# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Collision driver in CUDA."""

import collections
import itertools
from typing import Iterator, Tuple, Union

import jax
from jax import numpy as jp
from jax.extend import ffi
import mujoco
from mujoco.mjx._src import math
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import Contact
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import GeomType
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
from mujoco.mjx._src.cuda import _engine_collision_driver
from mujoco.mjx._src.cuda import engine_collision_convex
import numpy as np


ffi.register_ffi_target(
    'collision_driver_cuda',
    _engine_collision_driver.collision(),
    platform='CUDA',
)


def _get_body_has_plane(m: Model) -> np.ndarray:
  body_has_plane = [False] * m.nbody
  for i in range(m.nbody):
    start = m.body_geomadr[i]
    end = m.body_geomadr[i] + m.body_geomnum[i]
    for g in range(start, end):
      if m.geom_type[g] == GeomType.PLANE:
        body_has_plane[i] = True
        break
  return np.array(body_has_plane, dtype=np.uint32)


def _body_pairs(
    m: Union[Model, mujoco.MjModel],
) -> Iterator[Tuple[int, int]]:
  """Yields body pairs to check for collision."""
  # TODO(btaba): merge logic back into collision driver.
  exclude_signature = set(m.exclude_signature)
  geom_con = m.geom_contype | m.geom_conaffinity
  filterparent = not (m.opt.disableflags & DisableBit.FILTERPARENT)
  b_start = m.body_geomadr
  b_end = b_start + m.body_geomnum

  for b1 in range(m.nbody):
    if not geom_con[b_start[b1] : b_end[b1]].any():
      continue
    w1 = m.body_weldid[b1]
    w1_p = m.body_weldid[m.body_parentid[w1]]

    for b2 in range(b1, m.nbody):
      if not geom_con[b_start[b2] : b_end[b2]].any():
        continue
      signature = (b1 << 16) + (b2)
      if signature in exclude_signature:
        continue
      w2 = m.body_weldid[b2]
      # ignore self-collisions
      if w1 == w2:
        continue
      w2_p = m.body_weldid[m.body_parentid[w2]]
      # ignore parent-child collisions
      if filterparent and w1 != 0 and w2 != 0 and (w1 == w2_p or w2 == w1_p):
        continue
      yield b1, b2


def _geom_pairs(
    m: Union[Model, mujoco.MjModel],
) -> Iterator[Tuple[int, int, int, int]]:
  """Yields geom pairs to check for collision."""
  geom_con = m.geom_contype | m.geom_conaffinity
  b_start = m.body_geomadr
  b_end = b_start + m.body_geomnum
  for b1, b2 in _body_pairs(m):
    g1_range = [g for g in range(b_start[b1], b_end[b1]) if geom_con[g]]
    g2_range = [g for g in range(b_start[b2], b_end[b2]) if geom_con[g]]
    for g1, g2 in itertools.product(g1_range, g2_range):
      t1, t2 = m.geom_type[g1], m.geom_type[g2]
      # order pairs by geom_type for correct function mapping
      if t1 > t2:
        g1, g2, t1, t2 = g2, g1, t2, t1
      # ignore plane<>plane and plane<>hfield
      if (t1, t2) == (GeomType.PLANE, GeomType.PLANE):
        continue
      if (t1, t2) == (GeomType.PLANE, GeomType.HFIELD):
        continue
      # geoms must match contype and conaffinity on some bit
      mask = m.geom_contype[g1] & m.geom_conaffinity[g2]
      mask |= m.geom_contype[g2] & m.geom_conaffinity[g1]
      if not mask:
        continue
      yield g1, g2, t1, t2


def _get_ngeom_pair(m: Model) -> int:
  """Returns an upper bound on the number of colliding geom pairs."""
  n_geom_pair = 0
  for (*_,) in _geom_pairs(m):
    n_geom_pair += 1
  return n_geom_pair


def _get_ngeom_pair_type_offset(m: Model) -> np.ndarray:
  """Returns offsets into geom pair types."""
  geom_pair_type_count = collections.defaultdict(int)
  for *_, t1, t2 in _geom_pairs(m):
    geom_pair_type_count[(t1, t2)] += 1

  offsets = [0]
  # order according to sequential id = t1 + t2 * n_geom_types
  for t2 in range(len(GeomType)):
    for t1 in range(len(GeomType)):
      if t1 > t2:
        offsets.append(0)  # upper triangle only
        continue
      if (t1, t2) not in geom_pair_type_count:
        offsets.append(0)
      else:
        offsets.append(geom_pair_type_count[(t1, t2)])

  assert sum(offsets) == _get_ngeom_pair(m)
  return np.cumsum(offsets)[:-1]


def collision(
    m: Model,
    d: Data,
    depth_extension: float,
    gjk_iter: int,
    epa_iter: int,
    epa_best_count: int,
    multi_polygon_count: int,
    multi_tilt_angle: float,
) -> Contact:
  """GJK/EPA narrowphase routine."""
  ngeom = m.ngeom

  if not (m.geom_condim[0] == m.geom_condim).all():
    raise NotImplementedError(
        'm.geom_condim should be the same for all geoms. Different condim per'
        ' geom is not supported yet.'
    )
  if len(d.geom_xpos.shape) != 2:
    raise ValueError(
        f'd.geom_xpos should have 2d shape, got "{len(d.geom_xpos.shape)}".'
    )
  if len(d.geom_xmat.shape) != 3:
    raise ValueError(
        f'd.geom_xmat should have 3d shape, got "{len(d.geom_xmat.shape)}".'
    )
  if m.geom_size.shape[0] != ngeom:
    raise ValueError(
        f'm.geom_size.shape[0] should be ngeom ({ngeom}), '
        f'got "{m.geom_size.shape[0]}".'
    )
  if m.geom_dataid.shape != (ngeom,):
    raise ValueError(
        f'm.geom_dataid.shape should be (ngeom,) == ({ngeom},), got'
        f' "({m.geom_dataid.shape[0]},)".'
    )
  if m.npair > 0:
    raise NotImplementedError('m.npair > 0 is not supported.')

  # TODO(btaba): geom_margin/gap are not supported, we should throw an error in
  #   put_model.

  max_contact_points = d.contact.pos.shape[0]
  n_pts = max_contact_points
  body_pair_size = int((m.nbody * (m.nbody - 1) / 2 + 15) / 16) * 16
  n_geom_pair = _get_ngeom_pair(m)
  out_types = (
      # Output buffers.
      jax.ShapeDtypeStruct((n_pts,), dtype=jp.float32),  # dist
      jax.ShapeDtypeStruct((n_pts, 3), dtype=jp.float32),  # pos
      jax.ShapeDtypeStruct((n_pts, 3), dtype=jp.float32),  # normal
      jax.ShapeDtypeStruct((n_pts,), dtype=jp.int32),  # g1
      jax.ShapeDtypeStruct((n_pts,), dtype=jp.int32),  # g2
      jax.ShapeDtypeStruct((n_pts,), dtype=jp.float32),  # includemargin
      jax.ShapeDtypeStruct((n_pts, 5), dtype=jp.float32),  # friction
      jax.ShapeDtypeStruct((n_pts, mujoco.mjNREF), dtype=jp.float32),  # solref
      jax.ShapeDtypeStruct(
          (n_pts, mujoco.mjNREF), dtype=jp.float32
      ),  # solreffriction
      jax.ShapeDtypeStruct((n_pts, mujoco.mjNIMP), dtype=jp.float32),  # solimp
      # Buffers used for intermediate results.
      # TODO(btaba): combine and re-use buffers instead of having so many.
      jax.ShapeDtypeStruct((m.nbody, 6), dtype=jp.float32),  # dyn_body_aamm
      jax.ShapeDtypeStruct(
          (body_pair_size, 2), dtype=jp.int32
      ),  # col_body_pair
      jax.ShapeDtypeStruct((1,), dtype=jp.uint32),  # env_counter
      jax.ShapeDtypeStruct((1,), dtype=jp.uint32),  # env_counter2
      jax.ShapeDtypeStruct((1,), dtype=jp.uint32),  # env_offset
      jax.ShapeDtypeStruct((ngeom, 6), dtype=jp.float32),  # dyn_geom_aabb
      jax.ShapeDtypeStruct((n_geom_pair, 2), dtype=jp.int32),  # col_geom_pair
      jax.ShapeDtypeStruct((n_geom_pair,), dtype=jp.uint32),  # type_pair_env_id
      jax.ShapeDtypeStruct(
          (n_geom_pair * 2,), dtype=jp.uint32
      ),  # type_pair_geom_id
  )

  n_geom_types = len(GeomType)
  n_geom_type_pairs = n_geom_types * n_geom_types
  type_pair_offset = _get_ngeom_pair_type_offset(m)
  type_pair_count = np.zeros(n_geom_type_pairs, dtype=np.uint32)
  convex_vert, convex_vert_offset = engine_collision_convex.get_convex_vert(m)
  (
      dist,
      pos,
      normal,
      g1,
      g2,
      includemargin,
      friction,
      solref,
      solreffriction,
      solimp,
      *_,
  ) = ffi.ffi_call(
      'collision_driver_cuda',
      out_types,
      d.geom_xpos,
      d.geom_xmat,
      m.geom_size,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_priority,
      m.geom_margin,
      m.geom_gap,
      m.geom_solmix,
      m.geom_friction,
      m.geom_solref,
      m.geom_solimp,
      # TODO(btaba): allow vmapping over sizes via geom_aabb/rbound jax.Array.
      m.geom_aabb,
      m.geom_rbound,
      m.geom_dataid,
      m.geom_bodyid,
      m.body_parentid,
      m.body_weldid,
      m.body_contype,
      m.body_conaffinity,
      m.body_geomadr,
      m.body_geomnum.astype(np.uint32),
      _get_body_has_plane(m),
      m.pair_geom1,
      m.pair_geom2,
      m.exclude_signature,
      m.pair_margin,
      m.pair_gap,
      m.pair_friction,
      m.pair_solref,
      m.pair_solimp,
      convex_vert,
      convex_vert_offset,
      type_pair_offset.astype(np.uint32),
      type_pair_count,
      ngeom=np.uint32(ngeom),
      npair=np.uint32(m.npair),
      nbody=np.uint32(m.nbody),
      nexclude=np.uint32(m.nexclude),
      max_contact_points=np.uint32(max_contact_points),
      n_geom_pair=np.uint32(n_geom_pair),
      n_geom_types=np.uint32(n_geom_types),
      filter_parent=not (m.opt.disableflags & DisableBit.FILTERPARENT),
      depth_extension=np.float32(depth_extension),
      gjk_iteration_count=np.uint32(gjk_iter),
      epa_iteration_count=np.uint32(epa_iter),
      epa_best_count=np.uint32(epa_best_count),
      multi_polygon_count=np.uint32(multi_polygon_count),
      multi_tilt_angle=np.float32(multi_tilt_angle),
      vectorized=True,
  )

  c = Contact(
      dist=dist,
      pos=pos,
      frame=jax.vmap(math.make_frame)(normal),
      includemargin=includemargin,
      friction=friction,
      solref=solref,
      solreffriction=solreffriction,
      solimp=solimp,
      geom1=g1,
      geom2=g2,
      geom=jp.array([g1, g2]).T,
      efc_address=d.contact.efc_address,
      dim=d.contact.dim,
  )

  return c
