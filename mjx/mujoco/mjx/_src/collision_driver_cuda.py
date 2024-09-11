# Copyright 2023 DeepMind Technologies Limited
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
"""Runs collision checking for all geoms in a Model.

To do this, collision_driver builds a collision function table, and then runs
the collision functions serially on the parameters in the table.

For example, if a Model has three geoms:

geom   |   type
---------------
1      | sphere
2      | capsule
3      | sphere

collision_driver organizes it into these functions and runs them:

function       | geom pair
--------------------------
sphere_sphere  | (1, 3)
sphere_capsule | (1, 2), (2, 3)


Besides collision function, function tables are keyed on mesh id and condim,
in order to guarantee static shapes for contacts and jacobians.
"""

import itertools
from typing import Dict, Iterator, List, Tuple, Union

import jax
from jax import numpy as jp
import mujoco
from mujoco.mjx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx._src.collision_convex import box_box
from mujoco.mjx._src.collision_convex import capsule_convex
from mujoco.mjx._src.collision_convex import convex_convex
from mujoco.mjx._src.collision_convex import hfield_capsule
from mujoco.mjx._src.collision_convex import hfield_convex
from mujoco.mjx._src.collision_convex import hfield_sphere
from mujoco.mjx._src.collision_convex import plane_convex
from mujoco.mjx._src.collision_convex import sphere_convex
from mujoco.mjx._src.collision_primitive import capsule_capsule
from mujoco.mjx._src.collision_primitive import plane_capsule
from mujoco.mjx._src.collision_primitive import plane_cylinder
from mujoco.mjx._src.collision_primitive import plane_ellipsoid
from mujoco.mjx._src.collision_primitive import plane_sphere
from mujoco.mjx._src.collision_primitive import sphere_capsule
from mujoco.mjx._src.collision_primitive import sphere_sphere
from mujoco.mjx._src.collision_sdf import capsule_cylinder
from mujoco.mjx._src.collision_sdf import capsule_ellipsoid
from mujoco.mjx._src.collision_sdf import cylinder_cylinder
from mujoco.mjx._src.collision_sdf import ellipsoid_cylinder
from mujoco.mjx._src.collision_sdf import ellipsoid_ellipsoid
from mujoco.mjx._src.collision_types import FunctionKey
from mujoco.mjx._src.types import Contact
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import GeomType
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np

#=================================================================
# mjx_cuda_collision
#-----------------------------------------------------------------
from mujoco.mjx._src import mjx_cuda_collision
import jax.lax as lax
import time
#-----------------------------------------------------------------
g_print_time = False
g_contact_init = True
# fixed-size 
g_compilation_size : Dict[FunctionKey, int] = {}
#-----------------------------------------------------------------
# [Broad-phase]
g_broadphase_method = 0   # broadphase_off
#g_broadphase_method = 1   # NxN
#g_broadphase_method = 2   # sort
#-----------------------------------------------------------------
# [GJK setup]
g_run_mujoco = False
g_run_gjk = True
g_compress_result = False
# [GJK global variables] to avoid allocating memory per step (true??)
g_contact_counter = jp.zeros(0, jp.uint32)
g_contact_dist = jp.zeros(0, jp.float32)
g_contact_pos = jp.zeros(0, jp.float32)
g_contact_frame = jp.zeros(0, jp.float32)
g_contact_normal = jp.zeros(0, jp.float32)
g_contact_simplex = jp.zeros(0, jp.float32)
g_contact_pairs = jp.zeros(0, jp.int32)
g_out = jp.zeros(1, jp.uint32)
#-----------------------------------------------------------------
# def [merge_convex_vert]
#     merge convex mesh vertices into one array to run convex-convex cases all togeter 
#-----------------------------------------------------------------
g_merge_convex_mesh = True
g_convex_initialized = False
g_convex_vertex_array : jax.Array
g_convex_vertex_offset : jax.Array
#-----------------------------------------------------------------
# def merge_convex_vert(m: Union[Model, mujoco.MjModel], ) -> bool :

#   global g_convex_initialized
  
#   if(g_convex_initialized) : 
#     return False

#   global g_convex_vertex_array
#   global g_convex_vertex_offset

#   mesh_convex_size = len(m.mesh_convex)

#   total_convex_vertex_size = 0
#   for cm in m.mesh_convex : 
#     if cm is not None: total_convex_vertex_size += cm.vert.shape[0]
  
#   g_convex_vertex_array = jp.zeros((total_convex_vertex_size, 3), jp.float32)
#   convex_vertex_offset_np = np.zeros(mesh_convex_size + 1, np.int32)
#   convex_vertex_offset_np[0] = 0

#   # copy convex_vertex_data to jax with offset 
#   vertex_offset = 0
#   for cmi in range(mesh_convex_size) :
#     vertex_count = 0
#     if m.mesh_convex[cmi] is not None: 
#       vertex_count = m.mesh_convex[cmi].vert.shape[0]
#       g_convex_vertex_array = lax.dynamic_update_slice(g_convex_vertex_array, m.mesh_convex[cmi].vert, (vertex_offset, 0))
#     vertex_offset += vertex_count
#     convex_vertex_offset_np[cmi + 1] = vertex_offset
  
#   # copy vertex_offset data to jax
#   g_convex_vertex_offset = jp.array(convex_vertex_offset_np)

#   g_convex_initialized = 1

#   return True
#-----------------------------------------------------------------
#=================================================================



# pair-wise collision functions
_COLLISION_FUNC = {
    (GeomType.PLANE, GeomType.SPHERE): plane_sphere,
    (GeomType.PLANE, GeomType.CAPSULE): plane_capsule,
    (GeomType.PLANE, GeomType.BOX): plane_convex,
    (GeomType.PLANE, GeomType.ELLIPSOID): plane_ellipsoid,
    (GeomType.PLANE, GeomType.CYLINDER): plane_cylinder,
    (GeomType.PLANE, GeomType.MESH): plane_convex,
    (GeomType.HFIELD, GeomType.SPHERE): hfield_sphere,
    (GeomType.HFIELD, GeomType.CAPSULE): hfield_capsule,
    (GeomType.HFIELD, GeomType.BOX): hfield_convex,
    (GeomType.HFIELD, GeomType.MESH): hfield_convex,
    (GeomType.SPHERE, GeomType.SPHERE): sphere_sphere,
    (GeomType.SPHERE, GeomType.CAPSULE): sphere_capsule,
    (GeomType.SPHERE, GeomType.BOX): sphere_convex,
    (GeomType.SPHERE, GeomType.MESH): sphere_convex,
    (GeomType.CAPSULE, GeomType.CAPSULE): capsule_capsule,
    (GeomType.CAPSULE, GeomType.BOX): capsule_convex,
    (GeomType.CAPSULE, GeomType.ELLIPSOID): capsule_ellipsoid,
    (GeomType.CAPSULE, GeomType.CYLINDER): capsule_cylinder,
    (GeomType.CAPSULE, GeomType.MESH): capsule_convex,
    (GeomType.ELLIPSOID, GeomType.ELLIPSOID): ellipsoid_ellipsoid,
    (GeomType.ELLIPSOID, GeomType.CYLINDER): ellipsoid_cylinder,
    (GeomType.CYLINDER, GeomType.CYLINDER): cylinder_cylinder,
    (GeomType.BOX, GeomType.BOX): box_box,
    (GeomType.BOX, GeomType.MESH): convex_convex,
    (GeomType.MESH, GeomType.MESH): convex_convex,
}


# geoms for which we ignore broadphase
_GEOM_NO_BROADPHASE = {GeomType.HFIELD, GeomType.PLANE}


def has_collision_fn(t1: GeomType, t2: GeomType) -> bool:
  """Returns True if a collision function exists for a pair of geom types."""
  return (t1, t2) in _COLLISION_FUNC


def geom_pairs(
    m: Union[Model, mujoco.MjModel],
    #-----------------------------------------------------------------
    d: Data  # to access d.geom_xpos, d.geom_xmat 
    #-----------------------------------------------------------------
) -> Iterator[Tuple[int, int, int]]:
  """Yields geom pairs to check for collisions.

  Args:
    m: a MuJoCo or MJX model

  Yields:
    geom1, geom2, and pair index if defined in <pair> (else -1)
  """
  pairs = set()

  # -----------------------------------------------------------------
  global g_broadphase_method, g_out
  pool_size = 0
  pool_set = set() 
  use_pool = False 
  np.set_printoptions(precision=2, suppress=True)

  if(d and (g_broadphase_method > 0)) :
    
    global g_contact_init 
    use_pool = g_contact_init

    if(use_pool) : 

      global g_print_time
      start_time = 0.0
      if(g_print_time) : start_time = time.perf_counter()

      max_geom_pairs = _numeric(m, 'max_geom_pairs')
      max_geom_pairs = 20

      # calculate bvh_aabb_dyn
      d_bvh_aabb_dyn = jp.zeros(m.ngeom * 6, dtype=jp.float32)
      d_bvh_aabb_dyn.reshape((m.ngeom, 3, 2))
      mjx_cuda_collision.calculate_bvh_aabb_dyn(m, d, d_bvh_aabb_dyn, g_out)

      col_counter = jp.zeros(1, dtype=jp.uint32)
      #print('HERE', m.ngeom * 2 * max_geom_pairs, max_geom_pairs)

      col_pair = jp.zeros(m.ngeom * 2 * max_geom_pairs, dtype=jp.uint32)
    
      if(g_broadphase_method == 1) : 
        mjx_cuda_collision.broad_phase_nxn(m, d, d_bvh_aabb_dyn, col_counter, col_pair, max_geom_pairs, g_out)
    
      elif(g_broadphase_method == 2) :
        mjx_cuda_collision.broad_phase_sort(m, d, d_bvh_aabb_dyn, col_counter, col_pair, max_geom_pairs, g_out)

      col_counter.block_until_ready()
      col_pair.block_until_ready()
      pool_size = jax.device_get(col_counter[0])
      pool = jax.device_get(col_pair[0:pool_size * 2])

      if(g_print_time) :
        end_time = time.perf_counter()
        elapsed_time = (time.perf_counter() - start_time) * 1e4 # milli
        print(f"  - broad-phase: {elapsed_time:.2f} ms")

    # if the result is full NxN -> no reason to use 
    if(use_pool) : 
      # if(pool_size == int(m.ngeom * (m.ngeom-1) / 2)) : use_pool = False
      # else : 
      for i  in range (pool_size) :
        pool_set.add((pool[i*2], pool[i*2+1]))
    
  # -----------------------------------------------------------------

  for i in range(m.npair):
    g1, g2 = m.pair_geom1[i], m.pair_geom2[i]
    # order pairs by geom_type for correct function mapping
    if m.geom_type[g1] > m.geom_type[g2]:
      g1, g2 = g2, g1

    # -----------------------------------------------------------------
    p1, p2 = g1, g2 
    if(p1 > p2) : p1, p2 = g2, g1
    if(use_pool and ((p1, p2) not in pool_set)) : continue 
    # -----------------------------------------------------------------

    pairs.add((g1, g2))
    yield g1, g2, i

  exclude_signature = set(m.exclude_signature)
  geom_con = m.geom_contype | m.geom_conaffinity
  filterparent = not (m.opt.disableflags & DisableBit.FILTERPARENT)
  b_start = m.body_geomadr
  b_end = b_start + m.body_geomnum

  for b1 in range(m.nbody):
    if not geom_con[b_start[b1]:b_end[b1]].any():
      continue
    w1 = m.body_weldid[b1]
    w1_p = m.body_weldid[m.body_parentid[w1]]

    for b2 in range(b1, m.nbody):
      if not geom_con[b_start[b2]:b_end[b2]].any():
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

        # -----------------------------------------------------------------
        p1, p2 = g1, g2 
        if(p1 > p2) : p1, p2 = g2, g1
        if(use_pool and ((p1, p2) not in pool_set)) : continue
        # -----------------------------------------------------------------

        # geoms must match contype and conaffinity on some bit
        mask = m.geom_contype[g1] & m.geom_conaffinity[g2]
        mask |= m.geom_contype[g2] & m.geom_conaffinity[g1]
        if not mask:
          continue

        if (g1, g2) not in pairs:
          pairs.add((g1, g2))
          yield g1, g2, -1

 

def _geom_groups(
    m: Union[Model, mujoco.MjModel],
    # -----------------------------------------------------------------
    d: Data 
    # -----------------------------------------------------------------
) -> Dict[FunctionKey, List[Tuple[int, int, int]]]:
  """Returns geom pairs to check for collision grouped by collision function.

  The grouping consists of:
    - The collision function to run, which is determined by geom types
    - For mesh geoms, convex functions are run for each distinct mesh in the
      model, because the convex functions expect static mesh size. If a sphere
      collides with a cube and a tetrahedron, sphere_convex is called twice.
    - The condim of the collision. This ensures that the size of the resulting
      constraint jacobian is determined at compile time.

  Args:
    m: a MuJoCo or MJX model

  Returns:
    a dict with grouping key and values geom1, geom2, pair index
  """
  groups = {}

  for g1, g2, ip in geom_pairs(m, d):
    types = m.geom_type[g1], m.geom_type[g2]
    data_ids = m.geom_dataid[g1], m.geom_dataid[g2]
    if ip > -1:
      condim = m.pair_dim[ip]
    elif m.geom_priority[g1] > m.geom_priority[g2]:
      condim = m.geom_condim[g1]
    elif m.geom_priority[g1] < m.geom_priority[g2]:
      condim = m.geom_condim[g2]
    else:
      condim = max(m.geom_condim[g1], m.geom_condim[g2])

    #-----------------------------------------------------------------
    #  key = FunctionKey(types, data_ids, condim)
    #-----------------------------------------------------------------
    global g_merge_convex_mesh
    if(g_merge_convex_mesh == 1 and g_run_gjk) : 
      key = FunctionKey(types, (-1, -1), condim)
    else : 
      key = FunctionKey(types, data_ids, condim)
    #-----------------------------------------------------------------

    if types[0] == mujoco.mjtGeom.mjGEOM_HFIELD:
      # add static grid bounds to the grouping key for hfield collisions
      geom_rbound_hfield = (
          m.geom_rbound_hfield if isinstance(m, Model) else m.geom_rbound
      )
      nrow, ncol = m.hfield_nrow[data_ids[0]], m.hfield_ncol[data_ids[0]]
      xsize, ysize = m.hfield_size[data_ids[0]][:2]
      xtick, ytick = (2 * xsize) / (ncol - 1), (2 * ysize) / (nrow - 1)
      xbound = int(np.ceil(2 * geom_rbound_hfield[g2] / xtick)) + 1
      xbound = min(xbound, ncol)
      ybound = int(np.ceil(2 * geom_rbound_hfield[g2] / ytick)) + 1
      ybound = min(ybound, nrow)
      key = FunctionKey(types, data_ids, condim, (xbound, ybound))

    groups.setdefault(key, []).append((g1, g2, ip))

  return groups


def _contact_groups(m: Model, d: Data) -> Dict[FunctionKey, Contact]:
  """Returns contact groups to check for collisions.

  Contacts are grouped the same way as _geom_groups.  Only one contact is
  emitted per geom pair, even if the collision function emits multiple contacts.

  Args:
    m: MJX model
    d: MJX data

  Returns:
    a dict where the key is the grouping and value is a Contact
  """
  groups = {}
  eps = mujoco.mjMINVAL

  #-----------------------------------------------------------------
  global g_print_time
  #-----------------------------------------------------------------


  for key, geom_ids in _geom_groups(m, d).items():
    geom = np.array(geom_ids)
    geom1, geom2, ip = geom.T
    geom1, geom2, ip = geom1[ip == -1], geom2[ip == -1], ip[ip != -1]
    params = []

    
    if ip.size > 0:
      # pair contacts get their params from m.pair_* fields
      params.append((
          m.pair_margin[ip] - m.pair_gap[ip],
          jp.clip(m.pair_friction[ip], a_min=eps),
          m.pair_solref[ip],
          m.pair_solreffriction[ip],
          m.pair_solimp[ip]
      ))
    if geom1.size > 0 and geom2.size > 0:
      # other contacts get their params from geom fields
      margin = jp.maximum(m.geom_margin[geom1], m.geom_margin[geom2])
      gap = jp.maximum(m.geom_gap[geom1], m.geom_gap[geom2])
      solmix1, solmix2 = m.geom_solmix[geom1], m.geom_solmix[geom2]
      mix = solmix1 / (solmix1 + solmix2)
      mix = jp.where((solmix1 < eps) & (solmix2 < eps), 0.5, mix)
      mix = jp.where((solmix1 < eps) & (solmix2 >= eps), 0.0, mix)
      mix = jp.where((solmix1 >= eps) & (solmix2 < eps), 1.0, mix)
      mix = mix[:, None]  # for correct broadcasting
      # friction: max
      friction = jp.maximum(m.geom_friction[geom1], m.geom_friction[geom2])
      solref1, solref2 = m.geom_solref[geom1], m.geom_solref[geom2]
      # reference standard: mix
      solref_standard = mix * solref1 + (1 - mix) * solref2
      # reference direct: min
      solref_direct = jp.minimum(solref1, solref2)
      is_standard = (solref1[:, [0, 0]] > 0) & (solref2[:, [0, 0]] > 0)
      solref = jp.where(is_standard, solref_standard, solref_direct)
      solreffriction = jp.zeros(geom1.shape + (mujoco.mjNREF,))
      # impedance: mix
      solimp = mix * m.geom_solimp[geom1] + (1 - mix) * m.geom_solimp[geom2]

      pri = m.geom_priority[geom1] != m.geom_priority[geom2]
      if pri.any():
        # use priority geom when specified instead of mixing
        gp1, gp2 = m.geom_priority[geom1], m.geom_priority[geom2]
        gp = np.where(gp1 > gp2, geom1, geom2)[pri]
        friction = friction.at[pri].set(m.geom_friction[gp])
        solref = solref.at[pri].set(m.geom_solref[gp])
        solimp = solimp.at[pri].set(m.geom_solimp[gp])
     
      # unpack 5d friction:
      friction = friction[:, [0, 0, 1, 2, 2]]
      params.append((margin - gap, friction, solref, solreffriction, solimp))

    params = map(jp.concatenate, zip(*params))
    includemargin, friction, solref, solreffriction, solimp = params

    #-----------------------------------------------------------------
    # resize params to have the compilation size
    #-----------------------------------------------------------------
    global g_compilation_size 
    if(len(g_compilation_size) > 0) : 
      if(key in g_compilation_size) : 
        size_before = geom.shape[0]
        size = g_compilation_size[key]
        if(size_before < size) : 
          start_time = 0.0
          if(g_print_time) : start_time = time.perf_counter()

          start_time = time.perf_counter()
          includemargin = jp.resize(includemargin, size)
          friction = jp.resize(friction, (size, 5))
          solref = jp.resize(solref, (size, 2))
          solreffriction = jp.resize(solreffriction, (size, 2))
          solimp = jp.resize(solimp, (size, 5))
          #geom = jp.resize(geom, (size, 2))
          #geom = geom.at[size_before-1:size,:].set(-1)
          if(g_print_time) : 
            elapsed_time = (time.perf_counter() - start_time) * 1e4 # milli
            print(f"  - resize param: {elapsed_time:.2f} ms ", "(" , size_before , "->" , size , ")")          
    #-----------------------------------------------------------------

    groups[key] = Contact(
        # dist, pos, frame get filled in by collision functions:
        dist=None,
        pos=None,
        frame=None,
        includemargin=includemargin,
        friction=friction,
        solref=solref,
        solreffriction=solreffriction,
        solimp=solimp,
        dim=d.contact.dim,
        geom1=jp.array(geom[:, 0]),
        geom2=jp.array(geom[:, 1]),
        geom=jp.array(geom[:, :2]),
        efc_address=d.contact.efc_address,
    )

  #-----------------------------------------------------------------
  # add new empty groups if it disapears 
  #-----------------------------------------------------------------
  if(len(groups) != len(g_compilation_size)) :
    for key in g_compilation_size : 
      if key not in groups: 
        start_time = 0.0
        if(g_print_time) : start_time = time.perf_counter()
        size = g_compilation_size[key]
        groups[key] = Contact(
        dist=jp.full((size, 2), 100000000.0),
        pos=None,
        frame=None,
        includemargin=jp.zeros(size, jp.float32),
        friction=jp.zeros((size, 5), jp.float32),
        solref=jp.zeros((size, 2), jp.float32),
        solreffriction=jp.zeros((size, 2), jp.float32),
        solimp=jp.zeros((size, 5), jp.float32),
        dim=d.contact.dim,
        geom1=jp.zeros(0, jp.int32),
        geom2=jp.zeros(0, jp.int32),
        geom=jp.full((0, 2), -1),
        efc_address=d.contact.efc_address,
        )
        if(g_print_time) : 
          elapsed_time = (time.perf_counter() - start_time) * 1e4 # milli
          print(f"  - create key: {elapsed_time:.2f} ms")        
  #-----------------------------------------------------------------

  return groups


def _numeric(m: Union[Model, mujoco.MjModel], name: str) -> int:
  id_ = support.name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
  return int(m.numeric_data[id_]) if id_ >= 0 else -1


def make_condim(m: Union[Model, mujoco.MjModel], d: Data) -> np.ndarray:
  """Returns the dims of the contacts for a Model."""
  if m.opt.disableflags & DisableBit.CONTACT:
    return np.empty(0, dtype=int)

  group_counts = {k: len(v) for k, v in _geom_groups(m, d).items()}

  # max_geom_pairs limits the number of pairs we process in a collision function
  # by first running a primitive broad phase culling on the pairs
  max_geom_pairs = _numeric(m, 'max_geom_pairs')

  if max_geom_pairs > -1:
    for k in group_counts:
      if set(k.types) & _GEOM_NO_BROADPHASE:
        continue
      group_counts[k] = min(group_counts[k], max_geom_pairs)

  # max_contact_points limits the number of contacts emitted by selecting the
  # contacts with the most penetration after calling collision functions
  max_contact_points = _numeric(m, 'max_contact_points')

  condim_counts = {}
  for k, v in group_counts.items():
    func = _COLLISION_FUNC[k.types]
    num_contacts = condim_counts.get(k.condim, 0) + func.ncon * v  # pytype: disable=attribute-error
    if max_contact_points > -1:
      num_contacts = min(max_contact_points, num_contacts)
    condim_counts[k.condim] = num_contacts

  dims = sum(([c] * condim_counts[c] for c in sorted(condim_counts)), [])

  return np.array(dims)


def collision(m: Model, d: Data) -> Data:
  """Collides geometries."""
  if d.ncon == 0:
    return d

  groups = _contact_groups(m, d)
  max_geom_pairs = _numeric(m, 'max_geom_pairs')
  max_contact_points = _numeric(m, 'max_contact_points')

  # #-----------------------------------------------------------------
  # global g_contact_init
  # if(g_contact_init == False) : 
  #   # build a merged convex mesh 
  #   merge_convex_vert(m)
  #   # initial key 
  #   for key, contact in groups.items():
  #     g_compilation_size[key] = contact.geom.shape[0]
  #   g_contact_init = True
  # #-----------------------------------------------------------------
  global g_compilation_size
  for key, contact in groups.items():
    g_compilation_size[key] = contact.geom.shape[0]
  
  # run collision functions on groups
  for key, contact in groups.items():
    # determine which contacts we'll use for collision testing by running a
    # broad phase cull if requested
    if (
        max_geom_pairs > -1
        and contact.geom.shape[0] > max_geom_pairs
        and not set(key.types) & _GEOM_NO_BROADPHASE
    ):
      pos1, pos2 = d.geom_xpos[contact.geom.T]
      size1, size2 = m.geom_rbound[contact.geom.T]
      dist = jax.vmap(jp.linalg.norm)(pos2 - pos1) - (size1 + size2)
      _, idx = jax.lax.top_k(-dist, k=max_geom_pairs)
      contact = jax.tree_util.tree_map(lambda x, idx=idx: x[idx], contact)
    # run the collision function specified by the grouping key

    #-----------------------------------------------------------------
    global g_compress_result, g_contact_counter, g_contact_dist, g_contact_pos, g_contact_frame, g_contact_normal, g_contact_simplex, g_contact_pairs, g_out
    candidate_pair_count_max = g_compilation_size[key] # contact.geom.shape[0]
    candidate_pair_count = contact.geom.shape[0]

    func = _COLLISION_FUNC[key.types]
    ncon = func.ncon  # pytype: disable=attribute-error

    if(g_run_mujoco) : 
      #-----------------------------------------------------------------
      g_contact_dist, g_contact_pos, g_contact_frame = func(m, d, key, contact.geom)
      #-----------------------------------------------------------------

    if(g_run_gjk) : 

      g_contact_pos = jp.zeros((candidate_pair_count_max * ncon, 3), dtype=jp.float32)
      g_contact_dist = jp.full(candidate_pair_count_max * ncon, 100000000.0)
      g_contact_frame = jp.zeros((candidate_pair_count_max * ncon, 3, 3), dtype=jp.float32)

      g_contact_counter = jp.zeros(candidate_pair_count_max, dtype=jp.uint32)
      g_contact_normal = jp.zeros((candidate_pair_count_max, 3), dtype=jp.float32)
      g_contact_simplex = jp.zeros((candidate_pair_count_max, 12), dtype=jp.float32) # float3 * 4
      if(g_compress_result) : g_contact_pairs = jp.zeros((candidate_pair_count_max, 2), dtype=jp.int32)
      else : g_contact_pairs = jp.zeros(0, dtype=jp.int32)
      g_out = jp.zeros(1, dtype=jp.uint32)

      (
          g_contact_counter,
          g_contact_pos,
          g_contact_dist,
          g_contact_frame,
          g_contact_normal,
          g_contact_simplex,
          g_contact_pairs,
      ) = mjx_cuda_collision.gjk_epa(m, d,
                                candidate_pair_count_max, candidate_pair_count, contact.geom, key.types[0], key.types[1], 
                                m.g_convex_vertex_array, m.g_convex_vertex_offset, 
                                1000000000.0, 12, 12, 12, 8, 1.0, ncon, g_compress_result, g_out);
      # jax.debug.print("n_con={x}, n_geom={y}", x=(g_contact_dist < 0).sum(), y=candidate_pair_count_max)
      # jax.debug.print("g_contact_counter={x}", x=g_contact_counter)
      # jax.debug.print("g_contact_dist={x}", x=g_contact_dist)
      # import IPython; IPython.embed(user_ns=dict(globals(), **locals()))
   
    #-----------------------------------------------------------------
    if(candidate_pair_count < candidate_pair_count_max) : 
      geom = contact.geom
      geom1 = contact.geom1
      geom2 = contact.geom2
      geom = jp.resize(contact.geom, (candidate_pair_count_max, 2))
      geom1 = jp.resize(contact.geom1, candidate_pair_count_max)
      geom2 = jp.resize(contact.geom2, candidate_pair_count_max)
      geom = geom.at[candidate_pair_count:candidate_pair_count_max,:].set(-1)
      geom1 = geom1.at[candidate_pair_count:candidate_pair_count_max].set(-1)
      geom2 = geom2.at[candidate_pair_count:candidate_pair_count_max].set(-1)
      contact = contact.replace(geom = geom, geom1=geom1, geom2=geom2)
    #-----------------------------------------------------------------

      
    if ncon > 1:
      # repeat contacts to match the number of collisions returned
      repeat_fn = lambda x, r=ncon: jp.repeat(x, r, axis=0)
      contact = jax.tree_util.tree_map(repeat_fn, contact)

   
    if(candidate_pair_count <candidate_pair_count_max) :
      g_contact_dist = g_contact_dist.at[(candidate_pair_count)*ncon:(candidate_pair_count_max)*ncon].set(100000000.0)

    if(False):
      print("candidate_pair_count", candidate_pair_count, "candidate_pair_count_max", candidate_pair_count_max)
      print("geom", contact.geom)
      print("g_contact_counter", g_contact_counter)
      print("g_contact_pos", g_contact_pos)
      print("g_contact_dist", g_contact_dist)
      print("g_contact_frame", g_contact_frame)



    groups[key] = contact.replace(dist=g_contact_dist, pos=g_contact_pos, frame=g_contact_frame)

  # collapse contacts together, ensuring they are grouped by condim
  condim_groups = {}
  for key, contact in groups.items():
    condim_groups.setdefault(key.condim, []).append(contact)

  # limit the number of contacts per condim group if requested
  if max_contact_points > -1:
    for key, contacts in condim_groups.items():
      contact = jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *contacts)
      if contact.geom.shape[0] > max_contact_points:
        _, idx = jax.lax.top_k(-contact.dist, k=max_contact_points)
        contact = jax.tree_util.tree_map(lambda x, idx=idx: x[idx], contact)
      condim_groups[key] = [contact]

  contacts = sum([condim_groups[k] for k in sorted(condim_groups)], [])
  contact = jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *contacts)

  return d.replace(contact=contact)
