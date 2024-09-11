"""Collision cuda functions."""

import numpy as np
import jax
from jax import numpy as jp
from jax.extend import ffi
from mujoco import mjx
from .. import _mjx_cuda_collision
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import Model

ffi.register_ffi_target(
    "calculate_bvh_aabb_dyn_cuda", _mjx_cuda_collision.bvh_aabb_dyn(), platform="CUDA"
)

ffi.register_ffi_target(
    "collision_broad_phase_nxn_cuda", _mjx_cuda_collision.broad_phase_nxn(), platform="CUDA"
)

ffi.register_ffi_target(
    "collision_broad_phase_sort_cuda", _mjx_cuda_collision.broad_phase_sort(), platform="CUDA"
) 

ffi.register_ffi_target(
    "gjk_epa_cuda", _mjx_cuda_collision.gjk_epa(), platform="CUDA"
)


def calculate_bvh_aabb_dyn(m: Model, d: Data,  d_bvh_aabb_dyn, out) -> Data:

  out_types = (
     jax.ShapeDtypeStruct(out.shape, dtype=jp.uint32), 
  )

  (
    out,
  ) = ffi.ffi_call(
      "calculate_bvh_aabb_dyn_cuda",
      out_types,
      d_bvh_aabb_dyn, #d.bvh_aabb_dyn,
      m.geom_aabb, 
      d.geom_xpos,
      d.geom_xmat, 
      m_ngeom = np.uint32(m.ngeom),
      vectorized=True,
  )

  return True 

def broad_phase_nxn(m: Model, d: Data,  d_bvh_aabb_dyn, col_counter,  col_pair, _max_geom_pairs, out) -> Data:

  out_types = (
     jax.ShapeDtypeStruct(out.shape, dtype=jp.uint32),  
  )

  (
    out,
  ) = ffi.ffi_call(
      "collision_broad_phase_nxn_cuda",
      out_types,
      col_counter,
      col_pair, 
      d_bvh_aabb_dyn,
      m_ngeom = np.uint32(m.ngeom),
      max_geom_pairs = np.uint32(_max_geom_pairs),
      vectorized=True,
  )

  return True 


def broad_phase_sort(m: Model, d: Data, d_bvh_aabb_dyn, col_counter, col_pair, _max_geom_pairs, out) -> Data:

  out_types = (
     jax.ShapeDtypeStruct(out.shape, dtype=jp.uint32),  
  )
  
  # buffer size = (m.ngeom + 1) * 4 (and multiples of 16)
  buffer_size = int((m.ngeom + 1 + 15) / 16) * 16 * 4
  buffer = jp.zeros(buffer_size, dtype=jp.uint32)

  (
     out,
  ) = ffi.ffi_call(
      "collision_broad_phase_sort_cuda",
      out_types,
      col_counter,
      col_pair, 
      d_bvh_aabb_dyn, #d.bvh_aabb_dyn,
      buffer, 
      m_ngeom = np.uint32(m.ngeom),
      max_geom_pairs = np.uint32(_max_geom_pairs),
      vectorized=True,
  )
 
  return True 

def gjk_epa(m: Model, d: Data, 
                         contact_counter,
                         contact_pos, contact_dist, contact_frame, contact_normal, contact_simplex, contact_pairs, 
                         _candidate_pair_count_max, _candidate_pair_count, candidate_pairs, 
                         _key_types0, _key_types1,
                         convex_vertex_array, convex_vertex_offset,
                         _depthExtension, _gjkIterationCount, _epaIterationCount, _epaBestCount, _multiPolygonCount, _multiTiltAngle, _ncon, _compress_result, out) -> Data:

  ncon_total = contact_pairs * ncon
  out_types = (
    jax.ShapeDtypeStruct(ncon_total, dtype=jp.uint32),  # contact_counter
    jax.ShapeDtypeStruct((ncon_total, 3), dtype=jp.float32),  # contact_pos
    jax.ShapeDtypeStruct(ncon_total, dtype=jp.float32),  # contact_dist
    jax.ShapeDtypeStruct((ncon_total, 3, 3), dtype=jp.float32),  # contact_frame
    jax.ShapeDtypeStruct((ncon_total, 3), dtype=jp.float32),  # contact_normal
    jax.ShapeDtypeStruct((ncon_total, 12), dtype=jp.float32),  # contact_simplex
    jax.ShapeDtypeStruct(0, dtype=jp.uint32),  # contact_pairs
  )

  (
    out,
  ) = ffi.ffi_call(
      "gjk_epa_cuda",
      out_types,
      candidate_pairs,
      d.geom_xpos,
      d.geom_xmat,
      m.geom_size,
      m.geom_dataid,
      convex_vertex_array,
      convex_vertex_offset,
      ncon = np.uint32(_ncon),
      m_ngeom = np.uint32(m.ngeom),
      candidate_pair_count_max = np.uint32(_candidate_pair_count_max),  # allocated size, including dummy. 
      candidate_pair_count = np.uint32(_candidate_pair_count),
      key_types0 = np.int32(_key_types0),
      key_types1 = np.int32(_key_types1),
      depthExtension = np.float32(_depthExtension),
      gjkIterationCount = np.uint32(_gjkIterationCount),
      epaIterationCount = np.uint32(_epaIterationCount),
      epaBestCount = np.uint32(_epaBestCount),
      multiPolygonCount = np.uint32(_multiPolygonCount),
      multiTiltAngle = np.float32(_multiTiltAngle),
      compress_result = np.uint32(_compress_result),
      vectorized=True,
  )

  return True 
