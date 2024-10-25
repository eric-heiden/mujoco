// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "engine_collision_driver.h"  // mjx/cuda

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cstdio>

#include <cuda_runtime_api.h>  // cuda
#include <helper_math.h>  // cuda_samples
#include <mujoco/mujoco.h>
#include "engine_collision_common.h"  // mjx/cuda
#include "engine_collision_convex.h"  // mjx/cuda
#include "engine_util_blas.cu.h"  // mjx/cuda
#include <xla/ffi/api/ffi.h>

namespace mujoco::mjx::cuda {

namespace {

bool assert_cuda(const char* msg) {
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    printf("[CUDA-ERROR] [%s] (%d:%s) \n", msg ? msg : "", (int)last_error,
           cudaGetErrorString(last_error));
  }
  return (last_error == cudaSuccess);
}

bool assert_count(int count, const int expected, const char* msg) {
  if (count != expected) {
    printf("[CUDA-ERROR] [%s] Invalid size (%d != %d) elements.\n",
           msg ? msg : "", count, expected);
    return false;
  }
  return true;
}

bool assert_that(bool cond, const char* msg) {
  if (!cond) {
    printf("[CUDA-ERROR] [%s].\n", msg ? msg : "");
    return false;
  }
  return true;
}

__device__ __forceinline__ void xposmat_to_float4(const float* xpos,
                                                  const float* xmat, int gi,
                                                  float4* mat) {
  mat[0] = make_float4(xmat[gi * 9 + 0], xmat[gi * 9 + 1], xmat[gi * 9 + 2],
                       xpos[gi * 3 + 0]);
  mat[1] = make_float4(xmat[gi * 9 + 3], xmat[gi * 9 + 4], xmat[gi * 9 + 5],
                       xpos[gi * 3 + 1]);
  mat[2] = make_float4(xmat[gi * 9 + 6], xmat[gi * 9 + 7], xmat[gi * 9 + 8],
                       xpos[gi * 3 + 2]);
}

}  // namespace

// Bisect left the value v in the array x, with [a, b] as the initial range.
template <typename T>
__device__ __forceinline__ uint bisection(const T* __restrict x, const T v,
                                          uint a, uint b) {
  uint c = 0;
  while (b - a > 1) {
    c = (a + b) / 2;
    if (x[c] <= v) {
      a = c;
    } else {
      b = c;
    }
  }
  c = a;
  if (c != b && (x[b] <= v)) c = b;
  return c;
}

__global__ void init(const uint max_contact_points, const uint nenv,
                     float* __restrict dist, float* __restrict pos,
                     float* __restrict normal, int* __restrict g1,
                     int* __restrict g2, float* __restrict includemargin,
                     float* __restrict friction, float* __restrict solref,
                     float* __restrict solreffriction,
                     float* __restrict solimp) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nenv * max_contact_points) return;
  dist[tid] = 1e12f;
  pos[tid * 3] = 0.0f;
  pos[tid * 3 + 1] = 0.0f;
  pos[tid * 3 + 2] = 0.0f;
  normal[tid * 3] = 0.0f;
  normal[tid * 3 + 1] = 0.0f;
  normal[tid * 3 + 2] = 1.0f;
  g1[tid] = -1;
  g2[tid] = -1;
  includemargin[tid] = 0.0f;
  solref[tid * 2] = 0.02f;
  solref[tid * 2 + 1] = 1.0f;
  solimp[tid * 5] = 0.9f;
  solimp[tid * 5 + 1] = 0.95f;
  solimp[tid * 5 + 2] = 0.001f;
  solimp[tid * 5 + 3] = 0.5f;
  solimp[tid * 5 + 4] = 2.0f;
  friction[tid * 5] = 1.0f;
  friction[tid * 5 + 1] = 1.0f;
  friction[tid * 5 + 2] = 0.005f;
  friction[tid * 5 + 3] = 0.0001f;
  friction[tid * 5 + 4] = 0.0001f;
  solreffriction[tid * 2] = 0.0;
  solreffriction[tid * 2 + 1] = 0.0f;
}

__global__ void init_buffers(const uint nenv,
                             uint* __restrict col_body_pair_count,
                             uint* __restrict col_body_pair_offset,
                             uint* __restrict col_geom_pair_count) {
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nenv) return;
  col_body_pair_offset[tid] = 0;
  col_body_pair_count[tid] = 0;
  col_geom_pair_count[tid] = 0;
}

__global__ void get_dyn_body_aamm(
    const uint nenv, const uint nbody, const uint nmodel, const uint ngeom,
    uint* __restrict body_geomnum, int* __restrict body_geomadr,
    float* __restrict geom_margin, float* __restrict geom_xpos,
    float* __restrict geom_rbound, float* __restrict dyn_body_aamm) {
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nenv * nbody) return;
  const int bid = tid % nbody;
  const uint env_id = tid / nbody;
  const uint model_id = env_id % nmodel;
  float aamm[] = {__FLT_MAX__,  __FLT_MAX__,  __FLT_MAX__,
                  -__FLT_MAX__, -__FLT_MAX__, -__FLT_MAX__};
  for (int i = 0; i < body_geomnum[bid]; i++) {
    // TODO(btaba): do not use bodies/geoms that never collide.
    int g = body_geomadr[bid] + i;
#pragma unroll
    for (int j = 0; j < 3; j++) {
      float aamm_min = geom_xpos[(env_id * ngeom + g) * 3 + j] -
                       geom_rbound[model_id * ngeom + g] -
                       geom_margin[model_id * ngeom + g];
      float aamm_max = geom_xpos[(env_id * ngeom + g) * 3 + j] +
                       geom_rbound[model_id * ngeom + g] +
                       geom_margin[model_id * ngeom + g];
      aamm[j] = (aamm_min < aamm[j]) ? aamm_min : aamm[j];
      aamm[j + 3] = (aamm_max > aamm[j + 3]) ? aamm_max : aamm[j + 3];
    }
  }
#pragma unroll
  for (int i = 0; i < 6; i++) {
    dyn_body_aamm[tid * 6 + i] = aamm[i];
  }
}

__device__ __forceinline__ uint map_body_pair_nxn(const uint tid,
                                                  const uint nenv,
                                                  const uint nbody) {
  if (tid >= nenv * nbody * nbody) return FULL_MASK;
  const uint body_pair_id = tid % (nbody * nbody);
  const uint body1 = body_pair_id / nbody;
  const uint body2 = body_pair_id % nbody;
  return (body1 < body2) ? body1 + body2 * nbody : FULL_MASK;
}

__global__ void get_body_pairs_nxn(
    const uint nenv, const uint nbody, const bool filter_parent,
    const uint nexclude, int* __restrict body_parentid,
    int* __restrict body_weldid, int* __restrict body_contype,
    int* __restrict body_conaffinity, uint* __restrict body_has_plane,
    int* __restrict exclude_signature, float* __restrict dyn_body_aamm,
    int* __restrict col_body_pair, uint* __restrict col_body_pair_count) {
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint map_coord = map_body_pair_nxn(tid, nenv, nbody);
  if (map_coord == FULL_MASK) return;

  const uint env_id = tid / (nbody * nbody);
  const uint body1 = map_coord % nbody;
  const uint body2 = map_coord / nbody;

  if ((!body_contype[body1] && !body_conaffinity[body1]) ||
      (!body_contype[body2] && !body_conaffinity[body2])) {
    return;
  }

  // TODO(btaba): use thrust for the set intersection.
  int signature = (body1 << 16) + body2;
  for (int i = 0; i < nexclude; i++) {
    if (exclude_signature[i] == signature) return;
  }

  int w1 = body_weldid[body1];
  int w2 = body_weldid[body2];
  if (w1 == w2) {
    return;
  }

  int w1_p = body_weldid[body_parentid[w1]];
  int w2_p = body_weldid[body_parentid[w2]];
  if (filter_parent && w1 != 0 && w2 != 0 && (w1 == w2_p || w2 == w1_p)) {
    return;
  }

  const float* aamm1 = dyn_body_aamm + (env_id * nbody + body1) * 6;
  const float* aamm2 = dyn_body_aamm + (env_id * nbody + body2) * 6;
  const bool separating = ((aamm1[0] > aamm2[3]) || (aamm1[1] > aamm2[4]) ||
                           (aamm1[2] > aamm2[5]) || (aamm2[0] > aamm1[3]) ||
                           (aamm2[1] > aamm1[4]) || (aamm2[2] > aamm1[5]));
  if (separating && !(body_has_plane[body1] || body_has_plane[body2])) {
    return;
  }

  uint idx = atomicAdd(&col_body_pair_count[env_id], 1);
  uint nbody_pair = ((nbody * (nbody - 1) / 2 + 15) / 16) * 16;
  col_body_pair[(env_id * nbody_pair + idx) * 2] = body1;
  col_body_pair[(env_id * nbody_pair + idx) * 2 + 1] = body2;
}

__global__ void get_dyn_geom_aabb(const uint nenv, const uint nmodel,
                                  const uint ngeom,
                                  float* __restrict geom_xpos,
                                  float* __restrict geom_xmat,
                                  float* __restrict geom_aabb,
                                  float* __restrict dyn_aabb) {
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nenv * ngeom) return;
  const uint env_id = tid / ngeom;
  const uint gid = tid % ngeom;

  float4 mat[3];
  xposmat_to_float4(geom_xpos, geom_xmat, env_id * ngeom + gid, mat);

  const float aabb[3] = {geom_aabb[gid * 6 + 3], geom_aabb[gid * 6 + 4],
                         geom_aabb[gid * 6 + 5]};
  // Grab the center of the AABB in the geom frame.
  const float3 aabb_pos = make_float3(
      geom_aabb[gid * 6], geom_aabb[gid * 6 + 1], geom_aabb[gid * 6 + 2]);

  const float3 corner[8] = {make_float3(-aabb[0], -aabb[1], -aabb[2]),
                            make_float3(aabb[0], -aabb[1], -aabb[2]),
                            make_float3(-aabb[0], aabb[1], -aabb[2]),
                            make_float3(aabb[0], aabb[1], -aabb[2]),
                            make_float3(-aabb[0], -aabb[1], aabb[2]),
                            make_float3(aabb[0], -aabb[1], aabb[2]),
                            make_float3(-aabb[0], aabb[1], aabb[2]),
                            make_float3(aabb[0], aabb[1], aabb[2])};

  float aabb_max[3] = {-__FLT_MAX__, -__FLT_MAX__, -__FLT_MAX__};
  float aabb_min[3] = {__FLT_MAX__, __FLT_MAX__, __FLT_MAX__};

#pragma unroll
  for (int k = 0; k < 8; k++) {
    float3 r;
    mulMatVec3(r, mat, corner[k] + aabb_pos);

    aabb_max[0] = (aabb_max[0] > r.x) ? aabb_max[0] : r.x;
    aabb_max[1] = (aabb_max[1] > r.y) ? aabb_max[1] : r.y;
    aabb_max[2] = (aabb_max[2] > r.z) ? aabb_max[2] : r.z;
    aabb_min[0] = (aabb_min[0] < r.x) ? aabb_min[0] : r.x;
    aabb_min[1] = (aabb_min[1] < r.y) ? aabb_min[1] : r.y;
    aabb_min[2] = (aabb_min[2] < r.z) ? aabb_min[2] : r.z;
  }

  dyn_aabb[tid * 6 + 0] = mat[0].w + aabb_min[0];
  dyn_aabb[tid * 6 + 1] = mat[1].w + aabb_min[1];
  dyn_aabb[tid * 6 + 2] = mat[2].w + aabb_min[2];
  dyn_aabb[tid * 6 + 3] = mat[0].w + aabb_max[0];
  dyn_aabb[tid * 6 + 4] = mat[1].w + aabb_max[1];
  dyn_aabb[tid * 6 + 5] = mat[2].w + aabb_max[2];
}

__global__ void get_geom_pairs_nxn(
    const uint nenv, const uint ngeom, const uint nbody, const uint n_geom_pair,
    const uint* __restrict body_geomnum, const int* __restrict body_geomadr,
    const int* __restrict geom_contype, const int* __restrict geom_conaffinity,
    const int* __restrict geom_type, const float* __restrict geom_margin,
    const float* __restrict dyn_geom_aabb, const int* __restrict col_body_pair,
    const uint* __restrict col_body_pair_count,
    const uint* __restrict col_body_pair_offset, int* __restrict col_geom_pair,
    uint* __restrict col_geom_pair_count) {
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint env_id = bisection<uint>(col_body_pair_offset, tid, 0, nenv - 1);
  const int body_pair_id = tid - col_body_pair_offset[env_id];
  if (body_pair_id >= col_body_pair_count[env_id]) return;

  uint nbody_pair = ((nbody * (nbody - 1) / 2 + 15) / 16) * 16;
  const int body1 = col_body_pair[(env_id * nbody_pair + body_pair_id) * 2];
  const int body2 = col_body_pair[(env_id * nbody_pair + body_pair_id) * 2 + 1];

  // TODO(btaba): try sweep and prune or BVH instead of checking all pairs.
  for (int g1 = 0; g1 < body_geomnum[body1]; g1++) {
    int geom1 = body_geomadr[body1] + g1;
    for (int g2 = 0; g2 < body_geomnum[body2]; g2++) {
      int geom2 = body_geomadr[body2] + g2;

      // Check plane/hfield types.
      const int type1 = geom_type[geom1];
      const int type2 = geom_type[geom2];
      int skip_type = ((type1 == mjGEOM_HFIELD || type1 == mjGEOM_PLANE) &&
                       (type2 == mjGEOM_HFIELD || type2 == mjGEOM_PLANE));

      // Checks contype and conaffinity.
      int skip_con = !((geom_contype[geom1] & geom_conaffinity[geom2]) |
                       (geom_contype[geom2] & geom_conaffinity[geom1]));

      // TODO(btaba): check overlap with explicit pairs.

      // Check AABB intersection.
      const float* aabb1 = dyn_geom_aabb + (env_id * ngeom + geom1) * 6;
      const float* aabb2 = dyn_geom_aabb + (env_id * ngeom + geom2) * 6;
      const bool separating = ((aabb1[0] > aabb2[3]) || (aabb1[1] > aabb2[4]) ||
                               (aabb1[2] > aabb2[5]) || (aabb2[0] > aabb1[3]) ||
                               (aabb2[1] > aabb1[4]) || (aabb2[2] > aabb1[5]));

      if (separating | skip_con | skip_type) {
        continue;
      }

      if (type1 > type2) std::swap(geom1, geom2);
      const int pair_id = atomicAdd(&col_geom_pair_count[env_id], 1);
      col_geom_pair[(env_id * n_geom_pair + pair_id) * 2] = geom1;
      col_geom_pair[(env_id * n_geom_pair + pair_id) * 2 + 1] = geom2;
    }
  }
}

__global__ void group_contacts_by_type(
    const uint nenv, const uint n_geom_pair, const uint n_geom_types,
    const int* geom_type, const int* __restrict col_geom_pair,
    const uint* __restrict col_geom_pair_count,
    const uint* __restrict col_geom_pair_offset,
    const uint* __restrict type_pair_offset, uint* __restrict type_pair_env_id,
    uint* __restrict type_pair_geom_id, uint* __restrict type_pair_count) {
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint env_id = bisection<uint>(col_geom_pair_offset, tid, 0, nenv - 1);
  const int pair_id = tid - col_geom_pair_offset[env_id];
  if (pair_id >= col_geom_pair_count[env_id]) return;

  uint geom1 = col_geom_pair[(env_id * n_geom_pair + pair_id) * 2];
  uint geom2 = col_geom_pair[(env_id * n_geom_pair + pair_id) * 2 + 1];

  // Get the grouping key.
  int type1 = geom_type[geom1];
  int type2 = geom_type[geom2];
  const int group_key = type1 + type2 * n_geom_types;

  // Add a geom pair to the group and record the env_id.
  const uint n_type_pair = atomicAdd(&type_pair_count[group_key], 1);
  int type_pair_id = type_pair_offset[group_key] * nenv + n_type_pair;
  type_pair_env_id[type_pair_id] = env_id;
  type_pair_geom_id[type_pair_id * 2] = geom1;
  type_pair_geom_id[type_pair_id * 2 + 1] = geom2;
}

template <typename T>
__global__ void set_zero(const uint count, T* __restrict ptr) {
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  ptr[tid] = 0;
}

__global__ void get_contact_solver_params(
    const uint nenv, const uint nmodel, const uint ngeom,
    const int max_contact_pts, const int n_contact_pts,
    const int* __restrict geom1, const int* __restrict geom2,
    const int* __restrict geom_priority, const float* __restrict geom_solmix,
    const float* __restrict geom_friction, const float* __restrict geom_solref,
    const float* __restrict geom_solimp, const float* __restrict geom_margin,
    const float* __restrict geom_gap, const uint* __restrict env_contact_offset,
    float* __restrict includemargin, float* __restrict friction,
    float* __restrict solref, float* __restrict solreffriction,
    float* __restrict solimp) {
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_contact_pts) return;

  const uint env_id = bisection<uint>(env_contact_offset, tid, 0, nenv - 1);
  const uint model_id = env_id % nmodel;
  const uint pt_id =
      env_id * max_contact_pts + tid - env_contact_offset[env_id];

  const int g1 = geom1[pt_id] + model_id * ngeom;
  const int g2 = geom2[pt_id] + model_id * ngeom;

  // TODO(btaba): add explicit geom pair logic.
  float margin =
      geom_margin[g1] > geom_margin[g2] ? geom_margin[g1] : geom_margin[g2];
  float gap = geom_gap[g1] > geom_gap[g2] ? geom_gap[g1] : geom_gap[g2];
  float solmix1 = geom_solmix[g1];
  float solmix2 = geom_solmix[g2];
  float mix = solmix1 / (solmix1 + solmix2);
  mix = (solmix1 < mjMINVAL) && (solmix2 < mjMINVAL) ? 0.5 : mix;
  mix = (solmix1 < mjMINVAL) && (solmix2 >= mjMINVAL) ? 0.0 : mix;
  mix = (solmix1 >= mjMINVAL) && (solmix2 < mjMINVAL) ? 1.0 : mix;
  const float* friction1 = geom_friction + g1 * 3;
  const float* friction2 = geom_friction + g2 * 3;
  const float* solref1 = geom_solref + g1 * mjNREF;
  const float* solref2 = geom_solref + g2 * mjNREF;
  const float* solimp1 = geom_solimp + g1 * mjNIMP;
  const float* solimp2 = geom_solimp + g2 * mjNIMP;
  const int p1 = geom_priority[g1], p2 = geom_priority[g2];

  mix = p1 == p2 ? mix : ((p1 > p2) ? 1.0 : 0.0);
  bool is_standard = (solref1[0] > 0) && (solref2[0] > 0);
  float solref_[mjNREF];
#pragma unroll
  for (int i = 0; i < mjNREF; i++) {
    solref_[i] = mix * solref1[i] + (1 - mix) * solref2[i];
    solref_[i] = is_standard ? solref_[i] : min(solref1[i], solref2[i]);
  }

  float solimp_[mjNIMP];
#pragma unroll
  for (int i = 0; i < mjNIMP; i++) {
    solimp_[i] = mix * solimp1[i] + (1 - mix) * solimp2[i];
  }

  float friction_[3];
#pragma unroll
  for (int i = 0; i < 3; i++) {
    friction_[i] = max(friction1[i], friction2[i]);
  }

  includemargin[tid] = margin - gap;
  friction[tid * 5] = friction_[0];
  friction[tid * 5 + 1] = friction_[0];
  friction[tid * 5 + 2] = friction_[1];
  friction[tid * 5 + 3] = friction_[2];
  friction[tid * 5 + 4] = friction_[2];
#pragma unroll
  for (int i = 0; i < mjNREF; i++) {
    solref[tid * mjNREF + i] = solref_[i];
  }
#pragma unroll
  for (int i = 0; i < mjNIMP; i++) {
    solimp[tid * mjNIMP + i] = solimp_[i];
  }
}

bool collision(cudaStream_t stream, CollisionInput& input,
               CollisionOutput& output) {
  if (input.ngeom == 0) return true;

  // Initialize the output data.
  const int blockSize = 256;
  const int npts = input.max_contact_points * input.nenv;
  int gridSize = (npts + blockSize - 1) / blockSize;
  init<<<gridSize, blockSize, 0, stream>>>(
      input.max_contact_points, input.nenv, output.dist, output.pos,
      output.normal, output.g1, output.g2, output.includemargin,
      output.friction, output.solref, output.solreffriction, output.solimp);
  assert_cuda("kernel::init");

  // Initialize environment buffers.
  gridSize = (input.nenv + blockSize - 1) / blockSize;
  init_buffers<<<gridSize, blockSize, 0, stream>>>(
      input.nenv, output.env_counter, output.env_offset, output.env_counter2);
  assert_cuda("kernel::init_buffers");

  // Generate body AAMMs.
  // TODO(btaba): construct and use geom_xpos covariance-aligned frame.
  gridSize = (input.nenv * input.nbody + blockSize - 1) / blockSize;
  gridSize = gridSize > 0 ? gridSize : 1;
  get_dyn_body_aamm<<<gridSize, blockSize, 0, stream>>>(
      input.nenv, input.nbody, input.nmodel, input.ngeom, input.body_geomnum,
      input.body_geomadr, input.geom_margin, input.geom_xpos, input.geom_rbound,
      output.dyn_body_aamm);
  assert_cuda("kernel::dyn_body_aamm");

  // Generate body pairs (broadphase).
  // TODO(btaba): try sweep and prune for broadphase instead.
  // TODO(btaba): if we wind up really using nxn body pairs, consider filtering
  //   a pre-computed body pair list, rather than computing it on the fly.
  uint nxn = input.nenv * input.nbody * input.nbody;
  gridSize = (nxn + blockSize - 1) / blockSize;
  uint* col_body_pair_count = output.env_counter;
  get_body_pairs_nxn<<<gridSize, blockSize, 0, stream>>>(
      input.nenv, input.nbody, input.filter_parent, input.nexclude,
      input.body_parentid, input.body_weldid, input.body_contype,
      input.body_conaffinity, input.body_has_plane, input.exclude_signature,
      output.dyn_body_aamm, output.col_body_pair, col_body_pair_count);
  assert_cuda("kernel::body_pairs");

  // Get geom AABBs in global frame.
  gridSize = (input.nenv * input.ngeom + blockSize - 1) / blockSize;
  get_dyn_geom_aabb<<<gridSize, blockSize, 0, stream>>>(
      input.nenv, input.nmodel, input.ngeom, input.geom_xpos, input.geom_xmat,
      input.geom_aabb, output.dyn_geom_aabb);
  assert_cuda("kernel::dyn_geom_aabb");

  // TODO(btaba): get explicit geom pairs and add them for each env.

  // Generate geom pairs (midphase).
  cudaDeviceSynchronize();
  int total_body_pairs = thrust::reduce(thrust::device, col_body_pair_count,
                                        col_body_pair_count + input.nenv, 0);
  thrust::device_ptr<uint> body_pair_offset(output.env_offset);
  thrust::exclusive_scan(thrust::device, col_body_pair_count,
                         col_body_pair_count + input.nenv, body_pair_offset);
  uint* col_geom_pair_count = output.env_counter2;
  gridSize = (total_body_pairs + blockSize - 1) / blockSize;
  gridSize = gridSize > 0 ? gridSize : 1;
  get_geom_pairs_nxn<<<gridSize, blockSize, 0, stream>>>(
      input.nenv, input.ngeom, input.nbody, input.n_geom_pair,
      input.body_geomnum, input.body_geomadr, input.geom_contype,
      input.geom_conaffinity, input.geom_type, input.geom_margin,
      output.dyn_geom_aabb, output.col_body_pair, output.env_counter,
      body_pair_offset.get(), output.col_geom_pair, col_geom_pair_count);
  assert_cuda("kernel::geom_pairs");

  // Initialize type pair count.
  gridSize =
      (input.n_geom_types * input.n_geom_types + blockSize - 1) / blockSize;
  set_zero<uint>
      <<<gridSize, blockSize, 0, stream>>>(
          input.n_geom_types * input.n_geom_types, output.type_pair_count);
  assert_cuda("kernel::init_type_pair_count");

  // Group geom pairs by type.
  // TODO(btaba): try moving the grouping into the templated narrowphase call.
  cudaDeviceSynchronize();
  int total_geom_pairs = thrust::reduce(thrust::device, col_geom_pair_count,
                                        col_geom_pair_count + input.nenv, 0);
  thrust::device_ptr<uint> col_geom_pair_offset(output.env_offset);
  thrust::exclusive_scan(thrust::device, col_geom_pair_count,
                         col_geom_pair_count + input.nenv,
                         col_geom_pair_offset);
  gridSize = (total_geom_pairs + blockSize - 1) / blockSize;
  gridSize = gridSize > 0 ? gridSize : 1;
  group_contacts_by_type<<<gridSize, blockSize, 0, stream>>>(
      input.nenv, input.n_geom_pair, input.n_geom_types, input.geom_type,
      output.col_geom_pair, col_geom_pair_count, col_geom_pair_offset.get(),
      input.type_pair_offset, output.type_pair_env_id, output.type_pair_geom_id,
      output.type_pair_count);
  assert_cuda("kernel::group_contacts_by_type");

  // Initialize the env contact counter.
  gridSize = (input.nenv * input.ngeom + blockSize - 1) / blockSize;
  uint* env_contact_count = output.env_counter;
  set_zero<uint>
      <<<gridSize, blockSize, 0, stream>>>(input.nenv, env_contact_count);
  assert_cuda("kernel::init_env_contact_count");

  // Dispatch to narrowphase collision functions.
  narrowphase(stream, input, output);

  // Generate contact solver params.
  cudaDeviceSynchronize();
  int n_contact_pts = thrust::reduce(thrust::device, env_contact_count,
                                     env_contact_count + input.nenv, 0);
  thrust::device_ptr<uint> env_contact_offset(output.env_offset);
  thrust::exclusive_scan(thrust::device, env_contact_count,
                         env_contact_count + input.nenv, env_contact_offset);
  gridSize = (n_contact_pts + blockSize - 1) / blockSize;
  gridSize = gridSize > 0 ? gridSize : 1;
  get_contact_solver_params<<<gridSize, blockSize, 0, stream>>>(
      input.nenv, input.nmodel, input.ngeom, input.max_contact_points,
      n_contact_pts, output.g1, output.g2, input.geom_priority,
      input.geom_solmix, input.geom_friction, input.geom_solref,
      input.geom_solimp, input.geom_margin, input.geom_gap,
      env_contact_offset.get(), output.includemargin, output.friction,
      output.solref, output.solreffriction, output.solimp);
  assert_cuda("kernel::get_contact_solver_params");

  return true;
}

xla::ffi::Error LaunchKernel_Collision_Driver(
    cudaStream_t stream,
    // Data.
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_xpos,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_xmat,
    // Model.
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_size,
    xla::ffi::Buffer<xla::ffi::DataType::S32> geom_type,
    xla::ffi::Buffer<xla::ffi::DataType::S32> geom_contype,
    xla::ffi::Buffer<xla::ffi::DataType::S32> geom_conaffinity,
    xla::ffi::Buffer<xla::ffi::DataType::S32> geom_priority,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_margin,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_gap,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_solmix,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_friction,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_solref,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_solimp,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_aabb,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_rbound,
    xla::ffi::Buffer<xla::ffi::DataType::S32> geom_dataid,
    xla::ffi::Buffer<xla::ffi::DataType::S32> geom_bodyid,
    xla::ffi::Buffer<xla::ffi::DataType::S32> body_parentid,
    xla::ffi::Buffer<xla::ffi::DataType::S32> body_weldid,
    xla::ffi::Buffer<xla::ffi::DataType::S32> body_contype,
    xla::ffi::Buffer<xla::ffi::DataType::S32> body_conaffinity,
    xla::ffi::Buffer<xla::ffi::DataType::S32> body_geomadr,
    xla::ffi::Buffer<xla::ffi::DataType::U32> body_geomnum,
    xla::ffi::Buffer<xla::ffi::DataType::U32> body_has_plane,
    xla::ffi::Buffer<xla::ffi::DataType::S32> pair_geom1,
    xla::ffi::Buffer<xla::ffi::DataType::S32> pair_geom2,
    xla::ffi::Buffer<xla::ffi::DataType::S32> exclude_signature,
    xla::ffi::Buffer<xla::ffi::DataType::F32> pair_margin,
    xla::ffi::Buffer<xla::ffi::DataType::F32> pair_gap,
    xla::ffi::Buffer<xla::ffi::DataType::F32> pair_friction,
    xla::ffi::Buffer<xla::ffi::DataType::F32> pair_solref,
    xla::ffi::Buffer<xla::ffi::DataType::F32> pair_solimp,
    xla::ffi::Buffer<xla::ffi::DataType::F32> convex_vert,
    xla::ffi::Buffer<xla::ffi::DataType::U32> convex_vert_offset,
    xla::ffi::Buffer<xla::ffi::DataType::U32> type_pair_offset,
    xla::ffi::Buffer<xla::ffi::DataType::U32> type_pair_count,
    // Static arguments.
    uint ngeom, uint npair, uint nbody, uint nexclude, uint max_contact_points,
    uint n_geom_pair, uint n_geom_types, bool filter_parent,
    // GJK/EPA arguments.
    float depth_extension, uint gjk_iteration_count, uint epa_iteration_count,
    uint epa_best_count, uint multi_polygon_count, float multi_tilt_angle,
    // Output buffers.
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> dist,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> pos,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> normal,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::S32>> g1,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::S32>> g2,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> includemargin,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> friction,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> solref,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> solreffriction,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> solimp,
    // Buffers used for internal computation.
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> dyn_body_aamm,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::S32>> col_body_pair,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> env_counter,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> env_counter2,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> env_offset,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> dyn_geom_aabb,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::S32>> col_geom_pair,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>>
        type_pair_env_id,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>>
        type_pair_geom_id) {
  CollisionInput input;
  input.ngeom = ngeom;
  input.npair = npair;
  input.nbody = nbody;
  input.nexclude = nexclude;
  input.max_contact_points = max_contact_points;
  input.n_geom_pair = n_geom_pair;
  input.n_geom_types = n_geom_types;
  input.filter_parent = filter_parent;

  // Get the batch size of mjx.Data.
  input.nenv = 1;
  for (int i = 0; i < geom_xpos.dimensions().size() - 1; i++) {
    input.nenv *= geom_xpos.dimensions()[i];
  }
  input.nenv /= ngeom;
  if (input.nenv == 0) {
    return xla::ffi::Error(
        XLA_FFI_Error_Code_INVALID_ARGUMENT,
        std::string("Batch size of mjx.Data calculated in LaunchKernel_GJK_EPA "
                    "is 0."));
  }

  // Get the batch size of mjx.Model.
  input.nmodel = 1;
  for (int i = 0; i < geom_size.dimensions().size() - 1; i++) {
    input.nmodel *= geom_size.dimensions()[i];
  }
  input.nmodel /= ngeom;
  if (input.nmodel == 0) {
    return xla::ffi::Error(
        XLA_FFI_Error_Code_INVALID_ARGUMENT,
        std::string(
            "Batch size of mjx.Model calculated in LaunchKernel_GJK_EPA "
            "is 0."));
  }
  if (input.nmodel > 1 && input.nmodel != input.nenv) {
    return xla::ffi::Error(
        XLA_FFI_Error_Code_INVALID_ARGUMENT,
        std::string(
            "Batch size of mjx.Model is greater than 1 and does not match the "
            "batch size of mjx.Data in LaunchKernel_GJK_EPA."));
  }

  bool c = true;
  int nge = ngeom * input.nenv;
  int ngm = ngeom * input.nmodel;
  c &= assert_count(geom_xpos.element_count(), nge * 3, "geom_xpos");
  c &= assert_count(geom_xmat.element_count(), nge * 9, "geom_xmat");
  c &= assert_count(geom_size.element_count(), ngm * 3, "geom_size");
  c &= assert_count(geom_type.element_count(), ngeom, "geom_type");
  c &= assert_count(geom_contype.element_count(), ngeom, "geom_contype");
  c &=
      assert_count(geom_conaffinity.element_count(), ngeom, "geom_conaffinity");
  c &= assert_count(geom_priority.element_count(), ngeom, "geom_priority");
  c &= assert_count(geom_margin.element_count(), ngm, "geom_margin");
  c &= assert_count(geom_gap.element_count(), ngm, "geom_gap");
  c &= assert_count(geom_solmix.element_count(), ngm, "geom_solmix");
  c &= assert_count(geom_friction.element_count(), ngm * 3, "geom_friction");
  c &= assert_count(geom_solref.element_count(), ngm * mjNREF, "geom_solref");
  c &= assert_count(geom_solimp.element_count(), ngm * mjNIMP, "geom_solimp");
  c &= assert_count(geom_aabb.element_count(), ngeom * 6, "geom_aabb");
  c &= assert_count(geom_rbound.element_count(), ngm, "geom_rbound");
  c &= assert_count(geom_dataid.element_count(), ngeom, "geom_dataid");
  c &= assert_count(geom_bodyid.element_count(), ngeom, "geom_bodyid");
  c &= assert_count(body_parentid.element_count(), nbody, "body_parentid");
  c &= assert_count(body_weldid.element_count(), nbody, "body_weldid");
  c &= assert_count(body_contype.element_count(), nbody, "body_contype");
  c &=
      assert_count(body_conaffinity.element_count(), nbody, "body_conaffinity");
  c &= assert_count(body_geomadr.element_count(), nbody, "body_geomadr");
  c &= assert_count(body_geomnum.element_count(), nbody, "body_geomnum");
  c &= assert_count(body_has_plane.element_count(), nbody, "body_has_plane");
  c &= assert_count(pair_geom1.element_count(), npair, "pair_geom1");
  c &= assert_count(pair_geom2.element_count(), npair, "pair_geom2");
  c &= assert_count(exclude_signature.element_count(), nexclude,
                    "exclude_signature");
  c &= assert_count(pair_margin.element_count(), npair, "pair_margin");
  c &= assert_count(pair_gap.element_count(), npair, "pair_gap");
  c &= assert_count(pair_friction.element_count(), npair, "pair_friction");
  c &= assert_count(pair_solref.element_count(), npair, "pair_solref");
  c &= assert_count(pair_solimp.element_count(), npair, "pair_solimp");
  c &= assert_that(n_geom_types == mjxGEOM_size, "Unexpected mjGEOM_size.");
  // TODO(btaba): check shapes for convex_vert/offset.
  c &= assert_count(type_pair_offset.element_count(),
                    n_geom_types * n_geom_types, "type_pair_offset");
  if (!c) {
    return xla::ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                           "Invalid input size");
  }

  // Data.
  input.geom_xpos = geom_xpos.typed_data();
  input.geom_xmat = geom_xmat.typed_data();
  // Model.
  input.geom_size = geom_size.typed_data();
  input.geom_type = geom_type.typed_data();
  input.geom_contype = geom_contype.typed_data();
  input.geom_conaffinity = geom_conaffinity.typed_data();
  input.geom_priority = geom_priority.typed_data();
  input.geom_margin = geom_margin.typed_data();
  input.geom_gap = geom_gap.typed_data();
  input.geom_solmix = geom_solmix.typed_data();
  input.geom_friction = geom_friction.typed_data();
  input.geom_solref = geom_solref.typed_data();
  input.geom_solimp = geom_solimp.typed_data();
  input.geom_aabb = geom_aabb.typed_data();
  input.geom_rbound = geom_rbound.typed_data();
  input.geom_dataid = geom_dataid.typed_data();
  input.geom_bodyid = geom_bodyid.typed_data();
  input.body_parentid = body_parentid.typed_data();
  input.body_weldid = body_weldid.typed_data();
  input.body_contype = body_contype.typed_data();
  input.body_conaffinity = body_conaffinity.typed_data();
  input.body_geomadr = body_geomadr.typed_data();
  input.body_geomnum = body_geomnum.typed_data();
  input.body_has_plane = body_has_plane.typed_data();
  input.pair_geom1 = pair_geom1.typed_data();
  input.pair_geom2 = pair_geom2.typed_data();
  input.exclude_signature = exclude_signature.typed_data();
  input.pair_margin = pair_margin.typed_data();
  input.pair_gap = pair_gap.typed_data();
  input.pair_friction = pair_friction.typed_data();
  input.pair_solref = pair_solref.typed_data();
  input.pair_solimp = pair_solimp.typed_data();
  input.convex_vert = convex_vert.typed_data();
  input.convex_vert_offset = convex_vert_offset.typed_data();
  input.type_pair_offset = type_pair_offset.typed_data();
  // GJK/EPA parameters.
  input.depth_extension = depth_extension;
  input.gjk_iteration_count = gjk_iteration_count;
  input.epa_iteration_count = epa_iteration_count;
  input.epa_best_count = epa_best_count;
  input.multi_polygon_count = multi_polygon_count;
  input.multi_tilt_angle = multi_tilt_angle;

  uint npts = input.nenv * max_contact_points;
  c = true;
  c &= assert_count(type_pair_count.element_count(),
                    input.n_geom_types * input.n_geom_types, "type_pair_count");
  c &= assert_count(dist->element_count(), npts, "dist");
  c &= assert_count(pos->element_count(), npts * 3, "pos");
  c &= assert_count(normal->element_count(), npts * 3, "normal");
  c &= assert_count(g1->element_count(), npts, "g1");
  c &= assert_count(g2->element_count(), npts, "g2");
  c &= assert_count(includemargin->element_count(), npts, "includemargin");
  c &= assert_count(friction->element_count(), npts * 5, "friction");
  c &= assert_count(solref->element_count(), npts * mjNREF, "solref");
  c &= assert_count(solreffriction->element_count(), npts * mjNREF,
                    "solreffriction");
  c &= assert_count(solimp->element_count(), npts * mjNIMP, "solimp");
  c &= assert_count(dyn_body_aamm->element_count(),
                    input.nenv * input.nbody * 6, "dyn_body_aamm");
  int nbody_pair_buf_size =
      ((input.nbody * (input.nbody - 1) / 2 + 15) / 16) * 16;
  c &= assert_count(col_body_pair->element_count(),
                    input.nenv * nbody_pair_buf_size * 2, "col_body_pair");
  c &= assert_count(env_counter->element_count(), input.nenv, "env_counter");
  c &= assert_count(env_counter2->element_count(), input.nenv, "env_counter2");
  c &= assert_count(env_offset->element_count(), input.nenv, "env_offset");
  c &= assert_count(dyn_geom_aabb->element_count(), ngeom * input.nenv * 6,
                    "dyn_geom_aabb");
  c &= assert_count(col_geom_pair->element_count(),
                    input.nenv * input.n_geom_pair * 2, "col_geom_pair");
  c &= assert_count(type_pair_env_id->element_count(),
                    input.nenv * input.n_geom_pair, "type_pair_env_id");
  c &= assert_count(type_pair_geom_id->element_count(),
                    input.nenv * input.n_geom_pair * 2, "type_pair_geom_id");
  if (!c) {
    return xla::ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                           "Invalid output size");
  }

  CollisionOutput output;
  output.dist = dist->typed_data();
  output.pos = pos->typed_data();
  output.normal = normal->typed_data();
  output.g1 = g1->typed_data();
  output.g2 = g2->typed_data();
  output.includemargin = includemargin->typed_data();
  output.friction = friction->typed_data();
  output.solref = solref->typed_data();
  output.solreffriction = solreffriction->typed_data();
  output.solimp = solimp->typed_data();
  output.dyn_body_aamm = dyn_body_aamm->typed_data();
  output.col_body_pair = col_body_pair->typed_data();
  output.env_counter = env_counter->typed_data();
  output.env_offset = env_offset->typed_data();
  output.dyn_geom_aabb = dyn_geom_aabb->typed_data();
  output.col_geom_pair = col_geom_pair->typed_data();
  output.env_counter2 = env_counter2->typed_data();
  output.type_pair_count = type_pair_count.typed_data();
  output.type_pair_env_id = type_pair_env_id->typed_data();
  output.type_pair_geom_id = type_pair_geom_id->typed_data();

  if (!collision(stream, input, output)) {
    return xla::ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                           std::string("CUDA error::collision"));
  }

  cudaDeviceSynchronize();
  return xla::ffi::Error::Success();
}

}  // namespace mujoco::mjx::cuda
