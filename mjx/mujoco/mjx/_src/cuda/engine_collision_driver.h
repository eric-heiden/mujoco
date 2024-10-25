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

#ifndef MUJOCO_PYTHON_MJX_CUDA_ENGINE_COLLISION_DRIVER_H_
#define MUJOCO_PYTHON_MJX_CUDA_ENGINE_COLLISION_DRIVER_H_

#include <driver_types.h>  // cuda
#include <mujoco/mjmodel.h>
#include <xla/ffi/api/ffi.h>

namespace mujoco::mjx::cuda {

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
    // TODO(btaba): support condim.
    // xla::ffi::Buffer<xla::ffi::DataType::U32> geom_condim,
    // xla::ffi::Buffer<xla::ffi::DataType::U32> pair_dim,
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
        type_pair_geom_id);
}

#endif  // MUJOCO_PYTHON_MJX_CUDA_ENGINE_COLLISION_DRIVER_H_
