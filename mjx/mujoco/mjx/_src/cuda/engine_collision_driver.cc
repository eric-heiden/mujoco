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

#include <driver_types.h>  // cuda
#include <pybind11/pybind11.h>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

namespace mujoco::mjx::cuda {

namespace ffi = xla::ffi;

static const auto *kCollisionDriver =
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        // Data.
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_xpos
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_xmat
        // Model.
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_size
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // geom_type
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // geom_contype
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // geom_conaffinity
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // geom_priority
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_margin
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_gap
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_solmix
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_friction
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_solref
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_solimp
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_aabb
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_rbound
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // geom_dataid
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // geom_bodyid
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // body_parentid
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // body_weldid
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // body_contype
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // body_conaffinity
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // body_geomadr
        .Arg<ffi::Buffer<ffi::DataType::U32>>()  // body_geomnum
        .Arg<ffi::Buffer<ffi::DataType::U32>>()  // body_has_plane
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // pair_geom1
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // pair_geom2
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // exclude_signature
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // pair_margin
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // pair_gap
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // pair_friction
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // pair_solref
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // pair_solimp
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // convex_vert
        .Arg<ffi::Buffer<ffi::DataType::U32>>()  // convex_vert_offset
        .Arg<ffi::Buffer<ffi::DataType::U32>>()  // type_pair_offset
        .Arg<ffi::Buffer<ffi::DataType::U32>>()  // type_pair_count
        // Static Arguments.
        .Attr<uint>("ngeom")
        .Attr<uint>("npair")
        .Attr<uint>("nbody")
        .Attr<uint>("nexclude")
        .Attr<uint>("max_contact_points")
        .Attr<uint>("n_geom_pair")
        .Attr<uint>("n_geom_types")
        .Attr<bool>("filter_parent")
        // GJK/EPA arguments.
        .Attr<float>("depth_extension")
        .Attr<uint>("gjk_iteration_count")
        .Attr<uint>("epa_iteration_count")
        .Attr<uint>("epa_best_count")
        .Attr<uint>("multi_polygon_count")
        .Attr<float>("multi_tilt_angle")
        // Output buffers.
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // dist
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // pos
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // normal
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // g1
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // g2
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // includemargin
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // friction
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // solref
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // solreffriction
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // solimp
        // Buffers used for internal computation.
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // dyn_body_aamm
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // col_body_pair
        .Ret<ffi::Buffer<ffi::DataType::U32>>()  // env_counter
        .Ret<ffi::Buffer<ffi::DataType::U32>>()  // env_counter2
        .Ret<ffi::Buffer<ffi::DataType::U32>>()  // env_offset
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // dyn_geom_aabb
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // col_geom_pair
        .Ret<ffi::Buffer<ffi::DataType::U32>>()  // type_pair_env_id
        .Ret<ffi::Buffer<ffi::DataType::U32>>()  // type_pair_geom_id
        .To(LaunchKernel_Collision_Driver)
        .release();

XLA_FFI_Error *collision(XLA_FFI_CallFrame *call_frame) {
  return kCollisionDriver->Call(call_frame);
}

namespace {

namespace py = pybind11;

template <typename T>
py::capsule EncapsulateFfiCall(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

PYBIND11_MODULE(_engine_collision_driver, m) {
  m.def("collision", []() { return EncapsulateFfiCall(collision); });
}

}  // namespace

}  // namespace mujoco::mjx::cuda
