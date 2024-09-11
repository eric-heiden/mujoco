#include "mjx_cuda_collision.h"

#ifdef _ENABLE_XLA_FFI_
#include <cstring>
#include <pybind11/pybind11.h>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

namespace mujoco 
{
  namespace mjx
  {
    namespace cuda
    {
        namespace ffi = xla::ffi;

        static const auto* kCalculate_BVH_AABB_dyn = ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t> >()  // stream
            .Arg<ffi::Buffer<ffi::DataType::F32> >()    // d_bvh_aabb_dyn
            .Arg<ffi::Buffer<ffi::DataType::F32> >()    // m_aabb
            .Arg<ffi::Buffer<ffi::DataType::F32> >()    // d_xpos
            .Arg<ffi::Buffer<ffi::DataType::F32> >()    // d_xmat
            .Attr<unsigned int>("m_ngeom")
            .Ret<ffi::Buffer<ffi::DataType::U32> >()    // out
            .To(LaunchKernel_Calculate_BVH_AABB_dyn)
            .release();

        XLA_FFI_Error* Calculate_BVH_AABB_dyn(XLA_FFI_CallFrame* call_frame)
        {
            return kCalculate_BVH_AABB_dyn->Call(call_frame);
        }

        static const auto* kCollisionBroadPhase_NxN = ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t> >()  // stream
            .Arg<ffi::Buffer<ffi::DataType::U32> >()    // col_counter_nxn
            .Arg<ffi::Buffer<ffi::DataType::U32> >()    // col_pair_nxn
            .Arg<ffi::Buffer<ffi::DataType::F32> >()    // d_bvh_aabb_dyn
            .Attr<unsigned int>("m_ngeom")
            .Attr<unsigned int>("max_geom_pairs")
            .Ret<ffi::Buffer<ffi::DataType::U32> >()    // out
            .To(LaunchKernel_CollisionBroadPhase_NxN)
            .release();

        XLA_FFI_Error* CollisionBroadPhase_NxN(XLA_FFI_CallFrame* call_frame)
        {
            return kCollisionBroadPhase_NxN->Call(call_frame);
        }

        static const auto* kCollisionBroadPhase_Sort = ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t> >()  // stream
            .Arg<ffi::Buffer<ffi::DataType::U32> >()    // col_counter_nxn
            .Arg<ffi::Buffer<ffi::DataType::U32> >()    // col_pair_nxn
            .Arg<ffi::Buffer<ffi::DataType::F32> >()    // d_bvh_aabb_dyn
            .Arg<ffi::Buffer<ffi::DataType::U32> >()    // buffer
            .Attr<unsigned int>("m_ngeom")
            .Attr<unsigned int>("max_geom_pairs")
            .Ret<ffi::Buffer<ffi::DataType::U32> >()    // out
            .To(LaunchKernel_CollisionBroadPhase_Sort)
            .release();

        XLA_FFI_Error* CollisionBroadPhase_Sort(XLA_FFI_CallFrame* call_frame)
        {
            return kCollisionBroadPhase_Sort->Call(call_frame);
        }

        static const auto* kGJK_EPA = ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t> >()  // stream
            .Arg<ffi::Buffer<ffi::DataType::U32> >() // d_contact_counter,
            .Arg<ffi::Buffer<ffi::DataType::F32> >() // d_contact_pos,
            .Arg<ffi::Buffer<ffi::DataType::F32> >() // d_contact_dist,
            .Arg<ffi::Buffer<ffi::DataType::F32> >() // d_contact_frame,
            .Arg<ffi::Buffer<ffi::DataType::F32> >() // d_contact_normal,
            .Arg<ffi::Buffer<ffi::DataType::F32> >() // d_contact_simplex,
            .Arg<ffi::Buffer<ffi::DataType::S32> >() // d_contact_pairs,
            .Arg<ffi::Buffer<ffi::DataType::S32> >() // d_candidate_pairs,
            .Arg<ffi::Buffer<ffi::DataType::F32> >() // d_xpos,
            .Arg<ffi::Buffer<ffi::DataType::F32> >() // d_xmat,
            .Arg<ffi::Buffer<ffi::DataType::F32> >() // d_geom_size,
            .Arg<ffi::Buffer<ffi::DataType::S32> >() // m_geom_dataid,
            .Arg<ffi::Buffer<ffi::DataType::F32> >() // d_convex_vertex_array,
            .Arg<ffi::Buffer<ffi::DataType::S32> >() // d_convex_vertex_offset,
            .Attr<unsigned int>("ncon")
            .Attr<unsigned int>("m_ngeom")
            .Attr<unsigned int>("candidate_pair_count_max")
            .Attr<unsigned int>("candidate_pair_count")
            .Attr<int>("key_types0")
            .Attr<int>("key_types1")
            .Attr<float>("depthExtension")
            .Attr<unsigned int>("gjkIterationCount")
            .Attr<unsigned int>("epaIterationCount")
            .Attr<unsigned int>("epaBestCount")
            .Attr<unsigned int>("multiPolygonCount")
            .Attr<float>("multiTiltAngle")
            .Attr<unsigned int>("compress_result")
            .Ret<ffi::Buffer<ffi::DataType::U32> >() // out
            .To(LaunchKernel_GJK_EPA)
            .release();

        XLA_FFI_Error* GJK_EPA(XLA_FFI_CallFrame* call_frame)
        {
            return kGJK_EPA->Call(call_frame);
        }

      namespace {

      namespace py = pybind11;

      template <typename T>
      py::capsule EncapsulateFfiCall(T *fn) {
        static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                      "Encapsulated function must be and XLA FFI handler");
        return py::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
      }

      PYBIND11_MODULE(_mjx_cuda_collision, m) {
        m.def("bvh_aabb_dyn", []() { return EncapsulateFfiCall(Calculate_BVH_AABB_dyn); });
        m.def("broad_phase_nxn", []() { return EncapsulateFfiCall(CollisionBroadPhase_NxN); });
        m.def("broad_phase_sort", []() { return EncapsulateFfiCall(CollisionBroadPhase_Sort); });
        m.def("gjk_epa", []() { return EncapsulateFfiCall(GJK_EPA); });
      }

      }  // namespace
    }
  }
}


#endif // _ENABLE_XLA_FFI_
