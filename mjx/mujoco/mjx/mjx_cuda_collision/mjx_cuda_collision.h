#ifndef MJX_CUDA_COLLISION_H
#define MJX_CUDA_COLLISION_H

#ifndef _WIN32
#   define _ENABLE_XLA_FFI_
#endif 

#ifdef _ENABLE_XLA_FFI_
#   include <driver_types.h>
#   include <xla/ffi/api/ffi.h>

namespace mujoco
{
    namespace mjx
    {
        namespace cuda
        {
            xla::ffi::Error LaunchKernel_Calculate_BVH_AABB_dyn(cudaStream_t stream,
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_bvh_aabb_dyn,   
                xla::ffi::Buffer<xla::ffi::DataType::F32> m_aabb,           
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_xpos,           
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_xmat,           
                const unsigned int m_ngeom,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> out);

            xla::ffi::Error LaunchKernel_CollisionBroadPhase_NxN(cudaStream_t stream,
                xla::ffi::Buffer<xla::ffi::DataType::U32> col_counter_nxn,  
                xla::ffi::Buffer<xla::ffi::DataType::U32> col_pair_nxn,     
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_bvh_aabb_dyn,   
                const unsigned int m_ngeom,
                const unsigned int max_geom_pairs,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> out);

            xla::ffi::Error LaunchKernel_CollisionBroadPhase_Sort(cudaStream_t stream,
                xla::ffi::Buffer<xla::ffi::DataType::U32> col_counter_sort,  
                xla::ffi::Buffer<xla::ffi::DataType::U32> col_pair_sort,   
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_bvh_aabb_dyn,  
                xla::ffi::Buffer<xla::ffi::DataType::U32> buffer,            
                const unsigned int m_ngeom,
                const unsigned int max_geom_pairs,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> out);

            xla::ffi::Error LaunchKernel_GJK_EPA(cudaStream_t stream,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> d_contact_counter,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_pos,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_dist,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_frame,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_normal,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_simplex,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::S32>> d_contact_pairs,
                xla::ffi::Buffer<xla::ffi::DataType::S32> d_candidate_pairs,
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_xpos,
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_xmat,
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_geom_size,
                xla::ffi::Buffer<xla::ffi::DataType::S32> m_geom_dataid,
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_convex_vertex_array,
                xla::ffi::Buffer<xla::ffi::DataType::S32> d_convex_vertex_offset,
                const unsigned int ncon,
                const unsigned int m_ngeom,
                const unsigned int candidatePairCountMax,
                const unsigned int candidatePairCount,
                const int pairGeomType1,
                const int pairGeomType2,
                const float depthExtension, 
                const unsigned int gjkIterationCount, 
                const unsigned int epaIterationCount, 
                const unsigned int epaBestCount, 
                const unsigned int multiPolygonCount, 
                const float multiTiltAngle,
                const unsigned int compress_result,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> out);

        }
    }
}  // namespace mujoco::mjx::cuda
#endif  //#ifdef _ENABLE_XLA_FFI_

#endif  // MJX_CUDA_COLLISION_H