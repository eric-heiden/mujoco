#ifndef MJX_CUDA_COLLISION_COMMON_H
#define MJX_CUDA_COLLISION_COMMON_H

#include <stdio.h>
#include <chrono>           // timer
#include <cuda_runtime.h>

// disabled XLA_FFI compilation on windows
#ifndef _WIN32
#   define _ENABLE_XLA_FFI_
#endif 

#ifdef _ENABLE_XLA_FFI_
#   include <driver_types.h>
#   include <xla/ffi/api/ffi.h>
#endif 

namespace mujoco
{
    namespace mjx
    {
        namespace cuda
        {
            // =============================================================
            // jax_array
            // =============================================================
            typedef int jax_array_type;
            static const jax_array_type
                jax_array_type_int = 0,
                jax_array_type_uint = 1,
                jax_array_type_float = 2,
                jax_array_type_default = jax_array_type_int;

            // jax_array is a structure with reference pointers to jax.Array in MuJoCo
            struct jax_array	// jax.Array
            {
                struct platform
                {
                    union
                    {
                        float* f;
                        float4* f4;
                        int* i;
                        unsigned int* ui;
                        void* ptr;
                    };

                    size_t sizeInBytes;

                    platform()
                    {
                        ptr = NULL; sizeInBytes = 0;
                    }
                };

                jax_array_type type;

                platform dev;   // All device memory is a reference pointer from jax.Array
                platform cpu;   // CPU memory is allocated for GPU-CPU transfer

                jax_array();
                jax_array(jax_array_type devArrayType, void* devPtr, const size_t devSizeInBytes, const char* name = NULL);
                ~jax_array();
            };

            // =============================================================
            // cuda
            // =============================================================
#define _ASSERT_(_func) if(!(_func)){ printf("[CUDA-ERROR] %s\n", #_func); return false; }
            bool assert_cuda(const char* msg);
            bool assert_array_alloc(jax_array& x, const size_t sizeToAllocInBytes, const char* msg);
            
            bool array_resize_cpu(jax_array& x, const size_t sizeToAllocInBytes);   // resize cpu 
            bool array_destroy_cpu(jax_array& x);                                   // destroy cpu
            bool array_copy_from_device(jax_array& x, const size_t sizeToCopyInBytes = 0x7FFFFFFF);
            bool array_get_at(unsigned int& r, jax_array& x, const int offset);     // r = x[offset]
            bool array_zero(jax_array& x, const size_t sizeInBytes);                // x = 0
        }
    }
}
// namespace mujoco::mjx::cuda
#endif  // MJX_CUDA_COLLISION_COMMON_H