#include "mjx_cuda_collision.common.h"
#include <cmath>
#include <cstdarg>  // va_list, va_start, va_end
#include <cstdio>   // printf, vprintf, fprintf, vfprintf
#include <string>
#include <string.h>

namespace mujoco
{
	namespace mjx
	{
		namespace cuda
		{
			// =============================================================
			// cuda
			// =============================================================
			bool assert_cuda(const char* msg)
			{
				cudaError_t last_error = cudaGetLastError();

				if (last_error != cudaSuccess)
				{
					printf("[CUDA-ERROR] [%s] (%d:%s) \n", msg ? msg : "", (int) last_error, cudaGetErrorString(last_error));
				}

				return (last_error == cudaSuccess);
			}

			// =============================================================
			// jax_array
			// =============================================================
			bool assert_array_alloc(jax_array& x, const size_t sizeToAllocInBytes, const char* msg)
			{
				if (x.dev.sizeInBytes < sizeToAllocInBytes)
				{
					printf("[CUDA-ERROR] [%s] Invalid memory allocation (%zu < %zu) bytes\n",
						msg ? msg : "", x.dev.sizeInBytes, sizeToAllocInBytes);
					return false;
				}

				return true;
			}

			bool array_destroy_cpu(jax_array& x)
			{
				if (x.cpu.ptr) { delete[] x.cpu.i; x.cpu.ptr = NULL; x.cpu.sizeInBytes = 0; }
				return true;
			}

			bool array_destroy(jax_array& x)
			{
				array_destroy_cpu(x);
				return true;
			}

			bool array_resize_cpu(jax_array& x, const size_t sizeToAllocInBytes)
			{
				if(sizeToAllocInBytes == x.cpu.sizeInBytes){ return true;} // already allocated 
				_ASSERT_(array_destroy_cpu(x));
				x.cpu.ptr = new char[sizeToAllocInBytes];
				x.cpu.sizeInBytes = sizeToAllocInBytes;
				return true;
			}

		
			bool array_copy_from_device(jax_array& x, const size_t sizeToCopyInBytes)
			{
				if (x.dev.sizeInBytes == 0) { return true; }
				_ASSERT_(array_resize_cpu(x, x.dev.sizeInBytes));
				size_t devSizeToCopyInBytes = x.dev.sizeInBytes < sizeToCopyInBytes ? x.dev.sizeInBytes : sizeToCopyInBytes;
				cudaDeviceSynchronize();
				cudaMemcpy(x.cpu.ptr, x.dev.ptr, devSizeToCopyInBytes, cudaMemcpyDeviceToHost);
				_ASSERT_(assert_cuda("array_copy_from_device"));
				return true;
			}

			bool array_get_at(unsigned int& r, jax_array& x, const int offset)
			{
				_ASSERT_(assert_array_alloc(x, sizeof(unsigned int), ""));
				cudaDeviceSynchronize();
				cudaMemcpy(&r, x.dev.ui + offset, sizeof(unsigned int), cudaMemcpyDeviceToHost);
				_ASSERT_(assert_cuda("array_get_at"));
				return true;
			}

			bool array_zero(jax_array& x, const size_t sizeInBytes)
			{
				if (sizeInBytes == 0) { return true; }
				_ASSERT_(x.dev.ptr);

				cudaMemset(x.dev.ptr, 0, sizeInBytes);
				_ASSERT_(assert_cuda("array_zero"));
				return true;
			}

			jax_array::jax_array()
			{
				type = jax_array_type_default;
			}

			jax_array::jax_array(jax_array_type devArrayType, void* devPtr, const size_t devSizeInBytes, const char* _name)
			{
				type = devArrayType;
				dev.ptr = devPtr;
				dev.sizeInBytes = devSizeInBytes;
			}

			jax_array::~jax_array()
			{
				array_destroy_cpu(*this);
				memset(&dev, 0, sizeof(platform));
			}
		}
	}
}

