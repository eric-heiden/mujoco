
pyenv activate apagom
SITE_PACKAGES=/usr/local/google/home/btaba/.pyenv/versions/apagom/lib/python3.12/site-packages/

export PATH=/opt/cuda/12.2.0/bin:${PATH}
export LD_LIBRARY_PATH=/opt/cudnn/8.9.7/lib:/opt/cuda/12.2.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export CMAKE_INCLUDE_PATH=/opt/cudnn/8.9.7/include:/opt/cuda/12.2.0/targets/x86_64-linux/include:${CMAKE_INCLUDE_PATH}
export CMAKE_LIBRARY_PATH=/opt/cudnn/8.9.7/lib:/opt/cuda/12.2.0/targets/x86_64-linux/lib:${CMAKE_LIBRARY_PATH}

g++-12 -fPIC -c mjx_cuda_collision.xla.ffi.cc -o mjx_cuda_collision.xla.ffi.cc.o -I/usr/local/cuda/targets/x86_64-linux/include/ -I${SITE_PACKAGES}/jaxlib/include/ -I${SITE_PACKAGES}/tensorflow/include/external/pybind11/include/ -I${SITE_PACKAGES}/tensorflow/include/external/local_config_python/python_include/ -I${SITE_PACKAGES}/mujoco/include && \
                g++-12 -fPIC -c mjx_cuda_collision.common.cpp -o mjx_cuda_collision.common.cpp.o -I/usr/local/cuda/targets/x86_64-linux/include/ -I${SITE_PACKAGES}/jaxlib/include/ -I${SITE_PACKAGES}/tensorflow/include/external/pybind11/include/ -I${SITE_PACKAGES}/tensorflow/include/external/local_config_python/python_include/ -I${SITE_PACKAGES}/mujoco/include/ && \
                nvcc -c -Xcompiler -fPIC -ccbin /usr/bin/g++-12  mjx_cuda_collision.cu -o mjx_cuda_collision.cu.o -I/usr/local/cuda/targets/x86_64-linux/include/ -I${SITE_PACKAGES}/jaxlib/include/ -I${SITE_PACKAGES}/tensorflow/include/external/pybind11/include/ -I${SITE_PACKAGES}/tensorflow/include/external/local_config_python/python_include/ -I${SITE_PACKAGES}/mujoco/include/ && \
                g++-12 -shared -o _mjx_cuda_collision.so mjx_cuda_collision.xla.ffi.cc.o mjx_cuda_collision.common.cpp.o mjx_cuda_collision.cu.o -L${SITE_PACKAGES}/tensorflow/ -ltensorflow_cc -L/usr/local/cuda/targets/x86_64-linux/lib/ -lcudart

cp *.so ../

# generate cubin
# nvcc -c -Xcompiler -fPIC -ccbin /usr/bin/g++-12 -cubin mjx_cuda_collision.cu -o mjx_cuda_collision.cubin -I/usr/local/cuda/targets/x86_64-linux/include/ -I${SITE_PACKAGES}/jaxlib/include/ -I${SITE_PACKAGES}/tensorflow/include/external/pybind11/include/ -I${SITE_PACKAGES}/tensorflow/include/external/local_config_python/python_include/ -I${SITE_PACKAGES}/mujoco/include/

# generate ptx
# nvcc -c -Xcompiler -fPIC -ccbin /usr/bin/g++-12  mjx_cuda_collision.cu -ptx   -I/usr/local/cuda/targets/x86_64-linux/include/ -I${SITE_PACKAGES}/jaxlib/include/ -I${SITE_PACKAGES}/tensorflow/include/external/pybind11/include/ -I${SITE_PACKAGES}/tensorflow/include/external/local_config_python/python_include/ -I${SITE_PACKAGES}/mujoco/include/
