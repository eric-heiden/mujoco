# Build CUDA code externally
pyenv virtualenv 3.12 mjx-312
pyenv activate mjx-312

cd $HOME
mkdir git
cd git

git clone https://github.com/NVIDIA/cuda-samples
git clone https://github.com/openxla/xla
git clone https://github.com/google-deepmind/mujoco

pip install mujoco
# for pybind
pip install tensorflow
pip install tensorflow --no-binary=tensorflow
# need newest version for FFI
pip install --upgrade jax[cuda] jaxlib 

SITE_PACKAGES=$HOME/.pyenv/versions/mjx-312/lib/python3.12/site-packages

export PATH=/opt/cuda/12.2.0/bin:${PATH}
export LD_LIBRARY_PATH=/opt/cudnn/8.9.7/lib:/opt/cuda/12.2.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export CMAKE_INCLUDE_PATH=/opt/cudnn/8.9.7/include:/opt/cuda/12.2.0/targets/x86_64-linux/include:${CMAKE_INCLUDE_PATH}
export CMAKE_LIBRARY_PATH=/opt/cudnn/8.9.7/lib:/opt/cuda/12.2.0/targets/x86_64-linux/lib:${CMAKE_LIBRARY_PATH}

cd ~/git/mujoco/mjx/mujoco/mjx/_src/cuda

g++-12 -fPIC -c engine_collision_driver.cc -o engine_collision_driver.cc.o -I/usr/local/cuda/targets/x86_64-linux/include/ -I${SITE_PACKAGES}/jaxlib/include/ -I${SITE_PACKAGES}/tensorflow/include/external/pybind11/include/ -I${SITE_PACKAGES}/tensorflow/include/external/local_config_python/python_include/ -I${SITE_PACKAGES}/mujoco/include
g++-12 -fPIC -c engine_collision_convex.cc -o engine_collision_convex.cc.o -I/usr/local/cuda/targets/x86_64-linux/include/ -I${SITE_PACKAGES}/jaxlib/include/ -I${SITE_PACKAGES}/tensorflow/include/external/pybind11/include/ -I${SITE_PACKAGES}/tensorflow/include/external/local_config_python/python_include/ -I${SITE_PACKAGES}/mujoco/include -I${HOME}/git/cuda-samples/Common
nvcc -x cu -c -Xcompiler -fPIC -ccbin /usr/bin/g++-12  engine_collision_driver.cu.cc -o engine_collision_driver.cu.cc.o -I${HOME}/git/xla -I/usr/local/cuda/targets/x86_64-linux/include/ -I${SITE_PACKAGES}/jaxlib/include/ -I${SITE_PACKAGES}/tensorflow/include/external/pybind11/include/ -I${SITE_PACKAGES}/tensorflow/include/external/local_config_python/python_include/ -I${SITE_PACKAGES}/mujoco/include/ -I${HOME}/git/cuda-samples/Common
nvcc -x cu -c -Xcompiler -fPIC -ccbin /usr/bin/g++-12  engine_collision_convex.cu.cc -o engine_collision_convex.cu.cc.o -I${HOME}/git/xla -I/usr/local/cuda/targets/x86_64-linux/include/ -I${SITE_PACKAGES}/jaxlib/include/ -I${SITE_PACKAGES}/tensorflow/include/external/pybind11/include/ -I${SITE_PACKAGES}/tensorflow/include/external/local_config_python/python_include/ -I${SITE_PACKAGES}/mujoco/include/ -I${HOME}/git/cuda-samples/Common -I.

ln -s ${SITE_PACKAGES}/tensorflow/libtensorflow_cc.so.2  ${SITE_PACKAGES}/tensorflow/libtensorflow_cc.so 
g++-12 -shared -o _engine_collision_convex.so engine_collision_convex.cu.cc.o engine_collision_convex.cc.o  -L${SITE_PACKAGES}/tensorflow/ -ltensorflow_cc -L/usr/local/cuda/targets/x86_64-linux/lib/ -lcudart
g++-12 -shared -o _engine_collision_driver.so engine_collision_convex.cu.cc.o engine_collision_convex.cc.o engine_collision_driver.cc.o engine_collision_driver.cu.cc.o  -L${SITE_PACKAGES}/tensorflow/ -ltensorflow_cc -L/usr/local/cuda/targets/x86_64-linux/lib/ -lcudart

cd ~/git/mujoco/mjx
pip install -e .

# test it...
cd ~/git/mujoco/mjx/mujoco/mjx
MJX_CUDA_COLLISION_FULL=true python viewer.py --mjcf=test_data/convex.xml

