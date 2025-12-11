## OpenCV Requirement

```
# 1) Get sources
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir -p build && cd build

# 2) Configure: disable CUDA/OpenCL/OpenGL/Vulkan, keep only CPU+SIMD
cmake -S .. -B . \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  \
  -D WITH_CUDA=OFF \
  -D WITH_CUDNN=OFF \
  -D OPENCV_DNN_CUDA=OFF \
  -D WITH_OPENCL=OFF \
  -D WITH_OPENGL=OFF \
  -D WITH_VULKAN=OFF \
  -D WITH_OPENVX=OFF \
  -D WITH_HALIDE=OFF \
  -D WITH_IPP=OFF \
  -D WITH_TBB=OFF \
  -D CPU_BASELINE=SSE4_2 \
  -D CPU_DISPATCH="AVX;AVX2;FMA3;AVX512_SKX" \
  -D ENABLE_FAST_MATH=ON \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_LIST=core,imgproc


# 3) Build & install
cmake --build . -j$(nproc)
sudo cmake --install .
```



## Build and Run
```
cmake ..
cmake --build . --config Release # build the whole project
cmake --build build --target test_nms # only build the nms binary
ctest -R test_sobel --output-on-failure
```