cmake_minimum_required(VERSION 3.5)
project(cuda_practice)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDA_NVCC_FLAGS -std=c++11)

include_directories(${PROJECT_SOURCE_DIR})

cuda_add_executable(test_convolution
  test/test_convolution.cu)
target_link_libraries(test_convolution ${OpenCV_LIBS})

cuda_add_executable(live_camera_filter src/live_camera_filter.cu)
target_link_libraries(live_camera_filter ${OpenCV_LIBS})

cuda_add_executable(test_tensor 
  src/tensor.cu
  test/test_tensor.cu)

cuda_add_executable(test_cuda_sort test/test_cuda_sort.cu)
