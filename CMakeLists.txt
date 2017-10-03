cmake_minimum_required(VERSION 3.5)
project(cuda_practice)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDA_NVCC_FLAGS -std=c++11)

include_directories(${PROJECT_SOURCE_DIR})

cuda_add_executable(test_convolution
  test/test_convolution.cu)
target_link_libraries(test_convolution ${OpenCV_LIBS})

cuda_add_executable(test_cuda_sort test/test_cuda_sort.cu)

cuda_add_executable(test_cuda_hog test/test_cuda_hog.cu)
