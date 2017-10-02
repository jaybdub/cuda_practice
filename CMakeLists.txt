cmake_minimum_required(VERSION 3.5)
project(cuda_practice)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

cuda_add_executable(test_convolution
  test/test_convolution.cu)
