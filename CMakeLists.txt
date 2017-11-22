cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuda_practice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR})

# OPENCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(argument_parser src/argument_parser.cpp)

add_executable(main src/main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main ${OpenCV_LIBS} argument_parser)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
