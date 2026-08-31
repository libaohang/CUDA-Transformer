cmake_minimum_required(VERSION 3.18)
project(custom-gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(${CMAKE_SOURCE_DIR}/custom-gemm/src ${CMAKE_SOURCE_DIR}/custom-gemm/common)

add_library(gemm_lib src/kernels/gemm.cu)

add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm gemm_lib cublas)