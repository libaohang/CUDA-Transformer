#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + \
                ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(err__)); \
        }                                                                        \
    } while (0)

// Catches launch-time errors (bad grid/block config, invalid args) right
// where the launch happens, instead of surfacing at some unrelated later
// CUDA call.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }