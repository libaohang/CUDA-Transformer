#pragma once
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
#include <string>

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t status__ = (expr);                                      \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                               \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + \
                ":" + std::to_string(__LINE__) + " code=" + std::to_string(status__)); \
        }                                                                      \
    } while (0)

inline cublasHandle_t getCublasHandle() {
    // Reuses PyTorch's handle so it stays on the current stream —
    // don't cublasCreate() your own.
    return at::cuda::getCurrentCUDABlasHandle();
}