#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------- CUDA error-checking ----------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Checks the error state after a kernel launch (launches don't return
// cudaError_t directly, so you need this separately from CUDA_CHECK).
#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel launch error at %s:%d — %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------- GPU timing (cudaEvent-based) ----------

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    // Returns elapsed time in milliseconds.
    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// ---------- Convenience: run + time a kernel launch multiple times ----------

// Usage: float ms = timeKernel(num_iters, [&](){ myKernel<<<grid, block>>>(...); });
template <typename KernelLaunch>
float timeKernel(int num_iters, KernelLaunch launch) {
    GpuTimer timer;
    // Warm-up (avoids counting one-time JIT/caching costs).
    launch();
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.start();
    for (int i = 0; i < num_iters; ++i) {
        launch();
    }
    CUDA_CHECK_LAST();
    float ms = timer.stop();
    return ms / num_iters;
}

// ---------- GFLOPS calculation for GEMM ----------

// GEMM does 2*M*N*K FLOPs (M*N*K multiplies + M*N*K adds).
inline double gemmGFLOPS(int M, int N, int K, float ms) {
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double seconds = ms / 1000.0;
    return (flops / seconds) / 1e9;
}