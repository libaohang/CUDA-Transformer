#pragma once

#include <cuda_runtime.h>

// Launches the naive GEMM kernel: C = A @ B
// A: M x K, B: K x N, C: M x N (all row-major, device pointers)
void launchGemmNaive(const float* d_A, const float* d_B, float* d_C,
                      int M, int N, int K, cudaStream_t stream = 0);