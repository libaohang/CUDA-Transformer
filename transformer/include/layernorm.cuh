#pragma once
#include <cuda_runtime.h>

void launchLayernormNaive(const float* X, const float* gamma, const float* beta,
                           float* out, float* mean_out, float* rstd_out, int rows,
                           int cols, float eps, cudaStream_t stream);
void launchLayernormBackward(const float* grad_out, const float* in, const float* gamma,
                              const float* mean_out, const float* rstd_out,
                              float* grad_in, float* grad_gamma, float* grad_beta,
                              int rows, int cols, cudaStream_t stream);