#pragma once
#include <cuda_runtime.h>

void launchSoftmaxNaive(const float* Z, float* out, int rows, int cols, cudaStream_t stream);
void launchSoftmaxBackward(const float* grad_out, const float* y, float* grad_in,
                            int rows, int cols, cudaStream_t stream);