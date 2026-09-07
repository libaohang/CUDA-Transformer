#include "../../include/layernorm.cuh"
#include <cmath>

__global__ void layernorm_naive_kernel(const float* __restrict__ X,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        float* __restrict__ out,
                                        float* __restrict__ mean_out,
                                        float* __restrict__ rstd_out,
                                        int rows, int cols, float eps){

    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in  = X   + row * cols;
    float*       row_out = out + row * cols;

    // Pass 1: row mean
    float local_sum = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_sum += row_in[i];

    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + s];
        __syncthreads();
    }
    float row_mean = sdata[0] / cols;
    if (threadIdx.x == 0) mean_out[row] = row_mean;
    __syncthreads();

    // Pass 2: row variance
    local_sum = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float d = row_in[i] - row_mean;
        local_sum += d * d;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float row_var = sdata[0] / cols;
    float rstd = rsqrtf(row_var + eps);
    if (threadIdx.x == 0) rstd_out[row] = rstd;

    // Pass 3: write normalized output
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_out[i] = (row_in[i] - row_mean) * rstd * gamma[i] + beta[i];

}


void launchLayernormNaive(const float* X, const float* gamma, const float* beta,
                          float* out, float* mean_out, float* rstd_out, int rows,
                          int cols, float eps, cudaStream_t stream) {
    int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);
    size_t smem = threads * sizeof(float);
    layernorm_naive_kernel<<<grid, block, smem, stream>>>(X, gamma, beta, out, mean_out, 
        rstd_out, rows, cols, eps);
}