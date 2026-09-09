#include "../../include/layernorm.cuh"
#include "../../common/cuda_utils.cuh"
#include <cmath>

__global__ void layernorm_naive_kernel(const float* __restrict__ X,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        float* __restrict__ out,
                                        float* __restrict__ mean_out,
                                        float* __restrict__ rstd_out,
                                        int rows, int cols, float eps){

    extern __shared__ float smem[];
    float* sdata = smem;
    float* qdata = smem + blockDim.x;

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in  = X   + row * cols;
    float*       row_out = out + row * cols;

    float local_sum = 0;
    float local_sum_sq = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x){
        float s = row_in[i];
        local_sum += s;
        local_sum_sq += s * s;
    }

    sdata[threadIdx.x] = local_sum;
    qdata[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            qdata[threadIdx.x] += qdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float row_mean = sdata[0] / cols;
    float row_var = (qdata[0] / cols) - row_mean * row_mean;
    float rstd = rsqrtf(row_var + eps);

    if (threadIdx.x == 0){
        mean_out[row] = row_mean;
        rstd_out[row] = rstd;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_out[i] = (row_in[i] - row_mean) * rstd * gamma[i] + beta[i];
}


void launchLayernormNaive(const float* X, const float* gamma, const float* beta,
                          float* out, float* mean_out, float* rstd_out, int rows,
                          int cols, float eps, cudaStream_t stream) {
    int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);
    size_t smem = 2 * threads * sizeof(float);  // sdata + qdata
    layernorm_naive_kernel<<<grid, block, smem, stream>>>(X, gamma, beta, out, mean_out, 
        rstd_out, rows, cols, eps);
}

__global__ void layernorm_backward_kernel(const float* __restrict__ grad_out,
                                         const float* __restrict__ in,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ mean_out,
                                         const float* __restrict__ rstd_out,
                                         float* __restrict__ grad_in,
                                         float* grad_gamma,
                                         float* grad_beta,
                                         int rows, int cols) {
    extern __shared__ float smem[];
    float* sdata = smem;
    float* qdata = smem + blockDim.x;

    int row = blockIdx.x;
    if (row >= rows) return;

    const float  mean_row = mean_out[row];
    const float  rstd_row = rstd_out[row];
    const float*   dy_row = grad_out + row * cols;
    const float*   in_row = in       + row * cols;
    float*         dx_row = grad_in  + row * cols;

    float local_grad = 0;
    float local_grad_xhat = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x){
        float dL_dyi = dy_row[i];
        float gamma_i = gamma[i];
        float xhat_i = (in_row[i] - mean_row) * rstd_row;

        atomicAdd(&grad_gamma[i], xhat_i * dL_dyi);
        atomicAdd(&grad_beta[i], dL_dyi);

        local_grad += gamma_i * dL_dyi;
        local_grad_xhat += gamma_i * dL_dyi * xhat_i;
    }

    sdata[threadIdx.x] = local_grad;
    qdata[threadIdx.x] = local_grad_xhat;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            qdata[threadIdx.x] += qdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean_grad = sdata[0] / cols;
    float mean_grad_xhat = qdata[0] / cols;

    for (int i = threadIdx.x; i < cols; i += blockDim.x){
        float xhat_i = (in_row[i] - mean_row) * rstd_row;
        dx_row[i] = rstd_row * (gamma[i] * dy_row[i] - mean_grad - xhat_i * mean_grad_xhat);
    }
}

void launchLayernormBackward(const float* grad_out, const float* in, const float* gamma,
                              const float* mean_out, const float* rstd_out,
                              float* grad_in, float* grad_gamma, float* grad_beta,
                              int rows, int cols, cudaStream_t stream) {
    int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);
    size_t smem = 2 * threads * sizeof(float);  // sdata + qdata, same as forward
    layernorm_backward_kernel<<<grid, block, smem, stream>>>(
        grad_out, in, gamma, mean_out, rstd_out,
        grad_in, grad_gamma, grad_beta, rows, cols
    );
    CUDA_CHECK_LAST();
}