#include <cublas_v2.h>
#include <math_constants.h>

#include "../../include/mlp.cuh"
#include "../../common/cuda_utils.cuh"
#include "../../common/cublas_context.cuh"


void cublasGemmRowMajor(cublasHandle_t handle,
                         const float* d_A, const float* d_B, float* d_C,
                         int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N));
}

__global__ void activation_kernel(const float* __restrict__ b1,
                                float* __restrict__ pre,
                                float* __restrict__ post,
                                int rows, int d_ff){
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_pre = pre + row * d_ff;
    float* row_post = post + row * d_ff;

    float a = sqrtf(2.0f / CUDART_PI_F);

    for(int i = threadIdx.x; i < d_ff; i += blockDim.x){
        float x = row_pre[i] + b1[i];
        row_pre[i] = x;
        row_post[i] = 0.5f * x * (1.0f + tanhf(a * (x + 0.044715f * x * x * x)));
    }
}

void launchActivation(const float* __restrict__ b1,
                        float* __restrict__ pre,
                        float* __restrict__ post,
                        int rows, int d_ff, cudaStream_t stream) {
    int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);
    activation_kernel<<<grid, block, 0, stream>>>(b1, pre, post, rows, d_ff);
    CUDA_CHECK_LAST();
}

__global__ void add_kernel(const float* __restrict__ b2,
                                float* __restrict__ out,
                                int rows, int d_model){
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_out = out + row * d_model;

    for(int i = threadIdx.x; i < d_model; i += blockDim.x){
        row_out[i] += b2[i];
    }
}

void launchAdd(const float* __restrict__ b2,
                float* __restrict__ out,
                int rows, int d_model, 
                cudaStream_t stream) {
    int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);
    add_kernel<<<grid, block, 0, stream>>>(b2, out, rows, d_model);
    CUDA_CHECK_LAST();
}

void launchMLP(const float* X, const float* W1, const float* W2,
                       const float* b1, const float* b2,
                       float* out, float* pre, float* post,
                       int rows, int d_ff, int d_model,
                       cudaStream_t stream) {
    cublasHandle_t handle = getCublasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    cublasGemmRowMajor(handle, X, W1, pre, rows, d_ff, d_model);
    launchActivation(b1, pre, post, rows, d_ff, stream);
    cublasGemmRowMajor(handle, post, W2, out, rows, d_model, d_ff);
    launchAdd(b2, out, rows, d_model, stream);
}