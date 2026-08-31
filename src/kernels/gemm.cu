#include "../gemm.cuh"
#include <cuda/std/type_traits>

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void launchGemmNaive(const float* d_A, const float* d_B, float* d_C,
                      int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(cuda::ceil_div(N, block.x), cuda::ceil_div(M, block.y));
    gemm_naive_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K, int tile_K) {
    __shared__ float A_tile[blockDim.y][tile_K];
    __shared__ float B_tile[tile_K][blockDim.x];

    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
    
    float value = 0.0;
    
    for
    for (int k = 0; k < tile_K; k += blockDim.x){
        if (tile_K * tileIdx + k < K){
            A_tile[threadIdx.y][k] = A[C_row_idx * K + tile_K * tileIdx + k];
        }
    }

    for (int k = 0; k < tile_K; k += blockDim.y){
        if (tile_K * tileIdx + k < K){
            B_tile[k][threadIdx.x] = B[(tile_K * tileIdx + k) * N + C_col_idx];
        }
    }

    __syncthreads();


    for (int i = 0; i < K; ++i){
        value += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x]
    }
}