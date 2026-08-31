#include "../gemm.cuh"
#include <cuda/cmath>

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
    dim3 grid(cuda::ceil_div(N, static_cast<int>(block.x)), cuda::ceil_div(M, static_cast<int>(block.y)));
    gemm_naive_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_K>
__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float A_tile[BLOCK_DIM_Y][TILE_K];
    __shared__ float B_tile[TILE_K][BLOCK_DIM_X];

    int const C_col_idx = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    int const C_row_idx = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    int const num_tiles = (K + TILE_K - 1) / TILE_K;

    float value = 0.0f;

    for (int tileIdx = 0; tileIdx < num_tiles; ++tileIdx) {
        for (int k = threadIdx.x; k < TILE_K; k += BLOCK_DIM_X) {
            int global_k = tileIdx * TILE_K + k;
            A_tile[threadIdx.y][k] =
                (C_row_idx < M && global_k < K) ? A[C_row_idx * K + global_k] : 0.0f;
        }

        for (int k = threadIdx.y; k < TILE_K; k += BLOCK_DIM_Y) {
            int global_k = tileIdx * TILE_K + k;
            B_tile[k][threadIdx.x] =
                (global_k < K && C_col_idx < N) ? B[global_k * N + C_col_idx] : 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_K; ++i) {
            value += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (C_row_idx < M && C_col_idx < N) {
        C[C_row_idx * N + C_col_idx] = value;
    }
}

void launchGemmTiled(const float* d_A, const float* d_B, float* d_C,
                      int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(cuda::ceil_div(N, static_cast<int>(block.x)), cuda::ceil_div(M, static_cast<int>(block.y)));
    gemm_tiled_kernel<16, 16, 32><<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}