#include <cstdio>
#include <vector>
#include <cublas_v2.h>

#include "../common/utils.cuh"
#include "../common/matrix.cuh"
#include "../src/gemm.cuh"

// Reference GEMM via cuBLAS. Handles the row-major/col-major layout
// mismatch by swapping operand order.
void cublasGemmRowMajor(cublasHandle_t handle,
                         const float* d_A, const float* d_B, float* d_C,
                         int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    printf("GEMM: M=%d, N=%d, K=%d\n\n", M, N, K);

    // ---------- Host-side setup ----------
    std::vector<float> h_A, h_B, h_C_naive, h_C_cublas;
    matrixRandomInit(h_A, M, K);
    matrixRandomInit(h_B, K, N, -1.0f, 1.0f, /*seed=*/123);
    matrixZeroInit(h_C_naive, M, N);
    matrixZeroInit(h_C_cublas, M, N);

    // ---------- Device allocation ----------
    float *d_A, *d_B, *d_C_naive, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&d_C_naive, sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, sizeof(float) * M * N));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));

    // ---------- cuBLAS setup ----------
    cublasHandle_t handle;
    cublasCreate(&handle);

    // ---------- Correctness check ----------
    launchGemmNaive(d_A, d_B, d_C_naive, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasGemmRowMajor(handle, d_A, d_B, d_C_cublas, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C_naive, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C_cublas, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    auto cmp = matrixCompare(h_C_naive, h_C_cublas, M, N);
    matrixCompareReport(cmp, "Naive vs cuBLAS");

    if (M <= 8 && N <= 8) {  // only dump small matrices
        matrixPrint(h_C_naive, M, N, "C_naive");
        matrixPrint(h_C_cublas, M, N, "C_cublas");
    }

    // ---------- Performance comparison ----------
    const int num_iters = 20;

    float ms_naive = timeKernel(num_iters, [&]() {
        launchGemmNaive(d_A, d_B, d_C_naive, M, N, K);
    });

    float ms_cublas = timeKernel(num_iters, [&]() {
        cublasGemmRowMajor(handle, d_A, d_B, d_C_cublas, M, N, K);
    });

    double gflops_naive = gemmGFLOPS(M, N, K, ms_naive);
    double gflops_cublas = gemmGFLOPS(M, N, K, ms_cublas);

    printf("\n--- Performance ---\n");
    printf("Naive:  %8.4f ms  |  %8.2f GFLOPS\n", ms_naive, gflops_naive);
    printf("cuBLAS: %8.4f ms  |  %8.2f GFLOPS\n", ms_cublas, gflops_cublas);
    printf("Naive achieves %.2f%% of cuBLAS performance\n",
           100.0 * gflops_naive / gflops_cublas);

    // ---------- Cleanup ----------
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_cublas);

    return cmp.passed ? 0 : 1;
}