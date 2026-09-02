#include "../../include/softmax.cuh"
#include <cfloat>

// One block per row. blockDim.x threads cooperate on that row.
__global__ void softmax_naive_kernel(const float* __restrict__ Z,
                                      float* __restrict__ out,
                                      int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in  = Z   + row * cols;
    float*       row_out = out + row * cols;

    // Pass 1: row max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, row_in[i]);

    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Pass 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_sum += expf(row_in[i] - row_max);

    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // Pass 3: write normalized output
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_out[i] = expf(row_in[i] - row_max) / row_sum;
}

void launchSoftmaxNaive(const float* Z, float* out, int rows, int cols, cudaStream_t stream) {
    int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);
    size_t smem = threads * sizeof(float);
    softmax_naive_kernel<<<grid, block, smem, stream>>>(Z, out, rows, cols);
}