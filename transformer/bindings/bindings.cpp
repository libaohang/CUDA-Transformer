#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Forward-declare the launcher defined in src/softmax/softmax_naive.cu
void launchSoftmaxNaive(const float* Z, float* out, int rows, int cols, cudaStream_t stream);

torch::Tensor softmax_naive(torch::Tensor input) {
    // --- Validation ---
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [rows, cols]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int rows = input.size(0);
    int cols = input.size(1);

    auto out = torch::empty_like(input);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launchSoftmaxNaive(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        rows, cols,
        stream
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_naive", &softmax_naive, "Naive row-wise softmax (CUDA)");
}