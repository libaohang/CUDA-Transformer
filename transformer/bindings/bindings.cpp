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

void launchLayernormNaive(const float* X, const float* gamma, const float* beta,
                           float* out, float* mean_out, float* rstd_out, int rows,
                           int cols, float eps, cudaStream_t stream);

std::vector<torch::Tensor> layernorm_naive(torch::Tensor input,
                                            torch::Tensor gamma,
                                            torch::Tensor beta,
                                            double eps) {
    // --- Validation ---
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA tensors");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [rows, cols]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");
    TORCH_CHECK(gamma.size(0) == input.size(1), "gamma size must match input cols");
    TORCH_CHECK(beta.size(0) == input.size(1), "beta size must match input cols");

    int rows = input.size(0);
    int cols = input.size(1);

    auto out      = torch::empty_like(input);
    auto mean_out = torch::empty({rows}, input.options());
    auto rstd_out = torch::empty({rows}, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launchLayernormNaive(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        mean_out.data_ptr<float>(),
        rstd_out.data_ptr<float>(),
        rows, cols,
        static_cast<float>(eps),
        stream
    );

    return {out, mean_out, rstd_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_naive", &softmax_naive, "Naive row-wise softmax (CUDA)");
    m.def("layernorm_naive", &layernorm_naive, "Naive row-wise layernorm (CUDA)",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);
}