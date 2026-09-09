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

void launchSoftmaxBackward(const float* grad_out, const float* y, float* grad_in,
                            int rows, int cols, cudaStream_t stream);

torch::Tensor softmax_backward(torch::Tensor grad_output, torch::Tensor y) {
    TORCH_CHECK(grad_output.is_cuda() && y.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(grad_output.sizes() == y.sizes(), "shape mismatch");
    TORCH_CHECK(grad_output.is_contiguous() && y.is_contiguous(), "inputs must be contiguous");

    int rows = y.size(0);
    int cols = y.size(1);
    auto grad_input = torch::empty_like(y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launchSoftmaxBackward(
        grad_output.data_ptr<float>(),
        y.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        rows, cols, stream
    );
    return grad_input;
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

void launchLayernormBackward(const float* grad_out, const float* in, const float* gamma,
                              const float* mean_out, const float* rstd_out,
                              float* grad_in, float* grad_gamma, float* grad_beta,
                              int rows, int cols, cudaStream_t stream);

std::vector<torch::Tensor> layernorm_backward(torch::Tensor grad_output,
                                               torch::Tensor input,
                                               torch::Tensor gamma,
                                               torch::Tensor mean,
                                               torch::Tensor rstd) {
    TORCH_CHECK(grad_output.is_cuda() && input.is_cuda() && gamma.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(mean.is_cuda() && rstd.is_cuda(), "mean/rstd must be CUDA tensors");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "grad_output must be float32");
    TORCH_CHECK(input.sizes() == grad_output.sizes(), "shape mismatch between input and grad_output");
    TORCH_CHECK(grad_output.is_contiguous() && input.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(gamma.size(0) == input.size(1), "gamma size must match input cols");

    int rows = input.size(0);
    int cols = input.size(1);

    auto grad_input = torch::empty_like(input);
    auto grad_gamma = torch::zeros_like(gamma);  // zeros_like: required for atomicAdd
    auto grad_beta  = torch::zeros_like(gamma);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launchLayernormBackward(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        grad_beta.data_ptr<float>(),
        rows, cols, stream
    );

    return {grad_input, grad_gamma, grad_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_naive", &softmax_naive, "Naive row-wise softmax (CUDA)");
    m.def("softmax_backward", &softmax_backward, "Softmax backprop (CUDA)");
    m.def("layernorm_naive", &layernorm_naive, "Naive row-wise layernorm (CUDA)",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);
    m.def("layernorm_backward", &layernorm_backward, "Naive layernorm backward (CUDA)",
      py::arg("grad_output"), py::arg("input"), py::arg("gamma"),
      py::arg("mean"), py::arg("rstd"));
}