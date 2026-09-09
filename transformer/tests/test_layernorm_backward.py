import torch
import transformer_kernels_cuda as tk

def test_layernorm_backward():
    rows, cols = 128, 4096
    eps = 1e-5

    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, requires_grad=True)
    gamma = torch.randn(cols, device="cuda", dtype=torch.float32, requires_grad=True)
    beta = torch.randn(cols, device="cuda", dtype=torch.float32, requires_grad=True)

    # Reference via PyTorch autograd
    y_ref = torch.nn.functional.layer_norm(x, (cols,), weight=gamma, bias=beta, eps=eps)
    grad_output = torch.randn_like(y_ref)
    y_ref.backward(grad_output)

    grad_x_ref = x.grad.clone()
    grad_gamma_ref = gamma.grad.clone()
    grad_beta_ref = beta.grad.clone()

    # Your kernels: forward (to get mean/rstd), then backward
    with torch.no_grad():
        out, mean, rstd = tk.layernorm_naive(x, gamma, beta, eps)
        grad_x, grad_gamma, grad_beta = tk.layernorm_backward(grad_output, x, gamma, mean, rstd)

    torch.testing.assert_close(grad_x, grad_x_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(grad_gamma, grad_gamma_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(grad_beta, grad_beta_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_layernorm_backward()
    print("layernorm_backward test passed!")