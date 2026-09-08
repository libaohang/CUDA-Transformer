import torch
import transformer_kernels_cuda as tk

def test_softmax_backward():
    rows, cols = 128, 4096
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, requires_grad=True)

    # Reference: let PyTorch compute forward + backward via autograd
    y_ref = torch.softmax(x, dim=-1)
    grad_output = torch.randn_like(y_ref)
    y_ref.backward(grad_output)
    grad_ref = x.grad.clone()

    # Your kernel: forward (to get y), then your backward kernel
    with torch.no_grad():
        y = tk.softmax_naive(x)
        grad_input = tk.softmax_backward(grad_output, y)

    torch.testing.assert_close(grad_input, grad_ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_softmax_backward()
    print("softmax_backward test passed!")