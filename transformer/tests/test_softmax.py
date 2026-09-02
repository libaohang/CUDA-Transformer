import torch
import transformer_kernels_cuda as tk

def test_softmax_naive():
    x = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    out = tk.softmax_naive(x)
    ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)