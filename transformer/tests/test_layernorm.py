import torch
import transformer_kernels_cuda as tk

def test_layernorm_naive():
    rows, cols = 128, 4096
    eps = 1e-5

    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
    beta = torch.randn(cols, device="cuda", dtype=torch.float32)

    out, mean_out, rstd_out = tk.layernorm_naive(x, gamma, beta, eps)

    ref = torch.nn.functional.layer_norm(x, (cols,), weight=gamma, bias=beta, eps=eps)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    # optional: sanity check mean/rstd against a manual computation
    ref_mean = x.mean(dim=-1)
    ref_var = x.var(dim=-1, unbiased=False)
    ref_rstd = (ref_var + eps).rsqrt()

    torch.testing.assert_close(mean_out, ref_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rstd_out, ref_rstd, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_layernorm_naive()
    print("layernorm_naive test passed!")