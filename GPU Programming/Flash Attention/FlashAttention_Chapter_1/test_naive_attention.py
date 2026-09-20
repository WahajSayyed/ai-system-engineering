"""Chapter 1 test: our naive attention must agree with PyTorch's own SDPA
(forced onto its plain 'math' backend, which is the textbook algorithm)."""
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel   # PyTorch >= 2.3

from naive_attention import naive_attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa_math_backend(causal):
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 128, 64
    q, k, v = (torch.randn(B, H, N, D, device=DEVICE) for _ in range(3))

    ours = naive_attention(q, k, v, causal=causal)
    with sdpa_kernel(SDPBackend.MATH):
        theirs = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # fp32 on both sides; differences come only from op ordering.
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-4)


def test_rows_of_p_sum_to_one():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 32, 16, device=DEVICE)
    k = torch.randn(1, 1, 32, 16, device=DEVICE)
    p = torch.softmax(q @ k.transpose(-2, -1) / 16 ** 0.5, dim=-1)
    torch.testing.assert_close(p.sum(-1), torch.ones(1, 1, 32, device=DEVICE))
