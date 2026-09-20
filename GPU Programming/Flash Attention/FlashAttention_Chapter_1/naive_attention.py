"""Chapter 1: textbook attention, one PyTorch op at a time.

This file is the *oracle* for the whole series: every CUDA kernel we write later
is checked against reference_attention().
"""
import math

import torch


def naive_attention(q, k, v, causal=False, scale=None):
    """Attention exactly as written in the paper's equations.

    q, k, v : [B, H, N, D]  (batch, heads, sequence length, head dim)
    returns : [B, H, N, D]
    """
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    s = q @ k.transpose(-2, -1)          # S = Q K^T          -> [B, H, N, N]
    s = s * scale                        # S / sqrt(d)
    if causal:
        n_q, n_k = s.shape[-2], s.shape[-1]
        keep = torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).tril()
        s = s.masked_fill(~keep, float("-inf"))   # query i may not see keys j > i
    p = torch.softmax(s, dim=-1)         # P = row-wise softmax -> [B, H, N, N]
    return p @ v                         # O = P V             -> [B, H, N, D]


def reference_attention(q, k, v, causal=False):
    """fp32 oracle. Inputs in fp16/bf16 are upcast first, so a low-precision
    kernel is compared against a more precise answer, not against another
    low-precision computation."""
    return naive_attention(q.float(), k.float(), v.float(), causal=causal)
