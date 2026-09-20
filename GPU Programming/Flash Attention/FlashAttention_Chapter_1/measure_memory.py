"""Chapter 1 exercise: how much extra GPU memory does naive attention need?
Needs a CUDA GPU. Prediction (see the chapter): peak extra ~= 2 x one S matrix."""
import torch

from naive_attention import naive_attention


@torch.no_grad()
def peak_extra_bytes(B, H, N, D, dtype=torch.float16, causal=False):
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=dtype) for _ in range(3))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = naive_attention(q, k, v, causal=causal)
    torch.cuda.synchronize()               # GPU work is asynchronous: wait before reading stats
    peak = torch.cuda.max_memory_allocated()
    del out
    return peak - before


if __name__ == "__main__":
    B, H, D = 1, 8, 64
    dtype = torch.float16
    bytes_per = torch.finfo(dtype).bits // 8
    print(f"GPU: {torch.cuda.get_device_name(0)}   B={B} H={H} D={D} dtype={dtype}")
    for N in (1024, 2048, 4096, 8192):
        one_S = B * H * N * N * bytes_per
        try:
            extra = peak_extra_bytes(B, H, N, D, dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"N={N:5d}  out of memory (one S = {one_S / 2**20:8.1f} MiB)")
            break
        print(f"N={N:5d}  peak extra = {extra / 2**20:9.1f} MiB   one S = {one_S / 2**20:9.1f} MiB   ratio = {extra / one_S:4.2f}")
