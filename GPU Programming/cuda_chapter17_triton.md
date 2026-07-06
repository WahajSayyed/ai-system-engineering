# Chapter 17: OpenAI Triton — Python GPU Kernels

> **Prerequisites:** Chapters 1–16 (CUDA fundamentals, shared memory, warp mechanics, profiling, PyTorch extensions)
> **Language:** Python
> **Domain:** ML / AI, GPU Programming
> **Estimated Time:** 8–10 hours

---

## 17.1 Why Triton Exists

Writing CUDA kernels in C++ is powerful but verbose. For ML workloads, the gap between "what the researcher wants" and "what the GPU programmer implements" is wide. OpenAI's Triton closes that gap: it lets you write GPU kernels in Python, at a level of abstraction just above raw CUDA but far below PyTorch's operator-level API.

### The Abstraction Stack

```
PyTorch (operator level)         ← you call torch.matmul()
──────────────────────────────
Triton (tile level)              ← you write tiled loops over blocks
──────────────────────────────
CUDA (thread level)              ← you manage individual threads
──────────────────────────────
PTX / SASS (instruction level)   ← compiler output
```

Triton's key insight: **ML kernels are naturally tile-decomposable.** Matrix multiplications, attention, softmax — all fit a pattern of loading tiles, computing locally, and writing back. Triton exposes this tile abstraction directly, letting its compiler handle the low-level warp, shared memory, and scheduling decisions.

### What Triton Gives You

| Feature | Triton | CUDA C++ |
|---|---|---|
| Language | Python | C/C++ |
| Thread abstraction | Tile (block of elements) | Individual thread |
| Shared memory mgmt | Automatic | Manual |
| Warp-level decisions | Compiler | Programmer |
| Auto-tuning | Built-in `@triton.autotune` | Manual |
| Interop with PyTorch | Native | Via extensions |

---

## 17.2 Installation & Environment

```bash
# Triton ships with PyTorch >= 2.0 on CUDA
pip install triton

# Verify
python -c "import triton; print(triton.__version__)"

# Optional: for development/bleeding edge
pip install triton-nightly
```

Check GPU compatibility:
```python
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability())  # Triton requires >= (7, 0) — Volta+
```

---

## 17.3 Triton's Core Programming Model

### 17.3.1 The `@triton.jit` Decorator

Every Triton kernel is a Python function decorated with `@triton.jit`. Unlike regular Python, inside a `@triton.jit` function:

- Variables represent **blocks of values** (tensors), not scalars
- Operations are **SIMD** across the block
- The compiler lowers everything to LLVM IR → PTX

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr,          # Pointer to input tensor X
    y_ptr,          # Pointer to input tensor Y
    out_ptr,        # Pointer to output tensor
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Tile size (compile-time constant)
):
    # Each kernel instance handles one tile
    pid = tl.program_id(axis=0)   # Which tile am I?

    # Compute the range of indices this tile covers
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask out-of-bounds accesses (last tile may be partial)
    mask = offsets < n_elements

    # Load tiles from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store result
    tl.store(out_ptr + offsets, output, mask=mask)
```

### 17.3.2 Launching a Triton Kernel

```python
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()

    # Grid = number of tiles needed
    # Lambda receives meta dict containing compile-time constants
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch — similar syntax to CUDA but in Python
    add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=1024,  # Passed as constexpr
    )
    return output

# Test
x = torch.randn(2**20, device='cuda')
y = torch.randn(2**20, device='cuda')
z = add(x, y)
print(torch.allclose(z, x + y))  # True
```

### 17.3.3 `tl.constexpr` — Compile-Time vs Runtime Values

Parameters annotated with `tl.constexpr` are known at **compile time** and baked into the kernel. This lets the compiler unroll loops, choose vector widths, and generate optimal code.

```python
@triton.jit
def example(ptr, N, BLOCK: tl.constexpr):
    # BLOCK is compile-time — tl.arange REQUIRES constexpr
    offsets = tl.arange(0, BLOCK)   # ✓ BLOCK is constexpr
    # N is runtime — cannot use directly in tl.arange
```

---

## 17.4 `tl.load` and `tl.store` with Masking

These are the primary memory operations in Triton. Masking is what makes the last (partial) tile safe.

### Signature

```python
tl.load(pointer, mask=None, other=0.0)
tl.store(pointer, value, mask=None)
```

### Worked Example: Masked Load/Store

```python
@triton.jit
def masked_scale_kernel(
    in_ptr, out_ptr, n, scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n                          # Boolean block-tensor

    x = tl.load(in_ptr + offs, mask=mask, other=0.0)  # 0.0 for OOB
    tl.store(out_ptr + offs, x * scale, mask=mask)     # Ignore OOB writes
```

### 2D Pointer Arithmetic

For matrices (row-major layout):

```python
@triton.jit
def matrix_scale_kernel(
    in_ptr, out_ptr,
    M, N, stride_m, stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column offsets for this tile
    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 2D mask
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)

    # Pointer computation using strides (handles non-contiguous tensors)
    ptrs = in_ptr + row_offs[:, None] * stride_m + col_offs[None, :] * stride_n

    x = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(
        out_ptr + row_offs[:, None] * stride_m + col_offs[None, :] * stride_n,
        x * 2.0,
        mask=mask,
    )
```

---

## 17.5 Auto-Tuning with `@triton.autotune`

Auto-tuning searches over a set of **configs** — different `BLOCK_SIZE`, number of warps, pipeline stages — and picks the fastest one for your GPU.

```python
import triton
import triton.language as tl

# Define search space
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],   # Re-tune when this changes
)
@triton.jit
def autotuned_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

The first call for each unique `key` value runs all configs and caches the winner. Subsequent calls use the cached best config.

---

## 17.6 Project 1: Fused Softmax

Softmax over rows of a matrix. Fusing the three passes (max, sum-exp, normalize) into one kernel eliminates multiple global memory round-trips.

### Naive (unfused) approach (3 passes over data):
```python
# Pass 1: max per row
# Pass 2: exp and sum
# Pass 3: divide
# Each pass reads/writes HBM → very slow for large matrices
```

### Triton fused softmax (1 pass):

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024},num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles ONE row
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Load entire row (with masking for last partial block)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Numerically stable softmax: subtract row max
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # Exponentiate
    numerator = tl.exp(row)

    # Sum for normalization
    denominator = tl.sum(numerator, axis=0)

    # Normalize
    softmax_out = numerator / denominator

    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape

    # Pad BLOCK_SIZE to next power of 2 >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Ensure within limits
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)

    y = torch.empty_like(x)

    fused_softmax_kernel[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# Verification and benchmark
import time

M, N = 4096, 4096
x = torch.randn(M, N, device='cuda', dtype=torch.float32)

ref = torch.softmax(x, dim=1)
out = triton_softmax(x)
print(f"Max error: {(ref - out).abs().max().item():.2e}")  # Should be ~1e-7

# Timing
def benchmark(fn, *args, n_iters=100):
    # Warmup
    for _ in range(10): fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters): fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000  # ms

t_torch  = benchmark(lambda: torch.softmax(x, dim=1))
t_triton = benchmark(lambda: triton_softmax(x))
print(f"PyTorch: {t_torch:.3f} ms | Triton: {t_triton:.3f} ms")
```

---

## 17.7 Project 2: Triton GEMM (Matrix Multiplication)

GEMM is the most important kernel in ML. We implement `C = A @ B` with tiled shared-memory blocking.

### Algorithmic Structure

```
For each tile (pid_m, pid_n) in the output C:
    acc = zeros(BLOCK_M, BLOCK_N)
    For k_tile in range(K // BLOCK_K):
        a_tile = load A[pid_m*BM : (pid_m+1)*BM, k_tile*BK : (k_tile+1)*BK]
        b_tile = load B[k_tile*BK : (k_tile+1)*BK, pid_n*BN : (pid_n+1)*BN]
        acc += dot(a_tile, b_tile)
    store acc → C[pid_m*BM, pid_n*BN]
```

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Tiled GEMM with L2 cache-friendly swizzled tile ordering.
    """
    pid = tl.program_id(0)

    # --- Swizzled tile ordering for L2 cache reuse ---
    # Instead of row-major tile ordering, group tiles in vertical stripes
    # so adjacent tiles in a group share the same A rows → L2 hits.
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Compute pointer blocks ---
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k  = tl.arange(0, BLOCK_K)

    # Pointers to first tile of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # --- Main accumulation loop ---
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Boundary masks for K dimension
        k_mask = offs_k < K - k * BLOCK_K

        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        # Dot product accumulation — Triton generates tensor core ops here
        acc += tl.dot(a, b)

        # Advance pointers to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- Write output tile ---
    c = acc.to(tl.float16)  # Cast to output dtype

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs  = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask  = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible shapes"
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# Test
M, N, K = 512, 512, 512
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

ref = torch.mm(a, b)
out = triton_matmul(a, b)

print(f"Max error: {(ref - out).abs().max().item():.3f}")  # ~0.5 for fp16
print(f"Allclose: {torch.allclose(ref, out, atol=1.0)}")
```

### Understanding `tl.dot`

`tl.dot(a, b)` is Triton's tile-level matrix multiply. On Ampere+ GPUs, Triton automatically generates **tensor core** (WMMA/MMA) instructions when shapes and dtypes permit. This is the key to getting cuBLAS-competitive performance without writing any PTX.

---

## 17.8 Advanced: Fused Attention Kernel (Flash Attention Style)

Flash Attention avoids materializing the full attention score matrix by computing softmax in tiles. Here's a simplified Triton implementation of the forward pass:

```python
@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    seqlen_q, seqlen_k, head_dim,
    scale,                        # 1 / sqrt(head_dim)
    BLOCK_M: tl.constexpr,        # Tile size along Q sequence
    BLOCK_N: tl.constexpr,        # Tile size along K/V sequence
    HEAD_DIM: tl.constexpr,
):
    # Each program processes BLOCK_M query tokens
    start_m = tl.program_id(0)
    offs_m  = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, HEAD_DIM)

    # Load Q tile
    q = tl.load(
        Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=(offs_m[:, None] < seqlen_q),
        other=0.0,
    )

    # Running statistics for online softmax
    m_i  = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i  = tl.zeros((BLOCK_M,),              dtype=tl.float32)
    acc  = tl.zeros((BLOCK_M, HEAD_DIM),     dtype=tl.float32)

    # Iterate over K/V tiles
    for start_n in range(0, seqlen_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K and V tiles
        k = tl.load(
            K_ptr + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kk,
            mask=(offs_n[:, None] < seqlen_k),
            other=0.0,
        )
        v = tl.load(
            V_ptr + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vk,
            mask=(offs_n[:, None] < seqlen_k),
            other=0.0,
        )

        # Attention scores: QK^T * scale
        qk = tl.dot(q, tl.trans(k)) * scale   # (BLOCK_M, BLOCK_N)

        # Causal mask (optional)
        # qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))

        # Online softmax update (Flash Attention algorithm)
        m_ij = tl.max(qk, axis=1)             # New row max
        m_new = tl.maximum(m_i, m_ij)         # Running max

        alpha = tl.exp(m_i - m_new)           # Correction factor
        p    = tl.exp(qk - m_new[:, None])    # Softmax numerator

        l_i  = alpha * l_i + tl.sum(p, axis=1)
        acc  = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v)
        m_i  = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < seqlen_q),
    )
```

This demonstrates the **online softmax** trick: by tracking running max `m_i` and sum `l_i`, we can compute softmax over the full sequence without storing the full `(seqlen_q, seqlen_k)` attention matrix.

---

## 17.9 Inspecting Triton's Generated Code

Triton provides tools to see what it compiles to:

```python
# After calling a kernel, inspect the compiled PTX
import os
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

# Run kernel once to trigger compilation
out = triton_softmax(x)

# Get the compiled kernel object
compiled = fused_softmax_kernel.cache[next(iter(fused_softmax_kernel.cache))]

# Print PTX
print(compiled.asm['ptx'][:3000])   # First 3000 chars

# Print TTIR (Triton IR)
print(compiled.asm['ttir'])

# Print LLVMIR
print(compiled.asm['llvmir'][:2000])
```

You can also set environment variables to dump compilation artifacts:
```bash
TRITON_PRINT_AUTOTUNING=1 python your_script.py    # See autotune results
MLIR_ENABLE_DUMP=1 python your_script.py           # Dump MLIR passes
```

---

## 17.10 Profiling Triton Kernels

Triton kernels appear in Nsight Systems/Compute under their kernel names:

```python
# Profile with PyTorch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(10):
        out = triton_softmax(x)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

For Nsight Compute:
```bash
ncu --set full -o triton_profile python your_script.py
```

---

## 17.11 Triton vs. Torch.compile + CUDA Graphs

| Approach | When to use |
|---|---|
| `torch.compile` | Fast iteration, standard ops, no custom math |
| Triton kernel | Custom fused ops, novel algorithms (e.g., new attention variants) |
| Raw CUDA C++ | Maximum control, systems-level work (Chapter 18–19) |
| Triton + `torch.compile` | Best of both: compile discovers your Triton ops |

You can register Triton kernels as custom ops that `torch.compile` can optimize around:

```python
@torch.library.custom_op("mylib::softmax", mutates_args=())
def my_softmax(x: torch.Tensor) -> torch.Tensor:
    return triton_softmax(x)

@my_softmax.register_fake
def _(x):
    return torch.empty_like(x)
```

---

## 17.12 Common Pitfalls & Debugging

### Pitfall 1: Non-power-of-2 BLOCK_SIZE
```python
# ❌ Will crash — tl.arange requires power-of-2 at compile time
BLOCK_SIZE = 100

# ✓ Always use powers of 2
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

### Pitfall 2: Forgetting masks on the last tile
```python
# ❌ Silent OOB read — may return garbage or segfault
x = tl.load(ptr + offsets)

# ✓ Always mask
x = tl.load(ptr + offsets, mask=offsets < n, other=0.0)
```

### Pitfall 3: Float16 accumulation overflow
```python
# ❌ Accumulating in float16 loses precision in GEMM
acc = tl.zeros((BM, BN), dtype=tl.float16)

# ✓ Always accumulate in float32, cast at store
acc = tl.zeros((BM, BN), dtype=tl.float32)
...
tl.store(ptr, acc.to(tl.float16), ...)
```

### Pitfall 4: Strides vs contiguous assumption
```python
# ❌ Assumes row-major contiguous
ptr + row * N + col

# ✓ Use actual tensor strides (handles transposed, sliced tensors)
ptr + row * stride_row + col * stride_col
```

### Pitfall 5: `num_stages` and software pipelining
`num_stages` in `triton.Config` controls the software pipeline depth — how many tiles are prefetched concurrently. Higher stages hide memory latency but use more registers. If register pressure exceeds the budget, CUDA spills to local memory and performance collapses. If in doubt, start with `num_stages=2` or `3`.

---

## 17.13 Performance Mental Model

When writing Triton kernels, think in terms of the roofline model (Chapter 12):

```
Arithmetic Intensity = FLOPs / Bytes_accessed

Softmax (M×N matrix):
  FLOPs ≈ 5×M×N   (max, subtract, exp, sum, div)
  Bytes ≈ 2×M×N×2  (read fp16 + write fp16)
  AI    ≈ 1.25 FLOPs/Byte → memory-bound

GEMM (M×N×K):
  FLOPs = 2×M×N×K
  Bytes = 2×(MK + KN + MN)×2
  AI (large) >> 100 FLOPs/Byte → compute-bound
```

For memory-bound kernels (softmax, layer norm), the win comes from **fusion** — doing more work per byte loaded. For compute-bound kernels (GEMM), the win comes from **tensor core utilization** — ensuring tiles are sized and typed to trigger MMA instructions.

---

## 17.14 Chapter Summary

| Concept | Key Point |
|---|---|
| `@triton.jit` | Compiles Python to PTX; variables are SIMD blocks |
| `tl.program_id` | Identifies which tile this kernel instance handles |
| `tl.load`/`tl.store` | Global memory access; always mask on last tile |
| `tl.constexpr` | Compile-time constants; required for `tl.arange` |
| `@triton.autotune` | Benchmark multiple configs; caches the winner |
| `tl.dot` | Tile-level matmul; triggers tensor cores on Ampere+ |
| Online softmax | Track running `m_i`, `l_i` to avoid materializing full attention |
| L2 swizzling | Group tiles to maximize L2 hit rate in GEMM |
| `num_stages` | Software pipeline depth; balance latency hiding vs register pressure |

---

## 17.15 Exercises

**Exercise 1 — Vector Operations**
Implement a fused `ReLU + scale` kernel in Triton. Benchmark it against the PyTorch equivalent. What's the speedup? Why?

**Exercise 2 — Layer Normalization**
Implement fused layer normalization (mean, variance, normalize, scale, shift in one kernel). Verify against `torch.nn.LayerNorm`. Hint: use `tl.sum` and `tl.sqrt`.

**Exercise 3 — Transposed GEMM**
Modify the GEMM kernel to support `C = A^T @ B`. What changes in the pointer arithmetic and strides?

**Exercise 4 — Auto-tune Exploration**
Add configs with `num_stages=4` and `num_stages=5` to the softmax kernel's autotune. On your GPU, which `num_stages` wins? Profile with `ncu` to understand why.

**Exercise 5 — Causal Flash Attention**
Add a causal mask to the flash attention kernel (lower-triangular: query position ≥ key position). Verify the output matches `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True`.

---

## 17.16 What's Next

Chapter 18 dives below Triton into the CUDA compiler pipeline itself: reading PTX assembly, writing inline PTX with `asm()`, using `cuobjdump` to inspect SASS (GPU machine code), and understanding register allocation — giving you the full picture from Python all the way down to silicon.

---

*Chapter 17 of 20 — CUDA GPU Programming: Beginner to Expert*
