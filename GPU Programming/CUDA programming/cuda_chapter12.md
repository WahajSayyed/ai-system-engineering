# Chapter 12 — cuBLAS, cuDNN & CUDA Libraries

## Learning objectives

By the end of this chapter you will:

- Know the full CUDA library ecosystem and when each library applies
- Call cuBLAS GEMM directly from Python for all three BLAS levels
- Set up cuDNN convolution descriptors and understand its algorithm search
- Understand why library kernels outperform hand-written ones by 5–20×
- Know when to use libraries vs write custom kernels
- Profile library calls to understand what is happening under the hood

---

## 12.1 The CUDA library ecosystem

NVIDIA ships highly optimised libraries for every major compute domain.
Before writing a custom kernel, always check if a library covers the operation.

```
Library         Domain                      Python access
──────────────────────────────────────────────────────────────────────
cuBLAS          Dense linear algebra        CuPy (cp.linalg, cp.dot)
                (GEMM, BLAS L1/L2/L3)       cupy.cublas directly

cuDNN           Deep learning primitives    PyTorch/TensorFlow (auto)
                (conv, BN, attention)       cupy.cudnn

cuFFT           Fast Fourier transforms     cp.fft.*

cuSPARSE        Sparse linear algebra       scipy.sparse on GPU (cupy)

cuRAND          Random number generation    cp.random.*

cuSOLVER        Dense/sparse solvers        cp.linalg.solve, svd, etc.

Thrust          Parallel algorithms         Wrapped by CuPy
                (sort, scan, reduce)

NCCL            Multi-GPU collective ops    torch.distributed (Ch 15)

TensorRT        Inference optimisation      tensorrt Python package

cutlass         Template GEMM/conv          cutlass Python (Ch 14 preview)
```

**The 80/20 rule:** cuBLAS and cuDNN account for ~80% of compute time in
most deep learning workloads. Master these two and you understand where
GPU time goes in PyTorch.

---

## 12.2 BLAS primer — the three levels

BLAS (Basic Linear Algebra Subprograms) is organised into three levels by
the ratio of FLOPs to memory accesses (arithmetic intensity):

```
Level 1 — Vector operations        O(N) FLOPs,  O(N) memory   AI ≈ 1
  dot(x, y)        → scalar
  axpy(a, x, y)    → y = a*x + y
  nrm2(x)          → ||x||₂
  scal(a, x)       → x = a*x

Level 2 — Matrix-vector operations O(N²) FLOPs, O(N²) memory  AI ≈ 1
  gemv(A, x)       → y = αAx + βy
  ger(x, y)        → A = αxy^T + A  (outer product)

Level 3 — Matrix-matrix operations O(N³) FLOPs, O(N²) memory  AI ≈ N
  gemm(A, B)       → C = αAB + βC   ← the most important operation in ML
  syrk, trsm, ...

Key insight: only Level 3 has arithmetic intensity that grows with N.
GEMM with N=1024 has AI ≈ 170 FLOP/byte → fully compute-bound → GPU excels.
Level 1/2 operations are always memory-bound regardless of N.
```

---

## 12.3 cuBLAS GEMM — the most important GPU operation

GEMM computes: `C = α × A × B + β × C`

The `α` and `β` scalars make GEMM a fused multiply-add — you can accumulate
into an existing matrix C without a separate kernel launch.

### Via CuPy (simplest)

```python
import cupy as cp
import numpy as np

M, K, N = 2048, 2048, 2048

A = cp.random.randn(M, K, dtype=cp.float32)
B = cp.random.randn(K, N, dtype=cp.float32)

# CuPy dispatches to cuBLAS GEMM automatically
C = A @ B               # or cp.dot(A, B) or cp.matmul(A, B)

# With alpha/beta (fused: C = 2*A@B + 0.5*C)
C_out = cp.zeros((M, N), dtype=cp.float32)
cp.cublas.sgemm('N', 'N', N, M, K,
                2.0,              # alpha
                B, N,             # B and its leading dimension
                A, K,             # A and its leading dimension
                0.5,              # beta
                C_out, N)         # C and its leading dimension

# Benchmarking
def benchmark(fn, n_warmup=5, n_rep=20):
    for _ in range(n_warmup): fn()
    cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(n_rep):
        s = cp.cuda.Event(); e = cp.cuda.Event()
        s.record(); fn(); e.record(); e.synchronize()
        times.append(cp.cuda.get_elapsed_time(s, e))
    return np.min(times)

t = benchmark(lambda: A @ B)
flops   = 2 * M * K * N
gflops  = flops / (t * 1e-3) / 1e9
print(f"cuBLAS GEMM {M}×{K}×{N}: {t:.2f} ms  {gflops:.0f} GFLOP/s")

# Compare to your tiled kernel from Chapter 5
# Expected gap: cuBLAS ≈ 5–20× faster (Tensor Cores, larger tiles, tuning)
```

### Understanding why cuBLAS is faster

```
Your Chapter 5 tiled kernel (TILE=16):
  - Tile size: 16×16 = 256 floats per shared mem load
  - Arithmetic intensity per tile: 2×16 / (2×16²×4/16²) = 16 FLOP/byte
  - Peak achieved: ~180 GFLOP/s on T4

cuBLAS internally uses:
  - TILE=128 or 256 (much larger tiles → higher AI)
  - Register tiling (further reduces shared mem pressure)
  - Tensor Cores for FP16/BF16/TF32 (Ch 14): 8× hardware throughput
  - Double buffering: prefetch next tile while computing current
  - Persistent kernels: blocks stay alive across multiple tiles
  - Architecture-specific tuning: different kernel for sm_75 vs sm_80 vs sm_90
  - Auto-selection: tests multiple algorithms at first call, caches the best
```

### cuBLAS batched GEMM

For transformer attention: compute Q@K^T for each head in a batch:

```python
import cupy as cp

# Batched GEMM: compute C[b] = A[b] @ B[b] for each batch b
batch = 32    # batch size × n_heads combined
M, K, N = 128, 64, 128   # seq_len × d_head × seq_len (attention scores)

A_batch = cp.random.randn(batch, M, K, dtype=cp.float32)
B_batch = cp.random.randn(batch, K, N, dtype=cp.float32)

# CuPy handles batched matmul via broadcasting
C_batch = A_batch @ B_batch   # shape: (batch, M, N)
# Internally dispatches to cublasGemmStridedBatchedEx

print(f"Batched GEMM output shape: {C_batch.shape}")

# For mixed precision (FP16 compute, FP32 accumulate):
A_fp16 = A_batch.astype(cp.float16)
B_fp16 = B_batch.astype(cp.float16)
C_fp32 = (A_fp16 @ B_fp16).astype(cp.float32)  # auto mixed precision
```

---

## 12.4 BLAS Level 1 and 2 from Python

```python
import cupy as cp

# ── Level 1: Vector operations ────────────────────────────────────────────────
x = cp.random.randn(1_000_000, dtype=cp.float32)
y = cp.random.randn(1_000_000, dtype=cp.float32)

dot_result = cp.dot(x, y)              # cuBLAS sdot
norm       = cp.linalg.norm(x)         # cuBLAS snrm2
cp.multiply(2.0, x, out=x)            # cuBLAS sscal (scale)
cp.add(cp.multiply(2.0, x), y, out=y) # cuBLAS saxpy (y = ax + y)

# ── Level 2: Matrix-vector ────────────────────────────────────────────────────
A = cp.random.randn(1000, 1000, dtype=cp.float32)
v = cp.random.randn(1000,       dtype=cp.float32)

# y = A @ v — cuBLAS sgemv
result = A @ v                         # shape: (1000,)

# Outer product: A += x * y^T — cuBLAS sger
outer = cp.outer(x[:10], y[:10])       # small example

# ── When Level 1/2 are bottlenecks ───────────────────────────────────────────
# These are always memory-bound — the GPU is not the right tool if
# the bottleneck is a dot product or gemv. Consider:
# 1. Fusing multiple Level 1 ops into one custom kernel (Ch 4 ElementwiseKernel)
# 2. Batching many small Level 2 ops into one batched Level 3 op
# 3. Restructuring the algorithm to use GEMM instead of GEMV
```

---

## 12.5 cuFFT via CuPy

```python
import cupy as cp
import numpy as np

# ── 1D FFT ────────────────────────────────────────────────────────────────────
N    = 2**20   # 1M points — power of 2 for maximum efficiency
sig  = cp.random.randn(N, dtype=cp.float32)

# Forward FFT (real input → complex output)
freq = cp.fft.rfft(sig)                # shape: (N//2 + 1,) complex64
print(f"FFT output shape: {freq.shape}, dtype: {freq.dtype}")

# Inverse FFT
sig_back = cp.fft.irfft(freq, n=N)    # restore original length
print(f"Round-trip error: {float(cp.max(cp.abs(sig_back - sig))):.2e}")

# ── 2D FFT (image processing) ─────────────────────────────────────────────────
H, W   = 1024, 1024
image  = cp.random.randn(H, W, dtype=cp.float32)
spec2d = cp.fft.rfft2(image)           # 2D FFT
filt   = cp.ones_like(spec2d)
filt[:, W//4:] = 0                     # low-pass filter: zero high frequencies
filtered = cp.fft.irfft2(filt * spec2d, s=(H, W))

# ── Batched FFT (many signals at once) ────────────────────────────────────────
batch   = 256
sig_batch = cp.random.randn(batch, N, dtype=cp.float32)

# FFT along last axis for all batch elements simultaneously
freq_batch = cp.fft.rfft(sig_batch, axis=-1)   # shape: (256, N//2+1)

# Benchmark: single vs batched
t_single = 0
for _ in range(batch):
    s = cp.cuda.Event(); e = cp.cuda.Event()
    s.record()
    cp.fft.rfft(sig_batch[0])
    e.record(); e.synchronize()
    t_single += cp.cuda.get_elapsed_time(s, e)

s = cp.cuda.Event(); e = cp.cuda.Event()
s.record()
cp.fft.rfft(sig_batch, axis=-1)
e.record(); e.synchronize()
t_batch = cp.cuda.get_elapsed_time(s, e)

print(f"256 sequential FFTs: {t_single:.1f} ms")
print(f"1 batched FFT:       {t_batch:.1f} ms")
print(f"Batching speedup:    {t_single/t_batch:.1f}x")
```

---

## 12.6 cuDNN for deep learning primitives

cuDNN handles the core neural network operations. PyTorch and TensorFlow
call it automatically — but understanding the API demystifies what happens
inside `torch.nn.Conv2d` or `torch.nn.LayerNorm`.

### Convolution via CuPy's cuDNN wrapper

```python
import cupy as cp
from cupy import cudnn

# Input:  (N, C_in,  H,   W)   — NCHW format (cuDNN default)
# Filter: (C_out, C_in, kH, kW)
# Output: (N, C_out, H', W')

N, C_in, H, W = 32, 64, 56, 56      # batch, channels, height, width
C_out, kH, kW = 128, 3, 3           # output channels, kernel size

x      = cp.random.randn(N, C_in,  H,  W,  dtype=cp.float32)
weight = cp.random.randn(C_out, C_in, kH, kW, dtype=cp.float32)
bias   = cp.zeros(C_out, dtype=cp.float32)

# cuDNN convolution via CuPy
# CuPy wraps cuDNN under the hood for conv operations
# Access via cudnn module for explicit control

# Simpler: use CuPy's signal.convolve or cupyx.scipy.ndimage
from cupyx.scipy.ndimage import convolve
# For full DL convolutions, PyTorch's cuDNN integration is the standard path

# ── PyTorch + cuDNN (the real production path) ────────────────────────────────
try:
    import torch
    import torch.nn as nn

    # PyTorch Conv2d calls cuDNN automatically
    conv = nn.Conv2d(C_in, C_out, kernel_size=kH, padding=1).cuda()
    x_t  = torch.randn(N, C_in, H, W, device='cuda')

    # Enable cuDNN benchmark mode: auto-selects fastest algorithm
    torch.backends.cudnn.benchmark = True

    # Warm up (triggers algorithm search and caching)
    for _ in range(3):
        out = conv(x_t)
    torch.cuda.synchronize()

    # Benchmark
    import time
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        out = conv(x_t)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / 100

    flops_conv = 2 * N * C_out * C_in * kH * kW * H * W
    print(f"Conv2d ({N}×{C_in}×{H}×{W} → {C_out} ch): {t:.2f} ms")
    print(f"GFLOP/s: {flops_conv/t/1e9:.0f}")

except ImportError:
    print("PyTorch not available — install with: pip install torch")
```

### Understanding cuDNN algorithm selection

cuDNN does not use one algorithm for all convolutions. It maintains a
catalogue of algorithms and selects the best for each (input size, filter
size, GPU architecture):

```
cuDNN convolution algorithms:
  IMPLICIT_GEMM         Lowering: im2col + GEMM (baseline)
  IMPLICIT_PRECOMP_GEMM Pre-computed im2col indices
  GEMM                  Explicit im2col matrix + cuBLAS GEMM
  DIRECT                Direct convolution (best for large kernels)
  FFT                   FFT-based convolution (best for large inputs)
  FFT_TILING            Tiled FFT for memory efficiency
  WINOGRAD              Winograd algorithm (fastest for 3×3 convolutions)
  WINOGRAD_NONFUSED     Non-fused Winograd

For 3×3 convolutions (the most common in ResNet, VGG):
  Winograd reduces arithmetic from 9 multiplications per output to 4
  → 2.25× algorithmic speedup before any hardware tricks
  torch.backends.cudnn.benchmark = True triggers search for your exact sizes
```

---

## 12.7 cuSPARSE — sparse linear algebra

For sparse matrices (most elements zero), dense operations waste compute
on zeros. cuSPARSE provides sparse formats and operations:

```python
import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np

# Create a sparse matrix (90% zeros)
N   = 10_000
nnz = int(N * N * 0.10)   # 10% non-zero

# Build sparse COO format, convert to CSR (standard for SpMV)
rows = np.random.randint(0, N, nnz)
cols = np.random.randint(0, N, nnz)
vals = np.random.randn(nnz).astype(np.float32)

# CuPy sparse matrix (backed by cuSPARSE)
A_sparse = csp.csr_matrix(
    (cp.array(vals), (cp.array(rows), cp.array(cols))),
    shape=(N, N)
)

x = cp.random.randn(N, dtype=cp.float32)

# Sparse matrix-vector multiply (SpMV) — cuSPARSE scsrmv
y = A_sparse @ x   # dispatches to cuSPARSE

# Benchmark sparse vs dense
A_dense = cp.array(A_sparse.toarray())

s = cp.cuda.Event(); e = cp.cuda.Event()
s.record(); _ = A_sparse @ x; e.record(); e.synchronize()
t_sparse = cp.cuda.get_elapsed_time(s, e)

s.record(); _ = A_dense @ x; e.record(); e.synchronize()
t_dense = cp.cuda.get_elapsed_time(s, e)

print(f"Sparse SpMV: {t_sparse:.3f} ms")
print(f"Dense  GEMV: {t_dense:.3f} ms")
print(f"Sparse speedup: {t_dense/t_sparse:.1f}x  (at 10% density)")
# Sparse wins for density < ~5–10%; dense wins above that
```

---

## 12.8 Library vs custom kernel — decision framework

```
Use the library when:
  ✅ Operation is a standard BLAS/DNN primitive (GEMM, conv, BN, attention)
  ✅ Input sizes are standard (powers of 2, typical DL shapes)
  ✅ You need FP16/BF16 with Tensor Cores (library handles this automatically)
  ✅ You need multi-GPU support (NCCL handles collective ops)
  ✅ Correctness is critical (libraries are extensively tested)

Write a custom kernel when:
  ✅ Your operation is not expressible as a standard primitive
  ✅ You need to fuse operations (e.g. GEMM + custom activation + quantise)
  ✅ You have non-standard data layouts the library doesn't support
  ✅ The library has overhead from descriptor setup that dominates for small ops
  ✅ You need a custom reduction operator (not sum/max/min)
  ✅ Library kernels don't achieve good performance on your specific shapes

The most common reason to write custom kernels in production ML:
  Kernel fusion — combining multiple operations into one kernel to avoid
  intermediate HBM reads/writes. Libraries apply each operation separately;
  a fused custom kernel saves N passes over the data.

Example: LayerNorm = mean + variance + normalise + scale + shift
  Library approach: 5 separate kernels, 5 passes over data
  Custom fused:     1 kernel, 1-2 passes over data → 3–5× faster
```

---

## 12.9 Profiling library calls

Library calls appear in Nsight Systems as CUDA kernel launches from
within the library, not your code. Use Nsight to see what's inside:

```python
import cupy as cp
import numpy as np

# Add NVTX markers around library calls to identify them in the timeline
cp.cuda.nvtx.RangePush("cuBLAS GEMM 2048x2048")
M = K = N = 2048
A = cp.random.randn(M, K, dtype=cp.float32)
B = cp.random.randn(K, N, dtype=cp.float32)
C = A @ B
cp.cuda.Stream.null.synchronize()
cp.cuda.nvtx.RangePop()

cp.cuda.nvtx.RangePush("cuFFT 1M points")
sig  = cp.random.randn(2**20, dtype=cp.float32)
freq = cp.fft.rfft(sig)
cp.cuda.Stream.null.synchronize()
cp.cuda.nvtx.RangePop()

# Run under: nsys profile python this_script.py
# In the timeline: expand the NVTX regions to see the internal kernels
# cuBLAS launches 1-3 kernels internally
# cuFFT launches several kernels (mixed-radix decomposition)
```

### What Nsight shows for a cuBLAS GEMM

```
NVTX: cuBLAS GEMM 2048x2048
  └── CUDA kernel: ampere_sgemm_128x128_tn (or similar name)
      SM utilisation:     ~98%
      Achieved GFLOP/s:   ~12,000  (A100 FP32)
      Shared mem used:    ~96 KB
      Register use:       ~128 per thread
      Occupancy:          ~50% (register-limited, intentionally)
      Tile size:          128×128×8 (much larger than your Ch 5 kernel)
```

Note the register count: 128 registers per thread at 50% occupancy is a
deliberate tradeoff — more registers enables a larger tile, which gives
higher arithmetic intensity and outweighs the lower occupancy.
This is the compute-bound case from Chapter 10 section 10.8.

---

## 12.10 Complete benchmarking suite

```python
import cupy as cp
import numpy as np

def run_library_benchmarks():
    """
    Benchmark the main library operations.
    Compare against theoretical peak to see efficiency.
    """
    results = {}

    # T4 specs (update for your GPU):
    # Peak FP32 TFLOP/s: 8.1,  Peak HBM BW: 300 GB/s
    PEAK_FLOPS = 8_100  # GFLOP/s
    PEAK_BW    = 300    # GB/s

    def bmark(fn, n_warmup=5, n_rep=20):
        for _ in range(n_warmup): fn()
        cp.cuda.Stream.null.synchronize()
        ts = []
        for _ in range(n_rep):
            s = cp.cuda.Event(); e = cp.cuda.Event()
            s.record(); fn(); e.record(); e.synchronize()
            ts.append(cp.cuda.get_elapsed_time(s, e))
        return np.min(ts)

    # ── GEMM ──────────────────────────────────────────────────────────────
    for sz in [512, 1024, 2048, 4096]:
        A = cp.random.randn(sz, sz, dtype=cp.float32)
        B = cp.random.randn(sz, sz, dtype=cp.float32)
        t = bmark(lambda: A @ B)
        gf = 2 * sz**3 / (t * 1e-3) / 1e9
        results[f'gemm_{sz}'] = (t, gf, gf/PEAK_FLOPS*100)
        print(f"GEMM {sz:4d}×{sz}: {t:7.2f} ms  {gf:7.0f} GFLOP/s  "
              f"({gf/PEAK_FLOPS*100:.1f}% of peak)")

    # ── FFT ───────────────────────────────────────────────────────────────
    for exp in [18, 20, 22]:
        N   = 2**exp
        sig = cp.random.randn(N, dtype=cp.float32)
        t   = bmark(lambda: cp.fft.rfft(sig))
        # FFT FLOP count: 5 * N * log2(N)
        gf  = 5 * N * exp / (t * 1e-3) / 1e9
        print(f"FFT  2^{exp} ({N//1024}K pts): {t:7.2f} ms  {gf:7.0f} GFLOP/s")

    # ── Vector ops (Level 1) ──────────────────────────────────────────────
    N = 50_000_000
    x = cp.random.randn(N, dtype=cp.float32)
    y = cp.random.randn(N, dtype=cp.float32)

    t   = bmark(lambda: cp.dot(x, y))
    bw  = 2 * N * 4 / (t * 1e-3) / 1e9
    print(f"DOT  {N//1e6:.0f}M floats:  {t:7.2f} ms  {bw:7.0f} GB/s  "
          f"({bw/PEAK_BW*100:.1f}% of BW peak)")

    t   = bmark(lambda: x + y)
    bw  = 3 * N * 4 / (t * 1e-3) / 1e9
    print(f"AXPY {N//1e6:.0f}M floats:  {t:7.2f} ms  {bw:7.0f} GB/s  "
          f"({bw/PEAK_BW*100:.1f}% of BW peak)")

    return results

run_library_benchmarks()
```

---

## Hands-on exercises

### Exercise 1 — GEMM efficiency ladder

Run `cp.matmul` (cuBLAS) for square matrices from 64×64 to 8192×8192.
Plot GFLOP/s vs matrix size:

```python
sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
for N in sizes:
    A = cp.random.randn(N, N, dtype=cp.float32)
    B = cp.random.randn(N, N, dtype=cp.float32)
    # Measure and record GFLOP/s
```

Questions:
- At what size does cuBLAS reach >80% of peak FLOP/s?
- Why is efficiency low for small matrices?
- What does this tell you about when to batch small GEMMs vs run them
  sequentially?

### Exercise 2 — cuBLAS vs your Chapter 5 kernel

On your GPU, run both `matmul_tiled` (Chapter 5) and `cp.matmul` (cuBLAS)
for M=K=N = 512, 1024, 2048, 4096.

```
Size    Your kernel (GFLOP/s)   cuBLAS (GFLOP/s)   Gap ratio
──────────────────────────────────────────────────────────────
512
1024
2048
4096
```

Does the gap grow or shrink as size increases? Why?

### Exercise 3 — FFT vs direct convolution

For a 1D signal of length N and a filter of length K, both FFT-based
and direct convolution compute the same result. Benchmark both for:

```python
import cupy as cp
from cupyx.scipy.signal import fftconvolve, convolve

N_values = [1_000, 10_000, 100_000, 1_000_000]
K = 64   # filter length

for N in N_values:
    sig    = cp.random.randn(N,   dtype=cp.float32)
    filt   = cp.random.randn(K,   dtype=cp.float32)
    # Benchmark fftconvolve vs direct convolve
    # At what N does FFT-based become faster?
```

This crossover point is why cuDNN switches from direct to FFT algorithm
at a certain input size.

### Exercise 4 — Sparse vs dense crossover

Vary the sparsity of a 5000×5000 matrix and benchmark SpMV (sparse
matrix-vector multiply) vs dense GEMV:

```python
for density in [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
    # Build sparse matrix at this density
    # Benchmark sparse @ x vs dense @ x
    # Record crossover density
```

At what density does dense GEMV become faster than SpMV?
Why does the crossover happen (think about memory access patterns)?

### Exercise 5 — Kernel fusion opportunity analysis

This code runs four separate CuPy operations:

```python
import cupy as cp

N = 10_000_000
x = cp.random.randn(N, dtype=cp.float32)
W = cp.random.randn(N, N, dtype=cp.float32)   # large — skip for memory

# Fusion candidate: element-wise pipeline
def pipeline_unfused(x):
    a = cp.exp(-x)              # pass 1: exp
    b = 1.0 / (1.0 + a)        # pass 2: sigmoid
    c = b * (1.0 - b)           # pass 3: sigmoid derivative
    d = c / cp.sqrt(            # pass 4: normalise
            cp.sum(c ** 2) + 1e-8)
    return d
```

a) How many passes over the N-element data does this make?
b) How many bytes are read+written total?
c) Write a fused version using `cp.ElementwiseKernel` that does all
   four steps in a single pass.
d) Benchmark fused vs unfused. What speedup do you get?
e) Is this kernel memory-bound or compute-bound? Use the roofline
   model to verify.

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Library first | cuBLAS / cuDNN cover ~80% of DL compute — always check before writing kernels |
| BLAS levels | L1/L2 always memory-bound; L3 (GEMM) compute-bound for large N — GPU excels |
| GEMM α/β | Fused multiply-accumulate: `C = αAB + βC` avoids a separate kernel for scaling |
| cuBLAS gap | 5–20× faster than hand-written TILE=16 kernel — Tensor Cores + large tiles |
| cuDNN benchmark | `torch.backends.cudnn.benchmark=True` auto-selects fastest conv algorithm |
| Winograd | 3×3 convolutions need only 4 muls vs 9 — cuDNN selects automatically |
| Library vs custom | Custom when: non-standard op, fusion needed, unusual shapes/layout |
| Fusion benefit | N operations → 1 pass: N-1 fewer HBM round-trips → 2–5× speedup |
| Sparse crossover | cuSPARSE wins for density < ~5–10%; dense wins above (random access penalty) |
| Profiling libraries | NVTX markers + Nsight Systems shows internal cuBLAS/cuDNN kernel names |

---

## Phase 3 complete — what's next

You have now completed **Phase 3: Performance Engineering** (Chapters 9–12).
You can profile any kernel, identify its bottleneck, choose between library
and custom implementations, and apply the core optimisation techniques.

**Phase 4 begins with Chapter 13: Custom CUDA Extensions for PyTorch.**
You will write a kernel in CUDA C, bind it to Python with pybind11,
register it as an autograd Function with correct forward and backward passes,
and benchmark it against the equivalent PyTorch built-in. This is the
workflow used by teams building production ML infrastructure.

Paste your Exercise 1 GEMM efficiency ladder and Exercise 5 fusion
speedup — both connect directly to Chapter 13's motivation.
