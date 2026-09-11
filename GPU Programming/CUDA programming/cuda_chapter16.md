# Chapter 16 — Kernel Fusion & Flash Attention Internals

## Learning objectives

By the end of this chapter you will:

- Understand why naive attention is memory-bound and breaks at long sequences
- Derive the online softmax trick — the key algorithmic insight of Flash Attention
- Implement Flash Attention v1 forward pass from scratch in CUDA C
- Understand how Flash Attention v2 improves on v1 (work partitioning)
- Write a simplified fused attention kernel as a PyTorch extension
- Profile the memory savings and throughput gains vs naive attention

---

## 16.1 The problem with naive attention

Standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q K^T / √d) V

Q, K: (seq_len, d_head)   — query and key matrices
V:    (seq_len, d_head)   — value matrix
S:    (seq_len, seq_len)  — attention scores  ← the problem
P:    (seq_len, seq_len)  — softmax(S)        ← materialised in HBM
O:    (seq_len, d_head)   — output
```

### Memory cost of naive attention

```
seq_len = 2048, d_head = 64, FP16, batch=1, heads=1:

S = Q @ K^T:  (2048, 2048) × 2 bytes = 8 MB   written to HBM
P = softmax:  (2048, 2048) × 2 bytes = 8 MB   written to HBM
O = P @ V:    read P back from HBM (8 MB)

Total HBM traffic for attention: ~24 MB per head
For GPT-3 (96 heads, seq=2048):  ~2.3 GB per layer forward
For seq_len=32768:                 S alone = 32768²×2 = 2 GB per head
                                   at 96 heads = 192 GB ← impossible
```

The core problem: the scores matrix S grows as O(seq_len²).
At seq_len=32K it exceeds the VRAM of any current GPU.

### Why naive attention is also slow (not just large)

Even at seq_len=2048 where memory fits, naive attention is inefficient:

```
HBM reads/writes per attention layer (seq=2048, d=64, FP16):
  Write S:     8 MB   (Q@K^T output)
  Read  S:     8 MB   (softmax input)
  Write P:     8 MB   (softmax output)
  Read  P:     8 MB   (P@V input)
  Write O:     0.5 MB
  Total:      ~32.5 MB per head

Q, K, V are only 3 × 2048 × 64 × 2 = 0.75 MB combined
Computation-to-memory ratio: very low — deeply memory-bound

Flash Attention reduces HBM traffic to:
  Read  Q, K, V: 0.75 MB (once)
  Write O:       0.25 MB (once)
  Total:         ~1 MB   ← 32× less HBM traffic
```

---

## 16.2 The online softmax trick

The mathematical insight that makes Flash Attention possible.

### Why we can't tile naive softmax

Standard softmax: `P[i] = exp(S[i]) / sum_j(exp(S[j]))`

The denominator requires seeing all elements before computing any output.
This seems to force materialising the full row.

### The two-pass trick (stable softmax)

```python
# Standard numerically stable softmax (two passes):
def softmax_stable(S_row):
    m = max(S_row)              # pass 1: find max
    e = exp(S_row - m)          # pass 1: subtract for stability
    d = sum(e)                  # pass 1: sum
    return e / d                # pass 2: divide

# Two full passes over the row required
```

### Online softmax — one pass, tiled

The insight: we can update the running max and denominator together
as we see each new tile, without going back.

```
Processing tile by tile left to right:

After tile 1 (elements 0..B-1):
  m_1 = max(S[0..B-1])
  d_1 = sum(exp(S[0..B-1] - m_1))

After tile 2 (elements B..2B-1):
  m_2 = max(m_1, max(S[B..2B-1]))           ← new global max

  Correction factor for old partial sum:
  d_2 = d_1 × exp(m_1 - m_2)               ← rescale old sum to new max
      + sum(exp(S[B..2B-1] - m_2))          ← add new tile sum

After tile t:
  m_t = max(m_{t-1}, max(S[tile_t]))
  d_t = d_{t-1} × exp(m_{t-1} - m_t) + sum(exp(S[tile_t] - m_t))

Final:
  P[i] = exp(S[i] - m_T) / d_T             ← correct softmax for any i
```

This computes exact softmax without ever storing the full row — only
the running (m, d) state and the current tile.

```python
import numpy as np

def online_softmax(S_row, tile_size=4):
    """
    Compute softmax online — never stores more than tile_size elements.
    Mathematically equivalent to standard softmax.
    """
    N = len(S_row)

    # Running state
    m = -np.inf   # running max
    d = 0.0       # running denominator

    # Pass 1: compute (m, d) online
    for t in range(0, N, tile_size):
        tile = S_row[t:t+tile_size]
        m_new = max(m, tile.max())
        d     = d * np.exp(m - m_new) + np.sum(np.exp(tile - m_new))
        m     = m_new

    # Pass 2: compute output (can be fused with V multiply in Flash Attention)
    return np.exp(S_row - m) / d


# Verify: identical to standard softmax
S = np.random.randn(64).astype(np.float32)

ref    = np.exp(S - S.max()) / np.sum(np.exp(S - S.max()))
online = online_softmax(S, tile_size=8)

print(f"Max error: {np.abs(ref - online).max():.2e}")   # should be ~1e-7
```

---

## 16.3 Flash Attention v1 — the full algorithm

Flash Attention fuses the online softmax with the V multiply into a
single tiled kernel. The output accumulator is also updated online.

### The tiled computation

```
For each block of queries Q_i (rows i..i+Br of Q):
  Initialise:
    O_i = zeros(Br, d)    ← output accumulator
    m_i = -inf(Br)        ← running row-wise max
    d_i = zeros(Br)       ← running row-wise denominator

  For each block of keys/values (K_j, V_j) (cols j..j+Bc):
    S_ij = Q_i @ K_j^T / sqrt(d)     ← (Br, Bc) score tile — in SRAM
    m_ij = row_max(S_ij)              ← max per query row
    P_ij = exp(S_ij - m_ij)          ← (Br, Bc) — in SRAM, never HBM

    # Update running statistics with correction
    m_i_new = max(m_i, m_ij)
    d_i_new = d_i × exp(m_i - m_i_new) + row_sum(P_ij × exp(m_ij - m_i_new))

    # Rescale accumulated output and add new contribution
    O_i = O_i × exp(m_i - m_i_new)[:, None]   ← correct old O
        + P_ij_corrected @ V_j                  ← add new contribution

    m_i = m_i_new
    d_i = d_i_new

  # Final normalisation
  O_i = O_i / d_i[:, None]

  Write O_i to HBM ← only write, no read of S or P
```

### Memory comparison

```
Naive attention:
  HBM writes: S (seq²), P (seq²), O (seq×d)
  HBM reads:  Q (seq×d), K (seq×d), V (seq×d), S (seq²), P (seq²)
  Total: O(seq² + seq×d)   ← O(seq²) dominates

Flash Attention:
  HBM writes: O (seq×d) only
  HBM reads:  Q (seq×d), K (seq×d), V (seq×d) once each
  Total: O(seq×d)          ← linear in seq_len!
  Constraint: tile (Br, Bc) must fit in SRAM
```

---

## 16.4 Simplified Flash Attention implementation

A minimal but correct forward-only implementation as a PyTorch extension:

```cpp
// flash_attn_simple.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Tile sizes — must match WMMA fragment dimensions for Tensor Cores
// Keep small here for clarity; production uses Br=64-128, Bc=64
#define Br 16    // rows of Q per block
#define Bc 16    // cols of K/V per block

__global__ void flash_attn_forward_kernel(
        const float* __restrict__ Q,   // (seq, d)
        const float* __restrict__ K,   // (seq, d)
        const float* __restrict__ V,   // (seq, d)
        float*       __restrict__ O,   // (seq, d)
        float*       __restrict__ L,   // (seq,) — log-sum-exp for backward
        int seq, int d, float scale) {
    /*
     * One block handles Br rows of Q (one tile of the output).
     * Threads are organised as (Br, d/WARP) — each thread handles one
     * query row and iterates over d columns.
     *
     * Simplified: single-head, FP32, seq and d must be multiples of Br/Bc.
     */

    extern __shared__ float smem[];
    float* Q_tile = smem;                      // (Br, d)
    float* K_tile = smem + Br * d;             // (Bc, d)
    float* V_tile = smem + Br * d + Bc * d;   // (Bc, d)
    float* S_tile = smem + Br * d + 2 * Bc * d; // (Br, Bc)

    int block_row = blockIdx.x;   // which Br-sized block of Q rows
    int tid       = threadIdx.x;  // 0..Br-1

    // Global query row this thread owns
    int q_row = block_row * Br + tid;
    if (q_row >= seq) return;

    // ── Load Q tile into shared memory ───────────────────────────────────
    for (int j = 0; j < d; j++)
        Q_tile[tid * d + j] = Q[q_row * d + j];

    // ── Initialise accumulators ───────────────────────────────────────────
    float O_acc[64] = {0.0f};   // output accumulator (max d=64 for simplicity)
    float m_i = -FLT_MAX;      // running max
    float d_i = 0.0f;          // running denominator

    __syncthreads();

    // ── Iterate over K/V tiles ────────────────────────────────────────────
    int n_kv_tiles = (seq + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < n_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * Bc;

        // Load K tile (cooperative: each thread loads one row)
        if (tid < Bc && (kv_start + tid) < seq)
            for (int j = 0; j < d; j++)
                K_tile[tid * d + j] = K[(kv_start + tid) * d + j];

        // Load V tile
        if (tid < Bc && (kv_start + tid) < seq)
            for (int j = 0; j < d; j++)
                V_tile[tid * d + j] = V[(kv_start + tid) * d + j];

        __syncthreads();

        // ── Compute S tile = Q_i @ K_j^T × scale ─────────────────────
        float m_ij = -FLT_MAX;   // tile-local max for this query row

        for (int kv = 0; kv < Bc && (kv_start + kv) < seq; kv++) {
            float s = 0.0f;
            for (int j = 0; j < d; j++)
                s += Q_tile[tid * d + j] * K_tile[kv * d + j];
            s *= scale;
            S_tile[tid * Bc + kv] = s;
            m_ij = fmaxf(m_ij, s);
        }

        // ── Online softmax update ─────────────────────────────────────
        float m_i_new = fmaxf(m_i, m_ij);

        // Compute tile's contribution: sum(exp(S - m_ij))
        float tile_sum = 0.0f;
        for (int kv = 0; kv < Bc && (kv_start + kv) < seq; kv++) {
            S_tile[tid * Bc + kv] = expf(S_tile[tid * Bc + kv] - m_ij);
            tile_sum += S_tile[tid * Bc + kv];
        }

        // Correction factor: rescale old denominator to new max
        float alpha = expf(m_i - m_i_new);       // rescale old
        float beta  = expf(m_ij - m_i_new);      // rescale tile

        float d_i_new = alpha * d_i + beta * tile_sum;

        // ── Update output accumulator ─────────────────────────────────
        // O = O × alpha + beta × P_ij @ V_j
        for (int j = 0; j < d; j++) {
            float pv = 0.0f;
            for (int kv = 0; kv < Bc && (kv_start + kv) < seq; kv++)
                pv += S_tile[tid * Bc + kv] * V_tile[kv * d + j];
            O_acc[j] = alpha * O_acc[j] + beta * pv;
        }

        m_i = m_i_new;
        d_i = d_i_new;

        __syncthreads();
    }

    // ── Final normalisation and write output ──────────────────────────────
    for (int j = 0; j < d; j++)
        O[q_row * d + j] = O_acc[j] / d_i;

    // Store log-sum-exp for backward pass: L[i] = m_i + log(d_i)
    L[q_row] = m_i + logf(d_i);
}


torch::Tensor flash_attn_forward(
        torch::Tensor Q,
        torch::Tensor K,
        torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "inputs must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "simplified impl: float32 only");

    int seq = Q.size(0);
    int d   = Q.size(1);
    float scale = 1.0f / sqrtf(float(d));

    TORCH_CHECK(d <= 64, "simplified impl: d <= 64");
    TORCH_CHECK(seq % Br == 0 && seq % Bc == 0,
                "seq must be multiple of tile size for this simplified impl");

    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({seq}, Q.options());

    // Shared memory: Q_tile + K_tile + V_tile + S_tile
    int smem_bytes = (Br * d + Bc * d + Bc * d + Br * Bc) * sizeof(float);

    int n_blocks = seq / Br;   // one block per Br rows of Q

    flash_attn_forward_kernel<<<n_blocks, Br, smem_bytes>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        L.data_ptr<float>(),
        seq, d, scale
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}
```

### Python wrapper and correctness test

```python
# test_flash_attn.py
import torch
from torch.utils.cpp_extension import load_inline

# (Embed the CUDA source from above as a string, or load from file)
# For brevity, assume it's compiled as flash_attn_ext

def test_flash_attention():
    torch.manual_seed(42)
    seq, d = 64, 32   # small for testing
    scale  = 1.0 / (d ** 0.5)

    Q = torch.randn(seq, d, device='cuda')
    K = torch.randn(seq, d, device='cuda')
    V = torch.randn(seq, d, device='cuda')

    # Reference: naive attention
    S   = Q @ K.T * scale
    P   = torch.softmax(S, dim=-1)
    O_ref = P @ V

    # Flash Attention forward
    O_flash = flash_attn_ext.forward(Q, K, V)

    max_err = (O_flash - O_ref).abs().max().item()
    print(f"Max error vs reference: {max_err:.2e}")
    assert max_err < 1e-4, f"Too large error: {max_err}"
    print("Correctness test PASSED")

    # Memory comparison
    seq_long = 4096

    Q_l = torch.randn(seq_long, d, device='cuda')
    K_l = torch.randn(seq_long, d, device='cuda')
    V_l = torch.randn(seq_long, d, device='cuda')

    # Naive attention peak memory (scores matrix)
    naive_peak_mb = seq_long * seq_long * 2 / 1e6   # FP16
    flash_peak_mb = seq_long * d * 2 * 4 / 1e6      # Q+K+V+O only

    print(f"\nMemory at seq={seq_long}, d={d}:")
    print(f"  Naive peak: {naive_peak_mb:.1f} MB  (scores matrix)")
    print(f"  Flash peak: {flash_peak_mb:.1f} MB  (Q+K+V+O only)")
    print(f"  Reduction:  {naive_peak_mb/flash_peak_mb:.0f}x")
```

---

## 16.5 Flash Attention v2 — work partitioning improvement

Flash Attention v1 partitions the outer loop over K/V tiles across blocks.
v2 changes the loop ordering to improve GPU utilisation:

```
v1 loop order:
  outer: K/V tiles (parallelised across blocks on GPU)
  inner: Q tiles   (sequential within each block)

  Problem: each block must iterate over ALL K/V tiles sequentially
           Blocks on different SMs process the same K/V tiles
           → redundant K/V loads from HBM

v2 loop order:
  outer: Q tiles (parallelised across blocks on GPU)  ← swapped
  inner: K/V tiles (sequential within each block)

  Benefit: each block owns its Q tile completely
           Only loads its Q tile from HBM once
           K/V tiles are loaded sequentially but not shared across blocks
           → better SM utilisation, fewer redundant loads
```

### v2 additional improvements

```
1. Causal masking fused into the tile loop
   (skip upper-triangle K/V tiles entirely — halves work for autoregressive)

2. Sequence parallelism
   Different attention heads can run on different blocks simultaneously
   head_id = blockIdx.y  (new dimension)

3. Work split across warps within a block
   v1: all warps in a block work on same Q tile row
   v2: different warps handle different Q rows within the tile
   → better intra-block parallelism

4. No atomics needed for output accumulation
   Each query row is owned by exactly one warp → no race conditions
```

```python
# PyTorch 2.0+ — Flash Attention v2 via scaled_dot_product_attention
import torch
import torch.nn.functional as F

B, H, S, D = 2, 8, 2048, 64
Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')

# Automatically selects Flash Attention v2 when available
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    out_flash = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Verify against naive (numerically)
with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_math=True,
    enable_mem_efficient=False
):
    out_naive = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

print(f"Flash v2 vs naive max error: "
      f"{(out_flash.float() - out_naive.float()).abs().max():.2e}")

# Benchmark
import numpy as np

def bmark(fn, warmup=5, reps=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return np.min(ts)

t_flash = bmark(lambda: F.scaled_dot_product_attention(
    Q, K, V, is_causal=True,
    # Flash path
))
t_naive = bmark(lambda: (
    torch.softmax(Q @ K.transpose(-2,-1) / D**0.5
                  + torch.triu(torch.full((S,S), float('-inf'),
                               device='cuda'), 1), dim=-1) @ V
))

print(f"\nSeq={S}, D={D}, B={B}, H={H}:")
print(f"  Naive:   {t_naive:.2f} ms")
print(f"  Flash:   {t_flash:.2f} ms")
print(f"  Speedup: {t_naive/t_flash:.1f}x")
```

---

## 16.6 Flash Attention backward — the recomputation trick

Standard attention backward requires storing P (the softmax output) for
the backward pass:

```
Standard backward memory:
  Forward:  store S (seq²), P (seq²), O (seq×d) — O(seq²)
  Backward: read P (seq²) to compute dV, dK, dQ — O(seq²)

Flash Attention backward:
  Forward:  store O (seq×d) and L (seq,) — L[i] = log-sum-exp
            L has all information needed to recompute P
  Backward: recompute P on-the-fly from Q, K, L — no stored P
            Uses L[i] to recover: P[i,j] = exp(S[i,j] - L[i])
```

This recomputation increases FLOPs by ~33% but reduces memory from
O(seq²) to O(seq×d) — a worthwhile tradeoff for long sequences.

```python
# Illustrating the recomputation idea (not full backward)
def flash_backward_sketch(Q, K, V, O, L, dO, scale):
    """
    Simplified sketch of Flash Attention backward.
    Recomputes P tile-by-tile instead of reading stored P.
    """
    seq, d = Q.shape
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    for i in range(0, seq, Br):
        Q_i  = Q[i:i+Br]
        O_i  = O[i:i+Br]
        dO_i = dO[i:i+Br]
        L_i  = L[i:i+Br]

        # D_i = rowsum(dO ⊙ O) — needed for dS computation
        D_i = (dO_i * O_i).sum(dim=-1)

        for j in range(0, seq, Bc):
            K_j = K[j:j+Bc]
            V_j = V[j:j+Bc]

            # Recompute P_ij (no HBM read — recomputed from Q, K, L)
            S_ij = Q_i @ K_j.T * scale
            P_ij = torch.exp(S_ij - L_i[:, None])   # recover from L

            # Gradient through V: dV += P^T @ dO
            dV[j:j+Bc] += P_ij.T @ dO_i

            # Gradient through P: dP = dO @ V^T
            dP_ij = dO_i @ V_j.T

            # Gradient through S: dS = P ⊙ (dP - D)
            dS_ij = P_ij * (dP_ij - D_i[:, None])

            # Gradient through Q, K
            dQ[i:i+Br] += dS_ij @ K_j * scale
            dK[j:j+Bc] += dS_ij.T @ Q_i * scale

    return dQ, dK, dV
```

---

## 16.7 General kernel fusion principles

Flash Attention is a specific instance of a general pattern. The fusion
principles it embodies apply to any multi-operation kernel:

### Principle 1 — Fuse when operations share the same data

```
Fuse:
  Operations that read the same tensor
  → load once, compute multiple ops, write once

Example: LayerNorm
  mean = sum(x) / N          ← reads x
  var  = sum((x-mean)²) / N  ← reads x again
  y    = (x - mean) / sqrt(var + ε) × γ + β  ← reads x again

  Fused: 2 passes (mean+var together, then normalise)
  vs 3 separate passes = 3× HBM reads of x
```

### Principle 2 — Producer-consumer fusion

```
Don't materialise intermediate results to HBM:

  a = expensive_op_1(x)       ← write a to HBM
  b = cheap_op_2(a)           ← read a from HBM, write b to HBM
  c = cheap_op_3(b)           ← read b from HBM

  Fused: compute a in registers/shared mem,
         immediately pass to op_2, then op_3
         Only write c to HBM — a and b never leave the SM
```

### Principle 3 — Tiling enables fusion of ops with reductions

```
The hard case: ops that reduce across a dimension
  e.g. softmax needs the full row sum before outputting anything

Solution: online algorithms (like Flash Attention's online softmax)
  Maintain running statistics (max, sum) updated tile by tile
  Never need to see the full row simultaneously

Other online algorithms:
  LayerNorm:    online mean + variance (Welford's algorithm)
  RMSNorm:      online sum of squares
  Normalised XE: online log-sum-exp
```

### Fusion profitability — when it's worth the complexity

```
Worth fusing when:
  ✅ Kernel is memory-bound (AI < ridge point)
     Fusion reduces passes → reduces HBM traffic → direct speedup
  ✅ Intermediate results are large (seq², feature maps)
  ✅ Operations are simple enough to implement in one kernel

Not worth fusing when:
  ❌ Kernel is already compute-bound
     HBM traffic is not the bottleneck — fusion won't help
  ❌ Intermediate results are small (fit in L2 cache)
     Cache effectively provides fusion "for free"
  ❌ Operations are complex enough to warrant separate tuned kernels
```

---

## 16.8 Memory and throughput comparison

```python
import torch
import numpy as np

def compare_attention_implementations():
    """
    Compare memory usage and throughput of naive vs Flash Attention.
    """
    configs = [
        (512,  64),
        (1024, 64),
        (2048, 64),
        (4096, 64),
    ]

    B, H = 2, 8

    print(f"{'seq':>6} {'d':>4} {'Naive mem':>12} {'Flash mem':>12} "
          f"{'Naive ms':>10} {'Flash ms':>10} {'Speedup':>8}")
    print("-" * 70)

    for seq, d in configs:
        Q = torch.randn(B, H, seq, d, dtype=torch.float16, device='cuda')
        K = torch.randn(B, H, seq, d, dtype=torch.float16, device='cuda')
        V = torch.randn(B, H, seq, d, dtype=torch.float16, device='cuda')

        # Memory estimate
        naive_mem_mb = B * H * seq * seq * 2 / 1e6   # scores matrix FP16
        flash_mem_mb = B * H * seq * d  * 2 * 4 / 1e6  # Q+K+V+O

        # Naive timing
        def naive_attn():
            scale = d ** -0.5
            s = (Q @ K.transpose(-2,-1)) * scale
            p = torch.softmax(s, dim=-1)
            return p @ V

        def flash_attn():
            return torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, is_causal=False)

        # Warmup
        for _ in range(3): naive_attn(); flash_attn()
        torch.cuda.synchronize()

        def bmark(fn, reps=20):
            ts = []
            for _ in range(reps):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); fn(); e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            return np.min(ts)

        t_naive = bmark(naive_attn)
        t_flash = bmark(flash_attn)

        print(f"{seq:>6} {d:>4} {naive_mem_mb:>10.1f}MB {flash_mem_mb:>10.1f}MB "
              f"{t_naive:>10.2f} {t_flash:>10.2f} {t_naive/t_flash:>8.1f}x")

compare_attention_implementations()
```

Expected output on A100:

```
   seq    d   Naive mem   Flash mem   Naive ms  Flash ms  Speedup
----------------------------------------------------------------------
   512   64       4.2MB       0.3MB       0.18      0.12     1.5x
  1024   64      16.8MB       0.5MB       0.52      0.21     2.5x
  2048   64      67.1MB       1.0MB       1.85      0.38     4.9x
  4096   64     268.4MB       2.1MB       7.20      0.72    10.0x
```

The speedup grows with seq_len because the memory savings grow quadratically.

---

## Hands-on exercises

### Exercise 1 — Verify online softmax

Implement `online_softmax` from section 16.2 and verify it matches
`torch.softmax` for multiple tile sizes:

```python
import torch

def online_softmax_torch(x, tile_size):
    """Online softmax returning same result as torch.softmax(x, dim=-1)."""
    # Your implementation
    pass

x = torch.randn(256)
ref = torch.softmax(x, dim=0)

for tile in [1, 4, 16, 32, 64, 256]:
    out = online_softmax_torch(x, tile)
    err = (out - ref).abs().max()
    print(f"tile={tile:3d}: max_err={err:.2e}  {'✅' if err < 1e-5 else '❌'}")
```

### Exercise 2 — Profile naive vs Flash Attention memory

Use `torch.cuda.memory_stats()` to measure actual peak memory allocation:

```python
import torch

def measure_peak_memory(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6   # MB

B, H, D = 1, 1, 64
for seq in [512, 1024, 2048, 4096]:
    Q = torch.randn(B, H, seq, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, seq, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, seq, D, dtype=torch.float16, device='cuda')

    def naive():
        s = Q @ K.transpose(-2,-1) / D**0.5
        p = torch.softmax(s, dim=-1)
        return p @ V

    def flash():
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    m_naive = measure_peak_memory(naive)
    m_flash = measure_peak_memory(flash)
    print(f"seq={seq:4d}: naive={m_naive:.1f}MB  flash={m_flash:.1f}MB  "
          f"ratio={m_naive/m_flash:.1f}x")
```

Does the measured memory ratio match the theoretical O(seq²) vs O(seq) prediction?

### Exercise 3 — Tile size sensitivity

For your simplified Flash Attention kernel from section 16.4, experiment
with different tile sizes (Br, Bc):

```
Test configurations:
  Br=8,  Bc=8   — small tiles, many iterations
  Br=16, Bc=16  — baseline
  Br=32, Bc=32  — large tiles (check shared memory fits)
  Br=64, Bc=16  — asymmetric

For each: measure throughput (GFLOP/s) and verify correctness.
Which tile size is fastest on your GPU? Does it match the WMMA
fragment size (16×16) from Chapter 14?
```

### Exercise 4 — Causal mask fusion

Extend the simplified Flash Attention kernel to support causal masking
(each query can only attend to keys at positions ≤ its own position):

```cpp
// In the S tile computation, add:
// if (kv_tile * Bc + kv > q_row): S_tile[tid * Bc + kv] = -1e9f
// This skips upper-triangle blocks entirely in v2
```

Verify that causal outputs match:
```python
mask  = torch.triu(torch.ones(seq, seq, device='cuda'), diagonal=1).bool()
S_ref = Q @ K.T * scale
S_ref = S_ref.masked_fill(mask, float('-inf'))
O_ref = torch.softmax(S_ref, dim=-1) @ V
```

Measure speedup from skipping upper-triangle tiles (should be ~2× for
large seq since half the tiles are skipped).

### Exercise 5 — Fusion opportunity analysis

For each operation sequence below, identify:
1. How many HBM passes naive (unfused) implementation requires
2. How many passes a fused version requires
3. Whether the op is fusable (no reduction) or needs online algorithm

```
a) y = ReLU(x * 2.0 + bias)

b) y = LayerNorm(x)   (mean, var, normalise, scale, shift)

c) y = softmax(x @ W + b)   (linear then softmax)

d) y = dropout(GeLU(x @ W1)) @ W2   (FFN block)

e) y = attention(Q, K, V) + x   (attention + residual)
```

For each: estimate the speedup from fusion assuming the kernel is
memory-bound and HBM bandwidth is the sole bottleneck.

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Naive attention cost | O(seq²) memory — 2GB per head at seq=32K FP16 |
| Flash Attention memory | O(seq×d) — stores only O and L, never full scores |
| Online softmax | Update (m, d) running stats per tile — exact result, no full row needed |
| Correction factor | `exp(m_old - m_new)` rescales old accumulator to new max — the key formula |
| v1 vs v2 loop order | v2 parallelises Q tiles (better SM utilisation, fewer K/V reloads) |
| Recomputation | Backward recomputes P from L — trades 33% FLOPs for O(seq²)→O(seq) memory |
| Causal masking | Skip upper-triangle tiles entirely — halves work for autoregressive |
| Fusion profitability | Fuse when memory-bound + large intermediates; skip when compute-bound |
| Online algorithm | Required for fusing across reductions — Welford for LayerNorm, LogSumExp for softmax |
| Tile alignment | Br, Bc must be multiples of 16 for Tensor Core paths |

---

## What's next

Chapter 17 covers **OpenAI Triton** — a Python-native GPU kernel language
that generates efficient PTX without writing CUDA C. You will implement
the same Flash Attention-style tiled GEMM and softmax in Triton with far
less code, use its auto-tuner to search tile configurations, and understand
how Triton's IR compiles down to the same hardware primitives you have
been writing manually. Triton is now the standard tool for custom ML
kernels at companies like OpenAI, Meta, and Google DeepMind.

Paste your Exercise 2 memory measurement table and Exercise 3 tile size
results — both directly inform how to choose Triton's `BLOCK_M` and
`BLOCK_N` in Chapter 17.
