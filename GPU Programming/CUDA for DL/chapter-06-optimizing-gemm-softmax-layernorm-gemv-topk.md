# Chapter 6 — Optimizing GEMM, Softmax, LayerNorm, GEMV & Top-K

*Part 5: The Optimization Ladder — Memory-Bound & Compute-Bound Kernels. Confirmed scope from the book's companion repo: `book.cu/4_optim/README.md`, "CUDA Kernel Optimizations" — five operations, each as a numbered ladder of kernels in its own `kernels/` folder, benchmarked against PyTorch with a shared harness.*

This is the chapter where every naive kernel from Chapters 3–5 gets its fix. The single-thread-per-row anti-pattern you've now seen three times — Chapter 4's `softmax_kernel<<<batch_size,1>>>`, Chapter 5's non-cooperative `gemv_kernel`, Chapter 5's naive `topk_kernel` — all get properly parallelized versions in this chapter's real source. Even better: the repo's own code comments cite exactly where these techniques come from, so the "expert resources" from this course's bibliography aren't abstract references — they're the book's own stated sources: **Simon Boehm's "How to Optimize a CUDA Matmul Kernel" blog** (cited directly in `gemm/README.md`), **Maharshi Pandya's CUDA optimization blog** (Apache-2.0, cited in the softmax/GEMV kernel headers), and **Andrej Karpathy's `llm.c`** (cited directly in the LayerNorm warp kernel).

Confirmed structure, identical across all five operations:

```
<operation>/
├── main.py          # JIT-compiles all kernels, verifies correctness, benchmarks, plots
├── wrapper.cpp       # PyBind11 bindings (same pattern as Chapter 5, one file, all kernels)
└── kernels/
    ├── 0_*.cu        # usually a cuBLAS/PyTorch baseline
    ├── 1_*.cu        # naive
    └── ...           # each subsequent file fixes one named bottleneck
```

Two harness details worth flagging before the kernels themselves: **correctness is checked against PyTorch at a 1e-2 tolerance** (looser than Chapter 5's 1e-4 — because this chapter's GEMM kernels compute in **FP16**, which simply has less precision to work with than FP32), and kernels are **JIT-compiled via `torch.utils.cpp_extension.load()`** on first run rather than pre-built with `setup.py` — a lighter-weight alternative to Chapter 5's ahead-of-time extension build, useful for exactly this kind of "benchmark 5 variants of one op" workflow. The book's own stated headline result: **best performers typically land 1.2–1.6× faster than PyTorch**, with 100% of kernels passing correctness.

---

## 6.1 GEMM: The Full Optimization Ladder (Kernels 0–6)

`gemm/README.md` states its lineage directly: *"classical optimizations... implementing techniques from scratch"*, citing **Simon Boehm's blog** by name — confirming exactly what Chapter 1 told you to expect. This snapshot's `gemm/kernels/` contains the CUDA-core portion of that ladder (0 through 6); the Tensor Core continuation (MMA/WMMA/WGMMA) is a separate topic, next chapter.

**Kernel 1 — Naive**, one thread per output element, FP16 throughout (note the explicit `__hadd`/`__hmul` intrinsics — CUDA's half-precision type needs these rather than plain `+`/`*` in this codebase):

```cuda
typedef __half fp16;

__global__ void gemm_naive(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < M && y < N) {
    fp16 tmp = __float2half(0.0f);
    for (int i = 0; i < K; ++i)
      tmp = __hadd(tmp, __hmul(A[x * K + i], B[i * N + y]));
    C[x * N + y] = tmp;
  }
}
```

The kernel's own doc comment names its problem precisely: *"Each element of B is read M times by different threads"* — zero reuse, exactly Chapter 3's naive GEMM diagnosis.

**Kernel 2 — Global memory coalescing.** No shared memory yet — just a smarter thread-to-output mapping:

```cuda
template <const uint BLOCKSIZE>
__global__ void gemm_gmem_coalesce(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  if (cRow < M && cCol < N) {
    fp16 tmp = __float2half(0.0f);
    for (int i = 0; i < K; ++i)
      tmp = __hadd(tmp, __hmul(A[cRow * K + i], B[i * N + cCol]));
    C[cRow * N + cCol] = tmp;
  }
}
```

The only change: a 1D thread block, with `cCol` derived from `threadIdx.x % BLOCKSIZE` instead of a separate `threadIdx.y`. That single change means consecutive `threadIdx.x` values now map to consecutive `cCol` values — consecutive threads read consecutive columns of `B` — restoring the coalescing Chapter 2's `tensor_add_3d` first taught you to reason about, with zero algorithmic change to *what's* being computed.

**Kernel 3 — Shared-memory blocking**, the real fix for GEMM's data-reuse problem (Chapter 3 §3.3.2):

```cuda
template <const int BLOCKSIZE>
__global__ void gemm_smem_blocking(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint cRow = blockIdx.x, cCol = blockIdx.y;
  __shared__ fp16 As[BLOCKSIZE * BLOCKSIZE];
  __shared__ fp16 Bs[BLOCKSIZE * BLOCKSIZE];
  const uint threadCol = threadIdx.x % BLOCKSIZE, threadRow = threadIdx.x / BLOCKSIZE;
  A += cRow * BLOCKSIZE * K; B += cCol * BLOCKSIZE; C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  fp16 tmp = __float2half(0.0f);
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];   // each thread loads ONE element
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();                                                        // wait for the whole tile to land
    A += BLOCKSIZE; B += BLOCKSIZE * N;
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
      tmp = __hadd(tmp, __hmul(As[threadRow * BLOCKSIZE + dotIdx], Bs[dotIdx * BLOCKSIZE + threadCol]));
    __syncthreads();                                                        // wait before overwriting the tile
  }
  C[threadRow * N + threadCol] = tmp;
}
```

The two `__syncthreads()` calls are load-bearing, not defensive: the first ensures every thread's tile-load has landed in shared memory before *any* thread starts computing with it (otherwise you'd read garbage or half-written data); the second ensures every thread is *done* computing with the current tile before any thread starts overwriting it with the next K-tile's data. Miss either one and you get a race condition, not a compile error — exactly the kind of bug Chapter 5 §5.6's debugging methodology is built to catch. The kernel's own doc comment states the payoff directly: global memory accesses drop from **O(K) per thread to O(K/BLOCKSIZE)**, because each element loaded into shared memory gets reused `BLOCKSIZE` times before being evicted.

**Kernel 4 — 1D block tiling (register blocking).** Shared memory fixed the *block's* reuse; this fixes each *thread's*: instead of one output element, each thread now owns `TM=8` output elements, held in a register array:

```cuda
fp16 threadResults[TM];  // one thread, TM accumulators, all live in registers
// ...
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  fp16 tmpB = Bs[dotIdx * BN + threadCol];               // load ONE B value...
  for (uint resIdx = 0; resIdx < TM; ++resIdx)            // ...reuse it TM times
    threadResults[resIdx] = __hadd(threadResults[resIdx],
                                    __hmul(As[(threadRow * TM + resIdx) * BK + dotIdx], tmpB));
}
```

The comment nails the win: *"each element of B loaded once, reused TM times (once per output row)"* — arithmetic intensity goes up again, this time by reuse *within a single thread's registers*, the fastest memory tier in Chapter 1's hierarchy.

**Kernel 5 — 2D block tiling.** The natural extension: each thread now computes a `TM×TN` tile via an explicit outer product, reusing *both* a loaded `A` value and a loaded `B` value multiple times each:

```cuda
fp16 regM[TM], regN[TN];
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  for (uint i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
  for (uint i = 0; i < TN; ++i) regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
      threadResults[resIdxM * TN + resIdxN] = __hadd(threadResults[resIdxM * TN + resIdxN],
                                                       __hmul(regM[resIdxM], regN[resIdxN]));
}
```

`TM` values of `A` and `TN` values of `B`, loaded once, produce `TM×TN` multiply-adds — reuse squared relative to kernel 4.

**Kernel 6 — Vectorized memory access.** The last rung doesn't touch the *math* at all — it changes how bytes move. Loads/stores use a 4-element-wide vector type (`int2`, reinterpreted as 4×FP16 = 8 bytes) instead of one scalar `fp16` at a time, with an alignment check and scalar fallback:

```cuda
// (paraphrased structure — real file checks pointer alignment before vectorizing)
if (/* gmemSrc and smemDst are both 8-byte aligned */)
  *reinterpret_cast<VecType *>(smemDst) = *reinterpret_cast<const VecType *>(gmemSrc);
else
  /* fall back to a scalar loop */
```

Four elements per memory instruction instead of one means **4× fewer memory instructions** for the same number of bytes moved — directly attacking instruction-issue overhead rather than bandwidth itself, the last lever available once the access pattern and reuse are already as good as kernels 2–5 can make them.

Put the six rungs next to Chapter 1 §1.6.2's abstract "optimization layers" list and they match exactly: coalescing (K2) → shared-memory reuse (K3) → register tiling (K4–K5) → vectorization (K6). This is that list, as real, benchmarked code.

**Deep dive: the tile-size progression, converted into arithmetic intensity, using the kernels' own real constants.** Chapter 3 §3.3.2 derived naive GEMM's actual arithmetic intensity as 0.25 FLOP/byte and its *ideal* (perfect-reuse) intensity as 64.0 FLOP/byte for one specific problem size. There's a general formula connecting shared-memory tile size directly to arithmetic intensity, independent of the overall matrix size: for a GEMM tiled into `BM×BN`-sized output blocks, each block's full `K`-depth slice of `A` and `B` is read from global memory exactly once, giving

```
Arithmetic intensity ≈ 0.5 / (1/BM + 1/BN)
```

Plugging in this ladder's own confirmed tile constants (pulled directly from each kernel's launch function):

| Kernel | BM × BN | Arithmetic intensity |
|---|---|---|
| Naive (no tiling ⇒ BM=BN=1) | 1×1 | **0.25** FLOP/byte — matches Chapter 3's naive figure exactly |
| K3 (shared-memory blocking, `BLOCKSIZE=32`) | 32×32 | **8.0** FLOP/byte |
| K4/K5 (block tiling, `BM=BN=64`) | 64×64 | **16.0** FLOP/byte |
| K6 (vectorized, `BM=BN=128`) | 128×128 | **32.0** FLOP/byte |
| Ideal (whole matrix as one tile) | ∞ | **64.0** FLOP/byte — matches Chapter 3's ideal ceiling exactly |

Every rung of this ladder (all benchmarked at this chapter's confirmed `M=N=K=4096`, large enough that the tile sizes above are a small fraction of the matrix and the approximation holds tightly) is a single, larger point on the *exact same curve* Chapter 3 only had two points on. Doubling the tile dimension exactly doubles the arithmetic intensity — the real reason kernel 6 chooses `BM=BN=128` "for better memory reuse" (its own comment) instead of some arbitrary larger number: it's the next natural doubling in a formula-governed progression, not a guess. Note this figure only concerns *global*-memory traffic — the register-level `TM`/`TN` reuse kernels 4–6 add on top affects shared-memory and register traffic, not this number, so it stacks with rather than duplicates the tile-size effect above.

## 6.2 Softmax: Fixing Chapter 4's Anti-Pattern (Kernels 0–4)

Kernel 1, **online softmax**, first reduces three passes to two using a classic rescaling identity — but it's still one thread per row:

```cuda
__global__ void softmax_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M) {
        float m = -INFINITY, L = 0.0f;
        for (int col = 0; col < N; col++) {
            float curr = matd[row * N + col];
            if (curr > m) { L = L * expf(m - curr); m = curr; }   // rescale existing sum when max changes
            L += expf(curr - m);
        }
        for (int col = 0; col < N; col++)
            resd[row * N + col] = expf(matd[row * N + col] - m) / L;
    }
}
```

The rescaling trick — `exp(x - m_new) = exp(x - m_old) · exp(m_old - m_new)` — lets you update a running sum *without* already knowing the final max, collapsing the max-pass and sum-pass into one. This exact identity, generalized across tiles instead of just across one row, is the core idea Flash Attention (Part 7) builds its entire fused kernel around.

**Kernel 3, warp-shuffle**, is the actual fix for Chapter 4's `softmax_kernel<<<batch_size, 1>>>`:

```cuda
__global__ void softmax_kernel_3(float* xd, float* resd, int M, int N) {
    __shared__ float smem[1024];
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= M) return;

    float local_max = -INFINITY, local_norm = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {          // <-- the fix: N spread across blockDim.x threads
        float x = xd[row * N + i];
        if (x > local_max) { local_norm *= expf(local_max - x); local_max = x; }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    float val = local_max;
    for (int offset = 16; offset > 0; offset /= 2)        // warp-level max reduction
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    if (blockDim.x > 32) {                                 // if more than one warp, combine across warps too
        if (tid % 32 == 0) smem[tid / 32] = val;
        __syncthreads();
        if (tid < 32) {
            val = (tid < (blockDim.x + 31) / 32) ? smem[tid] : -INFINITY;
            for (int offset = 16; offset > 0; offset /= 2)
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    // (an equivalent second reduction combines local_norm the same way)
}
```

Compare the loop header to Chapter 4's version directly: instead of one thread scanning all `N` columns alone, `for (int i = tid; i < N; i += blockDim.x)` spreads those `N` columns across every thread in the block, each accumulating its own **local** online max/sum. Those per-thread partial results then get combined via `__shfl_down_sync` — a warp-level primitive that lets threads within a warp exchange register values directly, with no shared memory and no explicit synchronization needed *within* the warp (all 32 lanes execute in lockstep by hardware guarantee). For blocks larger than one warp, a second, smaller reduction stage combines the per-warp results through shared memory. This is precisely the two-pass, cooperative-reduction kernel Chapter 4 Exercise 1 and Chapter 3 Exercise 4 both asked you to design — here it is, in the book's real, working form.

**Deep dive: the actual serial-work reduction, at this benchmark's largest confirmed row width.** This chapter's own sweep tests softmax up to `(M, N) = (256, 8192)`. Under Chapter 4's single-thread-per-row kernel, one thread does roughly `3 × 8192 ≈ 24,576` sequential operations to fully process its row (a max pass, an exp+sum pass, a normalize pass). Under kernel 3, with a typical 256-thread block, that same 8192-wide row gets split so each thread only walks `8192 / 256 = 32` elements per pass — **≈96 sequential operations per thread, a 256× reduction** — with the combination cost being just a handful of `__shfl_down_sync` steps (`log₂(32) = 5` steps to reduce fully within one warp, plus one small additional stage if the block spans more than one warp). Trading 24,576 serial steps for 96 serial steps plus ~5 shuffle steps is the whole story of why this fix matters more, not less, as rows get wider — exactly the direction real vocabulary-sized softmaxes (Chapter 3's closing note) push you.

## 6.3 LayerNorm: The Warp-Cooperative Version (Kernels 0–2)

Kernel 2's header credits its source directly: *"Based on `llm.c`/`llmc/layernorm.cuh` kernel3"* — Andrej Karpathy's minimal GPT-2 training repo. One **warp** (not one block, not one thread) handles one row:

```cuda
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void layernorm_kernel_2(float* out, float* mean, float* rstd, const float* inp,
                                     const float* weight, const float* bias, int N, int C) {
    int lane_id = threadIdx.x % 32, warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    int idx = blockIdx.x * num_warps + warp_id;   // this WARP handles row `idx`
    if (idx >= N) return;
    const float* x = inp + idx * C;
    // ...each of the 32 lanes strides through C, accumulates a partial sum,
    // warpReduceSum() combines the 32 partial sums into the row's mean and variance...
}
```

Assigning **one warp per row** (rather than one thread, or one whole multi-warp block) is a deliberate granularity choice: it's fine-grained enough that many rows can be processed concurrently across a block's several warps, while still giving each row 32 cooperating lanes for its reduction — no cross-warp shared-memory synchronization needed at all for a single row's mean/variance, since a warp's 32 threads already execute in lockstep.

**Deep dive: why warp-per-row beats block-per-row here, concretely.** This chapter's own sweep goes up to `C = 4096` (hidden dimension). Under this kernel's actual warp-per-row design, each of the 32 lanes strides through `4096 / 32 = 128` elements. Compare that to the alternative this kernel *didn't* choose — a full 256-thread block per row — which would give each thread only `4096 / 256 = 16` elements, finer-grained per-row parallelism, but at a real cost: combining 256 threads' partial sums needs a shared-memory reduction across 8 separate warps, with explicit `__syncthreads()` barriers, for *every single row*. The warp-per-row design accepts coarser per-row parallelism (128 elements/lane instead of 16) in exchange for needing **zero cross-warp synchronization** to finish a row — and, more importantly, lets a single 256-thread block process **8 independent rows simultaneously** (one per warp) rather than one row at a time. With `N` up to 256 rows in this chapter's own largest test case, that's the difference between needing `256` separate block launches (one row each, if you'd gone with a whole-block-per-row design) versus just `256/8 = 32` block launches to cover the same batch — fewer, more fully-occupied blocks, which is exactly why `llm.c`'s own kernel3 (the source this file credits) makes this specific tradeoff.

## 6.4 GEMV: Fixing Chapter 5's Non-Cooperative Kernel (Kernels 0–4)

Kernel 2, **coalesced warp**, is the direct fix for Chapter 5's `gemv_kernel`, which used `threadsPerBlock.x = 1` — one thread doing an entire row's dot product alone:

```cuda
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
// Each block = one warp (32 threads) = one output row.
// Each thread strides through the row's N columns (stride = blockDim.x = 32),
// accumulating a partial dot-product sum, then warpReduceSum() combines all 32
// partial sums into the row's final output value.
```

Same shape of fix as softmax's kernel 3: replace "one thread does all the work for this row" with "32 threads split the row's work, then combine via warp shuffle." Chapter 5 Exercise 4 asked you to design exactly this — the book's real version confirms a warp-per-row, shuffle-reduction design is the standard answer.

**Deep dive: a concrete bandwidth target for this exact benchmark, on your own hardware.** This chapter's own sweep tops out at `(M, N) = (256, 8192)`. Since GEMV's cost is almost entirely reading the `M×N` matrix once (Chapter 5 §5.4's deep dive: arithmetic intensity ≈ 0.5 FLOP/byte, so this is squarely a bandwidth problem, not a compute one), the ideal, bandwidth-bound time is just total bytes ÷ peak bandwidth:

```
Bytes ≈ M×N×4 (matrix) + N×4 (input vector) + M×4 (output) = 256×8192×4 + 8192×4 + 256×4 ≈ 8,422,400 bytes (≈8.03 MiB)

RTX 3090 (936 GB/s):  8,422,400 / 936e9 ≈ 9.0 microseconds
Tesla T4  (320 GB/s):  8,422,400 / 320e9 ≈ 26.3 microseconds
```

Those two numbers are your real, computed targets for Chapter 5 Exercise 4's "measure achieved bandwidth against theoretical peak" ask — if your own timed run of this chapter's warp-cooperative kernel at this exact shape lands within a small multiple of 9.0µs (3090) or 26.3µs (T4), it's genuinely close to the hardware's actual ceiling for this operation; a large gap tells you there's real headroom left, likely in launch overhead or in the specific access pattern, worth chasing down with Chapter 10's profiling tools.

## 6.5 Top-K: A Real (Partial) Fix for Chapter 5's MoE Bug (Kernels 0–2)

Kernel 0 here is the same naive, single-thread-per-row insertion sort from Chapter 5's `topk_kernel` — same strict-inequality tie-break, same bug class. Kernel 2, **warp-parallel**, replaces the serial insertion sort with a proper parallel reduction, carrying the *index* alongside the *value* through every reduction step so the selected position survives:

```cuda
__device__ __forceinline__ ValueIndex warp_reduce_max_with_idx(ValueIndex val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        ValueIndex other;
        other.value = __shfl_down_sync(0xffffffff, val.value, offset);
        other.index = __shfl_down_sync(0xffffffff, val.index, offset);
        if (other.value > val.value) val = other;    // still a strict >, note for Exercise 4 below
    }
    return val;
}
// block_reduce_max_with_idx() extends this across multiple warps via shared memory,
// the same warp-then-block pattern as softmax kernel 3 and layernorm kernel 2.
```

This is a real, substantial improvement over Chapter 5's version — finding the row's max is now a genuine `O(log n)`-depth parallel reduction across all threads, not an `O(n·k)` serial scan on one thread — but look closely at the comparison: it's still `other.value > val.value`, the same **strict, non-deterministic-under-ties** inequality that caused the §5.5 MoE divergence bug. Better parallelism doesn't automatically mean better numerical robustness; those are two separate problems, and this kernel only solved the first one. Fully closing the loop — deterministic tie-breaking *and* proper parallelism together — is Exercise 4 below.

**Deep dive: how close the warp-parallel version gets to the theoretical 32× ceiling.** This chapter's confirmed benchmark sweep (`main.py`'s own `configs` list) is `[(256,8), (512,16), (1024,32), (2048,64), (4096,128)]` — five `(N, K)` pairs, largest `N=4096, K=128`. At that size, Chapter 5's naive kernel does up to `N×K = 4096×128 = 524,288` sequential comparison/shift operations on a *single* thread (worst case: every element requires scanning all `K` current slots before either inserting or rejecting it). Kernel 2 spreads `N` across a 32-lane warp — each thread handles `N/32 = 128` elements, building its own local top-`K` candidates at a cost of up to `(N/32)×K = 128×128 = 16,384` operations, but with all 32 threads doing this **simultaneously**, so the wall-clock-equivalent cost is just those 16,384 steps, not 32× that. Merging the 32 threads' local candidate lists into one final top-128 needs roughly `K` rounds of warp-level max-extraction, each `log₂(32) = 5` shuffle steps deep — about `128 × 5 = 640` additional steps.

```
Naive (single thread):        524,288 sequential steps
Warp-parallel (32 lanes):      16,384 (local build) + 640 (merge) ≈ 17,024 steps
Observed reduction:            524,288 / 17,024 ≈ 30.8×
```

The theoretical ceiling from spreading `N` across exactly 32 lanes, with zero merge overhead, would be exactly 32×. The real design lands at **≈30.8×** — within about 4% of that ceiling — meaning the merge stage's cost is genuinely small next to the local-build savings, and this kernel is capturing nearly all of the parallelism this specific axis (splitting `N` across a warp) has to offer. That's a materially different, and much better, story than kernel 2's *tie-breaking* behavior, which — as the paragraph above shows — didn't improve at all along with the speed.

## 6.6 Measuring the Speedups Yourself (with `load_inline`)

This chapter has made a lot of speedup claims backed by the book's own measured tables. This section lets you generate your own numbers, on your own hardware, for all five operations — naive vs. optimized, side by side.

**A provenance note covering all five pairs below, rather than repeating it five times:** the confirmed-verbatim pieces are `warpReduceSum` (§6.3), `warp_reduce_max_with_idx` (§6.5), and the online-softmax rescaling logic (§6.2). Where this chapter quoted a kernel's *structure* in prose rather than its full body (softmax kernel 3's block-level combination stage, layernorm kernel 2's full body, GEMV kernel 2's full body, top-k's block-level extension) — because the original text summarized rather than fully reproduced those sections — I've written **simplified, single-warp completions** below (exactly 32 threads per row, sidestepping the multi-warp block-combination stage the real kernels need for larger thread counts) that implement the *same idea* correctly and unambiguously, rather than risk reconstructing an intricate multi-stage reduction from a partial description. GEMM's tiled kernel is adapted from the book's confirmed FP16 kernel 3 to plain FP32, for a simpler side-by-side comparison against PyTorch.

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ================= GEMM: naive vs. shared-memory tiled (FP32 adaptation of §6.1) =================
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[row*K+k] * B[k*N+col];
        C[row*N+col] = sum;
    }
}
#define TILE 16
__global__ void gemm_smem_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K) ? A[row*K + t*TILE + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t*TILE+threadIdx.y < K) ? B[(t*TILE+threadIdx.y)*N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; ++i) sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = sum;
}

// ================= Softmax: naive (one thread/row) vs. single-warp (§6.2's idea) =================
__global__ void softmax_naive_kernel(const float* in, float* out, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float m = -INFINITY;
        for (int i = 0; i < N; ++i) m = fmaxf(m, in[row*N+i]);
        float s = 0.0f;
        for (int i = 0; i < N; ++i) s += expf(in[row*N+i]-m);
        for (int i = 0; i < N; ++i) out[row*N+i] = expf(in[row*N+i]-m) / s;
    }
}
__global__ void softmax_warp32_kernel(const float* in, float* out, int M, int N) {
    int row = blockIdx.x, lane = threadIdx.x;               // exactly 32 threads (one warp) per row
    float local_max = -INFINITY, local_norm = 0.0f;
    for (int i = lane; i < N; i += 32) {                    // online softmax, §6.2's rescaling identity
        float x = in[row*N+i];
        if (x > local_max) { local_norm *= expf(local_max - x); local_max = x; }
        local_norm += expf(x - local_max);
    }
    float max_val = local_max;
    for (int o = 16; o > 0; o /= 2) max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, o));
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    local_norm *= expf(local_max - max_val);
    float sum_val = local_norm;
    for (int o = 16; o > 0; o /= 2) sum_val += __shfl_down_sync(0xffffffff, sum_val, o);
    sum_val = __shfl_sync(0xffffffff, sum_val, 0);
    for (int i = lane; i < N; i += 32) out[row*N+i] = expf(in[row*N+i]-max_val) / sum_val;
}

// ================= LayerNorm: naive (one thread/row) vs. single-warp (§6.3's idea) =================
__device__ __forceinline__ float warpReduceSum(float val) {   // confirmed verbatim, §6.3
    for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__global__ void layernorm_naive_kernel(const float* x, float* out, int M, int N, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float mean = 0.0f;
        for (int i = 0; i < N; ++i) mean += x[row*N+i];
        mean /= N;
        float var = 0.0f;
        for (int i = 0; i < N; ++i) { float d = x[row*N+i]-mean; var += d*d; }
        var /= N;
        float rstd = rsqrtf(var + eps);
        for (int i = 0; i < N; ++i) out[row*N+i] = (x[row*N+i]-mean) * rstd;
    }
}
__global__ void layernorm_warp_kernel(const float* x, float* out, int M, int N, float eps) {
    int row = blockIdx.x, lane = threadIdx.x;               // exactly 32 threads (one warp) per row
    float sum = 0.0f;
    for (int i = lane; i < N; i += 32) sum += x[row*N+i];
    sum = __shfl_sync(0xffffffff, warpReduceSum(sum), 0);
    float mean = sum / N;
    float var_sum = 0.0f;
    for (int i = lane; i < N; i += 32) { float d = x[row*N+i]-mean; var_sum += d*d; }
    var_sum = __shfl_sync(0xffffffff, warpReduceSum(var_sum), 0);
    float rstd = rsqrtf(var_sum/N + eps);
    for (int i = lane; i < N; i += 32) out[row*N+i] = (x[row*N+i]-mean) * rstd;
}

// ================= GEMV: naive (one thread/row) vs. single-warp (§6.4's idea) =================
__global__ void gemv_naive_kernel(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) sum += A[row*N+col] * x[col];
        y[row] = sum;
    }
}
__global__ void gemv_warp_kernel(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x, lane = threadIdx.x;               // one warp (32 threads) per output row
    float sum = 0.0f;
    for (int col = lane; col < N; col += 32) sum += A[row*N+col] * x[col];
    sum = warpReduceSum(sum);
    if (lane == 0) y[row] = sum;
}

// ================= Top-K: naive insertion sort (§5.4) vs. single-warp reduction (§6.5's idea) =================
__global__ void topk_naive_kernel(const float* input, float* values, int* indices, int batch_size, int n, int k) {
    int b = blockIdx.x;
    if (b < batch_size) {
        const float* row = input + b * n;
        float* vr = values + b * k; int* ir = indices + b * k;
        for (int i = 0; i < k; ++i) { vr[i] = -INFINITY; ir[i] = -1; }
        for (int i = 0; i < n; ++i) {
            float val = row[i];
            for (int j = 0; j < k; ++j) {
                if (val > vr[j]) {
                    for (int m = k-1; m > j; --m) { vr[m]=vr[m-1]; ir[m]=ir[m-1]; }
                    vr[j] = val; ir[j] = i; break;
                }
            }
        }
    }
}
struct ValueIndex { float value; int index; };
__device__ __forceinline__ ValueIndex warp_reduce_max_with_idx(ValueIndex val) {   // confirmed verbatim, §6.5
    for (int offset = 16; offset > 0; offset >>= 1) {
        ValueIndex other;
        other.value = __shfl_down_sync(0xffffffff, val.value, offset);
        other.index = __shfl_down_sync(0xffffffff, val.index, offset);
        if (other.value > val.value) val = other;
    }
    return val;
}
__global__ void topk_warp_kernel(const float* input, float* values, int* indices, int batch_size, int n, int k) {
    int b = blockIdx.x, lane = threadIdx.x;                 // one warp (32 threads) per row
    if (b >= batch_size) return;
    const float* row = input + b * n;
    extern __shared__ float taken[];
    for (int i = lane; i < n; i += 32) taken[i] = 0.0f;
    __syncthreads();
    for (int r = 0; r < k; ++r) {
        ValueIndex best = {-INFINITY, -1};
        for (int i = lane; i < n; i += 32)
            if (taken[i] == 0.0f && row[i] > best.value) { best.value = row[i]; best.index = i; }
        best = warp_reduce_max_with_idx(best);
        best.value = __shfl_sync(0xffffffff, best.value, 0);
        best.index = __shfl_sync(0xffffffff, best.index, 0);
        if (lane == 0) { values[b*k+r] = best.value; indices[b*k+r] = best.index; taken[best.index] = 1.0f; }
        __syncthreads();
    }
}

// ================= Launchers =================
torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0),K=A.size(1),N=B.size(1); auto C=torch::empty({M,N},A.options());
    dim3 t(16,16), g((N+15)/16,(M+15)/16);
    gemm_naive_kernel<<<g,t>>>(A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),M,N,K); return C;
}
torch::Tensor gemm_smem(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0),K=A.size(1),N=B.size(1); auto C=torch::empty({M,N},A.options());
    dim3 t(TILE,TILE), g((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    gemm_smem_kernel<<<g,t>>>(A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),M,N,K); return C;
}
torch::Tensor softmax_naive(torch::Tensor in) {
    int M=in.size(0),N=in.size(1); auto out=torch::empty_like(in);
    int threads=256, blocks=(M+threads-1)/threads;
    softmax_naive_kernel<<<blocks,threads>>>(in.data_ptr<float>(),out.data_ptr<float>(),M,N); return out;
}
torch::Tensor softmax_warp(torch::Tensor in) {
    int M=in.size(0),N=in.size(1); auto out=torch::empty_like(in);
    softmax_warp32_kernel<<<M,32>>>(in.data_ptr<float>(),out.data_ptr<float>(),M,N); return out;
}
torch::Tensor layernorm_naive(torch::Tensor x, double eps) {
    int M=x.size(0),N=x.size(1); auto out=torch::empty_like(x);
    int threads=256, blocks=(M+threads-1)/threads;
    layernorm_naive_kernel<<<blocks,threads>>>(x.data_ptr<float>(),out.data_ptr<float>(),M,N,(float)eps); return out;
}
torch::Tensor layernorm_warp(torch::Tensor x, double eps) {
    int M=x.size(0),N=x.size(1); auto out=torch::empty_like(x);
    layernorm_warp_kernel<<<M,32>>>(x.data_ptr<float>(),out.data_ptr<float>(),M,N,(float)eps); return out;
}
torch::Tensor gemv_naive(torch::Tensor A, torch::Tensor x) {
    int M=A.size(0),N=A.size(1); auto y=torch::empty({M},A.options());
    int threads=256, blocks=(M+threads-1)/threads;
    gemv_naive_kernel<<<blocks,threads>>>(A.data_ptr<float>(),x.data_ptr<float>(),y.data_ptr<float>(),M,N); return y;
}
torch::Tensor gemv_warp(torch::Tensor A, torch::Tensor x) {
    int M=A.size(0),N=A.size(1); auto y=torch::empty({M},A.options());
    gemv_warp_kernel<<<M,32>>>(A.data_ptr<float>(),x.data_ptr<float>(),y.data_ptr<float>(),M,N); return y;
}
std::vector<torch::Tensor> topk_naive(torch::Tensor in, int64_t k) {
    int B=in.size(0),n=in.size(1);
    auto v=torch::empty({B,k},in.options()), idx=torch::empty({B,k},in.options().dtype(torch::kInt32));
    topk_naive_kernel<<<B,1>>>(in.data_ptr<float>(),v.data_ptr<float>(),idx.data_ptr<int>(),B,n,(int)k);
    return {v, idx};
}
std::vector<torch::Tensor> topk_warp(torch::Tensor in, int64_t k) {
    int B=in.size(0),n=in.size(1);
    auto v=torch::empty({B,k},in.options()), idx=torch::empty({B,k},in.options().dtype(torch::kInt32));
    topk_warp_kernel<<<B,32,n*sizeof(float)>>>(in.data_ptr<float>(),v.data_ptr<float>(),idx.data_ptr<int>(),B,n,(int)k);
    return {v, idx};
}
"""

cpp_source = r"""
torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_smem(torch::Tensor A, torch::Tensor B);
torch::Tensor softmax_naive(torch::Tensor in);
torch::Tensor softmax_warp(torch::Tensor in);
torch::Tensor layernorm_naive(torch::Tensor x, double eps);
torch::Tensor layernorm_warp(torch::Tensor x, double eps);
torch::Tensor gemv_naive(torch::Tensor A, torch::Tensor x);
torch::Tensor gemv_warp(torch::Tensor A, torch::Tensor x);
std::vector<torch::Tensor> topk_naive(torch::Tensor in, int64_t k);
std::vector<torch::Tensor> topk_warp(torch::Tensor in, int64_t k);
"""

ch6 = load_inline(
    name="ch6_optim_kernels", cpp_sources=cpp_source, cuda_sources=cuda_source,
    functions=["gemm_naive", "gemm_smem", "softmax_naive", "softmax_warp",
               "layernorm_naive", "layernorm_warp", "gemv_naive", "gemv_warp",
               "topk_naive", "topk_warp"],
    verbose=True,
)
```

A single benchmarking harness — CUDA events (Chapter 2 §2.6), correctness first, then timing — run against all five pairs:

```python
def bench(fn, *args, iters=50):
    for _ in range(5): fn(*args)                     # warmup
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters            # ms per call

def compare(name, naive_fn, opt_fn, *args, atol=1e-3):
    out_naive = naive_fn(*args); out_opt = opt_fn(*args)
    diff = (out_naive - out_opt).abs().max().item()
    t_naive, t_opt = bench(naive_fn, *args), bench(opt_fn, *args)
    print(f"{name:10s} match={diff:.1e}  naive={t_naive:.3f}ms  optimized={t_opt:.3f}ms  speedup={t_naive/t_opt:.1f}x")

device = "cuda"
torch.manual_seed(0)

A, B = torch.randn(1024, 512, device=device), torch.randn(512, 1024, device=device)
compare("gemm", ch6.gemm_naive, ch6.gemm_smem, A, B)

S = torch.randn(256, 4096, device=device)
compare("softmax", ch6.softmax_naive, ch6.softmax_warp, S)

X = torch.randn(256, 4096, device=device)
compare("layernorm", lambda x: ch6.layernorm_naive(x, 1e-5), lambda x: ch6.layernorm_warp(x, 1e-5), X)

Amat, xvec = torch.randn(4096, 4096, device=device), torch.randn(4096, device=device)
compare("gemv", ch6.gemv_naive, ch6.gemv_warp, Amat, xvec)

R = torch.randn(256, 4096, device=device)
def tk_naive(r): return ch6.topk_naive(r, 8)[0]
def tk_warp(r):  return ch6.topk_warp(r, 8)[0]
compare("topk", tk_naive, tk_warp, R)
```

Every row should show `match` near zero and a real `speedup` greater than 1×. Note `layernorm` here omits the learned affine (gamma/beta) parameters real `nn.LayerNorm` applies — this demo compares two raw normalization kernels against each other, not against PyTorch's own layer, so that's an intentional simplification, not a discrepancy to chase. Your own numbers won't match the book's H100 table exactly — these are simplified, single-warp, FP32 reproductions on whatever GPU you're running, not the book's tuned FP16/multi-warp kernels — but the *shape* of the result (naive loses, cooperative reduction wins, by a real and repeatable margin) should hold on your RTX 3090 and T4 alike.

---

## Hands-On Lab

```bash
cd book.cu/4_optim
for op in gemm softmax layernorm gemv topK; do
  echo "=== $op ==="
  cd $op && python main.py && cd ..
done
```

Each run JIT-compiles every kernel, prints a correctness table (all should read `PASS`), then latency/throughput/speedup summary tables, and writes a `<operation>_performance.png` plot.

1. **Recompute the full GEMM speedup, kernel 1 → kernel 6.** From the printed GFLOPS table, compute the ratio between the naive kernel and the vectorized kernel at your largest benchmarked size. Compare that ratio to Chapter 3's own naive-GEMM measurement (Chapter 3, Hands-On Lab step 2) — same starting point, now with the full ladder applied.
2. **Compare the softmax fix quantitatively.** Time kernel 0 (or 1) against kernel 3 (warp-shuffle) at a row width of at least 4096 — wide enough that Chapter 4's single-thread-per-row version has real work to parallelize away.
3. **Run both hardware profiles.** Execute all five benchmarks on your RTX 3090 and your T4. Note where the speedup-vs-PyTorch curve differs between them — fewer SMs and a smaller register file on the T4 can shift which rung of the ladder gives the biggest marginal win.
4. **Deliberately break kernel 3's synchronization.** Comment out the *second* `__syncthreads()` in GEMM's `gemm_smem_blocking` (kernel 3) and re-run. The correctness check should now report a failure — read the printed `max_diff`/`mean_diff` and confirm it's inconsistent from run to run (a signature of a genuine race condition, not a deterministic bug).

## Exercises

1. **Extend the GEMM ladder yourself.** Write a 7th kernel that increases `BK` (the K-tile size) in kernel 6, retune `BM`/`BN`/`TM`/`TN` to keep shared-memory usage within your GPU's per-SM limit (Chapter 1's table), and benchmark against kernel 6.
2. **Explain softmax kernel 3's synchronization points.** Why does `__syncthreads()` appear right after the per-thread accumulation loop but *before* any warp-shuffle call? What could go wrong if it were removed?
3. **Compare your Chapter 5 Exercise 4 design to the real GEMV kernel 2.** You were asked to design a cooperative-reduction GEMV before seeing this chapter's answer — how close did your design come to `warpReduceSum` plus a warp-per-row launch config?
4. **Close the MoE bug properly.** Extend top-k kernel 2's `warp_reduce_max_with_idx` to break ties deterministically (e.g., prefer the lower index when `other.value == val.value`, not just when it's strictly greater), then re-run Chapter 5 §5.5's full ablation (custom softmax + your fixed, *parallel* top-k) and confirm both correctness *and* near-zero MoE token divergence simultaneously.
5. **Read one `main.py` closely** and identify exactly how it decides a kernel "passes" correctness (the 1e-2 tolerance check) — then explain, using Chapter 1's numerical-precision discussion, why FP16 GEMM needs a looser tolerance than Chapter 5's FP32 ops did.

---

**Next:** Chapter 7 — Tensor Core Programming: WMMA & Hopper's WGMMA (Part 6). Every kernel in this chapter still computed its multiply-adds with ordinary CUDA cores. The `5_tensor_cores` folder picks up exactly where GEMM kernel 6 left off, handing that same multiply-accumulate work to dedicated hardware.
