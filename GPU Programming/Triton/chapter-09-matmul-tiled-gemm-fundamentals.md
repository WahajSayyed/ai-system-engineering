# Chapter 9 — Matrix Multiplication I: Tiled GEMM Fundamentals

## 9.1 A Different Kind of Performance Problem

Chapter 7 was about a bandwidth-bound operation: softmax does almost no arithmetic per byte moved, so the whole game was touching global memory as few times as possible. Matrix multiplication is the opposite regime, and it's worth being precise about why before writing a single line of kernel code.

For `C = A @ B` with `A` of shape `(M, K)` and `B` of shape `(K, N)`, the computation performs `2 * M * N * K` floating-point operations (a multiply and an add per output element per reduction step), while the data involved is only `O(M*K + K*N + M*N)` elements. As `M`, `N`, `K` grow, the FLOP count grows with the *product* of all three dimensions while the memory footprint only grows with their *pairwise sums* — arithmetic intensity (FLOPs per byte) increases with matrix size, and for large matrices, matmul becomes **compute-bound**: the ceiling on performance is the GPU's raw arithmetic throughput (specifically, its tensor cores), not HBM bandwidth.

This changes what "optimizing" even means here. Chapter 7's goal was *touch each byte as few times as possible*. This chapter's goal is *reuse each byte you've already paid to load as many times as possible* — load a tile of `A` and a tile of `B` into on-chip memory once, and get as many multiply-accumulate operations out of that one load as you can before moving on. This is exactly the tiling/blocking principle from classical HPC (the same idea behind cache-blocked BLAS implementations on CPUs), applied through Triton's tile-level programming model.

## 9.2 The Blocked Algorithm, Conceptually

Each program instance is responsible for computing one `(BLOCK_SIZE_M, BLOCK_SIZE_N)` tile of the output `C`. It does this by walking across the shared `K` dimension in chunks of `BLOCK_SIZE_K`: at each step, it loads a `(BLOCK_SIZE_M, BLOCK_SIZE_K)` tile of `A` and a `(BLOCK_SIZE_K, BLOCK_SIZE_N)` tile of `B`, multiplies them, and adds the result into a running accumulator. After walking the entire `K` dimension, the accumulator holds the finished output tile, which is written to `C` once.

Every element of the `A` tile loaded at a given step is reused `BLOCK_SIZE_N` times (once per output column in the tile); every element of the `B` tile is reused `BLOCK_SIZE_M` times (once per output row). This reuse — happening entirely from on-chip memory, not by re-reading from HBM — is where a tiled GEMM's performance advantage over a naive triple-loop implementation actually comes from.

## 9.3 A Naive Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_naive(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Wraparound trick: keeps row/col offsets valid even past M/N,
    # so the A/B loads below never need an M- or N-boundary mask.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

A few details worth being deliberate about, since each one is a small design decision, not an arbitrary choice:

**The `% M` / `% N` wraparound trick.** `offs_am`/`offs_bn` are taken modulo `M`/`N` *before* being used to build `a_ptrs`/`b_ptrs`. This guarantees every address computed for the `A`/`B` loads stays in-bounds, even for the last, partially-out-of-range tile along `M` or `N` — at the cost of occasionally reading a row/column that gets *wrapped back* to an already-valid one (redundant, but never wrong, since these particular rows/columns are exactly the ones that will be discarded by `c_mask` at the final store anyway). The payoff: the only mask you need *inside the K-loop* is the one for the `K` boundary (`offs_k[None, :] < k_remaining`), rather than a combined K-*and*-M/N mask recomputed every iteration. This is a genuine simplification, not just cleverness for its own sake — masks that don't need to combine multiple boundary conditions are cheaper to compute, every iteration, for the entire duration of the K-loop.

**`tl.dot(a, b, accumulator)`.** The three-argument form fuses the multiply with the accumulate — `accumulator = a @ b + accumulator` in one call — rather than computing `a @ b` and adding it separately. This maps directly onto the GPU's tensor-core fused-multiply-add instructions; you'll get the full precision/tensor-core treatment in Chapter 10, but for now, recognize this as *the* op that turns tile-level linear algebra into a single hardware instruction rather than a python-level abstraction over many scalar ops.

**`tl.float32` accumulator, even though the output is cast to `float16`.** Every partial product across the entire `K`-loop is summed in full `float32` precision, and only the *final* result is cast down to the output dtype. If you accumulated directly in `float16`, rounding error would compound over potentially thousands of `K`-steps for large matrices, producing a measurably less accurate result. This is a real numerical-stability decision, not a default you can ignore — Chapter 10 goes deeper into precision tradeoffs, but the pattern "accumulate wide, store narrow" is worth internalizing right now.

## 9.4 Why Tile *Order* Matters: L2 Cache Reuse

The naive kernel above computes `pid_m = pid // grid_n`, `pid_n = pid % grid_n` — a plain row-major sweep across output tiles. This is correct, but it leaves real performance on the table, and the reason is about **which tiles are being computed *concurrently*, on different SMs, at the same moment** — not about any single program's own behavior.

Tiles in the same row of `C` all share the same rows of `A`. Tiles in the same column of `C` all share the same columns of `B`. If nearby SMs, running concurrently, are working on output tiles that are also *spatially close* in `C`, then the `A`/`B` data one SM just loaded into the shared L2 cache is likely to still be resident when a neighboring SM wants the same data moments later — a genuine cache hit, saved from a full HBM round-trip. Row-major ordering doesn't guarantee this: it sweeps an entire row of `C` (potentially needing *every* column-tile of `B`) before moving to the next row, and by the time it revisits those same `B` tiles for row two, they may well have been evicted from L2 by everything loaded in between — especially once `B` itself is larger than the L2 cache, which is routine for real workloads.

Triton's own documentation quantifies this with a concrete example: for a matmul where each matrix is 9 tiles by 9 tiles, <cite index="36-1">computing the output in row-major ordering requires loading 90 tiles into on-chip memory to produce the first 9 output tiles, whereas a grouped ordering achieves the same 9 output tiles while loading only 54 tiles</cite> — and <cite index="36-1">this reduction in redundant loads translates to a measured improvement of more than 10% on real hardware, reported as an increase from 220 to 245 TFLOPS on an A100</cite> in Triton's own benchmarks. This is not a marginal micro-optimization; it's one of the largest single wins available in a hand-written matmul kernel, and it costs nothing at the algorithm level — it's purely a change in *which order* programs are assigned to tiles.

## 9.5 Grouped (Swizzled) Tile Ordering

The fix reorders which `(pid_m, pid_n)` pair each program ID maps to, so that programs launched close together in ID (and therefore likely to execute concurrently on nearby SMs) are assigned to *spatially adjacent* output tiles, sharing as much `A`/`B` data as possible. The official formula:

```python
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
# The last group may have fewer than GROUP_SIZE_M rows if num_pid_m doesn't divide evenly.
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

Read this as: instead of sweeping one full row of `C` at a time, group `GROUP_SIZE_M` consecutive rows of tiles together, and within that group, sweep *columns* — so `GROUP_SIZE_M * num_pid_n` consecutive program IDs are all working within the same narrow horizontal band of `C`, sharing `A` tiles across that whole band and reusing `B` tiles across the group before moving to the next band. `group_size_m` (as distinct from the constexpr `GROUP_SIZE_M`) handles the ragged last group correctly when `num_pid_m` isn't an exact multiple of `GROUP_SIZE_M` — without it, the last few rows of tiles would either be miscomputed or left out of the grid entirely.

**A subtlety worth flagging, not glossing over**: you may encounter kernel code online using a slightly different — but *usually* equivalent — expression, `pid_m = first_pid_m + (pid % group_size_m)` (note: `pid % group_size_m`, not `(pid % num_pid_in_group) % group_size_m`). The two agree whenever `group_size_m == GROUP_SIZE_M` (every group except possibly the last), and can disagree specifically in that final, ragged group. If you're debugging a matmul kernel and results are only wrong in the last row-band of tiles, this exact formula is one of the first places to look.

**`tl.swizzle2d`** is a built-in Triton utility that provides a similar group-then-sweep remapping without you hand-deriving the arithmetic above — worth knowing it exists (you'll see it in production kernels, and it appears in GPU MODE's grouped-matmul teaching material), though writing the formula out by hand once, as you're about to in this chapter's exercises, is the better way to actually understand what it's doing before you rely on the convenience wrapper.

## 9.6 Autotuning the Matmul: Bringing Chapter 8 to Bear

Unlike softmax's fairly forgiving performance landscape, the optimal `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, and `GROUP_SIZE_M` for a matmul genuinely depend on the specific `M`, `N`, `K` and the GPU you're running on — exactly the situation Chapter 8 was built for:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                       num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                       num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                       num_stages=4, num_warps=4),
        # ... additional candidate tile-size/pipelining combinations ...
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

This is a representative shape of the search space, not a literal, version-pinned config list to copy verbatim — the exact candidates in the official tutorial change between Triton releases as better defaults are discovered. What should *not* change is the reasoning: `key=['M', 'N', 'K']` because the optimal tile configuration is genuinely shape-dependent, and every `Config` bundles a tile-size choice with `num_stages`/`num_warps`, exactly as Chapter 8 described.

## 9.7 Hands-On

**Exercise 1 — Implement and test the naive kernel.** Build `matmul_kernel_naive` from §9.3 in full. Test against `torch.matmul` with matrix sizes deliberately *not* multiples of your block sizes (e.g., `M=1000, N=1000, K=1000` against `BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32`), confirming both the K-loop masking and the `c_mask` at final store are correct.

**Exercise 2 — Add grouped ordering.** Modify your kernel to use the `pid_m`/`pid_n` formula from §9.5 in place of the naive row-major mapping. Confirm correctness is completely unaffected (same output, different computation order) — a good sanity check that this optimization is purely about scheduling, not semantics.

**Exercise 3 — Measure the L2 effect directly.** Benchmark both orderings (naive vs. grouped) in achieved TFLOP/s across a range of matrix sizes, following the `triton.testing.perf_report` pattern from Chapter 7. You should see the gap between the two orderings *widen* as matrix size grows past your GPU's L2 cache capacity — small matrices may show little difference (everything fits in L2 regardless of order), while large matrices should show the grouped version pulling ahead measurably, consistent with the reasoning in §9.4.

**Exercise 4 — Sweep `GROUP_SIZE_M` itself.** Fix all other tile parameters and vary `GROUP_SIZE_M` (try 1, 4, 8, 16, 32) at a large, L2-stressing matrix size. `GROUP_SIZE_M=1` should behave close to naive row-major (no meaningful grouping); very large values should start to lose their benefit too (a full column-major sweep has its own reuse problems, symmetric to row-major's). Find the regime where the effect is real and report what you observe — this is meant to build intuition, not to find one universally correct number.

**Exercise 5 — Autotune it.** Wrap your grouped-ordering kernel with `@triton.autotune` using a representative config list like §9.6's, keyed on `['M', 'N', 'K']`. Run with `TRITON_PRINT_AUTOTUNING=1` (Chapter 8, §8.3) and confirm a search happens for each new shape you try. Compare autotuned performance against your best hand-picked single configuration from Exercise 4.

**Exercise 6 — Benchmark against `torch.matmul`.** Compare your best autotuned kernel's TFLOP/s against PyTorch's own `torch.matmul` (backed by cuBLAS) across several sizes. Getting within a reasonable margin of cuBLAS with this much less code — and with the ability to fuse in arbitrary custom logic that cuBLAS can't — is the actual point of this exercise, not necessarily beating it outright.

## 9.8 Check Your Understanding

1. Explain, without re-reading §9.1, why matrix multiplication's optimization goal ("maximize reuse of loaded data") is fundamentally different from fused softmax's ("minimize the number of memory round-trips"). What property of the computation causes this difference?
2. Walk through the `% M` wraparound trick in your own words: what would go wrong, specifically, if you removed it and did *not* replace it with an equivalent M/N mask on the `A`/`B` loads?
3. Why does grouped tile ordering improve performance without changing a single arithmetic operation the kernel performs? What, precisely, is different between the row-major and grouped versions?
4. Why is the matmul accumulator kept in `float32` even when the inputs and final output are `float16`? What would you expect to observe if you accumulated directly in `float16` for a very large `K`?

## 9.9 What's Next

You now have a correct, L2-aware, autotuned GEMM — the algorithmic core that essentially every high-performance kernel in this curriculum builds on, directly or by analogy. Chapter 10 goes one level deeper into what `tl.dot` is actually doing: mixed-precision accumulation, tensor-core mapping, and the real accuracy/throughput tradeoffs involved in choosing fp16, bf16, or fp8 inputs — the material this chapter deliberately deferred so the tiling algorithm itself could stay the center of attention.
