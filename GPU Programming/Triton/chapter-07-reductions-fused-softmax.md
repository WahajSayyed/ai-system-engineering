# Chapter 7 — Reductions & Fused Softmax

## 7.1 Why Softmax Is a Memory-Bandwidth Problem, Not a Compute Problem

Before writing any code, it's worth being precise about *what kind* of performance problem softmax actually is, because the entire design of this chapter's kernel follows from the answer.

Row-wise softmax over a matrix `x` of shape `(M, N)` computes, per row: a max, a subtraction, an exponential, a sum, and a division. That's roughly 4–5 arithmetic operations per element. Compare that to the amount of *data movement* a naive implementation requires, and you'll see the imbalance immediately. Here's a numerically-stable naive PyTorch implementation:

```python
def naive_softmax(x):
    x_max = x.max(dim=1)[0]                  # read MN, write M
    z = x - x_max[:, None]                    # read MN + M, write MN
    numerator = torch.exp(z)                  # read MN, write MN
    denominator = numerator.sum(dim=1)        # read MN, write M
    out = numerator / denominator[:, None]    # read MN + M, write MN
    return out
```

Each line is its own separate CUDA kernel launch under the hood (that's what "naive" means here — no fusion), and each one must read its input from, and write its output back to, global GPU memory (HBM) — the slow, high-latency memory tier, as distinct from the on-chip SRAM (registers and shared memory) you already know from CUDA. Add up the traffic across all five steps and this naive implementation reads **5MN + 2M** elements from DRAM and writes back **3MN + 2M** elements — roughly **8MN** elements of total memory traffic for a computation that, if it stayed on-chip, would only ever need to read `x` once (`MN` elements) and write the result once (`MN` elements): **2MN** elements, a full **4x** reduction in memory traffic.

This is the roofline-model insight from your CS149/CUDA background applied directly: softmax has very low **arithmetic intensity** (FLOPs per byte moved), so its performance ceiling is set by memory bandwidth, not by the GPU's compute throughput. A "faster" softmax isn't about doing the math quicker — the math is nearly free — it's about **not going back and forth to HBM five times for the same MN elements.**

## 7.2 The Fix: Keep the Whole Row On-Chip

The fusion strategy is direct: if a single program instance loads an *entire row* into on-chip memory (registers/shared memory, addressed through Triton's abstraction rather than managed by hand) and performs the max, subtract, exponentiate, sum, and divide **without ever writing intermediate results back to HBM**, then the *only* HBM traffic for the whole operation is one read of the row and one write of the result — exactly the 2MN-element floor from §7.1.

This requires one structural constraint you haven't needed until now: **`BLOCK_SIZE` must be at least as large as `n_cols`**, so that `tl.arange(0, BLOCK_SIZE)` can address the *entire* row in a single load, with masking (Chapter 4) covering the padding past `n_cols`. In practice, `BLOCK_SIZE` is set to the next power of two `≥ n_cols` (Triton's utility `triton.next_power_of_2` does this). This is a meaningfully different design point from the matmul kernel you'll build in Chapter 9, where tiles are deliberately *smaller* than the full problem and the kernel loops over chunks — here, each program's tile *is* the entire reduction axis, by design, precisely so no cross-program communication or multi-pass streaming is needed.

**A terminology note worth being precise about**, since it causes real confusion later: this "load the whole row once" strategy is fusion, but it is *not* the "online softmax" technique you'll meet in Chapter 17. Online softmax is a recursive reformulation needed specifically when the reduction axis is *too large to fit in one block* (as in attention, where the key/value sequence length can be enormous) — it lets you stream chunks and update a running max/sum incrementally. Here, because the whole row fits in SRAM, you don't need that machinery at all; you load once, hold the entire row in registers, and compute directly. Keep this distinction in mind so you don't over-engineer this chapter's kernel or wonder why it looks simpler than what you'll write in Chapter 17.

## 7.3 The Numerically Stable Formula, Recapped

Naively computing `exp(x_i) / sum(exp(x_j))` overflows for even moderately large `x_i` (float32 overflows around `exp(88)`). The standard fix — which you already used in the masking discussion in Chapter 4 — is to subtract the row's max before exponentiating:

```
softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

This is mathematically identical (the max cancels in the ratio) but numerically safe, since `x_i - max(x) <= 0` for all `i`, keeping every exponential in `(0, 1]`. This is exactly why the masked-out lanes in the max-reduction step must use `other=-float('inf')` (Chapter 4, §4.2) — a masked-out lane must never win the max, and `-inf` guarantees it never does, regardless of what real data happens to be in the row.

## 7.4 Writing the Kernel

```python
import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                    n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

Walk through what's new versus everything in Part II:

- **`other=-float('inf')` on the load** is the exact scenario Chapter 4 §4.2 asked you to reason through in the abstract — here it's load-bearing for real. Any padding lane past `n_cols` becomes `-inf`, which `tl.max` correctly ignores, and which — after the subtraction — becomes `exp(-inf) = 0`, correctly contributing nothing to the sum either. **One masking decision, made correctly, keeps both the max-reduction and the sum-reduction correct without any special-casing.**
- **`tl.max(row, axis=0)` and `tl.sum(numerator, axis=0)`** are genuine on-chip reductions across the block — the compiler generates the appropriate intra-block reduction (conceptually similar to a CUDA warp/block-level tree reduction, but you never write the tree yourself).
- **The `for row_idx in tl.range(...)` loop** is not there because a single row is too big for one program (it isn't — that's the whole point of §7.2) — it's there because, as you'll see in §7.5, we deliberately launch *fewer programs than rows* and have each one process multiple rows. This is precisely the grid-stride pattern you built by hand in Chapter 3, §3.6 — recognize it here as the same idiom doing real work.

## 7.5 Persistent Programs: Choosing `num_programs` from Occupancy

Here's a design decision the official tutorial makes that's easy to miss if you only skim the kernel body: **the launch grid is not `(n_rows,)`.** Instead, the number of program instances is computed from the GPU's actual **occupancy** — how many program instances can physically be resident on the hardware at once — and capped at `n_rows`:

```python
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

# Precompile once to learn actual register/shared-memory usage:
kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                                BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
                                num_warps=num_warps, grid=(1,))
kernel._init_handles()
n_regs = kernel.n_regs
size_smem = kernel.metadata.shared

occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
occupancy = min(occupancy, SIZE_SMEM // size_smem)
num_programs = min(NUM_SM * occupancy, n_rows)

kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                              BLOCK_SIZE, num_stages)
```

The reasoning, made explicit: every program instance that runs concurrently on an SM consumes a share of that SM's fixed register file and shared-memory budget. `n_regs` (registers actually used per thread, discovered by *compiling* the kernel once via `.warmup()` before ever launching it for real) and `num_warps` together tell you how many *warps'* worth of registers one program instance needs; dividing the SM's total register budget by that tells you how many program instances can be simultaneously resident per SM — that's `occupancy`. Multiply by `NUM_SM` (the number of SMs on the whole GPU) and you get the true hardware-limited ceiling on concurrent program instances, independent of how many rows your problem actually has.

If `n_rows` is smaller than that ceiling, you launch exactly `n_rows` programs (one per row, no looping needed). If `n_rows` is *larger* — the common case for real workloads — you launch only as many **persistent** programs as the hardware can actually run at once, and each one loops over multiple rows via the grid-stride pattern in §7.4. This avoids the overhead of launching (and the OS/driver-level bookkeeping of scheduling) potentially millions of program instances for a matrix with millions of rows, when the GPU could only ever run a few thousand of them simultaneously anyway. This is your first real encounter with **occupancy** as a first-class performance concept — you'll formalize registers, shared memory, and occupancy properly in Chapter 15, and use profiling tools to *measure* rather than *compute by hand* in Chapter 22, but the reasoning here is the real thing, not a simplification.

## 7.6 The Host-Side Wrapper

```python
def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200_000 else 2  # more stages if there's headroom
    y = torch.empty_like(x)
    # ... occupancy computation from §7.5 ...
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                                  BLOCK_SIZE, num_stages)
    return y
```

Note `x.stride(0)` is passed explicitly rather than assumed — the row-stride handling from Chapter 5 applies here exactly as it does everywhere else; this kernel works correctly on a non-contiguous row view for free, as long as you pass the real stride.

## 7.7 Correctness Testing on an Irregular Shape

```python
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, dim=1)
assert torch.allclose(y_triton, y_torch)
```

`1823` rows and `781` columns are deliberately *not* round numbers. `781` is not a power of two, so `BLOCK_SIZE = triton.next_power_of_2(781) = 1024` — meaning every row's load and store genuinely exercises the mask (`col_offsets < 781` is false for the last 243 lanes of every row). `1823` rows, likely larger than the hardware-occupancy ceiling on most GPUs, exercises the persistent-program loop from §7.5. This single test, in other words, is deliberately designed to fail loudly if either the masking or the occupancy/looping logic is wrong — a good habit to copy into your own test design, not just this kernel: pick shapes that stress every non-trivial code path, not round numbers that happen to sidestep them.

## 7.8 Benchmarking: GB/s, Not GFLOP/s

Because §7.1 established this is a bandwidth-bound operation, the right unit for benchmarking is **achieved memory bandwidth (GB/s)**, not FLOP/s — a FLOP/s number would make an already-fast, low-arithmetic-intensity kernel look artificially unimpressive and would tell you nothing about whether you're near the hardware's actual ceiling. Triton's benchmarking utility (`triton.testing.perf_report`, which you'll use repeatedly from here on) supports exactly this:

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'], x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider', line_vals=['triton', 'torch', 'naive_softmax'],
        line_names=['Triton', 'Torch', 'Naive Softmax'],
        ylabel='GB/s', plot_name='softmax-performance', args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    elif provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)
```

The `gbps` formula directly encodes the §7.1 analysis: `2 * numel * element_size` is exactly the theoretical floor (one read, one write of the whole matrix), divided by measured time. Running this, expect: **Triton comes out roughly on par with `torch.softmax`** (which is itself already a hand-fused CUDA kernel inside PyTorch — you're not beating an unoptimized baseline here, you're matching an expert-written one with far less code), and **meaningfully faster (rule-of-thumb: around 4x) than the naive unfused version**, confirming the §7.1 arithmetic in practice rather than just in theory.

## 7.9 A Performance Detail Worth Naming: `eviction_policy`

Recall `eviction_policy` from Chapter 4, §4.5 — a hint you were told to file away for later. Here's the "later." Because each row of `x` is read **exactly once** and never revisited by any other program instance, the input load in this kernel is a natural candidate for `eviction_policy="evict_first"`: telling the cache hierarchy "don't bother keeping this around, it won't be reused," freeing up cache capacity for data that *will* be reused. This is the official tutorial's actual justification for the hint, not a hypothetical — worth confirming for yourself in the exercises below by measuring whether it makes a detectable difference on your specific hardware (it may be small or within noise on some GPUs/shapes, which is itself a useful, honest result to observe rather than assume).

## 7.10 Hands-On

**Exercise 1 — Run the tutorial kernel end to end.** Implement §7.4–§7.6 in full, verify correctness with the irregular-shape test in §7.7, and reproduce the benchmark in §7.8 on your own hardware. Record the actual GB/s numbers for Triton, `torch.softmax`, and `naive_softmax` at a few different `N` values, and compare against your GPU's advertised peak HBM bandwidth — what fraction of peak is each implementation achieving?

**Exercise 2 — Break the occupancy assumption on purpose.** Modify the host wrapper to always launch `n_rows` programs (i.e., skip the occupancy calculation and grid-stride loop entirely — one program per row, unconditionally). Confirm correctness is unaffected, then benchmark this against the persistent-program version at a large `n_rows` (e.g., `M=100_000`). Quantify the launch-overhead cost this exposes.

**Exercise 3 — `eviction_policy` A/B test.** Add `eviction_policy="evict_first"` to the input load in §7.4, and benchmark before/after on your hardware at a few shapes. Report what you find, including if the difference is within noise — that's a legitimate and informative outcome, not a failed exercise.

**Exercise 4 — Extend to a masked softmax.** Real transformer workloads often need a softmax that ignores certain positions (e.g., padding tokens) — add an additional boolean mask input (distinct from the boundary mask you already have) that forces specific columns to contribute `0` to the output regardless of their value, and confirm it against a PyTorch reference using `masked_fill(-inf)` before `torch.softmax`.

**Exercise 5 — Where does this approach break?** Construct a case where `n_cols` is large enough that `BLOCK_SIZE = triton.next_power_of_2(n_cols)` would require an impractically large tile (e.g., `n_cols = 200,000`, as in a very large vocabulary logits row). Reason through, in writing, why the "whole row in one block" strategy from §7.2 stops being viable here, and what you'd need to do differently — you're not expected to solve this yet; you're building the motivating question that Chapter 17's online-softmax formulation answers.

## 7.11 Check Your Understanding

1. Explain, without looking back at §7.1, why softmax's performance ceiling is set by memory bandwidth rather than compute throughput. What property of the computation makes this true?
2. Why must `BLOCK_SIZE` be at least `n_cols` for this specific kernel design, and what would go wrong (not just "be suboptimal," but actually *wrong*) if you tried to use a `BLOCK_SIZE` smaller than `n_cols` without changing anything else about the algorithm?
3. What does the occupancy calculation in §7.5 actually compute, in your own words, and why is `n_regs` obtained by *compiling* the kernel (`.warmup()`) rather than estimated some other way?
4. Why is `other=-float('inf')` correct for both the max-reduction *and* (indirectly, after subtraction and exponentiation) the sum-reduction in the same kernel, using a single masked load?

## 7.12 What's Next

You've now built a real, correctly-masked, occupancy-aware fused kernel and benchmarked it honestly against a bandwidth floor — the full loop this curriculum will repeat for every kernel from here on. Chapter 8 formalizes **autotuning**: instead of hand-picking `num_warps=8` and `num_stages=4` as this chapter did, you'll let Triton search a configuration space and cache the winner automatically — the mechanism that makes the matrix-multiplication kernel in Chapter 9 (where the right tile sizes depend heavily on problem shape and hardware) actually practical to write once and run fast everywhere.
