# Chapter 9 — Profiling with Nsight Systems & Nsight Compute

## Learning objectives

By the end of this chapter you will:

- Distinguish when to use Nsight Systems vs Nsight Compute
- Read a Nsight Systems timeline and identify bottlenecks at the application level
- Interpret Nsight Compute's roofline model and place your kernel on it
- Know the 10 most important per-kernel metrics and what each tells you
- Apply a structured 5-step optimisation workflow to any kernel
- Profile the kernels you wrote in Chapters 5–7 and find their real bottleneck

---

## 9.1 Two tools, two questions

NVIDIA provides two complementary profilers. Knowing which to reach for
first saves significant time.

```
Nsight Systems  →  "Where is my application spending time?"
                   Application-level timeline
                   Shows: kernel launches, transfers, CPU/GPU overlap,
                          stream activity, API calls, NVTX regions
                   Granularity: milliseconds
                   Use first — find the expensive kernels

Nsight Compute  →  "Why is this kernel slow?"
                   Per-kernel deep analysis
                   Shows: memory throughput, occupancy, warp efficiency,
                          instruction mix, roofline position, stall reasons
                   Granularity: GPU clock cycles
                   Use second — after you know which kernel to fix
```

The workflow is always: Systems first → identify the bottleneck kernel →
Compute second → understand why → fix → repeat.

---

## 9.2 Nsight Systems — application timeline

### Installation and basic usage

```bash
# Nsight Systems ships with the CUDA toolkit
# Verify installation
nsys --version

# Profile a Python script — collect CUDA, OS, and NVTX traces
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=report1 \
    python your_script.py

# Open the GUI
nsys-ui report1.nsys-rep
```

For Google Colab or remote servers without a display:

```bash
# Generate a stats summary to stdout (no GUI needed)
nsys stats report1.nsys-rep

# Export to SQLite for programmatic analysis
nsys export --type=sqlite report1.nsys-rep
```

### Reading the timeline

The Nsight Systems GUI shows horizontal swim lanes:

```
Row: CUDA HW (GPU 0)
  ├── Compute     [kernel bars — width = execution time]
  ├── MemCpy H2D  [transfer bars]
  └── MemCpy D2H  [transfer bars]

Row: Streams
  ├── Stream 7    [operations on this stream]
  ├── Stream 8    [operations on this stream]
  └── ...

Row: CUDA API     [cudaLaunchKernel, cudaMemcpy calls from CPU]
Row: CPU (Thread) [Python/CPU activity]
Row: NVTX         [your custom region markers]
```

**What to look for:**

```
❶ Gaps between kernel bars
  → GPU idle time. Cause: CPU is busy preparing the next launch
    Fix: async launches, CUDA graphs (Ch 19)

❷ Sequential MemCpy + Kernel (no overlap)
  → Missing double buffering or using pageable memory
    Fix: pinned memory + streams (Ch 8)

❸ One very wide kernel bar dominating
  → This is your target for Nsight Compute

❹ Many tiny kernel bars
  → Launch overhead dominates. Fix: kernel fusion or CUDA graphs

❺ CPU row consistently busy while GPU row is idle
  → CPU is the bottleneck. Fix: async pipeline or prefetch
```

### Adding NVTX markers for clarity

```python
import cupy as cp

def profiled_pipeline(data):
    cp.cuda.nvtx.RangePush("Preprocessing")
    cleaned = cp.abs(data - cp.mean(data))
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush("FFT")
    spectrum = cp.fft.fft(cleaned)
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush("Power spectrum")
    power = cp.abs(spectrum) ** 2
    cp.cuda.nvtx.RangePop()

    return power

data = cp.random.rand(10_000_000, dtype=cp.float32)
result = profiled_pipeline(data)
cp.cuda.Stream.null.synchronize()
```

NVTX regions appear as coloured bars in the timeline, making it trivial
to correlate Python code regions with GPU activity.

---

## 9.3 Nsight Compute — kernel deep dive

### Basic usage

```bash
# Profile all kernels in a script (can be slow for many kernels)
ncu python your_script.py

# Profile only kernels matching a name pattern
ncu --kernel-name "matmul" python your_script.py

# Collect a specific metric set
ncu --set full python your_script.py

# Export for GUI
ncu --export report.ncu-rep python your_script.py
ncu-ui report.ncu-rep
```

From Python, you can bracket only the kernel you care about:

```python
import cupy as cp
from cupy.cuda import nvtx

# Only the code between these calls gets profiled
cp.cuda.profiler.start()
result = my_expensive_kernel(data)
cp.cuda.Stream.null.synchronize()
cp.cuda.profiler.stop()
```

---

## 9.4 The roofline model — your most important mental tool

The roofline model answers: **is this kernel compute-bound or memory-bound?**
This single question determines which optimisations will actually help.

### Arithmetic intensity

```
Arithmetic Intensity (AI) = FLOPs executed / Bytes transferred to/from HBM

Units: FLOP/byte

Examples:
  Vector addition (c = a + b):
    FLOPs:  N multiplications = N
    Bytes:  3N floats × 4 bytes = 12N bytes
    AI = N / 12N = 0.083 FLOP/byte  ← very memory-bound

  Matrix multiply (C = A × B, size N×N):
    FLOPs:  2N³  (N² dot products, each N MACs)
    Bytes:  3N² × 4 bytes (read A, B; write C)
    AI = 2N³ / 12N² = N/6 FLOP/byte  ← compute-bound for large N
    For N=1024: AI = 170 FLOP/byte
```

### The roofline chart

```
GFLOP/s
  │                              ╔══════════════════════╗
  │                          ╔══╝  Compute roof         ║ Peak FLOP/s
  │                      ╔══╝     (e.g. 312 TFLOP/s    ║ (A100 FP16)
  │                  ╔══╝          on A100)             ║
  │              ╔══╝                                   ║
  │          ╔══╝  Memory-bound region                  ║ Compute-bound
  │      ╔══╝     (slope = memory bandwidth)            ║ region
  │  ╔══╝         (e.g. 2000 GB/s on A100)              ║
  └──────────────────────────────────────────────────────► AI (FLOP/byte)
                              ↑
                        Ridge point
                    (memory BW / peak FLOP/s)
                    A100: 2000/312000 ≈ 0.006... wait
                    A100 FP32: 19.5 TFLOP/s
                    Ridge = 2000 GB/s / 19500 GFLOP/s ≈ 103 FLOP/byte

Kernel positions:
  Vector add (AI=0.08):  far left of ridge → memory-bound
                          optimise: coalescing, fewer passes, fusion
  Matmul N=1024 (AI=170): right of ridge → compute-bound
                          optimise: Tensor Cores, better instruction mix
```

### Reading the roofline in Nsight Compute

In the GUI: **Details** tab → **Roofline Analysis** section.
Your kernel appears as a dot. Its position tells you everything:

```
Dot far below the memory roof line:
  → Memory-bound AND poor bandwidth utilisation
  → Fix: coalescing (Ch 6), fewer global reads (Ch 5)

Dot near/on the memory roof line:
  → Memory-bound but well-optimised for bandwidth
  → Fix: reduce arithmetic intensity (fuse ops, use cache)

Dot far below compute roof:
  → Compute-bound AND poor compute utilisation
  → Fix: occupancy (Ch 10), instruction-level parallelism

Dot near/on compute roof:
  → Compute-bound and well-optimised
  → Minor gains only — consider algorithmic changes
```

---

## 9.5 The 10 metrics that matter

Nsight Compute exposes hundreds of metrics. These 10 cover 90% of
optimisation decisions:

### Memory metrics

```
① sm__throughput.avg.pct_of_peak_sustained_elapsed
  What: overall SM utilisation %
  Good: > 80%
  Low means: kernel is not keeping SMs busy

② l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second  (GB/s)
  What: actual global memory read bandwidth achieved
  Compare to: GPU peak HBM bandwidth (spec sheet)
  Good: > 70% of peak for memory-bound kernels

③ l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum /
   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
  What: sectors per request = coalescing efficiency
  Good: close to 1.0 (1 sector per request = perfect coalescing)
  High means: stride access or random access patterns

④ l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  What: shared memory bank conflicts (load)
  Good: 0
  Non-zero means: add padding (TILE+1 trick from Ch 5–6)
```

### Compute metrics

```
⑤ sm__warps_active.avg.pct_of_peak_sustained_active  (occupancy %)
  What: average % of warp slots that are active
  Good: > 50% (more is better but diminishing returns above 75%)
  Low means: register pressure, shared memory limits, or small blocks

⑥ smsp__thread_inst_executed_op_fadd_pred_on.sum  (FLOP count)
  What: actual floating point add instructions executed
  Use to: verify your FLOPs/s calculation

⑦ smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
  What: memory access efficiency %
  Good: 100%
  Low = same as metric ③ — non-coalesced access
```

### Warp/scheduler metrics

```
⑧ smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
  What: % of cycles warps stall waiting for global memory loads
  Good: < 20%
  High means: memory latency is the bottleneck
  Fix: increase occupancy (more warps to hide latency), prefetch

⑨ smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct
  What: % of cycles stalled on branch resolution
  High means: warp divergence (Ch 1 / Ch 3)
  Fix: restructure branches so all warp threads take same path

⑩ smsp__average_warp_latency_per_inst_issued.ratio
  What: average warp latency in clock cycles
  Good: < 20 cycles
  High means: instruction-level latency not hidden
  Fix: increase ILP (instruction-level parallelism) in kernel
```

---

## 9.6 Profiling your Chapter 5–7 kernels

Let's apply Nsight Compute to the kernels you already wrote.

### Profiling matmul_naive vs matmul_tiled

```python
import cupy as cp
import numpy as np
from numba import cuda, float32

TILE = 16

@cuda.jit
def matmul_naive(A, B, C):
    row, col = cuda.grid(2)
    M, K = A.shape; _, N = B.shape
    if row < M and col < N:
        acc = float32(0.0)
        for k in range(K):
            acc += A[row, k] * B[k, col]
        C[row, col] = acc

@cuda.jit
def matmul_tiled(A, B, C):
    A_tile = cuda.shared.array((TILE, TILE), dtype=float32)
    B_tile = cuda.shared.array((TILE, TILE), dtype=float32)
    row = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    ty, tx = cuda.threadIdx.y, cuda.threadIdx.x
    M, K = A.shape; _, N = B.shape
    acc = float32(0.0)
    for tile_idx in range((K + TILE - 1) // TILE):
        a_col = tile_idx * TILE + tx
        b_row = tile_idx * TILE + ty
        A_tile[ty, tx] = A[row, a_col] if (row < M and a_col < K) else float32(0.0)
        B_tile[ty, tx] = B[b_row, col] if (b_row < K and col < N) else float32(0.0)
        cuda.syncthreads()
        for k in range(TILE):
            acc += A_tile[ty, k] * B_tile[k, tx]
        cuda.syncthreads()
    if row < M and col < N:
        C[row, col] = acc

M = K = N = 1024
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
A_dev = cuda.to_device(A)
B_dev = cuda.to_device(B)
C_dev = cuda.device_array((M, N), dtype=np.float32)

threads = (TILE, TILE)
blocks  = ((N + TILE-1)//TILE, (M + TILE-1)//TILE)

# Warm up
matmul_naive[blocks, threads](A_dev, B_dev, C_dev)
matmul_tiled[blocks, threads](A_dev, B_dev, C_dev)
cuda.synchronize()

# Profile — run this script under ncu:
# ncu --set full --kernel-name "matmul_naive" python this_script.py
cp.cuda.profiler.start()
matmul_naive[blocks, threads](A_dev, B_dev, C_dev)
cuda.synchronize()
cp.cuda.profiler.stop()
```

**What you will see in Nsight Compute for matmul_naive:**

```
Metric                              Naive       Tiled (expected)
─────────────────────────────────────────────────────────────────
Global load bandwidth               ~12 GB/s    ~80 GB/s
Memory access efficiency            ~3–5%       ~95%
Occupancy                           ~25%        ~50%
Long scoreboard stall %             ~85%        ~30%
GFLOP/s achieved                    ~12         ~180
Roofline position                   Deep memory Memory roof
                                    bound       (near ridge)
```

The naive kernel's 85% long scoreboard stall tells the whole story:
threads spend 85% of their time waiting for global memory loads.

---

## 9.7 Structured optimisation workflow

Apply this 5-step process to every kernel you want to optimise:

```
Step 1 — Measure baseline
  Run under Nsight Systems, identify the most expensive kernel.
  Record: time (ms), GFLOP/s, GB/s.

Step 2 — Classify the bottleneck
  Check roofline position:
    Left of ridge → memory-bound
    Right of ridge → compute-bound
  Never optimise compute in a memory-bound kernel or vice versa.

Step 3 — Find the root cause
  Memory-bound:
    Low bandwidth?   → check coalescing (metric ③)
    High bandwidth?  → reduce data movement (fusion, caching)
  Compute-bound:
    Low occupancy?   → check register/shared mem limits (Ch 10)
    High stalls?     → check instruction latency (metric ⑩)

Step 4 — Apply one fix at a time
  Change one thing. Re-profile. Measure improvement.
  Multiple simultaneous changes make it impossible to know what helped.

Step 5 — Repeat until:
  Memory-bound: achieved BW > 80% of peak HBM bandwidth
  Compute-bound: achieved GFLOP/s > 80% of peak FLOP/s
  Or: further gains require algorithmic changes (not micro-optimisation)
```

### Worked example — optimising the histogram kernel

```python
# From Chapter 4 — naive histogram with atomics
from numba import cuda
import numpy as np

@cuda.jit
def histogram_v1(data, hist, n_bins, lo, hi):
    i = cuda.grid(1)
    if i < data.shape[0]:
        val    = data[i]
        bin_id = int((val - lo) / (hi - lo) * n_bins)
        bin_id = min(max(bin_id, 0), n_bins - 1)
        cuda.atomic.add(hist, bin_id, 1)   # global atomic — high contention
```

Step 1 — Profile: 50 ms for 10M elements.

Step 2 — Roofline: memory-bound (low AI — just reads and increments).

Step 3 — Root cause: Nsight shows `atomic_contention` stalls dominating.
Multiple threads hammering the same bin serialise.

Step 4 — Fix: **privatisation** — each block builds a private histogram
in shared memory (no contention), then atomically merges to global memory
(far fewer global atomics):

```python
@cuda.jit
def histogram_v2(data, hist, n_bins, lo, hi):
    """Privatised histogram — shared memory per block."""
    from numba import float32, int32

    # Private histogram in shared memory — one per block
    private_hist = cuda.shared.array(256, dtype=int32)  # assume n_bins <= 256

    tid = cuda.threadIdx.x
    i   = cuda.grid(1)

    # Initialise private histogram to zero
    if tid < n_bins:
        private_hist[tid] = 0
    cuda.syncthreads()

    # Accumulate into fast shared memory
    if i < data.shape[0]:
        val    = data[i]
        bin_id = int((val - lo) / (hi - lo) * n_bins)
        bin_id = min(max(bin_id, 0), n_bins - 1)
        cuda.atomic.add(private_hist, bin_id, 1)   # shared atomic — low contention

    cuda.syncthreads()

    # Merge private → global (one atomic per bin per block)
    if tid < n_bins:
        cuda.atomic.add(hist, tid, private_hist[tid])


# Benchmark v1 vs v2
import numpy as np
from numba import cuda

N       = 10_000_000
n_bins  = 256
data    = np.random.randn(N).astype(np.float32)
hist_v1 = np.zeros(n_bins, dtype=np.int32)
hist_v2 = np.zeros(n_bins, dtype=np.int32)

data_dev   = cuda.to_device(data)
hist_v1_dev = cuda.to_device(hist_v1)
hist_v2_dev = cuda.to_device(hist_v2)

threads = 256
blocks  = (N + threads - 1) // threads

start = cuda.event(timing=True); end = cuda.event(timing=True)

start.record()
histogram_v1[blocks, threads](data_dev, hist_v1_dev, n_bins, -4.0, 4.0)
end.record(); end.synchronize()
t_v1 = cuda.event_elapsed_time(start, end)

start.record()
histogram_v2[blocks, threads](data_dev, hist_v2_dev, n_bins, -4.0, 4.0)
end.record(); end.synchronize()
t_v2 = cuda.event_elapsed_time(start, end)

print(f"v1 (naive atomics):    {t_v1:.2f} ms")
print(f"v2 (privatised):       {t_v2:.2f} ms")
print(f"Speedup:               {t_v1/t_v2:.1f}x")
print(f"Results match: {np.array_equal(hist_v1_dev.copy_to_host(),
                                        hist_v2_dev.copy_to_host())}")
```

Expected speedup: 5–15× depending on the number of bins and contention level.

Step 5 — Re-profile v2: atomic contention stalls should drop from ~70%
to <10%. The remaining time should be dominated by global memory bandwidth
— indicating the kernel is now well-optimised for this algorithm.

---

## 9.8 Quick-reference metric cheatsheet

```
Symptom in Nsight Compute          Root cause             Chapter fix
────────────────────────────────────────────────────────────────────────
Low global bandwidth (<50% peak)   Non-coalesced access   Ch 6
High sectors/request (>>1)         Non-coalesced access   Ch 6
High shared bank conflicts         Missing +1 padding     Ch 5, Ch 6
Low occupancy (<25%)               Register/smem pressure Ch 10
High long-scoreboard stall (>50%)  Memory latency         Ch 5 (tiling)
High branch stall (>20%)           Warp divergence        Ch 1, Ch 3
Low GFLOP/s vs roofline            Under-utilised compute Ch 10, Ch 14
Kernel time << transfer time       Transfer dominates      Ch 8 (pipeline)
Many tiny kernels, high overhead   Launch latency          Ch 19 (graphs)
```

---

## Hands-on exercises

### Exercise 1 — Profile your matmul kernels

Run both `matmul_naive` and `matmul_tiled` from Chapter 5 under Nsight
Compute (or collect metrics programmatically). Record:

```
Kernel           BW (GB/s)   GFLOP/s   Occupancy   Scoreboard stall %
──────────────────────────────────────────────────────────────────────
matmul_naive
matmul_tiled
cuBLAS (cp @)
```

Where does each kernel sit on the roofline? Are they memory or compute bound?

### Exercise 2 — NVTX timeline audit

Instrument the Chapter 4 signal processing pipeline with NVTX markers
and profile it under Nsight Systems. Answer:

- What fraction of wall time is GPU compute vs H2D/D2H transfer vs CPU?
- Is there any overlap between transfers and compute?
- Which operation is the single biggest GPU time consumer?

### Exercise 3 — Classify these kernels without running them

Given only arithmetic intensity (AI), classify each as memory or compute
bound and predict which optimisation direction will help. Use a T4 GPU:
peak GFLOP/s = 8,141 FP32, peak HBM bandwidth = 300 GB/s,
ridge point = 300/8141 ≈ 27 FLOP/byte.

```
Kernel                AI (FLOP/byte)    Memory or Compute bound?
────────────────────────────────────────────────────────────────
Vector addition        0.08
Element-wise ReLU      0.25
1D convolution (k=3)   1.5
Dense layer (N=512)    85
Attention (seq=2048)   varies by impl
```

### Exercise 4 — Fix the histogram and measure

Run the `histogram_v1` vs `histogram_v2` benchmark from section 9.7 on
your GPU. Then answer:

- What speedup do you see?
- Profile v2 — what is now the dominant stall reason?
- What would a v3 look like? Is there a further optimisation to apply?

### Exercise 5 — Roofline calculation by hand

For your `matmul_tiled` kernel with TILE=16, M=K=N=1024, float32:

a) Calculate the theoretical arithmetic intensity assuming perfect tiling
   (only tile loads count as HBM traffic, not per-element).

b) Calculate the achieved arithmetic intensity from your profiler numbers
   (use actual GFLOP/s and actual GB/s from Exercise 1).

c) Is the achieved AI close to theoretical? If not, what explains the gap?
   (Hint: padding zeros for out-of-bounds tiles, syncthreads overhead, etc.)

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Systems vs Compute | Systems = where time goes (application); Compute = why kernel is slow |
| NVTX markers | Label your code regions — makes timeline readable immediately |
| Roofline model | Memory-bound (left of ridge) vs compute-bound (right) — determines fix direction |
| Arithmetic intensity | FLOPs / bytes — the single most important kernel characteristic |
| Top metric: scoreboard stall | >50% means memory latency dominates — fix with tiling or more occupancy |
| Top metric: sectors/request | >1 means non-coalesced access — fix with Ch 6 techniques |
| Top metric: occupancy | <25% means under-utilised — fix with Ch 10 techniques |
| Privatisation pattern | Shared memory → global atomic: reduces contention 5–15× for histograms |
| One fix at a time | Multiple simultaneous changes make causality impossible to determine |
| 80% rule | Stop micro-optimising when you hit 80% of the relevant roofline ceiling |

---

## What's next

Chapter 10 dives into **occupancy, warp efficiency, and launch
configuration** — the systematic method for choosing block sizes, managing
register pressure, and ensuring the GPU scheduler always has enough warps
to hide latency. With Nsight Compute from this chapter, you will be able
to measure occupancy directly and verify that your changes actually improve
it rather than guessing.

Paste your Exercise 1 metric table and Exercise 3 classification answers
and we will discuss before moving on.
