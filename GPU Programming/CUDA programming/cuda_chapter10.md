# Chapter 10 — Occupancy, Warp Efficiency & Launch Configuration

## Learning objectives

By the end of this chapter you will:

- Understand occupancy precisely — what limits it and why higher isn't always better
- Calculate theoretical occupancy by hand from register and shared memory usage
- Choose block sizes systematically rather than by intuition
- Identify and fix register pressure causing occupancy collapse
- Understand warp divergence's quantitative impact on efficiency
- Use the CUDA Occupancy API and Nsight Compute to measure and tune

---

## 10.1 What occupancy actually means

Occupancy is the ratio of active warps to the maximum warps an SM can
theoretically hold simultaneously:

```
Occupancy = active warps per SM / maximum warps per SM

Maximum warps per SM (hardware limit):
  Volta / Turing / Ampere / Hopper: 64 warps per SM

Example:
  Your kernel has 256 threads/block = 8 warps/block
  SM can fit 4 blocks simultaneously → 32 warps active
  Occupancy = 32 / 64 = 50%
```

Why does occupancy matter? The GPU hides memory latency by switching
between warps. When a warp stalls waiting for global memory (~400 cycles),
the scheduler picks another ready warp to run. If there are no other
warps, the SM sits idle for ~400 cycles.

```
Low occupancy (2 warps):
  Warp 0: [work][STALL 400 cycles         ][work]
  Warp 1:            [work][STALL 400 cycles    ][work]
  SM:     [busy][IDLE─────────────────────]

High occupancy (16 warps):
  Warp  0: [work][stall...                ]
  Warp  1:       [work][stall...          ]
  Warp  2:             [work][stall...    ]
  ...
  SM:     [busy][busy][busy][busy][busy]  ← latency fully hidden
```

The key insight: **occupancy is a tool for hiding memory latency, not a
goal in itself.** A compute-bound kernel can run at 100% efficiency with
25% occupancy if warps never stall.

---

## 10.2 What limits occupancy — the three constraints

Three hardware resources limit how many blocks fit simultaneously on an SM.
The binding constraint (the one that runs out first) determines occupancy.

### Constraint 1 — Thread count

```
Max threads per SM: 2048  (all modern GPUs since Volta)
Max threads per block: 1024

If block size = 256 threads:
  Blocks per SM ≤ 2048 / 256 = 8 blocks
  Warps  per SM = 8 × (256/32) = 64 warps → 100% occupancy

If block size = 1024 threads:
  Blocks per SM ≤ 2048 / 1024 = 2 blocks
  Warps  per SM = 2 × (1024/32) = 64 warps → still 100% occupancy

If block size = 64 threads:
  Blocks per SM ≤ 2048 / 64 = 32 blocks... but block limit = 32 ✓
  Warps  per SM = 32 × (64/32) = 64 warps → 100% occupancy
  (Any smaller block risks hitting the 32 blocks/SM limit)
```

Thread count alone rarely limits occupancy — the other two do.

### Constraint 2 — Register usage (the most common limiter)

Each SM has a fixed register file: **65,536 32-bit registers**.
Every thread in every active warp needs its registers simultaneously.

```
Registers per thread (from --ptxas-options=-v or Nsight Compute)
Let's say: 32 registers/thread

Max threads from register budget:
  65,536 registers / 32 registers per thread = 2,048 threads → 100% occupancy

If your kernel uses 64 registers/thread:
  65,536 / 64 = 1,024 threads → 50% occupancy ← register-limited

If your kernel uses 128 registers/thread:
  65,536 / 128 = 512 threads → 25% occupancy ← severe register pressure
```

Register count is determined by the compiler — you cannot set it directly.
You influence it by:
- Reducing local variable count (reuse variables)
- Limiting loop unrolling (`#pragma unroll 4` instead of full unroll)
- Using `__launch_bounds__` to tell the compiler the maximum threads/block

### Constraint 3 — Shared memory

Each SM has a fixed shared memory bank: **48–228 KB** (configurable,
48 KB default on most configs).

```
Shared memory per block:
  matmul TILE=16, float32: 2 × 16×16×4 = 2048 bytes = 2 KB

Max blocks from shared memory budget (48 KB default):
  48 KB / 2 KB = 24 blocks... but thread limit gives 8 blocks
  → Thread count is binding, not shared memory (good)

matmul TILE=32, float32: 2 × 32×32×4 = 8192 bytes = 8 KB
  48 KB / 8 KB = 6 blocks
  Thread limit: 2048 / 1024 = 2 blocks → threads are binding

Extreme: 24 KB shared memory per block:
  48 KB / 24 KB = 2 blocks
  2 × 1024 threads = 2048 → 100% from thread count
  BUT: if block = 512 threads: 2 × 512 = 1024 threads → 50% occupancy
  → Shared memory is now binding
```

---

## 10.3 Calculating occupancy by hand

Use the CUDA Occupancy Calculator logic manually. Given:

```
GPU specs (A100):
  Max warps per SM:        64
  Max threads per SM:      2048
  Max blocks per SM:       32
  Register file per SM:    65,536
  Shared memory per SM:    48 KB (default config)
  Max shared mem per block: 48 KB

Kernel parameters:
  Threads per block:  256  (= 8 warps)
  Registers/thread:   40
  Shared mem/block:   6 KB
```

Step 1 — Blocks limited by threads:
```
  2048 / 256 = 8 blocks
```

Step 2 — Blocks limited by registers:
```
  Registers needed per block: 256 × 40 = 10,240
  Max blocks: floor(65536 / 10240) = 6 blocks
```

Step 3 — Blocks limited by shared memory:
```
  Max blocks: floor(48 KB / 6 KB) = 8 blocks
```

Step 4 — Binding constraint = minimum = 6 blocks (register-limited)

Step 5 — Active warps and occupancy:
```
  Active blocks: 6
  Active warps:  6 × 8 = 48 warps
  Occupancy:     48 / 64 = 75%
```

### Python: CUDA Occupancy API

```python
from numba import cuda
import numpy as np

@cuda.jit
def my_kernel(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        # Some computation
        x = arr[i] * 2.0 + 1.0
        arr[i] = x * x

# Query occupancy for different block sizes
for block_size in [64, 128, 256, 512, 1024]:
    # Numba exposes occupancy via the dispatcher
    occ = my_kernel.get_max_grid_size(block_size)
    print(f"Block size {block_size:4d}: {occ}")

# CuPy / CUDA C approach using cupyx
import cupy as cp

kernel_code = r"""
extern "C" __global__
void sample_kernel(float* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = arr[i] * 2.0f + 1.0f;
}
"""

kernel = cp.RawKernel(kernel_code, 'sample_kernel')

# Get occupancy per block size
N = 1_000_000
arr = cp.ones(N, dtype=cp.float32)

for bs in [64, 128, 256, 512, 1024]:
    blocks = (N + bs - 1) // bs
    # Time it
    start = cp.cuda.Event(); end = cp.cuda.Event()
    # warm up
    kernel((blocks,), (bs,), (arr, N))
    cp.cuda.Stream.null.synchronize()
    start.record()
    for _ in range(20):
        kernel((blocks,), (bs,), (arr, N))
    end.record(); end.synchronize()
    t = cp.cuda.get_elapsed_time(start, end) / 20
    print(f"Block size {bs:4d}: {t:.3f} ms")
```

---

## 10.4 `__launch_bounds__` — telling the compiler your intent

The compiler over-allocates registers when it doesn't know how many threads
per block you will use. `__launch_bounds__` constrains it:

```c
// CUDA C — tell compiler: max 256 threads/block, min 4 blocks/SM
__global__ __launch_bounds__(256, 4)
void my_kernel(float* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] *= 2.0f;
}
```

With `maxThreadsPerBlock=256`:
- Compiler allocates registers assuming max 256 threads/block
- It can use more registers per thread (256 × 40 = 10,240 < 256 × 63 = 16,128)
- Result: compiler may use 40+ registers freely without limiting occupancy

With `minBlocksPerSM=4`:
- Compiler guarantees at least 4 blocks fit per SM
- Forces register usage ≤ 65536 / (4 × 256) = 64 registers/thread
- Compiler will spill to local memory if it needs more

In Numba, use the `max_registers` parameter:

```python
from numba import cuda

@cuda.jit(max_registers=32)   # cap register use at 32/thread
def reg_limited_kernel(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] *= 2.0
```

**The tradeoff:** fewer registers → higher occupancy → better latency hiding,
BUT the compiler may spill to local memory (slow) to fit within the cap.
Always measure — register capping sometimes hurts performance.

---

## 10.5 Warp divergence — quantitative impact

From Chapter 1 you know divergence causes serialisation. Now let's
quantify it and measure it with Nsight Compute.

### The efficiency formula

```
Warp execution efficiency = active threads / (32 × clock cycles used)

No divergence (32 active, 1 pass):
  Efficiency = 32 / (32 × 1) = 100%

50% divergence (16 active per pass, 2 passes):
  Efficiency = (16 + 16) / (32 × 2) = 50%

1-thread divergence (31 + 1, 2 passes):
  Efficiency = (31 + 1) / (32 × 2) = 50%  ← same cost as 50% divergence!
```

### Measuring divergence in Nsight Compute

```bash
ncu --metrics \
  smsp__thread_inst_executed_pred_on_per_inst_executed.ratio,\
  smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct \
  python script.py
```

```
smsp__thread_inst_executed_pred_on_per_inst_executed.ratio:
  1.0 = no divergence (all 32 threads active on every instruction)
  0.5 = 50% divergence on average
  0.03 = catastrophic divergence (1/32 active)

smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct:
  % of cycles where warps are stalled waiting for branch resolution
  > 15% warrants investigation
```

### Fixing divergence — restructuring branches

```python
# Divergent version — alternating threads take different paths
@cuda.jit
def divergent_kernel(arr, out):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        if arr[i] > 0:       # half the warp goes left, half right
            out[i] = arr[i] * 2.0
        else:
            out[i] = arr[i] * -1.0

# Non-divergent version 1 — arithmetic avoids the branch entirely
@cuda.jit
def branchless_kernel(arr, out):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        val = arr[i]
        # Use multiplication by sign instead of if/else
        sign = 1.0 if val > 0 else -1.0
        out[i] = val * abs(sign) * 2.0 + val * (sign - 1.0) * 0.5
        # Or more simply:
        out[i] = val * 2.0 if val > 0 else val * -1.0
        # GPU predication handles this without divergence when compiled optimally

# Non-divergent version 2 — sort data so all positive come first
# (not always possible, but eliminates divergence at data layout level)
```

For the specific case of `max(x, 0)` (ReLU), `min`, `max`, `abs`:

```c
// CUDA C — these compile to single predicated instructions (no divergence)
out[i] = fmaxf(arr[i], 0.0f);   // ReLU — single instruction, no branch
out[i] = fabsf(arr[i]);          // abs  — single instruction
out[i] = fminf(arr[i], 1.0f);   // clamp max
```

---

## 10.6 Choosing block size systematically

Rather than guessing `256` every time, use this decision process:

### Rule 1 — Block size must be a multiple of 32

Any block size that is not a multiple of 32 wastes warp slots.
Block size 100 → 4 warps but the last warp has only 4 active threads
(28 idle threads per warp in last warp = 87.5% of that warp wasted).

### Rule 2 — Sweet spots are 128, 256, 512

```
Block size   Warps   Notes
─────────────────────────────────────────────────────
64           2       Too few warps — poor latency hiding
128          4       Good for register-heavy kernels  
256          8       ← Best default starting point
512          16      Good for memory-bound kernels
1024         32      Use only when shared memory forces it
```

### Rule 3 — Use the occupancy API to sweep

```python
import cupy as cp
import numpy as np

kernel_code = r"""
extern "C" __global__
void target_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = sqrtf(a[i] * a[i] + b[i] * b[i]);
}
"""

kernel = cp.RawKernel(kernel_code, 'target_kernel')
N = 10_000_000
a = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)
c = cp.zeros(N, dtype=cp.float32)

print(f"{'Block':>8}  {'Time(ms)':>10}  {'GB/s':>8}")
print("-" * 32)

for block_size in [64, 128, 256, 512, 1024]:
    blocks = (N + block_size - 1) // block_size

    # Warm up
    for _ in range(3):
        kernel((blocks,), (block_size,), (a, b, c, N))
    cp.cuda.Stream.null.synchronize()

    # Measure
    times = []
    for _ in range(20):
        start = cp.cuda.Event(); end = cp.cuda.Event()
        start.record()
        kernel((blocks,), (block_size,), (a, b, c, N))
        end.record(); end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))

    t = np.min(times)
    bw = (3 * N * 4) / (t * 1e-3) / 1e9
    print(f"{block_size:>8}  {t:>10.3f}  {bw:>8.1f}")
```

### Rule 4 — Shared memory kernels: let tile size drive block size

For tiled kernels, block size = tile size² is natural:
- TILE=16 → 256 threads/block
- TILE=32 → 1024 threads/block (check shared mem doesn't limit occupancy)

### Rule 5 — For 2D grids, prefer (32, 8) or (16, 16) over (8, 32)

```
(32, 8): warp = all 32 threads in x-dim → perfect coalescing
         8 rows per block → good occupancy
(16, 16): 256 threads, balanced, standard matmul config
(8, 32):  warp spans 8 x-columns and parts of 4 y-rows → messy coalescing
```

---

## 10.7 Register spilling — detection and mitigation

When a kernel exceeds the register file budget, the compiler spills
registers to **local memory** — which is actually global memory with a
per-thread namespace. Local memory access has the same ~400 cycle latency
as global memory.

### Detecting spills

```bash
# Check register and spill stats during compilation
nvcc -ptxas-options=-v my_kernel.cu

# Output example:
ptxas info: Compiling entry function 'my_kernel' for 'sm_80'
ptxas info: Function properties for my_kernel
  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  Uses 64 registers, 4096+0 bytes smem, 400 bytes cmem[0]
```

Spill stores/loads > 0 means register spilling is occurring.

In Nsight Compute: look for `l1tex__t_bytes_pipe_lsu_mem_local` metric.
Any non-zero value indicates local memory (spill) traffic.

### Reducing register pressure

```python
# High register usage — many live variables simultaneously
@cuda.jit
def register_heavy(a, b, c, d, out):
    i = cuda.grid(1)
    if i < out.shape[0]:
        # All these variables are live simultaneously → many registers
        v0 = a[i] * 2.0
        v1 = b[i] + 1.0
        v2 = c[i] - 0.5
        v3 = d[i] * v0
        v4 = v1 + v2
        v5 = v3 * v4
        v6 = v5 + v0 * v2
        v7 = v6 - v1 * v3
        out[i] = v7

# Lower register usage — reuse variables, compute sequentially
@cuda.jit
def register_light(a, b, c, d, out):
    i = cuda.grid(1)
    if i < out.shape[0]:
        # Reuse 'acc' — fewer live variables simultaneously
        acc  = a[i] * 2.0
        tmp  = b[i] + 1.0
        acc += tmp
        tmp  = c[i] - 0.5
        acc *= tmp
        tmp  = d[i] * acc
        out[i] = tmp
```

---

## 10.8 Occupancy vs ILP — the real tradeoff

High occupancy is not always the right target. Sometimes having fewer,
more capable threads beats having many simple ones.

**Instruction-level parallelism (ILP):** a single thread can have multiple
independent operations in flight simultaneously. Modern GPUs can issue
multiple instructions per clock to the same warp if they are independent.

```
Low ILP (dependent chain — no parallelism within thread):
  x = a[i] * 2.0
  y = x + 1.0        ← depends on x
  z = y * y           ← depends on y
  out[i] = z         ← dependent chain, one instruction at a time

High ILP (independent operations — GPU can pipeline them):
  x0 = a[i]   * 2.0  ┐
  x1 = a[i+1] * 2.0  │ All independent — GPU issues simultaneously
  x2 = a[i+2] * 2.0  │
  x3 = a[i+3] * 2.0  ┘
  out[i]   = x0 + 1.0
  out[i+1] = x1 + 1.0
  ...
```

The technique of processing multiple elements per thread to increase ILP
is called **thread coarsening**:

```python
# Standard: one element per thread
@cuda.jit
def scale_v1(arr, out):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        out[i] = arr[i] * 2.0 + 1.0

# Thread coarsening: four elements per thread → higher ILP
@cuda.jit
def scale_v4(arr, out):
    i = cuda.grid(1) * 4    # each thread handles 4 consecutive elements
    n = arr.shape[0]
    if i     < n: out[i]     = arr[i]     * 2.0 + 1.0
    if i + 1 < n: out[i + 1] = arr[i + 1] * 2.0 + 1.0
    if i + 2 < n: out[i + 2] = arr[i + 2] * 2.0 + 1.0
    if i + 3 < n: out[i + 3] = arr[i + 3] * 2.0 + 1.0
```

Thread coarsening lowers occupancy (fewer threads) but can increase
throughput for compute-bound kernels because the GPU pipeline stays
fuller with independent instructions. For memory-bound kernels it
rarely helps.

---

## 10.9 Occupancy tuning decision tree

```
Start: measure kernel occupancy with Nsight Compute

Is occupancy < 25%?
  YES → find the binding constraint:
    ncu metric: sm__warps_active vs theoretical max
    High register use (>64/thread)?
      → try __launch_bounds__ or reduce local variables
    High shared memory (>16 KB/block)?
      → reduce tile size or increase smem config
    Block size < 64?
      → increase block size (minimum 128 recommended)

Is occupancy 25–50%?
  Is kernel memory-bound (scoreboard stall > 50%)?
    YES → increase occupancy — more warps hide latency
          try smaller block size (more blocks) or reduce registers
    NO  (compute-bound)?
      → occupancy may not matter — check ILP instead
          try thread coarsening

Is occupancy > 50%?
  → Occupancy is probably not your bottleneck
    Profile for other issues:
      Non-coalesced access (Ch 6)?
      Bank conflicts (Ch 5)?
      Warp divergence (section 10.5)?
      Instruction latency (ILP, section 10.8)?
```

---

## Hands-on exercises

### Exercise 1 — Occupancy calculator by hand

Given this kernel profile output from Nsight Compute:
```
Threads per block:    512
Registers per thread: 48
Shared memory/block:  12 KB

GPU: A100
  Max warps/SM:      64
  Registers/SM:      65,536
  Shared mem/SM:     48 KB (default)
  Max blocks/SM:     32
  Max threads/SM:    2048
```

Calculate:
- Blocks limited by thread count
- Blocks limited by registers
- Blocks limited by shared memory
- Binding constraint and number of active blocks
- Final occupancy %

### Exercise 2 — Block size sweep

Run the block size sweep from section 10.6 on your GPU for the
`target_kernel` (vector magnitude). Record the full table.

Then answer:
- Which block size gives the best throughput?
- Is the difference significant (>10%) between block sizes?
- Does the best block size match your theoretical occupancy prediction?

### Exercise 3 — Register pressure experiment

Write a kernel that intentionally uses many registers (8+ local float
variables all live simultaneously). Run it, check the register count
via `ptxas -v` or Nsight Compute, then apply `__launch_bounds__` to cap
registers. Measure:

```
Version          Registers/thread   Occupancy   Time (ms)
──────────────────────────────────────────────────────────
Uncapped
__launch_bounds__(256)
__launch_bounds__(256, 4)
max_registers=32  (Numba)
```

Does capping registers improve or hurt performance? Why?

### Exercise 4 — Thread coarsening experiment

Take the vector scale kernel from section 10.8 and benchmark 4 versions
(1, 2, 4, 8 elements per thread) on a 100M element array:

```python
# Implement scale_v1, scale_v2, scale_v4, scale_v8
# For each: record time (ms), GB/s, and effective occupancy

# Questions:
# At what coarsening factor does throughput peak?
# Does coarsening help more for float32 or float64? Why?
# What happens to occupancy as coarsening increases?
```

### Exercise 5 — Full occupancy audit of matmul_tiled

Using Nsight Compute on the `matmul_tiled` kernel from Chapter 5:

a) What is the actual achieved occupancy?
b) What is the binding constraint (threads / registers / shared memory)?
c) If you increase TILE_SIZE from 16 to 32, how does occupancy change?
   Calculate theoretically first, then verify by measurement.
d) Does higher or lower occupancy correlate with better matmul performance
   in this case? What does that tell you about whether matmul is
   memory-bound or compute-bound at this size?

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Occupancy definition | Active warps / max warps per SM — tool for hiding latency, not a goal |
| Three constraints | Threads, registers, shared memory — the minimum determines occupancy |
| Register limit | 65,536 registers / SM; >64 registers/thread → occupancy below 50% |
| `__launch_bounds__` | Tell compiler max threads/block → tighter register allocation |
| Warp divergence cost | Even 1 divergent thread = 50% efficiency (2 passes for 32 threads) |
| Block size default | 256 threads is the right starting point; sweep 128–512 to confirm |
| 2D block convention | (32, 8) or (16, 16) — never (8, 32) — x-dim must cover columns |
| Thread coarsening | Multiple elements per thread → higher ILP, lower occupancy |
| Register spilling | Spills to local (global) memory → check ptxas -v for spill count |
| Occupancy vs ILP | Memory-bound: maximise occupancy; compute-bound: maximise ILP |

---

## What's next

Chapter 11 covers **atomic operations and lock-free algorithms** — the
mechanisms for safely writing to shared locations from thousands of
concurrent threads. You will implement histogram (again, but correctly
at scale), sparse accumulation, and learn when atomics are fast vs when
they destroy performance. The privatisation pattern from Chapter 9 will
be generalised into a reusable technique.

Paste your block size sweep table from Exercise 2 and your matmul
occupancy audit from Exercise 5 — both connect directly to Chapter 11's
content.
