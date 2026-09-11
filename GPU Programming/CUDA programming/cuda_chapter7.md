# Chapter 7 — Parallel Reductions & Scan

## Learning objectives

By the end of this chapter you will:

- Understand why a serial reduction loop is the wrong mental model for GPUs
- Implement a tree-based parallel reduction from scratch — the foundational pattern
- Use warp shuffle intrinsics (`__shfl_down_sync`) for warp-level reduction without shared memory
- Build a complete multi-block reduction that handles arrays of any size
- Implement inclusive and exclusive prefix scan (the building block of many ML ops)
- Understand where reduction and scan appear in real ML workloads

---

## 7.1 Why serial reduction is wrong on a GPU

The serial approach — one thread sums all N elements in a loop — is the
worst possible way to use a GPU. It occupies one thread while 16,000 others
sit idle.

```
Serial (CPU mental model):
  total = 0
  for i in range(N):
      total += arr[i]
  Time: O(N) steps, 1 thread active

Parallel reduction (GPU mental model):
  Round 1: N/2 threads each add 2 elements  → N/2 partial sums
  Round 2: N/4 threads each add 2 elements  → N/4 partial sums
  Round 3: N/8 threads each add 2 elements  → N/8 partial sums
  ...
  Round log2(N): 1 thread adds final 2      → 1 result
  Time: O(log N) steps, N/2 threads active in round 1
```

For N = 1,024: serial takes 1,024 steps. Parallel takes 10 steps.
The GPU does 100× fewer sequential steps while using all available threads.

---

## 7.2 Naive reduction — the interleaved pattern

The most natural parallel reduction divides work by having threads stride
across the array. It works but has two problems we will fix iteratively.

```python
from numba import cuda, float32
import numpy as np

BLOCK = 256

@cuda.jit
def reduce_naive(arr, partial_sums):
    """
    Interleaved addressing reduction.
    Each block reduces its chunk to one partial sum.
    """
    shared = cuda.shared.array(BLOCK, dtype=float32)

    tid = cuda.threadIdx.x
    i   = cuda.grid(1)

    # Load from global into shared
    shared[tid] = arr[i] if i < arr.shape[0] else float32(0.0)
    cuda.syncthreads()

    # Tree reduction — stride halves each round
    stride = BLOCK // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Thread 0 writes block result
    if tid == 0:
        partial_sums[cuda.blockIdx.x] = shared[0]
```

**Problem 1 — Warp divergence:** In the last few rounds, only a few threads
do work while the rest idle. When `stride < 32`, all active threads fit in
one warp but the inactive threads in the same warp still burn clock cycles.

**Problem 2 — Shared memory bank conflicts:** With interleaved addressing,
threads access `shared[tid]` and `shared[tid + stride]`. When `stride = 1`,
threads 0 and 1 both access bank 0 and bank 1 — a 2-way conflict. At larger
strides the pattern gets worse.

---

## 7.3 Sequential addressing — fixing bank conflicts

Reverse the stride direction so threads access contiguous elements:

```python
@cuda.jit
def reduce_sequential(arr, partial_sums):
    """
    Sequential addressing — no bank conflicts, same work.
    """
    shared = cuda.shared.array(BLOCK, dtype=float32)

    tid = cuda.threadIdx.x
    i   = cuda.grid(1)

    shared[tid] = arr[i] if i < arr.shape[0] else float32(0.0)
    cuda.syncthreads()

    # ── Key change: stride starts at BLOCK//2 and decreases ──────────────
    # Active threads are always 0..stride-1 (contiguous)
    stride = BLOCK // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]   # sequential bank access ✅
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        partial_sums[cuda.blockIdx.x] = shared[0]
```

```
Round 1 (stride=128): threads 0–127 each add shared[tid] + shared[tid+128]
Round 2 (stride=64):  threads 0–63  each add shared[tid] + shared[tid+64]
Round 3 (stride=32):  threads 0–31  each add shared[tid] + shared[tid+32]
Round 4 (stride=16):  threads 0–15  ...
...
Round 8 (stride=1):   thread 0      adds shared[0] + shared[1]
```

Active threads are always contiguous → no bank conflicts. However, warp
divergence still occurs in the last 5 rounds (stride < 32) because threads
in the same warp split on the `if tid < stride` branch.

---

## 7.4 Warp shuffle intrinsics — reduction without shared memory

For the final warp-level reduction (the last 5 rounds), there is a better
tool: **warp shuffle** (`__shfl_down_sync`). Shuffles let threads exchange
register values directly — no shared memory needed, no `syncthreads`, no
bank conflicts.

```
__shfl_down_sync(mask, val, delta):
  Each thread receives the value from the thread delta lanes ahead.
  Thread i gets the value of thread i+delta.

  Before shuffle (delta=16):
    Lane:  0   1   2   3  ...  15   16  17  18  ... 31
    Val:   v0  v1  v2  v3 ...  v15  v16 v17 v18 ... v31

  After shuffle_down(mask, val, 16):
    Lane 0 gets v16, Lane 1 gets v17, ..., Lane 15 gets v31
    Lanes 16–31 get their own values (out of range)
```

Full warp reduction using shuffles:

```python
# CUDA C — warp-level reduction (embed via cp.RawKernel)
warp_reduce_code = r"""
__device__ float warp_reduce_sum(float val) {
    // Full mask: all 32 lanes participate
    unsigned int mask = 0xffffffff;

    // 5 rounds of shuffle-down: 16, 8, 4, 2, 1
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);

    return val;   // Lane 0 holds the warp sum
}

extern "C" __global__
void reduce_warp_shuffle(const float* arr, float* partial_sums, int n) {
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lid = tid % 32;       // lane id within warp
    int wid = tid / 32;       // warp id within block

    // Step 1: each thread loads one element
    float val = (i < n) ? arr[i] : 0.0f;

    // Step 2: warp-level reduction via shuffle
    val = warp_reduce_sum(val);

    // Step 3: first lane of each warp writes to shared memory
    __shared__ float warp_sums[32];   // max 32 warps per block (1024/32)
    if (lid == 0)
        warp_sums[wid] = val;

    __syncthreads();

    // Step 4: first warp reduces the warp sums
    int n_warps = (blockDim.x + 31) / 32;
    val = (tid < n_warps) ? warp_sums[tid] : 0.0f;

    if (wid == 0)
        val = warp_reduce_sum(val);

    // Step 5: thread 0 writes block result
    if (tid == 0)
        partial_sums[blockIdx.x] = val;
}
"""

import cupy as cp
import numpy as np

reduce_kernel = cp.RawKernel(warp_reduce_code, 'reduce_warp_shuffle',
                              options=('--std=c++14',))

def gpu_sum(arr_gpu, block_size=256):
    """Full reduction: handles arrays of any size via two-pass."""
    N = len(arr_gpu)
    n_blocks = (N + block_size - 1) // block_size

    partial = cp.zeros(n_blocks, dtype=cp.float32)
    reduce_kernel((n_blocks,), (block_size,), (arr_gpu, partial, N))

    # If more than one block, reduce partial sums recursively
    if n_blocks == 1:
        return float(partial[0])
    else:
        return gpu_sum(partial, block_size)

# Test
N   = 10_000_000
arr = cp.random.rand(N, dtype=cp.float32)

gpu_result = gpu_sum(arr)
cpu_result = float(cp.sum(arr))    # CuPy's built-in (reference)
print(f"GPU sum:  {gpu_result:.4f}")
print(f"CuPy sum: {cpu_result:.4f}")
print(f"Error:    {abs(gpu_result - cpu_result):.2e}")
```

Why shuffles are faster than shared memory for the last 5 rounds:
- No `syncthreads()` needed — warp executes in lockstep
- No shared memory bank conflicts
- Lower latency: register-to-register vs register-to-shared-to-register

---

## 7.5 Multi-block reduction — handling arbitrary N

A single block can reduce at most `BLOCK` elements. For N > BLOCK, you
need a two-stage strategy:

```
Stage 1: Launch ceil(N / BLOCK) blocks.
          Each block reduces its BLOCK elements → one partial sum.
          Result: ceil(N / BLOCK) partial sums in global memory.

Stage 2: Launch 1 block to reduce the partial sums.
          Result: 1 final sum.

For N = 10,000,000, BLOCK = 256:
  Stage 1: 39,063 blocks → 39,063 partial sums
  Stage 2: 1 block       → 1 final sum

If N is very large and Stage 2 still has > BLOCK partial sums:
  Recurse until result is 1 element.
```

The recursive `gpu_sum` function in section 7.4 implements exactly this.
Most production code uses two fixed passes (the second pass is always
small enough for one block if the first-pass block count ≤ 1024).

---

## 7.6 Generalising — reduction over any binary operator

Reduction is not limited to sum. Any **associative** operation works:

```python
reduce_op_code = r"""
__device__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffff;
    val = fmaxf(val, __shfl_down_sync(mask, val, 16));
    val = fmaxf(val, __shfl_down_sync(mask, val, 8));
    val = fmaxf(val, __shfl_down_sync(mask, val, 4));
    val = fmaxf(val, __shfl_down_sync(mask, val, 2));
    val = fmaxf(val, __shfl_down_sync(mask, val, 1));
    return val;
}

__device__ float warp_reduce_min(float val) {
    unsigned mask = 0xffffffff;
    val = fminf(val, __shfl_down_sync(mask, val, 16));
    val = fminf(val, __shfl_down_sync(mask, val, 8));
    val = fminf(val, __shfl_down_sync(mask, val, 4));
    val = fminf(val, __shfl_down_sync(mask, val, 2));
    val = fminf(val, __shfl_down_sync(mask, val, 1));
    return val;
}
"""
```

Common reductions in ML:

```
Sum       → layer norm (mean), softmax denominator, loss aggregation
Max       → softmax numerically stable max, argmax pre-step
Min       → gradient clipping thresholds
And/Or    → early exit conditions
LogSumExp → numerically stable log-softmax
```

---

## 7.7 Prefix scan — the second great parallel primitive

A **prefix scan** (also called prefix sum or cumulative sum) computes for
each position i the aggregate of all elements before it (exclusive) or
including it (inclusive).

```
Input:     [3,  1,  7,  0,  4,  1,  6,  3]
Inclusive: [3,  4, 11, 11, 15, 16, 22, 25]   ← each element + all before
Exclusive: [0,  3,  4, 11, 11, 15, 16, 22]   ← each element = sum of all before
```

Prefix scan appears constantly in GPU programming:
- **Stream compaction**: keep only elements passing a filter — scan tells
  each thread where to write its output
- **Memory allocation**: each thread needs a variable-size buffer — scan
  computes offsets
- **Histogram to CDF**: in image processing and sampling
- **Attention mask construction**: causal masks in transformers

### Work-efficient parallel scan (Blelloch algorithm)

The Blelloch scan runs in two phases: upsweep (reduction) and downsweep
(distribution). It is work-efficient — O(N) total operations, O(log N) depth.

```python
scan_code = r"""
extern "C" __global__
void exclusive_scan(float* arr, float* out, int n) {
    /*
     * Blelloch work-efficient exclusive prefix scan.
     * Handles one block of up to 2*BLOCK elements.
     * For simplicity: n must equal blockDim.x * 2.
     */
    extern __shared__ float tile[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = 1;

    // Load two elements per thread
    int ai = 2 * tid;
    int bi = 2 * tid + 1;
    tile[ai] = (ai < n) ? arr[bid * n + ai] : 0.0f;
    tile[bi] = (bi < n) ? arr[bid * n + bi] : 0.0f;

    // ── Upsweep (reduce) phase ─────────────────────────────────────────
    // Build a partial sum tree bottom-up
    int d = n >> 1;
    while (d > 0) {
        __syncthreads();
        if (tid < d) {
            int left  = offset * (2 * tid + 1) - 1;
            int right = offset * (2 * tid + 2) - 1;
            tile[right] += tile[left];
        }
        offset <<= 1;
        d >>= 1;
    }

    // Clear the last element (set identity for exclusive scan)
    if (tid == 0)
        tile[n - 1] = 0.0f;

    // ── Downsweep phase ────────────────────────────────────────────────
    // Distribute the partial sums top-down
    d = 1;
    while (d < n) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int left  = offset * (2 * tid + 1) - 1;
            int right = offset * (2 * tid + 2) - 1;
            float temp    = tile[left];
            tile[left]    = tile[right];
            tile[right]  += temp;
        }
        d <<= 1;
    }

    __syncthreads();

    // Write results
    if (ai < n) out[bid * n + ai] = tile[ai];
    if (bi < n) out[bid * n + bi] = tile[bi];
}
"""

import cupy as cp
import numpy as np

BLOCK = 512    # each block handles 2*BLOCK = 1024 elements

scan_kernel = cp.RawKernel(
    scan_code, 'exclusive_scan',
    options=('--std=c++14',)
)

def gpu_exclusive_scan(arr_gpu):
    """Exclusive prefix scan for arrays that fit in one block pass."""
    N   = len(arr_gpu)
    out = cp.zeros(N, dtype=cp.float32)
    shared_bytes = N * 4   # float32 shared array

    scan_kernel(
        (1,), (BLOCK,),
        (arr_gpu, out, N),
        shared_mem=shared_bytes
    )
    return out

# Test
arr = cp.array([3, 1, 7, 0, 4, 1, 6, 3], dtype=cp.float32)
result = gpu_exclusive_scan(arr)
print("Input:    ", cp.asnumpy(arr))
print("GPU scan: ", cp.asnumpy(result))
print("Expected: ", np.array([0, 3, 4, 11, 11, 15, 16, 22]))
```

### Visualising the Blelloch algorithm

```
Input:   [3]  [1]  [7]  [0]  [4]  [1]  [6]  [3]
          │    │    │    │    │    │    │    │

── Upsweep (build partial sum tree) ──────────────────────────────
Step 1:  [3]  [4]  [7]  [7]  [4]  [5]  [6]  [9]
              ↑              ↑         ↑         ↑
            (1+3)          (0+7)     (1+4)     (3+6)

Step 2:  [3]  [4]  [7] [11]  [4]  [5]  [6] [14]
                        ↑                       ↑
                     (4+7)                  (5+9)

Step 3:  [3]  [4]  [7] [11]  [4]  [5]  [6] [25]
                                                ↑
                                           (11+14)

Set last = 0: [3] [4] [7] [11] [4] [5] [6] [0]

── Downsweep (distribute) ────────────────────────────────────────
Step 1:  [3]  [4]  [7]  [0]  [4]  [5]  [6] [11]
                         ↑                      ↑
                    (was 25→0,             (was 0→11,
                     right=0+11=11)         left=0)

Step 2:  [3]  [4]  [0]  [7]  [4]  [5] [11] [16]
               ↑         ↑              ↑        ↑

Step 3:  [0]  [3]  [4] [11] [11] [15] [16] [22]

Result: [0, 3, 4, 11, 11, 15, 16, 22]  ✅ exclusive scan
```

---

## 7.8 CuPy built-in reductions — when to use them

For production code that doesn't need custom logic, CuPy wraps the
optimised cuDNN/Thrust implementations:

```python
import cupy as cp

arr = cp.random.rand(10_000_000, dtype=cp.float32)
mat = cp.random.rand(1000, 1000, dtype=cp.float32)

# Reductions
total     = cp.sum(arr)
row_sums  = cp.sum(mat, axis=1)     # reduce along columns → (1000,)
col_sums  = cp.sum(mat, axis=0)     # reduce along rows    → (1000,)
maximum   = cp.max(arr)
argmax    = cp.argmax(arr)

# Scan
cumsum    = cp.cumsum(arr)          # inclusive prefix sum
cumprod   = cp.cumprod(arr[:100])   # inclusive prefix product

# Numerically stable log-sum-exp (for softmax)
def log_sum_exp(x):
    m = cp.max(x)
    return m + cp.log(cp.sum(cp.exp(x - m)))
```

Use CuPy's built-ins by default. Write custom reduction kernels only when
you need:
- A custom reduction operator (e.g. weighted sum with a condition)
- Fusion with another operation (reduce + elementwise in one pass)
- Sub-warp granularity control

---

## 7.9 Where reduction and scan appear in ML

```
Operation            Primitive used        Where
────────────────────────────────────────────────────────────────────
Batch mean           Sum reduction         Layer norm, batch norm
Softmax              Max + sum reduction   Attention, output layer
Cross-entropy loss   Sum reduction         Loss aggregation
Gradient norm        Sum-of-squares        Gradient clipping
Top-k sampling       Partial reduction     LLM token sampling
Causal mask          Exclusive scan        Attention mask offsets
Stream compaction    Exclusive scan        Sparse attention
KV cache offsets     Exclusive scan        Paged attention (vLLM)
Histogram            Atomic + reduction    Quantisation calibration
```

The most performance-critical use: **softmax**.

```
Softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

Requires two reductions per row:
  Pass 1: max reduction    → subtract for numerical stability
  Pass 2: sum reduction    → divide to normalise

Naive implementation: 3 passes over the data (max, exp+sum, divide)
Flash Attention's trick: online softmax computes max and sum in 1 pass
  → the key algorithmic insight of Chapter 16
```

---

## Hands-on exercises

### Exercise 1 — Benchmark reduction implementations

Implement all three reduction variants (naive interleaved, sequential
addressing, warp shuffle) and benchmark them on arrays of size
1M, 10M, 100M float32:

```python
sizes = [1_000_000, 10_000_000, 100_000_000]
for N in sizes:
    arr = cp.random.rand(N, dtype=cp.float32)
    # Time each variant and record GB/s and time
    # Expected: warp shuffle ≈ 2–3× faster than naive interleaved
```

Also compare against `cp.sum()`. How close does your shuffle reduction
get to CuPy's built-in?

### Exercise 2 — Reduction correctness stress test

Reductions are prone to floating-point non-associativity. Test:

```python
N = 100_000_000
arr = cp.ones(N, dtype=cp.float32)   # known sum = N

gpu_result = gpu_sum(arr)
cpu_result = float(N)

print(f"Expected: {cpu_result}")
print(f"Got:      {gpu_result}")
print(f"Error:    {abs(gpu_result - cpu_result)}")   # likely non-zero!

# Now try with random values — compare to np.sum (double precision ref)
arr_rand = cp.random.rand(N, dtype=cp.float32)
gpu_sum_val = gpu_sum(arr_rand)
cpu_sum_val = float(np.sum(cp.asnumpy(arr_rand).astype(np.float64)))
print(f"Relative error vs float64 ref: {abs(gpu_sum_val - cpu_sum_val) / cpu_sum_val:.2e}")
```

Explain: why is the GPU sum of N ones not exactly N?
What determines the magnitude of the floating-point error?

### Exercise 3 — Row-wise softmax

Implement a numerically stable softmax over the rows of a matrix using
your reduction kernel. Each row is independent — one block per row.

```python
softmax_code = r"""
__device__ float warp_reduce_max(float val) { /* from section 7.6 */ }
__device__ float warp_reduce_sum(float val) { /* from section 7.4 */ }

extern "C" __global__
void row_softmax(const float* inp, float* out, int cols) {
    /*
     * One block per row.
     * Step 1: find row max (reduction)
     * Step 2: compute exp(x - max) and row sum (reduction)
     * Step 3: divide by sum (elementwise)
     */
    int row = blockIdx.x;
    // your implementation
}
"""
# Test: output rows should each sum to 1.0
# Verify against cp.softmax or scipy.special.softmax
```

Measure: for a `(1024, 1024)` matrix, how does your kernel compare to
`cp.exp(x) / cp.sum(cp.exp(x), axis=1, keepdims=True)` (CuPy version)?

### Exercise 4 — Verify the Blelloch scan

Run the exclusive scan from section 7.7 on:

```python
# Test 1: small known input
arr = cp.array([3, 1, 7, 0, 4, 1, 6, 3], dtype=cp.float32)
# Expected: [0, 3, 4, 11, 11, 15, 16, 22]

# Test 2: all ones (exclusive scan of N ones = [0, 1, 2, ..., N-1])
arr = cp.ones(512, dtype=cp.float32)
expected = cp.arange(512, dtype=cp.float32)

# Test 3: compare against cp.cumsum (which gives inclusive scan)
# exclusive[i] = inclusive[i-1], exclusive[0] = 0
arr = cp.random.rand(512, dtype=cp.float32)
gpu_excl  = gpu_exclusive_scan(arr)
cupy_incl = cp.cumsum(arr)
# Verify: gpu_excl[1:] should equal cupy_incl[:-1]
```

### Exercise 5 — Stream compaction using scan

Implement stream compaction: given an array of floats, keep only
elements greater than 0.5, writing them compactly to an output array.

```python
def stream_compact(arr_gpu, threshold=0.5):
    """
    Keep elements > threshold, pack them into output.
    Returns (compacted_array, count).

    Algorithm:
      1. Compute flag array: flags[i] = 1 if arr[i] > threshold else 0
      2. Exclusive scan of flags → output indices
      3. Scatter: for each i where flags[i]=1, out[scan[i]] = arr[i]
      4. Total count = scan[-1] + flags[-1]
    """
    flags = (arr_gpu > threshold).astype(cp.int32)
    indices = cp.cumsum(flags) - 1    # CuPy cumsum for simplicity
    count = int(cp.sum(flags))

    out = cp.zeros(count, dtype=arr_gpu.dtype)
    # Write a scatter kernel that uses indices to place elements
    # (or use CuPy fancy indexing as a reference)
    out_ref = arr_gpu[arr_gpu > threshold]   # reference answer

    return out_ref, count

arr = cp.random.rand(1_000_000, dtype=cp.float32)
result, n = stream_compact(arr)
print(f"Input size: {len(arr)}")
print(f"Output size: {n}  (expected ~{int(len(arr) * 0.5)})")
```

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Parallel reduction | O(log N) depth vs O(N) serial — use all threads simultaneously |
| Sequential addressing | Active threads always contiguous → no bank conflicts |
| Warp shuffle | Register-to-register exchange — no shared mem, no syncthreads, fastest |
| `__shfl_down_sync` | 5 rounds (16,8,4,2,1) reduce a full warp to lane 0 |
| Multi-block reduction | Two-pass: each block → partial sum, then reduce partial sums |
| Blelloch scan | Upsweep + downsweep = O(N) work, O(log N) depth, work-efficient |
| Exclusive vs inclusive | Exclusive: identity at position 0; Inclusive: first element at position 0 |
| Associativity required | Only associative operators work in parallel reduction (sum, max, min, and, or) |
| Stream compaction | Scan → scatter: fundamental GPU algorithm for variable output sizes |
| ML connections | Softmax, layer norm, loss, top-k, attention masks all use reduction/scan |

---

## What's next

Chapter 8 covers **CUDA streams, concurrency, and asynchronous execution**
— how to overlap data transfers with computation, run multiple kernels
simultaneously, and build pipelined GPU workflows. This is where the
"GPU is always busy" design principle becomes concrete: you will measure
the wall-clock speedup of a pipelined pipeline against a sequential one,
and build the mental model for how production inference servers keep the
GPU saturated.

Paste your benchmark table from Exercise 1 and the floating-point error
analysis from Exercise 2 — both have interesting answers worth discussing
before moving on.
