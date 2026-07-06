# Chapter 11 — Atomic Operations & Lock-Free Algorithms

## Learning objectives

By the end of this chapter you will:

- Understand what makes an atomic operation atomic and why it is needed
- Know every CUDA atomic — when each is fast vs when it serialises
- Implement a production-quality histogram using privatisation + warp reduction
- Use warp-level vote functions (`__ballot_sync`, `__any_sync`, `__all_sync`)
- Implement lock-free data structures: stack, work queue
- Know the memory ordering model and when to use `__threadfence()`

---

## 11.1 The race condition problem

Without atomics, concurrent writes to the same memory location produce
undefined results:

```python
# Two threads simultaneously execute:
# Thread A:  hist[bin] += 1
# Thread B:  hist[bin] += 1
#
# What actually happens at the hardware level:
#
#  Thread A: LOAD  hist[bin] → reg_A  (reads 5)
#  Thread B: LOAD  hist[bin] → reg_B  (reads 5)
#  Thread A: ADD   reg_A, 1  → reg_A  (reg_A = 6)
#  Thread B: ADD   reg_B, 1  → reg_B  (reg_B = 6)
#  Thread A: STORE reg_A → hist[bin]  (writes 6)
#  Thread B: STORE reg_B → hist[bin]  (writes 6)
#
# Result: hist[bin] = 6  instead of 7  ← lost update
#
# With 1000 threads all incrementing simultaneously:
# result could be anywhere between 1 and 1000
```

An **atomic operation** performs the read-modify-write as a single
indivisible hardware transaction. No other thread can interleave.

---

## 11.2 CUDA atomic operations — complete reference

### Integer atomics (global and shared memory)

```c
// All return the OLD value before the operation

int atomicAdd(int* addr, int val);        // *addr += val
int atomicSub(int* addr, int val);        // *addr -= val
int atomicExch(int* addr, int val);       // *addr  = val  (exchange)
int atomicMin(int* addr, int val);        // *addr  = min(*addr, val)
int atomicMax(int* addr, int val);        // *addr  = max(*addr, val)
int atomicAnd(int* addr, int val);        // *addr &= val
int atomicOr (int* addr, int val);        // *addr |= val
int atomicXor(int* addr, int val);        // *addr ^= val

// Compare-and-swap — the fundamental primitive
int atomicCAS(int* addr, int compare, int val);
// If *addr == compare: *addr = val, return old *addr
// Else:               return *addr unchanged
```

### Float atomics

```c
// Native float atomic (all CUDA versions)
float atomicAdd(float* addr, float val);

// Half precision (requires sm_70+, Volta+)
__half2 atomicAdd(__half2* addr, __half2 val);

// No native atomicMin/Max for float — use CAS loop instead:
__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        // Only update if val > current value
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}
```

### In Numba

```python
from numba import cuda
import numpy as np

@cuda.jit
def atomic_demo(arr, result):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        cuda.atomic.add(result, 0, arr[i])       # float or int add
        cuda.atomic.max(result, 1, arr[i])        # max
        cuda.atomic.min(result, 2, arr[i])        # min
        cuda.atomic.compare_and_swap(result, 3,   # CAS
                                     0, arr[i])

N = 1_000_000
arr = np.random.rand(N).astype(np.float32)
result = np.zeros(4, dtype=np.float32)

arr_dev    = cuda.to_device(arr)
result_dev = cuda.to_device(result)

threads = 256
blocks  = (N + threads - 1) // threads
atomic_demo[blocks, threads](arr_dev, result_dev)

r = result_dev.copy_to_host()
print(f"Sum: {r[0]:.2f}  (ref: {arr.sum():.2f})")
print(f"Max: {r[1]:.4f}  (ref: {arr.max():.4f})")
print(f"Min: {r[2]:.4f}  (ref: {arr.min():.4f})")
```

---

## 11.3 Atomic performance — when they are fast vs slow

Not all atomics are equal in cost. Performance depends entirely on
**contention** — how many threads compete for the same address.

```
Contention level    Behaviour                    Effective cost
────────────────────────────────────────────────────────────────────
Zero contention     Executes in ~1 warp cycle    ~Same as non-atomic
                    (no competing requests)

Low contention      Hardware queues requests     ~5–20 cycles
                    (L2 cache arbitration)

High contention     Serialised at memory         O(N) cycles for
                    controller                   N competing threads
                    (same address, many threads)

Extreme contention  Global bottleneck —          100–1000× slower
                    one thread at a time         than non-atomic
```

### Measuring contention impact

```python
atomic_bench_code = r"""
extern "C" __global__
void atomic_one_bucket(float* hist, const float* data, int n) {
    // Extreme contention: ALL threads write to hist[0]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&hist[0], data[i]);
}

extern "C" __global__
void atomic_many_buckets(float* hist, const float* data,
                          int n, int n_bins) {
    // Low contention: threads spread across n_bins buckets
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = (int)(data[i] * n_bins) % n_bins;
        atomicAdd(&hist[bin], data[i]);
    }
}
"""

import cupy as cp
import numpy as np

kernels = cp.RawModule(code=atomic_bench_code,
                       options=('--std=c++14',))
k_one  = kernels.get_function('atomic_one_bucket')
k_many = kernels.get_function('atomic_many_buckets')

N      = 5_000_000
n_bins = 1024
data   = cp.random.rand(N, dtype=cp.float32)
hist1  = cp.zeros(1,      dtype=cp.float32)
hist2  = cp.zeros(n_bins, dtype=cp.float32)

threads = 256
blocks  = (N + threads - 1) // threads

def time_kernel(fn, args, n_reps=20):
    for _ in range(3): fn(*args)  # warmup
    cp.cuda.Stream.null.synchronize()
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(n_reps): fn(*args)
    end.record(); end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / n_reps

t1 = time_kernel(k_one,  ((blocks,), (threads,), (hist1, data, N)))
t2 = time_kernel(k_many, ((blocks,), (threads,), (hist2, data, N, n_bins)))

print(f"1 bucket  (max contention): {t1:.2f} ms")
print(f"{n_bins} buckets (low contention): {t2:.2f} ms")
print(f"Contention penalty: {t1/t2:.0f}x slower")
# Typical: 50–200x penalty for single-bucket contention
```

---

## 11.4 Privatisation — the general contention fix

The privatisation pattern from Chapter 9 generalised:

```
Problem:  N threads compete for K shared output locations (K << N)
          → O(N/K) contention per location → serialisation

Solution: Give each block its own private copy of the K locations
          → contention within block only
          → merge private copies to global at end (K atomics per block)

Contention reduction:
  Before: N threads / K locations = N/K threads per location
  After:  block_size threads / K locations (per block)
          + n_blocks merge operations (one per block)
  For block_size=256, N=10M, K=256:
    Before: 10M/256 = 39,063 threads per location
    After:  256/256 = 1 thread per location (no contention within block!)
            + 39,063 merge atomics (low since K=256 now spreads the merges)
```

### Production histogram with privatisation + warp reduction

```python
histogram_code = r"""
extern "C" __global__
void histogram_privatised(
        const float* data, int* hist,
        int n, int n_bins, float lo, float hi) {
    /*
     * Three-stage histogram:
     * 1. Warp-level: each warp accumulates into warp-private registers
     * 2. Block-level: warp results merge into shared memory
     * 3. Global: shared memory merges into global histogram
     *
     * Handles n_bins <= blockDim.x (common case: 256 bins, 256 threads)
     */
    extern __shared__ int smem_hist[];   // n_bins integers

    int tid  = threadIdx.x;
    int i    = blockIdx.x * blockDim.x + threadIdx.x;

    // Stage 0: initialise shared histogram
    if (tid < n_bins)
        smem_hist[tid] = 0;
    __syncthreads();

    // Stage 1: accumulate into shared memory (shared atomics — low latency)
    if (i < n) {
        float val  = data[i];
        int bin    = (int)((val - lo) / (hi - lo) * n_bins);
        bin        = min(max(bin, 0), n_bins - 1);
        atomicAdd(&smem_hist[bin], 1);   // shared atomic — ~5 cycle latency
    }
    __syncthreads();

    // Stage 2: merge shared → global (global atomic — one per bin per block)
    if (tid < n_bins)
        atomicAdd(&hist[tid], smem_hist[tid]);
}
"""

import cupy as cp
import numpy as np

hist_kernel = cp.RawKernel(
    histogram_code, 'histogram_privatised',
    options=('--std=c++14',)
)

def gpu_histogram(data_gpu, n_bins=256, lo=-4.0, hi=4.0):
    N       = len(data_gpu)
    hist    = cp.zeros(n_bins, dtype=cp.int32)
    threads = 256
    blocks  = (N + threads - 1) // threads
    smem    = n_bins * 4   # n_bins * sizeof(int)

    hist_kernel(
        (blocks,), (threads,),
        (data_gpu, hist, N, n_bins, lo, hi),
        shared_mem=smem
    )
    return hist

# Test and benchmark
N    = 10_000_000
data = cp.random.randn(N, dtype=cp.float32)

# Correctness
gpu_hist = gpu_histogram(data, n_bins=256)
np_hist, _ = np.histogram(cp.asnumpy(data), bins=256, range=(-4.0, 4.0))
print(f"Max error vs numpy: {int(cp.max(cp.abs(gpu_hist - cp.array(np_hist))))}")

# Benchmark vs naive
naive_code = r"""
extern "C" __global__
void histogram_naive(const float* data, int* hist,
                      int n, int n_bins, float lo, float hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = (int)((data[i] - lo) / (hi - lo) * n_bins);
        bin = min(max(bin, 0), n_bins - 1);
        atomicAdd(&hist[bin], 1);   // global atomic — high contention
    }
}
"""
naive_kernel = cp.RawKernel(naive_code, 'histogram_naive')

hist_naive = cp.zeros(256, dtype=cp.int32)
threads = 256; blocks = (N + threads - 1) // threads

def run_naive():
    hist_naive[:] = 0
    naive_kernel((blocks,), (threads,), (data, hist_naive, N, 256, -4.0, 4.0))

def run_priv():
    return gpu_histogram(data)

# Time both
for name, fn in [("Naive global atomic", run_naive),
                  ("Privatised shared",   run_priv)]:
    for _ in range(3): fn()
    cp.cuda.Stream.null.synchronize()
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(20): fn()
    end.record(); end.synchronize()
    t = cp.cuda.get_elapsed_time(start, end) / 20
    print(f"{name:25s}: {t:.2f} ms")
```

---

## 11.5 Compare-and-swap — the universal atomic primitive

`atomicCAS` (compare-and-swap) is the fundamental atomic from which all
others can be built. It is also the basis of lock-free data structures.

```
atomicCAS(addr, expected, desired):
  Atomically:
    old = *addr
    if old == expected:
        *addr = desired
    return old

  If return == expected: you won the CAS — you changed the value
  If return != expected: someone else changed it first — retry
```

### Building a spin lock with CAS

```python
lock_code = r"""
__device__ void lock(int* mutex) {
    // Spin until we acquire the lock
    // atomicCAS returns 0 (unlocked) only when we set it to 1 (locked)
    while (atomicCAS(mutex, 0, 1) != 0) {
        // busy wait — GPU thread spins
        // In practice: use sparingly, only for very short critical sections
    }
}

__device__ void unlock(int* mutex) {
    atomicExch(mutex, 0);   // set back to 0 (unlocked)
}

extern "C" __global__
void critical_section_demo(float* shared_val, float* data, int* mutex, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        lock(mutex);
        // Critical section — only one thread at a time
        *shared_val = *shared_val * data[i] + data[i];
        unlock(mutex);
    }
}
"""
# Note: spin locks on GPU are dangerous — use only for trivially short
# critical sections. Long critical sections = deadlock or severe serialisation.
# Prefer atomicAdd/Max/Min or privatisation over spin locks.
```

### Lock-free stack with CAS

```python
stack_code = r"""
// Lock-free stack using linked list in pre-allocated node pool
struct Node {
    float  value;
    int    next;   // index into node pool (-1 = null)
};

extern "C" __global__
void stack_push(Node* pool, int* top, float* values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int   new_node = i;             // each thread owns one node in pool
    pool[new_node].value = values[i];

    int old_top;
    do {
        old_top = *top;             // read current top
        pool[new_node].next = old_top;
        // CAS: if top hasn't changed, set top = new_node
    } while (atomicCAS(top, old_top, new_node) != old_top);
    // If CAS fails (another thread changed top), retry with new top value
}

extern "C" __global__
void stack_pop(Node* pool, int* top, float* output, int* out_count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int old_top, new_top;
    do {
        old_top = *top;
        if (old_top == -1) return;    // stack empty
        new_top = pool[old_top].next;
        // CAS: if top hasn't changed, pop it
    } while (atomicCAS(top, old_top, new_top) != old_top);

    int slot = atomicAdd(out_count, 1);
    output[slot] = pool[old_top].value;
}
"""
```

---

## 11.6 Warp vote functions

Warp vote functions allow threads within a warp to communicate boolean
conditions without shared memory or atomics. They are warp-synchronous
(no `__syncthreads()` needed) and execute in a single cycle.

```c
unsigned __ballot_sync(unsigned mask, int predicate);
// Returns a 32-bit mask where bit i is set if thread i's predicate is true
// All threads in mask must call simultaneously

int __any_sync(unsigned mask, int predicate);
// Returns 1 if ANY thread in mask has predicate == true

int __all_sync(unsigned mask, int predicate);
// Returns 1 if ALL threads in mask have predicate == true

int __popc(unsigned mask);
// Population count — number of set bits (how many threads had true)
```

### Use case 1 — Count threads passing a predicate (warp-efficient)

```python
vote_code = r"""
extern "C" __global__
void count_positive_warp(const float* data, int* count, int n) {
    /*
     * Count elements > 0 using warp vote instead of atomicAdd per thread.
     * Reduces global atomics from N to N/32.
     */
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int pred = (i < n) && (data[i] > 0.0f);

    // Ballot: each thread reports its predicate, get 32-bit mask
    unsigned mask   = 0xffffffff;
    unsigned ballot = __ballot_sync(mask, pred);

    // Only lane 0 of each warp does the atomic add
    // popcount(ballot) = number of threads in this warp with pred=true
    if ((threadIdx.x % 32) == 0) {
        atomicAdd(count, __popc(ballot));
    }
}
"""

import cupy as cp
import numpy as np

vote_kernel = cp.RawKernel(vote_code, 'count_positive_warp',
                            options=('--std=c++14',))

N     = 10_000_000
data  = cp.random.randn(N, dtype=cp.float32)
count = cp.zeros(1, dtype=cp.int32)

threads = 256
blocks  = (N + threads - 1) // threads
vote_kernel((blocks,), (threads,), (data, count, N))

gpu_count = int(count[0])
ref_count = int(cp.sum(data > 0))
print(f"GPU count: {gpu_count}  Reference: {ref_count}  Match: {gpu_count == ref_count}")
```

Speedup: 32× fewer `atomicAdd` calls vs one per thread.

### Use case 2 — Early exit when all warps satisfy condition

```python
early_exit_code = r"""
extern "C" __global__
void find_any_negative(const float* data, int* found, int n) {
    /*
     * Set found=1 and exit as soon as any thread finds a negative value.
     * Uses __any_sync for warp-level early exit.
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if already found by another block
    if (*found) return;

    if (i < n) {
        int is_negative = (data[i] < 0.0f);

        // Warp vote: did ANY thread in this warp find negative?
        if (__any_sync(0xffffffff, is_negative)) {
            atomicExch(found, 1);   // set flag — only one atomic per warp
        }
    }
}
"""
```

### Use case 3 — Warp-level stream compaction

```python
compact_code = r"""
extern "C" __global__
void warp_compact(const float* inp, float* out,
                   int* out_count, float threshold, int n) {
    /*
     * Keep elements > threshold, write them compactly.
     * Uses ballot to compute intra-warp output offsets without scan kernel.
     */
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;

    int   pred   = (i < n) && (inp[i] > threshold);
    unsigned ballot = __ballot_sync(0xffffffff, pred);
    int   warp_count = __popc(ballot);

    // Compute this thread's output index within the warp
    // Count set bits below this lane (prefix popcount)
    unsigned mask_below = (1u << lane) - 1u;
    int local_idx = __popc(ballot & mask_below);

    // First lane of warp claims a contiguous block of output slots
    int warp_offset = 0;
    if (lane == 0 && warp_count > 0)
        warp_offset = atomicAdd(out_count, warp_count);

    // Broadcast warp_offset to all lanes using shuffle
    warp_offset = __shfl_sync(0xffffffff, warp_offset, 0);

    // Each selected thread writes to its slot
    if (pred)
        out[warp_offset + local_idx] = inp[i];
}
"""
```

This pattern — ballot → popcount → prefix popcount → single atomic per
warp — is 32× more efficient than one atomic per thread and avoids a
full scan kernel. It is used in sparse attention implementations.

---

## 11.7 Memory ordering and `__threadfence()`

Atomic operations on the same address are sequentially consistent.
But what about atomics on *different* addresses? The GPU has a relaxed
memory model — stores from one thread are not guaranteed to be visible
to other threads in any particular order.

```
Thread A:                   Thread B:
  data[i] = result;           while (flag[0] == 0) {} // spin
  __threadfence();            // fence ensures data visible before flag
  atomicAdd(&flag[0], 1);     val = data[i];  // safe after fence
```

`__threadfence()` levels:

```
__threadfence_block()   Ensures memory ops visible to ALL threads in same BLOCK
                        Cheaper than full fence

__threadfence()         Ensures memory ops visible to ALL threads on GPU
                        Use when communicating across blocks via global mem

__threadfence_system()  Ensures visibility to CPU and other GPUs (NVLink)
                        Most expensive — use only for host-device signalling
```

### Global synchronisation via flags (advanced)

```c
// Pattern: block 0 waits for all other blocks to finish phase 1
// before starting phase 2 — poor man's grid-wide sync
__global__ void two_phase_kernel(float* data, int* block_done,
                                  int n_blocks) {
    // Phase 1: all blocks do work
    do_phase1(data);

    __threadfence();   // ensure phase 1 results are globally visible

    // Signal completion
    if (threadIdx.x == 0)
        atomicAdd(block_done, 1);

    // Block 0 waits for everyone
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0)
            while (*block_done < n_blocks) {}  // spin wait
        __syncthreads();  // make sure all threads in block 0 see the signal
        do_phase2(data);  // only block 0 does phase 2
    }
}
// WARNING: This is fragile. Prefer cooperative groups (Ch 19) for grid sync.
```

---

## 11.8 Atomic performance summary

```
Operation          Memory    Latency (no contention)   Contention scaling
────────────────────────────────────────────────────────────────────────────
atomicAdd (int)    Global    ~20 cycles                O(N) for N threads
atomicAdd (float)  Global    ~20 cycles                O(N) for N threads
atomicAdd (int)    Shared    ~5 cycles                 O(threads/SM) — low
atomicCAS          Global    ~20 cycles + retry loop   O(N) retries
__ballot_sync      Register  ~1 cycle                  No contention (warp)
__shfl_sync        Register  ~1 cycle                  No contention (warp)

Decision hierarchy (fastest to slowest):
  1. Warp shuffle / vote   — no memory, register only (~1 cycle)
  2. Shared memory atomic  — on-chip, low contention  (~5 cycles)
  3. L2 atomic (sm_90+)    — cached atomic            (~20 cycles)
  4. Global memory atomic  — off-chip, contention     (20–1000+ cycles)
```

---

## Hands-on exercises

### Exercise 1 — Contention benchmark

Reproduce the contention benchmark from section 11.3. Run it for
1, 4, 16, 64, 256, 1024 bins with N=10M elements. Record the time
and compute the contention penalty ratio vs the 1024-bin case.

```
n_bins    Time (ms)    Relative to 1024 bins
────────────────────────────────────────────
1
4
16
64
256
1024
```

At what number of bins does contention become negligible?

### Exercise 2 — Implement atomicMaxFloat

Implement the float maximum atomic from section 11.2 and verify it:

```python
max_float_code = r"""
__device__ float atomicMaxFloat(float* addr, float val) {
    // Your CAS-loop implementation here
}

extern "C" __global__
void find_max(const float* data, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicMaxFloat(result, data[i]);
}
"""
# Test: does it correctly find the max of a 10M element array?
# Compare to cp.max() as reference
# Measure: how does it compare in speed to cp.max()?
```

### Exercise 3 — Histogram tuning study

Run the privatised histogram from section 11.4 for three bin counts:

```python
for n_bins in [32, 256, 1024]:
    # Run both naive and privatised
    # Record speedup
    # Question: why does privatisation help less for n_bins=1024
    #           than for n_bins=32?
```

Hint: think about what happens when n_bins > threads per block.

### Exercise 4 — Warp ballot stream compaction

Implement and test the `warp_compact` kernel from section 11.6.
Verify it against `cp.compress` or fancy indexing as a reference:

```python
import cupy as cp

N         = 1_000_000
data      = cp.random.randn(N, dtype=cp.float32)
threshold = 0.0

# Your warp_compact result
out_gpu, count_gpu = run_warp_compact(data, threshold)

# Reference
ref = data[data > threshold]

print(f"Count match: {int(count_gpu) == len(ref)}")
print(f"Values match: {cp.allclose(cp.sort(out_gpu[:int(count_gpu)]),
                                    cp.sort(ref))}")
# Benchmark: warp_compact vs cp boolean indexing vs scan-based compact
```

### Exercise 5 — Lock-free vs locked histogram

Implement two histogram versions:
- **Locked**: use a spin lock (from section 11.5) to protect each bin
- **Lock-free**: use direct `atomicAdd` (your privatised version)

Benchmark both for N=5M elements, 256 bins.

Then answer:
- Why is the spin-lock version slower despite doing the same work?
- Under what conditions would a spin lock ever be appropriate on GPU?
- What is the key difference between CPU lock contention and GPU
  lock contention (hint: think about what a stalled GPU warp does
  vs a stalled CPU thread)?

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Race condition | Read-modify-write is NOT atomic by default — lost updates are silent |
| atomicCAS | Universal primitive — builds all other atomics and lock-free structures |
| Contention penalty | 1 bucket, 10M threads = 10M-way serial — O(N) slowdown |
| Privatisation | Shared-memory private copy → merge → 32–1000× less contention |
| `__ballot_sync` | 32-bit predicate mask across warp — single cycle, no memory |
| Warp compaction | ballot → popcount → 1 atomic per warp — 32× fewer global atomics |
| `__threadfence()` | Ensures store ordering across blocks — needed for flag patterns |
| Shared vs global atomic | Shared: ~5 cycles, limited contention; Global: ~20+ cycles, O(N) |
| Spin locks on GPU | Dangerous — spinning warps block the scheduler; prefer atomics |
| Decision hierarchy | Shuffle → shared atomic → L2 atomic → global atomic |

---

## What's next

Chapter 12 covers **cuBLAS, cuDNN and the CUDA library ecosystem** —
when to use highly optimised library kernels instead of writing your own,
how to call cuBLAS GEMM from Python, set up cuDNN convolution descriptors,
and understand the performance gap between hand-written and library kernels
(and why it exists). After Chapter 12 you will have covered all of Phase 3
and be ready for the advanced ML-focused chapters.
