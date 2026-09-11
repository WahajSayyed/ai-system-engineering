# Chapter 3 — Thread Indexing & Memory Spaces

## Learning objectives

By the end of this chapter you will:

- Compute correct thread indices in 1D, 2D, and 3D grids confidently
- Know every CUDA memory space, what it costs, and when to use each
- Use `__shared__` memory to dramatically accelerate a real kernel
- Understand why global memory access patterns matter (coalescing preview)
- Diagnose index bugs by working through the math manually

---

## 3.1 The full thread coordinate system

In Chapter 2 you used `cuda.grid(1)` as a shortcut. Here we unpack exactly what
it does, because understanding the raw variables is essential for debugging and
for writing 2D/3D kernels correctly.

CUDA provides six built-in read-only variables inside every kernel:

```
┌─────────────────────────────────────────────────────────────┐
│  gridDim   (how many blocks in the grid,   as x/y/z)       │
│  blockIdx  (which block this thread is in, as x/y/z)       │
│  blockDim  (how many threads per block,    as x/y/z)       │
│  threadIdx (which thread within the block, as x/y/z)       │
└─────────────────────────────────────────────────────────────┘
```

All four exist in every kernel, whether you use 1D, 2D, or 3D.
Unused dimensions default to 1 (for dim) or 0 (for idx).

### 1D indexing (vectors, flat arrays)

```
Grid:  [ Block 0 ][ Block 1 ][ Block 2 ] ...[ Block gridDim.x-1 ]
Block: [ T0  T1  T2 ... T(blockDim.x-1) ]

Global thread index:
    i = blockIdx.x * blockDim.x + threadIdx.x
```

In Python/Numba:
```python
@cuda.jit
def kernel_1d(arr):
    i = cuda.grid(1)          # shorthand for the formula above
    if i < arr.shape[0]:
        arr[i] *= 2.0
```

In CUDA C:
```c
__global__ void kernel_1d(float* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] *= 2.0f;
}
```

### 2D indexing (matrices, images)

For a matrix of shape `(rows, cols)`, map rows to the y-dimension and columns
to the x-dimension. This convention matches how CUDA hardware accesses memory
most efficiently (more on this in section 3.4).

```
Grid (blockIdx.y, blockIdx.x):
  ┌────────┬────────┬────────┐
  │ (0,0)  │ (0,1)  │ (0,2)  │   ← blockIdx.y = 0
  ├────────┼────────┼────────┤
  │ (1,0)  │ (1,1)  │ (1,2)  │   ← blockIdx.y = 1
  └────────┴────────┴────────┘

Row index:  row = blockIdx.y * blockDim.y + threadIdx.y
Col index:  col = blockIdx.x * blockDim.x + threadIdx.x
```

```python
@cuda.jit
def kernel_2d(matrix):
    row, col = cuda.grid(2)       # returns (row, col)
    if row < matrix.shape[0] and col < matrix.shape[1]:
        matrix[row, col] *= 2.0

# Launch config
threads = (16, 16)                # 256 threads per block — common sweet spot
blocks  = (
    (cols + 15) // 16,            # x-dimension covers columns
    (rows + 15) // 16             # y-dimension covers rows
)
kernel_2d[blocks, threads](matrix_dev)
```

> **Why 16×16?** A 16×16 block = 256 threads = 8 warps. It divides most image
> dimensions cleanly and fits comfortably within the 1024 thread-per-block limit.

### 3D indexing (volumes, video frames, batched ops)

```python
@cuda.jit
def kernel_3d(volume):
    z, y, x = cuda.grid(3)
    if z < volume.shape[0] and y < volume.shape[1] and x < volume.shape[2]:
        volume[z, y, x] *= 2.0

threads = (8, 8, 8)               # 512 threads per block
blocks  = (
    (D + 7) // 8,
    (H + 7) // 8,
    (W + 7) // 8
)
```

The 3D case is common in medical imaging (CT volumes), video processing (batch ×
H × W), and 3D convolutions in deep learning.

### The flat-index trick for 2D matrices in 1D kernels

Sometimes you want to process a 2D matrix but launch a 1D kernel (simpler
launch config, sometimes better occupancy). Convert between flat and 2D indices:

```python
@cuda.jit
def matrix_kernel_1d(matrix):
    # matrix is shape (rows, cols), stored row-major in memory
    flat_idx = cuda.grid(1)
    rows, cols = matrix.shape
    if flat_idx < rows * cols:
        row = flat_idx // cols
        col = flat_idx %  cols
        matrix[row, col] *= 2.0
```

---

## 3.2 Memory spaces — the complete map

This is the most important conceptual map in all of CUDA. Every performance
decision you make connects back to it.

```
Per-thread  ──── Registers      (~1 cycle)    Fastest. Private to each thread.
                 Local memory   (~400 cycles) Register spill — avoid this.

Per-block   ──── Shared memory  (~5 cycles)   Programmable L1 cache. Key to fast kernels.

Per-kernel  ──── Constant mem   (~5 cycles)   Read-only, broadcast to all threads.
                 Texture mem    (~varies)      Read-only, spatial locality cache.

Global      ──── Global memory  (~400 cycles) Main VRAM. Where all your arrays live.

Off-chip    ──── Host (CPU RAM) (~ms)         Needs explicit cudaMemcpy.
```

### Registers

Every thread gets its own private set of registers. All scalar variables you
declare inside a kernel live here by default.

```c
__global__ void example(float* a) {
    float x = a[threadIdx.x];   // x lives in a register — fast
    float y = x * x;            // y lives in a register — fast
    a[threadIdx.x] = y;
}
```

There are a limited number of registers per SM (~65,536 on modern GPUs). If your
kernel uses too many, the compiler **spills** to local memory, which is actually
off-chip and slow. You can check register usage with `--ptxas-options=-v` in nvcc
or Nsight Compute.

### Local memory

Despite the name, local memory is **not fast** — it is thread-private storage in
global memory, used only when registers run out. You never explicitly allocate it;
the compiler decides. Large arrays declared inside a kernel often end up here:

```c
__global__ void risky(float* out) {
    float temp[1024];   // ← probably spills to local (slow global) memory
    // ...
}
```

If Nsight shows high "local memory" usage, your kernel has a spill problem.

### Shared memory

Shared memory is the most important memory space you'll program explicitly.
It is:

- On-chip (inside the SM) — ~5 cycle access latency
- Shared between **all threads in the same block** — enables communication
- Programmer-controlled — you decide what goes in it
- Limited: 48–228 KB per SM depending on GPU (you configure the split with `cudaFuncSetAttribute`)

Declare it with `__shared__` in CUDA C, or `cuda.shared.array` in Numba:

```c
// CUDA C
__global__ void with_shared(float* data, int n) {
    __shared__ float tile[256];       // allocated once per block, shared by all 256 threads

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        tile[threadIdx.x] = data[i];  // each thread loads one element from global → shared

    __syncthreads();                   // wait for ALL threads in block to finish loading

    // now use tile[...] — fast shared memory reads
    if (i < n)
        data[i] = tile[threadIdx.x] * 2.0f;
}
```

```python
# Numba
from numba import cuda, float32

@cuda.jit
def with_shared(data):
    tile = cuda.shared.array(256, dtype=float32)  # static size required at compile time

    i   = cuda.grid(1)
    tid = cuda.threadIdx.x

    if i < data.shape[0]:
        tile[tid] = data[i]

    cuda.syncthreads()   # ← critical: barrier for the whole block

    if i < data.shape[0]:
        data[i] = tile[tid] * 2.0
```

`__syncthreads()` / `cuda.syncthreads()` is a **block-wide barrier** — no thread
passes it until every thread in the block has reached it. Forgetting this causes
race conditions where some threads read shared memory before others have written it.

### Constant memory

64 KB of read-only memory cached aggressively. Perfect for values that every
thread reads identically (filter coefficients, lookup tables, model weights during
inference):

```c
__constant__ float filter[64];     // declared at file scope

// On host, load it once:
cudaMemcpyToSymbol(filter, host_filter, 64 * sizeof(float));
```

When all threads in a warp read the same constant memory address simultaneously,
it costs essentially one cycle (broadcast). If threads read different addresses,
it serializes — so constant memory works best for uniform reads.

### Global memory

Everything starts here. All arrays you allocate with `cudaMalloc` / `cuda.to_device`
/ `cp.array` live in global memory. It is large (GBs) but slow (~400–800 cycles
latency). Every optimization in Chapters 5–12 is ultimately about reducing how
much time your threads spend waiting on global memory.

---

## 3.3 Shared memory in practice — 1D stencil

A **stencil** computation updates each element based on its neighbors.
The 1D version: `out[i] = (in[i-1] + in[i] + in[i+1]) / 3.0`.

This is a great example for shared memory because each input element is read by
**up to three threads** — without shared memory, each read hits slow global memory
three times. With shared memory, each element is loaded from global memory once
and read from fast shared memory up to three times.

```python
from numba import cuda, float32
import numpy as np

BLOCK = 256

@cuda.jit
def stencil_naive(inp, out):
    """Without shared memory — each global read happens up to 3x."""
    i = cuda.grid(1)
    if 0 < i < inp.shape[0] - 1:
        out[i] = (inp[i-1] + inp[i] + inp[i+1]) / 3.0

@cuda.jit
def stencil_shared(inp, out):
    """With shared memory — each element loaded from global once."""
    # +2 for the left and right halo (boundary) elements
    tile = cuda.shared.array(BLOCK + 2, dtype=float32)

    i   = cuda.grid(1)
    tid = cuda.threadIdx.x

    # Load interior elements
    if i < inp.shape[0]:
        tile[tid + 1] = inp[i]

    # Load halo elements (boundary threads fetch neighbors)
    if tid == 0 and i > 0:
        tile[0] = inp[i - 1]           # left halo
    if tid == BLOCK - 1 and i < inp.shape[0] - 1:
        tile[BLOCK + 1] = inp[i + 1]   # right halo

    cuda.syncthreads()

    # Compute using only fast shared memory
    if 0 < i < inp.shape[0] - 1:
        out[i] = (tile[tid] + tile[tid + 1] + tile[tid + 2]) / 3.0


# Run both and compare timing
N   = 10_000_000
inp = np.random.rand(N).astype(np.float32)
out = np.zeros(N, dtype=np.float32)

inp_dev = cuda.to_device(inp)
out_dev = cuda.device_array(N, dtype=np.float32)

blocks = (N + BLOCK - 1) // BLOCK

# Time naive
start = cuda.event(); end = cuda.event()
start.record()
stencil_naive[blocks, BLOCK](inp_dev, out_dev)
end.record(); end.synchronize()
t_naive = cuda.event_elapsed_time(start, end)

# Time shared
start.record()
stencil_shared[blocks, BLOCK](inp_dev, out_dev)
end.record(); end.synchronize()
t_shared = cuda.event_elapsed_time(start, end)

print(f"Naive:  {t_naive:.2f} ms")
print(f"Shared: {t_shared:.2f} ms")
print(f"Speedup: {t_naive/t_shared:.2f}x")
```

On a T4, you typically see 1.5–2.5× speedup. The gain grows with arithmetic
intensity — in Chapter 5 you'll see 5–10× on matrix multiplication.

---

## 3.4 Global memory access patterns — coalescing preview

Even though coalescing is covered deeply in Chapter 6, you need to know one rule
now or your Chapter exercises will have mystery slowdowns.

**The rule:** threads in the same warp should access consecutive memory addresses.

```
Warp threads:  T0    T1    T2    T3  ...  T31
               │     │     │     │         │
               ▼     ▼     ▼     ▼         ▼

Coalesced:    [0]   [1]   [2]   [3]  ... [31]   ← 1 memory transaction
Strided×2:   [0]   [2]   [4]   [6]  ... [62]   ← 2 transactions
Column-major:[0]  [cols] [2×cols]...            ← 32 transactions (worst)
```

For a 2D array stored row-major (C order, which is NumPy's default), accessing
`array[row, col]` where `col` varies across threads (x-dimension) is coalesced.
Accessing `array[row, col]` where `row` varies across threads (and col is fixed)
is uncoalesced — threads jump by `cols` elements in memory.

This is why the convention is: **x-dimension = columns = the fast-varying index**.

---

## 3.5 Choosing the right memory space — decision guide

```
Is the data read-only and identical for all threads?
    YES → constant memory (≤ 64 KB) or texture memory
    NO  ↓

Is the data private to one thread, scalar or small array?
    YES → register (compiler handles this automatically)
    NO  ↓

Is the data shared between threads in the same block,
or loaded from global memory more than once?
    YES → shared memory (__shared__)
    NO  ↓

Default: global memory
    → optimize access pattern for coalescing (Ch 6)
    → consider caching with __ldg() (Ch 6)
```

---

## 3.6 Bank conflicts in shared memory

Shared memory is organized into 32 **banks** (one per warp lane). When all
threads in a warp access elements from different banks, all accesses happen in
parallel — one clock cycle. When two or more threads access the same bank, the
accesses serialize — this is a **bank conflict**.

```
32 shared memory banks:   Bank 0  Bank 1  Bank 2 ...  Bank 31
                           │       │       │             │
Array elements:           [0]    [1]    [2]   ...    [31]
                          [32]   [33]   [34]  ...    [63]

Thread 0 reads [0]  → Bank 0  ┐
Thread 1 reads [1]  → Bank 1  │  No conflicts → 1 cycle
Thread 2 reads [2]  → Bank 2  │
...                            ┘

Thread 0 reads [0]  → Bank 0  ┐
Thread 1 reads [32] → Bank 0  │  2-way conflict → 2 cycles
Thread 2 reads [64] → Bank 0  │  N-way conflict → N cycles
```

The classic fix: add a padding element to the shared memory array.

```c
// Without padding — threads accessing every 32nd element → bank conflicts
__shared__ float tile[32][32];

// With padding (+1 column) — strides the bank mapping, breaks conflicts
__shared__ float tile[32][33];
```

You won't optimize for bank conflicts until Chapter 5 (matrix multiply), but
knowing they exist explains occasional mysterious slowdowns.

---

## 3.7 Memory space reference card

```
┌─────────────────┬──────────┬───────────┬─────────────┬────────────────────────┐
│ Memory          │ Latency  │ Size      │ Scope       │ Declare (CUDA C)       │
├─────────────────┼──────────┼───────────┼─────────────┼────────────────────────┤
│ Register        │ 1 cy     │ ~255/thd  │ Per thread  │ (automatic)            │
│ Local           │ ~400 cy  │ 512 KB    │ Per thread  │ (compiler spill)       │
│ Shared          │ ~5 cy    │ 48–228 KB │ Per block   │ __shared__ float x[]   │
│ Constant        │ ~5 cy*   │ 64 KB     │ All threads │ __constant__ float x[] │
│ Texture         │ varies   │ varies    │ All threads │ texture<float> t       │
│ Global          │ ~400 cy  │ GBs       │ All threads │ float* (passed in)     │
│ Host (CPU RAM)  │ ~ms      │ GBs       │ CPU only    │ (malloc / numpy)       │
└─────────────────┴──────────┴───────────┴─────────────┴────────────────────────┘
* When all threads read same address (broadcast). Different addresses → slower.
```

---

## Hands-on exercises

### Exercise 1 — Index arithmetic by hand

A kernel is launched with:
```python
blocks  = (3, 2)     # gridDim.x=3, gridDim.y=2
threads = (4, 4)     # blockDim.x=4, blockDim.y=4
```

For the thread with `blockIdx = (1, 1)` and `threadIdx = (2, 3)`:

a) What is `row` (the global y-index)?
b) What is `col` (the global x-index)?
c) How many total threads are launched?
d) If you were processing an image of size `(8, 12)`, is this launch config sufficient?

Work it out from the formulas — don't run code yet.

### Exercise 2 — Implement and benchmark 2D shared memory tiling

Implement a kernel that copies a 2D matrix using shared memory as a staging area:

```python
TILE = 16

@cuda.jit
def matrix_copy_tiled(src, dst):
    tile = cuda.shared.array((TILE, TILE), dtype=float32)

    row = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + cuda.threadIdx.x

    # Step 1: load from global → shared
    if row < src.shape[0] and col < src.shape[1]:
        tile[cuda.threadIdx.y, cuda.threadIdx.x] = src[row, col]

    cuda.syncthreads()

    # Step 2: write from shared → global
    if row < dst.shape[0] and col < dst.shape[1]:
        dst[row, col] = tile[cuda.threadIdx.y, cuda.threadIdx.x]
```

Run this for a `(4096, 4096)` float32 matrix.

Then implement a naive version that copies directly without shared memory.
Compare bandwidth. Is there a difference? Why or why not? (Hint: think about
what the L1 cache is already doing for a sequential copy.)

### Exercise 3 — Constant memory for a lookup table

Implement a kernel that applies a 256-element lookup table (LUT) to an array of
uint8 values (pixel intensities, for example). Put the LUT in constant memory.

```c
// CUDA C version (embed with cp.RawKernel)
__constant__ float lut[256];

__global__ void apply_lut(const unsigned char* inp, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = lut[inp[i]];
}
```

```python
# Host side — load the LUT into constant memory before kernel launch
# (with cp.RawKernel you pass constants via a separate cudaMemcpyToSymbol call,
#  or pass the lut array as a kernel argument for simplicity in Python)
```

For the Python-friendly version, pass the LUT as a regular `const float*`
argument and measure: does moving it to a `__constant__` array change performance?
Explain the result in terms of how the warp accesses the LUT.

### Exercise 4 — Spot the race condition

This kernel is supposed to compute a running prefix (each element = sum of all
previous elements). Find the race condition and explain what could go wrong:

```python
@cuda.jit
def buggy_prefix(arr):
    tile = cuda.shared.array(256, dtype=float32)
    tid  = cuda.threadIdx.x
    i    = cuda.grid(1)

    tile[tid] = arr[i]
    # Bug is here — where is the missing syncthreads?

    if tid > 0:
        tile[tid] += tile[tid - 1]   # reads neighbor that may not be written yet

    arr[i] = tile[tid]
```

Fix it with a single added line. Then explain: would the fix be sufficient for a
correct prefix sum, or does the algorithm itself need more passes?

### Exercise 5 — Memory space diagnosis

For each scenario, state which memory space you would use and why:

a) A convolutional filter of shape `(3, 3)` applied to every pixel in an image —
   the filter is the same for every thread.

b) A per-thread accumulator that sums 1000 values in a loop before writing to
   global memory once.

c) A 512-element histogram that all threads in a block cooperatively build
   before one thread writes it out.

d) An 8 GB dataset of float32 values that a kernel processes one element at
   a time, with no reuse.

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Thread index (1D) | `i = blockIdx.x * blockDim.x + threadIdx.x` |
| Thread index (2D) | `row = blockIdx.y * blockDim.y + threadIdx.y`, same for col with x |
| Bounds guard | Always check against array shape — launched threads > array size is normal |
| Shared memory | On-chip, per-block, ~5 cycle latency — the key to fast kernels |
| `__syncthreads()` | Required barrier after writing shared memory before reading it |
| Global memory | Slow (~400 cycles) — minimize reads, coalesce access patterns |
| Constant memory | Fast for broadcast reads — ideal for filter kernels and LUTs |
| Bank conflicts | Avoid by padding shared arrays; `tile[N][33]` instead of `tile[N][32]` |

---

## What's next

Chapter 4 goes deep on **Numba and CuPy** — device arrays, streams, memory
pools, JIT compilation internals, and how to call cuBLAS/cuFFT from Python.
You'll build a small GPU-accelerated signal processing pipeline that ties together
everything from Chapters 2 and 3.

Paste your exercise answers and timing results here and we'll review before
moving to Chapter 4.
