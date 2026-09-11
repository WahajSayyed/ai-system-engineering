# Chapter 6 — Memory Access Patterns & Coalescing

## Learning objectives

By the end of this chapter you will:

- Understand exactly how the GPU memory controller groups thread accesses into transactions
- Know the coalescing rules and predict memory efficiency from access patterns alone
- Measure effective memory bandwidth and compare against your GPU's theoretical peak
- Fix non-coalesced kernels — including the classic matrix transpose problem
- Use `__ldg()` and the read-only cache to improve irregular access patterns
- Profile memory bottlenecks with Nsight Compute

---

## 6.1 How global memory actually works — transactions

The GPU memory controller does not fetch individual bytes or even individual
floats. It fetches **cache lines of 128 bytes** at a time. When a warp of 32
threads issues memory accesses, the hardware looks at all 32 addresses together
and groups them into the minimum number of 128-byte transactions needed to
satisfy all requests.

```
32 threads × 4 bytes (float32) = 128 bytes of data needed

Best case — all 32 addresses are consecutive:
  Thread 0  → address 0
  Thread 1  → address 4
  Thread 2  → address 8
  ...
  Thread 31 → address 124
  ─────────────────────────────────
  Hardware sees: one contiguous 128-byte region
  Transactions:  1       ← COALESCED ✅
  Efficiency:    128/128 = 100%

Worst case — all 32 addresses are in different cache lines:
  Thread 0  → address 0
  Thread 1  → address 128
  Thread 2  → address 256
  ...
  Thread 31 → address 3968
  ─────────────────────────────────
  Hardware sees: 32 different 128-byte regions
  Transactions:  32      ← UNCOALESCED ❌
  Efficiency:    128/(32×128) = 3.1%
```

The ratio of **useful bytes fetched / total bytes transferred** is
**memory efficiency**. Uncoalesced access wastes bandwidth by fetching
cache lines where only 1 of 32 elements is actually used.

---

## 6.2 Access patterns — from perfect to catastrophic

### Pattern 1 — Sequential (perfect coalescing)

```python
@cuda.jit
def sequential_read(arr, out):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        out[i] = arr[i] * 2.0
# Thread 0 reads arr[0], Thread 1 reads arr[1], ...
# 1 transaction per 32 threads → 100% efficiency
```

### Pattern 2 — Stride-2 access (50% efficiency)

```python
@cuda.jit
def stride2_read(arr, out):
    i = cuda.grid(1)
    if i < arr.shape[0] // 2:
        out[i] = arr[i * 2] * 2.0   # every other element
# Thread 0 reads arr[0], Thread 1 reads arr[2], ...
# Elements are in the same cache line but alternating — 2 transactions
# Useful bytes: 64/128 = 50% efficiency
```

### Pattern 3 — Stride-32 access (3% efficiency)

```python
@cuda.jit
def stride32_read(arr, out):
    i = cuda.grid(1)
    if i < arr.shape[0] // 32:
        out[i] = arr[i * 32] * 2.0
# Thread 0 reads arr[0], Thread 1 reads arr[32], ...
# Each thread's address is in a different cache line
# 32 transactions per warp → 3.1% efficiency
```

### Pattern 4 — Random access (worst case)

```python
@cuda.jit
def random_read(arr, idx, out):
    i = cuda.grid(1)
    if i < idx.shape[0]:
        out[i] = arr[idx[i]]   # arbitrary scatter/gather
# Completely unpredictable — up to 32 transactions per warp
# Effective bandwidth: ~3% of peak
```

---

## 6.3 Measuring the impact — bandwidth benchmark

```python
import numpy as np
import cupy as cp

def measure_bandwidth(kernel_fn, n_bytes_accessed, n_warmup=3, n_repeat=20):
    """Measure effective memory bandwidth in GB/s."""
    for _ in range(n_warmup):
        kernel_fn()
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(n_repeat):
        start = cp.cuda.Event()
        end   = cp.cuda.Event()
        start.record()
        kernel_fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))

    t_ms  = np.min(times)   # use minimum — closest to hardware peak
    bw    = n_bytes_accessed / (t_ms * 1e-3) / 1e9
    return bw, t_ms


N = 32_000_000   # 32M floats = 128 MB
a = cp.random.rand(N, dtype=cp.float32)
b = cp.zeros(N, dtype=cp.float32)

# Sequential — measure peak achievable bandwidth
bw_seq, _ = measure_bandwidth(lambda: cp.copyto(b, a), N * 4 * 2)
print(f"Sequential bandwidth:  {bw_seq:.1f} GB/s")

# Strided accesses via CuPy slicing
bw_s2, _ = measure_bandwidth(lambda: cp.copyto(b[::1], a[::2][:N//2]), N * 4)
print(f"Stride-2 bandwidth:    {bw_s2:.1f} GB/s")

# Compare against theoretical peak (check nvidia-smi)
# T4:   300 GB/s theoretical
# A100: 2000 GB/s theoretical
# H100: 3350 GB/s theoretical
```

---

## 6.4 The matrix transpose problem

Matrix transpose is the **classic coalescing case study** because a naive
implementation is perfectly coalesced on reads but completely uncoalesced
on writes — or vice versa. You cannot naively win both.

### Setup

```
Source matrix A (M × N) stored row-major:
  A[0,0]  A[0,1]  A[0,2]  ...  A[0,N-1]
  A[1,0]  A[1,1]  ...
  ...

Destination B (N × M) = transpose of A:
  B[j,i] = A[i,j]
```

### Naive transpose — reads coalesced, writes not

```python
from numba import cuda, float32

@cuda.jit
def transpose_naive(A, B):
    """
    B[col, row] = A[row, col]
    Reads  from A: coalesced   (threads read consecutive columns → consecutive addresses)
    Writes to  B: UNCOALESCED  (threads write to consecutive rows of B → strided in memory)
    """
    row, col = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:
        B[col, row] = A[row, col]
```

Why are the writes uncoalesced?

```
A is row-major: A[row, 0], A[row, 1], ... A[row, N-1] are consecutive.
  → Reading A[row, col] with col varying across threads = consecutive = coalesced ✅

B is also row-major: B[0, col], B[1, col], ... are NOT consecutive.
  B[col, row] with col varying across threads:
    Thread 0 writes B[0,   row] → offset = 0 * M + row
    Thread 1 writes B[1,   row] → offset = 1 * M + row
    Thread 2 writes B[2,   row] → offset = 2 * M + row
    Stride between consecutive threads = M elements = M * 4 bytes
    For M=1024: stride = 4096 bytes = 32 cache lines between threads
    → 32 transactions per warp ❌
```

### Tiled transpose — fix both with shared memory

Use shared memory as a staging area: read coalesced from A into a shared
tile, then write coalesced to B from a transposed view of that tile.

```python
TILE = 32

@cuda.jit
def transpose_tiled(A, B):
    """
    Coalesced reads from A and coalesced writes to B,
    using shared memory to perform the transpose on-chip.
    """
    # +1 padding avoids bank conflicts on the column read
    tile = cuda.shared.array((TILE, TILE + 1), dtype=float32)

    # Input tile position (reading from A)
    x = cuda.blockIdx.x * TILE + cuda.threadIdx.x   # column in A
    y = cuda.blockIdx.y * TILE + cuda.threadIdx.y   # row in A

    ty = cuda.threadIdx.y
    tx = cuda.threadIdx.x

    # Phase 1: Coalesced read from A into shared memory tile
    # Threads in a warp have same ty, varying tx → consecutive columns → coalesced
    if y < A.shape[0] and x < A.shape[1]:
        tile[ty, tx] = A[y, x]   # row-major read from A ✅ coalesced

    cuda.syncthreads()

    # Output tile position (writing to B) — transposed block coordinates
    x_out = cuda.blockIdx.y * TILE + cuda.threadIdx.x   # column in B
    y_out = cuda.blockIdx.x * TILE + cuda.threadIdx.y   # row in B

    # Phase 2: Coalesced write to B from transposed shared memory
    # Read tile[tx, ty] — swapped indices perform the transpose
    # Threads in a warp have same ty, varying tx → consecutive x_out → coalesced
    if y_out < B.shape[0] and x_out < B.shape[1]:
        B[y_out, x_out] = tile[tx, ty]   # ✅ coalesced write


# Run and benchmark
def benchmark_transpose(M=4096, N=4096):
    A = np.random.randn(M, N).astype(np.float32)
    A_dev = cuda.to_device(A)
    B_dev = cuda.device_array((N, M), dtype=np.float32)

    threads = (TILE, TILE)
    blocks  = (
        (N + TILE - 1) // TILE,
        (M + TILE - 1) // TILE
    )

    # Warm up
    transpose_naive[blocks, threads](A_dev, B_dev)
    transpose_tiled[blocks, threads](A_dev, B_dev)
    cuda.synchronize()

    start = cuda.event(timing=True)
    end   = cuda.event(timing=True)

    start.record()
    for _ in range(10):
        transpose_naive[blocks, threads](A_dev, B_dev)
    end.record(); end.synchronize()
    t_naive = cuda.event_elapsed_time(start, end) / 10

    start.record()
    for _ in range(10):
        transpose_tiled[blocks, threads](A_dev, B_dev)
    end.record(); end.synchronize()
    t_tiled = cuda.event_elapsed_time(start, end) / 10

    # Verify
    B_naive = B_dev.copy_to_host()
    transpose_tiled[blocks, threads](A_dev, B_dev)
    B_tiled = B_dev.copy_to_host()
    print(f"Correct: {np.allclose(B_naive, A.T) and np.allclose(B_tiled, A.T)}")

    bytes_moved = M * N * 4 * 2   # read A + write B
    bw_naive = bytes_moved / (t_naive * 1e-3) / 1e9
    bw_tiled = bytes_moved / (t_tiled * 1e-3) / 1e9

    print(f"Naive: {t_naive:.2f} ms  {bw_naive:.1f} GB/s")
    print(f"Tiled: {t_tiled:.2f} ms  {bw_tiled:.1f} GB/s")
    print(f"Speedup: {t_naive / t_tiled:.2f}x")

benchmark_transpose()
```

Expected results on a T4:
```
Naive:  ~8 ms   ~40 GB/s   (well below 300 GB/s peak — uncoalesced writes)
Tiled:  ~1.5 ms ~220 GB/s  (close to peak — both reads and writes coalesced)
Speedup: ~5x
```

---

## 6.5 The `__ldg()` read-only cache

For read-only global memory accesses (data that won't change during the
kernel), the GPU has a separate **read-only cache** (also called the
texture cache or L1 read-only path). It is 32–48 KB per SM and is
independent of the regular L1 data cache.

Using it: in CUDA C, `__ldg()` (load through global cache) routes the
load through this path. In Numba, arrays marked `const` use it
automatically, or you can use `cuda.ldg()`:

```c
// CUDA C
__global__ void scale(const float* __restrict__ inp, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __ldg(&inp[i]) * 2.0f;   // read-only cache path
}
```

```python
# Numba
from numba import cuda

@cuda.jit
def scale_ldg(inp, out):
    i = cuda.grid(1)
    if i < inp.shape[0]:
        out[i] = cuda.ldg(inp, i) * 2.0   # routes through read-only cache
```

When does `__ldg()` help?

```
Helps:
  - Lookup tables accessed by many threads (same address, broadcast)
  - Read-only weight matrices in inference kernels
  - Any data with spatial locality that fits in 32 KB

Does NOT help:
  - Uncoalesced accesses (bandwidth is still wasted)
  - Write paths (read-only cache is for loads only)
  - Data that changes during the kernel
```

---

## 6.6 Coalescing in 2D kernels — the row vs column convention

This is the most common source of accidental uncoalescing in 2D kernels.
A quick rule to check every 2D kernel you write:

```
For a row-major 2D array arr[row, col]:
  arr[row, col] where col = f(threadIdx.x)  → COALESCED   ✅
  arr[row, col] where row = f(threadIdx.x)  → UNCOALESCED ❌

The x-dimension of threadIdx must map to the fastest-varying
(column) index of any row-major array you access.
```

Common mistake — transposed launch config:

```python
# WRONG — threadIdx.x varies along rows (slow dimension)
@cuda.jit
def bad_2d_kernel(arr):
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if row < arr.shape[0] and col < arr.shape[1]:
        arr[row, col] *= 2.0
# Warp threads have same col (threadIdx.y fixed), different row
# → arr[0,col], arr[1,col], arr[2,col]... = column access = uncoalesced ❌

# CORRECT — threadIdx.x varies along columns (fast dimension)
@cuda.jit
def good_2d_kernel(arr):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < arr.shape[0] and col < arr.shape[1]:
        arr[row, col] *= 2.0
# Warp threads have same row (threadIdx.y fixed), different col
# → arr[row,0], arr[row,1], arr[row,2]... = row access = coalesced ✅
```

---

## 6.7 Profiling memory access with Nsight Compute

Nsight Compute exposes two key metrics for memory coalescing:

```
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
  = actual 32-byte sectors fetched from L2/HBM

l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
  = number of warp-level load requests

Memory efficiency = requests / sectors × 100%
  Perfect coalescing: 1 request = 1 sector → 100%
  Worst case:         1 request = 32 sectors → 3.1%
```

Command line profiling:

```bash
# Profile a CUDA executable
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    ./your_program

# For Python scripts
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    python your_script.py
```

In Nsight Compute GUI: look at the **Memory Workload Analysis** section.
The **Global Memory** row shows load/store efficiency as a percentage.
Anything below 80% warrants investigation.

---

## 6.8 Summary of coalescing rules

```
Rule 1 — Alignment
  Access starting addresses should be aligned to 128 bytes.
  cudaMalloc always returns 256-byte aligned memory. ✅ (automatic)

Rule 2 — Consecutiveness
  Warp threads should access consecutive addresses.
  Thread k should access base + k * element_size.

Rule 3 — x-dimension = fast index
  In 2D/3D kernels, threadIdx.x must map to the column (fastest) dimension.

Rule 4 — Avoid strides
  Any stride > 1 between consecutive thread addresses reduces efficiency.
  Stride of S = S transactions instead of 1.

Rule 5 — Use shared memory to fix unavoidable non-coalescing
  If the algorithm requires non-coalesced access (e.g. transpose),
  load into shared memory first (coalesced), then access freely (on-chip).
```

---

## Hands-on exercises

### Exercise 1 — Predict the transaction count

For each access pattern below, predict: number of 128-byte transactions
per warp, and memory efficiency. Then verify with a microbenchmark.

```
a) Thread i reads arr[i]                     (float32, 32 threads)
b) Thread i reads arr[i * 2]                 (stride 2)
c) Thread i reads arr[i * 16]                (stride 16)
d) Thread i reads arr[i * 32]                (stride 32)
e) Thread i reads arr[N - 1 - i]             (reversed — still coalesced?)
f) Thread i reads arr[i] where arr is float64 (8 bytes each — how many transactions?)
```

### Exercise 2 — Bandwidth ladder

Implement and benchmark all four access patterns from section 6.2 on a
32M float32 array. Plot or tabulate the results:

```python
from numba import cuda
import numpy as np

N = 32_000_000

@cuda.jit
def bw_sequential(src, dst):
    i = cuda.grid(1)
    if i < src.shape[0]:
        dst[i] = src[i]

@cuda.jit
def bw_stride2(src, dst):
    i = cuda.grid(1)
    if i < src.shape[0] // 2:
        dst[i] = src[i * 2]

@cuda.jit
def bw_stride32(src, dst):
    i = cuda.grid(1)
    if i < src.shape[0] // 32:
        dst[i] = src[i * 32]

# Implement bw_stride8 and bw_stride16 yourself.
# Measure bandwidth for each. Does it degrade linearly with stride?
```

### Exercise 3 — Fix a non-coalesced kernel

This kernel computes the column mean of a matrix. It is correct but slow:

```python
@cuda.jit
def col_mean_slow(matrix, means):
    """
    matrix: (rows, cols) float32
    means:  (cols,) float32 — mean of each column
    One thread per column.
    """
    col = cuda.grid(1)
    if col < matrix.shape[1]:
        total = float32(0.0)
        for row in range(matrix.shape[0]):
            total += matrix[row, col]   # ← column access — uncoalesced
        means[col] = total / matrix.shape[0]
```

Questions:
- Why is `matrix[row, col]` uncoalesced here? (thread varies over col,
  but the inner loop varies over row — what does that mean for a warp?)
- Rewrite it so the inner loop access is coalesced. Hint: think about
  what dimension each thread should own.
- Measure the speedup. For a `(4096, 4096)` matrix, what do you see?

### Exercise 4 — Verify the +1 bank conflict padding in transpose

Run `transpose_tiled` with and without the `+1` padding on a `4096×4096`
matrix:

```python
# Version A: with padding (as in section 6.4)
tile_padded   = cuda.shared.array((TILE, TILE + 1), dtype=float32)

# Version B: without padding
tile_unpadded = cuda.shared.array((TILE, TILE), dtype=float32)
```

Measure the timing difference. Is it significant? Use Nsight Compute
(if available) to check `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`
and confirm the conflict count drops to zero with padding.

### Exercise 5 — End-to-end: coalesced matmul B access

Revisit Chapter 5's `matmul_tiled`. The B matrix access pattern inside
the tile load is:

```python
B_tile[ty, tx] = B[b_row, col]
# b_row = tile_idx * TILE_SIZE + ty   (varies with threadIdx.y)
# col   = blockIdx.x * TILE + tx      (varies with threadIdx.x)
```

For a warp (fixed ty, varying tx): threads read `B[b_row, col]` with
col varying — this is a row access. Is this coalesced?

Now consider the naive matmul:
```python
acc += A[row, k] * B[k, col]
# For fixed k, varying col across warp: B[k, col] → is this coalesced?
# For fixed k, varying row across warp (if you launched differently)?
```

Write a short analysis (3–5 sentences) of the memory access quality of
both the naive and tiled matmul kernels, for both A and B.

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Cache line = 128 bytes | GPU fetches 128 bytes at a time regardless of how much you use |
| Coalescing | 32 consecutive thread addresses → 1 transaction → 100% efficiency |
| Stride-S access | S transactions per warp → 1/S efficiency |
| x = fast index | threadIdx.x must map to the column (fastest) dimension of row-major arrays |
| Transpose problem | Coalesced reads + uncoalesced writes — fix with shared memory staging |
| +1 padding | Eliminates bank conflicts in shared memory for column-access patterns |
| `__ldg()` | Routes through read-only cache — helps for broadcast reads, not strided |
| Nsight metric | `l1tex` sector/request ratio = coalescing efficiency |
| Fix strategy | If access is unavoidably non-coalesced → load coalesced into shared mem first |

---

## What's next

Chapter 7 covers **parallel reductions and scan** — the algorithmic
primitive behind every GPU sum, max, softmax, and layer norm. You will
implement a tree-based reduction, learn warp shuffle intrinsics
(`__shfl_down_sync`), and build up to a complete prefix scan. These
patterns appear in every non-trivial ML kernel.

Paste your bandwidth ladder results from Exercise 2 and the col_mean
speedup from Exercise 3, and we will review the numbers together before
moving on.
