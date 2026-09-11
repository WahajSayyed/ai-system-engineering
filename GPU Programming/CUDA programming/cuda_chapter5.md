# Chapter 5 — Shared Memory & Synchronization

## Learning objectives

By the end of this chapter you will:

- Understand exactly why shared memory produces large speedups over naive global memory kernels
- Implement tiled matrix multiplication from scratch — the canonical shared memory algorithm
- Use `__syncthreads()` correctly and understand what goes wrong without it
- Identify and fix bank conflicts in shared memory
- Measure how close your tiled kernel gets to cuBLAS (the gold standard)
- Know the shared memory / L1 cache configuration options on modern GPUs

---

## 5.1 The core problem — global memory latency dominates naive kernels

Recall from Chapter 3: global memory (HBM) has ~400–800 cycle latency. For a
naive matrix multiplication `C = A × B` of shape `(N, N)`, each output element
`C[i,j]` requires a dot product of row `i` of A and column `j` of B — N
multiply-accumulate operations, each reading from global memory.

```
Naive kernel — what happens per output element:
  C[i,j] = sum over k of A[i,k] * B[k,j]

  For N=1024:
    - 1024 reads from A (global memory, row i)
    - 1024 reads from B (global memory, col j)
    - Total: 2048 global memory reads per output element
    - Output elements: 1024 × 1024 = 1,048,576
    - Total global reads: ~2 BILLION
    - Each read: ~400 cycles
    - Result: completely memory-bound — GPU spends 99% of time waiting
```

The key insight: **A[i,k] is read once for every output element in row i**.
For N=1024, element `A[i,0]` is read 1024 times — once for each column of C.
We are paying 400-cycle global memory cost 1024 times for the same value.

Shared memory fixes this: load A and B tiles into shared memory once (1 global
read per element), then all threads in the block reuse those values from
fast on-chip shared memory (~5 cycles).

---

## 5.2 Tiled matrix multiplication — the algorithm

The idea: divide A and B into square tiles of size `TILE × TILE`. Each block
computes one tile of C. To do so, it iterates over the tiles of A and B,
loading each pair into shared memory and accumulating the partial dot product.

```
Matrix A (M × K)          Matrix B (K × N)
┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
│   │   │   │   │  tile 0 │   │   │   │   │
│ A0│ A1│ A2│ A3│ ──────► │B0 │B1 │B2 │B3 │
│   │   │   │   │         │   │   │   │   │
└───┴───┴───┴───┘         └───┴───┴───┴───┘

Each block loads one tile of A and one tile of B into shared memory,
computes partial C += A_tile × B_tile, then advances to next tile pair.

Total global reads per element:
  Naive:  2N reads (one per k iteration)
  Tiled:  2N/TILE reads (one per tile, TILE elements shared)
  Speedup from memory: TILE× reduction in global reads
```

For `TILE=16`: **16× fewer global memory reads** than the naive version.
For `TILE=32`: **32× fewer**.

---

## 5.3 Implementation — step by step

### Naive baseline (for comparison)

```python
from numba import cuda, float32
import numpy as np

@cuda.jit
def matmul_naive(A, B, C):
    """
    Naive matrix multiply — every element read from global memory every time.
    A: (M, K), B: (K, N), C: (M, N)
    """
    row, col = cuda.grid(2)
    M, K = A.shape
    K2, N = B.shape

    if row < M and col < N:
        acc = float32(0.0)
        for k in range(K):
            acc += A[row, k] * B[k, col]   # both reads from global memory
        C[row, col] = acc
```

### Tiled version with shared memory

```python
from numba import cuda, float32
import numpy as np

TILE_SIZE = 16   # 16×16 = 256 threads per block, 8 warps

@cuda.jit
def matmul_tiled(A, B, C):
    """
    Tiled matrix multiply using shared memory.
    Each block computes a TILE_SIZE × TILE_SIZE tile of C.
    A: (M, K), B: (K, N), C: (M, N)
    """
    # Shared memory tiles — allocated once per block
    A_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    B_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)

    # Global position this thread is responsible for
    row = cuda.blockIdx.y * TILE_SIZE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE_SIZE + cuda.threadIdx.x

    # Local thread position within the tile
    ty = cuda.threadIdx.y
    tx = cuda.threadIdx.x

    M, K = A.shape
    K2, N = B.shape

    acc = float32(0.0)

    # Iterate over tiles along the K dimension
    n_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

    for tile_idx in range(n_tiles):
        # ── Phase 1: Cooperative load from global → shared ────────────────
        # Each thread loads exactly one element of A and one of B

        a_col = tile_idx * TILE_SIZE + tx   # column in A
        b_row = tile_idx * TILE_SIZE + ty   # row in B

        # Bounds check — K may not be a multiple of TILE_SIZE
        if row < M and a_col < K:
            A_tile[ty, tx] = A[row, a_col]
        else:
            A_tile[ty, tx] = float32(0.0)  # padding

        if b_row < K and col < N:
            B_tile[ty, tx] = B[b_row, col]
        else:
            B_tile[ty, tx] = float32(0.0)  # padding

        # ── Phase 2: Synchronise — wait for ALL threads to finish loading ─
        cuda.syncthreads()   # ← CRITICAL: no thread computes before all load

        # ── Phase 3: Compute partial dot product from shared memory ───────
        for k in range(TILE_SIZE):
            acc += A_tile[ty, k] * B_tile[k, tx]  # fast shared memory reads

        # ── Phase 4: Synchronise before next tile load ────────────────────
        cuda.syncthreads()   # ← CRITICAL: no thread loads next tile before
                             #   all threads finish reading current tile

    # Write result to global memory (once per output element)
    if row < M and col < N:
        C[row, col] = acc
```

**The two `syncthreads()` calls serve different purposes:**

1. **After loading** — ensures every thread's shared memory write is visible
   before any thread starts reading. Without this, thread 0 might start
   computing with stale data from the previous tile still in `A_tile[0,0]`.

2. **After computing** — ensures all threads finish reading the current tile
   before any thread overwrites it with the next tile's data. Without this,
   a fast thread could start loading tile `i+1` while a slow thread is still
   reading tile `i`.

### Running and benchmarking

```python
import numpy as np
from numba import cuda
import time

def benchmark_matmul(M=1024, K=1024, N=1024):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    A_dev = cuda.to_device(A)
    B_dev = cuda.to_device(B)
    C_dev = cuda.device_array((M, N), dtype=np.float32)

    threads = (TILE_SIZE, TILE_SIZE)
    blocks  = (
        (N + TILE_SIZE - 1) // TILE_SIZE,
        (M + TILE_SIZE - 1) // TILE_SIZE
    )

    # Warm up (JIT compile)
    matmul_naive[blocks, threads](A_dev, B_dev, C_dev)
    matmul_tiled[blocks, threads](A_dev, B_dev, C_dev)
    cuda.synchronize()

    # Time naive
    start = cuda.event(timing=True)
    end   = cuda.event(timing=True)

    start.record()
    for _ in range(5):
        matmul_naive[blocks, threads](A_dev, B_dev, C_dev)
    end.record()
    end.synchronize()
    t_naive = cuda.event_elapsed_time(start, end) / 5

    # Time tiled
    start.record()
    for _ in range(5):
        matmul_tiled[blocks, threads](A_dev, B_dev, C_dev)
    end.record()
    end.synchronize()
    t_tiled = cuda.event_elapsed_time(start, end) / 5

    # Verify correctness
    C_naive = C_dev.copy_to_host()
    matmul_tiled[blocks, threads](A_dev, B_dev, C_dev)
    C_tiled = C_dev.copy_to_host()
    C_ref   = A @ B
    print(f"Naive vs ref  max err: {np.max(np.abs(C_naive - C_ref)):.2e}")
    print(f"Tiled vs ref  max err: {np.max(np.abs(C_tiled - C_ref)):.2e}")

    # Performance
    flops = 2 * M * K * N
    print(f"\nNaive:  {t_naive:.2f} ms  {flops/t_naive/1e9:.1f} GFLOP/s")
    print(f"Tiled:  {t_tiled:.2f} ms  {flops/t_tiled/1e9:.1f} GFLOP/s")
    print(f"Speedup: {t_naive/t_tiled:.2f}x")

    # Compare with cuBLAS via CuPy
    import cupy as cp
    A_cp = cp.asarray(A)
    B_cp = cp.asarray(B)

    # Warm up
    _ = A_cp @ B_cp
    cp.cuda.Stream.null.synchronize()

    ev_s = cp.cuda.Event(); ev_e = cp.cuda.Event()
    ev_s.record()
    for _ in range(5):
        C_cp = A_cp @ B_cp
    ev_e.record(); ev_e.synchronize()
    t_cublas = cp.cuda.get_elapsed_time(ev_s, ev_e) / 5

    print(f"cuBLAS: {t_cublas:.2f} ms  {flops/t_cublas/1e9:.1f} GFLOP/s")
    print(f"vs cuBLAS: our tiled is {t_cublas/t_tiled*100:.1f}% of cuBLAS speed")

benchmark_matmul()
```

Typical results on a T4 for `1024×1024` float32:

```
Naive:   ~180 ms    ~12 GFLOP/s
Tiled:   ~12 ms    ~180 GFLOP/s   (~15x speedup)
cuBLAS:  ~2 ms    ~1100 GFLOP/s
```

Our tiled kernel gets about 15–20% of cuBLAS performance. The gap comes from
cuBLAS using `TILE_SIZE=128` or larger, double-buffering, register tiling, and
Tensor Cores (Chapter 14). But the principle is identical — we've demonstrated
the core technique.

---

## 5.4 Bank conflicts — diagnosis and fix

Recall from Chapter 3: shared memory has 32 banks. Threads in a warp accessing
the same bank serialize. In tiled matmul, the B_tile access pattern causes
bank conflicts:

```
Accessing B_tile[k, tx] for fixed k, varying tx across warp:
  Thread 0 → B_tile[k, 0]  → Bank 0  ┐
  Thread 1 → B_tile[k, 1]  → Bank 1  │  No conflict — all different banks ✅
  Thread 2 → B_tile[k, 2]  → Bank 2  │
  ...                                  ┘

Accessing A_tile[ty, k] for fixed ty, varying k in the inner loop:
  All threads read A_tile[ty, 0], then A_tile[ty, 1], etc.
  Same row, sequential columns → sequential banks → no conflict ✅

But transposing B before loading can create conflicts in other layouts.
The classic conflict case — column-major access of a square tile:
  Thread 0 → tile[0, tx] → Bank tx
  Thread 1 → tile[1, tx] → Bank tx  ← all threads hit same bank!
  → 16-way or 32-way bank conflict
```

### Fixing bank conflicts with padding

```python
TILE_SIZE = 16
PAD = 1    # one extra column breaks the bank alignment

@cuda.jit
def matmul_tiled_noconflict(A, B, C):
    # +1 padding in second dimension breaks bank conflict pattern
    A_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE + PAD), dtype=float32)
    B_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE + PAD), dtype=float32)

    # Rest of kernel identical — just change shared array declarations
    row = cuda.blockIdx.y * TILE_SIZE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE_SIZE + cuda.threadIdx.x
    ty  = cuda.threadIdx.y
    tx  = cuda.threadIdx.x
    M, K = A.shape
    _, N = B.shape
    acc = float32(0.0)

    for tile_idx in range((K + TILE_SIZE - 1) // TILE_SIZE):
        a_col = tile_idx * TILE_SIZE + tx
        b_row = tile_idx * TILE_SIZE + ty

        A_tile[ty, tx] = A[row, a_col] if (row < M and a_col < K) else float32(0.0)
        B_tile[ty, tx] = B[b_row, col] if (b_row < K and col < N) else float32(0.0)

        cuda.syncthreads()

        for k in range(TILE_SIZE):
            acc += A_tile[ty, k] * B_tile[k, tx]

        cuda.syncthreads()

    if row < M and col < N:
        C[row, col] = acc
```

Why does `+1` help?

```
Without padding — TILE_SIZE=16, 32 banks:
  tile[0][0]  → Bank 0
  tile[0][1]  → Bank 1
  ...
  tile[0][15] → Bank 15
  tile[1][0]  → Bank 16    ← row 1 starts at bank 16
  tile[1][16] → Bank 0     ← column 16 wraps to bank 0

With padding — tile[row][17 columns]:
  tile[0][0]  → Bank 0
  tile[1][0]  → Bank 17    ← different bank than tile[0][0]
  tile[2][0]  → Bank 2     ← (34 % 32 = 2)
  No two rows start on the same bank → no column-access conflicts
```

---

## 5.5 Shared memory configuration

On modern GPUs the on-chip SRAM is split between L1 cache and shared memory.
You can control the split:

```python
# CUDA C — set per kernel
cudaFuncSetAttribute(
    matmul_tiled,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared   // maximise shared memory
);

# Available splits on Ampere (A100):
#   Default: 32 KB shared / 32 KB L1
#   Max shared: 100 KB shared / ~28 KB L1
#   Max L1:     32 KB shared / 96 KB L1
```

For matmul with large tiles (TILE=32), maximising shared memory is beneficial.
For memory-bound kernels with random access patterns, maximising L1 helps more.

In Numba, you can't control this split directly — it requires CUDA C. This is
one of the reasons production matmul kernels are written in CUDA C or Triton
rather than Numba.

---

## 5.6 The syncthreads contract — what goes wrong without it

This is important enough to make explicit with broken examples.

### Bug 1 — Missing syncthreads after load

```python
@cuda.jit
def buggy_no_sync_after_load(A, B, C):
    A_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    B_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    ty, tx = cuda.threadIdx.y, cuda.threadIdx.x
    row = cuda.blockIdx.y * TILE_SIZE + ty
    col = cuda.blockIdx.x * TILE_SIZE + tx
    M, K = A.shape; _, N = B.shape
    acc = float32(0.0)

    for tile_idx in range((K + TILE_SIZE - 1) // TILE_SIZE):
        a_col = tile_idx * TILE_SIZE + tx
        b_row = tile_idx * TILE_SIZE + ty
        A_tile[ty, tx] = A[row, a_col] if (row < M and a_col < K) else float32(0.0)
        B_tile[ty, tx] = B[b_row, col] if (b_row < K and col < N) else float32(0.0)

        # ← syncthreads() MISSING HERE
        # Thread (0,0) might start computing before thread (15,15) has
        # finished writing its element to shared memory.
        # Result: silently wrong answers — no crash, no error.

        for k in range(TILE_SIZE):
            acc += A_tile[ty, k] * B_tile[k, tx]

        cuda.syncthreads()

    if row < M and col < N:
        C[row, col] = acc
```

This is insidious: it often produces *nearly* correct results because threads
within a warp execute in lockstep (so at least 32 threads are synchronized).
It only fails visibly when threads from *different warps* race — which is
non-deterministic. Bugs like this are extremely hard to reproduce.

### Bug 2 — syncthreads inside a conditional

```python
@cuda.jit
def buggy_sync_in_conditional(arr, out):
    tile = cuda.shared.array(256, dtype=float32)
    i    = cuda.grid(1)
    tid  = cuda.threadIdx.x

    tile[tid] = arr[i] if i < arr.shape[0] else float32(0.0)

    if tid < 128:
        cuda.syncthreads()   # ← DEADLOCK
        # Only threads 0-127 reach this barrier.
        # Threads 128-255 never reach it.
        # The block hangs forever — all threads must hit the same syncthreads.

    out[i] = tile[tid]
```

**Rule: `__syncthreads()` must be reached by ALL threads in the block, on ALL
code paths.** Putting it inside an `if` that only some threads take causes a
deadlock. The GPU hangs until the watchdog kills it.

---

## 5.7 Choosing tile size

```
TILE_SIZE   Threads/block   Warps/block   Shared mem used   Notes
──────────────────────────────────────────────────────────────────────
8           64              2             2 × 256 B = 512 B  Low occupancy
16          256             8             2 × 1 KB  = 2 KB   ← sweet spot
32          1024            32            2 × 4 KB  = 8 KB   Max threads
64          4096            ×             Too many threads   ILLEGAL (>1024)
```

For float32, `TILE_SIZE=16` is the standard starting point:
- 256 threads/block → 8 warps → good occupancy
- 2 KB shared memory used → well within the 48 KB default limit
- 16 is the native Tensor Core tile dimension (convenient for Chapter 14)

For float16 with Tensor Cores, `TILE_SIZE=16` maps to the exact `16×16×16`
WMMA fragment size. No coincidence.

---

## Hands-on exercises

### Exercise 1 — Implement and verify tiled matmul

Copy the `matmul_tiled` kernel from section 5.3. Run `benchmark_matmul()`.

Record and report:
- Naive GFLOP/s
- Tiled GFLOP/s
- Speedup vs naive
- % of cuBLAS performance
- Max numerical error vs NumPy reference

### Exercise 2 — Tile size experiment

Modify `TILE_SIZE` to 8, 16, and 32. Run `benchmark_matmul(M=2048, K=2048, N=2048)`
for each. Fill in the table:

```
TILE_SIZE   Time (ms)   GFLOP/s   Speedup vs naive
─────────────────────────────────────────────────
8
16
32
```

Which tile size wins? Does the answer surprise you? (Hint: think about
occupancy — 32×32 = 1024 threads/block = full block, but fewer blocks
can run simultaneously.)

### Exercise 3 — Diagnose a syncthreads bug

Run `buggy_no_sync_after_load` from section 5.6 on a `512×512` matrix.
Compare its output to NumPy. 

Questions to answer:
- Do you see wrong results? Are they consistently wrong or flaky?
- Does the error depend on matrix size? Try `64×64` vs `1024×1024`.
- Why might small matrices give correct results even without syncthreads?

### Exercise 4 — Rectangular matrices

The kernel in section 5.3 assumes square tiles but allows non-square matrices.
Test it with:

```python
benchmark_matmul(M=512, K=2048, N=1024)   # non-square
benchmark_matmul(M=100, K=200,  N=300)    # not multiples of TILE_SIZE
```

Does it give correct results? Where does the bounds-checking guard activate?
Trace through the tile loop manually for one edge-case tile.

### Exercise 5 — Add a third syncthreads and explain why it's wrong

```python
@cuda.jit
def matmul_too_many_syncs(A, B, C):
    A_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    B_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    ty, tx = cuda.threadIdx.y, cuda.threadIdx.x
    row = cuda.blockIdx.y * TILE_SIZE + ty
    col = cuda.blockIdx.x * TILE_SIZE + tx
    M, K = A.shape; _, N = B.shape
    acc = float32(0.0)

    for tile_idx in range((K + TILE_SIZE - 1) // TILE_SIZE):
        a_col = tile_idx * TILE_SIZE + tx
        b_row = tile_idx * TILE_SIZE + ty
        A_tile[ty, tx] = A[row, a_col] if (row < M and a_col < K) else float32(0.0)
        B_tile[ty, tx] = B[b_row, col] if (b_row < K and col < N) else float32(0.0)
        cuda.syncthreads()

        for k in range(TILE_SIZE):
            acc += A_tile[ty, k] * B_tile[k, tx]
            cuda.syncthreads()   # ← added inside inner loop

        cuda.syncthreads()

    if row < M and col < N:
        C[row, col] = acc
```

Questions:
- Does this produce correct results?
- What is the performance impact?
- Why is `syncthreads()` inside the inner loop unnecessary (and expensive)?

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Why tiling works | Reduces global memory reads by TILE_SIZE× — trades 400-cycle for 5-cycle |
| Tile loop | Load A_tile + B_tile → sync → compute partial dot → sync → next tile |
| First syncthreads | After load: ensures all writes to shared mem are visible before reads |
| Second syncthreads | After compute: prevents overwriting tile before all threads finish reading it |
| Bank conflicts | Avoid column-major access; add +1 padding to shared array second dimension |
| Tile size 16 | Sweet spot: 256 threads, 8 warps, 2 KB shared, fits Tensor Core tiles |
| syncthreads in conditional | Deadlock — all threads in block must reach the same syncthreads call |
| vs cuBLAS | Our kernel reaches ~15–20%; gap is tile size, double buffering, Tensor Cores |

---

## What's next

Chapter 6 goes deep on **global memory access patterns and coalescing** — the
other half of memory optimization. You will learn exactly how the memory
controller groups thread accesses into transactions, why stride patterns are
catastrophic, and how to restructure kernels (including transposing B before
matmul) to always achieve coalesced access. Combined with Chapter 5's shared
memory, you will have the two fundamental tools of CUDA optimization.

Paste your exercise results — especially the tile size table from Exercise 2
and the syncthreads bug behavior from Exercise 3 — and we will discuss before
moving on.
