# Chapter 5: Shared Memory & Tiling

If Global Memory is a slow hard drive, Shared Memory is a tiny, ultra-fast RAM located directly inside the SM (Streaming Multiprocessor). 

**The Golden Rule of Tiling**: Because reading from Global Memory is slow, we load a chunk of data from Global Memory into Shared Memory *once*, do a ton of math on it using all the threads in the block, and then write the final result back to Global Memory.

## 1. Why Shared Memory is Special
1. **Speed**: ~20-30 cycles to access, compared to ~400-800 cycles for Global Memory.
2. **Scope**: It is shared among all threads in a *single block*. Thread Block A cannot see Block B's shared memory.
3. **Manual Management**: Unlike CPU cache which works automatically, the programmer must explicitly load data into shared memory.
4. **Synchronization**: Because threads run concurrently, if Thread 0 loads the data, Thread 5 can't use it until Thread 0 is finished. We must use `cuda.syncthreads()` to build a "fence" so all threads wait for the load to finish.

## 2. The Case Study: Matrix Multiplication (GEMM)
In Deep Learning, Matrix Multiplication (GEMM) is the core of Linear layers and Attention. 

**Naive GEMM Problem**:
To compute one element of output matrix `C`, a thread must read an entire row of `A` and an entire column of `B`.
If `A` and `B` are 1024x1024, that's 2048 reads from Global Memory *per thread*. 
Worse, neighboring threads read the *exact same* data! Thread (0,0) and Thread (0,1) both read the exact same row of `A`. This wastes massive amounts of memory bandwidth.

**Tiled GEMM Solution**:
We divide the matrix into 16x16 "tiles". 
1. A block of 256 threads (16x16) cooperatively loads a 16x16 tile of `A` and a 16x16 tile of `B` into Shared Memory.
2. `cuda.syncthreads()` (Wait for the load to finish!)
3. The threads compute partial dot products using the fast Shared Memory.
4. They loop through all tiles, accumulating the results in a local register.
5. Finally, they write the single result to Global Memory `C`.

---

## Hands-on Exercise: Tiled Matrix Multiplication

This is the "Hello World" of advanced CUDA. Read the comments carefully, as the indexing gets a bit tricky. 

```python
import os
os.environ["NUMBA_CUDA_ARCH"] = "7.5"
from numba import cuda, float32
import numpy as np
import math

# 1. Naive GEMM (For comparison)
@cuda.jit
def naive_gemm(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

# 2. Tiled GEMM
TILE_SIZE = 16

@cuda.jit
def tiled_gemm(A, B, C):
    # Define a 16x16 shared memory array for tiles of A and B
    s_A = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
    s_B = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)

    # Get global thread coordinates
    row, col = cuda.grid(2)
    
    # Local coordinates inside the tile (0 to 15)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Accumulator for the final result
    tmp = 0.0

    # Loop over the tiles of A and B needed to compute this block's output
    num_tiles = math.ceil(A.shape[1] / TILE_SIZE)
    for t in range(num_tiles):
        # 1. Load data into shared memory cooperatively
        # Guard against out-of-bounds if matrix isn't perfectly divisible by 16
        if row < A.shape[0] and (t * TILE_SIZE + tx) < A.shape[1]:
            s_A[ty, tx] = A[row, t * TILE_SIZE + tx]
        else:
            s_A[ty, tx] = 0.0

        if (t * TILE_SIZE + ty) < B.shape[0] and col < B.shape[1]:
            s_B[ty, tx] = B[t * TILE_SIZE + ty, col]
        else:
            s_B[ty, tx] = 0.0

        # 2. Synchronize! Wait for all threads to finish loading the tile
        cuda.syncthreads()

        # 3. Compute partial dot product using shared memory
        for k in range(TILE_SIZE):
            tmp += s_A[ty, k] * s_B[k, tx]

        # 4. Synchronize again! Ensure computation is done before loading the next tile
        cuda.syncthreads()

    # Write final result back to global memory
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp

# --- Prepare Data ---
N = 512
A_cpu = np.random.random((N, N)).astype(np.float32)
B_cpu = np.random.random((N, N)).astype(np.float32)
C_cpu = np.zeros((N, N), dtype=np.float32)

A_gpu = cuda.to_device(A_cpu)
B_gpu = cuda.to_device(B_cpu)
C_naive = cuda.device_array((N, N), dtype=np.float32)
C_tiled = cuda.device_array((N, N), dtype=np.float32)

# --- Launch Config ---
threads_per_block = (TILE_SIZE, TILE_SIZE)
blocks_per_grid = (math.ceil(N / TILE_SIZE), math.ceil(N / TILE_SIZE))

# --- Warm up ---
naive_gemm[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_naive)
tiled_gemm[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_tiled)
cuda.synchronize()

# --- Benchmark Naive ---
start = cuda.event(); end = cuda.event()
start.record()
for _ in range(50):
    naive_gemm[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_naive)
end.record(); end.synchronize()
naive_time = start.time_till(end) / 50.0

# --- Benchmark Tiled ---
start.record()
for _ in range(50):
    tiled_gemm[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_tiled)
end.record(); end.synchronize()
tiled_time = start.time_till(end) / 50.0

# --- Verify ---
expected = A_cpu @ B_cpu
naive_err = np.max(np.abs(C_naive.copy_to_host() - expected))
tiled_err = np.max(np.abs(C_tiled.copy_to_host() - expected))

print(f"Naive GEMM Time: {naive_time:.4f} ms (Error: {naive_err:.4e})")
print(f"Tiled GEMM Time: {tiled_time:.4f} ms (Error: {tiled_err:.4e})")
print(f"Speedup: {naive_time / tiled_time:.2f}x")
```

### Your Task & Analysis:
1.  **Run the code**. You should see a massive speedup (usually 5x to 10x) using Shared Memory!
2.  **The "Aha!" Moment**: Look at the `cuda.syncthreads()` calls. If you comment out the *first* `cuda.syncthreads()`, what conceptually happens? (Thread A might start computing math on `s_A` before Thread B has finished loading the data into it). 
3.  **Experiment**: Try changing `TILE_SIZE` to `8` and `32`. (Note: In Numba, `cuda.shared.array` requires the shape to be a static constant, so you must change the `TILE_SIZE = 8` at the top of the file, but the `threads_per_block` will automatically adapt). See how it affects the speed.