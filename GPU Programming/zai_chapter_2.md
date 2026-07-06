# Chapter 2: CUDA Programming Model — Threads, Blocks, Grids

Now that you've written your first kernel, let's look at the core execution model. Understanding how to map data to threads is the most fundamental skill in CUDA.

## 1. Built-in Variables
When you launch a kernel, the CUDA runtime secretly passes a few variables to every single thread. You use these to figure out what data each thread should touch:

| Variable | Description | Example (Block Size = 256) |
| :--- | :--- | :--- |
| `threadIdx.x` | Thread index inside its block | 0 to 255 |
| `blockIdx.x` | Block index inside the grid | 0 to N |
| `blockDim.x` | Size of the block (number of threads) | 256 |
| `gridDim.x` | Size of the grid (number of blocks) | Depends on data size |

*(Note: These can also be `.y` and `.z` for 2D/3D grids, which we will use for images/matrices).*

## 2. 1D vs 2D Indexing
For a 1D array (like a flat list of numbers), the math is easy: `idx = cuda.grid(1)`.

But what if you are working with a 2D matrix (like an image or a batch of AI embeddings)? 
CUDA allows you to launch 2D grids of 2D blocks. 

**Visualizing a 2D Block:**
Imagine a block of threads as a 2D grid of 16x16 threads.
*   `threadIdx.x` goes from 0 to 15.
*   `threadIdx.y` goes from 0 to 15.

To find the global X and Y coordinates, Numba provides `cuda.grid(2)`, which returns a tuple: `(x, y)`.

To map a 2D thread coordinate to a 1D flattened array (because memory is always physically 1D in the hardware), you use:
`idx = y * matrix_width + x`

## 3. A Crucial Rule: Guarding Bounds
GPUs are designed to process data in chunks of 32 (warps). If your array size isn't a perfect multiple of your block size, you will launch extra threads that have no data to work on.

If those extra threads try to read memory that doesn't exist, your program will crash (Segfault or Out of Bounds). Therefore, **every kernel must have a bounds check**:

```python
if idx < a.size:
    # do work
```

---

## Hands-on Exercise: 2D Matrix Operations

Let's apply this to a classic ML scenario: adding a bias to a 2D batch of activations, or simply adding two matrices together. 

We will write a 2D kernel using Numba.

```python
import os
os.environ["NUMBA_CUDA_ARCH"] = "7.5"

from numba import cuda
import numpy as np
import time

# 1. Write a 2D Kernel
@cuda.jit
def matrix_add_kernel(A, B, C):
    # Get the 2D global thread coordinates (x, y)
    x, y = cuda.grid(2)
    
    # Get the dimensions of the matrix
    # (Numba arrays have .shape just like NumPy)
    height, width = C.shape
    
    # Guard against out-of-bounds
    if x < width and y < height:
        # Flatten the 2D index to 1D to access the memory
        # Row-major order: row_index * width + column_index
        flat_idx = y * width + x
        C.flat[flat_idx] = A.flat[flat_idx] + B.flat[flat_idx]

# 2. Prepare Data (e.g., a batch of 10,000 images, each 784 pixels = 28x28)
# Let's use a manageable size: 5000 x 5000 matrix
rows, cols = 5000, 5000
A_cpu = np.random.random((rows, cols)).astype(np.float32)
B_cpu = np.random.random((rows, cols)).astype(np.float32)
C_cpu = np.zeros_like(A_cpu)

# Move to GPU
A_gpu = cuda.to_device(A_cpu)
B_gpu = cuda.to_device(B_cpu)
C_gpu = cuda.device_array_like(A_cpu)

# 3. Configure the 2D Kernel Launch
# We use 16x16 threads per block (16 * 16 = 256 threads, a standard choice)
threads_per_block = (16, 16)

# Calculate how many blocks we need in X and Y directions
blocks_x = (cols + threads_per_block[0] - 1) // threads_per_block[0]
blocks_y = (rows + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_x, blocks_y)

print(f"Launching Grid: {blocks_per_grid} blocks, {threads_per_block} threads per block")

# 4. Warm-up (Compile the kernel without timing it)
matrix_add_kernel[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)
cuda.synchronize()

# 5. Proper GPU Timing using CUDA Events
start = cuda.event()
end = cuda.event()

start.record() # Start timer on GPU
# Run the kernel 100 times to get a good average
for _ in range(100):
    matrix_add_kernel[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)
end.record() # Stop timer on GPU
end.synchronize() # Wait for GPU to finish

# Calculate time (events record milliseconds)
gpu_time_ms = start.time_till(end) / 100.0
print(f"Average GPU Kernel Time: {gpu_time_ms:.4f} ms")

# 6. Verify correctness
C_cpu_result = A_cpu + B_cpu
C_gpu_result = C_gpu.copy_to_host()
print(f"Max error: {np.max(np.abs(C_gpu_result - C_cpu_result))}")
```

### Your Task:
1. Run this 2D matrix code. Notice how we used `cuda.event()` for timing instead of `time.time()`. This is the standard way to benchmark GPU code!
2. **Concept Check**: If I have a matrix of size `1000 x 1000`, and my `threads_per_block` is `(32, 32)`, how many blocks will be launched in the grid? (Calculate `blocks_x` and `blocks_y`).
3. **Experiment**: Try changing `threads_per_block` to `(8, 8)` and then `(32, 32)`. Does the execution time change? Why do you think 16x16 or 32x32 are usually preferred over 8x8?

## Questions
1. Why 5000 was selected for rows and columns
2. The logic behind the calculation of blocks size
blocks_x = (cols + threads_per_block[0] - 1) // threads_per_block[0]


Great questions! These get to the heart of how a GPU maps work to hardware. Let's break them down.

### 1. Why 5000 for rows and columns?
It was an arbitrary choice to make the dataset large enough to keep the GPU busy for a measurable amount of time. 

In GPU programming, there is overhead just to launch a kernel (sending instructions from the CPU to the GPU). If the dataset is too small (e.g., 10x10), the GPU will finish the math in a nanosecond, and the execution time will be 99% overhead. 

5000 x 5000 = 25,000,000 elements (25 million). At 4 bytes per `float32`, that's 100 MB of data per matrix. This is enough work to make the GPU actually stretch its legs and show a difference between different block configurations. In ML, this is similar to processing a large batch of data (e.g., 25 million pixels or a large embedding layer).

### 2. The Block Size Calculation Logic (Ceiling Division)
This formula is a classic programming trick called **integer ceiling division**. 

**The Problem:**
We have a matrix with `cols = 5000`. We want to put `threads_per_block[0] = 16` threads in a block. 
How many blocks do we need to cover all 5000 columns?

If we do standard division `5000 // 16`, we get `312`. 
But 312 blocks * 16 threads = 4992 threads. We are short 8 threads! The last 8 columns of our matrix would not get processed.

We need a way to "round up" to the next whole block (313 blocks). 

**The Formula:** `(N + D - 1) // D`

Let's plug in the numbers: `(5000 + 16 - 1) // 16`
1. `5000 + 15 = 5015`
2. `5015 // 16 = 313` (since 313 * 16 = 5008)

We launch 313 blocks, which gives us 5008 threads. The first 5000 threads do the work. The last 8 threads hit our **bounds check** (`if x < width:`) and simply do nothing (idle). 

*Why not just use `math.ceil(5000 / 16)`?* 
Because doing floating-point division (`/`) and then converting back to an integer is slower. The `(N + D - 1) // D` trick uses purely integer math, which is faster and avoids floating-point precision errors. You will see this exact formula in almost every CUDA kernel ever written!

---

### Concept Check Question
Before we move to Chapter 3, let's make sure this clicks. 

If you have an array of `1000` elements, and you choose a block size of `32` threads. 
Using the formula `(1000 + 32 - 1) // 32`, how many blocks will you launch? And how many of the threads in the very last block will actually do work (vs hitting the bounds check and idling)?