# Chapter 1: Introduction to GPU Computing & Environment Setup

## 1. CPU vs GPU: Latency vs Throughput
To program a GPU effectively, you must unlearn some CPU optimizations.

*   **CPU (Latency-optimized)**: Has a few very powerful cores (e.g., 8-16). It has massive caches (L1, L2, L3) designed to fetch data as quickly as possible for a single thread. It excels at sequential tasks and complex branching (if/else).
*   **GPU (Throughput-optimized)**: Has thousands of weak cores. It has small caches but massive memory bandwidth. It excels at doing the *same math* on *different pieces of data* simultaneously. 

**Analogy**: A CPU is a Ferrari—fast, carries 2 people, gets there quickly. A GPU is a bus—slower top speed, but carries 100 people at once. If you need to move 10,000 people, buses win.

## 2. The GPU Architecture at a Glance
When you write CUDA code, you need a mental model of the hardware.

*   **SM (Streaming Multiprocessor)**: The fundamental building block of a GPU. A T4 has 40 SMs. Think of an SM as a mini-computer inside the GPU.
*   **Cores**: Inside each SM are smaller cores (INT, FP32, FP64, Tensor Cores). A T4 has 2,560 CUDA cores total.
*   **Warp**: The most critical concept. Threads on a GPU are grouped into "warps" of **32 threads**. All 32 threads execute the *exact same instruction* at the exact same time. If they diverge (e.g., an `if` statement where half the threads go true and half go false), both paths are executed sequentially. This is called **warp divergence**.
*   **Memory Hierarchy**: 
    *   *Global Memory*: The big 16GB VRAM. Slow (high latency), but large.
    *   *Shared Memory*: A tiny, ultra-fast scratchpad (usually 64KB) located *inside* each SM. Programmers manage this manually.

## 3. The CUDA Execution Model
In Python, you call a function. In CUDA, you launch a **Kernel**.

A kernel runs on the GPU. When you launch it, you configure a **Grid** of **Thread Blocks**.
*   **Thread**: A single execution unit.
*   **Block**: A group of threads (max 1024 threads per block). Threads in the same block can share memory and synchronize.
*   **Grid**: A group of Blocks. 

**Example**: If I have a 1D array of 10,000 elements, I might launch a grid of 100 blocks, where each block has 100 threads. `100 * 100 = 10,000`.

## 4. Environment Setup (Google Colab / Linux)
Since you are using Colab/Linux, let's get your tools ready.

**Step 1: Verify the GPU**
Open a terminal (or a Colab cell) and run:
```bash
nvidia-smi
```
You should see the T4, Driver Version, and CUDA Version.

**Step 2: Install Python Libraries**
We will primarily use **CuPy** (NumPy drop-in replacement for GPU) and **Numba** (Python JIT compiler that supports CUDA). 
Run this in your Python environment:
```bash
pip install cupy-cuda12x numba numpy torch
```
*(Note: In Colab, these are usually pre-installed, but it's good practice to ensure they are updated).*

---

## 5. Hands-on Exercise: Hello, GPU!

Let's write your first CUDA code in Python. We will do this two ways: using CuPy (high-level, looks exactly like NumPy) and Numba CUDA (lower-level, exposes the Thread/Block/Grid model).

### Part A: CuPy (The "NumPy" way)
CuPy automatically maps array operations to CUDA kernels. This is the easiest way to get started.

```python
import cupy as cp
import numpy as np
import time

# 1. Create data on the CPU (Host)
N = 10_000_000
a_cpu = np.random.rand(N).astype(np.float32)
b_cpu = np.random.rand(N).astype(np.float32)

# 2. Move data to the GPU (Device)
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_gpu)

# 3. Do math on the GPU
start = time.time()
c_gpu = a_gpu + b_gpu  # This automatically launches a CUDA kernel!
cp.cuda.Stream.null.synchronize() # Wait for GPU to finish
gpu_time = time.time() - start

# 4. Move result back to CPU
c_cpu = cp.asnumpy(c_gpu)

# 5. Compare with CPU
start = time.time()
c_cpu_expected = a_cpu + b_cpu
cpu_time = time.time() - start

print(f"GPU Time: {gpu_time:.5f} seconds")
print(f"CPU Time: {cpu_time:.5f} seconds")
print(f"Max error: {np.max(np.abs(c_cpu - c_cpu_expected))}")
```
*Notice Step 2 and Step 4? Data transfer between CPU and GPU is slow. We will learn to optimize this later.*

### Part B: Numba CUDA (The "Close-to-the-metal" way)
Now, let's manually write the kernel. Numba lets us write CUDA kernels in Python using the `@cuda.jit` decorator.

```python
from numba import cuda
import numpy as np
import time

# 1. Write the Kernel
# Think of this as the code that ONE single thread will run
@cuda.jit
def vector_add_kernel(a, b, c):
    # Calculate the global thread index
    # This tells the thread WHICH element of the array to work on
    idx = cuda.grid(1) 
    
    # Guard against out-of-bounds access
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

# 2. Prepare data
N = 10_000_000
a_cpu = np.random.rand(N).astype(np.float32)
b_cpu = np.random.rand(N).astype(np.float32)
c_cpu = np.zeros_like(a_cpu)

# Move to GPU
a_gpu = cuda.to_device(a_cpu)
b_gpu = cuda.to_device(b_cpu)
c_gpu = cuda.device_array_like(a_cpu)

# 3. Configure the Kernel Launch
# We need to decide how many threads per block, and how many blocks
threads_per_block = 256
# Calculate blocks needed to cover all N elements (ceiling division)
blocks_per_grid = (a_cpu.size + (threads_per_block - 1)) // threads_per_block

print(f"Launching {blocks_per_grid} blocks, each with {threads_per_block} threads...")

# 4. Launch the Kernel! (Syntax: kernel[grid, block](args))
start = time.time()
vector_add_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)
cuda.synchronize() # Wait for GPU to finish
gpu_time = time.time() - start

# 5. Copy back and verify
c_cpu = c_gpu.copy_to_host()
print(f"GPU Time: {gpu_time:.5f} seconds")
print(f"First 5 elements: {c_cpu[:5]}")
```

## Explaining cuda.grid(1)
In a CPU, if you write a for loop, one thread does the work sequentially: i = 0, then i = 1, then i = 2, etc.

On a GPU, thousands of threads run the exact same code at the exact same time. But how does Thread #5 know it should work on array index #5?

That's what ``cuda.grid(1)`` does. It calculates the global thread index.
In CUDA, threads are grouped into Blocks, and Blocks are grouped into a Grid.

- threadIdx.x: The thread's local ID inside its block (0 to 255 if block size is 256).
- blockIdx.x: The block's ID inside the grid (0 to N).
- blockDim.x: The size of the block (e.g., 256).

The formula for a thread's global ID is:
- global_id = threadIdx.x + (blockIdx.x * blockDim.x)

Numba's ``cuda.grid(1)`` is just a shortcut function that calculates this exact formula for you in 1D. The (1) tells Numba we are working with a 1-dimensional array.


### Your Task:
1. Run the above code in your environment.
2. **Experiment**: In the Numba example, change `threads_per_block` to `32`, then `128`, then `512`. Notice how the number of blocks automatically adjusts. 
3. **Question to think about**: In `cuda.grid(1)`, what exactly does `idx` equal to for the 5th thread in the 2nd block? (Assuming 1D indexing).