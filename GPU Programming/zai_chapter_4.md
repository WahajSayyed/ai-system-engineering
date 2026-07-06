# Chapter 4: Memory Coalescing & Access Patterns

If a GPU core is faster than its memory, how do we feed it? The answer is **Coalescing**. 

## 1. What is Coalesced Memory Access?
Remember that threads execute in groups of 32, called a **warp**. 
When a warp executes an instruction that reads from Global Memory (VRAM), the GPU doesn't fetch 32 separate numbers. Instead, it fetches data in **chunks of 128 bytes** (32 threads × 4 bytes for a float32).

*   **Coalesced (Good)**: If Thread 0 reads index 0, Thread 1 reads index 1, Thread 2 reads index 2... up to Thread 31 reading index 31, the GPU hardware combines these into a single 128-byte memory transaction. This is 100% efficient.
*   **Uncoalesced (Terrible)**: If Thread 0 reads index 0, Thread 1 reads index 1000, Thread 2 reads index 2000, the GPU cannot combine them. It forces the hardware to issue 32 separate 128-byte memory transactions. It fetches 4096 bytes of data, throws away 4064 bytes, and keeps 128 bytes. This causes a **32x performance penalty**!

## 2. Array of Structures (AoS) vs. Structure of Arrays (SoA)
This is the most common mistake beginners make when moving from CPUs to GPUs.

Imagine you are processing a batch of 3D coordinates (x, y, z) for a 3D model or a point cloud. 

**AoS (Array of Structures) - CPU Friendly, GPU Hostile:**
`[[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]...]`
If you want to do math on all the `x` values, Thread 0 reads `x1` at index 0. Thread 1 reads `x2` at index 3. Thread 2 reads `x3` at index 6. This is a **strided** access pattern. It is completely uncoalesced. The GPU wastes massive bandwidth.

**SoA (Structure of Arrays) - GPU Friendly:**
`x_array = [x1, x2, x3...]`
`y_array = [y1, y2, y3...]`
`z_array = [z1, z2, z3...]`
Now, Thread 0 reads `x_array[0]`, Thread 1 reads `x_array[1]`. Perfectly coalesced!

*(Note: In PyTorch/TensorFlow, tensors are usually laid out correctly for the GPU, but when you write custom kernels, you must be aware of this).*

---

## Hands-on Exercise: Proving the Coalescing Penalty

Let's write two kernels. One accesses memory sequentially (coalesced). The other accesses memory with a stride (uncoalesced). We'll time them to see the massive difference.

```python
import os
os.environ["NUMBA_CUDA_ARCH"] = "7.5"
from numba import cuda
import numpy as np
import time

# 1. Coalesced Kernel: Thread i reads index i
@cuda.jit
def coalesced_kernel(input_array, output_array):
    idx = cuda.grid(1)
    if idx < input_array.size:
        # Perfectly contiguous access
        output_array[idx] = input_array[idx] * 2.0

# 2. Strided Kernel: Thread i reads index i * STRIDE
STRIDE = 32 # Simulating an AoS layout where we only want 1 out of every 32 elements
@cuda.jit
def strided_kernel(input_array, output_array):
    idx = cuda.grid(1)
    # Ensure we don't read out of bounds based on the stride
    if idx * STRIDE < input_array.size:
        # Strided access -> uncoalesced!
        output_array[idx] = input_array[idx * STRIDE] * 2.0

# Prepare Data
N = 10_000_000
a_gpu = cuda.to_device(np.random.random(N, dtype=np.float32))
b_gpu = cuda.device_array(N, dtype=np.float32)

# Kernel Launch Config
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# Warm up
coalesced_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu)
strided_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu)
cuda.synchronize()

# Benchmark Coalesced
start = cuda.event()
end = cuda.event()
start.record()
for _ in range(100):
    coalesced_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu)
end.record()
end.synchronize()
coal_time = start.time_till(end) / 100.0

# Benchmark Strided
start.record()
for _ in range(100):
    strided_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu)
end.record()
end.synchronize()
stride_time = start.time_till(end) / 100.0

print(f"Coalesced Kernel Time:  {coal_time:.4f} ms")
print(f"Strided Kernel Time:    {stride_time:.4f} ms")
print(f"Slowdown factor:        {stride_time/coal_time:.2f}x")
```

### Your Task:
1.  Run this code. You should see the strided kernel is significantly slower.
2.  **Experiment**: Change the `STRIDE` variable to `2`, then `4`, then `64`. Notice how the slowdown factor changes. (A stride of 2 means the hardware fetches a 128-byte chunk but only uses half of it, cutting your effective bandwidth in half).
3.  **ML Context Question**: In PyTorch, when you have a tensor of shape `[Batch_Size, Sequence_Length, Hidden_Size]`, and you write a custom kernel to process the `Hidden_Size` dimension, which dimension must be contiguous in memory to ensure coalesced access? 
