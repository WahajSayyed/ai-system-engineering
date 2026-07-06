# Chapter 3: Memory Hierarchy Fundamentals

If you only remember one thing from this chapter, let it be this: **GPUs are incredibly fast at math, but terribly slow at fetching data.** 

A modern GPU core can do hundreds of math operations in the time it takes to fetch a single number from main memory. Therefore, the secret to GPU programming isn't writing clever math; it's *feeding the beast* so the cores never have to wait.

## 1. The GPU Memory Hierarchy
GPUs have different types of memory, each with a trade-off between Size and Speed.

1.  **Global Memory (VRAM)**
    *   **Size**: Huge (16GB on your T4).
    *   **Speed**: Slow (high latency, ~400-800 clock cycles).
    *   **Scope**: All threads can access it.
    *   *Analogy*: The hard drive of the GPU.
2.  **Shared Memory (`__shared__`)**
    *   **Size**: Tiny (usually 64KB per SM).
    *   **Speed**: Very fast (~20-30 cycles).
    *   **Scope**: Only threads *inside the same block* can share it.
    *   *Analogy*: A shared whiteboard in a small office room. One block can write to it, and all threads in that block can read it.
3.  **Registers**
    *   **Size**: Tiniest (a few KB per SM, divided among threads).
    *   **Speed**: Instant (1 cycle).
    *   **Scope**: Private to a single thread.
    *   *Analogy*: A sticky note on an individual worker's desk.
4.  **L1 / L2 Cache**: Hardware-managed caches that sit between Global Memory and Shared/Registers to automatically speed up repeated reads.

## 2. The PCIe Bottleneck (Host to Device Transfer)
Before the GPU can do any math, the data must travel from CPU RAM to GPU VRAM via the PCIe bus. This is agonizingly slow compared to GPU memory.

**Golden Rule of GPU Programming**: *Minimize data transfers.* 
Never send data to the GPU in a loop, do a tiny calculation, and bring it back. Send it once, do as much work as possible, and bring it back once.

## 3. Querying the Hardware
Let's look at the actual limits of your T4 GPU using Numba.

```python
import os
os.environ["NUMBA_CUDA_ARCH"] = "7.5"
from numba import cuda

# Get the device handle (Device 0 is the first GPU)
device = cuda.get_current_device()

print(f"--- {device.name.decode()} Specs ---")
print(f"Total VRAM (Global Memory): {device.total_memory / 1e9:.2f} GB")
print(f"Max Threads per Block: {device.MAX_THREADS_PER_BLOCK}")
print(f"Max Shared Memory per Block: {device.MAX_SHARED_MEMORY_PER_BLOCK} bytes")
print(f"Max Registers per Block: {device.MAX_REGISTERS_PER_BLOCK}")
print(f"Warp Size: {device.WARP_SIZE}")
```

---

## Hands-on Exercise: Measuring the Transfer Cost

Let's prove that the PCIe bottleneck is real. We will write a script that measures:
1.  How long it takes to do math on the CPU.
2.  How long it takes to transfer data to the GPU.
3.  How long it takes to do math on the GPU.

We'll use a simple operation: multiply an array by 2. We will use Numba's `@vectorize` which is a quick way to write element-wise kernels without manually setting up grids.

```python
import os
os.environ["NUMBA_CUDA_ARCH"] = "7.5"
from numba import cuda, vectorize
import numpy as np
import time

# 1. Prepare Data
N = 50_000_000 # 50 million floats (~200 MB)
a_cpu = np.random.random(N, dtype=np.float32)
b_cpu = np.random.random(N, dtype=np.float32)

# 2. CPU Baseline (warm-up first)
_ = a_cpu * 2.0 
start = time.perf_counter()
c_cpu = a_cpu * 2.0
cpu_time = time.perf_counter() - start
print(f"CPU Math Time: {cpu_time:.5f} seconds")

# 3. GPU Math Only (Data already on GPU)
a_gpu = cuda.to_device(a_cpu)
b_gpu = cuda.to_device(b_cpu)

# @vectorize compiles a scalar function into a CUDA kernel
@vectorize(['float32(float32)'], target='cuda')
def gpu_multiply(x):
    return x * 2.0

# Warm-up the GPU kernel
_ = gpu_multiply(a_gpu)
cuda.synchronize()

# Time JUST the GPU math
start = cuda.event()
end = cuda.event()
start.record()
c_gpu = gpu_multiply(a_gpu)
end.record()
end.synchronize()
gpu_math_time = start.time_till(end) / 1000.0 # convert ms to s
print(f"GPU Math Only Time: {gpu_math_time:.5f} seconds")

# 4. The Full Round-Trip (Transfer + Math + Transfer back)
start = cuda.event()
end = cuda.event()
start.record()

# Transfer CPU -> GPU
a_to_gpu = cuda.to_device(a_cpu)
# Math on GPU
c_temp = gpu_multiply(a_to_gpu)
# Transfer GPU -> CPU
c_result = c_temp.copy_to_host()

end.record()
end.synchronize()
total_gpu_time = start.time_till(end) / 1000.0

print(f"GPU Total Round-Trip Time: {total_gpu_time:.5f} seconds")
print(f"Data Transfer Overhead: {total_gpu_time - gpu_math_time:.5f} seconds")
```

### Your Task & Analysis:
1. Run the code above.
2. Look at the output. You should see that `GPU Math Only Time` is blazingly fast compared to the CPU. 
3. However, the `GPU Total Round-Trip Time` is likely much slower than the CPU! 
4. **Question**: Based on the output, what percentage of the total round-trip time was spent just moving data across the PCIe bus? 

This exercise is the foundation of why we write "Fused Kernels" in deep learning (like FlashAttention)—doing multiple operations on the GPU *before* sending data back to the CPU.