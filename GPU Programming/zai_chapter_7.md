# Chapter 7: Atomics, Reductions & Scan

## 1. Race Conditions & Atomic Operations
Imagine 100 threads all trying to increment a single counter: `counter += 1`. 
This operation actually consists of three hardware steps:
1. **Read** the current value of `counter` into a register.
2. **Add** 1 to the register.
3. **Write** the register back to `counter`.

If Thread A and Thread B do this at the exact same time, they might both read `5`, both add 1 to get `6`, and both write `6`. The counter should be `7`, but it's `6`. This is a **Race Condition**.

**The Fix**: `cuda.atomic.add(out, 0, val)`
Atomic operations perform the read-add-write cycle as a single, unbreakable (atomic) hardware instruction. No other thread can interrupt it.

**The Cost**: Atomics are *slow* because the hardware has to serialize access. If 1000 threads try to atomic-add to the same address simultaneously, they form a line and execute one by one. 
*Optimization Tip*: In Chapter 6, we used warp shuffles to reduce 32 numbers down to 1, and *then* used a single atomic add per warp. This is much faster than 32 threads doing an atomic add!

## 2. Parallel Scan (Prefix Sum)
A Prefix Sum calculates the running total of an array.
Input:  `[3, 1, 7, 0, 4, 1, 6, 3]`
Output: `[3, 4, 11, 11, 15, 16, 22, 25]`

**Why is this important for ML/AI?**
*   **Attention Masks**: Calculating cumulative sequence lengths for variable-length sequences in Transformers.
*   **Sparse Operations**: Segment sums in Graph Neural Networks (GNNs).
*   **Stream Compaction**: Removing padded tokens from batches.

**The CPU way** is sequential: `out[i] = out[i-1] + in[i]`. This takes `O(N)` time and cannot be parallelized.

**The GPU way** requires a tree-based algorithm. The most famous is the **Hillis-Steele Scan** (Inclusive Scan). It works by doubling the step size in each iteration.

```
Step 1 (offset=1): out[i] = in[i] + in[i-1]
Step 2 (offset=2): out[i] = out[i] + out[i-2]
Step 3 (offset=4): out[i] = out[i] + out[i-4]
```
It takes `O(log N)` steps, and every step can be done in parallel!

---

## Hands-on Exercise: Hillis-Steele Scan

The trickiest part of writing a scan in CUDA is that a thread cannot read a value from Global Memory if another thread is simultaneously updating it. We must use **Double Buffering** (two Shared Memory arrays) and swap them at each step.

```python
import os
os.environ["NUMBA_CUDA_ARCH"] = "7.5"
from numba import cuda
import numpy as np
import math

# We will process the array in blocks of 1024 elements
BLOCK_SIZE = 1024

@cuda.jit
def hillis_steele_scan_kernel(in_arr, out_arr):
    # Double buffer: we need two shared memory arrays
    s_front = cuda.shared.array(BLOCK_SIZE, dtype=np.float32)
    s_back = cuda.shared.array(BLOCK_SIZE, dtype=np.float32)
    
    tid = cuda.threadIdx.x
    # Each block processes a chunk of the array
    block_offset = cuda.blockIdx.x * cuda.blockDim.x
    
    # 1. Load data from Global Memory to Shared Memory (front buffer)
    if block_offset + tid < in_arr.size:
        s_front[tid] = in_arr[block_offset + tid]
    else:
        s_front[tid] = 0.0
    cuda.syncthreads()
    
    # Set up pointers for swapping
    front = s_front
    back = s_back
    
    # 2. The Scan Loop
    step = 1
    while step < BLOCK_SIZE:
        if tid >= step:
            back[tid] = front[tid] + front[tid - step]
        else:
            back[tid] = front[tid] # Just copy if no element to add
            
        cuda.syncthreads() # Wait for all threads to finish writing to back
        
        # Swap buffers! The back buffer becomes the front for the next step
        temp = front
        front = back
        back = temp
        
        step *= 2 # Double the step size (1, 2, 4, 8, 16...)

    # 3. Write final result back to Global Memory
    # (Note: because we swapped, the final data is in 'front')
    if block_offset + tid < in_arr.size:
        out_arr[block_offset + tid] = front[tid]

# --- Prepare Data ---
N = 1024 # Keeping it to 1 block for simplicity of the algorithm
arr_cpu = np.random.random(N).astype(np.float32)
expected = np.cumsum(arr_cpu) # NumPy's CPU prefix sum

arr_gpu = cuda.to_device(arr_cpu)
out_gpu = cuda.device_array(N, dtype=np.float32)

# --- Launch Config ---
threads_per_block = BLOCK_SIZE
blocks_per_grid = 1 # Only 1 block for this example

# Warm up
hillis_steele_scan_kernel[blocks_per_grid, threads_per_block](arr_gpu, out_gpu)
cuda.synchronize()

# Benchmark
start = cuda.event(); end = cuda.event()
start.record()
for _ in range(100):
    hillis_steele_scan_kernel[blocks_per_grid, threads_per_block](arr_gpu, out_gpu)
end.record(); end.synchronize()
gpu_time = start.time_till(end) / 100.0

# Verify
result = out_gpu.copy_to_host()
error = np.max(np.abs(result - expected))

print(f"CPU Cumsum:     {expected[:5]}...")
print(f"GPU Scan:       {result[:5]}...")
print(f"Max Error:      {error:.4e}")
print(f"GPU Time:       {gpu_time:.4f} ms")
```

### Your Task & Analysis:
1.  **Run the code**. Verify that it matches NumPy's `cumsum`.
2.  **Understand Double Buffering**: Look at the `temp = front; front = back; back = temp` swap. Why do we need two buffers? What would happen if we just tried to do `front[tid] = front[tid] + front[tid - step]` all in one array? *(Hint: Thread 5 might update its value, and then Thread 6 tries to read Thread 5's value. Did Thread 5 read the *old* value or the *new* value?)*
3.  **A Note on PyTorch**: In the real world, you rarely write custom scans. PyTorch has `torch.cumsum()`, which uses highly optimized CUDA libraries (like CUB) that can scan arrays of billions of elements across thousands of blocks. But understanding *how* it works under the hood is crucial for AI engineering.
