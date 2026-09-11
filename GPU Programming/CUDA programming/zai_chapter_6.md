# Chapter 6: Warp-Level Programming & Divergence

Up until now, we've treated threads as independent workers. But the hardware doesn't process threads individually. It processes them in groups of 32, called **Warps**.

## 1. Warp Divergence
Remember the SIMT (Single Instruction, Multiple Thread) model: all 32 threads in a warp must execute the *exact same instruction* at the exact same time. 

What happens if you write an `if/else` statement?
```python
if thread_idx % 2 == 0:
    # Do Math A
else:
    # Do Math B
```
The hardware cannot execute Math A and Math B simultaneously. Instead, it executes them sequentially:
1. Threads 0, 2, 4... execute Math A. (Threads 1, 3, 5... are disabled/idle).
2. Threads 1, 3, 5... execute Math B. (Threads 0, 2, 4... are disabled/idle).

This is called **Warp Divergence**. Your warp just did twice the work at 50% capacity. 

**Rule of thumb**: If you must use `if` statements, try to ensure the condition evaluates the same way for the entire warp (e.g., `if block_idx > 5:` is fine because all threads in a block share the same block index).

## 2. Warp-Level Primitives (The Magic)
Because threads in a warp execute in lockstep, they can communicate with each other directly through hardware registers—**without using Shared Memory and without needing `syncthreads()`!**

These are called **Warp Shuffle** instructions. A thread can "shuffle" its value to another thread in the same warp. 
*   `__shfl_sync`: Broadcast a value from a specific lane to all lanes.
*   `__shfl_up_sync`: Get the value from the lane with a lower index (great for prefix sums/scans).
*   `__shfl_down_sync`: Get the value from a lane with a higher index (great for reductions/sums).

*(Note: In Numba, these are accessed via `cuda.shfl_sync`, `cuda.shfl_down_sync`, etc.)*

## 3. The Butterfly Reduction
Imagine we want to sum 32 numbers held by 32 threads in a warp.
A naive way: Thread 1 adds Thread 0's value. Thread 2 adds Thread 1's result... This takes 31 steps and causes massive divergence!

The GPU way is a **Tree (Butterfly) Reduction**:
1. Lane 16 sends its value to Lane 0. Lane 0 adds them.
2. Lane 8 sends to Lane 0. Lane 0 adds them.
3. Lane 4 sends to Lane 0. Lane 0 adds them.
4. Lane 2 sends to Lane 0. Lane 0 adds them.
5. Lane 1 sends to Lane 0. Lane 0 adds them.

In just 5 steps, Lane 0 has the sum of all 32 numbers!

---

## Hands-on Exercise: Warp-Level Reduction

Let's implement this. We will write a kernel that sums an array. Instead of using Shared Memory for the final steps, we will use `cuda.shfl_down_sync`.

```python
import os
os.environ["NUMBA_CUDA_ARCH"] = "7.5"
from numba import cuda
import numpy as np
import math

# Helper function to do a warp-level reduction
# This sums 32 values inside a warp so that Lane 0 holds the final sum
@cuda.jit(device=True)
def warp_sum(val):
    # cuda.shfl_down_sync(mask, val, offset)
    # mask=0xffffffff means all 32 threads participate
    val += cuda.shfl_down_sync(0xffffffff, val, 16)
    val += cuda.shfl_down_sync(0xffffffff, val, 8)
    val += cuda.shfl_down_sync(0xffffffff, val, 4)
    val += cuda.shfl_down_sync(0xffffffff, val, 2)
    val += cuda.shfl_down_sync(0xffffffff, val, 1)
    return val # Only lane 0 has the correct sum, but all lanes return it

@cuda.jit
def reduction_kernel(arr, out):
    # 1. Load data from Global Memory to registers
    idx = cuda.grid(1)
    if idx < arr.size:
        my_val = arr[idx]
    else:
        my_val = 0.0

    # 2. Do Warp Reduction
    # After this, Lane 0 of every warp holds the sum of that warp
    warp_result = warp_sum(my_val)

    # 3. Use Shared Memory to collect the results from each warp
    # (Assuming block size is a multiple of 32 for simplicity)
    s_partial = cuda.shared.array(32, dtype=np.float32) # Max 32 warps per block
    tid = cuda.threadIdx.x
    lane = tid % 32
    warp_id = tid // 32

    # Only Lane 0 of each warp writes to shared memory
    if lane == 0:
        s_partial[warp_id] = warp_result

    # Wait for all warps to write their partial sums
    cuda.syncthreads()

    # 4. Final Reduction: The first warp sums up the partial sums
    if warp_id == 0:
        # How many warps were in this block?
        num_warps = cuda.blockDim.x // 32
        
        # Load the partial sum for this lane (or 0 if out of bounds)
        if lane < num_warps:
            my_val = s_partial[lane]
        else:
            my_val = 0.0

        # Sum the partial sums using the same warp primitive!
        final_block_sum = warp_sum(my_val)

        # Lane 0 of Warp 0 adds the block's sum to the global output
        if lane == 0:
            cuda.atomic.add(out, 0, final_block_sum)

# --- Prepare Data ---
N = 1_000_000
arr_cpu = np.random.random(N, dtype=np.float32).astype(np.float32)
expected_sum = np.sum(arr_cpu)

arr_gpu = cuda.to_device(arr_cpu)
out_gpu = cuda.device_array(1, dtype=np.float32) # Single float to hold the sum

# --- Launch Config ---
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# Warm up
reduction_kernel[blocks_per_grid, threads_per_block](arr_gpu, out_gpu)
cuda.synchronize()

# Benchmark
start = cuda.event(); end = cuda.event()
start.record()
for _ in range(100):
    out_gpu = cuda.device_array(1, dtype=np.float32) # reset
    reduction_kernel[blocks_per_grid, threads_per_block](arr_gpu, out_gpu)
end.record(); end.synchronize()
gpu_time = start.time_till(end) / 100.0

result = out_gpu.copy_to_host()[0]

print(f"CPU Sum:      {expected_sum:.4f}")
print(f"GPU Sum:      {result:.4f}")
print(f"Time:         {gpu_time:.4f} ms")
```

### Your Task & Analysis:
1.  Run the code. You should see the sums match (with tiny floating-point differences).
2.  **Understand the Shuffle**: Look at `val += cuda.shfl_down_sync(0xffffffff, val, 16)`. When Lane 16 executes this, it asks the hardware for the value held by Lane 32 (which doesn't exist). What does it get back? *(Answer: For out-of-bounds shuffles, it just returns its own value, so the addition does nothing, which is exactly what we want).*
3.  **The Power of Registers**: Notice that inside `warp_sum`, we didn't use `syncthreads()` or Shared Memory. We did it entirely in hardware registers. This is the fastest way to do math on a GPU.
