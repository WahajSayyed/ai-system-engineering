# Chapter 8: Occupancy & Launch Configuration

## 1. What is Occupancy?
Imagine a factory (the SM) with 4 assembly lines (Warp Schedulers). To keep the factory running at 100% capacity, you need enough workers (Warps) ready to work.

If you only launch a few threads, the factory is mostly empty. If a worker has to wait for a delivery (Global Memory fetch), the entire assembly line stops. 

**Occupancy** is the ratio of *Active Warps* on an SM to the *Maximum Possible Warps* an SM can hold.
*   **High Occupancy**: The SM has lots of warps. If Warp A stalls waiting for memory, the SM instantly switches to Warp B. This is how a GPU hides its memory latency!

## 2. The Limits of an SM
On your T4 (Turing architecture), each SM has hard limits:
*   **Max Threads per SM**: 1536 (which is 48 warps).
*   **Max Thread Blocks per SM**: 16.
*   **Max Shared Memory per SM**: 64 KB.
*   **Max Registers per SM**: 65,536 (256 KB).

**The Puzzle**: These limits fight against each other.
If you allocate a block size of 1024 threads, you can only fit 1 block on the SM (1536 / 1024 = 1). That gives you 1024 active threads (occupancy = 66%).
If you allocate a block size of 256 threads, you can fit 6 blocks on the SM (6 * 256 = 1536). That gives you 100% occupancy!

## 3. The Hidden Killers: Registers & Shared Memory
When the Numba compiler turns your Python code into GPU machine code, it assigns **registers** to variables. 
*   If a kernel uses 32 registers per thread, 1536 threads * 32 = 49,152 registers. This fits in the 65,536 limit! (100% occupancy).
*   If you write a very complex kernel that uses 64 registers per thread, 1536 * 64 = 98,304. This exceeds the 65,536 limit! The hardware is forced to reduce the number of active threads to 1024. Occupancy drops to 66%.

Similarly, if you allocate 48 KB of Shared Memory per block, you can only fit 1 block on the SM (since the SM only has 64 KB total), drastically reducing occupancy.

---

## Hands-on Exercise: The Occupancy Calculator

Numba provides a tool to query the theoretical occupancy of a compiled kernel. Let's see how block size and shared memory affect it.

We will take a simple kernel and ask Numba how many blocks will fit on an SM.

```python
import os
os.environ["NUMBA_CUDA_ARCH"] = "7.5"
from numba import cuda
import numpy as np

# 1. A simple kernel that uses a little bit of shared memory
@cuda.jit
def example_kernel(arr, out):
    # Allocate some shared memory (e.g., 16 KB)
    s_data = cuda.shared.array(4096, dtype=np.float32) # 4096 * 4 bytes = 16,384 bytes
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    s_data[tid] = arr[bid * cuda.blockDim.x + tid]
    cuda.syncthreads()
    
    out[bid * cuda.blockDim.x + tid] = s_data[tid] * 2.0

# Prepare dummy data
N = 1_000_000
arr_gpu = cuda.to_device(np.random.random(N, dtype=np.float32))
out_gpu = cuda.device_array(N, dtype=np.float32)

# 2. Numba's Occupancy Calculator
# We pass the compiled kernel function and the block size
for block_size in [128, 256, 512, 1024]:
    # Get how many blocks of this size can fit on one SM
    active_blocks = cuda.occupancy.get_active_blocks_per_multiprocessor(
        example_kernel, block_size, 0
    )
    
    # Calculate Occupancy
    max_threads_per_sm = 1536 # T4 Turing limit
    active_threads = active_blocks * block_size
    occupancy = (active_threads / max_threads_per_sm) * 100.0
    
    print(f"Block Size: {block_size:4d} | Active Blocks/SM: {active_blocks:2d} | "
          f"Active Threads/SM: {active_threads:4d} | Occupancy: {occupancy:.1f}%")

# 3. Let's test a kernel that uses TOO MUCH shared memory
@cuda.jit
def heavy_smem_kernel(arr, out):
    # Allocate 48 KB of shared memory (12232 floats * 4 bytes = 48928 bytes)
    s_data = cuda.shared.array(12232, dtype=np.float32)
    # ... kernel work ...

print("\n--- Testing Heavy Shared Memory Kernel ---")
for block_size in [128, 256, 512]:
    # Numba needs to know the dynamic shared memory size (we set 0, it uses the static 48KB)
    active_blocks = cuda.occupancy.get_active_blocks_per_multiprocessor(
        heavy_smem_kernel, block_size, 0
    )
    active_threads = active_blocks * block_size
    occupancy = (active_threads / 1536) * 100.0
    print(f"Block Size: {block_size:4d} | Active Blocks/SM: {active_blocks:2d} | "
          f"Occupancy: {occupancy:.1f}%")
```

### Your Task & Analysis:
1.  **Run the code**. Look at the output for the `example_kernel`.
2.  Notice how `block_size = 128` might give you 8 active blocks, but `block_size = 1024` gives you 1. Yet, they might both result in 100% occupancy (or close to it). 
3.  Look at the `heavy_smem_kernel` output. Because it uses 48 KB of shared memory per block, and the SM only has 64 KB total, how many blocks can fit on the SM at once? (It should be 1). What does that do to the occupancy for small block sizes?
4.  **The Golden Rule**: High occupancy is good, but it is *not* the only metric that matters. A kernel at 50% occupancy that does 10x less memory access will still be faster. We use occupancy as a guideline to pick block sizes (usually 128, 256, or 512 are safe bets).