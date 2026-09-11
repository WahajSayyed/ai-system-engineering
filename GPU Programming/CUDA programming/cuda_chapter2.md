# Chapter 2 — Environment Setup & First Kernel

## Learning objectives

By the end of this chapter you will:

- Have a working CUDA development environment (toolkit, Python bindings)
- Understand the host/device mental model and memory transfer lifecycle
- Write, compile, and run your first GPU kernel three ways: Numba, CuPy, and raw CUDA C
- Measure kernel execution time correctly using CUDA events
- Understand what `nvcc` does and how the compilation pipeline works

---

## 2.1 The host/device mental model

Before installing anything, cement one concept that every piece of CUDA code depends on.

Your system has two processors with **separate memory spaces**:

```
CPU (host)                        GPU (device)
────────────────                  ────────────────
CPU RAM (DDR5)                    VRAM / HBM
  └─ your Python variables          └─ GPU arrays
  └─ numpy arrays                   └─ kernel inputs/outputs

         ◄──── PCIe bus (≈16–32 GB/s) ────►
               (the expensive crossing)
```

Key rules that govern everything:

1. The CPU cannot directly read GPU memory. You must explicitly copy.
2. A GPU kernel cannot access CPU RAM (except with Unified Memory — Chapter 15).
3. Data transfers over PCIe are slow (~10–30 GB/s) compared to GPU memory bandwidth (~2–3 TB/s on H100). Minimize them.
4. The pattern is always: **allocate → copy to device → compute → copy back → free**.

---

## 2.2 Environment setup

### Option A — Google Colab (recommended if you don't have a local GPU)

Colab gives you a free T4 GPU (16 GB VRAM). Everything below is pre-installed.

```python
# Verify GPU is available
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

### Option B — Local installation

Install in this order. Skipping steps causes the most common setup failures.

**Step 1 — CUDA Toolkit**

Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). Pick your OS, architecture, and installer type. For Ubuntu:

```bash
# After running the installer commands from NVIDIA's site:
nvcc --version          # should print release 12.x
nvidia-smi              # should show your GPU and driver version
```

If `nvcc` is missing but `nvidia-smi` works, your driver is installed but not the toolkit. Run the full installer, not just the driver.

**Step 2 — Python packages**

```bash
pip install numba cupy-cuda12x
# Replace cuda12x with your CUDA version (cuda11x, cuda12x, etc.)
# CuPy has separate wheels per CUDA version — mismatch = silent import error
```

**Step 3 — Verify everything**

```python
import numpy as np
import numba
from numba import cuda
import cupy as cp

print(f"Numba version : {numba.__version__}")
print(f"CuPy version  : {cp.__version__}")
print(f"GPU detected  : {cuda.is_available()}")
print(f"Device name   : {cuda.get_current_device().name.decode()}")

# Quick sanity check — allocate a GPU array and bring it back
x_gpu = cp.array([1.0, 2.0, 3.0])
x_cpu = cp.asnumpy(x_gpu)
print(f"Round-trip OK : {x_cpu}")
```

---

## 2.3 The NVCC compilation pipeline

When you write CUDA C, `nvcc` (the NVIDIA CUDA compiler) does several things:

```
your_kernel.cu
      │
      ▼
  nvcc splits the file
      │
      ├──► Host code (C++) ──────► gcc/clang ──► host object
      │
      └──► Device code (CUDA C) ─► PTX (virtual ISA)
                                       │
                                       ▼
                                   SASS (real GPU machine code)
                                       │
                                       ▼
                                   .cubin (embedded in binary)
```

PTX is an intermediate representation — think of it like CUDA bytecode. SASS is the actual GPU instructions for your specific architecture. You rarely need to look at either until Chapter 18, but knowing they exist explains error messages you'll encounter.

For Python (Numba/CuPy), this pipeline runs transparently at import or first call — the JIT compiler handles it. You'll notice a ~1-2 second pause the first time a Numba kernel runs; that's compilation.

---

## 2.4 Your first kernel — vector addition

Vector addition is the GPU equivalent of "Hello World". Every element is independent, so it maps perfectly to parallel execution.

The serial CPU version:

```python
import numpy as np

def add_vectors_cpu(a, b, c):
    for i in range(len(a)):
        c[i] = a[i] + b[i]
```

The mental shift for GPU: instead of a loop, we write what **one thread does for one element**, then launch millions of threads to cover all elements simultaneously.

### Version 1 — Numba `@cuda.jit`

This is the most educational starting point. You write real CUDA-style kernel code in Python.

```python
import numpy as np
from numba import cuda

# ── The kernel ────────────────────────────────────────────────────────────────
@cuda.jit
def add_vectors_gpu(a, b, c):
    # Each thread computes its own index
    i = cuda.grid(1)          # equivalent to: blockIdx.x * blockDim.x + threadIdx.x

    # Guard: don't go out of bounds if array size isn't a multiple of block size
    if i < a.shape[0]:
        c[i] = a[i] + b[i]

# ── Host code (setup + launch) ────────────────────────────────────────────────
N = 1_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.zeros(N, dtype=np.float32)

# Copy to device (host → device transfer)
a_dev = cuda.to_device(a)
b_dev = cuda.to_device(b)
c_dev = cuda.device_array(N, dtype=np.float32)  # allocate only, no copy

# Launch configuration
threads_per_block = 256                                    # must be multiple of 32
blocks_per_grid   = (N + threads_per_block - 1) // threads_per_block  # ceiling division

# Launch the kernel
add_vectors_gpu[blocks_per_grid, threads_per_block](a_dev, b_dev, c_dev)

# Copy result back (device → host transfer)
c_result = c_dev.copy_to_host()

# Verify
np.testing.assert_allclose(c_result, a + b, rtol=1e-5)
print("Correct!")
```

**Three things to study in this code:**

1. `cuda.grid(1)` — this is the global thread index. Each thread calls this and gets a unique integer from 0 to (total threads - 1). This is how each thread knows which element it owns.

2. `if i < a.shape[0]` — the bounds guard. If N=1,000,000 and `threads_per_block=256`, you launch `ceil(1M/256) = 3907` blocks × 256 threads = 1,000,192 threads. The last 192 threads would read past the array without this guard.

3. `(N + threads_per_block - 1) // threads_per_block` — ceiling integer division without floating point. This pattern appears in nearly every CUDA launch. Memorise it.

### Version 2 — CuPy (NumPy-style, fastest to write)

CuPy lets you use GPU arrays with a NumPy-identical API. Under the hood it runs optimised CUDA kernels, but you don't write them:

```python
import cupy as cp
import numpy as np

N = 1_000_000
a_cpu = np.random.rand(N).astype(np.float32)
b_cpu = np.random.rand(N).astype(np.float32)

# Transfer to GPU
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_cpu)

# Compute on GPU — this launches a CUDA kernel transparently
c_gpu = a_gpu + b_gpu

# Transfer back
c_cpu = cp.asnumpy(c_gpu)

np.testing.assert_allclose(c_cpu, a_cpu + b_cpu, rtol=1e-5)
print("Correct!")
```

CuPy is excellent for prototyping and for operations already covered by its library (FFT, linalg, random, etc.). When you need to implement custom operations not in CuPy's library, you reach for Numba or raw CUDA C.

### Version 3 — Raw CUDA C with Python bindings (Numba `cuda.jit` with explicit C string)

For cases where you want to write real CUDA C but still call from Python, CuPy's `RawKernel` lets you embed a C kernel string directly:

```python
import cupy as cp
import numpy as np

# Raw CUDA C kernel as a Python string
kernel_code = r"""
extern "C" __global__
void add_vectors(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""

# Compile the kernel (happens once, cached after)
add_kernel = cp.RawKernel(kernel_code, 'add_vectors')

N = 1_000_000
a_gpu = cp.random.rand(N, dtype=cp.float32)
b_gpu = cp.random.rand(N, dtype=cp.float32)
c_gpu = cp.zeros(N, dtype=cp.float32)

threads_per_block = 256
blocks_per_grid   = (N + threads_per_block - 1) // threads_per_block

# Launch: (grid, block, args)
add_kernel(
    (blocks_per_grid,),
    (threads_per_block,),
    (a_gpu, b_gpu, c_gpu, N)
)

print("Max error:", float(cp.max(cp.abs(c_gpu - (a_gpu + b_gpu)))))
```

This is the style closest to what you'll write in Chapter 13 when building PyTorch CUDA extensions. The CUDA C keywords to notice:

- `__global__` — this function runs on the GPU and is called from the CPU (host)
- `blockIdx.x`, `blockDim.x`, `threadIdx.x` — the three built-in variables that every thread uses to compute its unique index
- There's no loop — the parallelism comes from launching many instances of this function, not from iteration

---

## 2.5 Measuring execution time correctly

**Do not use `time.time()` or `time.perf_counter()` to benchmark GPU kernels.** GPU kernels launch asynchronously — the CPU returns from the launch call before the GPU has finished. You'll time the launch overhead (~microseconds), not the actual compute.

The correct tool is **CUDA events**, which are timestamps recorded on the GPU timeline.

### With Numba:

```python
from numba import cuda
import numpy as np

N = 10_000_000
a_dev = cuda.to_device(np.random.rand(N).astype(np.float32))
b_dev = cuda.to_device(np.random.rand(N).astype(np.float32))
c_dev = cuda.device_array(N, dtype=np.float32)

threads_per_block = 256
blocks_per_grid   = (N + threads_per_block - 1) // threads_per_block

# Create events
start = cuda.event()
end   = cuda.event()

# Record start, launch, record end
start.record()
add_vectors_gpu[blocks_per_grid, threads_per_block](a_dev, b_dev, c_dev)
end.record()

# Synchronize — wait for GPU to finish
end.synchronize()

elapsed_ms = cuda.event_elapsed_time(start, end)
print(f"GPU time: {elapsed_ms:.3f} ms")

# Effective memory bandwidth
bytes_touched = 3 * N * 4  # 3 arrays × N elements × 4 bytes (float32)
bandwidth_gb_s = (bytes_touched / elapsed_ms * 1e-3) / 1e9
print(f"Bandwidth: {bandwidth_gb_s:.1f} GB/s")
```

### With CuPy:

```python
import cupy as cp

N = 10_000_000
a = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)

start = cp.cuda.Event()
end   = cp.cuda.Event()

start.record()
c = a + b
end.record()
end.synchronize()

elapsed_ms = cp.cuda.get_elapsed_time(start, end)
print(f"GPU time: {elapsed_ms:.3f} ms")
```

The bandwidth calculation above is important: vector addition is **memory-bound** (it reads two arrays and writes one, doing almost no compute per element). The bandwidth you measure tells you how close you are to the GPU's theoretical peak — a well-written vector add on an A100 should approach ~2 TB/s.

---

## 2.6 The full lifecycle in one picture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Host (CPU)                                                         │
│                                                                     │
│  1. Allocate CPU arrays (numpy)                                     │
│  2. Allocate GPU arrays (cuda.device_array / cp.zeros)             │
│  3. Copy CPU → GPU  (H2D transfer)                                  │
│                │                                                    │
│                ▼                                                    │
│       [PCIe bus — the bottleneck]                                   │
│                │                                                    │
│                ▼                                                    │
│  ┌─────────────────────────────────┐                               │
│  │  Device (GPU)                   │                               │
│  │  4. kernel<<<grid, block>>>()   │                               │
│  │     threads execute in parallel │                               │
│  │  5. Result lives in GPU memory  │                               │
│  └─────────────────────────────────┘                               │
│                │                                                    │
│                ▼                                                    │
│       [PCIe bus back]                                               │
│                │                                                    │
│                ▼                                                    │
│  6. Copy GPU → CPU  (D2H transfer)                                  │
│  7. Verify / use result                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2.7 Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `CUDA driver version is insufficient` | Driver too old for toolkit | Update driver to match CUDA version |
| `numba.cuda.cudadrv.error.CudaDriverError` | GPU not detected | Check `nvidia-smi`; try rebooting |
| `CuPy import error: cannot open shared object` | CuPy wheel / CUDA version mismatch | `pip install cupy-cuda12x` matching your `nvcc --version` |
| Wrong results, no crash | Missing bounds guard (`if i < n`) | Always add the bounds check |
| Kernel runs instantly (suspiciously fast) | Timed the async launch, not the kernel | Use CUDA events, not wall-clock time |
| `cudaErrorLaunchFailure` | Block size > 1024, or 0 blocks | `threads_per_block` must be in [1, 1024] |

---

## Hands-on exercises

### Exercise 1 — Verify your setup

Run the verification script from section 2.2. Then run the Numba vector addition from section 2.4. Copy the output here so we can confirm everything is working before moving on.

### Exercise 2 — Scale and measure

Modify the Numba vector addition to run for different array sizes and measure bandwidth for each:

```python
import numpy as np
from numba import cuda

# Fill in the measurement loop
for N in [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
    # ... your timing code here
    print(f"N={N:>12,}  time={elapsed_ms:.3f} ms  bandwidth={bw:.1f} GB/s")
```

Questions to answer from your results:
- At what array size does GPU bandwidth peak? Why is it low for small N?
- What is the peak bandwidth you observe, and how does it compare to your GPU's spec (look up `nvidia-smi --query-gpu=memory.total` and compare)?

### Exercise 3 — 2D grid indexing

Extend vector addition to **matrix addition** — two 2D arrays of shape `(M, N)`. The key change is computing a 2D thread index:

```python
@cuda.jit
def add_matrices_gpu(a, b, c):
    row, col = cuda.grid(2)       # 2D version of cuda.grid(1)
    if row < a.shape[0] and col < a.shape[1]:
        c[row, col] = a[row, col] + b[row, col]

# Launch with a 2D grid
M, N = 1024, 1024
threads = (16, 16)                # 16×16 = 256 threads per block
blocks  = (
    (M + threads[0] - 1) // threads[0],
    (N + threads[1] - 1) // threads[1]
)
```

Implement it, verify correctness against NumPy, and measure the bandwidth. Does it match the 1D version for the same number of total elements?

### Exercise 4 — Spot the bugs

Each snippet below has one bug. Identify it and fix it.

**Snippet A:**
```python
@cuda.jit
def buggy_kernel(a, b, c):
    i = cuda.grid(1)
    c[i] = a[i] + b[i]   # what's missing?
```

**Snippet B:**
```python
N = 500_000
threads_per_block = 1024
blocks_per_grid = N // threads_per_block   # what's wrong with this?
```

**Snippet C:**
```python
import time
start = time.perf_counter()
add_vectors_gpu[blocks, threads](a_dev, b_dev, c_dev)
elapsed = time.perf_counter() - start
print(f"Time: {elapsed*1000:.3f} ms")   # why is this wrong?
```

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Host vs device | Separate memory spaces; explicit copies over PCIe |
| Thread index | `i = blockIdx.x * blockDim.x + threadIdx.x` — every thread's identity |
| Bounds guard | Always check `if i < n` — launched threads often exceed array size |
| Launch config | `ceil(N / threads_per_block)` blocks; threads_per_block = multiple of 32, ≤ 1024 |
| Async execution | GPU kernels are async; use CUDA events to time them |
| Memory-bound ops | Vector addition is memory-bound; peak bandwidth ≈ your target |

---

## What's next

Chapter 3 dives deep into **thread indexing and memory spaces** — you'll learn 1D, 2D, and 3D indexing patterns, understand every memory type (global, shared, local, constant, texture), and see why the choice of memory space can make a kernel 10–100× faster or slower.

When you've attempted the exercises, paste your results and we'll go through them before moving on.
