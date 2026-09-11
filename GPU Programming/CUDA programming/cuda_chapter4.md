# Chapter 4 — CUDA from Python: Numba & CuPy Deep Dive

## Learning objectives

By the end of this chapter you will:

- Know when to use Numba vs CuPy vs raw CUDA C — and why
- Write multi-dimensional Numba kernels with full control over launch config
- Use CuPy's full API: device arrays, custom kernels, memory pools, streams
- Profile and compare both libraries on real workloads
- Build a small GPU-accelerated signal processing pipeline combining both

---

## 4.1 The Python GPU stack — choosing your tool

Before diving in, establish a clear decision framework. These tools are
complementary, not competing.

```
Problem type                          Best tool
─────────────────────────────────────────────────────────────────────
NumPy-compatible array ops            CuPy          (drop-in replacement)
Custom parallel logic, full control   Numba @cuda.jit
Production ML kernels, PyTorch ops    CUDA C + pybind11  (Ch 13)
Pure Python, tile-based kernels       Triton             (Ch 17)
─────────────────────────────────────────────────────────────────────

Rule of thumb:
  Can CuPy express it in one line?  →  use CuPy
  Need custom indexing / shared mem? →  use Numba
  Need maximum performance?          →  CUDA C (later chapters)
```

A common real-world pattern: prototype in CuPy, identify bottlenecks,
rewrite hot kernels in Numba or CUDA C.

---

## 4.2 Numba deep dive

### 4.2.1 JIT compilation internals

When you decorate a function with `@cuda.jit`, Numba:

1. Inspects the Python bytecode
2. Infers types from the first call's arguments (specialisation)
3. Compiles to PTX via LLVM
4. Caches the compiled kernel (subsequent calls skip compilation)

```python
from numba import cuda
import numpy as np

@cuda.jit
def my_kernel(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] *= 2.0

# First call: ~1-2 seconds (compilation)
arr = cuda.to_device(np.ones(1_000_000, dtype=np.float32))
my_kernel[1024, 256](arr)

# Second call: microseconds (cached)
my_kernel[1024, 256](arr)
```

**Type specialisation matters.** A kernel compiled for `float32` is a
different compiled artifact from one for `float64`. Calling with the
wrong dtype triggers recompilation:

```python
arr_f32 = cuda.to_device(np.ones(1000, dtype=np.float32))
arr_f64 = cuda.to_device(np.ones(1000, dtype=np.float64))

my_kernel[4, 256](arr_f32)   # compiles version A
my_kernel[4, 256](arr_f64)   # compiles version B (separate kernel)
```

Pre-specialise to avoid surprises in production:

```python
from numba import float32 as nb_f32

@cuda.jit('void(float32[:])')   # explicit signature — fails fast on wrong dtype
def my_kernel_typed(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] *= 2.0
```

### 4.2.2 Device functions — GPU-side helpers

Not everything needs to be a kernel. `@cuda.jit(device=True)` creates a
function that runs on the GPU but can only be called from another kernel
(not from the CPU). Use it to decompose complex kernels:

```python
@cuda.jit(device=True)
def clamp(val, lo, hi):
    """Clamp a value — runs on GPU, called from kernels."""
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

@cuda.jit
def normalize_kernel(arr, lo, hi):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] = clamp(arr[i], lo, hi)
```

Device functions are **inlined** by the compiler — no function call overhead.
They are the GPU equivalent of `inline` C functions.

### 4.2.3 2D kernel — image processing example

Applying a brightness/contrast adjustment to an image:

```python
from numba import cuda, float32, uint8
import numpy as np

@cuda.jit
def brightness_contrast(img_in, img_out, brightness, contrast):
    """
    img_in:  (H, W, 3) uint8
    img_out: (H, W, 3) uint8
    """
    row, col = cuda.grid(2)
    H, W, C = img_in.shape

    if row < H and col < W:
        for c in range(C):
            # Read from global memory
            pixel = float(img_in[row, col, c])

            # Apply: new = contrast * (pixel - 128) + 128 + brightness
            new_val = contrast * (pixel - 128.0) + 128.0 + brightness

            # Clamp to [0, 255] and write back
            new_val = max(0.0, min(255.0, new_val))
            img_out[row, col, c] = uint8(new_val)


# Usage
import numpy as np

H, W = 1080, 1920
img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

img_dev     = cuda.to_device(img)
img_out_dev = cuda.device_array_like(img)

TILE = 16
blocks = ((W + TILE - 1) // TILE,
          (H + TILE - 1) // TILE)
threads = (TILE, TILE)

brightness_contrast[blocks, threads](img_dev, img_out_dev, 20.0, 1.2)
result = img_out_dev.copy_to_host()
```

Note the loop over `C` (channels) inside the kernel. With only 3 channels,
unrolling into the thread dimension isn't worth it — let one thread handle all
three channels for its pixel.

### 4.2.4 Atomic operations in Numba

When multiple threads need to write to the same memory location safely:

```python
from numba import cuda
import numpy as np

@cuda.jit
def histogram_kernel(data, hist, n_bins, min_val, max_val):
    i = cuda.grid(1)
    if i < data.shape[0]:
        # Compute which bin this value falls into
        val    = data[i]
        bin_id = int((val - min_val) / (max_val - min_val) * n_bins)
        bin_id = min(bin_id, n_bins - 1)   # clamp to last bin

        # Atomic add — thread-safe increment
        cuda.atomic.add(hist, bin_id, 1)


N      = 5_000_000
data   = np.random.randn(N).astype(np.float32)
hist   = np.zeros(256, dtype=np.int32)

data_dev = cuda.to_device(data)
hist_dev = cuda.to_device(hist)

threads = 256
blocks  = (N + threads - 1) // threads

histogram_kernel[blocks, threads](data_dev, hist_dev, 256, -4.0, 4.0)
hist_result = hist_dev.copy_to_host()

# Verify against numpy
np_hist, _ = np.histogram(data, bins=256, range=(-4.0, 4.0))
print("Max difference:", np.max(np.abs(hist_result - np_hist)))
```

`cuda.atomic.add` is correct but slow under high contention (many threads
hitting the same bin). Chapter 11 covers the privatisation technique to
fix this.

### 4.2.5 Numba device array API

```python
# Allocate without initialising (fastest — like malloc)
arr = cuda.device_array(1_000_000, dtype=np.float32)

# Allocate matching shape/dtype of an existing array
arr2 = cuda.device_array_like(some_numpy_array)

# Transfer host → device
arr_dev = cuda.to_device(numpy_array)

# Transfer device → host (allocates new numpy array)
numpy_array = arr_dev.copy_to_host()

# Transfer into existing numpy array (avoids allocation)
out = np.empty(1_000_000, dtype=np.float32)
arr_dev.copy_to_host(out)

# Check properties
print(arr_dev.shape)
print(arr_dev.dtype)
print(arr_dev.strides)
```

---

## 4.3 CuPy deep dive

### 4.3.1 The drop-in NumPy replacement

CuPy mirrors the NumPy API almost exactly. Most code works by replacing
`np.` with `cp.`:

```python
import numpy as np
import cupy  as cp

# Everything below runs on the GPU
x  = cp.random.randn(1_000_000, dtype=cp.float32)
y  = cp.fft.fft(x)                          # GPU FFT
z  = cp.linalg.norm(x)                      # GPU norm
idx = cp.argsort(x)                         # GPU sort
m  = cp.random.randn(1000, 1000, dtype=cp.float32)
u, s, vh = cp.linalg.svd(m)                 # GPU SVD

# Move back to CPU
result = cp.asnumpy(z)
```

### 4.3.2 Memory management and the memory pool

By default, `cudaMalloc` (raw GPU allocation) is expensive — it can take
milliseconds. CuPy uses a **memory pool** to amortize this cost: freed
arrays go back into the pool and are reused for future allocations.

```python
import cupy as cp

# Check memory pool stats
pool = cp.get_default_memory_pool()

x = cp.zeros(10_000_000, dtype=cp.float32)   # allocates from pool
del x                                          # returns to pool (not freed)

print(f"Pool used:  {pool.used_bytes()  / 1e6:.1f} MB")
print(f"Pool total: {pool.total_bytes() / 1e6:.1f} MB")

# Free all pooled memory back to OS (useful before large allocations)
pool.free_all_blocks()

# Pinned memory pool (for fast H2D/D2H transfers — Chapter 15)
pinned_pool = cp.get_default_pinned_memory_pool()
```

For tight memory situations, pre-allocate a fixed-size pool:

```python
# Reserve 8 GB upfront to avoid fragmentation
cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
```

### 4.3.3 CuPy custom kernels

Three levels of custom kernel in CuPy, from simple to flexible:

**Level 1 — ElementwiseKernel (per-element, automatic indexing)**

```python
import cupy as cp

# CuPy handles the grid/block/index — you just write the per-element logic
sigmoid = cp.ElementwiseKernel(
    in_params  = 'float32 x',          # input
    out_params = 'float32 y',          # output
    operation  = 'y = 1.0f / (1.0f + expf(-x))',
    name       = 'sigmoid'
)

x = cp.random.randn(1_000_000, dtype=cp.float32)
y = sigmoid(x)
```

**Level 2 — ReductionKernel (parallel reduction)**

```python
# Compute L2 norm: sqrt(sum(x^2))
l2_norm = cp.ReductionKernel(
    in_params    = 'float32 x',
    out_params   = 'float32 y',
    map_expr     = 'x * x',            # applied per element
    reduce_expr  = 'a + b',            # how to combine
    post_map_expr= 'y = sqrtf(a)',     # final transform
    identity     = '0',                # identity element for reduction
    name         = 'l2_norm'
)

x    = cp.array([3.0, 4.0], dtype=cp.float32)
norm = l2_norm(x)
print(norm)   # 5.0
```

**Level 3 — RawKernel (full CUDA C, maximum control)**

```python
import cupy as cp

softmax_kernel_code = r"""
extern "C" __global__
void softmax_row(const float* inp, float* out, int cols) {
    int row = blockIdx.x;
    const float* row_in  = inp + row * cols;
    float*       row_out = out + row * cols;

    // Step 1: find max for numerical stability
    float max_val = row_in[0];
    for (int j = 1; j < cols; j++)
        if (row_in[j] > max_val) max_val = row_in[j];

    // Step 2: compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        row_out[j] = expf(row_in[j] - max_val);
        sum += row_out[j];
    }

    // Step 3: normalise
    for (int j = 0; j < cols; j++)
        row_out[j] /= sum;
}
"""

softmax_kernel = cp.RawKernel(softmax_kernel_code, 'softmax_row')

rows, cols = 1024, 512
inp = cp.random.randn(rows, cols, dtype=cp.float32)
out = cp.zeros_like(inp)

# One block per row
softmax_kernel((rows,), (1,), (inp, out, cols))

# Verify: each row should sum to 1.0
print("Row sums:", cp.allclose(out.sum(axis=1), cp.ones(rows)))
```

### 4.3.4 CuPy streams

Streams let you overlap computation with data transfer — multiple operations
can run concurrently on separate streams:

```python
import cupy as cp
import numpy as np

N = 10_000_000

# Default stream — operations are sequential
a = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)
c = a + b   # waits for a and b to be ready

# Custom streams — overlap independent operations
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

with stream1:
    a = cp.random.rand(N, dtype=cp.float32)
    fft_a = cp.fft.fft(a)              # FFT on stream 1

with stream2:
    b = cp.random.rand(N, dtype=cp.float32)
    fft_b = cp.fft.fft(b)              # FFT on stream 2 — concurrent with stream 1

# Synchronise both before combining
stream1.synchronize()
stream2.synchronize()

result = fft_a + fft_b                 # back on default stream
```

Streams are covered in depth in Chapter 8. For now, know they exist and
that `cp.cuda.Stream()` is the CuPy way to create one.

### 4.3.5 Interoperability — Numba ↔ CuPy

The two libraries share the same GPU memory. A CuPy array can be passed
directly to a Numba kernel without copying:

```python
import cupy as cp
from numba import cuda
import numpy as np

@cuda.jit
def scale_kernel(arr, factor):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] *= factor

# Create array in CuPy
x_cp = cp.ones(1_000_000, dtype=cp.float32)

# Pass directly to Numba kernel — zero copy
threads = 256
blocks  = (x_cp.shape[0] + threads - 1) // threads
scale_kernel[blocks, threads](x_cp, 3.14)

# Result is still a CuPy array
print(x_cp[:5])   # [3.14, 3.14, 3.14, 3.14, 3.14]
```

This zero-copy interop means you can: use CuPy for data loading and
preprocessing, Numba for custom kernels, and PyTorch for model inference
— all on the same GPU memory without any redundant transfers.

---

## 4.4 Timing and benchmarking properly

Always benchmark GPU code with warm-up runs and multiple repetitions:

```python
import cupy as cp
import numpy as np

def benchmark(fn, n_warmup=3, n_repeat=20):
    """
    Benchmark a GPU function correctly:
    - warm up to trigger JIT / memory pool
    - use CUDA events for accurate timing
    - return mean and std in milliseconds
    """
    # Warm-up
    for _ in range(n_warmup):
        fn()
    cp.cuda.Stream.null.synchronize()

    # Timed runs
    times = []
    for _ in range(n_repeat):
        start = cp.cuda.Event()
        end   = cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))

    times = np.array(times)
    print(f"  mean: {times.mean():.3f} ms  std: {times.std():.3f} ms"
          f"  min: {times.min():.3f} ms")
    return times


N = 10_000_000
a = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)

print("Vector addition:")
benchmark(lambda: a + b)

print("FFT:")
benchmark(lambda: cp.fft.fft(a))
```

---

## 4.5 Capstone — GPU signal processing pipeline

Combine everything from Chapters 2–4 into a realistic pipeline:
generate a noisy signal, denoise it with FFT, apply a custom gain
kernel, and compute statistics.

```python
import numpy as np
import cupy  as cp
from numba import cuda, float32

# ── Step 1: Custom gain kernel (Numba) ───────────────────────────────────────
@cuda.jit
def apply_gain_kernel(signal, gain_table, out):
    """
    Apply a per-sample gain from a lookup table.
    gain_table has 1024 entries; index by quantising the signal value.
    """
    i = cuda.grid(1)
    if i < signal.shape[0]:
        # Quantise signal to [0, 1023] for LUT lookup
        val      = signal[i]
        lut_idx  = min(int((val + 4.0) / 8.0 * 1023), 1023)
        lut_idx  = max(lut_idx, 0)
        out[i]   = signal[i] * gain_table[lut_idx]


# ── Step 2: FFT-based denoising (CuPy) ───────────────────────────────────────
def fft_denoise(signal_gpu, cutoff_ratio=0.1):
    """Zero out high-frequency components above cutoff_ratio."""
    N      = len(signal_gpu)
    freqs  = cp.fft.rfft(signal_gpu)
    cutoff = int(len(freqs) * cutoff_ratio)
    freqs[cutoff:] = 0.0 + 0.0j
    return cp.fft.irfft(freqs, n=N)


# ── Pipeline ─────────────────────────────────────────────────────────────────
N = 1_000_000

# Generate noisy signal on CPU, transfer once
t          = np.linspace(0, 1, N, dtype=np.float32)
clean      = np.sin(2 * np.pi * 440 * t)          # 440 Hz tone
noise      = np.random.randn(N).astype(np.float32) * 0.3
noisy      = clean + noise

gain_table = np.clip(np.linspace(0.5, 1.5, 1024), 0, 2).astype(np.float32)

# Single H2D transfer
noisy_gpu      = cp.asarray(noisy)
gain_table_gpu = cp.asarray(gain_table)

# Step A: FFT denoise (CuPy)
start = cp.cuda.Event(); end = cp.cuda.Event()
start.record()
denoised_gpu = fft_denoise(noisy_gpu, cutoff_ratio=0.05)
end.record(); end.synchronize()
print(f"FFT denoise:  {cp.cuda.get_elapsed_time(start, end):.2f} ms")

# Step B: Apply gain (Numba custom kernel)
out_gpu = cp.zeros(N, dtype=cp.float32)
threads = 256
blocks  = (N + threads - 1) // threads

start.record()
apply_gain_kernel[blocks, threads](denoised_gpu, gain_table_gpu, out_gpu)
end.record(); end.synchronize()
print(f"Gain kernel:  {cp.cuda.get_elapsed_time(start, end):.2f} ms")

# Step C: Statistics (CuPy — stays on GPU)
start.record()
mean  = float(cp.mean(out_gpu))
std   = float(cp.std(out_gpu))
rms   = float(cp.sqrt(cp.mean(out_gpu ** 2)))
end.record(); end.synchronize()
print(f"Statistics:   {cp.cuda.get_elapsed_time(start, end):.2f} ms")

# Single D2H transfer (only what you need)
result = cp.asnumpy(out_gpu)

print(f"\nSignal stats — mean: {mean:.4f}  std: {std:.4f}  RMS: {rms:.4f}")
```

The design principle here: **one H2D transfer, one D2H transfer, all
processing on GPU**. Every intermediate result stays in VRAM.

---

## 4.6 Numba vs CuPy — quick reference

```
Feature                     Numba @cuda.jit     CuPy
──────────────────────────────────────────────────────────────────
Custom indexing logic       ✅ full control      ❌ ElementwiseKernel only
Shared memory               ✅ cuda.shared.array ❌ (not exposed in Python)
Atomic operations           ✅ cuda.atomic.add   ❌ (limited)
NumPy API compatibility     ❌ write kernels     ✅ near-complete
Built-in FFT/linalg         ❌                   ✅ cuFFT, cuBLAS wrapped
Memory pool                 ❌ manual            ✅ automatic
Stream support              ✅ cuda.stream()     ✅ cp.cuda.Stream()
PyTorch interop             ✅ (via DLPack)      ✅ (via DLPack)
Compilation overhead        ~1–2s first call    ~0.1s first call
Learning curve              Medium              Low
```

---

## Hands-on exercises

### Exercise 1 — JIT type specialisation

Write a Numba kernel that squares each element. Call it with:
- `float32` array
- `float64` array
- `int32` array

Measure the time of the first call vs second call for each dtype.
How many compiled variants are cached? Use `my_kernel.inspect_types()`
to see Numba's type inference output.

### Exercise 2 — CuPy ElementwiseKernel vs ufunc

Implement ReLU three ways and benchmark all three on a `(10M,)` float32 array:

```python
# Method A: CuPy expression
def relu_cupy(x):
    return cp.maximum(x, 0)

# Method B: ElementwiseKernel
relu_ew = cp.ElementwiseKernel(
    'float32 x', 'float32 y',
    'y = x > 0 ? x : 0',
    'relu_ew'
)

# Method C: Numba kernel (write this yourself)
@cuda.jit
def relu_numba(arr):
    # your implementation
    pass
```

Which is fastest? Which has the lowest first-call overhead? Explain why.

### Exercise 3 — Memory transfer audit

Rewrite this inefficient code to minimise PCIe transfers:

```python
import numpy as np
import cupy as cp

a = np.random.rand(1_000_000).astype(np.float32)

# Inefficient version — count the transfers
a_gpu = cp.asarray(a)
mean  = float(cp.mean(a_gpu))        # D2H here
a_cpu = cp.asnumpy(a_gpu)            # D2H here
a_cpu = a_cpu - mean
a_gpu2 = cp.asarray(a_cpu)          # H2D here
result = cp.std(a_gpu2)
final  = float(result)               # D2H here

# Your task: rewrite so only 1 H2D and 1 D2H transfer occur
# All intermediate ops must stay on GPU
```

Count the transfers in the original, then write the efficient version
and count again.

### Exercise 4 — Custom kernel correctness

Implement a `clip` kernel using `cp.RawKernel` that clamps each element
to `[lo, hi]`. Verify it matches `cp.clip()`:

```python
clip_code = r"""
extern "C" __global__
void clip_kernel(const float* inp, float* out, float lo, float hi, int n) {
    // your implementation
}
"""
# Test with:
x = cp.random.randn(1_000_000, dtype=cp.float32)
# cp.clip(x, -1.0, 1.0) should match your kernel exactly
```

### Exercise 5 — Pipeline optimisation challenge

In the capstone pipeline (section 4.5), add a fourth step:
**normalise the output to zero mean and unit variance**
(i.e. `(x - mean) / std`) using a Numba kernel.

Then answer:
- How many global memory reads does your normalisation kernel do?
- Could you compute mean, std, and normalise in a single kernel pass?
  Why or why not?

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Tool selection | CuPy for array ops; Numba for custom kernels; CUDA C for max perf |
| JIT compilation | First call compiles; type-specialised; use explicit signatures in prod |
| Device functions | `@cuda.jit(device=True)` — GPU helpers, inlined, zero overhead |
| CuPy memory pool | Allocations are reused; call `free_all_blocks()` before large allocs |
| ElementwiseKernel | CuPy shortcut for per-element ops — no indexing boilerplate |
| RawKernel | Full CUDA C in a Python string — maximum flexibility from Python |
| Streams | Independent operation queues — enables compute/transfer overlap |
| Numba ↔ CuPy | Zero-copy interop — share GPU memory across both libraries |
| Benchmarking | Warm up first; use CUDA events; measure min not mean for peak perf |
| Transfer discipline | One H2D, one D2H — all intermediate results stay on GPU |

---

## What's next

Chapter 5 goes deep on **shared memory and synchronisation** — the
technique that unlocks 5–10× speedups over naive global memory kernels.
The centrepiece is tiled matrix multiplication: you will implement it
from scratch, understand every line, and measure how close you get to
cuBLAS performance.

Paste your exercise results here and we will review before moving on.
