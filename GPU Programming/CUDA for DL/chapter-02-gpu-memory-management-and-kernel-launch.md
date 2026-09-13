# Chapter 2 — GPU Memory Management and Kernel Launch

*Part 1: CUDA Fundamentals & Memory Management. Confirmed title, straight from the book's companion repo: `book.cu/0_vecadd/README.md` states these examples are "from Chapter 02: GPU Memory Management and Kernel Launch."*

Chapter 1 was all mental model. This chapter is where you actually compile and run something. Every code sample below is the **real, unmodified source from the book's official companion repository** (`github.com/Infatoshi/book.cu`, directory `0_vecadd/`) — not a reconstruction. I've pulled it directly so you're building from the same material the book's readers are. I've added the theory scaffolding (the "why," and one professional technique the repo's own code doesn't use yet) around it.

Three progressively harder examples live in this chapter's folder: `vecadd.cu` (fixed 8 elements, one block), `vecadd_scalable.cu` (1M elements, many blocks, boundary checks), `tensor_add_3d.cu` (3D indexing, the pattern real tensors use).

---

## 2.1 Toolchain & Build

You need: an NVIDIA GPU, the CUDA Toolkit (`nvcc` on your `PATH`), and (optionally) GNU Make. Confirm both before touching any `.cu` file — this is literally Chapter 1's closing exercise, so you should already have the numbers:

```bash
nvidia-smi        # driver + max supported CUDA version
nvcc --version    # installed CUDA Toolkit / compiler version
```

The book's own Makefile for this chapter (verbatim):

```makefile
# CUDA Vector Addition Examples - Makefile
# Chapter 02: GPU Memory Management and Kernel Launch

NVCC = nvcc
NVCC_FLAGS = -O2
TARGETS = vecadd vecadd_scalable tensor_add_3d

all: $(TARGETS)

vecadd: vecadd.cu
	$(NVCC) $(NVCC_FLAGS) vecadd.cu -o vecadd

vecadd_scalable: vecadd_scalable.cu
	$(NVCC) $(NVCC_FLAGS) vecadd_scalable.cu -o vecadd_scalable

tensor_add_3d: tensor_add_3d.cu
	$(NVCC) $(NVCC_FLAGS) tensor_add_3d.cu -o tensor_add_3d

clean:
	rm -f $(TARGETS)

run: all
	./vecadd
	./vecadd_scalable
	./tensor_add_3d
```

Nothing exotic: `-O2` host-code optimization, one target per `.cu` file, `nvcc` handles splitting host code (compiled with your system's C++ compiler) from device code (compiled to PTX→SASS) transparently. `make all && make run` builds and runs all three.

---

## 2.2 Example 1 — Basic Vector Addition (`vecadd.cu`)

**Goal:** the smallest possible correct CUDA program. Eight elements, one block, one thread per element, no boundary checks (deliberately — see the warning in the docstring below).

```cuda
__global__ void vectorAdd(float *a, float *b, float *c) {
    // Get the thread index within the block
    int i = threadIdx.x;

    // Perform element-wise addition: c[i] = a[i] + b[i]
    c[i] = a[i] + b[i];
}
```

That's the entire kernel. `__global__` marks it as device code, callable from the host, running on the GPU. `threadIdx.x` is a built-in, per-thread value — thread 0 gets `i=0`, thread 7 gets `i=7`, and so on. Note the explicit **warning in the book's own docstring**: *"This kernel does not perform bounds checking. It assumes the number of threads launched exactly matches the vector size."* That's the entire subject of the next example.

The `main()` function is the part worth reading closely, because its 8-step structure is the skeleton **every single CUDA program in this course will follow**, all the way through the CUTLASS capstone in Part 11:

```cuda
int main() {
    int n = 8;
    size_t size = n * sizeof(float);

    // STEP 1: Allocate host (CPU) memory using standard malloc
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (h_a == nullptr || h_b == nullptr || h_c == nullptr) {
        std::cerr << "Error: Failed to allocate host memory" << std::endl;
        return 1;
    }

    // STEP 2: Initialize input vectors with test data
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;        // Vector A: [0, 1, 2, 3, 4, 5, 6, 7]
        h_b[i] = (float)(i * 2);  // Vector B: [0, 2, 4, 6, 8, 10, 12, 14]
        // Expected result: C = [0, 3, 6, 9, 12, 15, 18, 21]
    }

    // STEP 3: Allocate device (GPU) memory using cudaMalloc
    float *d_a, *d_b, *d_c;
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed for d_a: " << cudaGetErrorString(err) << std::endl;
        free(h_a); free(h_b); free(h_c);
        return 1;
    }
    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) { /* ...same pattern, freeing what's already allocated... */ }
    err = cudaMalloc((void**)&d_c, size);
    if (err != cudaSuccess) { /* ... */ }

    // STEP 4: Copy data from host (CPU) to device (GPU)
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    // ...error check...
    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    // ...error check...

    // STEP 5: Launch CUDA kernel on the GPU
    vectorAdd<<<1, 8>>>(d_a, d_b, d_c);
    err = cudaGetLastError();          // catches launch-config errors
    // ...error check...
    err = cudaDeviceSynchronize();     // blocks host until the kernel finishes
    // ...error check...

    // STEP 6: Copy results back from device (GPU) to host (CPU)
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    // ...error check...

    // STEP 7: Verify correctness with an epsilon-tolerant float comparison
    bool success = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        float diff = std::abs(expected - h_c[i]);
        if (diff > std::numeric_limits<float>::epsilon()) {
            std::cout << "Error at index " << i << ": Got " << h_c[i]
                      << ", expected " << expected << std::endl;
            success = false;
            break;
        }
    }
    if (success) std::cout << "Success! All elements are correct." << std::endl;

    // STEP 8: Clean up — host memory with free(), device memory with cudaFree()
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return success ? 0 : 1;
}
```

Four facts worth pulling out explicitly, connecting straight back to Chapter 1:

1. **`h_`/`d_` prefix convention.** This isn't required by the compiler — it's a naming discipline the book uses throughout precisely because host and device pointers point into *physically separate* memory (1.3.1) and mixing them up is a classic, hard-to-debug CUDA bug (dereferencing a device pointer on the host segfaults; passing a host pointer to a kernel silently corrupts memory instead of crashing).
2. **Every CUDA call returns a `cudaError_t`, and this code checks every single one.** That's not paranoia — CUDA calls fail silently by default. Skipping error checks is the single most common reason a "should be simple" kernel produces wrong output with zero diagnostic information.
3. **`vectorAdd<<<1, 8>>>(...)`** — 1 block, 8 threads per block, 1×8 = 8 threads total, exactly matching `n`. This launch is asynchronous (1.3.1): the host moves on to the next line immediately, which is *why* `cudaGetLastError()` (catches launch-time errors, e.g. invalid config) and `cudaDeviceSynchronize()` (blocks until the kernel actually finishes, catches runtime errors, e.g. an illegal memory access) are two separate, necessary calls.
4. **Floating-point comparison uses an epsilon tolerance**, never `==`. Rounding differences between CPU and GPU floating-point arithmetic are expected, not bugs.

## 2.3 Example 2 — Scalable Vector Addition (`vecadd_scalable.cu`)

Real data doesn't fit in one block (a block caps out at 1024 threads on every current architecture). This example scales to 1,000,000 elements using the pattern you'll type thousands of times over this course:

```cuda
__global__ void vectorAddScalable(float *a, float *b, float *c, int n) {
    // Calculate global thread index across all blocks
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check to ensure we don't access out-of-range elements
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

Two changes from Example 1, both essential:

- **Global indexing**, `blockIdx.x * blockDim.x + threadIdx.x`, instead of just `threadIdx.x`. `blockIdx.x` identifies *which* block a thread belongs to; multiplying by `blockDim.x` (threads per block) and adding the thread's own local `threadIdx.x` gives every thread, across every block, a unique index into the full array. This is the line Chapter 1 (1.3.2) told you to memorize — here it is, doing real work.
- **The bounds check, `if (i < n)`.** The book's own docstring is blunt about why: *"Without it, threads that exceed the vector size would access invalid memory, causing undefined behavior."* You'll deliberately break this in the exercises below to see what "undefined behavior" actually looks like under a memory checker.

Grid-size calculation — the other new piece:

```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
// n=1,000,000, threadsPerBlock=256:
// blocksPerGrid = (1,000,000 + 255) / 256 = 3,908 blocks
// total threads = 3,908 * 256 = 1,000,448  (448 more than n — the bounds check discards them safely)
```

`(n + threadsPerBlock - 1) / threadsPerBlock` is integer-arithmetic ceiling division — it's the standard idiom because `n` almost never divides evenly by your block size, and you'd rather over-launch by a few threads (safely discarded by the bounds check) than under-launch and silently drop real data. **256 as the block size isn't arbitrary**: it's a multiple of the 32-thread warp size (Chapter 1, 1.3.2) — exactly 8 warps — which is a safe, broadly-good default for a memory-bound elementwise kernel like this one, before you've profiled anything.

The launch and verification:

```cuda
vectorAddScalable<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
err = cudaGetLastError();
err = cudaDeviceSynchronize();
// ... copy back ...

// For 1M elements, checking every one is wasteful — sample instead:
// first 10 and last 10 elements only
for (int i = 0; i < 10 && success; ++i) { /* check h_a[i]+h_b[i] == h_c[i] */ }
for (int i = n - 10; i < n && success; ++i) { /* same check */ }
```

Sampling the first and last 10 elements is a pragmatic verification strategy once `n` gets large — full verification of every element is itself an O(n) CPU loop that would dwarf the GPU work you're trying to measure.

## 2.4 Example 3 — 3D Tensor Addition (`tensor_add_3d.cu`)

Deep learning tensors are almost never 1D. This example generalizes to a `depth × height × width` tensor (think: a batch of feature maps) and introduces `dim3` for genuinely multi-dimensional grids and blocks — plus, for the first time in this chapter, an actual CPU reference implementation timed and compared against the GPU result.

**CPU reference** (the pattern Chapter 1, 1.6.1 called "CPU-first development," starting here in earnest):

```cuda
void tensorAdd3D_cpu(const float* A, const float* B, float* C, int depth, int height, int width) {
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // Row-major flattening: fastest-varying dimension (width) is contiguous
                int index = d * (height * width) + h * width + w;
                C[index] = A[index] + B[index];
            }
        }
    }
}
```

**GPU kernel**, using 3D thread coordinates directly instead of manually decomposing a 1D index:

```cuda
__global__ void tensorAdd3D_kernel(const float* A, const float* B, float* C, int depth, int height, int width) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;  // Width  (x-dimension)
    int h = blockIdx.y * blockDim.y + threadIdx.y;  // Height (y-dimension)
    int d = blockIdx.z * blockDim.z + threadIdx.z;  // Depth  (z-dimension)

    if (d < depth && h < height && w < width) {
        int index = d * (height * width) + h * width + w;
        C[index] = A[index] + B[index];
    }
}
```

The bounds check is now a 3-way `&&` — one clause per dimension, since ceiling division (below) can overshoot in x, y, *and* z independently.

Launch configuration with `dim3`:

```cuda
dim3 threadsPerBlock(8, 8, 8);   // 8*8*8 = 512 threads/block — under the 1024 hardware cap

dim3 blocksPerGrid(
    (width  + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
    (depth  + threadsPerBlock.z - 1) / threadsPerBlock.z
);
// For depth=32, height=128, width=128, block=(8,8,8):
// blocksPerGrid = (16, 16, 4) → 1,024 blocks × 512 threads/block = 524,288 threads total
```

**Why `w` maps to `blockIdx.x`/`threadIdx.x` and not, say, `d`:** the index formula `d*(height*width) + h*width + w` makes `w` the fastest-varying (contiguous) dimension in memory — row-major layout, same convention NumPy/PyTorch use by default. Mapping the *x*-dimension of the thread grid to the *contiguous* memory dimension means adjacent threads within a warp (which vary fastest in `threadIdx.x`) touch adjacent memory addresses. That's memory coalescing — introduced conceptually in Chapter 1 (1.4.1) — showing up as a concrete indexing decision for the first time. Get this mapping backwards (map `x` to `depth` instead) and the kernel is still *correct*, just meaningfully slower, because each warp's 32 threads now stride through memory instead of reading it contiguously.

The example also times the CPU path and reports it:

```cuda
auto start = std::chrono::high_resolution_clock::now();
tensorAdd3D_cpu(h_A, h_B, h_C_cpu, depth, height, width);
auto end = std::chrono::high_resolution_clock::now();
auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
// ...
std::cout << "CPU computation took " << cpu_time.count() << " ms" << std::endl;
std::cout << "GPU computation completed (asynchronous, timing not measured)" << std::endl;
```

Worth noticing explicitly: **the book's own example times the CPU path but not the GPU path** ("timing not measured" is printed verbatim). That's a real gap, and it's your first hands-on exercise below — `std::chrono` around a `cudaDeviceSynchronize()` call would work, but the *correct* tool for timing GPU work is CUDA events, which I'll introduce now since you'll want it for every chapter from here on.

## 2.5 The Memory Model, Recap

Three concepts this chapter put into your hands rather than just your head:

- **Host memory** (`malloc`/`free`) and **device memory** (`cudaMalloc`/`cudaFree`) are separate allocations in separate address spaces — every example allocates both, and frees both, explicitly.
- **`cudaMemcpy(dst, src, size, direction)`** is the only bridge between them, and it's synchronous (blocks the host until the transfer completes) in the form used here. `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost` are the two directions you've now used; a third, `cudaMemcpyDeviceToDevice`, appears later once you're moving data GPU-to-GPU (Part 10).
- **The launch-then-check-twice idiom**: `kernel<<<...>>>(...)` (async) → `cudaGetLastError()` (did the *launch* fail — bad config, too many threads?) → `cudaDeviceSynchronize()` (did *execution* fail — illegal address, before you trust the result). Every example in this chapter uses exactly this sequence; it will not be spelled out again in future chapters' walkthroughs, but the pattern doesn't go away.

## 2.6 Add-On: Timing GPU Kernels with CUDA Events

*Not in the book's repo for this chapter — this is a standard CUDA idiom worth having now, since you'll want real numbers (not just "it printed Success") starting with the next chapter's CPU-vs-GPU comparisons.*

`std::chrono` around a kernel launch measures launch overhead + your own host code, not GPU execution time, unless you force a sync — and even then you're including sync overhead. **CUDA events** are timestamps recorded *by the GPU itself* into a stream, giving you accurate device-side timing:

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
vectorAddScalable<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
cudaEventRecord(stop);

cudaEventSynchronize(stop);  // wait for the 'stop' event specifically
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "Kernel time: " << milliseconds << " ms" << std::endl;

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

You'll use this exact pattern to generate the first real performance numbers of the course in Chapter 3's exercises, and it's the manual precursor to what Nsight Compute automates for you starting in Part 9.

---

## Hands-On Lab

1. **Build and run the real thing.**
   ```bash
   git clone https://github.com/Infatoshi/book.cu.git
   cd book.cu/0_vecadd
   make all && make run
   ```
   Confirm all three print `Success!`.

2. **Break the bounds check on purpose.** In a copy of `vecadd.cu`, launch `vectorAdd<<<1, 16>>>(d_a, d_b, d_c)` against the 8-element arrays (8 extra threads now read/write past the end of `d_c`). Run it under `compute-sanitizer` (ships with the CUDA Toolkit):
   ```bash
   compute-sanitizer ./vecadd
   ```
   Read the report. This is what "undefined behavior" (the warning in `vectorAddScalable`'s docstring) looks like in practice, and it's your first hands-on look at a tool you'll rely on properly in Part 9.

3. **Add real GPU timing.** Instrument `vecadd_scalable.cu` with the CUDA-events pattern from §2.6. Sweep `threadsPerBlock` across `{32, 64, 128, 256, 512, 1024}` and record the kernel time at each. On a purely memory-bound, 1-FLOP-per-element kernel like this, expect the differences to be small — that's itself a useful, real result (you're bandwidth-bound, not launch-config-bound; ties directly back to Chapter 1, 1.4.1).

4. **Run it on both of your machines.** Build and run all three examples on your RTX 3090 box and your Tesla T4. Compare the GPU-vs-CPU timing gap on `tensor_add_3d` (32×128×128 = 524,288 elements) between the two — the T4's lower core count and bandwidth (Chapter 1's table) should show up as a smaller GPU speedup than the 3090.

## Exercises

1. **2D kernel from scratch.** Write a `matrixAdd2D` kernel (and host driver) for two `256×256` matrices, using 2D `dim3` grid/block configuration analogous to the 3D tensor example's x/y mapping, but without the z-dimension. Verify against a CPU reference.
2. **Coalescing, deliberately broken.** Take `tensorAdd3D_kernel` and swap the roles of `w` and `d` in the index formula (map `x`→depth, `z`→width) while keeping the same launch configuration. It should still produce correct output. Time it with CUDA events against the original mapping and explain the difference using the coalescing argument from §2.4.
3. **Occupancy arithmetic.** For your RTX 3090 (82 SMs) and your T4 (40 SMs), compute how many of `vecadd_scalable`'s 3,908 blocks (256 threads each) could theoretically be resident *simultaneously* if each SM can host at most 2048 threads — i.e., how many "waves" of blocks are needed to cover the whole grid on each GPU. (You don't need real occupancy-calculator numbers yet — Part 9 covers that properly — just the arithmetic.)
4. **Error-checking macro.** Every example above repeats the same `if (err != cudaSuccess) { ...; return 1; }` block by hand. Write a `CUDA_CHECK(call)` macro that wraps any CUDA API call, checks its return value, prints the file/line and `cudaGetErrorString`, and exits on failure. Refactor `vecadd_scalable.cu` to use it. (This is a near-universal idiom in real CUDA codebases — not from this book's repo, but you'll want it for every chapter from here on.)

---

**Next:** Chapter 3 — Building the Core Operations from Scratch (Part 2). CPU-first development applied to the operations that actually make up a neural network: elementwise ops, transpose, naive GEMM, softmax, and convolutions.
