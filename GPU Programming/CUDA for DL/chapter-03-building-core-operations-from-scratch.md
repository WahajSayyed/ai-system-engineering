# Chapter 3 — Building the Core Operations from Scratch

*Part 2: Building Deep Learning Primitives from Scratch. Confirmed title, from the book's companion repo: `book.cu/1_naive/README.md` states these implementations are "extracted from Chapter 03: Building the Core Operations from Scratch."*

Chapter 2 taught you the mechanics of a kernel. This chapter applies them to the eight operations that, combined, make up almost every neural network: elementwise add, matrix add, transpose, GEMM, softmax, 1D convolution, 2D convolution, and max pooling. Every kernel below is the **real, unmodified source** from `book.cu/1_naive/` — I pulled it directly from the repo, organized exactly as the book ships it (`elementwise/`, `transpose/`, `gemm/`, `softmax/`, `conv1d/`, `conv2d/`, `maxpool2d/`).

The book's own framing for this chapter is blunt and worth stating up front: these implementations **prioritize correctness over performance**. Every kernel is naive on purpose — one thread per output element, no shared memory, no tiling, no vectorization — and every kernel is paired with a CPU reference for verification. That's "CPU-first development," and it's the discipline every later optimization chapter builds on: you cannot know a "faster" kernel is *correct* unless you already have a trusted baseline to check it against.

---

## 3.0 The Shared Harness: `common.h`

Before the operations themselves, the book introduces a small shared header used by all eight examples — and it directly delivers something Chapter 2's exercises asked you to build yourself:

```cuda
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

/**
 * Timer class for measuring execution time of operations
 * Automatically prints elapsed time when destroyed (RAII pattern)
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    Timer(const std::string& operation_name = "") : name(operation_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        if (!name.empty()) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << name << " took " << duration.count() << " ms" << std::endl;
        }
    }
    void reset() { start_time = std::chrono::high_resolution_clock::now(); }
    long long elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }
};

/**
 * CUDA error checking macro
 * Wraps CUDA calls and exits with error message if operation fails
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

template<typename T>
void allocate_host(T** ptr, size_t size) {
    *ptr = (T*)malloc(size * sizeof(T));
    if (*ptr == nullptr) { std::cerr << "Failed to allocate host memory" << std::endl; exit(1); }
}

template<typename T>
void allocate_device(T** d_ptr, size_t size) {
    CUDA_CHECK(cudaMalloc((void**)d_ptr, size * sizeof(T)));
}

template<typename T>
void copy_to_device(T* d_dst, const T* h_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copy_to_host(T* h_dst, const T* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T> void free_host(T* ptr) { free(ptr); }
template<typename T> void free_device(T* d_ptr) { CUDA_CHECK(cudaFree(d_ptr)); }

bool verify_results(const float* result, const float* reference, size_t size, float tolerance = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(result[i] - reference[i]) > tolerance) {
            std::cout << "Verification failed at index " << i
                      << ": got " << result[i] << ", expected " << reference[i] << std::endl;
            return false;
        }
    }
    return true;
}
```

Two things worth calling out directly:

- **`CUDA_CHECK(call)`** is exactly the macro Chapter 2's exercises asked you to write yourself — wrapping any CUDA call, checking its return, printing file/line and `cudaGetErrorString` on failure. Seeing the book's own version confirms it's the standard idiom, not something I invented for the exercise.
- **`Timer` answers Chapter 2's other open question** (recall `tensor_add_3d.cu` printed *"GPU computation completed (asynchronous, timing not measured)"*). Here, every example wraps its GPU section in a scoped `Timer`, and — critically — calls `CUDA_CHECK(cudaDeviceSynchronize())` *inside* that scope, before the `Timer`'s destructor captures the end time. That forces the host to wait for the kernel to actually finish before the clock stops, so `std::chrono` correctly captures device execution time. It's coarser than the CUDA-events approach from Chapter 2 §2.6 (it also includes sync-call overhead, and can't overlap-time multiple concurrent streams), but for the single-kernel, single-stream measurements this chapter needs, it's simple and accurate.

Every example includes and uses this file — I won't repeat the `#include "common.h"` line in the code blocks below, but every `.cu` file starts with it.

## 3.1 CPU-First Development Methodology

The book states the pattern explicitly, and it's the same four steps for every operation in this chapter:

1. **Implement the CPU version** — clear, sequential, obviously-correct code.
2. **Implement the GPU kernel** — parallelize the same logic, one thread per output element.
3. **Verify correctness** — compare GPU output against the CPU reference with `verify_results` (floating-point tolerance, never exact equality).
4. **Measure performance** — get a real baseline number before anything is optimized.

And the indexing conventions are consistent across all eight files — this is the vocabulary you'll now use for the rest of the course:

```cuda
// 1D indexing (vectors, 1D conv)
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D indexing (matrices, 2D operations)
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

// Boundary checks
if (row < height && col < width) {
    // Compute...
}
```

## 3.2 Element-wise Operations

These reinforce Chapter 2's pattern but now run inside the shared harness, with a CPU reference and real timing for the first time.

**Vector addition** (`elementwise/vector_add.cu`) — 1,000,000 elements, the same global-index/bounds-check pattern from Chapter 2, now properly timed and verified:

```cuda
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// main() — abbreviated to the pattern; full I/O and setup follow §3.1's four steps
int n = 1000000;
// ...allocate host + device, initialize h_a[i]=i, h_b[i]=2i...
{
    Timer cpu_timer("CPU Vector Addition");
    vector_add_cpu(h_a, h_b, h_c_cpu, n);
}
// ...copy to device...
int threadsPerBlock = 256;  // must be multiple of 32 for warp efficiency
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
{
    Timer gpu_timer("GPU Vector Addition");
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
// ...copy back, verify_results(h_c_gpu, h_c_cpu, n)...
```

The book explicitly lists this operation's real use cases beyond the toy example: **bias addition in linear layers, residual connections in ResNets/Transformers, elementwise steps inside attention.**

**Matrix addition** (`elementwise/matrix_add.cu`) — the 2D generalization, 1024×1024 matrices:

```cuda
void matrix_add_cpu(const float* A, const float* B, float* C, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            int index = row * num_cols + col;
            C[index] = A[index] + B[index];
        }
    }
}

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int num_rows, int num_cols) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < num_rows && column < num_cols) {
        int index = row * num_cols + column;
        C[index] = A[index] + B[index];
    }
}
// Launch: dim3 threadsPerBlock(16, 16); → 256 threads/block, 8 warps
```

Real use cases per the book: **batch normalization's bias-add step, and layer aggregation** (combining outputs from parallel branches).

### 3.2.1 Deep Dive: What These Kernels Actually Cost, in Real Bytes

It's worth working the arithmetic all the way through once, rather than leaving "memory-bound" as an abstract label, because every later chapter's speedup claims are measured against numbers exactly like these.

**Vector add, `n = 1,000,000`:** every element requires one read of `a`, one read of `b`, and one write to `c` — three 4-byte transfers, zero reuse of anything.

```
Bytes moved = 3 × n × 4 bytes = 3 × 1,000,000 × 4 = 12,000,000 bytes ≈ 11.4 MiB
FLOPs       = n = 1,000,000
Arithmetic intensity = 1,000,000 / 12,000,000 ≈ 0.083 FLOP/byte
```

That's the exact same 0.083 FLOP/byte figure Chapter 1 §1.4.1 used as its worked example of a deeply memory-bound operation — now tied to this chapter's actual 1,000,000-element test case instead of a generic illustration. Matrix add's 1024×1024 case works out identically (same one-read-one-read-one-write shape, just more elements), landing at the same ≈0.083 FLOP/byte — the two "elementwise" kernels in this section are, arithmetic-intensity-wise, the *same* operation wearing a different shape.

**Why `dim3 threadsPerBlock(16, 16)` for a 1024×1024 matrix is a clean choice, concretely:** 256 threads is 8 warps (a reasonable, divisible-by-32 block size — Chapter 2's reasoning, reapplied), and `1024 / 16 = 64` exactly — every block is entirely full, with zero boundary-check waste at the matrix's edges. Contrast this with, say, a 1000×1000 matrix at the same block size: `ceil(1000/16) = 63` blocks per dimension, meaning the last row and column of blocks are only partially populated, and the `if (row < num_rows && column < num_cols)` check that looked like defensive boilerplate in Chapter 2 is doing real, non-trivial work at those edges.

**A worthwhile caveat about `verify_results`'s fixed `1e-5f` tolerance:** it's a reasonable default for these small-magnitude test values, but a *fixed absolute* tolerance quietly breaks down as values grow — accumulated floating-point error scales with the magnitude and the number of operations summed, so a sum of thousands of large values can legitimately differ from a reference by far more than `1e-5` while still being "correct" FP32 arithmetic. You'll feel this directly once GEMM's `K`-length accumulation (§3.3.2) grows large, or once Chapter 4's cuBLAS comparisons need a *relative* tolerance (`|result - reference| / |reference|`) instead of this chapter's simpler absolute one.

## 3.3 Matrix Operations

### 3.3.1 Transpose — the coalescing trade-off, made concrete

`transpose/transpose.cu` is the first example where the CPU and GPU code both have an *asymmetric* memory access pattern baked into the algorithm itself, not just an implementation detail:

```cuda
void transpose_cpu(const float* in, float* out, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; ++row) {
        for (int column = 0; column < num_cols; ++column) {
            // Reads row-major (sequential); writes column-major (strided)
            out[column * num_rows + row] = in[row * num_cols + column];
        }
    }
}

__global__ void transpose_kernel(const float* in, float* out, int num_rows, int num_cols) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < num_rows && column < num_cols) {
        int in_index = row * num_cols + column;      // coalesced read
        int out_index = column * num_rows + row;      // strided write — the bottleneck
        out[out_index] = in[in_index];
    }
}
```

Read the book's own comment on this kernel closely: *"threads write with stride `num_rows`... this is the main performance bottleneck: strided writes prevent memory coalescing."* Threads in the same warp vary in `column` (since `column` maps to `threadIdx.x`), so their **reads** from `in[row*num_cols+column]` are adjacent — coalesced. But their **writes** to `out[column*num_rows+row]` land `num_rows` elements apart in memory — one memory transaction per thread instead of one transaction per warp. Same total bytes moved as `vector_add`, dramatically worse achieved bandwidth, purely because of *which* dimension you write contiguously into.

The book's own docstring previews exactly how Part 5 fixes this: *"load input tile to shared memory, transpose in shared memory, write coalesced."* That's the shared-memory tile-transpose technique — stage a block into fast on-chip memory, do the index-swap there, then write back out in a coalesced pattern. You'll implement it in Part 5.

**Deep dive: counting actual memory transactions, not just "coalesced vs. not."** The GPU's memory controller doesn't service one thread's request at a time — it services a whole warp's requests together, in units of fixed-size memory transactions (historically 32-byte sectors, aggregated into 128-byte cache-line-sized bursts on current architectures). This is what "coalescing" is actually a statement about:

- **Coalesced read:** 32 threads in a warp, consecutive `column` values, each reading 4 bytes at `in[row*num_cols + column]` — those 32 threads' addresses span exactly `32 × 4 = 128` contiguous bytes. One 128-byte transaction satisfies the *entire warp's* read in a single trip to memory.
- **Strided write:** those same 32 threads write to `out[column*num_rows + row]` — addresses that jump by `num_rows × 4` bytes between consecutive threads. For any `num_rows` larger than 32, no two threads' writes land in the same 128-byte window at all. The hardware has no choice but to issue **up to 32 separate transactions** to service the same warp's store.

Same 128 bytes of true data moved, per warp, either way — but up to **32× more transactions issued** for the write than the read. That gap between "bytes your algorithm needs to move" and "transactions the hardware actually issues to move them" is the real, mechanical reason a strided-access kernel can move identical total data yet measure dramatically slower: transaction *count*, not just byte *count*, consumes memory-controller cycles.

### 3.3.2 GEMM — the "compute-bound in theory, memory-bound in practice" trap

`gemm/gemm.cu` is the most important kernel in this chapter, because it's naive GEMM's *specific* failure mode that motivates roughly a third of this entire course (all of Part 5–6).

```cuda
void gemm_cpu(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
    for (int row = 0; row < M_rows; ++row) {
        for (int column = 0; column < N_cols; ++column) {
            float sum = 0.0f;
            for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
                sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
            }
            C[row * N_cols + column] = sum;
        }
    }
}

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M_rows && column < N_cols) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
            sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
        }
        C[row * N_cols + column] = sum;
    }
}
// Test size: M=512, N=512, K=256 → 2*512*512*256 = 134,217,728 FLOPs
```

Recall Chapter 1 §1.4.1: GEMM's *theoretical* arithmetic intensity is high, because each element of `A` and `B` can in principle be reused `N` or `M` times respectively. This naive kernel **does not realize that reuse at all**. Every thread independently walks its own full `K`-length row of `A` and column of `B` straight from global memory — if two threads in the same block share the same row of `A` (they do, for every thread in the same output row), that row gets re-fetched from global memory once *per thread*, with zero caching of the reuse in shared memory or registers. On top of that, the book's own comment flags the access pattern itself as bad: reading `B[k_idx * N_cols + column]` strides through memory by `N_cols` elements per step — the CPU version even calls this out explicitly as *"cache-unfriendly, strided"* for the same reason a CPU BLAS implementation also has to tile for cache locality, not just a GPU one for coalescing.

Net effect: a kernel whose *math* looks compute-bound is, in this naive form, memory-bound in practice, because nothing is reused before being evicted. The book's own measured range makes the gap concrete:

| Operation | Naive GFLOPS | Optimized GFLOPS | Speedup |
|---|---|---|---|
| GEMM | 10–50 | 1,000–5,000 | 100–200× |
| Conv2D | 5–20 | 1,000–10,000 | 200–500× |
| Softmax | 1–5 | 100–500 | 100–200× |

*(Book's own figures, "approximate and GPU-dependent.")* That 100–200× GEMM gap is exactly the compounding optimization stack from Chapter 1 §1.6.2–1.6.3 (coalescing → shared-memory tiling → register blocking → vectorization → tensor cores), which Part 5–6 build rung by rung on top of this exact kernel.

### Deep dive: working the roofline numbers for real, on this exact kernel

Chapter 1 §1.4.1 introduced arithmetic intensity in the abstract. Here it is worked through completely, using this section's actual `M=512, N=512, K=256` test case — the same exercise Exercise 2 below asks you to redo yourself, done once here so you have a worked reference.

**FLOPs**, confirmed already: `2×M×N×K = 2×512×512×256 = 134,217,728`.

**Actual memory traffic this kernel issues**, assuming no cache reuse across threads at all (the honest worst case for a kernel with no shared memory): each of the `M×N = 262,144` threads independently reads its own `K=256`-element row of `A` and column of `B` — `2×256×4 = 2,048` bytes per thread — plus one 4-byte write to `C`.

```
Total ≈ 262,144 × 2,052 bytes ≈ 537.9 MB
Actual arithmetic intensity ≈ 134,217,728 / 537,919,488 ≈ 0.25 FLOP/byte
```

**Ideal memory traffic**, assuming every element of `A`, `B`, and `C` is read or written from global memory *exactly once* (the best any tiling scheme could ever achieve, since every element genuinely must cross the global-memory boundary at least once):

```
Unique data = (M×K + K×N + M×N) × 4 bytes = (131,072 + 131,072 + 262,144) × 4 = 2,097,152 bytes (exactly 2 MiB)
Ideal arithmetic intensity = 134,217,728 / 2,097,152 = 64.0 FLOP/byte, exactly
```

Now compare both numbers against real ridge points, computed from confirmed hardware specs (peak FP32 CUDA-core throughput ÷ peak memory bandwidth):

| GPU | Peak FP32 | Bandwidth | Ridge point |
|---|---|---|---|
| RTX 3090 | 35.58 TFLOPS | 936 GB/s | ≈ 38.0 FLOP/byte |
| Tesla T4 | 8.1 TFLOPS | 320 GB/s | ≈ 25.3 FLOP/byte |

The naive kernel's **actual** intensity (0.25) sits nowhere near either ridge point — deeply memory-bound on both GPUs, exactly as diagnosed. But the **ideal** intensity (64.0) sits *above* both ridge points — meaning a perfectly-tiled version of this exact same GEMM would be genuinely compute-bound on either your 3090 or your T4. That's the real, numeric target Part 5's tiled GEMM kernels are chasing: not "faster" in some vague sense, but specifically *closing the gap from 0.25 up toward 64.0* by converting one-time-use global loads into many-time-reused shared-memory and register values.

**One honest caveat, worth not glossing over:** the "0.25 FLOP/byte" figure describes traffic *issued to the memory subsystem in the total absence of caching*, not necessarily traffic that actually reaches DRAM. Real GPUs have L1 and L2 caches that can catch some of this reuse automatically and "for free" — every one of the `M=512` output rows re-reads the *exact same* `B` matrix, and a big enough L2 cache (a few MB on both your GPUs) can hold a meaningful fraction of a 512×256 `B` between different rows' thread blocks, softening the real DRAM hit somewhat. That's part of why naive GEMM measures at "only" 10–50 GFLOPS rather than something even more catastrophic. The difference shared-memory tiling (Part 5) makes isn't that caching didn't exist before — it's that tiling makes reuse **guaranteed and programmer-controlled** instead of opportunistic and subject to eviction, which is a much stronger and more predictable guarantee than hoping the L2 cache happens to still hold the data you need.

## 3.4 Neural Network Primitives

### 3.4.1 Softmax — an algorithmic inefficiency, not a memory one

`softmax/softmax.cu` introduces a *different* class of naive-kernel problem than GEMM's. This one isn't primarily about memory access patterns — it's about **redundant computation across threads**:

```cuda
void softmax_cpu(const float* in, float* out, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; ++row) {
        float max_val = in[row * num_cols];
        for (int col = 1; col < num_cols; ++col)
            if (in[row * num_cols + col] > max_val) max_val = in[row * num_cols + col];

        float sum_exp = 0.0f;
        for (int col = 0; col < num_cols; ++col)
            sum_exp += expf(in[row * num_cols + col] - max_val);

        for (int col = 0; col < num_cols; ++col)
            out[row * num_cols + col] = expf(in[row * num_cols + col] - max_val) / sum_exp;
    }
}

__global__ void softmax_naive_kernel(const float* in, float* out, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows && column < num_cols) {
        // Every one of the num_cols threads in this row redundantly
        // recomputes the SAME max and the SAME sum from scratch:
        float max_val = -1e20f;
        for (int col_idx = 0; col_idx < num_cols; ++col_idx)
            if (in[row * num_cols + col_idx] > max_val) max_val = in[row * num_cols + col_idx];

        float sum_exp = 0.0f;
        for (int col_idx = 0; col_idx < num_cols; ++col_idx)
            sum_exp += expf(in[row * num_cols + col_idx] - max_val);

        out[row * num_cols + column] = expf(in[row * num_cols + column] - max_val) / sum_exp;
    }
}
```

Walk the complexity through: the CPU version is `O(num_rows * num_cols)` — three passes over each row, but the passes are shared across all columns in that row. The naive GPU kernel launches one thread *per column*, and every one of those `num_cols` threads independently re-scans the *entire row* twice (once for max, once for sum) before computing its own single output value. That's `O(num_cols)` redundant work multiplied by `num_cols` threads per row — `O(num_cols²)` total work per row instead of `O(num_cols)`. For the book's own test size (128 rows × 1000 columns), that's roughly a thousand-fold more total arithmetic than the CPU version does, entirely from redundancy, before you even account for the repeated global-memory reads underneath it.

The book's own docstring names the fix directly: *"parallel reduction for max," "parallel reduction for sum,"* possibly via a *"two-pass kernel: first computes max and sum, second computes softmax."* That's precisely the shift from *thread-per-output-element, independent redundant work* to *thread-cooperative reduction via shared memory and warp shuffles* — Part 5, Chapter 16 in this course, and the same "online softmax" idea that makes Flash Attention (Part 7) possible at all.

Notice also the **numerical stability technique** used identically on both CPU and GPU: subtracting `max_val` before `expf()`. Softmax is mathematically `exp(x_i) / sum(exp(x_j))` regardless of any constant shift, but computing it that way risks `expf()` overflowing to `inf` for large logits; subtracting the row's max first guarantees every exponent argument is `≤ 0`, so `expf()` never overflows. This trick reappears every time you touch attention scores for the rest of the course.

**Deep dive: the precise redundancy factor, not just "roughly."** For this section's actual test size (128 rows × 1,000 columns), it's worth computing the real ratio rather than eyeballing it:

```
CPU total work        ≈ num_rows × 3 × num_cols = 128 × 3 × 1,000 = 384,000 elementwise operations
Naive GPU total work  ≈ (num_rows × num_cols) threads × (≈2 × num_cols) ops/thread
                      = 128,000 threads × 2,000 ops/thread ≈ 256,128,000 operations
Redundancy factor     ≈ 256,128,000 / 384,000 ≈ 667×
```

At this specific test size, the naive kernel performs roughly **667× more total arithmetic** than the CPU version — not because the algorithm is different (it's mathematically identical), but purely because every one of the 1,000 threads in a row re-derives the same max and the same sum from scratch instead of any of them being shared. Widen `num_cols` (as a real vocabulary-sized softmax over, say, 50,000 logits would) and this ratio grows linearly with `num_cols` — the redundancy gets *worse*, not better, at exactly the sequence lengths and vocabulary sizes real LLM inference cares about, which is precisely why Part 5's fix (thread-cooperative reduction) isn't optional polish but a genuine necessity at scale.

### 3.4.2 1D Convolution — the sliding-window pattern

`conv1d/conv1d.cu`, valid (no) padding, output size = `input_size - kernel_size + 1`:

```cuda
void conv1d_cpu(const float* in, float* out, const float* kernel, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    for (int i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size; ++j)
            sum += in[i + j] * kernel[j];
        out[i] = sum;
    }
}

__global__ void conv1d_kernel(const float* in, float* out, const float* kernel, int input_size, int kernel_size) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    if (output_idx < output_size) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < kernel_size; ++k_idx)
            sum += in[output_idx + k_idx] * kernel[k_idx];
        out[output_idx] = sum;
    }
}
// Test: input_size=100,000, kernel_size=32, kernel = averaging filter (1/32 each)
```

The book frames this as the pattern behind **signal filtering/smoothing/edge detection, time-series analysis, and text/character-level convolutions** (e.g. causal convolutions in WaveNet-style architectures). Note the kernel weights are read identically by every thread — the book's own docstring flags this as a candidate for **constant memory** (a small, cached, read-only device memory space you haven't used yet) rather than plain global memory; that's a real, standard CUDA feature, not just a hint — you'll want it as soon as you're optimizing this kernel.

**Deep dive: why constant memory fits this specific access pattern so well.** Every CUDA device exposes a small (typically 64 KB total) region of **constant memory** — read-only from the device side, backed by a small, dedicated per-SM cache separate from the regular L1/L2 hierarchy. Its defining hardware behavior: when every thread in a warp reads the *same* address in constant memory during the same instruction, the hardware services all 32 threads with a **single broadcast read** — one access, not 32. Look back at `conv1d_kernel`'s inner loop: for a fixed `k_idx`, every thread (regardless of its own `output_idx`) reads the exact same `kernel[k_idx]` — precisely the access pattern constant memory's broadcast mechanism was built for. Ordinary global memory has no such broadcast guarantee; that same access pattern there is 32 separate reads that merely *happen* to hit the same cached line if you're lucky. Moving `kernel[]` from a plain `float*` to `__constant__ float kernel[]` (populated via `cudaMemcpyToSymbol` instead of `cudaMemcpy`) turns "hopefully cached" into "guaranteed, hardware-native broadcast" — for exactly the same reason shared-memory tiling turns GEMM's "hopefully cached" reuse into a guarantee (§3.3.2's closing caveat, one memory space over).

### 3.4.3 2D Convolution — the CNN core

`conv2d/conv2d.cu` generalizes the same sliding-window idea to two dimensions, and the book's test case doubles as an actual image-processing demo — a 3×3 Laplacian-style edge-detection filter:

```cuda
void conv2d_cpu(const float* in, float* out, const float* kernel, int height, int width, int kernel_dim) {
    int output_h = height - kernel_dim + 1;
    int output_w = width - kernel_dim + 1;
    for (int r = 0; r < output_h; ++r) {
        for (int c = 0; c < output_w; ++c) {
            float sum = 0.0f;
            for (int kr = 0; kr < kernel_dim; ++kr)
                for (int kc = 0; kc < kernel_dim; ++kc)
                    sum += in[(r+kr) * width + (c+kc)] * kernel[kr * kernel_dim + kc];
            out[r * output_w + c] = sum;
        }
    }
}

__global__ void conv2d_kernel(const float* in, float* out, const float* kernel, int height, int width, int kernel_dim) {
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    int output_h = height - kernel_dim + 1;
    int output_w = width - kernel_dim + 1;
    if (output_row < output_h && output_col < output_w) {
        float sum = 0.0f;
        for (int kr = 0; kr < kernel_dim; ++kr)
            for (int kc = 0; kc < kernel_dim; ++kc)
                sum += in[(output_row+kr) * width + (output_col+kc)] * kernel[kr * kernel_dim + kc];
        out[output_row * output_w + output_col] = sum;
    }
}
// Test: 256×256 image, 3×3 kernel = {-1,-1,-1, -1,8,-1, -1,-1,-1} (Laplacian edge detector)
```

Same shape of inefficiency as GEMM: each thread re-reads its own `kernel_dim × kernel_dim` input window from global memory, and neighboring output threads' windows overlap heavily (a 3×3 kernel means adjacent output pixels share up to 6 of their 9 input reads) with zero reuse. That overlap is exactly what shared-memory **tiling** exploits once you optimize this in Part 5 — load one input tile into shared memory once, let every thread in the block reuse it.

**Deep dive: seeing the 6-of-9 overlap directly.** Write out the two input windows that horizontally-adjacent output pixels `(r, c)` and `(r, c+1)` each read, for a 3×3 kernel — columns `c..c+2` versus columns `c+1..c+3`, both across rows `r..r+2`:

```
Output (r, c) reads:      Output (r, c+1) reads:
 c   c+1  c+2                c+1  c+2  c+3
[X]  [X]  [X]               [X]  [X]  [ ]
[X]  [X]  [X]               [X]  [X]  [ ]
[X]  [X]  [X]               [X]  [X]  [ ]
```

Of each window's 9 reads, **6 land on identical input elements** (the two shared columns, `c+1` and `c+2`, across all 3 rows) — only the 3 in the rightmost new column differ. Two neighboring threads, computed completely independently in this naive kernel, are fetching the same 6 values from global memory twice, with the GPU having no way to know that unless you tell it — which is exactly what a shared-memory tile does: load the input region once per block, and let every thread's overlapping window read from that shared copy instead of re-issuing 6 redundant global loads per neighboring pixel. For a larger kernel (5×5, 7×7 — common in early CNN layers), the overlap fraction grows even higher, and so does the potential win from tiling.

### 3.4.4 2D Max Pooling — spatial reduction

`maxpool2d/maxpool2d.cu` — downsampling by taking the max over non-overlapping `pool_dim × pool_dim` windows:

```cuda
void maxpool2d_cpu(const float* in, float* out, int height, int width, int pool_dim) {
    int output_h = height / pool_dim, output_w = width / pool_dim;
    for (int r = 0; r < output_h; ++r) {
        for (int c = 0; c < output_w; ++c) {
            float max_val = -1e20f;
            for (int pr = 0; pr < pool_dim; ++pr)
                for (int pc = 0; pc < pool_dim; ++pc) {
                    float val = in[(r*pool_dim+pr) * width + (c*pool_dim+pc)];
                    if (val > max_val) max_val = val;
                }
            out[r * output_w + c] = max_val;
        }
    }
}

__global__ void maxpool2d_kernel(const float* in, float* out, int height, int width, int pool_dim) {
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    int output_h = height / pool_dim, output_w = width / pool_dim;
    if (output_row < output_h && output_col < output_w) {
        float max_val = -1e20f;
        for (int pr = 0; pr < pool_dim; ++pr)
            for (int pc = 0; pc < pool_dim; ++pc) {
                float val = in[(output_row*pool_dim+pr) * width + (output_col*pool_dim+pc)];
                if (val > max_val) max_val = val;
            }
        out[output_row * output_w + output_col] = max_val;
    }
}
// Test: 256×256 input, 2x2 pooling → 128×128 output (4x downsampling)
```

The book frames its role precisely: **reduces spatial dimensions, preserves the strongest activation per window, and improves translation invariance** (a small shift in the input often doesn't change which value is the local max). Unlike the other kernels here, this one's windows are **non-overlapping** — no wasted re-reads across neighboring outputs — so it doesn't have GEMM/conv2d's data-reuse problem; its cost is genuinely just `pool_dim²` reads per output, no more, no less.

**Deep dive: "translation invariance," made concrete.** A tiny 1D illustration carries the idea without needing a full 2D grid: take the 4 values `[1, 5, 3, 2]` and 2-wide max-pooling, giving windows `[1,5]` and `[3,2]` → outputs `[5, 3]`. Now shift every value one position to the right (a 1-pixel translation), padding with a repeat of the first value: `[1, 1, 5, 3]` → windows `[1,1]` and `[5,3]` → outputs `[1, 5]`. The output *did* change here — translation invariance is a tendency, not a guarantee, and it's strongest when the shift is small relative to the pooling window and the maximum value doesn't fall right at a window boundary. Try the same shift with a *wider* pool (say, 4-wide over 8 values) and you'll find the max survives far more shifts unchanged, simply because a bigger window has to move further before the true maximum crosses out of it — which is exactly why deeper CNN layers (operating on already-downsampled, lower-resolution feature maps) get more effective translation invariance from the same pool size than shallow ones do.

## 3.5 Why "Naive" Means "Correct but Slow"

The book's own summary of what every kernel above is missing, verbatim in spirit:

- **No shared memory** — every thread loads directly from global memory, every time.
- **No tiling/blocking** — no attempt to keep any block of data resident in fast memory for reuse across threads.
- **No vectorization** — one thread computes exactly one output element, one scalar load/store at a time.
- **Redundant computation where it applies** — softmax being the clearest example, recomputing the same reduction independently in every thread.

Every one of these gaps is a *named*, specific chapter later in this course: shared-memory tiling and vectorization (Part 5), warp-level reductions (Part 5, softmax/layernorm), tensor cores (Part 6). You now have working, verified, honestly-slow baselines for eight operations — which is exactly what CPU-first, correctness-first development is supposed to produce.

## 3.6 Running These Kernels from Python (with `load_inline`)

Every kernel above has been shown as it appears in the book's own `.cu` files — a raw `__global__` function plus a hardcoded `main()`. To actually call one from Python without leaving your existing PyTorch workflow, you need the same idea Chapter 5 formalizes in full (kernel → binding → wrapper), but in its lightest possible form. `torch.utils.cpp_extension.load_inline` compiles a CUDA/C++ source string **in memory, on the spot** — no `Makefile`, no separate `.cu` file on disk, no ahead-of-time `setup.py build_ext` step like Chapter 5 uses. It's the right tool for exactly this situation: trying out a kernel you already have as text, right now, in a Python session.

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------- 3.2 Element-wise ----------
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    a = a.contiguous(); b = b.contiguous();
    auto c = torch::empty_like(a);
    int n = a.numel();
    int threads = 256, blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int num_rows, int num_cols) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < num_rows && column < num_cols) {
        int index = row * num_cols + column;
        C[index] = A[index] + B[index];
    }
}
torch::Tensor matrix_add(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dim() == 2, "A, B must be 2D CUDA tensors");
    auto C = torch::empty_like(A);
    int num_rows = A.size(0), num_cols = A.size(1);
    dim3 threads(16, 16), blocks((num_cols + 15) / 16, (num_rows + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), num_rows, num_cols);
    return C;
}

// ---------- 3.3.1 Transpose ----------
__global__ void transpose_kernel(const float* in, float* out, int num_rows, int num_cols) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < num_rows && column < num_cols) {
        out[column * num_rows + row] = in[row * num_cols + column];
    }
}
torch::Tensor transpose_naive(torch::Tensor in) {
    TORCH_CHECK(in.is_cuda() && in.dim() == 2, "input must be a 2D CUDA tensor");
    int num_rows = in.size(0), num_cols = in.size(1);
    auto out = torch::empty({num_cols, num_rows}, in.options());
    dim3 threads(16, 16), blocks((num_cols + 15) / 16, (num_rows + 15) / 16);
    transpose_kernel<<<blocks, threads>>>(in.data_ptr<float>(), out.data_ptr<float>(), num_rows, num_cols);
    return out;
}

// ---------- 3.3.2 GEMM ----------
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M_rows && column < N_cols) {
        float sum = 0.0f;
        for (int k = 0; k < K_shared_dim; ++k)
            sum += A[row * K_shared_dim + k] * B[k * N_cols + column];
        C[row * N_cols + column] = sum;
    }
}
torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "inner dimensions must match");
    auto C = torch::empty({M, N}, A.options());
    dim3 threads(16, 16), blocks((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

// ---------- 3.4.1 Softmax ----------
__global__ void softmax_naive_kernel(const float* in, float* out, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows && column < num_cols) {
        float max_val = -1e20f;
        for (int c = 0; c < num_cols; ++c)
            if (in[row * num_cols + c] > max_val) max_val = in[row * num_cols + c];
        float sum_exp = 0.0f;
        for (int c = 0; c < num_cols; ++c)
            sum_exp += expf(in[row * num_cols + c] - max_val);
        out[row * num_cols + column] = expf(in[row * num_cols + column] - max_val) / sum_exp;
    }
}
torch::Tensor softmax_naive(torch::Tensor in) {
    TORCH_CHECK(in.is_cuda() && in.dim() == 2, "input must be a 2D CUDA tensor");
    int num_rows = in.size(0), num_cols = in.size(1);
    auto out = torch::empty_like(in);
    dim3 threads(16, 16), blocks((num_cols + 15) / 16, (num_rows + 15) / 16);
    softmax_naive_kernel<<<blocks, threads>>>(in.data_ptr<float>(), out.data_ptr<float>(), num_rows, num_cols);
    return out;
}

// ---------- 3.4.2 Conv1D ----------
__global__ void conv1d_kernel(const float* in, float* out, const float* kernel, int input_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) sum += in[idx + k] * kernel[k];
        out[idx] = sum;
    }
}
torch::Tensor conv1d_naive(torch::Tensor in, torch::Tensor kernel) {
    TORCH_CHECK(in.is_cuda() && kernel.is_cuda(), "inputs must be CUDA tensors");
    int input_size = in.size(0), kernel_size = kernel.size(0);
    int output_size = input_size - kernel_size + 1;
    auto out = torch::empty({output_size}, in.options());
    int threads = 256, blocks = (output_size + threads - 1) / threads;
    conv1d_kernel<<<blocks, threads>>>(in.data_ptr<float>(), out.data_ptr<float>(), kernel.data_ptr<float>(), input_size, kernel_size);
    return out;
}

// ---------- 3.4.3 Conv2D ----------
__global__ void conv2d_kernel(const float* in, float* out, const float* kernel, int height, int width, int kdim) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x, orow = blockIdx.y * blockDim.y + threadIdx.y;
    int oh = height - kdim + 1, ow = width - kdim + 1;
    if (orow < oh && oc < ow) {
        float sum = 0.0f;
        for (int kr = 0; kr < kdim; ++kr)
            for (int kc = 0; kc < kdim; ++kc)
                sum += in[(orow + kr) * width + (oc + kc)] * kernel[kr * kdim + kc];
        out[orow * ow + oc] = sum;
    }
}
torch::Tensor conv2d_naive(torch::Tensor in, torch::Tensor kernel) {
    TORCH_CHECK(in.is_cuda() && kernel.is_cuda(), "inputs must be CUDA tensors");
    int height = in.size(0), width = in.size(1), kdim = kernel.size(0);
    int oh = height - kdim + 1, ow = width - kdim + 1;
    auto out = torch::empty({oh, ow}, in.options());
    dim3 threads(16, 16), blocks((ow + 15) / 16, (oh + 15) / 16);
    conv2d_kernel<<<blocks, threads>>>(in.data_ptr<float>(), out.data_ptr<float>(), kernel.data_ptr<float>(), height, width, kdim);
    return out;
}

// ---------- 3.4.4 MaxPool2D ----------
__global__ void maxpool2d_kernel(const float* in, float* out, int height, int width, int pdim) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x, orow = blockIdx.y * blockDim.y + threadIdx.y;
    int oh = height / pdim, ow = width / pdim;
    if (orow < oh && oc < ow) {
        float m = -1e20f;
        for (int pr = 0; pr < pdim; ++pr)
            for (int pc = 0; pc < pdim; ++pc) {
                float v = in[(orow * pdim + pr) * width + (oc * pdim + pc)];
                if (v > m) m = v;
            }
        out[orow * ow + oc] = m;
    }
}
torch::Tensor maxpool2d_naive(torch::Tensor in, int64_t pool_dim) {
    TORCH_CHECK(in.is_cuda() && in.dim() == 2, "input must be a 2D CUDA tensor");
    int height = in.size(0), width = in.size(1), pdim = (int)pool_dim;
    int oh = height / pdim, ow = width / pdim;
    auto out = torch::empty({oh, ow}, in.options());
    dim3 threads(16, 16), blocks((ow + 15) / 16, (oh + 15) / 16);
    maxpool2d_kernel<<<blocks, threads>>>(in.data_ptr<float>(), out.data_ptr<float>(), height, width, pdim);
    return out;
}
"""

cpp_source = r"""
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor matrix_add(torch::Tensor A, torch::Tensor B);
torch::Tensor transpose_naive(torch::Tensor in);
torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor softmax_naive(torch::Tensor in);
torch::Tensor conv1d_naive(torch::Tensor in, torch::Tensor kernel);
torch::Tensor conv2d_naive(torch::Tensor in, torch::Tensor kernel);
torch::Tensor maxpool2d_naive(torch::Tensor in, int64_t pool_dim);
"""

ch3 = load_inline(
    name="ch3_naive_kernels",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["vector_add", "matrix_add", "transpose_naive", "gemm_naive",
               "softmax_naive", "conv1d_naive", "conv2d_naive", "maxpool2d_naive"],
    verbose=True,
)
```

Note the launcher bodies (`vector_add`, `matrix_add`, and so on) are new — they're the C++ glue Chapter 5 §5.1 will formalize as a distinct "binding" layer — but every `__global__` kernel body is copied verbatim from the sections above, unchanged. The 16×16 block shapes used here for the 2D kernels match this chapter's own established convention (§3.2's `matrix_add` launch), applied consistently to the kernels above whose own exact launch configuration wasn't quoted line-for-line earlier in this chapter.

A single verification script exercises all eight against their PyTorch/`torch.nn.functional` equivalents — the same CPU-first, verify-before-trusting discipline from §3.1, just with PyTorch itself standing in as the "CPU reference":

```python
import torch.nn.functional as F

device = "cuda"
torch.manual_seed(0)

def check(name, custom_out, ref_out, atol=1e-4):
    max_diff = (custom_out - ref_out).abs().max().item()
    print(f"{name:12s} max_diff={max_diff:.2e}  {'PASS' if max_diff < atol else 'FAIL'}")

a, b = torch.randn(1_000_000, device=device), torch.randn(1_000_000, device=device)
check("vector_add", ch3.vector_add(a, b), a + b)

A, B = torch.randn(1024, 1024, device=device), torch.randn(1024, 1024, device=device)
check("matrix_add", ch3.matrix_add(A, B), A + B)

M = torch.randn(512, 256, device=device)
check("transpose", ch3.transpose_naive(M), M.t().contiguous())

A2, B2 = torch.randn(512, 256, device=device), torch.randn(256, 512, device=device)
check("gemm", ch3.gemm_naive(A2, B2), A2 @ B2, atol=1e-3)   # a looser tolerance — see §3.2.1's note on fixed vs. relative tolerance

S = torch.randn(128, 1000, device=device)
check("softmax", ch3.softmax_naive(S), F.softmax(S, dim=1))

sig, kern1d = torch.randn(100_000, device=device), torch.full((32,), 1/32, device=device)
ref_1d = F.conv1d(sig.view(1, 1, -1), kern1d.view(1, 1, -1)).view(-1)
check("conv1d", ch3.conv1d_naive(sig, kern1d), ref_1d)

img = torch.randn(256, 256, device=device)
kern2d = torch.tensor([[-1.,-1.,-1.], [-1.,8.,-1.], [-1.,-1.,-1.]], device=device)
ref_2d = F.conv2d(img.view(1, 1, 256, 256), kern2d.view(1, 1, 3, 3)).view(254, 254)
check("conv2d", ch3.conv2d_naive(img, kern2d), ref_2d)

ref_pool = F.max_pool2d(img.view(1, 1, 256, 256), 2).view(128, 128)
check("maxpool2d", ch3.maxpool2d_naive(img, 2), ref_pool)
```

Every `check()` call should print `PASS`. The one deliberately loosened tolerance (`gemm`, `atol=1e-3`) isn't a bug in the harness — it's §3.2.1's earlier point about fixed-tolerance verification showing up in practice: a 256-term accumulation genuinely differs from PyTorch's own (likely cuBLAS-backed, different summation order) result by more than `1e-4`, even though both are "correct" FP32 arithmetic.

---

## Hands-On Lab

1. **Build and run the real thing**, using the book's actual Makefile:
   ```bash
   cd book.cu/1_naive
   make all      # builds all 8 targets into out/
   make run      # runs all 8, printing CPU/GPU timing + verification for each
   ```
   Or individually: `make gemm && ./out/gemm`, `make run_softmax`, etc.

2. **Compute achieved GFLOPS for your own GEMM run.** The program prints total FLOPs (134,217,728 for the default 512×512×256 case) and GPU time in ms. GFLOPS = `FLOPs / (time_ms * 1e6)`. Run it on both your RTX 3090 and your T4, and compare your two numbers against the book's stated naive range (10–50 GFLOPS) — and against your GPUs' confirmed FP32 peaks (35.58 TFLOPS for the RTX 3090, 8.1 TFLOPS for the T4) to see just how far under peak a naive kernel really runs.

3. **Quantify softmax's redundant work directly.** For the default test (128 rows × 1000 columns), compute: total "useful" work the CPU does (`3 * num_rows * num_cols` operations, roughly) versus total work the naive GPU kernel does (`~2 * num_cols` operations *per thread*, times `num_rows * num_cols` threads). Express the ratio as a multiple.

4. **Run `compute-sanitizer` on all eight binaries** (a good habit now, ahead of Part 9): `compute-sanitizer ./out/gemm`, etc. All eight should report clean — if any doesn't, you've found a real bug worth understanding before moving on.

## Exercises

1. **"Same" padding for `conv2d`.** The book's implementation uses valid (no) padding, so output is smaller than input. Modify `conv2d_kernel` to support "same" padding (output dimensions equal input dimensions) by treating out-of-bounds input reads as zero. Verify against a CPU reference you write yourself.
2. **GEMM roofline, on a different kernel.** §3.3.2's deep dive already worked GEMM's own actual-vs-ideal arithmetic intensity (0.25 vs. 64.0 FLOP/byte) against both your GPUs' ridge points. Now do the same full derivation for **transpose** instead: compute its arithmetic intensity (it's a pure data-movement op, so think carefully about what "ideal" reuse even means when there's no redundant re-reading to eliminate — the 32× transaction-count gap from §3.3.1's deep dive is the more relevant lens here than FLOP/byte). Explain in one paragraph why transpose's bottleneck is fundamentally different in *kind* from GEMM's, not just in degree.
3. **Argmax pooling.** Extend `maxpool2d_kernel` to also output, per pooling window, the *flattened index* of the maximum element (not just its value). You'll need this in Part 3 to implement max-pool's backward pass (gradients flow only through the position that was the max).
4. **Two-pass softmax.** Rewrite `softmax_naive_kernel` as two kernels: the first computes per-row max and sum-of-exp (one thread per row, or better, one block per row with a shared-memory reduction) and writes them to small per-row arrays; the second reads those precomputed values and writes the final softmax output. Verify it's still numerically identical to the naive version, and time both versions against each other on your own hardware — even this simple two-pass restructuring (a "poor man's" optimization ahead of Part 5's real warp-shuffle version) should show a real speedup.
5. **1D convolution in constant memory.** CUDA's `__constant__` memory space is a small (typically 64KB), cached, read-only region ideal for data every thread reads identically — exactly `conv1d`'s filter kernel. Rewrite `conv1d_kernel` to declare the kernel weights as `__constant__` (`cudaMemcpyToSymbol` instead of `cudaMemcpy` for that one array) and compare timing against the global-memory version.

---

**Next:** Chapter 4 — Backpropagation & a Neural Net in Pure CUDA (Part 3). Autodiff theory, the chain rule as a graph traversal, and an MNIST MLP rebuilt five ways: NumPy → PyTorch → single-threaded C → custom CUDA kernels → CUDA + cuBLAS.
