# Chapter 5: The Memory Hierarchy and Shared Memory

## 1. Goal

By the end of this chapter you will be able to:

- Explain the GPU memory hierarchy (registers → shared memory → L2 → HBM) in terms of size, latency, and scope.
- Explain *why* the naive kernels from Chapter 4 waste memory bandwidth, in concrete numbers.
- Write a **tiled matrix multiply** kernel that stages data through shared memory, with correct `__syncthreads()` placement.
- Verify the kernel against a CPU reference, including on input sizes that aren't multiples of the tile width.

This chapter's kernel is not attention-specific — it's a generic `C = A @ B`. That's deliberate. It's "the building block of everything after it" (as the syllabus puts it): in Chapter 9 onward, the exact same tiling pattern gets reused to compute `S = QKᵀ` and `O = PV` inside the real FlashAttention kernel. Get the tiling pattern right here, on a problem simple enough to reason about, and Part 2 becomes mostly plumbing.

## 2. Concepts

### 2.1 Why Chapter 4's kernels are slow

The naive `QKᵀ` kernel from Chapter 4 had each thread compute one output element by reading a full row of `Q` and a full column of `Kᵀ` directly from global memory (HBM):

```cpp
// Chapter 4, roughly:
for (int k = 0; k < d; ++k) {
    acc += Q[row * d + k] * K[col * d + k];
}
```

Look at the reuse pattern. Every thread in the same block-row reads the *same* row of `Q` — but each of them re-fetches it from HBM independently, because nothing is shared between threads. If a block computes a 16×16 tile of the output, every element of the input `Q` and `K` tiles gets read from HBM 16 times (once per thread that needs it), instead of once.

This is exactly the "memory-bound" problem from Chapter 1, showing up again one level down: it's not just that attention as a whole re-reads `S` from HBM — it's that even a single tile of a single matmul re-reads its own inputs many times over if you don't do something about it.

### 2.2 The memory hierarchy

CUDA GPUs expose a hierarchy of memories that trade off size and scope against speed. Rough numbers for the two target GPUs:

| Level | Scope | Size (approx.) | Latency (approx.) |
|---|---|---|---|
| Registers | per-thread | 255 32-bit registers/thread (both T4 and RTX 3090) | ~1 cycle |
| Shared memory | per-block (on-chip SRAM) | 48 KB/block by default; up to 64 KB/SM (T4) or ~100 KB/SM (RTX 3090) with opt-in | ~20–30 cycles |
| L2 cache | per-GPU, shared by all SMs | 4 MB (T4) / 6 MB (RTX 3090) | ~200 cycles |
| HBM (global memory) | per-GPU, off-chip | 16 GB (T4) / 24 GB (RTX 3090) | ~400–800 cycles |

The pattern: the closer memory is to the compute units, the smaller and faster it is, and the more narrowly scoped (private to a thread, vs. shared by a block, vs. visible to the whole GPU). This is the same reason a CPU has L1/L2/L3 cache — except here, shared memory is not a cache the hardware manages for you. It's a scratchpad *you* manage explicitly. That's both the opportunity and the extra work in this chapter.

The 48 KB default is a compile-time-visible limit for statically declared `__shared__` arrays (what this chapter uses). Going above it needs an explicit opt-in call, `cudaFuncSetAttribute`, which Chapter 9 covers when the real kernel's shared-memory budget gets tight.

### 2.3 Tiling: turning re-reads from HBM into re-reads from shared memory

The fix: instead of every thread independently fetching from HBM, have the *block* cooperatively load a tile of `A` and a tile of `B` into shared memory once, synchronize, and then have every thread in the block reuse that same on-chip copy for its own accumulation.

Concretely, for a `TILE_WIDTH × TILE_WIDTH` block computing a `TILE_WIDTH × TILE_WIDTH` output tile:

- **Without tiling:** each output element does `K` reads from `A` and `K` reads from `B`, all from HBM. A `TILE_WIDTH × TILE_WIDTH` tile of output does `2 × TILE_WIDTH² × K` HBM reads.
- **With tiling:** each `TILE_WIDTH × TILE_WIDTH` tile of `A` (and of `B`) is loaded from HBM exactly once per tile-step, by exactly `TILE_WIDTH²` threads doing one read each. Over `K / TILE_WIDTH` tile-steps, that's `2 × TILE_WIDTH × K` HBM reads for the whole output tile.

That's a reduction in HBM traffic by a factor of `TILE_WIDTH` — for `TILE_WIDTH = 16`, a 16× cut in global memory traffic for the same arithmetic. The arithmetic intensity (FLOPs per byte moved from HBM) goes up by the same factor, which is exactly the lever Chapter 1 said attention needed to pull.

### 2.4 `__syncthreads()`: why two barriers per tile-step, not zero or one

Shared memory is only useful if all threads agree on when it's safe to read what's been written. `__syncthreads()` is a block-wide barrier: no thread in the block proceeds past it until every thread has reached it. The tiled loop needs exactly two:

1. **After loading, before computing.** Threads load different elements of the tile in parallel. A thread that needs `As[ty][k]` for some `k` it didn't personally load must wait until *whichever thread did* load it has finished. Skip this barrier and you read a shared-memory slot that some other thread hasn't written yet — a race, and on real hardware it doesn't crash, it just silently returns garbage or stale data from the previous tile.

2. **After computing, before the next load.** The same shared-memory buffer (`As`, `Bs`) is reused on the next loop iteration for the next tile. A thread that's still reading `As[ty][k]` for the *current* tile must not have that buffer overwritten by a fast thread that's already moved on to loading the *next* tile. Skip this barrier and a slow thread computes with a mix of old-tile and new-tile data.

Both bugs are classic "works on small inputs, silently wrong on large ones" bugs, because whether the race is actually hit depends on scheduling, not on your code being "mostly right."

## 3. Code walkthrough

This chapter ships five files, following the same split Chapter 1 used between the implementation, its test, and dependency-free Python companions:

- **`tiled_matmul.h`** — declares the host launcher, so host code can call the kernel without needing to see CUDA kernel syntax.
- **`tiled_matmul.cu`** — the kernel and its launcher only. No `main()` here — this plays the same role `naive_attention.py` played in Chapter 1: the reusable piece everything else calls.
- **`test_tiled_matmul.cpp`** — the test driver: builds random inputs, calls the kernel through `tiled_matmul.h`, checks the result against a CPU reference. The C++ counterpart to `test_naive_attention.py`.
- **`tiled_matmul_numpy.py`** — a numpy twin of the tiling algorithm (same nested-loop structure, no CUDA), so the *blocking pattern itself* can be checked before any C++ syntax gets in the way — the same role `verify_numpy.py` played in Chapter 1.
- **`tiling_savings.py`** — the HBM-traffic, arithmetic-intensity, and shared-memory numbers from §2.2–2.3, computed straight from the formulas — the same role `roofline_numbers.py` played in Chapter 1.

Compile the CUDA pieces together into one test binary:

```
nvcc -O3 -arch=sm_75 tiled_matmul.cu test_tiled_matmul.cpp -o tiled_matmul_test   # Tesla T4
nvcc -O3 -arch=sm_86 tiled_matmul.cu test_tiled_matmul.cpp -o tiled_matmul_test   # RTX 3090
```

**`tiled_matmul.h`**

```cpp
#pragma once

// Tiled matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// Kernel + host launcher live in tiled_matmul.cu; declared here so host code
// (test_tiled_matmul.cpp) doesn't need to see CUDA kernel syntax to call it.

void launchTiledMatMul(const float* dA, const float* dB, float* dC,
                        int M, int K, int N);
```

**`tiled_matmul.cu`**

```cpp
// tiled_matmul.cu
// Chapter 5: tiled matrix multiply using shared memory.
// Device kernel + host launcher only -- no main() here; see test_tiled_matmul.cpp
// for the test driver, and tiled_matmul.h for the launcher's declaration.

#include "tiled_matmul.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define TILE_WIDTH 16

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

__global__ void tiledMatMulKernel(const float* A, const float* B, float* C,
                                   int M, int K, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;   // row of C this thread owns
    int col = blockIdx.x * TILE_WIDTH + tx;   // col of C this thread owns

    float acc = 0.0f;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_WIDTH + tx;
        int bRow = t * TILE_WIDTH + ty;

        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();  // wait for the whole tile to land in shared memory

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();  // wait for everyone to finish reading before the next load overwrites
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launchTiledMatMul(const float* dA, const float* dB, float* dC,
                        int M, int K, int N) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
               (M + TILE_WIDTH - 1) / TILE_WIDTH);
    tiledMatMulKernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}
```

**`test_tiled_matmul.cpp`**

```cpp
// test_tiled_matmul.cpp
// Chapter 5: host-side test driver for the tiled matmul kernel.
// Builds random inputs, runs the GPU kernel via tiled_matmul.h, checks the
// result against a plain triple-loop CPU reference.

#include "tiled_matmul.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

// Plain triple-loop CPU reference.
static void cpuMatMul(const std::vector<float>& A, const std::vector<float>& B,
                       std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

int main() {
    int M = 250, K = 130, N = 300;  // deliberately not multiples of TILE_WIDTH (16)

    std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
    for (auto& x : hA) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (auto& x : hB) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    launchTiledMatMul(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cpuMatMul(hA, hB, hRef, M, K, N);

    float maxErr = 0.0f;
    for (int i = 0; i < M * N; ++i) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    printf("Max abs error vs CPU reference: %e\n", maxErr);
    bool pass = maxErr < 1e-3f;
    printf(pass ? "PASS\n" : "FAIL\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return pass ? 0 : 1;
}
```

**A note on the correctness check.** Every other chapter verifies against the Chapter 1 PyTorch attention oracle. This one doesn't, on purpose: `tiled_matmul.cu` computes a generic `A @ B`, not attention, so there's no attention output to compare it to yet. `test_tiled_matmul.cpp` checks it instead against a straightforward triple-loop CPU implementation, `cpuMatMul`, which is the same "obviously correct, slow" role the Chapter 1 oracle plays — and `tiled_matmul_numpy.py` checks the *algorithm* the same way, one level earlier, before any CUDA is involved at all. When this exact tiling pattern gets specialized into the real `S = QKᵀ` kernel in Chapter 11, *that* kernel goes back to being checked against Chapter 1.

### Line-by-line

- **`As`, `Bs`** — two `TILE_WIDTH × TILE_WIDTH` shared-memory tiles, one to hold a piece of `A`, one to hold a piece of `B`. Every thread in the block sees the same `As` and `Bs`.
- **`row`, `col`** — each thread owns exactly one output element `C[row][col]`, chosen by combining which block it's in (`blockIdx`) with its position inside the block (`threadIdx`) — the same indexing scheme as Chapter 4's naive kernel.
- **`numTiles`** — how many `TILE_WIDTH`-sized steps it takes to walk across the `K` dimension. The ceiling division (`(K + TILE_WIDTH - 1) / TILE_WIDTH`) handles `K` not dividing evenly.
- **The load** — `As[ty][tx] = A[row * K + aCol]`. Each thread loads exactly one element of `A` and one of `B` into shared memory. Notice the *thread's own* `(ty, tx)` picks the shared-memory slot, while `(row, aCol)` / `(bRow, col)` pick the global-memory source — those are different index expressions on purpose, since the tile being loaded doesn't correspond 1:1 to the output element this thread will eventually write.
- **The bounds check** — `(row < M && aCol < K) ? A[...] : 0.0f`. When `M`, `K`, or `N` isn't a multiple of `TILE_WIDTH`, the last tile along any dimension is partially out of bounds. Reading out of bounds would be undefined behavior; padding with `0.0f` instead is mathematically inert for a sum of products (anything times a `0` contributes `0`), so the result stays correct without needing a separate code path for edge tiles.
- **First `__syncthreads()`** — every thread in the block has now written its one element of `As` and `Bs`. Nobody may read from either array until this point is reached by everybody.
- **The inner `for` over `k`** — this is where the actual work happens: a `TILE_WIDTH`-long dot product, but now every read (`As[ty][k]`, `Bs[k][tx]`) comes from shared memory instead of HBM. `#pragma unroll` asks the compiler to unroll this fixed-length (`TILE_WIDTH`, known at compile time) loop, trading code size for removing loop-overhead instructions — a preview of Chapter 7.
- **Second `__syncthreads()`** — every thread has finished reading `As`/`Bs` for this tile-step. Only now is it safe for the next iteration to start overwriting them with the next tile.
- **The final write** — `C[row * N + col] = acc`, guarded by the same bounds check pattern as the load, since a thread can own a `(row, col)` that's outside the true output shape when `M` or `N` isn't a multiple of `TILE_WIDTH`.

## 4. C++ decoded

**`__shared__ float As[TILE_WIDTH][TILE_WIDTH];`**
`__shared__` is a CUDA-specific storage qualifier, on top of ordinary C++ array syntax. It tells the compiler: allocate this array once *per block* (not once per thread — every thread in the block sees the exact same `As`), place it in on-chip shared memory instead of registers or global memory, and its contents live for as long as the block is running. There's no Python equivalent to "memory scoped to a group of parallel workers" — the closest mental model is a `multiprocessing.Array` created fresh for each pool of worker processes and thrown away when they finish, except here it's on-chip and far faster.

Because `TILE_WIDTH` is a compile-time constant (`#define`), the size of `As` is known at compile time. This is what makes it "static" shared memory, as opposed to the `extern __shared__` dynamically-sized version Chapter 9 introduces for when the tile size isn't fixed until kernel-launch time.

**2D indexing: `As[ty][tx]`**
This looks like indexing a 2D array in Python (`As[ty][tx]`), but the compiler doesn't store `As` as an array of pointers to rows the way, say, a Python list of lists would be. Because the shape `TILE_WIDTH × TILE_WIDTH` is fixed at compile time, the compiler lays `As` out as one flat, contiguous block of `TILE_WIDTH * TILE_WIDTH` floats in shared memory, row-major — and rewrites every `As[ty][tx]` into the flat address `ty * TILE_WIDTH + tx` for you, at compile time, with no runtime cost. This is the same row-major flattening from Chapter 3's `Q[i*d + j]`, except here the compiler is doing the flattening on your behalf because the array's dimensions are statically known, instead of you writing the flat index by hand because the array was allocated with `cudaMalloc` at a size only known at runtime.

**`#pragma unroll`**
Not C++ syntax at all — a *pragma* is a directive to the compiler, outside the language proper, that this particular compiler (`nvcc`) understands. It says: this `for` loop has a fixed, compile-time-known number of iterations (`TILE_WIDTH`), so instead of generating one copy of the loop body plus a loop counter and a branch, generate `TILE_WIDTH` copies of the body back to back with no branching. This trades a larger compiled kernel for fewer instructions executed per thread — more on when this trade is worth it in Chapter 7.

## 5. Common pitfalls

- **Missing the first `__syncthreads()`.** A thread reads `As[ty][k]` for a `k` some other thread was responsible for loading, before that thread has written it. Symptom: results are wrong, and the *specific* way they're wrong can change between runs, because it depends on warp scheduling.
- **Missing the second `__syncthreads()`.** A fast thread starts overwriting `As`/`Bs` for the next tile while a slow thread in the same block is still reading the current tile's values. Symptom: same as above — nondeterministic wrongness, often only showing up at larger `K`.
- **Forgetting the boundary `? ... : 0.0f` guard.** Either an out-of-bounds global memory read (undefined behavior, can crash or silently read garbage), or — if you instead skip the bounds check on the *write* — writing past the end of `C`, corrupting unrelated memory.
- **Using the wrong index to pick the shared-memory slot vs. the global-memory address.** It's easy to write `As[row][col]` out of habit instead of `As[ty][tx]`. `row`/`col` are global output coordinates that can be arbitrarily large; the shared-memory tile only has `TILE_WIDTH` slots in each dimension. This is an out-of-bounds write into shared memory, usually corrupting `Bs` or another block's memory rather than crashing cleanly.
- **Oversizing `TILE_WIDTH` for the shared-memory budget.** Two `float` tiles of `TILE_WIDTH × TILE_WIDTH` use `2 × TILE_WIDTH² × 4` bytes of shared memory. At `TILE_WIDTH = 16` that's 2 KB — comfortably under the 48 KB default. Push `TILE_WIDTH` up carelessly (or add more shared arrays later, as the real attention kernel will) and you can exceed the per-block budget, which is a launch failure, not a slow kernel — `cudaGetLastError()` after the launch is what catches this, which is exactly why the error-check macro wraps every CUDA call.

## 6. Exercises

1. **Break the barriers on purpose.** Comment out just the first `__syncthreads()` and run the test. Does it fail? Now restore it and comment out only the second one instead. Does *that* fail, and does it fail the same way? (It may not fail reliably every run — that unreliability is itself the lesson.)
2. **Boundary math.** With `M = 250`, `K = 130`, `N = 300`, and `TILE_WIDTH = 16`, how many tile-steps does the kernel take (`numTiles`)? For the *last* tile-step, which threads are loading real data and which are loading the zero-pad? Work it out on paper, then confirm your loaded-vs-padded thread count against what the bounds-check expression actually decides at that boundary.
3. **Shared memory budget.** Compute the shared memory used per block in bytes as a function of `TILE_WIDTH`. Given the SM-level shared memory totals in §2.2, what's the largest `TILE_WIDTH` (as a power of 2) you could use while still fitting at least 2 concurrently-resident blocks per SM on the T4? On the RTX 3090? Check your answer against `tiling_savings.py`, which computes this directly.
4. **Toward `QKᵀ`.** Attention needs `S = Q @ Kᵀ`, not `A @ B` — the second matrix is transposed. Modify the kernel to compute `C = A @ Bᵀ` where `B` is `N × K` instead of `K × N`. Which index expression changes: the load of `Bs`, the inner product loop, or both? (This is most of the real work behind Chapter 11.)
5. **Measure the win.** Using `cudaEvent_t` timing (a preview of Chapter 14), compare this tiled kernel's runtime against Chapter 4's naive `QKᵀ` kernel on a matching problem size. Does the speedup roughly track the `TILE_WIDTH`× reduction in HBM traffic predicted in §2.3, or is it smaller? If smaller, what other costs (e.g. the `__syncthreads()` barriers themselves) might be eating into the win?
