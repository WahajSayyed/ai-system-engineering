# Chapter 6: Coalescing and Vectorized Loads

## 1. Goal

By the end of this chapter you will be able to:

- Explain what "coalesced" global memory access means in terms of the fixed-size (32-byte) sectors the hardware actually moves.
- Predict, for a given warp access pattern, roughly how many memory transactions it costs — and check that prediction with a small simulator.
- Write a **vectorized load** (`float4`) that moves 4 floats in one instruction instead of one, applied to a `[N, D]` tile shaped exactly like a Q, K, or V tile will be.
- Know precisely when `float4` is legal — the alignment and divisibility requirements — and what happens when those aren't met.

This chapter is about a lever that sits *on top of* Chapter 5's tiling, not a replacement for it: tiling got the data into shared memory with far less HBM traffic; this chapter is about doing those loads themselves more efficiently, in preparation for the real Q/K/V tile loads in Chapter 10.

## 2. Concepts

### 2.1 What "coalescing" means, mechanically

A **warp** is a group of 32 threads that execute the same instruction together, in lockstep (Chapter 7 goes into this properly — for now, the fact that matters is: when one thread in a warp issues a load, so do the other 31, all at once, and the hardware handles all 32 addresses as a single request).

On both target GPUs (Turing T4, Ampere RTX 3090), global memory is serviced through the L2 cache in fixed **32-byte sectors**. When a warp's 32 threads issue a load instruction, the hardware looks at all 32 addresses together, works out which 32-byte sectors those addresses fall into, and issues one memory transaction per *distinct* sector needed:

- **Fully coalesced:** 32 threads read 32 consecutive 4-byte floats, starting at a sector-aligned address. That's 128 bytes total, falling into exactly 4 sectors. 4 transactions move exactly the 128 bytes requested — zero waste.
- **Strided / scattered:** 32 threads each read a float 128 bytes apart from the next (e.g., a column access into a row-major matrix). Each thread now lands in its *own* sector. 32 transactions move 32 × 32 = 1024 bytes to deliver 128 bytes of actual data — 8× more traffic for the same useful bytes.

This is the same memory-bound story from Chapter 1, one level down: even loading a tile of `Q` — which Chapter 1 assumed you could just "read once" — can quietly cost 8× the memory traffic it should, if the access pattern inside a warp is wrong.

**Alignment** compounds this: transactions are always sector-aligned, so a request that starts mid-sector can spill into one extra sector it wouldn't otherwise need. This isn't just a performance detail for vectorized loads — a `float4` load whose address isn't a multiple of 16 bytes is not merely slower, it's reading the wrong bytes relative to what the pointer arithmetic implies, and can produce a runtime misaligned-address error. Alignment for vector loads is a correctness requirement, not a tuning knob.

### 2.2 Vectorized loads: fewer instructions for the same bytes

`float4` is a built-in CUDA struct holding 4 consecutive floats (16 bytes), loadable and storable in a single 128-bit instruction (`LDG.128` / `STG.128` at the PTX/SASS level) instead of four separate 32-bit ones (`LDG.32`).

Even when the scalar version was *already* fully coalesced, vectorizing can still help: each instruction has fixed overhead (fetch, decode, address generation), so moving the same bytes through a quarter as many instructions reduces pressure on the load/store units — which matters most in tile-loading code, where each thread only needs a handful of elements and instruction overhead is a real fraction of the total time, not dwarfed by the transfer itself.

The fp16 analog is `half2`: two `__half` values (4 bytes total) in one 32-bit load. Chapter 21 introduces `__half`/`__half2` properly once dtypes are handled formally — for now it's worth knowing the pattern is the same idea, just at a different width.

### 2.3 Applying this to Q, K, V-shaped tiles

Once Part 2 starts, a tile of Q loaded into shared memory is `[Br, d]`: `Br` rows (queries), each row `d` floats wide, where `d` is the head dimension — 32, 64, or 128 in practice. Every row is contiguous in memory, which is exactly what `float4` wants: 4 consecutive elements, one instruction.

The catch is alignment *across* rows, not within one. Row `i` starts at flat offset `i * d` elements. For every row's start to be a legal `float4` base address (a multiple of 4 elements, 16 bytes), you need:

1. `d` itself to be a multiple of 4 — true for 32, 64, 128; **false** for something like 63.
2. The buffer's base pointer to be 16-byte aligned — guaranteed by `cudaMalloc`, which aligns allocations to at least 256 bytes.

Both hold for realistic head dimensions, which is why this technique is safe to reach for on Q/K/V tiles specifically — and why it silently (or loudly) breaks the moment you point it at a tile whose row width isn't a multiple of 4.

## 3. Code walkthrough

Same five-file split as Chapter 5:

- **`tile_loads.h`** — declares the four launchers.
- **`tile_loads.cu`** — the kernels and launchers, no `main()`.
- **`test_tile_loads.cpp`** — the test driver.
- **`tile_loads_numpy.py`** — pure-arithmetic grounding for row alignment (§2.3), runnable with no GPU.
- **`coalescing_numbers.py`** — a sector-touching simulator for §2.1, the tool for Exercise 3.

Compile:

```
nvcc -O3 -arch=sm_75 tile_loads.cu test_tile_loads.cpp -o tile_loads_test   # Tesla T4
nvcc -O3 -arch=sm_86 tile_loads.cu test_tile_loads.cpp -o tile_loads_test   # RTX 3090
```

**`tile_loads.h`**

```cpp
#pragma once

// Chapter 6: coalescing and vectorized loads. Kernels + host launchers live
// in tile_loads.cu; declared here so test_tile_loads.cpp doesn't need to see
// CUDA kernel syntax to call them.

// Part 1: coalesced vs strided access (SS2.1).
void launchCopyCoalesced(const float* dIn, float* dOut, int n);
void launchCopyStrided(const float* dIn, float* dOut, int n, int stride);

// Part 2: scalar vs float4-vectorized load of a [N, D] tile, D == 64 (SS2.2-2.3).
// D and ROWS_PER_BLOCK are fixed in tile_loads.cu; N must be a multiple of
// ROWS_PER_BLOCK (8) for this chapter's kernels -- boundary handling for
// arbitrary N returns properly in Chapter 10.
void launchScalarTileLoad(const float* dIn, float* dOut, int N);
void launchVectorizedTileLoad(const float* dIn, float* dOut, int N);
```

**`tile_loads.cu`**

```cpp
// tile_loads.cu
// Chapter 6: coalescing and vectorized loads.
// Two families of kernels, library-only (no main()) -- see
// test_tile_loads.cpp for the driver, tile_loads.h for the declarations.
//
//   1. copyCoalescedKernel / copyStridedKernel
//      A minimal A/B comparison: same amount of data copied, only the
//      *pattern* of addresses a warp touches differs.
//   2. scalarTileLoadKernel / vectorizedTileLoadKernel
//      Load-then-store-back of a [N, D] tile shaped like a Q/K/V tile
//      (D = head dim, a multiple of 4) -- once with one float per thread,
//      once with one float4 (4 floats) per thread.

#include "tile_loads.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Part 1: coalesced vs strided access, isolated from everything else.
// Every thread WRITES its own index i (always coalesced); only the READ
// index differs, so any transaction-count difference comes purely from the
// read pattern.
// ---------------------------------------------------------------------------

__global__ void copyCoalescedKernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

__global__ void copyStridedKernel(const float* in, float* out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        long long idx = (static_cast<long long>(i) * stride) % n;  // avoid 32-bit overflow
        out[i] = in[static_cast<int>(idx)];
    }
}

void launchCopyCoalesced(const float* dIn, float* dOut, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    copyCoalescedKernel<<<blocks, threads>>>(dIn, dOut, n);
    CUDA_CHECK(cudaGetLastError());
}

void launchCopyStrided(const float* dIn, float* dOut, int n, int stride) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    copyStridedKernel<<<blocks, threads>>>(dIn, dOut, n, stride);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Part 2: loading a Q/K/V-shaped [N, D] tile, scalar vs vectorized.
// D must be a multiple of 4 for the vectorized kernel (SS2.3 / Exercise 1) --
// realistic head dims (32, 64, 128) all satisfy this.
// ---------------------------------------------------------------------------

#define D 64
#define ROWS_PER_BLOCK 8

__global__ void scalarTileLoadKernel(const float* in, float* out, int N) {
    __shared__ float tile[ROWS_PER_BLOCK][D];

    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    int col = threadIdx.x;

    tile[threadIdx.y][col] = in[row * D + col];   // one float per thread: LDG.32
    __syncthreads();
    out[row * D + col] = tile[threadIdx.y][col];   // one float per thread: STG.32
}

__global__ void vectorizedTileLoadKernel(const float* in, float* out, int N) {
    __shared__ alignas(16) float tile[ROWS_PER_BLOCK][D];

    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    int vecCol = threadIdx.x;   // which group of 4 columns this thread owns

    const float4* inRow = reinterpret_cast<const float4*>(&in[row * D]);
    float4* tileRow = reinterpret_cast<float4*>(&tile[threadIdx.y][0]);
    tileRow[vecCol] = inRow[vecCol];   // 4 floats in one instruction: LDG.128
    __syncthreads();

    float4* outRow = reinterpret_cast<float4*>(&out[row * D]);
    outRow[vecCol] = tileRow[vecCol];   // 4 floats in one instruction: STG.128
}

void launchScalarTileLoad(const float* dIn, float* dOut, int N) {
    dim3 block(D, ROWS_PER_BLOCK);
    dim3 grid(N / ROWS_PER_BLOCK);
    scalarTileLoadKernel<<<grid, block>>>(dIn, dOut, N);
    CUDA_CHECK(cudaGetLastError());
}

void launchVectorizedTileLoad(const float* dIn, float* dOut, int N) {
    dim3 block(D / 4, ROWS_PER_BLOCK);
    dim3 grid(N / ROWS_PER_BLOCK);
    vectorizedTileLoadKernel<<<grid, block>>>(dIn, dOut, N);
    CUDA_CHECK(cudaGetLastError());
}
```

**`test_tile_loads.cpp`**

```cpp
// test_tile_loads.cpp
// Chapter 6: host-side test driver.
//   1. Coalesced vs strided copy: both must reproduce the same values
//      (out[i] == in[(i*stride) % n]) -- the point of this chapter isn't
//      that one is "correct" and the other isn't, it's that they cost very
//      different numbers of memory transactions. See coalescing_numbers.py
//      for that count, and Chapter 24 for measuring it for real with
//      Nsight Compute.
//   2. Scalar vs vectorized tile load: both must exactly reproduce the
//      input (a pure load-then-store-back), and must exactly agree with
//      each other.

#include "tile_loads.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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

static bool testCopyKernels() {
    const int n = 4096;
    const int stride = 33;

    std::vector<float> hIn(n), hCoalesced(n), hStrided(n);
    for (int i = 0; i < n; ++i) hIn[i] = static_cast<float>(i);

    float *dIn, *dCoalesced, *dStrided;
    CUDA_CHECK(cudaMalloc(&dIn, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCoalesced, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dStrided, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    launchCopyCoalesced(dIn, dCoalesced, n);
    launchCopyStrided(dIn, dStrided, n, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hCoalesced.data(), dCoalesced, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hStrided.data(), dStrided, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (hCoalesced[i] != hIn[i]) { ok = false; break; }
        long long idx = (static_cast<long long>(i) * stride) % n;
        if (hStrided[i] != hIn[idx]) { ok = false; break; }
    }

    printf("copy kernels (coalesced + strided, n=%d, stride=%d): %s\n",
           n, stride, ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dCoalesced); cudaFree(dStrided);
    return ok;
}

static bool testTileLoadKernels() {
    const int N = 128;   // rows; must be a multiple of ROWS_PER_BLOCK (8)
    const int D = 64;    // head-dim-sized row width, matches tile_loads.cu

    std::vector<float> hIn(N * D), hScalarOut(N * D), hVectorOut(N * D);
    for (int i = 0; i < N * D; ++i) hIn[i] = static_cast<float>(i % 997) * 0.01f;

    float *dIn, *dScalarOut, *dVectorOut;
    CUDA_CHECK(cudaMalloc(&dIn, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dScalarOut, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dVectorOut, N * D * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));

    launchScalarTileLoad(dIn, dScalarOut, N);
    launchVectorizedTileLoad(dIn, dVectorOut, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hScalarOut.data(), dScalarOut, N * D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVectorOut.data(), dVectorOut, N * D * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N * D; ++i) {
        if (hScalarOut[i] != hIn[i] || hVectorOut[i] != hIn[i]) { ok = false; break; }
    }

    printf("tile load kernels (scalar + vectorized, N=%d, D=%d): %s\n", N, D, ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dScalarOut); cudaFree(dVectorOut);
    return ok;
}

int main() {
    bool ok = true;
    ok &= testCopyKernels();
    ok &= testTileLoadKernels();
    return ok ? 0 : 1;
}
```

### Line-by-line

- **`copyCoalescedKernel` / `copyStridedKernel`** — deliberately hold the *write* side fixed (`out[i]`, always contiguous) and vary only the *read* index, so any difference in behavior is isolated to the read pattern, not muddied by the write also changing.
- **`(static_cast<long long>(i) * stride) % n`** — the product `i * stride` can exceed what a 32-bit `int` holds well before `i` reaches typical array sizes; casting to `long long` before multiplying avoids silently wrapping around. The result is cast back to `int` only after the modulo, once it's safely back in range.
- **`scalarTileLoadKernel`** — identical structure to Chapter 5's tile load: each thread reads one element into shared memory, syncs, then writes it back out. `D = 64` and `ROWS_PER_BLOCK = 8` are fixed compile-time constants (like Chapter 5's `TILE_WIDTH`), so `tile` can be statically sized.
- **`vectorizedTileLoadKernel`** — same shared-memory shape, but only `D/4 = 16` threads span a row instead of `D = 64`. Each thread reinterprets its slice of `in`, `tile`, and `out` as `float4*` and moves 4 elements per instruction. `vecCol` indexes *groups* of 4 columns, not individual columns.
- **`alignas(16)`** on the shared `tile` array — makes the 16-byte alignment requirement for the `float4` reinterpret explicit, rather than relying on it happening to already be true.
- **`reinterpret_cast<const float4*>(&in[row * D])`** — treats the address of `in[row*D]` (a `float*`) as if it pointed to `float4`s instead. This is a reinterpretation of the same bytes, not a conversion of values — see §4.

## 4. C++ decoded

**`reinterpret_cast<const float4*>(&in[row * D])`**
A `reinterpret_cast` tells the compiler: don't convert this value, just treat the same bits as a different type. Here, a `const float*` (pointing at one `float`) becomes a `const float4*` (pointing at 4 consecutive `float`s packaged as one struct). No data moves and no conversion happens — it's the C++ equivalent of viewing the same bytes through a different lens, similar in spirit to a NumPy array's `.view()` (reinterpreting the same buffer as a different dtype) rather than `.astype()` (which actually converts values). The requirement that comes with this power: the address being reinterpreted must actually be validly aligned for the target type, or the reinterpretation is undefined behavior — the compiler trusts you completely here and checks nothing.

**`float4`**
A CUDA built-in vector type: a plain struct of 4 `float` fields (accessible as `.x .y .z .w`), sized and aligned so the compiler can emit a single 128-bit load or store instruction for it, instead of four separate 32-bit ones. It's not "4 floats you loop over" — it's one hardware-recognized unit.

**`alignas(16)`**
A C++11 alignment specifier placed on a declaration. `alignas(16) float tile[...]` tells the compiler: guarantee this array starts at a memory address that's a multiple of 16 bytes, no matter what alignment the surrounding context would otherwise give it. Without it, nothing in the language *forces* misalignment — but relying on an alignment you haven't actually asked for is exactly the kind of thing that works by accident today and breaks on a different compiler flag, block size, or architecture tomorrow.

## 5. Common pitfalls

- **`float4` on an unaligned or non-multiple-of-4 offset.** The single most common way this technique breaks. Symptom ranges from wrong values (silently reading across the intended boundary) to an explicit misaligned-address runtime error — never a slow-but-correct result.
- **Assuming vectorizing always helps.** If a kernel is already DRAM-bandwidth-bound (moving a lot of data relative to its instruction count), cutting the instruction count further does very little — the bottleneck was never instruction issue rate. Vectorizing pays off most when a kernel issues many small per-thread loads, where instruction overhead is a real fraction of the time.
- **Conflating "coalesced" with "vectorized."** They're independent axes. Chapter 5's scalar loads were already fully coalesced — vectorizing them doesn't fix a bandwidth problem, since there wasn't one; it only reduces instruction count. Conversely, a `float4` load issued at a strided or misaligned offset is still badly behaved — vectorizing a bad access pattern doesn't fix the pattern.
- **Forgetting the divisibility requirement when reusing this on a different shape.** This chapter's kernels hard-assume `D` is a multiple of 4 and `N` is a multiple of `ROWS_PER_BLOCK`. Point them at shapes that don't satisfy this and you get silently wrong indexing, not a clean failure — see Exercise 1.
- **32-bit overflow in index arithmetic.** `i * stride` for `int i, stride` can overflow before you'd expect, especially once `stride` gets large in the transaction-counting exercises. The strided copy kernel casts to `long long` specifically to sidestep this — a real bug class, not just a style preference.

## 6. Exercises

1. **Break the divisibility assumption.** Change `D` in `tile_loads.cu` to `63` and try to adapt `vectorizedTileLoadKernel` to it without changing anything else about the launch configuration. What goes wrong first — a compile-time issue, a wrong answer, or a runtime crash? Now check your reasoning against `tile_loads_numpy.py`'s alignment table for `d=63`.
2. **See the alignment, don't just reason about it.** Add a small `printf` in `vectorizedTileLoadKernel` (guarded to fire once) that prints `reinterpret_cast<uintptr_t>(&in[row * D]) % 16` for the first few rows. Confirm it's always `0` for `D=64`. Predict, then check, what it looks like for `D=63`.
3. **Transaction counting.** Use `coalescing_numbers.py`'s `simulate_warp_load` to compute the sectors touched for `stride` in `(1, 2, 4, 8, 32, 33)`. At what stride does the waste stop getting worse, and why does that ceiling exist for a 32-thread warp doing 4-byte loads?
4. **Vectorize the coalescing demo.** Rewrite `copyCoalescedKernel` to move `float4`s instead of individual `float`s. Does this change the number of memory transactions relative to the scalar coalesced version, or only the instruction count? Check your prediction with `coalescing_numbers.py`'s vectorized case.
5. **`half2`, on paper only.** Chapter 21 covers `__half`/`__half2` for real — for now, work out: if Q/K/V were stored in fp16, how many elements would one `half2` load move, and what's the smallest head dimension `d` that keeps every row `half2`-aligned? Compare that requirement to the `float4`/fp32 one from §2.3.
