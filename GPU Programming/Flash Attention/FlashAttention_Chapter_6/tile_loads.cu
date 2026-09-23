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
//
// Compile as part of the test binary:
//   nvcc -O3 -arch=sm_75 tile_loads.cu test_tile_loads.cpp -o tile_loads_test   (Tesla T4)
//   nvcc -O3 -arch=sm_86 tile_loads.cu test_tile_loads.cpp -o tile_loads_test   (RTX 3090)

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
