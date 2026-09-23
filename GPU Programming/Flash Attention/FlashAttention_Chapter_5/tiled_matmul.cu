// tiled_matmul.cu
// Chapter 5: tiled matrix multiply using shared memory.
// Device kernel + host launcher only -- no main() here; see test_tiled_matmul.cpp
// for the test driver, and tiled_matmul.h for the launcher's declaration.
//
// Compile as part of the test binary:
//   nvcc -O3 -arch=sm_75 tiled_matmul.cu test_tiled_matmul.cpp -o tiled_matmul_test   (Tesla T4)
//   nvcc -O3 -arch=sm_86 tiled_matmul.cu test_tiled_matmul.cpp -o tiled_matmul_test   (RTX 3090)

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
