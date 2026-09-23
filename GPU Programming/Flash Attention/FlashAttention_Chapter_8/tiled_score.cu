// tiled_score.cu
// Chapter 8: templates and compile-time configuration.
// Computes C[M,N] = A[M,D] @ B[N,D]^T -- foreshadows S = Q K^T (Chapter 11) --
// with the tile shape (Br, Bc, D) as compile-time template parameters instead
// of Chapter 5/6's per-file #defines.
//
// Compile as part of the test binary:
//   nvcc -O3 -arch=sm_75 tiled_score.cu test_tiled_score.cpp -o tiled_score_test   (Tesla T4)
//   nvcc -O3 -arch=sm_86 tiled_score.cu test_tiled_score.cpp -o tiled_score_test   (RTX 3090)

#include "tiled_score.h"
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
// Templated tile loader: Rows and D are template parameters, so the same
// source serves both the [Br, D] "A" tile and the [Bc, D] "B" tile below --
// two call sites, one function, no duplicated loop body. D's if constexpr
// branch (Chapter 6's vectorized vs scalar load) is resolved per
// instantiation, not per call.
// ---------------------------------------------------------------------------

template <int Rows, int D>
__device__ __forceinline__ void loadTileRowMajor(const float* src, float dst[][D],
                                                  int tileStart, int numRows,
                                                  int tid, int numThreads) {
    if constexpr (D % 4 == 0) {
        constexpr int D4 = D / 4;   // computed once, at compile time, per instantiation
        for (int idx = tid; idx < Rows * D4; idx += numThreads) {
            int r = idx / D4, c4 = idx % D4;
            int globalRow = tileStart + r;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (globalRow < numRows) {
                v = reinterpret_cast<const float4*>(&src[globalRow * D])[c4];
            }
            reinterpret_cast<float4*>(&dst[r][0])[c4] = v;
        }
    } else {
        for (int idx = tid; idx < Rows * D; idx += numThreads) {
            int r = idx / D, c = idx % D;
            int globalRow = tileStart + r;
            dst[r][c] = (globalRow < numRows) ? src[globalRow * D + c] : 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// The kernel itself. Block size is deliberately decoupled from Br/Bc/D (see
// the strided idx += numThreads loops below) -- one launch config (256
// threads, flat 1D block) works across every tile shape this file supports.
// ---------------------------------------------------------------------------

template <int Br, int Bc, int D>
__global__ void tiledScoreKernel(const float* A, const float* B, float* C, int M, int N) {
    // A: [M, D] row-major (foreshadows Q)
    // B: [N, D] row-major (foreshadows K, used transposed below)
    // C: [M, N] row-major (foreshadows S = Q K^T)
    static_assert(Br > 0 && Bc > 0 && D > 0, "tile dimensions must be positive");

    constexpr int sharedBytes = (Br * D + Bc * D) * sizeof(float);
    static_assert(sharedBytes <= 48 * 1024,
                  "Br/Bc/D exceed the 48KB default shared memory budget (Ch5 SS2.2) -- "
                  "shrink the tile, or see Ch9 for the cudaFuncSetAttribute opt-in");

    __shared__ float As[Br][D];
    __shared__ float Bs[Bc][D];

    int tileRow = blockIdx.y * Br;
    int tileCol = blockIdx.x * Bc;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    loadTileRowMajor<Br, D>(A, As, tileRow, M, tid, numThreads);
    loadTileRowMajor<Bc, D>(B, Bs, tileCol, N, tid, numThreads);
    __syncthreads();

    for (int idx = tid; idx < Br * Bc; idx += numThreads) {
        int r = idx / Bc, c = idx % Bc;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < D; ++k) {
            acc += As[r][k] * Bs[c][k];   // A row r . B row c => (A @ B^T)[r,c]
        }
        int globalRow = tileRow + r, globalCol = tileCol + c;
        if (globalRow < M && globalCol < N) {
            C[globalRow * N + globalCol] = acc;
        }
    }
}

// Explicit instantiation: forces the compiler to generate code for exactly
// these four (Br, Bc, D) combinations in this translation unit, rather than
// relying on the implicit instantiation the <<<...>>> calls below would
// trigger anyway. In a single-file build like this one the practical effect
// is small (see Exercise 5) -- the real payoff shows up in bigger projects
// that split kernel definitions and their instantiations across files, so
// each translation unit only compiles the variants it actually needs.
template __global__ void tiledScoreKernel<64, 64, 32>(const float*, const float*, float*, int, int);
template __global__ void tiledScoreKernel<64, 64, 64>(const float*, const float*, float*, int, int);
template __global__ void tiledScoreKernel<64, 64, 63>(const float*, const float*, float*, int, int);
template __global__ void tiledScoreKernel<32, 32, 128>(const float*, const float*, float*, int, int);

template <int Br, int Bc, int D>
void launchTiledScoreImpl(const float* dA, const float* dB, float* dC, int M, int N) {
    dim3 block(256);
    dim3 grid((N + Bc - 1) / Bc, (M + Br - 1) / Br);
    tiledScoreKernel<Br, Bc, D><<<grid, block>>>(dA, dB, dC, M, N);
    CUDA_CHECK(cudaGetLastError());
}

// The dispatch function: D is a runtime int here (unlike the kernel's D
// above), because that's what a caller actually has. Note Br/Bc aren't just
// D echoed three ways -- D=128 needs a smaller tile (32x32) to fit the
// shared-memory budget from SS2.3; D=32/63/64 all fit comfortably at 64x64.
void launchTiledScore(const float* dA, const float* dB, float* dC, int M, int N, int D) {
    switch (D) {
        case 32:  launchTiledScoreImpl<64, 64, 32>(dA, dB, dC, M, N); break;
        case 64:  launchTiledScoreImpl<64, 64, 64>(dA, dB, dC, M, N); break;
        case 63:  launchTiledScoreImpl<64, 64, 63>(dA, dB, dC, M, N); break;
        case 128: launchTiledScoreImpl<32, 32, 128>(dA, dB, dC, M, N); break;
        default:
            fprintf(stderr, "launchTiledScore: unsupported D=%d (supported: 32, 63, 64, 128)\n", D);
            exit(1);
    }
}
