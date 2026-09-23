// flash_skeleton.cu
// Chapter 9: kernel skeleton -- grid layout, [B,H,N,D] stride arithmetic,
// and the shared memory budget/opt-in. The kernel establishes where every
// block's data lives (global and shared) and records it for verification;
// Chapter 10 fills in the actual tile-load loops.
//
// Compile as part of the test binary:
//   nvcc -O3 -arch=sm_75 flash_skeleton.cu test_flash_skeleton.cpp -o flash_skeleton_test   (Tesla T4)
//   nvcc -O3 -arch=sm_86 flash_skeleton.cu test_flash_skeleton.cpp -o flash_skeleton_test   (RTX 3090)

#include "flash_skeleton.h"
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

template <int Br, int Bc, int D>
__global__ void flashSkeletonKernel(const float* Q, const float* K, const float* V,
                                     float* O, int B, int H, int N,
                                     BlockInfo* debugOut) {
    // ---- shared-memory layout: WHERE each piece will live (Chapter 10 fills them in) ----
    extern __shared__ float smem[];
    float* Qs = smem;                 // [Br, D]
    float* Ks = Qs + Br * D;          // [Bc, D]
    float* Vs = Ks + Bc * D;          // [Bc, D]
    float* m  = Vs + Bc * D;          // [Br]  running row max (Ch2, used from Ch12)
    float* l  = m + Br;               // [Br]  running row sum (Ch2, used from Ch12)
    (void)Qs; (void)Ks; (void)Vs; (void)m; (void)l;   // unused until Ch10-12

    // ---- grid layout: which (batch, head, Q-tile) this block owns ----
    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int qTile = blockIdx.x;
    int qStart = qTile * Br;

    // ---- [B, H, N, D] stride arithmetic: this block's (batch, head) slice ----
    long long bhOffset = ((long long)batch * H + head) * (long long)N * D;
    const float* Qbh = Q + bhOffset;   // Qbh[n*D+d] == Q[batch,head,n,d]
    const float* Kbh = K + bhOffset;
    const float* Vbh = V + bhOffset;
    float* Obh = O + bhOffset;
    (void)Qbh; (void)Kbh; (void)Vbh; (void)Obh;   // used starting Chapter 10

    // Chapters 10-13 load Qbh/Kbh/Vbh into Qs/Ks/Vs, compute scores, run the
    // online softmax through m/l, and accumulate into Obh. This chapter
    // stops once every block can correctly answer "which data is mine, and
    // where does my shared memory live" -- recorded here for verification.
    if (threadIdx.x == 0) {
        int blockLinear = (blockIdx.z * H + blockIdx.y) * gridDim.x + blockIdx.x;
        debugOut[blockLinear] = BlockInfo{batch, head, qStart, bhOffset};
    }
}

template <int Br, int Bc, int D>
void launchFlashSkeletonImpl(const float* Q, const float* K, const float* V, float* O,
                              int B, int H, int N, BlockInfo* debugOut) {
    constexpr size_t sharedBytes = (size_t)(Br * D + 2 * Bc * D + 2 * Br) * sizeof(float);

    // Static shared memory has no opt-in (SS2.4) -- this call is what lets a
    // DYNAMIC request above the 48KB default actually succeed. Comment it
    // out to see the launch below fail instead (Exercise 1).
    CUDA_CHECK(cudaFuncSetAttribute(flashSkeletonKernel<Br, Bc, D>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)sharedBytes));

    int numQTiles = (N + Br - 1) / Br;
    dim3 grid(numQTiles, H, B);
    dim3 block(256);

    flashSkeletonKernel<Br, Bc, D><<<grid, block, sharedBytes>>>(Q, K, V, O, B, H, N, debugOut);
    CUDA_CHECK(cudaGetLastError());
}

void launchFlashSkeleton(const float* Q, const float* K, const float* V, float* O,
                         int B, int H, int N, int D_runtime, BlockInfo* debugOut) {
    switch (D_runtime) {
        case 64: launchFlashSkeletonImpl<64, 64, 64>(Q, K, V, O, B, H, N, debugOut); break;
        default:
            fprintf(stderr, "launchFlashSkeleton: unsupported D=%d (supported: 64)\n", D_runtime);
            exit(1);
    }
}
