// test_flash_skeleton.cpp
// Chapter 9: host-side test driver. Independently recomputes the expected
// (batch, head, qStart, bhOffset) for every block from the same [B,H,N,D]
// shape, and checks it against what each block actually wrote -- this is a
// direct test of SS2.1/SS2.2, not of any attention math (there isn't any yet).

#include "flash_skeleton.h"

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

int main() {
    const int B = 2, H = 4, N = 200, D = 64, Br = 64;
    const int numQTiles = (N + Br - 1) / Br;   // 4, with a partial last tile (8 valid rows)
    const int numBlocks = B * H * numQTiles;

    // Q/K/V/O contents don't matter for this chapter -- only shapes and offsets do.
    std::vector<float> hQKV(B * H * N * D, 0.0f);
    std::vector<BlockInfo> hDebug(numBlocks);

    float *dQ, *dK, *dV, *dO;
    BlockInfo* dDebug;
    size_t tensorBytes = hQKV.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dQ, tensorBytes));
    CUDA_CHECK(cudaMalloc(&dK, tensorBytes));
    CUDA_CHECK(cudaMalloc(&dV, tensorBytes));
    CUDA_CHECK(cudaMalloc(&dO, tensorBytes));
    CUDA_CHECK(cudaMalloc(&dDebug, numBlocks * sizeof(BlockInfo)));
    CUDA_CHECK(cudaMemcpy(dQ, hQKV.data(), tensorBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hQKV.data(), tensorBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hQKV.data(), tensorBytes, cudaMemcpyHostToDevice));

    launchFlashSkeleton(dQ, dK, dV, dO, B, H, N, D, dDebug);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hDebug.data(), dDebug, numBlocks * sizeof(BlockInfo), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int batch = 0; batch < B; ++batch) {
        for (int head = 0; head < H; ++head) {
            for (int qTile = 0; qTile < numQTiles; ++qTile) {
                int blockLinear = (batch * H + head) * numQTiles + qTile;
                BlockInfo got = hDebug[blockLinear];

                int expectedQStart = qTile * Br;
                long long expectedBhOffset = ((long long)batch * H + head) * (long long)N * D;

                if (got.batch != batch || got.head != head ||
                    got.qStart != expectedQStart || got.bhOffset != expectedBhOffset) {
                    ok = false;
                    printf("MISMATCH block(batch=%d,head=%d,qTile=%d): "
                           "got={batch=%d,head=%d,qStart=%d,bhOffset=%lld} "
                           "expected={batch=%d,head=%d,qStart=%d,bhOffset=%lld}\n",
                           batch, head, qTile, got.batch, got.head, got.qStart, got.bhOffset,
                           batch, head, expectedQStart, expectedBhOffset);
                }
            }
        }
    }

    printf("grid layout + [B,H,N,D] stride arithmetic, %d blocks (B=%d,H=%d,N=%d,numQTiles=%d): %s\n",
           numBlocks, B, H, N, numQTiles, ok ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dDebug);
    return ok ? 0 : 1;
}
