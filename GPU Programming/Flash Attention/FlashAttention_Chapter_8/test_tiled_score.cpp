// test_tiled_score.cpp
// Chapter 8: host-side test driver. Exercises all four dispatch cases --
// D=32, 64, 128 through the float4 branch of loadTileRowMajor's if
// constexpr, and D=63 through the scalar fallback branch -- against a
// plain triple-loop CPU reference for A @ B^T.

#include "tiled_score.h"

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

static void cpuMatMulT(const std::vector<float>& A, const std::vector<float>& B,
                        std::vector<float>& C, int M, int N, int D) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += A[i * D + k] * B[j * D + k];
            C[i * N + j] = acc;
        }
}

static bool testOneConfig(int D, int M, int N) {
    std::vector<float> hA(M * D), hB(N * D), hC(M * N), hRef(M * N);
    for (auto& x : hA) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (auto& x : hB) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    launchTiledScore(dA, dB, dC, M, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cpuMatMulT(hA, hB, hRef, M, N, D);

    float maxErr = 0.0f;
    for (size_t i = 0; i < hC.size(); ++i) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    bool ok = maxErr < 1e-2f;   // fp32 accumulation over up to D=128 terms
    printf("D=%3d (M=%d, N=%d): max abs error = %e -> %s\n", D, M, N, maxErr, ok ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok;
}

int main() {
    bool ok = true;
    ok &= testOneConfig(32, 200, 150);
    ok &= testOneConfig(64, 200, 150);
    ok &= testOneConfig(63, 200, 150);   // exercises the scalar (non-float4) branch
    ok &= testOneConfig(128, 200, 150);
    return ok ? 0 : 1;
}
