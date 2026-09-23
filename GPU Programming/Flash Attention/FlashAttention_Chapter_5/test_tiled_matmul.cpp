// test_tiled_matmul.cpp
// Chapter 5: host-side test driver for the tiled matmul kernel.
// Builds random inputs, runs the GPU kernel via tiled_matmul.h, checks the
// result against a plain triple-loop CPU reference.
//
// Compile together with tiled_matmul.cu -- see the header comment there for
// the nvcc command line.

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

// Plain triple-loop CPU reference. This chapter's kernel is a generic
// A @ B, not attention, so there's no Chapter 1 oracle to check against yet
// -- see the chapter notes for when that changes (Chapter 11, S = Q K^T).
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
