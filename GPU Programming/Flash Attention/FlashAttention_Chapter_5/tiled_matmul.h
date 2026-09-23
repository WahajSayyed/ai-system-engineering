#pragma once

// Tiled matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// Kernel + host launcher live in tiled_matmul.cu; declared here so host code
// (test_tiled_matmul.cpp) doesn't need to see CUDA kernel syntax to call it.

void launchTiledMatMul(const float* dA, const float* dB, float* dC,
                        int M, int K, int N);
