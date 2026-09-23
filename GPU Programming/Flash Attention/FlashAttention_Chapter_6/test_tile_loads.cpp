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
