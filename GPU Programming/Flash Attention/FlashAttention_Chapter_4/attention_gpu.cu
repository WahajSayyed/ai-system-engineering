// Chapter 4: host code for the three-kernel GPU attention.
#include "attention_gpu.h"

#include <cmath>
#include <cuda_runtime.h>

#include "attention_kernels.cuh"
#include "cuda_check.h"

// Integer ceiling division: the number of size-b blocks needed to cover a items.
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static void timer_begin(cudaEvent_t t0) { CUDA_CHECK(cudaEventRecord(t0)); }

static float timer_end(cudaEvent_t t0, cudaEvent_t t1) {
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));          // wait until the GPU has reached t1
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    return ms;
}

void attention_naive_gpu(const float* hQ, const float* hK, const float* hV,
                         float* hO, float* hL, int N, int d, bool causal,
                         KernelTimes* times) {
    const size_t bytes_nd = static_cast<size_t>(N) * d * sizeof(float);   // Q, K, V, O
    const size_t bytes_nn = static_cast<size_t>(N) * N * sizeof(float);   // S (the matrix we dislike)
    const size_t bytes_n  = static_cast<size_t>(N) * sizeof(float);       // L

    // ---- 1. allocate DEVICE memory ---------------------------------------------------
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr, *dS = nullptr, *dL = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, bytes_nd));
    CUDA_CHECK(cudaMalloc(&dK, bytes_nd));
    CUDA_CHECK(cudaMalloc(&dV, bytes_nd));
    CUDA_CHECK(cudaMalloc(&dO, bytes_nd));
    CUDA_CHECK(cudaMalloc(&dS, bytes_nn));
    CUDA_CHECK(cudaMalloc(&dL, bytes_n));

    // ---- 2. copy the inputs host -> device -------------------------------------------
    CUDA_CHECK(cudaMemcpy(dQ, hQ, bytes_nd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, bytes_nd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, bytes_nd, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    KernelTimes kt;

    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const dim3 block2d(16, 16);                                    // 256 threads per block

    // ---- 3a. kernel 1: S = scale * Q K^T. One thread per element of S (N x N) --------
    const dim3 grid_s(ceil_div(N, block2d.x), ceil_div(N, block2d.y));
    timer_begin(t0);
    scores_kernel<<<grid_s, block2d>>>(dQ, dK, dS, N, d, scale, causal);
    CUDA_CHECK_LAUNCH();
    kt.scores_ms = timer_end(t0, t1);

    // ---- 3b. kernel 2: row softmax. One thread per row ---------------------------------
    const int threads_1d = 128;
    timer_begin(t0);
    softmax_rows_kernel<<<ceil_div(N, threads_1d), threads_1d>>>(dS, dL, N);
    CUDA_CHECK_LAUNCH();
    kt.softmax_ms = timer_end(t0, t1);

    // ---- 3c. kernel 3: O = P V. One thread per element of O (N x d) ---------------------
    const dim3 grid_o(ceil_div(d, block2d.x), ceil_div(N, block2d.y));
    timer_begin(t0);
    pv_kernel<<<grid_o, block2d>>>(dS, dV, dO, N, d);
    CUDA_CHECK_LAUNCH();
    kt.pv_ms = timer_end(t0, t1);

    CUDA_CHECK(cudaDeviceSynchronize());                           // surface any asynchronous error

    // ---- 4. copy the results device -> host --------------------------------------------
    CUDA_CHECK(cudaMemcpy(hO, dO, bytes_nd, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hL, dL, bytes_n, cudaMemcpyDeviceToHost));

    // ---- 5. clean up ------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO));
    CUDA_CHECK(cudaFree(dS));
    CUDA_CHECK(cudaFree(dL));

    if (times != nullptr) *times = kt;
}
