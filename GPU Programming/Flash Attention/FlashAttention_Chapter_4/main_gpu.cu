// Chapter 4: test the three-kernel GPU attention against the CPU reference from Chapter 3,
// then time the three kernels.
//
// Build (from this folder), for the RTX 3090 (sm_86) or the T4 (sm_75):
//   nvcc -O2 -arch=sm_86 -I../ch03_code -o attn_gpu main_gpu.cu attention_gpu.cu ../ch03_code/tiled_attention.cpp
//   nvcc -O2 -arch=sm_75 -I../ch03_code -o attn_gpu main_gpu.cu attention_gpu.cu ../ch03_code/tiled_attention.cpp
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "attention_gpu.h"
#include "cuda_check.h"
#include "tiled_attention.h"        // attention_naive(): the CPU reference from Chapter 3

void lcg_fill(std::vector<float>& x, uint32_t seed, float scale) {     // same generator as Chapters 2 and 3
    uint32_t state = seed;
    for (size_t i = 0; i < x.size(); ++i) {
        state = 1664525u * state + 1013904223u;
        float u = static_cast<float>(state >> 8) / 16777216.0f;
        x[i] = (2.0f * u - 1.0f) * scale;
    }
}

// Largest |a[i] - b[i]|; returns NaN if any difference is NaN (see Chapter 3, section 3.7).
float max_abs_diff(const float* a, const float* b, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (!(diff <= worst)) worst = diff;
    }
    return worst;
}

int main() {
#ifndef CPU_EMULATION
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s | compute capability %d.%d | %d SMs | %.1f GiB | warp size %d | max threads/block %d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1073741824.0, prop.warpSize, prop.maxThreadsPerBlock);
#else
    printf("CPU EMULATION of the CUDA launch model (not a GPU)\n");
#endif

    // ---- correctness -------------------------------------------------------------------
    const int Ns[] = {1, 5, 37, 64, 100, 257};
    const int ds[] = {1, 8, 33, 64};
    const float tol = 1e-4f;   // GPU and CPU differ in fused multiply-add and in expf/logf rounding
    int configs = 0, failures = 0;
    float worst_o = 0.0f, worst_l = 0.0f;

    for (int N : Ns) {
        for (int d : ds) {
            std::vector<float> Q(N * d), K(N * d), V(N * d);
            lcg_fill(Q, 1u, 1.0f);
            lcg_fill(K, 2u, 1.0f);
            lcg_fill(V, 3u, 1.0f);
            for (bool causal : {false, true}) {
                std::vector<float> O_ref(N * d), L_ref(N), O(N * d), L(N);
                attention_naive(Q.data(), K.data(), V.data(), O_ref.data(), L_ref.data(), N, d, causal);
                attention_naive_gpu(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, causal);
                const float eo = max_abs_diff(O.data(), O_ref.data(), static_cast<size_t>(N) * d);
                const float el = max_abs_diff(L.data(), L_ref.data(), N);
                if (!(eo <= worst_o)) worst_o = eo;
                if (!(el <= worst_l)) worst_l = el;
                ++configs;
                if (!(eo < tol && el < tol)) {
                    if (++failures <= 5) {
                        printf("FAIL N=%d d=%d causal=%d  |dO|=%.3e |dL|=%.3e\n", N, d, causal, eo, el);
                    }
                }
            }
        }
    }

    // Large scores: without subtracting the row max, expf overflows float32 (Chapter 1, section 1.3).
    {
        const int N = 64, d = 16;
        std::vector<float> Q(N * d), K(N * d), V(N * d), O_ref(N * d), L_ref(N), O(N * d), L(N);
        lcg_fill(Q, 4u, 20.0f);
        lcg_fill(K, 5u, 20.0f);
        lcg_fill(V, 6u, 1.0f);
        attention_naive(Q.data(), K.data(), V.data(), O_ref.data(), L_ref.data(), N, d, false);
        attention_naive_gpu(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, false);
        const float eo = max_abs_diff(O.data(), O_ref.data(), static_cast<size_t>(N) * d);
        const float el = max_abs_diff(L.data(), L_ref.data(), N);
        ++configs;
        if (!(eo < 1e-3f && el < 1e-3f)) {
            ++failures;
            printf("FAIL large-scores case  |dO|=%.3e |dL|=%.3e\n", eo, el);
        }
    }
    printf("%d configurations, %d failures, worst |dO| = %.3e, worst |dL| = %.3e\n",
           configs, failures, worst_o, worst_l);

    // ---- timing ------------------------------------------------------------------------
#ifndef CPU_EMULATION
    if (failures == 0) {
        const int d = 64;
        printf("\n%6s %10s %10s %10s %10s   %s\n", "N", "scores ms", "softmax ms", "pv ms", "total ms", "S matrix");
        for (int N : {512, 1024, 2048, 4096}) {
            std::vector<float> Q(N * d), K(N * d), V(N * d), O(N * d), L(N);
            lcg_fill(Q, 1u, 1.0f);
            lcg_fill(K, 2u, 1.0f);
            lcg_fill(V, 3u, 1.0f);
            KernelTimes kt;
            attention_naive_gpu(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, false);   // warm-up
            float best_s = 1e30f, best_m = 1e30f, best_p = 1e30f;
            for (int rep = 0; rep < 5; ++rep) {
                attention_naive_gpu(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, false, &kt);
                best_s = std::min(best_s, kt.scores_ms);
                best_m = std::min(best_m, kt.softmax_ms);
                best_p = std::min(best_p, kt.pv_ms);
            }
            printf("%6d %10.3f %10.3f %10.3f %10.3f   %.0f MiB\n", N, best_s, best_m, best_p,
                   best_s + best_m + best_p, static_cast<double>(N) * N * 4 / 1048576.0);
        }
    }
#endif
    return failures == 0 ? 0 : 1;
}
