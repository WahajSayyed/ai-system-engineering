// Chapter 3: test the tiled implementation against the naive one, over many shapes.
#include "tiled_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

void lcg_fill(std::vector<float>& x, uint32_t seed, float scale) {     // same generator as Chapter 2
    uint32_t state = seed;
    for (size_t i = 0; i < x.size(); ++i) {
        state = 1664525u * state + 1013904223u;
        float u = static_cast<float>(state >> 8) / 16777216.0f;
        x[i] = (2.0f * u - 1.0f) * scale;
    }
}

// Largest |a[i] - b[i]|. Unlike std::max, this does NOT lose a NaN: comparisons with NaN are
// false, so `!(diff <= worst)` is true both when diff is larger and when diff is NaN.
float max_abs_diff(const float* a, const float* b, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (!(diff <= worst)) worst = diff;
    }
    return worst;
}

int main() {
    const size_t Ns[] = {1, 5, 37, 64, 100, 257};
    const size_t ds[] = {1, 8, 33};
    const size_t tiles[][2] = {{1, 1}, {4, 7}, {16, 16}, {64, 32}, {128, 128}, {1000, 1000}};

    int configs = 0, failures = 0;
    float worst_o = 0.0f, worst_l = 0.0f;

    for (size_t N : Ns) {
        for (size_t d : ds) {
            std::vector<float> Q(N * d), K(N * d), V(N * d);
            lcg_fill(Q, 1u, 1.0f);
            lcg_fill(K, 2u, 1.0f);
            lcg_fill(V, 3u, 1.0f);
            for (bool causal : {false, true}) {
                std::vector<float> O_ref(N * d), L_ref(N);
                attention_naive(Q.data(), K.data(), V.data(), O_ref.data(), L_ref.data(), N, d, causal);
                for (const auto& t : tiles) {
                    std::vector<float> O(N * d), L(N);
                    attention_tiled(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, t[0], t[1], causal);
                    const float eo = max_abs_diff(O.data(), O_ref.data(), N * d);
                    const float el = max_abs_diff(L.data(), L_ref.data(), N);
                    if (!(eo <= worst_o)) worst_o = eo;            // NaN-safe running maximum
                    if (!(el <= worst_l)) worst_l = el;
                    ++configs;
                    if (!(eo < 1e-5f && el < 1e-5f)) {          // '!(a && b)' also catches NaN
                        if (++failures <= 5) {
                            printf("FAIL N=%zu d=%zu Br=%zu Bc=%zu causal=%d  |dO|=%.3e |dL|=%.3e\n",
                                   N, d, t[0], t[1], causal, eo, el);
                        }
                    }
                }
            }
        }
    }
    printf("%d configurations, %d failures, worst |dO| = %.3e, worst |dL| = %.3e\n",
           configs, failures, worst_o, worst_l);

    // One case dumped in full, so check_cpp.py can compare against float64 Python.
    const size_t N = 37, d = 8;
    std::vector<float> Q(N * d), K(N * d), V(N * d), O(N * d), L(N);
    lcg_fill(Q, 1u, 1.0f);
    lcg_fill(K, 2u, 1.0f);
    lcg_fill(V, 3u, 1.0f);
    attention_tiled(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, 8, 16, true);
    for (size_t i = 0; i < N; ++i) {
        for (size_t x = 0; x < d; ++x) printf("O %zu %zu %.9g\n", i, x, O[i * d + x]);
        printf("L %zu %.9g\n", i, L[i]);
    }
    return failures == 0 ? 0 : 1;
}
