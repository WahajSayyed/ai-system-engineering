#include "tiled_attention.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
void lcg_fill(std::vector<float>& x, uint32_t seed, float scale) {
    uint32_t state = seed;
    for (size_t i = 0; i < x.size(); ++i) { state = 1664525u * state + 1013904223u;
        x[i] = (2.0f * (static_cast<float>(state >> 8) / 16777216.0f) - 1.0f) * scale; }
}
double secs(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
int main() {
    const size_t d = 64, N = 12288;    // S = 12288^2 * 4 B = 576 MiB
    std::vector<float> Q(N * d), K(N * d), V(N * d), O(N * d), L(N);
    lcg_fill(Q, 1u, 1.0f); lcg_fill(K, 2u, 1.0f); lcg_fill(V, 3u, 1.0f);
    auto t0 = std::chrono::steady_clock::now();
    attention_naive(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, false);
    double tn = secs(t0);
    printf("N=%zu naive: %.0f ms (S = %.0f MiB)\n", N, tn * 1e3, N * N * 4.0 / (1 << 20));
    for (size_t B : {64, 256}) {
        t0 = std::chrono::steady_clock::now();
        attention_tiled(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, B, B, false);
        double t = secs(t0);
        printf("N=%zu tiled B=%zu: %.0f ms (%.2fx vs naive)\n", N, B, t * 1e3, tn / t);
    }
}
