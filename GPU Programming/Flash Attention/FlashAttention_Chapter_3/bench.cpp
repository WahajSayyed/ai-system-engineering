// Chapter 3 (optional): time the naive and tiled CPU implementations, single thread.
// Build: g++ -std=c++17 -O2 -c tiled_attention.cpp && g++ -std=c++17 -O2 -o bench bench.cpp tiled_attention.o
#include "tiled_attention.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

void lcg_fill(std::vector<float>& x, uint32_t seed, float scale) {
    uint32_t state = seed;
    for (size_t i = 0; i < x.size(); ++i) {
        state = 1664525u * state + 1013904223u;
        x[i] = (2.0f * (static_cast<float>(state >> 8) / 16777216.0f) - 1.0f) * scale;
    }
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    const size_t d = 64;
    for (size_t N : {1024, 2048, 4096}) {
        std::vector<float> Q(N * d), K(N * d), V(N * d), O(N * d), L(N);
        lcg_fill(Q, 1u, 1.0f); lcg_fill(K, 2u, 1.0f); lcg_fill(V, 3u, 1.0f);

        double best_naive = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            attention_naive(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, false);
            best_naive = std::min(best_naive, seconds_since(t0));
        }
        printf("N=%zu d=%zu  naive              : %8.1f ms   (S matrix = %.0f MiB)\n",
               N, d, best_naive * 1e3, N * N * 4.0 / (1 << 20));

        for (size_t B : {16, 32, 64, 128, 256}) {
            double best = 1e30;
            for (int rep = 0; rep < 3; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                attention_tiled(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, B, B, false);
                best = std::min(best, seconds_since(t0));
            }
            printf("N=%zu d=%zu  tiled Br=Bc=%-4zu   : %8.1f ms   (%.2fx vs naive)\n", N, d, B, best * 1e3, best_naive / best);
        }
    }
    return 0;
}
