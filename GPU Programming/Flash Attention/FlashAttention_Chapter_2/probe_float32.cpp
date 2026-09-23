#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
struct MD { float m; float d; };
MD push(MD s, float x) { float mn = std::max(s.m, x); if (mn == -INFINITY) return s; return MD{mn, s.d * expf(s.m - mn) + expf(x - mn)}; }
int main() {
    for (uint32_t seed : {1u, 2u, 3u, 2024u, 99u}) {
        for (size_t n : {16u, 256u, 4096u, 65536u}) {
            std::vector<float> x(n);
            uint32_t st = seed;
            for (size_t i = 0; i < n; ++i) { st = 1664525u * st + 1013904223u; x[i] = (2.0f * (static_cast<float>(st >> 8) / 16777216.0f) - 1.0f) * 8.0f; }
            float m = -INFINITY; for (float v : x) m = std::max(m, v);
            float d3 = 0; for (float v : x) d3 += expf(v - m);
            MD s{-INFINITY, 0}; int changes = 0;
            for (float v : x) { float before = s.m; s = push(s, v); if (s.m != before) ++changes; }
            printf("seed=%u n=%6zu changes=%2d d3=%.9g d_online=%.9g rel=%.2e\n", seed, n, changes, d3, s.d, fabsf(d3 - s.d) / d3);
        }
    }
}
