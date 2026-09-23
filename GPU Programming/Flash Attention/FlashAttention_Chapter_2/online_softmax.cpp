// Chapter 2: online softmax in C++ (scalar, CPU only).
// Build:  g++ -std=c++17 -O2 -Wall -Wextra -o online_softmax online_softmax.cpp
// Run:    ./online_softmax
#include <algorithm>   // std::max, std::min
#include <cmath>       // expf, INFINITY, fabsf
#include <cstdint>     // uint32_t
#include <cstdio>      // printf
#include <vector>      // std::vector

// ---------------------------------------------------------------------------
// Deterministic pseudo-random numbers. The Python checker (check_cpp.py)
// reproduces the exact same sequence, so both languages see identical inputs.
// ---------------------------------------------------------------------------
void lcg_fill(std::vector<float>& x, uint32_t seed, float scale) {
    uint32_t state = seed;
    for (size_t i = 0; i < x.size(); ++i) {
        state = 1664525u * state + 1013904223u;                    // wraps modulo 2^32
        float u = static_cast<float>(state >> 8) / 16777216.0f;    // 24-bit integer / 2^24: exact
        x[i] = (2.0f * u - 1.0f) * scale;
    }
}

// ---------------------------------------------------------------------------
// 1. Safe softmax, three passes.
// ---------------------------------------------------------------------------
std::vector<float> softmax_3pass(const std::vector<float>& x) {
    const size_t n = x.size();

    float m = -INFINITY;                       // pass 1: maximum
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, x[i]);
    }

    float d = 0.0f;                            // pass 2: normaliser
    for (size_t i = 0; i < n; ++i) {
        d += expf(x[i] - m);
    }

    std::vector<float> y(n);                   // pass 3: outputs
    for (size_t i = 0; i < n; ++i) {
        y[i] = expf(x[i] - m) / d;
    }
    return y;
}

// ---------------------------------------------------------------------------
// 2. Online normaliser: the running state is a (max, sum) pair.
// ---------------------------------------------------------------------------
struct MD {
    float m;   // running maximum
    float d;   // running sum of exp(value - m)
};

// Fold one more score into the running state.
MD push(MD s, float x) {
    float m_new = std::max(s.m, x);
    if (m_new == -INFINITY) {                  // everything so far is masked
        return s;
    }
    return MD{m_new, s.d * expf(s.m - m_new) + expf(x - m_new)};
}

MD online_stats(const std::vector<float>& x) {
    MD s{-INFINITY, 0.0f};
    for (float xi : x) {                       // range-based for: like `for xi in x`
        s = push(s, xi);
    }
    return s;
}

std::vector<float> softmax_online(const std::vector<float>& x) {
    MD s = online_stats(x);
    std::vector<float> y(x.size(), 0.0f);      // all zeros: also our answer for a fully masked row
    if (s.d == 0.0f) {
        return y;
    }
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = expf(x[i] - s.m) / s.d;
    }
    return y;
}

// ---------------------------------------------------------------------------
// 3. Merge two partial states (what makes tiling and parallel reduction work).
// ---------------------------------------------------------------------------
MD merge(MD a, MD b) {
    float m = std::max(a.m, b.m);
    if (m == -INFINITY) {
        return MD{-INFINITY, 0.0f};
    }
    return MD{m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

MD stats_blocked(const std::vector<float>& x, size_t block) {
    MD total{-INFINITY, 0.0f};
    for (size_t start = 0; start < x.size(); start += block) {
        size_t end = std::min(start + block, x.size());
        MD part{-INFINITY, 0.0f};
        for (size_t i = start; i < end; ++i) {
            part = push(part, x[i]);
        }
        total = merge(total, part);
    }
    return total;
}

// ---------------------------------------------------------------------------
// 4. One pass for sum_j softmax(s)_j * v_j  (v_j is a single number here;
//    vector-valued v_j is the same update on every component: Chapter 3).
// ---------------------------------------------------------------------------
float weighted_average_online(const std::vector<float>& s, const std::vector<float>& v) {
    float m = -INFINITY;
    float d = 0.0f;
    float o = 0.0f;                            // running UNNORMALISED output
    for (size_t j = 0; j < s.size(); ++j) {
        float m_new = std::max(m, s[j]);
        if (m_new == -INFINITY) {
            continue;
        }
        float scale = expf(m - m_new);         // shrink everything accumulated so far
        float w = expf(s[j] - m_new);          // weight of the new element
        d = d * scale + w;
        o = o * scale + w * v[j];
        m = m_new;
    }
    return d == 0.0f ? 0.0f : o / d;           // normalise once, at the end
}

float weighted_average_reference(const std::vector<float>& s, const std::vector<float>& v) {
    std::vector<float> p = softmax_3pass(s);
    float acc = 0.0f;
    for (size_t j = 0; j < s.size(); ++j) {
        acc += p[j] * v[j];
    }
    return acc;
}

// ---------------------------------------------------------------------------
int main() {
    const size_t n = 16;
    std::vector<float> x(n);
    lcg_fill(x, 12345u, 4.0f);

    std::vector<float> y3 = softmax_3pass(x);
    std::vector<float> y2 = softmax_online(x);

    float max_diff = 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, fabsf(y3[i] - y2[i]));
        sum += y2[i];
        printf("y3 %zu %.7f\n", i, y3[i]);
        printf("y2 %zu %.7f\n", i, y2[i]);
    }
    printf("max |3pass - online| = %.3e   sum(online) = %.7f\n", max_diff, sum);

    MD ref = online_stats(x);
    for (size_t block : {1, 3, 5, 16}) {
        MD s = stats_blocked(x, block);
        printf("blocked block=%2zu  m=%.7f  d=%.7f  |d - d_seq| = %.2e\n",
               block, s.m, s.d, fabsf(s.d - ref.d));
    }

    std::vector<float> v(n);
    lcg_fill(v, 777u, 2.0f);
    float a = weighted_average_online(x, v);
    float b = weighted_average_reference(x, v);
    printf("weighted average: one-pass = %.7f  reference = %.7f\n", a, b);

    std::vector<float> masked = x;
    masked[0] = masked[1] = masked[2] = -INFINITY;     // a row that starts masked
    float c = weighted_average_online(masked, v);
    std::vector<float> tail_s(masked.begin() + 3, masked.end());
    std::vector<float> tail_v(v.begin() + 3, v.end());
    float e = weighted_average_reference(tail_s, tail_v);
    printf("masked start:     one-pass = %.7f  reference = %.7f\n", c, e);

    // A longer row: now the two algorithms round differently.
    std::vector<float> big(4096);
    lcg_fill(big, 1u, 8.0f);
    std::vector<float> b3 = softmax_3pass(big);
    std::vector<float> b2 = softmax_online(big);
    float big_diff = 0.0f;
    float big_sum3 = 0.0f;
    float big_sum2 = 0.0f;
    for (size_t i = 0; i < big.size(); ++i) {
        big_diff = std::max(big_diff, fabsf(b3[i] - b2[i]));
        big_sum3 += b3[i];
        big_sum2 += b2[i];
    }
    printf("n=4096: max |3pass - online| = %.3e   sum(3pass) = %.6f   sum(online) = %.6f\n",
           big_diff, big_sum3, big_sum2);
    return 0;
}
