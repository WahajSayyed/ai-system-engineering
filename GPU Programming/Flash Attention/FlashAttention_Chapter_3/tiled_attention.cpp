// Chapter 3: attention on the CPU, naive and tiled (FlashAttention-2 loop order).
#include "tiled_attention.h"

#include <algorithm>   // std::min, std::max, std::fill
#include <cmath>       // expf, logf, std::sqrt, INFINITY
#include <vector>      // std::vector

// ---------------------------------------------------------------------------
// Reference implementation: O = softmax(Q K^T / sqrt(d)) V, one row at a time,
// with the whole N x N matrix in memory.
// ---------------------------------------------------------------------------
void attention_naive(const float* Q, const float* K, const float* V,
                     float* O, float* L,
                     size_t N, size_t d, bool causal) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> S(N * N);                        // <- the matrix FlashAttention avoids

    for (size_t i = 0; i < N; ++i) {
        const float* q = Q + i * d;                     // pointer to row i of Q
        for (size_t j = 0; j < N; ++j) {
            const float* k = K + j * d;                 // pointer to row j of K
            float dot = 0.0f;
            for (size_t x = 0; x < d; ++x) {
                dot += q[x] * k[x];
            }
            S[i * N + j] = (causal && j > i) ? -INFINITY : dot * scale;
        }
    }

    for (size_t i = 0; i < N; ++i) {
        float* s = &S[i * N];                           // pointer to row i of S
        float m = -INFINITY;
        for (size_t j = 0; j < N; ++j) m = std::max(m, s[j]);
        float l = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            s[j] = expf(s[j] - m);
            l += s[j];
        }
        float* o = O + i * d;
        std::fill(o, o + d, 0.0f);
        for (size_t j = 0; j < N; ++j) {
            const float* v = V + j * d;
            const float p = s[j] / l;
            for (size_t x = 0; x < d; ++x) o[x] += p * v[x];
        }
        L[i] = m + logf(l);
    }
}

// ---------------------------------------------------------------------------
// Tiled implementation. Q tiles in the outer loop, K/V tiles in the inner loop.
// The only buffers are one score tile, and per-row running statistics.
// ---------------------------------------------------------------------------
void attention_tiled(const float* Q, const float* K, const float* V,
                     float* O, float* L,
                     size_t N, size_t d, size_t Br, size_t Bc, bool causal) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    std::vector<float> S(Br * Bc);      // score tile; row r starts at S[r * Bc]
    std::vector<float> m(Br);           // running max, one per row of the Q tile
    std::vector<float> l(Br);           // running sum
    std::vector<float> acc(Br * d);     // running UNNORMALISED output; row r starts at acc[r * d]

    for (size_t r0 = 0; r0 < N; r0 += Br) {              // ---- loop over Q tiles
        const size_t br = std::min(Br, N - r0);          // rows in this tile (last one may be short)
        std::fill(m.begin(), m.begin() + br, -INFINITY);
        std::fill(l.begin(), l.begin() + br, 0.0f);
        std::fill(acc.begin(), acc.begin() + br * d, 0.0f);

        for (size_t c0 = 0; c0 < N; c0 += Bc) {          // ---- loop over K/V tiles
            const size_t bc = std::min(Bc, N - c0);      // columns in this tile
            if (causal && c0 > r0 + br - 1) {
                break;                                   // tile lies entirely right of the diagonal
            }

            // S = scale * Q_i K_j^T   (br x bc)
            for (size_t r = 0; r < br; ++r) {
                const float* q = Q + (r0 + r) * d;
                for (size_t c = 0; c < bc; ++c) {
                    const float* k = K + (c0 + c) * d;
                    float dot = 0.0f;
                    for (size_t x = 0; x < d; ++x) {
                        dot += q[x] * k[x];
                    }
                    const bool masked = causal && (c0 + c) > (r0 + r);
                    S[r * Bc + c] = masked ? -INFINITY : dot * scale;
                }
            }

            // Online-softmax update, one row of the tile at a time.
            for (size_t r = 0; r < br; ++r) {
                float* s = &S[r * Bc];                   // pointer to this row of the tile
                float tile_max = -INFINITY;
                for (size_t c = 0; c < bc; ++c) tile_max = std::max(tile_max, s[c]);

                const float m_new = std::max(m[r], tile_max);
                const float m_safe = (m_new == -INFINITY) ? 0.0f : m_new;   // Chapter 2 guard
                const float alpha = expf(m[r] - m_safe);                    // exp(m_old - m_new)

                float row_sum = 0.0f;
                for (size_t c = 0; c < bc; ++c) {
                    s[c] = expf(s[c] - m_safe);          // overwrite scores with unnormalised probabilities
                    row_sum += s[c];
                }
                l[r] = alpha * l[r] + row_sum;

                float* a = &acc[r * d];                  // pointer to this row of the accumulator
                for (size_t x = 0; x < d; ++x) a[x] *= alpha;        // shrink what we had
                for (size_t c = 0; c < bc; ++c) {                    // add P_tile * V_tile
                    const float* v = V + (c0 + c) * d;
                    const float p = s[c];
                    for (size_t x = 0; x < d; ++x) a[x] += p * v[x];
                }
                m[r] = m_new;
            }
        }

        // Normalise once, write the results for this Q tile.
        for (size_t r = 0; r < br; ++r) {
            const float* a = &acc[r * d];
            float* o = O + (r0 + r) * d;
            const float inv = (l[r] > 0.0f) ? 1.0f / l[r] : 0.0f;
            for (size_t x = 0; x < d; ++x) o[x] = a[x] * inv;
            L[r0 + r] = m[r] + logf(l[r]);
        }
    }
}
