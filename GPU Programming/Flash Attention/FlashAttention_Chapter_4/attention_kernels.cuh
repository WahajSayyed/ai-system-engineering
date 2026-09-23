// Chapter 4: naive attention as three GPU kernels (device code only).
//
//   S = scale * Q K^T   (masked)     [N x N]     scores_kernel
//   P = softmax(S) per row           [N x N]     softmax_rows_kernel   (in place; also writes L)
//   O = P V                          [N x d]     pv_kernel
//
// All matrices are row-major float arrays: element (row, col) of an [R x C] matrix is at row * C + col.
#pragma once
#include <cmath>   // INFINITY, expf, logf, fmaxf

// One thread per element of S. Launched on a 2-D grid of 2-D blocks.
__global__ void scores_kernel(const float* Q, const float* K, float* S,
                              int N, int d, float scale, bool causal) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   // column of S: which key
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // row of S:    which query
    if (i >= N || j >= N) return;                          // the grid is rounded UP: extra threads exit

    if (causal && j > i) {                                 // query i may not look at key j > i
        S[static_cast<size_t>(i) * N + j] = -INFINITY;
        return;
    }
    const float* q = Q + static_cast<size_t>(i) * d;       // row i of Q
    const float* k = K + static_cast<size_t>(j) * d;       // row j of K
    float dot = 0.0f;
    for (int x = 0; x < d; ++x) {
        dot += q[x] * k[x];
    }
    S[static_cast<size_t>(i) * N + j] = dot * scale;
}

// One thread per ROW of S. Safe softmax in three sequential passes over the row, in place.
// Also writes the row logsumexp L[i] = max + log(sum), as in Chapter 3.
__global__ void softmax_rows_kernel(float* S, float* L, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // which row
    if (i >= N) return;

    float* s = S + static_cast<size_t>(i) * N;             // row i of S
    float m = -INFINITY;
    for (int j = 0; j < N; ++j) m = fmaxf(m, s[j]);        // pass 1: max
    float l = 0.0f;
    for (int j = 0; j < N; ++j) {                          // pass 2: exp and sum
        s[j] = expf(s[j] - m);
        l += s[j];
    }
    const float inv = 1.0f / l;
    for (int j = 0; j < N; ++j) s[j] *= inv;               // pass 3: normalise
    L[i] = m + logf(l);
}

// One thread per element of O. O[i][c] = sum_j P[i][j] * V[j][c].
__global__ void pv_kernel(const float* P, const float* V, float* O, int N, int d) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;   // output column, 0 .. d-1
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // output row (query)
    if (i >= N || c >= d) return;

    const float* p = P + static_cast<size_t>(i) * N;       // row i of P
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        acc += p[j] * V[static_cast<size_t>(j) * d + c];
    }
    O[static_cast<size_t>(i) * d + c] = acc;
}
