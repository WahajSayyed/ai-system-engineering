// Chapter 3: declarations for the two attention implementations.
// All matrices are row-major float arrays: element (row, col) of an [R x C] matrix
// lives at index row * C + col.
#pragma once          // include this header at most once per source file
#include <cstddef>    // size_t

// Reference: builds the full N x N score matrix.
//   Q, K, V : [N x d] inputs (read-only)
//   O       : [N x d] output      L : [N] row logsumexp (m + log(sum))
void attention_naive(const float* Q, const float* K, const float* V,
                     float* O, float* L,
                     size_t N, size_t d, bool causal);

// Tiled forward pass: never allocates anything of size N x N.
//   Br, Bc  : tile sizes (rows of Q per tile, rows of K/V per tile)
void attention_tiled(const float* Q, const float* K, const float* V,
                     float* O, float* L,
                     size_t N, size_t d, size_t Br, size_t Bc, bool causal);
