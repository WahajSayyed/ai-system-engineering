#pragma once

// Chapter 6: coalescing and vectorized loads. Kernels + host launchers live
// in tile_loads.cu; declared here so test_tile_loads.cpp doesn't need to see
// CUDA kernel syntax to call them.

// Part 1: coalesced vs strided access (SS2.1).
void launchCopyCoalesced(const float* dIn, float* dOut, int n);
void launchCopyStrided(const float* dIn, float* dOut, int n, int stride);

// Part 2: scalar vs float4-vectorized load of a [N, D] tile, D == 64 (SS2.2-2.3).
// D and ROWS_PER_BLOCK are fixed in tile_loads.cu; N must be a multiple of
// ROWS_PER_BLOCK (8) for this chapter's kernels -- boundary handling for
// arbitrary N returns properly in Chapter 10.
void launchScalarTileLoad(const float* dIn, float* dOut, int N);
void launchVectorizedTileLoad(const float* dIn, float* dOut, int N);
