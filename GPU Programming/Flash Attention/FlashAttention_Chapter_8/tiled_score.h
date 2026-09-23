#pragma once

// Chapter 8: A @ B^T (foreshadows S = Q K^T), templated on tile shape.
// Br, Bc, D are compile-time template parameters inside tiled_score.cu --
// this header only exposes the runtime dispatch entry point, since the
// caller's D is a runtime value (SS2.5), not something it can template on.

void launchTiledScore(const float* dA, const float* dB, float* dC,
                       int M, int N, int D);
