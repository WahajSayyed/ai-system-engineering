#pragma once

// Chapter 7: warp-level reductions and a standalone warp-softmax kernel.
// ROW_LEN is fixed at 32 (one warp handles one row) inside warp_reduce.cu --
// Chapter 9 generalizes beyond a single warp.

void launchReductionComparison(const float* dIn, float* dOutXor, float* dOutDown, int numRows);
void launchWarpSoftmax(const float* dIn, float* dOut, int numRows);
