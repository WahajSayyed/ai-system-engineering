// test_warp_reduce.cpp
// Chapter 7: host-side test driver.
//   1. reductionComparisonKernel: checks that the XOR reduction is correct
//      in EVERY lane, and that the DOWN reduction is correct only in lane 0
//      -- both are working as designed, not as a bug in either one.
//   2. warpSoftmaxKernel: checked against a straightforward CPU softmax.

#include "warp_reduce.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

#define ROW_LEN 32

static bool testReductionComparison() {
    const int numRows = 1024;
    std::vector<float> hIn(numRows * ROW_LEN), hOutXor(numRows * ROW_LEN), hOutDown(numRows * ROW_LEN);
    srand(0);
    for (auto& x : hIn) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    float *dIn, *dOutXor, *dOutDown;
    size_t bytes = hIn.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dIn, bytes));
    CUDA_CHECK(cudaMalloc(&dOutXor, bytes));
    CUDA_CHECK(cudaMalloc(&dOutDown, bytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    launchReductionComparison(dIn, dOutXor, dOutDown, numRows);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hOutXor.data(), dOutXor, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hOutDown.data(), dOutDown, bytes, cudaMemcpyDeviceToHost));

    bool xorOk = true, downLane0Ok = true;
    int downOtherLanesWrong = 0;
    for (int row = 0; row < numRows; ++row) {
        float rowSum = 0.0f;
        for (int lane = 0; lane < ROW_LEN; ++lane) rowSum += hIn[row * ROW_LEN + lane];

        for (int lane = 0; lane < ROW_LEN; ++lane) {
            if (fabsf(hOutXor[row * ROW_LEN + lane] - rowSum) > 1e-3f) xorOk = false;
        }
        if (fabsf(hOutDown[row * ROW_LEN + 0] - rowSum) > 1e-3f) downLane0Ok = false;
        for (int lane = 1; lane < ROW_LEN; ++lane) {
            if (fabsf(hOutDown[row * ROW_LEN + lane] - rowSum) > 1e-3f) downOtherLanesWrong++;
        }
    }

    printf("XOR reduction correct in every lane: %s\n", xorOk ? "PASS" : "FAIL");
    printf("DOWN reduction correct in lane 0: %s\n", downLane0Ok ? "PASS" : "FAIL");
    printf("DOWN reduction: %d / %d non-lane-0 entries differ from the row sum "
           "(expected -- only lane 0 is guaranteed correct)\n",
           downOtherLanesWrong, numRows * (ROW_LEN - 1));

    cudaFree(dIn); cudaFree(dOutXor); cudaFree(dOutDown);
    return xorOk && downLane0Ok;
}

static void cpuSoftmaxRow(const float* x, float* out, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i) m = fmaxf(m, x[i]);
    float l = 0.0f;
    for (int i = 0; i < n; ++i) { out[i] = expf(x[i] - m); l += out[i]; }
    for (int i = 0; i < n; ++i) out[i] /= l;
}

static bool testWarpSoftmax() {
    const int numRows = 1024;
    std::vector<float> hIn(numRows * ROW_LEN), hOut(numRows * ROW_LEN), hRef(numRows * ROW_LEN);
    for (auto& x : hIn) x = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;

    float *dIn, *dOut;
    size_t bytes = hIn.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dIn, bytes));
    CUDA_CHECK(cudaMalloc(&dOut, bytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    launchWarpSoftmax(dIn, dOut, numRows);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost));

    for (int row = 0; row < numRows; ++row) {
        cpuSoftmaxRow(&hIn[row * ROW_LEN], &hRef[row * ROW_LEN], ROW_LEN);
    }

    float maxErr = 0.0f;
    for (size_t i = 0; i < hOut.size(); ++i) maxErr = fmaxf(maxErr, fabsf(hOut[i] - hRef[i]));
    bool ok = maxErr < 1e-3f;
    printf("warp softmax max abs error vs CPU reference: %e -> %s\n", maxErr, ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

int main() {
    bool ok = true;
    ok &= testReductionComparison();
    ok &= testWarpSoftmax();
    return ok ? 0 : 1;
}
