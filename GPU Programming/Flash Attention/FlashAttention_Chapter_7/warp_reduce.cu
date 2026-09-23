// warp_reduce.cu
// Chapter 7: warp-level reductions via shuffle instructions, and a
// standalone warp-wide softmax built entirely out of them -- no shared
// memory, no __syncthreads(), everything stays in registers within one warp.
// Library only, no main() -- see test_warp_reduce.cpp for the driver.
//
// Compile as part of the test binary:
//   nvcc -O3 -arch=sm_75 warp_reduce.cu test_warp_reduce.cpp -o warp_reduce_test   (Tesla T4)
//   nvcc -O3 -arch=sm_86 warp_reduce.cu test_warp_reduce.cpp -o warp_reduce_test   (RTX 3090)

#include "warp_reduce.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define FULL_MASK 0xffffffff
#define WARPS_PER_BLOCK 8
#define ROW_LEN 32   // one warp handles exactly one row; Chapter 9 generalizes this

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Warp-level reductions. Every lane calls these with its own `val`; after
// they return, `val` has been combined with every other lane's -- entirely
// via register-to-register shuffles, no shared memory involved.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warpReduceMaxXor(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(FULL_MASK, val, offset);
        val = fmaxf(val, other);
    }
    return val;   // every lane holds the row max
}

__device__ __forceinline__ float warpReduceSumXor(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
    return val;   // every lane holds the row sum
}

__device__ __forceinline__ float warpReduceSumDown(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;   // ONLY lane 0 holds the row sum -- other lanes are not the answer
}

// ---------------------------------------------------------------------------
// Kernel 1: the XOR-vs-DOWN difference, made directly checkable.
// See test_warp_reduce.cpp for what each output is actually checked against.
// ---------------------------------------------------------------------------

__global__ void reductionComparisonKernel(const float* in, float* outXor,
                                           float* outDown, int numRows) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    float x = in[row * ROW_LEN + lane];

    outXor[row * ROW_LEN + lane] = warpReduceSumXor(x);
    outDown[row * ROW_LEN + lane] = warpReduceSumDown(x);
}

// ---------------------------------------------------------------------------
// Kernel 2: softmax across one warp's 32 elements -- Chapter 2's
// max / exp / sum / divide, entirely in registers.
// ---------------------------------------------------------------------------

__global__ void warpSoftmaxKernel(const float* in, float* out, int numRows) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;

    float x = in[row * ROW_LEN + lane];

    float m = warpReduceMaxXor(x);   // every lane: the row max
    float p = __expf(x - m);          // unnormalized softmax numerator
    float l = warpReduceSumXor(p);   // every lane: the row sum

    out[row * ROW_LEN + lane] = p / l;
}

void launchReductionComparison(const float* dIn, float* dOutXor, float* dOutDown, int numRows) {
    dim3 block(ROW_LEN, WARPS_PER_BLOCK);
    dim3 grid(numRows / WARPS_PER_BLOCK);
    reductionComparisonKernel<<<grid, block>>>(dIn, dOutXor, dOutDown, numRows);
    CUDA_CHECK(cudaGetLastError());
}

void launchWarpSoftmax(const float* dIn, float* dOut, int numRows) {
    dim3 block(ROW_LEN, WARPS_PER_BLOCK);
    dim3 grid(numRows / WARPS_PER_BLOCK);
    warpSoftmaxKernel<<<grid, block>>>(dIn, dOut, numRows);
    CUDA_CHECK(cudaGetLastError());
}
