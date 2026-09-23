// Chapter 4: error-checking macros for the CUDA runtime API.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap EVERY CUDA runtime call:  CUDA_CHECK(cudaMalloc(&p, bytes));
// If the call fails, print where and why, then stop the program.
#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        const cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error at %s:%d\n  call : %s\n  error: %s (%d)\n",  \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err_),             \
                    static_cast<int>(err_));                                         \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    } while (0)

// A kernel launch (<<<...>>>) returns nothing, so ask the runtime for the error state
// right after it. This catches bad launch configurations (e.g. too many threads per block).
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
