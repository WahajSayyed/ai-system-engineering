// ============================================================================
//  TEST-ONLY STAND-IN for <cuda_runtime.h>.   THIS IS NOT CUDA AND NOT A GPU.
//
//  It runs a "kernel launch" as ordinary nested C++ loops on the CPU, one thread after
//  another, so that the index arithmetic and boundary checks of our kernels can be tested
//  on a machine with no GPU. It is only valid for kernels whose threads never
//  communicate (no __syncthreads, no shared memory, no atomics), which is true of the
//  three kernels in this chapter.
// ============================================================================
#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define __global__
#define __device__
#define __host__

struct uint3 { unsigned x = 0, y = 0, z = 0; };
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

// The "built-in variables". A real GPU gives every thread its own; here the launcher sets them
// before each call of the kernel body.
inline uint3 threadIdx, blockIdx;
inline dim3 blockDim, gridDim;

enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2 };
inline cudaError_t emu_last_error = cudaSuccess;

inline const char* cudaGetErrorString(cudaError_t e) {
    switch (e) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
    }
    return "unknown error";
}
inline cudaError_t cudaGetLastError() { cudaError_t e = emu_last_error; emu_last_error = cudaSuccess; return e; }
inline cudaError_t cudaPeekAtLastError() { return emu_last_error; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };

// "Device" memory is ordinary heap memory of exactly the requested size, so AddressSanitizer
// can catch out-of-bounds accesses.
template <class T> cudaError_t cudaMalloc(T** p, size_t bytes) {
    *p = static_cast<T*>(std::malloc(bytes));
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind) {
    std::memcpy(dst, src, bytes);
    return cudaSuccess;
}

struct EmuEvent { std::chrono::steady_clock::time_point t; };
using cudaEvent_t = EmuEvent*;
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new EmuEvent(); return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e) { e->t = std::chrono::steady_clock::now(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    *ms = std::chrono::duration<float, std::milli>(b->t - a->t).count();
    return cudaSuccess;
}

// ---- launch bookkeeping, so tests can inspect what was launched ------------------------
struct EmuLaunch { std::string name; dim3 grid; dim3 block; };
inline std::vector<EmuLaunch> emu_log;

template <class F>
void emu_launch(const char* name, dim3 grid, dim3 block, F&& kernel_call) {
    const unsigned long long threads_per_block =
        static_cast<unsigned long long>(block.x) * block.y * block.z;
    // Mirror the real runtime's refusal of impossible configurations (limits from the CUDA guide:
    // at most 1024 threads per block; block dimensions at most (1024, 1024, 64)).
    if (threads_per_block == 0 || threads_per_block > 1024 || block.x > 1024 || block.y > 1024 ||
        block.z > 64 || grid.x == 0 || grid.y == 0 || grid.z == 0) {
        emu_last_error = cudaErrorInvalidValue;
        return;
    }
    emu_log.push_back({name, grid, block});
    gridDim = grid;
    blockDim = block;
    for (unsigned bz = 0; bz < grid.z; ++bz)
        for (unsigned by = 0; by < grid.y; ++by)
            for (unsigned bx = 0; bx < grid.x; ++bx) {
                blockIdx = {bx, by, bz};
                for (unsigned tz = 0; tz < block.z; ++tz)
                    for (unsigned ty = 0; ty < block.y; ++ty)
                        for (unsigned tx = 0; tx < block.x; ++tx) {
                            threadIdx = {tx, ty, tz};
                            kernel_call();
                        }
            }
}

// emulate_cuda.py rewrites   name<<<grid, block>>>(args);   into   EMU_LAUNCH(name, grid, block, args);
#define EMU_LAUNCH(name, grid, block, ...) emu_launch(#name, (grid), (block), [&]() { name(__VA_ARGS__); })
