# Chapter 7: Warps, Shuffles, and Reductions

## 1. Goal

By the end of this chapter you will be able to:

- Explain SIMT execution and warp divergence: what a warp actually is, and what happens when threads in one take different branches.
- Write a warp-level **max** and **sum** reduction using shuffle instructions, with no shared memory and no `__syncthreads()`.
- Explain the real difference between `__shfl_down_sync` (answer ends up in lane 0 only) and `__shfl_xor_sync` (answer ends up in every lane), and pick the right one for the job.
- Build a **standalone warp-softmax kernel**: Chapter 2's max → exp → sum → divide, done entirely in registers across one warp's 32 lanes.

## 2. Concepts

### 2.1 SIMT execution: what a warp actually is

Chapter 6 used the word "warp" as a preview; here's the real definition. A **warp** is a fixed group of 32 consecutive threads (indices `0-31`, `32-63`, and so on within a block) that the hardware schedules as one unit. This execution model is called **SIMT** — Single Instruction, Multiple Threads: every thread in a warp executes the *same instruction* at the *same time*, just on its own data and its own registers. It's not 32 independent threads that happen to run close together; it's one instruction stream, fanned out across 32 lanes.

This is exactly what makes a shuffle instruction possible: since all 32 lanes are executing the *same* `__shfl_xor_sync` call at the *same* moment, the hardware can wire lane-to-lane register reads directly, without ever touching memory.

### 2.2 Warp divergence

SIMT execution has a cost when threads in a warp disagree about which branch to take:

```cpp
if (lane < 16) {
    // path A
} else {
    // path B
}
```

The warp can't run path A on some lanes and path B on others *simultaneously* — there's only one instruction stream. Instead, the hardware runs the whole warp through path A with the path-B lanes masked off (predicated to do nothing), then runs the whole warp through path B with the path-A lanes masked off. Both paths execute, serially, for every lane — the ones "not currently active" just discard their results. A perfectly balanced 50/50 branch inside a single warp doesn't parallelize the two halves; it costs roughly the sum of both paths' time. This has no effect on *correctness* — it's a silent performance tax, not a bug — but it's exactly the reason Chapter 11's causal masking (`if` on whether a key position is visible) needs care about which lanes disagree and how often.

### 2.3 Register-to-register communication: shuffles

Every reduction so far in this series has gone through shared memory: write to `__shared__`, `__syncthreads()`, read back. A **shuffle** instruction skips both steps — a thread can read another lane's *register* directly, with no shared memory array involved and no barrier needed beyond the shuffle instruction itself.

Two variants matter here:

- **`__shfl_down_sync(mask, val, delta)`** — each lane's result is its own `val` combined with the val held by the lane `delta` positions *higher* (lane `i` reads lane `i + delta`). If `i + delta` is out of range, the lane reads back its own value instead. Run in a loop with `delta = 16, 8, 4, 2, 1`, this is the classic tree reduction — but only **lane 0** ends up holding the fully combined result. Every other lane holds some partial, not-generally-useful intermediate value.
- **`__shfl_xor_sync(mask, val, laneMask)`** — each lane exchanges with the lane at `myLane XOR laneMask`. Because XOR pairing is symmetric (if lane `a` reads from lane `b`, lane `b` also reads from lane `a`), running this in the same loop (`laneMask = 16, 8, 4, 2, 1`) produces a **butterfly** reduction where *every* lane ends up holding the full result — no separate broadcast step needed.

For online softmax, every lane needs the row max and the row sum (each lane subtracts the same max and divides by the same sum), so the XOR version is the one actually used in this chapter's softmax kernel. The down version is still worth knowing — it's what you reach for when only one lane needs the answer (e.g. a single thread writing a per-row scalar out to global memory).

Both intrinsics take a **mask** argument: a 32-bit value, one bit per lane, marking which lanes are participating in this exact call. `0xffffffff` (all 32 bits set) means "the full warp is here." This isn't a formality — a lane whose bit is missing from the mask, or a lane that executes the instruction while its bit *is* set but it's no longer actually converged with the others (say, because of an earlier divergent branch), is undefined behavior. The `_sync` in the name exists because these replaced older, unmasked shuffle intrinsics that had exactly this failure mode.

## 3. Code walkthrough

Five files:

- **`warp_reduce.h`** — declares the two launchers.
- **`warp_reduce.cu`** — the reduction helpers, both kernels, and the launchers. No `main()`.
- **`test_warp_reduce.cpp`** — the test driver.
- **`warp_reduce_numpy.py`** — a step-by-step numpy trace of both reduction patterns, no GPU needed — this is the tool for Exercise 1, and the thing that caught a subtle bug (below) before any CUDA got written.

Compile:

```
nvcc -O3 -arch=sm_75 warp_reduce.cu test_warp_reduce.cpp -o warp_reduce_test   # Tesla T4
nvcc -O3 -arch=sm_86 warp_reduce.cu test_warp_reduce.cpp -o warp_reduce_test   # RTX 3090
```

**`warp_reduce.h`**

```cpp
#pragma once

// Chapter 7: warp-level reductions and a standalone warp-softmax kernel.
// ROW_LEN is fixed at 32 (one warp handles one row) inside warp_reduce.cu --
// Chapter 9 generalizes beyond a single warp.

void launchReductionComparison(const float* dIn, float* dOutXor, float* dOutDown, int numRows);
void launchWarpSoftmax(const float* dIn, float* dOut, int numRows);
```

**`warp_reduce.cu`**

```cpp
// warp_reduce.cu
// Chapter 7: warp-level reductions via shuffle instructions, and a
// standalone warp-wide softmax built entirely out of them -- no shared
// memory, no __syncthreads(), everything stays in registers within one warp.
// Library only, no main() -- see test_warp_reduce.cpp for the driver.

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
```

**`test_warp_reduce.cpp`**

```cpp
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
```

### Line-by-line

- **`warpReduceMaxXor` / `warpReduceSumXor`** — five iterations (`offset = 16, 8, 4, 2, 1` — that's `log2(32)`), each combining the current value with the lane `offset` away by XOR. After the loop, every lane has participated in every pairing, so every lane holds the full reduction.
- **`warpReduceSumDown`** — same five-iteration shape, but `__shfl_down_sync` instead of `__shfl_xor_sync`. Only lane 0's chain of partners stays "live" through all 5 steps (lane 0 always has a valid partner at every offset, since `0 + offset < 32` for every offset used); the other lanes' final values are real numbers, just not the row sum.
- **`reductionComparisonKernel`** — runs both reductions side by side on the same input, purely so the difference between them is something you can print and check, not just something you take on faith.
- **`warpSoftmaxKernel`** — the actual point of the chapter: Chapter 2's "3-pass safe softmax" (subtract the max, exponentiate, divide by the sum), except now every one of those three quantities that needs to be shared across the row (`m`, `l`) gets there via a 5-instruction shuffle reduction instead of a shared-memory round trip. `__expf` is CUDA's fast approximate exponential intrinsic; Chapter 12 covers the accuracy/speed tradeoff between it, `expf`, and `exp2f` in depth — here it's fine because softmax normalizes away most of the approximation error and the test tolerance (`1e-3`) reflects that.

## 4. C++ decoded

**`__shfl_xor_sync(mask, var, laneMask)`**
Every lane calls this with its own `var`. Each lane's return value is the `var` held by the lane at `myLaneID XOR laneMask` — not a memory read, a direct register exchange between two specific lanes, resolved by the hardware's crossbar. Because XOR is its own inverse (`a XOR b XOR b == a`), whoever lane `a` reads from is also reading from lane `a` — the exchange is always a two-way, symmetric swap, which is exactly why repeating it with halving `laneMask` values produces a butterfly network where every lane converges on the same answer.

**`__shfl_down_sync(mask, var, delta)`**
Same idea, asymmetric: lane `i` receives the `var` held by lane `i + delta`. If `i + delta >= 32` (no such lane), the intrinsic returns the calling lane's own `var` unchanged instead. This asymmetry is exactly why only lane 0 ends up correct — lane 0's partner chain never runs out of valid partners across the 5 steps, but higher lanes' partner chains do, at different points, and are quietly "reading themselves" from then on instead of accumulating more of the row.

**The `0xffffffff` mask**
A 32-bit unsigned literal in hexadecimal, written as `0x` followed by 8 hex digits. Each hex digit is 4 bits, so 8 digits of `f` (`1111` in binary) is 32 bits, all set to `1`. Read as a bitmask over lane IDs 0 through 31, "all bits set" means "every lane in the warp is expected to hit this instruction." It's the same hex-literal syntax Python uses (`0xffffffff` means the same thing in both languages) — the CUDA-specific part is what the bits *mean* here: lane participation, not a plain integer value.

**`__device__ __forceinline__`**
`__device__` marks a function as callable only from other device code (kernels or other `__device__` functions) — as distinct from `__global__` (a kernel, callable from the host, launched with `<<<...>>>`) or `__host__` (ordinary CPU-callable code; a function can even be marked `__host__ __device__` to compile both ways). `__forceinline__` is a stronger version of the standard `inline` keyword: instead of merely suggesting that the compiler *may* inline the function, it tells `nvcc` to always substitute the function body directly at the call site. For a 5-instruction helper called on every single element of every row, the overhead of an actual function call would rival the work being done — inlining also lets the compiler allocate registers across the reduction and its caller as one unit, rather than at a function boundary.

## 5. Common pitfalls

- **A mask that doesn't cover every lane that executes the instruction.** If some lanes have diverged away (an early `return`, a different branch) and aren't included in the mask, or if the mask excludes a lane that still reaches the shuffle, the result is undefined — in the worst case, a hang, since a shuffle is a synchronizing operation for the lanes it names.
- **Assuming `__shfl_down_sync`'s result is valid in every lane.** It isn't, by design. Using its output outside lane 0 without an explicit broadcast (e.g. `__shfl_sync(mask, val, 0)` to copy lane 0's value to everyone) is a real, easy-to-make bug — see Exercise 2.
- **Using the down-reduction where the xor-reduction was needed.** Online softmax needs the max and sum in *every* lane (every lane subtracts/divides). Swapping in `warpReduceSumDown` for `warpReduceSumXor` inside `warpSoftmaxKernel` would silently produce garbage in 31 out of every 32 output elements — no error, no crash, just wrong numbers most places.
- **Forgetting these kernels assume exactly one full warp per row.** `ROW_LEN` is hardcoded to 32 and the reductions assume all 32 lanes are real, active row elements. Point this code at a row of any other length and the indexing is simply wrong — Chapters 9-12 build the version that handles arbitrary row lengths across multiple warps.
- **Reasoning about divergence cost from "correctness" instead of "both paths run."** A divergent `if/else` inside a warp is not a bug — the result is correct — but it is not free, either. The mistake is assuming a balanced branch "parallelizes" the two paths; SIMT execution means it serializes them instead.

## 6. Exercises

1. **Trace it by hand.** Run `warp_reduce_numpy.py`'s 8-lane toy trace for values `[1..8]`. At each of the 3 steps, write down what you'd expect for a couple of lanes before running it, then compare. Confirm every XOR lane ends at `36` and only DOWN lane 0 does.
2. **Fix the down-version.** Add a broadcast to `warpReduceSumDown`'s result — after the reduction loop, use `__shfl_sync(FULL_MASK, val, 0)` to copy lane 0's value into every lane — and confirm the broadcasted result now matches `warpReduceSumXor`'s output exactly, for every lane.
3. **Break the mask on purpose.** Change `FULL_MASK` to `0x0000ffff` (only lanes 0-15) inside `warpReduceSumXor` and run the test. Does it fail to compile, hang, crash, or silently produce wrong numbers for lanes 16-31? Explain why, from the rule that every lane executing the instruction must have its bit set in the mask.
4. **Measure divergence, indirectly.** Write a small kernel where `lane < 16` does a cheap loop (100 iterations of `x += 1.0f`) and the other half does the same loop 1,000,000 times, inside an `if/else`. Using `cudaEvent_t` timing (a preview of Chapter 14), does the warp finish closer to the cheap path's cost or the expensive path's cost? What does that tell you about how SIMT handles a divergent branch?
5. **Beyond one warp.** Extend `warpSoftmaxKernel` conceptually to `ROW_LEN = 64` — two warps' worth of one row — without writing the code yet. What, specifically, does `warpReduceSumXor` fail to account for once a row spans more than one warp? (This is exactly the gap Chapters 9-12 close with shared memory and cross-warp communication.)
