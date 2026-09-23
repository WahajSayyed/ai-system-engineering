# Chapter 9: Kernel Design and Indexing

## 1. Goal

By the end of this chapter you will be able to:

- Explain why FlashAttention's grid is 3D — batch x head x Q-tile — and why the K/V dimension deliberately *isn't* a grid axis.
- Write the `[B, H, N, D]` stride arithmetic that locates one block's slice of Q, K, V, and O, correctly, including for shapes large enough that `int` arithmetic silently breaks.
- Compute a real kernel's total shared-memory budget (Q tile + K tile + V tile + the online-softmax running stats) and know exactly when it needs more than the 48KB default.
- Use `extern __shared__` and `cudaFuncSetAttribute` together to request and receive that larger budget — and know what happens if you request it without the second call.

This chapter starts Part 2: everything from here through Chapter 14 builds one real, working (if not yet fast) attention kernel. Chapter 9's job specifically is the skeleton — the indexing and memory layout every later chapter depends on being correct — not yet the tile-loading loops (Chapter 10) or the actual score computation (Chapters 11-13).

## 2. Concepts

### 2.1 Mapping the algorithm onto blocks: batch x head x Q-tile

Attention is computed independently for every `(batch, head)` pair — nothing in `softmax(QKᵀ/√d)V` for one head depends on any other head, or any other item in the batch. Within one `(batch, head)`, FlashAttention further splits the query sequence into tiles of `Br` rows each, and — critically for what follows — those Q-tiles are *also* independent of each other: each one computes its own slice of the output by looping over *all* the K/V tiles internally (that loop is Chapters 10-13's job, and it happens *inside* a single block, sequentially).

That gives exactly three axes of embarrassingly-parallel work — batch, head, and Q-tile — and zero axes for K/V-tiles, which is why the grid is:

```cpp
dim3 grid(numQTiles, H, B);
```

`blockIdx.x = qTile`, `blockIdx.y = head`, `blockIdx.z = batch`. The choice of *which* axis carries which quantity isn't arbitrary: CUDA caps `gridDim.y` and `gridDim.z` at 65,535 each, while `gridDim.x` can be over two billion. `numQTiles` grows with sequence length — for a long enough sequence it's the one axis with real potential to be large — so it goes on `x`, the axis with no practical ceiling; `H` and `B` are typically small (tens to low hundreds) and comfortably fit on `y`/`z`.

### 2.2 `[B, H, N, D]` stride arithmetic

Every kernel so far has indexed a 2D matrix. Q, K, and V are 4D tensors: `[B, H, N, D]` — batch, head, sequence position, head dimension — stored contiguously in that order (the same row-major idea from Chapter 3's `Q[i*d + j]`, just with two more dimensions folded in). The flat offset of element `(b, h, n, d)` is:

```
offset = ((b * H + h) * N + n) * D + d
```

A block only needs the *start* of its `(batch, head)` slice — everything else (`n`, `d`) is relative to that:

```cpp
long long bhOffset = ((long long)batch * H + head) * (long long)N * D;
const float* Qbh = Q + bhOffset;   // Qbh[n * D + d] is Q[batch, head, n, d]
```

This layout isn't incidental — it's *why* the tiling and coalescing techniques from Chapters 5-6 work at all here. Because everything for one `(batch, head)` is one contiguous `[N, D]` block (exactly the shape Chapters 5-8 have been building toward), a block can locate its slice with one pointer add and then treat it exactly like the 2D matrices every earlier chapter already knows how to tile and load. If `D` were the *outermost* dimension instead, one `(batch, head)`'s data would be scattered across the whole tensor with a huge stride between elements — every technique from Chapter 6 onward would degrade toward the strided, uncoalesced case that chapter warned about.

**Why `long long`, not `int`:** `batch * H + head` and `N * D` are each individually modest, but their *product* — the byte or element count spanning everything before this block's slice — grows with the full size of the tensor. For a shape like `B=32, H=64, N=8192, D=128` (a plausible long-context training batch), the element offset for the last `(batch, head)` pair is `2,146,435,072` — within a hair of `INT32_MAX` (`2,147,483,647`), and the *byte* equivalent of that same offset overflows a 32-bit integer outright. `int` arithmetic here doesn't fail loudly — it wraps silently to a small or negative number, and the kernel reads or writes the wrong memory without any error at all. Casting to `long long` before the multiplication (not after) is what avoids it — the cast has to happen before the overflow-prone multiply, not on the already-wrapped result.

### 2.3 The shared-memory budget, for real this time

Chapter 8's budget only accounted for two tiles (`A`, `B`). A real attention block needs shared memory for the Q tile, the K tile, the V tile, *and* the two small running-statistics arrays Chapter 2's online softmax needs (`m`, the running row max; `l`, the running row sum) — Chapter 12 is where those get used, but the space for them belongs in the same layout this chapter establishes:

```
bytes = (Br*D + Bc*D + Bc*D + Br + Br) * sizeof(float)
      = (Br*D + 2*Bc*D + 2*Br) * sizeof(float)
```

For a perfectly ordinary-looking configuration — `Br = Bc = D = 64` — the three tiles alone total exactly `49,152` bytes: precisely 48KB, with zero room to spare. Add the two 64-element stats arrays (512 more bytes) and the real total is `49,664` bytes — *over* the 48KB default. This isn't a contrived worst case to motivate a rarely-needed feature; it's what happens the first time you count everything a real block actually needs, for a completely unremarkable tile size.

### 2.4 The 48KB default vs. the opt-in limit, precisely

Chapter 5 mentioned that going past 48KB per block needs an opt-in; here's the mechanism. The 48KB default applies to a block's *total* shared memory — static and dynamic combined — and **only dynamic shared memory can opt into more**. A statically-sized `__shared__ float As[Br][D]` (Chapters 5 and 8's style) has no opt-in path at all; its size is fixed at compile time and the compiler enforces the 48KB ceiling on it directly (which is exactly why Chapter 8's `static_assert` checked against that number with no escape hatch). To go beyond it, shared memory has to be **dynamic** — declared with `extern __shared__` and sized at launch time — *and* the kernel has to explicitly request the larger ceiling via `cudaFuncSetAttribute` before that launch. Skip the second step and a launch requesting more than 48KB of dynamic shared memory simply fails at runtime, caught the same way Chapter 5's oversized-`TILE_WIDTH` launch failure was: `cudaGetLastError()` after the launch. The true hardware ceiling for the opt-in differs by architecture — 64KB on Turing (the T4), roughly 100KB on Ampere (the RTX 3090) — so a tile configuration that fits with the opt-in on one card can still fail to fit on the other (Exercise 2).

## 3. Code walkthrough

Four files:

- **`flash_skeleton.h`** — declares the launcher and the small `BlockInfo` struct used to verify indexing.
- **`flash_skeleton.cu`** — the skeleton kernel, the shared-memory layout, and the launcher (which is where `cudaFuncSetAttribute` gets called).
- **`test_flash_skeleton.cpp`** — the test driver: an independent, host-side recomputation of every block's expected `(batch, head, qStart, bhOffset)`, checked against what each block actually wrote.
- **`flash_skeleton_numpy.py`** — the `[B,H,N,D]` offset arithmetic and the shared-memory budget arithmetic, checkable with no GPU (and, for the offset formula, checked directly against real NumPy memory layout, not just hand algebra).

Compile:

```
nvcc -O3 -arch=sm_75 flash_skeleton.cu test_flash_skeleton.cpp -o flash_skeleton_test   # Tesla T4
nvcc -O3 -arch=sm_86 flash_skeleton.cu test_flash_skeleton.cpp -o flash_skeleton_test   # RTX 3090
```

**`flash_skeleton.h`**

```cpp
#pragma once
#include <cstdint>

// Chapter 9: the kernel skeleton -- grid layout, [B,H,N,D] stride
// arithmetic, and the shared-memory budget/opt-in. No tile loading or
// score computation yet (Chapters 10-13).

struct BlockInfo {
    int batch;
    int head;
    int qStart;
    long long bhOffset;
};

// Q, K, V, O are all [B, H, N, D], row-major. debugOut has B*H*numQTiles
// entries, one per block, in (batch, head, qTile) row-major order.
void launchFlashSkeleton(const float* Q, const float* K, const float* V, float* O,
                         int B, int H, int N, int D_runtime,
                         BlockInfo* debugOut);
```

**`flash_skeleton.cu`**

```cpp
// flash_skeleton.cu
// Chapter 9: kernel skeleton -- grid layout, [B,H,N,D] stride arithmetic,
// and the shared memory budget/opt-in. The kernel establishes where every
// block's data lives (global and shared) and records it for verification;
// Chapter 10 fills in the actual tile-load loops.

#include "flash_skeleton.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

template <int Br, int Bc, int D>
__global__ void flashSkeletonKernel(const float* Q, const float* K, const float* V,
                                     float* O, int B, int H, int N,
                                     BlockInfo* debugOut) {
    // ---- shared-memory layout: WHERE each piece will live (Chapter 10 fills them in) ----
    extern __shared__ float smem[];
    float* Qs = smem;                 // [Br, D]
    float* Ks = Qs + Br * D;          // [Bc, D]
    float* Vs = Ks + Bc * D;          // [Bc, D]
    float* m  = Vs + Bc * D;          // [Br]  running row max (Ch2, used from Ch12)
    float* l  = m + Br;               // [Br]  running row sum (Ch2, used from Ch12)
    (void)Qs; (void)Ks; (void)Vs; (void)m; (void)l;   // unused until Ch10-12

    // ---- grid layout: which (batch, head, Q-tile) this block owns ----
    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int qTile = blockIdx.x;
    int qStart = qTile * Br;

    // ---- [B, H, N, D] stride arithmetic: this block's (batch, head) slice ----
    long long bhOffset = ((long long)batch * H + head) * (long long)N * D;
    const float* Qbh = Q + bhOffset;   // Qbh[n*D+d] == Q[batch,head,n,d]
    const float* Kbh = K + bhOffset;
    const float* Vbh = V + bhOffset;
    float* Obh = O + bhOffset;
    (void)Qbh; (void)Kbh; (void)Vbh; (void)Obh;   // used starting Chapter 10

    // Chapters 10-13 load Qbh/Kbh/Vbh into Qs/Ks/Vs, compute scores, run the
    // online softmax through m/l, and accumulate into Obh. This chapter
    // stops once every block can correctly answer "which data is mine, and
    // where does my shared memory live" -- recorded here for verification.
    if (threadIdx.x == 0) {
        int blockLinear = (blockIdx.z * H + blockIdx.y) * gridDim.x + blockIdx.x;
        debugOut[blockLinear] = BlockInfo{batch, head, qStart, bhOffset};
    }
}

template <int Br, int Bc, int D>
void launchFlashSkeletonImpl(const float* Q, const float* K, const float* V, float* O,
                              int B, int H, int N, BlockInfo* debugOut) {
    constexpr size_t sharedBytes = (size_t)(Br * D + 2 * Bc * D + 2 * Br) * sizeof(float);

    // Static shared memory has no opt-in (SS2.4) -- this call is what lets a
    // DYNAMIC request above the 48KB default actually succeed. Comment it
    // out to see the launch below fail instead (Exercise 1).
    CUDA_CHECK(cudaFuncSetAttribute(flashSkeletonKernel<Br, Bc, D>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)sharedBytes));

    int numQTiles = (N + Br - 1) / Br;
    dim3 grid(numQTiles, H, B);
    dim3 block(256);

    flashSkeletonKernel<Br, Bc, D><<<grid, block, sharedBytes>>>(Q, K, V, O, B, H, N, debugOut);
    CUDA_CHECK(cudaGetLastError());
}

void launchFlashSkeleton(const float* Q, const float* K, const float* V, float* O,
                         int B, int H, int N, int D_runtime, BlockInfo* debugOut) {
    switch (D_runtime) {
        case 64: launchFlashSkeletonImpl<64, 64, 64>(Q, K, V, O, B, H, N, debugOut); break;
        default:
            fprintf(stderr, "launchFlashSkeleton: unsupported D=%d (supported: 64)\n", D_runtime);
            exit(1);
    }
}
```

**`test_flash_skeleton.cpp`**

```cpp
// test_flash_skeleton.cpp
// Chapter 9: host-side test driver. Independently recomputes the expected
// (batch, head, qStart, bhOffset) for every block from the same [B,H,N,D]
// shape, and checks it against what each block actually wrote -- this is a
// direct test of SS2.1/SS2.2, not of any attention math (there isn't any yet).

#include "flash_skeleton.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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

int main() {
    const int B = 2, H = 4, N = 200, D = 64, Br = 64;
    const int numQTiles = (N + Br - 1) / Br;   // 4, with a partial last tile (8 valid rows)
    const int numBlocks = B * H * numQTiles;

    // Q/K/V/O contents don't matter for this chapter -- only shapes and offsets do.
    std::vector<float> hQKV(B * H * N * D, 0.0f);
    std::vector<BlockInfo> hDebug(numBlocks);

    float *dQ, *dK, *dV, *dO;
    BlockInfo* dDebug;
    size_t tensorBytes = hQKV.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dQ, tensorBytes));
    CUDA_CHECK(cudaMalloc(&dK, tensorBytes));
    CUDA_CHECK(cudaMalloc(&dV, tensorBytes));
    CUDA_CHECK(cudaMalloc(&dO, tensorBytes));
    CUDA_CHECK(cudaMalloc(&dDebug, numBlocks * sizeof(BlockInfo)));
    CUDA_CHECK(cudaMemcpy(dQ, hQKV.data(), tensorBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hQKV.data(), tensorBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hQKV.data(), tensorBytes, cudaMemcpyHostToDevice));

    launchFlashSkeleton(dQ, dK, dV, dO, B, H, N, D, dDebug);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hDebug.data(), dDebug, numBlocks * sizeof(BlockInfo), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int batch = 0; batch < B; ++batch) {
        for (int head = 0; head < H; ++head) {
            for (int qTile = 0; qTile < numQTiles; ++qTile) {
                int blockLinear = (batch * H + head) * numQTiles + qTile;
                BlockInfo got = hDebug[blockLinear];

                int expectedQStart = qTile * Br;
                long long expectedBhOffset = ((long long)batch * H + head) * (long long)N * D;

                if (got.batch != batch || got.head != head ||
                    got.qStart != expectedQStart || got.bhOffset != expectedBhOffset) {
                    ok = false;
                    printf("MISMATCH block(batch=%d,head=%d,qTile=%d): "
                           "got={batch=%d,head=%d,qStart=%d,bhOffset=%lld} "
                           "expected={batch=%d,head=%d,qStart=%d,bhOffset=%lld}\n",
                           batch, head, qTile, got.batch, got.head, got.qStart, got.bhOffset,
                           batch, head, expectedQStart, expectedBhOffset);
                }
            }
        }
    }

    printf("grid layout + [B,H,N,D] stride arithmetic, %d blocks (B=%d,H=%d,N=%d,numQTiles=%d): %s\n",
           numBlocks, B, H, N, numQTiles, ok ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dDebug);
    return ok ? 0 : 1;
}
```

### Line-by-line

- **`BlockInfo`** — a plain host/device-shared struct, one instance written per block, used purely to verify indexing; it has nothing to do with the real kernel's eventual output.
- **`Qs`/`Ks`/`Vs`/`m`/`l`** — five pointers, computed once each via pointer arithmetic over the single `extern __shared__` buffer, establishing the *layout* Chapter 10 loads real data into. Nothing is written through them yet in this chapter — the `(void)` casts exist purely to tell the compiler these are deliberately unused for now, not a mistake.
- **`bhOffset`** — computed once as `long long`, per §2.2's overflow argument, and reused for all four tensors (Q, K, V, O share the same `[B,H,N,D]` shape and therefore the same per-`(batch,head)` offset).
- **`launchFlashSkeletonImpl`**'s `cudaFuncSetAttribute` call — this is what makes the `<<<grid, block, sharedBytes>>>` launch below it succeed instead of failing; note the kernel argument is `flashSkeletonKernel<Br, Bc, D>` — the specific instantiation, using Chapter 8's explicit-instantiation-adjacent syntax (a template with its arguments filled in names one concrete function, usable as a function pointer here).
- **`launchFlashSkeleton`** — Chapter 8's dispatch pattern again: a runtime `switch` on `D_runtime` calling the one matching template instantiation. Only `D=64` is wired up here, deliberately — this chapter is about getting one configuration's indexing exactly right, not about breadth of configurations (Chapter 8 already covered dispatch itself).
- **`test_flash_skeleton.cpp`**'s triple-nested loop — recomputes, independently and from scratch, what every single block *should* have found for itself, and compares field by field. `N=200` with `Br=64` deliberately produces a partial last tile (`qStart=192`, only 8 valid rows) — this chapter doesn't load any data yet, so the partial tile isn't handled specially here, but the `qStart` value itself must still come out correct for Chapter 10 to handle it.

## 4. C++ decoded

**`extern __shared__ float smem[];`**
Unlike Chapters 5 and 8's `__shared__ float As[Br][D]` — a fixed size, known to the compiler at compile time — `extern __shared__` declares shared memory whose size is supplied at *launch* time, as the third argument inside `<<<grid, block, sharedBytes>>>` (every earlier chapter left this as the implicit default of `0`). The `extern` keyword doesn't have its ordinary C++ meaning here ("defined in another translation unit") — in this specific context it means "the size of this array isn't given here; it arrives externally, from the launch." A kernel may declare at most *one* such array; several logically distinct pieces (`Qs`, `Ks`, `Vs`, `m`, `l`) share the one buffer and are separated only by the pointer arithmetic that follows.

**`Ks = Qs + Br * D;`**
Ordinary C++ pointer arithmetic: adding an integer `n` to a `float*` advances it by `n * sizeof(float)` bytes, not `n` bytes. Since `Qs` occupies exactly `Br*D` floats, `Ks` computed this way starts at precisely the byte immediately after `Qs`'s last element — no gap, no overlap, and no copy: `Ks` is simply another name for a location inside the same underlying buffer `Qs` points into. The nearest Python analogy is slicing one flat buffer into consecutive views (`buf[0:BrD]`, `buf[BrD:BrD+BcD]`, ...) — except a Python slice makes a new object (or, for a memoryview, a new view object), while this is just arithmetic on a raw address.

**`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`**
A CUDA Runtime API call, made once per kernel (per process) before any launch that needs more than the 48KB default of *dynamic* shared memory. The first argument must name one concrete, instantiated kernel — `flashSkeletonKernel<Br, Bc, D>` written without `<<<...>>>` decays to exactly that, a function pointer to one specific compiled instantiation, which is why this call has to live inside `launchFlashSkeletonImpl` (a function template itself), rather than somewhere that only sees the bare, uninstantiated template. Forgetting this call doesn't affect compilation at all — the `.cu` file compiles fine either way — it surfaces only at runtime, when the oversized launch fails and `cudaGetLastError()` reports it (Exercise 1).

**`dim3 grid(numQTiles, H, B);`**
`dim3` is a small CUDA struct with three `unsigned int` fields (`x`, `y`, `z`), default-initialized to `1`; constructing it with three arguments sets all three explicitly. Inside the kernel, `blockIdx.x/.y/.z` read back exactly these three values for whichever block is currently executing — `gridDim.x/.y/.z` (used in `blockLinear`'s computation) read back the *sizes* passed here, available from any thread, in any block, without needing them passed in separately as ordinary arguments.

## 5. Common pitfalls

- **Forgetting `cudaFuncSetAttribute` for a dynamic shared-memory request over 48KB.** The `.cu` file compiles without any complaint; the failure shows up only at the launch, and only if you're checking `cudaGetLastError()` — which is precisely why every kernel in this series has wrapped launches in `CUDA_CHECK` from Chapter 5 onward.
- **Computing `bhOffset` (or any large tensor offset) in `int` instead of `long long`.** Unlike a compile-time `static_assert` catching a bad configuration, an `int` overflow here is a silent runtime wraparound — no crash, no error, just a wrong address, read or written without complaint. Cast to `long long` *before* the multiplication that could overflow, not after — casting the already-wrapped `int` result doesn't recover the lost bits.
- **Putting a possibly-large axis on `gridDim.y` or `gridDim.z`.** Both are capped at 65,535; only `gridDim.x` has no such practical limit. `numQTiles` (which scales with sequence length) belongs on `x` for exactly this reason — see Exercise 3 for what happens if you swap it onto `y` or `z` for a long enough sequence.
- **Assuming the shared-memory budget only needs the tiles you're actively "using" in a given chapter.** This chapter doesn't touch `Qs`/`Ks`/`Vs`/`m`/`l` at all yet, but the *space* for all five is requested up front — the budget describes what the finished kernel (Chapter 13) will need, not what any one intermediate chapter happens to touch.
- **Reusing one GPU's opt-in ceiling as if it were universal.** 64KB (Turing/T4) and roughly 100KB (Ampere/RTX 3090) are different hard limits — a tile configuration validated on one card can still fail `cudaFuncSetAttribute` (or the launch itself) on the other. See Exercise 2.

## 6. Exercises

1. **See the failure `cudaFuncSetAttribute` prevents.** Comment out the `cudaFuncSetAttribute` call in `launchFlashSkeletonImpl` and rerun the test. What does `CUDA_CHECK` report for the launch, and does it happen at the `cudaFuncSetAttribute` line or the `<<<...>>>` line once it's gone?
2. **Find the crossover.** Recompute the shared-memory budget from §2.3 for `Br = Bc = 128`, `D = 64`. Does it fit under the Turing (T4) opt-in ceiling of 64KB? Under the Ampere (RTX 3090) ceiling of roughly 100KB? At what `Br = Bc` (with `D=64` fixed) does it stop fitting on the RTX 3090 too?
3. **Swap the grid axes.** Change `dim3 grid(numQTiles, H, B)` to put `numQTiles` on `z` instead of `x` (and shift the others accordingly), updating the kernel's `blockIdx` reads to match. At roughly what sequence length (given `Br=64`) would `numQTiles` alone exceed `gridDim`'s 65,535 cap on that axis?
4. **Overflow, by hand.** For `B=32, H=64, N=8192, D=128`, compute the element offset `(B*H-1)*N*D` and the corresponding byte offset (`x 4`) yourself — using Python's arbitrary-precision integers is fine for the "true" answer — and confirm which one exceeds `INT32_MAX` (`2^31 - 1`). Check your numbers against `flash_skeleton_numpy.py`'s overflow section.
5. **A compile-time backstop.** Chapter 8 used `static_assert` to catch an oversized *static* shared-memory request at compile time. Add a `static_assert` to `flashSkeletonKernel` that catches a dynamic request too large to fit *even with the opt-in* — using, say, the RTX 3090's ~100KB ceiling as the hard limit — so a badly-chosen template instantiation fails to compile rather than only failing at launch. What value would you choose if you wanted the assert to protect the T4 instead, and why is picking the *smaller* of the two GPUs' ceilings the safer default for a `static_assert` meant to apply everywhere?
