# Chapter 8: Templates and Compile-Time Configuration

## 1. Goal

By the end of this chapter you will be able to:

- Explain why real FlashAttention kernels are template-heavy, in terms of what the compiler can and can't do with a value it only learns at runtime.
- Write a kernel templated on `<int Br, int Bc, int D>` — the three shapes that will matter for the rest of this series: the query tile height, the key/value tile width, and the head dimension.
- Use `static_assert` to catch a bad tile configuration at *compile* time, and `if constexpr` to compile away an entire unused code path rather than merely skipping it at runtime.
- Write a **dispatch function**: the ordinary runtime `if`/`switch` that bridges a runtime value (a model's actual head dimension) onto a small set of pre-compiled template instantiations.

This is one of the more C++-heavy chapters in the series (along with Chapter 14's PyTorch binding and Chapter 17's inline PTX) — worth taking slowly if the syntax is new.

## 2. Concepts

### 2.1 Why FlashAttention kernels are template-heavy

Every kernel so far has hardcoded its tile shape as a `#define`: Chapter 5's `TILE_WIDTH`, Chapter 6's `D`, Chapter 7's `ROW_LEN`. That worked because each chapter only needed *one* shape. A real attention kernel doesn't get that luxury — it needs to run well for head dimensions of 32, 64, or 128, and for tile sizes tuned differently per GPU. Chapter 5 already covers *why* the shape has to be known at compile time: the compiler needs actual numbers, not runtime variables, to fully unroll loops, size `__shared__` arrays, and allocate exactly the registers each thread needs — that's what turned a slow, general loop into a fast, specific one.

The tension: you can't have *one* compiled kernel that's simultaneously optimal for `D=32` and `D=128` — those need different unrolled code, different register counts, different shared-memory layouts. But you also don't want to hand-copy Chapter 5's file four times, once per head dimension, and maintain four near-identical copies forever.

A C++ **template** is the resolution: write the kernel's logic exactly once, as source, and let the compiler generate a separate, independently-optimized compiled version for every combination of template arguments the program actually uses. Coming from Python, this is the part that doesn't have a direct analogy — in Python, one function handles every shape uniformly, at runtime, with no separate "copies" ever created. A C++ template is closer to a code generator that runs at compile time: `tiledScoreKernel<64, 64, 32>` and `tiledScoreKernel<32, 32, 128>` are two *entirely separate* pieces of compiled machine code, each as fully specialized as if you'd hand-written it for that one shape — the compiler just wrote both copies for you, from one source.

### 2.2 What the compiler actually does with a template parameter

A template parameter like `D` in `template <int Br, int Bc, int D>` isn't a variable that gets passed in at runtime — it's a compile-time constant, exactly like `TILE_WIDTH` was in Chapter 5, except the *same source file* can now be instantiated with several different constants instead of one. Anywhere `D` appears in the function body, the compiler substitutes the literal number for that specific instantiation before it does anything else: `for (int k = 0; k < D; ++k)` becomes, for `tiledScoreKernel<64,64,32>`, a loop the compiler *knows* runs exactly 32 times — fully unrollable with `#pragma unroll`, exactly like Chapter 5's `TILE_WIDTH`-bounded loop.

You can go further and derive *new* compile-time constants from a template parameter with an ordinary `constexpr` declaration — `constexpr int D4 = D / 4;` inside the kernel body computes `D4` once, at compile time, for each instantiation; no division ever happens on the GPU. `constexpr` isn't a CUDA-specific keyword — it's standard C++ for "the compiler can and must compute this before the program runs" — but template parameters are exactly the kind of value that make `constexpr` arithmetic possible in the first place.

### 2.3 Catching bad configurations before the program ever runs

Chapter 5 pointed out that an oversized `TILE_WIDTH` blows the shared-memory budget as a *launch failure* — something you only discover by running the kernel and checking `cudaGetLastError()`. With `Br`, `Bc`, `D` as template parameters, the total shared-memory footprint (`Br*D + Bc*D` floats) is itself a compile-time constant — which means a `static_assert` can check it *before the program is ever built*, not just before it's ever run:

```cpp
static_assert(sharedBytes <= 48 * 1024, "...");
```

If this fails, `nvcc` refuses to produce a binary at all, with an error message pointing at this line — a strictly earlier place to catch the mistake than a runtime `cudaGetLastError()` check, and one that requires no GPU, no launch, and no test run to trigger.

### 2.4 `if constexpr`: deleting a branch, not just skipping it

An ordinary `if (D % 4 == 0) { ... } else { ... }` compiles *both* branches into the function and picks one at runtime with an actual branch instruction — wasteful when the condition's answer is already fixed at compile time for a given instantiation, and occasionally impossible: if the two branches contain code that only type-checks for certain template arguments, an ordinary `if` still requires *both* to compile, for every instantiation, whether taken or not.

`if constexpr` fixes both problems. Its condition must be something the compiler can evaluate at compile time (`D % 4 == 0`, where `D` is a template parameter, qualifies), and the branch that isn't taken is deleted from that instantiation entirely — not "skipped at runtime," but never emitted as code in the first place. For `loadTileRowMajor<Br, 32>`, the scalar fallback branch doesn't exist anywhere in the compiled kernel; only the `float4` branch does. This is exactly Chapter 6's vectorized-vs-scalar tile load, now folded into one function that specializes itself per instantiation instead of needing two separately hand-written kernels.

### 2.5 Dispatch: bridging a runtime value onto compile-time templates

Here's the mismatch this whole chapter is building toward: `Br`, `Bc`, `D` must be known at compile time to write `tiledScoreKernel<Br, Bc, D>` — but the actual head dimension a model uses is a runtime value (read from a config, passed as a function argument). You cannot write `tiledScoreKernel<Br, Bc, someIntVariable>` — that's not legal C++, template arguments can't be ordinary variables.

The standard fix is a **dispatch function**: an ordinary runtime `if`/`switch` that inspects the runtime value and calls whichever *already-compiled* template instantiation matches it. This isn't a workaround specific to this chapter's toy kernel — every real FlashAttention implementation has a dispatch layer of exactly this shape, and the kernel this series builds starting in Chapter 9 will too.

## 3. Code walkthrough

Five files, same split as recent chapters:

- **`tiled_score.h`** — declares just the dispatch entry point, `launchTiledScore`. The template machinery stays inside the `.cu` file.
- **`tiled_score.cu`** — the templated kernel, the templated load helper, explicit instantiations, and the dispatch function.
- **`test_tiled_score.cpp`** — the test driver, exercising all four dispatch cases.
- **`tiled_score_numpy.py`** — the same tiling algorithm as an ordinary parameterized Python function, to make the Python-vs-C++-templates contrast from §2.1 concrete.

Compile:

```
nvcc -O3 -arch=sm_75 tiled_score.cu test_tiled_score.cpp -o tiled_score_test   # Tesla T4
nvcc -O3 -arch=sm_86 tiled_score.cu test_tiled_score.cpp -o tiled_score_test   # RTX 3090
```

**`tiled_score.h`**

```cpp
#pragma once

// Chapter 8: A @ B^T (foreshadows S = Q K^T), templated on tile shape.
// Br, Bc, D are compile-time template parameters inside tiled_score.cu --
// this header only exposes the runtime dispatch entry point, since the
// caller's D is a runtime value (SS2.5), not something it can template on.

void launchTiledScore(const float* dA, const float* dB, float* dC,
                       int M, int N, int D);
```

**`tiled_score.cu`**

```cpp
// tiled_score.cu
// Chapter 8: templates and compile-time configuration.
// Computes C[M,N] = A[M,D] @ B[N,D]^T -- foreshadows S = Q K^T (Chapter 11) --
// with the tile shape (Br, Bc, D) as compile-time template parameters instead
// of Chapter 5/6's per-file #defines.
//
// Compile as part of the test binary:
//   nvcc -O3 -arch=sm_75 tiled_score.cu test_tiled_score.cpp -o tiled_score_test   (Tesla T4)
//   nvcc -O3 -arch=sm_86 tiled_score.cu test_tiled_score.cpp -o tiled_score_test   (RTX 3090)

#include "tiled_score.h"
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

// ---------------------------------------------------------------------------
// Templated tile loader: Rows and D are template parameters, so the same
// source serves both the [Br, D] "A" tile and the [Bc, D] "B" tile below --
// two call sites, one function, no duplicated loop body. D's if constexpr
// branch (Chapter 6's vectorized vs scalar load) is resolved per
// instantiation, not per call.
// ---------------------------------------------------------------------------

template <int Rows, int D>
__device__ __forceinline__ void loadTileRowMajor(const float* src, float dst[][D],
                                                  int tileStart, int numRows,
                                                  int tid, int numThreads) {
    if constexpr (D % 4 == 0) {
        constexpr int D4 = D / 4;   // computed once, at compile time, per instantiation
        for (int idx = tid; idx < Rows * D4; idx += numThreads) {
            int r = idx / D4, c4 = idx % D4;
            int globalRow = tileStart + r;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (globalRow < numRows) {
                v = reinterpret_cast<const float4*>(&src[globalRow * D])[c4];
            }
            reinterpret_cast<float4*>(&dst[r][0])[c4] = v;
        }
    } else {
        for (int idx = tid; idx < Rows * D; idx += numThreads) {
            int r = idx / D, c = idx % D;
            int globalRow = tileStart + r;
            dst[r][c] = (globalRow < numRows) ? src[globalRow * D + c] : 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// The kernel itself. Block size is deliberately decoupled from Br/Bc/D (see
// the strided idx += numThreads loops below) -- one launch config (256
// threads, flat 1D block) works across every tile shape this file supports.
// ---------------------------------------------------------------------------

template <int Br, int Bc, int D>
__global__ void tiledScoreKernel(const float* A, const float* B, float* C, int M, int N) {
    // A: [M, D] row-major (foreshadows Q)
    // B: [N, D] row-major (foreshadows K, used transposed below)
    // C: [M, N] row-major (foreshadows S = Q K^T)
    static_assert(Br > 0 && Bc > 0 && D > 0, "tile dimensions must be positive");

    constexpr int sharedBytes = (Br * D + Bc * D) * sizeof(float);
    static_assert(sharedBytes <= 48 * 1024,
                  "Br/Bc/D exceed the 48KB default shared memory budget (Ch5 SS2.2) -- "
                  "shrink the tile, or see Ch9 for the cudaFuncSetAttribute opt-in");

    __shared__ float As[Br][D];
    __shared__ float Bs[Bc][D];

    int tileRow = blockIdx.y * Br;
    int tileCol = blockIdx.x * Bc;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    loadTileRowMajor<Br, D>(A, As, tileRow, M, tid, numThreads);
    loadTileRowMajor<Bc, D>(B, Bs, tileCol, N, tid, numThreads);
    __syncthreads();

    for (int idx = tid; idx < Br * Bc; idx += numThreads) {
        int r = idx / Bc, c = idx % Bc;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < D; ++k) {
            acc += As[r][k] * Bs[c][k];   // A row r . B row c => (A @ B^T)[r,c]
        }
        int globalRow = tileRow + r, globalCol = tileCol + c;
        if (globalRow < M && globalCol < N) {
            C[globalRow * N + globalCol] = acc;
        }
    }
}

// Explicit instantiation: forces the compiler to generate code for exactly
// these four (Br, Bc, D) combinations in this translation unit, rather than
// relying on the implicit instantiation the <<<...>>> calls below would
// trigger anyway. In a single-file build like this one the practical effect
// is small (see Exercise 5) -- the real payoff shows up in bigger projects
// that split kernel definitions and their instantiations across files, so
// each translation unit only compiles the variants it actually needs.
template __global__ void tiledScoreKernel<64, 64, 32>(const float*, const float*, float*, int, int);
template __global__ void tiledScoreKernel<64, 64, 64>(const float*, const float*, float*, int, int);
template __global__ void tiledScoreKernel<64, 64, 63>(const float*, const float*, float*, int, int);
template __global__ void tiledScoreKernel<32, 32, 128>(const float*, const float*, float*, int, int);

template <int Br, int Bc, int D>
void launchTiledScoreImpl(const float* dA, const float* dB, float* dC, int M, int N) {
    dim3 block(256);
    dim3 grid((N + Bc - 1) / Bc, (M + Br - 1) / Br);
    tiledScoreKernel<Br, Bc, D><<<grid, block>>>(dA, dB, dC, M, N);
    CUDA_CHECK(cudaGetLastError());
}

// The dispatch function: D is a runtime int here (unlike the kernel's D
// above), because that's what a caller actually has. Note Br/Bc aren't just
// D echoed three ways -- D=128 needs a smaller tile (32x32) to fit the
// shared-memory budget from SS2.3; D=32/63/64 all fit comfortably at 64x64.
void launchTiledScore(const float* dA, const float* dB, float* dC, int M, int N, int D) {
    switch (D) {
        case 32:  launchTiledScoreImpl<64, 64, 32>(dA, dB, dC, M, N); break;
        case 64:  launchTiledScoreImpl<64, 64, 64>(dA, dB, dC, M, N); break;
        case 63:  launchTiledScoreImpl<64, 64, 63>(dA, dB, dC, M, N); break;
        case 128: launchTiledScoreImpl<32, 32, 128>(dA, dB, dC, M, N); break;
        default:
            fprintf(stderr, "launchTiledScore: unsupported D=%d (supported: 32, 63, 64, 128)\n", D);
            exit(1);
    }
}
```

**`test_tiled_score.cpp`**

```cpp
// test_tiled_score.cpp
// Chapter 8: host-side test driver. Exercises all four dispatch cases --
// D=32, 64, 128 through the float4 branch of loadTileRowMajor's if
// constexpr, and D=63 through the scalar fallback branch -- against a
// plain triple-loop CPU reference for A @ B^T.

#include "tiled_score.h"

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

static void cpuMatMulT(const std::vector<float>& A, const std::vector<float>& B,
                        std::vector<float>& C, int M, int N, int D) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += A[i * D + k] * B[j * D + k];
            C[i * N + j] = acc;
        }
}

static bool testOneConfig(int D, int M, int N) {
    std::vector<float> hA(M * D), hB(N * D), hC(M * N), hRef(M * N);
    for (auto& x : hA) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (auto& x : hB) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    launchTiledScore(dA, dB, dC, M, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cpuMatMulT(hA, hB, hRef, M, N, D);

    float maxErr = 0.0f;
    for (size_t i = 0; i < hC.size(); ++i) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    bool ok = maxErr < 1e-2f;   // fp32 accumulation over up to D=128 terms
    printf("D=%3d (M=%d, N=%d): max abs error = %e -> %s\n", D, M, N, maxErr, ok ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok;
}

int main() {
    bool ok = true;
    ok &= testOneConfig(32, 200, 150);
    ok &= testOneConfig(64, 200, 150);
    ok &= testOneConfig(63, 200, 150);   // exercises the scalar (non-float4) branch
    ok &= testOneConfig(128, 200, 150);
    return ok ? 0 : 1;
}
```

### Line-by-line

- **`template <int Rows, int D>` on `loadTileRowMajor`** — one function, called twice below with different `Rows` (`Br` for the A tile, `Bc` for the B tile). Chapters 5 and 6 would have needed two hand-written, near-duplicate loops for this; a template needs the loop body written once.
- **`float dst[][D]`** — a function parameter written this way is adjusted by the compiler to `float (*dst)[D]`: a pointer to arrays of `D` floats. This is why `D` can be *deduced* from the argument at a call site that doesn't specify it explicitly (see §4) — `D` is baked into the parameter's actual type, not just used inside the function body.
- **`if constexpr (D % 4 == 0)`** — resolved once per instantiation, at compile time. For every call in this file, `D` is a template parameter, so this is always statically known — never a runtime branch.
- **`static_assert(sharedBytes <= 48 * 1024, ...)`** inside `tiledScoreKernel` — fires at compile time for any instantiation whose tile is too big. Try instantiating `tiledScoreKernel<64, 64, 128>` yourself (Exercise 1) to see it happen.
- **The explicit instantiation block** — four `template __global__ void tiledScoreKernel<...>(...)` lines naming exactly the combinations this file supports, immediately after the template definition.
- **`launchTiledScoreImpl`** — a *function* template (not a kernel), one instantiation per `(Br, Bc, D)`, each launching the matching kernel instantiation.
- **`launchTiledScore`** — the dispatch function from §2.5: an ordinary runtime `switch` on a runtime `int D`, calling the one pre-compiled `launchTiledScoreImpl` instantiation that matches. Notice `Br`/`Bc` aren't a fixed function of `D` — `D=128` needs a smaller tile to fit the shared-memory budget, which the dispatch table encodes explicitly rather than computing.

## 4. C++ decoded

**`template <int Br, int Bc, int D>`**
A *non-type* template parameter list: unlike the more familiar `template <typename T>` (a placeholder for a *type*), `int Br` here is a placeholder for a compile-time *integer value*. Every distinct set of `(Br, Bc, D)` values used anywhere in the program causes the compiler to generate one distinct compiled function — `tiledScoreKernel<64,64,32>` and `tiledScoreKernel<32,32,128>` share no compiled code at all, only source.

**Template argument deduction (and when it doesn't apply)**
When a template parameter appears in a function argument's *type*, the compiler can often figure out its value from the argument itself, without you writing it in `<...>`. `loadTileRowMajor<Br, D>(A, As, ...)` gives both explicitly, but `D` didn't strictly need to be — since `As` has static type `float (*)[D]`, calling `loadTileRowMajor<Br>(A, As, ...)` (only `Br` explicit) would let the compiler deduce `D` from `As`'s type on its own. `Br`, though, never appears in any parameter's type — it's only used inside the function body as a loop bound — so it can *never* be deduced and must always be given explicitly. The same asymmetry shows up starkly at the kernel launch site: `tiledScoreKernel<Br, Bc, D><<<grid, block>>>(...)` must give all three explicitly, always, because *none* of them appear in a parameter type the compiler could inspect — `A`, `B`, `C` are all just `const float*` / `float*`, carrying no shape information at all.

**`constexpr`**
A general C++ keyword (not CUDA-specific) marking a value or function as computable at compile time. `constexpr int D4 = D / 4;` is not "a variable that happens to get the value early" — it's an instruction to the compiler to perform the division once, while compiling, and treat `D4` from then on as if you'd typed the literal number yourself.

**`static_assert(condition, "message")`**
Also standard C++, not CUDA-specific. Unlike a runtime `assert` (which checks a condition while the program is *running* and can be disabled in release builds), `static_assert`'s condition must be evaluable at compile time — a template parameter, or an expression built purely from one, qualifies. If the condition is false, compilation stops with an error containing your message, before any code is generated for that instantiation at all.

**`if constexpr (condition) { ... } else { ... }`**
Standard C++17. The condition must be a compile-time constant. Whichever branch isn't taken is removed from that instantiation's compiled output entirely — not dead code the optimizer *might* strip, but code that was never emitted for that instantiation in the first place. An ordinary runtime `if` with the same condition would still compile and keep both branches; `if constexpr` is a different, earlier mechanism.

**`make_float4(0.0f, 0.0f, 0.0f, 0.0f)`**
A CUDA-provided constructor function for the `float4` vector type from Chapter 6 — a plain, ordinary function call (not special syntax) that returns a `float4` with all four fields set, used here as the zero-pad value for an out-of-bounds vectorized load.

## 5. Common pitfalls

- **Trying to template on a runtime value.** `tiledScoreKernel<Br, Bc, someRuntimeInt>` is not legal C++ — template arguments must be compile-time constants. This is precisely the gap the dispatch function exists to close; if you find yourself wanting to do this directly, you want a `switch` over the supported values instead.
- **Forgetting `if constexpr` and using a plain `if` on a template-parameter condition.** It'll still work *functionally* in simple cases, but both branches stay in every instantiation's compiled code (larger binary, and in trickier cases, a branch that doesn't even type-check for some instantiation will fail to compile even though it's never taken).
- **A `static_assert` that isn't actually evaluable at compile time.** The condition must be built entirely from compile-time constants (template parameters, `constexpr` values, literals) — passing an ordinary runtime variable into a `static_assert` condition is a compile error, not a deferred runtime check.
- **Assuming explicit instantiation changes behavior in a single-file build.** Here, it mostly documents intent (Exercise 5 has you verify this directly) — its real value is controlling what gets compiled *per translation unit* in a multi-file project, which this chapter's single `.cu` file doesn't need.
- **Picking `Br`/`Bc` as if they were independent of `D`.** The shared-memory budget couples all three — see `D=128`'s smaller tile in the dispatch table. Treating tile size as a constant chosen once, regardless of head dimension, is exactly how you trip the `static_assert` in Exercise 1.

## 6. Exercises

1. **Trigger the `static_assert` on purpose.** Add a fifth explicit instantiation, `tiledScoreKernel<64, 64, 128>`, and try to compile. Confirm it fails with the message from §2.3, and compute by hand why `64*128*2*4` bytes exceeds the 48KB budget while `32*128*2*4` (the dispatch table's actual choice for `D=128`) doesn't.
2. **Exploit deduction.** Change one of the `loadTileRowMajor<Br, D>` / `loadTileRowMajor<Bc, D>` call sites in `tiledScoreKernel` to drop the second template argument (`loadTileRowMajor<Br>(...)`) and confirm it still compiles and produces the same result — `D` gets deduced from the `dst` argument's type, exactly as §4 describes.
3. **Add a fifth dispatch case.** Work out a `Br`/`Bc` for `D=256` that fits the 48KB budget (assume `Br == Bc` for simplicity), add the explicit instantiation, the `launchTiledScoreImpl` call, and a new `case 256:` in `launchTiledScore`, and confirm `test_tiled_score.cpp` passes with a new `testOneConfig(256, ...)` call.
4. **Which loops does `#pragma unroll` actually unroll?** `tiledScoreKernel` has three loops using the `idx += numThreads` pattern (two inside `loadTileRowMajor`, one for the output), and one `#pragma unroll`-annotated loop over `k < D`. Why can the compiler fully unroll the `k` loop but not the `idx`-strided ones, even though `Br*Bc`, `Br*D4`, etc. are all compile-time constants too? (Hint: what is `numThreads` — a template parameter, or something else?)
5. **Check the explicit-instantiation claim.** Comment out all four `template __global__ void tiledScoreKernel<...>` lines and rebuild. Does `test_tiled_score.cpp` still pass? What does that tell you about where explicit instantiation actually matters, versus where it's just documentation, in a single-file build like this one?
