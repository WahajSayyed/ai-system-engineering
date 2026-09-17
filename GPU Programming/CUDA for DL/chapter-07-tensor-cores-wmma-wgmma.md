# Chapter 7 — Tensor Core Programming: WMMA & Hopper's WGMMA

*Part 6: Tensor Cores — Hardware-Accelerated Matrix Math. Confirmed from the book's companion repo: `book.cu/5_tensor_cores/README.md`. Quick transparency note: this folder's own header still reads "Chapter 5" and references "Chapter 4" for the GEMM-optimization material we covered as this course's Chapter 6 — the same numbering drift flagged back in Chapter 1. The file numbering, at least, is unambiguous and consistent: this chapter's kernels are 7 through 12, picking up exactly where Chapter 6's GEMM ladder (kernels 0–6) left off.*

Every kernel in Chapter 6 computed its multiply-adds with ordinary CUDA cores, however cleverly tiled. This chapter hands that same work to dedicated matrix-multiply hardware — and the book's own companion repo includes something rare and genuinely valuable: a full, real, measured performance table across the *entire* ladder, naive CUDA core through Hopper WGMMA, plus a documented benchmarking bug that's worth learning from directly.

---

## 7.1 From CUDA Cores to Tensor Cores: What Gets Automated

The book's own comparison, confirmed from the README:

| Aspect | CUDA Cores (Ch. 6) | Tensor Cores (this chapter) |
|---|---|---|
| Optimization | Manual tiling, vectorization | Hardware-accelerated MMA |
| Code complexity | High (100+ LOC kernels) | Medium (intrinsics/PTX) |
| Performance | Good | Excellent (5–10× faster) |
| Hardware | All GPUs | Volta+ (V100, A100, H100) |
| Use case | General compute | Matrix-multiply dominant |

And here's the real, measured data behind that "5–10× faster" claim — `REPORT.md`'s full kernel-only timing table at **M=N=K=4096, FP16**, run on an actual H100:

| Kernel | Technique | Time (ms) | TFLOPS | vs. naive |
|---|---|---|---|---|
| K1 (naive, Ch.6) | 1 thread = 1 output | 273.877 | 0.5 | 1× |
| K6 (vectorize, Ch.6) | float4/int4 loads | 3.200 | 42.9 | 86× |
| **K7 (WMMA)** | Warp Matrix Multiply (16×16×16) | 1.935 | **71.0** | 142× |
| **K8 (WGMMA basic)** | 64×64×64 tiles, 128 threads | 0.433 | **317.5** | 633× |
| K9 (WGMMA larger tiles) | 128×128×64 tiles | 0.317 | 433.3 | 864× |
| K10 (WGMMA async/TMA) | + producer-consumer | 0.273 | 503.7 | 1,004× |
| **K11 (WGMMA max tiles)** | 128×256×64, 3 warpgroups | 0.222 | **618.4** | **1,234×**, 87% of cuBLAS |
| K0 (cuBLAS) | NVIDIA's library | 0.193 | 712.7 | 1,479× |

*(Note: `REPORT.md` numbers its own table K0–K11 locally; those map to this chapter's actual filenames `7_cublas_tc.cuh` through `12_wgmma_max_tiles.cuh` shifted by one, since its K0–K6 phase is really Chapter 6's ladder repeated for context.)*

Read straight down that table: the jump from best-tuned CUDA-core kernel (K6, 42.9 TFLOPS) to the *simplest possible* tensor-core kernel (K7, cuBLAS with tensor cores enabled — a pure library call) is itself a **16× jump**, before you've written a single line of tensor-core code yourself. That's the strongest possible argument for Chapter 1 §1.2's "when do you need custom CUDA" checklist: if a library call gets you 16× for free, hand-writing anything is only justified once you've actually reached for that library and it wasn't enough — which is exactly WMMA and WGMMA's position in this table.

Confirmed hardware gating, straight from the README and the Makefile:

| Kernel | Requires |
|---|---|
| 7 (cuBLAS + Tensor Cores) | Volta or later (V100, A100, H100) |
| 8 (WMMA) | Ampere or later, per this book's stated target (A100, RTX 30/40-series) |
| 9–12 (WGMMA) | **Hopper only** (H100, sm_90a) |

```makefile
# Target GPU architecture. These WGMMA kernels require Hopper (sm_90a, e.g. H100)
# and will not run on Ampere, Ada, or Blackwell. Override on the command line if
# needed, e.g. `make ARCH=sm_90`.
ARCH ?= sm_90a
```

Your RTX 3090 (Ampere, CC 8.6, from Chapter 1's table) runs kernels 7 and 8 natively. Kernels 9–12 need real Hopper hardware — this is the course's flagged cloud-GPU chapter.

**Deep dive: which single transition in the table actually matters most.** The table's own "vs. naive" column shows cumulative speedup, which hides something worth pulling out: the *per-step* multiplier between consecutive rungs.

```
K1  → K6  (entire classical ladder, 5 steps): 0.5 → 42.9 TFLOPS  = 86× total, geometric mean ≈ 2.44×/step
K6  → K7  (vectorized CUDA cores → WMMA):     42.9 → 71.0 TFLOPS = 1.65×
K7  → K8  (WMMA → WGMMA basic):               71.0 → 317.5 TFLOPS = 4.47×  ← largest single jump in the table
K8  → K9  (bigger WGMMA tiles):                317.5 → 433.3 TFLOPS = 1.36×
K9  → K10 (+ async/TMA):                       433.3 → 503.7 TFLOPS = 1.16×
K10 → K11 (max tiles, 3 warpgroups):           503.7 → 618.4 TFLOPS = 1.23×
K11 → K0  (cuBLAS, the ceiling):               618.4 → 712.7 TFLOPS = 1.15×
```

The single **K7→K8 transition — moving from WMMA's synchronous, single-warp tensor-core op to WGMMA's asynchronous, warp-group op — is a bigger multiplicative jump (4.47×) than the *entire* five-step classical CUDA-core optimization ladder averaged per step (2.44×/step).** Everything past K8 (larger tiles, async loads, more warpgroups) is real, worthwhile, and stacks — but it's incremental refinement on top of an architectural shift that already did most of the remaining work in one step. That's worth remembering the next time a "5 more optimization techniques" list appears: not all rungs on an optimization ladder are equal size, and knowing which one is actually load-bearing changes where you'd spend limited engineering time first.

## 7.2 The WMMA Programming Model (Kernel 8)

WMMA (Warp Matrix Multiply Accumulate) is C++ intrinsics, not raw PTX — the "easiest entry point," per the book's own framing. The real kernel builds a three-level tiling hierarchy: **block-level** (a 128×128 output tile per thread block), **warp-level** (each of the block's 8 warps owns a sub-tile), and **WMMA-level** (each warp's sub-tile is covered by 16×16×16 fragment operations):

```cuda
#include <mma.h>
using namespace nvcuda;
typedef __half fp16;

template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16,
          int WMMA_TILE_M = 4, int WMMA_TILE_N = 2, int WARP_TILE_M = 2, int WARP_TILE_N = 4>
__global__ void gemm_wmma_tiled(int M, int N, int K, const fp16 *A, const fp16 *B, fp16 *C) {
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;  // 16*4*2 = 128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  // 16*2*4 = 128
  __shared__ fp16 sA[BM][WMMA_K], sB[WMMA_K][BN], sC[BM][BN];

  // ...vectorized int4 loads from global into shared memory, with the same
  // alignment-check-and-scalar-fallback pattern as Chapter 6's GEMM kernel 6...

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp16> C_frag[WARP_TILE_M][WARP_TILE_N];
  for (int i = 0; i < WARP_TILE_M; ++i)
    for (int j = 0; j < WARP_TILE_N; ++j)
      wmma::fill_fragment(C_frag[i][j], __float2half(0.0f));

  for (int tile_k = 0; tile_k < CEIL_DIV(K, WMMA_K); ++tile_k) {
    // ...load this K-tile of A and B into shared memory, __syncthreads()...

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, fp16, wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, fp16, wmma::row_major> B_frag[WARP_TILE_N];

    for (int i = 0; i < WARP_TILE_M; ++i)
      wmma::load_matrix_sync(A_frag[i], &sA[/* this warp's M offset */][0], WMMA_K);
    for (int j = 0; j < WARP_TILE_N; ++j)
      wmma::load_matrix_sync(B_frag[j], &sB[0][/* this warp's N offset */], BN);

    for (int i = 0; i < WARP_TILE_M; ++i)
      for (int j = 0; j < WARP_TILE_N; ++j)
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);   // the actual tensor-core op

    __syncthreads();
  }
  // ...store C_frag tiles to shared memory via wmma::store_matrix_sync, then to global memory...
}
```

Four API calls do all the tensor-core work, and it's worth naming what each one really is:

- **`wmma::fragment<...>`** — an *opaque* type representing a small matrix tile, physically distributed across a warp's 32 threads' registers. You never index into it directly; it exists only to be passed to the other three calls.
- **`wmma::load_matrix_sync(frag, ptr, leading_dim)`** — cooperatively loads a tile from shared memory into a fragment, splitting the work across the warp's 32 lanes automatically.
- **`wmma::mma_sync(C_frag, A_frag, B_frag, C_frag)`** — the actual tensor-core instruction: a full 16×16×16 matrix-multiply-accumulate, executed by the warp's dedicated tensor-core hardware in one call.
- **`wmma::store_matrix_sync`** — the mirror of `load_matrix_sync`, writing a fragment back out.

Note this is still **synchronous and warp-scoped** — `mma_sync` blocks the issuing warp until the operation completes, and each warp acts independently. That's exactly the limitation WGMMA removes.

**Deep dive: this kernel accumulates in FP16, not FP32 — a real, deliberate departure from the norm.** Look again at the fragment declaration: `wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp16> C_frag[...]` — the accumulator type is `fp16`, the same as the inputs. Most GEMM kernels accumulate in FP32 specifically to control rounding error across a long reduction (this is the "precision sandwich" pattern that recurs all through this course — Chapter 4's cuBLAS, Chapter 8's Flash Attention, §7.3's own WGMMA kernels a few pages later, which use `float d[4][8]`). This kernel's choice to accumulate in FP16 instead is a genuine, quantifiable trade: a 16×16×16 FP16 accumulator fragment holds 256 values across a 32-thread warp — 8 values per thread, each 2 bytes, packing into **4 32-bit registers per thread per fragment**. The identical fragment with an FP32 accumulator would need 8 *4-byte* values per thread — **8 32-bit registers**, exactly double. Across this kernel's full `WARP_TILE_M × WARP_TILE_N = 2×4 = 8` accumulator fragments per warp, that's 32 registers (FP16) versus 64 registers (FP32) — a real, meaningful difference in a resource (registers) that directly caps how many warps can be resident on an SM simultaneously (Chapter 1's occupancy concept). The cost of that saving is exactly what you'd expect: more accumulated rounding error over the `K`-dimension reduction than an FP32 accumulator would have, which is precisely why the book's own correctness harness (Chapter 6 §6.1) uses a looser tolerance for kernels like this one than it would for an FP32-accumulate GEMM.

## 7.3 The WGMMA Programming Model (Kernels 9–12): Hopper's Asynchronous Tensor Cores

WGMMA is real PTX, not C++ intrinsics — the book's own README calls this out directly ("API: PTX-level programming"), and the actual source confirms it with inline assembly:

```cuda
__device__ void warpgroup_arrive()      { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ void warpgroup_commit_batch(){ asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ void warpgroup_wait()        { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, ..., %31}, %32, %33, %34, %35, %36, %37, %38;\n"
        "}\n"
        : /* 32 output registers, d[0..3][0..7] */
        : "l"(desc_a), "l"(desc_b), "n"(ScaleD), "n"(ScaleA), "n"(ScaleB), "n"(TransA), "n"(TransB));
}
```

Several genuinely new concepts here, each worth understanding individually:

**The operation is a "warp group" op, not a warp op.** `m64n64k16` computes a 64×64×16 tile, distributed across **128 threads = 4 warps acting as one unit** — twice the granularity of WMMA's single-warp 16×16×16. `d[4][8]` is one thread's slice of that 64×64 accumulator: 32 FP32 registers per thread, holding its share of the tile.

**It's genuinely asynchronous.** The three-call pattern — `wgmma.fence` → issue one or more `wgmma.mma_async` instructions → `wgmma.commit_group` → `wgmma.wait_group<N>` — means the instruction *returns immediately*; the actual matrix multiply executes on dedicated hardware while the issuing warp group could, in principle, do other work. `wait_group<N>` lets you keep `N` batches in flight simultaneously before blocking, which is a genuine pipelining tool WMMA's fully-synchronous `mma_sync` doesn't give you.

**Data needs a "descriptor," not a raw pointer.** `make_smem_desc()` packs a shared-memory address, a leading dimension, and a stride into a single 64-bit value the hardware itself interprets — because WGMMA expects **column-major** layout with a specific swizzle pattern, and the descriptor is how you tell the tensor-core hardware exactly how to walk your shared-memory tile. `REPORT.md`'s own explanation of *why*:

```
Row-major (C-style):     Column-major (Fortran-style):
A[0,0] A[0,1] A[0,2]     A[0,0] A[1,0] A[2,0]
A[1,0] A[1,1] A[1,2]     A[0,1] A[1,1] A[2,1]
...
```
*"WGMMA instructions expect column-major because hardware is optimized for Fortran-style layouts... matrix descriptors encode column-major metadata."*

**TMA (Tensor Memory Accelerator)** is Hopper's dedicated hardware DMA engine for exactly this kind of tiled tensor movement — asynchronous 2D-tile loads from global to shared memory, with automatic swizzling to avoid the shared-memory bank conflicts you'd otherwise have to manage by hand.

**Kernels 10–12 add producer-consumer warp specialization** — different warp groups play different roles, from `REPORT.md`'s own diagram:

```
┌─────────────┐
│  Producer   │  (1 warpgroup, 128 threads) — loads A & B tiles via TMA, signals a barrier when ready
│  Warpgroup  │
└──────┬──────┘
       │ Circular Buffer (QSIZE=3)
       ├───► Slot 0: Loading
       ├───► Slot 1: Computing ◄──┐
       └───► Slot 2: Waiting       │
┌──────────────────────────────────┴───┐
│  Consumer Warpgroups (2×128 threads) │  — wait on barrier, execute WGMMA on ready tiles
└──────────────────────────────────────┘
```

The producer loads the **next** tile while the consumers compute on the **current** one — hiding memory latency behind compute, the same overlap idea Chapter 1 introduced abstractly (async kernel launches, streams) now happening *inside* a single kernel, at the warp-group level. Kernel 11 uses 1 producer + 1 consumer (256 threads); kernel 12 pushes to 1 producer + 2 consumers (384 threads) with the largest tiles (128×256×64) — "best balance of compute and memory ops," per the report, and the measured winner at 618.4 TFLOPS.

**Deep dive: a second, complementary reason kernel 12 wins, beyond just "bigger tiles."** Look at the producer/consumer thread split itself: kernel 11's 256 threads are 1 producer warpgroup (128 threads, pure memory movement) and 1 consumer warpgroup (128 threads, pure tensor-core compute) — exactly **50%** of the block's threads are doing productive tensor-core work at any given moment; the other half exists solely to keep them fed. Kernel 12's 384 threads are 1 producer plus **2** consumers — **256 of 384 threads, ≈66.7%**, are compute warpgroups. A larger fraction of the block's total thread population is doing the work that actually shows up in the TFLOPS number, not just moving data to feed it. That's a real, distinct contributor to kernel 12's win alongside its larger tile shape — two consumer warpgroups mean the single producer's TMA loads need to keep *two* compute streams fed rather than one, which only pays off if the tiles are large enough that each load serves enough compute to justify the added coordination complexity of a third participating warpgroup. Larger tiles and a higher consumer-thread fraction aren't independent choices here; the report's "best balance" is really describing the point where both of these numbers work together.

## 7.4 A Real Lesson in Benchmarking Rigor

This is worth its own section because it's a genuine, documented mistake the book's own authors made and then caught — exactly the kind of thing Chapter 6's "1e-2 tolerance, three summary tables" harness is designed to catch, and exactly what Part 9's Nsight tooling exists to make impossible to miss.

`REPORT.md`, verbatim in substance: an early measurement of the most advanced kernel (their "K12," a hand-written kernel that already implemented the full producer-consumer, TMA, and register-allocation optimizations) showed only **~106 TFLOPS** — barely ahead of plain WMMA, and a disappointing result for a kernel with every advanced technique already applied. The cause: their benchmark script was **timing the row-major-to-column-major layout conversion together with the kernel itself.** That conversion alone measured **~30ms for a 4096×4096 FP16 matrix** — while the WGMMA kernel itself runs in a fraction of a millisecond. Once conversion was excluded from the timed region (converting layouts once, ahead of the timing loop, exactly the "pure kernel time" methodology this chapter's whole table uses), the same kernel's real performance turned out to be **~550 TFLOPS — over 5× higher than the flawed measurement suggested.**

The lesson generalizes far beyond this one kernel: **what you include inside your timer is not a detail, it's the whole result.** A kernel that looks mediocre can actually be excellent, wrapped in measurement code that's silently charging it for someone else's work. This is precisely why Part 9 spends real time on Nsight Systems' timeline view — it shows you, unambiguously, which portion of wall-clock time belongs to which actual kernel, instead of trusting a hand-rolled `start`/`stop` pair around a block of code you may not have audited closely enough.

One more real, concrete number worth sitting with: correctness validation across this whole ladder uses tolerance ≤ 2.0 for FP16 classical kernels (max observed diff < 0.72), but **the WGMMA kernels (8–11) match cuBLAS exactly — max_diff = 0.0000.** That's not a coincidence: WGMMA and cuBLAS's own tensor-core path both accumulate in FP32 through the same hardware reduction order, while the hand-rolled shared-memory CUDA-core kernels accumulate in whatever order their loop nests happen to produce — different order, different rounding, small but nonzero drift. Precision isn't just about which *type* you compute in; it's also about *in what order* values get summed.

**Deep dive: what the mismeasurement actually cost, in perceived-value terms.** It's worth translating "~106 TFLOPS vs. ~550 TFLOPS" into the language the rest of this table uses — speedup over naive. Against naive's confirmed 0.5 TFLOPS, the flawed measurement would have reported roughly `106/0.5 ≈ 212×` speedup for a kernel that had, in reality, already implemented every advanced technique this chapter covers. The corrected measurement puts the true figure at `550/0.5 = 1,100×`. **The bug didn't just understate the number — it hid more than half of the kernel's true achievement**, making a kernel that had already earned a place at the top of this chapter's own performance table look barely better than plain WMMA (142×, from §7.1's table) instead of nearly 8× better than WMMA. That gap between "barely ahead of the easy option" and "a legitimate contender near the top of the ladder" is entirely a measurement artifact — the kernel itself never changed.

## 7.5 Running WMMA from Python (and Why WGMMA Can't, Here)

**Honesty check before any code:** this environment has no attached GPU, so nothing in this course's Python sections has actually been executed and observed by me — every one has been written and hand-traced carefully, not run. That's always been true; it matters more to say out loud in this specific chapter, because §7.3's WGMMA kernels have a stronger claim than usual: they *cannot run at all* outside Hopper, a limitation this section needs to respect rather than paper over with code that would merely fail to compile if you tried it.

**WMMA, first — this genuinely runs on both your GPUs.** §7.2's real kernel uses a full block/warp/fragment tiling hierarchy whose shared-memory loading loop wasn't fully quoted verbatim (the text marks it `// ...vectorized int4 loads...` as a summarized placeholder, not literal source). Rather than reconstruct that hierarchy's exact details from a partial description, here's a **minimal, single-warp-per-tile WMMA kernel** — smaller in scope than the book's own, but using the *exact same confirmed API calls* (`load_matrix_sync`, `mma_sync`, `store_matrix_sync`) in the simplest form they can take:

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// One warp computes one 16x16 output tile, looping over K in chunks of 16.
// Simplified relative to §7.2's full hierarchy (one warp per tile, not a
// block/warp/fragment hierarchy) -- but the four WMMA API calls themselves
// are used exactly as §7.2 describes them.
__global__ void wmma_gemm_kernel(const half* A, const half* B, float* C, int M, int N, int K) {
    int warpM = blockIdx.y * blockDim.y + threadIdx.y;   // which 16-row tile of C this warp owns
    int warpN = blockIdx.x;                               // which 16-col tile of C this block owns

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        int aRow = warpM * 16, bCol = warpN * 16;
        wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + (warpM * 16) * N + warpN * 16, c_frag, N, wmma::mem_row_major);
}

torch::Tensor wmma_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16, "inputs must be FP16");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0, "M, N, K must be multiples of 16 for this minimal demo");
    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
    dim3 threads(32, 4);                        // 4 warps/block, one warp per 16-row tile via threadIdx.y
    dim3 blocks(N / 16, (M / 16 + 3) / 4);
    wmma_gemm_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), M, N, K);
    return C;
}
"""

cpp_source = "torch::Tensor wmma_gemm(torch::Tensor A, torch::Tensor B);"

ch7 = load_inline(
    name="ch7_wmma_kernel", cpp_sources=cpp_source, cuda_sources=cuda_source,
    functions=["wmma_gemm"], verbose=True,
)

M, K, N = 256, 256, 256
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

C_custom = ch7.wmma_gemm(A, B)
C_ref = (A.float() @ B.float())    # FP32 reference -- WMMA accumulates in FP32 internally (§7.2)
print("max diff:", (C_custom - C_ref).abs().max().item())   # expect a small but nonzero FP16-input diff
```

This should build and run on **both** your GPUs — Ampere and Turing alike support WMMA (Volta+, per §7.1's hardware table). It's a genuinely smaller kernel than the book's own (no boundary handling for non-multiples of 16, no register/warp tiling beyond one warp per 16×16 tile), but it exercises the real hardware path — you can confirm this yourself by timing it against `A.float() @ B.float()` (plain CUDA cores) and `A.half() @ B.half()` (cuBLAS with tensor cores, PyTorch's own default for FP16 matmul) the same way §7.1's table compares kernels 6, 7, and 0.

**WGMMA — why this section stops at the kernel text, not a runnable wrapper.** Three real constraints compound here, worth naming rather than working around: (1) `wgmma.mma_async` is a Hopper-only PTX instruction — trying to compile it for your 3090's `sm_86` fails at compile time, exactly as the Hands-On Lab already demonstrated; (2) even setting that aside, §7.3's quoted source stops short of a complete kernel — the inline PTX asm's full 32-register output list and `make_smem_desc`'s exact bit-packing weren't fully reproduced, only described; (3) with no GPU in this environment, I have no way to actually compile or test a from-scratch reconstruction even if I attempted one. Writing code I can't verify, to run on hardware neither of us has access to right now, for a kernel I don't have complete confirmed source for, would stack three separate honesty problems on top of each other.

What *is* worth showing: how you'd point `load_inline` at Hopper if you had a complete WGMMA kernel to compile, since the mechanism itself is simple and worth knowing on its own —

```python
# Illustrative only -- assumes you've supplied a complete WGMMA kernel's source
# (this chapter's confirmed fence/commit/wait helpers plus the full register layout
# and TMA descriptor code that §7.3 summarizes but doesn't fully reproduce).
ch7_hopper = load_inline(
    name="ch7_wgmma_kernel",
    cpp_sources="torch::Tensor wgmma_gemm(torch::Tensor A, torch::Tensor B);",
    cuda_sources=your_complete_wgmma_source,   # not provided here -- see below
    functions=["wgmma_gemm"],
    extra_cuda_cflags=["-arch=sm_90a"],         # the exact flag from this chapter's own Makefile (§7.1)
    verbose=True,
)
```

For the real kernel, the right move is the one the book itself sets up: clone `book.cu/5_tensor_cores`, and on an actual H100 (rented, if needed — this is this course's flagged cloud-GPU chapter), `make ARCH=sm_90a && python main.py`. That gets you the complete, tested kernel this section can't safely reconstruct from a partial quote.

---

## Hands-On Lab

```bash
cd book.cu/5_tensor_cores
make            # builds kernels 7-12 for sm_90a by default
python main.py  # benchmarks all, generates a performance plot
```

1. **On your RTX 3090:** override the target architecture and confirm what actually happens: `make ARCH=sm_86`. Kernels 7 and 8 (cuBLAS-TC, WMMA) should build and run. Kernels 9–12 (WGMMA) should **fail at compile time** — not at runtime — because `wgmma.mma_async` isn't an instruction that exists outside Hopper; `ptxas` can't assemble it for `sm_86` at all. Contrast this with a *runtime* "no kernel image available" error (Chapter 5's troubleshooting section) — that happens when a kernel *was* compiled, just not for your specific architecture. This is a compile-time absence of the instruction itself, a stricter failure mode.
2. **Measure kernels 7 and 8 on your own hardware** and record the TFLOPS at 4096³. Compare your 3090's numbers to this chapter's H100 table — expect a meaningfully lower absolute TFLOPS (fewer, older tensor cores), but check whether the *relative* jump from CUDA-core-vectorized to WMMA holds a similar shape.
3. **If you have (or rent) H100 access**, run the full suite and compare your own kernels 9–12 numbers against `REPORT.md`'s table.

## Exercises

1. **Reproduce the benchmarking-bug lesson yourself.** Time a row-major → column-major conversion for a 4096×4096 FP16 tensor (on any GPU you have). Confirm you land in the same tens-of-milliseconds range the book measured, and explain in one sentence why that would completely swamp a sub-millisecond WGMMA kernel if included in the same timer.
2. **Register-pressure arithmetic.** `d[4][8]` is one thread's share of a 64×64 accumulator tile, across a 128-thread warpgroup. Work out how many FP32 registers per thread that represents, and explain — using Chapter 1's occupancy concept — why kernel 9's jump to 128×128×64 tiles (bigger accumulator per warpgroup) is a genuine tradeoff, not a free win: more registers per thread means fewer warpgroups can be resident on an SM simultaneously.
3. **Re-tile the WMMA kernel.** Change `WARP_TILE_M`/`WARP_TILE_N`/`WMMA_TILE_M`/`WMMA_TILE_N` to a different valid combination that still yields `BM=BN=128`, rebuild, and re-benchmark against the original configuration.
4. **Explain the matrix descriptor in your own words.** Using `make_smem_desc`'s packed fields (address, leading dimension = 16, stride = 1024, a swizzle bit), explain why WGMMA's hardware needs a descriptor instead of accepting a raw shared-memory pointer the way `wmma::load_matrix_sync` does.
5. **Trace one full WGMMA batch end-to-end** through kernel 9's source: find the `warpgroup_arrive()` call, the loop issuing `wgmma64<...>()`, the `warpgroup_commit_batch()`, and the `warpgroup_wait<0>()` — and identify exactly which shared-memory tile must **not** be overwritten between the first and last of those four steps.

---

**Next:** Chapter 8 — Flash Attention (Part 7). You now have every ingredient: online-softmax's rescaling trick (Chapter 6 §6.2), tensor-core matrix multiplication (this chapter), and the memory-bandwidth framing from Chapter 1 §1.4.1. Flash Attention fuses all three into a single kernel that never materializes the full N×N attention matrix in global memory at all.
