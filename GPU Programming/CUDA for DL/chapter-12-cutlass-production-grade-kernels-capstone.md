# Chapter 12 — CUTLASS & Production-Grade Kernels (Capstone)

*Part 11: CUTLASS & Production-Grade Kernels. Confirmed from the book's companion repo: `book.cu/9_cutlass/README.md`, "CUTLASS Examples from Chapter 11 of the book" — the same +1 numbering drift flagged since Chapter 1 (this folder self-labels "Chapter 11," one less than this course's Chapter 12).*

Real hardware note, stated directly in the book's own README and worth repeating verbatim in spirit: **no single GPU runs all three of this chapter's examples.** `unofficial/` and `official/` need Hopper (`sm_90a`); `fp4/` needs Blackwell (`sm_100a`). Treat this chapter as a hardware-specific tour rather than something to run end-to-end on one machine — with one genuine exception worth knowing about, found in the repo's own internal engineering notes: `official/official_ampere_gemm/gemm_sm80_official.cu` is explicitly documented as **"Compatible with A100, RTX 3090"** — meaning your own 3090 can actually run one real piece of this capstone, even though the top-level README's blanket "Hopper only" framing for `official/` would suggest otherwise. That kind of documentation-vs-reality gap has shown up before in this course (Chapter 4's swapped file labels, Chapter 9's file-naming drift) — here it works in your favor.

---

## 12.1 The Question This Whole Course Has Been Building Toward

Chapter 1 §1.2 asked you to judge, before writing a single kernel, whether custom CUDA was worth its cost against an existing library. This chapter answers that question with real, measured numbers — `official/docs/COMPARISON.md`'s own benchmark, run on an actual H100 at an 8192×8192×8192 FP16 GEMM (FP32 accumulator):

| Implementation | GFLOPS | vs. cuBLAS | vs. From-Scratch |
|---|---|---|---|
| From-Scratch Ampere (SM80) | 429,000 | 0.59× | 1.00× (baseline) |
| Official Ampere (SM80) | ~465,000 | 0.71× | **8.5× faster** |
| From-Scratch Hopper (SM90) | 511,000 | 0.67× | 1.00× (baseline) |
| Official Hopper (SM90) | ~547,000 | 0.82× | **8.6× faster** |
| PyTorch cuBLAS (reference) | 726,000 | 1.00× | 12× faster than from-scratch |

A hand-written CUTLASS kernel with **correct API usage** — right types, right layouts, a real `CollectiveBuilder` pipeline — still loses to NVIDIA's own official example by **8.5–8.6×**, and the official example itself still trails cuBLAS by roughly 30%. That's this course's Chapter 1 §1.2 checklist, empirically resolved: correctness isn't the same as performance, and neither hand-written CUDA nor "using CUTLASS" alone gets you to peak — every layer, all the way up to a vendor's own decades-tuned library, still has headroom above it.

## 12.2 What the Hand-Written Version Gets Right

`unofficial/single_gpu_gemm.cu`'s own docstring states its scope precisely: a single-GPU FP16 GEMM (FP32 accumulator) targeting Hopper, built entirely from CUTLASS 3.x's template-composition API rather than a hand-rolled kernel body:

```cuda
using ArchTag = cutlass::arch::Sm90;
using ElementInput = cutlass::half_t;        // FP16 inputs — tensor-core eligible
using ElementAccumulator = float;             // FP32 accumulation — the precision-sandwich pattern, again
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;

using TileShape = Shape<_128, _256, _64>;     // thread-block tile: 128 rows x 256 cols x 64 K-depth
using ClusterShape = Shape<_2, _1, _1>;       // 2 thread blocks cooperating as one cluster (Hopper-only)
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    ElementInput, LayoutA, AlignmentA, ElementInput, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

This is the conceptual leap this whole capstone is built around: instead of writing a kernel body — the raw `wgmma.mma_async` inline assembly Chapter 7 §7.3 walked through by hand, with its own explicit fence/commit/wait sequencing — you **declare a specification** (data types, layouts, tile shape, cluster shape, scheduling policy) via template parameters, and `CollectiveBuilder`'s own metaprogramming assembles a working, hardware-optimized kernel matching that spec at *compile time*. It's Chapter 7's hand-rolled WGMMA pattern, generated correctly and consistently by a library instead of typed out by hand every time.

**A transparency note worth being direct about:** this file's *actual, current* configuration (`TileShape<_128,_256,_64>`, `ClusterShape<_2,_1,_1>`) already matches `COMPARISON.md`'s description of the **optimized/official** Hopper configuration, not the document's own "from-scratch" narrative (which describes `Shape<_128,_128,_64>` with no clusters, `Shape<_1,_1,_1>`). The most likely explanation is exactly what you'd expect from a repo maintained over an 8-month MEAP: the hand-written example was refined *after* that comparison document was written, and the two simply drifted out of sync — the same kind of documentation lag you've now seen in several chapters. The code above is what the file genuinely contains today; treat `COMPARISON.md`'s numbers and checklist as real and valuable in their own right, just not a byte-for-byte diff of this exact file's current state.

## 12.3 What Actually Moves the Needle

`COMPARISON.md`'s own optimization analysis, condensed to its real substance:

- **Rectangular tiles (128×256) beat square tiles (128×128).** Better memory coalescing along the N dimension, and a larger accumulator amortizes load/store overhead per useful FLOP computed.
- **Smaller K-per-stage (32 vs. 64) means more frequent outer-product updates**, which reduces register spilling — a direct echo of Chapter 6 §6.1's register-blocking rationale, tuned one level higher.
- **Thread Block Clusters** (`Shape<_2,_1,_1>`) are a genuinely new concept this course hasn't covered yet, distinct from anything in Chapter 7: instead of one thread block loading its own tile independently, **2 cooperating thread blocks share data via TMA multicast** — one load from global memory feeds both blocks, cutting memory traffic for that data roughly in half. `COMPARISON.md`'s own measured impact: **~15–20% for memory-bound cases.** Where WGMMA (Chapter 7) is about how *one* warp group computes, clusters are about how *multiple thread blocks* cooperate — a coordination layer above anything single-block kernels can express.

`COMPARISON.md`'s own checklist for closing the from-scratch-to-official gap, kept in the source's own words:

> **Ampere:** rectangular tiles, tune K (32/64/128), explicit 3–5 pipeline stages, tune warp shape, profile with `ncu`.
> **Hopper:** enable clusters (`Shape<_2,_1,_1>`), larger tiles (128×256+), let `KernelScheduleAuto` pick a warp-specialized variant, verify TMA usage via the `ncu` metric `lts__t_sectors_srcunit_tex`.

That last item is worth pausing on: it's Chapter 10's profiling discipline, applied to verify a *specific hardware feature* (TMA) is actually active — not just that the kernel runs, but that it's using the mechanism you intended.

## 12.4 Why Even "Official" Still Trails cuBLAS

The remaining ~30% gap between official CUTLASS examples and cuBLAS isn't a quality problem with CUTLASS itself — `COMPARISON.md` is direct about why: **"CUTLASS examples demonstrate features, not peak performance."** cuBLAS carries years of auto-tuning across an enormous space of shapes, plus heuristics for selecting among many pre-tuned kernel variants per problem size. Closing that last gap in CUTLASS itself means reaching for the **CUTLASS Profiler** — a separate tool for auto-tuning tile/cluster/stage configurations against your own specific problem-size distribution, rather than hand-guessing template parameters. The lesson recurses one more level than you might expect: from-scratch code loses to a library's example code, and a library's example code loses to that same library's *auto-tuning tool*. Each rung is still "know when to stop hand-tuning and reach for something built to search the space for you" — Chapter 1 §1.2's question, one more time, at the top of the stack.

## 12.5 The Blackwell Capstone: nvFP4 Block-Scaled GEMM

The book's final example isolates CUTLASS's own example 72a — a GEMM with **FP4 inputs and BF16 outputs**, on Blackwell's SM100 architecture:

```
Input A:     FP4 (row-major) + per-block scale factors
Input B:     FP4 (column-major) + per-block scale factors
Output C/D:  BF16 (row-major)
Accumulator: FP32
Operation:   D = alpha * (A @ B) + beta * C
```

**"Block-scaled"** is not a new idea at this point in the course — it's Chapter 9 §9.4's block-wise quantization granularity scheme, the same one Chapter 9 Exercise 3 asked you to combine with INT4 packing by hand. Here, that exact idea — different scale factors for different sub-blocks of a tensor, so quantization error doesn't get forced through a single global scale — is implemented as **native hardware support in Blackwell's tensor cores**, not something you build yourself in a kernel. The README states the payoff directly: block-scaled FP4 delivers **2× the throughput of FP8**, and **4× the throughput of FP8 on Hopper.**

The real, measured result, from the book's own benchmark run:

> **Average kernel time: 1.93 ms. Performance: 4.55 PFLOPS (4,553 TFLOPS). Problem size: 8192×8192×8192, batch=8.**

**This is worth putting next to Chapter 3's very first measured GEMM number.** Chapter 3 §3.3.2 reported naive GEMM running at **10–50 GFLOPS**. This chapter's FP4 Blackwell result — **4,553,000 GFLOPS** — is roughly **90,000× to 450,000×** faster than where this entire course started. That factor is the whole optimization stack from Chapter 1 §1.6.2, compounding exactly the way §1.6.3 promised it would: coalescing, shared-memory tiling, register blocking, vectorization, warp-level primitives, tensor cores, clusters, and finally a native hardware format purpose-built for exactly this kind of low-precision, high-throughput matrix multiplication.

And the correctness discipline hasn't changed at all from Chapter 3's very first CPU-first kernel: `verify.sh` runs a smaller (1024×1024×1024) case, computes both the GPU result and a CPU reference, and checks agreement within an FP4-appropriate tolerance before the benchmark script is trusted at all — the same "verify, then measure" order this entire course has followed in every single chapter.

## 12.6 Closing the Loop

Chapter 1 §1.6.2 laid out an abstract six-layer optimization stack before you'd written a single kernel. Looking back, every layer got a real, measured chapter of its own:

| Layer (Ch. 1's framing) | Where this course demonstrated it |
|---|---|
| Memory coalescing | Chapter 6, GEMM kernel 2 |
| Shared-memory reuse | Chapter 6, GEMM kernel 3 |
| Register tiling & vectorization | Chapter 6, GEMM kernels 4–6 |
| Warp-level primitives | Chapter 6's softmax/GEMV/top-k warp kernels |
| Tensor cores | Chapter 7 (WMMA, WGMMA) |
| Library/template level | This chapter (CUTLASS, and cuBLAS as the ceiling above even CUTLASS) |

Two other threads ran underneath that stack, recurring by accident of real code rather than deliberate course design — which is exactly why they were worth following: the **single-thread-per-row anti-pattern** (Chapter 4's softmax, Chapter 5's GEMV and top-k, Chapter 8's naive attention) and its real fixes; and the **precision-sandwich pattern** (accumulate in higher precision than you multiply in — Chapter 4's cuBLAS MNIST, Chapter 7's WGMMA, Chapter 8's Flash Attention, Chapter 9's quantization, and this chapter's FP4 GEMM). Neither was planned as a "theme" going in — both simply kept appearing in the book's own real, shipped code, chapter after chapter, which is a better argument for their importance than any amount of abstract framing could have been.

---

## Hands-On Lab

1. **Run the one piece of this capstone your own hardware actually supports.** Clone CUTLASS and attempt to build `official/official_ampere_gemm/gemm_sm80_official.cu` for your RTX 3090 (`sm_86`, part of the Ampere family this file targets). You'll need CUTLASS's headers on your include path and the correct `-arch`/`-gencode` flags for `sm_86`.
2. **Diff the two Hopper configurations directly**, if you have access to both files: `unofficial/single_gpu_gemm.cu` and `official/official_hopper_gemm/gemm_sm90_official.cu`. Compare their `TileShape`, `ClusterShape`, and scheduling template parameters side by side.
3. **If you have Hopper cloud access:** build and run both the unofficial and official Hopper examples, reproduce something close to `COMPARISON.md`'s ~8.6× gap yourself, then run `ncu` with the `lts__t_sectors_srcunit_tex` metric mentioned in its own checklist to confirm TMA is genuinely active in the official kernel.
4. **If you have Blackwell cloud access:** run `fp4/build.sh`, `fp4/verify.sh`, and `fp4/benchmark.sh` in sequence, and see how close your own measured PFLOPS comes to the book's reported 4.55 PFLOPS at the same problem size.

## Exercises

1. **Prioritize the optimization checklist.** Using `COMPARISON.md`'s own checklist from §12.3, pick the three changes you'd make *first* to close the from-scratch-vs-official gap on Ampere, and justify your ordering using this course's own arithmetic-intensity framework (Chapter 1 §1.4.1).
2. **Explain Thread Block Clusters in your own words.** What does `Shape<_2,_1,_1>` actually do differently from `Shape<_1,_1,_1>`? Why is this a genuinely new concept relative to Chapter 7's WMMA/WGMMA coverage, rather than just "WGMMA with extra steps"?
3. **Compute the full course-spanning speedup yourself.** Using Chapter 3's stated naive-GEMM range (10–50 GFLOPS) and this chapter's 4,553 TFLOPS figure, compute the exact ratio, and give your own reasoned estimate of which single layer from §12.6's table contributed the largest multiplicative factor to that gap.
4. **Connect block-scaled FP4 back to Chapter 9.** Explain how this chapter's per-block FP4 scale factors are the natural hardware-native conclusion of the exact idea Chapter 9 Exercise 3 asked you to build by hand (combining group/block granularity with sub-byte packing).
5. **Describe what a profiler-driven auto-tuner needs to search.** Without needing to run it yourself, describe — using everything you now know about tile shapes, cluster shapes, pipeline stages, and kernel scheduling — what the CUTLASS Profiler mentioned in §12.4 would need to search over to find a configuration that closes CUTLASS's remaining gap to cuBLAS for a specific problem size.

---

## Course Complete

That's the full arc: from `threadIdx.x` on an 8-element vector add in Chapter 2, through hand-rolled backpropagation, a transformer wired into PyTorch's own autograd, the complete GEMM optimization ladder, tensor cores, Flash Attention, quantization, real profiling tools, multi-GPU scaling, and finally a 4.55-PFLOPS Blackwell kernel built from a template library rather than a single line of hand-written kernel code. Every chapter's core claims were checked against the book's real, official companion source rather than reconstructed from memory — including the places where that source contradicted itself, which turned out to be some of the most useful material in the whole course.

Whenever you're ready, this is a natural point to fold back into your own GitHub learning hub alongside the Triton and GPU-architecture material already there — CUTLASS's CuTe layout algebra in particular would make a strong next stop, since it's the piece this chapter deliberately treated as a black box (`Shape<...>`, `CollectiveBuilder`) rather than opening up.
