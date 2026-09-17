# Chapter 26 — Cross-Platform Portability

## 26.1 Systematizing What's Been Scattered

Portability seams have surfaced repeatedly in this curriculum without ever being addressed as their own subject: Chapter 13's `libdevice`/`ocml`+`ockl` backend split, Chapter 15, §15.6's AMD general-purpose-vs-accumulation-VGPR nuance, Chapter 16's `amd_mfma`/`amd_rotating_shared` layout kinds, Chapter 19's brief CDNA aside. This chapter treats portability systematically rather than as a recurring footnote — and the single fact underlying most of what follows is worth understanding deeply before anything else.

## 26.2 The Single Most Consequential Hardware Difference: Warp Size

(cite index="24-1">NVIDIA executes threads in warps of 32; AMD's datacenter GPUs — the CDNA/Instinct series, including the MI300X — execute wavefronts of 64</cite>. This is, in a genuinely direct quote from a comparative analysis of the two platforms, (cite index="24-1">"the most-quoted hardware difference"</cite> between them, and for good reason: it's the granularity at which the hardware schedules SIMT execution, and it touches nearly everything downstream — reductions, divergence cost, occupancy.

Concretely: (cite index="22-1">an MI300X has 304 Compute Units, 4 SIMD units per CU, and a wavefront size of 64</cite>, queryable directly via `rocminfo`. Just as Chapter 7 and Chapter 19 taught you to query `NUM_SM` rather than hardcode an assumed SM count, portable code should query warp/wavefront size rather than assume 32 — (cite index="23-1">portable code should use the `warpSize` built-in to query it directly, since hipified code carried over from a CUDA-assuming codebase requires careful review to ensure it doesn't implicitly assume a wave size of 32</cite>.

**The consequence of getting this wrong is large and quantified, not a vague caveat**: (cite index="28-1">if a kernel's control flow, synchronization, or memory coalescing is implicitly tuned for warp32, running it unchanged on wave64 hardware can strand lanes or force the compiler to add masking/shuffles — the "same" kernel can end up roughly 2x off peak performance purely from the execution-width mismatch, before even considering differences in cache behavior or matrix-math units</cite>. This is worth sitting with: a kernel that's correct and well-tuned on NVIDIA hardware can lose *half its performance* on AMD hardware for a reason that has nothing to do with the algorithm itself — purely a mismatch between an implicit width assumption and the actual hardware.

**And this connects directly and precisely to Chapter 6's `tl.where` discussion.** Recall that per-lane divergent branches force the hardware to execute both paths, masking off inactive lanes each time. (cite index="24-1">Divergence on NVIDIA wastes lanes in groups of 32; the same divergence can idle up to 64 lanes on AMD's datacenter GPUs</cite>. The predication cost Chapter 6 described — "both branches always evaluated" — isn't just conceptually present on both platforms; it's **literally twice as expensive per divergence event** on wave64 hardware, in terms of lanes idled per masked branch. A `tl.where` pattern that was a negligible cost on NVIDIA hardware may be a meaningfully larger one on AMD's datacenter accelerators, for a precisely quantifiable reason.

## 26.3 A Terminology Map for Reading AMD Documentation

Your NVIDIA-trained vocabulary maps onto AMD's terminology fairly directly, once you know the pairing:

| NVIDIA term | AMD term | Where you've already met the NVIDIA side |
|---|---|---|
| SM (Streaming Multiprocessor) | CU (Compute Unit) | Chapters 7, 15, 19's `NUM_SM`-based occupancy/persistent-kernel calculations |
| Shared memory | LDS (Local Data Share) | Chapter 15's on-chip memory discussion |
| Registers | VGPRs (Vector General-Purpose Registers) | Chapter 15, §15.6's AMD register-pool nuance |
| Tensor cores (`wmma`/`mma`/`wgmma`/`tcgen05`) | Matrix cores (MFMA instructions) | Chapter 10, Chapter 16's `amd_mfma` layout |

Two things are worth being precise about here. First: the **concept** of a memory hierarchy (Chapter 15) transfers directly — LDS occupies the same conceptual tier shared memory does. Second: the **specific capacities do not transfer** — (cite index="26-1">LDS capacity is 64 KB on CDNA2 and 128 KB on CDNA3</cite>, numbers that don't match NVIDIA's per-SM shared memory budgets and that also differ *between AMD generations themselves*. Chapter 16's shared-memory-exhaustion failure mode (a layout conversion that fit at one tile size and failed to compile at a larger one) is exactly the kind of thing that needs re-checking per target platform *and* per target generation — never assumed to port unchanged just because "it's still AMD" or "it's still NVIDIA."

## 26.4 Matrix-Core Tuning Is Genuinely a Different Search Space

Here's a concrete, counter-intuitive illustration of why Chapter 8's autotuning discipline matters even more across platforms than within one: AMD's MFMA instructions come in different tile-shape variants, selected via a `matrix_instr_nonkdim` parameter (`16` selects `mfma_16x16`, `32` selects `mfma_32x32`). The empirical finding, direct from AMD's own tuning guidance: (cite index="22-1">for GEMM kernels on an MI300X, `mfma_16x16` typically outperforms `mfma_32x32`, even for large tile/GEMM sizes</cite>. If your NVIDIA-trained intuition says "bigger MMA tiles are usually better" (a reasonable prior from Chapter 9-10's tensor-core discussion), this is a direct counter-example — **the autotuning search space itself, not just the winning configuration within it, may need to differ by platform.** Porting a config list discovered on NVIDIA hardware to AMD isn't just "re-benchmark the same candidates" — it may mean adding genuinely different tunable dimensions AMD's architecture exposes that NVIDIA's doesn't.

## 26.5 `OPTIMIZE_EPILOGUE`: Chapter 16's Lesson, as a Real, One-Line AMD Knob

This is a genuinely satisfying, concrete instance of an abstract concept from earlier in this curriculum becoming a literal, documented tuning decision. Recall Chapter 16's discussion of `ttg.convert_layout` cost — an MMA-layout tensor sometimes needs to be converted to a different layout before it can be efficiently stored, and that conversion isn't free. AMD's Triton backend exposes exactly this trade-off as a tunable environment variable: (cite index="22-1">by default, MFMA instruction results are converted to a blocked layout — via an LDS-mediated data exchange between threads — specifically to enable maximum-width global stores (`global_store_dwordx4`); setting `OPTIMIZE_EPILOGUE=1` skips this conversion, storing results directly in the MFMA layout instead, at the cost of reduced global-store efficiency, though the impact on overall kernel execution time is usually minimal — AMD's own guidance recommends turning it on in most cases</cite>. This is Chapter 16's abstract "sometimes a layout conversion costs more than it's worth, and a targeted optimization pass or manual override can eliminate it" made completely concrete: a single environment variable, backed by exactly the layout mechanics Chapter 16 taught you to recognize.

## 26.6 Vendor-Specific Autotuning Dimensions Beyond `num_warps`/`num_stages`

Chapter 8 taught you `num_warps` and `num_stages` as platform-agnostic autotuning dimensions. In practice, each vendor exposes additional, genuinely platform-specific dimensions worth adding to your search space when targeting that platform specifically: (cite index="29-1">AMD GPUs expose `waves_per_eu`; newer NVIDIA generations (Hopper and Blackwell) expose `num_consumer_groups` and `num_buffers_warp_spec` — parameters specifically tied to the warp-specialization concepts from Chapter 25</cite>. Early experimentation with these parameters has shown meaningful improvement in some kernels. The practical consequence for your own autotuning `Config` lists (Chapter 8): a config space that's complete for one platform is very likely *incomplete* for another, not merely mistuned — matching Chapter 8's own principle that `key` and the search space should reflect what genuinely affects performance, now extended across a vendor boundary.

## 26.7 A Practical, Honest Caveat: Graphs vs. Runtime Adaptability

Worth a brief, honest mention given how directly it bears on production LLM-serving work: (cite index="29-1">using CUDA or HIP graphs — which capture and replay a fixed kernel/grid configuration specifically to reduce launch overhead — limits the flexibility to adapt kernel block size and launch grid to the actual runtime data (for instance, the real number of tokens in a batch)</cite>. This is a genuine, current tension between two things this curriculum has taught you to value: the launch-overhead reduction from Chapter 19's persistent-kernel/graph-style thinking, and the ability to adapt a kernel's configuration to real, data-dependent workload shape. Neither choice is universally correct — it's a real trade-off to make deliberately, informed by your specific deployment's actual variability in input shapes.

## 26.8 A Practical Portability Checklist

- **Never hardcode warp size as 32.** Query it explicitly, exactly as you already query `NUM_SM`/register-file size (Chapter 7, §7.5).
- **Re-run autotuning (Chapter 8) fully on each target platform.** Don't assume a discovered `Config` list — or even the same *search dimensions* — transfers; §26.4 and §26.6 both give concrete evidence this genuinely doesn't hold.
- **Re-check shared-memory/LDS capacity assumptions (Chapter 16) per target and per generation**, not just per vendor — capacities differ meaningfully even within one vendor's own product line (CDNA2 vs. CDNA3).
- **Look for vendor-specific layout/epilogue optimizations** (like `OPTIMIZE_EPILOGUE`) rather than assuming the default behavior is already optimal on a non-default platform.
- **Use Chapter 13's `is_cuda()`/`is_hip()` pattern** for any backend-conditional logic — `libdevice` paths, `extern_libs`, or now, platform-specific autotuning dimensions.
- **Test on every platform you claim to support.** Correctness and performance on one vendor's hardware are not evidence of either on another's.

## 26.9 Hands-On

**Exercise 1 — Query, don't assume, warp size.** For any kernel from this curriculum, add explicit warp/wavefront-size querying rather than an implicit assumption of 32. If you have access to AMD Instinct hardware, confirm it reports 64; if not, reason through precisely what would need to change in a kernel that had silently assumed 32 everywhere.

**Exercise 2 (AMD hardware required) — Rediscover, don't port, the autotuning search.** Take Chapter 9's matmul and run the *full* autotuning search fresh on AMD hardware, rather than porting the NVIDIA-discovered config list. Compare the winning configuration's shape against the NVIDIA winner, and — if feasible — add `matrix_instr_nonkdim` as a new, AMD-specific search dimension and confirm it changes the outcome, directly testing §26.4's claim.

**Exercise 3 (AMD hardware required) — Confirm `OPTIMIZE_EPILOGUE` directly.** Benchmark a matmul kernel with `OPTIMIZE_EPILOGUE=0` versus `=1` for your specific shapes, confirm whether AMD's "usually beneficial" guidance holds, and dump TTGIR for both (Chapter 14) to connect the observed difference back to Chapter 16's `convert_layout` discussion directly.

**Exercise 4 — Quantify the divergence-cost difference.** Construct a kernel with heavy, structured per-lane branching (a `tl.where` condition true for roughly half the lanes in each warp/wavefront). If you have access to both NVIDIA and AMD hardware, measure the actual divergence cost on each; if not, reason through, using §26.2's numbers, how the cost should differ given the warp=32 vs. wavefront=64 distinction.

**Exercise 5 — Per-generation shared-memory budget checking.** Take a kernel that was close to Chapter 16's shared-memory-exhaustion boundary, and check whether it fits within LDS budget on both a CDNA2 target (64 KB) and a CDNA3 target (128 KB) — a concrete exercise in per-*generation*, not just per-*vendor*, resource budgeting.

## 26.10 Check Your Understanding

1. Explain, using §26.2's numbers, why a kernel tuned implicitly for warp=32 can lose roughly half its performance on wave=64 hardware — what, mechanically, is being wasted?
2. Why does Chapter 6's `tl.where` cost analysis need to be revisited, quantitatively, when moving from NVIDIA to AMD hardware?
3. Using `OPTIMIZE_EPILOGUE` as your example, explain in your own words how a compiler-level concept (layout conversion cost, Chapter 16) can end up exposed as a simple, vendor-specific environment variable.
4. Why is "re-benchmark the same autotuning config list on the new platform" an insufficient porting strategy, based on §26.4 and §26.6?

## 26.11 What's Next

Chapter 27 turns to how this curriculum's techniques show up in real, production kernel libraries — FlashAttention, Liger Kernels, and the fine-tuning-adjacent kernels you may already have touched in your own work — and what a mature, maintained Triton kernel codebase actually looks like end to end, informed by everything from correctness discipline (Chapter 21) to profiling (Chapter 22) to the portability practices this chapter just built.
