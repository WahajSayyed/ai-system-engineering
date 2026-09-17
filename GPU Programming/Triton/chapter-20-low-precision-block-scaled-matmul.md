# Chapter 20 — Low-Precision & Block-Scaled Matmul

## 20.1 Cashing In Chapter 10's Deferred Promise

Chapter 10, §10.6 showed you fp8's severe accuracy problem and stated plainly that production kernels essentially never use bare, unscaled fp8 — they pair it with per-block scaling factors, deferred to "Chapter 20." This is that chapter: the machinery that makes fp8, and the even more aggressive fp4 formats, actually usable in real, accuracy-sensitive workloads.

## 20.2 The Core Idea: Scale Locally, Not Globally

A single global scale factor for an entire tensor is too coarse an instrument. Real weight and activation tensors have magnitude that varies meaningfully across different regions — one row might have values an order of magnitude larger than another. Quantizing the whole tensor against one shared scale forces every region into the same narrow representable range, wasting precision on the regions whose natural magnitude doesn't need it and clipping the regions that exceed it.

**Microscaling** fixes this by assigning a *separate* scale factor to every small, contiguous chunk of the tensor along the matmul's reduction (`K`) dimension — adapting to *local* magnitude variation rather than forcing one global compromise. This is precisely what makes an extremely aggressive format like fp4 (as little as 2–3 bits of genuine precision) usable at all in practice: without fine-grained local scaling, fp4's dynamic range is far too narrow for real tensor magnitude variation; with it, each small chunk gets its own locally-appropriate scale.

The Open Compute Project's **Microscaling (MX)** standard formalizes this precisely: (cite index="59-1">a 1D block of `VEC_SIZE` elements — 32, for the standard mxfp4/mxfp6/mxfp8 formats — shares one scale factor, quantized along the matmul's reduction dimension</cite>. The scale format itself is equally deliberate: (cite index="59-1">MX scales use the `e8m0` format — zero mantissa bits, eight exponent bits — meaning each scale factor is constrained to an exact power of two, from 2⁻¹²⁷ to 2¹²⁷, with the value 255 reserved for NaN</cite>. This "exponent-only" design isn't an accuracy compromise for its own sake — it makes dequantization on hardware a cheap, shift-like rescale rather than a general floating-point multiply. The trade-off is real, though: the scale itself is quantized to the nearest power of two, a small additional source of error layered on top of the element-level fp4/fp8 quantization.

## 20.3 The Format Menu

- **`mxfp4`** — 4-bit elements in the `e2m1` format (1 sign, 2 exponent, 1 mantissa bit), `VEC_SIZE=32`, `e8m0` scales. This is the OCP-standard, vendor-generic format, (cite index="28-1">supported on both NVIDIA and AMD GPUs</cite> — the portable baseline.
- **`mxfp8`** — 8-bit elements (`e4m3` or `e5m2`, the same formats from Chapter 10, §10.6), using the identical 32-wide `e8m0` block-scaling scheme. A noticeably gentler quantization than fp4, with the same scaling machinery underneath.
- **`nvfp4`** — NVIDIA's own fp4 variant, on NVIDIA hardware specifically. The concrete, meaningful difference from plain `mxfp4`: (cite index="29-1">`nvfp4`'s scale factors are stored as `float8_e4m3fn` rather than the coarser, exponent-only `e8m0` format `mxfp4` uses</cite>. This is a real, deliberate design choice — a higher-precision scale format costs a little more storage per scale, in exchange for measurably better overall accuracy than the plain OCP baseline at the same 4-bit element precision. This is exactly *why* two competing fp4 standards exist: NVIDIA specifically engineered `nvfp4` to improve on OCP `mxfp4`'s accuracy, not merely to have a differently-named equivalent.

## 20.4 `tl.dot_scaled`: The Language-Level API

```python
triton.language.dot_scaled(
    lhs, lhs_scale, lhs_format,
    rhs, rhs_scale, rhs_format,
    acc=None, fast_math=False,
    lhs_k_pack=True, rhs_k_pack=True,
    out_dtype=triton.language.float32,
)
```

`lhs`/`rhs` hold the quantized elements — (cite index="34-1">fp4 elements packed two-per-byte into a `uint8` tensor (the first element in the lower bits), or fp8/bf16 elements stored directly</cite>. `lhs_format`/`rhs_format` are strings naming the element format (`"e2m1"` for fp4, `"e4m3"`/`"e5m2"` for fp8, or `"bf16"`/`"fp16"`). `lhs_scale`/`rhs_scale` carry the per-block scale factors — `e8m0`, represented as a `uint8` tensor — shaped `[M, K // group_size]` and `[N, K // group_size]` respectively, where `group_size` is `32` for `e8m0`-scaled formats.

This should feel structurally familiar: `tl.dot_scaled` is to microscaled formats exactly what `tl.dot` (Chapter 10) is to fp16/bf16/fp32 — **one call, dispatching automatically to the appropriate hardware instruction (or a software fallback) based on your declared input formats**, rather than you selecting a specific instruction by hand.

**The software-emulation fallback is worth understanding explicitly, since it changes what "portable" means here.** (cite index="34-1">On hardware without native microscaling tensor-core support, `lhs`/`rhs` are automatically upcast to `bf16` before an ordinary dot product is performed</cite> — with a documented AMD CDNA3-specific exception upcasting to `fp16` instead when one input is already `fp16`. The consequence: a `tl.dot_scaled` kernel **runs correctly on essentially any Triton-supported GPU**, but is only **hardware-accelerated** — genuinely faster, not merely "not broken" — on the newest generations with native microscaling tensor cores. Know which regime you're actually in before drawing performance conclusions from a benchmark.

**A specific, currently-relevant API detail worth flagging directly**: `lhs_k_pack`/`rhs_k_pack` control which dimension fp4 packing runs along, and getting this wrong for your specific format has been a genuine, documented source of user confusion — production code should verify the current documentation's exact requirement for the format combination in use, rather than assuming the defaults are correct by analogy to a different format.

## 20.5 A Preview of the Hardware Underneath (Chapter 25's Territory)

NVIDIA's fifth-generation tensor cores — the ones with native microscaling support, on compute capability 10 (Blackwell) — are literally named **`tcgen05`** in NVIDIA's own instruction-set vocabulary. On supported hardware, `tl.dot_scaled` compiles down to the `tcgen05_mma_scaled` instruction (or AMD's CDNA4 equivalent); Gluon exposes this primitive directly, alongside a companion `tcgen05_copy` operation for efficiently staging scale factors into tensor memory, for situations where `tl.dot_scaled`'s automatic scheduling isn't sufficient and hand control is warranted. Treat this purely as a forward pointer — Chapter 25 is where you'll actually use these primitives directly; this chapter's job is giving you the `tl.dot_scaled` level fluently first.

## 20.6 An Honest Note: This Corner of Triton Is Still Actively Stabilizing

Worth knowing before you rely on this machinery for anything production-critical — block-scaled matmul support is genuinely newer and less battle-tested than most of what this curriculum has covered so far, and the evidence is concrete and recent:

- A documented compiler failure for a specific scaling granularity combination (`BlockWise1x128` with fp8 `e4m3` inputs) — `tl.dot_scaled` simply fails to compile for that configuration as of a late-2025 report.
- An **undocumented breaking shape change**: `rhs_scale`'s expected shape changed between Triton versions from `(K//32, N)` to `(N, K//32)` — exactly the kind of change that silently produces wrong results (not a crash) if you upgrade Triton without re-checking your scale tensor's shape convention.
- Friction around scale tensor dtypes — `dot_scaled` has, at points, rejected the semantically-correct `float8_e8m0fnu` scale dtype and required a `uint8` reinterpretation (`scale.view(torch.uint8)`) as a workaround.
- A genuine, **very recently fixed** correctness bug — as of days before this chapter was written — where `e8m0` scale values of zero, or subnormal-in-bf16 values, produced **completely wrong results across every element** of the affected test cases (16,384 of 16,384 elements wrong, in the reported case), only patched in early September 2026.

None of this means the feature is unusable — it means **testing your own block-scaled kernels thoroughly against a full-precision reference is not optional here**, especially around edge cases like near-zero or subnormal scale values, and checking your specific Triton version's release notes against known fixes before shipping anything built on this chapter is a reasonable, current practice rather than excessive caution.

## 20.7 Hands-On

**Exercise 1 — `mxfp8`, and a direct comparison against Chapter 10's bare fp8.** Implement a small `tl.dot_scaled`-based matmul using `mxfp8`, verify against a `float32` reference, and directly compare its error against the *unscaled* fp8 attempt from Chapter 10, Exercise 5. Quantify, with real numbers, how much accuracy block-scaling actually recovers — making this chapter's motivating claim measured rather than assumed.

**Exercise 2 — `mxfp4`, and the cost of going lower.** Repeat Exercise 1 with `mxfp4`, and compare its error against `mxfp8`'s. Quantify how much additional error the more aggressive 4-bit element format introduces on top of the same scaling machinery.

**Exercise 3 (requires NVIDIA Blackwell-class hardware, optional) — `mxfp4` vs. `nvfp4`.** If available, compare `mxfp4` and `nvfp4` accuracy at matched element precision, and attribute any difference you observe to the scale-format distinction from §20.3 (`e8m0` vs. `float8_e4m3`).

**Exercise 4 — Defeat block-scaling on purpose.** Construct a test case with a very wide dynamic range *within* a single 32-element `K`-dimension scaling group (rather than across groups) — deliberately violating the assumption block-scaling relies on. Confirm accuracy degrades sharply, illustrating concretely that block-scaling helps with magnitude variation *across* blocks, not *within* one.

**Exercise 5 — A real due-diligence check.** Check your installed Triton version against the fix dates for the issues cited in §20.6. This is a genuine exercise in the practice this chapter is asking you to adopt, not a hypothetical one.

**Exercise 6 (exploratory, no special hardware needed) — Confirm the software-emulation fallback directly.** Using Chapter 14 §14.7's GPU-free compilation technique, compile the same `tl.dot_scaled` kernel for a Blackwell (`cc10`) target and for an older target lacking native microscaling support, and inspect the generated code for each. Confirm you can see `tcgen05_mma_scaled`-family instructions in the former and a `bf16`-upcast software path in the latter — direct, hands-on evidence for §20.4's fallback claim.

## 20.8 Check Your Understanding

1. Explain, without referring back to §20.2, why a single global scale factor is insufficient for a format as narrow as fp4, and why per-32-element block scaling addresses this specifically.
2. Why does the `e8m0` scale format restrict scales to exact powers of two, and what does this trade away in exchange for cheaper hardware dequantization?
3. What, precisely, distinguishes `nvfp4` from `mxfp4`, and why does that specific difference translate into better accuracy?
4. A `tl.dot_scaled` kernel runs correctly, but delivers no speedup over an equivalent `tl.dot` kernel using a higher-precision format. What's the first thing you should check, given §20.4?

## 20.9 What's Next

That completes Part V — you've built forward and backward attention, group GEMM, and now the current frontier of low-precision matmul. Part VI turns to a different kind of skill: Chapter 21 covers correctness tooling — the interpreter mode that lets you debug kernel logic without a GPU at all, and `triton-viz` for visualizing what a kernel is actually doing — the tools you'll want to have had all along, now formalized for the more demanding kernels ahead.
