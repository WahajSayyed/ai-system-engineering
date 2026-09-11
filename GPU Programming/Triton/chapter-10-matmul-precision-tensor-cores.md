# Chapter 10 — Matrix Multiplication II: Precision, Mixed Types & Tensor Cores

## 10.1 What Chapter 9 Deliberately Left Unexamined

Chapter 9 told you the matmul accumulator should be `float32` regardless of input dtype, and left it there so the tiling algorithm could stay the center of attention. This chapter opens `tl.dot` back up and asks the question Chapter 9 postponed: **what actually happens to the *inputs*, and what hardware executes the multiply?**

The short answer is: `tl.dot` doesn't do one thing — it dispatches to different tensor-core instructions depending on your input dtypes, and for one specific dtype (`float32`) it makes a default precision decision that will surprise anyone who assumes "float32 in, full IEEE float32 math" without reading this chapter.

## 10.2 `tl.dot`'s Actual Signature

```python
tl.dot(input, other, acc=None, input_precision=None, allow_tf32=None,
       max_num_imprecise_acc=None, out_dtype=None)
```

`input`/`other` accept `int8`, `float8_e5m2` (and related fp8 variants), `float16`, `bfloat16`, or `float32`. Whichever of these you pass determines which tensor-core MMA (matrix-multiply-accumulate) instruction the compiler emits — exactly as `num_warps`/`num_stages` are compiler-facing hints rather than semantic changes, **you never choose the specific hardware instruction yourself; you choose the dtype, and the compiler picks the matching instruction.** This is the direct payoff of Triton's abstraction level from Chapter 1: on raw CUDA, selecting the right `wmma`/`mma.sync`/`wgmma` variant for a given dtype and GPU generation is your responsibility; here it's the compiler's.

## 10.3 fp16 vs. bf16: A Real Trade-off, Not a Default to Copy Blindly

Both are 16-bit floating-point formats, and both are common `tl.dot` input dtypes, but they split their 16 bits differently:

- **float16**: 1 sign bit, 5 exponent bits, 10 mantissa bits. More mantissa → more precision within its representable range. Fewer exponent bits → a much narrower dynamic range than `float32`, meaning values much larger or smaller than roughly `65504`/`6×10⁻⁵` overflow or underflow. This is precisely why early mixed-precision training needed **loss scaling** (multiplying the loss by a constant before backprop, then dividing gradients back down) — without it, gradients routinely underflowed to zero in fp16.
- **bfloat16**: 1 sign bit, 8 exponent bits, 7 mantissa bits. The exponent field is the *same width as float32's* — meaning bf16 has essentially the same dynamic range as full fp32, at the cost of noticeably less mantissa precision than fp16. This is why bf16 became the default for most modern LLM training and inference: you get fp32-like numerical *range* (no loss scaling needed, dramatically fewer overflow surprises) in exchange for accepting coarser precision within that range.

Neither format is strictly better — it's a genuine trade-off between precision and range, and the exercises in this chapter will have you construct inputs where each format visibly wins over the other, rather than taking "just use bf16" as an unexamined default.

## 10.4 The `float32` Surprise: TF32 Is the Default, Not IEEE

Here is the fact in this chapter most likely to change how you think about a kernel you've already written: **when you pass `float32` tensors into `tl.dot` on an NVIDIA GPU with tensor cores, Triton does *not*, by default, perform full IEEE-754 float32 multiplication.** (cite index="20-1">The default input precision for tensor-core-capable devices is "tf32"</cite> — TensorFloat-32, a format that (cite index="28-1">borrows the 10-bit mantissa of FP16 for precision while keeping the 8-bit exponent of FP32 for dynamic range</cite>, packed inside a 32-bit container. Your kernel's `float32` inputs are silently truncated to this reduced-mantissa format before the tensor-core multiply happens — you did not opt into this by passing a special dtype; it happens because `float32` inputs on tensor-core hardware default to TF32 unless you explicitly say otherwise.

This is a real accuracy consideration, not a curiosity: (cite index="20-1">TF32 truncates rather than rounds the float32 mantissa, which can bias the result</cite> — for best numerical results when you do want TF32, you should round explicitly rather than rely on truncation, or use a tensor descriptor's `round_f32_to_tf32=True` option (Chapter 5) rather than letting silent truncation happen. If you've ever compared a Triton kernel's `float32` matmul against a NumPy/CPU reference and seen a gap larger than you expected, TF32 truncation is one of the first places to look — *before* assuming your kernel logic is wrong.

You control this via `input_precision` — three options on NVIDIA hardware (AMD has a narrower set, described below): `"tf32"` (the default just described), `"tf32x3"` (§10.5), and `"ieee"` (full-precision float32 math, computed in software rather than on the tensor cores — meaningfully slower, but bit-accurate). `allow_tf32` is a deprecated boolean alias for setting `input_precision="tf32"`; you can only specify one of `input_precision`/`allow_tf32`, not both. On AMD, only `"ieee"` is available generally, with `"tf32"` available on CDNA3 hardware specifically.

There's also a global override, useful for debugging without touching kernel source: the `TRITON_F32_DEFAULT` environment variable (`ieee`, `tf32`, or `tf32x3`) changes the default for every `tl.dot` call in the process. This is a genuinely practical technique: if you suspect a numerical discrepancy might be TF32-related, rerun with `TRITON_F32_DEFAULT=ieee` and see if the gap closes — no code changes required, and you get a clean before/after comparison.

## 10.5 Splitting the Difference: `tf32x3`

Full `"ieee"` precision is accurate but abandons the tensor cores entirely — a large speed penalty. Plain `"tf32"` uses the tensor cores at full speed but accepts real mantissa loss. `"tf32x3"` is a middle path: (cite index="24-1">it implements a technique that recovers additional precision by decomposing each float32 operand and performing three separate TF32 tensor-core matmuls, then combining the partial results</cite> — roughly, splitting each `float32` value into a TF32-representable "big" part and a residual "small" part, computing the big×big, big×small, and small×big cross terms on the tensor cores, and summing them to recover much of the precision a single TF32 pass would lose. You pay roughly 3x the tensor-core time of a single `"tf32"` pass, but far less than the `"ieee"` software-dot penalty, while landing much closer to true float32 accuracy. (Triton also exposes analogous `bf16x3`/`bf16x6` tricks at the MLIR-dialect level, following the same idea with bf16 as the base format.)

This gives you a genuine three-point dial for `float32` matmuls: **`ieee`** (correct, slow) — **`tf32x3`** (very close to correct, moderate cost) — **`tf32`** (noticeably less precise, fastest) — and which point is right depends entirely on whether your downstream computation is sensitive to the accumulated error, which is exactly what this chapter's exercises will have you measure rather than guess.

## 10.6 fp8: Two Different Trade-offs at 8 Bits

Beyond 16-bit formats, `tl.dot` accepts 8-bit floating-point inputs — the current frontier for LLM inference throughput (you'll build a real block-scaled fp8 matmul kernel in Chapter 20; this section is the precision groundwork for that). The two common variants split their 8 bits differently, mirroring the fp16-vs-bf16 trade-off one level down:

- **`float8_e5m2`** — 5 exponent bits, 2 mantissa bits: wide dynamic range, very coarse precision.
- **`float8_e4m3`** (Triton's fp8e4nv/fp8e4b15 variants) — 4 exponent bits, 3 mantissa bits: narrower range, one more bit of precision.

At this bit width, the precision loss is severe enough that production fp8 kernels essentially never use a bare fp8 matmul — they pair it with **per-block scaling factors** (Chapter 20) that rescale each tile of data into fp8's usable range before the multiply, and rescale the result back afterward. Treat this section as motivating *why* that machinery in Chapter 20 exists, not as something to use unscaled today.

One more fp8-specific `tl.dot` argument worth knowing about now: **`max_num_imprecise_acc`**. On some tensor-core generations, accumulating many fp8 partial products directly in hardware is itself a reduced-precision operation compared to the fp32-accumulator ideal from Chapter 9. `max_num_imprecise_acc` caps how many K-dimension steps are allowed to use this faster-but-less-precise accumulation path before the compiler is required to periodically flush into a fully precise accumulator — a direct, hardware-forced nuance on the "always accumulate wide" principle from Chapter 9, §9.3: at fp8, "wide accumulation" isn't free even when you ask for it, and this parameter is how you trade accuracy for throughput explicitly rather than getting a fixed, unexamined default.

## 10.7 `out_dtype`: Decoupling Accumulation from Output

`out_dtype` lets you specify `tl.dot`'s returned dtype independent of both the input dtypes and the internal accumulator precision — e.g., accumulate in `float32` (as Chapter 9 did) but request the returned tile already cast to `float16`, saving you the explicit `.to(tl.float16)` call Chapter 9's kernel performed by hand after the K-loop. Functionally equivalent to what you already did; useful to know it can live inside `tl.dot` itself when you want it there.

## 10.8 Practical Debugging Habit: Isolate Precision from Logic

Given everything above, add this to your debugging toolkit alongside the `other=` correctness bug from Chapter 4 and the cache-inspection habits from Chapter 2: **when a Triton matmul kernel's results don't match a reference closely enough, don't assume your kernel logic is wrong before ruling out precision.** Rerun with `TRITON_F32_DEFAULT=ieee` (for fp32 inputs) or recompute your reference in the *same* reduced-precision format your kernel actually used (not naively in fp64/fp32, which will always show a "discrepancy" against a lower-precision kernel that isn't actually a bug). A huge fraction of "my Triton kernel gives a different answer than PyTorch" reports are precision-format mismatches between the test and the kernel, not logic errors — confirm this before you start debugging masks and offsets.

## 10.9 Hands-On

**Exercise 1 — Measure the `ieee` / `tf32x3` / `tf32` speed-accuracy ladder.** Take Chapter 9's matmul kernel with `float32` inputs. Run it three times — `input_precision="ieee"`, `"tf32x3"`, and the default `"tf32"` — against a reference computed in `torch.float64` (cast down for comparison). For each, record both wall-clock time (via `triton.testing.do_bench`) and a numerical error metric (e.g., max absolute difference or relative error) against the float64 reference. Confirm the ordering matches §10.5's description: `ieee` most accurate and slowest, `tf32` fastest and least accurate, `tf32x3` in between on both axes.

**Exercise 2 — Construct a case where fp16 beats bf16.** Build a matmul input where every value has a similar, moderate magnitude (no risk of overflow) and where fine-grained precision matters — e.g., inputs clustered tightly around a value where bf16's coarser mantissa causes visibly more rounding than fp16's. Confirm fp16 gives a smaller error against a float32 reference than bf16 does in this specific regime.

**Exercise 3 — Construct a case where bf16 beats fp16.** Now build an input with a wide dynamic range — some values very large, some very small, within the same matrix — sized so fp16 genuinely overflows to `inf` or underflows to `0` for some elements, while bf16 (matching fp32's exponent range) represents all of them without over/underflow. Confirm fp16's result contains `inf`/`nan` or clearly wrong near-zero values where bf16's does not.

**Exercise 4 — Use `TRITON_F32_DEFAULT` as a debugging tool.** Take any `float32` kernel from this chapter, and without changing a single line of its source, run it twice from the shell with `TRITON_F32_DEFAULT=tf32` and `TRITON_F32_DEFAULT=ieee`. Confirm the *only* thing that changes is numerical accuracy and runtime — not correctness of the surrounding logic — reinforcing §10.8's debugging habit by having actually done it once.

**Exercise 5 (optional, requires fp8-capable hardware) — A first look at fp8 error.** Cast a matmul's inputs to `float8_e5m2` and (if your Triton build exposes it) an `e4m3` variant, run both through `tl.dot`, and compare error against a float32 reference. Don't attempt to fix the error with scaling yet — the goal is simply to *see* how much larger the error is at fp8 than anything in Exercises 1–3, motivating why Chapter 20's block-scaling machinery exists at all.

## 10.10 Check Your Understanding

1. Someone tells you, "I passed `float32` tensors into `tl.dot`, so I got full float32 precision." Explain precisely why this statement is false by default on NVIDIA tensor-core hardware, and what you'd need to change to make it true.
2. Why does bf16 need no loss scaling in mixed-precision training while fp16 historically did? Answer in terms of the bit layout, not just "bf16 is newer/better."
3. What is `tf32x3` actually trading, and against what two alternatives is it positioned? Why would a real kernel choose it over either alternative?
4. Why does `max_num_imprecise_acc` exist specifically as an fp8 concern, when Chapter 9 already established that Triton accumulates in `float32` regardless of input dtype? What does this reveal about the limits of "the compiler handles it"?

## 10.11 What's Next

You now understand both halves of a production-grade GEMM: the tiling and scheduling algorithm (Chapter 9) and the precision decisions `tl.dot` makes on your behalf (this chapter) — together, this is most of what separates a "correct" matmul kernel from a genuinely fast, numerically-considered one. Chapter 11 shifts domains entirely: random number generation inside a kernel, using Triton's Philox-based PRNG, building toward a fused, memory-efficient dropout implementation.
