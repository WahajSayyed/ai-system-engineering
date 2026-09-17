# Chapter 24 — torch.compile & TorchInductor Internals

## 24.1 Why This Chapter Comes After, Not Before

Chapter 1, §1.5 mentioned in passing that `torch.compile` is arguably Triton's largest deployment surface — most PyTorch users benefit from Triton without ever writing it. This chapter makes that claim concrete: how TorchInductor (the default `torch.compile` backend) actually turns your model into Triton kernels, and — the more important question for someone who's just spent twenty-three chapters learning to write Triton by hand — precisely when that automatic process is good enough, and when it structurally isn't.

## 24.2 The Pipeline: FX Graph → InductorIR → Triton

TorchInductor's own internal pipeline is structurally analogous to the one Chapter 14 walked you through for Triton itself, just one level of abstraction higher: (cite index="62-1">an FX graph is first translated into InductorIR, a loop-level intermediate representation describing how operations execute through nested loops, at which point TorchInductor performs optimizations — loop tiling, reordering, and fusion — before generating final Triton code</cite>. Where Chapter 14's pipeline took a single kernel's Python source down to PTX, this pipeline takes a *whole model graph* down to a *collection of Triton kernels* — and Triton is its actual output target, not an implementation detail hidden further down. Everything from Part IV of this curriculum is directly useful here: once Inductor has emitted Triton, you can read it with exactly the tools Chapter 14 gave you.

## 24.3 Fusion: Chapter 7's Lesson, Automated

Here's the satisfying part: (cite index="59-1">Inductor automatically groups dependent operations together into single, efficient Triton kernels, keeping data in faster memory close to the register and cutting down on kernel-launch overhead</cite>. This is *exactly* Chapter 7's fused-softmax lesson — avoid redundant round-trips to global memory by keeping intermediate results on-chip — except Inductor is attempting it **automatically, across your entire model graph**, without you writing a line of Triton.

You can watch this happen directly: (cite index="59-1">running your script with the right `TORCH_LOGS` setting outputs the generated Triton kernels to your terminal</cite>, and the generated kernel names encode exactly which operations got fused — a kernel named something like `triton_per_fused_add_mul_sum_0` tells you, directly from its name, that an add, a multiply, and a sum reduction were fused into one kernel. This is Chapter 14's "go read the compiler's actual output" habit, applied one level higher up the stack.

## 24.4 The Real, Structural Limitation: Matmul Isn't Fused the Same Way

Here's the fact this chapter most wants you to walk away with, because it directly explains why Parts III and V of this curriculum remain valuable in a `torch.compile`-first world. (cite index="62-1">Matrix multiplication is a significant exception to Inductor's general fusion machinery — due to its critical role in deep learning workloads, matmul is not generated natively by TorchInductor at all; instead, it's routed through a separate, hand-optimized Triton template (or cuBLAS)</cite>. This template supports only *limited* fusion of simple pointwise operations directly around it (a bias-add, an activation) — (cite index="62-1">it does not fuse more complex surrounding operations, like gather or scatter, into the matmul itself</cite>.

A vivid, concrete illustration: (cite index="62-1">an FX graph representing a gather-then-matmul-then-scatter pattern is compiled by default into three separate, unfused Triton kernels — one for the gather, one using the matmul template, one for the scatter — rather than a single fused kernel</cite>. This is a genuine, structural gap, not a temporary rough edge: the operation that benefits *most* from custom fusion (matmul, exactly as Chapter 9's whole L2-locality argument demonstrated) is the one operation Inductor's general-purpose fusion pass is least able to reach into.

## 24.5 A Concrete, Current Case Where the Compiler Measurably Loses

This isn't a hypothetical caveat — it's a measured, recent finding, and it's directly relevant to work you may already be doing. A 2026 study evaluating whether `torch.compile` could automatically achieve the same kind of cross-matmul fusion a hand-written kernel gets, for a **SwiGLU**-style gated pattern (fusing computation across *two separate weight matrices* — exactly the shape of a gated MLP block in a modern LLM, directly relevant if you've worked with Qwen-style architectures), found: (cite index="56-1">`torch.compile` with `mode="max-autotune"` was *slower* than eager PyTorch, achieving only 0.35–0.94x of eager performance — worse at small batch sizes, converging toward parity only at large ones — because the compiler's Triton-generated kernels and CUDA graph overhead could not compensate for its inability to fuse across the two separate weight matrices</cite>. Critically: (cite index="56-1">neither `fullgraph=True`, wrapping the operation in a custom `torch.autograd.Function`, nor enabling `coordinate_descent_tuning` meaningfully improved this — all remained within roughly ±4% of the baseline compiled performance</cite>. This is precisely the situation where a hand-written, properly fused Triton kernel — built with exactly the techniques from Chapters 9, 17, and 19 of this curriculum, and integrated via Chapter 23's `triton_op` — genuinely, measurably outperforms what the automatic compiler can currently produce, not as a theoretical possibility but as a documented, recent research result.

## 24.6 `max-autotune`, Precisely

`torch.compile(model, mode="max-autotune")` does something structurally familiar: (cite index="53-1">it profiles multiple Triton kernel configurations *and* multiple competing matmul implementations (Triton-generated vs. cuBLAS, and in some integrations, CUTLASS) to select whichever measures fastest</cite> — the same "compile several candidates, benchmark, keep the winner" idea as Chapter 8's `@triton.autotune`, now operating at the whole-model level and choosing between entire *backends*, not just tile-size configurations. The cost is exactly what you'd expect from Chapter 6/8's compile-time-versus-runtime-speed theme, now at model scale: (cite index="53-1">first compilation under `max-autotune` is substantially slower</cite>. The available modes form a clear spectrum: `default` (balanced compile time vs. runtime), `reduce-overhead` (CUDA graphs, less Python/CPU overhead, more memory), `max-autotune` (longest compile, benchmarks the most alternatives), and `max-autotune-no-cudagraphs`.

## 24.7 Graph Breaks: Vocabulary for What Chapter 23 Already Showed You

(cite index="53-1">A graph break occurs when TorchDynamo encounters code it cannot trace into the FX graph; the model then runs as multiple separately-compiled subgraphs with eager Python execution stitched in between</cite>. You've already seen exactly this: Chapter 23's plain `torch.autograd.Function` (tier 1) is precisely such a break point — `torch.compile` cannot trace into it, full stop. `torch.compile(model, fullgraph=True)` **errors on any break** rather than silently falling back to eager for the untraceable parts — a genuinely useful diagnostic mode for confirming a model compiles cleanly end-to-end, and precisely the tool that would have let you *verify*, rather than merely assert, Chapter 23's claim that a properly-built `triton_op` (tier 3) doesn't cause a break where a plain `autograd.Function` does.

## 24.8 Relevance to Your Own Domain: LLM Serving

Worth a direct, honest mention given how closely this connects to real LLM infrastructure work: (cite index="60-1">vLLM builds custom compiler passes on top of `torch.compile`'s infrastructure specifically for LLM serving — choosing between cuBLAS, Triton, and CUTLASS backends per operation, performing prologue/epilogue fusion, and integrating CUDA graphs with FlashAttention/FlashInfer</cite>. And **FlexAttention** — a PyTorch API letting you express custom attention variants (custom masking, custom score modifications) *without* hand-writing a kernel for each variant — (cite index="60-1">uses `torch.compile` under the hood to automatically produce a custom Triton template</cite> from your specification. This is a genuinely elegant middle ground between "hand-write every attention variant yourself" (Chapters 17–18) and "accept whatever a fixed, generic kernel offers" — worth knowing exists, and worth comparing against your own hand-written attention kernel if you ever need a variant it doesn't already cover well.

## 24.9 An Honest, Current Caveat: Inductor's Codegen Has Real Rough Edges Too

Consistent with this curriculum's treatment of other maturing tooling (Chapter 20's block-scaled matmul, Chapter 21's interpreter mode), it's worth knowing Inductor's automatic Triton generation isn't flawless either. A documented, real issue: combining `torch.bucketize` inside a fused epilogue with `max_autotune_gemm_backends` restricted to Triton has produced a `NameError` about an undefined `XBLOCK` variable — a genuine bug in the generated code's own template substitution, not a user error. The practical lesson is the same one this curriculum has repeated in every chapter dealing with frontier tooling: verify, don't assume — especially when combining less-common operations with more aggressive compilation modes.

## 24.10 A Decision Framework

Putting this chapter's evidence together into an actual working policy:

- **Default: let `torch.compile`/Inductor handle it.** For ordinary elementwise- and reduction-heavy model code, automatic fusion is frequently good enough, and costs zero extra engineering effort — directly consistent with this curriculum's repeated anti-overengineering stance (Chapter 1, §1.6; Chapter 23, §23.1).
- **Reach for a hand-written kernel specifically when**: (a) the operation is matmul-adjacent and needs fusion *across* the matmul itself — Inductor's demonstrated structural blind spot (§24.4–24.5), with SwiGLU/gated-MLP patterns as a concrete, currently-relevant example; (b) profiling (Chapter 22) shows Inductor's generated kernel is measurably worse than a hand-tuned alternative for *your specific* shape/hardware combination — always measure, never assume in either direction; (c) you need scheduling control a generic template structurally can't express — persistent/grouped scheduling (Chapter 19), block-scaled precision control (Chapter 20), or warp specialization (Chapter 25).
- **The actual, common production workflow**: write the model in ordinary PyTorch, let `torch.compile` handle everything by default, profile (Chapter 22), and replace *only* the specific bottleneck operations it handles poorly with hand-written Triton kernels, integrated via Chapter 23's `triton_op` — not "hand-write everything," not "trust the compiler blindly," a measured, evidence-driven middle path.

## 24.11 Hands-On

**Exercise 1 — Confirm automatic fusion, and read the generated kernel's name.** Compile an ordinary elementwise-plus-reduction sequence (bias-add, ReLU, sum, say) with `torch.compile`, use `TORCH_LOGS` to view the generated Triton code, and confirm it's a single fused kernel. Read the generated kernel's name and confirm it names the operations you expect.

**Exercise 2 — Try to reproduce the SwiGLU finding yourself.** Build a small SwiGLU-style gated module (two separate weight matrices), and benchmark: eager PyTorch, `torch.compile(mode="max-autotune")`, and a hand-written fused Triton kernel using this curriculum's own techniques. See whether you can reproduce something resembling §24.5's finding — a genuine attempt to verify a real, current, surprising research claim with your own measurements, not accept it on faith.

**Exercise 3 — Confirm the gather-matmul-scatter fusion gap.** Construct the pattern from §24.4, compile it, and use `TORCH_LOGS` to confirm it produces multiple separate Triton kernels rather than one fused kernel — direct, hands-on evidence of Inductor's matmul-fusion blind spot.

**Exercise 4 — Connect graph breaks back to Chapter 23's tiers.** Wrap a model containing a plain `torch.autograd.Function` (Chapter 23, tier 1) in `torch.compile(fullgraph=True)` and confirm it errors with a graph break. Then swap in a properly-built `triton_op` (tier 3) and confirm `fullgraph=True` now succeeds — the direct, hands-on payoff of a distinction Chapter 23 asked you to take partly on description.

**Exercise 5 — Quantify the `max-autotune` compile-time cost.** Benchmark first-compilation time versus steady-state call time for a matmul-heavy model under `max-autotune`, making the compile-time-versus-runtime-speed tradeoff (Chapters 6 and 8, now at whole-model scale) concrete with your own numbers.

**Exercise 6 (exploratory) — FlexAttention, if relevant to your work.** If you have a model using attention, try PyTorch's FlexAttention API with a custom score-modification function, and use `TORCH_LOGS` to inspect the Triton template it auto-generates. Compare it, conceptually, against your own hand-written Chapter 17 flash-attention kernel — what does the auto-generated version handle well, and where might a hand-written variant still be justified?

## 24.12 Check Your Understanding

1. Explain, in your own words, why matmul is structurally harder for Inductor's general fusion pass to absorb than an elementwise-plus-reduction sequence.
2. Using the SwiGLU finding from §24.5, explain why `fullgraph=True` alone doesn't fix the underlying performance gap — what does `fullgraph=True` actually guarantee, and what does it not guarantee?
3. Why does this chapter recommend "let the compiler handle it by default, profile, then selectively hand-write bottlenecks" rather than either extreme? What would go wrong with always hand-writing everything, and what would go wrong with never doing so?
4. How does a graph break relate directly to Chapter 23's distinction between a plain `torch.autograd.Function` and a properly-wrapped `triton_op`?

## 24.13 What's Next

That completes the compiler- and integration-focused material building toward production use. Part VII turns to the genuine frontier: Chapter 25 covers warp specialization, TMA, and Gluon — the lower-level dialect you reach for precisely when, as this chapter's SwiGLU example demonstrated, neither the automatic compiler nor a standard hand-written Triton kernel is enough, and you need Hopper/Blackwell-level hardware control directly.
