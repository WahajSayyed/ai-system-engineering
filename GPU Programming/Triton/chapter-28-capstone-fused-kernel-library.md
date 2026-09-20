# Chapter 28 — Capstone: A Small Fused-Kernel Library

## 28.1 What This Chapter Is

This chapter doesn't teach new material — it's a structured project synthesizing everything from Parts I through VII into one coherent deliverable: **a small, cohesive fused-kernel library for a transformer block**, built, autotuned, tested, integrated, and profiled end to end, exactly the way Chapter 27's real production libraries are built. Think of it as a small, honest "mini-Liger" — not because it needs to match Liger Kernel's scope or maturity, but because building even a small version of that thing, correctly and with the right discipline at every step, is the single best confirmation that this curriculum's individual chapters have actually composed into real capability.

## 28.2 Project Scope

Implement fused, autotuned, tested, and profiled Triton kernels for:

1. **RMSNorm** (forward + backward) — a simpler cousin of Chapter 12's LayerNorm.
2. **RoPE** (forward + backward) — rotary position embeddings, an elementwise/pairwise rotation.
3. **Fused causal attention** (forward + backward) — Chapters 17–18, the hardest and most direct reuse of prior work.

Then assemble these into a minimal `TransformerBlock` `nn.Module`, wire it into a tiny real training loop, and benchmark the whole thing honestly against a pure-PyTorch reference — the same spirit as Chapter 9's matmul comparing itself against cuBLAS: getting close to expert-written performance with far less code is the actual point, not necessarily beating it outright.

**Before writing any kernel code**, write down your success criteria — this is deliberate, not a formality. Borrow Chapter 21, §21.9's edge-case checklist and Chapter 22's benchmarking methodology, and commit to them *up front*: what shapes will you test, what tolerance is appropriate for the precision you're using (Chapter 10, Chapter 21, §21.8), and what performance bar (relative to the PyTorch baseline) would count as success. Deciding this after the fact, once you've seen your own numbers, is exactly the kind of motivated reasoning this discipline exists to prevent.

## 28.3 Phase 1: RMSNorm — The "Learn the Loop" Kernel

Start here because it's the smallest complete forward/backward pair that still has a real cross-program reduction to get right.

- **Forward**: one program per row, `BLOCK_SIZE >= N` (Chapter 7's whole-row-in-one-block pattern), computing `y = x / sqrt(mean(x²) + eps) * weight` — simpler than Chapter 12's LayerNorm since there's no mean-centering, but otherwise the same fused-reduction shape.
- **Backward**: `dx` is local per row (Chapter 12, §12.3's pattern applies directly); `dweight` is a genuine cross-row reduction — the *same* bucketed-lock synchronization from Chapter 12, §12.4 is the correct tool here, not a simplification you get to skip.
- **Test**: non-power-of-two `N`, gradient check against `torch.autograd` (Chapter 21, Chapter 23's `gradcheck`).
- **Autotune**: `num_warps`/`num_stages` via `@triton.autotune` (Chapter 8).
- **Integrate**: wrap as a `torch.library.triton_op` (Chapter 23), and confirm `torch.compile(fullgraph=True)` succeeds without a graph break (Chapter 24).
- **Profile**: confirm high Memory SOL%, low Compute SOL% via Nsight Compute (Chapter 22) — RMSNorm should be bandwidth-bound, exactly like Chapter 7's softmax, and now you can *measure* that rather than assume it.

## 28.4 Phase 2: RoPE

A good second kernel precisely because it's small and clean: a paired-rotation of adjacent (or interleaved) elements along the head dimension, parameterized by position. Its backward is a genuinely satisfying, self-contained derivation — the gradient of a rotation is (close to) the inverse rotation, a small, complete exercise in deriving a backward pass by hand rather than reaching for one you've already seen. Test correctness against a reference RoPE implementation across non-power-of-two sequence lengths and head dimensions, and integrate it into the same `triton_op`-based structure as RMSNorm.

## 28.5 Phase 3: Fused Causal Attention — The Hard Core

This is where Chapters 17–18 pay off directly, not as a reference to consult but as code to reuse and adapt:

- **Forward**: online softmax, the two-stage (`STAGE=1`/`STAGE=2`) causal masking split (Chapter 17, §17.6), saved log-sum-exp for backward.
- **Backward**: the three-kernel structure — preprocessing (`Delta`), `dK`/`dV`, `dQ` (Chapter 18) — including the de-scaling discipline from Chapter 18, §18.4.
- **Autotune**: `BLOCK_M`/`BLOCK_N`/`num_warps`/`num_stages` (Chapter 8), keyed appropriately on sequence length and head dimension.
- **Correctness**: `gradcheck` against a reference implementation (Chapter 23), the full edge-case checklist (Chapter 21, §21.9), and — directly reusing Chapter 18, Exercise 6's methodology — a **peak-memory measurement** confirming genuine `O(N)` rather than `O(N²)` scaling, the actual headline claim of this whole kernel family.
- **Profile**: confirm the transition to compute-bound behavior at large sequence lengths (Chapter 22's SOL% measurement) — attention's arithmetic intensity should look qualitatively different from RMSNorm's.

## 28.6 Phase 4: Assembly — A Minimal `TransformerBlock`

Wire RMSNorm, RoPE, and fused attention (plus an ordinary PyTorch MLP, or your own fused SwiGLU as a stretch goal per §28.8) into a real `nn.Module`. Two concrete gates before moving on:

- **`torch.compile(fullgraph=True)` must succeed on the whole block**, with no graph breaks anywhere (Chapter 24, §24.7) — direct, hands-on confirmation that every `triton_op` integration point (Chapter 23) was done correctly, not just individually testable in isolation.
- **A convergence test** (Chapter 27, §27.2's concept, now actually implemented rather than only discussed): run a small number of real training steps on a toy dataset, once with your fused block and once with a pure-PyTorch reference block, and confirm the loss curves match within a reasonable tolerance. This is the capstone's real correctness bar — not because per-kernel tests aren't valuable, but because this is the one test that would catch a subtle numerical issue accumulating silently across many steps, exactly the failure mode Chapter 27 introduced this concept to guard against.

## 28.7 Phase 5: Honest, End-to-End Benchmarking

Measure your assembled block against the pure-PyTorch reference: wall-clock throughput (tokens/second), peak memory, and — wherever feasible — achieved TFLOP/s or GB/s against the theoretical peak your hardware supports (Chapter 22's SOL%-style measurement). Write up the result the way this curriculum has modeled honest benchmarking from Chapter 9 onward: getting close to, or matching, expert-tuned performance with dramatically less code than a hand-tuned CUDA equivalent is the real, legitimate win — not an obligation to beat cuBLAS or a production library outright.

## 28.8 Phase 6: Stretch Goals

Pick based on your own interest and hardware access — none of these are required to consider the capstone complete:

- **A block-scaled (fp8/`mxfp4`) variant** of the attention or a linear layer (Chapter 20), with the accuracy-vs-baseline measurement Chapter 20's own exercises modeled.
- **A Gluon, warp-specialized variant** of a kernel, if you have Hopper/Blackwell hardware access (Chapter 25) — benchmarked honestly against your standard Triton-language version, per Chapter 25, §25.9's "measure, don't assume" discipline.
- **A group-GEMM MoE variant** of the MLP layer (Chapter 19), if you're interested in extending this toward the Mixture-of-Experts territory Chapter 27's Unsloth discussion touched on.
- **Cross-platform correctness** on AMD hardware, if available, following Chapter 26's portability checklist — even if performance parity isn't the goal, confirming correctness on a second vendor is a genuine, non-trivial exercise.

## 28.9 A Self-Assessment Checklist Across the Whole Curriculum

A genuine final check — not "did you read every chapter," but "can you do this":

- **Part I–II (Foundations, Programming Model)**: Can you explain, without notes, why Triton's tile-level abstraction changes what a kernel author is responsible for versus raw CUDA? Can you write correct masking for an arbitrary, non-block-aligned shape from memory?
- **Part III (Core Kernels)**: Did your RMSNorm and attention kernels need `other=` values chosen deliberately for their reductions, and did you get them right the first time or catch a bug via Chapter 4's method?
- **Part IV (Compiler & Architecture)**: Can you read your own kernels' TTGIR and identify at least one layout anchor and, if present, a `ttg.convert_layout`?
- **Part V (Attention & Advanced Kernels)**: Can you explain, to someone else, why the backward pass recomputes `P` rather than storing it, and derive the `Delta` identity from memory?
- **Part VI (Debugging & Performance)**: Did you use interpreter mode *and* real hardware profiling as genuinely different tools for genuinely different bug classes, not interchangeably?
- **Part VII (Frontier & Production)**: Can you state, precisely, when you'd reach for Gluon instead of standard Triton, and when you'd trust `torch.compile` instead of writing a kernel by hand?

If any of these feel shaky, that's useful information about where to spend more deliberate practice — not a sign the curriculum failed, since several of these (Part IV and Part VII especially) are genuinely deep, and fluency there is built through repeated exposure, not one pass through a chapter.

## 28.10 Closing: How to Keep Learning as This Keeps Changing

A theme running through this entire curriculum, stated openly rather than glossed over, is that large parts of Triton are genuinely still evolving: the linear-layout unification (Chapter 16), block-scaled matmul's real, recent correctness bugs (Chapter 20), interpreter mode's own rough edges (Chapter 21), Inductor's matmul-fusion blind spot (Chapter 24), Gluon's active hardening (Chapter 25). This isn't a weakness of the material — it's an honest reflection of working with a genuinely live, actively-developed compiler and ecosystem, and the discipline this curriculum has modeled throughout (check current release notes, verify rather than assume, measure rather than trust a cited number blindly) is the actual, durable skill to carry forward, more so than any specific API surface that will keep shifting under you.

Concretely, going forward: read Triton's own release notes and changelog for your installed version before depending on a frontier feature; the GPU MODE community and lecture series remain a live, current source of exactly this kind of practitioner knowledge; Triton-Puzzles remains available for continued deliberate practice on the fundamentals; and — perhaps most usefully, given everything Part IV taught you — Triton's own compiler source is open, and reading it directly, with the TTIR/TTGIR/LLVM vocabulary Chapter 14 gave you, is a genuinely available option when documentation lags behind the code, which it sometimes will.

You've built a fused-kernel library that thinks about memory, compilation, layouts, precision, and correctness the same way the production libraries in Chapter 27 do. That's the actual capstone — not a specific benchmark number, but the fact that you now reach for the right tool, and ask the right question, without needing to be told which one applies.
