# Chapter 6 — Compile-Time Specialization & Control Flow

## 6.1 Two Different Kinds of "Decision" in a Triton Kernel

Every `if` and `for` you write inside a `@triton.jit` function resolves in one of three fundamentally different ways, and confusing them is a common source of both correctness bugs and unnecessary recompilation. This chapter is about telling them apart on sight:

1. **Trace-time (Python-level) decisions** — resolved once, while Triton is turning your Python source into IR, using only `constexpr` values. The untaken branch of an `if`, or the un-taken iterations of a loop, may never even become part of the compiled kernel.
2. **Compile-time device decisions** — a genuine loop or branch construct exists in the compiled kernel, but the compiler may choose to unroll or specialize it further using hints you provide.
3. **Runtime device decisions** — actual data-dependent control flow (or, for per-element decisions, predication) that executes on the GPU using values not known until the kernel runs.

## 6.2 Trace-Time Branching on `constexpr`

You've already used this without necessarily naming it. When a condition is built entirely from `constexpr` values (kernel arguments annotated `tl.constexpr`, or Python literals), the `if` is evaluated **while Triton is tracing your function**, exactly like a normal Python `if` — because at that moment, `BLOCK_SIZE` or `IS_CAUSAL` genuinely *is* a plain Python `int`/`bool`, not a device value yet.

```python
@triton.jit
def activation_kernel(x_ptr, out_ptr, n_elements, IS_GELU: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if IS_GELU:
        result = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    else:
        result = tl.maximum(x, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask)
```

When `IS_GELU=True` is passed, the compiled kernel contains **only the GELU branch** — the `tl.maximum(x, 0.0)` line was never traced into IR at all for that specialization. Call the same function with `IS_GELU=False` and you get a second, entirely separate compiled kernel containing only the ReLU line. This is precisely the caching behavior from Chapter 2 and Chapter 3 §3.4 made concrete: **two `constexpr` values → two binaries, each containing only its own branch, with zero runtime branch cost** — you're not paying for an `if` on the GPU at all in this case; you're paying for it in compile time and cache entries instead.

You'll see this pattern used for more than simple flags. It's common to compute a `constexpr` itself via a Python-level conditional expression, based on another compile-time-known property such as a dtype:

```python
v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
```

This line — taken directly from Triton's fused-attention kernel, which you'll implement in Chapter 17 — picks a different memory layout order depending on whether the value tensor is FP8, entirely at trace time. `V.dtype.element_ty` is known at trace time because it comes from the tensor's declared dtype, not from data.

## 6.3 Device-Level Branching: Scalar `if` vs. Elementwise `tl.where`

This is the one that trips people coming from CUDA (where per-thread branching is normal, if costly under divergence) and people coming from NumPy/PyTorch (where `if some_tensor:` is a routine — if occasionally ambiguous — thing to write).

**A Python `if` on a genuinely runtime, *scalar* value works and compiles to a real conditional** in the generated device code. But **a Python `if` on a tensor with more than one element does not work** — Triton, like NumPy and PyTorch, cannot decide what a multi-element tensor's truth value even *means*, and will raise an error rather than guess.

For a per-element decision — "for each lane, pick value A or value B depending on a per-lane condition" — you don't branch at all. You use `tl.where`, which computes **both** sides for every lane and selects elementwise:

```python
# WRONG — mask is a tensor, this raises an error, not a per-lane branch:
if mask:
    y = a
else:
    y = b

# CORRECT — evaluates both `a` and `b` for every lane, then selects per-element:
y = tl.where(mask, a, b)
```

The documentation is explicit that `tl.where`'s two value arguments are **"always evaluated regardless of the value of condition"** — this is worth sitting with, because it means `tl.where` is not free the way a branch that skips work would be; you pay the cost of computing both `a` and `b` for every lane, always. If `a` and `b` are cheap (the usual case — e.g., choosing between `-inf` and a loaded value), this is a non-issue. If one branch is expensive and rarely needed, `tl.where` is the wrong tool, and you should restructure so the expensive path is a genuine (scalar-conditioned or `constexpr`-conditioned) branch instead.

If this feels familiar, it should: it's the exact same trade-off as CUDA warp divergence and predication. A per-thread `if`/`else` where threads in a warp disagree forces the hardware to execute *both* paths serially, masking off the inactive lanes each time — which is mechanically the same "pay for both sides" cost `tl.where` makes explicit and unavoidable at the tile level. Triton doesn't hide this cost from you; it just moves the vocabulary from "warp divergence" to "which lanes does this select."

## 6.4 Loops: `range()`, `tl.static_range()`, and `tl.range()`

All three read like a normal Python `for` loop; they compile very differently.

**Plain Python `range()`** works with either a compile-time-known bound or a genuinely runtime one (e.g., a loop bound computed from `tl.cdiv(K, BLOCK_K)` where `K` is a runtime kernel argument), and compiles down to a real loop construct in Triton's IR — it is *not* guaranteed to be fully unrolled at trace time just because the bound happens to be a Python `int`. Whether the backend further unrolls it is left to the compiler's own optimization passes.

**`tl.static_range(...)`** has the same call signature as `range()`, but explicitly instructs the frontend to **fully unroll the loop at compile time** — every iteration becomes distinct, separately-scheduled instructions in the compiled kernel, with no loop-control overhead at all. Use it when you specifically want that (small, fixed iteration counts where unrolling exposes more instruction-level parallelism), and be aware it will grow your binary size and compile time proportionally to the trip count. It's also the practical fix if you ever hit a confusing compile-time `NameError` or SSA-related error from a variable defined inside a plain `range()` loop and used after the loop exits — a known frontend rough edge that `static_range`'s full unrolling sidesteps entirely, since there's no loop-exit merge point left to get wrong.

**`tl.range(...)`** is the one to reach for when you want a genuine device loop *and* fine control over how the compiler schedules it — this is the tool you'll use for the K-dimension reduction loop in the matmul kernel (Chapter 9) and the Q/K/V tiling loop in attention (Chapter 17). Its extra keyword arguments are compiler directives, not semantic changes to what the loop computes:

- **`num_stages`** — pipeline this specific loop into the given number of in-flight stages. Note this is subtly different from passing `num_stages` as a kernel-launch/autotune parameter (Chapter 8): the launch-level `num_stages` only pipelines loads that feed a `tl.dot`, while `tl.range(..., num_stages=N)` tries to pipeline *most* loads inside that particular loop, whether or not they feed a dot product.
- **`loop_unroll_factor`** — partial unrolling: replicate the loop body this many times per actual loop iteration, trading binary size/compile time for reduced loop-control overhead, without going as far as full `static_range`-style unrolling.
- **`warp_specialize`** — a Hopper/Blackwell-era hint enabling the compiler to assign different warps in a program instance to different roles within the loop (e.g., some warps issuing loads, others computing) — a preview of the manual warp specialization you'll do explicitly in Gluon (Chapter 25).
- **`disable_licm`** — disables loop-invariant code motion for this loop, occasionally useful when the compiler's LICM pass is hoisting something you don't want hoisted (an advanced, rarely-needed escape hatch).

The practical rule of thumb: **write a plain loop first, get it correct, and only reach for `tl.static_range`/`tl.range`'s extra arguments once you're profiling (Part VI) and have a specific reason** — these are performance-tuning knobs, not correctness requirements.

## 6.5 `tl.static_assert`: Catching Bad Specializations at Compile Time

`tl.static_assert(condition)` checks a condition **at compile time**, using only `constexpr`-known quantities, and — unlike a runtime assertion — it doesn't require any debug flag to be enabled; it's always active, because it costs nothing at runtime (it never survives into the compiled kernel at all; it either passes silently during compilation or aborts compilation with an error). The canonical real-world example, taken directly from Triton's fused-attention kernel:

```python
tl.static_assert(BLOCK_N <= HEAD_DIM)
```

This enforces a shape relationship the algorithm depends on (the key/value tile width can't exceed the head dimension for the tiling scheme used) — and it fails *at kernel-compilation time*, with a clear error pointing at the violated invariant, rather than producing a wrong answer or a confusing device-side crash the first time someone launches the kernel with an unexpected `BLOCK_N`/`HEAD_DIM` combination.

Contrast this with a genuine *runtime* assertion (`tl.device_assert`, which does require a debug flag to be active, since it has real runtime cost — it's a device-side check against actual data). Use `tl.static_assert` for anything that depends only on shapes/`constexpr` parameters known at compile time; reach for a runtime assertion only when the invariant genuinely depends on data you can't know until the kernel executes.

## 6.6 The Trade-Off, Made Concrete

Every mechanism in this chapter is a variation on the same trade: **push more decisions to compile time, and you remove runtime branch/loop overhead — at the cost of more compiled variants, longer compile times, and larger binaries.** You now have enough vocabulary to state this precisely instead of vaguely:

| Mechanism | What you gain at runtime | What it costs |
|---|---|---|
| `constexpr` branch (§6.2) | Zero — the untaken branch was never compiled in | One extra cached kernel variant per distinct `constexpr` value combination |
| `tl.static_range` full unroll (§6.4) | No loop-control overhead; more scheduling freedom | Compile time and binary size scale with trip count |
| `tl.range(num_stages=N)` (§6.4) | Overlapped memory latency with compute (software pipelining) | More registers/shared memory live at once — can reduce occupancy (Chapter 15, Chapter 22) |
| `tl.where` (§6.3) | A genuine per-lane decision without divergence | Both branches' cost is paid, always, for every lane |

None of these are free wins — they're all "move the cost somewhere else" tools, and knowing *where* the cost moved to is what lets you use them deliberately rather than by copying a tutorial without understanding why a particular knob was set the way it was.

## 6.7 Hands-On

**Exercise 1 — Confirm branch elimination.** Implement `activation_kernel` from §6.2. Clear your Triton cache, run it once with `IS_GELU=True` and once with `IS_GELU=False`, and confirm two separate cache entries appear (Chapter 2, §2.4). If you're comfortable inspecting IR (Chapter 14 gives you the full toolkit; a rough pass is fine here), dump each specialization's IR and confirm the untaken branch's instructions are genuinely absent from each — not merely predicated off.

**Exercise 2 — Trigger and fix the tensor-`if` error.** Deliberately write `if mask:` where `mask` is a multi-element tensor (as in the "WRONG" snippet in §6.3), run it, and read the actual error Triton raises. Then fix it with `tl.where` and confirm correctness. The goal is to make this error immediately recognizable the next time you see it, rather than something you have to re-debug from scratch.

**Exercise 3 — Loop variants on a reduction.** Implement a kernel that sums a vector in chunks using a device loop over `tl.cdiv(N, BLOCK_SIZE)` iterations, once with plain `range()` and once with `tl.static_range()`. Confirm both produce identical results, then compare compile time for a large fixed trip count (e.g., 64) between the two — you should see `tl.static_range` take measurably longer to compile as the trip count grows, which is the binary-size/compile-time cost from §6.6 made visible.

**Exercise 4 — `tl.static_assert` in practice.** Add a `tl.static_assert` to any earlier kernel enforcing a real invariant it depends on (e.g., `BLOCK_SIZE` being a power of two, needed for `tl.arange` to behave as expected). Deliberately violate it by passing a non-power-of-two `BLOCK_SIZE`, and confirm you get a compile-time error rather than a silent wrong answer or a device-side crash.

## 6.8 Check Your Understanding

1. Why is it meaningless to ask "how much runtime overhead does this `constexpr` `if` add" — what's wrong with the premise of the question?
2. A colleague writes `if attention_mask:` where `attention_mask` is a `(BLOCK_M, BLOCK_N)` tensor, and it fails. Explain *why* it fails (not just "use `tl.where` instead") in terms of what a Python `if` actually requires.
3. You have a loop with a trip count of exactly 4, always. Would you reach for `tl.static_range`, plain `range()`, or `tl.range(loop_unroll_factor=...)`? Justify the choice, including what you'd give up with each alternative.
4. Why does `tl.static_assert` not require any debug flag to be enabled, while a genuine runtime device assertion does?

## 6.9 What's Next

That completes Part II — you now have the full mechanical toolkit: the SPMD model, pointers and masking, multi-dimensional addressing, and compile-time/runtime control flow. Part III starts putting it to work on real, complete kernels: Chapter 7 builds fused softmax, your first kernel that combines a reduction with elementwise work in a single pass — and where the `other=-float('inf')` masking pattern from Chapter 4 finally gets to matter for real.
