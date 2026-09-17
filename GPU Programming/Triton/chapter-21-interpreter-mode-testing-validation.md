# Chapter 21 — Correctness: Interpreter Mode, Testing & Numerical Validation

## 21.1 Why This Chapter, Now

Chapters 17–20 built kernels complex enough that "read the Python source and reason it through" stops being a viable debugging strategy on its own. This chapter formalizes the tools that make correctness tractable at that complexity — several of which would have helped in earlier chapters too, but which earn their keep most clearly now.

## 21.2 Interpreter Mode: What It Actually Does

```bash
TRITON_INTERPRET=1 python your_script.py
```

Setting this environment variable makes Triton bypass compilation entirely: (cite index="63-1">every Triton kernel is instead simulated by an interpreter using NumPy equivalents of Triton's operations, processing each program instance sequentially — one at a time — rather than in parallel</cite>. This single fact — sequential, not parallel — is the source of both this tool's biggest strength and its biggest blind spot, covered in §21.3.

The practical payoff is substantial: because you're now running genuine, ordinary Python/NumPy code rather than a compiled GPU binary, **you get real debugging affordances back**. (cite index="63-1">Plain Python `print()` works directly inside your kernel — `print(tensor)` for a whole tensor, `print(tensor.handle.data[idx])` for individual values</cite> — and, more powerfully still, (cite index="66-1">you can insert real Python breakpoints directly in your kernel code</cite>, dropping into `pdb` exactly as you would in any other Python function. Neither of these is available when debugging a real, compiled, running-on-the-GPU kernel — this is the single biggest reason to reach for interpreter mode first.

## 21.3 What Interpreter Mode Can and Cannot Catch

This is the section worth internalizing precisely, because misunderstanding it leads directly to false confidence.

**It reliably catches**: incorrect indexing and offset arithmetic, wrong masking logic, incorrect algorithm structure — any bug that would be wrong regardless of whether execution happened to be parallel or sequential. For the large majority of bugs in Part III's kernels, this is exactly the right tool, and it's dramatically cheaper than debugging on real hardware.

**It cannot catch race conditions or synchronization bugs — structurally, not as a limitation someone forgot to fix.** Because the interpreter processes program instances one at a time, there is no actual concurrent execution for a race to occur *in*. Recall Chapter 12's lock-based `dw`/`db` accumulation kernel, and the exercise asking you to remove `tl.debug_barrier()` and see what happens: **interpreter mode could never have caught that bug**, no matter how thoroughly you tested under it, because the race it protects against only exists when multiple program instances are genuinely running at the same time. A kernel passing every interpreter-mode test can still have a serious concurrency bug — this is not a hypothetical caveat, it's a specific, concrete gap you've already met in this curriculum.

**It cannot reproduce real hardware numerics precisely, either.** A NumPy simulation does not replicate TF32 truncation (Chapter 10), tensor-core-specific rounding behavior, or other hardware-specific numerical quirks. "Passes under the interpreter" is necessary evidence of correct *logic*, but not sufficient evidence of correct *numerics on real hardware* — you still need a real-hardware test pass for that.

**A further, genuinely current honest caveat**: interpreter mode itself has real, active limitations and bugs, not a perfectly transparent simulation of Triton's semantics. `bfloat16` operations are not directly supported under the interpreter — you must explicitly `tl.cast` a `bfloat16` tensor to `float32` before operating on it there. And, concretely, a documented issue only days old at the time of this writing shows `tl.zeros` failing under interpreter mode with a confusing, unrelated-looking error message — evidence that this tool, like Chapter 20's block-scaled matmul support, is real, useful, actively-developed software with genuine rough edges, not an infallible oracle.

## 21.4 GPU-Side Debugging Primitives: `tl.static_print`, `tl.device_print`, `tl.device_assert`

These work on a **real, compiled, GPU-executing** kernel — not a simulation — which is exactly when you'd reach for them: a bug that only manifests on real hardware (a precision issue, a race condition already localized by other means) needs visibility into the *actual* execution, not the interpreter's stand-in for it.

- **`tl.static_print`** — prints a compile-time-known (`constexpr`) value during compilation itself. Useful for confirming what a specialization actually resolved to, without needing to run anything.
- **`tl.device_print`** — prints genuine runtime tensor values, from the real, executing GPU kernel.
- **`tl.device_assert`** — a genuine runtime check against real data, executing only when `TRITON_DEBUG=1` is set — precisely confirming, now with a direct citation, Chapter 6, §6.5's claim that this differs fundamentally from `tl.static_assert` (which requires no debug flag, because it costs nothing at runtime — it never survives compilation either way).

## 21.5 `compute-sanitizer`: Catching What the Interpreter Structurally Cannot

NVIDIA's `compute-sanitizer` is the tool that closes the exact gap §21.3 identified: it instruments your **real, compiled kernel**, running on **real hardware**, specifically to detect race conditions and memory-access errors (out-of-bounds reads/writes, uninitialized memory) — the class of bug that requires genuine concurrent execution to exist in the first place, which the interpreter cannot provide by construction.

```bash
compute-sanitizer python your_script.py
```

**This is, concretely, the tool you would use to rigorously verify Chapter 12's lock-based `dw`/`db` kernel is genuinely race-free** — a claim interpreter-mode testing is fundamentally unable to establish, no matter how many test cases you throw at it. On AMD hardware, the ROCm-provided LLVM AddressSanitizer serves the analogous role — the same portability theme from Chapters 13, 15, and 16 reappearing here in the tooling domain: the *concept* of hardware-level sanitization exists on both platforms, but the specific tool and invocation differ by vendor.

## 21.6 `triton-viz`: Visualization Built on the Interpreter

`triton-viz` runs your kernels through Triton's interpreter — inheriting both its strengths (no GPU required, vendor-agnostic, genuinely portable for teaching and debugging) and its structural limitation (no real concurrency, so it shares the interpreter's blindness to races). What it adds: (cite index="67-1">visualization, profiling, and memory-safety analysis tools</cite> built on top of that interpreted execution — most usefully, a way to actually **see** which memory addresses your kernel touches and in what pattern, turning Chapter 15's abstract coalescing discussion into something visually inspectable rather than purely conceptual. It also exposes a configurable `TRITON_VIZ_NUM_SMS` setting, letting the CPU-based interpreter emulate multiple concurrent "SMs" — a partial enhancement beyond bare `TRITON_INTERPRET=1`'s single-sequential-instance model, though still not genuine hardware parallelism, and not a substitute for `compute-sanitizer` when a real race is suspected.

## 21.7 A Brief, Honest Mention of the Wider Ecosystem

Beyond Triton's own official tooling, third-party packages have emerged specifically addressing the interpreter's "not your real kernel" limitation — tools that localize a numerical divergence by bisecting the *actual compiled kernel's* output against a reference (across output axes, `program_id` space, or named intermediate values), rather than re-simulating the kernel through NumPy. Worth knowing this category of tool exists as the ecosystem's response to a real, acknowledged gap — treat any specific such package as something to evaluate on its own merits if you encounter a debugging problem it's aimed at, not as endorsed, load-bearing curriculum material.

## 21.8 Formalizing Numerical Comparison: `torch.testing.assert_close`

Earlier chapters used `torch.allclose` loosely; it's worth being precise now about what "close enough" actually means:

```python
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

The underlying check is `|actual - expected| <= atol + rtol * |expected|` — **both** terms matter, and understanding why both are needed is a real testing-design skill, not a formality:

- **`rtol`** (relative tolerance) scales with the magnitude of the expected value — appropriate when you expect error proportional to the value's size (ordinary floating-point rounding behaves this way).
- **`atol`** (absolute tolerance) is a fixed floor — essential specifically for values *near zero*, where a purely relative tolerance would demand unreasonably tight precision on a quantity that's supposed to be small in the first place (`rtol * |expected|` shrinks toward zero exactly when `expected` does).

**Tolerance choice must match the precision actually in use — tie this directly back to Chapter 10.** A kernel computing in `bf16` or using TF32-truncated `tl.dot` inputs will, correctly and expectedly, fail a tolerance appropriate for `float64` comparison — that's not a bug in the kernel, it's the tolerance being wrong for the numerics actually involved. Choosing an appropriately loose tolerance *for the precision genuinely in play* is a deliberate, informed decision — the same "isolate precision from logic before assuming a bug" habit from Chapter 10, §10.8, now formalized as a concrete testing parameter rather than a debugging afterthought.

## 21.9 A Systematic Edge-Case Checklist

This curriculum has been asking you, implicitly, since Chapter 4 to test with irregular shapes, boundary conditions, and adversarial values. It's worth consolidating that pattern into an explicit checklist you can actually run through, rather than reconstruct from memory each time:

- **Non-power-of-two, non-block-size-multiple shapes** — forces masking logic to actually execute (Chapter 4, Chapter 7's `1823 × 781` test).
- **Both** exact-multiple-of-block-size *and* non-multiple shapes — `constexpr`-branched code paths (Chapter 6) can genuinely differ between them; testing only one leaves the other's correctness unverified.
- **Very small inputs** (a single row, a single block) and **very large inputs** (stressing occupancy and persistent-kernel logic, Chapters 7 and 19) — both ends of the size spectrum, not just a "medium, comfortable" size.
- **Adversarial numeric values**: all-identical values, all-negative values feeding a max-reduction (Chapter 4's deliberately-broken `other=` exercise), extreme magnitudes for precision/overflow testing (Chapter 10), and — worth calling out specifically — **zero and near-zero values**, deliberately, every time.
- **Transposed and non-contiguous inputs** (Chapter 5) — confirm a kernel genuinely uses passed-in strides rather than assuming row-major contiguity.
- **Gradient checks against `torch.autograd`** for anything with a backward pass (Chapters 11, 12, 18) — forward-pass correctness alone says nothing about gradient correctness.

**The "zero/near-zero values, deliberately, every time" item deserves its own concrete motivation, not just abstract advice.** Chapter 20, §20.6 described a real, very recently fixed correctness bug in `tl.dot_scaled`: an `e8m0` scale value of exactly zero, or a subnormal-in-bf16 scale, produced **completely wrong results across every element** of the affected test cases. This is exactly the class of bug a systematic "always explicitly test degenerate/zero values" checklist item exists to catch — not a hypothetical cautionary tale, a specific, dated, real one from earlier in this very curriculum.

## 21.10 A Debugging Decision Tree

Putting this chapter's tools into an order you'd actually reach for them:

1. **A kernel gives a wrong answer.** Run it under `TRITON_INTERPRET=1` first — cheapest, no GPU required, catches the large majority of logic bugs.
2. **It fails under the interpreter too** — use `print()`/`pdb` directly inside the kernel (§21.2) to isolate the bug, or `triton-viz` if a memory-access-pattern visualization would help you see the problem rather than just print numbers at it.
3. **It passes under the interpreter but fails on real hardware** — the bug is not in your algorithm's logic. Suspect, in order: a precision/numerics issue (Chapter 10, §10.8 — check your tolerance against the actual dtype in use first), a genuine race condition (reach for `compute-sanitizer`, §21.5 — especially if the kernel involves any locking/atomics, per Chapter 12), or a compiler-stage issue specific to your target hardware (Chapter 14's TTIR/TTGIR/LLVM tracing, especially if the bug only appears on one specific GPU generation).

## 21.11 Hands-On

**Exercise 1 — Find a planted bug with the interpreter.** Take any kernel from earlier in this curriculum, deliberately introduce a subtle indexing bug, and locate it using `TRITON_INTERPRET=1` with real `print()`/`pdb`. Note how this compares, in speed and clarity, to debugging the same class of bug on real GPU hardware directly.

**Exercise 2 — Reproduce the bf16 interpreter limitation.** Run a `bfloat16`-using kernel under `TRITON_INTERPRET=1`, observe the failure described in §21.3, and fix it with the documented `tl.cast`-to-`float32` workaround.

**Exercise 3 — Confirm the interpreter's blind spot, directly.** Take Chapter 12's lock-based LayerNorm backward kernel, remove `tl.debug_barrier()` (as Chapter 12's own Exercise 5 suggested), and this time try to detect the resulting race two ways: first under `TRITON_INTERPRET=1` (confirm it finds nothing, no matter how you test), then — if you have NVIDIA hardware — under `compute-sanitizer`. Making §21.3's claim concrete, with your own evidence, rather than accepting it as asserted.

**Exercise 4 — Visualize memory access with `triton-viz`.** Run one of your existing kernels (Chapter 9's matmul is a good choice) through `triton-viz` and inspect its memory-access visualization. Connect what you observe directly to Chapter 15's coalescing discussion — do the visualized access patterns look coalesced for the configuration you chose?

**Exercise 5 — Tolerance must match precision, demonstrated.** Take a kernel using `bf16` or TF32-truncated `tl.dot` (Chapter 10), write a `torch.testing.assert_close` test with a tolerance appropriately loose for that precision, confirm it passes — then deliberately tighten the tolerance toward `float64`-appropriate levels and confirm it now (correctly) fails. The point is feeling the difference between "the kernel is wrong" and "the test's expectations don't match the numerics in play."

**Exercise 6 — Run the checklist against your own past work.** Build a small, reusable test harness implementing §21.9's checklist, and run it against two or three kernels from earlier chapters that you considered "done." See whether it surfaces anything you hadn't previously caught — genuinely re-examining your own earlier work with systematic rigor, rather than assuming past correctness holds.

## 21.12 Check Your Understanding

1. Explain, in terms of *how* the interpreter executes program instances, precisely why it cannot detect race conditions — not just that it doesn't, but why it structurally can't.
2. A kernel passes every test under `TRITON_INTERPRET=1`. What, specifically, has this established, and what has it *not* established?
3. Why does `torch.testing.assert_close`'s formula need both `atol` and `rtol`, rather than either alone? Construct a scenario where `rtol` alone would be misleading.
4. Using Chapter 20's `e8m0`-zero-scale bug as your example, explain why "test edge cases" is more useful as a concrete, itemized checklist than as general advice.

## 21.13 What's Next

You can now establish correctness with real confidence and real tools, distinguishing logic bugs from race conditions from precision mismatches. Chapter 22 turns from "is it correct" to "is it fast" — profiling with Proton and Nsight Compute/Systems, reading occupancy and register-spill metrics on real hardware (closing the loop on Chapter 15's hand-computed estimates), and a repeatable benchmarking methodology you can trust.
