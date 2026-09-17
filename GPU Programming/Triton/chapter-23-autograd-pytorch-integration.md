# Chapter 23 — Autograd & PyTorch Integration

## 23.1 What This Chapter Formalizes

Chapters 11, 12, and 18 each built a real forward/backward kernel pair — dropout, LayerNorm, flash attention — and each time, wiring them together for testing was informal: a rough, hand-rolled `autograd.Function`, good enough to verify gradients matched a reference, not yet a properly integrated PyTorch operator usable inside a real, trainable model. This chapter formalizes that integration properly, across three tiers of increasing capability — and, just as importantly, tells you honestly when you don't need the more complex tiers at all.

**The three tiers, and PyTorch's own stated preference among them**: (cite index="50-1">prefer using Triton kernels with no `torch.library` custom-operator wrapper at all, because that is simpler — only reach for `torch.library.custom_op` or `torch.library.triton_op` when you specifically need the subsystem integration they provide</cite>. This is worth taking seriously as a default, not just a disclaimer: the added machinery in this chapter is a genuine cost (more code, more API surface, more to get subtly wrong), justified only when you actually need what it buys you.

1. **Plain `torch.autograd.Function`** — the classic API, what Chapters 11/12/18 already approximated informally. Works fine for eager-mode training. **Opaque to `torch.compile`/`torch.export`** — they will never trace into it; it's a black-box boundary.
2. **`torch.library.custom_op`** — formal dispatcher registration (a real schema, declared `device_types`, declared `mutates_args`), enabling proper `register_autograd`/`register_fake`/etc. integration with PyTorch's subsystems. **Still opaque** to `torch.compile`/`torch.export` in the same way as tier 1.
3. **`torch.library.triton_op`** (PyTorch 2.6+) — the current best practice specifically when your op's implementation is one or more Triton kernels. Unlike the tier below it, (cite index="49-1">this makes the implementation visible to `torch.compile` and `torch.export`, allowing them to trace into and optimize the actual Triton kernel launches</cite>, rather than treating the whole operator as an unoptimizable black box.

## 23.2 Tier 1: Plain `torch.autograd.Function`, Done Properly

```python
class SeededDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed):
        output = seeded_dropout(x, p, seed)   # Chapter 11's kernel
        ctx.p = p
        ctx.seed = seed
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = seeded_dropout_backward(grad_output, ctx.p, ctx.seed)  # Chapter 11, §11.5
        return grad_input, None, None
```

This is exactly what Chapter 11, §11.5 gestured at, now shown as the real, complete pattern: a proper class, `.apply()`-based usage, usable inside an actual `nn.Module`. Its limitation is specific and worth naming precisely: if you wrap a model containing this in `torch.compile`, compilation will hit this `Function` and treat it as an opaque node in the graph — correct, but with **zero opportunity for `torch.compile` to fuse anything across its boundary or further optimize the Triton kernel launches inside it.** For eager-mode-only usage, this limitation never matters. The moment `torch.compile` compatibility matters, it does.

## 23.3 Tiers 2–3: `torch.library`, and Why `triton_op` Specifically Matters

`torch.library.custom_op` adds real dispatcher formality — a schema, explicit `device_types`, and a declaration of which arguments the op mutates (`mutates_args`) — enabling PyTorch's subsystems (autograd registration, fake-tensor/meta shape inference for tracing contexts, `vmap` support) to interact with your op correctly. This formality matters even outside `torch.compile`: fake-tensor support, for instance, lets other parts of PyTorch reason about your op's output shape and dtype without ever executing real data through it.

**`torch.library.triton_op` is the one genuinely new capability this chapter is really about.** The registration pattern:

```python
from torch.library import triton_op, wrap_triton

@triton_op("mylib::seeded_dropout", mutates_args={})
def seeded_dropout_op(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    wrap_triton(_seeded_dropout)[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output
```

**The critical, easy-to-miss requirement**: every Triton kernel call inside a `triton_op`-decorated function must go through `wrap_triton(kernel)[grid](...)` rather than calling `kernel[grid](...)` directly. This is precisely what makes the kernel launch itself visible and traceable to `torch.compile`/`torch.export` — omit it, and you've written a `triton_op` that's functionally identical to an opaque `custom_op`, silently losing the entire point of choosing this tier.

### Wiring Up Backward: `register_autograd` and `setup_context`

```python
def setup_context(ctx, inputs, output):
    x, p, seed = inputs
    ctx.p = p
    ctx.seed = seed

def backward(ctx, grad_output):
    grad_input = torch.empty_like(grad_output)
    n_elements = grad_output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    wrap_triton(_seeded_dropout_backward)[grid](grad_output, grad_input, n_elements, ctx.p, ctx.seed, BLOCK_SIZE=1024)
    return grad_input, None, None

seeded_dropout_op.register_autograd(backward, setup_context=setup_context)
```

**A real, concrete constraint on `setup_context` worth understanding precisely, because it directly affects Chapters 17–18's flash attention integration**: (cite index="52-1">`setup_context` receives only `inputs` and `output` — the only quantities it can save are things that are literally among those inputs/outputs, or derived from them. If you need to save a non-input *intermediate* value computed during forward, you must explicitly return it as an additional output from forward, specifically so it becomes available to `setup_context`</cite>.

This is exactly the situation Chapter 17, §17.7 already put you in: flash attention's forward computes and needs to save `L` (the log-sum-exp) for backward — but `L` is an *intermediate* the algorithm computes internally, not a natural input, and not (by itself) the primary output `O`. **The fix is direct**: your `triton_op`-wrapped attention forward must `return O, L` (both, as a tuple) rather than `O` alone, specifically so `setup_context` has access to `L` and can save it via `ctx.save_for_backward(q, k, v, o, L)`. This isn't a new concept — it's Chapter 17's own design, now understood at the level of the formal API constraint that makes it necessary.

**Backward calling its own Triton kernels needs the identical treatment as forward.** Chapters 11, 12, and 18's real backward passes are themselves Triton kernels — `_seeded_dropout_backward`, `_layer_norm_bwd_dx_fused`, `_attn_bwd_dkdv`/`_attn_bwd_dq` — and every one of those calls must also go through `wrap_triton`, exactly as forward's did. The visibility-to-`torch.compile` requirement applies symmetrically to both directions, not just forward.

## 23.4 A Genuine, Documented Current Limitation: Heuristics/Autotune Ordering

Recall Chapter 8, §8.6's decorator-ordering rule: `@triton.autotune` must be the outer decorator, `@triton.heuristics` the inner one (closer to `@triton.jit`). PyTorch's own `torch.compile` integration documentation states essentially the same constraint from its own vantage point: (cite index="48-1">`triton.heuristics` can be used standalone, or before `triton.autotune` — but not after; if both are used together, `heuristics` must come first</cite>. Read together, this isn't two separate rules to memorize — it's the **same underlying constraint**, confirmed to matter for `torch.compile` compatibility specifically, not merely a Triton-internal preference you could safely ignore once `torch.compile` enters the picture.

## 23.5 Testing Formally: `opcheck` and `gradcheck`

Chapters 11, 12, and 18 validated gradients informally — comparing your Triton kernel's computed gradient against `torch.autograd`'s gradient for the *same* mathematical operation. That's a good check, but it has a blind spot: it confirms "I implemented the same math as the reference," not "the gradient I compute is actually the correct derivative of my own forward function." Two more formal tools close that gap:

- **`torch.library.opcheck(op, args)`** — validates a registered custom op for correct usage of the `torch.library` APIs themselves: schema correctness, correct autograd registration, fake-tensor/meta compatibility, and compatibility with dynamic-shape compilation (`aot_dispatch_dynamic`). This is the right tool for confirming your `triton_op` registration itself is sound, independent of whether the underlying math is correct.
- **`torch.autograd.gradcheck()`** — a rigorous, **finite-differences-based** numerical gradient checker: it perturbs inputs by a small amount and directly measures the resulting change in output, comparing that empirical slope against your `backward`'s analytically-computed gradient. This is a genuinely independent check — it doesn't rely on `torch.autograd`'s own computation of the same operation being correct (which is what the informal "compare against `torch.autograd`" approach implicitly assumes); it verifies your specific `backward` implementation is the correct derivative of your specific `forward`, full stop.

## 23.6 `register_fake`, Briefly

For `torch.compile`/`torch.export` to reason about your op's output shape and dtype without ever running real data through it (needed for graph tracing and shape propagation), you register a "fake" implementation — one that computes only the output's shape/dtype/device, using `torch.empty`-style placeholder tensors, without doing any real computation. Treat this as a piece of the full integration picture worth knowing exists; this chapter won't derive it in depth, since the mechanics are a straightforward extension of what §23.3 already covered.

## 23.7 Hands-On

**Exercise 1 — Tier 1, properly.** Wrap Chapter 11's seeded dropout in a real `torch.autograd.Function` (as sketched in §23.2), and confirm it works correctly inside an actual `nn.Module`, in a small training loop.

**Exercise 2 — Upgrade to tier 3, and observe the difference directly.** Reimplement the same dropout kernel using `torch.library.triton_op` + `wrap_triton` + `register_autograd`. Then wrap a small model using each version in `torch.compile`, and confirm you can observe `torch.compile` tracing into (and potentially optimizing) the `triton_op` version's kernel launches, while the plain `autograd.Function` version causes a graph break at that boundary — make the tier 1 vs. tier 3 distinction something you've actually seen, not just read about.

**Exercise 3 — Wire up LayerNorm.** Wrap Chapter 12's LayerNorm forward and backward (including its lock-based `dw`/`db` accumulation) as a `triton_op`, being careful that **every** Triton kernel call in both forward and backward goes through `wrap_triton`.

**Exercise 4 — Confront the `setup_context` constraint directly.** Wrap Chapters 17–18's flash attention forward/backward as a `triton_op`. You'll need to modify forward to `return O, L` rather than `O` alone, specifically so `setup_context` can access and save `L` — a genuine, concrete instance of §23.3's constraint, not a hypothetical one.

**Exercise 5 — Formal validation.** Run `torch.library.opcheck` against your Chapter 11 dropout `triton_op`, and `torch.autograd.gradcheck` against its backward. Check whether either surfaces anything your earlier, informal manual testing (Chapters 11/12/18's own exercises) hadn't caught.

**Exercise 6 — Reproduce and fix the heuristics/autotune ordering limitation.** Deliberately stack `triton.heuristics` and `triton.autotune` in the wrong order (heuristics outer, autotune inner) inside a `triton_op`-wrapped kernel, confirm the failure §23.4 describes, then fix the ordering and confirm it resolves — connecting Chapter 8's rule to a concrete, current `torch.compile`-specific consequence you've triggered yourself.

## 23.8 Check Your Understanding

1. Why does PyTorch's own documentation recommend using plain, unwrapped Triton kernels by default, reserving `custom_op`/`triton_op` for when their specific integration is actually needed? What's the cost of reaching for `triton_op` unconditionally?
2. Explain, precisely, why `wrap_triton` is required around every kernel call inside a `triton_op`-decorated function, and what would silently be lost if you forgot it on just one call.
3. Using flash attention's `L` as your example, explain why `setup_context`'s restriction to "inputs and outputs only" forced a specific change to how Chapter 17's forward function returns its results.
4. What does `torch.autograd.gradcheck` verify that comparing your kernel's gradient against `torch.autograd`'s own gradient for the same operation does not?

## 23.9 What's Next

You can now wire a Triton kernel into a real, trainable, `torch.compile`-compatible PyTorch model properly. Chapter 24 looks at the other direction of this same relationship: how `torch.compile`/TorchInductor *automatically* generates Triton kernels from an ordinary PyTorch model, without you writing any Triton at all — and, concretely, how to tell when that automatic generation is good enough versus when a hand-written kernel like the ones you've built throughout this curriculum is still the better choice.
