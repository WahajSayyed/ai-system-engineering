# Chapter 11 — Randomness & Low-Memory Dropout

## 11.1 What "Low-Memory" Actually Refers To

Standard dropout — zero out each activation independently with probability `p`, scale survivors by `1/(1-p)` so the expected value is unchanged — sounds like a purely elementwise operation, and computationally it is. The memory problem is hiding in the *training* use case specifically: during backpropagation, the gradient flowing through a dropout layer must be zeroed at **exactly the same positions** that were zeroed in the forward pass (mathematically, dropout's backward is `grad_input = grad_output * mask / (1 - p)`, using the identical `mask` from forward). A conventional implementation therefore has to **store the entire mask tensor** generated during the forward pass, so backward can reuse it — a full extra tensor, the same shape as your activations, that has to be written once (forward) and read once (backward), purely as bookkeeping for a computation that is otherwise nearly free.

This is structurally the same kind of problem Chapter 7 solved for softmax: extra memory traffic that isn't inherent to the math, only to a naive implementation strategy. The fix here is different in character, though — it's not about fusing multiple passes over the *same* data into one kernel launch; it's about **never storing the mask at all**, because it turns out you can cheaply regenerate the exact same mask on demand, in the backward pass, without persisting anything but a tiny seed value. That's the actual meaning of "low-memory" in this chapter's title.

## 11.2 Why This Is Possible: Counter-Based PRNGs

The trick rests on a specific *kind* of random number generator. Conventional PRNGs (Mersenne Twister, and most PRNGs you've used from `random`/`numpy.random`) are **sequential**: they maintain a single hidden internal state, and generating the `N`-th random number requires having already stepped through the previous `N-1`. This is a poor fit for a GPU kernel, where thousands of program instances want to generate a random value *simultaneously*, each for its own independent element, with no natural ordering between them and no appetite for synchronizing on shared mutable state.

Triton's PRNG is built on the **Philox** algorithm instead — (cite index="10-1">Triton's implementation of pseudo-random number generation is based on the Philox algorithm</cite>, whose defining property is captured in the title of the paper that introduced it: (cite index="10-1">"Parallel Random Numbers: As Easy as 1, 2, 3"</cite> (Salmon, Moraes, Dror & Shaw, 2011). Philox is **counter-based**: it's a pure function of a `(seed, counter)` pair — no hidden state, no sequential dependency between different counter values. Given the same `seed` and the same `counter` (in Triton's API, an integer *offset*), you always get the same pseudo-random output; different `offset` values, even computed completely independently by unrelated program instances with no coordination whatsoever, give statistically independent outputs. This is precisely what makes it trivially parallel: every one of your kernel's program instances can compute its own random value for its own element by calling the same pure function with its own offset, with zero synchronization overhead and zero shared state to manage.

**The consequence for dropout is direct**: because generating the mask is a pure, stateless function of `(seed, offset)`, the *backward* pass doesn't need the forward pass's mask handed to it at all — it just calls the identical function with the identical seed and offsets, and gets the identical mask back, recomputed rather than retrieved. You trade a full tensor's worth of stored memory for a cheap, already-necessary-anyway PRNG call.

## 11.3 The Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)

    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output
```

`tl.rand(seed, offsets)` — generates a block of uniformly-distributed `float32` values in `[0, 1)`, one per offset, using Philox internally. `x_keep = random > p` gives you a `True`/`False` decision per element with `P(keep) = 1 - p`, and `tl.where(x_keep, x / (1 - p), 0.0)` implements **inverted dropout**: survivors are divided by the keep-probability so the *expected value* of the output equals the input, unchanged — this is exactly why inference-time dropout can just be skipped entirely (identity function) rather than needing its own separate rescaling logic. This mirrors the exact `tl.where` predication mechanics from Chapter 6, §6.3 — computed here for a genuinely per-element, data-dependent condition.

Notice `seed` is passed as an ordinary runtime argument, **not** `constexpr`. There's no reason for it to be — the compiled kernel's logic doesn't change based on the seed's value the way it would for a shape or a feature flag (Chapter 3, §3.4); only the *data* that flows through an already-fixed computation changes. Making `seed` `constexpr` would be a straightforward specialization-cache bug in the making (Chapter 8, §8.3): a distinct compiled kernel — and, if autotuned, a distinct search — for every seed value you ever pass, for a value the kernel's structure never actually depends on.

## 11.4 Determinism, Demonstrated

```python
x = torch.randn(size=(10,), device=DEVICE)
output1 = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)
```

`output1` and `output2` are **bit-identical** — same seed, same offsets, same Philox output, same mask, every time, on every call, with nothing stored or communicated between the two calls beyond the seed itself. `output3`, with a different seed, produces an independent mask. This determinism is a genuine practical feature, not just a curiosity: it means you can reproduce an exact dropout pattern for debugging (fix the seed and a bug either reproduces or doesn't, rather than depending on run-to-run randomness), and it's the entire mechanism that makes the backward-pass mask regeneration in §11.2 possible at all.

## 11.5 Sketching the Backward Pass

You'll build the full, formally-integrated `torch.autograd.Function` version of this in Chapter 23, but the core idea is worth previewing now, since it's the actual payoff of everything in this chapter:

```python
@triton.jit
def _seeded_dropout_backward(grad_output_ptr, grad_input_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)

    random = tl.rand(seed, offsets)          # identical mask, recomputed, not stored
    x_keep = random > p
    grad_input = tl.where(x_keep, grad_output / (1 - p), 0.0)

    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
```

The forward pass needs to persist exactly two scalars for backward to use — `seed` and `p` — rather than a full mask tensor. A recent paper applying this exact pattern in a reinforcement-learning training loop quantifies the saving concretely: (cite index="14-1">storing one RNG seed per rollout, a few bytes, keeps mask-storage cost at O(K · seed) rather than the O(K · T · |H|) that explicit per-step mask materialization would require</cite> — at scale, this genuinely is close to free in memory, which is the whole point of this chapter's title.

## 11.6 Beyond Uniform `[0, 1)`: Other Distributions Exist

`tl.rand` covers the uniform-float case, which is all dropout needs — but it isn't the only primitive Triton exposes. The official tutorial itself points further: (cite index="10-1">if you need it, Triton also provides other random number generation strategies</cite>, and directs you to `python/triton/language/random.py` for the full set (integer variants, and non-uniform distributions) rather than enumerating them all inline in the tutorial. Treat this as a pointer to go looking the moment a future kernel needs something other than uniform floats — dropout is the canonical use case in this curriculum, but not the only place in-kernel randomness matters (data augmentation kernels and certain stochastic-rounding schemes are two others you may encounter later).

## 11.7 A Practical Pitfall: Seed Reuse Across Unrelated Calls

Because Philox is a pure function of `(seed, offset)`, reusing the **same** `seed` for two logically-different dropout applications that also happen to index the same range of `offsets` will produce the **identical** mask for both — almost certainly not what you want if, say, two different dropout layers in the same model happen to process same-shaped tensors and get called with the same hardcoded seed. In practice, derive a distinct seed per call — e.g., combine a per-run base seed with a per-layer index or a call counter — rather than hardcoding one seed value everywhere in your model. This is exactly analogous to a bug class you already know from ordinary software engineering (reusing a nonce or an IV), transplanted into kernel-level PRNG usage.

## 11.8 Hands-On

**Exercise 1 — Reproduce the tutorial's determinism demo.** Implement `_seeded_dropout` and `seeded_dropout` exactly as in §11.3, and reproduce the three-call comparison from §11.4 (`seed=123` twice, `seed=512` once), printing all three outputs side by side. Confirm the first two are bit-identical and the third differs.

**Exercise 2 — Extend to a matrix with one seed per row (official challenge).** Generalize the kernel to operate on a 2D input, using a *vector* of seeds — one per row — rather than a single global seed. You'll need to fold the row index into how you derive each row's effective seed/offset space so that different rows don't accidentally produce correlated masks (a direct instance of the §11.7 pitfall, deliberately constructed rather than stumbled into).

**Exercise 3 — Add stride support (official challenge).** Make your matrix version correct for a non-contiguous input (a transposed or sliced view), using the stride-passing discipline from Chapter 5 rather than assuming row-major contiguity.

**Exercise 4 (advanced, official challenge) — Sparse Johnson-Lindenstrauss transform.** Implement a kernel that performs a random projection *without ever materializing the projection matrix in memory* — generate each entry of the (conceptually huge) projection matrix on the fly from a seed and its `(row, col)` coordinates, exactly the same "regenerate, don't store" principle as this chapter's dropout mask, applied to a different structure entirely. This is a genuinely harder exercise; treat it as optional deep practice rather than a required step.

**Exercise 5 — Wire up and test the backward pass.** Implement `_seeded_dropout_backward` from §11.5, wrap both kernels in a minimal hand-rolled `torch.autograd.Function` (you'll formalize this properly in Chapter 23 — a rough version now is fine), and verify gradients match `torch.nn.functional.dropout`'s own backward pass for the same input and an equivalent (though not identical, since the underlying RNGs differ) dropout mask, by checking the *statistical* properties (fraction zeroed, scaling of survivors) rather than expecting bit-identical masks against PyTorch's own generator.

**Exercise 6 — Reproduce the seed-collision pitfall on purpose.** Deliberately call `seeded_dropout` with the same `seed` for what should be two logically independent dropout applications over same-shaped tensors, and confirm the masks are identical when they shouldn't be for your use case. Fix it by deriving distinct per-call seeds, and confirm the masks become independent.

## 11.9 Check Your Understanding

1. In your own words: what specifically does a *sequential* PRNG require that makes it a poor fit for a GPU kernel, and what property of Philox avoids that requirement?
2. Why is `seed` passed as an ordinary runtime argument rather than `constexpr`, given everything you know from Chapter 3 about when a value should be one or the other?
3. Explain precisely what memory cost this chapter's approach eliminates, and what small cost it accepts in exchange. Is the exchange "free" — is there truly zero downside?
4. What would go wrong, concretely, if two unrelated dropout layers in the same model were called with the same hardcoded seed and overlapping offset ranges?

## 11.10 What's Next

Chapter 12 returns to fused, reduction-heavy kernels — layer normalization — but with a new wrinkle this chapter's softmax didn't need: a genuinely fused **backward pass** that itself requires careful reduction handling (via Welford's algorithm) and atomic accumulation across programs, building directly on both the reduction techniques from Chapter 7 and the forward/backward pairing pattern you just previewed in §11.5.
