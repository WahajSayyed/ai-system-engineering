# Chapter 17 — Flash Attention I: Forward Pass

## 17.1 Cashing In a Foreshadowing From Chapter 7

Chapter 7, §7.2 flagged something and deliberately left it unresolved: fused softmax's "load the whole row into one block" strategy only works because the row fits. Attention's softmax is computed over the **key/value sequence length** — for real workloads (long-context LLM inference, training on long documents) this can be far too large to fit in a single block's worth of on-chip memory. You cannot load the whole reduction axis at once here. You have to **stream** it in chunks, and maintain a **running, incrementally-updated softmax state** as each new chunk arrives — never seeing the "whole row" at any single point in time, and still producing exactly the same answer as if you had. That algorithm is **online softmax**, and this chapter is where it finally earns its keep.

## 17.2 What Each Program Computes

(cite index="53-1">Each Triton program instance computes one block of queries, for one attention head, for one batch element</cite> — the launch grid is `(triton.cdiv(N_CTX, BLOCK_M), batch * nheads)`. Concretely: one program owns a `BLOCK_M`-sized chunk of query positions, for one specific `(batch, head)` pair, and is responsible for producing that chunk's full attention output by streaming across the *entire* relevant key/value sequence.

The query chunk `Q` for this program is loaded **once**, at the very start, and — exactly as the tutorial's own comment states — "stays in SRAM throughout": every subsequent chunk of `K`/`V` streamed past reuses this same resident `Q`, never reloading it. This is the same reuse principle from Chapter 9's matmul tiling (load once, reuse across many multiply-accumulates), applied here to an entire streaming loop rather than a single tile.

Three pieces of state are carried across that streaming loop — this triple **is** the online-softmax algorithm, and understanding how each piece updates correctly, chunk by chunk, is this chapter's actual subject:

- **`m_i`** — the running row-wise max seen *so far* (shape `(BLOCK_M,)`), initialized to `-inf`.
- **`l_i`** — the running softmax denominator sum *so far* (shape `(BLOCK_M,)`), initialized to `1.0` (§17.4 explains precisely why `1.0`, not `0.0`).
- **`acc`** — the running weighted-value accumulator *so far* (shape `(BLOCK_M, HEAD_DIM)`), initialized to zeros.

## 17.3 A Small, Real Optimization: `exp2` Instead of `exp`

Before the main algorithm, one detail worth understanding rather than copying blindly:

```python
qk_scale = sm_scale
qk_scale *= 1.44269504  # 1 / ln(2)
```

The softmax scale is pre-multiplied, once, by `1/ln(2)`, and every exponentiation in this kernel uses `tl.math.exp2` (base-2) rather than `tl.exp` (base-*e*) — mathematically equivalent after this rescaling (`exp(x) = exp2(x * log2(e))`), but genuinely faster on the hardware: base-2 exponentiation maps directly onto a native GPU special-function-unit instruction, while natural-base exponentiation typically requires extra scaling on top of that same underlying primitive. This is a small, concrete instance of a general habit worth having: when a production kernel does something that looks like an arbitrary numerical choice, look for a hardware-mapping reason before assuming it's cosmetic.

## 17.4 The Online Softmax Update, Line by Line

Here is the update applied for each streamed chunk of `K`/`V` (a `BLOCK_N`-sized slice) within the loop:

```python
qk = tl.dot(q, k)                                  # (1) Q @ K^T for this chunk

# --- causal masking, only in the "on-band" call (§17.6) ---
# qk = tl.where(causal_mask, qk, -float("inf"))

m_ij = tl.maximum(m_i, tl.max(qk, axis=1))         # (2) combine old max with this chunk's max
p = tl.math.exp2(qk * qk_scale - m_ij[:, None])    # (3) stable exponentiation vs. the NEW combined max
l_ij = tl.sum(p, axis=1)                           # (4) this chunk's contribution to the denominator

alpha = tl.math.exp2(m_i - m_ij)                   # (5) rescaling factor for ALL previously accumulated state
l_i = l_i * alpha + l_ij                           # (6) rescale old denominator sum, add new contribution
acc = acc * alpha[:, None]                         # (7) rescale old accumulator BEFORE adding new contribution
acc += tl.dot(p.to(v.dtype), v)                    # (8) P @ V for this chunk, added into the rescaled accumulator

m_i = m_ij                                         # (9) commit the updated max for the next iteration
```

Walk through *why* each step is there, not just what it computes:

- **Step (2)** is the heart of "online": you cannot know the row's true global max until you've seen every chunk, so instead you track the max seen *so far* and update it as you go. `m_ij` is never wrong — it's simply the best information available at this point in the stream.
- **Step (3)** is the same numerical-stability move from Chapter 7, §7.3 (subtract the max before exponentiating) — except the max being subtracted is itself a moving target, refined chunk by chunk rather than known in advance.
- **Step (5) is the actual crux of the whole algorithm.** Every value already accumulated into `l_i` and `acc` before this chunk was computed *relative to the old max*, `m_i`. Now that the max has been updated to `m_ij`, that old accumulated state is stated relative to the wrong reference point — it must be rescaled by `exp(m_i - m_ij)` (in base 2: `exp2(m_i - m_ij)`) to remain mathematically consistent with the new reference point, *before* this chunk's new contribution is added in. This is precisely what makes the algorithm correct without ever revisiting a previous chunk: past work is *corrected forward*, never recomputed from scratch.
- **Steps (6)–(8)** apply that correction to both pieces of running state — the denominator sum and the weighted-value accumulator — and then fold in the new chunk's contribution to each.
- **Step (9)** commits the updated max, ready for the next iteration's rescaling to use as "the old max."

**The `l_i = 1.0` initialization, explained precisely.** On the very first loop iteration, `m_i` is still `-inf`. So `alpha = exp2(m_i - m_ij) = exp2(-inf - <finite>) = exp2(-inf) = 0`. That zero multiplies `l_i`'s dummy initial value of `1.0` in step (6) — `1.0 * 0 = 0` — meaning the placeholder value is provably discarded on the very first update, regardless of what it was set to. It's initialized to `1.0` here, not because `1.0` carries any special meaning, but simply because it's guaranteed to be multiplied away before it could ever matter — a detail worth confirming for yourself in this chapter's exercises rather than taking on faith.

**After the loop finishes streaming the entire relevant K/V range**, one final normalization applies what Chapter 7 did per-row in a single step, here done once at the very end of the streamed accumulation: `acc = acc / l_i[:, None]`.

## 17.5 Why Chapter 16 Was a Prerequisite

This kernel contains **two** `tl.dot` calls per loop iteration — `Q @ K^T` (step 1) and `P @ V` (step 8) — with a genuine elementwise/reduction computation (steps 2–7) in between. Exactly the situation Chapter 16 built vocabulary for: the first `tl.dot`'s output lands in an MMA layout; the online-softmax arithmetic between the two dots naturally propagates something layout-adjacent to that; and the *second* `tl.dot` requires its own `dot_operand` layout for `P` and `V`, structurally dictated by the tensor-core instruction being issued — not necessarily the same shape of distribution the first dot's output arrived in. This is precisely why real, performance-tuned Triton attention kernels have meaningful `ttg.convert_layout` activity in their TTGIR around exactly this boundary, and why Chapter 16's vocabulary (layout anchors, conversion cost, `RemoveLayoutConversions`) isn't academic here — it's the difference between a correct attention kernel and a *fast* one. You'll confirm this directly in this chapter's exercises.

## 17.6 Causal Masking, Done Efficiently: The Two-Stage Split

A naive causal implementation would loop over the entire K/V range once, checking a causal condition (`key_position <= query_position`) on *every single chunk*, masking out the (many) chunks that are entirely in the future and need no partial masking, and correctly handling the (one) chunk that straddles the causal boundary. This wastes work: most chunks don't need masking logic at all — they're either entirely valid or entirely invalid, and only exactly one chunk per query block genuinely straddles the boundary.

The tutorial's actual design splits this into two distinct calls, controlled by a `constexpr STAGE`:

```python
if STAGE == 1:
    lo, hi = 0, start_m * BLOCK_M              # "off-band": strictly before the diagonal
elif STAGE == 2:
    lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M   # "on-band": the diagonal block itself
    lo = tl.multiple_of(lo, BLOCK_M)
else:
    lo, hi = 0, N_CTX                            # non-causal: the whole range, no masking anywhere
```

For causal attention, the outer kernel calls this helper **twice**: once with `STAGE=1` (every key position in this range is provably, unconditionally visible to this query block — no per-element masking check is needed *at all*), and once with `STAGE=2` (exactly the one block where query and key positions genuinely interleave, where a real per-element `tl.where`-based mask is necessary). For non-causal attention, the helper is called once with the "process everything, mask nothing" branch. This is a real, named optimization — (cite index="55-1">splitting the K/V loop into non-causal blocks that skip the mask check entirely and a causal boundary block that applies it, specifically to reduce branch/masking overhead</cite> across the large majority of chunks that never needed it in the first place.

**This is also a direct, concrete payoff of Chapter 6.** `STAGE` is a `constexpr`, so the `if STAGE == 1: ... elif STAGE == 2: ... else: ...` branching is resolved entirely at **trace time** (Chapter 6, §6.2) — the compiled kernel for the `STAGE=1` call contains **zero** masking instructions whatsoever, not a masked-off branch that still costs something. The masking code path exists, as compiled machine instructions, only in the one specialization that actually needs it. This is precisely the "two constexpr values, two binaries, only one contains the untaken branch" fact from Chapter 6 — here doing genuinely load-bearing work in one of the most performance-critical kernels in this entire curriculum, not just a toy illustration.

## 17.7 Saving State for Backward: the Log-Sum-Exp Trick

After the streaming loop and final normalization, one more small output is produced: `L_i = m_i + log2(l_i)` — the **log-sum-exp**, saved out per query position for the backward pass (Chapter 18) to reuse, exactly the "forward saves, backward reuses" pattern from Chapter 12. This specific choice — storing one combined value rather than `m_i` and `l_i` separately — is a genuine, documented refinement of the algorithm over time: earlier Triton attention implementations saved both quantities separately, and later versions (cite index="21-1">specifically sped up the forward pass by storing only the LSE instead of `m` and `l` independently</cite>. Worth noting as a small, real example of how these kernels evolve — not every design choice you see in a mature reference kernel was there from the first version; some are measured, incremental refinements.

## 17.8 `tl.static_assert(BLOCK_N <= HEAD_DIM)`, Now in Context

Recall Chapter 6, §6.5's example of `tl.static_assert` — taken, without further explanation at the time, directly from this kernel. Now you have the context: this specific tiling scheme's addressing arithmetic for `K`/`V` chunks depends on a real relationship between how wide a `BLOCK_N` chunk is and the head dimension `HEAD_DIM`, and violating it would silently produce incorrect addressing rather than an obviously-wrong result. `tl.static_assert` catches this at compile time, for every specialization, at zero runtime cost — exactly per Chapter 6's description, now shown doing real work rather than serving as an isolated syntax example.

## 17.9 Hands-On

**Exercise 1 — Non-causal forward pass, in full.** Implement the non-causal path (`STAGE=3`, no masking) completely — query loading, the K/V streaming loop with the full online-softmax update from §17.4, and final normalization. Test against a straightforward PyTorch reference (`torch.softmax(Q @ K.transpose(-1,-2) * scale, dim=-1) @ V`, computed in one shot, no streaming) across a few `(batch, heads, seqlen, head_dim)` configurations. The point of this test isn't just "does it match" — it's confirming that an algorithm computed *incrementally, in chunks, with a running max that changes as it goes* produces **exactly** the same answer as one computed with the whole row visible at once.

**Exercise 2 — Add causal masking.** Implement the two-stage split from §17.6 (`STAGE=1` + `STAGE=2` calls). Test against a PyTorch reference using a causal (lower-triangular) mask, deliberately choosing a `seqlen` that is **not** a multiple of `BLOCK_M`/`BLOCK_N`, so both the diagonal-block causal masking *and* ordinary sequence-boundary masking are exercised simultaneously in the same test.

**Exercise 3 — Confirm the layout-anchor tension from Chapter 16, directly.** Dump the TTGIR for your kernel (Chapter 14, §14.7) and locate the layout attached around each of the two `tl.dot` calls. Identify whether — and exactly where — a `ttg.convert_layout` appears between them, connecting this chapter's concrete kernel back to Chapter 16's abstract discussion.

**Exercise 4 — Quantify what the two-stage causal split actually saves.** Implement a deliberately "un-split" version: a single loop over the *entire* K/V range that checks a causal condition on every chunk (rather than the two-call `STAGE=1`/`STAGE=2` split), and confirm it's still correct. Benchmark it against the properly split version from Exercise 2, and quantify the overhead §17.6 claims the split avoids.

**Exercise 5 — Verify the `l_i = 1.0` trick by hand.** Trace through (on paper, or by adding a debug print via interpreter mode, previewed in Chapter 21) the very first loop iteration's `alpha` value, and confirm it is exactly `0`, making the dummy initial `l_i = 1.0` provably irrelevant to the final result.

**Exercise 6 (exploratory) — Does the initializer value actually matter?** Change `l_i`'s initial value to `0.0` instead of `1.0` and confirm the kernel still produces correct results. This is meant to build a healthy skepticism: not every detail in a reference implementation is load-bearing, and distinguishing "this specific value matters" from "this value was an arbitrary but harmless choice" is itself a skill worth practicing on real code.

## 17.10 Check Your Understanding

1. Explain, without referring back to §17.1, precisely why fused softmax's "whole row in one block" strategy (Chapter 7) cannot be reused directly for attention's softmax.
2. Walk through, in your own words, why the rescaling step `acc = acc * alpha[:, None]` must happen **before** `acc += tl.dot(p, v)`, not after. What would go wrong if the order were reversed?
3. Why does the `STAGE=1`/`STAGE=2` split produce a genuinely different *compiled kernel* for each call, rather than the same kernel executing a runtime branch — and why does that distinction matter for performance?
4. What does storing `L_i = m_i + log2(l_i)`, rather than `m_i` and `l_i` separately, save, and what does this tell you about how to read a mature reference kernel's design choices in general?

## 17.11 What's Next

You've built the forward pass; you haven't yet built its gradient. Chapter 18 tackles flash attention's backward pass — which needs its own careful reduction handling (recomputing values rather than storing everything from forward, similar in spirit to Chapter 11's regenerate-don't-store dropout strategy) and introduces new numerical-stability considerations specific to differentiating through an online, streaming softmax.
