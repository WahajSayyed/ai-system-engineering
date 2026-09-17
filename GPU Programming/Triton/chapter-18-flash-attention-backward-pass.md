# Chapter 18 — Flash Attention II: Backward Pass & Numerical Stability

## 18.1 Why Three Kernels, Not One

Chapter 17's forward pass was a single kernel. Backward is genuinely different: it's implemented as **three** separate kernels — a small preprocessing kernel, a kernel that computes `dK`/`dV` together, and a kernel that computes `dQ` — rather than one monolithic backward kernel mirroring forward's structure. This isn't incidental complexity; the split between the `dQ` kernel and the `dK`/`dV` kernel specifically (cite index="41-1">simplifies the logic around causal masking loops and reduces the amount of on-chip memory (SRAM) needed per thread block</cite>, compared to fusing everything into one kernel that would need to hold both parallelization schemes' state simultaneously. The preprocessing kernel exists for a different, more subtle reason you'll see in §18.3.

## 18.2 The Central Design Principle: Recompute, Don't Store

Here is the fact that makes this whole chapter possible, and it should feel familiar: **neither forward nor backward ever materializes the full attention-probability matrix** — the conceptual `(N, N)` matrix of softmax outputs you'd get from a naive, unfused implementation. Forward (Chapter 17) only ever saves two things per query position: the actual output `O` (shape `(N, HEAD_DIM)`) and the log-sum-exp `L` (shape `(N,)`, called `M` in the backward-pass code you're about to read — same quantity, renamed; flagging this now so it doesn't read as a new variable). Backward **reconstructs** the attention probabilities on the fly, one `(query-block, key-block)` tile at a time, using only `Q`, `K`, and that tiny saved `M` — never the full matrix, at any point, in either direction.

This should feel like a direct echo of Chapter 11's dropout strategy — regenerate cheaply from a small saved quantity rather than store an entire large tensor — and that's not a coincidence of this curriculum's design; it's the same principle FlashAttention's own authors apply, explicitly, to a different tensor. The original paper states the analogous case for the dropout mask directly: (cite index="44-1">rather than storing the O(N²)-sized dropout mask from the forward pass, the pseudo-random generator states are saved instead, and the mask is regenerated in the backward pass — reducing the extra memory required from O(N²) to O(N)</cite>. This chapter's `P`-matrix recomputation is the identical philosophy, applied to the attention probabilities themselves rather than a dropout mask — a genuinely recurring design pattern across this whole family of kernels, not a one-off trick.

**A genuinely useful clarification about reconstruction here versus the online algorithm in Chapter 17**: rebuilding `P` in backward is *simpler* than computing it in forward. In forward, `m_i` and `l_i` were moving targets, refined incrementally as new K/V chunks streamed in. In backward, `M` (renamed `L_i` from Chapter 17) is already the **final, correct** normalizer for the whole row — there's nothing left to refine. Reconstructing a tile of `P` is a single, direct computation: `p = exp2(qk * scale - M)`, no running max, no rescaling correction, no online update at all. You've already done the hard algorithmic work in Chapter 17; backward gets to use its finished output.

## 18.3 The Math: `dS`, and the `Delta` Identity

The gradient of a softmax-weighted sum with respect to its pre-softmax scores follows a standard identity — for a row of attention scores `S`, probabilities `P = softmax(S)`, and an incoming gradient `dP` on those probabilities:

```
dS_ij = P_ij * (dP_ij - Delta_i),   where Delta_i = sum_j( P_ij * dP_ij )
```

This is the softmax Jacobian applied per row — the same shape of identity underlying any softmax backward, now being applied block-by-block to attention's specific probabilities. Computing `Delta_i` directly, from its definition, would require the full `(N, N)` matrices `P` and `dP` to be simultaneously available for that row — exactly the thing this whole kernel family exists to avoid.

Here's the identity that makes it cheap instead. Recall from Chapter 17 that the forward output is `O_i = sum_j(P_ij * V_j)`, and note that `dP_ij = dO_i · V_j` (the gradient of the output with respect to the probabilities, dotted with the incoming output gradient). Substituting:

```
Delta_i = sum_j( P_ij * dP_ij ) = sum_j( P_ij * (dO_i · V_j) ) = dO_i · sum_j( P_ij * V_j ) = dO_i · O_i
```

**`Delta_i` — a quantity whose definition involves the full attention matrix — turns out to equal a simple row-wise dot product between two tensors you already have in full: `dO` and `O`, both shaped `(N, HEAD_DIM)`, never `(N, N)`.** This is the entire reason the preprocessing kernel exists as its own separate, cheap step: `Delta = rowsum(dO * O)` is computed **once**, up front, from small tensors — precisely because (cite index="41-1">recomputing this particular term the "expensive" way — from the actual attention probabilities — would cost more than recomputing the attention scores themselves would</cite>. This is a genuinely important distinction to internalize: "recompute rather than store" isn't a blanket rule applied identically everywhere in this kernel — `P` itself is cheap to recompute per-tile from `Q`/`K`/`M`, while `Delta` is cheap to *precompute once* via a mathematical identity that sidesteps needing `P`/`dP` in full at all. Both are instances of "avoid materializing the expensive `(N,N)` object," achieved by two different means.

## 18.4 `_attn_bwd_dkdv`: Fixed Key/Value Block, Streamed Queries

This kernel parallelizes across **key/value blocks** — the mirror image of forward's parallelization across query blocks. For a program's fixed `(K, V)` tile, it loops over every relevant block of `Q`:

```python
qkT = tl.dot(k, tl.trans(q))              # recompute scores for this (K-block, Q-block) tile
pT = tl.math.exp2(qkT * qk_scale - m)     # single-shot reconstruction using the saved M — no online update
dv += tl.dot(pT.to(do.dtype), do)         # accumulate this Q-block's contribution to dV
dpT = tl.dot(v, tl.trans(do))             # gradient w.r.t. this tile of P
dsT = pT * (dpT - Delta)                  # the softmax-gradient identity from §18.3
dk += tl.dot(dsT.to(q.dtype), q)          # accumulate this Q-block's contribution to dK
```

(Variable names and exact transpose placements vary slightly across versions; the structure above is the essential shape.) Each term traces directly back to §18.3: `pT` is the recomputed probability tile; `dv`'s update is the direct softmax-weighted-sum gradient (`dV = P^T @ dO`); `dpT` is the raw gradient flowing back through `P`; `dsT` applies the Jacobian correction using the precomputed `Delta`; and `dk`'s update flows from `dS` back through the `Q @ K^T` product.

**A real, easy-to-miss numerical detail worth flagging explicitly**: recall from Chapter 17 that `K` (or, in the backward code, a scaled copy of it) is pre-multiplied by `sm_scale / ln(2)` to enable the `exp2` trick. Because that scaling was folded into `K` *before* this kernel ever sees it, the resulting `dK` comes out scaled by that same folded factor — and must be explicitly **de-scaled** at the end (`dk *= sm_scale`) to represent the true gradient with respect to the *original*, unscaled `K`. This is a genuine, real gotcha class: whenever you fold a constant into an input for a compiler/numerical trick (Chapter 17's `exp2` optimization being exactly such a trick), you must remember to account for that folding on the *gradient* side too, or the result is silently wrong by a fixed multiplicative factor — not obviously broken, not a crash, just quietly incorrect until someone checks the actual numbers.

## 18.5 `_attn_bwd_dq`: Fixed Query Block, Streamed Keys/Values

This kernel mirrors forward's own parallelization exactly — fixed `Q`, streamed `K`/`V`:

```python
qk = tl.dot(q, tl.trans(k))
p = tl.math.exp2(qk * qk_scale - m)
dp = tl.dot(do, tl.trans(v))
ds = p * (dp - Delta)
dq += tl.dot(ds.to(k.dtype), k)
```

The same four quantities from §18.3–§18.4 (`p`, `dp`, `ds`, and the final matmul into the output gradient) reappear here, just accumulating into `dQ` instead of `dK`/`dV`, streamed from the query's perspective rather than the key/value's. If `dQ`'s computation involved any analogous folded scaling, the same de-scaling discipline from §18.4 applies here too — worth checking explicitly in whatever specific implementation you're reading or writing, rather than assuming it's symmetric by default.

## 18.6 Causal Masking, Briefly Revisited

The same two-stage masked/unmasked split from Chapter 17, §17.6 reappears independently within *each* of the `dK`/`dV` and `dQ` kernels (the preprocessing kernel needs no masking at all — `Delta` doesn't depend on which positions are causally valid). Production implementations typically add one further refinement here: subdividing the specific diagonal/boundary region into even finer sub-blocks (via a factor sometimes named `BLK_SLICE_FACTOR`), specifically to minimize how much genuinely-masked, wasted computation happens right at the causal boundary. Treat this as a refinement of a concept you already have, not a new one to learn from scratch — the underlying idea (avoid paying masking cost on blocks that provably don't need it) is exactly Chapter 17, §17.6's.

## 18.7 Hands-On

**Exercise 1 — Verify the `Delta` identity for yourself.** For a small, fully-materializable test case, compute `Delta` two ways: (a) directly from its definition, `sum_j(P_ij * dP_ij)`, using the *full* `(N, N)` probability and gradient matrices (fine to materialize at this small scale, purely for verification); and (b) via `rowsum(dO * O)` as in §18.3. Confirm they match to floating-point tolerance — don't take the identity on faith; derive your own numerical confidence in it.

**Exercise 2 — Implement `_attn_bwd_preprocess` and `_attn_bwd_dkdv` (non-causal first).** Verify `dK` and `dV` against `torch.autograd`'s own gradients for a small, non-causal attention setup, remembering the de-scaling step from §18.4.

**Exercise 3 — Implement `_attn_bwd_dq`, then assemble the full backward.** Verify `dQ` similarly, then wire all three kernels together behind a minimal hand-rolled `torch.autograd.Function` (previewing Chapter 23's formal treatment), and confirm `dQ`, `dK`, and `dV` **simultaneously** match a full, straightforward PyTorch reference computed end-to-end.

**Exercise 4 — Break the de-scaling step on purpose.** Remove the `dk *= sm_scale` correction and rerun your correctness tests. Confirm the failure mode is exactly what §18.4 predicts: not a crash, not an obviously wrong shape or a NaN — a result that's numerically wrong by a fixed multiplicative factor, the specific signature of a forgotten de-scale.

**Exercise 5 — Add causal masking to both backward kernels.** Following the pattern from Chapter 17, §17.6 (as extended in §18.6), add the masked/unmasked split to `_attn_bwd_dkdv` and `_attn_bwd_dq`. Test against a causal PyTorch reference with a sequence length that isn't a multiple of your block sizes.

**Exercise 6 (exploratory) — Measure the actual memory win, not just speed.** Benchmark your full custom forward+backward against PyTorch's own autograd running through a naive, unfused attention implementation (materializing the full `(N, N)` matrix), at a long sequence length. Measure **peak memory usage**, not just wall-clock time — the headline benefit of this entire kernel family is `O(N)` memory instead of `O(N²)`, and this exercise is where you confirm that claim with your own numbers rather than taking it as received wisdom.

## 18.8 Check Your Understanding

1. Explain, in your own words, why reconstructing `P` in the backward pass is described as *simpler* than computing it was in the forward pass, given that both ultimately compute the same softmax probabilities.
2. Derive (or re-derive from memory) why `Delta_i = sum_j(P_ij * dP_ij)` equals `dO_i · O_i`. What specific property of attention's forward computation (`O_i = sum_j P_ij V_j`) makes this substitution valid?
3. Why does the preprocessing kernel exist as a separate step rather than folding `Delta`'s computation inline into the `dK`/`dV` or `dQ` kernels?
4. A colleague's from-scratch reimplementation of this kernel passes all correctness tests at low precision tolerance but fails a stricter tolerance check by a suspiciously *constant* factor across every element. What would you check first, given this chapter?

## 18.9 What's Next

You've now built a complete, numerically-careful forward and backward pass for one of the most important kernels in modern deep learning. Chapter 19 shifts to a different problem entirely: batched matrix multiplication where each batch element has a *different* size — group GEMM — and the persistent-kernel pattern (first previewed all the way back in Chapter 3, §3.6, and used implicitly in Chapter 7's softmax) that makes handling that irregularity efficient.
