# Chapter 4 — Pointers, Loads, Stores & Masking

## 4.1 What a Pointer Argument Actually Is

When you write:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

`x_ptr` is a **typed device memory address** — Triton infers its pointee type (`float16`, `float32`, etc.) from the PyTorch tensor's dtype at launch time. It is *not* a Python object you can inspect or dereference directly; it only becomes meaningful inside `tl.load`/`tl.store`, or when you do arithmetic on it.

Pointer arithmetic in Triton is elementwise and broadcasting, same as NumPy-style tensor arithmetic:

```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)   # shape (BLOCK_SIZE,), int32
ptrs = x_ptr + offsets                              # shape (BLOCK_SIZE,), tensor of pointers
```

`x_ptr + offsets` doesn't move a single pointer forward — it produces a **tensor of pointers**, one per offset, each pointing to `x_ptr + offsets[i] * sizeof(element)`. Note the arithmetic is in *elements*, not bytes — Triton scales by the pointee's element size for you, matching how C pointer arithmetic works, not how `memcpy`-style byte arithmetic works. This is the object `tl.load` actually consumes.

## 4.2 `tl.load`: Full Anatomy

The full signature you'll be using across the rest of this curriculum:

```python
tl.load(pointer, mask=None, other=None, boundary_check=(), padding_option='',
         cache_modifier='', eviction_policy='', volatile=False)
```

`tl.load` behaves differently depending on what `pointer` is:

- **Single-element pointer** (a scalar pointer) → loads a scalar. `mask`/`other` must also be scalars.
- **N-dimensional tensor of pointers** (what `x_ptr + offsets` produces) → loads an N-dimensional tensor. `mask` and `other` are implicitly broadcast to `pointer.shape`, and `other` is implicitly cast to the pointer's element type. This is the case you've used in every kernel so far.
- **A block pointer** (from `make_block_ptr`, Chapter 5) → a different addressing mode entirely, where `boundary_check` replaces `mask`/`other` (they're mutually exclusive with block pointers — mixing them is an error).

You already know `mask`: *"if `mask[idx]` is false, do not load the data at `pointer[idx]`."* What you haven't examined yet is **`other`**, and it deserves real attention because getting it wrong is one of the most common sources of *silently incorrect* Triton kernels.

### The `other` Argument Is Not Optional in Spirit, Even Though It's Optional in Syntax

If `other` is `None` (the default) and `mask[idx]` is false, **the loaded value at that position is undefined** — not zero, not NaN-safe, literally unspecified garbage from whatever was in the register or was read. If you never *use* the masked-out lanes again (e.g., you immediately mask the corresponding store too), this is harmless. But the moment a masked-out value participates in a **reduction** — a sum, a max, a dot product — undefined becomes a live correctness bug, and it's the kind that often passes on small test cases and fails silently on real workloads with irregular shapes.

Concretely:

```python
# WRONG for a row-max reduction: masked-out (past end-of-row) lanes are undefined,
# and undefined values can corrupt tl.max.
row = tl.load(row_ptr + col_offsets, mask=col_offsets < n_cols)
row_max = tl.max(row, axis=0)   # BUG: garbage lanes may exceed real data

# CORRECT: masked-out lanes are explicitly -inf, which tl.max ignores correctly.
row = tl.load(row_ptr + col_offsets, mask=col_offsets < n_cols, other=-float('inf'))
row_max = tl.max(row, axis=0)   # correct
```

You'll see this exact pattern in the fused-softmax kernel (Chapter 7): `other=-float('inf')` for a max-reduction, `other=0.0` for a sum-reduction. **The right value of `other` depends entirely on the operation the masked-out lanes will subsequently participate in** — it's not a fixed convention, it's a per-kernel correctness decision you have to reason through.

## 4.3 `tl.store`: Anatomy and the Asymmetry with `load`

```python
tl.store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='')
```

Notice `tl.store` has no `other` — it doesn't need one. When `mask[idx]` is false, the store for that lane simply doesn't happen; there's no "what value should the skipped write use" question the way there is for reads. This asymmetry (load needs a fallback value for masked lanes because the *result* is used downstream; store doesn't, because a skipped write just... doesn't write) is worth internalizing so you stop reflexively looking for an `other=` parameter on stores.

## 4.4 `boundary_check`: The Other Masking Mechanism

You'll notice `boundary_check` in both signatures above, alongside `mask`. These are **not interchangeable** — `boundary_check` is specifically for **block pointers** (`make_block_ptr`, Chapter 5), where the compiler tracks tensor bounds automatically and you tell it *which dimensions* to bounds-check rather than supplying an explicit boolean mask tensor yourself:

```python
# Manual mask (what you've done so far) — pointer is a raw tensor-of-pointers:
x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)

# Block pointer (Chapter 5) — bounds are implicit in the block pointer's metadata:
x = tl.load(block_ptr, boundary_check=(0,), padding_option='zero')
```

For now, treat `boundary_check` as a preview — you'll use it once block pointers are introduced next chapter. The mechanism to internalize *today* is manual masking, since it's what you'll reach for most in hand-rolled indexing.

## 4.5 `cache_modifier` and `eviction_policy`: Performance Hints, Not Correctness

Both `tl.load` and `tl.store` accept hints that map to NVIDIA PTX cache-control instructions:

- **`cache_modifier`** on load: `".ca"` (cache at all levels — default-ish behavior), `".cg"` (cache at global/L2 level, skip L1), `".cv"` (don't cache, always fetch fresh — use when you know the data was just modified by another agent and can't trust a cached copy). On store: `".wb"`, `".cg"`, `".cs"` (streaming — for data you won't touch again soon), `".wt"` (write-through).
- **`eviction_policy`**: `"evict_first"` or `"evict_last"` — a hint to the cache replacement policy about whether this data is likely to be reused soon.

These do not affect correctness — only performance, and only on NVIDIA GPUs (they're PTX-specific hints; the ROCm/AMD backend has its own analogous mechanisms). You'll see `eviction_policy="evict_first"` show up in the fused-softmax tutorial (Chapter 7) for input rows that are read once and never revisited — a real, measurable optimization, not a curiosity. We'll return to this with actual profiling evidence in Chapter 22; for now, just recognize the parameters when you see them and don't worry about tuning them yet.

## 4.6 Common Bugs in This Layer

Worth naming explicitly, because these are exactly the bugs that eat the most debugging time for people new to Triton:

1. **Forgetting `mask` entirely** on a kernel where the problem size isn't guaranteed to divide `BLOCK_SIZE` evenly. This usually manifests as either a hard crash (illegal memory access) or — worse — silently wrong results if the out-of-bounds read happens to land on other valid GPU memory.
2. **Wrong (or missing) `other`** feeding a reduction, as in §4.2 — the bug that passes on padded/aligned test inputs and fails on real ones.
3. **Mixing `mask`/`other` with a block pointer.** As noted above, this is an outright error — block pointers use `boundary_check`/`padding_option` instead. The error message is usually clear, but confusing if you don't already know the two mechanisms are mutually exclusive.
4. **Assuming pointer arithmetic is byte-based.** `x_ptr + 1` advances by *one element* of `x_ptr`'s dtype, not one byte. This matters when you start mixing pointers of different dtypes in the same kernel (e.g., an `int32` index tensor alongside an `fp16` data tensor) — the same offset tensor does *not* mean the same byte distance for both.
5. **Reusing an offset tensor across differently-shaped loads without re-deriving the mask.** Easy to do when refactoring; the mask was correct for the original load's bounds, not necessarily for a second load with different valid range.

## 4.7 Hands-On

**Exercise 1 — Break `other` on purpose.** Take the row-max snippet from §4.2, deliberately set `other=0.0` instead of `other=-float('inf')`, and construct an input row where every real value is negative (e.g., `torch.rand(BLOCK_SIZE) - 2.0`, with `n_cols < BLOCK_SIZE` so masking is active). Confirm the reduction now returns the wrong max value (0.0, from the masked-out lanes, rather than the true row maximum) — you want to *see* this bug happen once, deliberately, so you recognize it instantly in the wild.

**Exercise 2 — Triton-Puzzles 3 (gpu-mode/Triton-Puzzles).** This puzzle specifically exercises masked loads/stores where the mask is genuinely non-trivial (not just a single boundary condition) — a good forcing function for the mental model in §4.1–4.2.

**Exercise 3 — Vector-add with explicit `other` variants.** Modify the Chapter 2 vector-add kernel so that instead of `x + y`, it computes `x + y` but where `x`'s masked-out (out-of-bounds) lanes are loaded with `other=100.0` and immediately stored to a *separate debug output* (unmasked, so you can actually see them). Confirm that positions past `n_elements` in this debug output show `100.0`, and that this has *no effect whatsoever* on the real, correctly-masked output — proving to yourself that `other` only matters when the masked lane's value actually propagates somewhere.

**Exercise 4 — Read the generated behavior at a boundary.** For `n_elements = 1000` and `BLOCK_SIZE = 256`, `triton.cdiv(1000, 256) = 4` programs launch, covering `[0, 1024)`. Manually compute, for program `pid=3`, exactly which offsets in its `mask` are `True` vs `False`. Then add a `print`-based sanity check inside a CPU-side wrapper (not inside the kernel — Triton kernels don't support Python `print` for tensor values without interpreter mode, which you'll meet in Chapter 21) that confirms your hand computation against `torch.arange`.

## 4.8 Check Your Understanding

1. Why does `tl.store` have no `other` parameter while `tl.load` does? What does this tell you about the difference between what happens to masked-out lanes on a read vs. a write?
2. You're writing a kernel that computes a masked sum-reduction. What should `other` be, and why would `other=-float('inf')` be wrong here even though it was correct for the max-reduction example?
3. What's the practical difference between passing `mask=` to `tl.load` and passing `boundary_check=` — and why can't you mix them on the same call?
4. `cache_modifier` and `eviction_policy` are described as affecting performance, not correctness. Is there a scenario where a wrong `cache_modifier` choice (e.g., `.cv` — never cache — on data read repeatedly) could make a kernel *catastrophically* slow rather than just suboptimal? Reason about what it does mechanically.

## 4.9 What's Next

Chapter 5 moves from flat, one-dimensional offset tensors to genuinely multi-dimensional tensors — strides, `make_block_ptr`, and the `boundary_check`/`padding_option` mechanism you saw previewed in §4.4. This is the machinery that makes the matrix-multiplication kernel in Chapter 9 possible, and it directly resolves the "2D grid" exercise you did in Chapter 3.
