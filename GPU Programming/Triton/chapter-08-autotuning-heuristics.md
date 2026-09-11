# Chapter 8 — Autotuning & Heuristics

## 8.1 Why This Chapter Exists

Every kernel so far has had its `BLOCK_SIZE`, `num_warps`, and (in Chapter 7) `num_stages` chosen by you, once, by hand. That's fine for a vector-add — the optimal block size barely matters across shapes. It stops being fine the moment the optimal configuration genuinely *depends* on the problem size and the specific GPU you're running on, which is exactly the situation you'll hit in Chapter 9's matrix multiplication: the best tile sizes for a `(512, 512) @ (512, 512)` matmul on an A100 are not the best tile sizes for a `(8192, 8192) @ (8192, 8192)` matmul on an H100. Hand-picking one configuration and shipping it means leaving real performance on the table for every shape/hardware combination that configuration wasn't tuned for.

Autotuning is Triton's answer: **describe a search space of candidate configurations, and let the runtime benchmark all of them the first time it sees a given problem shape, caching the winner** so every subsequent call with that same shape just uses the answer directly. You already have every prerequisite for understanding *how* this works — it's built entirely on mechanisms from Part I: `constexpr` specialization (Chapter 3, §3.4) and the JIT compilation/caching lifecycle (Chapter 2, §2.3–2.4). Autotuning doesn't introduce new compiler machinery; it's an outer loop that compiles and benchmarks *several* specializations and keeps the best.

## 8.2 `triton.Config`: One Candidate Point in the Search Space

```python
triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4)
```

A `Config` bundles exactly the launch parameters you've been setting by hand since Chapter 2: a dict of `constexpr` kernel arguments (`kwargs`), plus `num_warps` and `num_stages`. Each distinct `Config` corresponds to a distinct compiled specialization — this should feel familiar, since it's the same "different `constexpr` values → different cached binary" fact from Chapter 3, applied to a whole list of candidates rather than one hardcoded choice.

## 8.3 `@triton.autotune`: Full Mechanics

```python
@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['x_size'],
)
@triton.jit
def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    ...
```

The `key` argument is the piece that determines *when* re-tuning happens: it's a list of kernel argument names whose values, taken together, form a cache key. **The first time the kernel is called with a given combination of `key` values, every config in the list is compiled and benchmarked, and the fastest one is recorded against that key** in an in-memory dictionary the `Autotuner` maintains for the lifetime of the process. Every subsequent call with the *same* key values skips the search entirely and just runs the previously-found winner. Change `x_size` to a value not seen before, and the full search runs again for that new key.

This is a direct generalization of the JIT specialization behavior from Chapter 2 — instead of one cache entry per `(dtype, constexpr-values, compute-capability)` tuple, you now also get one *search result* per distinct value of whatever arguments you name in `key`. Choose `key` to be exactly the runtime values that plausibly change which config is optimal (problem shape is almost always the right choice; something like a scalar `alpha` coefficient almost never should be, since it's unlikely to change which tile size wins and would otherwise blow up the number of searches you pay for).

**A precision worth correcting from how Chapter 2 described this in passing**: the autotuning result cache described above is, by default, an **in-memory, per-process** dictionary — it is *not* automatically persisted to disk the way compiled kernel binaries are (Chapter 2, §2.4). If you restart your Python process, the autotuning search runs again from scratch on first use, even though the underlying *compiled kernels* it discovered before are still sitting in the on-disk `~/.triton/cache`. If you want the search results themselves to survive process restarts, pass `cache_results=True` to `@triton.autotune` explicitly — this is an opt-in, not the default.

### Watching It Happen

```bash
TRITON_PRINT_AUTOTUNING=1 python your_script.py
```

Setting this environment variable makes Triton print, to stdout, how long the autotuning search took and which configuration won, every time a new search is triggered. This is the single most useful debugging habit to build in this chapter — turn it on the first time you run any autotuned kernel, so you're never wondering *whether* a search happened, or surprised by a mysterious pause on first launch.

## 8.4 `reset_to_zero` and `restore_value`: Correctness of the Benchmark Itself

Here's a subtlety that has nothing to do with whether your *kernel* is correct, and everything to do with whether *benchmarking it fairly* is correct. Autotuning works by running your kernel many times — once per config, and typically several repetitions per config for stable timing. If your kernel has a **side effect** — it accumulates into an output via `tl.atomic_add`, for instance, rather than overwriting it — then running it repeatedly during the search means each repetition sees the *already-modified* output from the previous repetition, not a clean starting state. The measured correctness (and sometimes the measured *timing*, if accumulation changes control flow) becomes contaminated by search order, not a fair per-config comparison.

Two hooks solve this:

- **`reset_to_zero=['output_ptr']`** — zeroes the named tensor argument(s) before every single benchmarked run of every config, guaranteeing each config starts from a clean slate.
- **`restore_value=['some_tensor']`** — more general: snapshots the named tensor(s) before a run and restores the original contents afterward, for cases where "reset to zero" isn't the right clean state (e.g., an in-place update kernel where the correct starting state is the *original input*, not zero).

The rule of thumb: **any kernel argument your kernel mutates in a way that isn't a simple overwrite needs one of these two hooks**, or your autotuning search is silently measuring (and potentially selecting) the wrong thing.

## 8.5 `prune_configs_by`: Taming a Combinatorial Search Space

A realistic matmul autotuning config list (Chapter 9) searches over `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `GROUP_M`, `num_warps`, and `num_stages` simultaneously — the cross product can easily reach dozens of candidates, each requiring a real compilation and a real benchmarked run the first time a new shape is seen. `prune_configs_by` lets you cut this down *before* paying the full compile-and-bench cost for every candidate:

- **`early_config_prune`** — a function you supply that filters the config list down using cheap, known-bad-combination logic (e.g., discard configs where `BLOCK_K` exceeds `BLOCK_M` for architectural reasons you already know rule them out) — before any compilation happens at all.
- **`perf_model` + `top_k`** — a cheap analytical cost model estimates each surviving config's likely running time, and only the estimated best `top_k` are actually compiled and benchmarked for real. This trades a small risk of missing the true optimum (if the cost model is imperfect) for a large reduction in first-call latency.

You won't need this machinery for small kernels like vector-add or softmax — it becomes worth reaching for exactly when your config list starts costing real, noticeable time to search in full, which is squarely a Chapter 9 concern.

## 8.6 `@triton.heuristics`: The Skip-the-Search Door

Not every `constexpr` value benefits from being *searched* — some can be **derived directly** from other arguments with a simple formula, with no ambiguity about what the right value is. Forcing such a value into the autotuning search space wastes compile time benchmarking configs that differ only in a value you could have just computed. `@triton.heuristics` exists for exactly this case:

```python
@triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
@triton.jit
def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    ...
```

`values` is a dict mapping a `constexpr` argument name to a function that computes it from a dict of the kernel's *other* arguments, indexed **by name** (`args['x_size']`, not by position — older examples floating around online index positionally and are outdated/incorrect against the current API; if you copy a snippet using positional indexing, rewrite it to use argument names). This is exactly the logic behind Chapter 7's `BLOCK_SIZE = triton.next_power_of_2(n_cols)` — there is no meaningful "search" to do there; the value is fully determined by `n_cols`, and `@triton.heuristics` is the idiomatic way to express "compute this, don't search it" as a declarative decorator rather than inline host-side Python.

### Combining Both, and the Decorator Order That Matters

Autotuning and heuristics compose — search over what genuinely needs searching, derive what doesn't:

```python
@triton.autotune(
    configs=[
        triton.Config(kwargs=dict(BLOCK_SIZE_ROWS=r, num_stages=s), num_warps=w, num_stages=s)
        for r in (16, 32, 64, 128) for s in (2, 3, 4) for w in (2, 4, 8)
    ],
    key=['N_COLS'],
)
@triton.heuristics(values=dict(
    BLOCK_SIZE_COLS=lambda args: triton.next_power_of_2(args['N_COLS']),
))
@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_rows,
                    N_COLS: tl.constexpr, BLOCK_SIZE_ROWS: tl.constexpr,
                    BLOCK_SIZE_COLS: tl.constexpr, num_stages: tl.constexpr):
    ...
```

Here, `BLOCK_SIZE_ROWS`, `num_warps`, and `num_stages` are genuinely searched (their optimal values plausibly depend on hardware and problem size in ways not reducible to a formula), while `BLOCK_SIZE_COLS` is derived deterministically and never searched at all. **Decorator order is not cosmetic**: `@triton.autotune` must be the outer decorator and `@triton.heuristics` the inner one (closer to `@triton.jit`) — reversing them raises an error, because the heuristics-computed values need to already exist before the autotuner's config-benchmarking logic runs. If you see this ordering constraint violated in older tutorial code, it's a genuine bug, not a style choice.

## 8.7 The Real Cost: First-Call Latency, Multiplied

Recall from Chapter 2 that a single kernel specialization already pays a JIT-compilation tax on first call. Autotuning multiplies that tax by the number of configs actually benchmarked (after any pruning from §8.5) for every *new* `key` value your kernel encounters. In an interactive/research setting this is usually a non-issue — you pay it once per shape and move on. In a **production inference service**, where a new request shape might trigger a fresh search mid-request, this can show up as a real, user-visible latency spike the first time a shape is seen.

The standard mitigations, worth knowing even before you hit this problem for real:
- **Pre-warm** the service by calling the kernel once with every shape you expect to see in production, during startup, before accepting real traffic.
- **Enable `cache_results=True`** (§8.3) so the search results themselves persist across process restarts, not just the compiled binaries.
- **Constrain `key`** to genuinely necessary dimensions, and bucket/round problem sizes where possible (e.g., pad sequence lengths to the nearest power of two) so you encounter far fewer distinct keys in practice.

## 8.8 Hands-On

**Exercise 1 — Autotune vector-add.** Take the Chapter 2 vector-add kernel and wrap it in `@triton.autotune` with 4–5 `Config`s varying `BLOCK_SIZE` and `num_warps`, keyed on `n_elements`. Run with `TRITON_PRINT_AUTOTUNING=1` and confirm you can see the search happen once per distinct size, and observe that a second call with the same size skips straight to the cached winner (no printed search message).

**Exercise 2 — Autotune + heuristics on Chapter 7's softmax.** Rewrite the fused-softmax kernel using the pattern in §8.6: autotune over `num_warps`/`num_stages` combinations keyed on `n_cols`, and derive `BLOCK_SIZE` via `@triton.heuristics` rather than computing it in the host wrapper as Chapter 7 did. Confirm correctness is unchanged, then benchmark against Chapter 7's hand-picked fixed configuration (`num_warps=8, num_stages=4`) across a range of `n_cols` values — does autotuning find a better configuration for any of them? Report where it does and doesn't matter.

**Exercise 3 — Break a benchmark on purpose, then fix it.** Write a small kernel that accumulates into its output via `tl.atomic_add` rather than `tl.store`. Autotune it *without* `reset_to_zero`, and construct a case where the reported "fastest" config is actually influenced by leftover accumulated state from a previous config's benchmarking runs (this may show up as a correctness assertion failing after autotuning, even though the exact same kernel logic works fine when called normally without autotuning). Add `reset_to_zero=['output_ptr']` and confirm the problem disappears.

**Exercise 4 — Measure the multiplied first-call cost.** Time the very first call to an autotuned kernel with `N` configs in its list, versus the same kernel hardcoded to a single fixed config (no autotuning at all), for a few values of `N` (e.g., 2, 8, 32 configs). Confirm the first-call cost scales roughly with the number of configs actually benchmarked, and that the *second* call (same shape) is essentially identical between the two versions.

## 8.9 Check Your Understanding

1. Why is choosing `key` correctly (not too broad, not too narrow) important for autotuning to actually be useful in practice? What goes wrong in each direction — too many distinct keys vs. too few?
2. Why does the *default* autotuning result cache not survive a process restart, even though the underlying compiled kernel binaries do?
3. Give a concrete example (not from this chapter) of a `constexpr` value that should be handled with `@triton.heuristics` rather than included in the `@triton.autotune` search space, and explain why searching it would be wasteful.
4. A kernel writes its entire output via `tl.store` (full overwrite, no accumulation) on every call. Does it need `reset_to_zero` or `restore_value` for correct autotuning? Justify your answer from the mechanism in §8.4, not just intuition.

## 8.10 What's Next

You now have the full toolkit needed to make Chapter 9's matrix-multiplication kernel practical rather than merely correct: tiled addressing (Chapter 5), compile-time specialization (Chapter 6), and now a principled way to search tile sizes rather than guess them. Chapter 9 builds the tiled GEMM algorithm itself — L2-cache-aware tile ordering, the K-dimension reduction loop, and a realistic autotuning config space that puts everything from this chapter to direct, load-bearing use.
