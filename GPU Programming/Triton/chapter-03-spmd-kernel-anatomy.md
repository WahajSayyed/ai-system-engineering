# Chapter 3 — SPMD Kernel Anatomy

## 3.1 Recap: What "SPMD" Actually Buys You

Single-Program-Multiple-Data means: you write **one** function, and the hardware runs many independent instances of it concurrently, each operating on a different slice of data. This isn't unique to Triton — CUDA is SPMD too, at the thread level. What's specific to Triton is *where* the "instance" boundary is drawn.

- In CUDA, the instance is a **thread**. You write scalar code; 32 of these threads execute in lockstep as a warp; you manually orchestrate how threads within a block cooperate (shared memory, `__syncthreads()`).
- In Triton, the instance is a **program**, and a program operates on a **tile** (a block of data) using vectorized tensor operations. You never write "thread 17 does X" — you write "this program loads/computes/stores a `BLOCK_SIZE`-shaped chunk," and the compiler decides how many actual GPU threads cooperate to execute that chunk's worth of work.

This chapter is about making that second bullet concrete and mechanical, using the vector-add kernel from Chapter 2 as the running example.

## 3.2 `program_id` and the Grid

```python
pid = tl.program_id(axis=0)
```

Every program instance, when it runs, can ask "which one am I?" — that's what `tl.program_id` answers. It's the direct analog of `blockIdx.x` in CUDA (not `threadIdx.x` — this is the detail people new to Triton most often get wrong). There is no Triton-level equivalent of `threadIdx`; thread-level identity is exactly the thing Triton takes away from you in exchange for the compiler managing coalescing and synchronization.

The grid can have up to **three axes** (0, 1, 2) — the same dimensionality limit as a CUDA launch grid's `(gridDim.x, gridDim.y, gridDim.z)`. `triton.language.num_programs(axis)` gives you the total count of program instances along a given axis — the analog of `gridDim.x`.

```python
pid_m = tl.program_id(axis=0)   # e.g. row-tile index
pid_n = tl.program_id(axis=1)   # e.g. column-tile index
```

You'll use exactly this two-axis pattern starting with the matrix-multiplication kernel in Chapter 9, where each program is responsible for one `(BLOCK_M, BLOCK_N)` output tile, addressed by `(pid_m, pid_n)`.

## 3.3 From `program_id` to a Data Range

The pattern you saw in vector-add generalizes to essentially every kernel you'll write:

```python
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
```

Read this as: *"I am program number `pid`. My job is the half-open range `[pid * BLOCK_SIZE, pid * BLOCK_SIZE + BLOCK_SIZE)`."* `tl.arange(0, BLOCK_SIZE)` materializes a small tensor `[0, 1, ..., BLOCK_SIZE-1]`, added to `block_start` to get absolute offsets. The `mask` exists because `n_elements` is rarely an exact multiple of `BLOCK_SIZE` — the last program's range spills past the end of the array, and every `tl.load`/`tl.store` touching those offsets must be masked or it reads/writes out of bounds.

This is the single most repeated idiom in Triton kernel-writing: **compute your tile's coordinates from `program_id`, build an offset tensor with `tl.arange`, mask for boundaries.** Everything from Chapter 4 onward is a variation on this.

## 3.4 `tl.constexpr`, Precisely

`BLOCK_SIZE: tl.constexpr` is doing more work than it looks like. A `constexpr` argument is baked into the compiled kernel at **compile time**, not passed as a runtime value — which has three consequences:

1. **It can be used where Triton needs a compile-time-known shape.** `tl.arange(0, BLOCK_SIZE)` requires `BLOCK_SIZE` to be known at compile time because the compiler needs to statically determine tensor shapes and layouts (register allocation, vectorization width) — you cannot pass a runtime Python `int` here.
2. **Different values trigger different compilations.** As you saw in Chapter 2, calling `add_kernel[grid](..., BLOCK_SIZE=1024)` and then `BLOCK_SIZE=2048` compiles and caches two distinct binaries. This is *why* autotuning (Chapter 8) works the way it does — it's literally compiling several variants and benchmarking which one wins.
3. **It enables compile-time branching.** Code like `if BLOCK_SIZE >= 1024:` inside a `@triton.jit` function, when `BLOCK_SIZE` is `constexpr`, is resolved and specialized at compile time — dead branches are eliminated entirely, not evaluated at runtime. This is how Triton kernels stay branch-free on the actual GPU despite having Python-level conditionals in the source (more on this in Chapter 6).

A practical rule of thumb: **tile/block dimensions, feature flags (e.g., `IS_CAUSAL: tl.constexpr`), and anything that affects tensor shape or triggers a different code path should be `constexpr`. Actual data (pointers, `n_elements`, alpha/beta scalars) should not be** — making a frequently-varying runtime value `constexpr` would explode your cache with near-duplicate compiled kernels for every distinct value you ever pass.

## 3.5 Mapping a Triton "Program" onto CUDA's Hierarchy

Since you already have the CUDA mental model, here's the direct correspondence — and where it breaks down.

| CUDA concept | Triton concept | Relationship |
|---|---|---|
| Grid `(gridDim.x, .y, .z)` | Launch `grid` tuple | Same dimensionality (≤3), same purpose |
| Block index `blockIdx.{x,y,z}` | `tl.program_id(axis={0,1,2})` | Direct equivalent |
| Thread index `threadIdx.x` | *(no equivalent)* | You don't address individual threads in Triton |
| Thread block (CTA) | **A program instance** | A Triton program compiles down to work executed collectively by the threads of one CUDA thread block |
| `__shared__` memory, manual staging | *(no equivalent — implicit)* | Compiler allocates and manages shared memory when your access pattern needs it |
| `__syncthreads()` | *(no equivalent — implicit)* | Compiler inserts synchronization as needed within a program |
| `num_warps` in `<<<...>>>` config | `num_warps=` kernel-launch argument | Same meaning: how many warps cooperate per program instance/block |

The mental model that resolves most early confusion: **a Triton "program" is roughly one CUDA thread block**, and the `BLOCK_SIZE`-sized tile of data it processes is spread across that block's warps/threads *by the compiler*, using whatever vector width and memory-coalescing pattern it determines is optimal for your access pattern. When you write `x = tl.load(x_ptr + offsets, mask=mask)` for a 1024-element `offsets` tensor, the compiler decides how those 1024 loads get distributed across (say) 4 warps × 32 threads = 128 threads, each handling several elements — you don't see this distribution, but it's happening, and it's exactly the part of CUDA programming Triton is designed to take off your hands.

This is also why `num_warps` matters as a *performance* knob even though it's not part of your kernel's logic: more warps per program means more parallelism *within* a program instance (better latency hiding for memory operations) but also more register/shared-memory pressure shared across those warps. You'll tune this directly starting in Chapter 8.

## 3.6 Beyond One-Tile-Per-Program: `num_programs` and Grid-Stride Loops

Nothing forces a 1:1 mapping between program instances and tiles of work. A common pattern — and the direct ancestor of the **persistent kernel** pattern you'll formalize in Chapter 19 — launches *fewer* programs than there are tiles, and has each program loop over multiple tiles:

```python
@triton.jit
def add_kernel_grid_stride(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)
    # Each program strides across the full problem, BLOCK_SIZE at a time.
    for block_start in range(pid * BLOCK_SIZE, n_elements, num_progs * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)
```

Launched with a *fixed* grid size (e.g., exactly enough programs to saturate the GPU's SM count) rather than one program per tile, this trades launch-overhead for a loop inside the kernel. Why this matters in practice: on kernels with many small tiles, the overhead of launching one program per tile can be significant relative to the work each tile does; a persistent, grid-strided kernel amortizes that overhead. Keep this pattern in the back of your mind — it resurfaces for real in Chapter 19 (Group GEMM & Persistent Kernels) and again in Chapter 25 (warp-specialized persistent matmul).

## 3.7 Hands-On

**Exercise 1 — Read before you write.** Take the vector-add kernel from Chapter 2 and, without changing any code, answer for `size = 98432` and `BLOCK_SIZE = 1024`: how many program instances does `grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)` launch? What range of offsets does the *last* program instance handle, and why is `mask` non-trivial only for that final program?

**Exercise 2 — Triton-Puzzles 1–2 (gpu-mode/Triton-Puzzles).** Puzzle 1 has you write a kernel that adds a constant to every element of a vector using a *single* program instance (no grid parallelism at all) — useful for isolating `tl.load`/`tl.store`/`constexpr` mechanics from grid mechanics. Puzzle 2 reintroduces multiple programs. Do both; they're short.

**Exercise 3 — 2D grid.** Extend vector-add to operate on a 2D matrix (shape `(M, N)`) rather than a flat vector, using a 2D grid: `program_id(axis=0)` indexes row-tiles, `program_id(axis=1)` indexes column-tiles. You'll need `tl.arange` along two dimensions and a 2D mask (`row_mask[:, None] & col_mask[None, :]`). This is deliberately a preview of the indexing machinery formalized in Chapter 5 — expect it to feel slightly unresolved; that's fine.

**Exercise 4 — Grid-stride variant.** Implement the `add_kernel_grid_stride` pattern above, launch it with a small fixed grid (e.g., `grid=(32,)` regardless of `n_elements`), and confirm it still produces correct output for a large vector. Then use `time.perf_counter()` (with proper warm-up and `torch.cuda.synchronize()`, per Chapter 2) to compare its launch overhead against the one-program-per-tile version at a large `n_elements` — the difference will be small for this trivial kernel, but the *pattern* is what you're practicing.

## 3.8 Check Your Understanding

1. Why is `tl.program_id(axis=0)` the analog of `blockIdx.x`, not `threadIdx.x`? What would it mean, mechanically, if Triton *did* expose per-thread identity?
2. A kernel argument is `constexpr`. Someone calls the kernel 50 times in a training loop, each time with a different value for that argument (say, a sequence length that changes per batch). What happens to the kernel cache, and why might this be a performance problem?
3. In the CUDA-to-Triton mapping table, `num_warps` appears on both sides but means something different in each context in terms of *who* controls it. Explain.
4. Sketch (in words, not code) how you'd extend the grid-stride vector-add to two dimensions.

## 3.9 What's Next

Chapter 4 formalizes pointers, `tl.load`/`tl.store`, and masking — the mechanics you've been using in this chapter's examples but haven't yet examined in detail (boundary conditions, the `other=` argument for masked loads, and why the mask pattern generalizes to every kernel with irregular problem sizes). Chapter 5 then extends indexing to genuinely multi-dimensional tensors with strides, which is what Exercise 3 above was foreshadowing.
