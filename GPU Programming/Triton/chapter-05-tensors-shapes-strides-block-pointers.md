# Chapter 5 — Tensors, Shapes, Strides & Block Pointers

## 5.1 From a Flat Vector to a Real Tensor

Every kernel so far has treated memory as one long flat array. Real workloads are matrices and higher-rank tensors, and PyTorch tensors aren't necessarily laid out the way their logical shape suggests — a transposed view, a sliced batch, a non-contiguous tensor from a previous op, all have the *same* logical shape but different physical memory layouts. The concept that bridges "logical shape" and "physical memory address" is the **stride**.

**A stride tells you how many elements to skip in memory to move one step along a given dimension.** For a row-major (C-contiguous) matrix of shape `(M, N)`, the stride is `(N, 1)`: moving one step along the row dimension (dimension 0) skips `N` elements (a full row); moving one step along the column dimension (dimension 1) skips 1 element (adjacent in memory). PyTorch exposes this directly: `tensor.stride(0)`, `tensor.stride(1)`.

This matters immediately because **you always pass strides into your kernel explicitly** — Triton has no built-in notion of "this tensor is row-major" or "this tensor is transposed." It only knows what you tell it via the stride arguments.

## 5.2 Manual 2D Indexing: The Workhorse Pattern

This is the pattern underlying the matrix-multiplication kernel in Chapter 9, the attention kernel in Chapter 17, and honestly the majority of real Triton kernels you'll encounter — internalize it here rather than re-deriving it each time.

```python
@triton.jit
def row_sum_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_m, stride_n,          # x.stride(0), x.stride(1)
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)                          # one program per row
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    # Address of element [row, col] = base + row*stride_m + col*stride_n
    row_ptr = x_ptr + row * stride_m
    x = tl.load(row_ptr + col_offsets * stride_n, mask=mask, other=0.0)

    tl.store(out_ptr + row, tl.sum(x, axis=0))
```

Read `row_ptr + col_offsets * stride_n` as: *"start at the beginning of this row, then step `stride_n` elements per column."* For a contiguous row-major matrix `stride_n == 1`, so this collapses to the flat-vector pattern you already know — the stride machinery is there precisely for the cases where it *doesn't* collapse (transposed inputs, strided slices, batched tensors).

### Two Dimensions at Once: Broadcasting with `None`

When a single program needs to address a full 2D **tile** (not just one row), you build the offset tensor for each axis separately and broadcast them together — exactly like NumPy:

```python
row_offsets = tl.arange(0, BLOCK_M)          # shape (BLOCK_M,)
col_offsets = tl.arange(0, BLOCK_N)          # shape (BLOCK_N,)

ptrs = (x_ptr
        + row_offsets[:, None] * stride_m     # shape (BLOCK_M, 1)
        + col_offsets[None, :] * stride_n)    # shape (1, BLOCK_N)
        # broadcasts to shape (BLOCK_M, BLOCK_N)

mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
tile = tl.load(ptrs, mask=mask, other=0.0)
```

`row_offsets[:, None]` reshapes to a column `(BLOCK_M, 1)`; `col_offsets[None, :]` reshapes to a row `(1, BLOCK_N)`. Adding them broadcasts to a full `(BLOCK_M, BLOCK_N)` grid of addresses — one pointer per element of the tile, computed in a single vectorized expression rather than a nested loop. This is the exact mechanism you half-built in the Chapter 3 "2D grid" exercise; now you have the full tool.

The 2D mask follows the same broadcasting logic: `row_offsets[:, None] < M` is `True`/`False` per row, `col_offsets[None, :] < N` is `True`/`False` per column, and `&` combines them elementwise so a tile element is valid only if *both* its row and column are in bounds — correctly handling tiles that spill past the matrix edge in either dimension independently.

## 5.3 Strides Make Transposes Free

Because addressing is entirely stride-driven, a "transposed" view of a matrix costs nothing to express — you just swap which stride goes with which offset tensor:

```python
# Normal (M, N) row-major access:
ptrs = x_ptr + row_offsets[:, None] * stride_m + col_offsets[None, :] * stride_n

# The SAME underlying memory, addressed as if it were (N, M) — no data movement:
ptrs_T = x_ptr + col_offsets[:, None] * stride_n + row_offsets[None, :] * stride_m
```

This is exactly how `torch.Tensor.t()` / `.transpose()` work under the hood (a metadata-only operation — strides are swapped, no bytes move), and it's why passing a PyTorch tensor's `.stride()` values into your kernel — rather than assuming row-major — makes your kernel correct for transposed and non-contiguous inputs "for free." Get in the habit of never hardcoding an assumption like "column stride is always 1"; always take strides as kernel arguments.

## 5.4 Block Pointers — Deprecated as of Triton 3.7

For several years, Triton offered a second, higher-level way to express exactly the addressing pattern in §5.2: `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)`. It packaged a tile's shape, strides, starting offset, and a coalescing hint (`order`) into one object, paired with `boundary_check=`/`padding_option=` instead of manual `mask`/`other`. You'll still see it in a large fraction of existing tutorials, blog posts, and production kernels written before mid-2026.

**As of Triton 3.7 (current stable), `make_block_ptr` is deprecated** — it now emits a deprecation warning, and on Triton's development branch it has already been removed entirely, redirecting users to the **tensor descriptor** API described below. If you're reading someone else's kernel using `make_block_ptr`, you can still understand it directly from the manual-indexing mental model in §5.2 — it was always a convenience wrapper over the same underlying addressing math, not a different execution model.

## 5.5 Tensor Descriptors — The Current Replacement

`tl.make_tensor_descriptor` is the API you should reach for going forward when you want the "describe a tile once, load/store it declaratively" convenience that `make_block_ptr` used to provide:

```python
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))
```

The important architectural point: **on NVIDIA GPUs with TMA (Tensor Memory Accelerator) support — Hopper and Blackwell — a tensor descriptor compiles down to an actual TMA descriptor**, and `desc.load`/`desc.store` become hardware-accelerated asynchronous copy operations, not just a syntactic convenience. This is a genuinely different (and faster) execution path on that hardware, not merely cleaner syntax — you're directly meeting the machinery covered in depth in Chapter 25 (Warp Specialization, TMA & Gluon). On older GPUs (Ampere and earlier) or the AMD/Intel backends, it lowers to conventional addressed loads/stores, functionally equivalent to §5.2's manual pattern.

Two setup details worth knowing now so they don't surprise you later:
- Tensor descriptors need a **global memory allocation** for descriptor storage, which means you must register an allocator once per process: `triton.set_allocator(alloc_fn)` (shown in the snippet above via `alloc_fn`).
- `strides` for the leading dimensions must be multiples of 16 bytes, and the last dimension must be contiguous — a real hardware constraint of the TMA unit, not an arbitrary API restriction. If your tensor doesn't satisfy this, you fall back to manual indexing.

### Where This Curriculum Uses Which

- **Chapters 7–9 (softmax, matmul fundamentals):** manual pointer arithmetic (§5.2). It's the most transparent way to *learn* what's happening, works identically on every backend, and has no setup overhead.
- **Chapter 17 onward (fused attention) and Chapter 25 (Gluon/TMA):** tensor descriptors, where the TMA hardware path is actually the point of the exercise.

Don't feel you need to master tensor descriptors today — the goal of this section is that you recognize them, understand *why* they exist (a real hardware feature, not API churn for its own sake), and know why `make_block_ptr` shows up in older material you'll inevitably read.

## 5.6 The `order` Argument: A Coalescing Hint

Both `make_block_ptr` (historically) and the layout metadata behind tensor descriptors accept an **`order`** — a permutation telling the compiler which dimension is physically contiguous, used to optimize memory coalescing. A useful mental trick: `order` lets you describe a "virtual transpose" at zero cost. If a tensor `k` is stored row-major with shape `(T, K)` but you want to *address* it as though it were `(K, T)` (common in attention kernels, where you want K^T without materializing a transposed copy), you swap both the `shape`/`strides` *and* the `order` to match — the compiler then knows which axis to prioritize for coalesced access, even though the "logical" view has been flipped. We'll use this concretely in Chapter 17.

## 5.7 Hands-On

**Exercise 1 — Manual row-wise kernel.** Implement `row_sum_kernel` from §5.2 fully, test it against `torch.sum(x, dim=1)` for a matrix with `N` **not** a power of two and **not** a multiple of your `BLOCK_N` (forces the mask to matter), and confirm correctness.

**Exercise 2 — 2D tile load with broadcasting.** Implement a kernel that loads a `(BLOCK_M, BLOCK_N)` tile from an `(M, N)` matrix using the full broadcasting pattern in §5.2 (not one-row-per-program), and writes the tile unchanged to an output matrix. Test with `M, N` that don't divide evenly by `BLOCK_M, BLOCK_N`, confirming the 2D mask correctly excludes out-of-bounds rows *and* columns independently — construct a test case where a tile is partially valid in both dimensions simultaneously (not just one).

**Exercise 3 — Prove transposes are free.** Take a PyTorch matrix `x` of shape `(M, N)`, create `x_t = x.t()` (a view, not a copy), and confirm `x_t.stride()` is the reverse of `x.stride()`. Then write a single kernel that can process *either* `x` or `x_t` correctly by simply passing in the tensor's own `.stride(0)`/`.stride(1)` — no code branching on which one you were given.

**Exercise 4 — Triton-Puzzles 2D indexing puzzles (gpu-mode/Triton-Puzzles).** Work through the puzzles involving matrix/2D inputs — they're specifically designed to exercise the broadcasting pattern from §5.2 under slightly awkward shapes.

**Exercise 5 (optional, requires Hopper/Blackwell) — Tensor descriptors.** If you have access to an H100/B200-class GPU, run the `inplace_abs` example from §5.5 verbatim, then modify it to operate on a matrix whose last dimension is *not* contiguous (e.g., a transposed view) and observe what happens — you should either get a clear error (violated contiguity requirement) or a fallback path, which is itself instructive about the hardware constraint in §5.5.

## 5.8 Check Your Understanding

1. Why does Triton need you to pass `stride_m`/`stride_n` explicitly rather than inferring them from `M`, `N` alone?
2. Explain, in terms of strides, why `x.t()` in PyTorch is "free" (no memory copy), and why your kernel can handle a transposed input correctly without any special-casing as long as you use the tensor's own strides.
3. What is the *functional* (not just syntactic) difference between a tensor descriptor on an H100 versus on an older GPU without TMA support?
4. Why does this curriculum still teach manual pointer arithmetic in depth, given that tensor descriptors are the more modern API?

## 5.9 What's Next

You now have every mechanical piece needed to read and write real kernels: grids, `program_id`, masking, and multi-dimensional addressing. Chapter 6 covers the last foundational piece — compile-time control flow (`constexpr` branching, loops, `tl.static_assert`) — before Part III has you build a full sequence of production-pattern kernels starting with fused softmax in Chapter 7.
