# Chapter 2: Tensor Operations & Memory

## Introduction

Chapter 1 covered *what* tensors are. This chapter covers something more subtle and more important for writing correct, efficient PyTorch code: **how tensors are stored in memory, and how operations interact with that storage.**

A huge class of PyTorch bugs — silent data corruption, unexpected `RuntimeError: view size is not compatible`, gradients that don't flow the way you expect — come from not understanding the difference between a **view** and a **copy**. This chapter builds that mental model from the ground up.

By the end of this chapter you'll be able to:
- Distinguish operations that return views from operations that return copies
- Reason about strides and understand why some reshapes are "free" and others aren't
- Use in-place operations safely (and know when *not* to)
- Fix `contiguous()`-related errors instead of just sprinkling `.contiguous()` everywhere and hoping

---

## 2.1 Storage, Views, and Why They Matter

Every tensor has an underlying **storage** — a flat, one-dimensional block of memory holding its actual data — plus metadata (`shape`, `stride`, `offset`) that tells PyTorch how to interpret that flat block as an N-dimensional array.

A **view** is a new tensor object that points to the *same* underlying storage as another tensor, just with different metadata. No data is copied. A **copy** allocates entirely new storage.

```python
import torch

x = torch.arange(6)                # storage: [0, 1, 2, 3, 4, 5]
y = x.view(2, 3)                    # same storage, new shape metadata

print(y)
# tensor([[0, 1, 2],
#         [3, 4, 5]])

y[0, 0] = 99
print(x)
# tensor([99,  1,  2,  3,  4,  5])   <- x changed too!
```

This is the single most important fact in this chapter: **`y` and `x` share memory.** Modifying one modifies the other. This is a feature, not a bug — it's what makes operations like `.view()`, `.reshape()` (sometimes), slicing, and `.transpose()` extremely cheap. But it's a common source of surprising bugs if you don't expect it.

Check whether two tensors share storage with `data_ptr()`:

```python
print(x.data_ptr() == y.data_ptr())  # True
```

---

## 2.2 Which Operations Return Views vs. Copies?

### Operations that (usually) return views

- Slicing: `x[1:3]`, `x[:, 0]`
- `.view()`
- `.reshape()` — *when possible* (see 2.4)
- `.transpose()` / `.permute()`
- `.squeeze()` / `.unsqueeze()`
- `.expand()`

```python
x = torch.arange(12).reshape(3, 4)
row = x[1]          # view
row[0] = -1
print(x[1, 0])       # -1, x was modified through the view
```

### Operations that always return copies

- `.clone()` — explicit copy, same storage layout
- Most arithmetic ops (`x + 1`, `x * 2`) unless done in-place
- `torch.tensor(x)` (copies, unlike `torch.as_tensor`)
- Indexing with a boolean mask or an integer index tensor ("advanced indexing")

```python
x = torch.arange(6)
mask = x > 2
y = x[mask]      # COPY — advanced indexing never returns a view
y[0] = 99
print(x)         # unchanged: tensor([0, 1, 2, 3, 4, 5])
```

> **Rule of thumb:** basic slicing with `:` and integers returns a view; indexing with a list, tensor, or boolean mask returns a copy. This distinction ("basic" vs. "advanced" indexing) is the same one NumPy uses.

### Making an explicit copy

When you want to be safe rather than clever, use `.clone()`:

```python
y = x.clone()   # independent storage, same values
```

If you need a copy that's also detached from the autograd graph (relevant from Chapter 3 onward), use `.clone().detach()`.

---

## 2.3 Strides: How PyTorch Reads Memory

The **stride** of a tensor tells you how many storage elements to skip to move one step along each dimension. This is the mechanism that makes views possible.

```python
x = torch.arange(12).reshape(3, 4)
print(x.stride())   # (4, 1)
```

This means: to move one step along dimension 0 (rows), skip 4 elements in storage; to move one step along dimension 1 (columns), skip 1 element. That matches a standard row-major (C-contiguous) layout.

Now transpose it:

```python
xt = x.t()
print(xt.shape)    # torch.Size([4, 3])
print(xt.stride())  # (1, 4)
```

**No data was moved.** `xt` is still backed by the exact same flat storage as `x` — only the stride metadata changed, which is why `.transpose()` and `.t()` are effectively free, O(1) operations regardless of tensor size.

```python
print(x.data_ptr() == xt.data_ptr())  # True
```

This is a big deal in performance-sensitive code (like the multi-node GPU pipelines you work with): transposing a huge tensor doesn't cost you a memory copy — until you try to `.view()` it, which brings us to the next section.

---

## 2.4 Contiguity: Why `.view()` Sometimes Fails

A tensor is **contiguous** when its elements are laid out in memory in the same order you'd get by reading it row-by-row (standard C order), with no "gaps" introduced by a prior view operation like `.transpose()`.

`.view()` requires the tensor to be contiguous, because it only changes shape metadata — it doesn't touch memory. If the requested new shape can't be expressed purely via the existing stride, `.view()` raises an error:

```python
x = torch.arange(12).reshape(3, 4)
xt = x.t()          # transposed, no longer contiguous

print(xt.is_contiguous())  # False

xt.view(12)
# RuntimeError: view size is not compatible with input tensor's
# size and stride ...
```

There are two ways to fix this:

```python
# Option 1: .reshape() — falls back to a copy automatically if needed
y = xt.reshape(12)     # works, silently copies if necessary

# Option 2: .contiguous() then .view() — explicit about the copy
y = xt.contiguous().view(12)   # forces a real memory copy first
```

### `.view()` vs. `.reshape()`

| | `.view()` | `.reshape()` |
|---|---|---|
| Requires contiguous input | Yes | No |
| Copies data if needed | Never (raises error instead) | Yes, automatically, if required |
| Performance | Always O(1) when it works | O(1) if possible, otherwise a full copy |

**Practical guidance:** use `.reshape()` by default — it's more forgiving. Reach for `.view()` when you specifically want a guarantee that no copy happens (e.g., in a tight, memory-conscious training loop) and you'd rather get an explicit error than a silent, expensive copy.

```python
x = torch.rand(4, 1000, 1000)  # large tensor
xt = x.transpose(0, 1)          # cheap: view, no copy

# xt.view(-1)          # would raise — not contiguous
xt.reshape(-1)          # works, but silently copies ~16MB of data
```

If you're chasing a memory or performance bug and suspect a hidden copy, `.view()` will surface it immediately as an error rather than letting it happen silently.

---

## 2.5 In-Place Operations

Operations suffixed with an underscore modify the tensor **in place**, mutating its existing storage instead of allocating new memory:

```python
x = torch.tensor([1., 2., 3.])
x.add_(1)        # x is now [2., 3., 4.] — no new tensor created
x.mul_(2)        # x is now [4., 6., 8.]

x.zero_()         # fills with zeros in place
x.fill_(5.0)      # fills with a specific value in place
```

In-place ops save memory — useful when working with large tensors on GPUs with limited VRAM (relevant to your RTX 3090 workflow). But they come with real caveats:

1. **They can silently break autograd.** Modifying a tensor in place that's needed for a gradient computation raises a `RuntimeError` during `.backward()`, or worse, silently produces wrong gradients if PyTorch doesn't catch it. We'll cover this precisely in Chapter 3.

2. **They can invalidate views you don't expect.** Since in-place ops mutate shared storage, any other tensor that's a view into the same memory sees the change too — this is sometimes exactly what you want, and sometimes a nasty bug.

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
# x.add_(1)   # would raise a RuntimeError here once .backward() is called,
              # because autograd needs the original value of x
```

> **Practical guidance:** avoid in-place ops on any tensor that has `requires_grad=True` and participates in a graph you'll call `.backward()` on, unless you specifically know what you're doing. Outside of autograd contexts (e.g., manipulating raw data, buffers, or inference-only code), in-place ops are a safe and effective memory optimization.

---

## 2.6 Core Tensor Operations Reference

A working set of operations you'll use constantly:

```python
x = torch.tensor([[1., 2.], [3., 4.]])

# Element-wise math
x + 1, x - 1, x * 2, x / 2
torch.exp(x), torch.log(x), torch.sqrt(x)

# Reductions
x.sum()             # sum of all elements
x.sum(dim=0)         # sum along rows -> tensor([4., 6.])
x.mean(dim=1)        # mean along columns -> tensor([1.5, 3.5])
x.max(), x.min()
x.argmax(dim=1)       # index of max value along each row

# Matrix operations
x @ x                 # matrix multiplication (preferred syntax)
torch.matmul(x, x)     # equivalent
x.T                    # transpose (2D shorthand for .t())

# Comparison
x == 2                # element-wise boolean tensor
torch.allclose(x, x)    # approximate equality (careful with float precision)

# Concatenation and stacking
a = torch.zeros(2, 3)
b = torch.ones(2, 3)
torch.cat([a, b], dim=0)     # shape (4, 3) — joins along existing dim
torch.stack([a, b], dim=0)    # shape (2, 2, 3) — introduces a new dim
```

`torch.cat` vs `torch.stack` trips people up constantly: `cat` joins tensors along an *existing* dimension (no new dimension is created), while `stack` creates a *new* dimension. Use `stack` when combining several equal-shaped tensors into a batch; use `cat` when extending along a dimension that already exists (e.g., concatenating sequences).

---

## Summary

- Tensors are metadata (`shape`, `stride`, `offset`) wrapped around a flat storage buffer. Views share storage; copies don't.
- Basic slicing, `.view()`, `.transpose()`, and `.permute()` return views (free, O(1)); advanced indexing and most arithmetic return copies.
- Strides explain *why* transposition is free and *why* `.view()` sometimes fails on non-contiguous tensors.
- `.reshape()` is the forgiving option (copies automatically if needed); `.view()` is the strict option (errors instead of silently copying).
- In-place ops (`add_`, `mul_`, etc.) save memory but interact dangerously with autograd — avoid them on tensors that need gradients unless you're certain of the semantics.
- `torch.cat` joins along an existing dimension; `torch.stack` creates a new one.

## Exercises

1. Create a `(4, 4)` tensor, take a transpose, and try to `.view(-1)` it. Observe the error, then fix it two different ways.
2. Write a function that takes two tensors and returns `True` if they share the same underlying storage, using `data_ptr()`.
3. Given a batch of 3D tensors of shape `(8, 3, 3)`, stack them into a `(8, 3, 3)` batch using `torch.stack`, then concatenate the same list along `dim=0` using `torch.cat`. Explain the shape difference.
4. Predict — without running the code — what happens to `x` after each of the following, and why:
   ```python
   x = torch.arange(6)
   y = x.view(2, 3)
   y[:, 0] = -1
   z = x[x > 0].clone()
   z[0] = 999
   ```

**Next:** Chapter 3 introduces PyTorch's autograd engine — how computational graphs are built, how `requires_grad` and `backward()` work, and how gradients accumulate.
