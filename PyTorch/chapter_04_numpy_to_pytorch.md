# Chapter 4: From NumPy to PyTorch

## Introduction

If you've done any scientific computing in Python, NumPy's `ndarray` API will feel deeply familiar in PyTorch — that's by design. But the overlap is also where subtle bugs hide: memory-sharing surprises, dtype mismatches, and autograd interactions that don't exist in NumPy at all. This chapter closes the loop between Chapters 1–3 by nailing down exactly how the two libraries interact.

By the end of this chapter you'll be able to:
- Convert between NumPy arrays and PyTorch tensors correctly, choosing the right method for the situation
- Predict when a conversion shares memory vs. copies it
- Avoid the autograd + NumPy interaction that raises PyTorch's most confusing error message
- Recognize dtype and device pitfalls specific to NumPy interop

---

## 4.1 The Three Conversion Functions

PyTorch gives you three ways to go from NumPy to a tensor, and they are **not interchangeable**:

| Function | Copies data? | Notes |
|---|---|---|
| `torch.tensor(np_array)` | Always copies | Safest default; independent from the source |
| `torch.from_numpy(np_array)` | Never copies | Shares memory; fastest; CPU only |
| `torch.as_tensor(np_array)` | Copies only if necessary | Smart default — shares memory when it can |

```python
import numpy as np
import torch

np_array = np.array([1, 2, 3])

a = torch.tensor(np_array)      # independent copy
b = torch.from_numpy(np_array)   # shares memory with np_array
c = torch.as_tensor(np_array)    # shares memory with np_array (no copy needed here)
```

Verify the memory sharing directly:

```python
np_array[0] = 999
print(a)   # tensor([1, 2, 3])   <- unaffected, it was a copy
print(b)   # tensor([999, 2, 3]) <- changed! shared memory
print(c)   # tensor([999, 2, 3]) <- changed! shared memory
```

This mirrors exactly what you learned about views vs. copies in Chapter 2 — `from_numpy` and `as_tensor` are effectively creating a *view* of the NumPy array's memory, not a new tensor.

### Going the other direction: tensor → NumPy

```python
t = torch.tensor([1., 2., 3.])
np_from_t = t.numpy()    # shares memory, just like from_numpy in reverse

t[0] = 999
print(np_from_t)    # array([999., 2., 3.]) <- changed
```

`.numpy()` only works on **CPU tensors**. If your tensor is on a GPU, you must move it to CPU first:

```python
t_gpu = torch.tensor([1., 2., 3.]).to("cuda")
# t_gpu.numpy()          # RuntimeError
t_gpu.cpu().numpy()       # works — .cpu() returns a CPU copy first
```

> **Rule of thumb:** default to `torch.as_tensor()` when converting from NumPy — it avoids unnecessary copies but still does the right thing if a copy is genuinely required (e.g., dtype conversion forces a copy regardless of which function you use).

---

## 4.2 The Autograd Trap

Here's the interaction that catches almost everyone at least once: **`.numpy()` refuses to work on a tensor that's part of an active computational graph.**

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2

# y.numpy()
# RuntimeError: Can't call numpy() on Tensor that requires grad.
# Use tensor.detach().numpy() instead.
```

The error message tells you exactly what to do, and it's worth understanding *why*: NumPy arrays have no concept of a computational graph. If PyTorch let you silently convert a graph-tracked tensor to NumPy, any further computation you did in NumPy would be invisible to autograd — a foot-gun for silently broken gradients. So PyTorch refuses outright rather than letting you accidentally detach part of your graph without realizing it.

The fix, exactly as suggested:

```python
y_numpy = y.detach().numpy()   # detach from the graph first, then convert
```

This is directly connected to Chapter 3's `.detach()` — this is the single most common place you'll reach for it in practice: converting a loss or a metric to NumPy/Python for logging, plotting with matplotlib, or feeding into a non-PyTorch library.

```python
losses = []
for epoch in range(epochs):
    loss = compute_loss(...)
    loss.backward()
    optimizer.step()
    losses.append(loss.detach().cpu().numpy())   # detach -> cpu -> numpy, in order
```

Note the order when a tensor is both on GPU *and* requires grad: `.detach()` before `.cpu()` before `.numpy()` is the safe, conventional order (though `.cpu()` and `.detach()` are commutative — `.numpy()` must come last regardless).

---

## 4.3 Dtype Mismatches

NumPy's default float dtype is `float64` (double precision). PyTorch's default is `float32`. This mismatch is a frequent source of silent bugs and unnecessary memory/compute overhead when converting data pipelines that started life in NumPy or pandas.

```python
np_array = np.array([1.0, 2.0, 3.0])
print(np_array.dtype)   # float64

t = torch.from_numpy(np_array)
print(t.dtype)   # torch.float64  <- inherited, not float32!
```

This matters because most neural network layers expect `float32` inputs by default. Feeding in `float64` tensors won't necessarily error immediately, but it can cause dtype-mismatch errors deeper in a model, or silently double your memory footprint and slow down GPU computation (GPUs are heavily optimized for `float32`/`float16`, less so for `float64`).

Fix explicitly at the boundary:

```python
t = torch.from_numpy(np_array).float()          # cast to float32
# or, when creating from scratch:
t = torch.as_tensor(np_array, dtype=torch.float32)
```

The same issue shows up in reverse with integer arrays — NumPy's default int type varies by platform (`int32` on Windows, `int64` on Linux/Mac for `np.array([1,2,3])`), while PyTorch's indexing operations (e.g., `nn.Embedding`, cross-entropy loss targets) specifically expect `torch.int64` (`torch.long`). Always check `.dtype` explicitly rather than assuming.

```python
labels_np = np.array([0, 1, 2])
labels = torch.as_tensor(labels_np, dtype=torch.long)   # be explicit
```

---

## 4.4 Common Migration Pitfalls (NumPy → PyTorch)

For anyone porting NumPy code or thinking in NumPy idioms, a few naming and behavioral differences trip people up regularly:

| NumPy | PyTorch | Note |
|---|---|---|
| `np.concatenate` | `torch.cat` | Same behavior, different name |
| `arr.copy()` | `t.clone()` | Same behavior, different name |
| `arr.astype(np.float32)` | `t.float()` / `t.to(torch.float32)` | |
| `np.reshape(arr, shape)` | `t.reshape(shape)` | Very similar; PyTorch's is more forgiving about copies (Chapter 2) |
| `arr.T` | `t.T` (2D) / `t.transpose(0, 1)` / `t.permute(...)` | `.T` on tensors with ndim > 2 reverses *all* dims — often not what you want; prefer `.permute()` for 3D+ |
| `np.dot(a, b)` | `a @ b` / `torch.matmul` | |
| Broadcasting rules | Broadcasting rules | Identical semantics (Chapter 1) |
| In-place `arr += 1` | `t.add_(1)` | NumPy's `+=` is in-place by default; PyTorch's isn't unless you use the `_` method (careful with autograd, Chapter 3) |

### The `.T` trap on 3D+ tensors

This one deserves its own callout because it's easy to get wrong silently:

```python
x = torch.rand(2, 3, 4)
print(x.T.shape)   # torch.Size([4, 3, 2])  -- reverses ALL dimensions
```

If you only wanted to swap two specific dimensions (e.g., in a `(batch, seq, features)` tensor, swapping `seq` and `features`), use `.transpose(dim0, dim1)` or `.permute(...)` instead:

```python
x.transpose(1, 2).shape   # torch.Size([2, 4, 3]) -- only swaps dims 1 and 2
x.permute(0, 2, 1).shape   # same result, explicit about all dimensions
```

### `+=` and autograd

In NumPy, `arr += 1` is just a convenient in-place operation with no side effects beyond mutating `arr`. In PyTorch, if the tensor is part of an autograd graph, the equivalent (`t += 1`, which Python desugars to `t.__iadd__`, effectively an in-place op) can raise the same "leaf Variable... in-place operation" error from Chapter 3:

```python
t = torch.tensor([1.0, 2.0], requires_grad=True)
# t += 1   # RuntimeError, same root cause as Section 3.6
t = t + 1   # safe: creates a new tensor instead of mutating in place
```

---

## 4.5 A Practical Interop Pattern

A pattern you'll use constantly when working with data pipelines (e.g., pandas → NumPy → PyTorch for tabular data, or OpenCV → NumPy → PyTorch for the kind of image pipelines you've built for your conveyor sorting system):

```python
import numpy as np

# Simulate data arriving as NumPy (e.g., from OpenCV, pandas, or a sensor)
raw_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)  # HWC, uint8

# Convert to a PyTorch tensor and prepare it for a model
tensor_image = (
    torch.from_numpy(raw_image)      # share memory (no copy yet)
    .permute(2, 0, 1)                  # HWC -> CHW, the layout PyTorch expects
    .float()                            # uint8 -> float32 (this forces a copy)
    .div(255.0)                          # normalize to [0, 1]
)

print(tensor_image.shape, tensor_image.dtype)
# torch.Size([3, 224, 224]) torch.float32
```

Each step here is deliberate: `from_numpy` avoids an unnecessary copy at the boundary, `.permute()` reorders dimensions for PyTorch's channel-first convention, and `.float()` is where an actual memory copy happens (uint8 → float32 can't be a view — the underlying byte representation is fundamentally different).

---

## Summary

- `torch.tensor()` always copies; `torch.from_numpy()` never copies; `torch.as_tensor()` copies only when necessary — pick deliberately.
- `.numpy()` only works on CPU tensors with `requires_grad=False`; the standard safe conversion order is `.detach().cpu().numpy()`.
- NumPy defaults to `float64`, PyTorch defaults to `float32` — always check `.dtype` explicitly at NumPy/PyTorch boundaries to avoid silent slowdowns or mismatches.
- Naming differs (`concatenate`/`cat`, `copy`/`clone`) even where behavior is identical.
- `.T` reverses *all* dimensions on tensors with ndim > 2 — use `.transpose()` or `.permute()` for anything beyond a 2D matrix.

## Exercises

1. Create a NumPy array of `float64` values, convert it to a PyTorch tensor three different ways (`tensor`, `from_numpy`, `as_tensor`), and verify which ones share memory by mutating the original array.
2. Take a tensor with `requires_grad=True`, run it through a couple of operations, and convert the result to NumPy correctly (handle both the autograd and device cases).
3. Given a NumPy array simulating a batch of images with shape `(N, H, W, C)` and dtype `uint8`, write the conversion pipeline to get a `(N, C, H, W)` `float32` tensor normalized to `[0, 1]`, ready to feed into a CNN (Chapter 10).
4. Explain why `t.T` on a `(2, 3, 4)` tensor is usually a bug when the intent was to swap only two axes, and rewrite it correctly with `.permute()`.

**Next:** Chapter 5 begins Part II — the `nn.Module` system, where you'll assemble the tensors and autograd mechanics from Part I into your first real neural network layers.
