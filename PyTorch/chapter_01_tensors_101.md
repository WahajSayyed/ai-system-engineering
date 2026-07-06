# Chapter 1: Tensors 101

## Introduction

Every PyTorch program — from a two-line linear regression to a multi-node LLM training run — starts with the same building block: the **tensor**. A tensor is a multi-dimensional array, conceptually identical to a NumPy `ndarray`, but with two superpowers that NumPy arrays don't have:

1. It can live on a GPU (or other accelerators) and run operations there.
2. It can track the history of operations applied to it, which enables automatic differentiation (covered in Chapter 3).

This chapter covers everything you need to create, inspect, and index tensors confidently. We won't touch autograd or neural networks yet — just the data structure itself.

By the end of this chapter you'll be able to:
- Create tensors from Python data, NumPy arrays, and built-in factory functions
- Understand and control `dtype` and `device`
- Index and slice tensors like NumPy arrays, plus PyTorch-specific tricks
- Reason about shapes and use broadcasting correctly

---

## 1.1 What Is a Tensor?

A tensor is defined by its **rank** (number of dimensions), **shape** (size along each dimension), and **dtype** (the type of the values it holds).

| Rank | Name | Example |
|------|------|---------|
| 0 | Scalar | a single number: `torch.tensor(3.14)` |
| 1 | Vector | `torch.tensor([1, 2, 3])` |
| 2 | Matrix | `torch.tensor([[1, 2], [3, 4]])` |
| 3+ | N-D tensor | a batch of RGB images: `(batch, channels, height, width)` |

```python
import torch

scalar = torch.tensor(3.14)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])
tensor3d = torch.zeros(2, 3, 4)  # rank-3 tensor

print(scalar.ndim, vector.ndim, matrix.ndim, tensor3d.ndim)
# 0 1 2 3
```

---

## 1.2 Creating Tensors

### From Python data

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
# tensor([[1, 2],
#         [3, 4]])
```

`torch.tensor()` always **copies** the input data into a new tensor. This is safe but has a memory cost — keep this in mind when converting large NumPy arrays.

### From NumPy arrays

```python
import numpy as np

np_array = np.array([[1, 2], [3, 4]])
x_np = torch.from_numpy(np_array)
```

`torch.from_numpy()` shares memory with the original NumPy array — modifying one modifies the other. If you want a zero-copy conversion but aren't sure whether the source is a NumPy array or a Python list, use `torch.as_tensor()`, which avoids copying whenever possible and falls back to copying otherwise. We'll cover this distinction in depth in Chapter 4.

### From another tensor

New tensors can inherit the shape and dtype of an existing tensor:

```python
x_ones = torch.ones_like(x_data)      # same shape/dtype as x_data, filled with 1s
x_rand = torch.rand_like(x_data, dtype=torch.float)  # override dtype
```

### Factory functions

These are the workhorses you'll use constantly to initialize weights, masks, and placeholders:

```python
shape = (2, 3)

torch.rand(shape)     # uniform random in [0, 1)
torch.randn(shape)    # standard normal (mean 0, std 1)
torch.ones(shape)     # all ones
torch.zeros(shape)    # all zeros
torch.eye(3)          # 3x3 identity matrix
torch.arange(0, 10, 2)   # tensor([0, 2, 4, 6, 8]) — like Python's range()
torch.linspace(0, 1, 5)  # 5 evenly spaced values between 0 and 1
torch.empty(shape)    # uninitialized memory — fast, but contains garbage values
```

> **Practical tip:** `torch.empty()` is faster than `torch.zeros()` because it skips initialization, but only use it when you're about to overwrite every element (e.g., pre-allocating a buffer you'll fill in a loop).

---

## 1.3 Tensor Attributes: Shape, Dtype, Device

Every tensor carries three pieces of metadata that you should check reflexively when debugging:

```python
tensor = torch.rand(3, 4)

print(f"Shape:  {tensor.shape}")   # torch.Size([3, 4])
print(f"Dtype:  {tensor.dtype}")   # torch.float32
print(f"Device: {tensor.device}")  # cpu
```

### Dtype

The default float dtype is `torch.float32`. Common dtypes you'll encounter:

| dtype | Use case |
|-------|----------|
| `torch.float32` | Default for weights and activations |
| `torch.float16` / `torch.bfloat16` | Mixed-precision training (Chapter 15) |
| `torch.int64` | Default for indices, class labels |
| `torch.bool` | Masks |

You can specify or change dtype explicitly:

```python
t = torch.zeros(2, 4, dtype=torch.int32)
t = t.float()          # cast to float32
t = t.to(torch.float64)  # cast to float64 (double precision)
```

**Mismatched dtypes are one of the most common sources of runtime errors** — e.g., feeding an `int64` label tensor into a loss function that expects `float32`, or trying to multiply a `float32` tensor by a `float64` one. When in doubt, print `.dtype`.

### Device

By default, tensors are created on the CPU. Moving a tensor to a GPU (if available) is done with `.to()` or `.cuda()`:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = torch.rand(3, 4).to(device)
```

A critical rule you'll hit constantly: **operations between two tensors require both to be on the same device.** Trying to add a CPU tensor to a CUDA tensor raises a `RuntimeError`. This becomes especially relevant later when you're juggling multiple GPUs in your NUC/multi-node setup — always be explicit about which device a tensor lives on.

```python
a = torch.rand(3, 3)                 # cpu
b = torch.rand(3, 3).to("cuda")      # cuda:0
# a + b  -> RuntimeError: expected all tensors on same device
a = a.to("cuda")
a + b   # works
```

---

## 1.4 Indexing and Slicing

PyTorch indexing follows NumPy conventions closely.

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

x[0]        # first row -> tensor([0, 1, 2, 3])
x[:, 0]     # first column -> tensor([0, 4, 8])
x[1, 2]     # single element -> tensor(6)
x[0:2, 1:3] # sub-matrix -> tensor([[1, 2], [5, 6]])
x[-1]       # last row -> tensor([8, 9, 10, 11])
```

### Boolean (mask) indexing

Extremely common for filtering — e.g., zeroing out padding tokens or selecting positive values:

```python
x = torch.tensor([-1, 2, -3, 4, -5])
mask = x > 0
print(mask)        # tensor([False, True, False, True, False])
print(x[mask])      # tensor([2, 4])

x[x < 0] = 0        # in-place: clip negative values to zero
```

### Fancy (integer array) indexing

```python
x = torch.tensor([10, 20, 30, 40, 50])
idx = torch.tensor([0, 2, 4])
print(x[idx])   # tensor([10, 30, 50])
```

### Modifying values in place

```python
x = torch.zeros(3, 3)
x[1, :] = torch.tensor([1., 2., 3.])
```

---

## 1.5 Shapes and Broadcasting

Broadcasting lets PyTorch perform element-wise operations on tensors of different shapes without explicit copying, following the same rules as NumPy:

1. Align shapes from the **rightmost** dimension.
2. Two dimensions are compatible if they're equal, or if one of them is 1.
3. Missing dimensions are treated as size 1.

```python
a = torch.ones(4, 3)       # shape (4, 3)
b = torch.tensor([1., 2., 3.])  # shape (3,)

result = a + b   # b is broadcast across each of the 4 rows
print(result.shape)  # torch.Size([4, 3])
```

A classic real-world example: adding a bias vector to every row of a batch of activations, or normalizing a batch of images by per-channel mean/std:

```python
images = torch.rand(8, 3, 32, 32)          # (batch, channels, H, W)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

normalized = (images - mean) / std   # broadcasts across batch, H, and W
```

> **Debugging tip:** the single most common shape bug is silently-successful broadcasting that produces the *wrong* result rather than an error — e.g., adding a `(3,)` vector when you meant a `(3, 1)` column vector. Always print `.shape` after an operation you're unsure about, and prefer `.view()`/`.unsqueeze()` (Chapter 2) to make intended broadcast dimensions explicit rather than relying on implicit alignment.

---

## 1.6 Useful Inspection Habits

A few one-liners worth internalizing — you'll type these dozens of times a day once you're debugging real models:

```python
t.shape       # dimensions
t.dtype       # data type
t.device      # cpu / cuda:0 / mps
t.numel()     # total number of elements
t.dim()       # number of dimensions (same as ndim)
t.requires_grad  # whether autograd is tracking this tensor (Chapter 3)
```

---

## Summary

- Tensors are PyTorch's core data structure: N-dimensional arrays with a `dtype` and a `device`.
- `torch.tensor()` copies data; `torch.from_numpy()` shares memory with NumPy; factory functions (`zeros`, `rand`, `arange`, etc.) cover most initialization needs.
- Always be deliberate about `dtype` and `device` — mismatches are among the most common runtime errors in PyTorch code.
- Indexing and slicing mirror NumPy, with boolean masks and fancy indexing as powerful additions.
- Broadcasting rules let you combine differently-shaped tensors — but silent shape bugs are a real risk, so verify shapes explicitly when in doubt.

## Exercises

1. Create a `(5, 5)` tensor of random integers between 0 and 10 using `torch.randint()`. Replace all values greater than 5 with 0 using boolean masking.
2. Given a tensor of shape `(10, 3)` representing 10 RGB pixel values, write a broadcasting expression to normalize each channel by a different scalar mean.
3. Create two tensors on different devices (or simulate this by reasoning through it if you don't have a GPU available) and predict what happens when you try to add them. Then fix it.
4. Use `torch.arange` and `.reshape()` to build a `(4, 4)` tensor, then extract the anti-diagonal (top-right to bottom-left) using indexing.

**Next:** Chapter 2 dives into tensor operations and memory layout — views vs. copies, strides, and why `.contiguous()` matters.
