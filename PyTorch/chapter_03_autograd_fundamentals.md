# Chapter 3: Autograd Fundamentals

## Introduction

Everything up to this point has been about tensors as static data. This chapter introduces the feature that turns PyTorch into a deep learning framework: **autograd**, PyTorch's automatic differentiation engine.

Autograd is what lets you write `loss.backward()` and have every gradient in a million-parameter network computed correctly, without you deriving a single derivative by hand. Understanding *how* it works — not just that it works — will save you hours of debugging later, especially once you're writing custom training loops, custom losses, or RL objectives like GRPO.

By the end of this chapter you'll be able to:
- Explain what a computational graph is and how PyTorch builds one dynamically
- Use `requires_grad`, `.backward()`, and `.grad` correctly
- Understand gradient accumulation and why you call `optimizer.zero_grad()`
- Control autograd with `torch.no_grad()`, `.detach()`, and `retain_graph`
- Avoid the most common autograd errors

---

## 3.1 What Is a Computational Graph?

When you perform operations on tensors that have `requires_grad=True`, PyTorch doesn't just compute the result — it also records *how* that result was computed, building a **directed acyclic graph (DAG)** of operations as it goes. This is called a computational graph, and it's built **dynamically**, one operation at a time, as your Python code actually executes (this is PyTorch's "define-by-run" model, as opposed to older "define-and-run" frameworks that required building the whole graph upfront).

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y * 3

print(z)
# tensor(12., grad_fn=<MulBackward0>)
```

Notice `grad_fn=<MulBackward0>`. This is PyTorch telling you: "this tensor was produced by a multiplication, and I know how to compute the gradient of that operation." Every tensor produced from an operation on a `requires_grad=True` tensor carries a `grad_fn` that points backward to the operation that created it — that chain of `grad_fn`s *is* the computational graph.

```
x (leaf, requires_grad=True)
  │  x ** 2
  ▼
y  (grad_fn=PowBackward0)
  │  y * 3
  ▼
z  (grad_fn=MulBackward0)
```

Tensors you create directly (like `x` above) are called **leaf tensors** — they're the starting points of the graph, typically your model's parameters or inputs.

---

## 3.2 `requires_grad`: Turning Tracking On

Only tensors with `requires_grad=True` (and results derived from them) get tracked. Regular tensors don't carry the overhead of graph-building at all.

```python
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0)  # requires_grad=False by default

y = x * w
print(y.requires_grad)  # True — inherits from x, since x requires grad
```

You can also turn tracking on for an existing tensor in place:

```python
x = torch.tensor(2.0)
x.requires_grad_(True)   # trailing underscore: in-place, as covered in Chapter 2
```

In practice, you rarely set `requires_grad=True` manually — `nn.Module` parameters (Chapter 5) have it set to `True` automatically when created. You mostly interact with `requires_grad` when freezing layers for transfer learning (Chapter 14) or debugging why a gradient isn't flowing.

---

## 3.3 Computing Gradients with `.backward()`

Calling `.backward()` on a scalar tensor triggers **reverse-mode automatic differentiation**: PyTorch walks the computational graph backward from that tensor, applying the chain rule at each node, and accumulates the resulting gradients into the `.grad` attribute of every leaf tensor that required gradients.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2       # y = x^2
y.backward()      # computes dy/dx

print(x.grad)     # tensor(4.)   since dy/dx = 2x = 2*2 = 4
```

A slightly larger example — a small chain of operations:

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1    # y = x^2 + 2x + 1
y.backward()

print(x.grad)   # tensor(8.)   dy/dx = 2x + 2 = 2*3 + 2 = 8
```

### Why does `.backward()` require a scalar?

`.backward()` computes the gradient of *one* output with respect to its inputs — mathematically, that only makes direct sense when the output is a scalar (e.g., a loss value). If `y` isn't a scalar, you must supply a `gradient` argument, which acts as the upstream gradient in a vector-Jacobian product:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2                      # y is a vector: [1, 4, 9]

# y.backward()                  # RuntimeError: grad can be implicitly
                                  # created only for scalar outputs

y.backward(torch.ones_like(y))    # treats each element as contributing equally
print(x.grad)                     # tensor([2., 4., 6.])
```

In practice, you'll call `.backward()` on a scalar loss the vast majority of the time, so this edge case rarely comes up directly — but understanding it demystifies error messages you'll eventually hit.

---

## 3.4 Gradient Accumulation

Here's a detail that trips up almost everyone the first time: **gradients accumulate (add up) across multiple `.backward()` calls by default; they are not overwritten.**

```python
x = torch.tensor(2.0, requires_grad=True)

y1 = x ** 2
y1.backward()
print(x.grad)   # tensor(4.)

y2 = x ** 2
y2.backward()
print(x.grad)   # tensor(8.)  <- accumulated, not overwritten!
```

This is *intentional* — it's what allows techniques like gradient accumulation over multiple mini-batches (simulating a larger batch size than fits in memory) to work by simply calling `.backward()` several times before a single optimizer step. But it means that in a standard training loop, you **must** zero out gradients at the start of each iteration, or they'll silently accumulate across batches and corrupt your training:

```python
for batch in dataloader:
    optimizer.zero_grad()   # reset .grad to zero (or None) before this step
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()
```

We'll build this loop for real in Chapter 8, but the reason `zero_grad()` exists at all is entirely explained by what you just learned in this section.

---

## 3.5 Controlling the Graph: `no_grad`, `detach`, and Freeing Memory

Building the computational graph has a real memory and compute cost — PyTorch has to store intermediate activations to compute gradients later. When you don't need gradients (inference, evaluation, or manipulating tensors outside of training), turn tracking off explicitly.

### `torch.no_grad()`

A context manager that disables graph-building for everything inside it:

```python
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x ** 2
    print(y.requires_grad)   # False — no graph was built

y2 = x ** 2
print(y2.requires_grad)      # True — tracking resumes outside the block
```

Use this for evaluation loops, inference, and any manual weight updates.

### `.detach()`

Returns a new tensor that shares the same storage (recall Chapter 2's discussion of views!) but is **detached from the computational graph** — it has `requires_grad=False` and no `grad_fn`, so gradients won't flow through it.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y_detached = y.detach()

print(y_detached.requires_grad)   # False
```

This is essential when you want to use a tensor's *value* somewhere without dragging its gradient history along — e.g., logging a loss value, or in RL algorithms (relevant to your GRPO work) where you need to treat certain quantities as constants during a particular backward pass.

### `@torch.no_grad()` as a decorator

Also usable directly on functions, which is common in evaluation code:

```python
@torch.no_grad()
def evaluate(model, x):
    return model(x)
```

---

## 3.6 Common Autograd Errors (and What They Actually Mean)

### "RuntimeError: element 0 of tensors does not require grad"

You called `.backward()` on a tensor with `requires_grad=False`. Check that your loss traces back to a `requires_grad=True` leaf tensor (usually your model's parameters).

### "RuntimeError: Trying to backward through the graph a second time"

By default, PyTorch **frees the graph** after a single `.backward()` call, to save memory. If you need to call `.backward()` more than once on the same graph (rare, but happens with certain multi-loss setups), pass `retain_graph=True`:

```python
y.backward(retain_graph=True)   # keep the graph alive for another backward() call
y.backward()                     # now this works too
```

Overusing `retain_graph=True` is usually a sign of a design problem (e.g., re-running a forward pass unnecessarily) rather than something you need routinely.

### "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation"

Directly connects back to Chapter 2's in-place operations warning:

```python
x = torch.tensor(2.0, requires_grad=True)
x.add_(1)   # RuntimeError — can't modify a leaf tensor with requires_grad in place
```

Fix: use out-of-place ops (`x = x + 1`) for anything that requires gradients, and reserve in-place ops for `requires_grad=False` tensors or contexts wrapped in `torch.no_grad()`.

---

## 3.7 A Complete Minimal Example: Gradient Descent by Hand

Putting it all together — fitting `y = wx + b` to data using nothing but autograd and manual updates, no `nn.Module` or optimizer yet (those come in Chapters 5–6):

```python
torch.manual_seed(0)

# Toy data: y = 2x + 1, plus noise
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_data = torch.tensor([3.0, 5.0, 7.0, 9.0])

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01

for epoch in range(100):
    y_pred = w * x_data + b
    loss = ((y_pred - y_data) ** 2).mean()

    loss.backward()

    with torch.no_grad():           # don't track the manual update itself
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()               # manual equivalent of optimizer.zero_grad()
        b.grad.zero_()

print(w.item(), b.item())   # should converge close to w=2.0, b=1.0
```

Every line here maps directly to something covered in this chapter: `requires_grad` on the leaves, `.backward()` building and consuming the graph, `torch.no_grad()` protecting the manual update from being tracked, and explicit `.zero_()` calls preventing gradient accumulation across epochs.

---

## Summary

- Autograd builds a dynamic computational graph as operations execute on tensors with `requires_grad=True`.
- `.backward()` walks that graph in reverse, applying the chain rule, and accumulates results into `.grad` on leaf tensors.
- Gradients **accumulate** by default — always zero them between optimization steps.
- `torch.no_grad()` and `.detach()` are how you opt out of graph tracking, for inference, evaluation, or treating a value as a constant.
- Most autograd errors trace back to one of three things: a missing `requires_grad`, a graph that's already been freed, or an in-place op on a tracked leaf tensor.

## Exercises

1. Compute the gradient of `f(x) = 3x^3 - 2x^2 + x` at `x = 2.0` using autograd, and verify it by hand using calculus.
2. Modify the gradient descent example in 3.7 to fit `y = w1*x1 + w2*x2 + b` (two features) instead of one.
3. Deliberately trigger and then fix each of the three errors described in Section 3.6.
4. Explain, in your own words, why `optimizer.zero_grad()` exists — tie your answer directly to what `.backward()` does to `.grad`.

**Next:** Chapter 4 covers NumPy interop in more depth — `torch.from_numpy()` vs `torch.as_tensor()`, common migration pitfalls, and where autograd and NumPy interoperability intersect.
