# Chapter 16: Custom Autograd Functions

## Introduction

Every model built so far has relied entirely on PyTorch's automatic differentiation — you write a `forward()` using standard tensor operations, and autograd derives the backward pass for you. This works for the overwhelming majority of cases. But occasionally you need an operation autograd can't handle automatically: a non-differentiable function that needs a custom gradient approximation, a numerically unstable operation that benefits from a hand-derived, more stable backward formula, or a custom CUDA kernel (Chapter 25) that needs to be wired into the autograd graph manually.

`torch.autograd.Function` is the mechanism for exactly this: defining a new differentiable operation by writing its forward and backward passes explicitly, then plugging it into the same autograd system every built-in operation uses.

By the end of this chapter you'll be able to:
- Explain when you actually need a custom `autograd.Function` (and when you don't)
- Implement `forward()` and `backward()` for a custom operation correctly
- Use `ctx` to save tensors needed for the backward pass
- Verify a custom backward implementation is correct using `gradcheck`
- Implement a straight-through estimator, a common real-world use case

---

## 16.1 When Do You Actually Need This?

This is worth establishing clearly upfront, because reaching for `torch.autograd.Function` is rarely the right first move: **if your operation can be expressed as a composition of existing differentiable PyTorch operations, autograd already handles it correctly — write it as a normal function or `nn.Module` and let Chapter 3's machinery do its job.** Every example in this book so far, including the from-scratch attention implementation in Chapter 12, works this way, with zero custom autograd code.

Custom `autograd.Function` implementations are genuinely needed in a narrower set of cases:

- **Non-differentiable operations that need a gradient anyway** — e.g., a hard threshold or rounding operation used in quantization (Chapter 26), where you want to use a discrete operation in the forward pass but still train through it using an approximate gradient (Section 16.4's straight-through estimator).
- **Custom CUDA kernels** — wiring a hand-written CUDA operation (Chapter 25) into PyTorch's autograd graph requires defining both directions explicitly, since autograd has no way to automatically differentiate through raw CUDA code.
- **Numerical stability** — some operations have a mathematically equivalent but numerically better-behaved backward formula than what falls out of naively differentiating the forward computation step by step; writing a custom backward lets you use that better formula directly.
- **Memory optimization** — a custom backward pass can sometimes avoid storing certain intermediate activations that autograd's default approach would otherwise keep, recomputing them instead (a manual, targeted version of the broader gradient checkpointing technique covered in Chapter 24).

If none of these apply to what you're building, you don't need this chapter's machinery — but understanding it is essential once you do, and it also demystifies what every built-in PyTorch operation is doing under the hood.

---

## 16.2 The `torch.autograd.Function` Structure

A custom autograd function is a class with two static methods: `forward()`, computing the operation's output, and `backward()`, computing gradients with respect to each input given the gradient with respect to the output.

The simplest possible non-trivial example — a custom implementation of squaring, written the "hard way" specifically to show the mechanics:

```python
import torch

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)   # save x -- needed to compute the gradient later
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors     # retrieve what was saved in forward
        grad_input = grad_output * 2 * x   # d/dx(x^2) = 2x, chained with grad_output
        return grad_input

# Using it:
square = Square.apply   # .apply, not calling the class directly -- see below
x = torch.tensor(3.0, requires_grad=True)
y = square(x)
y.backward()
print(x.grad)   # tensor(6.) -- correct: d/dx(x^2) at x=3 is 2*3=6
```

Several details here, each essential:

- **`ctx`** ("context") is an object autograd provides automatically — it's your channel for passing information from `forward()` to `backward()`. It is *not* something you instantiate yourself.
- **`ctx.save_for_backward(x)`**: the correct, memory-safe way to save tensors needed later — use this rather than just storing `x` as a plain attribute on `ctx` (e.g., `ctx.x = x`), because `save_for_backward` integrates properly with autograd's memory management (in particular, it correctly handles cases involving in-place operations and avoids creating reference cycles that could leak memory). Use `ctx.save_for_backward()` specifically for tensors; store non-tensor values (like a scalar hyperparameter) as plain attributes instead — `ctx.my_scalar = 5.0` is fine.
- **`backward()`'s signature**: receives `ctx` plus one argument per forward output (here, one, since `forward` returns one tensor) — `grad_output` is the gradient of the final loss with respect to *this operation's output*, supplied by autograd as it walks the graph backward from the loss (exactly the reverse-mode process described conceptually in Chapter 3). `backward()` must return one value per forward *input* (here, also one, matching `forward(ctx, x)`'s single input `x`) — the gradient of the loss with respect to that input.
- **The chain rule, made explicit**: `grad_input = grad_output * 2 * x` is exactly the chain rule — `2 * x` is this operation's own local derivative ($d(x^2)/dx$), multiplied by `grad_output` (the gradient flowing in from everything downstream). This multiplication is the manual version of what autograd normally does for you automatically when you don't write a custom function.
- **`.apply`, not direct instantiation**: you call `Square.apply(x)`, never `Square()(x)` or `Square.forward(x)` directly — `.apply` is what correctly registers the operation in the computational graph (attaching a `grad_fn` to the output, exactly as discussed in Chapter 3, Section 3.1) and sets up `ctx` properly. Calling `forward()` directly bypasses the graph-building machinery entirely, the same category of mistake as calling `.forward()` directly on an `nn.Module` in Chapter 5.

---

## 16.3 Multiple Inputs and Outputs

`forward()` and `backward()` generalize directly to multiple inputs and outputs — the correspondence between them is positional and must match exactly.

```python
class ScaledAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, scale):
        ctx.save_for_backward(x, y)
        ctx.scale = scale   # non-tensor value: plain attribute, not save_for_backward
        return scale * (x + y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        scale = ctx.scale
        # d/dx[scale*(x+y)] = scale, d/dy[scale*(x+y)] = scale
        grad_x = grad_output * scale
        grad_y = grad_output * scale
        grad_scale = None   # scale is a Python float, not a tensor -- no gradient needed for it
        return grad_x, grad_y, grad_scale

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
out = ScaledAdd.apply(x, y, 5.0)
out.backward()
print(x.grad, y.grad)   # tensor(5.) tensor(5.)
```

Two rules to keep straight:

- **`backward()` must return exactly as many values as `forward()` received as inputs** (here, three: `x`, `y`, `scale`), in the same order, even for inputs that don't need a gradient — return `None` for those, as done for `scale` here (a plain Python float can't have `.grad`, since it was never a tensor with `requires_grad` in the first place).
- If `forward()` returns multiple outputs, `backward()` receives one `grad_output` argument per output, in the same order.

---

## 16.4 A Real Use Case: The Straight-Through Estimator

A genuinely practical example: rounding a continuous value to the nearest integer has zero gradient almost everywhere (the derivative of a step function is zero except at the discontinuities, where it's undefined) — yet you sometimes need to train through a rounding operation, most notably in quantization-aware training (Chapter 26), where weights or activations are rounded to a lower-precision representation during the forward pass, but the model still needs to learn via gradients as if rounding hadn't happened.

The **straight-through estimator (STE)** is a standard trick: use the true (non-differentiable) operation in the forward pass, but pretend, for gradient purposes, that the backward pass is the identity function — passing `grad_output` straight through unchanged, as if the rounding operation weren't there at all:

```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)   # true rounding in the forward pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output   # identity gradient -- "pretend" round() has gradient 1 everywhere

round_ste = RoundSTE.apply

x = torch.tensor([1.2, 2.7, 3.5], requires_grad=True)
y = round_ste(x)
print(y)   # tensor([1., 3., 4.])

loss = y.sum()
loss.backward()
print(x.grad)   # tensor([1., 1., 1.]) -- gradient flows through as if round() were the identity
```

This is a deliberate, principled approximation — not a bug or a workaround forced by a limitation — that's widely used precisely because it empirically works well: it lets a model's other parameters receive a meaningful learning signal even though the exact operation used in the forward pass has no true, useful gradient of its own. You'll encounter this exact pattern (or close variants) any time you work with quantization or with discrete/categorical operations that need to remain trainable end-to-end.

---

## 16.5 Verifying Correctness with `gradcheck`

A hand-derived backward pass is a common source of subtle bugs — an easy-to-miss sign error or a misapplied chain rule can silently produce a model that trains, but incorrectly (a category of bug that's notoriously difficult to catch just by watching a loss curve, since a wrong-but-plausible gradient often still reduces loss, just not as well or as correctly as it should). PyTorch provides `torch.autograd.gradcheck` specifically to catch this: it numerically approximates the gradient (via finite differences) and compares it against your analytical `backward()` implementation.

```python
from torch.autograd import gradcheck

# gradcheck requires double precision (float64) for numerical accuracy
x = torch.randn(5, dtype=torch.float64, requires_grad=True)

test_passed = gradcheck(Square.apply, (x,), eps=1e-6, atol=1e-4)
print(test_passed)   # True if the analytical and numerical gradients match closely
```

**Always run `gradcheck` on any custom `backward()` implementation before trusting it in a real training run.** Note the `dtype=torch.float64` requirement — `gradcheck`'s finite-difference approximation needs the extra numerical precision `float32` can't reliably provide, so this check should be run in isolation as a correctness test (analogous in spirit to a unit test), separate from your actual training precision (which, per Chapter 15, will typically be `float32`, `float16`, or `bfloat16`).

Deliberately verifying a broken implementation is also instructive:

```python
class BuggySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * x   # BUG: missing the factor of 2

x = torch.randn(5, dtype=torch.float64, requires_grad=True)
# gradcheck(BuggySquare.apply, (x,))   # raises an AssertionError, correctly catching the bug
```

`gradcheck` would correctly flag this: the numerical gradient (computed by nudging `x` slightly and measuring the change in output) would be roughly `2x`, while the buggy analytical backward returns just `x` — a real, catchable discrepancy `gradcheck` is specifically designed to surface.

---

## 16.6 Using a Custom Function Inside an `nn.Module`

Custom autograd functions integrate naturally with the `nn.Module` system from Chapter 5 — call `.apply()` from within `forward()`, exactly like any other operation:

```python
import torch.nn as nn

class QuantizedLinear(nn.Module):
    """A toy example: a linear layer whose weights are rounded to the
    nearest 0.1 in the forward pass, but trained using the STE."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        quantized_weight = round_ste(self.weight * 10) / 10   # round to nearest 0.1, STE gradient
        return x @ quantized_weight.T + self.bias

layer = QuantizedLinear(4, 2)
x = torch.rand(3, 4)
out = layer(x)
loss = out.sum()
loss.backward()
print(layer.weight.grad is not None)   # True -- gradients flow through despite the rounding
```

This is a simplified illustration of the core mechanism behind real quantization-aware training techniques (Chapter 26 covers this properly) — the pattern of "use a custom `autograd.Function` to make a fundamentally non-differentiable forward operation trainable via an approximate backward pass" generalizes well beyond this toy example.

---

## Summary

- Reach for `torch.autograd.Function` only when an operation genuinely can't be expressed via existing differentiable PyTorch operations — composing built-in ops and letting autograd handle the backward pass automatically is correct and sufficient for the vast majority of models.
- A custom function defines `forward(ctx, ...)` and `backward(ctx, grad_output, ...)` as static methods; `ctx.save_for_backward()` stores tensors needed later, and non-tensor values are stored as plain `ctx` attributes.
- `backward()` must return one gradient per forward input (in order, using `None` for inputs that don't need a gradient), implementing the chain rule explicitly by multiplying the operation's local derivative against the incoming `grad_output`.
- Always invoke a custom function via `.apply()`, never by calling the class or `forward()` directly.
- The straight-through estimator — using the true operation in `forward()` but the identity in `backward()` — is a standard, principled technique for training through non-differentiable operations like rounding.
- `gradcheck` (with `float64` tensors) numerically verifies a custom `backward()` implementation and should be run on any hand-written backward pass before trusting it in training.

## Exercises

1. Implement a custom `autograd.Function` for `f(x) = 1/x` (reciprocal), including a correct `backward()`, and verify it with `gradcheck`.
2. Implement a custom `autograd.Function` for a clamped ReLU variant — `f(x) = max(0, min(x, 6))` (sometimes called ReLU6) — and verify both the forward output and the backward gradient (which should be zero outside `[0, 6]` and one inside it) against `gradcheck`.
3. Deliberately introduce a sign error into a custom `backward()` implementation, run `gradcheck`, and confirm it fails as expected — then fix the bug and confirm it passes.
4. Using the straight-through estimator pattern from Section 16.4, implement a binary sign function (`f(x) = +1 if x >= 0 else -1`) usable in a small classifier's forward pass, trainable end-to-end via STE gradients, and confirm gradients flow through it correctly on a toy example.

**Next:** Chapter 17 covers hooks — forward and backward hooks for inspecting and modifying activations and gradients without changing a model's `forward()` code, an essential debugging tool for the training issues you'll encounter at scale.
