# Chapter 5: The `nn.Module` System

## Introduction

Part I gave you the raw materials: tensors, autograd, memory semantics. Part II starts assembling them into actual neural networks. Everything in PyTorch's neural network API is built on a single, elegant abstraction: `nn.Module`.

Once you understand `nn.Module` deeply — how it tracks parameters, how submodules compose, how `forward()` fits into the picture — every layer type in `torch.nn` (`Linear`, `Conv2d`, `LSTM`, `TransformerEncoder`, all of them) will make sense as a variation on the same pattern, rather than a new thing to memorize.

By the end of this chapter you'll be able to:
- Explain what `nn.Module` actually does under the hood
- Define custom layers and models by subclassing `nn.Module`
- Understand how parameters are registered and discovered automatically
- Compose modules out of other modules, and navigate a model's structure
- Use `nn.Sequential` for straightforward, linear architectures

---

## 5.1 What `nn.Module` Actually Is

At its core, `nn.Module` is a Python class that does three things for you automatically:

1. **Tracks parameters** — any `nn.Parameter` (a tensor with `requires_grad=True` by default) you assign as an attribute is automatically registered and discoverable.
2. **Tracks submodules** — any `nn.Module` you assign as an attribute is automatically registered as a child module, building a tree structure.
3. **Provides a consistent interface** — `.parameters()`, `.to(device)`, `.train()` / `.eval()`, `.state_dict()`, and more, all work uniformly across every module, whether it's a single linear layer or an entire transformer.

The simplest possible custom module — a linear layer implemented by hand, without using `nn.Linear`, to see exactly what's happening:

```python
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()   # ALWAYS call this first — sets up internal bookkeeping
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias

layer = MyLinear(4, 2)
x = torch.rand(3, 4)   # batch of 3 samples, 4 features each
out = layer(x)
print(out.shape)   # torch.Size([3, 2])
```

Two things to notice:

- `super().__init__()` **must** be called before you assign any parameters or submodules. Skipping it causes cryptic errors, because `nn.Module` overrides Python's `__setattr__` to intercept parameter/module assignment — that machinery isn't initialized until `super().__init__()` runs.
- We called `layer(x)`, not `layer.forward(x)`. This matters — see Section 5.3.

---

## 5.2 `nn.Parameter`: What Makes a Tensor "Learnable"

`nn.Parameter` is a thin subclass of `torch.Tensor` with one job: when assigned as an attribute of an `nn.Module`, it gets automatically registered so that `.parameters()` can find it.

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

layer = MyLinear(4, 2)

for name, param in layer.named_parameters():
    print(name, param.shape, param.requires_grad)
# weight torch.Size([2, 4]) True
# bias torch.Size([2]) True
```

Compare this to what happens with a plain tensor:

```python
class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 4)   # NOT an nn.Parameter!

m = Broken()
print(list(m.parameters()))   # [] -- empty! optimizer.step() would do nothing
```

This is a real and common bug: forgetting `nn.Parameter(...)` around a tensor you intend to train. The tensor still has `requires_grad=True` if you set it manually, and gradients will still compute — but the optimizer (Chapter 6) won't know it exists, because `.parameters()` won't return it. Always wrap learnable tensors in `nn.Parameter`.

---

## 5.3 `forward()` and `__call__`: Why You Never Call `.forward()` Directly

Every `nn.Module` subclass defines a `forward()` method describing what happens when data passes through it. But you never call `.forward()` directly — you call the module itself, `layer(x)`.

The reason: `nn.Module.__call__` (inherited by every module) wraps your `forward()` with important bookkeeping — registered **hooks** (forward hooks, pre-hooks — covered in depth in Chapter 17), autograd graph bookkeeping, and mode-dependent behavior (`.train()` vs `.eval()`, relevant to layers like Dropout and BatchNorm in Chapter 19). Calling `.forward()` directly bypasses all of that silently.

```python
out = layer(x)          # correct — goes through __call__, runs hooks, etc.
out = layer.forward(x)  # works today, but skips hook machinery — avoid this
```

**Rule:** always call the module as if it were a function. Never call `.forward()` explicitly, even though it "works" in simple cases.

---

## 5.4 Composing Modules: Submodules and the Module Tree

Real models are built by composing smaller modules. Assign an `nn.Module` as an attribute of another `nn.Module`, and it's automatically registered as a **submodule** — this is the same mechanism as `nn.Parameter`, just one level up.

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(784, 128, 10)
print(model)
```

```
MLP(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
```

That printed structure isn't a coincidence — `nn.Module` builds an actual tree of submodules, and `.parameters()` recursively walks that entire tree:

```python
for name, param in model.named_parameters():
    print(name, param.shape)
# fc1.weight torch.Size([128, 784])
# fc1.bias   torch.Size([128])
# fc2.weight torch.Size([10, 128])
# fc2.bias   torch.Size([10])
```

Notice `relu` contributes nothing — `nn.ReLU()` has no learnable parameters, it's a pure function wrapped as a module for composability. This recursive discovery is exactly what makes `optimizer = torch.optim.Adam(model.parameters())` (Chapter 6) work correctly no matter how deeply nested your architecture is — a transformer with dozens of layers, each containing attention and feedforward submodules, still exposes every parameter through a single flat call to `.parameters()`.

### Nesting arbitrarily deep

```python
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = MLP(dim, dim * 4, dim)

    def forward(self, x):
        return x + self.mlp(x)   # residual connection

class DeepModel(nn.Module):
    def __init__(self, dim, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

model = DeepModel(dim=64, num_blocks=6)
print(sum(p.numel() for p in model.parameters()))   # total parameter count
```

### `nn.ModuleList` vs. a plain Python list

This deserves emphasis because it's a common and subtle bug:

```python
class Broken(nn.Module):
    def __init__(self, dim, num_blocks):
        super().__init__()
        self.blocks = [Block(dim) for _ in range(num_blocks)]  # plain list!

m = Broken(64, 6)
print(len(list(m.parameters())))   # 0 -- the blocks are invisible to nn.Module!
```

A plain Python `list` doesn't participate in `nn.Module`'s attribute-tracking machinery, so any modules stored inside it are invisible to `.parameters()`, `.to(device)`, and `.state_dict()`. **Always use `nn.ModuleList` (for a list of modules) or `nn.ModuleDict` (for a dict of modules)** instead of built-in Python containers whenever the contents are `nn.Module` instances.

---

## 5.5 `nn.Sequential`: The Fast Path for Linear Architectures

When a model is just a straight pipeline of layers with no branching, residuals, or conditional logic, `nn.Sequential` avoids writing a `forward()` method at all:

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

x = torch.rand(32, 784)
out = model(x)   # automatically chains each layer's output into the next
print(out.shape)   # torch.Size([32, 10])
```

This is exactly equivalent to writing an `MLP`-style class with a `forward()` that calls each layer in order. `nn.Sequential` is convenient, but it only works for strictly linear data flow — the moment you need a residual connection, a skip connection, multiple inputs, or any conditional branching (as in the `Block` example above with `x + self.mlp(x)`), you need a custom `forward()`.

You can also name the layers for a more readable structure:

```python
model = nn.Sequential(
    nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
    ),
)
# or, with explicit names via an OrderedDict:
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(784, 128)),
    ("relu1", nn.ReLU()),
    ("fc2", nn.Linear(128, 10)),
]))
print(model.fc1)   # accessible by name
```

---

## 5.6 Inspecting and Navigating a Model

A few methods you'll reach for constantly when debugging or modifying an existing architecture (relevant, for instance, when fine-tuning or freezing layers of a pretrained model in Chapter 14):

```python
model = MLP(784, 128, 10)

# List all named submodules
for name, module in model.named_modules():
    print(name, "->", module.__class__.__name__)

# List only direct children
for name, module in model.named_children():
    print(name, "->", module.__class__.__name__)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params}, Trainable: {trainable_params}")

# Access a submodule directly by attribute
print(model.fc1.weight.shape)
```

`named_modules()` includes the model itself plus every submodule at every depth; `named_children()` gives only the immediate, one-level-down children — useful when you want to iterate over a model's top-level structure without descending into nested blocks.

---

## Summary

- `nn.Module` is the base class underlying every layer and model in PyTorch. It automatically tracks parameters and submodules assigned as attributes.
- Always call `super().__init__()` first in a custom module's `__init__`.
- Wrap learnable tensors in `nn.Parameter` — a plain tensor, even with `requires_grad=True`, won't be discovered by `.parameters()`.
- Call modules as `model(x)`, never `model.forward(x)` — the former runs through `__call__`, which handles hooks and mode-dependent behavior.
- Submodules compose into a tree; use `nn.ModuleList`/`nn.ModuleDict` instead of plain Python containers so PyTorch can track modules stored inside a list or dict.
- `nn.Sequential` is a convenient shortcut for strictly linear architectures; anything with branching or residual connections needs a custom `forward()`.

## Exercises

1. Implement a custom `nn.Module` for a 2-layer MLP with a residual connection around the second layer (i.e., `output = layer2(activation(layer1(x))) + x`), and verify it with `print(model)` and `.named_parameters()`.
2. Rewrite the `Broken` example from Section 5.4 (plain Python list of modules) to use `nn.ModuleList`, and confirm `.parameters()` now finds all the submodules.
3. Build a model using `nn.Sequential` with an `OrderedDict` giving each layer a meaningful name, then access one of the layers by name and print its weight shape.
4. Write a function that takes any `nn.Module` and returns the total parameter count and trainable parameter count as a tuple, using `named_parameters()`.

**Next:** Chapter 6 covers loss functions and optimizers — `nn.functional`, `torch.optim`, and how to write a custom loss function, completing the pieces needed for a full training loop.
