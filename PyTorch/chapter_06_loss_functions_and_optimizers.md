# Chapter 6: Loss Functions & Optimizers

## Introduction

Chapter 5 gave you a model architecture that can produce predictions. This chapter gives you the other two pieces every training loop needs: a way to measure how wrong those predictions are (a **loss function**), and a way to use that measurement to update the model's parameters (an **optimizer**).

Both concepts are simple in isolation, but PyTorch's API around them has enough sharp edges — `nn.CrossEntropyLoss`'s input expectations being the most notorious — that this chapter is worth reading closely even if you've used other frameworks before.

By the end of this chapter you'll be able to:
- Choose the right built-in loss function for a task and use it correctly
- Understand the distinction between `torch.nn` loss modules and their `torch.nn.functional` counterparts
- Configure and use `torch.optim` optimizers correctly, including the parameter groups pattern
- Write a custom loss function of your own

---

## 6.1 `nn.functional` vs. `nn`: Two Ways to Call the Same Thing

Almost every operation in `torch.nn` — layers, activations, loss functions — has a stateless functional equivalent in `torch.nn.functional` (conventionally imported as `F`).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(3, 4)

# As a module (has state, needs to be instantiated)
relu_module = nn.ReLU()
out1 = relu_module(x)

# As a function (stateless, called directly)
out2 = F.relu(x)

torch.equal(out1, out2)   # True — identical computation
```

**When to use which:**

- Use the `nn` module version (`nn.ReLU`, `nn.CrossEntropyLoss`) when the operation is part of your model's structure — i.e., you assign it in `__init__` so it shows up in `print(model)` and is tracked consistently. This matters most for layers with actual learnable parameters or internal state (`nn.Linear`, `nn.BatchNorm2d`, `nn.Dropout` — the latter behaves differently in train vs. eval mode, covered in Chapter 19).
- Use the `F` functional version for simple, stateless operations applied inline inside `forward()`, especially activation functions, where instantiating a module in `__init__` just to call it once feels like unnecessary ceremony.

```python
class MLP(nn.Module):
    def __init__(self, in_f, hidden_f, out_f):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden_f)
        self.fc2 = nn.Linear(hidden_f, out_f)

    def forward(self, x):
        x = F.relu(self.fc1(x))   # functional style — no state needed
        return self.fc2(x)
```

Loss functions follow the same pattern (`nn.CrossEntropyLoss` vs. `F.cross_entropy`), and both are commonly used — `nn.CrossEntropyLoss()` instantiated once and reused is slightly more conventional in training loops, but they compute identically.

---

## 6.2 Common Loss Functions

### Regression: `nn.MSELoss`

Mean squared error — the default choice for continuous-valued targets.

```python
loss_fn = nn.MSELoss()
predictions = torch.tensor([2.5, 0.0, 2.1])
targets = torch.tensor([3.0, -0.5, 2.0])
loss = loss_fn(predictions, targets)
print(loss)   # tensor(0.1467)
```

### Classification: `nn.CrossEntropyLoss` — read this section carefully

This is the loss function you'll use constantly, and it's also the single biggest source of shape/dtype confusion for people new to PyTorch, because it does more than its name suggests.

**`nn.CrossEntropyLoss` expects raw, unnormalized logits — not probabilities.** It internally applies `log_softmax` and then negative log-likelihood in a single, numerically stable step. If you've already applied `softmax` to your model's output before passing it in, you're double-applying softmax, and your model will train incorrectly (usually silently — it may still "sort of" learn, but suboptimally, which makes this bug especially insidious).

```python
# Model outputs RAW LOGITS — no softmax applied
logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]])   # shape (batch=2, num_classes=3)

# Targets are CLASS INDICES (int64/long), NOT one-hot vectors
targets = torch.tensor([0, 1])   # shape (batch=2,)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)
print(loss)
```

Three requirements, all commonly violated by newcomers:

1. **Input must be raw logits**, shape `(batch, num_classes)` — do not apply `softmax` or `log_softmax` yourself first.
2. **Targets must be class indices**, shape `(batch,)`, dtype `torch.long` — not one-hot encoded vectors. (If you have one-hot targets, e.g. from a data source that produces them, convert with `targets.argmax(dim=1)`.)
3. **Targets dtype matters**: passing `float` targets raises a `RuntimeError` — `nn.CrossEntropyLoss` specifically requires `torch.long`.

```python
# Common mistake #1: applying softmax before the loss
probs = F.softmax(logits, dim=1)
# loss_fn(probs, targets)   # runs without error, but is WRONG — double softmax

# Common mistake #2: one-hot targets
one_hot_targets = torch.tensor([[1, 0, 0], [0, 1, 0]])
# loss_fn(logits, one_hot_targets)   # RuntimeError: expected scalar type Long
```

If you specifically need `softmax` probabilities elsewhere (e.g., for inference-time confidence scores), apply it *separately* from the loss computation, on the raw logits:

```python
logits = model(x)
loss = loss_fn(logits, targets)          # loss uses raw logits directly
probs = F.softmax(logits, dim=1)          # separate call, for interpretation only
predicted_class = probs.argmax(dim=1)     # or equivalently, logits.argmax(dim=1)
```

Note that `argmax` on logits and `argmax` on softmax(logits) always agree, since softmax is monotonic — you don't need softmax at all just to get a predicted class, only when you want an actual probability value.

### Binary classification: `nn.BCEWithLogitsLoss`

For binary (or multi-label) classification, use `nn.BCEWithLogitsLoss`, not `nn.BCELoss` — the "WithLogits" version combines a sigmoid and binary cross-entropy in one numerically stable step, exactly analogous to how `CrossEntropyLoss` combines softmax internally.

```python
logits = torch.tensor([1.5, -0.3, 2.0])     # raw logits, shape (batch,)
targets = torch.tensor([1.0, 0.0, 1.0])       # float targets, 0 or 1

loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, targets)
```

As with cross-entropy, avoid manually applying `sigmoid()` before this loss — `BCEWithLogitsLoss` does it internally, more stably than a separate `sigmoid` + `BCELoss` pair (it avoids intermediate overflow/underflow issues that show up in naive implementations).

---

## 6.3 Optimizers: `torch.optim`

An optimizer takes the gradients computed by `.backward()` (Chapter 3) and updates each parameter accordingly. PyTorch's optimizer API is deliberately uniform — swapping SGD for Adam is a one-line change.

```python
model = MLP(784, 128, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# or:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Every optimizer is constructed the same way: pass it the parameters it should manage (almost always `model.parameters()`) plus hyperparameters like learning rate.

### The standard training step

```python
for x_batch, y_batch in dataloader:      # dataloader introduced properly in Chapter 7
    optimizer.zero_grad()                  # Chapter 3: gradients accumulate, so reset first
    predictions = model(x_batch)
    loss = loss_fn(predictions, y_batch)
    loss.backward()                        # compute gradients
    optimizer.step()                        # apply the update using those gradients
```

This four-line sequence — `zero_grad()`, forward pass, `backward()`, `step()` — is the heartbeat of virtually every PyTorch training loop you will ever write, from a toy MLP to a multi-billion-parameter LLM fine-tune.

### Common optimizers, briefly

| Optimizer | Notes |
|---|---|
| `torch.optim.SGD` | Simple, well-understood; often needs `momentum` to converge well |
| `torch.optim.Adam` | Adaptive learning rates per-parameter; strong default choice |
| `torch.optim.AdamW` | Adam with *decoupled* weight decay — generally preferred over Adam for modern deep learning, including transformer fine-tuning |

For the LLM fine-tuning work you've done with Qwen2.5-Coder (SFT/GRPO), `AdamW` is the standard choice you'll see in `SFTTrainer` configs and most published training recipes — its decoupled weight decay behaves more predictably than plain Adam's, especially at the learning rates and batch sizes typical of LLM fine-tuning.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

### Parameter groups: different hyperparameters for different parts of a model

A pattern you'll need the moment you want, say, a lower learning rate for a pretrained backbone and a higher one for a newly added classification head (directly relevant to transfer learning, Chapter 14):

```python
optimizer = torch.optim.Adam([
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(), "lr": 1e-3},
])
```

Each dictionary is a **parameter group**; any hyperparameter not specified falls back to a default you can also pass to the optimizer constructor. This is the same mechanism that learning rate schedulers (Chapter 18) hook into when they adjust `lr` per group over the course of training.

---

## 6.4 Writing a Custom Loss Function

Most custom losses are just plain Python functions operating on tensors — since loss functions are typically stateless, you often don't even need an `nn.Module` subclass:

```python
def custom_mse_with_l1_penalty(predictions, targets, model, l1_lambda=1e-4):
    mse = F.mse_loss(predictions, targets)
    l1_penalty = sum(p.abs().sum() for p in model.parameters())
    return mse + l1_lambda * l1_penalty
```

When a loss needs configuration state (e.g., class weights fixed at construction time) or you want it to appear cleanly in a model summary, wrap it as an `nn.Module`, following exactly the pattern from Chapter 5:

```python
class WeightedMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight   # a scalar or per-sample tensor

    def forward(self, predictions, targets):
        return (self.weight * (predictions - targets) ** 2).mean()

loss_fn = WeightedMSELoss(weight=2.0)
loss = loss_fn(predictions, targets)
```

A concrete, more realistic example: a custom loss relevant to structured-output fine-tuning work like your docstring-generation projects, combining a standard cross-entropy term with a penalty for missing a required structural token (a simplified illustration of the kind of programmatic verifier logic used in GRPO/RLVR-style reward shaping):

```python
def structured_output_loss(logits, targets, required_token_id, penalty=0.5):
    ce = F.cross_entropy(logits, targets)
    predicted_ids = logits.argmax(dim=-1)
    missing_token = (predicted_ids == required_token_id).float().mean() < 0.5
    penalty_term = penalty if missing_token else 0.0
    return ce + penalty_term
```

This is intentionally simplified — real reward/verifier functions for RLVR are typically non-differentiable and used outside the standard `.backward()` path entirely (they score complete generations, not per-token logits) — but the pattern of *combining a differentiable loss term with additional structural signal* is directly relevant groundwork for the GRPO material later in this curriculum.

---

## Summary

- `torch.nn` module versions and `torch.nn.functional` versions compute identically; use modules for stateful layers assigned in `__init__`, functional calls for simple stateless operations inline in `forward()`.
- `nn.CrossEntropyLoss` expects raw logits (not softmax output) and integer class-index targets (not one-hot, dtype `torch.long`) — this is the most common source of subtle training bugs for newcomers.
- `nn.BCEWithLogitsLoss` similarly expects raw logits and combines sigmoid + BCE in one numerically stable step.
- Every optimizer follows the same `zero_grad()` → forward → `backward()` → `step()` pattern; `AdamW` is the standard modern default, especially for LLM fine-tuning.
- Parameter groups let you assign different hyperparameters (e.g., learning rate) to different parts of a model — essential for transfer learning and scheduling.
- Custom losses are usually just functions; wrap them as `nn.Module` subclasses only when they carry configuration state.

## Exercises

1. Take a batch of logits with shape `(8, 5)` (8 samples, 5 classes) and class-index targets, and compute `nn.CrossEntropyLoss` two ways: directly, and manually via `F.log_softmax` + `F.nll_loss`. Confirm they match.
2. Deliberately pass one-hot targets to `nn.CrossEntropyLoss`, observe the error, and fix it.
3. Build a two-layer MLP, set up an `AdamW` optimizer with two parameter groups (different learning rates for each layer), and print `optimizer.param_groups` to confirm the configuration.
4. Write a custom loss function that combines `nn.MSELoss` with an L2 penalty on a specific submodule's weights only (not the whole model), and explain why you might want to penalize only part of a model rather than all of it.

**Next:** Chapter 7 covers `Dataset` and `DataLoader` — how to load, batch, and shuffle data efficiently, including writing custom datasets and collate functions.
