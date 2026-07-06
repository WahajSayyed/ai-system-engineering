# Chapter 32: Debugging & Testing PyTorch Code

## Introduction

Every chapter in this book has built toward writing correct, efficient PyTorch code — but "correct" has mostly been demonstrated by example, not by a systematic practice of verification. This final chapter closes that gap: reproducibility (making bugs and results repeatable in the first place), `torch.testing` (writing genuine unit tests for tensor code), and a catalog of the failure modes this book has flagged throughout — collected here as a single, practical debugging reference for when something inevitably goes wrong in a real project.

By the end of this chapter you'll be able to:
- Set up full reproducibility across PyTorch, NumPy, and Python's own randomness
- Write meaningful unit tests for PyTorch code using `torch.testing.assert_close`
- Systematically debug NaN/Inf losses, shape mismatches, and silent training failures
- Use a consolidated checklist that ties every debugging technique in this book back to one reference

---

## 32.1 Reproducibility: Seeding Everything

A bug that only sometimes appears is far harder to debug than one that appears reliably — the first step toward systematic debugging is often just making an unstable-seeming result *deterministically* reproducible, so you can actually iterate on a fix and confirm it worked.

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)               # Python's own random module
    np.random.seed(seed)             # NumPy (Chapter 4) -- affects any NumPy-based data augmentation, etc.
    torch.manual_seed(seed)           # PyTorch's CPU random number generator
    torch.cuda.manual_seed_all(seed)   # PyTorch's GPU random number generator(s), all devices

    # Force deterministic algorithm selection where available -- see caveat below
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

Three separate random number generators — Python's, NumPy's, and PyTorch's — need seeding independently, since they're entirely separate implementations with no shared state; forgetting one is a common reason a script *seems* to be seeded (some things repeat) while other parts (a NumPy-based data augmentation step, say) keep varying between runs.

### The determinism/performance trade-off

`torch.backends.cudnn.deterministic = True` deserves an explicit callout: cuDNN (the library backing most of PyTorch's GPU convolution and RNN implementations) often has multiple algorithm choices for the same operation, some faster but non-deterministic (producing tiny numerical differences between runs due to floating-point summation order varying with parallel execution) and some slower but exactly deterministic. Setting `deterministic = True` forces the slower, reproducible path — worth doing specifically while debugging (when you need a fix to be verifiably reproducible) but often worth reverting for actual production training runs, where the resulting slowdown isn't justified once the code is already correct and stable.

```python
torch.use_deterministic_algorithms(True)   # a broader, stricter version -- raises an error instead of
                                              # silently using a non-deterministic op where no
                                              # deterministic implementation exists at all
```

`torch.use_deterministic_algorithms(True)` is a stronger, PyTorch-wide version of the same idea, covering more than just cuDNN — worth reaching for when debugging a result that seems to vary between runs in ways you can't otherwise account for, though be aware it can raise an error for certain operations with no deterministic implementation available at all, at which point you'd need to substitute an alternative operation or accept that specific source of nondeterminism.

### `DataLoader` workers and seeding

A subtlety directly connecting back to Chapter 7: `DataLoader` worker processes (`num_workers > 0`, Section 7.5) are separate processes with their *own* random state, not automatically synchronized with the main process's seed — if your `Dataset.__getitem__` uses randomness (e.g., random data augmentation), you need an explicit `worker_init_fn` to seed each worker deterministically:

```python
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, worker_init_fn=worker_init_fn)
```

---

## 32.2 Writing Real Tests: `torch.testing.assert_close`

Throughout this book, correctness checks have used `torch.allclose` (Chapters 4, 16, 25, 30) and manual `print`-and-inspect debugging. For genuine, maintainable unit tests — the kind worth keeping in a test suite rather than running once and discarding — `torch.testing.assert_close` is the more appropriate tool: it raises a clear, informative `AssertionError` on failure (rather than silently returning `False`, as `allclose` does), making it suitable for use inside an actual test framework like `pytest`.

```python
import torch
from torch.testing import assert_close

def test_linear_layer_shape():
    layer = torch.nn.Linear(10, 5)
    x = torch.rand(3, 10)
    output = layer(x)
    assert output.shape == (3, 5)

def test_custom_relu_matches_builtin():
    def my_relu(x):
        return torch.where(x > 0, x, torch.zeros_like(x))

    x = torch.randn(100)
    assert_close(my_relu(x), torch.relu(x))   # raises a detailed error on any mismatch, unlike allclose

def test_gradient_flows():
    layer = torch.nn.Linear(10, 5)
    x = torch.rand(3, 10, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None   # Chapter 3 -- confirms gradients actually reached the input
    assert not torch.isnan(x.grad).any()   # Chapter 15/32.3 -- confirms no NaN corruption
```

`assert_close` accepts `rtol`/`atol` tolerances exactly like `allclose` (Chapters 25/27's verification patterns), but produces output specifically designed for debugging test failures — reporting exactly which elements mismatched and by how much, rather than a bare `True`/`False`. **Practical guidance:** reach for `assert_close` (inside an actual test file, run via `pytest`) for anything you expect to run repeatedly as part of ongoing development — a custom autograd function (Chapter 16), a custom CUDA kernel (Chapter 25), a data preprocessing pipeline (Chapter 7) — and reserve one-off `print`/`allclose` checks for genuinely exploratory, single-use debugging.

### A minimal test suite structure

```python
# test_model.py
import torch
import pytest
from mymodel import FashionMNIST_CNN   # Chapter 10

@pytest.fixture
def model():
    torch.manual_seed(0)   # Section 32.1 -- reproducible fixture
    return FashionMNIST_CNN()

def test_output_shape(model):
    x = torch.rand(4, 1, 28, 28)
    output = model(x)
    assert output.shape == (4, 10)

def test_output_is_finite(model):
    x = torch.rand(4, 1, 28, 28)
    output = model(x)
    assert torch.isfinite(output).all()   # Section 32.3 -- catches NaN/Inf immediately, not deep in training

def test_backward_produces_gradients(model):
    x = torch.rand(4, 1, 28, 28, requires_grad=True)
    output = model(x)
    output.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
```

This kind of lightweight test suite — shape checks, finiteness checks, gradient-flow checks — catches an entire class of bugs (a typo in a `reshape` call, Chapter 10's shape-tracing concern; a layer accidentally excluded from the gradient path) at the moment they're introduced, via `pytest` in a normal development workflow, rather than discovering them only after an expensive, hours-long training run has already produced a suspiciously bad result.

---

## 32.3 Debugging NaN and Inf Losses

A NaN (or Inf) loss is one of the most common — and most frustrating, if approached without a system — failures in deep learning. This section consolidates the specific causes flagged throughout this book into one systematic debugging procedure.

### Step 1: Confirm exactly where the NaN first appears

```python
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}!")
        return True
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}!")
        return True
    return False

# Sprinkled through a forward pass during debugging:
x = layer1(x)
check_for_nan(x, "after layer1")
x = layer2(x)
check_for_nan(x, "after layer2")
```

For a more systematic version of this across an entire model without manually instrumenting every layer, directly reuse Chapter 17's forward hooks:

```python
def make_nan_check_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor) and torch.isnan(output).any():
            print(f"NaN detected in output of {name}")
    return hook

for name, layer in model.named_modules():   # Chapter 5, Section 5.6
    layer.register_forward_hook(make_nan_check_hook(name))
```

This is precisely the pattern from Chapter 17, Section 17.4's gradient health check, applied to forward activations instead of backward gradients — running both together on a model that's producing NaN losses is often the fastest way to localize the problem to a specific layer, rather than only knowing the *final* loss is NaN.

### Step 2: Match the location to a known cause

A checklist of the specific NaN/Inf causes flagged throughout this book, roughly in order of how commonly they're the actual culprit:

| Symptom location | Likely cause | Chapter reference |
|---|---|---|
| NaN in loss, `float16` training | Gradient underflow/overflow without proper scaling | Ch. 15, Section 15.3 |
| NaN after a `log()` or `log_softmax` | Input contains exactly 0, or logits are extreme | Ch. 6, Section 6.2 (avoid manual softmax + separate log) |
| NaN after several epochs, previously fine | Exploding gradients — check if clipping is applied | Ch. 8, Section 8.5 |
| Inf in attention scores | Missing or wrong scaling factor ($\sqrt{d_k}$) | Ch. 12, Section 12.2 |
| NaN specific to one GPU in a DDP run | A single process encountered a genuinely bad batch (corrupt data) | Ch. 23 |
| NaN immediately at step 0, before any training | Bad weight initialization, or a data preprocessing bug (e.g., unnormalized inputs with extreme values) | Ch. 9, Section 9.1 |

### Step 3: Check learning rate and data as a fast first pass

Before diving into hooks and instrumentation, two fast checks are worth trying first, since they account for a large fraction of real-world NaN cases: **reduce the learning rate by 10x** (Chapter 18) and re-run a few steps — if the NaN disappears, the original learning rate was likely too aggressive for the current loss landscape; and **check the input data directly** for unexpected extreme values or actual NaNs already present in the dataset (surprisingly common with real-world data, especially anything derived from sensor readings or web-scraped sources) using a simple assertion in the data loading path:

```python
def check_batch(x, y):
    assert torch.isfinite(x).all(), "Input batch contains NaN/Inf!"
    assert torch.isfinite(y).all() if y.dtype.is_floating_point else True

for x_batch, y_batch in train_loader:
    check_batch(x_batch, y_batch)   # fails fast, at the data-loading step, not deep inside the model
    # ... rest of training step ...
```

---

## 32.4 Debugging Shape Mismatches

`RuntimeError: mat1 and mat2 shapes cannot be multiplied` (or similar) is arguably the single most common error in PyTorch code, spanning nearly every chapter in this book — a systematic approach beats trial-and-error:

1. **Print shapes at every step of a suspect forward pass** — the cheapest, fastest diagnostic, directly extending the shape-tracing habit established in Chapter 10, Section 10.4:

```python
def forward(self, x):
    print(f"input: {x.shape}")
    x = self.conv1(x)
    print(f"after conv1: {x.shape}")
    x = self.flatten(x)
    print(f"after flatten: {x.shape}")
    x = self.fc1(x)   # error likely occurs HERE if flatten's output doesn't match fc1's expected input
    return x
```

2. **Compute expected shapes by hand first**, using the formulas from Chapter 10 (convolution output shape) or straightforward multiplication (flattened size) — comparing the hand-computed expectation against the actual printed shape immediately localizes whether the bug is in your shape *reasoning* or in the code *not matching* that reasoning.

3. **Check batch dimension consistency** — a very common variant: a function that works correctly for a single sample silently breaks (or, worse, silently produces a *wrong but shaped-correctly* result) when given a batch, typically from an operation that should have included a `dim=` argument (Chapter 2's reshape operations, Chapter 12's softmax `dim=-1` requirement) but defaulted to operating over the wrong axis instead.

---

## 32.5 A Consolidated Debugging Checklist

A single reference tying together every major debugging tool introduced across this entire book, organized by what you're trying to diagnose:

| Problem | Tool | Chapter |
|---|---|---|
| "Is my custom `backward()` correct?" | `gradcheck` | Ch. 16, Section 16.5 |
| "Is my custom CUDA kernel correct?" | `gradcheck` + fair benchmarking | Ch. 25, Section 25.5 |
| "Where is time actually being spent?" | `torch.profiler` | Ch. 20 |
| "Is the GPU or the data loader the bottleneck?" | Synthetic-data test | Ch. 20, Section 20.4 |
| "What are this layer's activations/gradients?" | Forward/backward hooks | Ch. 17 |
| "Is my model overfitting?" | Train/val loss gap | Ch. 19, Section 19.1 |
| "Why is my loss NaN?" | Hooks + the checklist in 32.3 | Ch. 15, 17, 32.3 |
| "Why is this shape wrong?" | Manual shape tracing | Ch. 10, Section 10.4; 32.4 |
| "Does my exported model match the original?" | `assert_close`/`allclose` on outputs | Ch. 25, 27 |
| "Is my custom optimizer correct?" | Step-by-step comparison against the built-in | Ch. 28, Section 28.4 |
| "Is my result actually reproducible?" | `set_seed()`, deterministic algorithms | Ch. 32, Section 32.1 |

This table is deliberately meant to function as a genuine reference — when something goes wrong in a future project, the fastest path forward is usually recognizing which row of this table matches the symptom, then revisiting the corresponding chapter's specific technique, rather than debugging from first principles each time.

---

## Summary

- Full reproducibility requires seeding Python's `random`, NumPy, and PyTorch (CPU and CUDA) independently, plus explicit `DataLoader` worker seeding — `torch.use_deterministic_algorithms(True)` trades performance for stronger determinism guarantees, worth using while debugging and reconsidering for production runs.
- `torch.testing.assert_close`, used inside a real test suite (`pytest`), is the appropriate tool for maintainable correctness checks — shape, finiteness, and gradient-flow tests catch bugs at introduction time rather than after an expensive training run.
- NaN/Inf debugging benefits from a systematic procedure: localize with hooks, match the location against a checklist of known causes (numerical instability, exploding gradients, bad initialization, data issues), and try fast first passes (lower learning rate, check input data) before deeper instrumentation.
- Shape mismatches are best debugged by printing shapes through a forward pass, computing expected shapes by hand, and checking for missing `dim=` arguments — the single most common category of PyTorch error across this entire book.
- The consolidated debugging table in Section 32.5 is worth keeping as a standing reference — matching a symptom to the right tool is usually faster than re-deriving a debugging approach from scratch.

## Exercises

1. Write a `set_seed()` function (Section 32.1) and verify it produces identical model initialization, identical `DataLoader` batch ordering (with `shuffle=True`), and identical training loss curves across two separate runs of the same script.
2. Write a small `pytest` test suite (Section 32.2) for a model of your choice from an earlier chapter, covering output shape, output finiteness, and gradient flow to every parameter — intentionally introduce a bug (e.g., accidentally freeze a layer) and confirm your test suite catches it.
3. Deliberately construct a training scenario that produces a NaN loss (e.g., an unreasonably high learning rate on an unnormalized dataset), then use the forward-hook NaN-detection pattern from Section 32.3 to localize exactly which layer's output first contains a NaN.
4. Take a model with a deliberately introduced shape bug (e.g., an incorrect flattened size passed to a `Linear` layer, Chapter 10, Section 10.4), and practice the systematic shape-debugging procedure from Section 32.4 to locate and fix it without simply guessing at the correct number.

---

# Part VI Recap, and Curriculum Conclusion

Part VI covered the specialized, advanced techniques that sit atop everything built in Parts I-V: custom CUDA extensions tying directly into your CUDA curriculum, quantization for efficient deployment, model export for production, custom optimizers, and the RL/LoRA/GRPO machinery underlying modern LLM fine-tuning — closing with this chapter's debugging and testing practices, which apply to every single piece of code written across all 32 chapters.

You now have a complete, code-first path from `torch.tensor([1, 2, 3])` to a working understanding of the exact techniques behind your own Qwen2.5-Coder GRPO pipeline, your CUDA kernel work, and the deployment concerns relevant to your conveyor sorting system — the full arc this curriculum set out to cover.
