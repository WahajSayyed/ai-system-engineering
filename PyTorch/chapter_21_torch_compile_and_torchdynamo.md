# Chapter 21: `torch.compile` & TorchDynamo

## Introduction

Chapter 20 established profiling as the first step toward performance — and one common finding, especially for CPU-bound training loops, is that a meaningful amount of time is lost not to the actual mathematical computation, but to Python-level overhead: dispatching individual operations one at a time, launching many small GPU kernels separately when they could be fused into one. `torch.compile` addresses this directly: it captures your model's computation as a graph, applies optimizations across that graph as a whole (operator fusion being the most impactful), and generates faster code — often with just one line added to existing, unmodified model code.

By the end of this chapter you'll be able to:
- Explain what `torch.compile` and TorchDynamo actually do, at a conceptual level
- Apply `torch.compile` to a model and measure the resulting speedup correctly
- Understand graph breaks — what causes them and why they matter
- Use `fullgraph=True` and PyTorch's diagnostic tooling to find and fix graph breaks
- Choose an appropriate compilation mode for a given situation

---

## 21.1 What `torch.compile` Actually Does

`torch.compile` wraps a model or function and, the first time it's called with a given set of input shapes/types, traces through the Python code to build an intermediate graph representation of the computation — this tracing component is called **TorchDynamo**. That graph is then handed to a compiler backend (by default, **TorchInductor**) which generates optimized, often fused, GPU (or CPU) kernels for it. Subsequent calls with the same input characteristics reuse the compiled version directly, skipping Python-level dispatch overhead entirely.

```python
import torch
import torch.nn as nn

model = FashionMNIST_CNN().to("cuda")   # Chapter 10
compiled_model = torch.compile(model)

x = torch.rand(64, 1, 28, 28, device="cuda")
output = compiled_model(x)   # first call: triggers tracing + compilation, slower than usual
output = compiled_model(x)   # subsequent calls: uses the compiled, optimized version
```

The **first call is slower**, not faster — this is compilation overhead, and it's a critical detail for correctly measuring `torch.compile`'s benefit: benchmarking only the first call (or forgetting to run a warm-up call before timing) will make `torch.compile` look like it *hurts* performance, when in fact you've only measured the one-time compilation cost, not the steady-state speedup it provides across the many subsequent calls in a real training run.

```python
import time

# WRONG way to benchmark -- includes compilation overhead in the measurement
start = time.time()
output = compiled_model(x)
torch.cuda.synchronize()
print(f"First call: {time.time() - start:.3f}s")   # misleadingly slow

# CORRECT way -- warm up first, then measure steady-state performance
compiled_model(x)   # warm-up call, triggers compilation, result discarded
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    output = compiled_model(x)
torch.cuda.synchronize()   # Chapter 20, Section 20.4 -- required for accurate GPU timing
print(f"Average per call: {(time.time() - start) / 100 * 1000:.3f}ms")
```

---

## 21.2 Applying `torch.compile` to a Training Loop

`torch.compile` works directly on `nn.Module` instances, integrating into the exact training loop structure from Chapter 8/9 with no other changes required:

```python
model = FashionMNIST_CNN().to(device)
model = torch.compile(model)   # the only line added compared to earlier chapters

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
```

A few practical realities worth setting expectations around:

- **Compilation happens per unique input shape.** If your batches vary in shape (e.g., the last batch of an epoch is smaller when `drop_last=False`, Chapter 7), `torch.compile` recompiles for that new shape the first time it's seen — usually a minor, one-time cost per distinct shape, but worth knowing about if you observe periodic slowdowns at predictable points (like the end of every epoch).
- **`torch.compile` composes with mixed precision (Chapter 15)** — wrapping an already-`autocast`-enabled training step works as expected, and the two techniques are commonly used together in practice.
- **Speedup varies substantially by model.** Models with many small, simple operations (where kernel launch overhead is a larger fraction of total time) tend to benefit more than models already dominated by a few very large operations (like one enormous matrix multiplication, where launch overhead was never a significant fraction of the total time to begin with).

---

## 21.3 Graph Breaks

TorchDynamo can't trace through arbitrary Python code — certain constructs (data-dependent control flow based on tensor values, calls to unsupported library functions, certain uses of `.item()`) force it to fall back to normal, uncompiled Python execution for that portion of the code, then resume compiled execution afterward. This fallback point is called a **graph break**, and it's a strict performance concern, not a correctness one: `torch.compile(fullgraph=False)` (the default) handles graph breaks gracefully and still produces correct results, but each break interrupts kernel fusion opportunities across that boundary, limiting the speedup compiled regions on either side of a break can independently achieve.

```python
@torch.compile
def fn(x):
    y = x.sum()
    if y > 0:              # data-dependent branching on a tensor's VALUE, not just its shape
        return x + y.item()
    return x - y.item()

print(fn(torch.ones(3, 3)))   # still produces the correct result -- but with a graph break
```

Branching on `y > 0`, where `y` is a tensor whose actual runtime value depends on the input data, is a canonical graph-break trigger: TorchDynamo would need to know, at compile time, which branch to trace — but the branch taken depends on data only available at runtime, so it can't build a single unified graph across that `if` statement. The code still runs and still produces correct output; it just runs that specific branching logic in regular, uncompiled Python rather than as part of a fused compiled graph.

### `fullgraph=True`: forcing errors instead of silent graph breaks

For understanding and eliminating graph breaks deliberately (rather than letting them pass silently), `fullgraph=True` makes any unsupported construct raise an explicit error instead of falling back:

```python
compiled_fn = torch.compile(fn, fullgraph=True)

try:
    compiled_fn(torch.ones(3, 3))
except Exception as e:
    print(e)
```

```
torch._dynamo.exc.Unsupported: Data-dependent branching
Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`).
Dynamo does not support tracing dynamic control flow.
Hint: This graph break is fundamental - it is unlikely that Dynamo will ever
be able to trace through your code. Consider finding a workaround.
Hint: Use `torch.cond` to express dynamic control flow.
```

PyTorch's graph-break diagnostics are specifically designed to be actionable — the error names the exact construct that broke tracing and typically suggests a concrete workaround (here, `torch.cond`, a primitive specifically for expressing this kind of dynamic branching in a traceable way, beyond the scope of this introductory chapter). **Practical guidance:** develop with the default `fullgraph=False` to get a working, correct baseline first; once things work end to end, try `fullgraph=True` specifically to surface and evaluate graph breaks you may want to eliminate for additional performance — treating it as a diagnostic tool during optimization, not a requirement for every compiled model to satisfy.

### Diagnosing graph breaks in an existing model

For a model where you don't already know where breaks occur, PyTorch's logging flags surface this directly without needing `fullgraph=True`:

```python
import torch._dynamo

torch._dynamo.config.verbose = True

# Or, via environment variable before running the script:
# TORCH_LOGS="graph_breaks" python train.py
```

Running with `TORCH_LOGS="graph_breaks"` prints a report of every graph break encountered during a run, along with the source location responsible — the practical starting point for deciding which breaks are worth fixing (typically ones inside a model's hot inner loop, called many times) versus which are harmless (a break in one-time setup code, called once and never again).

---

## 21.4 Common Graph Break Causes and Workarounds

A few patterns worth recognizing directly, since they account for a large share of graph breaks encountered in practice:

### `.item()` and other direct data access

```python
# Breaks: .item() extracts a Python scalar, forcing a sync point Dynamo can't trace through cleanly
def fn(x):
    total = x.sum().item()
    return x / total

# Workaround: keep the computation in tensor form as long as possible
def fn_fixed(x):
    total = x.sum()
    return x / total   # division by a 0-dim tensor works fine, no .item() needed
```

This connects directly to Chapter 8, Section 8.4's guidance about `.item()` — the same operation that's necessary for safely accumulating a loss value for logging (to avoid leaking the autograd graph) is also a common graph-break source when used inside a function you intend to compile. The resolution in both cases is the same underlying principle: use `.item()` at the boundary where you genuinely need a Python scalar (logging, printing), and avoid it inside computation that should stay as pure tensor operations.

### Print statements and logging

Printing or logging inside a compiled function also triggers a graph break by default, since these are side-effecting operations Dynamo can't safely fold into a pure computational graph. If you have debug prints left in a `forward()` method, they're worth removing (or gating behind a flag that's `False` during compiled execution) before compiling that model for real training.

### Unsupported third-party library calls

Calls into libraries TorchDynamo doesn't know how to trace through (some NumPy operations mixed into a `forward()`, certain custom C++ extensions without special registration) will break tracing at that call. The general fix is either to express the same logic using standard PyTorch tensor operations Dynamo does understand, or, for a genuinely necessary external call, wrapping it appropriately so Dynamo treats it as an opaque, non-traced unit rather than attempting (and failing) to trace through its internals.

---

## 21.5 Compilation Modes

`torch.compile` accepts a `mode` argument trading off compilation time against runtime performance:

```python
model = torch.compile(model, mode="default")          # balanced -- good default choice
model = torch.compile(model, mode="reduce-overhead")    # lower Python overhead, uses CUDA graphs internally
model = torch.compile(model, mode="max-autotune")        # most aggressive kernel-level optimization search
```

- **`"default"`**: a reasonable balance of compilation time and runtime speedup — the right starting point for most use cases.
- **`"reduce-overhead"`**: additionally uses CUDA graphs to further cut CPU-side dispatch overhead, particularly beneficial for models where many small operations dominate (exactly the CPU-bound scenario diagnosed via profiling in Chapter 20). Comes with a real caveat: CUDA graphs capture and replay a fixed sequence of GPU operations, which interacts poorly with any dynamic behavior (data-dependent shapes or control flow) — best suited to models with a highly regular, static computation graph.
- **`"max-autotune"`**: searches more exhaustively over possible kernel implementations and fusion strategies to find the fastest configuration, at the cost of substantially longer compilation time upfront. Worth using specifically for a model you'll run many, many times (a long training run, or a model being deployed for repeated inference) where the extra one-time compilation cost is easily amortized — not worth it for quick experimentation or a model you'll only run a handful of times.

**Practical guidance:** start with `"default"`, confirm `torch.compile` is providing a measurable benefit for your specific model (via the correctly-warmed-up benchmarking pattern from Section 21.1), and only reach for `"reduce-overhead"` or `"max-autotune"` once you have a stable, production-shaped training or inference setup where the additional compilation cost and mode-specific caveats are worth investigating further.

---

## Summary

- `torch.compile` traces a model's computation (via TorchDynamo) into a graph and compiles it (via TorchInductor) into fused, optimized kernels — reducing the Python-level dispatch overhead that profiling (Chapter 20) often reveals as a bottleneck.
- The first call after compiling is slower due to one-time compilation cost — always warm up before benchmarking, and use `torch.cuda.synchronize()` for accurate GPU timing (Chapter 20, Section 20.4).
- Graph breaks occur when Dynamo can't trace through a piece of code (commonly: data-dependent branching, `.item()`, print statements, unsupported library calls) — they preserve correctness but limit fusion opportunities across the break.
- `fullgraph=True` turns graph breaks into explicit, diagnosable errors; `TORCH_LOGS="graph_breaks"` surfaces them during normal (non-`fullgraph`) execution.
- Compilation modes (`"default"`, `"reduce-overhead"`, `"max-autotune"`) trade off compilation time against runtime performance — start with `"default"` and move to more aggressive modes only once a stable, high-repetition use case justifies the extra upfront cost.

## Exercises

1. Benchmark the `FashionMNIST_CNN` from Chapter 10 with and without `torch.compile`, using the correct warm-up methodology from Section 21.1, and report the steady-state speedup observed.
2. Write a small function containing a data-dependent `if` statement on a tensor's value, compile it with `fullgraph=True`, and read the resulting error to confirm it correctly identifies the branching as the cause.
3. Find or construct a model containing a stray `print()` statement inside its `forward()` method, run it with `TORCH_LOGS="graph_breaks"`, and confirm the print statement is reported as a graph break source. Remove it and confirm the break disappears.
4. Compare `"default"` and `"reduce-overhead"` compilation modes on the same model, measuring both total compilation time (the first, warm-up call) and steady-state per-batch time (subsequent calls) for each.

**Next:** Chapter 22 begins multi-GPU training — comparing `DataParallel` and `DistributedDataParallel`, and explaining why the latter is the standard choice for any serious multi-GPU setup, including the multi-node configuration in your own lab.
