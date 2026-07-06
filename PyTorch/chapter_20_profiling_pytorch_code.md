# Chapter 20: Profiling PyTorch Code

## Introduction

Parts I through IV covered how to build and train models correctly. Part V shifts focus to a different question: is your training run actually using your hardware efficiently? Intuition about where time is being spent — "it's probably the data loading" or "the GPU must be the bottleneck" — is frequently wrong, and optimizing the wrong part of a pipeline wastes effort while leaving the real bottleneck untouched. `torch.profiler` gives you a precise, measured answer instead of a guess.

This chapter opens Part V by establishing profiling as the first step before any performance optimization — including the more advanced techniques covered later in this part (`torch.compile` in Chapter 21, multi-GPU training in Chapters 22-23, memory optimization in Chapter 24).

By the end of this chapter you'll be able to:
- Profile a training loop's CPU and GPU time using `torch.profiler`
- Distinguish between a GPU-bound, CPU-bound, and data-loading-bound training loop
- Read and interpret a profiler trace, including the Chrome trace viewer
- Use profiling to identify and confirm the effect of common bottlenecks

---

## 20.1 Why "It Feels Slow" Isn't a Diagnosis

A training loop can be slow for fundamentally different reasons, each requiring a completely different fix:

- **GPU-bound**: the GPU is the bottleneck — it's continuously busy, and the CPU/data pipeline is keeping up with it easily. Fixes: mixed precision (Chapter 15), `torch.compile` (Chapter 21), or a more efficient model architecture.
- **CPU-bound**: Python-level overhead (kernel launch overhead, excessive small operations, inefficient custom code) is the bottleneck — the GPU sits idle waiting for the CPU to issue the next instruction. Fixes: `torch.compile` (Chapter 21, which can fuse operations and reduce launch overhead), batching operations more efficiently, reducing unnecessary `.item()` or `.cpu()` calls that force synchronization.
- **Data-loading-bound**: the GPU sits idle waiting for the next batch of data to be loaded and preprocessed. Fixes: more `DataLoader` workers, `pin_memory` (Chapter 7), or faster preprocessing.

Applying a GPU-focused optimization (like mixed precision) to a data-loading-bound pipeline yields essentially no speedup, because the GPU was never the constraint in the first place — the training loop simply spends more time idle, waiting slightly less long for data that still arrives too slowly. **Profiling exists specifically to distinguish these cases before you spend time optimizing.**

---

## 20.2 A Minimal Profiling Example

`torch.profiler.profile` wraps a section of code and records detailed timing and memory information about every operation executed within it:

```python
import torch
from torch.profiler import profile, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNIST_CNN().to(device)   # Chapter 10
x_batch = torch.rand(64, 1, 28, 28, device=device)
y_batch = torch.randint(0, 10, (64,), device=device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    optimizer.zero_grad()
    output = model(x_batch)
    loss = loss_fn(output, y_batch)
    loss.backward()
    optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

```
-------------------------------  ------------  ------------  ------------
                           Name    Self CPU %     CPU total    CUDA total
-------------------------------  ------------  ------------  ------------
                aten::conv2d          2.1%      1.203ms       0.891ms
           aten::convolution          1.8%      1.150ms       0.885ms
    aten::cudnn_convolution           15.3%      0.982ms       0.845ms
                  aten::addmm          8.7%      0.421ms       0.312ms
                  aten::relu           3.2%      0.198ms       0.156ms
...
-------------------------------  ------------  ------------  ------------
```

`activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]` tells the profiler to track both CPU-side (Python/dispatch overhead) and GPU-side (actual kernel execution) time — comparing these two columns for the same operation is often the fastest way to identify whether you're CPU-bound or GPU-bound for a given part of the model, following the categories in Section 20.1. `record_shapes=True` additionally records the tensor shapes involved in each operation, useful for spotting unexpectedly large or oddly-shaped tensors driving up cost in an operation you didn't expect to be expensive.

---

## 20.3 Profiling a Full Training Loop with `schedule`

Profiling a single forward/backward pass (Section 20.2) is useful for isolated experiments, but profiling an entire training loop needs a different approach — you don't want to record every single step (the resulting trace would be enormous and slow to generate), and the first few steps are often unrepresentative (CUDA kernels compiling and caching for the first time, `DataLoader` workers spinning up). `torch.profiler.schedule` addresses both concerns:

```python
from torch.profiler import profile, schedule, ProfilerActivity

my_schedule = schedule(
    wait=5,     # skip the first 5 steps entirely (warm-up, not measured at all)
    warmup=2,    # profile these steps, but discard the results (further warm-up)
    active=5,     # actually record these steps
    repeat=1,      # repeat this wait/warmup/active cycle this many times
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=my_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profiler"),
) as prof:
    for step, (x_batch, y_batch) in enumerate(train_loader):
        if step >= (5 + 2 + 5):   # wait + warmup + active
            break

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

        prof.step()   # tells the profiler a training step has completed
```

`prof.step()`, called once per training iteration, is what drives the `wait`/`warmup`/`active`/`repeat` schedule forward — without it, the profiler has no way to know where one step ends and the next begins. `on_trace_ready=torch.profiler.tensorboard_trace_handler(...)` writes the recorded trace to disk in a format TensorBoard can visualize directly, which is generally a more practical way to explore a full training-loop trace than reading the printed table from Section 20.2, especially once GPU/CPU overlap and idle gaps become relevant (Section 20.4).

---

## 20.4 Reading a Trace: Identifying Idle Gaps

The printed table from Section 20.2 shows aggregate time per operation type, but it doesn't directly show *when* the GPU was idle waiting for something else — for that, the visual, timeline-based trace view (via TensorBoard's PyTorch Profiler plugin, or the Chrome trace viewer at `chrome://tracing`) is far more informative.

```python
prof.export_chrome_trace("trace.json")
```

Loading `trace.json` in `chrome://tracing` (or the equivalent TensorBoard view) shows a timeline with separate rows for CPU threads and each GPU stream, letting you visually inspect: is the GPU row continuously busy, or are there visible gaps? A gap in the GPU timeline that lines up with a `DataLoader`-related operation on the CPU timeline is a strong, direct visual confirmation of a data-loading bottleneck — exactly the kind of evidence that turns "I think it's the data loading" into a confirmed diagnosis rather than a guess.

### A practical bottleneck-isolation technique: the synthetic-data test

A simpler diagnostic, not requiring the full trace viewer, that's worth knowing as a fast first check: replace the real `DataLoader` with a loop that yields pre-generated, already-on-GPU tensors, and re-measure training speed.

```python
import time

# Pre-generate fake batches, already on the correct device -- removes DataLoader entirely
fake_batches = [
    (torch.rand(64, 1, 28, 28, device=device), torch.randint(0, 10, (64,), device=device))
    for _ in range(20)
]

start = time.time()
for x_batch, y_batch in fake_batches:
    optimizer.zero_grad()
    loss = loss_fn(model(x_batch), y_batch)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()   # ensure all GPU work has actually finished before stopping the timer
elapsed = time.time() - start
print(f"Time with synthetic data: {elapsed:.3f}s")
```

**If training speed with synthetic, pre-loaded data is meaningfully faster than with the real `DataLoader`, the bottleneck is in data loading** — confirming the diagnosis before spending time tuning `num_workers` or rewriting a `Dataset` (Chapter 7). If speed is roughly the same either way, the bottleneck lies elsewhere (model compute, CPU-side overhead), and data pipeline changes won't help.

**`torch.cuda.synchronize()` is essential here** and is worth understanding precisely: CUDA operations are launched *asynchronously* from the CPU's perspective — `optimizer.step()` returning doesn't mean the GPU has actually finished that work, only that the instruction has been queued. Without an explicit synchronization point, a naive CPU-side timer would measure only how long it took to *launch* the GPU work, not how long the GPU actually took to *execute* it — producing a misleadingly fast (and essentially meaningless) benchmark number.

---

## 20.5 Memory Profiling

`torch.profiler` also tracks memory allocation, useful for identifying which operations are driving peak GPU memory usage — directly relevant groundwork for the memory optimization techniques in Chapter 24:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    output = model(x_batch)
    loss = loss_fn(output, y_batch)
    loss.backward()

print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
```

A simpler, lower-overhead alternative for just tracking peak memory over a training run, without the full profiler's more detailed (and more expensive) instrumentation:

```python
torch.cuda.reset_peak_memory_stats()

# ... run one or more training steps ...

peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
```

This lightweight check is worth running routinely (not just when actively debugging an OOM) as a sanity check whenever you change batch size, model architecture, or sequence length — on hardware like your RTX 3090's 24GB, knowing exactly how much headroom a given configuration leaves is directly useful for deciding how much you can safely scale up before hitting a wall.

---

## 20.6 A Profiling Checklist

A practical sequence to follow whenever a training run feels slower than expected, rather than guessing:

1. **Confirm the GPU is actually the bottleneck** using the synthetic-data test (Section 20.4) before investing time in GPU-side optimizations.
2. **If GPU-bound**, use the `key_averages()` table (Section 20.2) to identify which specific operations dominate CUDA time — this points directly at candidates for Chapter 15's mixed precision or Chapter 21's `torch.compile`.
3. **If data-loading-bound**, revisit Chapter 7's `num_workers`, `pin_memory`, and whether `__getitem__` is doing unnecessarily expensive work that could be precomputed once rather than repeated every epoch.
4. **If CPU-bound with the GPU otherwise idle**, look for excessive small operations, unnecessary `.item()`/`.cpu()` synchronization points scattered through the training loop (each one forces the CPU to wait for the GPU, discussed further in Chapter 21), or Python-level overhead in a custom `forward()`.
5. **Check peak memory** (Section 20.5) alongside speed — a configuration that's fast but sits right at the edge of available GPU memory is fragile and worth knowing about before it causes an OOM in the middle of a longer run.

---

## Summary

- Training slowness has distinct root causes (GPU-bound, CPU-bound, data-loading-bound) requiring different fixes — profile before optimizing, rather than guessing.
- `torch.profiler.profile`, combined with `ProfilerActivity.CPU` and `ProfilerActivity.CUDA`, records per-operation CPU and GPU time; `key_averages().table()` summarizes this into a readable ranked view.
- `schedule` + `prof.step()` profiles a representative slice of a full training loop, skipping unrepresentative warm-up steps; `export_chrome_trace()` or TensorBoard's profiler plugin gives a visual timeline for spotting idle GPU gaps.
- A synthetic pre-loaded-data test is a fast, low-effort way to confirm or rule out a data-loading bottleneck before investing in `DataLoader` tuning.
- `torch.cuda.synchronize()` is required for accurate CPU-side timing of GPU work, since CUDA operations are launched asynchronously.
- `profile_memory=True` and `torch.cuda.max_memory_allocated()` identify memory bottlenecks, worth checking routinely alongside speed.

## Exercises

1. Profile a single training step of a model from an earlier chapter (e.g., `FashionMNIST_CNN` from Chapter 10) using the minimal example from Section 20.2, and identify the top 3 most expensive operations by CUDA time.
2. Run the synthetic-data test from Section 20.4 on a real training loop with `num_workers=0`, then repeat with `num_workers=4`. Use the result to determine whether that particular pipeline was data-loading-bound, and by how much.
3. Deliberately introduce a `.item()` call inside a tight loop that doesn't need one (e.g., calling `.item()` on every individual output value rather than only the final aggregated loss) and use profiling with `torch.cuda.synchronize()` timing to measure the resulting slowdown.
4. Use `export_chrome_trace()` on a full training loop profiled with `schedule`, load it in `chrome://tracing`, and identify at least one visible gap in the GPU timeline — describe what CPU-side operation appears to be causing it.

**Next:** Chapter 21 covers `torch.compile` and TorchDynamo — using graph capture and compilation to reduce the CPU-side overhead and kernel launch costs that profiling in this chapter is designed to reveal.
