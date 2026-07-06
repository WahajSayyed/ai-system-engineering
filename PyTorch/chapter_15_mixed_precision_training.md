# Chapter 15: Mixed Precision Training

## Introduction

Every model trained so far in this book has used `float32` for weights and activations. Modern GPUs — including your RTX 3090's Ampere-generation Tensor Cores — can perform matrix multiplications substantially faster in lower precision (`float16` or `bfloat16`), while using roughly half the memory. **Automatic mixed precision (AMP)** lets PyTorch use lower precision where it's numerically safe to do so, while keeping full `float32` precision for operations that need it, all with minimal code changes.

This chapter opens Part IV, which shifts focus from architecture design to training mechanics at scale. Mixed precision is a natural starting point: it's one of the highest-leverage, lowest-effort performance improvements available, often yielding a meaningful speedup and memory reduction from adding just a few lines to an existing training loop.

By the end of this chapter you'll be able to:
- Explain what mixed precision training actually does and why it works
- Use `torch.autocast` and `torch.amp.GradScaler` correctly in a training loop
- Understand the difference between `float16` and `bfloat16`, and when to use each
- Recognize and debug the most common AMP-related issues

---

## 15.1 Why Mixed Precision Works

`float32` uses 32 bits per number: 1 sign bit, 8 exponent bits, 23 mantissa (precision) bits. `float16` uses 16 bits: 1 sign, 5 exponent, 10 mantissa. Fewer bits means less memory per value and faster computation on hardware with dedicated support (Tensor Cores) — but also a much smaller representable range and less precision.

The key insight behind mixed precision: **not every part of a training step needs full precision equally**. Matrix multiplications and convolutions — the bulk of the compute in a forward pass — tolerate reduced precision well. Certain operations (loss computation, and gradient accumulation for parameters with small-magnitude gradients) are much more sensitive to reduced precision and benefit from staying in `float32`. AMP automates this decision per-operation, using a curated list of which operations are safe to run in reduced precision and which should stay in `float32`, rather than requiring you to specify this manually op by op.

> **A note on the API.** Older PyTorch code (and many tutorials, papers' reference implementations, and even some of your own earlier scripts) uses `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler`. As of PyTorch 2.4, these are deprecated in favor of the device-agnostic `torch.amp.autocast("cuda", ...)` and `torch.amp.GradScaler("cuda")` — same behavior, explicit device argument. This chapter uses the current API; if you encounter the older form in existing code, it still works but will emit a `FutureWarning`, and it's worth updating.

---

## 15.2 `torch.autocast`: Automatic Per-Operation Precision

`torch.autocast` is a context manager that wraps the forward pass (model + loss computation), automatically running eligible operations in reduced precision while leaving precision-sensitive operations in `float32`:

```python
import torch
import torch.nn as nn

device = torch.device("cuda")
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

x_batch = torch.rand(64, 784, device=device)
y_batch = torch.randint(0, 10, (64,), device=device)

optimizer.zero_grad()

with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(x_batch)
    loss = loss_fn(output, y_batch)
    print(output.dtype)   # torch.float16 -- Linear layers ran in reduced precision
    print(loss.dtype)      # torch.float32 -- loss computation stays in float32 for stability

loss.backward()   # deliberately OUTSIDE the autocast block -- see below
optimizer.step()
```

**`loss.backward()` belongs outside the `autocast` block.** Backward ops automatically run in whatever dtype their corresponding forward op used — autocast handles this correctly without needing to be active during the backward pass itself, and PyTorch's documentation specifically advises against wrapping `.backward()` in `autocast`.

Notice that `output.dtype` is `float16` (the `Linear` layers ran in reduced precision) while `loss.dtype` is `float32` — autocast automatically upcast the loss computation, since operations like cross-entropy are on the list of precision-sensitive operations that autocast always runs in `float32` regardless of the surrounding context's requested dtype.

---

## 15.3 `GradScaler`: Preventing Gradient Underflow

`float16`'s limited range creates a specific problem: gradients are often small numbers, and in `float16`, sufficiently small values can underflow to exactly zero, silently vanishing rather than contributing to the parameter update. **Gradient scaling** addresses this by multiplying the loss by a large scale factor before `.backward()`, which proportionally scales up all the gradients, keeping small values safely away from `float16`'s underflow threshold. The scaling is undone before the optimizer actually applies the update, so it has no effect on the mathematical result — it exists purely to preserve numerical precision at the intermediate gradient-computation step.

```python
scaler = torch.amp.GradScaler("cuda")

for x_batch, y_batch in train_loader:
    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(x_batch)
        loss = loss_fn(output, y_batch)

    scaler.scale(loss).backward()   # scales the loss before backward, producing scaled gradients
    scaler.step(optimizer)            # unscales gradients, checks for inf/NaN, then steps if safe
    scaler.update()                    # adjusts the scale factor for the next iteration
```

Four `GradScaler` methods, each with a specific job:

- **`scaler.scale(loss)`**: multiplies the loss by the current scale factor before `.backward()` is called on it.
- **`scaler.step(optimizer)`**: first unscales the gradients back to their true values, checks whether any are `inf` or `NaN` (which would indicate the scale factor was too aggressive for this step), and only calls the underlying `optimizer.step()` if the gradients are clean — skipping the update entirely on the rare step where overflow is detected, rather than corrupting the model with a bad update.
- **`scaler.update()`**: adjusts the scale factor for future iterations — growing it periodically (to use as much of `float16`'s range as safely possible) and shrinking it whenever an overflow was detected in `step()`.
- **A single `GradScaler` instance should persist across the entire training run** — it's lightweight, but its internal scale factor needs to accumulate its adjustment history across iterations to converge on an appropriate value; creating a fresh one every iteration would defeat this entirely.

This exact four-line pattern — `scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()`, in place of the plain `loss.backward()` / `optimizer.step()` from Chapter 8 — is the complete, standard mixed precision training step. Note `optimizer.zero_grad()` still appears exactly where it did before (Chapter 3's gradient accumulation logic is entirely unaffected by AMP).

---

## 15.4 `bfloat16`: A Simpler Alternative on Modern GPUs

`bfloat16` (brain float16) is an alternative 16-bit format with a different bit allocation: 1 sign bit, **8** exponent bits (same as `float32`), and only 7 mantissa bits. This trade-off matters directly: `bfloat16` has the *same dynamic range* as `float32` (since it keeps all 8 exponent bits), at the cost of less precision within that range — the opposite trade-off from `float16`, which has more precision but a much narrower range.

The practical consequence: **`bfloat16` doesn't suffer from the underflow problem `GradScaler` was built to solve**, because its exponent range matches `float32`'s. This means `bfloat16` training typically doesn't need gradient scaling at all:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(x_batch)
    loss = loss_fn(output, y_batch)

loss.backward()      # no GradScaler needed with bfloat16
optimizer.step()
```

Your RTX 3090 (Ampere architecture) has hardware support for `bfloat16` on its Tensor Cores, making this a genuinely practical choice, not just a theoretical one. **Practical guidance:** prefer `bfloat16` over `float16` when your GPU supports it (Ampere generation or newer — Volta and Turing do not) — it's simpler code (no `GradScaler` machinery needed) and more numerically robust, at a small cost in precision that's rarely significant for typical deep learning workloads. This is also why `bfloat16` has become the standard choice for large-scale LLM training and fine-tuning, including the kind of Unsloth/SFTTrainer workflows you've used for Qwen2.5-Coder — those libraries typically default to or strongly recommend `bfloat16` specifically for this reason.

```python
# Check bfloat16 support directly
print(torch.cuda.is_bf16_supported())   # True on Ampere (RTX 30xx) and newer
```

---

## 15.5 A Complete Mixed Precision Training Loop

Integrating AMP into the full Chapter 8 training loop, using `float16` with `GradScaler` (the more universally compatible choice across older GPU generations) — swap to the `bfloat16` pattern from Section 15.4 directly if your hardware supports it:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNIST_CNN().to(device)   # from Chapter 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            output = model(x_batch)
            loss = loss_fn(output, y_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                                            # unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)       # Chapter 8, Section 8.5
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x_batch.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")
```

Two details worth flagging, both easy to get subtly wrong:

- **`enabled=(device.type == "cuda")`** on both `autocast` and `GradScaler`: this is the "convenience argument" mentioned in PyTorch's own AMP documentation — passing `enabled=False` turns both into complete no-ops, letting the exact same code path run correctly on CPU-only machines (where AMP with `float16` isn't beneficial and isn't fully supported the same way) without needing a separate `if`/`else` branch.
- **`scaler.unscale_(optimizer)` before `clip_grad_norm_`**: gradient clipping needs to operate on the *true, unscaled* gradient magnitudes to compute a meaningful norm — clipping the scaled gradients directly would clip based on an arbitrary, scale-factor-dependent magnitude that has nothing to do with the actual gradient norm you intended to bound. Calling `scaler.unscale_()` explicitly before clipping (rather than letting `scaler.step()` do the unscaling implicitly) is required whenever you need to inspect or modify gradients — such as clipping — between `backward()` and `step()`.

---

## 15.6 Checkpointing with Mixed Precision

`GradScaler` has its own state (the current scale factor and its adjustment history) that should be saved and restored alongside model and optimizer state, extending the checkpointing pattern from Chapter 8, Section 8.6:

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict(),   # AMP-specific addition
}
torch.save(checkpoint, "checkpoint.pt")

# Resuming:
checkpoint = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scaler.load_state_dict(checkpoint["scaler_state_dict"])
```

Skipping this doesn't break correctness outright, but the scaler will restart its scale-factor search from scratch after resuming, which can cause a few unstable iterations right after resuming — similar in spirit to the optimizer-state-loading concern from Chapter 8.

---

## Summary

- Mixed precision runs eligible operations (mainly matrix multiplications and convolutions) in reduced precision while keeping precision-sensitive operations (like loss computation) in `float32`, automated per-operation by `torch.autocast`.
- `float16` has a narrow dynamic range, making gradients prone to underflow — `GradScaler` addresses this by scaling the loss before `.backward()` and unscaling before the optimizer step, skipping updates on rare overflow.
- `bfloat16` has `float32`'s exponent range and doesn't need `GradScaler` — prefer it on Ampere-or-newer GPUs (including the RTX 3090) for simpler, more numerically robust code; it's also the standard choice for modern LLM training and fine-tuning.
- `.backward()` should be called outside the `autocast` context; `scaler.unscale_()` must be called explicitly before any gradient manipulation (like clipping) that happens between `backward()` and `step()`.
- Save and restore `scaler.state_dict()` alongside model and optimizer state when checkpointing.

## Exercises

1. Train the `FashionMNIST_CNN` from Chapter 10 with and without mixed precision (`float16` + `GradScaler`), measuring wall-clock time per epoch and peak GPU memory usage (`torch.cuda.max_memory_allocated()`) for both. Report the speedup and memory reduction.
2. Repeat Exercise 1 using `bfloat16` instead (no `GradScaler`), and compare training stability (loss curve smoothness) against the `float16` version.
3. Deliberately set an extremely large initial scale factor (`torch.amp.GradScaler("cuda", init_scale=2**24)`) and observe what happens to training in the first few iterations — connect what you see to `scaler.step()`'s overflow-skipping behavior described in Section 15.3.
4. Extend the checkpointing example in Section 15.6 into a full save/resume test: train for a few epochs, save a checkpoint including scaler state, restart the script, load everything back, and confirm training loss continues smoothly rather than spiking.

**Next:** Chapter 16 covers custom autograd functions — `torch.autograd.Function` — for implementing operations with hand-written forward and backward passes, useful when you need behavior autograd can't derive automatically.
