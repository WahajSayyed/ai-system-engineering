# Chapter 8: The Training Loop, Deconstructed

## Introduction

You now have every individual piece: tensors and autograd (Part I), `nn.Module` (Chapter 5), loss functions and optimizers (Chapter 6), and data loading (Chapter 7). This chapter assembles them into a complete, production-shaped training loop — and covers the details that separate a toy loop from one that's robust enough to survive a multi-hour (or multi-day) real training run: gradient clipping, checkpointing, validation, and `train()`/`eval()` mode switching.

By the end of this chapter you'll be able to:
- Write a complete training loop with proper train/eval mode handling
- Track and log training and validation metrics correctly
- Apply gradient clipping to stabilize training
- Save and resume training from checkpoints
- Recognize the difference between an epoch, a step, and a batch

---

## 8.1 The Full Anatomy of a Training Loop

Here is a complete, realistic training loop. Every piece has been introduced individually in earlier chapters — this section is about how they fit together, plus the details a toy example usually skips.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(784, 128, 10).to(device)          # Chapter 5
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)   # Chapter 6
loss_fn = nn.CrossEntropyLoss()                 # Chapter 6

num_epochs = 10

for epoch in range(num_epochs):
    # ---- Training phase ----
    model.train()                                # Section 8.2
    running_loss = 0.0

    for x_batch, y_batch in train_loader:          # Chapter 7
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)   # Section 8.4

    train_loss = running_loss / len(train_loader.dataset)

    # ---- Validation phase ----
    model.eval()                                   # Section 8.2
    val_loss = 0.0
    correct = 0

    with torch.no_grad():                            # Chapter 3
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)

            val_loss += loss.item() * x_batch.size(0)
            correct += (predictions.argmax(dim=1) == y_batch).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
          f"val_acc={val_accuracy:.4f}")
```

The rest of this chapter explains each non-obvious piece of this loop in detail.

---

## 8.2 `model.train()` and `model.eval()`: Why They're Not Optional

`nn.Module` has two modes, and switching between them is not cosmetic — it changes the actual computation performed by certain layers.

- **`Dropout`** (Chapter 19) randomly zeroes activations during training as a regularizer, but must be fully disabled during evaluation — you want deterministic, complete predictions at inference time, not randomly dropped units.
- **`BatchNorm`** (Chapter 19) uses per-batch statistics during training but switches to fixed, accumulated running statistics during evaluation, so that inference behavior doesn't depend on which other samples happen to be in the current batch.

```python
model.train()   # Dropout active, BatchNorm uses batch statistics
model.eval()     # Dropout disabled, BatchNorm uses running statistics
```

If you forget to call `model.eval()` before validation, dropout stays active and BatchNorm keeps using batch statistics — your validation metrics will be noisy and pessimistically biased, and won't reflect your model's true inference-time behavior. If you forget to call `model.train()` again before resuming training after an evaluation pass, you'll silently train with dropout disabled, which quietly removes a regularizer you were probably counting on. **Toggle mode explicitly at the start of every training phase and every evaluation phase — never assume the mode carries over correctly.**

A model with no dropout or batchnorm layers technically doesn't need this — but making it a reflexive habit costs nothing and prevents an entire category of hard-to-notice bugs the moment you add such a layer later.

---

## 8.3 `torch.no_grad()` During Validation — and Why It Matters Beyond Correctness

You saw `torch.no_grad()` in Chapter 3 for controlling the autograd graph. In a validation loop, it does double duty:

1. **Correctness of intent**: you're not going to call `.backward()` during validation, so there's no reason to build a graph at all.
2. **Real performance and memory impact**: without it, PyTorch would still allocate memory for and track every intermediate activation across the entire validation set, exactly as if you were about to backpropagate through it. For a validation set of any real size, this can meaningfully increase memory usage and slow down the pass for no benefit.

```python
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        predictions = model(x_batch)
        # ... compute metrics
```

Forgetting `torch.no_grad()` during validation is one of the most common causes of an "it worked fine on small data, then OOM'd on the full validation set" bug — worth checking first if you hit unexpected memory pressure specifically during an eval pass.

---

## 8.4 `.item()`: Extracting a Python Number from a Loss Tensor

`loss` is a scalar tensor with `requires_grad=True` — accumulating it directly (`running_loss += loss`) would keep the *entire computational graph* for every batch alive simultaneously, since `running_loss` would itself become part of the graph, causing a severe and steadily worsening memory leak over the course of an epoch.

```python
running_loss += loss.item() * x_batch.size(0)   # correct: .item() extracts a plain Python float
# running_loss += loss                            # WRONG: leaks the graph, grows memory every batch
```

`.item()` extracts the underlying Python scalar from a single-element tensor, fully detached from autograd — exactly the operation you want for logging and metric accumulation. This pairs with the `.detach()` discussion from Chapter 4: `.item()` is effectively `.detach()` plus "and also give me a plain Python number, not a tensor at all."

Multiplying by `x_batch.size(0)` (the batch size) before accumulating, then dividing by the total dataset size at the end, correctly computes a per-sample average even when the final batch of an epoch is smaller than the rest (the classic case where `drop_last=False`, from Chapter 7).

---

## 8.5 Gradient Clipping

Certain architectures and training regimes — RNNs (Chapter 11) and LLM fine-tuning being the most common cases you'll encounter — are prone to **exploding gradients**: gradient magnitudes that occasionally spike to very large values, causing a single bad update to destabilize or even corrupt the model's weights.

**Gradient clipping** caps the norm of the gradients before the optimizer step, preventing any single update from being too large:

```python
for x_batch, y_batch in train_loader:
    optimizer.zero_grad()
    predictions = model(x_batch)
    loss = loss_fn(predictions, y_batch)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # clip BEFORE step()

    optimizer.step()
```

`clip_grad_norm_` computes the combined norm across *all* parameters' gradients together (not per-parameter) and scales them down proportionally if that combined norm exceeds `max_norm`, preserving the gradient's direction while limiting its magnitude. The order matters: clipping must happen **after `.backward()`** (gradients need to exist first) and **before `optimizer.step()`** (so the clipped values are what actually get applied).

`max_norm=1.0` is a common starting point, but the right value is architecture- and task-dependent — this is exactly the kind of hyperparameter you'd tune alongside learning rate. For the kind of GRPO/RLVR training you've worked with, gradient clipping is standard practice and often set more conservatively (smaller `max_norm`) than typical supervised fine-tuning, since RL-style objectives tend to produce noisier gradients.

---

## 8.6 Checkpointing: Saving and Resuming Training

A checkpoint should capture everything needed to resume training exactly where it left off — not just the model weights, but optimizer state (e.g., Adam's per-parameter momentum estimates) and the epoch number.

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_loss": train_loss,
    "val_loss": val_loss,
}
torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
```

`model.state_dict()` returns an ordered dictionary mapping each parameter/buffer name to its tensor — this is the standard, portable way to save weights (as opposed to `torch.save(model, path)`, which pickles the entire model object and class definition, and is more fragile across code changes — prefer `state_dict()` in essentially all cases).

### Resuming from a checkpoint

```python
checkpoint = torch.load("checkpoint_epoch_9.pt", map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1

for epoch in range(start_epoch, num_epochs):
    # ... continue training
```

Loading the **optimizer's** state dict, not just the model's, matters more than people expect — Adam/AdamW maintain running estimates of gradient moments per parameter, and restarting with those reset to zero effectively restarts the optimizer's "warm-up" behavior, which can cause a visible bump in loss right after resuming even though the model weights themselves are unchanged.

`map_location=device` ensures a checkpoint saved on GPU can still be loaded correctly on a machine without a GPU (or a different GPU index) — a real concern in a multi-node setup, or when a training run needs to move between machines with different GPU configurations.

### Saving only the best model

A common pattern: track the best validation loss seen so far, and only checkpoint when it improves, to avoid accumulating dozens of large checkpoint files unnecessarily:

```python
best_val_loss = float("inf")

# inside the epoch loop, after computing val_loss:
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best_model.pt")
```

---

## 8.7 Epoch vs. Step vs. Batch: Terminology Worth Being Precise About

These terms get used loosely, but precision matters once you're reading papers, configuring learning rate schedulers (Chapter 18), or setting up logging:

- **Batch**: one group of samples processed together in a single forward/backward pass (`batch_size` samples).
- **Step** (or **iteration**): one call to `optimizer.step()` — i.e., one weight update. Under standard training, one step corresponds to exactly one batch, but this is *not* always true — with gradient accumulation (calling `.backward()` several times before a single `.step()`, mentioned in Chapter 3), one step spans multiple batches.
- **Epoch**: one complete pass through the entire training dataset.

```python
steps_per_epoch = len(train_loader)          # number of batches in one epoch
total_steps = steps_per_epoch * num_epochs
```

This distinction matters concretely once you get to learning rate schedulers in Chapter 18 — some schedulers step per-batch (`scheduler.step()` inside the batch loop), others step per-epoch, and mixing these up is a common source of a learning rate schedule that doesn't behave as intended.

---

## Summary

- A complete training loop = `model.train()`, iterate batches, `zero_grad()` → forward → `backward()` → (optional clip) → `step()`, then `model.eval()` + `torch.no_grad()` for validation.
- `model.train()`/`model.eval()` aren't cosmetic — they change the actual computation for layers like Dropout and BatchNorm. Toggle them explicitly at every phase transition.
- Use `torch.no_grad()` during validation for both correctness and real memory/performance savings.
- Accumulate loss with `.item()`, never the raw tensor, to avoid leaking the computational graph across an entire epoch.
- Gradient clipping (`clip_grad_norm_`, applied after `backward()` and before `step()`) stabilizes training against exploding gradients, especially relevant for RNNs and RL-style fine-tuning.
- Checkpoints should include model state, optimizer state, and epoch number — not just weights — to allow training to resume without a jarring restart of the optimizer's internal state.
- Be precise about batch vs. step vs. epoch — this precision matters once schedulers and gradient accumulation enter the picture.

## Exercises

1. Take the training loop from Section 8.1 and deliberately remove `model.eval()` before the validation phase. Run it on a model that includes a `Dropout` layer and observe how validation accuracy changes compared to the correct version.
2. Add gradient clipping to a training loop, and experiment with `max_norm` values of `0.1`, `1.0`, and `10.0` — record how training loss curves differ.
3. Implement checkpoint saving that keeps only the 3 most recent checkpoints (deleting older ones) plus a separate "best model so far" checkpoint that's never deleted.
4. Write a training loop that uses gradient accumulation — calling `.backward()` 4 times before each `optimizer.step()` — and explain, using the epoch/step/batch definitions from 8.7, how many actual optimizer steps occur in one epoch over a dataset with 1000 samples and `batch_size=32`.

**Next:** Chapter 9 puts everything from Part II together into a full, end-to-end MLP classification example — completing Part II before Part III introduces convolutional, recurrent, and attention-based architectures.
