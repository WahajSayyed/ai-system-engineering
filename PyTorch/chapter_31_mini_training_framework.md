# Chapter 31: Writing a Mini Training Framework

## Introduction

This chapter is the capstone of the entire book: consolidating the training loop (Chapter 8), mixed precision (Chapter 15), learning rate scheduling (Chapter 18), distributed training (Chapters 22-23), and checkpointing into a single, reusable `Trainer` class — in the spirit of Hugging Face's `Trainer`/`SFTTrainer`, which you've already used directly in your own Qwen2.5-Coder fine-tuning work. Building a simplified version yourself is the best way to fully understand what those production frameworks are actually doing under the hood, rather than treating them as opaque configuration objects.

By the end of this chapter you'll have a complete, working `Trainer` class that:
- Runs a full training loop with train/eval phases, correctly handling every detail from Chapter 8
- Supports mixed precision (Chapter 15) and gradient clipping (Chapter 8) transparently
- Integrates a learning rate scheduler (Chapter 18) with correct per-step/per-epoch handling
- Optionally supports DDP (Chapters 22-23) without the calling code needing to change
- Checkpoints and resumes correctly, including all the state pieces from Chapter 8/15/23

---

## 31.1 Designing the Interface First

Before writing any implementation, it's worth deciding what the *calling code* should look like — good framework design starts from the user-facing API, then works backward to the implementation. The goal: a user should be able to configure and run a full training job in a handful of lines, with all of this book's correctness details (mode toggling, gradient accumulation reset, `no_grad` during eval, and so on) handled automatically and correctly, exactly once, inside the framework rather than re-implemented (and potentially re-broken) in every individual training script.

```python
# The target usage, before writing a single line of Trainer implementation:
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
    use_amp=True,
    grad_clip_norm=1.0,
    checkpoint_dir="./checkpoints",
)

trainer.train(num_epochs=10)
```

Everything in this configuration maps to a specific chapter's concept: `use_amp` to Chapter 15, `grad_clip_norm` to Chapter 8 Section 8.5, `scheduler` to Chapter 18, `checkpoint_dir` to Chapter 8 Section 8.6. The `Trainer`'s entire job is orchestrating these pieces correctly and consistently, so this configuration-level interface is all a user needs to interact with.

---

## 31.2 The Core `Trainer` Class

Building incrementally — starting with the essential structure, then layering in each feature from the chapters above.

```python
import torch
import torch.nn as nn
import os

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        scheduler_step_per_batch=False,   # Chapter 18, Section 18.2 -- per-step vs per-epoch
        use_amp=False,
        grad_clip_norm=None,
        checkpoint_dir=None,
        device=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = checkpoint_dir

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Chapter 15 -- AMP setup, correctly disabled when not on CUDA
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.current_epoch = 0
        self.best_val_loss = float("inf")

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
```

This constructor is deliberately just configuration and setup — no training logic yet — following the same principle as Chapter 5's `nn.Module.__init__` versus `forward()`: `__init__` establishes state, the methods below (`train_epoch`, `evaluate`, `train`) define behavior.

---

## 31.3 The Training Epoch: Every Chapter 8 Detail, Once

```python
    def train_epoch(self):
        self.model.train()   # Chapter 8, Section 8.2
        running_loss = 0.0
        num_samples = 0

        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()   # Chapter 3/8 -- reset before every step, always

            # Chapter 15 -- autocast is a no-op when use_amp=False, via the `enabled=` flag
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)   # Chapter 15, Section 15.5 -- unscale before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None and self.scheduler_step_per_batch:
                self.scheduler.step()   # Chapter 18, Section 18.2 -- per-step schedulers (OneCycleLR, etc.)

            running_loss += loss.item() * x_batch.size(0)   # Chapter 8, Section 8.4 -- .item(), never the tensor
            num_samples += x_batch.size(0)

        if self.scheduler is not None and not self.scheduler_step_per_batch:
            self.scheduler.step()   # Chapter 18, Section 18.2 -- per-epoch schedulers (StepLR, CosineAnnealingLR)

        return running_loss / num_samples
```

Every line here traces back to a specific, previously-discussed correctness concern — this method is, in a real sense, the distilled sum of Chapters 3, 8, 15, and 18's individual lessons, expressed once, correctly, rather than needing to be remembered and re-applied by hand in every new training script. The `scheduler_step_per_batch` flag directly encodes Chapter 18, Section 18.2's critical distinction, making it an explicit, impossible-to-forget configuration choice rather than an easy-to-miss detail buried in a hand-written loop.

---

## 31.4 Evaluation: `no_grad`, `eval()`, and Aggregated Metrics

```python
    @torch.no_grad()   # Chapter 3, Section 3.5 -- decorator form, applies to the whole method
    def evaluate(self):
        if self.val_loader is None:
            return None

        self.model.eval()   # Chapter 8, Section 8.2
        running_loss = 0.0
        num_samples = 0

        for x_batch, y_batch in self.val_loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)

            running_loss += loss.item() * x_batch.size(0)
            num_samples += x_batch.size(0)

        return running_loss / num_samples
```

`@torch.no_grad()` decorating the entire method (rather than a `with` block inside it) is a clean pattern worth calling back to from Chapter 9, Section 9.5 — appropriate here since *every* line of `evaluate()` should run without gradient tracking, so wrapping the whole method is simpler and equally correct compared to indenting a `with` block around the loop body.

---

## 31.5 Checkpointing: Every Piece of State, Correctly

```python
    def save_checkpoint(self, path):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),   # Chapter 15, Section 15.6
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.current_epoch = checkpoint["epoch"] + 1

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
```

This method is precisely the union of every checkpointing detail scattered across three earlier chapters — model and optimizer state from Chapter 8, Section 8.6; scaler state from Chapter 15, Section 15.6; scheduler state, needed for correctly resuming a schedule mid-cycle rather than restarting it from step 0 (Chapter 18). Consolidating all of it into one `save_checkpoint`/`load_checkpoint` pair is exactly the kind of "correct once, reused everywhere" benefit a framework is meant to provide — a hand-written training script that forgets even one of these four state components will *appear* to resume correctly, then behave subtly wrong (a scaler restarting its scale-factor search, or a scheduler restarting its decay from the beginning) in a way that's easy to miss without specifically testing for it.

---

## 31.6 The Top-Level `train()` Loop

```python
    def train(self, num_epochs, start_epoch=None):
        if start_epoch is not None:
            self.current_epoch = start_epoch

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            log_msg = f"Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" | val_loss={val_loss:.4f}"
            print(log_msg)

            if self.checkpoint_dir is not None:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, "last.pt"))

                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "best.pt"))
```

Note the loop starts from `self.current_epoch`, not always `0` — this is what makes resuming training (via `load_checkpoint()` followed by `train()`) pick up exactly where a previous run left off, rather than restarting the epoch count, directly closing the loop on the checkpointing mechanics from Section 31.5.

---

## 31.7 Adding DDP Support Without Changing the Calling Code

A well-designed framework should let a user opt into distributed training (Chapters 22-23) largely through configuration, not by rewriting their training script. A minimal extension:

```python
    def __init__(self, ..., use_ddp=False, rank=0):   # extending the constructor from Section 31.2
        # ... existing setup ...
        self.use_ddp = use_ddp
        self.rank = rank
        self.is_main_process = (not use_ddp) or (rank == 0)   # Chapter 23, Section 23.4's rank==0 pattern

        if use_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[rank])

    def save_checkpoint(self, path):
        if not self.is_main_process:
            return   # Chapter 23, Section 23.4 -- only rank 0 writes checkpoints

        model_to_save = self.model.module if self.use_ddp else self.model   # Chapter 23's .module access
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": model_to_save.state_dict(),
            # ... rest unchanged from Section 31.5 ...
        }
        torch.save(checkpoint, path)
```

`self.is_main_process`, computed once and reused throughout (guarding checkpointing here, and equally applicable to guarding the `print()` logging in Section 31.6), is exactly Chapter 23, Section 23.4's `rank == 0` pattern, elevated from an ad-hoc check scattered through a hand-written script into a single, reusable `Trainer` attribute — the calling code in Section 31.1 doesn't need to know or care whether it's running in a distributed setting at all; it just calls `trainer.train(...)`, and the `Trainer` internally handles rank-aware behavior correctly and consistently.

---

## 31.8 Using the Complete `Trainer`

Bringing it back to the target interface from Section 31.1 — now fully implemented, and directly reusable across every model in this book:

```python
model = FashionMNIST_CNN()   # Chapter 10
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    scheduler_step_per_batch=False,   # CosineAnnealingLR is per-epoch, Chapter 18, Section 18.2
    use_amp=True,
    grad_clip_norm=1.0,
    checkpoint_dir="./checkpoints",
)

trainer.train(num_epochs=10)

# Resuming later, in a fresh script:
trainer.load_checkpoint("./checkpoints/last.pt")
trainer.train(num_epochs=20)   # continues from wherever it left off
```

This is the entire payoff of the chapter: the same handful of lines now correctly runs mixed precision, gradient clipping, a correctly-scheduled learning rate, full checkpoint/resume support, and (with `use_ddp=True`) distributed training — with every individual correctness detail from Chapters 8, 15, 18, and 23 handled exactly once, inside the `Trainer`, rather than needing to be remembered and re-implemented by hand for every new project.

---

## Summary

- A good framework interface is designed from the calling code backward — deciding what a user should type before writing the implementation that makes it work.
- The `Trainer`'s `train_epoch()` and `evaluate()` methods consolidate every correctness detail from Chapter 8 (mode toggling, `zero_grad`, `.item()`), Chapter 15 (autocast, `GradScaler`, unscaling before clipping), and Chapter 18 (per-step vs. per-epoch scheduler stepping) into one place, applied consistently across every future training run.
- Checkpointing must save and restore model, optimizer, scaler, and scheduler state together — omitting any one causes a resumed run to behave subtly, silently wrong rather than failing outright.
- DDP support can be added via configuration (`use_ddp`, `rank`) without changing the calling code at all, by internally applying Chapter 23's `rank == 0` and `.module` patterns inside the framework rather than requiring the user to handle them.
- This `Trainer` is a simplified but structurally accurate model of what production frameworks like Hugging Face's `Trainer`/`SFTTrainer` are doing internally — understanding it demystifies the configuration options those frameworks expose.

## Exercises

1. Implement the complete `Trainer` class from this chapter, train the `FashionMNIST_CNN` from Chapter 10 with it for a few epochs, and confirm the results match a hand-written training loop (Chapter 9) on the same data and hyperparameters.
2. Add early stopping to `Trainer`: stop training if validation loss hasn't improved for a configurable number of epochs (`patience`), directly analogous to `ReduceLROnPlateau`'s patience concept from Chapter 18, Section 18.7.
3. Extend `Trainer` to log metrics to a simple CSV file (one row per epoch) in addition to printing them, guarded by `self.is_main_process` exactly as checkpointing is in Section 31.7.
4. Test the resume functionality end to end: train for 5 epochs, stop, reload the checkpoint in a fresh script, and continue training for 5 more epochs — confirm the loss curve is smooth across the resume point, with no unexpected spike (connecting back to Chapter 15, Section 15.6's discussion of what happens when scaler state is *not* correctly restored).

**Next:** Chapter 32 closes the book with debugging and testing practices — reproducibility, `torch.testing`, and a consolidated reference for diagnosing the failure modes flagged throughout every chapter, including several that a `Trainer` like this one can still hit if misconfigured.
