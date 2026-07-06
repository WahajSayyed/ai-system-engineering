# Chapter 18: Learning Rate Scheduling

## Introduction

Every training loop so far has used a fixed learning rate for the entire run. In practice, the optimal learning rate typically isn't constant over the course of training: a larger learning rate early on helps the model make rapid progress and can help it avoid getting stuck in poor local regions of the loss landscape, while a smaller learning rate later allows finer convergence toward a good minimum without overshooting it. **Learning rate schedulers** automate changing the learning rate over time according to a defined policy, and correctly integrating one is one of the more impactful, low-cost improvements you can make to a training run.

By the end of this chapter you'll be able to:
- Use `torch.optim.lr_scheduler` correctly, including the critical distinction between per-epoch and per-step schedulers
- Explain and apply warmup, and why it matters especially for transformers
- Use cosine annealing and `OneCycleLR`, two of the most widely used modern scheduling policies
- Combine schedulers (e.g., warmup followed by decay) using `SequentialLR`
- Recognize the most common scheduler integration mistakes

---

## 18.1 The Basic Scheduler Pattern

A scheduler wraps an optimizer and adjusts its learning rate according to a rule, called once per some unit of training progress (Chapter 8, Section 8.7's distinction between epoch and step matters a great deal here — see Section 18.2):

```python
import torch

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(30):
    # ... training for one epoch, using optimizer.step() as usual (Chapter 8) ...
    scheduler.step()   # called once per epoch, AFTER the epoch's training is done
    print(f"Epoch {epoch}: lr={scheduler.get_last_lr()[0]:.6f}")
```

`StepLR` here multiplies the learning rate by `gamma=0.1` every `step_size=10` epochs — a simple, classic "step decay" policy. `scheduler.get_last_lr()` returns the current learning rate(s) as a list (one per parameter group, connecting back to Chapter 6, Section 6.3) — useful for logging and sanity-checking that the schedule is behaving as intended.

**`scheduler.step()` does not train the model** — it only updates the learning rate the optimizer will use on subsequent calls to `optimizer.step()`. The two `.step()` calls are entirely separate mechanisms that happen to share a name; conflating them is a common point of confusion for newcomers.

---

## 18.2 Per-Epoch vs. Per-Step Schedulers: A Critical Distinction

This is the single most common source of a learning rate schedule not behaving as intended, and it directly extends the epoch/step/batch precision established in Chapter 8, Section 8.7.

**Some schedulers are designed to be called once per epoch; others are designed to be called once per training step (i.e., once per batch, inside the training loop).** Calling a per-step scheduler once per epoch (or vice versa) doesn't raise an error — it silently produces a schedule that unfolds far too slowly or far too quickly relative to what was intended.

```python
# PER-EPOCH scheduler: step() called once per epoch, after the epoch's batches are done
scheduler_epoch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        # ... optimizer.zero_grad(), forward, backward, optimizer.step() ...
        pass
    scheduler_epoch.step()   # OUTSIDE the batch loop

# PER-STEP scheduler: step() called once per batch, inside the training loop
scheduler_step = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=num_epochs * len(train_loader)
)
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        # ... optimizer.zero_grad(), forward, backward, optimizer.step() ...
        scheduler_step.step()   # INSIDE the batch loop
```

**Always check a scheduler's documentation for which pattern it expects** before adding it to a training loop — `StepLR`, `MultiStepLR`, `ExponentialLR`, and `ReduceLROnPlateau` are conventionally per-epoch; `OneCycleLR` and `CosineAnnealingWarmRestarts` are conventionally per-step (or per-step is at least the common, intended usage, given how their internal cycle length is specified). `total_steps=num_epochs * len(train_loader)` in the `OneCycleLR` example above is exactly why this distinction matters mechanically: that scheduler needs to know the *total number of `.step()` calls* it will receive across the entire run to correctly shape its one-cycle curve, which is only meaningful if you're actually calling it once per batch as intended.

---

## 18.3 Warmup: Why Transformers Need It

**Warmup** means starting training with a small (or zero) learning rate and gradually increasing it over the first portion of training, before switching to the main decay schedule. This connects directly to the pre-norm vs. post-norm discussion in Chapter 13, Section 13.2: transformers — especially deep ones, and especially with post-norm — can be unstable in the very first steps of training, when weights are still close to their random initialization and gradients can be poorly scaled. A large learning rate applied immediately risks a bad early update that the model never fully recovers from; warmup gives the model a gentler on-ramp.

```python
def warmup_lambda(current_step, warmup_steps):
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)   # linearly ramp from 0 to 1
    return 1.0   # full learning rate after warmup completes

warmup_steps = 1000
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: warmup_lambda(step, warmup_steps)
)

for step in range(total_steps):
    # ... training step ...
    scheduler.step()   # per-step, matching the per-step nature of this warmup schedule
```

`LambdaLR` is a flexible, general-purpose scheduler: it multiplies the optimizer's *base* learning rate by whatever your `lr_lambda` function returns at each step, letting you implement essentially any custom schedule as a plain Python function. This is directly relevant to the LLM fine-tuning work you've done — a linear or cosine warmup over the first few hundred to a few thousand steps, before decay, is close to the default recipe in most modern training frameworks (including what `SFTTrainer` configures under the hood for your Qwen2.5-Coder fine-tuning runs).

---

## 18.4 Cosine Annealing

**Cosine annealing** decays the learning rate following a cosine curve, smoothly decreasing from an initial value down toward (typically) zero or a small minimum, with the characteristic property that the rate of decrease is small at the very start and very end of the schedule, and steepest in the middle:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

for epoch in range(num_epochs):
    # ... training for one epoch ...
    scheduler.step()   # per-epoch usage, as configured here via T_max=num_epochs
```

`T_max` is the number of steps (matching whatever unit you call `.step()` on — epochs here) over which the cosine curve completes a full half-period, going from the initial learning rate down to `eta_min`. Cosine annealing has become one of the most common default choices in modern deep learning — including LLM pretraining and fine-tuning recipes — largely because this shape (aggressive decrease in the middle, gentle at both ends) tends to work well empirically across a wide range of tasks, without requiring the careful manual tuning of step boundaries that `StepLR`/`MultiStepLR` demand.

### Warm restarts

`CosineAnnealingWarmRestarts` repeats the cosine decay cycle multiple times, "restarting" the learning rate back up to its initial value periodically — the idea being that periodically increasing the learning rate again can help the model escape a poor local region it may have settled into during a previous decay cycle:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2   # first cycle: 10 steps; each subsequent cycle doubles in length
)
```

---

## 18.5 `OneCycleLR`: Warmup and Decay in a Single Policy

`OneCycleLR` combines warmup and decay into a single, widely-used policy: the learning rate increases from a low starting value up to a specified `max_lr` over roughly the first 25-30% of training, then decreases — often down to a value even smaller than the starting one — over the remainder. It's popularized as part of the "super-convergence" line of research, which found that a single well-tuned cycle like this can sometimes let a model reach good accuracy faster than a constant or simple-decay schedule.

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),   # per-step usage -- Section 18.2
    pct_start=0.3,   # 30% of training spent ramping up, 70% decaying
)

for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()   # per-step, inside the batch loop -- required for OneCycleLR
```

`total_steps` must exactly match the number of times `.step()` will actually be called across the entire run — miscounting this (a common mistake when, say, changing `num_epochs` without updating `total_steps` to match) causes the schedule to either finish early (leaving the rest of training at whatever the final scheduled rate happened to be) or never reach its intended decay phase.

---

## 18.6 Combining Schedulers: Warmup Then Decay

A common real-world pattern — and close to standard practice for training transformers, including LLM fine-tuning — combines linear warmup with a subsequent cosine decay. `SequentialLR` chains multiple schedulers together, switching between them at specified milestones:

```python
warmup_steps = 500
total_steps = 10000

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
)
decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, decay_scheduler],
    milestones=[warmup_steps],   # switch from warmup to decay at this step
)

for step in range(total_steps):
    # ... training step ...
    scheduler.step()   # per-step, driving whichever sub-scheduler is currently active
```

`LinearLR` here ramps the learning rate linearly from `1%` (`start_factor=0.01`) of the base learning rate up to `100%` (`end_factor=1.0`) over `total_iters=500` steps — a slightly more configurable built-in alternative to the manual `LambdaLR` warmup from Section 18.3. `SequentialLR`'s `milestones=[warmup_steps]` tells it to use `warmup_scheduler` for the first 500 steps, then hand off to `decay_scheduler` for the remainder — this exact warmup-then-cosine-decay shape is close to a default recipe you'd recognize from configuring learning rate schedules in Hugging Face's `Trainer`/`SFTTrainer`, underlying much of your own fine-tuning work.

---

## 18.7 `ReduceLROnPlateau`: Scheduling Based on Metrics, Not Steps

Every scheduler covered so far follows a predetermined schedule based purely on step or epoch count. `ReduceLROnPlateau` is different: it monitors a metric (typically validation loss) and reduces the learning rate only when that metric stops improving — a reactive, rather than predetermined, schedule.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

for epoch in range(num_epochs):
    # ... training phase ...
    val_loss = ...   # computed during validation, as in Chapter 8/9

    scheduler.step(val_loss)   # NOTE: takes the metric as an argument, unlike other schedulers
```

`patience=3` means the scheduler waits 3 epochs without improvement in `val_loss` before reducing the learning rate by `factor=0.5` (halving it). This scheduler's `.step()` call signature is distinct from every other scheduler in this chapter — it requires the metric value as an argument, since its decision to reduce the learning rate depends on that value rather than on step count alone. This is easy to forget when switching between scheduler types in existing code, since every other scheduler discussed here takes no arguments to `.step()`.

---

## Summary

- Schedulers wrap an optimizer and adjust its learning rate over time; `scheduler.step()` updates the rate, it does not train the model — that remains `optimizer.step()`'s job.
- The per-epoch vs. per-step distinction is critical and scheduler-specific — check documentation, and ensure the total step/epoch count passed to a scheduler (e.g., `OneCycleLR`'s `total_steps`) matches how often `.step()` is actually called.
- Warmup — gradually increasing the learning rate at the start of training — improves stability for transformers and other architectures sensitive to early large updates.
- Cosine annealing and `OneCycleLR` are widely-used modern defaults; `SequentialLR` combines warmup and decay policies into a single scheduler.
- `ReduceLROnPlateau` reacts to a monitored metric rather than following a predetermined schedule, and uniquely requires that metric as an argument to `.step()`.

## Exercises

1. Train a small model with three different schedulers — constant learning rate, `StepLR`, and `CosineAnnealingLR` — logging the learning rate at each epoch with `get_last_lr()`, and plot all three schedules on one chart to visually compare their shapes.
2. Deliberately call a per-step scheduler (`OneCycleLR`) once per epoch instead of once per batch, and observe how the resulting learning rate curve differs from the intended schedule — connect this to the `total_steps` mismatch described in Section 18.2.
3. Implement a linear warmup followed by cosine decay using `SequentialLR`, as in Section 18.6, and verify (by printing `get_last_lr()`) that the learning rate is exactly `0` (or `start_factor * base_lr`) at step 0, reaches `max_lr` at the warmup/decay boundary, and approaches `eta_min` by the final step.
4. Set up `ReduceLROnPlateau` on a model trained with a validation loss that plateaus partway through training (e.g., using early stopping-style synthetic data), and confirm the learning rate reduction fires at the expected epoch given your chosen `patience`.

**Next:** Chapter 19 covers regularization techniques — dropout, weight decay, label smoothing, and mixup — completing Part IV's toolkit for training models that generalize well, not just models that fit their training data.
