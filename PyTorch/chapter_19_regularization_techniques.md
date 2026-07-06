# Chapter 19: Regularization Techniques

## Introduction

Every technique in this book so far has focused on getting a model to fit its training data effectively. **Regularization** addresses the other half of the problem: getting a model to generalize well to data it hasn't seen, rather than memorizing quirks specific to the training set — a gap known as **overfitting**. This chapter closes out Part IV with the standard regularization toolkit: dropout, weight decay, label smoothing, and mixup, each addressing overfitting through a different mechanism.

By the end of this chapter you'll be able to:
- Recognize overfitting from training/validation loss curves
- Use `nn.Dropout` correctly, including precisely how train/eval mode affects it
- Understand weight decay mechanically, and why `AdamW`'s decoupled version differs from plain `Adam`'s
- Apply label smoothing to prevent overconfident predictions
- Implement mixup as a data-level augmentation technique

---

## 19.1 Recognizing Overfitting

Before applying regularization, it's worth being precise about what you're looking for. The classic signature of overfitting, visible in the train/val loss logging pattern established in Chapter 8/9:

```python
# Healthy training:      train_loss decreases, val_loss decreases roughly alongside it
# Overfitting:            train_loss keeps decreasing, val_loss plateaus or starts increasing
```

A model with far more capacity (parameters) than the training data can meaningfully constrain will eventually start fitting noise specific to the training set rather than the underlying pattern you actually want it to learn — training loss keeps improving because the model is increasingly good at reproducing the exact training examples, while validation loss stalls or worsens because that memorization doesn't transfer to new data. Every technique in this chapter works by making that kind of memorization harder or less rewarding, through different mechanisms.

---

## 19.2 Dropout: Randomly Disabling Units

Dropout, briefly introduced in the Chapter 9 capstone example, randomly zeroes a fraction of activations during training, forcing the network to not rely too heavily on any single unit or narrow combination of units — since any given unit might be zeroed out on a given forward pass, the network is pushed toward representations that are robust to the loss of individual units, which tends to generalize better.

```python
import torch
import torch.nn as nn

dropout = nn.Dropout(p=0.5)   # p = probability of zeroing each element

x = torch.ones(1, 10)
dropout.train()
print(dropout(x))
# tensor([[2., 0., 2., 2., 0., 2., 0., 2., 0., 0.]])  -- roughly half zeroed, survivors scaled up
```

Notice the surviving values are `2.0`, not `1.0` — this is **inverted dropout**, PyTorch's implementation: during training, surviving activations are scaled by `1/(1-p)` (here, `1/0.5 = 2`) to compensate for the zeroed ones, keeping the *expected* sum of activations roughly constant whether or not dropout is applied. This is precisely what allows dropout to be a complete no-op at evaluation time, rather than requiring a separate rescaling step during inference:

```python
dropout.eval()
print(dropout(x))
# tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])  -- unchanged, dropout fully disabled
```

This is exactly the mechanism referenced in Chapter 8, Section 8.2 when discussing why `model.train()`/`model.eval()` toggling matters — `nn.Dropout` is one of the two canonical layers (alongside `nn.BatchNorm`) whose behavior depends entirely on that mode flag, and forgetting to call `.eval()` before validation means dropout stays active, producing noisy, artificially degraded validation metrics that don't reflect the model's true inference-time behavior.

### Where to place dropout, and typical values

Dropout is typically placed after activation functions in fully-connected layers (as in the Chapter 9 `FashionMNISTClassifier`), and is used more sparingly (or not at all) in convolutional layers, where `BatchNorm` (Section 19.5) often already provides a substantial regularizing effect on its own. Common `p` values range from `0.1` to `0.5` — higher for layers with more parameters relative to the amount of training data available, since those are the layers most prone to overfitting in the first place.

---

## 19.3 Weight Decay: Penalizing Large Weights

**Weight decay** adds a penalty proportional to the magnitude of the model's weights to the training objective, encouraging the optimizer to prefer smaller weight values when doing so doesn't meaningfully hurt the primary loss. The intuition: large weights let a model fit sharp, high-variance patterns in the training data — exactly the kind of fitting that tends not to generalize — so discouraging unnecessarily large weights is a broad, effective regularizer.

Mathematically, the classic (L2) formulation adds $\frac{\lambda}{2} \|w\|^2$ to the loss, whose gradient contributes a term proportional to $\lambda w$ to every parameter's gradient — pulling each weight slightly toward zero on every update, in proportion to its current magnitude.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### `AdamW` vs. plain `Adam` with `weight_decay`: a subtle but important distinction

This is worth understanding precisely, since it's a genuinely common point of confusion, and directly connects back to the `AdamW` recommendation from Chapter 6, Section 6.3. Plain `torch.optim.Adam` also accepts a `weight_decay` argument, but implements it by adding the L2 penalty term directly *into the gradient* before Adam's adaptive learning rate machinery processes it — which means the effective weight decay applied ends up interacting with (and being distorted by) each parameter's adaptive learning rate. `AdamW` ("Adam with decoupled Weight decay") instead applies the weight decay update *separately*, directly to the weights, entirely decoupled from the gradient-based adaptive step:

```python
# Adam: weight decay is folded into the gradient before Adam's adaptive scaling -- interacts unpredictably
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

# AdamW: weight decay is applied directly to weights, independent of Adam's adaptive scaling
optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

This decoupling is precisely why `AdamW` has become the standard default over plain `Adam` across modern deep learning, and specifically in LLM fine-tuning — the decoupled version behaves more predictably and consistently across different learning rates and model scales, which matters when you're tuning hyperparameters for something like the Qwen2.5-Coder SFT/GRPO runs, since a weight decay value that behaves sensibly at one learning rate should continue to behave sensibly if the learning rate changes, which is far less true for plain `Adam`'s coupled version.

### Excluding certain parameters from weight decay

A common refinement, particularly in transformer training: apply weight decay to weight matrices, but exclude biases and normalization parameters (`LayerNorm`'s learnable scale/shift, Chapter 13), since penalizing those toward zero doesn't carry the same "prefer simpler patterns" justification and can actively hurt training. This uses exactly the parameter groups pattern from Chapter 6/14:

```python
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "bias" in name or "norm" in name.lower():
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": 0.01},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=1e-3)
```

---

## 19.4 Label Smoothing: Discouraging Overconfidence

`nn.CrossEntropyLoss` (Chapter 6, Section 6.2), used with standard integer class-index targets, implicitly asks the model to push the predicted probability for the correct class toward exactly `1.0` and every other class toward exactly `0.0` — an infinitely confident target that a model can only approach by driving its logits to increasingly extreme values, which tends to correlate with overfitting and generally produces poorly-calibrated confidence estimates (a model that says "99.99% confident" far more often than it's actually correct 99.99% of the time).

**Label smoothing** softens this target: instead of a one-hot target, it uses a slightly "smoothed" distribution — mostly concentrated on the correct class, but with a small amount of probability mass spread across the other classes:

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

logits = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])

loss = loss_fn(logits, target)
```

With `label_smoothing=0.1` on a 3-class problem, instead of the target distribution being `[1.0, 0.0, 0.0]`, `CrossEntropyLoss` internally uses something like `[0.933, 0.033, 0.033]` — the correct class still receives the overwhelming majority of the target probability, but the model is no longer asked to drive its confidence to the theoretical extreme, which in practice tends to produce better-calibrated, more robust predictions with a negligible amount of extra code (a single argument to a loss function you're already using).

This is a genuinely low-cost regularizer — unlike dropout or mixup, it requires no architectural change and doesn't slow down training, which is part of why it's become a common default in modern image classification and language model training recipes alike.

---

## 19.5 A Brief Note on `BatchNorm` as a Regularizer

`nn.BatchNorm2d` (used throughout modern CNN architectures, including the `torchvision.models` backbones from Chapter 14) normalizes activations using per-batch statistics during training — mentioned here specifically because, as a side effect of that batch-dependent normalization, it introduces a small amount of noise into training (since the exact normalization applied to a given sample depends on which other samples happen to share its batch), which itself has a mild regularizing effect, similar in spirit to (though mechanically distinct from) dropout. This is a secondary benefit rather than `BatchNorm`'s primary purpose (which is training stability and allowing higher learning rates), but it's part of why architectures with heavy `BatchNorm` usage sometimes need less additional dropout than architectures without it. A full treatment of `BatchNorm`'s mechanics is out of scope for this chapter — the relevant point here is purely its role as one of several complementary regularization sources you're likely already using without necessarily thinking of it as regularization.

---

## 19.6 Mixup: Data-Level Regularization

Every technique so far modifies the model or the loss function. **Mixup** instead modifies the *data*: it trains on convex combinations of pairs of training examples and their labels, rather than on the original examples directly.

```python
def mixup_data(x, y, num_classes, alpha=0.2):
    """x: (batch, ...) input tensor. y: (batch,) integer class labels."""
    batch_size = x.size(0)

    lam = torch.distributions.Beta(alpha, alpha).sample().item()   # mixing coefficient
    perm = torch.randperm(batch_size)                                # random pairing within the batch

    mixed_x = lam * x + (1 - lam) * x[perm]

    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[perm]

    return mixed_x, mixed_y

# Inside the training loop:
x_batch, y_batch = next(iter(train_loader))
mixed_x, mixed_y = mixup_data(x_batch, y_batch, num_classes=10, alpha=0.2)

logits = model(mixed_x)
log_probs = torch.nn.functional.log_softmax(logits, dim=1)
loss = -(mixed_y * log_probs).sum(dim=1).mean()   # cross-entropy against a SOFT target
```

Two mechanical details worth noting: mixup requires a soft (non-integer) target, so the standard `nn.CrossEntropyLoss` (which expects integer class indices, Chapter 6, Section 6.2) can't be used directly — the manual `log_softmax` + weighted sum shown above is the standard workaround, computing cross-entropy against an arbitrary probability distribution rather than a one-hot target. The mixing coefficient `lam` is drawn from a `Beta(alpha, alpha)` distribution, whose shape depends on `alpha` — small values of `alpha` (e.g., `0.2`, used above) produce a distribution concentrated near `0` and `1` (mostly one image dominating, with a small amount of the other blended in), while `alpha=1.0` produces a uniform distribution over the mixing ratio.

The regularizing effect: the model is trained on inputs that don't correspond to any single real class, with correspondingly blended targets — this discourages the sharp, overconfident decision boundaries that come from training exclusively on "pure" examples with hard one-hot labels, pushing the model toward smoother, more linear behavior between classes, which tends to generalize better and also improves robustness to slightly out-of-distribution inputs at inference time.

---

## 19.7 Combining Regularization Techniques

These techniques are not mutually exclusive, and real training recipes commonly combine several:

```python
model = FashionMNIST_CNN().to(device)   # includes nn.Dropout, Chapter 10

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)   # weight decay
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)                                 # label smoothing

# ... training loop optionally also applying mixup_data() per batch ...
```

**Practical guidance:** don't apply every technique in this chapter simultaneously by default — each adds a hyperparameter to tune, and over-regularizing (too much dropout, too aggressive weight decay) can hurt performance just as much as under-regularizing, by preventing the model from fitting even the genuine signal in the training data. Start simple (weight decay via `AdamW`, which you likely already have from Chapter 6's default guidance, plus modest dropout if your architecture doesn't already include `BatchNorm` throughout), and add label smoothing or mixup specifically if you observe the overfitting signature from Section 19.1 persisting after those first steps.

---

## Summary

- Overfitting shows up as training loss continuing to improve while validation loss plateaus or worsens — every technique in this chapter addresses that gap through a different mechanism.
- `nn.Dropout` randomly zeroes activations during training (with inverted scaling to keep expected magnitude constant) and is fully disabled in eval mode — another concrete reason `model.train()`/`model.eval()` toggling matters.
- Weight decay penalizes large weights; `AdamW`'s decoupled implementation is preferred over plain `Adam`'s coupled version because it behaves more predictably across learning rates, which is part of why `AdamW` is the standard default (Chapter 6).
- Label smoothing softens classification targets away from one-hot extremes, discouraging overconfidence at essentially no implementation cost.
- Mixup trains on convex combinations of input pairs and their labels, a data-level regularizer requiring a soft-target loss formulation rather than standard `CrossEntropyLoss`.
- These techniques combine well but shouldn't all be applied maximally by default — start simple and add regularization specifically in response to observed overfitting.

## Exercises

1. Train the `FashionMNIST_CNN` from Chapter 10 with dropout `p` values of `0.0`, `0.3`, and `0.7`, and compare the train/val loss gap for each — identify which setting shows the clearest overfitting signature and which shows signs of over-regularization (both train and val performance suffering).
2. Train the same model with plain `Adam` and `AdamW`, both using `weight_decay=0.1` (a relatively large value), and compare final validation accuracy — connect any difference you observe to the coupled vs. decoupled weight decay explanation in Section 19.3.
3. Add `label_smoothing=0.1` to an existing classification training run and compare the model's confidence on a few individual test predictions (via `softmax` probabilities, Chapter 9, Section 9.5) against a version trained without label smoothing.
4. Implement the full `mixup_data` training loop from Section 19.6 on the FashionMNIST classifier, and visually inspect a few mixed images (as blended pixel arrays) alongside their soft labels to build intuition for what the model is actually being trained on.

---

# Part IV Recap

Part IV covered the training-mechanics toolkit that sits around model architecture: mixed precision for speed and memory, custom autograd for operations beyond what automatic differentiation handles natively, hooks for introspection and debugging, learning rate scheduling for shaping training dynamics over time, and regularization for generalization. Combined with Part II's training loop and Part III's architectures, you now have essentially everything needed to train a serious model end to end.

**Next:** Part V shifts focus to performance and systems — starting with Chapter 20, profiling PyTorch code with `torch.profiler` to find out exactly where a training run is actually spending its time, before optimizing anything further.
