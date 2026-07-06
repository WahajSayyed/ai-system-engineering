# Chapter 14: Transfer Learning

## Introduction

Every model built so far in this book has been trained from randomly initialized weights. For many real-world tasks — especially with limited data — that's wasteful: a model pretrained on a large, general dataset has already learned broadly useful features (edges, textures, shapes for vision; syntax, semantics for language) that transfer well to a new, related task. **Transfer learning** means starting from those pretrained weights instead of random initialization, then adapting the model to your specific problem.

This chapter closes out Part III with the practical mechanics of transfer learning in PyTorch: loading pretrained models from `torchvision`, deciding what to freeze versus fine-tune, and the parameter-group patterns needed to train frozen and unfrozen parts of a model together correctly.

By the end of this chapter you'll be able to:
- Load and adapt a pretrained model from `torchvision.models`
- Replace a model's classification head for a new task
- Freeze layers correctly, and understand what freezing actually does mechanically
- Choose between feature extraction and fine-tuning, and combine both with differential learning rates
- Apply the same normalization the pretrained model was trained with — a step that's easy to forget and quietly ruins results

---

## 14.1 Loading a Pretrained Model

`torchvision.models` provides a wide range of architectures with weights pretrained on ImageNet (1.28M images, 1000 classes) available for download:

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print(model)
```

The `weights=` argument (the modern API, replacing the older `pretrained=True` boolean) accepts a specific weights enum, giving you explicit control over exactly which pretrained checkpoint you get — useful since some architectures have multiple available pretrained versions trained with different recipes. `DEFAULT` selects the best-performing available weights for that architecture.

```python
weights = models.ResNet18_Weights.DEFAULT
print(weights.meta["categories"][:5])   # first 5 of the 1000 ImageNet class names
```

---

## 14.2 Replacing the Classification Head

A pretrained ResNet18 was trained to output 1000 ImageNet class logits — almost never what you actually want. The standard adaptation is to replace the final layer with a new one sized for your task, while keeping everything before it:

```python
print(model.fc)   # Linear(in_features=512, out_features=1000, bias=True)

num_classes = 10   # e.g., your own dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

print(model.fc)   # Linear(in_features=512, out_features=10, bias=True) -- new, randomly initialized
```

This directly applies Chapter 5's understanding of `nn.Module` composition: `model.fc` is just an attribute like any other, and reassigning it swaps out that submodule entirely — the new `nn.Linear` is freshly, randomly initialized (Chapter 5, Section 5.2), while every other layer in `model` retains its pretrained weights untouched.

Different architectures name their final layer differently — worth checking with `print(model)` before assuming `.fc` is correct:

```python
# ResNet family: final layer is `.fc`
model_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)

# VGG / EfficientNet family: final layer is inside `.classifier`, often a Sequential
model_vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
print(model_vgg.classifier)   # a Sequential — the last element is the layer to replace
model_vgg.classifier[-1] = nn.Linear(model_vgg.classifier[-1].in_features, num_classes)
```

---

## 14.3 Freezing Layers: What It Actually Does

"Freezing" a layer means preventing its weights from being updated during training. Mechanically, this is just setting `requires_grad = False` on its parameters — directly connecting back to Chapter 3's autograd fundamentals.

```python
for param in model.parameters():
    param.requires_grad = False

# Then unfreeze just the new classification head
for param in model.fc.parameters():
    param.requires_grad = True
```

It's worth being precise about what this does and doesn't do:

- **Gradients still compute and flow through frozen layers during backpropagation** (they have to, in order to reach earlier unfrozen layers, or simply because the graph passes through them) — `requires_grad = False` only prevents the optimizer from *updating* those parameters, it doesn't remove them from the forward/backward computation.
- **The optimizer must still only be given trainable parameters**, or it'll waste memory tracking momentum/state for parameters that never change:

```python
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),   # only trainable params
    lr=1e-3
)
```

Passing `model.parameters()` directly (unfiltered) to the optimizer when some are frozen isn't strictly wrong — the optimizer will just never update the frozen ones, since their `.grad` will be `None` (no gradient is retained for a `requires_grad=False` leaf) — but filtering is more explicit and avoids the optimizer maintaining unnecessary internal state for parameters it'll never touch.

### Verifying what's frozen

A useful sanity check before starting a training run, since silently forgetting to freeze/unfreeze the intended layers is a common source of confusing results (a "fine-tuning" run that mysteriously doesn't converge, or a "feature extraction" run that overfits, are both worth checking against this first):

```python
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
```

---

## 14.4 Feature Extraction vs. Fine-Tuning

Two distinct strategies, differing in how much of the pretrained model you allow to change:

### Feature extraction: freeze everything except the new head

Treat the pretrained backbone as a fixed feature extractor, training only the newly added classification head. This is fast, requires little data, and is a strong choice when your dataset is small or closely related to the original pretraining data (ImageNet, for general natural images):

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)   # new layer, requires_grad=True by default

optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)   # only the head has params to train
```

### Fine-tuning: unfreeze some or all layers, train with a small learning rate

Allow the pretrained weights themselves to adapt to the new task, typically with a much smaller learning rate than you'd use for training from scratch — the pretrained weights are already in a good region of the loss landscape, and large updates risk destroying that useful initialization ("catastrophic forgetting" of the pretrained features):

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Everything is trainable by default (requires_grad=True) -- no freezing needed here
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)   # note: much smaller lr than 1e-3
```

### A common middle ground: freeze early layers, fine-tune later ones

Early layers in a CNN learn generic, broadly transferable features (edges, colors, simple textures — recall Chapter 10, Section 10.5's discussion of growing channel depth and increasing abstraction with depth); later layers learn more task-specific, abstract features. A common and often effective strategy freezes the early layers (keeping their generic features fixed) while fine-tuning the later layers (adapting the more abstract, task-relevant representations) plus the new head:

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last block (layer4) and the new head
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True
```

### Differential learning rates: fine-tune everything, but not equally

Rather than a hard freeze/unfreeze split, you can keep every layer trainable but assign different learning rates to different parts of the model — smaller for the pretrained backbone, larger for the new head — using exactly the parameter groups pattern introduced in Chapter 6, Section 6.3:

```python
optimizer = torch.optim.AdamW([
    {"params": model.layer1.parameters(), "lr": 1e-6},
    {"params": model.layer2.parameters(), "lr": 1e-6},
    {"params": model.layer3.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-3},
])
```

This approach lets the whole network adapt somewhat while still respecting the intuition that pretrained early layers need only gentle nudging, while the randomly-initialized head needs to move much further from its starting point.

**Practical guidance:** feature extraction first, as a fast baseline and sanity check that the overall pipeline works; then fine-tuning (full or partial, or with differential learning rates) if you have enough data and feature extraction alone isn't sufficient. This progression mirrors, at a larger scale, the SFT-first-then-further-training sequencing you've used in your own LLM fine-tuning work — establish a stable baseline before allowing more aggressive adaptation.

---

## 14.5 Preprocessing: Matching the Pretrained Model's Expectations

This is the step most likely to be silently skipped, and doing so quietly degrades results without raising an obvious error. A pretrained model expects input preprocessed *exactly* the way its original training data was — different input statistics than what it saw during pretraining means every subsequent layer receives out-of-distribution activations from the very first layer onward.

`torchvision`'s weights objects conveniently bundle the correct preprocessing transform directly:

```python
weights = models.ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
print(preprocess)
```

```
ImageClassification(
    crop_size=[224]
    resize_size=[256]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    interpolation=InterpolationMode.BILINEAR
)
```

Use it directly in your data pipeline, exactly the way `transforms.Compose` was used in Chapter 9, Section 9.1:

```python
from torchvision import datasets

dataset = datasets.ImageFolder(root="./my_data", transform=preprocess)
```

The specific `mean`/`std` values here are ImageNet's per-channel statistics — the same normalization concept introduced in Chapter 9, but the values themselves must match what the *pretrained model* was trained with, not values recomputed from your own dataset. This is worth double-checking explicitly any time you adapt an existing data pipeline to use a pretrained model, since a working, syntactically-correct pipeline using the *wrong* normalization values is one of the more insidious bugs in transfer learning — training will still proceed and often still "sort of" work, just noticeably worse than it should.

---

## 14.6 A Complete Transfer Learning Example

Putting it together — feature extraction on a small custom image classification dataset, reusing the exact training loop from Chapter 8/9 unchanged:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load pretrained model and its matching preprocessing
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
preprocess = weights.transforms()

# 2. Freeze the backbone, replace the head
for param in model.parameters():
    param.requires_grad = False

num_classes = 5   # e.g., 5 custom categories
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 3. Data, using the pretrained model's own preprocessing
train_dataset = datasets.ImageFolder(root="./my_data/train", transform=preprocess)
val_dataset = datasets.ImageFolder(root="./my_data/val", transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Loss and optimizer -- only the head has trainable parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)

# 5. Training loop -- identical to Chapter 8/9, no changes needed
for epoch in range(10):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
    # ... validation phase, exactly as in Chapter 8
```

Notice, once again, that the training loop itself required zero changes — the same structural payoff observed when moving from MLP to CNN in Chapter 10. Transfer learning only changes *how the model is constructed and which parameters are trainable*; everything downstream in the pipeline stays the same.

---

## Summary

- `torchvision.models` provides pretrained architectures via a `weights=` argument; use `weights.transforms()` to get the exact preprocessing the model expects.
- Replace a model's final layer (name varies by architecture — check with `print(model)`) to adapt it to a new number of classes.
- Freezing is just `requires_grad = False` — it stops the optimizer from updating those parameters, but gradients still flow through frozen layers during backpropagation on their way to earlier unfrozen ones.
- Feature extraction (freeze everything but the head) is fast and data-efficient; full or partial fine-tuning (small learning rate, or differential learning rates per layer group) adapts more of the model at the cost of needing more data and careful tuning.
- Always match the pretrained model's original preprocessing exactly — mismatched normalization is a silent, easy-to-miss source of degraded performance.

## Exercises

1. Load a pretrained `resnet18`, freeze all layers, replace the head for a 3-class problem, and print `named_parameters()` with their `requires_grad` status to confirm only the head is trainable.
2. Implement the "freeze early layers, fine-tune later ones" strategy from Section 14.4 on a different architecture (e.g., `resnet34`), unfreezing the last two residual blocks instead of just one.
3. Set up an optimizer with three parameter groups (early layers, late layers, head) using three different learning rates, and verify the configuration by printing `optimizer.param_groups`.
4. Deliberately use the wrong normalization statistics (e.g., plain `[0.5, 0.5, 0.5]` mean/std instead of ImageNet's) when fine-tuning a pretrained model on a small dataset, and compare final validation accuracy against the correctly-preprocessed version.

---

# Part III Recap

Part III covered the major architecture families built on top of Part II's infrastructure: CNNs for spatial data, RNNs for sequential data, attention and transformers for relating arbitrary positions directly, and transfer learning for leveraging pretrained models rather than always training from scratch.

**Next:** Part IV shifts focus from architecture to training mechanics at scale, starting with Chapter 15 — mixed precision training, using `torch.cuda.amp` to train faster and with less memory, directly relevant to getting the most out of your RTX 3090's Tensor Cores.
