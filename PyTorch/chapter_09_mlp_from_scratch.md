# Chapter 9: Building an MLP from Scratch

## Introduction

This chapter is the capstone of Part II. Rather than introducing new concepts, it walks through a single, complete, end-to-end project — an image classifier trained on FashionMNIST — that uses every piece from Chapters 5 through 8 together, exactly as you'd write it in a real project. If any step feels unfamiliar, that's a signal to revisit the relevant earlier chapter before continuing to Part III.

By the end of this chapter you'll have:
- A complete, runnable training script structured the way real projects are structured
- A working mental checklist for building any new classification project from scratch
- A trained model, saved checkpoint, and a simple inference function to test it on new data

---

## 9.1 The Task and the Data

FashionMNIST is a drop-in replacement for the classic MNIST digit dataset: 28×28 grayscale images across 10 clothing categories (T-shirt, trouser, pullover, etc.), with 60,000 training images and 10,000 test images. It's small enough to train quickly on a single GPU (or even CPU, slowly) while being meaningfully harder than plain MNIST — a good testbed for a first real project.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Loading data with `torchvision`

`torchvision.datasets` provides many standard datasets as ready-made `Dataset` objects (Chapter 7) that download automatically and apply a `transform` pipeline to each sample:

```python
transform = transforms.Compose([
    transforms.ToTensor(),                    # PIL Image -> tensor, scales to [0, 1]
    transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST's known mean/std
])

full_train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)
```

`transforms.ToTensor()` handles the exact NumPy/PIL → PyTorch conversion concerns from Chapter 4 for you — converting a `(H, W, C)` `uint8` image to a `(C, H, W)` `float32` tensor scaled to `[0, 1]`, the same pipeline you built manually in Section 4.5. `transforms.Normalize` then standardizes pixel values using the dataset's known per-channel mean and standard deviation, which typically helps training converge faster and more stably.

### A proper train/validation split

`full_train_dataset` gives us 60,000 images, but we want a held-out validation set distinct from the final test set, so we can monitor generalization during training without touching test data until the very end:

```python
train_size = int(0.9 * len(full_train_dataset))   # 54,000
val_size = len(full_train_dataset) - train_size     # 6,000

train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)       # reproducible split
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
```

Everything here follows Chapter 7's guidance directly: `shuffle=True` only for the training loader, a fixed seed for a reproducible split, and `num_workers` for parallel loading.

---

## 9.2 Defining the Model

A three-layer MLP, following the composition patterns from Chapter 5:

```python
class FashionMNISTClassifier(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, num_classes=10, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()   # (batch, 1, 28, 28) -> (batch, 784)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)   # raw logits — no softmax (Chapter 6, Section 6.2)
        return logits

model = FashionMNISTClassifier().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

A few deliberate choices worth noting, each traceable to an earlier chapter:

- **`nn.Flatten()`** collapses the `(batch, 1, 28, 28)` image into `(batch, 784)` — a module rather than a manual `.view()` call, so it's visible in `print(model)` and composes cleanly.
- **`self.dropout` is instantiated once and reused** — reusing the same `nn.Dropout` module across multiple calls in `forward()` is fine and standard; it's stateless aside from its train/eval mode (Chapter 8, Section 8.2), which is shared correctly since it's all one module.
- **`fc3` returns raw logits**, not softmax output — because we'll use `nn.CrossEntropyLoss`, which expects exactly that (Chapter 6, Section 6.2).
- **`F.relu`** used functionally inline, while the parameterized layers (`Linear`, `Dropout`) are `nn` modules assigned in `__init__` — following the guidance from Chapter 6, Section 6.1.

---

## 9.3 Loss, Optimizer, and the Training Loop

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 15
best_val_loss = float("inf")

for epoch in range(num_epochs):
    # ---- Training ----
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            val_loss += loss.item() * x_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
          f"val_acc={val_accuracy:.4f}")

    # ---- Checkpointing ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }, "best_fashion_mnist_model.pt")
        print(f"  -> Saved new best model (val_loss={val_loss:.4f})")
```

This is, line for line, the Chapter 8 training loop applied to a real dataset and model — `zero_grad → forward → backward → clip → step`, `train()`/`eval()` toggling, `torch.no_grad()` during validation, `.item()` for loss accumulation, and best-model checkpointing.

---

## 9.4 Final Evaluation on the Test Set

The test set should be touched exactly once, after all training and hyperparameter decisions are finalized — using it for anything during development risks tuning your model to it indirectly, defeating its purpose as a genuinely held-out measure of generalization.

```python
checkpoint = torch.load("best_fashion_mnist_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        predictions = logits.argmax(dim=1)

        test_correct += (predictions == y_batch).sum().item()
        test_total += y_batch.size(0)

test_accuracy = test_correct / test_total
print(f"Final test accuracy: {test_accuracy:.4f}")
```

A well-tuned MLP of this size typically reaches roughly 88-89% test accuracy on FashionMNIST — a useful baseline number to compare against once you build a CNN for the same task in Chapter 10, which should meaningfully outperform it by exploiting spatial structure the MLP discards entirely (recall that `nn.Flatten()` throws away all 2D spatial relationships between pixels — this is precisely the limitation convolutional layers are designed to address).

---

## 9.5 A Simple Inference Function

Wrapping prediction into a clean, reusable function — the shape you'd actually deploy or call from other code:

```python
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

@torch.no_grad()
def predict(model, image_tensor, device):
    """image_tensor: shape (1, 28, 28), already normalized like training data"""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)   # add batch dim -> (1, 1, 28, 28)
    logits = model(image_tensor)
    probs = F.softmax(logits, dim=1)                        # for interpretability only
    predicted_idx = probs.argmax(dim=1).item()
    confidence = probs[0, predicted_idx].item()
    return CLASS_NAMES[predicted_idx], confidence

# Example usage on a single test sample
sample_image, true_label = test_dataset[0]
predicted_class, confidence = predict(model, sample_image, device)
print(f"Predicted: {predicted_class} ({confidence:.1%} confidence) | "
      f"True: {CLASS_NAMES[true_label]}")
```

Note `@torch.no_grad()` used as a function decorator here (Chapter 3, Section 3.5) — a clean way to guarantee an inference function never accidentally builds a graph, without needing a `with` block wrapped around every call site. Also note `softmax` is applied *only* here, for a human-readable confidence score — never during training, where `CrossEntropyLoss` needed raw logits directly (Chapter 6).

---

## 9.6 A Reusable Checklist

Distilling this chapter into a checklist you can apply to any new classification project:

1. **Data**: load with an appropriate `Dataset`, split into train/val/test, wrap in `DataLoader`s with correct `shuffle` settings (Chapter 7).
2. **Model**: define an `nn.Module`, output raw logits from the final layer, verify parameter counts and structure with `print(model)` (Chapter 5).
3. **Loss & optimizer**: match the loss function to the task (`CrossEntropyLoss` for multi-class, `BCEWithLogitsLoss` for binary), default to `AdamW` unless you have a specific reason otherwise (Chapter 6).
4. **Training loop**: `train()`/`eval()` mode toggling, `torch.no_grad()` during validation, `.item()` for logging, gradient clipping if needed, checkpoint on best validation metric (Chapter 8).
5. **Final evaluation**: touch the test set exactly once, after everything else is finalized.
6. **Inference**: wrap prediction in a clean function, decorated with `torch.no_grad()`, that handles the full pipeline from raw input to interpretable output.

This checklist will stay valid as you move into Part III's more sophisticated architectures — CNNs, RNNs, and transformers all slot into the exact same surrounding structure; only the model definition in step 2 changes.

---

## Summary

- This chapter combined `Dataset`/`DataLoader` (Ch. 7), `nn.Module` composition (Ch. 5), loss/optimizer choice (Ch. 6), and the full training loop (Ch. 8) into one complete, working project.
- A proper project separates train/validation/test data, uses validation to guide decisions during development, and touches the test set exactly once at the end.
- The resulting checklist — data, model, loss/optimizer, training loop, final evaluation, inference — generalizes to essentially any supervised learning project you'll build for the rest of this book.

## Exercises

1. Run the full pipeline in this chapter and record your final test accuracy. Then try increasing `hidden_size` to 512 and reducing `dropout` to 0.1 — does validation accuracy improve, get worse, or show signs of overfitting (training loss dropping while validation loss rises)?
2. Add a learning rate scheduler stub — even just manually halving the optimizer's `lr` every 5 epochs — and observe its effect on the training curve. (Proper schedulers are covered in Chapter 18.)
3. Modify the `predict()` function to accept a batch of images at once (rather than a single image) and return predictions for all of them in one forward pass.
4. Using `sklearn.metrics.confusion_matrix` (or a manual tensor-based implementation) on the test set predictions, identify which two clothing categories the model confuses most often, and form a hypothesis for why, given what an MLP with `nn.Flatten()` can and can't represent about an image.

---

# Part II Recap

Part II took you from a bare `nn.Module` to a complete, checkpointed, evaluated classification pipeline. Every mechanical piece of a PyTorch project — data loading, model definition, loss/optimizer selection, the training loop, and inference — is now in place.

**Next:** Part III begins with Chapter 10, where convolutional layers replace `nn.Flatten()` and give the model genuine spatial awareness — the first of several architecture families (CNNs, RNNs, attention, transformers) that all plug into the exact same surrounding training infrastructure you just built.
