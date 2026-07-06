# Chapter 10: Convolutional Networks

## Introduction

Chapter 9 ended with a limitation: the MLP classifier used `nn.Flatten()` to collapse a 28×28 image into a flat 784-element vector, throwing away every spatial relationship between pixels — as far as that model was concerned, shuffling all the pixels in a consistent way across the whole dataset wouldn't have changed anything about what it could learn. Convolutional layers exist specifically to fix that: they process images while preserving and exploiting spatial structure.

This chapter builds up `nn.Conv2d` from first principles, then assembles a CNN classifier for the same FashionMNIST task from Chapter 9, so you can directly compare the two architectures on identical data.

By the end of this chapter you'll be able to:
- Explain what a convolution operation actually computes, and why it's well-suited to images
- Use `nn.Conv2d` correctly, including reasoning about kernel size, stride, and padding
- Use pooling layers to downsample feature maps
- Calculate output shapes through a stack of conv/pool layers by hand
- Build and train a complete CNN classifier

---

## 10.1 What a Convolution Actually Computes

A convolutional layer slides a small learnable filter (the **kernel**) across the input, computing a dot product between the kernel and each local patch it covers. Unlike a fully-connected layer — where every output depends on every input, each with its own independent weight — a convolution has two properties that make it dramatically more efficient and better-suited to images:

1. **Local connectivity**: each output value depends only on a small local neighborhood of the input (e.g., a 3×3 patch), not the entire image.
2. **Parameter sharing**: the *same* kernel weights are applied at every spatial position. A kernel that learns to detect a vertical edge detects vertical edges anywhere in the image, using one set of learned weights rather than a separate set per location.

This is precisely the kind of feature you were engineering by hand in your HSV color detection and OpenCV shape detection pipelines for the conveyor sorting system — a convolutional network learns comparable local, spatially-invariant feature detectors (edges, corners, textures, and eventually much more abstract shapes) directly from data, rather than requiring you to specify the detection logic explicitly.

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)

x = torch.rand(1, 1, 8, 8)   # (batch, channels, height, width)
out = conv(x)
print(out.shape)   # torch.Size([1, 4, 6, 6])
```

`in_channels=1` matches a grayscale input; `out_channels=4` means the layer learns 4 independent 3×3 kernels, each producing its own output feature map — so a single-channel input becomes a 4-channel output, one channel per learned filter.

---

## 10.2 The Core Parameters: Kernel Size, Stride, and Padding

### Kernel size

The spatial size of the sliding filter — most commonly `3` (a 3×3 kernel) or `5` in modern architectures. Larger kernels see more context per position but have more parameters and are more expensive to compute.

```python
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)   # 3x3 kernel
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 5))  # non-square kernel, if needed
```

### Stride

How many pixels the kernel moves between each position. `stride=1` (the default) slides the kernel one pixel at a time, producing a densely overlapping output. `stride=2` skips every other position, roughly halving the output's spatial dimensions — a common way to downsample without a separate pooling layer.

```python
conv_stride1 = nn.Conv2d(1, 4, kernel_size=3, stride=1)
conv_stride2 = nn.Conv2d(1, 4, kernel_size=3, stride=2)

x = torch.rand(1, 1, 8, 8)
print(conv_stride1(x).shape)   # torch.Size([1, 4, 6, 6])
print(conv_stride2(x).shape)   # torch.Size([1, 4, 3, 3])
```

### Padding

Adds zeros around the input's border before convolving. Without padding, a convolution shrinks the spatial size on every layer (since the kernel can't center itself on border pixels), which limits how deep a network can go before running out of spatial dimensions entirely. `padding="same"` (or the equivalent explicit integer) keeps the output spatial size equal to the input size when `stride=1`.

```python
conv_no_pad = nn.Conv2d(1, 4, kernel_size=3, padding=0)   # default
conv_same = nn.Conv2d(1, 4, kernel_size=3, padding=1)      # 3x3 kernel needs padding=1 to preserve size
# or, more explicitly:
conv_same_v2 = nn.Conv2d(1, 4, kernel_size=3, padding="same")

x = torch.rand(1, 1, 8, 8)
print(conv_no_pad(x).shape)    # torch.Size([1, 4, 6, 6])  — shrunk
print(conv_same(x).shape)      # torch.Size([1, 4, 8, 8])  — preserved
```

### The output shape formula

For a square input of size $H$, kernel size $k$, stride $s$, and padding $p$:

$$H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1$$

```python
# Verify: H=8, k=3, s=1, p=0  ->  (8 + 0 - 3)/1 + 1 = 6   matches conv_no_pad output above
# Verify: H=8, k=3, s=1, p=1  ->  (8 + 2 - 3)/1 + 1 = 8   matches conv_same output above
```

Being able to compute this by hand — or at least sanity-check it — matters in practice: mismatched shapes between a convolutional stack and the fully-connected layers that follow it are one of the most common errors when building CNNs from scratch, and the error message (a matrix-multiplication shape mismatch, often several layers downstream of the actual cause) rarely points directly at the offending conv layer.

---

## 10.3 Pooling: Downsampling Feature Maps

Pooling layers reduce the spatial size of feature maps, independent of any learned parameters — they apply a fixed aggregation function (max or average) over local regions.

```python
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)   # halves H and W

x = torch.rand(1, 4, 8, 8)
out = maxpool(x)
print(out.shape)   # torch.Size([1, 4, 4, 4])
```

`MaxPool2d` (take the maximum value in each region) is far more common than `AvgPool2d` in practice — it tends to preserve the strongest activations (interpretable as "was this feature present anywhere in this region," which is often exactly what you want for classification), while also providing a small amount of translation invariance: a feature shifted by a pixel or two within the pooling window still produces the same max value.

Pooling and strided convolution both downsample spatial dimensions — modern architectures often use strided convolutions instead of separate pooling layers, since a strided conv can learn *how* to downsample rather than using a fixed rule. Both approaches remain common, and you'll see both in practice; this curriculum uses explicit pooling layers for clarity in the examples below.

---

## 10.4 Building a Complete CNN Classifier

Assembling everything into a CNN for FashionMNIST, directly comparable to the MLP from Chapter 9 — same data, same training loop, only the model architecture changes:

```python
import torch.nn.functional as F

class FashionMNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1: 1 -> 32 channels, 28x28 -> 14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32 -> 64 channels, 14x14 -> 7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   # (batch, 32, 14, 14)
        x = self.pool2(F.relu(self.conv2(x)))    # (batch, 64, 7, 7)
        x = self.flatten(x)                        # (batch, 64*7*7) = (batch, 3136)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)                         # raw logits (Chapter 6)
        return logits

model = FashionMNIST_CNN().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

Trace the shape through the network to confirm the `64 * 7 * 7` in `fc1` isn't a magic number:

- Input: `(batch, 1, 28, 28)`
- After `conv1` (padding=1, so size preserved) + `pool1` (halves): `(batch, 32, 14, 14)`
- After `conv2` (padding=1, preserved) + `pool2` (halves): `(batch, 64, 7, 7)`
- After `flatten`: `(batch, 64*7*7)` = `(batch, 3136)`

This kind of by-hand shape tracing — using the formula from Section 10.2 at each step — is a habit worth building deliberately; it's the fastest way to debug a `RuntimeError: mat1 and mat2 shapes cannot be multiplied` error, which is what you get when the flattened size doesn't match what `fc1` expects.

### Training

The training loop is **identical** to Chapter 9's — this is the whole point of the architecture/infrastructure separation established in Part II. Only the model class changes:

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ... exact same training loop as Chapter 9, Section 9.3, just with this model
```

A well-trained version of this CNN typically reaches roughly 91-92% test accuracy on FashionMNIST — a clear improvement over the MLP's ~88-89% from Chapter 9, using a comparable or smaller number of parameters, purely from exploiting spatial structure the MLP discarded.

---

## 10.5 Growing Channels, Shrinking Space: The Standard CNN Pattern

Notice the pattern in `FashionMNIST_CNN`: channel count *increases* (1 → 32 → 64) while spatial size *decreases* (28 → 14 → 7) as you go deeper. This is close to universal in CNN design, for an intuitive reason: early layers detect simple, local features (edges, colors, textures) and don't need many channels to represent them; deeper layers detect increasingly abstract, complex combinations of those simple features (shapes, object parts) and benefit from more channels to represent a richer vocabulary of patterns — while the actual spatial resolution needed to represent "is this abstract pattern present, and roughly where" shrinks accordingly.

This same principle underlies the backbone of a YOLO-style detector like YOLO-World, which you've integrated into your conveyor sorting pipeline — a sequence of downsampling convolutional blocks with growing channel depth, producing feature maps at multiple scales that a detection head then uses to localize and classify objects.

---

## 10.6 A Note on Input Channels: Grayscale vs. RGB

FashionMNIST is grayscale (`in_channels=1`). For RGB images — the case for essentially all real-world vision tasks, including anything processing camera feeds from a conveyor system — the first convolutional layer simply takes `in_channels=3`:

```python
first_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
```

Each of the 32 output filters in this case learns a kernel that spans *all three* input channels simultaneously (a `3×3×3` kernel per output filter, not three separate `3×3` kernels) — the convolution naturally mixes information across color channels from the very first layer, rather than processing R, G, and B independently.

---

## Summary

- Convolutions exploit local connectivity and parameter sharing, learning spatially-invariant feature detectors directly from data — this is what `nn.Flatten()`-based MLPs discard entirely.
- Kernel size, stride, and padding jointly determine output spatial size, computable via a standard formula — tracing shapes by hand is the fastest way to debug shape-mismatch errors in CNNs.
- `MaxPool2d` downsamples feature maps using a fixed (non-learned) aggregation; strided convolutions offer a learnable alternative to the same goal.
- The standard CNN pattern — growing channel depth, shrinking spatial size — reflects a shift from simple local features early in the network to abstract, high-level features deeper in it.
- Swapping an MLP for a CNN (Chapter 9 → Chapter 10) requires no changes to the surrounding data pipeline, loss/optimizer setup, or training loop — only the model definition changes, which is the payoff of the clean separation established in Part II.

## Exercises

1. By hand, compute the output shape after each layer of a network with: input `(1, 3, 32, 32)`, `Conv2d(3, 16, kernel_size=5, padding=2)`, `MaxPool2d(2)`, `Conv2d(16, 32, kernel_size=3, padding=1)`, `MaxPool2d(2)`. Then verify your answer by running the layers in code.
2. Modify `FashionMNIST_CNN` to use a strided convolution (`stride=2`) instead of `MaxPool2d` for downsampling, adjusting `padding` and channel counts as needed to keep the final flattened size correct. Train it and compare accuracy and parameter count against the pooled version.
3. Add a third convolutional block (128 channels) to `FashionMNIST_CNN`, correctly updating the input size of `fc1`. Does test accuracy improve, and does training take noticeably longer?
4. Build a small CNN for 3-channel (RGB) `64×64` input images with 5 output classes, using the growing-channels/shrinking-space pattern from Section 10.5, and print the model summary with parameter count.

**Next:** Chapter 11 introduces recurrent networks — `nn.RNN`, `nn.LSTM`, and `nn.GRU` — for sequential data where order matters, a fundamentally different kind of structure than the spatial locality convolutions exploit.
