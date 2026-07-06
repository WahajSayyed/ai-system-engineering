# Chapter 7: Thresholding & Binarization

> *"Thresholding is not a decision — it is the revelation of a decision the image has already made. Your job is to find the right question to ask it."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain what binarization does and why it is a fundamental preprocessing step
2. Apply all five `cv2.threshold` types and understand when each is appropriate
3. Derive and implement Otsu's method from its mathematical foundation
4. Explain why Otsu's method fails on non-bimodal images and what to do instead
5. Apply adaptive thresholding correctly, choosing `blockSize` and `C` with principled reasoning
6. Build a complete preprocessing pipeline for document binarization
7. Combine threshold types with bitwise logic for multi-band segmentation
8. Use `cv2.inRange` for multi-channel thresholding (revisited from Chapter 3)
9. Diagnose thresholding failures and apply the right corrective strategy
10. Benchmark all methods and understand the speed vs quality tradeoff

---

## 7.1 What Is Thresholding?

Thresholding converts a continuous-valued grayscale image into a **binary image** — a map of decisions. Every pixel becomes either 0 (background, black) or `maxval` (foreground, white). This reduces an image with 256 possible intensity levels to just 2, collapsing uncertainty into a yes/no judgment for every pixel.

Why reduce information so aggressively? Because most downstream algorithms — contour detection, connected components, shape analysis, OCR — are designed to work on binary inputs. Thresholding is the bridge between raw pixel values and structured, discrete objects.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Conceptual demonstration: what thresholding does
intensities = np.arange(256)
threshold_val = 127

binary = np.where(intensities > threshold_val, 255, 0)
print("After threshold=127:")
print(f"  Pixel 0   → {binary[0]}")     # 0   (dark background)
print(f"  Pixel 127 → {binary[127]}")   # 0   (at threshold → background)
print(f"  Pixel 128 → {binary[128]}")   # 255 (above threshold → foreground)
print(f"  Pixel 255 → {binary[255]}")   # 255 (bright foreground)
```

### 7.1.1 When Thresholding Works and When It Fails

Thresholding works well when there is a clear **intensity difference** between the objects of interest and the background. It struggles — sometimes catastrophically — when:

- **Non-uniform illumination** creates a gradient across the image (shadows, uneven lighting)
- The object and background have **similar intensities** in some regions
- **Noise** causes individual pixels to cross the threshold spuriously
- The image has **more than two meaningful intensity classes** (e.g., three objects of different brightness)

Understanding these failure modes tells you which thresholding method to apply in each situation. We will revisit each failure mode with worked examples.

---

## 7.2 cv2.threshold: Global Thresholding

The function `cv2.threshold` applies a single threshold value to the entire image — one number decides the fate of every pixel:

```python
retval, dst = cv2.threshold(src, thresh, maxval, type)
```

| Parameter | Description |
|-----------|-------------|
| `src` | Input image — **must be grayscale** (single channel) |
| `thresh` | Threshold value (0–255 for uint8) |
| `maxval` | Value assigned to pixels that pass the threshold (typically 255) |
| `type` | Threshold operation type (see Section 7.3) |
| `retval` | The threshold actually used (same as `thresh` unless Otsu/Triangle) |
| `dst` | Output binary image, same size as `src`, dtype uint8 |

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    rng = np.random.default_rng(42)
    img = np.zeros((200, 300), dtype=np.uint8)
    # Gradient background + bright object
    for x in range(300):
        img[:, x] = int(x / 300 * 80)   # gradient background 0→80
    cv2.circle(img, (150, 100), 70, 200, -1)   # bright circle, value=200
    cv2.rectangle(img, (20, 130), (100, 180), 160, -1)  # medium-bright rect

# Apply THRESH_BINARY with threshold=127
retval, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print(f"Return value: {retval}")  # 127.0 (same as thresh for non-Otsu)
print(f"Output dtype: {binary.dtype}")   # uint8
print(f"Unique values: {np.unique(binary)}")  # [0, 255]
```

---

## 7.3 The Five Threshold Types

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a grayscale ramp to demonstrate all threshold types
img_ramp = np.tile(np.arange(256, dtype=np.uint8), (60, 1))  # 60×256 gradient
thresh = 127
maxval = 255

types = [
    (cv2.THRESH_BINARY,     "THRESH_BINARY",
     "0 if pix≤thresh, maxval if pix>thresh"),
    (cv2.THRESH_BINARY_INV, "THRESH_BINARY_INV",
     "maxval if pix≤thresh, 0 if pix>thresh"),
    (cv2.THRESH_TRUNC,      "THRESH_TRUNC",
     "thresh if pix>thresh, pix otherwise (clips highs)"),
    (cv2.THRESH_TOZERO,     "THRESH_TOZERO",
     "0 if pix≤thresh, pix otherwise (zeros lows)"),
    (cv2.THRESH_TOZERO_INV, "THRESH_TOZERO_INV",
     "pix if pix≤thresh, 0 otherwise (zeros highs)"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes.flat[0].imshow(img_ramp, cmap="gray", vmin=0, vmax=255)
axes.flat[0].set_title("Input: gradient 0→255"); axes.flat[0].axis("off")

for ax, (ttype, name, description) in zip(axes.flat[1:], types):
    _, result = cv2.threshold(img_ramp, thresh, maxval, ttype)
    ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{name}\n{description}", fontsize=8)
    ax.axis("off")

plt.suptitle(f"All threshold types (thresh={thresh}, maxval={maxval})", fontsize=12)
plt.tight_layout(); plt.show()
```

### 7.3.1 Detailed Type Descriptions

```python
import numpy as np

# For a pixel value p, threshold T, maxval M:
p = np.array([50, 100, 127, 128, 200, 255])
T, M = 127, 255

print(f"Input:              {p}")
print(f"THRESH_BINARY:      {np.where(p > T, M, 0)}")
# 50→0, 100→0, 127→0, 128→M, 200→M, 255→M
# "foreground if bright, background if dark"

print(f"THRESH_BINARY_INV:  {np.where(p > T, 0, M)}")
# 50→M, 100→M, 127→M, 128→0, 200→0, 255→0
# "foreground if dark, background if bright" (useful: dark text on bright paper)

print(f"THRESH_TRUNC:       {np.minimum(p, T)}")
# Clips values above T to exactly T; values below unchanged
# 50→50, 100→100, 127→127, 128→127, 200→127, 255→127
# Not binary — output still has range [0, T]

print(f"THRESH_TOZERO:      {np.where(p > T, p, 0)}")
# Keep values above T, zero out everything ≤ T
# 50→0, 100→0, 127→0, 128→128, 200→200, 255→255
# Not binary — useful for zeroing out background

print(f"THRESH_TOZERO_INV:  {np.where(p > T, 0, p)}")
# Keep values ≤ T, zero out above T
# 50→50, 100→100, 127→127, 128→0, 200→0, 255→0
# Not binary — keeps the background intensity intact
```

> **Key detail:** Only `THRESH_BINARY` and `THRESH_BINARY_INV` produce true binary (0/255) output. `THRESH_TRUNC`, `THRESH_TOZERO`, and `THRESH_TOZERO_INV` produce images where values span a range. All five can be combined with `THRESH_OTSU` or `THRESH_TRIANGLE`.

---

## 7.4 Otsu's Method: Automatic Global Thresholding

### 7.4.1 The Problem with Manual Thresholds

Setting a threshold manually requires trial and error and is brittle — a threshold that works on one image may fail on another with different lighting or contrast. Otsu's method, introduced by Nobuyuki Otsu in 1979, solves this by automatically finding the optimal threshold from the image's histogram.

### 7.4.2 Mathematical Derivation

Otsu's method treats thresholding as an optimization problem. Given a threshold `t`, pixels are assigned to one of two classes:
- Class 1 (background): pixels with intensity ≤ t
- Class 2 (foreground): pixels with intensity > t

For each candidate threshold `t` (scanning all 256 levels), Otsu computes the **between-class variance**:

```
σ²_b(t) = w₁(t) · w₂(t) · [μ₁(t) - μ₂(t)]²
```

Where:
- `w₁(t)` = fraction of pixels in class 1 (cumulative probability up to t)
- `w₂(t)` = fraction of pixels in class 2 = 1 - w₁(t)
- `μ₁(t)` = mean intensity of class 1 pixels
- `μ₂(t)` = mean intensity of class 2 pixels

The optimal threshold `t*` maximizes `σ²_b(t)`. Maximizing between-class variance is mathematically equivalent to minimizing within-class variance (the sum of per-class intensity variances weighted by class size). An optimally split histogram will have two tight, well-separated clusters.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def otsu_manual(img_gray):
    """
    Manual implementation of Otsu's thresholding algorithm.
    Shows the full derivation step by step.

    Returns (optimal_threshold, between_class_variances)
    """
    assert img_gray.ndim == 2, "Input must be grayscale"
    assert img_gray.dtype == np.uint8

    # Step 1: Compute normalized histogram (probability distribution)
    n_pixels = img_gray.size
    hist = np.bincount(img_gray.ravel(), minlength=256)
    p = hist / n_pixels    # probability of each intensity level

    # Step 2: Precompute cumulative sums for efficiency
    # w1(t) = cumulative probability (weight of class 1 up to t)
    # mu_global = overall mean intensity
    cum_prob = np.cumsum(p)                      # w1(t) for each t
    cum_mean = np.cumsum(np.arange(256) * p)     # μ_total(t): cumulative mean

    mu_global = cum_mean[-1]   # global mean = sum(i * P(i))

    # Step 3: Compute between-class variance for each threshold
    # w1(t) and w2(t)
    w1 = cum_prob                # shape (256,)
    w2 = 1.0 - w1

    # μ1(t) = cum_mean(t) / w1(t)
    # μ2(t) = (μ_global - cum_mean(t)) / w2(t)
    # σ²_b(t) = w1(t) * w2(t) * (μ1 - μ2)²

    # Safe division (avoid div by zero at extremes)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu1 = np.where(w1 > 0, cum_mean / w1, 0)
        mu2 = np.where(w2 > 0, (mu_global - cum_mean) / w2, 0)

    sigma_b_sq = w1 * w2 * (mu1 - mu2) ** 2

    # Step 4: Find threshold that maximizes between-class variance
    t_star = int(np.argmax(sigma_b_sq))

    return t_star, sigma_b_sq


# ── Test on a synthetic bimodal image ─────────────────────────────────────────
img_bimodal = np.zeros((200, 400), dtype=np.uint8)
img_bimodal[:, :200] = 60    # background region: mean=60
img_bimodal[:, 200:] = 190   # foreground region: mean=190

# Add Gaussian noise
noise = np.random.normal(0, 15, img_bimodal.shape)
img_bimodal = np.clip(img_bimodal.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# Our manual Otsu
t_manual, sigma_b = otsu_manual(img_bimodal)
print(f"Manual Otsu threshold:  {t_manual}")

# OpenCV's Otsu
retval, binary_otsu = cv2.threshold(img_bimodal, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"OpenCV Otsu threshold:  {int(retval)}")

# They should be identical (or differ by at most 1 due to implementation)
print(f"Difference: {abs(t_manual - int(retval))}")   # 0 or 1

# ── Visualize the between-class variance curve ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

hist_vals = np.bincount(img_bimodal.ravel(), minlength=256)
axes[0].bar(range(256), hist_vals, color="steelblue", alpha=0.7)
axes[0].axvline(t_manual, color="red", linewidth=2, label=f"Otsu t={t_manual}")
axes[0].set_title("Histogram with Otsu threshold")
axes[0].set_xlabel("Intensity"); axes[0].legend()

axes[1].plot(sigma_b, color="darkgreen", linewidth=1.5)
axes[1].axvline(t_manual, color="red", linewidth=2, label=f"t*={t_manual}")
axes[1].set_title("Between-class variance σ²_b(t)")
axes[1].set_xlabel("Threshold t"); axes[1].legend()

axes[2].imshow(binary_otsu, cmap="gray", vmin=0, vmax=255)
axes[2].set_title(f"Otsu binary (t={int(retval)})")
axes[2].axis("off")

plt.tight_layout(); plt.show()
```

### 7.4.3 Using Otsu in OpenCV

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (200, 300), dtype=np.uint8)

# Basic Otsu: combine THRESH_BINARY with THRESH_OTSU
# The threshold value (second argument) is IGNORED when THRESH_OTSU is set
retval, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold found: {retval:.0f}")

# Otsu with inverted output (for dark text on bright background)
retval_inv, binary_inv = cv2.threshold(img, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(f"Same threshold, inverted output: {retval_inv:.0f}")
# retval == retval_inv (same threshold, just different output mapping)

# Otsu TRUNC (less common but valid)
retval_trunc, trunc = cv2.threshold(img, 0, 255,
                                     cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

# ── Gold-standard pipeline: Gaussian pre-blur then Otsu ──────────────────────
# Noise corrupts the histogram and shifts the Otsu threshold. Pre-blurring
# smooths noise → cleaner bimodal histogram → more accurate Otsu.
blurred = cv2.GaussianBlur(img, (5, 5), 0)
retval_clean, binary_clean = cv2.threshold(blurred, 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Blur+Otsu threshold:  {retval_clean:.0f}")
```

### 7.4.4 The Triangle Method: Alternative Auto-Threshold

OpenCV also implements the **Triangle method** (Zack, Rogers & Latt, 1977), an automatic threshold algorithm that works better when the histogram is **unimodal** — when the foreground is a small, sparse set of bright pixels on a predominantly dark background (or vice versa):

```python
import cv2
import numpy as np

# THRESH_TRIANGLE: finds threshold at the "knee" of a unimodal histogram
# Works better than Otsu for: fluorescence microscopy, sparse foreground objects
img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (200, 300), dtype=np.uint8)

retval_triangle, binary_triangle = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
)
print(f"Triangle threshold: {retval_triangle:.0f}")

# Compare Otsu vs Triangle
retval_otsu, binary_otsu = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"Otsu threshold:     {retval_otsu:.0f}")
```

### 7.4.5 When Otsu Fails: Diagnosis

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def diagnose_otsu(img_gray, title="Image"):
    """
    Plot the histogram and Otsu threshold for diagnosis.
    Shows why Otsu may fail on non-bimodal images.
    """
    # Compute Otsu threshold
    retval, binary = cv2.threshold(img_gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hist = np.bincount(img_gray.ravel(), minlength=256)

    # Assess bimodality: count histogram modes
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hist, height=hist.max() * 0.1, distance=20)
    n_modes = len(peaks)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"{title} (Otsu t={retval:.0f})")
    axes[0].axis("off")

    axes[1].bar(range(256), hist, color="steelblue", alpha=0.8, width=1)
    axes[1].axvline(retval, color="red", linewidth=2.5,
                     label=f"Otsu threshold = {retval:.0f}")
    for pk in peaks:
        axes[1].axvline(pk, color="green", linewidth=1, linestyle="--", alpha=0.7)
    axes[1].set_title(f"Histogram — {n_modes} mode(s) detected")
    axes[1].set_xlabel("Intensity"); axes[1].legend()

    plt.tight_layout(); plt.show()

    if n_modes < 2:
        print(f"WARNING: Histogram has {n_modes} mode(s). "
              f"Otsu works best with bimodal histograms.")
        print("  → Consider: adaptive thresholding, or manual threshold selection.")
    elif n_modes > 2:
        print(f"WARNING: Histogram has {n_modes} modes. "
              f"Otsu may split incorrectly.")
        print("  → Consider: multi-level thresholding (cv2.THRESH_OTSU iteratively), "
              "clustering, or adaptive thresholding.")
    else:
        print(f"Histogram is bimodal — Otsu should work well.")

    return retval, binary


# ── Case 1: Bimodal (Otsu works) ──────────────────────────────────────────────
img_good = np.zeros((200, 300), dtype=np.uint8)
img_good[:100, :] = np.random.normal(60, 15, (100, 300)).clip(0, 255).astype(np.uint8)
img_good[100:, :] = np.random.normal(190, 15, (100, 300)).clip(0, 255).astype(np.uint8)

# ── Case 2: Non-uniform lighting (Otsu fails) ─────────────────────────────────
img_gradient_bg = np.zeros((200, 300), dtype=np.uint8)
for x in range(300):
    bg_val = int(x / 300 * 180)   # gradient background 0→180
    img_gradient_bg[:, x] = np.random.normal(bg_val, 10, 200).clip(0, 255)
# Add a bright object in the middle
cv2.circle(img_gradient_bg, (150, 100), 50, 220, -1)

diagnose_otsu(img_good, "Bimodal (Otsu should work)")
diagnose_otsu(img_gradient_bg, "Non-uniform lighting (Otsu will fail)")
```

---

## 7.5 Adaptive Thresholding

Adaptive thresholding is the solution to non-uniform illumination. Instead of computing one global threshold, it computes a **local threshold for each pixel** based on the pixel values in a neighborhood around it.

### 7.5.1 How Adaptive Thresholding Works

For each pixel `(x, y)`:
1. Extract a block of size `blockSize × blockSize` centered at `(x, y)`
2. Compute a local statistic T_local of that block (mean or Gaussian-weighted mean)
3. Subtract a constant `C`: `threshold(x, y) = T_local - C`
4. Apply: pixel → 255 if pixel > threshold(x,y), else 0

Because `threshold(x, y)` varies across the image, dark regions get low thresholds and bright regions get high thresholds — automatically compensating for illumination variation.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Create a classic failure case for global thresholding ─────────────────────
# Simulate a document photographed under non-uniform lighting
img_doc = np.zeros((300, 400), dtype=np.uint8)

# Add text (simulated as dark rectangles on lighter background)
for y in range(40, 260, 28):
    cv2.line(img_doc, (30, y), (370, y), 180, 2)   # "text lines"
for y in range(44, 260, 28):
    for x in range(30, 370, 20):
        if np.random.random() > 0.4:
            cv2.rectangle(img_doc, (x, y-4), (x+12, y+2), 40, -1)  # "letters"

# Simulate non-uniform lighting: bright left, dark right
lighting = np.zeros((300, 400), dtype=np.float32)
for x in range(400):
    lighting[:, x] = 120 * np.exp(-((x - 0) / 200) ** 2)   # bright left peak
lighting += np.random.normal(0, 8, (300, 400))
img_doc_lit = np.clip(img_doc.astype(np.float32) + lighting, 0, 255).astype(np.uint8)

# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType,
#                        blockSize, C)
#
# adaptiveMethod:
#   cv2.ADAPTIVE_THRESH_MEAN_C     — threshold = mean(block) - C
#   cv2.ADAPTIVE_THRESH_GAUSSIAN_C — threshold = Gaussian-weighted-mean(block) - C
#
# blockSize: size of neighborhood (must be ODD, ≥ 3)
#   Small (e.g. 11): responds to fine local detail, may include noise
#   Large (e.g. 51): smoother thresholds, misses fine detail
#
# C: constant subtracted from mean (typically 2–15)
#   C > 0: threshold is below local mean → more conservative (less foreground)
#   C < 0: threshold is above local mean → more aggressive (more foreground)
#   C = 0: threshold exactly equals local mean

# Global Otsu (fails on non-uniform lighting)
_, binary_global = cv2.threshold(img_doc_lit, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive mean
binary_mean = cv2.adaptiveThreshold(img_doc_lit, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=21, C=5)

# Adaptive Gaussian (usually better — Gaussian weighting reduces sensitivity to outliers)
binary_gauss = cv2.adaptiveThreshold(img_doc_lit, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      blockSize=21, C=5)

# Adaptive Gaussian with inverted output (for dark text → white background)
binary_gauss_inv = cv2.adaptiveThreshold(img_doc_lit, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV,
                                          blockSize=21, C=5)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, im, title in zip(axes.flat, [
    img_doc, img_doc_lit, binary_global,
    binary_mean, binary_gauss, binary_gauss_inv
], [
    "Original text", "With non-uniform lighting",
    "Global Otsu (fails)",
    "Adaptive Mean (blockSize=21, C=5)",
    "Adaptive Gaussian (blockSize=21, C=5)",
    "Adaptive Gaussian INV (dark text → white)"
]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9); ax.axis("off")
plt.suptitle("Global vs Adaptive Thresholding on Non-Uniform Illumination", fontsize=12)
plt.tight_layout(); plt.show()
```

### 7.5.2 Parameter Tuning Guide

Choosing `blockSize` and `C` requires understanding what each parameter controls:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def adaptive_param_grid(img_gray, block_sizes, C_values):
    """
    Visualize adaptive thresholding over a parameter grid.
    Shows blockSize × C_value combinations.
    """
    n_rows = len(block_sizes)
    n_cols = len(C_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    for i, bs in enumerate(block_sizes):
        for j, c in enumerate(C_values):
            result = cv2.adaptiveThreshold(
                img_gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=bs, C=c
            )
            ax = axes[i][j] if n_rows > 1 else axes[j]
            ax.imshow(result, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"bs={bs}, C={c}", fontsize=8)
            ax.axis("off")

    plt.suptitle("Adaptive Gaussian: blockSize × C parameter grid", fontsize=12)
    plt.tight_layout(); plt.show()


img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((200, 300), dtype=np.uint8)
    for y in range(30, 180, 25):
        cv2.line(img, (20, y), (280, y), 170, 2)
    cv2.circle(img, (150, 100), 60, 30, -1)
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# Preprocess with median blur to reduce noise effect on local thresholds
img_prep = cv2.medianBlur(img, 3)

adaptive_param_grid(
    img_prep,
    block_sizes=[11, 21, 51],
    C_values=[2, 5, 10, 15]
)
```

**Parameter interpretation:**

| Parameter | Effect of increasing | Typical range |
|-----------|---------------------|---------------|
| `blockSize` | Broader neighborhood → smoother thresholds, less sensitive to local texture | 11–51 (must be odd) |
| `C` | Higher C → threshold moves further below local mean → less foreground (thinner/fewer blobs) | 2–15 |
| `C < 0` | Threshold above local mean → more aggressive foreground extraction | Rare, use cautiously |

**Rule of thumb:** Start with `blockSize=11, C=5` (Gaussian). Increase `blockSize` if you see too much noise. Increase `C` if background is bleeding into foreground. Decrease `C` if genuine foreground is disappearing.

### 7.5.3 Adaptive Thresholding Internals

Understanding how adaptive thresholding computes local thresholds helps you debug and tune it:

```python
import cv2
import numpy as np


def adaptive_threshold_manual(img_gray, block_size, C,
                               method="gaussian"):
    """
    Manual implementation of adaptive thresholding.
    Shows exactly what cv2.adaptiveThreshold does internally.
    """
    assert block_size % 2 == 1, "blockSize must be odd"
    pad = block_size // 2
    h, w = img_gray.shape
    img_f = img_gray.astype(np.float32)

    if method == "mean":
        # Local mean via box blur
        local_mean = cv2.blur(img_f, (block_size, block_size))
    else:
        # Local Gaussian-weighted mean
        # OpenCV uses sigma = (block_size - 1) / 6 for the Gaussian
        sigma = (block_size - 1) / 6.0
        local_mean = cv2.GaussianBlur(img_f, (block_size, block_size), sigma)

    # Threshold at each pixel
    threshold_map = local_mean - C
    binary = (img_f > threshold_map).astype(np.uint8) * 255

    return binary, threshold_map


img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(50, 200, (100, 150), dtype=np.uint8)

manual_result, thresh_map = adaptive_threshold_manual(img, 21, 5, "gaussian")
cv2_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 5)

diff = np.abs(manual_result.astype(int) - cv2_result.astype(int))
print(f"Manual vs cv2 max difference: {diff.max()} px")
# Should be 0 (exact match for MEAN) or small for Gaussian (floating-point differences)

# Visualize the threshold map — this shows what "local threshold" each pixel uses
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original"); axes[0].axis("off")

im = axes[1].imshow(thresh_map, cmap="hot")
axes[1].set_title("Local threshold map\n(varies per pixel)")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1])

axes[2].imshow(cv2_result, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Adaptive threshold result"); axes[2].axis("off")
plt.tight_layout(); plt.show()
```

---

## 7.6 Multi-Level Thresholding

### 7.6.1 Combining Thresholds with Bitwise Operations

For images with more than two meaningful intensity ranges, combine multiple thresholds:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a 3-region image: dark (0-85), medium (86-170), bright (171-255)
img_multi = np.zeros((200, 300), dtype=np.uint8)
img_multi[:, :100]  = np.random.normal(50,  15, (200, 100)).clip(0, 255)
img_multi[:, 100:200] = np.random.normal(128, 15, (200, 100)).clip(0, 255)
img_multi[:, 200:]  = np.random.normal(210, 15, (200, 100)).clip(0, 255)

# Extract each region using threshold bands
_, mask_dark   = cv2.threshold(img_multi, 85,  255, cv2.THRESH_BINARY_INV)  # pix ≤ 85
_, mask_bright = cv2.threshold(img_multi, 170, 255, cv2.THRESH_BINARY)      # pix > 170

# Medium: NOT dark AND NOT bright
mask_medium_raw = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_bright))
_, mask_above_85 = cv2.threshold(img_multi, 85, 255, cv2.THRESH_BINARY)
_, mask_below_170 = cv2.threshold(img_multi, 170, 255, cv2.THRESH_BINARY_INV)
mask_medium = cv2.bitwise_and(mask_above_85, mask_below_170)

# Visualize: color-code the three regions
colored = np.zeros((*img_multi.shape, 3), dtype=np.uint8)
colored[mask_dark   > 0] = [255,  50,  50]   # red: dark
colored[mask_medium > 0] = [ 50, 255,  50]   # green: medium
colored[mask_bright > 0] = [ 50,  50, 255]   # blue: bright

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, im, title in zip(axes,
    [img_multi, mask_dark, mask_medium, colored],
    ["Input", "Dark mask (≤85)", "Medium mask (86-170)", "3-region color map"]):
    ax.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
              cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
    ax.set_title(title, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
```

### 7.6.2 Multi-Otsu Thresholding

For images with multiple classes, Otsu's method can be extended to find multiple thresholds. OpenCV does not have a built-in multi-Otsu, but we can implement it or use scikit-image:

```python
import numpy as np
import cv2


def multi_otsu(img_gray, n_thresholds=2):
    """
    Multi-level Otsu's method: find n_thresholds that maximally separate
    (n_thresholds + 1) intensity classes.

    For n_thresholds=1, equivalent to standard Otsu.
    For n_thresholds=2, finds two thresholds separating 3 classes.

    This is a brute-force implementation — use scikit-image for production.
    """
    assert 1 <= n_thresholds <= 3, "Support 1-3 thresholds"

    n_pixels = img_gray.size
    hist = np.bincount(img_gray.ravel(), minlength=256) / n_pixels

    best_var  = -1
    best_thresholds = None

    if n_thresholds == 1:
        # Standard Otsu
        t, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [int(t)]

    elif n_thresholds == 2:
        # Two thresholds → three classes
        # σ²_b = w0*(μ0-μg)² + w1*(μ1-μg)² + w2*(μ2-μg)²
        mu_global = sum(i * hist[i] for i in range(256))

        for t1 in range(1, 254):
            for t2 in range(t1 + 1, 255):
                # Class 0: [0, t1], Class 1: (t1, t2], Class 2: (t2, 255]
                w0 = hist[:t1+1].sum()
                w1 = hist[t1+1:t2+1].sum()
                w2 = hist[t2+1:].sum()

                if w0 < 1e-10 or w1 < 1e-10 or w2 < 1e-10:
                    continue

                # Class means
                mu0 = sum(i * hist[i] for i in range(0, t1+1)) / w0
                mu1 = sum(i * hist[i] for i in range(t1+1, t2+1)) / w1
                mu2 = sum(i * hist[i] for i in range(t2+1, 256)) / w2

                var_b = (w0 * (mu0 - mu_global)**2 +
                         w1 * (mu1 - mu_global)**2 +
                         w2 * (mu2 - mu_global)**2)

                if var_b > best_var:
                    best_var = var_b
                    best_thresholds = [t1, t2]

        return best_thresholds


# Faster with scikit-image (recommended for production):
def multi_otsu_skimage(img_gray, n_classes=3):
    """Use scikit-image's optimized multi-Otsu (much faster than manual)."""
    try:
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(img_gray, classes=n_classes)
        return list(thresholds.astype(int))
    except ImportError:
        print("scikit-image not available. Install: pip install scikit-image")
        return None


img_test = np.zeros((200, 300), dtype=np.uint8)
img_test[:, :100]  = np.random.normal(50,  10, (200, 100)).clip(0, 255)
img_test[:, 100:200] = np.random.normal(128, 10, (200, 100)).clip(0, 255)
img_test[:, 200:]  = np.random.normal(210, 10, (200, 100)).clip(0, 255)

thresholds = multi_otsu(img_test, n_thresholds=2)
if thresholds:
    print(f"Multi-Otsu thresholds (2): {thresholds}")
    # Apply to segment
    labels = np.digitize(img_test, bins=thresholds)   # 0, 1, 2
    segmented = (labels * 127).astype(np.uint8)        # 0→0, 1→127, 2→254
```

---

## 7.7 Thresholding in Color Images

`cv2.threshold` and `cv2.adaptiveThreshold` operate on single-channel images only. For color images, use one of these strategies:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("sample.jpg")
if img_bgr is None:
    img_bgr = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.circle(img_bgr, (150, 100), 80, (60, 180, 220), -1)

# ── Strategy 1: Convert to grayscale, then threshold ─────────────────────────
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, binary_gray = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ── Strategy 2: Threshold each channel separately, combine with AND ──────────
_, mask_b = cv2.threshold(img_bgr[:, :, 0], 100, 255, cv2.THRESH_BINARY)
_, mask_g = cv2.threshold(img_bgr[:, :, 1], 100, 255, cv2.THRESH_BINARY)
_, mask_r = cv2.threshold(img_bgr[:, :, 2], 100, 255, cv2.THRESH_BINARY)
combined = cv2.bitwise_and(cv2.bitwise_and(mask_b, mask_g), mask_r)

# ── Strategy 3: cv2.inRange — most practical for color segmentation ───────────
# (Revisited from Chapter 3 — the canonical tool for color ranges)
lower = np.array([50, 100, 150], dtype=np.uint8)   # BGR lower bound
upper = np.array([100, 200, 255], dtype=np.uint8)  # BGR upper bound
mask_range = cv2.inRange(img_bgr, lower, upper)   # 0 or 255

# ── Strategy 4: Threshold in HSV for robust color isolation ──────────────────
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
# Isolate bright cyan-ish pixels (from the circle above)
lower_hsv = np.array([85, 100, 100], dtype=np.uint8)
upper_hsv = np.array([100, 255, 255], dtype=np.uint8)
mask_hsv   = cv2.inRange(hsv, lower_hsv, upper_hsv)

# ── Strategy 5: Threshold the L channel of LAB (ignores color, only brightness) ─
lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
L   = lab[:, :, 0]   # Perceptual lightness
_, binary_lab = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("Color thresholding strategies:")
print(f"  Grayscale Otsu: {binary_gray.sum()//255} white pixels")
print(f"  Channel AND:    {combined.sum()//255} white pixels")
print(f"  cv2.inRange:    {mask_range.sum()//255} white pixels")
print(f"  HSV range:      {mask_hsv.sum()//255} white pixels")
print(f"  LAB L* Otsu:    {binary_lab.sum()//255} white pixels")
```

---

## 7.8 Complete Project: Document Binarization Pipeline

A production-quality document binarization system handles: varying illumination, noise, shadows, skewed text, and different paper textures. Let us build one step by step:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def assess_image_quality(img_gray: np.ndarray) -> dict:
    """
    Assess image properties to guide thresholding strategy selection.

    Returns a dict with keys:
        'contrast'     : RMS contrast (std of pixel values)
        'bimodality'   : Bimodality coefficient (0=unimodal, 1=perfect bimodal)
        'illumination' : 'uniform' or 'non-uniform' (based on spatial variance of local means)
        'noise_level'  : Estimated noise std
    """
    # Contrast: standard deviation of pixel intensities
    contrast = float(img_gray.std())

    # Bimodality coefficient (Sarle's, 1994)
    # BC = (skewness² + 1) / kurtosis
    # BC > 0.555 suggests bimodal or multimodal distribution
    vals = img_gray.ravel().astype(np.float64)
    mu = vals.mean()
    sigma = vals.std() + 1e-8
    n = len(vals)
    skewness = float(((vals - mu)**3).mean() / sigma**3)
    kurtosis = float(((vals - mu)**4).mean() / sigma**4)
    bc = (skewness**2 + 1) / (kurtosis + 3 * (n-1)**2 / ((n-2)*(n-3)))
    bimodality = min(max(bc, 0), 1)  # clip to [0, 1]

    # Non-uniform illumination: measure spatial variation of local means
    local_mean = cv2.blur(img_gray.astype(np.float32), (51, 51))
    illumination_var = float(local_mean.std())
    illumination = "non-uniform" if illumination_var > 20 else "uniform"

    # Noise estimate: median absolute deviation of high-frequency component
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 1.0)
    hf = img_gray.astype(np.float32) - blurred.astype(np.float32)
    noise_level = float(np.median(np.abs(hf)) * 1.4826)  # MAD-based std estimate

    return {
        "contrast"     : contrast,
        "bimodality"   : bimodality,
        "illumination" : illumination,
        "noise_level"  : noise_level,
    }


def select_threshold_strategy(quality: dict) -> str:
    """
    Select the best thresholding strategy based on image quality assessment.
    """
    if quality["illumination"] == "non-uniform":
        return "adaptive_gaussian"

    if quality["contrast"] < 15:
        return "adaptive_gaussian"   # low contrast → global often fails

    if quality["bimodality"] > 0.555:
        if quality["noise_level"] < 10:
            return "otsu"                    # clean bimodal → Otsu is ideal
        else:
            return "blur_otsu"               # noisy bimodal → pre-blur + Otsu

    return "adaptive_gaussian"               # fallback: adaptive is safest


def binarize_document(img_bgr_or_gray: np.ndarray,
                       strategy: Optional[str] = None,
                       invert: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Binarize a document image using an automatically or manually selected strategy.

    Parameters
    ----------
    img_bgr_or_gray : np.ndarray
        Input BGR or grayscale image.
    strategy : str or None
        Force a specific strategy: 'otsu', 'blur_otsu', 'adaptive_gaussian',
        'adaptive_mean', 'triangle'. If None, auto-select.
    invert : bool
        If True, output dark foreground on white background (for dark text).
        If False, white foreground on black background.

    Returns
    -------
    binary : np.ndarray (H, W) uint8, values 0 and 255
    info   : dict with strategy used, threshold found, quality metrics
    """
    # ── Step 1: Convert to grayscale ──────────────────────────────────────────
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray.copy()

    # ── Step 2: Assess image quality ──────────────────────────────────────────
    quality = assess_image_quality(gray)

    # ── Step 3: Select strategy ───────────────────────────────────────────────
    if strategy is None:
        strategy = select_threshold_strategy(quality)

    # ── Step 4: Preprocessing ────────────────────────────────────────────────
    # Denoise slightly before thresholding (reduces false positives)
    if quality["noise_level"] > 8:
        preprocessed = cv2.medianBlur(gray, 3)
    else:
        preprocessed = gray

    # ── Step 5: Apply thresholding ────────────────────────────────────────────
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold_val = None

    if strategy == "otsu":
        retval, binary = cv2.threshold(preprocessed, 0, 255,
                                        thresh_type + cv2.THRESH_OTSU)
        threshold_val = float(retval)

    elif strategy == "blur_otsu":
        blurred = cv2.GaussianBlur(preprocessed, (5, 5), 0)
        retval, binary = cv2.threshold(blurred, 0, 255,
                                        thresh_type + cv2.THRESH_OTSU)
        threshold_val = float(retval)

    elif strategy == "triangle":
        retval, binary = cv2.threshold(preprocessed, 0, 255,
                                        thresh_type + cv2.THRESH_TRIANGLE)
        threshold_val = float(retval)

    elif strategy == "adaptive_mean":
        binary = cv2.adaptiveThreshold(preprocessed, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        thresh_type, blockSize=25, C=8)

    else:  # adaptive_gaussian (default)
        # Block size heuristic: ~1/20 of image width, odd
        h, w = gray.shape
        block_size_hint = max(11, (w // 20) | 1)  # ensure odd
        binary = cv2.adaptiveThreshold(preprocessed, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        thresh_type,
                                        blockSize=block_size_hint, C=5)

    # ── Step 6: Post-processing (optional cleanup) ────────────────────────────
    # Remove small isolated noise blobs via morphological opening
    if quality["noise_level"] > 5:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    info = {
        "strategy"     : strategy,
        "threshold"    : threshold_val,
        "quality"      : quality,
    }

    return binary, info


# ── Demo: Test on multiple scenarios ──────────────────────────────────────────
def create_test_images():
    """Create four challenging test cases."""
    images = {}

    # 1. Clean document (Otsu works well)
    clean = np.full((200, 300), 240, dtype=np.uint8)
    for y in range(40, 180, 25):
        cv2.line(clean, (20, y), (280, y), 220, 1)
    for y in range(42, 180, 25):
        for x in range(20, 280, 18):
            cv2.rectangle(clean, (x, y-7), (x+10, y), 40, -1)
    images["1_clean_document"] = clean

    # 2. Non-uniform lighting (adaptive wins)
    uneven = clean.copy().astype(np.float32)
    for x in range(300):
        uneven[:, x] += 80 * np.exp(-((x - 280) / 100)**2)  # bright right edge
    images["2_uneven_lighting"] = np.clip(uneven, 0, 255).astype(np.uint8)

    # 3. Noisy document
    noisy = clean.copy()
    noise = np.random.normal(0, 20, clean.shape)
    images["3_noisy_document"] = np.clip(noisy.astype(np.float32) + noise,
                                          0, 255).astype(np.uint8)

    # 4. Low-contrast document (subtle text)
    low_contrast = np.full((200, 300), 180, dtype=np.uint8)
    for y in range(42, 180, 25):
        for x in range(20, 280, 18):
            cv2.rectangle(low_contrast, (x, y-7), (x+10, y), 145, -1)  # small contrast
    images["4_low_contrast"] = low_contrast

    return images

test_images = create_test_images()
fig, axes = plt.subplots(len(test_images), 3, figsize=(12, 14))

for i, (name, img) in enumerate(test_images.items()):
    binary, info = binarize_document(img, invert=True)   # invert for dark text

    # Show original, binary, and quality info
    axes[i][0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[i][0].set_title(f"{name}\nNoise={info['quality']['noise_level']:.1f}", fontsize=8)
    axes[i][0].axis("off")

    axes[i][1].imshow(binary, cmap="gray", vmin=0, vmax=255)
    t = info["threshold"]
    t_str = f"t={t:.0f}" if t else "adaptive"
    axes[i][1].set_title(f"Strategy: {info['strategy']}\n{t_str}", fontsize=8)
    axes[i][1].axis("off")

    # Quality profile
    q = info["quality"]
    axes[i][2].bar(["contrast", "bimodality\n×100", "noise"],
                   [q["contrast"], q["bimodality"]*100, q["noise_level"]],
                   color=["steelblue", "coral", "seagreen"])
    axes[i][2].set_title(f"Illumination: {q['illumination']}", fontsize=8)
    axes[i][2].set_ylim(0, max(100, q["contrast"] + 10))

plt.suptitle("Adaptive Document Binarization Pipeline", fontsize=13, y=1.01)
plt.tight_layout(); plt.show()
```

---

## 7.9 Thresholding Performance and Limitations

### 7.9.1 Speed Comparison

```python
import cv2
import numpy as np
import time

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (1080, 1920), dtype=np.uint8)  # Full HD

methods = {
    "Global (fixed t=127)"    : lambda i: cv2.threshold(i, 127, 255, cv2.THRESH_BINARY),
    "Otsu"                    : lambda i: cv2.threshold(i, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU),
    "Blur+Otsu"               : lambda i: cv2.threshold(cv2.GaussianBlur(i,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU),
    "Adaptive Mean (bs=21)"   : lambda i: cv2.adaptiveThreshold(i, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5),
    "Adaptive Gauss (bs=21)"  : lambda i: cv2.adaptiveThreshold(i, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5),
    "Adaptive Gauss (bs=51)"  : lambda i: cv2.adaptiveThreshold(i, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5),
}

n_runs = 30
print(f"Speed benchmark on {img.shape[1]}×{img.shape[0]} grayscale image ({n_runs} runs):")
print("─" * 60)
base_time = None
for name, func in methods.items():
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(img)
        times.append(time.perf_counter() - t0)
    mean_ms = np.mean(times) * 1000
    if base_time is None:
        base_time = mean_ms
    print(f"  {name:<35}: {mean_ms:6.2f} ms  ({mean_ms/base_time:.1f}×)")
```

### 7.9.2 Decision Guide: Choosing the Right Method

```python
# ════════════════════════════════════════════════════════════════════════════
# THRESHOLDING DECISION GUIDE
# ════════════════════════════════════════════════════════════════════════════

# Q1: Does the image have uniform illumination?
#   YES → Q2
#   NO  → Use ADAPTIVE_GAUSSIAN

# Q2: Is the histogram bimodal (two clear peaks with a valley)?
#   YES → Q3
#   NO  → Use ADAPTIVE_GAUSSIAN (or try TRIANGLE for unimodal with sparse FG)

# Q3: Is there significant noise?
#   YES → Use Gaussian blur (5×5, σ=0) → then OTSU
#   NO  → Use OTSU directly

# Q4: Need real-time performance?
#   YES → Use BINARY with fixed threshold (fastest) or OTSU (near-fastest)
#   NO  → Adaptive Gaussian is best quality for non-uniform lighting

# Q5: Is the foreground dark on a bright background (text, handwriting)?
#   YES → Use THRESH_BINARY_INV (or THRESH_BINARY + bitwise_not result)
#   NO  → Use THRESH_BINARY

# QUICK REFERENCE:
# cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)          # manual, fast
# cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+THRESH_OTSU)# auto global
# cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+THRESH_OTSU)# robust global
# cv2.adaptiveThreshold(img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 5)  # local
```

---

## 7.10 Summary

Thresholding converts a grayscale image to binary by comparing each pixel to a threshold value. The method and parameters must be matched to the image's statistical properties:

**`cv2.threshold` with 5 types:**
- `THRESH_BINARY` / `THRESH_BINARY_INV`: true binary output
- `THRESH_TRUNC`: clips at threshold, not binary
- `THRESH_TOZERO` / `THRESH_TOZERO_INV`: zeros one side, not binary

**Otsu's method (`THRESH_OTSU`):** Automatically finds the threshold that maximizes between-class variance. Works best on bimodal histograms. Pre-blur with Gaussian when image is noisy. Returns the threshold value via `retval`.

**Triangle method (`THRESH_TRIANGLE`):** Better than Otsu for unimodal histograms with a sparse foreground.

**`cv2.adaptiveThreshold`:** Computes a local threshold per pixel based on a neighborhood. The solution for non-uniform illumination. Two methods: `ADAPTIVE_THRESH_MEAN_C` (faster) and `ADAPTIVE_THRESH_GAUSSIAN_C` (better quality). Tune `blockSize` (neighborhood extent) and `C` (offset below local mean).

**Multi-level thresholding:** Combine `cv2.threshold` calls with bitwise AND/OR. Use `cv2.inRange` for color channels. Use `skimage.filters.threshold_multiotsu` for automatic multi-class segmentation.

**Diagnostic workflow:** Check histogram bimodality before applying Otsu. Check illumination uniformity via spatial variance of local means. Let image quality drive strategy selection.

---

## 7.11 Exercises

### Warm-up

**7.1** Create a synthetic 400×600 grayscale image with three horizontal bands of intensities 50, 130, and 210. Apply all five `cv2.threshold` types with `thresh=100` and `thresh=170`. Display all 10 results in a 2×5 grid. Label each panel.

**7.2** Implement the Otsu threshold manually using NumPy (without `cv2.threshold`), following the between-class variance formula from Section 7.4.2. Test it on five different images and compare to `cv2.THRESH_OTSU`. Report the maximum discrepancy across all images.

**7.3** Use `cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_GAUSSIAN_C` and visualize how the output changes as `blockSize` goes from 5 to 101 in steps of 10, and C goes from 0 to 20 in steps of 5. Create a 10×5 parameter grid showing all 50 combinations.

### Core

**7.4** Load a real document image (or generate a synthetic one with simulated shadows). Apply: (a) global Otsu, (b) Gaussian pre-blur then Otsu, (c) adaptive mean, (d) adaptive Gaussian. Compute the fraction of "connected regions" (use `cv2.connectedComponentsWithStats`) for each — a cleaner binarization should have fewer, larger, more coherent components. Rank the methods by this metric.

**7.5** Build an `interactive_threshold_tuner(img)` function that opens an OpenCV window with trackbars for: method selection (0=binary, 1=otsu, 2=adaptive_mean, 3=adaptive_gaussian), threshold value (for binary), blockSize (for adaptive), and C (for adaptive). Show the input and binary output side by side. Display the histogram with the current threshold overlaid.

**7.6** Implement `hysteresis_threshold(img, low, high)` that applies two thresholds: pixels above `high` are definitely foreground; pixels between `low` and `high` are foreground only if connected to a definite-foreground pixel. (This is essentially what Canny edge detection does internally.) Use `cv2.connectedComponentsWithStats` or flood fill for the connectivity step.

**7.7** Apply the `binarize_document` pipeline from Section 7.8 to 5 real images (or 5 synthetic test cases with different challenges). For each, report the strategy auto-selected, the threshold used, and show a visual comparison of the result vs the "wrong" strategy applied to the same image.

### Challenge

**7.8** Implement a complete **receipt/invoice scanner** that: (a) detects the document corners using edge detection and contour approximation, (b) applies `four_point_transform` from Chapter 5 to perspective-correct the document, (c) binarizes using the `binarize_document` pipeline, (d) runs `pytesseract.image_to_string` on the result. Report the OCR accuracy on a test receipt.

**7.9** Implement **locally adaptive Otsu**: divide the image into overlapping tiles (e.g., 64×64 with 32-pixel stride), apply Otsu's method to each tile to get a local optimal threshold, and bilinearly interpolate between tile thresholds to get a smooth threshold map. Apply this per-pixel varying threshold to binarize the image. Compare to `cv2.adaptiveThreshold` on a document with non-uniform lighting.

**7.10** **Histogram backprojection for background separation**: compute the histogram of a manually selected background region (a small ROI). Use `cv2.calcBackProject` to create a probability map where each pixel is assigned the probability that it belongs to the background. Threshold this probability map to separate foreground from background. Compare to Otsu on the same image and evaluate on a dataset of 10 images.

---

## Further Reading

- **OpenCV thresholding tutorial:** https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- **Otsu, N. (1979). "A threshold selection method from gray-level histograms."** IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66. — the original paper
- **Otsu's method, Wikipedia:** https://en.wikipedia.org/wiki/Otsu%27s_method — clear mathematical summary
- **scikit-image thresholding:** https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html — broader set of algorithms including multi-Otsu, Li, minimum, and more
- **PyImageSearch adaptive thresholding:** https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
