# Chapter 9: Edge Detection & Gradients

> *"An edge is where the world decides to change its mind. Your job as a CV engineer is to find those decisions reliably — even when the image is noisy, blurry, or poorly lit."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain image gradients mathematically as discrete approximations of derivatives
2. Apply Sobel, Scharr, and Prewitt operators and understand their directional sensitivity
3. Compute gradient magnitude and direction maps from component gradients
4. Apply the Laplacian operator and explain zero-crossing edge detection
5. Explain every stage of the Canny algorithm: smoothing, gradient computation, NMS, hysteresis
6. Tune Canny thresholds intelligently using histogram percentiles
7. Compute the Laplacian of Gaussian (LoG) and relate it to scale-space edge detection
8. Detect edges in color images using multi-channel gradient fusion
9. Build a complete, auto-tuned edge detection pipeline
10. Compare all detectors and choose the right one for each application

---

## 9.1 What Is an Edge?

An **edge** is a location in an image where intensity changes rapidly over a short distance. Edges carry most of the semantically meaningful information in an image — object boundaries, surface discontinuities, depth changes, and lighting transitions all manifest as intensity gradients.

The fundamental mathematical definition: a 1D intensity function `I(x)` has an edge at position `x₀` if `|dI/dx|` is locally maximal and exceeds a threshold.

```python
import numpy as np
import matplotlib.pyplot as plt

# Illustrate what an edge looks like in 1D
x = np.linspace(0, 100, 500)

# Step edge (ideal)
step = np.where(x < 50, 50.0, 200.0)

# Ramp edge (realistic — limited by camera optics)
ramp = np.where(x < 42, 50.0,
        np.where(x < 58, 50 + (x - 42) / 16 * 150, 200.0))

# Noisy ramp (what we actually see)
rng = np.random.default_rng(42)
noisy_ramp = ramp + rng.normal(0, 8, ramp.shape)

# First derivative (gradient) of each signal
dx = x[1] - x[0]
d_step  = np.gradient(step,       dx)
d_ramp  = np.gradient(ramp,       dx)
d_noisy = np.gradient(noisy_ramp, dx)

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, signal, deriv, title in zip(
    axes[0],
    [step, ramp, noisy_ramp],
    [d_step, d_ramp, d_noisy],
    ["Step edge (ideal)", "Ramp edge (realistic)", "Noisy ramp (real-world)"]
):
    ax.plot(x, signal, 'b-', linewidth=2)
    ax.set_title(title); ax.set_xlabel("Position x"); ax.set_ylabel("Intensity")
    ax.grid(alpha=0.3)

for ax, deriv, title in zip(
    axes[1],
    [d_step, d_ramp, d_noisy],
    ["d(step)/dx", "d(ramp)/dx", "d(noisy)/dx"]
):
    ax.plot(x, deriv, 'r-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(title); ax.set_xlabel("Position x"); ax.set_ylabel("Gradient")
    ax.grid(alpha=0.3)

plt.suptitle("Edge = Peak of Gradient Magnitude", fontsize=13)
plt.tight_layout(); plt.show()
```

**Key insight from this plot:**
- The step edge has an infinitely sharp gradient spike → unrealistic
- The ramp edge has a broad gradient hump → edge location is the hump center
- The noisy ramp has a jagged gradient → **smoothing before differentiation is essential**

This motivates every edge detector: smooth first, then differentiate.

---

## 9.2 Discrete Derivatives: Finite Differences

Digital images are sampled at integer positions. We must approximate derivatives using **finite differences**:

```
Forward difference:  dI/dx ≈ I[y, x+1] - I[y, x]
Backward difference: dI/dx ≈ I[y, x] - I[y, x-1]
Central difference:  dI/dx ≈ (I[y, x+1] - I[y, x-1]) / 2
```

As a convolution kernel, the central difference is: `[-1, 0, 1]` (along x) and `[-1, 0, 1]ᵀ` (along y).

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(img, (150, 100), 70, 200, -1)
    cv2.rectangle(img, (30, 130), (120, 180), 150, -1)

img_f = img.astype(np.float32)

# Central difference kernels (1×3 horizontal, 3×1 vertical)
K_dx = np.array([[-1, 0, 1]], dtype=np.float32)    # ∂I/∂x
K_dy = np.array([[-1], [0], [1]], dtype=np.float32) # ∂I/∂y

Gx = cv2.filter2D(img_f, -1, K_dx)  # horizontal gradient
Gy = cv2.filter2D(img_f, -1, K_dy)  # vertical gradient

# Gradient magnitude: G = sqrt(Gx² + Gy²)
G_mag = np.sqrt(Gx**2 + Gy**2)

# Gradient direction: θ = arctan2(Gy, Gx)
G_dir = np.arctan2(Gy, Gx)   # range: [-π, π]
G_dir_deg = np.degrees(G_dir) % 360   # [0, 360]

print(f"Gradient magnitude range: [{G_mag.min():.1f}, {G_mag.max():.1f}]")
print(f"Note: magnitude range exceeds 255 — use float32, not uint8!")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, im, title in zip(axes.flat, [
    img,
    np.clip(Gx + 128, 0, 255).astype(np.uint8),
    np.clip(Gy + 128, 0, 255).astype(np.uint8),
    np.clip(G_mag / G_mag.max() * 255, 0, 255).astype(np.uint8),
    G_dir_deg.astype(np.uint8),
    np.abs(Gx).astype(np.uint8)
], ["Original", "Gx (horizontal grad)", "Gy (vertical grad)",
    "Magnitude |G|", "Direction θ (mapped)", "|Gx| only"]):
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 9.3 Sobel Operator

The Sobel operator is a gradient estimator that incorporates Gaussian smoothing. The 3×3 Sobel kernels are:

```
       ∂/∂x (Gx)          ∂/∂y (Gy)
┌─────────────────┐    ┌─────────────────┐
│ -1   0  +1 │    │ -1  -2  -1 │
│ -2   0  +2 │    │  0   0   0 │
│ -1   0  +1 │    │ +1  +2  +1 │
└─────────────────┘    └─────────────────┘
```

Each Sobel kernel is separable: Gx = `[1, 2, 1]ᵀ × [-1, 0, 1]` and Gy = `[-1, 0, 1]ᵀ × [1, 2, 1]`. The `[1, 2, 1]` factor is a binomial (Gaussian-like) smoothing in the perpendicular direction.

### 9.3.1 cv2.Sobel

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(img, (150, 100), 70, 200, -1)
    cv2.rectangle(img, (30, 130), (120, 180), 150, -1)
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# cv2.Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0, borderType)
# ddepth: output depth — MUST be signed float to capture negative gradients
#   cv2.CV_16S: 16-bit signed integer
#   cv2.CV_32F: 32-bit float (preferred)
#   cv2.CV_64F: 64-bit float (highest precision)
# dx: order of x derivative (0 or 1)
# dy: order of y derivative (0 or 1)
# ksize: 1, 3, 5, or 7 (1 = Scharr optimal filter)

# First derivatives (gradient)
Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)  # ∂I/∂x
Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)  # ∂I/∂y

# WARNING: if you use uint8 output, negative values wrap to 255
# Always use float output for Sobel
Gx_wrong = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)  # WRONG — wraps negatives!
Gx_right = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) # CORRECT

print(f"Gx (uint8) min: {Gx_wrong.min()}, max: {Gx_wrong.max()}")
print(f"Gx (float) min: {Gx_right.min():.1f}, max: {Gx_right.max():.1f}")

# Gradient magnitude (L2 norm)
G_L2 = np.sqrt(Gx**2 + Gy**2)

# Gradient magnitude (L1 approximation — faster, slightly less accurate)
G_L1 = np.abs(Gx) + np.abs(Gy)

# Gradient direction (angle perpendicular to edge)
G_dir = np.arctan2(Gy, Gx)

# Visualize — convert to uint8 for display
def norm_to_uint8(arr):
    """Normalize a float array to [0, 255] uint8 for display."""
    a_min, a_max = arr.min(), arr.max()
    if a_max == a_min:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - a_min) / (a_max - a_min) * 255).astype(np.uint8)

# A common pattern: absolute value then convert
Gx_vis = cv2.convertScaleAbs(Gx)  # abs() then saturate to uint8
Gy_vis = cv2.convertScaleAbs(Gy)
G_L1_vis = cv2.convertScaleAbs(G_L1)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, im, title in zip(axes.flat,
    [img, Gx_vis, Gy_vis, norm_to_uint8(G_L2), norm_to_uint8(np.abs(G_dir)),
     norm_to_uint8(G_L2 - G_L1.max() * 0 + G_L1 * 0)],  # reuse
    ["Original", "|Gx| (∂/∂x)", "|Gy| (∂/∂y)",
     "Magnitude L2 (√Gx²+Gy²)", "Direction θ", "|Gx| + |Gy| (L1 approx)"]):
    pass  # placeholder

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, im, title in zip(axes.flat,
    [img, Gx_vis, Gy_vis, norm_to_uint8(G_L2), G_L1_vis, norm_to_uint8(G_dir + np.pi)],
    ["Original", "|Gx| (∂I/∂x)", "|Gy| (∂I/∂y)",
     "Magnitude L2", "Magnitude L1", "Direction θ"]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

### 9.3.2 Kernel Size and the Scale Tradeoff

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(img, (150, 100), 70, 200, -1)

# ksize=1 in cv2.Sobel uses the optimized Scharr filter (ksize must be 1, 3, 5, or 7)
# ksize=3: standard 3×3 Sobel
# ksize=5: 5×5 Sobel (more smoothing, captures coarser edges)
# ksize=7: 7×7 Sobel (even more smoothing)

results = {}
for ksize in [1, 3, 5, 7]:
    Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(Gx**2 + Gy**2)
    results[ksize] = mag

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
for ax, (ksize, mag) in zip(axes[1:], results.items()):
    label = "Scharr\n(ksize=1)" if ksize == 1 else f"Sobel\n(ksize={ksize})"
    ax.imshow(norm_to_uint8(mag), cmap="gray")
    ax.set_title(f"{label}\nFine edges" if ksize <= 3 else f"{label}\nCoarse edges")
    ax.axis("off")
plt.suptitle("Sobel Gradient Magnitude: Effect of Kernel Size", fontsize=12)
plt.tight_layout(); plt.show()

# Tradeoff:
# Small ksize (1, 3): sensitive to fine detail, also sensitive to noise
# Large ksize (5, 7): smoother response, captures only coarser edges
```

### 9.3.3 Scharr Operator: Optimized 3×3 Gradient

The Scharr operator (`ksize=1` in `cv2.Sobel`, or `cv2.Scharr`) is an optimized 3×3 gradient filter that minimizes rotational asymmetry error compared to standard Sobel:

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(img, (150, 100), 70, 200, -1)

# Scharr kernel (optimal 3×3 gradient filter):
# Gx:                  Gy:
# [ -3   0   3]        [ -3  -10  -3]
# [-10   0  10]        [  0    0   0]
# [ -3   0   3]        [  3   10   3]

# Two equivalent ways to call Scharr:
Scharr_x_v1 = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=cv2.FILTER_SCHARR)
Scharr_y_v1 = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=cv2.FILTER_SCHARR)

Scharr_x_v2 = cv2.Scharr(img, cv2.CV_32F, 1, 0)
Scharr_y_v2 = cv2.Scharr(img, cv2.CV_32F, 0, 1)

print(f"Same result: {np.allclose(Scharr_x_v1, Scharr_x_v2)}")   # True

Sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)

# Scharr has better rotational symmetry than Sobel at 3×3
# Use Scharr when you need accurate gradient direction,
# especially for feature matching and corner detection
```

---

## 9.4 Laplacian: Second Derivative Edge Detection

The Laplacian operator computes the **second derivative** (∂²I/∂x² + ∂²I/∂y²). Edges appear as **zero-crossings** of the Laplacian.

The discrete 3×3 Laplacian kernel:
```
┌─────────────┐
│  0   1   0 │
│  1  -4   1 │
│  0   1   0 │
└─────────────┘
```
(or the 8-connected version with all 1s and center = -8)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(img, (150, 100), 70, 200, -1)
    cv2.rectangle(img, (30, 130), (120, 180), 150, -1)

# cv2.Laplacian(src, ddepth, ksize=1, scale=1, delta=0, borderType)
# ksize: aperture size for Sobel operator (must be odd; ksize=1 uses the standard 3×3 Laplacian)
# ALWAYS use float output to capture negative values

# ── Basic Laplacian ────────────────────────────────────────────────────────────
laplacian = cv2.Laplacian(img, cv2.CV_32F, ksize=1)
laplacian_k3 = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
laplacian_k5 = cv2.Laplacian(img, cv2.CV_32F, ksize=5)

print(f"Laplacian ksize=1 range: [{laplacian.min():.0f}, {laplacian.max():.0f}]")
# Note: includes negative values — zero-crossings mark edges

# ── Absolute Laplacian for display ───────────────────────────────────────────
lap_abs    = cv2.convertScaleAbs(laplacian)
lap_k3_abs = cv2.convertScaleAbs(laplacian_k3)
lap_k5_abs = cv2.convertScaleAbs(laplacian_k5)

# ── Laplacian is noise-sensitive: pre-blur first ──────────────────────────────
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
lap_blurred = cv2.Laplacian(blurred, cv2.CV_32F, ksize=1)
lap_blurred_abs = cv2.convertScaleAbs(lap_blurred)

# ── Zero-crossing detection: where sign changes = edges ────────────────────────
def zero_crossings(lap_float, threshold=10.0):
    """
    Detect zero-crossings of the Laplacian.
    A zero-crossing occurs where the sign changes and the magnitude is large enough.
    """
    h, w = lap_float.shape
    zc = np.zeros((h, w), dtype=np.uint8)

    # For each pixel, check if it has a neighbor of opposite sign
    lap_pos = lap_float > threshold    # positive regions
    lap_neg = lap_float < -threshold   # negative regions

    # Erode each and check if they touch
    se = np.ones((3, 3), dtype=np.uint8)
    dilated_pos = cv2.dilate(lap_pos.astype(np.uint8), se)
    dilated_neg = cv2.dilate(lap_neg.astype(np.uint8), se)

    # Zero-crossings: where positive and negative dilations overlap
    zc = cv2.bitwise_and(dilated_pos, dilated_neg)
    return zc * 255

zc_map = zero_crossings(laplacian, threshold=15)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, im, title in zip(axes.flat,
    [img, lap_abs, lap_k3_abs,
     lap_k5_abs, lap_blurred_abs, zc_map],
    ["Original", "Laplacian ksize=1", "Laplacian ksize=3",
     "Laplacian ksize=5", "Blurred then Laplacian", "Zero-crossings"]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title); ax.axis("off")
plt.suptitle("Laplacian Edge Detection", fontsize=12)
plt.tight_layout(); plt.show()
```

### 9.4.1 Comparing Sobel and Laplacian

```python
# Key differences:
#
# Sobel (first derivative):
#   ✓ Directional — separate Gx and Gy give gradient direction
#   ✓ Thick edges — good for visualization
#   ✓ Stable — less sensitive to noise than Laplacian
#   ✗ Not isotropic for 3×3 (Scharr improves this)
#   ✗ Edges are thick (need NMS or Canny for thin edges)
#
# Laplacian (second derivative):
#   ✓ Isotropic — single kernel for all directions
#   ✓ Zero-crossings give precise edge locations
#   ✗ Very sensitive to noise (second derivative amplifies noise)
#   ✗ Double-response at edges (positive peak, zero-crossing, negative peak)
#   ✗ Cannot give edge direction
#
# In practice: Canny (which uses Sobel internally + NMS + hysteresis)
# is almost always preferred over raw Sobel or Laplacian.
```

---

## 9.5 Laplacian of Gaussian (LoG)

The **Laplacian of Gaussian** (LoG) combines Gaussian smoothing with the Laplacian. It is equivalent to applying a Gaussian blur first (to suppress noise) and then the Laplacian:

```
LoG(σ, x, y) = ∇²G(σ, x, y)
             = (1/πσ⁴) · (r² - 2σ²) / (2σ²) · exp(-r² / 2σ²)
```

where `r² = x² + y²`. This is often called the "Mexican hat" function because of its shape.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace


def log_kernel(ksize, sigma):
    """
    Compute the Laplacian of Gaussian kernel analytically.
    """
    k = ksize // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    r_sq = x**2 + y**2
    kernel = -(1 / (np.pi * sigma**4)) * (1 - r_sq / (2 * sigma**2)) * np.exp(-r_sq / (2 * sigma**2))
    # Normalize so sum = 0 (center negative, surround positive)
    kernel -= kernel.mean()
    return kernel.astype(np.float32)


img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(img, (150, 100), 70, 200, -1)
    cv2.rectangle(img, (30, 130), (120, 180), 150, -1)
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# Method 1: blur then Laplacian
def log_blur_then_laplacian(img, sigma):
    ksize = int(6 * sigma) | 1   # make odd
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return cv2.Laplacian(blurred, cv2.CV_32F, ksize=1)

# Method 2: apply LoG kernel directly
def log_direct(img, sigma):
    ksize = int(6 * sigma) | 1
    kernel = log_kernel(ksize, sigma)
    return cv2.filter2D(img.astype(np.float32), -1, kernel)

# Compare at multiple scales
sigmas = [1.0, 2.0, 4.0]

fig, axes = plt.subplots(len(sigmas), 3, figsize=(12, 10))
for i, sigma in enumerate(sigmas):
    log1 = log_blur_then_laplacian(img, sigma)
    log2 = log_direct(img, sigma)

    axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[i, 0].set_title(f"Original (σ={sigma})"); axes[i, 0].axis("off")

    axes[i, 1].imshow(cv2.convertScaleAbs(log1), cmap="gray")
    axes[i, 1].set_title(f"Blur→Laplacian σ={sigma}"); axes[i, 1].axis("off")

    axes[i, 2].imshow(cv2.convertScaleAbs(log2), cmap="gray")
    axes[i, 2].set_title(f"Direct LoG σ={sigma}"); axes[i, 2].axis("off")

plt.suptitle("Laplacian of Gaussian at Multiple Scales", fontsize=12)
plt.tight_layout(); plt.show()

# ── LoG and scale space ──────────────────────────────────────────────────────
# LoG at scale σ detects blobs of radius r ≈ √2 σ
# Large σ → detects large blobs, ignores fine detail
# Small σ → detects fine edges, sensitive to noise
# This is the foundation of SIFT (Difference of Gaussians ≈ LoG)
```

---

## 9.6 Canny Edge Detector: The Gold Standard

The Canny edge detector (John Canny, 1986) is the most widely used edge detection algorithm in practice. It produces thin, well-localized, connected edges. Understanding its stages is essential for tuning it correctly.

### 9.6.1 The Four Stages of Canny

```
Stage 1: Gaussian Smoothing
    Suppress noise before computing gradients.
    σ controls the scale of edges to detect.

Stage 2: Gradient Computation (Sobel)
    Compute Gx and Gy using Sobel operators.
    Compute magnitude G = √(Gx²+Gy²) and direction θ.

Stage 3: Non-Maximum Suppression (NMS)
    For each pixel, check if it is a LOCAL MAXIMUM in the gradient direction.
    If not → suppress to 0 (produces thin, 1-pixel-wide edges).
    This is the step that converts thick gradient ridges to thin edges.

Stage 4: Double Threshold (Hysteresis)
    Apply two thresholds T_low and T_high:
    - G > T_high  → "strong" edge: always included
    - T_low < G ≤ T_high → "weak" edge: included only if connected to a strong edge
    - G ≤ T_low   → not an edge: discarded
    This step connects edges while rejecting isolated noise.
```

### 9.6.2 cv2.Canny

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(img, (150, 100), 70, 200, -1)
    cv2.rectangle(img, (30, 130), (120, 180), 150, -1)
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
# image:      grayscale uint8 input (Canny does its own internal Gaussian blur)
# threshold1: lower hysteresis threshold
# threshold2: upper hysteresis threshold
# apertureSize: Sobel kernel size (3, 5, or 7)
# L2gradient:  True = L2 gradient magnitude (more accurate), False = L1 (faster)
#
# Rule of thumb: threshold2 ≈ 2×threshold1 to 3×threshold1

# Effect of threshold values
threshold_pairs = [
    (50,  100),   # loose: more edges, more noise
    (100, 200),   # moderate
    (200, 400),   # strict: only strong edges
    (50,  250),   # wide gap: few edges, well-connected
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes.flat[0].imshow(img, cmap="gray"); axes.flat[0].set_title("Original"); axes.flat[0].axis("off")

for ax, (t1, t2) in zip(axes.flat[1:], threshold_pairs):
    edges = cv2.Canny(img, t1, t2)
    edge_pct = edges.sum() // 255 / edges.size * 100
    ax.imshow(edges, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"T_low={t1}, T_high={t2}\n({edge_pct:.1f}% edge pixels)")
    ax.axis("off")

axes.flat[5].axis("off")
plt.suptitle("Canny Threshold Effect", fontsize=12)
plt.tight_layout(); plt.show()

# L2gradient comparison
edges_L1 = cv2.Canny(img, 100, 200, L2gradient=False)
edges_L2 = cv2.Canny(img, 100, 200, L2gradient=True)
diff = np.abs(edges_L1.astype(int) - edges_L2.astype(int))
print(f"L1 vs L2 gradient: {(diff > 0).sum()} pixels differ "
      f"({(diff > 0).mean() * 100:.1f}% of edge pixels)")
```

### 9.6.3 Pre-smoothing and Scale

Canny does not apply Gaussian smoothing internally — you must pre-blur:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(img, (200, 150), 100, 200, -1)
    cv2.rectangle(img, (30, 200), (150, 280), 150, -1)
    noise = np.random.normal(0, 15, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# Effect of pre-blur on Canny
blur_configs = [
    ("No blur", img),
    ("σ=0.5 (k=3)", cv2.GaussianBlur(img, (3, 3), 0.5)),
    ("σ=1.5 (k=9)", cv2.GaussianBlur(img, (9, 9), 1.5)),
    ("σ=3.0 (k=19)", cv2.GaussianBlur(img, (19, 19), 3.0)),
    ("σ=5.0 (k=31)", cv2.GaussianBlur(img, (31, 31), 5.0)),
]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, (label, blurred) in zip(axes, blur_configs):
    edges = cv2.Canny(blurred, 50, 150)
    n_edge = edges.sum() // 255
    ax.imshow(edges, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{label}\n{n_edge} edge px")
    ax.axis("off")
plt.suptitle("Effect of Pre-Smoothing on Canny (T_low=50, T_high=150)", fontsize=12)
plt.tight_layout(); plt.show()

# Insight:
# No blur: many noise edges, fragmented
# σ=0.5–1.5: fine edges detected, manageable noise
# σ=3.0+: only major edges, fine detail lost
# Choose σ based on the scale of edges you care about
```

### 9.6.4 Automatic Threshold Selection for Canny

Manual threshold selection is tedious and brittle across different images. Two approaches for automation:

```python
import cv2
import numpy as np


def canny_auto_sigma(img_gray, sigma=0.33, pre_blur_sigma=1.0):
    """
    Canny with automatically determined thresholds.

    Strategy: thresholds are set relative to the MEDIAN pixel intensity.
    σ=0.33 means: T_low = median*(1-0.33), T_high = median*(1+0.33)

    Works well when the image has a roughly symmetric intensity distribution.
    From: Adrian Rosebrock, PyImageSearch.
    """
    # Optional pre-blur
    if pre_blur_sigma > 0:
        ksize = int(6 * pre_blur_sigma) | 1
        img_proc = cv2.GaussianBlur(img_gray, (ksize, ksize), pre_blur_sigma)
    else:
        img_proc = img_gray

    median_val = np.median(img_proc)
    T_low  = int(max(0,   (1.0 - sigma) * median_val))
    T_high = int(min(255, (1.0 + sigma) * median_val))

    edges = cv2.Canny(img_proc, T_low, T_high)
    return edges, T_low, T_high


def canny_auto_percentile(img_gray, low_pct=10, high_pct=90, pre_blur_sigma=1.0):
    """
    Canny with percentile-based thresholds.

    Thresholds are set at given percentiles of the gradient magnitude distribution.
    More robust than the median method for images with extreme contrast.
    """
    if pre_blur_sigma > 0:
        ksize = int(6 * pre_blur_sigma) | 1
        img_proc = cv2.GaussianBlur(img_gray, (ksize, ksize), pre_blur_sigma)
    else:
        img_proc = img_gray

    # Compute gradient magnitude
    Gx = cv2.Sobel(img_proc, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img_proc, cv2.CV_32F, 0, 1, ksize=3)
    G  = np.sqrt(Gx**2 + Gy**2)

    T_low  = float(np.percentile(G, low_pct))
    T_high = float(np.percentile(G, high_pct))

    edges = cv2.Canny(img_proc, T_low, T_high)
    return edges, T_low, T_high


# ── Test both methods ─────────────────────────────────────────────────────────
img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(img, (200, 150), 100, 200, -1)
    cv2.rectangle(img, (30, 200), (150, 280), 150, -1)
    noise = np.random.normal(0, 12, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

edges_sigma, T1, T2 = canny_auto_sigma(img)
edges_pct, Tp1, Tp2 = canny_auto_percentile(img, low_pct=15, high_pct=85)
edges_manual = cv2.Canny(cv2.GaussianBlur(img, (9,9), 1.5), 50, 150)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, im, title in zip(axes, [
    img, edges_manual, edges_sigma, edges_pct
], [
    "Input",
    "Manual (50/150)",
    f"Auto-σ ({T1:.0f}/{T2:.0f})",
    f"Percentile ({Tp1:.0f}/{Tp2:.0f})"
]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 9.7 Edge Detection in Color Images

Converting to grayscale before edge detection discards color information. Color edges — boundaries between differently-colored regions at similar brightness — are invisible to grayscale detectors.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def color_edge_magnitude(img_bgr):
    """
    Compute gradient magnitude in a color image using the Di Zenzo method.

    For each pixel, computes the 2×2 structure tensor and finds the
    direction of maximum gradient across all channels.

    This detects edges that exist in any channel, including pure hue changes.
    """
    img_f = img_bgr.astype(np.float32)

    # Compute gradients for each channel
    channels = cv2.split(img_f)
    Gxx = Gyy = Gxy = np.zeros_like(img_f[:,:,0])

    for ch in channels:
        Gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        Gxx += Gx * Gx
        Gyy += Gy * Gy
        Gxy += Gx * Gy

    # Structure tensor eigenvalue: maximum gradient direction
    # λ_max = (Gxx + Gyy + sqrt((Gxx-Gyy)² + 4*Gxy²)) / 2
    diff = Gxx - Gyy
    F = np.sqrt(diff**2 + 4 * Gxy**2)
    lambda_max = (Gxx + Gyy + F) / 2.0

    return np.sqrt(np.maximum(0, lambda_max))


def simple_channel_max(img_bgr):
    """
    Simpler approach: compute gradient magnitude per channel, take maximum.
    Less theoretically rigorous but often works well in practice.
    """
    channels = cv2.split(img_bgr.astype(np.float32))
    max_mag = np.zeros(img_bgr.shape[:2], dtype=np.float32)
    for ch in channels:
        Gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(Gx**2 + Gy**2)
        max_mag = np.maximum(max_mag, mag)
    return max_mag


# ── Create test image with a pure hue edge (same brightness, different color) ──
img_color = np.zeros((200, 300, 3), dtype=np.uint8)
img_color[:, :150] = [0, 0, 200]     # red (BGR)
img_color[:, 150:] = [0, 200, 0]     # green (BGR)
# Both regions have similar luminance (~82) — grayscale sees nothing!

gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
edges_gray  = cv2.Canny(gray, 50, 150)

dizenzo_mag = color_edge_magnitude(img_color)
ch_max_mag  = simple_channel_max(img_color)

print(f"Grayscale edge pixels: {edges_gray.sum() // 255}")
print(f"Di Zenzo edge pixels:  {(dizenzo_mag > 50).sum()}")
print(f"Channel max pixels:    {(ch_max_mag  > 50).sum()}")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, im, title in zip(axes, [
    cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB),
    gray, edges_gray,
    norm_to_uint8(dizenzo_mag), norm_to_uint8(ch_max_mag)
], ["Color input\n(pure hue edge)", "Grayscale\n(hue edge invisible)",
    "Canny on grayscale\n(misses hue edge)",
    "Di Zenzo method\n(detects hue edge!)",
    "Channel max\n(also detects)"]):
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0 if im.ndim == 2 else None)
    ax.set_title(title, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 9.8 Complete Project: Auto-Tuned Edge Detection Pipeline

A production pipeline that adapts its parameters to each image:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class EdgeConfig:
    """Configuration for an edge detection run."""
    method:        str        # 'canny', 'sobel', 'laplacian', 'log'
    pre_blur_sigma: float
    t_low:         Optional[float]
    t_high:        Optional[float]
    sobel_ksize:   int = 3
    log_sigma:     float = 2.0


def detect_edges(img_gray: np.ndarray,
                 config: Optional[EdgeConfig] = None,
                 auto_tune: bool = True) -> Tuple[np.ndarray, EdgeConfig]:
    """
    Detect edges with optional automatic parameter selection.

    Parameters
    ----------
    img_gray : np.ndarray  Grayscale uint8 input
    config   : EdgeConfig  Manual configuration (used if auto_tune=False)
    auto_tune : bool       If True, select parameters from image statistics

    Returns
    -------
    edges  : np.ndarray (H, W) uint8
    config : EdgeConfig  Parameters actually used
    """
    if auto_tune:
        config = _auto_config(img_gray)

    # Pre-blur
    if config.pre_blur_sigma > 0:
        k = int(6 * config.pre_blur_sigma) | 1
        processed = cv2.GaussianBlur(img_gray, (k, k), config.pre_blur_sigma)
    else:
        processed = img_gray

    if config.method == "canny":
        # Auto thresholds if not specified
        if config.t_low is None or config.t_high is None:
            Gx = cv2.Sobel(processed, cv2.CV_32F, 1, 0, ksize=3)
            Gy = cv2.Sobel(processed, cv2.CV_32F, 0, 1, ksize=3)
            G  = np.sqrt(Gx**2 + Gy**2)
            config = EdgeConfig(
                method="canny",
                pre_blur_sigma=config.pre_blur_sigma,
                t_low=float(np.percentile(G, 20)),
                t_high=float(np.percentile(G, 80)),
            )
        edges = cv2.Canny(processed,
                          float(config.t_low), float(config.t_high),
                          L2gradient=True)

    elif config.method == "sobel":
        Gx = cv2.Sobel(processed, cv2.CV_32F, 1, 0, ksize=config.sobel_ksize)
        Gy = cv2.Sobel(processed, cv2.CV_32F, 0, 1, ksize=config.sobel_ksize)
        G  = np.sqrt(Gx**2 + Gy**2)
        # Threshold by percentile
        t = config.t_high or float(np.percentile(G, 70))
        edges = ((G > t) * 255).astype(np.uint8)

    elif config.method == "laplacian":
        lap = cv2.Laplacian(processed, cv2.CV_32F, ksize=1)
        t = config.t_low or 15.0
        edges = zero_crossings(lap, threshold=t)

    elif config.method == "log":
        k = int(6 * config.log_sigma) | 1
        blurred = cv2.GaussianBlur(processed, (k, k), config.log_sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=1)
        t = config.t_low or 10.0
        edges = zero_crossings(lap, threshold=t)

    else:
        raise ValueError(f"Unknown method: {config.method}")

    return edges, config


def _auto_config(img_gray: np.ndarray) -> EdgeConfig:
    """Select edge detection configuration from image statistics."""
    # Noise estimate
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 1.0)
    noise_est = float(np.std(img_gray.astype(float) - blurred.astype(float)))

    # Contrast estimate
    contrast = float(img_gray.std())

    # Select pre-blur sigma based on noise
    if noise_est < 3:
        pre_blur = 0.5
    elif noise_est < 8:
        pre_blur = 1.0
    else:
        pre_blur = 2.0

    # Canny is default — best general-purpose detector
    return EdgeConfig(
        method="canny",
        pre_blur_sigma=pre_blur,
        t_low=None,  # will be auto-computed
        t_high=None,
    )


# ── Demo on multiple image types ──────────────────────────────────────────────
def make_test_images():
    images = {}

    # Clean, high-contrast
    clean = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(clean, (150, 100), 70, 200, -1)
    cv2.rectangle(clean, (30, 130), (120, 180), 150, -1)
    images["clean"] = clean

    # Noisy
    noisy = clean.copy()
    noise = np.random.normal(0, 25, clean.shape)
    images["noisy"] = np.clip(noisy.astype(float) + noise, 0, 255).astype(np.uint8)

    # Low contrast
    low_c = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(low_c, (150, 100), 70, 140, -1)
    cv2.rectangle(low_c, (30, 130), (120, 180), 100, -1)
    images["low_contrast"] = low_c

    return images

test_imgs = make_test_images()

fig, axes = plt.subplots(len(test_imgs), 4, figsize=(16, 10))
methods = ["canny", "sobel", "laplacian", "log"]

for row, (name, img) in enumerate(test_imgs.items()):
    axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[row, 0].set_title(f"{name}\n(input)", fontsize=9)
    axes[row, 0].axis("off")

    for col, method in enumerate(methods[:-1], start=1):
        edges, used_config = detect_edges(img,
            EdgeConfig(method=method, pre_blur_sigma=1.0,
                       t_low=None, t_high=None), auto_tune=False)
        axes[row, col].imshow(edges, cmap="gray", vmin=0, vmax=255)
        axes[row, col].set_title(f"{method}", fontsize=9)
        axes[row, col].axis("off")

plt.suptitle("Edge Detection Methods × Image Types", fontsize=13)
plt.tight_layout(); plt.show()
```

---

## 9.9 Comparison of All Edge Detectors

```python
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# EDGE DETECTOR COMPARISON
# ════════════════════════════════════════════════════════════════════════════
#
# METHOD       SPEED   NOISE   EDGE    DIRECTION  THIN   USE WHEN
#              (rel)   SENS.   THICK   INFO      EDGES
# ─────────────────────────────────────────────────────────────────────────
# Sobel         ★★★★    High    Yes     ✓         ✗    Quick gradient,
#                                                        directional analysis
# Scharr        ★★★★    High    Yes     ✓✓        ✗    Better angular accuracy
#                                                        than Sobel (3×3)
# Laplacian     ★★★★    Very    Yes     ✗         ~    Not recommended directly;
#              High     (ZC)              use LoG instead
# LoG           ★★★     Medium  Yes     ✗         ~    Scale-space analysis,
#                                                        SIFT preprocessor
# Canny         ★★★     Low     ✗       ✗         ✓    BEST GENERAL PURPOSE
#              (pre-blur                                 thin, connected edges
#               needed)
#
# ════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE: OpenCV functions
# ════════════════════════════════════════════════════════════════════════════

import cv2
img_gray = np.zeros((100, 100), dtype=np.uint8)

# Sobel X and Y gradients
Gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)        # ∂I/∂x
Gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)        # ∂I/∂y
G  = cv2.magnitude(Gx, Gy)                                   # √(Gx²+Gy²)
angle = cv2.phase(Gx, Gy, angleInDegrees=True)              # direction [0,360]

# Scharr (more accurate than Sobel for 3×3)
Sx = cv2.Scharr(img_gray, cv2.CV_32F, 1, 0)
Sy = cv2.Scharr(img_gray, cv2.CV_32F, 0, 1)

# Laplacian
lap = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=1)          # sum of 2nd derivatives

# Canny (recommended default)
edges = cv2.Canny(img_gray, threshold1=50, threshold2=150,  # L1 gradient
                   apertureSize=3, L2gradient=False)
edges = cv2.Canny(img_gray, threshold1=50, threshold2=150,  # L2 gradient (better)
                   L2gradient=True)

# cv2.magnitude and cv2.phase — compute from Gx, Gy arrays
# cv2.cartToPolar(Gx, Gy)  → (magnitude, angle) in one call
mag, ang = cv2.cartToPolar(Gx, Gy, angleInDegrees=True)

# ════════════════════════════════════════════════════════════════════════════
# DECISION GUIDE
# ════════════════════════════════════════════════════════════════════════════
# Need thin, connected edges?         → Canny (always pre-blur)
# Need gradient direction?            → Sobel or Scharr
# Need scale-space blobs?             → LoG (blob detection basis)
# Need the fastest possible option?   → Sobel (L1 magnitude, ksize=3)
# Color image edges?                  → Per-channel Sobel + max, or Di Zenzo
# Very noisy image?                   → Increase Gaussian pre-blur sigma
# Fine detail edges?                  → Small sigma (1.0) + Canny
# Only major boundaries?              → Large sigma (3+) + Canny
```

---

## 9.10 Summary

**Gradients** are the mathematical foundation of edge detection. The discrete central difference approximates ∂I/∂x and ∂I/∂y. Gradient magnitude `|G| = √(Gx²+Gy²)` and direction `θ = arctan2(Gy, Gx)` are the two fundamental quantities. Always use float output (`cv2.CV_32F`) to capture negative gradients.

**Sobel operator** (3×3 to 7×7): first derivative filter with built-in Gaussian smoothing. Directional — gives both Gx and Gy. Thick edges (need post-processing for thin results). Scharr (`ksize=1`) is the optimized 3×3 variant with better rotational symmetry.

**Laplacian** (second derivative): single kernel, isotropic, zero-crossings mark edges. Very noise-sensitive — almost always use after Gaussian pre-blur (= LoG).

**LoG** (Laplacian of Gaussian): combine Gaussian smoothing + Laplacian. σ controls scale — detects blobs/edges at radius r≈√2σ. Foundational for SIFT.

**Canny** (the standard): 4-stage algorithm (Gaussian → Sobel → NMS → hysteresis). Produces thin, connected, well-localized 1-pixel-wide edges. Auto-tune thresholds using gradient magnitude percentiles.

**Color edge detection**: grayscale conversion misses hue-only edges. Use per-channel gradient maximum or Di Zenzo structure tensor for full-color edge detection.

---

## 9.11 Exercises

### Warm-up

**9.1** Compute Gx, Gy, magnitude, and direction using cv2.Sobel on a grayscale image. Display all four results. Draw 20 arrows at random non-zero-gradient locations showing the gradient direction and magnitude using `cv2.arrowedLine`.

**9.2** Show the duality of Sobel and Canny: apply Sobel (magnitude) and Canny to the same image with roughly equivalent sensitivity. Compute the Intersection over Union (IoU) between the two edge maps: `IoU = |A ∩ B| / |A ∪ B|`. What do you find?

**9.3** Implement a `canny_nms(Gx, Gy)` function that performs non-maximum suppression manually: for each pixel, check its neighbors along the gradient direction (rounded to 4 angles: 0°, 45°, 90°, 135°) and suppress if not a local maximum. Compare your result to cv2.Canny's intermediate NMS output.

### Core

**9.4** Build a `gradient_visualization(img_gray, step=20)` function that computes the gradient field on a grid (every `step` pixels) and visualizes it as: (a) a quiver/arrow plot using `matplotlib.pyplot.quiver`, (b) an HSV image where hue = direction and value = magnitude. This is the classic "gradient flow" visualization.

**9.5** Implement `auto_canny_scale(img_gray, sigmas=[0.5, 1, 2, 4])` that runs Canny at multiple pre-blur scales and combines the results with an OR operation (multi-scale edge detection). For each scale, use automatic percentile-based thresholds. Display the individual scale results and the combined map.

**9.6** Compare Canny with Sobel thresholding (magnitude > T) on 5 test images. For each, vary T to match the number of Canny edge pixels. Compute the following metrics: (a) average edge pixel neighborhood connectivity (how many edge neighbors does each edge pixel have), (b) fraction of edge pixels that are isolated (0 edge neighbors). Canny should show consistently better connectivity.

**9.7** Implement the `color_edge_magnitude` (Di Zenzo) function and compare it to grayscale-converted Canny on images containing: (a) a pure hue edge (same luminance, different color), (b) a textured object on a similarly-bright background, (c) a natural photograph. Score detection quality qualitatively.

### Challenge

**9.8** Implement a **full Canny algorithm from scratch** using only NumPy and `cv2.Sobel`. The pipeline: (1) Gaussian pre-blur, (2) compute Gx, Gy, magnitude, direction, (3) non-maximum suppression (round angle to 4 directions, check neighbors), (4) double threshold (strong/weak/none), (5) hysteresis (DFS/BFS flood from strong edges). Verify against `cv2.Canny` on 5 images and report pixel-level accuracy.

**9.9** Build a **real-time edge detector** that reads from webcam (or video file), applies Canny with trackbar-controlled pre-blur sigma and both thresholds, overlays the edges on the original frame in red (using `cv2.addWeighted`), and displays FPS. The overlay should be updated at every frame. Optimize for 30fps on HD resolution.

**9.10** Implement **edge-based image comparison**: given two images of the same scene taken at slightly different exposures, compute: (a) Canny edges for each, (b) align them using `cv2.findHomography` + `cv2.warpPerspective`, (c) compute the Chamfer distance between edge maps (each edge pixel's distance to the nearest edge in the other map), (d) identify regions with large Chamfer distance (significant structural differences). Visualize the difference map.

---

## Further Reading

- **OpenCV Canny tutorial:** https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
- **Canny, J. (1986). "A computational approach to edge detection."** IEEE TPAMI 8(6), 679–698 — the original paper
- **Marr, D. & Hildreth, E. (1980). "Theory of edge detection."** Proceedings of the Royal Society B, 207(1167), 187–217 — foundational LoG paper
- **Di Zenzo, S. (1986). "A note on the gradient of a multi-image."** Computer Vision, Graphics, and Image Processing — the color gradient method
- **Sobel, I. (1968). Camera models and machine perception.** Stanford AI Lab Memo — original Sobel filter
- **Szeliski, R. (2022). "Computer Vision: Algorithms and Applications" 2nd ed., §4.2** — thorough treatment of edge detection theory
