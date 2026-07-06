# Chapter 8: Morphological Operations

> *"Morphology treats an image as a set. The structuring element is a probe you drag through that set — everything it fits inside becomes foreground; everything it doesn't becomes background."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain morphological operations as set operations on binary images
2. Apply erosion and dilation and predict their effects on foreground shape
3. Design structuring elements for targeted effects (rectangle, cross, ellipse, custom)
4. Use opening, closing, gradient, top-hat, and black-hat operations with `cv2.morphologyEx`
5. Control operation strength via `iterations` and kernel size
6. Apply morphological operations to grayscale images and interpret the results
7. Build morphological sequences to solve real noise-removal and object-separation problems
8. Use hit-or-miss transform for template matching on binary images
9. Design complete morphological preprocessing pipelines for segmentation

---

## 8.1 Mathematical Morphology: The Core Idea

Mathematical morphology treats images not as arrays of numbers but as **sets of pixels**. A binary image is a set of foreground pixel positions: `F = {(x, y) : image[y, x] == 255}`.

A **structuring element** (SE) is a small shape — also a set of pixel positions — that acts as a probe. As you translate the SE across the image, its relationship to the foreground set determines the output.

Two primitive operations define all of morphology:

- **Erosion:** A pixel stays foreground only if the SE fits **entirely** inside the foreground when centered at that pixel
- **Dilation:** A pixel becomes foreground if the SE **touches any part** of the foreground when centered at that pixel

Every other morphological operation is built by combining erosion and dilation.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_test_binary(shape=(200, 300), add_noise=True):
    """Create a synthetic binary image for demonstration."""
    img = np.zeros(shape, dtype=np.uint8)
    # Several shapes: circle, rectangle, small dots (noise)
    cv2.circle(img, (80, 100), 50, 255, -1)
    cv2.rectangle(img, (170, 50), (270, 160), 255, -1)
    cv2.ellipse(img, (150, 170), (60, 25), 30, 0, 360, 255, -1)
    if add_noise:
        # Salt noise: random foreground specks
        rng = np.random.default_rng(42)
        noise_coords = rng.integers(0, min(shape), size=(2, 200))
        img[noise_coords[0], noise_coords[1]] = 255
        # Holes in foreground: random black specks
        hole_coords = rng.integers(20, 80, size=(2, 60))
        img[hole_coords[0], hole_coords[1]] = 0
    return img


img_bin = make_test_binary()
print(f"Binary image shape: {img_bin.shape}")
print(f"Foreground pixels:  {img_bin.sum() // 255}")
print(f"Background pixels:  {(img_bin == 0).sum()}")
```

---

## 8.2 Structuring Elements

The structuring element is the "shape of the probe." Its size and shape determine what features survive each morphological operation.

### 8.2.1 cv2.getStructuringElement

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# cv2.getStructuringElement(shape, ksize, anchor=(-1,-1))
# shape:  cv2.MORPH_RECT     — filled rectangle
#         cv2.MORPH_CROSS    — plus-sign cross (horizontal + vertical)
#         cv2.MORPH_ELLIPSE  — ellipse (circle for square ksize)
# ksize:  (width, height) — both should be odd for symmetric anchor

sizes_to_show = [(3, 3), (5, 5), (7, 7), (11, 11)]
shapes = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]
shape_names = ["MORPH_RECT", "MORPH_CROSS", "MORPH_ELLIPSE"]

# Show all structuring elements
fig, axes = plt.subplots(len(shapes), len(sizes_to_show),
                          figsize=(10, 7))
for i, (shape, name) in enumerate(zip(shapes, shape_names)):
    for j, size in enumerate(sizes_to_show):
        se = cv2.getStructuringElement(shape, size)
        axes[i, j].imshow(se, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
        axes[i, j].set_title(f"{name}\n{size[0]}×{size[1]}", fontsize=8)
        axes[i, j].axis("off")
        # Print numeric kernel
        if size == (5, 5):
            print(f"\n{name} 5×5:\n{se}")

plt.suptitle("Structuring Element Gallery", fontsize=12)
plt.tight_layout(); plt.show()
```

**Output for 5×5 kernels:**
```
MORPH_RECT 5×5:
[[1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]]

MORPH_CROSS 5×5:
[[0 0 1 0 0]
 [0 0 1 0 0]
 [1 1 1 1 1]
 [0 0 1 0 0]
 [0 0 1 0 0]]

MORPH_ELLIPSE 5×5:
[[0 0 1 0 0]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [0 0 1 0 0]]
```

### 8.2.2 Custom Structuring Elements

```python
import numpy as np
import cv2

# Any binary NumPy array can serve as a structuring element

# Horizontal bar (5×1) — removes/connects objects along horizontal axis
se_hbar = np.ones((1, 15), dtype=np.uint8)

# Vertical bar (1×15)
se_vbar = np.ones((15, 1), dtype=np.uint8)

# Diagonal structuring element
se_diag = np.eye(7, dtype=np.uint8)

# Ring (hollow circle) — useful for detecting ring-shaped objects
se_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
se_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
pad = (11 - 7) // 2
se_ring[pad:pad+7, pad:pad+7] -= se_inner   # outer - inner = ring

# L-shaped structuring element
se_L = np.zeros((7, 7), dtype=np.uint8)
se_L[:, 0] = 1   # left column
se_L[-1, :] = 1  # bottom row

print(f"Horizontal bar: {se_hbar.shape}  {se_hbar}")
print(f"Diagonal SE:\n{se_diag}")
```

---

## 8.3 Erosion

**Erosion** shrinks foreground regions. A foreground pixel survives erosion only if the entire structuring element, when centered at that pixel, lies within the foreground.

**Effect on binary images:**
- Removes pixels from the boundary of white regions
- Eliminates small foreground specks (noise) that are smaller than the SE
- Disconnects touching objects if the connection is narrower than the SE
- Shrinks all foreground shapes

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = make_test_binary(add_noise=True)

# cv2.erode(src, kernel, anchor=(-1,-1), iterations=1, borderType, borderValue)
# iterations: how many times to apply the operation (equivalent to larger SE)

se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
se7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
se15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

eroded_3   = cv2.erode(img, se3)
eroded_7   = cv2.erode(img, se7)
eroded_15  = cv2.erode(img, se15)
eroded_3x3 = cv2.erode(img, se3, iterations=3)   # 3 passes with 3×3 ≈ 1 pass with 7×7

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for ax, im, title in zip(axes,
    [img, eroded_3, eroded_7, eroded_15, eroded_3x3],
    ["Original", "Erode SE=3×3", "Erode SE=7×7", "Erode SE=15×15",
     "Erode 3×3\n(3 iterations)"]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title); ax.axis("off")
plt.suptitle("Erosion Effects", fontsize=12)
plt.tight_layout(); plt.show()

# ── What erosion removes ──────────────────────────────────────────────────────
noise_removed = img.copy()
noise_removed[eroded_3 == 0] = 128   # gray = pixels removed by erosion
# All the random noise specks (smaller than 3×3) are gone
print(f"Pixels removed by 3×3 erosion: {(img.astype(int) - eroded_3.astype(int)).clip(0).sum() // 255}")
```

### 8.3.1 Erosion on Grayscale Images

Morphological operations also work on grayscale images, where they have a different interpretation:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_gray = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    img_gray = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(img_gray, (100, 100), 60, 200, -1)
    cv2.rectangle(img_gray, (170, 60), (260, 150), 150, -1)

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Grayscale erosion: minimum of neighborhood (darkens image)
# Each pixel becomes the minimum value in its SE neighborhood
eroded_gray = cv2.erode(img_gray, se)

# Grayscale dilation: maximum of neighborhood (brightens image)
dilated_gray = cv2.dilate(img_gray, se)

print("Grayscale morphology interpretation:")
print("  Erosion  = local MINIMUM filter (darkens, shrinks bright regions)")
print("  Dilation = local MAXIMUM filter (brightens, expands bright regions)")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, im, title in zip(axes,
    [img_gray, eroded_gray, dilated_gray],
    ["Original", "Grayscale Erosion\n(local min → darker)", "Grayscale Dilation\n(local max → brighter)"]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 8.4 Dilation

**Dilation** grows foreground regions. A pixel becomes (or stays) foreground if any part of the SE, centered at that pixel, overlaps with foreground.

**Effect on binary images:**
- Adds pixels to the boundary of white regions
- Fills small holes (black specks) within foreground regions
- Connects nearby objects if the gap is smaller than the SE
- Grows all foreground shapes

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = make_test_binary(add_noise=False)
# Add holes manually for demonstration
img[40:60, 40:60] = 0    # hole in circle
img[80:100, 200:220] = 0  # hole in rectangle

se5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
se11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
se21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

dilated_5  = cv2.dilate(img, se5)
dilated_11 = cv2.dilate(img, se11)
dilated_21 = cv2.dilate(img, se21)

# Dilation connects the two closest objects if gap < SE size
gap_image = np.zeros((100, 300), dtype=np.uint8)
cv2.rectangle(gap_image, (20,  30), (80,  70), 255, -1)    # left object
cv2.rectangle(gap_image, (100, 30), (160, 70), 255, -1)   # 20px gap
cv2.rectangle(gap_image, (200, 30), (260, 70), 255, -1)   # 40px gap from prev
se_gap = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
dilated_gap = cv2.dilate(gap_image, se_gap)

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, im, title in zip(axes.flat,
    [img, dilated_5, dilated_11,
     dilated_21, gap_image, dilated_gap],
    ["Original (with holes)", "Dilate SE=5×5", "Dilate SE=11×11",
     "Dilate SE=21×21", "Objects with gaps", "Dilation connects objects"]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title); ax.axis("off")
plt.suptitle("Dilation Effects", fontsize=12)
plt.tight_layout(); plt.show()
```

### 8.4.1 The Duality of Erosion and Dilation

Erosion and dilation are dual operations:

```python
import numpy as np
import cv2

img = make_test_binary(add_noise=False)
se  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Duality: eroding foreground = dilating background (and vice versa)
# Erosion(img) == NOT Dilation(NOT img)

eroded   = cv2.erode(img, se)
dilated  = cv2.dilate(img, se)

inv_img      = cv2.bitwise_not(img)
dil_inv      = cv2.dilate(inv_img, se)
duality_check = cv2.bitwise_not(dil_inv)  # should equal eroded

print(f"Duality holds (erode == NOT(dilate(NOT(img)))): "
      f"{np.array_equal(eroded, duality_check)}")  # True
```

---

## 8.5 cv2.morphologyEx: The Compound Operations

OpenCV provides `cv2.morphologyEx` for compound operations that combine erosion and dilation:

```python
import cv2
import numpy as np

# cv2.morphologyEx(src, op, kernel, anchor, iterations, borderType, borderValue)
#
# op codes:
#   cv2.MORPH_ERODE     — erosion (same as cv2.erode)
#   cv2.MORPH_DILATE    — dilation (same as cv2.dilate)
#   cv2.MORPH_OPEN      — opening: erosion → dilation
#   cv2.MORPH_CLOSE     — closing: dilation → erosion
#   cv2.MORPH_GRADIENT  — morphological gradient: dilation - erosion
#   cv2.MORPH_TOPHAT    — top-hat: src - opening
#   cv2.MORPH_BLACKHAT  — black-hat: closing - src
#   cv2.MORPH_HITMISS   — hit-or-miss (binary only)
```

### 8.5.1 Opening: Erosion then Dilation

Opening = `dilate(erode(img, se), se)`

**Effect:** Removes small bright objects (smaller than SE) while preserving larger shapes. The erosion kills small objects; the subsequent dilation restores the shape of surviving objects.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Test image: large shapes + small noise specks
img = make_test_binary(add_noise=True)
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
# Equivalent to:
# opened_manual = cv2.dilate(cv2.erode(img, se), se)
# assert np.array_equal(opened, opened_manual)

# What was removed (noise)?
noise_mask = cv2.bitwise_and(img, cv2.bitwise_not(opened))  # in original but not after opening

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(img,        cmap="gray"); axes[0].set_title("Input (with noise specks)"); axes[0].axis("off")
axes[1].imshow(opened,     cmap="gray"); axes[1].set_title("After Opening\n(noise removed, large shapes intact)"); axes[1].axis("off")
axes[2].imshow(noise_mask, cmap="gray"); axes[2].set_title("Removed pixels\n(the noise)"); axes[2].axis("off")
plt.tight_layout(); plt.show()

print(f"Noise pixels removed: {noise_mask.sum() // 255}")
```

### 8.5.2 Closing: Dilation then Erosion

Closing = `erode(dilate(img, se), se)`

**Effect:** Fills small dark holes within foreground regions. The dilation fills holes; the subsequent erosion restores the outer shape.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Test image with holes in objects
img_holes = make_test_binary(add_noise=False)
# Add many small holes
rng = np.random.default_rng(10)
hole_ys = rng.integers(10, 190, 150)
hole_xs = rng.integers(10, 290, 150)
for y, x in zip(hole_ys, hole_xs):
    if img_holes[y, x] == 255:   # only add holes in foreground
        img_holes[y-1:y+2, x-1:x+2] = 0   # 3×3 hole

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
closed = cv2.morphologyEx(img_holes, cv2.MORPH_CLOSE, se)

# What was filled?
filled_mask = cv2.bitwise_and(cv2.bitwise_not(img_holes), closed)  # in closed but not original

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(img_holes,  cmap="gray"); axes[0].set_title("Input (with holes)"); axes[0].axis("off")
axes[1].imshow(closed,     cmap="gray"); axes[1].set_title("After Closing\n(holes filled, outer shape intact)"); axes[1].axis("off")
axes[2].imshow(filled_mask,cmap="gray"); axes[2].set_title("Filled pixels\n(the holes)"); axes[2].axis("off")
plt.tight_layout(); plt.show()
```

### 8.5.3 Opening vs Closing: The Decision Rule

```
Opening  — use when you want to REMOVE small foreground features:
           small noise specks, thin protrusions, narrow bridges
           "erosion-dominant" effect

Closing  — use when you want to FILL small background features:
           small holes, gaps, thin cracks in foreground
           "dilation-dominant" effect

Combined (open then close) — remove noise AND fill holes simultaneously
```

### 8.5.4 Morphological Gradient

Gradient = `dilate(img) - erode(img)`

The result is the **boundary** of foreground regions — a morphological edge detector.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = make_test_binary(add_noise=False)
se  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
se7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

gradient_thin  = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se)
gradient_thick = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se7)

# Manual verification
eroded  = cv2.erode(img, se)
dilated = cv2.dilate(img, se)
gradient_manual = cv2.subtract(dilated, eroded)   # or: (dilated.astype(int) - eroded.astype(int)).clip(0, 255)
print(f"Gradient matches manual: {np.array_equal(gradient_thin, gradient_manual)}")  # True

# Gradient variants (not in OpenCV but useful)
# Internal gradient: img - erode(img)  → inner boundary
# External gradient: dilate(img) - img → outer boundary
internal_grad = cv2.subtract(img, eroded)    # pixels at inner edge
external_grad = cv2.subtract(dilated, img)   # pixels at outer edge

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, im, title in zip(axes.flat,
    [img, gradient_thin, gradient_thick,
     internal_grad, external_grad,
     cv2.bitwise_or(internal_grad, external_grad)],
    ["Original", "Morphological Gradient\n(SE=3×3, thin)",
     "Morphological Gradient\n(SE=7×7, thick)",
     "Internal gradient\n(inner boundary)",
     "External gradient\n(outer boundary)",
     "Full boundary\n(internal ∪ external)"]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
```

### 8.5.5 Top-Hat and Black-Hat

These two operations extract fine detail that is brighter (top-hat) or darker (black-hat) than their surroundings:

```
Top-Hat   = src - Opening(src)    → bright features SMALLER than SE
Black-Hat = Closing(src) - src    → dark features SMALLER than SE
```

Top-hat and black-hat are extremely powerful for **illumination correction** and extracting small features from non-uniform backgrounds.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_tophat_blackhat():
    # Create image: dark background with small bright spots + large bright blob
    img = np.zeros((200, 400), dtype=np.uint8)

    # Large bright region (will be "normalized away" by opening)
    img[50:150, 50:350] = 100   # large background variation

    # Small bright objects on the large region
    for x in [100, 200, 300]:
        cv2.circle(img, (x, 100), 12, 220, -1)   # bright dots
    cv2.circle(img, (150, 100), 60, 130, -1)  # medium object

    se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

    tophat   = cv2.morphologyEx(img, cv2.MORPH_TOPHAT,   se_large)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se_large)

    print("Top-hat result: only the small bright dots remain")
    print(f"  Original range: [{img.min()}, {img.max()}]")
    print(f"  Top-hat range:  [{tophat.min()}, {tophat.max()}]")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(tophat, cmap="gray"); axes[1].set_title("Top-Hat\n(small bright features)"); axes[1].axis("off")
    axes[2].imshow(blackhat, cmap="gray"); axes[2].set_title("Black-Hat\n(small dark features)"); axes[2].axis("off")
    plt.tight_layout(); plt.show()


def demonstrate_tophat_illumination():
    """
    Classic use case: top-hat removes non-uniform background illumination.
    """
    # Simulated image with gradient background + text
    img = np.zeros((200, 400), dtype=np.uint8)
    # Non-uniform background: bright in center
    Y, X = np.mgrid[0:200, 0:400]
    bg = (60 + 80 * np.exp(-((X - 200)**2 + (Y - 100)**2) / (2 * 100**2))).astype(np.uint8)
    img = bg.copy()
    # Add dark "text"
    for y in range(50, 160, 30):
        for x in range(40, 360, 40):
            cv2.rectangle(img, (x, y), (x+20, y+18), 20, -1)

    # Problem: threshold fails on non-uniform background
    _, bad_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Solution: top-hat removes background, then threshold
    se_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    tophat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se_bg)
    # Wait — we want to find dark text, so use BLACK-HAT instead
    blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se_bg)
    _, good_thresh = cv2.threshold(blackhat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, im, title in zip(axes,
        [img, bad_thresh, blackhat_img, good_thresh],
        ["Input (non-uniform bg)", "Direct Otsu (fails)",
         "Black-Hat result", "Otsu on Black-Hat (works)"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.show()


demonstrate_tophat_blackhat()
demonstrate_tophat_illumination()
```

---

## 8.6 Structuring Element Shape Effects

The **shape** of the SE determines which features are affected:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compare_se_shapes(img_bin, kernel_size=11):
    """Show how SE shape changes the morphological effect."""
    se_rect  = cv2.getStructuringElement(cv2.MORPH_RECT,    (kernel_size, kernel_size))
    se_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,   (kernel_size, kernel_size))
    se_ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    se_hbar  = np.ones((1, kernel_size), dtype=np.uint8)
    se_vbar  = np.ones((kernel_size, 1), dtype=np.uint8)

    results = []
    names   = []
    for se, name in [
        (se_rect,  f"RECT {kernel_size}×{kernel_size}"),
        (se_cross, f"CROSS {kernel_size}×{kernel_size}"),
        (se_ellip, f"ELLIPSE {kernel_size}×{kernel_size}"),
        (se_hbar,  f"H-BAR 1×{kernel_size}"),
        (se_vbar,  f"V-BAR {kernel_size}×1"),
    ]:
        results.append(cv2.erode(img_bin, se))
        names.append(name)

    n = len(results) + 1
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes.flat[0].imshow(img_bin, cmap="gray")
    axes.flat[0].set_title("Original"); axes.flat[0].axis("off")
    for ax, result, name in zip(axes.flat[1:], results, names):
        ax.imshow(result, cmap="gray")
        ax.set_title(f"Erode\n{name}", fontsize=9)
        ax.axis("off")
    plt.suptitle("Effect of SE Shape on Erosion", fontsize=12)
    plt.tight_layout(); plt.show()


# Create test image with objects of different orientations
img = np.zeros((200, 300), dtype=np.uint8)
cv2.rectangle(img, (20, 80), (130, 110), 255, -1)   # horizontal bar
cv2.rectangle(img, (150, 20), (180, 180), 255, -1)  # vertical bar
cv2.circle(img, (250, 100), 50, 255, -1)             # circle

compare_se_shapes(img, kernel_size=15)

# Key insight:
# - RECT removes anything smaller than kernel_size × kernel_size
# - H-BAR preserves wide horizontal objects, removes narrow/vertical
# - V-BAR preserves tall vertical objects, removes narrow/horizontal
# - ELLIPSE has smooth, isotropic effect (best for "natural" shapes)
# - CROSS is a compromise: cheaper than RECT, more coverage than lines
```

---

## 8.7 Iterations vs Kernel Size

Multiple iterations with a small SE is nearly equivalent to one pass with a larger SE, but the relationship is not exact for non-convex shapes:

```python
import cv2
import numpy as np

img = make_test_binary(add_noise=False)
se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
se9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# 3 iterations of 3×3 SE ≈ 1 iteration of 7×7 SE (for convex shapes)
# Exact equivalence holds for rectangular SEs (Minkowski sum)
iter3_result = cv2.erode(img, se3, iterations=3)
single9_result = cv2.erode(img, se9)

diff = np.abs(iter3_result.astype(int) - single9_result.astype(int))
print(f"3 iter 3×3 vs 1 iter 9×9: max diff = {diff.max()}, "
      f"differing pixels = {(diff > 0).sum()}")
# Small difference — the dilation-of-SE rule: n×(d-1)+1 = effective size
# For ellipse: 3 iters × (3-1) + 1 = 7 effective; but 9 is slightly larger

# Performance: iterations vs large SE
import time

n_runs = 100
t0 = time.perf_counter()
for _ in range(n_runs): cv2.erode(img, se3, iterations=5)
t_iter = (time.perf_counter() - t0) / n_runs * 1000

se11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
t0 = time.perf_counter()
for _ in range(n_runs): cv2.erode(img, se11)
t_large = (time.perf_counter() - t0) / n_runs * 1000

print(f"5 iter × 3×3:  {t_iter:.3f} ms")
print(f"1 iter × 11×11: {t_large:.3f} ms")
# Multiple iterations of small SE is usually faster than one large SE
# due to separability and cache efficiency
```

---

## 8.8 Hit-or-Miss Transform

The **hit-or-miss transform** (`cv2.MORPH_HITMISS`) is the only morphological operation that requires a two-part structuring element: a "hit" region (must be foreground) and a "miss" region (must be background). It detects specific binary patterns.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a binary image with various dot patterns
img = np.zeros((100, 200), dtype=np.uint8)
# Isolated single pixels (what we want to find)
img[20, 20] = 255
img[20, 80] = 255
img[60, 120] = 255
# Non-isolated pixels
img[40:50, 40:60] = 255   # block
img[70:80, 160:180] = 255  # another block

# Hit-or-miss kernel to find isolated pixels:
# Hit (1): center pixel must be foreground
# Miss (-1): all 8 neighbors must be background
# Don't care (0): not constrained
K_hitmiss = np.array([
    [-1, -1, -1],
    [-1,  1, -1],
    [-1, -1, -1]
], dtype=np.int8)

isolated = cv2.morphologyEx(img, cv2.MORPH_HITMISS, K_hitmiss)

print(f"Input foreground pixels: {img.sum() // 255}")
print(f"Isolated pixels found:   {isolated.sum() // 255}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img,      cmap="gray"); axes[0].set_title("Input"); axes[0].axis("off")
axes[1].imshow(isolated, cmap="gray"); axes[1].set_title("Hit-or-Miss: isolated pixels"); axes[1].axis("off")
plt.tight_layout(); plt.show()
```

---

## 8.9 Complete Project: Cell Counting Pipeline

Morphological operations are foundational in biological image analysis. Let us build a complete pipeline that separates, counts, and measures overlapping circular cells:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List


@dataclass
class Cell:
    """Detected cell properties."""
    label:    int
    area:     int
    centroid: tuple
    bbox:     tuple   # (x, y, w, h)
    radius:   float   # equivalent circle radius


def generate_cell_image(n_cells=25, h=400, w=500, seed=42):
    """
    Generate a synthetic microscopy image with overlapping circular cells.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.uint8)

    for _ in range(n_cells):
        cx = rng.integers(30, w - 30)
        cy = rng.integers(30, h - 30)
        r  = rng.integers(20, 45)
        brightness = rng.integers(150, 240)
        cv2.circle(img, (cx, cy), r, int(brightness), -1)

    # Add noise
    noise = rng.normal(0, 8, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Add background gradient
    Y, X = np.mgrid[0:h, 0:w]
    bg_gradient = (20 * X / w + 10 * Y / h).astype(np.uint8)
    img = np.clip(img.astype(np.int32) + bg_gradient, 0, 255).astype(np.uint8)

    return img


def count_cells(img_gray: np.ndarray, min_area: int = 400) -> tuple:
    """
    Count and measure cells in a grayscale microscopy image.

    Pipeline:
    1. Threshold to get binary foreground
    2. Open (remove noise) then close (fill holes)
    3. Distance transform + watershed to separate touching cells
    4. Connected components to label each cell
    5. Extract measurements

    Returns (labeled_image, cells_list, visualization)
    """
    # ── Step 1: Threshold ─────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Step 2: Morphological cleanup ─────────────────────────────────────────
    # Opening: remove small noise specks
    se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se_open, iterations=2)
    # Closing: fill small holes within cells
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, se_close, iterations=2)

    # ── Step 3: Separate touching cells with distance transform + watershed ────
    # Distance transform: each foreground pixel gets its distance to background
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    # Sure foreground: pixels far from background (centers of cells)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Sure background: pixels definitely not cells
    sure_bg = cv2.dilate(cleaned, se_close, iterations=3)

    # Unknown region: between sure_fg and sure_bg (the touching/overlapping area)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label the markers for watershed
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1   # add 1 so background is labeled 1, not 0
    markers[unknown == 255] = 0   # unknown region = 0 (watershed fills these)

    # Apply watershed on the color image (watershed needs 3-channel input)
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # ── Step 4: Extract cell measurements ─────────────────────────────────────
    cells: List[Cell] = []
    for label_id in range(2, n_markers + 1):  # skip 1 (background) and -1 (watershed border)
        mask = (markers == label_id).astype(np.uint8) * 255
        area = int(mask.sum() // 255)

        if area < min_area:
            continue

        # Centroid via moments
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            continue

        x, y, w, h = cv2.boundingRect(mask)
        radius = np.sqrt(area / np.pi)

        cells.append(Cell(
            label=label_id,
            area=area,
            centroid=(cx, cy),
            bbox=(x, y, w, h),
            radius=float(radius),
        ))

    # ── Step 5: Visualization ─────────────────────────────────────────────────
    # Color each cell differently
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(99)

    for cell in cells:
        color = tuple(int(c) for c in rng.integers(80, 255, 3))
        vis[markers == cell.label] = (
            int(vis[markers == cell.label, 0].mean() * 0.5 + color[0] * 0.5),
            int(vis[markers == cell.label, 1].mean() * 0.5 + color[1] * 0.5),
            int(vis[markers == cell.label, 2].mean() * 0.5 + color[2] * 0.5),
        )
        # Outline
        cv2.circle(vis, cell.centroid, int(cell.radius),
                   (255, 255, 255), 1, cv2.LINE_AA)
        # Label
        cv2.putText(vis, str(cell.label - 1),
                    (cell.centroid[0] - 5, cell.centroid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # Watershed borders
    vis[markers == -1] = [0, 0, 255]   # red watershed lines

    return markers, cells, vis, dist_transform, cleaned


# ── Run the pipeline ──────────────────────────────────────────────────────────
img_cells = generate_cell_image(n_cells=25)
markers, cells, vis, dist_t, cleaned = count_cells(img_cells)

print(f"Detected {len(cells)} cells")
if cells:
    areas = [c.area for c in cells]
    radii = [c.radius for c in cells]
    print(f"Area:   mean={np.mean(areas):.0f}, std={np.std(areas):.0f}, "
          f"min={min(areas)}, max={max(areas)}")
    print(f"Radius: mean={np.mean(radii):.1f} px")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, im, title in zip(axes.flat, [
    img_cells, cleaned,
    (dist_t / dist_t.max() * 255).astype(np.uint8),
    vis, img_cells, cleaned
], [
    "Raw input", "After morphological cleanup",
    "Distance transform", "Cell detection result",
    "", ""
]):
    if im.ndim == 2:
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(title); ax.axis("off")

for ax in axes.flat[4:]: ax.set_visible(False)
plt.suptitle(f"Cell Counting Pipeline: {len(cells)} cells detected", fontsize=13)
plt.tight_layout(); plt.show()
```

---

## 8.10 Morphological Operation Quick Reference

```python
import cv2
import numpy as np

# ════════════════════════════════════════════════════════════
# MORPHOLOGICAL OPERATIONS QUICK REFERENCE
# ════════════════════════════════════════════════════════════

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
img = np.zeros((100, 200), dtype=np.uint8)

# Basic operations
cv2.erode(img, se)                              # shrink FG, remove small FG blobs
cv2.dilate(img, se)                             # grow FG, fill small BG holes

# Compound operations (cv2.morphologyEx)
cv2.morphologyEx(img, cv2.MORPH_OPEN,      se)  # erode→dilate: remove small FG noise
cv2.morphologyEx(img, cv2.MORPH_CLOSE,     se)  # dilate→erode: fill small BG holes
cv2.morphologyEx(img, cv2.MORPH_GRADIENT,  se)  # dilate - erode: boundary outline
cv2.morphologyEx(img, cv2.MORPH_TOPHAT,    se)  # img - open: bright spots on dark bg
cv2.morphologyEx(img, cv2.MORPH_BLACKHAT,  se)  # close - img: dark spots on bright bg
cv2.morphologyEx(img, cv2.MORPH_HITMISS,   se)  # detect specific binary pattern

# Structuring elements
cv2.getStructuringElement(cv2.MORPH_RECT,    (k, k))  # filled rectangle
cv2.getStructuringElement(cv2.MORPH_CROSS,   (k, k))  # + cross shape
cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))  # ellipse/circle
np.ones((1, k), dtype=np.uint8)                        # horizontal bar
np.ones((k, 1), dtype=np.uint8)                        # vertical bar

# Key constraints
# - ksize should be odd for centered anchor
# - For MORPH_HITMISS: use int8 kernel with 1=hit, -1=miss, 0=don't care
# - iterations > 1 is equivalent to larger SE (computationally cheaper)
# - Gray images: erosion=local_min, dilation=local_max

# ════════════════════════════════════════════════════════════
# WHEN TO USE WHICH OPERATION
# ════════════════════════════════════════════════════════════
#
# Remove salt noise (small bright specks):    OPEN  (erode→dilate)
# Fill holes in foreground:                   CLOSE (dilate→erode)
# Remove salt + fill holes:                   OPEN then CLOSE
# Extract edges/boundaries:                   GRADIENT
# Find bright spots on dark background:       TOP-HAT
# Find dark spots on bright background:       BLACK-HAT
# Correct non-uniform illumination:           TOP-HAT or BLACK-HAT
# Separate touching objects:                  ERODE then distance transform
# Find specific pixel patterns:               HIT-OR-MISS
# Thin skeleton:                              iterate ERODE until convergence
```

---

## 8.11 Summary

Morphological operations treat binary images as sets and probe them with a structuring element. The two primitives — erosion (minimum fitting) and dilation (any overlap) — compose into all other operations:

**Opening** (erode→dilate): removes noise smaller than SE, preserves large shapes. Use when the input has salt noise or thin protrusions.

**Closing** (dilate→erode): fills holes smaller than SE, preserves large shapes. Use when the foreground has internal holes or thin gaps.

**Gradient** (dilate − erode): the morphological edge, a thick outline of objects.

**Top-hat** (img − opening): isolates small bright features; powerful for illumination correction.

**Black-hat** (closing − img): isolates small dark features; the dark-text companion to top-hat.

**Structuring element shape matters:** ellipse for isotropic effects; horizontal/vertical bars for directional selection; cross for 4-connected effects; custom shapes for template-like matching.

**Grayscale morphology:** erosion = local minimum, dilation = local maximum. Useful for background estimation, normalization, and lighting correction.

---

## 8.12 Exercises

### Warm-up

**8.1** Create a 400×600 binary image with 5 large circles, 20 medium rectangles, and 200 random 1-pixel salt noise dots. Apply opening with SE sizes 3, 7, 11 and count how many foreground pixels are removed at each step. Plot the foreground pixel count vs SE size.

**8.2** Demonstrate dilation connectivity: create two rectangles separated by gaps of 5, 15, and 30 pixels. For each gap, find the minimum SE size that connects them with dilation. Show the result for all three gaps.

**8.3** Apply all 7 `cv2.morphologyEx` operations (open, close, gradient, tophat, blackhat, hitmiss, plus erode and dilate) to the same binary image. Display them in a 3×3 grid with clear labels and descriptions.

### Core

**8.4** Build a `text_region_detector(img_gray)` function that uses morphological operations to segment text regions in a scanned document. Strategy: (1) threshold, (2) dilate horizontally (connect letters into words), (3) dilate vertically (connect words into lines), (4) find bounding boxes of connected components. Test on a synthetic document image.

**8.5** Implement `morphological_thinning(binary_img, max_iterations=100)` that iteratively erodes the image until no more pixels can be removed (the skeleton). Use `cv2.erode` with a 3×3 cross SE. Compare to `skimage.morphology.skeletonize`. Show the progression at every 10 iterations.

**8.6** Build an `illumination_corrector(img_gray, se_size=51)` that uses black-hat or top-hat morphology to remove non-uniform illumination and normalize the image for subsequent thresholding. Demonstrate on an image with a strong circular lighting gradient. Compare thresholding quality before and after correction.

### Challenge

**8.7** Implement a **fingerprint enhancement pipeline** using morphological operations: (1) normalize contrast, (2) Gaussian blur, (3) threshold, (4) apply successive morphological open and close to clean up ridges, (5) apply thinning to produce 1-pixel-wide ridges. Use a synthetic fingerprint-like striped image for testing.

**8.8** Build a **license plate localization** step using morphological operations: (1) grayscale + threshold, (2) apply horizontal morphological close to merge characters, (3) find bounding boxes of candidate regions, (4) filter by aspect ratio (license plates are typically 2:1 to 5:1 width:height). Test on synthetic images with text blocks.

**8.9** Implement the **watershed-based cell segmentation pipeline** from Section 8.9 on a real microscopy image (download from a public dataset such as BBBC). Extend it to: (a) estimate cell roundness (4π·area / perimeter²), (b) classify cells as "round" (roundness > 0.7) vs "elongated", (c) visualize the two classes with different colors, (d) generate a summary statistics table.

---

## Further Reading

- **OpenCV morphological transformations tutorial:** https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- **Serra, J. (1982). "Image Analysis and Mathematical Morphology"** — the foundational textbook
- **Soille, P. (2003). "Morphological Image Analysis"** (2nd ed.) — comprehensive reference with applications
- **PyImageSearch morphological operations guide:** https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/
- **Watershed algorithm for touching objects:** https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
