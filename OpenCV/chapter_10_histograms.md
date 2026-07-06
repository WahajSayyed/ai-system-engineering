# Chapter 10: Histograms & Histogram Equalization

> *"The histogram is the image's identity card. Before you process anything, read the histogram — it tells you what the image contains and why your algorithm might fail."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain what an image histogram is and how to interpret it diagnostically
2. Compute 1D, 2D, and multi-channel histograms using `cv2.calcHist`
3. Normalize, compare, and visualize histograms
4. Apply global histogram equalization using `cv2.equalizeHist` and understand the CDF method
5. Apply CLAHE (`cv2.createCLAHE`) and tune its parameters correctly
6. Compute and use histogram backprojection (`cv2.calcBackProject`) for object tracking
7. Compare histograms with all four OpenCV comparison metrics
8. Build a complete image retrieval system using histogram signatures
9. Apply histogram-based techniques to color images via LAB and HSV color spaces

---

## 10.1 What Is a Histogram?

A **histogram** is a function that counts how many pixels in an image have each possible intensity value. For a uint8 grayscale image, there are 256 possible values (0–255), and the histogram `H[v]` counts how many pixels have value `v`.

The histogram is a **projection** — it discards all spatial information and retains only the global distribution of values. This is both its strength (robust to position changes, translations, small distortions) and its limitation (two images with identical histograms can look completely different).

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_histogram(img_gray, title=""):
    """Display an image alongside its histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Image{': ' + title if title else ''}")
    axes[0].axis("off")

    hist = np.bincount(img_gray.ravel(), minlength=256)
    axes[1].bar(range(256), hist, color="steelblue", alpha=0.8, width=1)
    axes[1].set_xlabel("Pixel intensity (0=black, 255=white)")
    axes[1].set_ylabel("Number of pixels")
    axes[1].set_title("Grayscale Histogram")
    axes[1].set_xlim([0, 255])
    plt.tight_layout()
    plt.show()


# ── Examples of diagnostic histogram shapes ──────────────────────────────────
def make_diagnostic_images():
    h, w = 200, 300
    imgs = {}

    # 1. Bimodal: dark background + bright objects → ideal for Otsu
    bimodal = np.zeros((h, w), dtype=np.uint8)
    bimodal[:] = 50    # dark background
    cv2.circle(bimodal, (150, 100), 70, 200, -1)   # bright circle
    bimodal += np.random.normal(0, 10, bimodal.shape).astype(np.int8)
    imgs["bimodal"] = np.clip(bimodal, 0, 255).astype(np.uint8)

    # 2. Overexposed: most pixels near 255
    overexp = np.clip(np.random.normal(200, 30, (h, w)), 0, 255).astype(np.uint8)
    imgs["overexposed"] = overexp

    # 3. Underexposed: most pixels near 0
    underexp = np.clip(np.random.normal(55, 25, (h, w)), 0, 255).astype(np.uint8)
    imgs["underexposed"] = underexp

    # 4. Low contrast: all pixels in a narrow range
    low_contrast = np.clip(np.random.normal(128, 10, (h, w)), 0, 255).astype(np.uint8)
    imgs["low_contrast"] = low_contrast

    # 5. High contrast / good exposure
    good = np.clip(np.random.uniform(0, 255, (h, w)), 0, 255).astype(np.uint8)
    imgs["good_contrast"] = good

    return imgs

diag_imgs = make_diagnostic_images()

fig, axes = plt.subplots(2, 5, figsize=(18, 6))
for col, (name, img) in enumerate(diag_imgs.items()):
    hist = np.bincount(img.ravel(), minlength=256)
    axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(name.replace("_", " "), fontsize=9)
    axes[0, col].axis("off")
    axes[1, col].fill_between(range(256), hist, color="steelblue", alpha=0.7)
    axes[1, col].set_xlim([0, 255])
    axes[1, col].set_xticks([0, 128, 255])
plt.suptitle("Histogram Diagnostic Shapes", fontsize=12)
plt.tight_layout(); plt.show()

# ── Reading a histogram ───────────────────────────────────────────────────────
print("Histogram diagnostic guide:")
print("  Bimodal (two peaks)     → threshold-able image (Otsu works)")
print("  All values near 255     → overexposed, use gamma correction")
print("  All values near 0       → underexposed, brighten")
print("  Narrow central cluster  → low contrast, equalize or stretch")
print("  Uniform spread 0-255    → good contrast")
```

---

## 10.2 cv2.calcHist: The Core Function

`cv2.calcHist` is the primary OpenCV histogram function. Its interface is unusual and frequently misunderstood — memorize the argument structure:

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

# cv2.calcHist(images, channels, mask, histSize, ranges, accumulate=False)
#
# images   : LIST of source images, e.g. [img]
# channels : LIST of channel indices, e.g. [0] for blue, [1] green, [2] red
# mask     : binary mask (uint8); None = use entire image
# histSize : LIST of bin counts, e.g. [256] for all values
# ranges   : LIST of range [min, max), e.g. [0, 256]
#
# Return value: np.ndarray of shape (histSize[0], 1) for 1D histogram

# ── 1D grayscale histogram ────────────────────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256])
print(f"Full hist shape: {hist_full.shape}")   # (256, 1)
print(f"Full hist dtype: {hist_full.dtype}")   # float32

# Equivalent using NumPy (faster for simple cases)
hist_np = np.bincount(gray.ravel(), minlength=256).astype(np.float32)
print(f"Match: {np.allclose(hist_full.ravel(), hist_np)}")   # True

# ── Masked histogram (only compute over a region) ─────────────────────────────
mask = np.zeros_like(gray)
mask[50:150, 100:250] = 255   # ROI: rows 50-150, cols 100-250

hist_masked = cv2.calcHist([gray], [0], mask, [256], [0, 256])
n_pixels_roi = mask.sum() // 255
print(f"Masked hist total: {hist_masked.sum():.0f} = {n_pixels_roi} pixels")

# ── Binned histogram (fewer bins = coarser) ───────────────────────────────────
hist_16bins = cv2.calcHist([gray], [0], None, [16], [0, 256])
# Each bin covers 256/16 = 16 intensity values
print(f"16-bin hist shape: {hist_16bins.shape}")  # (16, 1)

# ── Color channel histograms ──────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
colors_bgr = {"blue": 0, "green": 1, "red": 2}
for name, ch_idx in colors_bgr.items():
    hist = cv2.calcHist([img], [ch_idx], None, [256], [0, 256])
    ax.plot(hist, color=name, alpha=0.8, label=f"{name.upper()} channel")
ax.set_xlim([0, 255])
ax.set_xlabel("Pixel intensity"); ax.set_ylabel("Count")
ax.set_title("Per-channel BGR Histogram"); ax.legend()
plt.tight_layout(); plt.show()
```

### 10.2.1 2D Histograms: Two-Channel Joint Distribution

A 2D histogram counts co-occurrences of values in two channels. This is the foundation of histogram backprojection:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

# 2D Hue-Saturation histogram (most useful for color analysis)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# H: 0-179 (180 bins), S: 0-255 (256 bins)
hist_2d = cv2.calcHist(
    [hsv],
    [0, 1],           # channels: Hue (0) and Saturation (1)
    None,             # no mask
    [180, 256],       # bins: 180 for H, 256 for S
    [0, 180, 0, 256]  # ranges: H in [0,180), S in [0,256)
)
print(f"2D hist shape: {hist_2d.shape}")   # (180, 256) — axis 0=H, axis 1=S

# Normalize for display
hist_2d_norm = cv2.normalize(hist_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Input image"); axes[0].axis("off")

# Display 2D histogram: X=Saturation, Y=Hue
axes[1].imshow(hist_2d_norm, origin="lower", aspect="auto",
               extent=[0, 255, 0, 180], cmap="hot")
axes[1].set_xlabel("Saturation (0=gray, 255=vivid)")
axes[1].set_ylabel("Hue (0=red, 90=cyan, 179=red)")
axes[1].set_title("2D Hue-Saturation Histogram")
plt.colorbar(axes[1].images[0], ax=axes[1], label="Count (normalized)")
plt.tight_layout(); plt.show()
```

---

## 10.3 Histogram Normalization and Comparison

### 10.3.1 Normalizing Histograms

Comparison between histograms of images of different sizes requires normalization:

```python
import cv2
import numpy as np

img1 = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img1 is None:
    img1 = np.random.randint(0, 256, (300, 400), dtype=np.uint8)
img2 = cv2.resize(img1, None, fx=0.5, fy=0.5)   # half size, fewer total pixels

hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

print(f"hist1 sum: {hist1.sum():.0f} = {img1.size} pixels")  # full image
print(f"hist2 sum: {hist2.sum():.0f} = {img2.size} pixels")  # 1/4 the pixels

# cv2.normalize(src, dst, alpha, beta, norm_type)
# NORM_MINMAX: scale to [alpha, beta]
# NORM_L1:     normalize so sum = alpha (probability distribution, sum=1)
# NORM_L2:     normalize so L2 norm = alpha
# NORM_INF:    normalize so max = alpha

hist1_norm = cv2.normalize(hist1, None, 0, 1, cv2.NORM_MINMAX)  # scale to [0,1]
hist2_norm = cv2.normalize(hist2, None, 0, 1, cv2.NORM_MINMAX)

hist1_prob = cv2.normalize(hist1, None, 1, 0, cv2.NORM_L1)    # probability (sum=1)
hist2_prob = cv2.normalize(hist2, None, 1, 0, cv2.NORM_L1)

print(f"Probability hist sum: {hist1_prob.sum():.6f}")  # 1.0
```

### 10.3.2 Histogram Comparison with cv2.compareHist

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (300, 400), dtype=np.uint8)

# Create test images for comparison
img_same    = img.copy()
img_brighter = np.clip(img.astype(int) + 50, 0, 255).astype(np.uint8)
img_darker   = np.clip(img.astype(int) - 50, 0, 255).astype(np.uint8)
img_noisy    = np.clip(img.astype(float) + np.random.normal(0, 20, img.shape),
                       0, 255).astype(np.uint8)
img_random   = np.random.randint(0, 256, img.shape, dtype=np.uint8)

def compare_to(h_ref, h_test, name):
    """Compare two histograms using all four metrics."""
    # Normalize to probability distributions first
    h1 = cv2.normalize(h_ref,  None, 1, 0, cv2.NORM_L1)
    h2 = cv2.normalize(h_test, None, 1, 0, cv2.NORM_L1)

    correl    = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)     # -1 to 1, 1=identical
    chi_sq    = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)     # 0 to ∞, 0=identical
    intersect = cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)  # 0 to 1 (normalized), 1=identical
    bhat_dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) # 0 to 1, 0=identical

    print(f"  {name:<20}: Correl={correl:+.4f}  χ²={chi_sq:.4f}  "
          f"Intersect={intersect:.4f}  Bhattacharyya={bhat_dist:.4f}")

h_ref = cv2.calcHist([img], [0], None, [256], [0, 256])

print("Histogram comparison (ref = original):")
print(f"  {'Image':<20}  Correlation(↑=better)  Chi²(↓=better)  "
      f"Intersect(↑=better)  Bhattacharyya(↓=better)")
print("─" * 80)
compare_to(h_ref, cv2.calcHist([img_same],    [0], None, [256], [0,256]), "Same image")
compare_to(h_ref, cv2.calcHist([img_brighter],[0], None, [256], [0,256]), "+50 brighter")
compare_to(h_ref, cv2.calcHist([img_darker],  [0], None, [256], [0,256]), "-50 darker")
compare_to(h_ref, cv2.calcHist([img_noisy],   [0], None, [256], [0,256]), "Gaussian noise")
compare_to(h_ref, cv2.calcHist([img_random],  [0], None, [256], [0,256]), "Random image")
```

**Metric summary:**

| Metric | Perfect Match | Unrelated | Notes |
|--------|--------------|-----------|-------|
| `HISTCMP_CORREL` | 1.0 | ~0 | Like Pearson correlation |
| `HISTCMP_CHISQR` | 0 | Large | Sensitive to bin magnitude differences |
| `HISTCMP_INTERSECT` | 1.0 (normalized) | ~0 | Robust, commonly used |
| `HISTCMP_BHATTACHARYYA` | 0 | ~1 | Related to Bhattacharyya coefficient; best for probability distributions |

---

## 10.4 Histogram Equalization

### 10.4.1 The Mathematics of Equalization

Histogram equalization transforms pixel values so that the output histogram is approximately uniform — every intensity level has roughly the same number of pixels. This maximizes contrast by making full use of the available dynamic range.

The transformation function is derived from the **Cumulative Distribution Function (CDF)** of the histogram:

```
CDF(v) = Σᵢ₌₀ᵛ H[i] / N_pixels   (cumulative probability up to value v)

T(v) = round((L - 1) · CDF(v))   (transform: stretch CDF to [0, L-1])
```

Where L = 256 for uint8. The output pixel value of an input pixel with value `v` becomes `T(v)`.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def equalize_hist_manual(img_gray):
    """
    Manual implementation of histogram equalization.
    Shows the mathematics step by step.
    """
    # Step 1: Compute histogram
    hist = np.bincount(img_gray.ravel(), minlength=256)
    n_pixels = img_gray.size

    # Step 2: Compute CDF (cumulative sum of normalized histogram)
    cdf = hist.cumsum() / n_pixels   # CDF values in [0, 1]

    # Step 3: Create lookup table (LUT)
    # Map each input value v to output T(v) = (L-1) * CDF(v)
    L = 256
    lut = np.round((L - 1) * cdf).astype(np.uint8)

    # Step 4: Apply LUT
    equalized = lut[img_gray]

    return equalized, cdf, lut


# ── Load a low-contrast image ──────────────────────────────────────────────────
img_gray = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    img_gray = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(img_gray, (150, 100), 70, 100, -1)
    cv2.rectangle(img_gray, (20, 130), (120, 180), 80, -1)
    img_gray = np.clip(img_gray.astype(float) + np.random.normal(0,8,img_gray.shape),
                       0, 255).astype(np.uint8)

# Manual equalization
equalized_manual, cdf, lut = equalize_hist_manual(img_gray)

# cv2.equalizeHist — must be grayscale, uint8
equalized_cv2 = cv2.equalizeHist(img_gray)

# Verify they match
print(f"Manual vs cv2 max difference: {np.abs(equalized_manual.astype(int) - equalized_cv2.astype(int)).max()}")
# Should be 0

# ── Visualize: before and after ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

# Row 0: images
axes[0,0].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
axes[0,0].set_title("Original"); axes[0,0].axis("off")
axes[0,1].imshow(equalized_cv2, cmap="gray", vmin=0, vmax=255)
axes[0,1].set_title("Equalized"); axes[0,1].axis("off")
axes[0,2].imshow(lut.reshape(1, -1), cmap="gray", aspect="auto")
axes[0,2].set_title("LUT: T(v) — the mapping function")
axes[0,2].set_xlabel("Input value v"); axes[0,2].set_ylabel("Output T(v)")

# Row 1: histograms and CDF
hist_orig = np.bincount(img_gray.ravel(), minlength=256)
hist_eq   = np.bincount(equalized_cv2.ravel(), minlength=256)

axes[1,0].fill_between(range(256), hist_orig, color="steelblue", alpha=0.7)
axes[1,0].set_title("Original histogram"); axes[1,0].set_xlim([0, 255])

axes[1,1].fill_between(range(256), hist_eq, color="coral", alpha=0.7)
axes[1,1].set_title("Equalized histogram (flat/uniform)")
axes[1,1].set_xlim([0, 255])

axes[1,2].plot(cdf * 255, color="green", linewidth=2)
axes[1,2].set_title("CDF (should become diagonal after equalization)")
axes[1,2].set_xlabel("Input value"); axes[1,2].set_ylabel("Output value")
axes[1,2].set_xlim([0, 255])

plt.tight_layout(); plt.show()
```

### 10.4.2 When Equalization Works and When It Fails

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Case 1: Works — low contrast, narrow histogram
img_good = np.clip(np.random.normal(128, 20, (200, 300)), 0, 255).astype(np.uint8)

# Case 2: Works — dark image
img_dark = np.clip(np.random.normal(50, 15, (200, 300)), 0, 255).astype(np.uint8)

# Case 3: Problematic — non-uniform illumination
img_uneven = np.zeros((200, 300), dtype=np.uint8)
for x in range(300):
    bg = int(x / 300 * 180)
    img_uneven[:, x] = np.clip(np.random.normal(bg, 10, 200), 0, 255)

# Case 4: Problematic — already-wide histogram
img_wide = np.random.randint(0, 256, (200, 300), dtype=np.uint8)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
images = [img_good, img_dark, img_uneven, img_wide]
titles = ["Low contrast", "Dark", "Uneven lighting", "Wide histogram"]

for col, (im, title) in enumerate(zip(images, titles)):
    eq = cv2.equalizeHist(im)
    axes[0, col].imshow(np.hstack([im, eq]), cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(f"{title}\nBefore | After", fontsize=9)
    axes[0, col].axis("off")
    h_b = np.bincount(im.ravel(), minlength=256)
    h_a = np.bincount(eq.ravel(), minlength=256)
    axes[1, col].fill_between(range(256), h_b, alpha=0.5, color="blue", label="Before")
    axes[1, col].fill_between(range(256), h_a, alpha=0.5, color="red",  label="After")
    axes[1, col].legend(fontsize=7); axes[1, col].set_xlim([0, 255])
plt.suptitle("Histogram Equalization: Cases Where It Works and Where It Doesn't", fontsize=11)
plt.tight_layout(); plt.show()
```

---

## 10.5 CLAHE: Contrast Limited Adaptive Histogram Equalization

Global histogram equalization has two weaknesses:
1. It treats the entire image as one unit — non-uniform illumination gets amplified, not corrected
2. Regions with low contrast that were originally "correct" may get over-enhanced, amplifying noise

**CLAHE** (Karel Zuiderveld, 1994) solves both problems by:
1. Dividing the image into small tiles (e.g., 8×8)
2. Applying histogram equalization independently to each tile
3. **Clipping the histogram** at `clipLimit` before computing the CDF — this limits contrast amplification, preventing noise blow-up
4. Using **bilinear interpolation** between neighboring tiles to avoid tile boundary artifacts

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def demo_clahe_parameters(img_gray):
    """Show the effect of CLAHE clipLimit and tileGridSize parameters."""
    # cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
    # clipLimit: histogram clip threshold
    #   Low  (1-2):  very limited amplification, close to original
    #   Moderate (3-8): good contrast, controlled noise
    #   High (40+): strong enhancement but more noise (not recommended)
    # tileGridSize: (cols, rows) of tiles
    #   Small tiles (4×4): very local adaptation, may create "patchy" look
    #   Large tiles (16×16): broader adaptation, smoother result

    clip_limits    = [1.0, 2.0, 5.0, 10.0]
    tile_grids     = [(4,4), (8,8), (16,16)]

    fig, axes = plt.subplots(len(tile_grids), len(clip_limits) + 1, figsize=(15, 9))

    for row, tg in enumerate(tile_grids):
        axes[row, 0].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"Original\n(tile={tg})", fontsize=8)
        axes[row, 0].axis("off")

        for col, cl in enumerate(clip_limits, start=1):
            clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=tg)
            result = clahe.apply(img_gray)
            axes[row, col].imshow(result, cmap="gray", vmin=0, vmax=255)
            axes[row, col].set_title(f"clip={cl}, tile={tg}", fontsize=8)
            axes[row, col].axis("off")

    plt.suptitle("CLAHE Parameter Grid: clipLimit × tileGridSize", fontsize=12)
    plt.tight_layout(); plt.show()


# ── CLAHE on grayscale ────────────────────────────────────────────────────────
img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((300, 400), dtype=np.uint8)
    Y, X = np.mgrid[0:300, 0:400]
    bg = (40 + 80 * np.exp(-((X - 200)**2 + (Y - 150)**2) / (2*100**2))).astype(np.uint8)
    img = bg.copy()
    for y in range(50, 260, 30):
        cv2.line(img, (30, y), (370, y), 20, 2)

# Global equalization
global_eq = cv2.equalizeHist(img)

# CLAHE (recommended defaults: clipLimit=2.0, tileGridSize=(8,8))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(img)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, im, title in zip(axes,
    [img, global_eq, clahe_result],
    ["Original", "Global Equalization\n(may over-enhance)", "CLAHE (clipLimit=2, tile=8×8)"]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()

demo_clahe_parameters(img)

# ── CLAHE on color images — CORRECT approach ─────────────────────────────────
# NEVER apply CLAHE to all 3 BGR channels independently — this shifts colors
# CORRECT: apply only to the luminance channel

def clahe_color_bgr(img_bgr, clip_limit=2.0, tile_size=(8, 8)):
    """
    Apply CLAHE to a color image correctly by working on the L channel in LAB.

    Returns contrast-enhanced BGR image with colors preserved.
    """
    # Convert to LAB (L = perceptual lightness, a/b = color)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # Apply CLAHE only to L (luminance)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    L_enhanced = clahe.apply(L)

    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([L_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


img_color = cv2.imread("sample.jpg")
if img_color is None:
    img_color = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)

# Wrong approach: CLAHE on each BGR channel independently (shifts colors)
def clahe_color_wrong(img_bgr):
    channels = list(cv2.split(img_bgr))
    clahe = cv2.createCLAHE(clipLimit=2.0)
    return cv2.merge([clahe.apply(ch) for ch in channels])

correct = clahe_color_bgr(img_color)
wrong   = clahe_color_wrong(img_color)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, im, title in zip(axes,
    [img_color, correct, wrong],
    ["Original", "CLAHE via LAB\n(colors preserved)", "CLAHE on BGR channels\n(color shift!)"]):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 10.6 Histogram Backprojection

Histogram backprojection answers the question: **"Where in this image are pixels that look like the pixels in this reference sample?"** Given a reference object and its histogram, backprojection creates a probability map where each pixel's value reflects how likely it is to belong to the reference object.

### 10.6.1 How Backprojection Works

```
1. Build a model histogram H_model from a sample of the target object
   (typically the Hue-Saturation histogram in HSV space)

2. For each pixel p in the query image:
   - Find the histogram bin that p belongs to
   - Set backprojection(p) = H_model[bin]

3. Normalize result to [0, 255]

4. Smooth the result (optional, helps with noise)
5. Threshold to get a binary mask
```

The result is a grayscale image where bright pixels have color statistics matching the target object.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_backprojection(sample_bgr, target_bgr,
                              h_bins=180, s_bins=256,
                              smooth_kernel=5):
    """
    Find pixels in target that look like the sample, using 2D HS histogram.

    Parameters
    ----------
    sample_bgr : np.ndarray   Sample/reference image of the target object
    target_bgr : np.ndarray   Full scene image to search in
    h_bins : int              Number of Hue bins
    s_bins : int              Number of Saturation bins
    smooth_kernel : int       Kernel size for post-smoothing (odd)

    Returns
    -------
    probability_map : np.ndarray (H, W) uint8, 0-255
    binary_mask     : np.ndarray (H, W) uint8, 0 or 255
    """
    # Convert to HSV
    sample_hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)

    # Build 2D Hue-Saturation model histogram from sample
    hist_model = cv2.calcHist(
        [sample_hsv],
        [0, 1],           # H and S channels
        None,
        [h_bins, s_bins],
        [0, 180, 0, 256]
    )

    # Normalize histogram to [0, 255] (scale counts to probability-like values)
    cv2.normalize(hist_model, hist_model, 0, 255, cv2.NORM_MINMAX)

    # Backproject: for each pixel in target, look up its H,S bin in hist_model
    prob_map = cv2.calcBackProject(
        [target_hsv],       # image to search in
        [0, 1],             # H and S channels
        hist_model,         # model histogram
        [0, 180, 0, 256],   # ranges matching histogram ranges
        scale=1             # scale factor (1 = direct lookup)
    )
    # prob_map shape: (H, W) uint8 — each pixel = hist_model[H_bin, S_bin]

    # Post-smooth (optional): helps merge scattered pixels into coherent regions
    if smooth_kernel > 1:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (smooth_kernel, smooth_kernel))
        cv2.filter2D(prob_map, -1, se, prob_map)   # in-place convolution

    # Threshold to binary mask
    _, binary = cv2.threshold(prob_map, 50, 255, cv2.THRESH_BINARY)

    return prob_map, binary


# ── Demo: find skin-toned regions in a scene ─────────────────────────────────
# Create a synthetic "scene" and "sample"
scene = np.zeros((200, 400, 3), dtype=np.uint8)
# Add skin-tone regions (BGR approx)
cv2.circle(scene, (100, 100), 60, [100, 130, 190], -1)   # skin-tone face
cv2.ellipse(scene, (300, 150), (60, 40), 0, 0, 360, [0, 200, 0], -1)  # non-skin green

# Sample: just the skin-tone region
sample = np.full((50, 50, 3), [100, 130, 190], dtype=np.uint8)
noise = np.random.normal(0, 10, sample.shape).astype(np.int8)
sample = np.clip(sample.astype(int) + noise, 0, 255).astype(np.uint8)

prob_map, binary = histogram_backprojection(sample, scene)

result = cv2.bitwise_and(scene, scene, mask=binary)

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, im, title in zip(axes,
    [sample, scene, prob_map, binary, result],
    ["Sample\n(reference object)", "Scene\n(search target)",
     "Probability map\n(bright = matches sample)",
     "Binary mask\n(thresholded)", "Result\n(object found)"]):
    if im.ndim == 3:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 10.7 Histogram-Based Image Retrieval

One of the most practical applications of histograms: given a query image, find the most similar images in a database. Histograms serve as compact image signatures:

```python
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


@dataclass
class ImageRecord:
    """A record in the image database."""
    name:      str
    image:     np.ndarray    # BGR
    signature: np.ndarray    # histogram signature


def compute_signature(img_bgr: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """
    Compute a compact histogram signature for an image.

    Strategy: compute per-channel histograms in HSV space (H, S, V),
    then concatenate into a single feature vector.
    Using 32 bins per channel = 96-dimensional vector.

    HSV is preferable to BGR because:
    - H channel is rotation-invariant within color
    - S and V separate color from brightness
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sig = []
    for ch, n, rng in [(0, n_bins, [0, 180]),  # Hue
                       (1, n_bins, [0, 256]),  # Saturation
                       (2, n_bins, [0, 256])]: # Value
        h = cv2.calcHist([hsv], [ch], None, [n], rng)
        cv2.normalize(h, h, 1, 0, cv2.NORM_L1)  # probability
        sig.append(h.ravel())
    return np.concatenate(sig)


class ImageDatabase:
    """Simple histogram-based image retrieval system."""

    def __init__(self, n_bins: int = 32):
        self.n_bins  = n_bins
        self.records: List[ImageRecord] = []

    def add(self, name: str, img_bgr: np.ndarray):
        sig = compute_signature(img_bgr, self.n_bins)
        self.records.append(ImageRecord(name=name, image=img_bgr, signature=sig))

    def query(self, query_img: np.ndarray, top_k: int = 5,
              metric: int = cv2.HISTCMP_BHATTACHARYYA
              ) -> List[Tuple[float, ImageRecord]]:
        """
        Find the top_k most similar images to query_img.

        Returns list of (distance, record) sorted by distance ascending.
        For BHATTACHARYYA: lower = more similar.
        For CORREL: higher = more similar.
        """
        q_sig = compute_signature(query_img, self.n_bins)
        # Reshape for compareHist: needs (N, 1)
        q_h = q_sig.reshape(-1, 1).astype(np.float32)

        distances = []
        for rec in self.records:
            r_h = rec.signature.reshape(-1, 1).astype(np.float32)
            dist = cv2.compareHist(q_h, r_h, metric)
            distances.append((dist, rec))

        # Sort: ascending for BHATTACHARYYA/chi-sq, descending for CORREL/intersect
        reverse = (metric in [cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT])
        distances.sort(key=lambda x: x[0], reverse=reverse)

        return distances[:top_k]


# ── Demo: Build and query a synthetic database ────────────────────────────────
def make_colored_image(color_bgr, shape=(100, 100)):
    """Create a solid-color image with slight noise."""
    img = np.full((*shape, 3), color_bgr, dtype=np.uint8)
    noise = np.random.normal(0, 15, img.shape).astype(np.int8)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

rng = np.random.default_rng(42)

# Create 10 images of 5 colors (2 each)
colors = {
    "red1":    [30, 30, 200], "red2":    [25, 25, 210],
    "blue1":   [200, 50, 30], "blue2":   [210, 55, 25],
    "green1":  [30, 200, 50], "green2":  [25, 210, 45],
    "yellow1": [20, 210, 220],"yellow2": [15, 205, 225],
    "purple1": [200, 30, 130],"purple2": [205, 25, 135],
}

db = ImageDatabase(n_bins=32)
for name, color in colors.items():
    db.add(name, make_colored_image(color))

# Query: a new red image (should match red1 and red2)
query = make_colored_image([28, 28, 205])
results = db.query(query, top_k=5, metric=cv2.HISTCMP_BHATTACHARYYA)

print("Query: new red image")
print("Top 5 results (lower Bhattacharyya = more similar):")
for dist, rec in results:
    print(f"  {rec.name:<12}: Bhattacharyya distance = {dist:.4f}")

# ── Visualization ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 6, figsize=(16, 3))
axes[0].imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
axes[0].set_title("Query", fontweight="bold"); axes[0].axis("off")
for i, (dist, rec) in enumerate(results, 1):
    axes[i].imshow(cv2.cvtColor(rec.image, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"#{i}: {rec.name}\nd={dist:.3f}", fontsize=8)
    axes[i].axis("off")
plt.suptitle("Histogram-Based Image Retrieval", fontsize=12)
plt.tight_layout(); plt.show()
```

---

## 10.8 Summary

**Histograms** capture the global distribution of pixel intensities. `cv2.calcHist` handles 1D, 2D, and masked histograms with a flexible but unusual API (all arguments in lists). Always normalize before comparing histograms across images of different sizes.

**Histogram comparison** with `cv2.compareHist` offers four metrics: Correlation (↑), Chi-square (↓), Intersection (↑), Bhattacharyya (↓). Bhattacharyya on normalized (probability) histograms is the most reliable for image matching.

**Histogram equalization** (`cv2.equalizeHist`) applies the CDF transformation to redistribute intensities uniformly. Effective for globally underexposed or low-contrast images. Fails on non-uniform illumination.

**CLAHE** (`cv2.createCLAHE`) fixes equalization's limitations by processing local tiles with contrast limiting. For color images, always apply CLAHE to the L (or V) channel only, not to all three channels — use LAB or HSV color space.

**Histogram backprojection** (`cv2.calcBackProject`) creates a probability map showing where target-like pixels occur. Use 2D Hue-Saturation histograms in HSV space for robust color-based segmentation and tracking.

---

## 10.9 Exercises

### Warm-up

**10.1** Write a function `histogram_diagnostics(img_gray)` that computes and prints: total pixels, mean, std, 5th and 95th percentile, fraction below 30 ("dark"), fraction above 225 ("bright"), and a diagnosis string ("overexposed", "underexposed", "low contrast", or "good"). Test on 5 different images.

**10.2** Implement `histogram_stretch(img_gray, low_pct=2, high_pct=98)` that stretches the histogram by clipping at given percentiles and remapping to [0, 255]. This is "contrast stretching" — simpler than equalization but often sufficient. Compare to `cv2.equalizeHist` on three test images.

**10.3** Plot the CDF of a grayscale image before and after histogram equalization. Verify that after equalization, the CDF is approximately a straight diagonal line (uniform distribution). Show both CDFs on the same axes.

### Core

**10.4** Build a `multi_histogram_visualizer(img_bgr)` that shows the image alongside: (a) individual BGR channel histograms, (b) HSV channel histograms with correct axis labels (H: 0–179, S/V: 0–255), (c) the 2D Hue-Saturation histogram as a heatmap. All in a single figure.

**10.5** Apply CLAHE to a real (or realistic synthetic) medical-style grayscale image (e.g., a simulated X-ray: dark background, structures of varying brightness). Compare global equalization, CLAHE with clipLimit=2, and CLAHE with clipLimit=10. Use SSIM (structural similarity — `from skimage.metrics import structural_similarity`) to measure which preserves original structure best while improving contrast.

**10.6** Implement a `color_histogram_backprojector` class that: (a) takes a sample image and builds a 2D HS model, (b) on each call to `locate(scene)` applies backprojection, smoothing, and thresholding, and (c) returns the probability map, binary mask, and the center of mass of the detected region. Test with a colored ball sample and a scene containing that ball.

**10.7** Implement histogram-based image retrieval on a real dataset: create 20 images divided into 4 categories (5 per category) by varying hue. Build a database and query with all 20 images. Compute retrieval precision@1, precision@3, and mean average precision (mAP).

### Challenge

**10.8** Implement **histogram specification** (also called histogram matching): given a source image and a target image, transform the source image so its histogram matches the target's histogram. This generalizes equalization (which targets a uniform histogram). Use the inverse CDF approach: build source CDF, build target CDF, compose them to get the mapping LUT. Demonstrate on: same image → different brightness target; natural photograph → artistic (high-contrast) target.

**10.9** Build a **real-time color tracker** using histogram backprojection and `cv2.CamShift`: (a) let the user click-and-drag to select a target object, (b) compute the HS histogram of the selection, (c) on each new frame, run backprojection and `cv2.CamShift` to find the new location, (d) draw the rotated bounding box on the frame. Test with a video file or webcam.

**10.10** Implement **earth mover's distance (EMD)** as a histogram similarity metric using `cv2.EMD`. Compare EMD to Bhattacharyya on 20 image pairs (10 similar, 10 dissimilar). EMD considers the "cost of transportation" between histogram bins, making it more sensitive to perceptually similar but quantized differently histograms. Show where EMD outperforms simpler metrics.

---

## Further Reading

- **OpenCV histogram tutorial:** https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
- **OpenCV CLAHE documentation:** https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- **Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"** in Graphics Gems IV — original CLAHE paper
- **Swain, M. & Ballard, D. (1991). "Color Indexing"** IJCV — original histogram backprojection paper
- **PyImageSearch CLAHE guide:** https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
