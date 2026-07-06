# Chapter 6: Filtering & Smoothing

> *"Every image processing algorithm either removes information (to reduce noise) or enhances information (to reveal structure). Filtering is the primitive operation from which both arise."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain 2D convolution mathematically and implement it manually with `cv2.filter2D`
2. Understand the difference between correlation and convolution
3. Design custom kernels for sharpening, embossing, and edge enhancement
4. Apply box blur, Gaussian blur, and understand their noise-reduction properties
5. Explain when and why median blur outperforms Gaussian blur for salt-and-pepper noise
6. Implement and understand bilateral filtering for edge-preserving smoothing
7. Use separable filters for computational efficiency
8. Handle border effects correctly with `cv2.copyMakeBorder`
9. Build an adaptive noise reduction pipeline
10. Benchmark all filters and understand the speed/quality tradeoff

---

## 6.1 Convolution: The Mathematical Foundation

### 6.1.1 What Is Convolution?

At its core, spatial filtering is a sliding-window computation. For each output pixel, you take a weighted sum of a neighborhood of input pixels, where the weights are defined by a **kernel** (also called a filter, mask, or structuring element).

Mathematically, 2D convolution of image `I` with kernel `K` is:

```
(I * K)[y, x] = Σᵢ Σⱼ I[y + i, x + j] · K[i, j]
```

where the sum is over all kernel indices (i, j). The kernel "slides" over every pixel in the image.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ── Manual convolution demonstration on a tiny image ─────────────────────────
# This shows exactly what OpenCV does internally

def manual_conv2d(img, kernel, padding="zero"):
    """
    Manual implementation of 2D convolution (educational).
    For real use, always use cv2.filter2D.
    """
    img_f = img.astype(np.float64)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad the image
    if padding == "zero":
        padded = np.pad(img_f, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    elif padding == "reflect":
        padded = np.pad(img_f, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    else:
        padded = np.pad(img_f, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

    h, w = img_f.shape
    output = np.zeros_like(img_f)

    # Slide the kernel over every pixel
    for y in range(h):
        for x in range(w):
            region = padded[y:y + kh, x:x + kw]
            output[y, x] = np.sum(region * kernel)

    return output


# Test: apply a simple 3×3 averaging kernel
img_gray = np.array([
    [10,  20,  30,  40,  50],
    [60,  70,  80,  90, 100],
    [110,120, 130, 140, 150],
    [160,170, 180, 190, 200],
    [210,220, 230, 240, 250],
], dtype=np.uint8)

kernel_avg = np.ones((3, 3), dtype=np.float64) / 9.0
result_manual = manual_conv2d(img_gray.astype(np.float64), kernel_avg)
print(f"Manual convolution at center [2,2]: {result_manual[2,2]:.2f}")
# = mean of 3×3 neighborhood around (2,2) = mean([70..190]) = 130.0 ✓

# Compare to OpenCV
result_cv2 = cv2.filter2D(img_gray, -1, kernel_avg.astype(np.float32))
print(f"cv2.filter2D at center [2,2]: {result_cv2[2,2]:.2f}")
# Should match (minor float differences)
```

### 6.1.2 Convolution vs Correlation

There is an important distinction that causes confusion:

- **Correlation** slides the kernel *as-is* over the image: `(I ⊛ K)[y,x] = Σ I[y+i, x+j] · K[i,j]`
- **Convolution** flips the kernel 180° first: `(I * K)[y,x] = Σ I[y+i, x+j] · K[-i,-j]`

For symmetric kernels (Gaussian, box blur), convolution and correlation give identical results. For asymmetric kernels (directional edge detectors, Sobel), they differ.

**OpenCV's `cv2.filter2D` performs correlation, not convolution.** The function is named "filter2D" rather than "convolve" for this reason. If you need true convolution with an asymmetric kernel, flip it first with `np.flip(kernel)` or `cv2.flip(kernel, -1)`.

```python
import cv2
import numpy as np

# Asymmetric kernel to demonstrate the difference
kernel_asym = np.array([
    [1, 2, 3],
    [0, 0, 0],
    [0, 0, 0]
], dtype=np.float32) / 6.0

img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# Correlation (what cv2.filter2D does)
result_corr = cv2.filter2D(img, -1, kernel_asym)

# True convolution: flip kernel 180° then correlate
kernel_flipped = cv2.flip(kernel_asym, -1)   # flip both axes
result_conv    = cv2.filter2D(img, -1, kernel_flipped)

# For symmetric kernels, correlation == convolution
kernel_sym = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16.0
r_corr = cv2.filter2D(img, -1, kernel_sym)
r_conv = cv2.filter2D(img, -1, cv2.flip(kernel_sym, -1))
print(f"Symmetric kernel: max diff = {np.abs(r_corr.astype(int) - r_conv.astype(int)).max()}")
# → 0 (identical)
```

### 6.1.3 cv2.filter2D: The Universal Kernel Filter

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_gray(path="sample.jpg", fallback_size=(300, 400)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        rng = np.random.default_rng(42)
        img = (rng.random(fallback_size) * 255).astype(np.uint8)
        cv2.putText(img, "Sample", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, 200, 3, cv2.LINE_AA)
    return img

img = load_gray()

# cv2.filter2D(src, ddepth, kernel, anchor=(-1,-1), delta=0, borderType=BORDER_REFLECT_101)
# ddepth = -1 → output has same depth as source
# anchor = (-1,-1) → kernel center
# delta  = value added to result (rarely needed)

# ── Identity kernel (no change) ───────────────────────────────────────────────
K_identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)
identity_result = cv2.filter2D(img, -1, K_identity)
assert np.array_equal(img, identity_result)

# ── Custom kernels ────────────────────────────────────────────────────────────
# Sharpening kernel: identity + negative Laplacian
K_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)
sharpened = cv2.filter2D(img, -1, K_sharpen)

# Edge detection (approximation)
K_edge = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)
edges = cv2.filter2D(img, cv2.CV_32F, K_edge)
edges_vis = cv2.convertScaleAbs(edges)

# Emboss effect
K_emboss = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]
], dtype=np.float32)
embossed = cv2.filter2D(img, cv2.CV_32F, K_emboss)
# Normalize to [0, 255]: add 128 to shift from [-255,255] to centered at 128
embossed_vis = np.clip(embossed + 128, 0, 255).astype(np.uint8)

# Horizontal motion blur (5×1 kernel)
K_motion_h = np.ones((1, 15), dtype=np.float32) / 15
motion_blur = cv2.filter2D(img, -1, K_motion_h)

# Box blur (manually, to understand what cv2.blur does internally)
K_box = np.ones((7, 7), dtype=np.float32) / 49
box_blur_manual = cv2.filter2D(img, -1, K_box)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, im, title in zip(axes.flat,
    [img, sharpened, edges_vis, embossed_vis, motion_blur, box_blur_manual],
    ["Original", "Sharpen (Laplacian+identity)",
     "Edge detection (8-neighbor)", "Emboss (offset Laplacian)",
     "Motion blur (H, k=15)", "Box blur (7×7)"]):
    ax.imshow(im, cmap="gray"); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

### 6.1.4 Kernel Design Principles

Understanding these rules lets you design kernels from scratch:

```python
import numpy as np

# ── Rule 1: Sum to 1 → no brightness change ──────────────────────────────────
K_blur   = np.ones((5, 5)) / 25         # sum=1 ✓ — brightness preserved
K_edge   = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])  # sum=0 ✓ — zero response to constant

# ── Rule 2: Sum to 0 → zero response to uniform regions ──────────────────────
# Edge kernels should sum to 0: they respond to intensity change, not level
print(f"Edge kernel sum: {K_edge.sum()}")   # 0

# ── Rule 3: Kernel separability → separable filters are 5-10× faster ─────────
# A 2D kernel K is separable if K = u * v^T (outer product of two 1D vectors)
# Example: Gaussian kernel is separable

def is_separable(kernel, tol=1e-8):
    """Check if a 2D kernel is separable (rank-1 matrix)."""
    U, s, Vt = np.linalg.svd(kernel)
    # Separable iff only one non-zero singular value
    return s[1] < tol * s[0]

K_gaussian_2d = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0

K_edge_laplacian = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float32)

print(f"Gaussian 3×3 is separable: {is_separable(K_gaussian_2d)}")      # True
print(f"Laplacian is separable:    {is_separable(K_edge_laplacian)}")   # False

# Gaussian separable decomposition:
# K_gauss = [1,2,1]^T * [1,2,1] / 16
v_gauss = np.array([1, 2, 1], dtype=np.float32)[:, np.newaxis] / 4.0
u_gauss = v_gauss.T
K_gauss_reconstructed = v_gauss @ u_gauss
print(f"Reconstruction error: {np.abs(K_gaussian_2d - K_gauss_reconstructed).max():.8f}")
```

---

## 6.2 Border Handling

Every filter that requires a neighborhood around a pixel faces a problem at the image edges: there are no pixels outside the boundary. OpenCV provides several border padding strategies:

```python
import cv2
import numpy as np

img = np.array([[10, 20, 30, 40],
                [50, 60, 70, 80],
                [90,100,110,120],
                [130,140,150,160]], dtype=np.uint8)

# cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
border_size = 2

borders = {}
for btype, name in [
    (cv2.BORDER_CONSTANT,    "CONSTANT (0)"),
    (cv2.BORDER_REPLICATE,   "REPLICATE"),
    (cv2.BORDER_REFLECT,     "REFLECT (dcba|abcd)"),
    (cv2.BORDER_REFLECT_101, "REFLECT_101 (dcb|abcd)"),   # default for most filters
    (cv2.BORDER_WRAP,        "WRAP"),
]:
    padded = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                 btype, value=0)
    borders[name] = padded

print("REPLICATE border (stretches edge pixels):")
print(borders["REPLICATE"])
# Top rows: [10 10 10 20 30 40 40 40]

print("\nREFLECT_101 border (default, doesn't repeat edge pixel):")
print(borders["REFLECT_101"])
# Top rows: [30 20 10 20 30 40 30 20]   (reflects but skips the edge pixel itself)

# Practical impact: REFLECT_101 avoids the "doubling" artifact at edges
# Use BORDER_REFLECT_101 for Gaussian/bilateral filtering
# Use BORDER_CONSTANT for perspective correction (black borders look correct)
# Use BORDER_REPLICATE when you can't afford dark artifacts (e.g., portrait blur)
```

---

## 6.3 Box Blur and Averaging Filters

The box blur is the simplest possible smoothing filter: it replaces each pixel with the unweighted average of its neighborhood.

### 6.3.1 cv2.blur and cv2.boxFilter

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

# cv2.blur: simple box filter (normalized)
box3  = cv2.blur(img, (3, 3))   # 3×3 averaging kernel
box7  = cv2.blur(img, (7, 7))   # 7×7 averaging kernel
box15 = cv2.blur(img, (15, 15)) # 15×15 averaging kernel

# cv2.boxFilter: same but with more control
# normalize=True: divide by ksize.area (same as cv2.blur)
# normalize=False: compute unnormalized sum (useful for integral computations)
box_norm  = cv2.boxFilter(img, -1, (7, 7), normalize=True)
box_unnorm= cv2.boxFilter(img, cv2.CV_32F, (7, 7), normalize=False)  # float output

print(f"Normalized box range:   [{box_norm.min()}, {box_norm.max()}]")
print(f"Unnormalized box range: [{box_unnorm.min():.0f}, {box_unnorm.max():.0f}]")
# Unnormalized values are 49× larger (49 = 7×7 pixels summed)

# ── Properties of the box filter ─────────────────────────────────────────────
# Pros:  Fast (implementable via integral images in O(1) per pixel regardless of k)
# Cons:  Ringing artifacts, does NOT model camera optics or human vision blur
# Cons:  The rectangular kernel causes a "boxy" frequency response (sinc-like)
# Use for: fast downscaling preview, integral-image-based operations

# ── Box filter via integral image (understanding the O(1) trick) ──────────────
# cv2.integral computes the summed area table
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
integral = cv2.integral(img_gray)   # shape: (H+1, W+1)

def box_filter_integral(integral_img, x, y, ksize):
    """Compute box filter output at (x, y) using integral image — O(1)."""
    k = ksize // 2
    x1, y1 = max(0, x - k), max(0, y - k)
    x2, y2 = min(integral_img.shape[1]-1, x + k + 1), min(integral_img.shape[0]-1, y + k + 1)
    area = (x2 - x1) * (y2 - y1)
    total = (integral_img[y2, x2] - integral_img[y1, x2]
             - integral_img[y2, x1] + integral_img[y1, x1])
    return total / area if area > 0 else 0.0

# Test at position (50, 50) with k=7
val_integral = box_filter_integral(integral, 50, 50, 7)
val_direct   = float(box7[50, 50].mean() if img.ndim == 3 else box7[50, 50])
print(f"Integral image result at (50,50): {val_integral:.2f}")
```

---

## 6.4 Gaussian Blur

Gaussian blur is the gold standard of spatial smoothing. It models camera defocus, pre-processes images for edge detection, and is the foundation of scale-space theory.

### 6.4.1 The Gaussian Function

The 2D Gaussian kernel with standard deviation σ is:

```
G(x, y) = (1 / (2πσ²)) · exp(-(x² + y²) / (2σ²))
```

This function:
- Is maximal at the center (x=0, y=0)
- Decays smoothly in all directions
- Is isotropic (circular symmetry)
- Is separable: G(x,y) = G_1D(x) · G_1D(y)

The separability is key for performance: a 2D Gaussian of size k×k can be applied as two 1D passes of size k×1 and 1×k, reducing the number of multiply-adds from k² to 2k.

### 6.4.2 cv2.GaussianBlur

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg")
if img is None:
    rng = np.random.default_rng(42)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 150), 100, (0, 200, 220), -1)
    cv2.rectangle(img, (50, 50), (150, 250), (180, 80, 20), -1)

# cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=0, borderType=BORDER_REFLECT_101)
# ksize: MUST be odd and positive, e.g. (3,3), (5,5), (7,7), (11,11)
# sigmaX: standard deviation in X direction
# sigmaY: if 0, uses sigmaX (isotropic Gaussian)
# If sigmaX=0 AND ksize given, sigma is computed from ksize via:
#   sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

blurs = {
    "Original"        : img,
    "σ=1 (k=5)"      : cv2.GaussianBlur(img, (5,  5),  1.0),
    "σ=2 (k=11)"     : cv2.GaussianBlur(img, (11, 11), 2.0),
    "σ=5 (k=21)"     : cv2.GaussianBlur(img, (21, 21), 5.0),
    "σ=10 (k=41)"    : cv2.GaussianBlur(img, (41, 41), 10.0),
    "Anisotropic σx=2,σy=8": cv2.GaussianBlur(img, (11, 41), 2.0, 8.0),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (title, im) in zip(axes.flat, blurs.items()):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(title); ax.axis("off")
plt.suptitle("Gaussian Blur: Effect of σ", fontsize=13)
plt.tight_layout(); plt.show()

# ── The sigma-ksize relationship ──────────────────────────────────────────────
# Rule: ksize ≈ 6σ (capture 99.7% of Gaussian energy, 3σ on each side)
# In practice: ksize = int(6 * sigma) | 1   (make odd)
def sigma_to_ksize(sigma):
    k = int(6 * sigma)
    return k if k % 2 == 1 else k + 1

for sigma in [0.5, 1.0, 2.0, 3.0, 5.0]:
    k = sigma_to_ksize(sigma)
    print(f"  σ={sigma:.1f} → ksize={k}")
# σ=0.5 → ksize=3
# σ=1.0 → ksize=7  (note: OpenCV uses ksize=5 for sigma=1 in many tutorials)
# σ=2.0 → ksize=13
# σ=3.0 → ksize=19
# σ=5.0 → ksize=31

# ── Getting the actual Gaussian kernel OpenCV uses ─────────────────────────────
kernel_1d = cv2.getGaussianKernel(ksize=9, sigma=2.0, ktype=cv2.CV_64F)
kernel_2d = kernel_1d @ kernel_1d.T   # outer product → 2D kernel
print(f"\nGaussian kernel (9×9, σ=2):\n{kernel_2d.round(4)}")
print(f"Sum = {kernel_2d.sum():.8f}")   # should be 1.0
```

### 6.4.3 Gaussian Blur for Noise Reduction

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add controlled Gaussian noise to an image
def add_gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

img_clean = cv2.imread("sample.jpg")
if img_clean is None:
    img_clean = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.circle(img_clean, (150, 100), 60, (0, 180, 220), -1)

img_noisy = add_gaussian_noise(img_clean, std=25)

# Compare denoising with different sigma values
denoised = {
    "Noisy (σ_noise=25)": img_noisy,
    "Gaussian σ=0.5"    : cv2.GaussianBlur(img_noisy, (3,3),  0.5),
    "Gaussian σ=1.5"    : cv2.GaussianBlur(img_noisy, (9,9),  1.5),
    "Gaussian σ=3.0"    : cv2.GaussianBlur(img_noisy, (19,19),3.0),
    "Gaussian σ=6.0"    : cv2.GaussianBlur(img_noisy, (37,37),6.0),
}

# Measure PSNR (Peak Signal-to-Noise Ratio) for each
def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255**2 / mse)

print("PSNR comparison (higher is better):")
for name, img_d in denoised.items():
    p = psnr(img_clean, img_d)
    print(f"  {name:<30}: {p:.2f} dB")
# Optimal sigma balances noise removal vs edge blurring
# → there is an optimal sigma that maximizes PSNR
```

### 6.4.4 Scale Space: Gaussian Pyramid (Preview)

One of the most important applications of Gaussian blur is building scale-space representations. The **Gaussian pyramid** blurs and downsamples successively:

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

# Build Gaussian pyramid (5 levels)
pyramid = [img]
for level in range(4):
    # pyrDown = Gaussian blur then downsample by 2
    pyramid.append(cv2.pyrDown(pyramid[-1]))

for i, level_img in enumerate(pyramid):
    print(f"Level {i}: {level_img.shape}")
# Level 0: (256, 256, 3)
# Level 1: (128, 128, 3)
# Level 2: (64,  64,  3)
# Level 3: (32,  32,  3)
# Level 4: (16,  16,  3)

# Equivalent to:
# level_1 = cv2.GaussianBlur(img, (5,5), 1.0)[::2, ::2]   # blur then subsample
```

---

## 6.5 Median Blur

The median filter replaces each pixel with the **median** of its neighborhood — the value that splits the sorted list of neighborhood values in half. Unlike the box or Gaussian filters, the median is a non-linear operation.

### 6.5.1 Why Median Is Special

The median filter has two critically important properties:

1. **Salt-and-pepper noise immunity:** A salt pixel (value=255) in a small neighborhood cannot affect the median unless more than half the neighborhood is corrupted. In contrast, even a single outlier pixel can dramatically shift a mean.

2. **Edge preservation:** If the filter window straddles an edge (e.g., half the pixels are 10, half are 200), the median lands at either 10 or 200 — it cannot produce intermediate blended values. This is why median filtering preserves edges that Gaussian blur destroys.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Demonstrate salt-and-pepper vs Gaussian noise ─────────────────────────────
img_clean = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img_clean is None:
    img_clean = np.zeros((200, 300), dtype=np.uint8)
    for y in range(20, 180, 20):
        cv2.line(img_clean, (20, y), (280, y), 200, 2)

# Add salt-and-pepper noise
def add_salt_pepper(img, density=0.05):
    """density: fraction of pixels corrupted."""
    noisy = img.copy()
    n = int(density * img.size)
    # Salt
    coords = [np.random.randint(0, d, n//2) for d in img.shape]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, d, n//2) for d in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

img_sp = add_salt_pepper(img_clean, density=0.10)

# cv2.medianBlur(src, ksize)
# ksize: MUST be odd positive integer (1, 3, 5, 7, ...)
# Input can be uint8 (any channels) or float32 (single channel only for large k)
median3 = cv2.medianBlur(img_sp, 3)
median5 = cv2.medianBlur(img_sp, 5)
median7 = cv2.medianBlur(img_sp, 7)

# Compare: Gaussian also applied
gauss5 = cv2.GaussianBlur(img_sp, (5, 5), 1.5)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)

print("Salt-and-pepper noise (10%): PSNR comparison")
print(f"  Noisy image:        {psnr(img_clean, img_sp):.2f} dB")
print(f"  Gaussian (5×5):     {psnr(img_clean, gauss5):.2f} dB")
print(f"  Median (3×3):       {psnr(img_clean, median3):.2f} dB")
print(f"  Median (5×5):       {psnr(img_clean, median5):.2f} dB")
print(f"  Median (7×7):       {psnr(img_clean, median7):.2f} dB")
# Median consistently outperforms Gaussian for salt-and-pepper noise

# ── Visual comparison ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for ax, im, title in zip(axes,
    [img_clean, img_sp, gauss5, median3, median5],
    ["Clean", "Salt & pepper (10%)",
     "Gaussian (5×5)", "Median (3×3)", "Median (5×5)"]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{title}\n{psnr(img_clean,im):.1f} dB")
    ax.axis("off")
plt.tight_layout(); plt.show()
```

### 6.5.2 Median Blur Behavior at Edges

```python
import numpy as np
import cv2

# Create a sharp edge image
edge_img = np.zeros((100, 100), dtype=np.uint8)
edge_img[:, 50:] = 200   # left half = 0, right half = 200

# Apply both filters
gaussian_edge = cv2.GaussianBlur(edge_img, (9, 9), 3.0)
median_edge   = cv2.medianBlur(edge_img, 9)

# Sample a horizontal profile through the middle
profile_orig  = edge_img[50, :]
profile_gauss = gaussian_edge[50, :]
profile_med   = median_edge[50, :]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(profile_orig,  'k-',  linewidth=2, label="Original")
plt.plot(profile_gauss, 'b--', linewidth=2, label="Gaussian (9×9, σ=3)")
plt.plot(profile_med,   'r-',  linewidth=2, label="Median (9×9)")
plt.axvline(50, color='gray', linestyle=':', alpha=0.7)
plt.xlabel("Column (x)"); plt.ylabel("Pixel value")
plt.title("Edge profile: Gaussian vs Median")
plt.legend(); plt.grid(alpha=0.3)
plt.show()
# Key result: Gaussian blurs the edge into a smooth ramp
# Median preserves the edge sharpness — the profile stays step-like
```

---

## 6.6 Bilateral Filter

The bilateral filter is the most sophisticated blur in OpenCV's standard toolkit. It smooths regions of similar color while **preserving** edges — unlike Gaussian blur, which blurs everything equally.

### 6.6.1 How It Works

A regular Gaussian blur weights neighbors only by their spatial distance from the center pixel. The bilateral filter weights neighbors by **two** factors:

1. **Spatial weight (σ_space):** Pixels physically closer to center get higher weight (same as Gaussian)
2. **Color weight (σ_color):** Pixels with similar intensity/color to center get higher weight; dissimilar pixels (across an edge) get near-zero weight

```
BilateralFilter(I)[y,x] = Σ_{i,j} G_spatial(dist(p,q)) · G_color(|I[p]-I[q]|) · I[q]
                           ─────────────────────────────────────────────────────────────
                           Normalization factor
```

At an edge, the pixels on the other side have very different color — their `G_color` weight drops to near zero, so they barely contribute. The edge is preserved.

### 6.6.2 cv2.bilateralFilter

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg")
if img is None:
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 150), 100, (0, 180, 220), -1)
    cv2.rectangle(img, (50, 50), (150, 250), (180, 80, 20), -1)
    # Add Gaussian noise
    noise = np.random.normal(0, 20, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, borderType)
#
# d: diameter of each pixel's neighborhood.
#    If d ≤ 0, it is computed from sigmaSpace: d ≈ 5 * sigmaSpace
#    Larger d = slower; recommended: d=5 for real-time, d=9 for offline
#
# sigmaColor: filter sigma in the color space.
#    Larger value → colors farther apart are mixed.
#    ~10: subtle, preserves most color differences
#    ~75: good general purpose
#    ~150: strong smoothing but starts to lose color detail
#
# sigmaSpace: filter sigma in coordinate space.
#    Larger value → pixels farther away influence each other
#    ~10: local influence only
#    ~75: wide influence
#    Recommended: sigmaColor ≈ sigmaSpace for typical use

results = {
    "Gaussian σ=3"      : cv2.GaussianBlur(img, (19, 19), 3.0),
    "Bilateral d=5  sC=25  sS=25"  : cv2.bilateralFilter(img, 5,  25,  25),
    "Bilateral d=9  sC=75  sS=75"  : cv2.bilateralFilter(img, 9,  75,  75),
    "Bilateral d=15 sC=150 sS=15"  : cv2.bilateralFilter(img, 15, 150, 15),
    "Bilateral d=9  sC=150 sS=75"  : cv2.bilateralFilter(img, 9,  150, 75),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes.flat[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes.flat[0].set_title("Original"); axes.flat[0].axis("off")

for ax, (title, result) in zip(axes.flat[1:], results.items()):
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9); ax.axis("off")
plt.suptitle("Bilateral Filter Parameter Effects", fontsize=13)
plt.tight_layout(); plt.show()
```

### 6.6.3 Bilateral Filter Trade-offs

```python
import cv2
import numpy as np
import time

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

# Timing comparison
def time_filter(func, *args, n=20):
    t0 = time.perf_counter()
    for _ in range(n): func(*args)
    return (time.perf_counter() - t0) / n * 1000

t_gauss = time_filter(cv2.GaussianBlur, img, (9, 9), 3.0)
t_bil5  = time_filter(cv2.bilateralFilter, img, 5,  75, 75)
t_bil9  = time_filter(cv2.bilateralFilter, img, 9,  75, 75)
t_bil15 = time_filter(cv2.bilateralFilter, img, 15, 75, 75)

print(f"Timing on {img.shape[1]}×{img.shape[0]} image:")
print(f"  Gaussian (9×9):      {t_gauss:.2f} ms")
print(f"  Bilateral d=5:       {t_bil5:.2f} ms  ({t_bil5/t_gauss:.1f}× Gaussian)")
print(f"  Bilateral d=9:       {t_bil9:.2f} ms  ({t_bil9/t_gauss:.1f}× Gaussian)")
print(f"  Bilateral d=15:      {t_bil15:.2f} ms ({t_bil15/t_gauss:.1f}× Gaussian)")
# Bilateral is O(d²) per pixel → quadratic in d
# Gaussian is O(k) per pixel via separable implementation → linear in k

# ── When to use bilateral ─────────────────────────────────────────────────────
# Use bilateral filter when:
# 1. You need to denoise while preserving edges (portrait processing, depth maps)
# 2. You're preprocessing for edge detection (Canny works better on bilaterally filtered images)
# 3. You're creating a stylized "painting" or "cartoon" effect
# 4. Speed is not critical (d≤9 for real-time at HD resolution)

# Avoid bilateral filter when:
# 1. Processing at 4K in real-time → use guided filter or fast approximation
# 2. Noise is salt-and-pepper type → use median instead
# 3. You need reproducible deterministic output → bilateral is numerically complex
```

### 6.6.4 Cartoon Effect with Bilateral Filter

A popular creative application of bilateral filtering:

```python
import cv2
import numpy as np


def cartoonize(img_bgr, n_passes=5, d=9, sigma_color=200, sigma_space=200,
               edge_threshold1=100, edge_threshold2=200, edge_thickness=3):
    """
    Create a cartoon effect by:
    1. Repeatedly applying bilateral filter (smooths while preserving edges)
    2. Detecting edges on the grayscale version
    3. Combining smooth color with dark edges

    Parameters
    ----------
    n_passes : int
        Number of bilateral filter passes (more = more paint-like)
    d : int
        Bilateral filter diameter
    sigma_color, sigma_space : float
        Bilateral filter sigma values (high = more smoothing)
    edge_threshold1/2 : float
        Canny thresholds for edge detection
    edge_thickness : int
        Dilation iterations for edges
    """
    # Step 1: Smoothed color via iterated bilateral filter
    smoothed = img_bgr.copy()
    for _ in range(n_passes):
        smoothed = cv2.bilateralFilter(smoothed, d, sigma_color, sigma_space)

    # Step 2: Edge detection on grayscale
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.medianBlur(gray, 5)  # pre-smooth to reduce noise in edges
    edges = cv2.Canny(blurred_gray, edge_threshold1, edge_threshold2)

    # Thicken edges slightly for a drawn-line look
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_thickness, edge_thickness))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    # Step 3: Invert edges (black lines) and combine
    edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_mask = 255 - edge_mask   # invert: 0=edge, 255=non-edge

    cartoon = cv2.bitwise_and(smoothed, edge_mask)
    return cartoon, smoothed, edges


# Demo
img = cv2.imread("sample.jpg")
if img is None:
    # Create a colorful test image
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    cv2.circle(img, (250, 200), 120, (20, 180, 220), -1)
    cv2.rectangle(img, (50, 280), (200, 380), (180, 60, 20), -1)
    cv2.ellipse(img, (380, 300), (80, 60), 30, 0, 360, (40, 200, 40), -1)

cartoon, smooth, edges = cartoonize(img)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, im, title in zip(axes,
    [img, smooth, cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR), cartoon],
    ["Original", "Bilateral smoothed", "Edge overlay", "Cartoon result"]):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 6.7 Separable Filters

For large kernels, the separability property provides a major speedup. A separable 2D kernel can be decomposed into two 1D passes:

```python
import cv2
import numpy as np
import time

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

# cv2.sepFilter2D(src, ddepth, kernelX, kernelY, anchor, delta, borderType)
# Applies kernelX along columns (1D horizontal), then kernelY along rows (1D vertical)
# This is equivalent to filter2D with the outer-product kernel, but faster

# Gaussian decomposition
sigma = 3.0
kernel_1d = cv2.getGaussianKernel(sigma_to_ksize(sigma), sigma, cv2.CV_32F)
# kernel_1d has shape (k, 1)

# Method 1: 2D filter (O(k²) per pixel)
ksize = sigma_to_ksize(sigma)
kernel_2d = kernel_1d @ kernel_1d.T   # outer product
t0 = time.perf_counter()
for _ in range(50):
    result_2d = cv2.filter2D(img, -1, kernel_2d)
t_2d = (time.perf_counter() - t0) / 50 * 1000

# Method 2: Separable filter (O(2k) per pixel)
t0 = time.perf_counter()
for _ in range(50):
    result_sep = cv2.sepFilter2D(img, -1, kernel_1d, kernel_1d)
t_sep = (time.perf_counter() - t0) / 50 * 1000

# Verify identical result
max_err = np.abs(result_2d.astype(int) - result_sep.astype(int)).max()
print(f"σ={sigma}, ksize={ksize}:")
print(f"  2D filter:      {t_2d:.2f} ms")
print(f"  Separable:      {t_sep:.2f} ms  ({t_sep/t_2d:.2f}× relative)")
print(f"  Max difference: {max_err} (should be 0 or 1 from rounding)")
print(f"  Theoretical speedup: 2k/{ksize} = {2.0/ksize:.3f}× work per pixel")
# For large kernels, sepFilter2D is faster by O(k/2) factor

# ── GaussianBlur is automatically separable internally ────────────────────────
# cv2.GaussianBlur uses sepFilter2D under the hood — no need to call sepFilter2D
# manually for Gaussian. Use it for custom separable kernels.

# Example: Horizontal Sobel in X, then smooth in Y
sobel_x = np.array([[-1, 0, 1]], dtype=np.float32)   # shape (1, 3)
smooth_y = np.array([[1, 2, 1]], dtype=np.float32).T / 4.0   # shape (3, 1)
# Apply: horizontal derivative, vertical smoothing = Sobel X kernel
sobel_result = cv2.sepFilter2D(img, cv2.CV_32F,
                                kernelX=sobel_x,
                                kernelY=smooth_y)
```

---

## 6.8 Comparing All Smoothing Filters

A systematic comparison helps you internalize when to use which filter:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def load_test_image():
    img = cv2.imread("sample.jpg")
    if img is None:
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(img, (200, 150), 90, (0, 180, 220), -1)
        cv2.rectangle(img, (50, 200), (150, 290), (180, 60, 20), -1)
        # Add mixed noise
        gauss_noise = np.random.normal(0, 15, img.shape).astype(np.float32)
        sp_noisy    = img.copy()
        idx = np.random.choice(img.size // 3, img.size // 60)
        sp_noisy.reshape(-1, 3)[idx] = 255
        sp_noisy.reshape(-1, 3)[idx + img.size//90] = 0
        img = np.clip(sp_noisy.astype(np.float32) + gauss_noise, 0, 255).astype(np.uint8)
    return img

img = load_test_image()
k = 9
sigma = 3.0

filters = {
    "Box (9×9)"          : lambda i: cv2.blur(i, (k, k)),
    f"Gaussian σ={sigma}" : lambda i: cv2.GaussianBlur(i, (sigma_to_ksize(sigma),)*2, sigma),
    "Median (9×9)"        : lambda i: cv2.medianBlur(i, k),
    "Bilateral d=9"       : lambda i: cv2.bilateralFilter(i, k, 75, 75),
}

# Apply and time
results = {}
timings = {}
for name, func in filters.items():
    t0 = time.perf_counter()
    for _ in range(20):
        result = func(img)
    timings[name] = (time.perf_counter() - t0) / 20 * 1000
    results[name] = result

# Display
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes.flat[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes.flat[0].set_title("Input (mixed noise)"); axes.flat[0].axis("off")

for ax, (name, result) in zip(axes.flat[1:], results.items()):
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{name}\n{timings[name]:.1f} ms", fontsize=9)
    ax.axis("off")
plt.suptitle("Smoothing Filter Comparison (ksize≈9)", fontsize=13)
plt.tight_layout(); plt.show()

# Print timing table
print("\nTiming on this image:")
ref = timings["Box (9×9)"]
for name, t in timings.items():
    print(f"  {name:<25}: {t:6.2f} ms  ({t/ref:.1f}× box)")
```

---

## 6.9 Adaptive Noise Reduction Pipeline

Real images often have spatially varying noise levels. A production-quality pipeline should adapt the filter strength to local noise:

```python
import cv2
import numpy as np


def estimate_local_noise(img_gray: np.ndarray, ksize: int = 7) -> np.ndarray:
    """
    Estimate local noise level as the standard deviation within each patch.
    Returns a float32 noise map, same size as input.
    """
    img_f = img_gray.astype(np.float32)
    # Local mean via Gaussian blur
    local_mean = cv2.GaussianBlur(img_f, (ksize, ksize), ksize / 4.0)
    # Local variance: E[X²] - E[X]²
    local_mean_sq = cv2.GaussianBlur(img_f ** 2, (ksize, ksize), ksize / 4.0)
    local_var  = np.maximum(0, local_mean_sq - local_mean ** 2)
    local_std  = np.sqrt(local_var)
    return local_std


def adaptive_denoise(img_bgr: np.ndarray,
                      low_noise_threshold: float = 15.0,
                      high_noise_threshold: float = 40.0) -> np.ndarray:
    """
    Adaptive denoising pipeline:
    - Low-noise regions: light bilateral filter
    - High-noise regions: stronger Gaussian blur
    - Intermediate: interpolated blend

    Returns denoised BGR image.
    """
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    noise_map = estimate_local_noise(gray)

    # Normalize noise map to [0, 1] blending weight
    # 0 = use light filter; 1 = use strong filter
    blend = np.clip((noise_map - low_noise_threshold) /
                     (high_noise_threshold - low_noise_threshold),
                     0.0, 1.0)[:, :, np.newaxis]   # (H, W, 1) for broadcasting

    # Two denoising passes
    light  = cv2.bilateralFilter(img_bgr, 7,   50,  50).astype(np.float32)
    strong = cv2.GaussianBlur(img_bgr, (11, 11), 4.0).astype(np.float32)

    # Blend based on local noise level
    denoised = (light * (1.0 - blend) + strong * blend).clip(0, 255).astype(np.uint8)

    return denoised, noise_map


# ── Demo ──────────────────────────────────────────────────────────────────────
# Create a spatially varying noise scenario
img_clean = cv2.imread("sample.jpg")
if img_clean is None:
    img_clean = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(img_clean, (200, 150), 90, (0, 180, 220), -1)
    cv2.rectangle(img_clean, (50, 200), (150, 290), (180, 60, 20), -1)

# Add high noise on left half, low noise on right half
img_noisy = img_clean.copy().astype(np.float32)
img_noisy[:, :200] += np.random.normal(0, 45, img_clean[:, :200].shape)   # high noise left
img_noisy[:, 200:] += np.random.normal(0, 5,  img_clean[:, 200:].shape)   # low noise right
img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

denoised, noise_map = adaptive_denoise(img_noisy)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, im, title in zip(axes,
    [img_clean, img_noisy,
     (noise_map / noise_map.max() * 255).astype(np.uint8), denoised],
    ["Clean", "Noisy (spatially varying)",
     "Noise estimate map", "Adaptive denoised"]):
    if im.ndim == 3:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(im, cmap="hot")
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 6.10 Filter Parameter Quick Reference

```python
import cv2
import numpy as np

# ════════════════════════════════════════════════════════
# QUICK REFERENCE: All smoothing filters
# ════════════════════════════════════════════════════════

# cv2.blur(src, ksize)
#   ksize: (width, height) — can be non-square, can be even
#   Fast, uniform weight, not great quality
cv2.blur(np.zeros((100,100,3),dtype=np.uint8), (5,5))

# cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=0, borderType=BORDER_REFLECT_101)
#   ksize: MUST be odd (3,3)(5,5)(7,7)...; or (0,0) to auto-compute from sigma
#   sigmaX: controls blur width; sigmaY=0 → same as sigmaX
#   sigmaX=0 with ksize given → sigma auto from ksize
#   BEST FOR: Gaussian noise, pre-processing for edge detection, scale space
cv2.GaussianBlur(np.zeros((100,100,3),dtype=np.uint8), (9,9), 0)

# cv2.medianBlur(src, ksize)
#   ksize: MUST be odd positive int; for uint8 can be any odd; float32 single channel ≤5
#   BEST FOR: Salt-and-pepper noise, preserving edges
cv2.medianBlur(np.zeros((100,100,3),dtype=np.uint8), 5)

# cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, borderType=BORDER_REFLECT_101)
#   d: neighborhood diameter; d≤0 → auto from sigmaSpace (≈ 5*sigmaSpace)
#   sigmaColor: how much color difference is tolerated (~10=strict, ~75=medium, ~150=permissive)
#   sigmaSpace: spatial extent (~10=local, ~75=wide)
#   CANNOT operate in-place
#   BEST FOR: Edge-preserving smoothing, portrait processing, cartoon effect pre-processing
cv2.bilateralFilter(np.zeros((100,100,3),dtype=np.uint8), 9, 75, 75)

# cv2.filter2D(src, ddepth, kernel, anchor=(-1,-1), delta=0, borderType)
#   ddepth=-1: output same dtype as input
#   ddepth=cv2.CV_32F: float output (needed for kernels with negative values)
#   kernel: numpy array, any size (odd preferred for symmetric anchor)
#   Performs CORRELATION not convolution
#   BEST FOR: Custom kernels, sharpening, embossing, motion blur, arbitrary filters
cv2.filter2D(np.zeros((100,100,3),dtype=np.uint8), -1,
             np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32))

# cv2.sepFilter2D(src, ddepth, kernelX, kernelY, anchor, delta, borderType)
#   kernelX: 1D row kernel, shape (1, k) or (k,)
#   kernelY: 1D col kernel, shape (k, 1) or (k,)
#   BEST FOR: Large separable custom kernels where performance matters
cv2.sepFilter2D(np.zeros((100,100,3),dtype=np.uint8), -1,
                np.ones((1,7),dtype=np.float32)/7,
                np.ones((7,1),dtype=np.float32)/7)

# ════════════════════════════════════════════════════════
# Border types (use as borderType parameter)
# ════════════════════════════════════════════════════════
# cv2.BORDER_CONSTANT   (0)  : fill with constant (default color=black)
# cv2.BORDER_REPLICATE       : repeat edge pixel: aaaa|abcd|dddd
# cv2.BORDER_REFLECT         : mirror including edge: dcba|abcd|dcba
# cv2.BORDER_REFLECT_101     : mirror excluding edge: dcb|abcd|cba  ← DEFAULT for filters
# cv2.BORDER_WRAP            : tile: abcd|abcd|abcd
```

---

## 6.11 Summary

This chapter established filtering as the foundational image processing operation:

**Convolution / cv2.filter2D:**
- Slides a kernel over the image computing weighted sums
- OpenCV performs *correlation* (not convolution) — for asymmetric kernels, pre-flip with `cv2.flip`
- Kernel sums to 1 → preserves brightness; sums to 0 → zero response to uniform regions
- Separable kernels split into two 1D passes for O(k) instead of O(k²) work

**Box Blur (cv2.blur):** Uniform average, fast via integral images, visible rectangular artifacts, not recommended for quality applications.

**Gaussian Blur (cv2.GaussianBlur):** Weighted average matching Gaussian distribution. Gold standard for pre-processing. Separable internally. Best for Gaussian noise. Use `ksize ≈ 6σ`.

**Median Blur (cv2.medianBlur):** Non-linear. Replaces with neighborhood median. Immune to salt-and-pepper outliers. Preserves edges. Cannot be factored into separable passes.

**Bilateral Filter (cv2.bilateralFilter):** Combines spatial and color proximity weights. Smooths flat regions, stops at edges. Significantly slower than Gaussian — use `d≤9` for real-time.

**The key choice rule:**
- Gaussian noise → Gaussian blur (or bilateral for quality)
- Salt-and-pepper noise → Median blur
- Blurring near edges → Bilateral filter
- Custom operation → filter2D with explicit kernel
- Large separable custom kernel → sepFilter2D

---

## 6.12 Exercises

### Warm-up

**6.1** Write a function `apply_motion_blur(img, angle_deg, length)` that creates a motion blur kernel of a given length and angle and applies it with `cv2.filter2D`. Test it at angles 0°, 45°, and 90° with length=30.

**6.2** Demonstrate that `cv2.GaussianBlur` with sigma=3 and the corresponding ksize is exactly equivalent to `cv2.sepFilter2D` with `cv2.getGaussianKernel`. Show that the max absolute difference is ≤1 (rounding).

**6.3** Add salt-and-pepper noise at densities 5%, 10%, and 25% to a test image. For each, apply median blur with ksize=3, 5, and 7. Plot a 3×3 grid of results and a PSNR heat map.

### Core

**6.4** Implement a `wiener_sharpen(img, sigma_blur, k=0.01)` function that performs deconvolution-based sharpening: compute the Fourier transform, apply a Wiener filter in the frequency domain, return the sharpened image. Compare to the simple sharpening kernel from Section 6.1.

**6.5** Build a `filter_benchmark(img, filters_dict)` function that takes a dict of `{name: func}` filter functions, runs each 50 times, and returns a DataFrame (or dict) with: name, mean_time_ms, std_time_ms, output_psnr (if ground truth given). Display a bar chart of execution times.

**6.6** Implement a `stacked_gaussian_blur(img, sigmas)` function that applies a series of Gaussian blurs in sequence. Show mathematically (and verify numerically) that blurring with σ₁ then σ₂ is equivalent to a single blur with σ_total = √(σ₁² + σ₂²). This is the **scale-space addition theorem**.

**6.7** Implement the `unsharp_mask(img, sigma, amount, threshold=0)` sharpening filter: `output = img + amount * (img - GaussianBlur(img))`. The `threshold` parameter prevents sharpening in already-sharp regions. Test with amount=0.5, 1.0, 1.5 and visualize the difference images.

### Challenge

**6.8** Implement a **Non-Local Means (NLM) denoising approximation**: for each pixel, find the K most similar patches (in a search window) and average them. Compare to `cv2.fastNlMeansDenoisingColored` and measure PSNR and runtime. Start with a small image (128×128) to keep runtime manageable.

**6.9** Build an **interactive filter tuner** using trackbars that supports switching between all four filters (box/Gaussian/median/bilateral) with appropriate parameter trackbars for each. Display input, output, and a difference-amplified image (|(input - output)| × 5) side by side. Show FPS in the title bar.

**6.10** Implement **guided image filtering**: given a guidance image G and an input image P, the guided filter produces an output Q = linear_transform(G) that smooths P but preserves edges in G. The guidance image can be the input itself (self-guided) or a different image (e.g., RGB image guiding a depth map). Demonstrate: (a) self-guided denoising, (b) depth map upsampling guided by a high-resolution RGB image.

---

## Further Reading

- **OpenCV smoothing tutorial:** https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- **OpenCV image filtering reference:** https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
- **Bilateral filter original paper:** Tomasi & Manduchi, "Bilateral filtering for gray and color images," ICCV 1998
- **Non-local means denoising:** Buades, Coll & Morel, "A non-local algorithm for image denoising," CVPR 2005
- **Scale-space theory:** Lindeberg, "Feature detection with automatic scale selection," IJCV 1998
- **cv2.fastNlMeansDenoisingColored documentation:** https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html
