# Chapter 13: Corner & Keypoint Detection

> *"A corner is the simplest invariant structure in an image — it looks the same from nearby viewpoints, survives mild blur, and appears at a consistent location regardless of lighting. Everything else in feature detection builds on this foundation."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain what makes a good feature for tracking and matching
2. Derive the structure tensor (second-moment matrix) from first principles
3. Implement Harris corner detection and understand the role of the k parameter
4. Apply `cv2.cornerHarris` and `cv2.cornerSubPix` for sub-pixel accuracy
5. Explain the Shi-Tomasi improvement over Harris and use `cv2.goodFeaturesToTrack`
6. Apply the FAST corner detector and understand its circle-based test
7. Use `cv2.FastFeatureDetector` with all configuration options
8. Visualize and compare keypoints using `cv2.drawKeypoints`
9. Build a repeatable feature detection pipeline robust to noise and scale
10. Compare all three detectors quantitatively on repeatability and localization accuracy

---

## 13.1 What Makes a Good Feature?

Before studying any specific detector, we need to understand what makes a feature **useful** for computer vision tasks like tracking, panoramic stitching, or visual odometry.

### 13.1.1 The Three Requirements

A good feature (keypoint, corner, or interest point) must satisfy three properties:

**1. Repeatability:** The same physical point should be detected in multiple images of the same scene, even under different viewpoints, illumination, or scale. A feature that appears in one image but not another is useless for matching.

**2. Distinctiveness:** The local neighborhood around the detected point should look different from most other points in the image. This allows it to be uniquely matched against a database.

**3. Localization accuracy:** The detected position should be as close as possible to the true physical location. Poor localization introduces geometric error into everything downstream — homography estimation, 3D reconstruction, pose estimation.

### 13.1.2 Why Corners?

Consider three local image regions:

```
Flat region:       Edge:              Corner:
────────────       ─────┃─────        ─────┃─────
────────────       ─────┃─────        ─────┼─────
────────────       ─────┃─────        ─────┃─────

No change in       Change only in     Change in ALL
any direction      one direction      directions
```

If you slide a small window over a **flat region**, the appearance does not change in any direction — you cannot tell where you are. An **edge** only changes when you move perpendicular to it — you can locate yourself along the edge but not along it. A **corner** changes when you move in any direction — it unambiguously marks a specific location.

This intuition — measuring how intensity changes in all directions within a local window — is the foundation of every corner detector.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_corner_intuition():
    """Create the three canonical region types and show their appearance change."""
    N = 80

    # Flat region
    flat = np.full((N, N), 128, dtype=np.uint8)
    flat += np.random.normal(0, 3, (N, N)).astype(np.int8)

    # Edge region (vertical edge at center)
    edge = np.zeros((N, N), dtype=np.uint8)
    edge[:, :N//2] = 80
    edge[:, N//2:] = 180

    # Corner region (top-left dark, rest lighter)
    corner = np.zeros((N, N), dtype=np.uint8)
    corner[:N//2, :N//2] = 50
    corner[:N//2, N//2:] = 160
    corner[N//2:, :] = 160

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, (img, title) in enumerate(zip(
        [flat, edge, corner],
        ["Flat region", "Edge region", "Corner region"]
    )):
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(title); axes[0, col].axis("off")

        # Show how appearance changes as we shift a 20×20 window
        shifts = np.arange(-20, 21, 4)
        ssd_h = []   # Sum of squared differences for horizontal shifts
        ssd_d = []   # diagonal shifts

        cx, cy = N // 2, N // 2
        w_size = 15
        ref = img[cy-w_size//2:cy+w_size//2, cx-w_size//2:cx+w_size//2].astype(float)

        for s in shifts:
            # Horizontal shift
            nx = np.clip(cx + s, w_size//2, N - w_size//2)
            patch = img[cy-w_size//2:cy+w_size//2, nx-w_size//2:nx+w_size//2].astype(float)
            if patch.shape == ref.shape:
                ssd_h.append(np.sum((ref - patch)**2))
            else:
                ssd_h.append(0)

        axes[1, col].plot(shifts, ssd_h, 'b-o', markersize=4, linewidth=2, label="Horizontal")
        axes[1, col].set_title("SSD vs shift distance"); axes[1, col].legend(fontsize=8)
        axes[1, col].set_xlabel("Shift (pixels)"); axes[1, col].grid(alpha=0.3)

    plt.suptitle("Why Corners Are Good Features: Appearance Change Under Shift", fontsize=12)
    plt.tight_layout(); plt.show()


visualize_corner_intuition()
```

---

## 13.2 The Structure Tensor: Mathematical Foundation

Every corner detector studied in this chapter derives from the same 2×2 matrix — the **structure tensor** (also called the second-moment matrix or Harris matrix).

### 13.2.1 Derivation

The sum of squared differences (SSD) of a small window shifted by (Δx, Δy) is:

```
E(Δx, Δy) = Σ_{(x,y) in W} [I(x + Δx, y + Δy) - I(x, y)]²
```

Using a first-order Taylor expansion: `I(x + Δx, y + Δy) ≈ I(x, y) + Iₓ·Δx + Iᵧ·Δy`

This simplifies to:

```
E(Δx, Δy) ≈ [Δx  Δy] · M · [Δx  Δy]ᵀ
```

Where **M** is the structure tensor:

```
M = Σ_{W} w(x,y) · [Iₓ²   IₓIᵧ]
                     [IₓIᵧ  Iᵧ²  ]
```

The weight function w(x,y) is either a uniform box (original Harris) or a Gaussian (more common).

### 13.2.2 Interpreting the Eigenvalues

The eigenvalues λ₁ ≥ λ₂ of M describe the local geometry:

```
         λ₂
          ↑
    Corner│    Corner
 λ₂ >> 0 │  λ₁ ≈ λ₂ >> 0
          │
    ──────┼──────────────→ λ₁
   Edge   │
 λ₁ >> 0 │   Flat region
 λ₂ ≈ 0  │   λ₁ ≈ λ₂ ≈ 0
```

| Region | λ₁ | λ₂ | Interpretation |
|--------|-----|-----|----------------|
| Flat | ≈ 0 | ≈ 0 | No gradient in any direction |
| Edge | Large | ≈ 0 | Gradient in one direction only |
| Corner | Large | Large | Gradient in all directions |

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_structure_tensor(img_gray, block_size=3, ksize=3):
    """
    Compute the structure tensor M at every pixel.

    M = [[Σ Ix², Σ Ix·Iy],
         [Σ Ix·Iy, Σ Iy²]]

    Returns Ixx, Iyy, Ixy (each smoothed over blockSize×blockSize window).
    """
    img_f = img_gray.astype(np.float32)

    # Compute image gradients
    Ix = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=ksize)

    # Products of gradients (elements of M)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Smooth over neighborhood (Gaussian window is better than box)
    sigma = block_size / 2.0
    Ixx = cv2.GaussianBlur(Ixx, (block_size*2+1, block_size*2+1), sigma)
    Iyy = cv2.GaussianBlur(Iyy, (block_size*2+1, block_size*2+1), sigma)
    Ixy = cv2.GaussianBlur(Ixy, (block_size*2+1, block_size*2+1), sigma)

    return Ixx, Iyy, Ixy


def compute_eigenvalues(Ixx, Iyy, Ixy):
    """
    Compute eigenvalues of M at each pixel analytically.

    For 2×2 symmetric matrix [[a, b], [b, c]]:
    λ₁,₂ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)
    """
    half_trace = (Ixx + Iyy) / 2.0
    discriminant = np.sqrt(((Ixx - Iyy) / 2.0)**2 + Ixy**2)
    lambda1 = half_trace + discriminant   # larger eigenvalue
    lambda2 = half_trace - discriminant   # smaller eigenvalue
    return lambda1, lambda2


# ── Visualize eigenvalues on a test image ────────────────────────────────────
img = np.zeros((200, 300), dtype=np.uint8)
cv2.rectangle(img, (50, 40), (150, 140), 200, -1)
cv2.circle(img, (230, 100), 50, 180, -1)
img = img + np.random.normal(0, 5, img.shape).astype(np.int8)
img = np.clip(img, 0, 255).astype(np.uint8)

Ixx, Iyy, Ixy = compute_structure_tensor(img, block_size=3)
l1, l2 = compute_eigenvalues(Ixx, Iyy, Ixy)

# Classify each pixel
lmin = np.minimum(l1, l2)
lmax = np.maximum(l1, l2)
corner_map = np.minimum(l1, l2)   # Shi-Tomasi score = min(λ₁, λ₂)
edge_map   = lmax - lmin           # Edges have large spread between eigenvalues

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes[0,0].imshow(img, cmap="gray"); axes[0,0].set_title("Original"); axes[0,0].axis("off")
axes[0,1].imshow(l1, cmap="hot");   axes[0,1].set_title("λ₁ (larger eigenvalue)"); axes[0,1].axis("off")
axes[0,2].imshow(l2, cmap="hot");   axes[0,2].set_title("λ₂ (smaller eigenvalue)"); axes[0,2].axis("off")

axes[1,0].imshow(lmin, cmap="hot");    axes[1,0].set_title("min(λ₁,λ₂)\n→ Shi-Tomasi score"); axes[1,0].axis("off")
axes[1,1].imshow(edge_map, cmap="hot"); axes[1,1].set_title("λ₁ - λ₂\n→ Edge strength"); axes[1,1].axis("off")

# Mark detected corners in original
threshold = lmin.max() * 0.15
corner_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
corner_coords = np.argwhere(lmin > threshold)
for y, x in corner_coords[::5]:   # subsample for display
    cv2.circle(corner_img, (x, y), 2, (0, 255, 0), -1)
axes[1,2].imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
axes[1,2].set_title("Corners: min(λ₁,λ₂) > threshold"); axes[1,2].axis("off")

plt.suptitle("Structure Tensor Eigenvalues: The Foundation of All Corner Detectors", fontsize=12)
plt.tight_layout(); plt.show()
```

---

## 13.3 Harris Corner Detector

### 13.3.1 The Harris Response Function

Harris and Stephens (1988) avoided computing eigenvalues explicitly (expensive) by defining a response function that approximates the same classification:

```
R = det(M) - k · trace(M)²
  = λ₁λ₂ - k(λ₁ + λ₂)²
```

Where k is the Harris free parameter (typically 0.04–0.06).

The sign and magnitude of R classify each pixel:
- **R >> 0:** Corner (both eigenvalues large)
- **R < 0 (large magnitude):** Edge (one eigenvalue large, one small)
- **|R| ≈ 0:** Flat region (both eigenvalues small)

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def harris_response_manual(img_gray, block_size=3, ksize=3, k=0.04):
    """
    Compute Harris corner response R = det(M) - k·trace(M)²
    manually to demonstrate the derivation.
    """
    Ixx, Iyy, Ixy = compute_structure_tensor(img_gray, block_size, ksize)

    det_M   = Ixx * Iyy - Ixy * Ixy    # det(M) = λ₁λ₂
    trace_M = Ixx + Iyy                  # trace(M) = λ₁ + λ₂
    R = det_M - k * trace_M**2

    return R


def apply_harris(img_gray, block_size=3, ksize=3, k=0.04, threshold_frac=0.01):
    """
    Apply Harris corner detection and return corner pixel coordinates.

    Parameters
    ----------
    block_size : int   Neighborhood size for structure tensor computation
    ksize      : int   Aperture size for Sobel gradient (1, 3, 5, 7)
    k          : float Harris free parameter (0.04–0.06)
    threshold_frac : float   Threshold as fraction of max response

    Returns
    -------
    corners : np.ndarray (N, 2) — (x, y) corner coordinates
    R       : np.ndarray (H, W) float32 — Harris response map
    """
    R = cv2.cornerHarris(
        img_gray.astype(np.float32),
        blockSize=block_size,
        ksize=ksize,
        k=k
    )
    # R shape: (H, W), dtype: float32
    # Positive peaks = corners; negative = edges; ~0 = flat

    # Threshold and find local maxima
    threshold = threshold_frac * R.max()
    # Dilate response map to find local maxima (non-maximum suppression)
    R_dilated = cv2.dilate(R, None)  # 3×3 max-filter
    corners_mask = (R == R_dilated) & (R > threshold)

    ys, xs = np.where(corners_mask)
    corners = np.column_stack([xs, ys])  # (x, y) format

    return corners, R


# ── Test on a synthetic image with known corners ──────────────────────────────
def make_test_image():
    img = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), 200, -1)
    cv2.circle(img,    (300, 150), 80, 180, -1)
    pts = np.array([[30, 270], [150, 240], [100, 290]], np.int32)
    cv2.fillPoly(img, [pts], 150)
    noise = np.random.normal(0, 6, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


img = make_test_image()
corners, R = apply_harris(img)

print(f"Harris detected {len(corners)} corners")
print(f"R range: [{R.min():.1f}, {R.max():.1f}]")

# Visualize response map
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")

# Color-coded response: red=corner, blue=edge, gray=flat
R_vis = np.zeros((*img.shape, 3), dtype=np.uint8)
R_vis[R > 0.01 * R.max()]   = [0, 0, 200]   # red: corners
R_vis[R < -0.01 * R.max()]  = [200, 0, 0]   # blue: edges
axes[1].imshow(R_vis)
axes[1].set_title("Harris response\nRed=corner, Blue=edge"); axes[1].axis("off")

# Detected corners
corner_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x, y in corners:
    cv2.circle(corner_vis, (x, y), 4, (0, 255, 0), -1)
axes[2].imshow(cv2.cvtColor(corner_vis, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Detected {len(corners)} corners"); axes[2].axis("off")

plt.suptitle("Harris Corner Detection", fontsize=12)
plt.tight_layout(); plt.show()
```

### 13.3.2 Parameter Sensitivity

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def harris_parameter_study(img_gray):
    """Visualize the effect of each Harris parameter."""
    configs = [
        # (block_size, ksize, k, label)
        (2, 3, 0.04,  "block=2, k=0.04\n(standard)"),
        (5, 3, 0.04,  "block=5, k=0.04\n(wider neighborhood)"),
        (2, 5, 0.04,  "block=2, ksize=5\n(larger gradient)"),
        (2, 3, 0.02,  "block=2, k=0.02\n(more corners)"),
        (2, 3, 0.08,  "block=2, k=0.08\n(fewer corners)"),
    ]

    fig, axes = plt.subplots(1, len(configs), figsize=(18, 4))
    for ax, (bs, ks, k, label) in zip(axes, configs):
        corners, R = apply_harris(img_gray, block_size=bs, ksize=ks, k=k)
        vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for x, y in corners:
            cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{label}\n{len(corners)} corners", fontsize=8)
        ax.axis("off")

    plt.suptitle("Harris Parameter Sensitivity", fontsize=12)
    plt.tight_layout(); plt.show()


img = make_test_image()
harris_parameter_study(img)

# Parameter guidelines:
# blockSize: neighborhood size (2–5). Larger = smoother response, fewer corners.
# ksize:     Sobel aperture (1, 3, 5, 7). Larger = less noise-sensitive.
# k:         Harris free parameter (0.04–0.06). Smaller k = more sensitive = more corners.
#            Theoretical range: 0 < k < 0.25. Values outside 0.02–0.15 rarely used.
```

### 13.3.3 Sub-Pixel Refinement with cornerSubPix

Harris gives integer-precision corners. For applications requiring sub-pixel accuracy (calibration, stereo reconstruction), use `cv2.cornerSubPix`:

```python
import cv2
import numpy as np


def refine_corners_subpix(img_gray, corners_xy, win_size=5, max_iter=30, eps=0.001):
    """
    Refine corner locations to sub-pixel accuracy.

    Parameters
    ----------
    corners_xy : np.ndarray (N, 2) int — initial (x, y) corner locations
    win_size   : int   Search window half-size (window = (2w+1) × (2w+1))
    max_iter   : int   Maximum iterations
    eps        : float Convergence threshold (pixels)

    Returns
    -------
    corners_refined : np.ndarray (N, 1, 2) float32 — sub-pixel (x, y)
    """
    # cornerSubPix requires float32 corners in shape (N, 1, 2)
    corners_f32 = corners_xy.astype(np.float32).reshape(-1, 1, 2)

    # Termination criteria: stop after max_iter OR when change < eps pixels
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        eps
    )

    # zeroZone=(-1,-1) means no dead-zone (use the full window)
    cv2.cornerSubPix(
        img_gray,
        corners_f32,
        winSize=(win_size, win_size),
        zeroZone=(-1, -1),
        criteria=criteria
    )

    return corners_f32


img = make_test_image()
corners_int, R = apply_harris(img, threshold_frac=0.02)

if len(corners_int) > 0:
    corners_subpix = refine_corners_subpix(img, corners_int)

    # Compare: how much does sub-pixel refinement shift corners?
    shifts = np.linalg.norm(
        corners_subpix.reshape(-1, 2) - corners_int.astype(float),
        axis=1
    )
    print(f"Sub-pixel refinement shifts:")
    print(f"  Mean: {shifts.mean():.3f} px")
    print(f"  Max:  {shifts.max():.3f} px")
    print(f"  Corners with shift > 0.5px: {(shifts > 0.5).sum()}")

    # For camera calibration: sub-pixel accuracy is critical
    # A shift of 0.3px in detected corners causes ~0.3° error in calibration
```

---

## 13.4 Shi-Tomasi: A Better Corner Score

J. Shi and C. Tomasi (1994) proposed a small but important improvement: replace Harris's algebraic response with the **minimum eigenvalue** directly:

```
R_Harris    = λ₁λ₂ - k(λ₁ + λ₂)²   ← depends on empirical k
R_ShiTomasi = min(λ₁, λ₂)           ← directly measures "worst-case" directional gradient
```

A corner is accepted if `min(λ₁, λ₂) > threshold`. This is geometrically cleaner: it directly measures how much intensity change occurs in the direction of minimum variation. If even the worst direction has significant change, it is definitely a corner.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def shi_tomasi_manual(img_gray, block_size=3, ksize=3):
    """Compute Shi-Tomasi response = min eigenvalue of structure tensor."""
    Ixx, Iyy, Ixy = compute_structure_tensor(img_gray, block_size, ksize)
    l1, l2 = compute_eigenvalues(Ixx, Iyy, Ixy)
    return np.minimum(l1, l2)


# Compare Harris vs Shi-Tomasi response maps
img = make_test_image()

R_harris = cv2.cornerHarris(img.astype(np.float32), 3, 3, 0.04)
R_shitomasi = shi_tomasi_manual(img)

# Also use OpenCV's direct min-eigenvalue function
R_mineig = cv2.cornerMinEigenVal(img.astype(np.float32), blockSize=3, ksize=3)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, R, title in zip(axes,
    [R_harris, R_shitomasi, R_mineig],
    ["Harris R = det(M) - k·trace²", "Shi-Tomasi = min(λ₁,λ₂) [manual]",
     "cv2.cornerMinEigenVal"]):
    R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ax.imshow(R_norm, cmap="hot"); ax.set_title(title, fontsize=9); ax.axis("off")
plt.suptitle("Harris vs Shi-Tomasi Response Maps", fontsize=12)
plt.tight_layout(); plt.show()

print("Shi-Tomasi advantage: No empirical k parameter.")
print("  Harris requires tuning k (0.04 is standard but not universal).")
print("  Shi-Tomasi threshold is more geometrically interpretable.")
```

### 13.4.1 cv2.goodFeaturesToTrack: The Practical Interface

`cv2.goodFeaturesToTrack` is the highest-level interface — it handles detection, non-maximum suppression, and optional sub-pixel refinement in one call:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def good_features_demo(img_bgr_or_gray):
    """
    Demonstrate cv2.goodFeaturesToTrack with all parameter options.
    """
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray

    # cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance,
    #                          mask=None, blockSize=3, useHarrisDetector=False,
    #                          k=0.04)
    #
    # maxCorners   : Maximum number of corners to return (sorted by quality)
    # qualityLevel : Minimum accepted quality as fraction of max quality (0–1)
    #                A corner is kept if quality ≥ qualityLevel × max_quality
    # minDistance  : Minimum Euclidean distance between returned corners (pixels)
    #                Acts as non-maximum suppression with a geometric constraint
    # useHarrisDetector: False = Shi-Tomasi (default), True = Harris
    # k            : Harris free parameter (only when useHarrisDetector=True)

    # Standard Shi-Tomasi: top 50 corners, quality ≥ 1%, at least 10px apart
    corners_st = cv2.goodFeaturesToTrack(gray, maxCorners=50,
                                          qualityLevel=0.01, minDistance=10)

    # Harris variant
    corners_h = cv2.goodFeaturesToTrack(gray, maxCorners=50,
                                          qualityLevel=0.01, minDistance=10,
                                          useHarrisDetector=True, k=0.04)

    # Tight quality: only very strong corners
    corners_strict = cv2.goodFeaturesToTrack(gray, maxCorners=200,
                                              qualityLevel=0.1, minDistance=5)

    # With a mask: only detect corners in a specific region
    mask = np.zeros_like(gray)
    mask[50:200, 50:300] = 255  # only left portion
    corners_masked = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10, mask=mask)

    print("goodFeaturesToTrack results:")
    for name, cnrs in [("Shi-Tomasi (q=0.01, d=10)", corners_st),
                        ("Harris (q=0.01, d=10)", corners_h),
                        ("Strict (q=0.1, d=5)", corners_strict),
                        ("Masked (left region only)", corners_masked)]:
        n = len(cnrs) if cnrs is not None else 0
        print(f"  {name:<35}: {n} corners")

    # corners shape: (N, 1, 2) float32 — extract as (N, 2)
    if corners_st is not None:
        pts = corners_st.reshape(-1, 2)
        print(f"\nCorner format: {corners_st.shape}  → reshape to {pts.shape}")
        print(f"First 3 corners (x, y): {pts[:3]}")

    # Visualize all four results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    results = [
        (corners_st,     "Shi-Tomasi\nq=0.01, d=10", (0,255,0)),
        (corners_h,      "Harris\nq=0.01, d=10",     (255,100,0)),
        (corners_strict, "Shi-Tomasi\nq=0.10, d=5",  (0,100,255)),
        (corners_masked, "Masked\n(left region)",     (255,0,255)),
    ]
    for ax, (cnrs, title, color) in zip(axes, results):
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if cnrs is not None:
            for pt in cnrs.reshape(-1, 2):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 4,
                           color[::-1], -1)  # BGR
        n = len(cnrs) if cnrs is not None else 0
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title}\n{n} corners", fontsize=9)
        ax.axis("off")
    plt.suptitle("goodFeaturesToTrack Parameter Effects", fontsize=12)
    plt.tight_layout(); plt.show()


img = make_test_image()
good_features_demo(img)
```

### 13.4.2 Sub-Pixel Corners from goodFeaturesToTrack

```python
import cv2
import numpy as np


def good_features_with_subpix(img_gray, max_corners=100, quality=0.01, min_dist=10):
    """
    Detect corners with Shi-Tomasi and immediately refine to sub-pixel accuracy.

    This is the complete production pipeline for feature tracking applications.
    """
    # Step 1: Integer-precision corners
    corners = cv2.goodFeaturesToTrack(img_gray, max_corners, quality, min_dist)

    if corners is None or len(corners) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    # Step 2: Sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners_refined = cv2.cornerSubPix(
        img_gray, corners.copy(),
        winSize=(5, 5), zeroZone=(-1, -1),
        criteria=criteria
    )

    return corners_refined   # shape (N, 1, 2) float32


img = make_test_image()
corners_precise = good_features_with_subpix(img)
print(f"Detected {len(corners_precise)} corners with sub-pixel precision")
print(f"Example: {corners_precise[:3].reshape(-1, 2)}")
```

---

## 13.5 FAST: Features from Accelerated Segment Test

Harris and Shi-Tomasi compute eigenvalues of a 2×2 matrix at every pixel — expensive for real-time applications. **FAST** (Rosten & Drummond, 2006) takes a completely different approach that is one to two orders of magnitude faster.

### 13.5.1 The Circle Test

FAST examines a circle of 16 pixels at radius 3 around each candidate pixel `p`:

```
       ○ ○ ○
     ○       ○
   ○           ○
   ○     p     ○
   ○           ○
     ○       ○
       ○ ○ ○
```

A pixel `p` is a corner if there exist at least N (usually 9 or 12) **contiguous** pixels on the circle that are all either:
- Brighter than `I(p) + threshold`, or
- Darker than `I(p) - threshold`

This test has an accelerated pre-filter: first check pixels at positions 1, 5, 9, and 13. If at least 3 of these 4 are not all brighter or darker than p, the pixel cannot be a corner and is rejected immediately. This eliminates ~75% of candidates cheaply.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fast_circle_visualization(img_gray, px, py, threshold=20):
    """
    Visualize the FAST circle test at a specific pixel.
    Shows which circle pixels are brighter/darker than p ± threshold.
    """
    # Bresenham circle of radius 3 — the 16 pixel positions
    # Relative to center pixel (0,0)
    circle_offsets = [
        (-3,0), (-3,1), (-2,2), (-1,3), (0,3),
        (1,3), (2,2), (3,1), (3,0), (3,-1),
        (2,-2), (1,-3), (0,-3), (-1,-3), (-2,-2), (-3,-1)
    ]

    Ip = float(img_gray[py, px])
    high = Ip + threshold
    low  = Ip - threshold

    # Classify circle pixels
    labels = []
    for dx, dy in circle_offsets:
        nx, ny = px + dx, py + dy
        if 0 <= ny < img_gray.shape[0] and 0 <= nx < img_gray.shape[1]:
            Iq = float(img_gray[ny, nx])
            if Iq > high:
                labels.append("bright")
            elif Iq < low:
                labels.append("dark")
            else:
                labels.append("similar")
        else:
            labels.append("outside")

    # Check FAST condition: ≥9 contiguous same-class pixels
    def longest_run(labels_list, label_type):
        doubled = labels_list + labels_list   # circular
        max_run = 0
        current = 0
        for l in doubled:
            if l == label_type:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return min(max_run, 16)

    run_bright = longest_run(labels, "bright")
    run_dark   = longest_run(labels, "dark")
    is_corner  = (run_bright >= 9) or (run_dark >= 9)

    return labels, is_corner, run_bright, run_dark


# ── Apply FAST via OpenCV ─────────────────────────────────────────────────────
img = make_test_image()

# Create FAST detector object
fast = cv2.FastFeatureDetector_create()

# Default parameters
print(f"Default threshold: {fast.getThreshold()}")           # 10
print(f"Non-max suppression: {fast.getNonmaxSuppression()}")  # True
print(f"Type: {fast.getType()}")   # cv2.FastFeatureDetector_TYPE_9_16 = 2

# Detect keypoints
kps = fast.detect(img, None)
print(f"\nFAST (default settings): {len(kps)} keypoints")

# Access keypoint properties
if kps:
    kp = kps[0]
    print(f"Keypoint[0]: pt={kp.pt}, size={kp.size}, "
          f"angle={kp.angle:.1f}, response={kp.response:.2f}")
    # pt      : (x, y) float32
    # size    : diameter of meaningful keypoint neighborhood
    # angle   : -1 (FAST doesn't compute orientation)
    # response: corner response value (used for ranking)
    # octave  : scale level (0 for FAST, which is scale-invariant)


# cv2.drawKeypoints visualization
# flags control what information is drawn:
# cv2.DRAW_MATCHES_FLAGS_DEFAULT:         small circle at keypoint location
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: circle sized by keypoint.size,
#                                             + line showing orientation
kp_img_default = cv2.drawKeypoints(img, kps, None,
                                     color=(0, 255, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
kp_img_rich    = cv2.drawKeypoints(img, kps, None,
                                     color=(0, 255, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(cv2.cvtColor(kp_img_default, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"FAST default flags\n{len(kps)} keypoints"); axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(kp_img_rich, cv2.COLOR_BGR2RGB))
axes[1].set_title("FAST rich keypoints\n(size shown as circle radius)"); axes[1].axis("off")
plt.suptitle("cv2.drawKeypoints Visualization Modes", fontsize=12)
plt.tight_layout(); plt.show()
```

### 13.5.2 FAST Parameters and Variants

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fast_parameter_study(img_gray):
    """Compare FAST with different threshold, NMS, and type settings."""
    configs = [
        # (threshold, nms, type, label)
        (10,  True,  cv2.FastFeatureDetector_TYPE_9_16,  "thresh=10, NMS=T\nTYPE_9_16"),
        (25,  True,  cv2.FastFeatureDetector_TYPE_9_16,  "thresh=25, NMS=T\nTYPE_9_16"),
        (10,  False, cv2.FastFeatureDetector_TYPE_9_16,  "thresh=10, NMS=F\n(many duplicates)"),
        (10,  True,  cv2.FastFeatureDetector_TYPE_7_12,  "thresh=10, NMS=T\nTYPE_7_12"),
        (10,  True,  cv2.FastFeatureDetector_TYPE_5_8,   "thresh=10, NMS=T\nTYPE_5_8"),
    ]

    fig, axes = plt.subplots(1, len(configs), figsize=(18, 4))
    for ax, (thresh, nms, ftype, label) in zip(axes, configs):
        det = cv2.FastFeatureDetector_create(threshold=thresh,
                                              nonmaxSuppression=nms,
                                              type=ftype)
        kps = det.detect(img_gray, None)
        vis = cv2.drawKeypoints(img_gray, kps, None,
                                  color=(0, 255, 0),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{label}\n{len(kps)} kps", fontsize=8)
        ax.axis("off")

    plt.suptitle("FAST Parameter Study", fontsize=12)
    plt.tight_layout(); plt.show()

    print("FAST Type meanings:")
    print("  TYPE_9_16: 9 contiguous pixels out of 16 (strict — fewer, better corners)")
    print("  TYPE_7_12: 7 out of 12 (moderate)")
    print("  TYPE_5_8:  5 out of 8  (lenient — many detections, some noisy)")
    print("\nThreshold: intensity difference required (lower = more sensitive = more kps)")
    print("NMS: non-maximum suppression — removes duplicate keypoints in small neighborhoods")


img = make_test_image()
fast_parameter_study(img)

# ── FAST speed benchmark ──────────────────────────────────────────────────────
import time

img_large = cv2.resize(img, (1280, 720))
gray_large = img_large

fast_det = cv2.FastFeatureDetector_create(threshold=10)
harris_t = time.perf_counter()
_, R_h = apply_harris(gray_large)
t_harris = (time.perf_counter() - harris_t) * 1000

fast_t = time.perf_counter()
kps_fast = fast_det.detect(gray_large, None)
t_fast = (time.perf_counter() - fast_t) * 1000

print(f"\nSpeed on 1280×720:")
print(f"  Harris:         {t_harris:.2f} ms")
print(f"  FAST:           {t_fast:.2f} ms")
print(f"  Speedup:        {t_harris / t_fast:.1f}×")
```

---

## 13.6 Comparing All Three Detectors

### 13.6.1 Feature Comparison Table

```python
# ════════════════════════════════════════════════════════════════════
# CORNER DETECTOR COMPARISON
# ════════════════════════════════════════════════════════════════════
#
#               Harris      Shi-Tomasi    FAST
# ────────────────────────────────────────────────────────────────────
# Paper year    1988         1994          2006
# Algorithm     Structure    Min            Circle
#               tensor +     eigenvalue     segment
#               response fn                test
#
# Speed         Moderate     Moderate       Very fast (10-50× Harris)
# (relative)    (baseline)   (~same)        (~1ms vs ~30ms for 1MP)
#
# Scale         ✗ No         ✗ No           ✗ No
# invariance    (needs SIFT  (needs SIFT    (needs ORB or SIFT)
#               or pyramid)  or pyramid)
#
# Rotation      ✗ No         ✗ No           ✗ No
# invariance    (gradient    (gradient      (pure circle test)
#               direction    direction
#               changes)     changes)
#
# Parameter     k (0.04)     None!          threshold, NMS, type
# sensitivity   blockSize    blockSize      (threshold is main knob)
#               ksize        ksize
#
# Repeatability Good         Good           Good (similar scale)
#
# Localization  Good with    Good with      Integer precision
# accuracy      subpix       subpix         (no subpix support)
#
# NMS           Manual       Built-in       Built-in flag
#               (needed)     (minDistance)
#
# Best use      When you     Default for    Real-time SLAM, tracking,
# case          need to      tracking,      mobile devices, when speed
#               tune k or    pyramid        > accuracy tradeoff OK
#               understand   detection
#               theory
```

### 13.6.2 Head-to-Head Quantitative Comparison

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List


def detect_harris(img_gray, max_n=500) -> List:
    """Detect corners with Harris, return as KeyPoint list."""
    corners, R = apply_harris(img_gray, threshold_frac=0.005)
    # Sort by Harris response
    strengths = [float(R[y, x]) for x, y in corners]
    sorted_idx = np.argsort(strengths)[::-1][:max_n]
    kps = [cv2.KeyPoint(float(corners[i][0]), float(corners[i][1]),
                         size=5.0, response=strengths[i])
           for i in sorted_idx]
    return kps


def detect_shi_tomasi(img_gray, max_n=500) -> List:
    """Detect corners with Shi-Tomasi, return as KeyPoint list."""
    corners = cv2.goodFeaturesToTrack(img_gray, max_n, 0.005, 5)
    if corners is None:
        return []
    return [cv2.KeyPoint(float(x), float(y), size=5.0)
            for (x, y) in corners.reshape(-1, 2)]


def detect_fast(img_gray) -> List:
    """Detect corners with FAST."""
    det = cv2.FastFeatureDetector_create(threshold=10)
    return det.detect(img_gray, None)


def repeatability_score(kps1: List, kps2: List, overlap_threshold=5.0) -> float:
    """
    Compute repeatability: fraction of keypoints in img1 that
    have a corresponding keypoint in img2 within overlap_threshold pixels.
    """
    if not kps1 or not kps2:
        return 0.0

    pts1 = np.array([kp.pt for kp in kps1])
    pts2 = np.array([kp.pt for kp in kps2])

    matched = 0
    for p in pts1:
        dists = np.linalg.norm(pts2 - p, axis=1)
        if dists.min() < overlap_threshold:
            matched += 1
    return matched / len(kps1)


def run_detector_comparison(img_gray):
    """Compare all three detectors on several criteria."""
    # Create transformed versions
    img_noisy  = cv2.GaussianBlur(img_gray, (5, 5), 2)
    img_noisy  = np.clip(img_noisy.astype(float) +
                          np.random.normal(0, 10, img_gray.shape), 0, 255).astype(np.uint8)
    rows, cols = img_gray.shape
    M = cv2.getRotationMatrix2D((cols//2, rows//2), 5, 1.0)
    img_rotated = cv2.warpAffine(img_gray, M, (cols, rows))
    img_bright  = np.clip(img_gray.astype(int) + 30, 0, 255).astype(np.uint8)

    detectors = {
        "Harris":     detect_harris,
        "Shi-Tomasi": detect_shi_tomasi,
        "FAST":       detect_fast,
    }

    results = {}
    for name, det_fn in detectors.items():
        # Measure speed
        t0 = time.perf_counter()
        for _ in range(20):
            kps_orig = det_fn(img_gray)
        t_ms = (time.perf_counter() - t0) / 20 * 1000

        kps_noisy   = det_fn(img_noisy)
        kps_rotated = det_fn(img_rotated)
        kps_bright  = det_fn(img_bright)

        rep_noise  = repeatability_score(kps_orig, kps_noisy)
        rep_rotate = repeatability_score(kps_orig, kps_rotated)
        rep_bright = repeatability_score(kps_orig, kps_bright)

        results[name] = {
            "n_detected"      : len(kps_orig),
            "time_ms"         : t_ms,
            "rep_noise"       : rep_noise,
            "rep_rotation"    : rep_rotate,
            "rep_illumination": rep_bright,
            "kps_orig"        : kps_orig,
        }

    # Print results table
    print(f"\n{'Metric':<25} {'Harris':>10} {'Shi-Tomasi':>12} {'FAST':>8}")
    print("─" * 60)
    metrics = [
        ("n_detected",       "N detected"),
        ("time_ms",          "Time (ms)"),
        ("rep_noise",        "Repeatability (noise)"),
        ("rep_rotation",     "Repeatability (5° rot)"),
        ("rep_illumination", "Repeatability (+30 bright)"),
    ]
    for key, label in metrics:
        row = {n: results[n][key] for n in results}
        fmt = ".2f" if key != "n_detected" else "d"
        vals = [f"{v:{fmt}}" for v in row.values()]
        print(f"  {label:<23}" + "".join(f"{v:>12}" for v in vals))

    return results


img = make_test_image()
results = run_detector_comparison(img)

# Visual comparison side by side
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (name, res) in zip(axes, results.items()):
    vis = cv2.drawKeypoints(img, res["kps_orig"][:200], None,
                              color=(0, 255, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{name}\n{res['n_detected']} keypoints, {res['time_ms']:.1f}ms",
                  fontsize=9)
    ax.axis("off")
plt.suptitle("Corner Detector Comparison", fontsize=12)
plt.tight_layout(); plt.show()
```

---

## 13.7 Non-Maximum Suppression

Both Harris and FAST require NMS to thin response maps into isolated keypoints. Understanding NMS helps you tune detection density:

```python
import numpy as np
import cv2


def nms_response_map(R: np.ndarray, min_distance: int = 10,
                     threshold_frac: float = 0.01) -> np.ndarray:
    """
    Non-maximum suppression for a response map R.
    Returns (N, 2) array of (x, y) corner locations.

    Strategy: iteratively select the strongest remaining response,
    suppress all responses within min_distance pixels.
    """
    R_copy = R.copy()
    threshold = threshold_frac * R.max()
    corners = []

    while True:
        # Find current maximum
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(R_copy)
        if max_val < threshold:
            break

        mx, my = max_loc   # (x, y) format
        corners.append((mx, my))

        # Suppress neighborhood
        y1 = max(0, my - min_distance)
        y2 = min(R_copy.shape[0], my + min_distance + 1)
        x1 = max(0, mx - min_distance)
        x2 = min(R_copy.shape[1], mx + min_distance + 1)
        R_copy[y1:y2, x1:x2] = 0

    return np.array(corners) if corners else np.empty((0, 2), dtype=int)


def compare_nms_distances(img_gray):
    """Show how min_distance controls corner density."""
    _, R = apply_harris(img_gray, threshold_frac=0.001)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, dist in zip(axes, [1, 5, 10, 20, 40]):
        corners = nms_response_map(R, min_distance=dist, threshold_frac=0.001)
        vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for x, y in corners:
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"min_dist={dist}\n{len(corners)} corners", fontsize=9)
        ax.axis("off")
    plt.suptitle("NMS min_distance Effect on Corner Density", fontsize=12)
    plt.tight_layout(); plt.show()


img = make_test_image()
compare_nms_distances(img)
```

---

## 13.8 Complete Project: Feature Point Tracker Initializer

A production-quality feature detection pipeline that initializes tracking points for a video sequence:

```python
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt


@dataclass
class TrackerConfig:
    """Configuration for feature detection."""
    detector:       str   = "shi_tomasi"   # "harris" | "shi_tomasi" | "fast"
    max_features:   int   = 200
    quality_level:  float = 0.01           # for shi_tomasi/harris
    min_distance:   int   = 15             # minimum pixel spacing
    fast_threshold: int   = 15             # for FAST
    use_subpix:     bool  = True           # sub-pixel refinement
    grid_mask:      bool  = False          # enforce grid-based distribution
    grid_cells:     int   = 5             # NxN grid when grid_mask=True


class FeatureInitializer:
    """
    Initialize tracking feature points in a grayscale image.
    Supports Harris, Shi-Tomasi, and FAST detectors with
    optional grid-based distribution enforcement.
    """

    def __init__(self, config: TrackerConfig = None):
        self.config = config or TrackerConfig()
        self._setup_detector()

    def _setup_detector(self):
        cfg = self.config
        if cfg.detector == "fast":
            self._fast = cv2.FastFeatureDetector_create(
                threshold=cfg.fast_threshold
            )

    def detect(self, img_gray: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect feature points.

        Returns
        -------
        np.ndarray (N, 2) float32 — (x, y) coordinates
        """
        cfg = self.config

        if cfg.grid_mask:
            return self._detect_grid(img_gray, mask)

        if cfg.detector == "fast":
            kps = self._fast.detect(img_gray, mask)
            if not kps:
                return np.empty((0, 2), dtype=np.float32)
            # Sort by response, keep top N
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[:cfg.max_features]
            pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
        else:
            use_harris = (cfg.detector == "harris")
            corners = cv2.goodFeaturesToTrack(
                img_gray, cfg.max_features, cfg.quality_level,
                cfg.min_distance, mask=mask,
                useHarrisDetector=use_harris, k=0.04
            )
            if corners is None:
                return np.empty((0, 2), dtype=np.float32)
            pts = corners.reshape(-1, 2)

        # Sub-pixel refinement
        if cfg.use_subpix and len(pts) > 0:
            pts = pts.reshape(-1, 1, 2).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.01)
            cv2.cornerSubPix(img_gray, pts, (5, 5), (-1, -1), criteria)
            pts = pts.reshape(-1, 2)

        return pts

    def _detect_grid(self, img_gray: np.ndarray,
                      global_mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Detect features enforcing at most max_features/N² points per grid cell.
        This prevents all features clustering in one textured region
        (e.g., only in the corner of the frame).
        """
        h, w = img_gray.shape
        n = self.config.grid_cells
        pts_per_cell = max(1, self.config.max_features // (n * n))
        all_pts = []

        for row in range(n):
            for col in range(n):
                # Build cell mask
                y1 = int(row * h / n)
                y2 = int((row + 1) * h / n)
                x1 = int(col * w / n)
                x2 = int((col + 1) * w / n)

                cell_mask = np.zeros_like(img_gray)
                cell_mask[y1:y2, x1:x2] = 255

                if global_mask is not None:
                    cell_mask = cv2.bitwise_and(cell_mask, global_mask)

                # Detect in cell
                cfg = self.config
                corners = cv2.goodFeaturesToTrack(
                    img_gray, pts_per_cell, cfg.quality_level,
                    cfg.min_distance // 2, mask=cell_mask
                )
                if corners is not None:
                    all_pts.append(corners.reshape(-1, 2))

        if not all_pts:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.vstack(all_pts).astype(np.float32)

        # Sub-pixel
        if self.config.use_subpix:
            pts_cv = pts.reshape(-1, 1, 2)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.01)
            cv2.cornerSubPix(img_gray, pts_cv, (5, 5), (-1, -1), criteria)
            pts = pts_cv.reshape(-1, 2)

        return pts

    def visualize(self, img_gray: np.ndarray, pts: np.ndarray,
                   title: str = "") -> np.ndarray:
        """Draw detected points on the image."""
        vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for x, y in pts.astype(int):
            cv2.circle(vis, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
        n = len(pts)
        cv2.putText(vis, f"{title}: {n} pts",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2, cv2.LINE_AA)
        return vis


# ── Demo: Compare uniform vs grid distribution ────────────────────────────────
def make_skewed_image():
    """Image with texture concentrated in one corner — shows grid value."""
    img = np.zeros((300, 400), dtype=np.uint8)
    # Dense texture only in top-left
    for y in range(0, 100, 8):
        for x in range(0, 150, 8):
            cv2.rectangle(img, (x, y), (x+5, y+5), 180, -1)
    # Sparse elsewhere
    cv2.circle(img, (300, 200), 60, 150, -1)
    cv2.rectangle(img, (200, 200), (380, 280), 130, -1)
    noise = np.random.normal(0, 5, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


img_skewed = make_skewed_image()

# Without grid: all features cluster in top-left texture
cfg_uniform = TrackerConfig(detector="shi_tomasi", max_features=100,
                              min_distance=10, use_subpix=True, grid_mask=False)
# With grid: features distributed across entire image
cfg_grid    = TrackerConfig(detector="shi_tomasi", max_features=100,
                              min_distance=5, use_subpix=True,
                              grid_mask=True, grid_cells=5)

init_uniform = FeatureInitializer(cfg_uniform)
init_grid    = FeatureInitializer(cfg_grid)

pts_uniform = init_uniform.detect(img_skewed)
pts_grid    = init_grid.detect(img_skewed)

vis_uniform = init_uniform.visualize(img_skewed, pts_uniform, "Uniform")
vis_grid    = init_grid.visualize(img_skewed, pts_grid, "Grid (5×5)")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(img_skewed, cmap="gray"); axes[0].set_title("Input (texture only top-left)"); axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(vis_uniform, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Uniform detection\n→ clusters in texture"); axes[1].axis("off")
axes[2].imshow(cv2.cvtColor(vis_grid, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Grid-enforced detection\n→ spread across frame"); axes[2].axis("off")
plt.suptitle("Grid-Based vs Uniform Feature Distribution", fontsize=12)
plt.tight_layout(); plt.show()

print(f"Uniform: {len(pts_uniform)} points (mostly clustered)")
print(f"Grid:    {len(pts_grid)} points (distributed)")
```

---

## 13.9 Quick Reference

```python
import cv2
import numpy as np

img = np.zeros((256, 256), dtype=np.uint8)

# ── Harris corner detection ──────────────────────────────────────────────────
R = cv2.cornerHarris(
    img.astype(np.float32),
    blockSize=3,   # neighborhood size (gradient averaging window)
    ksize=3,       # Sobel aperture size (1,3,5,7)
    k=0.04         # Harris free parameter (0.04–0.06 typical)
)
# R > 0 = corner, R < 0 = edge, |R| ≈ 0 = flat

# Sub-pixel refinement
corners_f32 = np.array([[[100.0, 100.0]]], dtype=np.float32)
cv2.cornerSubPix(img, corners_f32, (5,5), (-1,-1),
                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

# Min eigenvalue (Shi-Tomasi response without NMS)
min_eig = cv2.cornerMinEigenVal(img.astype(np.float32), blockSize=3, ksize=3)

# ── Shi-Tomasi (goodFeaturesToTrack) ─────────────────────────────────────────
corners = cv2.goodFeaturesToTrack(
    img,
    maxCorners=200,      # return at most N strongest corners
    qualityLevel=0.01,   # threshold as fraction of best corner quality
    minDistance=10,      # minimum Euclidean distance between corners
    mask=None,           # optional binary mask (uint8)
    blockSize=3,         # structure tensor window
    useHarrisDetector=False,  # True = Harris response, False = Shi-Tomasi
    k=0.04               # Harris k (only used if useHarrisDetector=True)
)
# corners shape: (N, 1, 2) float32 or None

# ── FAST ─────────────────────────────────────────────────────────────────────
fast = cv2.FastFeatureDetector_create(
    threshold=10,           # intensity difference threshold
    nonmaxSuppression=True, # suppress non-maxima
    type=cv2.FastFeatureDetector_TYPE_9_16  # 9/16, 7/12, or 5/8
)
kps = fast.detect(img, None)   # list of cv2.KeyPoint

# ── Drawing keypoints ────────────────────────────────────────────────────────
vis = cv2.drawKeypoints(img, kps, None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT            # small dot
    # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # sized circle + orientation
)

# ── KeyPoint attributes ───────────────────────────────────────────────────────
# kp.pt       : (x, y) float — position
# kp.size     : diameter of meaningful neighborhood
# kp.angle    : orientation in degrees (-1 if not computed)
# kp.response : detector response (used for ranking)
# kp.octave   : pyramid level (0 for scale-invariant detectors)
# kp.class_id : cluster id (for classification use cases)
```

---

## 13.10 Summary

**What makes a good feature:** Repeatability, distinctiveness, and localization accuracy. Corners satisfy all three: they change appearance under any shift direction, making them uniquely locatable.

**Structure tensor M:** The 2×2 matrix `[[Σ Iₓ², Σ IₓIᵧ], [Σ IₓIᵧ, Σ Iᵧ²]]` summarizes local gradient distribution. Its eigenvalues λ₁, λ₂ classify every pixel: both large = corner; one large = edge; both small = flat.

**Harris detector (`cv2.cornerHarris`):** Approximates eigenvalue classification via `R = det(M) - k·trace(M)²`. Avoids expensive eigenvalue computation. Requires empirical `k` (0.04 standard). Returns a float response map; apply NMS manually or via dilation. Refine to sub-pixel with `cv2.cornerSubPix`.

**Shi-Tomasi (`cv2.goodFeaturesToTrack`):** Uses `R = min(λ₁, λ₂)` — cleaner, no empirical k. One-call interface handles detection + NMS + optional sub-pixel. Returns (N, 1, 2) float32. The default choice for optical flow initialization (`cv2.calcOpticalFlowPyrLK` in Chapter 19).

**FAST (`cv2.FastFeatureDetector`):** Circle-based test on 16 pixels. No eigenvalues, no convolutions — 10–50× faster than Harris/Shi-Tomasi. Threshold controls sensitivity. TYPE_9_16 (default) requires 9 contiguous pixels. Does not compute orientation. Ideal for real-time SLAM, tracking on mobile devices.

**Grid-based distribution:** Prevents all features clustering in one textured region. Split the image into N×N cells and detect independently in each for spatially uniform coverage.

---

## 13.11 Exercises

### Warm-up

**13.1** Implement `corner_classification_demo(img)` that computes the structure tensor manually, finds all three eigenvalue combinations, and produces a color-coded visualization: green=corner (λ₁>T, λ₂>T), red=edge (λ₁>T, λ₂<T/10), gray=flat. Compare to `cv2.cornerHarris`.

**13.2** Apply `cv2.goodFeaturesToTrack` to a real image with three different `(qualityLevel, minDistance)` combinations: (0.001, 3), (0.01, 10), (0.1, 20). Report: number of corners, fraction in the top-left quadrant, fraction in the bottom-right. Do this for 5 different images and observe if the relative distribution changes.

**13.3** Benchmark all three detectors on a 1920×1080 grayscale image: measure detection time (50 runs), count keypoints, report keypoints per millisecond. Use a bar chart to visualize the throughput comparison.

### Core

**13.4** Implement a `corner_repeatability_study(img, n_trials=20)` function: for each trial, add Gaussian noise (σ=15), detect corners with all three methods, and compute the repeatability score (fraction of original corners re-detected within 5 pixels). Plot distributions as box plots for all three methods.

**13.5** Implement the Harris corner response from scratch using only `cv2.Sobel` and `cv2.GaussianBlur`. Verify numerically that your result matches `cv2.cornerHarris` (max absolute difference < 1% of max response). Then show a side-by-side: corners from your implementation vs `cv2.cornerHarris`.

**13.6** Build a `pyramid_corner_detect(img_gray, n_levels=4, scale=0.5)` function that: (a) builds an image pyramid, (b) applies Shi-Tomasi at each level, (c) maps corner coordinates back to the original scale, (d) removes duplicates within 5 pixels. Compare the multi-scale detected corners against single-scale detection on an image with both coarse and fine features.

**13.7** Implement `grid_detector_compare(img, grid_sizes=[3,5,8])`: detect Shi-Tomasi features using uniform detection vs 3×3, 5×5, and 8×8 grids. For each, compute the spatial entropy (how uniformly distributed the features are across the image using a grid-based histogram). Plot entropy vs grid_size.

### Challenge

**13.8** Build an **adaptive threshold FAST detector**: given a target number of keypoints T, binary-search over FAST threshold values to find the threshold that produces closest to T keypoints. The search should converge in ≤10 iterations. Demonstrate that you can reliably produce 50, 100, 200, and 500 keypoints on 5 different images.

**13.9** Implement a **scale-space corner detector**: build a Gaussian pyramid, compute Harris response at each scale, and combine detections. A corner at scale s should only be accepted if it is also a local maximum in scale (i.e., stronger than the response at scales s-1 and s+1 at the same spatial location). This is the foundation of SIFT's DoG detector. Compare to single-scale Harris on an image viewed from two different distances.

**13.10** Build a **complete feature tracking initializer** for a video: (a) on frame 0, detect features using grid-based Shi-Tomasi; (b) on each subsequent frame, apply `cv2.calcOpticalFlowPyrLK` to track; (c) when the number of tracked features drops below 50% of the initial count, re-detect and merge new features while avoiding already-tracked positions; (d) visualize tracks as colored trails. Test on a video file or webcam feed and report the mean track lifetime in frames.

---

## Further Reading

- **OpenCV Harris tutorial:** https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
- **OpenCV Shi-Tomasi tutorial:** https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html
- **OpenCV FAST tutorial:** https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html
- **Harris, C. & Stephens, M. (1988). "A combined corner and edge detector."** Proc. 4th Alvey Vision Conference — the original Harris paper
- **Shi, J. & Tomasi, C. (1994). "Good features to track."** CVPR — the Shi-Tomasi paper
- **Rosten, E. & Drummond, T. (2006). "Machine learning for high-speed corner detection."** ECCV — the original FAST paper
- **OpenCV feature detection reference:** https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html
