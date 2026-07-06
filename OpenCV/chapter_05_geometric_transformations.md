# Chapter 5: Geometric Transformations

> *"All image formation is a projection. Understanding how points map between planes is the key that unlocks camera calibration, augmented reality, panoramic stitching, and document scanning."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Build and apply the 2×3 affine transformation matrix for translation, rotation, scaling, shearing, and reflection
2. Understand homogeneous coordinates and why they unify all linear spatial transforms
3. Use `cv2.warpAffine` and `cv2.warpPerspective` with all border mode options
4. Compute rotation matrices about arbitrary centers with `cv2.getRotationMatrix2D`
5. Derive a perspective transform from four point correspondences with `cv2.getPerspectiveTransform`
6. Understand what homography is, how it relates to perspective, and how to compute it robustly with `cv2.findHomography` + RANSAC
7. Apply perspective correction to build a document scanner
8. Use `cv2.remap` for arbitrary pixel-level remapping and lens distortion simulation
9. Chain transforms by matrix multiplication
10. Choose border modes and interpolation methods appropriately

---

## 5.1 The Geometry of Image Transformations

Before touching any OpenCV function, we need a mental model of what a transformation does at the mathematical level. This pays dividends throughout the rest of the book.

### 5.1.1 The Two Viewpoints: Forward and Inverse Mapping

When you apply a transformation to an image, there are two ways to think about what happens:

**Forward mapping** asks: "Given pixel (x, y) in the source, where does it go in the destination?" The problem with forward mapping is that multiple source pixels might land on the same destination pixel (collisions), and some destination pixels might receive no contribution at all (holes).

**Inverse mapping** asks: "Given pixel (x', y') in the destination, where does it come from in the source?" OpenCV's `cv2.warpAffine` and `cv2.warpPerspective` use inverse mapping — for each destination pixel, they compute the source location and interpolate the source value. This guarantees every destination pixel is filled, with no holes.

```
Inverse mapping (what OpenCV does):

For each (x', y') in dst:
    (x, y) = M_inverse * (x', y')          # transform destination → source
    dst[y', x'] = interpolate(src, x, y)   # sample source at fractional coords
```

This is why you pass the *forward* transformation matrix to `warpAffine` but OpenCV internally inverts it (unless you pass `cv2.WARP_INVERSE_MAP`).

### 5.1.2 Homogeneous Coordinates

To represent translation as a matrix multiplication (instead of vector addition), we use **homogeneous coordinates**: a point (x, y) in 2D becomes (x, y, 1) in 3D homogeneous space.

A point in homogeneous coordinates (wx, wy, w) maps to Euclidean (x/w, y/w) when w ≠ 1.

```python
import numpy as np

# Euclidean point
p = np.array([100, 200])

# Homogeneous equivalent
p_h = np.array([100, 200, 1])

# Now translation, rotation, scaling, and perspective can ALL be matrices
# Translation by (tx, ty):
#   [1  0  tx] [x]   [x + tx]
#   [0  1  ty] [y] = [y + ty]
#   [0  0   1] [1]   [  1   ]

def make_translation_matrix(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0,  1]], dtype=np.float64)

T = make_translation_matrix(50, 100)
p_transformed = T @ p_h    # matrix multiply
print(f"Translated: {p_transformed[:2]}")   # [150, 300]
```

### 5.1.3 The Transformation Hierarchy

Transformations form a strict hierarchy — each level is a special case of the next:

```
Translation (2 DOF)
  └─ Rigid / Euclidean (3 DOF: tx, ty, θ)
       └─ Similarity (4 DOF: tx, ty, θ, s)
            └─ Affine (6 DOF: any 2×3 matrix)
                 └─ Projective / Homography (8 DOF: any 3×3 matrix, up to scale)
```

| Transform | Matrix | DOF | Preserves | Use case |
|-----------|--------|-----|-----------|----------|
| Translation | 2×3 (bottom row fixed) | 2 | Shape, size, angles | Shift crop window |
| Euclidean | 2×3 (rotation + translation) | 3 | Shape, size | Image registration |
| Similarity | 2×3 (rotation + scale + translation) | 4 | Shape, angles | Thumbnail rescaling |
| Affine | 2×3 (any) | 6 | Parallel lines, ratios | Skew correction |
| Projective | 3×3 | 8 | Straight lines only | Perspective correction, AR |

---

## 5.2 Affine Transformations

An affine transformation maps every point (x, y) to (x', y') via:

```
x' = M[0,0]*x + M[0,1]*y + M[0,2]
y' = M[1,0]*x + M[1,1]*y + M[1,2]
```

Or in matrix form with homogeneous coordinates: `[x', y', 1]ᵀ = M_3x3 * [x, y, 1]ᵀ` where the 2×3 OpenCV matrix is the top two rows of M_3x3.

### 5.2.1 Translation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_or_create(path="sample.jpg", shape=(300, 400, 3)):
    img = cv2.imread(path)
    if img is None:
        img = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (350, 250), (0, 180, 220), -1)
        cv2.circle(img, (200, 150), 80, (220, 80, 20), -1)
        cv2.putText(img, "Test", (120, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 3, cv2.LINE_AA)
    return img

img = load_or_create()
h, w = img.shape[:2]

# Translation: move (tx, ty) pixels right and down
tx, ty = 80, 50

# 2x3 affine matrix for pure translation
M_translate = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

# cv2.warpAffine(src, M, dsize, flags, borderMode, borderValue)
# dsize = (output_width, output_height)
translated = cv2.warpAffine(img, M_translate, (w, h))

# To see the full translated image without clipping, expand canvas:
translated_full = cv2.warpAffine(img, M_translate, (w + tx, h + ty))
print(f"Original: {img.shape}, Translated (same size): {translated.shape}, Full: {translated_full.shape}")
```

### 5.2.2 Rotation

`cv2.getRotationMatrix2D` builds the 2×3 rotation matrix for rotation about an arbitrary center:

```python
import cv2
import numpy as np

img = load_or_create()
h, w = img.shape[:2]

# Rotate 30° clockwise about the image center
# cv2.getRotationMatrix2D(center, angle, scale)
# angle: POSITIVE = counter-clockwise (OpenCV convention)
center = (w // 2, h // 2)
angle  = 30    # degrees counter-clockwise
scale  = 1.0   # no zoom

M_rot = cv2.getRotationMatrix2D(center, angle, scale)
print(f"Rotation matrix:\n{M_rot}")
# [[cos(θ)  sin(θ)   (1-cos)·cx - sin·cy]
#  [-sin(θ) cos(θ)   sin·cx + (1-cos)·cy]]

rotated = cv2.warpAffine(img, M_rot, (w, h))

# ── Rotate without cropping corners ──────────────────────────────────────────
def rotate_bound(image, angle_deg):
    """
    Rotate image by angle_deg without clipping corners.
    Expands the canvas to fit the rotated image.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Compute new bounding dimensions
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Adjust the translation column to center in the new canvas
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    return cv2.warpAffine(image, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))

rotated_full = rotate_bound(img, 45)
print(f"Original: {img.shape[:2]}, Rotated 45° (no clip): {rotated_full.shape[:2]}")
```

### 5.2.3 Scaling

```python
import cv2
import numpy as np

img = load_or_create()
h, w = img.shape[:2]

# Scale about origin (top-left)
sx, sy = 1.5, 0.75  # 1.5× horizontally, 0.75× vertically

M_scale = np.float32([
    [sx, 0,  0],
    [0,  sy, 0]
])
scaled = cv2.warpAffine(img, M_scale, (int(w * sx), int(h * sy)))

# Scale about image center (avoids shift to corner)
cx, cy = w / 2.0, h / 2.0
M_scale_center = np.float32([
    [sx, 0,  cx * (1 - sx)],
    [0,  sy, cy * (1 - sy)]
])
scaled_center = cv2.warpAffine(img, M_scale_center, (w, h))

# NOTE: For simple isotropic scaling, cv2.resize is cleaner and faster
small  = cv2.resize(img, None, fx=0.5,  fy=0.5,  interpolation=cv2.INTER_AREA)
large  = cv2.resize(img, None, fx=2.0,  fy=2.0,  interpolation=cv2.INTER_CUBIC)
exact  = cv2.resize(img, (320, 240),             interpolation=cv2.INTER_LINEAR)
```

### 5.2.4 Shearing

```python
import cv2
import numpy as np

img = load_or_create()
h, w = img.shape[:2]

# Horizontal shear: x' = x + shear_factor * y
shear_x = 0.3

M_shear_x = np.float32([
    [1, shear_x, 0],
    [0, 1,       0]
])
sheared_x = cv2.warpAffine(img, M_shear_x, (int(w + h * shear_x), h))

# Vertical shear: y' = shear_factor * x + y
shear_y = 0.2
M_shear_y = np.float32([
    [1,       0, 0],
    [shear_y, 1, 0]
])
sheared_y = cv2.warpAffine(img, M_shear_y, (w, int(h + w * shear_y)))
```

### 5.2.5 Flipping

```python
import cv2
import numpy as np

img = load_or_create()

# cv2.flip is the idiomatic way
flip_h    = cv2.flip(img, 1)    # flipCode=1: horizontal
flip_v    = cv2.flip(img, 0)    # flipCode=0: vertical
flip_both = cv2.flip(img, -1)   # flipCode=-1: both

# Equivalent via affine (educational — cv2.flip is faster)
h, w = img.shape[:2]
M_flip_h = np.float32([[-1, 0, w - 1], [0, 1, 0]])   # mirror x: x' = w-1-x
flip_h_affine = cv2.warpAffine(img, M_flip_h, (w, h))
```

### 5.2.6 Computing Affine Matrix from 3 Point Correspondences

When you know where 3 specific points should map to, `cv2.getAffineTransform` computes the exact 2×3 matrix:

```python
import cv2
import numpy as np

img = load_or_create()
h, w = img.shape[:2]

# Source points: top-left, top-right, bottom-left (float32 required)
src_pts = np.float32([
    [0,   0  ],
    [w-1, 0  ],
    [0,   h-1]
])

# Destination points: apply skew and mild shear
dst_pts = np.float32([
    [w * 0.1, h * 0.15],
    [w * 0.9, h * 0.05],
    [w * 0.2, h * 0.90]
])

M = cv2.getAffineTransform(src_pts, dst_pts)
print(f"Computed affine matrix:\n{M}")

warped = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

# Draw the corresponding points for verification
for (sx, sy), (dx, dy) in zip(src_pts.astype(int), dst_pts.astype(int)):
    cv2.circle(img,    (sx, sy), 8, (0, 255, 0), -1)
    cv2.circle(warped, (dx, dy), 8, (0, 255, 0), -1)
```

### 5.2.7 Chaining Affine Transforms

Matrix multiplication chains transforms. The order matters:

```python
import numpy as np
import cv2

img = load_or_create()
h, w = img.shape[:2]

# Helper: promote 2x3 → 3x3 for matrix multiplication
def to_3x3(M_2x3):
    return np.vstack([M_2x3, [0, 0, 1]])

def to_2x3(M_3x3):
    return M_3x3[:2, :]

# Three individual transforms
M_t = to_3x3(np.float32([[1, 0, -w/2], [0, 1, -h/2]]))   # translate to origin
M_r = to_3x3(cv2.getRotationMatrix2D((0, 0), 45, 1.0))    # rotate about origin
M_b = to_3x3(np.float32([[1, 0,  w/2], [0, 1,  h/2]]))   # translate back

# Combined: rotate about image center = translate → rotate → translate back
# Matrix multiplication order: rightmost applied first
M_combined = M_b @ M_r @ M_t
warped = cv2.warpAffine(img, to_2x3(M_combined), (w, h))

# Verify: this equals getRotationMatrix2D(center, 45, 1.0)
M_direct = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
diff = np.abs(to_2x3(M_combined) - M_direct)
print(f"Max difference from direct rotation: {diff.max():.8f}")   # ~0 (floating-point eps)
```

---

## 5.3 Border Modes

When a transformation maps source pixels outside the image bounds, OpenCV needs a strategy for filling those areas:

```python
import cv2
import numpy as np

img = load_or_create()
h, w = img.shape[:2]

M = cv2.getRotationMatrix2D((w//2, h//2), 20, 1.0)

border_modes = [
    (cv2.BORDER_CONSTANT,   "BORDER_CONSTANT\n(default, black fill)"),
    (cv2.BORDER_REPLICATE,  "BORDER_REPLICATE\n(stretch edge pixels)"),
    (cv2.BORDER_REFLECT,    "BORDER_REFLECT\n(mirror at edge: dcba|abcd)"),
    (cv2.BORDER_REFLECT_101,"BORDER_REFLECT_101\n(mirror, don't repeat edge: dcb|abcd)"),
    (cv2.BORDER_WRAP,       "BORDER_WRAP\n(tile the image)"),
]

results = []
for mode, label in border_modes:
    warped = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=mode,
                             borderValue=(128, 128, 128))  # gray fill for CONSTANT
    # Add label
    cv2.putText(warped, label.split('\n')[0], (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    results.append(warped)

# BORDER_CONSTANT with custom color
warped_colored = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 100, 0))  # blue fill
```

---

## 5.4 Perspective Transformations

A perspective (projective) transformation maps straight lines to straight lines but does **not** preserve parallelism. It models what happens when you photograph a flat surface from an angle.

### 5.4.1 The 3×3 Homography Matrix

The perspective transform uses a 3×3 matrix with 8 degrees of freedom (9 elements, scale-invariant):

```
dst_x = (H[0,0]*x + H[0,1]*y + H[0,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
dst_y = (H[1,0]*x + H[1,1]*y + H[1,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
```

The division by the third row (w = H[2,0]x + H[2,1]y + H[2,2]) is what makes it projective — it causes the "foreshortening" effect you see when looking at a tilted plane.

### 5.4.2 cv2.getPerspectiveTransform and cv2.warpPerspective

Four point correspondences determine the perspective transform uniquely:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Simulated document scanner ────────────────────────────────────────────────
# Create a test document image
doc = np.full((400, 300, 3), 240, dtype=np.uint8)  # off-white background
# Add text lines
for y in range(60, 360, 30):
    cv2.line(doc, (30, y), (270, y), (180, 180, 180), 1)
cv2.putText(doc, "DOCUMENT", (50, 40),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
cv2.rectangle(doc, (20, 20), (280, 380), (40, 40, 40), 3)

# Simulate photographing at an angle: apply a perspective warp to "distort" it
h_d, w_d = doc.shape[:2]

# A perspective distortion (like taking a photo from bottom-left angle)
src_pts_fwd = np.float32([
    [0,     0    ],
    [w_d-1, 0    ],
    [w_d-1, h_d-1],
    [0,     h_d-1]
])

# Destination: trapezoidal shape simulating perspective view
dst_pts_fwd = np.float32([
    [40,   60  ],
    [340,  20  ],
    [380,  480 ],
    [0,    500 ]
])

M_fwd = cv2.getPerspectiveTransform(src_pts_fwd, dst_pts_fwd)
distorted = cv2.warpPerspective(doc, M_fwd, (400, 520))

# ── Now correct the perspective back (the document scanner step) ──────────────
# The user (or algorithm) identifies the 4 corners of the document
# in the distorted image and we warp back to a rectangle.

# Known corners of the distorted document
doc_corners_in_photo = dst_pts_fwd   # same as our forward destination

# Target rectangle (output document size)
output_w, output_h = 300, 400
target_rect = np.float32([
    [0,         0        ],
    [output_w-1,0        ],
    [output_w-1,output_h-1],
    [0,         output_h-1]
])

# Compute the perspective correction matrix
M_correct = cv2.getPerspectiveTransform(doc_corners_in_photo, target_rect)
corrected  = cv2.warpPerspective(distorted, M_correct, (output_w, output_h),
                                  flags=cv2.INTER_CUBIC)

# Visual comparison
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
for ax, im, title in zip(axes,
    [doc, distorted, corrected],
    ["Original document", "Distorted (camera angle)", "Perspective-corrected"]):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

### 5.4.3 cv2.findHomography: Robust Estimation with RANSAC

When point correspondences are noisy or contain outliers (as they always are in real-world feature matching), `cv2.getPerspectiveTransform` (which requires exactly 4 points) breaks down. `cv2.findHomography` robustly estimates the homography from many correspondences using RANSAC:

```python
import cv2
import numpy as np

# cv2.findHomography(srcPoints, dstPoints, method, ransacReprojThreshold)
# method:
#   0                    — least-squares (all points, no outlier rejection)
#   cv2.RANSAC           — Random Sample Consensus (robust)
#   cv2.LMEDS            — Least Median of Squares (robust, no threshold needed)
#   cv2.RHO              — PROSAC/RHO (faster than RANSAC for well-conditioned data)

# Simulate matched point pairs with 20% outliers
np.random.seed(42)
n_pts = 50

# True homography (a mild perspective warp)
H_true = np.array([
    [1.2,  0.1,  30.0],
    [0.05, 0.9, -15.0],
    [0.001, 0.0002, 1.0]
], dtype=np.float64)

# Generate source points
src_pts = np.column_stack([
    np.random.uniform(50, 400, n_pts),
    np.random.uniform(50, 300, n_pts)
]).astype(np.float32)

# Generate (noisy) destination points
src_h   = np.column_stack([src_pts, np.ones(n_pts)])
dst_h   = (H_true @ src_h.T).T
dst_pts = (dst_h[:, :2] / dst_h[:, 2:3]).astype(np.float32)
dst_pts += np.random.normal(0, 2, dst_pts.shape)   # add Gaussian noise

# Inject 10 outliers
n_out = 10
dst_pts[:n_out] += np.random.uniform(50, 150, (n_out, 2))

# ── Method 1: RANSAC (recommended) ────────────────────────────────────────────
H_ransac, inlier_mask = cv2.findHomography(
    src_pts, dst_pts,
    cv2.RANSAC,
    ransacReprojThreshold=5.0   # pixels
)
n_inliers = inlier_mask.sum()
print(f"RANSAC: {n_inliers}/{n_pts} inliers ({n_inliers/n_pts*100:.0f}%)")
print(f"Estimated H:\n{H_ransac}")

# ── Method 2: LMEDS (no threshold needed, better with <50% outliers) ─────────
H_lmeds, mask_lmeds = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
print(f"\nLMEDS inliers: {mask_lmeds.sum()}")

# ── Apply to warp a full image ─────────────────────────────────────────────────
img = load_or_create()
h_img, w_img = img.shape[:2]

warped = cv2.warpPerspective(img, H_ransac, (w_img, h_img),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
```

### 5.4.4 When to Use Affine vs Perspective

```python
# Affine (2×3 / warpAffine) — use when:
# - You have 3 known point pairs
# - The transformation is caused by camera movement WITHOUT rotation about the viewing axis
# - You're working with: copy/paste at an angle, simple skew correction
# - Parallelism must be preserved (e.g., grid lines stay parallel)

# Perspective (3×3 / warpPerspective) — use when:
# - You have 4+ known point pairs
# - The object is a flat plane photographed at an angle
# - Parallelism is NOT preserved (far lines appear to converge)
# - Use cases: document scanning, whiteboard rectification, AR planar tracking

# Rule of thumb:
# - Is the scene a flat plane? → Perspective
# - Is it a 3D object / camera panning? → Affine or homography
# - Are there noisy correspondences? → findHomography (RANSAC)
```

---

## 5.5 Interpolation: Sampling Between Pixels

When a transformation maps a destination pixel to a fractional source coordinate (e.g., x=142.7, y=88.3), OpenCV must interpolate the source value. The choice of interpolation affects quality and speed:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = load_or_create()

# Upscale 4× using each interpolation method
METHODS = {
    "NEAREST"  : cv2.INTER_NEAREST,
    "LINEAR"   : cv2.INTER_LINEAR,
    "CUBIC"    : cv2.INTER_CUBIC,
    "AREA"     : cv2.INTER_AREA,    # not meaningful for upscaling
    "LANCZOS4" : cv2.INTER_LANCZOS4,
}

results = {}
for name, method in METHODS.items():
    results[name] = cv2.resize(img, None, fx=4.0, fy=4.0, interpolation=method)

# Compare on a small crop (to see pixel differences clearly)
crop_region = (80, 120, 40, 40)   # x, y, w, h
crops = {name: res[crop_region[1]:crop_region[1]+crop_region[3]*4,
                   crop_region[0]:crop_region[0]+crop_region[2]*4]
         for name, res in results.items()}

# Nearest: pixelated — visible squares
# Linear: smooth but slightly blurry
# Cubic: smooth, slightly sharper than linear
# Lanczos4: highest quality, slight ringing artifacts near sharp edges
```

**Interpolation method selection guide:**

| Method | `cv2.INTER_*` | Speed | Quality | Use when |
|--------|--------------|-------|---------|----------|
| Nearest | `NEAREST` | ★★★★★ | ★☆☆☆☆ | Pixel art, mask operations |
| Bilinear | `LINEAR` | ★★★★☆ | ★★★☆☆ | General upscaling (default) |
| Bicubic | `CUBIC` | ★★★☆☆ | ★★★★☆ | Higher quality upscaling |
| Area | `AREA` | ★★★★☆ | ★★★★☆ | **Downscaling** (moire-free) |
| Lanczos4 | `LANCZOS4` | ★★☆☆☆ | ★★★★★ | Highest quality upscaling |

---

## 5.6 Custom Pixel-Level Remapping

`cv2.remap` is the most general spatial transformation — you directly specify where every destination pixel comes from in the source:

```python
import cv2
import numpy as np

img = load_or_create()
h, w = img.shape[:2]

# cv2.remap(src, map1, map2, interpolation, borderMode, borderValue)
# map1: x-coordinates (column) in source for each destination pixel
# map2: y-coordinates (row) in source for each destination pixel
# Both maps have shape (dst_h, dst_w) and dtype float32

# Build coordinate grids
Y, X = np.mgrid[0:h, 0:w].astype(np.float32)

# ── Example 1: Horizontal wave distortion ────────────────────────────────────
amplitude = 15.0   # pixels
frequency = 0.03   # cycles per pixel

# Each row is shifted horizontally by a sine wave
map_x_wave = X + amplitude * np.sin(frequency * 2 * np.pi * Y)
map_y_wave = Y.copy()   # no vertical displacement

wave_distorted = cv2.remap(img, map_x_wave, map_y_wave,
                            cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)

# ── Example 2: Barrel (fisheye) distortion ────────────────────────────────────
cx, cy = w / 2.0, h / 2.0
k1 = 0.0003   # barrel coefficient (positive = barrel, negative = pincushion)

# Normalize coordinates to [-1, 1] relative to center
x_norm = (X - cx) / cx
y_norm = (Y - cy) / cy
r_sq   = x_norm**2 + y_norm**2

# Barrel distortion model: undistorted_r = r * (1 + k1 * r^2)
# Map back to pixel coordinates
factor = 1.0 + k1 * r_sq
map_x_barrel = cx + (x_norm * factor) * cx
map_y_barrel = cy + (y_norm * factor) * cy

barrel = cv2.remap(img, map_x_barrel.astype(np.float32),
                   map_y_barrel.astype(np.float32),
                   cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)

# ── Example 3: Polar coordinate transform (cartesian → polar) ─────────────────
# Map: for each output pixel (angle, radius) → source pixel (x, y)
out_h, out_w = h, w
max_r = min(cx, cy)

# Output: rows = radius (0 to max_r), cols = angle (0 to 2π)
theta_arr = np.linspace(0, 2 * np.pi, out_w, dtype=np.float32)
r_arr     = np.linspace(0, max_r,     out_h, dtype=np.float32)
THETA, R  = np.meshgrid(theta_arr, r_arr)

map_x_polar = (cx + R * np.cos(THETA)).astype(np.float32)
map_y_polar = (cy + R * np.sin(THETA)).astype(np.float32)

polar = cv2.remap(img, map_x_polar, map_y_polar,
                  cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

# Display all remapping results
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, im, title in zip(axes.flat,
    [img, wave_distorted, barrel, polar],
    ["Original", "Wave distortion", "Barrel distortion", "Polar transform"]):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 5.7 Complete Project: Document Scanner

Combining perspective correction, mouse interaction, and image processing into a functional document scanner:

```python
import cv2
import numpy as np
from typing import List, Tuple, Optional


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    Works regardless of the order they were clicked.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    pts  = pts.reshape(4, 2)

    # Sum: top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    # Difference: top-right has smallest diff, bottom-left has largest diff
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left

    return rect


def four_point_transform(image: np.ndarray,
                          pts: np.ndarray) -> np.ndarray:
    """
    Apply a perspective transform to extract and rectify a quadrilateral region.

    Parameters
    ----------
    image : np.ndarray  BGR image
    pts   : np.ndarray  (4, 2) float array of corner points

    Returns
    -------
    np.ndarray  Rectified, perspective-corrected crop
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Compute output width: max of top and bottom edge lengths
    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    out_w = int(max(width_top, width_bottom))

    # Compute output height: max of left and right edge lengths
    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_h = int(max(height_left, height_right))

    # Target rectangle (axis-aligned output)
    dst = np.float32([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1]
    ])

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h),
                                  flags=cv2.INTER_CUBIC)
    return warped


def auto_detect_document(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Attempt to automatically detect a document's four corners using
    edge detection and contour finding.

    Returns (4, 2) float32 array of corners, or None if detection fails.
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # Dilate to close small gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    # Find contours and look for the largest quadrilateral
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        # Approximate contour to polygon
        peri    = cv2.arcLength(contour, True)
        approx  = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            # Found a quadrilateral
            area = cv2.contourArea(approx)
            img_area = image.shape[0] * image.shape[1]
            if area > 0.1 * img_area:   # must be at least 10% of image area
                return approx.reshape(4, 2).astype(np.float32)

    return None


def scan_document(image_path: str) -> dict:
    """
    Full document scanning pipeline:
    1. Load image
    2. Attempt auto-detection of document corners
    3. Apply perspective correction
    4. Enhance contrast for readability

    Returns dict with 'original', 'corrected', 'enhanced', 'corners'
    """
    img = cv2.imread(image_path)
    if img is None:
        # Create synthetic test document
        img = np.full((520, 400, 3), 200, dtype=np.uint8)
        for y in range(80, 460, 28):
            cv2.line(img, (40, y), (360, y), (160, 160, 160), 1)
        cv2.putText(img, "INVOICE #0042", (60, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.rectangle(img, (20, 20), (380, 500), (30, 30, 30), 3)

        # Simulate perspective distortion
        h_i, w_i = img.shape[:2]
        src = np.float32([[0,0],[w_i,0],[w_i,h_i],[0,h_i]])
        dst = np.float32([[30,20],[w_i-10,0],[w_i+20,h_i],[0,h_i+15]])
        H   = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, H, (w_i+50, h_i+30))

    corners = auto_detect_document(img)

    if corners is not None:
        corrected = four_point_transform(img, corners)
    else:
        print("Auto-detection failed — using full image")
        h_i, w_i = img.shape[:2]
        corners   = np.float32([[0,0],[w_i,0],[w_i,h_i],[0,h_i]])
        corrected = img.copy()

    # Enhance: convert to grayscale, apply adaptive threshold for "scanned" look
    gray_doc  = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    enhanced  = cv2.adaptiveThreshold(
        gray_doc, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=11, C=7
    )

    return {
        "original"  : img,
        "corners"   : corners,
        "corrected" : corrected,
        "enhanced"  : enhanced
    }


# ── Run the scanner ────────────────────────────────────────────────────────────
result = scan_document("document.jpg")   # will use synthetic if file missing

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, (key, title) in zip(axes, [
    ("original",  "Original (with distortion)"),
    ("corrected", "Perspective corrected"),
    ("enhanced",  "Scanned (binarized)")
]):
    im = result[key]
    if im.ndim == 2:
        ax.imshow(im, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 5.8 Summary

**Affine transformations** use a 2×3 matrix and are applied with `cv2.warpAffine`. They preserve parallelism and are appropriate for translation, rotation, scaling, shearing, and any combination thereof. Use `cv2.getRotationMatrix2D` for rotation, `cv2.getAffineTransform` for 3-point correspondences, and chain transforms via matrix multiplication.

**Perspective transformations** use a 3×3 matrix and are applied with `cv2.warpPerspective`. They preserve straight lines but not parallelism. Use `cv2.getPerspectiveTransform` for 4 exact points or `cv2.findHomography` with RANSAC for robust estimation from many noisy correspondences.

**Border modes** control what fills areas outside the source. `BORDER_REFLECT_101` is the most common choice for filtering; `BORDER_CONSTANT` with black is the default for rectification.

**Interpolation methods**: `INTER_AREA` for downscaling; `INTER_LINEAR` or `INTER_CUBIC` for upscaling; `INTER_NEAREST` for masks.

**`cv2.remap`** provides the most general per-pixel mapping — use it for lens distortion, artistic effects, and polar transforms.

---

## 5.9 Exercises

### Warm-up

**5.1** Write a function `translate(img, tx, ty)` that translates an image by (tx, ty) pixels using `cv2.warpAffine`. Handle the case where the canvas should expand to show the full translated image (no clipping).

**5.2** Use `rotate_bound` from Section 5.2.2 to rotate an image from 0° to 360° in steps of 10° and save all frames as a GIF or video. Use `cv2.VideoWriter` to write the video.

**5.3** Demonstrate `cv2.remap` by applying a horizontal sine-wave distortion to a checkerboard image. Show three amplitudes (5px, 15px, 30px) side by side. Label each panel with the amplitude.

### Core

**5.4** Implement a `perspective_correct(img, corners)` function where `corners` is a (4,2) array of float32 points in any order. Use `order_points` to sort them, then call `four_point_transform`. Test it with at least two different quadrilateral shapes and compare the output aspect ratios to the expected values.

**5.5** Build a `TileTransformer` class that tiles an image `n×n` times using `warpAffine` with translation matrices. Then apply a global perspective warp to the entire tiled canvas. Show the result for n=3.

**5.6** Use `cv2.findHomography` with RANSAC to register two images: (a) an original image and (b) a version of it that has been perspective-warped and had 30% of pixels randomly corrupted with noise. Report the number of inliers found, the reprojection error on inliers, and overlay the registered image on the original using `cv2.addWeighted`.

### Challenge

**5.7** Implement a **full interactive document scanner** using mouse clicks: the user clicks 4 corners of the document, the application shows a live rubber-band quadrilateral, and on ENTER applies `four_point_transform` and shows the result. Add a "enhance" key that binarizes the output using `cv2.adaptiveThreshold`.

**5.8** Implement the `barrel_undistort(img, k1, k2=0)` function that corrects barrel/pincushion distortion using `cv2.remap`. Build an interactive version with two trackbars controlling k1 and k2. Demonstrate that the result visually straightens lines that were curved.

**5.9** Build a **virtual camera simulator** using `cv2.warpPerspective`: given an image of a flat plane, simulate what it looks like when the virtual camera rotates around the x-axis (tilt) and y-axis (pan). Parameterize by three angles (pan, tilt, roll) and compute the 3D→2D projection matrix analytically. Render for 12 evenly spaced pan angles (0°–330°) and stitch them into a single display strip.

---

## Further Reading

- **OpenCV geometric transformations reference:** https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
- **OpenCV homography tutorial:** https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
- **Hartley & Zisserman, "Multiple View Geometry in Computer Vision"** — the definitive reference on projective geometry for computer vision
- **Richard Szeliski, "Computer Vision: Algorithms and Applications" (2nd ed, 2022):** Chapter 2 covers image formation and spatial transforms
