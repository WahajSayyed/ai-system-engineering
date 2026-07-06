# Chapter 2: Images as Arrays — The NumPy Foundation

> *"NumPy is to Python what a scalpel is to a surgeon — the tool that makes everything else possible."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain how digital images map to NumPy `ndarray` objects and describe every attribute: `shape`, `dtype`, `strides`, `flags`
2. Use all forms of NumPy indexing — basic, slice, fancy, boolean — to read and write pixels
3. Apply NumPy broadcasting rules to perform per-channel and per-row/column image operations without loops
4. Perform safe arithmetic on images using proper dtype management
5. Use masking to selectively modify image regions
6. Concatenate, stack, split, and tile images using NumPy
7. Convert between `uint8`, `float32`, and `float64` image representations correctly
8. Profile image operations and apply vectorization principles for performance
9. Read images using Pillow and understand the NumPy ↔ Pillow ↔ OpenCV conversion chain
10. Build a reusable image utility library used throughout the rest of this book

---

## 2.1 The ndarray: A Mental Model

NumPy's `ndarray` (N-dimensional array) is the single data structure that underlies virtually all scientific computing in Python. For image processing, it is not just a convenient representation — it *is* the image. OpenCV has no separate image object. When you call `cv2.imread()`, you get an `ndarray`. When you pass an image to `cv2.cvtColor()`, you are passing an `ndarray`. The entire OpenCV Python API speaks `ndarray`.

Let us build a thorough mental model before touching any image files.

### 2.1.1 Array Attributes You Must Know

Every `ndarray` carries a set of metadata that describes its shape, type, and memory layout. These attributes are your first diagnostic tools.

```python
import numpy as np
import cv2

# Load a sample image (or create one if unavailable)
img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# ── The most important attributes ────────────────────────────────────────────

print(f"shape    : {img.shape}")
# (480, 640, 3)
# Interpretation: 480 rows × 640 columns × 3 channels
# shape[0] = height, shape[1] = width, shape[2] = number of channels

print(f"dtype    : {img.dtype}")
# uint8
# unsigned 8-bit integer: values 0–255

print(f"ndim     : {img.ndim}")
# 3 (three axes: row, column, channel)

print(f"size     : {img.size}")
# 921600 = 480 × 640 × 3 total elements

print(f"nbytes   : {img.nbytes}")
# 921600 bytes = 900 KB (each uint8 is 1 byte)

print(f"itemsize : {img.itemsize}")
# 1 byte per element (uint8)

print(f"strides  : {img.strides}")
# (1920, 3, 1)
# To move one row down:    skip 1920 bytes (= 640 cols × 3 channels × 1 byte)
# To move one column right:  skip 3 bytes (= 3 channels × 1 byte)
# To move one channel:       skip 1 byte
```

The `strides` attribute is particularly illuminating. It tells NumPy exactly how many bytes to skip in memory to move one step along each axis. Understanding strides explains why certain slicing operations are O(1) — they just change metadata pointers, not data.

### 2.1.2 The Memory Layout of a Color Image

A color image with shape `(H, W, 3)` and `dtype=uint8` is stored as a flat sequence of bytes in memory. The default layout is **C-contiguous** (row-major), meaning rows are stored consecutively:

```
Memory address:  0   1   2   3   4   5   6   7   8   9  ...
Data:           B00 G00 R00 B01 G01 R01 B02 G02 R02 B03 ...
                |_______|   |_______|   |_______|
                pixel(0,0)  pixel(0,1)  pixel(0,2)
```

Where `Bxy`, `Gxy`, `Rxy` denote the Blue, Green, Red values of the pixel at row `x`, column `y`.

This layout — channels interleaved, columns fastest, rows slowest — is called **HWC** format (Height × Width × Channel). It is what OpenCV uses natively. Deep learning frameworks often prefer **CHW** format (Channel × Height × Width). We will handle this conversion explicitly when needed.

```python
import numpy as np

# Verify the memory layout with a tiny 2x2 color image
tiny = np.array([
    [[10, 20, 30],   [40, 50, 60]],   # row 0: pixel(0,0) and pixel(0,1)
    [[70, 80, 90],   [100, 110, 120]] # row 1: pixel(1,0) and pixel(1,1)
], dtype=np.uint8)

print(f"Shape  : {tiny.shape}")    # (2, 2, 3)
print(f"Strides: {tiny.strides}")  # (6, 3, 1)
#   Moving down 1 row  → skip 6 bytes (= 2 cols × 3 channels × 1 byte)
#   Moving right 1 col → skip 3 bytes (= 3 channels × 1 byte)
#   Moving 1 channel   → skip 1 byte

# The flat byte sequence in memory:
print(f"Flat bytes: {tiny.flatten()}")
# [ 10  20  30  40  50  60  70  80  90 100 110 120]
#  |_B__|_G__|_R__| pixel(0,0)
#            |_B__|_G__|_R__| pixel(0,1)
#                       |_B__|_G__|_R__| pixel(1,0)
#                                  |_B__|_G__|_R__| pixel(1,1)
```

---

## 2.2 Data Types in Image Processing

The `dtype` (data type) of your image array determines the valid value range, memory usage per pixel, and what arithmetic operations will behave correctly. Using the wrong dtype is one of the most common sources of silent bugs in image processing pipelines.

### 2.2.1 The dtype Landscape

| dtype | Range | Bytes/pixel | Primary Use |
|-------|-------|-------------|-------------|
| `uint8` | 0 – 255 | 1 | Display, storage, most OpenCV operations |
| `uint16` | 0 – 65,535 | 2 | Medical imaging (DICOM), HDR, depth maps |
| `int16` | -32,768 – 32,767 | 2 | Intermediate computations (e.g., Sobel output) |
| `int32` | -2B – 2B | 4 | Accumulation buffers, histograms |
| `float32` | ±3.4×10³⁸ | 4 | Deep learning, intermediate processing, float images |
| `float64` | ±1.8×10³⁰⁸ | 8 | High-precision scientific work |
| `bool` | True/False | 1 | Binary masks |

The everyday workhorse is `uint8`. But you will reach for `float32` constantly for intermediate computations. Here is why:

```python
import numpy as np

# ── The uint8 arithmetic problem ─────────────────────────────────────────────
a = np.array([200], dtype=np.uint8)
b = np.array([100], dtype=np.uint8)

# Direct numpy addition: wraps around (modular arithmetic)
print(f"uint8 + uint8  : {a + b}")   # [44]  ← WRONG: 300 mod 256 = 44

# The correct approach for intermediate image math:
a_f = a.astype(np.float32)
b_f = b.astype(np.float32)
result = np.clip(a_f + b_f, 0, 255).astype(np.uint8)
print(f"float32 + clip : {result}")  # [255] ← CORRECT: saturated addition

# OpenCV's safe arithmetic uses saturation semantics:
# cv2.add() saturates at 255 (see Chapter 1)
# cv2.subtract() saturates at 0
```

### 2.2.2 Normalizing to [0.0, 1.0]

Many image processing algorithms and all deep learning frameworks expect images normalized to `[0.0, 1.0]` as `float32`. The conversion is simple but the direction matters:

```python
import numpy as np
import cv2

img_uint8 = cv2.imread("sample.jpg")
if img_uint8 is None:
    img_uint8 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# uint8 [0, 255] → float32 [0.0, 1.0]
img_f32 = img_uint8.astype(np.float32) / 255.0

print(f"uint8  dtype: {img_uint8.dtype}, range: [{img_uint8.min()}, {img_uint8.max()}]")
print(f"float32 dtype: {img_f32.dtype}, range: [{img_f32.min():.4f}, {img_f32.max():.4f}]")

# float32 [0.0, 1.0] → uint8 [0, 255]
# ALWAYS clip before casting — values outside [0, 1] will wrap silently
img_back = np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8)

# Verify round-trip fidelity
diff = np.abs(img_uint8.astype(np.int32) - img_back.astype(np.int32))
print(f"Max round-trip error: {diff.max()} (should be 0)")
```

> **NumPy 2.0 note:** Starting with NumPy 2.0 (released June 2024), scalar promotion rules changed. Scalar-array operations now follow stricter rules: `np.uint8_array / 255` now returns `float64` due to Python scalar literal `255` being treated as a `float64` scalar. To stay in `float32`, be explicit: `img.astype(np.float32) / np.float32(255.0)`.

```python
# NumPy 2.0 safe normalization
img_f32 = img_uint8.astype(np.float32) / np.float32(255.0)   # stays float32
img_f64 = img_uint8.astype(np.float32) / 255.0               # may upcast to float64
print(f"Safe:   {img_f32.dtype}")   # float32
print(f"Unsafe: {img_f64.dtype}")   # float64 in NumPy 2.x
```

### 2.2.3 Type Conversion Cheat Sheet

```python
import numpy as np

img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# To float32 [0, 1] — most common
f32 = img.astype(np.float32) / np.float32(255.0)

# To float64 [0, 1] — high precision
f64 = img.astype(np.float64) / 255.0

# float32 [0, 1] → uint8 [0, 255]
back_uint8 = np.clip(f32 * 255.0, 0, 255).astype(np.uint8)

# uint8 → int16 for signed arithmetic (e.g., image differences)
i16 = img.astype(np.int16)

# Difference without overflow
diff = img.astype(np.int16) - img[::-1].astype(np.int16)  # can be negative
diff_abs = np.abs(diff).clip(0, 255).astype(np.uint8)

# uint16 (e.g., from depth camera) → uint8 for display
depth_u16 = np.random.randint(500, 5000, (480, 640), dtype=np.uint16)
# Normalize to full 8-bit range
depth_vis = ((depth_u16.astype(np.float32) - depth_u16.min()) /
             (depth_u16.max() - depth_u16.min()) * 255).astype(np.uint8)
```

---

## 2.3 Indexing: Reading and Writing Pixels

NumPy provides several indexing mechanisms. Understanding all of them is essential — you will use every one of them in a real computer vision pipeline.

### 2.3.1 Basic (Scalar) Indexing

The simplest form: access individual elements by their indices.

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# Access a single pixel — returns an array of 3 channel values (BGR)
pixel = img[100, 200]          # row=100, col=200
print(f"Pixel at (x=200, y=100): BGR={pixel}")
# e.g., [142  87  63]

# Access individual channels
blue  = img[100, 200, 0]       # Blue channel value
green = img[100, 200, 1]       # Green channel value
red   = img[100, 200, 2]       # Red channel value
print(f"B={blue}, G={green}, R={red}")

# Write a single pixel value — modifies in place
img[100, 200] = [0, 255, 0]    # Set to pure green (in BGR)
img[100, 200, 2] = 255         # Set only the Red channel

# Grayscale image — single value per pixel
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
value = gray[100, 200]         # Returns a Python scalar (uint8)
print(f"Grayscale pixel: {value}")
gray[100, 200] = 128           # Set to mid-gray
```

> **Performance note:** Accessing individual pixels in a Python loop is extremely slow — even for a 640×480 image, a double loop visits 307,200 pixels. At ~1 µs per Python iteration, that is ~0.3 seconds just for access. Always vectorize. We will demonstrate this explicitly in Section 2.8.

### 2.3.2 Slice Indexing — Regions of Interest (ROI)

Slicing is how you extract and manipulate rectangular regions:

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

h, w = img.shape[:2]

# Basic slice: img[row_start:row_end, col_start:col_end]
# Note: end index is EXCLUSIVE (standard Python)

# Extract top-left 100×150 region
roi = img[0:100, 0:150]          # shape: (100, 150, 3)
print(f"ROI shape: {roi.shape}")

# Extract center crop
cy, cx = h // 2, w // 2
crop = img[cy-100:cy+100, cx-150:cx+150]   # 200×300 center crop

# Every other row and column (downsampling by 2)
downsampled = img[::2, ::2]      # shape: (240, 320, 3)
print(f"Downsampled: {downsampled.shape}")

# Reverse rows (vertical flip) — returns a VIEW, not a copy
flipped_v = img[::-1, :, :]
# Reverse cols (horizontal flip)
flipped_h = img[:, ::-1, :]
# Both flips
flipped_both = img[::-1, ::-1, :]

# Extract a single channel — returns 2D array
blue_channel  = img[:, :, 0]    # shape: (480, 640)
green_channel = img[:, :, 1]
red_channel   = img[:, :, 2]

# Slice assignment — paste a region
canvas = np.zeros_like(img)
canvas[50:150, 75:225] = img[50:150, 75:225]  # Copy a region to canvas
```

**The View vs Copy distinction with slices:**

```python
import numpy as np

img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
original_pixel = img[50, 50].copy()

# Slices return VIEWS — same underlying memory
roi_view = img[40:60, 40:60]          # not a copy!
roi_view[:] = 0                        # zeros out the region in img too
print(f"img[50,50] after modifying roi_view: {img[50, 50]}")
# [0, 0, 0] — img was modified!

# Restore and use .copy() for independence
img[40:60, 40:60] = original_pixel
roi_copy = img[40:60, 40:60].copy()   # independent copy
roi_copy[:] = 255                      # img is NOT modified
print(f"img[50,50] after modifying roi_copy: {img[50, 50]}")
# original value — img is intact ✓
```

### 2.3.3 Fancy (Advanced) Indexing

Fancy indexing uses arrays of indices instead of slices. Unlike slices, fancy indexing always returns a **copy**.

```python
import numpy as np

img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# Index with arrays of row and column indices
rows = np.array([10, 50, 100, 200])
cols = np.array([15, 75, 150, 300])

# Gather pixels at these (row, col) pairs — returns shape (4, 3)
pixels = img[rows, cols]
print(f"Selected pixels:\n{pixels}")

# Gather entire rows
selected_rows = img[[0, 100, 200, 300]]    # shape: (4, 640, 3)

# Gather entire columns
selected_cols = img[:, [0, 100, 200, 300]] # shape: (480, 4, 3)

# Scatter: write values to specific locations
img[rows, cols] = [255, 0, 0]  # Set those pixels to blue (BGR)
```

### 2.3.4 Boolean (Mask) Indexing

Boolean indexing uses a binary mask to select pixels. This is one of the most powerful and frequently used operations in image processing.

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# ── Example 1: Threshold on grayscale ────────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a boolean mask — True where pixel is bright (> 200)
bright_mask = gray > 200         # shape: (480, 640), dtype: bool
print(f"Bright pixels: {bright_mask.sum()}")    # number of True values
print(f"Fraction:      {bright_mask.mean():.3f}")

# Apply mask: set all bright pixels to white
result = img.copy()
result[bright_mask] = [255, 255, 255]   # only modifies pixels where mask=True
# Note: boolean mask on a 3-channel image modifies all channels

# ── Example 2: Color range selection (skin-tone detection) ───────────────────
# In BGR: select pixels where all channels are in a certain range
lower = np.array([20, 50, 100], dtype=np.uint8)
upper = np.array([60, 150, 220], dtype=np.uint8)

# cv2.inRange creates a binary mask (0 or 255, NOT bool)
mask_cv = cv2.inRange(img, lower, upper)
print(f"cv2.inRange dtype: {mask_cv.dtype}")    # uint8
print(f"cv2.inRange values: {np.unique(mask_cv)}")  # [0, 255]

# Convert to bool for NumPy boolean indexing
mask_bool = mask_cv > 0   # or: mask_cv.astype(bool)

# ── Example 3: Dark pixel replacement ────────────────────────────────────────
dark_mask = gray < 30              # pixels darker than threshold
dark_result = img.copy()
dark_result[dark_mask] = [200, 200, 200]   # set dark pixels to light gray

# ── Example 4: Combining masks with logical operators ────────────────────────
# Pixels that are bright AND in the left half of the image
left_half = np.zeros(gray.shape, dtype=bool)
left_half[:, :gray.shape[1]//2] = True

bright_left_mask = bright_mask & left_half       # AND
bright_or_left   = bright_mask | left_half       # OR
bright_not       = ~bright_mask                  # NOT (invert)

print(f"Bright AND left half: {bright_left_mask.sum()} pixels")
```

---

## 2.4 Broadcasting: Vectorized Operations Without Loops

Broadcasting is NumPy's mechanism for applying operations between arrays of different shapes. It is the key to writing fast, loop-free image processing code.

### 2.4.1 The Broadcasting Rules

NumPy compares shapes element-by-element from the trailing axis inward. Two dimensions are compatible if they are equal, or if one of them is 1 (which gets "stretched").

```
Image shape:   (480, 640, 3)
Scalar:               ()      → broadcast to (480, 640, 3)
Per-channel:        (3,)      → broadcast to (480, 640, 3)
Per-row:      (480,  1, 1)    → broadcast to (480, 640, 3)
Per-column:   (  1, 640, 1)   → broadcast to (480, 640, 3)
```

### 2.4.2 Per-Channel Operations

The most common broadcasting scenario: apply a different value to each channel.

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

# ── Brightness adjustment ─────────────────────────────────────────────────────
# Add 50 to all channels — WRONG with uint8 (may overflow)
# brighter_wrong = img + 50   # uint8 wraps!

# Correct: use float32 intermediate
img_f = img.astype(np.float32)
brighter = np.clip(img_f + 50.0, 0, 255).astype(np.uint8)

# ── Per-channel scaling ──────────────────────────────────────────────────────
# Boost only the Red channel (index 2 in BGR)
# Wrong naive approach: loops
# Correct: broadcasting with explicit channel array
boost = np.array([1.0, 1.0, 1.5], dtype=np.float32)  # shape: (3,)
# img_f has shape (480, 640, 3)
# boost has shape (3,) → broadcasts to (480, 640, 3)
boosted = np.clip(img_f * boost, 0, 255).astype(np.uint8)

# ── White balance simulation ─────────────────────────────────────────────────
# Apply different gains to each BGR channel
# Warmer image: boost red, reduce blue
wb_gains = np.array([0.85, 1.0, 1.2], dtype=np.float32)  # [B_gain, G_gain, R_gain]
warm_img = np.clip(img_f * wb_gains, 0, 255).astype(np.uint8)

# ── Subtract per-channel mean (mean centering) ──────────────────────────────
# Used in deep learning preprocessing
channel_means = img_f.mean(axis=(0, 1))   # shape: (3,) — mean per channel
print(f"Channel means (BGR): {channel_means}")
# e.g., [102.3  98.7  91.2]
mean_centered = img_f - channel_means     # broadcasts (480,640,3) - (3,)
print(f"Centered range: [{mean_centered.min():.1f}, {mean_centered.max():.1f}]")
```

### 2.4.3 Per-Row and Per-Column Operations

```python
import numpy as np

img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8).astype(np.float32)

# ── Vignette effect: darken edges, brighten center ─────────────────────────
h, w = img.shape[:2]

# Create a 1D gradient along columns: bright in center, dark at edges
col_fade = np.abs(np.linspace(-1, 1, w))       # shape: (640,)
col_weight = 1.0 - col_fade * 0.5              # shape: (640,) — range [0.5, 1.0]

# Reshape for broadcasting: need shape (1, 640, 1)
col_weight = col_weight.reshape(1, w, 1)       # shape: (1, 640, 1)
# Broadcasts with (480, 640, 3) → each column gets its weight, applied to all channels

# Same for rows
row_fade = np.abs(np.linspace(-1, 1, h))       # shape: (480,)
row_weight = 1.0 - row_fade * 0.5             # shape: (480,)
row_weight = row_weight.reshape(h, 1, 1)       # shape: (480, 1, 1)

# Combined vignette: multiply both weights
vignette_mask = row_weight * col_weight        # shape: (480, 640, 1) via broadcasting
print(f"Vignette mask shape: {vignette_mask.shape}")

vignette_img = np.clip(img * vignette_mask, 0, 255).astype(np.uint8)

# ── Horizontal gradient overlay ──────────────────────────────────────────────
# Add a gradient that goes from 0 (left) to 80 (right) to Red channel only
gradient = np.linspace(0, 80, w, dtype=np.float32)  # shape: (640,)
gradient = gradient.reshape(1, w)                    # shape: (1, 640)

# Apply only to Red channel (index 2)
img_copy = img.copy()
img_copy[:, :, 2] = np.clip(img_copy[:, :, 2] + gradient, 0, 255)
```

### 2.4.4 Image Arithmetic: Blending and Compositing

```python
import numpy as np
import cv2

# Create two test images
h, w = 400, 600
img1 = np.zeros((h, w, 3), dtype=np.uint8)
img2 = np.zeros((h, w, 3), dtype=np.uint8)

# Solid colors
img1[:] = [0, 100, 200]    # Orange (BGR: B=0, G=100, R=200)
img2[:] = [100, 50, 0]     # Dark blue (BGR: B=100, G=50, R=0)

# ── Alpha blending: out = α*img1 + (1-α)*img2 ────────────────────────────────
alpha = 0.6   # 60% img1, 40% img2
blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, gamma=0)
# gamma=0 means no brightness offset added

# Manual equivalent:
blended_manual = np.clip(
    img1.astype(np.float32) * alpha + img2.astype(np.float32) * (1 - alpha),
    0, 255
).astype(np.uint8)

# ── Linear blend transition (alpha varies by column) ─────────────────────────
alpha_map = np.linspace(0, 1, w, dtype=np.float32)  # 0.0 → 1.0 left to right
alpha_map = alpha_map.reshape(1, w, 1)              # (1, W, 1) for broadcasting

blend_transition = np.clip(
    img1.astype(np.float32) * (1 - alpha_map) +
    img2.astype(np.float32) * alpha_map,
    0, 255
).astype(np.uint8)

# ── Absolute difference (change detection) ────────────────────────────────────
# Use int16 to handle signed differences
diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
diff_u8 = diff.clip(0, 255).astype(np.uint8)

# ── Screen blend mode (like Photoshop) ────────────────────────────────────────
# screen(a, b) = 1 - (1-a)(1-b)
a = img1.astype(np.float32) / 255.0
b = img2.astype(np.float32) / 255.0
screen = (1.0 - (1.0 - a) * (1.0 - b))
screen_u8 = (screen * 255).clip(0, 255).astype(np.uint8)

# ── Multiply blend mode ───────────────────────────────────────────────────────
multiply = (a * b * 255).clip(0, 255).astype(np.uint8)
```

---

## 2.5 Image Manipulation with NumPy

### 2.5.1 Cropping, Padding, and Resizing

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

h, w = img.shape[:2]

# ── Cropping with slice ───────────────────────────────────────────────────────
def crop_center(image, crop_h, crop_w):
    """Crop a centered region of size crop_h × crop_w."""
    h, w = image.shape[:2]
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]

center_crop = crop_center(img, 200, 300)
print(f"Center crop shape: {center_crop.shape}")  # (200, 300, 3)

# ── Padding with np.pad ──────────────────────────────────────────────────────
# Add 20px black border on all sides
padded_black = np.pad(
    img,
    pad_width=((20, 20), (20, 20), (0, 0)),  # (top, bot), (left, right), no channel pad
    mode="constant",
    constant_values=0
)
print(f"After padding: {padded_black.shape}")  # (520, 680, 3)

# Reflective padding (avoids edge artifacts in filtering)
padded_reflect = np.pad(
    img,
    pad_width=((10, 10), (10, 10), (0, 0)),
    mode="reflect"
)

# ── Resizing ─────────────────────────────────────────────────────────────────
# Use cv2.resize — do NOT use array slicing for resizing (it just skips rows)
# cv2.resize(src, (width, height), interpolation)  ← NOTE: width first!
small = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
large = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_LINEAR)

# Resize maintaining aspect ratio
def resize_keep_aspect(image, target_width):
    h, w = image.shape[:2]
    ratio = target_width / w
    new_h = int(h * ratio)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)

resized = resize_keep_aspect(img, 400)
print(f"Aspect-preserved resize: {resized.shape}")
```

**Interpolation methods for resizing:**

| Flag | Method | Use When |
|------|--------|----------|
| `cv2.INTER_NEAREST` | Nearest neighbor | Speed critical, pixel art |
| `cv2.INTER_LINEAR` | Bilinear (default) | General upsampling |
| `cv2.INTER_AREA` | Pixel area relation | **Downsampling** (avoids moire) |
| `cv2.INTER_CUBIC` | Bicubic (4×4) | Higher quality upsampling |
| `cv2.INTER_LANCZOS4` | Lanczos (8×8) | Highest quality upsampling |

> **Rule of thumb:** Use `INTER_AREA` when making images smaller. Use `INTER_LINEAR` or `INTER_CUBIC` when making them larger.

### 2.5.2 Flipping, Rotating, and Transposing

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# ── Flipping ──────────────────────────────────────────────────────────────────
# Using cv2 (preferred — guaranteed C-contiguous output)
flip_h = cv2.flip(img, 1)       # horizontal flip (flip code 1)
flip_v = cv2.flip(img, 0)       # vertical flip (flip code 0)
flip_both = cv2.flip(img, -1)   # both (flip code -1)

# Using NumPy (returns views — may need np.ascontiguousarray)
flip_h_np = np.ascontiguousarray(img[:, ::-1])
flip_v_np = np.ascontiguousarray(img[::-1, :])

# ── Rotating 90/180/270 degrees ───────────────────────────────────────────────
# cv2.rotate is the fastest for exact 90° increments
rot90  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rot180 = cv2.rotate(img, cv2.ROTATE_180)
rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Using NumPy np.rot90 (returns views)
rot90_np = np.ascontiguousarray(np.rot90(img, k=1))  # 1 = 90° CCW

# ── Transposing axes ─────────────────────────────────────────────────────────
# Swap rows and columns (diagonal flip)
transposed = img.transpose(1, 0, 2)   # new shape: (640, 480, 3)
print(f"Original: {img.shape}, Transposed: {transposed.shape}")

# HWC → CHW (needed for deep learning frameworks)
chw = img.transpose(2, 0, 1)   # (3, 480, 640)
print(f"CHW shape: {chw.shape}")

# CHW → HWC (back from deep learning output)
hwc = chw.transpose(1, 2, 0)   # (480, 640, 3)
```

### 2.5.3 Concatenating and Stacking Images

```python
import numpy as np
import cv2

# Two images of the same height
img1 = np.zeros((400, 300, 3), dtype=np.uint8)
img2 = np.zeros((400, 300, 3), dtype=np.uint8)
img1[:] = [255, 100, 50]
img2[:] = [50, 100, 255]

# ── Side by side (horizontal concatenation) ──────────────────────────────────
side_by_side = np.concatenate([img1, img2], axis=1)   # along columns
print(f"Side by side: {side_by_side.shape}")           # (400, 600, 3)
# Equivalently:
side_by_side = np.hstack([img1, img2])

# ── Top and bottom (vertical concatenation) ──────────────────────────────────
top_bottom = np.concatenate([img1, img2], axis=0)     # along rows
top_bottom = np.vstack([img1, img2])
print(f"Top/bottom: {top_bottom.shape}")              # (800, 300, 3)

# ── Create a 2×2 image grid ─────────────────────────────────────────────────
def make_grid(images, nrow, padding=2, pad_value=128):
    """
    Arrange a list of images into a grid.

    Parameters
    ----------
    images : list of np.ndarray
        All images must have the same shape (H, W, C).
    nrow : int
        Number of images per row.
    padding : int
        Pixels of padding between images.
    pad_value : int
        Background fill value (0=black, 255=white, 128=gray).

    Returns
    -------
    np.ndarray
        Grid image.
    """
    n = len(images)
    h, w = images[0].shape[:2]
    c = images[0].shape[2] if images[0].ndim == 3 else 1
    ncol = (n + nrow - 1) // nrow  # ceiling division

    # Pad images list to fill grid evenly
    pad_img = np.full((h, w, c) if c > 1 else (h, w),
                      pad_value, dtype=images[0].dtype)
    images = images + [pad_img] * (nrow * ncol - n)

    rows = []
    for i in range(0, len(images), nrow):
        row_imgs = images[i:i + nrow]
        if padding > 0:
            pad_strip = np.full(
                (h, padding, c) if c > 1 else (h, padding),
                pad_value, dtype=images[0].dtype
            )
            row_with_pads = []
            for j, ri in enumerate(row_imgs):
                row_with_pads.append(ri)
                if j < len(row_imgs) - 1:
                    row_with_pads.append(pad_strip)
            row = np.concatenate(row_with_pads, axis=1)
        else:
            row = np.concatenate(row_imgs, axis=1)
        rows.append(row)

    if padding > 0:
        h_pad = np.full((padding, row.shape[1], c) if c > 1
                        else (padding, row.shape[1]),
                        pad_value, dtype=images[0].dtype)
        rows_with_pads = []
        for j, r in enumerate(rows):
            rows_with_pads.append(r)
            if j < len(rows) - 1:
                rows_with_pads.append(h_pad)
        grid = np.concatenate(rows_with_pads, axis=0)
    else:
        grid = np.concatenate(rows, axis=0)

    return grid


# Usage
img_list = [
    np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    for _ in range(6)
]
grid = make_grid(img_list, nrow=3, padding=4)
print(f"Grid shape: {grid.shape}")   # (204, 312, 3) with 4px padding, 3 cols, 2 rows
```

### 2.5.4 Splitting into Channels and Merging

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)

# ── Split with cv2 ────────────────────────────────────────────────────────────
b, g, r = cv2.split(img)
print(f"Channel shapes: B={b.shape}, G={g.shape}, R={r.shape}")
# Each is (200, 300) — 2D array

# ── Split with NumPy (faster, returns views) ──────────────────────────────────
b_np = img[:, :, 0]   # view — same memory
g_np = img[:, :, 1]
r_np = img[:, :, 2]

# ── Merge back ────────────────────────────────────────────────────────────────
merged = cv2.merge([b, g, r])
print(f"Merged shape: {merged.shape}")  # (200, 300, 3)

# ── Merge with np.stack ───────────────────────────────────────────────────────
merged_np = np.stack([b, g, r], axis=-1)  # axis=-1 = last axis (channels)
print(f"np.stack result: {merged_np.shape}")  # (200, 300, 3)

# ── Build a grayscale image from only the Red channel ─────────────────────────
# Display each channel as grayscale alongside the original
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
axes[0].imshow(img_rgb);      axes[0].set_title("Original");  axes[0].axis("off")
axes[1].imshow(b, cmap="Blues_r"); axes[1].set_title("B Channel"); axes[1].axis("off")
axes[2].imshow(g, cmap="Greens_r"); axes[2].set_title("G Channel"); axes[2].axis("off")
axes[3].imshow(r, cmap="Reds_r");  axes[3].set_title("R Channel"); axes[3].axis("off")
plt.tight_layout()
plt.show()
```

---

## 2.6 Masking: Selective Image Modification

Masks are fundamental to nearly every non-trivial image processing task. A mask tells an algorithm *where* to operate. Understanding how to create, combine, and apply masks is essential.

### 2.6.1 Creating Masks

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

h, w = img.shape[:2]

# ── Rectangular mask ─────────────────────────────────────────────────────────
rect_mask = np.zeros((h, w), dtype=np.uint8)   # start with black (0)
rect_mask[100:300, 150:450] = 255               # white in the rectangle

# ── Circular mask ─────────────────────────────────────────────────────────────
circle_mask = np.zeros((h, w), dtype=np.uint8)
center = (w // 2, h // 2)
radius = min(h, w) // 3
cv2.circle(circle_mask, center, radius, color=255, thickness=-1)  # filled

# ── Elliptical mask ───────────────────────────────────────────────────────────
ellipse_mask = np.zeros((h, w), dtype=np.uint8)
cv2.ellipse(
    ellipse_mask,
    center=(w//2, h//2),
    axes=(w//3, h//4),
    angle=30,
    startAngle=0, endAngle=360,
    color=255, thickness=-1
)

# ── Gradient (soft) mask ──────────────────────────────────────────────────────
# Create a soft vignette mask that fades to black at edges
Y, X = np.mgrid[0:h, 0:w]   # coordinate grids, shape (h, w) each
cy, cx = h / 2, w / 2
# Distance from center, normalized
dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
soft_mask = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
print(f"Soft mask range: [{soft_mask.min():.3f}, {soft_mask.max():.3f}]")
# Center=1.0, edges approach 0.0
```

### 2.6.2 Applying Masks

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

h, w = img.shape[:2]

# Create a circular mask
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (w//2, h//2), min(h,w)//3, 255, -1)
bool_mask = mask.astype(bool)

# ── Method 1: Boolean indexing ────────────────────────────────────────────────
result1 = img.copy()
result1[~bool_mask] = 0   # set outside circle to black

# ── Method 2: cv2.bitwise_and (fastest, in-place friendly) ───────────────────
# mask must be single-channel uint8 (0 or 255)
result2 = cv2.bitwise_and(img, img, mask=mask)
# where mask=0 → output=0; where mask=255 → output=src

# ── Method 3: 3-channel mask for per-pixel float blending ────────────────────
soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)  # (h, w, 3)
blended = (img.astype(np.float32) * soft_mask_3ch).clip(0, 255).astype(np.uint8)

# ── Combining with a background ───────────────────────────────────────────────
background = np.full_like(img, fill_value=50)  # dark gray background
mask_f = (mask / 255.0).reshape(h, w, 1)       # float mask, shape (h,w,1)

composite = (
    img.astype(np.float32) * mask_f +
    background.astype(np.float32) * (1.0 - mask_f)
).clip(0, 255).astype(np.uint8)

# ── cv2.copyTo: safe masked copy ─────────────────────────────────────────────
# Copies src to dst only where mask is non-zero
canvas = np.full_like(img, 200)  # light gray canvas
img.copyTo(canvas, mask=mask)    # copy circle region of img onto canvas
# Note: this is a method on ndarray when using OpenCV Python bindings
# Equivalent to: canvas[bool_mask] = img[bool_mask]
```

---

## 2.7 Coordinate Grids and Spatial Operations

Many image processing algorithms need to work with explicit pixel coordinates. NumPy provides efficient tools for this.

### 2.7.1 np.mgrid and np.meshgrid

```python
import numpy as np
import matplotlib.pyplot as plt

h, w = 300, 400

# ── np.mgrid: generate coordinate grids ──────────────────────────────────────
# Returns Y coordinates (rows) and X coordinates (columns)
Y, X = np.mgrid[0:h, 0:w]      # Y shape: (300, 400), X shape: (300, 400)
print(f"Y[0,0]={Y[0,0]}, Y[-1,0]={Y[-1,0]}")   # 0, 299  (row indices)
print(f"X[0,0]={X[0,0]}, X[0,-1]={X[0,-1]}")   # 0, 399  (col indices)

# ── np.meshgrid: alternative (prefer mgrid for integer grids) ─────────────────
x_range = np.linspace(0, 1, w)
y_range = np.linspace(0, 1, h)
X_float, Y_float = np.meshgrid(x_range, y_range)
# X_float[i, j] = x coordinate of pixel (i, j) in [0, 1]
# Y_float[i, j] = y coordinate of pixel (i, j) in [0, 1]

# ── Distance map from center ─────────────────────────────────────────────────
cy, cx = h / 2, w / 2
dist = np.sqrt((X - cx)**2 + (Y - cy)**2)   # shape: (h, w)
print(f"Max distance from center: {dist.max():.1f}")

# ── Radial gradient (useful for vignette, lens distortion sim) ────────────────
max_dist = np.sqrt(cx**2 + cy**2)
radial = (dist / max_dist * 255).clip(0, 255).astype(np.uint8)

# ── Concentric circles using modulo ──────────────────────────────────────────
rings = (dist % 30).astype(np.uint8) * 8   # rings every 30 pixels

# ── Checkerboard using XOR of coordinates ─────────────────────────────────────
checker_size = 30
checker = ((X // checker_size + Y // checker_size) % 2 * 255).astype(np.uint8)
# checker[i,j] = 255 if (row//30 + col//30) is odd, else 0

# ── Angle map (for directional operations) ────────────────────────────────────
angle_map = np.arctan2(Y - cy, X - cx)   # range: [-π, π]
angle_deg = np.degrees(angle_map) % 360  # normalize to [0, 360]
```

### 2.7.2 Practical: Building Synthetic Test Images

Having synthetic test images is invaluable for testing algorithms without depending on real photos.

```python
import numpy as np
import cv2

def make_test_image(h=512, w=512):
    """
    Create a synthetic test image with known structure:
    useful for verifying algorithms.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Background: smooth gradient
    Y, X = np.mgrid[0:h, 0:w]
    bg = (X / w * 128).astype(np.uint8)   # left=0, right=128 in all channels
    img[:, :, 0] = bg        # Blue gradient
    img[:, :, 1] = (Y / h * 128).astype(np.uint8)  # Green gradient (vertical)

    # Draw colored rectangles
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)    # Blue filled
    cv2.rectangle(img, (250, 50), (400, 200), (0, 255, 0), -1)   # Green filled
    cv2.rectangle(img, (450, 50), (w-50, 200), (0, 0, 255), -1)  # Red filled

    # Circles
    cv2.circle(img, (w//4, h//2), 80, (255, 255, 0), -1)         # Cyan circle
    cv2.circle(img, (w//2, h//2), 80, (255, 0, 255), -1)         # Magenta
    cv2.circle(img, (3*w//4, h//2), 80, (0, 255, 255), -1)       # Yellow

    # Add text
    cv2.putText(img, "TEST IMAGE", (w//4, h-50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # Add Gaussian noise
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


test = make_test_image()
print(f"Test image shape: {test.shape}")
```

---

## 2.8 Performance: Vectorization vs Loops

This section makes the performance gap between Python loops and NumPy vectorization concrete and measurable.

### 2.8.1 The Cost of Pixel Loops

```python
import numpy as np
import cv2
import time

img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

# ── Method 1: Python loops — DO NOT use in production ─────────────────────────
def brightness_loop(image, delta):
    """Increase brightness using Python loops. Intentionally slow."""
    out = image.copy()
    h, w = out.shape[:2]
    for y in range(h):
        for x in range(w):
            for c in range(3):
                val = int(out[y, x, c]) + delta
                out[y, x, c] = min(255, max(0, val))
    return out

# ── Method 2: NumPy vectorized — preferred ────────────────────────────────────
def brightness_vectorized(image, delta):
    """Increase brightness using vectorized NumPy."""
    return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)

# ── Method 3: cv2.add — saturating arithmetic, fastest ───────────────────────
def brightness_cv2(image, delta):
    """Increase brightness using OpenCV's saturating add."""
    offset = np.full_like(image, delta, dtype=np.uint8)
    return cv2.add(image, offset)

# ── Benchmark ─────────────────────────────────────────────────────────────────
print("Benchmarking on 512×512 image...")

# Loop (only measure once — it's very slow)
t0 = time.perf_counter()
result_loop = brightness_loop(img, 30)
t1 = time.perf_counter()
print(f"Python loop:     {t1-t0:.3f}s")

# NumPy (average of 100 runs)
times = []
for _ in range(100):
    t0 = time.perf_counter()
    result_np = brightness_vectorized(img, 30)
    times.append(time.perf_counter() - t0)
print(f"NumPy vectorized: {np.mean(times)*1000:.2f}ms (avg of 100 runs)")

# cv2 (average of 100 runs)
times = []
for _ in range(100):
    t0 = time.perf_counter()
    result_cv2 = brightness_cv2(img, 30)
    times.append(time.perf_counter() - t0)
print(f"cv2.add:          {np.mean(times)*1000:.2f}ms (avg of 100 runs)")

# ── Verify all three give the same result ─────────────────────────────────────
print(f"\nResults match: {np.array_equal(result_loop, result_np)}")
```

**Typical output on a modern machine:**
```
Python loop:      12.847s
NumPy vectorized:  0.81ms  (~15,000× faster)
cv2.add:           0.34ms  (~37,000× faster)
```

The lesson: a Python loop over pixels is **never** acceptable for real-time or even batch image processing. Every operation in this book will be vectorized.

### 2.8.2 Profiling Your Image Pipeline

```python
import numpy as np
import cv2
import cProfile
import pstats
import io

def image_pipeline(img):
    """Sample pipeline to profile."""
    # Step 1: Convert to float
    f = img.astype(np.float32) / 255.0

    # Step 2: Normalize per channel
    means = f.mean(axis=(0, 1))
    stds  = f.std(axis=(0, 1)) + 1e-8
    f_norm = (f - means) / stds

    # Step 3: Compute edge energy (gradient magnitude)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)

    # Step 4: Threshold
    threshold = magnitude.mean() + magnitude.std()
    edges = (magnitude > threshold).astype(np.uint8) * 255

    return edges

img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

# Profile
pr = cProfile.Profile()
pr.enable()
for _ in range(10):
    image_pipeline(img)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(10)  # top 10 functions
print(s.getvalue())
```

---

## 2.9 Working with Pillow (PIL)

OpenCV is not the only way to load images in Python. Pillow (PIL) is the other major image I/O library. Understanding the conversion between them matters because:

1. Some datasets/tools provide Pillow images
2. Torchvision (PyTorch's image library) uses Pillow by default
3. Some image formats are better handled by Pillow (e.g., animated GIFs before OpenCV 4.11, TIFF metadata)

### 2.9.1 Pillow Basics

```python
from PIL import Image
import numpy as np
import cv2

# ── Open with Pillow ──────────────────────────────────────────────────────────
pil_img = Image.open("sample.jpg")
print(f"Pillow mode  : {pil_img.mode}")    # 'RGB' — note: NOT BGR!
print(f"Pillow size  : {pil_img.size}")    # (width, height) — note: reversed vs OpenCV!

# ── Pillow → NumPy ────────────────────────────────────────────────────────────
arr_rgb = np.array(pil_img)       # shape: (H, W, 3), dtype=uint8, RGB order
print(f"NumPy from Pillow: {arr_rgb.shape}, {arr_rgb.dtype}")  # (H, W, 3), uint8

# ── Pillow → OpenCV (RGB → BGR) ───────────────────────────────────────────────
arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

# ── OpenCV → Pillow ───────────────────────────────────────────────────────────
cv_img = cv2.imread("sample.jpg")
if cv_img is None:
    cv_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Must convert BGR → RGB before creating Pillow image
cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
pil_from_cv = Image.fromarray(cv_rgb)
print(f"Pillow from OpenCV: {pil_from_cv.mode}, {pil_from_cv.size}")

# ── Grayscale ─────────────────────────────────────────────────────────────────
pil_gray = pil_img.convert("L")         # Pillow grayscale
arr_gray = np.array(pil_gray)           # shape: (H, W), dtype=uint8
cv_gray  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)   # same result via OpenCV
```

### 2.9.2 The Conversion Matrix

```
File on disk (JPEG/PNG)
      │
      ▼
cv2.imread()      →  np.ndarray (H, W, 3) dtype=uint8, BGR order
      │
      │  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      ▼
np.ndarray (H, W, 3) dtype=uint8, RGB order
      │
      │  Image.fromarray(arr_rgb)
      ▼
PIL.Image (mode='RGB')
      │
      │  pil_img.convert("L")
      ▼
PIL.Image (mode='L') — grayscale
      │
      │  np.array(pil_gray)
      ▼
np.ndarray (H, W) dtype=uint8
```

**Key differences between OpenCV and Pillow:**

| Property | OpenCV | Pillow |
|----------|--------|--------|
| Channel order | BGR | RGB |
| Image type | `np.ndarray` | `PIL.Image` |
| Size attribute | `img.shape` = `(H, W, C)` | `img.size` = `(W, H)` |
| Default dtype | `uint8` | `uint8` |
| Read function | `cv2.imread()` | `Image.open()` |

---

## 2.10 Building the Book's Utility Library

Let us consolidate everything from this chapter into a reusable utility module that we will import throughout the rest of the book. Create a file called `cvbook/utils.py`:

```python
"""
cvbook/utils.py
─────────────────────────────────────────────────────────────────────────────
Reusable image utility functions for the OpenCV Python Book.
Used in every chapter — keep this file in your PYTHONPATH or project root.
─────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ── I/O ──────────────────────────────────────────────────────────────────────

def imload(path, color_space="bgr"):
    """
    Load an image from disk with validation.

    Parameters
    ----------
    path : str or Path
        Path to the image file.
    color_space : str
        Output color space: 'bgr' (default), 'rgb', 'gray'.

    Returns
    -------
    np.ndarray
        Loaded image.

    Raises
    ------
    FileNotFoundError
        If the file does not exist on disk.
    ValueError
        If OpenCV fails to decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    flag = cv2.IMREAD_GRAYSCALE if color_space == "gray" else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)

    if img is None:
        raise ValueError(f"OpenCV could not decode image: {path}")

    if color_space == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def imsave(path, img, quality=95):
    """
    Save an image to disk with validation.

    Parameters
    ----------
    path : str or Path
    img : np.ndarray
    quality : int
        JPEG quality (0–100). Ignored for lossless formats.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success = cv2.imwrite(str(path), img, params)
    if not success:
        raise IOError(f"Failed to save image to {path}")
    return path


# ── Display ───────────────────────────────────────────────────────────────────

def imshow(img, title="", figsize=None, cmap=None, axis="off"):
    """
    Display an image in a Jupyter notebook using matplotlib.

    Handles BGR→RGB conversion automatically for 3-channel images.
    """
    if figsize is None:
        h, w = img.shape[:2]
        figsize = (min(w / 80, 14), min(h / 80, 10))

    fig, ax = plt.subplots(figsize=figsize)

    if img.ndim == 3 and img.shape[2] == 3:
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(display)
    elif img.ndim == 2:
        ax.imshow(img, cmap=cmap or "gray")
    else:
        ax.imshow(img)

    ax.set_title(title)
    ax.axis(axis)
    plt.tight_layout()
    plt.show()
    return fig, ax


def imshow_grid(images, titles=None, nrow=4, figsize=None,
                bgr_to_rgb=True, cmap="gray"):
    """
    Display a list of images as a grid.

    Parameters
    ----------
    images : list of np.ndarray
    titles : list of str, optional
    nrow : int
        Number of images per row.
    bgr_to_rgb : bool
        Convert BGR images before display.
    """
    n = len(images)
    ncol = (n + nrow - 1) // nrow

    if figsize is None:
        figsize = (nrow * 3.5, ncol * 3.5)

    fig, axes = plt.subplots(ncol, nrow, figsize=figsize)
    if ncol == 1:
        axes = [axes] if nrow == 1 else [axes]
    axes = np.array(axes).flatten()

    for i, img in enumerate(images):
        ax = axes[i]
        if img.ndim == 3 and bgr_to_rgb:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


# ── Type Conversion ───────────────────────────────────────────────────────────

def to_float32(img):
    """Convert uint8 image [0,255] to float32 [0.0, 1.0]."""
    return img.astype(np.float32) / np.float32(255.0)


def to_uint8(img):
    """Convert float32 [0.0, 1.0] to uint8 [0, 255]. Clips before casting."""
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def ensure_bgr(img):
    """Ensure image is 3-channel BGR. Converts grayscale if needed."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def ensure_gray(img):
    """Ensure image is single-channel grayscale."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ── Geometry ─────────────────────────────────────────────────────────────────

def resize_to_width(img, width):
    """Resize image to target width, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (width, new_h), interpolation=interp)


def resize_to_height(img, height):
    """Resize image to target height, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = height / h
    new_w = int(w * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, height), interpolation=interp)


def pad_to_square(img, pad_value=0):
    """Pad image with border to make it square."""
    h, w = img.shape[:2]
    side = max(h, w)
    top    = (side - h) // 2
    bottom = side - h - top
    left   = (side - w) // 2
    right  = side - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=pad_value)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def image_info(img, name="image"):
    """Print a concise summary of image properties."""
    if img is None:
        print(f"{name}: None")
        return
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    mb = img.nbytes / 1024**2
    print(f"{name:20s} | shape={img.shape} | dtype={img.dtype} | "
          f"range=[{img.min()}, {img.max()}] | {mb:.2f} MB")
```

---

## 2.11 Summary

This chapter established the complete NumPy foundation for image processing:

**Data Model:**
- Images are `ndarray` objects with shape `(H, W)` for grayscale and `(H, W, 3)` for color
- Key attributes: `shape`, `dtype`, `strides`, `nbytes`, `flags`
- Memory is stored row-major (C-contiguous); channels are interleaved (HWC)

**Data Types:**
- `uint8` is the display type (0–255); arithmetic wraps around — use float32 for intermediate math
- Normalize to `[0.0, 1.0]` float32 for most processing; `np.float32(255.0)` keeps dtype clean in NumPy 2.x
- Always `clip()` before casting back to `uint8`

**Indexing:**
- Basic indexing: `img[row, col]` — slow in loops
- Slice indexing: `img[y1:y2, x1:x2]` — returns views, use `.copy()` for independence
- Fancy indexing: arrays of indices — always returns copies
- Boolean indexing: mask arrays — the workhorse of selective modification

**Broadcasting:**
- Per-channel: scalar `(3,)` broadcasts with `(H, W, 3)` — no loop needed
- Per-row: reshape to `(H, 1, 1)` to broadcast with all columns and channels
- Per-column: reshape to `(1, W, 1)` — same principle

**Performance:**
- Python pixel loops: ~12 seconds for 512×512 — never acceptable
- NumPy vectorized: ~0.8ms — 15,000× faster
- `cv2` functions: ~0.3ms — fastest path

**Interoperability:**
- Pillow → NumPy: `np.array(pil_img)` gives RGB; convert to BGR for OpenCV
- OpenCV → Pillow: convert BGR→RGB, then `Image.fromarray(arr_rgb)`

---

## 2.12 Exercises

### Warm-up

**2.1** Given a loaded BGR image, write a one-liner (single line of code) for each of the following: (a) extract the top-left 200×300 crop, (b) flip it horizontally, (c) extract only the Red channel as a 2D array, (d) downsample by 4× using striding.

**2.2** Create a `300×300` `uint8` image where the value of each pixel at `(row, col)` equals `(row + col) % 256`. Do this with a single NumPy expression — no loops. Display the result.

**2.3** Convert a `uint8` image to `float32 [0,1]` and back to `uint8`. Print the maximum absolute difference between the original and round-tripped image. Explain why it might not be exactly 0.

### Core

**2.4** Implement a function `channel_swap(img)` that takes a BGR image and returns a new image where the Blue and Red channels are swapped — without using `cv2.cvtColor`. Do it using NumPy indexing only.

**2.5** Implement a soft circular vignette effect: create a float32 mask that is 1.0 at the image center and decreases smoothly to `sigma` (configurable) at the edges. Multiply it with the image. Use `np.mgrid` and no loops. Show the original and vignetted images side by side.

**2.6** Write a function `safe_blend(img1, img2, alpha_map)` where `alpha_map` is a 2D float32 array with values in `[0.0, 1.0]` specifying per-pixel blending weight. The function should handle both grayscale and color images. Test it with: (a) a constant `alpha=0.5`, (b) a horizontal gradient alpha map (left=img1, right=img2).

**2.7** Write a function `tile_images(img, ny, nx)` that tiles an image `ny` times vertically and `nx` times horizontally using `np.tile`. Then write an alternative using `np.concatenate`. Time both approaches on a 512×512 image with `ny=4, nx=4`. Compare results.

### Challenge

**2.8** Implement a `brightness_contrast(img, alpha, beta)` function that computes `out = clip(alpha * img + beta, 0, 255)`. Write four implementations: (a) Python loop, (b) NumPy vectorized, (c) using `cv2.convertScaleAbs`, (d) using `np.clip` on a float32 intermediate. Benchmark all four on a 1080p image. Plot a bar chart of execution times.

**2.9** Implement a function `patch_extractor(img, patch_size, stride)` that extracts all non-overlapping (or strided-overlapping) patches from an image and returns them as a 4D array of shape `(N_patches, patch_size, patch_size, C)`. Use `np.lib.stride_tricks.as_strided` for the implementation — **no loops**. Verify correctness by reconstructing the original image from patches.

**2.10** Build a function `compute_image_statistics(img)` that returns a dictionary containing: per-channel mean, std, min, max, median, 5th and 95th percentiles; overall entropy (use `np.histogram` with 256 bins, compute `−Σ p log2 p`); and the percentage of pixels that are saturated (value == 0 or value == 255). Test it on at least three different images and compare the statistics.

---

## Further Reading

- **NumPy documentation — Array Indexing:** https://numpy.org/doc/stable/reference/arrays.indexing.html
- **NumPy documentation — Broadcasting:** https://numpy.org/doc/stable/user/basics.broadcasting.html
- **NumPy 2.0 Migration Guide:** https://numpy.org/doc/stable/release/2.0.0-notes.html  
- **NumPy documentation — Memory Layout and Strides:** https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
- **`np.lib.stride_tricks` tutorial:** https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html
- **Pillow documentation:** https://pillow.readthedocs.io/

---

*Next up: **Chapter 3 — Color Spaces & Channels**, where we go deep into BGR, RGB, HSV, HLS, LAB, and YCrCb — understanding which color space to use for which problem, and why hue-based segmentation fails in BGR.*
