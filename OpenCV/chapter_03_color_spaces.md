# Chapter 3: Color Spaces & Channels

> *"The choice of color space is not cosmetic — it is the difference between an algorithm that works and one that doesn't."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain what a color space is and why multiple color spaces exist
2. Convert fluently between BGR, RGB, grayscale, HSV, HLS, LAB, XYZ, and YCrCb
3. Describe the geometry, channel meaning, and OpenCV value ranges for each color space
4. Explain the grayscale luminance formula and why it is not a simple average
5. Use HSV for robust color-based segmentation and handle the red-hue wraparound correctly
6. Explain perceptual uniformity and use LAB for color-difference measurements (Delta E)
7. Use YCrCb for skin-tone detection and illumination-robust segmentation
8. Choose the right color space for a given task using a principled decision framework
9. Build a complete multi-color object tracker using `cv2.inRange`
10. Understand OpenCV's scaling conventions for each color space

---

## 3.1 What Is a Color Space?

A **color space** is a mathematical model that maps observable colors to numbers. The same physical color can be described by many different coordinate systems — just as a point in the real world can be described by Cartesian (x, y, z) or spherical (r, θ, φ) coordinates. Each description is complete and interconvertible, but each makes different operations easy or hard.

### 3.1.1 Why Multiple Color Spaces Exist

Consider the problem of detecting a red traffic light.

In **BGR space**, red pixels have high R, low G, and low B. But at dusk, an orange streetlight also creates high R values. At noon in direct sunlight, a red object may still have a fairly high B channel due to scattered light. The three channels are coupled — a change in illumination shifts all three.

In **HSV space**, red has a specific hue angle (~0° or ~360°), regardless of whether the scene is bright or dark, indoors or outdoors. The illumination is captured entirely in the V (Value) channel and the color in H (Hue). You can lock onto red with a narrow H range and ignore V.

In **LAB space**, the L channel captures all lightness information. The a and b channels capture only color (chrominance), completely decoupled from brightness. A red patch in a dim room and a red patch in bright sunlight will have similar a and b values but different L values.

Each color space encodes the same underlying information differently — and your choice of color space directly determines how hard or easy each subsequent algorithm will be.

### 3.1.2 The `cv2.cvtColor` Function

All color space conversions in OpenCV go through a single function:

```python
dst = cv2.cvtColor(src, code)
```

Where `code` is one of hundreds of constants of the form `cv2.COLOR_<FROM>2<TO>`. Some key principles:

- The source image must have the correct number of channels for the input color space
- Most conversions expect `uint8` or `float32` input
- The output dtype and value range depend on the target color space
- Conversions through intermediate spaces (e.g., BGR→XYZ→LAB) are handled internally

```python
import cv2
import numpy as np

# List all available conversion codes
all_codes = [x for x in dir(cv2) if x.startswith("COLOR_")]
print(f"Total conversion codes: {len(all_codes)}")   # ~300+

# Print BGR-origin conversions
bgr_codes = [x for x in all_codes if x.startswith("COLOR_BGR")]
print(f"BGR conversions available: {len(bgr_codes)}")
for code in sorted(bgr_codes)[:20]:
    print(f"  cv2.{code} = {getattr(cv2, code)}")
```

---

## 3.2 BGR and RGB: The Foundation

### 3.2.1 Why BGR and Not RGB

As established in Chapter 1, OpenCV uses **BGR** (Blue, Green, Red) channel ordering. This is a historical artifact from Windows GDI (Graphics Device Interface) APIs, which stored 24-bit color as BGR in memory. OpenCV adopted this convention in its early Intel years, and it has persisted ever since.

Every function in OpenCV assumes BGR input. This includes `cv2.imread`, `cv2.VideoCapture`, and all color space converters. The `cv2.COLOR_BGR2RGB` code merely reverses the channel order — no mathematical transformation occurs.

```python
import cv2
import numpy as np

# Create a pure red pixel — RGB: (255, 0, 0) — to demonstrate the ordering
red_bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)    # BGR: B=0, G=0, R=255
red_rgb = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2RGB)
print(f"BGR encoding of red: {red_bgr[0,0]}")    # [0 0 255]
print(f"RGB encoding of red: {red_rgb[0,0]}")    # [255 0 0]

# The two most common conversions you will use daily
img_bgr = cv2.imread("sample.jpg")
if img_bgr is None:
    img_bgr = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # for matplotlib / PyTorch
img_bgr_back = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # back to OpenCV

# Verify round-trip is lossless
assert np.array_equal(img_bgr, img_bgr_back), "Round-trip should be lossless"
```

### 3.2.2 Visualizing BGR Channels

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_channels_bgr(img_bgr, figsize=(14, 4)):
    """
    Display a BGR image alongside its three individual channels,
    each shown in its natural color tint.
    """
    b, g, r = cv2.split(img_bgr)

    # Create colorized single-channel views for intuitive display
    # Blue channel: embed in blue slot of a 3-channel image
    blue_vis  = np.zeros_like(img_bgr); blue_vis[:, :, 0] = b
    green_vis = np.zeros_like(img_bgr); green_vis[:, :, 1] = g
    red_vis   = np.zeros_like(img_bgr); red_vis[:, :, 2]  = r

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    images = [img_bgr, blue_vis, green_vis, red_vis]
    titles = ["Original (BGR)", "Blue Channel", "Green Channel", "Red Channel"]

    for ax, im, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("BGR Channel Decomposition", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


img = cv2.imread("sample.jpg")
if img is None:
    # Create a synthetic scene with distinct color regions
    img = np.zeros((200, 600, 3), dtype=np.uint8)
    img[:, :200]  = [200, 50, 20]    # blue-dominant region
    img[:, 200:400] = [30, 180, 30]  # green-dominant region
    img[:, 400:]  = [20, 40, 220]    # red-dominant region

visualize_channels_bgr(img)
```

---

## 3.3 Grayscale: Luminance, Not Average

Converting to grayscale is the most common single operation in image processing. The intuitive approach — averaging the three channels — is wrong for an important reason: the human eye does not treat all wavelengths equally.

### 3.3.1 The Luminance Formula

The perceptual luminance formula is specified by ITU-R BT.601 (for standard definition video) and BT.709 (for HD video):

```
BT.601:  Y = 0.299·R + 0.587·G + 0.114·B
BT.709:  Y = 0.2126·R + 0.7152·G + 0.0722·B
```

The human eye is most sensitive to green (~58–71%), moderately sensitive to red (~21–30%), and least sensitive to blue (~7–11%). This reflects the relative density of the three types of cone cells in the human fovea.

OpenCV's `cv2.COLOR_BGR2GRAY` uses the BT.601 formula by default.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg")
if img is None:
    # Create a synthetic image: pure red, green, blue patches of equal "brightness"
    img = np.zeros((200, 600, 3), dtype=np.uint8)
    img[:, :200]  = [0, 0, 200]     # Red (BGR)
    img[:, 200:400] = [0, 200, 0]   # Green (BGR)
    img[:, 400:]  = [200, 0, 0]     # Blue (BGR)

# ── Method 1: cv2 conversion (BT.601, correct) ───────────────────────────────
gray_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── Method 2: Naive channel average (incorrect) ──────────────────────────────
gray_avg = img.mean(axis=2).astype(np.uint8)

# ── Method 3: Manual BT.601 formula ─────────────────────────────────────────
# Note: in BGR order, R=index 2, G=index 1, B=index 0
b = img[:, :, 0].astype(np.float32)
g = img[:, :, 1].astype(np.float32)
r = img[:, :, 2].astype(np.float32)
gray_manual = (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255).astype(np.uint8)

# ── Method 4: BT.709 (used for HD content) ──────────────────────────────────
gray_bt709 = (0.2126 * r + 0.7152 * g + 0.0722 * b).clip(0, 255).astype(np.uint8)

print("Grayscale values for pure R=200, G=200, B=200 patches (equal input):")
h, w = img.shape[:2]
print(f"  cv2 (BT.601) : R-patch={gray_cv2[h//2, 50]}, "
      f"G-patch={gray_cv2[h//2, 250]}, B-patch={gray_cv2[h//2, 500]}")
print(f"  naive avg    : R-patch={gray_avg[h//2, 50]}, "
      f"G-patch={gray_avg[h//2, 250]}, B-patch={gray_avg[h//2, 500]}")
# Expected: Green patch is brightest, blue is darkest in cv2 result
# Expected: All patches equal in naive avg (all = 200/3*... wait — only one channel is set)
# With pure red patch: R=200, G=0, B=0 → cv2: 0.299*200 = 59.8 ≈ 60
# With pure green: 0.587*200 = 117.4 ≈ 117
# With pure blue: 0.114*200 = 22.8 ≈ 23
# naive avg: all 200/3 ≈ 66

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, gray, title in zip(axes,
                            [gray_cv2, gray_avg, gray_manual, gray_bt709],
                            ["cv2 (BT.601)", "Naive Average",
                             "Manual BT.601", "BT.709"]):
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

### 3.3.2 When to Load Directly as Grayscale

```python
import cv2
import numpy as np

# Option 1: Load directly as grayscale
gray = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
print(f"Direct load gray shape: {gray.shape}")  # (H, W) — 2D

# Option 2: Load BGR, then convert — identical result
bgr = cv2.imread("sample.jpg")
if bgr is not None:
    gray2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print(f"Converted gray shape: {gray2.shape}")  # (H, W)
    # These are identical
    if gray is not None:
        print(f"Results match: {np.array_equal(gray, gray2)}")  # True

# Option 3: Keep as single-channel 3D array (sometimes needed for consistency)
if gray is not None:
    gray3d = gray[:, :, np.newaxis]   # shape: (H, W, 1)
    # Or: gray3d = np.expand_dims(gray, axis=-1)

# Converting grayscale back to BGR (for drawing colored overlays on gray image)
if gray is not None:
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    print(f"Gray→BGR shape: {gray_bgr.shape}")  # (H, W, 3)
    # All three channels are equal: gray_bgr[:,:,0] == gray_bgr[:,:,1] == gray_bgr[:,:,2]
```

---

## 3.4 HSV: Hue, Saturation, Value

HSV is the workhorse color space for practical color-based segmentation. Its power comes from separating **what color** (Hue) from **how much of it** (Saturation) from **how bright** (Value). This separation makes it far easier to specify a color range that is robust to lighting changes.

### 3.4.1 The HSV Geometry

HSV is a cylindrical coordinate system:

```
         White (V=1)
            │
            │
            │
 ─ ─ ─ ─ ─ ●─ ─ ─ ─ ─   (S=0 axis: grays)
          / │ \
         /  │  \
        /   │   \
 ──────●────┼────●──────   (H is the angle in this plane)
        \   │   /           S is the distance from center
         \  │  /            V is the height
          \ │ /
            ●
         Black (V=0)
```

- **H (Hue):** The color angle, 0°–360°. Red is at 0° (and 360°), then yellow at 60°, green at 120°, cyan at 180°, blue at 240°, magenta at 300°.
- **S (Saturation):** How "pure" the color is. S=0 is gray (no color), S=1 is fully saturated (pure color).
- **V (Value):** Brightness. V=0 is black (regardless of H and S), V=1 is the full brightness.

### 3.4.2 OpenCV's HSV Ranges — The Critical Gotcha

OpenCV stores HSV values as `uint8` (0–255), but the natural HSV ranges are `H:[0°, 360°]`, `S:[0%, 100%]`, `V:[0%, 100%]`. OpenCV scales them as follows:

| Channel | Natural Range | OpenCV uint8 Range | Scale Factor |
|---------|--------------|---------------------|--------------|
| H (Hue) | 0° – 360° | **0 – 179** | ÷ 2 (to fit in uint8) |
| S (Saturation) | 0.0 – 1.0 | 0 – 255 | × 255 |
| V (Value) | 0.0 – 1.0 | 0 – 255 | × 255 |

The H channel is halved (0–179, not 0–255) because 360 does not fit in a `uint8`. This is a source of frequent bugs when people try to specify hue ranges from outside references. When you read a color picker that says "Hue = 120°" (green), you must use H = 60 in OpenCV.

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Convert BGR → HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

print(f"H range: [{h.min()}, {h.max()}]")    # [0, 179]
print(f"S range: [{s.min()}, {s.max()}]")    # [0, 255]
print(f"V range: [{v.min()}, {v.max()}]")    # [0, 255]

# ── How to look up the HSV value of a specific BGR color ────────────────────
def bgr_to_hsv_value(bgr_color):
    """Convert a single BGR color to its OpenCV HSV representation."""
    pixel = np.uint8([[bgr_color]])          # shape: (1, 1, 3)
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    return hsv_pixel[0, 0]

# Pure colors in BGR → their HSV
colors = {
    "Red"    : [0, 0, 255],
    "Green"  : [0, 255, 0],
    "Blue"   : [255, 0, 0],
    "Yellow" : [0, 255, 255],
    "Cyan"   : [255, 255, 0],
    "Magenta": [255, 0, 255],
    "White"  : [255, 255, 255],
    "Black"  : [0, 0, 0],
    "Gray"   : [128, 128, 128],
}

print("\nBGR → HSV lookup table:")
print(f"{'Color':<10} {'BGR':<20} {'HSV (OpenCV uint8)':<25} {'H° (natural)'}")
print("─" * 75)
for name, bgr in colors.items():
    hsv_val = bgr_to_hsv_value(bgr)
    h_deg = hsv_val[0] * 2  # convert back to degrees
    print(f"{name:<10} BGR={bgr!s:<17} HSV={list(hsv_val)!s:<23} H={h_deg}°")
```

**Output:**
```
Color      BGR                  HSV (OpenCV uint8)        H° (natural)
───────────────────────────────────────────────────────────────────────
Red        BGR=[0, 0, 255]      HSV=[0, 255, 255]         H=0°
Green      BGR=[0, 255, 0]      HSV=[60, 255, 255]        H=120°
Blue       BGR=[255, 0, 0]      HSV=[120, 255, 255]       H=240°
Yellow     BGR=[0, 255, 255]    HSV=[30, 255, 255]        H=60°
Cyan       BGR=[255, 255, 0]    HSV=[90, 255, 255]        H=180°
Magenta    BGR=[255, 0, 255]    HSV=[150, 255, 255]       H=300°
White      BGR=[255, 255, 255]  HSV=[0, 0, 255]           H=0°
Black      BGR=[0, 0, 0]        HSV=[0, 0, 0]             H=0°
Gray       BGR=[128, 128, 128]  HSV=[0, 0, 128]           H=0°
```

### 3.4.3 Color Segmentation with cv2.inRange

`cv2.inRange` is the primary tool for HSV-based segmentation. It creates a binary mask where pixels within `[lower, upper]` bounds are 255 and everything else is 0:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg")
if img is None:
    # Synthetic image: colored objects on gray background
    img = np.full((300, 500, 3), 100, dtype=np.uint8)  # gray
    cv2.circle(img, (100, 150), 80, [0, 50, 220], -1)   # red circle (BGR)
    cv2.circle(img, (300, 150), 80, [30, 200, 30], -1)  # green circle
    cv2.circle(img, (420, 150), 60, [220, 50, 0], -1)   # blue circle

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ── Isolate green objects ─────────────────────────────────────────────────────
# Green in OpenCV HSV: H ≈ 35–85, S > 50, V > 50
lower_green = np.array([35, 50, 50],  dtype=np.uint8)
upper_green = np.array([85, 255, 255], dtype=np.uint8)
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# ── Isolate blue objects ──────────────────────────────────────────────────────
lower_blue = np.array([100, 50, 50],  dtype=np.uint8)
upper_blue = np.array([140, 255, 255], dtype=np.uint8)
mask_blue  = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply masks to original image
result_green = cv2.bitwise_and(img, img, mask=mask_green)
result_blue  = cv2.bitwise_and(img, img, mask=mask_blue)

# Visualize
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, im, title in zip(axes,
    [img, hsv, mask_green, result_green, result_blue],
    ["Original (BGR)", "HSV", "Green mask", "Green isolated", "Blue isolated"]):
    if im.ndim == 2:
        ax.imshow(im, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(title); ax.axis("off")
plt.tight_layout()
plt.show()
```

### 3.4.4 The Red Wraparound Problem

Red occupies both ends of the hue circle — near H=0° and H=360° (H=0 and H=179 in OpenCV). A single `cv2.inRange` call cannot capture both ends simultaneously. You need two masks combined with a bitwise OR:

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, [0, 0, 220], -1)   # red circle

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ── WRONG: single range misses some reds ─────────────────────────────────────
# lower_red_wrong = np.array([170, 50, 50])
# upper_red_wrong = np.array([10, 255, 255])  ← invalid! lower > upper

# ── CORRECT: two ranges, then combine ────────────────────────────────────────
# Red: H near 0 (lower end of hue circle)
lower_red1 = np.array([0,   50,  50],  dtype=np.uint8)
upper_red1 = np.array([10,  255, 255], dtype=np.uint8)

# Red: H near 179 (upper end of hue circle, wraps around)
lower_red2 = np.array([170, 50,  50],  dtype=np.uint8)
upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Combine with bitwise OR
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

result = cv2.bitwise_and(img, img, mask=mask_red)
print(f"Red pixels found: {mask_red.sum() // 255}")


def segment_color(img_bgr, color_name):
    """
    Segment a named color from a BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
    color_name : str
        One of: 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple'

    Returns
    -------
    mask : np.ndarray (uint8, 0 or 255)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # HSV ranges in OpenCV uint8 units
    ranges = {
        "red"    : [(  0,  50,  50), ( 10, 255, 255),
                    (170,  50,  50), (179, 255, 255)],  # two-range wraparound
        "orange" : [( 11,  50,  50), ( 25, 255, 255)],
        "yellow" : [( 26,  50,  50), ( 34, 255, 255)],
        "green"  : [( 35,  50,  50), ( 85, 255, 255)],
        "cyan"   : [( 86,  50,  50), ( 99, 255, 255)],
        "blue"   : [(100,  50,  50), (140, 255, 255)],
        "purple" : [(141,  50,  50), (169, 255, 255)],
    }

    if color_name not in ranges:
        raise ValueError(f"Unknown color '{color_name}'. "
                         f"Choose from: {list(ranges)}")

    r = ranges[color_name]
    if len(r) == 2:
        # Single range
        mask = cv2.inRange(hsv,
                           np.array(r[0], dtype=np.uint8),
                           np.array(r[1], dtype=np.uint8))
    else:
        # Two-range (wraparound — currently only red)
        m1 = cv2.inRange(hsv,
                         np.array(r[0], dtype=np.uint8),
                         np.array(r[1], dtype=np.uint8))
        m2 = cv2.inRange(hsv,
                         np.array(r[2], dtype=np.uint8),
                         np.array(r[3], dtype=np.uint8))
        mask = cv2.bitwise_or(m1, m2)

    return mask
```

### 3.4.5 Interactive HSV Range Finder

In practice, the best way to determine HSV ranges is empirically with an interactive tool. Here is a trackbar-based HSV range finder:

```python
import cv2
import numpy as np


def hsv_range_finder(img_bgr):
    """
    Interactive HSV range finder using OpenCV trackbars.
    Adjust sliders to tune lower/upper bounds for segmentation.
    Press 'q' or ESC to exit and print the final values.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    window_name = "HSV Range Finder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create trackbars: (name, window, default, max, callback)
    cv2.createTrackbar("H_lower", window_name, 0,   179, lambda x: None)
    cv2.createTrackbar("H_upper", window_name, 179, 179, lambda x: None)
    cv2.createTrackbar("S_lower", window_name, 50,  255, lambda x: None)
    cv2.createTrackbar("S_upper", window_name, 255, 255, lambda x: None)
    cv2.createTrackbar("V_lower", window_name, 50,  255, lambda x: None)
    cv2.createTrackbar("V_upper", window_name, 255, 255, lambda x: None)

    while True:
        hl = cv2.getTrackbarPos("H_lower", window_name)
        hu = cv2.getTrackbarPos("H_upper", window_name)
        sl = cv2.getTrackbarPos("S_lower", window_name)
        su = cv2.getTrackbarPos("S_upper", window_name)
        vl = cv2.getTrackbarPos("V_lower", window_name)
        vu = cv2.getTrackbarPos("V_upper", window_name)

        lower = np.array([hl, sl, vl], dtype=np.uint8)
        upper = np.array([hu, su, vu], dtype=np.uint8)

        mask   = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

        display = np.hstack([img_bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
        cv2.imshow(window_name, display)

        key = cv2.waitKey(10) & 0xFF
        if key in [ord("q"), 27]:  # 'q' or ESC
            break

    cv2.destroyAllWindows()
    print(f"\nFinal HSV range:")
    print(f"  lower = np.array([{hl}, {sl}, {vl}])")
    print(f"  upper = np.array([{hu}, {su}, {vu}])")
    return lower, upper


# Usage: hsv_range_finder(img)   # requires desktop environment
```

---

## 3.5 HLS: Hue, Lightness, Saturation

HLS is closely related to HSV. The key difference is that the **L (Lightness)** channel defines maximum color at L=0.5 (not L=1.0 as in HSV's Value channel). In HSV, increasing V toward 1 moves toward pure saturated color; in HLS, maximum color purity is at the midpoint L=0.5, with white at L=1.0 and black at L=0.0.

```
HSV:   Black (V=0) → Pure Color (V=1, S=1) → Can't reach White
HLS:   Black (L=0) → Pure Color (L=0.5, S=1) → White (L=1)
```

### 3.5.1 OpenCV HLS Ranges

OpenCV stores HLS as:
- **H:** 0–179 (same halving as HSV)
- **L:** 0–255
- **S:** 0–255

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
h, l, s = cv2.split(hls)

print(f"HLS ranges: H=[{h.min()},{h.max()}], L=[{l.min()},{l.max()}], S=[{s.min()},{s.max()}]")
```

### 3.5.2 HLS vs HSV: When to Use Which

HLS has one significant advantage over HSV: it handles the **white and bright color** distinction better. In HSV, white (H=0, S=0, V=255) and bright yellow (H=30, S=255, V=255) differ only in S — a small change makes white look yellow in HSV. In HLS, white has L=255 and S=0, which is geometrically far from bright yellow.

A common practical use of HLS over HSV is **lane detection** in autonomous vehicles:

```python
import cv2
import numpy as np

def detect_white_lane_markings(img_bgr):
    """
    Detect white lane markings using HLS color space.
    White is better detected in HLS than HSV.
    """
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)

    # White: any hue (H is irrelevant), high lightness, low saturation
    # L > 200, S < 50 (nearly desaturated, bright)
    lower_white = np.array([0,   200,  0], dtype=np.uint8)
    upper_white = np.array([179, 255, 50], dtype=np.uint8)
    mask_white  = cv2.inRange(hls, lower_white, upper_white)

    # Yellow lane markings (can also add in HLS space)
    # Yellow in HLS: H ≈ 15–30, L ≈ 100–200, S > 100
    lower_yellow = np.array([15, 80,  100], dtype=np.uint8)
    upper_yellow = np.array([35, 200, 255], dtype=np.uint8)
    mask_yellow  = cv2.inRange(hls, lower_yellow, upper_yellow)

    mask_lanes = cv2.bitwise_or(mask_white, mask_yellow)
    return mask_lanes, mask_white, mask_yellow


img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (200, 400, 3), dtype=np.uint8)

mask_all, mask_white, mask_yellow = detect_white_lane_markings(img)
```

---

## 3.6 LAB: Lightness, a*, b*

The **CIELAB (L\*a\*b\*)** color space was defined by the International Commission on Illumination (CIE) in 1976. It was designed to be **perceptually uniform** — equal numerical distances in LAB space should correspond to equal perceived color differences by a human observer.

### 3.6.1 The Three LAB Channels

- **L\* (Lightness):** Perceptual brightness. L=0 is pure black, L=100 is pure white.
- **a\* (Green–Red axis):** Negative values indicate green; positive values indicate red/magenta.
- **b\* (Blue–Yellow axis):** Negative values indicate blue; positive values indicate yellow.

```
       +a* (Red/Magenta)
          │
          │
 -b*      │       +b*
(Blue) ───┼─────── (Yellow)
          │
          │
       -a* (Green)
```

The geometry immediately suggests two things:
1. Skin tones cluster in positive a* (reddish), moderate positive b* (slightly warm), medium L*
2. Natural foliage clusters in negative a* (green), slightly positive b* (yellowish), moderate L*

### 3.6.2 OpenCV's LAB Scaling

Because L, a, b are float values with different natural ranges, OpenCV scales them for `uint8` storage:

| Channel | Natural Range | OpenCV uint8 | Formula |
|---------|--------------|--------------|---------|
| L* | 0 – 100 | 0 – 255 | L_cv = L_natural * 255/100 |
| a* | -128 – +127 | 0 – 255 | a_cv = a_natural + 128 |
| b* | -128 – +127 | 0 – 255 | b_cv = b_natural + 128 |

So in OpenCV uint8 LAB: **neutral gray has L=~128, a=128, b=128** (not a=0, b=0 as in the natural space). Red objects will have a > 128 (positive a*). Blue objects will have b < 128 (negative b*).

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.zeros((200, 600, 3), dtype=np.uint8)
    img[:, :200]  = [0, 0, 200]     # red region (BGR)
    img[:, 200:400] = [0, 200, 0]   # green region
    img[:, 400:]  = [200, 0, 0]     # blue region

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(lab)

print("LAB values at color patches (OpenCV uint8 scaling):")
h = lab.shape[0] // 2
print(f"  Red   region: L={L[h,50]}, a={a[h,50]}, b={b[h,50]}")
# Red:   high a (>128), moderate b
print(f"  Green region: L={L[h,250]}, a={a[h,250]}, b={b[h,250]}")
# Green: low a (<128), low b
print(f"  Blue  region: L={L[h,500]}, a={a[h,500]}, b={b[h,500]}")
# Blue:  low b (<128), moderate a

# Convert to natural float range for intuition
lab_f32 = cv2.cvtColor(img.astype(np.float32) / 255.0,
                        cv2.COLOR_BGR2Lab)
# Now L in [0,100], a in [-127,127], b in [-127,127]
Lf, af, bf = cv2.split(lab_f32)
print("\nNatural float LAB ranges (L:[0,100], a:[-127,127], b:[-127,127]):")
print(f"  Red   region: L={Lf[h,50]:.1f}, a={af[h,50]:.1f}, b={bf[h,50]:.1f}")
print(f"  Green region: L={Lf[h,250]:.1f}, a={af[h,250]:.1f}, b={bf[h,250]:.1f}")
print(f"  Blue  region: L={Lf[h,500]:.1f}, a={af[h,500]:.1f}, b={bf[h,500]:.1f}")
```

### 3.6.3 Color Difference: Delta E

The primary reason to use LAB for color comparison is that Euclidean distance in LAB space approximates perceived color difference. This metric is called **ΔE (Delta E)**:

```
ΔE = sqrt((L1*-L2*)² + (a1*-a2*)² + (b1*-b2*)²)
```

The CIE76 ΔE formula above is the simplest form. Its interpretation:

| ΔE Value | Perception |
|----------|------------|
| < 1 | Not perceptible to most observers |
| 1–2 | Perceptible only with close inspection |
| 2–10 | Perceptible at a glance |
| 11–49 | Colors are more similar than opposite |
| 100 | Maximum difference (black vs white) |

```python
import cv2
import numpy as np


def delta_e_cie76(color1_bgr, color2_bgr):
    """
    Compute perceptual color difference (CIE76 Delta E)
    between two BGR colors.

    Parameters
    ----------
    color1_bgr, color2_bgr : array-like of shape (3,)
        BGR pixel values as uint8.

    Returns
    -------
    float
        Delta E value. <2 is barely perceptible, >10 is clearly different.
    """
    # Convert individual pixels to LAB float32
    # cv2.COLOR_BGR2Lab on float32 gives natural range:
    # L: [0,100], a: [-127,127], b: [-127,127]
    c1 = np.array([[color1_bgr]], dtype=np.float32) / 255.0
    c2 = np.array([[color2_bgr]], dtype=np.float32) / 255.0
    lab1 = cv2.cvtColor(c1, cv2.COLOR_BGR2Lab)[0, 0]
    lab2 = cv2.cvtColor(c2, cv2.COLOR_BGR2Lab)[0, 0]

    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return float(np.sqrt(dL**2 + da**2 + db**2))


def delta_e_image(img1_bgr, img2_bgr):
    """
    Compute per-pixel Delta E map between two same-size BGR images.
    Returns a float32 array of shape (H, W).
    Useful for image comparison, watermark detection, quality assessment.
    """
    assert img1_bgr.shape == img2_bgr.shape

    # Convert to float32 LAB
    lab1 = cv2.cvtColor(img1_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lab2 = cv2.cvtColor(img2_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

    diff = lab1.astype(np.float32) - lab2.astype(np.float32)
    de_map = np.sqrt(np.sum(diff**2, axis=2))  # shape: (H, W)
    return de_map


# ── Demonstration ─────────────────────────────────────────────────────────────
# Pairs of colors and their perceptual difference
test_pairs = [
    ([0, 0, 255],   [0, 0, 255],   "Same red vs same red"),
    ([0, 0, 255],   [0, 0, 200],   "Bright red vs dark red"),
    ([0, 0, 255],   [0, 30, 230],  "Red vs orange-red"),
    ([0, 0, 255],   [0, 255, 0],   "Red vs green"),
    ([0, 0, 255],   [255, 0, 0],   "Red vs blue"),
    ([0, 0, 0],     [255, 255, 255], "Black vs white"),
]

print("Color Difference (CIE76 ΔE):")
print("─" * 60)
for c1, c2, description in test_pairs:
    de = delta_e_cie76(c1, c2)
    perception = ("identical" if de < 1 else
                  "barely perceptible" if de < 2 else
                  "noticeable" if de < 10 else
                  "very different")
    print(f"  {description:<35} ΔE = {de:6.2f}  ({perception})")
```

### 3.6.4 LAB-Based Color Segmentation

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    # Add a skin-tone region
    cv2.rectangle(img, (50, 50), (180, 180), [100, 130, 190], -1)  # approx skin (BGR)
    # Add a red region
    cv2.circle(img, (300, 100), 60, [20, 30, 200], -1)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# ── Skin detection in LAB (classic Kovac approach) ────────────────────────────
# Skin: L=[20,220], a=[140,180], b=[100,170] (uint8 OpenCV LAB)
lower_skin_lab = np.array([20,  140, 100], dtype=np.uint8)
upper_skin_lab = np.array([220, 180, 170], dtype=np.uint8)
mask_skin_lab  = cv2.inRange(lab, lower_skin_lab, upper_skin_lab)

# ── Red object detection in LAB ──────────────────────────────────────────────
# Red: high a* (>160 in uint8 = > +32 natural), any L, b neutral to positive
lower_red_lab = np.array([50,  155, 120], dtype=np.uint8)
upper_red_lab = np.array([255, 255, 200], dtype=np.uint8)
mask_red_lab  = cv2.inRange(lab, lower_red_lab, upper_red_lab)

result_skin = cv2.bitwise_and(img, img, mask=mask_skin_lab)
result_red  = cv2.bitwise_and(img, img, mask=mask_red_lab)
```

---

## 3.7 XYZ: The Device-Independent Foundation

The CIE XYZ color space (defined in 1931) is the mathematical foundation from which LAB (and many other spaces) are derived. XYZ values represent colors in terms of the tristimulus responses of the CIE standard observer — a model of human cone cell sensitivities.

You will encounter XYZ directly less often than HSV or LAB, but it appears as an intermediate step in:
- BGR → XYZ → LAB (how OpenCV implements BGR2LAB internally)
- Color management and ICC profile conversions
- Illuminant corrections

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
X, Y, Z = cv2.split(xyz)

print("XYZ color space:")
print(f"  Shape: {xyz.shape}, Dtype: {xyz.dtype}")
print(f"  X range: [{X.min()}, {X.max()}]")
print(f"  Y range: [{Y.min()}, {Y.max()}]")   # Y = luminance channel
print(f"  Z range: [{Z.min()}, {Z.max()}]")

# The Y channel in XYZ is the luminance — identical to grayscale conversion
gray_from_xyz = Y
gray_standard = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# These won't be exactly equal because OpenCV uses slightly different
# gamma and white-point assumptions, but they are visually similar
```

---

## 3.8 YCrCb: Luminance and Chrominance

YCrCb (sometimes written YCbCr) separates a color image into one luminance component (Y) and two chrominance (color difference) components:

- **Y:** Luma (brightness), computed similarly to grayscale
- **Cr (Chroma Red):** Red chrominance — the difference between red and luma
- **Cb (Chroma Blue):** Blue chrominance — the difference between blue and luma

YCrCb is used extensively in video compression (JPEG, MPEG, H.264, HEVC all operate in a YCbCr-like space), because the human visual system is less sensitive to chrominance than luminance — the Cr and Cb channels can be subsampled (4:2:0, 4:2:2) without perceptible quality loss.

### 3.8.1 OpenCV YCrCb Ranges

OpenCV stores YCrCb as uint8:
- **Y:** 0–255 (luminance)
- **Cr:** 0–255 (centered at 128, meaning Cr=128 is neutral)
- **Cb:** 0–255 (centered at 128)

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(ycrcb)

print("YCrCb channels:")
print(f"  Y  (luma):        range [{Y.min()}, {Y.max()}]")
print(f"  Cr (red diff):    range [{Cr.min()}, {Cr.max()}]")
print(f"  Cb (blue diff):   range [{Cb.min()}, {Cb.max()}]")
```

### 3.8.2 Skin Detection in YCrCb

YCrCb is widely used for skin detection. Empirically, human skin pixels cluster in a remarkably consistent region of the Cr-Cb plane, largely independent of illumination and ethnicity:

**Skin in YCrCb:** Cr ∈ [133, 173], Cb ∈ [77, 127]

This range was established by Chai and Ngan (1999) and remains widely used. The key insight is that Cr and Cb together capture the "reddish-warm" quality of skin while the Y component handles all illumination variation.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_skin_ycrcb(img_bgr):
    """
    Detect skin pixels using the Chai-Ngan YCrCb skin model.

    Empirical skin range:
      Cr: [133, 173]  (reddish chrominance)
      Cb: [77, 127]   (cool chrominance — skin is warm, so Cb is lower)
      Y:  [0, 255]    (any brightness)

    Returns
    -------
    mask : np.ndarray (H, W), uint8, 0 or 255
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    lower_skin = np.array([0,   133,  77], dtype=np.uint8)   # [Y, Cr, Cb]
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    return mask


def detect_skin_hsv(img_bgr):
    """Alternative skin detection in HSV."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  20,  70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


# Create a test face image (synthetic skin tone)
img_test = np.zeros((300, 300, 3), dtype=np.uint8)
# Approximate caucasian skin tone in BGR: ~[100, 130, 190]
cv2.ellipse(img_test, (150, 140), (80, 100), 0, 0, 360, [105, 135, 195], -1)
# Eyes (non-skin)
cv2.circle(img_test, (120, 120), 15, [40, 40, 40], -1)
cv2.circle(img_test, (180, 120), 15, [40, 40, 40], -1)

mask_ycrcb = detect_skin_ycrcb(img_test)
mask_hsv   = detect_skin_hsv(img_test)

result_ycrcb = cv2.bitwise_and(img_test, img_test, mask=mask_ycrcb)
result_hsv   = cv2.bitwise_and(img_test, img_test, mask=mask_hsv)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, im, title in zip(axes,
    [img_test, mask_ycrcb, mask_hsv, result_ycrcb],
    ["Original", "YCrCb Skin Mask", "HSV Skin Mask", "YCrCb Result"]):
    if im.ndim == 2:
        ax.imshow(im, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 3.9 Comparing Color Spaces on a Real Task

This is the most important section of the chapter. Let us put all five color spaces head-to-head on a single realistic problem — isolating a brightly colored object under varying illumination — and measure their robustness.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compare_color_space_segmentation(img_bgr, target_bgr_sample, figsize=(18, 10)):
    """
    Compare color-based segmentation in BGR, HSV, HLS, LAB, and YCrCb.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image.
    target_bgr_sample : array-like (3,)
        A BGR pixel from the target object (used to auto-compute ranges).
    figsize : tuple
    """
    # Compute color space values of the target sample
    sample = np.uint8([[target_bgr_sample]])
    hsv_sample    = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0, 0]
    hls_sample    = cv2.cvtColor(sample, cv2.COLOR_BGR2HLS)[0, 0]
    lab_sample    = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)[0, 0]
    ycrcb_sample  = cv2.cvtColor(sample, cv2.COLOR_BGR2YCrCb)[0, 0]

    print("Sample pixel values across color spaces:")
    print(f"  BGR   : {list(target_bgr_sample)}")
    print(f"  HSV   : H={hsv_sample[0]} ({hsv_sample[0]*2}°), S={hsv_sample[1]}, V={hsv_sample[2]}")
    print(f"  HLS   : H={hls_sample[0]} ({hls_sample[0]*2}°), L={hls_sample[1]}, S={hls_sample[2]}")
    print(f"  LAB   : L={lab_sample[0]}, a={lab_sample[1]} (natural={lab_sample[1]-128}), b={lab_sample[2]} (natural={lab_sample[2]-128})")
    print(f"  YCrCb : Y={ycrcb_sample[0]}, Cr={ycrcb_sample[1]}, Cb={ycrcb_sample[2]}")

    # Build segmentation masks with tolerance
    tol_bgr = 40
    tol_h   = 15   # hue tolerance in OpenCV units (= 30°)
    tol_sv  = 60
    tol_l   = 50
    tol_ab  = 25
    tol_ycc = 30

    masks = {}

    # BGR
    lower_bgr = np.clip(np.array(target_bgr_sample) - tol_bgr, 0, 255).astype(np.uint8)
    upper_bgr = np.clip(np.array(target_bgr_sample) + tol_bgr, 0, 255).astype(np.uint8)
    masks["BGR"] = cv2.inRange(img_bgr, lower_bgr, upper_bgr)

    # HSV
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_hsv = np.clip([int(hsv_sample[0]) - tol_h, int(hsv_sample[1]) - tol_sv, int(hsv_sample[2]) - tol_sv], 0, [179, 255, 255]).astype(np.uint8)
    upper_hsv = np.clip([int(hsv_sample[0]) + tol_h, 255, 255], 0, [179, 255, 255]).astype(np.uint8)
    masks["HSV"] = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    # HLS
    hls_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    lower_hls = np.clip([int(hls_sample[0]) - tol_h, int(hls_sample[1]) - tol_l, int(hls_sample[2]) - tol_sv], 0, [179, 255, 255]).astype(np.uint8)
    upper_hls = np.clip([int(hls_sample[0]) + tol_h, int(hls_sample[1]) + tol_l, 255], 0, [179, 255, 255]).astype(np.uint8)
    masks["HLS"] = cv2.inRange(hls_img, lower_hls, upper_hls)

    # LAB
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lower_lab = np.clip([0, int(lab_sample[1]) - tol_ab, int(lab_sample[2]) - tol_ab], 0, 255).astype(np.uint8)
    upper_lab = np.clip([255, int(lab_sample[1]) + tol_ab, int(lab_sample[2]) + tol_ab], 0, 255).astype(np.uint8)
    masks["LAB"] = cv2.inRange(lab_img, lower_lab, upper_lab)

    # YCrCb
    ycrcb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower_ycc = np.clip([0, int(ycrcb_sample[1]) - tol_ycc, int(ycrcb_sample[2]) - tol_ycc], 0, 255).astype(np.uint8)
    upper_ycc = np.clip([255, int(ycrcb_sample[1]) + tol_ycc, int(ycrcb_sample[2]) + tol_ycc], 0, 255).astype(np.uint8)
    masks["YCrCb"] = cv2.inRange(ycrcb_img, lower_ycc, upper_ycc)

    # Plot
    n = len(masks)
    fig, axes = plt.subplots(2, n + 1, figsize=figsize)

    # Row 0: masks
    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    for i, (name, mask) in enumerate(masks.items()):
        pct = mask.sum() // 255 / mask.size * 100
        axes[0, i+1].imshow(mask, cmap="gray")
        axes[0, i+1].set_title(f"{name} mask\n({pct:.1f}% of pixels)")
        axes[0, i+1].axis("off")

    # Row 1: results
    axes[1, 0].axis("off")
    for i, (name, mask) in enumerate(masks.items()):
        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        axes[1, i+1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, i+1].set_title(f"{name} result"); axes[1, i+1].axis("off")

    plt.suptitle("Color Space Segmentation Comparison", fontsize=13)
    plt.tight_layout()
    plt.show()
    return masks


# Create test: orange object on gray background
test_img = np.full((300, 400, 3), 120, dtype=np.uint8)
cv2.circle(test_img, (200, 150), 80, [20, 120, 230], -1)  # orange in BGR

compare_color_space_segmentation(test_img, target_bgr_sample=[20, 120, 230])
```

---

## 3.10 Color Space Selection Guide

This section provides a principled decision framework. After understanding the theory and seeing it in code, the real skill is knowing **which space to use for a given problem**.

### 3.10.1 The Decision Table

| Task | Best Space | Why |
|------|-----------|-----|
| Color object detection/tracking | **HSV** | H channel isolates color from brightness |
| Skin detection | **YCrCb** or **LAB** | Cr, Cb (or a*, b*) stable under illumination |
| White/bright object detection | **HLS** | L channel correctly distinguishes white vs saturated |
| Color difference measurement | **LAB** | Euclidean distance ≈ perceived difference (ΔE) |
| Image enhancement / smoothing | **LAB** | Operate on L only; color channels unchanged |
| Feature extraction (SIFT, HOG) | **Grayscale** | Most classical features are designed for grayscale |
| Edge detection | **Grayscale** or **LAB L*** | L channel gives perceptually meaningful edges |
| Background subtraction | **HSV** or **YCrCb** | Illumination changes mostly in V or Y only |
| Video compression preprocessing | **YCrCb** | Human system less sensitive to Cr, Cb |
| Deep learning preprocessing | **RGB** (normalized) | PyTorch / TensorFlow expect RGB |
| Shadow removal | **HSV** (S, V channels) | Shadows mainly affect V; H stays constant |
| Document binarization | **LAB L\*** or **Grayscale** | Removes color noise; keeps text contrast |

### 3.10.2 Why BGR Segmentation Fails Under Lighting Changes

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a green object under two lighting conditions
green_bgr = np.array([30, 180, 30], dtype=np.uint8)  # bright green

# Simulate darker lighting: multiply by 0.4
green_dark = (green_bgr.astype(np.float32) * 0.4).astype(np.uint8)

print(f"Bright green (BGR): {green_bgr}")
print(f"Dark green   (BGR): {green_dark}")

# HSV values
bright_hsv = cv2.cvtColor(np.uint8([[green_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
dark_hsv   = cv2.cvtColor(np.uint8([[green_dark]]), cv2.COLOR_BGR2HSV)[0, 0]

print(f"\nBright green (HSV): H={bright_hsv[0]}, S={bright_hsv[1]}, V={bright_hsv[2]}")
print(f"Dark green   (HSV): H={dark_hsv[0]}, S={dark_hsv[1]}, V={dark_hsv[2]}")

# Key insight: H is nearly identical!
# Only V changes significantly — and we can ignore V entirely in the range check
print(f"\nH difference: {abs(int(bright_hsv[0]) - int(dark_hsv[0]))} "
      f"(small → same hue)")
print(f"V difference: {abs(int(bright_hsv[2]) - int(dark_hsv[2]))} "
      f"(large → illumination captured in V)")
print(f"BGR B difference: {abs(int(green_bgr[0]) - int(green_dark[0]))}")
print(f"BGR G difference: {abs(int(green_bgr[1]) - int(green_dark[1]))}")
print(f"BGR R difference: {abs(int(green_bgr[2]) - int(green_dark[2]))}")
# All three BGR channels changed significantly, making range detection hard
```

---

## 3.11 Complete Project: Multi-Color Object Tracker

Let us build a complete, production-quality multi-color object detector that applies the principles from this chapter:

```python
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class ColorTarget:
    """Defines a color target for segmentation."""
    name: str
    lower1: np.ndarray          # HSV lower bound (range 1)
    upper1: np.ndarray          # HSV upper bound (range 1)
    lower2: Optional[np.ndarray] = None  # For red wraparound
    upper2: Optional[np.ndarray] = None
    display_bgr: Tuple[int, int, int] = (255, 255, 255)  # color for drawing

    def get_mask(self, hsv_img: np.ndarray) -> np.ndarray:
        """Compute the binary mask for this color target."""
        mask = cv2.inRange(hsv_img, self.lower1, self.upper1)
        if self.lower2 is not None:
            mask2 = cv2.inRange(hsv_img, self.lower2, self.upper2)
            mask = cv2.bitwise_or(mask, mask2)
        return mask


# Pre-defined color targets (HSV OpenCV uint8 ranges)
COLORS = {
    "red": ColorTarget(
        name="red",
        lower1=np.array([0,   60, 60]),
        upper1=np.array([10,  255, 255]),
        lower2=np.array([170, 60,  60]),
        upper2=np.array([179, 255, 255]),
        display_bgr=(0, 0, 220),
    ),
    "green": ColorTarget(
        name="green",
        lower1=np.array([35, 60, 60]),
        upper1=np.array([85, 255, 255]),
        display_bgr=(0, 200, 0),
    ),
    "blue": ColorTarget(
        name="blue",
        lower1=np.array([100, 60, 60]),
        upper1=np.array([140, 255, 255]),
        display_bgr=(220, 0, 0),
    ),
    "yellow": ColorTarget(
        name="yellow",
        lower1=np.array([22, 60, 60]),
        upper1=np.array([38, 255, 255]),
        display_bgr=(0, 220, 220),
    ),
    "orange": ColorTarget(
        name="orange",
        lower1=np.array([11, 60, 60]),
        upper1=np.array([21, 255, 255]),
        display_bgr=(0, 140, 255),
    ),
}


def detect_color_objects(
    img_bgr: np.ndarray,
    targets: List[ColorTarget],
    min_area: int = 500,
    morph_kernel_size: int = 5
) -> dict:
    """
    Detect and localize colored objects in a BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format.
    targets : list of ColorTarget
        Colors to detect.
    min_area : int
        Minimum contour area to report (filters noise).
    morph_kernel_size : int
        Kernel size for morphological cleanup of masks.

    Returns
    -------
    dict mapping color name → list of dicts with keys:
        'bbox'     : (x, y, w, h) bounding box
        'center'   : (cx, cy) centroid
        'area'     : pixel area of the detection
        'mask'     : binary mask (same size as input)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Morphological structuring element for noise cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (morph_kernel_size, morph_kernel_size))

    results = {}

    for target in targets:
        # 1. Get raw mask
        mask = target.get_mask(hsv)

        # 2. Morphological cleanup:
        #    - Opening removes small noise (erosion then dilation)
        #    - Closing fills small holes (dilation then erosion)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue  # skip tiny blobs

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Centroid via image moments
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Individual contour mask
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)

            detections.append({
                "bbox"   : (x, y, w, h),
                "center" : (cx, cy),
                "area"   : area,
                "mask"   : contour_mask,
            })

        results[target.name] = detections

    return results


def draw_detections(img_bgr: np.ndarray,
                     results: dict,
                     targets: List[ColorTarget]) -> np.ndarray:
    """
    Draw bounding boxes, centroids, and labels on the image.

    Returns a new annotated BGR image.
    """
    annotated = img_bgr.copy()
    target_map = {t.name: t for t in targets}

    for color_name, detections in results.items():
        target = target_map.get(color_name)
        color_draw = target.display_bgr if target else (255, 255, 255)

        for det in detections:
            x, y, w, h = det["bbox"]
            cx, cy      = det["center"]

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color_draw, 2)

            # Centroid dot
            cv2.circle(annotated, (cx, cy), 5, color_draw, -1)

            # Label
            label = f"{color_name} ({det['area']:.0f}px)"
            cv2.putText(annotated, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_draw, 2,
                        cv2.LINE_AA)

    return annotated


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create a synthetic test scene with known colored objects
    scene = np.full((400, 600, 3), 80, dtype=np.uint8)  # dark gray background

    # Place colored circles (in BGR)
    cv2.circle(scene, (100, 200), 70, [20, 30, 210], -1)    # red
    cv2.circle(scene, (300, 200), 70, [20, 200, 20], -1)    # green
    cv2.circle(scene, (500, 200), 70, [210, 20, 20], -1)    # blue
    cv2.rectangle(scene, (50, 290), (200, 380), [0, 200, 220], -1)  # yellow
    cv2.circle(scene, (400, 330), 50, [0, 130, 255], -1)    # orange

    # Add slight noise to make it realistic
    noise = np.random.normal(0, 8, scene.shape).astype(np.float32)
    scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Run detection
    active_targets = [COLORS[c] for c in ["red", "green", "blue", "yellow", "orange"]]
    detections = detect_color_objects(scene, active_targets, min_area=200)

    # Print summary
    print("Detection Results:")
    for color_name, dets in detections.items():
        print(f"  {color_name}: {len(dets)} object(s) found")
        for i, det in enumerate(dets):
            print(f"    [{i}] center={det['center']}, area={det['area']:.0f}px, "
                  f"bbox={det['bbox']}")

    # Draw and display
    annotated = draw_detections(scene, detections, active_targets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Scene"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detected Objects"); axes[1].axis("off")
    plt.tight_layout()
    plt.show()
```

---

## 3.12 Color Space Conversion Reference Table

A complete reference for all conversions you will commonly need:

```python
import cv2

# Comprehensive conversion code reference
CONVERSIONS = {
    # ── To and from BGR ──────────────────────────────────────────────────────
    "BGR  → RGB"    : cv2.COLOR_BGR2RGB,
    "RGB  → BGR"    : cv2.COLOR_RGB2BGR,
    "BGR  → GRAY"   : cv2.COLOR_BGR2GRAY,
    "GRAY → BGR"    : cv2.COLOR_GRAY2BGR,
    "BGR  → HSV"    : cv2.COLOR_BGR2HSV,
    "HSV  → BGR"    : cv2.COLOR_HSV2BGR,
    "BGR  → HLS"    : cv2.COLOR_BGR2HLS,
    "HLS  → BGR"    : cv2.COLOR_HLS2BGR,
    "BGR  → LAB"    : cv2.COLOR_BGR2LAB,
    "LAB  → BGR"    : cv2.COLOR_LAB2BGR,
    "BGR  → XYZ"    : cv2.COLOR_BGR2XYZ,
    "XYZ  → BGR"    : cv2.COLOR_XYZ2BGR,
    "BGR  → YCrCb"  : cv2.COLOR_BGR2YCrCb,
    "YCrCb→ BGR"    : cv2.COLOR_YCrCb2BGR,
    "BGR  → LUV"    : cv2.COLOR_BGR2LUV,
    "LUV  → BGR"    : cv2.COLOR_LUV2BGR,
    "BGR  → YUV"    : cv2.COLOR_BGR2YUV,
    "YUV  → BGR"    : cv2.COLOR_YUV2BGR,
    # ── Alpha channel variants ────────────────────────────────────────────────
    "BGR  → BGRA"   : cv2.COLOR_BGR2BGRA,
    "BGRA → BGR"    : cv2.COLOR_BGRA2BGR,
    "BGRA → GRAY"   : cv2.COLOR_BGRA2GRAY,
    "BGR  → RGBA"   : cv2.COLOR_BGR2RGBA,
    # ── Direct RGB variants ───────────────────────────────────────────────────
    "RGB  → GRAY"   : cv2.COLOR_RGB2GRAY,
    "RGB  → HSV"    : cv2.COLOR_RGB2HSV,
    "RGB  → LAB"    : cv2.COLOR_RGB2LAB,
    "RGB  → YCrCb"  : cv2.COLOR_RGB2YCrCb,
}

print("Color Conversion Code Reference:")
print("─" * 50)
for name, code in CONVERSIONS.items():
    print(f"  cv2.cvtColor(img, cv2.{cv2._c_to_str(code) or code:40}) # {name}")
```

---

## 3.13 Summary

This chapter covered color spaces at both the mathematical and practical level:

**Why Color Spaces Matter:**
- Different color spaces separate (or mix) color from brightness differently
- The wrong color space makes algorithms brittle; the right one makes them robust
- The choice of color space is an algorithmic decision, not an implementation detail

**The Color Spaces and Their Strengths:**

| Space | Key Property | Primary Use |
|-------|-------------|-------------|
| BGR/RGB | Device-native | I/O, display, DL preprocessing |
| Grayscale | Single intensity channel | Feature extraction, edge detection |
| HSV | H separates color from brightness | Color segmentation, object tracking |
| HLS | L=1.0 is white (not saturated color) | White/bright detection, lane markings |
| LAB | Perceptual uniformity | Color difference (ΔE), color-aware smoothing |
| XYZ | Device-independent foundation | Color science, ICC profiles |
| YCrCb | Y/Cr/Cb separates luma from chroma | Skin detection, video compression |

**Critical Details to Remember:**
- OpenCV HSV: H ∈ [0, 179] (halved to fit uint8) — divide natural degrees by 2
- OpenCV LAB: a and b are shifted: neutral gray = (128, 128) in uint8
- Red wraps around in HSV: need two `cv2.inRange` calls combined with `bitwise_or`
- CIE76 ΔE: values < 1 imperceptible, 1–2 barely visible, > 10 clearly different
- Skin detection: YCrCb is the most reliable; Cr ∈ [133, 173], Cb ∈ [77, 127]
- `cv2.cvtColor` on `float32` input uses full natural ranges; on `uint8` it scales

---

## 3.14 Exercises

### Warm-up

**3.1** Write a function `show_all_spaces(img_bgr)` that converts a BGR image to all six color spaces (grayscale, HSV, HLS, LAB, XYZ, YCrCb) and displays all seven images (including original) in a grid. Label each image with the space name and the range of each channel.

**3.2** Write a function `bgr_to_all_values(bgr_pixel)` that, given a BGR pixel as a list/array, prints its representation in every color space discussed in this chapter. Format the output clearly. Test it on: pure red, pure white, and a skin-tone approximation of your choice.

**3.3** Why does OpenCV store the Hue channel in range [0, 179] instead of [0, 255]? What is the consequence of this for your code? Write an example that demonstrates a bug caused by ignoring this, and then fix it.

### Core

**3.4** Implement a `color_palette_extractor(img_bgr, n_colors=5)` function that finds the `n` dominant colors in an image using k-means clustering in LAB space (use `cv2.kmeans`). Return the colors as BGR arrays and display them as a palette strip. Compare results between clustering in BGR vs LAB space.

**3.5** Illuminate robustness test: Take any colored object image (real or synthetic). Simulate three lighting conditions by multiplying the V channel in HSV: ×0.3 (dim), ×1.0 (normal), ×1.5 (overexposed, clipped at 255). For each, try to segment the object using fixed `cv2.inRange` thresholds in BGR vs HSV. Quantify: how many object pixels are correctly found in each space?

**3.6** Implement `compute_delta_e_map(img1_bgr, img2_bgr)` that returns a float32 heat map of per-pixel perceptual color difference between two same-size images. Normalize the output to [0, 255] and display it with a colormap. Apply it to: (a) the same image vs itself + Gaussian noise, (b) same image vs a brightened version (+50 to all channels), (c) same image vs a color-shifted version.

**3.7** Implement the complete `segment_color` function from Section 3.4.4 with all seven colors, and add a `refine_mask(mask, kernel_size=5, iterations=2)` post-processing step that applies morphological open followed by close. Test your segmentation on an image with multiple colored objects and report detection accuracy.

### Challenge

**3.8** Implement a **histogram-based color model**: given a small sample region (ROI) from an image, compute 2D histograms in (Cr, Cb) for YCrCb, (H, S) for HSV, and (a, b) for LAB. Normalize each histogram to a probability distribution. To detect the color elsewhere, use `cv2.calcBackProject` to project the histogram onto a new image, creating a probability map. Compare the three spaces' back-projection maps.

**3.9** Build a **real-time multi-color tracker** (if webcam is available) or a **video file processor** that: (1) detects all active color targets in each frame, (2) tracks their centroids across frames using a simple nearest-neighbor assignment, (3) draws trajectory trails (the last 30 centroid positions) for each tracked object, (4) displays detection count and FPS in the corner.

**3.10** **Color constancy simulation:** Take a photograph with a known white patch. Implement a simplified white balance correction using: (a) Gray World Assumption (normalize so channel means equal the global mean), (b) White Patch / Max-RGB (divide by per-channel maximum), (c) LAB-based correction (shift the L-independent a and b channels toward zero). Compute ΔE between a manually selected white patch before and after each correction. Which method produces the smallest ΔE on that patch?

---

## Further Reading

- **OpenCV color conversion documentation:** https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html  
- **CIE CIELAB specification:** https://en.wikipedia.org/wiki/CIELAB_color_space  
- **Delta E primer:** http://zschuessler.github.io/DeltaE/learn/  
- **Chai & Ngan (1999) skin detection in YCrCb** — the original reference for the Cr/Cb ranges  
- **LearnOpenCV: Color spaces comparison:** https://learnopencv.com/color-spaces-in-opencv-cpp-python/  
- **ITU-R BT.601 (standard definition) and BT.709 (HD) luminance formulas:** https://www.itu.int/rec/R-REC-BT.601/

---

*Next up: **Chapter 4 — Drawing, Annotation & Basic UI**, where we use OpenCV's drawing API — lines, circles, rectangles, polygons, text, arrows — and build interactive windows with mouse callbacks and trackbars.*
