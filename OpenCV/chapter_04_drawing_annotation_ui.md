# Chapter 4: Drawing, Annotation & Basic UI

> *"The best algorithm is useless if you can't see what it's doing. Drawing is debugging made visual."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Use every OpenCV drawing primitive: lines, arrows, rectangles, circles, ellipses, polygons, and markers
2. Render text with correct sizing using `cv2.getTextSize` and place it precisely
3. Understand `lineType` choices (`LINE_4`, `LINE_8`, `LINE_AA`) and when each matters
4. Build reusable, professional annotation functions for bounding boxes, labels, and overlays
5. Create named windows and control their properties
6. Write a keyboard-driven event loop with `cv2.waitKey`
7. Implement mouse callbacks for all mouse event types
8. Build interactive parameter tuning tools with trackbars
9. Add semi-transparent overlay annotations using alpha blending
10. Build a complete, keyboard-and-mouse-driven annotation tool

---

## 4.1 The Drawing API: Foundations

### 4.1.1 How OpenCV Drawing Works

OpenCV's drawing functions are **in-place** — they modify the image array directly and do not return a new image. This is a deliberate design choice for performance; allocating a new array on every draw call would be expensive in tight video loops.

```python
import cv2
import numpy as np

# All drawing is in-place — img is modified
img = np.zeros((400, 600, 3), dtype=np.uint8)

# The function returns img (same object), but that return value is rarely used
ret = cv2.line(img, (0, 0), (100, 100), (255, 0, 0), 2)
print(f"Return is same object: {ret is img}")  # True

# Practical consequence: always work on a .copy() if you need the original
canvas = np.full((400, 600, 3), 40, dtype=np.uint8)   # dark gray background
working = canvas.copy()   # draw on working, canvas stays pristine
cv2.circle(working, (300, 200), 80, (0, 200, 100), -1)
# canvas is unchanged; working has the circle
```

### 4.1.2 Universal Drawing Arguments

Every drawing function shares a common set of parameters. Understanding them once means understanding all drawing functions:

| Parameter | Type | Description |
|-----------|------|-------------|
| `img` | `np.ndarray` | Target image (modified in-place) |
| `color` | `tuple(B, G, R)` | Color as BGR tuple; use `int` values 0–255 |
| `thickness` | `int` | Stroke width in pixels; `-1` fills closed shapes |
| `lineType` | `int` | `cv2.LINE_4`, `cv2.LINE_8` (default), or `cv2.LINE_AA` |
| `shift` | `int` | Number of fractional bits in point coordinates (usually 0) |

**lineType in depth:**

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Visual comparison of line types on a diagonal line
h, w = 120, 400
imgs = []
labels = []

for ltype, name in [(cv2.LINE_4, "LINE_4\n(4-connected)"),
                     (cv2.LINE_8, "LINE_8\n(8-connected, default)"),
                     (cv2.LINE_AA, "LINE_AA\n(anti-aliased)")]:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Steep diagonal to make aliasing obvious
    cv2.line(img, (20, 100), (w-20, 20), (0, 200, 255), 2, ltype)
    imgs.append(img)
    labels.append(name)

fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for ax, im, label in zip(axes, imgs, labels):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(label, fontsize=10); ax.axis("off")
plt.suptitle("LINE_4 vs LINE_8 vs LINE_AA", y=1.02); plt.tight_layout(); plt.show()
```

> **Rule of thumb:** Always use `cv2.LINE_AA` (anti-aliased) for any drawing that will be displayed on screen or saved to disk. The staircase aliasing of `LINE_8` is distracting in visualizations. Only use `LINE_8` when pixel-perfect reproducibility matters (e.g., mask generation).

### 4.1.3 Color Specification

```python
import cv2
import numpy as np

# ── BGR tuples ────────────────────────────────────────────────────────────────
RED     = (0,   0,   255)
GREEN   = (0,   255, 0)
BLUE    = (255, 0,   0)
YELLOW  = (0,   255, 255)
CYAN    = (255, 255, 0)
MAGENTA = (255, 0,   255)
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0)
GRAY    = (128, 128, 128)
ORANGE  = (0,   165, 255)

# ── From hex string (useful when copying from web/Figma) ──────────────────────
def hex_to_bgr(hex_color: str):
    """Convert '#RRGGBB' hex string to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

teal    = hex_to_bgr("#008080")   # (128, 128, 0)
coral   = hex_to_bgr("#FF6B6B")   # (107, 107, 255) in BGR
indigo  = hex_to_bgr("#4B0082")
print(f"Teal in BGR:   {teal}")
print(f"Coral in BGR:  {coral}")

# ── Grayscale images: single scalar ──────────────────────────────────────────
gray_img = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(gray_img, (100, 100), 60, 200, 3)  # scalar 200 = mid-light gray
```

---

## 4.2 Lines and Arrows

### 4.2.1 cv2.line

```python
import cv2
import numpy as np

canvas = np.zeros((400, 600, 3), dtype=np.uint8)

# cv2.line(img, pt1, pt2, color, thickness=1, lineType=LINE_8, shift=0)
# pt1, pt2: (x, y) tuples — NOTE: x=column, y=row (not numpy [row,col])

# Simple line
cv2.line(canvas, (50, 50), (550, 50), WHITE, 2, cv2.LINE_AA)

# Thick line
cv2.line(canvas, (50, 100), (550, 100), (0, 200, 255), 6, cv2.LINE_AA)

# Thin, anti-aliased diagonal
cv2.line(canvas, (50, 150), (550, 380), (100, 255, 100), 1, cv2.LINE_AA)

# Dashed line (not built-in — implement manually)
def draw_dashed_line(img, pt1, pt2, color, thickness=1,
                     dash_len=15, gap_len=8, line_type=cv2.LINE_AA):
    """
    Draw a dashed line from pt1 to pt2.
    OpenCV has no built-in dashed line — we simulate by drawing short segments.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    total  = np.hypot(dx, dy)
    if total == 0:
        return
    # Unit vector along the line
    ux, uy = dx / total, dy / total
    dist = 0.0
    drawing = True
    while dist < total:
        seg_end = min(dist + (dash_len if drawing else gap_len), total)
        if drawing:
            p1 = (int(x1 + ux * dist),    int(y1 + uy * dist))
            p2 = (int(x1 + ux * seg_end), int(y1 + uy * seg_end))
            cv2.line(img, p1, p2, color, thickness, line_type)
        dist += dash_len if drawing else gap_len
        drawing = not drawing

draw_dashed_line(canvas, (50, 200), (550, 200), YELLOW, thickness=2)
draw_dashed_line(canvas, (50, 250), (550, 350), (200, 100, 255), thickness=2,
                 dash_len=20, gap_len=12)
```

### 4.2.2 cv2.arrowedLine

```python
import cv2
import numpy as np

canvas = np.zeros((300, 500, 3), dtype=np.uint8)

# cv2.arrowedLine(img, pt1, pt2, color, thickness, line_type, shift, tipLength)
# tipLength: fraction of the line length for the arrowhead (default 0.1)

# Simple arrow (tip at pt2)
cv2.arrowedLine(canvas, (50, 150), (400, 150), WHITE, 2, cv2.LINE_AA)

# Short arrow with larger tip
cv2.arrowedLine(canvas, (50, 50), (200, 50), YELLOW, 3, cv2.LINE_AA,
                tipLength=0.3)

# Annotation: draw a flow field of arrows
def draw_flow_field(img, step=40):
    """Draw random unit vectors as arrows on a grid."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(42)
    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            angle = rng.uniform(0, 2 * np.pi)
            length = int(step * 0.4)
            ex = int(x + np.cos(angle) * length)
            ey = int(y + np.sin(angle) * length)
            cv2.arrowedLine(img, (x, y), (ex, ey), CYAN, 1, cv2.LINE_AA,
                            tipLength=0.35)

draw_flow_field(canvas)
```

---

## 4.3 Rectangles

```python
import cv2
import numpy as np

canvas = np.zeros((400, 600, 3), dtype=np.uint8)

# cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift)
# pt1 = top-left corner, pt2 = bottom-right corner (both as (x, y))

# Outline rectangle
cv2.rectangle(canvas, (50, 50), (250, 200), GREEN, 2, cv2.LINE_AA)

# Filled rectangle (thickness=-1)
cv2.rectangle(canvas, (300, 50), (550, 200), BLUE, -1)

# Thin border on a filled rectangle (draw twice: fill then outline)
cv2.rectangle(canvas, (50, 250), (250, 370), RED, -1)
cv2.rectangle(canvas, (50, 250), (250, 370), WHITE, 1, cv2.LINE_AA)

# ── Using cv2.Rect alternative form ──────────────────────────────────────────
# cv2.rectangle also accepts a Rect object: (x, y, width, height)
# Useful when you have a bounding box as (x, y, w, h) from contours/detectors
x, y, w, h = 300, 250, 200, 120
cv2.rectangle(canvas, (x, y), (x + w, y + h), YELLOW, 2, cv2.LINE_AA)

# ── Rounded rectangle (not built-in, use cv2.drawContours or polylines) ──────
def draw_rounded_rect(img, pt1, pt2, color, radius=15,
                      thickness=2, line_type=cv2.LINE_AA):
    """
    Draw a rectangle with rounded corners.
    pt1=(x1,y1) top-left, pt2=(x2,y2) bottom-right.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius

    # Four straight edges (inset by radius)
    cv2.line(img, (x1 + r, y1),     (x2 - r, y1),     color, thickness, line_type)  # top
    cv2.line(img, (x1 + r, y2),     (x2 - r, y2),     color, thickness, line_type)  # bottom
    cv2.line(img, (x1,     y1 + r), (x1,     y2 - r), color, thickness, line_type)  # left
    cv2.line(img, (x2,     y1 + r), (x2,     y2 - r), color, thickness, line_type)  # right

    # Four arc corners (90-degree arcs)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90,  color, thickness, line_type)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90,  color, thickness, line_type)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90,  color, thickness, line_type)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90,  color, thickness, line_type)

draw_rounded_rect(canvas, (50, 330), (250, 390), MAGENTA, radius=12)
```

---

## 4.4 Circles and Ellipses

### 4.4.1 cv2.circle

```python
import cv2
import numpy as np

canvas = np.zeros((400, 600, 3), dtype=np.uint8)

# cv2.circle(img, center, radius, color, thickness, lineType, shift)
# center: (x, y)   radius: int pixels

# Outline circle
cv2.circle(canvas, (150, 150), 80, CYAN, 2, cv2.LINE_AA)

# Filled circle
cv2.circle(canvas, (350, 150), 80, RED, -1)

# Concentric circles
for r in range(20, 100, 15):
    # Interpolate color: inner=yellow, outer=blue
    t = r / 100.0
    color = (int(255 * t), int(255 * (1 - t)), int(255 * (1 - t)))
    cv2.circle(canvas, (500, 150), r, color, 2, cv2.LINE_AA)

# Ring: filled circle with a smaller filled black circle on top
cv2.circle(canvas, (150, 320), 70, WHITE, -1)
cv2.circle(canvas, (150, 320), 50, BLACK, -1)  # "erase" center
```

### 4.4.2 cv2.ellipse

The ellipse function is more complex than circle and merits a detailed explanation:

```python
import cv2
import numpy as np

canvas = np.zeros((400, 600, 3), dtype=np.uint8)

# cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color,
#             thickness=-1, lineType=LINE_8, shift=0)
#
# center:     (x, y) center of the ellipse
# axes:       (semi-major, semi-minor) half-axis lengths in pixels
# angle:      rotation of the MAJOR axis in DEGREES, CLOCKWISE
# startAngle: start of the arc, measured CLOCKWISE from the major axis
# endAngle:   end of the arc, measured CLOCKWISE from the major axis
# → 0 to 360 = full ellipse; 0 to 180 = upper half

# Full upright ellipse
cv2.ellipse(canvas, (150, 100), (100, 50), 0, 0, 360, GREEN, 2, cv2.LINE_AA)

# Rotated 45 degrees
cv2.ellipse(canvas, (350, 100), (100, 50), 45, 0, 360, YELLOW, 2, cv2.LINE_AA)

# Half ellipse (arc): upper half
cv2.ellipse(canvas, (500, 150), (80, 50), 0, 0, 180, CYAN, 3, cv2.LINE_AA)

# Filled ellipse sector (like a pie slice)
cv2.ellipse(canvas, (150, 280), (90, 60), 0, 30, 210, MAGENTA, -1)

# ── Drawing a pie/donut chart ─────────────────────────────────────────────────
def draw_pie_chart(img, center, radius, values, colors, start_angle=0):
    """Draw a simple pie chart using cv2.ellipse sectors."""
    total = sum(values)
    angle = start_angle
    for val, color in zip(values, colors):
        sweep = val / total * 360
        cv2.ellipse(img, center, (radius, radius), 0,
                    angle, angle + sweep, color, -1)
        # White separator line
        rad = np.radians(angle)
        ex = int(center[0] + radius * np.cos(rad))
        ey = int(center[1] + radius * np.sin(rad))
        cv2.line(img, center, (ex, ey), WHITE, 2, cv2.LINE_AA)
        angle += sweep

pie_data   = [30, 20, 25, 15, 10]
pie_colors = [RED, BLUE, GREEN, YELLOW, CYAN]
draw_pie_chart(canvas, (450, 300), 80, pie_data, pie_colors)
```

---

## 4.5 Polygons and Polylines

### 4.5.1 cv2.polylines and cv2.fillPoly

```python
import cv2
import numpy as np

canvas = np.zeros((500, 600, 3), dtype=np.uint8)

# Points must be shape (N, 1, 2) and dtype int32
# The reshape(-1, 1, 2) idiom converts a list of (x, y) tuples

# ── Open polyline ─────────────────────────────────────────────────────────────
pts_open = np.array([[50, 50], [150, 150], [250, 80], [350, 200], [450, 100]],
                     dtype=np.int32).reshape(-1, 1, 2)

cv2.polylines(canvas, [pts_open], isClosed=False, color=CYAN,
              thickness=2, lineType=cv2.LINE_AA)

# ── Closed polyline (polygon outline) ────────────────────────────────────────
pts_closed = np.array([[50, 280], [150, 230], [250, 280], [220, 370], [80, 370]],
                       dtype=np.int32).reshape(-1, 1, 2)

cv2.polylines(canvas, [pts_closed], isClosed=True, color=YELLOW,
              thickness=2, lineType=cv2.LINE_AA)

# ── Filled polygon ────────────────────────────────────────────────────────────
pts_filled = np.array([[350, 250], [480, 230], [530, 320], [480, 410], [350, 390]],
                       dtype=np.int32)

cv2.fillPoly(canvas, [pts_filled], color=RED)
cv2.polylines(canvas, [pts_filled.reshape(-1, 1, 2)], True, WHITE, 2, cv2.LINE_AA)

# ── Regular polygon helper ────────────────────────────────────────────────────
def regular_polygon(center, radius, n_sides, rotation_deg=0):
    """
    Compute vertices of a regular polygon.

    Parameters
    ----------
    center : (x, y)
    radius : float   — circumradius (center to vertex)
    n_sides : int
    rotation_deg : float   — rotation offset in degrees

    Returns
    -------
    np.ndarray of shape (n_sides, 1, 2), dtype int32
    """
    angles = [np.radians(rotation_deg + 360 * i / n_sides)
              for i in range(n_sides)]
    pts = [(int(center[0] + radius * np.cos(a)),
            int(center[1] + radius * np.sin(a)))
           for a in angles]
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

# Draw a hexagon, pentagon, and triangle
shapes = [
    (regular_polygon((120, 430), 50, 6, -90), CYAN),
    (regular_polygon((270, 430), 50, 5, -90), GREEN),
    (regular_polygon((420, 430), 50, 3, -90), MAGENTA),
]
for pts, color in shapes:
    cv2.fillPoly(canvas, [pts], color=color)
    cv2.polylines(canvas, [pts], True, WHITE, 1, cv2.LINE_AA)

# ── Multiple polygons in one call ─────────────────────────────────────────────
triangles = [
    np.array([[500, 50], [580, 200], [420, 200]], dtype=np.int32),
    np.array([[510, 70], [570, 180], [440, 180]], dtype=np.int32),
]
cv2.fillPoly(canvas, triangles, color=(80, 80, 200))
```

---

## 4.6 Markers

`cv2.drawMarker` draws predefined scientific-plot-style markers at a point — useful for visualizing keypoints, centroids, and detected features:

```python
import cv2
import numpy as np

canvas = np.zeros((300, 700, 3), dtype=np.uint8)

# cv2.drawMarker(img, position, color, markerType, markerSize, thickness, line_type)
marker_types = [
    (cv2.MARKER_CROSS,          "CROSS"),
    (cv2.MARKER_TILTED_CROSS,   "TILTED_CROSS"),
    (cv2.MARKER_STAR,           "STAR"),
    (cv2.MARKER_DIAMOND,        "DIAMOND"),
    (cv2.MARKER_SQUARE,         "SQUARE"),
    (cv2.MARKER_TRIANGLE_UP,    "TRIANGLE_UP"),
    (cv2.MARKER_TRIANGLE_DOWN,  "TRIANGLE_DOWN"),
]

for i, (mtype, name) in enumerate(marker_types):
    x = 50 + i * 90
    cv2.drawMarker(canvas, (x, 100), CYAN, mtype,
                   markerSize=30, thickness=2, line_type=cv2.LINE_AA)
    # Label below
    cv2.putText(canvas, name, (x - 30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)

# Scatter plot visualization
rng = np.random.default_rng(42)
n = 50
xs = (rng.random(n) * 600 + 50).astype(int)
ys = (rng.random(n) * 150 + 200).astype(int)
classes = rng.integers(0, 3, n)
class_colors = [RED, GREEN, BLUE]
class_markers = [cv2.MARKER_CIRCLE if hasattr(cv2, 'MARKER_CIRCLE')
                 else cv2.MARKER_CROSS,
                 cv2.MARKER_DIAMOND, cv2.MARKER_TRIANGLE_UP]
# Note: MARKER_CIRCLE was added in OpenCV 4.x contrib; use MARKER_CROSS as fallback

for x, y, c in zip(xs, ys, classes):
    cv2.drawMarker(canvas, (x, y), class_colors[c],
                   cv2.MARKER_CROSS, markerSize=12, thickness=2, line_type=cv2.LINE_AA)
```

---

## 4.7 Text Rendering

Text in OpenCV requires more care than the geometric primitives. The key challenge is **positioning** — `cv2.putText` places text with the bottom-left corner of the first character at the origin point, not the top-left. Getting this wrong shifts text up or down unexpectedly.

### 4.7.1 Fonts and Basic putText

```python
import cv2
import numpy as np

canvas = np.zeros((600, 700, 3), dtype=np.uint8)

# Available fonts
fonts = [
    (cv2.FONT_HERSHEY_SIMPLEX,        "HERSHEY_SIMPLEX"),
    (cv2.FONT_HERSHEY_PLAIN,          "HERSHEY_PLAIN"),
    (cv2.FONT_HERSHEY_DUPLEX,         "HERSHEY_DUPLEX"),
    (cv2.FONT_HERSHEY_COMPLEX,        "HERSHEY_COMPLEX"),
    (cv2.FONT_HERSHEY_TRIPLEX,        "HERSHEY_TRIPLEX"),
    (cv2.FONT_HERSHEY_COMPLEX_SMALL,  "HERSHEY_COMPLEX_SMALL"),
    (cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, "HERSHEY_SCRIPT_SIMPLEX"),
    (cv2.FONT_HERSHEY_SCRIPT_COMPLEX, "HERSHEY_SCRIPT_COMPLEX"),
    (cv2.FONT_ITALIC,                 "ITALIC (modifier flag)"),
]

# cv2.putText(img, text, org, fontFace, fontScale, color,
#             thickness=1, lineType=LINE_8, bottomLeftOrigin=False)
# org: (x, y) of the BOTTOM-LEFT of the first character baseline

for i, (font, name) in enumerate(fonts):
    y = 50 + i * 60
    cv2.putText(canvas, f"{name}: The quick brown fox",
                (10, y), font, 0.6, WHITE, 1, cv2.LINE_AA)

# ── FONT_ITALIC is a MODIFIER, not a standalone font ─────────────────────────
# Combine with bitwise OR
italic_duplex = cv2.FONT_HERSHEY_DUPLEX | cv2.FONT_ITALIC
cv2.putText(canvas, "DUPLEX + ITALIC", (10, 580),
            italic_duplex, 0.8, YELLOW, 1, cv2.LINE_AA)
```

### 4.7.2 cv2.getTextSize — Accurate Text Placement

The most important text function that beginners skip. Without it, you cannot reliably center text, draw background boxes, or align text to baselines:

```python
import cv2
import numpy as np

# cv2.getTextSize(text, fontFace, fontScale, thickness)
# Returns: (width, height), baseline
# where baseline is the distance from the bottom of the text to the baseline

text = "Hello, OpenCV!"
font = cv2.FONT_HERSHEY_DUPLEX
scale = 1.2
thickness = 2

(text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

print(f"Text dimensions: {text_w} × {text_h} pixels")
print(f"Baseline:        {baseline} pixels below text origin")

canvas = np.full((200, 500, 3), 40, dtype=np.uint8)

# Place the text centered in the canvas
cx, cy = 250, 100
org_x = cx - text_w // 2
org_y = cy + text_h // 2   # putText uses bottom-left, so offset upward

# Draw text bounding box (for debugging)
top_left     = (org_x, org_y - text_h - baseline)
bottom_right = (org_x + text_w, org_y + baseline)
cv2.rectangle(canvas, top_left, bottom_right, GRAY, 1)

# Draw the baseline
cv2.line(canvas, (org_x, org_y), (org_x + text_w, org_y), RED, 1)

# Draw the text
cv2.putText(canvas, text, (org_x, org_y), font, scale, WHITE, thickness, cv2.LINE_AA)

# Explanation of the coordinate system:
# org_y (the origin passed to putText) is the BASELINE
# Characters sit ABOVE the baseline (text_h above)
# Descenders (g, p, y, etc.) extend BELOW the baseline (baseline pixels below)
```

### 4.7.3 A Production-Grade Text Utility

Text with a legible background box is essential for real CV applications — raw white text on a busy image background is often unreadable:

```python
import cv2
import numpy as np


def put_text_with_background(
    img,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=2,
    padding=6,
    alpha=0.7,
    line_type=cv2.LINE_AA,
):
    """
    Render text with a semi-transparent background box.

    Parameters
    ----------
    img : np.ndarray   BGR image (modified in-place)
    text : str
    org : (x, y)       Bottom-left of text baseline
    alpha : float      Background opacity (0=transparent, 1=opaque)
    padding : int      Pixels of padding around text
    """
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = org
    # Box coordinates
    bx1 = x - padding
    by1 = y - th - baseline - padding
    bx2 = x + tw + padding
    by2 = y + baseline + padding

    # Clamp to image bounds
    h_img, w_img = img.shape[:2]
    bx1 = max(0, bx1); by1 = max(0, by1)
    bx2 = min(w_img, bx2); by2 = min(h_img, by2)

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Text on top
    cv2.putText(img, text, (x, y), font, font_scale,
                text_color, thickness, line_type)

    return img


def put_text_centered(
    img,
    text,
    center,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.8,
    color=(255, 255, 255),
    thickness=2,
    line_type=cv2.LINE_AA,
    with_background=False,
    bg_color=(0, 0, 0),
    bg_alpha=0.6,
):
    """
    Render text centered at `center` (cx, cy).
    """
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cx, cy = center
    org = (cx - tw // 2, cy + th // 2)

    if with_background:
        put_text_with_background(img, text, org, font, font_scale,
                                  color, bg_color, thickness, alpha=bg_alpha)
    else:
        cv2.putText(img, text, org, font, font_scale, color, thickness, line_type)

    return img


# ── Multi-line text helper ─────────────────────────────────────────────────────
def put_multiline_text(
    img,
    lines,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    color=(255, 255, 255),
    thickness=1,
    line_spacing=1.4,
    line_type=cv2.LINE_AA,
):
    """
    Render a list of strings as multi-line text.

    Parameters
    ----------
    lines : list of str
    line_spacing : float   Multiplier on line height (1.0 = tight, 1.5 = spacious)
    """
    x, y = org
    for line in lines:
        (_, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, line_type)
        y += int((th + baseline) * line_spacing)
    return img


# ── Demo ──────────────────────────────────────────────────────────────────────
canvas = np.full((350, 600, 3), 60, dtype=np.uint8)
cv2.rectangle(canvas, (0, 0), (599, 349), (80, 80, 80), 1)

put_text_with_background(canvas, "cv2.putText with background",
                          (20, 60), font_scale=0.8, padding=8)

put_text_centered(canvas, "Centered text, no bg",
                   (300, 140), font_scale=0.9, color=YELLOW)

put_text_centered(canvas, "Centered + background",
                   (300, 200), font_scale=0.9, color=WHITE,
                   with_background=True, bg_color=BLUE)

put_multiline_text(canvas,
    ["Line 1: Detection Results", "Line 2: 14 objects found",
     "Line 3: Confidence: 0.92", "Line 4: FPS: 28.4"],
    org=(20, 270), font_scale=0.6, line_spacing=1.5)
```

---

## 4.8 Windows and the Display Loop

### 4.8.1 Named Windows

```python
import cv2
import numpy as np

img = np.zeros((300, 500, 3), dtype=np.uint8)

# ── Window creation ───────────────────────────────────────────────────────────
# cv2.namedWindow(name, flags)
# flags:
#   cv2.WINDOW_NORMAL     — resizable by user, can go fullscreen
#   cv2.WINDOW_AUTOSIZE   — sized to fit image, not resizable (default for imshow)
#   cv2.WINDOW_KEEPRATIO  — maintain aspect ratio when resizing
#   cv2.WINDOW_FREERATIO  — allow arbitrary aspect ratio

cv2.namedWindow("Resizable Window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Fixed Window",    cv2.WINDOW_AUTOSIZE)

# Set window size programmatically (only works for WINDOW_NORMAL)
cv2.resizeWindow("Resizable Window", 800, 600)

# Move window to specific screen position
cv2.moveWindow("Resizable Window", 100, 100)

cv2.imshow("Resizable Window", img)
cv2.imshow("Fixed Window", img)

cv2.waitKey(1000)  # show for 1 second then continue
cv2.destroyAllWindows()
```

### 4.8.2 The waitKey Event Loop

`cv2.waitKey` is the heartbeat of every OpenCV desktop application. It processes all pending window events (redraws, mouse events, keyboard events) and optionally waits for a key:

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)

cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
cv2.imshow("Demo", img)

# ── Single-image display with key detection ───────────────────────────────────
while True:
    key = cv2.waitKey(30) & 0xFF   # & 0xFF: mask to 8 bits (fixes Linux 32-bit issue)

    if key == 27:              # ESC
        print("ESC pressed — exiting")
        break
    elif key == ord("q"):      # q
        print("q pressed — exiting")
        break
    elif key == ord("s"):      # s — save screenshot
        cv2.imwrite("screenshot.png", img)
        print("Saved screenshot.png")
    elif key == ord("g"):      # g — toggle grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Demo", img)
    elif key == 0:             # up arrow
        print("Up arrow")
    elif key == 1:             # down arrow (platform-dependent)
        print("Down arrow")
    elif key != 255:           # any other key (255 = no key)
        print(f"Key pressed: {key} (char: {chr(key) if 32 <= key < 127 else '?'})")

cv2.destroyAllWindows()

# ── Common waitKey patterns ───────────────────────────────────────────────────
# cv2.waitKey(0)    → wait forever (for static image display)
# cv2.waitKey(1)    → minimal wait (for video/real-time loops)
# cv2.waitKey(33)   → ~30 fps cap (1000ms / 30 ≈ 33ms)
# cv2.waitKey(1000) → pause 1 second

# ── Arrow key codes (platform varies slightly) ────────────────────────────────
KEY_CODES = {
    27   : "ESC",
    13   : "ENTER",
    32   : "SPACE",
    8    : "BACKSPACE",
    9    : "TAB",
    # Arrow keys on most platforms:
    81   : "LEFT",     82   : "UP",
    83   : "RIGHT",    84   : "DOWN",
    # macOS alternative:
    2424832: "LEFT",  2490368: "UP",
    2555904: "RIGHT", 2621440: "DOWN",
}
```

---

## 4.9 Mouse Callbacks

Mouse callbacks allow your application to respond to mouse events in real time. They are the foundation of interactive annotation tools, ROI selection, and drawing applications.

### 4.9.1 Mouse Event Types

```python
import cv2

# All mouse event types
mouse_events = {
    cv2.EVENT_MOUSEMOVE       : "Mouse moved (no button)",
    cv2.EVENT_LBUTTONDOWN     : "Left button pressed",
    cv2.EVENT_RBUTTONDOWN     : "Right button pressed",
    cv2.EVENT_MBUTTONDOWN     : "Middle button pressed",
    cv2.EVENT_LBUTTONUP       : "Left button released",
    cv2.EVENT_RBUTTONUP       : "Right button released",
    cv2.EVENT_MBUTTONUP       : "Middle button released",
    cv2.EVENT_LBUTTONDBLCLK   : "Left button double-click",
    cv2.EVENT_RBUTTONDBLCLK   : "Right button double-click",
    cv2.EVENT_MBUTTONDBLCLK   : "Middle button double-click",
    cv2.EVENT_MOUSEWHEEL      : "Scroll wheel (vertical)",
    cv2.EVENT_MOUSEHWHEEL     : "Scroll wheel (horizontal)",
}

# flags parameter bitmask
mouse_flags = {
    cv2.EVENT_FLAG_LBUTTON    : "Left button held",
    cv2.EVENT_FLAG_RBUTTON    : "Right button held",
    cv2.EVENT_FLAG_MBUTTON    : "Middle button held",
    cv2.EVENT_FLAG_CTRLKEY    : "Ctrl key held",
    cv2.EVENT_FLAG_SHIFTKEY   : "Shift key held",
    cv2.EVENT_FLAG_ALTKEY     : "Alt key held",
}

print("Mouse event types:")
for code, description in mouse_events.items():
    print(f"  cv2.{cv2._c_to_str(code) if hasattr(cv2, '_c_to_str') else code}: {description}")
```

### 4.9.2 Mouse Callback Signature and Registration

```python
import cv2
import numpy as np

# The callback signature is FIXED — exactly these 5 parameters:
# def callback(event, x, y, flags, param):
#   event : int   — one of cv2.EVENT_* constants
#   x, y  : int   — cursor position in the WINDOW (not image, if scaled)
#   flags : int   — bitmask of cv2.EVENT_FLAG_* values
#   param : any   — user data passed at registration time

def my_mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left click at ({x}, {y}), flags={flags:#010b}")
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        print(f"Drag at ({x}, {y})")
    elif event == cv2.EVENT_MOUSEWHEEL:
        # flags > 0 = scroll up, flags < 0 = scroll down
        direction = "up" if flags > 0 else "down"
        print(f"Scroll {direction} at ({x}, {y})")

img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.namedWindow("Mouse Demo")

# Register the callback — must be done AFTER namedWindow
# cv2.setMouseCallback(windowName, callback, param=None)
cv2.setMouseCallback("Mouse Demo", my_mouse_callback)

cv2.imshow("Mouse Demo", img)
cv2.waitKey(3000)   # show for 3 seconds
cv2.destroyAllWindows()
```

### 4.9.3 Interactive ROI Selector

A complete, production-quality rectangular ROI selector using mouse callbacks:

```python
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ROISelector:
    """
    Interactive rectangular ROI selector using mouse callbacks.

    Usage:
        selector = ROISelector(img)
        roi = selector.select()   # blocks until user finishes
        if roi is not None:
            x, y, w, h = roi
    """
    original_img: np.ndarray
    window_name: str = "ROI Selector"
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2

    # State (not constructor args)
    _drawing: bool = field(default=False, init=False)
    _start:   Optional[Tuple[int, int]] = field(default=None, init=False)
    _end:     Optional[Tuple[int, int]] = field(default=None, init=False)
    _done:    bool = field(default=False, init=False)
    _display: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        self._display = self.original_img.copy()

    def _callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start   = (x, y)
            self._end     = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._end = (x, y)
            # Live preview
            self._display = self.original_img.copy()
            cv2.rectangle(self._display, self._start, self._end,
                          self.color, self.thickness, cv2.LINE_AA)
            # Show coordinates
            put_text_with_background(
                self._display,
                f"({self._start[0]},{self._start[1]}) → ({x},{y})",
                (10, 25), font_scale=0.6, padding=5
            )
            cv2.imshow(self.window_name, self._display)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._end     = (x, y)
            self._done    = True
            self._display = self.original_img.copy()
            cv2.rectangle(self._display, self._start, self._end,
                          self.color, self.thickness, cv2.LINE_AA)
            cv2.imshow(self.window_name, self._display)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: reset selection
            self._drawing = False
            self._start   = None
            self._end     = None
            self._done    = False
            self._display = self.original_img.copy()
            cv2.imshow(self.window_name, self._display)

    def select(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Show the window and let user draw a ROI.

        Returns (x, y, w, h) or None if cancelled (ESC/q).
        Controls:
            Left-drag  : draw rectangle
            Right-click: reset
            ENTER      : confirm selection
            ESC / q    : cancel
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._callback)
        cv2.imshow(self.window_name, self._display)

        instructions = [
            "Left drag: draw ROI",
            "Right click: reset",
            "ENTER: confirm",
            "ESC/q: cancel",
        ]
        for i, line in enumerate(instructions):
            put_text_with_background(
                self._display,
                line, (10, 25 + i * 28),
                font_scale=0.55, padding=4
            )
        cv2.imshow(self.window_name, self._display)

        roi = None
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, ord("\r")):   # ENTER
                if self._done and self._start and self._end:
                    x1, y1 = self._start
                    x2, y2 = self._end
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    if w > 0 and h > 0:
                        roi = (x, y, w, h)
                break
            elif key in (27, ord("q")):  # ESC / q
                break

        cv2.destroyWindow(self.window_name)
        return roi
```

### 4.9.4 Point Collection Tool

```python
import cv2
import numpy as np
from typing import List, Tuple


def collect_points(img_bgr, n_points=None, window_name="Point Collector"):
    """
    Interactively collect click coordinates from an image.

    Parameters
    ----------
    img_bgr : np.ndarray
    n_points : int or None
        If given, stop after n_points. If None, collect until ENTER.

    Returns
    -------
    list of (x, y) tuples
    """
    points: List[Tuple[int, int]] = []
    display = img_bgr.copy()

    def _callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw point
            cv2.circle(display, (x, y), 5, RED, -1, cv2.LINE_AA)
            cv2.circle(display, (x, y), 6, WHITE, 1, cv2.LINE_AA)
            # Label with index
            put_text_with_background(display, str(len(points)), (x + 8, y),
                                      font_scale=0.6)
            # Draw line to previous point
            if len(points) > 1:
                cv2.line(display, points[-2], points[-1], YELLOW, 1, cv2.LINE_AA)
            cv2.imshow(window_name, display)

            if n_points and len(points) >= n_points:
                # Signal completion via a flag in closure
                param["done"] = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Undo last point
            if points:
                points.pop()
                # Redraw
                nonlocal display
                display = img_bgr.copy()
                for i, (px, py) in enumerate(points):
                    cv2.circle(display, (px, py), 5, RED, -1, cv2.LINE_AA)
                    put_text_with_background(display, str(i + 1), (px + 8, py),
                                              font_scale=0.6)
                    if i > 0:
                        cv2.line(display, points[i-1], (px, py), YELLOW, 1, cv2.LINE_AA)
                cv2.imshow(window_name, display)

    state = {"done": False}
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _callback, state)
    cv2.imshow(window_name, display)

    while not state["done"]:
        key = cv2.waitKey(20) & 0xFF
        if key in (13,):   # ENTER
            break
        elif key in (27, ord("q")):
            points.clear()
            break

    cv2.destroyWindow(window_name)
    return points
```

---

## 4.10 Trackbars: Interactive Parameter Tuning

Trackbars are the fastest way to tune algorithm parameters interactively. The canonical use case: instead of editing code to change a threshold, move a slider and see the result instantly.

### 4.10.1 Creating and Reading Trackbars

```python
import cv2
import numpy as np

# ── Null callback (common pattern when you poll in a loop) ────────────────────
def nothing(val):
    pass  # callback does nothing — we read via getTrackbarPos in the loop

img = np.zeros((100, 500, 3), dtype=np.uint8)
cv2.namedWindow("Trackbar Demo")
cv2.imshow("Trackbar Demo", img)

# cv2.createTrackbar(trackbarName, windowName, value, count, onChange)
# value = initial position
# count = maximum value (minimum is always 0)
cv2.createTrackbar("Red",        "Trackbar Demo", 0,   255, nothing)
cv2.createTrackbar("Green",      "Trackbar Demo", 0,   255, nothing)
cv2.createTrackbar("Blue",       "Trackbar Demo", 0,   255, nothing)
cv2.createTrackbar("Brightness", "Trackbar Demo", 128, 255, nothing)

while True:
    r = cv2.getTrackbarPos("Red",        "Trackbar Demo")
    g = cv2.getTrackbarPos("Green",      "Trackbar Demo")
    b = cv2.getTrackbarPos("Blue",       "Trackbar Demo")
    bright = cv2.getTrackbarPos("Brightness", "Trackbar Demo")

    # Apply brightness: scale channels
    factor = bright / 128.0
    color_img = np.zeros((100, 500, 3), dtype=np.uint8)
    color_img[:] = [
        int(b * factor),
        int(g * factor),
        int(r * factor),
    ]

    label = f"BGR=({b},{g},{r})  Brightness={bright}  Factor={factor:.2f}"
    put_text_with_background(color_img, label, (10, 70), font_scale=0.55)
    cv2.imshow("Trackbar Demo", color_img)

    if cv2.waitKey(30) & 0xFF in (27, ord("q")):
        break

cv2.destroyAllWindows()
```

### 4.10.2 Real-Time Blur Tuner — A Complete Example

```python
import cv2
import numpy as np


def blur_tuner(img_bgr):
    """
    Interactive blur parameter tuner.
    Shows Gaussian, Median, and Bilateral blur side by side.
    Trackbars control kernel sizes and parameters.
    Press ESC or q to exit.
    """
    window = "Blur Tuner — ESC to exit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1200, 500)

    def nothing(v): pass

    # Gaussian
    cv2.createTrackbar("Gaussian ksize", window, 1,  25, nothing)  # will be made odd
    cv2.createTrackbar("Gaussian sigma", window, 0,  50, nothing)  # 0 = auto

    # Median
    cv2.createTrackbar("Median ksize",   window, 1,  25, nothing)  # must be odd ≥ 1

    # Bilateral
    cv2.createTrackbar("Bilateral d",    window, 5,  25, nothing)  # diameter
    cv2.createTrackbar("Bilateral sColor", window, 75, 200, nothing)
    cv2.createTrackbar("Bilateral sSpace", window, 75, 200, nothing)

    def make_odd(v, minimum=1):
        """Ensure kernel size is odd and ≥ minimum."""
        v = max(minimum, v)
        return v if v % 2 == 1 else v + 1

    while True:
        gk = make_odd(cv2.getTrackbarPos("Gaussian ksize", window))
        gs = cv2.getTrackbarPos("Gaussian sigma", window) / 10.0
        mk = make_odd(cv2.getTrackbarPos("Median ksize", window))
        bd = max(1, cv2.getTrackbarPos("Bilateral d", window))
        bsc = cv2.getTrackbarPos("Bilateral sColor", window)
        bss = cv2.getTrackbarPos("Bilateral sSpace", window)

        # Apply filters
        gauss    = cv2.GaussianBlur(img_bgr, (gk, gk), gs)
        median   = cv2.medianBlur(img_bgr, mk)
        bilateral = cv2.bilateralFilter(img_bgr, bd, bsc, bss)

        # Annotate each result
        def annotate(im, label):
            out = im.copy()
            put_text_with_background(out, label, (10, 30),
                                      font_scale=0.6, padding=5)
            return out

        g_label = f"Gaussian k={gk}, σ={gs:.1f}"
        m_label = f"Median k={mk}"
        b_label = f"Bilateral d={bd}, sC={bsc}, sS={bss}"

        row = np.hstack([annotate(gauss, g_label),
                          annotate(median, m_label),
                          annotate(bilateral, b_label)])

        # Resize if too wide
        max_w = 1200
        if row.shape[1] > max_w:
            scale = max_w / row.shape[1]
            row = cv2.resize(row, None, fx=scale, fy=scale)

        cv2.imshow(window, row)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyWindow(window)


# Usage
img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
# blur_tuner(img)   # uncomment for desktop
```

---

## 4.11 Semi-Transparent Overlays

`cv2.addWeighted` blends two images. Combined with drawing on a copy, this produces the semi-transparent overlays used in modern CV visualization:

```python
import cv2
import numpy as np


def draw_transparent_rect(img, pt1, pt2, color, alpha=0.4,
                           border_color=None, border_thickness=2):
    """
    Draw a semi-transparent filled rectangle.

    alpha=0 → invisible, alpha=1 → fully opaque (covers background).
    """
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if border_color:
        cv2.rectangle(img, pt1, pt2, border_color, border_thickness, cv2.LINE_AA)


def draw_transparent_circle(img, center, radius, color, alpha=0.4):
    overlay = img.copy()
    cv2.circle(overlay, center, radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_transparent_polygon(img, pts, color, alpha=0.4,
                               border_color=None, border_thickness=2):
    """Draw a semi-transparent filled polygon."""
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border_color:
        cv2.polylines(img, [pts], True, border_color, border_thickness, cv2.LINE_AA)


# ── Demo: overlay panel with stats ────────────────────────────────────────────
img = cv2.imread("sample.jpg")
if img is None:
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    # Add some "objects" for demo
    cv2.circle(img, (200, 200), 80, (20, 180, 220), -1)
    cv2.rectangle(img, (400, 150), (580, 320), (200, 80, 20), -1)

annotated = img.copy()

# Sidebar info panel (dark semi-transparent box in top-left)
draw_transparent_rect(annotated, (0, 0), (220, 160), BLACK, alpha=0.6)

stats = [
    "Frame: 0042",
    "Objects: 2 detected",
    "FPS: 29.8",
    "Confidence: 0.91",
    "Model: YOLOv8n",
]
put_multiline_text(annotated, stats, (10, 25), font_scale=0.55,
                   color=WHITE, thickness=1, line_spacing=1.5)

# Highlight a detection with transparent fill
draw_transparent_rect(annotated, (120, 120), (280, 280), GREEN, alpha=0.25,
                       border_color=GREEN, border_thickness=2)
put_text_with_background(annotated, "ball 0.94", (120, 115),
                          font_scale=0.6, bg_color=(0, 150, 0))

# Highlight another detection
draw_transparent_rect(annotated, (400, 150), (580, 320), BLUE, alpha=0.2,
                       border_color=BLUE, border_thickness=2)
put_text_with_background(annotated, "car 0.88", (400, 145),
                          font_scale=0.6, bg_color=(150, 0, 0))

# Crosshair at screen center
h, w = annotated.shape[:2]
cx, cy = w // 2, h // 2
cv2.line(annotated, (cx - 20, cy), (cx + 20, cy), WHITE, 1, cv2.LINE_AA)
cv2.line(annotated, (cx, cy - 20), (cx, cy + 20), WHITE, 1, cv2.LINE_AA)
cv2.circle(annotated, (cx, cy), 15, WHITE, 1, cv2.LINE_AA)
```

---

## 4.12 Jupyter Notebook Display Helper

`cv2.imshow` does not work in Jupyter. Here is a robust display utility for notebooks that works in all environments:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def show(img, title="", figsize=None, cmap=None):
    """
    Display an image in Jupyter — handles BGR→RGB, grayscale, float32.
    Drop-in replacement for cv2.imshow in notebook environments.
    """
    if img is None:
        print(f"[show] Image is None — nothing to display.")
        return

    # Auto figure size based on image dimensions
    if figsize is None:
        h, w = img.shape[:2]
        scale = min(10.0 / max(h, w) * 100, 1.5)
        figsize = (w * scale / 80, h * scale / 80)
        figsize = (max(4, figsize[0]), max(3, figsize[1]))

    fig, ax = plt.subplots(figsize=figsize)

    # Handle different image types
    if img.ndim == 2:
        # Grayscale
        ax.imshow(img, cmap=cmap or "gray", vmin=0,
                   vmax=255 if img.dtype == np.uint8 else 1.0)
    elif img.ndim == 3 and img.shape[2] == 3:
        # Color — convert BGR→RGB if uint8
        if img.dtype == np.uint8:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            # float32 assumed RGB, clamp to [0,1]
            ax.imshow(np.clip(img, 0, 1))
    elif img.ndim == 3 and img.shape[2] == 4:
        # BGRA → RGBA
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        ax.imshow(rgba)
    else:
        ax.imshow(img)

    if title:
        ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    return fig, ax


def show_grid(images, titles=None, nrow=4, figsize=None,
              cmap="gray", suptitle=None):
    """
    Display a list of images as a grid in Jupyter.

    Parameters
    ----------
    images  : list of np.ndarray
    titles  : list of str (optional, same length as images)
    nrow    : images per row
    suptitle: overall title
    """
    n    = len(images)
    ncol = (n + nrow - 1) // nrow
    if figsize is None:
        figsize = (nrow * 3.5, ncol * 3.5)

    fig, axes = plt.subplots(ncol, nrow, figsize=figsize)
    if ncol == 1 and nrow == 1:
        axes = [[axes]]
    elif ncol == 1:
        axes = [axes]
    elif nrow == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]

    for i, img in enumerate(images):
        ax = axes_flat[i]
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0,
                       vmax=255 if img.dtype == np.uint8 else 1.0)
        elif img.ndim == 3 and img.dtype == np.uint8:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(np.clip(img, 0, 1))

        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    if suptitle:
        plt.suptitle(suptitle, fontsize=12, y=1.01)

    plt.tight_layout()
    plt.show()
    return fig
```

---

## 4.13 Complete Project: Annotation Tool

Bringing together drawing, text, mouse callbacks, and keyboard handling into a single, complete interactive annotation tool:

```python
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class Tool(Enum):
    RECTANGLE = "rectangle"
    CIRCLE    = "circle"
    POINT     = "point"
    TEXT      = "text"


@dataclass
class Annotation:
    tool    : Tool
    pts     : List[Tuple[int, int]]
    color   : Tuple[int, int, int]
    label   : str = ""
    thickness: int = 2


class AnnotationTool:
    """
    A complete interactive image annotation tool built with OpenCV.

    Keyboard controls:
        r   — rectangle tool
        c   — circle tool
        p   — point tool
        z   — undo last annotation
        s   — save annotated image
        ESC — exit

    Mouse controls:
        Left drag   — draw current shape
        Right click — cancel in-progress drawing
    """

    TOOL_COLORS = {
        Tool.RECTANGLE : (0,  255,  0),   # green
        Tool.CIRCLE    : (255, 80,  0),   # blue-orange
        Tool.POINT     : (0,  200, 255),  # yellow
        Tool.TEXT      : (255, 255, 0),   # cyan
    }

    def __init__(self, img_bgr: np.ndarray, window_name: str = "Annotation Tool"):
        self.original    = img_bgr.copy()
        self.window_name = window_name
        self.annotations : List[Annotation] = []
        self.current_tool : Tool = Tool.RECTANGLE

        # Drawing state
        self._drawing    = False
        self._start_pt   : Optional[Tuple[int, int]] = None
        self._current_pt : Optional[Tuple[int, int]] = None
        self._display    : np.ndarray = img_bgr.copy()

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_annotation(self, img, ann: Annotation):
        color = ann.color
        if ann.tool == Tool.RECTANGLE and len(ann.pts) == 2:
            cv2.rectangle(img, ann.pts[0], ann.pts[1], color, ann.thickness, cv2.LINE_AA)
        elif ann.tool == Tool.CIRCLE and len(ann.pts) == 2:
            r = int(np.hypot(ann.pts[1][0] - ann.pts[0][0],
                              ann.pts[1][1] - ann.pts[0][1]))
            cv2.circle(img, ann.pts[0], r, color, ann.thickness, cv2.LINE_AA)
        elif ann.tool == Tool.POINT and ann.pts:
            cv2.circle(img, ann.pts[0], 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, ann.pts[0], 6, WHITE,  1, cv2.LINE_AA)
        if ann.label:
            put_text_with_background(img, ann.label,
                                      (ann.pts[0][0], ann.pts[0][1] - 5),
                                      font_scale=0.6, bg_color=color,
                                      text_color=BLACK)

    def _redraw(self):
        """Redraw everything from scratch."""
        self._display = self.original.copy()
        # Committed annotations
        for ann in self.annotations:
            self._draw_annotation(self._display, ann)
        # In-progress shape (live preview)
        if self._drawing and self._start_pt and self._current_pt:
            color = self.TOOL_COLORS[self.current_tool]
            if self.current_tool == Tool.RECTANGLE:
                cv2.rectangle(self._display, self._start_pt,
                               self._current_pt, color, 2, cv2.LINE_AA)
            elif self.current_tool == Tool.CIRCLE:
                r = int(np.hypot(
                    self._current_pt[0] - self._start_pt[0],
                    self._current_pt[1] - self._start_pt[1]))
                cv2.circle(self._display, self._start_pt, r, color, 2, cv2.LINE_AA)

        # HUD: tool name + count
        self._draw_hud()

    def _draw_hud(self):
        hud_lines = [
            f"Tool: {self.current_tool.value.upper()}  [r/c/p]",
            f"Annotations: {len(self.annotations)}",
            "z=undo  s=save  ESC=exit",
        ]
        draw_transparent_rect(self._display, (0, 0), (220, 75), BLACK, alpha=0.55)
        put_multiline_text(self._display, hud_lines, (8, 18),
                           font_scale=0.48, color=WHITE, line_spacing=1.4)

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing   = True
            self._start_pt  = (x, y)
            self._current_pt = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            self._current_pt = (x, y)
            if self._drawing:
                self._redraw()
                cv2.imshow(self.window_name, self._display)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._drawing:
                self._drawing = False
                color = self.TOOL_COLORS[self.current_tool]
                ann = Annotation(
                    tool=self.current_tool,
                    pts=[self._start_pt, (x, y)],
                    color=color,
                    thickness=2,
                )
                # Filter out accidental zero-size clicks
                if (self.current_tool == Tool.POINT or
                        abs(x - self._start_pt[0]) > 3 or
                        abs(y - self._start_pt[1]) > 3):
                    self.annotations.append(ann)

                self._start_pt   = None
                self._current_pt = None
                self._redraw()
                cv2.imshow(self.window_name, self._display)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Cancel in-progress
            self._drawing    = False
            self._start_pt   = None
            self._current_pt = None
            self._redraw()
            cv2.imshow(self.window_name, self._display)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> List[Annotation]:
        """
        Launch the annotation window. Blocks until user exits.
        Returns the list of annotations.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._redraw()
        cv2.imshow(self.window_name, self._display)

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):       # ESC / q — exit
                break
            elif key == ord("r"):           # rectangle tool
                self.current_tool = Tool.RECTANGLE
                self._redraw()
            elif key == ord("c"):           # circle tool
                self.current_tool = Tool.CIRCLE
                self._redraw()
            elif key == ord("p"):           # point tool
                self.current_tool = Tool.POINT
                self._redraw()
            elif key == ord("z"):           # undo
                if self.annotations:
                    self.annotations.pop()
                    self._redraw()
            elif key == ord("s"):           # save
                output_path = "annotated_output.png"
                cv2.imwrite(output_path, self._display)
                print(f"Saved: {output_path}")

            cv2.imshow(self.window_name, self._display)

        cv2.destroyWindow(self.window_name)
        return self.annotations

    def export_labels(self) -> List[Dict]:
        """Export annotations as a list of dicts (YOLO-like format)."""
        h, w = self.original.shape[:2]
        labels = []
        for i, ann in enumerate(self.annotations):
            if ann.tool == Tool.RECTANGLE and len(ann.pts) == 2:
                x1, y1 = ann.pts[0]
                x2, y2 = ann.pts[1]
                x, y = min(x1,x2), min(y1,y2)
                bw, bh = abs(x2-x1), abs(y2-y1)
                labels.append({
                    "id"    : i,
                    "tool"  : ann.tool.value,
                    "bbox"  : [x, y, bw, bh],
                    "bbox_n": [x/w, y/h, bw/w, bh/h],  # normalized
                    "label" : ann.label,
                })
        return labels


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create a test image
    img = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.circle(img,     (200, 200), 80, (20, 180, 220), -1)
    cv2.rectangle(img,  (380, 150), (580, 340), (200, 80, 20), -1)
    cv2.putText(img, "Annotate me!", (150, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

    tool = AnnotationTool(img)
    annotations = tool.run()

    print(f"\nTotal annotations: {len(annotations)}")
    for ann in annotations:
        print(f"  {ann.tool.value}: pts={ann.pts}, color={ann.color}")

    labels = tool.export_labels()
    for lbl in labels:
        print(f"  BBOX (normalized): {lbl['bbox_n']}")
```

---

## 4.14 Summary

This chapter covered the complete OpenCV drawing and UI stack:

**Drawing Primitives:**
- `cv2.line` / `cv2.arrowedLine` — lines with optional arrowheads; use `LINE_AA` for display
- `cv2.rectangle` — outline or filled; combine fill + outline for bordered boxes
- `cv2.circle` / `cv2.ellipse` — circles, rings, arcs, and pie sectors
- `cv2.polylines` / `cv2.fillPoly` — arbitrary polygon outlines and fills
- `cv2.drawMarker` — scientific plot-style markers (cross, diamond, star, etc.)
- All functions are in-place; use `.copy()` to preserve the original

**Text:**
- `cv2.putText` — renders text; origin is the bottom-left of the baseline (not top-left!)
- `cv2.getTextSize` — returns `(width, height), baseline`; essential for centering and boxes
- Always use `LINE_AA` for rendered text; background boxes improve readability dramatically

**Windows and Event Loop:**
- `cv2.namedWindow` + `cv2.WINDOW_NORMAL` for resizable windows
- `cv2.waitKey(1)` in video loops; `waitKey(0)` for static display
- `key & 0xFF` masks to 8 bits for cross-platform compatibility

**Mouse Callbacks:**
- `cv2.setMouseCallback` — registers a 5-parameter callback `(event, x, y, flags, param)`
- 12 event types: click down/up, move, double-click, scroll wheel
- `flags` bitmask: detect modifier keys (Ctrl, Shift, Alt) and held buttons

**Trackbars:**
- `cv2.createTrackbar(name, window, initial, max, callback)` — always attach to an existing window
- `cv2.getTrackbarPos(name, window)` — poll value in event loop (most common pattern)
- Kernel sizes must be odd; use a `make_odd` helper

**Semi-transparent Overlays:**
- `cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)` — the standard alpha blend pattern
- Draw on a copy, then blend back — preserves the original for subsequent redraws

---

## 4.15 Exercises

### Warm-up

**4.1** Draw the following on a 500×500 black canvas and display it: a red filled circle at the center, a yellow rectangle outline whose corners are at ¼ and ¾ of the width/height, a blue diagonal line from top-left to bottom-right, a green triangle using `cv2.fillPoly`, and white text "OpenCV Drawing" centered at the top. Use `LINE_AA` throughout.

**4.2** Write a function `draw_crosshair(img, center, size=20, color=WHITE, thickness=1)` that draws a crosshair (two crossed lines with a circle) at the specified center. Use it to mark 10 random points on a test image.

**4.3** Use `cv2.getTextSize` to write a function `draw_text_box(img, text, center)` that draws text centered exactly at `center` with a tight background box whose padding is 6 pixels on each side. Verify with multiple font scales.

### Core

**4.4** Build a `draw_detection_box(img, bbox, label, confidence, color)` function that draws a professional bounding box annotation: a colored rectangle, a filled label tab in the same color protruding upward from the top-left corner, white text inside the tab showing `"label: conf"`, and a subtle drop shadow (gray rectangle offset by 2px). Test it with 5 overlapping boxes.

**4.5** Implement `draw_heatmap_overlay(img, heatmap, alpha=0.5)` that takes a grayscale heatmap (same size as img), applies the `cv2.COLORMAP_JET` colormap, and blends it semi-transparently over the image. Test with a synthetic Gaussian heatmap centered at the image center.

**4.6** Build a mouse-driven "rubber-band" line drawing tool: left-drag draws a temporary preview line, releasing the button commits it permanently. Right-click undoes the last committed line. The tool should work in a loop with `waitKey(1)` and redraw every frame.

**4.7** Implement a `draw_grid(img, n_rows, n_cols, color=GRAY, thickness=1)` function that divides the image into a regular grid by drawing horizontal and vertical lines. Optionally label each cell with its (row, col) index in small text.

### Challenge

**4.8** Build a **trackbar-based Canny edge detector tuner**: display the original image alongside the Canny edge output. Four trackbars control: `threshold1` (0–300), `threshold2` (0–300), `aperture` (3/5/7 — integer 0–2 mapped), and `L2gradient` (0/1). When the edge image is updated, also display the edge density (fraction of edge pixels) in a text overlay. Smooth the display so it updates in real time.

**4.9** Extend the `AnnotationTool` from Section 4.13 to support: (a) a **freehand line** tool (left-drag paints a polyline of connected segments), (b) **color selection** via number keys 1–5 selecting from a preset palette, (c) an **opacity slider** trackbar that controls the background transparency of all drawn shapes, and (d) export of all annotations as a JSON file.

**4.10** Build a **multi-image comparison viewer**: given a list of images and titles, display them in a single window with a trackbar to scrub between them. When the user left-clicks on the display, draw a vertical crosshair line that follows the mouse (persists after button release), overlaid on all images so you can compare specific columns. Press `c` to clear all crosshairs.

---

## Further Reading

- **OpenCV drawing functions reference:** https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html  
- **OpenCV HighGUI reference:** https://docs.opencv.org/4.x/d7/dfc/group__highgui.html  
- **OpenCV mouse and trackbar tutorial (LearnOpenCV):** https://learnopencv.com/mouse-and-trackbar-in-opencv-gui/  
- **OpenCV HighGUI features overview:** https://opencv.org/opencv-highgui/  
- **cv2.addWeighted documentation:** https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19  

---

*Next up: **Chapter 5 — Geometric Transformations**, where we move from drawing on images to spatially transforming them: resizing, rotating, affine and perspective warps, homography estimation, and image stitching foundations.*
