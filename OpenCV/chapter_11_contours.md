# Chapter 11: Contours & Shape Analysis

> *"A contour is a boundary expressed as a list of points. Everything we know about an object's shape lives in that list."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain what a contour is and find contours using `cv2.findContours`
2. Use all four retrieval modes and two approximation methods correctly
3. Draw contours and understand the hierarchy (parent-child relationships)
4. Compute geometric properties: area, perimeter, centroid, bounding boxes
5. Use image moments to extract shape statistics up to third order
6. Compute convex hull and convexity defects for shape analysis
7. Approximate polygons with `cv2.approxPolyDP` (Douglas-Peucker algorithm)
8. Fit minimum enclosing circles, rotated rectangles, ellipses, and lines
9. Match shapes using Hu moments and `cv2.matchShapes`
10. Build a complete shape classifier that identifies circles, rectangles, triangles, and polygons

---

## 11.1 What Is a Contour?

A **contour** is a curve joining all continuous points along the boundary of a connected region that share the same intensity or color. In practice, you almost always find contours on **binary images** (after thresholding or edge detection).

In OpenCV, a contour is represented as a NumPy array of shape `(N, 1, 2)` where N is the number of points and each point is `[x, y]` (column first, then row — **not** NumPy's `[row, col]` convention).

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_shapes_binary(h=300, w=400):
    """Create a binary image with several distinct shapes."""
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(img,    (70,  80),  50, 255, -1)         # circle
    cv2.rectangle(img, (160, 40), (280, 140), 255, -1)  # rectangle
    pts = np.array([[330, 40], [390, 140], [270, 140]], np.int32)
    cv2.fillPoly(img,  [pts], 255)                       # triangle
    cv2.ellipse(img,   (80, 220), (60, 35), 20, 0, 360, 255, -1)  # ellipse
    pts2 = np.array([[200, 180], [260, 200], [300, 250],  # pentagon
                     [240, 290], [160, 260]], np.int32)
    cv2.fillPoly(img, [pts2], 255)
    # Add a shape with a hole (donut) to demonstrate hierarchy
    cv2.circle(img, (340, 230), 55, 255, -1)   # outer circle
    cv2.circle(img, (340, 230), 30, 0,   -1)   # inner hole
    return img


img_bin = make_shapes_binary()
print(f"Binary image shape: {img_bin.shape}")
print(f"Foreground pixels:  {img_bin.sum() // 255}")
```

---

## 11.2 cv2.findContours: Finding All Contours

```python
import cv2
import numpy as np

img_bin = make_shapes_binary()

# cv2.findContours(image, mode, method, offset=(0,0))
# Returns: contours, hierarchy
#
# image:  binary uint8 (IMPORTANT: findContours may modify the input — pass a copy)
# mode:   retrieval mode — how to organize the contour hierarchy
# method: approximation method — how to store contour points
#
# Returns:
#   contours : list of np.ndarray(N, 1, 2) int32 — one per contour
#   hierarchy: np.ndarray(1, N_contours, 4) — [Next, Prev, FirstChild, Parent]

# ── Retrieval modes ────────────────────────────────────────────────────────────
# cv2.RETR_EXTERNAL  — only outermost contours (no children)
# cv2.RETR_LIST      — all contours, no hierarchy (flat list)
# cv2.RETR_CCOMP     — two-level hierarchy: outer + holes
# cv2.RETR_TREE      — full hierarchy tree (nested at all depths)

# ── Approximation methods ──────────────────────────────────────────────────────
# cv2.CHAIN_APPROX_NONE    — store ALL boundary points (may be thousands for a circle)
# cv2.CHAIN_APPROX_SIMPLE  — compress horizontal/vertical/diagonal runs
#                            (circle: ~32 points instead of hundreds)
# cv2.CHAIN_APPROX_TC89_L1 — Teh-Chin chain approximation
# cv2.CHAIN_APPROX_TC89_KCOS — another Teh-Chin variant

# IMPORTANT: Always pass a copy to protect the original
contours_ext, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_all, _ = cv2.findContours(img_bin.copy(), cv2.RETR_LIST,     cv2.CHAIN_APPROX_SIMPLE)
contours_tree, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(f"RETR_EXTERNAL (outer only):  {len(contours_ext)} contours")
print(f"RETR_LIST (all, flat):       {len(contours_all)} contours")
print(f"RETR_TREE (full hierarchy):  {len(contours_tree)} contours")
# EXTERNAL = 6 (circle, rect, triangle, ellipse, pentagon, outer-donut)
# LIST = 7 (all above + inner hole of donut)
# TREE = same count as LIST, but with parent-child info

print(f"\nFirst contour shape: {contours_ext[0].shape}")   # (N, 1, 2)
print(f"First contour dtype: {contours_ext[0].dtype}")    # int32
print(f"First contour sample points:\n{contours_ext[0][:5]}")
# [[[ x1  y1 ]]
#  [[ x2  y2 ]]
#  ...]
```

### 11.2.1 Retrieval Mode Comparison

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bin = make_shapes_binary()

modes = [
    (cv2.RETR_EXTERNAL, "RETR_EXTERNAL\nOnly outermost"),
    (cv2.RETR_LIST,     "RETR_LIST\nAll, no hierarchy"),
    (cv2.RETR_CCOMP,    "RETR_CCOMP\n2-level hierarchy"),
    (cv2.RETR_TREE,     "RETR_TREE\nFull tree"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (mode, title) in zip(axes, modes):
    contours, _ = cv2.findContours(img_bin.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
    canvas = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    # Color each contour differently
    rng = np.random.default_rng(42)
    for i, cnt in enumerate(contours):
        color = tuple(int(c) for c in rng.integers(80, 255, 3))
        cv2.drawContours(canvas, [cnt], -1, color, 2)

    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{title}\n{len(contours)} contours", fontsize=9)
    ax.axis("off")
plt.suptitle("Contour Retrieval Mode Comparison", fontsize=12)
plt.tight_layout(); plt.show()
```

### 11.2.2 Approximation Methods: Points Stored

```python
import cv2
import numpy as np

img_circle = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(img_circle, (100, 100), 70, 255, -1)

# None: stores all boundary pixels
c_none, _   = cv2.findContours(img_circle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# Simple: compresses runs of collinear points
c_simple, _ = cv2.findContours(img_circle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Circle boundary — CHAIN_APPROX_NONE:   {len(c_none[0])} points")
print(f"Circle boundary — CHAIN_APPROX_SIMPLE: {len(c_simple[0])} points")
# NONE:   ~440 points (every pixel on the boundary)
# SIMPLE: ~32 points  (only direction changes are stored)

img_rect = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img_rect, (30, 30), (170, 170), 255, -1)

c_r_none, _   = cv2.findContours(img_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c_r_simple, _ = cv2.findContours(img_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Rectangle — CHAIN_APPROX_NONE:   {len(c_r_none[0])} points")
print(f"Rectangle — CHAIN_APPROX_SIMPLE: {len(c_r_simple[0])} points")
# NONE:   ~560 points (every pixel along the 4 sides)
# SIMPLE: 4 points   (just the 4 corners)

# Rule: Use CHAIN_APPROX_SIMPLE for almost all use cases.
# Use CHAIN_APPROX_NONE only when you need every single boundary pixel.
```

---

## 11.3 Drawing Contours

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bin = make_shapes_binary()
contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset)
# contourIdx: -1 = all contours; n = only the n-th contour
# thickness:  -1 = fill the contour (filled interior)
# maxLevel:   draw hierarchy levels up to maxLevel (only with hierarchy)

canvas_all   = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
canvas_one   = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
canvas_filled = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
canvas_colored = np.zeros((*img_bin.shape, 3), dtype=np.uint8)

# Draw all contours in green
cv2.drawContours(canvas_all, contours, -1, (0, 255, 0), 2)

# Draw only the first contour in red
cv2.drawContours(canvas_one, contours, 0, (0, 0, 255), 3)

# Draw filled contours
cv2.drawContours(canvas_filled, contours, -1, (0, 180, 0), -1)

# Draw each contour in a different color
rng = np.random.default_rng(42)
for i, cnt in enumerate(contours):
    color = tuple(int(c) for c in rng.integers(80, 255, 3))
    cv2.drawContours(canvas_colored, contours, i, color, 2)
    # Label with index
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(canvas_colored, str(i), (cx-8, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, img, title in zip(axes,
    [canvas_all, canvas_one, canvas_filled, canvas_colored],
    ["All contours", "First contour only", "Filled contours", "Colored by index"]):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## 11.4 Contour Hierarchy

The `hierarchy` array from `cv2.findContours` with `RETR_TREE` encodes the parent-child structure:

```python
import cv2
import numpy as np

img_bin = make_shapes_binary()
contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# hierarchy shape: (1, N_contours, 4)
# hierarchy[0][i] = [Next, Previous, FirstChild, Parent]
#   Next:       index of next contour at same level (-1 if none)
#   Previous:   index of previous contour at same level (-1 if none)
#   FirstChild: index of first child contour (-1 if no children)
#   Parent:     index of parent contour (-1 if top-level)

h = hierarchy[0]   # shape: (N, 4)
print(f"Number of contours: {len(contours)}")
print(f"Hierarchy shape: {h.shape}")
print("\nContour hierarchy table:")
print(f"{'Idx':<5} {'Next':<6} {'Prev':<6} {'Child':<7} {'Parent':<7} {'Level'}")
print("─" * 50)

def get_level(h, idx):
    """Get nesting level (0 = top level)."""
    level = 0
    parent = h[idx][3]
    while parent != -1:
        level += 1
        parent = h[parent][3]
    return level

for i, (cnt, hi) in enumerate(zip(contours, h)):
    level = get_level(h, i)
    area  = cv2.contourArea(cnt)
    is_hole = (hi[3] != -1 and level % 2 == 1)   # odd nesting = hole
    print(f"{i:<5} {hi[0]:<6} {hi[1]:<6} {hi[2]:<7} {hi[3]:<7} "
          f"{'  '*level}Level {level} {'(HOLE)' if is_hole else ''} area={area:.0f}")

# For the donut shape:
# outer circle: level=0, has a child (the hole)
# inner hole:   level=1 (child of outer), no children of its own
```

---

## 11.5 Geometric Properties

### 11.5.1 Area, Perimeter, and Basic Metrics

```python
import cv2
import numpy as np

img_bin = make_shapes_binary()
contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    # Area
    area = cv2.contourArea(cnt)        # pixel area (Green's formula)

    # Perimeter (arc length)
    # True: treat as closed curve; False: open curve
    perimeter = cv2.arcLength(cnt, True)

    # Bounding rectangle (axis-aligned, smallest up-right rectangle)
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = w * h

    # Circularity / compactness: ratio of area to square of perimeter
    # Circle has max circularity = 1 (4π·area/perimeter²)
    # Line has circularity near 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Aspect ratio of bounding box
    aspect_ratio = w / h if h > 0 else 0

    # Extent: ratio of contour area to bounding box area
    extent = area / bbox_area if bbox_area > 0 else 0

    # Solidity: ratio of contour area to convex hull area
    hull   = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    print(f"Contour {i}: area={area:.0f}  perim={perimeter:.1f}  "
          f"bbox={w}×{h}  circ={circularity:.3f}  "
          f"ext={extent:.3f}  solid={solidity:.3f}")
```

### 11.5.2 Image Moments

Image moments are weighted averages of pixel coordinates. They encode shape position, size, and geometry:

```python
import cv2
import numpy as np


def explain_moments(cnt):
    """
    Compute and explain all raw moments up to 3rd order,
    plus derived quantities.
    """
    M = cv2.moments(cnt)
    # M is a dict with keys like 'm00', 'm10', 'm01', 'm20', etc.
    # m_pq = Σ_x Σ_y (x^p · y^q · I(x,y))  [raw moments]
    # mu_pq = Σ (x-cx)^p · (y-cy)^q · I   [central moments, translation-invariant]
    # nu_pq = mu_pq / m00^((p+q)/2+1)      [normalized, scale-invariant]

    # m00: zeroth moment = area (for binary images = pixel area)
    area = M["m00"]
    if area == 0:
        return {}

    # Centroid (center of mass)
    cx = M["m10"] / M["m00"]   # x_centroid = m10 / m00
    cy = M["m01"] / M["m00"]   # y_centroid = m01 / m00

    # Second-order central moments → used for orientation
    mu20 = M["mu20"] / M["m00"]   # = variance in x
    mu02 = M["mu02"] / M["m00"]   # = variance in y
    mu11 = M["mu11"] / M["m00"]   # = covariance xy

    # Principal orientation angle (angle of major axis, in radians)
    # This is the angle that makes the cross-moment mu11 = 0
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    theta_deg = np.degrees(theta)

    # Equivalent ellipse axes (semi-major and semi-minor)
    # lambda1, lambda2 = eigenvalues of the covariance matrix
    delta = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)
    lambda1 = (mu20 + mu02 + delta) / 2.0
    lambda2 = (mu20 + mu02 - delta) / 2.0
    semi_major = 2 * np.sqrt(lambda1) if lambda1 > 0 else 0
    semi_minor = 2 * np.sqrt(abs(lambda2)) if lambda2 > 0 else 0

    return {
        "area"      : area,
        "centroid"  : (cx, cy),
        "theta_deg" : theta_deg,
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "eccentricity": np.sqrt(1 - (semi_minor / semi_major)**2)
                        if semi_major > 0 else 0
    }


img_bin = make_shapes_binary()
contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Moment-based shape analysis:")
print("─" * 75)
for i, cnt in enumerate(contours):
    props = explain_moments(cnt)
    if props:
        print(f"Shape {i}: area={props['area']:.0f}, "
              f"center=({props['centroid'][0]:.1f},{props['centroid'][1]:.1f}), "
              f"θ={props['theta_deg']:.1f}°, "
              f"semi-axes=({props['semi_major']:.1f}, {props['semi_minor']:.1f}), "
              f"eccentricity={props['eccentricity']:.3f}")
```

---

## 11.6 Convex Hull and Convexity Defects

### 11.6.1 Convex Hull

The convex hull is the smallest convex polygon that contains all contour points:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bin = make_shapes_binary()
contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

canvas = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    # returnPoints=True: returns (x,y) coordinates of hull vertices
    hull_pts = cv2.convexHull(cnt, returnPoints=True)    # shape: (K, 1, 2)

    # Draw original contour in green
    cv2.drawContours(canvas, [cnt], -1, (0, 200, 0), 1)
    # Draw hull in red
    cv2.drawContours(canvas, [hull_pts], -1, (0, 0, 255), 2)

    # Hull area and solidity
    cnt_area  = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull_pts)
    solidity  = cnt_area / hull_area if hull_area > 0 else 0
    # Low solidity → concave shape (star, hand, letter C)
    # High solidity → convex shape (circle, rectangle)

    cx, cy = int(cv2.moments(cnt)["m10"] / max(cv2.moments(cnt)["m00"], 1)), \
             int(cv2.moments(cnt)["m01"] / max(cv2.moments(cnt)["m00"], 1))
    cv2.putText(canvas, f"S={solidity:.2f}", (cx-20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.title("Green=Contour, Red=Convex Hull, S=Solidity")
plt.axis("off"); plt.tight_layout(); plt.show()
```

### 11.6.2 Convexity Defects

Convexity defects are the "dents" in a contour relative to its convex hull. They are extremely useful for hand gesture recognition (finger gaps) and concave shape analysis:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_convexity_defects(cnt, img_bgr):
    """
    Draw a contour, its convex hull, and all convexity defects.

    Each defect is represented by (start_point, end_point, farthest_point, depth).
    The farthest_point is the deepest "valley" in the contour.
    """
    canvas = img_bgr.copy()

    # Draw contour
    cv2.drawContours(canvas, [cnt], -1, (0, 255, 0), 2)

    # Hull — must pass returnPoints=False to get indices (required for defects)
    hull_idx = cv2.convexHull(cnt, returnPoints=False)   # hull point indices
    hull_pts = cv2.convexHull(cnt, returnPoints=True)    # hull coordinates

    # Draw hull
    cv2.drawContours(canvas, [hull_pts], -1, (0, 0, 255), 2)

    # Convexity defects
    if len(cnt) > 3:
        defects = cv2.convexityDefects(cnt, hull_idx)
        # defects shape: (N_defects, 1, 4)
        # Each row: [start_idx, end_idx, farthest_idx, depth * 256]
        # depth is in units of 1/256 pixels (multiply by 1/256 to get pixels)

        if defects is not None:
            for defect in defects:
                s, e, f, d = defect[0]
                start    = tuple(cnt[s][0])   # start point of defect arc
                end      = tuple(cnt[e][0])   # end point
                farthest = tuple(cnt[f][0])   # deepest valley point
                depth    = d / 256.0          # depth in pixels

                if depth > 5:   # filter trivial defects
                    cv2.line(canvas, start, end, (0, 165, 255), 1)
                    cv2.circle(canvas, farthest, 5, (255, 0, 0), -1)
                    # Label depth
                    cv2.putText(canvas, f"{depth:.0f}px", farthest,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
    return canvas


# Create a hand-like shape with "fingers" (convexity defects between fingers)
hand = np.zeros((300, 300), dtype=np.uint8)
pts = np.array([
    [100, 280], [90, 180],  [70, 160],   # palm left
    [60, 80],   [80, 50],   [100, 80],   # finger 1 (thumb-like)
    [120, 150], [130, 70],  [150, 50],   # finger 2
    [170, 80],  [180, 150], [190, 70],   # finger 3
    [210, 50],  [230, 80],  [240, 150],  # finger 4
    [250, 70],  [270, 50],  [285, 80],   # finger 5
    [290, 160], [290, 280]               # palm right
], dtype=np.int32)
cv2.fillPoly(hand, [pts], 255)

contours, _ = cv2.findContours(hand.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

canvas = cv2.cvtColor(hand, cv2.COLOR_GRAY2BGR)
result = visualize_convexity_defects(cnt, canvas)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Convexity Defects\nGreen=Contour, Red=Hull, Blue=Defect points, Orange=Depth")
plt.axis("off"); plt.tight_layout(); plt.show()

# Applications:
# - Count fingers: number of defects with depth > threshold
# - Detect hand gestures: pattern of defect depths and angles
# - Quality control: detect product dents or missing parts
```

---

## 11.7 Polygon Approximation

`cv2.approxPolyDP` implements the **Douglas-Peucker algorithm**: iteratively removes points that are within `epsilon` of the simplified curve until no more can be removed.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# The key parameter: epsilon
# Larger epsilon → fewer points, rougher approximation
# Epsilon is typically set as a fraction of the arc length
# Common rule: epsilon = 0.01 to 0.05 × arcLength(contour, True)

img_bin = make_shapes_binary()
contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(2, len(contours), figsize=(16, 5))
for col, cnt in enumerate(contours):
    perimeter = cv2.arcLength(cnt, True)
    epsilon   = 0.03 * perimeter   # 3% of perimeter
    approx    = cv2.approxPolyDP(cnt, epsilon, True)

    # Draw original vs approximated
    canvas_orig  = np.zeros((300, 400, 3), dtype=np.uint8)
    canvas_approx = canvas_orig.copy()
    cv2.drawContours(canvas_orig,  [cnt],   -1, (0, 255, 0), 2)
    cv2.drawContours(canvas_approx,[approx],-1, (0, 100, 255), 2)
    # Mark approximated vertices
    for pt in approx:
        cv2.circle(canvas_approx, tuple(pt[0]), 4, (255, 0, 0), -1)

    axes[0, col].imshow(cv2.cvtColor(canvas_orig,  cv2.COLOR_BGR2RGB))
    axes[0, col].set_title(f"Original\n{len(cnt)} pts", fontsize=9)
    axes[0, col].axis("off")
    axes[1, col].imshow(cv2.cvtColor(canvas_approx,cv2.COLOR_BGR2RGB))
    axes[1, col].set_title(f"Approx (ε=3%)\n{len(approx)} pts", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle("Polygon Approximation: Original vs Douglas-Peucker", fontsize=12)
plt.tight_layout(); plt.show()

# Key insight: the number of vertices after approximation identifies the shape:
# 3 vertices → triangle
# 4 vertices → quadrilateral (check aspect ratio → square vs rectangle)
# 5 vertices → pentagon
# ~8-12 vertices → roughly octagonal or noisy polygon
# Many vertices → curve/ellipse/circle
```

---

## 11.8 Bounding Shapes: Enclosing Geometries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bin = make_shapes_binary()
contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

canvas = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    # ── Axis-aligned bounding box ─────────────────────────────────────────────
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (255, 0, 0), 1)  # blue

    # ── Minimum area rotated rectangle ───────────────────────────────────────
    # Returns: RotatedRect = (center, (width, height), angle)
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect).astype(np.int32)   # 4 corner points
    cv2.drawContours(canvas, [box], -1, (0, 165, 255), 1)  # orange

    # Access rect properties
    (rx, ry), (rw, rh), angle = rect

    # ── Minimum enclosing circle ──────────────────────────────────────────────
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(canvas, (int(cx), int(cy)), int(radius), (0, 255, 0), 1)  # green

    # ── Fitted ellipse (needs ≥ 5 points) ────────────────────────────────────
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(canvas, ellipse, (0, 255, 255), 1)   # yellow

    # ── Best-fit line ─────────────────────────────────────────────────────────
    # For elongated shapes — fits a line through all contour points
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    # vx, vy: direction vector; x0, y0: point on the line

# Legend
legend_y = 270
for color, label in [
    ((255, 0, 0),    "Blue  = boundingRect"),
    ((0, 165, 255),  "Orange= minAreaRect"),
    ((0, 255, 0),    "Green = minEnclosingCircle"),
    ((0, 255, 255),  "Yellow= fitEllipse"),
]:
    cv2.putText(canvas, label, (5, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    legend_y += 12

plt.figure(figsize=(9, 6))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.title("Bounding Shapes: Rect, RotatedRect, Circle, Ellipse")
plt.axis("off"); plt.tight_layout(); plt.show()

# ── Efficiency comparison ─────────────────────────────────────────────────────
# For the same object, area(original) ≤ area(fitted_ellipse) ≤ area(min_circle) ≤ area(bounding_rect)
# The ratio of contour area to enclosing shape area measures how well the shape fits
```

---

## 11.9 Shape Matching with Hu Moments

Hu moments are seven values derived from normalized central moments that are invariant to translation, rotation, and scale (the seventh is also skew-invariant):

```python
import cv2
import numpy as np

# cv2.HuMoments(moments_dict) → 7×1 array of Hu moments
# cv2.matchShapes(cnt1, cnt2, method, parameter) → similarity score (0 = identical)
#   method: cv2.CONTOURS_MATCH_I1, I2, or I3

img_bin = make_shapes_binary()
contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Compute Hu moments for all shapes
print("Hu Moment Signatures:")
for i, cnt in enumerate(contours):
    M = cv2.moments(cnt)
    hu = cv2.HuMoments(M)   # shape: (7, 1)
    # Log-transform for better numerical comparison (Hu moments span many orders of magnitude)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    print(f"  Shape {i} (area={cv2.contourArea(cnt):.0f}): {hu_log.ravel()}")

# matchShapes: compare two shapes by their Hu moments
# Returns: 0 = identical, larger = more different
# NOTE: matchShapes accepts contours OR images directly
def compare_all_shapes(contours):
    """Build a similarity matrix between all shapes."""
    n = len(contours)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = cv2.matchShapes(
                contours[i], contours[j], cv2.CONTOURS_MATCH_I2, 0.0
            )
    return dist_matrix

dist = compare_all_shapes(contours)
print("\nShape similarity matrix (lower = more similar):")
shape_names = ["circle", "rect", "triangle", "ellipse", "pentagon", "donut"]
names = shape_names[:len(contours)]
for i, row in enumerate(dist):
    formatted = "  ".join(f"{v:.3f}" for v in row)
    print(f"  {names[i]:<10}: [{formatted}]")

# The diagonal should be 0 (shape compared to itself)
# Off-diagonal values show dissimilarity
# Circle vs ellipse should have low distance (similar roundness)
# Triangle vs rectangle should have high distance
```

---

## 11.10 Complete Project: Shape Classifier

A production-quality system that classifies shapes by their geometric properties:

```python
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt


@dataclass
class ShapeProperties:
    """Complete shape analysis record."""
    contour:        np.ndarray
    area:           float
    perimeter:      float
    centroid:       tuple
    bbox:           tuple         # (x, y, w, h)
    circularity:    float
    solidity:       float
    aspect_ratio:   float
    extent:         float
    n_approx_verts: int           # vertices after polygon approximation
    orientation:    float         # degrees
    is_convex:      bool
    label:          str           # classified shape name
    confidence:     float


def analyze_contour(cnt: np.ndarray, epsilon_frac: float = 0.04) -> ShapeProperties:
    """
    Compute all geometric properties for a single contour.
    """
    area      = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Centroid
    M  = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    # Derived metrics
    circularity  = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    hull         = cv2.convexHull(cnt)
    hull_area    = cv2.contourArea(hull)
    solidity     = area / hull_area if hull_area > 0 else 0
    aspect_ratio = float(w) / h if h > 0 else 0
    extent       = area / (w * h) if w * h > 0 else 0

    # Polygon approximation
    epsilon = epsilon_frac * perimeter
    approx  = cv2.approxPolyDP(cnt, epsilon, True)
    n_verts = len(approx)

    # Orientation (from second-order moments)
    mu20 = M.get("mu20", 0) / M["m00"] if M["m00"] != 0 else 0
    mu02 = M.get("mu02", 0) / M["m00"] if M["m00"] != 0 else 0
    mu11 = M.get("mu11", 0) / M["m00"] if M["m00"] != 0 else 0
    orientation = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))

    is_convex = cv2.isContourConvex(approx)

    # Classify shape
    label, confidence = classify_shape(circularity, n_verts, solidity, aspect_ratio)

    return ShapeProperties(
        contour=cnt, area=area, perimeter=perimeter, centroid=(cx, cy),
        bbox=(x, y, w, h), circularity=circularity, solidity=solidity,
        aspect_ratio=aspect_ratio, extent=extent, n_approx_verts=n_verts,
        orientation=orientation, is_convex=is_convex, label=label, confidence=confidence
    )


def classify_shape(circularity: float, n_verts: int,
                   solidity: float, aspect_ratio: float) -> tuple:
    """
    Classify a shape using a rule-based decision tree.
    Returns (label, confidence).
    """
    # Circles: very high circularity
    if circularity > 0.85:
        confidence = min(circularity / 1.0, 1.0)
        return "circle", confidence

    # Ellipses: moderate circularity, convex
    if 0.6 < circularity <= 0.85 and solidity > 0.90:
        return "ellipse", circularity

    # Triangles: 3 vertices
    if n_verts == 3:
        return "triangle", 0.9

    # Squares / rectangles: 4 vertices
    if n_verts == 4:
        # Distinguish square from rectangle by aspect ratio
        if 0.85 < aspect_ratio < 1.15:
            return "square", 0.85
        else:
            return "rectangle", 0.9

    # Pentagons, hexagons, etc.
    if n_verts == 5:
        return "pentagon", 0.85
    if n_verts == 6:
        return "hexagon", 0.85
    if 7 <= n_verts <= 10:
        return "polygon", 0.7

    # Very low solidity → concave or complex shape
    if solidity < 0.6:
        return "concave_shape", 0.6

    # Default fallback
    return f"polygon ({n_verts}v)", 0.5


def detect_and_classify_shapes(
    img_bgr_or_gray: np.ndarray,
    min_area: int = 500,
    visualize: bool = True,
) -> list:
    """
    Full shape detection and classification pipeline.

    Pipeline:
    1. Convert to grayscale if needed
    2. Threshold (Otsu)
    3. Morphological cleanup (opening)
    4. Find contours
    5. Analyze and classify each contour
    6. Visualize

    Returns list of ShapeProperties.
    """
    # ── Preprocessing ─────────────────────────────────────────────────────────
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        display_img = img_bgr_or_gray.copy()
    else:
        gray = img_bgr_or_gray
        display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)

    # ── Find contours ──────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # ── Analyze and classify ───────────────────────────────────────────────────
    shapes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        props = analyze_contour(cnt)
        shapes.append(props)

    # ── Visualization ──────────────────────────────────────────────────────────
    if visualize:
        annotated = display_img.copy()

        COLORS = {
            "circle":        (0,   200,   0),   # green
            "ellipse":       (255, 165,   0),   # orange
            "triangle":      (0,   0,   255),   # red
            "square":        (255, 0,   255),   # magenta
            "rectangle":     (0,   255, 255),   # yellow
            "pentagon":      (255, 100, 100),   # light red
            "hexagon":       (100, 100, 255),   # light blue
        }

        for props in shapes:
            color = COLORS.get(props.label, (200, 200, 200))
            cx, cy = props.centroid

            # Draw contour
            cv2.drawContours(annotated, [props.contour], -1, color, 2)

            # Label with shape name and confidence
            label_str = f"{props.label} ({props.confidence:.0%})"
            # Semi-transparent background for label
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated,
                          (cx - tw//2 - 3, cy - th - 5),
                          (cx + tw//2 + 3, cy + 3),
                          (0, 0, 0), -1)
            cv2.putText(annotated, label_str, (cx - tw//2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Draw centroid
            cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Input"); axes[0].axis("off")
        axes[1].imshow(binary, cmap="gray"); axes[1].set_title("Binary"); axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Classified ({len(shapes)} shapes)"); axes[2].axis("off")
        plt.tight_layout(); plt.show()

    return shapes


# ── Run the classifier ─────────────────────────────────────────────────────────
img_bin = make_shapes_binary()
detected = detect_and_classify_shapes(img_bin, min_area=300)

print(f"\nClassification Results ({len(detected)} shapes):")
print("─" * 65)
for i, props in enumerate(detected):
    print(f"  Shape {i}: {props.label:<15} conf={props.confidence:.0%}  "
          f"area={props.area:.0f}  circ={props.circularity:.3f}  "
          f"solid={props.solidity:.3f}  verts={props.n_approx_verts}")
```

---

## 11.11 Advanced: pointPolygonTest

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# pointPolygonTest finds the distance from a point to a contour
# Returns: positive inside, negative outside, 0 on boundary
# measureDist=True: returns actual signed distance
# measureDist=False: returns +1, 0, or -1 (faster, ~2-3× speedup)

img_bin = make_shapes_binary()
contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]   # use first shape (circle)

# Test specific points
test_points = [(70, 80), (5, 5), (1000, 1000), (70, 30)]
for pt in test_points:
    # Clamp to valid image bounds for safety
    if 0 <= pt[0] < img_bin.shape[1] and 0 <= pt[1] < img_bin.shape[0]:
        dist = cv2.pointPolygonTest(cnt, pt, measureDist=True)
        location = "INSIDE" if dist > 0 else ("ON BOUNDARY" if dist == 0 else "OUTSIDE")
        print(f"  Point {pt}: distance = {dist:.2f} ({location})")

# Create a distance map from one contour
h, w = img_bin.shape
dist_map = np.zeros((h, w), dtype=np.float32)
for y in range(h):
    for x in range(w):
        dist_map[y, x] = cv2.pointPolygonTest(cnt, (x, y), measureDist=True)

# Normalize for display: clamp to [-50, 50] and map to [0, 255]
dist_vis = np.clip(dist_map, -50, 50)
dist_vis = ((dist_vis + 50) / 100 * 255).astype(np.uint8)
dist_color = cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)
# Blue = outside (negative distance), Red = inside (positive distance), Green = on boundary

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(dist_color, cv2.COLOR_BGR2RGB))
plt.title("pointPolygonTest Distance Map\nBlue=outside, Green=boundary, Red=inside")
plt.colorbar(); plt.axis("off"); plt.tight_layout(); plt.show()
```

---

## 11.12 Summary

Contours are the structured boundary representations that bridge raw binary images and shape reasoning.

**Finding contours:** `cv2.findContours` requires binary input, a copy of the image, a retrieval mode (EXTERNAL for isolated shapes, TREE for full hierarchy), and an approximation method (SIMPLE almost always). Returns a list of `(N, 1, 2)` int32 arrays.

**Hierarchy:** `RETR_TREE` gives the parent-child relationship. `hierarchy[0][i] = [Next, Prev, Child, Parent]`. Holes in objects are children of their outer contour.

**Geometric properties:**
- `cv2.contourArea`: Green's formula area
- `cv2.arcLength`: perimeter
- `cv2.moments`: raw, central, and normalized moments → centroid, orientation, eccentricity
- Derived: circularity (4π·area/perim²), solidity (area/hull_area), extent, aspect_ratio

**Shape features:** convex hull via `cv2.convexHull`; defects via `cv2.convexityDefects` (requires `returnPoints=False` hull); polygon approximation via `cv2.approxPolyDP` with epsilon ≈ 3–5% of perimeter.

**Enclosing shapes:** `cv2.boundingRect` (axis-aligned), `cv2.minAreaRect` (rotated), `cv2.minEnclosingCircle`, `cv2.fitEllipse` (needs ≥5 points), `cv2.fitLine`.

**Shape matching:** Hu moments via `cv2.HuMoments` are invariant to translation, rotation, and scale. `cv2.matchShapes` compares contours by Hu moments.

---

## 11.13 Exercises

### Warm-up

**11.1** Create a binary image with 5 different shapes (circle, square, rectangle, triangle, hexagon). Find all external contours and for each print: area, perimeter, circularity, number of approxPolyDP vertices, and label. Verify the labels are correct.

**11.2** Demonstrate the difference between `CHAIN_APPROX_NONE` and `CHAIN_APPROX_SIMPLE` on a circle, a square, and a diagonal line. For each shape, report the number of points stored by each method and compute the compression ratio.

**11.3** Show the contour hierarchy for an image containing: a large outer rectangle, two smaller circles inside it, and one of those circles containing a filled triangle inside a hollow region. Use `RETR_TREE` and print the full hierarchy tree with indentation showing nesting levels.

### Core

**11.4** Implement `filter_contours(contours, min_area, max_area, min_circularity, max_aspect_ratio)` that filters a list of contours by multiple geometric criteria. Apply it to an image with 20 shapes of varying sizes and geometries, filtering to keep only "roughly circular" objects (circularity > 0.7) above a minimum size.

**11.5** Build a `shape_sorter(img_binary)` that finds all contours, computes their area, and returns them sorted by: (a) area descending, (b) circularity descending, (c) centroid x-coordinate ascending. Display the sorted order visually by labeling each contour 1–N in sorted order.

**11.6** Implement a `coin_counter(img_bgr)` function: threshold → find circular contours (circularity > 0.75) → estimate coin radius using `minEnclosingCircle` → classify coin size by relative radius → return count and estimated total value. Test on a synthetic image of circles of three sizes (representing different denominations).

**11.7** Implement `polygon_similarity(cnt1, cnt2)` that compares two polygons using both `cv2.matchShapes` (Hu moments) and a second metric of your choice (Fourier descriptors of the contour, or aspect ratio + vertex count distance). Compare the two metrics on 5 shape pairs and discuss which is more discriminative.

### Challenge

**11.8** Build a **defect detection system** for manufactured parts: create a "perfect" reference shape (e.g., a rectangle with rounded corners). Then create 10 test images: 5 with the correct shape and 5 with deliberate defects (missing corner, extra protrusion, wrong size). Use contour analysis (area, convexity defects, Hu moment matching) to classify each as PASS or FAIL. Report accuracy.

**11.9** Implement **contour-based document layout analysis**: given a scanned document image (synthetic or real), find all text blocks (connected components after morphological closing), filter by area and aspect ratio, draw bounding boxes color-coded by estimated text density (mean intensity within the block), and output an XML-like layout description: `<block x=.. y=.. w=.. h=.. density=../>`.

**11.10** Build a **gesture recognition demo**: create synthetic images of 5 hand gestures (0, 1, 2, 3, 4 fingers extended) using `cv2.fillPoly`. For each, count the number of significant convexity defects (depth > 20px) and use this to predict the finger count. Verify on all 5 gestures and report accuracy. Extend to handle noisy inputs (add morphological closing before analysis).

---

## Further Reading

- **OpenCV contours tutorial:** https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
- **OpenCV contour features:** https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
- **OpenCV contour properties:** https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html
- **Hu, M. K. (1962). "Visual pattern recognition by moment invariants."** IRE Transactions on Information Theory, 8(2), 179–187 — the original Hu moments paper
- **Douglas, D. & Peucker, T. (1973). "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature."** The Canadian Cartographer — the original DP approximation algorithm
- **Sklansky, J. (1982). "Finding the convex hull of a simple polygon."** Pattern Recognition Letters — the convex hull algorithm used in OpenCV
