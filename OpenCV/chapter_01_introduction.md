# Chapter 1: Introduction to Computer Vision & OpenCV

> *"OpenCV's purpose is to help turn 'seeing' into perception."*  
> — Gary Bradski, founder of OpenCV

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain what computer vision is and why it is a hard problem
2. Describe OpenCV's history, architecture, and place in the modern CV ecosystem
3. Choose the correct OpenCV Python package for your use case
4. Install and verify a working OpenCV + Python environment
5. Read, display, and save an image using OpenCV
6. Understand the fundamental data representation — images as NumPy arrays
7. Explain the difference between OpenCV's BGR convention and RGB
8. Describe the OpenCV module structure and know where to find help

---

## 1.1 What Is Computer Vision?

Before writing a single line of OpenCV code, let us build the right mental model of what we are trying to do — and why it is surprisingly hard.

### 1.1.1 The Perception Gap

A human child can look at a photograph and instantly say *"there is a dog sitting on a red couch."* That child has no idea how they did it. The computation happened involuntarily, in roughly 150 milliseconds, across hundreds of millions of neurons refined by 540 million years of visual evolution.

A computer sees something entirely different: a rectangular grid of numbers.

```
[[[ 59  69  47]
  [ 60  70  48]
  [ 61  71  49]
  ...
  [201 193 178]
  [199 191 176]
  [197 189 174]]]
```

**Computer vision** is the field dedicated to bridging this gap — to writing algorithms that can extract *meaning* from these grids of numbers. At its core, every computer vision problem is a function approximation problem:

```
f(pixel_grid) → meaningful_output
```

Where "meaningful output" might be:

- A class label: `"cat"`, `"not-cat"`
- A bounding box: `[x=120, y=45, w=200, h=180]`
- A segmentation mask: a pixel-wise labeling
- A depth map: distance at every pixel
- A 3D pose: the orientation of an object in space
- A set of matched keypoints between two images

### 1.1.2 Why Is Vision Hard?

Computer vision is difficult for reasons that are both obvious and subtle. Understanding them will make you a better practitioner.

**Viewpoint variation.** The same object looks completely different from different angles. A mug photographed from above looks like a circle; from the side, a cylinder.

**Illumination variation.** A face lit from the front versus a face in shadow can differ by orders of magnitude in pixel values, yet it is the same face. Lighting changes can be more dramatic than the object changes you care about.

**Scale variation.** A car occupies 10,000 pixels when driving beside you and 50 pixels when it is a kilometer away. Algorithms must handle this gracefully.

**Deformation.** Cats, humans, and most objects of interest are non-rigid. The same cat can contort into a dozen configurations.

**Occlusion.** Objects hide behind other objects. We often see only parts of what we want to detect.

**Background clutter.** Natural scenes have enormous visual complexity. A leopard on a rock in a forest is an extreme case of camouflage, but all backgrounds present challenges.

**Intra-class variation.** The category "chair" contains thousands of distinct shapes, materials, and colors. Yet we want one algorithm to recognize all of them.

Classical computer vision tackled these challenges through hand-crafted features and geometric reasoning. Modern deep learning tackles them by learning representations from enormous datasets. OpenCV supports **both paradigms** fluently — and knowing when to use each is a core skill this book will develop.

### 1.1.3 The Computer Vision Application Space

Where does computer vision show up in the real world? Almost everywhere:

| Domain | Application | Core CV Problem |
|--------|-------------|-----------------|
| Manufacturing | Defect detection on assembly lines | Anomaly detection, segmentation |
| Agriculture | Crop disease identification | Classification, segmentation |
| Healthcare | Cell counting in microscopy | Detection, counting |
| Autonomous vehicles | Lane detection, pedestrian detection | Segmentation, real-time detection |
| Retail | Cashier-free checkout (e.g., Amazon Go) | Tracking, recognition |
| Security | Face recognition access control | Detection, recognition, verification |
| AR/VR | Environment understanding, hand tracking | Pose estimation, depth |
| Photography | Portrait mode blur (bokeh effect) | Segmentation, depth |
| Satellite imagery | Building detection, deforestation tracking | Segmentation, change detection |
| Sports | Player tracking, ball tracking, pose analysis | Tracking, pose estimation |

You will be able to build prototypes for all of these domains by the end of this book.

---

## 1.2 A Brief History of OpenCV

Understanding where OpenCV came from gives you context for its design decisions — and explains why certain things work the way they do.

### 1.2.1 The Intel Years (1999–2008)

OpenCV was started at Intel in 1999 by Gary Bradski, and the first release came out in 2000. Vadim Pisarevsky joined Gary Bradski to manage Intel's Russian software OpenCV team.

The motivation was explicitly commercial but publicly beneficial. Gary Bradski wanted to accelerate computer vision research by giving everyone the same infrastructure. As he later explained, the idea was that if you could speed up computer vision by giving researchers a shared foundation, you would drive demand for computing power — which was Intel's core business.

The first alpha version of OpenCV was released to the public at the IEEE Conference on Computer Vision and Pattern Recognition in 2000, and five betas were released between 2001 and 2005. The first 1.0 version was released in 2006.

A defining moment came in 2005. OpenCV was used on Stanley, the vehicle that won the 2005 DARPA Grand Challenge. The DARPA Grand Challenge was a race for autonomous vehicles across 132 miles of desert — Stanley, developed at Stanford, won the $2 million prize and demonstrated that software-based perception was ready for the real world. OpenCV was part of its visual perception stack.

### 1.2.2 The Willow Garage Era and OpenCV 2.x (2008–2012)

After 2008, OpenCV moved into two new homes — companies named Willow Garage and Itseez. Willow Garage was focused on cutting-edge robotics technology, and Itseez created the best-in-class Computer Vision algorithms.

The most important release of this era was OpenCV 2.0. In the 2.0 release, C++ became the primary library language. OpenCV also got automatically generated Python bindings — that is used worldwide since then. This was the architectural pivot that made OpenCV's Python interface what it is today: the Python bindings are auto-generated wrappers around the C++ implementation, which means you get nearly C-speed execution while writing Python code.

### 1.2.3 OpenCV 3.x and the Great Refactoring (2015–2018)

OpenCV 3.0 was a major architectural overhaul. The library was split into two repositories:

- **`opencv`** — the stable core, maintained by the OpenCV team
- **`opencv_contrib`** — cutting-edge and community-contributed algorithms, including some under non-free licenses

This split has a direct consequence for you: if you install `opencv-python` from PyPI, you get the core only. If you install `opencv-contrib-python`, you get both. We will discuss this in Section 1.4.

OpenCV 3.x also greatly expanded deep learning support through the DNN module, introduced support for CUDA-accelerated operations, and launched official Android and iOS support.

### 1.2.4 OpenCV 4.x — The Current Era (2018–present)

OpenCV 4.0 (2018) was another significant leap:

- The old C API (functions like `cvLoadImage`) was fully removed. The C++ API is now the only official interface.
- Optimized kernels through OpenCV's Hardware Abstraction Layer (HAL)
- Massively expanded DNN module supporting ONNX, TensorFlow, Caffe, Darknet and more

The 4.x series has continued to evolve rapidly. As of early 2026, the latest release is 4.13.0.92 (February 2026). Recent highlights include GIF encode/decode, improved PNG and animated PNG handling, animated WebP support, and a new HAL for RISC-V RVV 1.0 platforms in OpenCV 4.12.

A version 5.0 is in active development as of 2025–2026, which will bring further API modernization and expanded deep learning support.

### 1.2.5 The OpenCV Timeline at a Glance

```
1999  Gary Bradski starts OpenCV at Intel
2000  First alpha release at CVPR; first public release
2005  OpenCV used on Stanley (DARPA Grand Challenge winner)
2006  OpenCV 1.0 — stable C API
2009  OpenCV 2.0 — C++ API + auto-generated Python bindings
2011  GPU acceleration via CUDA
2012  OpenCV.org non-profit foundation; mobile (Android/iOS) support
2015  OpenCV 3.0 — opencv / opencv_contrib split; DNN module
2018  OpenCV 4.0 — removed legacy C API; improved DNN
2020  OpenCV 4.3 — ONNX Runtime and TFLite support
2023  OpenCV 4.9 — Python 3.12 support; improved ArUco
2024  OpenCV 4.10/4.11 — expanded DNN, new ONNX layers
2026  OpenCV 4.13 — current stable; OpenCV 5.x in development
```

---

## 1.3 The OpenCV Ecosystem

OpenCV is not a monolith. It is a carefully organized collection of modules. Understanding this structure lets you navigate the documentation efficiently.

### 1.3.1 Core Modules

The following are the modules you will use throughout this book:

| Module | What It Does |
|--------|-------------|
| `core` | Fundamental data structures (`Mat`, `UMat`), basic array ops, math |
| `imgproc` | Image processing: filtering, transforms, histograms, morphology |
| `imgcodecs` | Reading and writing images (PNG, JPEG, TIFF, WebP, GIF, etc.) |
| `highgui` | Display windows, trackbars, mouse callbacks, simple GUI |
| `videoio` | Video capture and writing (webcams, files, RTSP streams) |
| `calib3d` | Camera calibration, stereo, 3D reconstruction, pose estimation |
| `features2d` | Feature detection and matching (ORB, AKAZE, FAST, etc.) |
| `objdetect` | Object detection (Haar cascades, HOG, QR codes, ArUco) |
| `dnn` | Deep neural network inference |
| `video` | Optical flow, background subtraction, tracking |
| `photo` | Computational photography (inpainting, denoising, HDR) |
| `stitching` | Image panorama stitching |
| `ml` | Classical machine learning (SVM, k-NN, decision trees, etc.) |

### 1.3.2 opencv_contrib Modules

These are in the separate `contrib` repository and require `opencv-contrib-python`:

| Module | What It Does |
|--------|-------------|
| `xfeatures2d` | SIFT, SURF (patent-expired but still in contrib), BRIEF |
| `aruco` | ArUco and ChArUco marker detection/pose |
| `tracking` | Advanced trackers: CSRT, KCF, MIL, GOTURN |
| `face` | Face recognition (Eigenfaces, Fisherfaces, LBPH) |
| `text` | Text detection and recognition |
| `bgsegm` | Advanced background subtraction |
| `ximgproc` | Extended image processing (superpixels, edge-preserving filters) |

> **Note on SIFT:** SIFT was patented by the University of British Columbia until March 2020. Since the patent expired, SIFT was moved from `xfeatures2d` back into the main `features2d` module starting with OpenCV 4.4. You no longer need `contrib` just for SIFT. However, it is still good practice to install `opencv-contrib-python` to have the full toolkit available.

### 1.3.3 OpenCV and the Broader Python Ecosystem

OpenCV does not exist in isolation. Understanding how it integrates with other libraries is important:

```
NumPy          ← OpenCV's native array type is np.ndarray. No conversion needed.
Matplotlib     ← Visualization in Jupyter notebooks (better than cv2.imshow for notebooks)
scikit-image   ← Complementary image processing (some algorithms are only there)
Pillow (PIL)   ← Alternative image I/O; easy format conversion
PyTorch        ← Model training; export to ONNX → load in cv2.dnn
TensorFlow     ← Same pattern: train → export → cv2.dnn
MediaPipe      ← High-level pipelines for face, hand, pose (wraps TFLite)
Ultralytics    ← YOLOv8/v11 object detection; ONNX export works with cv2.dnn
```

The key architectural insight: **OpenCV images are just NumPy arrays.** This is the single most important fact about OpenCV's Python interface. Everything in the NumPy ecosystem — fancy indexing, broadcasting, vectorized operations — works directly on OpenCV images. We will exploit this heavily.

---

## 1.4 Installation

### 1.4.1 Choosing the Right Package

Four official packages are available on PyPI:

| Package | GPU/CUDA | Extra Algorithms | Headless | Use When |
|---------|----------|-----------------|----------|----------|
| `opencv-python` | No | No | No | Quick start, most common |
| `opencv-contrib-python` | No | Yes | No | **Recommended default** |
| `opencv-python-headless` | No | No | Yes | Servers, Docker (no GUI) |
| `opencv-contrib-python-headless` | No | Yes | Yes | Servers with full algorithms |

**Recommendation:** Use `opencv-contrib-python` for local development. Use `opencv-contrib-python-headless` for servers and containers. Never install more than one of these simultaneously — they will conflict.

> **Why not `opencv-python`?** You might think you do not need the extras yet. But algorithms like SIFT, ArUco markers, advanced trackers, and more are in `contrib`. Installing `contrib` costs nothing and saves you from uninstalling and reinstalling later.

### 1.4.2 Environment Setup with Conda (Recommended)

Using a virtual environment is non-negotiable for serious Python work. Conda handles both Python packages and native dependencies cleanly.

```bash
# Create a dedicated environment
conda create -n opencv-book python=3.11 -y
conda activate opencv-book

# Install OpenCV (contrib version)
pip install opencv-contrib-python

# Install the scientific stack
pip install numpy matplotlib scikit-image Pillow jupyterlab ipykernel

# Register the kernel so Jupyter sees it
python -m ipykernel install --user --name opencv-book --display-name "OpenCV Book"

# Launch JupyterLab
jupyter lab
```

### 1.4.3 Environment Setup with pip + venv

```bash
# Create environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install packages
pip install opencv-contrib-python numpy matplotlib scikit-image Pillow jupyterlab
```

### 1.4.4 Verify Your Installation

After installing, run this verification script:

```python
import cv2
import numpy as np
import sys

print("=" * 50)
print(f"Python version : {sys.version}")
print(f"OpenCV version : {cv2.__version__}")
print(f"NumPy version  : {np.__version__}")
print("=" * 50)

# Check build info (shows what backends are available)
build_info = cv2.getBuildInformation()

# Extract key lines
for line in build_info.split("\n"):
    for keyword in ["OpenCV modules", "Python 3", "CUDA", "OpenCL", "LAPACK", "BLAS"]:
        if keyword in line:
            print(line.strip())

# Confirm we have contrib modules
# SIFT should be importable if contrib is installed
sift = cv2.SIFT_create()
print("\nSIFT (contrib check) : OK ✓")

# Confirm numpy interop
img = np.zeros((100, 100, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Color conversion test : {img.shape} → {gray.shape} ✓")
```

**Expected output:**

```
==================================================
Python version : 3.11.x ...
OpenCV version : 4.13.0
NumPy version  : 2.x.x
==================================================
  OpenCV modules: ... imgproc imgcodecs highgui ...
  Python 3: YES (ver 3.11.x ...)
  CUDA: NO
  OpenCL: YES
SIFT (contrib check) : OK ✓
Color conversion test : (100, 100, 3) → (100, 100) ✓
```

If you see `SIFT (contrib check) : OK ✓`, you have a complete installation.

### 1.4.5 Platform-Specific Notes

**macOS (Apple Silicon / M1/M2/M3)**

The PyPI wheels support Apple Silicon natively since OpenCV 4.6. No special steps needed. However, `cv2.imshow()` may behave unexpectedly in some terminal environments — use Jupyter with matplotlib for display instead.

```bash
pip install opencv-contrib-python
```

**Windows**

The PyPI wheels work directly on Windows. If you encounter DLL errors on import, install the latest [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

**Linux (Ubuntu 22.04 / 24.04)**

```bash
# System dependencies for video/display support
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

pip install opencv-contrib-python
```

For headless servers (no display), use `opencv-contrib-python-headless` to avoid X11 dependencies.

**Docker**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-contrib-python-headless numpy matplotlib
```

---

## 1.5 Core Concepts: Images as NumPy Arrays

Before reading/writing images, you need to understand the data structure you will be working with every single day.

### 1.5.1 What Is a Digital Image?

A **grayscale image** is a 2D array of intensity values:

```
shape: (height, width)
dtype: uint8  (values 0–255, most common)
       uint16 (values 0–65535, medical/HDR imaging)
       float32 (values 0.0–1.0, common in deep learning)
```

A **color image** is a 3D array — height × width × channels:

```
shape: (height, width, 3)   # for 3-channel color (BGR in OpenCV)
dtype: uint8                # most common
```

The `3` in `(height, width, 3)` represents the three color channels. In OpenCV (and this is critical), the channel order is **BGR** — Blue, Green, Red — *not* RGB. We will discuss why shortly.

```python
import numpy as np

# Manually construct a small 4x4 grayscale image
gray_image = np.array([
    [0,   50,  100, 150],
    [50,  100, 150, 200],
    [100, 150, 200, 250],
    [150, 200, 250, 255]
], dtype=np.uint8)

print(f"Shape: {gray_image.shape}")   # (4, 4)
print(f"Dtype: {gray_image.dtype}")   # uint8
print(f"Min: {gray_image.min()}")     # 0
print(f"Max: {gray_image.max()}")     # 255
```

A `uint8` (unsigned 8-bit integer) stores values from 0 to 255. This maps directly to display intensity: 0 is black, 255 is white for grayscale, and for color channels, each channel independently ranges 0–255.

### 1.5.2 Understanding Image Coordinates

Image coordinates follow **matrix convention**, not Cartesian convention:

```
             columns (x) →
           ┌─────────────────────┐
    rows   │ (0,0)  (0,1)  (0,2) │
    (y)    │ (1,0)  (1,1)  (1,2) │
     ↓     │ (2,0)  (2,1)  (2,2) │
           └─────────────────────┘
```

- **Row index** = y-coordinate (increases downward)
- **Column index** = x-coordinate (increases rightward)
- **Origin (0, 0)** is the top-left corner

When you access a pixel at position `(x=100, y=50)`, you write `image[50, 100]` — row first, then column.

```python
import numpy as np

# 3x4 image (3 rows, 4 columns) — grayscale
img = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
], dtype=np.uint8)

# Access pixel at row=1, col=2 (which is x=2, y=1)
pixel = img[1, 2]
print(f"Pixel at (x=2, y=1): {pixel}")  # 70

# For color images: img[row, col] returns array of 3 channel values
color_img = np.zeros((3, 4, 3), dtype=np.uint8)
color_img[1, 2] = [100, 150, 200]       # sets B=100, G=150, R=200
print(f"Color pixel at [1,2]: {color_img[1, 2]}")  # [100 150 200]
```

### 1.5.3 The BGR Convention Explained

OpenCV stores color images in **BGR** order (Blue, Green, Red) rather than the more intuitive RGB order. This is a historical artifact from when OpenCV was developed against Windows APIs, which used BGR as their native format.

This matters enormously when:
1. Displaying images with matplotlib (which expects RGB)
2. Passing images to deep learning models (which typically expect RGB)
3. Interpreting channel values manually

```python
import cv2
import numpy as np

# Create a pure red pixel in OpenCV
# In BGR: B=0, G=0, R=255
red_bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)

# This same image in matplotlib would display as BLUE
# because matplotlib interprets the first channel as Red

# To display correctly in matplotlib, convert BGR → RGB
red_rgb = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2RGB)
# Now: R=255, G=0, B=0 ← correct for matplotlib

print(f"BGR pixel: {red_bgr[0,0]}")   # [0 0 255]
print(f"RGB pixel: {red_rgb[0,0]}")   # [255 0 0]
```

> **Memorize this rule:** OpenCV reads and writes BGR. Matplotlib and most deep learning frameworks use RGB. Always convert when switching between them.

---

## 1.6 Your First OpenCV Programs

Now let's write real code. We will build up from the simplest possible program to a complete image inspection tool.

### 1.6.1 Reading and Displaying an Image

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Read an image ─────────────────────────────────────────────────────────
#
# cv2.imread(path, flag)
#   flag = cv2.IMREAD_COLOR       → BGR, 3-channel (default)
#   flag = cv2.IMREAD_GRAYSCALE   → 1-channel, uint8
#   flag = cv2.IMREAD_UNCHANGED   → include alpha channel if present
#
img_bgr = cv2.imread("sample.jpg", cv2.IMREAD_COLOR)

# ── 2. Always check for None ──────────────────────────────────────────────────
#
# imread returns None silently if the file is not found.
# This is a common source of confusing errors.
if img_bgr is None:
    raise FileNotFoundError("Image not found. Check the path.")

# ── 3. Inspect the array ──────────────────────────────────────────────────────
print(f"Shape  : {img_bgr.shape}")   # (height, width, channels)
print(f"Dtype  : {img_bgr.dtype}")   # uint8
print(f"Size   : {img_bgr.size}")    # total number of elements = h * w * c
print(f"Height : {img_bgr.shape[0]} px")
print(f"Width  : {img_bgr.shape[1]} px")
print(f"Channels: {img_bgr.shape[2]}")

# ── 4. Display with matplotlib (correct BGR→RGB conversion) ──────────────────
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title("Image loaded with OpenCV")
plt.axis("off")   # hide axis ticks
plt.tight_layout()
plt.show()
```

### 1.6.2 Reading a Grayscale Image

```python
import cv2
import matplotlib.pyplot as plt

# Load directly as grayscale
img_gray = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)

print(f"Grayscale shape: {img_gray.shape}")   # (height, width) — no channel dimension!
print(f"Pixel range: [{img_gray.min()}, {img_gray.max()}]")

# Display — matplotlib uses 'gray' colormap for 2D arrays
plt.figure(figsize=(8, 6))
plt.imshow(img_gray, cmap="gray")
plt.title("Grayscale image")
plt.colorbar(label="Intensity (0=black, 255=white)")
plt.axis("off")
plt.show()
```

> **Common mistake:** Displaying a grayscale image without `cmap="gray"` gives a green-yellow-red false color. Always pass `cmap="gray"` for single-channel images.

### 1.6.3 Saving an Image

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")

# Save as PNG (lossless)
success = cv2.imwrite("output.png", img)
print(f"PNG saved: {success}")

# Save as JPEG with quality control
# Quality: 0 (worst) to 100 (best), default 95
encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
success = cv2.imwrite("output.jpg", img, encode_params)
print(f"JPEG saved: {success}")

# imwrite returns True on success, False on failure
# It does NOT raise an exception on failure — always check the return value
```

**Supported formats:**
- **PNG** — lossless, supports alpha, preferred for intermediate outputs
- **JPEG** — lossy, small files, use for final output
- **TIFF** — lossless, multi-page, common in scientific/medical imaging
- **WebP** — Google's format, good compression + quality balance
- **BMP** — uncompressed, rarely used
- **GIF** — animated GIF support added in OpenCV 4.11/4.12

### 1.6.4 Using cv2.imshow() (Desktop Only)

In addition to matplotlib, OpenCV has its own display window. This is primarily useful for desktop applications and real-time video (we will use it extensively in the video chapters).

```python
import cv2

img = cv2.imread("sample.jpg")

# Create a named window and display
cv2.imshow("My Image", img)

# waitKey(ms) — wait for a key press
# waitKey(0)  → wait indefinitely
# waitKey(1)  → wait 1 ms (used in video loops)
# Returns the ASCII code of the key pressed
key = cv2.waitKey(0)

if key == ord('q') or key == 27:  # 'q' or ESC
    print("Closing window")

# Always destroy windows after use
cv2.destroyAllWindows()
```

> **Jupyter Note:** `cv2.imshow()` does **not** work reliably inside Jupyter notebooks. Use `matplotlib.pyplot.imshow()` for notebooks. In Chapter 4 (Drawing & UI) we will build a helper function that works in both environments.

### 1.6.5 A Comprehensive Image Inspector

Let us combine everything into a useful utility function you can reuse throughout the book:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def inspect_image(img, title="Image", convert_bgr=True):
    """
    Display an image and print its key properties.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    title : str
        Title shown above the image.
    convert_bgr : bool
        If True, convert BGR→RGB before display.
        Set to False if img is already RGB or grayscale.
    """
    # ── Properties ─────────────────────────────────────────────────────────
    print(f"{'─'*40}")
    print(f"Title    : {title}")
    print(f"Shape    : {img.shape}")
    print(f"Dtype    : {img.dtype}")
    print(f"Min/Max  : {img.min()} / {img.max()}")
    print(f"Mean     : {img.mean():.2f}")
    print(f"Std      : {img.std():.2f}")

    if img.ndim == 3:
        h, w, c = img.shape
        print(f"Size     : {w} x {h} px, {c} channels")
        print(f"Memory   : {img.nbytes / 1024:.1f} KB")
    else:
        h, w = img.shape
        print(f"Size     : {w} x {h} px (grayscale)")
        print(f"Memory   : {img.nbytes / 1024:.1f} KB")
    print(f"{'─'*40}")

    # ── Display ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2 if img.ndim == 3 else 1,
                             figsize=(12, 5) if img.ndim == 3 else (6, 5))

    if img.ndim == 3:
        # Show image
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if convert_bgr else img
        axes[0].imshow(display_img)
        axes[0].set_title(f"{title}\n{w}x{h}, {c}ch, {img.dtype}")
        axes[0].axis("off")

        # Show channel histograms
        colors = ("blue", "green", "red")
        channel_names = ("B", "G", "R") if convert_bgr else ("R", "G", "B")
        for i, (color, name) in enumerate(zip(colors, channel_names)):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            axes[1].plot(hist, color=color, alpha=0.8, label=name)
        axes[1].set_title("Channel Histograms")
        axes[1].set_xlabel("Pixel Intensity")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].set_xlim([0, 256])

    else:
        axes.imshow(img, cmap="gray")
        axes.set_title(f"{title}\n{w}x{h}, grayscale, {img.dtype}")
        axes.axis("off")

    plt.tight_layout()
    plt.show()


# ── Usage ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img = cv2.imread("sample.jpg")
    if img is not None:
        inspect_image(img, title="sample.jpg")
    else:
        # Create a synthetic test image if no file available
        img_synthetic = np.zeros((200, 300, 3), dtype=np.uint8)
        img_synthetic[:, :100] = [255, 0, 0]    # Blue region
        img_synthetic[:, 100:200] = [0, 255, 0]  # Green region
        img_synthetic[:, 200:] = [0, 0, 255]     # Red region
        inspect_image(img_synthetic, title="Synthetic BGR image")
```

### 1.6.6 Creating Images from Scratch

You do not always need to load from disk. Here is how to create images programmatically:

```python
import cv2
import numpy as np

# ── 1. Solid color images ─────────────────────────────────────────────────
# Black image (all zeros)
black = np.zeros((480, 640, 3), dtype=np.uint8)

# White image
white = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Gray image
gray = np.full((480, 640, 3), fill_value=128, dtype=np.uint8)

# Specific color (in BGR)
blue_img = np.zeros((200, 300, 3), dtype=np.uint8)
blue_img[:] = [255, 0, 0]   # Pure blue

# ── 2. Gradient image ────────────────────────────────────────────────────
# Horizontal gradient (left=black, right=white)
gradient = np.zeros((200, 256), dtype=np.uint8)
for x in range(256):
    gradient[:, x] = x

# More efficient using broadcasting:
gradient_fast = np.tile(np.arange(256, dtype=np.uint8), (200, 1))

# ── 3. Checkerboard pattern ───────────────────────────────────────────────
def make_checkerboard(size=8, n_tiles=8):
    """Create a checkerboard image."""
    tile = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    board = np.kron(np.ones((n_tiles//2, n_tiles//2), dtype=np.uint8), tile)
    board = np.kron(board, np.ones((size, size), dtype=np.uint8))
    return board

checker = make_checkerboard(size=32, n_tiles=8)
print(f"Checkerboard shape: {checker.shape}")   # (256, 256)

# ── 4. Clone an image (deep copy) ────────────────────────────────────────
original = cv2.imread("sample.jpg") if cv2.imread("sample.jpg") is not None \
           else np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# IMPORTANT: np.copy() or img.copy() — NOT simple assignment
wrong_copy = original          # same memory → modifying wrong_copy modifies original
right_copy = original.copy()  # independent copy

wrong_copy[0, 0] = [0, 0, 0]
print(f"Original after modifying wrong_copy: {original[0, 0]}")   # Changed!

right_copy[0, 0] = [255, 255, 255]
print(f"Original after modifying right_copy: {original[0, 0]}")   # Unchanged ✓
```

> **Critical pitfall — shallow vs deep copy:** Assignment `b = a` does not copy the image; it creates another reference to the same data. When you modify `b`, you modify `a` too. Always use `img.copy()` to make an independent copy.

---

## 1.7 Understanding Image Memory Layout

This section is slightly advanced but critical for understanding performance and avoiding subtle bugs.

### 1.7.1 C-Contiguous Arrays

NumPy arrays can be stored in different memory orderings. OpenCV **requires C-contiguous arrays** (row-major, also called C-order). When you load an image with `cv2.imread`, you get a C-contiguous array. Certain NumPy operations (like `np.flipud`, some slices) can produce non-contiguous arrays.

```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg") if cv2.imread("sample.jpg") is not None \
      else np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Check contiguity
print(f"C-contiguous: {img.flags['C_CONTIGUOUS']}")  # True (freshly loaded)

# Non-contiguous after certain operations
flipped = np.flipud(img)
print(f"After flipud: {flipped.flags['C_CONTIGUOUS']}")  # May be False

# Fix: use np.ascontiguousarray()
flipped_contiguous = np.ascontiguousarray(flipped)
print(f"After fix: {flipped_contiguous.flags['C_CONTIGUOUS']}")  # True ✓

# OpenCV will often throw cryptic errors with non-contiguous arrays
# When in doubt, wrap with np.ascontiguousarray()
```

### 1.7.2 Memory Size Calculations

Understanding how much memory an image uses is important for pipeline design:

```python
def image_memory_usage(shape, dtype=np.uint8):
    """Calculate memory usage of an image."""
    h, w = shape[:2]
    c = shape[2] if len(shape) == 3 else 1
    bytes_per_pixel = np.dtype(dtype).itemsize
    total_bytes = h * w * c * bytes_per_pixel

    print(f"Dimensions   : {w} x {h} x {c}")
    print(f"Dtype        : {dtype}")
    print(f"Bytes/pixel  : {bytes_per_pixel}")
    print(f"Total memory : {total_bytes:,} bytes")
    print(f"             : {total_bytes / 1024:.1f} KB")
    print(f"             : {total_bytes / 1024**2:.2f} MB")

# A typical webcam frame
image_memory_usage((480, 640, 3), np.uint8)
# 640 x 480 x 3 x 1 byte = 921,600 bytes = 900 KB

# A 4K image
image_memory_usage((2160, 3840, 3), np.uint8)
# 3840 x 2160 x 3 x 1 byte = 24,883,200 bytes ≈ 23.7 MB

# A float32 image (common in deep learning)
image_memory_usage((480, 640, 3), np.float32)
# 4x larger than uint8
```

At 30 fps with a 1080p stream, you are moving `1920 × 1080 × 3 × 30 ≈ 186 MB/s` of raw image data. This is why efficient in-place operations and memory reuse matter in video applications.

---

## 1.8 OpenCV Module Reference & Getting Help

### 1.8.1 The Python API Structure

All OpenCV functions are accessed through the `cv2` module. The naming convention follows the C++ API:

```python
# C++ style:          cv::cvtColor(src, dst, cv::COLOR_BGR2RGB)
# Python equivalent:  cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
#                     (no dst — Python returns new arrays)

# C++ constants use :: namespace
# Python uses . module access
# cv::IMREAD_COLOR  →  cv2.IMREAD_COLOR
```

### 1.8.2 Reading Function Documentation

```python
import cv2

# Method 1: Python help system
help(cv2.imread)

# Method 2: Access the docstring directly
print(cv2.imread.__doc__)

# Method 3: Check function signature
import inspect
# Note: cv2 functions are C extensions; inspect may not show full signature
```

The official documentation is at **https://docs.opencv.org/4.x/**. Always use the `4.x` URL to get the current stable documentation rather than a pinned version.

### 1.8.3 Finding Constants

OpenCV has hundreds of named constants. The best way to find them:

```python
import cv2

# List all constants starting with a prefix
color_constants = [x for x in dir(cv2) if x.startswith("COLOR_BGR")]
print(color_constants[:10])
# ['COLOR_BGR2BGR555', 'COLOR_BGR2BGR565', 'COLOR_BGR2BGRA',
#  'COLOR_BGR2GRAY', 'COLOR_BGR2HLS', ...]

# Find all IMREAD constants
imread_flags = [(x, getattr(cv2, x)) for x in dir(cv2) if x.startswith("IMREAD_")]
for name, val in imread_flags:
    print(f"cv2.{name} = {val}")
```

---

## 1.9 Common Beginner Mistakes

Let us explicitly enumerate the mistakes that trip up almost every OpenCV beginner. Consider this a preemptive debugging guide.

### Mistake 1: Not Checking imread Return Value

```python
# ❌ Wrong — will crash with confusing error downstream
img = cv2.imread("nonexistent.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # NoneType error here

# ✓ Correct
img = cv2.imread("path/to/image.jpg")
assert img is not None, f"Failed to load image"
```

### Mistake 2: Displaying BGR Image with Matplotlib

```python
# ❌ Wrong — colors will look wrong (reds appear blue, blues appear red)
img = cv2.imread("photo.jpg")
plt.imshow(img)

# ✓ Correct — always convert for matplotlib
img = cv2.imread("photo.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

### Mistake 3: Modifying an Image In-Place Unintentionally

```python
# ❌ Wrong — original is destroyed
backup = original
cv2.rectangle(backup, (10, 10), (100, 100), (0, 255, 0), 2)
# backup and original now both have the rectangle

# ✓ Correct
backup = original.copy()
cv2.rectangle(backup, (10, 10), (100, 100), (0, 255, 0), 2)
# original is untouched
```

### Mistake 4: Forgetting That Slicing Creates a View, Not a Copy

```python
# ❌ Wrong — roi is a VIEW into img, not a copy
roi = img[50:150, 100:200]
roi[:] = 0  # This zeros out the same region in img!

# ✓ Correct — explicit copy
roi = img[50:150, 100:200].copy()
roi[:] = 0  # img is unaffected
```

### Mistake 5: Off-by-One in Coordinate Systems

```python
# OpenCV uses (x, y) for drawing functions — column first, row second
# NumPy uses [row, col] — row first
# These are REVERSED

img = np.zeros((100, 200, 3), dtype=np.uint8)

# NumPy access: [row, col] = [y, x]
img[30, 50] = [255, 0, 0]   # row=30, col=50 → y=30, x=50

# OpenCV drawing: (x, y) = (col, row)
cv2.circle(img, center=(50, 30), radius=5, color=(0, 255, 0), thickness=-1)
# center=(x=50, y=30) = (col=50, row=30) — same pixel as above ✓
```

### Mistake 6: Value Overflow with uint8

```python
# ❌ Wrong — values overflow/wrap with uint8 arithmetic
img1 = np.array([[[200, 200, 200]]], dtype=np.uint8)
img2 = np.array([[[100, 100, 100]]], dtype=np.uint8)
result = img1 + img2
print(result)  # [[[ 44  44  44]]]  ← WRONG! 200+100=300, but wraps to 44

# ✓ Correct — use cv2.add() for saturated addition (clips at 255)
result = cv2.add(img1, img2)
print(result)  # [[[255 255 255]]] ← Correctly clamped ✓

# Or cast to int before arithmetic
result = (img1.astype(np.int32) + img2.astype(np.int32)).clip(0, 255).astype(np.uint8)
```

---

## 1.10 Summary

In this chapter, we built the conceptual and practical foundation for everything that follows:

**Computer Vision Fundamentals:**
- Computer vision bridges the gap between pixel grids and semantic meaning
- Core challenges: viewpoint, illumination, scale, deformation, occlusion, background variation
- Classical CV and deep learning are complementary — OpenCV supports both

**OpenCV History & Ecosystem:**
- Founded by Gary Bradski at Intel in 1999; first release in 2000
- Major milestones: DARPA Grand Challenge 2005, Python bindings in 2.0 (2009), DNN module in 3.x (2015), C API removal in 4.0 (2018)
- Current release: 4.13.x (2026); OpenCV 5.x in development
- Install `opencv-contrib-python` for the full toolkit

**The Data Model:**
- Images are NumPy arrays: `(height, width)` for grayscale, `(height, width, 3)` for color
- OpenCV uses **BGR** channel order — convert to RGB for matplotlib and most deep learning frameworks
- Coordinate system: row=y, col=x; origin at top-left
- Always use `.copy()` for independent image copies
- Use `cv2.add()` for saturated arithmetic on uint8 images

**Core Operations Learned:**
- `cv2.imread()` — load an image (check for `None`!)
- `cv2.imwrite()` — save an image
- `cv2.imshow()` + `cv2.waitKey()` — display (desktop only)
- `cv2.cvtColor()` — convert between color spaces
- `np.zeros()`, `np.ones()`, `np.full()` — create images

---

## 1.11 Exercises

### Warm-up

**1.1** Install OpenCV and run the verification script from Section 1.4.4. Report your OpenCV version.

**1.2** Write a script that loads any image and prints: width, height, number of channels, data type, minimum pixel value, maximum pixel value, and total memory usage in MB.

**1.3** Create a `400 × 600` image manually using NumPy with three equal-width vertical bands of pure red, pure green, and pure blue (in BGR). Save it as `bands.png` and display it. Verify the colors are correct.

### Core

**1.4** Write a function `show_channels(img_bgr)` that takes a BGR image and displays it as a 2×2 subplot: the original image (correctly converted to RGB), plus one image for each of the three channels in grayscale. Label each subplot.

**1.5** Demonstrate the uint8 overflow bug. Create a 100×100 white image (`dtype=uint8`, all pixels = 200). Add 100 to every pixel using (a) direct NumPy addition and (b) `cv2.add()`. Display and explain the results.

**1.6** Load an image and create a region of interest (ROI) — a 100×100 crop from the center. Make a copy of the ROI, set all pixels to zero (black), and display the original image alongside the result of replacing the center crop with the zeroed ROI. Demonstrate the difference between using a view and a `.copy()`.

### Challenge

**1.7** Write a function `image_info_card(path)` that takes an image path, loads the image, and produces a single matplotlib figure with: the image itself, individual R/G/B channel histograms, and a text box showing all metadata (width, height, dtype, min, max, mean, std, file size on disk). Make it look polished.

**1.8** Create a `512 × 512` image that displays a smooth 2D gradient — the top-left corner should be black `(0,0,0)`, the top-right should be pure blue `(255,0,0)` in BGR, the bottom-left should be pure green `(0,255,0)`, and the bottom-right should be pure red `(0,0,255)`. Use NumPy broadcasting — do not use any loops. Hint: use `np.linspace` and broadcasting rules.

**1.9** Write a `safe_imread` function that: (a) raises a descriptive `FileNotFoundError` if the path does not exist, (b) raises a `ValueError` if the loaded array is `None` (can happen with corrupted files), (c) optionally accepts a `target_size=(width, height)` argument and resizes the image using `cv2.resize` if provided, and (d) optionally accepts a `color_space` argument (`"bgr"`, `"rgb"`, `"gray"`) and returns the image in the requested color space. Write unit tests for each behavior.

---

## Further Reading

- **Official OpenCV Python tutorials:** https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html  
- **OpenCV Anniversary and history:** https://opencv.org/anniversary/  
- **"Learning OpenCV 3" by Bradski & Kaehler** — the canonical reference book from the library's creator  
- **NumPy documentation — Array memory layout:** https://numpy.org/doc/stable/reference/arrays.ndarray.html#memory-layout  
- **PyPI page for opencv-contrib-python:** https://pypi.org/project/opencv-contrib-python/  

---

*Next up: **Chapter 2 — Images as Arrays: The NumPy Foundation**, where we go deep on NumPy operations for image manipulation: slicing, masking, broadcasting, and the vectorized operations that make OpenCV fast.*
