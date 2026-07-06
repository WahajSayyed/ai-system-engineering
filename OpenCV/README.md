# 📷 OpenCV with Python: From Beginner to Advanced

> *A structured, hands-on guide to computer vision — written in the spirit of clear, rigorous, and practical machine learning education.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange?logo=jupyter)](https://jupyter.org/)

---

## 🧭 About This Book

This is an open-source, GitHub-hosted book on **Computer Vision using OpenCV and Python** — designed to take you from zero to production-ready. Whether you're a curious beginner running your first pixel operation or a practitioner building real-time video pipelines, this book grows with you.

The philosophy here mirrors that of great ML textbooks: **ground everything in code**, explain the math only when it illuminates, and never skip the *why*. Each chapter is a self-contained Jupyter notebook with runnable code, annotated outputs, and carefully chosen exercises.

> **"The best way to learn computer vision is to build things that see."**

---

## 📚 Table of Contents

### Part I — Foundations

| Chapter | Title | Topics |
|---------|-------|--------|
| [Ch 01](chapters/01_introduction/) | **Introduction to Computer Vision & OpenCV** | What is CV, OpenCV history, ecosystem, installation, first program |
| [Ch 02](chapters/02_image_basics/) | **Images as Arrays — The NumPy Foundation** | Pixels, color channels, dtypes, image I/O, matplotlib display |
| [Ch 03](chapters/03_color_spaces/) | **Color Spaces & Channels** | BGR, RGB, HSV, LAB, grayscale, channel splitting/merging |
| [Ch 04](chapters/04_drawing_annotation/) | **Drawing, Annotation & Basic UI** | Lines, circles, rectangles, text, mouse callbacks, trackbars |
| [Ch 05](chapters/05_image_transformations/) | **Geometric Transformations** | Resize, rotate, flip, affine/perspective transforms, homography |

### Part II — Image Processing

| Chapter | Title | Topics |
|---------|-------|--------|
| [Ch 06](chapters/06_filtering_smoothing/) | **Filtering & Smoothing** | Convolution, blur kernels, Gaussian, median, bilateral filter |
| [Ch 07](chapters/07_thresholding/) | **Thresholding & Binarization** | Global, adaptive, Otsu's method, multi-level thresholding |
| [Ch 08](chapters/08_morphological_ops/) | **Morphological Operations** | Erosion, dilation, opening, closing, gradient, top-hat |
| [Ch 09](chapters/09_edge_gradients/) | **Edge Detection & Gradients** | Sobel, Laplacian, Canny, LoG, gradient magnitude & direction |
| [Ch 10](chapters/10_histograms/) | **Histograms & Histogram Equalization** | Intensity histograms, CLAHE, backprojection, color histograms |
| [Ch 11](chapters/11_contours/) | **Contours & Shape Analysis** | Finding/drawing contours, moments, bounding boxes, convex hull, shape matching |
| [Ch 12](chapters/12_frequency_domain/) | **Frequency Domain Processing** | Fourier Transform (DFT), frequency filtering, phase/magnitude |

### Part III — Feature Engineering & Matching

| Chapter | Title | Topics |
|---------|-------|--------|
| [Ch 13](chapters/13_feature_detection/) | **Corner & Keypoint Detection** | Harris, Shi-Tomasi, FAST, keypoint visualization |
| [Ch 14](chapters/14_feature_descriptors/) | **Feature Descriptors: SIFT, ORB, AKAZE** | SIFT deep-dive, ORB (fast & free), AKAZE, descriptor anatomy |
| [Ch 15](chapters/15_feature_matching/) | **Feature Matching & Homography** | BFMatcher, FLANN, ratio test, RANSAC, image stitching |

### Part IV — Segmentation

| Chapter | Title | Topics |
|---------|-------|--------|
| [Ch 16](chapters/16_classical_segmentation/) | **Classical Segmentation Methods** | Region growing, watershed, GrabCut, mean-shift |
| [Ch 17](chapters/17_kmeans_segmentation/) | **Clustering-Based Segmentation** | K-means on pixels, color quantization, superpixels (SLIC) |

### Part V — Video & Real-Time Processing

| Chapter | Title | Topics |
|---------|-------|--------|
| [Ch 18](chapters/18_video_basics/) | **Working with Video** | VideoCapture, VideoWriter, frame-by-frame processing, codecs |
| [Ch 19](chapters/19_background_subtraction/) | **Background Subtraction & Motion** | MOG2, KNN, optical flow (Farneback, Lucas-Kanade), motion vectors |
| [Ch 20](chapters/20_object_tracking/) | **Object Tracking** | CSRT, KCF, MIL, GOTURN, multi-object tracking, tracker comparison |
| [Ch 21](chapters/21_realtime_pipeline/) | **Building Real-Time Pipelines** | Threading, frame queues, latency optimization, RTSP/IP cameras |

### Part VI — Classical Detection & Recognition

| Chapter | Title | Topics |
|---------|-------|--------|
| [Ch 22](chapters/22_haar_hog/) | **Haar Cascades & HOG Detectors** | Face/eye detection, HOG+SVM for pedestrians, sliding windows |
| [Ch 23](chapters/23_template_matching/) | **Template Matching** | Methods compared, multi-scale matching, NMS, limitations |
| [Ch 24](chapters/24_ocr/) | **OCR with OpenCV & Tesseract** | Preprocessing pipeline, pytesseract, EasyOCR, scene text |

### Part VII — Deep Learning Integration

| Chapter | Title | Topics |
|---------|-------|--------|
| [Ch 25](chapters/25_dnn_module/) | **OpenCV's DNN Module** | Loading ONNX/Caffe/TF models, inference, blob preprocessing |
| [Ch 26](chapters/26_object_detection_dl/) | **Deep Object Detection** | YOLOv8 via ONNX, SSD MobileNet, NMS, real-time inference |
| [Ch 27](chapters/27_semantic_segmentation/) | **Semantic Segmentation** | DeepLab, FCN via DNN module, mask overlay, class colorization |
| [Ch 28](chapters/28_face_analysis/) | **Face Analysis Pipeline** | Detection, alignment, recognition (ArcFace), landmark detection |
| [Ch 29](chapters/29_pose_estimation/) | **Human Pose Estimation** | OpenPose, MediaPipe integration, skeleton drawing, keypoint parsing |

### Part VIII — 3D Vision & Camera Geometry

| Chapter | Title | Topics |
|---------|-------|--------|
| [Ch 30](chapters/30_camera_calibration/) | **Camera Calibration** | Pinhole model, intrinsics/extrinsics, checkerboard, distortion correction |
| [Ch 31](chapters/31_stereo_vision/) | **Stereo Vision & Depth** | Stereo rectification, disparity maps, StereoBM/SGBM, depth estimation |
| [Ch 32](chapters/32_aruco_pose/) | **ArUco Markers & Pose Estimation** | Marker generation, detection, solvePnP, 3D overlay, AR basics |

### Part IX — Production & Performance

| Chapter | Title | Topics |
|---------|-------|--------|
| [Ch 33](chapters/33_performance/) | **Performance Optimization** | Profiling, UMat/OpenCL, CUDA basics, vectorized ops, benchmarking |
| [Ch 34](chapters/34_deployment/) | **Deploying OpenCV Applications** | FastAPI service, Docker container, edge devices (Raspberry Pi, Jetson) |
| [Ch 35](chapters/35_projects/) | **Capstone Projects** | Document scanner, real-time face attendance, lane detection, AR marker app |

---

## 🗂️ Repository Structure

```
opencv-python-book/
│
├── README.md                    ← You are here
├── environment.yml              ← Conda environment
├── requirements.txt             ← pip requirements
├── CONTRIBUTING.md
├── LICENSE
│
├── chapters/
│   ├── 01_introduction/
│   │   ├── notebook.ipynb       ← Main chapter notebook
│   │   ├── README.md            ← Chapter summary & objectives
│   │   └── assets/              ← Images used in notebook
│   ├── 02_image_basics/
│   │   └── ...
│   └── ...
│
├── data/                        ← Shared datasets & sample images
│   ├── images/
│   ├── videos/
│   └── models/
│
└── utils/                       ← Shared Python utilities
    ├── display.py
    ├── download.py
    └── benchmarks.py
```

---

## ⚙️ Setup & Installation

### Option 1 — Conda (Recommended)

```bash
git clone https://github.com/YOUR_USERNAME/opencv-python-book.git
cd opencv-python-book

conda env create -f environment.yml
conda activate opencv-book

jupyter lab
```

### Option 2 — pip + venv

```bash
git clone https://github.com/YOUR_USERNAME/opencv-python-book.git
cd opencv-python-book

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-contrib-python` | ≥ 4.9 | Full OpenCV with extras (SIFT, SURF, etc.) |
| `numpy` | ≥ 1.24 | Array operations |
| `matplotlib` | ≥ 3.8 | Visualization in notebooks |
| `Pillow` | ≥ 10.0 | Image I/O supplement |
| `scikit-image` | ≥ 0.22 | Supplementary image processing |
| `pytesseract` | ≥ 0.3 | OCR interface |
| `ultralytics` | ≥ 8.0 | YOLOv8 |
| `mediapipe` | ≥ 0.10 | Pose/face pipelines |
| `jupyterlab` | ≥ 4.0 | Notebook environment |

> **Note:** Use `opencv-contrib-python` (not `opencv-python`) to get patented algorithms like SIFT and extra modules.

---

## 📖 How to Use This Book

This book is designed to be read **sequentially** for beginners, but each chapter is also self-contained for practitioners who want to jump to a specific topic.

Each chapter notebook follows this structure:

```
1. Learning Objectives          ← What you'll know by the end
2. Theory (concise)             ← The minimum math that matters
3. Code Walkthrough             ← Step-by-step, heavily commented
4. Visual Experiments           ← Tweak parameters, see what changes
5. Common Pitfalls              ← What trips people up
6. Exercises                    ← Graded: Warm-up / Core / Challenge
7. Summary & Further Reading
```

---

## 🧑‍💻 Who Is This For?

| Background | Recommended Starting Point |
|------------|---------------------------|
| Complete beginner to CV | Chapter 1 — start here |
| Know Python, new to OpenCV | Chapter 2 or 3 |
| Know OpenCV basics, want depth | Part III (Feature Engineering) |
| ML practitioner adding CV | Part VII (Deep Learning) |
| Building production systems | Part IX (Performance & Deployment) |

**Prerequisites:** Comfortable with Python (lists, loops, functions). NumPy basics are introduced in Chapter 2 — no prior CV experience required.

---

## 🤝 Contributing

Contributions, corrections, and improvements are warmly welcomed!

- Found a bug or typo? → Open an **Issue**
- Want to add an exercise or improve an explanation? → Open a **Pull Request**
- Have a project idea for Chapter 35? → Start a **Discussion**

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a PR.

---

## 📜 License

This book and all accompanying code are released under the **MIT License** — free to use, share, and build upon with attribution.

See [LICENSE](LICENSE) for full terms.

---

## ✍️ Author & Acknowledgments

Written and maintained with care for learners at every level.

Special thanks to the OpenCV, NumPy, and open-source Python communities whose work makes all of this possible.

---

## 📬 Stay Updated

Star ⭐ this repository to follow along as new chapters are published.

Chapters are released in order. Part I is complete; subsequent parts will be added progressively.

---

*"Code is the best textbook."*
