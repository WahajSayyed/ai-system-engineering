# Chapter 2: Python & the Scientific Stack

> *"NumPy is to Python what a scalpel is to a surgeon — precise, fast, and the tool everything else is built around."*

---

## Table of Contents

1. [Why the Scientific Stack?](#1-why-the-scientific-stack)
2. [NumPy — The Engine of Numerical Computing](#2-numpy--the-engine-of-numerical-computing)
   - 2.1 [Arrays vs Python Lists](#21-arrays-vs-python-lists)
   - 2.2 [Creating Arrays](#22-creating-arrays)
   - 2.3 [Array Attributes and Dtypes](#23-array-attributes-and-dtypes)
   - 2.4 [Indexing and Slicing](#24-indexing-and-slicing)
   - 2.5 [Vectorized Operations](#25-vectorized-operations)
   - 2.6 [Broadcasting](#26-broadcasting)
   - 2.7 [Aggregations and Reductions](#27-aggregations-and-reductions)
   - 2.8 [Reshaping and Stacking](#28-reshaping-and-stacking)
   - 2.9 [Linear Algebra with NumPy](#29-linear-algebra-with-numpy)
   - 2.10 [Random Number Generation](#210-random-number-generation)
3. [Pandas — Data Wrangling at Scale](#3-pandas--data-wrangling-at-scale)
   - 3.1 [Series and DataFrames](#31-series-and-dataframes)
   - 3.2 [Loading Data](#32-loading-data)
   - 3.3 [Inspecting and Profiling](#33-inspecting-and-profiling)
   - 3.4 [Selecting and Filtering](#34-selecting-and-filtering)
   - 3.5 [Handling Missing Values](#35-handling-missing-values)
   - 3.6 [Transforming Columns](#36-transforming-columns)
   - 3.7 [GroupBy and Aggregation](#37-groupby-and-aggregation)
   - 3.8 [Merging and Joining](#38-merging-and-joining)
   - 3.9 [Pandas ↔ NumPy Bridge](#39-pandas--numpy-bridge)
4. [Matplotlib — Visualization Foundations](#4-matplotlib--visualization-foundations)
   - 4.1 [The Figure/Axes Object Model](#41-the-figureaxes-object-model)
   - 4.2 [Essential Plot Types for ML](#42-essential-plot-types-for-ml)
   - 4.3 [Customization and Style](#43-customization-and-style)
5. [Seaborn — Statistical Visualization](#5-seaborn--statistical-visualization)
6. [Scikit-learn — The ML Framework](#6-scikit-learn--the-ml-framework)
   - 6.1 [The Estimator API](#61-the-estimator-api)
   - 6.2 [Transformers and Pipelines](#62-transformers-and-pipelines)
   - 6.3 [The fit/transform/predict Contract](#63-the-fittransformpredict-contract)
7. [Performance: NumPy vs Pure Python](#7-performance-numpy-vs-pure-python)
8. [Putting It All Together: A Complete EDA Workflow](#8-putting-it-all-together-a-complete-eda-workflow)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. Why the Scientific Stack?

Python is a general-purpose language. Out of the box, it is not fast at numerical computation. A pure Python loop over a million numbers is roughly **100× slower** than an equivalent C operation.

The scientific stack solves this by wrapping highly optimized C, C++, and Fortran code in a clean Python interface:

- **NumPy**: Contiguous memory arrays + vectorized C operations. The foundation.
- **Pandas**: Labeled, tabular data structures built on NumPy arrays.
- **Matplotlib**: Low-level, highly configurable 2D plotting library.
- **Seaborn**: Statistical visualization built on Matplotlib.
- **Scikit-learn**: Consistent API for training, evaluating, and deploying ML models.

Understanding these tools deeply is not optional — they are the medium in which all ML work happens. An ML practitioner who doesn't know NumPy well is like a carpenter who doesn't know their chisel.

---

## 2. NumPy — The Engine of Numerical Computing

### 2.1 Arrays vs Python Lists

The fundamental NumPy data structure is the **ndarray** (N-dimensional array). It differs from Python lists in critical ways:

| Property | Python List | NumPy ndarray |
|---|---|---|
| Element types | Mixed (`[1, "a", 3.0]`) | Homogeneous (all same dtype) |
| Memory layout | Scattered pointers | Contiguous block |
| Vectorized ops | No (need loops) | Yes (C speed) |
| Multidimensional | Nested lists (ugly) | Native n-D |
| Speed (numerical) | Slow | ~100× faster |

```python
import numpy as np
import time

# Speed comparison: sum 10 million numbers
n = 10_000_000

# Python list
python_list = list(range(n))
start = time.perf_counter()
total = sum(python_list)
python_time = time.perf_counter() - start

# NumPy array
numpy_array = np.arange(n)
start = time.perf_counter()
total = numpy_array.sum()
numpy_time = time.perf_counter() - start

print(f"Python list sum: {python_time:.4f}s")
print(f"NumPy array sum: {numpy_time:.4f}s")
print(f"Speedup:         {python_time / numpy_time:.1f}×")
# Typical output: Speedup ~50-150×
```

Why is NumPy faster?
1. **Contiguous memory**: All elements stored in a single memory block. Cache-friendly.
2. **No type checking per element**: Every element is the same dtype, so the CPU doesn't check types in a loop.
3. **Vectorized SIMD instructions**: Modern CPUs can process multiple array elements simultaneously (SSE, AVX).
4. **BLAS/LAPACK**: Matrix operations use highly optimized numerical libraries.

---

### 2.2 Creating Arrays

```python
import numpy as np

# ── From Python sequences ──────────────────────────────────────────────────
a = np.array([1, 2, 3, 4, 5])               # 1D array, dtype inferred (int64)
b = np.array([1.0, 2.0, 3.0])               # 1D array, dtype float64
c = np.array([[1, 2, 3],
              [4, 5, 6]])                    # 2D array, shape (2, 3)

# ── Constant arrays ─────────────────────────────────────────────────────────
zeros     = np.zeros((3, 4))                # 3×4 array of 0.0 (float64)
ones      = np.ones((2, 3), dtype=np.int32) # 2×3 array of 1 (int32)
full      = np.full((3, 3), fill_value=7.0) # 3×3 array of 7.0
empty     = np.empty((2, 2))               # Uninitialized (fast, values are garbage)
identity  = np.eye(4)                      # 4×4 identity matrix

# ── Range and linspace ───────────────────────────────────────────────────────
arange  = np.arange(0, 10, 2)             # [0, 2, 4, 6, 8]  (start, stop, step)
linspace = np.linspace(0, 1, 5)           # [0, 0.25, 0.5, 0.75, 1.0] (start, stop, num_points)
logspace = np.logspace(0, 3, 4)           # [1, 10, 100, 1000] — logarithmically spaced

print("arange: ", arange)
print("linspace:", linspace)
print("logspace:", logspace)

# ── Random arrays (we'll cover the RNG in 2.10) ──────────────────────────────
rng = np.random.default_rng(seed=42)       # Reproducible random number generator
rand_uniform  = rng.random((3, 3))         # Uniform [0, 1)
rand_normal   = rng.standard_normal((3, 3))# Standard normal N(0, 1)
rand_integers = rng.integers(0, 10, (3, 3))# Integers in [0, 10)

# ── Special constructions ────────────────────────────────────────────────────
diag_matrix = np.diag([1, 2, 3, 4])       # Diagonal matrix from 1D array
flat_diag   = np.diag(diag_matrix)         # Extract diagonal from 2D array
triu        = np.triu(c)                   # Upper triangular
tril        = np.tril(c)                   # Lower triangular
```

---

### 2.3 Array Attributes and Dtypes

```python
import numpy as np

a = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])

# ── Key attributes ────────────────────────────────────────────────────────────
print(a.shape)    # (2, 3)   — tuple of dimension sizes
print(a.ndim)     # 2        — number of dimensions
print(a.size)     # 6        — total number of elements
print(a.dtype)    # float64  — data type of elements
print(a.itemsize) # 8        — bytes per element (float64 = 8 bytes)
print(a.nbytes)   # 48       — total memory in bytes (6 × 8)

# ── Data types (dtype) ───────────────────────────────────────────────────────
# Choosing the right dtype matters for memory and precision.

# Floating point:
#   float16: 2 bytes, ~3 decimal digits   — used in deep learning (half precision)
#   float32: 4 bytes, ~7 decimal digits   — standard in PyTorch/GPU
#   float64: 8 bytes, ~15 decimal digits  — NumPy default, standard in Scikit-learn

# Integer:
#   int8, int16, int32, int64
#   uint8: [0, 255]  — pixel values, class labels (memory efficient)

# Boolean:
#   bool_: True/False — used in masks

# Type conversions
a_int   = a.astype(np.int32)
a_fp32  = a.astype(np.float32)

print(f"\nOriginal dtype: {a.dtype}, memory: {a.nbytes} bytes")
print(f"float32 dtype:  {a_fp32.dtype}, memory: {a_fp32.nbytes} bytes")
print(f"int32 dtype:    {a_int.dtype}, memory: {a_int.nbytes} bytes")
# float32 uses half the memory of float64 — critical for large datasets and GPU training

# ── Overflow danger ───────────────────────────────────────────────────────────
x = np.array([200], dtype=np.int8)   # int8 max is 127
print(x + 100)   # Wraps around: [-44]  (silent overflow — dangerous!)
```

**Rule of thumb for dtype choice:**
- Scikit-learn: use `float64` (default, expected)
- PyTorch / GPU: use `float32` (4× less memory, GPU-native)
- Labels/indices: `int32` or `int64`
- Pixel values: `uint8`

---

### 2.4 Indexing and Slicing

Mastering indexing is essential — it's how you select data, create masks, and avoid unnecessary copies.

```python
import numpy as np

# ── 1D indexing ───────────────────────────────────────────────────────────────
a = np.array([10, 20, 30, 40, 50, 60, 70, 80])

print(a[0])      # 10     — first element
print(a[-1])     # 80     — last element
print(a[2:5])    # [30 40 50]  — slice [start:stop) (stop is exclusive)
print(a[::2])    # [10 30 50 70] — every 2nd element
print(a[::-1])   # reverse

# ── 2D indexing ───────────────────────────────────────────────────────────────
M = np.array([[1,  2,  3,  4],
              [5,  6,  7,  8],
              [9, 10, 11, 12]])

print(M[0, 0])    # 1     — row 0, col 0
print(M[1, 2])    # 7     — row 1, col 2
print(M[0, :])    # [1 2 3 4]  — entire row 0
print(M[:, 1])    # [2 6 10]   — entire column 1
print(M[0:2, 1:3])# [[2 3]
                   #  [6 7]]   — submatrix

# ── Views vs Copies ───────────────────────────────────────────────────────────
# CRITICAL: NumPy slices return VIEWS, not copies.
# Modifying a view modifies the original array. This saves memory but can cause bugs.

original = np.array([1, 2, 3, 4, 5])
view     = original[1:4]    # NOT a copy — shares memory
view[0]  = 999

print(original)  # [  1 999   3   4   5]  ← original was modified!

# To get an independent copy:
copy = original[1:4].copy()
copy[0] = 0
print(original)  # unchanged

# ── Boolean (mask) indexing ───────────────────────────────────────────────────
# The most powerful indexing pattern in ML — filtering data without loops

data = np.array([3.1, -1.2, 5.5, -0.8, 2.3, -4.1, 6.0])
mask = data > 0                                  # Boolean array: [T F T F T F T]
print(data[mask])                                # [3.1 5.5 2.3 6.0]
print(data[data > 0])                            # Same, inline
print(data[(data > 0) & (data < 4)])             # [3.1 2.3]  — AND condition
print(data[(data < -2) | (data > 5)])            # [-4.1  5.5  6.0] — OR condition

# In ML: selecting samples where a condition holds
X = np.random.default_rng(0).standard_normal((100, 5))  # 100 samples, 5 features
y = np.random.default_rng(0).integers(0, 2, 100)        # Binary labels
X_positive = X[y == 1]   # All samples where label = 1
print(f"Positive class samples: {X_positive.shape}")

# ── Fancy indexing ────────────────────────────────────────────────────────────
# Index with an array of indices. Always returns a COPY.
a = np.array([10, 20, 30, 40, 50])
idx = np.array([0, 2, 4])
print(a[idx])    # [10 30 50]

# Select specific rows from a matrix
M = np.arange(20).reshape(4, 5)
rows = np.array([0, 2, 3])
print(M[rows])   # Rows 0, 2, 3
```

---

### 2.5 Vectorized Operations

The cardinal rule of NumPy: **never loop over array elements when a vectorized operation exists.**

```python
import numpy as np

# ── Element-wise arithmetic ──────────────────────────────────────────────────
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([5.0, 6.0, 7.0, 8.0])

print(a + b)    # [6.  8. 10. 12.]
print(a * b)    # [ 5. 12. 21. 32.]
print(a ** 2)   # [ 1.  4.  9. 16.]
print(a / b)    # [0.2  0.33 0.43 0.5]

# ── Universal functions (ufuncs) ──────────────────────────────────────────────
# These operate element-wise and are highly optimized.
x = np.linspace(0, 2 * np.pi, 6)
print(np.sin(x).round(2))   # [ 0.    0.95  0.59 -0.59 -0.95  0.  ]
print(np.exp(np.array([0, 1, 2])))  # [1.   2.72 7.39]
print(np.log(np.array([1, np.e, np.e**2])))  # [0. 1. 2.]
print(np.abs(np.array([-3, -1, 0, 2])))      # [3 1 0 2]
print(np.sqrt(np.array([1, 4, 9, 16])))      # [1. 2. 3. 4.]

# ── Comparison operations return boolean arrays ──────────────────────────────
a = np.array([1, 5, 3, 8, 2])
print(a > 3)          # [F  T  F  T  F]
print(a == 5)         # [F  T  F  F  F]
print(np.any(a > 7))  # True  — is any element > 7?
print(np.all(a > 0))  # True  — are all elements > 0?

# ── The real power: replace for-loops entirely ───────────────────────────────

# BAD: Python loop to standardize features
def standardize_slow(X):
    result = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            result[i, j] = (X[i, j] - X[:, j].mean()) / X[:, j].std()
    return result

# GOOD: Fully vectorized
def standardize_fast(X):
    mean = X.mean(axis=0)   # Shape (n_features,) — mean of each column
    std  = X.std(axis=0)    # Shape (n_features,) — std of each column
    return (X - mean) / std  # Broadcasting handles the shape mismatch

rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 20))  # 1000 samples, 20 features

import time
start = time.perf_counter(); standardize_slow(X); slow_t = time.perf_counter() - start
start = time.perf_counter(); standardize_fast(X); fast_t = time.perf_counter() - start
print(f"Loop:       {slow_t:.4f}s")
print(f"Vectorized: {fast_t:.6f}s")
print(f"Speedup:    {slow_t/fast_t:.0f}×")
# Typical speedup: 500–2000×
```

---

### 2.6 Broadcasting

Broadcasting is NumPy's mechanism for performing arithmetic between arrays of **different but compatible shapes**. It is one of the most powerful and initially confusing features.

**Broadcasting Rules:**
1. If arrays have different numbers of dimensions, prepend 1s to the shape of the smaller array.
2. Arrays with size 1 along a dimension are "stretched" to match the other array.
3. If sizes don't match and neither is 1, raise an error.

```python
import numpy as np

# ── Scalar broadcast ─────────────────────────────────────────────────────────
a = np.array([1.0, 2.0, 3.0])
print(a * 2)          # [2. 4. 6.]  — scalar broadcast to all elements

# ── 1D row broadcast over 2D matrix ─────────────────────────────────────────
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])              # shape (3, 3)
row = np.array([10, 20, 30])          # shape (3,)  → treated as (1, 3)

print(M + row)
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]

# ── Column broadcast ─────────────────────────────────────────────────────────
col = np.array([[10],
                [20],
                [30]])                # shape (3, 1)
print(M + col)
# [[11 12 13]
#  [24 25 26]
#  [37 38 39]]

# ── Why broadcasting matters in ML ───────────────────────────────────────────
# Standardizing a dataset: subtract column mean, divide by column std
X = np.random.default_rng(0).standard_normal((1000, 4))  # (1000, 4)
mean = X.mean(axis=0)   # (4,)  — one mean per feature
std  = X.std(axis=0)    # (4,)  — one std per feature

# This works because (1000,4) - (4,) broadcasts mean across all rows
X_scaled = (X - mean) / std

print(f"Original  — mean: {X.mean(axis=0).round(2)}, std: {X.std(axis=0).round(2)}")
print(f"Scaled    — mean: {X_scaled.mean(axis=0).round(10)}, std: {X_scaled.std(axis=0).round(2)}")

# ── Outer product via broadcasting ───────────────────────────────────────────
# Useful for computing pairwise distances, kernels, etc.
a = np.array([1, 2, 3])      # shape (3,)
b = np.array([10, 20, 30])   # shape (3,)

# Reshape to column and row vectors
outer = a[:, np.newaxis] * b[np.newaxis, :]  # (3,1) * (1,3) → (3,3)
print(outer)
# [[ 10  20  30]
#  [ 20  40  60]
#  [ 30  60  90]]

# ── Pairwise Euclidean distances (used in KNN, K-Means) ──────────────────────
def pairwise_sq_distances(X, Y):
    """
    Compute squared Euclidean distance between every row of X and every row of Y.
    X: (n, d), Y: (m, d) → output: (n, m)
    Uses the identity: ||x - y||² = ||x||² + ||y||² - 2 x·yᵀ
    """
    XX = (X ** 2).sum(axis=1, keepdims=True)   # (n, 1)
    YY = (Y ** 2).sum(axis=1, keepdims=True).T  # (1, m)
    XY = X @ Y.T                                # (n, m)
    return XX + YY - 2 * XY                     # Broadcasting: (n,1) + (1,m) → (n,m)

rng = np.random.default_rng(0)
A = rng.standard_normal((5, 3))
B = rng.standard_normal((4, 3))
D = pairwise_sq_distances(A, B)
print(f"Distance matrix shape: {D.shape}")   # (5, 4)
```

---

### 2.7 Aggregations and Reductions

```python
import numpy as np

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])

# ── Global aggregations ───────────────────────────────────────────────────────
print(X.sum())       # 45.0  — sum of all elements
print(X.mean())      # 5.0
print(X.std())       # 2.582
print(X.min())       # 1.0
print(X.max())       # 9.0
print(np.median(X))  # 5.0

# ── Axis-wise aggregations ────────────────────────────────────────────────────
# axis=0: collapse rows → result has shape (n_cols,)
# axis=1: collapse cols → result has shape (n_rows,)
print(X.sum(axis=0))    # [12. 15. 18.]  — sum of each column
print(X.sum(axis=1))    # [ 6. 15. 24.]  — sum of each row
print(X.mean(axis=0))   # [4. 5. 6.]     — mean of each column
print(X.max(axis=1))    # [3. 6. 9.]     — max of each row

# ── keepdims: preserve shape for broadcasting ────────────────────────────────
col_mean = X.mean(axis=0, keepdims=True)   # shape (1, 3), not (3,)
print(col_mean.shape)     # (1, 3)
print((X - col_mean))     # Works cleanly — no shape errors

# ── Cumulative operations ─────────────────────────────────────────────────────
a = np.array([1, 2, 3, 4, 5])
print(np.cumsum(a))     # [ 1  3  6 10 15]
print(np.cumprod(a))    # [  1   2   6  24 120]

# ── Sorting and argsort ───────────────────────────────────────────────────────
scores = np.array([0.9, 0.3, 0.7, 0.1, 0.5])
print(np.sort(scores))         # [0.1 0.3 0.5 0.7 0.9]
sorted_idx = np.argsort(scores) # [3 1 4 2 0]  ← indices that would sort the array
print(scores[sorted_idx])      # [0.1 0.3 0.5 0.7 0.9]

# Top-k predictions (used in evaluation):
top2_idx = np.argsort(scores)[::-1][:2]    # Descending, take 2
print(f"Top-2 indices: {top2_idx}")         # [0 2]

# np.argmax and np.argmin
print(np.argmax(scores))  # 0  — index of maximum value
print(np.argmin(scores))  # 3  — index of minimum value

# ── np.where — vectorized if/else ────────────────────────────────────────────
labels = np.array([0.1, 0.7, 0.4, 0.9, 0.3])
binary = np.where(labels >= 0.5, 1, 0)   # threshold at 0.5
print(binary)   # [0 1 0 1 0]
```

---

### 2.8 Reshaping and Stacking

```python
import numpy as np

# ── Reshape ───────────────────────────────────────────────────────────────────
a = np.arange(12)             # [0, 1, 2, ..., 11]
b = a.reshape(3, 4)           # 3 rows × 4 cols
c = a.reshape(2, 2, 3)        # 3D: 2 × 2 × 3
d = a.reshape(-1, 4)          # -1 means "figure out this dimension" → (3, 4)

print(b)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# ── Flatten and ravel ─────────────────────────────────────────────────────────
flat = b.flatten()    # Returns a copy: [0 1 2 ... 11]
ravel = b.ravel()     # Returns a view when possible (more efficient)

# ── Adding dimensions: np.newaxis ─────────────────────────────────────────────
v = np.array([1, 2, 3])         # shape (3,)
col = v[:, np.newaxis]           # shape (3, 1) — column vector
row = v[np.newaxis, :]           # shape (1, 3) — row vector

# Equivalent using None:
col2 = v[:, None]
row2 = v[None, :]

# ── Transpose ─────────────────────────────────────────────────────────────────
M = np.arange(6).reshape(2, 3)  # (2, 3)
print(M.T.shape)                 # (3, 2)
print(M.T)

# ── Stacking ──────────────────────────────────────────────────────────────────
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.vstack([a, b]))          # Vertical stack → (2, 3)
print(np.hstack([a, b]))          # Horizontal stack → (6,)
print(np.stack([a, b], axis=0))   # New axis at 0 → (2, 3)
print(np.stack([a, b], axis=1))   # New axis at 1 → (3, 2)
print(np.concatenate([a, b]))     # Concatenate along existing axis

# ── Splitting ─────────────────────────────────────────────────────────────────
X = np.arange(24).reshape(6, 4)
parts = np.split(X, 3, axis=0)   # Split into 3 equal chunks along rows
print([p.shape for p in parts])   # [(2,4), (2,4), (2,4)]

# ── Practical: add a bias column to a feature matrix ─────────────────────────
X = np.random.default_rng(0).standard_normal((5, 3))  # 5 samples, 3 features
bias = np.ones((5, 1))
X_with_bias = np.hstack([bias, X])   # (5, 4) — first column is all 1s
print(X_with_bias.shape)             # (5, 4)
```

---

### 2.9 Linear Algebra with NumPy

Linear algebra is the mathematical backbone of ML. We'll cover the theory in Chapter 4, but here are the NumPy tools:

```python
import numpy as np

# ── Matrix multiplication ──────────────────────────────────────────────────
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A @ B)        # Matrix multiply (preferred syntax, Python 3.5+)
print(np.dot(A, B)) # Equivalent
# [[19 22]
#  [43 50]]

# ── np.linalg — the linear algebra module ──────────────────────────────────
# Determinant
print(np.linalg.det(A))          # -2.0

# Inverse
A_inv = np.linalg.inv(A)
print(A_inv)
print(A @ A_inv)                  # Should be identity matrix

# Eigenvalues and eigenvectors (used in PCA)
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:",  eigenvalues)
print("Eigenvectors:", eigenvectors)

# Singular Value Decomposition: A = U Σ Vᵀ
# Used in PCA, recommendation systems, text analysis
U, S, Vt = np.linalg.svd(A)
print("U:", U)
print("S (singular values):", S)
print("Vt:", Vt)
# Reconstruct: A = U @ np.diag(S) @ Vt
A_reconstructed = U @ np.diag(S) @ Vt
print("Reconstruction error:", np.allclose(A, A_reconstructed))  # True

# Solving linear systems: Ax = b  →  x = A⁻¹b
b = np.array([1, 2])
x = np.linalg.solve(A, b)        # More numerically stable than inv(A) @ b
print("Solution x:", x)           # [-1.   1. ]
print("Verify Ax = b:", np.allclose(A @ x, b))  # True

# Norms (used in regularization, distance computation)
v = np.array([3.0, 4.0])
print(np.linalg.norm(v))          # 5.0  — L2 (Euclidean) norm
print(np.linalg.norm(v, ord=1))   # 7.0  — L1 (Manhattan) norm
print(np.linalg.norm(v, ord=np.inf)) # 4.0 — L∞ (Max) norm
```

---

### 2.10 Random Number Generation

```python
import numpy as np

# ── ALWAYS use the new Generator API (NumPy 1.17+) ────────────────────────────
# The old np.random.seed() / np.random.randn() API is legacy.
# The new API is statistically better and more explicit.

rng = np.random.default_rng(seed=42)   # Reproducible generator

# Distributions
uniform     = rng.random(5)                  # Uniform [0, 1)
normal      = rng.standard_normal((3, 3))    # N(0, 1)
normal_mv   = rng.normal(loc=5, scale=2, size=10)  # N(μ=5, σ=2)
integers    = rng.integers(0, 10, size=(3, 3))      # Integers [0, 10)
binomial    = rng.binomial(n=10, p=0.3, size=100)   # Binomial
choice      = rng.choice([10, 20, 30, 40], size=5)  # Random sample from array
permutation = rng.permutation(np.arange(10))         # Shuffle

# ── Reproducibility in ML projects ────────────────────────────────────────────
# Always set a global seed at the top of your script.
# For experiments: vary the seed and report mean ± std across multiple runs.

SEED = 42
rng = np.random.default_rng(SEED)

# Train-test split (manual, to illustrate):
n = 100
indices = rng.permutation(n)         # Shuffled indices
train_idx = indices[:80]
test_idx  = indices[80:]

# ── Generating synthetic datasets ─────────────────────────────────────────────
# Useful for testing algorithms before real data is available.

def make_classification_data(n=200, n_features=2, seed=42):
    """Two-class Gaussian blobs."""
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=n // 2)
    X1 = rng.multivariate_normal(mean=[3, 3], cov=[[1, -0.3], [-0.3, 1]], size=n // 2)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    return X, y

X, y = make_classification_data()
print(f"X shape: {X.shape}, y distribution: {np.bincount(y)}")
```

---

## 3. Pandas — Data Wrangling at Scale

### 3.1 Series and DataFrames

Pandas has two core data structures:

- **Series**: A 1D labeled array. Think: a single column with a named index.
- **DataFrame**: A 2D table. Think: a spreadsheet with named rows and columns.

```python
import pandas as pd
import numpy as np

# ── Series ────────────────────────────────────────────────────────────────────
s = pd.Series([10, 20, 30, 40], name='scores')
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# Name: scores, dtype: int64

# Custom index
s_idx = pd.Series(
    [0.92, 0.87, 0.95, 0.78],
    index=['alice', 'bob', 'carol', 'dave'],
    name='accuracy'
)
print(s_idx['alice'])     # 0.92
print(s_idx[['alice', 'carol']])  # Multiple labels

# ── DataFrame ─────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'age':    [25, 32, 28, 45, 38],
    'income': [50000, 80000, 62000, 120000, 95000],
    'target': [0, 1, 0, 1, 1]
})

print(df)
print(f"\nShape: {df.shape}")         # (5, 3)
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()}")  # [0, 1, 2, 3, 4]

# Accessing columns
print(df['age'])             # Series
print(df[['age', 'income']]) # DataFrame (list of column names)

# Accessing rows by integer position
print(df.iloc[0])            # First row as Series
print(df.iloc[0:3])          # First 3 rows

# Accessing by label
print(df.loc[0])             # Row with index label 0
print(df.loc[0:2, 'age':'income'])  # Rows 0-2, columns 'age' to 'income'
```

---

### 3.2 Loading Data

```python
import pandas as pd

# ── CSV ───────────────────────────────────────────────────────────────────────
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv',
    sep=',',               # Delimiter (use '\t' for TSV)
    header=0,              # Row number for column names (None if no header)
    index_col='id',        # Column to use as row index
    usecols=['col1', 'col2', 'target'],   # Load only specific columns
    dtype={'age': np.int32, 'income': np.float32},  # Specify dtypes
    na_values=['NA', 'N/A', '-', '?'],   # Strings to treat as NaN
    nrows=10000,           # Load only first 10k rows (useful for large files)
    parse_dates=['date'],  # Parse date strings as datetime
)

# ── Excel ──────────────────────────────────────────────────────────────────────
df_xl = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# ── JSON ───────────────────────────────────────────────────────────────────────
df_json = pd.read_json('data.json')

# ── From Scikit-learn Datasets ─────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df_cancer = pd.DataFrame(data.data, columns=data.feature_names)
df_cancer['target'] = data.target
print(df_cancer.head())

# ── From a URL ─────────────────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
# df_titanic = pd.read_csv(url)  # Uncomment when online

# ── Saving data ─────────────────────────────────────────────────────────────────
df.to_csv('cleaned_data.csv', index=False)       # Don't write row index
df.to_parquet('data.parquet')                    # Efficient binary format
```

---

### 3.3 Inspecting and Profiling

**Do this first, every time, before any modeling.**

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# ── First look ─────────────────────────────────────────────────────────────────
print(df.shape)          # (569, 31)
print(df.dtypes)         # Data types of each column
print(df.head())         # First 5 rows
print(df.tail())         # Last 5 rows
print(df.sample(5))      # 5 random rows

# ── Data types and memory ──────────────────────────────────────────────────────
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 569 entries, 0 to 568
# Data columns (total 31 columns):
# ...
# dtypes: float64(30), int64(1)
# memory usage: 137.9+ KB

# ── Summary statistics ─────────────────────────────────────────────────────────
print(df.describe())     # count, mean, std, min, 25%, 50%, 75%, max for each column
print(df.describe(include='all'))  # Include object/categorical columns too

# ── Missing values ─────────────────────────────────────────────────────────────
print(df.isnull().sum())                             # NaN count per column
print(df.isnull().sum() / len(df) * 100)             # NaN percentage per column
print(df.isnull().any(axis=1).sum())                 # Rows with at least one NaN

# ── Class distribution (for classification problems) ──────────────────────────
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True))     # As proportions

# ── Unique values ─────────────────────────────────────────────────────────────
for col in df.columns[:5]:
    print(f"{col}: {df[col].nunique()} unique values")

# ── Correlation matrix ─────────────────────────────────────────────────────────
corr = df.corr()
print(corr['target'].sort_values(ascending=False).head(10))  # Features most correlated with target
```

---

### 3.4 Selecting and Filtering

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age':       [25, 32, 28, 45, 38, 22, 55],
    'income':    [50000, 80000, 62000, 120000, 95000, 35000, 140000],
    'education': ['BSc', 'MSc', 'BSc', 'PhD', 'MSc', 'BSc', 'PhD'],
    'target':    [0, 1, 0, 1, 1, 0, 1]
})

# ── Boolean filtering ─────────────────────────────────────────────────────────
print(df[df['age'] > 30])
print(df[(df['age'] > 30) & (df['income'] > 80000)])
print(df[df['education'].isin(['MSc', 'PhD'])])
print(df[df['income'].between(50000, 100000)])

# ── .query() — readable string-based filtering ───────────────────────────────
print(df.query("age > 30 and education == 'PhD'"))
print(df.query("30 < age < 50 and income > 70000"))

# ── .loc and .iloc ────────────────────────────────────────────────────────────
# .loc:  label-based — use column names, index labels
# .iloc: integer-based — use row/col numbers

# Select rows where target==1, only 'age' and 'income' columns
print(df.loc[df['target'] == 1, ['age', 'income']])

# Select first 3 rows, first 2 columns (by position)
print(df.iloc[:3, :2])

# Assign values to a subset
df.loc[df['age'] < 25, 'income'] = 30000    # Conditional assignment

# ── Selecting feature matrix X and target vector y ────────────────────────────
# Standard pattern used before feeding data into any model:
X = df.drop(columns=['target']).values   # NumPy array (n_samples, n_features)
y = df['target'].values                  # NumPy array (n_samples,)
print(X.shape, y.shape)
```

---

### 3.5 Handling Missing Values

Missing data is one of the most common real-world problems. Understanding *why* data is missing matters:

- **MCAR** (Missing Completely At Random): missingness is unrelated to any variable. Rare.
- **MAR** (Missing At Random): missingness depends on observed variables but not on the missing value itself.
- **MNAR** (Missing Not At Random): missingness depends on the missing value itself (e.g., high earners don't report income). Hardest to handle.

```python
import pandas as pd
import numpy as np

# Create a dataset with missing values
df = pd.DataFrame({
    'age':    [25, np.nan, 28, 45, np.nan, 22, 55],
    'income': [50000, 80000, np.nan, 120000, 95000, np.nan, 140000],
    'gender': ['M', 'F', np.nan, 'M', 'F', 'F', np.nan],
    'target': [0, 1, 0, 1, 1, 0, 1]
})

print("Missing values:\n", df.isnull().sum())
print("Missing %:\n", (df.isnull().sum() / len(df) * 100).round(1))

# ── Strategy 1: Drop rows or columns ─────────────────────────────────────────
df_dropped_rows = df.dropna()                        # Drop all rows with any NaN
df_dropped_cols = df.dropna(axis=1)                  # Drop columns with any NaN
df_thresh = df.dropna(thresh=3)                      # Keep rows with at least 3 non-NaN

# Only use this if:
#   - You have plenty of data and the missing rows are few (<5%)
#   - Missingness is MCAR (random)

# ── Strategy 2: Simple imputation ─────────────────────────────────────────────
df_imputed = df.copy()

# Numerical: fill with mean, median, or a constant
df_imputed['age']    = df['age'].fillna(df['age'].median())     # Median more robust to outliers
df_imputed['income'] = df['income'].fillna(df['income'].mean())

# Categorical: fill with mode
df_imputed['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# ── Strategy 3: Forward/backward fill (for time series) ───────────────────────
ts = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0, np.nan])
print(ts.ffill())   # Forward fill: [1, 1, 1, 4, 5, 5]
print(ts.bfill())   # Backward fill: [1, 4, 4, 4, 5, NaN]

# ── Strategy 4: Scikit-learn imputers (preferred for ML pipelines) ─────────────
from sklearn.impute import SimpleImputer, KNNImputer

# SimpleImputer
imp_median = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent', 'constant'
X_imputed  = imp_median.fit_transform(df[['age', 'income']])

# KNNImputer: impute using k nearest neighbors — more sophisticated
knn_imp = KNNImputer(n_neighbors=3)
X_knn   = knn_imp.fit_transform(df[['age', 'income']])

# ── Strategy 5: Add a missing indicator ──────────────────────────────────────
# Often more informative than just imputing — the fact that data is missing
# can itself be a useful signal.
from sklearn.impute import MissingIndicator

indicator = MissingIndicator()
missing_flags = indicator.fit_transform(df[['age', 'income']])
# Returns boolean array indicating which values were originally missing
```

---

### 3.6 Transforming Columns

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name':   ['Alice Johnson', 'Bob Smith', 'Carol White'],
    'salary': ['$85,000', '$120,000', '$67,500'],
    'date':   ['2023-01-15', '2022-06-01', '2024-03-22'],
    'score':  [0.92, 0.78, 0.85]
})

# ── .apply() — apply a function to each element or row/column ─────────────────
# Apply to a column (element-wise)
df['salary_clean'] = df['salary'].str.replace('[$,]', '', regex=True).astype(float)
df['first_name']   = df['name'].apply(lambda x: x.split()[0])
df['score_pct']    = df['score'].apply(lambda x: f"{x*100:.1f}%")

# Apply to rows (axis=1)
df['salary_per_score'] = df.apply(
    lambda row: row['salary_clean'] / row['score'], axis=1
)

# ── .map() — element-wise mapping via dict or function ────────────────────────
grade_map = {'A': 4.0, 'B': 3.0, 'C': 2.0}
grades = pd.Series(['A', 'B', 'A', 'C', 'B'])
print(grades.map(grade_map))   # [4.0, 3.0, 4.0, 2.0, 3.0]

# ── String operations (.str accessor) ─────────────────────────────────────────
text = pd.Series(['  Hello World  ', 'python ML', 'DEEP learning'])
print(text.str.lower())
print(text.str.strip())
print(text.str.replace(' ', '_'))
print(text.str.contains('ML'))     # Boolean mask
print(text.str.split(' '))         # Split into list
print(text.str.len())              # Length of each string

# ── Datetime operations (.dt accessor) ────────────────────────────────────────
df['date'] = pd.to_datetime(df['date'])
print(df['date'].dt.year)
print(df['date'].dt.month)
print(df['date'].dt.day_of_week)   # 0=Monday, 6=Sunday
print(df['date'].dt.days_in_month)

# ── Type conversions ──────────────────────────────────────────────────────────
df['salary_clean'] = df['salary_clean'].astype(np.float32)
df['score']        = pd.to_numeric(df['score'], errors='coerce')  # 'coerce' → NaN on error

# ── pd.cut and pd.qcut — binning continuous values ───────────────────────────
ages = pd.Series([15, 22, 35, 47, 58, 72, 84])
bins = pd.cut(ages, bins=[0, 18, 35, 55, 100],
              labels=['teen', 'young_adult', 'middle_age', 'senior'])
print(bins)

# pd.qcut: quantile-based binning (equal-frequency bins)
qbins = pd.qcut(ages, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print(qbins)
```

---

### 3.7 GroupBy and Aggregation

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing', 'Engineering', 'HR'],
    'gender':     ['M', 'F', 'F', 'M', 'M', 'F', 'F'],
    'salary':     [95000, 72000, 88000, 65000, 78000, 91000, 68000],
    'experience': [5, 3, 7, 2, 4, 6, 3],
    'rating':     [4.2, 3.8, 4.5, 3.5, 4.0, 4.3, 3.7]
})

# ── Basic groupby ─────────────────────────────────────────────────────────────
by_dept = df.groupby('department')

print(by_dept['salary'].mean())
print(by_dept['salary'].agg(['mean', 'std', 'min', 'max']))

# ── Multiple groupby keys ─────────────────────────────────────────────────────
print(df.groupby(['department', 'gender'])['salary'].mean())

# ── Custom aggregation with .agg() ────────────────────────────────────────────
agg_result = by_dept.agg(
    avg_salary      = ('salary', 'mean'),
    max_salary      = ('salary', 'max'),
    avg_experience  = ('experience', 'mean'),
    headcount       = ('salary', 'count'),
    avg_rating      = ('rating', lambda x: x.mean().round(2))
)
print(agg_result)

# ── Transform: same index as original ─────────────────────────────────────────
# Useful for creating group-level features without collapsing the DataFrame
df['dept_avg_salary']  = df.groupby('department')['salary'].transform('mean')
df['salary_vs_dept']   = df['salary'] - df['dept_avg_salary']
df['dept_salary_rank'] = df.groupby('department')['salary'].rank(ascending=False)
print(df[['department', 'salary', 'dept_avg_salary', 'salary_vs_dept', 'dept_salary_rank']])

# ── Filter: keep only groups satisfying a condition ──────────────────────────
big_depts = df.groupby('department').filter(lambda x: len(x) >= 3)
print(big_depts)  # Only Engineering (3 rows) is kept

# ── Value counts and cross-tabulation ─────────────────────────────────────────
print(df['department'].value_counts())
print(pd.crosstab(df['department'], df['gender']))
```

---

### 3.8 Merging and Joining

```python
import pandas as pd

employees = pd.DataFrame({
    'emp_id':   [1, 2, 3, 4],
    'name':     ['Alice', 'Bob', 'Carol', 'Dave'],
    'dept_id':  [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id':   [10, 20, 40],
    'dept_name': ['Engineering', 'Marketing', 'Finance']
})

# ── pd.merge — SQL-style joins ─────────────────────────────────────────────────
# Inner join: only rows with matching keys in BOTH tables
inner = pd.merge(employees, departments, on='dept_id', how='inner')
print(inner)   # Alice, Bob, Carol (Dave's dept 30 not in departments)

# Left join: all rows from left, NaN for unmatched right rows
left = pd.merge(employees, departments, on='dept_id', how='left')
print(left)    # All 4 employees; Dave has NaN for dept_name

# Outer join: all rows from both
outer = pd.merge(employees, departments, on='dept_id', how='outer')
print(outer)   # Includes Finance even though no employee has dept_id=40

# ── Merge on different column names ───────────────────────────────────────────
other = pd.DataFrame({'id': [1, 2, 3], 'score': [0.9, 0.7, 0.8]})
merged = pd.merge(employees, other, left_on='emp_id', right_on='id')

# ── Concatenating DataFrames ──────────────────────────────────────────────────
# Useful for combining train/test splits or appending batches
df1 = employees.iloc[:2]   # First 2 rows
df2 = employees.iloc[2:]   # Last 2 rows
combined = pd.concat([df1, df2], ignore_index=True)   # Reset index

# Concatenate horizontally (side by side)
combined_cols = pd.concat([employees, other.drop('id', axis=1)], axis=1)
```

---

### 3.9 Pandas ↔ NumPy Bridge

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': [7.0, 8.0, 9.0]})

# ── DataFrame to NumPy ─────────────────────────────────────────────────────────
X = df.values                    # NumPy array (no copy if dtypes match)
X = df.to_numpy()                # Preferred method
X = df.to_numpy(dtype=np.float32)

# ── NumPy to DataFrame ─────────────────────────────────────────────────────────
arr = np.random.default_rng(0).standard_normal((5, 3))
df_from_arr = pd.DataFrame(arr, columns=['feature_0', 'feature_1', 'feature_2'])

# ── Extracting X and y for Scikit-learn ────────────────────────────────────────
df_ml = pd.DataFrame({
    'age': [25, 32, 28, 45], 'income': [50, 80, 62, 120],
    'target': [0, 1, 0, 1]
})
X = df_ml.drop(columns='target').to_numpy(dtype=np.float64)
y = df_ml['target'].to_numpy()

# OR — Scikit-learn 0.24+ accepts DataFrames directly
# model.fit(df_ml.drop('target', axis=1), df_ml['target'])
```

---

## 4. Matplotlib — Visualization Foundations

### 4.1 The Figure/Axes Object Model

Matplotlib has two interfaces:
- **PyPlot interface** (`plt.plot(...)`): Quick and stateful. Good for simple plots.
- **Object-oriented interface** (`fig, ax = plt.subplots()`): Explicit and composable. **Always use this for anything non-trivial.**

```python
import matplotlib.pyplot as plt
import numpy as np

# ── Object-oriented interface (preferred) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))   # fig = canvas, ax = plot area

x = np.linspace(0, 2 * np.pi, 100)

ax.plot(x, np.sin(x), color='steelblue', linewidth=2, label='sin(x)')
ax.plot(x, np.cos(x), color='tomato',    linewidth=2, label='cos(x)', linestyle='--')

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('y', fontsize=13)
ax.set_title('Sine and Cosine Functions', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.3, 1.3)
ax.axhline(y=0, color='black', linewidth=0.8)  # Horizontal line at y=0

plt.tight_layout()
plt.savefig('figures/02_sincos.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Multiple subplots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))   # 2 rows, 3 cols

# axes is a 2D array — axes[row, col]
axes[0, 0].set_title('Plot (0,0)')
axes[1, 2].set_title('Plot (1,2)')

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
```

---

### 4.2 Essential Plot Types for ML

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

rng = np.random.default_rng(42)

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1. Line plot — loss curves ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
epochs = np.arange(1, 51)
train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.1 + rng.normal(0, 0.02, 50)
val_loss   = 2.5 * np.exp(-0.06 * epochs) + 0.2 + rng.normal(0, 0.03, 50)
ax1.plot(epochs, train_loss, label='Train loss', color='steelblue')
ax1.plot(epochs, val_loss,   label='Val loss',   color='tomato')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Curves'); ax1.legend(); ax1.grid(alpha=0.3)

# ── 2. Scatter plot — 2D feature space ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
X0 = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 80)
X1 = rng.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], 80)
ax2.scatter(X0[:, 0], X0[:, 1], c='steelblue', alpha=0.6, label='Class 0', s=30)
ax2.scatter(X1[:, 0], X1[:, 1], c='tomato',    alpha=0.6, label='Class 1', s=30)
ax2.set_title('Feature Space'); ax2.legend(); ax2.grid(alpha=0.3)

# ── 3. Histogram — feature distribution ───────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
data = rng.normal(5, 2, 500)
ax3.hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
ax3.axvline(data.mean(), color='tomato', linewidth=2, label=f'Mean={data.mean():.2f}')
ax3.axvline(np.median(data), color='green', linewidth=2, linestyle='--', label=f'Median={np.median(data):.2f}')
ax3.set_title('Feature Distribution'); ax3.legend(); ax3.grid(alpha=0.3)

# ── 4. Bar chart — feature importance ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
features = ['petal_len', 'petal_wid', 'sepal_len', 'sepal_wid']
importances = [0.44, 0.40, 0.10, 0.06]
colors_bar = ['#2196F3' if i < 2 else '#90CAF9' for i in range(4)]
bars = ax4.barh(features, importances, color=colors_bar)
ax4.bar_label(bars, fmt='%.2f', padding=4)
ax4.set_title('Feature Importances'); ax4.set_xlabel('Importance')
ax4.grid(axis='x', alpha=0.3)

# ── 5. Confusion matrix heatmap ───────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
cm = np.array([[45, 3, 2], [2, 48, 0], [1, 1, 48]])
im = ax5.imshow(cm, cmap='Blues', interpolation='nearest')
ax5.set_xticks([0, 1, 2]); ax5.set_yticks([0, 1, 2])
ax5.set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
ax5.set_yticklabels(['Class 0', 'Class 1', 'Class 2'])
for i in range(3):
    for j in range(3):
        ax5.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)
ax5.set_title('Confusion Matrix')
ax5.set_xlabel('Predicted'); ax5.set_ylabel('Actual')
plt.colorbar(im, ax=ax5)

# ── 6. Box plot — comparing distributions ─────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
data_groups = [rng.normal(0, 1, 100), rng.normal(1, 1.5, 100), rng.normal(-0.5, 0.8, 100)]
bp = ax6.boxplot(data_groups, labels=['Model A', 'Model B', 'Model C'],
                 patch_artist=True, notch=True)
colors_box = ['#2196F3', '#FF5722', '#4CAF50']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax6.set_title('CV Score Distributions'); ax6.set_ylabel('Score'); ax6.grid(axis='y', alpha=0.3)

# ── 7. ROC curve ───────────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
from sklearn.metrics import roc_curve, auc
y_true  = rng.integers(0, 2, 200)
y_score = np.clip(y_true + rng.normal(0, 0.5, 200), 0, 1)
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
ax7.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
ax7.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
ax7.set_xlabel('False Positive Rate'); ax7.set_ylabel('True Positive Rate')
ax7.set_title('ROC Curve'); ax7.legend(); ax7.grid(alpha=0.3)

# ── 8. Learning curve ─────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
train_sizes = np.array([50, 100, 200, 400, 600, 800])
train_scores = 1 - 0.8 * np.exp(-train_sizes / 200) + rng.normal(0, 0.01, 6)
test_scores  = 1 - 1.2 * np.exp(-train_sizes / 200) + rng.normal(0, 0.01, 6)
ax8.plot(train_sizes, train_scores, 'o-', color='steelblue', label='Train score')
ax8.plot(train_sizes, test_scores,  's-', color='tomato',    label='Val score')
ax8.fill_between(train_sizes, train_scores - 0.02, train_scores + 0.02, alpha=0.15, color='steelblue')
ax8.fill_between(train_sizes, test_scores  - 0.02, test_scores  + 0.02, alpha=0.15, color='tomato')
ax8.set_xlabel('Training set size'); ax8.set_ylabel('Score')
ax8.set_title('Learning Curve'); ax8.legend(); ax8.grid(alpha=0.3)

# ── 9. Correlation heatmap ─────────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
corr = np.corrcoef(rng.standard_normal((8, 50)))
im2 = ax9.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax9.set_title('Correlation Matrix'); plt.colorbar(im2, ax=ax9)

plt.suptitle('Essential ML Visualization Patterns', fontsize=16, fontweight='bold', y=1.01)
plt.savefig('figures/02_ml_plots.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### 4.3 Customization and Style

```python
import matplotlib.pyplot as plt
import numpy as np

# ── Consistent style for publication/notebook ──────────────────────────────────
plt.rcParams.update({
    'figure.dpi':       100,
    'figure.figsize':   (8, 5),
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.titleweight': 'bold',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'lines.linewidth':  2.0,
    'legend.framealpha':0.9,
})

# ── Colormaps ─────────────────────────────────────────────────────────────────
# For continuous data: 'viridis' (perceptually uniform), 'plasma', 'coolwarm'
# For diverging data: 'RdBu_r', 'seismic'
# For qualitative: 'tab10', 'Set2', 'Paired'

# ── Saving figures with consistent quality ────────────────────────────────────
# fig.savefig('output.png', dpi=150, bbox_inches='tight', facecolor='white')
# fig.savefig('output.pdf', bbox_inches='tight')   # Vector format for papers
```

---

## 5. Seaborn — Statistical Visualization

Seaborn is built on Matplotlib and specializes in statistical plots. It integrates natively with Pandas DataFrames.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Load data as DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

# ── Pairplot — all pairwise relationships ─────────────────────────────────────
# One of the most useful plots for understanding a classification dataset
fig = sns.pairplot(df, hue='species', diag_kind='kde',
                   plot_kws={'alpha': 0.6, 's': 30},
                   palette='Set2')
fig.fig.suptitle('Iris Dataset — Pairplot', y=1.01, fontsize=14, fontweight='bold')
plt.savefig('figures/02_pairplot.png', dpi=120, bbox_inches='tight')
plt.show()

# ── Heatmap — correlation matrix ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = df.drop('species', axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask upper triangle
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, square=True)
ax.set_title('Feature Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/02_correlation_heatmap.png', dpi=150)
plt.show()

# ── Violin plot — distribution by class ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.violinplot(data=df, x='species', y='petal length (cm)',
               palette='Set2', inner='quartile', ax=axes[0])
axes[0].set_title('Petal Length by Species')

sns.boxplot(data=df, x='species', y='sepal width (cm)',
            palette='Set2', ax=axes[1])
sns.stripplot(data=df, x='species', y='sepal width (cm)',
              color='black', alpha=0.3, size=3, ax=axes[1])
axes[1].set_title('Sepal Width by Species')
plt.tight_layout()
plt.savefig('figures/02_violin_box.png', dpi=150)
plt.show()

# ── Distribution plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for species, color in zip(iris.target_names, ['#2196F3', '#FF5722', '#4CAF50']):
    mask = df['species'] == species
    sns.kdeplot(df.loc[mask, 'petal length (cm)'], ax=ax,
                label=species, color=color, fill=True, alpha=0.3)
ax.set_xlabel('Petal Length (cm)')
ax.set_title('KDE — Petal Length by Species')
ax.legend()
plt.tight_layout()
plt.savefig('figures/02_kde.png', dpi=150)
plt.show()

# ── Regression plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sns.regplot(data=df, x='petal length (cm)', y='petal width (cm)',
            scatter_kws={'alpha': 0.4}, line_kws={'color': 'tomato'}, ax=ax)
ax.set_title('Petal Length vs. Petal Width (with regression line)')
plt.tight_layout()
plt.savefig('figures/02_regplot.png', dpi=150)
plt.show()
```

---

## 6. Scikit-learn — The ML Framework

### 6.1 The Estimator API

Scikit-learn's most important design decision: **every object follows the same API.**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# All estimators follow: Estimator(hyperparameters)
model   = LogisticRegression(C=1.0, max_iter=200, random_state=42)
tree    = DecisionTreeClassifier(max_depth=5)
scaler  = StandardScaler()
pca     = PCA(n_components=2)

# All estimators follow: .fit(X, y) for supervised, .fit(X) for unsupervised
# All transformers follow: .transform(X)
# All supervised models follow: .predict(X)

# Inspection: all hyperparameters accessible via get_params()
print(model.get_params())

# Fitted attributes end with underscore (convention)
scaler.fit([[1, 2], [3, 4], [5, 6]])
print(scaler.mean_)     # [3. 4.]  — learned from data
print(scaler.scale_)    # [1.63 1.63]
```

---

### 6.2 Transformers and Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ── ColumnTransformer: apply different transforms to different columns ─────────
numerical_features   = ['age', 'income', 'years_experience']
categorical_features = ['education', 'job_title']

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer,   numerical_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')

# ── Full pipeline: preprocessing + model ─────────────────────────────────────
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(n_estimators=100, random_state=42))
])

# This pipeline:
# 1. Scales numerics, encodes categoricals
# 2. Trains on the result
# 3. At predict time, applies the SAME transformation learned during training
# → No data leakage possible
print(full_pipeline)
```

---

### 6.3 The fit/transform/predict Contract

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

rng = np.random.default_rng(42)
X = rng.standard_normal((200, 5))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
model  = LogisticRegression()

# CORRECT workflow:
scaler.fit(X_train)             # Learn mean/std from training data ONLY
X_train_scaled = scaler.transform(X_train)   # Apply to training data
X_test_scaled  = scaler.transform(X_test)    # Apply SAME transform to test data

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# WRONG (data leakage):
# scaler.fit(X)               # ← Never fit on the whole dataset
# X_scaled = scaler.transform(X)

# fit_transform is ONLY acceptable on training data
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform in one step (training only)
X_test_scaled  = scaler.transform(X_test)         # transform only (test)

print(f"Fitted params: mean={scaler.mean_[:3].round(2)}, scale={scaler.scale_[:3].round(2)}")
```

---

## 7. Performance: NumPy vs Pure Python

A comprehensive benchmark to build intuition for *when* vectorization matters:

```python
import numpy as np
import time

def benchmark(name, func, *args, n_runs=5):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - start)
    mean_t = np.mean(times)
    print(f"{name:40s}: {mean_t*1000:8.3f} ms")
    return result

n = 1_000_000
x_list  = list(range(n))
x_array = np.arange(n, dtype=np.float64)

print("=" * 55)
print("Operation: Sum of 1 million elements")
print("=" * 55)

benchmark("Python built-in sum(list)",     sum, x_list)
benchmark("NumPy array.sum()",             lambda x: x.sum(), x_array)
benchmark("NumPy np.sum(array)",           np.sum, x_array)

print()
print("=" * 55)
print("Operation: Element-wise multiply + add (1M elements)")
print("=" * 55)

x_arr = np.random.default_rng(0).standard_normal(n)
y_arr = np.random.default_rng(1).standard_normal(n)

def loop_multiply_add(x, y):
    return [xi * 2 + yi for xi, yi in zip(x, y)]

benchmark("Python list comprehension",     loop_multiply_add, list(x_arr), list(y_arr))
benchmark("NumPy vectorized",              lambda a, b: a * 2 + b, x_arr, y_arr)

print()
print("=" * 55)
print("Operation: Matrix multiply (1000×1000)")
print("=" * 55)

A = np.random.default_rng(0).standard_normal((1000, 1000))
B = np.random.default_rng(1).standard_normal((1000, 1000))

benchmark("NumPy A @ B (BLAS)",            lambda a, b: a @ b, A, B)
# Python nested loop version would take ~10 minutes — not worth benchmarking
```

---

## 8. Putting It All Together: A Complete EDA Workflow

```python
# =============================================================================
# Complete EDA Workflow on the Titanic Dataset
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# ── Load data ─────────────────────────────────────────────────────────────────
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.data.copy()
df['survived'] = titanic.target.astype(int)

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())

# ── Missing value audit ───────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
missing_df = missing_df[missing_df['count'] > 0].sort_values('pct', ascending=False)
print("\nMissing values:\n", missing_df)

# ── Numerical summary ─────────────────────────────────────────────────────────
numerical_cols = ['age', 'fare', 'sibsp', 'parch']
print("\nNumerical summary:\n", df[numerical_cols].describe().round(2))

# ── Target distribution ───────────────────────────────────────────────────────
print("\nSurvival rate:", df['survived'].mean().round(3))
print(df['survived'].value_counts())

# ── Key relationships ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Survival by class
survival_by_class = df.groupby('pclass')['survived'].mean()
axes[0, 0].bar(survival_by_class.index.astype(str), survival_by_class.values,
               color=['#2196F3', '#FF9800', '#F44336'])
axes[0, 0].set_title('Survival Rate by Passenger Class')
axes[0, 0].set_xlabel('Class'); axes[0, 0].set_ylabel('Survival Rate')
axes[0, 0].set_ylim(0, 1)

# Survival by sex
survival_by_sex = df.groupby('sex')['survived'].mean()
axes[0, 1].bar(survival_by_sex.index, survival_by_sex.values,
               color=['#F48FB1', '#64B5F6'])
axes[0, 1].set_title('Survival Rate by Sex')
axes[0, 1].set_xlabel('Sex'); axes[0, 1].set_ylabel('Survival Rate')
axes[0, 1].set_ylim(0, 1)

# Age distribution by survival
df['age'] = pd.to_numeric(df['age'], errors='coerce')
ax = axes[0, 2]
for survived, color, label in [(0, '#F44336', 'Did not survive'), (1, '#4CAF50', 'Survived')]:
    mask = df['survived'] == survived
    ax.hist(df.loc[mask, 'age'].dropna(), bins=25, alpha=0.6, color=color, label=label)
ax.set_title('Age Distribution by Survival')
ax.set_xlabel('Age'); ax.set_ylabel('Count'); ax.legend()

# Fare distribution (log scale)
df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
axes[1, 0].hist(np.log1p(df['fare'].dropna()), bins=30, color='steelblue', alpha=0.7)
axes[1, 0].set_title('Log(Fare) Distribution')
axes[1, 0].set_xlabel('log(1 + fare)'); axes[1, 0].set_ylabel('Count')

# Survival heatmap: class × sex
pivot = df.groupby(['pclass', 'sex'])['survived'].mean().unstack()
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
            ax=axes[1, 1], vmin=0, vmax=1)
axes[1, 1].set_title('Survival Rate: Class × Sex')

# Correlation of numeric features
num_df = df[['age', 'fare', 'sibsp', 'parch', 'survived']].apply(pd.to_numeric, errors='coerce')
corr = num_df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1, 2], vmin=-1, vmax=1, square=True)
axes[1, 2].set_title('Correlation Matrix')

plt.suptitle('Titanic Dataset — Exploratory Data Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/02_titanic_eda.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Key findings (document these!) ───────────────────────────────────────────
print("""
Key EDA Findings:
-----------------
1. Survival rate: 38.2% — class imbalance matters.
2. Strong effect of sex: women survived at much higher rates than men.
3. Passenger class strongly predicts survival (1st > 2nd > 3rd).
4. Age: children had higher survival rates; elderly had lower.
5. Fare is right-skewed — log transform recommended.
6. 'Age' has 20% missing values — needs imputation, not dropping.
7. 'Cabin' has 77% missing — likely not usable without feature engineering.
""")
```

---

## 9. Summary

### ✅ Must-Remember Mental Models

**NumPy:**
- An ndarray is a **contiguous block of typed memory**. That's why it's fast.
- `shape` tells you dimensions. `dtype` tells you the type. Check both immediately after loading any data.
- **Views vs copies**: Slices return views. Fancy indexing returns copies. Modifying a view modifies the original.
- **Vectorize everything.** If you're writing a Python loop over array elements, there is almost certainly a vectorized alternative.
- **Broadcasting rule**: dimensions are compatible if they are equal or one of them is 1. Shapes are aligned from the right.
- `axis=0` → operate along rows (collapse rows). `axis=1` → operate along columns (collapse columns).

**Pandas:**
- `df.info()` and `df.describe()` are your first two commands on any new dataset. Always.
- `.loc` is label-based. `.iloc` is integer-based. Never mix them up.
- `groupby` + `transform` preserves the original DataFrame's index — essential for creating group-level features.
- Always check for missing values before modeling. Understand *why* data is missing.
- `df.values` or `df.to_numpy()` converts to NumPy. Scikit-learn accepts either.

**Matplotlib / Seaborn:**
- Always use the **object-oriented interface** (`fig, ax = plt.subplots()`). The pyplot stateful interface causes bugs in loops and subplots.
- For EDA: pairplot (Seaborn), distribution histograms, correlation heatmap.
- For model evaluation: loss curves, confusion matrix, ROC curve, learning curves.
- Save every important figure with `savefig()` before `show()`.

**Scikit-learn:**
- **Every estimator follows the same contract**: `fit()`, `transform()`, `predict()`.
- **Always use Pipelines.** They prevent data leakage by ensuring transformers are fit only on training data.
- **Fitted attributes end with `_`** (e.g., `scaler.mean_`, `model.coef_`). Check them to understand what the model learned.
- **The golden rule**: fit transformers on training data, apply to test data. Never the reverse.

---

## 10. Exercises

**NumPy:**

1. Create a 5×5 matrix where element `(i, j)` equals `i * j`. Do this using broadcasting without any loops.

2. Given a 2D array `X` of shape `(100, 4)`, compute the z-score normalization along columns using only NumPy operations (no Scikit-learn). Verify that the result has mean ≈ 0 and std ≈ 1 for each column.

3. Implement the softmax function vectorized: `softmax(z) = exp(z) / sum(exp(z))`. Handle numerical stability by subtracting the max before exponentiating. Test it on `z = np.array([2.0, 1.0, 0.1])`.

4. Given a 1D array of class labels `y = np.array([0, 2, 1, 0, 2, 2, 1])`, convert it to a one-hot encoded matrix of shape `(7, 3)` using only NumPy (no Scikit-learn). Hint: use fancy indexing.

5. Implement batched matrix multiplication: given a 3D array `A` of shape `(32, 4, 4)` (32 matrices of size 4×4) and a 2D array `B` of shape `(4, 4)`, compute `A @ B` using NumPy broadcasting. Verify against a loop version.

**Pandas:**

6. Load the breast cancer dataset from Scikit-learn and convert it to a Pandas DataFrame. Find the 5 features most correlated with the target. Are any features nearly perfectly correlated with each other (|corr| > 0.95)? What does this imply?

7. Simulate a dataset with missing values: generate a 200×5 DataFrame of random floats, then randomly set 10% of values to NaN. Compare three imputation strategies (mean, median, KNN with k=5) by measuring how well each recovers the original values. (Hint: save the original, then introduce NaNs, impute, and compute MAE vs original.)

8. Using the Titanic dataset from Section 8, create the following new features using Pandas:
   - `family_size = sibsp + parch + 1`
   - `is_alone = (family_size == 1).astype(int)`
   - `title` = extracted from the `name` column (Mr., Mrs., Miss., etc.)
   - `fare_per_person = fare / family_size`
   
   Then compute survival rates for each title.

**Matplotlib / Seaborn:**

9. Reproduce the bias-variance visualization from Chapter 1 (polynomial regression) but this time use the full Matplotlib object-oriented interface, add confidence intervals (use `ax.fill_between()`), and save the figure as both `.png` and `.pdf`.

10. Create a publication-quality correlation heatmap for the breast cancer dataset. Show only the lower triangle, annotate with values, and highlight correlations |r| > 0.8 in bold.

**Challenge:**

11. Implement a memory-efficient moving average using NumPy's `stride_tricks`. Given a 1D time series of 10,000 points, compute the rolling mean with window size 50 — without any loops and without creating intermediate arrays larger than necessary.
    
    ```python
    from numpy.lib.stride_tricks import sliding_window_view
    # Hint: sliding_window_view(x, window_shape=50) → shape (9951, 50)
    # Then take the mean along axis=1
    ```

---

## 11. References

- Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. — The NumPy paper.
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*. — The Pandas paper.
- Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90–95.
- Waskom, M. (2021). Seaborn: Statistical Data Visualization. *Journal of Open Source Software*, 6(60), 3021.
- NumPy Documentation: https://numpy.org/doc/stable/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Matplotlib Documentation: https://matplotlib.org/stable/contents.html
- Scikit-learn Documentation: https://scikit-learn.org/stable/

---

*Previous: [Chapter 1 — The Machine Learning Landscape](01_ml_landscape.md)*  
*Next: [Chapter 3 — Data Wrangling & Exploratory Data Analysis](03_eda.md)*

*In Chapter 3, we go deeper into the full EDA process — detecting outliers, handling skewed distributions, understanding feature interactions, and building the data intuition that separates strong ML practitioners from those who just run models.*

---

> **Chapter 2 complete.** All code is in `notebooks/02_python_stack.ipynb`. Every code block is self-contained and executable.
