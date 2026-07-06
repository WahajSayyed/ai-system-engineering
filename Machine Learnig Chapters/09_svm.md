# Chapter 9: Support Vector Machines

> *"The SVM is perhaps the most theoretically elegant supervised learning algorithm. It finds the unique decision boundary that is maximally far from all training examples — and does so with a beautiful geometric and mathematical structure."*

> *"The kernel trick is one of the greatest ideas in machine learning: it transforms an impossible non-linear problem in the original space into a tractable linear problem in a high-dimensional feature space — without ever computing the transformation explicitly."*

---

## Table of Contents

1. [The Maximum Margin Classifier](#1-the-maximum-margin-classifier)
2. [The Hard-Margin SVM](#2-the-hard-margin-svm)
   - 2.1 [Geometric Intuition](#21-geometric-intuition)
   - 2.2 [The Optimization Problem](#22-the-optimization-problem)
   - 2.3 [Support Vectors](#23-support-vectors)
3. [The Soft-Margin SVM](#3-the-soft-margin-svm)
   - 3.1 [Slack Variables](#31-slack-variables)
   - 3.2 [The C Parameter](#32-the-c-parameter)
4. [The Dual Problem and Kernel Trick](#4-the-dual-problem-and-kernel-trick)
   - 4.1 [Lagrangian Duality](#41-lagrangian-duality)
   - 4.2 [The Kernel Trick](#42-the-kernel-trick)
   - 4.3 [Common Kernels](#43-common-kernels)
5. [The Hinge Loss](#5-the-hinge-loss)
6. [SVMs for Regression — SVR](#6-svms-for-regression--svr)
7. [Multi-class SVMs](#7-multi-class-svms)
8. [SVM with Scikit-learn](#8-svm-with-scikit-learn)
9. [When to Use SVMs](#9-when-to-use-svms)
10. [Case Study: Text Classification with Linear SVM](#10-case-study-text-classification-with-linear-svm)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. The Maximum Margin Classifier

Suppose you have two linearly separable classes. There are infinitely many hyperplanes that separate them. Which one should you choose?

The SVM chooses the hyperplane with the **maximum margin** — the one that is furthest from the nearest training point of each class. Intuitively, a larger margin means the classifier is more "confident" about its decision and will generalize better to new data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification

rng = np.random.default_rng(42)

# ── Linearly separable data ────────────────────────────────────────────────────
X = np.array([
    [1.0, 1.5], [1.5, 2.0], [2.0, 1.0], [1.0, 2.5],   # Class 0
    [3.5, 3.5], [4.0, 3.0], [3.5, 4.5], [4.5, 4.0],   # Class 1
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, title, color_set in zip(
    axes,
    ['Many valid separating hyperplanes',
     'SVM: Maximum Margin Hyperplane',
     'Support Vectors and Margin Width'],
    ['gray', 'steelblue', 'steelblue']
):
    ax.scatter(X[y==0, 0], X[y==0, 1], s=80, color='#EF9A9A',
               edgecolors='k', zorder=5, label='Class 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], s=80, color='#A5D6A7',
               edgecolors='k', zorder=5, label='Class 1')
    ax.set_xlim(0, 5.5); ax.set_ylim(0, 5.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

# Plot multiple arbitrary separating lines on axes[0]
ax = axes[0]
x_line = np.linspace(0, 5.5, 100)
for slope, intercept, alpha in [
    (1.0, -0.5, 0.4), (0.5, 0.5, 0.4), (2.0, -2.5, 0.4),
    (0.8, 0.2, 0.4), (1.5, -1.5, 0.4)
]:
    ax.plot(x_line, slope*x_line + intercept, 'gray', lw=1.5, alpha=alpha)
ax.text(0.3, 4.8, 'All separate the data\nbut which is "best"?',
        fontsize=9, color='gray')
ax.legend(fontsize=8)

# SVM decision boundary on axes[1]
ax = axes[1]
svm = SVC(kernel='linear', C=1e6)  # Hard margin (very large C)
svm.fit(X, y)
w  = svm.coef_[0]
b  = svm.intercept_[0]
# Decision boundary: w[0]*x1 + w[1]*x2 + b = 0
# → x2 = -(w[0]*x1 + b) / w[1]
db_x2 = -(w[0]*x_line + b) / w[1]
ax.plot(x_line, db_x2, 'steelblue', lw=2.5, label='Decision boundary')
# Margin boundaries: w·x + b = ±1
margin_pos = (-(w[0]*x_line + b) + 1) / w[1]
margin_neg = (-(w[0]*x_line + b) - 1) / w[1]
ax.plot(x_line, margin_pos, 'steelblue', lw=1.5, linestyle='--', alpha=0.6)
ax.plot(x_line, margin_neg, 'steelblue', lw=1.5, linestyle='--', alpha=0.6)
ax.fill_between(x_line, margin_neg, margin_pos, alpha=0.1, color='steelblue')
ax.legend(fontsize=8)

# Support vectors on axes[2]
ax = axes[2]
db_x2 = -(w[0]*x_line + b) / w[1]
margin_pos = (-(w[0]*x_line + b) + 1) / w[1]
margin_neg = (-(w[0]*x_line + b) - 1) / w[1]
ax.scatter(X[y==0, 0], X[y==0, 1], s=80, color='#EF9A9A',
           edgecolors='k', zorder=5)
ax.scatter(X[y==1, 0], X[y==1, 1], s=80, color='#A5D6A7',
           edgecolors='k', zorder=5)
ax.plot(x_line, db_x2, 'steelblue', lw=2.5)
ax.plot(x_line, margin_pos, 'steelblue', lw=1.5, linestyle='--')
ax.plot(x_line, margin_neg, 'steelblue', lw=1.5, linestyle='--')
ax.fill_between(x_line, margin_neg, margin_pos, alpha=0.1, color='steelblue')

# Highlight support vectors
sv_idx = svm.support_
ax.scatter(X[sv_idx, 0], X[sv_idx, 1], s=250, facecolors='none',
           edgecolors='gold', linewidths=3, zorder=6, label='Support vectors')

# Draw margin width arrow
margin_width = 2 / np.linalg.norm(w)
ax.annotate('', xy=(3.0, db_x2[50]+margin_width/2*w[1]/np.linalg.norm(w)),
            xytext=(3.0, db_x2[50]-margin_width/2*w[1]/np.linalg.norm(w)),
            arrowprops=dict(arrowstyle='<->', color='tomato', lw=2))
ax.text(3.2, db_x2[50], f'Margin\n= {margin_width:.3f}',
        color='tomato', fontsize=9)
ax.legend(fontsize=8)

plt.suptitle('Support Vector Machine: Maximum Margin Classifier',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_svm_margin.png', dpi=140, bbox_inches='tight')
plt.show()

margin = 2 / np.linalg.norm(w)
print(f"SVM margin width: {margin:.4f}")
print(f"Number of support vectors: {svm.n_support_}")
print(f"Support vector indices: {svm.support_}")
print(f"Weight vector w: {w.round(4)}")
print(f"Bias b: {b:.4f}")
```

---

## 2. The Hard-Margin SVM

### 2.1 Geometric Intuition

A **hyperplane** in $\mathbb{R}^d$ is defined by:

$$\mathbf{w}^T\mathbf{x} + b = 0$$

The **signed distance** from point $\mathbf{x}^{(i)}$ to the hyperplane is:

$$d^{(i)} = \frac{\mathbf{w}^T\mathbf{x}^{(i)} + b}{\|\mathbf{w}\|}$$

The **functional margin** of point $i$ with label $y^{(i)} \in \{-1, +1\}$ is:

$$\hat{\gamma}^{(i)} = y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b)$$

For a correctly classified point: $\hat{\gamma}^{(i)} > 0$. The **geometric margin** is $\hat{\gamma}^{(i)} / \|\mathbf{w}\|$.

---

### 2.2 The Optimization Problem

We want to maximize the **minimum margin** across all training points.

By convention, we normalize so that the minimum functional margin = 1 (we can always rescale $\mathbf{w}$ and $b$):

$$y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 \quad \forall i$$

The geometric margin then equals $1/\|\mathbf{w}\|$. Maximizing $1/\|\mathbf{w}\|$ is equivalent to minimizing $\|\mathbf{w}\|^2$:

$$\boxed{\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 \quad \forall i}$$

This is a **convex quadratic program** with linear constraints — exactly one global minimum.

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
Hard-Margin SVM: Primal Problem
═══════════════════════════════════════════════════════════════════════

Objective:    min_{w, b}  (1/2) ||w||²
Constraints:  y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1   for all i = 1, ..., n

Intuition:
  • ||w||² = 1/margin²  →  minimizing ||w||² = maximizing margin
  • Factor of 1/2 is for clean derivatives (convention only)
  • Constraints say: each point must be on the correct side,
    at least 1 unit away from the decision boundary
    (in the w-normalized coordinate system)

KKT Conditions (Karush-Kuhn-Tucker):
  For each support vector i (active constraint):
    αᵢ > 0  AND  y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) = 1
  For non-support vectors:
    αᵢ = 0  (constraint is inactive: y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) > 1)

Key insight: The solution w* depends ONLY on the support vectors!
  w* = Σᵢ αᵢ yᵢ xᵢ   (sum over support vectors only)

This means: you can delete all non-support-vector training points
and the trained SVM remains identical.
═══════════════════════════════════════════════════════════════════════
""")
```

---

### 2.3 Support Vectors

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# ── Demonstrate that only support vectors matter ──────────────────────────────
rng = np.random.default_rng(42)
n   = 40

# Generate separable data
X_c0 = rng.multivariate_normal([1, 1], [[0.3, 0], [0, 0.3]], n//2)
X_c1 = rng.multivariate_normal([3, 3], [[0.3, 0], [0, 0.3]], n//2)
X    = np.vstack([X_c0, X_c1])
y    = np.hstack([-np.ones(n//2), np.ones(n//2)])

svm = SVC(kernel='linear', C=1e6)
svm.fit(X, y)

sv_mask    = np.zeros(n, dtype=bool)
sv_mask[svm.support_] = True
n_sv       = sv_mask.sum()

# Retrain using ONLY the support vectors
X_sv = X[sv_mask]
y_sv = y[sv_mask]
svm_sv = SVC(kernel='linear', C=1e6)
svm_sv.fit(X_sv, y_sv)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def plot_svm_boundary(ax, X, y, svm, title, highlight_sv=True):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1],
                colors=['#FFCDD2', '#E8F5E9'], alpha=0.5)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               colors=['tomato', 'steelblue', 'green'],
               linewidths=[1.5, 2.5, 1.5],
               linestyles=['--', '-', '--'])
    ax.scatter(X[y==-1, 0], X[y==-1, 1], s=40, color='#EF9A9A', edgecolors='k',
               linewidths=0.5, zorder=5)
    ax.scatter(X[y== 1, 0], X[y== 1, 1], s=40, color='#A5D6A7', edgecolors='k',
               linewidths=0.5, zorder=5)
    if highlight_sv:
        sv = svm.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
                   edgecolors='gold', linewidths=3, zorder=6,
                   label=f'{len(sv)} support vectors')
    ax.set_title(title); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

plot_svm_boundary(axes[0], X, y, svm,
                  f'Full dataset ({n} points)\n{n_sv} support vectors identified')
plot_svm_boundary(axes[1], X_sv, y_sv, svm_sv,
                  f'Retrained on support vectors ONLY ({n_sv} points)\n'
                  f'Identical decision boundary!')

plt.suptitle('Support Vectors are ALL That Matter\n'
             'Delete all other training points → same SVM',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_support_vectors.png', dpi=140, bbox_inches='tight')
plt.show()

w1, w2 = svm.coef_[0], svm_sv.coef_[0]
b1, b2 = svm.intercept_[0], svm_sv.intercept_[0]
print(f"Full SVM:   w={w1.round(4)}, b={b1:.4f}")
print(f"SV-only SVM: w={w2.round(4)}, b={b2:.4f}")
print(f"Identical: {np.allclose(w1, w2, atol=1e-3) and np.isclose(b1, b2, atol=1e-3)}")
```

---

## 3. The Soft-Margin SVM

### 3.1 Slack Variables

Real data is rarely linearly separable. The **soft-margin SVM** introduces **slack variables** $\xi_i \geq 0$ that allow some points to violate the margin:

$$y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 - \xi_i$$

- $\xi_i = 0$: point correctly classified, outside margin
- $0 < \xi_i \leq 1$: point inside margin but correctly classified
- $\xi_i > 1$: point misclassified

The modified objective **penalizes** slack:

$$\boxed{\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1-\xi_i, \quad \xi_i \geq 0}$$

### 3.2 The C Parameter

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification

rng = np.random.default_rng(42)
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                            n_informative=2, class_sep=0.8, random_state=42)
y = 2*y - 1   # Convert to {-1, +1}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
C_values   = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]

for ax, C in zip(axes.flatten(), C_values):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X, y)

    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               colors=['tomato', 'black', 'green'],
               linewidths=[1.5, 2.5, 1.5],
               linestyles=['--', '-', '--'])
    ax.scatter(X[y==-1, 0], X[y==-1, 1], s=20, color='tomato',
               edgecolors='k', linewidths=0.3, zorder=5, alpha=0.8)
    ax.scatter(X[y== 1, 0], X[y== 1, 1], s=20, color='steelblue',
               edgecolors='k', linewidths=0.3, zorder=5, alpha=0.8)
    sv = svm.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors='none',
               edgecolors='gold', linewidths=2, zorder=6)

    margin = 2 / np.linalg.norm(svm.coef_[0])
    n_sv   = len(svm.support_)
    train_acc = svm.score(X, y)

    ax.set_title(f'C={C}\nMargin={margin:.3f}, #SV={n_sv}, TrainAcc={train_acc:.3f}',
                 fontsize=9)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=7)

plt.suptitle('Soft-Margin SVM: Effect of C\n'
             'Small C → wide margin, many SV (underfitting)\n'
             'Large C → narrow margin, few SV (overfitting risk)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_soft_margin_C.png', dpi=130, bbox_inches='tight')
plt.show()

print("""
C Parameter Interpretation:
────────────────────────────────────────────────────────────────
C is the INVERSE of regularization strength (like in LogisticRegression):
  Small C → large regularization → wide margin → more misclassifications OK
  Large C → small regularization → narrow margin → tries to classify all points

Mathematically:
  SVM loss = (1/2)||w||² + C Σ ξᵢ
  C controls the tradeoff between margin width and training error.
  C → ∞: hard margin (no violations allowed)
  C → 0: ignore the data, maximize margin only (all slack)

Dual interpretation — C bounds the dual variables:
  0 ≤ αᵢ ≤ C   (vs hard margin: 0 ≤ αᵢ without upper bound)
────────────────────────────────────────────────────────────────
""")
```

---

## 4. The Dual Problem and Kernel Trick

### 4.1 Lagrangian Duality

The power of SVMs comes from converting the primal problem to its **dual form** using Lagrange multipliers.

**Lagrangian:**
$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i \left[y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) - 1\right]$$

Setting derivatives to zero and substituting back:

$$\boxed{\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y^{(i)} y^{(j)} \langle\mathbf{x}^{(i)}, \mathbf{x}^{(j)}\rangle \quad \text{s.t.} \quad \sum_i \alpha_i y^{(i)} = 0, \; 0 \leq \alpha_i \leq C}$$

The dual formulation depends on training data **only through inner products** $\langle\mathbf{x}^{(i)}, \mathbf{x}^{(j)}\rangle$. This is the doorway to the kernel trick.

The primal solution recovered from dual:
$$\mathbf{w}^* = \sum_{i=1}^n \alpha_i y^{(i)} \mathbf{x}^{(i)}$$

Prediction:
$$\hat{y}(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i y^{(i)} \langle\mathbf{x}^{(i)}, \mathbf{x}\rangle + b\right)$$

---

### 4.2 The Kernel Trick

The kernel trick replaces the inner product $\langle\mathbf{x}^{(i)}, \mathbf{x}^{(j)}\rangle$ with a **kernel function** $K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$:

$$\hat{y}(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i y^{(i)} K(\mathbf{x}^{(i)}, \mathbf{x}) + b\right)$$

The kernel implicitly computes a high-dimensional feature map $\phi(\mathbf{x})$ such that $K(\mathbf{x}, \mathbf{z}) = \langle\phi(\mathbf{x}), \phi(\mathbf{z})\rangle$ — but **never explicitly computes $\phi$**.

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
THE KERNEL TRICK — Why It's Remarkable
══════════════════════════════════════════════════════════════════════

Problem: Data is not linearly separable in the original space.
Solution: Map to a higher-dimensional space where it IS separable.

Example: 1D data, not linearly separable
  Points: x = {-2, -1, 1, 2}
  Labels:  y = { -1,  1, 1, -1}  (positive class in the middle)
  No threshold in 1D separates these.

Transformation: φ(x) = [x, x²]  (map to 2D)
  φ(-2) = [-2, 4]
  φ(-1) = [-1, 1]
  φ( 1) = [ 1, 1]
  φ( 2) = [ 2, 4]
  
  Now a hyperplane in 2D separates them! (e.g., x₂ = 2)

Kernel: K(x, z) = φ(x)·φ(z) = xz + (xz)² = xz + x²z²
  This is the polynomial kernel of degree 2.
  We NEVER compute φ(x) explicitly — just evaluate K(x, z)!

For RBF kernel: K(x, z) = exp(-γ||x-z||²)
  The implicit feature space is INFINITE dimensional.
  Yet we only compute a single scalar K(x, z).
  This is the computational miracle of the kernel trick.
══════════════════════════════════════════════════════════════════════
""")

# ── Visual demonstration: linear → non-linear via kernel ──────────────────────
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons

datasets = {
    'Circles': make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42),
    'Moons':   make_moons(n_samples=200, noise=0.15, random_state=42),
}

kernels = ['linear', 'rbf', 'poly']

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for row, (ds_name, (X_ds, y_ds)) in enumerate(datasets.items()):
    for col, kernel in enumerate(kernels):
        ax = axes[row, col]

        svm = SVC(kernel=kernel, C=1.0, gamma='scale', degree=3)
        svm.fit(X_ds, y_ds)

        x_min, x_max = X_ds[:,0].min()-0.4, X_ds[:,0].max()+0.4
        y_min, y_max = X_ds[:,1].min()-0.4, X_ds[:,1].max()+0.4
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                              np.linspace(y_min, y_max, 250))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r')
        ax.scatter(X_ds[y_ds==0, 0], X_ds[y_ds==0, 1], s=25,
                   color='#EF9A9A', edgecolors='k', linewidths=0.3, zorder=5)
        ax.scatter(X_ds[y_ds==1, 0], X_ds[y_ds==1, 1], s=25,
                   color='#A5D6A7', edgecolors='k', linewidths=0.3, zorder=5)
        sv = svm.support_vectors_
        ax.scatter(sv[:,0], sv[:,1], s=120, facecolors='none',
                   edgecolors='gold', linewidths=2, zorder=6)
        acc = svm.score(X_ds, y_ds)
        ax.set_title(f'{ds_name} — {kernel} kernel\nAcc={acc:.3f}, #SV={len(sv)}',
                     fontsize=9)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.tick_params(labelsize=7)

plt.suptitle('Kernel SVM: Different Kernels on Non-Linear Data\n'
             'Gold circles = support vectors',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_kernel_svm.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

### 4.3 Common Kernels

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
KERNEL REFERENCE TABLE
══════════════════════════════════════════════════════════════════════════════

Kernel          Formula                            Parameters   When to use
──────────────────────────────────────────────────────────────────────────────
Linear          K(x,z) = xᵀz                       C only       Linearly separable,
                                                                  high-dimensional (text),
                                                                  large n (use LinearSVC)

Polynomial      K(x,z) = (γxᵀz + r)^d             γ, r, d, C   Low-degree non-linearity
                d=2: quadratic interactions                       Genomics, NLP embeddings
                
RBF (Gaussian)  K(x,z) = exp(-γ||x-z||²)           γ, C         Default choice for
                = exp(-||x-z||²/(2σ²))                            non-linear problems
                γ = 1/(2σ²)                                       Works for most problems
                
Sigmoid         K(x,z) = tanh(γxᵀz + r)            γ, r, C      Neural-network-like;
                                                                  rarely outperforms RBF
                
Laplacian       K(x,z) = exp(-γ||x-z||₁)           γ, C         Robust to outliers;
                                                                  not available in sklearn

Matern          Various                              ν, γ, C      Gaussian processes;
                                                                  smooth functions

Notes:
  • All kernels must satisfy Mercer's theorem (be positive semi-definite)
    to guarantee a valid inner product in some feature space
  • RBF is usually the best default; try linear first if n >> d
  • γ in RBF controls the "reach" of each training point:
    Large γ → small σ → local influence → complex boundary → overfit risk
    Small γ → large σ → global influence → smooth boundary → underfit risk
══════════════════════════════════════════════════════════════════════════════
""")

# ── Visualize RBF kernel as a similarity function ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

x_ref = np.array([0.0])
x_range = np.linspace(-4, 4, 500)

ax = axes[0]
for gamma, color, ls in [(0.1, '#EF9A9A', '-'), (0.5, '#FFCC80', '--'),
                          (1.0, '#A5D6A7', '-.'), (5.0, '#90CAF9', ':')]:
    K_vals = np.exp(-gamma * (x_range - x_ref[0])**2)
    ax.plot(x_range, K_vals, color=color, lw=2.5, ls=ls,
            label=f'γ={gamma}')
ax.axvline(0, color='gray', lw=1, linestyle='--')
ax.set_xlabel('x'); ax.set_ylabel('K(0, x)')
ax.set_title('RBF Kernel: K(0, x) = exp(-γ|x|²)\n'
             'Large γ → narrow similarity region')
ax.legend(); ax.grid(True, alpha=0.3)

# Kernel matrix visualization
ax2 = axes[1]
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
rng = np.random.default_rng(42)
X_km = np.sort(rng.standard_normal((30, 2)), axis=0)
K_rbf = rbf_kernel(X_km, gamma=1.0)
im = ax2.imshow(K_rbf, cmap='Blues', aspect='auto')
plt.colorbar(im, ax=ax2)
ax2.set_title('RBF Kernel Matrix K[i,j] = K(xᵢ, xⱼ)\n'
              'Diagonal = 1 (self-similarity); off-diag = similarity')
ax2.set_xlabel('Sample index'); ax2.set_ylabel('Sample index')

# Polynomial kernel degree effect
ax3 = axes[2]
x_p = np.linspace(-2, 2, 300).reshape(-1, 1)
for d, color in zip([1, 2, 3, 5], ['#EF9A9A', '#A5D6A7', '#90CAF9', '#CE93D8']):
    K_p = polynomial_kernel(np.array([[1.0]]), x_p, degree=d, gamma=1.0, coef0=1.0)[0]
    ax3.plot(x_p.ravel(), K_p, color=color, lw=2.5, label=f'd={d}')
ax3.axvline(1, color='gray', lw=1, linestyle='--')
ax3.set_xlabel('x'); ax3.set_ylabel('K(1, x) = (x + 1)^d')
ax3.set_title('Polynomial Kernel Degree Effect\nHigher d → more complex boundary')
ax3.legend(); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/09_kernels.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 5. The Hinge Loss

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
SVM AS REGULARIZED HINGE LOSS MINIMIZATION
══════════════════════════════════════════════════════════════════════

The soft-margin SVM primal problem can be written as:
  min_{w,b}  (1/n) Σᵢ max(0, 1 - yᵢ(wᵀxᵢ + b)) + (λ/2)||w||²
                   ▲ hinge loss                      ▲ L2 regularization

Where λ = 1/(nC).

Hinge loss:  L(yᵢ, fᵢ) = max(0, 1 - yᵢfᵢ)
  • yᵢfᵢ ≥ 1:  point correctly classified outside margin → loss = 0
  • 0 < yᵢfᵢ < 1:  point inside margin → loss = 1 - yᵢfᵢ  (positive)
  • yᵢfᵢ ≤ 0:  point misclassified → loss = 1 - yᵢfᵢ  (large)

This connects SVMs to:
  • Logistic regression: replace hinge with log-loss
  • Linear regression: replace hinge with squared error
  • All use L2 regularization on weights
""")

# ── Compare hinge vs log-loss vs 0-1 loss ────────────────────────────────────
z = np.linspace(-3, 3, 500)   # y * f(x): positive = correct prediction

hinge   = np.maximum(0, 1 - z)
log_l   = np.log1p(np.exp(-z)) / np.log(2)   # Log-loss (nats → bits)
sq_h    = np.maximum(0, 1 - z)**2             # Squared hinge
zero_one = (z < 0).astype(float)             # 0-1 loss (non-convex, non-smooth)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(z, zero_one, 'k--', lw=2,   label='0-1 Loss (ideal but intractable)')
ax.plot(z, hinge,   color='steelblue', lw=2.5, label='Hinge Loss (SVM)')
ax.plot(z, log_l,   color='tomato',    lw=2.5, label='Log Loss (Logistic)')
ax.plot(z, sq_h,    color='green',     lw=2.5, label='Squared Hinge')
ax.axvline(0, color='gray', lw=1, linestyle=':')
ax.axvline(1, color='gray', lw=1, linestyle=':', label='Margin boundary')
ax.set_xlabel('yf(x) — Functional margin')
ax.set_ylabel('Loss')
ax.set_title('Loss Functions for Binary Classification\n'
             'All are upper bounds on 0-1 loss (convex surrogates)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(-0.2, 4)

# Hinge gradient (subgradient)
ax2 = axes[1]
hinge_grad = np.where(z < 1, -1.0, 0.0)   # Subgradient of hinge
log_grad   = -1 / (1 + np.exp(z)) / np.log(2)   # Derivative of log-loss

ax2.plot(z, hinge_grad, color='steelblue', lw=2.5, label="Hinge subgradient")
ax2.plot(z, log_grad,   color='tomato',    lw=2.5, label="Log-loss gradient")
ax2.axhline(0, color='gray', lw=1); ax2.axvline(0, color='gray', lw=1, linestyle=':')
ax2.axvline(1, color='gray', lw=1, linestyle=':')
ax2.set_xlabel('yf(x)'); ax2.set_ylabel('Gradient / Subgradient')
ax2.set_title('Gradients: Hinge vs Log-Loss\n'
              'Hinge: sparse gradient (zero when correctly classified)\n'
              'Log-loss: always non-zero (never fully "satisfied")')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.suptitle('Hinge Loss: The SVM Objective as Empirical Risk Minimization',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_hinge_loss.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 6. SVMs for Regression — SVR

**Support Vector Regression (SVR)** uses an ε-insensitive tube: predictions within ε of the true value incur no loss.

$$L_\epsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \epsilon)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(42)
n   = 80
X_r = np.sort(rng.uniform(0, 10, n)).reshape(-1, 1)
y_r = np.sin(X_r.ravel()) * 2 + rng.normal(0, 0.4, n)

X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_r)
X_plot_sc = scaler.transform(X_plot)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

configs = [
    ('RBF, ε=0.1, C=1',   SVR(kernel='rbf',  epsilon=0.1, C=1.0)),
    ('RBF, ε=0.5, C=1',   SVR(kernel='rbf',  epsilon=0.5, C=1.0)),
    ('RBF, ε=0.1, C=100', SVR(kernel='rbf',  epsilon=0.1, C=100.0)),
]

for ax, (title, svr) in zip(axes, configs):
    svr.fit(X_sc, y_r)
    y_pred = svr.predict(X_plot_sc)
    y_train_pred = svr.predict(X_sc)

    ax.scatter(X_r.ravel(), y_r, s=20, color='steelblue', alpha=0.6, label='Data')
    ax.plot(X_plot.ravel(), y_pred, 'tomato', lw=2.5, label='SVR prediction')
    ax.fill_between(X_plot.ravel(),
                    y_pred - svr.epsilon,
                    y_pred + svr.epsilon,
                    alpha=0.25, color='tomato', label=f'ε-tube (ε={svr.epsilon})')

    # Show support vectors
    sv_x = X_r[svr.support_].ravel()
    sv_y = y_r[svr.support_]
    ax.scatter(sv_x, sv_y, s=120, facecolors='none', edgecolors='gold',
               linewidths=2.5, zorder=6, label=f'{len(svr.support_)} SVs')

    from sklearn.metrics import mean_squared_error, r2_score
    r2 = r2_score(y_r, y_train_pred)
    ax.set_title(f'{title}\nR²={r2:.3f}', fontsize=9)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle('Support Vector Regression (SVR)\n'
             'Points inside the ε-tube have zero loss',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_svr.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
SVR Parameters:
  epsilon (ε) : Width of the insensitive tube.
                Large ε → more tolerance → fewer support vectors
                Small ε → tighter fit → more support vectors
  C           : Same as classification SVM.
                Large C → penalizes errors outside tube more
  kernel      : Same kernels as classification SVM
  
Key: SVR solves both the fit AND the tube width simultaneously.
""")
```

---

## 7. Multi-class SVMs

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

print("""
Multi-class SVM Strategies:
══════════════════════════════════════════════════════════════════════

1. One-vs-Rest (OvR): Train K binary classifiers.
   Classifier k: class k vs all others.
   Predict: class with highest decision function score.
   Sklearn: decision_function_shape='ovr' (default)

2. One-vs-One (OvO): Train K(K-1)/2 binary classifiers.
   Classifier (i,j): class i vs class j.
   Predict: majority vote across all classifiers.
   Sklearn: decision_function_shape='ovo'

   OvO is often preferred for SVMs because each binary subproblem
   is smaller (n/K samples) and faster to solve.
══════════════════════════════════════════════════════════════════════
""")

for strategy, dfs in [('OvR', 'ovr'), ('OvO', 'ovo')]:
    svm = SVC(kernel='rbf', C=10, gamma='scale',
              decision_function_shape=dfs, random_state=42)
    svm.fit(X_tr_s, y_tr)
    print(f"\n{strategy} ({svm.n_support_.sum()} total SVs):")
    print(f"Test accuracy: {svm.score(X_te_s, y_te):.4f}")
    print(classification_report(y_te, svm.predict(X_te_s),
                                target_names=iris.target_names))
```

---

## 8. SVM with Scikit-learn

```python
import numpy as np
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Classification: SVC vs LinearSVC ─────────────────────────────────────────
cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# SVC (kernel methods) — O(n²) to O(n³) training, best for n < ~100k
# LinearSVC (linear only) — O(n) with SGD solver, best for large n or text
models = {
    'SVC (RBF)': Pipeline([
        ('sc', StandardScaler()),
        ('m',  SVC(kernel='rbf', C=10, gamma='scale',
                   probability=True, random_state=42))
    ]),
    'SVC (Linear)': Pipeline([
        ('sc', StandardScaler()),
        ('m',  SVC(kernel='linear', C=1.0,
                   probability=True, random_state=42))
    ]),
    'LinearSVC': Pipeline([
        ('sc', StandardScaler()),
        ('m',  LinearSVC(C=1.0, max_iter=5000, random_state=42))
    ]),
}

print(f"\n{'Model':<20} {'CV AUC':>10} {'Test AUC':>10} {'Test Acc':>10}")
print("-" * 55)

for name, pipe in models.items():
    pipe.fit(X_tr, y_tr)
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring='roc_auc').mean()
    try:
        test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:,1])
    except AttributeError:
        df = pipe.named_steps['m'].decision_function(
            pipe.named_steps['sc'].transform(X_te))
        test_auc = roc_auc_score(y_te, df)
    test_acc = pipe.score(X_te, y_te)
    print(f"{name:<20} {cv_auc:>10.4f} {test_auc:>10.4f} {test_acc:>10.4f}")

# ── Tuning: Grid search over C and gamma ─────────────────────────────────────
print("\nGrid Search — RBF SVM:")
pipe_gs = Pipeline([
    ('sc', StandardScaler()),
    ('m',  SVC(kernel='rbf', probability=True, random_state=42))
])
param_grid = {
    'm__C':     [0.01, 0.1, 1, 10, 100],
    'm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
}
gs = GridSearchCV(pipe_gs, param_grid, cv=5, scoring='roc_auc',
                  n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)
print(f"  Best params: {gs.best_params_}")
print(f"  CV AUC:      {gs.best_score_:.4f}")
print(f"  Test AUC:    {roc_auc_score(y_te, gs.predict_proba(X_te)[:,1]):.4f}")

# ── SVM key sklearn parameters ────────────────────────────────────────────────
print("""
sklearn SVC Key Parameters:
─────────────────────────────────────────────────────────────────────
C             : Regularization (default=1.0). Tune via CV.
kernel        : 'linear', 'rbf', 'poly', 'sigmoid' (default='rbf')
gamma         : RBF/poly/sigmoid bandwidth.
                'scale' = 1/(n_features * X.var()) [default, recommended]
                'auto'  = 1/n_features
                float   = explicit value
degree        : Polynomial kernel degree (default=3)
coef0         : Polynomial/sigmoid kernel constant term (default=0)
probability   : Enable predict_proba (adds Platt scaling cost)
class_weight  : 'balanced' for imbalanced classes
cache_size    : Kernel cache in MB (increase for large datasets)
max_iter      : -1 = no limit; increase if ConvergenceWarning

LinearSVC vs SVC(kernel='linear'):
  LinearSVC: Uses liblinear (faster for large datasets)
  SVC:       Uses libsvm (slower but more features: probability, etc.)
─────────────────────────────────────────────────────────────────────
""")
```

---

## 9. When to Use SVMs

```python
print("""
SVM PRACTICAL GUIDE
══════════════════════════════════════════════════════════════════════

STRENGTHS:
  ✓ Theoretically well-founded (maximum margin, Mercer's theorem)
  ✓ Effective in high-dimensional spaces (text, genomics)
  ✓ Works well with small datasets (n < ~10,000)
  ✓ The kernel trick enables non-linear classification without
    explicitly computing features
  ✓ Memory efficient: only stores support vectors after training
  ✓ Robust to outliers that are far from the margin

WEAKNESSES:
  ✗ Doesn't scale well: O(n²) to O(n³) for kernel SVMs
    → Use RandomForest/XGBoost for n > 10,000
  ✗ Sensitive to feature scaling (MUST standardize)
  ✗ Requires careful tuning of C and γ
  ✗ SVC.predict_proba requires Platt scaling (extra cost)
  ✗ Not naturally probabilistic (no calibrated probabilities)
  ✗ Hard to interpret (no feature importance)
  ✗ Kernel choice and tuning requires expertise

WHEN TO CHOOSE SVM:
  ✓ Text classification (LinearSVC is very competitive)
  ✓ Image classification (before deep learning, SVMs with HOG dominated)
  ✓ Bioinformatics (n << d, e.g., 200 samples × 20,000 genes)
  ✓ When n < 10,000 and you suspect non-linear patterns
  ✓ When you need the maximum margin guarantee for interpretability

WHEN NOT TO USE SVM:
  ✗ Large datasets (n > 100,000): use XGBoost/LightGBM/LinearSVC
  ✗ Probabilistic predictions needed: use Logistic Regression or
    calibrated Random Forest
  ✗ Feature importance needed: use Random Forest or linear model
  ✗ Online/streaming learning: SGD-based models are better

SCALING ALTERNATIVES:
  For large datasets, replace kernel SVM with:
    • LinearSVC (linear kernel, liblinear)
    • SGDClassifier(loss='hinge') (stochastic gradient descent SVM)
    • RandomForestClassifier (usually better, always faster)
    • XGBoostClassifier (best accuracy on tabular data)
══════════════════════════════════════════════════════════════════════
""")
```

---

## 10. Case Study: Text Classification with Linear SVM

```python
# =============================================================================
# Text Classification with LinearSVC — the classic SVM application
# Dataset: 20 Newsgroups (text categorization)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load a 4-class subset ──────────────────────────────────────────────────
categories = ['sci.med', 'sci.space', 'comp.graphics', 'rec.sport.hockey']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                       shuffle=True, random_state=42,
                                       remove=('headers', 'footers', 'quotes'))
newsgroups_test  = fetch_20newsgroups(subset='test',  categories=categories,
                                       shuffle=True, random_state=42,
                                       remove=('headers', 'footers', 'quotes'))

X_train_raw = newsgroups_train.data
X_test_raw  = newsgroups_test.data
y_train     = newsgroups_train.target
y_test      = newsgroups_test.target

print(f"Train samples: {len(X_train_raw)}")
print(f"Test samples:  {len(X_test_raw)}")
print(f"Categories: {newsgroups_train.target_names}")

# ── 2. Why Linear SVM for Text ─────────────────────────────────────────────────
print("""
Why Linear SVM excels at text:
  1. Text features (TF-IDF) are very high-dimensional (d >> n)
  2. In high dimensions, data is usually linearly separable
  3. Linear SVM has competitive accuracy to kernel SVM but O(n) training
  4. The margin objective provides built-in regularization against noisy words
  5. LinearSVC uses liblinear (much faster than SVC for large d)
""")

# ── 3. TF-IDF + LinearSVC pipeline ───────────────────────────────────────────
svm_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features    = 50_000,
        ngram_range     = (1, 2),    # Unigrams and bigrams
        min_df          = 2,         # Ignore very rare terms
        max_df          = 0.95,      # Ignore very common terms
        sublinear_tf    = True,      # Apply log(1+tf) instead of tf
        strip_accents   = 'unicode',
        analyzer        = 'word',
        token_pattern   = r'\b[a-zA-Z]{2,}\b'  # Words only, min 2 chars
    )),
    ('clf', LinearSVC(C=1.0, max_iter=5000, random_state=42))
])

svm_pipe.fit(X_train_raw, y_train)
y_pred_svm = svm_pipe.predict(X_test_raw)

print(f"\nLinearSVC Test Accuracy: {(y_pred_svm == y_test).mean():.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_svm,
                            target_names=newsgroups_test.target_names))

# ── 4. Compare classifiers ────────────────────────────────────────────────────
classifiers = {
    'LinearSVC (C=1.0)':     Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50_000, ngram_range=(1,2),
                                   sublinear_tf=True)),
        ('clf',   LinearSVC(C=1.0, max_iter=5000, random_state=42))
    ]),
    'LinearSVC (C=0.1)':     Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50_000, ngram_range=(1,2),
                                   sublinear_tf=True)),
        ('clf',   LinearSVC(C=0.1, max_iter=5000, random_state=42))
    ]),
    'Logistic Regression':    Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50_000, ngram_range=(1,2),
                                   sublinear_tf=True)),
        ('clf',   LogisticRegression(C=1.0, max_iter=2000, random_state=42,
                                      n_jobs=-1))
    ]),
    'Naive Bayes':            Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50_000, sublinear_tf=True)),
        ('clf',   MultinomialNB(alpha=0.1))
    ]),
    'SGD (hinge = SVM)':     Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50_000, ngram_range=(1,2),
                                   sublinear_tf=True)),
        ('clf',   SGDClassifier(loss='hinge', alpha=1e-4, max_iter=100,
                                 random_state=42, n_jobs=-1))
    ]),
}

print(f"\n{'Classifier':<30} {'Test Acc':>10} {'Notes'}")
print("-" * 70)

for name, clf in classifiers.items():
    clf.fit(X_train_raw, y_train)
    acc = clf.score(X_test_raw, y_test)
    print(f"{name:<30} {acc:>10.4f}")

# ── 5. Most informative features per class ────────────────────────────────────
vectorizer = svm_pipe.named_steps['tfidf']
svm_clf    = svm_pipe.named_steps['clf']
feat_names = np.array(vectorizer.get_feature_names_out())

print("\nTop discriminative words per class (SVM coefficients):")
for i, cls_name in enumerate(newsgroups_train.target_names):
    coefs = svm_clf.coef_[i]
    top_pos = feat_names[np.argsort(coefs)[-10:]][::-1]
    top_neg = feat_names[np.argsort(coefs)[:10]]
    print(f"\n  {cls_name}:")
    print(f"    Most positive: {', '.join(top_pos)}")
    print(f"    Most negative: {', '.join(top_neg)}")

# ── 6. Confusion matrix ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_svm),
    display_labels=[c.split('.')[-1] for c in newsgroups_test.target_names]
)
disp.plot(ax=axes[0], colorbar=True, cmap='Blues')
axes[0].set_title(f'LinearSVC Confusion Matrix\nTest Acc = {(y_pred_svm==y_test).mean():.4f}')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

# C sensitivity
C_range = np.logspace(-3, 2, 20)
accs    = []
for C in C_range:
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50_000, sublinear_tf=True)),
        ('clf',   LinearSVC(C=C, max_iter=5000, random_state=42))
    ])
    pipe.fit(X_train_raw, y_train)
    accs.append(pipe.score(X_test_raw, y_test))

axes[1].semilogx(C_range, accs, 'o-', color='steelblue', lw=2.5)
axes[1].axvline(C_range[np.argmax(accs)], color='tomato', lw=2, linestyle='--',
                label=f'Best C = {C_range[np.argmax(accs)]:.3f}')
axes[1].set_xlabel('C (inverse regularization)')
axes[1].set_ylabel('Test Accuracy')
axes[1].set_title('LinearSVC: C Sensitivity\n20 Newsgroups (4 categories)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/09_text_classification.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 11. Summary

### ✅ Must-Remember Mental Models

**1. The SVM objective (primal):**
$$\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i\xi_i \quad\text{s.t.}\quad y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \geq 1-\xi_i$$
Minimizing $\|\mathbf{w}\|^2$ = maximizing the margin. $C$ controls the tradeoff.

**2. Support vectors define the solution completely.**
Only the handful of training points on or inside the margin matter. Delete the rest and get the same model. This makes SVMs memory-efficient.

**3. The kernel trick in one sentence:**
Replace all inner products $\langle\mathbf{x}^{(i)}, \mathbf{x}^{(j)}\rangle$ with $K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$, which implicitly computes a dot product in a (possibly infinite-dimensional) feature space — without ever computing the transformation explicitly.

**4. RBF kernel = infinite-dimensional feature space.**
$K(\mathbf{x},\mathbf{z}) = \exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)$. Parameter $\gamma$: large = local influence = complex boundary = overfit risk. Small = global influence = smooth boundary = underfit risk.

**5. C and γ are always tuned together for RBF SVM.**
Typical grid: $C \in \{0.01, 0.1, 1, 10, 100\}$, $\gamma \in \{0.001, 0.01, 0.1, 1, \text{'scale'}\}$.

**6. SVM as hinge loss minimization:**
$$\min \frac{1}{n}\sum_i \max(0, 1-y_i f(\mathbf{x}_i)) + \frac{\lambda}{2}\|\mathbf{w}\|^2$$
Connects SVMs to the broader family of regularized ERM. Replace hinge with log-loss → logistic regression.

**7. Feature scaling is mandatory.** SVMs are not scale-invariant. Always `StandardScaler` before SVC.

**8. For text and large datasets: `LinearSVC`.**
$O(n)$ training vs $O(n^2)$–$O(n^3)$ for kernel SVM. Competitive accuracy on high-dimensional sparse data.

**9. SVMs vs Random Forests/GBM on tabular data:**
Modern gradient boosting usually wins. SVMs shine on small, high-dimensional datasets (genomics, text, images without deep learning).

---

## 12. Exercises

1. **Prove the margin width**: Show that the distance between the two margin hyperplanes $\mathbf{w}^T\mathbf{x}+b=1$ and $\mathbf{w}^T\mathbf{x}+b=-1$ equals $2/\|\mathbf{w}\|$.

2. **Kernel matrix**: Show that the RBF kernel matrix $K$ where $K_{ij} = \exp(-\gamma\|\mathbf{x}^{(i)}-\mathbf{x}^{(j)}\|^2)$ is symmetric positive semi-definite for any $\gamma > 0$.

3. **Hinge loss gradient**: Derive the subgradient of the hinge loss $\max(0, 1-yf)$ with respect to $f$. Why is it not differentiable at $yf=1$? How does SGD handle this?

4. **C vs. λ relationship**: Show that the soft-margin SVM objective $(1/2)\|\mathbf{w}\|^2 + C\sum\xi_i$ is equivalent to $(1/n)\sum\xi_i + \frac{\lambda}{2}\|\mathbf{w}\|^2$ with $\lambda = 1/(nC)$.

5. **Implement a linear SVM with SGD**: Use `sklearn.linear_model.SGDClassifier(loss='hinge')` to train a linear SVM on the breast cancer dataset. Compare to `LinearSVC`. Show they converge to the same solution.

6. **Kernel comparison experiment**: On the circles and moons datasets, systematically compare: linear, polynomial (d=2,3), RBF (γ=0.1,1,10) kernels. For each: plot decision boundary, count support vectors, report train/test accuracy.

7. **Challenge — Implement the kernel perceptron**: The kernel perceptron uses the same update rule as the regular perceptron but with kernel evaluations. Implement it and compare to SVC on the circles dataset.

---

## 13. References

- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. — Original SVM paper.
- Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992). A training algorithm for optimal margin classifiers. *COLT*. — Kernel trick introduction.
- Platt, J. (1999). Probabilistic Outputs for SVMs. — Platt scaling for probability calibration.
- Hsieh, C. J., et al. (2008). A Dual Coordinate Descent Method for Large-scale Linear SVM. *ICML*. — liblinear.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *ESL*, Chapter 12: Support Vector Machines.

---

*Next: [Chapter 10 — K-Nearest Neighbors & Naive Bayes](10_knn_nb.md)*
