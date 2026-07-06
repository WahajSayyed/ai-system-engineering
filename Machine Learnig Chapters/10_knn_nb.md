# Chapter 10: K-Nearest Neighbors & Naive Bayes

> *"KNN is the simplest possible model: 'you are like your neighbors.' It has no training phase — just memory. Its simplicity is its greatest weakness and, sometimes, its greatest strength."*

> *"Naive Bayes is wrong by construction — real features are rarely independent. And yet it works remarkably well, especially for text. Understanding why teaches you something deep about the difference between a good probability model and a good classifier."*

---

## Table of Contents

1. [K-Nearest Neighbors](#1-k-nearest-neighbors)
   - 1.1 [The Algorithm](#11-the-algorithm)
   - 1.2 [The Geometry of KNN](#12-the-geometry-of-knn)
   - 1.3 [Distance Metrics](#13-distance-metrics)
   - 1.4 [Choosing K](#14-choosing-k)
   - 1.5 [The Curse of Dimensionality](#15-the-curse-of-dimensionality)
   - 1.6 [KNN for Regression](#16-knn-for-regression)
   - 1.7 [Efficient KNN: KD-Trees and Ball Trees](#17-efficient-knn-kd-trees-and-ball-trees)
   - 1.8 [KNN with Scikit-learn](#18-knn-with-scikit-learn)
2. [Naive Bayes](#2-naive-bayes)
   - 2.1 [Bayes' Theorem Revisited](#21-bayes-theorem-revisited)
   - 2.2 [The Naive Independence Assumption](#22-the-naive-independence-assumption)
   - 2.3 [Gaussian Naive Bayes](#23-gaussian-naive-bayes)
   - 2.4 [Multinomial Naive Bayes](#24-multinomial-naive-bayes)
   - 2.5 [Bernoulli Naive Bayes](#25-bernoulli-naive-bayes)
   - 2.6 [Laplace Smoothing](#26-laplace-smoothing)
   - 2.7 [Why Naive Bayes Works Despite Being Wrong](#27-why-naive-bayes-works-despite-being-wrong)
   - 2.8 [Naive Bayes with Scikit-learn](#28-naive-bayes-with-scikit-learn)
3. [Comparing KNN and Naive Bayes](#3-comparing-knn-and-naive-bayes)
4. [Case Study: Spam Detection](#4-case-study-spam-detection)
5. [Summary](#5-summary)
6. [Exercises](#6-exercises)
7. [References](#7-references)

---

## 1. K-Nearest Neighbors

### 1.1 The Algorithm

KNN makes a prediction for a new point $\mathbf{x}$ by:
1. Finding the $k$ training points closest to $\mathbf{x}$ (by some distance metric)
2. **Classification**: predict the majority class among the $k$ neighbors
3. **Regression**: predict the mean (or weighted mean) of the $k$ neighbors' target values

There is **no training phase**. KNN is a **lazy learner** — it stores all training data and does all computation at prediction time.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KNNScratch:
    """
    K-Nearest Neighbors from scratch.
    Supports classification (majority vote) and regression (mean).
    """

    def __init__(self, k=5, task='classification', distance='euclidean',
                 weights='uniform'):
        """
        Parameters
        ----------
        k          : number of neighbors
        task       : 'classification' or 'regression'
        distance   : 'euclidean', 'manhattan', 'chebyshev', 'minkowski'
        weights    : 'uniform' (equal vote) or 'distance' (inverse-distance weighting)
        """
        self.k        = k
        self.task     = task
        self.distance = distance
        self.weights  = weights

    def fit(self, X, y):
        """KNN 'training': just store the data."""
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y)
        return self

    def _compute_distances(self, x, p=2):
        """Compute distances from x to all training points."""
        diff = self.X_train_ - x
        if self.distance == 'euclidean':
            return np.sqrt((diff**2).sum(axis=1))
        elif self.distance == 'manhattan':
            return np.abs(diff).sum(axis=1)
        elif self.distance == 'chebyshev':
            return np.abs(diff).max(axis=1)
        else:  # minkowski
            return (np.abs(diff)**p).sum(axis=1)**(1/p)

    def _predict_one(self, x):
        dists    = self._compute_distances(x)
        knn_idx  = np.argsort(dists)[:self.k]
        knn_dists = dists[knn_idx]
        knn_y    = self.y_train_[knn_idx]

        if self.weights == 'distance':
            # Inverse distance weighting (handle zero-distance case)
            w = 1.0 / (knn_dists + 1e-10)
        else:
            w = np.ones(self.k)

        if self.task == 'classification':
            # Weighted majority vote
            classes = np.unique(knn_y)
            class_weights = {c: w[knn_y == c].sum() for c in classes}
            return max(class_weights, key=class_weights.get)
        else:
            # Weighted mean
            return np.average(knn_y, weights=w)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def predict_proba(self, X):
        """Class probability estimates (classification only)."""
        classes = np.unique(self.y_train_)
        probas  = []
        for x in X:
            dists    = self._compute_distances(x)
            knn_idx  = np.argsort(dists)[:self.k]
            knn_y    = self.y_train_[knn_idx]
            w        = (1.0/(dists[knn_idx]+1e-10)
                        if self.weights == 'distance' else np.ones(self.k))
            w        /= w.sum()
            p = np.array([w[knn_y == c].sum() for c in classes])
            probas.append(p)
        return np.array(probas)

    def score(self, X, y):
        y_pred = self.predict(X)
        if self.task == 'classification':
            return (y_pred == y).mean()
        else:
            ss_res = ((y - y_pred)**2).sum()
            ss_tot = ((y - y.mean())**2).sum()
            return 1 - ss_res / ss_tot


# ── Verify against sklearn ────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)
scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

knn_scratch = KNNScratch(k=5, task='classification')
knn_scratch.fit(X_tr_s, y_tr)

knn_sklearn = KNeighborsClassifier(n_neighbors=5)
knn_sklearn.fit(X_tr_s, y_tr)

print("KNN — Scratch vs Sklearn (Iris, k=5):")
print(f"  Scratch: {knn_scratch.score(X_te_s, y_te):.4f}")
print(f"  Sklearn: {knn_sklearn.score(X_te_s, y_te):.4f}")
```

---

### 1.2 The Geometry of KNN

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                            n_informative=2, class_sep=1.0, random_state=42)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
k_values  = [1, 3, 5, 10, 20, 50]

for ax, k in zip(axes.flatten(), k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_sc, y)

    xx, yy = np.meshgrid(np.linspace(X_sc[:,0].min()-0.3, X_sc[:,0].max()+0.3, 250),
                          np.linspace(X_sc[:,1].min()-0.3, X_sc[:,1].max()+0.3, 250))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.35, cmap='RdBu_r')
    ax.contour(xx, yy, Z, colors='black', linewidths=0.8, alpha=0.5)
    ax.scatter(X_sc[y==0, 0], X_sc[y==0, 1], s=18, color='#EF9A9A',
               edgecolors='k', linewidths=0.3, alpha=0.8)
    ax.scatter(X_sc[y==1, 0], X_sc[y==1, 1], s=18, color='#A5D6A7',
               edgecolors='k', linewidths=0.3, alpha=0.8)

    train_acc = knn.score(X_sc, y)
    ax.set_title(f'k={k}\nTrain Acc={train_acc:.3f}', fontsize=9)
    ax.tick_params(labelsize=7)

plt.suptitle('KNN Decision Boundaries at Different k Values\n'
             'k=1: extremely jagged (overfit). Large k: smooth (underfit).',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_knn_boundaries.png', dpi=130, bbox_inches='tight')
plt.show()

print("""
KNN Geometry:
  k=1: Voronoi diagram — each training point is its own region
       Perfect training accuracy (unless ties), but very high variance
  k=n: Predict the majority class everywhere — ignores features entirely
  
Decision boundary smoothness:
  Small k → irregular, complex boundary → low bias, high variance
  Large k → smooth, simple boundary → high bias, low variance
  
  This is another manifestation of the bias-variance tradeoff!
""")
```

---

### 1.3 Distance Metrics

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
DISTANCE METRICS FOR KNN
══════════════════════════════════════════════════════════════════════

Minkowski distance (generalization):
  d_p(x, z) = (Σᵢ |xᵢ - zᵢ|ᵖ)^(1/p)

  p=1: Manhattan (L1)   — sum of absolute differences
       Robust to outliers; useful for high dimensions
  p=2: Euclidean (L2)   — straight-line distance [default]
       Sensitive to scale; must standardize!
  p=∞: Chebyshev (L∞)  — maximum component difference

Other metrics:
  Cosine similarity:  1 - (x·z)/(||x|| ||z||)
    Measures ANGLE between vectors, not magnitude
    Standard for text (TF-IDF vectors)
    
  Hamming distance:   fraction of positions that differ
    For binary/categorical features
    
  Mahalanobis:  sqrt((x-z)ᵀ Σ⁻¹ (x-z))
    Accounts for feature correlations
    Equivalent to Euclidean after whitening transform

CRITICAL: KNN is sensitive to feature scale!
  Feature with range [0, 1000] will dominate distance over [0, 1].
  ALWAYS StandardScaler before KNN.
══════════════════════════════════════════════════════════════════════
""")

# ── Unit balls for different Minkowski metrics ─────────────────────────────────
theta  = np.linspace(0, 2*np.pi, 1000)
p_vals = [1, 1.5, 2, 3, 10, np.inf]
colors_p = ['#EF9A9A', '#FFCC80', '#A5D6A7', '#90CAF9', '#CE93D8', '#B0BEC5']

fig, ax = plt.subplots(figsize=(7, 7))

for p, color in zip(p_vals, colors_p):
    if p == np.inf:
        r = np.ones_like(theta)
    else:
        r = 1 / (np.abs(np.cos(theta))**p + np.abs(np.sin(theta))**p)**(1/p)
    label = f'p={p}' if p != np.inf else 'p=∞ (L∞/Chebyshev)'
    ax.plot(r*np.cos(theta), r*np.sin(theta), color=color, lw=2.5, label=label)

ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
ax.set_title('Unit Balls for Minkowski Metrics\n'
             '{x : d_p(0, x) ≤ 1}', fontsize=12, fontweight='bold')
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
plt.tight_layout()
plt.savefig('figures/10_distance_metrics.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Demonstrate scale sensitivity ─────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

cancer  = load_breast_cancer()
X_c, y_c = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, stratify=y_c,
                                            test_size=0.2, random_state=42)

results_scale = {}
for scaler_name, scaler_obj in [
    ('No scaling',       None),
    ('StandardScaler',   StandardScaler()),
    ('MinMaxScaler',     MinMaxScaler()),
]:
    if scaler_obj:
        X_tr_s = scaler_obj.fit_transform(X_tr)
        X_te_s = scaler_obj.transform(X_te)
    else:
        X_tr_s, X_te_s = X_tr, X_te

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr_s, y_tr)
    results_scale[scaler_name] = knn.score(X_te_s, y_te)

print("\nKNN Accuracy — Effect of Feature Scaling (breast cancer, k=5):")
for name, acc in results_scale.items():
    bar = '█' * int(acc * 40)
    print(f"  {name:<20}: {acc:.4f}  {bar}")
print("\n→ StandardScaler dramatically improves KNN performance.")
```

---

### 1.4 Choosing K

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

k_range = range(1, 51)
train_accs, cv_accs, test_accs = [], [], []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr_s, y_tr)
    train_accs.append(knn.score(X_tr_s, y_tr))
    test_accs.append( knn.score(X_te_s, y_te))
    cv = cross_val_score(knn, X_tr_s, y_tr, cv=5, scoring='accuracy')
    cv_accs.append(cv.mean())

best_k_cv   = list(k_range)[np.argmax(cv_accs)]
best_k_test = list(k_range)[np.argmax(test_accs)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(k_range, train_accs, 'o-', color='steelblue', lw=2, markersize=3, label='Train')
ax.plot(k_range, cv_accs,    's-', color='green',     lw=2, markersize=3, label='CV (5-fold)')
ax.plot(k_range, test_accs,  '^-', color='tomato',    lw=2, markersize=3, label='Test')
ax.axvline(best_k_cv, color='black', lw=2, linestyle='--',
           label=f'Best CV k = {best_k_cv}')
ax.set_xlabel('k (Number of Neighbors)')
ax.set_ylabel('Accuracy')
ax.set_title(f'KNN: Accuracy vs k\n'
             f'Best CV k={best_k_cv} (acc={max(cv_accs):.4f})')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Effect of distance weighting
ax2 = axes[1]
for weights, color, label in [('uniform', 'steelblue', 'Uniform weights'),
                                ('distance', 'tomato', 'Distance weights')]:
    cv_w = []
    for k in k_range:
        knn_w = KNeighborsClassifier(n_neighbors=k, weights=weights)
        cv_w.append(cross_val_score(knn_w, X_tr_s, y_tr, cv=5).mean())
    ax2.plot(k_range, cv_w, 'o-', color=color, lw=2, markersize=3, label=label)

ax2.set_xlabel('k'); ax2.set_ylabel('CV Accuracy')
ax2.set_title('Uniform vs Distance-Weighted KNN\n'
              'Distance weighting helps when k is larger')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.suptitle('Choosing k: Cross-Validation and Distance Weighting',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_knn_k_selection.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"\nBest k by CV:   {best_k_cv}  (CV acc = {max(cv_accs):.4f})")
print(f"Best k by test: {best_k_test}  (Test acc = {max(test_accs):.4f})")
print(f"\nTips for choosing k:")
print(f"  - Rule of thumb: k = sqrt(n_train) = {int(np.sqrt(len(X_tr)))}")
print(f"  - Always use odd k for binary classification (breaks ties)")
print(f"  - Always use cross-validation to choose k")
print(f"  - Distance weighting often improves performance for larger k")
```

---

### 1.5 The Curse of Dimensionality

This is the most important limitation of KNN.

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
THE CURSE OF DIMENSIONALITY
══════════════════════════════════════════════════════════════════════

In high dimensions, distances lose their meaning.

Thought experiment: n=1000 uniformly distributed points in d dimensions.
  How large must a hypercube's side length l be to capture 10% of the data?

  Volume of hypercube: l^d
  Volume of unit hypercube: 1
  To capture fraction f: l^d = f  →  l = f^(1/d)
""")

fractions = [0.01, 0.05, 0.10, 0.25]
dims      = np.arange(1, 101)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
for f, color in zip(fractions, ['#EF9A9A','#FFCC80','#A5D6A7','#90CAF9']):
    side_lengths = f**(1/dims)
    ax.plot(dims, side_lengths, lw=2.5, color=color,
            label=f'f = {int(f*100)}% of data')
ax.set_xlabel('Dimensions d'); ax.set_ylabel('Required side length l = f^(1/d)')
ax.set_title('Side Length to Capture f% of Data\n'
             'As d grows, must search almost the ENTIRE space')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

# Distance concentration: all points become equidistant
ax2 = axes[1]
rng = np.random.default_rng(42)
n_pts = 1000
dims_check = [2, 5, 10, 20, 50, 100, 200, 500]
dist_ratios = []   # (max_dist - min_dist) / min_dist

for d in dims_check:
    X_d = rng.uniform(0, 1, (n_pts, d))
    ref = X_d[0]
    dists = np.sqrt(((X_d[1:] - ref)**2).sum(axis=1))
    ratio = (dists.max() - dists.min()) / (dists.min() + 1e-10)
    dist_ratios.append(ratio)

ax2.semilogx(dims_check, dist_ratios, 'o-', color='steelblue', lw=2.5)
ax2.set_xlabel('Dimensions d')
ax2.set_ylabel('(max_dist - min_dist) / min_dist')
ax2.set_title('Distance Concentration\n'
              'All distances become similar → KNN neighbors meaningless')
ax2.grid(True, alpha=0.3)

# KNN accuracy degrades in high dimensions
ax3 = axes[2]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

dims_test   = [2, 5, 10, 20, 50, 100, 200]
knn_accs    = []
lr_accs     = []

for d in dims_test:
    X_d  = rng.standard_normal((500, d))
    # Only first 2 dimensions are informative
    y_d  = (X_d[:, 0] + X_d[:, 1] > 0).astype(int)

    sc   = StandardScaler()
    X_ds = sc.fit_transform(X_d)

    knn_cv = cross_val_score(
        KNeighborsClassifier(n_neighbors=5), X_ds, y_d, cv=5
    ).mean()
    lr_cv  = cross_val_score(
        LogisticRegression(max_iter=500), X_ds, y_d, cv=5
    ).mean()
    knn_accs.append(knn_cv)
    lr_accs.append(lr_cv)

ax3.semilogx(dims_test, knn_accs, 'o-', color='steelblue', lw=2.5, label='KNN (k=5)')
ax3.semilogx(dims_test, lr_accs,  's-', color='tomato',    lw=2.5, label='Logistic Reg.')
ax3.axhline(0.5, color='gray', lw=1, linestyle='--', label='Random')
ax3.set_xlabel('Total dimensions (only 2 informative)')
ax3.set_ylabel('CV Accuracy')
ax3.set_title('KNN vs Logistic Reg in High Dimensions\n'
              'KNN degrades; linear model stable')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

plt.suptitle('The Curse of Dimensionality — Why KNN Fails in High Dimensions',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_curse_of_dimensionality.png', dpi=130, bbox_inches='tight')
plt.show()

print("""
Practical implications:
  • KNN works well when d ≤ 20 and features are informative
  • For high-dimensional data (d >> 100): use linear models or tree ensembles
  • Dimensionality reduction (PCA, UMAP) before KNN can help
  • Feature selection removes irrelevant dimensions that hurt distance metrics
""")
```

---

### 1.6 KNN for Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

rng  = np.random.default_rng(42)
n    = 150
X_r  = np.sort(rng.uniform(0, 8, n)).reshape(-1, 1)
y_r  = np.sin(X_r.ravel()) * 2 + 0.5*X_r.ravel() + rng.normal(0, 0.4, n)

X_plot = np.linspace(0, 8, 400).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, k in zip(axes, [1, 5, 20]):
    knn_r = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn_r.fit(X_r, y_r)
    y_pred = knn_r.predict(X_plot)

    ax.scatter(X_r.ravel(), y_r, s=15, color='steelblue', alpha=0.6, label='Data')
    ax.plot(X_plot.ravel(), y_pred, 'tomato', lw=2.5, label=f'KNN k={k}')
    ax.plot(X_plot.ravel(), np.sin(X_plot.ravel())*2 + 0.5*X_plot.ravel(),
            'k--', lw=1.5, alpha=0.4, label='True')

    cv_r2 = cross_val_score(knn_r, X_r, y_r, cv=5, scoring='r2').mean()
    ax.set_title(f'k={k} (distance weighting)\nCV R²={cv_r2:.4f}')
    ax.set_xlabel('x'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle('KNN Regression at Different k Values\n'
             'k=1: interpolates through every point (overfit)\n'
             'Large k: over-smoothed (underfit)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_knn_regression.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 1.7 Efficient KNN: KD-Trees and Ball Trees

Naive KNN is $O(nd)$ per query (compute distance to all $n$ training points). For large $n$, this is too slow.

```python
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.datasets import make_classification

print("""
KNN SEARCH ALGORITHMS
══════════════════════════════════════════════════════════════════════

Brute Force: O(n·d) per query — compute all distances
  Good when: n small, d large (high-dimensional)

KD-Tree: Binary space partitioning tree
  Build: O(n log n)
  Query: O(log n) average, O(n) worst case
  Good when: d ≤ ~30
  Algorithm='kd_tree' in sklearn

Ball Tree: Hypersphere-based partitioning
  Build: O(n log n)
  Query: O(log n) average
  Good when: d > 30, non-Euclidean metrics
  Algorithm='ball_tree' in sklearn

Rule of thumb in sklearn:
  n < 30:           'brute'
  d < 20, n large:  'kd_tree'
  d > 20 or non-L2: 'ball_tree'
  Default 'auto':   sklearn decides automatically
══════════════════════════════════════════════════════════════════════
""")

# Speed comparison
results = {}
for n_train in [1_000, 5_000, 10_000]:
    X_sp, y_sp = make_classification(n_samples=n_train+200, n_features=10,
                                      random_state=42)
    X_tr_sp, X_te_sp = X_sp[:n_train], X_sp[n_train:]
    y_tr_sp, y_te_sp = y_sp[:n_train], y_sp[n_train:]

    for algo in ['brute', 'kd_tree', 'ball_tree']:
        knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
        knn.fit(X_tr_sp, y_tr_sp)
        start = time.perf_counter()
        knn.predict(X_te_sp)
        elapsed = time.perf_counter() - start
        results[(n_train, algo)] = elapsed * 1000   # ms

print(f"\nQuery time (ms) for 200 test samples, d=10:")
print(f"{'n_train':<12} {'Brute':>12} {'KD-Tree':>12} {'Ball-Tree':>12}")
print("-" * 52)
for n in [1_000, 5_000, 10_000]:
    print(f"{n:<12} "
          f"{results[(n,'brute')]:>12.3f} "
          f"{results[(n,'kd_tree')]:>12.3f} "
          f"{results[(n,'ball_tree')]:>12.3f}")
```

---

### 1.8 KNN with Scikit-learn

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# Full pipeline: scale → KNN
pipe_knn = Pipeline([
    ('sc',  StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 10, 15, 20, 25, 30],
    'knn__weights':     ['uniform', 'distance'],
    'knn__metric':      ['euclidean', 'manhattan', 'chebyshev'],
}

gs = GridSearchCV(pipe_knn, param_grid, cv=5, scoring='roc_auc',
                  n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)

print("KNN Grid Search — Breast Cancer:")
print(f"  Best params: {gs.best_params_}")
print(f"  CV AUC:      {gs.best_score_:.4f}")
print(f"  Test AUC:    {roc_auc_score(y_te, gs.predict_proba(X_te)[:,1]):.4f}")
print(f"  Test Acc:    {gs.score(X_te, y_te):.4f}")
print(f"\n{classification_report(y_te, gs.predict(X_te), target_names=cancer.target_names)}")
```

---

## 2. Naive Bayes

### 2.1 Bayes' Theorem Revisited

For classification, we want: $P(\text{class} = k | \mathbf{x})$

By Bayes' theorem:
$$P(y=k | \mathbf{x}) = \frac{P(\mathbf{x} | y=k) \cdot P(y=k)}{P(\mathbf{x})}$$

- $P(y=k)$: **prior** — class frequency in training data
- $P(\mathbf{x}|y=k)$: **likelihood** — how probable is this feature vector given class $k$?
- $P(\mathbf{x})$: **evidence** — constant for all classes (normalizer)

We predict:
$$\hat{y} = \arg\max_k P(y=k|\mathbf{x}) = \arg\max_k P(\mathbf{x}|y=k) \cdot P(y=k)$$

---

### 2.2 The Naive Independence Assumption

Computing $P(\mathbf{x}|y=k)$ for a $d$-dimensional feature vector requires estimating a joint distribution over $d$ variables — exponentially complex.

**The "naive" assumption**: features are **conditionally independent** given the class:

$$P(\mathbf{x}|y=k) = \prod_{j=1}^d P(x_j | y=k)$$

This simplifies the problem dramatically: we only need to estimate $d$ univariate distributions instead of one $d$-variate joint.

```python
import numpy as np

print("""
WHY "NAIVE" INDEPENDENCE MAKES THE PROBLEM TRACTABLE
════════════════════════════════════════════════════════════════════

Without independence: estimate P(x₁, x₂, ..., xd | y=k)
  For d binary features: 2^d - 1 parameters per class
  For d=30 features:     2^30 - 1 ≈ 1 BILLION parameters per class
  → Exponential in d. Completely intractable.

With independence: P(x₁,...,xd | y=k) = Π P(xⱼ | y=k)
  For d binary features: d parameters per class
  For d=30 features:     30 parameters per class
  → Linear in d. Always tractable.

The independence assumption is WRONG for almost all real datasets.
  Example: "buy" and "free" in email are positively correlated.
  The model treats them as independent — which is factually false.

Yet Naive Bayes often works well anyway. Why?

For CLASSIFICATION (not probability estimation), we only need:
  P(y=0|x) vs P(y=1|x) — the RANKING of classes, not exact values.
  
  Even if P(y=k|x) is badly calibrated due to the wrong independence
  assumption, the class with the HIGHEST probability is often still
  correctly identified.
  
  Analogy: You can rank people by height without knowing exact heights.
════════════════════════════════════════════════════════════════════
""")
```

---

### 2.3 Gaussian Naive Bayes

For **continuous features**, assume each feature $x_j$ given class $k$ follows a Gaussian distribution:

$$P(x_j | y=k) = \frac{1}{\sqrt{2\pi\sigma_{jk}^2}} \exp\left(-\frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2}\right)$$

Parameters learned from data: $\mu_{jk} = $ mean of feature $j$ in class $k$; $\sigma_{jk}^2 = $ variance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class GaussianNBScratch:
    """
    Gaussian Naive Bayes from scratch.
    For each class k and feature j:
      Estimate μ_jk = mean, σ²_jk = variance from training data.
    Predict: argmax_k [log P(y=k) + Σⱼ log N(xⱼ; μ_jk, σ²_jk)]
    """

    def fit(self, X, y):
        self.classes_    = np.unique(y)
        self.n_classes_  = len(self.classes_)
        n, d             = X.shape

        # Prior: log P(y=k) = log(n_k / n)
        self.log_prior_  = np.array([
            np.log(np.sum(y == k) / n) for k in self.classes_
        ])

        # Likelihood parameters: mean and variance per class per feature
        self.theta_      = np.zeros((self.n_classes_, d))   # means
        self.sigma_      = np.zeros((self.n_classes_, d))   # variances

        for i, k in enumerate(self.classes_):
            X_k               = X[y == k]
            self.theta_[i]    = X_k.mean(axis=0)
            self.sigma_[i]    = X_k.var(axis=0) + 1e-9   # Add epsilon for stability

        return self

    def _log_likelihood(self, X):
        """
        Compute log P(x | y=k) for each class k.
        Returns array of shape (n_samples, n_classes).
        """
        n = len(X)
        log_liks = np.zeros((n, self.n_classes_))

        for i in range(self.n_classes_):
            # log N(xⱼ; μ_jk, σ²_jk) = -0.5*log(2πσ²) - (x-μ)²/(2σ²)
            log_liks[:, i] = np.sum(
                -0.5 * np.log(2 * np.pi * self.sigma_[i])
                - 0.5 * ((X - self.theta_[i])**2) / self.sigma_[i],
                axis=1
            )
        return log_liks

    def predict_log_proba(self, X):
        """Log posterior: log P(y=k|x) ∝ log P(y=k) + log P(x|y=k)"""
        log_post = self._log_likelihood(X) + self.log_prior_
        # Normalize (log-sum-exp trick)
        log_norm = np.log(np.exp(log_post - log_post.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
        return log_post - log_norm - log_post.max(axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._log_likelihood(X) + self.log_prior_, axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ── Test on Iris ──────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

gnb_scratch  = GaussianNBScratch().fit(X_tr, y_tr)
gnb_sklearn  = GaussianNB().fit(X_tr, y_tr)

print("Gaussian Naive Bayes — Scratch vs Sklearn:")
print(f"  Scratch: {gnb_scratch.score(X_te, y_te):.4f}")
print(f"  Sklearn: {gnb_sklearn.score(X_te, y_te):.4f}")

# ── Learned parameters ────────────────────────────────────────────────────────
print("\nLearned parameters (μ per class per feature):")
print(f"{'Feature':<25}", end='')
for cls in iris.target_names:
    print(f" {cls:>12}", end='')
print()
print("-" * 65)
for j, feat in enumerate(iris.feature_names):
    print(f"{feat:<25}", end='')
    for i in range(3):
        print(f" {gnb_scratch.theta_[i,j]:>12.4f}", end='')
    print()

# ── Visualize learned Gaussians ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
colors_cls = ['tomato', 'steelblue', 'green']

for ax, (feat_idx, feat_name) in zip(axes.flatten(),
                                      enumerate(iris.feature_names)):
    x_range = np.linspace(X[:,feat_idx].min()-0.5,
                          X[:,feat_idx].max()+0.5, 300)
    for i, (cls_name, color) in enumerate(zip(iris.target_names, colors_cls)):
        mu    = gnb_scratch.theta_[i, feat_idx]
        sigma = np.sqrt(gnb_scratch.sigma_[i, feat_idx])
        pdf   = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range-mu)/sigma)**2)
        ax.plot(x_range, pdf, color=color, lw=2.5,
                label=f'{cls_name} μ={mu:.2f}, σ={sigma:.2f}')
        # Actual histogram
        data_cls = X_tr[y_tr==i, feat_idx]
        ax.hist(data_cls, bins=15, density=True, color=color,
                alpha=0.25, edgecolor='white')
    ax.set_title(feat_name); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle('Gaussian Naive Bayes: Learned Per-Class Distributions\n'
             'Histogram = data, Curve = fitted Gaussian',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_gnb_distributions.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 2.4 Multinomial Naive Bayes

Used for **discrete count features** — the standard model for **text classification** with word counts or TF-IDF.

$$P(x_j | y=k) = \frac{(\text{count of feature } j \text{ in class } k) + \alpha}{\sum_{j'} (\text{count of feature } j' \text{ in class } k) + \alpha \cdot d}$$

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class MultinomialNBScratch:
    """
    Multinomial Naive Bayes from scratch.
    Designed for word count features.
    """

    def __init__(self, alpha=1.0):
        """alpha: Laplace smoothing parameter"""
        self.alpha = alpha

    def fit(self, X, y):
        """
        X: (n_samples, n_features) non-negative feature matrix
        y: (n_samples,) class labels
        """
        n, d          = X.shape
        self.classes_ = np.unique(y)
        K             = len(self.classes_)

        # Prior: P(y=k) = n_k / n
        self.log_prior_ = np.array([
            np.log(np.sum(y == k) / n) for k in self.classes_
        ])

        # Likelihood: smoothed feature counts per class
        # P(xⱼ | y=k) = (N_kj + α) / (N_k + α·d)
        # where N_kj = total count of feature j in class k
        self.log_likelihood_ = np.zeros((K, d))
        for i, k in enumerate(self.classes_):
            X_k    = X[y == k]
            counts = X_k.sum(axis=0) + self.alpha   # N_kj + α
            total  = counts.sum()                    # N_k + α·d
            self.log_likelihood_[i] = np.log(counts / total)

        return self

    def predict_log_proba(self, X):
        # log P(y=k|x) ∝ log P(y=k) + Σⱼ xⱼ log P(xⱼ|y=k)
        return X @ self.log_likelihood_.T + self.log_prior_

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X, y):
        return (self.predict(X) == y).mean()


# ── Quick test on 20 Newsgroups (4 categories) ────────────────────────────────
cats   = ['sci.med', 'sci.space', 'comp.graphics', 'rec.sport.hockey']
train  = fetch_20newsgroups(subset='train', categories=cats,
                             remove=('headers', 'footers', 'quotes'))
test   = fetch_20newsgroups(subset='test',  categories=cats,
                             remove=('headers', 'footers', 'quotes'))

# TF-IDF features
tfidf  = TfidfVectorizer(max_features=30000, sublinear_tf=True)
X_tr_t = tfidf.fit_transform(train.data).toarray()
X_te_t = tfidf.transform(test.data).toarray()

# Our implementation vs sklearn
mnb_scratch = MultinomialNBScratch(alpha=1.0)
mnb_scratch.fit(X_tr_t, train.target)

mnb_sk = MultinomialNB(alpha=1.0)
mnb_sk.fit(X_tr_t, train.target)

print("Multinomial Naive Bayes on 20 Newsgroups (4 categories):")
print(f"  Scratch:  {mnb_scratch.score(X_te_t, test.target):.4f}")
print(f"  Sklearn:  {mnb_sk.score(X_te_t, test.target):.4f}")

# Complement NB — often better for text
cnb = ComplementNB(alpha=1.0)
cnb.fit(X_tr_t, train.target)
print(f"  ComplementNB: {cnb.score(X_te_t, test.target):.4f}")
print("  (Complement NB: trains on the COMPLEMENT of each class → handles imbalance)")
```

---

### 2.5 Bernoulli Naive Bayes

For **binary features** (feature present/absent, e.g., binary bag of words):

$$P(x_j | y=k) = p_{jk}^{x_j} \cdot (1 - p_{jk})^{1-x_j}$$

Where $p_{jk} = P(x_j = 1 | y=k)$.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

# Binary vectorizer: 1 if word appears, 0 otherwise
binary_vec = CountVectorizer(max_features=30000, binary=True)
X_tr_bin   = binary_vec.fit_transform(train.data)
X_te_bin   = binary_vec.transform(test.data)

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_tr_bin, train.target)

print(f"\nBernoulli NB on 20 Newsgroups:")
print(f"  Test accuracy: {bnb.score(X_te_bin, test.target):.4f}")
print(f"  (vs Multinomial NB: {mnb_sk.score(X_te_t, test.target):.4f})")
print(f"  Bernoulli uses binary features; Multinomial uses counts")
```

---

### 2.6 Laplace Smoothing

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
LAPLACE SMOOTHING — WHY IT'S ESSENTIAL
════════════════════════════════════════════════════════════════════

Problem: What if a word never appears in class k's training data?
  P(word="unicorn" | y=spam) = 0/n_spam = 0

Then:  P(email | y=spam) = Π P(wᵢ | y=spam) = 0  (one zero kills everything)

The model refuses to classify any email with an unseen word as spam,
regardless of all other evidence. This is called the zero-frequency problem.

Solution: Laplace smoothing (add-α smoothing)
  P(xⱼ | y=k) = (count(xⱼ, k) + α) / (total_count(k) + α·d)

  α=1: Laplace smoothing (add 1 to every count)
  α<1: Lidstone smoothing (softer)
  α=0: MLE (no smoothing — zero-frequency problem)

Effect of α:
  Large α → stronger smoothing → all words get similar probability
             → less distinction between classes → underfitting
  Small α → weaker smoothing → rare words matter more
             → can overfit to training vocabulary
  
  Cross-validate α like any other hyperparameter!
════════════════════════════════════════════════════════════════════
""")

# Demonstrate alpha effect
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
accs   = []

cv_vec = CountVectorizer(max_features=20000)
X_tr_c = cv_vec.fit_transform(train.data)
X_te_c = cv_vec.transform(test.data)

for alpha in alphas:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_tr_c, train.target)
    accs.append(mnb.score(X_te_c, test.target))

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(alphas, accs, 'o-', color='steelblue', lw=2.5, markersize=8)
ax.set_xlabel('Laplace smoothing parameter α')
ax.set_ylabel('Test Accuracy')
ax.set_title('Effect of Laplace Smoothing (α) on Multinomial NB\n'
             '20 Newsgroups — 4 categories')
ax.axvline(alphas[np.argmax(accs)], color='tomato', lw=2, linestyle='--',
           label=f'Best α = {alphas[np.argmax(accs)]}')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/10_laplace_smoothing.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"\nBest alpha: {alphas[np.argmax(accs)]}")
print(f"Best test accuracy: {max(accs):.4f}")
```

---

### 2.7 Why Naive Bayes Works Despite Being Wrong

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)

print("""
WHY NAIVE BAYES WORKS: THE GENERATIVE VS DISCRIMINATIVE VIEW
════════════════════════════════════════════════════════════════════

Two paradigms for learning P(y|x):

GENERATIVE MODEL (Naive Bayes):
  Explicitly models P(x|y) and P(y), then uses Bayes' theorem.
  Pros: Works with few samples (strong assumption reduces variance)
        Handles missing features naturally
        Easy to update incrementally (online learning)
  Cons: Model is wrong (features not independent) → miscalibrated probabilities

DISCRIMINATIVE MODEL (Logistic Regression):
  Directly models P(y|x) without modeling P(x|y).
  Pros: Doesn't need to model the distribution of features
        More accurate when n is large
        Better calibrated probabilities
  Cons: Needs more data; can't handle missing values as easily

Key insight (Ng & Jordan, 2002):
  Naive Bayes converges to its optimal performance with O(log n) samples.
  Logistic Regression needs O(n) samples to converge.
  
  → Naive Bayes is BETTER with very few samples.
  → Logistic Regression is BETTER with many samples.
  
  The crossover point is typically around n = 30-100 for simple problems.
════════════════════════════════════════════════════════════════════
""")

# Empirically verify: NB better with few samples
n_features = 10
n_total    = 1000
X, y = make_classification(n_samples=n_total, n_features=n_features,
                            n_informative=5, random_state=42)

sample_sizes = [10, 20, 50, 100, 200, 500, 1000]
gnb_accs, lr_accs = [], []

for n_s in sample_sizes:
    gnb_cv = []
    lr_cv  = []
    for seed in range(30):
        rng_s = np.random.default_rng(seed)
        idx   = rng_s.choice(n_total, n_s, replace=False)
        X_s, y_s = X[idx], y[idx]
        sc    = StandardScaler()
        X_ss  = sc.fit_transform(X_s)
        # 3-fold CV (or LOO for very small n)
        if n_s >= 6:
            cv_folds = min(3, n_s // 2)
            gnb_cv.append(cross_val_score(GaussianNB(), X_ss, y_s, cv=cv_folds).mean())
            lr_cv.append(cross_val_score(
                LogisticRegression(max_iter=500, C=1.0), X_ss, y_s, cv=cv_folds).mean())
    gnb_accs.append(np.mean(gnb_cv))
    lr_accs.append(np.mean(lr_cv))

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(sample_sizes, gnb_accs, 'o-', color='steelblue', lw=2.5,
            label='Gaussian Naive Bayes (generative)')
ax.semilogx(sample_sizes, lr_accs,  's-', color='tomato',    lw=2.5,
            label='Logistic Regression (discriminative)')
ax.set_xlabel('Training set size n')
ax.set_ylabel('CV Accuracy')
ax.set_title('Generative vs Discriminative:\nNaive Bayes Wins on Small Data,\n'
             'Logistic Regression Wins with More Data')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/10_nb_vs_lr.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 2.8 Naive Bayes with Scikit-learn

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

print("""
Naive Bayes Variants — Quick Reference:
──────────────────────────────────────────────────────────────────
GaussianNB:    Continuous features, assumes Gaussian distribution
               No hyperparameters (except var_smoothing ≈ 1e-9)
               Use for: real-valued features, general classification

MultinomialNB: Non-negative integer features (word counts)
               alpha: Laplace smoothing (default=1.0)
               Use for: text classification, count data

BernoulliNB:   Binary features (word presence/absence)
               alpha: Laplace smoothing (default=1.0)
               binarize: threshold to convert continuous to binary
               Use for: binary bag-of-words, boolean features

ComplementNB:  Variant of MultinomialNB trained on complement classes
               Often better for imbalanced text classification
               alpha: same as MultinomialNB
──────────────────────────────────────────────────────────────────
""")

# GaussianNB — works directly on continuous features, no scaling needed
# (it learns the distribution parameters itself)
iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_tr, y_tr)
print(f"GaussianNB on Iris:")
print(f"  Test acc: {gnb.score(X_te, y_te):.4f}")
print(f"  Class priors: {gnb.class_prior_.round(3)}")
print(f"  Theta (means) shape: {gnb.theta_.shape}")

# Incremental learning: partial_fit for streaming data
print("\nGaussianNB partial_fit (streaming/incremental learning):")
gnb_incremental = GaussianNB()
batch_size = 20
for i in range(0, len(X_tr), batch_size):
    X_batch = X_tr[i:i+batch_size]
    y_batch = y_tr[i:i+batch_size]
    gnb_incremental.partial_fit(X_batch, y_batch, classes=np.unique(y))
print(f"  After streaming all {len(X_tr)} samples in batches of {batch_size}:")
print(f"  Test acc: {gnb_incremental.score(X_te, y_te):.4f}")
print(f"  (vs batch: {gnb.score(X_te, y_te):.4f})")
```

---

## 3. Comparing KNN and Naive Bayes

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(42)

# Different scenarios
scenarios = {
    'Gaussian clusters\n(NB assumption met)': make_blobs(
        n_samples=300, centers=3, random_state=42
    ),
    'Non-linear boundary\n(neither assumes)': make_moons(
        n_samples=300, noise=0.2, random_state=42
    ),
    'High-dimensional\n(d=50, 5 informative)': make_classification(
        n_samples=300, n_features=50, n_informative=5, random_state=42
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (scenario_name, (X_sc, y_sc)) in zip(axes, scenarios.items()):
    scaler = StandardScaler()
    X_scs  = scaler.fit_transform(X_sc)

    knn_cv = cross_val_score(KNeighborsClassifier(n_neighbors=7), X_scs, y_sc, cv=5).mean()
    gnb_cv = cross_val_score(GaussianNB(), X_scs, y_sc, cv=5).mean()

    methods = ['KNN (k=7)', 'Gaussian NB']
    accs    = [knn_cv, gnb_cv]
    colors  = ['steelblue', 'tomato']
    bars    = ax.bar(methods, accs, color=colors, alpha=0.8)
    ax.bar_label(bars, fmt='%.4f', padding=4)
    ax.set_ylabel('CV Accuracy')
    ax.set_title(f'{scenario_name}')
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis='y', alpha=0.3)
    winner = methods[np.argmax(accs)]
    ax.set_xlabel(f'Winner: {winner}', color='green', fontweight='bold')

plt.suptitle('KNN vs Naive Bayes: Scenario-Dependent Performance',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_knn_vs_nb.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
KNN vs Naive Bayes — When to Use Which:
─────────────────────────────────────────────────────────────────────
              KNN                        Naive Bayes
─────────────────────────────────────────────────────────────────────
Training      No training (lazy)         Fast (count statistics)
Prediction    Slow (O(n) per query)      Fast (O(d·K) per query)
Scaling       REQUIRED                   Not needed (learns scale)
High dim      Poor (CoD)                 Better (linear in d)
Text data     Terrible                   Excellent
Small data    Good (non-parametric)      Better (strong prior)
Large data    Slow at predict time       Fast always
Missing values Need imputation           Handle naturally (skip feature)
Interpretability  None                  Good (feature probabilities)
─────────────────────────────────────────────────────────────────────
Rule of thumb:
  Text/high-dimensional sparse: MultinomialNB
  Small n, continuous features: GaussianNB or KNN
  Medium n, non-linear: KNN with k tuned by CV
  Large n: neither (use Random Forest or GBM)
─────────────────────────────────────────────────────────────────────
""")
```

---

## 4. Case Study: Spam Detection

```python
# =============================================================================
# Email Spam Detection: Naive Bayes vs LinearSVC vs Logistic Regression
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# ── Simulate spam classification using newsgroups ────────────────────────────
# Use 'talk.politics.misc' and 'talk.religion.misc' as "spam" (unwanted)
# Use 'sci.space' and 'comp.graphics' as "ham" (wanted)
SPAM_CATS = ['talk.politics.misc', 'talk.religion.misc']
HAM_CATS  = ['sci.space', 'comp.graphics']
ALL_CATS  = HAM_CATS + SPAM_CATS

train_data = fetch_20newsgroups(subset='train', categories=ALL_CATS,
                                 remove=('headers','footers','quotes'))
test_data  = fetch_20newsgroups(subset='test',  categories=ALL_CATS,
                                 remove=('headers','footers','quotes'))

# Binary: spam=1, ham=0
y_tr_bin = (train_data.target >= 2).astype(int)
y_te_bin = (test_data.target  >= 2).astype(int)

print(f"Training: {len(y_tr_bin)} emails ({y_tr_bin.sum()} spam, {(1-y_tr_bin).sum()} ham)")
print(f"Testing:  {len(y_te_bin)} emails ({y_te_bin.sum()} spam, {(1-y_te_bin).sum()} ham)")
print(f"Spam rate (train): {y_tr_bin.mean():.2%}")

# ── Pipeline comparison ───────────────────────────────────────────────────────
classifiers = {
    'MultinomialNB (α=0.1)': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=30000, sublinear_tf=True,
                                   ngram_range=(1,2))),
        ('clf',   MultinomialNB(alpha=0.1))
    ]),
    'ComplementNB (α=0.1)': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=30000, sublinear_tf=True,
                                   ngram_range=(1,2))),
        ('clf',   ComplementNB(alpha=0.1))
    ]),
    'BernoulliNB (binary)': Pipeline([
        ('tfidf', CountVectorizer(max_features=30000, binary=True)),
        ('clf',   BernoulliNB(alpha=1.0))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=30000, sublinear_tf=True,
                                   ngram_range=(1,2))),
        ('clf',   LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1))
    ]),
    'LinearSVC': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=30000, sublinear_tf=True,
                                   ngram_range=(1,2))),
        ('clf',   LinearSVC(C=1.0, max_iter=5000))
    ]),
}

results_spam = {}
print(f"\n{'Classifier':<28} {'CV Acc':>8} {'Test Acc':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 78)

for name, clf in classifiers.items():
    clf.fit(train_data.data, y_tr_bin)
    y_pred = clf.predict(test_data.data)
    cv_acc = cross_val_score(clf, train_data.data, y_tr_bin, cv=5).mean()

    from sklearn.metrics import precision_score, recall_score, f1_score
    test_acc = (y_pred == y_te_bin).mean()
    prec     = precision_score(y_te_bin, y_pred)
    rec      = recall_score(y_te_bin, y_pred)
    f1       = f1_score(y_te_bin, y_pred)
    results_spam[name] = {'test_acc': test_acc, 'f1': f1}
    print(f"{name:<28} {cv_acc:>8.4f} {test_acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")

# ── Most spam-associated words ────────────────────────────────────────────────
best_nb = classifiers['MultinomialNB (α=0.1)']
best_nb.fit(train_data.data, y_tr_bin)
vectorizer = best_nb.named_steps['tfidf']
nb_clf     = best_nb.named_steps['clf']
feat_names = np.array(vectorizer.get_feature_names_out())

# Log probability difference: high positive = more spam-like
log_prob_diff = nb_clf.feature_log_prob_[1] - nb_clf.feature_log_prob_[0]
top_spam      = feat_names[np.argsort(log_prob_diff)[-15:]][::-1]
top_ham       = feat_names[np.argsort(log_prob_diff)[:15]]

print(f"\nTop spam-associated words (MultinomialNB):")
print(f"  {', '.join(top_spam)}")
print(f"\nTop ham-associated words:")
print(f"  {', '.join(top_ham)}")

# ── Visualization ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Model comparison
model_names = list(results_spam.keys())
f1_scores   = [results_spam[n]['f1'] for n in model_names]
colors_m    = ['#A5D6A7' if 'NB' in n else '#90CAF9' for n in model_names]
bars_m      = axes[0].barh(model_names, f1_scores, color=colors_m, alpha=0.9)
axes[0].bar_label(bars_m, fmt='%.4f', padding=4)
axes[0].set_xlabel('F1 Score')
axes[0].set_title('Spam Detection — F1 Score\nGreen = Naive Bayes, Blue = Discriminative')
axes[0].set_xlim(0.8, 1.0)
axes[0].grid(axis='x', alpha=0.3)

# Most discriminative words
sort_idx = np.argsort(log_prob_diff)
top_k    = 10
words    = np.concatenate([feat_names[sort_idx[:top_k]],
                            feat_names[sort_idx[-top_k:]]])
diffs    = np.concatenate([log_prob_diff[sort_idx[:top_k]],
                            log_prob_diff[sort_idx[-top_k:]]])
colors_w = ['#A5D6A7' if d < 0 else '#EF9A9A' for d in diffs]

axes[1].barh(words, diffs, color=colors_w, alpha=0.8)
axes[1].axvline(0, color='black', lw=1)
axes[1].set_xlabel('log P(w|spam) - log P(w|ham)')
axes[1].set_title('Most Discriminative Words\nRed = spam-like, Green = ham-like')
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('Email Spam Detection with Naive Bayes',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_spam_detection.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 5. Summary

### ✅ Must-Remember Mental Models

**KNN:**

**1. KNN is a non-parametric, lazy learner.** No training — just memorize data. Prediction is $O(n \cdot d)$ per query (naive) or $O(\log n)$ with KD/Ball tree.

**2. KNN prediction rule:**
- Classification: majority class among $k$ nearest neighbors
- Regression: mean (or distance-weighted mean) of $k$ neighbors

**3. Feature scaling is mandatory for KNN.** Distance metrics are sensitive to scale. Always `StandardScaler` before KNN.

**4. $k$ controls the bias-variance tradeoff:**
- Small $k$ = low bias, high variance (complex, overfitting boundary)
- Large $k$ = high bias, low variance (smooth, underfitting boundary)
- Always tune $k$ via cross-validation.

**5. The curse of dimensionality kills KNN in high dimensions.** As $d$ grows, all points become equidistant. KNN degrades while linear models maintain performance.

**Naive Bayes:**

**6. The Naive Bayes formula:**
$$\hat{y} = \arg\max_k P(y=k) \prod_{j=1}^d P(x_j | y=k)$$
Work in log space to avoid underflow: sum log-probabilities, don't multiply raw probabilities.

**7. Which NB variant to use:**
- Continuous features: `GaussianNB`
- Word counts: `MultinomialNB`
- Binary features (word present/absent): `BernoulliNB`
- Imbalanced text: `ComplementNB`

**8. Always use Laplace smoothing.** `alpha=1.0` (default). Without it, any unseen feature causes the entire prediction to collapse to zero.

**9. NB vs Logistic Regression:**
- NB wins with very few samples (generative model, strong prior)
- LR wins with more data (learns the decision boundary directly)
- Crossover: typically $n \approx 30$–$100$

**10. NB is calibrated poorly but classifies well.** The independence assumption makes probabilities wrong, but the ranking of classes is usually correct. Use `CalibratedClassifierCV` if you need reliable probabilities.

---

## 6. Exercises

**KNN:**

1. **Implement KNN for regression with cross-validation**: on the California Housing dataset (using only 2 features for visualization), implement KNN regression from scratch. Plot the predicted surface as a heatmap for $k \in \{1, 5, 20\}$.

2. **Distance metric comparison**: on the breast cancer dataset, compare Euclidean, Manhattan, and Chebyshev distances for $k=5$. Which performs best? Use cross-validation. Does scaling affect the comparison?

3. **Dimensionality reduction before KNN**: apply PCA (retaining 95% variance) before KNN on the breast cancer dataset. Compare accuracy with and without PCA. How many components are retained?

4. **Challenge — Approximate KNN**: Implement Locality-Sensitive Hashing (LSH) for approximate nearest neighbor search. Show that it produces results similar to exact KNN but faster on a large dataset ($n=50,000$).

**Naive Bayes:**

5. **Naive Bayes from scratch**: Implement `GaussianNB` from scratch with full support for `predict_proba` and `partial_fit` (incremental learning). Verify against sklearn.

6. **Calibration analysis**: Train a MultinomialNB and a LogisticRegression on 20 Newsgroups. Plot calibration curves for both. Show that NB is poorly calibrated (overconfident) while LR is better calibrated.

7. **Feature log-probability ratio**: For a trained MultinomialNB spam classifier, plot the top-20 features for each class by $\log P(w|y=k) - \log P(w|y\neq k)$. Do the top words make intuitive sense?

8. **Challenge — Continuous and discrete features**: Real datasets often mix continuous and categorical features. Implement a "Mixed Naive Bayes" that uses GaussianNB for continuous features and MultinomialNB for discrete features, combining their log-likelihoods. Test on the adult income dataset.

---

## 7. References

- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Trans. Information Theory*, 13(1), 21–27. — Original KNN convergence proof.
- Fix, E., & Hodges, J. L. (1951). Discriminatory analysis, nonparametric discrimination: consistency properties. *USAF School of Aviation Medicine Technical Report*. — Original KNN paper.
- Naive Bayes: Good, I. J. (1965). *The Estimation of Probabilities: An Essay on Modern Bayesian Methods*. MIT Press.
- Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers. *NeurIPS*. — Generative vs discriminative comparison.
- Rennie, J. D. M., et al. (2003). Tackling the Poor Assumptions of Naive Bayes Text Classifiers. *ICML*. — ComplementNB.
- Zhang, H. (2004). The Optimality of Naive Bayes. *FLAIRS Conference*. — Why NB works despite wrong assumptions.

---

*Previous: [Chapter 9 — Support Vector Machines](09_svm.md)*  
*Next: [Chapter 11 — Unsupervised Learning: K-Means, DBSCAN & Hierarchical Clustering](11_unsupervised.md)*

*In Chapter 11, we move to unsupervised learning — finding structure in data without labels. We implement K-Means from scratch, derive the EM interpretation, introduce DBSCAN for density-based clustering, and show how to evaluate clustering quality.*

---

> **Chapter 10 complete.** All code is in `notebooks/10_knn_nb.ipynb`.
