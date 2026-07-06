# Chapter 12: Dimensionality Reduction — PCA, t-SNE & UMAP

> *"All dimensionality reduction methods are lies — they compress information. The question is: which lies are useful for your problem?"*

> *"PCA is the workhorse: interpretable, fast, mathematically principled. t-SNE is the storyteller: reveals cluster structure beautifully but distorts global geometry. UMAP is the modern standard: faster than t-SNE, preserves more structure, and scales."*

---

## Table of Contents

1. [Why Reduce Dimensions?](#1-why-reduce-dimensions)
2. [Principal Component Analysis (PCA)](#2-principal-component-analysis-pca)
   - 2.1 [The Geometric Intuition](#21-the-geometric-intuition)
   - 2.2 [PCA via Eigendecomposition](#22-pca-via-eigendecomposition)
   - 2.3 [PCA via SVD](#23-pca-via-svd)
   - 2.4 [PCA from Scratch](#24-pca-from-scratch)
   - 2.5 [Explained Variance and the Scree Plot](#25-explained-variance-and-the-scree-plot)
   - 2.6 [Choosing the Number of Components](#26-choosing-the-number-of-components)
   - 2.7 [PCA for Preprocessing](#27-pca-for-preprocessing)
   - 2.8 [Limitations of PCA](#28-limitations-of-pca)
3. [Kernel PCA](#3-kernel-pca)
4. [t-SNE — t-Distributed Stochastic Neighbor Embedding](#4-t-sne--t-distributed-stochastic-neighbor-embedding)
   - 4.1 [The Algorithm](#41-the-algorithm)
   - 4.2 [The Crowding Problem and the t-Distribution](#42-the-crowding-problem-and-the-t-distribution)
   - 4.3 [Using t-SNE Correctly](#43-using-t-sne-correctly)
5. [UMAP — Uniform Manifold Approximation and Projection](#5-umap--uniform-manifold-approximation-and-projection)
   - 5.1 [The Intuition](#51-the-intuition)
   - 5.2 [UMAP vs t-SNE](#52-umap-vs-t-sne)
6. [Linear Discriminant Analysis (LDA)](#6-linear-discriminant-analysis-lda)
7. [Comparing All Methods](#7-comparing-all-methods)
8. [Case Study: Visualizing the MNIST Digit Space](#8-case-study-visualizing-the-mnist-digit-space)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. Why Reduce Dimensions?

High-dimensional data causes several problems:

- **The Curse of Dimensionality**: distances become meaningless; all points appear equidistant
- **Visualization**: impossible beyond 3D
- **Computational cost**: many algorithms scale poorly with $d$
- **Overfitting**: too many features relative to samples
- **Correlated features**: many features carry redundant information

Dimensionality reduction finds a low-dimensional representation that preserves the most important structure of the data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_swiss_roll

rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# High-dimensional data is hard to visualize
digits = load_digits()
axes[0].imshow(digits.images[:16].reshape(4, 4, 8, 8)
               .transpose(0, 2, 1, 3).reshape(32, 32),
               cmap='gray_r', interpolation='nearest')
axes[0].set_title('MNIST digits: 64 dimensions\n'
                   '(8×8 pixel images — impossible to visualize directly)')
axes[0].axis('off')

# The manifold hypothesis: high-dim data lies on a low-dim manifold
X_swiss, t = make_swiss_roll(1000, noise=0.1, random_state=42)
ax3d = fig.add_subplot(1, 3, 2, projection='3d')
ax3d.scatter(X_swiss[:,0], X_swiss[:,1], X_swiss[:,2],
             c=t, cmap='viridis', s=15, alpha=0.7)
ax3d.set_title('Swiss Roll: 3D data\nlying on a 2D manifold')
ax3d.tick_params(labelsize=7)

# After dimensionality reduction
from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_swiss)
sc = axes[2].scatter(X_tsne[:,0], X_tsne[:,1], c=t, cmap='viridis', s=15, alpha=0.7)
axes[2].set_title('t-SNE unrolls the manifold\n(2D preserves structure)')
axes[2].grid(True, alpha=0.3)
plt.colorbar(sc, ax=axes[2], label='Position on roll')

plt.suptitle('Dimensionality Reduction: From High-D to Low-D',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_motivation.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
Three main goals of dimensionality reduction:
  1. VISUALIZATION: project to 2D/3D for human inspection
  2. PREPROCESSING: remove noise/redundancy before ML model
  3. COMPRESSION: store/transmit data more efficiently

Two families:
  LINEAR: PCA, LDA — find linear subspace
          Fast, interpretable, can project new data
  NON-LINEAR: t-SNE, UMAP, autoencoders — find curved manifold
              Better for complex structure, but usually for visualization only
""")
```

---

## 2. Principal Component Analysis (PCA)

### 2.1 The Geometric Intuition

PCA finds the directions (principal components) of **maximum variance** in the data. The first PC is the direction along which the data varies the most. The second PC is orthogonal to the first and captures the second-most variance, and so on.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Correlated 2D data
mean  = [0, 0]
cov   = [[3, 2.5], [2.5, 3]]
X     = rng.multivariate_normal(mean, cov, 200)

# Compute PCA manually
X_c   = X - X.mean(axis=0)
cov_m = X_c.T @ X_c / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov_m)
# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Original data with PCs
ax = axes[0]
ax.scatter(X[:,0], X[:,1], s=20, alpha=0.5, color='steelblue')
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
scale = 2.5
for i, (ev, color, label) in enumerate(zip(
    eigenvalues, ['tomato', 'green'],
    ['PC1 (max variance)', 'PC2 (min variance)']
)):
    v = eigenvectors[:, i] * np.sqrt(ev) * scale
    ax.annotate('', xy=v, xytext=-v,
                arrowprops=dict(arrowstyle='<->', color=color, lw=2.5))
    ax.text(*v*1.15, label, color=color, fontsize=8, fontweight='bold')
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.set_title('PCA: Principal Components\n'
             'Direction of maximum variance')

# Project to 1D (PC1)
ax2 = axes[1]
X_proj_1d = X_c @ eigenvectors[:, 0]   # Project onto PC1
X_recon   = np.outer(X_proj_1d, eigenvectors[:, 0]) + X.mean(axis=0)
ax2.scatter(X[:,0], X[:,1], s=20, alpha=0.4, color='steelblue', label='Original')
ax2.scatter(X_recon[:,0], X_recon[:,1], s=15, alpha=0.6, color='tomato',
            label='Projection onto PC1')
for orig, proj in zip(X[:20], X_recon[:20]):
    ax2.plot([orig[0], proj[0]], [orig[1], proj[1]], 'gray', lw=0.5, alpha=0.5)
v = eigenvectors[:, 0] * 5
ax2.plot([-v[0], v[0]], [-v[1], v[1]], 'tomato', lw=2.5, label='PC1 axis')
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
ax2.set_title('Projection onto PC1\nGray lines = reconstruction error')
ax2.legend(fontsize=7)

# 2D projection (whitened)
ax3 = axes[2]
X_pca = X_c @ eigenvectors
ax3.scatter(X_pca[:,0], X_pca[:,1], s=20, alpha=0.5, color='steelblue')
ax3.axhline(0, color='tomato', lw=1.5, linestyle='--', label='PC1')
ax3.axvline(0, color='green',  lw=1.5, linestyle='--', label='PC2')
ax3.set_xlabel(f'PC1 (var={eigenvalues[0]:.2f})')
ax3.set_ylabel(f'PC2 (var={eigenvalues[1]:.2f})')
ax3.set_title('Data in PCA space\nPCs are uncorrelated (orthogonal)')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

plt.suptitle('PCA: Finding Directions of Maximum Variance',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_pca_geometry.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 2.2 PCA via Eigendecomposition

**Derivation:**

We want the unit vector $\mathbf{v}_1$ that maximizes $\text{Var}(\mathbf{v}_1^T \mathbf{X}) = \mathbf{v}_1^T \boldsymbol{\Sigma} \mathbf{v}_1$ subject to $\|\mathbf{v}_1\| = 1$.

By Lagrange multipliers: $\boldsymbol{\Sigma} \mathbf{v}_1 = \lambda_1 \mathbf{v}_1$

So $\mathbf{v}_1$ is an **eigenvector** of the covariance matrix $\boldsymbol{\Sigma}$, and the variance explained equals the eigenvalue $\lambda_1$.

The first $k$ PCs are the eigenvectors corresponding to the $k$ largest eigenvalues.

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
PCA via Eigendecomposition — Step by Step:
═══════════════════════════════════════════════════════════════════

Given: Data matrix X ∈ R^{n×d}

Step 1: Center the data
  X_c = X - mean(X, axis=0)
  
Step 2: Compute covariance matrix
  Σ = (1/(n-1)) X_c^T X_c    ← shape (d, d)
  
Step 3: Eigendecompose Σ
  Σ = Q Λ Q^T
  Q: eigenvectors (columns) — the principal component directions
  Λ: diagonal matrix of eigenvalues — variance explained by each PC
  
Step 4: Sort by descending eigenvalue
  λ₁ ≥ λ₂ ≥ ... ≥ λd ≥ 0
  (All eigenvalues ≥ 0 because Σ is positive semi-definite)
  
Step 5: Project data
  X_pca = X_c @ Q[:, :k]     ← keep only top-k PCs
  Shape: (n, k)
  
Step 6: Reconstruct (optional)
  X_recon = X_pca @ Q[:, :k]^T + mean(X)
  
Key relationships:
  Variance of i-th principal component = λᵢ
  Fraction of variance explained by PC i = λᵢ / Σⱼ λⱼ
  PCs are orthonormal: Q^T Q = I
  PCA scores are uncorrelated: cov(X_pca) = Λ (diagonal)
═══════════════════════════════════════════════════════════════════
""")
```

---

### 2.3 PCA via SVD

In practice, PCA is computed via **Singular Value Decomposition (SVD)** of the centered data matrix, which is more numerically stable than eigendecomposing the covariance matrix.

$$\mathbf{X}_c = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

- $\mathbf{V}$: right singular vectors = principal component directions (same as eigenvectors of $\mathbf{X}_c^T \mathbf{X}_c$)
- $\boldsymbol{\Sigma}$: singular values; eigenvalues $\lambda_i = \sigma_i^2 / (n-1)$
- $\mathbf{U} \boldsymbol{\Sigma}$: PCA scores (projections)

```python
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X    = iris.data.astype(float)
X_c  = X - X.mean(axis=0)

# ── Method 1: Eigendecomposition of covariance matrix ─────────────────────────
Sigma = X_c.T @ X_c / (len(X) - 1)
eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(Sigma)
idx = np.argsort(eigenvalues_cov)[::-1]
eigenvalues_cov  = eigenvalues_cov[idx]
eigenvectors_cov = eigenvectors_cov[:, idx]

X_pca_eig = X_c @ eigenvectors_cov[:, :2]

# ── Method 2: SVD of centered data matrix ─────────────────────────────────────
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
eigenvalues_svd = s**2 / (len(X) - 1)
X_pca_svd       = X_c @ Vt[:2].T    # Equivalent to U[:, :2] * s[:2]

# ── Method 3: sklearn (uses SVD internally) ────────────────────────────────────
from sklearn.decomposition import PCA
pca_sk = PCA(n_components=2)
X_pca_sk = pca_sk.fit_transform(X)

print("PCA: Three Methods Comparison")
print(f"\nEigenvalues (Cov):  {eigenvalues_cov[:2].round(6)}")
print(f"Eigenvalues (SVD):  {eigenvalues_svd[:2].round(6)}")
print(f"Sklearn var:        {pca_sk.explained_variance_.round(6)}")

# Verify projections match (up to sign)
sign_flip_1 = np.sign(X_pca_eig[0, 0]) * np.sign(X_pca_svd[0, 0])
sign_flip_2 = np.sign(X_pca_eig[0, 0]) * np.sign(X_pca_sk[0, 0])
print(f"\nProjections match: {np.allclose(np.abs(X_pca_eig), np.abs(X_pca_svd), atol=1e-5)}")
print(f"Sklearn matches:   {np.allclose(np.abs(X_pca_eig), np.abs(X_pca_sk), atol=1e-5)}")
print("\nNote: PCA results are unique up to sign flips of each component.")
print("Sklearn uses SVD for numerical stability (avoids squaring the matrix).")
```

---

### 2.4 PCA from Scratch

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

class PCAScratch:
    """
    PCA from scratch using SVD.
    Follows sklearn's API: fit → transform → inverse_transform.
    """

    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten       = whiten

    def fit(self, X):
        n, d = X.shape
        k    = self.n_components or min(n, d)

        # Center
        self.mean_ = X.mean(axis=0)
        X_c        = X - self.mean_

        # SVD (economy/thin SVD)
        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

        # Store top-k components
        self.components_          = Vt[:k]              # (k, d)
        self.explained_variance_  = s[:k]**2 / (n - 1)
        total_var = (X_c**2).sum() / (n - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        self.singular_values_     = s[:k]
        self.n_components_        = k
        self.n_samples_           = n
        self.n_features_          = d
        return self

    def transform(self, X):
        X_c    = X - self.mean_
        X_proj = X_c @ self.components_.T    # (n, k)
        if self.whiten:
            X_proj /= (self.singular_values_ / np.sqrt(self.n_samples_ - 1))
        return X_proj

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_proj):
        """Reconstruct original-space data from PCA projections."""
        if self.whiten:
            X_proj = X_proj * (self.singular_values_ / np.sqrt(self.n_samples_ - 1))
        return X_proj @ self.components_ + self.mean_

    @property
    def noise_variance_(self):
        """Variance of discarded components (used in probabilistic PCA)."""
        total_var = self.explained_variance_.sum() / self.explained_variance_ratio_.sum()
        return (total_var - self.explained_variance_.sum()) / max(
            self.n_features_ - self.n_components_, 1
        )


# ── Test ───────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA

cancer  = load_breast_cancer()
X_raw   = cancer.data
scaler  = StandardScaler()
X       = scaler.fit_transform(X_raw)

pca_mine = PCAScratch(n_components=10)
X_mine   = pca_mine.fit_transform(X)

pca_sk = PCA(n_components=10)
X_sk   = pca_sk.fit_transform(X)

print("PCA Scratch vs Sklearn (Breast Cancer, 30→10 dims):")
print(f"\nExplained variance ratio:")
for i, (m, s) in enumerate(zip(pca_mine.explained_variance_ratio_,
                                 pca_sk.explained_variance_ratio_)):
    print(f"  PC{i+1:2d}: scratch={m:.6f}, sklearn={s:.6f}, match={np.isclose(m,s)}")

print(f"\nTotal variance explained: "
      f"{pca_mine.explained_variance_ratio_.sum()*100:.2f}% (scratch), "
      f"{pca_sk.explained_variance_ratio_.sum()*100:.2f}% (sklearn)")

# Reconstruction error
X_recon_mine = pca_mine.inverse_transform(X_mine)
X_recon_sk   = pca_sk.inverse_transform(X_sk)
recon_err = np.mean((X - X_recon_mine)**2)
print(f"\nReconstruction MSE (10 components): {recon_err:.6f}")
print(f"(Lower = more information preserved)")
```

---

### 2.5 Explained Variance and the Scree Plot

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = StandardScaler().fit_transform(cancer.data)

pca = PCA().fit(X)
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scree plot
ax = axes[0]
ax.bar(range(1, len(evr)+1), evr*100, color='steelblue', alpha=0.8)
ax.plot(range(1, len(evr)+1), evr*100, 'o-', color='tomato', lw=2, markersize=5)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('Scree Plot\n(Bar heights show individual PC variance)')
ax.grid(axis='y', alpha=0.3)

# Cumulative variance
ax2 = axes[1]
ax2.plot(range(1, len(cum_evr)+1), cum_evr*100, 'o-', color='steelblue',
         lw=2.5, markersize=5)
for threshold, color in [(0.80, 'green'), (0.90, 'orange'), (0.95, 'tomato')]:
    n_comps = np.argmax(cum_evr >= threshold) + 1
    ax2.axhline(threshold*100, color=color, lw=1.5, linestyle='--',
                label=f'{threshold*100:.0f}% → {n_comps} PCs')
    ax2.axvline(n_comps, color=color, lw=1, linestyle=':')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained (%)')
ax2.set_title('Cumulative Variance\n(Choose K where curve flattens)')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# PCA loadings heatmap (first 5 components)
ax3 = axes[2]
n_show = 5
loadings = pca.components_[:n_show]
im = ax3.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
ax3.set_xticks(range(len(cancer.feature_names)))
ax3.set_xticklabels([f.replace(' (mean)', '').replace(' (worst)', 'W')
                      for f in cancer.feature_names], rotation=90, fontsize=6)
ax3.set_yticks(range(n_show))
ax3.set_yticklabels([f'PC{i+1}\n({evr[i]*100:.1f}%)' for i in range(n_show)],
                    fontsize=8)
plt.colorbar(im, ax=ax3, label='Loading')
ax3.set_title('PCA Loadings (first 5 PCs)\n'
              'Red=positive, Blue=negative contribution')

plt.suptitle('PCA: Explained Variance Analysis — Breast Cancer Dataset',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_pca_variance.png', dpi=140, bbox_inches='tight')
plt.show()

for thr in [0.80, 0.90, 0.95, 0.99]:
    n = np.argmax(cum_evr >= thr) + 1
    print(f"  {thr*100:.0f}% variance explained by {n:2d} components "
          f"(of {len(evr)} total)")
```

---

### 2.6 Choosing the Number of Components

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_raw, y = cancer.data, cancer.target
X = StandardScaler().fit_transform(X_raw)

# Strategy 1: Fixed threshold (e.g., 95% variance)
pca_95 = PCA(n_components=0.95)   # sklearn accepts float = variance threshold
X_95   = pca_95.fit_transform(X)
print(f"95% variance threshold: {pca_95.n_components_} components "
      f"(from {X.shape[1]})")

# Strategy 2: Cross-validate classification accuracy vs n_components
n_range = list(range(1, 31))
cv_accs = []
for n in n_range:
    pipe = Pipeline([
        ('pca', PCA(n_components=n)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    cv = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
    cv_accs.append(cv.mean())

best_n = n_range[np.argmax(cv_accs)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(n_range, cv_accs, 'o-', color='steelblue', lw=2.5, markersize=6)
ax.axvline(best_n, color='tomato', lw=2, linestyle='--',
           label=f'Best n={best_n} (AUC={max(cv_accs):.4f})')
ax.axhline(cross_val_score(
    Pipeline([('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    X, y, cv=5, scoring='roc_auc').mean(),
    color='green', lw=1.5, linestyle=':', label='No PCA baseline')
ax.set_xlabel('Number of PCA components')
ax.set_ylabel('CV AUC')
ax.set_title('PCA Components vs Classification Performance\n'
             'Downstream task tells you the right number')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Strategy 3: MLE for optimal n (Minka 2000)
pca_mle = PCA(n_components='mle')
pca_mle.fit(X)
print(f"MLE optimal components: {pca_mle.n_components_}")

axes[1].axis('off')
table_data = [
    ['Method', 'How to use', 'Best for'],
    ['Fixed %\n(e.g., 95%)', 'PCA(n_components=0.95)', 'Preprocessing\nbefore ML'],
    ['Scree plot\nelbow', 'Visual inspection', 'Exploratory\nanalysis'],
    ['CV accuracy', 'Loop + cross_val_score', 'When PCA →\nclassifier'],
    ['MLE', 'PCA(n_components="mle")', 'Probabilistic\nmodeling'],
]
table = axes[1].table(cellText=table_data[1:], colLabels=table_data[0],
                       cellLoc='left', loc='center', bbox=[0, 0.1, 1, 0.85])
table.auto_set_font_size(False); table.set_fontsize(9)
axes[1].set_title('Strategies for Choosing n_components', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/12_pca_n_components.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 2.7 PCA for Preprocessing

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

print("PCA as Preprocessing — Speed and Accuracy Comparison:")
print(f"\n{'Configuration':<35} {'CV AUC':>10} {'Test AUC':>10}")
print("-" * 60)

import time
from sklearn.metrics import roc_auc_score

configs = [
    ('SVM RBF (no PCA)',
     Pipeline([('sc', StandardScaler()),
               ('m',  SVC(kernel='rbf', C=10, gamma='scale',
                          probability=True, random_state=42))])),
    ('SVM RBF + PCA(95%)',
     Pipeline([('sc',  StandardScaler()),
               ('pca', PCA(n_components=0.95)),
               ('m',   SVC(kernel='rbf', C=10, gamma='scale',
                           probability=True, random_state=42))])),
    ('SVM RBF + PCA(10)',
     Pipeline([('sc',  StandardScaler()),
               ('pca', PCA(n_components=10)),
               ('m',   SVC(kernel='rbf', C=10, gamma='scale',
                           probability=True, random_state=42))])),
    ('LR (no PCA)',
     Pipeline([('sc', StandardScaler()),
               ('m',  LogisticRegression(max_iter=2000, C=1.0))])),
    ('LR + PCA(95%)',
     Pipeline([('sc',  StandardScaler()),
               ('pca', PCA(n_components=0.95)),
               ('m',   LogisticRegression(max_iter=2000, C=1.0))])),
]

for name, pipe in configs:
    t0 = time.perf_counter()
    cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring='roc_auc')
    pipe.fit(X_tr, y_tr)
    test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:,1])
    elapsed  = time.perf_counter() - t0
    print(f"{name:<35} {cv.mean():>10.4f} {test_auc:>10.4f}  ({elapsed:.2f}s)")

print("""
\nKey PCA preprocessing insights:
  • PCA before SVM: reduces d → fewer distance computations in kernel
  • PCA acts as regularization: removes noisy dimensions
  • For linear models: PCA rarely helps unless d >> n
  • Always fit PCA on training set only → apply to test set (Pipeline handles this)
  • PCA whitening (whiten=True): scales each PC to unit variance → 
    can improve convergence of gradient-based methods
""")
```

---

### 2.8 Limitations of PCA

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_s_curve

rng = np.random.default_rng(42)

# PCA fails on non-linear manifolds
X_swiss, color = make_swiss_roll(1000, noise=0.05, random_state=42)
X_s, color_s   = make_s_curve(1000, noise=0.05, random_state=42)

from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Swiss roll original (3D → use 2 views)
ax = axes[0]
ax.scatter(X_swiss[:,0], X_swiss[:,2], c=color, cmap='viridis',
           s=12, alpha=0.7)
ax.set_title('Swiss Roll\n(3D, view from above)')
ax.set_xlabel('x'); ax.set_ylabel('z'); ax.grid(True, alpha=0.3)

# PCA of swiss roll
X_pca_sw = pca2.fit_transform(X_swiss)
ax2 = axes[1]
ax2.scatter(X_pca_sw[:,0], X_pca_sw[:,1], c=color, cmap='viridis',
            s=12, alpha=0.7)
evr = pca2.explained_variance_ratio_
ax2.set_title(f'PCA (2D) — Swiss Roll FAILS\n'
              f'Colors mixed (PC1={evr[0]:.1%}, PC2={evr[1]:.1%})')
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2'); ax2.grid(True, alpha=0.3)

# S-curve
ax3 = axes[2]
ax3.scatter(X_s[:,0], X_s[:,2], c=color_s, cmap='viridis', s=12, alpha=0.7)
ax3.set_title('S-Curve\n(3D, view from above)')
ax3.set_xlabel('x'); ax3.set_ylabel('z'); ax3.grid(True, alpha=0.3)

# PCA of S-curve
X_pca_s = pca2.fit_transform(X_s)
ax4 = axes[3]
ax4.scatter(X_pca_s[:,0], X_pca_s[:,1], c=color_s, cmap='viridis',
            s=12, alpha=0.7)
ax4.set_title('PCA (2D) — S-Curve FAILS\n'
              'Cannot unroll curved manifolds')
ax4.set_xlabel('PC1'); ax4.set_ylabel('PC2'); ax4.grid(True, alpha=0.3)

plt.suptitle('PCA Limitation: Cannot Handle Non-Linear Manifolds\n'
             '→ Need Kernel PCA, t-SNE, or UMAP',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_pca_limits.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

## 3. Kernel PCA

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

datasets = [
    ('Circles', *make_circles(300, noise=0.05, factor=0.4, random_state=42)),
    ('Moons',   *make_moons(300, noise=0.07, random_state=42)),
]

kernels = [
    ('Linear PCA',   PCA(n_components=2)),
    ('RBF kPCA',     KernelPCA(n_components=2, kernel='rbf', gamma=10)),
    ('Poly kPCA',    KernelPCA(n_components=2, kernel='poly', degree=3)),
]

cmap = plt.cm.RdBu_r

for row, (ds_name, X_d, y_d) in enumerate(datasets):
    sc    = StandardScaler()
    X_dsc = sc.fit_transform(X_d)

    # Original data
    axes[row, 0].scatter(X_dsc[:,0], X_dsc[:,1], c=y_d, cmap=cmap,
                         s=20, alpha=0.8, edgecolors='k', linewidths=0.2)
    axes[row, 0].set_title(f'{ds_name} Original')
    axes[row, 0].grid(True, alpha=0.3)

    for col, (kern_name, kpca) in enumerate(kernels, start=1):
        X_k = kpca.fit_transform(X_dsc)
        axes[row, col].scatter(X_k[:,0], X_k[:,1], c=y_d, cmap=cmap,
                               s=20, alpha=0.8, edgecolors='k', linewidths=0.2)
        axes[row, col].set_title(f'{kern_name}')
        axes[row, col].grid(True, alpha=0.3)

plt.suptitle('Kernel PCA: Non-Linear Dimensionality Reduction\n'
             'Same kernel trick as SVM, applied to PCA',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_kernel_pca.png', dpi=130, bbox_inches='tight')
plt.show()

print("""
Kernel PCA:
  Applies the kernel trick to PCA.
  K(x, z) = φ(x)·φ(z)  →  PCA in the (infinite-dimensional) feature space.
  
  Same kernels as SVM: RBF, polynomial, sigmoid.
  Limitation: Cannot project new test points analytically (needs fit_transform).
  sklearn solution: use fit_inverse_transform=True for approximate inversion.
""")
```

---

## 4. t-SNE — t-Distributed Stochastic Neighbor Embedding

### 4.1 The Algorithm

t-SNE is a **non-linear** dimensionality reduction technique designed primarily for visualization. It preserves **local structure** (nearby points in high-D stay nearby in low-D) while allowing global structure to be distorted.

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
t-SNE ALGORITHM
════════════════════════════════════════════════════════════════════

High-dimensional space:
  For each pair (i,j): compute pᵢⱼ = similarity as conditional probability
  
  p(j|i) = exp(-||xᵢ-xⱼ||² / (2σᵢ²)) / Σ_{k≠i} exp(-||xᵢ-xₖ||² / (2σᵢ²))
  
  pᵢⱼ = (p(j|i) + p(i|j)) / (2n)  [symmetrized]
  
  σᵢ is chosen so that the perplexity of pᵢ equals the target perplexity.
  Perplexity ≈ effective number of neighbors ≈ 5-50 (hyperparameter).

Low-dimensional space:
  Initialize Y randomly in 2D.
  For each pair (i,j): compute qᵢⱼ using t-distribution (Student with 1 df):
  
  qᵢⱼ = (1 + ||yᵢ-yⱼ||²)⁻¹ / Σ_{k≠l} (1 + ||yₖ-yₗ||²)⁻¹

Objective:
  Minimize KL(P || Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ / qᵢⱼ)
  Gradient descent on Y (the 2D positions).

Why t-distribution in low-D?
  Gaussian would cause the "crowding problem" — no room for medium-distance
  points. t-distribution has heavier tails, giving more space.
  
Key properties:
  ✓ Reveals local cluster structure beautifully
  ✗ Global distances not preserved (clusters far apart ≠ far in high-D)
  ✗ Non-deterministic (random init, multiple runs may differ)
  ✗ Slow: O(n²) naive; O(n log n) with Barnes-Hut approximation
  ✗ Cannot project new points (must rerun on full dataset + new points)
════════════════════════════════════════════════════════════════════
""")
```

---

### 4.2 The Crowding Problem and the t-Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, norm

z = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Gaussian vs t-distribution tails
ax = axes[0]
ax.plot(z, norm.pdf(z), color='steelblue', lw=2.5, label='Gaussian (high-D)')
ax.plot(z, t_dist.pdf(z, df=1), color='tomato', lw=2.5,
        label='t-distribution (low-D, df=1)')
ax.fill_between(z, t_dist.pdf(z, df=1), norm.pdf(z),
                where=np.abs(z)>1.5, alpha=0.3, color='tomato',
                label='Extra "room" from heavier tails')
ax.set_xlabel('Distance'); ax.set_ylabel('Density')
ax.set_title('The t-Distribution Solution to Crowding\n'
             'Heavy tails allow medium-distance points to spread out')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Crowding problem illustration
ax2 = axes[1]
# In 2D: many points can be at equal distance from center
# In 10D: MANY more points can be at equal distance → all compress into same region
dims   = np.arange(1, 21)
# Surface area of d-sphere grows, but most volume is near the surface
# Volume of d-ball at radius r vs r-delta
shell_volume_ratio = [(d/2 * np.log(1/0.9)) for d in dims]
ax2.plot(dims, shell_volume_ratio, 'o-', color='steelblue', lw=2.5)
ax2.set_xlabel('Dimensions')
ax2.set_ylabel('Surface fraction of sphere (proxy)')
ax2.set_title('The Crowding Problem\n'
              'In high-D, all neighbors are at ~equal distance\n'
              '→ cannot fit them all in 2D without distortion')
ax2.grid(True, alpha=0.3)
ax2.text(12, 3, 'In 10D+: all points\nappear equidistant',
         fontsize=9, color='tomato',
         bbox=dict(boxstyle='round', fc='lightyellow'))

plt.tight_layout()
plt.savefig('figures/12_tsne_crowding.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 4.3 Using t-SNE Correctly

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y   = digits.data, digits.target

# BEST PRACTICE: PCA first to reduce to ~50 dims, then t-SNE
pca50 = PCA(n_components=50, random_state=42)
X_50  = pca50.fit_transform(X)
print(f"Variance retained by PCA(50): {pca50.explained_variance_ratio_.sum()*100:.1f}%")

# Run t-SNE with different perplexities
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
perplexities = [5, 30, 50]
cmap_digits  = plt.cm.tab10

# Row 1: Raw data → t-SNE
for col, perp in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    ax = axes[0, col]
    scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y,
                         cmap=cmap_digits, s=15, alpha=0.7)
    ax.set_title(f't-SNE (perplexity={perp})\nRaw data (64D)')
    ax.grid(True, alpha=0.2)

# Row 2: PCA(50) → t-SNE
for col, perp in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_50)

    ax = axes[1, col]
    scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y,
                         cmap=cmap_digits, s=15, alpha=0.7)
    ax.set_title(f't-SNE (perplexity={perp})\nPCA(50) → t-SNE [recommended]')
    ax.grid(True, alpha=0.2)

# Add colorbar
cbar = plt.colorbar(scatter, ax=axes, orientation='vertical', fraction=0.02,
                     pad=0.02)
cbar.set_ticks(range(10))
cbar.set_label('Digit class')

plt.suptitle('t-SNE: Effect of Perplexity and Preprocessing\n'
             'PCA → t-SNE is the recommended workflow',
             fontsize=12, fontweight='bold')
plt.savefig('figures/12_tsne_digits.png', dpi=120, bbox_inches='tight')
plt.show()

print("""
t-SNE Best Practices:
  ✓ Always run PCA first (reduce to 20-100 dims) — speeds up t-SNE 10-100×
  ✓ Use init='pca' (PCA initialization, more stable than random)
  ✓ Perplexity ∈ [5, 50]; try 30 as default
  ✓ n_iter ≥ 1000 for convergence
  ✓ Run multiple times — results vary (non-deterministic with random seeds)
  ✗ Don't interpret cluster sizes or inter-cluster distances
  ✗ Don't use for downstream ML (information is distorted)
  ✗ Don't use for datasets n > 10,000 (slow, use UMAP instead)
""")
```

---

## 5. UMAP — Uniform Manifold Approximation and Projection

### 5.1 The Intuition

UMAP (McInnes et al., 2018) is grounded in Riemannian geometry and algebraic topology. Like t-SNE, it preserves local structure — but with key improvements:

- **Faster**: $O(n \log n)$ vs $O(n^2)$ for t-SNE
- **Preserves more global structure**: relative positions of clusters are more meaningful
- **Scalable**: works on millions of points
- **Supports projection of new data**: `transform()` method works

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X, y   = digits.data, digits.target
X_50   = PCA(n_components=50, random_state=42).fit_transform(X)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not installed. Install with: pip install umap-learn")

if UMAP_AVAILABLE:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap_d = plt.cm.tab10

    # UMAP with different n_neighbors
    for ax, n_nbrs, title in zip(
        axes,
        [5, 15, 50],
        ['n_neighbors=5\n(local structure)', 'n_neighbors=15\n(balanced)',
         'n_neighbors=50\n(global structure)']
    ):
        reducer = umap.UMAP(n_neighbors=n_nbrs, min_dist=0.1,
                            n_components=2, random_state=42)
        X_umap  = reducer.fit_transform(X_50)
        scatter = ax.scatter(X_umap[:,0], X_umap[:,1], c=y,
                             cmap=cmap_d, s=12, alpha=0.7)
        ax.set_title(f'UMAP — {title}')
        ax.grid(True, alpha=0.2)

    plt.colorbar(scatter, ax=axes[-1], label='Digit')
    plt.suptitle('UMAP: Effect of n_neighbors Parameter\n'
                 'Smaller = more local focus; Larger = more global',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/12_umap_digits.png', dpi=120, bbox_inches='tight')
    plt.show()

print("""
UMAP Key Parameters:
  n_neighbors  : Controls local/global balance (default=15)
                 Small → local structure, fragmented clusters
                 Large → global structure, more connected
  min_dist     : Minimum distance in low-D space (default=0.1)
                 Small → tightly packed clusters
                 Large → uniform spread (good for visualization)
  n_components : Output dimensions (default=2 for viz, higher for preprocessing)
  metric       : Distance metric (default='euclidean')
  random_state : For reproducibility
""")
```

---

### 5.2 UMAP vs t-SNE

```python
print("""
t-SNE vs UMAP Comparison:
════════════════════════════════════════════════════════════════════
                    t-SNE               UMAP
────────────────────────────────────────────────────────────────────
Speed               O(n log n)          O(n log n) faster constant
Speed (n=10k)       ~minutes            ~seconds
Scalability         n < 10k             n > 1M possible
Global structure    Poor                Better
Local structure     Excellent           Excellent
New data            No (refit needed)   Yes (transform())
Reproducibility     Variable            Better (random_state)
Theoretical basis   Probability         Riemannian geometry +
                    matching            topological analysis
Cluster separation  Over-emphasizes     More faithful
Preprocessing       PCA first           PCA first (recommended)
Hyperparameters     perplexity          n_neighbors, min_dist
Default use         Visualization       Visualization + preprocessing
────────────────────────────────────────────────────────────────────
Rule of thumb:
  ≤ 10,000 points: either t-SNE or UMAP; t-SNE often looks "nicer"
  > 10,000 points: UMAP is the clear choice
  Downstream ML:   neither (use PCA for linear preprocessing)
════════════════════════════════════════════════════════════════════
""")
```

---

## 6. Linear Discriminant Analysis (LDA)

LDA is a **supervised** dimensionality reduction method. Unlike PCA, it uses class labels to find directions that **maximize class separability**.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine   = load_wine()
X, y   = wine.data, wine.target
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

pca = PCA(n_components=2)
lda = LinearDiscriminantAnalysis(n_components=2)

X_pca = pca.fit_transform(X_sc)
X_lda = lda.fit_transform(X_sc, y)   # LDA uses labels!

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.cm.Set1

for ax, X_proj, method, subtitle in zip(
    axes,
    [X_pca, X_lda],
    ['PCA (unsupervised)', 'LDA (supervised)'],
    ['Maximizes variance', 'Maximizes class separability']
):
    for cls, color, name in zip([0,1,2], ['tomato','steelblue','green'],
                                  wine.target_names):
        mask = y == cls
        ax.scatter(X_proj[mask,0], X_proj[mask,1], s=40, color=color,
                   edgecolors='k', linewidths=0.4, alpha=0.8, label=name)

    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    knn   = KNeighborsClassifier(5)
    knn.fit(X_proj, y)
    acc   = knn.score(X_proj, y)
    ax.set_title(f'{method}\n({subtitle})\nKNN-5 train acc: {acc:.4f}')
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('PCA vs LDA: Unsupervised vs Supervised Reduction\n'
             'LDA explicitly separates classes using labels',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_pca_vs_lda.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
LDA Properties:
  • Finds at most K-1 discriminant directions (K = number of classes)
  • Maximizes: between-class scatter / within-class scatter
  • Assumes: each class has Gaussian distribution, equal covariance matrices
  • LDA is also a linear classifier (like logistic regression)
  • When assumptions hold: LDA is optimal (Bayes-optimal)
  
LDA vs PCA:
  PCA: unsupervised, maximizes total variance
  LDA: supervised, maximizes class separability
  LDA is better for classification preprocessing when classes are well-separated
""")
```

---

## 7. Comparing All Methods

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

digits = load_digits()
X, y   = digits.data, digits.target

# Subsample for speed
idx = np.random.default_rng(42).choice(len(X), 500, replace=False)
X_s, y_s = X[idx], y[idx]
X_sc = StandardScaler().fit_transform(X_s)

methods = {
    'PCA':         PCA(n_components=2),
    'Kernel PCA\n(RBF)': KernelPCA(n_components=2, kernel='rbf', gamma=0.01),
    'LDA':         LinearDiscriminantAnalysis(n_components=2),
    't-SNE':       TSNE(n_components=2, perplexity=30, random_state=42,
                        init='pca', learning_rate='auto'),
    'Isomap':      Isomap(n_components=2, n_neighbors=10),
}

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
cmap = plt.cm.tab10

for ax, (name, method) in zip(axes, methods.items()):
    if hasattr(method, 'fit_transform'):
        if 'LDA' in name:
            X_2d = method.fit_transform(X_sc, y_s)
        else:
            X_2d = method.fit_transform(X_sc)
    else:
        method.fit(X_sc)
        X_2d = method.transform(X_sc)

    scatter = ax.scatter(X_2d[:,0], X_2d[:,1], c=y_s, cmap=cmap,
                         s=15, alpha=0.7)
    ax.set_title(name, fontsize=9); ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)

plt.colorbar(scatter, ax=axes, label='Digit', fraction=0.01)
plt.suptitle('Dimensionality Reduction Method Comparison (MNIST subset, n=500)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_method_comparison.png', dpi=120, bbox_inches='tight')
plt.show()

print("""
Method Selection Guide:
════════════════════════════════════════════════════════════════════
Goal                  Best method          Notes
────────────────────────────────────────────────────────────────────
Preprocessing (ML)    PCA                  Fast, interpretable, scalable
Visualization         UMAP > t-SNE         UMAP faster; t-SNE nicer
Non-linear preprocessing  Kernel PCA       Use rbf kernel, tune gamma
Supervised reduction  LDA                  Only K-1 components max
Large dataset viz     UMAP                 t-SNE too slow for n>10k
Interpretability      PCA                  Loadings tell you "what is PC1"
Anomaly detection     PCA reconstruction   High reconstruction error = anomaly
Compression           PCA                  Optimal for linear compression
════════════════════════════════════════════════════════════════════
""")
```

---

## 8. Case Study: Visualizing the MNIST Digit Space

```python
# =============================================================================
# Full pipeline: MNIST digits → PCA preprocessing → UMAP visualization
# Goal: understand the structure of the digit space
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load MNIST (scikit-learn's 8×8 version) ───────────────────────────────
digits = load_digits()
X, y   = digits.data, digits.target
print(f"Dataset: {X.shape} (n_samples, n_features={digits.data.shape[1]})")
print(f"Classes: {digits.target_names}")

# ── 2. PCA preprocessing ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
pca50  = PCA(n_components=50, random_state=42)
X_50   = pca50.fit_transform(X_sc)
print(f"\nPCA(50) retains {pca50.explained_variance_ratio_.sum()*100:.1f}% variance")

# ── 3. t-SNE visualization ────────────────────────────────────────────────────
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
            random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_50)

# ── 4. K-Means on PCA space ───────────────────────────────────────────────────
km = KMeans(n_clusters=10, n_init=20, random_state=42)
km_labels = km.fit_predict(X_50)
ari_km = adjusted_rand_score(y, km_labels)

# ── 5. Comprehensive visualization ────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))

# Main t-SNE scatter
ax1 = fig.add_subplot(2, 3, (1, 2))
cmap_d = plt.cm.tab10
scatter = ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap=cmap_d,
                      s=12, alpha=0.7)
plt.colorbar(scatter, ax=ax1, label='Digit Class')
ax1.set_title(f't-SNE Visualization of MNIST Digits\n'
              f'(n={len(X)}, 64→50→2 dimensions via PCA+t-SNE)',
              fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.2)

# Annotate cluster centers
for cls in range(10):
    mask   = y == cls
    cx, cy = X_tsne[mask, 0].mean(), X_tsne[mask, 1].mean()
    ax1.annotate(str(cls), (cx, cy), fontsize=14, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

# PCA scree plot
ax2 = fig.add_subplot(2, 3, 3)
pca_full = PCA().fit(X_sc)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_) * 100
ax2.plot(range(1, len(cum_var)+1), cum_var, 'o-', color='steelblue',
         lw=2, markersize=3)
for thr, color in [(80,'green'), (95,'tomato')]:
    n_c = np.argmax(cum_var >= thr) + 1
    ax2.axhline(thr, color=color, lw=1.5, linestyle='--',
                label=f'{thr}% → {n_c} PCs')
    ax2.axvline(n_c, color=color, lw=1, linestyle=':')
ax2.set_xlabel('# Components'); ax2.set_ylabel('Cumulative Variance %')
ax2.set_title('PCA Scree: MNIST digits (64D)')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# Sample digits per class
ax3 = fig.add_subplot(2, 3, 4)
n_per_class = 5
sample_imgs = []
for cls in range(10):
    cls_idx = np.where(y == cls)[0][:n_per_class]
    sample_imgs.append(digits.images[cls_idx])
grid = np.concatenate([np.concatenate(imgs, axis=1)
                        for imgs in sample_imgs], axis=0)
ax3.imshow(grid, cmap='gray_r', interpolation='nearest', aspect='auto')
ax3.set_yticks(np.arange(10)*8 + 4)
ax3.set_yticklabels(range(10), fontsize=8)
ax3.set_xticks([])
ax3.set_title('Sample Digits (5 per class)')

# K-Means vs True labels in t-SNE space
ax4 = fig.add_subplot(2, 3, 5)
ax4.scatter(X_tsne[:,0], X_tsne[:,1], c=km_labels, cmap=plt.cm.Set1,
            s=12, alpha=0.6)
ax4.set_title(f'K-Means Clustering (K=10) in t-SNE Space\n'
              f'ARI={ari_km:.4f} (vs true labels)')
ax4.grid(True, alpha=0.2)

# PCA top-2 visualization
ax5 = fig.add_subplot(2, 3, 6)
pca2  = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_sc)
ax5.scatter(X_pca2[:,0], X_pca2[:,1], c=y, cmap=cmap_d, s=8, alpha=0.5)
ax5.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
ax5.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
ax5.set_title('PCA (2D) for comparison\n'
              '(Much less separation than t-SNE)')
ax5.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('figures/12_mnist_case_study.png', dpi=110, bbox_inches='tight')
plt.show()

print(f"\nKey findings:")
print(f"  PCA(2) captures {pca2.explained_variance_ratio_.sum()*100:.1f}% of variance")
print(f"  PCA(50) captures {pca50.explained_variance_ratio_.sum()*100:.1f}%")
print(f"  K-Means(10) on PCA(50) space: ARI={ari_km:.4f}")
print(f"  (ARI=1.0 would mean perfect match to true labels)")
```

---

## 9. Summary

### ✅ Must-Remember Mental Models

**1. PCA objective:** Find orthogonal directions of maximum variance. PC1 = direction of maximum variance; PC2 = direction of maximum variance orthogonal to PC1; etc.

**2. PCA via SVD (always use this in practice):**
$$\mathbf{X}_c = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T \Rightarrow \text{PCA scores} = \mathbf{U}\boldsymbol{\Sigma}, \quad \text{directions} = \mathbf{V}^T$$

**3. Explained variance ratio:** $\lambda_i / \sum_j \lambda_j$. Use the scree plot to find the "elbow." For 95% variance: `PCA(n_components=0.95)`.

**4. Always center (and usually scale) before PCA.** Without centering, PC1 captures the mean offset, not variance. Without scaling, high-variance features dominate.

**5. PCA is a linear method.** It cannot unroll curved manifolds (swiss roll, s-curve). For non-linear structure → Kernel PCA, t-SNE, or UMAP.

**6. t-SNE is for visualization only:**
- Preserves local structure (nearby points)
- Distorts global distances (inter-cluster distances are meaningless)
- Cannot project new data points
- Run `PCA(50) → t-SNE` for speed and stability

**7. UMAP is the modern standard:**
- Faster than t-SNE
- Better global structure preservation
- Supports `transform()` for new data
- Default parameters: `n_neighbors=15, min_dist=0.1`

**8. LDA is supervised dimensionality reduction:**
- Uses class labels to maximize separability
- Finds at most $K-1$ directions (for $K$ classes)
- Better than PCA for classification preprocessing when classes are separated

**9. The reconstruction error** = variance of discarded components. For anomaly detection: high reconstruction error → unusual point.

**10. Practical workflow:**
- For ML preprocessing: `StandardScaler → PCA(n_components=0.95)`
- For visualization: `StandardScaler → PCA(50) → t-SNE or UMAP`
- For classification: compare PCA vs LDA on validation set

---

## 10. Exercises

1. **Prove optimality of PCA**: Show using Lagrange multipliers that the first principal component is the eigenvector of the covariance matrix corresponding to the largest eigenvalue.

2. **Reconstruction error**: For the digits dataset, plot reconstruction error (MSE) as a function of number of PCA components. At what k does the "elbow" occur? Compare to the scree plot.

3. **PCA for noise reduction**: Add Gaussian noise (σ=0.5) to the iris dataset features. Apply PCA with k=2, then reconstruct. Compare the SNR before and after PCA denoising.

4. **t-SNE pitfalls**: Generate three nested Gaussian clusters (sizes 50, 50, 50). Run t-SNE 5 times with different random seeds. Show that (a) cluster separation varies across runs, (b) changing perplexity from 5 to 100 gives dramatically different layouts. Write a comment on what conclusions you can/cannot draw from t-SNE.

5. **UMAP for preprocessing**: On the digits dataset, use UMAP with `n_components=10` as a preprocessing step before KNN classification. Compare to PCA(10) preprocessing. Does non-linear reduction help?

6. **Challenge — Incremental PCA**: sklearn's `IncrementalPCA` allows fitting PCA on large datasets that don't fit in memory. Implement a simplified version that processes data in batches using the online update rule for the mean and covariance matrix.

---

## 11. References

- Pearson, K. (1901). On Lines and Planes of Closest Fit. *Philosophical Magazine*, 2(11), 559–572. — Original PCA paper.
- Van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579–2605. — t-SNE.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*. — UMAP.
- Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic Problems. *Annals of Eugenics*, 7(2), 179–188. — LDA.
- Minka, T. P. (2000). Automatic Choice of Dimensionality for PCA. *NeurIPS*. — MLE for PCA dimensionality.

---
*Next: [Chapter 13 — Feature Engineering & Selection](13_feature_engineering.md)*
