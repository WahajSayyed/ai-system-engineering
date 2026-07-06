# Chapter 11: Unsupervised Learning — K-Means, DBSCAN & Hierarchical Clustering

> *"Clustering is the art of finding structure in data that nobody labeled for you. The challenge is not the algorithm — it's defining what 'similar' means for your problem."*

> *"K-Means is to clustering what linear regression is to supervised learning: the first thing you try, the baseline everything else is measured against, and often the thing that works best in practice."*

---

## Table of Contents

1. [What Is Unsupervised Learning?](#1-what-is-unsupervised-learning)
2. [K-Means Clustering](#2-k-means-clustering)
   - 2.1 [The Algorithm: Lloyd's Method](#21-the-algorithm-lloyds-method)
   - 2.2 [K-Means as EM](#22-k-means-as-em)
   - 2.3 [K-Means from Scratch](#23-k-means-from-scratch)
   - 2.4 [Choosing K: The Elbow Method and Silhouette](#24-choosing-k-the-elbow-method-and-silhouette)
   - 2.5 [K-Means++: Smart Initialization](#25-k-means-smart-initialization)
   - 2.6 [Limitations of K-Means](#26-limitations-of-k-means)
3. [DBSCAN — Density-Based Clustering](#3-dbscan--density-based-clustering)
   - 3.1 [Core Points, Border Points, and Noise](#31-core-points-border-points-and-noise)
   - 3.2 [DBSCAN Algorithm](#32-dbscan-algorithm)
   - 3.3 [Choosing eps and min_samples](#33-choosing-eps-and-min_samples)
4. [Hierarchical Clustering](#4-hierarchical-clustering)
   - 4.1 [Agglomerative Clustering](#41-agglomerative-clustering)
   - 4.2 [Linkage Criteria](#42-linkage-criteria)
   - 4.3 [The Dendrogram](#43-the-dendrogram)
5. [Evaluating Clustering Quality](#5-evaluating-clustering-quality)
6. [Clustering with Scikit-learn](#6-clustering-with-scikit-learn)
7. [Case Study: Customer Segmentation](#7-case-study-customer-segmentation)
8. [Summary](#8-summary)
9. [Exercises](#9-exercises)
10. [References](#10-references)

---

## 1. What Is Unsupervised Learning?

In supervised learning, every training example has a label. In **unsupervised learning**, there are no labels — only inputs $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$. The algorithm must discover structure on its own.

**Clustering** is the most common unsupervised task: partition data into groups (clusters) such that:
- Points within a cluster are **similar** to each other
- Points in different clusters are **dissimilar**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles

rng = np.random.default_rng(42)

# Three different clustering scenarios
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

datasets = [
    ('Well-separated Gaussians\n(K-Means works perfectly)',
     *make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)),
    ('Crescent-shaped\n(K-Means fails, DBSCAN works)',
     *make_moons(n_samples=300, noise=0.08, random_state=42)),
    ('Varying density\n(DBSCAN struggles too)',
     *make_blobs(n_samples=[100, 200, 50],
                  centers=[[-2,0],[2,0],[0,3]],
                  cluster_std=[0.3, 1.0, 0.2], random_state=42)),
]

cmap = plt.cm.Set1
for ax, (title, X, y) in zip(axes, datasets):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=20, alpha=0.8,
               edgecolors='k', linewidths=0.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)

plt.suptitle('Clustering Scenarios — No Single Algorithm Wins on All',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_clustering_scenarios.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
Clustering applications:
  • Customer segmentation (group customers by behavior)
  • Document clustering (group similar documents/articles)
  • Image segmentation (group pixels by color/texture)
  • Anomaly detection (points that don't belong to any cluster)
  • Gene expression analysis (group genes by expression patterns)
  • Social network community detection
  • Recommendation systems (cluster users by taste)
""")
```

---

## 2. K-Means Clustering

### 2.1 The Algorithm: Lloyd's Method

K-Means partitions $n$ points into $K$ clusters by minimizing the **within-cluster sum of squares (WCSS)**:

$$J = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|^2$$

Where $\boldsymbol{\mu}_k$ is the centroid (mean) of cluster $k$.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

rng   = np.random.default_rng(42)
X, _  = make_blobs(n_samples=200, centers=3, cluster_std=0.7, random_state=42)

# ── Visualize K-Means steps ────────────────────────────────────────────────────
def kmeans_step_by_step(X, K, n_steps=6, seed=0):
    """Show K-Means convergence step by step."""
    rng_local = np.random.default_rng(seed)
    # Random initialization
    centroids = X[rng_local.choice(len(X), K, replace=False)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    cmap = plt.cm.Set1

    for step, ax in enumerate(axes.flatten()):
        if step > 0:
            # Assign step
            dists   = np.array([np.sum((X - c)**2, axis=1) for c in centroids])
            labels  = np.argmin(dists, axis=0)
            # Update step
            new_centroids = np.array([X[labels == k].mean(axis=0)
                                       for k in range(K)])
            centroids = new_centroids

        # Compute labels for display
        dists  = np.array([np.sum((X - c)**2, axis=1) for c in centroids])
        labels = np.argmin(dists, axis=0)
        wcss   = sum(np.sum((X[labels==k] - centroids[k])**2)
                     for k in range(K))

        # Plot
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap,
                   s=20, alpha=0.6, edgecolors='k', linewidths=0.2)
        for k, (cx, cy) in enumerate(centroids):
            ax.scatter(cx, cy, s=250, c=[cmap(k/K)], marker='*',
                       edgecolors='black', linewidths=1.5, zorder=5)
        ax.set_title(f'Step {step}: {"Init" if step==0 else "Assign+Update"}\nWCSS={wcss:.1f}',
                     fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('K-Means Lloyd\'s Algorithm Step by Step\n'
                 'Stars = centroids; Colors = cluster assignments',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/11_kmeans_steps.png', dpi=140, bbox_inches='tight')
    plt.show()

kmeans_step_by_step(X, K=3)

print("""
K-Means Algorithm (Lloyd's Method):
═══════════════════════════════════════════════════════
1. Initialize K centroids (randomly or K-Means++)
2. Repeat until convergence:
   a. ASSIGN:  label each point with its nearest centroid
              label[i] = argmin_k ||x[i] - μ_k||²
   b. UPDATE:  move each centroid to the mean of its assigned points
              μ_k = (1/|C_k|) Σ_{i∈C_k} x[i]
3. Converged when no assignments change

Convergence guarantee: WCSS decreases monotonically each iteration.
Time complexity: O(n·K·d·T) where T = number of iterations.
Space complexity: O(n·K) for distance matrix.

Important: K-Means finds a LOCAL minimum of WCSS, not necessarily global.
  → Run multiple times with different initializations, keep best WCSS.
═══════════════════════════════════════════════════════
""")
```

---

### 2.2 K-Means as EM

K-Means is a special case of the **Expectation-Maximization (EM)** algorithm:

- **E-step** (Assign): compute the expected cluster membership (hard assignment: each point belongs to exactly one cluster)
- **M-step** (Update): maximize the likelihood by updating centroids to the mean

The full **Gaussian Mixture Model (GMM)** generalizes K-Means with soft assignments and full covariance matrices — but that's a topic for Chapter 12.

```python
import numpy as np

print("""
K-Means as EM — The Connection:

K-Means minimizes: J = Σ_k Σ_{i∈C_k} ||x_i - μ_k||²

This is equivalent to maximum likelihood estimation of a Gaussian Mixture
Model where:
  • All clusters have equal, spherical covariance: σ²I
  • Assignments are hard (one cluster per point)

Full GMM (soft K-Means):
  • Each point has a soft probability of belonging to each cluster
  • Clusters can have different shapes (full covariance matrices)
  • EM gives closed-form updates for means, covariances, and mixing weights

Intuition:
  K-Means: 100% certain assignment → treats each point as owned by one cluster
  GMM:     Soft assignment → treats each point as partially belonging to all
  
  As the variance σ² → 0, GMM soft assignments become hard → reduces to K-Means
""")
```

---

### 2.3 K-Means from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansScratch:
    """
    K-Means clustering from scratch with K-Means++ initialization.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 n_init=10, init='k-means++', random_state=42):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.tol          = tol
        self.n_init       = n_init
        self.init         = init
        self.random_state = random_state

    def _init_centroids_random(self, X, rng):
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        return X[idx].copy()

    def _init_centroids_kmeanspp(self, X, rng):
        """
        K-Means++ initialization:
        1. Choose first centroid uniformly at random
        2. For each subsequent centroid: choose proportional to D²(x)
           where D(x) = distance to nearest already-chosen centroid
        """
        n = len(X)
        centroids = [X[rng.integers(0, n)]]

        for _ in range(1, self.n_clusters):
            # Distance to nearest centroid for each point
            dists = np.array([min(np.sum((x - c)**2) for c in centroids)
                              for x in X])
            # Sample proportional to squared distance
            probs = dists / dists.sum()
            idx   = rng.choice(n, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def _single_run(self, X, rng):
        """One complete K-Means run from initialization to convergence."""
        n, d = X.shape

        # Initialize centroids
        if self.init == 'k-means++':
            centroids = self._init_centroids_kmeanspp(X, rng)
        else:
            centroids = self._init_centroids_random(X, rng)

        labels   = np.zeros(n, dtype=int)
        wcss_history = []

        for iteration in range(self.max_iter):
            # E-step: assign each point to nearest centroid
            dists  = np.array([np.sum((X - c)**2, axis=1)
                               for c in centroids])     # (K, n)
            new_labels = np.argmin(dists, axis=0)       # (n,)

            # Compute WCSS
            wcss = sum(np.sum((X[new_labels == k] - centroids[k])**2)
                       for k in range(self.n_clusters)
                       if np.sum(new_labels == k) > 0)
            wcss_history.append(wcss)

            # Check convergence: no label changes
            if np.all(new_labels == labels) and iteration > 0:
                break
            labels = new_labels

            # M-step: update centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0
                else centroids[k]                   # Keep centroid if cluster is empty
                for k in range(self.n_clusters)
            ])

            # Check convergence: centroid movement
            centroid_shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids

            if centroid_shift < self.tol:
                break

        return labels, centroids, wcss_history[-1]

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        X   = np.array(X, dtype=float)

        best_labels    = None
        best_centroids = None
        best_wcss      = np.inf

        # Run n_init times with different initializations, keep best
        for _ in range(self.n_init):
            labels, centroids, wcss = self._single_run(X, rng)
            if wcss < best_wcss:
                best_wcss      = wcss
                best_labels    = labels
                best_centroids = centroids

        self.labels_         = best_labels
        self.cluster_centers_ = best_centroids
        self.inertia_        = best_wcss
        return self

    def predict(self, X):
        X     = np.array(X, dtype=float)
        dists = np.array([np.sum((X - c)**2, axis=1)
                          for c in self.cluster_centers_])
        return np.argmin(dists, axis=0)

    def fit_predict(self, X):
        return self.fit(X).labels_


# ── Test ───────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=42)

km_scratch = KMeansScratch(n_clusters=3, n_init=10)
km_scratch.fit(X)

km_sklearn = KMeans(n_clusters=3, n_init=10, random_state=42)
km_sklearn.fit(X)

# ARI: 1.0 = perfect, 0 = random
ari_scratch = adjusted_rand_score(y_true, km_scratch.labels_)
ari_sklearn  = adjusted_rand_score(y_true, km_sklearn.labels_)

print("K-Means — Scratch vs Sklearn:")
print(f"  Scratch WCSS: {km_scratch.inertia_:.2f}, ARI: {ari_scratch:.4f}")
print(f"  Sklearn WCSS: {km_sklearn.inertia_:.2f}, ARI: {ari_sklearn:.4f}")
```

---

### 2.4 Choosing K: The Elbow Method and Silhouette

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs
import matplotlib.cm as cm

X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)

k_range  = range(2, 11)
wcss     = []
sil_avgs = []
sil_stds = []

for k in k_range:
    km   = KMeans(n_clusters=k, n_init=10, random_state=42)
    lbls = km.fit_predict(X)
    wcss.append(km.inertia_)
    sil   = silhouette_samples(X, lbls)
    sil_avgs.append(sil.mean())
    sil_stds.append(sil.std())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ── Elbow plot ────────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(k_range, wcss, 'o-', color='steelblue', lw=2.5, markersize=8)
# Mark the elbow (largest second derivative)
diffs2 = np.diff(np.diff(wcss))
elbow  = list(k_range)[np.argmax(diffs2) + 1]
ax.axvline(elbow, color='tomato', lw=2, linestyle='--',
           label=f'Elbow at K={elbow}')
ax.set_xlabel('Number of Clusters K')
ax.set_ylabel('WCSS (Inertia)')
ax.set_title('Elbow Method\n(Look for the "elbow" — diminishing returns)')
ax.legend(); ax.grid(True, alpha=0.3)

# ── Silhouette plot ───────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.errorbar(k_range, sil_avgs, yerr=sil_stds, fmt='s-',
             color='tomato', lw=2.5, markersize=8, capsize=5)
ax2.axvline(list(k_range)[np.argmax(sil_avgs)], color='steelblue', lw=2,
            linestyle='--',
            label=f'Best K={list(k_range)[np.argmax(sil_avgs)]}')
ax2.set_xlabel('Number of Clusters K')
ax2.set_ylabel('Average Silhouette Score')
ax2.set_title('Silhouette Score\n(Higher is better; max=1, bad<0)')
ax2.legend(); ax2.grid(True, alpha=0.3)

# ── Silhouette analysis for best K ────────────────────────────────────────────
best_k = list(k_range)[np.argmax(sil_avgs)]
km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels_best = km_best.fit_predict(X)
sil_vals = silhouette_samples(X, labels_best)

ax3 = axes[2]
y_lower = 10
cmap_sil = cm.nipy_spectral

for k in range(best_k):
    sil_k = np.sort(sil_vals[labels_best == k])
    size_k = len(sil_k)
    y_upper = y_lower + size_k
    color   = cmap_sil(float(k) / best_k)
    ax3.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_k,
                       facecolor=color, edgecolor=color, alpha=0.7)
    ax3.text(-0.05, y_lower + size_k/2, str(k), fontsize=9)
    y_lower = y_upper + 5

ax3.axvline(sil_vals.mean(), color='tomato', lw=2, linestyle='--',
            label=f'Average = {sil_vals.mean():.3f}')
ax3.set_xlabel('Silhouette Coefficient')
ax3.set_ylabel('Cluster')
ax3.set_title(f'Silhouette Analysis — K={best_k}\n'
              f'Each band = one cluster, width = silhouette')
ax3.legend(fontsize=8)
ax3.set_yticks([])

plt.suptitle('Choosing K: Elbow Method and Silhouette Analysis',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_choosing_k.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"""
Silhouette Score explained:
  s(i) = (b(i) - a(i)) / max(a(i), b(i))
  
  a(i) = mean distance to other points in SAME cluster (cohesion)
  b(i) = mean distance to points in NEAREST OTHER cluster (separation)
  
  s(i) ≈ +1: point well-matched to its cluster
  s(i) ≈  0: point on the boundary between clusters
  s(i) ≈ -1: point likely in the WRONG cluster

Best K by elbow:     {elbow}
Best K by silhouette: {best_k}  (score = {max(sil_avgs):.4f})
True K in data:       4
""")
```

---

### 2.5 K-Means++: Smart Initialization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(n_samples=500, centers=5, cluster_std=1.0, random_state=42)

# Compare random init vs K-Means++ across many runs
n_trials = 30
random_inertias  = []
kpp_inertias     = []

rng = np.random.default_rng(0)
for seed in range(n_trials):
    km_rand = KMeans(n_clusters=5, init='random', n_init=1, random_state=seed)
    km_rand.fit(X)
    random_inertias.append(km_rand.inertia_)

    km_pp = KMeans(n_clusters=5, init='k-means++', n_init=1, random_state=seed)
    km_pp.fit(X)
    kpp_inertias.append(km_pp.inertia_)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.hist(random_inertias, bins=15, alpha=0.7, color='tomato',
        label=f'Random init\nmean={np.mean(random_inertias):.0f}')
ax.hist(kpp_inertias,    bins=15, alpha=0.7, color='steelblue',
        label=f'K-Means++ init\nmean={np.mean(kpp_inertias):.0f}')
ax.set_xlabel('WCSS (Inertia)')
ax.set_ylabel('Count')
ax.set_title(f'K-Means++ vs Random Initialization\n'
             f'({n_trials} trials, K=5)')
ax.legend(); ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.scatter(range(n_trials), sorted(random_inertias), s=40,
            color='tomato', label='Random', alpha=0.8)
ax2.scatter(range(n_trials), sorted(kpp_inertias), s=40,
            color='steelblue', label='K-Means++', alpha=0.8)
ax2.set_xlabel('Trial (sorted by inertia)')
ax2.set_ylabel('WCSS')
ax2.set_title('Sorted WCSS per Trial\n'
              'K-Means++ consistently finds lower WCSS')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/11_kmeanspp.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"Random init:   mean={np.mean(random_inertias):.1f}, "
      f"std={np.std(random_inertias):.1f}")
print(f"K-Means++ init: mean={np.mean(kpp_inertias):.1f}, "
      f"std={np.std(kpp_inertias):.1f}")
print(f"\nK-Means++ advantages:")
print(f"  - O(log K) approximation guarantee on WCSS")
print(f"  - Fewer iterations needed to converge")
print(f"  - Default in sklearn (always use it)")
```

---

### 2.6 Limitations of K-Means

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Limitation 1: Must specify K
# Limitation 2: Assumes spherical clusters
# Limitation 3: Sensitive to outliers
# Limitation 4: Not robust to different density clusters

scenarios = [
    ('Non-convex shapes\n(moons)',
     *make_moons(300, noise=0.05, random_state=0), 2),
    ('Concentric circles',
     *make_circles(300, noise=0.05, factor=0.5, random_state=0), 2),
    ('Different sizes',
     *make_blobs(n_samples=[50,300,100], centers=[[-2,0],[2,0],[0,3]],
                 cluster_std=[0.3,1.0,0.3], random_state=0), 3),
    ('Different densities',
     *make_blobs(n_samples=[200,50], centers=[[0,0],[5,0]],
                 cluster_std=[1.0, 0.2], random_state=0), 2),
]

cmap = plt.cm.Set1

for col, (title, X_s, y_s, K) in enumerate(scenarios):
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_s)
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    km.fit(X_sc)
    labels = km.labels_

    # Top row: true clusters
    axes[0, col].scatter(X_sc[:,0], X_sc[:,1], c=y_s, cmap=cmap,
                         s=18, alpha=0.8, edgecolors='k', linewidths=0.2)
    axes[0, col].set_title(f'{title}\n(True clusters)', fontsize=8)

    # Bottom row: K-Means result
    axes[1, col].scatter(X_sc[:,0], X_sc[:,1], c=labels, cmap=cmap,
                         s=18, alpha=0.8, edgecolors='k', linewidths=0.2)
    centers = km.cluster_centers_
    axes[1, col].scatter(centers[:,0], centers[:,1], s=200, c='gold',
                         marker='*', edgecolors='k', linewidths=1, zorder=5)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(y_s, labels)
    axes[1, col].set_title(f'K-Means result\nARI={ari:.3f}', fontsize=8)

for ax in axes.flatten():
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=6)

axes[0,0].set_ylabel('True Labels', fontsize=9)
axes[1,0].set_ylabel('K-Means Labels', fontsize=9)

plt.suptitle('K-Means Limitations\n'
             'Fails for non-convex, unequal size/density clusters',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_kmeans_limits.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

## 3. DBSCAN — Density-Based Clustering

### 3.1 Core Points, Border Points, and Noise

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters as dense regions separated by sparse regions.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=200, noise=0.07, random_state=42)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

eps         = 0.3
min_samples = 5

print(f"""
DBSCAN KEY CONCEPTS
════════════════════════════════════════════════════════════════════

Parameters:
  eps         (ε): Neighborhood radius.
                   A point's ε-neighborhood = all points within distance ε.
  min_samples (m): Minimum points required in ε-neighborhood for core point.

Point types:
  Core point:    Has ≥ min_samples neighbors within distance ε
                 → Dense region, definitely in a cluster
  Border point:  Has < min_samples neighbors, but is in the ε-neighborhood
                 of a core point → on the edge of a cluster
  Noise point:   Not a core point and not reachable from any core point
                 → Outlier, labeled -1

Connectivity:
  Directly density-reachable: point q is directly density-reachable from p
    if p is a core point AND q is in p's ε-neighborhood
  Density-reachable: chain of directly density-reachable steps
  Density-connected: both reachable from a common core point

A cluster = maximum set of density-connected points.

Advantages over K-Means:
  ✓ Does NOT require specifying K
  ✓ Finds arbitrary-shaped clusters
  ✓ Explicitly identifies outliers (noise points)
  ✓ Works for varying density if eps is tuned carefully
  
Disadvantages:
  ✗ Sensitive to eps and min_samples
  ✗ Struggles with very different densities across clusters
  ✗ O(n²) naive; O(n log n) with spatial index
════════════════════════════════════════════════════════════════════
""")

# Visualize core/border/noise classification
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(radius=eps).fit(X_sc)
neighborhoods = nbrs.radius_neighbors(X_sc, return_distance=False)
n_neighbors   = np.array([len(n) - 1 for n in neighborhoods])  # -1 for self

is_core   = n_neighbors >= min_samples
is_noise  = np.zeros(len(X_sc), dtype=bool)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
# Color by point type BEFORE actual clustering
types = np.where(is_core, 2, 1)  # Simplified: core=2, other=1
colors_pt = np.where(is_core, '#A5D6A7', '#90CAF9')

ax.scatter(X_sc[is_core, 0],  X_sc[is_core, 1],
           s=30, color='#A5D6A7', edgecolors='k',
           linewidths=0.4, label=f'Core points ({is_core.sum()})', zorder=4)
ax.scatter(X_sc[~is_core, 0], X_sc[~is_core, 1],
           s=30, color='#90CAF9', edgecolors='k',
           linewidths=0.4, label=f'Potential border/noise ({(~is_core).sum()})')

# Draw epsilon circle around a few core points
sample_cores = np.where(is_core)[0][:3]
for idx in sample_cores:
    circle = plt.Circle(X_sc[idx], eps, fill=False, color='tomato',
                        alpha=0.5, linewidth=1.5)
    ax.add_patch(circle)

ax.set_title(f'DBSCAN: Core Point Identification\n'
             f'ε={eps}, min_samples={min_samples}')
ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# Actual DBSCAN result
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=eps, min_samples=min_samples)
db_labels = db.fit_predict(X_sc)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise    = (db_labels == -1).sum()

ax2 = axes[1]
unique_labels = set(db_labels)
colors_db     = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 1)))

for k, col in zip(sorted(l for l in unique_labels if l != -1), colors_db):
    mask = db_labels == k
    ax2.scatter(X_sc[mask, 0], X_sc[mask, 1], s=25, color=col,
                edgecolors='k', linewidths=0.3, alpha=0.8, label=f'Cluster {k}')

if n_noise > 0:
    mask = db_labels == -1
    ax2.scatter(X_sc[mask, 0], X_sc[mask, 1], s=40, color='black',
                marker='x', linewidths=1.5, label=f'Noise ({n_noise})')

ax2.set_title(f'DBSCAN Result\n{n_clusters} clusters, {n_noise} noise points')
ax2.legend(fontsize=8); ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)

plt.suptitle('DBSCAN: Density-Based Clustering', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_dbscan.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 3.2 DBSCAN Algorithm

```python
import numpy as np

class DBSCANScratch:
    """DBSCAN from scratch using breadth-first search."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps         = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        n       = len(X)
        labels  = np.full(n, -2, dtype=int)  # -2 = unvisited
        cluster = 0

        # Precompute neighborhoods
        from sklearn.neighbors import NearestNeighbors
        nbrs   = NearestNeighbors(radius=self.eps).fit(X)
        neighs = nbrs.radius_neighbors(X, return_distance=False)

        for i in range(n):
            if labels[i] != -2:     # Already processed
                continue

            neighbors = neighs[i]

            if len(neighbors) < self.min_samples:
                labels[i] = -1      # Noise (for now)
                continue

            # Start new cluster
            labels[i] = cluster
            seeds      = list(neighbors)
            seeds.remove(i) if i in seeds else None

            j = 0
            while j < len(seeds):
                q = seeds[j]
                if labels[q] == -1:
                    labels[q] = cluster     # Border → assign to cluster
                if labels[q] != -2:
                    j += 1
                    continue
                labels[q] = cluster
                q_neighbors = neighs[q]
                if len(q_neighbors) >= self.min_samples:
                    seeds.extend([x for x in q_neighbors if x not in seeds])
                j += 1

            cluster += 1

        return labels


# Test
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

X, _ = make_moons(200, noise=0.07, random_state=42)
X_sc = StandardScaler().fit_transform(X)

scratch_labels = DBSCANScratch(eps=0.3, min_samples=5).fit_predict(X_sc)
sklearn_labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(X_sc)

from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(sklearn_labels, scratch_labels)
print(f"DBSCAN Scratch vs Sklearn: ARI = {ari:.6f}")
print(f"Scratch clusters: {len(set(scratch_labels)) - (1 if -1 in scratch_labels else 0)}")
print(f"Sklearn clusters: {len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)}")
```

---

### 3.3 Choosing eps and min_samples

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

X, _ = make_moons(300, noise=0.07, random_state=42)
X_sc  = StandardScaler().fit_transform(X)

# ── K-distance graph to choose eps ───────────────────────────────────────────
# Plot the distance to the k-th nearest neighbor (sorted)
# The "elbow" suggests a good value for eps
# Use k = min_samples - 1

k = 4   # min_samples = 5, so look at 4th neighbor
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_sc)
dists, _ = nbrs.kneighbors(X_sc)
k_dists  = np.sort(dists[:, k])[::-1]   # Sort descending

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.plot(range(len(k_dists)), k_dists, color='steelblue', lw=2)
ax.axhline(0.3, color='tomato', lw=2, linestyle='--',
           label='eps = 0.3 (elbow)')
ax.set_xlabel('Points sorted by k-distance')
ax.set_ylabel(f'{k}-th nearest neighbor distance')
ax.set_title(f'K-Distance Graph (k={k})\n'
             f'Elbow → good eps value')
ax.legend(); ax.grid(True, alpha=0.3)

# ── Grid of eps × min_samples ──────────────────────────────────────────────────
ax2 = axes[1]
eps_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
mps_vals = [3, 5, 8, 12]
results_grid = np.zeros((len(mps_vals), len(eps_vals)))

for i, mps in enumerate(mps_vals):
    for j, eps in enumerate(eps_vals):
        db = DBSCAN(eps=eps, min_samples=mps)
        lbls = db.fit_predict(X_sc)
        n_clust = len(set(lbls)) - (1 if -1 in lbls else 0)
        results_grid[i, j] = n_clust

im = ax2.imshow(results_grid, cmap='RdYlGn', aspect='auto',
                vmin=0, vmax=5)
ax2.set_xticks(range(len(eps_vals)))
ax2.set_xticklabels(eps_vals)
ax2.set_yticks(range(len(mps_vals)))
ax2.set_yticklabels(mps_vals)
ax2.set_xlabel('eps'); ax2.set_ylabel('min_samples')
ax2.set_title('Number of Clusters\n(Green=2 good; White=1 merge; Red=too many)')
for i in range(len(mps_vals)):
    for j in range(len(eps_vals)):
        ax2.text(j, i, int(results_grid[i,j]), ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax2)

# Best configuration
ax3 = axes[2]
db_best = DBSCAN(eps=0.3, min_samples=5)
best_labels = db_best.fit_predict(X_sc)
n_clust = len(set(best_labels)) - (1 if -1 in best_labels else 0)
n_noise = (best_labels == -1).sum()

colors_best = plt.cm.Set1(np.linspace(0, 1, max(n_clust, 1)))
for k, col in zip(sorted(l for l in set(best_labels) if l != -1), colors_best):
    mask = best_labels == k
    ax3.scatter(X_sc[mask,0], X_sc[mask,1], s=25, color=col,
                edgecolors='k', linewidths=0.3, label=f'Cluster {k}')
if n_noise:
    ax3.scatter(X_sc[best_labels==-1,0], X_sc[best_labels==-1,1],
                s=50, c='black', marker='x', label=f'Noise ({n_noise})')
ax3.set_title(f'Best DBSCAN (eps=0.3, min_samples=5)\n'
              f'{n_clust} clusters, {n_noise} noise')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

plt.suptitle('Tuning DBSCAN Parameters', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_dbscan_tuning.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

## 4. Hierarchical Clustering

### 4.1 Agglomerative Clustering

Agglomerative (bottom-up) hierarchical clustering:
1. Start: each point is its own cluster
2. Repeatedly merge the two closest clusters
3. Stop when all points are in one cluster (or when K clusters remain)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

X, y_true = make_blobs(n_samples=50, centers=3, cluster_std=0.7, random_state=42)

print("""
Agglomerative Clustering Algorithm:
════════════════════════════════════════════════════════════════════

Input: n points, linkage criterion
Initialize: n clusters {x₁}, {x₂}, ..., {xₙ}

While n_clusters > 1:
  1. Find the two clusters Cᵢ, Cⱼ with minimum inter-cluster distance
     (distance defined by linkage criterion)
  2. Merge Cᵢ and Cⱼ into one cluster
  3. Update distance matrix

The result is a hierarchy (dendrogram), not a flat partition.
You can cut the dendrogram at any level to get K clusters.

Advantages:
  ✓ No need to specify K upfront (decide after seeing dendrogram)
  ✓ Produces the full hierarchy of clusterings
  ✓ Deterministic (no random initialization)
  ✓ Can use any distance metric and linkage criterion

Disadvantages:
  ✗ O(n³) time, O(n²) memory (naive)
  ✗ Cannot undo a merge (greedy — may be suboptimal)
  ✗ Sensitive to linkage criterion and distance metric
════════════════════════════════════════════════════════════════════
""")
```

---

### 4.2 Linkage Criteria

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

print("""
LINKAGE CRITERIA — How to Define Distance Between Clusters
════════════════════════════════════════════════════════════════════

Single linkage:   d(A, B) = min_{a∈A, b∈B} d(a, b)
  = distance between CLOSEST pair of points
  Tends to create elongated, "chaining" clusters
  Good for non-convex clusters but sensitive to noise/outliers

Complete linkage: d(A, B) = max_{a∈A, b∈B} d(a, b)
  = distance between FURTHEST pair of points
  Creates compact, roughly equal-size clusters
  Sensitive to outliers (one outlier = max distance)

Average linkage:  d(A, B) = (1/|A||B|) Σ_{a∈A} Σ_{b∈B} d(a, b)
  = average over ALL pairs between clusters
  Balance between single and complete; less sensitive to outliers

Ward linkage:     d(A, B) = ΔWCSS when merging A and B
  = increase in within-cluster sum of squares
  Minimizes total variance; tends to produce equal-size spherical clusters
  DEFAULT in sklearn — usually the best choice

Centroid linkage: d(A, B) = d(centroid_A, centroid_B)
  Can cause inversions in the dendrogram
════════════════════════════════════════════════════════════════════
""")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
X_ds, y_ds = make_blobs(n_samples=200, centers=3, cluster_std=0.8, random_state=42)
X_moons, _ = make_moons(200, noise=0.07, random_state=42)

datasets  = [X_ds, X_moons]
ds_names  = ['Blob data', 'Moon data']
linkages  = ['ward', 'complete', 'average', 'single']
cmap_lnk  = plt.cm.Set1

for row, (X_d, ds_name) in enumerate(zip(datasets, ds_names)):
    sc    = StandardScaler()
    X_dsc = sc.fit_transform(X_d)
    n_cls = 2 if row == 1 else 3

    for col, link in enumerate(linkages):
        ax = axes[row, col]
        agg = AgglomerativeClustering(n_clusters=n_cls, linkage=link)
        lbls = agg.fit_predict(X_dsc)

        ax.scatter(X_dsc[:,0], X_dsc[:,1], c=lbls, cmap=cmap_lnk,
                   s=18, alpha=0.8, edgecolors='k', linewidths=0.2)
        from sklearn.metrics import adjusted_rand_score
        true = y_ds if row == 0 else np.zeros(200)
        ax.set_title(f'{ds_name}\n{link} linkage', fontsize=8)
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)

plt.suptitle('Linkage Criteria Comparison\nTop: Blobs, Bottom: Moons',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_linkage_criteria.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

### 4.3 The Dendrogram

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=30, centers=3, cluster_std=0.6, random_state=42)
X_sc = StandardScaler().fit_transform(X)

# Compute linkage matrix (Ward)
Z = linkage(X_sc, method='ward')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Full dendrogram ────────────────────────────────────────────────────────────
ax = axes[0]
dend = dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8,
                  color_threshold=3.5)
ax.axhline(3.5, color='tomato', lw=2, linestyle='--',
           label='Cut at height=3.5 → 3 clusters')
ax.set_title('Hierarchical Clustering Dendrogram (Ward)\n'
             'Height = merge cost; Horizontal cut = flat clustering')
ax.set_xlabel('Sample index')
ax.set_ylabel('Merge height (ΔWCSS)')
ax.legend(fontsize=8)

# ── Cutting the dendrogram ─────────────────────────────────────────────────────
ax2 = axes[1]
cut_heights = [8, 5, 3.5, 2]
x_pos = np.arange(len(cut_heights))
n_clusters_at_cut = [len(set(fcluster(Z, h, criterion='distance')))
                     for h in cut_heights]

ax2.barh(x_pos, n_clusters_at_cut, color='steelblue', alpha=0.8)
ax2.set_yticks(x_pos)
ax2.set_yticklabels([f'height={h}' for h in cut_heights])
ax2.set_xlabel('Number of Clusters')
ax2.set_title('Cutting the Dendrogram\n'
              'Higher cut → fewer clusters\nLower cut → more clusters')
for pos, n in zip(x_pos, n_clusters_at_cut):
    ax2.text(n + 0.1, pos, str(n), va='center', fontsize=10)
ax2.set_xlim(0, max(n_clusters_at_cut) + 1)
ax2.grid(axis='x', alpha=0.3)

plt.suptitle('Reading a Dendrogram: Height = Merging Cost',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_dendrogram.png', dpi=140, bbox_inches='tight')
plt.show()

# Where to cut: look for the largest vertical gap (long branches)
merge_heights = Z[:, 2]
gaps = np.diff(sorted(merge_heights))
best_cut = sorted(merge_heights)[np.argmax(gaps) + 1]
n_best = len(set(fcluster(Z, best_cut, criterion='distance')))
print(f"\nLargest gap in merge heights at: {best_cut:.3f}")
print(f"Suggested K at this cut: {n_best}")
print(f"(True K = 3)")
```

---

## 5. Evaluating Clustering Quality

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score,
                              davies_bouldin_score,
                              calinski_harabasz_score)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

print("""
CLUSTERING EVALUATION METRICS
════════════════════════════════════════════════════════════════════

EXTERNAL METRICS (require ground truth labels):
  Adjusted Rand Index (ARI):
    Measures similarity between predicted and true clusterings.
    Range: [-1, 1]. 1.0 = perfect, 0 = random, negative = worse than random.
    Adjusted for chance: random partitions score ≈ 0.
    
  Normalized Mutual Information (NMI):
    Information-theoretic similarity between clusterings.
    Range: [0, 1]. 1.0 = perfect agreement.
    Normalized by entropy, handles different numbers of clusters.

  Homogeneity:  Each cluster contains only members of a single class.
  Completeness: All members of a class are in the same cluster.
  V-measure:    Harmonic mean of homogeneity and completeness.

INTERNAL METRICS (no ground truth needed — use in practice):
  Silhouette Score:
    Cohesion vs separation for each point. Range: [-1, 1]. Higher is better.
    
  Davies-Bouldin Index:
    Average ratio of within-cluster scatter to between-cluster separation.
    Range: [0, ∞). LOWER is better.
    
  Calinski-Harabasz Score (Variance Ratio):
    Ratio of between-cluster dispersion to within-cluster dispersion.
    Range: [0, ∞). HIGHER is better.
════════════════════════════════════════════════════════════════════
""")

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
X_sc = StandardScaler().fit_transform(X)

algorithms = {
    'K-Means (K=4)':  KMeans(n_clusters=4, n_init=10, random_state=42),
    'K-Means (K=3)':  KMeans(n_clusters=3, n_init=10, random_state=42),
    'K-Means (K=6)':  KMeans(n_clusters=6, n_init=10, random_state=42),
    'Agglomerative':  AgglomerativeClustering(n_clusters=4, linkage='ward'),
    'DBSCAN':         DBSCAN(eps=0.4, min_samples=5),
}

print(f"\n{'Algorithm':<20} {'ARI':>7} {'NMI':>7} {'Silhouette':>12} "
      f"{'Davies-B':>10} {'Calinski':>10}")
print("-" * 72)

for name, alg in algorithms.items():
    lbls = alg.fit_predict(X_sc)
    n_clusters = len(set(lbls)) - (1 if -1 in lbls else 0)

    ari = adjusted_rand_score(y_true, lbls)
    nmi = normalized_mutual_info_score(y_true, lbls)

    if n_clusters >= 2 and len(set(lbls)) >= 2:
        sil = silhouette_score(X_sc, lbls)
        db  = davies_bouldin_score(X_sc, lbls)
        ch  = calinski_harabasz_score(X_sc, lbls)
    else:
        sil = db = ch = float('nan')

    print(f"{name:<20} {ari:>7.4f} {nmi:>7.4f} {sil:>12.4f} "
          f"{db:>10.4f} {ch:>10.1f}")
```

---

## 6. Clustering with Scikit-learn

```python
import numpy as np
from sklearn.cluster import (KMeans, MiniBatchKMeans, DBSCAN,
                               AgglomerativeClustering, MeanShift,
                               SpectralClustering)
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("""
sklearn Clustering API Quick Reference:
────────────────────────────────────────────────────────────────────────
Algorithm           Key Parameters              Best For
────────────────────────────────────────────────────────────────────────
KMeans              n_clusters, n_init,         General, large n, spherical
                    max_iter, init              Default: n_init=10, k-means++

MiniBatchKMeans     n_clusters, batch_size      Very large datasets (>100k)
                    max_iter                    Approximate K-Means with SGD

DBSCAN              eps, min_samples            Non-convex, outlier detection
                    metric, algorithm           No K needed

AgglomerativeClustering n_clusters, linkage,   Hierarchical, any shape with
                    affinity, connectivity      suitable linkage

MeanShift           bandwidth, bin_seeding      No K needed, finds modes
                                                Slow for large data

SpectralClustering  n_clusters, affinity,       Graph-based, non-convex
                    n_neighbors                 Slow: O(n³) eigendecomposition

GaussianMixture     n_components, covariance_   Soft assignments, elliptical
(not in cluster)    _type, max_iter             clusters, probabilistic
────────────────────────────────────────────────────────────────────────
""")

X, y = make_blobs(n_samples=1000, centers=4, random_state=42)
sc   = StandardScaler()
X_sc = sc.fit_transform(X)

# MiniBatchKMeans for large data
from sklearn.metrics import adjusted_rand_score
mbkm = MiniBatchKMeans(n_clusters=4, batch_size=100, n_init=10, random_state=42)
lbls_mb = mbkm.fit_predict(X_sc)
print(f"MiniBatchKMeans ARI: {adjusted_rand_score(y, lbls_mb):.4f}")

# Spectral clustering for non-convex
from sklearn.datasets import make_moons
X_m, y_m = make_moons(300, noise=0.07, random_state=42)
X_m_sc   = StandardScaler().fit_transform(X_m)
sc_clust = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                               n_neighbors=10, random_state=42)
lbls_sc  = sc_clust.fit_predict(X_m_sc)
print(f"SpectralClustering (moons) ARI: {adjusted_rand_score(y_m, lbls_sc):.4f}")
```

---

## 7. Case Study: Customer Segmentation

```python
# =============================================================================
# Customer Segmentation with K-Means
# Synthetic RFM (Recency, Frequency, Monetary) data
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

rng = np.random.default_rng(42)
n   = 1000

# ── 1. Generate synthetic RFM customer data ───────────────────────────────────
# RFM: Recency (days since last purchase), Frequency (# purchases), Monetary ($)
df = pd.DataFrame({
    'customer_id':    range(n),
    'recency':        np.clip(rng.exponential(30, n), 1, 365).astype(int),
    'frequency':      rng.negative_binomial(5, 0.4, n) + 1,
    'monetary':       np.clip(rng.lognormal(5.5, 1.2, n), 10, 5000).round(2),
    'age':            rng.integers(18, 75, n),
    'n_categories':   rng.integers(1, 8, n),
    'avg_basket_size':np.clip(rng.lognormal(4, 0.8, n), 5, 500).round(2),
})

print("Customer Dataset:")
print(df.describe().round(2))

# ── 2. Feature selection and scaling ─────────────────────────────────────────
features = ['recency', 'frequency', 'monetary', 'age', 'n_categories',
            'avg_basket_size']
X = df[features].values
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)

# ── 3. Find optimal K ─────────────────────────────────────────────────────────
k_range = range(2, 11)
wcss    = []
sil     = []

for k in k_range:
    km   = KMeans(n_clusters=k, n_init=10, random_state=42)
    lbls = km.fit_predict(X_sc)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X_sc, lbls))

best_k_sil = list(k_range)[np.argmax(sil)]
print(f"\nBest K by silhouette: {best_k_sil}")

# ── 4. Final clustering ────────────────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k_sil, n_init=20, random_state=42)
df['segment'] = km_final.fit_predict(X_sc)
segment_names = {0: 'Champions', 1: 'Loyal', 2: 'At-Risk',
                  3: 'New', 4: 'Dormant'}

# ── 5. Segment profiling ───────────────────────────────────────────────────────
profile = df.groupby('segment')[features].mean().round(2)
profile['size'] = df.groupby('segment').size()
print("\nSegment Profiles:")
print(profile)

# ── 6. Visualization ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Elbow and silhouette
ax = axes[0, 0]
ax2_twin = ax.twinx()
ax.plot(k_range, wcss, 'o-', color='steelblue', lw=2, label='WCSS')
ax2_twin.plot(k_range, sil, 's-', color='tomato', lw=2, label='Silhouette')
ax.axvline(best_k_sil, color='black', lw=1.5, linestyle='--')
ax.set_xlabel('K'); ax.set_ylabel('WCSS', color='steelblue')
ax2_twin.set_ylabel('Silhouette', color='tomato')
ax.set_title(f'Optimal K Selection\nBest K={best_k_sil}')
ax.grid(True, alpha=0.3)

# PCA projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)
cmap_seg = plt.cm.Set1

ax3 = axes[0, 1]
for seg in sorted(df['segment'].unique()):
    mask = df['segment'] == seg
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], s=15, alpha=0.6,
                cmap=cmap_seg, label=f'Seg {seg}',
                color=cmap_seg(seg / best_k_sil))
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax3.set_title('Customer Segments (PCA projection)')
ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

# Segment sizes
ax4 = axes[0, 2]
sizes  = df['segment'].value_counts().sort_index()
colors_pie = [cmap_seg(i/best_k_sil) for i in range(best_k_sil)]
ax4.pie(sizes, labels=[f'Seg {i}\n(n={s})' for i, s in sizes.items()],
        colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax4.set_title('Segment Sizes')

# RFM profiles
for ax_idx, (feat_x, feat_y) in enumerate([
    ('recency', 'monetary'),
    ('frequency', 'monetary'),
    ('recency', 'frequency')
]):
    ax5 = axes[1, ax_idx]
    for seg in sorted(df['segment'].unique()):
        mask = df['segment'] == seg
        ax5.scatter(df.loc[mask, feat_x], df.loc[mask, feat_y],
                    s=12, alpha=0.4, color=cmap_seg(seg / best_k_sil),
                    label=f'Seg {seg}')
    ax5.set_xlabel(feat_x); ax5.set_ylabel(feat_y)
    ax5.set_title(f'{feat_x} vs {feat_y}')
    ax5.legend(fontsize=6); ax5.grid(True, alpha=0.3)

plt.suptitle('Customer Segmentation with K-Means\n'
             'RFM + Demographic Features',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_customer_segmentation.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

## 8. Summary

### ✅ Must-Remember Mental Models

**1. K-Means objective:** Minimize WCSS = $\sum_k \sum_{i \in C_k} \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|^2$. The two-step Lloyd's algorithm (assign → update) is guaranteed to decrease WCSS monotonically.

**2. Always use K-Means++ initialization.** It gives $O(\log K)$ approximation guarantee and converges faster than random init. Default in sklearn.

**3. Run K-Means multiple times (`n_init=10`).** K-Means finds local optima — multiple runs with different inits give the best result.

**4. Choosing K — two complementary methods:**
- Elbow: plot WCSS vs K, look for diminishing returns
- Silhouette: $s(i) = (b-a)/\max(a,b)$, higher is better, peak at best K

**5. DBSCAN needs only two parameters:** `eps` (neighborhood radius) and `min_samples`. Use the k-distance graph to find eps: plot the k-th nearest neighbor distance (sorted), look for the "knee."

**6. DBSCAN advantages over K-Means:** no need to specify K, finds arbitrary shapes, explicitly identifies outliers (labeled -1). Limitation: struggles with clusters of very different densities.

**7. Agglomerative clustering linkage rules:**
- Ward (default): minimizes WCSS increase — best for compact spherical clusters
- Single: closest pair — elongated chains, sensitive to noise
- Complete: furthest pair — compact equal-size clusters
- Average: mean of all pairs — balanced

**8. The dendrogram cuts:** Longer vertical lines = larger merge cost. Cut at a height where you see the largest gap between consecutive merge heights.

**9. Evaluation without labels:** Use Silhouette Score or Davies-Bouldin. With labels: use ARI (adjusts for chance, range [-1,1]).

**10. Clustering is subjective.** There is no single "correct" clustering. The right algorithm depends on your data geometry and business question.

---

## 9. Exercises

1. **K-Means convergence proof**: Show that the WCSS objective strictly decreases (or stays the same) at each iteration of K-Means. Where does the algorithm terminate?

2. **Implement Mini-Batch K-Means**: Modify `KMeansScratch` to use stochastic gradient descent with mini-batches. Compare convergence speed vs. full K-Means on 100,000 2D points.

3. **Gaussian Mixture Model**: Implement the EM algorithm for a Gaussian Mixture Model (soft K-Means). Compare to K-Means on the moons dataset. Show that GMM gives soft cluster memberships.

4. **DBSCAN sensitivity**: On the moons dataset, systematically vary eps from 0.05 to 1.0 (with min_samples=5). Plot: number of clusters vs eps, noise points vs eps. Identify the stable region where DBSCAN gives the correct answer.

5. **Customer segmentation analysis**: Using the synthetic customer data from Section 7, interpret the segments business-wise. Write a brief "segment description" for each cluster (e.g., "Champions: high frequency, recent, high spend"). Suggest a marketing action for each segment.

6. **Challenge — K-Means for image compression**: Load a color image (use `matplotlib.image.imread`). Reshape to (n_pixels, 3). Apply K-Means with K ∈ {2, 4, 8, 16, 32}. Replace each pixel with its cluster centroid color. Compare compression ratios and visual quality.

---

## 10. References

- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Trans. Information Theory*, 28(2). — Original K-Means paper.
- Arthur, D., & Vassilvitskii, S. (2007). K-Means++: The Advantages of Careful Seeding. *SODA*. — K-Means++ initialization.
- Ester, M., et al. (1996). A Density-Based Algorithm for Discovering Clusters. *KDD*. — Original DBSCAN paper.
- Ward, J. H. (1963). Hierarchical Grouping to Optimize an Objective Function. *JASA*, 58(301). — Ward linkage.
- Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. *CSDA*, 20. — Silhouette score.

---
*Next: [Chapter 12 — Dimensionality Reduction: PCA, t-SNE, UMAP](12_dim_reduction.md)*
