# Chapter 7: Decision Trees & Rule-Based Learning

> *"Decision trees are the only machine learning model you can explain to your grandmother and your CEO using the same diagram."*

> *"Understanding decision trees deeply is not optional — they are the building blocks of Random Forests, Gradient Boosting, and XGBoost, which together dominate structured data competitions worldwide."*

---

## Table of Contents

1. [What Is a Decision Tree?](#1-what-is-a-decision-tree)
2. [The Anatomy of a Tree](#2-the-anatomy-of-a-tree)
3. [How Trees Learn: The CART Algorithm](#3-how-trees-learn-the-cart-algorithm)
   - 3.1 [Impurity Measures for Classification](#31-impurity-measures-for-classification)
   - 3.2 [Splitting Criterion](#32-splitting-criterion)
   - 3.3 [The Greedy Search](#33-the-greedy-search)
   - 3.4 [Splitting for Regression](#34-splitting-for-regression)
4. [Building a Tree from Scratch](#4-building-a-tree-from-scratch)
5. [Overfitting and Pruning](#5-overfitting-and-pruning)
   - 5.1 [Why Trees Overfit](#51-why-trees-overfit)
   - 5.2 [Pre-Pruning: Stopping Criteria](#52-pre-pruning-stopping-criteria)
   - 5.3 [Post-Pruning: Cost-Complexity Pruning](#53-post-pruning-cost-complexity-pruning)
6. [Feature Importance](#6-feature-importance)
7. [Decision Trees for Regression](#7-decision-trees-for-regression)
8. [Visualizing and Interpreting Trees](#8-visualizing-and-interpreting-trees)
9. [Decision Trees with Scikit-learn](#9-decision-trees-with-scikit-learn)
10. [Strengths and Weaknesses](#10-strengths-and-weaknesses)
11. [Case Study: Classifying Wine Quality](#11-case-study-classifying-wine-quality)
12. [Summary](#12-summary)
13. [Exercises](#13-exercises)
14. [References](#14-references)

---

## 1. What Is a Decision Tree?

A decision tree is a model that makes predictions by learning a **hierarchical sequence of if-then-else rules** from data. It partitions the feature space into rectangular regions, assigning a prediction to each region.

Consider predicting whether a loan applicant will default:

```
                    Income > $50k?
                   /              \
                NO                 YES
               /                     \
       Credit score > 600?         Debt ratio > 0.4?
       /              \            /               \
     NO               YES        YES               NO
     ↓                 ↓          ↓                 ↓
   DEFAULT          REPAY      DEFAULT            REPAY
```

No scaling, no matrix operations, no gradient required. Just a series of comparisons on the raw feature values. The model is fully interpretable: you can trace *exactly why* any prediction was made.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Visualize a hand-crafted decision tree as a diagram ──────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 8)
ax.axis('off')

def draw_node(ax, x, y, text, color='#E3F2FD', width=2.2, height=0.8):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle='round,pad=0.1',
                          facecolor=color, edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold',
            wrap=True, multialignment='center')

def draw_leaf(ax, x, y, text, color):
    box = FancyBboxPatch((x - 1.0, y - 0.5), 2.0, 1.0,
                          boxstyle='round,pad=0.1',
                          facecolor=color, edgecolor='gray', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

def draw_edge(ax, x1, y1, x2, y2, label, side):
    ax.annotate('', xy=(x2, y2+0.4), xytext=(x1, y1-0.4),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx + (0.3 if side=='right' else -0.3), my + 0.1,
            label, fontsize=8, color='#1565C0', fontweight='bold')

# Root
draw_node(ax, 7, 7, 'Income\n> $50k?')
# Level 1
draw_node(ax, 3.5, 5, 'Credit Score\n> 600?')
draw_node(ax, 10.5, 5, 'Debt Ratio\n> 0.4?')
# Level 2
draw_node(ax, 1.5, 3, 'Employment\n> 2 years?')
draw_node(ax, 5.5, 3, 'Loan Amount\n< $10k?')
draw_node(ax, 8.5, 3, 'Age\n> 30?')
# Leaves
draw_leaf(ax, 0.8, 1, '🔴 DEFAULT\n[High Risk]', '#FFCDD2')
draw_leaf(ax, 2.5, 1, '🟡 REVIEW\n[Med Risk]', '#FFF9C4')
draw_leaf(ax, 4.5, 1, '🟢 REPAY\n[Low Risk]', '#C8E6C9')
draw_leaf(ax, 6.8, 1, '🔴 DEFAULT\n[High Risk]', '#FFCDD2')
draw_leaf(ax, 7.8, 1, '🟡 REVIEW\n[Med Risk]', '#FFF9C4')  # dummy for spacing
draw_leaf(ax, 9.5, 1, '🔴 DEFAULT\n[High Risk]', '#FFCDD2')
draw_leaf(ax, 11.5, 1, '🟢 REPAY\n[Low Risk]', '#C8E6C9')

# Edges
draw_edge(ax, 7, 7, 3.5, 5, 'NO', 'left')
draw_edge(ax, 7, 7, 10.5, 5, 'YES', 'right')
draw_edge(ax, 3.5, 5, 1.5, 3, 'NO', 'left')
draw_edge(ax, 3.5, 5, 5.5, 3, 'YES', 'right')
draw_edge(ax, 10.5, 5, 8.5, 3, 'YES', 'left')
draw_edge(ax, 10.5, 5, 11.5, 1, 'NO', 'right')

ax.set_title('Decision Tree — Loan Default Prediction\n'
             'Each internal node = one feature test; each leaf = a prediction',
             fontsize=13, fontweight='bold', y=0.98)

# Annotations
ax.text(0.5, 0.5, 'Root Node\n(first split)', transform=ax.transAxes,
        fontsize=8, color='#1565C0', ha='right')

plt.tight_layout()
plt.savefig('figures/07_tree_diagram.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 2. The Anatomy of a Tree

```python
import numpy as np

print("""
DECISION TREE TERMINOLOGY
═══════════════════════════════════════════════════════════════════
Node Types:
  Root Node    : The topmost node; contains the first splitting rule.
                 Receives ALL training samples.
  Internal Node: A node with at least one child.
                 Applies a split condition: feature_j < threshold
  Leaf Node    : A terminal node with no children.
                 Stores the prediction: majority class (classification)
                 or mean target value (regression).

Tree Properties:
  Depth        : Length of the longest path from root to leaf.
                 Depth=0 → just a root (predicts the same for all).
  Branching    : CART always produces binary trees (two children per node).
  Pure Leaf    : A leaf where all samples belong to the same class.
                 Impurity = 0. Perfect (but likely overfit).

Key Hyperparameters:
  max_depth          : Maximum depth of tree.
  min_samples_split  : Minimum samples required to split a node.
  min_samples_leaf   : Minimum samples required in a leaf.
  max_features       : Number of features to consider at each split.
  min_impurity_decrease : Minimum impurity reduction to make a split.
  ccp_alpha          : Cost-complexity pruning parameter (post-pruning).

The fundamental bias-variance tradeoff for trees:
  Deeper tree  → lower bias, higher variance  → overfitting
  Shallower tree → higher bias, lower variance → underfitting
═══════════════════════════════════════════════════════════════════
""")

# ── Visualize a tree's partition of feature space ─────────────────────────────
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

rng = np.random.default_rng(42)
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                            n_informative=2, random_state=42)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
depths = [1, 2, 4, None]  # None = unlimited depth

for ax, depth in zip(axes, depths):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-0.3, X[:,0].max()+0.3, 300),
        np.linspace(X[:,1].min()-0.3, X[:,1].max()+0.3, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.35,
                colors=['#EF9A9A', '#A5D6A7'])
    ax.contour(xx, yy, Z, colors=['black'], linewidths=0.8, alpha=0.5)
    ax.scatter(X[:,0], X[:,1], c=y, cmap='RdYlGn',
               s=18, edgecolors='k', linewidths=0.3, alpha=0.8)
    train_acc = clf.score(X, y)
    n_leaves  = clf.get_n_leaves()
    depth_label = f'depth={depth}' if depth else 'depth=None'
    ax.set_title(f'{depth_label}\n{n_leaves} leaves, train acc={train_acc:.2%}',
                 fontsize=9)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂' if depth == 1 else '')
    ax.tick_params(labelsize=7)

plt.suptitle('Decision Boundaries at Different Tree Depths\n'
             'Each rectangle = a leaf region',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_tree_partitions.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 3. How Trees Learn: The CART Algorithm

**CART** (Classification and Regression Trees, Breiman et al. 1984) is the algorithm behind scikit-learn's `DecisionTreeClassifier` and `DecisionTreeRegressor`.

The core idea: **at each node, find the feature $j$ and threshold $t$ that produces the best split** — where "best" means the resulting child nodes are as pure as possible.

### 3.1 Impurity Measures for Classification

**What is impurity?** A node is impure when it contains samples from multiple classes. A perfectly pure node has impurity = 0.

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Three impurity measures ───────────────────────────────────────────────────

def gini_impurity(p):
    """
    Gini Impurity: G = 1 - Σ_k p_k²
    Probability of misclassifying a randomly chosen sample.
    Range: [0, 1 - 1/K]
    Max for K classes: 1 - 1/K  (achieved at uniform distribution)
    """
    p = np.array(p)
    return 1 - np.sum(p**2)

def entropy(p, base=2):
    """
    Entropy (Information Gain): H = -Σ_k p_k log(p_k)
    Expected information content. Range: [0, log(K)]
    Max: log₂(K) bits (achieved at uniform distribution)
    """
    p = np.array(p)
    p = p[p > 0]   # 0 * log(0) = 0 by convention
    return -np.sum(p * np.log2(p)) if base == 2 else -np.sum(p * np.log(p))

def misclassification_error(p):
    """
    Misclassification Error: E = 1 - max_k(p_k)
    Fraction of samples that would be misclassified.
    Range: [0, 1 - 1/K]
    Less sensitive than Gini/Entropy — rarely used for splitting.
    """
    return 1 - np.max(p)

# ── Compare on binary classification (K=2) ───────────────────────────────────
p1 = np.linspace(0.001, 0.999, 500)   # P(class 1)
p0 = 1 - p1                            # P(class 0)

gini  = [gini_impurity([p, 1-p]) for p in p1]
ent   = [entropy([p, 1-p])       for p in p1]
mce   = [misclassification_error([p, 1-p]) for p in p1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(p1, gini, color='steelblue', lw=2.5, label='Gini Impurity')
ax.plot(p1, np.array(ent)/2, color='tomato', lw=2.5,
        label='Entropy/2 (normalized)')
ax.plot(p1, mce, color='green', lw=2.5, label='Misclassification Error')
ax.axvline(0.5, color='gray', lw=1, linestyle='--', label='Max impurity at p=0.5')
ax.set_xlabel('P(class 1)')
ax.set_ylabel('Impurity')
ax.set_title('Binary Impurity Measures\n'
             'All peak at p=0.5 (maximum uncertainty)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Numerical comparison at key points
print("Impurity at key class distributions:")
print(f"\n{'Distribution':<30} {'Gini':>8} {'Entropy':>10} {'Mis.Err':>10}")
print("-" * 62)

examples = [
    ([1.0, 0.0],          'Pure class 0  [1, 0]'),
    ([0.9, 0.1],          '90% class 0   [0.9, 0.1]'),
    ([0.7, 0.3],          '70% class 0   [0.7, 0.3]'),
    ([0.5, 0.5],          '50/50 split   [0.5, 0.5]'),
    ([0.33, 0.33, 0.34],  '3-class equal [0.33,0.33,0.34]'),
    ([0.8, 0.1, 0.1],     '3-class skewed [0.8,0.1,0.1]'),
]

for probs, label in examples:
    g = gini_impurity(probs)
    e = entropy(probs)
    m = misclassification_error(probs)
    print(f"{label:<30} {g:>8.4f} {e:>10.4f} {m:>10.4f}")

# ── Visualize impurity reduction from a split ─────────────────────────────────
ax2 = axes[1]

# Parent node: [50% class 0, 50% class 1] — maximum impurity
# Split produces: Left=[80% class 0, 20% class 1], Right=[20% class 0, 80% class 1]
parent = [0.5, 0.5]
left   = [0.8, 0.2]    # n_left  = 60 samples
right  = [0.2, 0.8]    # n_right = 40 samples
n_left, n_right, n_total = 60, 40, 100

g_parent = gini_impurity(parent)
g_left   = gini_impurity(left)
g_right  = gini_impurity(right)
g_weighted = (n_left/n_total * g_left + n_right/n_total * g_right)
ig_gini  = g_parent - g_weighted

e_parent = entropy(parent)
e_left   = entropy(left)
e_right  = entropy(right)
e_weighted = (n_left/n_total * e_left + n_right/n_total * e_right)
ig_entropy = e_parent - e_weighted

categories = ['Parent\n[50/50]', 'Left child\n[80/20]', 'Right child\n[20/80]']
gini_vals  = [g_parent, g_left, g_right]
ent_vals   = [e_parent, e_left, e_right]

x_pos = np.arange(3)
width = 0.35
ax2.bar(x_pos - width/2, gini_vals, width, color='steelblue', alpha=0.8, label='Gini')
ax2.bar(x_pos + width/2, np.array(ent_vals)/2, width, color='tomato', alpha=0.8,
        label='Entropy/2')
ax2.set_xticks(x_pos); ax2.set_xticklabels(categories)
ax2.set_ylabel('Impurity')
ax2.set_title(f'Impurity Before and After a Split\n'
              f'ΔGini = {ig_gini:.4f},  ΔEntropy = {ig_entropy:.4f} bits')
ax2.legend(); ax2.grid(axis='y', alpha=0.3)

# Annotate the information gain
ax2.annotate(f'Information gain\n(Gini): {ig_gini:.4f}',
             xy=(0, g_parent), xytext=(0.5, g_parent+0.05),
             arrowprops=dict(arrowstyle='->', color='steelblue'),
             color='steelblue', fontsize=9)

plt.tight_layout()
plt.savefig('figures/07_impurity_measures.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"\nExample split: parent Gini={g_parent:.4f}")
print(f"  Left  child: n={n_left}, Gini={g_left:.4f}")
print(f"  Right child: n={n_right}, Gini={g_right:.4f}")
print(f"  Weighted child Gini: {g_weighted:.4f}")
print(f"  Gini Gain: {ig_gini:.4f}   ← maximize this")
```

---

### 3.2 Splitting Criterion

At each node, CART evaluates ALL possible splits $(j, t)$ for each feature $j$ and all possible thresholds $t$, and selects the one maximizing the impurity reduction:

$$\text{Gain}(j, t) = \text{Impurity}(\text{parent}) - \frac{n_L}{n}\text{Impurity}(L) - \frac{n_R}{n}\text{Impurity}(R)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# ── Find the best split for a single feature ──────────────────────────────────
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)

def information_gain_gini(y_parent, y_left, y_right):
    """Weighted Gini impurity reduction."""
    n = len(y_parent)
    n_L, n_R = len(y_left), len(y_right)
    if n_L == 0 or n_R == 0:
        return 0.0
    g_parent = gini_impurity(y_parent)
    g_left   = gini_impurity(y_left)
    g_right  = gini_impurity(y_right)
    return g_parent - (n_L/n * g_left + n_R/n * g_right)

def find_best_split_for_feature(X_feature, y):
    """
    Find the threshold that maximizes information gain for a single feature.
    Evaluates all midpoints between consecutive unique values.
    """
    # Sort by feature value
    sort_idx  = np.argsort(X_feature)
    X_sorted  = X_feature[sort_idx]
    y_sorted  = y[sort_idx]

    best_gain      = -np.inf
    best_threshold = None
    gains = []
    thresholds = []

    # Try splitting at midpoint between each consecutive pair of unique values
    unique_vals = np.unique(X_sorted)
    candidate_thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

    for t in candidate_thresholds:
        left_mask  = X_feature <= t
        right_mask = ~left_mask
        gain = information_gain_gini(y, y[left_mask], y[right_mask])
        gains.append(gain)
        thresholds.append(t)

        if gain > best_gain:
            best_gain      = gain
            best_threshold = t

    return best_threshold, best_gain, thresholds, gains


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, feat_idx, feat_name in zip(
    axes.flatten(),
    range(4),
    iris.feature_names
):
    threshold, gain, all_thresholds, all_gains = find_best_split_for_feature(
        X[:, feat_idx], y
    )

    # Top: scatter of feature values colored by class
    ax_top = ax
    for cls, color, name in zip([0,1,2], ['tomato','green','steelblue'], iris.target_names):
        mask = y == cls
        ax_top.scatter(X[mask, feat_idx],
                       [cls + np.random.uniform(-0.1, 0.1) for _ in range(mask.sum())],
                       s=15, alpha=0.5, color=color, label=name)
    ax_top.axvline(threshold, color='black', lw=2.5, linestyle='--',
                   label=f'Best threshold = {threshold:.2f}\nGain = {gain:.4f}')
    ax_top.set_xlabel(feat_name); ax_top.set_ylabel('Class (jittered)')
    ax_top.set_title(f'{feat_name}\nBest split: {threshold:.2f}, Gain = {gain:.4f}')
    ax_top.legend(fontsize=7)
    ax_top.grid(True, alpha=0.3)

plt.suptitle('Best Split for Each Feature — Iris Dataset\n'
             'CART evaluates all candidate thresholds and picks max Gini gain',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_best_splits.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Find the globally best split (across all features) ────────────────────────
print("Best split for each feature (Iris dataset):")
print(f"\n{'Feature':<30} {'Best Threshold':>16} {'Gini Gain':>12}")
print("-" * 62)

best_overall_gain = -np.inf
best_overall_feat = None
best_overall_thresh = None

for feat_idx, feat_name in enumerate(iris.feature_names):
    thresh, gain, _, _ = find_best_split_for_feature(X[:, feat_idx], y)
    marker = " ← CHOSEN" if gain > best_overall_gain else ""
    if gain > best_overall_gain:
        best_overall_gain   = gain
        best_overall_feat   = feat_name
        best_overall_thresh = thresh
    print(f"{feat_name:<30} {thresh:>16.4f} {gain:>12.4f}{marker}")

print(f"\nRoot split: {best_overall_feat} <= {best_overall_thresh:.4f}")
print(f"Gini gain: {best_overall_gain:.4f}")
```

---

### 3.3 The Greedy Search

CART is a **greedy algorithm**: at each node, it makes the locally optimal split. It does not backtrack. This means:
- Finding the globally optimal tree is NP-hard
- CART finds a good tree quickly, but not necessarily the best tree
- The greedy nature contributes to variance → addressed by ensembles (Chapter 8)

```python
import numpy as np

print("""
CART Algorithm Pseudocode:
══════════════════════════════════════════════════════════════════
BuildTree(X, y, depth=0):
  
  1. If stopping criteria met:
       Return leaf node with prediction = majority_class(y)  [or mean(y)]
  
  2. For each feature j in {1, ..., d}:
       For each candidate threshold t in candidates(X[:, j]):
         Compute Gain(j, t) = Gini(y) - weighted_Gini(y_left, y_right)
  
  3. (j*, t*) = argmax_{j,t} Gain(j, t)
  
  4. Split data:
       X_left,  y_left  = samples where X[:, j*] <= t*
       X_right, y_right = samples where X[:, j*] >  t*
  
  5. Return InternalNode(
       feature = j*,
       threshold = t*,
       left  = BuildTree(X_left,  y_left,  depth+1),
       right = BuildTree(X_right, y_right, depth+1)
     )

Stopping Criteria (pre-pruning):
  • depth >= max_depth
  • |samples in node| < min_samples_split
  • |samples in best child| < min_samples_leaf
  • All samples have the same class label (pure node)
  • Best gain < min_impurity_decrease
══════════════════════════════════════════════════════════════════

Time complexity:
  At each node: O(d × n log n) to find best split
    d = number of features
    n = samples at node (sorting each feature takes O(n log n))
  Total: O(d × n log n × n_nodes)
         ≈ O(d × n² log n) for a balanced tree of depth log(n)

Space complexity:
  O(n_nodes) for the tree structure
  O(n) for sample tracking during training
""")
```

---

### 3.4 Splitting for Regression

For regression trees, the impurity measure is the **variance** (or equivalently, the mean squared error) of the target values in each node:

$$\text{Variance}(\mathcal{S}) = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} (y^{(i)} - \bar{y}_\mathcal{S})^2$$

$$\text{Gain}(j, t) = \text{Var}(parent) - \frac{n_L}{n}\text{Var}(L) - \frac{n_R}{n}\text{Var}(R)$$

```python
import numpy as np

def variance_reduction(y_parent, y_left, y_right):
    """Variance reduction (regression splitting criterion)."""
    n = len(y_parent)
    n_L, n_R = len(y_left), len(y_right)
    if n_L == 0 or n_R == 0:
        return 0.0
    return (y_parent.var() -
            n_L/n * y_left.var() -
            n_R/n * y_right.var())

# Show that splitting reduces variance
np.random.seed(42)
y_parent = np.array([1.2, 3.5, 1.1, 8.9, 9.2, 3.8, 1.0, 8.5, 3.2, 9.1])

# Bad split
y_L1 = y_parent[:5];  y_R1 = y_parent[5:]
# Good split (separates high and low values)
y_L2 = y_parent[y_parent < 5]; y_R2 = y_parent[y_parent >= 5]

print("Variance reduction — regression trees:")
print(f"\nParent variance: {y_parent.var():.4f}")
print(f"\nBad split:  Left var={y_L1.var():.4f}, Right var={y_R1.var():.4f}")
print(f"  Gain = {variance_reduction(y_parent, y_L1, y_R1):.4f}")
print(f"\nGood split: Left var={y_L2.var():.4f}, Right var={y_R2.var():.4f}")
print(f"  Gain = {variance_reduction(y_parent, y_L2, y_R2):.4f}  ← much better")
print(f"\nLeaf prediction = mean of samples in that leaf")
print(f"  Left leaf:  mean = {y_L2.mean():.2f}")
print(f"  Right leaf: mean = {y_R2.mean():.2f}")
```

---

## 4. Building a Tree from Scratch

```python
import numpy as np
from collections import Counter

class Node:
    """A single node in a decision tree."""
    def __init__(self):
        # For internal nodes
        self.feature_idx  = None   # Which feature to split on
        self.threshold    = None   # Split threshold
        self.left         = None   # Left child (feature <= threshold)
        self.right        = None   # Right child (feature > threshold)
        self.gain         = None   # Impurity gain of this split
        # For leaf nodes
        self.is_leaf      = False
        self.prediction   = None   # Majority class (classification) or mean (regression)
        self.n_samples    = None   # How many training samples reached this node
        self.class_counts = None   # Class distribution at this leaf

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(pred={self.prediction}, n={self.n_samples})"
        return (f"Node(feature={self.feature_idx}, "
                f"threshold={self.threshold:.4f}, gain={self.gain:.4f})")


class DecisionTreeScratch:
    """
    CART Decision Tree for binary and multi-class classification.
    Implements:
      - Gini impurity splitting
      - Pre-pruning (max_depth, min_samples_split, min_samples_leaf)
      - Prediction with majority-class leaves
      - Feature importance (mean impurity decrease)
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0,
                 random_state=None):
        self.max_depth           = max_depth
        self.min_samples_split   = min_samples_split
        self.min_samples_leaf    = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state        = random_state
        self.root_                = None
        self.n_features_          = None
        self.n_classes_           = None
        self.feature_importances_ = None

    # ── Impurity ───────────────────────────────────────────────────────────────
    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        probs  = counts / len(y)
        return 1.0 - np.sum(probs**2)

    # ── Best split ─────────────────────────────────────────────────────────────
    def _best_split(self, X, y):
        n_samples  = len(y)
        g_parent   = self._gini(y)
        best_gain  = self.min_impurity_decrease
        best_feat  = None
        best_thresh = None

        for feat_idx in range(self.n_features_):
            feat_vals   = X[:, feat_idx]
            unique_vals = np.unique(feat_vals)
            # Candidate thresholds: midpoints between consecutive unique values
            thresholds  = (unique_vals[:-1] + unique_vals[1:]) / 2

            for t in thresholds:
                left_mask  = feat_vals <= t
                right_mask = ~left_mask
                n_L = left_mask.sum()
                n_R = right_mask.sum()

                if n_L < self.min_samples_leaf or n_R < self.min_samples_leaf:
                    continue

                g_L = self._gini(y[left_mask])
                g_R = self._gini(y[right_mask])
                gain = g_parent - (n_L/n_samples * g_L + n_R/n_samples * g_R)

                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = feat_idx
                    best_thresh = t

        return best_feat, best_thresh, best_gain

    # ── Recursive tree builder ─────────────────────────────────────────────────
    def _build(self, X, y, depth, importance_tracker):
        node = Node()
        node.n_samples    = len(y)
        node.class_counts = np.bincount(y, minlength=self.n_classes_)
        node.prediction   = int(np.argmax(node.class_counts))

        # Stopping criteria
        stop = (
            self._gini(y) == 0 or                          # Pure node
            len(y) < self.min_samples_split or             # Too few samples
            (self.max_depth is not None and depth >= self.max_depth)
        )

        if stop:
            node.is_leaf = True
            return node

        # Find best split
        feat_idx, threshold, gain = self._best_split(X, y)

        if feat_idx is None:   # No valid split found
            node.is_leaf = True
            return node

        # Record gain for feature importance
        importance_tracker[feat_idx] += gain * len(y)

        # Split data
        left_mask  = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        node.feature_idx = feat_idx
        node.threshold   = threshold
        node.gain        = gain
        node.left  = self._build(X[left_mask],  y[left_mask],  depth+1, importance_tracker)
        node.right = self._build(X[right_mask], y[right_mask], depth+1, importance_tracker)

        return node

    # ── Public interface ───────────────────────────────────────────────────────
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.n_classes_  = len(np.unique(y))
        importance_tracker = np.zeros(self.n_features_)

        self.root_ = self._build(X, y, depth=0,
                                  importance_tracker=importance_tracker)

        # Normalize feature importances
        total = importance_tracker.sum()
        self.feature_importances_ = (importance_tracker / total
                                     if total > 0 else importance_tracker)
        return self

    def _predict_one(self, x, node):
        """Traverse the tree for a single sample."""
        if node.is_leaf:
            return node.prediction
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root_) for x in X])

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_depth(self, node=None):
        if node is None:
            node = self.root_
        if node.is_leaf:
            return 0
        return 1 + max(self.get_depth(node.left),
                       self.get_depth(node.right))

    def get_n_leaves(self, node=None):
        if node is None:
            node = self.root_
        if node.is_leaf:
            return 1
        return self.get_n_leaves(node.left) + self.get_n_leaves(node.right)

    def print_tree(self, node=None, depth=0, feature_names=None):
        if node is None:
            node = self.root_
        indent = "│   " * depth
        if node.is_leaf:
            print(f"{indent}└── Leaf: class={node.prediction}, "
                  f"n={node.n_samples}, dist={node.class_counts}")
        else:
            fname = (feature_names[node.feature_idx]
                     if feature_names else f"feature_{node.feature_idx}")
            print(f"{indent}├── [{fname} <= {node.threshold:.4f}]  "
                  f"gain={node.gain:.4f}, n={node.n_samples}")
            self.print_tree(node.left,  depth+1, feature_names)
            self.print_tree(node.right, depth+1, feature_names)


# ── Test on Iris ──────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# Our implementation
my_tree = DecisionTreeScratch(max_depth=3, min_samples_leaf=1)
my_tree.fit(X_tr, y_tr)

# Scikit-learn reference
sk_tree  = DecisionTreeClassifier(max_depth=3, random_state=42)
sk_tree.fit(X_tr, y_tr)

print("Decision Tree from Scratch vs Scikit-learn:")
print(f"\n  Scratch  — Train: {my_tree.score(X_tr, y_tr):.4f}, "
      f"Test: {my_tree.score(X_te, y_te):.4f}, "
      f"Depth: {my_tree.get_depth()}, Leaves: {my_tree.get_n_leaves()}")
print(f"  Sklearn  — Train: {sk_tree.score(X_tr, y_tr):.4f}, "
      f"Test: {sk_tree.score(X_te, y_te):.4f}, "
      f"Depth: {sk_tree.get_depth()}, Leaves: {sk_tree.get_n_leaves()}")

print("\nTree Structure (from scratch):")
my_tree.print_tree(feature_names=iris.feature_names)

print("\nFeature Importances:")
for name, imp in zip(iris.feature_names, my_tree.feature_importances_):
    bar = '█' * int(imp * 40)
    print(f"  {name:<30}: {imp:.4f}  {bar}")
```

---

## 5. Overfitting and Pruning

### 5.1 Why Trees Overfit

A fully grown tree (no stopping criteria) will **perfectly memorize the training data** by creating one leaf per sample. Training accuracy = 100%. Test accuracy = terrible.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

rng = np.random.default_rng(42)
X, y = make_classification(n_samples=400, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

depths = range(1, 21)
train_scores, test_scores, cv_scores = [], [], []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_tr, y_tr)
    train_scores.append(clf.score(X_tr, y_tr))
    test_scores.append( clf.score(X_te, y_te))
    cv_scores.append(cross_val_score(clf, X_tr, y_tr, cv=5).mean())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(depths, train_scores, 'o-', color='steelblue', lw=2, label='Train accuracy')
ax.plot(depths, test_scores,  's-', color='tomato',    lw=2, label='Test accuracy')
ax.plot(depths, cv_scores,    '^-', color='green',     lw=2, label='CV accuracy (5-fold)')
best_depth = depths[np.argmax(cv_scores)]
ax.axvline(best_depth, color='black', lw=2, linestyle='--',
           label=f'Best CV depth = {best_depth}')
ax.set_xlabel('Max Depth'); ax.set_ylabel('Accuracy')
ax.set_title('Overfitting in Decision Trees\n'
             'Train accuracy always increases; test peaks then drops')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.05)

# Number of leaves vs depth
ax2 = axes[1]
n_leaves = []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_tr, y_tr)
    n_leaves.append(clf.get_n_leaves())

ax2.bar(depths, n_leaves, color='steelblue', alpha=0.8)
ax2.axvline(best_depth, color='tomato', lw=2, linestyle='--',
            label=f'Best CV depth = {best_depth}')
ax2.set_xlabel('Max Depth'); ax2.set_ylabel('Number of Leaves')
ax2.set_title('Tree Complexity vs Depth\n'
              'Each leaf = one rectangular region in feature space')
ax2.legend(); ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/07_overfitting.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"Fully grown tree (no max_depth):")
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_tr, y_tr)
print(f"  Train accuracy: {clf_full.score(X_tr, y_tr):.4f}")
print(f"  Test accuracy:  {clf_full.score(X_te, y_te):.4f}")
print(f"  Depth: {clf_full.get_depth()}, Leaves: {clf_full.get_n_leaves()}")
print(f"\nPruned tree (max_depth={best_depth}):")
clf_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
clf_pruned.fit(X_tr, y_tr)
print(f"  Train accuracy: {clf_pruned.score(X_tr, y_tr):.4f}")
print(f"  Test accuracy:  {clf_pruned.score(X_te, y_te):.4f}")
print(f"  Depth: {clf_pruned.get_depth()}, Leaves: {clf_pruned.get_n_leaves()}")
```

---

### 5.2 Pre-Pruning: Stopping Criteria

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# Grid search over pre-pruning parameters
param_grid = {
    'max_depth':          [3, 5, 7, 10, None],
    'min_samples_split':  [2, 5, 10, 20],
    'min_samples_leaf':   [1, 3, 5, 10],
    'min_impurity_decrease': [0.0, 0.001, 0.005, 0.01],
}

gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
gs.fit(X_tr, y_tr)

print(f"Best pre-pruning parameters: {gs.best_params_}")
print(f"CV AUC: {gs.best_score_:.4f}")
print(f"Test AUC: {__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(y_te, gs.predict_proba(X_te)[:,1]):.4f}")
```

---

### 5.3 Post-Pruning: Cost-Complexity Pruning

**Cost-complexity pruning** (also called "weakest link pruning") systematically removes subtrees that do not sufficiently reduce impurity relative to tree complexity.

The cost-complexity measure:

$$R_\alpha(T) = R(T) + \alpha \cdot |T|$$

Where $R(T)$ is the misclassification rate, $|T|$ is the number of leaves, and $\alpha$ is the **pruning parameter** (analogous to regularization strength).

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# Get the full sequence of pruning alphas
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_tr, y_tr)

path = clf_full.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas  = path.ccp_alphas
impurities  = path.impurities

# Train a tree at each alpha level
clfs      = []
train_aucs = []
cv_aucs    = []
test_aucs  = []
n_leaves   = []

for alpha in ccp_alphas[:-1]:  # Last alpha gives a single-node tree
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    clf.fit(X_tr, y_tr)
    clfs.append(clf)
    n_leaves.append(clf.get_n_leaves())

    train_aucs.append(roc_auc_score(y_tr, clf.predict_proba(X_tr)[:,1]))
    cv_auc = cross_val_score(
        DecisionTreeClassifier(random_state=42, ccp_alpha=alpha),
        X_tr, y_tr, cv=5, scoring='roc_auc'
    ).mean()
    cv_aucs.append(cv_auc)
    test_aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:,1]))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# AUC vs alpha
ax = axes[0]
ax.semilogx(ccp_alphas[:-1], train_aucs, 'o-', color='steelblue',
            lw=2, label='Train AUC', markersize=4)
ax.semilogx(ccp_alphas[:-1], cv_aucs,    's-', color='green',
            lw=2, label='CV AUC', markersize=4)
ax.semilogx(ccp_alphas[:-1], test_aucs,  '^-', color='tomato',
            lw=2, label='Test AUC', markersize=4)
best_alpha = ccp_alphas[np.argmax(cv_aucs)]
ax.axvline(best_alpha, color='black', lw=1.5, linestyle='--',
           label=f'Best α = {best_alpha:.5f}')
ax.set_xlabel('ccp_alpha (pruning strength)')
ax.set_ylabel('AUC')
ax.set_title('Cost-Complexity Pruning\n(Larger α → more pruning)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Number of leaves vs alpha
ax2 = axes[1]
ax2.semilogx(ccp_alphas[:-1], n_leaves, 'o-', color='steelblue', lw=2)
ax2.axvline(best_alpha, color='tomato', lw=1.5, linestyle='--',
            label=f'Best α = {best_alpha:.5f}')
ax2.set_xlabel('ccp_alpha'); ax2.set_ylabel('Number of Leaves')
ax2.set_title('Tree Complexity vs Pruning Strength')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# Final pruned tree performance
best_clf = clfs[np.argmax(cv_aucs)]
ax3 = axes[2]
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred_best = best_clf.predict(X_te)
disp = ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred_best),
                               display_labels=cancer.target_names)
disp.plot(ax=ax3, colorbar=False, cmap='Blues')
ax3.set_title(f'Pruned Tree (α={best_alpha:.5f})\n'
              f'Depth={best_clf.get_depth()}, Leaves={best_clf.get_n_leaves()}\n'
              f'Test AUC={max(test_aucs):.4f}')

plt.suptitle('Cost-Complexity Pruning (Post-Pruning)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_ccp_pruning.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"\nBest pruned tree:")
print(f"  ccp_alpha:    {best_alpha:.6f}")
print(f"  Depth:        {best_clf.get_depth()}")
print(f"  Leaves:       {best_clf.get_n_leaves()}")
print(f"  CV AUC:       {max(cv_aucs):.4f}")
print(f"  Test AUC:     {test_aucs[np.argmax(cv_aucs)]:.4f}")
```

---

## 6. Feature Importance

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
feature_names = cancer.feature_names

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_tr, y_tr)

# ── Method 1: Mean Decrease in Impurity (MDI) ─────────────────────────────────
# Computed during training: sum of weighted impurity decreases at splits
# that use this feature (normalized so they sum to 1)
# Bias: favors high-cardinality features (more candidate thresholds)
mdi_importances = clf.feature_importances_

# ── Method 2: Permutation Importance ─────────────────────────────────────────
# Shuffle each feature independently, measure AUC drop
# More reliable than MDI, but slower
perm_result = permutation_importance(clf, X_te, y_te, n_repeats=20,
                                      random_state=42, scoring='roc_auc')
perm_mean = perm_result.importances_mean
perm_std  = perm_result.importances_std

# Sort by MDI
sort_idx = np.argsort(mdi_importances)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# MDI importance
ax = axes[0]
bars = ax.bar(range(len(mdi_importances)),
              mdi_importances[sort_idx],
              color='steelblue', alpha=0.8)
ax.set_xticks(range(len(mdi_importances)))
ax.set_xticklabels([feature_names[i] for i in sort_idx],
                   rotation=90, fontsize=7)
ax.set_title('Feature Importance: Mean Decrease in Impurity (MDI)\n'
             '(Computed during training — fast but biased toward high cardinality)')
ax.set_ylabel('Importance (sum = 1)')
ax.grid(axis='y', alpha=0.3)

# Permutation importance
ax2 = axes[1]
perm_sorted_idx = np.argsort(perm_mean)[::-1]
ax2.bar(range(len(perm_mean)),
        perm_mean[perm_sorted_idx],
        yerr=perm_std[perm_sorted_idx],
        color='tomato', alpha=0.8,
        error_kw={'capsize': 3})
ax2.set_xticks(range(len(perm_mean)))
ax2.set_xticklabels([feature_names[i] for i in perm_sorted_idx],
                    rotation=90, fontsize=7)
ax2.set_title('Feature Importance: Permutation Importance\n'
              '(Evaluated on test set — unbiased, shows true predictive power)')
ax2.set_ylabel('Mean AUC decrease (±1 std)')
ax2.axhline(0, color='black', lw=0.8)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Feature Importance Methods — Breast Cancer',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_feature_importance.png', dpi=140, bbox_inches='tight')
plt.show()

# Comparison table
print("\nTop 10 Features — MDI vs Permutation:")
print(f"\n{'Feature':<35} {'MDI':>8} {'Perm Mean':>12} {'Perm Std':>10}")
print("-" * 70)
for i in sort_idx[:10]:
    print(f"{feature_names[i]:<35} {mdi_importances[i]:>8.4f} "
          f"{perm_mean[i]:>12.4f} {perm_std[i]:>10.4f}")

print("""
\nWhen MDI and Permutation disagree:
  - MDI ranks a feature high but Permutation ranks it low:
    → The feature splits on many thresholds (high cardinality bias)
       but doesn't actually generalize well
  - Permutation ranks a feature high but MDI ranks it low:
    → The feature wasn't used much during training (correlated with another feature)
       but independently holds predictive signal
  
  Rule: Trust Permutation Importance more for feature selection.
        Use MDI for understanding the tree's internal structure.
""")
```

---

## 7. Decision Trees for Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
n   = 200

# Non-linear function: sine wave with trend
X_r = np.sort(rng.uniform(0, 10, n)).reshape(-1, 1)
y_r = np.sin(X_r.ravel()) + 0.1*X_r.ravel() + rng.normal(0, 0.3, n)

X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.25, random_state=42)
X_plot = np.linspace(0, 10, 500).reshape(-1, 1)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

depths_reg = [1, 2, 4, 8, None, 'best']
titles = ['depth=1\n(underfitting)', 'depth=2', 'depth=4\n(good)',
          'depth=8', 'depth=None\n(overfit)', 'Best depth (CV)']

cv_mse = []
for d in [1,2,3,4,5,6,7,8,None]:
    reg = DecisionTreeRegressor(max_depth=d, random_state=42)
    score = cross_val_score(reg, X_tr, y_tr, cv=5,
                            scoring='neg_mean_squared_error').mean()
    cv_mse.append(-score)

best_depth = [1,2,3,4,5,6,7,8,None][np.argmin(cv_mse)]

for ax, (depth, title) in zip(axes.flatten(), zip(depths_reg, titles)):
    d = best_depth if depth == 'best' else depth
    title_use = f'Best depth={best_depth} (CV)' if depth == 'best' else title

    reg = DecisionTreeRegressor(max_depth=d, random_state=42)
    reg.fit(X_tr, y_tr)
    y_plot  = reg.predict(X_plot)
    y_pred  = reg.predict(X_te)
    rmse    = np.sqrt(mean_squared_error(y_te, y_pred))
    r2      = r2_score(y_te, y_pred)

    ax.scatter(X_tr.ravel(), y_tr, s=12, alpha=0.5,
               color='steelblue', label='Train')
    ax.scatter(X_te.ravel(), y_te, s=12, alpha=0.5,
               color='tomato', marker='s', label='Test')
    ax.plot(X_plot.ravel(), y_plot, color='black', lw=2, label='Prediction')
    ax.plot(X_plot.ravel(),
            np.sin(X_plot.ravel()) + 0.1*X_plot.ravel(),
            'g--', lw=1.5, alpha=0.5, label='True')
    ax.set_title(f'{title_use}\nRMSE={rmse:.4f}, R²={r2:.4f}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle('Decision Tree Regressor at Different Depths\n'
             'Step-function predictions = piecewise constant approximation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_regression_trees.png', dpi=140, bbox_inches='tight')
plt.show()

print("Key property of regression trees:")
print("  Predictions are PIECEWISE CONSTANT (step functions)")
print("  Each leaf predicts mean(y) for samples in that region")
print("  This is a fundamental limitation compared to linear models")
print("  → For smooth functions, you need many leaves to approximate well")
print("  → Gradient boosted trees overcome this by stacking many trees")
```

---

## 8. Visualizing and Interpreting Trees

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_tr, y_tr)

# ── Method 1: plot_tree ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(clf,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          impurity=True,
          precision=3,
          fontsize=9,
          ax=ax)
ax.set_title('Decision Tree — Iris Dataset (depth=3)\n'
             'Node color = majority class, Darker = purer',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_tree_visualization.png', dpi=120, bbox_inches='tight')
plt.show()

# ── Method 2: export_text (ASCII representation) ──────────────────────────────
tree_text = export_text(clf,
                        feature_names=list(iris.feature_names),
                        max_depth=4)
print("Tree as Text Rules:")
print(tree_text)

# ── Method 3: Extract rules as human-readable if-then statements ──────────────
def extract_rules(tree, feature_names, class_names):
    """
    Walk the tree and extract each root-to-leaf path as a list of rules.
    Returns a list of (conditions, predicted_class, n_samples, purity) tuples.
    """
    tree_  = tree.tree_
    rules  = []

    def recurse(node, conditions):
        if tree_.children_left[node] == -1:   # Leaf
            class_idx = np.argmax(tree_.value[node][0])
            n_samples = int(tree_.n_node_samples[node])
            purity    = tree_.value[node][0][class_idx] / n_samples
            rules.append((list(conditions),
                          class_names[class_idx],
                          n_samples,
                          purity))
        else:
            feat  = feature_names[tree_.feature[node]]
            thresh = tree_.threshold[node]
            recurse(tree_.children_left[node],
                    conditions + [f"{feat} <= {thresh:.3f}"])
            recurse(tree_.children_right[node],
                    conditions + [f"{feat} >  {thresh:.3f}"])

    recurse(0, [])
    return rules

rules = extract_rules(clf, iris.feature_names, iris.target_names)
print("\nExtracted If-Then Rules:")
print("=" * 60)
for i, (conditions, cls, n, purity) in enumerate(rules, 1):
    print(f"\nRule {i} → {cls} (n={n}, purity={purity:.2%}):")
    for cond in conditions:
        print(f"  IF {cond}")
print("=" * 60)
```

---

## 9. Decision Trees with Scikit-learn

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      cross_val_score, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── Complete tuning workflow ───────────────────────────────────────────────────
cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# Note: Decision trees do NOT need feature scaling.
# Tree splits compare feature_j <= threshold which is scale-invariant.
# We include it here only to demonstrate the Pipeline pattern.

# Grid search: pre-pruning parameters
param_grid = {
    'max_depth':             [3, 5, 7, 10, None],
    'min_samples_split':     [2, 10, 20],
    'min_samples_leaf':      [1, 5, 10],
    'max_features':          [None, 'sqrt', 'log2', 0.5],
    'criterion':             ['gini', 'entropy'],
    'ccp_alpha':             [0.0, 0.001, 0.005, 0.01],
}

gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1, verbose=0
)
gs.fit(X_tr, y_tr)

best = gs.best_estimator_
print("Grid Search Complete:")
print(f"  Best params:  {gs.best_params_}")
print(f"  CV AUC:       {gs.best_score_:.4f}")
print(f"  Test AUC:     {roc_auc_score(y_te, best.predict_proba(X_te)[:,1]):.4f}")
print(f"  Test Acc:     {best.score(X_te, y_te):.4f}")
print(f"  Tree depth:   {best.get_depth()}")
print(f"  Num leaves:   {best.get_n_leaves()}")
print(f"\n{classification_report(y_te, best.predict(X_te), target_names=cancer.target_names)}")

# ── Compare across datasets ───────────────────────────────────────────────────
datasets = [
    ('Iris (3-class)',   load_iris()),
    ('Breast Cancer',   load_breast_cancer()),
    ('Wine (3-class)',  load_wine()),
]

print(f"\n{'Dataset':<25} {'Baseline Acc':>14} {'Tuned Acc':>12} {'Improvement':>13}")
print("-" * 68)

for name, data in datasets:
    X_d, y_d = data.data, data.target
    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
        X_d, y_d, stratify=y_d, test_size=0.2, random_state=42
    )
    # Baseline: default tree
    clf_base = DecisionTreeClassifier(random_state=42)
    clf_base.fit(X_tr_d, y_tr_d)
    base_acc = clf_base.score(X_te_d, y_te_d)

    # Tuned: max_depth chosen by CV
    best_cv   = -np.inf
    best_d    = None
    for d in [2, 3, 4, 5, 6, 8, 10, None]:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        cv  = cross_val_score(clf, X_tr_d, y_tr_d, cv=5, scoring='accuracy').mean()
        if cv > best_cv:
            best_cv = cv; best_d = d
    clf_tuned = DecisionTreeClassifier(max_depth=best_d, random_state=42)
    clf_tuned.fit(X_tr_d, y_tr_d)
    tuned_acc = clf_tuned.score(X_te_d, y_te_d)

    print(f"{name:<25} {base_acc:>14.4f} {tuned_acc:>12.4f} "
          f"{tuned_acc-base_acc:>+13.4f}  (best depth={best_d})")
```

---

## 10. Strengths and Weaknesses

```python
print("""
DECISION TREES — STRENGTHS AND WEAKNESSES
══════════════════════════════════════════════════════════════════════

STRENGTHS:
  ✓ Interpretability
      Every prediction can be traced to an exact set of rules.
      "If petal length > 2.45cm AND petal width <= 1.75cm, then versicolor"
      Ideal for regulated industries (finance, medicine, law).

  ✓ No feature scaling required
      Splits are comparisons (x_j <= t) — scale-invariant.
      Works natively with mixed data types.

  ✓ Handles non-linear relationships
      Unlike linear models, can capture step-like non-linearities.
      No feature engineering required for simple non-linearity.

  ✓ Implicit feature selection
      Irrelevant features are simply never split on.
      Works with high-dimensional data.

  ✓ Fast inference
      O(depth) per prediction — just traverse the tree.

  ✓ Handles mixed feature types
      Continuous, ordinal, and even binary features natively.

WEAKNESSES:
  ✗ High variance (unstable)
      Small changes in training data → completely different tree structure.
      This is the primary reason we use ensembles (Random Forests, boosting).

  ✗ Axis-aligned splits only
      A diagonal boundary requires many splits to approximate.
      Example: y = 1 if x₁ + x₂ > 1 → tree needs many steps.

  ✗ Piecewise constant predictions (regression)
      Cannot extrapolate beyond training data range.
      Cannot fit smooth curves without many leaves.

  ✗ Poor on high-dimensional sparse data
      Text classification, genomics → linear models usually better.

  ✗ Prone to overfitting without regularization
      An unpruned tree perfectly memorizes training data.
      Requires careful tuning of stopping/pruning parameters.

  ✗ Biased toward features with more categories
      MDI feature importance is biased — high cardinality features
      have more split candidates and appear important spuriously.

BOTTOM LINE:
  Use decision trees when:
    • Interpretability is paramount
    • A simple, inspectable model is needed as a baseline
    • You're on your way to building a Random Forest or GBM

  Don't use decision trees when:
    • You need the best possible predictive accuracy (use RF or GBM)
    • Data is high-dimensional and sparse (use logistic regression or SVMs)
    • You need smooth probability estimates (use logistic regression or calibrated RF)
══════════════════════════════════════════════════════════════════════
""")
```

---

## 11. Case Study: Classifying Wine Quality

```python
# =============================================================================
# Full Pipeline: Wine Quality → EDA → Tree → Pruning → Rules → Report
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      cross_val_score, StratifiedKFold,
                                      learning_curve)
from sklearn.tree import (DecisionTreeClassifier, plot_tree,
                           export_text)
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load and inspect ───────────────────────────────────────────────────────
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
class_names   = wine.target_names

print(f"Dataset: {X.shape[0]} wines, {X.shape[1]} chemical features")
print(f"Classes: {class_names} (cultivar types)")
print(f"Class distribution: {np.bincount(y)}")
print(f"\nFeature ranges:")
for fn, mn, mx in zip(feature_names, X.min(0), X.max(0)):
    print(f"  {fn:<30}: [{mn:.2f}, {mx:.2f}]")

# ── 2. Split ──────────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# ── 3. Tuning: find best depth via CV + cost-complexity pruning ───────────────
# Step A: depth grid search
cv     = StratifiedKFold(5, shuffle=True, random_state=42)
depths = [1, 2, 3, 4, 5, 6, None]
cv_acc = []
for d in depths:
    scores = cross_val_score(
        DecisionTreeClassifier(max_depth=d, random_state=42),
        X_tr, y_tr, cv=cv, scoring='accuracy'
    )
    cv_acc.append(scores.mean())

best_depth = depths[np.argmax(cv_acc)]
print(f"\nBest depth by CV: {best_depth}")

# Step B: post-pruning on top of best depth
clf_temp = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
clf_temp.fit(X_tr, y_tr)
path     = clf_temp.cost_complexity_pruning_path(X_tr, y_tr)
alphas   = path.ccp_alphas[:-1]

cv_acc_pruned = []
for alpha in alphas:
    scores = cross_val_score(
        DecisionTreeClassifier(max_depth=best_depth, ccp_alpha=alpha, random_state=42),
        X_tr, y_tr, cv=cv, scoring='accuracy'
    )
    cv_acc_pruned.append(scores.mean())

best_alpha = alphas[np.argmax(cv_acc_pruned)] if len(alphas) > 0 else 0.0
print(f"Best ccp_alpha by CV: {best_alpha:.6f}")

# ── 4. Final model ────────────────────────────────────────────────────────────
clf_final = DecisionTreeClassifier(
    max_depth=best_depth, ccp_alpha=best_alpha, random_state=42
)
clf_final.fit(X_tr, y_tr)
y_pred = clf_final.predict(X_te)
test_acc = clf_final.score(X_te, y_te)

print(f"\nFinal tree: depth={clf_final.get_depth()}, leaves={clf_final.get_n_leaves()}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"\n{classification_report(y_te, y_pred, target_names=class_names)}")

# ── 5. Comprehensive visualization ────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# Tree visualization
ax_tree = fig.add_subplot(gs[0, :])
plot_tree(clf_final,
          feature_names=feature_names,
          class_names=class_names,
          filled=True, rounded=True,
          impurity=True, precision=3,
          fontsize=8, ax=ax_tree)
ax_tree.set_title(f'Final Decision Tree — Wine Dataset\n'
                  f'Depth={clf_final.get_depth()}, Leaves={clf_final.get_n_leaves()}, '
                  f'Test Acc={test_acc:.4f}',
                  fontsize=12, fontweight='bold')

# Confusion matrix
ax_cm = fig.add_subplot(gs[1, 0])
disp  = ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                                display_labels=class_names)
disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
ax_cm.set_title('Confusion Matrix')

# Feature importance
ax_fi = fig.add_subplot(gs[1, 1])
fi   = clf_final.feature_importances_
fi_sort = np.argsort(fi)[::-1]
# Show only top features (non-zero)
nonzero = fi > 0
bars = ax_fi.barh([feature_names[i] for i in fi_sort if fi[i] > 0][::-1],
                   sorted(fi[nonzero]),
                   color='steelblue', alpha=0.8)
ax_fi.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
ax_fi.set_xlabel('MDI Importance')
ax_fi.set_title('Feature Importance (MDI)')
ax_fi.grid(axis='x', alpha=0.3)

# Learning curve
ax_lc = fig.add_subplot(gs[1, 2])
train_sizes, tr_sc, val_sc = learning_curve(
    clf_final, X_tr, y_tr,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5, scoring='accuracy', n_jobs=-1
)
ax_lc.plot(train_sizes, tr_sc.mean(1), 'o-', color='steelblue', lw=2, label='Train')
ax_lc.fill_between(train_sizes, tr_sc.mean(1)-tr_sc.std(1),
                    tr_sc.mean(1)+tr_sc.std(1), alpha=0.2, color='steelblue')
ax_lc.plot(train_sizes, val_sc.mean(1), 's-', color='tomato', lw=2, label='CV')
ax_lc.fill_between(train_sizes, val_sc.mean(1)-val_sc.std(1),
                    val_sc.mean(1)+val_sc.std(1), alpha=0.2, color='tomato')
ax_lc.set_xlabel('Training set size'); ax_lc.set_ylabel('Accuracy')
ax_lc.set_title('Learning Curve')
ax_lc.legend(fontsize=9); ax_lc.grid(True, alpha=0.3)

plt.savefig('figures/07_wine_case_study.png', dpi=120, bbox_inches='tight')
plt.show()

# ── 6. Extract and display rules ──────────────────────────────────────────────
print("\nDecision Rules:")
print(export_text(clf_final, feature_names=list(feature_names)))
```

---

## 12. Summary

### ✅ Must-Remember Mental Models

**1. The CART algorithm — one sentence:**
At each node, find the feature $j$ and threshold $t$ that maximally reduces weighted impurity in the children. Repeat recursively. Stop when a criterion is met.

**2. Gini vs. Entropy:**
- Gini: $1 - \sum p_k^2$ — faster to compute, slightly favors larger partitions.
- Entropy: $-\sum p_k \log p_k$ — information-theoretic, penalizes impurity more at extremes.
- In practice: rarely matters which you choose. Try both in grid search.

**3. Gini impurity formula for binary case:**
$$G = 2p(1-p) \qquad \text{(ranges from 0 to 0.5)}$$

**4. Trees overfit aggressively.** An unpruned tree always achieves 100% train accuracy. Always tune `max_depth` via cross-validation. Use `ccp_alpha` for post-pruning.

**5. Feature importance — which to use:**
- MDI (`.feature_importances_`): fast, available after training, biased toward high-cardinality features.
- Permutation Importance: evaluated on held-out data, unbiased, slower.
- Default: use permutation importance for feature selection decisions.

**6. No feature scaling required.** Tree splits are comparisons ($x_j \leq t$), which are scale-invariant. StandardScaler before a tree is harmless but unnecessary.

**7. Regression trees predict the mean of the leaf's training samples** — always a piecewise constant function. Cannot extrapolate beyond the training range.

**8. Trees are the foundation of ensembles:**
- Random Forest (Chapter 8): many trees, each on a random data/feature subset → reduces variance
- Gradient Boosting (Chapter 8): trees trained sequentially on residuals → reduces bias
- Understanding a single tree well is essential for understanding these.

**9. Extract rules for interpretability.** `export_text()` and `plot_tree()` give human-readable rules. This is decision trees' greatest advantage over every other model.

---

## 13. Exercises

**Conceptual:**

1. A node has 40 samples: 30 class A, 10 class B. Calculate the Gini impurity, entropy (base-2), and misclassification error. Verify that Gini impurity ≤ 2 × misclassification error.

2. Consider a dataset with a diagonal decision boundary ($y=1$ if $x_1 + x_2 > 0$). How many splits does a depth-$d$ tree need to approximate this boundary to an error of $\epsilon$? Draw it. What does this tell you about tree limitations?

3. Explain why CART with Gini impurity applied to a balanced binary problem will never choose a split that puts all samples in one child (prove this algebraically).

**Coding:**

4. **Extend the scratch implementation**: Add support for (a) entropy as the splitting criterion, (b) soft probability predictions (`predict_proba` — return class proportions in the leaf), (c) regression (using variance reduction).

5. **Cost-complexity pruning curve**: Using the breast cancer dataset, plot the full pruning path and show: (a) number of leaves vs $\alpha$, (b) train/test/CV accuracy vs $\alpha$. Identify the $\alpha$ that maximizes CV accuracy and evaluate the pruned tree.

6. **Stability experiment**: Generate 50 bootstrap samples from the Iris dataset. For each, train a depth-5 decision tree. Visualize the variation in: (a) tree structure (feature at root node), (b) feature importances across the 50 trees. This illustrates the high variance of single trees.

7. **Rule extraction system**: Build a function `tree_to_rules(clf, feature_names, class_names, min_purity=0.9)` that extracts all leaf rules with purity ≥ `min_purity`. Format them as human-readable SQL-like WHERE clauses. Apply to breast cancer dataset.

**Challenge:**

8. **ID3 from scratch**: Implement the ID3 algorithm (uses Information Gain with Entropy, not Gini, and originally designed for categorical features). Support:
   - Entropy splitting criterion
   - Categorical features (split on equality rather than threshold)
   - Multi-way splits (one branch per category value)
   - Compare ID3 vs CART on the adult income dataset (mix of categorical and continuous features)

---

## 14. References

- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Chapman & Hall. — The original CART paper.
- Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*, 1(1), 81–106. — The ID3 algorithm.
- Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann. — The C4.5 extension of ID3.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Chapter 9: Additive Models, Trees, and Related Methods.
- Scikit-learn Decision Tree documentation: https://scikit-learn.org/stable/modules/tree.html

---

*Previous: [Chapter 6 — Logistic Regression & Classification](06_logistic_regression.md)*  
*Next: [Chapter 8 — Ensemble Methods: Random Forests, Bagging & Boosting](08_ensembles.md)*

*In Chapter 8, we see how combining many weak trees creates models that dominate structured data competitions. We derive the variance reduction of bagging, implement Random Forests from scratch, and build gradient boosting step by step.*

---

> **Chapter 7 complete.** All code is in `notebooks/07_decision_trees.ipynb`.  
> Run `my_tree.print_tree()` on your own dataset to see the learned rules in plain English.
