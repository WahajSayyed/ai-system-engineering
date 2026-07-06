# Chapter 8: Ensemble Methods — Random Forests, Bagging & Boosting

> *"If one doctor's opinion is uncertain, ask ten doctors and take the majority. Ensembles work for the same reason."*

> *"Random Forests and Gradient Boosting are the two most important algorithms for structured data in production today. Understanding them from first principles — not just their API — is what separates practitioners from button-pushers."*

---

## Table of Contents

1. [Why Combine Models?](#1-why-combine-models)
2. [The Bias-Variance Decomposition Revisited](#2-the-bias-variance-decomposition-revisited)
3. [Bagging — Bootstrap Aggregating](#3-bagging--bootstrap-aggregating)
   - 3.1 [The Bootstrap](#31-the-bootstrap)
   - 3.2 [Bagging from Scratch](#32-bagging-from-scratch)
   - 3.3 [Out-of-Bag Evaluation](#33-out-of-bag-evaluation)
4. [Random Forests](#4-random-forests)
   - 4.1 [The Key Insight: Decorrelating Trees](#41-the-key-insight-decorrelating-trees)
   - 4.2 [Random Forest from Scratch](#42-random-forest-from-scratch)
   - 4.3 [Hyperparameter Guide](#43-hyperparameter-guide)
   - 4.4 [Feature Importance in Random Forests](#44-feature-importance-in-random-forests)
5. [Boosting — Learning from Mistakes](#5-boosting--learning-from-mistakes)
   - 5.1 [AdaBoost: Reweighting Hard Examples](#51-adaboost-reweighting-hard-examples)
   - 5.2 [Gradient Boosting: The General Framework](#52-gradient-boosting-the-general-framework)
   - 5.3 [Gradient Boosting from Scratch](#53-gradient-boosting-from-scratch)
6. [XGBoost — Extreme Gradient Boosting](#6-xgboost--extreme-gradient-boosting)
7. [LightGBM — Fast Gradient Boosting](#7-lightgbm--fast-gradient-boosting)
8. [Stacking — Ensembling Ensembles](#8-stacking--ensembling-ensembles)
9. [Choosing the Right Ensemble](#9-choosing-the-right-ensemble)
10. [Case Study: Tabular Data Showdown](#10-case-study-tabular-data-showdown)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Why Combine Models?

A single decision tree has high variance: retrain on a slightly different dataset and you get a completely different tree. A single linear model has high bias: it can't capture non-linear patterns. What if we could combine many models to get the best of both worlds?

**The core principle of ensemble learning:**

> Combine a set of diverse, independently-trained models. Their errors tend to cancel out; their strengths reinforce.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(42)

# ── Demonstrate cancellation of errors with majority voting ──────────────────
# Each model is a weak classifier with ~70% accuracy
# Three independent models majority-vote → much higher accuracy

def simulate_ensemble_accuracy(n_models, p_each=0.70, n_trials=100_000):
    """
    Probability that majority of n_models independent classifiers
    (each with accuracy p_each) is correct.
    For odd n: majority = ceil(n/2) or more correct.
    """
    correct_threshold = n_models // 2 + 1
    from scipy.stats import binom
    # P(k or more correct) = 1 - CDF(k-1)
    return 1 - binom.cdf(correct_threshold - 1, n_models, p_each)

n_models_range = np.arange(1, 52, 2)   # Odd numbers only for clean majority vote
p_individuals  = [0.55, 0.65, 0.70, 0.80]
colors_ind     = ['#EF9A9A', '#FFCC80', '#A5D6A7', '#90CAF9']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for p, color in zip(p_individuals, colors_ind):
    probs = [simulate_ensemble_accuracy(n, p) for n in n_models_range]
    ax.plot(n_models_range, probs, 'o-', color=color, lw=2,
            label=f'p_individual = {p:.2f}')
ax.axhline(0.5, color='gray', lw=1, linestyle='--', label='Random (50%)')
ax.set_xlabel('Number of Models in Ensemble')
ax.set_ylabel('Ensemble Accuracy')
ax.set_title('Ensemble Accuracy via Majority Vote\n'
             '(Models assumed independent — theoretical)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.4, 1.05)

# ── Show on real data: individual trees vs. ensemble ─────────────────────────
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

n_ensemble_sizes = [1, 3, 5, 10, 25, 50, 100]
ensemble_accs    = []

for n in n_ensemble_sizes:
    predictions = []
    for seed in range(n):
        # Each tree sees a random bootstrap sample
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        clf = DecisionTreeClassifier(max_depth=None, random_state=seed)
        clf.fit(X_train[idx], y_train[idx])
        predictions.append(clf.predict(X_test))
    # Majority vote
    vote = np.array(predictions).mean(axis=0) >= 0.5
    ensemble_accs.append(accuracy_score(y_test, vote.astype(int)))

ax2 = axes[1]
ax2.plot(n_ensemble_sizes, ensemble_accs, 'o-', color='steelblue', lw=2.5,
         markersize=8, label='Ensemble (bagging)')
ax2.axhline(ensemble_accs[0], color='tomato', lw=2, linestyle='--',
            label=f'Single tree: {ensemble_accs[0]:.3f}')
ax2.set_xlabel('Number of Trees')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Real Data: Accuracy vs Ensemble Size\n'
              '(Moon dataset, deep trees, bootstrap sampling)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.8, 1.0)

plt.suptitle('Why Ensembles Work: Errors Cancel, Strengths Reinforce',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_ensemble_motivation.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"Single tree accuracy:   {ensemble_accs[0]:.4f}")
print(f"100-tree ensemble:      {ensemble_accs[-1]:.4f}")
print(f"Improvement:            {ensemble_accs[-1] - ensemble_accs[0]:+.4f}")
```

---

## 2. The Bias-Variance Decomposition Revisited

The expected test error of any model decomposes as:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

- **Bagging** targets **variance**: average many high-variance, low-bias models → variance decreases, bias unchanged.
- **Boosting** targets **bias**: sequentially combine many low-variance, high-bias models → bias decreases, variance may increase slightly.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

rng = np.random.default_rng(42)

# ── Empirical bias-variance decomposition ──────────────────────────────────────
# True function: y = sin(2πx) + N(0, 0.3)
# We train 200 models on 200 different training sets of size 50
# At each x, compute:
#   Bias²  = (mean_prediction - true_value)²
#   Variance = var(predictions)

def true_function(x):
    return np.sin(2 * np.pi * x)

n_datasets = 200
n_train    = 50
x_test     = np.linspace(0, 1, 100)
y_true     = true_function(x_test)
noise_std  = 0.3

# Models to compare
models_bv = {
    'Shallow Tree\n(depth=2, high bias)':  DecisionTreeRegressor(max_depth=2),
    'Deep Tree\n(depth=None, high var)':   DecisionTreeRegressor(max_depth=None),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, (model_name, model_proto) in enumerate(models_bv.items()):
    predictions = []
    for seed in range(n_datasets):
        rng_local = np.random.default_rng(seed)
        x_tr = rng_local.uniform(0, 1, n_train)
        y_tr = true_function(x_tr) + rng_local.normal(0, noise_std, n_train)
        from sklearn.base import clone
        m = clone(model_proto)
        m.fit(x_tr.reshape(-1,1), y_tr)
        predictions.append(m.predict(x_test.reshape(-1,1)))

    predictions = np.array(predictions)   # (n_datasets, n_test)
    mean_pred   = predictions.mean(axis=0)
    variance    = predictions.var(axis=0)
    bias_sq     = (mean_pred - y_true)**2
    total_err   = bias_sq + variance

    # Plot predictions and statistics
    ax_top = axes[0, col]
    for i in range(0, min(50, n_datasets)):
        ax_top.plot(x_test, predictions[i], color='steelblue', alpha=0.05, lw=1)
    ax_top.plot(x_test, y_true,   'k-',  lw=2.5, label='True function', zorder=5)
    ax_top.plot(x_test, mean_pred,'r-',  lw=2.5, label='Mean prediction', zorder=4)
    ax_top.set_title(f'{model_name}\n(showing 50/200 trained models)')
    ax_top.set_ylim(-2.5, 2.5)
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    ax_bot = axes[1, col]
    ax_bot.plot(x_test, bias_sq,   color='tomato',    lw=2, label=f'Bias²  (mean={bias_sq.mean():.4f})')
    ax_bot.plot(x_test, variance,  color='steelblue', lw=2, label=f'Var    (mean={variance.mean():.4f})')
    ax_bot.plot(x_test, total_err, color='black',     lw=2, linestyle='--',
                label=f'Total  (mean={total_err.mean():.4f})')
    ax_bot.fill_between(x_test, 0, bias_sq,  alpha=0.25, color='tomato')
    ax_bot.fill_between(x_test, 0, variance, alpha=0.25, color='steelblue')
    ax_bot.set_xlabel('x'); ax_bot.set_ylabel('Error component')
    ax_bot.set_title('Bias²  vs  Variance  vs  Total Error')
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, alpha=0.3)

plt.suptitle('Bias-Variance Decomposition: Shallow vs Deep Tree\n'
             'Shallow = high bias (underfits).  Deep = high variance (overfits).',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_bias_variance.png', dpi=130, bbox_inches='tight')
plt.show()

print("Summary — Bias-Variance Decomposition:")
print("  Shallow tree:  high Bias², low Variance  → underfitting")
print("  Deep tree:     low Bias², high Variance  → overfitting")
print("\nBagging strategy: average many deep trees")
print("  Deep tree Bias² unchanged (still low)")
print("  Deep tree Variance divided by n_trees (approximately)")
print("  → Best of both worlds")
```

---

## 3. Bagging — Bootstrap Aggregating

### 3.1 The Bootstrap

The **bootstrap** generates $B$ new datasets, each by sampling $n$ observations from the original dataset **with replacement**.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ── What the bootstrap does ───────────────────────────────────────────────────
original = np.arange(1, 11)   # 10 samples, labeled 1-10
n = len(original)
B = 5

print("Bootstrap sampling demonstration:")
print(f"Original dataset: {original}")
print(f"\n{'Bootstrap':>12} {'Sample':<35} {'OOB (not sampled)'}")
print("-" * 70)

for b in range(B):
    idx    = rng.choice(n, n, replace=True)
    sample = original[idx]
    oob    = original[~np.isin(original, sample)]
    print(f"Bootstrap {b+1:>2}:  {sorted(sample.tolist())}  {sorted(oob.tolist())}")

# Expected fraction of samples included in a bootstrap:
# P(sample i is included) = 1 - (1 - 1/n)^n → 1 - 1/e ≈ 0.632 as n→∞
p_included = 1 - (1 - 1/n)**n
print(f"\nP(sample included) = 1 - (1-1/n)^n = {p_included:.4f}")
print(f"Expected: 1 - 1/e = {1 - 1/np.e:.4f}")
print(f"Expected OOB fraction: {1 - p_included:.4f} ≈ 36.8%")

# Verify empirically
n_large = 10_000
B_large = 1_000
included_fracs = []
for b in range(B_large):
    idx = rng.choice(n_large, n_large, replace=True)
    included_fracs.append(len(np.unique(idx)) / n_large)
print(f"\nEmpirical avg fraction included (n={n_large}, B={B_large}): "
      f"{np.mean(included_fracs):.4f}")
```

---

### 3.2 Bagging from Scratch

```python
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

class BaggingClassifierScratch:
    """
    Bootstrap Aggregating (Bagging) Classifier.
    Trains B estimators on bootstrap samples; predicts by majority vote.
    """

    def __init__(self, base_estimator, n_estimators=100,
                 max_samples=1.0, random_state=42):
        """
        Parameters
        ----------
        base_estimator : sklearn estimator — the weak learner to bag
        n_estimators   : number of bootstrap models
        max_samples    : fraction of samples in each bootstrap (default: 100%)
        """
        self.base_estimator  = base_estimator
        self.n_estimators    = n_estimators
        self.max_samples     = max_samples
        self.random_state    = random_state
        self.estimators_     = []
        self.oob_indices_    = []   # Out-of-bag indices for each estimator

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n   = len(y)
        k   = int(n * self.max_samples)

        self.estimators_ = []
        self.oob_indices_ = []

        for i in range(self.n_estimators):
            # Bootstrap sample
            boot_idx  = rng.choice(n, k, replace=True)
            oob_idx   = np.array([j for j in range(n) if j not in set(boot_idx)])

            X_boot, y_boot = X[boot_idx], y[boot_idx]

            # Train a clone of the base estimator
            est = clone(self.base_estimator)
            est.set_params(random_state=i)
            est.fit(X_boot, y_boot)

            self.estimators_.append(est)
            self.oob_indices_.append(oob_idx)

        return self

    def predict_proba(self, X):
        """Average class probabilities across all estimators."""
        all_probas = np.array([est.predict_proba(X) for est in self.estimators_])
        return all_probas.mean(axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def oob_score(self, X, y):
        """
        OOB score: for each sample, predict using only estimators
        that did NOT see it during training.
        """
        n = len(y)
        n_classes = len(np.unique(y))
        oob_proba = np.zeros((n, n_classes))
        oob_count = np.zeros(n)

        for est, oob_idx in zip(self.estimators_, self.oob_indices_):
            if len(oob_idx) == 0:
                continue
            proba = est.predict_proba(X[oob_idx])
            oob_proba[oob_idx] += proba
            oob_count[oob_idx] += 1

        # Samples that appeared in at least one OOB set
        valid_mask = oob_count > 0
        oob_pred   = np.argmax(oob_proba[valid_mask], axis=1)
        return accuracy_score(y[valid_mask], oob_pred)


# ── Test and compare ──────────────────────────────────────────────────────────
cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# Our implementation
bag_scratch = BaggingClassifierScratch(
    base_estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100, random_state=42
)
bag_scratch.fit(X_tr, y_tr)

# Sklearn reference
from sklearn.ensemble import BaggingClassifier
bag_sklearn = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100, random_state=42, oob_score=True
)
bag_sklearn.fit(X_tr, y_tr)

print("Bagging — Scratch vs Sklearn:")
print(f"\n  Scratch  — Test Acc: {bag_scratch.score(X_te, y_te):.4f}, "
      f"OOB Acc: {bag_scratch.oob_score(X_tr, y_tr):.4f}")
print(f"  Sklearn  — Test Acc: {bag_sklearn.score(X_te, y_te):.4f}, "
      f"OOB Acc: {bag_sklearn.oob_score_:.4f}")

single_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
single_tree.fit(X_tr, y_tr)
print(f"\n  Single Tree — Test Acc: {single_tree.score(X_te, y_te):.4f}")
print(f"  Bagging improvement: "
      f"{bag_sklearn.score(X_te, y_te) - single_tree.score(X_te, y_te):+.4f}")
```

---

### 3.3 Out-of-Bag Evaluation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# Track OOB score as n_estimators grows
n_range   = [1, 5, 10, 20, 50, 100, 200]
oob_scores_bag = []
test_scores_bag = []

for n in n_range:
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=None),
        n_estimators=n, oob_score=True, random_state=42
    )
    bag.fit(X_tr, y_tr)
    oob_scores_bag.append(bag.oob_score_)
    test_scores_bag.append(bag.score(X_te, y_te))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogx(n_range, oob_scores_bag,  'o-', color='green',
            lw=2.5, label='OOB score (no test set needed!)')
ax.semilogx(n_range, test_scores_bag, 's-', color='tomato',
            lw=2.5, label='Test score')
ax.set_xlabel('Number of estimators')
ax.set_ylabel('Accuracy')
ax.set_title('OOB Score Tracks Test Score\n'
             'OOB = free cross-validation, no data held out!')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
cv_scores = cross_val_score(
    BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=None),
                      n_estimators=100, random_state=42),
    X_tr, y_tr, cv=5
)
methods = ['OOB Score', '5-fold CV', 'Test Set']
values  = [oob_scores_bag[-1],
           cv_scores.mean(),
           test_scores_bag[-1]]
errors  = [0, cv_scores.std(), 0]

bars = ax2.bar(methods, values, color=['green','steelblue','tomato'], alpha=0.8)
ax2.errorbar([0,1,2], values, yerr=[0, cv_scores.std()*2, 0],
             fmt='none', color='black', capsize=8, lw=2)
ax2.bar_label(bars, fmt='%.4f', padding=5)
ax2.set_ylabel('Accuracy')
ax2.set_title('OOB vs Cross-Validation vs Test Set\n'
              'OOB is nearly as reliable as 5-fold CV')
ax2.set_ylim(0.9, 1.01)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/08_oob_score.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"OOB Score:    {oob_scores_bag[-1]:.4f}")
print(f"5-fold CV:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Test Score:   {test_scores_bag[-1]:.4f}")
print("\nKey insight: OOB score is free — uses the ~36.8% samples")
print("not seen by each tree as a built-in validation set.")
print("No need to hold out a separate validation set for hyperparameter tuning!")
```

---

## 4. Random Forests

### 4.1 The Key Insight: Decorrelating Trees

Bagging reduces variance by averaging. But if all trees are highly correlated (because they all tend to use the same strong features at the root), averaging doesn't help as much.

**The variance of the average of $B$ correlated variables:**

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B X_b\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Where $\rho$ is the pairwise correlation between trees and $\sigma^2$ is their individual variance.

As $B \to \infty$: variance $\to \rho \sigma^2$ (not zero!). Correlation is the bottleneck.

**Random Forest's solution**: at each split, only consider a random subset of $m$ features (not all $d$). This **decorrelates** the trees, making the ensemble more powerful.

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Illustrate the variance formula ──────────────────────────────────────────
sigma_sq = 1.0   # Individual tree variance
B_range  = np.arange(1, 201)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for rho, color, label in [(0.9, '#EF9A9A', 'ρ=0.9 (highly correlated)'),
                            (0.5, '#FFCC80', 'ρ=0.5 (moderate)'),
                            (0.2, '#A5D6A7', 'ρ=0.2 (weakly correlated)'),
                            (0.0, '#90CAF9', 'ρ=0.0 (independent)')]:
    var_ensemble = rho * sigma_sq + (1 - rho) / B_range * sigma_sq
    ax.plot(B_range, var_ensemble, color=color, lw=2.5, label=label)

ax.axhline(sigma_sq, color='black', lw=1.5, linestyle=':', label='Single tree variance')
ax.set_xlabel('Number of Trees B')
ax.set_ylabel('Ensemble Variance')
ax.set_title('Variance of Ensemble Average\nVar = ρσ² + (1-ρ)σ²/B\n'
             'Correlation ρ limits how much bagging helps')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
# Floor variance (as B→∞) vs correlation
rho_range = np.linspace(0, 1, 200)
floor_var = rho_range * sigma_sq
ax2.fill_between(rho_range, 0, floor_var, alpha=0.3, color='tomato',
                 label='Floor variance (irreducible)')
ax2.fill_between(rho_range, floor_var, sigma_sq, alpha=0.3, color='steelblue',
                 label='Reducible by adding more trees')
ax2.plot(rho_range, floor_var, color='tomato', lw=2.5)
ax2.axvline(0.9, color='gray', lw=1.5, linestyle='--', label='Bagging (ρ≈0.9)')
ax2.axvline(0.3, color='green', lw=1.5, linestyle='--', label='Random Forest (ρ≈0.3)')
ax2.set_xlabel('Pairwise Tree Correlation ρ')
ax2.set_ylabel('Variance Floor (B→∞)')
ax2.set_title('Random Forest Insight:\nReducing Correlation Reduces Variance Floor')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.suptitle('Why Random Forests Beat Pure Bagging: Decorrelation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_rf_variance.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
Random Forest vs Bagging:
────────────────────────────────────────────────────────────
Bagging:       Bootstrap sample + full feature set
               → Trees correlated because same strong feature
                 appears at root in most trees

Random Forest: Bootstrap sample + RANDOM m features at each split
               → Trees decorrelated: different features drive
                 different trees → variance floor drops dramatically

Typical m values:
  Classification: m = √d     (e.g., d=30 → m=5)
  Regression:     m = d/3    (e.g., d=30 → m=10)
  These are starting points — tune m as a hyperparameter.
────────────────────────────────────────────────────────────
""")
```

---

### 4.2 Random Forest from Scratch

```python
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class RandomForestScratch:
    """
    Random Forest Classifier — built from scratch.
    Key additions over BaggingClassifier:
      1. max_features: only consider m features at each split
      2. OOB score built-in
      3. Feature importance via MDI
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 max_features='sqrt', min_samples_leaf=1,
                 random_state=42):
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.max_features    = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state    = random_state

    def _get_max_features(self, d):
        if self.max_features == 'sqrt':    return max(1, int(np.sqrt(d)))
        if self.max_features == 'log2':    return max(1, int(np.log2(d)))
        if isinstance(self.max_features, float): return max(1, int(self.max_features * d))
        if isinstance(self.max_features, int):   return self.max_features
        return d   # None → use all features (equivalent to bagging)

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        m    = self._get_max_features(d)
        self.n_classes_ = len(np.unique(y))

        self.estimators_   = []
        self.oob_indices_  = []
        self.n_features_   = d

        for i in range(self.n_estimators):
            # 1. Bootstrap sample
            boot_idx = rng.choice(n, n, replace=True)
            oob_idx  = np.setdiff1d(np.arange(n), np.unique(boot_idx))

            # 2. Random feature subset — key RF innovation
            # We pass max_features to the tree, which then picks m features
            # randomly at each node (not once globally per tree)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=m,           # <-- The critical parameter
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(0, 2**31))
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.estimators_.append(tree)
            self.oob_indices_.append(oob_idx)

        # Feature importances: average MDI across trees
        self.feature_importances_ = np.mean(
            [t.feature_importances_ for t in self.estimators_], axis=0
        )
        return self

    def predict_proba(self, X):
        all_probas = np.array([t.predict_proba(X) for t in self.estimators_])
        return all_probas.mean(axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def oob_score(self, X, y):
        n = len(y)
        oob_proba = np.zeros((n, self.n_classes_))
        oob_count = np.zeros(n)

        for tree, oob_idx in zip(self.estimators_, self.oob_indices_):
            if len(oob_idx) == 0: continue
            proba = tree.predict_proba(X[oob_idx])
            oob_proba[oob_idx] += proba
            oob_count[oob_idx] += 1

        valid = oob_count > 0
        return accuracy_score(y[valid], np.argmax(oob_proba[valid], axis=1))


# ── Benchmark ────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

rf_scratch  = RandomForestScratch(n_estimators=100, max_depth=None,
                                   max_features='sqrt', random_state=42)
rf_scratch.fit(X_tr, y_tr)

rf_sklearn  = RandomForestClassifier(n_estimators=100, max_depth=None,
                                      max_features='sqrt', random_state=42,
                                      oob_score=True)
rf_sklearn.fit(X_tr, y_tr)

print("Random Forest — Scratch vs Sklearn:")
print(f"\n  Scratch  — Test Acc: {rf_scratch.score(X_te, y_te):.4f}, "
      f"OOB Acc: {rf_scratch.oob_score(X_tr, y_tr):.4f}")
print(f"  Sklearn  — Test Acc: {rf_sklearn.score(X_te, y_te):.4f}, "
      f"OOB Acc: {rf_sklearn.oob_score_:.4f}")
print(f"\nTest AUC (Sklearn): "
      f"{roc_auc_score(y_te, rf_sklearn.predict_proba(X_te)[:,1]):.4f}")
```

---

### 4.3 Hyperparameter Guide

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      validation_curve)
import warnings
warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# ── 1. n_estimators: accuracy plateaus ────────────────────────────────────────
n_range = [1, 5, 10, 20, 50, 100, 200, 300, 500]
tr_acc, te_acc = [], []
for n in n_range:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    tr_acc.append(rf.score(X_tr, y_tr))
    te_acc.append(rf.score(X_te, y_te))

axes[0,0].semilogx(n_range, tr_acc, 'o-', color='steelblue', label='Train', lw=2)
axes[0,0].semilogx(n_range, te_acc, 's-', color='tomato',    label='Test',  lw=2)
axes[0,0].set_xlabel('n_estimators'); axes[0,0].set_ylabel('Accuracy')
axes[0,0].set_title('n_estimators\n(Accuracy plateaus ~100-200; more = slower)')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# ── 2. max_depth: bias-variance ───────────────────────────────────────────────
d_range = [2, 3, 4, 5, 6, 8, 10, 15, None]
d_labels = [str(d) for d in d_range]
cv_d = []
for d in d_range:
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42),
        X_tr, y_tr, cv=5, scoring='roc_auc'
    )
    cv_d.append(cv.mean())

axes[0,1].bar(d_labels, cv_d, color='steelblue', alpha=0.8)
axes[0,1].set_xlabel('max_depth'); axes[0,1].set_ylabel('CV AUC')
axes[0,1].set_title('max_depth\n(None = unlimited usually best for RF)')
axes[0,1].grid(axis='y', alpha=0.3)

# ── 3. max_features: the critical RF parameter ────────────────────────────────
d = X.shape[1]
mf_range  = [1, 2, 3, int(np.sqrt(d)), int(d/3), d//2, d]
mf_labels = ['1', '2', '3', f'√d={int(np.sqrt(d))}', f'd/3={d//3}',
             f'd/2={d//2}', f'd={d}']
cv_mf = []
for mf in mf_range:
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42),
        X_tr, y_tr, cv=5, scoring='roc_auc'
    )
    cv_mf.append(cv.mean())

bars = axes[0,2].bar(mf_labels, cv_mf, color='tomato', alpha=0.8)
axes[0,2].set_xlabel('max_features')
axes[0,2].set_ylabel('CV AUC')
axes[0,2].set_title('max_features — The Critical RF Parameter\n'
                    '(√d is standard; tune carefully)')
axes[0,2].grid(axis='y', alpha=0.3)
# Mark default
axes[0,2].bar_label(bars, fmt='%.4f', padding=2, fontsize=7)

# ── 4. min_samples_leaf ───────────────────────────────────────────────────────
msl_range = [1, 2, 3, 5, 10, 20]
cv_msl = []
for msl in msl_range:
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=100, min_samples_leaf=msl, random_state=42),
        X_tr, y_tr, cv=5, scoring='roc_auc'
    )
    cv_msl.append(cv.mean())

axes[1,0].plot(msl_range, cv_msl, 'o-', color='green', lw=2.5)
axes[1,0].set_xlabel('min_samples_leaf'); axes[1,0].set_ylabel('CV AUC')
axes[1,0].set_title('min_samples_leaf\n(Larger = smoother predictions, less overfit)')
axes[1,0].grid(True, alpha=0.3)

# ── 5. Summary table ─────────────────────────────────────────────────────────
axes[1,1].axis('off')
table_data = [
    ['n_estimators',      '100–500', '100', 'More trees = more stable'],
    ['max_depth',         'None',    'None', 'Deep trees + bagging = good'],
    ['max_features',      '√d (cls)', 'd/3 (reg)', 'Tune this first!'],
    ['min_samples_leaf',  '1',       '1',   'Increase for noisy data'],
    ['min_samples_split', '2',       '2',   'Rarely needs tuning'],
    ['bootstrap',         'True',    'True', 'Disable → pasting, rarely better'],
    ['oob_score',         'True',    'True', 'Free validation — always enable'],
    ['n_jobs',            '-1',      '-1',  'Parallelize across all cores'],
    ['class_weight',      'None',    'N/A', '"balanced" for imbalanced data'],
]
table = axes[1,1].table(
    cellText=table_data,
    colLabels=['Parameter', 'Clf default', 'Reg default', 'Notes'],
    cellLoc='left', loc='center',
    bbox=[0, 0.05, 1, 0.9]
)
table.auto_set_font_size(False)
table.set_fontsize(8)
axes[1,1].set_title('Random Forest Hyperparameter Guide', fontweight='bold')

# ── 6. Warm start: add trees incrementally ────────────────────────────────────
rf_warm = RandomForestClassifier(n_estimators=10, warm_start=True, random_state=42)
n_steps = [10, 20, 40, 60, 80, 100, 150, 200]
oob_warm = []

for n in n_steps:
    rf_warm.set_params(n_estimators=n, oob_score=True)
    rf_warm.fit(X_tr, y_tr)
    oob_warm.append(rf_warm.oob_score_)

axes[1,2].plot(n_steps, oob_warm, 'o-', color='purple', lw=2.5)
axes[1,2].set_xlabel('n_estimators'); axes[1,2].set_ylabel('OOB Accuracy')
axes[1,2].set_title('warm_start=True\n(Add trees incrementally; stop when OOB plateaus)')
axes[1,2].grid(True, alpha=0.3)

plt.suptitle('Random Forest Hyperparameter Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_rf_hyperparams.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

### 4.4 Feature Importance in Random Forests

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

# MDI importance with std across trees
mdi_mean  = rf.feature_importances_
mdi_std   = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)

# Permutation importance on test set
perm_res  = permutation_importance(rf, X_te, y_te, n_repeats=20,
                                    random_state=42, scoring='roc_auc')

sort_mdi  = np.argsort(mdi_mean)[::-1]
sort_perm = np.argsort(perm_res.importances_mean)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# MDI with error bars (std across trees)
ax = axes[0]
top_n = 15
ax.barh(range(top_n),
        mdi_mean[sort_mdi[:top_n]][::-1],
        xerr=mdi_std[sort_mdi[:top_n]][::-1] * 2,
        color='steelblue', alpha=0.8, error_kw={'capsize': 3})
ax.set_yticks(range(top_n))
ax.set_yticklabels([cancer.feature_names[i] for i in sort_mdi[:top_n]][::-1],
                   fontsize=8)
ax.set_xlabel('MDI Importance (±2 std across trees)')
ax.set_title('Random Forest MDI Feature Importance\n'
             'Error bars = variability across 200 trees', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Permutation importance
ax2 = axes[1]
ax2.barh(range(top_n),
         perm_res.importances_mean[sort_perm[:top_n]][::-1],
         xerr=perm_res.importances_std[sort_perm[:top_n]][::-1] * 2,
         color='tomato', alpha=0.8, error_kw={'capsize': 3})
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([cancer.feature_names[i] for i in sort_perm[:top_n]][::-1],
                    fontsize=8)
ax2.set_xlabel('Mean AUC drop when permuted (±2 std)')
ax2.set_title('Permutation Importance (Test Set)\n'
              'More reliable: measures actual predictive power', fontweight='bold')
ax2.axvline(0, color='black', lw=0.8)
ax2.grid(axis='x', alpha=0.3)

plt.suptitle('Random Forest Feature Importance: MDI vs Permutation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_rf_importance.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 5. Boosting — Learning from Mistakes

### 5.1 AdaBoost: Reweighting Hard Examples

AdaBoost (Freund & Schapire, 1997) trains classifiers **sequentially**. After each round, samples that were misclassified are given higher weight, forcing the next classifier to focus on the hard examples.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class AdaBoostScratch:
    """
    AdaBoost for binary classification {-1, +1}.
    Uses decision stumps (max_depth=1) as weak learners.
    """

    def __init__(self, n_estimators=50, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        """
        y must be in {-1, +1}
        """
        rng = np.random.default_rng(self.random_state)
        n   = len(y)

        # Initialize uniform sample weights
        w   = np.ones(n) / n

        self.estimators_    = []
        self.alphas_        = []
        self.stage_errors_  = []

        for t in range(self.n_estimators):
            # Train a weak learner on weighted samples
            # sklearn doesn't use sample_weight in the same way — use a stump
            stump = DecisionTreeClassifier(max_depth=1,
                                           random_state=int(rng.integers(0, 2**31)))
            stump.fit(X, y, sample_weight=w)

            y_pred = stump.predict(X)

            # Weighted error
            incorrect  = (y_pred != y).astype(float)
            err        = np.dot(w, incorrect) / w.sum()
            err        = np.clip(err, 1e-10, 1 - 1e-10)   # Numerical safety

            # Classifier weight α: confident when error is low
            alpha = 0.5 * np.log((1 - err) / err)

            # Update sample weights:
            # Misclassified samples get higher weight: multiply by e^α
            # Correctly classified get lower weight: multiply by e^{-α}
            # Note: y * y_pred = +1 if correct, -1 if incorrect
            w = w * np.exp(-alpha * y * y_pred)
            w = w / w.sum()   # Renormalize to probability distribution

            self.estimators_.append(stump)
            self.alphas_.append(alpha)
            self.stage_errors_.append(err)

        self.alphas_ = np.array(self.alphas_)
        return self

    def predict(self, X):
        """Final prediction: weighted majority vote"""
        weighted_sum = sum(alpha * estimator.predict(X)
                           for alpha, estimator in
                           zip(self.alphas_, self.estimators_))
        return np.sign(weighted_sum).astype(int)

    def staged_predict(self, X):
        """Yield predictions after each stage for learning curve."""
        weighted_sum = np.zeros(len(X))
        for alpha, estimator in zip(self.alphas_, self.estimators_):
            weighted_sum += alpha * estimator.predict(X)
            yield np.sign(weighted_sum).astype(int)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


# ── Test on classification data ───────────────────────────────────────────────
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                            random_state=42)
y_ab = 2*y - 1   # Convert {0,1} to {-1,+1}

X_tr, X_te, y_tr, y_te = train_test_split(X, y_ab, test_size=0.3, random_state=42)

ada = AdaBoostScratch(n_estimators=200, random_state=42)
ada.fit(X_tr, y_tr)

# Staged accuracy
tr_staged = [np.mean(p == y_tr) for p in ada.staged_predict(X_tr)]
te_staged = [np.mean(p == y_te) for p in ada.staged_predict(X_te)]

from sklearn.ensemble import AdaBoostClassifier
ada_sk = AdaBoostClassifier(n_estimators=200, random_state=42, algorithm='SAMME')
ada_sk.fit(X_tr, y_tr)

print("AdaBoost — Scratch vs Sklearn:")
print(f"  Scratch: {ada.score(X_te, y_te):.4f}")
print(f"  Sklearn: {ada_sk.score(X_te, y_te):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(tr_staged, color='steelblue', lw=2, label='Train acc (scratch)')
ax.plot(te_staged, color='tomato',    lw=2, label='Test acc (scratch)')
ax.set_xlabel('Boosting round'); ax.set_ylabel('Accuracy')
ax.set_title('AdaBoost Staged Accuracy\n'
             'Note: test accuracy continues improving past train plateau')
ax.legend(); ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(ada.stage_errors_, color='green', lw=2, label='Weak learner error')
ax2.plot(ada.alphas_ / ada.alphas_.max(), color='purple', lw=2,
         label='Classifier weight α (normalized)')
ax2.axhline(0.5, color='gray', lw=1, linestyle='--', label='Random (50%)')
ax2.set_xlabel('Boosting round'); ax2.set_ylabel('Value')
ax2.set_title('AdaBoost: Stump Error and Weight\n'
              'Early stumps have low error → high weight')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/08_adaboost.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 5.2 Gradient Boosting: The General Framework

Gradient boosting (Friedman, 2001) unifies boosting with gradient descent. Instead of reweighting samples, it **fits each new tree to the residuals of the current ensemble**.

**The algorithm:**

$$F_0(\mathbf{x}) = \bar{y} \quad \text{(initialize with mean)}$$

For $m = 1, \ldots, M$:
1. Compute **pseudo-residuals** (negative gradient of loss):
$$r_i^{(m)} = -\left[\frac{\partial L(y_i, F(\mathbf{x}^{(i)}))}{\partial F(\mathbf{x}^{(i)})}\right]_{F = F_{m-1}}$$
2. Fit a tree $h_m$ to the residuals $\{(\mathbf{x}^{(i)}, r_i^{(m)})\}$
3. Update: $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$

For MSE loss: $L = \frac{1}{2}(y - F)^2 \Rightarrow r_i = y_i - F(\mathbf{x}^{(i)})$ (ordinary residuals).  
For log-loss: $r_i = y_i - \sigma(F(\mathbf{x}^{(i)}))$ (probability residuals).

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
GRADIENT BOOSTING — KEY INTUITION
══════════════════════════════════════════════════════════════════════

Think of it as gradient descent in function space:
  • In parameter space (neural nets): update w ← w - η ∇_w Loss
  • In function space (GBM):         F ← F + η h  where h ≈ -∇_F Loss

Each tree h_m learns to correct what the current ensemble F_{m-1} gets wrong.

For MSE loss: pseudo-residuals = y - F_{m-1}(x)
  → Each tree literally fits the prediction errors of the ensemble so far.
  
  Round 0: predict mean(y)
  Round 1: predict residuals from round 0
  Round 2: predict residuals from rounds 0+1
  ...
  Round M: final prediction = sum of all trees × learning rate

For log-loss (classification): residuals = y - P(y=1|x)
  → Each tree fits the error in predicted PROBABILITY, not class.
  → After logit transform, binary cross-entropy grad simplifies elegantly.

Learning rate η:
  Small η → need more trees, but more regularization, usually better
  Large η → fewer trees needed, but may overstep optima
  Typical: η ∈ [0.01, 0.3]; use smaller η with larger n_estimators
══════════════════════════════════════════════════════════════════════
""")
```

---

### 5.3 Gradient Boosting from Scratch

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

class GradientBoostingRegressorScratch:
    """
    Gradient Boosting for regression (MSE loss).
    Implements the Friedman (2001) algorithm from scratch.
    Pseudo-residuals for MSE = y - F(x).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_leaf=5, random_state=42):
        self.n_estimators    = n_estimators
        self.learning_rate   = learning_rate
        self.max_depth       = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state    = random_state

    def fit(self, X, y):
        # Initialize with constant prediction = mean(y)
        # For MSE: F_0 = argmin_c Σ(yi - c)² = mean(y)
        self.F0_ = y.mean()
        F        = np.full(len(y), self.F0_)

        self.trees_   = []
        self.train_mse_ = []

        for m in range(self.n_estimators):
            # Step 1: Compute pseudo-residuals
            # For MSE: r_i = y_i - F_{m-1}(x_i)
            residuals = y - F

            # Step 2: Fit a shallow tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + m
            )
            tree.fit(X, residuals)

            # Step 3: Update ensemble
            update = tree.predict(X)
            F      = F + self.learning_rate * update

            self.trees_.append(tree)
            self.train_mse_.append(mean_squared_error(y, F))

        return self

    def predict(self, X):
        F = np.full(len(X), self.F0_)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X):
        """Yield prediction after each stage (for plotting)."""
        F = np.full(len(X), self.F0_)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
            yield F.copy()


# ── Visualize the boosting process step by step ───────────────────────────────
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

rng = np.random.default_rng(42)
X_r = np.sort(rng.uniform(0, 6, 150)).reshape(-1, 1)
y_r = np.sin(X_r.ravel()) * 3 + X_r.ravel() * 0.3 + rng.normal(0, 0.4, 150)

gb = GradientBoostingRegressorScratch(
    n_estimators=100, learning_rate=0.1, max_depth=3
)
gb.fit(X_r, y_r)

X_plot = np.linspace(0, 6, 300).reshape(-1, 1)

fig, axes = plt.subplots(2, 4, figsize=(16, 9))

stages_to_show = [1, 2, 3, 5, 10, 20, 50, 100]
stage_preds    = list(gb.staged_predict(X_plot))

for ax, stage in zip(axes.flatten(), stages_to_show):
    F_stage = stage_preds[stage - 1]
    # Residuals at this stage
    F_train_stage = list(gb.staged_predict(X_r))[stage - 1]
    residuals     = y_r - F_train_stage

    ax.scatter(X_r.ravel(), y_r, s=12, alpha=0.4, color='steelblue', label='Data')
    ax.plot(X_plot.ravel(), F_stage, 'tomato', lw=2.5,
            label=f'F_{stage}(x)')
    # Show residuals as tick marks
    ax.vlines(X_r.ravel(), F_train_stage, y_r,
              colors='gray', alpha=0.2, lw=0.7)
    mse = mean_squared_error(y_r, F_train_stage)
    ax.set_title(f'After {stage} tree{"s" if stage>1 else ""}\nMSE={mse:.4f}')
    ax.set_xlabel('x'); ax.set_ylim(-4, 8)
    if stage == 1: ax.legend(fontsize=7)

plt.suptitle('Gradient Boosting — Sequential Tree Fitting\n'
             'Each tree fits the RESIDUALS of the current ensemble',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_gbm_stages.png', dpi=130, bbox_inches='tight')
plt.show()

# Compare with sklearn GBM
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

gb_scratch = GradientBoostingRegressorScratch(n_estimators=100, learning_rate=0.1)
gb_scratch.fit(X_tr, y_tr)

gb_sk = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42)
gb_sk.fit(X_tr, y_tr)

print("Gradient Boosting Regression — Scratch vs Sklearn:")
print(f"  Scratch: R²={r2_score(y_te, gb_scratch.predict(X_te)):.4f}")
print(f"  Sklearn: R²={r2_score(y_te, gb_sk.predict(X_te)):.4f}")
```

---

## 6. XGBoost — Extreme Gradient Boosting

XGBoost (Chen & Guestrin, 2016) extends gradient boosting with second-order (Newton) approximations, regularization, and engineering optimizations that make it 10–100× faster than vanilla GBM.

**Key improvements over vanilla GBM:**

1. **Second-order Taylor approximation** of the loss → better tree leaf values
2. **L1 + L2 regularization** on tree leaf weights built into the objective
3. **Column subsampling** at each tree and each split (like Random Forests)
4. **Approximate split finding** for large datasets
5. **Sparsity-aware splits** — handles missing values natively

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")

housing = fetch_california_housing()
X, y    = housing.data, housing.target
feature_names = housing.feature_names

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

if XGB_AVAILABLE:
    # ── XGBoost key hyperparameters ───────────────────────────────────────────
    xgb_model = xgb.XGBRegressor(
        n_estimators    = 500,         # Number of trees
        learning_rate   = 0.05,        # η — step size shrinkage
        max_depth       = 6,           # Tree depth (3-10 typical)
        subsample       = 0.8,         # Row sampling per tree (bagging)
        colsample_bytree= 0.8,         # Column sampling per tree (RF-like)
        colsample_bylevel=0.8,         # Column sampling per level
        min_child_weight= 5,           # Minimum sum of weights in a child
        reg_alpha       = 0.1,         # L1 regularization on leaf weights
        reg_lambda      = 1.0,         # L2 regularization on leaf weights
        gamma           = 0.05,        # Minimum loss reduction for split
        random_state    = 42,
        n_jobs          = -1,
        verbosity       = 0,
        eval_metric     = 'rmse',
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set      = [(X_tr, y_tr), (X_te, y_te)],
        verbose       = False,
        early_stopping_rounds = 30     # Stop if no improvement for 30 rounds
    )

    r2_xgb  = r2_score(y_te, xgb_model.predict(X_te))
    rmse_xgb = np.sqrt(mean_squared_error(y_te, xgb_model.predict(X_te)))
    print(f"XGBoost — Test R²: {r2_xgb:.4f}, RMSE: {rmse_xgb:.4f}")
    print(f"Best iteration: {xgb_model.best_iteration}")

    # ── Learning curves from eval history ────────────────────────────────────
    evals_result = xgb_model.evals_result()
    fig, axes   = plt.subplots(1, 2, figsize=(13, 5))

    tr_rmse = evals_result['validation_0']['rmse']
    te_rmse = evals_result['validation_1']['rmse']
    rounds  = range(len(tr_rmse))

    axes[0].plot(rounds, tr_rmse, color='steelblue', lw=2, label='Train RMSE')
    axes[0].plot(rounds, te_rmse, color='tomato',    lw=2, label='Test RMSE')
    axes[0].axvline(xgb_model.best_iteration, color='black', lw=1.5,
                    linestyle='--', label=f'Best iter={xgb_model.best_iteration}')
    axes[0].set_xlabel('Boosting round'); axes[0].set_ylabel('RMSE')
    axes[0].set_title('XGBoost Learning Curves\n(Early stopping prevents overfit)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Feature importance
    fi_gain   = xgb_model.get_booster().get_score(importance_type='gain')
    fi_cover  = xgb_model.get_booster().get_score(importance_type='cover')
    fi_weight = xgb_model.get_booster().get_score(importance_type='weight')

    ax2 = axes[1]
    fi_df = pd.DataFrame({
        'feature': list(fi_gain.keys()),
        'gain':    list(fi_gain.values()),
    }).sort_values('gain', ascending=True).tail(8)
    bars = ax2.barh(fi_df['feature'], fi_df['gain'], color='steelblue', alpha=0.8)
    ax2.set_xlabel('Importance (Gain: average gain per split)')
    ax2.set_title('XGBoost Feature Importance\n'
                  '(Gain = improvement in loss when feature is used)')
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle('XGBoost — California Housing Regression',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/08_xgboost.png', dpi=140, bbox_inches='tight')
    plt.show()

print("""
XGBoost Hyperparameter Guide:
──────────────────────────────────────────────────────────────────────
n_estimators     : 100–2000; use early stopping, not fixed number
learning_rate    : 0.01–0.3; smaller = more trees needed but better
max_depth        : 3–10; default 6; deeper = more complex patterns
subsample        : 0.5–1.0; row sampling (like bagging)
colsample_bytree : 0.5–1.0; column sampling per tree
min_child_weight : 1–20; larger = more conservative (regularization)
gamma            : 0–5; minimum split gain (regularization)
reg_alpha        : 0–5; L1 regularization on leaf weights
reg_lambda       : 1–10; L2 regularization on leaf weights (default=1)

Quick tuning recipe:
  1. Start: n_estimators=500, lr=0.1, max_depth=6, subsample=0.8,
            colsample_bytree=0.8; use early stopping
  2. Tune: max_depth and min_child_weight (main bias-variance knobs)
  3. Tune: subsample and colsample_bytree (variance reduction)
  4. Lower lr to 0.05 or 0.01, refit with more trees
  5. Fine-tune: gamma, reg_alpha, reg_lambda
──────────────────────────────────────────────────────────────────────
""")
```

---

## 7. LightGBM — Fast Gradient Boosting

LightGBM (Ke et al., 2017) makes gradient boosting faster and more memory-efficient with two innovations:

1. **Leaf-wise tree growth** (vs. level-wise in XGBoost): grows the leaf with the highest gain, creating asymmetric trees that converge faster.
2. **GOSS** (Gradient-based One-Side Sampling): keeps all samples with large gradients, subsamples those with small gradients — focuses computation where it matters.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not installed. Install with: pip install lightgbm")

if LGB_AVAILABLE:
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_tr, X_va, X_te = X[:15000], X[15000:18000], X[18000:]
    y_tr, y_va, y_te = y[:15000], y[15000:18000], y[18000:]

    lgb_model = lgb.LGBMRegressor(
        n_estimators      = 1000,
        learning_rate     = 0.05,
        num_leaves        = 63,       # 2^max_depth - 1; key LGB parameter
        max_depth         = -1,       # -1 = no limit; controlled by num_leaves
        min_child_samples = 20,
        subsample         = 0.8,
        subsample_freq    = 1,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1
    )

    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False),
                 lgb.log_evaluation(period=-1)]

    lgb_model.fit(
        X_tr, y_tr,
        eval_set        = [(X_va, y_va)],
        callbacks       = callbacks
    )

    r2_lgb   = r2_score(y_te, lgb_model.predict(X_te))
    rmse_lgb = np.sqrt(mean_squared_error(y_te, lgb_model.predict(X_te)))
    print(f"LightGBM — Test R²: {r2_lgb:.4f}, RMSE: {rmse_lgb:.4f}")
    print(f"Best iteration: {lgb_model.best_iteration_}")

print("""
LightGBM vs XGBoost vs sklearn GBM:
────────────────────────────────────────────────────────────────────────
Feature               sklearn GBM     XGBoost         LightGBM
────────────────────────────────────────────────────────────────────────
Tree growth           Level-wise      Level-wise      Leaf-wise (faster)
Speed                 Slow            Fast            Very fast
Memory                High            Medium          Low
Categorical features  Manual encode   Manual encode   Native support
Missing values        No              Yes             Yes
GPU support           No              Yes             Yes
Typical use           Baseline/learn  Production      Production/large data
Best for              Small datasets  General         Large datasets

LightGBM-specific hyperparameters:
  num_leaves     : Controls complexity. Analogous to 2^max_depth.
                   Default 31. Range: 10-300. Use with min_child_samples.
  min_child_samples: Minimum samples in leaf (default 20). Regularization.
  subsample_freq : Frequency of row subsampling (every N trees).

Rule of thumb: LightGBM ≈ 3-10× faster than XGBoost on most datasets.
               Accuracy is usually comparable or slightly better.
────────────────────────────────────────────────────────────────────────
""")
```

---

## 8. Stacking — Ensembling Ensembles

**Stacking** uses a **meta-learner** to combine the predictions of multiple base models. Unlike bagging (average) or boosting (sequential), stacking learns *how to best combine* diverse models.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

# ── Stacking Architecture ─────────────────────────────────────────────────────
# Level 0 (Base Learners): diverse models
# Level 1 (Meta-learner): learns to combine base predictions

# Key: base learners must produce out-of-fold predictions to avoid
# leakage into the meta-learner → StackingClassifier handles this

base_learners = [
    ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lr',  Pipeline([('sc', StandardScaler()),
                      ('m', LogisticRegression(C=1.0, max_iter=1000))])),
    ('svc', Pipeline([('sc', StandardScaler()),
                      ('m', SVC(probability=True, random_state=42))])),
]

# Meta-learner: simple logistic regression
meta_learner = LogisticRegression(C=1.0, max_iter=1000)

stacking_clf = StackingClassifier(
    estimators      = base_learners,
    final_estimator = meta_learner,
    cv              = StratifiedKFold(5, shuffle=True, random_state=42),
    stack_method    = 'predict_proba',   # Pass probabilities to meta-learner
    n_jobs          = -1,
    passthrough     = False   # Set True to also pass original features to meta
)
stacking_clf.fit(X_tr, y_tr)

# Evaluate all models
models_to_eval = {
    'Decision Tree':    DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':    RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Reg.':    Pipeline([('sc', StandardScaler()),
                                   ('m', LogisticRegression(max_iter=1000))]),
    'SVM':              Pipeline([('sc', StandardScaler()),
                                   ('m', SVC(probability=True, random_state=42))]),
    'Stacking':         stacking_clf,
}

print(f"\n{'Model':<25} {'CV AUC':>10} {'Test AUC':>10}")
print("-" * 48)

results = {}
for name, model in models_to_eval.items():
    if name != 'Stacking':
        model.fit(X_tr, y_tr)
    cv_auc   = cross_val_score(model if name != 'Stacking' else
                               StackingClassifier(
                                   estimators=base_learners,
                                   final_estimator=meta_learner,
                                   cv=3, n_jobs=-1
                               ),
                               X_tr, y_tr, cv=5, scoring='roc_auc').mean()
    test_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
    results[name] = test_auc
    print(f"{name:<25} {cv_auc:>10.4f} {test_auc:>10.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#EF9A9A', '#FFCC80', '#A5D6A7', '#90CAF9', '#CE93D8', '#B0BEC5']
names  = list(results.keys())
aucs   = list(results.values())
bars   = ax.barh(names, aucs, color=colors, alpha=0.9)
ax.bar_label(bars, fmt='%.4f', padding=4)
ax.axvline(results['Random Forest'], color='gray', lw=1.5, linestyle='--')
ax.set_xlabel('Test AUC')
ax.set_title('Stacking vs Individual Models — Breast Cancer\n'
             'Stacking combines diverse models to exceed any individual')
ax.set_xlim(0.97, 1.01)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/08_stacking.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
\nStacking Best Practices:
  1. Use DIVERSE base learners (tree + linear + kernel = good)
  2. Base learners must be fit on out-of-fold predictions → prevents leakage
  3. Meta-learner should be simple (logistic regression, ridge)
     Complex meta-learner → overfitting to base learner outputs
  4. passthrough=True: also pass original features to meta-learner
     (can help when base learners don't fully capture some signal)
  5. Stacking rarely wins by a huge margin over a well-tuned single model
     Use when every percentage point matters (Kaggle competitions)
""")
```

---

## 9. Choosing the Right Ensemble

```python
print("""
ENSEMBLE METHOD SELECTION GUIDE
══════════════════════════════════════════════════════════════════════════════

METHOD COMPARISON:
                    RF          Bagging     AdaBoost    GBM         XGBoost/LGB
────────────────────────────────────────────────────────────────────────────────
Bias reduction      ✗           ✗           ✓✓          ✓✓          ✓✓
Variance reduction  ✓✓          ✓           ✓           ✓           ✓
Speed               Fast        Fast        Medium      Slow        Fast
Robustness to noise ✓✓          ✓✓          ✗           ✓           ✓
Interpretability    Medium      Low         Low         Low         Low
Overfitting risk    Low         Low         High        Medium      Low-Med
Needs tuning        Low         Low         Medium      High        High
Missing values      ✗           ✗           ✗           ✗           ✓✓
Baseline quality    Excellent   Good        Good        Excellent   Excellent

DECISION TREE:
  When: Interpretability required, baseline, rules needed
  When not: Best accuracy needed

RANDOM FOREST:
  When: Strong baseline needed with minimal tuning
        Noisy data, outliers present
        Need reliable feature importance
        OOB evaluation is convenient
        Parallel training needed
  Typical accuracy: 5-15% better than single tree

GRADIENT BOOSTING (sklearn):
  When: Educational purposes, understanding the algorithm
        Small datasets where speed is not a concern
  Not for: Large datasets (too slow)

XGBOOST:
  When: Best accuracy on tabular data needed
        Large datasets (millions of rows)
        Missing values present
        Need L1/L2 regularization on trees
  Standard for: Kaggle competitions, production ML

LIGHTGBM:
  When: XGBoost but 3-10× faster
        Very large datasets (100M+ rows)
        Need native categorical support
        Memory-constrained environments

STACKING:
  When: Squeezing last 0.1-0.5% AUC improvement
        Ensembling models from diverse families
  Not for: When you need interpretability or fast inference

PRACTICAL RECOMMENDATION:
  1. Start with Random Forest (zero tuning, solid baseline)
  2. If more accuracy needed: try XGBoost/LightGBM
  3. Tune the gradient boosting model carefully
  4. Use stacking only when single-model performance plateaus
══════════════════════════════════════════════════════════════════════════════
""")
```

---

## 10. Case Study: Tabular Data Showdown

```python
# =============================================================================
# Full Benchmark: All major ensemble methods on California Housing
# Goal: demonstrate practical workflow from baseline to best model
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      KFold, RandomizedSearchCV)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               BaggingRegressor, ExtraTreesRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load and split ─────────────────────────────────────────────────────────
housing = fetch_california_housing()
X, y    = housing.data, housing.target
feature_names = housing.feature_names

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
def engineer_features(X, feature_names):
    """Add domain-relevant interaction features."""
    df = pd.DataFrame(X, columns=feature_names)
    df['rooms_per_household']   = df['AveRooms'] / (df['AveOccup'] + 1e-5)
    df['bedrooms_per_room']     = df['AveBedrms'] / (df['AveRooms'] + 1e-5)
    df['log_population']        = np.log1p(df['Population'])
    df['log_AveOccup']          = np.log1p(df['AveOccup'])
    df['MedInc_sq']             = df['MedInc'] ** 2
    return df.values

X_tr_eng = engineer_features(X_tr, feature_names)
X_te_eng  = engineer_features(X_te, feature_names)

# ── 3. Model zoo ──────────────────────────────────────────────────────────────
models = {
    'Ridge Regression (baseline)': Pipeline([
        ('sc', StandardScaler()),
        ('m',  Ridge(alpha=1.0))
    ]),
    'Single Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
    'Bagging': BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=None),
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=200, max_features=0.33, random_state=42, n_jobs=-1
    ),
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=200, max_features=0.33, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42
    ),
}

# Add XGBoost and LightGBM if available
try:
    import xgboost as xgb
    models['XGBoost'] = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbosity=0
    )
except ImportError: pass

try:
    import lightgbm as lgb
    models['LightGBM'] = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbose=-1
    )
except ImportError: pass

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
results = []
kf = KFold(5, shuffle=True, random_state=42)

print(f"\n{'Model':<32} {'Train R²':>9} {'Test R²':>9} {'Test RMSE':>10} {'Test MAE':>10}")
print("-" * 75)

for name, model in models.items():
    X_use = X_tr_eng if name not in ['Ridge Regression (baseline)'] else X_tr_eng
    model.fit(X_use, y_tr)
    y_pred_tr = model.predict(X_use)
    y_pred_te = model.predict(X_te_eng)

    train_r2  = r2_score(y_tr, y_pred_tr)
    test_r2   = r2_score(y_te, y_pred_te)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
    test_mae  = mean_absolute_error(y_te, y_pred_te)

    results.append({
        'Model': name, 'Train R²': train_r2, 'Test R²': test_r2,
        'RMSE': test_rmse, 'MAE': test_mae
    })
    print(f"{name:<32} {train_r2:>9.4f} {test_r2:>9.4f} "
          f"{test_rmse:>10.4f} {test_mae:>10.4f}")

# ── 5. Visualization ──────────────────────────────────────────────────────────
df_results = pd.DataFrame(results).sort_values('Test R²', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
colors_bar = ['#EF9A9A' if r < 0.7 else '#FFCC80' if r < 0.8
              else '#A5D6A7' if r < 0.85 else '#90CAF9'
              for r in df_results['Test R²']]
bars = ax.barh(df_results['Model'], df_results['Test R²'],
               color=colors_bar, alpha=0.9)
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=8)
ax.set_xlabel('Test R²')
ax.set_title('Test R² by Model\n(California Housing with Feature Engineering)')
ax.set_xlim(0.5, 1.0)
ax.axvline(df_results['Test R²'].max(), color='gold', lw=2, linestyle='--',
           label=f'Best: {df_results["Test R²"].max():.4f}')
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)

ax2 = axes[1]
ax2.scatter(df_results['RMSE'], df_results['Test R²'],
            s=150, color=colors_bar, zorder=5, edgecolors='k', linewidths=0.5)
for _, row in df_results.iterrows():
    ax2.annotate(row['Model'].split('(')[0].strip(),
                 (row['RMSE'], row['Test R²']),
                 textcoords='offset points', xytext=(5, 3), fontsize=7)
ax2.set_xlabel('Test RMSE (lower is better)')
ax2.set_ylabel('Test R² (higher is better)')
ax2.set_title('RMSE vs R²\n(Top-right corner = best)')
ax2.grid(True, alpha=0.3)

plt.suptitle('Ensemble Methods — Tabular Data Benchmark\n'
             'California Housing (n=20,640, d=8 + 5 engineered features)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_benchmark.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 11. Summary

### ✅ Must-Remember Mental Models

**1. The fundamental ensemble equation:**
$$\text{Ensemble Error} = \underbrace{\rho \sigma^2}_{\text{irreducible}} + \underbrace{\frac{1-\rho}{B}\sigma^2}_{\text{reducible by adding trees}}$$
Correlation $\rho$ between trees is the bottleneck. Random Forests reduce $\rho$; pure bagging does not.

**2. Bagging reduces variance, not bias.**
It averages many high-variance, low-bias models. The base learner must have low bias (deep trees) for this to work. Bagging a high-bias model (shallow tree) doesn't help much.

**3. Boosting reduces bias, not variance.**
It sequentially corrects the ensemble's errors. Each weak learner targets what the ensemble gets wrong. Can overfit on noisy datasets.

**4. Gradient Boosting = gradient descent in function space.**
Pseudo-residuals are the negative gradient of the loss. Fitting trees to residuals is equivalent to taking gradient steps. For MSE loss, residuals = ordinary residuals.

**5. OOB score = free cross-validation.**
Each tree only sees ~63% of the data. The remaining ~37% is a natural validation set. `oob_score=True` computes this at no extra cost — no need for a separate validation set when hyperparameter tuning a Random Forest.

**6. The `max_features` parameter is the key Random Forest hyperparameter.**
Default `sqrt(d)` for classification, `d/3` for regression. Smaller = more decorrelated trees = less variance but more bias. Always tune this.

**7. XGBoost/LightGBM are the defaults for tabular data competitions.**
Start here when you need maximum accuracy. Use early stopping to find optimal `n_estimators`. Tune `max_depth` and `learning_rate` first.

**8. Feature importances from single trees are unreliable.** 
Random Forest MDI averages across many trees → more stable but still biased.
Use permutation importance on a held-out set for reliable feature ranking.

**9. Stacking hierarchy:**
Level 0 = diverse base learners → Level 1 = meta-learner on out-of-fold predictions. Base learner predictions must be generated via cross-validation to prevent leakage.

**10. Practical hierarchy:**
Ridge/Logistic → Decision Tree → Bagging → Random Forest → XGBoost/LightGBM → Stacking.  
Each step costs more compute and tuning time for diminishing returns.

---

## 12. Exercises

**Conceptual:**

1. **Prove the variance formula**: Show that the variance of the average of $B$ identically distributed random variables, each with variance $\sigma^2$ and pairwise correlation $\rho$, equals $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. What is the limit as $B\to\infty$?

2. **AdaBoost analysis**: In AdaBoost, show that the weight update rule $w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))$ causes the total weight on correctly classified samples to equal $\frac{1}{2}(1-\epsilon_t)$ and on misclassified samples to equal $\frac{1}{2}\epsilon_t$ (after renormalization) — i.e., misclassified samples get weight $\frac{1}{2}$ regardless of their original weight.

3. **Gradient Boosting for log-loss**: Derive the pseudo-residuals for binary cross-entropy loss: $L = -[y \log p + (1-y)\log(1-p)]$ where $p = \sigma(F)$. Show that the pseudo-residuals are $r_i = y_i - p_i$, exactly like logistic regression.

**Coding:**

4. **Bias-variance experiment**: Empirically measure bias² and variance for (a) a single decision tree, (b) a 100-tree Random Forest, and (c) a 100-round Gradient Boosting model. Use 200 bootstrap training sets of size 100 from a sine-wave dataset. Compare the three models' bias/variance decomposition.

5. **Extend GBM from scratch**: Add support for (a) classification with log-loss, (b) subsampling (`subsample` < 1), (c) column subsampling (`colsample_bytree` < 1). Compare to sklearn's `GradientBoostingClassifier` on the breast cancer dataset.

6. **OOB feature importance**: Using a Random Forest with `oob_score=True`, implement OOB-based permutation importance: for each feature, shuffle its values only in the OOB samples for each tree, measure the change in OOB accuracy. Compare to the standard MDI importance.

7. **Early stopping**: Implement early stopping for your GBM from scratch — use a validation set and stop training when validation loss doesn't improve for `patience` rounds. Plot the training/validation loss curve and mark the optimal stopping point.

**Challenge:**

8. **Implement XGBoost's second-order tree building**: In standard GBM we fit trees to first-order gradients (residuals). XGBoost uses a second-order Taylor approximation with both gradient $g_i$ and Hessian $h_i$. The optimal leaf weight is:
   $$w^* = -\frac{\sum_{i\in\text{leaf}} g_i}{\sum_{i\in\text{leaf}} h_i + \lambda}$$
   And the gain for a split is:
   $$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$
   Implement this for MSE loss (where $g_i = \hat{y}_i - y_i$ and $h_i = 1$). Verify that this recovers the ordinary residual-fitting rule when $\lambda = 0$.

---

## 13. References

- Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123–140. — Original bagging paper.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. — Original RF paper.
- Freund, Y., & Schapire, R. E. (1997). A Decision-Theoretic Generalization of On-Line Learning. *JCSS*, 55(1), 119–139. — AdaBoost.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232. — Original GBM paper.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*. — XGBoost.
- Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*. — LightGBM.
- Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241–259. — Stacking.

---

*Previous: [Chapter 7 — Decision Trees](07_decision_trees.md)*  
*Next: [Chapter 9 — Support Vector Machines](09_svm.md)*

*In Chapter 9, we derive the maximum-margin classifier, the kernel trick, and the dual problem — showing how SVMs find the unique decision boundary that is furthest from all training examples.*

---

> **Chapter 8 complete.** All code is in `notebooks/08_ensembles.ipynb`.  
> The `GradientBoostingRegressorScratch` implementation shows exactly what happens round by round — run `staged_predict` and watch the ensemble improve.
