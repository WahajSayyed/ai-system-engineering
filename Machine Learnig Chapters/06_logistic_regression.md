# Chapter 6: Logistic Regression & Binary Classification

> *"Logistic regression is a linear model in disguise. Its elegance lies in the fact that the non-linearity comes for free from the sigmoid — and the loss function is perfectly convex."*

---

## Table of Contents

1. [From Regression to Classification](#1-from-regression-to-classification)
2. [The Logistic Regression Model](#2-the-logistic-regression-model)
   - 2.1 [The Sigmoid Function](#21-the-sigmoid-function)
   - 2.2 [Decision Boundary](#22-decision-boundary)
   - 2.3 [Probabilistic Interpretation](#23-probabilistic-interpretation)
3. [Learning: Maximum Likelihood Estimation](#3-learning-maximum-likelihood-estimation)
   - 3.1 [The Binary Cross-Entropy Loss](#31-the-binary-cross-entropy-loss)
   - 3.2 [Deriving the Gradient](#32-deriving-the-gradient)
   - 3.3 [Implementing from Scratch](#33-implementing-from-scratch)
4. [Evaluating a Classifier](#4-evaluating-a-classifier)
   - 4.1 [The Confusion Matrix](#41-the-confusion-matrix)
   - 4.2 [Precision, Recall, F1](#42-precision-recall-f1)
   - 4.3 [ROC Curve and AUC](#43-roc-curve-and-auc)
   - 4.4 [Precision-Recall Curve](#44-precision-recall-curve)
   - 4.5 [Threshold Selection](#45-threshold-selection)
5. [Regularization in Logistic Regression](#5-regularization-in-logistic-regression)
6. [Multi-class Classification](#6-multi-class-classification)
   - 6.1 [One-vs-Rest (OvR)](#61-one-vs-rest-ovr)
   - 6.2 [Softmax Regression](#62-softmax-regression)
7. [Class Imbalance](#7-class-imbalance)
8. [Interpreting Logistic Regression](#8-interpreting-logistic-regression)
9. [Logistic Regression with Scikit-learn](#9-logistic-regression-with-scikit-learn)
10. [Case Study: Breast Cancer Diagnosis](#10-case-study-breast-cancer-diagnosis)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. From Regression to Classification

In Chapter 5 we predicted a **continuous output** $y \in \mathbb{R}$. Now we predict a **discrete category**: spam or not, malignant or benign, click or no-click.

**Why not just use linear regression for classification?**

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Binary dataset: two well-separated Gaussian blobs
n   = 80
X0  = rng.normal(2, 0.8, (n//2, 1))
X1  = rng.normal(5, 0.8, (n//2, 1))
X   = np.vstack([X0, X1])
y   = np.hstack([np.zeros(n//2), np.ones(n//2)])

# Fit linear regression
w   = np.polyfit(X.ravel(), y, 1)
x_line = np.linspace(-1, 8, 300)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Problem 1: linear regression produces values outside [0, 1]
ax = axes[0]
ax.scatter(X.ravel(), y, s=40, alpha=0.7,
           c=['steelblue' if yi==0 else 'tomato' for yi in y])
ax.plot(x_line, np.polyval(w, x_line), 'k-', lw=2, label='Linear regression')
ax.axhline(0, color='gray', lw=0.8); ax.axhline(1, color='gray', lw=0.8)
ax.axhline(0.5, color='green', lw=1.5, linestyle='--', label='Threshold 0.5')
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('x'); ax.set_ylabel('y (class label)')
ax.set_title('Linear Regression on Binary Data\n'
             'Predicts <0 and >1 — meaningless as probability!')
ax.legend(fontsize=8)

# Problem 2: extreme outlier breaks the linear classifier
X_ext = np.vstack([X0, X1, np.array([[20.]])])
y_ext = np.hstack([y, [1.]])
w_ext = np.polyfit(X_ext.ravel(), y_ext, 1)

ax2 = axes[1]
ax2.scatter(X.ravel(), y, s=40, alpha=0.7,
            c=['steelblue' if yi==0 else 'tomato' for yi in y], zorder=3)
ax2.scatter([20], [1], s=200, color='darkred', zorder=5, marker='*',
            label='Outlier added')
ax2.plot(x_line, np.polyval(w, x_line),   'k-',  lw=2, label='Without outlier')
ax2.plot(x_line, np.polyval(w_ext, x_line),'k--', lw=2, label='With outlier')
ax2.axhline(0.5, color='green', lw=1.5, linestyle=':')
ax2.set_ylim(-0.5, 1.5); ax2.set_xlim(-1, 8)
ax2.set_xlabel('x'); ax2.set_title('Linear Regression: Vulnerable to Outliers\n'
                                    'Decision boundary shifts dramatically')
ax2.legend(fontsize=7)

# Solution: logistic regression squashes output to (0, 1)
from scipy.special import expit as sigmoid

ax3 = axes[2]
w_log = [-6.0, 2.7]   # Approximate logistic fit
ax3.scatter(X.ravel(), y, s=40, alpha=0.7,
            c=['steelblue' if yi==0 else 'tomato' for yi in y])
ax3.plot(x_line, sigmoid(w_log[0] + w_log[1]*x_line), 'green', lw=2.5,
         label='Logistic regression')
ax3.axhline(0.5, color='gray', lw=1, linestyle='--', label='Decision boundary')
ax3.fill_between(x_line, 0, sigmoid(w_log[0]+w_log[1]*x_line),
                 alpha=0.1, color='tomato')
ax3.fill_between(x_line, sigmoid(w_log[0]+w_log[1]*x_line), 1,
                 alpha=0.1, color='steelblue')
ax3.set_ylim(-0.1, 1.1)
ax3.set_xlabel('x'); ax3.set_ylabel('P(y=1 | x)')
ax3.set_title('Logistic Regression\nOutputs calibrated probability in (0, 1)')
ax3.legend(fontsize=8)

plt.suptitle('Why Linear Regression Fails for Classification',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_why_not_linear.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 2. The Logistic Regression Model

### 2.1 The Sigmoid Function

Logistic regression uses the **sigmoid** (logistic) function to map a linear combination of features to a probability:

$$\hat{p} = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$$

The sigmoid has key properties that make it ideal for binary classification:
- Output is always in $(0, 1)$ — a valid probability
- Differentiable everywhere — gradient-based training works
- Derivative has a beautiful closed form: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- Symmetric: $\sigma(-z) = 1 - \sigma(z)$

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Numerically stable sigmoid."""
    # Avoid overflow: for z << 0, use 1 - sigmoid(-z) form
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

z = np.linspace(-8, 8, 500)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Sigmoid and its derivative
ax = axes[0]
sig  = sigmoid(z)
dsig = sig * (1 - sig)   # σ'(z) = σ(z)(1 - σ(z))

ax.plot(z, sig,  color='steelblue', lw=2.5, label='σ(z)')
ax.plot(z, dsig, color='tomato',    lw=2.5, label="σ'(z) = σ(1-σ)", linestyle='--')
ax.axhline(0.5, color='gray', lw=1, linestyle=':', label='σ(0) = 0.5')
ax.axvline(0,   color='gray', lw=1, linestyle=':')
ax.set_xlabel('z = w^T x')
ax.set_title('Sigmoid Function and Its Derivative\n'
             'Max gradient = 0.25 at z=0 (vanishing gradient concern!)')
ax.legend(); ax.grid(True, alpha=0.3)

# Effect of weights on the sigmoid shape
ax2 = axes[1]
for w, ls, color in [(0.5, '--', '#90CAF9'), (1.0, '-', 'steelblue'),
                      (2.0, '-.', '#1565C0'), (5.0, ':', 'darkblue')]:
    ax2.plot(z, sigmoid(w*z), lw=2, label=f'w={w}', linestyle=ls, color=color)
ax2.axhline(0.5, color='gray', lw=1, linestyle='--')
ax2.axvline(0,   color='gray', lw=1, linestyle='--')
ax2.set_xlabel('z'); ax2.set_title('Effect of Weight Magnitude\nLarger |w| → sharper decision boundary')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# Sigmoid vs tanh vs softplus
ax3 = axes[2]
ax3.plot(z, sigmoid(z),          color='steelblue', lw=2, label='sigmoid (0,1)')
ax3.plot(z, np.tanh(z),          color='tomato',    lw=2, label='tanh (-1,1)')
ax3.plot(z, np.log1p(np.exp(z)), color='green',     lw=2, label='softplus (0,∞)')
ax3.set_xlabel('z')
ax3.set_title('Sigmoid vs Tanh vs Softplus\n(All related: tanh(z) = 2σ(2z)-1)')
ax3.legend(); ax3.grid(True, alpha=0.3)

plt.suptitle('The Sigmoid Function', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_sigmoid.png', dpi=140, bbox_inches='tight')
plt.show()

# Numerical properties of sigmoid
print("Sigmoid properties:")
print(f"  σ(0)   = {sigmoid(0):.4f}   (boundary point)")
print(f"  σ(+∞)  → 1.0000          (asymptote)")
print(f"  σ(-∞)  → 0.0000          (asymptote)")
print(f"  σ'(0)  = {sigmoid(0)*(1-sigmoid(0)):.4f}   (max gradient)")
print(f"  σ(-z)  = 1 - σ(z):  {sigmoid(-2.0):.4f} + {sigmoid(2.0):.4f} = {sigmoid(-2.0)+sigmoid(2.0):.4f}")
```

---

### 2.2 Decision Boundary

The decision boundary is the surface where $\hat{p} = 0.5$, i.e., $\mathbf{w}^T\mathbf{x} = 0$.

For 2D features, this is a **straight line** — hence "linear classifier."

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles

rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Generate three datasets
datasets = [
    make_classification(n_samples=300, n_features=2, n_redundant=0,
                        n_informative=2, random_state=42),
    make_moons(n_samples=300, noise=0.2, random_state=42),
    make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42),
]

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

for ax, (X_d, y_d), title in zip(
    axes,
    datasets,
    ['Linearly Separable\n(LR works perfectly)',
     'Moon-shaped\n(LR fails — not linearly separable)',
     'Concentric circles\n(LR fails — non-linear boundary)']
):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_d)

    lr = LogisticRegression(random_state=42)
    lr.fit(X_sc, y_d)

    # Decision boundary
    x_min, x_max = X_sc[:,0].min()-0.5, X_sc[:,0].max()+0.5
    y_min, y_max = X_sc[:,1].min()-0.5, X_sc[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.7, vmin=0, vmax=1)
    ax.contour(xx,  yy, Z, levels=[0.5], colors='black', linewidths=2)
    scatter = ax.scatter(X_sc[:,0], X_sc[:,1], c=y_d,
                         cmap='RdBu_r', s=20, edgecolors='k', linewidths=0.3, alpha=0.8)
    acc = lr.score(X_sc, y_d)
    ax.set_title(f'{title}\nAccuracy = {acc:.2%}')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

plt.suptitle('Logistic Regression Decision Boundaries\n'
             '(Background color = predicted probability of class 1)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_decision_boundaries.png', dpi=140, bbox_inches='tight')
plt.show()

# ── The decision boundary equation ───────────────────────────────────────────
print("""
Logistic Regression Decision Boundary:
  σ(w^T x) = 0.5
  ↔  w^T x = 0
  ↔  w₀ + w₁x₁ + w₂x₂ = 0
  ↔  x₂ = -(w₀ + w₁x₁) / w₂    (line in 2D)

For the linear dataset:
""")
lr_linear = LogisticRegression(random_state=42)
X_lin, y_lin = datasets[0]
X_lin_sc = StandardScaler().fit_transform(X_lin)
lr_linear.fit(X_lin_sc, y_lin)
w = lr_linear.coef_[0]
b = lr_linear.intercept_[0]
print(f"  Weights: w₁={w[0]:.4f}, w₂={w[1]:.4f}")
print(f"  Bias:    b={b:.4f}")
print(f"  Decision boundary: {w[1]:.4f}x₂ = -{b:.4f} - {w[0]:.4f}x₁")
print(f"  i.e.,   x₂ = {-b/w[1]:.4f} + {-w[0]/w[1]:.4f} x₁")
```

---

### 2.3 Probabilistic Interpretation

Logistic regression models the **conditional probability** of class 1:

$$P(y=1 | \mathbf{x}; \mathbf{w}) = \sigma(\mathbf{w}^T\mathbf{x}) = \hat{p}$$

$$P(y=0 | \mathbf{x}; \mathbf{w}) = 1 - \sigma(\mathbf{w}^T\mathbf{x}) = 1 - \hat{p}$$

The **log-odds** (logit) is linear:

$$\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \log\frac{\hat{p}}{1-\hat{p}} = \mathbf{w}^T\mathbf{x}$$

This is the origin of the name: the **log**istic function maps **log**-odds to probability.

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Log-odds interpretation ────────────────────────────────────────────────────
p = np.linspace(0.001, 0.999, 500)
log_odds = np.log(p / (1 - p))   # logit function

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(p, log_odds, color='steelblue', lw=2.5)
ax.axhline(0, color='gray', lw=1, linestyle='--', label='log-odds = 0 → p = 0.5')
ax.axvline(0.5, color='gray', lw=1, linestyle='--')
ax.set_xlabel('Probability p')
ax.set_ylabel('Log-odds = log(p / (1-p))')
ax.set_title('Logit Function: Probability → Log-odds\n(Inverse of sigmoid)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax2 = axes[1]
odds_vals = np.logspace(-2, 2, 500)
ax2.semilogx(odds_vals, odds_vals/(1+odds_vals), color='tomato', lw=2.5)
ax2.axhline(0.5, color='gray', lw=1, linestyle='--', label='p=0.5 at odds=1')
ax2.axvline(1,   color='gray', lw=1, linestyle='--')
ax2.set_xlabel('Odds = p/(1-p)')
ax2.set_ylabel('Probability p')
ax2.set_title('Odds → Probability\nOdds=2 means event 2× more likely than not')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/06_logit.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
Interpreting logistic regression coefficients:

  log(p/(1-p)) = w₀ + w₁x₁ + ... + wₐxₐ

  A unit increase in xⱼ changes the LOG-ODDS by wⱼ.
  Equivalently: the ODDS are multiplied by exp(wⱼ).

  Example: w₁ = 0.5 for 'age' in a disease model
    → Each year of age multiplies the odds of disease by exp(0.5) ≈ 1.65
    → i.e., odds increase by 65% per year
    
  This is why logistic regression is widely used in medicine and economics:
  coefficients have a natural interpretation as log-odds ratios.
""")
```

---

## 3. Learning: Maximum Likelihood Estimation

### 3.1 The Binary Cross-Entropy Loss

Under the logistic model, the likelihood of observing $y^{(i)} \in \{0,1\}$ is:

$$p(y^{(i)} | \mathbf{x}^{(i)}; \mathbf{w}) = \hat{p}^{y^{(i)}} (1-\hat{p})^{1-y^{(i)}}$$

This is a compact way to write: $\hat{p}$ if $y^{(i)}=1$, and $(1-\hat{p})$ if $y^{(i)}=0$.

Log-likelihood over all $n$ samples:

$$\ell(\mathbf{w}) = \sum_{i=1}^n \left[y^{(i)} \log \hat{p}^{(i)} + (1-y^{(i)}) \log(1-\hat{p}^{(i)})\right]$$

Negating to get the **loss** to minimize:

$$\boxed{L(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^n \left[y^{(i)} \log \hat{p}^{(i)} + (1-y^{(i)}) \log(1-\hat{p}^{(i)})\right]}$$

This is **binary cross-entropy** (BCE).

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Visualize BCE loss behavior ────────────────────────────────────────────────
p_hat = np.linspace(0.001, 0.999, 500)

bce_y1 = -np.log(p_hat)        # Loss when y=1: penalizes low predicted probability
bce_y0 = -np.log(1 - p_hat)    # Loss when y=0: penalizes high predicted probability

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(p_hat, bce_y1, color='tomato',    lw=2.5, label='y=1: L = -log(p̂)')
ax.plot(p_hat, bce_y0, color='steelblue', lw=2.5, label='y=0: L = -log(1-p̂)')
ax.axvline(0.5, color='gray', lw=1, linestyle='--')
ax.set_xlabel('Predicted probability p̂')
ax.set_ylabel('Loss')
ax.set_title('Binary Cross-Entropy Loss\n'
             'Penalizes confident wrong predictions infinitely!')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 5)

# ── Loss landscape is convex ──────────────────────────────────────────────────
# Unlike neural networks, logistic regression loss has ONE global minimum
ax2 = axes[1]
w = np.linspace(-5, 5, 300)

# Generate toy data
rng = np.random.default_rng(0)
n   = 50
X_s = rng.normal(0, 1, (n, 2))
X_s = np.hstack([np.ones((n,1)), X_s])
y_s = (X_s[:,1] + X_s[:,2] > 0).astype(float)

def bce_loss_1d(w1_val, w_other=None):
    """BCE loss as function of w1 with other weights fixed."""
    w = np.array([0.0, w1_val, 0.5])
    z     = X_s @ w
    p_hat = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    loss  = -np.mean(y_s * np.log(p_hat+1e-10) + (1-y_s)*np.log(1-p_hat+1e-10))
    return loss

loss_vals = [bce_loss_1d(wi) for wi in w]
ax2.plot(w, loss_vals, color='steelblue', lw=2.5)
ax2.set_xlabel('w₁ (one weight)')
ax2.set_ylabel('BCE Loss')
ax2.set_title('BCE Loss is Convex in w\n'
              'One global minimum → gradient descent guaranteed to find it')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/06_bce_loss.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 3.2 Deriving the Gradient

The gradient of the BCE loss:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})$$

Where $\hat{\mathbf{p}} = \sigma(\mathbf{X}\mathbf{w})$.

This has the exact same form as the MSE gradient for linear regression — a remarkable simplification arising from the choice of sigmoid + cross-entropy.

```python
import numpy as np

print("""
Derivation of the BCE Gradient:
════════════════════════════════════════════════════════

Step 1: Loss function
  L(w) = -(1/n) Σᵢ [yᵢ log σ(wᵀxᵢ) + (1-yᵢ) log(1 - σ(wᵀxᵢ))]

Step 2: Let zᵢ = wᵀxᵢ and p̂ᵢ = σ(zᵢ)

Step 3: Chain rule
  ∂L/∂w = (1/n) Σᵢ ∂L/∂p̂ᵢ × ∂p̂ᵢ/∂zᵢ × ∂zᵢ/∂w

Step 4: Each term
  ∂L/∂p̂ᵢ = -(yᵢ/p̂ᵢ - (1-yᵢ)/(1-p̂ᵢ))
  ∂p̂ᵢ/∂zᵢ = σ(zᵢ)(1 - σ(zᵢ)) = p̂ᵢ(1 - p̂ᵢ)
  ∂zᵢ/∂w  = xᵢ

Step 5: Multiply and simplify (magic cancellation!):
  ∂L/∂p̂ᵢ × ∂p̂ᵢ/∂zᵢ
  = -(yᵢ/p̂ᵢ - (1-yᵢ)/(1-p̂ᵢ)) × p̂ᵢ(1-p̂ᵢ)
  = -(yᵢ(1-p̂ᵢ) - (1-yᵢ)p̂ᵢ)
  = -(yᵢ - p̂ᵢ)
  = (p̂ᵢ - yᵢ)

Step 6: Final gradient (vectorized)
  ∂L/∂w = (1/n) Σᵢ (p̂ᵢ - yᵢ) xᵢ  =  (1/n) Xᵀ(p̂ - y)

This is IDENTICAL in form to the MSE gradient:
  MSE gradient:  (2/n) Xᵀ(Xw - y)
  BCE gradient:  (1/n) Xᵀ(σ(Xw) - y)

Key insight: the sigmoid's gradient exactly cancels the derivative
of the log in BCE — this is not a coincidence. It's why we use BCE
with sigmoid, and MSE with linear output.
════════════════════════════════════════════════════════
""")
```

---

### 3.3 Implementing from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using gradient descent.
    Supports L2 regularization.
    """

    def __init__(self, lr=0.1, n_epochs=1000, C=1.0,
                 batch_size=None, tol=1e-6, random_state=42):
        """
        Parameters
        ----------
        lr          : learning rate
        n_epochs    : maximum number of epochs
        C           : inverse regularization strength (like sklearn). Larger C = less regularization.
        batch_size  : None for batch GD, int for mini-batch SGD
        tol         : convergence tolerance
        random_state: for reproducibility
        """
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.C            = C
        self.batch_size   = batch_size
        self.tol          = tol
        self.random_state = random_state
        self.history_     = {'loss': [], 'acc': []}

    @staticmethod
    def _sigmoid(z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    @staticmethod
    def _bce_loss(y, p_hat, w, C):
        n = len(y)
        eps = 1e-12
        nll = -np.mean(y * np.log(p_hat + eps) + (1-y) * np.log(1-p_hat + eps))
        l2  = (1 / (2 * C * n)) * np.dot(w[1:], w[1:])   # Don't regularize bias
        return nll + l2

    def fit(self, X, y):
        rng     = np.random.default_rng(self.random_state)
        n, d    = X.shape
        # Add bias column
        X_b     = np.hstack([np.ones((n, 1)), X])
        self.w_ = rng.standard_normal(d + 1) * 0.01
        prev_loss = np.inf

        for epoch in range(self.n_epochs):
            if self.batch_size is None:
                indices = np.arange(n)
            else:
                indices = rng.permutation(n)

            for start in range(0, n, self.batch_size or n):
                idx     = indices[start:start+(self.batch_size or n)]
                Xb, yb  = X_b[idx], y[idx]
                z       = Xb @ self.w_
                p_hat   = self._sigmoid(z)
                # Gradient = BCE gradient + L2 regularization gradient
                grad    = (1/len(yb)) * Xb.T @ (p_hat - yb)
                reg_grad = np.zeros_like(self.w_)
                reg_grad[1:] = self.w_[1:] / (self.C * n)
                self.w_ -= self.lr * (grad + reg_grad)

            # Track loss on full dataset
            p_full  = self._sigmoid(X_b @ self.w_)
            loss    = self._bce_loss(y, p_full, self.w_, self.C)
            acc     = accuracy_score(y, (p_full >= 0.5).astype(int))
            self.history_['loss'].append(loss)
            self.history_['acc'].append(acc)

            if abs(prev_loss - loss) < self.tol:
                print(f"Converged at epoch {epoch}")
                break
            prev_loss = loss

        return self

    def predict_proba(self, X):
        X_b  = np.hstack([np.ones((len(X), 1)), X])
        p    = self._sigmoid(X_b @ self.w_)
        return np.column_stack([1-p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ── Test on breast cancer dataset ─────────────────────────────────────────────
cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

# Train from scratch
lr_scratch = LogisticRegressionScratch(lr=0.5, n_epochs=500, C=1.0)
lr_scratch.fit(X_tr_sc, y_train)

# Compare with sklearn
from sklearn.linear_model import LogisticRegression
lr_sklearn = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_sklearn.fit(X_tr_sc, y_train)

print("Model Comparison — Breast Cancer")
print(f"  Scratch  — Test Acc: {lr_scratch.score(X_te_sc, y_test):.4f}, "
      f"AUC: {roc_auc_score(y_test, lr_scratch.predict_proba(X_te_sc)[:,1]):.4f}")
print(f"  Sklearn  — Test Acc: {lr_sklearn.score(X_te_sc, y_test):.4f}, "
      f"AUC: {roc_auc_score(y_test, lr_sklearn.predict_proba(X_te_sc)[:,1]):.4f}")

# ── Training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(len(lr_scratch.history_['loss']))
axes[0].plot(epochs, lr_scratch.history_['loss'], color='steelblue', lw=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
axes[0].set_title('Training Loss (from scratch)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, lr_scratch.history_['acc'], color='tomato', lw=2)
axes[1].axhline(lr_sklearn.score(X_tr_sc, y_train), color='green', lw=2,
                linestyle='--', label='sklearn final train acc')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/06_training_curves.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 4. Evaluating a Classifier

### 4.1 The Confusion Matrix

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_tr_s, y_tr)
y_pred = lr.predict(X_te_s)

cm = confusion_matrix(y_te, y_pred)
TN, FP, FN, TP = cm.ravel()

print("=" * 55)
print("THE CONFUSION MATRIX — COMPLETE BREAKDOWN")
print("=" * 55)
print(f"""
                  Predicted
                  Neg    Pos
Actual  Neg  │  TN={TN:>3} │  FP={FP:>3} │  ← False Positive Rate = FP/(TN+FP)
        Pos  │  FN={FN:>3} │  TP={TP:>3} │  ← True Positive Rate (Recall) = TP/(TP+FN)

Metric              Formula             Value    Meaning
───────────────────────────────────────────────────────────────────────
Accuracy            (TP+TN)/(TP+TN+FP+FN)  {(TP+TN)/(TP+TN+FP+FN):.4f}  Fraction correct
Precision           TP/(TP+FP)         {TP/(TP+FP+1e-10):.4f}  Of predicted +, how many are +?
Recall (Sensitivity)TP/(TP+FN)         {TP/(TP+FN+1e-10):.4f}  Of actual +, how many found?
Specificity         TN/(TN+FP)         {TN/(TN+FP+1e-10):.4f}  Of actual -, how many found?
F1 Score            2×P×R/(P+R)        {2*TP/(2*TP+FP+FN+1e-10):.4f}  Harmonic mean of P and R
False Positive Rate FP/(TN+FP)         {FP/(TN+FP+1e-10):.4f}  Fraction of neg predicted as +
False Negative Rate FN/(TP+FN)         {FN/(TP+FN+1e-10):.4f}  Fraction of pos predicted as -
PPV (Precision)     TP/(TP+FP)         {TP/(TP+FP+1e-10):.4f}  Same as precision
NPV                 TN/(TN+FN)         {TN/(TN+FN+1e-10):.4f}  Negative predictive value
""")

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=cancer.target_names)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('Confusion Matrix — Breast Cancer\n'
             '(Benign=0, Malignant=1)', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_confusion_matrix.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 4.2 Precision, Recall, F1

```python
import numpy as np
import matplotlib.pyplot as plt

print("""
PRECISION vs RECALL — THE FUNDAMENTAL TRADEOFF
════════════════════════════════════════════════════════════════════

PRECISION: "Of everything I labeled positive, how many were actually positive?"
  → Minimizes false positives
  → Critical when: false positives are costly
  → Examples: spam filter (don't want to flag real emails), 
               recommender (don't recommend irrelevant items)
  
RECALL: "Of all actual positives, how many did I find?"
  → Minimizes false negatives
  → Critical when: false negatives are costly
  → Examples: cancer screening (must catch all cancer), 
               fraud detection (can't miss fraud)
  
F1: Harmonic mean of precision and recall
  F1 = 2 × (P × R) / (P + R)
  → Use when both precision AND recall matter
  → Harmonic mean penalizes extreme imbalance
  
Fβ (general): Fβ = (1+β²) × (P × R) / (β²P + R)
  β < 1: weight precision more
  β > 1: weight recall more
  β = 2 (F2): recall twice as important as precision (medical)
  
NEVER use accuracy alone on imbalanced datasets:
  If 99% of transactions are legitimate, a model that always
  predicts "not fraud" has 99% accuracy but is completely useless.
════════════════════════════════════════════════════════════════════
""")

# Visualize the precision-recall tradeoff
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# F1 as a function of P and R
P_grid = np.linspace(0.01, 1, 100)
R_grid = np.linspace(0.01, 1, 100)
PP, RR = np.meshgrid(P_grid, R_grid)
F1_grid = 2 * PP * RR / (PP + RR + 1e-10)

ax = axes[0]
cf = ax.contourf(PP, RR, F1_grid, levels=20, cmap='RdYlGn')
ax.contour(PP, RR, F1_grid, levels=[0.5, 0.6, 0.7, 0.8, 0.9], colors='white',
           linewidths=1, alpha=0.5)
plt.colorbar(cf, ax=ax, label='F1 Score')
ax.set_xlabel('Precision'); ax.set_ylabel('Recall')
ax.set_title('F1 Score as Function of Precision and Recall\n'
             'High F1 requires BOTH high precision AND recall')

# Show specific points
points = [(0.9, 0.5), (0.5, 0.9), (0.8, 0.8), (0.6, 0.6)]
for p, r in points:
    f1 = 2*p*r/(p+r)
    ax.plot(p, r, 'w*', markersize=12, zorder=5)
    ax.annotate(f'F1={f1:.2f}', (p, r), textcoords='offset points',
                xytext=(8, 5), color='white', fontsize=9, fontweight='bold')

# Fβ for different β values
ax2 = axes[1]
beta_vals = [0.5, 1.0, 2.0, 5.0]
colors_b  = ['steelblue', 'green', 'tomato', 'purple']
for beta, color in zip(beta_vals, colors_b):
    # Fix R=0.7, vary P
    P_range = np.linspace(0.01, 1, 200)
    R_fixed = 0.7
    Fb = (1+beta**2) * P_range * R_fixed / (beta**2 * P_range + R_fixed + 1e-10)
    ax2.plot(P_range, Fb, color=color, lw=2, label=f'F{beta} (R=0.7 fixed)')
ax2.axvline(0.7, color='gray', lw=1.5, linestyle='--', label='P=R=0.7')
ax2.set_xlabel('Precision (Recall fixed at 0.7)')
ax2.set_ylabel('Fβ Score')
ax2.set_title('Fβ Scores — Weighting Precision vs Recall\n'
              'High β → rewards high recall (critical applications)')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/06_f1_scores.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 4.3 ROC Curve and AUC

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

models = {
    'Logistic Regression':   LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Random Forest':          RandomForestClassifier(n_estimators=100, random_state=42),
    'Random Classifier':      None,  # Baseline
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for name, model in models.items():
    if model is None:
        ax.plot([0,1], [0,1], 'k--', lw=1.5, label='Random (AUC=0.50)')
        continue
    model.fit(X_tr_s, y_tr)
    proba = model.predict_proba(X_te_s)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_te, proba)
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC={auc_val:.4f})')

ax.fill_between([0,1], [0,1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
ax.set_title('ROC Curves\n(Higher AUC = better, max=1.0, random=0.5)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Understanding what AUC means ─────────────────────────────────────────────
print("""
AUC (Area Under ROC Curve) — What It Really Means:
════════════════════════════════════════════════════════════
AUC = P(score of random positive > score of random negative)
  = probability of correctly ranking a random positive-negative pair
  
AUC = 1.0: Perfect — all positives ranked above all negatives
AUC = 0.5: Random — no discrimination ability
AUC < 0.5: Worse than random (invert predictions → AUC > 0.5)

AUC is:
  ✓ Threshold-independent (evaluates the full score distribution)
  ✓ Invariant to class imbalance (unlike accuracy)
  ✓ Widely used in medicine, credit scoring, information retrieval
  ✗ Can be misleading with severe class imbalance (use PR-AUC instead)
════════════════════════════════════════════════════════════
""")

# ROC at different threshold values
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_tr_s, y_tr)
proba_lr = lr_model.predict_proba(X_te_s)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_te, proba_lr)

ax2 = axes[1]
sc = ax2.scatter(fpr_lr[1:], tpr_lr[1:], c=thresholds_lr[1:],
                 cmap='RdYlGn_r', s=20, zorder=5)
plt.colorbar(sc, ax=ax2, label='Decision threshold')
ax2.plot(fpr_lr, tpr_lr, 'steelblue', lw=1.5, alpha=0.5)
ax2.plot([0,1], [0,1], 'k--', lw=1)

# Mark the default threshold (0.5)
idx_05 = np.argmin(np.abs(thresholds_lr - 0.5))
ax2.scatter([fpr_lr[idx_05]], [tpr_lr[idx_05]], s=200, color='black',
            marker='*', zorder=10, label=f'Threshold=0.5')

# Mark the optimal threshold (Youden's J: max TPR - FPR)
j_scores = tpr_lr - fpr_lr
idx_opt  = np.argmax(j_scores)
ax2.scatter([fpr_lr[idx_opt]], [tpr_lr[idx_opt]], s=200, color='gold',
            marker='*', zorder=10,
            label=f'Optimal threshold={thresholds_lr[idx_opt]:.3f}')
ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
ax2.set_title('ROC with Threshold Values\nGold star = Youden\'s J optimal threshold')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/06_roc_curves.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 4.4 Precision-Recall Curve

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# PR curve
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_tr_s, y_tr)
proba_lr = lr_model.predict_proba(X_te_s)[:, 1]

precision, recall, thresholds_pr = precision_recall_curve(y_te, proba_lr)
ap = average_precision_score(y_te, proba_lr)

ax = axes[0]
ax.step(recall, precision, where='post', color='steelblue', lw=2.5,
        label=f'LR (AP={ap:.4f})')
ax.axhline(y_te.mean(), color='tomato', lw=2, linestyle='--',
           label=f'Random (AP={y_te.mean():.3f})')
ax.fill_between(recall, precision, alpha=0.2, color='steelblue', step='post')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve\n'
             '(Better than ROC for imbalanced datasets)')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

# Precision and Recall vs Threshold
ax2 = axes[1]
ax2.plot(thresholds_pr, precision[:-1], color='steelblue', lw=2.5, label='Precision')
ax2.plot(thresholds_pr, recall[:-1],    color='tomato',    lw=2.5, label='Recall')
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
ax2.plot(thresholds_pr, f1_scores,      color='green',     lw=2.5, label='F1')
best_thresh = thresholds_pr[np.argmax(f1_scores)]
ax2.axvline(best_thresh, color='black', lw=1.5, linestyle='--',
            label=f'Best F1 threshold = {best_thresh:.3f}')
ax2.set_xlabel('Decision threshold'); ax2.set_ylabel('Score')
ax2.set_title('Precision, Recall, F1 vs Threshold\n'
              'Choose threshold based on application cost')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/06_pr_curve.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 4.5 Threshold Selection

```python
import numpy as np

print("""
THRESHOLD SELECTION — BUSINESS CONTEXT MATTERS
════════════════════════════════════════════════════════════════

Default threshold = 0.5 is almost never optimal.
Choose your threshold based on the COST of each error type.

Application        FP cost    FN cost    Recommendation
─────────────────────────────────────────────────────────────
Spam filter        High       Low        Lower threshold (miss less)
Cancer screening   Low        Very High  Lower threshold (catch all cancer)
Fraud detection    Medium     High       Lower threshold (catch more fraud)
Content moderation Medium     Medium     0.5 default, tune with A/B test
Legal decisions    High       High       High threshold (burden of proof)
─────────────────────────────────────────────────────────────

Methods for threshold selection:
  1. Youden's J:     max(TPR - FPR)  — balances sensitivity/specificity
  2. Max F1:         max(F1 score)   — balances precision/recall
  3. Cost-sensitive: min(cost_FP × FP + cost_FN × FN)
  4. Business rule:  "catch 95% of fraud" → find threshold giving recall=0.95
════════════════════════════════════════════════════════════════
""")

# Cost-sensitive threshold selection
proba_lr_all = lr_model.predict_proba(X_te_s)[:, 1]
thresholds   = np.linspace(0.01, 0.99, 200)

# Scenario: cancer screening
# FN (missed cancer) costs 10× more than FP (unnecessary follow-up)
cost_FP, cost_FN = 1, 10

costs = []
for t in thresholds:
    y_pred_t = (proba_lr_all >= t).astype(int)
    TP = ((y_pred_t == 1) & (y_te == 1)).sum()
    TN = ((y_pred_t == 0) & (y_te == 0)).sum()
    FP = ((y_pred_t == 1) & (y_te == 0)).sum()
    FN = ((y_pred_t == 0) & (y_te == 1)).sum()
    total_cost = cost_FP * FP + cost_FN * FN
    costs.append(total_cost)

optimal_threshold = thresholds[np.argmin(costs)]
print(f"Optimal threshold (FN cost = 10× FP cost): {optimal_threshold:.3f}")
print(f"Default threshold (0.5) cost:              {costs[np.argmin(np.abs(thresholds-0.5))]}")
print(f"Optimal threshold cost:                    {min(costs)}")
```

---

## 5. Regularization in Logistic Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

# Sklearn's LogisticRegression uses C = 1/λ (INVERSE of regularization strength)
# Large C = less regularization; Small C = more regularization
C_values = np.logspace(-4, 4, 50)

results = {'l1': {'train': [], 'val': [], 'n_zeros': []},
           'l2': {'train': [], 'val': [], 'n_zeros': []}}

for C in C_values:
    for penalty in ['l1', 'l2']:
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        lr = LogisticRegression(C=C, penalty=penalty, solver=solver,
                                max_iter=2000, random_state=42)
        cv   = cross_val_score(lr, X_tr_s, y_tr, cv=5, scoring='roc_auc')
        lr.fit(X_tr_s, y_tr)
        train_auc = roc_auc_score(y_tr, lr.predict_proba(X_tr_s)[:,1])
        results[penalty]['train'].append(train_auc)
        results[penalty]['val'].append(cv.mean())
        results[penalty]['n_zeros'].append((np.abs(lr.coef_[0]) < 1e-6).sum())

from sklearn.metrics import roc_auc_score
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, penalty, color in zip(axes, ['l1', 'l2'], ['steelblue', 'tomato']):
    ax.semilogx(C_values, results[penalty]['train'], '--', color=color,
                lw=2, label='Train AUC')
    ax.semilogx(C_values, results[penalty]['val'],   '-',  color=color,
                lw=2.5, label='CV AUC')
    best_C  = C_values[np.argmax(results[penalty]['val'])]
    best_cv = max(results[penalty]['val'])
    ax.axvline(best_C, color='black', lw=1.5, linestyle='--',
               label=f'Best C={best_C:.4f}')
    ax.set_xlabel('C (1/regularization)'); ax.set_ylabel('AUC')
    ax.set_title(f'{penalty.upper()} Regularization\nBest C={best_C:.4f}, CV AUC={best_cv:.4f}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    if penalty == 'l1':
        ax_twin = ax.twinx()
        ax_twin.semilogx(C_values, results[penalty]['n_zeros'],
                         ':', color='green', lw=2, alpha=0.7)
        ax_twin.set_ylabel('# Zero Coefficients', color='green')
        ax_twin.tick_params(axis='y', colors='green')

plt.suptitle('Regularization Strength in Logistic Regression\n'
             '(C = 1/λ — smaller C = stronger regularization)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_regularization.png', dpi=140, bbox_inches='tight')
plt.show()

# Final model with best C
best_C_l2 = C_values[np.argmax(results['l2']['val'])]
lr_best = LogisticRegression(C=best_C_l2, penalty='l2', max_iter=2000, random_state=42)
lr_best.fit(X_tr_s, y_tr)
print(f"\nFinal Model (L2, C={best_C_l2:.4f}):")
print(f"  Test AUC: {roc_auc_score(y_te, lr_best.predict_proba(X_te_s)[:,1]):.4f}")
print(f"  Test Acc: {lr_best.score(X_te_s, y_te):.4f}")
```

---

## 6. Multi-class Classification

### 6.1 One-vs-Rest (OvR)

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

iris  = load_iris()
X, y  = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

# OvR: train 3 binary classifiers, pick the class with highest score
lr_ovr = LogisticRegression(multi_class='ovr', C=1.0, max_iter=1000, random_state=42)
lr_ovr.fit(X_tr_s, y_tr)

print("One-vs-Rest (OvR) Multiclass Logistic Regression:")
print(f"Coefficient matrix shape: {lr_ovr.coef_.shape}  (K × d = 3 classifiers × 4 features)")
print(f"\nTest accuracy: {lr_ovr.score(X_te_s, y_te):.4f}")
print("\nClassification Report:")
print(classification_report(y_te, lr_ovr.predict(X_te_s),
                            target_names=iris.target_names))
```

---

### 6.2 Softmax Regression

Softmax regression (multinomial logistic regression) extends logistic regression directly to $K$ classes.

$$P(y=k | \mathbf{x}; \mathbf{W}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$

Loss: **categorical cross-entropy** — generalization of BCE:

$$L(\mathbf{W}) = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \hat{p}_{ik}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data[:, :2], iris.target   # Use only 2 features for visualization

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# Softmax (multinomial) logistic regression
lr_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                C=1.0, max_iter=1000, random_state=42)
lr_softmax.fit(X_sc, y)

# ── Decision boundary visualization ──────────────────────────────────────────
x_min, x_max = X_sc[:,0].min()-0.5, X_sc[:,0].max()+0.5
y_min, y_max = X_sc[:,1].min()-0.5, X_sc[:,1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))
grid   = np.c_[xx.ravel(), yy.ravel()]
Z_prob = lr_softmax.predict_proba(grid)
Z_pred = lr_softmax.predict(grid).reshape(xx.shape)

colors = ['#EF9A9A', '#A5D6A7', '#90CAF9']
cmap   = plt.matplotlib.colors.ListedColormap(colors)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.contourf(xx, yy, Z_pred, alpha=0.5, cmap=cmap)
ax.contour( xx, yy, Z_pred, colors='black', linewidths=1, alpha=0.5)
for cls, color, name in zip([0,1,2], ['tomato','green','steelblue'], iris.target_names):
    mask = y == cls
    ax.scatter(X_sc[mask,0], X_sc[mask,1], s=40, color=color,
               edgecolors='k', linewidths=0.5, label=name, alpha=0.8)
ax.set_xlabel('Sepal length (scaled)'); ax.set_ylabel('Sepal width (scaled)')
ax.set_title('Softmax Regression — Decision Boundaries\n(3-class, 2 features)')
ax.legend(fontsize=9)

# Probability simplex (3-class probabilities)
ax2 = axes[1]
for cls, color, name in zip([0,1,2], ['tomato','green','steelblue'], iris.target_names):
    mask = y == cls
    probs = lr_softmax.predict_proba(X_sc[mask])
    ax2.scatter(probs[:,0], probs[:,1], s=30, color=color, alpha=0.6, label=name)
ax2.set_xlabel('P(setosa)'); ax2.set_ylabel('P(versicolor)')
ax2.set_title('Predicted Probability Space\n(P(virginica) = 1 - P(setosa) - P(versicolor))')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/06_softmax.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Softmax from scratch ──────────────────────────────────────────────────────
def softmax(Z):
    """Z: (n, K) matrix of logits"""
    Z_shifted = Z - Z.max(axis=1, keepdims=True)   # Numerical stability
    exp_Z     = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

# Test
Z_test = np.array([[2.0, 1.0, 0.5],
                   [0.1, 3.0, 0.2]])
probs  = softmax(Z_test)
print("Softmax outputs (each row sums to 1):")
print(probs.round(4))
print("Row sums:", probs.sum(axis=1))
```

---

## 7. Class Imbalance

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Generate imbalanced dataset: 95% negative, 5% positive
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=5,
    weights=[0.95, 0.05],   # 5% positive class
    random_state=42
)

print(f"Class distribution: {np.bincount(y)}")
print(f"Positive rate: {y.mean():.2%}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

# ── Strategy 1: Ignore imbalance (baseline) ───────────────────────────────────
lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(X_tr_s, y_tr)

# ── Strategy 2: Class weighting ───────────────────────────────────────────────
lr_weighted = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_weighted.fit(X_tr_s, y_tr)

# ── Strategy 3: SMOTE oversampling ────────────────────────────────────────────
try:
    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_s, y_tr)
    lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_smote.fit(X_tr_sm, y_tr_sm)
    smote_available = True
except ImportError:
    smote_available = False
    print("imblearn not installed; skipping SMOTE")

# ── Strategy 4: Threshold adjustment ─────────────────────────────────────────
lr_thresh = LogisticRegression(max_iter=1000, random_state=42)
lr_thresh.fit(X_tr_s, y_tr)

print("\nComparison of Imbalance Strategies:")
print(f"\n{'Strategy':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 78)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

strategies = [
    ('Baseline (no handling)', lr_base, X_te_s, 0.5),
    ('Class weighting',        lr_weighted, X_te_s, 0.5),
    ('Threshold=0.2',          lr_thresh, X_te_s, 0.2),
]
if smote_available:
    strategies.append(('SMOTE oversampling', lr_smote, X_te_s, 0.5))

for name, model, X_te_eval, thresh in strategies:
    proba = model.predict_proba(X_te_eval)[:, 1]
    y_pred = (proba >= thresh).astype(int)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, proba)
    print(f"{name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {auc:>10.4f}")

print("""
\nKey takeaways for class imbalance:
  1. Accuracy is misleading on imbalanced data (baseline gets 95% by predicting all-negative)
  2. Use AUC-ROC and AUC-PR as primary metrics, plus recall for the minority class
  3. class_weight='balanced': simple, effective, no data modification
  4. SMOTE: creates synthetic minority samples — good for severe imbalance
  5. Threshold adjustment: tune to your application's precision/recall tradeoff
  6. Always use stratified splits (stratify=y) to maintain class ratios
""")
```

---

## 8. Interpreting Logistic Regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.stats

cancer   = load_breast_cancer()
X, y     = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_te_s   = scaler.transform(X_te)

lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
lr.fit(X_tr_s, y_tr)

# ── Coefficient interpretation ────────────────────────────────────────────────
coefs     = lr.coef_[0]
intercept = lr.intercept_[0]

# Bootstrapped confidence intervals
n_bootstrap = 200
boot_coefs  = np.zeros((n_bootstrap, len(coefs)))
rng = np.random.default_rng(42)

for i in range(n_bootstrap):
    idx = rng.choice(len(X_tr_s), len(X_tr_s), replace=True)
    Xb, yb = X_tr_s[idx], y_tr[idx]
    try:
        lr_b = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        lr_b.fit(Xb, yb)
        boot_coefs[i] = lr_b.coef_[0]
    except:
        boot_coefs[i] = coefs   # fallback

ci_low  = np.percentile(boot_coefs, 2.5,  axis=0)
ci_high = np.percentile(boot_coefs, 97.5, axis=0)
significant = (ci_low > 0) | (ci_high < 0)

# Create interpretation DataFrame
df_interp = pd.DataFrame({
    'feature':    cancer.feature_names,
    'coef':       coefs,
    'odds_ratio': np.exp(coefs),
    'ci_low':     ci_low,
    'ci_high':    ci_high,
    'ci_low_or':  np.exp(ci_low),
    'ci_high_or': np.exp(ci_high),
    'significant': significant
}).sort_values('coef', ascending=False)

print("Logistic Regression Coefficients (standardized features):")
print(f"\n{'Feature':<35} {'Coef':>8} {'OR':>8} {'95% CI':>20} {'Sig?':>6}")
print("-" * 82)
for _, row in df_interp.iterrows():
    ci_str = f"[{row['ci_low_or']:.3f}, {row['ci_high_or']:.3f}]"
    sig    = "***" if row['significant'] else ""
    print(f"{row['feature']:<35} {row['coef']:>+8.4f} {row['odds_ratio']:>8.3f} "
          f"{ci_str:>20} {sig:>6}")

# ── Coefficient plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 10))
df_plot = df_interp.copy()
y_pos   = range(len(df_plot))
colors  = ['tomato' if c > 0 else 'steelblue' for c in df_plot['coef']]

ax.barh(list(y_pos), df_plot['coef'], color=colors, alpha=0.8)
ax.errorbar(df_plot['coef'], list(y_pos),
            xerr=[df_plot['coef'] - df_plot['ci_low'],
                  df_plot['ci_high'] - df_plot['coef']],
            fmt='none', color='black', capsize=3, lw=1.5)
ax.axvline(0, color='black', lw=1.5)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(df_plot['feature'], fontsize=8)
ax.set_xlabel('Coefficient (standardized)\nPositive = increases P(malignant)')
ax.set_title('Logistic Regression Coefficients\nwith 95% Bootstrap CI')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/06_coefficients.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 9. Logistic Regression with Scikit-learn

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

cancer  = load_breast_cancer()
X, y    = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ── Option 1: Manual pipeline with GridSearchCV ───────────────────────────────
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(max_iter=2000, random_state=42))
])

param_grid = {
    'model__C':       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'model__penalty': ['l1', 'l2'],
    'model__solver':  ['liblinear'],  # Required for L1
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_tr, y_tr)

print(f"Grid Search Best params: {gs.best_params_}")
print(f"CV AUC:                  {gs.best_score_:.4f}")
print(f"Test AUC:                {roc_auc_score(y_te, gs.predict_proba(X_te)[:,1]):.4f}")

# ── Option 2: LogisticRegressionCV (faster, built-in) ────────────────────────
pipe_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegressionCV(Cs=10, cv=5, max_iter=2000,
                                    scoring='roc_auc', random_state=42, n_jobs=-1))
])
pipe_cv.fit(X_tr, y_tr)
best_C_cv = pipe_cv.named_steps['model'].C_[0]
print(f"\nLogisticRegressionCV Best C: {best_C_cv:.6f}")
print(f"Test AUC:                    {roc_auc_score(y_te, pipe_cv.predict_proba(X_te)[:,1]):.4f}")
print(f"Test Accuracy:               {pipe_cv.score(X_te, y_te):.4f}")
print("\nClassification Report:")
print(classification_report(y_te, pipe_cv.predict(X_te),
                            target_names=cancer.target_names))

# ── Key sklearn parameters ────────────────────────────────────────────────────
print("""
sklearn LogisticRegression Key Parameters:
─────────────────────────────────────────────────────────────────────
C             : 1/λ (inverse regularization). Default=1.0.
                Large C = weak regularization (can overfit).
                Small C = strong regularization (can underfit).

penalty       : 'l2' (default), 'l1', 'elasticnet', None
                Use 'l1' for feature selection
                Use 'elasticnet' + l1_ratio for both

solver        : Algorithm for optimization
                'lbfgs'     : Default. Good for L2, multinomial. No L1.
                'liblinear' : Good for small datasets, supports L1.
                'saga'      : Best for large datasets, all penalties.
                'newton-cg' : Fast for L2 but no L1.

max_iter      : Maximum iterations. Increase if ConvergenceWarning.
multi_class   : 'auto', 'ovr', 'multinomial'
class_weight  : None, 'balanced', or dict {0: w0, 1: w1}
random_state  : For reproducibility (especially with solver='saga')
─────────────────────────────────────────────────────────────────────
""")
```

---

## 10. Case Study: Breast Cancer Diagnosis

```python
# =============================================================================
# Full Pipeline: EDA → Feature Selection → Model → Calibration → Report
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load and split ─────────────────────────────────────────────────────────
cancer   = load_breast_cancer()
X, y     = cancer.data, cancer.target
feature_names = cancer.feature_names

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
print(f"Train positive rate: {y_tr.mean():.2%}, Test: {y_te.mean():.2%}")

# ── 2. Pipeline with feature selection ───────────────────────────────────────
pipe = Pipeline([
    ('scaler',    StandardScaler()),
    ('select',    SelectFromModel(
        LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000),
        prefit=False
    )),
    ('model',     LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 20), cv=5, max_iter=2000,
        scoring='roc_auc', random_state=42, n_jobs=-1
    ))
])

pipe.fit(X_tr, y_tr)

# Selected features
selected_mask  = pipe.named_steps['select'].get_support()
selected_feats = feature_names[selected_mask]
print(f"\nFeature selection: {selected_mask.sum()} / {len(feature_names)} features selected")
print(f"Selected: {list(selected_feats)}")

# ── 3. Evaluation ─────────────────────────────────────────────────────────────
y_pred   = pipe.predict(X_te)
y_proba  = pipe.predict_proba(X_te)[:, 1]

print("\n" + "="*55)
print("FINAL MODEL EVALUATION")
print("="*55)
print(f"  Test Accuracy:   {(y_pred == y_te).mean():.4f}")
print(f"  Test AUC-ROC:    {roc_auc_score(y_te, y_proba):.4f}")
print(f"  Test AUC-PR:     {average_precision_score(y_te, y_proba):.4f}")
print(f"  Brier Score:     {brier_score_loss(y_te, y_proba):.4f}  (lower is better, 0=perfect)")
print(f"\n  Best C:          {pipe.named_steps['model'].C_[0]:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=cancer.target_names))

# ── 4. Comprehensive visualization ────────────────────────────────────────────
from sklearn.metrics import roc_curve, precision_recall_curve

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                               display_labels=cancer.target_names)
disp.plot(ax=axes[0,0], colorbar=False, cmap='Blues')
axes[0,0].set_title('Confusion Matrix')

# ROC curve
fpr, tpr, _ = roc_curve(y_te, y_proba)
axes[0,1].plot(fpr, tpr, color='steelblue', lw=2.5,
               label=f'AUC = {roc_auc_score(y_te, y_proba):.4f}')
axes[0,1].plot([0,1],[0,1],'k--',lw=1)
axes[0,1].set_xlabel('FPR'); axes[0,1].set_ylabel('TPR')
axes[0,1].set_title('ROC Curve'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_te, y_proba)
ap = average_precision_score(y_te, y_proba)
axes[0,2].step(rec, prec, color='tomato', lw=2.5, where='post', label=f'AP = {ap:.4f}')
axes[0,2].set_xlabel('Recall'); axes[0,2].set_ylabel('Precision')
axes[0,2].set_title('Precision-Recall Curve'); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

# Coefficient plot (selected features)
coefs = pipe.named_steps['model'].coef_[0]
sort_idx = np.argsort(np.abs(coefs))[::-1]
colors_c = ['tomato' if c > 0 else 'steelblue' for c in coefs[sort_idx]]
axes[1,0].barh(range(len(coefs)), coefs[sort_idx], color=colors_c, alpha=0.8)
axes[1,0].set_yticks(range(len(coefs)))
axes[1,0].set_yticklabels([selected_feats[i] for i in sort_idx], fontsize=7)
axes[1,0].axvline(0, color='black', lw=1)
axes[1,0].set_title('Coefficients (selected features)')
axes[1,0].grid(axis='x', alpha=0.3)

# Learning curve
train_sizes, train_sc, val_sc = learning_curve(
    pipe, X_tr, y_tr, train_sizes=np.linspace(0.1,1,8),
    cv=5, scoring='roc_auc', n_jobs=-1
)
axes[1,1].plot(train_sizes, train_sc.mean(1), 'o-', color='steelblue', label='Train')
axes[1,1].fill_between(train_sizes, train_sc.mean(1)-train_sc.std(1),
                        train_sc.mean(1)+train_sc.std(1), alpha=0.2, color='steelblue')
axes[1,1].plot(train_sizes, val_sc.mean(1),   's-', color='tomato', label='Validation')
axes[1,1].fill_between(train_sizes, val_sc.mean(1)-val_sc.std(1),
                        val_sc.mean(1)+val_sc.std(1), alpha=0.2, color='tomato')
axes[1,1].set_xlabel('Training set size'); axes[1,1].set_ylabel('AUC')
axes[1,1].set_title('Learning Curve')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

# Probability calibration
prob_true, prob_pred = np.histogram(y_proba, bins=10, density=False)
bin_centers = np.linspace(0.05, 0.95, 10)
mean_proba_bins = [y_proba[(y_proba >= l) & (y_proba < h)].mean()
                   for l, h in zip(np.linspace(0,0.9,10), np.linspace(0.1,1,10))]
true_frac_bins  = [y_te[(y_proba >= l) & (y_proba < h)].mean()
                   for l, h in zip(np.linspace(0,0.9,10), np.linspace(0.1,1,10))]
mean_proba_bins = [m for m in mean_proba_bins if not np.isnan(m)]
true_frac_bins  = [t for t in true_frac_bins  if not np.isnan(t)]
axes[1,2].plot([0,1],[0,1],'k--',lw=1.5, label='Perfect calibration')
axes[1,2].plot(mean_proba_bins, true_frac_bins, 's-', color='tomato', lw=2, label='LR calibration')
axes[1,2].set_xlabel('Mean predicted probability')
axes[1,2].set_ylabel('Fraction of positives')
axes[1,2].set_title('Calibration Curve\n(How reliable are the probabilities?)')
axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)

plt.suptitle('Breast Cancer Diagnosis — Full Model Evaluation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_case_study.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 11. Summary

### ✅ Must-Remember Mental Models

**1. Why logistic regression, not linear regression for classification:**
- Linear regression predicts unbounded values ($< 0$ or $> 1$) — invalid as probabilities.
- Linear regression is sensitive to outliers that distort the decision boundary.
- Logistic regression: $\hat{p} = \sigma(\mathbf{w}^T\mathbf{x}) \in (0, 1)$ always.

**2. The sigmoid and its derivative:**
$$\sigma(z) = \frac{1}{1+e^{-z}}, \quad \sigma'(z) = \sigma(z)(1-\sigma(z))$$
Max gradient = 0.25 at $z=0$. This causes vanishing gradients in deep networks.

**3. The gradient has a beautiful form:**
$$\nabla_\mathbf{w} L_\text{BCE} = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})$$
Identical in structure to the MSE gradient. Not a coincidence — it's by design.

**4. The confusion matrix — four cells you must know:**
TP (correctly flagged positive), TN (correctly dismissed), FP (false alarm), FN (missed positive).

**5. Which metric to use when:**
- Balanced classes: accuracy, F1
- Imbalanced (rare positives): AUC-PR, recall
- Ranking/scoring: AUC-ROC
- Medical screening: recall (sensitivity) — never miss a positive
- Spam filter: precision — don't flag real emails

**6. Decision threshold is a design choice, not a fixed constant.**
0.5 is almost never optimal. Choose based on cost of FP vs FN.

**7. Regularization in sklearn uses C = 1/λ.**
Larger C = weaker regularization. Always tune C via cross-validation.

**8. Softmax = multinomial logistic regression.**
Generalization to K classes. Loss = categorical cross-entropy. Output sums to 1.

**9. Class imbalance — your first tools:**
`class_weight='balanced'` and threshold adjustment. SMOTE for severe imbalance.

**10. Calibration matters for probabilities.**
A model with 97% AUC can still output badly calibrated probabilities. Always check the calibration curve if probabilities matter (medical, financial).

---

## 12. Exercises

1. **Prove the gradient simplification**: Starting from the binary cross-entropy loss and the chain rule, show explicitly that $\nabla_\mathbf{w} L = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})$.

2. **Implement logistic regression with momentum** (SGD + momentum). Compare convergence speed against plain gradient descent on the breast cancer dataset.

3. **ROC vs PR curve**: Create a severely imbalanced dataset (1% positive). Train logistic regression. Plot both ROC and PR curves. Show that ROC AUC looks great (0.95) while PR AUC reveals the true difficulty.

4. **Threshold optimization**: For the breast cancer dataset, implement a cost-sensitive threshold selection function that minimizes total cost given configurable FP and FN costs. Show how the optimal threshold changes as you vary the cost ratio from 1:1 to 1:20.

5. **Feature importance via permutation**: For the trained breast cancer logistic regression, implement permutation importance: randomly shuffle each feature one at a time, measure the drop in AUC, and rank features by importance. Compare to the absolute coefficient values.

6. **Calibration**: Train a logistic regression on breast cancer. Calibrate it using `sklearn.calibration.CalibratedClassifierCV` with Platt scaling and isotonic regression. Compare Brier scores before and after calibration.

7. **Challenge — Multinomial from scratch**: Implement softmax regression from scratch (NumPy only). Your implementation should support:
   - Softmax activation
   - Categorical cross-entropy loss  
   - Mini-batch gradient descent
   - L2 regularization
   Test on the Iris dataset and verify against `sklearn.linear_model.LogisticRegression(multi_class='multinomial')`.

---

## 13. References

- Cox, D. R. (1958). The regression analysis of binary sequences. *JRSS-B*, 20(2), 215–232. — Original logistic regression paper.
- Harrell, F. E. (2015). *Regression Modeling Strategies* (2nd ed.). Springer. — Definitive reference for logistic regression in biostatistics.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.
- Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities With Supervised Learning. *ICML*. — On calibration.

---

*Previous: [Chapter 5 — Linear Regression](05_linear_regression.md)*  
*Next: [Chapter 7 — Decision Trees](07_decision_trees.md)*

*In Chapter 7, we build decision trees from scratch — learning the mathematics of impurity (Gini, entropy), how the CART algorithm works, why trees overfit and how to prune them, and when they outperform linear models.*

---

> **Chapter 6 complete.** All code is in `notebooks/06_logistic_regression.ipynb`.
