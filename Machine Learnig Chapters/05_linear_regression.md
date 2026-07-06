# Chapter 5: Linear Regression — Predicting Continuous Values

> *"All models are wrong, but some are useful."*  
> — George Box

> *"The fact that linear regression still works so well in practice is not an accident — it works because the world often is approximately linear, and because regularization tames the cases where it isn't."*

---

## Table of Contents

1. [The Problem: Predicting Continuous Values](#1-the-problem-predicting-continuous-values)
2. [Simple Linear Regression](#2-simple-linear-regression)
   - 2.1 [The Model](#21-the-model)
   - 2.2 [Deriving the OLS Solution](#22-deriving-the-ols-solution)
   - 2.3 [Geometric Interpretation](#23-geometric-interpretation)
3. [Multiple Linear Regression](#3-multiple-linear-regression)
   - 3.1 [Matrix Formulation](#31-matrix-formulation)
   - 3.2 [Normal Equations](#32-normal-equations)
   - 3.3 [Gradient Descent Solution](#33-gradient-descent-solution)
4. [The Assumptions of Linear Regression](#4-the-assumptions-of-linear-regression)
5. [Evaluating a Regression Model](#5-evaluating-a-regression-model)
6. [Regularization](#6-regularization)
   - 6.1 [The Overfitting Problem](#61-the-overfitting-problem)
   - 6.2 [Ridge Regression (L2)](#62-ridge-regression-l2)
   - 6.3 [Lasso Regression (L1)](#63-lasso-regression-l1)
   - 6.4 [ElasticNet — Best of Both](#64-elasticnet--best-of-both)
   - 6.5 [Choosing Regularization Strength](#65-choosing-regularization-strength)
7. [Polynomial Regression](#7-polynomial-regression)
8. [Residual Diagnostics](#8-residual-diagnostics)
9. [Linear Regression with Scikit-learn](#9-linear-regression-with-scikit-learn)
10. [Case Study: Predicting California House Prices](#10-case-study-predicting-california-house-prices)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. The Problem: Predicting Continuous Values

**Regression** is the task of predicting a continuous output $y \in \mathbb{R}$ from an input $\mathbf{x} \in \mathbb{R}^d$.

Real-world regression problems:
- Predicting house price from square footage, location, age
- Predicting patient blood glucose level from clinical measurements
- Predicting product demand from price, marketing spend, seasonality
- Predicting stock return from financial ratios
- Predicting energy consumption from temperature, time of day, occupancy

Linear regression makes one assumption: **the target is (approximately) a linear function of the inputs.**

$$y \approx w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_d x_d$$

This is both its strength (interpretable, efficient, regularizable) and its limitation (can't capture non-linear relationships without feature engineering).

---

## 2. Simple Linear Regression

### 2.1 The Model

With one feature, the model is:

$$\hat{y} = w_0 + w_1 x$$

Where:
- $w_0$: **intercept** (bias) — value of $\hat{y}$ when $x = 0$
- $w_1$: **slope** — change in $\hat{y}$ per unit change in $x$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Generate data from a true linear relationship with noise ──────────────────
np.random.seed(42)
n = 80

x_true = np.linspace(0, 10, n)
w0_true, w1_true = 2.0, 1.5   # intercept=2, slope=1.5
noise  = np.random.normal(0, 1.5, n)
y      = w0_true + w1_true * x_true + noise

# ── Fit simple linear regression ──────────────────────────────────────────────
# scipy.stats.linregress gives us: slope, intercept, r, p, se
result = stats.linregress(x_true, y)
w1_hat = result.slope
w0_hat = result.intercept
r_sq   = result.rvalue**2

print(f"True:      w0={w0_true:.2f}, w1={w1_true:.2f}")
print(f"Estimated: w0={w0_hat:.4f}, w1={w1_hat:.4f}")
print(f"R² = {r_sq:.4f}")

# ── Full visualization ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
x_line = np.linspace(-0.5, 10.5, 200)
y_hat  = w0_hat + w1_hat * x_line

ax.scatter(x_true, y, alpha=0.6, color='steelblue', s=40, label='Observations', zorder=3)
ax.plot(x_line, y_hat, color='tomato', lw=2.5, label=f'Fit: ŷ = {w0_hat:.2f} + {w1_hat:.2f}x')
ax.plot(x_line, w0_true + w1_true * x_line, 'k--', lw=1.5, alpha=0.5, label='True line')

# Draw residuals for a few points
for i in [5, 20, 40, 60, 75]:
    y_pred_i = w0_hat + w1_hat * x_true[i]
    ax.plot([x_true[i], x_true[i]], [y[i], y_pred_i],
            color='gray', lw=1, linestyle='-', alpha=0.7)

# Annotate slope and intercept
ax.annotate('w₀ (intercept)', xy=(0, w0_hat), xytext=(0.5, w0_hat - 2.5),
            arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=10)
ax.annotate(f'slope w₁ = {w1_hat:.2f}', xy=(5, w0_hat + w1_hat*5),
            xytext=(6.5, w0_hat + w1_hat*5 - 3),
            arrowprops=dict(arrowstyle='->', color='purple'), color='purple', fontsize=10)

ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title(f'Simple Linear Regression\nR² = {r_sq:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Residual plot
ax2 = axes[1]
y_pred_all = w0_hat + w1_hat * x_true
residuals  = y - y_pred_all
ax2.scatter(y_pred_all, residuals, alpha=0.6, color='steelblue', s=40)
ax2.axhline(0, color='tomato', lw=2, linestyle='--')
ax2.set_xlabel('Fitted values ŷ'); ax2.set_ylabel('Residuals e = y - ŷ')
ax2.set_title('Residual Plot\n(Should be random around 0 — no pattern)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/05_simple_regression.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 2.2 Deriving the OLS Solution

We want to minimize the **Sum of Squared Residuals (SSR)**:

$$L(w_0, w_1) = \sum_{i=1}^n (y^{(i)} - w_0 - w_1 x^{(i)})^2$$

Taking partial derivatives and setting to zero:

$$\frac{\partial L}{\partial w_0} = -2\sum_i (y^{(i)} - w_0 - w_1 x^{(i)}) = 0$$

$$\frac{\partial L}{\partial w_1} = -2\sum_i x^{(i)}(y^{(i)} - w_0 - w_1 x^{(i)}) = 0$$

Solving this system of equations:

$$\boxed{w_1 = \frac{\sum_i (x^{(i)} - \bar{x})(y^{(i)} - \bar{y})}{\sum_i (x^{(i)} - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}}$$

$$\boxed{w_0 = \bar{y} - w_1 \bar{x}}$$

```python
import numpy as np

def simple_ols(x, y):
    """
    Closed-form OLS solution for simple linear regression.
    Derives w1 = Cov(x,y)/Var(x),  w0 = y_bar - w1 * x_bar
    """
    x_bar = x.mean()
    y_bar = y.mean()

    # Numerator: Σ(xi - x̄)(yi - ȳ) = covariance (unnormalized)
    cov_xy = np.sum((x - x_bar) * (y - y_bar))

    # Denominator: Σ(xi - x̄)² = variance (unnormalized)
    var_x  = np.sum((x - x_bar)**2)

    w1 = cov_xy / var_x
    w0 = y_bar - w1 * x_bar

    return w0, w1

# Verify against scipy
w0, w1 = simple_ols(x_true, y)
print(f"Manual OLS: w0={w0:.6f}, w1={w1:.6f}")
print(f"scipy:      w0={w0_hat:.6f}, w1={w1_hat:.6f}")
print(f"Match: {np.allclose([w0, w1], [w0_hat, w1_hat])}")

# ── Connection to Pearson correlation ─────────────────────────────────────────
# w1 = Cov(x,y) / Var(x)
#    = [Cov(x,y) / (σx σy)] * [σy / σx]
#    = r * (σy / σx)
r  = np.corrcoef(x_true, y)[0, 1]
w1_via_r = r * (y.std() / x_true.std())
print(f"\nw1 via Pearson r: {w1_via_r:.6f}  (match: {np.isclose(w1, w1_via_r)})")
print(f"Note: slope = correlation × (std_y / std_x)")
print(f"  r = {r:.4f},  std_y/std_x = {y.std()/x_true.std():.4f}")
```

---

### 2.3 Geometric Interpretation

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Loss surface: L(w0, w1) as a 2D bowl ─────────────────────────────────────
w0_vals = np.linspace(-2, 6, 80)
w1_vals = np.linspace(0.5, 2.5, 80)
W0, W1  = np.meshgrid(w0_vals, w1_vals)

# Compute loss for each (w0, w1) combination
L = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        y_pred = W0[i,j] + W1[i,j] * x_true
        L[i,j] = ((y - y_pred)**2).mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Contour plot
ax = axes[0]
contour = ax.contourf(W0, W1, L, levels=40, cmap='viridis', alpha=0.9)
plt.colorbar(contour, ax=ax, label='MSE Loss')
ax.plot(w0, w1, 'r*', markersize=16, zorder=5, label=f'OLS minimum\n({w0:.2f}, {w1:.2f})')
ax.set_xlabel('w₀ (intercept)'); ax.set_ylabel('w₁ (slope)')
ax.set_title('MSE Loss Surface — Simple Linear Regression\n(Convex bowl → unique global minimum)')
ax.legend()

# 1D slice: loss vs w1 with w0 fixed at optimal
ax2 = axes[1]
w1_range = np.linspace(0.5, 2.5, 200)
L_slice  = [((y - (w0 + w1_i * x_true))**2).mean() for w1_i in w1_range]
ax2.plot(w1_range, L_slice, color='steelblue', lw=2.5)
ax2.axvline(w1, color='tomato', lw=2, linestyle='--', label=f'OLS w₁={w1:.3f}')
ax2.set_xlabel('w₁ (slope)'); ax2.set_ylabel('MSE Loss')
ax2.set_title('Loss vs. w₁ (w₀ fixed at optimum)\n(Parabola — exactly one minimum)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/05_loss_surface.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 3. Multiple Linear Regression

### 3.1 Matrix Formulation

With $d$ features and $n$ samples, we write compactly:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$$

Where:
- $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$ — design matrix with a leading column of ones for the intercept
- $\mathbf{w} \in \mathbb{R}^{d+1}$ — weight vector $[w_0, w_1, \ldots, w_d]^T$
- $\hat{\mathbf{y}} \in \mathbb{R}^n$ — predictions

The MSE loss in matrix form:

$$L(\mathbf{w}) = \frac{1}{n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 = \frac{1}{n}(\mathbf{X}\mathbf{w} - \mathbf{y})^T(\mathbf{X}\mathbf{w} - \mathbf{y})$$

---

### 3.2 Normal Equations

Setting $\nabla_\mathbf{w} L = 0$:

$$\frac{2}{n}\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y}) = 0$$
$$\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}$$
$$\boxed{\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# Use 3 features for illustration
features = ['MedInc', 'HouseAge', 'AveRooms']
X_raw = df[features].values
y     = df['MedHouseVal'].values

# Add bias column (column of ones)
X = np.hstack([np.ones((len(X_raw), 1)), X_raw])

# ── Normal equations ──────────────────────────────────────────────────────────
XtX   = X.T @ X                           # (4, 4)
Xty   = X.T @ y                           # (4,)
w_ols = np.linalg.solve(XtX, Xty)         # Never use inv() — use solve()

print("Normal Equations — Multiple Linear Regression")
print(f"Design matrix shape: {X.shape}")
print(f"\nEstimated weights:")
print(f"  w₀ (intercept): {w_ols[0]:.4f}")
for feat, wi in zip(features, w_ols[1:]):
    print(f"  w_{feat:<12}: {wi:.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
y_hat = X @ w_ols
residuals = y - y_hat
mse   = (residuals**2).mean()
rmse  = np.sqrt(mse)
mae   = np.abs(residuals).mean()
ss_res = (residuals**2).sum()
ss_tot = ((y - y.mean())**2).sum()
r_sq  = 1 - ss_res / ss_tot

print(f"\nTraining metrics:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}  (same units as y)")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r_sq:.4f}  ({r_sq*100:.1f}% of variance explained)")

# ── Why we avoid explicit matrix inversion ───────────────────────────────────
# np.linalg.solve uses LU decomposition: O(d³) but numerically stable
# np.linalg.inv(XtX) @ Xty: also O(d³) but less stable due to intermediate inversion

# Demonstrate on an ill-conditioned system:
# If features are highly correlated, X^T X is nearly singular
rng = np.random.default_rng(42)
n, d = 100, 50
X_illcond  = rng.standard_normal((n, d))
X_illcond  = np.hstack([X_illcond, X_illcond[:, :10] + 1e-6 * rng.standard_normal((n, 10))])
# These last 10 columns are nearly identical to first 10 → ill-conditioned

print(f"\nCondition number of X^TX:")
print(f"  Well-conditioned: ~{np.linalg.cond(X.T @ X):.0f}")
print(f"  Ill-conditioned:  ~{np.linalg.cond(X_illcond.T @ X_illcond):.2e}")
print("  High condition number → solve() still works, inv() may give garbage")
```

---

### 3.3 Gradient Descent Solution

```python
import numpy as np
import matplotlib.pyplot as plt

def linear_regression_gd(X, y, lr=0.01, n_epochs=500, batch_size=None, seed=42):
    """
    Linear regression via gradient descent.
    Supports: Batch GD (batch_size=None), Mini-batch SGD (batch_size=int).

    Parameters
    ----------
    X          : (n, d) feature matrix WITH bias column
    y          : (n,) target vector
    lr         : learning rate
    n_epochs   : number of epochs
    batch_size : None for batch GD, int for mini-batch
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w    = rng.standard_normal(d) * 0.01   # Small random init

    history = {'loss': [], 'w': [w.copy()]}

    for epoch in range(n_epochs):
        if batch_size is None:
            # Full batch gradient
            grad = (2/n) * X.T @ (X @ w - y)
            w    = w - lr * grad
        else:
            # Mini-batch
            indices = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx  = indices[start:start+batch_size]
                Xb, yb = X[idx], y[idx]
                grad = (2/len(yb)) * Xb.T @ (Xb @ w - yb)
                w    = w - lr * grad

        loss = ((X @ w - y)**2).mean()
        history['loss'].append(loss)
        history['w'].append(w.copy())

    return w, history

# Scale features first — critical for gradient descent
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

features = ['MedInc', 'HouseAge', 'AveRooms']
X_raw = df[features].values
y     = df['MedHouseVal'].values

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_raw)
X_sc    = np.hstack([np.ones((len(X_sc), 1)), X_sc])

# Run gradient descent
w_gd, hist_gd = linear_regression_gd(X_sc, y, lr=0.05, n_epochs=200)

# Compare with normal equations (on same scaled data)
w_ols_sc = np.linalg.solve(X_sc.T @ X_sc, X_sc.T @ y)

print(f"Normal equations (scaled): {w_ols_sc.round(4)}")
print(f"Gradient descent (scaled): {w_gd.round(4)}")
print(f"Match: {np.allclose(w_ols_sc, w_gd, atol=1e-2)}")

# Plot convergence
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(hist_gd['loss'], color='steelblue', lw=2, label='GD loss')
ax.axhline(((X_sc @ w_ols_sc - y)**2).mean(), color='tomato', lw=2,
           linestyle='--', label='OLS optimum')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.set_title('Gradient Descent Convergence\n(Feature scaling is essential!)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/05_gd_convergence.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 4. The Assumptions of Linear Regression

Linear regression has formal assumptions. When they hold, OLS is BLUE (Best Linear Unbiased Estimator — Gauss-Markov theorem). When they're violated, the model still predicts, but inference (p-values, confidence intervals) becomes unreliable.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

print("""
LINEAR REGRESSION ASSUMPTIONS (Gauss-Markov)
════════════════════════════════════════════════════════════════
1. LINEARITY
   y = Xw + ε — the relationship is linear in parameters
   How to check: residual vs fitted plot (should show no pattern)
   How to fix:   add polynomial features, log-transform, feature engineering

2. INDEPENDENCE
   Errors ε^(i) are independent of each other
   How to check: Durbin-Watson test (for time series: look for autocorrelation)
   How to fix:   use time-series models (ARIMA), add lag features

3. HOMOSCEDASTICITY
   Var(ε^(i)) = σ² (constant) — same variance for all errors
   How to check: scale-location plot (should be flat)
   How to fix:   log-transform y, use WLS (Weighted Least Squares)

4. NORMALITY OF RESIDUALS
   ε ~ N(0, σ²) — needed for valid hypothesis tests and CIs
   How to check: Q-Q plot of residuals (should follow diagonal)
   How to fix:   transform y, remove outliers (but carefully!)

5. NO MULTICOLLINEARITY
   Features are not perfectly correlated with each other
   How to check: VIF > 10 is problematic
   How to fix:   remove/combine correlated features, use Ridge regularization

6. NO INFLUENTIAL OUTLIERS
   No single observation drives the entire fit
   How to check: Cook's distance > 4/n is flagged
   How to fix:   investigate the outlier — is it an error? Remove or robust regression.
════════════════════════════════════════════════════════════════
""")

# ── Demonstrating assumption violations visually ──────────────────────────────
n = 200
x = np.linspace(0, 10, n)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# A) Assumptions met
y_good = 2 + 1.5*x + rng.normal(0, 1, n)
w0g, w1g = np.polyfit(x, y_good, 1)[::-1]
r_good = y_good - (w0g + w1g*x)
axes[0,0].scatter(w0g + w1g*x, r_good, s=8, alpha=0.5, color='green')
axes[0,0].axhline(0, color='tomato', lw=1.5)
axes[0,0].set_title('✓ Assumptions Met\n(Random scatter, no pattern)')
axes[0,0].set_xlabel('Fitted'); axes[0,0].set_ylabel('Residual')

# B) Non-linearity
y_nonlin = 2 + 1.5*x + 0.2*x**2 + rng.normal(0, 1, n)
w0n, w1n = np.polyfit(x, y_nonlin, 1)[::-1]
r_nonlin = y_nonlin - (w0n + w1n*x)
axes[0,1].scatter(w0n + w1n*x, r_nonlin, s=8, alpha=0.5, color='tomato')
axes[0,1].axhline(0, color='black', lw=1.5)
# Add smoother
from scipy.ndimage import uniform_filter1d
idx_sort = np.argsort(w0n + w1n*x)
smoothed = uniform_filter1d(r_nonlin[idx_sort], size=20)
axes[0,1].plot(sorted(w0n + w1n*x), smoothed, 'b-', lw=2, label='Trend')
axes[0,1].set_title('✗ Non-linearity\n(U-shaped residual pattern)')
axes[0,1].set_xlabel('Fitted'); axes[0,1].legend(fontsize=8)

# C) Heteroscedasticity
y_hetero = 2 + 1.5*x + rng.normal(0, 0.1 + 0.5*x, n)
w0h, w1h = np.polyfit(x, y_hetero, 1)[::-1]
r_hetero = y_hetero - (w0h + w1h*x)
axes[0,2].scatter(w0h + w1h*x, r_hetero, s=8, alpha=0.5, color='orange')
axes[0,2].axhline(0, color='black', lw=1.5)
axes[0,2].set_title('✗ Heteroscedasticity\n(Fan/funnel shape)')
axes[0,2].set_xlabel('Fitted')

# D) Non-normal residuals (heavy tails)
y_nonnorm = 2 + 1.5*x + rng.standard_t(df=2, size=n)
w0nn, w1nn = np.polyfit(x, y_nonnorm, 1)[::-1]
r_nonnorm  = y_nonnorm - (w0nn + w1nn*x)
axes[0,3].scatter(w0nn + w1nn*x, r_nonnorm, s=8, alpha=0.5, color='purple')
axes[0,3].axhline(0, color='black', lw=1.5)
axes[0,3].set_title('✗ Heavy-tailed Residuals\n(Outlier clusters)')
axes[0,3].set_xlabel('Fitted')

# Q-Q plots for all four
for ax, r, title, color in zip(
    axes[1,:],
    [r_good, r_nonlin, r_hetero, r_nonnorm],
    ['✓ Normal residuals', '✗ Non-linear (still normal)', '✗ Heteroscedastic', '✗ Heavy tails (t₂)'],
    ['green', 'tomato', 'orange', 'purple']
):
    (osm, osr), (slope, intercept, r_val) = stats.probplot(r)
    ax.scatter(osm, osr, s=5, alpha=0.4, color=color)
    ax.plot([osm[0], osm[-1]], [slope*osm[0]+intercept, slope*osm[-1]+intercept],
            'k-', lw=1.5)
    ax.set_title(f'Q-Q Plot: {title}')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    ax.grid(True, alpha=0.3)

plt.suptitle('Linear Regression Assumption Checks', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_assumption_checks.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 5. Evaluating a Regression Model

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

housing = fetch_california_housing()
X, y    = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# ── Regression metrics explained ──────────────────────────────────────────────
residuals = y_test - y_pred

mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)

# Adjusted R²: penalizes for number of features
n, d  = X_test.shape
r2_adj = 1 - (1 - r2) * (n - 1) / (n - d - 1)

# MAPE: Mean Absolute Percentage Error
mape  = np.mean(np.abs(residuals / (y_test + 1e-8))) * 100

# Median Absolute Error: robust to outliers
medae = np.median(np.abs(residuals))

print("=" * 55)
print("REGRESSION EVALUATION METRICS")
print("=" * 55)
print(f"""
Metric        Value    Interpretation
───────────────────────────────────────────────────────
MSE           {mse:.4f}  Average squared error (penalizes large errors)
RMSE          {rmse:.4f}  Same units as y — 'typical' error magnitude
MAE           {mae:.4f}  Average absolute error (robust to outliers)
MedAE         {medae:.4f}  Median absolute error (very robust)
MAPE          {mape:.2f}%  Percentage error (scale-independent)
R²            {r2:.4f}  Fraction of variance explained [0, 1]
Adj. R²       {r2_adj:.4f}  R² penalized for # features

House values are in $100k units.
RMSE = {rmse:.4f} → typical prediction error ≈ ${rmse*100:.0f}
""")

print("Cross-validation R² (5-fold):")
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f"  Scores: {cv_scores.round(4)}")
print(f"  Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Actual vs Predicted plot ──────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual y'); ax.set_ylabel('Predicted ŷ')
ax.set_title(f'Actual vs. Predicted\nR² = {r2:.4f}')
ax.legend(); ax.grid(True, alpha=0.3)

# Residual distribution
ax2 = axes[1]
ax2.hist(residuals, bins=60, color='steelblue', alpha=0.7,
         edgecolor='white', density=True)
x_norm = np.linspace(residuals.min(), residuals.max(), 300)
from scipy import stats
ax2.plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
         color='tomato', lw=2, label='Normal fit')
ax2.axvline(0, color='black', lw=1.5, linestyle='--')
ax2.set_xlabel('Residual e = y - ŷ')
ax2.set_ylabel('Density')
ax2.set_title(f'Residual Distribution\nMean={residuals.mean():.4f}, Std={residuals.std():.4f}')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/05_evaluation.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 6. Regularization

### 6.1 The Overfitting Problem

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
n   = 40
X_reg = np.sort(rng.uniform(0, 1, n)).reshape(-1, 1)
y_reg = np.sin(2*np.pi*X_reg.ravel()) + rng.normal(0, 0.3, n)

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)
X_plot = np.linspace(0, 1, 300).reshape(-1, 1)

degrees = [1, 3, 9, 15]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, deg in zip(axes, degrees):
    pipe = Pipeline([
        ('poly',  PolynomialFeatures(degree=deg, include_bias=False)),
        ('model', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_plot = pipe.predict(X_plot)
    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    test_mse  = mean_squared_error(y_test,  pipe.predict(X_test))

    ax.scatter(X_train.ravel(), y_train, s=25, color='steelblue', alpha=0.8,
               zorder=3, label='Train')
    ax.scatter(X_test.ravel(),  y_test,  s=25, color='tomato',    alpha=0.8,
               zorder=3, label='Test', marker='s')
    ax.plot(X_plot.ravel(), y_plot, color='green', lw=2)
    ax.plot(X_plot.ravel(), np.sin(2*np.pi*X_plot.ravel()), 'k--',
            lw=1, alpha=0.4, label='True')
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f'Degree {deg}\nTrain={train_mse:.3f}, Test={test_mse:.3f}',
                 color='green' if test_mse < 0.3 else 'tomato')
    ax.set_xlabel('x')
    if deg == 1: ax.legend(fontsize=7)

plt.suptitle('Polynomial Regression: Underfitting → Good Fit → Overfitting',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_overfitting.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 6.2 Ridge Regression (L2)

**Ridge** adds an L2 penalty on the weights:

$$L_\text{Ridge}(\mathbf{w}) = \frac{1}{n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 + \alpha \|\mathbf{w}\|_2^2$$

Closed-form solution:

$$\mathbf{w}^*_\text{Ridge} = (\mathbf{X}^T\mathbf{X} + \alpha n \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

Adding $\alpha \mathbf{I}$ to $\mathbf{X}^T\mathbf{X}$ **regularizes** the inversion: even if $\mathbf{X}^T\mathbf{X}$ is nearly singular (multicollinear features), the ridge solution is stable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ── Implement Ridge from scratch ──────────────────────────────────────────────
def ridge_closed_form(X, y, alpha):
    """
    Ridge: w = (X^T X + alpha * I)^{-1} X^T y
    Note: we do NOT regularize the intercept (first column of X is all ones).
    Standard practice: add alpha to diagonal EXCEPT for the bias term.
    """
    n, d = X.shape
    I = np.eye(d)
    I[0, 0] = 0   # Don't regularize bias
    w = np.linalg.solve(X.T @ X + alpha * n * I, X.T @ y)
    return w

# Regenerate data
rng = np.random.default_rng(42)
n   = 40
X_r = np.sort(rng.uniform(0, 1, n)).reshape(-1, 1)
y_r = np.sin(2*np.pi*X_r.ravel()) + rng.normal(0, 0.3, n)

# Degree-9 polynomial features
poly  = PolynomialFeatures(degree=9, include_bias=True)
X_p   = poly.fit_transform(X_r)
scaler = StandardScaler(with_std=True)
X_p_sc = scaler.fit_transform(X_p[:, 1:])   # Don't scale bias column
X_p_sc = np.hstack([np.ones((n, 1)), X_p_sc])

X_plot = np.linspace(0, 1, 300).reshape(-1, 1)
X_plot_p  = poly.transform(X_plot)
X_plot_sc = scaler.transform(X_plot_p[:, 1:])
X_plot_sc = np.hstack([np.ones((300, 1)), X_plot_sc])

alphas = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, alpha in zip(axes.flatten(), alphas):
    w = ridge_closed_form(X_p_sc, y_r, alpha)
    y_pred_plot = X_plot_sc @ w

    train_mse = ((y_r - X_p_sc @ w)**2).mean()
    # Use all data as "test" for visualization simplicity
    label = f'α={alpha}, Train MSE={train_mse:.4f}'

    ax.scatter(X_r.ravel(), y_r, s=25, color='steelblue', alpha=0.8, zorder=3)
    ax.plot(X_plot.ravel(), y_pred_plot, color='tomato', lw=2.5, label=label)
    ax.plot(X_plot.ravel(), np.sin(2*np.pi*X_plot.ravel()), 'k--',
            lw=1.5, alpha=0.5, label='True function')
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f'Ridge  α={alpha}\n||w||₂ = {np.linalg.norm(w[1:]):.2f}')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle('Ridge Regression — Effect of Regularization Strength α\n'
             'Larger α → smoother curve → smaller weights',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_ridge_alpha.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Ridge coefficient paths ───────────────────────────────────────────────────
from sklearn.linear_model import Ridge

alphas_path = np.logspace(-4, 4, 100)
coef_paths  = []

for a in alphas_path:
    ridge = Ridge(alpha=a)
    ridge.fit(X_p_sc[:, 1:], y_r)   # sklearn handles bias separately
    coef_paths.append(ridge.coef_)

coef_paths = np.array(coef_paths)

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(coef_paths.shape[1]):
    ax.plot(np.log10(alphas_path), coef_paths[:, i], lw=1.5, alpha=0.8)
ax.axhline(0, color='black', lw=1)
ax.set_xlabel('log₁₀(α)'); ax.set_ylabel('Coefficient value')
ax.set_title('Ridge Coefficient Paths\n'
             'All coefficients → 0 as α → ∞ (smoothly, never exactly 0)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/05_ridge_paths.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 6.3 Lasso Regression (L1)

**Lasso** adds an L1 penalty — this promotes **sparsity**:

$$L_\text{Lasso}(\mathbf{w}) = \frac{1}{n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 + \alpha \|\mathbf{w}\|_1$$

Unlike Ridge, Lasso has **no closed-form solution** (the L1 norm is not differentiable at zero). Solved via coordinate descent or proximal gradient methods.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# Standardize
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)

# ── Lasso vs Ridge coefficient paths ─────────────────────────────────────────
alphas_path = np.logspace(-3, 1, 100)
lasso_coefs = []
ridge_coefs = []

for a in alphas_path:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_sc, y)
    lasso_coefs.append(lasso.coef_)

    ridge = Ridge(alpha=a * len(y))  # Ridge alpha is not normalized by n in sklearn
    ridge.fit(X_sc, y)
    ridge_coefs.append(ridge.coef_)

lasso_coefs = np.array(lasso_coefs)
ridge_coefs = np.array(ridge_coefs)

colors = plt.cm.tab10(np.linspace(0, 1, X.shape[1]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, (name, color) in enumerate(zip(feature_names, colors)):
    axes[0].plot(np.log10(alphas_path), lasso_coefs[:, i], lw=2, color=color, label=name)
    axes[1].plot(np.log10(alphas_path), ridge_coefs[:, i], lw=2, color=color, label=name)

for ax, title in zip(axes, ['Lasso (L1) Coefficient Paths\n(Features driven to exactly 0 — sparsity)',
                              'Ridge (L2) Coefficient Paths\n(Features shrunk toward 0 — never exactly 0)']):
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('log₁₀(α)'); ax.set_ylabel('Coefficient')
    ax.set_title(title)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/05_lasso_vs_ridge_paths.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Key difference: sparsity ──────────────────────────────────────────────────
from sklearn.linear_model import Lasso, Ridge

lasso_a  = Lasso(alpha=0.1,       max_iter=10000).fit(X_sc, y)
ridge_a  = Ridge(alpha=0.1*len(y)).fit(X_sc, y)

print("Lasso vs Ridge — sparsity comparison (α=0.1):")
print(f"\n{'Feature':<25} {'Lasso':>12} {'Ridge':>12}")
print("-" * 52)
for name, lc, rc in zip(feature_names, lasso_a.coef_, ridge_a.coef_):
    lasso_str = f"{lc:.4f}" if abs(lc) > 1e-6 else "0 (zeroed!)"
    print(f"{name:<25} {lasso_str:>12} {rc:>12.4f}")

n_zero_lasso = (np.abs(lasso_a.coef_) < 1e-6).sum()
print(f"\nLasso zero coefficients: {n_zero_lasso}/{len(feature_names)}")
print(f"Ridge zero coefficients: {(np.abs(ridge_a.coef_) < 1e-6).sum()}/{len(feature_names)}")
print("\nLasso performs automatic feature selection — some features are completely excluded.")
```

---

### 6.4 ElasticNet — Best of Both

$$L_\text{EN}(\mathbf{w}) = \frac{1}{n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 + \alpha\left[\rho\|\mathbf{w}\|_1 + \frac{1-\rho}{2}\|\mathbf{w}\|_2^2\right]$$

- $\rho = 1$: pure Lasso
- $\rho = 0$: pure Ridge
- $0 < \rho < 1$: blend of both

Use ElasticNet when: (1) you want sparsity but also stability with correlated features (Lasso tends to pick one arbitrarily from a correlated group), or (2) you have more features than samples.

```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import numpy as np

housing = fetch_california_housing()
X, y   = housing.data, housing.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search over alpha and l1_ratio
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet(max_iter=5000))
])

param_grid = {
    'model__alpha':    [0.001, 0.01, 0.1, 1.0],
    'model__l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
gs.fit(X_tr, y_tr)

print(f"ElasticNet Best params: {gs.best_params_}")
print(f"Best CV R²:             {gs.best_score_:.4f}")
print(f"Test R²:                {gs.score(X_te, y_te):.4f}")
```

---

### 6.5 Choosing Regularization Strength

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X, y    = housing.data, housing.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# RidgeCV and LassoCV find optimal alpha via cross-validation automatically
alphas = np.logspace(-4, 4, 200)

ridge_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RidgeCV(alphas=alphas, cv=5, scoring='r2'))
])
ridge_cv.fit(X_tr, y_tr)

lasso_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LassoCV(alphas=alphas, cv=5, max_iter=10000, n_jobs=-1))
])
lasso_cv.fit(X_tr, y_tr)

best_alpha_ridge = ridge_cv.named_steps['model'].alpha_
best_alpha_lasso = lasso_cv.named_steps['model'].alpha_

print("Automatic regularization strength selection via Cross-Validation:")
print(f"  Ridge optimal α: {best_alpha_ridge:.6f}")
print(f"  Lasso optimal α: {best_alpha_lasso:.6f}")
print(f"  Ridge test R²:   {ridge_cv.score(X_te, y_te):.4f}")
print(f"  Lasso test R²:   {lasso_cv.score(X_te, y_te):.4f}")

# ── Lasso CV path ─────────────────────────────────────────────────────────────
mse_path_mean = lasso_cv.named_steps['model'].mse_path_.mean(axis=-1)
alphas_path   = lasso_cv.named_steps['model'].alphas_

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogx(alphas_path, mse_path_mean, color='steelblue', lw=2)
ax.axvline(best_alpha_lasso, color='tomato', lw=2, linestyle='--',
           label=f'Best α={best_alpha_lasso:.4f}')
ax.set_xlabel('α (regularization strength)')
ax.set_ylabel('Mean CV MSE')
ax.set_title('LassoCV — Choosing α via Cross-Validation\n'
             'U-shape: too small = overfit, too large = underfit')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/05_lasso_cv.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 7. Polynomial Regression

Linear regression becomes **polynomial regression** when we add polynomial features. The model is still *linear in parameters* — it's linear regression applied to transformed inputs.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve

rng = np.random.default_rng(0)
n   = 100
X_p = np.sort(rng.uniform(0, 1, n)).reshape(-1, 1)
y_p = np.sin(2*np.pi*X_p.ravel()) + rng.normal(0, 0.2, n)

# ── Validation curve: find best polynomial degree ─────────────────────────────
pipe_poly = Pipeline([
    ('poly',   PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=0.01))
])

degrees    = np.arange(1, 16)
train_scores, val_scores = validation_curve(
    pipe_poly, X_p, y_p,
    param_name='poly__degree', param_range=degrees,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

train_mse_mean = -train_scores.mean(axis=1)
train_mse_std  = train_scores.std(axis=1)
val_mse_mean   = -val_scores.mean(axis=1)
val_mse_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(degrees, train_mse_mean, 'o-', color='steelblue', lw=2, label='Train MSE')
ax.fill_between(degrees, train_mse_mean - train_mse_std,
                train_mse_mean + train_mse_std, alpha=0.2, color='steelblue')
ax.plot(degrees, val_mse_mean,   's-', color='tomato',    lw=2, label='Validation MSE')
ax.fill_between(degrees, val_mse_mean - val_mse_std,
                val_mse_mean + val_mse_std, alpha=0.2, color='tomato')
ax.axvline(degrees[np.argmin(val_mse_mean)], color='green', lw=2, linestyle='--',
           label=f'Best degree = {degrees[np.argmin(val_mse_mean)]}')
ax.set_xlabel('Polynomial Degree'); ax.set_ylabel('MSE')
ax.set_title('Validation Curve — Finding Optimal Polynomial Degree\n'
             'Train keeps decreasing; validation has a sweet spot')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.3)
plt.tight_layout()
plt.savefig('figures/05_polynomial_validation.png', dpi=140, bbox_inches='tight')
plt.show()

best_deg = degrees[np.argmin(val_mse_mean)]
print(f"Best polynomial degree: {best_deg}")
print(f"Val MSE at best degree: {val_mse_mean[best_deg-1]:.4f}")
```

---

## 8. Residual Diagnostics

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y    = housing.data, housing.target

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])
pipe.fit(X, y)
y_hat     = pipe.predict(X)
residuals = y - y_hat
std_res   = (residuals - residuals.mean()) / residuals.std()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. Residuals vs Fitted
axes[0,0].scatter(y_hat, residuals, s=3, alpha=0.3, color='steelblue')
axes[0,0].axhline(0, color='tomato', lw=2, linestyle='--')
# Add lowess-style smooth
from scipy.ndimage import uniform_filter1d
idx_sort = np.argsort(y_hat)
smoothed = uniform_filter1d(residuals[idx_sort], size=500)
axes[0,0].plot(y_hat[idx_sort], smoothed, 'tomato', lw=2.5, label='Trend')
axes[0,0].set_xlabel('Fitted ŷ'); axes[0,0].set_ylabel('Residual')
axes[0,0].set_title('Residuals vs Fitted\n(Non-random pattern → non-linearity)')
axes[0,0].legend(fontsize=8)

# 2. Q-Q Plot
(osm, osr), (slope, intercept, _) = stats.probplot(residuals)
axes[0,1].scatter(osm, osr, s=3, alpha=0.3, color='steelblue')
axes[0,1].plot([osm[0], osm[-1]], [slope*osm[0]+intercept, slope*osm[-1]+intercept],
               'tomato', lw=2)
axes[0,1].set_xlabel('Theoretical quantiles'); axes[0,1].set_ylabel('Sample quantiles')
axes[0,1].set_title('Q-Q Plot\n(Deviations → non-normal residuals)')

# 3. Scale-Location (sqrt|residuals| vs fitted)
sqrt_abs_res = np.sqrt(np.abs(std_res))
axes[0,2].scatter(y_hat, sqrt_abs_res, s=3, alpha=0.3, color='steelblue')
smoothed2 = uniform_filter1d(sqrt_abs_res[idx_sort], size=500)
axes[0,2].plot(y_hat[idx_sort], smoothed2, 'tomato', lw=2.5)
axes[0,2].set_xlabel('Fitted ŷ'); axes[0,2].set_ylabel('√|Standardized residual|')
axes[0,2].set_title('Scale-Location\n(Funnel shape → heteroscedasticity)')

# 4. Residual histogram
axes[1,0].hist(residuals, bins=80, color='steelblue', alpha=0.7,
               density=True, edgecolor='white')
x_norm = np.linspace(residuals.min(), residuals.max(), 300)
axes[1,0].plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
               'tomato', lw=2, label='Normal')
axes[1,0].set_xlabel('Residual'); axes[1,0].set_ylabel('Density')
axes[1,0].set_title(f'Residual Histogram\nSkew={residuals.skew():.3f}  Kurt={residuals.kurtosis():.3f}')
axes[1,0].legend()

# 5. Cook's Distance (influence measure)
# Cook's distance ≈ (leverage × residual²) / (p × MSE)
# High Cook's D → single point strongly influences the fit
n, p  = X.shape
H     = X @ np.linalg.solve(X.T @ X, X.T)   # Hat matrix (leverage)
h     = np.diag(H)                            # Leverage scores
mse   = (residuals**2).mean()
cooks = residuals**2 * h / (p * mse * (1 - h)**2 + 1e-10)
threshold = 4 / n

axes[1,1].stem(range(len(cooks)), cooks, markerfmt='b.',
               linefmt='steelblue', basefmt='k-')
axes[1,1].axhline(threshold, color='tomato', lw=2, linestyle='--',
                  label=f'Threshold = 4/n = {threshold:.4f}')
n_influential = (cooks > threshold).sum()
axes[1,1].set_xlabel('Observation index')
axes[1,1].set_ylabel("Cook's Distance")
axes[1,1].set_title(f"Cook's Distance — Influential Points\n"
                    f"{n_influential} points above threshold ({n_influential/n*100:.1f}%)")
axes[1,1].legend(fontsize=8)
axes[1,1].set_ylim(0, min(threshold*20, cooks.max()*1.1))

# 6. Leverage vs Residual (Influence plot)
axes[1,2].scatter(h, std_res, s=5, alpha=0.3, color='steelblue')
axes[1,2].axhline(0, color='black', lw=0.8)
axes[1,2].axhline(3, color='orange', lw=1.5, linestyle='--', label='|std_res| = 3')
axes[1,2].axhline(-3, color='orange', lw=1.5, linestyle='--')
axes[1,2].axvline(2*p/n, color='tomato', lw=1.5, linestyle='--',
                  label=f'High leverage = 2p/n')
axes[1,2].set_xlabel('Leverage h'); axes[1,2].set_ylabel('Standardized residual')
axes[1,2].set_title('Influence Plot\n(Top-right = dangerous outliers)')
axes[1,2].legend(fontsize=8)

plt.suptitle('Residual Diagnostics — California Housing Linear Regression',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_residual_diagnostics.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"\nDiagnostic Summary:")
print(f"  Residual skewness: {float(np.array(residuals).flatten().__class__.__name__ and stats.skew(residuals)):.3f}")
print(f"  High leverage points (h > 2p/n): {(h > 2*p/n).sum()}")
print(f"  Influential points (Cook's D > 4/n): {n_influential}")
print(f"  Outliers (|std_res| > 3): {(np.abs(std_res) > 3).sum()}")
```

---

## 9. Linear Regression with Scikit-learn

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

housing = fetch_california_housing()
X, y    = housing.data, housing.target
feature_names = housing.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Compare all linear regression variants ────────────────────────────────────
models = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression())
    ]),
    'Ridge (auto-CV)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5))
    ]),
    'Lasso (auto-CV)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Lasso(alpha=0.01, max_iter=10000))
    ]),
    'ElasticNet': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000))
    ]),
    'Polynomial (deg=2)': Pipeline([
        ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=1.0))
    ]),
}

print(f"{'Model':<25} {'Train R²':>10} {'Test R²':>10} {'Test RMSE':>11} {'CV R² (mean±std)':>22}")
print("-" * 82)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    train_r2  = r2_score(y_train, y_pred_train)
    test_r2   = r2_score(y_test,  y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
    cv_str    = f"{cv_scores.mean():.4f}±{cv_scores.std():.4f}"

    print(f"{name:<25} {train_r2:>10.4f} {test_r2:>10.4f} {test_rmse:>11.4f} {cv_str:>22}")
```

---

## 10. Case Study: Predicting California House Prices

```python
# =============================================================================
# Full pipeline: EDA → preprocessing → multiple models → evaluation
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Load and inspect ──────────────────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("Dataset summary:")
print(f"  Shape: {df.shape}")
print(f"  Target: MedHouseVal (median house value in $100k)")
print(f"  Known issue: values capped at $500k (= 5.0 in data)")
n_capped = (df['MedHouseVal'] == 5.0).sum()
print(f"  Capped rows: {n_capped} ({n_capped/len(df)*100:.1f}%)")

# ── Feature engineering ───────────────────────────────────────────────────────
df['rooms_per_household']   = df['AveRooms']   / (df['AveOccup'] + 1e-5)
df['bedrooms_per_room']     = df['AveBedrms']  / (df['AveRooms'] + 1e-5)
df['log_population']        = np.log1p(df['Population'])
df['log_AveOccup']          = np.log1p(df['AveOccup'])
df['MedInc_sq']             = df['MedInc'] ** 2
df['coastal_proximity']     = (df['Longitude'] < -121.5).astype(float)

feature_cols = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'log_population', 'log_AveOccup', 'Latitude', 'Longitude',
    'rooms_per_household', 'bedrooms_per_room',
    'MedInc_sq', 'coastal_proximity'
]

X = df[feature_cols].values
y = df['MedHouseVal'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model comparison ──────────────────────────────────────────────────────────
alphas = np.logspace(-4, 4, 200)

models = {
    'OLS (no regularization)': Pipeline([
        ('sc', StandardScaler()),
        ('m',  LinearRegression())
    ]),
    'Ridge (CV-tuned α)': Pipeline([
        ('sc', StandardScaler()),
        ('m',  RidgeCV(alphas=alphas, cv=5))
    ]),
    'Ridge + Poly(deg=2)': Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('sc',   StandardScaler()),
        ('m',    RidgeCV(alphas=alphas, cv=5))
    ]),
}

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    cv     = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
    results[name] = {'RMSE': rmse, 'R2': r2, 'CV_mean': cv.mean(), 'CV_std': cv.std()}
    print(f"{name:<30}: RMSE={rmse:.4f}, R²={r2:.4f}, CV={cv.mean():.4f}±{cv.std():.4f}")

# Best model: interpret coefficients
best_pipe = models['Ridge (CV-tuned α)']
coefs     = best_pipe.named_steps['m'].coef_
best_alpha = best_pipe.named_steps['m'].alpha_

print(f"\nBest Ridge α: {best_alpha:.4f}")
print("\nFeature Coefficients (standardized — comparable):")
coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': coefs})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
for _, row in coef_df.iterrows():
    bar = '█' * int(abs(row['Coefficient']) * 3)
    sign = '+' if row['Coefficient'] > 0 else '-'
    print(f"  {row['Feature']:<25}: {row['Coefficient']:>+8.4f}  {sign}{bar}")

# ── Prediction error analysis ──────────────────────────────────────────────────
y_pred_best = best_pipe.predict(X_test)
errors      = y_test - y_pred_best

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(y_test, y_pred_best, s=5, alpha=0.3, color='steelblue')
axes[0].plot([0, 5], [0, 5], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual ($100k)'); axes[0].set_ylabel('Predicted ($100k)')
axes[0].set_title(f"Actual vs Predicted\nR²={r2_score(y_test, y_pred_best):.4f}")
axes[0].legend()

axes[1].hist(errors, bins=60, color='steelblue', alpha=0.7, edgecolor='white', density=True)
axes[1].axvline(0, color='tomato', lw=2, linestyle='--')
axes[1].set_xlabel('Prediction Error ($100k)'); axes[1].set_ylabel('Density')
axes[1].set_title(f'Error Distribution\nMean={errors.mean():.4f}, Std={errors.std():.4f}')

# Learning curve
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    best_pipe, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2', n_jobs=-1
)
axes[2].plot(train_sizes, train_scores.mean(axis=1), 'o-', color='steelblue', label='Train')
axes[2].fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1),
                     train_scores.mean(1)+train_scores.std(1), alpha=0.2, color='steelblue')
axes[2].plot(train_sizes, val_scores.mean(axis=1), 's-', color='tomato', label='Validation')
axes[2].fill_between(train_sizes, val_scores.mean(1)-val_scores.std(1),
                     val_scores.mean(1)+val_scores.std(1), alpha=0.2, color='tomato')
axes[2].set_xlabel('Training set size'); axes[2].set_ylabel('R² Score')
axes[2].set_title('Learning Curve\n(Gap = variance, low val = bias)')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/05_case_study.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

## 11. Summary

### ✅ Must-Remember Mental Models

**1. The OLS formula:**
$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
Use `np.linalg.solve(XtX, Xty)`, never explicit inverse.

**2. Regularization — when and which:**
- Default start: Ridge. Always try it.
- When you suspect only a few features matter: Lasso.
- Correlated features + sparsity wanted: ElasticNet.
- Tree-based models: regularization is built-in → no Ridge/Lasso needed.

**3. Ridge vs. Lasso geometry:**
- Ridge (L2): circular constraint → shrinks all weights proportionally → never exactly zero.
- Lasso (L1): diamond constraint → corners on axes → solution lands on corners → exact zeros → automatic feature selection.

**4. Scaling is mandatory before gradient descent (and Ridge/Lasso).**
Without it, features on different scales have different effective regularization strengths.

**5. The five residual plots to always run:**
Residuals vs Fitted, Q-Q plot, Scale-Location, Cook's Distance, Leverage vs Residual.

**6. R² interpretation:**
- R² = 0.6 means the model explains 60% of variance — not that it predicts within 60%.
- Always report RMSE alongside R²: RMSE is in the units of y.
- Adjusted R² penalizes for extra features — use it when comparing models with different numbers of features.

**7. Polynomial regression = feature engineering + linear regression.**
The model is still linear in parameters. Regularize polynomial models aggressively.

---

## 12. Exercises

1. **Derive the gradient** of the Lasso loss $L = \frac{1}{n}\|Xw-y\|^2 + \alpha\|w\|_1$. Why can't we set it to zero analytically? What is the subgradient at $w_j = 0$?

2. **Implement Ridge regression from scratch** (NumPy only). Verify it matches `sklearn.linear_model.Ridge` on the California Housing dataset for $\alpha \in \{0.01, 0.1, 1, 10\}$.

3. **Multicollinearity experiment**: generate 200 samples with features $x_1, x_2 = x_1 + \epsilon, x_3$. Fit OLS and Ridge. Show that OLS gives unstable coefficients when $\epsilon$ is small, while Ridge is stable. Measure this with the coefficient standard error across 50 different training sets.

4. **Learning curves**: for the California Housing dataset, plot learning curves for (a) LinearRegression, (b) Ridge, (c) polynomial degree-2 Ridge. What does the gap between train and validation tell you about each model's variance?

5. **Residual diagnostics**: deliberately violate each assumption (create a heteroscedastic dataset, a non-linear dataset, a dataset with autocorrelated errors) and show the corresponding diagnostic plots that detect each violation.

6. **Challenge — Weighted Least Squares (WLS)**: In WLS, different observations have different weights $w_i$. Derive the WLS solution $\hat{w} = (X^T W X)^{-1} X^T W y$ where $W = \text{diag}(w_1, \ldots, w_n)$. Implement it from scratch. Show that WLS can partially correct for heteroscedasticity by setting $w_i = 1/\hat{\sigma}_i^2$.

---

## 13. References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Chapters 3–4.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *JRSS-B*, 58(1), 267–288. — Original Lasso paper.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression. *Technometrics*, 12(1), 55–67.
- Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *JRSS-B*, 67(2), 301–320.

---

*Next: [Chapter 6 — Logistic Regression & Binary Classification](06_logistic_regression.md)*
