# Chapter 13: Feature Engineering & Selection

> *"Applied machine learning is basically feature engineering."*  
> — Andrew Ng

> *"A feature is just a number. But the right number — derived from domain knowledge, careful observation, and creative transformation — can be the difference between a mediocre model and a great one."*

---

## Table of Contents

1. [What Is Feature Engineering?](#1-what-is-feature-engineering)
2. [Encoding Categorical Variables](#2-encoding-categorical-variables)
   - 2.1 [One-Hot Encoding](#21-one-hot-encoding)
   - 2.2 [Ordinal Encoding](#22-ordinal-encoding)
   - 2.3 [Target Encoding](#23-target-encoding)
   - 2.4 [Frequency and Count Encoding](#24-frequency-and-count-encoding)
   - 2.5 [Binary and Hashing Encoding](#25-binary-and-hashing-encoding)
3. [Scaling and Normalization](#3-scaling-and-normalization)
4. [Handling Skewed Distributions](#4-handling-skewed-distributions)
5. [Handling Missing Values](#5-handling-missing-values)
6. [Creating New Features](#6-creating-new-features)
   - 6.1 [Polynomial and Interaction Features](#61-polynomial-and-interaction-features)
   - 6.2 [Domain-Specific Features](#62-domain-specific-features)
   - 6.3 [Date and Time Features](#63-date-and-time-features)
   - 6.4 [Binning and Discretization](#64-binning-and-discretization)
7. [Feature Selection](#7-feature-selection)
   - 7.1 [Filter Methods](#71-filter-methods)
   - 7.2 [Wrapper Methods](#72-wrapper-methods)
   - 7.3 [Embedded Methods](#73-embedded-methods)
   - 7.4 [Permutation Importance](#74-permutation-importance)
8. [Automated Feature Selection with Scikit-learn](#8-automated-feature-selection-with-scikit-learn)
9. [The Full Feature Engineering Pipeline](#9-the-full-feature-engineering-pipeline)
10. [Case Study: House Price Feature Engineering](#10-case-study-house-price-feature-engineering)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. What Is Feature Engineering?

**Feature engineering** is the process of using domain knowledge to transform raw data into features that better represent the underlying problem to predictive models.

It includes:
- **Encoding**: converting categorical variables to numbers
- **Scaling**: normalizing feature magnitudes
- **Transformation**: log, sqrt, Box-Cox for skewed distributions
- **Creation**: combining existing features into new, more informative ones
- **Selection**: removing irrelevant or redundant features

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

housing = fetch_california_housing(as_frame=True)
df      = housing.frame.copy()
feature_names = housing.feature_names

# ── Demonstrate impact of feature engineering ─────────────────────────────────
X_raw = df[feature_names].values
y     = df['MedHouseVal'].values

# Baseline: raw features
baseline_cv = cross_val_score(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X_raw, y, cv=5, scoring='r2'
)

# After feature engineering
df_eng = df[feature_names].copy()
df_eng['rooms_per_household']   = df_eng['AveRooms'] / (df_eng['AveOccup'] + 1e-5)
df_eng['bedrooms_per_room']     = df_eng['AveBedrms'] / (df_eng['AveRooms'] + 1e-5)
df_eng['log_population']        = np.log1p(df_eng['Population'])
df_eng['log_AveOccup']          = np.log1p(df_eng['AveOccup'])
df_eng['MedInc_sq']             = df_eng['MedInc'] ** 2
df_eng['MedInc_x_Rooms']        = df_eng['MedInc'] * df_eng['AveRooms']
df_eng['income_per_bedroom']    = df_eng['MedInc'] / (df_eng['AveBedrms'] + 1e-5)

X_eng = df_eng.values

engineered_cv = cross_val_score(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X_eng, y, cv=5, scoring='r2'
)

fig, ax = plt.subplots(figsize=(8, 4))
methods = ['Baseline\n(raw features)', 'After Feature\nEngineering']
means   = [baseline_cv.mean(), engineered_cv.mean()]
stds    = [baseline_cv.std(), engineered_cv.std()]
colors  = ['#90CAF9', '#A5D6A7']
bars    = ax.bar(methods, means, color=colors, alpha=0.9, width=0.4)
ax.errorbar([0, 1], means, yerr=stds, fmt='none', color='black',
            capsize=8, lw=2)
ax.bar_label(bars, labels=[f'{m:.4f}' for m in means], padding=5, fontsize=11)
ax.set_ylabel('CV R² Score')
ax.set_title('Impact of Feature Engineering on Random Forest\n'
             'California Housing Dataset')
ax.set_ylim(0.8, 0.88)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/13_fe_impact.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"Baseline CV R²:  {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}")
print(f"Engineered CV R²: {engineered_cv.mean():.4f} ± {engineered_cv.std():.4f}")
print(f"Improvement: {engineered_cv.mean() - baseline_cv.mean():+.4f}")
print(f"\nNew features added: {X_eng.shape[1] - X_raw.shape[1]} "
      f"(from {X_raw.shape[1]} to {X_eng.shape[1]})")
```

---

## 2. Encoding Categorical Variables

Machines work with numbers. Categorical variables must be converted.

### 2.1 One-Hot Encoding

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df_cat = pd.DataFrame({
    'color':     ['red', 'blue', 'green', 'red', 'blue'],
    'size':      ['S', 'M', 'L', 'XL', 'M'],
    'material':  ['cotton', 'silk', 'cotton', 'wool', 'silk'],
    'price':     [10, 25, 15, 30, 20]
})

print("Original DataFrame:")
print(df_cat)

# ── Manual one-hot encoding ───────────────────────────────────────────────────
def one_hot_manual(series):
    """One-hot encode a pandas Series."""
    unique_vals = series.unique()
    df_ohe = pd.DataFrame(
        {f'{series.name}_{v}': (series == v).astype(int)
         for v in unique_vals}
    )
    return df_ohe

ohe_color = one_hot_manual(df_cat['color'])
print("\nManual one-hot (color):")
print(ohe_color)

# ── sklearn OneHotEncoder ─────────────────────────────────────────────────────
ohe = OneHotEncoder(
    drop='first',           # Drop first category to avoid multicollinearity (dummy trap)
    sparse_output=False,    # Return dense array
    handle_unknown='ignore' # Ignore unseen categories at test time
)
X_ohe = ohe.fit_transform(df_cat[['color', 'size', 'material']])
print(f"\nsklearn OHE shape: {X_ohe.shape}")
print(f"Feature names: {ohe.get_feature_names_out()}")

# ── When NOT to one-hot encode ────────────────────────────────────────────────
print("""
One-Hot Encoding Guidelines:
  ✓ Low cardinality (< 20 unique values)
  ✓ No ordinal relationship (red/blue/green — no ordering)
  ✓ With linear models, SVMs, neural networks
  
  ✗ High cardinality (100+ categories) → feature explosion
    → Use target encoding, frequency encoding, or embeddings
  ✗ Tree models don't need OHE — they split on any threshold
    → But OHE doesn't hurt trees; use for consistency
  ✗ 'drop' parameter: use drop='first' to avoid the dummy variable trap
    (one category is implied by the others; keeps the matrix full-rank)
""")
```

---

### 2.2 Ordinal Encoding

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df_ord = pd.DataFrame({
    'education': ['HS', 'BSc', 'MSc', 'PhD', 'BSc', 'HS', 'PhD'],
    'rating':    ['Poor', 'Fair', 'Good', 'Excellent', 'Good', 'Fair', 'Excellent'],
    'size':      ['S', 'M', 'L', 'XL', 'S', 'L', 'M'],
})

# ── Manual ordinal encoding ────────────────────────────────────────────────────
edu_order    = {'HS': 0, 'BSc': 1, 'MSc': 2, 'PhD': 3}
rating_order = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
size_order   = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}

df_ord['edu_encoded']    = df_ord['education'].map(edu_order)
df_ord['rating_encoded'] = df_ord['rating'].map(rating_order)
df_ord['size_encoded']   = df_ord['size'].map(size_order)
print("Manual ordinal encoding:")
print(df_ord)

# ── sklearn OrdinalEncoder ────────────────────────────────────────────────────
oe = OrdinalEncoder(categories=[
    ['HS', 'BSc', 'MSc', 'PhD'],       # Must specify order explicitly!
    ['Poor', 'Fair', 'Good', 'Excellent'],
    ['S', 'M', 'L', 'XL']
])
X_ord = oe.fit_transform(df_ord[['education', 'rating', 'size']])
print(f"\nsklearn OrdinalEncoder output:")
print(X_ord)

print("""
Ordinal Encoding Rules:
  ✓ When the categorical variable has a MEANINGFUL ORDER
    (education level, rating, size, temperature: cold/warm/hot)
  ✓ The order must be specified explicitly — don't rely on alphabetical!
  ✗ Never use for nominal variables (colors, city names)
    → OHE instead
""")
```

---

### 2.3 Target Encoding

Target encoding replaces each category with the **mean of the target variable** for that category. Powerful but requires careful implementation to prevent data leakage.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

class TargetEncoderCV:
    """
    Target encoding with cross-validation to prevent leakage.
    For each fold, encode training samples using ONLY statistics
    from other folds (never from the current sample's fold).
    """

    def __init__(self, n_splits=5, smoothing=1.0, random_state=42):
        """
        smoothing: blend category mean with global mean
                   Higher = more regularization (pulls toward global mean)
        """
        self.n_splits    = n_splits
        self.smoothing   = smoothing
        self.random_state = random_state

    def fit(self, X, y):
        """Compute the final encoding mapping (global stats)."""
        self.global_mean_ = y.mean()
        self.encoding_map_ = {}

        for col in X.columns:
            # Smoothed target mean per category
            stats = y.groupby(X[col]).agg(['mean', 'count'])
            smoothing_factor = 1 / (1 + np.exp(-(stats['count'] - 1) / self.smoothing))
            self.encoding_map_[col] = (
                smoothing_factor * stats['mean'] +
                (1 - smoothing_factor) * self.global_mean_
            )
        return self

    def transform(self, X, y=None):
        """Apply encoding. If y is provided, use CV encoding (for train set)."""
        X_enc = X.copy()

        if y is not None:
            # CV encoding for training set
            kf = KFold(n_splits=self.n_splits, shuffle=True,
                       random_state=self.random_state)
            for col in X.columns:
                encoded = np.full(len(X), self.global_mean_)
                for train_idx, val_idx in kf.split(X):
                    fold_stats = y.iloc[train_idx].groupby(
                        X[col].iloc[train_idx]
                    ).agg(['mean', 'count'])
                    for cat in X[col].iloc[val_idx].unique():
                        if cat in fold_stats.index:
                            n = fold_stats.loc[cat, 'count']
                            m = fold_stats.loc[cat, 'mean']
                            sf = 1 / (1 + np.exp(-(n-1)/self.smoothing))
                            encoded[val_idx[X[col].iloc[val_idx] == cat]] = (
                                sf * m + (1-sf) * self.global_mean_
                            )
                X_enc[col] = encoded
        else:
            # Test set encoding using global map
            for col in X.columns:
                X_enc[col] = X[col].map(self.encoding_map_).fillna(self.global_mean_)

        return X_enc


# ── Demo: high-cardinality categorical ────────────────────────────────────────
rng = np.random.default_rng(42)
n   = 500
cities = ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix',
          'Philadelphia', 'San Antonio', 'Dallas', 'Austin', 'Seattle']

df_demo = pd.DataFrame({
    'city':   rng.choice(cities, n),
    'income': rng.normal(60000, 15000, n),
})

# True relationship: NYC/LA/Seattle → higher salary
salary_boost = {'NYC': 20000, 'LA': 15000, 'Seattle': 12000,
                'Chicago': 8000, 'San Antonio': 0, 'Phoenix': 0,
                'Philadelphia': 5000, 'Houston': 3000, 'Dallas': 2000, 'Austin': 7000}
df_demo['salary'] = (df_demo['income'] +
                     df_demo['city'].map(salary_boost) +
                     rng.normal(0, 5000, n))

# Compare encoding methods
y_demo   = df_demo['salary']
X_city   = df_demo[['city']]

# Method 1: Target encoding (CV)
te = TargetEncoderCV(smoothing=1.0)
te.fit(X_city, y_demo)
X_te = te.transform(X_city, y_demo)

# Method 2: OHE
from sklearn.preprocessing import OneHotEncoder
ohe_demo = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_ohe_demo = ohe_demo.fit_transform(X_city)

print("Target Encoding vs OHE — City salary boost:")
print(f"\nGlobal mean salary: {y_demo.mean():.0f}")
print(f"\n{'City':<15} {'True Boost':>12} {'TE Encoding':>13} {'TE - Mean':>10}")
print("-" * 55)
for city in sorted(cities):
    true_boost = salary_boost[city]
    te_val     = te.encoding_map_['city'].get(city, te.global_mean_)
    te_diff    = te_val - te.global_mean_
    print(f"{city:<15} {true_boost:>12,.0f} {te_val:>13,.0f} {te_diff:>+10,.0f}")

print("""
\nTarget Encoding Best Practices:
  ✓ Use cross-validation (K-fold) to prevent leakage on training set
  ✓ Apply smoothing to handle rare categories (pull toward global mean)
  ✓ Store global-mean mapping for test set transformation
  ✓ Excellent for high-cardinality features (100+ categories)
  ✓ Better than OHE for tree models on high-cardinality features
  ✗ Risk of overfitting if not done with CV
""")
```

---

### 2.4 Frequency and Count Encoding

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
df_freq = pd.DataFrame({
    'country': rng.choice(['USA', 'UK', 'China', 'India', 'Brazil',
                           'France', 'Germany', 'Japan'], 1000,
                          p=[0.4, 0.15, 0.15, 0.1, 0.08, 0.05, 0.04, 0.03]),
    'product': rng.choice(['A','B','C','D','E'], 1000)
})

# Frequency encoding: replace category with its frequency (proportion)
freq_map = df_freq['country'].value_counts(normalize=True)
df_freq['country_freq'] = df_freq['country'].map(freq_map)

# Count encoding: replace with raw count
count_map = df_freq['country'].value_counts()
df_freq['country_count'] = df_freq['country'].map(count_map)

print("Frequency and Count Encoding:")
print(df_freq.groupby('country')[['country_freq', 'country_count']].first()
      .sort_values('country_freq', ascending=False))

print("""
When to use Frequency Encoding:
  ✓ Captures the "popularity" signal (rare categories → small encoding)
  ✓ Works well when frequency itself is predictive
    (e.g., popular products → different behavior than niche ones)
  ✓ No cardinality explosion (one column per original column)
  ✗ Two categories with same frequency get same encoding
    → Loses identity information
  → Often used COMBINED with other encodings
""")
```

---

### 2.5 Binary and Hashing Encoding

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

# ── Hashing trick: for very high cardinality ─────────────────────────────────
# Hash each category to a fixed-size bit vector
# No need to store a vocabulary → great for streaming/online learning

categories = [f'category_{i}' for i in range(10000)]  # 10,000 unique categories

hasher = FeatureHasher(n_features=256, input_type='string')
X_hashed = hasher.transform([[cat] for cat in categories[:10]])
print(f"Hashing trick: 10,000 categories → {X_hashed.shape[1]} hash features")
print(f"Non-zero per row: {X_hashed.nnz / 10:.1f} (expected: ~1 per category)")

print("""
Encoding Method Selection Guide:
════════════════════════════════════════════════════════════════════
# Unique  Method              Notes
────────────────────────────────────────────────────────────────────
2         OHE or Binary       Binary: {0,1} directly
3-15      OHE (drop='first')  Standard choice
16-50     OHE or Target       Target if predictive; OHE otherwise
50-1000   Target Encoding     With CV smoothing; much better than OHE
1000+     Hashing or Target   Hashing for memory efficiency
Ordinal   OrdinalEncoder      Must specify explicit order!
────────────────────────────────────────────────────────────────────
For tree models: any method works (trees find splits on any encoding)
For linear models: OHE or Target (scale-sensitive, needs numeric)
For NNs: Embeddings (learned dense vectors) are best for high cardinality
════════════════════════════════════════════════════════════════════
""")
```

---

## 3. Scaling and Normalization

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                    RobustScaler, MaxAbsScaler,
                                    Normalizer, PowerTransformer)

rng  = np.random.default_rng(42)
# Dataset with different scales and an outlier
data = {
    'age':    rng.integers(18, 80, 200).astype(float),
    'income': rng.lognormal(11, 0.8, 200),   # Right-skewed
    'score':  rng.normal(75, 15, 200),
}
data['income'][5] = 2_000_000   # Outlier

df_scale = pd.DataFrame(data)
X = df_scale.values

scalers = {
    'Original':               None,
    'StandardScaler\n(z-score)': StandardScaler(),
    'MinMaxScaler\n[0,1]':       MinMaxScaler(),
    'RobustScaler\n(median/IQR)': RobustScaler(),
    'PowerTransformer\n(Yeo-Johnson)': PowerTransformer(method='yeo-johnson'),
}

fig, axes = plt.subplots(len(scalers), 3, figsize=(12, 16))

for row, (name, scaler) in enumerate(scalers.items()):
    X_scaled = scaler.fit_transform(X) if scaler else X.copy()

    for col, (feat, feat_name) in enumerate(zip(
        range(3), ['age', 'income', 'score']
    )):
        ax = axes[row, col]
        ax.hist(X_scaled[:, col], bins=30, color='steelblue',
                alpha=0.7, edgecolor='white', density=True)
        mu = X_scaled[:, col].mean()
        sd = X_scaled[:, col].std()
        ax.set_title(f'{feat_name} | {name}\nμ={mu:.2f}, σ={sd:.2f}', fontsize=7)
        ax.tick_params(labelsize=6)

plt.suptitle('Scaling Methods Comparison\n'
             'Note how each handles the income outlier differently',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/13_scaling_comparison.png', dpi=120, bbox_inches='tight')
plt.show()

print("""
Scaling Method Guide:
════════════════════════════════════════════════════════════════════
StandardScaler (z-score):
  Formula: (x - mean) / std
  Output: mean=0, std=1 (but unbounded range)
  ✓ Best default for most models
  ✗ Sensitive to outliers (mean and std are affected)

MinMaxScaler [0, 1]:
  Formula: (x - min) / (max - min)
  Output: exactly in [0, 1]
  ✓ Preserves zero values (good for sparse data)
  ✗ Very sensitive to outliers (one outlier stretches everything)

RobustScaler (median/IQR):
  Formula: (x - median) / IQR
  Output: centered at 0, robust range
  ✓ Best when data has outliers (uses robust statistics)
  ✗ Output range not fixed

PowerTransformer (Yeo-Johnson):
  Makes distributions more Gaussian
  ✓ Best for highly skewed features (income, population)
  ✓ Handles both positive and negative values (unlike Box-Cox)
  ✓ Automatically finds optimal λ

Normalizer (row-wise):
  Formula: x / ||x||₂ (per sample, not per feature)
  ✓ For text/document features (makes all doc vectors unit length)
  ✗ Not for tabular data (destroys feature scale info)

When NOT to scale:
  • Decision trees / Random Forests / XGBoost: scale-invariant
  • Naive Bayes: uses raw distributions
  • Neural networks: BatchNorm makes input scale less critical
  
When MUST scale:
  • KNN, SVM, PCA, Logistic Regression, Linear Regression
  • Gradient descent (without scaling: slow convergence or no convergence)
════════════════════════════════════════════════════════════════════
""")
```

---

## 4. Handling Skewed Distributions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PowerTransformer

rng   = np.random.default_rng(42)
n     = 500

# Generate skewed data
income      = rng.lognormal(10, 1.2, n)
transaction = rng.exponential(50, n)
age         = rng.normal(40, 15, n).clip(0, 100)

fig, axes = plt.subplots(3, 5, figsize=(18, 10))

datasets = [
    ('Income\n(log-normal)', income),
    ('Transaction\n(exponential)', transaction),
    ('Age\n(near-normal)', age),
]

transforms = [
    ('Original',     lambda x: x),
    ('log(x)',       lambda x: np.log(x + 1e-5)),
    ('log1p(x)',     lambda x: np.log1p(x)),
    ('sqrt(x)',      lambda x: np.sqrt(np.abs(x))),
    ('Yeo-Johnson',  None),   # handled separately
]

for row, (ds_name, data) in enumerate(datasets):
    for col, (t_name, t_func) in enumerate(transforms):
        ax = axes[row, col]

        if t_name == 'Yeo-Johnson':
            pt  = PowerTransformer(method='yeo-johnson')
            x_t = pt.fit_transform(data.reshape(-1,1)).ravel()
        elif t_name == 'log(x)' and np.any(data <= 0):
            x_t = np.log(data + data[data > 0].min())
        else:
            x_t = t_func(data) if t_func else data

        skew = pd.Series(x_t).skew()
        ax.hist(x_t, bins=40, color='steelblue', alpha=0.7,
                edgecolor='white', density=True)
        color = 'green' if abs(skew) < 0.5 else ('orange' if abs(skew) < 1 else 'tomato')
        ax.set_title(f'{t_name}\nskew={skew:.2f}',
                     color=color, fontsize=8)
        ax.tick_params(labelsize=6)

        if col == 0:
            ax.set_ylabel(ds_name, fontsize=8)

plt.suptitle('Skewness Correction Transforms\n'
             'Green=good (|skew|<0.5), Orange=moderate, Red=still skewed',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/13_skew_transforms.png', dpi=120, bbox_inches='tight')
plt.show()

print("""
Skewness Treatment Decision Tree:
────────────────────────────────────────────────────────────────
Is your model a tree ensemble (RF/XGBoost)?
  YES → No transformation needed (trees are rank-invariant)
  NO  → Continue below

Is the feature right-skewed (long right tail, skew > 0)?
  Data always positive: log1p(x) — simple and effective
  Data can be zero/negative: PowerTransformer(Yeo-Johnson)
  
Is the feature left-skewed (long left tail, skew < 0)?
  Reflect first: log1p(max(x) - x), then apply above

Rule of thumb: |skew| > 1.0 → definitely transform
               0.5 < |skew| < 1.0 → consider transforming
               |skew| < 0.5 → probably fine
────────────────────────────────────────────────────────────────
""")
```

---

## 5. Handling Missing Values

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
X_orig, y = cancer.data, cancer.target

# Introduce missing values at random (MCAR)
rng = np.random.default_rng(42)
X   = X_orig.astype(float).copy()
missing_mask = rng.random(X.shape) < 0.15   # 15% missing
X[missing_mask] = np.nan

print(f"Missing values introduced: {missing_mask.sum()} "
      f"({missing_mask.mean()*100:.1f}%)")
print(f"Features with any missing: {(missing_mask.any(axis=0)).sum()}")

# ── Compare imputation strategies ────────────────────────────────────────────
strategies = {
    'No imputation\n(drop rows)': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'Mean imputation': Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'Median imputation': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'KNN imputation\n(k=5)': Pipeline([
        ('sc1', StandardScaler()),
        ('imp', KNNImputer(n_neighbors=5)),
        ('sc2', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'MICE (Iterative)': Pipeline([
        ('imp', IterativeImputer(max_iter=10, random_state=42)),
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'Complete data\n(no missing)': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
}

results_imp = {}
for name, pipe in strategies.items():
    if 'drop rows' in name:
        # Remove rows with any missing
        mask_complete = ~np.isnan(X).any(axis=1)
        cv = cross_val_score(pipe, X[mask_complete], y[mask_complete],
                             cv=5, scoring='roc_auc')
    elif 'no missing' in name.lower():
        cv = cross_val_score(pipe, X_orig, y, cv=5, scoring='roc_auc')
    else:
        cv = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
    results_imp[name] = cv

fig, ax = plt.subplots(figsize=(10, 5))
names  = list(results_imp.keys())
means  = [r.mean() for r in results_imp.values()]
stds   = [r.std()  for r in results_imp.values()]
colors = ['#EF9A9A' if 'drop' in n else
          '#A5D6A7' if 'no missing' in n.lower() else
          '#90CAF9' for n in names]
bars   = ax.barh(names, means, color=colors, alpha=0.85)
ax.errorbar(means, range(len(means)), xerr=stds, fmt='none',
            color='black', capsize=5, lw=1.5)
ax.bar_label(bars, labels=[f'{m:.4f}' for m in means], padding=4)
ax.set_xlabel('CV ROC-AUC')
ax.set_title('Imputation Strategy Comparison\n'
             '(15% random missing; Blue=imputed, Red=dropped, Green=complete)')
ax.set_xlim(0.97, 1.02)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/13_imputation_comparison.png', dpi=140, bbox_inches='tight')
plt.show()

print("\nImputation Strategy Summary:")
for name, cv in results_imp.items():
    print(f"  {name.replace(chr(10),' '):<35}: "
          f"{cv.mean():.4f} ± {cv.std():.4f}")

print("""
\nMissing Value Strategy Guide:
════════════════════════════════════════════════════════════════════
MCAR (completely random): any method works
  → Simple mean/median imputation is often sufficient

MAR (random given other variables): 
  → KNN or MICE (Iterative Imputer) — use other features to impute

MNAR (not at random, depends on missing value):
  → The fact of missingness is informative! Add binary indicator column:
    is_missing_income = income.isnull().astype(int)
  → Then impute the value itself with median

For tree models (RF, XGBoost):
  → XGBoost handles missing values natively (learns best direction)
  → RandomForest: impute first (any strategy), add missingness indicator

Rule: More sophisticated ≠ always better.
  Mean/median imputation wins more often than expected in practice.
  The difference between strategies matters most when:
    - Missing % is high (> 30%)
    - Features are highly correlated (MICE leverages this)
════════════════════════════════════════════════════════════════════
""")
```

---

## 6. Creating New Features

### 6.1 Polynomial and Interaction Features

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# ── Manual interaction features ───────────────────────────────────────────────
df_feat = df.copy()

# Domain-driven interactions
df_feat['MedInc_x_HouseAge']   = df['MedInc'] * df['HouseAge']
df_feat['MedInc_x_AveRooms']   = df['MedInc'] * df['AveRooms']
df_feat['AveRooms_x_AveOccup'] = df['AveRooms'] * df['AveOccup']

print("Manual interaction features added:")
print(f"  New features: MedInc×HouseAge, MedInc×AveRooms, AveRooms×AveOccup")

# ── sklearn PolynomialFeatures ────────────────────────────────────────────────
# Creates all polynomial combinations up to degree d
# For d=2 and 3 features [a, b, c]:
# → [1, a, b, c, a², ab, ac, b², bc, c²]

poly_demo = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_demo = np.array([[1, 2, 3]])
X_poly = poly_demo.fit_transform(X_demo)
print(f"\nPolynomialFeatures degree=2, 3 features:")
print(f"  Input:  {X_demo[0]}")
print(f"  Output: {X_poly[0]}")
print(f"  Names:  {poly_demo.get_feature_names_out(['a','b','c'])}")

# ── Feature count explosion ────────────────────────────────────────────────────
print(f"\nFeature count explosion:")
for d in range(1, 5):
    for n_feats in [3, 5, 10, 20]:
        poly = PolynomialFeatures(degree=d, include_bias=False)
        n_out = poly.fit_transform(np.zeros((1, n_feats))).shape[1]
        if d == 1 or n_feats in [3, 10]:
            print(f"  degree={d}, n_features={n_feats:2d} → {n_out:5d} features")

print("""
\nPolynomial Feature Guidelines:
  ✓ degree=2: useful for linear models to capture interactions
  ✗ degree≥3: feature explosion, always use regularization (Ridge)
  ✓ interaction_only=True: only cross-products, no x² terms
    → reduces number of features, often better for sparse data
  ✓ Always pair with Ridge/Lasso regularization
  ✗ With tree models: poly features don't help (trees create interactions naturally)
""")
```

---

### 6.2 Domain-Specific Features

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("=" * 60)
print("Domain-Specific Feature Engineering: California Housing")
print("=" * 60)

# ── Geographic features ────────────────────────────────────────────────────────
# Major California cities: (lat, lon)
city_coords = {
    'San Francisco': (37.7749, -122.4194),
    'Los Angeles':   (34.0522, -118.2437),
    'San Diego':     (32.7157, -117.1611),
    'Sacramento':    (38.5816, -121.4944),
}

for city, (lat, lon) in city_coords.items():
    col_name = f'dist_{city.replace(" ", "_")}'
    df[col_name] = np.sqrt(
        (df['Latitude'] - lat)**2 + (df['Longitude'] - lon)**2
    )
    print(f"  {col_name}: range [{df[col_name].min():.2f}, {df[col_name].max():.2f}]")

# ── Housing density ────────────────────────────────────────────────────────────
df['rooms_per_household']   = df['AveRooms']   / df['AveOccup'].clip(lower=0.1)
df['bedrooms_per_room']     = df['AveBedrms']  / df['AveRooms'].clip(lower=0.1)
df['people_per_bedroom']    = df['AveOccup']   * df['AveRooms'] / \
                               df['AveBedrms'].clip(lower=0.1)

# ── Income features ────────────────────────────────────────────────────────────
df['log_MedInc']            = np.log1p(df['MedInc'])
df['MedInc_sq']             = df['MedInc'] ** 2
df['income_per_room']       = df['MedInc'] / df['AveRooms'].clip(lower=0.1)

# ── Population density proxies ─────────────────────────────────────────────────
df['log_Population']        = np.log1p(df['Population'])
df['log_AveOccup']          = np.log1p(df['AveOccup'])

# ── Geographic clusters ────────────────────────────────────────────────────────
# Are we in the Bay Area / LA / San Diego?
df['is_bay_area'] = (
    (df['Latitude'] > 37.0) & (df['Latitude'] < 38.5) &
    (df['Longitude'] > -122.8) & (df['Longitude'] < -121.5)
).astype(int)

df['is_la_area']  = (
    (df['Latitude'] > 33.5) & (df['Latitude'] < 34.5) &
    (df['Longitude'] > -119.0) & (df['Longitude'] < -117.5)
).astype(int)

print(f"\nTotal features after engineering: {len(df.columns) - 1} "
      f"(was {len(housing.feature_names)})")

# ── Evaluate engineered features ─────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

feature_cols = [c for c in df.columns if c != 'MedHouseVal']
X_eng = df[feature_cols].values
y     = df['MedHouseVal'].values
X_raw = housing.data

for name, X_use in [('Raw features', X_raw), ('Engineered features', X_eng)]:
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    cv = cross_val_score(rf, X_use, y, cv=5, scoring='r2')
    print(f"\n{name} (d={X_use.shape[1]}): R²={cv.mean():.4f} ± {cv.std():.4f}")
```

---

### 6.3 Date and Time Features

```python
import numpy as np
import pandas as pd

# ── Generate a realistic time-series dataset ──────────────────────────────────
rng = np.random.default_rng(42)
n   = 2000
dates = pd.date_range('2020-01-01', periods=n, freq='h')

df_time = pd.DataFrame({
    'timestamp': dates,
    'sales':     (100 +
                  30 * np.sin(np.arange(n) * 2 * np.pi / 24) +   # Daily cycle
                  20 * np.sin(np.arange(n) * 2 * np.pi / 168) +  # Weekly cycle
                  rng.normal(0, 10, n))
})

print("Raw datetime column:")
print(df_time.head())

# ── Extract time features ──────────────────────────────────────────────────────
dt  = df_time['timestamp']

# Calendar features
df_time['year']          = dt.dt.year
df_time['month']         = dt.dt.month          # 1-12
df_time['day']           = dt.dt.day            # 1-31
df_time['day_of_week']   = dt.dt.dayofweek      # 0=Monday, 6=Sunday
df_time['day_of_year']   = dt.dt.dayofyear      # 1-366
df_time['week_of_year']  = dt.dt.isocalendar().week.astype(int)
df_time['quarter']       = dt.dt.quarter        # 1-4
df_time['hour']          = dt.dt.hour           # 0-23
df_time['minute']        = dt.dt.minute
df_time['is_weekend']    = (dt.dt.dayofweek >= 5).astype(int)
df_time['is_business_hour'] = (
    (dt.dt.hour >= 9) & (dt.dt.hour < 17) &
    (dt.dt.dayofweek < 5)
).astype(int)

# ── Cyclical encoding (CRITICAL for time features) ───────────────────────────
# WRONG: encoding hour as 0-23 makes 23 and 0 far apart, but midnight is close to 11pm
# RIGHT: encode as sin/cos to preserve cyclical distance

def cyclical_encode(series, max_val):
    """Encode cyclic feature as (sin, cos) pair."""
    angle = 2 * np.pi * series / max_val
    return np.sin(angle), np.cos(angle)

df_time['hour_sin'], df_time['hour_cos']             = cyclical_encode(df_time['hour'], 24)
df_time['day_of_week_sin'], df_time['day_of_week_cos'] = cyclical_encode(df_time['day_of_week'], 7)
df_time['month_sin'], df_time['month_cos']           = cyclical_encode(df_time['month'], 12)

print("\nEngineered time features:")
print(df_time[['timestamp', 'hour', 'hour_sin', 'hour_cos',
               'is_weekend', 'is_business_hour']].head(8))

# Visualize cyclical encoding
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Problem with linear encoding
ax = axes[0]
hours = np.arange(24)
ax.scatter(hours, np.zeros_like(hours), c=hours, cmap='hsv', s=100, zorder=5)
ax.set_xlabel('Hour (linear encoding)')
ax.set_yticks([])
ax.set_title('Linear Encoding: Hour\n23 and 0 are far apart\n(but midnight is close to 11pm!)')
ax.grid(True, alpha=0.3)

# Cyclical encoding
ax2 = axes[1]
h_sin, h_cos = cyclical_encode(hours, 24)
scatter = ax2.scatter(h_sin, h_cos, c=hours, cmap='hsv', s=100, zorder=5)
for h in [0, 6, 12, 18, 23]:
    ax2.annotate(f'{h}h', (h_sin[h], h_cos[h]),
                 textcoords='offset points', xytext=(5, 5), fontsize=8)
plt.colorbar(scatter, ax=ax2, label='Hour')
ax2.set_xlabel('sin(2π·hour/24)'); ax2.set_ylabel('cos(2π·hour/24)')
ax2.set_title('Cyclical Encoding: Hour\n0 and 23 are now neighbors!')
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/13_cyclical_encoding.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 6.4 Binning and Discretization

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

rng = np.random.default_rng(42)
ages = rng.normal(40, 15, 500).clip(0, 100)

# ── Equal-width bins ───────────────────────────────────────────────────────────
age_ew = pd.cut(ages, bins=5, labels=['Child','Teen','Adult','Middle','Senior'])

# ── Equal-frequency bins (quantile-based) ─────────────────────────────────────
age_eq = pd.qcut(ages, q=5, labels=['Q1','Q2','Q3','Q4','Q5'])

# ── sklearn KBinsDiscretizer ───────────────────────────────────────────────────
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
age_kbd = kbd.fit_transform(ages.reshape(-1, 1)).ravel()

print("Binning comparison:")
print(f"\nEqual-width distribution:\n{pd.Series(age_ew).value_counts().sort_index()}")
print(f"\nEqual-frequency distribution:\n{pd.Series(age_eq).value_counts().sort_index()}")

print("""
\nWhen to use binning:
  ✓ Linear models: capture non-linear effects without polynomial features
  ✓ Reduce sensitivity to outliers
  ✓ When the relationship with target is step-like (tax brackets, age groups)
  ✓ Combine with OHE for smooth non-linear effects
  ✗ Tree models: don't need binning (they find their own splits)
  ✗ Can lose information if bins are too wide
""")
```

---

## 7. Feature Selection

### 7.1 Filter Methods

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import (SelectKBest, f_classif, chi2,
                                        mutual_info_classif,
                                        VarianceThreshold)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
feature_names = cancer.feature_names

# ── Remove zero-variance features ─────────────────────────────────────────────
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)
print(f"Variance Threshold: {X.shape[1]} → {X_vt.shape[1]} features")

# ── Univariate statistical tests ──────────────────────────────────────────────
X_nonneg = MinMaxScaler().fit_transform(X)   # chi2 requires non-negative

methods_filter = {
    'ANOVA F-test':         SelectKBest(f_classif,         k='all').fit(X, y),
    'Mutual Information':   SelectKBest(mutual_info_classif, k='all').fit(X, y),
    'Chi-squared':          SelectKBest(chi2,               k='all').fit(X_nonneg, y),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, selector) in zip(axes, methods_filter.items()):
    scores = selector.scores_
    # Normalize for comparison
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    sort_idx    = np.argsort(scores_norm)[::-1]

    colors = ['steelblue' if i < 10 else 'lightblue' for i in range(len(scores))]
    ax.barh(range(len(scores)), scores_norm[sort_idx],
            color=[colors[i] for i in range(len(scores))], alpha=0.8)
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels([feature_names[i] for i in sort_idx], fontsize=6)
    ax.set_xlabel('Normalized Score')
    ax.set_title(f'{name}\n(Top 10 highlighted)')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Filter Methods: Univariate Feature Ranking\n'
             'Fast, model-agnostic, ignores feature interactions',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/13_filter_methods.png', dpi=130, bbox_inches='tight')
plt.show()

# Top features by each method
print("\nTop 5 features by each method:")
for name, selector in methods_filter.items():
    top5 = feature_names[np.argsort(selector.scores_)[::-1][:5]]
    print(f"  {name:<25}: {', '.join(top5)}")
```

---

### 7.2 Wrapper Methods

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import (RFE, RFECV, SequentialFeatureSelector)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# ── RFE: Recursive Feature Elimination ────────────────────────────────────────
# 1. Train model on all features
# 2. Remove the feature with the lowest importance
# 3. Repeat until k features remain

rfe = RFE(
    estimator=LogisticRegression(max_iter=1000, C=1.0),
    n_features_to_select=10,
    step=1                    # Remove 1 feature per iteration
)
rfe.fit(X_sc, y)
selected_rfe = cancer.feature_names[rfe.support_]
print("RFE (Logistic Regression) — Top 10 features:")
# Show ranking: 1 = selected, higher = eliminated earlier
ranking = pd.Series(rfe.ranking_, index=cancer.feature_names).sort_values()
print(ranking.head(15).to_string())

# ── RFECV: RFE with Cross-Validation ──────────────────────────────────────────
# Automatically finds the optimal number of features
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000),
    step=1,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc',
    min_features_to_select=1,
    n_jobs=-1
)
rfecv.fit(X_sc, y)
print(f"\nRFECV optimal number of features: {rfecv.n_features_}")
print(f"Selected: {cancer.feature_names[rfecv.support_]}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4))
n_features = range(1, len(rfecv.cv_results_['mean_test_score']) + 1)
mean_scores = rfecv.cv_results_['mean_test_score']
std_scores  = rfecv.cv_results_['std_test_score']

ax.plot(n_features, mean_scores, 'o-', color='steelblue', lw=2, markersize=5)
ax.fill_between(n_features,
                mean_scores - std_scores,
                mean_scores + std_scores, alpha=0.2, color='steelblue')
ax.axvline(rfecv.n_features_, color='tomato', lw=2, linestyle='--',
           label=f'Optimal n={rfecv.n_features_}')
ax.set_xlabel('Number of Features')
ax.set_ylabel('CV AUC')
ax.set_title('RFECV: Finding Optimal Feature Count Automatically')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/13_rfecv.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
Wrapper Methods Tradeoffs:
  RFE:  fast, well-understood — requires specifying n_features
  RFECV: automatically finds n_features via CV — slower
  SFS (forward/backward): flexible, works with any scorer
        forward: start empty, add best feature each step
        backward: start full, remove worst feature each step
  
  Computational cost: O(d × n_folds × model_training_time)
  → Use Embedded methods instead for large d (> 100 features)
""")
```

---

### 7.3 Embedded Methods

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
feature_names = cancer.feature_names

# ── Lasso for feature selection ────────────────────────────────────────────────
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso_cv.fit(X_sc, y.astype(float))

selected_lasso = feature_names[lasso_cv.coef_ != 0]
print(f"Lasso (optimal α={lasso_cv.alpha_:.4f}):")
print(f"  Selected {len(selected_lasso)}/{len(feature_names)} features")
print(f"  Zero coefficients: {(lasso_cv.coef_ == 0).sum()}")

# ── Random Forest importance threshold ────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)
sfm = SelectFromModel(rf, threshold='median')
sfm.fit(X, y)
selected_rf = feature_names[sfm.get_support()]
print(f"\nRandom Forest (median threshold):")
print(f"  Selected {len(selected_rf)}/{len(feature_names)} features")

# ── Compare: all features vs embedded selection ────────────────────────────────
configs = [
    ('All features',   X),
    ('Lasso selected', X_sc[:, lasso_cv.coef_ != 0]),
    ('RF median',      X[:, sfm.get_support()]),
]

print(f"\n{'Config':<25} {'n_features':>12} {'CV AUC':>10}")
print("-" * 50)
for name, X_use in configs:
    pipe = Pipeline([
        ('sc', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    cv = cross_val_score(pipe, X_use, y, cv=5, scoring='roc_auc')
    print(f"{name:<25} {X_use.shape[1]:>12} {cv.mean():>10.4f} ± {cv.std():.4f}")

# ── Visualize Lasso path ───────────────────────────────────────────────────────
from sklearn.linear_model import lasso_path
alphas, coefs, _ = lasso_path(X_sc, y.astype(float), alphas=None,
                               max_iter=5000)

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(coefs.shape[0]):
    if np.abs(coefs[i]).max() > 0.01:
        ax.semilogx(alphas, coefs[i], lw=1.5, alpha=0.8,
                    label=feature_names[i] if np.abs(coefs[i]).max() > 0.1 else '')
ax.axvline(lasso_cv.alpha_, color='black', lw=2, linestyle='--',
           label=f'Optimal α={lasso_cv.alpha_:.4f}')
ax.set_xlabel('α (regularization strength)')
ax.set_ylabel('Coefficient value')
ax.set_title('Lasso Path: Features Being Zeroed Out\n'
             'As α increases, more features become exactly 0')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/13_lasso_path.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

### 7.4 Permutation Importance

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

# Permutation importance on test set
perm_imp = permutation_importance(rf, X_te, y_te, n_repeats=30,
                                   random_state=42, scoring='roc_auc')
perm_mean = perm_imp.importances_mean
perm_std  = perm_imp.importances_std

# Sort by importance
sort_idx = np.argsort(perm_mean)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Permutation importance vs MDI
mdi_imp  = rf.feature_importances_
mdi_sort = np.argsort(mdi_imp)[::-1]

ax = axes[0]
ax.barh(range(10), perm_mean[sort_idx[:10]][::-1],
        xerr=perm_std[sort_idx[:10]][::-1],
        color='steelblue', alpha=0.8, error_kw={'capsize': 4})
ax.set_yticks(range(10))
ax.set_yticklabels([cancer.feature_names[i] for i in sort_idx[:10]][::-1], fontsize=9)
ax.set_xlabel('Mean AUC decrease when permuted')
ax.set_title('Permutation Importance\n(Test set; ±1 std across 30 permutations)')
ax.axvline(0, color='black', lw=0.8); ax.grid(axis='x', alpha=0.3)

ax2 = axes[1]
ax2.barh(range(10), mdi_imp[mdi_sort[:10]][::-1],
         color='tomato', alpha=0.8)
ax2.set_yticks(range(10))
ax2.set_yticklabels([cancer.feature_names[i] for i in mdi_sort[:10]][::-1], fontsize=9)
ax2.set_xlabel('MDI Importance (normalized)')
ax2.set_title('MDI Importance (from training)\n(Biased toward high-cardinality features)')
ax2.grid(axis='x', alpha=0.3)

plt.suptitle('Permutation Importance vs MDI: Breast Cancer RF',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/13_permutation_importance.png', dpi=130, bbox_inches='tight')
plt.show()

print("""
Which importance to use:
  MDI:         Fast (free from training), biased, training-set only
  Permutation: Slow, unbiased, test-set (or OOB), recommended for selection
  SHAP:        Best: shows direction + magnitude per sample (see Chapter 32)
""")
```

---

## 8. Automated Feature Selection with Scikit-learn

```python
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFdr, SelectFpr,
    mutual_info_classif, f_classif,
    VarianceThreshold, SelectFromModel, RFE, RFECV,
    SequentialFeatureSelector
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target

print("Feature Selection Methods Compared (CV AUC):")
print(f"\n{'Method':<40} {'n_selected':>12} {'CV AUC':>10}")
print("-" * 65)

selectors = [
    ('No selection (all 30)',
     None, X),
    ('VarianceThreshold(0.1)',
     VarianceThreshold(0.1), X),
    ('SelectKBest(f_classif, k=10)',
     SelectKBest(f_classif, k=10), X),
    ('SelectKBest(MI, k=10)',
     SelectKBest(mutual_info_classif, k=10), X),
    ('SelectFromModel(RF, median)',
     SelectFromModel(RandomForestClassifier(100, random_state=42)), X),
    ('RFE(LR, n=10)',
     RFE(LogisticRegression(max_iter=1000), n_features_to_select=10), X),
]

for name, selector, X_use in selectors:
    if selector is None:
        n_sel = X_use.shape[1]
        pipe  = Pipeline([
            ('sc', StandardScaler()),
            ('clf', RandomForestClassifier(100, random_state=42))
        ])
        cv = cross_val_score(pipe, X_use, y, cv=5, scoring='roc_auc')
    else:
        pipe = Pipeline([
            ('sc',  StandardScaler()),
            ('sel', selector),
            ('clf', RandomForestClassifier(100, random_state=42))
        ])
        pipe.fit(X_use, y)
        n_sel = pipe['sel'].transform(X_use).shape[1]
        cv = cross_val_score(pipe, X_use, y, cv=5, scoring='roc_auc')

    print(f"{name:<40} {n_sel:>12} {cv.mean():>10.4f} ± {cv.std():.4f}")
```

---

## 9. The Full Feature Engineering Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                    PowerTransformer)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Simulate a realistic messy dataset ───────────────────────────────────────
rng = np.random.default_rng(42)
n   = 1000

df_messy = pd.DataFrame({
    # Numerical with missing and skew
    'age':          np.where(rng.random(n) < 0.1, np.nan,
                             rng.integers(18, 80, n).astype(float)),
    'income':       np.where(rng.random(n) < 0.05, np.nan,
                             rng.lognormal(10.5, 0.8, n)),
    'credit_score': rng.normal(650, 80, n).clip(300, 850),

    # Categorical with missing
    'occupation':   np.where(rng.random(n) < 0.08, None,
                             rng.choice(['Tech', 'Finance', 'Healthcare',
                                         'Education', 'Retail', 'Other'], n)),
    'education':    rng.choice(['HS', 'BSc', 'MSc', 'PhD'], n,
                               p=[0.35, 0.4, 0.18, 0.07]),

    # Ordinal
    'risk_level':   rng.choice(['Low', 'Medium', 'High'], n),

    # Target
    'default':      (rng.random(n) < 0.15).astype(int),
})

# ── Define feature groups ──────────────────────────────────────────────────────
numeric_skewed  = ['income']
numeric_regular = ['age', 'credit_score']
categorical_nom = ['occupation']
categorical_ord = ['education', 'risk_level']

# ── Build preprocessing sub-pipelines ─────────────────────────────────────────
num_skewed_pipe = Pipeline([
    ('imputer',     SimpleImputer(strategy='median')),
    ('transformer', PowerTransformer(method='yeo-johnson')),
])

num_regular_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

cat_nom_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

cat_ord_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# ── ColumnTransformer: apply each pipe to the right columns ────────────────────
preprocessor = ColumnTransformer([
    ('num_skew',  num_skewed_pipe,  numeric_skewed),
    ('num_reg',   num_regular_pipe, numeric_regular),
    ('cat_nom',   cat_nom_pipe,     categorical_nom),
    ('cat_ord',   cat_ord_pipe,     categorical_ord),
], remainder='drop')

# ── Full pipeline: preprocess → select → model ────────────────────────────────
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector',     SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=42),
        threshold='median'
    )),
    ('model',        GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    )),
])

X = df_messy.drop('default', axis=1)
y = df_messy['default']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                            test_size=0.2, random_state=42)
full_pipeline.fit(X_tr, y_tr)

cv_score = cross_val_score(full_pipeline, X_tr, y_tr, cv=5, scoring='roc_auc')
test_acc  = full_pipeline.score(X_te, y_te)

print("Full Feature Engineering Pipeline:")
print(f"  CV AUC:      {cv_score.mean():.4f} ± {cv_score.std():.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"\n{classification_report(y_te, full_pipeline.predict(X_te))}")

# Pipeline summary
print("\nPipeline steps:")
for step_name, step in full_pipeline.steps:
    print(f"  [{step_name}]: {type(step).__name__}")
    if hasattr(step, 'transformers'):
        for t_name, t, cols in step.transformers:
            print(f"    → {t_name}: {cols}")
```

---

## 10. Case Study: House Price Feature Engineering

```python
# =============================================================================
# California Housing: Systematic Feature Engineering + Selection + Evaluation
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data ───────────────────────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
df      = housing.frame.copy()
feature_names = list(housing.feature_names)

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    df[feature_names].values, df['MedHouseVal'].values,
    test_size=0.2, random_state=42
)

# ── 2. Systematic feature engineering ────────────────────────────────────────
def engineer_housing_features(df_input):
    """Apply domain-driven feature engineering to California Housing."""
    df_e = pd.DataFrame(df_input, columns=feature_names)

    # Ratio features (more informative than raw counts)
    df_e['rooms_per_person']     = df_e['AveRooms']   / df_e['AveOccup'].clip(0.1)
    df_e['bedrooms_per_person']  = df_e['AveBedrms']  / df_e['AveOccup'].clip(0.1)
    df_e['bedrooms_ratio']       = df_e['AveBedrms']  / df_e['AveRooms'].clip(0.1)
    df_e['people_per_room']      = df_e['AveOccup']   / df_e['AveRooms'].clip(0.1)

    # Log transforms for skewed features
    df_e['log_population']       = np.log1p(df_e['Population'])
    df_e['log_AveOccup']         = np.log1p(df_e['AveOccup'])
    df_e['log_AveRooms']         = np.log1p(df_e['AveRooms'])

    # Income features (most predictive)
    df_e['MedInc_sq']            = df_e['MedInc'] ** 2
    df_e['MedInc_log']           = np.log1p(df_e['MedInc'])
    df_e['income_per_room']      = df_e['MedInc'] / df_e['AveRooms'].clip(0.1)

    # Geographic features
    city_coords = {
        'SF':  (37.7749, -122.4194),
        'LA':  (34.0522, -118.2437),
        'SD':  (32.7157, -117.1611),
        'SAC': (38.5816, -121.4944),
    }
    for city, (lat, lon) in city_coords.items():
        df_e[f'dist_{city}'] = np.sqrt(
            (df_e['Latitude'] - lat)**2 + (df_e['Longitude'] - lon)**2
        )

    df_e['lat_x_lon']            = df_e['Latitude'] * df_e['Longitude']
    df_e['is_coastal']           = (df_e['dist_SF'] < 1.0) | (df_e['dist_LA'] < 1.0) | \
                                    (df_e['dist_SD'] < 0.8)
    df_e['is_coastal']           = df_e['is_coastal'].astype(float)

    # Age features
    df_e['age_sq']               = df_e['HouseAge'] ** 2
    df_e['new_house']            = (df_e['HouseAge'] < 10).astype(float)
    df_e['old_house']            = (df_e['HouseAge'] > 40).astype(float)

    # Interactions
    df_e['income_x_rooms']       = df_e['MedInc'] * df_e['AveRooms']
    df_e['income_x_coastal']     = df_e['MedInc'] * df_e['is_coastal']

    return df_e.values

X_tr_eng = engineer_housing_features(X_tr_raw)
X_te_eng  = engineer_housing_features(X_te_raw)

print(f"Features: {X_tr_raw.shape[1]} (raw) → {X_tr_eng.shape[1]} (engineered)")

# ── 3. Model comparison ───────────────────────────────────────────────────────
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=5, subsample=0.8, random_state=42)

results_fe = {}
for name, X_use_tr, X_use_te in [
    ('Raw features',       X_tr_raw, X_te_raw),
    ('Engineered features', X_tr_eng, X_te_eng),
]:
    model.fit(X_use_tr, y_tr)
    y_pred = model.predict(X_use_te)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    r2     = r2_score(y_te, y_pred)
    cv     = cross_val_score(model, X_use_tr, y_tr, cv=5, scoring='r2')
    results_fe[name] = {'RMSE': rmse, 'R2': r2, 'CV_R2': cv.mean()}
    print(f"\n{name}:")
    print(f"  Test R² = {r2:.4f}, RMSE = {rmse:.4f}")
    print(f"  CV R²   = {cv.mean():.4f} ± {cv.std():.4f}")

# ── 4. Feature importance after engineering ────────────────────────────────────
all_feat_names = (feature_names +
                  [c for c in pd.DataFrame(X_tr_eng).columns
                   if c not in feature_names])
model.fit(X_tr_eng, y_tr)

# Get feature names from the engineering function
df_sample = pd.DataFrame(X_tr_raw[:5], columns=feature_names)
df_eng_sample = pd.DataFrame(engineer_housing_features(X_tr_raw[:5]))
eng_feature_names = (feature_names +
                     [f'feat_{i}' for i in range(X_tr_eng.shape[1] - len(feature_names))])

# Simpler: just label by index
mdi_imp  = model.feature_importances_
sort_idx = np.argsort(mdi_imp)[::-1][:15]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
labels = [f'feat_{i}' for i in sort_idx]
ax.barh(range(15), mdi_imp[sort_idx][::-1], color='steelblue', alpha=0.8)
ax.set_yticks(range(15))
ax.set_yticklabels(labels[::-1], fontsize=8)
ax.set_xlabel('MDI Importance')
ax.set_title('Top 15 Features After Engineering\n(GBM MDI importance)')
ax.grid(axis='x', alpha=0.3)

ax2 = axes[1]
methods_cmp = list(results_fe.keys())
r2_vals     = [results_fe[m]['R2'] for m in methods_cmp]
colors_fe   = ['#90CAF9', '#A5D6A7']
bars = ax2.bar(methods_cmp, r2_vals, color=colors_fe, alpha=0.9)
ax2.bar_label(bars, labels=[f'{v:.4f}' for v in r2_vals], padding=5)
ax2.set_ylabel('Test R² Score')
ax2.set_title('Test R² Before vs After Feature Engineering')
ax2.set_ylim(0.82, 0.90); ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Feature Engineering Case Study: California Housing',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/13_case_study.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"\nSummary:")
for name, res in results_fe.items():
    print(f"  {name:<25}: R²={res['R2']:.4f}, RMSE={res['RMSE']:.4f}")
improvement = results_fe['Engineered features']['R2'] - results_fe['Raw features']['R2']
print(f"\n  R² improvement from feature engineering: {improvement:+.4f}")
```

---

## 11. Summary

### ✅ Must-Remember Mental Models

**1. Feature engineering is the highest-ROI activity in applied ML.**
Better features → better models, often more than better algorithms. Spend 70% of your time here.

**2. Encoding selection by cardinality:**
- Binary: use as-is (0/1)
- Low (2–15): OneHotEncoder with `drop='first'`
- Medium (15–50): OHE or target encoding
- High (50+): Target encoding with CV smoothing
- Ordinal: OrdinalEncoder with explicit order

**3. Target encoding requires cross-validation.**
Fitting target statistics on the same samples you're training on leaks the target → use K-fold within training only.

**4. Scaling is mandatory for distance/gradient methods:**
KNN, SVM, PCA, logistic regression, linear regression, neural networks.
Not needed for tree models (RF, XGBoost, LightGBM).

**5. Log-transform right-skewed features.**
`np.log1p(x)` for data with zeros; `PowerTransformer(yeo-johnson)` for signed data.

**6. Cyclical features need sin/cos encoding.**
Hour, day of week, month → encode as (sin, cos) pair so that the last value is adjacent to the first.

**7. Feature selection hierarchy:**
- Filter (fast, model-agnostic): univariate tests, mutual information
- Wrapper (accurate, slow): RFE, RFECV, sequential selection
- Embedded (best balance): Lasso path, RF importance threshold
- Permutation (most reliable): shuffle each feature, measure AUC drop on test set

**8. Always use Pipeline + ColumnTransformer.**
Guarantees that all transformations are fit on training data only and correctly applied to test data. Zero data leakage possible.

**9. Domain knowledge beats algorithms.**
A feature derived from business understanding (e.g., `income_per_bedroom`) almost always outperforms a blindly generated polynomial feature.

**10. More features ≠ better model.** Irrelevant features hurt linear models and KNN. They don't hurt tree models much, but they waste computation. Always validate with cross-validation.

---

## 12. Exercises

**Conceptual:**

1. Explain why target encoding without cross-validation leads to data leakage. Give a concrete example with small numbers showing the inflated training accuracy.

2. A dataset has a categorical column `zipcode` with 1,000 unique values. You apply one-hot encoding. What are the consequences for: (a) logistic regression, (b) random forest, (c) a neural network? Which approach would you recommend for each model?

3. You have a feature `temperature` that ranges from -20°C to +45°C. Your model is KNN. Which scaler would you use and why? What if the data has 5% outliers at ±100°C?

**Coding:**

4. **Ordinal encoding pitfall**: Generate a dataset where `education` has values `['PhD', 'BSc', 'HS', 'MSc']` in random order. Show that applying `OrdinalEncoder()` without specifying `categories` gives wrong ordering. Fix it.

5. **Target encoding experiment**: On the California Housing dataset, bin `Latitude` into 20 equal-width bins (high cardinality). Compare: OHE vs frequency encoding vs target encoding (with proper CV). Evaluate with a linear model (Ridge regression). Which wins?

6. **Full pipeline**: Build a `Pipeline` + `ColumnTransformer` for the Adult Census Income dataset (from sklearn's `fetch_openml('adult')`). It has: continuous features (age, hours-per-week), ordinal (education), nominal (occupation, workclass), and missing values. Target: income > 50K. Achieve ≥ 87% CV accuracy.

7. **Feature importance comparison**: On the breast cancer dataset, compute and plot: (a) ANOVA F-test scores, (b) mutual information, (c) Lasso coefficients, (d) Random Forest MDI, (e) permutation importance. Show a heatmap of all 5 rankings. Where do they agree/disagree?

**Challenge:**

8. **Automated feature engineering with interaction search**: Implement a function `find_best_interactions(X, y, max_features=5, cv=5)` that:
   - Generates all pairwise product interactions from the top-k individual features
   - Evaluates each new interaction feature via cross-validated AUC improvement
   - Returns the top interactions sorted by improvement
   - Test on the breast cancer dataset, compare AUC before and after adding top interactions

---

## 13. References

- Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly Media. — The definitive book on the topic.
- Domingos, P. (2012). A Few Useful Things to Know About Machine Learning. *CACM*, 55(10). — Includes "more features = more problems."
- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection: A Practical Approach*. CRC Press. — Free online at feat.engineering.
- Breiman, L. (2001). Statistical Modeling: The Two Cultures. *Statistical Science*, 16(3), 199–215. — On the role of feature engineering in ML.

---

*Previous: [Chapter 12 — Dimensionality Reduction](12_dim_reduction.md)*  
*Next: [Chapter 14 — Model Evaluation, Metrics & Validation](14_evaluation.md)*

*In Chapter 14, we go deep on model evaluation: cross-validation strategies, all major metrics for regression and classification, calibration, the test set protocol, and how to avoid the most common evaluation mistakes.*

---

> **Chapter 13 complete.** All code is in `notebooks/13_feature_engineering.ipynb`.  
> The `engineer_housing_features()` function is ready to drop into any California Housing experiment.
