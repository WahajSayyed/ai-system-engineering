# Chapter 3: Data Wrangling & Exploratory Data Analysis

> *"In data science, 80% of time is spent preparing data and 20% complaining about preparing data."*  
> — A widely shared truth in the ML community

---

## Table of Contents

1. [Why EDA Is Non-Negotiable](#1-why-eda-is-non-negotiable)
2. [The EDA Mindset: Questions Before Code](#2-the-eda-mindset-questions-before-code)
3. [Loading and First-Pass Inspection](#3-loading-and-first-pass-inspection)
4. [Understanding Distributions](#4-understanding-distributions)
   - 4.1 [Univariate Analysis — Numerical Features](#41-univariate-analysis--numerical-features)
   - 4.2 [Univariate Analysis — Categorical Features](#42-univariate-analysis--categorical-features)
   - 4.3 [Detecting and Handling Skewness](#43-detecting-and-handling-skewness)
5. [Outlier Detection and Treatment](#5-outlier-detection-and-treatment)
   - 5.1 [Visual Methods](#51-visual-methods)
   - 5.2 [Statistical Methods](#52-statistical-methods)
   - 5.3 [Isolation Forest for Multivariate Outliers](#53-isolation-forest-for-multivariate-outliers)
   - 5.4 [When to Remove vs. Keep Outliers](#54-when-to-remove-vs-keep-outliers)
6. [Bivariate and Multivariate Analysis](#6-bivariate-and-multivariate-analysis)
   - 6.1 [Numerical vs. Numerical](#61-numerical-vs-numerical)
   - 6.2 [Categorical vs. Numerical](#62-categorical-vs-numerical)
   - 6.3 [Categorical vs. Categorical](#63-categorical-vs-categorical)
   - 6.4 [Correlation Analysis — Deep Dive](#64-correlation-analysis--deep-dive)
7. [Target Variable Analysis](#7-target-variable-analysis)
8. [Feature Relationships and Interactions](#8-feature-relationships-and-interactions)
9. [Data Quality: Duplicates, Inconsistencies, Leakage](#9-data-quality-duplicates-inconsistencies-leakage)
   - 9.1 [Duplicate Detection](#91-duplicate-detection)
   - 9.2 [Inconsistent Data](#92-inconsistent-data)
   - 9.3 [Data Leakage — The Silent Killer](#93-data-leakage--the-silent-killer)
10. [Building a Reusable EDA Report](#10-building-a-reusable-eda-report)
11. [Case Study: End-to-End EDA on Housing Data](#11-case-study-end-to-end-eda-on-housing-data)
12. [Summary](#12-summary)
13. [Exercises](#13-exercises)
14. [References](#14-references)

---

## 1. Why EDA Is Non-Negotiable

Consider this sequence of events, which plays out constantly in real-world ML projects:

1. A data scientist receives a dataset, splits it, trains a model, gets 98% accuracy.
2. The model is deployed. It performs terribly.
3. Investigation reveals: the training data contained a column that directly encoded the target label — one that would not be available at prediction time. The 98% accuracy was a mirage.

EDA would have caught this in 10 minutes.

**EDA is the process of systematically interrogating your data before modeling.** It answers:

- Is the data what I think it is?
- Are there quality issues (missing values, duplicates, corrupted entries)?
- What are the distributions of individual features?
- How do features relate to each other and to the target?
- Are there outliers? Are they errors or genuine extreme values?
- Is there any data leakage?
- What preprocessing will this data require?

A model is only as good as the data it trains on. The most sophisticated algorithm cannot save you from a poorly understood dataset.

---

## 2. The EDA Mindset: Questions Before Code

EDA is not about running a fixed script. It is a **directed investigation**. Every plot and statistic should answer a specific question.

Before opening a notebook, write down:

```
1. What is the prediction task?
2. What does each row represent? (a customer, a transaction, a patient visit?)
3. What does each column represent?
4. How was the data collected? (survey, sensor, database export, web scraping?)
5. Over what time period?
6. Are there known data quality issues?
7. What features do I expect to be predictive, and why?
8. What would constitute data leakage in this problem?
```

These questions shape your entire analysis. Without them, EDA becomes aimless chart generation.

---

## 3. Loading and First-Pass Inspection

**The first 10 commands you run on any new dataset — every time, without exception.**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Plotting defaults ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        100,
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'lines.linewidth':   2.0,
})

# ── Load the California Housing dataset ───────────────────────────────────────
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("=" * 60)
print("STEP 1: Shape and memory")
print("=" * 60)
print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"Memory:  {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 60)
print("STEP 2: Column names and dtypes")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("STEP 3: First 5 rows")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("STEP 4: Random sample (catches sorting artifacts)")
print("=" * 60)
print(df.sample(5, random_state=42))

print("\n" + "=" * 60)
print("STEP 5: Summary statistics")
print("=" * 60)
print(df.describe().round(3))

print("\n" + "=" * 60)
print("STEP 6: Missing values")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_count': missing,
    'missing_pct':   missing_pct
})
print(missing_df[missing_df['missing_count'] > 0])
if missing_df['missing_count'].sum() == 0:
    print("No missing values found.")

print("\n" + "=" * 60)
print("STEP 7: Target variable summary")
print("=" * 60)
target = 'MedHouseVal'
print(df[target].describe().round(3))
print(f"\nSkewness:  {df[target].skew():.3f}")
print(f"Kurtosis:  {df[target].kurt():.3f}")

print("\n" + "=" * 60)
print("STEP 8: Duplicate rows")
print("=" * 60)
n_duplicates = df.duplicated().sum()
print(f"Duplicate rows: {n_duplicates} ({n_duplicates/len(df)*100:.2f}%)")

print("\n" + "=" * 60)
print("STEP 9: Cardinality of categorical columns")
print("=" * 60)
cat_cols = df.select_dtypes(include='object').columns
if len(cat_cols) > 0:
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values")
else:
    print("No categorical columns in this dataset.")

print("\n" + "=" * 60)
print("STEP 10: Range checks — are values physically plausible?")
print("=" * 60)
print(f"MedInc range:      {df['MedInc'].min():.2f} — {df['MedInc'].max():.2f}")
print(f"HouseAge range:    {df['HouseAge'].min():.0f} — {df['HouseAge'].max():.0f} years")
print(f"AveRooms range:    {df['AveRooms'].min():.2f} — {df['AveRooms'].max():.2f}")
print(f"Population range:  {df['Population'].min():.0f} — {df['Population'].max():.0f}")
print(f"Latitude range:    {df['Latitude'].min():.2f} — {df['Latitude'].max():.2f}")
print(f"Longitude range:   {df['Longitude'].min():.2f} — {df['Longitude'].max():.2f}")
```

**Sample output:**
```
STEP 1: Shape and memory
Rows:    20,640
Columns: 9
Memory:  1.42 MB

STEP 6: Missing values
No missing values found.

STEP 7: Target variable summary
count    20640.000
mean         2.069
std          1.154
min          0.150
25%          1.196
50%          1.797
75%          2.648
max          5.000

Skewness:  0.978
```

**The anomaly you should notice immediately:** `MedHouseVal` has a max of exactly 5.000. This is suspicious — it means house values above $500k were capped at $500k in the data. This is **censored data** and affects how we model the upper end of the distribution. You would *only* notice this from careful first-pass inspection.

---

## 4. Understanding Distributions

### 4.1 Univariate Analysis — Numerical Features

For every numerical feature, you want to understand:
- **Central tendency**: mean, median, mode
- **Spread**: range, IQR, standard deviation
- **Shape**: symmetric, skewed left/right, bimodal, uniform
- **Tails**: kurtosis (how heavy are the tails?)
- **Anomalies**: unexpected spikes, truncation, impossible values

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
target_col = 'MedHouseVal'
feature_cols = [c for c in numerical_cols if c != target_col]

# ── Comprehensive distribution plot for every feature ─────────────────────────
def plot_feature_distribution(df, col, ax_hist, ax_box):
    """
    For one feature: histogram + KDE on top, box plot below.
    Annotates mean, median, skewness, kurtosis.
    """
    data = df[col].dropna()

    # Histogram + KDE
    ax_hist.hist(data, bins=50, color='steelblue', alpha=0.6,
                 density=True, edgecolor='white', linewidth=0.3)
    kde_x = np.linspace(data.min(), data.max(), 300)
    kde   = stats.gaussian_kde(data)
    ax_hist.plot(kde_x, kde(kde_x), color='steelblue', linewidth=2)

    # Mean and median lines
    ax_hist.axvline(data.mean(),   color='tomato',  linewidth=1.8,
                    linestyle='--', label=f'Mean={data.mean():.2f}')
    ax_hist.axvline(data.median(), color='green',   linewidth=1.8,
                    linestyle=':',  label=f'Median={data.median():.2f}')

    ax_hist.set_title(col)
    ax_hist.legend(fontsize=8)
    ax_hist.set_ylabel('Density')

    # Annotation box
    skew  = data.skew()
    kurt  = data.kurt()
    text  = f'skew={skew:.2f}\nkurt={kurt:.2f}'
    color = 'tomato' if abs(skew) > 1 else ('orange' if abs(skew) > 0.5 else 'green')
    ax_hist.text(0.97, 0.95, text, transform=ax_hist.transAxes,
                 fontsize=8, va='top', ha='right',
                 bbox=dict(boxstyle='round', fc=color, alpha=0.3))

    # Box plot
    bp = ax_box.boxplot(data, vert=False, patch_artist=True,
                        widths=0.6, notch=True,
                        boxprops=dict(facecolor='steelblue', alpha=0.5),
                        medianprops=dict(color='tomato', linewidth=2),
                        flierprops=dict(marker='.', markersize=3, alpha=0.3))
    ax_box.set_xlabel(col)
    ax_box.set_yticks([])


n_features = len(feature_cols)
fig, axes = plt.subplots(n_features, 2, figsize=(12, n_features * 3.5))

for i, col in enumerate(feature_cols):
    plot_feature_distribution(df, col, axes[i, 0], axes[i, 1])

plt.suptitle('Univariate Feature Distributions — California Housing',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/03_univariate_distributions.png', dpi=130, bbox_inches='tight')
plt.show()

# ── Tabular summary ───────────────────────────────────────────────────────────
print("\nDistribution Summary:")
print(f"{'Feature':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8} {'Assessment'}")
print("-" * 80)

for col in feature_cols + [target_col]:
    data = df[col].dropna()
    skew = data.skew()
    kurt = data.kurt()
    if   abs(skew) > 1:   assessment = "⚠ Highly skewed"
    elif abs(skew) > 0.5: assessment = "~ Moderate skew"
    else:                 assessment = "✓ Approx. normal"
    print(f"{col:<20} {data.mean():>8.3f} {data.median():>8.3f} "
          f"{data.std():>8.3f} {skew:>8.3f} {kurt:>8.3f}  {assessment}")
```

---

### 4.2 Univariate Analysis — Categorical Features

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use the Adult Income dataset (census data) — rich in categorical features
from sklearn.datasets import fetch_openml

adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')
df_adult = adult.data.copy()
df_adult['income'] = (adult.target == '>50K').astype(int)

categorical_cols = df_adult.select_dtypes(include='category').columns.tolist()
print(f"Categorical columns: {categorical_cols}")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(categorical_cols[:9]):
    vc = df_adult[col].value_counts()

    # Truncate to top 10 categories for readability
    if len(vc) > 10:
        vc = vc.head(10)
        axes[i].set_title(f'{col} (top 10)')
    else:
        axes[i].set_title(col)

    bars = axes[i].barh(vc.index.astype(str), vc.values,
                        color='steelblue', alpha=0.75)
    axes[i].bar_label(bars, fmt='%d', padding=3, fontsize=8)
    axes[i].set_xlabel('Count')

    # Annotate cardinality and mode
    mode_val  = df_adult[col].mode()[0]
    mode_pct  = df_adult[col].value_counts(normalize=True).iloc[0] * 100
    axes[i].text(0.97, 0.02,
                 f'Cardinality: {df_adult[col].nunique()}\nMode: "{mode_val}" ({mode_pct:.1f}%)',
                 transform=axes[i].transAxes, fontsize=7.5,
                 va='bottom', ha='right',
                 bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Categorical Feature Distributions — Adult Income Dataset',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_categorical_distributions.png', dpi=130, bbox_inches='tight')
plt.show()

# ── Key metrics for categorical features ──────────────────────────────────────
print(f"\n{'Column':<20} {'Cardinality':>12} {'Mode':<25} {'Mode %':>8} {'Entropy':>8}")
print("-" * 78)

for col in categorical_cols:
    vc   = df_adult[col].value_counts(normalize=True)
    mode = vc.index[0]
    entropy = stats.entropy(vc.values, base=2)
    print(f"{col:<20} {df_adult[col].nunique():>12} {str(mode):<25} "
          f"{vc.iloc[0]*100:>7.1f}% {entropy:>8.3f}")
```

**What entropy tells you:**
- **Low entropy** (close to 0): one category dominates. The feature may not be informative.
- **High entropy**: categories are evenly distributed. The feature is maximally informative for splitting.

---

### 4.3 Detecting and Handling Skewness

Skewed distributions can hurt linear models, distance-based models, and neural networks. Tree-based models (Random Forests, XGBoost) are immune — they only use rank ordering, not absolute values.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PowerTransformer

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# ── The three most useful transforms for right-skewed data ────────────────────
skewed_cols = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
col = 'Population'   # Most skewed in the dataset
data = df[col].values

# Box-Cox requires strictly positive values
# Yeo-Johnson works for any values (including zero and negative)

transforms = {
    'Original':             data,
    'log1p':                np.log1p(data),
    'sqrt':                 np.sqrt(data),
    'Box-Cox':              stats.boxcox(data + 1)[0],     # +1 to avoid log(0)
    'Yeo-Johnson':          PowerTransformer(method='yeo-johnson').fit_transform(
                                data.reshape(-1, 1)).ravel(),
}

fig, axes = plt.subplots(2, 5, figsize=(18, 7))

for i, (name, transformed) in enumerate(transforms.items()):
    ax_hist = axes[0, i]
    ax_qq   = axes[1, i]

    # Histogram
    ax_hist.hist(transformed, bins=50, color='steelblue', alpha=0.7,
                 density=True, edgecolor='white', linewidth=0.3)
    skew = pd.Series(transformed).skew()
    ax_hist.set_title(f'{name}\nskew = {skew:.3f}',
                      color='green' if abs(skew) < 0.5 else 'tomato')
    ax_hist.set_ylabel('Density' if i == 0 else '')

    # Q-Q plot — the most diagnostic tool for normality
    # Points on the diagonal → data is normally distributed
    (osm, osr), (slope, intercept, r) = stats.probplot(transformed)
    ax_qq.scatter(osm, osr, s=4, alpha=0.4, color='steelblue')
    ax_qq.plot([osm[0], osm[-1]],
               [slope*osm[0]+intercept, slope*osm[-1]+intercept],
               color='tomato', linewidth=1.5)
    ax_qq.set_xlabel('Theoretical quantiles')
    ax_qq.set_ylabel('Sample quantiles' if i == 0 else '')
    ax_qq.set_title(f'Q-Q Plot (r={r:.3f})')

plt.suptitle(f'Transform Comparison — {col}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_skewness_transforms.png', dpi=130, bbox_inches='tight')
plt.show()

# ── Automated skewness detection and recommendation ───────────────────────────
print(f"\n{'Feature':<20} {'Skewness':>10} {'Recommendation'}")
print("-" * 60)

for col in df.columns[:-1]:   # All features except target
    skew = df[col].skew()
    if   abs(skew) < 0.5:  rec = "None needed"
    elif abs(skew) < 1.0:  rec = "Consider log1p or sqrt"
    elif abs(skew) < 2.0:  rec = "Apply log1p or Box-Cox"
    else:                  rec = "⚠ Strong skew — log1p required"
    direction = "right ►" if skew > 0 else "◄ left"
    print(f"{col:<20} {skew:>8.3f} {direction}   {rec}")
```

**When to transform and when not to:**

| Model Type | Transform needed? | Reason |
|---|---|---|
| Linear Regression | Yes | Assumes normally distributed residuals |
| Logistic Regression | Yes | Improves convergence and decision boundaries |
| KNN / SVM | Yes | Distance-based: scale and shape matter |
| Neural Networks | Yes | Gradient flow benefits from normalized inputs |
| Decision Tree | No | Only uses rank order of values |
| Random Forest | No | Ensemble of trees — rank-invariant |
| XGBoost / LightGBM | No | Tree-based — rank-invariant |

---

## 5. Outlier Detection and Treatment

### 5.1 Visual Methods

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, col in enumerate(df.columns[:-1]):
    ax = axes[i // 4, i % 4]

    # Box plot with outlier dots
    bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                    notch=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.5),
                    medianprops=dict(color='tomato', linewidth=2.5),
                    flierprops=dict(marker='.', markersize=3,
                                    alpha=0.4, color='gray'))

    # Count outliers using IQR rule
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    pct   = n_out / len(df) * 100

    ax.set_title(f'{col}\n{n_out} outliers ({pct:.1f}%)',
                 fontsize=10,
                 color='tomato' if pct > 5 else 'black')
    ax.set_xticks([])

plt.suptitle('Outlier Detection via Box Plots (IQR method)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_outliers_boxplot.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

### 5.2 Statistical Methods

Three complementary statistical methods for flagging outliers:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

def detect_outliers_summary(series, name="feature"):
    """
    Apply four outlier detection methods and compare their results.
    """
    data = series.dropna().values
    n    = len(data)
    results = {}

    # ── Method 1: Z-score (assumes normality) ─────────────────────────────────
    # |z| > 3  →  outlier  (3 std devs from the mean)
    # Only reliable when the distribution is approximately normal
    z_scores = np.abs(stats.zscore(data))
    z_mask   = z_scores > 3
    results['Z-score (|z|>3)'] = z_mask.sum()

    # ── Method 2: Modified Z-score (uses median — robust to outliers) ──────────
    # Better than z-score because it doesn't use the mean/std, which are themselves
    # sensitive to outliers.
    # |Mz| > 3.5  →  outlier
    median    = np.median(data)
    mad       = np.median(np.abs(data - median))   # Median Absolute Deviation
    mod_z     = 0.6745 * (data - median) / (mad + 1e-10)
    mod_mask  = np.abs(mod_z) > 3.5
    results['Modified Z-score'] = mod_mask.sum()

    # ── Method 3: IQR rule (Tukey's fences) ───────────────────────────────────
    # Classic boxplot outlier criterion. Non-parametric — no distribution assumption.
    # Lower fence = Q1 - 1.5*IQR, Upper fence = Q3 + 1.5*IQR
    Q1  = np.percentile(data, 25)
    Q3  = np.percentile(data, 75)
    IQR = Q3 - Q1
    iqr_mask = (data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)
    results['IQR (1.5×)'] = iqr_mask.sum()

    # ── Method 4: Extreme IQR (3×IQR) ────────────────────────────────────────
    # More conservative — only flags truly extreme values
    ext_mask = (data < Q1 - 3*IQR) | (data > Q3 + 3*IQR)
    results['IQR (3×) extreme'] = ext_mask.sum()

    print(f"\n{'─'*50}")
    print(f" Feature: {name}  (n={n:,})")
    print(f"{'─'*50}")
    for method, count in results.items():
        pct = count / n * 100
        bar = '█' * int(pct * 2)
        print(f"  {method:<25}: {count:>5,} ({pct:>5.2f}%) {bar}")

    return results

# Apply to AveOccup — the most outlier-prone feature
detect_outliers_summary(df['AveOccup'],    "AveOccup")
detect_outliers_summary(df['Population'],  "Population")
detect_outliers_summary(df['AveRooms'],    "AveRooms")


# ── Visualizing the outlier boundary ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

col  = 'AveOccup'
data = df[col].values
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1

# Left: scatter of all values with bounds highlighted
sorted_idx = np.argsort(data)
axes[0].scatter(range(len(data)), data[sorted_idx], s=2, alpha=0.3, color='steelblue')
axes[0].axhline(Q3 + 1.5*IQR, color='orange',  linewidth=1.5, linestyle='--', label='1.5×IQR')
axes[0].axhline(Q3 + 3.0*IQR, color='tomato',  linewidth=1.5, linestyle='--', label='3.0×IQR')
axes[0].set_title(f'{col} — Sorted values with outlier bounds')
axes[0].set_xlabel('Rank'); axes[0].set_ylabel(col)
axes[0].set_ylim(-1, 40)   # Zoom in — some values go to 1243!
axes[0].legend()

# Right: histogram zoomed in
axes[1].hist(data[data <= 10], bins=60, color='steelblue', alpha=0.7,
             density=True, edgecolor='white')
axes[1].axvline(Q3 + 1.5*IQR, color='orange', linewidth=2, linestyle='--',
                label=f'1.5×IQR bound = {Q3+1.5*IQR:.1f}')
axes[1].axvline(Q3 + 3.0*IQR, color='tomato',  linewidth=2, linestyle='--',
                label=f'3.0×IQR bound = {Q3+3.0*IQR:.1f}')
axes[1].set_title(f'{col} Distribution (zoomed to ≤10)')
axes[1].set_xlabel(col); axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/03_outlier_boundaries.png', dpi=130, bbox_inches='tight')
plt.show()

print(f"\nAveOccup max value: {data.max():.1f}")
print("Context: This represents 1,243 people per bedroom — almost certainly an error.")
print("A block of commercial properties with 1 occupant each, grouped incorrectly.")
```

---

### 5.3 Isolation Forest for Multivariate Outliers

The methods above detect outliers in **one dimension at a time**. But in ML, we care about **multivariate outliers** — points that are unusual given the combination of all features.

**Isolation Forest** works by:
1. Randomly selecting a feature and a random split point.
2. Recursively splitting until each point is isolated.
3. Outliers require fewer splits to isolate (they are in sparse regions).
4. Anomaly score = average path length (shorter path = more anomalous).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

features_for_od = df.drop('MedHouseVal', axis=1).values

# Standardize before Isolation Forest (good practice even though IF is tree-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_for_od)

# contamination: expected proportion of outliers
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,   # Flag ~2% of data as outliers
    random_state=42,
    n_jobs=-1
)
outlier_labels = iso_forest.fit_predict(X_scaled)   # -1 = outlier, 1 = inlier
anomaly_scores = iso_forest.decision_function(X_scaled)  # More negative = more anomalous

# How many outliers?
n_outliers = (outlier_labels == -1).sum()
print(f"Isolation Forest detected {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")

# ── Visualize: anomaly score vs two key features ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

is_outlier = outlier_labels == -1

axes[0].scatter(df.loc[~is_outlier, 'AveOccup'],
                df.loc[~is_outlier, 'MedInc'],
                c=anomaly_scores[~is_outlier], cmap='Blues', s=5, alpha=0.4,
                label='Inlier')
sc = axes[0].scatter(df.loc[is_outlier, 'AveOccup'],
                     df.loc[is_outlier, 'MedInc'],
                     c='tomato', s=25, alpha=0.7, zorder=5, label='Outlier (IF)')
axes[0].set_xlabel('AveOccup'); axes[0].set_ylabel('MedInc')
axes[0].set_xlim(0, 20); axes[0].set_ylim(0, 16)
axes[0].set_title('Isolation Forest Outliers\n(AveOccup vs MedInc)')
axes[0].legend()

# Distribution of anomaly scores
axes[1].hist(anomaly_scores, bins=60, color='steelblue', alpha=0.7,
             density=True, edgecolor='white')
threshold = np.percentile(anomaly_scores, 2)
axes[1].axvline(threshold, color='tomato', linewidth=2, linestyle='--',
                label=f'2% threshold = {threshold:.3f}')
axes[1].set_xlabel('Anomaly Score (more negative = more anomalous)')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Isolation Forest Anomaly Scores')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/03_isolation_forest.png', dpi=130, bbox_inches='tight')
plt.show()

# Compare features of outliers vs inliers
df['is_outlier'] = is_outlier
comparison = df.groupby('is_outlier').mean().round(3).T
comparison.columns = ['Inlier mean', 'Outlier mean']
comparison['ratio'] = (comparison['Outlier mean'] / comparison['Inlier mean']).round(2)
print("\nInlier vs Outlier feature comparison:")
print(comparison.sort_values('ratio', ascending=False))
```

---

### 5.4 When to Remove vs. Keep Outliers

This is not a purely statistical decision. It requires domain knowledge.

```
Decision framework for outlier treatment:
─────────────────────────────────────────────────────────────────

Q1: Is this value physically/logically impossible?
    (e.g., age = -5, income = $1 trillion)
    → YES: Fix or remove it. It's a data error.

Q2: Is this a measurement/recording error?
    (e.g., sensor glitch, data entry typo — "1243 occupants per room")
    → YES: Remove or impute. Document the action.

Q3: Is the outlier a genuine extreme value?
    (e.g., a billionaire in an income dataset, a 100-year-old in age data)
    → YES: Keep it. Cap it with a winsorization if needed.
    Your model must handle real-world extremes.

Q4: Will this outlier distort the model for the majority of cases?
    (e.g., linear regression is highly sensitive to extreme values)
    → YES: Consider winsorizing or using a robust model.
    → NO (tree-based models): safe to leave.

Q5: Does the outlier represent the population you care about?
    (e.g., fraud detection — the fraud cases ARE the outliers)
    → YES: Never remove them. They are the signal.
─────────────────────────────────────────────────────────────────
```

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# ── Winsorization: cap extreme values at a percentile ─────────────────────────
# A softer alternative to removal — keeps the data point but limits its influence.

def winsorize_column(series, lower_pct=0.01, upper_pct=0.99):
    """Cap values below lower_pct and above upper_pct quantiles."""
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower=lower, upper=upper)

df_clean = df.copy()
for col in ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']:
    original_range = f"{df[col].min():.1f} — {df[col].max():.1f}"
    df_clean[col]  = winsorize_column(df[col], 0.01, 0.99)
    new_range      = f"{df_clean[col].min():.1f} — {df_clean[col].max():.1f}"
    print(f"{col:<15}: {original_range:>25}  →  {new_range}")

# ── Log-transform instead of capping: preserves relative differences ──────────
# Preferred when the outliers represent genuine extreme values
df_clean['AveOccup_log']    = np.log1p(df['AveOccup'])
df_clean['Population_log']  = np.log1p(df['Population'])

print(f"\nAveOccup_log skewness:   {df_clean['AveOccup_log'].skew():.3f}  "
      f"(original: {df['AveOccup'].skew():.3f})")
print(f"Population_log skewness: {df_clean['Population_log'].skew():.3f}  "
      f"(original: {df['Population'].skew():.3f})")
```

---

## 6. Bivariate and Multivariate Analysis

### 6.1 Numerical vs. Numerical

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# ── Scatter plot matrix with marginal distributions ───────────────────────────
# The pairplot is the most information-dense single EDA plot.
# For n features, it shows n*(n-1)/2 scatter plots simultaneously.

# Select a subset of features for readability
plot_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']

# Cap outliers just for visualization
df_viz = df[plot_cols].copy()
df_viz['AveOccup'] = df_viz['AveOccup'].clip(upper=6)
df_viz['AveRooms'] = df_viz['AveRooms'].clip(upper=10)

# Color by target: discretize MedHouseVal into 3 terciles
df_viz['value_tier'] = pd.qcut(df_viz['MedHouseVal'], q=3,
                                labels=['Low', 'Medium', 'High'])

g = sns.pairplot(df_viz, hue='value_tier', diag_kind='kde',
                 vars=plot_cols[:-1],
                 plot_kws={'alpha': 0.3, 's': 15},
                 diag_kws={'fill': True, 'alpha': 0.5},
                 palette={'Low': '#F44336', 'Medium': '#FF9800', 'High': '#4CAF50'})

g.fig.suptitle('Pairplot — California Housing Features (colored by price tier)',
               y=1.02, fontsize=13, fontweight='bold')
plt.savefig('figures/03_pairplot.png', dpi=120, bbox_inches='tight')
plt.show()

# ── Individual scatter with regression line + correlation ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

pairs = [('MedInc', 'MedHouseVal'),
         ('AveRooms', 'MedHouseVal'),
         ('HouseAge', 'MedHouseVal')]

for ax, (x_col, y_col) in zip(axes, pairs):
    # Subsample for speed
    idx = np.random.default_rng(42).choice(len(df), 2000, replace=False)
    x = df[x_col].iloc[idx].clip(upper=df[x_col].quantile(0.99))
    y = df[y_col].iloc[idx]

    # Pearson correlation
    r, p = stats.pearsonr(x, y)

    # Spearman correlation (rank-based, robust to outliers and non-linearity)
    rho, _ = stats.spearmanr(x, y)

    ax.scatter(x, y, alpha=0.3, s=8, color='steelblue')

    # Regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m*x_line + b, color='tomato', linewidth=2)

    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}\nPearson r={r:.3f}, Spearman ρ={rho:.3f}')

plt.tight_layout()
plt.savefig('figures/03_scatter_regression.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

### 6.2 Categorical vs. Numerical

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from scipy import stats

adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')
df_adult = adult.data.copy()
df_adult['income'] = (adult.target == '>50K').astype(int)

# Convert categoricals to strings for clean plotting
for col in df_adult.select_dtypes('category').columns:
    df_adult[col] = df_adult[col].astype(str)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── 1. Violin plot: hours-per-week by education ────────────────────────────────
edu_order = df_adult.groupby('education')['hours-per-week'].median().sort_values().index
sns.violinplot(data=df_adult, x='education', y='hours-per-week',
               order=edu_order, palette='Blues_d',
               inner='quartile', cut=0, ax=axes[0, 0])
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(),
                            rotation=45, ha='right', fontsize=8)
axes[0, 0].set_title('Hours per Week by Education Level')

# ── 2. Box plot with strip: age by work class ──────────────────────────────────
workclass_order = df_adult.groupby('workclass')['age'].median().sort_values().index
sns.boxplot(data=df_adult, x='workclass', y='age', order=workclass_order,
            palette='Set2', ax=axes[0, 1])
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(),
                             rotation=45, ha='right', fontsize=8)
axes[0, 1].set_title('Age Distribution by Work Class')

# ── 3. Mean comparison with 95% CI (bar + error bars) ─────────────────────────
income_by_edu = df_adult.groupby('education')['income'].agg(['mean', 'sem', 'count'])
income_by_edu = income_by_edu.sort_values('mean', ascending=True)
ci95 = 1.96 * income_by_edu['sem']

axes[1, 0].barh(income_by_edu.index, income_by_edu['mean'],
                xerr=ci95, color='steelblue', alpha=0.7,
                error_kw={'ecolor': 'black', 'capsize': 3})
axes[1, 0].set_xlabel('>$50K income rate')
axes[1, 0].set_title('Income Rate by Education (±95% CI)')
axes[1, 0].set_xlim(0, 1)

# ── 4. ANOVA test: is the group difference statistically significant? ──────────
# H0: all group means are equal
# If p < 0.05, at least one group has a different mean — feature is informative.
groups = [df_adult.loc[df_adult['education']==edu, 'hours-per-week'].values
          for edu in df_adult['education'].unique()]
f_stat, p_value = stats.f_oneway(*groups)
eta_sq = (f_stat * (len(groups) - 1)) / (f_stat * (len(groups) - 1) + len(df_adult) - len(groups))

axes[1, 1].axis('off')
text = (
    f"One-way ANOVA:\nEducation → Hours per Week\n\n"
    f"F-statistic: {f_stat:.2f}\n"
    f"p-value:     {p_value:.2e}\n"
    f"η² (effect): {eta_sq:.4f}\n\n"
    f"Interpretation:\n"
    f"{'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05\n"
    f"Education explains {eta_sq*100:.1f}% of\nvariance in hours-per-week."
)
axes[1, 1].text(0.1, 0.5, text, transform=axes[1, 1].transAxes,
                fontsize=11, va='center',
                bbox=dict(boxstyle='round', fc='lightyellow', ec='gray'))
axes[1, 1].set_title('Statistical Test: ANOVA')

plt.suptitle('Categorical vs. Numerical Bivariate Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_cat_vs_num.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

### 6.3 Categorical vs. Categorical

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, cramers_v

# Note: cramers_v is in scipy.stats from version 1.7+
# For older versions: compute manually (shown in the function below)

def cramers_v_manual(x, y):
    """Cramér's V: measure of association between two categorical variables. Range: [0, 1]."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n    = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr    = r - ((r-1)**2)/(n-1)
    kcorr    = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Use Adult dataset
adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')
df_a = adult.data.copy()
df_a['income'] = adult.target.astype(str)

for col in df_a.select_dtypes('category').columns:
    df_a[col] = df_a[col].astype(str)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ── 1. Stacked bar: income rate by occupation ──────────────────────────────────
ct = pd.crosstab(df_a['occupation'], df_a['income'], normalize='index') * 100
ct = ct.sort_values('>50K', ascending=True)
ct.plot(kind='barh', stacked=True, ax=axes[0],
        color=['#90CAF9', '#EF9A9A'])
axes[0].set_title('Income by Occupation (% of each occupation)')
axes[0].set_xlabel('Percentage')
axes[0].legend(title='Income', loc='lower right')

# ── 2. Heatmap of cross-tabulation ────────────────────────────────────────────
ct2 = pd.crosstab(df_a['education'].replace({
    'Preschool':'<HS', '1st-4th':'<HS', '5th-6th':'<HS', '7th-8th':'<HS',
    '9th':'HS-some', '10th':'HS-some', '11th':'HS-some', '12th':'HS-some',
    'HS-grad':'HS-grad'
}), df_a['income'])

ct2_pct = ct2.div(ct2.sum(axis=1), axis=0) * 100
sns.heatmap(ct2_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            ax=axes[1], linewidths=0.5,
            cbar_kws={'label': '% with >50K income'})
axes[1].set_title('Education vs. Income (% >50K)')

# ── 3. Cramér's V matrix for all categorical features ─────────────────────────
cat_cols_small = ['workclass', 'education', 'marital-status', 'occupation',
                  'relationship', 'race', 'sex']
v_matrix = pd.DataFrame(np.zeros((len(cat_cols_small), len(cat_cols_small))),
                         index=cat_cols_small, columns=cat_cols_small)
for c1 in cat_cols_small:
    for c2 in cat_cols_small:
        v_matrix.loc[c1, c2] = cramers_v_manual(df_a[c1], df_a[c2])

mask = np.triu(np.ones_like(v_matrix, dtype=bool))
sns.heatmap(v_matrix, annot=True, fmt='.2f', cmap='Blues',
            mask=mask, vmin=0, vmax=1, ax=axes[2],
            linewidths=0.5, square=True)
axes[2].set_title("Cramér's V — Categorical Associations")

plt.suptitle('Categorical vs. Categorical Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_cat_vs_cat.png', dpi=130, bbox_inches='tight')
plt.show()

# Chi-square test for independence
chi2, p, dof, expected = chi2_contingency(pd.crosstab(df_a['sex'], df_a['income']))
print(f"\nChi-square test: sex vs income")
print(f"  χ² = {chi2:.2f},  dof = {dof},  p = {p:.2e}")
print(f"  {'Dependent (associated)' if p < 0.05 else 'Independent'} at α=0.05")
print(f"  Cramér's V = {cramers_v_manual(df_a['sex'], df_a['income']):.4f}")
print(f"  (V > 0.3 = strong, 0.1-0.3 = moderate, < 0.1 = weak)")
```

---

### 6.4 Correlation Analysis — Deep Dive

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# ── 1. Pearson, Spearman, and Kendall — when to use each ──────────────────────
print("Correlation with target (MedHouseVal):")
print(f"\n{'Feature':<20} {'Pearson':>10} {'Spearman':>10} {'Kendall':>10} {'Interpretation'}")
print("-" * 75)

for col in df.columns[:-1]:
    r_pearson,  _ = stats.pearsonr(df[col],  df['MedHouseVal'])
    r_spearman, _ = stats.spearmanr(df[col], df['MedHouseVal'])
    r_kendall,  _ = stats.kendalltau(df[col].sample(2000, random_state=42),
                                      df['MedHouseVal'].sample(2000, random_state=42))

    # If Spearman >> Pearson: monotonic but non-linear relationship
    # If Kendall ≈ Spearman: consistent, robust result
    delta = abs(r_spearman) - abs(r_pearson)
    if   delta > 0.1:  note = "Non-linear monotonic"
    elif delta > 0.05: note = "Mildly non-linear"
    else:              note = "Approximately linear"

    print(f"{col:<20} {r_pearson:>10.4f} {r_spearman:>10.4f} "
          f"{r_kendall:>10.4f}  {note}")

# ── 2. Full correlation matrix — three representations ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

corr_pearson  = df.corr(method='pearson')
corr_spearman = df.corr(method='spearman')

# Mask upper triangle
mask = np.triu(np.ones_like(corr_pearson, dtype=bool))

kws = dict(annot=True, fmt='.2f', linewidths=0.5,
           vmin=-1, vmax=1, square=True, mask=mask)

sns.heatmap(corr_pearson,  cmap='RdBu_r', ax=axes[0], **kws)
axes[0].set_title('Pearson Correlation\n(linear relationships)')

sns.heatmap(corr_spearman, cmap='RdBu_r', ax=axes[1], **kws)
axes[1].set_title('Spearman Correlation\n(monotonic relationships)')

# Difference matrix: where do they disagree?
diff = (corr_spearman - corr_pearson).abs()
sns.heatmap(diff, cmap='Oranges', ax=axes[2], annot=True, fmt='.2f',
            linewidths=0.5, square=True, mask=mask, vmin=0, vmax=0.3)
axes[2].set_title('|Spearman - Pearson|\n(high = non-linear relationship)')

plt.suptitle('Correlation Analysis — Pearson vs Spearman',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_correlation_comparison.png', dpi=130, bbox_inches='tight')
plt.show()

# ── 3. Multicollinearity detection — Variance Inflation Factor (VIF) ──────────
# High VIF (>5-10) means a feature is nearly redundant with other features.
# Multicollinearity doesn't hurt prediction, but hurts model interpretability.
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df.drop('MedHouseVal', axis=1).assign(const=1)  # Add intercept column
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns[:-1]   # Exclude constant
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1] - 1)]
vif_data = vif_data.sort_values('VIF', ascending=False)

print("\nVariance Inflation Factors (VIF):")
print(f"{'Feature':<20} {'VIF':>8}  {'Assessment'}")
print("-" * 50)
for _, row in vif_data.iterrows():
    if   row['VIF'] > 10: assessment = "🔴 High multicollinearity"
    elif row['VIF'] > 5:  assessment = "🟡 Moderate"
    else:                  assessment = "✓  Low"
    print(f"{row['Feature']:<20} {row['VIF']:>8.2f}  {assessment}")
```

---

## 7. Target Variable Analysis

Understanding your target is as important as understanding your features.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing, load_breast_cancer

# ── Case 1: Regression target ──────────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
df_h = housing.frame.copy()
target_reg = 'MedHouseVal'

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Distribution + normality tests
axes[0, 0].hist(df_h[target_reg], bins=60, color='steelblue', alpha=0.7,
                density=True, edgecolor='white')
axes[0, 0].set_title(f'{target_reg} Distribution')
axes[0, 0].set_xlabel('Median House Value ($100k)')

# Log-transformed
axes[0, 1].hist(np.log(df_h[target_reg]), bins=60, color='steelblue', alpha=0.7,
                density=True, edgecolor='white')
axes[0, 1].set_title(f'log({target_reg}) Distribution')

# Q-Q plot
(osm, osr), (slope, intercept, r) = stats.probplot(df_h[target_reg])
axes[0, 2].scatter(osm, osr, s=3, alpha=0.3, color='steelblue')
axes[0, 2].plot([osm[0], osm[-1]], [slope*osm[0]+intercept, slope*osm[-1]+intercept],
                color='tomato', linewidth=2)
axes[0, 2].set_title(f'Q-Q Plot (r={r:.4f})')
axes[0, 2].set_xlabel('Theoretical quantiles')

# ── Check for target capping ───────────────────────────────────────────────────
# This is a famous data quality issue in the California Housing dataset
cap_val = df_h[target_reg].max()
n_capped = (df_h[target_reg] == cap_val).sum()
print(f"Target max value: {cap_val}")
print(f"Values at maximum: {n_capped} ({n_capped/len(df_h)*100:.2f}%) ← CENSORED")
print("Recommendation: Remove or model these separately.")

# ── Case 2: Binary classification target ──────────────────────────────────────
cancer = load_breast_cancer(as_frame=True)
df_c   = cancer.frame.copy()
y      = cancer.target

print(f"\n{'='*50}")
print("Class imbalance analysis — Breast Cancer dataset")
print(f"{'='*50}")
vc     = pd.Series(y).value_counts()
vc_pct = vc / len(y) * 100

print(vc)
print(f"\nImbalance ratio: {vc.iloc[0] / vc.iloc[1]:.2f}:1")

# Compute optimal random baseline accuracy
majority_class_pct = vc_pct.iloc[0]
print(f"\n⚠  Baseline accuracy (always predict majority): {majority_class_pct:.1f}%")
print("   A model that only achieves this is useless.")

axes[1, 0].bar([cancer.target_names[0], cancer.target_names[1]],
               vc.values,
               color=['#EF9A9A', '#A5D6A7'])
axes[1, 0].set_title('Class Distribution — Breast Cancer')
axes[1, 0].set_ylabel('Count')
for i, (v, p) in enumerate(zip(vc.values, vc_pct.values)):
    axes[1, 0].text(i, v + 5, f'{p:.1f}%', ha='center', fontsize=11)

# ── Feature-target relationship for top features ──────────────────────────────
top_features = (df_c.drop('target', axis=1)
                .apply(lambda col: abs(stats.pointbiserialr(col, df_c['target'])[0]))
                .sort_values(ascending=False)
                .head(10))

axes[1, 1].barh(top_features.index[::-1], top_features.values[::-1],
                color='steelblue', alpha=0.8)
axes[1, 1].set_xlabel('|Point-Biserial Correlation|')
axes[1, 1].set_title('Top 10 Features Correlated with Target')
axes[1, 1].set_xlim(0, 1)

# ── KDE of top feature split by class ─────────────────────────────────────────
best_feature = top_features.index[0]
for label, color in [(0, '#EF9A9A'), (1, '#A5D6A7')]:
    mask = df_c['target'] == label
    sns.kdeplot(df_c.loc[mask, best_feature], ax=axes[1, 2],
                label=cancer.target_names[label], fill=True, alpha=0.4, color=color)
axes[1, 2].set_title(f'KDE — {best_feature}\nby class')
axes[1, 2].legend()

plt.suptitle('Target Variable Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_target_analysis.png', dpi=130, bbox_inches='tight')
plt.show()
```

---

## 8. Feature Relationships and Interactions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# ── Mutual Information: non-linear feature relevance ─────────────────────────
# Mutual Information measures how much knowing X reduces uncertainty about y.
# It captures non-linear relationships — unlike Pearson correlation.
# MI = 0: X tells us nothing about y
# MI > 0: larger = more informative

from sklearn.feature_selection import mutual_info_regression

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

mi_scores = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MI vs Pearson comparison
pearson_corr = X.corrwith(y).abs().sort_values(ascending=False)

ax = axes[0]
x_pos = np.arange(len(mi_series))
width = 0.35
bars1 = ax.bar(x_pos - width/2, mi_series[X.columns],
               width, label='Mutual Information', color='steelblue', alpha=0.8)
ax2   = ax.twinx()
bars2 = ax2.bar(x_pos + width/2, pearson_corr[X.columns],
                width, label='|Pearson r|', color='tomato', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(X.columns, rotation=45, ha='right')
ax.set_ylabel('Mutual Information', color='steelblue')
ax2.set_ylabel('|Pearson Correlation|', color='tomato')
ax.set_title('Mutual Information vs. Pearson Correlation\n(captures linear vs. all relationships)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# ── Interaction effects: feature × feature → target ──────────────────────────
# When feature A × feature B is more predictive than either alone

# Example: MedInc × Latitude interaction
# High income + Southern CA vs Northern CA behaves differently
df['MedInc_x_Lat'] = df['MedInc'] * df['Latitude']

# Scatter: binned AveRooms vs MedHouseVal, split by income quartile
income_quartile = pd.qcut(df['MedInc'], q=4, labels=['Q1\n(lowest)', 'Q2', 'Q3', 'Q4\n(highest)'])
ax2 = axes[1]
colors_q = ['#EF9A9A', '#FFE082', '#A5D6A7', '#90CAF9']

for q, color in zip(['Q1\n(lowest)', 'Q2', 'Q3', 'Q4\n(highest)'], colors_q):
    mask   = income_quartile == q
    rooms  = df.loc[mask, 'AveRooms'].clip(upper=8)
    values = df.loc[mask, 'MedHouseVal']
    ax2.scatter(rooms, values, alpha=0.2, s=5, color=color, label=q)

ax2.set_xlabel('Average Rooms per Household')
ax2.set_ylabel('Median House Value ($100k)')
ax2.set_title('Rooms vs. Price — Interaction with Income\n(same rooms, different price by income)')
ax2.legend(title='Income Quartile', loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('figures/03_feature_interactions.png', dpi=130, bbox_inches='tight')
plt.show()

# ── Permutation Importance (model-based feature relevance) ────────────────────
# More reliable than tree node impurity (which biases towards high-cardinality features)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

perm_imp = permutation_importance(rf, X_test, y_test,
                                   n_repeats=10, random_state=42, n_jobs=-1)
perm_series = pd.Series(perm_imp.importances_mean, index=X.columns)
perm_std    = pd.Series(perm_imp.importances_std,  index=X.columns)
perm_sorted = perm_series.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(perm_sorted.index[::-1], perm_sorted.values[::-1],
        xerr=perm_std[perm_sorted.index[::-1]].values,
        color='steelblue', alpha=0.8, error_kw={'capsize': 4})
ax.set_xlabel('Mean decrease in R² score')
ax.set_title('Permutation Importance (Random Forest)\n±1 std across 10 permutations')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('figures/03_permutation_importance.png', dpi=130, bbox_inches='tight')
plt.show()

print("\nPermutation Importance:")
for feat in perm_sorted.index:
    bar = '█' * max(0, int(perm_series[feat] * 100))
    print(f"  {feat:<20}: {perm_series[feat]:>7.4f} ± {perm_std[feat]:.4f}  {bar}")
```

---

## 9. Data Quality: Duplicates, Inconsistencies, Leakage

### 9.1 Duplicate Detection

```python
import numpy as np
import pandas as pd

# Create a dataset with deliberate quality issues
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'customer_id': np.arange(n),
    'age':         np.random.randint(18, 80, n),
    'income':      np.random.normal(60000, 20000, n),
    'product':     np.random.choice(['A', 'B', 'C'], n),
    'purchase':    np.random.randint(0, 2, n)
})

# Inject exact duplicates
df = pd.concat([df, df.iloc[10:15]], ignore_index=True)

# Inject near-duplicates (same customer, different records)
near_dup = df.iloc[20:25].copy()
near_dup['income'] += np.random.normal(0, 100, 5)  # Tiny income difference
df = pd.concat([df, near_dup], ignore_index=True)

# ── Exact duplicate detection ──────────────────────────────────────────────────
n_exact = df.duplicated().sum()
print(f"Exact duplicates: {n_exact}")

# Which rows are duplicated?
dupes = df[df.duplicated(keep=False)].sort_values('customer_id')
print(f"\nDuplicated rows (showing all occurrences):\n{dupes.head(10)}")

# Remove exact duplicates
df_deduped = df.drop_duplicates()
print(f"\nAfter removing exact duplicates: {len(df)} → {len(df_deduped)} rows")

# ── Near-duplicate detection (same key, slightly different values) ─────────────
n_near = df.duplicated(subset=['customer_id']).sum()
print(f"\nNear-duplicates (same customer_id): {n_near}")

# Decide: which record to keep? Most recent? Average the values?
# Strategy: keep the record with the highest income (or most recent timestamp)
df_deduped2 = df.sort_values('income', ascending=False).drop_duplicates(
    subset=['customer_id'], keep='first'
)
print(f"After removing near-duplicates: {len(df)} → {len(df_deduped2)} rows")
```

---

### 9.2 Inconsistent Data

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age':     [25, -5, 150, 30, 45, 200],
    'income':  [50000, 80000, -1000, 60000, 90000, 0],
    'gender':  ['Male', 'male', 'MALE', 'Female', 'F', 'M'],
    'zip_code':['12345', '1234', 'ABCDE', '90210', '10001', '00000'],
    'email':   ['a@b.com', 'invalid', 'c@d.com', 'e@f.com', 'no_at_sign', 'g@h.com'],
})

print("Raw data:\n", df, "\n")

# ── Rule-based validation ──────────────────────────────────────────────────────
issues = []

# Age: must be [0, 120]
invalid_age = (df['age'] < 0) | (df['age'] > 120)
print(f"Invalid ages: {invalid_age.sum()} rows — {df.loc[invalid_age, 'age'].tolist()}")
issues.append(('age', 'out_of_range', invalid_age.sum()))

# Income: must be non-negative
invalid_income = df['income'] < 0
print(f"Negative income: {invalid_income.sum()} rows")
issues.append(('income', 'negative', invalid_income.sum()))

# Gender: standardize
df['gender_clean'] = (df['gender']
                       .str.lower()
                       .str.strip()
                       .replace({'m': 'male', 'f': 'female'}))
print(f"\nGender before: {df['gender'].unique()}")
print(f"Gender after:  {df['gender_clean'].unique()}")

# Zip code: must be 5 digits
import re
valid_zip = df['zip_code'].str.match(r'^\d{5}$')
print(f"\nInvalid zip codes: {(~valid_zip).sum()}")

# Email: must contain @
valid_email = df['email'].str.contains('@', na=False)
print(f"Invalid emails: {(~valid_email).sum()}")

# ── Summary of data quality issues ────────────────────────────────────────────
print(f"\n{'─'*50}")
print("DATA QUALITY REPORT")
print(f"{'─'*50}")
print(f"Total rows: {len(df)}")
print(f"Invalid ages:    {invalid_age.sum()} ({invalid_age.mean()*100:.1f}%)")
print(f"Negative income: {invalid_income.sum()} ({invalid_income.mean()*100:.1f}%)")
print(f"Invalid zips:    {(~valid_zip).sum()} ({(~valid_zip).mean()*100:.1f}%)")
print(f"Invalid emails:  {(~valid_email).sum()} ({(~valid_email).mean()*100:.1f}%)")
```

---

### 9.3 Data Leakage — The Silent Killer

Data leakage occurs when information that would not be available at prediction time is used during training. It produces optimistic evaluation metrics that completely fail in production.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)
n = 1000

# Simulate a credit scoring dataset
df = pd.DataFrame({
    'credit_score':    np.random.randint(300, 850, n),
    'debt_ratio':      np.random.uniform(0.1, 0.8, n),
    'income':          np.random.normal(60000, 20000, n).clip(10000),
    'num_late_payments': np.random.poisson(1, n),
})

# True target: default probability
p_default = 1 / (1 + np.exp(-(
    -2 + 0.01 * (600 - df['credit_score']) +
    2 * df['debt_ratio'] +
    0.5 * df['num_late_payments']
)))
df['defaulted'] = (np.random.uniform(0, 1, n) < p_default).astype(int)

# ── LEAKED FEATURE: 'recovery_amount' is only known AFTER default ─────────────
# In reality, you can't know recovery amount before the loan decision.
df['recovery_amount'] = np.where(
    df['defaulted'] == 1,
    np.random.uniform(1000, 20000, n),
    0.0
)

# ── LEAKED FEATURE: 'account_closed' — a consequence of defaulting ─────────────
df['account_closed'] = np.where(
    df['defaulted'] == 1,
    np.random.binomial(1, 0.85, n),
    np.random.binomial(1, 0.05, n)
)

print("Data sample:")
print(df.head())
print(f"\nDefault rate: {df['defaulted'].mean():.2%}")

# ── Scenario 1: WITH leakage ───────────────────────────────────────────────────
features_with_leak = ['credit_score', 'debt_ratio', 'income', 'num_late_payments',
                       'recovery_amount', 'account_closed']

pipe_leak = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(random_state=42))
])

X_leak = df[features_with_leak]
y      = df['defaulted']
scores_leak = cross_val_score(pipe_leak, X_leak, y, cv=5, scoring='roc_auc')
print(f"\n[WITH LEAKAGE]    CV ROC-AUC: {scores_leak.mean():.4f} ± {scores_leak.std():.4f}")
print("                  ^^ Suspiciously high — looks great, totally wrong.")

# ── Scenario 2: WITHOUT leakage ────────────────────────────────────────────────
features_clean = ['credit_score', 'debt_ratio', 'income', 'num_late_payments']

pipe_clean = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(random_state=42))
])

X_clean = df[features_clean]
scores_clean = cross_val_score(pipe_clean, X_clean, y, cv=5, scoring='roc_auc')
print(f"[WITHOUT LEAKAGE] CV ROC-AUC: {scores_clean.mean():.4f} ± {scores_clean.std():.4f}")
print("                  ^^ Honest estimate of real-world performance.")

print(f"\nLeakage inflated ROC-AUC by: {scores_leak.mean() - scores_clean.mean():.4f}")

# ── Common sources of leakage — checklist ─────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════╗
║          DATA LEAKAGE CHECKLIST                                 ║
╠══════════════════════════════════════════════════════════════════╣
║ ✗ Features derived FROM the target                             ║
║   (e.g., recovery_amount, account_closed)                      ║
║                                                                 ║
║ ✗ Future data in the training set                              ║
║   (e.g., using price at t+1 to predict direction at t)         ║
║                                                                 ║
║ ✗ Fitting preprocessors on the full dataset                    ║
║   (e.g., StandardScaler fit before train/test split)           ║
║                                                                 ║
║ ✗ Target encoding without cross-validation                     ║
║   (e.g., replacing category with target mean using all data)   ║
║                                                                 ║
║ ✗ Using a unique identifier as a feature                       ║
║   (e.g., customer ID encodes time of joining → leaks recency)  ║
║                                                                 ║
║ ✗ Temporal leakage: training on future events                  ║
║   (e.g., fraud label assigned retroactively, used in train set)║
╚══════════════════════════════════════════════════════════════════╝
""")
```

---

## 10. Building a Reusable EDA Report

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def eda_report(df, target_col=None, max_cat_unique=20, figsize_base=5):
    """
    Automated EDA report generator.

    Parameters
    ----------
    df          : pd.DataFrame
    target_col  : str or None — the target column (for supervised tasks)
    max_cat_unique : int — max unique values to treat a column as categorical
    figsize_base   : int — base figure size multiplier
    """

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Also treat low-cardinality numeric columns as categorical
    for col in num_cols[:]:
        if df[col].nunique() <= max_cat_unique and col != target_col:
            cat_cols.append(col)
            num_cols.remove(col)

    if target_col and target_col in num_cols:
        num_cols.remove(target_col)
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)

    # ── Section 1: Overview ─────────────────────────────────────────────────────
    print("=" * 65)
    print("  EDA REPORT")
    print("=" * 65)
    print(f"  Shape:        {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory:       {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Numerical:    {len(num_cols)} features")
    print(f"  Categorical:  {len(cat_cols)} features")
    print(f"  Target:       {target_col or 'None specified'}")
    print(f"  Duplicates:   {df.duplicated().sum():,} rows ({df.duplicated().mean()*100:.2f}%)")

    # ── Section 2: Missing values ────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  MISSING VALUES")
    print(f"{'─'*65}")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            pct = count / len(df) * 100
            bar = '█' * int(pct / 2)
            flag = " ⚠" if pct > 20 else ""
            print(f"  {col:<25} {count:>6,} ({pct:>5.1f}%) {bar}{flag}")
    else:
        print("  ✓ No missing values detected.")

    # ── Section 3: Numerical feature summary ─────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  NUMERICAL FEATURES")
    print(f"{'─'*65}")
    print(f"  {'Feature':<22} {'Mean':>9} {'Std':>9} {'Skew':>7} {'Outliers%':>10}")
    print(f"  {'─'*60}")
    for col in num_cols:
        data = df[col].dropna()
        skew = data.skew()
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        n_out = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
        pct_out = n_out / len(data) * 100
        skew_flag = " ⚠" if abs(skew) > 1 else ""
        out_flag  = " ⚠" if pct_out > 10 else ""
        print(f"  {col:<22} {data.mean():>9.3f} {data.std():>9.3f} "
              f"{skew:>6.3f}{skew_flag} {pct_out:>8.1f}%{out_flag}")

    # ── Section 4: Categorical feature summary ────────────────────────────────────
    if cat_cols:
        print(f"\n{'─'*65}")
        print("  CATEGORICAL FEATURES")
        print(f"{'─'*65}")
        print(f"  {'Feature':<22} {'Cardinality':>13} {'Mode':<20} {'Mode%':>7}")
        print(f"  {'─'*60}")
        for col in cat_cols:
            vc      = df[col].value_counts(normalize=True)
            card    = df[col].nunique()
            mode    = str(vc.index[0])[:18]
            mode_pct= vc.iloc[0] * 100
            flag    = " ⚠ high cardinality" if card > 50 else ""
            print(f"  {col:<22} {card:>13} {mode:<20} {mode_pct:>6.1f}%{flag}")

    # ── Section 5: Target summary ─────────────────────────────────────────────────
    if target_col:
        print(f"\n{'─'*65}")
        print(f"  TARGET: {target_col}")
        print(f"{'─'*65}")
        target = df[target_col].dropna()
        if target.dtype in [np.float64, np.float32, np.int64] and target.nunique() > 10:
            print(f"  Type:    Regression (continuous)")
            print(f"  Mean:    {target.mean():.4f}")
            print(f"  Std:     {target.std():.4f}")
            print(f"  Skew:    {target.skew():.4f}")
            print(f"  Range:   [{target.min():.4f}, {target.max():.4f}]")
        else:
            print(f"  Type:    Classification (discrete)")
            vc = target.value_counts()
            for cls, count in vc.items():
                pct = count / len(target) * 100
                bar = '█' * int(pct / 2)
                print(f"  Class {cls}: {count:>6,} ({pct:>5.1f}%) {bar}")
            imbalance = vc.iloc[0] / vc.iloc[-1]
            if imbalance > 3:
                print(f"\n  ⚠ Class imbalance ratio: {imbalance:.1f}:1 — handle carefully")

    # ── Section 6: Correlation with target ────────────────────────────────────────
    if target_col and target_col in df.columns:
        target_num = pd.to_numeric(df[target_col], errors='coerce')
        print(f"\n{'─'*65}")
        print(f"  FEATURE-TARGET CORRELATIONS (Pearson)")
        print(f"{'─'*65}")
        corrs = {}
        for col in num_cols:
            r, p = stats.pearsonr(df[col].fillna(df[col].median()), target_num.fillna(0))
            corrs[col] = (r, p)
        corrs_sorted = sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
        for col, (r, p) in corrs_sorted:
            sig   = "✓" if p < 0.05 else "✗"
            bar   = '█' * int(abs(r) * 20)
            sign  = '+' if r > 0 else '-'
            print(f"  {sig} {col:<22} r={r:>+7.4f}  {'p<0.05' if p<0.05 else 'n.s.'}  {sign}{bar}")

    print(f"\n{'='*65}")
    print("  END OF REPORT")
    print(f"{'='*65}")


# Run on California Housing
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df_h = housing.frame.copy()
eda_report(df_h, target_col='MedHouseVal')
```

---

## 11. Case Study: End-to-End EDA on Housing Data

```python
# =============================================================================
# Complete EDA: From raw data to modeling-ready conclusions
# Dataset: California Housing
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("─" * 60)
print("CALIFORNIA HOUSING — EDA SUMMARY")
print("─" * 60)
print(f"""
Context:
  Each row = one census block group in California (1990 census)
  Prediction task: regression — predict median house value

Columns:
  MedInc     — Median income of block group (in $10,000s)
  HouseAge   — Median house age (years)
  AveRooms   — Average number of rooms per household
  AveBedrms  — Average number of bedrooms per household
  Population — Block group population
  AveOccup   — Average number of occupants per household
  Latitude   — Geographic latitude
  Longitude  — Geographic longitude
  MedHouseVal— Median house value (in $100,000s) ← TARGET
""")

# ── Finding 1: Geographic clustering is extremely strong ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sc = axes[0].scatter(df['Longitude'], df['Latitude'],
                     c=df['MedHouseVal'], cmap='viridis',
                     s=2, alpha=0.4, vmin=0.5, vmax=5)
plt.colorbar(sc, ax=axes[0], label='Median House Value ($100k)')
axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude')
axes[0].set_title('House Value by Geographic Location\n(San Francisco Bay = expensive cluster)')

sc2 = axes[1].scatter(df['Longitude'], df['Latitude'],
                      c=df['MedInc'], cmap='plasma',
                      s=2, alpha=0.4)
plt.colorbar(sc2, ax=axes[1], label='Median Income ($10k)')
axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude')
axes[1].set_title('Income by Geographic Location\n(Income and value track closely geographically)')

plt.suptitle('Geographic Analysis — California Housing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_geographic_analysis.png', dpi=130, bbox_inches='tight')
plt.show()

# ── Finding 2: Target is right-skewed and censored ────────────────────────────
print("\nFinding 2: Target (MedHouseVal) analysis")
cap_val   = df['MedHouseVal'].max()
n_capped  = (df['MedHouseVal'] == cap_val).sum()
print(f"  Max value: {cap_val} (appears {n_capped} times = {n_capped/len(df)*100:.1f}% of data)")
print(f"  Skewness: {df['MedHouseVal'].skew():.3f} — log transform recommended for linear models")

# ── Finding 3: AveOccup has extreme outliers ──────────────────────────────────
print("\nFinding 3: AveOccup extreme outliers")
Q1, Q3 = df['AveOccup'].quantile([0.25, 0.75])
IQR = Q3 - Q1
n_out = (df['AveOccup'] > Q3 + 1.5*IQR).sum()
print(f"  AveOccup max: {df['AveOccup'].max():.1f} occupants/room")
print(f"  IQR outliers: {n_out} ({n_out/len(df)*100:.1f}%)")
print(f"  Action: winsorize at 99th percentile OR log-transform")

# ── Finding 4: MedInc is the strongest predictor ──────────────────────────────
print("\nFinding 4: Feature-target correlations (Pearson, Spearman)")
for col in df.columns[:-1]:
    r_p, _ = stats.pearsonr(df[col],  df['MedHouseVal'])
    r_s, _ = stats.spearmanr(df[col], df['MedHouseVal'])
    print(f"  {col:<15}: Pearson={r_p:>+.4f},  Spearman={r_s:>+.4f}")

# ── Finding 5: Geography needs encoding ───────────────────────────────────────
print("\nFinding 5: Latitude/Longitude")
print("  Raw lat/lon are not directly useful for linear models.")
print("  Recommendations:")
print("    - Use cluster labels from K-Means (k=10-20 geographic clusters)")
print("    - Or: compute distance to major city centers (SF, LA, SD)")
print("    - Or: bin into grid cells")
print("    - Or: use lat × lon interaction term")

# ── Preprocessing recommendations ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREPROCESSING RECOMMENDATIONS")
print("=" * 60)
recommendations = {
    'MedInc':     'Slightly right-skewed. Try log1p for linear models.',
    'HouseAge':   'Uniform-ish. Cap capped values (50yr cap visible).',
    'AveRooms':   'Right-skewed. Winsorize at 99th pct.',
    'AveBedrms':  'Right-skewed. Winsorize at 99th pct.',
    'Population': 'Highly right-skewed. Apply log1p.',
    'AveOccup':   'Extremely right-skewed. Apply log1p + winsorize.',
    'Latitude':   'Bimodal (LA vs SF cluster). Engineer cluster features.',
    'Longitude':  'Bimodal. Engineer cluster features.',
    'MedHouseVal':'Censored at $500k. Apply log transform for linear models.',
}
for feat, rec in recommendations.items():
    print(f"  {feat:<15}: {rec}")
```

---

## 12. Summary

### ✅ Must-Remember Mental Models

**1. The 10 first-pass commands — run these before any modeling, every time:**
`shape` → `dtypes` → `head()` → `sample()` → `describe()` → `isnull().sum()` → `target distribution` → `duplicated().sum()` → `cardinality check` → `range plausibility`

**2. Three outlier detection methods and when to use them:**
- **Z-score**: only when distribution is approximately normal
- **IQR (Tukey's fences)**: always safe, non-parametric
- **Isolation Forest**: multivariate outliers — catches unusual *combinations* of values, not just extreme values in one dimension

**3. Skewness treatment table:**
- |skew| < 0.5: no transform needed
- 0.5 < |skew| < 1.0: consider log1p or sqrt
- |skew| > 1.0: apply log1p or Box-Cox
- Tree-based models: never need transforms (rank-invariant)

**4. Three correlation measures and when each applies:**
- **Pearson r**: linear relationships, normally distributed data
- **Spearman ρ**: monotonic (not necessarily linear), robust to outliers
- **Cramér's V**: association between two categorical variables
- When Spearman >> Pearson: the relationship is non-linear but monotonic

**5. VIF for multicollinearity:**
- VIF > 10: high multicollinearity — consider removing one of the correlated features
- Multicollinearity doesn't hurt prediction accuracy, but destroys interpretability of coefficients

**6. Data leakage sources — memorize these:**
- Features derived from the target (consequences, not causes)
- Future information in the training set
- Fitting preprocessors on the full dataset before splitting
- Temporal leakage (future data predicting past outcomes)

**7. Mutual Information vs. Pearson correlation:**
- Pearson: captures linear relationships only. Can be 0 even for a strong non-linear relationship.
- MI: captures any statistical dependence. Better for non-linear feature selection.

**8. EDA findings → preprocessing decisions:**
- Right-skewed numerical feature → log1p (for linear/distance models)
- Outliers → winsorize or log-transform (not blind removal)
- Missing values → impute (median for numeric, mode for categorical) + add indicator column
- High cardinality categorical → target encoding or embedding, not one-hot
- Geographic features → engineer distance/cluster features

---

## 13. Exercises

**Comprehension:**

1. You are given a dataset where a numerical feature has Pearson correlation of 0.02 with the target, but Mutual Information of 0.43. Draw a scatter plot that could produce this situation. What kind of relationship does it suggest?

2. Your colleague says: "I fit the StandardScaler on the entire dataset before the train-test split, but the scaler doesn't change the values of the test set because it just subtracts the mean — so there's no leakage." Are they correct? Explain precisely where the leakage occurs.

3. A feature has 30% missing values. List three questions you should ask before deciding how to handle the missingness.

**Coding:**

4. Load the `fetch_openml('titanic', version=1)` dataset. Perform a complete first-pass inspection (all 10 steps from Section 3). Write a one-paragraph EDA summary documenting your key findings and the preprocessing steps you would recommend.

5. For the breast cancer dataset (`load_breast_cancer`): detect multivariate outliers using Isolation Forest with contamination=0.05. Visualize the outliers projected onto the top-2 PCA components. Do the detected outliers share any characteristics (e.g., specific class, extreme feature values)?

6. Implement a function `compute_associations(df)` that:
   - For numerical pairs: computes Spearman correlation
   - For categorical pairs: computes Cramér's V
   - For numerical-categorical pairs: computes eta-squared (ω²) from ANOVA
   - Returns a single n×n association matrix
   - Visualizes it as a heatmap
   
   Apply it to the Adult dataset.

7. Create a temporal leakage example from scratch:
   - Generate 500 rows of simulated loan data with a `date` column spanning 2 years
   - Add a feature `month_default_rate` = average default rate for that month (computed over all rows including future months)
   - Train a logistic regression WITH and WITHOUT this feature, compare CV AUC scores
   - Show how to correctly compute `month_default_rate` using only historical data (no future leakage)

**Challenge:**

8. Build a complete, automated EDA pipeline that:
   - Accepts any Pandas DataFrame and optional target column
   - Automatically chooses the right analysis for each column pair (numerical/numerical, numerical/categorical, categorical/categorical)
   - Generates a full figure with labeled subplots
   - Returns a Python dictionary with: missing %, outlier %, skewness, top correlated features, recommended preprocessing steps
   - Test it on at least 3 different datasets (California Housing, Adult Income, Breast Cancer)

---

## 14. References

- Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley. — The foundational text.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.). Chapter 2: End-to-End ML Project.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *IEEE ICDM*.
- Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality. *Biometrika*, 52(3–4), 591–611.
- Pandas documentation: https://pandas.pydata.org/docs/user_guide/index.html
- Seaborn documentation: https://seaborn.pydata.org/
- SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html

---

*Previous: [Chapter 2 — Python & the Scientific Stack](02_python_stack.md)*  
*Next: [Chapter 4 — Mathematics for Machine Learning](04_math.md)*

*In Chapter 4, we build the mathematical foundation every ML practitioner needs: linear algebra (vectors, matrices, eigendecomposition), calculus (gradients, the chain rule), and probability & statistics (distributions, Bayes' theorem, MLE). We derive, not just describe.*

---

> **Chapter 3 complete.** All code is in `notebooks/03_eda.ipynb`. Every function is self-contained and runnable on the provided Scikit-learn datasets — no external data downloads required.
