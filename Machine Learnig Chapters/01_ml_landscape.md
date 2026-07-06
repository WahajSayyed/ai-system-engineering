# Chapter 1: The Machine Learning Landscape

> *"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."*
> — Tom Mitchell, 1997

---

## Table of Contents

1. [What Is Machine Learning?](#1-what-is-machine-learning)
2. [Why Machine Learning? The Motivation](#2-why-machine-learning-the-motivation)
3. [Types of Machine Learning Systems](#3-types-of-machine-learning-systems)
   - 3.1 [Supervised Learning](#31-supervised-learning)
   - 3.2 [Unsupervised Learning](#32-unsupervised-learning)
   - 3.3 [Semi-Supervised Learning](#33-semi-supervised-learning)
   - 3.4 [Self-Supervised Learning](#34-self-supervised-learning)
   - 3.5 [Reinforcement Learning](#35-reinforcement-learning)
4. [The Machine Learning Workflow](#4-the-machine-learning-workflow)
5. [Generalization: The Central Problem](#5-generalization-the-central-problem)
6. [The Bias-Variance Tradeoff](#6-the-bias-variance-tradeoff)
7. [No Free Lunch Theorem](#7-no-free-lunch-theorem)
8. [The Modern ML Stack](#8-the-modern-ml-stack)
9. [Your First End-to-End ML Program](#9-your-first-end-to-end-ml-program)
10. [Summary](#10-summary)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. What Is Machine Learning?

Traditional programming works like this:

```
Rules + Data → Computer → Answers
```

You explicitly write down every rule. If you want to detect spam emails, you write rules like:
- If the email contains "FREE MONEY", mark as spam.
- If the sender is unknown and the subject is in ALL CAPS, mark as spam.

This works for simple cases, but fails at scale. Language evolves. Spammers adapt. You can't write 10,000 rules and maintain them forever.

Machine learning inverts this paradigm:

```
Data + Answers → Machine Learning → Rules (a model)
```

You feed the machine *examples* — thousands of emails labeled "spam" or "not spam" — and the algorithm figures out the rules on its own. These learned rules are called a **model** or **hypothesis**.

More formally: **Machine learning is the study of algorithms that improve their performance on a task through experience (data), without being explicitly programmed for that task.**

This is not magic. A machine learning model is, at its core, a mathematical function:

```
ŷ = f(x; θ)
```

Where:
- `x` is the input (features, e.g., word counts in an email)
- `ŷ` is the predicted output (spam or not)
- `θ` (theta) are the **parameters** — the numbers the algorithm learns from data
- `f` is the model family (e.g., logistic regression, decision tree, neural network)

Learning = finding the values of `θ` that make `f(x; θ)` give the right answers as often as possible.

---

## 2. Why Machine Learning? The Motivation

Let's make this concrete with three examples that illustrate *why* we need ML:

### Example 1: Image Recognition

Consider classifying whether a photo contains a cat. The pixel values of a 224×224 RGB image form a vector of 224 × 224 × 3 = **150,528 numbers**. Writing rules over raw pixels is intractable. A convolutional neural network learns hierarchical features — edges → textures → parts → objects — directly from millions of labeled images.

### Example 2: Language Understanding

Human language is ambiguous, context-dependent, and constantly evolving. "The bank was steep" vs "The bank was closed" — the word "bank" means something completely different. No rule-based system handles this gracefully at scale. Language models trained on billions of text tokens learn these contextual representations automatically.

### Example 3: Personalization

Netflix has 260+ million subscribers. Writing explicit rules for which movie to recommend to each person is impossible. A recommendation system learns patterns from viewing history — "users who watched A and B also enjoyed C" — and generalizes to new users.

**In all three cases: the pattern is too complex or too high-dimensional for hand-crafted rules. ML finds the pattern automatically from data.**

---

## 3. Types of Machine Learning Systems

ML systems are classified along several axes. The most important is: **what kind of supervision is available during training?**

### 3.1 Supervised Learning

The training dataset consists of **input-output pairs**: `{(x⁽¹⁾, y⁽¹⁾), (x⁽²⁾, y⁽²⁾), ..., (x⁽ⁿ⁾, y⁽ⁿ⁾)}`.

The label `y` is provided by a human (or an automated process). The algorithm learns a mapping from inputs to outputs.

Two sub-types based on the nature of `y`:

#### Regression
`y` is a **continuous number**.

- Predicting house prices from square footage, location, and number of rooms.
- Predicting tomorrow's temperature.
- Predicting a patient's blood glucose level.

#### Classification
`y` is a **discrete category**.

- **Binary**: Spam or not spam. Malignant or benign tumor.
- **Multi-class**: Digit recognition (0–9). Species identification (cat, dog, bird).
- **Multi-label**: A photo can contain both "beach" AND "sunset" simultaneously.

**Key algorithms:** Linear/Logistic Regression, Decision Trees, Random Forests, SVMs, Neural Networks.

---

### 3.2 Unsupervised Learning

There are **no labels**. You only have `{x⁽¹⁾, x⁽²⁾, ..., x⁽ⁿ⁾}`. The algorithm must find structure, patterns, or compact representations on its own.

**Clustering**: Group similar data points together.
- Customer segmentation: group customers by purchasing behavior without being told what the groups are.
- K-Means, DBSCAN, Hierarchical Clustering.

**Dimensionality Reduction**: Compress high-dimensional data into fewer dimensions while preserving structure.
- PCA reduces a 1000-feature dataset to 2 dimensions for visualization.
- UMAP preserves local and global structure for exploratory analysis.

**Density Estimation**: Learn the underlying probability distribution of the data.
- Anomaly detection: a data point with very low probability is likely anomalous.

**Generative Modeling**: Learn to *generate* new data that looks like the training data.
- VAEs, Diffusion Models, GANs.

---

### 3.3 Semi-Supervised Learning

You have a **small amount of labeled data** and a **large amount of unlabeled data**. Labels are expensive (a radiologist labeling MRI scans = costly). Unlabeled data is cheap.

Semi-supervised learning leverages the structure in unlabeled data to improve the model trained on labeled data.

**Example**: You have 100 labeled chest X-rays and 10,000 unlabeled ones. A semi-supervised approach first learns useful visual representations from all 10,100 images, then fine-tunes on the 100 labeled ones. Performance far exceeds training on 100 labels alone.

---

### 3.4 Self-Supervised Learning

A special case of unsupervised learning where **labels are generated automatically from the data itself**. No human annotation required.

**Example — Masked Language Modeling (BERT)**:
Take the sentence: *"The cat sat on the [MASK]."*
The model must predict the masked word ("mat"). The "label" was created by masking a word from existing text — no human labeled anything.

**Example — Contrastive Learning (SimCLR, CLIP)**:
Two augmented views of the same image are created. The model is trained to produce similar representations for augmented versions of the same image, and dissimilar representations for different images.

Self-supervised learning is responsible for the most powerful pretrained models today: GPT, BERT, CLIP, DINO, and more. **It scales with data in a way that supervised learning cannot.**

---

### 3.5 Reinforcement Learning

An **agent** interacts with an **environment**, takes **actions**, and receives **rewards** (or penalties). The goal is to learn a **policy** — a mapping from states to actions — that maximizes cumulative reward over time.

```
State (s) → Agent → Action (a) → Environment → Next State (s') + Reward (r)
```

There are no labeled examples. The agent must *discover* what actions are good through trial and error.

**Famous applications:**
- AlphaGo / AlphaZero: learned to play Go at superhuman level.
- OpenAI Five: learned to play Dota 2 at professional level.
- RLHF (Reinforcement Learning from Human Feedback): used to fine-tune ChatGPT and Claude to be helpful and harmless.
- Robotic control, chip design, drug discovery.

RL is the hardest to apply in practice due to: sparse rewards, exploration/exploitation tradeoffs, and sample inefficiency. We cover it in Chapter 27.

---

### Summary: Learning Paradigms at a Glance

| Paradigm | Labels | Goal | Example |
|---|---|---|---|
| Supervised | All data labeled | Predict y from x | Spam detection |
| Unsupervised | No labels | Find structure | Customer segments |
| Semi-supervised | Few labels | Use unlabeled data | Medical imaging |
| Self-supervised | Auto-generated | Pretrain representations | BERT, GPT |
| Reinforcement | Reward signal | Maximize cumulative reward | Game playing, robotics |

---

## 4. The Machine Learning Workflow

Every ML project, regardless of domain, follows the same fundamental workflow. Skipping steps or doing them in the wrong order is the #1 source of failed projects.

```
┌─────────────────────────────────────────────────────────────────┐
│                    The ML Workflow                              │
│                                                                 │
│  1. Define Problem  →  2. Collect Data  →  3. EDA              │
│          ↑                                      ↓               │
│  8. Monitor         ←  7. Deploy        ←  4. Preprocess        │
│                                                 ↓               │
│                          6. Evaluate   ←  5. Model & Iterate   │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Define the Problem

This is the most underrated step. Before touching data or code:

- **What exactly are you predicting?** Be precise. "Predict customer churn" is vague. "Predict whether a customer will cancel their subscription within the next 30 days" is specific.
- **What type of ML problem is this?** Regression, classification, clustering?
- **What metric defines success?** Accuracy? Revenue uplift? Recall (for medical diagnosis)? Precision (for spam filters)?
- **What is the cost of each type of error?** A false negative for cancer detection is far worse than a false positive.
- **What data is available, and is it sufficient?**
- **What is the baseline?** A "dumb" model (e.g., always predict the majority class) is your starting point.

### Step 2: Collect and Understand Data

- Identify data sources (databases, APIs, user logs, sensors).
- Understand data provenance: where did it come from? How was it collected?
- Watch for **selection bias** (the data doesn't represent the real population), **label noise** (incorrect labels), and **temporal leakage** (future data leaking into training).

### Step 3: Exploratory Data Analysis (EDA)

Before building any model:
- Check shapes, data types, missing values.
- Visualize distributions of each feature.
- Look for correlations between features and the target.
- Find outliers.
- **EDA prevents you from spending days modeling garbage data.** We devote all of Chapter 3 to this.

### Step 4: Preprocess and Feature Engineer

Raw data is almost never model-ready:
- Handle missing values (imputation or removal).
- Encode categorical variables (one-hot encoding, ordinal encoding, target encoding).
- Scale numerical features (standardization, min-max normalization).
- Create new features (interaction terms, polynomial features, domain-specific transforms).

Feature engineering is an art. The best features often come from domain knowledge. Chapter 13 covers this in full.

### Step 5: Model Selection and Training

- Start with a simple baseline (logistic regression, linear regression).
- Try progressively more complex models.
- Use cross-validation — never evaluate on training data.
- Tune hyperparameters systematically.

### Step 6: Evaluation

- Evaluate on a **held-out test set** that the model has never seen.
- Use the right metrics for your problem.
- Perform **error analysis**: look at the specific examples your model gets wrong. Why?

### Step 7: Deployment

- Wrap your model in an API.
- Handle edge cases in production (missing features, unexpected input formats).
- Version your model.

### Step 8: Monitor

Models **degrade in production** because the real world changes:
- **Data drift**: the input distribution shifts (e.g., user demographics change).
- **Concept drift**: the relationship between inputs and outputs changes (e.g., fraudsters adapt their tactics).

Set up monitoring and retrain regularly.

---

## 5. Generalization: The Central Problem

The fundamental goal of supervised learning is not to memorize training data — it is to **generalize** to new, unseen data.

Consider a model that simply memorizes every training example. It achieves 100% accuracy on the training set. But on new data, it fails completely — it has learned nothing useful.

This is called **overfitting**.

The opposite problem — **underfitting** — occurs when the model is too simple to capture the true patterns in the data.

Let's make this concrete with a regression example.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# --- Generate toy data ---
np.random.seed(42)
n = 30
X = np.sort(np.random.uniform(0, 1, n))
y_true = np.sin(2 * np.pi * X)        # True underlying function
y = y_true + np.random.normal(0, 0.3, n)  # Add noise

X_plot = np.linspace(0, 1, 300)
y_plot_true = np.sin(2 * np.pi * X_plot)

# --- Fit three models of increasing complexity ---
degrees = [1, 4, 15]  # Underfitting, good fit, overfitting
colors   = ['steelblue', 'green', 'firebrick']
labels   = ['Degree 1 (Underfitting)', 'Degree 4 (Good fit)', 'Degree 15 (Overfitting)']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, deg, color, label in zip(axes, degrees, colors, labels):
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X.reshape(-1, 1), y)

    y_pred_train = model.predict(X.reshape(-1, 1))
    y_pred_plot  = model.predict(X_plot.reshape(-1, 1))

    train_mse = mean_squared_error(y, y_pred_train)

    ax.scatter(X, y, color='gray', alpha=0.7, label='Training data')
    ax.plot(X_plot, y_plot_true, 'k--', linewidth=1.5, label='True function')
    ax.plot(X_plot, y_pred_plot, color=color, linewidth=2, label=f'Model (deg={deg})')
    ax.set_title(f'{label}\nTrain MSE: {train_mse:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=8)
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('figures/01_overfitting_underfitting.png', dpi=150)
plt.show()
```

**What you'll observe:**
- Degree 1 (linear): Too rigid, can't capture the sinusoidal pattern → **underfitting**
- Degree 4: Captures the pattern well without memorizing noise → **good generalization**
- Degree 15: Wiggles through almost every training point → **overfitting** (low train error, but will fail on new data)

**The key insight:** Low training error does not mean good model. Always evaluate on held-out data.

---

## 6. The Bias-Variance Tradeoff

This is perhaps the single most important concept in classical machine learning. The expected prediction error of any model can be decomposed as:

```
Expected Error = Bias² + Variance + Irreducible Noise
```

Let's define each term:

### Bias
**Bias** is the error due to wrong assumptions in the learning algorithm.

A high-bias model (e.g., fitting a straight line to sinusoidal data) **systematically misses** the true relationship. It doesn't matter how much data you collect — it will always be wrong in the same way.

> High bias = underfitting = model too simple for the problem.

### Variance
**Variance** is the error due to the model's sensitivity to small fluctuations in the training data.

A high-variance model (e.g., a degree-15 polynomial) **memorizes noise** in the training data. If you trained it on a different random sample of data, you'd get a wildly different model.

> High variance = overfitting = model too complex for the available data.

### Irreducible Noise
This is the noise inherent in the problem itself. Even a perfect model can't predict it. You can't eliminate it — you can only build models that learn to ignore it.

### The Tradeoff

As you increase model complexity:
- Bias **decreases** (the model can fit more complex patterns)
- Variance **increases** (the model becomes sensitive to training data fluctuations)

```
Error
  │
  │   Total Error
  │    \         /
  │     \       /
  │      \     /  ← Optimal complexity
  │       \   /
  │  Bias² \_/ Variance
  │
  └──────────────────── Model Complexity
```

The sweet spot is where total error is minimized. Achieving it requires:
- **Cross-validation** to estimate generalization error.
- **Regularization** to reduce variance without increasing bias too much.
- **More data** — the single most reliable way to reduce variance.
- **Better features** — the most reliable way to reduce bias.

---

## 7. No Free Lunch Theorem

In 1996, David Wolpert proved the **No Free Lunch (NFL) theorem**: *averaged over all possible problems, every learning algorithm performs equally well.*

In other words: **there is no universally best ML algorithm.**

An algorithm that works brilliantly for image classification (CNN) performs no better than random guessing on problems it wasn't designed for. An algorithm that excels on tabular data (gradient boosting) might be outperformed by a simple linear model on a different tabular dataset.

**What this means in practice:**
1. Never commit to one algorithm before trying others.
2. The "default" algorithm for structured/tabular data in 2024 is gradient boosting (XGBoost, LightGBM). But always sanity-check with a simple linear baseline.
3. For images: CNNs or Vision Transformers. For text: Transformers. But verify with your data.
4. Problem-specific inductive biases matter. A CNN's assumption (local spatial structure matters) is right for images. It's wrong for time-series.

---

## 8. The Modern ML Stack

Before we write any code, let's understand the tools we'll use throughout this book.

### Core Libraries

| Library | Purpose | When you use it |
|---|---|---|
| **NumPy** | N-dimensional arrays, vectorized math | The foundation of everything |
| **Pandas** | Tabular data (DataFrames) | Data loading, cleaning, EDA |
| **Matplotlib / Seaborn** | Visualization | EDA, evaluation, debugging |
| **Scikit-learn** | Classical ML | Linear models, trees, SVMs, pipelines, evaluation |
| **XGBoost / LightGBM** | Gradient boosting | Best-in-class for tabular data |
| **PyTorch** | Deep learning | Neural networks, CNNs, RNNs, Transformers |
| **HuggingFace Transformers** | Pretrained models | NLP, vision, multimodal tasks |
| **Optuna** | Hyperparameter optimization | Systematic tuning |

### How They Fit Together

```
Raw Data (CSV, images, text)
       ↓
   Pandas / NumPy
   (load, clean, explore)
       ↓
  Scikit-learn Pipelines
  (preprocess, transform)
       ↓
  ┌──────────────────┐
  │  Classical ML     │ ← Scikit-learn, XGBoost, LightGBM
  │  (tabular data)   │
  └──────────────────┘
         OR
  ┌──────────────────┐
  │   Deep Learning   │ ← PyTorch + HuggingFace
  │ (images, text...) │
  └──────────────────┘
       ↓
  Evaluation (Scikit-learn metrics)
       ↓
  Deployment (FastAPI + Docker)
```

### Installing the Stack

```bash
# Create a virtual environment (always do this — never pollute your system Python)
python -m venv ml-env
source ml-env/bin/activate       # Linux/macOS
# ml-env\Scripts\activate        # Windows

# Core scientific stack
pip install numpy pandas matplotlib seaborn scikit-learn

# Gradient boosting
pip install xgboost lightgbm

# Deep learning (CPU version — replace 'cpu' with your CUDA version for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# HuggingFace
pip install transformers datasets

# Hyperparameter optimization
pip install optuna

# Jupyter for interactive exploration
pip install jupyterlab
```

Verify your installation:

```python
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers

print(f"NumPy:          {np.__version__}")
print(f"Pandas:         {pd.__version__}")
print(f"Scikit-learn:   {sklearn.__version__}")
print(f"PyTorch:        {torch.__version__}")
print(f"Transformers:   {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Expected output (versions may differ):**
```
NumPy:          1.26.4
Pandas:         2.2.1
Scikit-learn:   1.4.2
PyTorch:        2.2.2
Transformers:   4.40.0
CUDA available: False
```

---

## 9. Your First End-to-End ML Program

Let's put everything together with a complete, working ML pipeline on a real dataset. We'll use the **Iris dataset** — 150 flower samples, 4 features (sepal length/width, petal length/width), 3 species classes.

This is intentionally simple. The goal is to see the full workflow in one clean script before we dive into each piece in detail.

```python
# =============================================================================
# Chapter 1 — Your First End-to-End ML Pipeline
# Dataset: Iris (classic multiclass classification)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names   = iris.target_names

print("Dataset shape:", X.shape)          # (150, 4)
print("Classes:", class_names)            # ['setosa' 'versicolor' 'virginica']
print("Feature names:", feature_names)

# Convert to DataFrame for easier inspection
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [class_names[i] for i in y]
print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df['species'].value_counts())       # 50 samples per class — perfectly balanced

print("\nBasic statistics:")
print(df.describe().round(2))

# ── 2. Exploratory Data Analysis ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
colors = ['#2196F3', '#FF5722', '#4CAF50']

for ax, feature in zip(axes.flatten(), feature_names):
    for cls_idx, cls_name in enumerate(class_names):
        mask = df['species'] == cls_name
        ax.hist(df.loc[mask, feature], bins=20, alpha=0.6,
                color=colors[cls_idx], label=cls_name)
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {feature}')
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions by Species', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/01_iris_eda.png', dpi=150)
plt.show()

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
# stratify=y ensures each class is proportionally represented in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")  # 120
print(f"Test set size:     {X_test.shape[0]}")     # 30

# ── 4. Build Pipelines for Three Models ──────────────────────────────────────
# A Pipeline chains preprocessing + model into a single object.
# CRITICAL: The scaler is fit ONLY on training data, then applied to test data.
# Never fit the scaler on the full dataset before splitting — that's data leakage.

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=200, random_state=42))
    ]),
    'Decision Tree': Pipeline([
        ('clf', DecisionTreeClassifier(max_depth=4, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
}

# ── 5. Cross-Validation ───────────────────────────────────────────────────────
# We use 5-fold CV on the training set to estimate generalization performance.
# The test set is UNTOUCHED until final evaluation.
print("\n── Cross-Validation Results (5-fold, Training Set Only) ──")
cv_results = {}
for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ── 6. Train on Full Training Set and Evaluate on Test Set ───────────────────
print("\n── Test Set Evaluation ──")
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

# ── 7. Confusion Matrix for Best Model ───────────────────────────────────────
# Random Forest typically performs best here
best_pipeline = models['Random Forest']
y_pred_best = best_pipeline.predict(X_test)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred_best),
    display_labels=class_names
)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Random Forest (Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/01_confusion_matrix.png', dpi=150)
plt.show()

# ── 8. Feature Importance (Random Forest) ────────────────────────────────────
importances = best_pipeline.named_steps['clf'].feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color=['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
)
ax.set_ylabel('Feature Importance (Mean Decrease in Impurity)')
ax.set_title('Random Forest Feature Importances', fontweight='bold')
ax.bar_label(bars, fmt='%.3f', padding=3)
plt.tight_layout()
plt.savefig('figures/01_feature_importance.png', dpi=150)
plt.show()

print("\nFeature Importances:")
for idx in sorted_idx:
    print(f"  {feature_names[idx]:30s}: {importances[idx]:.4f}")

# ── 9. Making a Prediction on a New Sample ───────────────────────────────────
# This is what deployment looks like: a model receives new data and outputs a prediction.
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Likely setosa
predicted_class = best_pipeline.predict(new_flower)[0]
predicted_proba = best_pipeline.predict_proba(new_flower)[0]

print(f"\n── Predicting a New Flower ──")
print(f"Input features (sepal_len, sepal_wid, petal_len, petal_wid): {new_flower[0]}")
print(f"Predicted class: {class_names[predicted_class]}")
print(f"Class probabilities:")
for cls, prob in zip(class_names, predicted_proba):
    print(f"  {cls:15s}: {prob:.4f}")
```

**Sample output:**
```
Dataset shape: (150, 4)
Classes: ['setosa' 'versicolor' 'virginica']

── Cross-Validation Results (5-fold, Training Set Only) ──
Logistic Regression      : 0.9667 ± 0.0298
Decision Tree            : 0.9417 ± 0.0340
Random Forest            : 0.9583 ± 0.0280

── Test Set Evaluation ──
Random Forest
Test Accuracy: 1.0000

── Predicting a New Flower ──
Predicted class: setosa
Class probabilities:
  setosa         : 1.0000
  versicolor     : 0.0000
  virginica      : 0.0000
```

### What to notice in this code

1. **The Pipeline object** prevents data leakage. The `StandardScaler` is fit inside cross-validation folds, not on the whole dataset.
2. **Cross-validation** gives us an unbiased estimate of generalization performance before we ever touch the test set.
3. **The test set is used exactly once.** If we kept tweaking and re-evaluating on the test set, its results would become optimistic.
4. **`predict_proba`** gives probability estimates, not just a class label. You often care about confidence, not just the prediction.
5. **Feature importance** tells us which inputs drive predictions. Petal length and width dominate for Iris — this matches botanical knowledge.

---

## 10. Summary

Here are the core concepts from Chapter 1 that you must internalize:

### ✅ Must-Remember Mental Models

**1. What ML is, at its core:**
A model is a mathematical function `ŷ = f(x; θ)`. Learning = finding `θ` that maps inputs to correct outputs on new, unseen data.

**2. The 5 learning paradigms:**
- Supervised: labeled input-output pairs → predict y from x
- Unsupervised: no labels → find structure
- Semi-supervised: few labels + lots of unlabeled data
- Self-supervised: labels generated from data itself (BERT, GPT)
- Reinforcement: learn via rewards from environment interactions

**3. The ML workflow (never skip steps):**
```
Define Problem → Collect Data → EDA → Preprocess →
Baseline → Iterate → Evaluate (on test set, once) → Deploy → Monitor
```

**4. Generalization is the goal. Not training accuracy.**
A model that memorizes training data is useless. Always evaluate on held-out data.

**5. Bias-Variance Tradeoff:**
- High bias (underfitting) = model too simple
- High variance (overfitting) = model too complex for available data
- Fix underfitting: more complex model, better features
- Fix overfitting: regularization, more data, simpler model, dropout

**6. No Free Lunch:**
No single algorithm wins on all problems. Always try a simple baseline before anything complex. Compare multiple models with cross-validation.

**7. Data leakage is silent and deadly:**
Never fit a preprocessor (scaler, imputer) on data that includes the test set. Always use Pipelines. The test set is used exactly once.

**8. The modern ML stack:**
NumPy (arrays) → Pandas (tables) → Scikit-learn (classical ML + pipelines) → PyTorch (deep learning) → HuggingFace (pretrained models)

---

## 11. Exercises

Work through these before moving on. Understanding is confirmed by doing.

**Conceptual:**

1. Explain the difference between supervised and self-supervised learning. Give two examples of each.

2. A fraud detection model achieves 99.9% accuracy on a dataset where only 0.1% of transactions are fraudulent. Why is this number misleading? What metric would you use instead?

3. You train a linear regression model on 100 samples and get a training RMSE of 0.15 and a test RMSE of 0.18. Then you train a 200-layer neural network on the same data: training RMSE = 0.01, test RMSE = 0.45. Explain what happened in terms of bias and variance.

4. Your model performs perfectly on the test set but fails in production. Name three possible causes.

**Coding:**

5. Load the `wine` dataset from `sklearn.datasets.load_wine()`. Repeat the end-to-end pipeline from Section 9. Report cross-validation scores for all three models.

6. Modify the polynomial regression code from Section 5. Plot training MSE and test MSE as a function of polynomial degree (1 through 15). Where is the optimal degree? What does this curve tell you about the bias-variance tradeoff?

7. Add a fourth model to the pipeline in Section 9: `KNeighborsClassifier(n_neighbors=5)` from `sklearn.neighbors`. Does it need a scaler? Why or why not? (Hint: think about how KNN makes predictions.)

8. In the end-to-end pipeline, intentionally introduce data leakage by fitting the `StandardScaler` on the full dataset (before the train-test split) and re-run the pipeline. Does the test accuracy change? Why or why not? (On Iris it might not — think about *when* it would matter.)

**Challenge:**

9. Implement the **bias-variance decomposition** empirically:
   - Generate 50 different training datasets of 30 samples each from `y = sin(2πx) + noise`.
   - For polynomial degrees 1, 4, and 15, train a model on each of the 50 datasets.
   - For each degree, compute the variance of predictions (how much do predictions differ across the 50 models?) and the squared bias (how far is the average prediction from the true function?).
   - Plot bias², variance, and total error vs. degree.

---

## 12. References

- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill. — Source of the classic definition.
- Wolpert, D. H. (1996). The Lack of A Priori Distinctions Between Learning Algorithms. *Neural Computation*, 8(7), 1341–1390. — The No Free Lunch theorem.
- Geman, S., Bienenstock, E., & Doursat, R. (1992). Neural Networks and the Bias/Variance Dilemma. *Neural Computation*, 4(1), 1–58. — Foundational bias-variance paper.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04805*. — Self-supervised learning at scale.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*. — The gold standard for tabular ML.
- Scikit-learn documentation: https://scikit-learn.org/stable/
- PyTorch documentation: https://pytorch.org/docs/stable/

---

*Next: [Chapter 2 — Python & the Scientific Stack](02_python_stack.md)*

*In Chapter 2, we go deep on NumPy — the engine behind all ML computations — and build intuition for vectorized operations, broadcasting, and memory-efficient array manipulations. We also cover Pandas for data wrangling and Matplotlib/Seaborn for visualization.*

---

> **Chapter 1 complete.** All code is in `notebooks/01_ml_landscape.ipynb`. Run it, break it, modify it.
