# Machine Learning with Python — From Foundations to Mastery

> *A ground-up, code-first guide in the style of Sebastian Raschka — rigorous, practical, and always grounded in the math that matters.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

---

## 📖 About This Book

This is an **open-source, beginner-to-advanced machine learning book** written in Markdown, designed to be read on GitHub or exported to any format. Every concept is paired with clean, self-contained Python code. The philosophy:

- **No magic boxes.** We open every algorithm and look inside.
- **Math when it matters.** Not more, not less.
- **Code you can run today.** Every notebook is reproducible.
- **Modern stack.** Scikit-learn, PyTorch, HuggingFace, Optuna, and beyond.

Inspired by Sebastian Raschka's writing style — methodical, example-driven, building intuition before abstraction.

---

## 🗂️ Table of Contents

### Part I — Foundations
| # | Chapter | Key Topics |
|---|---------|-----------|
| 1 | [The Machine Learning Landscape](chapters/01_ml_landscape.md) | Supervised/unsupervised/RL, ML workflow, tools overview |
| 2 | [Python & the Scientific Stack](chapters/02_python_stack.md) | NumPy, Pandas, Matplotlib, Scikit-learn setup |
| 3 | [Data Wrangling & Exploratory Data Analysis](chapters/03_eda.md) | Cleaning, profiling, visualization patterns |
| 4 | [Mathematics for Machine Learning](chapters/04_math.md) | Linear algebra, calculus, probability, statistics |

### Part II — Classical Machine Learning
| # | Chapter | Key Topics |
|---|---------|-----------|
| 5 | [Linear Regression](chapters/05_linear_regression.md) | OLS, gradient descent, regularization (L1/L2) |
| 6 | [Logistic Regression & Classification](chapters/06_logistic_regression.md) | Sigmoid, cross-entropy, decision boundary |
| 7 | [Decision Trees](chapters/07_decision_trees.md) | Gini, entropy, pruning, ID3/CART |
| 8 | [Ensemble Methods](chapters/08_ensembles.md) | Random Forests, Bagging, AdaBoost, XGBoost, LightGBM |
| 9 | [Support Vector Machines](chapters/09_svm.md) | Margin, kernel trick, soft-margin SVM |
| 10 | [KNN & Naive Bayes](chapters/10_knn_nb.md) | Distance metrics, Bayes theorem, Laplace smoothing |
| 11 | [Unsupervised Learning](chapters/11_unsupervised.md) | K-Means, DBSCAN, hierarchical clustering |
| 12 | [Dimensionality Reduction](chapters/12_dim_reduction.md) | PCA, t-SNE, UMAP, when to use each |

### Part III — Model Engineering
| # | Chapter | Key Topics |
|---|---------|-----------|
| 13 | [Feature Engineering & Selection](chapters/13_feature_engineering.md) | Encoding, scaling, interaction terms, mutual info |
| 14 | [Model Evaluation & Validation](chapters/14_evaluation.md) | Bias-variance, CV, ROC-AUC, precision-recall |
| 15 | [Hyperparameter Tuning](chapters/15_hyperparameter_tuning.md) | Grid search, random search, Optuna (Bayesian) |
| 16 | [Pipelines & Scikit-learn Patterns](chapters/16_pipelines.md) | Pipeline, ColumnTransformer, custom transformers |
| 17 | [Handling Imbalanced Data](chapters/17_imbalanced.md) | SMOTE, class weights, threshold moving, evaluation |

### Part IV — Deep Learning
| # | Chapter | Key Topics |
|---|---------|-----------|
| 18 | [Neural Networks from Scratch](chapters/18_nn_from_scratch.md) | Forward pass, backprop, numpy only |
| 19 | [Deep Learning with PyTorch](chapters/19_pytorch_foundations.md) | Tensors, autograd, nn.Module, training loop |
| 20 | [Convolutional Neural Networks](chapters/20_cnns.md) | Conv layers, pooling, ResNet, image classification |
| 21 | [Recurrent Networks & LSTMs](chapters/21_rnns_lstms.md) | Vanishing gradient, LSTM gates, sequence modeling |
| 22 | [Transformers & Attention](chapters/22_transformers.md) | Self-attention, multi-head attention, positional encoding |
| 23 | [Transfer Learning & Fine-tuning](chapters/23_transfer_learning.md) | Pretrained models, feature extraction vs fine-tuning |

### Part V — Advanced & Modern Topics
| # | Chapter | Key Topics |
|---|---------|-----------|
| 24 | [NLP with Transformers](chapters/24_nlp_transformers.md) | Tokenization, BERT, GPT, HuggingFace pipelines |
| 25 | [Large Language Models](chapters/25_llms.md) | Prompting, RAG, LoRA fine-tuning |
| 26 | [Generative Models](chapters/26_generative.md) | VAEs, diffusion models, latent spaces |
| 27 | [Reinforcement Learning](chapters/27_rl.md) | MDPs, Q-learning, policy gradient, PPO |
| 28 | [Graph Neural Networks](chapters/28_gnns.md) | Message passing, GCN, GraphSAGE, PyG intro |

### Part VI — Production & MLOps
| # | Chapter | Key Topics |
|---|---------|-----------|
| 29 | [ML Project Structure & Reproducibility](chapters/29_mlops_structure.md) | DVC, Hydra, Git, experiment tracking (MLflow) |
| 30 | [Model Deployment](chapters/30_deployment.md) | FastAPI, Docker, cloud (AWS/GCP), model serving |
| 31 | [Monitoring & Drift Detection](chapters/31_monitoring.md) | Data drift, concept drift, Evidently, retraining triggers |
| 32 | [Responsible AI](chapters/32_responsible_ai.md) | Fairness, SHAP, LIME, differential privacy |

### Appendices
| | Appendix |
|-|----------|
| A | [Python Refresher](appendices/A_python_refresher.md) |
| B | [Math Reference Sheet](appendices/B_math_reference.md) |
| C | [Cheat Sheets](appendices/C_cheatsheets.md) |
| D | [Recommended Resources & Papers](appendices/D_resources.md) |

---

## 🧠 What You Must Always Remember — The Core Mental Models

> These are the fundamental truths of ML. Memorize them. Return to them whenever you are confused.

### 1. The Bias-Variance Tradeoff
- **High bias** = model too simple = underfitting = bad on train *and* test
- **High variance** = model too complex = overfitting = great on train, bad on test
- Your job is always to find the sweet spot. Regularization, more data, and cross-validation are your primary tools.

### 2. No Free Lunch Theorem
No single algorithm works best on all problems. Always experiment. Always validate.

### 3. Data > Algorithm
A simple model on clean, well-engineered data beats a complex model on raw data almost every time. Spend 80% of your time on data.

### 4. Train / Validation / Test Split — The Golden Rule
- **Train** on training data only.
- **Tune** hyperparameters on validation data.
- **Report** final performance on the test set — exactly once.
- Peeking at the test set during development = data leakage = invalid results.

### 5. The ML Workflow (Always Follow This)
```
Define Problem → Collect Data → EDA → Preprocess →
Baseline Model → Iterate (features, models, tuning) →
Evaluate → Deploy → Monitor
```

### 6. Gradient Descent in One Line
Move in the direction of the negative gradient of the loss, scaled by the learning rate. Everything in deep learning is a variation of this.

### 7. Cross-Entropy Is the Right Loss for Classification
Mean Squared Error for regression. Cross-entropy for classification. Using the wrong loss gives wrong gradients.

### 8. Scaling Matters
Always scale/normalize features before using distance-based models (KNN, SVM, PCA, neural networks). Tree-based models don't need it.

### 9. Correlation ≠ Causation
ML models find correlations. They do not discover causal relationships unless specifically designed to (e.g., causal inference methods).

### 10. The Confusion Matrix Is Your Best Friend
Accuracy is misleading on imbalanced datasets. Always look at precision, recall, F1, and the ROC-AUC curve.

### 11. Transformers Are Sequence-to-Sequence Attention Machines
The key innovation: every token attends to every other token (self-attention). This replaced RNNs for most sequence tasks.

### 12. Embeddings Are Compressed Representations
Whether word embeddings, image embeddings, or user embeddings — the goal is always to map high-dimensional data into a dense, meaningful lower-dimensional space.

---

## 🛠️ Environment Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/ml-python-book.git
cd ml-python-book

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies
```
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
torch>=2.2
torchvision>=0.17
transformers>=4.40
datasets>=2.18
optuna>=3.6
xgboost>=2.0
lightgbm>=4.3
umap-learn>=0.5
shap>=0.45
fastapi>=0.110
```

---

## 📁 Repository Structure

```
ml-python-book/
│
├── README.md                  ← You are here
├── requirements.txt
│
├── chapters/
│   ├── 01_ml_landscape.md
│   ├── 02_python_stack.md
│   └── ...                    ← One .md file per chapter
│
├── notebooks/
│   ├── 01_ml_landscape.ipynb
│   ├── 02_python_stack.ipynb
│   └── ...                    ← Executable Jupyter notebooks
│
├── code/
│   ├── ch05_linear_regression/
│   ├── ch08_ensembles/
│   └── ...                    ← Standalone Python scripts
│
├── datasets/
│   └── ...                    ← Small datasets used in examples
│
└── appendices/
    ├── A_python_refresher.md
    ├── B_math_reference.md
    ├── C_cheatsheets.md
    └── D_resources.md
```

---

## ✍️ Writing Style & Philosophy

This book follows Sebastian Raschka's principles:

1. **Implement first, library second.** We implement algorithms from scratch before reaching for Scikit-learn or PyTorch — because understanding the internals is non-negotiable.
2. **Annotated code.** Every non-obvious line of code is explained.
3. **Visual intuition.** Every major concept gets a figure or plot.
4. **Exercises at the end of every chapter.** Learning requires doing.
5. **References to original papers.** Science, not cargo-culting.

---

## 🤝 Contributing

Contributions are welcome! If you find a bug, a conceptual error, or want to add an exercise:

1. Fork the repo
2. Create your branch: `git checkout -b fix/chapter-05-typo`
3. Commit your changes: `git commit -m 'Fix gradient descent derivation in ch05'`
4. Push and open a Pull Request

Please follow the existing Markdown formatting style.

---

## 📄 License

This book is released under the [MIT License](LICENSE). You are free to use, share, and adapt it with attribution.

---

## 🙏 Acknowledgments

Inspired by:
- *Machine Learning with PyTorch and Scikit-Learn* — Sebastian Raschka
- *Hands-On Machine Learning* — Aurélien Géron
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman
- *Dive into Deep Learning* — d2l.ai

---

> *"The best way to learn machine learning is to implement it — not just to use it."*

**Let's start with Chapter 1 whenever you're ready. 🚀**
