# Chapter 4: Mathematics for Machine Learning

> *"The purpose of computing is insight, not numbers."*  
> — Richard Hamming

> *"You don't need to be a mathematician to do machine learning. But you do need enough mathematics to understand what your algorithm is actually doing — and why it sometimes fails."*

---

## Table of Contents

1. [Why Mathematics?](#1-why-mathematics)
2. [Linear Algebra](#2-linear-algebra)
   - 2.1 [Scalars, Vectors, Matrices, Tensors](#21-scalars-vectors-matrices-tensors)
   - 2.2 [Matrix Operations](#22-matrix-operations)
   - 2.3 [Norms — Measuring Size](#23-norms--measuring-size)
   - 2.4 [Special Matrices](#24-special-matrices)
   - 2.5 [Eigenvalues and Eigenvectors](#25-eigenvalues-and-eigenvectors)
   - 2.6 [Singular Value Decomposition](#26-singular-value-decomposition)
   - 2.7 [Linear Algebra in ML — Concrete Applications](#27-linear-algebra-in-ml--concrete-applications)
3. [Calculus and Optimization](#3-calculus-and-optimization)
   - 3.1 [Derivatives — The Rate of Change](#31-derivatives--the-rate-of-change)
   - 3.2 [Partial Derivatives and Gradients](#32-partial-derivatives-and-gradients)
   - 3.3 [The Chain Rule — Backbone of Backpropagation](#33-the-chain-rule--backbone-of-backpropagation)
   - 3.4 [Gradient Descent](#34-gradient-descent)
   - 3.5 [The Jacobian and Hessian](#35-the-jacobian-and-hessian)
   - 3.6 [Convexity — Why It Matters](#36-convexity--why-it-matters)
4. [Probability and Statistics](#4-probability-and-statistics)
   - 4.1 [Probability Fundamentals](#41-probability-fundamentals)
   - 4.2 [Random Variables and Distributions](#42-random-variables-and-distributions)
   - 4.3 [Key Distributions in ML](#43-key-distributions-in-ml)
   - 4.4 [Bayes' Theorem](#44-bayes-theorem)
   - 4.5 [Maximum Likelihood Estimation](#45-maximum-likelihood-estimation)
   - 4.6 [Information Theory — Entropy and KL Divergence](#46-information-theory--entropy-and-kl-divergence)
5. [Putting It All Together: Deriving Linear Regression](#5-putting-it-all-together-deriving-linear-regression)
6. [Numerical Stability — The Practitioner's Concern](#6-numerical-stability--the-practitioners-concern)
7. [Summary](#7-summary)
8. [Exercises](#8-exercises)
9. [References](#9-references)

---

## 1. Why Mathematics?

You can train a neural network without understanding the math. You cannot *debug* one without it.

Consider these real situations:
- Your loss explodes to `nan` after a few training steps. Without understanding gradients and numerical stability, you cannot fix it.
- Your logistic regression converges to terrible results. Without understanding convexity and the likelihood function, you don't know where to look.
- You want to explain why PCA works. Without eigendecomposition, you can only wave your hands.
- You need to choose between L1 and L2 regularization. Without understanding the geometry of norms, the choice is arbitrary.

This chapter covers the mathematics you *need* — linear algebra, calculus (focused on optimization), and probability/statistics. We derive results rather than just state them, because derivations build intuition that survives beyond any specific formula.

**What this chapter is not:** A complete mathematics textbook. For deeper treatment, see the references at the end.

---

## 2. Linear Algebra

### 2.1 Scalars, Vectors, Matrices, Tensors

**Scalar**: A single real number. Temperature, loss value, learning rate.
$$a \in \mathbb{R}$$

**Vector**: An ordered list of numbers. A data point with $n$ features, a weight vector, a gradient.
$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n \quad \text{(column vector by convention)}$$

**Matrix**: A 2D grid of numbers. A dataset of $m$ samples and $n$ features, a weight matrix in a neural network layer.
$$\mathbf{A} \in \mathbb{R}^{m \times n}$$

**Tensor**: A generalization to $N$ dimensions. Images are rank-3 tensors (height × width × channels). Batches of images are rank-4 tensors (batch × height × width × channels).

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Concrete representation of each ───────────────────────────────────────────
scalar = 3.14                                        # shape: ()
vector = np.array([1.0, 2.0, 3.0])                  # shape: (3,)
matrix = np.array([[1., 2., 3.],
                   [4., 5., 6.]])                    # shape: (2, 3)
tensor = np.zeros((4, 28, 28, 3))                    # shape: (4, 28, 28, 3)
                                                     # 4 images, 28×28 pixels, RGB

print(f"Scalar:  {scalar}              shape: {np.asarray(scalar).shape}")
print(f"Vector:  {vector}  shape: {vector.shape}")
print(f"Matrix:  shape = {matrix.shape}")
print(f"Tensor:  shape = {tensor.shape}")

# ── Geometric intuition ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Vector as arrow in 2D
ax = axes[0]
v1 = np.array([3.0, 2.0])
v2 = np.array([1.0, 3.5])
ax.annotate('', xy=v1, xytext=[0,0],
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.5))
ax.annotate('', xy=v2, xytext=[0,0],
            arrowprops=dict(arrowstyle='->', color='tomato', lw=2.5))
ax.annotate('', xy=v1+v2, xytext=[0,0],
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5, linestyle='dashed'))
# Parallelogram
ax.plot([v1[0], v1[0]+v2[0]], [v1[1], v1[1]+v2[1]], 'k--', lw=1, alpha=0.4)
ax.plot([v2[0], v1[0]+v2[0]], [v2[1], v1[1]+v2[1]], 'k--', lw=1, alpha=0.4)
ax.text(*v1*0.5 + [0.1, 0.1], 'v₁', fontsize=13, color='steelblue')
ax.text(*v2*0.5 + [-0.3, 0.1], 'v₂', fontsize=13, color='tomato')
ax.text(*(v1+v2)*0.5 + [0.1, 0.1], 'v₁+v₂', fontsize=12, color='green')
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 6.5)
ax.set_aspect('equal')
ax.set_title('Vector Addition (Parallelogram Law)')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.grid(True, alpha=0.3)

# Matrix as transformation
ax = axes[1]
# Unit square corners
square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]).T   # (2, 5)
A = np.array([[2., 1.], [0.5, 1.5]])                    # transformation matrix
transformed = A @ square

ax.plot(square[0], square[1], 'steelblue', lw=2, label='Original')
ax.fill(square[0], square[1], alpha=0.15, color='steelblue')
ax.plot(transformed[0], transformed[1], 'tomato', lw=2, label='Transformed')
ax.fill(transformed[0], transformed[1], alpha=0.15, color='tomato')
ax.set_aspect('equal')
ax.set_title('Matrix as Linear Transformation')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)

# Data matrix visualization
ax = axes[2]
rng = np.random.default_rng(42)
X = rng.standard_normal((8, 4))
im = ax.imshow(X, cmap='RdBu_r', aspect='auto', vmin=-2.5, vmax=2.5)
ax.set_title('Data Matrix X  (8 samples × 4 features)')
ax.set_xlabel('Features')
ax.set_ylabel('Samples')
ax.set_xticks(range(4)); ax.set_xticklabels([f'x{i+1}' for i in range(4)])
ax.set_yticks(range(8)); ax.set_yticklabels([f's{i+1}' for i in range(8)])
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('figures/04_vectors_matrices.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 2.2 Matrix Operations

```python
import numpy as np

# ── Transpose ─────────────────────────────────────────────────────────────────
# (A^T)_{ij} = A_{ji}  — rows become columns, columns become rows
A = np.array([[1., 2., 3.],
              [4., 5., 6.]])     # shape (2, 3)
print(f"A shape: {A.shape}")
print(f"A.T shape: {A.T.shape}")
# Key identity: (AB)^T = B^T A^T

# ── Matrix-Vector Multiplication ──────────────────────────────────────────────
# y = Ax:  transforms vector x from R^n to R^m
# This is the core operation in: linear regression, neural network layers, PCA
A = np.array([[1., 2.],
              [3., 4.],
              [5., 6.]])         # shape (3, 2): maps R^2 → R^3
x = np.array([1., 2.])          # shape (2,)
y = A @ x                       # shape (3,)
print(f"\nA @ x = {y}")

# Interpretation: y_i = dot product of row i of A with x
# Each output dimension is a weighted combination of input dimensions
print("Row 0 dot x:", np.dot(A[0], x))   # Same as y[0]

# ── Matrix-Matrix Multiplication ──────────────────────────────────────────────
# C = AB  where A ∈ R^{m×k} and B ∈ R^{k×n}  →  C ∈ R^{m×n}
# The k-dimension must match: "inner dimensions cancel"
A = np.random.default_rng(0).standard_normal((3, 4))   # (3, 4)
B = np.random.default_rng(1).standard_normal((4, 5))   # (4, 5)
C = A @ B                                               # (3, 5)
print(f"\nA:{A.shape} @ B:{B.shape} = C:{C.shape}")

# ── Properties of matrix multiplication ──────────────────────────────────────
# AB ≠ BA  (NOT commutative)
# (AB)C = A(BC)  (associative)
# A(B+C) = AB + AC  (distributive)

A = np.array([[1.,2.],[3.,4.]])
B = np.array([[5.,6.],[7.,8.]])
print(f"\nAB:\n{A @ B}")
print(f"BA:\n{B @ A}")
print(f"AB ≠ BA: {not np.allclose(A @ B, B @ A)}")

# ── Dot product ──────────────────────────────────────────────────────────────
# a · b = a^T b = Σ_i a_i b_i
# Geometric meaning: a · b = ||a|| ||b|| cos(θ)
# θ = angle between vectors
a = np.array([1., 0., 0.])   # Unit vector along x
b = np.array([0., 1., 0.])   # Unit vector along y
c = np.array([1., 1., 0.]) / np.sqrt(2)  # 45 degrees

print(f"\na · b (perpendicular): {np.dot(a, b):.4f}   (cos 90° = 0)")
print(f"a · c (45 degrees):    {np.dot(a, c):.4f}   (cos 45° ≈ 0.707)")
print(f"a · a (same vector):   {np.dot(a, a):.4f}   (cos 0° = 1)")

# ── Outer product ──────────────────────────────────────────────────────────────
# a ⊗ b = ab^T  — produces a matrix from two vectors
# Used in: attention mechanisms, low-rank approximations, covariance estimation
a = np.array([1., 2., 3.])    # (3,)
b = np.array([4., 5.])        # (2,)
outer = np.outer(a, b)        # (3, 2)
print(f"\nOuter product shape: {outer.shape}")
print(outer)

# ── Hadamard (element-wise) product ──────────────────────────────────────────
# A ⊙ B  — element-wise multiplication (same shape)
# Used in: LSTM gates, attention masking, element-wise activations
A = np.array([[1., 2.], [3., 4.]])
B = np.array([[5., 6.], [7., 8.]])
print(f"\nHadamard product (A ⊙ B):\n{A * B}")
```

---

### 2.3 Norms — Measuring Size

A **norm** is a function that assigns a non-negative "length" to a vector.

The **L_p norm** is defined as:
$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p \right)^{1/p}$$

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── The three norms you need to know ─────────────────────────────────────────
v = np.array([3., 4.])

L1 = np.linalg.norm(v, ord=1)        # |3| + |4| = 7     (Manhattan, "taxicab")
L2 = np.linalg.norm(v, ord=2)        # sqrt(9 + 16) = 5   (Euclidean)
Linf = np.linalg.norm(v, ord=np.inf) # max(|3|, |4|) = 4  (Chebyshev)

print(f"v = {v}")
print(f"L1 norm: {L1:.4f}  (sum of absolute values)")
print(f"L2 norm: {L2:.4f}  (Euclidean distance)")
print(f"L∞ norm: {Linf:.4f}  (max absolute value)")

# ── Unit balls — what does "length 1" look like for each norm? ────────────────
# The unit ball = {x : ||x|| ≤ 1}
# This geometry determines how regularization shapes the solution space.

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
theta = np.linspace(0, 2*np.pi, 500)

# L1 ball: diamond shape
# → induces SPARSITY: corners of the diamond are on axes → solutions tend to
#   have many zero components → this is why L1 regularization (Lasso) produces
#   sparse models
t = np.linspace(0, 2*np.pi, 1000)
# |x|+|y|=1 parameterization
r_L1 = 1 / (np.abs(np.cos(t)) + np.abs(np.sin(t)) + 1e-10)
axes[0].fill(r_L1*np.cos(t), r_L1*np.sin(t), alpha=0.3, color='steelblue')
axes[0].plot(r_L1*np.cos(t), r_L1*np.sin(t), color='steelblue', lw=2)
axes[0].set_title('L1 Unit Ball\n(Diamond → promotes sparsity)', fontsize=10)
axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='k', lw=0.5); axes[0].axvline(0, color='k', lw=0.5)
axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)

# L2 ball: circle
axes[1].fill(np.cos(theta), np.sin(theta), alpha=0.3, color='tomato')
axes[1].plot(np.cos(theta), np.sin(theta), color='tomato', lw=2)
axes[1].set_title('L2 Unit Ball\n(Circle → shrinks weights uniformly)', fontsize=10)
axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='k', lw=0.5); axes[1].axvline(0, color='k', lw=0.5)
axes[1].set_xlim(-1.5, 1.5); axes[1].set_ylim(-1.5, 1.5)

# L∞ ball: square
square_pts = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])
axes[2].fill(square_pts[:,0], square_pts[:,1], alpha=0.3, color='green')
axes[2].plot(square_pts[:,0], square_pts[:,1], color='green', lw=2)
axes[2].set_title('L∞ Unit Ball\n(Square → bounds max component)', fontsize=10)
axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)
axes[2].axhline(0, color='k', lw=0.5); axes[2].axvline(0, color='k', lw=0.5)
axes[2].set_xlim(-1.5, 1.5); axes[2].set_ylim(-1.5, 1.5)

plt.suptitle('Unit Balls for Different Norms\n'
             'The shape determines how regularization interacts with the loss surface',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_unit_balls.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Frobenius norm for matrices ───────────────────────────────────────────────
# ||A||_F = sqrt(Σ_ij A_ij²) = sqrt(trace(A^T A))
# = the L2 norm of the flattened matrix
A = np.array([[1., 2.], [3., 4.]])
frob = np.linalg.norm(A, 'fro')
print(f"\nFrobenius norm of A: {frob:.4f}")
print(f"Verify (manual):     {np.sqrt((A**2).sum()):.4f}")
```

**Key insight for ML:**
- **L2 regularization (Ridge)**: penalizes the L2 norm of weights → shrinks all weights toward zero proportionally → keeps all features, reduces their magnitude → circular constraint → smooth solution
- **L1 regularization (Lasso)**: penalizes the L1 norm → diamond-shaped constraint has corners on axes → optimizer tends to land on corners → sparse solutions (many weights = 0) → automatic feature selection

---

### 2.4 Special Matrices

```python
import numpy as np

# ── Identity matrix I ─────────────────────────────────────────────────────────
# AI = IA = A  (multiplicative identity)
I = np.eye(4)
A = np.random.default_rng(0).standard_normal((4, 4))
print("A @ I == A:", np.allclose(A @ I, A))
print("I @ A == A:", np.allclose(I @ A, A))

# ── Inverse matrix A⁻¹ ────────────────────────────────────────────────────────
# A A⁻¹ = A⁻¹ A = I
# Only exists for square, full-rank (invertible/non-singular) matrices
# In ML: we rarely compute inverses explicitly (numerically unstable + slow O(n³))
#         Instead we use: np.linalg.solve(A, b) to solve Ax=b
A_sq = np.array([[2., 1.], [1., 3.]])
A_inv = np.linalg.inv(A_sq)
print(f"\nA⁻¹:\n{A_inv.round(4)}")
print(f"A @ A⁻¹ ≈ I: {np.allclose(A_sq @ A_inv, np.eye(2))}")

# Solving Ax=b — the right way
b = np.array([4., 5.])
x_solve = np.linalg.solve(A_sq, b)     # Numerically stable
x_inv   = A_inv @ b                    # Works but less stable
print(f"\nSolving Ax=b:  x = {x_solve}")
print(f"Verify Ax=b:   {np.allclose(A_sq @ x_solve, b)}")

# ── Orthogonal matrices Q ──────────────────────────────────────────────────────
# Q^T Q = I  →  Q^T = Q⁻¹
# Columns are orthonormal (unit length, mutually perpendicular)
# Geometric meaning: rotations and reflections — preserve lengths and angles
# Key property: ||Qx|| = ||x||  →  orthogonal transforms preserve norms
from scipy.stats import ortho_group
Q = ortho_group.rvs(3, random_state=42)
print(f"\nQ^T Q ≈ I: {np.allclose(Q.T @ Q, np.eye(3))}")

v = np.array([1., 2., 3.])
Qv = Q @ v
print(f"||v||  = {np.linalg.norm(v):.4f}")
print(f"||Qv|| = {np.linalg.norm(Qv):.4f}  (preserved — orthogonal transforms are isometries)")

# ── Symmetric matrix A = A^T ──────────────────────────────────────────────────
# Covariance matrices are always symmetric positive semi-definite
# All eigenvalues of a real symmetric matrix are real
# Eigenvectors are orthogonal

def make_covariance(X):
    """Sample covariance matrix from data matrix X (n_samples × n_features)."""
    X_centered = X - X.mean(axis=0)
    return (X_centered.T @ X_centered) / (len(X) - 1)

rng = np.random.default_rng(42)
X = rng.multivariate_normal([0, 0, 0],
                             [[2., 0.8, 0.2],
                              [0.8, 1., 0.5],
                              [0.2, 0.5, 1.5]], size=500)
Sigma = make_covariance(X)
print(f"\nCovariance matrix:\n{Sigma.round(3)}")
print(f"Symmetric (Σ = Σᵀ): {np.allclose(Sigma, Sigma.T)}")
print(f"Eigenvalues (all real, all ≥ 0 for PSD): {np.linalg.eigvalsh(Sigma).round(4)}")

# ── Positive Definite (PD) and Positive Semi-Definite (PSD) ───────────────────
# PD:  x^T A x > 0  for all x ≠ 0  →  all eigenvalues > 0
# PSD: x^T A x ≥ 0  for all x     →  all eigenvalues ≥ 0
# Covariance matrices: PSD (possibly PD)
# Why it matters: a PSD matrix defines a valid inner product and a valid probability density

print(f"\nAll eigenvalues ≥ 0 (PSD): {np.all(np.linalg.eigvalsh(Sigma) >= -1e-10)}")
```

---

### 2.5 Eigenvalues and Eigenvectors

**Definition**: For a square matrix $\mathbf{A}$, a non-zero vector $\mathbf{v}$ is an **eigenvector** with **eigenvalue** $\lambda$ if:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

The matrix acts on $\mathbf{v}$ by only scaling it — it does not rotate it. $\lambda$ is the scaling factor.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── Computing eigendecomposition ─────────────────────────────────────────────
A = np.array([[3., 1.],
              [1., 3.]])

eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvectors[:, i] is the i-th eigenvector (COLUMN)

print("Matrix A:")
print(A)
print(f"\nEigenvalues:  {eigenvalues}")
print(f"Eigenvector 0: {eigenvectors[:, 0].round(4)}")
print(f"Eigenvector 1: {eigenvectors[:, 1].round(4)}")

# Verify: Av = λv
for i in range(2):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    lhs = A @ v
    rhs = lam * v
    print(f"\nAv_{i} = {lhs.round(6)}")
    print(f"λ_{i}v_{i} = {rhs.round(6)}")
    print(f"Equal: {np.allclose(lhs, rhs)}")

# ── Geometric interpretation ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Show how A transforms various vectors
theta_vals = np.linspace(0, 2*np.pi, 16, endpoint=False)
colors = plt.cm.tab20(np.linspace(0, 1, 16))

ax = axes[0]
for theta, color in zip(theta_vals, colors):
    v = np.array([np.cos(theta), np.sin(theta)])
    Av = A @ v
    ax.annotate('', xy=v, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.5))
    ax.annotate('', xy=Av, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=1.0,
                               linestyle='dashed'))

# Draw eigenvectors
for i, color in enumerate(['tomato', 'steelblue']):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    ax.annotate('', xy=v * 2.5, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color=color, lw=3))
    ax.text(*(v * 2.7), f'v{i+1} (λ={lam:.1f})', color=color, fontsize=11, fontweight='bold')

ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_title('Eigenvectors are not rotated by A\n'
             'Faded = original vectors, Solid = transformed')

# ── Eigendecomposition of covariance matrix ───────────────────────────────────
rng = np.random.default_rng(42)
X = rng.multivariate_normal([0, 0], [[3., 2.], [2., 2.]], size=300)
Sigma = np.cov(X.T)
eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(Sigma)  # eigh for symmetric

ax2 = axes[1]
ax2.scatter(X[:, 0], X[:, 1], alpha=0.3, s=15, color='steelblue')

# Draw eigenvectors scaled by sqrt(eigenvalue) = std along that direction
scale = 2.5
for i, (lam, color) in enumerate(zip(eigenvalues_cov[::-1], ['tomato', 'orange'])):
    v = eigenvectors_cov[:, -(i+1)]    # eigenvectors in ascending order from eigh
    ax2.annotate('', xy=scale * np.sqrt(lam) * v,
                 xytext=-scale * np.sqrt(lam) * v,
                 arrowprops=dict(arrowstyle='<->', color=color, lw=3))
    ax2.text(*(scale * np.sqrt(lam) * v + 0.2),
             f'PC{i+1}\nλ={lam:.2f}', color=color, fontsize=10, fontweight='bold')

ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
ax2.set_title('Eigenvectors of Covariance Matrix\n'
              '= Principal Components\n'
              'Length ∝ √eigenvalue = std in that direction')

plt.suptitle('Eigenvalues and Eigenvectors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_eigenvectors.png', dpi=140, bbox_inches='tight')
plt.show()

# ── The eigendecomposition: A = Q Λ Q^T ──────────────────────────────────────
# For symmetric matrices (like covariance matrices):
# A can be decomposed as A = Q Λ Q^T
# where Q = matrix of eigenvectors (orthogonal), Λ = diagonal eigenvalue matrix
Q   = eigenvectors_cov
Lam = np.diag(eigenvalues_cov)
A_reconstructed = Q @ Lam @ Q.T
print(f"\nEigendecomposition: A ≈ QΛQᵀ: {np.allclose(Sigma, A_reconstructed)}")
```

**Why eigenvalues matter in ML:**
- **PCA**: principal components are eigenvectors of the covariance matrix. Eigenvalues tell you how much variance is explained by each component.
- **Spectral methods**: graph clustering, manifold learning (Laplacian Eigenmaps).
- **Stability analysis**: eigenvalues of the Hessian determine if a critical point is a minimum, maximum, or saddle point.

---

### 2.6 Singular Value Decomposition

SVD is the most important matrix decomposition in ML. **Every matrix** (not just square ones) has an SVD.

$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

Where:
- $\mathbf{A} \in \mathbb{R}^{m \times n}$
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: left singular vectors (orthogonal) — "output space"
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: diagonal, non-negative singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $\mathbf{V}^T \in \mathbb{R}^{n \times n}$: right singular vectors (orthogonal) — "input space"

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Computing SVD ─────────────────────────────────────────────────────────────
A = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.],
              [10.,11.,12.]])   # (4, 3) — not square

U, s, Vt = np.linalg.svd(A, full_matrices=False)   # "economy" SVD

print(f"A shape: {A.shape}")
print(f"U shape: {U.shape}   (left singular vectors)")
print(f"s shape: {s.shape}   (singular values)")
print(f"Vt shape: {Vt.shape}  (right singular vectors, transposed)")
print(f"\nSingular values: {s.round(4)}")
# Note: s[2] ≈ 0 → A has rank 2 (rows are linearly dependent)

# Reconstruct: A = U @ diag(s) @ Vt
A_reconstructed = U @ np.diag(s) @ Vt
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# ── Low-rank approximation (the most important use of SVD) ────────────────────
# Keep only the top-k singular values → best rank-k approximation
# (Eckart–Young–Mirsky theorem: optimal in both Frobenius and spectral norm)

# Use an image to demonstrate compression
from sklearn.datasets import load_digits
digits = load_digits()
img = digits.images[0]   # 8×8 image

U_img, s_img, Vt_img = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
k_values = [1, 2, 3, 5, 8]

for ax, k in zip(axes, k_values):
    img_k = U_img[:, :k] @ np.diag(s_img[:k]) @ Vt_img[:k, :]
    error = np.linalg.norm(img - img_k, 'fro') / np.linalg.norm(img, 'fro')
    # Variance explained by top-k singular values
    var_explained = (s_img[:k]**2).sum() / (s_img**2).sum()
    ax.imshow(img_k, cmap='gray', interpolation='nearest')
    ax.set_title(f'k={k}\n{var_explained*100:.1f}% variance\nerr={error:.3f}', fontsize=9)
    ax.axis('off')

plt.suptitle('SVD Low-Rank Approximation — Digit Image',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_svd_compression.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Singular values and explained variance ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Singular values
axes[0].bar(range(1, len(s_img)+1), s_img, color='steelblue', alpha=0.8)
axes[0].set_xlabel('Component k'); axes[0].set_ylabel('Singular value σ_k')
axes[0].set_title('Singular Values (ordered)')

# Cumulative variance explained
cum_var = np.cumsum(s_img**2) / (s_img**2).sum()
axes[1].plot(range(1, len(s_img)+1), cum_var, 'o-', color='tomato', linewidth=2)
axes[1].axhline(0.95, color='gray', linestyle='--', label='95%')
axes[1].axhline(0.99, color='black', linestyle='--', label='99%')
axes[1].set_xlabel('Number of components k')
axes[1].set_ylabel('Cumulative variance explained')
axes[1].set_title('Scree Plot — Cumulative Variance')
axes[1].legend()
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('figures/04_svd_variance.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Connection to PCA ─────────────────────────────────────────────────────────
# PCA = SVD of the mean-centered data matrix
# If X_c = X - mean(X), then:
#   X_c = U Σ V^T
# Principal components = right singular vectors V
# Explained variance = σ_i² / (n-1)
# PCA scores = U Σ  (projection of data onto principal components)

print("\nSVD ↔ PCA connection:")
print("Right singular vectors V = principal component directions")
print("σ_i² / (n-1) = variance explained by PC_i")
print("U Σ = PCA scores (data projected onto PCs)")
```

---

### 2.7 Linear Algebra in ML — Concrete Applications

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ── Application 1: Linear Regression prediction ──────────────────────────────
# y_hat = X @ w  (matrix-vector multiply)
# n=100 samples, d=3 features
n, d = 100, 3
rng = np.random.default_rng(42)
X = rng.standard_normal((n, d))
w = np.array([2., -1., 0.5])     # True weights
y_hat = X @ w                    # Shape (100,)  — one prediction per sample
print(f"Prediction shape: {y_hat.shape}")

# ── Application 2: Neural network layer ──────────────────────────────────────
# h = activation(W x + b)
# W ∈ R^{d_out × d_in},  x ∈ R^{d_in}
batch_size = 32
d_in, d_out = 128, 64
X_batch = rng.standard_normal((batch_size, d_in))  # (32, 128)
W = rng.standard_normal((d_out, d_in)) * 0.01      # (64, 128)
b = np.zeros(d_out)                                # (64,)
h = np.maximum(0, X_batch @ W.T + b)              # ReLU: (32, 64)
print(f"\nNeural layer output shape: {h.shape}")

# ── Application 3: Covariance and PCA ────────────────────────────────────────
iris = load_iris()
X_iris = StandardScaler().fit_transform(iris.data)  # Standardize: mean=0, std=1

# Covariance matrix
Sigma = X_iris.T @ X_iris / (len(X_iris) - 1)  # (4, 4)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues   = eigenvalues[idx]
eigenvectors  = eigenvectors[:, idx]

# Project to 2D
X_pca = X_iris @ eigenvectors[:, :2]  # (150, 2)

print(f"\nPCA explained variance: {eigenvalues[:2] / eigenvalues.sum() * 100}")
print(f"Top-2 PCs explain: {eigenvalues[:2].sum() / eigenvalues.sum() * 100:.1f}% of variance")

# ── Application 4: Distance matrix (used in KNN, K-Means) ────────────────────
# ||x_i - x_j||² = x_i·x_i + x_j·x_j - 2 x_i·x_j
# Vectorized over all pairs using broadcasting + matrix multiply
def pairwise_distances(X):
    """Compute n×n pairwise squared Euclidean distance matrix."""
    sq_norms = (X**2).sum(axis=1, keepdims=True)   # (n, 1)
    D_sq = sq_norms + sq_norms.T - 2 * X @ X.T     # (n, n)
    return np.sqrt(np.maximum(D_sq, 0))              # Clip for numerical safety

D = pairwise_distances(X_pca[:10])
print(f"\nDistance matrix (first 10 Iris samples) shape: {D.shape}")
print(D.round(2))
```

---

## 3. Calculus and Optimization

### 3.1 Derivatives — The Rate of Change

The derivative $f'(x)$ tells us how $f$ changes for an infinitesimally small change in $x$:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

In ML, we almost never compute derivatives symbolically by hand. We need to:
1. Know the derivatives of common functions (see table below)
2. Apply the chain rule to compose them
3. Trust that autograd (PyTorch, JAX) handles the rest

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Numerical differentiation (approximation) ─────────────────────────────────
# Useful for checking analytical gradients (gradient checking)
def numerical_gradient(f, x, h=1e-5):
    """Central difference approximation of gradient."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus  = x.copy(); x_plus[i]  += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# ── Common derivatives used in ML ─────────────────────────────────────────────
x = np.linspace(-3, 3, 500)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

functions = [
    ('x²',          lambda x: x**2,                    lambda x: 2*x,
     'Quadratic loss'),
    ('|x|',         lambda x: np.abs(x),                lambda x: np.sign(x),
     'L1 loss'),
    ('σ(x)',        lambda x: 1/(1+np.exp(-x)),         lambda x: (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x))),
     'Sigmoid'),
    ('tanh(x)',     np.tanh,                             lambda x: 1 - np.tanh(x)**2,
     'Tanh activation'),
    ('ReLU',        lambda x: np.maximum(0, x),         lambda x: (x > 0).astype(float),
     'ReLU activation'),
    ('ln(x)',       lambda x: np.log(x + 3.01),         lambda x: 1/(x + 3.01),
     'Log (shifted)'),
    ('exp(x)',      np.exp,                              np.exp,
     'Exponential'),
    ('Leaky ReLU',  lambda x: np.where(x>=0, x, 0.1*x),lambda x: np.where(x>=0, 1., 0.1),
     'Leaky ReLU'),
]

for ax, (name, f, df, desc) in zip(axes.flatten(), functions):
    y  = f(x)
    dy = df(x)
    ax.plot(x, y,  color='steelblue', lw=2, label='f(x)')
    ax.plot(x, dy, color='tomato',    lw=2, label="f'(x)", linestyle='--')
    ax.set_title(f'{name}  ({desc})', fontsize=9)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 4)

plt.suptitle("Common Functions and Their Derivatives in ML", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_derivatives.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Derivative table ──────────────────────────────────────────────────────────
print("""
Common Derivatives in ML (memorize these):
─────────────────────────────────────────────────────────────────────
Function                  Derivative
─────────────────────────────────────────────────────────────────────
x^n                       n * x^(n-1)
e^x                       e^x                 (fixed point of differentiation)
ln(x)                     1/x
sigmoid σ(x)              σ(x)(1 - σ(x))      (output × (1 - output))
tanh(x)                   1 - tanh²(x)        (1 - output²)
ReLU: max(0, x)           0 if x<0, 1 if x>0  (undefined at 0, use 0 or 0.5)
softmax(x_i)              softmax(x_i)(δ_ij - softmax(x_j))
─────────────────────────────────────────────────────────────────────
""")
```

---

### 3.2 Partial Derivatives and Gradients

For a function of multiple variables $f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n)$:

The **partial derivative** $\frac{\partial f}{\partial x_i}$ is the derivative with respect to $x_i$ while holding all other variables constant.

The **gradient** $\nabla f(\mathbf{x})$ is the vector of all partial derivatives:
$$\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^T$$

**Key geometric fact**: The gradient points in the direction of steepest **ascent**. The negative gradient points in the direction of steepest **descent**. This is why gradient descent works.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ── Visualizing gradients on a 2D loss surface ────────────────────────────────
def loss_surface(w1, w2):
    """A simple 2D bowl-shaped loss surface."""
    return w1**2 + 3*w2**2 + w1*w2

def gradient(w1, w2):
    """Analytical gradient of loss_surface."""
    dL_dw1 = 2*w1 + w2
    dL_dw2 = 6*w2 + w1
    return np.array([dL_dw1, dL_dw2])

w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-2, 2, 100)
W1, W2 = np.meshgrid(w1, w2)
L = loss_surface(W1, W2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Contour plot with gradient field
ax = axes[0]
contour = ax.contourf(W1, W2, L, levels=30, cmap='viridis', alpha=0.8)
ax.contour(W1, W2, L, levels=15, colors='white', alpha=0.3, linewidths=0.5)
plt.colorbar(contour, ax=ax, label='Loss L(w₁, w₂)')

# Draw gradient vectors at grid points
skip = 10
for i in range(0, len(w1), skip):
    for j in range(0, len(w2), skip):
        g = gradient(w1[i], w2[j])
        g_norm = g / (np.linalg.norm(g) + 1e-8) * 0.2
        ax.annotate('', xy=[w1[i]+g_norm[0], w2[j]+g_norm[1]],
                    xytext=[w1[i], w2[j]],
                    arrowprops=dict(arrowstyle='->', color='white',
                                    lw=1.0, alpha=0.7))

ax.set_xlabel('w₁'); ax.set_ylabel('w₂')
ax.set_title('Loss Surface with Gradient Field\n'
             '(arrows point toward steepest ascent)')

# Gradient descent trajectory
ax2 = axes[1]
ax2.contourf(W1, W2, L, levels=30, cmap='viridis', alpha=0.8)
ax2.contour(W1, W2, L, levels=15, colors='white', alpha=0.3, linewidths=0.5)
plt.colorbar(ax2.contourf(W1, W2, L, levels=30, cmap='viridis', alpha=0), ax=ax2)

lr  = 0.1
w   = np.array([2.8, 1.8])
path = [w.copy()]
for _ in range(40):
    g = gradient(*w)
    w = w - lr * g
    path.append(w.copy())
path = np.array(path)

ax2.plot(path[:, 0], path[:, 1], 'w-o', markersize=4, lw=1.5, label='GD path')
ax2.plot(path[0, 0], path[0, 1], 'r*', markersize=15, label='Start')
ax2.plot(path[-1, 0], path[-1, 1], 'g*', markersize=15, label='End (≈minimum)')
ax2.set_xlabel('w₁'); ax2.set_ylabel('w₂')
ax2.set_title(f'Gradient Descent Trajectory\n(lr={lr}, {len(path)-1} steps)')
ax2.legend(fontsize=9)

plt.suptitle('Gradients and Gradient Descent', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_gradient_descent_surface.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Gradient of common ML loss functions ─────────────────────────────────────
print("Gradients of Loss Functions w.r.t. weights w")
print("─" * 60)
print("""
MSE: L = (1/n) Σ(y_hat - y)²
     ∂L/∂w = (2/n) X^T (X w - y)        [shape: (d,)]
     ∂L/∂w_j = (2/n) Σ_i x_ij (ŷ_i - y_i)

Binary Cross-Entropy: L = -(y log ŷ + (1-y) log(1-ŷ))
     where ŷ = sigmoid(w^T x)
     ∂L/∂w = (1/n) X^T (ŷ - y)          [same form as MSE!]
     Note: sigmoid's gradient cancels perfectly with cross-entropy

L2 Regularized Loss: L_reg = L + (λ/2) ||w||²
     ∂L_reg/∂w = ∂L/∂w + λ w            [just add λw to gradient]

L1 Regularized Loss: L_reg = L + λ ||w||_1
     ∂L_reg/∂w = ∂L/∂w + λ sign(w)      [sign: +1 or -1]
""")
```

---

### 3.3 The Chain Rule — Backbone of Backpropagation

The chain rule: if $z = f(g(x))$, then:
$$\frac{dz}{dx} = \frac{dz}{dg} \cdot \frac{dg}{dx}$$

For vectors — if $\mathbf{z} = f(\mathbf{g}(\mathbf{x}))$:
$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{g}} \cdot \frac{\partial \mathbf{g}}{\partial \mathbf{x}}$$

Backpropagation is **nothing more than the chain rule applied repeatedly** through a computational graph.

```python
import numpy as np

# ── Manual forward + backward pass for a tiny neural network ─────────────────
# Network: x → [Linear] → z1 → [ReLU] → a1 → [Linear] → z2 → [Sigmoid] → y_hat
# Loss: Binary Cross-Entropy  L = -(y log ŷ + (1-y) log(1-ŷ))

np.random.seed(42)
# Single sample, 3 input features, 4 hidden units, 1 output
d_in, d_h, d_out = 3, 4, 1
x  = np.array([0.5, -0.3, 0.8])   # input
y  = np.array([1.0])               # true label

# Parameters (randomly initialized for illustration)
W1 = np.random.randn(d_h, d_in) * 0.1    # (4, 3)
b1 = np.zeros(d_h)                        # (4,)
W2 = np.random.randn(d_out, d_h) * 0.1   # (1, 4)
b2 = np.zeros(d_out)                      # (1,)

# ── Forward pass ──────────────────────────────────────────────────────────────
z1    = W1 @ x + b1             # (4,)   — linear layer 1
a1    = np.maximum(0, z1)       # (4,)   — ReLU activation
z2    = W2 @ a1 + b2            # (1,)   — linear layer 2
y_hat = 1 / (1 + np.exp(-z2))  # (1,)   — sigmoid output

# Loss (binary cross-entropy)
eps   = 1e-12
L     = -(y * np.log(y_hat + eps) + (1-y) * np.log(1 - y_hat + eps))
print(f"Forward pass:")
print(f"  z1     = {z1.round(4)}")
print(f"  a1     = {a1.round(4)}")
print(f"  z2     = {z2.round(4)}")
print(f"  y_hat  = {y_hat.round(4)}")
print(f"  Loss L = {L[0]:.6f}")

# ── Backward pass (chain rule) ────────────────────────────────────────────────
# Start from loss, propagate backward

# dL/d(y_hat)  =  -(y / y_hat) + (1-y) / (1 - y_hat)
dL_dyhat = -(y / (y_hat + eps)) + (1-y) / (1 - y_hat + eps)  # (1,)

# dL/dz2 = dL/d(y_hat) * d(y_hat)/dz2
#         = (y_hat - y)  [simplification when using BCE + sigmoid together]
dL_dz2 = y_hat - y                                             # (1,)

# dL/dW2 = dL/dz2 * dz2/dW2  =  dL/dz2.T ⊗ a1
dL_dW2 = dL_dz2[:, None] * a1[None, :]                        # (1, 4)
dL_db2 = dL_dz2                                                # (1,)

# dL/da1 = W2^T * dL/dz2
dL_da1 = W2.T @ dL_dz2                                        # (4,)

# dL/dz1 = dL/da1 * da1/dz1  (ReLU derivative: 1 if z1>0, 0 otherwise)
dL_dz1 = dL_da1 * (z1 > 0).astype(float)                     # (4,)

# dL/dW1 = dL/dz1 ⊗ x
dL_dW1 = dL_dz1[:, None] * x[None, :]                        # (4, 3)
dL_db1 = dL_dz1                                               # (4,)

print(f"\nBackward pass gradients:")
print(f"  dL/dz2 = {dL_dz2.round(4)}")
print(f"  dL/dW2 shape: {dL_dW2.shape}")
print(f"  dL/da1 = {dL_da1.round(4)}")
print(f"  dL/dz1 = {dL_dz1.round(4)}")
print(f"  dL/dW1 shape: {dL_dW1.shape}")

# ── Gradient check: compare analytical vs. numerical ─────────────────────────
def forward_loss(W1, b1, W2, b2, x, y):
    z1    = W1 @ x + b1
    a1    = np.maximum(0, z1)
    z2    = W2 @ a1 + b2
    y_hat = 1 / (1 + np.exp(-z2))
    return -(y * np.log(y_hat + 1e-12) + (1-y) * np.log(1 - y_hat + 1e-12)).sum()

# Check dL/dW2[0, 0] numerically
h = 1e-5
W2_plus  = W2.copy(); W2_plus[0, 0]  += h
W2_minus = W2.copy(); W2_minus[0, 0] -= h
num_grad_W2_00 = (forward_loss(W1,b1,W2_plus,b2,x,y) -
                  forward_loss(W1,b1,W2_minus,b2,x,y)) / (2*h)

print(f"\nGradient check dL/dW2[0,0]:")
print(f"  Analytical: {dL_dW2[0, 0]:.8f}")
print(f"  Numerical:  {num_grad_W2_00:.8f}")
print(f"  Match: {np.isclose(dL_dW2[0,0], num_grad_W2_00, rtol=1e-4)}")
```

---

### 3.4 Gradient Descent

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_\mathbf{w} L(\mathbf{w}_t)$$

Where $\eta$ (eta) is the **learning rate** — the most important hyperparameter in deep learning.

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Three variants: Batch GD, SGD, Mini-batch GD ─────────────────────────────

def generate_regression_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, 2))
    X   = np.hstack([np.ones((n, 1)), X])   # Add bias column
    w_true = np.array([1.5, 2.0, -1.0])
    y   = X @ w_true + rng.normal(0, 0.5, n)
    return X, y, w_true

X, y, w_true = generate_regression_data(n=500)

def mse_loss(w, X, y):
    return ((X @ w - y)**2).mean()

def mse_gradient(w, X, y):
    return 2 * X.T @ (X @ w - y) / len(y)

# ── Batch Gradient Descent ────────────────────────────────────────────────────
def batch_gd(X, y, lr=0.1, n_epochs=100, seed=0):
    rng = np.random.default_rng(seed)
    w   = rng.standard_normal(X.shape[1])
    losses = [mse_loss(w, X, y)]
    for _ in range(n_epochs):
        g = mse_gradient(w, X, y)   # Gradient over ALL samples
        w = w - lr * g
        losses.append(mse_loss(w, X, y))
    return w, losses

# ── Stochastic Gradient Descent (SGD) ────────────────────────────────────────
def sgd(X, y, lr=0.01, n_epochs=10, seed=0):
    rng = np.random.default_rng(seed)
    w   = rng.standard_normal(X.shape[1])
    losses = [mse_loss(w, X, y)]
    n = len(y)
    for epoch in range(n_epochs):
        idx = rng.permutation(n)   # Shuffle each epoch
        for i in idx:
            xi, yi = X[i:i+1], y[i:i+1]
            g = mse_gradient(w, xi, yi)   # Gradient over ONE sample
            w = w - lr * g
        losses.append(mse_loss(w, X, y))
    return w, losses

# ── Mini-batch SGD ────────────────────────────────────────────────────────────
def mini_batch_sgd(X, y, lr=0.05, batch_size=32, n_epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    w   = rng.standard_normal(X.shape[1])
    losses = [mse_loss(w, X, y)]
    n = len(y)
    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]
            Xb, yb    = X[batch_idx], y[batch_idx]
            g = mse_gradient(w, Xb, yb)  # Gradient over BATCH
            w = w - lr * g
        losses.append(mse_loss(w, X, y))
    return w, losses

w_bgd,   losses_bgd   = batch_gd(X, y, lr=0.1,  n_epochs=100)
w_sgd,   losses_sgd   = sgd(X, y, lr=0.01, n_epochs=100)
w_mbsgd, losses_mbsgd = mini_batch_sgd(X, y, lr=0.05, batch_size=32, n_epochs=100)

print(f"True weights:       {w_true}")
print(f"Batch GD weights:   {w_bgd.round(4)}")
print(f"SGD weights:        {w_sgd.round(4)}")
print(f"Mini-batch weights: {w_mbsgd.round(4)}")

# ── Comparison plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

epochs_bgd   = range(len(losses_bgd))
epochs_sgd   = range(len(losses_sgd))
epochs_mbsgd = range(len(losses_mbsgd))

axes[0].plot(epochs_bgd,   losses_bgd,   label='Batch GD',    color='steelblue', lw=2)
axes[0].plot(epochs_sgd,   losses_sgd,   label='SGD',         color='tomato',    lw=2)
axes[0].plot(epochs_mbsgd, losses_mbsgd, label='Mini-batch',  color='green',     lw=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Convergence: GD Variants')
axes[0].legend()

# Zoom in to show noise in SGD
axes[1].plot(epochs_bgd[:50],   losses_bgd[:50],   label='Batch GD',   color='steelblue', lw=2)
axes[1].plot(epochs_sgd[:50],   losses_sgd[:50],   label='SGD',        color='tomato',    lw=2, alpha=0.7)
axes[1].plot(epochs_mbsgd[:50], losses_mbsgd[:50], label='Mini-batch', color='green',     lw=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE Loss')
axes[1].set_title('Convergence (zoomed: first 50 epochs)\nNote: SGD is noisy but fast per epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/04_gradient_descent_variants.png', dpi=140, bbox_inches='tight')
plt.show()

# ── Effect of learning rate ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for lr, color in [(0.001, '#EF9A9A'), (0.05, '#FFCC80'), (0.1, '#A5D6A7'),
                   (0.3, '#90CAF9'),   (1.0, '#CE93D8')]:
    _, losses = batch_gd(X, y, lr=lr, n_epochs=100)
    label = f'lr={lr}'
    if np.isnan(losses[-1]) or losses[-1] > 1000:
        label += ' (diverged!)'
        losses = [min(l, 10) for l in losses]
    ax.plot(losses, label=label, color=color, lw=2)

ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss (clipped at 10)')
ax.set_title('Effect of Learning Rate on Convergence\n'
             'Too small: slow. Too large: diverges.')
ax.legend()
ax.set_ylim(0, 8)
plt.tight_layout()
plt.savefig('figures/04_learning_rate.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 3.5 The Jacobian and Hessian

```python
import numpy as np

# ── Jacobian: generalization of the gradient to vector-valued functions ────────
# If f: R^n → R^m, the Jacobian J ∈ R^{m×n} where J_{ij} = ∂f_i/∂x_j
# The gradient is the Jacobian for m=1 (scalar-valued functions)

# Example: softmax Jacobian
def softmax(z):
    e = np.exp(z - z.max())   # Numerically stable
    return e / e.sum()

def softmax_jacobian(z):
    s   = softmax(z)
    n   = len(s)
    J   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i,j] = s[i] * (1 - s[j])
            else:
                J[i,j] = -s[i] * s[j]
    return J
    # Vectorized: J = diag(s) - outer(s, s)

z = np.array([2.0, 1.0, 0.5])
s = softmax(z)
J = softmax_jacobian(z)
print("Softmax Jacobian:")
print(J.round(4))
print(f"\nRow sums (should be 0, since Σ softmax(z_i) = 1):")
print(J.sum(axis=1).round(10))

# ── Hessian: second-order information ─────────────────────────────────────────
# H_{ij} = ∂²L / ∂w_i ∂w_j
# Diagonal Hessian: curvature in each direction individually
# The Hessian tells you the shape of the loss surface around a point

# For MSE loss: L = (1/n) ||Xw - y||²
# Gradient: ∇L = (2/n) X^T (Xw - y)
# Hessian:  H  = (2/n) X^T X   ← depends only on X, not on w or y!

rng = np.random.default_rng(42)
n, d = 100, 3
X    = rng.standard_normal((n, d))
H    = (2/n) * X.T @ X

eigenvalues_H = np.linalg.eigvalsh(H)
print(f"\nHessian of MSE loss:")
print(H.round(4))
print(f"\nEigenvalues of Hessian: {eigenvalues_H.round(4)}")
print(f"All positive: {np.all(eigenvalues_H > 0)}")
print("→ MSE loss is strictly convex → unique global minimum")

# Condition number: ratio of largest to smallest eigenvalue
# High condition number → ill-conditioned → gradient descent converges slowly
cond_num = eigenvalues_H.max() / eigenvalues_H.min()
print(f"\nCondition number: {cond_num:.2f}")
print("(1 = perfectly conditioned, large = ill-conditioned → slow convergence)")
```

---

### 3.6 Convexity — Why It Matters

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Convex vs. Non-convex functions ───────────────────────────────────────────
x = np.linspace(-3, 3, 500)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Convex: f(λa + (1-λ)b) ≤ λf(a) + (1-λ)f(b)
# → any local minimum is the global minimum → safe for gradient descent
f_convex   = x**2           # MSE loss w.r.t. 1D weight
axes[0].plot(x, f_convex, color='steelblue', lw=2.5)
a, b = -2., 1.5
for lam in [0.3, 0.6]:
    c = lam*a + (1-lam)*b
    # Chord between (a, f(a)) and (b, f(b))
    chord_val = lam*a**2 + (1-lam)*b**2
    axes[0].plot(c, c**2, 'o', color='tomato', markersize=8)
    axes[0].plot(c, chord_val, 's', color='green', markersize=8)
    axes[0].plot([a, b], [a**2, b**2], 'g--', alpha=0.5)
axes[0].set_title('Convex (f = x²)\nChord ≥ function → one global min')

# Non-convex: neural network loss surface
f_nonconvex = np.sin(3*x) + 0.3*x**2 - 0.5*x
axes[1].plot(x, f_nonconvex, color='tomato', lw=2.5)
local_min_x = x[np.where((np.diff(np.sign(np.diff(f_nonconvex))) > 0))[0] + 1]
local_min_y = np.sin(3*local_min_x) + 0.3*local_min_x**2 - 0.5*local_min_x
axes[1].scatter(local_min_x, local_min_y, s=100, zorder=5, color='darkred',
                label='Local minima')
global_min_x = local_min_x[np.argmin(local_min_y)]
axes[1].scatter([global_min_x], [min(local_min_y)], s=200, zorder=6,
                color='gold', marker='*', label='Global minimum')
axes[1].set_title('Non-convex (neural network)\nMany local minima + saddle points')
axes[1].legend(fontsize=8)

# Log-loss (cross-entropy) is convex in w (for logistic regression)
# But not jointly convex in all parameters of a deep network
p = np.linspace(0.01, 0.99, 500)
axes[2].plot(p, -np.log(p),     color='steelblue', lw=2, label='-log(p)  [y=1]')
axes[2].plot(p, -np.log(1-p),   color='tomato',    lw=2, label='-log(1-p) [y=0]')
axes[2].set_xlabel('Predicted probability p')
axes[2].set_ylabel('Loss')
axes[2].set_title('Cross-Entropy Loss\n(Convex in probability space)')
axes[2].legend(); axes[2].set_ylim(0, 5)

plt.suptitle('Convex vs. Non-Convex Optimization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_convexity.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
Convexity summary for ML losses:
─────────────────────────────────────────────────────────
Loss Function           Convex in w?    Notes
─────────────────────────────────────────────────────────
MSE (linear model)      YES             Unique global min
Binary Cross-Entropy    YES             (logistic regression)
  (logistic regression)
Hinge loss (SVM)        YES             Unique global min
Neural network losses   NO              Local minima, saddles
  (any hidden layer)
─────────────────────────────────────────────────────────
Why deep learning works despite non-convexity:
  1. Most local minima generalize almost as well as global min
  2. SGD noise helps escape shallow local minima
  3. Overparameterized networks have many paths to good solutions
─────────────────────────────────────────────────────────
""")
```

---

## 4. Probability and Statistics

### 4.1 Probability Fundamentals

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("""
Core probability rules:
────────────────────────────────────────────────────────────────
Rule                    Formula                   Intuition
────────────────────────────────────────────────────────────────
Sum rule               P(A) = Σ_B P(A, B)         Marginalization
Product rule           P(A, B) = P(A|B) P(B)      Joint = cond × marginal
Conditional prob       P(A|B) = P(A,B) / P(B)     Prob of A given B
Bayes' theorem         P(A|B) = P(B|A)P(A)/P(B)   Invert conditionals
Independence           P(A,B) = P(A) P(B)          No interaction
Law of total prob      P(B) = Σ_A P(B|A) P(A)     Marginalize over A
────────────────────────────────────────────────────────────────
""")

# ── Conditional probability and independence ─────────────────────────────────
# Simulate disease testing (classic example)
rng = np.random.default_rng(42)
n   = 100_000

# Disease prevalence = 1%
has_disease = rng.random(n) < 0.01

# Test: sensitivity (true positive rate) = 95%, specificity = 98%
test_positive = np.where(
    has_disease,
    rng.random(n) < 0.95,   # P(positive | disease) = 0.95
    rng.random(n) < 0.02    # P(positive | no disease) = 0.02
)

# P(disease | positive test) = ?
true_pos  = (has_disease & test_positive).sum()
false_pos = (~has_disease & test_positive).sum()
false_neg = (has_disease & ~test_positive).sum()
true_neg  = (~has_disease & ~test_positive).sum()

precision  = true_pos / (true_pos + false_pos)
recall     = true_pos / (true_pos + false_neg)

print("Medical Test Example: P(disease | positive test)")
print(f"  Disease prevalence:  1%")
print(f"  Test sensitivity:    95%")
print(f"  Test specificity:    98%")
print(f"\n  If you test positive:")
print(f"  P(disease | positive) = {precision:.4f}  ← {precision*100:.1f}% — much lower than intuition!")
print(f"  P(recall | disease)   = {recall:.4f}")
print(f"\n  Base rate neglect: most people guess ~95%, actual is ~{precision*100:.0f}%")
print(f"  Reason: false positives (0.02 × 99%) >> true positives (0.95 × 1%)")
```

---

### 4.2 Random Variables and Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("""
Key Concepts:
────────────────────────────────────────────────────────
Random variable X: maps outcomes to real numbers
PDF p(x): probability density function  (continuous)
PMF P(X=x): probability mass function   (discrete)
CDF F(x) = P(X ≤ x): cumulative distribution

Expected value E[X] = ∫ x p(x) dx       (mean)
Variance Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
Standard deviation σ = √Var(X)

Key identities:
  E[aX + b]     = a E[X] + b
  Var(aX + b)   = a² Var(X)
  E[X + Y]      = E[X] + E[Y]            (always)
  Var(X + Y)    = Var(X) + Var(Y) + 2Cov(X,Y)
  Var(X + Y)    = Var(X) + Var(Y)        (if X,Y independent)
────────────────────────────────────────────────────────
""")

# ── Computing moments ─────────────────────────────────────────────────────────
rng  = np.random.default_rng(42)
data = rng.normal(loc=5, scale=2, size=10000)

mean    = data.mean()
var     = data.var()
std     = data.std()
skew    = stats.skew(data)
kurt    = stats.kurtosis(data)  # Excess kurtosis (normal dist = 0)

print(f"Sample moments (N(5, 2²) data):")
print(f"  Mean:     {mean:.4f}  (true: 5.0)")
print(f"  Variance: {var:.4f}  (true: 4.0)")
print(f"  Std:      {std:.4f}  (true: 2.0)")
print(f"  Skewness: {skew:.4f}  (true: 0.0 — symmetric)")
print(f"  Kurtosis: {kurt:.4f}  (true: 0.0 — normal distribution)")
```

---

### 4.3 Key Distributions in ML

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
x_cont = np.linspace(-5, 5, 500)
x_int  = np.arange(0, 15)

distributions = [
    # (name, scipy_dist, params for plot, ml_use)
    ('Normal N(0,1)',    stats.norm(0, 1),    x_cont,
     'Prior on weights,\nGaussian noise,\nCentral Limit Theorem'),
    ('Normal N(μ,σ²)',  None,                 None,
     'Special case below'),
    ('Bernoulli B(p)',   stats.bernoulli(0.7), x_int[:3],
     'Binary outcomes:\nHeads/tails,\nclick/no-click'),
    ('Binomial B(n,p)',  stats.binom(10, 0.3), x_int,
     'Sum of Bernoullis:\n# successes in n trials'),
    ('Poisson λ',        stats.poisson(3.5),   x_int,
     'Count data:\nevent counts,\nword frequencies'),
    ('Uniform U(a,b)',   stats.uniform(0, 1),  np.linspace(-0.5, 1.5, 300),
     'Random initialization,\nbase sampler,\nlearning rate search'),
    ('Beta β(α,β)',      stats.beta(2, 5),      np.linspace(0, 1, 300),
     'Prior on probability,\nBayesian inference'),
    ('Exponential λ',    stats.expon(0, 1),    np.linspace(0, 6, 300),
     'Time between events,\nL1 prior (Laplace)'),
]

dist_objects = [
    ('Normal N(0,1)',   stats.norm(0,   1),    x_cont,   'steelblue'),
    ('Normal N(2,0.5)', stats.norm(2, 0.5),   x_cont,   'cornflowerblue'),
    ('Bernoulli(0.7)',  stats.bernoulli(0.7), [0,1],     'tomato'),
    ('Binomial(10,0.3)',stats.binom(10,0.3),  x_int,     'salmon'),
    ('Poisson(λ=3.5)',  stats.poisson(3.5),   x_int,     'green'),
    ('Uniform(0,1)',    stats.uniform(0,1),   np.linspace(-0.5,1.5,300), 'gold'),
    ('Beta(2,5)',        stats.beta(2,5),      np.linspace(0,1,300), 'purple'),
    ('Exponential(1)',  stats.expon(0,1),     np.linspace(0,6,300), 'orange'),
]

ml_uses = [
    'Weight initialization\nCentral Limit Theorem',
    'Prediction residuals\nGaussian processes',
    'Click/no-click\nBinary classification',
    'A/B test outcomes\nCoin flips',
    'Word counts\nEvent frequencies',
    'Random number generation\nHP search bounds',
    'Prior on probability\nBayesian modeling',
    'Time-to-event\nL1 regularization prior',
]

for ax, (name, dist, x_plot, color), use in zip(axes.flatten(), dist_objects, ml_uses):
    if dist is None:
        ax.set_visible(False)
        continue
    try:
        if name.startswith('Bernoulli') or name.startswith('Binomial') or name.startswith('Poisson'):
            probs = dist.pmf(x_plot)
            ax.bar(x_plot[:15], probs[:15], alpha=0.7, color=color, edgecolor='white')
        else:
            ax.plot(x_plot, dist.pdf(x_plot), color=color, lw=2.5)
            ax.fill_between(x_plot, dist.pdf(x_plot), alpha=0.25, color=color)
    except:
        pass

    try:
        mu  = dist.mean()
        std = dist.std()
        ax.set_title(f'{name}\nμ={mu:.2f}, σ={std:.2f}', fontsize=9)
    except:
        ax.set_title(name, fontsize=9)

    ax.text(0.98, 0.95, use, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    ax.grid(True, alpha=0.3)

plt.suptitle('Key Probability Distributions in Machine Learning',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_distributions.png', dpi=140, bbox_inches='tight')
plt.show()
```

---

### 4.4 Bayes' Theorem

$$P(\theta | \mathcal{D}) = \frac{P(\mathcal{D} | \theta) \cdot P(\theta)}{P(\mathcal{D})}$$

- $P(\theta)$: **Prior** — our belief about parameters before seeing data
- $P(\mathcal{D}|\theta)$: **Likelihood** — probability of data given parameters
- $P(\theta|\mathcal{D})$: **Posterior** — updated belief after seeing data
- $P(\mathcal{D})$: **Evidence** — normalizing constant (often intractable)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Bayesian updating — coin flip example ─────────────────────────────────────
# We observe n coin flips. We want to infer P(heads) = θ.
# Prior: Beta(α=2, β=2)  — weakly informative, centered at 0.5
# Likelihood: Binomial(n, θ)
# Posterior: Beta(α + heads, β + tails)  [conjugate prior!]

theta = np.linspace(0, 1, 500)

observations = [
    (0, 0,   'Prior only\nBeta(2,2)'),
    (1, 1,   '1H, 1T'),
    (3, 2,   '3H, 2T'),
    (7, 3,   '7H, 3T'),
    (15, 5,  '15H, 5T'),
    (30, 10, '30H, 10T'),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
colors_seq = plt.cm.Blues(np.linspace(0.4, 0.9, 6))

for ax, (heads, tails, label), color in zip(axes.flatten(), observations, colors_seq):
    alpha_prior, beta_prior = 2, 2
    alpha_post = alpha_prior + heads
    beta_post  = beta_prior  + tails

    prior     = stats.beta(alpha_prior, beta_prior)
    posterior = stats.beta(alpha_post,  beta_post)

    ax.plot(theta, prior.pdf(theta),     color='gray',  lw=1.5, linestyle='--', label='Prior')
    ax.plot(theta, posterior.pdf(theta), color=color,   lw=2.5, label='Posterior')
    ax.fill_between(theta, posterior.pdf(theta), alpha=0.3, color=color)
    ax.axvline(alpha_post / (alpha_post + beta_post), color='tomato',
               lw=2, linestyle=':', label=f'MAP={alpha_post/(alpha_post+beta_post):.2f}')
    ax.set_title(label + f'\nPosterior: Beta({alpha_post},{beta_post})', fontsize=10)
    ax.set_xlabel('θ = P(heads)')
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Bayesian Updating — Posterior concentrates with more data",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_bayesian_updating.png', dpi=140, bbox_inches='tight')
plt.show()

print("""
Bayesian vs. Frequentist:
─────────────────────────────────────────────────────────
Frequentist: θ is a fixed (unknown) constant.
             P(data | θ) → find θ that maximizes it (MLE).

Bayesian:    θ is a random variable with a prior distribution.
             We compute P(θ | data) via Bayes' theorem.
             Result is a full posterior distribution, not a point.

In ML:
  MLE = frequentist maximum likelihood estimation
  MAP = Maximum A Posteriori (mode of posterior)
  Full Bayesian = integrate over all θ (usually intractable)

L2 regularization = MAP estimation with Gaussian prior on weights
L1 regularization = MAP estimation with Laplace prior on weights
─────────────────────────────────────────────────────────
""")
```

---

### 4.5 Maximum Likelihood Estimation

MLE finds parameters that maximize the probability of observing the data.

$$\hat{\theta}_\text{MLE} = \arg\max_\theta \log P(\mathcal{D}|\theta) = \arg\max_\theta \sum_{i=1}^n \log p(x^{(i)}|\theta)$$

We maximize the **log-likelihood** instead of the likelihood because:
1. Sums are numerically stabler than products.
2. Logarithm is monotone → same argmax.
3. Results in additive gradients.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar

# ── MLE for Gaussian: derives the formula for sample mean and variance ─────────
# p(x | μ, σ²) = (1/√(2πσ²)) exp(-(x-μ)² / (2σ²))
# log p(x | μ, σ²) = -0.5 log(2πσ²) - (x-μ)² / (2σ²)
# Sum over n samples:
# ℓ(μ, σ²) = -n/2 log(2πσ²) - 1/(2σ²) Σ(x_i - μ)²
# Set ∂ℓ/∂μ = 0  →  μ̂_MLE = (1/n) Σ x_i = sample mean
# Set ∂ℓ/∂σ² = 0 →  σ̂²_MLE = (1/n) Σ(x_i - μ̂)²  (biased! vs unbiased n-1)

rng  = np.random.default_rng(42)
data = rng.normal(loc=3.0, scale=1.5, size=100)

mu_mle    = data.mean()
sigma_mle = data.std(ddof=0)   # MLE: divide by n (biased)
sigma_unb = data.std(ddof=1)   # Unbiased: divide by n-1

print("MLE for Gaussian parameters:")
print(f"  True:     μ=3.0,  σ=1.5")
print(f"  MLE:      μ̂={mu_mle:.4f}, σ̂={sigma_mle:.4f}  (biased)")
print(f"  Unbiased: μ̂={mu_mle:.4f}, σ̂={sigma_unb:.4f}  (divide by n-1)")

# ── MLE: minimizing NLL = minimizing MSE (for Gaussian noise) ─────────────────
# For linear regression with Gaussian noise:
#   y = Xw + ε,  ε ~ N(0, σ²)
# The MLE of w is exactly the Ordinary Least Squares solution!
# Minimizing NLL ↔ Minimizing MSE (they differ only by constants)

print("""
MLE ↔ Loss Function Connection:
─────────────────────────────────────────────────────────────────
Model assumption            MLE loss       Standard name
─────────────────────────────────────────────────────────────────
y | X ~ N(Xw, σ²I)          MSE            Linear Regression
y | X ~ Bernoulli(σ(Xw))    Binary cross-entropy  Logistic Regression
y | X ~ Categorical(softmax) Cross-entropy  Multiclass Classification
y | X ~ Poisson(exp(Xw))     Poisson deviance  Count regression
─────────────────────────────────────────────────────────────────
Every standard loss function is the NLL of some probability model.
""")

# ── Numerical MLE example ─────────────────────────────────────────────────────
# MLE for Poisson: data = word counts in documents
word_counts = rng.poisson(lam=7.3, size=500)

def neg_log_likelihood_poisson(lam, data):
    if lam <= 0:
        return np.inf
    return -np.sum(stats.poisson.logpmf(data, mu=lam))

result = minimize_scalar(neg_log_likelihood_poisson, bounds=(1, 20),
                         method='bounded', args=(word_counts,))
print(f"Poisson MLE: λ̂ = {result.x:.4f}  (true: 7.3, sample mean: {word_counts.mean():.4f})")
print("Note: MLE for Poisson = sample mean (provable analytically)")
```

---

### 4.6 Information Theory — Entropy and KL Divergence

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

# ── Shannon Entropy: H(p) = -Σ p(x) log p(x) ────────────────────────────────
# Entropy measures average uncertainty / information content.
# H=0: completely certain (one outcome has probability 1)
# H=log(n): maximum uncertainty (uniform over n outcomes)
# Unit: bits (log base 2) or nats (natural log)

def entropy(p, base=2):
    p = np.array(p)
    p = p[p > 0]   # Avoid log(0)
    return -np.sum(p * np.log2(p)) if base == 2 else -np.sum(p * np.log(p))

# Compare distributions
print("Entropy of different distributions (4 outcomes):")
uniform  = [0.25, 0.25, 0.25, 0.25]
skewed   = [0.7,  0.1,  0.1,  0.1]
certain  = [1.0,  0.0,  0.0,  0.0]
moderate = [0.5,  0.3,  0.1,  0.1]

for name, dist in [('Uniform [0.25,0.25,0.25,0.25]', uniform),
                   ('Moderate [0.5,0.3,0.1,0.1]',    moderate),
                   ('Skewed  [0.7,0.1,0.1,0.1]',     skewed),
                   ('Certain [1.0,0.0,0.0,0.0]',     certain)]:
    print(f"  H = {entropy(dist):.4f} bits   {name}")

print(f"\n  Max entropy = log2(4) = {np.log2(4):.4f} bits (uniform)")

# ── Cross-Entropy: H(p, q) = -Σ p(x) log q(x) ───────────────────────────────
# Measures expected surprise when using q to encode messages from p.
# In ML: p = true labels, q = model predictions
# Cross-entropy ≥ entropy, equality iff p = q

def cross_entropy(p_true, q_pred, eps=1e-12):
    p = np.array(p_true)
    q = np.array(q_pred) + eps
    return -np.sum(p * np.log(q))

y_true = [0, 0, 1, 0]           # One-hot: class 2 is correct
y_pred_good = [0.02, 0.03, 0.9, 0.05]   # Confident and correct
y_pred_bad  = [0.1,  0.6,  0.2, 0.1]    # Confident and wrong

print(f"\nCross-Entropy Loss:")
print(f"  Good prediction:  CE = {cross_entropy(y_true, y_pred_good):.4f}")
print(f"  Bad prediction:   CE = {cross_entropy(y_true, y_pred_bad):.4f}")
print(f"  Lower is better — cross-entropy penalizes confident wrong predictions harshly")

# ── KL Divergence: KL(p||q) = Σ p(x) log(p(x)/q(x)) = H(p,q) - H(p) ─────────
# Measures how much information is lost when using q instead of p.
# KL(p||q) ≥ 0  (Gibbs inequality)
# KL(p||q) = 0 iff p = q
# KL is NOT symmetric: KL(p||q) ≠ KL(q||p)

def kl_divergence(p, q, eps=1e-12):
    p = np.array(p) + eps
    q = np.array(q) + eps
    return np.sum(p * np.log(p / q))

p = [0.4, 0.3, 0.2, 0.1]
q = [0.25, 0.25, 0.25, 0.25]

print(f"\nKL Divergence:")
print(f"  KL(p||q) = {kl_divergence(p, q):.4f}")
print(f"  KL(q||p) = {kl_divergence(q, p):.4f}")
print(f"  KL is NOT symmetric!")
print(f"  KL(p||p) = {kl_divergence(p, p):.4f}  (zero when distributions are identical)")

# ── Where these appear in ML ────────────────────────────────────────────────
print("""
Information Theory in ML:
──────────────────────────────────────────────────────────────────
Concept         Formula                  ML application
──────────────────────────────────────────────────────────────────
Entropy H(p)    -Σ p log p               Decision tree splitting
                                          Feature selection (MI)
Cross-entropy   -Σ p log q               Classification loss function
H(p,q)                                    = NLL under categorical model

KL Divergence   Σ p log(p/q)             VAE regularization term
KL(p||q)                                  Policy optimization (PPO)
                                          Knowledge distillation

Mutual Info     I(X;Y) = H(X) - H(X|Y)  Feature relevance
I(X;Y)                                    Self-supervised learning

Note: Cross-entropy loss = H(y_true, y_pred)
      = H(y_true) + KL(y_true || y_pred)
      Since H(y_true) is constant w.r.t. model params,
      minimizing cross-entropy = minimizing KL divergence!
──────────────────────────────────────────────────────────────────
""")
```

---

## 5. Putting It All Together: Deriving Linear Regression

We now have all the tools to derive **Ordinary Least Squares (OLS) linear regression from scratch** — combining all four mathematical areas.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("""
=============================================================
 DERIVING LINEAR REGRESSION FROM FIRST PRINCIPLES
=============================================================

Setup:
  n samples, d features
  X ∈ R^{n×d}  (design matrix, includes bias column)
  y ∈ R^n      (target vector)
  w ∈ R^d      (weights to learn)

Model: ŷ = Xw

Probabilistic interpretation (MLE):
  y^(i) = w^T x^(i) + ε^(i),  ε^(i) ~ N(0, σ²)

Likelihood: L(w) = Π_i p(y^(i) | x^(i), w)
  = Π_i (1/√(2πσ²)) exp(-(y^(i) - w^T x^(i))² / (2σ²))

Log-likelihood: ℓ(w) = -n/2 log(2πσ²) - 1/(2σ²) Σ_i (y^(i) - w^T x^(i))²

Maximizing ℓ(w) ↔ Minimizing Σ(y - Xw)² = ||y - Xw||²_2

Loss function (MSE): L(w) = (1/n) ||Xw - y||²

Gradient: ∇_w L = (2/n) X^T (Xw - y)

Setting ∇_w L = 0:
  X^T (Xw - y) = 0
  X^T Xw = X^T y
  w = (X^T X)^{-1} X^T y   ← Normal Equations (Closed-Form Solution)
""")

# ── Closed-form OLS ────────────────────────────────────────────────────────────
np.random.seed(42)
n = 200

X_raw = np.random.randn(n, 2)
X     = np.hstack([np.ones((n, 1)), X_raw])   # Add bias (intercept) column
w_true = np.array([1.0, 2.5, -1.5])           # [intercept, w1, w2]
y     = X @ w_true + np.random.randn(n) * 0.5

# Normal equations: w = (X^T X)^{-1} X^T y
XtX   = X.T @ X                # (3, 3)
Xty   = X.T @ y                # (3,)
w_ols = np.linalg.solve(XtX, Xty)  # More stable than: np.linalg.inv(XtX) @ Xty

print("Closed-Form OLS (Normal Equations):")
print(f"  True weights:       {w_true}")
print(f"  Estimated weights:  {w_ols.round(4)}")

# Residuals and R²
y_hat    = X @ w_ols
residuals = y - y_hat
ss_res   = (residuals**2).sum()
ss_tot   = ((y - y.mean())**2).sum()
r_squared = 1 - ss_res / ss_tot
print(f"\n  MSE:  {(residuals**2).mean():.4f}")
print(f"  R²:   {r_squared:.4f}")

# ── Gradient descent solution (should match) ──────────────────────────────────
w_gd = np.zeros(3)
lr = 0.1
for _ in range(1000):
    grad = (2/n) * X.T @ (X @ w_gd - y)
    w_gd -= lr * grad

print(f"\nGradient Descent Solution (1000 steps, lr=0.1):")
print(f"  Estimated weights: {w_gd.round(4)}")
print(f"  Matches OLS:       {np.allclose(w_ols, w_gd, atol=1e-3)}")

# ── Why we prefer gradient descent for large n, d ────────────────────────────
print("""
When to use Normal Equations vs. Gradient Descent:
─────────────────────────────────────────────────────────
Normal Equations:
  ✓ Exact solution, no hyperparameters
  ✓ Fast for small datasets (n, d < 10,000)
  ✗ Requires computing (X^T X)^{-1}: O(d³) time, O(d²) memory
  ✗ Fails if X^T X is singular (collinear features)
  ✗ Cannot apply regularization easily (Ridge adds λI: w = (X^T X + λI)^{-1} X^T y)

Gradient Descent:
  ✓ Scales to millions of samples and thousands of features
  ✓ Works for non-convex losses (neural networks)
  ✓ Easy to add regularization (just modify gradient)
  ✗ Requires tuning learning rate and stopping criterion
  ✗ Approximate — never exactly converges
─────────────────────────────────────────────────────────
""")

# ── Geometric interpretation ──────────────────────────────────────────────────
print("""
Geometric Interpretation of OLS:
  The fitted values ŷ = Xw_ols are the ORTHOGONAL PROJECTION of y
  onto the column space of X.

  This means: (y - ŷ) ⊥ ŷ  (residuals are perpendicular to predictions)
  Verify: X^T (y - ŷ) = X^T e = 0  (normal equations!)
""")
e = y - X @ w_ols
print(f"  X^T e ≈ 0: {np.allclose(X.T @ e, 0, atol=1e-8)}")
print(f"  ŷ · e ≈ 0: {np.allclose(np.dot(y_hat, e), 0, atol=1e-6)}")
```

---

## 6. Numerical Stability — The Practitioner's Concern

```python
import numpy as np

print("=" * 60)
print("NUMERICAL STABILITY — REAL PROBLEMS IN ML")
print("=" * 60)

# ── Problem 1: Log(0) and overflow in softmax ────────────────────────────────
def softmax_unstable(z):
    """Will overflow for large z values."""
    return np.exp(z) / np.exp(z).sum()

def softmax_stable(z):
    """Subtract max before exponentiation — mathematically identical, numerically safe."""
    z_shifted = z - z.max()
    e = np.exp(z_shifted)
    return e / e.sum()

z_normal = np.array([2.0, 1.0, 0.1])
z_large  = np.array([1000., 999., 998.])   # exp(1000) = ∞

print("\nSoftmax stability:")
print(f"  Normal input: unstable={softmax_unstable(z_normal)}, stable={softmax_stable(z_normal)}")
try:
    result = softmax_unstable(z_large)
    print(f"  Large input unstable: {result}")
except:
    print(f"  Large input unstable: ERROR (overflow)")
print(f"  Large input stable:   {softmax_stable(z_large)}")

# ── Problem 2: Log-sum-exp trick ──────────────────────────────────────────────
# log Σ_i exp(z_i) = max(z) + log Σ_i exp(z_i - max(z))
def log_sum_exp_unstable(z):
    return np.log(np.exp(z).sum())

def log_sum_exp_stable(z):
    c = z.max()
    return c + np.log(np.exp(z - c).sum())

z_test = np.array([1000., 999., 998.])
print(f"\nLog-sum-exp:")
print(f"  Unstable: {log_sum_exp_unstable(z_test)}")   # inf
print(f"  Stable:   {log_sum_exp_stable(z_test):.4f}")

# ── Problem 3: Catastrophic cancellation ─────────────────────────────────────
# Subtracting two nearly equal numbers destroys precision
a = 1.0000001
b = 1.0000000
print(f"\nCatastrophic cancellation:")
print(f"  a = {a},  b = {b}")
print(f"  a - b = {a - b}  (only 1 significant digit left!)")
print(f"  Correct answer: 1e-7")
print(f"  This is why: (a² - b²) = (a+b)(a-b) is better than direct computation")

# ── Problem 4: Vanishing/exploding gradients ─────────────────────────────────
print("\nVanishing gradients:")
sigmoid_prime = lambda x: (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x)))
max_sigmoid_grad = sigmoid_prime(0)   # = 0.25 at x=0
print(f"  Max sigmoid gradient: {max_sigmoid_grad}")
print(f"  After 10 layers: {max_sigmoid_grad**10:.8f}  (essentially zero!)")
print(f"  After 20 layers: {max_sigmoid_grad**20:.2e}")
print(f"  → This is why ReLU replaced sigmoid in deep networks")
print(f"  → ReLU gradient = 1 for positive inputs: {1**10}")

# ── Problem 5: Division by zero in normalization ──────────────────────────────
def normalize_safe(X, eps=1e-8):
    """Add epsilon to std to prevent division by zero."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    return (X - mean) / (std + eps)

rng = np.random.default_rng(42)
X_test = np.column_stack([
    rng.standard_normal(100),   # Normal feature
    np.zeros(100),               # Constant feature (std=0!)
    rng.standard_normal(100),
])
print(f"\nNormalization with zero-std column:")
print(f"  Without eps: would produce inf/nan")
X_normed = normalize_safe(X_test)
print(f"  With eps=1e-8: {np.isnan(X_normed).sum()} NaNs,  {np.isinf(X_normed).sum()} Infs")

# ── Summary of numerical best practices ──────────────────────────────────────
print("""
Numerical Stability Checklist:
──────────────────────────────────────────────────────────────────
✓ Always use log-probabilities, not raw probabilities
✓ Implement softmax with max subtraction
✓ Use log-sum-exp trick wherever possible
✓ Add ε (1e-7 to 1e-8) to denominators and log arguments
✓ Use float32 for GPU, float64 for CPU (higher precision when needed)
✓ Use np.linalg.solve instead of explicit matrix inversion
✓ Clip logits before sigmoid/softmax in custom implementations
✓ Use PyTorch's built-in losses (they implement these tricks correctly)
✓ Monitor gradient norms — if they explode, use gradient clipping
✓ If loss is NaN: check learning rate, check for log(0), check input scale
──────────────────────────────────────────────────────────────────
""")
```

---

## 7. Summary

### ✅ Must-Remember Mental Models

**Linear Algebra:**
- A matrix is a linear transformation. $\mathbf{A}\mathbf{x}$ transforms vector $\mathbf{x}$ into a new space.
- **Dot product** = projection = similarity measure. $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$.
- **L2 norm** = Euclidean distance. **L1 norm** = sum of absolute values → induces sparsity.
- **Eigenvectors** are directions not rotated by a matrix — only scaled. Eigenvalues are the scale factors.
- **SVD** decomposes any matrix. Top-k truncation gives the best low-rank approximation (used in PCA, recommendation systems).
- **PCA = SVD of the centered data matrix.** Principal components = right singular vectors.

**Calculus:**
- **Gradient** points uphill. **Negative gradient** points downhill. Gradient descent follows the negative gradient.
- **Chain rule**: $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$. Backpropagation is just the chain rule applied to a computational graph.
- The gradient of MSE loss is $\nabla_w L = \frac{2}{n} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y})$.
- The gradient of sigmoid cross-entropy is $\nabla_w L = \frac{1}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$ — same form as MSE.
- **Convex loss → global minimum guaranteed.** Linear/logistic regression losses are convex. Neural network losses are not.

**Probability:**
- **MLE** finds parameters maximizing the likelihood of observed data.
- Every standard loss function is the negative log-likelihood of a probability model: MSE ↔ Gaussian noise, cross-entropy ↔ categorical distribution.
- **L2 regularization = MAP estimation with Gaussian prior.** L1 regularization = MAP with Laplace prior.
- **Entropy** measures uncertainty. **Cross-entropy** is the loss. **KL divergence** measures distribution mismatch.
- Minimizing cross-entropy ≡ minimizing KL divergence ≡ maximizing likelihood — all the same thing.

**Numerical:**
- Use `np.linalg.solve(A, b)` not `inv(A) @ b`.
- Softmax: subtract max before exponentiation.
- Always add $\epsilon$ to denominators and log arguments.
- Vanishing gradients: sigmoid gradient ≤ 0.25 per layer. After 20 layers: essentially zero. Use ReLU.

---

## 8. Exercises

**Linear Algebra:**

1. Prove that the L2 norm is invariant under orthogonal transformations: show that $\|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$ for any orthogonal matrix $\mathbf{Q}$.

2. Implement PCA from scratch using only NumPy (no Scikit-learn). Apply it to the Iris dataset (standardized). Show that your results match `sklearn.decomposition.PCA`. Report the explained variance ratio of each component.

3. Implement a rank-$k$ image compression function using SVD. Apply it to a grayscale image of your choice. Plot the compressed images for $k = 1, 5, 10, 50$ and the percentage of variance explained at each rank.

**Calculus:**

4. Derive the gradient of the Ridge (L2 regularized) regression loss:
   $$L(\mathbf{w}) = \frac{1}{n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$$
   Then implement the gradient descent update rule and verify your solution converges to $(\mathbf{X}^T\mathbf{X} + \lambda n \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$.

5. Implement a gradient checker function that compares analytical and numerical gradients for any function `f(w)`. The relative error should be below $10^{-5}$ for correct implementations. Test it on the MSE gradient.

6. Visualize the gradient descent path on a 2D non-convex surface (use $L = \sin(x) + \cos(y) + 0.1(x^2 + y^2)$). Show that: (a) different starting points lead to different local minima, (b) momentum helps escape shallow traps.

**Probability:**

7. Simulate the Monty Hall problem: verify through 100,000 trials that switching doors wins 2/3 of the time. Explain the result using Bayes' theorem.

8. Derive the cross-entropy gradient for multiclass logistic regression (softmax + cross-entropy). Show that $\nabla_z L = \hat{\mathbf{y}} - \mathbf{y}$ where $z$ are the logits. Verify with numerical gradient checking.

9. Show empirically that L2 regularization corresponds to a Gaussian prior. Implement both MAP estimation and L2-regularized OLS and verify they give the same result.

**Challenge:**

10. Implement automatic differentiation (autograd) for scalar-valued functions using forward-mode differentiation with dual numbers:
    ```python
    class Dual:
        def __init__(self, val, deriv):
            self.val   = val    # f(x)
            self.deriv = deriv  # f'(x)
        
        def __add__(self, other): ...
        def __mul__(self, other): ...
        # etc.
    
    # Usage: to compute f'(x0), evaluate f(Dual(x0, 1.0))
    ```
    Implement `+`, `-`, `*`, `/`, `**`, `sin`, `cos`, `exp`, `log`.
    Verify against numerical gradients on several functions.

---

## 9. References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 2 (Linear Algebra), Chapter 3 (Probability), Chapter 4 (Numerical Computation). MIT Press. — The canonical deep learning math reference.
- Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. — The best linear algebra textbook.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. — Authoritative Bayesian ML reference.
- Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press. — Free at mml-book.github.io.
- Petersen, K. B., & Pedersen, M. S. (2012). *The Matrix Cookbook*. — Formula reference: matrixcookbook.com.
- 3Blue1Brown: *Essence of Linear Algebra* series — youtube.com/3blue1brown — best geometric intuition.

---

*Previous: [Chapter 3 — Data Wrangling & EDA](03_eda.md)*  
*Next: [Chapter 5 — Linear Regression](05_linear_regression.md)*

*In Chapter 5, we put all this mathematics to work: we derive, implement, and fully understand linear regression — from the normal equations to regularization to the geometric interpretation of projection.*

---

> **Chapter 4 complete.** All code is in `notebooks/04_math.ipynb`. Every derivation is paired with a numerical verification — if you don't trust the math, check it with code.
