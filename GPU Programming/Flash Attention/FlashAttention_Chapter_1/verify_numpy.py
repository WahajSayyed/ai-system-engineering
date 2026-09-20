"""Numpy twin of the PyTorch listing in Chapter 1. Same operations, same order.
Used to check the math and produce the worked example (this sandbox has no torch / GPU)."""
import math
import numpy as np

def naive_attention_np(q, k, v, causal=False, scale=None):
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    s = q @ np.swapaxes(k, -2, -1)          # S = Q K^T           [.., N, N]
    s = s * scale
    if causal:
        n_q, n_k = s.shape[-2], s.shape[-1]
        mask = np.tril(np.ones((n_q, n_k), dtype=bool))
        s = np.where(mask, s, -np.inf)
    m = s.max(axis=-1, keepdims=True)        # safe softmax
    e = np.exp(s - m)
    p = e / e.sum(axis=-1, keepdims=True)    # P                    [.., N, N]
    return p @ v, s, p

def attention_row_from_definition(q, k, v, i, causal=False):
    """o_i = sum_j softmax_j(q_i . k_j / sqrt(d)) * v_j, written with plain loops."""
    d = q.shape[-1]
    n = k.shape[0]
    js = range(i + 1) if causal else range(n)
    logits = np.array([q[i] @ k[j] / math.sqrt(d) for j in js])
    w = np.exp(logits - logits.max())
    w = w / w.sum()
    return sum(w[t] * v[j] for t, j in enumerate(js))

np.set_printoptions(precision=4, suppress=True)

# ---- 1. tiny worked example ------------------------------------------------
Q = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
K = Q.copy()
V = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
O, S, P = naive_attention_np(Q, K, V)
print("== worked example (N=3, d=2) ==")
print("Q K^T (before scaling):\n", Q @ K.T)
print("S = QK^T / sqrt(2):\n", S)
print("P = softmax rows:\n", P)
print("row sums of P:", P.sum(-1))
print("O = P V:\n", O)
Oc, Sc, Pc = naive_attention_np(Q, K, V, causal=True)
print("causal P:\n", Pc)
print("causal O:\n", Oc)

# ---- 2. batched random check vs definition (loop) ---------------------------
rng = np.random.default_rng(0)
B, H, N, D = 2, 3, 16, 8
q = rng.standard_normal((B, H, N, D)); k = rng.standard_normal((B, H, N, D)); v = rng.standard_normal((B, H, N, D))
for causal in (False, True):
    out, _, _ = naive_attention_np(q, k, v, causal=causal)
    err = 0.0
    for b in range(B):
        for h in range(H):
            for i in range(N):
                ref = attention_row_from_definition(q[b, h], k[b, h], v[b, h], i, causal)
                err = max(err, np.abs(out[b, h, i] - ref).max())
    print(f"\nmax |vectorised - definition| (causal={causal}): {err:.2e}")

# ---- 3. why we subtract the row max -----------------------------------------
print("\n== overflow without max-subtraction ==")
print("exp(float16(12))  =", np.exp(np.float16(12.0)))
print("float16 max       =", np.finfo(np.float16).max, " ln(max) =", round(math.log(float(np.finfo(np.float16).max)), 3))
print("float32 exp(88)   =", np.exp(np.float32(88.0)), " exp(89) =", np.exp(np.float32(89.0)))
x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
with np.errstate(all="ignore"):
    naive = np.exp(x) / np.exp(x).sum()
safe = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
print("naive softmax([1000,1001,1002]) =", naive)
print("safe  softmax([1000,1001,1002]) =", safe)
print("shift invariance holds:", np.allclose(safe, np.exp(x - 1000) / np.exp(x - 1000).sum()))

# ---- 4. a fully masked row -> NaN (foreshadows Ch 12) ----------------------
row = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
with np.errstate(all="ignore"):
    print("\nsoftmax of an all -inf row:", np.exp(row - row.max()) / np.exp(row - row.max()).sum())
