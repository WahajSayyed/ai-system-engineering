import math
import numpy as np

def flash_attention_forward(Q, K, V, Br=64, Bc=64, causal=False):
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    O = np.empty_like(Q)
    L = np.empty(N, dtype=Q.dtype)
    for r0 in range(0, N, Br):                        # loop over Q tiles
        r1 = min(r0 + Br, N)
        Qi = Q[r0:r1]
        br = r1 - r0
        m = np.full(br, -np.inf, dtype=Q.dtype)       # running max, one per row
        l = np.zeros(br, dtype=Q.dtype)               # running sum
        acc = np.zeros((br, d), dtype=Q.dtype)        # running UNNORMALISED output
        for c0 in range(0, N, Bc):                    # loop over K/V tiles
            c1 = min(c0 + Bc, N)
            if causal and c0 > r1 - 1:                # tile is entirely right of the diagonal
                break
            Kj, Vj = K[c0:c1], V[c0:c1]
            S = (Qi @ Kj.T) * scale                   # [br, bc] score tile
            if causal and c1 - 1 > r0:                # tile touches the diagonal: mask inside it
                rows = np.arange(r0, r1)[:, None]
                cols = np.arange(c0, c1)[None, :]
                S = np.where(cols <= rows, S, -np.inf)
            m_new = np.maximum(m, S.max(axis=1))
            m_safe = np.where(np.isneginf(m_new), 0.0, m_new)   # Chapter 2 guard, vectorised
            alpha = np.exp(m - m_safe)                # rescale factor exp(m_old - m_new)
            P = np.exp(S - m_safe[:, None])           # unnormalised probabilities
            l = alpha * l + P.sum(axis=1)
            acc = alpha[:, None] * acc + P @ Vj
            m = m_new
        O[r0:r1] = acc / l[:, None]                   # normalise ONCE per Q tile
        L[r0:r1] = m + np.log(l)                      # logsumexp of each row
    return O, L
