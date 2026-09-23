"""Chapter 3: tiled attention (the FlashAttention-2 forward loop) on the CPU, in NumPy.

Single head: Q, K, V are [N, d] arrays. Nothing here ever allocates an N x N array
except the *reference* implementation (naive_attention_np) that we test against.
"""
import math

import numpy as np

NEG_INF = -np.inf


# --------------------------------------------------------------------------- #
# Reference: the Chapter 1 algorithm for one head (materialises S and P)
# --------------------------------------------------------------------------- #
def naive_attention_np(Q, K, V, causal=False, window=None):
    """window=w (with causal): query i may only see keys i-w+1 .. i  (sliding-window attention)."""
    N, d = Q.shape
    S = (Q @ K.T) / math.sqrt(d)
    if causal:
        allowed = np.tril(np.ones((N, N), dtype=bool))
        if window is not None:
            allowed &= np.triu(np.ones((N, N), dtype=bool), k=-(window - 1))
        S = np.where(allowed, S, NEG_INF)
    m = S.max(axis=1, keepdims=True)
    E = np.exp(S - m)
    l = E.sum(axis=1, keepdims=True)
    return (E / l) @ V, (m + np.log(l))[:, 0], S       # O, L (logsumexp), S


# --------------------------------------------------------------------------- #
# Tiled forward pass, Q-tiles in the outer loop (FlashAttention-2's loop order)
# --------------------------------------------------------------------------- #
def flash_attention_forward(Q, K, V, Br=64, Bc=64, causal=False, io=None, window=None,
                            _bad_rescale=False, _no_guard=False):
    """Returns (O, L). If `io` is a dict, counts the elements moved between
    'slow' memory (the full arrays) and 'fast' memory (the tile-sized locals)."""
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    O = np.empty_like(Q)
    L = np.empty(N, dtype=Q.dtype)
    for r0 in range(0, N, Br):                        # ---- loop over Q tiles
        r1 = min(r0 + Br, N)
        Qi = Q[r0:r1]                                 # load Q_i
        if io is not None: io["Q read"] += Qi.size
        br = r1 - r0
        m = np.full(br, NEG_INF, dtype=Q.dtype)       # running max, one per row
        l = np.zeros(br, dtype=Q.dtype)               # running sum, one per row
        acc = np.zeros((br, d), dtype=Q.dtype)        # running UNNORMALISED output
        for c0 in range(0, N, Bc):                    # ---- loop over K/V tiles
            c1 = min(c0 + Bc, N)
            if causal and c0 > r1 - 1:                # every column is right of every row: fully masked
                break                                 # (later tiles are even further right)
            if window is not None and c1 - 1 <= r0 - window:   # tile lies entirely left of the window
                continue
            Kj, Vj = K[c0:c1], V[c0:c1]               # load K_j, V_j
            if io is not None: io["K read"] += Kj.size; io["V read"] += Vj.size
            S = (Qi @ Kj.T) * scale                   # [br, bc] score tile
            if causal and (c1 - 1 > r0 or window is not None):   # tile needs masking inside it
                rows = np.arange(r0, r1)[:, None]
                cols = np.arange(c0, c1)[None, :]
                allowed = cols <= rows
                if window is not None:
                    allowed &= cols > rows - window
                S = np.where(allowed, S, NEG_INF)
            m_new = np.maximum(m, S.max(axis=1))
            m_safe = m_new if _no_guard else np.where(np.isneginf(m_new), 0.0, m_new)   # the Chapter 2 guard, vectorised
            alpha = np.exp(m - m_safe)                # rescale factor exp(m_old - m_new)
            if _bad_rescale:                          # the factor as PRINTED in the FA-2 paper (see chapter):
                alpha = np.where(alpha > 0, 1.0 / np.where(alpha > 0, alpha, 1.0), 0.0)   # its inverse
            P = np.exp(S - m_safe[:, None])           # [br, bc] unnormalised probabilities
            l = alpha * l + P.sum(axis=1)
            acc = alpha[:, None] * acc + P @ Vj
            m = m_new
        with np.errstate(divide="ignore"):
            L[r0:r1] = m + np.log(l)                  # logsumexp of the row
        O[r0:r1] = acc / np.where(l == 0, 1, l)[:, None]        # normalise ONCE per Q tile
        if io is not None: io["O written"] += acc.size; io["L written"] += br
    return O, L


def flash_attention_multihead(Q, K, V, **kw):
    """[B, H, N, d] inputs: every (b, h) pair is an independent single-head problem."""
    B, H, N, d = Q.shape
    O = np.empty_like(Q)
    L = np.empty((B, H, N), dtype=Q.dtype)
    for b in range(B):
        for h in range(H):
            O[b, h], L[b, h] = flash_attention_forward(Q[b, h], K[b, h], V[b, h], **kw)
    return O, L


# --------------------------------------------------------------------------- #
# Traffic model: elements moved between slow and fast memory, from loop structure
# --------------------------------------------------------------------------- #
def io_dry_run(N, d, Br, Bc, order):
    """Count element transfers for the two loop orders without doing any arithmetic."""
    t = dict.fromkeys(["reads", "writes"], 0)
    tiles_r = [(r0, min(r0 + Br, N)) for r0 in range(0, N, Br)]
    tiles_c = [(c0, min(c0 + Bc, N)) for c0 in range(0, N, Bc)]
    if order == "Q-outer":            # FlashAttention-2 order
        for r0, r1 in tiles_r:
            t["reads"] += (r1 - r0) * d                                   # Q_i
            for c0, c1 in tiles_c:
                t["reads"] += 2 * (c1 - c0) * d                           # K_j, V_j
            t["writes"] += (r1 - r0) * d + (r1 - r0)                      # O_i, L_i
    elif order == "KV-outer":         # original FlashAttention order (Algorithm 1 in the FA-1 paper)
        for c0, c1 in tiles_c:
            t["reads"] += 2 * (c1 - c0) * d                               # K_j, V_j
            for r0, r1 in tiles_r:
                br = r1 - r0
                t["reads"] += br * d + br * d + 2 * br                    # Q_i, O_i, l_i, m_i
                t["writes"] += br * d + 2 * br                            # O_i, l_i, m_i
    else:
        raise ValueError(order)
    return t["reads"] + t["writes"]


def causal_tile_counts(N, Br, Bc):
    total = math.ceil(N / Br) * math.ceil(N / Bc)
    kept = 0
    for r0 in range(0, N, Br):
        r1 = min(r0 + Br, N)
        for c0 in range(0, N, Bc):
            if not c0 > r1 - 1:
                kept += 1
    return kept, total


if __name__ == "__main__":
    import tracemalloc
    rng = np.random.default_rng(0)
    np.set_printoptions(precision=6, suppress=True)

    print("== A. correctness grid: tiled (float64) vs naive (float64) ==")
    worst_o = worst_l = 0.0
    n_cfg = 0
    for N in (1, 2, 5, 37, 64, 100, 257):
        for d in (1, 8, 33):
            Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
            for causal in (False, True):
                O_ref, L_ref, _ = naive_attention_np(Q, K, V, causal)
                for Br, Bc in ((1, 1), (4, 7), (16, 16), (64, 32), (128, 128), (1000, 1000)):
                    O, L = flash_attention_forward(Q, K, V, Br, Bc, causal)
                    worst_o = np.maximum(worst_o, np.abs(O - O_ref).max())   # np.maximum keeps NaN; built-in max() drops it
                    worst_l = np.maximum(worst_l, np.abs(L - L_ref).max())
                    n_cfg += 1
    print(f"  {n_cfg} configurations (ragged tiles, Br != Bc, tiles larger than N, causal on/off)")
    print(f"  worst |O - O_ref| = {worst_o:.2e}    worst |L - L_ref| = {worst_l:.2e}")

    print("\n== B. float32: tiled vs naive, both judged against float64 naive ==")
    N, d = 512, 64
    Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
    O64, L64, _ = naive_attention_np(Q, K, V, causal=True)
    Q32, K32, V32 = (a.astype(np.float32) for a in (Q, K, V))
    O32t, L32t = flash_attention_forward(Q32, K32, V32, 64, 64, causal=True)
    O32n, L32n, _ = naive_attention_np(Q32, K32, V32, causal=True)
    print(f"  tiled float32 vs float64 naive: max|dO| = {np.abs(O32t - O64).max():.2e}  max|dL| = {np.abs(L32t - L64).max():.2e}")
    print(f"  naive float32 vs float64 naive: max|dO| = {np.abs(O32n - O64).max():.2e}  max|dL| = {np.abs(L32n - L64).max():.2e}")

    print("\n== C. the saved logsumexp L lets us rebuild P without the max or the sum ==")
    N, d = 64, 16
    Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
    O, L = flash_attention_forward(Q, K, V, 16, 16, causal=True)
    _, _, S = naive_attention_np(Q, K, V, causal=True)
    P_from_L = np.exp(S - L[:, None])
    P_ref = np.exp(S - S.max(1, keepdims=True)); P_ref /= P_ref.sum(1, keepdims=True)
    print(f"  max |exp(S - L) - softmax(S)| = {np.abs(P_from_L - P_ref).max():.2e}")
    print(f"  max |row sums of exp(S - L) - 1| = {np.abs(P_from_L.sum(1) - 1).max():.2e}")

    print("\n== D. the rescale factor as printed in the FA-2 paper (an inverse) fails ==")
    Q, K, V = (rng.standard_normal((100, 16)) for _ in range(3))
    O_ref, _, _ = naive_attention_np(Q, K, V)
    O_ok, _ = flash_attention_forward(Q, K, V, 16, 16)
    O_bad, _ = flash_attention_forward(Q, K, V, 16, 16, _bad_rescale=True)
    print(f"  exp(m_old - m_new)      : max|dO| = {np.abs(O_ok - O_ref).max():.2e}")
    print(f"  exp(m_old - m_new)^-1   : max|dO| = {np.abs(O_bad - O_ref).max():.2e}")

    print("\n== E. multi-head layout [B, H, N, d] ==")
    B, H, N, d = 2, 3, 40, 8
    Q, K, V = (rng.standard_normal((B, H, N, d)) for _ in range(3))
    O, L = flash_attention_multihead(Q, K, V, Br=16, Bc=8, causal=True)
    err = np.max([np.abs(O[b, h] - naive_attention_np(Q[b, h], K[b, h], V[b, h], True)[0]).max()
                  for b in range(B) for h in range(H)])
    print(f"  max |O - reference| over {B * H} heads = {err:.2e}")
    print(f"  strides of a [B,H,N,d] float32 array in ELEMENTS: "
          f"{tuple(s // 4 for s in Q.astype(np.float32).strides)}  (formula: (H*N*d, N*d, d, 1) = {(H*N*d, N*d, d, 1)})")
    b, h, n, c = 1, 2, 17, 5
    off = ((b * H + h) * N + n) * d + c
    print(f"  Q[1,2,17,5] = {Q[b, h, n, c]:.6f}   Q.ravel()[((b*H+h)*N+n)*d+c] = {Q.ravel()[off]:.6f}")

    print("\n== E2. sliding-window attention: rows whose first tile is fully masked ==")
    N, d, w = 100, 16, 5
    Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
    O_ref, L_ref, _ = naive_attention_np(Q, K, V, causal=True, window=w)
    for Br, Bc in ((16, 16), (7, 9), (32, 8)):
        O, L = flash_attention_forward(Q, K, V, Br, Bc, causal=True, window=w)
        with np.errstate(all="ignore"):
            O_ng, _ = flash_attention_forward(Q, K, V, Br, Bc, causal=True, window=w, _no_guard=True)
        print(f"  window={w} Br={Br:>2} Bc={Bc:>2}: with guard max|dO| = {np.abs(O - O_ref).max():.2e};"
              f"  WITHOUT guard: {int(np.isnan(O_ng).any(axis=1).sum())} of {N} output rows are NaN")

    print("\n== F. causal: how many tiles are skipped? (N=1024) ==")
    for Br, Bc in ((64, 64), (128, 64), (32, 128)):
        kept, total = causal_tile_counts(1024, Br, Bc)
        print(f"  Br={Br:>3} Bc={Bc:>3}: processed {kept:>3} of {total:>3} tiles = {kept / total:5.1%}")

    print("\n== G. traffic: elements moved between slow and fast memory (N=1024, d=64) ==")
    N, d = 1024, 64
    alg0 = 4 * N * N + 4 * N * d
    floor = 4 * N * d
    print(f"  Chapter 1 numbers: Algorithm 0 = {alg0:,}   floor (Q,K,V in, O out) = {floor:,}")
    print(f"  square tiles Br = Bc = B:")
    print(f"  {'B':>5} {'T':>4} {'Q-outer':>10} {'KV-outer':>10} {'Alg0/Q-outer':>13} {'KV-outer/Q-outer':>17}")
    for B in (16, 32, 64, 128, 256, 1024):
        q = io_dry_run(N, d, B, B, "Q-outer"); k = io_dry_run(N, d, B, B, "KV-outer")
        print(f"  {B:>5} {math.ceil(N / B):>4} {q:>10,} {k:>10,} {alg0 / q:>13.2f} {k / q:>17.2f}")
    print("  Q-outer traffic at Br=64 for different Bc (it should not depend on Bc):")
    print("   ", {bc: io_dry_run(N, d, 64, bc, "Q-outer") for bc in (16, 64, 256, 1024)})
    io = dict.fromkeys(["Q read", "K read", "V read", "O written", "L written"], 0)
    flash_attention_forward(*(rng.standard_normal((N, d)).astype(np.float32) for _ in range(3)), Br=64, Bc=64, io=io)
    print(f"  instrumented real run (Br=Bc=64): {io}")
    print(f"     total = {sum(io.values()):,};  dry run = {io_dry_run(N, d, 64, 64, 'Q-outer'):,}")
    T = math.ceil(N / 64)
    print(f"     closed form 2Nd + 2Nd*T_r + N = {2 * N * d + 2 * N * d * T + N:,}")

    print("\n== H. tile footprint if everything sat in shared memory (upper bound; Chapter 9 refines) ==")
    print("   Q,K,V tiles fp16 (2 B); S, O-accumulator, m, l fp32 (4 B)")
    for Br, Bc, d in ((64, 64, 64), (128, 64, 64), (128, 128, 64), (128, 64, 128), (128, 128, 128)):
        b = 2 * (Br * d + 2 * Bc * d) + 4 * (Br * Bc + Br * d + 2 * Br)
        print(f"  Br={Br:>3} Bc={Bc:>3} d={d:>3}: {b:>7,} B = {b / 1024:6.1f} KiB")

    print("\n== I. peak Python-side allocation, beyond the inputs (numpy, float32, non-causal) ==")
    for N in (1024, 2048, 4096):
        d = 64
        Q, K, V = (rng.standard_normal((N, d)).astype(np.float32) for _ in range(3))
        tracemalloc.start(); naive_attention_np(Q, K, V); _, p_naive = tracemalloc.get_traced_memory(); tracemalloc.stop()
        tracemalloc.start(); flash_attention_forward(Q, K, V, 64, 64); _, p_tiled = tracemalloc.get_traced_memory(); tracemalloc.stop()
        print(f"  N={N:>5}: naive peak = {p_naive / 2**20:8.2f} MiB ({p_naive / (N * N * 4):4.1f} x one S)   "
              f"tiled peak = {p_tiled / 2**20:6.2f} MiB   (O + L alone = {(N * d + N) * 4 / 2**20:5.2f} MiB)")
