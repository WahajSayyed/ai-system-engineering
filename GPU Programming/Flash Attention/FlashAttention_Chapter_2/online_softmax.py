"""Chapter 2: from 3-pass softmax to a one-pass softmax-weighted sum.

Pure-Python loops on purpose: the loop structure is the point, and it is what we
translate to C++ (and later CUDA). numpy is used only to check answers.
"""
import math
import random

import numpy as np

NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# 1. Safe softmax: three passes over the input
# --------------------------------------------------------------------------- #
def softmax_3pass(x):
    m = NEG_INF
    for xi in x:                       # pass 1: maximum
        m = max(m, xi)
    d = 0.0
    for xi in x:                       # pass 2: normaliser
        d += math.exp(xi - m)
    return [math.exp(xi - m) / d for xi in x]   # pass 3: outputs


# --------------------------------------------------------------------------- #
# 2. Online normaliser: max and sum in ONE pass, then one pass for the outputs
# --------------------------------------------------------------------------- #
def push(m, d, x):
    """Fold one more score x into the running state (m, d).

    Invariant after the call:  m = max of everything seen so far,
                               d = sum over everything seen of exp(value - m).
    """
    m_new = max(m, x)
    if m_new == NEG_INF:               # everything seen so far is masked: nothing to accumulate
        return m, d
    d = d * math.exp(m - m_new) + math.exp(x - m_new)
    return m_new, d


def online_stats(x, trace=False):
    m, d = NEG_INF, 0.0
    for j, xj in enumerate(x):
        m_old = m
        m, d = push(m, d, xj)
        if trace:
            print(f"  j={j}  x={xj:5.2f}  m_old={m_old:5.2f}  m_new={m:5.2f}  "
                  f"rescale=exp(m_old-m_new)={math.exp(m_old - m):.4f}  d={d:.4f}")
    return m, d


def softmax_online(x):
    m, d = online_stats(x)
    if d == 0.0:                       # fully masked row: our convention is all zeros
        return [0.0] * len(x)
    return [math.exp(xi - m) / d for xi in x]


def softmax_online_unguarded(x):
    """The 'obvious' version, WITHOUT the masked-so-far guard. Kept to show the failure."""
    m, d = NEG_INF, 0.0
    with np.errstate(all="ignore"):
        for xi in x:
            m_new = max(m, xi)
            d = d * np.exp(m - m_new) + np.exp(xi - m_new)
            m = m_new
    return [float(np.exp(xi - m) / d) for xi in x]


# --------------------------------------------------------------------------- #
# 3. The merge operator: combine the states of two chunks of a row
# --------------------------------------------------------------------------- #
def merge(a, b):
    (ma, da), (mb, db) = a, b
    m = max(ma, mb)
    if m == NEG_INF:
        return (NEG_INF, 0.0)
    return (m, da * math.exp(ma - m) + db * math.exp(mb - m))


def stats_blocked(x, block):
    total = (NEG_INF, 0.0)
    for start in range(0, len(x), block):
        m, d = NEG_INF, 0.0
        for xi in x[start:start + block]:
            m, d = push(m, d, xi)
        total = merge(total, (m, d))
    return total


# --------------------------------------------------------------------------- #
# 4. One pass for what attention actually needs: sum_j softmax(s)_j * v_j
#    v_j may be a vector (numpy array); the same rescale applies to every component.
# --------------------------------------------------------------------------- #
def weighted_sum_online(s, V):
    m, d = NEG_INF, 0.0
    o = np.zeros_like(V[0], dtype=np.float64)
    for sj, vj in zip(s, V):
        m_new = max(m, sj)
        if m_new == NEG_INF:
            continue
        scale = math.exp(m - m_new)     # shrink everything accumulated so far
        w = math.exp(sj - m_new)        # weight of the new element
        d = d * scale + w
        o = o * scale + w * vj
        m = m_new
    return o / d if d > 0.0 else np.zeros_like(o)


def weighted_sum_reference(s, V):
    p = softmax_3pass(s)
    return sum(pj * vj for pj, vj in zip(p, V))


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    rng = random.Random(0)

    print("== A. trace of the online normaliser on x = [1, 3, 2, 4] ==")
    m, d = online_stats([1.0, 3.0, 2.0, 4.0], trace=True)
    direct = sum(math.exp(v - 4.0) for v in [1.0, 3.0, 2.0, 4.0])
    print(f"  final (m, d) = ({m}, {d:.6f});  direct sum exp(x - 4) = {direct:.6f}")

    print("\n== B. online vs 3-pass vs numpy on awkward inputs (max abs diff) ==")
    N = 257
    families = {
        "random normal":        [rng.gauss(0, 3) for _ in range(N)],
        "offset +1000":         [1000 + rng.gauss(0, 3) for _ in range(N)],
        "increasing":           [float(i) * 0.1 for i in range(N)],
        "decreasing":           [-float(i) * 0.1 for i in range(N)],
        "constant":             [7.0] * N,
        "single element":       [3.5],
        "-inf in the middle":   [rng.gauss(0, 1) if i % 5 else NEG_INF for i in range(N)],
        "-inf at the START":    [NEG_INF] * 3 + [rng.gauss(0, 1) for _ in range(N - 3)],
    }
    for name, x in families.items():
        ref = np.exp(np.array(x) - np.max(x)); ref = ref / ref.sum()
        a, b = np.array(softmax_3pass(x)), np.array(softmax_online(x))
        print(f"  {name:<20} |3pass-ref|={np.abs(a - ref).max():.2e}  "
              f"|online-ref|={np.abs(b - ref).max():.2e}  sum(online)={b.sum():.6f}")

    print("\n== C. the guard: a row that starts with -inf ==")
    x = [NEG_INF, NEG_INF, 1.0, 2.0, 0.5]
    print("  unguarded:", np.array(softmax_online_unguarded(x)))
    print("  guarded  :", np.array(softmax_online(x)))
    print("  reference:", np.array(softmax_3pass(x[2:])), "(the last three entries)")
    print("  all -inf row, guarded:", softmax_online([NEG_INF] * 3), "(our convention: zeros)")

    print("\n== D. merging chunks: block size and merge order do not matter ==")
    x = [rng.gauss(0, 4) for _ in range(1000)]
    m0, d0 = online_stats(x)
    for block in (1, 2, 3, 7, 64, 500, 1000):
        m, d = stats_blocked(x, block)
        print(f"  block={block:>4}  m={m:.6f}  d={d:.10f}  |d - d_sequential|/d = {abs(d - d0) / d0:.2e}")
    # random binary-tree merge order
    parts = []
    for start in range(0, len(x), 10):
        m, d = NEG_INF, 0.0
        for xi in x[start:start + 10]:
            m, d = push(m, d, xi)
        parts.append((m, d))
    worst = 0.0
    for _ in range(200):
        pool = parts[:]
        rng.shuffle(pool)
        while len(pool) > 1:
            i, j = rng.sample(range(len(pool)), 2)
            a, b = pool[i], pool[j]
            pool = [p for k, p in enumerate(pool) if k not in (i, j)] + [merge(a, b)]
        worst = max(worst, abs(pool[0][1] - d0) / d0)
    print(f"  200 random merge trees over 100 chunks: worst relative error in d = {worst:.2e}")

    print("\n== E. how often does the maximum change? (rescale is only needed then) ==")
    for n in (16, 256, 4096):
        trials, changes = 200, 0
        for _ in range(trials):
            m = NEG_INF
            for _ in range(n):
                v = rng.gauss(0, 1)
                if v > m:
                    changes += 1
                    m = v
        harmonic = sum(1.0 / k for k in range(1, n + 1))
        print(f"  N={n:>5}: mean max-updates per row = {changes / trials:5.2f}   (harmonic number H_N = {harmonic:5.2f})")
    inc = [float(i) for i in range(256)]
    changes = sum(1 for i in range(1, 256) if inc[i] > max(inc[:i])) + 1
    print(f"  increasing input, N=256: max changes on {changes} of 256 elements (worst case)")

    print("\n== F. one pass for the weighted sum, vector values (N=64, d=8) ==")
    worst = 0.0
    for trial in range(200):
        s = [rng.gauss(0, 3) for _ in range(64)]
        if trial % 4 == 0:
            s[:5] = [NEG_INF] * 5          # masked start
        V = [np.array([rng.gauss(0, 1) for _ in range(8)]) for _ in range(64)]
        worst = max(worst, np.abs(weighted_sum_online(s, V) - weighted_sum_reference(s, V)).max())
    print(f"  worst |one-pass - reference| over 200 rows: {worst:.2e}")
