# Chapter 3: Tiled Attention on the CPU

**Series:** Flash Attention from Scratch in CUDA · Part 0 (the algorithm before the kernel)
**Languages:** NumPy, then C++ (pointers, flat row-major indexing, header/source files, how compilation works)
**Builds on:** Chapter 1 (the `N × N` problem) and Chapter 2 (the online-softmax state `(m, d)` and how to merge it)

---

## What you will be able to do after this chapter

1. Split attention into tiles and compute the exact answer without ever building the `N × N` score matrix.
2. Write the FlashAttention-2 forward loop (Q tiles outside, K/V tiles inside) in NumPy and in C++.
3. Save the per-row logsumexp `L` that the backward pass will need, and explain why it is enough.
4. Handle causal masking at tile level: skip tiles, mask only the tiles that touch the diagonal.
5. Predict the memory traffic of a tile size, and see that **small tiles can move more data than not tiling at all**.
6. Read and write C++ that uses **pointers** and **flat row-major indexing** (`Q[i * d + x]`), which is what every CUDA kernel uses.
7. Explain the stages between a `.cpp` file and a running program (preprocess, compile, assemble, link), and read a linker error.
8. Test numerical code so that the tests can actually fail (§3.10).

## How this chapter is grounded

| Label | Meaning |
|---|---|
| **[ran]** | I executed it (Python 3 with NumPy, GCC 13.3) and pasted the real output. |
| **[source]** | From a named source (§3.17). Both FlashAttention papers were read directly for this chapter. |
| **[derived]** | Computed by a formula shown in the text. |
| **[predicted]** | My reasoning about something I did not measure. |
| **[language rule]** | A C++ or NumPy rule I state but did not demonstrate. |

Nothing here needs a GPU or PyTorch. Every file is beside this chapter, in `ch03_code/`.

---

## 3.1 The picture: cut the score matrix into tiles

Chapter 1 showed why we must never store the `N × N` matrix `S`. Chapter 2 showed we can process one row of `S` in pieces. Now do that for all rows at once, in blocks.

Take one head, `Q, K, V` of shape `[N, d]`. Choose two tile sizes: `Br` (rows of `Q` per tile) and `Bc` (rows of `K`/`V` per tile). Cut `Q` into `T_r = ⌈N/Br⌉` row tiles, and `K`, `V` into `T_c = ⌈N/Bc⌉` row tiles. The score matrix `S` then breaks into a grid of small `Br × Bc` tiles:

```
                  K/V tile:   j=0        j=1        j=2        j=3
                          ┌──────────┬──────────┬──────────┬──────────┐
 Q tile i=0 (rows 0..Br-1)│  S(0,0)  │  S(0,1)  │  S(0,2)  │  S(0,3)  │
                          ├──────────┼──────────┼──────────┼──────────┤
 Q tile i=1               │  S(1,0)  │  S(1,1)  │  S(1,2)  │  S(1,3)  │
                          ├──────────┼──────────┼──────────┼──────────┤
 Q tile i=2               │  S(2,0)  │  S(2,1)  │  S(2,2)  │  S(2,3)  │
                          └──────────┴──────────┴──────────┴──────────┘
```

We visit the grid **one Q tile at a time**: for a fixed Q tile `i`, walk left to right across `j = 0, 1, 2, …`. At each stop we compute the small tile `S(i, j)`, use it, and throw it away. The full grid never exists.

This loop order (Q tiles outside, K/V tiles inside) is the one FlashAttention-2's forward pass uses **[source]**. The original FlashAttention did it the other way round; §3.3 compares them.

---

## 3.2 One tile step

For each row of the current Q tile we keep three things, all small:

```
m    running max of the scores seen so far            (one number per row)
l    running sum of exp(score - m)                    (one number per row)
acc  running UNNORMALISED output  Σ exp(score - m) · v   (a row of length d)
```

They start at `m = −inf`, `l = 0`, `acc = 0`. For each K/V tile `j` (this is Chapter 2's update, done for a whole tile at once):

```
1.  S      = scale · Q_i · K_jᵀ                       [br × bc]   the score tile
2.  m_new  = max(m, rowmax(S))                         [br]
3.  α      = exp(m − m_new)                            [br]        rescale factor for the OLD state
    P      = exp(S − m_new)                            [br × bc]   unnormalised probabilities
4.  l      = α · l + rowsum(P)
5.  acc    = α · acc + P · V_j                         [br × d]
6.  m      = m_new
```

After the last tile:

```
O_i = acc / l                        the attention output for this Q tile (divide once)
L_i = m + log(l)                     the logsumexp of each row (saved for the backward pass)
```

This is FlashAttention-2's Algorithm 1 **[source]**. The two changes it makes to the original algorithm are exactly the two you see here: the output is kept **unscaled** and divided by `l` only once at the very end, and the only statistic saved for the backward pass is the logsumexp `L = m + log(l)` rather than both `m` and `l` **[source]**. Extra memory beyond the inputs and output is `O(N)`, for `L` **[source]**.

**How this relates to Chapter 2.** `(m, l)` is Chapter 2's `(m, d)` state. Step 4 is Chapter 2's update. Step 5 is Chapter 2's weighted-sum accumulator, with a vector `v` instead of a number. Computing `m_new` against the tile's row maximum and then `exp(S − m_new)` gives the same result as forming the tile's own `(m, d)` state and calling Chapter 2's `merge`; the algorithm just skips writing the tile's state down.

**A trap in the paper.** As I read the FlashAttention-2 paper, its Algorithm 1 (line 10) and its §3.1.1 write the rescale of the old output as `diag(e^{m_old − m_new})⁻¹ · O_old`. Taken literally, the inverse multiplies by `e^{m_new − m_old}`, which is at least 1, so old output would *grow* when the max increases. That contradicts the derivation in Chapter 2, where the factor must be `e^{m_old − m_new} ≤ 1`. **[ran]** I implemented both:

```
  exp(m_old - m_new)      : max|dO| = 3.33e-16
  exp(m_old - m_new)^-1   : max|dO| = 8.93e-01
```

Only the version without the inverse is correct, so I treat the printed `⁻¹` as a typographical slip in the paper. I did not check whether a later revision fixes it. The lesson is worth keeping: when you implement from a paper, test the algorithm; do not trust a single line.

---

## 3.3 NumPy: the FlashAttention-2 loop, line by line

This is the core of `tiled_attention.py`, with the optional debugging arguments removed (`core_listing.py` is exactly this text):

```python
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
```

**Reading it.**

- `range(0, N, Br)` steps by `Br`, so `r0` is the first row of each Q tile. `min(r0 + Br, N)` clips the last tile when `N` is not a multiple of `Br` ("ragged" tile). So `br` may be less than `Br`.
- `Q[r0:r1]` is a **view**: NumPy does not copy it **[language rule]**. It is a window onto the same memory, which is exactly what a C++ pointer to a row will be in §3.7.
- `S.max(axis=1)` is the maximum of each row of the tile; `np.maximum(m, ...)` compares elementwise with the running max.
- **The guard.** Where `m_new` is still `−inf`, every score so far in that row is masked. `exp(−inf − (−inf))` would be `NaN`. Replacing `−inf` by `0` in `m_safe` avoids it: then `alpha = exp(−inf − 0) = 0` and `P = exp(−inf − 0) = 0`, so a fully masked row contributes nothing, and `m` itself stays `−inf`. This is Chapter 2's guard in vectorised form.
- `Kj.T` is a view too (a transposed window, no copy).
- **Causal masking at tile level** follows the FlashAttention-2 paper's two rules **[source]**: (1) a tile whose every column index is to the right of every row index is fully masked, so skip it; since tiles are visited left to right, `break`. (2) Only tiles that touch the diagonal need the element-wise mask. Tiles fully below the diagonal need none.
- The only arrays that depend on `N` are the inputs, `O` and `L`. Everything else is tile-sized.

### Does it work?

**[ran]** 252 configurations: `N ∈ {1, 2, 5, 37, 64, 100, 257}`, `d ∈ {1, 8, 33}`, six `(Br, Bc)` pairs including `(1, 1)`, `Br ≠ Bc`, and tiles larger than `N`, causal on and off. float64, against the full-matrix reference from Chapter 1:

```
  252 configurations (ragged tiles, Br != Bc, tiles larger than N, causal on/off)
  worst |O - O_ref| = 1.55e-15    worst |L - L_ref| = 1.78e-15
```

**[ran]** float32: how much error does tiling add over the naive float32 version? Both compared with the float64 answer, `N = 512, d = 64`, causal:

```
  tiled float32 vs float64 naive: max|dO| = 4.07e-07  max|dL| = 4.85e-07
  naive float32 vs float64 naive: max|dO| = 3.82e-07  max|dL| = 4.91e-07
```

Tiling did not make float32 noticeably less accurate.

### The saved `L` is enough to rebuild `P`

The backward pass needs the probabilities `P`, but we never stored them. With `L` we can recompute any entry from the scores: `P = exp(S − L)`. **[ran]** (`N = 64, d = 16`, causal):

```
  max |exp(S - L) - softmax(S)| = 3.33e-16
  max |row sums of exp(S - L) - 1| = 6.66e-16
```

No max and no sum needed: `exp(S − L) = exp(S − m) / l` is already normalised. This is why one number per row is all the backward pass has to save (Chapters 22 and 23).

### Many heads: the same code, `B × H` times

`[B, H, N, d]` inputs are `B·H` independent problems. `flash_attention_multihead` in the file just loops over `(b, h)`. **[ran]** for `B=2, H=3, N=40, d=8` causal, `Br=16, Bc=8`:

```
  max |O - reference| over 6 heads = 5.55e-16
```

Nothing is shared between heads. In CUDA that independence becomes parallelism: each `(b, h)`, and in FlashAttention-2 each Q tile as well, can be its own thread block **[source: FlashAttention-2 parallelises along the sequence dimension as well as batch and heads]**.

### A `[B, H, N, d]` array is one flat array

Element `(b, h, n, c)` lives at flat index `((b·H + h)·N + n)·d + c`. **[ran]**:

```
  strides of a [B,H,N,d] float32 array in ELEMENTS: (960, 320, 8, 1)  (formula: (H*N*d, N*d, d, 1) = (960, 320, 8, 1))
  Q[1,2,17,5] = 0.893168   Q.ravel()[((b*H+h)*N+n)*d+c] = 0.893168
```

NumPy's `strides` (in bytes, divided here by 4) are exactly the multipliers in that formula. In C++ you write the formula yourself, so learn it now.

---

## 3.4 Causal masking: how many tiles disappear

**[ran]** `N = 1024`, counting tiles that survive the "entirely right of the diagonal" test:

```
  Br= 64 Bc= 64: processed 136 of 256 tiles = 53.1%
  Br=128 Bc= 64: processed  72 of 128 tiles = 56.2%
  Br= 32 Bc=128: processed 144 of 256 tiles = 56.2%
```

About half, as the FlashAttention-2 paper says (approximately half of the blocks for large sequences); the paper reports around 1.7 to 1.8 times speed-up relative to attention without the causal mask, measured on a GPU **[source]**. The surplus over 50% here is the diagonal tiles, which must still be computed.

### Where the guard is really needed: sliding windows

Plain causal self-attention never has a row whose first processed tile is fully masked (each row can always see its own diagonal element, and tile 0 is always visited). So the guard is never exercised. Other masks do exercise it. In **sliding-window** attention, query `i` may only see keys `i−w+1 … i`; then early tiles can be entirely outside the window for some rows.

**[ran]** `N = 100, d = 16, w = 5`, with and without the guard (`_no_guard=True`):

```
  window=5 Br=16 Bc=16: with guard max|dO| = 4.44e-16;  WITHOUT guard: 60 of 100 output rows are NaN
  window=5 Br= 7 Bc= 9: with guard max|dO| = 6.66e-16;  WITHOUT guard: 32 of 100 output rows are NaN
  window=5 Br=32 Bc= 8: with guard max|dO| = 4.44e-16;  WITHOUT guard: 76 of 100 output rows are NaN
```

Keep this in mind for §3.10: a guard that is never tested can be silently broken. Sliding windows return in Chapter 25.

---

## 3.5 What tiling costs in memory traffic

Chapter 1 counted the traffic of standard attention: `4N² + 4Nd` elements for the paper's Algorithm 0, and a floor of `4Nd` (read `Q, K, V`, write `O`). Now count it for the tiled loop. The model: "slow memory" is the full arrays; "fast memory" is the tile-sized locals. Count the elements crossing between them.

```
Q-outer (FlashAttention-2 order):
    read Q once                                   N·d
    read K and V once for EVERY Q tile            T_r · 2·N·d
    write O once, write L once                    N·d + N
    total = 2·N·d + 2·N·d·T_r + N

K/V-outer (the original FlashAttention order, Algorithm 1 of the first paper):
    read K, V once                                2·N·d
    for each of the T_c outer steps, over all Q tiles:
        read Q and O                              2·N·d
        write O                                   N·d
        read and write l and m                    4·N
    total = 2·N·d + T_c · (3·N·d + 4·N)
```

The second order is from the first paper's Algorithm 1, which loads and stores the partial output and the statistics at every inner step **[source]**.

I did not just trust these formulas. **[ran]** the real NumPy loop with counters inside it (`io=` argument), `N = 1024, d = 64, Br = Bc = 64`, float32:

```
  instrumented real run (Br=Bc=64): {'Q read': 65536, 'K read': 1048576, 'V read': 1048576, 'O written': 65536, 'L written': 1024}
     total = 2,229,248;  dry run = 2,229,248
     closed form 2Nd + 2Nd*T_r + N = 2,229,248
```

Now the numbers **[ran]** (`N = 1024, d = 64`, square tiles `Br = Bc = B`; from Chapter 1, Algorithm 0 = 4,456,448 elements and the floor is 262,144):

```
      B    T    Q-outer   KV-outer  Alg0/Q-outer  KV-outer/Q-outer
     16   64  8,520,704 12,976,128          0.52              1.52
     32   32  4,326,400  6,553,600          1.03              1.51
     64   16  2,229,248  3,342,336          2.00              1.50
    128    8  1,180,672  1,736,704          3.77              1.47
    256    4    656,384    933,888          6.79              1.42
   1024    1    263,168    331,776         16.93              1.26
```

Read this table carefully:

- **Tiling is not automatically a win.** With `16 × 16` tiles the FlashAttention-2 order moves almost **twice as much** data as standard attention (ratio 0.52). The `N × N` matrix is gone, but `K` and `V` are re-read `T_r = 64` times.
- **Traffic is about `2N²d / Br`.** It falls in proportion to the tile height. At `Br = 64` that gives `2·1024²·64/64 = 2.1M` (actual: 2.23M including the `2Nd + N` terms). Bigger tiles mean fewer passes over `K` and `V`.
- **`Bc` does not matter for traffic in this order.** **[ran]** at `Br = 64`, `Bc ∈ {16, 64, 256, 1024}` all give 2,229,248. `Bc` affects the on-chip footprint and loop overhead, not the bytes moved. (In the other order the roles swap.)
- **The FlashAttention-2 order moves about 1.3 to 1.5 times less than the original order** at these sizes, because it writes `O` once instead of reading and writing it at every step.
- **You need big tiles to approach the floor.** At `B = 1024` (one tile covers everything) the traffic is 263,168, essentially the floor plus `N` for `L`. At `B = 64` you are only 2× better than standard attention, nowhere near the 17× gap. The paper's analysis says the same thing in general terms: `Θ(N²d²/M)` accesses, where `M` is the on-chip memory size, so a bigger on-chip memory allows bigger tiles and fewer accesses **[source]**.

### What that means for your GPUs (derived from Chapter 1's numbers)

Take Chapter 1's workload (`B·H = 1024` heads, `N = 1024, d = 64`, fp16, forward only) and convert elements to bytes (× 2 × 1024 heads). Lower-bound memory times at the datasheet bandwidths Chapter 1 used (936 GB/s for the RTX 3090, 300 GB/s for the T4) **[derived]**:

```
Algorithm 0    :  9.13 GB | 3090  9.75 ms | T4 30.42 ms
tiled  B=64    :  4.57 GB | 3090  4.88 ms | T4 15.22 ms
tiled  B=128   :  2.42 GB | 3090  2.58 ms | T4  8.06 ms
tiled  B=256   :  1.34 GB | 3090  1.44 ms | T4  4.48 ms
compute at Tensor Core peak (Chapter 1): 3090 3.87 ms | T4 4.23 ms
```

For `B = 128`, memory time on the 3090 (2.58 ms) drops below the Tensor Core compute time (3.87 ms): the kernel would become compute-bound. On the T4 the memory time (8.06 ms) still exceeds compute time (4.23 ms). These are lower bounds under a model that ignores caches (some repeated `K`/`V` reads may hit the L2 cache); how much that helps is **[predicted]**, not measured. We measure real kernels from Chapter 4 on.

### How big can a tile be? A first look at the on-chip budget

Each tile step needs, on chip: a `Q` tile (`Br·d`), a `K` and a `V` tile (`2·Bc·d`), the score tile (`Br·Bc`), the output accumulator (`Br·d`), and `m`, `l` (`2·Br`). **[ran]** (`tiled_attention.py`, `Q, K, V` tiles in fp16 = 2 bytes; scores, accumulator, `m`, `l` in fp32 = 4 bytes; **an upper bound** that assumes everything sits in shared memory, whereas real kernels keep much of it in registers):

```
  Br= 64 Bc= 64 d= 64:  57,856 B =   56.5 KiB
  Br=128 Bc= 64 d= 64:  99,328 B =   97.0 KiB
  Br=128 Bc=128 d= 64: 148,480 B =  145.0 KiB
  Br=128 Bc= 64 d=128: 164,864 B =  161.0 KiB
  Br=128 Bc=128 d=128: 230,400 B =  225.0 KiB
```

Chapter 1 quoted, from secondary sources, on-chip limits per thread block of 64 KB for the T4 and 99 KB for the RTX 3090 (with an opt-in above 48 KB). Against those, `128 × 64` tiles at `d = 64` (97 KiB) would just fit on the 3090 and not on the T4, which is consistent with the FlashAttention paper's observation that the T4's smaller on-chip memory forces smaller blocks and a smaller speed-up **[source]**. Treat this as a preview: Chapter 9 does the real budget, and will confirm the limits with a device query.

### The memory saving, measured on the CPU

**[ran]** NumPy peak allocations beyond the inputs, float32, non-causal, measured with `tracemalloc`. This measures NumPy on the CPU, **not** PyTorch or a GPU:

```
  N= 1024: naive peak =    12.26 MiB ( 3.1 x one S)   tiled peak =   0.35 MiB   (O + L alone =  0.25 MiB)
  N= 2048: naive peak =    48.52 MiB ( 3.0 x one S)   tiled peak =   0.60 MiB   (O + L alone =  0.51 MiB)
  N= 4096: naive peak =   193.03 MiB ( 3.0 x one S)   tiled peak =   1.11 MiB   (O + L alone =  1.02 MiB)
```

The naive version quadruples every time `N` doubles; the tiled version doubles (it is dominated by `O` and `L`, which are `O(N)`). NumPy's eager evaluation keeps about three `N × N` temporaries alive, more than the two I predicted in Chapter 1 for PyTorch; the pattern differs between libraries, the scaling does not.

---

## 3.6 C++ part 2: pointers and flat row-major indexing

NumPy carries the shape with the array (`Q.shape`, `Q[i, j]`). A C++ pointer does not. Everything the CUDA kernels do is built on the ideas in this section.

### A matrix is one flat array

A `[rows × cols]` matrix stored **row-major** is one line of numbers, row after row. Element `(r, c)` is at index `r * cols + c`. **[ran]** (`pointers_demo.cpp`, a `2 × 3` matrix):

```cpp
    std::vector<float> M{0, 1, 2, 3, 4, 5};             //  [[0 1 2],
                                                         //   [3 4 5]]
    const float* p = M.data();
    printf("M[%zu][%zu] = p[%zu * %zu + %zu] = %g\n", r, c, r, cols, c, p[r * cols + c]);
```

```
M[1][2] = p[1 * 3 + 2] = 5
```

### What a pointer is

A **pointer** holds a memory address. `float*` is "address of a float". Key operations:

| C++ | Meaning | NumPy analogue |
|---|---|---|
| `const float* p = M.data();` | address of the first element | (the array itself) |
| `p[3]` or `*(p + 3)` | the element 3 positions along | `M.ravel()[3]` |
| `p + k` | address `k` **elements** further on | `M.ravel()[k:]` (a view, no copy) |
| `const float* row = p + i * cols;` | pointer to the **start of row `i`** | `M[i]` |
| `row[j]` | element `j` of that row | `M[i][j]` |
| `nullptr` | a pointer that points at nothing | `None` |
| `&x[k]` | address of element `k` of `x` | |

**[ran]**:

```
sizeof(float) = 4 bytes
p+1 is 4 bytes after p
*p = 0,  p[3] = 3,  *(p + 3) = 3
row1[0..2] = 3 4 5
row1 == &M[3]? yes
nothing == nullptr? yes
```

`p + 1` is 4 bytes after `p`: **pointer arithmetic counts in elements, not bytes**, and the compiler multiplies by the element size **[language rule]**. So `Q + i * d` is the row pointer for row `i` of an `[N × d]` matrix. It is exactly what NumPy's `Q[i]` view gives you, but with no size information attached: you must keep `N` and `d` yourself.

| NumPy | C++ |
|---|---|
| `Q[i]` | `const float* q = Q + i * d;` |
| `Q[i, x]` | `Q[i * d + x]`, or `q[x]` |
| `Q[r0:r1]` | `Q + r0 * d`, and remember the row count `br` |
| `Q.shape` | separate `N` and `d` parameters |
| `np.empty((N, d))` | `std::vector<float> O(N * d)` |

`std::vector<float>::data()` returns the pointer to the vector's storage. `&S[r * Bc]` (the address of element `r * Bc`) is the same idea: the pointer to the start of row `r` of the tile stored in `S`. You will see both in the code.

### `const` and pointers: read the type right to left

- `const float* x` means "`x` points to floats that I promise not to modify". Writing through it is a compile error **[ran]**:

```
e_const.cpp: In function 'void zero_first(const float*)':
e_const.cpp:2:10: error: assignment of read-only location '* x'
    2 |     x[0] = 0.0f;
```

- `float* x` means the floats may be modified. Our inputs `Q`, `K`, `V` are `const float*`; the outputs `O`, `L` are `float*`. The compiler now checks that the attention code never writes to its inputs.
- `float* const x` would mean "`x` always points at the same place" (the pointer is constant, not the data). We do not use it in this chapter **[language rule]**.

### Pointers versus references

Chapter 2 introduced references (`const std::vector<float>&`). A reference is an alias that must refer to something and can never be re-pointed; a pointer can be `nullptr`, can be moved (`p + 1`), and can point into the middle of an array. CUDA kernels take **pointers** as arguments (Chapter 4) **[preview]**, which is why this chapter uses them.

### Getting the stride wrong fails silently

Read `M[1][2]` with the wrong multiplier and nothing complains **[ran]**:

```
M[1][2] with the right stride (cols): 5
M[1][2] with the wrong stride (rows): 4
```

The compiler cannot know which multiplier is right. Only tests can (§3.10).

### A pointer can outlive its storage

`data()` gives a pointer into the vector's current storage. If the vector grows and reallocates, the old pointer dangles **[ran]** (`dangling.cpp`, no sanitizer):

```
before: p[0] = 1
did the storage move? yes
after : p[0] = 5.73222e+20   <- reading freed memory
```

No error; just a garbage number. With `-fsanitize=address` **[ran]**:

```
==357==ERROR: AddressSanitizer: heap-use-after-free on address 0x502000000010 ...
READ of size 4 at 0x502000000010 thread T0
    #0 ... in main .../dangling.cpp:9
freed by thread T0 here:
    #7 ... in main .../dangling.cpp:7
```

Rule: take pointers *after* the storage has stopped changing size. (On a GPU, freeing device memory while a kernel still uses it is the same bug, and it is about as quiet.)

### 32-bit indices overflow

`i * d + x` is computed in the type of its operands. We use `size_t` (64-bit) throughout. With 32-bit `int` it stops working sooner than you might think: a tensor with `B = 8, H = 32, N = 65,536, d = 128` has `8 × 32 × 65,536 × 128 = 2,147,483,648 = 2³¹` elements **[derived]**, one more than the largest signed 32-bit integer (`2³¹ − 1`). Offsets into big tensors must be 64-bit.

---

## 3.7 The C++ program, piece by piece

The program is split into three files:

| File | What it holds |
|---|---|
| `tiled_attention.h` | **Declarations**: the function signatures. Included by anyone who wants to call them. |
| `tiled_attention.cpp` | **Definitions**: the code. Compiled once. |
| `main_test.cpp` | `main`: the tests. Includes the header, calls the functions. |

### The header

```cpp
#pragma once          // include this header at most once per source file
#include <cstddef>    // size_t

void attention_naive(const float* Q, const float* K, const float* V,
                     float* O, float* L,
                     size_t N, size_t d, bool causal);

void attention_tiled(const float* Q, const float* K, const float* V,
                     float* O, float* L,
                     size_t N, size_t d, size_t Br, size_t Bc, bool causal);
```

**C++ decoded:**

- A **declaration** says "a function with this name and these types exists somewhere". It has no body. A **definition** is the function with its body. Every function may be declared many times but must be defined exactly once.
- **`#pragma once`** stops the header being pasted in twice if two included files both include it.
- **`bool`** is a true/false type (`true`/`false`, lowercase).
- A function that needs `N` and `d` gets them as parameters, because a `const float*` carries no shape.

### The reference implementation

```cpp
void attention_naive(const float* Q, const float* K, const float* V,
                     float* O, float* L,
                     size_t N, size_t d, bool causal) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> S(N * N);                        // <- the matrix FlashAttention avoids

    for (size_t i = 0; i < N; ++i) {
        const float* q = Q + i * d;                     // pointer to row i of Q
        for (size_t j = 0; j < N; ++j) {
            const float* k = K + j * d;                 // pointer to row j of K
            float dot = 0.0f;
            for (size_t x = 0; x < d; ++x) {
                dot += q[x] * k[x];
            }
            S[i * N + j] = (causal && j > i) ? -INFINITY : dot * scale;
        }
    }

    for (size_t i = 0; i < N; ++i) {
        float* s = &S[i * N];                           // pointer to row i of S
        float m = -INFINITY;
        for (size_t j = 0; j < N; ++j) m = std::max(m, s[j]);
        float l = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            s[j] = expf(s[j] - m);
            l += s[j];
        }
        float* o = O + i * d;
        std::fill(o, o + d, 0.0f);
        for (size_t j = 0; j < N; ++j) {
            const float* v = V + j * d;
            const float p = s[j] / l;
            for (size_t x = 0; x < d; ++x) o[x] += p * v[x];
        }
        L[i] = m + logf(l);
    }
}
```

**C++ decoded:**

- Each row of `Q`, `K`, `V`, `O` is reached by a **row pointer** (`Q + i * d`). The three-loop dot product is `Q @ K.T` written out.
- `S[i * N + j]` is row-major indexing of the `N × N` matrix.
- **`std::sqrt(static_cast<float>(d))`**: `d` is an integer type, so convert it to `float` first, or you would get a `double` back. `std::sqrt` has a `float` version, so the result stays `float`.
- **`(cond) ? a : b`** is Python's `a if cond else b`.
- **`std::fill(o, o + d, 0.0f)`** sets the range `[o, o + d)` to zero. The two arguments are pointers ("from here, up to but not including here"). `std::fill(v.begin(), v.end(), x)` on a vector works the same way; `begin()` and `end()` are what the standard library calls **iterators**, which behave like pointers.
- `logf` is the `float` natural log; `L[i] = m + logf(l)` is the logsumexp.

### The tiled implementation: setup and loops

```cpp
void attention_tiled(const float* Q, const float* K, const float* V,
                     float* O, float* L,
                     size_t N, size_t d, size_t Br, size_t Bc, bool causal) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    std::vector<float> S(Br * Bc);      // score tile; row r starts at S[r * Bc]
    std::vector<float> m(Br);           // running max, one per row of the Q tile
    std::vector<float> l(Br);           // running sum
    std::vector<float> acc(Br * d);     // running UNNORMALISED output; row r starts at acc[r * d]

    for (size_t r0 = 0; r0 < N; r0 += Br) {              // ---- loop over Q tiles
        const size_t br = std::min(Br, N - r0);          // rows in this tile (last one may be short)
        std::fill(m.begin(), m.begin() + br, -INFINITY);
        std::fill(l.begin(), l.begin() + br, 0.0f);
        std::fill(acc.begin(), acc.begin() + br * d, 0.0f);

        for (size_t c0 = 0; c0 < N; c0 += Bc) {          // ---- loop over K/V tiles
            const size_t bc = std::min(Bc, N - c0);      // columns in this tile
            if (causal && c0 > r0 + br - 1) {
                break;                                   // tile lies entirely right of the diagonal
            }
```

**C++ decoded:**

- These four `std::vector`s are the **entire** working set: `Br·Bc + 2Br + Br·d` floats. Nothing is `N × N`. In CUDA these become registers and shared memory.
- **The tile buffer has a fixed row length `Bc`**: row `r` starts at `S[r * Bc]`, even when the last tile has fewer columns (`bc < Bc`). This is how a kernel's tile buffer works too, and forgetting the difference between `Bc` (the buffer's row length) and `bc` (the columns actually in use) is a real bug. We test for exactly that in §3.10.
- **`N - r0`** is safe here only because `r0 < N` inside the loop. With unsigned types, `a - b` when `b > a` wraps to a huge number (Chapter 2). Any unsigned subtraction deserves a second look.
- `m.begin() + br` is "the position `br` elements into the vector".

### The scores and the online update

```cpp
            // S = scale * Q_i K_j^T   (br x bc)
            for (size_t r = 0; r < br; ++r) {
                const float* q = Q + (r0 + r) * d;
                for (size_t c = 0; c < bc; ++c) {
                    const float* k = K + (c0 + c) * d;
                    float dot = 0.0f;
                    for (size_t x = 0; x < d; ++x) {
                        dot += q[x] * k[x];
                    }
                    const bool masked = causal && (c0 + c) > (r0 + r);
                    S[r * Bc + c] = masked ? -INFINITY : dot * scale;
                }
            }

            // Online-softmax update, one row of the tile at a time.
            for (size_t r = 0; r < br; ++r) {
                float* s = &S[r * Bc];                   // pointer to this row of the tile
                float tile_max = -INFINITY;
                for (size_t c = 0; c < bc; ++c) tile_max = std::max(tile_max, s[c]);

                const float m_new = std::max(m[r], tile_max);
                const float m_safe = (m_new == -INFINITY) ? 0.0f : m_new;   // Chapter 2 guard
                const float alpha = expf(m[r] - m_safe);                    // exp(m_old - m_new)

                float row_sum = 0.0f;
                for (size_t c = 0; c < bc; ++c) {
                    s[c] = expf(s[c] - m_safe);          // overwrite scores with unnormalised probabilities
                    row_sum += s[c];
                }
                l[r] = alpha * l[r] + row_sum;

                float* a = &acc[r * d];                  // pointer to this row of the accumulator
                for (size_t x = 0; x < d; ++x) a[x] *= alpha;        // shrink what we had
                for (size_t c = 0; c < bc; ++c) {                    // add P_tile * V_tile
                    const float* v = V + (c0 + c) * d;
                    const float p = s[c];
                    for (size_t x = 0; x < d; ++x) a[x] += p * v[x];
                }
                m[r] = m_new;
            }
        }
```

**C++ decoded:**

- **Absolute vs tile-local indices.** Row `r` of the tile is row `r0 + r` of the whole matrix; column `c` is `c0 + c`. The causal test compares the **absolute** positions `(c0 + c) > (r0 + r)`. Mixing local and absolute indices is a classic tiling bug.
- **`a[x] *= alpha`** is `a[x] = a[x] * alpha`. The compound assignment operators `+=`, `-=`, `*=`, `/=` work as in Python.
- **`float* a = &acc[r * d];`** then `a[x]`: a pointer to one row of a flat array, indexed like a 1-D array. The same trick makes the code readable without a 2-D type.
- The scores are **overwritten** with the probabilities (`s[c] = expf(...)`) to save a buffer. That is fine because `S` is not used afterwards; on a GPU this kind of in-place reuse is normal.
- The three loops over `c` and `x` in the update are `P @ Vj` written out. On a GPU these become Tensor Core operations (Part 3).

### Finishing a Q tile

```cpp
        // Normalise once, write the results for this Q tile.
        for (size_t r = 0; r < br; ++r) {
            const float* a = &acc[r * d];
            float* o = O + (r0 + r) * d;
            const float inv = (l[r] > 0.0f) ? 1.0f / l[r] : 0.0f;
            for (size_t x = 0; x < d; ++x) o[x] = a[x] * inv;
            L[r0 + r] = m[r] + logf(l[r]);
        }
    }
}
```

`inv` is computed once per row and multiplied in, rather than dividing `d` times. A fully masked row (`l = 0`) gets an output of zeros, the convention from Chapter 2.

### The test driver, briefly

`main_test.cpp` runs 216 configurations against `attention_naive`. New syntax it uses:

```cpp
    const size_t Ns[] = {1, 5, 37, 64, 100, 257};              // a C-style array
    const size_t tiles[][2] = {{1, 1}, {4, 7}, {16, 16}, {64, 32}, {128, 128}, {1000, 1000}};
    for (size_t N : Ns) { ... }
    for (bool causal : {false, true}) { ... }
    for (const auto& t : tiles) { ... t[0] ... t[1] ... }      // auto: the compiler fills in the type
```

- `const size_t Ns[] = {...}` is a fixed-size array, and the range-`for` works on it directly.
- **`auto`** means "deduce the type from the initialiser"; `const auto& t` is a read-only reference to each element (a pair of tile sizes), so nothing is copied.
- **The error measure must not lose a NaN.** The test computes the maximum absolute difference with this helper:

```cpp
float max_abs_diff(const float* a, const float* b, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (!(diff <= worst)) worst = diff;      // true for a bigger diff AND for NaN
    }
    return worst;
}
```

  My first version used `worst = std::max(worst, diff)`. That silently ignores NaN, because every comparison with NaN is false. **[ran]**:

```
std::max(0.0f, NAN) = 0
std::max(NAN, 0.0f) = nan
```

  I found this while writing §3.10: an implementation whose outputs were **all NaN** passed the original test suite with `0 failures`. Python's built-in `max(0.0, float("nan"))` has the same problem (it returns `0.0`); NumPy's `np.maximum` and `.max()` propagate NaN. The Python tests use those.

---

## 3.8 How compilation works: from `.cpp` to a running program

Chapter 2 introduced compiling as one command. With several files you need the stages, because they are what you see when something goes wrong, and CUDA builds add more stages (Chapter 4; I have not run `nvcc` here).

`build.sh` does this:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -c tiled_attention.cpp -o tiled_attention.o
g++ -std=c++17 -O2 -Wall -Wextra -c main_test.cpp       -o main_test.o
g++ tiled_attention.o main_test.o -o main_test
```

Each `-c` compiles **one source file** to an **object file** (machine code, but not yet a program). The last line **links** the object files into an executable. **[ran]** with no warnings, and all 216 tests pass.

The stages, on our own file **[ran]**:

| Stage | Command | Result for `tiled_attention.cpp` |
|---|---|---|
| Preprocess | `g++ -E` | 127 lines of source become **38,485 lines**: every `#include` is pasted in as text |
| Compile to assembly | `g++ -S -O2` | 1,294 lines of assembly |
| Assemble | `g++ -c -O2` | a 9,656-byte object file |
| Link | `g++ a.o b.o -o prog` | the executable |

The object file lists the symbols it defines (`T`) and the ones it needs from elsewhere (`U`) **[ran]** (`nm`):

```
0000000000000000 T _Z15attention_naivePKfS0_S0_PfS1_mmb
0000000000000410 T _Z15attention_tiledPKfS0_S0_PfS1_mmmmb
                 U _Znwm
                 U expf
                 U logf
                 U memset
```

The strange names are **mangled**: C++ encodes the parameter types into the name (`PKf` is "pointer to const float", `m` is `unsigned long`, `b` is `bool`) so that functions can be overloaded. `nm -C` shows them demangled:

```
T attention_tiled(float const*, float const*, float const*, float*, float*, unsigned long, unsigned long, unsigned long, unsigned long, bool)
```

**Reading link errors.** If you forget an object file **[ran]**:

```
$ g++ main_test.o -o broken
/usr/bin/ld: main_test.o: in function `main':
main_test.cpp:(.text.startup+0x5f4): undefined reference to `attention_naive(float const*, float const*, float const*, float*, float*, unsigned long, unsigned long, bool)'
```

The **compiler** was satisfied by the header (a declaration); the **linker** could not find a definition. Now a subtler case: I edited a copy of the header so that `N` and `d` were declared `int` instead of `size_t`. It **compiled fine**, then failed at link time **[ran]**:

```
main_test.cpp:(.text+0x6ed): undefined reference to `attention_tiled(float const*, float const*, float const*, float*, float*, int, int, unsigned long, unsigned long, bool)'
```

Look at the demangled signature: it has `int, int` where the object file has `unsigned long, unsigned long`. **A declaration that does not match the definition produces a different function name, so you get "undefined reference" instead of a type error.** When you see that message, compare the signature in the error with your definition.

Chapter 4 adds a CUDA compiler that handles GPU code and CPU code from one `.cu` file; the header/source split and these link errors carry over unchanged **[preview]**.

---

## 3.9 Testing so that tests can fail

**[ran]** The C++ suite compares `attention_tiled` with `attention_naive` on 216 configurations (`N ∈ {1, 5, 37, 64, 100, 257}`, `d ∈ {1, 8, 33}`, six tile shapes including `(1, 1)`, `(4, 7)` and `(1000, 1000)`, causal on and off), in float32:

```
216 configurations, 0 failures, worst |dO| = 1.490e-07, worst |dL| = 9.537e-07
```

`check_cpp.py` builds the program, dumps one case (`N = 37, d = 8, Br = 8, Bc = 16`, causal) and compares it with the **float64** NumPy reference on bit-identical inputs (the LCG scale is a power of two, so both languages generate exactly the same floats) **[ran]**:

```
C++ float32 tiled vs float64 naive reference : max|dO| = 5.77e-08  max|dL| = 2.24e-07
Python float64 tiled vs float64 naive        : max|dO| = 1.11e-16  max|dL| = 4.44e-16
```

A passing test suite only means something if it can fail. **Mutation testing** checks that: break the code on purpose, one bug at a time, and see whether the tests notice. `mutants.py` does this to the C++ file. **[ran]**:

| Deliberate bug | Failures (of 216) | Worst `|dO|` |
|---|---|---|
| forget to shrink the old accumulator (`a[x] *= alpha`) | 102 | 0.33 |
| forget `alpha` when updating `l` | 102 | 0.18 |
| forget the `1/√d` scale | 144 | 0.87 |
| forget the causal mask inside a tile | 75 | 1.27 |
| forget the final division by `l` | 180 | 15.4 |
| row stride `bc` instead of `Bc` when reading `S` | 129 | 0.91 |
| every output is `NaN` | 216 | `nan` |
| **skip the `−inf` guard** | **0** | 1.49e-07 |
| ragged last tile ignored (`br = Br`), built with AddressSanitizer | (aborted) | `heap-buffer-overflow` in `attention_tiled` |

Seven of the eight numerical bugs are caught, and the ragged-tile bug is caught by AddressSanitizer (`-fsanitize=address`) at the exact line. Note that the stride bug (`bc` vs `Bc`) is only visible on ragged tiles, which is why the test grid includes sizes like `37` and `257`. Note also the NaN row: with my original `std::max` harness that mutant had scored **0 failures** (§3.7).

**The one survivor is the lesson.** Removing the guard changes nothing the suite can see, because **no test has a row whose first tile is fully masked** (§3.4). The same blind spot means the whole C++ suite passes under `-ffast-math` **[ran]**:

```
-ffast-math with the NaN-safe harness:
216 configurations, 0 failures, worst |dO| = 1.416e-07, worst |dL| = 9.537e-07
```

Chapter 2 showed `-ffast-math` breaking the `−INFINITY` guard in a masked-start row. Here nothing breaks only because no test reaches the guard. The NumPy sliding-window run (§3.4) is the test that does reach it. Do not take comfort from a green suite that never exercises the risky code: Exercise 5 asks you to add a window mask to the C++ and watch fast-math fail.

---

## 3.10 Is it faster on a CPU? (optional, and an honest surprise)

`bench.cpp` times both C++ versions on one thread (`-O2`, `d = 64`, no mask, best of 3). The machine reports one core, a 48 KiB L1d, 2 MiB L2, and a 260 MiB L3. **[ran]**:

```
N=1024 d=64  naive              :     60.8 ms   (S matrix = 4 MiB)
N=1024 d=64  tiled Br=Bc=64     :     78.8 ms   (0.77x vs naive)
N=2048 d=64  naive              :    240.6 ms   (S matrix = 16 MiB)
N=2048 d=64  tiled Br=Bc=64     :    310.3 ms   (0.78x vs naive)
N=2048 d=64  tiled Br=Bc=256    :    285.8 ms   (0.84x vs naive)
N=4096 d=64  naive              :    997.3 ms   (S matrix = 64 MiB)
N=4096 d=64  tiled Br=Bc=64     :   1257.5 ms   (0.79x vs naive)
N=4096 d=64  tiled Br=Bc=256    :   1157.5 ms   (0.86x vs naive)
```

On this machine **tiling was 14 to 27% slower** for `N` up to 4096 (every tile size I tried, 16 to 256, was slower; the table shows a subset). Then I raised `N` until the score matrix no longer fit in the L3 cache (`bench_big.cpp`, `N = 12288`, one run each) **[ran]**:

```
N=12288 naive: 10539 ms (S = 576 MiB)
N=12288 tiled B=64: 8666 ms (1.22x vs naive)
N=12288 tiled B=256: 8734 ms (1.21x vs naive)
```

Now the tiled version wins by about 1.2×. My reading is **[predicted]**: while `S` (up to 64 MiB) fits inside this machine's very large L3 cache, the naive version's extra traffic is cheap, and the tiled version pays for its extra `exp` and rescale work; once `S` (576 MiB) spills to main memory, avoiding it pays off. I did not confirm this with hardware counters, and the cache sizes come from a virtual machine's report, which may not be accurate. Read the whole thing with care:

- It is a **CPU**, single-threaded, scalar code. A GPU has a very different memory system, so no number here transfers.
- A run on a different machine could put the crossover elsewhere. Timing on this sandbox is noisy and is only colour.
- What does transfer is the **principle**: tiling removes traffic to the slow level of the memory hierarchy, and it helps exactly when that traffic is the bottleneck (Chapter 1). On the GPU, where the naive version is memory-bound already, we expect it to matter at much smaller sizes. We measure that from Chapter 4 on.

---

## 3.11 How this maps onto CUDA (a preview)

| This chapter | In the CUDA kernel |
|---|---|
| Outer loop over Q tiles (`r0`) | **Grid dimension**: one thread block per Q tile per head, since Q tiles are independent (Chapter 9) |
| Inner loop over K/V tiles (`c0`) | A loop inside the kernel |
| `S`, `acc`, `m`, `l` (tile-sized buffers) | Registers and shared memory |
| `Q + i * d`, `V + (c0 + c) * d` | Kernel pointer arguments plus offsets; the `[B, H, N, d]` stride formula of §3.3 |
| `L` (logsumexp) | Saved output for the backward pass (Chapters 13, 22, 23) |
| `Br`, `Bc`, `d` as run-time parameters | **Compile-time template parameters** (Chapter 8) |
| Ragged last tile (`min(Br, N - r0)`) | Boundary handling when loading tiles (Chapter 10) |
| `std::vector<float>` | Device memory from `cudaMalloc` (Chapter 4) |
| The guard, the `bc` vs `Bc` distinction | The same bugs, harder to see |

---

## 3.12 Exercises

1. **Trace by hand.** With `N = 4`, `Br = Bc = 2`, non-causal, write down `m`, `l` and `acc` for the first Q tile after each of its two K/V tiles. Then check with a `print` inside `flash_attention_forward`.
2. **Confirm the traffic formula.** Predict the total traffic for `N = 2048, d = 128, Br = 64, Bc = 256` (Q-outer). Run `io_dry_run` and the instrumented `io=` counter. Are they equal? Was your prediction?
3. **Break the paper's factor.** Run `flash_attention_forward(..., _bad_rescale=True)` and explain, using Chapter 2's derivation, why the error is large. Why did I not simply get `NaN`?
4. **A tile-size sweep.** Using the traffic formula, for which `Br` does Q-outer traffic first fall below Algorithm 0 for `N = 1024, d = 64`? Verify against the table.
5. **Give the C++ a window mask.** Add a `window` parameter to `attention_tiled` and `attention_naive`, mirroring the Python. First run the suite with the guard removed (the mutant that survived), then again built with `-ffast-math` and the guard in place. What happens? Which of the two runs would have gone unnoticed before your new test?
6. **Row versus column.** Change `S[r * Bc + c]` to `S[c * Bc + r]` in both places it is used and run the tests. Which configurations fail, and why is it silent for `(1, 1)` tiles?
7. **Sanitise it.** Build `main_test` with `-fsanitize=address,undefined -g` and run. Then introduce an off-by-one in one of the `for` bounds and read the report.
8. **32-bit indices.** Change `size_t` to `int` in the indexing arithmetic of one loop. At what `N` and `d` would `i * d + x` overflow a 32-bit `int`?

---

## 3.13 Common pitfalls

- **Forgetting to rescale the old accumulator** (`acc *= alpha`) or the old sum. The output is right only when the max never changes.
- **`alpha` computed after overwriting `m`.** Compute `exp(m_old − m_new)` while both values are alive.
- **The paper's printed inverse** (§3.2). Test every formula you take from a paper.
- **Tile buffer stride:** `Bc` (the buffer's row length) versus `bc` (columns in use). Only ragged tiles reveal it.
- **Local versus absolute indices** in masks: use `(c0 + c) > (r0 + r)`.
- **Dividing by `l` at every step** instead of once at the end: correct but slower, and not what FlashAttention-2 does.
- **The `−inf` guard** never being tested (§3.9).
- **`std::max` and Python `max()` swallow NaN.** Use `np.maximum` in NumPy, or `!(diff <= worst)` in C++.
- **Dangling pointers** from `data()` after a vector grows (§3.6).
- **Unsigned subtraction** (`N - r0`) and **32-bit index overflow**.
- **Declaration and definition that disagree** (`undefined reference` at link time, §3.8).
- **Trusting one CPU timing.** It depends on cache sizes, flags and the size of the problem (§3.10).

---

## 3.14 Summary and bridge to Chapter 4

- Attention can be computed tile by tile with a running `(m, l, acc)` per row: one Q tile at a time, sweeping across K/V tiles, dividing by `l` once at the end. The result is exact (errors of 1e-15 in float64, and no worse than naive in float32).
- We save only the logsumexp `L = m + log(l)`; `exp(S − L)` rebuilds `P` for the backward pass.
- Traffic is about `2N²d / Br`: bigger tiles are better, and small tiles can be **worse** than standard attention. The on-chip memory size limits the tile size, which is the paper's `Θ(N²d²/M)`.
- Causal masking skips about half the tiles and masks only those on the diagonal. The `−inf` guard matters for other masks, and is easy to leave untested.
- In C++: pointers, `const float*`, row pointers (`Q + i * d`), flat row-major indexing, headers versus sources, the compile/link stages and how to read a link error, `-fsanitize=address`, and how to write tests that can fail.

**Next: Chapter 4, your first CUDA kernels.** We move to the GPU. Attention is written as three separate kernels (`QKᵀ`, softmax, `PV`), so you meet `__global__`, `threadIdx`, `blockIdx`, the `<<<grid, block>>>` launch, `cudaMalloc`/`cudaMemcpy` and an error-check macro. Everything in this chapter's tiled loop then comes back as a single fused kernel in Part 2.

---

## 3.15 Files for this chapter (`ch03_code/`)

| File | Purpose | Status |
|---|---|---|
| `tiled_attention.py` | NumPy naive and tiled attention, sliding window, counters; running it prints every table in §3.3 to §3.5 | **[ran]** |
| `core_listing.py` | the simplified listing of §3.3 (tested against the naive reference) | **[ran]** |
| `tiled_attention.h`, `tiled_attention.cpp` | the header and the C++ implementations of §3.7 | **[ran]** (GCC 13.3, no warnings) |
| `main_test.cpp` | 216-configuration test driver and the dump used by `check_cpp.py` | **[ran]** |
| `build.sh` | compile each file to an object, then link (run with `sh build.sh`) | **[ran]** |
| `check_cpp.py` | builds and runs the C++, compares with float64 NumPy | **[ran]** |
| `mutants.py` | mutation tests of §3.9 | **[ran]** |
| `pointers_demo.cpp`, `stride.cpp`, `dangling.cpp` | the pointer demos of §3.6 | **[ran]** |
| `nanmax.cpp` | `std::max` losing a NaN | **[ran]** |
| `bench.cpp`, `bench_big.cpp` | CPU timing of §3.10 (optional) | **[ran]** |

---

## 3.16 Verification status and loose ends

| Item | Status |
|---|---|
| NumPy tiled forward: 252-configuration grid, float32 comparison, `L` and `P` reconstruction, multi-head, strides, causal tile counts, sliding window with and without the guard | **Executed.** Outputs pasted above. |
| Traffic formulas | **Derived**, and checked against counters inside the running NumPy loop (exact match). |
| NumPy `tracemalloc` peaks | **Executed** (CPU/NumPy only, not PyTorch). |
| C++ tiled attention: 216 tests, C++↔Python cross-check, mutation tests, AddressSanitizer runs, `-ffast-math` run | **Executed.** GCC 13.3, `-std=c++17 -O2 -Wall -Wextra`, no warnings. |
| Compile stages, `nm` output, link errors | **Executed.** |
| Pointer, `const`, wrong-stride, dangling-pointer, and `std::max`/NaN demos | **Executed.** |
| CPU timing | **Executed**, on one single-core virtual machine, best of three (one run at `N = 12288`). Noisy. The explanation is **[predicted]**. |
| Bandwidths and on-chip limits for the T4 and RTX 3090 | From Chapter 1's sources, some secondary. **Derived** times are lower bounds. |
| Anything on a GPU; `nvcc` | **Not tested.** §3.11 and the `nvcc` remark in §3.8 are previews. |
| Paper claims | **Read directly from the papers.** They describe the papers' setups. |

**Loose ends.**

- **The paper's inverse.** My reading that the `⁻¹` in the FlashAttention-2 rescale is a typo rests on the algebra and on my test; I did not check other versions of the paper.
- **A bug I found in my own harness.** My first C++ test used `std::max` and could not see NaN (§3.7). All results in this chapter were re-run after the fix. Chapter 2's C++ printouts were also produced with `std::max` in a comparison, and its Python tests use the built-in `max`; I re-checked the Chapter 2 vectorised-sum tests for NaNs (none were present), but the Chapter 2 files still contain the old pattern.
- **The traffic model counts requests**, not bytes actually delivered by DRAM; caches will change real GPU numbers.
- **On-chip footprint** (§3.5) assumes all tiles sit in shared memory. Real kernels keep much of it in registers; Chapter 9 revisits it.
- **The CPU crossover** (§3.10) depends on the virtual machine's cache behaviour, which I could not inspect.

## 3.17 Sources

- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691. https://arxiv.org/abs/2307.08691 (§3.1.1: the two algorithm tweaks, Algorithm 1, causal-masking rules and the 1.7 to 1.8× figure; the extra-memory statement; the parallelisation along the sequence length)
- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* arXiv:2205.14135. https://arxiv.org/abs/2205.14135 (Algorithm 1: the K/V-outer loop with per-step reads and writes of `O`, `l`, `m`; the `Θ(N²d²/M)` IO analysis; the T4 observation; also read for Chapter 1)
- Milakov, Gimelshein. *Online normalizer calculation for softmax.* arXiv:1805.02867. https://arxiv.org/abs/1805.02867 (the recurrence from Chapter 2, reused here)
- Hardware figures (bandwidths, on-chip limits): the sources listed in Chapter 1, §1.12.
