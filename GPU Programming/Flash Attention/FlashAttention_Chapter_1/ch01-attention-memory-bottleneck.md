# Chapter 1: Attention and Why Memory Is the Bottleneck

**Series:** Flash Attention from Scratch in CUDA · Part 0 (the algorithm before the kernel)
**Language in this chapter:** Python / PyTorch only. C++ starts in Chapter 2.
**Hardware referenced:** RTX 3090 (Ampere, sm_86) and Tesla T4 (Turing, sm_75)

---

## What you will be able to do after this chapter

1. Write attention from its definition, check it against PyTorch's own implementation, and explain every line.
2. Say exactly which tensors are `N × N`, and calculate how much memory they take.
3. Count how many bytes a naive attention implementation moves through GPU memory, and compare that with the bytes the arithmetic actually needs.
4. Explain, with numbers, why this makes attention **memory-bound**, and place that on your own two GPUs.
5. State the two problems FlashAttention has to solve (previewed here, solved in Chapters 2 and 3).

## How this chapter is grounded

Every number below is one of four kinds, and is labelled that way:

| Label | Meaning |
|---|---|
| **[ran]** | I executed code and pasted its real output. |
| **[source]** | Taken from a named source (paper, datasheet, docs). Sources are listed in §1.12. |
| **[derived]** | Computed by formula from [source] inputs. Formulas are shown. |
| **[predicted]** | My reasoning about code I could not run. You are asked to measure it (§1.9). |

The sandbox I wrote this in has no GPU and no PyTorch. So the PyTorch listings were syntax-checked but not executed; the maths was executed in a NumPy twin that performs the same operations in the same order. Section 1.12 has the full status table. Treat your first run of the tests as the real test.

---

## 1.1 What attention computes

### The definition (one head)

Take a sequence of `N` tokens. Each token has been turned into three vectors of length `d` (the *head dimension*): a query, a key and a value. Stack them as matrices:

```
Q, K, V   each of shape [N, d]
```

Attention produces an output `O` of shape `[N, d]` in three steps:

```
S = Q · Kᵀ / √d          shape [N, N]   "scores": S[i][j] = how much query i matches key j
P = softmax(S, per row)  shape [N, N]   "probabilities": each row is positive and sums to 1
O = P · V                shape [N, d]   each output row is a weighted average of the value rows
```

Read row `i` of the result as: *"output i = average of all value vectors, where the weight on value j is how well query i matches key j."*

Two facts to hold onto for the rest of the series:

- `S` and `P` are **`N × N`**. `Q`, `K`, `V`, `O` are only **`N × d`**. In real models `N` (thousands) is much larger than `d` (64 to 128).
- The softmax is applied **across a whole row**: to normalise row `i`, you need all `N` scores in that row.

### Why divide by √d?

Scores are dot products of `d`-dimensional vectors. If the entries of `q` and `k` are independent with unit variance, the dot product has variance `d`, so its standard deviation grows like `√d`. Dividing by `√d` brings it back to about 1.

**[ran]** (random unit-variance vectors, 200,000 pairs each):

```
d= 16: std(q.k) =  3.996  (sqrt(d) =  4.000);  std(q.k/sqrt(d)) = 0.999
d= 64: std(q.k) =  7.990  (sqrt(d) =  8.000);  std(q.k/sqrt(d)) = 0.999
d=128: std(q.k) = 11.329  (sqrt(d) = 11.314);  std(q.k/sqrt(d)) = 1.001
```

Why it matters: unscaled scores are so spread out that the softmax collapses onto one key. **[ran]** (one random query against 1,024 random keys, `d = 64`, seed 0; this is a single draw, so treat it as an illustration):

```
  no scale: max prob = 0.798, entropy = 0.69 nats  (uniform over 1024 would be 6.93)
 1/sqrt(d): max prob = 0.017, entropy = 6.40 nats
```

### Batch and heads

Real models run many independent attention problems at once. The standard layout is:

```
Q, K, V, O   have shape [B, H, N, D]
             B = batch size, H = number of heads, N = sequence length, D = head dim
```

Every `(b, h)` pair is a separate, independent `N × N` problem. Nothing is shared between them. Remember this: in CUDA it will become the *grid*: one block of threads per (b, h, piece of the sequence).

### A worked example you can check by hand

**[ran]** `N = 3, d = 2`:

```
Q = K = [[1, 0],        V = [[1, 2],
         [0, 1],             [3, 4],
         [1, 1]]             [5, 6]]

Q·Kᵀ (before scaling):        S = Q·Kᵀ / √2:
[[1, 0, 1],                   [[0.7071, 0.0000, 0.7071],
 [0, 1, 1],                    [0.0000, 0.7071, 0.7071],
 [1, 1, 2]]                    [0.7071, 0.7071, 1.4142]]

P = softmax(S) per row:       row sums of P: [1, 1, 1]
[[0.4011, 0.1978, 0.4011],
 [0.1978, 0.4011, 0.4011],
 [0.2483, 0.2483, 0.5035]]

O = P·V:
[[3.0000, 4.0000],
 [3.4067, 4.4067],
 [3.5105, 4.5105]]
```

Check row 0 yourself: `0.4011·[1,2] + 0.1978·[3,4] + 0.4011·[5,6] = [3.0, 4.0]`.

### Causal masking

Language models generate text left to right, so token `i` must not look at tokens `j > i`. We enforce that by setting `S[i][j] = -inf` for `j > i` *before* the softmax; `exp(-inf) = 0`, so those weights become exactly zero.

**[ran]** same example with the causal mask:

```
P (causal):                   O (causal):
[[1.0000, 0.0000, 0.0000],    [[1.0000, 2.0000],
 [0.3302, 0.6698, 0.0000],     [2.3395, 3.3395],
 [0.2483, 0.2483, 0.5035]]     [3.5105, 4.5105]]
```

Row 0 can only see itself, so `P[0] = [1, 0, 0]` and `O[0] = V[0]`. Row 2 sees everything, so it equals the non-causal row 2.

---

## 1.2 The naive implementation in PyTorch

This is the file every later kernel gets tested against. Save it as `naive_attention.py`.

```python
import math

import torch


def naive_attention(q, k, v, causal=False, scale=None):
    """Attention exactly as written in the paper's equations.

    q, k, v : [B, H, N, D]  (batch, heads, sequence length, head dim)
    returns : [B, H, N, D]
    """
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    s = q @ k.transpose(-2, -1)          # S = Q K^T          -> [B, H, N, N]
    s = s * scale                        # S / sqrt(d)
    if causal:
        n_q, n_k = s.shape[-2], s.shape[-1]
        keep = torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).tril()
        s = s.masked_fill(~keep, float("-inf"))   # query i may not see keys j > i
    p = torch.softmax(s, dim=-1)         # P = row-wise softmax -> [B, H, N, N]
    return p @ v                         # O = P V             -> [B, H, N, D]


def reference_attention(q, k, v, causal=False):
    """fp32 oracle. Inputs in fp16/bf16 are upcast first, so a low-precision
    kernel is compared against a more precise answer, not against another
    low-precision computation."""
    return naive_attention(q.float(), k.float(), v.float(), causal=causal)
```

### Line by line

| Line | What it does | Note for a Python programmer |
|---|---|---|
| `q @ k.transpose(-2, -1)` | Batched matrix multiply. `transpose(-2, -1)` swaps the last two axes, turning `[B,H,N,D]` into `[B,H,D,N]`. | `@` is `torch.matmul`; it treats all leading axes as batch axes, so this is `B·H` independent `[N,D]·[D,N]` products. |
| `s * scale` | Element-wise multiply of the whole `[B,H,N,N]` tensor by a Python float. | This allocates a **new** `N × N` tensor. Remember that for §1.4. |
| `torch.ones(...).tril()` | Builds a lower-triangular boolean matrix: `True` where `j ≤ i`. | `tril()` keeps the lower triangle, including the diagonal. |
| `s.masked_fill(~keep, -inf)` | Where `keep` is `False` (the upper triangle), write `-inf`. | `~` on a bool tensor is logical NOT. `keep` is `[N,N]` and is broadcast across `B` and `H`. |
| `torch.softmax(s, dim=-1)` | Softmax along the last axis, i.e. across each row of keys. | PyTorch's softmax already subtracts the row max internally. See §1.3. |
| `p @ v` | `[B,H,N,N] · [B,H,N,D] → [B,H,N,D]`. | The second matmul. |

### Does it match PyTorch's own attention?

PyTorch ships `torch.nn.functional.scaled_dot_product_attention` (SDPA). Its documentation includes a reference implementation described as equivalent to what the fast kernels compute: it forms the scores, scales them, adds a mask bias, applies softmax, optionally dropout, then multiplies by `V` **[source]**. That is the same pipeline as `naive_attention`, which is why we use it as a cross-check.

Key facts from the docs **[source]**:

- Signature: `scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)`.
- `scale` can only be passed as a keyword argument; if omitted it defaults to `1/√d`.
- The function applies dropout whenever `dropout_p > 0` regardless of train/eval mode. We leave it at `0.0`.
- Recent PyTorch picks a fast backend automatically. We force the plain `MATH` backend for our test, so we compare against the textbook algorithm and not against a fused kernel.

Save as `test_naive_attention.py`:

```python
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel   # PyTorch >= 2.3

from naive_attention import naive_attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa_math_backend(causal):
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 128, 64
    q, k, v = (torch.randn(B, H, N, D, device=DEVICE) for _ in range(3))

    ours = naive_attention(q, k, v, causal=causal)
    with sdpa_kernel(SDPBackend.MATH):
        theirs = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # fp32 on both sides; differences come only from op ordering.
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-4)


def test_rows_of_p_sum_to_one():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 32, 16, device=DEVICE)
    k = torch.randn(1, 1, 32, 16, device=DEVICE)
    p = torch.softmax(q @ k.transpose(-2, -1) / 16 ** 0.5, dim=-1)
    torch.testing.assert_close(p.sum(-1), torch.ones(1, 1, 32, device=DEVICE))
```

Run: `pytest -q test_naive_attention.py`.

**What I verified.** I ran the NumPy twin of `naive_attention` against a from-the-definition, plain-loop version (`o_i = Σ_j softmax_j(q_i·k_j/√d) · v_j`) on random `[2, 3, 16, 8]` inputs. **[ran]** The maximum difference was `3.33e-16` (non-causal) and `4.44e-16` (causal), i.e. float64 rounding. The `pytest` file itself has **not** been run.

**If the SDPA test fails:** the tolerances are set for fp32. Errors near `1e-3` usually mean TF32 matmuls are on. PyTorch's default for `torch.set_float32_matmul_precision` is `"highest"` (full fp32) but some code sets it to `"high"`. Print it with `torch.get_float32_matmul_precision()`.

### The oracle rule (worth internalising)

`reference_attention` upcasts to fp32. Later, our fp16 kernels will be compared against **this**, not against a fp16 naive implementation. Comparing two low-precision computations hides errors in both.

---

## 1.3 Why softmax needs the row maximum

`softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`. Written this way it breaks for large `x`, because `exp` overflows.

**[ran]:**

```
float16 max value            = 65504  ->  ln(65504) = 11.09
exp(float16(12))             = inf    <- overflow at only x = 12
float32:  exp(88) = 1.65e+38,  exp(89) = inf
naive softmax([1000, 1001, 1002]) = [nan nan nan]
safe  softmax([1000, 1001, 1002]) = [0.09   0.2447 0.6652]
```

The fix is that softmax is unchanged if you subtract the same constant from every element: `softmax(x) = softmax(x − c)`. Choose `c = max(x)` so the largest exponent is `exp(0) = 1` and nothing overflows. **[ran]** the shifted and unshifted versions agree (`np.allclose` returned `True` when comparing `x − max` against `x − 1000`).

This running maximum is the `m` you will see in FlashAttention. In Chapter 2 we make it *incremental*, so it works when you have not yet seen the whole row.

**A trap to remember for Chapter 12.** A row that is entirely `-inf` has `max = -inf`, and `-inf − (−inf)` is `NaN`. **[ran]** `softmax([-inf, -inf, -inf])` gives `[nan nan nan]`. In plain causal attention every row has at least its own diagonal entry, so this does not happen, but it *will* happen for padded or skipped tiles inside a kernel.

---

## 1.4 Where the memory goes: the `N × N` matrices

Read `naive_attention` again and list the tensors that exist at the same time:

- Inputs/outputs: `Q, K, V, O`: each `N × d` (small).
- `s = q @ k.T`: `N × N`.
- `s = s * scale`: a **second** `N × N`. The first is still alive until the assignment finishes.
- `p = softmax(s)`: a third, while the scaled `s` is still alive.

### Size of one score matrix, per (batch, head)

Formula **[derived]**: `bytes = N² × bytes_per_element`.

**[ran]** (`roofline_numbers.py`):

```
      N       fp16       fp32
    512      0.5Mi      1.0Mi
   1024      2.0Mi      4.0Mi
   2048      8.0Mi     16.0Mi
   4096     32.0Mi     64.0Mi
   8192    128.0Mi    256.0Mi
  16384    512.0Mi   1024.0Mi
  32768   2048.0Mi   4096.0Mi
```

Doubling `N` quadruples the matrix. Compare with the inputs: at `N = 4096, d = 64` in fp16, `Q` is `4096 × 64 × 2 B = 0.5 MiB`, but one score matrix is 32 MiB, exactly `N/d = 64×` larger.

### Does it fit on your GPUs?

Setup: batch 1, 32 heads, fp16. Assumption **[predicted]**: about **two** `N × N` tensors are alive at the peak (from the list above). I have not measured this; Exercise 1 asks you to.

**[derived]** with 24 GiB (RTX 3090) and 16 GiB (T4) as the nominal memory sizes; the usable amount is a little lower:

```
N=  2048: S=  0.25 GiB, 2xS=  0.50 GiB | RTX 3090: fits | Tesla T4: fits
N=  4096: S=  1.00 GiB, 2xS=  2.00 GiB | RTX 3090: fits | Tesla T4: fits
N=  8192: S=  4.00 GiB, 2xS=  8.00 GiB | RTX 3090: fits | Tesla T4: fits
N= 12288: S=  9.00 GiB, 2xS= 18.00 GiB | RTX 3090: fits | Tesla T4: OOM
N= 16384: S= 16.00 GiB, 2xS= 32.00 GiB | RTX 3090: OOM  | Tesla T4: OOM
```

This ignores the model weights and everything else in memory, so real limits arrive sooner. And it is only about *capacity*. The more important cost is *time*.

A data point from the paper's memory benchmark **[source]** (A100 40 GB, batch 16, 8 heads, `d = 64`, fp16, forward plus backward, no dropout or mask):

| `N` | Standard PyTorch attention | FlashAttention |
|---|---|---|
| 4,096 | 17,024 MB | 836 MB |
| 8,192 | out of memory | 1,672 MB |
| 65,536 | out of memory | 13,376 MB |

Two things to read off it. FlashAttention's footprint doubles when `N` doubles (linear), while the standard version quadruples. And at `N = 4,096` the standard version used about 4.2 times one fp16 score matrix for that shape (`128 (batch × heads) × 4096² × 2 B = 4,096 MiB`; `17,024 / 4,096 ≈ 4.2`) **[derived]**. That is more than the roughly 2× I predict above for a forward-only run, which is what you would expect once autograd keeps intermediates alive for the backward pass. Exercise 1 has you check both cases.

---

## 1.5 Where the time goes: memory traffic and arithmetic intensity

### The execution model in three sentences

Each PyTorch operation (`@`, `*`, `masked_fill`, `softmax`) launches one or more **GPU kernels**. Each kernel reads its inputs from device memory, computes, and writes its outputs back to device memory. The next kernel then reads those outputs again from device memory. The paper describes the same model: kernels load from the large, slow memory into registers and on-chip memory, compute, and write results back **[source]**.

Every arrow through device memory costs time proportional to the bytes moved.

### Counting the traffic of `naive_attention`

Per `(b, h)`, non-causal, in units of elements:

| Operation | Reads | Writes |
|---|---|---|
| `S = Q·Kᵀ` | `Q`, `K` (2·N·d) | `S` (N²) |
| `S·scale` | `S` (N²) | `S'` (N²) |
| `softmax` | `S'` (N²) | `P` (N²) |
| `O = P·V` | `P`, `V` (N² + N·d) | `O` (N·d) |
| **Total** | | **6·N² + 4·N·d** |

The paper's minimal "standard attention" (its Algorithm 0) writes `S`, reads it back, writes `P`, reads it back, giving **4·N² + 4·N·d** **[source]**. It has no separate scaling pass. A real implementation could fold the `1/√d` into the matmul (for example by scaling `Q` first), but our eager code does not, so it makes two extra passes over an `N × N` tensor.

### The two quantities to compare

- **FLOPs** (floating-point operations) that the mathematics requires: `S = QKᵀ` costs `2·N²·d`, `O = PV` costs `2·N²·d`, so the matmuls total `4·N²·d`. **[derived]**
- **Bytes** that the implementation moves.

Their ratio is the **arithmetic intensity**: FLOPs per byte moved. A GPU has a peak compute rate (FLOP/s) and a peak memory rate (byte/s). Dividing one by the other gives the **ridge point**:

```
ridge = peak FLOP/s ÷ peak bytes/s        (FLOP per byte)
```

- If a kernel's intensity is **below** the ridge, memory delivers data slower than the arithmetic units could consume it. The kernel is **memory-bound**: its time is set by bytes moved, and making the arithmetic faster does nothing.
- If it is **above** the ridge, it is **compute-bound**.

(This is the roofline model. The paper also cites it, and defines the same compute-bound / memory-bound split; it lists matmul with a large inner dimension as compute-bound and reductions like softmax as memory-bound **[source]**.)

### The hardware numbers

Inputs used below **[source]**, with the arithmetic checks I could do **[derived]**:

| | RTX 3090 | Tesla T4 |
|---|---|---|
| Architecture | Ampere (GA102), compute capability 8.6 | Turing, compute capability 7.5 |
| Device memory | 24 GB GDDR6X | 16 GB GDDR6 |
| Memory bandwidth | 936 GB/s (384-bit × 19.5 Gbps ÷ 8 ✓) | 300 GB/s per NVIDIA datasheet; NVIDIA's product page says "320+" |
| FP32 (CUDA cores) | 35.6 TFLOPS (10,496 cores × 2 × 1.695 GHz ✓) | 8.1 TFLOPS |
| FP16 Tensor Cores, FP32 accumulate | ~71 TFLOPS dense (whitepaper figure via a third-party table; an NVIDIA forum reply says "around 70") | 65 TFLOPS (NVIDIA datasheet "mixed precision FP16/FP32") |
| SMs | 82 | 2,560 CUDA cores; at 64 per Turing SM, 40 (not directly sourced) |

We use FP16 Tensor Cores with FP32 accumulation because that is what FlashAttention kernels use.

### The numbers

Workload: the paper's GPT-2-medium shape, but forward pass only: `B = 64, H = 16, N = 1024, d = 64`, fp16. **[ran]** (`roofline_numbers.py`, all values derived from the formulas above):

```
FLOPs (2 matmuls)          :    274.9 GFLOP
Algorithm 0 (paper)       :     9.13 GB moved  -> intensity    30.1 FLOP/B
our eager code            :    13.42 GB moved  -> intensity    20.5 FLOP/B
floor: Q,K,V in, O out    :     0.54 GB moved  -> intensity   512.0 FLOP/B
ratio Algorithm 0 / floor  :  17.0x

== ridge point = peak FLOP/s / peak B/s ==
RTX 3090: fp32 ridge =   38.0 FLOP/B,  tensor-core ridge =   75.9 FLOP/B
Tesla T4: fp32 ridge =   27.0 FLOP/B,  tensor-core ridge =  216.7 FLOP/B
```

The **floor** row is the fewest bytes *any* implementation must move: read `Q`, `K`, `V` once, write `O` once (4·N·d elements). Everything above the floor is the `N × N` matrices going to memory and back.

And the resulting lower bounds on time **[derived]**. These use datasheet peaks; a real kernel is always slower, so read them as bounds, not predictions:

```
RTX 3090: memory floor (Alg 0)   9.75 ms | (eager) 14.34 ms | compute @fp32  7.72 ms | compute @tensor 3.87 ms
Tesla T4: memory floor (Alg 0)  30.42 ms | (eager) 44.74 ms | compute @fp32 33.94 ms | compute @tensor  4.23 ms
```

### What this says

1. **On the Tensor Core path, naive attention is memory-bound on both cards.** Its intensity (about 30 FLOP/B, or 20.5 for our eager code) is below the tensor-core ridge (76 on the 3090, 217 on the T4). Memory time alone (9.75 ms) is about 2.5× the compute time (3.87 ms) on the 3090, and about 7× on the T4 (30.4 ms vs 4.2 ms).
2. **"Memory-bound" depends on which compute path you use.** With plain FP32 on the T4, compute time (33.9 ms) is slightly above the memory floor (30.4 ms), so it is close to balanced. The bottleneck only becomes memory once the arithmetic is made fast, and making the arithmetic fast is exactly what we will spend Part 3 doing.
3. **Naive intensity is flat in `N`; the floor's grows with `N`.** **[ran]**, fp16, `d = 64`:

```
     N  naive (Alg 0)    floor
   128           21.3     64.0
   256           25.6    128.0
   512           28.4    256.0
  1024           30.1    512.0
  2048           31.0   1024.0
  4096           31.5   2048.0
```

Naive attention tops out near `d/2` FLOP/B (32 here), because every score is written and re-read. An implementation that never sends the scores to device memory has an intensity ceiling of about `N/2`, far above any ridge point. That gap is the whole opportunity.

4. **Softmax is the worst part.** Per element it does roughly a handful of arithmetic operations (compare, subtract, `exp`, add, divide) while moving a 2-byte read and a 2-byte write, so on the order of 1 FLOP/B. **[predicted]**: my estimate from counting operations, not a measurement. It is far below any ridge point on either card.

### Cross-check against the paper's own measurements

The paper measured standard attention against its fused kernel for GPT-2 medium (`N = 1024, d = 64, 16 heads, batch 64`) on an A100, forward plus backward **[source]**:

| | Standard | FlashAttention |
|---|---|---|
| Device-memory reads+writes | 40.3 GB | 4.4 GB |
| Runtime | 41.7 ms | 7.3 ms |

Two things to take from that:

- The paper also notes FlashAttention does **more** arithmetic (it recomputes values in the backward pass) and is still much faster, because it moves far less data. That is the memory-bound story in one sentence.
- These are not directly comparable to my forward-only numbers: theirs are forward plus backward, measured on an A100. Mine are a forward-only count from formulas. The 40.3 GB is the same order of magnitude as my 9.13 GB scaled up for a backward pass; no more than that is claimed.

---

## 1.6 Your two GPUs in this picture

Two facts from the paper's appendix on other hardware **[source]**, both about *your* cards:

- **RTX 3090:** measured speedups over standard PyTorch attention of about **2.5× to 4.5×**, slightly higher than on an A100. The authors attribute this to the 3090's lower memory bandwidth (roughly 900 GB/s vs 1.5 TB/s on the A100): the less bandwidth you have, the more you gain from moving fewer bytes.
- **T4:** **smaller gains**. The T4 has less on-chip memory than an A100, so FlashAttention must use smaller blocks, which means more passes and more memory traffic. This matches the paper's IO-complexity analysis.

So expect the payoff of the kernel we build to be larger on the 3090 than on the T4. We will measure it in Chapter 24 and see whether that holds for our implementation.

Two more facts worth noting now, because they shape later chapters:

- **On-chip memory limits (for Chapter 9).** Per NVIDIA's compute-capability table, the T4 (cc 7.5) allows up to 64 KB of shared memory per SM and per block, and the 3090 (cc 8.6) allows 100 KB per SM and 99 KB per block. Going above 48 KB per block requires an explicit opt-in call. **[source]** (NVIDIA developer blog table and NVIDIA forum threads). We will confirm this with `cudaDeviceProp` on your machines in Chapter 9 instead of trusting it.
- **The official library.** The reference FlashAttention-2 repository lists support for Ampere, Ada and Hopper GPUs, and says Turing GPUs like the T4 should use a separate repository or FlashAttention 1.x **[source]**. So for the Chapter 24 benchmarks, the 3090 can use the official kernel as a yardstick; the T4 cannot use FlashAttention-2.

### A naming note

The paper calls the big, slow memory "HBM". That is what an A100 has. Your cards use GDDR memory instead. The idea is identical, but from here on the series says **global memory** (or **device memory**) for the large off-chip memory and **shared memory / registers** for the small on-chip memory. When you read the paper, mentally translate "HBM" to "global memory".

---

## 1.7 The idea that solves it (preview only)

The paper states the goal as: never read or write the `N × N` matrices to and from device memory. It names two obstacles and one technique for each **[source]**:

| Obstacle | Technique | Chapter |
|---|---|---|
| The softmax needs the whole row, but the whole row does not fit on-chip. | **Tiling**: split `Q`, `K`, `V` into blocks, and compute softmax incrementally by carrying a running max and a running sum. | 2, 3 |
| The backward pass normally needs `S` and `P`, which we refuse to store. | **Recomputation**: save only the softmax statistics from the forward pass and recompute `S`, `P` block by block on-chip. | 22, 23 |

The result is a single fused kernel (matmul, softmax, matmul in one launch) that reads `Q`, `K`, `V` and writes `O` with `N × N` values living only in registers and shared memory. The paper's analysis gives device-memory accesses of `Θ(N²d²/M)` for FlashAttention against `Θ(Nd + N²)` for standard attention, where `M` is the on-chip memory size. For typical `d` (64 to 128) and `M` (around 100 KB), `d²` is many times smaller than `M`, so this is far fewer accesses **[source]**.

You do not need the proof yet. You need the picture: **make the arithmetic-to-bytes ratio high by keeping the `N × N` intermediates on-chip.**

---

## 1.8 Hands-on: run everything

Create a folder and put these files in it (they accompany this chapter):

| File | Needs GPU? | Purpose |
|---|---|---|
| `check_env.py` | uses it if present | Prints GPU name, compute capability, SM count, memory. |
| `naive_attention.py` | no | The oracle (§1.2). |
| `test_naive_attention.py` | no | Cross-check against PyTorch (§1.2). |
| `verify_numpy.py` | no | NumPy twin; prints the worked example and overflow demos (§1.1, §1.3). **[ran]** |
| `roofline_numbers.py` | no | Prints every table in §1.4 to §1.5. **[ran]** |
| `measure_memory.py` | **yes** | Exercise 1. |

Step 1: environment.

```bash
python check_env.py
```

`check_env.py`:

```python
import torch

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("name               :", p.name)
    print("compute capability :", f"{p.major}.{p.minor}", f"(sm_{p.major}{p.minor})")
    print("SM count           :", p.multi_processor_count)
    print("total memory       :", round(p.total_memory / 2**30, 2), "GiB")
    print("fp32 matmul mode   :", torch.get_float32_matmul_precision(), "(want 'highest' for oracle tests)")
```

Expect `8.6` on the 3090 and `7.5` on the T4. If you get something else, stop and check which GPU PyTorch sees.

Step 2: correctness.

```bash
pytest -q test_naive_attention.py
python verify_numpy.py
python roofline_numbers.py
```

Step 3 (GPU): memory measurement, `measure_memory.py`:

```python
import torch

from naive_attention import naive_attention


@torch.no_grad()
def peak_extra_bytes(B, H, N, D, dtype=torch.float16, causal=False):
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=dtype) for _ in range(3))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = naive_attention(q, k, v, causal=causal)
    torch.cuda.synchronize()               # GPU work is asynchronous: wait before reading stats
    peak = torch.cuda.max_memory_allocated()
    del out
    return peak - before


if __name__ == "__main__":
    B, H, D = 1, 8, 64
    dtype = torch.float16
    bytes_per = torch.finfo(dtype).bits // 8
    print(f"GPU: {torch.cuda.get_device_name(0)}   B={B} H={H} D={D} dtype={dtype}")
    for N in (1024, 2048, 4096, 8192):
        one_S = B * H * N * N * bytes_per
        try:
            extra = peak_extra_bytes(B, H, N, D, dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"N={N:5d}  out of memory (one S = {one_S / 2**20:8.1f} MiB)")
            break
        print(f"N={N:5d}  peak extra = {extra / 2**20:9.1f} MiB   one S = {one_S / 2**20:9.1f} MiB   ratio = {extra / one_S:4.2f}")
```

**Two Python-programmer notes about this script:**

- **GPU calls are asynchronous.** When Python runs `naive_attention(...)` it only *queues* the kernels and returns immediately. Reading memory statistics or timing without `torch.cuda.synchronize()` measures the queueing, not the work. Every benchmark in this series calls it.
- **`torch.no_grad()`** stops PyTorch from saving intermediates for autograd, which would otherwise keep extra `N × N` tensors alive and change the answer.

---

## 1.9 Exercises: predict, then measure

**Exercise 1: memory ratio.** Run `measure_memory.py` on each GPU.
My prediction **[predicted]**, from reading the code: the peak is about **2.0×** one `S` matrix (non-causal), plus a small `O` term of relative size `D/N` (about 6% at `N = 1024`, under 1% at `N = 8192`). The two-`S` moments are (a) the multiplication `s * scale`, where the old and new `s` coexist, and (b) the softmax, where the scaled `s` and the new `p` coexist. If your ratio is very different from 2, find out why before moving on; that is the point of the exercise. Also report the largest `N` that runs on each card and compare with the §1.4 table.

*Part b (training mode).* Remove `@torch.no_grad()`, create `q, k, v` with `requires_grad=True`, and run `naive_attention(...).sum().backward()` before reading the peak. Predict whether the ratio goes up or down, then measure. The paper's data point (about 4.2 score matrices at `N = 4,096`, §1.4) is your reference for what forward plus backward looks like on a different card.

**Exercise 2: causal memory.** Predict the ratio for `causal=True` in fp16, then measure. Derivation **[predicted]**: at the `masked_fill` call the live tensors are the scaled `s` (1 S), the boolean `keep` mask (`N²` bytes = 0.5 S in fp16), its negation `~keep` (another 0.5 S) and the new output (1 S), for a peak of about **3.0×**. Would the ratio be higher or lower in fp32, and by how much?

**Exercise 3: roofline by hand.** Pick your own shape (say `B = 1, H = 32, N = 4096, d = 128`). Compute the FLOPs, the Algorithm 0 bytes, the intensity, and both lower-bound times for each GPU using the formulas in §1.5. Then edit `roofline_numbers.py` to check your arithmetic.

**Exercise 4: map the paper to the code.** Read Section 2.2 and Algorithm 0 of the paper. Mark which line of `naive_attention` corresponds to each numbered step, and find where our eager code does extra passes that Algorithm 0 avoids.

**Exercise 5: break the softmax.** Write your own softmax without the max subtraction, feed it fp16 values, and find the smallest score at which it returns `inf`. (§1.3 gives you the threshold to expect.) Then fix it and confirm the answer is unchanged for inputs that did not overflow.

---

## 1.10 Common pitfalls

- **Masking with `0` instead of `-inf`.** A zero score still gets probability `exp(0) / Σ`. Masked entries must be `-inf` so their weight is exactly zero.
- **Forgetting the `1/√d` scale.** The output is still a valid attention output, just a badly conditioned one (see the §1.1 numbers), and it will not match PyTorch's.
- **Comparing fp16 to fp16.** Compare low-precision kernels against `reference_attention` (fp32), not against a low-precision naive version.
- **TF32 silently on.** It loosens tolerances to about `1e-3`. Check `torch.get_float32_matmul_precision()`.
- **Timing without `torch.cuda.synchronize()`.** You will measure launch overhead and think your code is very fast.
- **`is_causal` with `N_q ≠ N_k`.** In our square case it does not matter, but PyTorch's causal mask alignment for non-square shapes has its own convention; the docs' reference code builds it with `tril(diagonal=0)` from the top-left corner. We only use square cases in this series.
- **Assuming `attn_mask` and `is_causal` can be combined.** In PyTorch's reference code they are mutually exclusive (it asserts on that).

---

## 1.11 Summary and bridge to Chapter 2

- Attention is `softmax(Q Kᵀ / √d) V`. `Q, K, V, O` are `N × d`; the intermediate `S` and `P` are `N × N`.
- A straightforward implementation writes and rereads `N × N` matrices through device memory: about `4N² + 4Nd` elements for the paper's Algorithm 0, and about `6N² + 4Nd` for our eager PyTorch code. The theoretical floor is `4Nd`.
- For the paper's GPT-2-medium shape in fp16 that is about 17× more traffic than the floor. Its arithmetic intensity (about 30 FLOP/B) sits below the Tensor Core ridge point on both of your GPUs (about 76 on the 3090, about 217 on the T4). So it is **memory-bound** on the fast path.
- The way out is to keep `N × N` values on-chip: tile the computation, and solve the softmax-needs-the-whole-row problem with a running maximum and running sum.

**Next: Chapter 2, Online softmax.** We derive the incremental softmax in Python, then write your first C++ program to implement it. You will meet `#include`, `float`, `for` loops, `std::vector` and `expf`, with the Python equivalent for every construct.

---

## 1.12 Sources and verification status

### Verification status

| Item | Status |
|---|---|
| NumPy twin (`verify_numpy.py`): worked example, causal example, agreement with a loop implementation, overflow demos, all-`-inf` row | **Executed.** Outputs pasted above. |
| `1/√d` variance and sharpness check | **Executed** (small script, outputs pasted in §1.1). |
| `roofline_numbers.py`: every table in §1.4 and §1.5 | **Executed.** Inputs are datasheet values; outputs are formula results. |
| `naive_attention.py`, `test_naive_attention.py`, `measure_memory.py`, `check_env.py` | **Syntax-checked only.** No PyTorch or GPU in the environment they were written in. Your first run is the real test. |
| "≈2× one S" peak-memory claim and the causal "≈3×" claim | **Predicted from the code, not measured.** Exercises 1 and 2. |
| Paper figures (A100, 3090 and T4 speedup observations) | **Read directly from the paper.** They describe the paper's hardware and setup, not your machines. |
| GPU specifications | From datasheets and secondary sources; some cross-checked by arithmetic (see §1.5). |

### Known loose ends

- **T4 memory bandwidth** is 300 GB/s in NVIDIA's datasheet and "320+ GB/s" on NVIDIA's product page. I used 300 GB/s. With 320 the T4 tensor-core ridge is about 203 instead of 217; none of the conclusions change.
- **RTX 3090 Tensor Core peak (about 71 TFLOPS)** comes from a third-party table reproducing the GA102 whitepaper plus an NVIDIA forum reply of roughly the same number; I did not read the whitepaper table directly. This number only affects the "compute @tensor" time and ridge point.
- **The paper's GFLOPs column.** The paper's Figure 2 table lists 66.6 and 75.2 GFLOPs for standard vs FlashAttention. I could not reproduce the scale of that column from the configuration in the caption (my forward-only matmul count for that configuration is about 275 GFLOP), so I do not use it. The direction (FlashAttention does somewhat more arithmetic) is stated in the paper's text.
- **Shared-memory limits** per SM and per block come from NVIDIA's blog table and forum answers, not from the CUDA Programming Guide table directly. Chapter 9 checks them with the device query.

### Sources

- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* arXiv:2205.14135. https://arxiv.org/abs/2205.14135 (Sections 1, 2, 3, Appendix B, C, E.5; Figure 2)
- PyTorch documentation: `torch.nn.functional.scaled_dot_product_attention`. https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
- NVIDIA T4 Tensor Core GPU datasheet. https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf and product page https://www.nvidia.com/en-in/data-center/tesla-t4/
- NVIDIA Ampere GA102 GPU Architecture whitepaper. https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf
- RTX 3090 specification listings (memory 24 GB GDDR6X, 936 GB/s, 82 SMs, 10,496 CUDA cores, compute capability 8.6). https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
- NVIDIA developer forums on shared memory limits for compute capability 7.5. https://forums.developer.nvidia.com/t/max-shared-memory/144409
- NVIDIA compute-capability table (shared memory per SM and per block). https://developer.nvidia.com/zh-cn/blog/cuda-computing-power-cn
- FlashAttention-2 repository (supported GPUs). https://github.com/Dao-AILab/flash-attention
