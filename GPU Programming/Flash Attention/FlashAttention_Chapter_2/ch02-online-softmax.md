# Chapter 2: Online Softmax, the Key Trick

**Series:** Flash Attention from Scratch in CUDA · Part 0 (the algorithm before the kernel)
**Languages:** Python, then your first C++ file
**Builds on:** Chapter 1 (attention, the `N × N` score matrix, why softmax needs a row maximum)

---

## What you will be able to do after this chapter

1. Derive the online softmax recurrence yourself, and say why each term is there.
2. Write softmax as one running `(max, sum)` pair that is updated one score at a time.
3. Merge the `(max, sum)` states of two chunks of a row, in any order. This is what makes tiling (Chapter 3) and parallel reduction (Chapter 7) legal.
4. Compute the softmax-weighted sum `Σ softmax(s)ⱼ · vⱼ` in a **single pass**, without ever holding the whole row. This is the heart of FlashAttention.
5. Read, compile and run a small C++ program, and explain every construct in it using its Python equivalent.

## How this chapter is grounded

Every number is labelled:

| Label | Meaning |
|---|---|
| **[ran]** | I executed the code in this chapter and pasted the real output. This time that includes the C++ (GCC 13.3) and the Python. |
| **[source]** | Taken from a named source (Section 2.14). |
| **[derived]** | Computed by formula or algebra shown in the text. |
| **[predicted]** | My reasoning about something I did not measure. |

Standard mathematical results I did not re-prove are marked **[math fact]**, and C++ language rules that I did not demonstrate are marked **[language rule]**. Nothing in this chapter needs a GPU or PyTorch. Everything below can be reproduced on any machine with Python 3, NumPy and `g++`. The code files are alongside this chapter.

---

## 2.1 The problem: a row that arrives in pieces

From Chapter 1, the softmax of a row of scores `x₁ … x_N` is

```
softmax(x)ᵢ = exp(xᵢ − m) / d        where  m = max(x),   d = Σⱼ exp(xⱼ − m)
```

The `− m` keeps `exp` from overflowing (Chapter 1, §1.3). The catch is that **`m` depends on the whole row**, and `d` depends on `m`. The standard "safe softmax" therefore takes three passes over the data:

```python
import math

NEG_INF = float("-inf")

def softmax_3pass(x):
    m = NEG_INF
    for xi in x:                       # pass 1: maximum
        m = max(m, xi)
    d = 0.0
    for xi in x:                       # pass 2: normaliser
        d += math.exp(xi - m)
    return [math.exp(xi - m) / d for xi in x]   # pass 3: outputs
```

Milakov and Gimelshein count the memory traffic of this: three passes, which is **4 memory accesses per element** (read `x` in pass 1, read it in pass 2, read it and write `y` in pass 3) **[source]**.

Why this matters for FlashAttention: we want to process each row of `S` in tiles that live on-chip, one tile at a time, and never store the row. But pass 1 cannot finish until every tile has been seen. We need a way to get the right answer while seeing the row **incrementally**.

---

## 2.2 The idea: keep a running max and a running sum, and repair the sum

Process the elements one at a time. After `j` elements keep two numbers:

```
m_j = max(x₁ … x_j)                      the max so far
d_j = Σ_{k ≤ j} exp(x_k − m_j)           the sum so far, measured relative to the max so far
```

The final `m_N` and `d_N` are exactly the `m` and `d` that the 3-pass algorithm computes. The only question is how to get `(m_j, d_j)` from `(m_{j−1}, d_{j−1})` and the new element `x_j`.

### Derivation (worth doing on paper once)

Start from the definition and split off the last term:

```
d_j = Σ_{k ≤ j−1} exp(x_k − m_j)  +  exp(x_j − m_j)
```

The old sum was measured against the old max. Convert it by multiplying and dividing by `exp(m_{j−1})`:

```
exp(x_k − m_j) = exp(x_k − m_{j−1}) · exp(m_{j−1} − m_j)
```

So the first sum equals `d_{j−1} · exp(m_{j−1} − m_j)`, and

```
d_j = d_{j−1} · exp(m_{j−1} − m_j)  +  exp(x_j − m_j)          with  m_j = max(m_{j−1}, x_j)
```

This is Algorithm 3 in the online-softmax paper **[source]**. Two cases:

- **The max did not change** (`x_j ≤ m_{j−1}`): `m_j = m_{j−1}`, the factor is `exp(0) = 1`, and we just add `exp(x_j − m)`. Same as the 3-pass.
- **The max increased** (`x_j > m_{j−1}`): the factor `exp(m_{j−1} − m_j)` is between 0 and 1. It **shrinks** everything accumulated so far, because that sum was measured against a smaller max. The new element contributes `exp(0) = 1`.

The factor `exp(m_old − m_new)` is the **rescale factor**. It shows up everywhere in FlashAttention.

### A trace you can check by hand

**[ran]** `x = [1, 3, 2, 4]`:

```
  j=0  x= 1.00  m_old= -inf  m_new= 1.00  rescale=exp(m_old-m_new)=0.0000  d=1.0000
  j=1  x= 3.00  m_old= 1.00  m_new= 3.00  rescale=exp(m_old-m_new)=0.1353  d=1.1353
  j=2  x= 2.00  m_old= 3.00  m_new= 3.00  rescale=exp(m_old-m_new)=1.0000  d=1.5032
  j=3  x= 4.00  m_old= 3.00  m_new= 4.00  rescale=exp(m_old-m_new)=0.3679  d=1.5530
  final (m, d) = (4.0, 1.553002);  direct sum exp(x - 4) = 1.553002
```

Check step `j=1` yourself: `1.0 × 0.1353 + exp(3 − 3) = 1.1353`. At `j=0` the old max is `−inf`, so `exp(−inf) = 0` and the (empty) old sum contributes nothing. That is why we can start from `m = −inf, d = 0` with no special case.

### The code

```python
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


def online_stats(x):
    m, d = NEG_INF, 0.0
    for xj in x:
        m, d = push(m, d, xj)
    return m, d


def softmax_online(x):
    m, d = online_stats(x)
    if d == 0.0:                       # fully masked row: our convention is all zeros
        return [0.0] * len(x)
    return [math.exp(xi - m) / d for xi in x]
```

`softmax_online` still makes two passes (one for the statistics, one to produce the outputs), so it needs **3 memory accesses per element** instead of 4 **[source]**.

### Does it match?

**[ran]** Maximum absolute difference against a NumPy float64 reference, for awkward inputs (`N = 257` unless stated):

```
  random normal        |3pass-ref|=1.11e-16  |online-ref|=1.11e-16  sum(online)=1.000000
  offset +1000         |3pass-ref|=5.55e-17  |online-ref|=8.33e-17  sum(online)=1.000000
  increasing           |3pass-ref|=2.78e-17  |online-ref|=2.78e-17  sum(online)=1.000000
  decreasing           |3pass-ref|=2.78e-17  |online-ref|=2.78e-17  sum(online)=1.000000
  constant             |3pass-ref|=0.00e+00  |online-ref|=0.00e+00  sum(online)=1.000000
  single element       |3pass-ref|=0.00e+00  |online-ref|=0.00e+00  sum(online)=1.000000
  -inf in the middle   |3pass-ref|=1.73e-18  |online-ref|=6.94e-18  sum(online)=1.000000
  -inf at the START    |3pass-ref|=1.39e-17  |online-ref|=6.94e-18  sum(online)=1.000000
```

Both agree with the reference to about `1e-16` (double-precision rounding). The "increasing" row is the worst case for the online algorithm, because the maximum changes at every single step, and it is still exact to rounding.

---

## 2.3 The guard for masked entries

Look at the last line of `push`'s body: the `if m_new == NEG_INF` guard. It is easy to miss, and without it things break.

Causal masking (Chapter 1) sets scores to `−inf`. If a row **starts** with `−inf` entries, then `m_old = −inf` and `x = −inf`, so `m_new = −inf`, and the rescale factor becomes `exp(−inf − (−inf)) = exp(NaN) = NaN`. The NaN then poisons every later step.

**[ran]** `x = [-inf, -inf, 1.0, 2.0, 0.5]`:

```
  unguarded: [nan nan nan nan nan]
  guarded  : [0.       0.       0.231224 0.628532 0.140244]
  reference: [0.231224 0.628532 0.140244] (the last three entries)
  all -inf row, guarded: [0.0, 0.0, 0.0] (our convention: zeros)
```

The guard says: *if everything I have seen so far is masked, there is nothing to accumulate; skip the update.*

Two design decisions to note:

- Plain causal attention never has a row that starts masked (each row can always see its own diagonal element), but the tiles inside a kernel can, and so can padding and sliding-window patterns. We build the guard in now.
- A **fully** masked row has no meaningful softmax. We return zeros. That is our convention, not a law of nature: the plain softmax in Chapter 1 produced `NaN` for the same input.

---

## 2.4 Merging: combining the states of two chunks

Real kernels never walk a row one element at a time. They process it in **chunks** (tiles), and different threads handle different chunks. So we need to combine the `(m, d)` state of one chunk with the state of another.

```python
def merge(a, b):
    (ma, da), (mb, db) = a, b
    m = max(ma, mb)
    if m == NEG_INF:
        return (NEG_INF, 0.0)
    return (m, da * math.exp(ma - m) + db * math.exp(mb - m))
```

It is the same repair as before, applied to both sides: convert each sum to the new common max, then add.

### Why the order does not matter

The state `(m, d)` stands for the quantity `d · exp(m)`, stored in a scaled form so it cannot overflow. And `merge` computes `(da·e^{ma} + db·e^{mb})` in that same scaled form:

```
merge(a, b) represents:   d · e^m  =  da · e^{ma}  +  db · e^{mb}
```

Addition is associative and commutative, so merging is too, in exact arithmetic **[derived]**. (Side note: `m + ln(d)` is the log-sum-exp of the elements. FlashAttention saves exactly this number per row for the backward pass, as we saw in Chapter 1.)

### Checking it numerically

**[ran]** A row of 1,000 random scores processed in blocks of different sizes, each block reduced with `push`, then the block states combined with `merge`. Relative difference from the plain sequential result:

```
  block=   1  m=14.001926  d=2.3858819856  |d - d_sequential|/d = 0.00e+00
  block=   2  m=14.001926  d=2.3858819856  |d - d_sequential|/d = 3.72e-16
  block=   3  m=14.001926  d=2.3858819856  |d - d_sequential|/d = 7.45e-16
  block=   7  m=14.001926  d=2.3858819856  |d - d_sequential|/d = 1.86e-16
  block=  64  m=14.001926  d=2.3858819856  |d - d_sequential|/d = 5.58e-16
  block= 500  m=14.001926  d=2.3858819856  |d - d_sequential|/d = 1.86e-16
  block=1000  m=14.001926  d=2.3858819856  |d - d_sequential|/d = 0.00e+00
  200 random merge trees over 100 chunks: worst relative error in d = 7.45e-16
```

Block size 1 (merge every element) and block size 1,000 (no merging at all) agree. So do 200 random binary merge trees. In floating point the results differ in the last bits (rounding is not associative), but the mathematics does not depend on the order.

This one property gives us two things later:

- **Chapter 3:** loop over tiles of a row, merging each tile's state into the running state.
- **Chapter 7:** many threads each hold a partial `(m, d)` and combine them with warp shuffles.

---

## 2.5 One pass for what attention actually needs

Attention does not need the softmax **vector**. It needs the weighted sum `O = Σⱼ softmax(s)ⱼ · vⱼ`. That lets us go from two passes to one.

Keep a third running quantity, the **unnormalised output**:

```
o_j = Σ_{k ≤ j} exp(s_k − m_j) · v_k
```

Exactly the same algebra as for `d` gives

```
o_j = o_{j−1} · exp(m_{j−1} − m_j)  +  exp(s_j − m_j) · v_j
```

and at the end `O = o_N / d_N`. The rescale factor multiplies both accumulators, `d` and `o`. When `v_j` is a vector of length `d_head`, the same factor is applied to every component.

```python
import numpy as np

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
```

A design choice: **we divide by `d` once, at the end.** The original FlashAttention paper's Algorithm 1 instead normalises the output at every step **[source]**. The two are mathematically identical. We choose the deferred version because it is simpler and does less work per step. Chapter 20 revisits what the FlashAttention-2 kernel does.

**[ran]** Against the three-pass definition (`weighted_sum_reference`: softmax, then dot with `V`), 200 random rows of `N = 64` scores with vector values of length 8, and every fourth row starting with five masked entries:

```
  worst |one-pass - reference| over 200 rows: 2.00e-15
```

### Where the passes went

| Computation | Passes over the scores | Memory accesses per element |
|---|---|---|
| Safe softmax (vector output) | 3 | 4 **[source]** |
| Online softmax (vector output) | 2 | 3 **[source]** |
| Online softmax-weighted sum (what attention needs) | **1** | reads `s` and `v` once each **[derived]** |

The last row is the payoff. If the scores `s_j` are produced tile by tile and consumed immediately, they never need to be stored at all. That is the whole idea of FlashAttention, and Chapter 3 assembles it from these pieces.

---

## 2.6 What it costs, honestly

**Extra arithmetic.** Each online update computes two exponentials (the rescale factor and the new term) where the 3-pass version computes one per pass. The paper describes the additional cost as negligible, two extra operations per element **[source]**.

**How often is the rescale actually needed?** Only when the max changes. For random data that is rare: the expected number of times a running maximum updates over `N` independent values is the harmonic number `H_N ≈ ln N + 0.58` **[math fact]**. **[ran]** simulation (200 random rows per size):

```
  N=   16: mean max-updates per row =  3.38   (harmonic number H_N =  3.38)
  N=  256: mean max-updates per row =  6.08   (harmonic number H_N =  6.12)
  N= 4096: mean max-updates per row =  9.14   (harmonic number H_N =  8.90)
  increasing input, N=256: max changes on 256 of 256 elements (worst case)
```

So for typical inputs the rescale factor is exactly 1 almost all the time, but the worst case changes it on every element. Whether it is worth skipping the extra `exp` when the max is unchanged is a GPU question (all threads in a warp execute together), which we return to in Chapter 12. Here it is only worth knowing.

**Speed of standalone softmax.** The paper reports softmax up to 1.3× faster from the reduced memory traffic **[source]**. That is modest. A third-party JAX re-implementation, tested on TPUv2 and a T4, reported that neither of its online variants was reliably faster than the compiler-optimised baseline, and observed numerical stability problems for its fused softmax-plus-dot variant at large sizes **[source: repository README, anecdotal]**. So the honest summary is: online softmax on its own is a small win at best. Its real value is that it is **incremental and mergeable**, which lets us fuse softmax with the two matrix multiplies and never write the `N × N` matrix (Chapter 1's argument).

**Float32 accuracy.** Do the two algorithms agree exactly in float32? Not always. **[ran]** (`probe_float32.cpp`, 5 seeds per size, scores uniform in `[−8, 8]`): the largest relative disagreement between the 3-pass `d` and the online `d` was

```
n = 16 :  1.02e-07        n = 256  : 1.48e-07
n = 4096: 7.02e-07        n = 65536: 3.26e-06
```

and for several individual (seed, `n`) pairs it was exactly zero. Note that this measures how much the two methods disagree with each other, not which is closer to the truth. Both accumulate `n` float32 additions sequentially, and that accumulated rounding error grows with `n` whichever method you use. It is one reason later kernels accumulate in fp32 even when the inputs are fp16.

---

## 2.7 Your first C++ file

Everything above is now translated to C++. The program is `online_softmax.cpp`. I will show it in pieces, with a **C++ decoded** note after each piece for whatever is new.

### The Python-to-C++ dictionary for this chapter

| Python | C++ |
|---|---|
| `import math` | `#include <cmath>` |
| `math.exp(x)` | `expf(x)` (the `float` version) |
| `float("-inf")` | `-INFINITY` |
| `for i in range(n):` | `for (size_t i = 0; i < n; ++i) { ... }` |
| `for v in x:` | `for (float v : x) { ... }` |
| `max(a, b)` | `std::max(a, b)` |
| `x = [0.0] * n` | `std::vector<float> x(n, 0.0f);` |
| `len(x)` | `x.size()` |
| `def f(x): return ...` | `float f(const std::vector<float>& x) { return ...; }` |
| `(m, d)` tuple | `struct MD { float m; float d; };` |
| `print(f"{v:.7f}")` | `printf("%.7f\n", v);` |
| `a // b` on ints | `a / b` on ints (**different for negatives**, see below) |

### Building and running

```bash
g++ -std=c++17 -O2 -Wall -Wextra -o online_softmax online_softmax.cpp
./online_softmax
```

**C++ decoded: the compile step.** Python runs your source directly. C++ is **compiled** first: `g++` reads the source and produces a machine-code executable, then you run that. Consequences you will feel immediately: type errors and typos are reported before the program runs at all, and nothing happens until compilation succeeds. The flags: `-std=c++17` picks the language version, `-O2` turns on optimisation, `-Wall -Wextra` turn on warnings (always use them: several of the bugs below only show up as warnings). CUDA's `nvcc` compiler is a wrapper around this same process.

### Piece 1: headers and a random-number helper

```cpp
#include <algorithm>   // std::max, std::min
#include <cmath>       // expf, INFINITY, fabsf
#include <cstdint>     // uint32_t
#include <cstdio>      // printf
#include <vector>      // std::vector

void lcg_fill(std::vector<float>& x, uint32_t seed, float scale) {
    uint32_t state = seed;
    for (size_t i = 0; i < x.size(); ++i) {
        state = 1664525u * state + 1013904223u;                    // wraps modulo 2^32
        float u = static_cast<float>(state >> 8) / 16777216.0f;    // 24-bit integer / 2^24: exact
        x[i] = (2.0f * u - 1.0f) * scale;
    }
}
```

`lcg_fill` is a tiny linear congruential generator. We use it instead of a library random function so that Python (`check_cpp.py`) can produce the **identical** numbers and we can compare the two languages fairly.

**C++ decoded:**

- **`#include <vector>`** is closest to `import`, but it works by pasting the header's text into your file before compilation. Standard library names live in the `std` namespace, so you write `std::vector`, `std::max`. If you forget, you get this error **[ran]**:

```
e2.cpp:3:5: error: 'vector' was not declared in this scope
    3 |     vector<float> x(3);
      |     ^~~~~~
```

- **Every variable has a type, written before its name.** `uint32_t state = seed;` declares an unsigned 32-bit integer. Every function declares its parameter and return types. Python's `def lcg_fill(x, seed, scale)` becomes `void lcg_fill(std::vector<float>& x, uint32_t seed, float scale)`; `void` means "returns nothing".
- **`float` vs `double`.** `float` is 32 bits; `double` is 64 bits. **A Python `float` is a C `double`.** The suffix matters: `0.1f` is a float literal, `0.1` is a double **[ran]**:

```
0.1f stored exactly as 0.10000000149011612
0.1  stored exactly as 0.10000000000000001
```

  Everything in a CUDA attention kernel is 32-bit or narrower, so the suffix `f` will appear constantly.
- **Floats run out of integers at 2²⁴.** **[ran]** `16777216.0f + 1.0f` prints `16777216.0`: a `float` has a 24-bit mantissa, so above 16,777,216 it cannot represent every integer. This is why we build `u` from a 24-bit integer: it is exact.
- **`uint32_t`** is an integer with an exact width (32 bits, unsigned). For unsigned types, arithmetic that overflows **wraps around modulo 2³²** by definition **[language rule]**, and here we *want* that wrap: it is the generator. In Python the integers never overflow, so `check_cpp.py` masks with `& 0xFFFFFFFF` to reproduce the wrap.
- **`size_t`** is the unsigned integer type for sizes and indices (`x.size()` returns one). Being unsigned, it wraps instead of going negative **[ran]**:

```
size_t 0 - 1 = 18446744073709551615
```

  Also with `-Wall -Wextra`, comparing it to a plain `int` warns **[ran]**:

```
w4.cpp:4:23: warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<float>::size_type' {aka 'long unsigned int'} [-Wsign-compare]
    4 |     for (int i = 0; i < x.size(); ++i) t += x[i];
```

  So loop indices over a vector are `size_t`.
- **`static_cast<float>(state >> 8)`** is an explicit type conversion (Python: `float(...)`). `>> 8` shifts the bits right by 8 (integer divide by 256), leaving a 24-bit number.
- **`std::vector<float>&`** in the parameter list: the `&` makes `x` a **reference**, meaning `x` is the caller's vector, not a copy. Writing `x[i] = ...` therefore changes the caller's data, like mutating a Python list passed into a function. More on references in Piece 2.
- **`++i`** adds one to `i`. **`;`** ends every statement, and `{ }` delimit blocks (indentation means nothing to the compiler). A missing semicolon gives **[ran]**:

```
e1.cpp:4:5: error: expected ',' or ';' before 'printf'
    4 |     printf("%f\n", x);
```

  Note that the error is reported on the line *after* the mistake.

### Piece 2: the three-pass softmax

```cpp
std::vector<float> softmax_3pass(const std::vector<float>& x) {
    const size_t n = x.size();

    float m = -INFINITY;                       // pass 1: maximum
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, x[i]);
    }

    float d = 0.0f;                            // pass 2: normaliser
    for (size_t i = 0; i < n; ++i) {
        d += expf(x[i] - m);
    }

    std::vector<float> y(n);                   // pass 3: outputs
    for (size_t i = 0; i < n; ++i) {
        y[i] = expf(x[i] - m) / d;
    }
    return y;
}
```

**C++ decoded:**

- **`for (size_t i = 0; i < n; ++i)`** has three parts separated by semicolons: *initialise* (`size_t i = 0`), *keep going while* (`i < n`), *after each iteration* (`++i`). It is `for i in range(n)` spelled out.
- **`const std::vector<float>& x`**: read this right to left. `x` is a **reference** (`&`, an alias, not a copy) to a **`vector<float>`** that is **`const`** (read-only). This is the standard way to pass big read-only data.
  - *Why it matters:* in Python, passing a list never copies it. **In C++ the default is to copy.** Passing `std::vector<float> x` by value would copy every element on every call. **[ran]** `copycost.cpp`: 200 calls with a 4 MiB vector took about 68 to 69 ms by value and about 0.06 to 0.12 ms by `const&` (three runs). The benchmark is deliberately lopsided, since the function does almost no work, but it shows the copy dominating.
  - *The `const` is enforced.* Writing through it is a compile error **[ran]**:

```
e3.cpp:3:10: error: assignment of read-only location '(& x)->std::vector<float>::operator[](0)'
    3 |     x[0] = 2.0f * x[0];
```

- **`const size_t n`** is a constant: assigned once, never changed.
- **`std::max(m, x[i])`** needs both arguments to have the **same type**. `std::max(m, 0)` with a `float m` does not compile **[ran]**:

```
mx.cpp:3:47: error: no matching function for call to 'max(float&, int)'
```

  Write `0.0f`.
- **`expf`** is the single-precision exponential. There is also `exp` for doubles. We use the `f` versions so everything stays 32-bit (in CUDA device code, too; Chapter 12).
- **`-INFINITY`** is a macro from `<cmath>` that means `float` infinity. (Python: `float("-inf")`.)
- **`std::vector<float> y(n)`** creates a vector of `n` floats, all zero **[ran]** (`vector<float>(3) = 0 0 0`). It is like `[0.0] * n`.
- **`return y;`** returns the vector by value. Compilers avoid the copy for a returned local variable, so this is fine **[language rule]**; I did not measure it here.

### Piece 3: the running state as a `struct`, and the online update

```cpp
struct MD {
    float m;   // running maximum
    float d;   // running sum of exp(value - m)
};

// Fold one more score into the running state.
MD push(MD s, float x) {
    float m_new = std::max(s.m, x);
    if (m_new == -INFINITY) {                  // everything so far is masked
        return s;
    }
    return MD{m_new, s.d * expf(s.m - m_new) + expf(x - m_new)};
}

MD online_stats(const std::vector<float>& x) {
    MD s{-INFINITY, 0.0f};
    for (float xi : x) {                       // range-based for: like `for xi in x`
        s = push(s, xi);
    }
    return s;
}
```

**C++ decoded:**

- **`struct MD { float m; float d; };`** defines a new type with two named fields. Think of it as a `namedtuple` or a `@dataclass` with fixed types. Note the **semicolon after the closing brace**; forgetting it is a classic error.
- **`MD s{-INFINITY, 0.0f};`** and **`MD{m_new, ...}`** build an `MD` with the fields in declaration order (brace initialisation). Python: `MD(-inf, 0.0)`.
- **`s.m`** reads a field, just like Python.
- **`push(MD s, float x)` takes `s` by value**, so it receives a copy. For an 8-byte struct that is exactly what you want (cheap), and it means `push` cannot change the caller's `s`; it *returns* the new state, like the Python `m, d = push(m, d, x)`.
- **`for (float xi : x)`** is the range-based `for`. It copies each element into `xi`. For a `vector<float>` that is fine.
- **The `s` structs will become registers.** Two floats per row is the entire running state. In a CUDA kernel this will live in registers next to each thread's work (Chapter 12).

### Piece 4: outputs, merge and blocks

```cpp
std::vector<float> softmax_online(const std::vector<float>& x) {
    MD s = online_stats(x);
    std::vector<float> y(x.size(), 0.0f);      // all zeros: also our answer for a fully masked row
    if (s.d == 0.0f) {
        return y;
    }
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = expf(x[i] - s.m) / s.d;
    }
    return y;
}

MD merge(MD a, MD b) {
    float m = std::max(a.m, b.m);
    if (m == -INFINITY) {
        return MD{-INFINITY, 0.0f};
    }
    return MD{m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

MD stats_blocked(const std::vector<float>& x, size_t block) {
    MD total{-INFINITY, 0.0f};
    for (size_t start = 0; start < x.size(); start += block) {
        size_t end = std::min(start + block, x.size());
        MD part{-INFINITY, 0.0f};
        for (size_t i = start; i < end; ++i) {
            part = push(part, x[i]);
        }
        total = merge(total, part);
    }
    return total;
}
```

**C++ decoded:**

- **`std::vector<float> y(x.size(), 0.0f)`**: `n` copies of `0.0f`. Python: `[0.0] * len(x)`.
- **`start += block`** in the `for` header: a loop that steps by `block`. Python: `range(0, n, block)`.
- **`std::min(start + block, x.size())`** clamps the last, possibly shorter, block. Both arguments are `size_t`, so it compiles.
- **`==` on floats.** `m == -INFINITY` is safe here because infinity is an exact special value. Comparing computed floats with `==` in general is a bug waiting to happen; we only ever compare against exact sentinels (`-INFINITY`, `0.0f` for a sum that was never touched).
- **Indexing is unchecked.** `x[i]` does not test `i` against the size **[language rule]**. **[ran]** reading one past the end:

```
--- plain build ---
x[4] = 0.000000            (no error; exit code 0)
--- with -fsanitize=address ---
ERROR: AddressSanitizer: heap-buffer-overflow ... READ of size 4 ... oob.cpp:5
--- using x.at(4) instead of x[4] ---
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 4) >= this->size() (which is 4)
```

  The plain build silently printed a value that belongs to nothing. Python would have raised `IndexError`. So learn the habit now: when a C++ result looks wrong, rebuild with `-fsanitize=address` (or use `.at()`) before suspecting the algorithm. GPU kernels are equally unchecked by default (I have not tested that here), and they have their own tool for this, which we meet later.

### Piece 5: the one-pass weighted average

```cpp
float weighted_average_online(const std::vector<float>& s, const std::vector<float>& v) {
    float m = -INFINITY;
    float d = 0.0f;
    float o = 0.0f;                            // running UNNORMALISED output
    for (size_t j = 0; j < s.size(); ++j) {
        float m_new = std::max(m, s[j]);
        if (m_new == -INFINITY) {
            continue;
        }
        float scale = expf(m - m_new);         // shrink everything accumulated so far
        float w = expf(s[j] - m_new);          // weight of the new element
        d = d * scale + w;
        o = o * scale + w * v[j];
        m = m_new;
    }
    return d == 0.0f ? 0.0f : o / d;           // normalise once, at the end
}
```

**C++ decoded:**

- **`continue`** skips to the next loop iteration, same as Python.
- **`cond ? a : b`** is the ternary operator; Python: `a if cond else b`.
- **Here `v[j]` is a single number.** This keeps the C++ free of 2-D indexing for now. When each `v_j` is a vector of length `d_head`, the same `scale` multiplies every component of `o`. Doing that with flat, row-major arrays and pointers is the C++ topic of Chapter 3.
- **Variables must be initialised.** `float o = 0.0f;` matters. If you write `float d;` and then `d += v`, you use whatever bytes happened to be in that memory. GCC catches this with `-Wall` **[ran]**:

```
w5.cpp:5:11: warning: 'd' is used uninitialized [-Wuninitialized]
    5 |     float d;
```

  And here is the trap: the same program, built without the warnings, printed `3.000000` on three runs in a row. It happened to work. Python would have raised a `NameError`; C++ gives an answer that is right until it is not.

### Piece 6: `main`

```cpp
int main() {
    const size_t n = 16;
    std::vector<float> x(n);
    lcg_fill(x, 12345u, 4.0f);

    std::vector<float> y3 = softmax_3pass(x);
    std::vector<float> y2 = softmax_online(x);

    float max_diff = 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, fabsf(y3[i] - y2[i]));
        sum += y2[i];
        printf("y3 %zu %.7f\n", i, y3[i]);
        printf("y2 %zu %.7f\n", i, y2[i]);
    }
    printf("max |3pass - online| = %.3e   sum(online) = %.7f\n", max_diff, sum);

    MD ref = online_stats(x);
    for (size_t block : {1, 3, 5, 16}) {
        MD s = stats_blocked(x, block);
        printf("blocked block=%2zu  m=%.7f  d=%.7f  |d - d_seq| = %.2e\n",
               block, s.m, s.d, fabsf(s.d - ref.d));
    }

    std::vector<float> v(n);
    lcg_fill(v, 777u, 2.0f);
    float a = weighted_average_online(x, v);
    float b = weighted_average_reference(x, v);
    printf("weighted average: one-pass = %.7f  reference = %.7f\n", a, b);

    std::vector<float> masked = x;
    masked[0] = masked[1] = masked[2] = -INFINITY;     // a row that starts masked
    float c = weighted_average_online(masked, v);
    std::vector<float> tail_s(masked.begin() + 3, masked.end());
    std::vector<float> tail_v(v.begin() + 3, v.end());
    float e = weighted_average_reference(tail_s, tail_v);
    printf("masked start:     one-pass = %.7f  reference = %.7f\n", c, e);

    // A longer row: now the two algorithms round differently.
    std::vector<float> big(4096);
    lcg_fill(big, 1u, 8.0f);
    std::vector<float> b3 = softmax_3pass(big);
    std::vector<float> b2 = softmax_online(big);
    float big_diff = 0.0f;
    float big_sum3 = 0.0f;
    float big_sum2 = 0.0f;
    for (size_t i = 0; i < big.size(); ++i) {
        big_diff = std::max(big_diff, fabsf(b3[i] - b2[i]));
        big_sum3 += b3[i];
        big_sum2 += b2[i];
    }
    printf("n=4096: max |3pass - online| = %.3e   sum(3pass) = %.6f   sum(online) = %.6f\n",
           big_diff, big_sum3, big_sum2);
    return 0;
}
```

(`weighted_average_reference` is in the file; it is the 3-pass softmax followed by a dot product.)

**C++ decoded:**

- **`int main()`** is where execution starts; `return 0;` means success. There is no `if __name__ == "__main__":`.
- **`printf`** is C-style formatted output. `%.7f` is a float with 7 decimals, `%.3e` is scientific notation, `%zu` is the format for a `size_t`, `%2zu` pads it to width 2, and `\n` is a newline (`printf` does not add one). Floats are widened to doubles when passed to `printf`; that is expected. The format string is not checked against your arguments by the language, though `-Wall` will often warn.
- **`std::vector<float> masked = x;`** **copies** the whole vector (assignment copies in C++; in Python it would alias). We want a modified copy here, so that is correct.
- **`masked[0] = masked[1] = masked[2] = -INFINITY;`** chained assignment, same as Python.
- **`masked.begin() + 3`** is an *iterator*: a position inside the vector. `std::vector<float>(a, b)` builds a vector from the range `[a, b)`, so `tail_s` is `masked[3:]`. Python: `masked[3:]`.
- **`for (size_t block : {1, 3, 5, 16})`** loops over a braced list of values.
- **Integer division.** Not used in the program, but you will hit it: in C++ dividing two integers gives an integer, truncated **toward zero**. **[ran]**:

```
7 / 2      = 3
-7 / 2     = -3   (Python: -7 // 2 = -4)
-7 % 3     = -1   (Python: -7 % 3 = 2)
7 / 2.0f   = 3.5
(float)7/2 = 3.5
1 / 2   = 0          (from pitfalls.cpp)
1 / 2.0 = 0.5
```

  Python's `//` and `%` round down; C++'s `/` and `%` on negative integers do not. Index arithmetic in kernels leans on this.

---

## 2.8 Running it, and checking C++ against Python

**[ran]** Compile with `-Wall -Wextra`: no warnings. The program first prints the 16 values of each softmax vector (32 lines, omitted here); the summary lines that follow are:

```
max |3pass - online| = 0.000e+00   sum(online) = 0.9999999
blocked block= 1  m=3.7264748  d=2.3972869  |d - d_seq| = 0.00e+00
blocked block= 3  m=3.7264748  d=2.3972869  |d - d_seq| = 0.00e+00
blocked block= 5  m=3.7264748  d=2.3972869  |d - d_seq| = 0.00e+00
blocked block=16  m=3.7264748  d=2.3972869  |d - d_seq| = 0.00e+00
weighted average: one-pass = -0.0648239  reference = -0.0648239
masked start:     one-pass = -0.0854902  reference = -0.0854902
n=4096: max |3pass - online| = 2.794e-09   sum(3pass) = 1.000001   sum(online) = 1.000001
```

For 16 elements the two float32 algorithms agree to all printed digits. For 4,096 elements they differ in the ninth decimal.

`check_cpp.py` compiles the program, runs it, regenerates the same inputs in Python (the scale factors 4, 2 and 8 are powers of two, so the inputs are bit-identical in both languages) and compares with the float64 Python implementations. **[ran]**:

```
C++ float32 vs Python float64 (same inputs)
  softmax, 3-pass : max abs diff = 5.63e-08
  softmax, online : max abs diff = 5.63e-08
  blocked block= 1: |m - m64| = 3.8e-08, |d - d64| = 1.9e-07
  ...
  weighted average: |C++ - Python| = 2.61e-09
  masked start    : |C++ - Python| = 7.25e-08
```

Read the size of these numbers correctly: the C++ program prints 7 decimals, so a difference of about `5e-8` is just the rounding of the printed text (half a unit in the seventh decimal). The C++ and the Python implement the same algorithm and agree to the precision that was printed.

---

## 2.9 A hazard: fast-math

Compilers have a flag that allows aggressive floating-point rewrites: `-ffast-math`. One of the things it permits is assuming that no value is ever infinite or NaN **[documented GCC behaviour, not re-checked here]**. Our guard is all about `−INFINITY`. **[ran]** The same source built with `-O2 -ffast-math`:

```
weighted average: one-pass = -0.0648239  reference = -0.0648239
masked start:     one-pass = -nan  reference = -0.0854901
```

Normal build: the masked-start case gives `-0.0854902`. With `-ffast-math`: `-nan`. The unmasked case is unaffected. The behaviour is consistent with the compiler having stopped honouring the `−INFINITY` comparison; I have not inspected the generated code to say which instruction changed.

CUDA has a similarly named option (`--use_fast_math`). I have not tested what it does to this algorithm. Chapter 12 tests it before we rely on anything. Until then: **assume that a build option with "fast math" in the name can break masking.**

---

## 2.10 How this maps onto the GPU (a preview)

| This chapter | Where it goes |
|---|---|
| `(m, d)` per row | Two registers per query row inside the kernel (Chapters 12 and 13). |
| `push` | The inner-loop update as each new score arrives (Chapter 12). |
| `merge` | Combining tiles (Chapter 3) and combining partial states from different threads (Chapter 7). |
| The `o` accumulator, rescaled by `scale` | The output accumulator that is rescaled each time a new key/value tile changes the max (Chapter 13). |
| Divide by `d` once at the end | The final normalisation before writing `O` back. |
| `m + ln(d)` | The logsumexp `L` saved for the backward pass (Chapter 13). |

`expf` is also the name of the single-precision exponential in CUDA device code, and CUDA has a faster, less accurate variant. Both are Chapter 12 material; I have not verified their exact behaviour here.

---

## 2.11 Exercises

1. **By hand, then with code.** Trace the online update on `[2, 7, 7, 1]`. Note the tie at `j = 2`: what is the rescale factor? Check yourself with `online_stats(..., trace=True)` (the `trace` argument is in the file's version of `online_stats`).
2. **Prove `merge` is associative.** Use the "`d · e^m` is additive" argument from §2.4, or grind out the algebra. Then write a test that splits a random row into 3 chunks in every possible order.
3. **Skip the extra exponential.** Modify `push` so that when `x <= m` it computes only `exp(x − m)`, and when `x > m` it computes only the rescale factor and adds 1. Count `exp` calls for random and increasing rows, confirming the outputs match. (The counts should track §2.6.)
4. **Predict, then compile.** Without running it, what does each print in C++: `7 / 2`, `-7 / 2`, `7 / 2.0f`, `1 / 2 * 3.0`, and `size_t a = 3; size_t b = 5; a - b`? Check with a small program. (`div.cpp` and `pitfalls.cpp` cover most of these.)
5. **Break it on purpose.** Remove the guard (`if (m_new == -INFINITY)`) from the C++ `push` and run the masked-start test. Then put the guard back and rebuild with `-ffast-math`. Which failure did you expect, and which did you get?
6. **Sanitise.** Add an off-by-one to one of the loops (`i <= n`) and find it with `-fsanitize=address`. Then find it again with `.at()`.
7. **Design question.** A fully masked row returns zeros here, and `NaN` in naive softmax. What should attention output for a query that can see nothing? Which behaviour would you want when debugging, and which when training?

---

## 2.12 Common pitfalls

- **Forgetting the guard for `-inf`**, and getting NaN from `exp(−inf − (−inf))` (§2.3).
- **Forgetting the rescale of the *old* sum or output.** The most common bug when writing this by hand: update `m`, then add the new term to the old `d` without multiplying `d` by `exp(m_old − m_new)`. The result is right only when the max never changes, so tests on small or sorted-descending inputs pass.
- **Using the new max before computing the rescale factor.** Compute `exp(m_old − m_new)` while both values are still available; do not overwrite `m` first.
- **Integer division and truncation** (§2.7, Piece 6).
- **`float` vs `double` literals** (`0.1f` vs `0.1`).
- **Unsigned wrap-around** with `size_t`, especially in loops that count downward: `for (size_t i = n - 1; i >= 0; --i)` never terminates, because `i >= 0` is always true for an unsigned value **[language rule]**.
- **Uninitialised variables** (§2.7, Piece 5) and **unchecked indexing** (Piece 4). Build with `-Wall -Wextra` always.
- **Passing big containers by value** (Piece 2).
- **Fast-math** (§2.9).

---

## 2.13 Summary and bridge to Chapter 3

- Softmax needs the row max, which seems to force a full pass first. The recurrence `d_j = d_{j−1}·exp(m_{j−1} − m_j) + exp(x_j − m_j)` removes that requirement: the sum is *repaired* whenever the max changes.
- Safe softmax makes 3 passes and 4 memory accesses per element; online softmax makes 2 and 3 **[source]**. Softmax-weighted sums, which is what attention needs, take **one** pass.
- The `(m, d)` state is **mergeable**: any chunking and any merge order give the same answer (§2.4). This is the property that makes tiling and parallel reduction correct.
- On its own, online softmax is a modest speed-up. Its power is composability with the two matrix multiplies.
- In C++ you met: compilation, `#include`, `std::`, typed variables, `float`/`double`/`size_t`/`uint32_t`, `for` loops, references (`&`) and `const`, `struct`, `std::vector`, `std::max`, `expf`, `printf`, and the three big differences from Python: values are copied by default, integer division truncates, and nothing checks your indices.

**Next: Chapter 3, tiled attention on the CPU.** We put the pieces together: split `Q`, `K`, `V` into tiles, run the online update across tiles for every query row, and produce the full attention output without ever building the `N × N` matrix, first in NumPy and then in C++. That is where pointers, row-major flat indexing, and the `o` accumulator for vector values come in.

---

## 2.14 Sources and verification status

### Verification status

| Item | Status |
|---|---|
| Python implementations (`online_softmax.py`): trace, awkward-input table, guard demo, merge tests, max-update counts, vector one-pass test | **Executed.** Outputs pasted above. |
| C++ program (`online_softmax.cpp`): compiled with GCC 13.3, `-std=c++17 -O2 -Wall -Wextra`, no warnings; output pasted above | **Executed.** |
| C++ vs Python cross-check (`check_cpp.py`) | **Executed.** |
| Float32 disagreement probe (`probe_float32.cpp`) | **Executed** (5 seeds × 4 sizes). |
| Small C++ demos: `pitfalls.cpp`, `div.cpp`, `copycost.cpp`, `oob.cpp`, and the compile-error and warning snippets | **Executed.** The error snippets were compiled separately; only the messages are shown. |
| `-ffast-math` effect on the masked case | **Executed.** The *reason* was not investigated. |
| GPU behaviour of anything in this chapter | **Not tested.** Nothing here runs on a GPU; §2.10 is a preview. |
| Paper claims (4 vs 3 accesses per element, up to 1.3×, Algorithm 3, Algorithm 1's per-step normalisation) | **Read from the papers.** They describe the papers' setups, not your machines. |

### Known loose ends

- The **1.3×** speed-up is the paper's headline figure for its own benchmarks; I did not reproduce it and did not identify the hardware it was measured on.
- The JAX repository's observations (no reliable speed-up, instability for a fused variant) are one person's experiment, reported in a README. I mention them as a caution, not as a finding.
- The `copycost.cpp` benchmark uses a few C++ features not yet covered (a lambda and `<chrono>`). Treat it as a curiosity. An earlier version of it let the compiler reuse repeated identical calls and reported an implausibly small time for the `const&` case, so the file now passes a changing `start` argument to prevent that.
- `-ffast-math` semantics are stated from general knowledge of GCC, and I did not re-read its documentation for this chapter.

### Sources

- Milakov, Gimelshein. *Online normalizer calculation for softmax.* arXiv:1805.02867. https://arxiv.org/abs/1805.02867 (Algorithm 3, memory-access counts, speed-up claim)
- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention.* arXiv:2205.14135. https://arxiv.org/abs/2205.14135 (Algorithm 1, per-step output normalisation)
- jenkspt. *online-softmax-jax* (README). https://github.com/jenkspt/online-softmax-jax
