# Chapter 12: Attention Mechanisms

## Introduction

Chapter 11 processed sequences one step at a time, carrying information forward through a hidden state — a design with an inherent bottleneck: everything the model knows about a long sequence has to be compressed into that single fixed-size hidden state before it can influence a distant future timestep. **Attention** solves this differently: instead of compressing history into a running summary, it lets every position in a sequence look directly at every other position and decide, dynamically, which ones are relevant.

This chapter builds scaled dot-product attention entirely from scratch — no black-box `nn.MultiheadAttention` yet — so you understand precisely what's happening inside it before Chapter 13 assembles full transformer blocks out of it.

By the end of this chapter you'll be able to:
- Explain queries, keys, and values, and what the attention computation does conceptually
- Implement scaled dot-product attention from raw tensor operations
- Implement multi-head attention from scratch
- Apply masking for padding and causal (autoregressive) attention
- Understand why attention scales the way it does, and its computational trade-offs versus RNNs

---

## 12.1 The Core Idea: Queries, Keys, and Values

Attention reframes "which parts of the input are relevant to this position" as a soft, differentiable lookup, using three learned projections of the input:

- **Query (Q)**: "what am I looking for?" — one per output position.
- **Key (K)**: "what do I contain?" — one per input position, used to match against queries.
- **Value (V)**: "what do I actually contribute, if selected?" — one per input position, this is what actually gets returned.

The mechanism: for each query, compute a similarity score against every key (via a dot product), turn those scores into a probability distribution (via softmax), and use that distribution to compute a weighted sum of the values. Positions whose keys are highly similar to a given query contribute more to that query's output.

A helpful analogy: think of it like a soft database lookup — the query is your search term, the keys are indexed labels, the values are the actual data, and instead of retrieving a single exact match, you get a weighted blend of everything, weighted by how well each key matches your query.

---

## 12.2 Scaled Dot-Product Attention, Step by Step

The formula, from the original "Attention Is All You Need" paper:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's build this one operation at a time.

```python
import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)

seq_len, d_k, d_v = 4, 8, 8   # sequence length, key/query dim, value dim
Q = torch.rand(seq_len, d_k)
K = torch.rand(seq_len, d_k)
V = torch.rand(seq_len, d_v)
```

### Step 1: Similarity scores via dot product

```python
scores = Q @ K.T   # shape (seq_len, seq_len)
print(scores.shape)   # torch.Size([4, 4])
```

`scores[i, j]` is the dot-product similarity between query $i$ and key $j$ — how relevant position $j$ is to position $i$, before any normalization.

### Step 2: Scale by $\sqrt{d_k}$

```python
scaled_scores = scores / math.sqrt(d_k)
```

**Why scale at all?** As $d_k$ grows, dot products between random vectors tend to grow in magnitude too (the sum of more terms). Without scaling, these large values push the softmax in the next step into regions with extremely small gradients — the softmax becomes nearly one-hot, saturating and making training unstable. Dividing by $\sqrt{d_k}$ keeps the scores in a well-behaved range regardless of dimensionality, which is precisely the kind of numerical-stability concern you've likely encountered directly if you've debugged NaN losses during LLM fine-tuning.

### Step 3: Softmax to get attention weights

```python
attn_weights = F.softmax(scaled_scores, dim=-1)   # normalize across the "key" dimension
print(attn_weights.sum(dim=-1))   # tensor([1., 1., 1., 1.]) -- each row sums to 1
```

`dim=-1` is essential — softmax must normalize across the *key* dimension (so each query's attention weights form a valid probability distribution over all positions it could attend to), not across the query dimension. Getting this dimension wrong is a common and easy-to-miss bug: the shapes still work out, but the resulting weights mean something entirely different.

### Step 4: Weighted sum of values

```python
output = attn_weights @ V   # shape (seq_len, d_v)
print(output.shape)   # torch.Size([4, 8])
```

Each output position is now a weighted blend of all value vectors, weighted according to how relevant each corresponding key was to that position's query.

### Putting it together as one function

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))   # Section 12.4

    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V
    return output, attn_weights
```

Returning `attn_weights` alongside the output is common practice even outside of debugging — visualizing attention weights (e.g., as a heatmap) is one of the more interpretable windows into what a trained attention-based model is actually doing, compared to the largely opaque hidden states of an RNN.

---

## 12.3 Self-Attention vs. Cross-Attention

The example above is **self-attention**: Q, K, and V are all derived from the *same* input sequence — each position attends to (potentially) every other position in the same sequence. This is what powers the transformer encoder blocks in Chapter 13.

**Cross-attention** uses queries from one sequence and keys/values from a *different* sequence — for example, in a translation model, decoder queries attending to encoder keys/values from the source-language sequence. The mechanics (scaled dot-product attention itself) are identical; only where Q versus K/V come from differs:

```python
# Self-attention: Q, K, V all from the same source
x = torch.rand(seq_len, d_k)
output, _ = scaled_dot_product_attention(Q=x, K=x, V=x)

# Cross-attention: Q from one sequence, K/V from another
decoder_hidden = torch.rand(3, d_k)      # target sequence, length 3
encoder_output = torch.rand(seq_len, d_k)  # source sequence, length 4
output, _ = scaled_dot_product_attention(Q=decoder_hidden, K=encoder_output, V=encoder_output)
print(output.shape)   # torch.Size([3, 8]) -- one output per QUERY position (3), not per key position
```

Notice the output length matches the *query* sequence length, not the key/value sequence length — attention naturally handles Q and K/V sequences of different lengths, which is exactly what makes cross-attention useful for tasks relating two sequences of different sizes.

---

## 12.4 Masking

### Padding masks

Just as Chapter 11's `pack_padded_sequence` prevented an RNN from processing padding tokens, attention needs an explicit mechanism to prevent positions from attending to padding — since attention has no inherent sense of "sequence length," it will happily compute attention scores against padding positions unless told not to.

```python
seq_len = 5
scores = torch.rand(seq_len, seq_len)

# Suppose the last 2 positions are padding
padding_mask = torch.tensor([1, 1, 1, 0, 0])   # 1 = real token, 0 = padding
mask_2d = padding_mask.unsqueeze(0).expand(seq_len, seq_len)   # broadcast to (seq_len, seq_len)

masked_scores = scores.masked_fill(mask_2d == 0, float("-inf"))
attn_weights = F.softmax(masked_scores, dim=-1)
print(attn_weights[0])   # last two entries are ~0 -- padding contributes nothing
```

Setting masked positions to `-inf` *before* softmax (rather than zeroing the weights *after*) is essential: `softmax(-inf) = 0` exactly, cleanly excluding those positions from the probability distribution, while the remaining real positions' weights are correctly renormalized to still sum to 1. Zeroing after softmax would leave the excluded positions' probability mass "missing" rather than redistributed.

### Causal masks (for autoregressive models)

Language models that generate text one token at a time must not be allowed to attend to *future* tokens during training — otherwise the model could trivially "cheat" by looking ahead at the answer it's supposed to be predicting. A **causal mask** enforces this by masking out all positions after the current one:

```python
def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))   # lower-triangular matrix of 1s

mask = causal_mask(5)
print(mask)
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

`torch.tril` (lower triangular) zeroes out everything above the diagonal — row $i$ (query position $i$) can only attend to columns $0$ through $i$ (key positions up to and including itself), never anything later. This single mask is the entire mechanical difference between an encoder-style (bidirectional) and decoder-style (autoregressive/causal) transformer block, a distinction that becomes directly relevant in Chapter 13.

---

## 12.5 Multi-Head Attention

Rather than computing a single attention operation over the full-dimensional Q/K/V vectors, **multi-head attention** splits them into several smaller "heads," runs scaled dot-product attention independently in each, and concatenates the results. The intuition: different heads can learn to attend based on different notions of relevance simultaneously (e.g., one head tracking syntactic relationships, another tracking longer-range topical relevance) — a single attention operation is forced to blend all such notions into one shared weighting.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape

        Q = self.q_proj(x)   # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_head)
        Q = Q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ V   # (batch, num_heads, seq_len, d_head)

        # Merge heads back: (batch, num_heads, seq_len, d_head) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.out_proj(attn_output)

mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.rand(2, 10, 64)   # (batch=2, seq_len=10, d_model=64)
output = mha(x)
print(output.shape)   # torch.Size([2, 10, 64])
```

Two implementation details worth being deliberate about, both connecting directly back to Chapter 2:

- **`.transpose(1, 2)`** rearranges dimensions to group `(batch, num_heads)` together for the batched matrix multiply — this produces a non-contiguous tensor (Chapter 2, Section 2.4), which is why `.contiguous()` is required before the final `.view()` call when merging heads back.
- **`d_model % num_heads == 0`** is a hard requirement — the model dimension must divide evenly across heads, since each head operates on an equal-sized slice (`d_head = d_model // num_heads`).

### PyTorch's built-in `nn.MultiheadAttention`

Now that you understand the mechanics, in real projects you'll typically use PyTorch's built-in, more heavily optimized implementation rather than hand-rolling your own:

```python
mha_builtin = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

x = torch.rand(2, 10, 64)
output, attn_weights = mha_builtin(query=x, key=x, value=x)   # self-attention
print(output.shape)   # torch.Size([2, 10, 64])
```

`batch_first=True` matters here for the same consistency reason as `nn.LSTM` in Chapter 11. The built-in version also supports an `attn_mask` argument accepting exactly the kind of mask constructed in Section 12.4, plus additional optimizations (like fused kernels) that a hand-written implementation won't have — but understanding the from-scratch version is what makes debugging shape errors or unexpected attention behavior in the built-in version tractable.

---

## 12.6 Computational Trade-offs: Attention vs. Recurrence

Worth being explicit about, since it explains why attention-based architectures (transformers) have largely displaced RNNs for sequence modeling despite RNNs' conceptual elegance:

| | RNN/LSTM/GRU | Self-Attention |
|---|---|---|
| Per-step computation | Sequential — step $t$ depends on step $t-1$ | Parallel — all positions computed simultaneously |
| Path length between distant positions | $O(n)$ — information must flow through every intermediate step | $O(1)$ — any two positions connect directly |
| Memory/compute cost | $O(n)$ per sequence | $O(n^2)$ — every position attends to every other position |

Attention's $O(1)$ path length between any two positions is precisely what solves the long-range dependency problem from Section 11.2 far more directly than LSTM's gating mechanism does. But this comes at the cost of $O(n^2)$ memory and compute in sequence length — a real practical constraint you'll encounter directly when working with long-context LLMs, and the motivation behind numerous efficient-attention variants (not covered in this introductory chapter) developed specifically to address that quadratic cost.

---

## Summary

- Attention computes a weighted blend of values, where weights come from a softmax over scaled query-key dot-product similarities — a form of differentiable, soft lookup.
- Scaling by $\sqrt{d_k}$ prevents softmax saturation as dimensionality grows, keeping gradients well-behaved.
- Self-attention relates a sequence to itself; cross-attention relates queries from one sequence to keys/values from another — same mechanics, different sources.
- Masking (via `-inf` before softmax) excludes padding or future positions from consideration — causal masks are what make autoregressive language modeling possible.
- Multi-head attention runs several smaller attention operations in parallel, letting different heads specialize in different notions of relevance.
- Attention trades RNNs' sequential $O(n)$ computation and $O(n)$ long-range path length for parallel computation and $O(1)$ path length, at the cost of $O(n^2)$ memory/compute in sequence length.

## Exercises

1. Implement scaled dot-product attention without the scaling factor, and construct an example with `d_k=512` where you can observe the softmax becoming nearly one-hot (compare `attn_weights` with and without scaling).
2. Extend the causal mask function from Section 12.4 to also incorporate a padding mask, so a position is masked if it's either in the future *or* is a padding token.
3. Modify the `MultiHeadAttention` class from Section 12.5 to support cross-attention (separate `query_input` and `key_value_input` arguments), and verify it works when the two inputs have different sequence lengths.
4. Using the built-in `nn.MultiheadAttention`, pass a causal mask via the `attn_mask` argument (check the current PyTorch documentation for the expected mask format and dtype, since this differs from the boolean approach used in Section 12.4) and confirm attention weights are zero for future positions.

**Next:** Chapter 13 assembles multi-head attention, feedforward layers, and residual connections into full transformer encoder and decoder blocks using both `nn.TransformerEncoder` and a from-scratch mini transformer.
