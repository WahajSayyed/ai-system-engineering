# Chapter 13: Transformers in PyTorch

## Introduction

Chapter 12 built multi-head attention from scratch. This chapter assembles that attention mechanism, together with feedforward layers, residual connections, and normalization, into a complete **transformer block** — the architecture underlying essentially every modern large language model, including the Qwen2.5-Coder model you've fine-tuned. We'll build a mini transformer from scratch first, to see exactly how the pieces fit together, then use PyTorch's built-in `nn.TransformerEncoder` for a production-shaped version of the same thing.

By the end of this chapter you'll be able to:
- Explain why transformers need positional encoding, and implement it
- Assemble a complete transformer encoder block: attention, feedforward, residuals, and normalization
- Understand pre-norm vs. post-norm design and why it matters for training stability
- Use `nn.TransformerEncoder` and `nn.TransformerEncoderLayer` correctly
- Build a small transformer-based sequence classifier end to end

---

## 13.1 The Missing Piece: Positional Encoding

Attention (Chapter 12) computes weighted sums over a *set* of positions — nothing in the scaled dot-product attention computation itself depends on the *order* of those positions. If you shuffled the input sequence, self-attention would produce the same set of outputs, just correspondingly shuffled — attention is inherently **permutation-invariant**. This is a real problem for language: "the dog bit the man" and "the man bit the dog" contain the same tokens in a different order, with very different meanings.

**Positional encoding** solves this by injecting information about each position directly into the input embeddings, before attention ever sees them. The original transformer paper used a fixed sinusoidal encoding:

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions: sine
        pe[:, 1::2] = torch.cos(position * div_term)    # odd dimensions: cosine
        self.register_buffer("pe", pe.unsqueeze(0))      # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

pos_enc = PositionalEncoding(d_model=64)
x = torch.rand(2, 10, 64)
x_with_pos = pos_enc(x)
print(x_with_pos.shape)   # torch.Size([2, 10, 64]) -- same shape, position info added
```

A few details worth noting:

- **`register_buffer`**, not `nn.Parameter` (Chapter 5). The positional encoding is fixed, computed once from a formula — it's not learned, so it shouldn't appear in `.parameters()` or receive gradient updates. But it still needs to move with the model when you call `.to(device)`, and be saved/loaded with `state_dict()` — exactly what `register_buffer` provides, distinct from both a plain attribute (which `.to(device)` wouldn't touch) and `nn.Parameter` (which would incorrectly make it trainable).
- The sine/cosine pattern at different frequencies gives each position a unique, smoothly-varying signature, with a useful mathematical property: the encoding for position $p+k$ can be expressed as a linear function of the encoding for position $p$, which was hypothesized to make it easier for the model to learn to attend based on *relative* position.

### Learned positional embeddings — the more common modern approach

Many modern transformers, especially LLMs, instead use a simple learned embedding table for positions, exactly analogous to a token embedding:

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return x + self.pos_embedding(positions)
```

This is simpler to implement and lets the model learn whatever positional representation works best for the data, at the cost of not generalizing beyond `max_len` positions seen during training (unlike the sinusoidal version, which can in principle extrapolate to arbitrary lengths, though in practice modern long-context models typically use other techniques, such as rotary position embeddings, to handle very long sequences — worth knowing exists, though implementing them is beyond this introductory chapter).

---

## 13.2 Assembling a Transformer Encoder Block

A transformer encoder block combines four components in a specific arrangement: multi-head self-attention, a position-wise feedforward network, and residual connections with layer normalization around each:

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- Self-attention sub-block, with residual + pre-norm ---
        normed = self.norm1(x)
        attn_output, _ = self.self_attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_output)          # residual connection

        # --- Feedforward sub-block, with residual + pre-norm ---
        normed = self.norm2(x)
        ff_output = self.feedforward(normed)
        x = x + self.dropout(ff_output)             # residual connection

        return x

block = TransformerEncoderBlock(d_model=64, num_heads=8, d_ff=256)
x = torch.rand(2, 10, 64)
output = block(x)
print(output.shape)   # torch.Size([2, 10, 64]) -- same shape in and out, stackable
```

Notice the block's output shape exactly matches its input shape — this is deliberate and essential, since it's what allows blocks to be stacked arbitrarily deep, each one refining the same `(batch, seq_len, d_model)` representation.

### Why residual connections?

`x = x + self.dropout(attn_output)` rather than `x = self.dropout(attn_output)` — the residual (skip) connection adds the sub-block's output *to* its input, rather than replacing it. This directly addresses the same vanishing-gradient family of problems discussed for deep RNNs in Chapter 11: with a residual connection, gradients have a direct, unimpeded path (the `+ x` term) back to earlier layers during backpropagation, regardless of how deep the network is. Without residual connections, very deep transformers (real LLMs commonly have dozens of stacked blocks) become extremely difficult to train.

### Why layer normalization, and pre-norm vs. post-norm

`nn.LayerNorm` normalizes across the feature dimension (last dimension) for each individual sequence position independently — unlike `nn.BatchNorm` (Chapter 19), which normalizes across the batch dimension, `LayerNorm`'s per-position normalization doesn't depend on other samples in the batch, which suits variable-length sequences far better and requires no special handling for train vs. eval mode.

The code above uses **pre-norm**: normalization is applied *before* the attention/feedforward sub-layer (`self.norm1(x)` computed first, then fed into attention). The original transformer paper used **post-norm**: normalize *after* adding the residual (`self.norm1(x + attn_output)`). Pre-norm has become the dominant choice in modern LLMs, because it produces measurably more stable gradients early in training — deep post-norm transformers are notoriously sensitive to learning rate warmup and initialization, while pre-norm transformers train reliably with less careful tuning. This is a design detail worth recognizing when reading model source code or configs, since the two aren't interchangeable without adjusting other hyperparameters.

---

## 13.3 A Complete Mini Transformer for Sequence Classification

Stacking multiple encoder blocks, with an embedding layer and positional encoding at the front and a classification head at the end:

```python
class MiniTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids, padding_mask=None):
        x = self.embedding(token_ids) * math.sqrt(self.embedding.embedding_dim)  # standard scaling
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask=padding_mask)

        x = self.final_norm(x)
        pooled = x.mean(dim=1)          # mean-pool across sequence positions
        logits = self.classifier(pooled)
        return logits

model = MiniTransformerClassifier(
    vocab_size=10000, d_model=128, num_heads=8, d_ff=512, num_layers=4, num_classes=3
)

token_ids = torch.randint(1, 10000, (4, 20))   # (batch=4, seq_len=20)
logits = model(token_ids)
print(logits.shape)   # torch.Size([4, 3])

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

A few choices worth flagging:

- **`nn.ModuleList`** for the stacked blocks, not a plain Python list — directly applying the Chapter 5 lesson: a plain list would make the blocks invisible to `.parameters()`.
- **Embedding scaling** (`* math.sqrt(d_model)`) is a convention from the original transformer paper, keeping the embedding and positional encoding magnitudes comparable before they're added together.
- **Mean pooling** (`x.mean(dim=1)`) is one simple way to collapse a `(batch, seq_len, d_model)` sequence representation into a single `(batch, d_model)` vector for classification; many real models instead prepend a special `[CLS]` token and use its final representation, but mean pooling is simpler to reason about for a first implementation.

---

## 13.4 Using `nn.TransformerEncoder` and `nn.TransformerEncoderLayer`

Having built the mechanics by hand, in practice you'll typically use PyTorch's built-in, more heavily optimized transformer components rather than the hand-rolled version above:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128, nhead=8, dim_feedforward=512, dropout=0.1,
    batch_first=True, norm_first=True,   # norm_first=True -> pre-norm (Section 13.2)
)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

x = torch.rand(4, 20, 128)   # (batch, seq_len, d_model)
output = transformer_encoder(x)
print(output.shape)   # torch.Size([4, 20, 128])
```

`nn.TransformerEncoder` takes a single `nn.TransformerEncoderLayer` instance as a template and internally creates `num_layers` independent copies of it (with separately initialized weights, not shared) — directly analogous to the `nn.ModuleList` of `TransformerEncoderBlock`s built by hand in Section 13.3. `norm_first=True` selects pre-norm; leaving it at the default `False` uses post-norm, matching the original paper's design discussed in Section 13.2.

### Using it with masking

```python
seq_len = 20
causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)   # built-in helper (Ch. 12, Section 12.4)

output = transformer_encoder(x, mask=causal_mask)
```

`generate_square_subsequent_mask` produces exactly the kind of causal mask built by hand in Chapter 12 — using PyTorch's own convention for the mask format (additive, with `-inf` for masked positions), which matches what `nn.TransformerEncoderLayer` expects internally.

**Practical guidance:** build attention and transformer blocks from scratch (as in Sections 13.1-13.3) specifically to understand the mechanics and be able to debug shape or masking issues confidently — but reach for `nn.TransformerEncoder`/`nn.TransformerEncoderLayer` in real projects, since they include optimized attention kernels (including, on supported hardware, fused implementations that are substantially faster than the naive from-scratch version) that a hand-written implementation won't match without significant additional engineering.

---

## 13.5 Encoder-Only vs. Decoder-Only vs. Encoder-Decoder

Worth being explicit about this taxonomy, since it explains why different transformer-based models look architecturally different despite sharing the same core building block:

- **Encoder-only** (e.g., BERT-style models): bidirectional self-attention, no causal mask — every position can attend to every other position. Well-suited to understanding tasks (classification, the `MiniTransformerClassifier` built above).
- **Decoder-only** (e.g., GPT-style models, and the Qwen2.5-Coder model you've fine-tuned): causal self-attention only — each position can only attend to itself and earlier positions (Chapter 12, Section 12.4's causal mask). Well-suited to generation, since a model that generates text one token at a time must never see future tokens it hasn't produced yet.
- **Encoder-decoder** (e.g., the original transformer, T5-style models): an encoder processes the full input bidirectionally, and a separate decoder generates output autoregressively, using cross-attention (Chapter 12, Section 12.3) to attend back to the encoder's output at each step. Well-suited to sequence-to-sequence tasks like translation.

The vast majority of modern LLMs — including everything in the GPT, Llama, and Qwen families — are decoder-only, which is why the causal masking mechanics from Chapter 12 are the ones most directly relevant to the fine-tuning work in later chapters of this book (Chapter 30).

---

## Summary

- Attention is permutation-invariant by itself; positional encoding (sinusoidal or learned) injects order information into embeddings before attention is applied.
- A transformer encoder block combines multi-head self-attention and a feedforward network, each wrapped in a residual connection and layer normalization.
- Residual connections provide a direct gradient path through deep stacks of blocks; `nn.LayerNorm` normalizes per-position, independent of batch composition.
- Pre-norm (normalize before each sub-layer) is more stable for training deep transformers than the original post-norm design, and is the modern default.
- `nn.TransformerEncoder`/`nn.TransformerEncoderLayer` provide optimized, production-ready versions of exactly what you built by hand in this chapter.
- Encoder-only, decoder-only, and encoder-decoder architectures all share the same core block, differing in attention masking and the presence (or absence) of cross-attention — decoder-only, causal architectures underlie most modern LLMs, including Qwen2.5-Coder.

## Exercises

1. Implement the sinusoidal positional encoding from Section 13.1, plot the encoding values for the first 4 dimensions across the first 50 positions (using matplotlib), and describe the pattern you observe.
2. Modify `TransformerEncoderBlock` from Section 13.2 to use post-norm instead of pre-norm, and verify the output shape is unaffected — the architecture change is purely about training dynamics, not shape.
3. Extend `MiniTransformerClassifier` to accept a padding mask (built using the pattern from Chapter 12, Section 12.4) so that padding tokens don't influence the mean-pooled representation, and confirm this by comparing outputs on a padded vs. unpadded version of the same sequence.
4. Using `nn.TransformerEncoderLayer` with `norm_first=True` and a causal mask via `generate_square_subsequent_mask`, build a minimal decoder-only next-token-prediction model over a small vocabulary, and verify that changing a later token in the input sequence doesn't affect the model's output logits for earlier positions.

**Next:** Chapter 14 covers transfer learning — using `torchvision.models` and pretrained backbones, freezing layers, and fine-tuning strategies, closing out Part III before Part IV moves into training at scale.
