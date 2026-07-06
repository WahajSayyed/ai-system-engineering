# Chapter 11: Recurrent Networks

## Introduction

Convolutions (Chapter 10) exploit spatial locality — the assumption that nearby pixels are related. Sequential data — text, time series, sensor readings over time — has a different kind of structure: **order matters**, and elements can depend on others arbitrarily far back in the sequence. Recurrent neural networks (RNNs) are the classical architecture family designed for exactly this: processing a sequence one element at a time while maintaining a running summary of everything seen so far.

This chapter covers `nn.RNN`, and its two far more practically useful variants, `nn.LSTM` and `nn.GRU`, along with the sequence batching mechanics — padding, packing, masking — that make training on real, variable-length sequence data tractable.

By the end of this chapter you'll be able to:
- Explain the recurrent computation and why plain RNNs struggle with long sequences
- Use `nn.LSTM` and `nn.GRU` correctly, including their input/output shape conventions
- Batch variable-length sequences using padding and `pack_padded_sequence`
- Build a sequence classifier end to end

---

## 11.1 The Recurrent Idea

A recurrent layer processes a sequence step by step, maintaining a **hidden state** that gets updated at each step and carries information forward:

$$h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$$

At each timestep $t$, the layer combines the current input $x_t$ with the previous hidden state $h_{t-1}$ to produce a new hidden state $h_t$. The same weights ($W_{ih}$, $W_{hh}$) are reused at every timestep — this is a form of parameter sharing conceptually similar to a convolution's kernel sharing across spatial positions (Chapter 10), just applied across *time* instead of *space*.

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

x = torch.rand(4, 5, 10)   # (batch=4, seq_len=5, input_size=10)
output, hidden = rnn(x)

print(output.shape)   # torch.Size([4, 5, 20])  -- hidden state at EVERY timestep
print(hidden.shape)    # torch.Size([1, 4, 20])  -- hidden state at the FINAL timestep only
```

Two return values, both worth understanding precisely:
- **`output`**: the hidden state produced at *every* timestep — shape `(batch, seq_len, hidden_size)` with `batch_first=True`. Use this when you need a representation of every position (e.g., token-level tagging, or as input to an attention mechanism, Chapter 12).
- **`hidden`**: just the *final* hidden state — shape `(num_layers, batch, hidden_size)`. Use this when you need a single summary vector for the entire sequence (e.g., sequence classification).

`batch_first=True` is worth setting explicitly and consistently — without it, PyTorch's recurrent layers default to `(seq_len, batch, input_size)` ordering, which is inconsistent with the `(batch, ...)`-first convention used everywhere else in this book (`Dataset`/`DataLoader`, CNNs, etc.) and is a common source of silent shape bugs when it's forgotten.

---

## 11.2 Why Plain RNNs Struggle: The Vanishing Gradient Problem

Plain `nn.RNN` has a well-known practical limitation: gradients backpropagated through many timesteps tend to shrink toward zero (or, less commonly, explode — connecting directly back to Chapter 8's gradient clipping) as they're repeatedly multiplied by the same recurrent weight matrix. In practice, this means plain RNNs struggle to learn dependencies that span more than roughly a few dozen timesteps — information from early in a long sequence effectively gets "forgotten" before it can influence the final output.

This is precisely the problem **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) were designed to solve, and it's why you'll almost never use plain `nn.RNN` in practice — it's included here mainly as the conceptual starting point for understanding what LSTM and GRU add.

---

## 11.3 `nn.LSTM`: Gated Memory

LSTM introduces a separate **cell state** ($c_t$) alongside the hidden state, plus three learned **gates** (input, forget, output) that control how information flows into, out of, and is retained in that cell state at each timestep. The mechanism is more complex than a plain RNN, but the practical upshot is straightforward: LSTMs can learn to *selectively remember* information across long spans, because the forget gate can learn to keep the cell state nearly unchanged across many timesteps when appropriate, sidestepping the vanishing gradient problem.

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)

x = torch.rand(4, 5, 10)
output, (hidden, cell) = lstm(x)

print(output.shape)   # torch.Size([4, 5, 20])
print(hidden.shape)    # torch.Size([1, 4, 20])
print(cell.shape)       # torch.Size([1, 4, 20])  -- the extra cell state
```

The only structural difference from `nn.RNN` at the API level: LSTM returns a **tuple** `(hidden, cell)` as its second output instead of a single hidden tensor, reflecting the extra cell state it maintains internally.

### Multi-layer (stacked) LSTMs

Stacking multiple recurrent layers — feeding the output sequence of one LSTM as the input sequence to another — lets the model build progressively more abstract sequence representations, analogous to stacking convolutional blocks in Chapter 10:

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True, dropout=0.2)

x = torch.rand(4, 5, 10)
output, (hidden, cell) = lstm(x)

print(hidden.shape)   # torch.Size([2, 4, 20])  -- one hidden state per layer now
```

Note `dropout` here applies *between* stacked layers, not within a single layer's recurrence — and it's silently ignored (with a warning) if `num_layers=1`, since there's nothing between layers to apply it to.

### Bidirectional LSTMs

A **bidirectional** LSTM processes the sequence in both directions — forward and backward — and concatenates the results, useful whenever the entire sequence is available at once (as opposed to streaming/online prediction, where future timesteps genuinely aren't available yet):

```python
bilstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True, bidirectional=True)

x = torch.rand(4, 5, 10)
output, (hidden, cell) = bilstm(x)

print(output.shape)   # torch.Size([4, 5, 40])  -- 20*2, forward+backward concatenated
```

---

## 11.4 `nn.GRU`: A Simpler Gated Alternative

GRU is a streamlined alternative to LSTM — it merges the cell state into the hidden state and uses two gates instead of three, resulting in fewer parameters and often comparable performance to LSTM on many tasks, at somewhat lower computational cost.

```python
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)

x = torch.rand(4, 5, 10)
output, hidden = gru(x)   # note: single hidden tensor, no separate cell state — like plain RNN's API

print(output.shape)   # torch.Size([4, 5, 20])
print(hidden.shape)    # torch.Size([1, 4, 20])
```

**Practical guidance:** LSTM and GRU are both reasonable defaults, and neither dominates the other universally — GRU is a reasonable first choice for smaller datasets or when training speed matters more, given its lower parameter count; LSTM's separate cell state can be advantageous on tasks with very long-range dependencies. When in doubt, try both and compare — the API difference is only in the return values, so swapping one for the other in existing code is close to a one-line change.

---

## 11.5 Batching Variable-Length Sequences

Real sequence data — sentences, time series of different durations — rarely arrives with uniform length, which directly connects to the custom `collate_fn` and padding pattern introduced in Chapter 7, Section 7.6. This section extends that pattern with the mechanics specific to recurrent layers.

### The problem with naive padding

Padding sequences to a common length (as in Chapter 7) lets you batch them into a single tensor, but naively running a padded batch through an RNN/LSTM/GRU means the recurrent layer keeps "processing" padding tokens as if they were real data — updating its hidden state based on meaningless zero-vectors past the end of shorter sequences, which can corrupt the final hidden state you extract for classification.

### `pack_padded_sequence` and `pad_packed_sequence`

PyTorch's solution is a special `PackedSequence` representation that tells the recurrent layer exactly how long each sequence in the batch actually is, so it can skip computation on padding entirely:

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

sequences = [torch.rand(5, 10), torch.rand(3, 10), torch.rand(4, 10)]  # (seq_len, features)
lengths = torch.tensor([5, 3, 4])

padded = pad_sequence(sequences, batch_first=True)   # shape (3, 5, 10) — Chapter 7 pattern
print(padded.shape)

packed = pack_padded_sequence(
    padded, lengths, batch_first=True, enforce_sorted=False
)

lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
packed_output, (hidden, cell) = lstm(packed)   # LSTM accepts a PackedSequence directly

output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
print(output.shape)         # torch.Size([3, 5, 20]) -- padded again, for downstream use
print(output_lengths)        # tensor([5, 3, 4]) -- original lengths, preserved
```

`enforce_sorted=False` tells PyTorch the input sequences aren't pre-sorted by descending length — required in older PyTorch versions but now handled internally, at a small performance cost compared to pre-sorting. If performance matters and you're processing many batches, sorting sequences by length within each batch (and setting `enforce_sorted=True`) avoids that overhead.

**Practically:** `hidden` (the final hidden state) from a packed sequence is automatically the state at each sequence's *true* final timestep — not the padded length — so for sequence classification tasks where you only need the final hidden state, packing correctly handles this for you without any extra indexing logic.

---

## 11.6 A Complete Example: Sequence Classification

Classifying a batch of variable-length sequences (e.g., sentiment classification from tokenized text) into one of several categories, using an LSTM's final hidden state:

```python
class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)   # (batch, seq_len, embed_dim)

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)      # only need the final hidden state

        # hidden shape: (num_layers=1, batch, hidden_size) -> squeeze to (batch, hidden_size)
        logits = self.fc(hidden.squeeze(0))
        return logits

model = SequenceClassifier(vocab_size=5000, embed_dim=64, hidden_size=128, num_classes=3)

token_ids = torch.randint(1, 5000, (4, 6))   # (batch=4, seq_len=6), 0 reserved for padding
lengths = torch.tensor([6, 4, 5, 2])
logits = model(token_ids, lengths)
print(logits.shape)   # torch.Size([4, 3])
```

Note `padding_idx=0` in `nn.Embedding` — this tells the embedding layer to keep the embedding vector for token ID `0` fixed at zero and exclude it from gradient updates, since it represents padding and carries no real semantic content. This, combined with `pack_padded_sequence`, means padding tokens are cleanly excluded from the model's computation at both the embedding and recurrent stages. `lengths.cpu()` is required because `pack_padded_sequence` expects the lengths tensor on CPU even when the rest of the model runs on GPU — a small, easy-to-forget device-placement detail (connecting back to Chapter 1's device discussion).

The training loop for this model is, once again, unchanged from Chapter 8/9 — only `forward()` now takes an extra `lengths` argument, and the `DataLoader`'s `collate_fn` (Chapter 7) needs to produce both padded token IDs and lengths per batch.

---

## Summary

- Recurrent layers process sequences step by step, maintaining a hidden state that summarizes everything seen so far, sharing weights across timesteps.
- Plain `nn.RNN` suffers from vanishing gradients on long sequences; `nn.LSTM` (gated cell state) and `nn.GRU` (simpler, fewer parameters) both address this and are the practical defaults.
- `output` gives the hidden state at every timestep; `hidden` (and, for LSTM, `cell`) gives only the final state — choose based on whether you need per-position or whole-sequence representations.
- Naive padding corrupts recurrent computation unless paired with `pack_padded_sequence`/`pad_packed_sequence`, which tells the layer to skip padding positions entirely.
- `padding_idx` in `nn.Embedding` keeps padding tokens' embeddings fixed at zero and excluded from training.

## Exercises

1. Build a plain `nn.RNN`-based sequence classifier and an `nn.LSTM`-based one with matching hidden sizes, and compare their parameter counts using the `sum(p.numel() ...)` pattern from Chapter 5.
2. Given three sequences of lengths 7, 3, and 5, manually pad them, pack them with `pack_padded_sequence`, and verify by printing shapes that the final `hidden` state reflects each sequence's true last timestep rather than the padded length.
3. Modify `SequenceClassifier` to use a bidirectional LSTM, adjusting the shape of `self.fc`'s input accordingly, and explain in your own words why bidirectionality requires that adjustment.
4. Implement the same sequence classification task using `nn.GRU` instead of `nn.LSTM`, and note exactly which lines needed to change.

**Next:** Chapter 12 introduces attention mechanisms — a fundamentally different way of relating positions in a sequence to each other, built from scratch, forming the foundation for the transformer architecture in Chapter 13.
