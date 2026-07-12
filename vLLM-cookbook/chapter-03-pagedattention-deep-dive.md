# Chapter 3 — PagedAttention Deep Dive

> **Track:** Beginner → Intermediate  
> **Prerequisite:** Chapter 1 (why the problem exists), Chapter 2 (you've run inference)  
> **Source:** Original SOSP 2023 paper + vLLM v1 architecture docs

---

## Table of Contents

1. [What is a KV cache — precisely](#1-what-is-a-kv-cache--precisely)
2. [What a KV block actually stores](#2-what-a-kv-block-actually-stores)
3. [The block size choice: why 16 tokens?](#3-the-block-size-choice-why-16-tokens)
4. [Logical blocks vs physical blocks](#4-logical-blocks-vs-physical-blocks)
5. [The block table](#5-the-block-table)
6. [Walkthrough: a single request, step by step](#6-walkthrough-a-single-request-step-by-step)
7. [The free block pool](#7-the-free-block-pool)
8. [How attention computation works over blocks](#8-how-attention-computation-works-over-blocks)
9. [Block sharing: copy-on-write for parallel sampling](#9-block-sharing-copy-on-write-for-parallel-sampling)
10. [Prefix caching (Automatic Prefix Caching — APC)](#10-prefix-caching-automatic-prefix-caching--apc)
11. [Memory pressure: swapping and preemption](#11-memory-pressure-swapping-and-preemption)
12. [What `# GPU blocks: N` means in your logs](#12-what--gpu-blocks-n-means-in-your-logs)
13. [Why the math still works on non-contiguous blocks](#13-why-the-math-still-works-on-non-contiguous-blocks)
14. [PagedAttention overhead: the honest numbers](#14-pagedattention-overhead-the-honest-numbers)
15. [Practice exercises](#15-practice-exercises)
16. [Chapter summary](#16-chapter-summary)

---

## 1. What is a KV cache — precisely

Before understanding how PagedAttention manages the KV cache, you need a precise picture of what it actually is.

In a transformer, every token attends to every previous token using the **attention mechanism**:

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V
```

- **Q (Query):** Comes from the current token — "what am I looking for?"
- **K (Key):** Comes from all past tokens — "what do I contain?"
- **V (Value):** Comes from all past tokens — "what information should I pass forward?"

During **decode** (generating token by token), you only have one new query at each step. But you still need the keys and values from **every previous token**, at **every layer** of the model.

Rather than recomputing K and V for all past tokens on every step, you **cache** them. That cache is the **KV cache**.

### KV cache size — the actual formula

```
KV cache bytes per token =
    2                    ← one tensor for K, one for V
  × num_layers           ← stored for every transformer layer
  × num_kv_heads         ← stored for every attention head
  × head_dim             ← dimension of each head vector
  × dtype_bytes          ← 2 for float16/bfloat16, 4 for float32
```

**Concrete example — Llama-3.1-8B (bfloat16):**
```
num_layers   = 32
num_kv_heads = 8    (uses Grouped Query Attention — fewer KV heads than Q heads)
head_dim     = 128
dtype_bytes  = 2

KV per token = 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB per token
```

For a 4,096-token context: `128 KB × 4,096 = 512 MB` — for a **single request**.  
With 40 concurrent users: `512 MB × 40 = 20 GB` — just for KV cache.

This is why KV cache memory management is the defining challenge of LLM serving.

---

## 2. What a KV block actually stores

PagedAttention divides the KV cache into **fixed-size blocks**. Each block stores the K and V tensors for a fixed number of tokens (default: **16 tokens per block**).

```
One KV block stores:
  K vectors for tokens [t₀, t₁, ..., t₁₅]   ← one per token per layer per head
  V vectors for tokens [t₀, t₁, ..., t₁₅]   ← same

Block memory = 2 × block_size × num_kv_heads × head_dim × dtype_bytes
             = 2 × 16 × 8 × 128 × 2 bytes   (Llama-3.1-8B example)
             = 65,536 bytes ≈ 64 KB per block
```

**All blocks are the same size.** This is the key property that eliminates external fragmentation — every freed block is immediately usable by any new request, with no size mismatch.

---

## 3. The block size choice: why 16 tokens?

Block size is the main tuning knob in PagedAttention. It involves a genuine tradeoff:

| Block size | Effect |
|-----------|--------|
| Smaller (e.g. 4 tokens) | Less internal fragmentation (last block is less than half empty on average), but more blocks to track → more block table overhead |
| Larger (e.g. 64 tokens) | Less overhead per block, but last block wastes more memory (up to 63 slots empty) |
| **16 tokens (default)** | The vLLM authors tested many values and found 16 strikes the best balance |

At block size 16, the **maximum internal fragmentation per request is 15 tokens** — the last, partially filled block. With millions of requests, the average waste per block converges to `block_size / 2 = 8 tokens` — far better than the 60–80% waste of the pre-PagedAttention approach.

The block size can be changed via `--block-size {8,16,32}` when launching `vllm serve`, but 16 is correct for virtually all workloads.

---

## 4. Logical blocks vs physical blocks

The central idea of PagedAttention — directly from OS virtual memory — is to separate what the **sequence sees** from where **memory is actually stored**.

```
LOGICAL VIEW (per request — what the model sees):
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Logical     │ │ Logical     │ │ Logical     │ │ Logical     │
│ Block 0     │ │ Block 1     │ │ Block 2     │ │ Block 3     │
│ tokens 0–15 │ │ tokens16–31 │ │ tokens32–47 │ │ tokens48–55 │
│ (16 filled) │ │ (16 filled) │ │ (16 filled) │ │ (8 filled)  │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
  Block table maps logical → physical (anywhere in GPU memory)

PHYSICAL VIEW (GPU VRAM — actual locations):
┌──────────┐     ┌──────────┐
│ Phys #3  │     │ Phys #11 │     ← tokens 0–15
└──────────┘     └──────────┘
                 ┌──────────┐
                 │ Phys #2  │     ← tokens 16–31
                 └──────────┘
┌──────────┐
│ Phys #17 │                     ← tokens 32–47
└──────────┘
                              ┌──────────┐
                              │ Phys #5  │  ← tokens 48–55
                              └──────────┘
```

The sequence's tokens appear **contiguous** in the logical view. Physically, the blocks are **scattered** across GPU memory wherever free blocks exist. The attention kernel uses the **block table** to find them.

This is identical to how OS virtual memory works:
- OS: process sees contiguous virtual address space → physical RAM frames can be anywhere
- vLLM: sequence sees contiguous token positions → physical KV blocks can be anywhere

---

## 5. The block table

Each active request in vLLM has its own **block table** — a simple array mapping logical block index to physical block ID, plus how many token slots are filled in that block.

```python
# Simplified block table structure (from vLLM source)
class KVCacheBlock:
    block_id: int        # Physical block ID (index into GPU memory pool)
    block_hash: int      # Hash of block contents (used for prefix caching)
    ref_cnt: int         # Number of requests currently using this block
    # Doubly linked list pointers for the free queue
    prev_free_block: "KVCacheBlock | None"
    next_free_block: "KVCacheBlock | None"
```

**Example block table for a request with 55 tokens (block size 16):**

```
Logical Block 0  →  Physical Block #7   (16/16 filled)
Logical Block 1  →  Physical Block #1   (16/16 filled)
Logical Block 2  →  Physical Block #9   (16/16 filled)
Logical Block 3  →  Physical Block #14  (7/16 filled)  ← last block, partial
```

When the request generates token 56, vLLM writes it into logical block 3 (physical block 14), slot 8. When logical block 3 fills up (token 63), vLLM allocates a **new** physical block from the free pool and appends it to the block table.

---

## 6. Walkthrough: a single request, step by step

Let's trace exactly what happens when you call `llm.generate(["The capital of France is"], params)`.

**Step 0 — Tokenisation**
```
"The capital of France is"  →  [464, 3139, 286, 4881, 318]
                                  5 tokens
```

**Step 1 — Prefill (all prompt tokens processed in parallel)**

vLLM needs enough physical blocks to hold 5 tokens. With block size 16, it allocates **1 physical block** (block #7) from the free pool.

```
Block table after prefill:
  Logical Block 0  →  Physical Block #7   (5/16 filled)
```

All 5 prompt tokens' K and V vectors are computed and written into physical block #7.

**Step 2 — Decode, step 1: generate "Paris"**

The model generates token "Paris" (token ID: 6342).

- Write K and V for "Paris" into physical block #7, slot 6.
- Block table: Physical Block #7, now 6/16 filled.
- Output so far: "Paris"

**Step 3 — Decode, step 2: generate "."**

- Write K and V for "." into physical block #7, slot 7.
- Block table: Physical Block #7, now 7/16 filled.

**Step 4 — Continue decoding until block #7 fills up (16/16)**

At token 17, physical block #7 is full. For token 18:

- vLLM pops physical block #22 from the free block pool.
- Appends it to the block table.

```
Block table after block fills:
  Logical Block 0  →  Physical Block #7    (16/16 filled)
  Logical Block 1  →  Physical Block #22   (1/16 filled)
```

**Step 5 — Request completes**

The model generates `<EOS>`. Block #7 and block #22 are both returned to the free block pool. Their ref_cnt drops to 0 — they are immediately available for new requests.

**No memory is wasted** between requests. Any new request can use blocks #7 and #22 the moment this request finishes.

---

## 7. The free block pool

The free block pool is a **doubly linked list** of all unallocated physical blocks. Using a doubly linked list gives O(1) allocation and deallocation — critical for high-throughput serving.

```
At startup:
  All N GPU blocks → free pool
  [Block#0] ↔ [Block#1] ↔ [Block#2] ↔ ... ↔ [Block#N]

When a request arrives:
  Pop blocks from the head of the free list → assign to request

When a request finishes:
  Append its blocks to the tail of the free list → available immediately
```

The number of blocks (`N`) is computed at startup:

```
N = floor(available_VRAM / block_size_bytes)

Where available_VRAM = total_VRAM × gpu_memory_utilization − model_weights − activation_buffer
```

This is exactly the `# GPU blocks: N` line you see in vLLM startup logs. More blocks = more KV cache capacity = more concurrent requests the engine can serve.

---

## 8. How attention computation works over blocks

You might wonder: standard attention assumes K and V are in one contiguous tensor. How does it work when they're scattered across non-contiguous blocks?

The PagedAttention **kernel** is rewritten to handle this. Instead of reading from a single tensor, it:

1. Receives the block table for the current sequence.
2. **Iterates over each logical block** in order.
3. For each logical block, looks up the **physical block ID** in the block table.
4. Reads K and V from that physical memory location.
5. Computes partial attention scores.
6. Accumulates the final result.

```
Standard attention kernel:
  for t in range(seq_len):
      k = K[t]               ← contiguous memory access
      v = V[t]
      score = Q · k
      ...

PagedAttention kernel:
  for block_idx in range(num_blocks):
      phys_block = block_table[block_idx]     ← one indirection
      for slot in range(block_size):
          k = K_cache[phys_block][slot]       ← jump to physical location
          v = V_cache[phys_block][slot]
          score = Q · k
          ...
```

The **mathematics of attention is identical** — the same dot products, the same softmax, the same weighted sum. The only difference is one extra memory indirection per block. This adds ~20–26% overhead to the kernel itself, but the end-to-end throughput improves 2–4× because you can batch far more requests simultaneously.

---

## 9. Block sharing: copy-on-write for parallel sampling

PagedAttention enables a powerful optimisation: **multiple sequences can share the same physical blocks** for tokens they have in common.

### Parallel sampling (n > 1)

When you request `n=4` completions for the same prompt (e.g., to pick the best one), all 4 sequences share the **same prompt KV blocks**.

```python
params = SamplingParams(n=4, temperature=0.9, max_tokens=100)
outputs = llm.generate(["Summarise the following article:"], params)
# 4 different continuations generated from the same prompt
```

```
Prompt tokens → Physical Block #7 (ref_cnt = 4)
                                    ↑
               Sequence A block table: Logical 0 → Phys #7
               Sequence B block table: Logical 0 → Phys #7
               Sequence C block table: Logical 0 → Phys #7
               Sequence D block table: Logical 0 → Phys #7
```

All 4 sequences read from the same physical block. No duplication. Memory saving: up to 6–10% on parallel sampling workloads.

### When sequences diverge — Copy-on-Write

Once sequences start generating different tokens, their block tables diverge. When sequence A needs to **write** a new token to a shared block:

1. vLLM checks: `ref_cnt > 1` for this block.
2. If yes → **allocate a new physical block**, copy the shared block's content into it.
3. Sequence A now points to the new block. Sequence B–D still point to the original.
4. `ref_cnt` of the original block decreases by 1.

```
Before write (shared):
  Phys Block #7  ←  Seq A, B, C, D  (ref_cnt = 4)

After Seq A generates a diverging token:
  Phys Block #7  ←  Seq B, C, D    (ref_cnt = 3)   ← unchanged
  Phys Block #22 ←  Seq A           (ref_cnt = 1)   ← new copy, now A writes here
```

This is **exactly** the Copy-on-Write mechanism from operating systems (used in `fork()`). The paper reports **37–55% memory savings** on beam search workloads due to KV block sharing.

---

## 10. Prefix caching (Automatic Prefix Caching — APC)

Block sharing can be extended across different requests — not just within one request. This is called **Automatic Prefix Caching (APC)**.

### The key insight

Each KV block can be **uniquely identified** by its content: the token IDs within it, plus the hash of all preceding blocks (forming a chain). This is analogous to a Merkle tree.

```
Block 0 hash = hash(None, tokens[0:16])
Block 1 hash = hash(Block_0_hash, tokens[16:32])
Block 2 hash = hash(Block_1_hash, tokens[32:48])
...
```

If two requests share the same first 32 tokens, their Block 0 and Block 1 hashes will be **identical**. vLLM maintains a global hash table of all cached blocks. On a new request, it checks whether any of its leading blocks already exist in the cache.

### Cache hit example

```
Request 1:
  System prompt (32 tokens) + User message A (50 tokens) → generates response

  Block 0: hash("You are a helpful assistant. Answer..") → stored in cache
  Block 1: hash(block0_hash, "You are a helpful assista..") → stored in cache

Request 2 (same system prompt, different user message):
  System prompt (32 tokens) + User message B (60 tokens)

  Block 0 check: hash matches! → cache HIT → reuse Block 0, skip recomputing it
  Block 1 check: hash matches! → cache HIT → reuse Block 1, skip recomputing it
  Only user message tokens need to be computed. Time-to-first-token drops significantly.
```

### Eviction policy

The cache is bounded by available KV blocks. When the cache is full and a new block must be allocated:

1. Find all blocks with `ref_cnt = 0` (not currently in use by any active request).
2. Evict the **Least Recently Used (LRU)** block among those.
3. The evicted block goes back to the free pool.

This is identical to an LRU page replacement policy in OS memory management.

### Enabling APC in vLLM

```python
# Python API
llm = LLM(model="...", enable_prefix_caching=True)

# CLI server
vllm serve <model> --enable-prefix-caching
```

**When APC helps the most:**
- Same system prompt reused across many requests (chatbots, RAG)
- Few-shot examples prepended to every request
- Long shared context (e.g., a document, a codebase loaded once)

**When APC doesn't help:**
- Every request has a completely unique prompt
- Very short prompts (less than one full block = 16 tokens)

---

## 11. Memory pressure: swapping and preemption

What happens when all KV blocks are in use and a new high-priority request arrives?

vLLM has two strategies, chosen per workload:

### Strategy 1 — Swap (CPU offloading)

Move a lower-priority request's KV blocks from GPU memory to CPU RAM. The request is **paused**. When GPU memory frees up, swap the blocks back.

```
GPU blocks full + new request arrives
  → Choose a lower-priority running request
  → Copy its KV blocks from GPU VRAM → CPU RAM over PCIe
  → Free those GPU blocks for the new request
  → When priority request finishes, swap the paused request back
```

**Cost:** PCIe transfer speed (typically 16–32 GB/s). For small blocks (16 tokens × ~64 KB), individual transfers are fast.

### Strategy 2 — Recomputation

Simply discard the KV blocks of a paused request entirely. When it's resumed, rerun the prefill phase to rebuild the KV cache from scratch.

**When recompute beats swapping:** For short prompts, re-running the prefill is faster than the PCIe transfer. The paper notes this is a non-obvious insight made possible only by block-level eviction.

### The `swap_space` parameter

```python
# Reserve 4 GB of CPU RAM for swapping KV blocks (default)
llm = LLM(model="...", swap_space=4)  # GiB
```

On a CPU-only setup (`vllm-cpu`), all KV cache already lives in RAM — this concept maps to disk-backed overflow, which is rarely needed.

---

## 12. What `# GPU blocks: N` means in your logs

When vLLM starts, you'll see a line like:

```
INFO:     # GPU blocks: 2048, # CPU blocks: 512
```

This tells you exactly how much KV cache capacity you have.

```
Total token capacity = GPU blocks × block_size
                     = 2048 × 16
                     = 32,768 tokens

This is the total simultaneous token capacity across ALL active requests.
```

**To estimate max concurrent users:**

```
max_concurrent_users ≈ total_token_capacity / avg_context_length

Example: 32,768 tokens / 512 tokens per user ≈ 64 concurrent users
```

**To increase `# GPU blocks`:**
- Increase `gpu_memory_utilization` (use more VRAM)
- Decrease `max_model_len` (allows more blocks to fit)
- Use a smaller model or quantized model (frees more VRAM for KV cache)
- Add more GPU memory (more hardware)

---

## 13. Why the math still works on non-contiguous blocks

A common misconception: "Doesn't attention require contiguous memory to be correct?"

**No.** Attention is defined purely in terms of dot products between vectors:

```
score(i) = Q · Kᵢ / √d_k
weight(i) = softmax(score(i))
output = Σᵢ weight(i) × Vᵢ
```

These operations iterate over tokens individually. Whether the tokens' K and V vectors live at consecutive memory addresses is irrelevant to the **result**. The PagedAttention kernel traverses the block table to locate each K/V pair — one extra pointer dereference — but computes exactly the same dot products in the same order.

The **output is mathematically identical** to standard attention. No approximations. This is why vLLM's outputs are indistinguishable from a naive implementation in terms of quality.

---

## 14. PagedAttention overhead: the honest numbers

PagedAttention is not free. Here are the real numbers from the original paper:

| Metric | Impact |
|--------|--------|
| Per-kernel attention latency | +20–26% vs contiguous attention |
| End-to-end throughput | **+2–4× improvement** vs Orca (best prior system) |
| Memory efficiency | 60–80% waste → **<4% waste** |
| Beam search KV savings | 37–55% memory reduction |
| Parallel sampling KV savings | 6–10% memory reduction |

The per-kernel overhead is more than compensated by the ability to run far more requests in parallel. This is the core tradeoff: slightly slower per-operation, dramatically higher system-wide throughput.

---

## 15. Practice exercises

### Exercise 1 — Calculate KV cache size for your model

```python
def kv_cache_bytes_per_token(num_layers, num_kv_heads, head_dim, dtype_bytes=2):
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

# Llama-3.1-8B (GQA: 8 KV heads, 32 layers, 128 head_dim)
llama_8b = kv_cache_bytes_per_token(32, 8, 128)
print(f"Llama-3.1-8B: {llama_8b} bytes/token = {llama_8b/1024:.1f} KB/token")
print(f"  At 4096 tokens: {llama_8b * 4096 / 1e9:.2f} GB per request")

# OPT-125M (12 layers, 8 heads, 64 head_dim — matches your laptop model)
opt_125m = kv_cache_bytes_per_token(12, 8, 64)
print(f"\nOPT-125M: {opt_125m} bytes/token = {opt_125m/1024:.2f} KB/token")
print(f"  At 512 tokens:  {opt_125m * 512 / 1e6:.2f} MB per request")

# How many blocks needed for 512 tokens at block_size=16?
tokens = 512
block_size = 16
import math
blocks_needed = math.ceil(tokens / block_size)
print(f"\n512 tokens → {blocks_needed} blocks (block_size=16)")
print(f"Internal fragmentation of last block: {tokens % block_size} unused slots")
```

### Exercise 2 — Observe prefix caching in action

```python
import os, time
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"

from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    dtype="bfloat16",
    max_model_len=256,
    enable_prefix_caching=True,   # Enable APC
)
params = SamplingParams(temperature=0.0, max_tokens=20)

shared_prefix = "The history of computing begins with Charles Babbage who designed "

# First request — no cache, full computation
t0 = time.time()
out1 = llm.generate([shared_prefix + "the Analytical Engine in"], params)
t1 = time.time()
print(f"First request:  {(t1-t0)*1000:.0f}ms  (cold — no cache)")

# Second request — same prefix, different ending → cache HIT on prefix blocks
t2 = time.time()
out2 = llm.generate([shared_prefix + "his Difference Engine around"], params)
t3 = time.time()
print(f"Second request: {(t3-t2)*1000:.0f}ms  (warm — prefix blocks reused)")
# Second request should be meaningfully faster due to APC
```

### Exercise 3 — Parallel sampling and block sharing

```python
from vllm import LLM, SamplingParams
import os
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"

llm = LLM(model="facebook/opt-125m", dtype="bfloat16", max_model_len=256)

# n=4: generate 4 different continuations from the same prompt
# Internally, all 4 share the prompt's KV blocks (copy-on-write kicks in at divergence)
params = SamplingParams(n=4, temperature=1.0, max_tokens=30)
outputs = llm.generate(["Once upon a time in a land far away"], params)

print(f"Number of completions: {len(outputs[0].outputs)}")
for i, completion in enumerate(outputs[0].outputs):
    print(f"\n[{i}] {completion.text.strip()[:80]}")
    print(f"    finish_reason: {completion.finish_reason}")
```

### Exercise 4 — Block count math

```python
# Estimate GPU blocks from memory numbers
# (Run this to understand your log output)

def estimate_gpu_blocks(
    total_vram_gb,
    gpu_memory_utilization,
    model_weights_gb,
    num_layers,
    num_kv_heads,
    head_dim,
    block_size=16,
    dtype_bytes=2
):
    available = total_vram_gb * gpu_memory_utilization
    kv_pool = available - model_weights_gb - 2.0  # 2GB overhead estimate
    bytes_per_block = 2 * block_size * num_layers * num_kv_heads * head_dim * dtype_bytes
    n_blocks = int(kv_pool * 1e9 / bytes_per_block)
    total_tokens = n_blocks * block_size
    return n_blocks, total_tokens

# A100 40GB + Llama-3.1-8B
blocks, tokens = estimate_gpu_blocks(40, 0.90, 16, 32, 8, 128)
print(f"A100 40GB + Llama-3.1-8B: ~{blocks} GPU blocks, {tokens:,} token capacity")

# RTX 3090 24GB + Qwen2.5-7B
blocks, tokens = estimate_gpu_blocks(24, 0.90, 14, 28, 4, 128)
print(f"RTX 3090 24GB + Qwen2.5-7B: ~{blocks} GPU blocks, {tokens:,} token capacity")
```

---

## 16. Chapter summary

PagedAttention is one of the most elegant applications of OS concepts to ML systems. The entire design follows from one insight: **the KV cache has the same structure as a virtual memory address space**.

| Concept | Description |
|---------|-------------|
| KV block | Fixed-size unit storing K+V tensors for 16 tokens, all layers |
| Block size (16) | Chosen to balance internal fragmentation vs overhead |
| Logical block | The sequence's view — tokens appear contiguous |
| Physical block | Actual GPU memory location — can be anywhere |
| Block table | Per-request mapping: logical index → physical block ID |
| Free block pool | Doubly linked list of unallocated blocks; O(1) alloc/free |
| Copy-on-write | Shared blocks copied only when a sequence needs to diverge |
| Prefix caching (APC) | Blocks identified by hash; shared across requests with same prefix |
| Eviction | LRU policy on blocks with ref_cnt = 0 |
| Swapping | KV blocks moved CPU↔GPU under memory pressure |
| Recomputation | Alternative to swapping — re-run prefill when it's faster |
| `# GPU blocks: N` | Total KV cache capacity = N × 16 tokens |
| Kernel overhead | +20–26% per operation, but 2–4× system throughput gain |

---

## What's next

Chapter 4 — **Serving with the OpenAI-compatible API**: you'll start `vllm serve`, hit it with `curl` and the OpenAI Python SDK, and understand how all the PagedAttention machinery runs underneath every HTTP request you send.

---

*Source: [PagedAttention paper — SOSP 2023](https://arxiv.org/abs/2309.06180) | vLLM docs: https://docs.vllm.ai*
