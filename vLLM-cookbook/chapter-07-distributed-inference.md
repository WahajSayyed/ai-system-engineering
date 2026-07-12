# Chapter 7 — Distributed Inference

> **Track:** Advanced  
> **Prerequisite:** Chapters 1–6  
> **vLLM version:** 0.19.0 (April 2026)  
> **Sources:** vLLM Distributed Serving docs, vLLM Parallelism & Scaling docs, Red Hat Office Hours recap (Feb 2025), ROCm MoE Playbook (Nov 2025), Megatron-LM paper (Shoeybi et al., 2019), vLLM Data Parallel docs

---

## Why this chapter matters

In Chapter 6 you learned to make models *smaller* so they fit on fewer GPUs. This chapter is the other side of that equation: what do you do when the model is still too large — or when you need more throughput than one GPU can deliver?

A Llama-3.1-70B model at BF16 weighs ~140 GB. No single consumer or even datacenter GPU holds that. A Llama-3.1-405B is ~810 GB. Even after INT4 quantization it needs ~200 GB. You need multiple GPUs working together, and the way you wire them together determines whether you get low latency, high throughput, or both.

We'll build up from first principles — starting with the matrix math inside a single transformer layer, then showing exactly how each parallelism strategy splits that math across GPUs, what communication it requires, and when to use each.

---

## Table of Contents

1. [The bottleneck: one transformer layer, one GPU](#1-the-bottleneck-one-transformer-layer-one-gpu)
2. [Tensor parallelism — splitting within a layer](#2-tensor-parallelism--splitting-within-a-layer)
3. [Column-parallel and row-parallel linear layers](#3-column-parallel-and-row-parallel-linear-layers)
4. [How TP splits attention](#4-how-tp-splits-attention)
5. [AllReduce: the communication primitive](#5-allreduce-the-communication-primitive)
6. [TP constraint: attention head divisibility](#6-tp-constraint-attention-head-divisibility)
7. [NVLink vs PCIe: why interconnect bandwidth matters](#7-nvlink-vs-pcie-why-interconnect-bandwidth-matters)
8. [Pipeline parallelism — splitting across layers](#8-pipeline-parallelism--splitting-across-layers)
9. [The pipeline bubble problem in autoregressive inference](#9-the-pipeline-bubble-problem-in-autoregressive-inference)
10. [Data parallelism — independent replicas](#10-data-parallelism--independent-replicas)
11. [Expert parallelism — for MoE models](#11-expert-parallelism--for-moe-models)
12. [Combining parallelism strategies](#12-combining-parallelism-strategies)
13. [vLLM implementation: multiprocessing vs Ray](#13-vllm-implementation-multiprocessing-vs-ray)
14. [Single-node multi-GPU setup](#14-single-node-multi-gpu-setup)
15. [Multi-node setup with Ray](#15-multi-node-setup-with-ray)
16. [KV cache in distributed inference](#16-kv-cache-in-distributed-inference)
17. [Choosing a parallelism strategy — a decision guide](#17-choosing-a-parallelism-strategy--a-decision-guide)
18. [Common errors and fixes](#18-common-errors-and-fixes)
19. [Practice exercises](#19-practice-exercises)
20. [Chapter summary](#20-chapter-summary)

---

## 1. The bottleneck: one transformer layer, one GPU

Let's start with what actually happens inside one transformer layer when a single GPU processes a batch of tokens.

Every transformer layer contains two main compute blocks:

**Self-attention block:**
```
Input X  →  Q = X·W_Q  →  Attention(Q, K, V) = softmax(QKᵀ/√d)·V  →  Output·W_O
              K = X·W_K
              V = X·W_V
```

**Feed-forward (MLP) block:**
```
Input X  →  Gate = X·W_gate  →  SILU(Gate) ⊙ (X·W_up)  →  W_down  →  Output
```

For a 70B parameter model (Llama-3.1-70B), each transformer layer has roughly:
- 4 weight matrices for attention (Q, K, V, O): each ~(8192 × 8192) at BF16 = ~128 MB
- 3 weight matrices for MLP (gate, up, down): each ~(8192 × 28672) at BF16 = ~448 MB

That is roughly **1.2 GB per layer**, and there are **80 layers**. Total: ~140 GB.

The core challenge: **you cannot split a matrix multiply across GPUs in a naive way without getting the wrong answer**. If GPU 0 computes `X · W_Q[:4096]` and GPU 1 computes `X · W_Q[4096:]`, they each have a partial result — and neither GPU knows the full output. Tensor parallelism solves exactly this.

---

## 2. Tensor parallelism — splitting within a layer

Tensor parallelism (TP) splits individual weight matrices across GPUs so that each GPU computes a partial result, and the GPUs communicate at the end of each layer to combine their partial results into the correct full output.

The implementation in vLLM uses Megatron-LM's tensor parallel algorithm, originally developed for training (Shoeybi et al., 2019) and adapted for inference.

With TP=4, each GPU holds **1/4 of each weight matrix** in the model. The total model is spread evenly.

```
Without TP (1 GPU):
  GPU 0: Full W_Q (8192 × 8192), Full W_K, Full W_V, Full W_O
         Full W_gate, Full W_up, Full W_down
         Memory: ~140 GB for 70B model → DOES NOT FIT on one GPU

With TP=4:
  GPU 0: W_Q[:, 0:2048], W_K[:, 0:2048], W_V[:, 0:2048]  ← 1/4 of each matrix
  GPU 1: W_Q[:, 2048:4096], W_K[:, 2048:4096], ...
  GPU 2: W_Q[:, 4096:6144], ...
  GPU 3: W_Q[:, 6144:8192], ...
  Memory per GPU: ~35 GB → fits on one A100 40GB
```

Every GPU processes the **same input tokens** but against its own shard of the weight matrices. The key is choosing how to split each matrix — column-wise or row-wise — so that the partial results can be combined correctly.

---

## 3. Column-parallel and row-parallel linear layers

The Megatron-LM approach uses two complementary split strategies that work together to require only **one AllReduce per transformer block** instead of one per matrix multiply.

### Column-parallel (first linear layer in a block)

Split the weight matrix along **output features** (columns). Each GPU computes a partial output independently:

```
Full computation:    Y = X · W          where W is (d_in × d_out)
                                              X is (batch × d_in)
                                              Y is (batch × d_out)

Column-parallel (TP=4):
  GPU 0:  Y₀ = X · W[:,  0:d_out/4]   → partial output Y₀
  GPU 1:  Y₁ = X · W[:,  d_out/4:d_out/2]
  GPU 2:  Y₂ = X · W[:,  d_out/2:3*d_out/4]
  GPU 3:  Y₃ = X · W[:,  3*d_out/4:]
  
  Each GPU's result Y₀, Y₁, Y₂, Y₃ is a correct 1/4 slice of Y
  No communication needed here — each partial result is independently valid
```

For column-parallel, all GPUs receive the **same full input X** and compute different output slices. The outputs are independent — no communication needed at this stage.

### Row-parallel (second linear layer in a block)

Split along **input features** (rows). Each GPU computes a partial dot product, and all GPUs AllReduce to sum:

```
Full computation:    Z = Y · W          where W is (d_in × d_out)

Row-parallel (TP=4):
  GPU 0:  Z₀ = Y₀ · W[0:d_in/4,  :]   → partial sum for Z
  GPU 1:  Z₁ = Y₁ · W[d_in/4:d_in/2,:]
  GPU 2:  Z₂ = Y₂ · W[d_in/2:3*d_in/4,:]
  GPU 3:  Z₃ = Y₃ · W[3*d_in/4:,  :]
  
  Z = Z₀ + Z₁ + Z₂ + Z₃   ← AllReduce (sum) across all 4 GPUs
```

Here each GPU has a different input slice (Y₀, Y₁, Y₂, Y₃ from the column-parallel step above) and computes a partial contribution to the output. An **AllReduce sum** combines all four partial results into the correct final Z.

### Putting it together for an MLP block

For Llama's MLP (gate + up + down):

```
Input X (same on all GPUs)
  ↓
  [Column-parallel: W_gate, W_up]  ← Each GPU computes 1/TP of gate/up outputs
  ↓
  SILU activation on partial outputs  ← Elementwise, no communication needed
  ↓
  [Row-parallel: W_down]  ← Each GPU computes partial W_down output
  ↓
  AllReduce → full output  ← ONE AllReduce for the whole MLP block
```

Column parallelism applies to up-projection operations. Element-wise activation functions (e.g., SILU) operate on sharded outputs. Row parallelism is used in down-projection, with an all-reduce operation to aggregate final results. This elegant arrangement means only **one AllReduce per MLP block**, not three.

---

## 4. How TP splits attention

For self-attention, the split is along **attention heads**. With TP=4 and 32 heads:

```
GPU 0: Q heads  0–7,  K heads  0–7,  V heads  0–7
GPU 1: Q heads  8–15, K heads  8–15, V heads  8–15
GPU 2: Q heads 16–23, K heads 16–23, V heads 16–23
GPU 3: Q heads 24–31, K heads 24–31, V heads 24–31
```

Each GPU computes attention **only for its subset of heads**. The Q, K, V projections are column-parallel (each GPU computes a head shard independently). The output projection W_O is row-parallel (each GPU multiplies its head shard output, then AllReduce to combine):

```
Column-parallel:  [W_Q, W_K, W_V]  →  each GPU has 1/TP heads
Elementwise:       Attention(Q_i, K_i, V_i) per head i on each GPU
Row-parallel:     W_O  →  AllReduce to combine all head outputs
```

From the vLLM source `QKVParallelLinear`:
```python
if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```

This handles **Grouped Query Attention (GQA)** models like Llama-3 correctly: if there are fewer KV heads than TP GPUs, some GPUs replicate the KV heads rather than splitting them.

---

## 5. AllReduce: the communication primitive

After every attention block and every MLP block, all GPUs perform an **AllReduce**. This is the core communication cost of tensor parallelism.

AllReduce works as follows: each GPU sends its partial result to every other GPU in the TP group, and every GPU ends up with the final sum.

```
4 GPUs, each with partial result P₀, P₁, P₂, P₃:

  AllReduce:
    GPU 0 → all: P₀
    GPU 1 → all: P₁
    GPU 2 → all: P₂
    GPU 3 → all: P₃

  Each GPU computes: P₀ + P₁ + P₂ + P₃  (the correct full output)
```

For a 70B model with 80 transformer layers and TP=4:
- AllReduces per forward pass: ~160 (2 per layer × 80 layers)
- Data per AllReduce: proportional to hidden_dim × batch_tokens × bytes

The latency of each AllReduce is dominated by the **link bandwidth** between GPUs. This is why interconnect speed is critical for tensor parallelism.

---

## 6. TP constraint: attention head divisibility

Tensor parallelism has one hard constraint that trips up many users:

**`tensor_parallel_size` must evenly divide the number of attention heads (and KV heads for GQA models).**

```python
# Llama-3.1-8B: 32 Q-heads, 8 KV-heads
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=4)
# ✅ Works: 32 ÷ 4 = 8 Q-heads per GPU, 8 ÷ 4 = 2 KV-heads per GPU

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=3)
# ❌ Error: 32 Q-heads not divisible by 3
# "Total number of attention heads (32) must be divisible by tensor parallel size (3)"
```

Common valid TP sizes for popular models:

| Model | Q-heads | KV-heads | Valid TP values |
|-------|---------|----------|----------------|
| Llama-3.1-8B | 32 | 8 | 1, 2, 4, 8 |
| Llama-3.1-70B | 64 | 8 | 1, 2, 4, 8 |
| Llama-3.1-405B | 128 | 8 | 1, 2, 4, 8 |
| Qwen2.5-72B | 64 | 8 | 1, 2, 4, 8 |
| Mistral-7B | 32 | 8 | 1, 2, 4, 8 |

For quantized models, the weight tensor dimensions must also be divisible by TP. Quantization block sizes add additional constraints; for example, models with a quantization block size of 128 may not support TP=3 even if attention heads permit it.

---

## 7. NVLink vs PCIe: why interconnect bandwidth matters

The AllReduce cost depends entirely on how fast GPUs can communicate. There are two hardware configurations:

**NVLink (within a DGX/HGX node)**
- NVLink 4.0 (H100 SXM): 900 GB/s bidirectional per GPU
- All GPUs on the same node are connected directly in an all-to-all topology
- AllReduce is extremely fast — milliseconds even for large tensors
- TP across all GPUs on the node is practical and efficient

**PCIe (consumer GPUs, L40S, some cloud setups)**
- PCIe 5.0: ~128 GB/s total bandwidth (shared by all devices)
- Multiple GPUs go through a PCIe switch or the CPU
- AllReduce is significantly slower — the bottleneck for TP at high TP sizes

If the GPUs on the node do not have NVLink interconnect (e.g. L40S), leverage pipeline parallelism instead of tensor parallelism for higher throughput and lower communication overhead.

```
NVLink throughput (H100 SXM, TP=8):
  AllReduce for 1MB tensor: ~0.01ms → negligible overhead

PCIe throughput (L40S, TP=4):
  AllReduce for 1MB tensor: ~0.5ms → meaningful overhead at high QPS
```

**The practical rule:** use TP for all GPUs connected by NVLink. Switch to PP or DP for GPUs connected only by PCIe or InfiniBand across nodes.

---

## 8. Pipeline parallelism — splitting across layers

Pipeline parallelism (PP) takes a different approach: instead of splitting within each layer, it assigns **consecutive groups of layers to different GPUs**. Each GPU owns its layers entirely — no weight sharding — and passes activations to the next GPU.

```
Without PP (1 GPU, 80 layers):
  GPU 0: Layers 0–79, full model → requires 140 GB

With PP=2 (2 GPUs, 80 layers each):
  GPU 0: Layers 0–39  → requires 70 GB   (prefill + activations forward)
  GPU 1: Layers 40–79 → requires 70 GB   (receives activations, continues forward)
```

For a 2-node deployment with 8 GPUs per node and PP=2:
```
  Node 0 (GPU 0–7): Layers 0–39   (TP=8 within node)
  Node 1 (GPU 8–15): Layers 40–79  (TP=8 within node)
```

Communication between PP stages: GPU 7 (last GPU of stage 0) sends intermediate activations to GPU 8 (first GPU of stage 1). This is a **point-to-point send/receive**, not an AllReduce — lower communication volume than TP.

Pipeline Parallelism reduces memory constraints across GPUs but does not inherently decrease inference latency as tensor parallelism does.

---

## 9. The pipeline bubble problem in autoregressive inference

Pipeline parallelism has a severe structural problem with autoregressive LLM inference. Here is the precise issue:

In autoregressive decoding, each token depends on the previous token. This means each decode step must complete the **full forward pass** (all pipeline stages) before the next token can start. With 2 pipeline stages:

```
Step 1 — Token 1:
  Time 1:  GPU 0 processes token 1 → Stage 1 done
  Time 2:  GPU 1 processes token 1 → Stage 2 done → token 1 output

Step 2 — Token 2:
  Time 3:  GPU 0 processes token 2 → GPU 1 is IDLE waiting
  Time 4:  GPU 1 processes token 2

Pipeline efficiency:
  GPU 0: [active][idle][active][idle]...   → 50% utilization
  GPU 1: [idle][active][idle][active]...   → 50% utilization
```

The autoregressive token-by-token nature means that in any given moment, at most one stage is active while the others wait. Pipeline parallelism and autoregressive inference are fundamentally at odds: you can never achieve true pipeline parallelism in autoregressive inference and are therefore always forced to tolerate significant pipeline bubbles.

vLLM mitigates this somewhat with **micro-batch pipelining** — interleaving multiple requests across stages — but the fundamental bubble issue remains. This is why:

1. **TP is strongly preferred over PP for latency** — TP processes all layers in parallel for every step, with only AllReduce overhead
2. **PP is used primarily to span across nodes**, where cross-node communication is too slow for TP's AllReduce
3. The common practice: **TP within a node, PP across nodes**

---

## 10. Data parallelism — independent replicas

Data parallelism (DP) is conceptually the simplest strategy: run **independent complete copies** of the model on separate GPUs (or groups of GPUs), and route different requests to different replicas.

```
DP=4, 4 separate full model copies:

  GPU 0: Full model copy 1 → serves requests R1, R5, R9, ...
  GPU 1: Full model copy 2 → serves requests R2, R6, R10, ...
  GPU 2: Full model copy 3 → serves requests R3, R7, R11, ...
  GPU 3: Full model copy 4 → serves requests R4, R8, R12, ...

  No inter-GPU communication during inference
  4× throughput vs 1 GPU
```

**Key properties:**
- No communication overhead during inference (unlike TP)
- Requires enough VRAM per GPU to hold the full model
- Scales throughput linearly with the number of replicas
- Each replica has its own independent KV cache pool

**In vLLM**, data parallelism at the instance level is typically handled by an external load balancer (Kubernetes, nginx) routing requests to separate `vllm serve` instances. Starting from vLLM 0.9.0, vLLM also supports native DP via `--data-parallel-size` with a single API endpoint.

```bash
# Native DP in vLLM (V1 engine, v0.9.0+)
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --data-parallel-size 4    # 4 replicas, requires 4 GPUs total
  --tensor-parallel-size 1  # each replica uses 1 GPU

# Combined DP + TP: 4 replicas × 2 GPUs each = 8 GPUs total
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --data-parallel-size 2    # 2 replicas
  --tensor-parallel-size 4  # each replica spans 4 GPUs (TP)
```

---

## 11. Expert parallelism — for MoE models

Mixture-of-Experts (MoE) models like DeepSeek-V3, Mixtral, and Qwen3-MoE have a special structure: each layer contains many "expert" sub-networks (e.g., 256 experts in DeepSeek-V3), but only a small subset (e.g., 8) are activated per token.

Standard TP would shard each expert's weights across GPUs. Expert parallelism (EP) takes a different approach: **each GPU owns a different subset of complete experts**.

```
Without EP (TP=8, 256 experts):
  Each GPU holds all 256 experts, but each expert's weights are sharded 1/8
  Communication: AllReduce after every MoE layer

With EP (DP=8, EP=8, 256 experts):
  GPU 0: Experts  0–31  (32 complete experts)
  GPU 1: Experts 32–63
  GPU 2: Experts 64–95
  ...
  GPU 7: Experts 224–255
  Communication: AllToAll routing (tokens routed to correct expert GPUs)
```

Expert parallelism typically uses AllToAll communication (when DP>1) instead of AllReduce. AllToAll is more efficient for MoE because you only send tokens to the GPUs that hold the relevant experts.

```bash
# Single node, 8 GPUs, DeepSeek-V3
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 8
```

EP size is automatically calculated by vLLM; you do not set it directly.

**Important caveat from the docs:** Expert parallelism only activates when `TP_SIZE × DP_SIZE > 1`. With a pure PP configuration (TP=1, DP=1 per stage), EP does NOT activate even if `--enable-expert-parallel` is set.

---

## 12. Combining parallelism strategies

The strategies compose. The most common production configuration is **TP within nodes + PP across nodes**:

```
2 nodes × 8 GPUs each = 16 GPUs total

Configuration: TP=8, PP=2

Node 0 (GPUs 0–7): TP group, handles layers 0–39
  GPU 0: layers 0–39, attention heads 0–7 (of 64)
  GPU 1: layers 0–39, attention heads 8–15
  ...
  GPU 7: layers 0–39, attention heads 56–63

Node 1 (GPUs 8–15): TP group, handles layers 40–79
  GPU 8:  layers 40–79, attention heads 0–7
  ...
  GPU 15: layers 40–79, attention heads 56–63

Communication:
  Within Node 0: AllReduce over NVLink (fast)
  Within Node 1: AllReduce over NVLink (fast)
  Node 0 → Node 1: activation pass (InfiniBand, once per step)
```

The common practice is to set the tensor parallel size to the number of GPUs in each node, and the pipeline parallel size to the number of nodes. For example, if you have 16 GPUs across 2 nodes (8 GPUs per node), set the tensor parallel size to 8 and the pipeline parallel size to 2.

**When to use TP-only (no PP):**

You can also use pure TP across all GPUs if the interconnect supports it:

```bash
# 16 GPUs across 2 NVLink-connected nodes (e.g. DGX SuperPOD)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray
```

---

## 13. vLLM implementation: multiprocessing vs Ray

vLLM supports two distributed execution backends:

**Multiprocessing (default for single node):**
- Uses Python's `multiprocessing` + NCCL for GPU communication
- Automatically selected when all required GPUs are on the same node
- Lower overhead, simpler setup

**Ray (required for multi-node):**
- Distributed computing framework for Python
- Required when GPUs span multiple nodes
- vLLM uses Ray to manage worker processes across nodes
- Also available for single-node, but adds overhead

```python
# Single node — multiprocessing (default, no extra config needed)
from vllm import LLM
llm = LLM("meta-llama/Llama-3.1-70B-Instruct", tensor_parallel_size=4)

# Single node — explicit Ray backend
llm = LLM(
    "meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    distributed_executor_backend="ray",
)

# Multi-node — must use Ray
# (see multi-node section below)
```

Multiprocessing will be used by default when not running in a Ray placement group and if there are sufficient GPUs available on the same node for the configured `tensor_parallel_size`, otherwise Ray will be used.

---

## 14. Single-node multi-GPU setup

This is the most common case — you have a multi-GPU server (DGX, HGX, or a workstation with multiple GPUs) and want to run a model too large for one GPU.

### Python API

```python
from vllm import LLM, SamplingParams

# 70B model on 4×A100 80GB
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,     # Split across 4 GPUs
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
)

params = SamplingParams(temperature=0.7, max_tokens=200)
outputs = llm.generate(["What is tensor parallelism?"], params)
print(outputs[0].outputs[0].text)
```

### CLI server

```bash
# 70B model on 4 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --port 8000

# 405B model on 8 GPUs (needs 8×A100 80GB)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --port 8000
```

### Selecting which GPUs to use

```bash
# Use GPUs 0, 1, 2, 3 only
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4

# Use GPUs 4, 5, 6, 7 only (for a second instance on the same node)
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --port 8001
```

### Pipeline parallelism on a single node

If the GPU count doesn't evenly divide the model size, or if GPUs are not NVLink connected:

```bash
# 4 GPUs without NVLink (e.g. L40S) — use PP instead of TP
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 4 \
  --distributed-executor-backend ray
```

---

## 15. Multi-node setup with Ray

For models that require more GPUs than one node can provide (e.g., Llama-3.1-405B on commodity hardware, DeepSeek-R1 671B).

### Prerequisites

- Same Python environment and model path on all nodes
- Network connectivity between nodes (InfiniBand recommended for TP, Ethernet acceptable for PP)
- Ray cluster running

### Step 1 — Start the Ray cluster

On the head node:
```bash
pip install ray
ray start --head --port=6379
```

On each worker node:
```bash
ray start --address='<HEAD_NODE_IP>:6379'
```

Verify all nodes are connected:
```bash
ray status
# Should show N nodes available
```

### Step 2 — Launch vLLM across nodes

vLLM provides a helper script `examples/online_serving/run_cluster.sh`, but you can also launch directly:

```bash
# On the head node — 2 nodes × 8 GPUs, TP=8, PP=2
vllm serve /path/to/model \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --port 8000
```

### Alternatively: pure TP across all nodes (if high-speed interconnect)

```bash
# 16 GPUs total, TP=16 (requires NVLink or InfiniBand)
vllm serve /path/to/model \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray
```

### Alternatively: multiprocessing backend (two-node, no Ray)

```bash
# Node 0 (head, ip: 192.168.1.10)
vllm serve /path/to/model \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr 192.168.1.10

# Node 1 (worker)
vllm serve /path/to/model \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr 192.168.1.10 \
  --headless    # no API server on this node
```

Efficient tensor parallelism requires fast internode communication, preferably through high-speed network adapters such as InfiniBand.

---

## 16. KV cache in distributed inference

With TP, the KV cache is **sharded across GPUs** along the KV head dimension. Each GPU stores only the KV cache for its assigned KV heads.

```
Llama-3.1-70B: 8 KV heads, TP=8

  GPU 0: KV cache for KV head 0 only
  GPU 1: KV cache for KV head 1 only
  ...
  GPU 7: KV cache for KV head 7 only

Total KV cache capacity = sum of KV cache across all GPUs
→ With TP=8, you get 8× more total KV blocks than with a single GPU
```

This is a non-obvious benefit of tensor parallelism: you get more KV cache capacity, meaning you can serve more concurrent requests. The `# GPU blocks` shown in startup logs is **per GPU** — the total effective KV cache is `blocks × TP_size` (after accounting for the sharding).

With pipeline parallelism, each pipeline stage handles different layers. The KV cache for each layer is stored on the GPU that processes that layer. There is no cross-stage KV communication during the normal decode loop — only the forward activations move between stages.

---

## 17. Choosing a parallelism strategy — a decision guide

```
Step 1: Does the model fit on one GPU?
  YES → No distributed inference needed
  NO  → Continue

Step 2: Do all required GPUs exist on one node?
  YES → Use tensor parallelism (TP)
        tensor_parallel_size = number of GPUs needed
        Backend: multiprocessing (default)
  NO  → Continue to multi-node

Step 3: Does the node have NVLink between GPUs?
  YES (DGX/HGX node, NVLink) →
        TP within each node, PP across nodes
        tensor_parallel_size = GPUs per node
        pipeline_parallel_size = number of nodes
        Backend: Ray

  NO (L40S, consumer GPUs, PCIe only) →
        PP within node (avoids expensive AllReduce over slow PCIe)
        tensor_parallel_size = 1
        pipeline_parallel_size = number of GPUs
        Backend: Ray (or mp)

Step 4: Is the model an MoE (Mixtral, DeepSeek, Qwen-MoE)?
  YES → Consider expert parallelism
        DP + EP combination for high throughput
        TP + EP for low latency
        (See Section 11)

Step 5: Do you need to scale throughput rather than fit the model?
  Model fits on 1 GPU → Use data parallelism
  Run N replicas, load balance across them
  --data-parallel-size N  (or separate vllm serve instances)
```

**Summary table:**

| Strategy | Goal | Communication | Best hardware |
|----------|------|---------------|---------------|
| TP | Fit model / reduce latency | AllReduce per layer | NVLink, within node |
| PP | Fit model across nodes | Send/recv per stage | InfiniBand, across nodes |
| DP | Scale throughput | None during inference | Any |
| EP | Efficient MoE routing | AllToAll | NVLink / InfiniBand |
| TP + PP | Large model, multi-node | AllReduce + send/recv | NVLink within, IB across |
| TP + DP | Throughput + model size | AllReduce + none | NVLink |

---

## 18. Common errors and fixes

### `Total number of attention heads must be divisible by tensor parallel size`

```
ValueError: Total number of attention heads (32) must be divisible by tensor parallel size (3)
```

Fix: Choose a TP size that divides the number of attention heads. For a 32-head model, valid values are 1, 2, 4, 8, 16, 32.

### `NCCL error: unhandled cuda error, NCCL version...`

Usually a CUDA/driver version mismatch, or GPU-to-GPU communication failing. Check:
```bash
nvidia-smi topo -m   # Show NVLink/PCIe topology
# All GPUs should show NV (NVLink) or at least PIX (PCIe) connections
```

### `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

Don't call `torch.cuda` functions before initializing vLLM's distributed setup. Use `CUDA_VISIBLE_DEVICES` for GPU selection instead of `torch.cuda.set_device()`.

### Slow inference with multi-GPU (TP set too high)

With PCIe-connected GPUs, high TP causes expensive AllReduce operations. Solution: switch to PP or reduce TP size.

```bash
# Before: slow due to PCIe AllReduce overhead
--tensor-parallel-size 4

# After: faster for PCIe setups
--tensor-parallel-size 1 --pipeline-parallel-size 4
```

### Ray workers fail to start

Ensure:
- All nodes have the same Python version and vLLM version
- Model files exist at the same path on all nodes
- Firewall allows Ray's ports (6379, 8265)
- `ray status` shows all nodes before launching vLLM

---

## 19. Practice exercises

### Exercise 1 — Check what TP sizes your model supports

```python
def valid_tp_sizes(num_attention_heads: int, num_kv_heads: int, max_tp: int = 16) -> list:
    """Return valid tensor_parallel_size values for a given model."""
    valid = []
    for tp in range(1, max_tp + 1):
        # TP must divide num_attention_heads
        # Also: if tp > num_kv_heads, we need tp to be a multiple of num_kv_heads
        if num_attention_heads % tp != 0:
            continue
        if tp > num_kv_heads and tp % num_kv_heads != 0:
            continue
        valid.append(tp)
    return valid

# Llama-3.1-8B: 32 Q-heads, 8 KV-heads
print("Llama-3.1-8B:", valid_tp_sizes(32, 8))      # [1, 2, 4, 8]

# Llama-3.1-70B: 64 Q-heads, 8 KV-heads
print("Llama-3.1-70B:", valid_tp_sizes(64, 8))     # [1, 2, 4, 8]

# Mistral-7B: 32 Q-heads, 8 KV-heads
print("Mistral-7B:", valid_tp_sizes(32, 8))        # [1, 2, 4, 8]

# Falcon-40B: 64 Q-heads, 64 KV-heads (MHA, not GQA)
print("Falcon-40B:", valid_tp_sizes(64, 64))       # [1, 2, 4, 8, 16, ...]
```

### Exercise 2 — VRAM calculation for TP

```python
def vram_per_gpu_tp(
    model_params_b: float,    # billions of parameters
    dtype_bytes: int,         # 2 for bfloat16/float16, 1 for int8
    tp_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_context_len: int,
    batch_size: int = 32,
    gpu_utilization: float = 0.90,
) -> dict:
    """Estimate VRAM distribution with tensor parallelism."""
    
    # Model weights: evenly sharded across TP GPUs
    total_weight_gb = model_params_b * 1e9 * dtype_bytes / 1e9
    weights_per_gpu_gb = total_weight_gb / tp_size
    
    # KV cache per GPU: KV heads are sharded too
    kv_heads_per_gpu = max(1, num_kv_heads // tp_size)
    kv_bytes_per_token = 2 * kv_heads_per_gpu * head_dim * dtype_bytes
    kv_cache_gb = kv_bytes_per_token * max_context_len * batch_size / 1e9
    
    # CUDA/framework overhead
    overhead_gb = 2.5  # rough estimate
    
    total_needed_gb = weights_per_gpu_gb + kv_cache_gb + overhead_gb
    
    return {
        "weights_per_gpu_gb": round(weights_per_gpu_gb, 1),
        "kv_cache_gb": round(kv_cache_gb, 1),
        "overhead_gb": overhead_gb,
        "total_needed_gb": round(total_needed_gb, 1),
        "fits_in_80gb": total_needed_gb <= 80 * gpu_utilization,
    }

# Llama-3.1-70B on 4×A100 80GB (BF16)
result = vram_per_gpu_tp(
    model_params_b=70, dtype_bytes=2,
    tp_size=4,
    num_layers=80, num_kv_heads=8, head_dim=128,
    max_context_len=8192, batch_size=32,
)
print("Llama-3.1-70B, TP=4, A100 80GB:")
for k, v in result.items():
    print(f"  {k}: {v}")
```

### Exercise 3 — Single-node TP inference

```python
from vllm import LLM, SamplingParams
import torch

if torch.cuda.device_count() >= 2:
    # Only run if you have at least 2 GPUs
    llm = LLM(
        model="facebook/opt-1.3b",   # Small model for testing
        tensor_parallel_size=2,
        dtype="float16",
    )
    
    params = SamplingParams(temperature=0.0, max_tokens=50)
    outputs = llm.generate(
        ["Tensor parallelism splits model weights across"],
        params
    )
    print(outputs[0].outputs[0].text)
else:
    print("Need at least 2 GPUs for this exercise. "
          "Conceptually: LLM(..., tensor_parallel_size=N) "
          "splits each weight matrix across N GPUs.")
```

### Exercise 4 — Measure TP scaling

```python
import time, torch
from vllm import LLM, SamplingParams

if torch.cuda.device_count() < 4:
    print("Need 4 GPUs to run this exercise.")
else:
    prompts = [f"Tell me about topic number {i}:" for i in range(20)]
    params = SamplingParams(temperature=0.7, max_tokens=100)
    
    for tp in [1, 2, 4]:
        if torch.cuda.device_count() < tp:
            continue
        
        llm = LLM(
            model="facebook/opt-1.3b",
            tensor_parallel_size=tp,
            dtype="float16",
        )
        
        # Warm-up
        llm.generate(prompts[:2], params)
        
        t0 = time.time()
        outputs = llm.generate(prompts, params)
        elapsed = time.time() - t0
        
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / elapsed
        
        print(f"TP={tp}: {elapsed:.1f}s, {throughput:.0f} tokens/sec")
        
        del llm
        torch.cuda.empty_cache()
    
    # Expected: higher TP → lower latency per request, similar or higher throughput
    # due to more total VRAM available for KV cache
```

---

## 20. Chapter summary

| Concept | Key fact |
|---------|----------|
| Tensor parallelism | Splits weight matrices within each layer across N GPUs; one AllReduce per block |
| Column-parallel | Splits output dimension; each GPU computes a shard independently |
| Row-parallel | Splits input dimension; partial outputs combined via AllReduce |
| Attention TP | Splits by attention heads; each GPU computes a head shard |
| TP constraint | `tensor_parallel_size` must divide `num_attention_heads` (and `num_kv_heads` for GQA) |
| AllReduce | Communication primitive that sums partial results; cost scales with link bandwidth |
| NVLink | ~900 GB/s on H100; makes AllReduce nearly free within a node |
| PCIe | ~128 GB/s shared; high TP on PCIe hurts throughput — prefer PP instead |
| Pipeline parallelism | Assigns consecutive layer groups to different GPUs; only point-to-point activation passing |
| Pipeline bubble | Autoregressive decoding forces one stage to be idle while another runs; unavoidable |
| TP vs PP | TP reduces latency; PP reduces memory per node but adds bubbles |
| Data parallelism | Independent full-model replicas; no inference communication; scales throughput linearly |
| Expert parallelism | Distributes MoE experts across GPUs; uses AllToAll instead of AllReduce |
| Combined TP+PP | TP within nodes (NVLink), PP across nodes (InfiniBand); the standard multi-node recipe |
| KV cache with TP | KV heads sharded across GPUs; more total KV blocks = more concurrent requests |
| Multiprocessing | Default backend; single-node only; lower overhead |
| Ray | Required for multi-node; also supports single-node |
| `--tensor-parallel-size` | Main flag; set to number of GPUs on one node |
| `--pipeline-parallel-size` | Set to number of nodes |
| `--data-parallel-size` | Native DP in vLLM V1 (v0.9.0+) |
| `--enable-expert-parallel` | MoE models only; activates only when TP×DP > 1 |

---

## What's next

Chapter 8 — **Advanced Features & Production**: speculative decoding to cut latency, LoRA for serving multiple fine-tuned adapters from one base model, prefix caching revisited, multimodal models, and production monitoring with Prometheus.

---

*Sources: [vLLM Distributed Serving docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) · [Megatron-LM paper](https://arxiv.org/abs/1909.08053) · [Red Hat Office Hours recap](https://developers.redhat.com/articles/2025/02/06/distributed-inference-with-vllm) · [ROCm MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) · [vLLM Data Parallel docs](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) · vLLM 0.19.0*
