# vLLM — Beginner to Advanced Learning Plan

> **Version:** Based on vLLM `0.19.0` (April 2026)  
> **Setup:** CPU laptop (GPU concepts covered for deep understanding)  
> **Goal:** Master LLM inference and serving with vLLM — from first principles to production

---

## Table of Contents

- [Overview](#overview)
- [Chapter 1 — What is vLLM?](#chapter-1--what-is-vllm)
- [Chapter 2 — Installation & First Inference](#chapter-2--installation--first-inference)
- [Chapter 3 — PagedAttention Deep Dive](#chapter-3--pagedattention-deep-dive)
- [Chapter 4 — Serving with the OpenAI-Compatible API](#chapter-4--serving-with-the-openai-compatible-api)
- [Chapter 5 — Continuous Batching & Scheduling](#chapter-5--continuous-batching--scheduling)
- [Chapter 6 — Quantization](#chapter-6--quantization)
- [Chapter 7 — Distributed Inference](#chapter-7--distributed-inference)
- [Chapter 8 — Advanced Features & Production](#chapter-8--advanced-features--production)
- [Environment Notes](#environment-notes)
- [Resources](#resources)

---

## Overview

| Track | Chapters | Focus |
|-------|----------|-------|
| 🟣 Beginner | 1 – 3 | Concepts, installation, PagedAttention |
| 🟢 Intermediate | 4 – 5 | Serving, batching, scheduling |
| 🟠 Advanced | 6 – 7 | Quantization, distributed inference |
| 🔵 Production | 8 | Speculative decoding, LoRA, monitoring |

---

## Chapter 1 — What is vLLM?

**Track:** Beginner  
**Topics:**
- Why LLM inference is slow and expensive by default
- The KV cache problem: what it is and why memory gets wasted (60–80% waste in naive systems)
- Three types of memory fragmentation: internal, external, reservation
- What vLLM solves and how it achieves <4% KV cache memory waste
- vLLM's origins: UC Berkeley Sky Computing Lab (2023 SOSP paper)
- vLLM vs HuggingFace Transformers vs TGI — when to use what
- Current state: vLLM `0.19.0`, Python 3.10–3.13, Apache 2.0 license

**Key Concepts:**
- Prefill phase vs decode phase in LLM inference
- Autoregressive token generation
- GPU memory layout for LLMs (weights + activations + KV cache)

**Practice:**
- Read the original PagedAttention paper abstract
- Understand throughput vs latency tradeoffs conceptually
- Sketch (on paper) how memory is wasted in naive KV cache allocation

---

## Chapter 2 — Installation & First Inference

**Track:** Beginner  
**Topics:**
- Installation: `pip install vllm` (CPU mode for laptop)
- CPU vs GPU installation differences
- The `LLM` class — offline batch inference
- `SamplingParams` — controlling temperature, top_p, max_tokens
- Running a small model (e.g. `facebook/opt-125m` or a quantized Qwen/Llama)
- Streaming outputs with `AsyncLLMEngine`
- Understanding model loading time and memory footprint

**Key Concepts:**
- Offline inference vs online serving
- Model loading from HuggingFace Hub
- Tokenization pipeline inside vLLM

**Practice:**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(["Explain what a GPU is in one sentence."], params)
print(outputs[0].outputs[0].text)
```

**CPU-specific notes:**
- Use `--device cpu` flag or `device="cpu"` in `LLM()`
- Expect slower inference — this is normal; concepts transfer directly to GPU
- Stick to small models (125M–1.5B parameters) for local testing

---

## Chapter 3 — PagedAttention Deep Dive

**Track:** Beginner  
**Topics:**
- The OS virtual memory analogy: pages, page tables, page frames
- KV blocks: fixed-size units of key-value cache storage
- Block tables: how each request maps logical blocks to physical GPU memory
- Non-contiguous memory storage — how attention still works correctly
- Copy-on-write for beam search and parallel sampling
- Prefix caching (prompt caching): sharing KV blocks across requests with the same prefix
- How PagedAttention enables larger batch sizes → higher throughput

**Key Concepts:**
- KV cache = keys and values from the attention mechanism stored per token per layer
- Block size (e.g. 16 tokens/block) and its effect on granularity
- Free block pool and block reuse when a request finishes
- Why <4% memory waste is achieved vs 60–80% in naive systems

**Practice:**
- Trace through a 3-request example: allocate blocks, generate tokens, free blocks
- Compare memory layout diagrams: contiguous (old) vs paged (vLLM)
- Run two identical prefix requests and observe cache hit behavior (GPU users)

---

## Chapter 4 — Serving with the OpenAI-Compatible API

**Track:** Intermediate  
**Topics:**
- Starting the server: `vllm serve <model>`
- Key CLI flags: `--host`, `--port`, `--max-model-len`, `--device`, `--dtype`
- OpenAI-compatible endpoints: `/v1/completions`, `/v1/chat/completions`, `/v1/models`
- Using the OpenAI Python SDK against vLLM
- Tool calling / function calling support
- Sampling parameters via API: `temperature`, `top_p`, `top_k`, `stop`, `n`
- Structured outputs (JSON mode)
- Async requests and streaming (`stream=True`)

**Key Concepts:**
- Why OpenAI-compatibility matters for ecosystem integration
- LangChain / LlamaIndex integration via vLLM's OpenAI endpoint
- Chat template handling per model family (Llama, Mistral, Qwen, etc.)

**Practice:**
```bash
# Start the server
vllm serve facebook/opt-125m --device cpu --port 8000

# Query it
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "facebook/opt-125m", "prompt": "Paris is the capital of", "max_tokens": 10}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(response.choices[0].message.content)
```

---

## Chapter 5 — Continuous Batching & Scheduling

**Track:** Intermediate  
**Topics:**
- Static batching (old approach) and its GPU utilization problem
- Continuous batching: adding new requests mid-batch at the token level
- The scheduler: how vLLM decides which requests run each step
- Prefill vs decode separation and scheduling tradeoffs
- Chunked prefill: splitting large prompts to avoid blocking decode
- Preemption: what happens when GPU memory runs out (swap vs recompute)
- Request priority and fairness

**Key Concepts:**
- Why one slow/long request shouldn't block short requests (head-of-line blocking)
- Iteration-level scheduling vs request-level scheduling
- How throughput and latency are in tension — and how vLLM balances them
- `max_num_seqs` and `max_num_batched_tokens` configuration

**Practice:**
- Send 10 concurrent requests to your local server and observe ordering
- Experiment with `--max-num-seqs` and observe memory/throughput changes
- Understand the scheduler logs with `--log-level debug`

---

## Chapter 6 — Quantization

**Track:** Advanced  
**Topics:**
- Why quantization: fit larger models in less VRAM, faster matrix multiplications
- Weight-only quantization vs weight+activation quantization
- GPTQ: post-training quantization via second-order error minimization
- AWQ (Activation-aware Weight Quantization): protecting salient weights
- AutoRound: newer alternative to GPTQ
- INT4, INT8 formats and their hardware support
- FP8 (8-bit floating point): Hopper GPU support (H100, H200)
- Loading quantized models from HuggingFace (e.g. `TheBloke`, `neuralmagic`)
- Accuracy vs speed vs memory tradeoffs — benchmarking methodology

**Key Concepts:**
- Calibration datasets for quantization
- Perplexity as a quality metric for quantized models
- `--quantization` flag in vLLM: `gptq`, `awq`, `fp8`
- Why quantization helps more on GPU than CPU

**Practice:**
```python
# Load a pre-quantized GPTQ model
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16"
)
```

```bash
# Serve a quantized model
vllm serve neuralmagic/Llama-3.2-1B-Instruct-quantized.w8a8 \
  --quantization compressed-tensors
```

- Compare outputs of full-precision vs quantized model on the same prompts
- Measure tokens/second difference

---

## Chapter 7 — Distributed Inference

**Track:** Advanced  
**Topics:**
- Why single GPU isn't enough for large models (70B+ parameters)
- Tensor parallelism (TP): splitting weight matrices across GPUs
- Pipeline parallelism (PP): splitting layers across GPUs
- Data parallelism: independent replicas for throughput scaling
- Expert parallelism (EP): for Mixture-of-Experts models (DeepSeek, Mixtral)
- Multi-node inference with Ray
- `tensor_parallel_size` and `pipeline_parallel_size` configuration
- Communication overhead: NCCL, all-reduce operations
- When to use TP vs PP vs combination

**Key Concepts:**
- Model sharding strategies and their memory/latency tradeoffs
- Why TP reduces latency but PP adds pipeline bubbles
- `--tensor-parallel-size N` requires N GPUs
- Ray cluster setup for multi-node

**Practice (conceptual for CPU users):**
```python
# GPU example (for reference / cloud use)
llm = LLM(
    model="meta-llama/Llama-3-70b-hf",
    tensor_parallel_size=4,  # splits across 4 GPUs
    dtype="float16"
)
```

```bash
# CLI equivalent
vllm serve meta-llama/Meta-Llama-3-70B \
  --tensor-parallel-size 4 \
  --dtype float16
```

- Understand the math: a 70B model at float16 = ~140GB → needs 2× A100 80GB minimum
- Trace through how a single forward pass is split across 4 GPUs with TP

---

## Chapter 8 — Advanced Features & Production

**Track:** Production  
**Topics:**

### Speculative Decoding
- Draft model + verification model approach
- EAGLE speculative decoding (state-of-the-art, 2025)
- N-gram speculative decoding (no draft model needed)
- When speculative decoding helps (low batch sizes, high latency budgets)
- `--speculative-model` and related flags

### LoRA — Serving Fine-Tuned Adapters
- What LoRA is and why it's useful for serving
- Loading multiple LoRA adapters on one base model
- Dynamic LoRA loading at request time
- `--enable-lora` and `--max-loras` configuration

### Multimodal Models
- Vision-language models: LLaVA, Qwen-VL, Gemma 4
- Passing images in API requests
- Video input support (experimental)

### Prefix Caching (KV Cache Reuse)
- Automatic prefix caching: `--enable-prefix-caching`
- Use cases: system prompts, RAG context, few-shot examples
- Cache hit rate monitoring

### Production Monitoring
- Prometheus metrics endpoint (`/metrics`)
- Key metrics: `vllm:inter_token_latency_seconds`, `vllm:request_success_total`
- Logging and tracing with OpenTelemetry
- Health check endpoint: `/health`
- Docker deployment
- Kubernetes + KServe deployment patterns

**Practice:**
```bash
# Enable prefix caching
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --enable-prefix-caching \
  --device cpu

# Check metrics
curl http://localhost:8000/metrics
```

---

## Environment Notes

### CPU Laptop Setup (Your Config)
```bash
# Install vLLM for CPU
pip install vllm

# Recommended small models for CPU testing
# - facebook/opt-125m       (251MB, fast)
# - Qwen/Qwen2.5-0.5B       (1GB, good quality)
# - microsoft/phi-2          (5.5GB, great quality)
```

### GPU Concepts (For Deep Understanding)
Even on CPU, these GPU concepts are essential to know:
| Concept | Why it matters |
|---------|----------------|
| VRAM (GPU memory) | KV cache + weights must fit here |
| CUDA cores / tensor cores | Parallelism units that vLLM exploits |
| Memory bandwidth (GB/s) | Bottleneck for decode phase |
| FLOPs | Bottleneck for prefill phase |
| NVLink / PCIe | Multi-GPU communication speed |

### Recommended Cloud GPU for Hands-On GPU Practice
- Google Colab Free (T4, 16GB) — good for models up to 7B
- RunPod / Lambda Labs — A100 40GB for larger experiments
- Vast.ai — affordable spot instances

---

## Resources

| Resource | Link |
|----------|------|
| Official Docs | https://docs.vllm.ai |
| GitHub | https://github.com/vllm-project/vllm |
| Blog | https://blog.vllm.ai |
| Original Paper | [PagedAttention — SOSP 2023](https://arxiv.org/abs/2309.06180) |
| PyPI | https://pypi.org/project/vllm/ |
| Discord/Slack | https://slack.vllm.ai |
| Model Hub | https://huggingface.co/models?library=vllm |

---

## Progress Tracker

- [ ] Chapter 1 — What is vLLM?
- [ ] Chapter 2 — Installation & First Inference
- [ ] Chapter 3 — PagedAttention Deep Dive
- [ ] Chapter 4 — Serving with the OpenAI-Compatible API
- [ ] Chapter 5 — Continuous Batching & Scheduling
- [ ] Chapter 6 — Quantization
- [ ] Chapter 7 — Distributed Inference
- [ ] Chapter 8 — Advanced Features & Production

---

*Last updated: April 2026 | vLLM version: 0.19.0*
