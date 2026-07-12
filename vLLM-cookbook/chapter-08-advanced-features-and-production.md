# Chapter 8 — Advanced Features & Production

> **Track:** Production  
> **Prerequisite:** Chapters 1–7  
> **vLLM version:** 0.19.0 (April 2026)  
> **Sources:** vLLM Speculative Decoding docs, vLLM LoRA docs, vLLM Metrics/V1 design docs, Red Hat EAGLE3 blog (Jul 2025), JarvisLabs spec-decode benchmark (Dec 2025), vLLM Multi-LoRA blog (Feb 2026)

---

## Why this chapter exists

You now know how vLLM works end to end: KV cache paging, the scheduler, the OpenAI-compatible server, quantization, and distributed inference. This final chapter covers four things that separate a *working* deployment from a *production-grade* one:

1. **Speculative decoding** — cut per-token latency at low batch sizes without changing output quality
2. **LoRA serving** — serve dozens of fine-tuned adapters from one base model, per request
3. **Prefix caching revisited** — the production implications you couldn't fully appreciate until now
4. **Production monitoring** — Prometheus metrics, health checks, and the dashboards that tell you when something is wrong before your users do

Each topic is covered from first principles first, then concrete code.

---

## Table of Contents

1. [Speculative decoding — the core idea](#1-speculative-decoding--the-core-idea)
2. [Rejection sampling: why output quality is preserved](#2-rejection-sampling-why-output-quality-is-preserved)
3. [The acceptance rate: the key variable](#3-the-acceptance-rate-the-key-variable)
4. [Speculative decoding methods in vLLM](#4-speculative-decoding-methods-in-vllm)
5. [N-gram / prompt lookup decoding](#5-n-gram--prompt-lookup-decoding)
6. [EAGLE and EAGLE-3](#6-eagle-and-eagle-3)
7. [When speculative decoding helps — and when it hurts](#7-when-speculative-decoding-helps--and-when-it-hurts)
8. [LoRA — what it is and why you need to serve it efficiently](#8-lora--what-it-is-and-why-you-need-to-serve-it-efficiently)
9. [How vLLM serves multiple LoRA adapters](#9-how-vllm-serves-multiple-lora-adapters)
10. [LoRA configuration and serving](#10-lora-configuration-and-serving)
11. [Prefix caching in production](#11-prefix-caching-in-production)
12. [Production monitoring: the `/metrics` endpoint](#12-production-monitoring-the-metrics-endpoint)
13. [The essential metrics to watch](#13-the-essential-metrics-to-watch)
14. [PromQL queries for the five key signals](#14-promql-queries-for-the-five-key-signals)
15. [Health checks and readiness probes](#15-health-checks-and-readiness-probes)
16. [Benchmarking your deployment](#16-benchmarking-your-deployment)
17. [Practice exercises](#17-practice-exercises)
18. [Chapter summary and full course recap](#18-chapter-summary-and-full-course-recap)

---

## 1. Speculative decoding — the core idea

Go back to Chapter 5. The decode phase generates **one token per forward pass** of the entire model. Every step loads the full model weights from VRAM. For a 70B model at small batch sizes, you spend more time moving data (memory bandwidth bound) than doing computation.

The core insight of speculative decoding (Leviathan et al., 2023) is this: what if a small, cheap model proposed several tokens at once, and the large model verified them all in a **single forward pass**? If even some proposals are correct, you generate multiple tokens per large-model step.

Here is the mechanism precisely:

```
Standard decode — 5 tokens, 5 large-model passes:

  Pass 1:  Large model → token 1 ("The")
  Pass 2:  Large model → token 2 ("capital")
  Pass 3:  Large model → token 3 ("of")
  Pass 4:  Large model → token 4 ("France")
  Pass 5:  Large model → token 5 ("is")

  Total: 5 forward passes

Speculative decode — 5 tokens, 2 large-model passes (if draft accepts 4):

  Draft model (fast): proposes ["The", "capital", "of", "France", "is"]

  Pass 1:  Large model verifies 5 draft tokens in one pass
           → accepts ["The", "capital", "of", "France"] (4 tokens)
           → rejects "is", samples a replacement

  Draft model again: proposes next 5 tokens
  Pass 2:  Large model verifies → accepts all 5

  Total: 2 forward passes for 9 tokens ≈ 4.5 tokens per large-model step
```

The memory bandwidth cost of one large-model pass stays roughly constant whether you're verifying 1 token or 5 tokens (the extra compute is cheap relative to the weight-loading cost). So if the draft model is fast enough and accurate enough, you get 2–4× more tokens per second.

**The critical question is: does the output change?** No — and the rejection sampling mechanism is why.

---

## 2. Rejection sampling: why output quality is preserved

The large model doesn't blindly accept the draft model's tokens. It uses a mathematically rigorous **rejection sampling** procedure that ensures the final output distribution is **identical** to what you would have gotten running the large model alone.

Here is how it works for one draft token:

```python
# At position t, draft model proposed token d_t
# Large model assigns probability p_large(d_t) to that token
# Draft model assigned probability p_draft(d_t)

import random

acceptance_probability = min(1.0, p_large(d_t) / p_draft(d_t))

if random.random() < acceptance_probability:
    # Accept: token d_t is used
    pass
else:
    # Reject: sample a replacement from (p_large - p_draft) normalized
    # This correction ensures the marginal distribution is exactly p_large
    token = sample_from(max(0, p_large - p_draft))
```

Three things to note:

1. If `p_large(d_t) ≥ p_draft(d_t)` — the large model is at least as likely to pick this token as the draft → accept always.
2. If `p_large(d_t) < p_draft(d_t)` — the draft model was overconfident → accept with probability `p_large / p_draft`, and when rejecting, resample in a way that corrects for the bias.
3. The joint distribution of accepted tokens matches exactly what the large model would have produced alone.

In vLLM, speculative decoding aims to enhance inference efficiency while maintaining accuracy. Speculative decoding sampling is theoretically lossless up to the precision limits of hardware numerics. Floating-point errors might cause slight variations in output distributions — practically indistinguishable from standard decoding.

---

## 3. The acceptance rate: the key variable

The acceptance rate α is the fraction of draft tokens the large model accepts. It determines everything:

```
Expected tokens per large-model step ≈ (1 - α^(K+1)) / (1 - α)
where K = number of speculative tokens

α = 0.5, K = 5:  ≈ (1 - 0.5^6) / 0.5 ≈ 1.97 tokens/step (1.97× speedup)
α = 0.7, K = 5:  ≈ (1 - 0.7^6) / 0.3 ≈ 3.00 tokens/step (3× speedup)
α = 0.9, K = 5:  ≈ (1 - 0.9^6) / 0.1 ≈ 4.69 tokens/step (4.69× speedup)
α = 0.3, K = 5:  ≈ (1 - 0.3^6) / 0.7 ≈ 1.43 tokens/step (1.43× speedup)
```

High α requires the draft model to be highly correlated with the large model — same tokenizer vocabulary, fine-tuned on similar data, conditioned on the same context. Low α means the draft proposals are rejected frequently and the overhead of running the draft model is wasted.

vLLM (as of v0.9.1) exposes α per request via metrics: **draft acceptance rate**, **per-position acceptance rates**, and **mean acceptance length** (average tokens generated per large-model forward pass). Always inspect these when evaluating whether speculative decoding is actually helping.

---

## 4. Speculative decoding methods in vLLM

vLLM supports several speculative decoding approaches in V1 as of 0.19.0. They vary in whether they need separate model weights, how they generate proposals, and the use cases they target:

| Method | Config key | Draft weights? | Best for |
|--------|-----------|---------------|----------|
| N-gram / prompt lookup | `"ngram"` | No | Summarization, code with repetition |
| Suffix decoding | `"suffix"` | No | Code generation, agents |
| EAGLE | `"eagle"` | Yes | General chat, low-latency |
| EAGLE-3 | `"eagle3"` | Yes | General chat, highest quality |

**Important vLLM note:** Speculative decoding is not yet fully optimized in vLLM and does not always yield inter-token latency reductions for all prompt datasets or sampling parameters. The work to optimize it is ongoing. Also: speculative decoding is **not compatible with pipeline parallelism**. You can use tensor parallelism for the main model, but not PP.

---

## 5. N-gram / prompt lookup decoding

This is the simplest form: **no extra model weights needed**. Instead of a draft model, it looks for repeated n-grams in the prompt and uses them as draft tokens.

**How it works:**

```
Prompt: "The quick brown fox jumps over the lazy dog. The quick brown"
                                                        ↑
                         System scans prompt for "The quick brown" as a match
                         Finds it → proposes ["fox", "jumps", "over", ...] as draft tokens
```

It works best when the output is likely to repeat or paraphrase parts of the input: summarization, document Q&A, and code generation (especially copy-paste patterns).

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,   # propose 5 tokens per draft step
        "prompt_lookup_max": 4,        # max n-gram size to match
    },
)

params = SamplingParams(temperature=0.0, max_tokens=200)
outputs = llm.generate(["Summarize this: " + long_document], params)
```

```bash
# CLI equivalent
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
```

Measured speedups at QPS=1 on CNN/DailyMail (summarization): up to **2.8×** vs no speculation using Llama-3-70B on 4×H100. At high QPS, this advantage shrinks because the system becomes compute-bound.

---

## 6. EAGLE and EAGLE-3

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) is a speculative decoding method where the draft model is trained to predict the large model's **hidden states**, not just tokens. This makes it highly correlated with the target model.

**How EAGLE-3 works:**

The EAGLE-3 draft model takes as input:
- Token embeddings of the current sequence
- Hidden states from **three layers** of the verifier (target) model

It passes these through a lightweight MLP to predict the next token's hidden state, then samples a draft token from that. Because it has direct visibility into the verifier's internal representations, it achieves much higher acceptance rates than a standalone small model.

**In practice, EAGLE-3 is the state-of-the-art** for speculative decoding speedup on general workloads, delivering up to 2.5× speedup in production-like settings.

EAGLE-3 is supported in vLLM from v0.8.5+. v0.9.1 added CUDA graph support for EAGLE/EAGLE-3, which improves throughput.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    speculative_config={
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,  # EAGLE draft must run TP=1
    },
)

params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)
outputs = llm.generate(["Explain how transformers work:"], params)
```

```bash
# CLI: EAGLE-3 for Llama-3.3-70B with TP=4 main model
VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --seed 42 \
  --tensor-parallel-size 4 \
  --speculative-config '{"model":"yuhuili/EAGLE3-LLaMA3.3-Instruct-70B","num_speculative_tokens":3,"method":"eagle3","draft_tensor_parallel_size":1}'
```

**EAGLE draft models must run with `draft_tensor_parallel_size=1`**, even if the main model uses TP. This is a current vLLM limitation.

Available EAGLE/EAGLE-3 checkpoints on HuggingFace:
- `yuhuili/EAGLE-LLaMA3.1-Instruct-8B`
- `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`
- `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`
- `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` (via Speculators framework)

---

## 7. When speculative decoding helps — and when it hurts

This is the part most tutorials skip over. Speculative decoding is **not universally beneficial**. You need to understand the conditions:

**It helps when:**
- Batch size is small (1–8 requests) — decode is memory-bandwidth bound, so extra draft compute is "free"
- The acceptance rate α is high (>0.7) — depends heavily on task type
- Output is somewhat predictable (code, summarization, instruction following)
- You care about **inter-token latency (ITL)** / streaming feel, not just throughput

**It hurts when:**
- High QPS / large batch size — the system is already compute-bound; adding draft model overhead reduces throughput
- The acceptance rate is low (<0.4) — rejections cost time without benefit
- Using temperature=0 (greedy) with a high-quality EAGLE model — acceptance is near perfect, works great
- Creative writing with temperature=1.5 — highly random, acceptance rate drops

From benchmarks: **performance comparison shows spec decode delivering up to 1.5× speedup at QPS=1 with Llama-3-70B on ShareGPT using a draft model, and up to 2.8× on CNN/DailyMail with n-grams. However, in high-QPS environments, speculative decoding may introduce performance trade-offs. The extra compute required to propose and verify tokens can sometimes slow down the system when it is already compute-bound.**

**Practical guidance:**

```python
# Before enabling spec decode, measure your baseline
# Then enable and compare directly

# Always check acceptance rate after enabling
# vLLM exposes this in /metrics:
# vllm:spec_decode_draft_acceptance_rate
```

If you see acceptance rate below 0.5 in your production logs, either disable spec decode or switch to n-gram (which has lower overhead than EAGLE when acceptance is poor).

---

## 8. LoRA — what it is and why you need to serve it efficiently

LoRA (Low-Rank Adaptation, Hu et al., 2021) is a parameter-efficient fine-tuning method. Instead of retraining the full weight matrix W ∈ ℝ^(d×k), it trains two small matrices: A ∈ ℝ^(d×r) and B ∈ ℝ^(r×k), where r ≪ min(d, k) is the rank (typically 8–64).

During inference, the adapted output is:

```
y = x·W + x·A·B    (= x·(W + AB))
```

The base weights W stay frozen. The adapter A and B are tiny — a rank-16 adapter for a 4096×4096 weight matrix uses `4096×16 + 16×4096 = 131,072` parameters instead of `4096×4096 = 16,777,216`. That is **128× fewer parameters** for one layer.

**The serving problem:** You have one base model (say Llama-3.1-8B) and 50 fine-tuned LoRA adapters — one for SQL, one for French, one for medical notes, one per customer. The naive solution: deploy 50 separate model instances. Cost: 50× the VRAM and infrastructure.

The efficient solution: load the base model once, and dynamically merge the correct adapter per request. vLLM does exactly this.

---

## 9. How vLLM serves multiple LoRA adapters

When `enable_lora=True`, vLLM modifies each Linear layer's forward pass to:

```python
# Without LoRA:
output = input @ W

# With LoRA (adapter A, B active):
output = input @ W + (input @ A) @ B
```

The base weights W stay on GPU permanently. The adapter weights A and B are much smaller and can be swapped in and out per request in the same batch.

**Multiple adapters in one batch:** vLLM batches requests using different adapters together. In a single decode step, requests using `adapter_sql` and requests using `adapter_french` run simultaneously — each token uses its own adapter's A and B matrices applied on top of the same W.

**The LRU cache for adapters:** vLLM keeps up to `max_loras` adapters in GPU VRAM at once and uses an LRU (Least Recently Used) eviction policy. `max_cpu_loras` adapters can overflow to CPU RAM. When a request arrives using an adapter not currently on GPU, vLLM loads it from CPU → GPU (a brief latency spike). To reduce cold-start latency, warm up adapters by sending a dummy request before production traffic.

**Memory cost of adapters:** A rank-16 adapter for a 7B model adds roughly:
```
2 × (num_layers × 2 × d_model × rank × dtype_bytes)
= 2 × 32 × 2 × 4096 × 16 × 2 bytes = ~16 MB
```
Very small compared to the base model's ~14 GB. You can fit dozens of adapters on a single GPU.

---

## 10. LoRA configuration and serving

### Offline inference with one adapter

```python
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Download adapter
sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")

# Load base model with LoRA enabled
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_loras=4,            # max adapters resident on GPU simultaneously
    max_lora_rank=64,       # must be >= the rank of any adapter you load
    max_cpu_loras=8,        # adapters that can overflow to CPU RAM
)

params = SamplingParams(temperature=0.0, max_tokens=256, stop=["[/assistant]"])

outputs = llm.generate(
    [
        "[user] Write a SQL query: SELECT all users [/user] [assistant]",
        "[user] Write a SQL query: COUNT rows in table_name [/user] [assistant]",
    ],
    params,
    lora_request=LoRARequest(
        lora_name="sql_adapter",   # human-readable name
        lora_int_id=1,             # unique integer ID (used internally for caching)
        lora_path=sql_lora_path,
    ),
)

for output in outputs:
    print(output.outputs[0].text)
```

### Online serving with multiple adapters

```bash
# Start server with multiple adapters pre-loaded
vllm serve meta-llama/Llama-2-7b-hf \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --max-cpu-loras 8 \
  --lora-modules sql-lora=/path/to/sql-adapter \
  --lora-modules french-lora=/path/to/french-adapter \
  --lora-modules medical-lora=/path/to/medical-adapter
```

The adapters appear in the model list:

```bash
curl localhost:8000/v1/models | jq '.data[].id'
# "meta-llama/Llama-2-7b-hf"
# "sql-lora"
# "french-lora"
# "medical-lora"
```

Clients select an adapter just by setting `"model": "sql-lora"` in their request:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="sql-lora",   # ← selects the SQL adapter; base model weights unchanged
    messages=[{"role": "user", "content": "Write a query to get all users."}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

### Dynamic adapter loading at runtime

For production systems where adapters are added without server restart:

```bash
# Set env var to enable (comes with security warning — use in trusted env only)
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Load a new adapter
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "new_adapter", "lora_path": "/path/to/new-adapter"}'

# Unload an adapter
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "old_adapter"}'
```

**Security warning:** The vLLM docs note this feature comes with security risks and should not be used in production unless it is an isolated, fully trusted environment. In multi-tenant production, use the startup-time `--lora-modules` approach instead.

---

## 11. Prefix caching in production

We covered the mechanism in Chapter 3. Here we focus on the production implications.

**When APC delivers the most value:**

```
Use case 1 — System prompt sharing:
  Every user in your chatbot gets the same 500-token system prompt.
  Without APC: each request computes 500 tokens of prefill from scratch.
  With APC:    first request computes and caches. All subsequent hits skip it.
  Savings:     up to 500/avg_total_tokens reduction in TTFT.

Use case 2 — RAG with shared context:
  You inject a 2000-token document into every query.
  Without APC: each query runs 2000-token prefill.
  With APC:    document blocks are cached after the first request.
  Savings:     dramatic TTFT reduction for high-QPS RAG systems.

Use case 3 — Few-shot prompt templates:
  Same 800-token few-shot examples in every request.
  APC caches the few-shot blocks permanently (until evicted by LRU).
```

**Enabling and monitoring APC:**

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,   # enable APC
)
```

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-prefix-caching
```

**Measure your cache hit rate in production:**

```bash
# Prometheus metrics for APC
curl http://localhost:8000/metrics | grep prefix_cache

# vllm:prefix_cache_queries   — total queries to prefix cache
# vllm:prefix_cache_hits      — how many resulted in a hit
# vllm:gpu_prefix_cache_hit_rate  — current hit rate (gauge)
# vllm:cpu_prefix_cache_hit_rate  — CPU-side hit rate
```

A cache hit rate above 0.5 (50%) is generally good. Below 0.2 means prompts are too varied for APC to help.

**Designing prompts for high cache hit rates:**

Always put the **stable part of the prompt first**, and the variable part at the end:

```python
# Bad: variable part first → cache MISS every time
prompt = f"User name: {user_name}. " + SYSTEM_PROMPT + f"\n\nQuestion: {question}"

# Good: stable part first → cache HIT on system prompt blocks
prompt = SYSTEM_PROMPT + f"\n\nUser name: {user_name}. Question: {question}"
```

Reason: APC hashes blocks from the beginning of the sequence. If the first N tokens differ between requests, none of the blocks match.

---

## 12. Production monitoring: the `/metrics` endpoint

vLLM exposes a Prometheus-compatible `/metrics` endpoint. A Prometheus instance polls this every few seconds and stores the time series. Grafana visualises it.

```bash
# Check the raw metrics endpoint
curl http://localhost:8000/metrics

# You'll see output like:
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
# vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 12.0
#
# HELP vllm:kv_cache_usage_perc Fraction of used KV cache blocks.
# TYPE vllm:kv_cache_usage_perc gauge
# vllm:kv_cache_usage_perc{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.743
```

vLLM provides a reference Grafana dashboard in `examples/production_monitoring/`. The metrics in that dashboard are what the developers consider the most important for day-to-day operation.

**Prometheus scrape config:**

```yaml
# prometheus.yml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "vllm"
    metrics_path: /metrics
    static_configs:
      - targets: ["localhost:8000"]
```

---

## 13. The essential metrics to watch

vLLM exposes metrics grouped into three types: **Counters** (always increasing, reset on restart), **Gauges** (current value, goes up and down), **Histograms** (distribution of values in buckets, used for percentiles).

### Throughput and volume

| Metric | Type | What it tells you |
|--------|------|-------------------|
| `vllm:prompt_tokens_total` | Counter | Total prompt tokens processed since startup |
| `vllm:generation_tokens_total` | Counter | Total tokens generated — rate is your throughput |
| `vllm:request_success_total` | Counter | Finished requests by `finished_reason` (stop vs length) |

`finished_reason="length"` means requests are hitting `max_tokens` and being cut off. If this fraction is high, increase `max_tokens` or investigate whether users are sending very long requests.

### Queue and concurrency

| Metric | Type | What it tells you |
|--------|------|-------------------|
| `vllm:num_requests_running` | Gauge | Currently executing (in RUNNING state) |
| `vllm:num_requests_waiting` | Gauge | In queue waiting for KV cache |
| `vllm:num_requests_swapped` | Gauge | Preempted to CPU swap (V0 only; V1 uses RECOMPUTE) |

A growing `num_requests_waiting` while `num_requests_running` is near `max_num_seqs` means your server is at capacity. Consider scaling up.

### Memory

| Metric | Type | What it tells you |
|--------|------|-------------------|
| `vllm:kv_cache_usage_perc` | Gauge | Fraction of KV blocks in use (0–1) |
| `vllm:gpu_prefix_cache_hit_rate` | Gauge | APC hit rate |
| `vllm:num_preemptions_total` | Counter | How many requests have been preempted |

`kv_cache_usage_perc` above 0.95 means you're near KV capacity — you'll start seeing preemptions. If preemptions are frequent, you need more KV space (higher `gpu_memory_utilization`, quantization, or more GPUs).

### Latency (histograms)

| Metric | Type | What it tells you |
|--------|------|-------------------|
| `vllm:time_to_first_token_seconds` | Histogram | TTFT distribution |
| `vllm:inter_token_latency_seconds` | Histogram | ITL distribution |
| `vllm:e2e_request_latency_seconds` | Histogram | Total request latency |

These are histograms — use `histogram_quantile()` in PromQL to get percentiles.

---

## 14. PromQL queries for the five key signals

Once your Prometheus + Grafana stack is running, these queries cover the essential signals:

### Token throughput (tokens/sec)

```promql
rate(vllm:generation_tokens_total[1m])
```

### P95 time to first token

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))
)
```

### P99 inter-token latency

```promql
histogram_quantile(
  0.99,
  sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[5m]))
)
```

### KV cache utilisation

```promql
vllm:kv_cache_usage_perc
```

Alert when this exceeds 0.90 — you're about to run out of KV space.

### Request queue depth

```promql
vllm:num_requests_waiting
```

Alert when this is persistently > 0 — requests are queuing, which directly inflates TTFT.

### Prefix cache hit rate

```promql
vllm:gpu_prefix_cache_hit_rate
```

If this is low (<0.2) despite having APC enabled, your prompts aren't sharing enough prefix. Reorganise your prompt template.

---

## 15. Health checks and readiness probes

For Kubernetes and load balancers, vLLM provides two HTTP endpoints:

```bash
# Liveness check — is the server process alive?
curl http://localhost:8000/health
# Returns 200 OK when the server is running

# Models list — are models loaded and ready?
curl http://localhost:8000/v1/models
# Returns model list when engine is initialised
```

**Kubernetes readiness probe:**

```yaml
# deployment.yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60    # vLLM can take 30–120s to load weights
  periodSeconds: 10
  failureThreshold: 3

livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 120
  periodSeconds: 30
```

The `initialDelaySeconds` must be long enough for the model weights to finish loading. For large models (70B+ at BF16) this can exceed 3 minutes on slow storage.

---

## 16. Benchmarking your deployment

Before putting a vLLM deployment into production, benchmark it on **your actual workload**. vLLM ships a built-in benchmarking tool:

```bash
# Install benchmark dependencies
pip install vllm[bench]

# Benchmark against a running server
# Measures: median TTFT, P99 TTFT, median ITL, P99 ITL, throughput
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint-type openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 10    # requests/second
```

**The dataset matters.** ShareGPT is good for general chat. For code, use HumanEval or SWE-Bench. For summarisation, use CNN/DailyMail. Always benchmark on data that matches your production distribution.

**What to look for in benchmark output:**

```
Throughput: 45.3 requests/s
Median TTFT: 0.23s
P99 TTFT: 1.12s
Median ITL: 0.021s
P99 ITL: 0.045s

KV cache usage: 0.78
Preemptions: 12 (2.4%)
```

High P99 TTFT with normal median → occasional long queuing spikes. Try reducing `max_num_seqs`.  
High P99 ITL with normal median → occasional large batches. Try reducing `max_num_batched_tokens`.  
High preemption count → not enough KV cache. Try quantization, smaller `max_model_len`, or more GPU.

---

## 17. Practice exercises

### Exercise 1 — N-gram speculative decoding

```python
import os
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"

from vllm import LLM, SamplingParams
import time

# A summarization prompt where n-gram speculation helps:
# the model will often copy phrases from the document
document = """
The Amazon rainforest covers over 5.5 million square kilometres. It is home
to about 10% of all species on Earth. The forest produces 20% of the world's
oxygen and absorbs vast quantities of carbon dioxide. Deforestation threatens
to reduce the Amazon rainforest to a point where it can no longer produce
enough rain to sustain itself, leading to what scientists call a tipping point.
"""

prompt = f"Summarize the following passage:\n\n{document}\n\nSummary:"
params = SamplingParams(temperature=0.0, max_tokens=100)

# Without speculative decoding
llm_base = LLM(model="facebook/opt-125m", dtype="bfloat16", max_model_len=256)
t0 = time.time()
out_base = llm_base.generate([prompt], params)
t_base = time.time() - t0

# With n-gram speculative decoding
llm_spec = LLM(
    model="facebook/opt-125m",
    dtype="bfloat16",
    max_model_len=256,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_max": 4,
    },
)
t0 = time.time()
out_spec = llm_spec.generate([prompt], params)
t_spec = time.time() - t0

print(f"Base:     {t_base:.2f}s — {out_base[0].outputs[0].text.strip()[:80]}")
print(f"N-gram:   {t_spec:.2f}s — {out_spec[0].outputs[0].text.strip()[:80]}")
print(f"Speedup:  {t_base / t_spec:.2f}×")
```

### Exercise 2 — Multi-LoRA: per-request adapter selection

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download
import os
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"

# Download the SQL LoRA adapter (uses Llama-2-7b-hf base, but concept is the same)
# For CPU testing, skip if model is too large and just run with a small model
sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_loras=2,
    max_lora_rank=8,
    dtype="bfloat16",
)

params = SamplingParams(temperature=0.0, max_tokens=50, stop=["[/assistant]"])

# Request 1: uses the SQL adapter
sql_output = llm.generate(
    ["[user] Give me a SQL query to find all users [/user] [assistant]"],
    params,
    lora_request=LoRARequest("sql", 1, sql_lora_path),
)

# Request 2: no adapter (base model)
base_output = llm.generate(
    ["[user] Give me a SQL query to find all users [/user] [assistant]"],
    params,
)

print("SQL adapter:", sql_output[0].outputs[0].text.strip()[:100])
print("Base model:", base_output[0].outputs[0].text.strip()[:100])
```

### Exercise 3 — Query the Prometheus metrics

```python
import requests

def get_metric(metric_name: str, server: str = "http://localhost:8000") -> str:
    """Fetch a specific metric from the vLLM /metrics endpoint."""
    resp = requests.get(f"{server}/metrics")
    lines = resp.text.split("\n")
    for line in lines:
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            return line
    return f"Metric {metric_name} not found"

# Check key metrics
print(get_metric("vllm:kv_cache_usage_perc"))
print(get_metric("vllm:num_requests_running"))
print(get_metric("vllm:num_requests_waiting"))
print(get_metric("vllm:generation_tokens_total"))
```

### Exercise 4 — Design a prompt for maximum APC hit rate

```python
# Two equivalent prompts — one optimised for APC, one not

SYSTEM = "You are a helpful assistant. Always answer concisely."
CONTEXT = "The user is asking about Python programming. Be specific."

# Bad: variable user ID comes first → cache miss on every request
def bad_prompt(user_id: str, question: str) -> str:
    return f"User {user_id}: {question}\n\n{SYSTEM}\n{CONTEXT}"

# Good: stable content comes first → cache hit on SYSTEM + CONTEXT blocks
def good_prompt(user_id: str, question: str) -> str:
    return f"{SYSTEM}\n{CONTEXT}\n\nUser {user_id}: {question}"

# Demonstrate the difference:
import hashlib

def block_hash(text: str, block_size: int = 16) -> list:
    """Show which token blocks would be cache-hit candidates."""
    tokens = text.split()  # approximate tokenization
    blocks = [tokens[i:i+block_size] for i in range(0, len(tokens), block_size)]
    return [hashlib.md5(" ".join(b).encode()).hexdigest()[:8] for b in blocks]

q1 = "How do I reverse a string?"
q2 = "How do I sort a list?"
user1, user2 = "alice", "bob"

print("Bad prompt blocks (user1):", block_hash(bad_prompt(user1, q1))[:3])
print("Bad prompt blocks (user2):", block_hash(bad_prompt(user2, q2))[:3])
print("Match:", block_hash(bad_prompt(user1, q1))[:3] == block_hash(bad_prompt(user2, q2))[:3])

print("\nGood prompt blocks (user1):", block_hash(good_prompt(user1, q1))[:2])
print("Good prompt blocks (user2):", block_hash(good_prompt(user2, q2))[:2])
print("Match:", block_hash(good_prompt(user1, q1))[:2] == block_hash(good_prompt(user2, q2))[:2])
# First two blocks should match (stable SYSTEM+CONTEXT prefix)
```

### Exercise 5 — Compute theoretical speculative decoding speedup

```python
def theoretical_speedup(alpha: float, K: int) -> float:
    """
    Compute the expected speedup from speculative decoding.
    
    alpha: draft acceptance rate (0 to 1)
    K: number of speculative tokens proposed per step
    
    Returns: expected tokens per large-model forward pass
    """
    # Expected number of accepted tokens per step
    # From Chen et al. "Accelerating Large Language Model Decoding with
    # Speculative Sampling"
    accepted = (1 - alpha**(K+1)) / (1 - alpha) if alpha < 1.0 else K + 1
    return accepted

print("Theoretical tokens per large-model step:\n")
print(f"{'Alpha':>6} | K=3    | K=5    | K=8    | K=10")
print("-" * 45)
for alpha in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    results = [f"{theoretical_speedup(alpha, k):.2f}x" for k in [3, 5, 8, 10]]
    print(f"  {alpha:.2f}  | " + " | ".join(f"{r:>6}" for r in results))

# Key insight: low acceptance rate (0.3) makes speculative decoding
# barely worth it even at K=10. High acceptance (0.9) makes K=5 give 4.7x speedup.
```

---

## 18. Chapter summary and full course recap

### Chapter 8 summary

| Topic | Key fact |
|-------|----------|
| Speculative decoding | Draft model proposes K tokens; target verifies in one pass; rejection sampling preserves output distribution |
| Acceptance rate α | Determines speedup: `(1-α^(K+1))/(1-α)` tokens per step; must be >0.6 to justify overhead |
| N-gram / prompt lookup | No extra weights; matches prompt n-grams; best for summarisation, code copy patterns |
| EAGLE-3 | State-of-the-art; draft uses target's hidden states from 3 layers; up to 2.5× speedup |
| Spec decode limitations | Not PP-compatible; EAGLE draft must run TP=1; hurts high-QPS; benefits low-batch |
| LoRA math | y = xW + xAB; A ∈ ℝ^(d×r), B ∈ ℝ^(r×k); r ≪ min(d,k); 128× fewer params than full fine-tuning |
| Multi-LoRA serving | One base model on GPU; per-request adapter selection; LRU cache for GPU adapters |
| `--enable-lora` | Adds LoRA capability; `--lora-modules name=path` loads adapters; clients select via `"model"` field |
| Dynamic LoRA | `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`; use `/v1/load_lora_adapter`; development-only |
| Prefix caching (APC) | Put stable prompt parts first so block hashes match across requests |
| `/metrics` | Prometheus-compatible; all metrics prefixed `vllm:`; Counters, Gauges, Histograms |
| Must-watch metrics | `kv_cache_usage_perc`, `num_requests_waiting`, `time_to_first_token_seconds`, `inter_token_latency_seconds`, `num_preemptions_total` |
| `/health` | Liveness probe; use as Kubernetes readiness/liveness check |
| `vllm bench serve` | Built-in benchmarking tool; always benchmark on your actual dataset |

---

### Full course recap: vLLM from beginner to production

| Chapter | Core concept |
|---------|-------------|
| 1 | The KV cache memory problem; PagedAttention borrows OS paging → <4% waste |
| 2 (CPU) | `vllm-cpu`, `VLLM_CPU_KVCACHE_SPACE`, `LLM` class, `SamplingParams`, batch in one call |
| 2b (GPU) | `vllm`, VRAM formula, `gpu_memory_utilization`, `# GPU blocks` log |
| 3 | KV blocks (16 tokens), block tables (logical→physical), free pool, copy-on-write, APC |
| 4 | `vllm serve`, OpenAI API compatibility, streaming, structured outputs, tool calling |
| 5 | Continuous batching, V1 token-budget scheduler, chunked prefill, RECOMPUTE preemption |
| 6 | GPTQ (Hessian), AWQ (1% salient weights), INT8/FP8 W8A8, Marlin kernel (2.6–10.9× speedup) |
| 7 | TP splits within layers (AllReduce); PP splits across layers (pipeline bubbles); DP = replicas; NVLink vs PCIe |
| 8 | Speculative decoding, multi-LoRA, prefix caching in production, Prometheus monitoring |

The mental model that ties everything together: **KV cache is the central resource**. PagedAttention manages it efficiently. The scheduler decides who gets blocks. Continuous batching keeps the GPU saturated. Quantization shrinks the model to leave more room for KV cache. Distributed inference adds GPUs when one isn't enough. Speculative decoding makes each token cheaper. LoRA makes each model serve more use cases. And monitoring tells you when any of these break.

---

*Sources: [vLLM Speculative Decoding docs](https://docs.vllm.ai/en/v0.10.1/features/spec_decode.html) · [vLLM LoRA docs](https://docs.vllm.ai/en/stable/features/lora/) · [vLLM Metrics V1 design](https://docs.vllm.ai/en/stable/design/metrics/) · [Red Hat EAGLE-3 blog](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding) · [JarvisLabs spec-decode benchmark](https://docs.jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference) · [vLLM Multi-LoRA blog](https://blog.vllm.ai/2026/02/26/multi-lora.html) · vLLM 0.19.0*
