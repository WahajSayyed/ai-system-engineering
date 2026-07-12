# Chapter 5 — Continuous Batching & Scheduling

> **Track:** Intermediate  
> **Prerequisite:** Chapters 1–4  
> **vLLM version:** 0.19.0 (April 2026) — V1 engine (V0 was fully removed in Q3 2025)  
> **Sources:** vLLM official Optimization docs, V1 architecture blog (Sept 2025), Anatomy of vLLM blog (Sept 2025)

---

## Table of Contents

1. [The problem static batching creates](#1-the-problem-static-batching-creates)
2. [Continuous batching — the fix](#2-continuous-batching--the-fix)
3. [The engine step loop](#3-the-engine-step-loop)
4. [Request states: WAITING → RUNNING → FINISHED](#4-request-states-waiting--running--finished)
5. [The scheduler in V1: one unified token budget](#5-the-scheduler-in-v1-one-unified-token-budget)
6. [Prefill is compute-bound; decode is memory-bound](#6-prefill-is-compute-bound-decode-is-memory-bound)
7. [The prefill-decode conflict and head-of-line blocking](#7-the-prefill-decode-conflict-and-head-of-line-blocking)
8. [Chunked prefill — the solution](#8-chunked-prefill--the-solution)
9. [Preemption — when KV cache runs out](#9-preemption--when-kv-cache-runs-out)
10. [Key tuning parameters](#10-key-tuning-parameters)
11. [Latency metrics: TTFT, ITL, E2E](#11-latency-metrics-ttft-itl-e2e)
12. [The throughput–latency tradeoff](#12-the-throughputlatency-tradeoff)
13. [V1 process architecture](#13-v1-process-architecture)
14. [Practice exercises](#14-practice-exercises)
15. [Chapter summary](#15-chapter-summary)

---

## 1. The problem static batching creates

Before continuous batching existed, LLM servers used **static batching**: collect a group of requests, run the entire group through the model until every single request in the group is done, then start the next group.

```
Static batching — 4 requests in one batch:

Request A: 400 tokens to generate  ████████████████████████████████████████
Request B:  20 tokens to generate  ████
Request C:  15 tokens to generate  ███
Request D: 380 tokens to generate  ██████████████████████████████████████

                                   ├────────────────────────────────────────┤
                                   All 4 run together until ALL finish.
                                   B and C finish at step 20 but sit idle.
                                   GPU cycles wasted on padding.
```

The GPU idles on padding for B and C for ~380 steps while A and D finish. This is pure waste. Throughput tanks proportionally to variance in output length — and output length is always unpredictable.

---

## 2. Continuous batching — the fix

Continuous batching removes the static constraint. Requests join and leave the batch independently at every iteration. A request that finishes after 20 tokens frees its slot immediately. Another request takes that slot. The GPU stays busy.

```
Continuous batching — same 4 requests:

Step  1–20:  [A, B, C, D] — all running
Step 21:     B finishes → slot freed immediately → E joins
Step 22:     C finishes → slot freed → F joins
Step 21–400: [A, E, F, D] — GPU continuously occupied
```

The scheduler no longer waits for a fixed global batch to finish; it performs iteration-level scheduling so GPU work fills gaps and latency variance between requests is absorbed.

This is also called **iteration-level scheduling** or **in-flight batching** (TensorRT-LLM's term). It was introduced in the Orca paper (OSDI 2022) and is now standard in every serious inference framework.

---

## 3. The engine step loop

The heart of vLLM is a tight loop inside `EngineCore`. Every iteration of this loop is called a **step**. Each step has exactly three stages:

```
while requests_exist:
    ┌─────────────────────────────────────────────┐
    │ 1. SCHEDULE                                 │
    │    - Allocate KV blocks for new requests    │
    │    - Decide which requests run this step    │
    │    - Build {request_id: num_tokens} dict    │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │ 2. EXECUTE                                  │
    │    - One GPU forward pass                   │
    │    - Prefill new tokens + decode all active │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │ 3. UPDATE                                   │
    │    - Sample output tokens                   │
    │    - Check stop conditions                  │
    │    - Free KV blocks for finished requests   │
    │    - Stream tokens to clients               │
    └─────────────────────────────────────────────┘
```

At the core, there is a busy loop that pulls data from an internal input queue and runs an engine step. An engine step includes the scheduling and execution of one forward pass of the model.

Every step results in exactly **one new output token per active request**. There is no concept of "waiting for a request to fully complete" — the engine keeps running as long as any request is active.

---

## 4. Request states: WAITING → RUNNING → FINISHED

Every request in vLLM lives in exactly one of these states:

```
New request arrives
        │
        ▼
   [WAITING] ── in the scheduler queue, waiting for KV blocks
        │
        │  Scheduler selects it, allocates KV blocks
        ▼
   [RUNNING] ── active: prefill runs, then decode each step
        │
        │  EOS token / max_tokens / stop string hit
        ▼
  [FINISHED] ── KV blocks freed, response streamed to client
        │
        │  (if preempted — see section 9)
        ▼
   [WAITING] ── re-queued, blocks freed or recomputed later
```

The scheduler prioritizes decode requests — those already in the running queue. After that, it processes prefill requests from the waiting queue: retrieves computed blocks (for prefix caching), calls the KV-cache manager's `allocate_slots` function, then pops the request from WAITING and moves it to RUNNING.

---

## 5. The scheduler in V1: one unified token budget

V1 introduced a fundamentally cleaner scheduler design compared to V0.

vLLM V1 introduces a simple yet flexible scheduler. It removes the traditional distinction between "prefill" and "decode" phases by treating user-given prompt tokens and model-generated output tokens uniformly. Scheduling decisions are represented as a simple dictionary — `{request_id: num_tokens}` — which specifies the number of tokens to process for each request at each step.

The scheduler works within a single **token budget** per step:

```
Token budget = max_num_batched_tokens  (e.g. 8192)

Step N allocation:
  Request A (decoding): allocate 1 token     → budget: 8192 - 1   = 8191
  Request B (decoding): allocate 1 token     → budget: 8191 - 1   = 8190
  Request C (decoding): allocate 1 token     → budget: 8190 - 1   = 8189
  ...
  Request Z (decoding): allocate 1 token     → budget: 8189 - ...
  New request (prefill): allocate min(remaining_budget, prompt_length) tokens
                         → if prompt is 500 tokens and budget has 600 left,
                           schedule all 500 tokens in this step.
                         → if prompt is 1200 tokens and budget has 600 left,
                           schedule only 600 tokens (chunked prefill).
```

This representation is general enough to support chunked prefills, prefix caching, and speculative decoding. Chunked-prefill scheduling is seamlessly implemented: with a fixed token budget, the scheduler dynamically decides how many tokens to allocate to each request.

**Key difference from V0:** The V1 scheduler can mix both types of requests in the same step. In contrast, the V0 engine could only process either prefill or decode at once.

---

## 6. Prefill is compute-bound; decode is memory-bound

This distinction is fundamental to understanding why scheduling is hard.

**Prefill** processes all prompt tokens in one parallel pass. Processing N tokens at once means N × hidden_dim computations happen simultaneously — the GPU's CUDA cores are fully utilized. This is **compute-bound**: performance is limited by FLOPS (floating-point operations per second).

**Decode** generates one token at a time per request. For each decode step, the model must load its full weight matrices from VRAM and apply them to a tiny batch (often just 1–32 tokens). The bottleneck is **memory bandwidth** — how fast you can stream weights from VRAM to compute units. The compute cores sit underutilized.

```
Prefill:  [TTTTTTTTTTTTTTTTTTT]  ← many tokens, compute-bound
           All processed in parallel on CUDA cores

Decode:   [T]  [T]  [T]  [T]    ← one token per request per step
           Model weights streamed from VRAM repeatedly
           Memory bandwidth is the bottleneck
```

This has a critical implication: **running prefill and decode in the same step is actually beneficial** — the prefill fills the compute units while the decode fills the memory bandwidth pipeline. Together they better utilize the GPU's resources than either phase alone.

This is why chunked prefill (mixing prefill chunks with decode) improves GPU utilization.

---

## 7. The prefill-decode conflict and head-of-line blocking

Without chunked prefill, a large prefill request monopolizes an entire engine step.

```
Without chunked prefill — request P has a 4000-token prompt:

Step N:   P's full prefill (4000 tokens)      ← entire step consumed by P
          → all decode requests stalled for the full duration of this step

Step N+1: All decode requests resume
          → users who were mid-response see a visible pause in token generation
```

This is **head-of-line blocking**: one long request prevents all shorter requests from making progress. Without chunked prefill, we could end up with a single very long request monopolizing one engine step, disallowing other prefill requests to run. That would postpone all other requests and increase their latency.

The effect on users: if you are receiving streaming tokens from the server and someone else sends a large prompt, your token stream **pauses** visibly until that prefill completes.

---

## 8. Chunked prefill — the solution

Chunked prefill breaks large prompts into smaller pieces and interleaves them with decode steps.

Chunked prefill allows vLLM to process large prefills in smaller chunks and batch them together with decode requests. This feature helps improve both throughput and latency by better balancing compute-bound (prefill) and memory-bound (decode) operations. In V1, chunked prefill is enabled by default whenever possible.

```
With chunked prefill — same 4000-token request P, chunk size 512:

Step N:    [512 tokens of P's prefill] + [1 token each for A, B, C]
Step N+1:  [512 tokens of P's prefill] + [1 token each for A, B, C]
Step N+2:  [512 tokens of P's prefill] + [1 token each for A, B, C]
...
Step N+7:  [remaining tokens of P] + [1 token each for A, B, C]
Step N+8:  P now in decode phase → [1 token each for P, A, B, C]
```

A, B, and C continue receiving tokens every step throughout P's prefill. No visible pause.

### The scheduling order in V1 with chunked prefill

With chunked prefill enabled, the scheduling policy prioritizes decode requests. It batches all pending decode requests before scheduling any prefill operations. When there are available tokens in the `max_num_batched_tokens` budget, it schedules pending prefills. If a pending prefill request cannot fit into `max_num_batched_tokens`, it automatically chunks it.

**Priority order per step:**
1. All RUNNING requests (decode) get their 1 token allocated first
2. WAITING requests (prefill) get the remaining token budget
3. If a prefill doesn't fit, it's chunked to what remains

### Enabling chunked prefill

```python
# Python API
from vllm import LLM
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_num_batched_tokens=8192,   # token budget per step
)
# Chunked prefill is ON by default in V1

# CLI server
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-num-batched-tokens 8192

# In V1, enable via long_prefill_token_threshold for fine control
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --long-prefill-token-threshold 2048   # prompts >2048 tokens get chunked
```

---

## 9. Preemption — when KV cache runs out

Even with careful scheduling, GPU blocks can run out. This happens when many long-running requests accumulate more KV cache than the block pool can hold.

Due to the autoregressive nature of transformer architecture, there are times when KV cache space is insufficient to handle all batched requests. In such cases, vLLM can preempt requests to free up KV cache space for other requests. Preempted requests are recomputed when sufficient KV cache space becomes available again.

When you see this warning, preemption has occurred:

```
WARNING scheduler.py:1057 Sequence group 0 is preempted by
PreemptionMode.RECOMPUTE mode because there is not enough KV cache space.
This can affect the end-to-end performance.
total_cumulative_preemption_cnt=1
```

### V1 default: RECOMPUTE (not SWAP)

In vLLM V1, the default preemption mode is RECOMPUTE rather than SWAP, as recomputation has lower overhead in the V1 architecture.

In RECOMPUTE mode:
1. The preempted request's KV blocks are **freed entirely**
2. The request goes back to the WAITING queue
3. When resources free up, the request is re-prefilled from scratch

In SWAP mode (V0 default):
1. KV blocks are copied to CPU RAM over PCIe
2. Copied back to GPU when the request resumes
3. Higher latency if PCIe bandwidth is the bottleneck

### What to do if you see frequent preemptions

```bash
# Option 1: Give vLLM more VRAM for KV cache
vllm serve <model> --gpu-memory-utilization 0.95

# Option 2: Reduce how many requests can run concurrently
vllm serve <model> --max-num-seqs 128   # default is 256

# Option 3: Reduce max context length (less KV memory per request)
vllm serve <model> --max-model-len 4096

# Option 4: Use tensor parallelism (more GPUs = more total VRAM)
vllm serve <model> --tensor-parallel-size 2
```

---

## 10. Key tuning parameters

These are the parameters that directly control scheduling behaviour.

### `max_num_seqs`

Maximum number of requests that can be RUNNING simultaneously. This is the concurrency ceiling.

```bash
vllm serve <model> --max-num-seqs 256   # default
```

- Higher → more concurrent users, higher throughput, more KV memory consumed
- Lower → less KV pressure, lower tail latency, fewer preemptions

### `max_num_batched_tokens`

The token budget per engine step — the total number of tokens (prefill + decode) processed in one forward pass.

```bash
vllm serve <model> --max-num-batched-tokens 8192
```

Smaller values (e.g., 2048) achieve better inter-token latency (ITL) because there are fewer prefills slowing down decodes. Higher values achieve better time to first token (TTFT) as you can process more prefill tokens in a batch. For optimal throughput, we recommend setting `max_num_batched_tokens` > 8192, especially for smaller models on large GPUs.

| `max_num_batched_tokens` | TTFT | ITL | Throughput | When to use |
|--------------------------|------|-----|------------|-------------|
| 2048 | Slower | **Better** | Moderate | Interactive chat, low latency priority |
| 8192 | Better | Good | **High** | General purpose — recommended default |
| 32768 | Best | Slower | Highest | Batch jobs, throughput over latency |

### `max_model_len`

Hard cap on context length (prompt + output). Directly controls how large each request's KV footprint can be.

```bash
vllm serve <model> --max-model-len 4096
```

Reducing this is the most effective way to increase how many concurrent requests fit in memory.

### Scheduling policy

```bash
# Default: FCFS (First Come First Served)
vllm serve <model>

# Priority scheduling (lower number = higher priority)
vllm serve <model> --scheduling-policy priority
# Then pass "priority" field in requests via extra_body
```

---

## 11. Latency metrics: TTFT, ITL, E2E

Understanding these three metrics is essential for diagnosing performance.

**TTFT — Time To First Token**  
The time from when the client sends the request to when the first output token arrives. Dominated by: queue wait time + prefill time.

```
TTFT = queue_wait + prefill_time
```

Long TTFT means either:
- Server is overloaded (queue wait is high)
- Prompt is very long (prefill takes longer)
- `max_num_batched_tokens` is too small for your prompt lengths

**ITL — Inter-Token Latency** (also called TPOT: Time Per Output Token)  
The time between each successive output token after the first. Dominated by: one decode step per token.

```
ITL ≈ decode_step_time / num_concurrent_requests
```

High ITL means:
- Too many concurrent requests in the batch (each step covers more tokens total, taking longer)
- Prefill chunks are too large, interrupting decode steps
- GPU is memory-bandwidth saturated

**E2E — End-to-End Latency**  
The total time from request sent to full response received.

```
E2E = TTFT + (num_output_tokens × ITL)
```

**Throughput** (different from latency)  
The number of tokens generated per second across all requests. Maximised by large batch sizes, but at the cost of higher per-request latency.

---

## 12. The throughput–latency tradeoff

This is the fundamental tension in LLM serving. You cannot minimise both simultaneously.

```
Low max_num_batched_tokens / max_num_seqs:
  → Each step processes fewer tokens
  → Each step is faster
  → Lower ITL for individual requests
  → But: GPU underutilised, lower total throughput

High max_num_batched_tokens / max_num_seqs:
  → Each step processes more tokens across more requests
  → Higher GPU utilisation
  → Higher total throughput (tokens/second)
  → But: Each step takes longer → higher ITL per request
```

**Rule of thumb for choosing your defaults:**

| Workload | Recommended settings |
|----------|---------------------|
| Interactive chat (low latency priority) | `max_num_batched_tokens=2048`, `max_num_seqs=64` |
| Mixed (general API) | `max_num_batched_tokens=8192`, `max_num_seqs=256` |
| Batch processing (throughput priority) | `max_num_batched_tokens=32768`, `max_num_seqs=512` |

---

## 13. V1 process architecture

Understanding the process layout explains CPU requirements and how the engine avoids GPU idle time.

vLLM V1 uses a multi-process architecture where each process requires CPU resources. For a deployment with N GPUs, there are at minimum: 1 API server process — handles HTTP requests, tokenization, and input processing; 1 engine core process — runs the scheduler and coordinates GPU workers; N GPU worker processes — one per GPU, executes model forward passes. This means there are always at least 2 + N processes competing for CPU time. The engine core process runs a busy loop and is particularly sensitive to CPU starvation.

```
Process layout (single GPU):

┌─────────────────────┐    ZeroMQ IPC    ┌──────────────────────┐
│   API Server        │ ◄──────────────► │   EngineCore         │
│  (FastAPI/Uvicorn)  │                  │  (Scheduler + Loop)  │
│                     │                  │                      │
│ - HTTP handling     │                  │ - Scheduling         │
│ - Tokenization      │                  │ - KV block mgmt      │
│ - Response stream   │                  │ - Dispatches to GPU  │
└─────────────────────┘                  └──────────┬───────────┘
                                                     │ shared memory
                                          ┌──────────▼───────────┐
                                          │   GPU Worker (x N)   │
                                          │  (Model execution)   │
                                          │                      │
                                          │ - Forward pass       │
                                          │ - PagedAttention     │
                                          └──────────────────────┘
```

The separation of API server from EngineCore means tokenization and HTTP handling don't block the GPU. The EngineCore's busy loop runs uninterrupted, keeping the GPU saturated.

In the v0.6.0 release, vLLM introduced a multiprocessing API server utilising ZeroMQ for IPC, enabling overlap between the API server and AsyncLLM. vLLM V1 extends this by integrating the multiprocessing architecture deeper into the core of AsyncLLM, creating an isolated EngineCore execution loop that focuses exclusively on the scheduler and model executor.

---

## 14. Practice exercises

### Exercise 1 — Observe the scheduling log

```bash
# Start server with stats logging enabled
export VLLM_CPU_KVCACHE_SPACE=4
vllm serve facebook/opt-125m \
  --device cpu \
  --dtype bfloat16 \
  --max-model-len 256 \
  --disable-log-stats false \
  --port 8000

# In another terminal — send concurrent requests
python3 << 'EOF'
import threading
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def send_request(prompt, max_tokens):
    resp = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    print(f"Done: {prompt[:30]}... → {len(resp.choices[0].message.content)} chars")

# Send 5 concurrent requests with varying output lengths
threads = [
    threading.Thread(target=send_request, args=(f"Write a {n*20}-word essay about AI", n*20))
    for n in range(1, 6)
]
for t in threads: t.start()
for t in threads: t.join()
EOF
# Watch the server logs — notice how requests complete at different times
# but the server never stops processing
```

### Exercise 2 — Measure TTFT vs ITL under load

```python
import time, threading
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def measure_streaming_latency(prompt: str, max_tokens: int = 50):
    results = {"ttft": None, "itl_times": [], "start": time.time()}
    
    stream = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    
    last_token_time = None
    for chunk in stream:
        now = time.time()
        if chunk.choices[0].delta.content:
            if results["ttft"] is None:
                results["ttft"] = now - results["start"]
                last_token_time = now
            else:
                results["itl_times"].append(now - last_token_time)
                last_token_time = now
    
    avg_itl = sum(results["itl_times"]) / len(results["itl_times"]) if results["itl_times"] else 0
    print(f"TTFT: {results['ttft']*1000:.0f}ms | Avg ITL: {avg_itl*1000:.0f}ms")

# Baseline — single request
print("--- Single request ---")
measure_streaming_latency("Tell me about the history of computing.")

# Under load — send 5 simultaneous requests
print("\n--- 5 concurrent requests ---")
threads = [
    threading.Thread(target=measure_streaming_latency, 
                     args=(f"Request {i}: Tell me about topic {i}",))
    for i in range(5)
]
for t in threads: t.start()
for t in threads: t.join()
# Notice: TTFT increases under load (queue wait), ITL also increases (larger batch)
```

### Exercise 3 — Tune `max_num_batched_tokens`

```python
# This requires restarting the server with different flags each time.
# Compare throughput between two settings.

import subprocess, time
from openai import OpenAI

def benchmark(max_num_batched_tokens: int, num_requests: int = 10):
    # Start server in background (adjust for your setup)
    print(f"\n--- max_num_batched_tokens={max_num_batched_tokens} ---")
    
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
    
    prompts = [f"Explain topic {i} in detail:" for i in range(num_requests)]
    
    t0 = time.time()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as ex:
        futures = [
            ex.submit(
                client.chat.completions.create,
                model="facebook/opt-125m",
                messages=[{"role": "user", "content": p}],
                max_tokens=30,
            )
            for p in prompts
        ]
        results = [f.result() for f in futures]
    
    elapsed = time.time() - t0
    total_tokens = sum(r.usage.completion_tokens for r in results)
    print(f"Total time: {elapsed:.1f}s | Total tokens: {total_tokens}")
    print(f"Throughput: {total_tokens/elapsed:.1f} tokens/sec")

# Run with server started at different max_num_batched_tokens values
benchmark(max_num_batched_tokens=512)
benchmark(max_num_batched_tokens=8192)
```

### Exercise 4 — Simulate head-of-line blocking

```python
import time, threading
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def short_request():
    """A short request that should complete quickly."""
    start = time.time()
    resp = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=5,
    )
    elapsed = time.time() - start
    print(f"Short request done in {elapsed*1000:.0f}ms: {resp.choices[0].message.content!r}")

def long_prefill_request():
    """A request with a very long prompt — occupies a full prefill step."""
    long_prompt = "Summarise the following: " + ("word " * 200)  # ~200 token prompt
    resp = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[{"role": "user", "content": long_prompt}],
        max_tokens=20,
    )
    print(f"Long request done: {resp.choices[0].message.content[:50]!r}")

# Send long request first, then short request immediately after
# Without chunked prefill, the short request would wait for the full prefill
t_long = threading.Thread(target=long_prefill_request)
t_short = threading.Thread(target=short_request)

t_long.start()
time.sleep(0.05)   # Tiny delay so long request enters queue first
t_short.start()

t_long.join()
t_short.join()
```

---

## 15. Chapter summary

| Concept | Key fact |
|---------|----------|
| Static batching | Waits for all requests in a batch to finish; wastes GPU on padding |
| Continuous batching | Iteration-level scheduling; requests join/leave independently each step |
| Engine step | Three stages: Schedule → Execute (1 GPU forward pass) → Update |
| Request states | WAITING → RUNNING → FINISHED (or WAITING again if preempted) |
| V1 scheduler | Token budget `{request_id: num_tokens}`; mixes prefill + decode in one step |
| Prefill | Compute-bound; processes all prompt tokens in parallel |
| Decode | Memory-bandwidth-bound; one token per request per step |
| Head-of-line blocking | Long prefill monopolises a step, stalling all decode requests |
| Chunked prefill | Default in V1; breaks long prefills into chunks; decodes run alongside |
| `max_num_batched_tokens` | Token budget per step; small = lower ITL, large = higher throughput |
| `max_num_seqs` | Max concurrent RUNNING requests; default 256 |
| Preemption | V1 default is RECOMPUTE; frees KV blocks, re-queues the request |
| TTFT | Time To First Token: queue wait + prefill time |
| ITL | Inter-Token Latency: time between successive output tokens |
| V1 processes | API server + EngineCore + N GPU workers; requires 2+N CPU cores minimum |

---

## What's next

Chapter 6 — **Quantization**: now that you understand how requests flow through the engine, the next bottleneck is model weight size. Quantization reduces VRAM footprint, enabling larger models or more KV cache headroom. We'll cover GPTQ, AWQ, INT8, and FP8 with concrete examples for both CPU and GPU.

---

*Sources: [vLLM Optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/) · [V1 Architecture blog](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) · [Anatomy of vLLM](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) · vLLM 0.19.0*
