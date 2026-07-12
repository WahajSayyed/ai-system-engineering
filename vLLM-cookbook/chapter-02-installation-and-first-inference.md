# Chapter 2 — Installation & First Inference

> **Track:** Beginner  
> **Setup:** CPU laptop  
> **vLLM version:** 0.19.0 (April 2026)

---

## Table of Contents

1. [Two packages: `vllm` vs `vllm-cpu`](#1-two-packages-vllm-vs-vllm-cpu)
2. [Installation](#2-installation)
3. [CPU-specific environment variables](#3-cpu-specific-environment-variables)
4. [The `LLM` class — your inference engine](#4-the-llm-class--your-inference-engine)
5. [Key `LLM()` constructor arguments](#5-key-llm-constructor-arguments)
6. [The `SamplingParams` class — controlling generation](#6-the-samplingparams-class--controlling-generation)
7. [Running your first inference](#7-running-your-first-inference)
8. [Understanding `RequestOutput`](#8-understanding-requestoutput)
9. [Batch inference — the right way](#9-batch-inference--the-right-way)
10. [Chat models vs base models](#10-chat-models-vs-base-models)
11. [Recommended models for CPU laptops](#11-recommended-models-for-cpu-laptops)
12. [Offline vs online inference](#12-offline-vs-online-inference)
13. [What happens under the hood when you call `llm.generate()`](#13-what-happens-under-the-hood)
14. [Practice exercises](#14-practice-exercises)
15. [Chapter summary](#15-chapter-summary)

---

## 1. Two packages: `vllm` vs `vllm-cpu`

There are **two separate PyPI packages** depending on your hardware:

| Package | Hardware | Notes |
|---------|----------|-------|
| `vllm` | NVIDIA/AMD GPU | The main package. Requires CUDA/ROCm. |
| `vllm-cpu` | CPU only | Separate optimised wheel for CPU inference. Auto-detects AVX512/AVX2. |

On a CPU laptop, **use `vllm-cpu`**. It ships two shared libraries internally (`_C.so` for AVX512 and `_C_AVX2.so` for older CPUs) and picks the right one at import time — no manual config needed.

---

## 2. Installation

### CPU laptop (your setup)

```bash
# Check Python version first — vLLM requires 3.10–3.13
python --version

# Install vllm-cpu
pip install vllm-cpu

# Optional but recommended: faster memory allocation
sudo apt install libtcmalloc-minimal4        # Debian/Ubuntu
sudo dnf install gperftools-libs             # RHEL/Fedora

# Preload TCMalloc (add to your .bashrc/.zshrc for persistence)
export LD_PRELOAD=$(find /usr -name "libtcmalloc_minimal.so*" | head -1)
```

### Verify installation

```python
from vllm import LLM
print("vLLM imported successfully")
```

### GPU installation (for reference)

```bash
# Requires CUDA 12.1+ and Python 3.10–3.13
pip install vllm
```

---

## 3. CPU-specific environment variables

These variables control CPU inference behaviour. Set them before running any script or server:

```bash
# How much RAM to use for KV cache (in GB). Default is very small — always set this.
export VLLM_CPU_KVCACHE_SPACE=8      # Use 8GB of RAM for KV cache on a 16GB laptop

# Pin inference threads to specific CPU cores (improves performance)
# Find your core count: nproc
export VLLM_CPU_OMP_THREADS_BIND=0-7  # Use cores 0–7 for inference (leave 1–2 for OS)
```

**Why `VLLM_CPU_KVCACHE_SPACE` matters:** On CPU, vLLM uses RAM instead of VRAM for the KV cache. If you don't set this, vLLM defaults to a very small cache and you'll see warnings or OOM errors on longer generations.

**Practical rule of thumb for laptops:**

| Total RAM | `VLLM_CPU_KVCACHE_SPACE` |
|-----------|--------------------------|
| 8 GB | 2–3 |
| 16 GB | 6–8 |
| 32 GB | 12–16 |

---

## 4. The `LLM` class — your inference engine

The `LLM` class is the primary entrypoint for **offline inference** — meaning you run it directly in Python code, not through an HTTP server.

```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
```

When you instantiate `LLM`, it does all of the following:
1. Downloads the model weights from HuggingFace Hub (cached in `~/.cache/huggingface/`)
2. Loads the tokenizer
3. Initialises the vLLM engine with PagedAttention memory management
4. Allocates the KV cache pool in RAM (CPU) or VRAM (GPU)

> **Note:** `LLM` is for offline/batch use. For serving requests over HTTP, you use `AsyncLLMEngine` (covered in Chapter 4).

---

## 5. Key `LLM()` constructor arguments

```python
llm = LLM(
    model="facebook/opt-125m",   # HuggingFace model ID or local path
    dtype="bfloat16",            # Data type for weights and activations
    max_model_len=1024,          # Max total context length (prompt + output)
    trust_remote_code=False,     # Allow executing model's custom Python code
    tokenizer=None,              # Override tokenizer (defaults to same as model)
    seed=42,                     # Random seed for reproducibility
    download_dir=None,           # Custom download directory (default: HF cache)
)
```

### The `dtype` argument in depth

This is the most important argument for CPU users:

| `dtype` value | Memory per param | CPU support | Use when |
|---------------|-----------------|-------------|----------|
| `"float32"` | 4 bytes | ✅ Always | Maximum precision, most RAM |
| `"bfloat16"` | 2 bytes | ✅ Most modern CPUs | **Recommended for CPU** |
| `"float16"` | 2 bytes | ⚠️ Unstable on CPU | Avoid on CPU — use bfloat16 |
| `"auto"` | depends | ✅ | Reads model config, may pick float16 |

> **CPU rule:** Always explicitly set `dtype="bfloat16"`. `float16` has known instability issues on CPU torch backends. `auto` may silently choose float16.

### The `max_model_len` argument

Every model has a default maximum context length (e.g., OPT-125m supports 2048 tokens). On a laptop with limited RAM, you can reduce this to lower memory usage:

```python
# Reduce context window to save RAM
llm = LLM(model="facebook/opt-125m", max_model_len=512, dtype="bfloat16")
```

---

## 6. The `SamplingParams` class — controlling generation

`SamplingParams` controls *how* the model generates text. You pass it alongside your prompts.

```python
from vllm import SamplingParams

params = SamplingParams(
    temperature=0.8,      # Randomness: 0.0 = deterministic, 1.0+ = creative/chaotic
    top_p=0.95,           # Nucleus sampling: consider only top 95% probability mass
    top_k=50,             # Only sample from top-50 tokens at each step
    max_tokens=200,       # Maximum number of tokens to generate
    stop=["###", "\n\n"], # Stop generation when these strings appear
    n=1,                  # Number of output sequences per prompt
    seed=None,            # Seed for this specific generation (overrides LLM seed)
    presence_penalty=0.0, # Penalise tokens that have appeared at all (+ve = less repeat)
    frequency_penalty=0.0,# Penalise tokens proportional to how often they appeared
    repetition_penalty=1.0,# >1.0 reduces repetition (1.0 = no effect)
)
```

### Intuition for the key sampling parameters

**`temperature`** scales the probability distribution before sampling:

```
temperature = 0.0  → Always pick the highest-probability token (greedy). Deterministic.
temperature = 0.7  → Slightly more varied. Good for factual tasks.
temperature = 1.0  → Sample from the raw distribution. Balanced.
temperature = 1.5  → More creative/unpredictable. Good for creative writing.
temperature > 2.0  → Often incoherent.
```

**`top_p` (nucleus sampling):** At each step, sort all tokens by probability, then keep only the smallest set of tokens whose cumulative probability reaches `top_p`. Only sample from those. This cuts off the long tail of unlikely tokens.

**`top_k`:** Simpler version — just keep the top K tokens by probability.

**`max_tokens`:** Hard cap on output length. Does **not** include the input/prompt tokens. Keep this low on CPU to avoid slow generation.

**`stop` sequences:** Generation halts as soon as one of these strings appears in the output. Useful for structured outputs or conversation formatting.

**Greedy decoding** (no randomness, fastest, most deterministic):

```python
params = SamplingParams(temperature=0.0, max_tokens=100)
```

---

## 7. Running your first inference

```python
import os
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"  # 4GB KV cache for laptop

from vllm import LLM, SamplingParams

# 1. Load the model
llm = LLM(
    model="facebook/opt-125m",
    dtype="bfloat16",
    max_model_len=512,
)

# 2. Define sampling parameters
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

# 3. Run inference
outputs = llm.generate(["The capital of France is"], params)

# 4. Print results
for output in outputs:
    print(f"Prompt:    {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

**Expected output (approximate):**

```
Prompt:    'The capital of France is'
Generated: ' Paris. Paris is the most visited city in the world...'
```

> **First run note:** The model (~250MB for opt-125m) will be downloaded to `~/.cache/huggingface/` on first run. Subsequent runs are instant.

---

## 8. Understanding `RequestOutput`

`llm.generate()` returns a list of `RequestOutput` objects — one per input prompt. Here's the full structure:

```python
output = outputs[0]

output.request_id          # Unique ID for this request
output.prompt              # The original input string
output.prompt_token_ids    # Token IDs of the prompt
output.outputs             # List of CompletionOutput (one per n= setting)

# Each CompletionOutput:
completion = output.outputs[0]
completion.text            # The generated text string
completion.token_ids       # Token IDs of generated text
completion.cumulative_logprob  # Log probability of the generation
completion.finish_reason   # "stop", "length", or "abort"
```

**`finish_reason` values:**

| Value | Meaning |
|-------|---------|
| `"stop"` | Generation hit a stop string or EOS token |
| `"length"` | Hit `max_tokens` limit — output may be truncated |
| `"abort"` | Request was cancelled |

Always check `finish_reason` in production — `"length"` means the output was cut short.

---

## 9. Batch inference — the right way

The entire point of vLLM is **high-throughput batching**. Pass all your prompts in a single call:

```python
import os
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"

from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m", dtype="bfloat16", max_model_len=512)
params = SamplingParams(temperature=0.7, max_tokens=60)

# Pass ALL prompts in one call — vLLM batches them intelligently
prompts = [
    "The capital of France is",
    "Water boils at",
    "The speed of light is approximately",
    "Python was created by",
]

outputs = llm.generate(prompts, params)

for output in outputs:
    print(f">>> {output.prompt}")
    print(f"    {output.outputs[0].text.strip()}")
    print()
```

> **Important:** Do NOT call `llm.generate()` in a loop for each prompt individually. That defeats the entire purpose of vLLM and gives you near-zero throughput benefit. Always batch.

### Per-prompt sampling parameters

You can pass different `SamplingParams` for each prompt:

```python
params_list = [
    SamplingParams(temperature=0.0, max_tokens=20),   # Deterministic for prompt 0
    SamplingParams(temperature=1.2, max_tokens=100),  # Creative for prompt 1
    SamplingParams(temperature=0.7, max_tokens=50),   # Balanced for prompt 2
    SamplingParams(temperature=0.7, max_tokens=50),
]

outputs = llm.generate(prompts, params_list)
```

---

## 10. Chat models vs base models

There are two types of models you'll encounter:

**Base models** (`facebook/opt-125m`, `Qwen/Qwen2.5-0.5B`) complete text directly. Pass raw text:

```python
outputs = llm.generate(["The capital of France is"], params)
```

**Chat/instruction models** (`Qwen/Qwen2.5-0.5B-Instruct`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) expect a specific conversation format. Use `llm.chat()` instead:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

outputs = llm.chat(messages, params)
print(outputs[0].outputs[0].text)
```

`llm.chat()` handles the chat template formatting (applying `<|user|>`, `[INST]`, etc.) automatically based on the model's tokenizer config.

---

## 11. Recommended models for CPU laptops

| Model | Size | RAM needed | Good for |
|-------|------|-----------|----------|
| `facebook/opt-125m` | 250 MB | ~1 GB | Fastest test — low quality |
| `Qwen/Qwen2.5-0.5B` | 1 GB | ~2 GB | Good quality for size |
| `Qwen/Qwen2.5-0.5B-Instruct` | 1 GB | ~2 GB | Chat/instruction following |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2.2 GB | ~4 GB | Best small chat model |
| `Qwen/Qwen2.5-1.5B-Instruct` | 3 GB | ~6 GB | Strong quality for CPU |
| `microsoft/phi-2` | 5.5 GB | ~10 GB | High quality, needs more RAM |

**Start with `facebook/opt-125m`** to verify your installation works, then move to `Qwen/Qwen2.5-0.5B-Instruct` for real experiments.

---

## 12. Offline vs online inference

This chapter covers **offline inference** — the `LLM` class runs blocking inference in your Python process.

| | Offline (`LLM`) | Online (`vllm serve`) |
|---|---|---|
| Use case | Scripts, batch jobs, experiments | HTTP API, serving multiple users |
| Interface | Python function call | REST API (OpenAI-compatible) |
| Concurrency | Single process, one batch at a time | Async, handles concurrent requests |
| Class | `LLM` | `AsyncLLMEngine` (used internally) |

Chapter 4 covers the online server in full.

---

## 13. What happens under the hood

When you call `llm.generate(prompts, params)`, here's what vLLM does:

```
1. Tokenise all prompts                     → convert strings to token ID lists
2. Create a Request object per prompt       → assign request_id, store params
3. Add all requests to the scheduler queue  → WAITING state
4. Scheduler loop:
   a. Scheduler selects a batch of requests
   b. Allocates physical KV cache blocks    → from the free block pool
   c. Runs prefill for new requests         → builds KV cache
   d. Runs decode step for all active reqs  → generates one token each
   e. Checks stop conditions               → finish_reason set if done
   f. Frees KV blocks for completed reqs   → back to free pool
5. Collect all RequestOutput objects        → return to caller
```

The key insight: even with multiple prompts, vLLM's scheduler interleaves their prefill and decode steps using PagedAttention's non-contiguous block allocation — which is why batching gets you higher throughput.

---

## 14. Practice exercises

### Exercise 1 — Verify installation
```python
import os
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "2"

from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m", dtype="bfloat16", max_model_len=256)
out = llm.generate(["Hello, I am"], SamplingParams(max_tokens=20))
print(out[0].outputs[0].text)
# If this runs without error, your installation is working.
```

### Exercise 2 — Explore temperature
```python
prompt = ["The future of artificial intelligence will"]
llm = LLM(model="facebook/opt-125m", dtype="bfloat16", max_model_len=256)

for temp in [0.0, 0.5, 1.0, 1.5]:
    params = SamplingParams(temperature=temp, max_tokens=30, seed=42)
    out = llm.generate(prompt, params)
    print(f"temp={temp}: {out[0].outputs[0].text.strip()[:80]}")
```

### Exercise 3 — Batch vs single call
```python
import time

prompts = [f"Tell me a fact about the number {i}:" for i in range(1, 9)]
params = SamplingParams(temperature=0.7, max_tokens=30)

# Batch (correct way)
t0 = time.time()
outputs = llm.generate(prompts, params)
print(f"Batch: {time.time() - t0:.2f}s for {len(prompts)} prompts")

# Single (wrong way — for comparison only)
t0 = time.time()
for p in prompts:
    llm.generate([p], params)
print(f"Loop:  {time.time() - t0:.2f}s for {len(prompts)} prompts")
# You'll see batch is significantly faster.
```

### Exercise 4 — finish_reason check
```python
params_short = SamplingParams(temperature=0.7, max_tokens=5)  # Very short limit
params_long  = SamplingParams(temperature=0.7, max_tokens=200)

for params, label in [(params_short, "short"), (params_long, "long")]:
    out = llm.generate(["The history of computers began"], params)
    completion = out[0].outputs[0]
    print(f"{label}: finish_reason={completion.finish_reason!r}, text={completion.text[:60]!r}")
```

---

## 15. Chapter summary

| Concept | Key point |
|---------|-----------|
| Package | Use `vllm-cpu` on a laptop, `vllm` on GPU |
| `VLLM_CPU_KVCACHE_SPACE` | Always set this — controls how much RAM is used for KV cache |
| `dtype` | Use `"bfloat16"` on CPU — `"float16"` is unstable |
| `LLM` class | Offline inference only; loads model + engine + KV cache pool |
| `SamplingParams` | Controls randomness (`temperature`), length (`max_tokens`), stopping |
| `llm.generate()` | Always pass ALL prompts in one call for best throughput |
| `RequestOutput` | Check `finish_reason` — `"length"` means output was cut short |
| `llm.chat()` | For chat/instruction models — handles template formatting |
| Offline vs online | `LLM` = Python API; `vllm serve` = HTTP server (Chapter 4) |

---

## What's next

Chapter 3 dives into **PagedAttention** — now that you've run your first inference, you'll understand *exactly* what vLLM was doing with memory during those `llm.generate()` calls.

---

*vLLM version: 0.19.0 | Python: 3.10–3.13 | Docs: https://docs.vllm.ai*
