# Chapter 2b — Installation & First Inference (GPU)

> **Track:** Beginner  
> **Setup:** NVIDIA GPU (covers cloud GPUs, local NVIDIA, Google Colab)  
> **vLLM version:** 0.19.0 (April 2026)  
> **Companion to:** Chapter 2 (CPU) — the Python API, `LLM` class, `SamplingParams`, and `RequestOutput` are identical. This chapter covers GPU-specific installation, VRAM sizing, and GPU-only configuration.

---

## Table of Contents

1. [GPU requirements](#1-gpu-requirements)
2. [VRAM sizing — will your model fit?](#2-vram-sizing--will-your-model-fit)
3. [Installation](#3-installation)
4. [Verify GPU is visible to vLLM](#4-verify-gpu-is-visible-to-vllm)
5. [GPU-specific `LLM()` arguments](#5-gpu-specific-llm-arguments)
6. [The `gpu_memory_utilization` parameter in depth](#6-the-gpu_memory_utilization-parameter-in-depth)
7. [How vLLM allocates GPU memory at startup](#7-how-vllm-allocates-gpu-memory-at-startup)
8. [Your first GPU inference](#8-your-first-gpu-inference)
9. [Model recommendations by GPU VRAM](#9-model-recommendations-by-gpu-vram)
10. [Cloud GPU quick-start (Colab / RunPod)](#10-cloud-gpu-quick-start-colab--runpod)
11. [Common GPU errors and fixes](#11-common-gpu-errors-and-fixes)
12. [GPU vs CPU: what actually changes](#12-gpu-vs-cpu-what-actually-changes)
13. [Practice exercises](#13-practice-exercises)
14. [Chapter summary](#14-chapter-summary)

---

## 1. GPU requirements

### Minimum supported hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU architecture | Volta (SM 7.0) — e.g. V100 | Ampere or newer (A100, RTX 3090+) |
| VRAM | 8 GB | 16–80 GB |
| CUDA version | 12.1 | 12.4 (default in vLLM 0.19.0) |
| Driver version | 525.xx | 535.xx+ |
| OS | Linux (primary) | Ubuntu 20.04 / 22.04 |
| Python | 3.10 | 3.11 or 3.12 |

> **Windows note:** vLLM does not natively support Windows. Use **WSL2** (Windows Subsystem for Linux) or Docker.  
> **macOS note:** No CUDA support on Mac. Use the CPU build or a cloud GPU.

### Common GPU tiers for vLLM

| GPU | VRAM | Typical use |
|-----|------|-------------|
| RTX 3060 / 4060 | 8–12 GB | Small models (≤7B), experiments |
| RTX 3090 / 4090 | 24 GB | 7B–13B models comfortably |
| A10G | 24 GB | Cloud workloads, 7B–13B |
| L40S | 48 GB | 13B–34B models |
| A100 40GB | 40 GB | 13B–30B, production |
| A100 80GB | 80 GB | 70B models, production |
| H100 80GB | 80 GB | Highest throughput, 70B+ |

---

## 2. VRAM sizing — will your model fit?

Before downloading anything, calculate whether your model fits. This is the most common source of frustration for beginners.

### The formula

```
VRAM needed ≈ (Parameters × Bytes per parameter) + KV cache + ~2–4 GB overhead
```

### Bytes per parameter by dtype

| dtype | Bytes/param | 7B model | 13B model | 70B model |
|-------|-------------|----------|-----------|-----------|
| `float32` | 4 | 28 GB | 52 GB | 280 GB |
| `float16` | 2 | 14 GB | 26 GB | 140 GB |
| `bfloat16` | 2 | 14 GB | 26 GB | 140 GB |
| INT8 | 1 | 7 GB | 13 GB | 70 GB |
| INT4 | 0.5 | 3.5 GB | 6.5 GB | 35 GB |

**Rule of thumb:** `model_VRAM_GB ≈ parameters_B × 2` at float16/bfloat16.

### KV cache on top

After weights load, vLLM uses whatever VRAM is left for KV cache. More KV cache = more concurrent requests. This is controlled by `gpu_memory_utilization` (see section 6).

### Quick fit check

```python
import subprocess

# Check your GPU VRAM
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True
)
print(result.stdout)
# e.g: NVIDIA A100-SXM4-40GB, 40960 MiB
```

### Safety rule

Always reserve **2–4 GB** of VRAM for CUDA runtime, PyTorch overhead, and activations. If your calculation leaves less than 2 GB free after weights, the server will crash.

**Example:** RTX 3090 (24 GB) + Llama-3.1-8B at bfloat16:
```
Weights:   8B × 2 bytes = 16 GB
Overhead:  ~2 GB
KV cache:  24 - 16 - 2 = 6 GB remaining → ~20 concurrent requests at 1K tokens
```
✅ Fits comfortably.

**Example:** RTX 3060 (8 GB) + Llama-3.1-8B at bfloat16:
```
Weights:   8B × 2 bytes = 16 GB
Available: 8 GB
```
❌ Does not fit. Use a quantized model (INT4 ~4 GB) or a smaller model.

---

## 3. Installation

### Step 1 — Create a fresh Python environment (strongly recommended)

vLLM bundles its own PyTorch with custom CUDA kernels. Mixing with an existing PyTorch install causes subtle breakage.

```bash
# Using conda
conda create -n vllm python=3.11 -y
conda activate vllm

# Or using venv
python3.11 -m venv vllm-env
source vllm-env/bin/activate
```

### Step 2 — Check CUDA version

```bash
nvidia-smi           # Shows driver version and CUDA version in top-right corner
nvcc --version       # Shows CUDA Toolkit version (may differ from driver CUDA)
```

vLLM 0.19.0 ships with **CUDA 12.4** by default. This works with any driver ≥ 525.xx.

### Step 3 — Install vLLM

```bash
# Standard install (CUDA 12.4 — works for most setups)
pip install vllm

# If you need a specific CUDA version (e.g. CUDA 11.8 on older machines):
export VLLM_VERSION=0.19.0
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp311-cp311-manylinux1_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Using uv (faster alternative to pip)
uv pip install vllm
```

### Step 4 — Install FlashAttention (optional but recommended for Ampere+)

FlashAttention 2 significantly reduces memory usage and improves speed on Ampere GPUs (A100, RTX 3090, 4090) and newer.

```bash
pip install flash-attn --no-build-isolation
```

vLLM will automatically use FlashAttention when it detects it is available.

---

## 4. Verify GPU is visible to vLLM

```python
import torch

# Check CUDA is available
print(torch.cuda.is_available())          # True
print(torch.cuda.device_count())          # Number of GPUs
print(torch.cuda.get_device_name(0))      # e.g. "NVIDIA A100-SXM4-40GB"

total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Total VRAM: {total_vram:.1f} GB")
```

If `torch.cuda.is_available()` returns `False`:
- Driver not installed or outdated → update NVIDIA driver
- Running in WSL2 without GPU passthrough → check WSL2 CUDA setup
- Wrong Python environment → check you're in the right venv/conda env

---

## 5. GPU-specific `LLM()` arguments

All arguments from Chapter 2 (CPU) apply. These are the **GPU-only** additions:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",

    # --- GPU-specific ---
    dtype="bfloat16",               # bfloat16 recommended on Ampere+; float16 on older GPUs
    gpu_memory_utilization=0.90,    # Fraction of VRAM to use (default 0.90 = 90%)
    max_model_len=4096,             # Limit context to save KV cache VRAM
    tensor_parallel_size=1,         # Number of GPUs to shard model across
    enforce_eager=False,            # False = use CUDA graphs (faster); True = eager (safer)

    # --- Common (same as CPU) ---
    trust_remote_code=False,
    seed=42,
)
```

### `dtype` on GPU

| dtype | When to use |
|-------|-------------|
| `"bfloat16"` | **Default for Ampere+ GPUs** (A100, RTX 30/40 series, H100). Better numerical stability. |
| `"float16"` | Older Volta/Turing GPUs (V100, T4, RTX 20xx). Also used with AWQ quantization. |
| `"float32"` | Never use for inference — doubles memory, no quality gain. |
| `"auto"` | Reads model config. Usually picks float16, which is fine for most GPU setups. |

---

## 6. The `gpu_memory_utilization` parameter in depth

This is the most important GPU-specific parameter. It controls how much of your GPU's VRAM vLLM can use.

```python
llm = LLM(model="...", gpu_memory_utilization=0.90)
#                                               ^^^
#                              90% of total VRAM is available to vLLM
#                              10% reserved for CUDA/PyTorch overhead
```

**What vLLM does with that 90%:**
1. Loads model weights first (fixed cost)
2. Whatever VRAM remains → allocated to the KV cache block pool

```
Total VRAM:           40 GB (A100)
× gpu_memory_util:    × 0.90
= Available to vLLM:  36 GB

- Model weights:      - 16 GB  (8B model at bfloat16)
= KV cache budget:    = 20 GB  → large number of concurrent requests
```

### When to change `gpu_memory_utilization`

| Situation | Action |
|-----------|--------|
| OOM error on startup | Lower to 0.80 or 0.75 |
| Sharing GPU with other processes | Lower to 0.60–0.70 |
| Dedicated inference GPU | Keep at 0.90 (default) or raise to 0.95 |
| Multi-model on one GPU | Sum of all instances must not exceed 1.0 |

> **Important:** Lowering `gpu_memory_utilization` reduces KV cache size, which means fewer concurrent requests and lower throughput. It does NOT affect model quality.

---

## 7. How vLLM allocates GPU memory at startup

Understanding the startup sequence explains many "why is vLLM slow to start?" and OOM questions.

```
Step 1 — Profile run (dry run)
   vLLM runs a dummy forward pass with max sequence length
   to measure peak activation memory usage.

Step 2 — Weight loading
   Model weights loaded from disk to VRAM.
   Progress shown: "Loading model weights: 100%"

Step 3 — KV cache allocation
   vLLM calculates: available VRAM - weights - activation buffer
   Divides remainder into fixed-size KV blocks.
   Logs: "# GPU blocks: 2048, # CPU blocks: 512"

Step 4 — CUDA graph capture (if enforce_eager=False)
   Pre-captures common batch sizes as CUDA graphs for speed.
   This is why startup takes 30–120 seconds on first run.

Step 5 — Ready
   "Uvicorn running on http://0.0.0.0:8000"
```

The log line `# GPU blocks: N` directly tells you your KV cache capacity. More blocks = more concurrent users.

```bash
# See this during startup:
INFO: # GPU blocks: 2048, # CPU blocks: 512
# Each block = 16 tokens × num_layers × hidden_size bytes
# 2048 blocks × 16 tokens = 32,768 tokens of KV cache capacity
```

---

## 8. Your first GPU inference

```python
from vllm import LLM, SamplingParams

# Load model — vLLM auto-detects GPU
llm = LLM(
    model="facebook/opt-125m",     # Small model to test installation
    dtype="float16",               # OPT uses float16
    gpu_memory_utilization=0.85,
)

params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

outputs = llm.generate(
    ["The capital of France is", "Machine learning is"],
    params
)

for output in outputs:
    print(f"Prompt:    {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text.strip()!r}")
    print(f"Reason:    {output.outputs[0].finish_reason}")
    print()
```

### Serving a 7B instruction model (RTX 3090 / A100 / Colab T4)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",              # Ampere+: bfloat16
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)

params = SamplingParams(temperature=0.7, max_tokens=256)

# Using llm.chat() for instruction models
messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": "Explain PagedAttention in 3 sentences."},
]

outputs = llm.chat([messages], params)
print(outputs[0].outputs[0].text)
```

### Batch inference at scale (the GPU advantage)

On GPU, batching truly shines — all requests run in parallel on thousands of CUDA cores:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
params = SamplingParams(temperature=0.7, max_tokens=100)

# 50 prompts processed as one batch
prompts = [f"Write one sentence about topic #{i}:" for i in range(50)]
outputs = llm.generate(prompts, params)

print(f"Generated {len(outputs)} responses")
for o in outputs[:3]:
    print(f"  {o.outputs[0].text.strip()[:80]}")
```

---

## 9. Model recommendations by GPU VRAM

### 8 GB VRAM (RTX 3060, RTX 4060, Tesla T4)

```python
# Option A: Small model, full precision
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", dtype="float16")

# Option B: 7B model with INT4 quantization (fits in ~4 GB weights)
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    quantization="gptq",
    dtype="float16",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)
```

### 16 GB VRAM (RTX 3080, RTX 4080)

```python
# 7B model comfortably, 13B quantized
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
)
```

### 24 GB VRAM (RTX 3090, RTX 4090, A10G)

```python
# 13B models or 7B with large context
llm = LLM(
    model="Qwen/Qwen2.5-14B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_model_len=16384,
)
```

### 40–80 GB VRAM (A100, H100)

```python
# 70B models natively
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.92,
    max_model_len=32768,
)
```

### 70B on two GPUs (multi-GPU with tensor parallelism)

```python
# Requires 2× GPU with NVLink or PCIe
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=2,   # Split model across 2 GPUs
    gpu_memory_utilization=0.90,
)
```

> Tensor parallelism is covered in depth in Chapter 7. The key point: `tensor_parallel_size=N` requires exactly N GPUs.

---

## 10. Cloud GPU quick-start (Colab / RunPod)

### Google Colab (T4 — free tier, 16 GB VRAM)

```python
# Run in a Colab cell
!pip install vllm -q

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",                # T4 is Turing — use float16, not bfloat16
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

params = SamplingParams(temperature=0.7, max_tokens=200)
outputs = llm.generate(["Tell me about vLLM in 3 sentences:"], params)
print(outputs[0].outputs[0].text)
```

> **T4 note:** The T4 is a Turing architecture GPU. Use `dtype="float16"` — bfloat16 is not supported on Turing.

### RunPod / Lambda Labs (A100 / H100)

```bash
# SSH into your instance, then:
pip install vllm

# Download a model (recommended for large models — avoids re-downloading)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ./models/llama-3.1-8b

python3 << 'EOF'
from vllm import LLM, SamplingParams

llm = LLM(
    model="./models/llama-3.1-8b",
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
)
params = SamplingParams(temperature=0.7, max_tokens=100)
out = llm.generate(["Hello, what can you do?"], params)
print(out[0].outputs[0].text)
EOF
```

### Controlling which GPU(s) vLLM uses

```bash
# Single GPU — use GPU 0 only
export CUDA_VISIBLE_DEVICES=0
python your_script.py

# Use GPU 1 only (on a multi-GPU machine)
export CUDA_VISIBLE_DEVICES=1
python your_script.py

# Use GPUs 0 and 1 for tensor parallelism
export CUDA_VISIBLE_DEVICES=0,1
python your_script.py  # with tensor_parallel_size=2
```

---

## 11. Common GPU errors and fixes

### OOM: `torch.cuda.OutOfMemoryError` on startup

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate X GiB (GPU 0; 24.00 GiB total capacity; ...)
```

**Fixes (try in order):**

```python
# Fix 1: Reduce gpu_memory_utilization
llm = LLM(model="...", gpu_memory_utilization=0.80)

# Fix 2: Reduce max context length
llm = LLM(model="...", max_model_len=2048)

# Fix 3: Use a quantized model
llm = LLM(model="TheBloke/Llama-2-7B-GPTQ", quantization="gptq", dtype="float16")

# Fix 4: Use a smaller model
```

### OOM: `torch.cuda.OutOfMemoryError` during generation

This means your `max_model_len` is set but the KV cache still fills up under concurrent load. Reduce `max_model_len` or `gpu_memory_utilization`.

### `CUDA error: device-side assert triggered`

Usually a tokenizer issue — a token ID out of range. Make sure you're using the correct tokenizer for your model (`trust_remote_code=True` may be needed for some models).

### `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

Do not call any `torch.cuda` functions before creating the `LLM` object. If you need to check GPU info first, do it in a separate process or use `subprocess`.

```python
# WRONG — causes CUDA re-init error
import torch
torch.cuda.set_device(0)   # Don't do this before LLM()
llm = LLM(model="...")

# RIGHT — control GPU via environment variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM
llm = LLM(model="...")
```

### Model weights not found / 403 error from HuggingFace

Some models (Llama, Gemma) require accepting a license on HuggingFace first:

```bash
# Login to HuggingFace CLI
pip install huggingface_hub
huggingface-cli login      # Paste your HF token

# Then download
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

---

## 12. GPU vs CPU: what actually changes

The Python API (`LLM`, `SamplingParams`, `RequestOutput`, `llm.generate()`, `llm.chat()`) is **identical** between GPU and CPU. The differences are purely in setup and configuration:

| Aspect | CPU (`vllm-cpu`) | GPU (`vllm`) |
|--------|-----------------|--------------|
| Package | `pip install vllm-cpu` | `pip install vllm` |
| Memory type | RAM | VRAM |
| Memory config | `VLLM_CPU_KVCACHE_SPACE` env var | `gpu_memory_utilization` param |
| Recommended dtype | `bfloat16` | `bfloat16` (Ampere+) or `float16` (older) |
| Throughput | Tokens/sec × 10–100 slower | High throughput |
| CUDA graphs | Not applicable | Auto-enabled (`enforce_eager=False`) |
| Batch size ceiling | RAM / model size | VRAM / model size |
| Multi-device | Not typical | `tensor_parallel_size=N` |

The **concepts** (PagedAttention, continuous batching, sampling, batching strategy) are the same. GPU just executes them orders of magnitude faster.

---

## 13. Practice exercises

### Exercise 1 — VRAM check before loading

```python
import torch
import subprocess

# Check available VRAM
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
print(result.stdout)

# Calculate if a 7B model fits at bfloat16
params_b = 7       # billion parameters
bytes_per_param = 2  # bfloat16
weights_gb = params_b * bytes_per_param
overhead_gb = 3
print(f"7B model needs ~{weights_gb + overhead_gb} GB VRAM (weights + overhead)")
```

### Exercise 2 — Observe `# GPU blocks` in logs

```python
import logging
logging.basicConfig(level=logging.INFO)

from vllm import LLM
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.90)
# Watch for: INFO: # GPU blocks: XXXX

llm2 = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.50)
# Compare GPU blocks — lower utilization = fewer blocks = less KV cache
```

### Exercise 3 — Throughput comparison

```python
import time
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.85)
params = SamplingParams(temperature=0.7, max_tokens=50)
prompts = [f"Sentence number {i} starts with" for i in range(100)]

t0 = time.time()
outputs = llm.generate(prompts, params)
elapsed = time.time() - t0

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"Generated {total_tokens} tokens in {elapsed:.1f}s")
print(f"Throughput: {total_tokens / elapsed:.0f} tokens/sec")
```

### Exercise 4 — `gpu_memory_utilization` impact on blocks

```python
from vllm import LLM

for util in [0.5, 0.7, 0.9]:
    import subprocess, sys
    # Run in subprocess to avoid CUDA re-init
    code = f"""
from vllm import LLM
import logging
logging.basicConfig(level=logging.INFO)
llm = LLM(model='facebook/opt-125m', gpu_memory_utilization={util})
"""
    print(f"\n--- gpu_memory_utilization={util} ---")
    subprocess.run([sys.executable, "-c", code])
    # Compare "# GPU blocks" in output
```

---

## 14. Chapter summary

| Concept | Key point |
|---------|-----------|
| GPU requirement | Compute capability ≥ 7.0 (Volta), CUDA 12.1+, 8 GB+ VRAM |
| VRAM formula | `params_B × 2` GB for bfloat16/float16 weights + 2–4 GB overhead |
| Installation | `pip install vllm` in a fresh environment |
| dtype | `bfloat16` for Ampere+ (A100, RTX 30/40 series); `float16` for Turing (T4, V100) |
| `gpu_memory_utilization` | Controls VRAM fraction for vLLM; remainder stays free for OS/other processes |
| Startup sequence | Profile → weights → KV blocks → CUDA graphs → ready |
| `# GPU blocks` log | More blocks = more KV cache capacity = more concurrent users |
| Multi-GPU | `tensor_parallel_size=N` (details in Chapter 7) |
| `CUDA_VISIBLE_DEVICES` | Control which GPU(s) vLLM sees |
| API compatibility | Same Python API as CPU — `LLM`, `SamplingParams`, `llm.generate()` unchanged |

---

## What's next

Chapter 3 dives into **PagedAttention** — now that you've run inference on either CPU or GPU, you'll understand exactly what vLLM was doing with those KV blocks during your `llm.generate()` calls. The chapter explains why `# GPU blocks: 2048` matters and what it enables at scale.

---

*vLLM version: 0.19.0 | GPU docs: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/*
