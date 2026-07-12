# Chapter 6 — Quantization

> **Track:** Advanced  
> **Prerequisite:** Chapters 1–5  
> **vLLM version:** 0.19.0 (April 2026)  
> **Sources:** vLLM quantization docs, LLM Compressor docs, original AWQ/GPTQ papers, GPUStack benchmarks, JarvisLabs benchmarks

---

## Table of Contents

1. [Why quantization matters for serving](#1-why-quantization-matters-for-serving)
2. [Core vocabulary: W, A, bits, granularity](#2-core-vocabulary-w-a-bits-granularity)
3. [How quantization works mechanically](#3-how-quantization-works-mechanically)
4. [Calibration: static vs dynamic](#4-calibration-static-vs-dynamic)
5. [GPTQ — second-order weight quantization](#5-gptq--second-order-weight-quantization)
6. [AWQ — activation-aware weight quantization](#6-awq--activation-aware-weight-quantization)
7. [INT8 W8A8 — weights and activations at 8-bit](#7-int8-w8a8--weights-and-activations-at-8-bit)
8. [FP8 W8A8 — floating-point 8-bit](#8-fp8-w8a8--floating-point-8-bit)
9. [KV cache quantization](#9-kv-cache-quantization)
10. [Inference kernels: where real speedups come from](#10-inference-kernels-where-real-speedups-come-from)
11. [Hardware compatibility matrix](#11-hardware-compatibility-matrix)
12. [Loading pre-quantized models from HuggingFace](#12-loading-pre-quantized-models-from-huggingface)
13. [Quantizing your own model with LLM Compressor](#13-quantizing-your-own-model-with-llm-compressor)
14. [Choosing the right quantization method](#14-choosing-the-right-quantization-method)
15. [Measuring quality after quantization](#15-measuring-quality-after-quantization)
16. [Practice exercises](#16-practice-exercises)
17. [Chapter summary](#17-chapter-summary)

---

## 1. Why quantization matters for serving

Every chapter so far has shown that VRAM is the primary constraint in LLM serving. Quantization attacks that constraint directly by reducing the number of bits used to represent model weights, and optionally activations too.

The numbers are straightforward:

| Precision | Bytes per parameter | 7B model | 13B model | 70B model |
|-----------|--------------------|---------:|----------:|----------:|
| FP32 | 4 | 28 GB | 52 GB | 280 GB |
| BF16 / FP16 | 2 | **14 GB** | **26 GB** | **140 GB** |
| INT8 / FP8 | 1 | 7 GB | 13 GB | 70 GB |
| INT4 | 0.5 | 3.5 GB | 6.5 GB | 35 GB |

Going from BF16 to INT4 halves the weight memory twice over. A 70B model that required two A100 80GB GPUs at BF16 fits on a single A100 80GB at INT4 — with room left for KV cache.

There are two distinct benefits:

**Smaller model → more VRAM for KV cache.** With the same GPU, you fit more KV blocks, serve more concurrent users, and achieve higher throughput.

**Faster weight loading.** The decode phase (Chapter 5) is memory-bandwidth-bound: the GPU must stream model weights from VRAM on every step. Smaller weights stream faster. INT4 weights load 4× faster than FP16, directly increasing decode throughput.

However, quantization introduces **approximation error**. The question in every deployment is: how much quality are you willing to trade for which performance gain?

---

## 2. Core vocabulary: W, A, bits, granularity

Before diving into specific methods, the notation used throughout this chapter:

**W and A notation:**
- **W4A16** = Weights quantized to 4-bit, Activations kept at 16-bit (weight-only quantization)
- **W8A8** = Weights quantized to 8-bit, Activations quantized to 8-bit (weight+activation quantization)
- **W8A16** = Weights at 8-bit, Activations at 16-bit

**Granularity** — the scope over which a single scale factor applies:
- **Per-tensor**: one scale for the entire weight matrix. Simplest, least accurate.
- **Per-channel**: one scale per output channel (row of the weight matrix). More accurate.
- **Per-group**: one scale per group of `group_size` contiguous weights (e.g. 128). Most accurate for weight-only quantization.

**Group size**: the number of weights sharing one scale. Common values: 128 (default), 64, 32. Smaller group = more scales to store = larger overhead but better accuracy.

**Scale and zero-point**: every quantization maps a floating-point range to integer values using:
```
quantized = round(float / scale) + zero_point
dequantized = (quantized - zero_point) × scale
```
Symmetric quantization: `zero_point = 0`. Asymmetric: `zero_point ≠ 0`.

---

## 3. How quantization works mechanically

At its simplest, quantizing a weight matrix to INT4 means:

```python
# Original weight (FP16):
w = tensor([-0.32, 0.14, -1.87, 0.92, ...])   # float16, 2 bytes each

# Step 1: Find range
w_min, w_max = w.min(), w.max()   # e.g. -2.1, 1.4

# Step 2: Compute scale
scale = (w_max - w_min) / (2^4 - 1)   # = 3.5 / 15 = 0.233

# Step 3: Quantize → integers in [0, 15]
w_int4 = round((w - w_min) / scale)   # now stored as 4-bit integers

# Step 4: At inference time, dequantize before compute
w_fp16 = w_int4 * scale + w_min       # back to float for matmul
```

The key insight: you store the weight at 4 bits but **dequantize to FP16 before computing**. This is why W4A16 quantization saves memory (weights are small) but has a dequantization cost at inference time.

W8A8 is different — both weights and activations are quantized to 8-bit and the **matrix multiply itself runs at INT8**. This avoids the dequantization step and leverages INT8 tensor cores directly.

---

## 4. Calibration: static vs dynamic

Both activation and KV cache quantization require knowing the range of values at runtime. There are two strategies:

**Static (offline) quantization**: Run a small representative dataset (100–512 samples) through the model during quantization. Record min/max activation values. Bake the scales into the saved model. At inference time, use those pre-computed scales — no overhead.

**Dynamic (online) quantization**: Compute the scale for each activation tensor on the fly during every forward pass. More accurate (adapts to actual input), but adds compute overhead per step.

**Which to use:**

| Scenario | Recommendation |
|----------|---------------|
| Production serving, known workload distribution | Static — lower overhead, predictable |
| Research or unknown input distribution | Dynamic — adapts to anything |
| FP8 weight-only (no activation calibration) | Dynamic activation — no calibration data needed |

Weight quantization (GPTQ, AWQ) always requires calibration data regardless of static/dynamic activation choice.

---

## 5. GPTQ — second-order weight quantization

**What it is:** A post-training weight-only quantization algorithm that uses second-order information (the Hessian matrix of the loss with respect to weights) to minimize quantization error. Introduced in "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022).

**The core idea:** Not all weights are equally sensitive to quantization. GPTQ computes the Hessian to understand how quantizing each weight affects the model's output. It then quantizes weights one at a time, updating the remaining unquantized weights to compensate for the error introduced. The LLM Compressor GPTQ modifier "uses activations to calibrate a Hessian matrix, which is then used to determine optimal quantization values and orderings for the model weights."

**Key properties:**
- Weight-only: W4A16 or W8A16 (activations stay FP16)
- Requires calibration data (typically 128–512 samples from C4 or WikiText)
- Produces per-group scales (group_size=128 is standard)
- No backpropagation — one forward pass per layer
- Takes 1–4 hours for a 7B model on a single GPU

**Supported formats in vLLM:**
- GPTQ (original format)
- GPTQ with Marlin kernel (recommended — see section 10)
- Dynamic per-module GPTQ via GPTQModel

**Loading in vLLM:**

```python
from vllm import LLM, SamplingParams

# vLLM auto-detects quantization from the model's config.json
llm = LLM(model="TheBloke/Llama-2-7B-GPTQ")

# Or be explicit:
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16",        # GPTQ requires float16, not bfloat16
)
```

**Quantizing your own model with GPTQModel:**

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=2)  # increase batch_size if VRAM allows
model.save("Llama-3.2-1B-Instruct-gptq-4bit")
```

---

## 6. AWQ — activation-aware weight quantization

**What it is:** A hardware-friendly weight-only quantization method introduced in "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., MLSys 2024). AWQ finds that not all weights in an LLM are equally important. Protecting only 1% of salient weights can greatly reduce quantization error. To identify salient weight channels, we should refer to the activation distribution, not the weights themselves.

**The core idea:** During a calibration pass, AWQ records which weight channels consistently produce large-magnitude activations. These are the "salient" channels — they matter most to model quality. Instead of quantizing them with mixed precision (which is hardware-inefficient), AWQ mathematically derives per-channel scaling factors that effectively amplify salient channels before quantization and shrink them after, preserving accuracy while keeping all weights at the same bit-width.

AWQ does not rely on any backpropagation or reconstruction, so it generalizes to different domains and modalities without overfitting the calibration set.

**Key properties:**
- Weight-only: W4A16 (standard), W8A16
- Requires small calibration dataset (128+ samples)
- Per-channel scaling factors protect the 1% of salient weights
- Faster calibration than GPTQ (no Hessian computation)
- Excellent generalization across domains (does not overfit calibration set)

**Loading in vLLM:**

```python
from vllm import LLM

# vLLM auto-detects AWQ from config.json
llm = LLM(model="TheBloke/Llama-2-7B-AWQ")

# Explicit:
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="float16",        # AWQ also requires float16
)
```

**Quantizing with LLM Compressor (recommended for new models):**

Note: AutoAWQ (casper-hansen/AutoAWQ) was archived in May 2025 and is no longer maintained. The recommended replacement is vllm-project/llm-compressor.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
]

from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")

oneshot(model=model, dataset=ds, recipe=recipe)
model.save_pretrained("Meta-Llama-3-8B-Instruct-AWQ-W4A16")
tokenizer.save_pretrained("Meta-Llama-3-8B-Instruct-AWQ-W4A16")
```

---

## 7. INT8 W8A8 — weights and activations at 8-bit

**What it is:** Both weights and activations are quantized to INT8. The matrix multiply itself runs at INT8 precision using hardware INT8 tensor cores, which avoids the dequantization overhead of W4A16/W8A16.

**The core idea:** INT8 matrix multiplications are natively supported on modern GPUs and are significantly faster than FP16 on high-throughput workloads. The challenge is that LLM activations are harder to quantize than weights because they exhibit "outliers" — a small number of activation values that are much larger than the rest, making the quantization range hard to set.

INT8 computation is supported on NVIDIA GPUs with compute capability > 7.5 (Turing, Ampere, Ada Lovelace, Hopper). Note: INT8 is not supported on Blackwell (compute capability >= 10.0). Use FP8 instead on those GPUs.

**Calibration requirement:** Because activations are quantized, you need calibration data to estimate activation ranges. SmoothQuant is commonly applied before GPTQ to tame activation outliers.

**Loading in vLLM:**

```python
# If the model already has quantization_config in config.json (compressed-tensors format):
from vllm import LLM
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8")
# vLLM reads quantization_config automatically — no --quantization flag needed

# Or from a locally quantized model:
llm = LLM(model="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")
```

**Quantizing with LLM Compressor (W8A8 with SmoothQuant + GPTQ):**

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# SmoothQuant shifts quantization difficulty from activations to weights
# GPTQ then quantizes both weights and activations to INT8
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

model.save_pretrained("Meta-Llama-3-8B-Instruct-W8A8", save_compressed=True)
tokenizer.save_pretrained("Meta-Llama-3-8B-Instruct-W8A8")
```

---

## 8. FP8 W8A8 — floating-point 8-bit

**What it is:** Weights and activations quantized to 8-bit floating-point (FP8) instead of integer. FP8 comes in two formats:
- **E4M3** (1 sign + 4 exponent + 3 mantissa bits): higher precision, range up to ±448
- **E5M2** (1 sign + 5 exponent + 2 mantissa bits): wider range, less precision

vLLM supports FP8 (8-bit floating point) weight and activation quantization using hardware acceleration on Hopper (H100) and Ada Lovelace (RTX 4090, L40S) GPUs. For W8A8, only Hopper and Ada Lovelace are officially supported. Turing and Ampere GPUs are supported for W8A16 (weight-only FP8) using Marlin kernels.

Quantization of models with FP8 allows for a 2× reduction in model memory requirements and up to a 1.6× improvement in throughput with minimal impact on accuracy.

**Dynamic FP8 (no calibration needed):**

This is the simplest entry point to FP8. Dynamic quantization of a BF16/FP16 model to FP8 can be achieved with vLLM without any calibration data required.

```bash
# Enable FP8 on the fly at serving time — no pre-quantized model needed
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --dtype bfloat16
```

```python
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8",       # Weights quantized to FP8_E4M3 per-tensor
                              # Activations quantized dynamically per-tensor
    dtype="bfloat16",
)
```

In this mode, all Linear modules (except lm_head) have their weights quantized to FP8_E4M3 with a per-tensor scale. Activations have their min/max calculated during each forward pass for a dynamic per-tensor scale. Because of the dynamic activation scaling, latency improvements are limited in this mode compared to static FP8.

**Static FP8 (with calibration — better performance):**

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# ... load model and tokenizer ...

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",   # weights: FP8, activations: dynamic FP8
    ignore=["lm_head"],
)

oneshot(model=model, recipe=recipe)   # No calibration data needed for dynamic
model.save_pretrained("Meta-Llama-3-8B-Instruct-FP8-Dynamic")
```

**Loading pre-quantized FP8 models:**

```python
# Models quantized with compressed-tensors format (most NeuralMagic models)
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8")
# vLLM reads quantization_config from config.json automatically
```

---

## 9. KV cache quantization

Beyond quantizing model weights, you can also quantize the KV cache itself. This reduces the memory consumed by PagedAttention blocks, allowing more blocks (more concurrent requests) for the same VRAM.

vLLM supports FP8 datatypes (E4M3 and E5M2) for KV cache quantization, but not INT8. FP8 halves KV cache memory versus FP16.

```python
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8",           # quantize KV cache to FP8
    gpu_memory_utilization=0.90,
)
```

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90
```

**Hardware requirement:** KV cache FP8 quantization is supported on Hopper (H100, H200) and Ada Lovelace (RTX 4090, L40S) GPUs.

**Important caveat from the docs:** While research has explored 4-bit or 2-bit KV cache quantization, these approaches typically cause noticeable accuracy degradation. KV cache quantization provides relatively modest throughput improvements compared to weight quantization — it's usually not the first optimization to reach for.

---

## 10. Inference kernels: where real speedups come from

This is the most important section if you care about actual throughput numbers. **The quantization algorithm (GPTQ vs AWQ) is only half the story. The inference kernel determines the actual throughput.**

### The problem with naive dequantization

Standard W4A16 inference loads quantized (INT4) weights from HBM, dequantizes them to FP16, performs matrix multiplication, and repeats. The problem is a memory bandwidth bottleneck: compute units wait for data from HBM.

### Marlin kernel (Ampere/Hopper — A100+, H100+)

Marlin is a mixed-precision kernel developed by NeuralMagic (now Red Hat) and integrated into vLLM. It uses asynchronous memory copy to pipeline weight loading with computation — the next block of weights loads while the current block is being processed.

Compatible GPTQ and AWQ models can leverage the Marlin and Machete vLLM custom kernels to maximize throughput for both Ampere (A100+) and Hopper (H100+) NVIDIA GPUs.

Performance impact from JarvisLabs benchmarks (Qwen2.5-32B-Instruct, H200):

| Method | Throughput | vs baseline |
|--------|-----------|-------------|
| BF16 baseline | 461 tok/s | — |
| GPTQ (no Marlin) | 276 tok/s | -40% |
| GPTQ + Marlin | 712 tok/s | +54% |
| AWQ (no Marlin) | 67 tok/s | -85% |
| AWQ + Marlin | **741 tok/s** | **+61%** |

**Kernels matter more than algorithms: Marlin kernels provide massive speedups — 2.6× for GPTQ and 10.9× for AWQ.**

### Machete kernel (Hopper — H100+)

Machete is a newer kernel optimised specifically for Hopper tensor cores. vLLM selects Marlin or Machete automatically based on hardware.

### How vLLM selects kernels automatically

vLLM selects the best available kernel automatically based on:
1. GPU compute capability (Volta/Turing/Ampere/Ada/Hopper)
2. Quantization method (GPTQ/AWQ)
3. Batch size (Marlin outperforms at larger batch sizes)

For AWQ quantization, the official AWQ kernel serves as the default, while GPTQ models utilize the ExLlamaV2 kernel by default. The Marlin and Machete kernels are also available and offer enhanced performance particularly for larger batch sizes.

You generally do not need to specify kernels manually — vLLM handles this.

---

## 11. Hardware compatibility matrix

The following is from the official vLLM quantization docs. GPU architecture names map to compute capabilities: Volta = SM 7.0 (V100), Turing = SM 7.5 (T4, RTX 2080), Ampere = SM 8.0/8.6 (A100, RTX 3090), Ada = SM 8.9 (RTX 4090, L40S), Hopper = SM 9.0 (H100, H200).

| Method | Volta | Turing | Ampere | Ada | Hopper | Notes |
|--------|-------|--------|--------|-----|--------|-------|
| GPTQ (W4A16) | ✅ | ✅ | ✅ | ✅ | ✅ | ExLlamaV2 kernel default |
| GPTQ + Marlin | ❌ | ❌ | ✅ | ✅ | ✅ | Major speedup on Ampere+ |
| AWQ (W4A16) | ✅ | ✅ | ✅ | ✅ | ✅ | Standard AWQ kernel |
| AWQ + Marlin | ❌ | ❌ | ✅ | ✅ | ✅ | 10× speedup on Ampere+ |
| INT8 W8A8 | ❌ | ✅ | ✅ | ✅ | ✅ | Not on Blackwell (SM≥10) |
| FP8 W8A8 | ❌ | ❌ | ❌ | ✅ | ✅ | Ada Lovelace + Hopper only |
| FP8 W8A16 | ❌ | ❌ | ✅ | ✅ | ✅ | Marlin kernel |
| KV cache FP8 | ❌ | ❌ | ❌ | ✅ | ✅ | Ada + Hopper |

**CPU:** Most weight-only methods (GPTQ, AWQ, INT8 W8A16) work on CPU but without dedicated kernels — expect no throughput benefit over BF16, though memory reduction still applies.

---

## 12. Loading pre-quantized models from HuggingFace

The most common workflow for most teams — use a pre-quantized checkpoint rather than quantizing yourself. TheBloke (now less active), NeuralMagic/Red Hat, and official model authors publish quantized variants for nearly every major model.

### Auto-detection (recommended)

vLLM reads `quantization_config` from the model's `config.json` automatically. For most pre-quantized HuggingFace models, you need no extra flags:

```python
from vllm import LLM

# NeuralMagic / Red Hat checkpoints (compressed-tensors format)
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8")
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8")

# GPTQ models (TheBloke format)
llm = LLM(model="TheBloke/Llama-2-7B-GPTQ")

# AWQ models
llm = LLM(model="TheBloke/Llama-2-7B-AWQ")
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ")
```

### Serving pre-quantized models

```bash
# W8A8 from NeuralMagic — no --quantization flag needed
vllm serve neuralmagic/Llama-3.1-8B-Instruct-quantized.w8a8

# GPTQ
vllm serve TheBloke/Llama-2-7B-GPTQ

# AWQ
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ

# FP8 from HuggingFace (pre-quantized)
vllm serve neuralmagic/Meta-Llama-3-8B-Instruct-FP8

# Dynamic FP8 (quantize on load — no pre-quantized model needed)
vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8
```

### Finding pre-quantized models

Search HuggingFace with suffixes: `-AWQ`, `-GPTQ`, `-FP8`, `-W8A8`, `-quantized`.

Recommended sources:
- **NeuralMagic / Red Hat** (`neuralmagic/*`): High-quality W8A8, FP8 models using compressed-tensors format
- **Qwen official** (`Qwen/Qwen*-AWQ`, `Qwen/Qwen*-GPTQ`): Official Qwen quantized releases
- **TheBloke** (`TheBloke/*`): Large GPTQ/AWQ library, though less actively maintained since mid-2024
- **HuggingFace LLM Leaderboard collections**: Often include quantized variants

---

## 13. Quantizing your own model with LLM Compressor

LLM Compressor is the official quantization library for vLLM. It is maintained by the vLLM team and supports FP8, INT8, INT4, and other quantization formats.

```bash
pip install llmcompressor
```

### W4A16 AWQ (recommended for weight-only, best quality)

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")

recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
]

oneshot(model=model, dataset=ds, recipe=recipe)
model.save_pretrained("Meta-Llama-3-8B-Instruct-AWQ-W4A16")
tokenizer.save_pretrained("Meta-Llama-3-8B-Instruct-AWQ-W4A16")
```

### FP8 dynamic (fastest to apply, no calibration data)

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

# No dataset needed — dynamic activation quantization
oneshot(model=model, recipe=recipe)
model.save_pretrained("Meta-Llama-3-8B-Instruct-FP8-Dynamic")
tokenizer.save_pretrained("Meta-Llama-3-8B-Instruct-FP8-Dynamic")
```

### W8A8 INT8 with SmoothQuant (best throughput on Ampere for W8A8)

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

model.save_pretrained("Meta-Llama-3-8B-Instruct-W8A8", save_compressed=True)
tokenizer.save_pretrained("Meta-Llama-3-8B-Instruct-W8A8")
```

---

## 14. Choosing the right quantization method

Use this decision tree:

```
What GPU do you have?
├── Hopper (H100, H200)
│   └── Best choice: FP8 W8A8 (native hardware support, 1.6× speedup)
│       Second choice: AWQ + Marlin (W4A16)
│
├── Ada Lovelace (RTX 4090, L40S)
│   └── Best choice: FP8 W8A8 or AWQ + Marlin
│       For memory-constrained: AWQ W4A16 + Marlin
│
├── Ampere (A100, RTX 3090/3080)
│   └── Best choice for throughput: INT8 W8A8 or AWQ + Marlin
│       Best choice for memory: AWQ W4A16 + Marlin
│       Note: FP8 W8A8 not fully supported on Ampere
│
├── Turing (T4, RTX 2080)
│   └── Best choice: GPTQ W4A16 (Marlin not available)
│       INT8 W8A8 works but limited throughput benefit
│
├── Volta (V100)
│   └── GPTQ W4A16 only
│
└── CPU (your laptop)
    └── AWQ or GPTQ W4A16 for memory reduction
        No throughput benefit without optimized CPU kernels
        Stick with smaller BF16 models if RAM allows
```

**General rules:**
- If you have a pre-quantized model available → just use it, no need to quantize yourself
- If VRAM is the bottleneck → W4A16 (AWQ or GPTQ) gives 4× memory reduction
- If throughput is the bottleneck on Ampere/Hopper → W8A8 INT8 or FP8 uses hardware tensor cores directly
- If you're on Hopper → FP8 is the recommended default for production
- If you need to fine-tune later → don't quantize; or fine-tune then quantize post-training

---

## 15. Measuring quality after quantization

Always validate your quantized model. The standard metrics are:

**Perplexity** — measures how confidently the model predicts held-out text. Lower is better. A well-quantized INT4 model should show perplexity within 5–10% of the FP16 baseline on WikiText-2 or C4.

```python
# Quick sanity check — compare outputs side by side
from vllm import LLM, SamplingParams

test_prompts = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "def fibonacci(n):",
]
params = SamplingParams(temperature=0.0, max_tokens=50)

# Load both models and compare
llm_fp16 = LLM(model="meta-llama/Llama-3.2-1B-Instruct", dtype="bfloat16")
llm_quant = LLM(model="./my-quantized-model")

for p in test_prompts:
    out_fp16 = llm_fp16.generate([p], params)[0].outputs[0].text
    out_quant = llm_quant.generate([p], params)[0].outputs[0].text
    print(f"Prompt: {p}")
    print(f"  FP16:  {out_fp16.strip()[:80]}")
    print(f"  Quant: {out_quant.strip()[:80]}")
    print()
```

**Task benchmarks** — for production systems, evaluate on task-specific metrics:
- **HumanEval / MBPP**: coding tasks (Pass@1)
- **MMLU**: factual knowledge
- **GSM8K**: arithmetic reasoning
- **HellaSwag**: commonsense reasoning

The vLLM-compatible `lm-evaluation-harness` can benchmark any vLLM model:

```bash
pip install lm-eval
lm_eval \
  --model vllm \
  --model_args pretrained=./my-quantized-model,add_bos_token=True \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size auto
```

---

## 16. Practice exercises

### Exercise 1 — Load and compare a pre-quantized model

```python
import os, time
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"

from vllm import LLM, SamplingParams

params = SamplingParams(temperature=0.0, max_tokens=50)
prompt = ["Explain the water cycle in one sentence."]

# Full precision (CPU — use a tiny model)
print("Loading BF16 model...")
t0 = time.time()
llm_base = LLM(model="facebook/opt-125m", dtype="bfloat16", max_model_len=256)
print(f"Load time: {time.time()-t0:.1f}s")
out_base = llm_base.generate(prompt, params)[0].outputs[0].text

# On GPU, you would compare to a GPTQ/AWQ version:
# llm_quant = LLM(model="TheBloke/opt-125m-GPTQ", dtype="float16")
# For CPU, demonstrate the principle with a small model at different dtypes

print(f"BF16 output: {out_base.strip()}")
```

### Exercise 2 — Dynamic FP8 on GPU (requires Ada or Hopper)

```python
from vllm import LLM, SamplingParams

# No pre-quantized model needed — vLLM applies FP8 on load
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8",          # Dynamic FP8 — weights quantized to FP8_E4M3
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
)

params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(
    ["Explain the difference between FP8 and INT8 quantization."],
    params
)
print(outputs[0].outputs[0].text)
```

### Exercise 3 — VRAM comparison

```python
import torch

def vram_used_gb():
    return torch.cuda.memory_allocated() / 1e9

# GPU only
if torch.cuda.is_available():
    from vllm import LLM
    
    # BF16
    torch.cuda.empty_cache()
    vram_before = vram_used_gb()
    llm = LLM(model="facebook/opt-1.3b", dtype="bfloat16")
    vram_bf16 = vram_used_gb() - vram_before
    del llm
    torch.cuda.empty_cache()
    
    # AWQ (if available for this model — use TheBloke's AWQ version)
    # llm = LLM(model="TheBloke/opt-1.3b-AWQ", quantization="awq")
    # vram_awq = vram_used_gb() - vram_before
    
    print(f"BF16 VRAM usage: {vram_bf16:.2f} GB")
    # Expected: ~2.6 GB for 1.3B params at bfloat16 (2 bytes × 1.3B = 2.6 GB)
```

### Exercise 4 — KV cache FP8 (GPU, Ada/Hopper only)

```python
from vllm import LLM, SamplingParams

# Standard KV cache
llm_standard = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
)

# FP8 KV cache — halves KV cache memory, doubling concurrent capacity
llm_fp8_kv = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    kv_cache_dtype="fp8",       # KV blocks stored as FP8 instead of BF16
    gpu_memory_utilization=0.85,
)

# Compare # GPU blocks in startup logs:
# llm_fp8_kv should show approximately 2x more GPU blocks
```

---

## 17. Chapter summary

| Concept | Key fact |
|---------|----------|
| W4A16 | 4-bit weights, 16-bit activations — weight-only quantization (GPTQ, AWQ) |
| W8A8 | 8-bit weights and activations — matrix multiply runs at INT8 or FP8 |
| GPTQ | Second-order (Hessian-based) weight quantization; requires calibration data; supports W4A16 and W8A16 |
| AWQ | Activation-aware per-channel scaling; protects 1% salient weights; W4A16; no backprop |
| INT8 W8A8 | Hardware INT8 tensor cores; requires SmoothQuant for activation outliers; not on Blackwell |
| FP8 W8A8 | Ada + Hopper only; 2× memory reduction, 1.6× throughput; dynamic variant needs no calibration data |
| KV cache FP8 | `kv_cache_dtype="fp8"`; Ada + Hopper only; halves KV cache memory |
| Marlin kernel | Async memory copy pipeline for W4A16; 2.6–10.9× speedup on Ampere+; auto-selected |
| Machete kernel | Hopper-specific successor to Marlin; auto-selected on H100/H200 |
| LLM Compressor | Official vLLM quantization library; supports AWQ, GPTQ, FP8, INT8 |
| Auto-detection | vLLM reads `quantization_config` from `config.json` automatically for most HF models |
| Quality check | Always run perplexity + task benchmarks (HumanEval, GSM8K, MMLU) after quantization |

---

## What's next

Chapter 7 — **Distributed Inference**: with quantization you can fit large models on fewer GPUs, but some models require multiple GPUs regardless. Tensor parallelism and pipeline parallelism let you split a model across GPUs and nodes. We'll cover the mechanics, the tradeoffs, and the `tensor_parallel_size` and `pipeline_parallel_size` flags in vLLM.

---

*Sources: [vLLM Quantization docs](https://docs.vllm.ai/en/latest/features/quantization/) · [LLM Compressor](https://github.com/vllm-project/llm-compressor) · [AWQ paper](https://arxiv.org/abs/2306.00978) · [GPTQ paper](https://arxiv.org/abs/2210.17323) · [GPUStack quantization benchmarks](https://docs.gpustack.ai/2.0/performance-lab/references/the-impact-of-quantization-on-vllm-inference-performance/) · [JarvisLabs benchmarks](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) · vLLM 0.19.0*
