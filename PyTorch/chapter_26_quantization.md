# Chapter 26: Quantization

## Introduction

Every model in this book has used `float32` (or, with Chapter 15's mixed precision, `float16`/`bfloat16`) to represent weights and activations. **Quantization** goes further: representing values with even fewer bits — typically 8-bit or 4-bit integers — to shrink model size, reduce memory bandwidth, and speed up inference, particularly valuable for deploying models where compute or memory is constrained. This chapter covers the core quantization concepts, post-training quantization (PTQ), quantization-aware training (QAT), and the current PyTorch tooling ecosystem for applying them.

By the end of this chapter you'll be able to:
- Explain what quantization actually does numerically, and why it's harder than simply truncating bits
- Apply post-training dynamic and static quantization to a trained model
- Understand quantization-aware training and why it typically outperforms PTQ for aggressive quantization
- Navigate PyTorch's current quantization tooling landscape, including `torchao`
- Reason about the accuracy/efficiency trade-off quantization involves

---

## 26.1 What Quantization Actually Computes

Quantizing a tensor means mapping its `float32` values to a much smaller set of representable values — typically the 256 integers of `int8`, or the 16 integers of `int4` — using an affine transformation defined by a **scale** and a **zero-point**:

$$x_{quant} = \text{round}(x_{float} / \text{scale}) + \text{zero\_point}$$

```python
import torch

def quantize_tensor(x, num_bits=8):
    qmin, qmax = 0, 2 ** num_bits - 1   # e.g., 0 to 255 for int8 (unsigned)

    x_min, x_max = x.min().item(), x.max().item()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - x_min / scale

    x_quant = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return x_quant.to(torch.uint8), scale, zero_point

def dequantize_tensor(x_quant, scale, zero_point):
    return (x_quant.float() - zero_point) * scale

x = torch.tensor([-1.5, -0.3, 0.0, 0.8, 2.1])
x_quant, scale, zero_point = quantize_tensor(x)
x_dequant = dequantize_tensor(x_quant, scale, zero_point)

print(x_quant)      # tensor([  0,  70,  92, 133, 255], dtype=torch.uint8)
print(x_dequant)    # tensor([-1.5000, -0.3059, -0.0059,  0.7900,  2.1000])  -- close, not exact
```

The scale and zero-point are computed from the tensor's actual value range, mapping that continuous range onto the small number of discrete integers available — `scale` determines how much real-valued range each integer step represents, and `zero_point` shifts the mapping so that `0.0` in floating point maps to a valid integer (important, since many tensors — activations after a ReLU, for instance — have a meaningful concentration of exact zeros that benefit from being representable exactly).

**The dequantized result doesn't exactly match the original** — this is the fundamental, unavoidable cost of quantization: representing a continuous range with only 256 (or 16, for `int4`) discrete steps necessarily loses precision, exactly analogous to (though more aggressive than) the `float32`-to-`float16` precision loss discussed in Chapter 15, just applied to a far smaller number of representable values. The entire discipline of quantization is about minimizing the *practical* impact of this loss — on model accuracy — while maximizing the *practical* benefit — in memory and compute savings.

### Symmetric vs. asymmetric quantization

The example above is **asymmetric** (zero-point can be any value, mapping the tensor's actual min/max range). **Symmetric** quantization instead forces the range to be centered at zero (`zero_point = 0`), trading some representable range efficiency for simpler, faster arithmetic during inference — since a computation involving a zero-point of exactly 0 avoids an extra addition/subtraction at every use. Symmetric quantization is more common for weights (whose distributions are often naturally close to zero-centered); asymmetric is more common for activations (especially post-ReLU activations, which are entirely non-negative and thus poorly suited to a zero-centered symmetric range).

---

## 26.2 Why Quantize: Memory and Speed

Two distinct benefits, worth being clear about since they matter differently depending on your deployment target:

- **Memory**: `int8` uses a quarter of `float32`'s storage (and half of `float16`'s); `int4` uses an eighth. For a large model, this directly determines whether it fits on a given GPU or edge device at all — directly relevant to fitting a larger fine-tuned LLM onto your RTX 3090, or deploying a vision model onto more memory-constrained hardware in a robotics context.
- **Speed**: many hardware platforms have dedicated, faster integer arithmetic paths (distinct from the Tensor Core float paths discussed in Chapter 15), and moving less data (lower memory bandwidth) directly speeds up memory-bound operations — a substantial fraction of inference workloads, especially at the small batch sizes typical of real-time deployment (directly relevant to the low-latency inference needs of your conveyor sorting system's vision pipeline).

---

## 26.3 Post-Training Quantization (PTQ)

**Post-training quantization** applies quantization to an already-fully-trained model, with no further training required — the fastest, lowest-effort quantization approach, and the right starting point before considering the more involved QAT (Section 26.5).

### Dynamic quantization: the simplest form

Dynamic quantization quantizes weights ahead of time (once, at conversion), but computes each activation's quantization parameters (scale/zero-point) **dynamically, at inference time**, based on that specific input's actual range — no calibration data required, since activation ranges are computed on the fly rather than pre-estimated.

```python
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
model.eval()   # Chapter 8, Section 8.2 -- quantization is an inference-time transformation

quantized_model = quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

`{nn.Linear}` specifies which layer types to quantize — dynamic quantization is most commonly applied to `nn.Linear` layers specifically, since their weight matrices tend to dominate both memory footprint and compute cost in many architectures. This is a genuinely minimal-effort entry point: no calibration dataset, no retraining, one function call on an already-trained model.

### Static quantization: quantizing activations ahead of time too

Static quantization goes further, pre-computing quantization parameters for *activations* as well as weights — which requires a **calibration** step: running a representative sample of real data through the model once, observing the actual range of values at each activation, and using those observed ranges to fix the quantization parameters ahead of inference time (rather than computing them dynamically per-input, as dynamic quantization does). This typically yields better speed than dynamic quantization (since no per-inference range computation is needed) at the cost of requiring representative calibration data and being more sensitive to how well that calibration data matches real deployment inputs.

```python
from torch.ao.quantization import get_default_qconfig, prepare, convert

model.qconfig = get_default_qconfig("x86")   # target-hardware-specific quantization config
prepared_model = prepare(model)   # inserts observers to record activation ranges

# Calibration: run representative data through the model, observers record ranges
with torch.no_grad():
    for x_batch, _ in calibration_loader:   # a small, representative subset of real data
        prepared_model(x_batch)

quantized_model = convert(prepared_model)   # uses observed ranges to finalize quantization parameters
```

`get_default_qconfig("x86")` selects a quantization configuration tuned for a specific target deployment backend (here, `x86` CPUs) — the correct choice depends on where the quantized model will actually run, since different hardware backends support different quantization schemes efficiently. `prepare()` inserts **observer** modules throughout the model that passively record the range of values flowing through during the calibration pass, without altering the model's actual output at that stage; `convert()` then uses those recorded ranges to replace the observed layers with their actual quantized equivalents.

---

## 26.4 The Current PyTorch Quantization Landscape: `torch.ao` and `torchao`

This is worth addressing directly, since PyTorch's quantization tooling has evolved substantially and it's easy to find outdated guidance. `torch.ao.quantization` (used in Section 26.3 above) is PyTorch core's built-in quantization module, covering the classic eager-mode and FX-graph-mode PTQ/QAT workflows shown so far. **`torchao`** ("PyTorch Architecture Optimization") is a separate, actively-developed library — now the recommended path for most new quantization work, including the newer `torch.export`-based quantization flow and more advanced techniques (int4 weight-only quantization, mixed activation/weight schemes) that go beyond what `torch.ao.quantization`'s classic API covers well:

```python
# pip install torchao --break-system-packages
from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig, PerGroup

model = ...   # a trained model, e.g., a fine-tuned LLM

config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerGroup(32),   # quantize weights in groups of 32, not one scale for the whole tensor
)
quantize_(model, config)   # applies quantization in place, following the specified config
```

`quantize_()` (the trailing underscore following the standard PyTorch in-place-operation convention from Chapter 2) is `torchao`'s primary, unified entry point — a single function accepting different `Config` objects to select the specific quantization scheme, rather than the more manual `prepare`/`convert` sequence of the classic `torch.ao.quantization` API. `PerGroup(32)` illustrates **per-group quantization**: rather than one scale/zero-point for an entire weight tensor (as in the simplified example in Section 26.1), the tensor is divided into groups of 32 values, each with its own independently-computed scale — meaningfully improving accuracy for aggressive quantization (like `int4`) by letting the quantization parameters adapt to local variation within a large weight matrix, rather than being forced to accommodate the tensor's full global range with a single scale.

**Practical guidance:** `torchao` is directly relevant to your LLM fine-tuning work — it's the library Unsloth itself builds on for quantization-aware fine-tuning support, meaning familiarity with `torchao`'s `quantize_` API and configuration objects connects directly to tooling you're likely to encounter (or already have encountered) in that context, beyond this curriculum's own examples.

---

## 26.5 Quantization-Aware Training (QAT)

Post-training quantization applies quantization *after* training is complete — the model never experiences quantization's precision loss during training, so it has no opportunity to adapt its weights to be more robust to it. **Quantization-aware training** instead simulates quantization *during* training, using **fake quantization**: quantizing and immediately dequantizing values in the forward pass (so the model experiences quantization's precision loss), while still computing real, full-precision gradients for the backward pass.

This is precisely the **straight-through estimator** pattern from Chapter 16, Section 16.4, applied to its most common real-world use case — worth returning to directly, since the connection is exact, not just analogous:

```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        x_quant = torch.round(x / scale + zero_point).clamp(qmin, qmax)
        x_dequant = (x_quant - zero_point) * scale
        return x_dequant   # quantization's precision loss IS present in the forward pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None   # straight-through: identity gradient for x

fake_quantize = FakeQuantize.apply

class QATLinear(nn.Module):
    def __init__(self, in_features, out_features, num_bits=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.qmin, self.qmax = 0, 2 ** num_bits - 1

    def forward(self, x):
        scale = self.weight.abs().max().item() / self.qmax
        zero_point = self.qmax // 2
        fake_quantized_weight = fake_quantize(
            self.weight, scale, zero_point, self.qmin, self.qmax
        )
        return x @ fake_quantized_weight.T + self.bias
```

During training, `QATLinear` uses the fake-quantized (precision-reduced, then immediately restored to float) weight for its actual forward computation — meaning the loss the model is optimizing genuinely reflects quantization's precision loss — while gradients flow straight through the quantization step (via the straight-through estimator) exactly as in Chapter 16's rounding example, letting the optimizer adjust the underlying full-precision weight to compensate for, and become more robust to, that precision loss.

`torchao`'s QAT API follows the same conceptual flow via a `prepare`/`convert` sequence, letting you apply this at scale to a real model without hand-writing `FakeQuantize` modules for every layer type:

```python
from torchao.quantization.qat import QATConfig

base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4, weight_granularity=PerGroup(32))

quantize_(model, QATConfig(base_config, step="prepare"))   # inserts fake-quantization during training

# ... standard training loop, Chapter 8 — model trains WITH fake quantization active ...

quantize_(model, QATConfig(base_config, step="convert"))   # finalizes to genuinely quantized weights
```

**Practical guidance:** PTQ first, always — it's near-zero additional effort on top of a model you've already trained, and for many models and moderate quantization levels (`int8`), the accuracy loss is small enough that PTQ alone is sufficient. Move to QAT specifically when PTQ's accuracy degradation is unacceptable for your use case — most commonly encountered with more aggressive quantization (`int4` and below), which tends to be considerably more sensitive to whether the model had a chance to adapt to it during training. This mirrors published results on LLM quantization specifically: QAT has been shown to recover a substantial majority of the accuracy degradation that plain PTQ introduces, particularly for lower-bit weight quantization — directly relevant if you pursue more aggressive compression of a model like your fine-tuned Qwen2.5-Coder beyond what standard PTQ would preserve acceptably.

---

## Summary

- Quantization maps a continuous float range onto a small set of discrete integers via a scale and zero-point, trading representational precision for memory and speed — the precision loss is fundamental and unavoidable, and the goal is minimizing its practical impact rather than eliminating it.
- Dynamic quantization (weights pre-quantized, activation ranges computed at inference time) requires no calibration data and is the lowest-effort entry point; static quantization (both weights and activations pre-quantized, via a calibration pass) typically yields better speed at the cost of needing representative calibration data.
- `torch.ao.quantization` is PyTorch core's classic quantization module; `torchao` is the actively-developed, now-recommended library for most new quantization work, including advanced schemes like per-group `int4` quantization, and is what libraries like Unsloth build on for QAT-aware fine-tuning.
- Quantization-aware training uses fake quantization — Chapter 16's straight-through estimator, applied directly — to let a model adapt its weights to quantization's precision loss during training, typically recovering much of the accuracy PTQ alone would sacrifice, especially at aggressive bit-widths.
- Practical sequencing: try PTQ first as a near-zero-cost baseline; move to QAT specifically when PTQ's accuracy loss is unacceptable, most commonly at aggressive (`int4` or lower) quantization levels.

## Exercises

1. Implement `quantize_tensor`/`dequantize_tensor` from Section 26.1 for `int4` (16 levels) instead of `int8`, apply it to a sample tensor, and compare the dequantization error against the `int8` version.
2. Apply dynamic quantization (Section 26.3) to a trained `FashionMNISTClassifier` (Chapter 9), and compare model size on disk (via `torch.save`) and inference accuracy before and after quantization.
3. Implement the full `QATLinear` module from Section 26.5, train a small model using it for a few epochs, and compare its final accuracy after genuine conversion to quantized weights against the same architecture quantized via PTQ alone, at the same target bit-width.
4. Research (via `torchao`'s documentation) the difference between per-tensor and per-group (`PerGroup`) quantization granularity, and explain in your own words why per-group quantization tends to preserve more accuracy for aggressive (`int4`) weight quantization specifically.

**Next:** Chapter 27 covers model export and deployment — TorchScript, ONNX export, and `torch.export` — for taking a trained (and optionally quantized) PyTorch model out of the Python training environment and into a form suitable for production inference.
