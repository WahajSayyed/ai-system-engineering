# Chapter 27: Model Export & Deployment

## Introduction

Every model in this book so far has run inside a live Python process, with the model class definition available and PyTorch fully installed. Production deployment often can't assume that: a mobile app, an embedded robotics controller, or a low-latency inference server may need a model in a form that doesn't require the original Python source, doesn't carry PyTorch's full training-time overhead, or needs to run inside a completely different runtime (a C++ application, a different ML framework entirely) than the one it was trained in. This chapter covers the current PyTorch tooling for that transition: `torch.export`, and ONNX export built on top of it.

By the end of this chapter you'll be able to:
- Explain what "exporting" a model actually captures, and why it differs fundamentally from `torch.save`
- Use `torch.export` to produce a portable, serializable graph representation of a model
- Handle dynamic shapes correctly during export
- Export a model to ONNX for cross-framework and cross-runtime deployment
- Understand where TorchScript fits in this landscape today, and why new work shouldn't target it

---

## 27.1 Why Export at All? `state_dict()` Isn't Enough Outside Python

Recall Chapter 8, Section 8.6: `model.state_dict()` saves only the *weights* — the numeric parameter values — not the model's architecture. Loading a checkpoint requires the original Python class definition to already exist in the environment doing the loading, so the weights can be loaded back into a freshly-instantiated instance of that exact class. This is completely fine within a Python-to-Python workflow (which is what every checkpointing example in this book so far has assumed), but it's a hard blocker for deployment scenarios where the loading environment doesn't have — or shouldn't need — your training code at all.

**Exporting** a model, by contrast, captures both the architecture (as a computational graph, similar in spirit to the graph `torch.compile`'s TorchDynamo builds in Chapter 21) and the weights together, into a self-contained representation that a separate runtime can execute without needing your original `nn.Module` class definition, or in some cases without needing Python or PyTorch installed at all.

---

## 27.2 `torch.export`: The Current Standard

`torch.export` is PyTorch's current, recommended mechanism for capturing a model as a portable graph — superseding the older TorchScript approach (Section 27.5) for essentially all new work.

```python
import torch

model = FashionMNIST_CNN().to("cpu")   # Chapter 10
model.eval()   # Chapter 8, Section 8.2 -- export captures inference-time behavior

example_inputs = (torch.randn(1, 1, 28, 28),)   # a representative input, as a tuple
exported_program = torch.export.export(model, example_inputs)

print(exported_program)   # shows the captured graph, similar in spirit to a torch.compile trace
```

`torch.export.export` traces through the model using the provided `example_inputs`, producing an `ExportedProgram` — a graph representation using a normalized set of core PyTorch operators (ATen ops), with Python-level control flow and data structures eliminated from the graph itself (any genuinely necessary dynamic control flow must be handled explicitly, similar in spirit to Chapter 21's discussion of `torch.compile`'s graph breaks — `torch.export` is stricter than `torch.compile` about this, since it needs to produce one complete, self-contained graph, not a partially-compiled model still running inside a live Python process).

**Calling `model.eval()` before export matters for the same reason it matters before validation (Chapter 8, Section 8.2)** — the exported graph reflects whatever mode the model was in during tracing; exporting a model still in `.train()` mode would capture dropout/batchnorm's training-time behavior into a graph intended for inference, which is essentially always the wrong thing to ship.

### Saving and loading an `ExportedProgram`

```python
torch.export.save(exported_program, "model.pt2")

# In a separate environment, without the original FashionMNIST_CNN class definition available:
loaded_program = torch.export.load("model.pt2")
output = loaded_program.module()(torch.randn(1, 1, 28, 28))
```

This is the direct payoff described in Section 27.1: `loaded_program` can run inference without the original Python class definition present in the loading environment at all — only the exported `.pt2` file and PyTorch itself are required.

---

## 27.3 Handling Dynamic Shapes

By default, `torch.export` specializes the exported graph to the *exact* shapes of the `example_inputs` provided — attempting to run the exported model on a different batch size (or, for sequence models, a different sequence length) than what was used during export will fail unless you explicitly tell the exporter which dimensions are allowed to vary.

```python
from torch.export import Dim

batch_dim = Dim("batch", min=1, max=256)   # declares batch size can range from 1 to 256

exported_program = torch.export.export(
    model,
    example_inputs,
    dynamic_shapes={"x": {0: batch_dim}},   # dimension 0 of input "x" is dynamic
)

# Now works correctly for any batch size in the declared range:
output_batch_1 = exported_program.module()(torch.randn(1, 1, 28, 28))
output_batch_32 = exported_program.module()(torch.randn(32, 1, 28, 28))
```

`Dim("batch", min=1, max=256)` declares a named dynamic dimension with an explicit valid range — this range matters beyond documentation purposes: it's used during export to verify (via the shape-constraint reasoning `torch.export` performs internally) that the traced graph is actually valid across that entire declared range, not just at the single concrete shape used for tracing. This directly parallels the batch-size variability concern raised for `torch.compile` in Chapter 21, Section 21.2 (recompilation per distinct shape) — `torch.export`'s `dynamic_shapes` mechanism is the equivalent tool for declaring shape flexibility upfront, at export time, rather than accepting a new compilation per shape as `torch.compile` does at runtime.

**Practical guidance:** if your deployment scenario has a genuinely fixed input shape (e.g., a robotics vision pipeline always feeding the model images at one fixed resolution), exporting without `dynamic_shapes` is simpler and sufficient. Reserve the extra complexity of declaring dynamic dimensions for cases where the deployed model genuinely needs to handle a range of input sizes at inference time.

---

## 27.4 Exporting to ONNX

ONNX (Open Neural Network Exchange) is an open, framework-agnostic format for representing machine learning models — a model exported to ONNX can run in ONNX Runtime, TensorRT, OpenVINO, and numerous other runtimes and hardware-specific accelerators, independent of PyTorch entirely. This is the right target when your deployment environment isn't a PyTorch/Python environment at all, or when you need to target hardware-specific inference engines with their own optimized ONNX support.

```python
import torch

model = FashionMNIST_CNN()
model.eval()
example_inputs = (torch.randn(1, 1, 28, 28),)

onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
onnx_program.save("model.onnx")
```

**`dynamo=True` is the current, recommended ONNX exporter** — it builds directly on `torch.export`'s graph capture (Section 27.2) internally, meaning the same dynamic-shape declaration approach, robustness improvements, and reduced susceptibility to unsupported dynamic control flow that `torch.export` provides for `.pt2` export carry over to ONNX export as well. The older, default-for-backward-compatibility exporter (`dynamo=False`, or simply omitting the argument in older PyTorch versions) relies on the deprecated TorchScript tracing machinery discussed in Section 27.5, and is no longer the recommended path for new work — worth being aware of specifically because a meaningful amount of existing ONNX-export tutorial content online still shows the older, non-`dynamo` invocation.

### Verifying the exported ONNX model

Given the format conversion involved, it's worth explicitly verifying numerical agreement between the original PyTorch model and the exported ONNX model before trusting the exported version in production — connecting directly to the correctness-verification instinct established for custom autograd functions (Chapter 16's `gradcheck`) and custom CUDA kernels (Chapter 25):

```python
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("model.onnx")

test_input = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    pytorch_output = model(test_input).numpy()

onnx_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
onnx_output = ort_session.run(None, onnx_inputs)[0]

np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5)
print("PyTorch and ONNX Runtime outputs match.")
```

`rtol`/`atol` tolerances rather than exact equality are expected and appropriate here — different runtimes (PyTorch's own kernels vs. ONNX Runtime's) can produce tiny floating-point differences from differing operation ordering or kernel implementations, exactly the kind of numerical-precision nuance familiar from Chapter 15's mixed precision discussion, without indicating an actual export bug.

---

## 27.5 A Note on TorchScript

You will encounter TorchScript (`torch.jit.script` and `torch.jit.trace`) in a substantial amount of existing PyTorch code, tutorials, and documentation, since it was the standard export mechanism for years before `torch.export` existed. **It is now in maintenance mode, and new projects should target `torch.export` instead.** Worth understanding this transition explicitly, since it affects how to read older PyTorch codebases and material:

- `torch.jit.trace` records operations for one specific execution path (similar in spirit to `torch.export`'s tracing, but with weaker guarantees and less robust handling of dynamic control flow) — it can silently produce an incorrect graph for inputs that would take a different code path than the one used during tracing, a correctness risk `torch.export`'s stricter tracing and shape-constraint checking is specifically designed to catch instead of silently mishandling.
- `torch.jit.script` instead directly parses and compiles a *subset* of Python syntax, handling some dynamic control flow that tracing can't — but only supports a restricted subset of the language, often requiring genuine code changes (avoiding unsupported Python constructs) to get an existing model to script successfully.

**Practical guidance:** if you're starting new export or deployment work, use `torch.export` (Section 27.2) and, if targeting ONNX, `torch.onnx.export(..., dynamo=True)` (Section 27.4) — both are actively developed, more robust in their handling of the dynamic-shape and control-flow issues that plagued TorchScript, and are where PyTorch's own ongoing investment is concentrated. If you're maintaining or reading an existing codebase using `torch.jit.trace`/`torch.jit.script`, it's worth understanding what those calls do, but not worth building new functionality on top of them going forward.

---

## 27.6 A Deployment Decision Guide

A practical summary of which export target fits which situation, tying together this chapter's tools with the quantization work from Chapter 26:

- **Staying entirely within Python/PyTorch for inference** (a server-side inference API, for instance): plain `state_dict()` loading (Chapter 8) is sufficient — no export needed at all; consider `torch.compile` (Chapter 21) for inference-time speed.
- **Deploying to a non-Python or cross-framework environment, staying within the broader PyTorch/ONNX ecosystem**: `torch.export` directly (`.pt2` files), particularly relevant for edge/embedded deployment via tooling built on `torch.export` (such as ExecuTorch, PyTorch's dedicated edge-deployment runtime, worth researching further if targeting genuinely resource-constrained hardware — beyond this introductory chapter's scope).
- **Needing true framework-agnostic portability, or targeting a specific hardware-vendor runtime** (TensorRT, OpenVINO, a mobile inference engine): ONNX export via `torch.onnx.export(..., dynamo=True)`.
- **Combining with quantization** (Chapter 26): quantize first, then export the already-quantized model — both `torch.export` and the ONNX exporter can generally handle a quantized model's graph, though it's worth verifying numerically (Section 27.4's verification pattern) that quantization survived the export step correctly, given the additional format conversion involved.

For your own conveyor sorting system's vision pipeline — likely running inference directly within a Python process on your GPU server — plain `state_dict()` loading is probably sufficient day to day; `torch.export`/ONNX become relevant specifically if you ever need to deploy inference onto a more constrained or non-Python component of that pipeline (an embedded controller closer to the robotic arms, for instance).

---

## Summary

- `state_dict()` (Chapter 8) captures only weights and requires the original Python class definition to reload — exporting captures architecture and weights together into a self-contained, portable representation.
- `torch.export.export()` traces a model into an `ExportedProgram`, a normalized graph that can be saved (`.pt2`) and loaded/run without the original model class present — the current, recommended capture mechanism.
- `dynamic_shapes` with `torch.export.Dim` declares which input dimensions are allowed to vary at inference time, with an explicit valid range verified during export.
- `torch.onnx.export(..., dynamo=True)` builds on `torch.export` to produce framework-agnostic ONNX models for cross-runtime deployment — always verify numerical agreement between the original and exported model with an appropriate tolerance.
- TorchScript (`torch.jit.trace`/`torch.jit.script`) is in maintenance mode; new export work should target `torch.export` and the `dynamo=True` ONNX exporter instead.

## Exercises

1. Export the `FashionMNIST_CNN` from Chapter 10 with `torch.export.export()`, save it, load it in a way that doesn't reference the original class (e.g., in a fresh Python session or script), and run inference on a test input.
2. Add `dynamic_shapes` to the export from Exercise 1 to support a batch dimension ranging from 1 to 128, and verify the exported model correctly handles both a batch size of 1 and a batch size of 64.
3. Export the same model to ONNX using `torch.onnx.export(..., dynamo=True)`, load it with `onnxruntime.InferenceSession`, and verify numerical agreement with the original PyTorch model using the tolerance-based comparison from Section 27.4.
4. Research ExecuTorch (PyTorch's edge-deployment runtime, built on `torch.export`) via PyTorch's official documentation, and summarize in a few sentences what kind of deployment target it's designed for and how it relates to the `torch.export` workflow covered in this chapter.

**Next:** Chapter 28 covers building a custom optimizer from scratch — implementing an Adam-like or GRPO-style update rule directly, connecting the optimizer mechanics from Chapter 6 to the RL-adjacent training techniques used later in this curriculum.
