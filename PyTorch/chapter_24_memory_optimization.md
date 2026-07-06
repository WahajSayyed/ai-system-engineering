# Chapter 24: Memory Optimization

## Introduction

Every technique in this book so far has assumed a model, batch size, and sequence length that comfortably fit in available GPU memory. In practice, that assumption breaks down constantly — a larger model, a longer sequence, or a bigger batch than your hardware can hold is one of the most common walls you'll hit in real training work, whether that's on your RTX 3090's 24GB or coordinating memory across your multi-node setup. This chapter closes out Part V with the standard techniques for training larger models than would otherwise fit: gradient checkpointing (trading compute for memory), and a systematic approach to debugging out-of-memory (OOM) errors rather than guessing at fixes.

By the end of this chapter you'll be able to:
- Explain exactly what consumes GPU memory during training, and in what proportion
- Use gradient checkpointing to trade recomputation for reduced memory usage
- Understand activation offloading as a complementary technique
- Debug OOM errors systematically, using the profiling tools from Chapter 20
- Apply a practical checklist of memory-reduction techniques, in a sensible order of effort

---

## 24.1 What Actually Consumes GPU Memory

Before optimizing memory usage, it's worth being precise about where it goes during training — a habit of mind directly connected to Chapter 20's "profile before optimizing" principle:

- **Model parameters**: the weights themselves — fixed, doesn't grow with batch size.
- **Gradients**: one gradient tensor per parameter (Chapter 3) — same size as the parameters, also fixed regardless of batch size.
- **Optimizer state**: `AdamW` (Chapter 6) maintains two additional per-parameter buffers (momentum and variance estimates) — meaning `AdamW`'s optimizer state alone is roughly *twice* the size of the model's parameters, a detail that's easy to underestimate when budgeting memory.
- **Activations**: the intermediate outputs of every layer during the forward pass, kept in memory because the backward pass needs them to compute gradients (Chapter 3's computational graph, made concrete) — this is the component that scales directly with batch size and sequence length, and is typically the dominant, fastest-growing consumer of memory in a deep network.

```python
def estimate_model_memory_gb(model, optimizer_multiplier=2):
    param_count = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4   # float32, Chapter 1

    params_gb = param_count * bytes_per_param / 1e9
    gradients_gb = params_gb   # same size as parameters
    optimizer_gb = params_gb * optimizer_multiplier   # AdamW: 2x params

    total_gb = params_gb + gradients_gb + optimizer_gb
    print(f"Parameters: {params_gb:.2f} GB")
    print(f"Gradients: {gradients_gb:.2f} GB")
    print(f"Optimizer state: {optimizer_gb:.2f} GB")
    print(f"Total (excluding activations): {total_gb:.2f} GB")
    return total_gb

estimate_model_memory_gb(model)
```

Note this deliberately excludes activation memory, since — unlike the other three components — it isn't a fixed function of parameter count alone; it depends on batch size, sequence length, and how many layers' worth of intermediate values are being kept alive simultaneously. This is precisely the component gradient checkpointing (Section 24.2) targets.

Directly connecting back to Chapter 15: mixed precision reduces the *activation* memory footprint substantially (roughly half, using `float16`/`bfloat16` instead of `float32` for the forward pass), but note that a naive AMP setup still commonly keeps `float32` master weights and `AdamW`'s `float32` optimizer state — meaning mixed precision's memory savings are concentrated specifically in the activation component, not uniformly across all four categories above.

---

## 24.2 Gradient Checkpointing: Trading Compute for Memory

The core idea: instead of storing *every* intermediate activation from the forward pass (needed for the backward pass, per Chapter 3), gradient checkpointing stores only a subset of them — at strategically chosen "checkpoint" points — and **recomputes** the discarded activations on the fly during the backward pass, by re-running the relevant portion of the forward pass a second time.

This is a genuine trade-off, not a free optimization: you're spending extra forward-pass compute (typically adding roughly 20-30% to overall training time, though this varies by model) in exchange for a substantial reduction in peak memory usage — often allowing a meaningfully larger batch size or model than would otherwise fit at all, which frequently more than offsets the added per-step compute cost in terms of overall training throughput.

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Without checkpointing: return self.block(x)
        return checkpoint(self.block, x, use_reentrant=False)   # recomputes activations in backward()

model = nn.Sequential(*[CheckpointedBlock(256) for _ in range(12)])
x = torch.rand(32, 256, requires_grad=True)
output = model(x)
loss = output.sum()
loss.backward()   # each block's internal activations are recomputed here, not stored from forward()
```

`checkpoint(self.block, x, ...)` wraps a call to `self.block(x)`: during the forward pass, it runs the wrapped function normally but *discards* its intermediate activations immediately afterward, storing only the block's input (`x`) and output. During the backward pass, when gradients for this block are needed, it re-runs `self.block(x)` a second time (using the saved input) purely to regenerate those intermediate activations, immediately followed by the actual backward computation through them.

`use_reentrant=False` is worth setting explicitly — it selects the newer, recommended checkpointing implementation, which has fewer restrictions and edge cases (particularly around interactions with other advanced features like DDP, Chapter 22-23) than the older, default-for-backward-compatibility `use_reentrant=True` implementation.

### Applying it to a real transformer stack

Gradient checkpointing is especially common for transformer models (Chapter 13), where each layer's attention and feedforward activations, multiplied across dozens of stacked layers and long sequences, dominate memory usage — directly relevant to the kind of LLM fine-tuning work you've done, where checkpointing (or Unsloth's more specialized memory-efficient implementations building on similar underlying ideas) is often what makes fine-tuning feasible on a single consumer GPU like the RTX 3090 at all:

```python
class CheckpointedTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff)   # Chapter 13
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = checkpoint(block, x, mask, use_reentrant=False)
        return x
```

**Practical guidance:** checkpoint at the granularity of whole transformer blocks (or whole residual blocks in a CNN) rather than individual operations — checkpointing too finely adds recomputation overhead without meaningfully reducing peak memory, since the memory savings come from *not* storing many layers' worth of activations simultaneously, which requires checkpointing at a coarse enough granularity that a meaningful chunk of the network gets discarded and recomputed as a unit.

---

## 24.3 Activation Offloading: A Complementary Technique

A related but distinct technique: instead of discarding activations and recomputing them (gradient checkpointing's approach), **activation offloading** moves activations from GPU memory to CPU memory (or even disk) during the forward pass, then transfers them back to the GPU when needed during the backward pass. This trades GPU memory for CPU memory and data-transfer time (via PCIe bandwidth), rather than for extra compute.

```python
# Conceptual illustration -- in practice, this is typically handled by a library
# (e.g., DeepSpeed's ZeRO-Offload, or FSDP's CPU offload option) rather than
# implemented by hand, given the complexity of correctly overlapping transfers
# with compute for good performance.

class OffloadedActivation(torch.autograd.Function):   # Chapter 16
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x.cpu())   # move to CPU, freeing GPU memory
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output   # x would be moved back to GPU here if needed for a real gradient computation
```

This toy example is included specifically to make the underlying mechanism concrete — connecting directly back to Chapter 16's `autograd.Function` and Chapter 4's `.cpu()`/device-transfer discussion — but production use of activation offloading is essentially always via a framework that handles the transfer scheduling, overlap with compute, and edge cases correctly (DeepSpeed and Fully Sharded Data Parallel, or FSDP, both offer offloading options), rather than a hand-rolled implementation like the one above. **Practical guidance:** reach for gradient checkpointing first — it's simpler, has no data-transfer bandwidth dependency, and is directly supported via the standard `torch.utils.checkpoint` API shown in Section 24.2; consider offloading (via an established framework) specifically when checkpointing's added recompute cost is itself becoming a bottleneck, or when even checkpointed memory usage still doesn't fit.

---

## 24.4 Debugging OOM Errors Systematically

An out-of-memory error is one of the more disruptive failures to hit mid-training, and it's worth having a systematic response rather than randomly trying fixes. Building directly on Chapter 20's profiling tools:

### Step 1: Check peak memory usage, not just the error message

```python
torch.cuda.reset_peak_memory_stats()

# ... run one training step ...

peak_gb = torch.cuda.max_memory_allocated() / 1e9
reserved_gb = torch.cuda.max_memory_reserved() / 1e9
print(f"Peak allocated: {peak_gb:.2f} GB")
print(f"Peak reserved: {reserved_gb:.2f} GB")
```

`max_memory_allocated()` reports memory actually holding tensor data; `max_memory_reserved()` reports memory PyTorch's caching allocator has claimed from CUDA overall (which can be somewhat higher, since PyTorch retains freed memory in a cache for fast reuse rather than immediately returning it to CUDA) — a large gap between the two can sometimes indicate fragmentation, discussed next.

### Step 2: Check for fragmentation

Memory **fragmentation** occurs when many small, differently-sized allocations and deallocations over time leave GPU memory in a state where the *total* free memory is sufficient for a new allocation, but no single *contiguous* block is large enough — leading to an OOM error even though `nvidia-smi` might report meaningful free memory overall. This is more common with highly variable tensor shapes (e.g., training on sequences of very different lengths without bucketing/padding to consistent sizes, Chapter 7) than with consistently-shaped batches.

```python
print(torch.cuda.memory_summary())   # detailed breakdown, including fragmentation-relevant stats
```

A practical mitigation, when fragmentation is suspected, is setting the `PYTORCH_CUDA_ALLOC_CONF` environment variable before running your script, which changes the caching allocator's behavior to reduce fragmentation risk at some cost to allocation speed:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py
```

### Step 3: Bisect to find the actual culprit

If it's unclear whether the model, batch size, sequence length, or some specific operation is responsible, a systematic bisection — deliberately reducing one variable at a time and re-measuring peak memory — is far more reliable than guessing:

```python
for batch_size in [64, 32, 16, 8]:
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand(batch_size, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"batch_size={batch_size}: peak={peak_gb:.2f} GB")
```

Roughly linear scaling of peak memory with batch size confirms activations (Section 24.1) are the dominant, batch-size-dependent component — the expected, normal case. A sharp, disproportionate jump at one specific batch size, rather than smooth scaling, is worth investigating further; it can indicate something like a specific operation with unusually poor memory scaling behavior at that particular shape, findable via the `torch.profiler` memory tooling from Chapter 20, Section 20.5.

---

## 24.5 A Practical Memory-Reduction Checklist

In roughly increasing order of implementation effort — worth working through in this order rather than reaching for the most complex technique first:

1. **Reduce batch size.** The simplest lever, always available, though it may require adjusting the learning rate (Chapter 22, Section 22.5's effective batch size discussion applies here in reverse) and increases training time per epoch.
2. **Enable mixed precision (Chapter 15).** Often a substantial activation memory reduction essentially for free, alongside a speed improvement — usually the first thing worth trying if it isn't already in use.
3. **Clear unused variables and use `del` + `torch.cuda.empty_cache()` deliberately** when working interactively (e.g., in a notebook) where old tensors from previous cells may be lingering unnecessarily — less relevant inside a clean, well-structured training script, but a common source of "phantom" memory usage during interactive experimentation.
4. **Use gradient accumulation** (briefly introduced in Chapter 3, Section 3.4, and Chapter 8, Section 8.7) to simulate a larger effective batch size while keeping the actual per-step batch size (and thus peak activation memory) small.
5. **Apply gradient checkpointing (Section 24.2).** A substantial and reliable memory reduction at a moderate, predictable compute cost — the standard next step once the cheaper options above are exhausted or insufficient.
6. **Consider activation offloading or model sharding** (FSDP, covered conceptually in Section 24.3, and touched on further in later chapters) — reach for these once single-GPU techniques are genuinely insufficient and the model or batch requirements exceed what a single GPU can hold even with checkpointing.

**Practical guidance:** work through this list roughly in order — reducing batch size and enabling mixed precision are nearly free wins worth trying before anything more involved, while activation offloading and model sharding introduce meaningfully more complexity and are worth reserving for when the simpler techniques have been exhausted and are demonstrably insufficient.

---

## Summary

- GPU memory during training is consumed by parameters, gradients, optimizer state (roughly 2x parameters for `AdamW`), and activations — activations are the component that scales with batch size/sequence length and is usually the dominant, fastest-growing consumer.
- Gradient checkpointing discards intermediate activations during the forward pass and recomputes them during backward, trading roughly 20-30% extra compute for substantially reduced peak memory — apply it at the granularity of whole blocks, not individual operations.
- Activation offloading moves activations to CPU memory instead of recomputing them, trading GPU memory for PCIe transfer time — typically used via an established framework (DeepSpeed, FSDP) rather than hand-implemented.
- Debug OOM errors systematically: check peak memory with `torch.cuda.max_memory_allocated()`, check for fragmentation via `memory_summary()`, and bisect batch size/sequence length to confirm expected linear scaling versus an unexpected spike worth investigating further.
- Work through memory-reduction techniques roughly in order of increasing complexity — batch size reduction and mixed precision first, gradient checkpointing next, offloading/sharding last.

## Exercises

1. Using the `estimate_model_memory_gb` function from Section 24.1, compute the estimated parameter/gradient/optimizer memory for a model of your choice, then compare it against `torch.cuda.max_memory_allocated()` measured empirically after one real training step — the gap between the two is roughly the activation memory for that step.
2. Apply gradient checkpointing to a deep MLP or transformer stack (using the pattern from Section 24.2), and measure both the peak memory reduction and the wall-clock time increase per training step, using the profiling techniques from Chapter 20.
3. Run the batch-size bisection loop from Section 24.4, Step 3, on a model of your choice, and plot peak memory against batch size — confirm whether the scaling is roughly linear, as expected, or whether you observe a disproportionate jump worth investigating.
4. Combine gradient accumulation (simulating a large effective batch size with a small per-step batch size) with gradient checkpointing on the same model, and measure the total peak memory reduction achieved by combining both techniques compared to using neither.

---

# Part V Recap

Part V covered the performance and systems layer that sits around model training: profiling to identify actual bottlenecks rather than guessing, `torch.compile` for reducing dispatch overhead, multi-GPU training via DDP for scaling across GPUs and nodes, and memory optimization for training larger models than would otherwise fit. Combined with Parts I-IV, you now have the complete toolkit for training models correctly, efficiently, and at scale.

**Next:** Part VI moves into advanced, specialized topics — starting with Chapter 25, writing custom CUDA extensions for PyTorch, tying directly into your existing CUDA curriculum and letting you write hand-optimized kernels for operations that need performance beyond what standard PyTorch operations provide.
