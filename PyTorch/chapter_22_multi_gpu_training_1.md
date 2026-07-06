# Chapter 22: Multi-GPU Training I

## Introduction

Every training loop so far has assumed a single GPU. Once a model or dataset grows large enough that a single GPU either runs out of memory or takes too long to train on, the natural next step is spreading the work across multiple GPUs — directly relevant to your multi-node setup, where a single training job could in principle draw on more than one GPU across your NUC cluster and GPU server. PyTorch offers two fundamentally different approaches to this: the simple but limited `nn.DataParallel`, and the more capable, standard approach, `DistributedDataParallel` (DDP).

This chapter explains both, focused on understanding *why* DDP has become the standard choice, before Chapter 23 covers the practical mechanics of actually launching and running a DDP training job.

By the end of this chapter you'll be able to:
- Explain data parallelism as a strategy for multi-GPU training
- Use `nn.DataParallel` and understand precisely why it's discouraged for anything beyond quick experimentation
- Explain DDP's core mechanism — per-process replication and gradient synchronization via all-reduce
- Articulate the specific bottlenecks that make DataParallel slower than DDP, not just "DDP is newer"

---

## 22.1 The Basic Idea: Data Parallelism

Both techniques in this chapter implement the same core strategy, **data parallelism**: replicate the full model on every GPU, split each batch across the GPUs (each GPU processes a different slice of the batch), compute gradients independently on each GPU, then combine (synchronize) those gradients before updating the model — ensuring every replica ends up with identical, updated weights after each step, so they stay in sync across the entire training run.

This is a different strategy from **model parallelism** (splitting a single model's layers across multiple GPUs, because the model itself is too large to fit on one GPU) — a distinct and more complex technique, briefly touched on again in later performance chapters, but not the focus here. Data parallelism assumes the model fits comfortably on a single GPU; it's about processing more data per step by spreading the batch across GPUs, not about fitting a bigger model.

---

## 22.2 `nn.DataParallel`: The Simple, Discouraged Option

`nn.DataParallel` is the older, simpler API — a single line wrapping an existing model:

```python
import torch
import torch.nn as nn

model = FashionMNIST_CNN()   # Chapter 10
model = nn.DataParallel(model)   # wraps the model to use all available GPUs
model = model.to("cuda")

# Training loop is otherwise unchanged from earlier chapters
x_batch = torch.rand(256, 1, 28, 28).to("cuda")   # DataParallel splits this across GPUs automatically
output = model(x_batch)
```

Mechanically, on every single forward call, `DataParallel` (all running in a single Python process, on the main thread): splits the input batch into chunks, one per GPU; copies (replicates) the current model weights from the primary GPU to every other GPU; runs the forward pass independently on each GPU's chunk; gathers all the outputs back onto the primary GPU; and, during the backward pass, gathers gradients back to the primary GPU and applies the update there, before the whole broadcast-copy cycle repeats on the next forward call.

**This per-step weight replication and output/gradient gathering, all funneled through a single primary GPU, is precisely why `DataParallel` scales poorly.** The primary GPU becomes a bottleneck — it does extra work (gathering, and the actual optimizer step) that the other GPUs don't, and the repeated broadcast of weights every single forward pass is pure overhead that grows with model size and GPU count. It's also fundamentally single-process, single-node — `DataParallel` has no mechanism for coordinating across multiple machines at all, ruling it out entirely for a genuinely multi-node setup.

**Practical guidance: `nn.DataParallel` is useful only for the fastest possible way to try "does using more than one GPU help at all" on a single machine during early experimentation.** For anything resembling real training — and certainly for any multi-node setup — use `DistributedDataParallel` instead. This isn't a minor style preference; PyTorch's own documentation explicitly recommends DDP over `DataParallel` even for single-node, multi-GPU cases, given the mechanical inefficiencies just described.

---

## 22.3 `DistributedDataParallel`: The Standard Approach

DDP takes a fundamentally different architectural approach: instead of one Python process managing multiple GPUs, **DDP runs one independent process per GPU**, each with its own complete copy of the model. There's no central process gathering outputs or broadcasting weights every step — each process runs its own full forward and backward pass independently, on its own GPU, using only its own slice of the data.

The only synchronization point is gradients themselves, combined via an **all-reduce** operation: after each process computes its local gradients (via its own independent `.backward()` call), all-reduce efficiently sums those gradients across every process and ensures every process ends up with the identical, averaged result — without needing a central coordinator to gather everything through one bottleneck GPU. Each process then applies that identical gradient using its own local optimizer, and because every process started with identical weights and applied an identical gradient update, all replicas remain in sync without ever needing to broadcast full model weights on every step (the way `DataParallel` does).

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)

    model = FashionMNIST_CNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x_batch = torch.rand(32, 1, 28, 28, device=rank)   # each process's OWN slice of data
    y_batch = torch.randint(0, 10, (32,), device=rank)

    optimizer.zero_grad()
    output = ddp_model(x_batch)
    loss = loss_fn(output, y_batch)
    loss.backward()          # gradients synchronized (all-reduced) automatically during this call
    optimizer.step()          # each process applies the identical, synchronized gradient locally
```

A few concepts introduced here that Chapter 23 will cover in full operational detail — worth previewing now, since they're central to understanding *why* DDP is structured this way:

- **`rank`**: a unique integer identifying each process (`0` to `world_size - 1`) — effectively "which GPU/process am I."
- **`world_size`**: the total number of processes participating in training (commonly, though not necessarily, equal to the total number of GPUs across all machines involved).
- **`dist.init_process_group("nccl", ...)`**: establishes the communication backend processes use to talk to each other — `"nccl"` (NVIDIA Collective Communications Library) is the standard, GPU-optimized backend for CUDA training.
- **`loss.backward()` transparently triggers gradient synchronization** — this is the key architectural difference from a plain single-GPU `.backward()` call (Chapter 3): DDP registers hooks internally (conceptually similar to the backward hooks from Chapter 17, though DDP's are set up automatically) that fire as each parameter's local gradient becomes available, kicking off the all-reduce communication for that parameter in the background, overlapped with the remaining backward computation for other parameters — meaning gradient communication and computation happen concurrently rather than as separate sequential phases, a major contributor to DDP's efficiency advantage over `DataParallel`'s strictly sequential gather-then-update pattern.

---

## 22.4 Why DDP Scales Better: A Direct Comparison

| | `DataParallel` | `DistributedDataParallel` |
|---|---|---|
| Process model | Single process, multiple threads/GPUs | One independent process per GPU |
| Weight synchronization | Full weights re-broadcast every forward pass | Only gradients synchronized, via efficient all-reduce |
| Communication/compute overlap | None — strictly sequential gather-then-compute | Gradient communication overlaps with backward computation |
| Multi-node support | No — single machine only | Yes — designed for it from the ground up |
| Bottleneck | Primary GPU (gathering, optimizer step) | None inherent to the design — all-reduce is symmetric |
| Recommended for | Quick single-machine experimentation only | Essentially everything else |

The "no inherent bottleneck" point in the last row is worth being precise about: all-reduce is a genuinely distributed, symmetric operation — no single GPU does more communication work than any other, unlike `DataParallel`'s primary-GPU gather pattern. This is a large part of why DDP's performance scales meaningfully better as GPU count increases; `DataParallel`'s single-process bottleneck becomes proportionally worse the more GPUs you add, since the primary GPU's fixed gathering workload doesn't shrink even as the per-GPU compute workload does.

---

## 22.5 A Note on Batch Size and Learning Rate

Since each process now handles a *slice* of a larger effective batch (rather than the full batch each GPU processed under `DataParallel`'s splitting, which still produced a single combined gradient), it's worth being deliberate about what "batch size" refers to in a multi-GPU DDP context: the `batch_size` passed to each process's `DataLoader` is the **per-GPU (local) batch size**; the **effective (global) batch size** actually used for each gradient update is `local_batch_size × world_size`, since the all-reduced gradient is effectively averaged across that many samples' worth of independent forward/backward passes combined.

This matters directly for hyperparameter tuning: a learning rate tuned for a single-GPU run with `batch_size=64` is not automatically correct once you scale to 4 GPUs at `batch_size=64` each (an effective batch size of 256) — a larger effective batch size often benefits from a correspondingly larger learning rate (a widely-used heuristic scales the learning rate roughly linearly with the effective batch size increase, often combined with the warmup techniques from Chapter 18 to stabilize the larger initial steps). This connects the scheduling concepts from Chapter 18 directly to multi-GPU scaling — a topic Chapter 23 revisits with the concrete, runnable launch mechanics needed to actually configure and test this.

---

## Summary

- Data parallelism replicates a model across GPUs, splits each batch across them, and synchronizes gradients to keep all replicas identical — distinct from model parallelism, which splits a single model across GPUs because it doesn't fit on one.
- `nn.DataParallel` is single-process, re-broadcasts full model weights every forward pass, and funnels gathering/optimization through a single primary GPU — usable for quick single-machine experimentation only, and explicitly discouraged by PyTorch's own documentation for anything beyond that.
- `DistributedDataParallel` runs one independent process per GPU, synchronizing only gradients via all-reduce, with that communication overlapped with backward computation — this architectural difference is why DDP scales meaningfully better, including across multiple machines/nodes, which `DataParallel` cannot support at all.
- The per-GPU `batch_size` and the effective global batch size (`local_batch_size × world_size`) are different quantities — learning rate and warmup schedules (Chapter 18) typically need adjustment when scaling to more GPUs, since the effective batch size changes even if the per-GPU batch size doesn't.

## Exercises

1. If available, run the same training script on a single GPU, then wrapped in `nn.DataParallel` across 2+ GPUs on one machine, and measure the actual speedup achieved — compare it against the theoretical speedup you'd expect from simply having more GPUs, and relate any gap to the primary-GPU bottleneck described in Section 22.2.
2. Draw (on paper, or as a diagram) the sequence of operations `DataParallel` performs on a single forward+backward step across 2 GPUs, and a separate diagram for what 2 independent DDP processes do — label where synchronization occurs in each, and where the primary-GPU bottleneck sits in the `DataParallel` diagram but has no equivalent in the DDP diagram.
3. Given a single-GPU training run tuned with `batch_size=32` and `lr=1e-3`, compute the effective batch size and a linearly-scaled learning rate for the same run distributed across 4 GPUs using DDP with the same per-GPU `batch_size=32`.
4. Research (via PyTorch's official documentation) what `"nccl"` stands for and why it's preferred over the alternative `"gloo"` backend specifically for GPU-based distributed training — summarize the distinction in your own words.

**Next:** Chapter 23 covers the practical mechanics of actually running DDP training — process groups, launching with `torchrun`, and correctly synchronizing checkpointing, logging, and evaluation across multiple processes.
