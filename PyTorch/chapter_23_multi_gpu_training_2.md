# Chapter 23: Multi-GPU Training II

## Introduction

Chapter 22 explained *why* `DistributedDataParallel` is structured the way it is — one process per GPU, gradients synchronized via all-reduce. This chapter covers the practical mechanics of actually running that setup: launching multiple processes correctly with `torchrun`, setting up process groups, and handling the details that only matter once you have several processes running concurrently — checkpointing without every process fighting to write the same file, logging without duplicated output, and evaluation without duplicated or inconsistent metrics. This is the chapter that turns Chapter 22's conceptual understanding into a script you can actually run across your multi-node setup.

By the end of this chapter you'll be able to:
- Launch a DDP training job correctly using `torchrun`
- Understand `rank`, `local_rank`, and `world_size`, and use them correctly
- Use `DistributedSampler` to correctly partition data across processes
- Checkpoint, log, and evaluate correctly in a multi-process setting
- Reason about single-node multi-GPU vs. multi-node DDP configurations

---

## 23.1 `rank`, `local_rank`, and `world_size`, Precisely

These three values, briefly introduced in Chapter 22, need precise definitions before writing real launch code, especially once multiple *nodes* (machines) are involved — directly relevant to a setup spanning your GPU server and NUC cluster:

- **`world_size`**: the total number of processes across the *entire* training job, across all nodes combined. If you have 2 machines with 2 GPUs each, `world_size = 4`.
- **`rank`**: a unique global identifier for a process, from `0` to `world_size - 1`, unique across the *entire* job.
- **`local_rank`**: a process's identifier *within its own node* — if node A has GPUs 0 and 1, its two processes have `local_rank` 0 and 1, regardless of their global `rank`.

For a 2-node, 2-GPU-per-node setup, the mapping looks like:

| Node | Local Rank | Global Rank |
|---|---|---|
| Node A | 0 | 0 |
| Node A | 1 | 1 |
| Node B | 0 | 2 |
| Node B | 1 | 3 |

`local_rank` is what you use for `torch.cuda.set_device(local_rank)` — selecting which physical GPU *on this machine* the process should use. `rank` is what identifies the process globally, used for things like deciding which single process should handle checkpointing or logging (Section 23.4), since you want exactly one such process across the *whole job*, not one per node.

---

## 23.2 Launching with `torchrun`

`torchrun` (PyTorch's standard distributed launch utility, superseding the older `torch.distributed.launch`) handles spawning the correct number of processes and setting the relevant environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and the coordination address) automatically — your training script reads these from the environment rather than computing them manually.

### Single-node, multi-GPU

```bash
torchrun --standalone --nproc_per_node=2 train.py
```

`--nproc_per_node=2` spawns 2 processes on this machine (typically matching your GPU count); `--standalone` is a convenience flag for single-node jobs that handles the coordination setup automatically, without needing to specify network addresses explicitly.

### Multi-node

```bash
# On the "master" node (e.g., your GPU server):
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
    --master_addr="192.168.1.10" --master_port=29500 train.py

# On the second node (e.g., a NUC):
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 \
    --master_addr="192.168.1.10" --master_port=29500 train.py
```

`--nnodes=2` and `--node_rank` (0 for the first node, 1 for the second — set explicitly per machine, since each node needs to know its own position) together establish the full multi-node topology; `--master_addr`/`--master_port` point every node at the same coordination endpoint (conventionally hosted on node 0) so all processes across all machines can find each other and establish communication. This is the concrete mechanism by which a training job could span your GPU server and one or more NUCs as genuinely cooperating processes in a single distributed run, rather than separate, independent jobs.

### Reading the launch-provided values in your script

```python
import os
import torch
import torch.distributed as dist

def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size
```

`torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` as environment variables automatically for every process it spawns — your script reads them rather than computing or hardcoding them, which is precisely what makes the *same*, unmodified script correctly runnable across a single-GPU debug run, a single-node multi-GPU run, and a full multi-node run, purely by changing the `torchrun` command-line arguments used to launch it.

---

## 23.3 `DistributedSampler`: Correctly Partitioning Data

Chapter 7's `DataLoader` with `shuffle=True` isn't sufficient once multiple processes are each pulling from the same dataset — without coordination, different processes could end up drawing overlapping or duplicated samples within an epoch, undermining the "each process handles a distinct slice of the effective batch" assumption Chapter 22's gradient math depends on. `DistributedSampler` solves this by deterministically partitioning the dataset across processes based on `rank` and `world_size`:

```python
from torch.utils.data import DataLoader, DistributedSampler

train_dataset = ...   # any Dataset from Chapter 7

sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,          # PER-GPU batch size, Chapter 22, Section 22.5
    sampler=sampler,          # note: sampler handles shuffling now, not the shuffle= argument
    num_workers=4,
)
```

Two details that are easy to get wrong:

- **`shuffle=` on the `DataLoader` itself must be omitted (left at its default `False`) when using a `sampler`** — this is the same `sampler`/`shuffle` mutual exclusivity noted for `WeightedRandomSampler` back in Chapter 7, Section 7.7, and it applies identically here. Shuffling is controlled by `DistributedSampler`'s own `shuffle=True` argument instead.
- **`sampler.set_epoch(epoch)` must be called at the start of every epoch**, or every process will use the *same* shuffled order on every epoch, defeating the purpose of shuffling entirely:

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)   # ensures a different shuffle order each epoch, consistently across processes
    for x_batch, y_batch in train_loader:
        # ... training step ...
        pass
```

`set_epoch()` feeds the epoch number into `DistributedSampler`'s internal random seed, so every process independently derives the *same* shuffled ordering for that epoch (important — all processes need a consistent, non-overlapping partition) while still getting a genuinely different order from one epoch to the next (important — for the same reason shuffling matters in Chapter 7 generally, avoiding the model learning spurious patterns tied to a fixed sample order).

---

## 23.4 Checkpointing, Logging, and Evaluation: The `rank == 0` Pattern

With multiple processes all running the same script, most side effects that should happen exactly once — saving a checkpoint, writing a log line, printing a progress update — need to be explicitly restricted to a single process, conventionally `rank == 0` ("the main process"). Without this, every process would redundantly perform the same file write or print the same log line multiple times, and worse, multiple processes writing to the same checkpoint file concurrently risks file corruption.

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    ddp_model.train()

    for x_batch, y_batch in train_loader:
        # ... standard training step, Chapter 8 ...
        pass

    # Only rank 0 checkpoints and logs -- every other process skips this entirely
    if rank == 0:
        torch.save({
            "model_state_dict": ddp_model.module.state_dict(),   # .module -- see below
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }, "checkpoint.pt")
        print(f"Epoch {epoch} complete, checkpoint saved.")
```

**`ddp_model.module`, not `ddp_model`, when accessing the underlying model** — wrapping a model in `DDP` (Chapter 22, Section 22.3) adds a layer of wrapping around it, exactly analogous in spirit to how `nn.Sequential` or any composed module wraps its children (Chapter 5); the actual `FashionMNIST_CNN` instance (or whatever your model class is) lives at `ddp_model.module`, and that's what you want `.state_dict()` from for a clean, DDP-independent checkpoint — one that can be loaded later into a plain, non-distributed model for inference (Chapter 9's `predict()` pattern) without any DDP-specific wrapping getting in the way.

### Validation loss: aggregating across processes

Evaluation is subtler than checkpointing, because each process only sees its own slice of the validation set (via its own `DistributedSampler`) — if you want a single, correct validation loss representing the *entire* validation set (not just whatever fraction `rank 0` happened to see), the per-process partial results need to be explicitly combined, using the same all-reduce primitive from Chapter 22's gradient synchronization:

```python
def evaluate(ddp_model, val_loader, loss_fn, device, world_size):
    ddp_model.eval()
    local_loss_sum = torch.tensor(0.0, device=device)
    local_count = torch.tensor(0, device=device)

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = ddp_model(x_batch)
            loss = loss_fn(output, y_batch)
            local_loss_sum += loss * x_batch.size(0)
            local_count += x_batch.size(0)

    # Combine every process's partial sum into a single, globally-correct total
    dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    global_val_loss = (local_loss_sum / local_count).item()
    return global_val_loss   # identical, correct value on every process after this point
```

`dist.all_reduce` here is used directly and explicitly, rather than happening implicitly inside `.backward()` as it does for gradients — this is the same underlying communication primitive from Chapter 22, Section 22.3, just invoked manually for a value (validation loss) that DDP has no built-in awareness of needing to synchronize. After `all_reduce`, every process holds the identical, globally-correct sum — which is why it's safe to then compute `global_val_loss` and, if desired, only have `rank == 0` actually print or log it, since all processes now agree on the same number regardless of which one does the printing.

---

## 23.5 A Complete, Minimal DDP Training Script

Putting the pieces together into a runnable script (save as `train_ddp.py`, launched via the `torchrun` commands from Section 23.2):

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = FashionMNIST_CNN().to(device)   # Chapter 10
    ddp_model = DDP(model, device_ids=[local_rank])

    train_dataset = ...   # Chapter 9's FashionMNIST setup
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        ddp_model.train()

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(ddp_model(x_batch), y_batch)
            loss.backward()   # gradients synchronized automatically, Chapter 22
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} complete.")

    dist.destroy_process_group()   # clean shutdown, releasing distributed resources

if __name__ == "__main__":
    main()
```

`dist.destroy_process_group()` at the end is worth including deliberately — it releases the communication resources (network connections, NCCL state) the process group was holding, ensuring a clean exit rather than leaving dangling resources that could interfere with a subsequent run on the same node.

---

## Summary

- `world_size` is the total process count across the entire job; `rank` uniquely identifies a process globally; `local_rank` identifies it within its own node — `local_rank` selects the physical GPU, `rank` identifies the process for global coordination decisions (like which one checkpoints).
- `torchrun` spawns processes and sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE` as environment variables automatically; the same script runs unmodified across single-GPU, single-node-multi-GPU, and multi-node configurations purely via different `torchrun` arguments.
- `DistributedSampler` (used instead of `shuffle=True`) partitions data correctly across processes; `sampler.set_epoch(epoch)` must be called every epoch or shuffling silently stops varying across epochs.
- Restrict checkpointing, logging, and printing to `rank == 0` to avoid redundant work and file-write conflicts; access the underlying model via `ddp_model.module` for a clean, DDP-independent `state_dict()`.
- Aggregate metrics like validation loss across processes explicitly with `dist.all_reduce`, since each process otherwise only sees its own local slice of the data.

## Exercises

1. Launch the complete script from Section 23.5 with `torchrun --standalone --nproc_per_node=N` for whatever GPU count is available to you, and confirm via added print statements (guarded appropriately) that each process reports a distinct `rank`/`local_rank` and processes a distinct, non-overlapping subset of data each epoch.
2. Deliberately omit `sampler.set_epoch(epoch)` from the training loop, and add logging to print the first few sample indices each process sees each epoch — confirm the ordering is now identical across epochs, and explain why that undermines shuffling's purpose from Chapter 7.
3. Extend the `evaluate()` function from Section 23.4 to also correctly aggregate accuracy (correct prediction count), not just loss, across all processes using `dist.all_reduce`.
4. Using two machines on the same local network (or by simulating with `--node_rank` values and `localhost` for testing purposes), configure and run the multi-node `torchrun` commands from Section 23.2, confirming both nodes join the same training job and produce consistent, synchronized results.

**Next:** Chapter 24 covers memory optimization techniques — gradient checkpointing, activation offloading, and systematic OOM debugging — for training larger models than would otherwise fit in available GPU memory, whether on a single GPU or across the distributed setup covered in this chapter.
