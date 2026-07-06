# Chapter 7: Datasets & DataLoaders

## Introduction

Every training loop so far in this book has quietly assumed data arrives as a ready-made batch of tensors. This chapter fills in the machinery that actually produces those batches: PyTorch's `Dataset` and `DataLoader` abstractions.

This pairing is deceptively simple on the surface — `Dataset` describes *how to get one sample*, `DataLoader` handles *batching, shuffling, and parallel loading* on top of it — but getting comfortable writing custom versions of both is one of the highest-leverage skills in practical PyTorch work, since almost no real project uses off-the-shelf data exactly as provided.

By the end of this chapter you'll be able to:
- Explain the division of responsibility between `Dataset` and `DataLoader`
- Write a custom `Dataset` for arbitrary data (tabular, image, text)
- Configure `DataLoader` correctly (batching, shuffling, `num_workers`, pinned memory)
- Write a custom `collate_fn` for data that doesn't batch trivially (e.g., variable-length sequences)
- Use `Sampler` objects to control the order and selection of samples

---

## 7.1 The Division of Labor

- **`Dataset`** is responsible for exactly one thing: given an index, return one sample (and its label, if applicable). It knows nothing about batching, shuffling, or parallelism.
- **`DataLoader`** wraps a `Dataset` and handles everything around it: grouping samples into batches, shuffling order each epoch, loading in parallel with worker processes, and combining individual samples into batched tensors.

This separation is deliberate — it means you can write a `Dataset` once and get batching, shuffling, and multiprocessing essentially for free from `DataLoader`, without touching that logic yourself.

---

## 7.2 The Map-Style `Dataset`

The most common type of dataset is **map-style**: it implements `__len__` (how many samples exist) and `__getitem__` (given an index, return that sample). This is the interface you'll write 95% of the time.

```python
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features   # tensor of shape (N, num_features)
        self.labels = labels       # tensor of shape (N,)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

features = torch.randn(100, 4)
labels = torch.randint(0, 2, (100,))
dataset = TabularDataset(features, labels)

print(len(dataset))       # 100
x, y = dataset[0]          # calls __getitem__(0) under the hood
print(x.shape, y)          # torch.Size([4]) tensor(1)
```

That's the entire contract. `DataLoader` (Section 7.4) calls `__getitem__` repeatedly, collects the results, and stacks them into a batch.

### A more realistic example: loading images from disk

Real datasets rarely fit in memory as pre-loaded tensors. A common pattern is to store only file paths in `__init__` and load + preprocess each image lazily, inside `__getitem__`, so you're never holding more than one batch's worth of raw image data in memory at once:

```python
from PIL import Image
import os

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.endswith((".jpg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)   # e.g., torchvision transforms
        return image
```

Loading lazily inside `__getitem__` — rather than loading everything up front in `__init__` — is what allows `DataLoader`'s worker processes (Section 7.5) to parallelize the expensive part (disk I/O and decoding) across multiple CPU cores.

---

## 7.3 The Iterable-Style `Dataset`

For data that doesn't have a natural fixed length or random-access index — streaming data, data read sequentially from a database cursor, or data too large to enumerate — PyTorch offers `IterableDataset`, which implements `__iter__` instead of `__len__`/`__getitem__`:

```python
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    def __init__(self, data_source):
        self.data_source = data_source   # e.g., a generator, file handle, or queue

    def __iter__(self):
        for record in self.data_source:
            yield preprocess(record)
```

Map-style datasets are the right default for the vast majority of use cases (anything that fits an index, even if lazily loaded from disk). Reach for `IterableDataset` specifically when your data is a genuine stream — for example, tailing a live sensor feed from your conveyor sorting system, or reading from a source where "how many samples are there" isn't a well-defined question ahead of time.

---

## 7.4 `DataLoader`: Batching, Shuffling, and Iteration

`DataLoader` wraps any `Dataset` and turns it into an iterable of batches:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for x_batch, y_batch in dataloader:
    print(x_batch.shape, y_batch.shape)
    break
# torch.Size([16, 4]) torch.Size([16])
```

Note what happened automatically: `DataLoader` called `dataset[i]` 16 times (in shuffled order), got back 16 individual `(feature_tensor, label)` pairs, and **stacked** them into batched tensors — a `(16, 4)` feature batch and a `(16,)` label batch. This default stacking behavior is called **collation**, and it's handled by a default `collate_fn` you can override (Section 7.6).

### Key `DataLoader` arguments

```python
DataLoader(
    dataset,
    batch_size=32,       # samples per batch
    shuffle=True,          # reshuffle every epoch — True for training, False for eval
    num_workers=4,          # parallel worker processes for loading (Section 7.5)
    pin_memory=True,         # speeds up host-to-GPU transfer (Section 7.5)
    drop_last=False,          # drop the final incomplete batch if dataset size isn't divisible by batch_size
)
```

**`shuffle=True` is for training; `shuffle=False` for validation/test loaders.** Shuffling training data each epoch prevents the model from learning spurious patterns tied to sample order (e.g., if your data happens to be sorted by class). Evaluation doesn't need shuffling — you want deterministic, reproducible ordering for computing metrics.

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

## 7.5 `num_workers` and `pin_memory`: Performance Knobs

### `num_workers`

By default, `num_workers=0` means data loading happens in the **main process**, synchronously — your GPU sits idle while the CPU loads and preprocesses the next batch. Setting `num_workers > 0` spawns separate worker processes that load data in parallel, ideally keeping a queue of ready batches so the GPU is never waiting.

```python
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

A reasonable starting point is `num_workers` equal to the number of CPU cores available, though the actual optimum depends on how expensive your `__getitem__` is (heavy image decoding benefits more from parallelism than simple tensor indexing) — profile rather than guess, using the tools from Chapter 20.

Two practical gotchas:
- On some platforms (notably Windows, and in certain notebook/multiprocessing contexts), `num_workers > 0` requires your dataset-loading code to be guarded by `if __name__ == "__main__":` to avoid recursive process spawning errors.
- More workers isn't always better — beyond a certain point, inter-process communication overhead and memory usage from duplicated dataset state outweigh the parallelism benefit. This matters directly on your multi-node GPU/NUC setup, where CPU cores may be shared across several training or inference processes at once.

### `pin_memory`

When `pin_memory=True`, `DataLoader` places batches in **page-locked (pinned) host memory**, which allows faster, asynchronous transfer to the GPU via `.to(device, non_blocking=True)`. This is a small but free win for GPU-bound training:

```python
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)

for x_batch, y_batch in dataloader:
    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)
```

`pin_memory` only helps when training on GPU — it's a no-op (and pure overhead) for CPU-only training, so leave it `False` in that case.

---

## 7.6 Custom `collate_fn`: Handling Data That Doesn't Batch Trivially

The default collation logic assumes every sample in a batch has the *same shape*, so it can stack them along a new dimension. This breaks immediately for variable-length data — the canonical example being sequences of different lengths, extremely relevant to any NLP or LLM fine-tuning work:

```python
sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9])]
labels = [0, 1, 0]

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

dataset = SequenceDataset(sequences, labels)
# DataLoader(dataset, batch_size=3)  -- would fail with the default collate_fn:
# RuntimeError: stack expects each tensor to be equal size
```

The fix is a custom `collate_fn`: a function that receives a list of individual samples (exactly what `__getitem__` returns, one per sample in the batch) and is responsible for combining them into a batch however makes sense for your data:

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    sequences, labels = zip(*batch)   # unzip list of (seq, label) tuples
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(s) for s in sequences])
    labels = torch.tensor(labels)
    return padded, labels, lengths

dataloader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)

for padded, labels, lengths in dataloader:
    print(padded)
    print(lengths)
```

```
tensor([[1, 2, 3, 0],
        [4, 5, 0, 0],
        [6, 7, 8, 9]])
tensor([3, 2, 4])
```

`pad_sequence` pads every sequence in the batch to the length of the longest one, and returning `lengths` alongside the padded tensor lets downstream code (e.g., an RNN using `pack_padded_sequence`, covered in Chapter 11, or an attention mask in a transformer, Chapter 13) know which positions are real data versus padding. This exact pattern — variable-length inputs, padding, and an accompanying mask/length tensor — is the standard shape of the data pipeline behind essentially all LLM tokenization and batching, including the kind of SFT data preparation used in your Qwen2.5-Coder fine-tuning work.

---

## 7.7 `Sampler`: Controlling Selection Order

Under the hood, `shuffle=True` is actually shorthand for using a `RandomSampler`; `shuffle=False` uses a `SequentialSampler`. You can supply a custom `Sampler` directly when you need more control than a boolean flag provides — for example, oversampling a minority class to address class imbalance:

```python
from torch.utils.data import WeightedRandomSampler

# Suppose class 0 has 90 samples and class 1 has 10 (imbalanced)
class_counts = torch.tensor([90, 10])
sample_weights = 1.0 / class_counts[labels]   # inverse frequency per-sample weight

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
```

Note: `sampler` and `shuffle` are mutually exclusive arguments — supplying a custom `sampler` means you're taking over ordering entirely, so `shuffle` must be left at its default (`False`).

---

## Summary

- `Dataset` defines how to fetch one sample (`__len__` + `__getitem__` for map-style; `__iter__` for streaming data); `DataLoader` handles batching, shuffling, and parallel loading on top of it.
- Load expensive resources (images, files) lazily inside `__getitem__`, not eagerly in `__init__`, so `DataLoader` workers can parallelize the loading.
- `shuffle=True` for training loaders, `shuffle=False` for validation/test loaders.
- `num_workers > 0` parallelizes data loading across processes; `pin_memory=True` speeds up host-to-GPU transfer — both are free performance wins for GPU training, worth tuning via profiling.
- A custom `collate_fn` is required whenever samples don't naturally stack into a uniform-shaped batch — most commonly, variable-length sequences that need padding.
- `Sampler` objects (like `WeightedRandomSampler`) give you fine-grained control over sample selection order, beyond the basic `shuffle` boolean.

## Exercises

1. Write a custom `Dataset` that wraps a list of `(text, label)` pairs, tokenizes each text into a list of integer IDs (a simple whitespace split + vocabulary lookup is fine), and returns `(token_ids_tensor, label)`.
2. Build a `DataLoader` for the dataset from Exercise 1 with a custom `collate_fn` that pads sequences and also returns an attention mask (a tensor of 1s for real tokens and 0s for padding).
3. Experiment with `num_workers` values (0, 2, 4) on a dataset with an artificially slow `__getitem__` (add a `time.sleep(0.01)` to simulate disk I/O) and measure the wall-clock time to iterate through one epoch for each setting.
4. Given a dataset with 3 classes in a 100:50:10 ratio, construct a `WeightedRandomSampler` that would sample all three classes roughly equally, and verify it empirically by counting class frequencies across a few thousand sampled indices.

**Next:** Chapter 8 assembles everything from Part II so far into a complete, deconstructed training loop — covering gradient clipping, checkpointing, and validation, building toward your first end-to-end model in Chapter 9.
