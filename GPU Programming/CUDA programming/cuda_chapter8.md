# Chapter 8 — Streams, Concurrency & Asynchronous Execution

## Learning objectives

By the end of this chapter you will:

- Understand CUDA's execution model — why kernels launch asynchronously
- Create and manage multiple CUDA streams to run operations concurrently
- Overlap host-to-device transfers with kernel computation using `cudaMemcpyAsync`
- Use CUDA events to synchronise between streams precisely
- Build a multi-stream pipeline that keeps the GPU fully saturated
- Understand the hardware limits on concurrency (copy engines, SMs)
- Profile stream overlap with Nsight Systems timeline view

---

## 8.1 The default execution model — deceptively sequential

Without streams, CUDA operations appear sequential from the CPU's perspective
but execute asynchronously on the GPU:

```python
import cupy as cp
import numpy as np

arr = cp.random.rand(10_000_000, dtype=cp.float32)

# These three lines return to the CPU IMMEDIATELY — GPU hasn't finished yet
result1 = cp.sum(arr)          # launch kernel 1
result2 = arr * 2.0            # launch kernel 2
result3 = cp.fft.fft(arr)      # launch kernel 3

# GPU executes all three SEQUENTIALLY on the default stream
# CPU is free to do other work while GPU runs
# But the three GPU ops themselves cannot overlap — same stream
```

The **default stream** (stream 0) is a single queue. Operations in the same
stream execute in FIFO order — each waits for the previous to finish.

```
Default stream timeline:
  ──────────────────────────────────────────────────────► time
  [  kernel 1  ][  kernel 2  ][  kernel 3  ]
                ↑             ↑
              waits          waits
```

This is fine for dependent operations (output of op 1 feeds into op 2).
It is wasteful for independent operations that could run simultaneously.

---

## 8.2 CUDA streams — independent execution queues

A CUDA stream is an ordered queue of GPU operations. Operations in different
streams can execute concurrently — the hardware scheduler decides how to
interleave them based on resource availability.

```
Stream 1:  [  kernel A  ][  kernel C  ]
Stream 2:        [  kernel B  ]
                 ↑
              can overlap with A and C if enough SMs available

Timeline:
  ──────────────────────────────────────────────────────► time
  Stream 1: [ kernel A ][ kernel C ]
  Stream 2:    [ kernel B ]
```

### Creating and using streams in CuPy

```python
import cupy as cp
import numpy as np

# Create two independent streams
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

N = 5_000_000
a = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)

# Operations assigned to stream1
with stream1:
    result_fft = cp.fft.fft(a)    # runs on stream1
    norm_a     = cp.linalg.norm(a) # runs on stream1, after fft

# Operations assigned to stream2 — run concurrently with stream1
with stream2:
    result_sum = cp.sum(b)         # runs on stream2
    sorted_b   = cp.sort(b)        # runs on stream2, after sum

# Wait for both streams to finish before using results
stream1.synchronize()
stream2.synchronize()

print(f"FFT done, norm: {float(norm_a):.4f}")
print(f"Sum: {float(result_sum):.4f}")
```

### Creating and using streams in Numba

```python
from numba import cuda
import numpy as np

stream1 = cuda.stream()
stream2 = cuda.stream()

N = 5_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# Transfer on specific streams
a_dev = cuda.to_device(a, stream=stream1)
b_dev = cuda.to_device(b, stream=stream2)

@cuda.jit
def scale(arr, factor):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] *= factor

threads = 256
blocks  = (N + threads - 1) // threads

# Launch kernels on separate streams
scale[blocks, threads, stream1](a_dev, 2.0)
scale[blocks, threads, stream2](b_dev, 3.0)

# Synchronise
stream1.synchronize()
stream2.synchronize()

a_result = a_dev.copy_to_host()
b_result = b_dev.copy_to_host()
```

---

## 8.3 The key use case — overlapping transfers with compute

The most impactful use of streams in practice is overlapping PCIe data
transfers with GPU computation. While the GPU processes batch N, the CPU
can be transferring batch N+1.

Modern GPUs have **two DMA copy engines** — one for host-to-device (H2D)
and one for device-to-host (D2H). This means:

```
Can overlap simultaneously:
  ✅ H2D transfer + kernel execution
  ✅ D2H transfer + kernel execution
  ✅ H2D transfer + D2H transfer + kernel execution

Cannot overlap:
  ❌ Two H2D transfers (only one H2D engine)
  ❌ Two D2H transfers (only one D2H engine)
  ❌ Kernel + kernel if they together exceed SM capacity
```

### The requirement: pinned (page-locked) memory

Asynchronous transfers only work with **pinned memory** — CPU memory that
the OS guarantees will not be paged out. Regular `numpy` arrays use pageable
memory, which requires the CUDA driver to make an intermediate copy first,
breaking the overlap.

```python
import numpy as np
import cupy as cp

N = 10_000_000

# Regular pageable memory — async transfer cannot truly overlap
arr_pageable = np.random.rand(N).astype(np.float32)

# Pinned memory — async transfer can overlap with kernel
arr_pinned = cp.cuda.alloc_pinned_memory(N * 4)   # allocate pinned buffer
arr_pinned_np = np.frombuffer(arr_pinned, dtype=np.float32, count=N)
np.copyto(arr_pinned_np, np.random.rand(N).astype(np.float32))

# Alternative: use numpy with pinned allocator
pinned_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
# Now any cp.cuda.alloc_pinned_memory allocations use the pool
```

---

## 8.4 Double buffering — the pipelined pattern

Double buffering is the fundamental pattern for keeping the GPU busy while
transferring data. You maintain two buffers and alternate between them:

```
Without pipelining (sequential):
  [H2D batch 0][kernel batch 0][D2H batch 0][H2D batch 1][kernel batch 1][D2H batch 1]
  Total time = N * (transfer + compute + transfer)

With double buffering (pipelined):
  [H2D batch 0]
               [H2D batch 1][kernel batch 0]
                            [H2D batch 2   ][kernel batch 1][D2H batch 0]
                                            [H2D batch 3   ][kernel batch 2][D2H batch 1]
  Total time ≈ max(transfer, compute) * N  (overlap hides the smaller one)
```

### Full double buffering implementation

```python
import cupy as cp
import numpy as np

def process_dataset_pipelined(dataset, process_fn, chunk_size=1_000_000):
    """
    Process a large dataset using double buffering.
    Overlaps H2D transfer of chunk N+1 with kernel on chunk N.

    dataset:    large numpy array on CPU
    process_fn: function that takes a CuPy array and returns a CuPy array
    chunk_size: elements per chunk (must fit in VRAM)
    """
    N       = len(dataset)
    n_chunks = (N + chunk_size - 1) // chunk_size

    # Two streams: compute and transfer run on different streams
    stream_compute  = cp.cuda.Stream()
    stream_transfer = cp.cuda.Stream()

    # Two device buffers — alternate between them
    buf0 = cp.empty(chunk_size, dtype=cp.float32)
    buf1 = cp.empty(chunk_size, dtype=cp.float32)
    buffers = [buf0, buf1]

    # Results storage (CPU side)
    results = np.empty(N, dtype=np.float32)

    # Pinned host memory for async transfers
    pinned_in  = [cp.cuda.alloc_pinned_memory(chunk_size * 4) for _ in range(2)]
    pinned_out = [cp.cuda.alloc_pinned_memory(chunk_size * 4) for _ in range(2)]

    def get_chunk(chunk_idx):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, N)
        return dataset[start:end], start, end

    # ── Startup: transfer first chunk ─────────────────────────────────────
    data0, s0, e0 = get_chunk(0)
    chunk_len0 = e0 - s0

    # Copy to pinned buffer
    pin_arr0 = np.frombuffer(pinned_in[0], dtype=np.float32, count=chunk_len0)
    np.copyto(pin_arr0, data0)

    # Async H2D transfer on transfer stream
    with stream_transfer:
        buf0[:chunk_len0].set(pin_arr0)

    # ── Main pipeline loop ────────────────────────────────────────────────
    for chunk_idx in range(n_chunks):
        curr_buf = chunk_idx % 2
        next_buf = 1 - curr_buf

        data_curr, s_curr, e_curr = get_chunk(chunk_idx)
        chunk_len_curr = e_curr - s_curr

        # Wait for current chunk's transfer to finish before computing
        stream_compute.wait_event(stream_transfer.record())

        # Launch kernel on current chunk (compute stream)
        with stream_compute:
            out_gpu = process_fn(buffers[curr_buf][:chunk_len_curr])

        # While kernel runs: transfer NEXT chunk (transfer stream)
        if chunk_idx + 1 < n_chunks:
            data_next, s_next, e_next = get_chunk(chunk_idx + 1)
            chunk_len_next = e_next - s_next

            pin_arr_next = np.frombuffer(
                pinned_in[next_buf], dtype=np.float32, count=chunk_len_next
            )
            np.copyto(pin_arr_next, data_next)

            with stream_transfer:
                buffers[next_buf][:chunk_len_next].set(pin_arr_next)

        # Copy result back to CPU (also overlapped with next transfer)
        with stream_compute:
            out_gpu.get(out=np.frombuffer(
                pinned_out[curr_buf], dtype=np.float32, count=chunk_len_curr
            ))

        # Sync and collect result for this chunk
        stream_compute.synchronize()
        results[s_curr:e_curr] = np.frombuffer(
            pinned_out[curr_buf], dtype=np.float32, count=chunk_len_curr
        )

    stream_transfer.synchronize()
    return results


# ── Benchmark: pipelined vs sequential ────────────────────────────────────────
N = 20_000_000
dataset = np.random.rand(N).astype(np.float32)

def process_fn(x):
    return cp.sqrt(cp.abs(x)) * 2.0 + cp.sin(x)

# Sequential baseline
t0 = cp.cuda.Event(); t1 = cp.cuda.Event()
t0.record()
x_gpu  = cp.asarray(dataset)
out    = process_fn(x_gpu)
result_seq = cp.asnumpy(out)
t1.record(); t1.synchronize()
t_seq = cp.cuda.get_elapsed_time(t0, t1)

# Pipelined
import time
wall_start = time.perf_counter()
result_pip = process_dataset_pipelined(dataset, process_fn, chunk_size=2_000_000)
wall_end   = time.perf_counter()
t_pip = (wall_end - wall_start) * 1000

print(f"Sequential: {t_seq:.1f} ms")
print(f"Pipelined:  {t_pip:.1f} ms")
print(f"Speedup:    {t_seq/t_pip:.2f}x")
print(f"Results match: {np.allclose(result_seq, result_pip, rtol=1e-4)}")
```

---

## 8.5 CUDA events — precision synchronisation

Events are timestamps you can insert into any stream. They let you:
- Measure elapsed time between two points in GPU execution
- Synchronise between streams (stream B waits for event in stream A)
- Check whether a GPU operation has completed without blocking the CPU

```python
import cupy as cp

stream_a = cp.cuda.Stream()
stream_b = cp.cuda.Stream()

arr = cp.random.rand(5_000_000, dtype=cp.float32)

# ── Cross-stream synchronisation ──────────────────────────────────────────────
with stream_a:
    x = cp.fft.fft(arr)               # expensive op on stream A

# Record event in stream A when FFT finishes
event_fft_done = stream_a.record()

# Stream B must wait for the FFT before it can use x
stream_b.wait_event(event_fft_done)   # non-blocking on CPU
                                       # GPU enforces the ordering

with stream_b:
    norm = cp.linalg.norm(x)           # safe: guaranteed FFT has finished

stream_b.synchronize()
print(f"Norm: {float(norm):.4f}")


# ── Timing between streams ────────────────────────────────────────────────────
ev_start = cp.cuda.Event()
ev_mid   = cp.cuda.Event()
ev_end   = cp.cuda.Event()

with stream_a:
    ev_start.record()
    heavy_op1 = cp.sort(arr)
    ev_mid.record()

with stream_b:
    heavy_op2 = cp.fft.fft(arr)
    ev_end.record()

ev_end.synchronize()

t_sort = cp.cuda.get_elapsed_time(ev_start, ev_mid)
t_total = cp.cuda.get_elapsed_time(ev_start, ev_end)
print(f"Sort:       {t_sort:.2f} ms")
print(f"Total wall: {t_total:.2f} ms")
print(f"Overlap savings: {(t_sort - t_total + cp.cuda.get_elapsed_time(ev_mid, ev_end)):.2f} ms")
```

### Event synchronisation rules

```
stream.record()          → insert event into stream's queue
                           returns immediately to CPU
                           GPU timestamps when it processes this point

stream.wait_event(ev)    → GPU: do not start subsequent ops until ev fires
                           CPU: returns immediately (non-blocking)

ev.synchronize()         → CPU: BLOCK until GPU reaches this event
                           use at the end to collect results

ev.done                  → CPU: check without blocking (True/False)
                           use for polling patterns
```

---

## 8.6 Stream priorities

CUDA supports high and low priority streams. High-priority work preempts
low-priority work on the same SM when resources are contended:

```python
import cupy as cp

# Get priority range (device-specific)
# Most GPUs support priorities 0 (highest) to -1 (lowest)
hi_stream  = cp.cuda.Stream(priority=0)    # high priority
lo_stream  = cp.cuda.Stream(priority=-1)   # low priority

arr_large = cp.random.rand(50_000_000, dtype=cp.float32)
arr_small = cp.random.rand(1_000,      dtype=cp.float32)

# Submit large low-priority work first
with lo_stream:
    result_big = cp.sort(arr_large)    # long-running

# Then submit small high-priority work
with hi_stream:
    result_small = cp.sum(arr_small)   # short but urgent

# High-priority stream may finish before the sort even though
# it was submitted second — hardware can preempt/interleave
hi_stream.synchronize()
print(f"Small result ready: {float(result_small):.4f}")
lo_stream.synchronize()
```

Stream priorities matter for inference servers where latency-sensitive
requests (interactive queries) should not be blocked by background tasks
(logging, preprocessing, cache warming).

---

## 8.7 The null stream and synchronisation gotchas

The **null stream** (stream 0, the default) has a special property: it
synchronises with ALL other streams. This makes it a synchronisation
barrier, which can silently destroy the overlap you expect.

```python
import cupy as cp

stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

arr = cp.random.rand(10_000_000, dtype=cp.float32)

# ── This DOES overlap (two non-default streams) ───────────────────────────────
with stream1:
    a = cp.fft.fft(arr)

with stream2:
    b = cp.sort(arr)     # concurrent with FFT ✅

stream1.synchronize()
stream2.synchronize()

# ── This does NOT overlap (null stream forces serialisation) ──────────────────
with stream1:
    a = cp.fft.fft(arr)

c = cp.sum(arr)          # default (null) stream — waits for stream1 to finish
                          # AND forces stream1 to wait for c before continuing ❌

with stream1:
    d = arr * 2.0        # waits for null stream op to complete
```

**Rule:** If you want true concurrency, all operations must be on explicit
non-default streams. A single operation on the default stream serialises
everything around it.

In Numba, create explicit streams with `cuda.stream()`. Pass them to
kernel launches and `to_device()` / `copy_to_host()` calls.

---

## 8.8 Profiling stream overlap with Nsight Systems

Nsight Systems shows a timeline view where you can visually verify that
transfers and kernels overlap:

```bash
# Profile a Python script
nsys profile --trace=cuda,nvtx python your_script.py
nsys-ui report1.nsys-rep   # open GUI
```

What to look for in the timeline:
```
Streams row:   each stream gets its own horizontal lane
CUDA HW row:   shows actual hardware execution

Signs of good overlap:
  Stream 1 kernel bar overlaps with Stream 2 kernel bar
  MemcpyH2D bar overlaps with kernel bar in another stream

Signs of no overlap:
  All bars are sequential with no horizontal overlap
  → check for null stream usage or missing wait_event()
```

Adding NVTX markers lets you label regions in the timeline:

```python
import cupy as cp

# Mark regions for Nsight Systems
cp.cuda.nvtx.RangePush("Data preprocessing")
preprocessed = cp.sqrt(cp.abs(data))
cp.cuda.nvtx.RangePop()

cp.cuda.nvtx.RangePush("FFT computation")
spectrum = cp.fft.fft(preprocessed)
cp.cuda.nvtx.RangePop()
```

---

## 8.9 When streams help vs when they don't

```
Streams HELP when:
  ✅ Independent operations that don't share data
  ✅ Overlapping large H2D/D2H transfers with compute
  ✅ Multiple small kernels that together fill less than one SM's capacity
  ✅ Producer-consumer pipelines (preprocess → compute → postprocess)

Streams DON'T HELP when:
  ❌ Operations are data-dependent (must be sequential regardless)
  ❌ A single kernel already saturates all SMs (no room for concurrent kernel)
  ❌ Transfers use pageable memory (async copies silently become synchronous)
  ❌ One stream uses the null stream (serialises everything)

The saturation rule:
  A 100% occupancy kernel leaves no SM capacity for a second stream's kernel.
  Concurrent kernels only help when each is <50–70% occupancy.
  → check occupancy with Nsight Compute before expecting stream speedup
```

---

## 8.10 Practical pattern library

### Pattern 1 — Parallel independent computations

```python
import cupy as cp

streams = [cp.cuda.Stream() for _ in range(4)]
results = [None] * 4
data    = [cp.random.rand(2_000_000, dtype=cp.float32) for _ in range(4)]

for i, (s, d) in enumerate(zip(streams, data)):
    with s:
        results[i] = cp.fft.fft(d)   # all 4 FFTs run concurrently

for s in streams:
    s.synchronize()
```

### Pattern 2 — Producer-consumer with events

```python
import cupy as cp

preprocess_stream = cp.cuda.Stream()
compute_stream    = cp.cuda.Stream()

raw_data = cp.random.rand(10_000_000, dtype=cp.float32)

# Stage 1: preprocess
with preprocess_stream:
    cleaned = cp.abs(raw_data - cp.mean(raw_data))

# Record event after preprocessing
preprocess_done = preprocess_stream.record()

# Stage 2: compute — wait for preprocessing, then run concurrently
# with any remaining preprocessing work
compute_stream.wait_event(preprocess_done)

with compute_stream:
    result = cp.fft.fft(cleaned)
    norm   = cp.linalg.norm(result)

compute_stream.synchronize()
print(f"Result norm: {float(norm):.4f}")
```

### Pattern 3 — Inference server simulation

```python
import cupy as cp
import numpy as np
import queue
import threading

def inference_server(model_fn, batch_size=32, n_batches=20):
    """
    Simulates an inference server:
    - Transfer stream: load batches from CPU to GPU
    - Inference stream: run model
    - Result stream: copy results back
    All three overlap.
    """
    transfer_stream = cp.cuda.Stream()
    infer_stream    = cp.cuda.Stream()
    result_stream   = cp.cuda.Stream()

    # Simulate batches of input data
    batches = [np.random.rand(batch_size, 512).astype(np.float32)
               for _ in range(n_batches)]

    all_results  = []
    device_bufs  = [cp.empty((batch_size, 512), dtype=cp.float32) for _ in range(2)]
    result_bufs  = [cp.empty((batch_size, 256), dtype=cp.float32) for _ in range(2)]

    for i, batch in enumerate(batches):
        buf_idx = i % 2

        # Transfer current batch
        with transfer_stream:
            device_bufs[buf_idx].set(batch)

        ev_loaded = transfer_stream.record()
        infer_stream.wait_event(ev_loaded)

        # Run inference
        with infer_stream:
            output = model_fn(device_bufs[buf_idx])
            result_bufs[buf_idx][:] = output

        ev_done = infer_stream.record()
        result_stream.wait_event(ev_done)

        # Copy result back
        with result_stream:
            cpu_result = cp.asnumpy(result_bufs[buf_idx])

        result_stream.synchronize()
        all_results.append(cpu_result)

    return all_results


def simple_model(x):
    """Toy model: linear transform + ReLU."""
    W = cp.random.rand(512, 256, dtype=cp.float32) * 0.01
    return cp.maximum(x @ W, 0)

results = inference_server(simple_model, batch_size=64, n_batches=10)
print(f"Processed {len(results)} batches, each shape {results[0].shape}")
```

---

## Hands-on exercises

### Exercise 1 — Measure true async behaviour

Verify that kernel launches are truly asynchronous from the CPU:

```python
import cupy as cp
import time

N = 100_000_000
arr = cp.random.rand(N, dtype=cp.float32)

# Measure CPU time to "launch" vs actual GPU completion time
cpu_start = time.perf_counter()
result = cp.sort(arr)              # launch (returns immediately)
cpu_end   = time.perf_counter()

gpu_start = cp.cuda.Event(); gpu_end = cp.cuda.Event()
gpu_start.record()
result2 = cp.sort(arr)
gpu_end.record()
gpu_end.synchronize()

print(f"CPU launch time:   {(cpu_end - cpu_start)*1000:.3f} ms")
print(f"GPU execution time: {cp.cuda.get_elapsed_time(gpu_start, gpu_end):.3f} ms")
# Expected: CPU launch << GPU execution — confirms async launch
```

### Exercise 2 — Two-stream overlap

Verify that two independent operations on separate streams actually overlap:

```python
import cupy as cp

N = 20_000_000
a = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)

stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

# Measure sequential time (same stream)
ev0 = cp.cuda.Event(); ev1 = cp.cuda.Event()
ev0.record()
with stream1:
    r1 = cp.fft.fft(a)
    r2 = cp.fft.fft(b)   # same stream — sequential
stream1.synchronize()
ev1.record(); ev1.synchronize()
t_seq = cp.cuda.get_elapsed_time(ev0, ev1)

# Measure concurrent time (different streams)
ev0.record()
with stream1:
    r1 = cp.fft.fft(a)
with stream2:
    r2 = cp.fft.fft(b)   # different stream — concurrent
stream1.synchronize()
stream2.synchronize()
ev1.record(); ev1.synchronize()
t_par = cp.cuda.get_elapsed_time(ev0, ev1)

print(f"Sequential: {t_seq:.2f} ms")
print(f"Concurrent: {t_par:.2f} ms")
print(f"Overlap speedup: {t_seq/t_par:.2f}x")
# Note: speedup depends on GPU occupancy of FFT kernel
# If FFT already saturates all SMs, expect ~1x
```

### Exercise 3 — Pipeline benchmark

Compare sequential vs pipelined processing using the
`process_dataset_pipelined` function from section 8.4.

Test with three chunk sizes and record the speedup:

```python
N = 50_000_000
dataset = np.random.rand(N).astype(np.float32)

def process_fn(x):
    return cp.tanh(x * 2.0) + cp.sin(x)

for chunk_size in [500_000, 2_000_000, 10_000_000]:
    # Time sequential and pipelined
    # Record: chunk_size, t_seq (ms), t_pip (ms), speedup
    pass

# Questions:
# Which chunk size gives the best speedup?
# What happens when chunk_size is very small? Very large?
# At what chunk size does the pipelining stop helping?
```

### Exercise 4 — Event-based producer-consumer

Build a three-stage pipeline:
1. **Preprocess** (stream A): normalise input data to zero mean unit variance
2. **Compute** (stream B): apply FFT after preprocessing completes
3. **Postprocess** (stream C): compute power spectrum (abs² of FFT) after FFT completes

Use events to enforce ordering while allowing maximum concurrency:

```python
import cupy as cp

N = 10_000_000
raw = cp.random.rand(N, dtype=cp.float32) * 100 + 50   # unnormalised

stream_pre   = cp.cuda.Stream()
stream_fft   = cp.cuda.Stream()
stream_post  = cp.cuda.Stream()

# Your implementation:
# 1. preprocess on stream_pre
# 2. record event, make stream_fft wait
# 3. FFT on stream_fft
# 4. record event, make stream_post wait
# 5. power spectrum on stream_post
# 6. synchronise and collect result

# Verify: power spectrum values should all be non-negative
# Verify: total power (sum of power spectrum) ≈ N * var(normalised input)
```

### Exercise 5 — Null stream trap

This code is supposed to overlap two FFTs but silently does not.
Find and fix the bug:

```python
import cupy as cp

N = 20_000_000
a = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)

stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

ev_start = cp.cuda.Event()
ev_end   = cp.cuda.Event()

ev_start.record()         # ← this records on the DEFAULT stream

with stream1:
    r1 = cp.fft.fft(a)

with stream2:
    r2 = cp.fft.fft(b)

stream1.synchronize()
stream2.synchronize()

ev_end.record()           # ← this records on the DEFAULT stream

ev_end.synchronize()
t = cp.cuda.get_elapsed_time(ev_start, ev_end)
print(f"Time: {t:.2f} ms")   # probably equals sequential time — why?

# Fix:
# 1. Identify why ev_start.record() on the default stream is the problem
# 2. Rewrite the timing to use stream-specific events
# 3. Verify the timing now shows overlap
```

---

## Chapter summary

| Concept | Key takeaway |
|---|---|
| Default stream | Sequential FIFO queue — operations cannot overlap within it |
| Non-default streams | Independent queues — hardware schedules them concurrently |
| Pinned memory | Required for true async H2D/D2H overlap — pageable silently degrades |
| Double buffering | Transfer N+1 while computing N — hides transfer latency behind compute |
| `stream.record()` | Insert GPU timestamp into stream — returns immediately to CPU |
| `stream.wait_event()` | GPU ordering constraint — non-blocking on CPU |
| `ev.synchronize()` | CPU blocks until GPU reaches this event — use to collect results |
| Null stream trap | Default stream synchronises with ALL streams — silently kills overlap |
| Saturation limit | Concurrent kernels only help if each is <50–70% SM occupancy |
| Stream priorities | High-priority streams preempt low-priority — useful for inference servers |

---

## What's next

Chapter 9 covers **profiling with Nsight Systems and Nsight Compute** —
you will learn to read roofline models, interpret the SM efficiency and
memory throughput metrics, find the true bottleneck in any kernel, and
apply a structured optimisation workflow. With Chapters 5–8 complete you
now have real kernels to profile, which makes Chapter 9 immediately
practical rather than abstract.

Paste your Exercise 2 overlap speedup and Exercise 3 pipeline benchmark
table — the numbers will depend on your GPU's copy engine bandwidth and
SM count, and are worth discussing before moving to the profiling chapter.
