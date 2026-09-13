# Chapter 11 — Multi-GPU & Distributed Training

*Part 10: Multi-GPU & Distributed Training. Confirmed from the book's companion repo: `book.cu/8_distributed/README.md`. This folder's own files self-label as "Chapter 10" — one less than this course's Chapter 11, the same numbering drift flagged since Chapter 1. One small naming note: the top-level README's directory diagram calls the third example's folder `pipeline/`; the actual folder on disk is `pipeline_parallel/` — a harmless documentation/reality mismatch, not worth more than a mention.*

Three examples, confirmed directly: **tensor parallelism across 8 GPUs on one node**, **tensor parallelism across 16 GPUs on two nodes**, and **pipeline parallelism across up to 8 GPUs using CUDA streams.** This is explicitly this course's designated cloud-GPU chapter — the hardware requirements are real (single-node examples want 8×H100-class GPUs; multi-node wants two such nodes over InfiniBand), so treat the code below as something to read closely and reason about even if you can't run all of it on your own RTX 3090/T4 pair.

---

## 11.1 Tensor Parallelism: Splitting the Contraction Dimension

`tensor_parallel.cu`'s own docstring is more precise than the top-level README's looser summary, and it's the one worth trusting: the weight matrix `B` is split **by rows** (the `K` dimension — the dimension being *contracted* in the matmul), and the input `A` is split **by columns** to match. Every GPU computes a **partial product** — a full-size `M×N` matrix, but each entry holds only `1/P` of that entry's true inner-product sum — and `ncclAllReduce` with `ncclSum` combines those partial sums so every GPU ends up holding the *complete* result. This is the standard "row-parallel linear layer" pattern from Megatron-style tensor parallelism.

**Node-local GPU assignment** is a real, non-obvious detail worth understanding on its own:

```cuda
// Ranks on the same node share that node's GPUs, so device selection must use
// the node-local rank, not the global one. MPI_Comm_split_type groups ranks by
// shared-memory domain (i.e., by node).
MPI_Comm local_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
int local_rank, local_size;
MPI_Comm_rank(local_comm, &local_rank);
MPI_Comm_size(local_comm, &local_size);
// ...
CHECK_CUDA(cudaSetDevice(local_rank));   // NOT the global `rank`
```

On a 16-rank, 2-node job, global ranks 0–7 live on node 0 and 8–15 live on node 1. If every rank naively called `cudaSetDevice(rank)`, ranks 8–15 would try to select device indices 8–15 on node 1 — which only has devices 0–7. `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)` groups ranks that share memory (i.e., that live on the same physical node) into their own sub-communicator, giving each rank a **node-local** rank number that correctly indexes into that node's own 0–7 GPU range.

**NCCL bootstrap** follows the canonical pattern for initializing a communicator across MPI ranks:

```cuda
ncclUniqueId nccl_id;
if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&nccl_id));   // only rank 0 generates the ID
MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);  // MPI distributes it to everyone
ncclComm_t nccl_comm;
CHECK_NCCL(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));  // every rank joins using the shared ID
```

MPI handles *process launching and coordination*; NCCL handles the actual *GPU-to-GPU data path* (NVLink within a node, InfiniBand/Ethernet between nodes) — this bootstrap sequence is exactly where the two libraries' responsibilities hand off to each other.

The full tensor-parallel "step" is a two-line fusion of the two ideas — a local GEMM on this rank's shard, followed by a collective that combines every rank's partial result:

```cuda
auto tp_step = [&]() {
    CHECK_CUBLAS(cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, Kp,
                            &alpha, d_A, M, d_B, Kp, &beta, d_C, M));
    CHECK_NCCL(ncclAllReduce(d_C, d_C, static_cast<size_t>(M) * N,
                             ncclHalf, ncclSum, nccl_comm, stream));
};
```

And correctness is checked the same way Chapters 4 and 8 both checked their own precision-sensitive kernels: the distributed FP16 result is compared against an FP32 `cublasSgemm` reference computed once on rank 0, confirming the tensor-parallel result carries the *same* level of FP16 rounding error as an equivalent single-GPU FP16 computation would — not additional error introduced by the sharding or the collective itself.

## 11.2 Honest Scaling Metrics: Isolating Communication from Compute

This is the real methodology worth internalizing, and it's Chapter 10's profiling lesson applied with nothing but careful timer placement instead of a profiler: **time the local GEMM alone, then time the full step (GEMM + AllReduce), and subtract.**

```cuda
double comm_ms = step_ms_max - gemm_ms_max;
double speedup = baseline_ms / step_ms_max;
double efficiency = 100.0 * speedup / world_size;
// ...
std::cout << "AllReduce share of step: " << (100.0 * comm_ms / step_ms_max) << "%" << std::endl;
```

Confirmed real results from the top-level README:

| Configuration | Efficiency | Interconnect | Bandwidth |
|---|---|---|---|
| 8 GPUs, single node | ~100% | NVLink | ~600 GB/s |
| 16 GPUs, two nodes | ~99.8% | InfiniBand | ~25 GB/s (200 Gb/s) |

That's a **24× lower bandwidth** interconnect between nodes, and yet total scaling efficiency barely moves. The explanation is a multi-GPU-scale version of Chapter 1 §1.4.1's roofline argument: AllReduce's communication cost is a genuinely small *fraction* of total step time whenever the local GEMM (compute) is large relative to the amount of data being reduced — `K=4096`-scale FP16 GEMMs at hundreds of thousands of GFLOPS per GPU dwarf the cost of all-reducing one `M×N` matrix, even over a much slower link. The same "does data movement or computation dominate this operation" question that classified individual *kernels* as memory- or compute-bound back in Chapter 1 applies one level up, at the *collective-communication* scale — compute-per-step versus communication-volume-per-step.

## 11.3 Pipeline Parallelism: Overlapping Batches with Streams

`pipeline.cu` assigns one MLP layer per GPU (every GPU runs Linear+ReLU except the last, which runs Linear+Softmax) and compares two ways of driving batches through that pipeline. The naive version's own comments name its problem directly, twice:

```cuda
void process_batch_naive(GPULayer* layers, int num_gpus, float* h_input, float* h_output) {
    CUDA_CHECK(cudaMemcpy(layers[0].d_input_naive, h_input, /*...*/, cudaMemcpyHostToDevice));
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(layers[i].gpu_id));
        forward_linear_naive(&layers[i]);
        if (layers[i].has_relu) forward_relu_naive(&layers[i]);
        if (layers[i].has_softmax) forward_softmax_naive(&layers[i]);

        // --- Synchronization Point (MAJOR BOTTLENECK) ---
        CUDA_CHECK(cudaDeviceSynchronize());

        if (i < num_gpus - 1) {
            CUDA_CHECK(cudaMemcpy(layers[i+1].d_input_naive, layers[i].d_output_naive,
                                  /*...*/, cudaMemcpyDeviceToDevice));   // blocking D2D copy
            // --- Synchronization Point (MAJOR BOTTLENECK) ---
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    // ...copy final output back to host...
}
```

Two full-device synchronization points *per layer* mean every GPU sits completely idle while every other GPU works — with 4 GPUs, each one is active roughly a quarter of the time. The async version replaces every blocking call with its non-blocking counterpart, and replaces "wait for the whole device" with "wait for exactly the one event this data actually depends on":

```cuda
void process_batch_async(GPULayer* layers, int num_gpus, int batch_id, /*...*/) {
    int s = batch_id % NUM_STREAMS_PER_GPU;     // this batch's assigned stream (round-robin)
    CUDA_CHECK(cudaMemcpyAsync(layers[0].d_input[s], h_input, /*...*/, layers[0].streams[s]));

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(layers[i].gpu_id));
        if (i > 0) {
            // Wait ONLY for the previous layer's work on THIS micro-batch — nothing more.
            CUDA_CHECK(cudaStreamWaitEvent(layers[i].streams[s], layers[i-1].events[s], 0));
            CUDA_CHECK(cudaMemcpyAsync(layers[i].d_input[s], layers[i-1].d_output[s],
                                       /*...*/, cudaMemcpyDeviceToDevice, layers[i].streams[s]));
        }
        forward_linear_stream(&layers[i], s);
        if (layers[i].has_relu) forward_relu_stream(&layers[i], s);
        if (layers[i].has_softmax) forward_softmax_stream(&layers[i], s);

        // Mark this layer's work on this micro-batch done — the NEXT layer waits on this, not on the whole device.
        CUDA_CHECK(cudaEventRecord(layers[i].events[s], layers[i].streams[s]));
    }
    // ...async copy final output back to host...
}
```

`NUM_STREAMS_PER_GPU = 4` means up to four micro-batches can be genuinely in flight simultaneously, each pinned to its own stream. While GPU 3 finishes batch *N*'s final layer, GPU 2 can already be computing batch *N+1*, GPU 1 batch *N+2*, GPU 0 batch *N+3* — the same "staircase" software-pipelining idea a CPU's instruction pipeline uses, here applied across physically separate devices, with `cudaStreamWaitEvent`/`cudaEventRecord` enforcing only the *true* data dependencies (this layer's input genuinely needs the previous layer's output) rather than an entire device's worth of unrelated work.

## 11.4 A Genuinely Surprising Real Result

The pipeline example's own README reports a full scaling table across 1, 2, 4, and 8 GPUs, and it's worth reading closely rather than skimming past, because it doesn't behave the way "more GPUs, more speedup" intuition predicts:

| GPUs | Naive (samples/s) | Streams (samples/s) | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 601,000 | 1,285,000 | 2.14× | 214% |
| 2 | 323,000 | 1,010,000 | 3.13× | 157% |
| 4 | 167,000 | 698,000 | 4.17× | 104% |
| 8 | 84,000 | 279,000 | 3.32× | 42% |

Two things in this table are worth explaining honestly rather than glossing over:

**Efficiency exceeds 100% at 1, 2, and 4 GPUs.** That's not a measurement error — it's a real artifact of what "efficiency" means here (`speedup ÷ GPU count`) combined with what's actually producing the speedup. Even at **1 GPU**, `NUM_STREAMS_PER_GPU=4` lets four micro-batches overlap *on that same single device* — the async version's speedup at N=1 comes entirely from hiding memory-transfer latency behind other batches' compute via stream-level concurrency, and has **nothing to do with multi-GPU parallelism at all.** Dividing a real, legitimate 2.14× speedup by a GPU count of 1 mechanically produces "214%" — the metric formula itself is a slightly awkward fit at N=1, where there's no cross-GPU parallelism baseline being measured against in the first place. The underlying speedup is real; the *percentage-of-GPU-count* framing is what produces a number that looks odd.

**Efficiency collapses to 42% at 8 GPUs**, down from 104% at 4. This one *is* the diminishing-returns story Chapter 1 §1.6.4 warned about, playing out at the pipeline-stage level instead of the single-kernel level: more GPUs means more pipeline *stages*, which means more inter-GPU synchronization events per batch and a longer relative pipeline fill/drain period against a fixed `NUM_BATCHES=256` — overhead that grows with stage count, eating into the steady-state throughput gain the deeper pipeline was supposed to buy. The lesson generalizes directly: **scaling out (more GPUs) and scaling deep (more pipeline stages) both carry a synchronization tax that grows with the thing you're adding more of** — Chapter 1's warning about multiplying a slow kernel across more GPUs, and this chapter's own warning about chaining more pipeline stages, are the same shape of caution at two different levels of the stack.

---

## Hands-On Lab

*This chapter's examples genuinely need multi-GPU hardware — treat the following as a scaled-down or reasoning-based lab depending on what you have access to.*

1. **If you have 2–4 GPUs on one machine** (even consumer cards over PCIe, no NVLink required for correctness — just less overlap headroom): build and run `pipeline.cu` at `num_gpus = 1, 2, 4` and see whether you reproduce the qualitative shape of the table in §11.4 (real speedup from stream overlap even at low GPU counts) on your own hardware.
2. **Profile the async pipeline with `nsys`** (Chapter 10's tool) across at least 2 GPUs, and look directly at the timeline for the "staircase" overlap pattern — confirm visually that batch *N+1* genuinely begins on GPU *i−1* before batch *N* finishes on GPU *i*.
3. **Without multi-node access**, work through `tensor_parallel.cu`'s "honest scaling metrics" timing code by hand: using Chapter 1's bandwidth table (NVLink ~600 GB/s vs. InfiniBand ~25 GB/s here), estimate how much longer you'd expect the AllReduce step alone to take on the 16-GPU/InfiniBand configuration versus the 8-GPU/NVLink one, for a `K=4096` FP16 GEMM's output size.

## Exercises

1. **Explain the necessity of `MPI_Comm_split_type`.** What would actually go wrong — be specific about which rank, which GPU, and which error — if every rank on a 2-node, 16-rank job called `cudaSetDevice(rank)` using the *global* MPI rank instead of the node-local one?
2. **Quantify the interconnect gap, then explain why it barely matters.** Using §11.2's "AllReduce share of step" methodology and Chapter 1's bandwidth figures, estimate the AllReduce-only slowdown you'd expect moving from NVLink to InfiniBand (roughly 24×), then explain in one paragraph why total step *efficiency* barely drops despite that large per-collective slowdown.
3. **Derive the >100%-efficiency result precisely.** Using `NUM_STREAMS_PER_GPU=4`, explain mathematically why a single-GPU pipelined run can beat a single-GPU naive run by a real, meaningful factor with zero multi-GPU parallelism involved — then propose an alternative metric (not `speedup ÷ GPU count`) that wouldn't produce a number exceeding 100% in this specific case.
4. **Predict before you test.** If you changed `NUM_STREAMS_PER_GPU` from 4 to 8, would you expect throughput to keep improving, plateau, or regress? Justify your answer using Chapter 1's memory-hierarchy/occupancy reasoning (more concurrent streams means more simultaneously-resident buffers and work — what resource runs out first?), then check your prediction if you have the hardware to run it.
5. **Sketch overlapping tensor parallelism's own communication.** `tensor_parallel.cu`'s `tp_step` treats the local GEMM and the AllReduce as strictly sequential. Sketch, in words or pseudocode, what it would take to overlap one layer's AllReduce with the *next* layer's local GEMM — applying pipeline parallelism's core idea (§11.3) to tensor parallelism's own communication step.

---

**Next:** Chapter 12 — CUTLASS & Production-Grade Kernels (Part 11, the capstone). Every hand-written kernel across this entire course — from Chapter 3's naive GEMM through Chapter 7's hand-rolled WGMMA — has been building toward the same question: at what point does hand-tuning give way to a template library that already encodes all of it? This final chapter answers that directly, including the Blackwell FP4 GEMM the book uses to close out its own arc.
