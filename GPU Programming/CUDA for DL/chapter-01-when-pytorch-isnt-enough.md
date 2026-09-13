# Chapter 1 — When PyTorch Just Isn't Enough

*Part 0: Foundations & Mental Models. Maps to Book Ch. 1, sections 1.1–1.8 (confirmed against the actual liveBook chapter brief).*

This chapter has no kernels to write yet — that starts in Chapter 2. Its job is to build the mental model everything else in the course hangs off: what CUDA actually is, when it's worth reaching for, and why "memory, not FLOPs" is the sentence you'll come back to in almost every later chapter.

---

## 1.1 What is CUDA?

**CUDA** (Compute Unified Device Architecture) is NVIDIA's parallel-computing platform: a set of C/C++ language extensions, a compiler toolchain (`nvcc`), and a runtime/driver API for managing GPUs as general-purpose parallel processors, not just graphics chips. NVIDIA introduced it in 2006–2007.

It's useful to see where it sits in the stack you already use every day:

```
PyTorch / JAX  (Python, autograd, op dispatch)
      │
cuBLAS / cuDNN / cuBLASLt / CUTLASS  (NVIDIA's hand-tuned kernel libraries)
      │
CUDA runtime + driver API  (device mgmt, memory, kernel launch, streams)
      │
PTX  (a stable, architecture-independent virtual ISA — think "GPU bytecode")
      │
SASS  (the actual machine code for one specific GPU architecture)
      │
Physical GPU hardware
```

When you call `torch.matmul(a, b)`, PyTorch is (in the common case) dispatching straight into cuBLAS. When you write your own `.cu` file, `nvcc` compiles your `__global__` functions down through PTX to SASS for the architecture you target. Higher-level tools you already use, like Triton, also compile down to PTX — so "writing CUDA" and "writing Triton" ultimately produce the same kind of machine code, just via a different, higher-level front end.

**Compute capability** is the version number that matters most in this course, because it gates which hardware features are available to a kernel:

| GPU (yours / reference) | Architecture | Compute Capability | Notes |
|---|---|---|---|
| Tesla T4 | Turing | 7.5 | WMMA tensor cores supported (CC 7.0+) |
| RTX 3090 | Ampere (consumer, GA102) | 8.6 | 3rd-gen tensor cores, WMMA |
| A100 | Ampere (datacenter, GA100) | 8.0 | Referenced throughout the book for comparison |
| H100 | Hopper | 9.0 | WGMMA + TMA async tensor cores — **required** for Part 6 Ch. 19 |
| B200 | Blackwell (datacenter) | 10.0 | FP4 tensor cores — **required** for Part 11's capstone |

This table is the reason the curriculum flagged Parts 6 and 11 as cloud-GPU territory: your RTX 3090 (CC 8.6) and T4 (CC 7.5) will run everything through WMMA tensor cores just fine, but WGMMA is a Hopper-only instruction, and the FP4 CUTLASS example is Blackwell-only.

CUDA is also **NVIDIA-only**. Vendor-neutral alternatives exist (OpenCL, SYCL, Vulkan compute, ROCm/HIP for AMD), but essentially the entire production LLM training/inference stack today (cuBLAS, cuDNN, FlashAttention, CUTLASS, Megatron, vLLM's custom kernels) is CUDA, which is the practical justification for the whole course.

## 1.2 When Do You Need Custom CUDA?

The honest starting position is: **you usually don't**. cuBLAS, cuDNN, and PyTorch's built-in fused kernels represent years of NVIDIA engineering, and for standard ops (matmul, convolution, standard attention) they will beat a hand-rolled kernel written in an afternoon. `torch.compile` (which generates Triton kernels via Inductor) closes most of the remaining gap for arbitrary user code automatically.

Custom CUDA earns its cost when one of these is true:

1. **A genuinely new op** — a novel attention variant, an unusual quantization scheme, a custom activation with a nonstandard backward pass — that no library implements yet.
2. **Kernel fusion** — several small ops (say, a mask, a softmax, and a matmul) currently run as separate kernel launches, each round-tripping its result through slow global memory. Fusing them into one kernel avoids those intermediate writes/reads entirely. This is the *entire* reason FlashAttention exists (Part 7).
3. **Bleeding-edge hardware features libraries haven't caught up to** — e.g., writing raw WGMMA/TMA code on Hopper, or FP4 on Blackwell, before cuBLAS/CUTLASS fully exploit it for your exact shape.
4. **Unusual shapes or access patterns** general-purpose library kernels aren't tuned for — very small batch sizes, ragged sequences, sparse patterns.
5. **You're studying GPU performance itself** — which, pragmatically, is also why you and I are doing this course.

And the corresponding "don't bother" case: if PyTorch eager mode or `torch.compile` already saturates the roofline (see 1.4.1) for your op and shape, hand-written CUDA is pure maintenance cost with no payoff. This chapter's whole point is to give you the judgment to tell these cases apart *before* you spend a weekend writing a kernel — the book's own hook, "recognizing when you need a custom kernel instead of an existing library," is exactly this section.

## 1.3 CUDA Basics

### 1.3.1 Host and Device: Two Worlds Working Together

**Host** = the CPU and its RAM, running your OS and orchestrating the program.
**Device** = the GPU and its own, physically separate DRAM (GDDR6/GDDR6X on consumer cards like your 3090, HBM2e/HBM3 on datacenter cards like the A100/H100).

Because host and device memory are separate address spaces, data must be explicitly copied between them (`cudaMemcpy`, or `cudaMallocManaged` for a page-migrated "unified memory" convenience layer that's simpler but often slower in hot loops). They're connected by:

- **PCIe** (PCIe 4.0 x16 ≈ 25–32 GB/s per direction in practice) for a typical workstation/server link — this is what your RTX 3090 uses to talk to its host CPU.
- **NVLink** (900 GB/s per H100 SXM GPU) for tightly-coupled multi-GPU nodes — this is what makes Part 10's near-linear multi-GPU scaling possible.

Put concrete numbers next to each other and the design implication jumps out: your RTX 3090's *own* memory bus delivers 936 GB/s to its GDDR6X, while the PCIe link back to the host delivers roughly 30x less. Every unnecessary host↔device round trip, and every extra kernel launch that could have been fused, costs you disproportionately more than the same work done entirely on-device. This single fact motivates kernel fusion (1.2), the entire optimization ladder (1.6), and Flash Attention (Part 7).

Execution is asynchronous by default: the host issues a kernel launch into a **stream** (an ordered work queue) and immediately continues running host code — it does *not* wait for the kernel to finish unless you hit an explicit sync point (`cudaDeviceSynchronize()`) or an implicit one (a blocking `cudaMemcpy`). This asynchrony is what pipeline parallelism (Part 10, Ch. 29) exploits directly.

### 1.3.2 Kernels: Functions That Run on Thousands of Threads

A **kernel** is a function marked `__global__`, compiled for the device, that — when launched — is executed by many threads simultaneously, each running identical code (this execution model is called **SIMT**: Single Instruction, Multiple Thread) but usually operating on a different slice of data based on its own coordinates.

Launch syntax: `kernel<<<gridDim, blockDim>>>(args)`, where `gridDim` is the number of thread **blocks** and `blockDim` is the number of **threads** per block; both can be 1D, 2D, or 3D (`dim3`).

The hardware reality underneath that syntax:

- Threads execute in groups of **32**, called a **warp** — this number is fixed across every current NVIDIA architecture. A warp's 32 threads execute in lockstep on a warp scheduler; if threads in the same warp branch differently (**divergence**), the warp serializes both paths — still correct, just no longer fully parallel.
- A thread block is scheduled onto a single SM (Streaming Multiprocessor) and stays there for its whole lifetime — no mid-execution migration. How many blocks an SM can run *concurrently* depends on how many registers and how much shared memory each block consumes (this is **occupancy**, a concept you'll measure directly in Nsight Compute in Part 9).
- Every thread computes its own global index, almost always via some variant of:
  ```cuda
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ```
  This one line is the workhorse of nearly every CUDA kernel you'll ever write, and it's exactly what opens Chapter 2's vector-add kernel.
- Kernel launches aren't free — expect on the order of a few microseconds of fixed overhead each. For small ops launched thousands of times (as happens across the many tiny layers of an LLM during inference), that overhead compounds, which is another concrete argument for fusion.

## 1.4 Recognizing Parallel Opportunities

### 1.4.1 The Memory Hierarchy: Your Performance Bottleneck

Ordered fastest/smallest/closest-to-the-cores → slowest/largest/farthest:

| Tier | Scope | Approx. size | Approx. bandwidth / latency |
|---|---|---|---|
| Registers | per-thread | a few hundred KB per SM, split across resident threads | fastest, ~0 cycles |
| Shared memory / L1 | per-SM (programmer-managed scratchpad) | up to ~164–228 KB per SM (Ampere/Hopper, split with L1) | ~single-digit cycles, TB/s-class aggregate |
| L2 cache | chip-wide, shared by all SMs | tens of MB (e.g. ~40 MB on A100, ~50 MB on H100) | hundreds of cycles |
| Global memory (device DRAM) | whole GPU | 16 GB (T4) – 24 GB (3090) – 80 GB (A100/H100) | 320 GB/s (T4) – 936 GB/s (3090) – ~2.0 TB/s (A100) – 3.35 TB/s (H100 SXM) |
| Host memory | over PCIe/NVLink | system RAM | ~25–32 GB/s (PCIe4 x16) – up to 900 GB/s (NVLink) |

The central lesson of this section — and honestly of the entire book — is that **almost no modern GPU kernel is limited by how many FLOPs the cores can compute; it's limited by how fast data can be fed to them from this hierarchy.** This is formalized as the **roofline model**: compute a kernel's **arithmetic intensity** (FLOPs performed per byte moved to/from memory), and compare it against the GPU's *ridge point* (peak FLOPs ÷ peak bandwidth). Below the ridge point you're **memory-bound**; above it you're **compute-bound**.

Worked example: a naive vector add does 1 FLOP per element but moves 12 bytes (read 2×4 bytes, write 1×4 bytes) → arithmetic intensity ≈ 0.083 FLOP/byte. That's nowhere near any GPU's ridge point — it's deeply memory-bound on a T4, a 3090, or an H100 alike. A well-tiled GEMM, by contrast, reuses each value loaded into shared memory/registers many times before it's evicted, pushing arithmetic intensity up toward the compute-bound region — which is *exactly* why GEMM optimization (Part 5–6) is about maximizing data reuse in the fast tiers, not about launching more threads.

### 1.4.2 When Parallelism Doesn't Help

- **Serial dependency chains** — e.g. a recurrent scan where step *t* needs step *t−1*'s result — resist naive parallelization outright; they need restructuring (parallel-scan / associative-operator tricks) to parallelize at all.
- **Problems too small to amortize overhead** — if total work doesn't exceed the fixed kernel-launch + PCIe-transfer cost, moving to GPU is a net loss versus just running on CPU.
- **Heavy warp divergence** — algorithms with highly data-dependent per-element control flow (some sparse/graph algorithms) can lose most of a GPU's theoretical throughput to serialized divergent branches.
- **Bandwidth already saturated** — once a memory-bound kernel has saturated the memory bus, adding more threads does nothing; you need to change the *access pattern* (coalescing, caching, fusion), not the thread count.

This section's real message, which sets up the entire back half of the course: **"parallel" does not automatically mean "fast."** You have to identify what's actually limiting a given operation *before* you optimize it — which is the whole premise of profiling-driven optimization (Part 9, Nsight Compute).

## 1.5 Deep Learning Through the CUDA Lens

### 1.5.1 The Nature of Deep Learning Computation

Neural nets decompose, at the tensor-op level, into a small, repeated set of primitives: dense matrix multiplication (linear layers; attention's QKᵀ and softmax·V), elementwise nonlinearities (GELU/SiLU/ReLU), normalization (LayerNorm/RMSNorm), reductions (softmax, max/top-k for pooling and sampling), and pure data movement (embedding lookups, transpose, KV-cache concatenation).

Nearly every one of these applies the *same* operation independently across a huge number of elements/rows/tokens — a near-perfect match for SIMT execution, in sharp contrast to control-flow-heavy general-purpose code, which GPUs handle poorly.

Training adds a second full pass — **backpropagation** — that mirrors the forward pass operation-for-operation in reverse. Practically, this means almost every custom forward kernel you write eventually needs a matching backward kernel to be trainable end-to-end inside autograd — a theme Part 3 (hand-rolled backprop) and Part 4 (wiring custom kernels into PyTorch's autograd) both build around directly.

### 1.5.2 Why GPUs Excel at Deep Learning

The design-philosophy contrast: a CPU core spends most of its transistor budget on control logic, branch prediction, and large caches to make *one* thread of arbitrary code fast. A GPU SM spends its budget on many simple ALUs (CUDA cores) plus, since Volta (2017), dedicated **Tensor Cores** — betting that the workload has enough parallelism to keep thousands of simple lanes busy rather than needing one very smart lane.

Put your own hardware's numbers next to that claim: an RTX 3090 has 82 SMs × 128 FP32 CUDA cores = **10,496 CUDA cores**, plus 328 3rd-generation Tensor Cores — versus a high-end CPU's tens of cores. That's the SIMT trade explicitly: raw lane count over single-thread sophistication, which is exactly what a batch of matmuls over thousands of tokens wants.

Tensor Cores themselves are a *separate* hardware unit from ordinary CUDA cores — they execute small, fixed-shape matrix-multiply-accumulate operations (e.g. a 16×16×16 FP16→FP32 MMA) at far higher throughput than the same math expressed as generic FP32 CUDA-core instructions. That's precisely why Part 6 (Tensor Cores / WMMA / WGMMA) is treated as a distinct optimization tier sitting *above* ordinary CUDA-core GEMM tuning, not a variation of it.

And memory bandwidth shows up here too: single-sequence/small-batch LLM *decode* is dominated by reading model weights from device memory once per generated token — i.e. it's **memory-bandwidth-bound, not FLOP-bound.** That's exactly why NVIDIA's datacenter roadmap (A100 → H100 → H200/B200) has pushed memory bandwidth (~2.0 → 3.35 → higher TB/s) as hard as raw compute, and why every later "optimize this kernel" chapter in this course starts by asking the 1.4.1 question first: *is this memory-bound or compute-bound?*

## 1.6 The Optimization Stack

### 1.6.1 Starting Point: Naive Kernels

Every kernel in this course starts as a **naive** version: one thread per output element, no shared-memory reuse, no vectorization — verified correct against a CPU reference before anything else is touched. Correctness first, speed second. This is formalized in Part 2 (Chapter 6) as "CPU-first development," and it's the same discipline the book's own repo follows kernel-by-kernel.

### 1.6.2 Optimization Layers

1. **Memory access pattern** — coalesce global-memory accesses (adjacent threads touch adjacent addresses) so the hardware services a warp's loads in as few transactions as possible.
2. **Data reuse via shared memory** — stage a tile once into an SM's shared memory and let every thread in the block reuse it, cutting redundant global-memory traffic. This is the core idea behind every "tiled"/"blocked" GEMM.
3. **Register-level tiling & vectorization** — have each thread compute several output elements and use wide loads (`float4`, etc.) to move more bytes per instruction and better hide latency.
4. **Warp-level primitives** — `__shfl_*` intrinsics for intra-warp reductions/broadcasts without touching shared memory at all; used heavily in the optimized softmax/layernorm/top-k kernels (Part 5).
5. **Tensor Cores** — hand the actual multiply-accumulate work to dedicated hardware (WMMA on Ampere, WGMMA+TMA on Hopper) once the surrounding data movement is already efficient.
6. **Library/template level (CUTLASS)** — once you're hand-tuning tile sizes, pipeline stages, and swizzled layouts for the last 10–20%, you're re-deriving what CUTLASS already encodes as templates. The book's final chapter treats CUTLASS as the tier above hand-written kernels.

### 1.6.3 The Compounding Effect

These layers stack, they aren't alternatives — the book's own measured GEMM progression (naive → coalesced → shared-memory tiled → 1D block-tiled → 2D block-tiled → vectorized → cuBLAS-with-tensor-cores → WMMA → WGMMA) is a real example of exactly this: each rung typically buys another meaningful multiple of throughput, with the cumulative gain from naive to fully tuned commonly running into the 10–100× range *before* even reaching Tensor Cores.

This is also why profiling (Part 9) has to happen *between* rungs, not just at the end: each layer removes a specific bottleneck, and applying the wrong fix to an already-solved bottleneck wastes effort or actively regresses performance (e.g. adding more shared-memory tiling to a kernel that's now register-pressure-bound rather than memory-bound).

### 1.6.4 Scaling to Multiple GPUs

Once a single GPU's kernel is well-tuned, the next axis is spreading work across GPUs: **tensor parallelism** (shard one large matmul's rows/columns across GPUs, synchronized via NCCL all-reduce/all-gather over NVLink or InfiniBand) and **pipeline parallelism** (assign different layers to different GPUs, overlap batches in flight via CUDA streams). This only pays off once single-GPU kernels are already efficient — multiplying a slow kernel across 8 GPUs just produces a slow result 8× more expensively, which is exactly why this course places Part 10 (distributed) *after* Parts 5–8 (single-GPU optimization), not before.

### 1.6.5 Why This Matters

Every layer above translates directly into either training wall-clock time (cost) or inference latency/throughput (user experience and serving cost) for real LLMs. None of this is academic — it's the literal difference between a model that trains in days versus weeks, or serves at acceptable latency versus not.

## 1.7 Getting Ready to Build

### 1.7.1 Our Roadmap

The arc ahead, mapped to this curriculum's Parts:

**Part 1** (environment + first kernels) → **Part 2** (every core DL op from scratch, CPU-verified) → **Part 3** (hand-rolled backprop, full MNIST MLP) → **Part 4** (wiring custom kernels into real PyTorch autograd for a transformer) → **Part 5** (the GEMM/softmax/layernorm/top-k optimization ladder) → **Part 6** (Tensor Cores) → **Part 7** (Flash Attention) → **Part 8** (quantization) → **Part 9** (profiling mastery) → **Part 10** (multi-GPU) → **Part 11** (CUTLASS capstone).

## 1.8 Summary

- CUDA is NVIDIA's parallel-computing platform; your code compiles through PTX to architecture-specific SASS. Compute capability gates which hardware features (WMMA, WGMMA, FP4) are available to you.
- Reach for custom CUDA for new ops, fusion, bleeding-edge hardware features, or unusual shapes — not to reinvent what cuBLAS/cuDNN already do well.
- Host and device are separate memory spaces connected by a comparatively slow link (PCIe or NVLink); minimizing round trips and kernel-launch count matters as much as the kernel's own code.
- A kernel is executed by thousands of SIMT threads, grouped into 32-thread warps that execute in lockstep; divergence and occupancy are hardware realities, not abstractions.
- Almost everything is about the **memory hierarchy**, not raw FLOPs: classify every kernel as memory-bound or compute-bound before optimizing it.
- Deep learning's core ops are naturally data-parallel, which is why GPUs — and especially their dedicated Tensor Cores — excel at it; LLM inference in particular is usually memory-bandwidth-bound.
- Optimization is a stack of compounding layers (coalescing → shared memory → register tiling/vectorization → warp primitives → tensor cores → CUTLASS), applied in order, profiled between steps.
- Single-GPU efficiency is a prerequisite for multi-GPU scaling, not a parallel concern.

---

## Hands-On: Getting Your Environment Ready

No kernel code yet — Chapter 2 opens with `vecadd.cu`. Today's "lab" is confirming your toolchain and characterizing the hardware you'll be running every later chapter on.

```bash
# 1. Confirm the driver + max supported CUDA version
nvidia-smi

# 2. Confirm the installed CUDA Toolkit / compiler version
nvcc --version

# 3. Build and run the CUDA Samples' deviceQuery (clone once, from NVIDIA's cuda-samples repo)
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery
make
./deviceQuery
```

`deviceQuery` prints, per GPU: compute capability, SM count, total global memory, memory bus width/clock (from which you can hand-compute peak bandwidth), max threads per block, shared memory per block, and warp size. Run it on **both** of your machines (RTX 3090 box and the T4) and record the results — you'll compare against these numbers directly in Part 5–6 when kernels start reporting achieved-vs-peak bandwidth and TFLOPS.

## Exercises

1. **Bound classification.** For each of the following, state whether it's memory-bound or compute-bound and justify with an arithmetic-intensity argument: (a) elementwise vector add, (b) a 4096×4096×4096 FP32 GEMM, (c) an embedding table lookup for a batch of tokens, (d) LayerNorm over a hidden dimension of 4096, (e) top-k selection over a 128k-vocabulary logits vector.
2. **Roofline math.** Vector add moves 12 bytes per FLOP (arithmetic intensity ≈ 0.083). Compute the ridge point (peak FLOPs ÷ peak bandwidth) for your RTX 3090, your T4, and an H100 SXM (use the FP32 CUDA-core FLOPS figures, not tensor-core FLOPS). Confirm vector add sits far below all three ridge points on every device.
3. **Know your hardware.** Run `deviceQuery` (or equivalent) on your own GPU(s). Record compute capability, SM count, and memory bandwidth. Based on the compute-capability table in 1.1, list which later-course features (WMMA, WGMMA, FP4 tensor cores) each of your GPUs can and can't run natively.
4. **When to go custom.** Pick three operations in a hypothetical LLM inference server (e.g., RMSNorm, KV-cache append, top-p sampling). For each, argue — using the five criteria from 1.2 — whether a custom fused CUDA kernel is likely worth writing, or whether you'd expect PyTorch/cuBLAS/`torch.compile` to already be good enough.

---

**Next:** Chapter 2 — GPU Memory Management and Kernel Launch (Part 1). We'll write and run the first real kernel: scalable vector addition, with 1D/2D/3D indexing and explicit host↔device memory management.
