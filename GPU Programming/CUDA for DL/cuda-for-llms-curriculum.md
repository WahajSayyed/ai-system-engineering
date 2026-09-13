# CUDA GPU Programming for LLMs — Course Curriculum

**Primary source:** *CUDA for LLMs* — Elliot Arledge (Manning Publications, MEAP started Jan 2026, print est. Oct 2026, ISBN 9781633434899)
**Companion code:** [github.com/Infatoshi/book.cu](https://github.com/Infatoshi/book.cu)
**Supplementary sources:** see the [Expert Resources Bibliography](#expert-resources-bibliography) at the end.

> Research note: Manning doesn't publish a full public TOC for this title, so this curriculum was reconstructed from (a) the actual liveBook chapter-1 brief, (b) explicit chapter labels found inside the companion GitHub repo's READMEs (e.g. "Chapter 02: GPU Memory Management," "Chapter 04," "Chapter 8: Quantization," "Chapter 10: Distributed Computing," "Chapter 11" for CUTLASS), and (c) the repo's actual kernel files, which let me confirm exact progressions (e.g. the GEMM ladder `0_cublas → 1_naive → 2_gmem_coalesce → 3_smem_blocking → 4_1d_blocktiling → 5_2d_blocktiling → 6_vectorize → 7_cublas_tc → 8_wmma → 9_wgmma → ... → 12_wgmma_max_tiles`, which is a near-exact match to Simon Boehm's famous "How to Optimize a CUDA Matmul Kernel" blog). Where the book's own repo docs disagreed on numbering (they drifted during the MEAP's 8 months of revisions), I resolved it toward the most recently/explicitly labeled chapters and noted the gap (Ch. 9, profiling) that the book's blurb promises but which has no dedicated code folder.
>
> The book ships **56 kernels**, a PyTorch C++ extension pipeline, hand-rolled backprop ending in a single-file MNIST MLP, and coverage through Ampere/Hopper/Blackwell. I've expanded its ~11 book chapters into **30 finer-grained lessons across 12 parts** so each session is a manageable, single-topic unit with theory + a hands-on lab + exercises — the same granularity as your other curricula (Triton, FastAPI, DDIA, PyTorch).

---

## Who this is for / prerequisites

- Comfortable in Python and C/C++; no prior CUDA required (matches the book's stated audience).
- A CUDA-capable NVIDIA GPU for Parts 0–8 (your RTX 3090 box and T4 cover almost everything through Ampere-era tensor cores and quantization).
- **Hopper (sm_90, H100)** is required for WGMMA (Part 6, Ch. 19) and the CUTLASS Hopper examples (Part 11). **Blackwell (sm_100, B200)** is required for the nvFP4 CUTLASS example (Ch. 30). **8–16 H100s** are required to literally reproduce the book's distributed-training benchmarks (Part 10) — these two parts are designed to be studied theory-first and run on rented cloud GPUs (Lambda/RunPod/CoreWeave) only when you want to reproduce numbers.

## Environment / toolchain

- CUDA Toolkit 12.4+, `nvcc`, a C++17 compiler
- Python 3.10+, PyTorch 2.5+, `pybind11`, `uv`
- Nsight Systems + Nsight Compute (profiling, used from Part 9 onward but referenced throughout)
- `cuda-gdb`, `compute-sanitizer`
- CUTLASS (cloned automatically by the Part 11 build scripts)
- OpenMPI + NCCL for Part 10

---

## Part 0 — Foundations & Mental Models
*(Book Ch. 1 — "When PyTorch just isn't enough")*

1. **Why Custom CUDA, Why Now** — the optimization ladder from PyTorch → naive kernel → production kernel; when a custom kernel is (and isn't) worth writing; the LLM compute budget problem.
2. **The CUDA Programming Model** — host vs. device; kernels as thread-parallel functions; recognizing parallel opportunities; a first look at the memory hierarchy as *the* performance bottleneck; why deep learning ops are embarrassingly parallel.

## Part 1 — CUDA Fundamentals & Memory Management
*(Book Ch. 2 — "GPU Memory Management and Kernel Launch")*

3. **Your First Kernel** — toolchain setup, `nvcc`, `__global__` functions, `threadIdx`/`blockIdx`, launch syntax, 8-element vector add.
4. **Thread Hierarchy & Scalable Launches** — global thread indexing, boundary checks, grid-size calculation, scaling to 1M+ elements.
5. **Multi-Dimensional Indexing & Memory Model** — `dim3`, 2D/3D grids and blocks, 3D→1D flattening for tensors, `cudaMalloc`/`cudaMemcpy`, host vs. device memory.

## Part 2 — Building Deep Learning Primitives from Scratch
*(Book Ch. 3 — "Building the Core Operations from Scratch")*

6. **CPU-First Development Methodology** — why every kernel gets a CPU reference; elementwise ops (vector/matrix add) as the foundation of residual connections.
7. **Matrix Transpose & Naive GEMM** — correctness-over-performance GEMM; the operation at the heart of every linear layer.
8. **Softmax, Convolutions & Pooling** — naive softmax, 1D convolution (signal processing), 2D convolution (CNN core), max pooling / downsampling.

## Part 3 — Backpropagation & a Neural Net in Pure CUDA
*(Book Ch. 4)*

9. **Autodiff Theory** — computational graphs, forward vs. backward pass, the chain rule as a graph traversal, gradient rules for `+` and `×` nodes.
10. **MNIST MLP, Five Ways** — a 784→256→10 MLP rebuilt as: NumPy reference → PyTorch reference → single-threaded C → custom CUDA kernels → CUDA + cuBLAS. Direct host/device code comparison at every stage.

## Part 4 — Integrating CUDA into PyTorch: The Transformer
*(Book Ch. 5)*

11. **The PyTorch C++/CUDA Extension Pipeline** — `pybind11` bindings, `setup.py build_ext`, wrapping raw kernels as autograd-compatible ops, the "PyTorch baseline → custom CUDA" two-stage workflow.
12. **A Character-Level GPT in Custom CUDA** — embedding, layernorm, matmul, softmax, and activation kernels wired into a real training loop (dataset: *The Wonderful Wizard of Oz*).
13. **Transformer Inference: KV-Cache, GEMV & Top-K** — dense vs. MoE inference, KV-caching for autoregressive generation, GEMV kernels for decode-time speedups, top-K sampling; why MoE routing is a harder correctness problem than dense inference.

## Part 5 — The Optimization Ladder: Memory-Bound & Compute-Bound Kernels
*(Book Ch. 6)*

14. **Optimizing GEMM I — Memory Coalescing & Shared-Memory Tiling** — the canonical ladder: naive → global-memory coalescing → shared-memory blocking (mirrors Simon Boehm's kernel progression exactly).
15. **Optimizing GEMM II — Register Blocking & Vectorization** — 1D block-tiling, 2D block-tiling, vectorized loads (`float4`); measuring against cuBLAS at every rung.
16. **Optimizing Softmax & LayerNorm** — online (single-pass) softmax, shared-memory reduction, warp-shuffle reduction, vectorized loads.
17. **Optimizing GEMV & Top-K Selection** — warp-coalesced GEMV, vectorized GEMV vs. cuBLAS; naive vs. heap-based vs. warp-level Top-K.

## Part 6 — Tensor Cores: Hardware-Accelerated Matrix Math
*(Book Ch. 6/7 continuation — tensor core programming)*

18. **From CUDA Cores to Tensor Cores** — what tensor cores automate (coalescing, tiling) vs. what you still control; cuBLAS-with-tensor-cores as the zero-effort baseline; **WMMA** fragments and warp-level 16×16×16 MMA (Volta/Ampere+).
19. **WGMMA & Hopper's Asynchronous Tensor Cores** — warp-group MMA (128 threads / 4 warps), PTX-level programming, larger tiles, async loads via the Tensor Memory Accelerator (TMA), pushing tile sizes to the hardware limit. *(Requires Hopper.)*

## Part 7 — Flash Attention
*(Book Ch. 7 — "Flash attention implementations")*

20. **The Memory-Bandwidth Problem in Attention** — why materializing the N×N score matrix is the bottleneck; the online-softmax trick; tiling parameters (Br/Bc).
21. **Building Flash Attention: Naive → Fused → Tensor Cores** — 3-kernel naive baseline (QKᵀ → softmax → ×V) vs. a single fused kernel with online softmax and WMMA 16×16×16 tiles; benchmarking against PyTorch's native Flash Attention and understanding the remaining gap (BF16, `cp.async`, warp specialization, swizzled shared memory — previewing CUTLASS).

## Part 8 — Quantization for Inference
*(Book Ch. 8 — "Quantization")*

22. **Quantization Theory** — symmetric vs. asymmetric, dynamic vs. static scales, min-max vs. percentile calibration, memory-compression math (4× for INT8, 8× for INT4).
23. **Implementing Quant/Dequant Kernels** — FP32→INT8, FP32→INT4 with packing; granularity schemes hands-on: tensor-wise, group-wise, block-wise, channel-wise; a look at AWQ-style weight-only quantization for LLM inference.

## Part 9 — Profiling, Debugging & Performance Engineering
*(Book Ch. 9 — implied by the book's Nsight-focused promise; no dedicated repo folder)*

24. **Nsight Systems** — timeline profiling, finding kernel-launch gaps, stream overlap, CPU/GPU sync stalls.
25. **Nsight Compute & the Roofline Model** — occupancy, memory throughput vs. compute throughput, bank conflicts, warp divergence, register/shared-memory pressure, reading a kernel's "speed of light" report.
26. **Debugging CUDA** — `cuda-gdb` walkthroughs, `compute-sanitizer` for races/memory errors, common silent-failure patterns (uninitialized memory, race conditions in reductions).

## Part 10 — Multi-GPU & Distributed Training
*(Book Ch. 10 — "Distributed Computing")*

27. **Multi-GPU Fundamentals** — NCCL primitives, peer-to-peer access, NVLink vs. InfiniBand bandwidth, when to reach for tensor vs. pipeline parallelism.
28. **Tensor Parallelism at Scale** — sharding a large GEMM across 8 GPUs (single node, NVLink) then 16 GPUs across two nodes (InfiniBand); measuring scaling efficiency (~100% single-node, ~99.8% multi-node in the book's own benchmarks).
29. **Pipeline Parallelism** — naive blocking synchronization vs. async CUDA-stream overlap; the book's own numbers show a **7.8×** throughput jump (110 → 396 batches/s) from that overlap alone.

## Part 11 — CUTLASS & Production-Grade Kernels (Capstone)
*(Book Ch. 11 — "CUTLASS")*

30. **CUTLASS Capstone** — why hand-written kernels top out where template libraries win; CUTLASS 3.x + CuTe layouts; hand-written vs. official single/multi-GPU Hopper GEMM; Blackwell nvFP4 GEMM as the book's final, most advanced example. Wrap-up: revisiting the optimization ladder end-to-end (naive → coalesced → tiled → vectorized → tensor cores → CUTLASS) and what to reach for in real LLM inference/training stacks.

---

## Expert Resources Bibliography

Used to fill gaps, add rigor, and cross-check the book's approach:

- **"Programming Massively Parallel Processors"** (Kirk, Hwu, El Hajj) — the standard CUDA theory textbook; used to deepen Parts 1–2.
- **NVIDIA CUDA C++ Programming Guide** & **Best Practices Guide** — canonical reference for every API used throughout.
- **Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"** — the book's own GEMM kernel numbering (`0_cublas` → `6_vectorize`) is essentially this blog post; used directly for Part 5.
- **GPU MODE (formerly CUDA-MODE)** lecture series & Discord — community deep-dives on kernels, Triton, and profiling.
- **Lei Mao's blog** — clear write-ups on memory coalescing, bank conflicts, and quantization math.
- **Tri Dao et al., FlashAttention / FlashAttention-2 / FlashAttention-3 papers** — theoretical backbone for Part 7.
- **NVIDIA CUTLASS docs + CuTe layout algebra guide** — for Part 11.
- **NVIDIA Nsight Systems / Nsight Compute documentation** — for Part 9.
- **NCCL documentation** — for Part 10.
- **Horace He, "Making Deep Learning Go Brrrr From First Principles"** — framing for compute-bound vs. memory-bound intuition used throughout.

---

## Next steps

Say the word and we'll start with **Part 0, Chapter 1** — theory first, then a hands-on kernel, then exercises — and move through the list at whatever pace you want. We can also reorder (e.g., jump straight to Flash Attention or Tensor Cores) if you'd rather follow your current interests instead of strict book order.
