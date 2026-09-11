Here's a comprehensive, structured learning plan for GPU programming with CUDA, designed for a Python developer going from zero to expert.

Here's the complete curriculum at a glance:

| # | Chapter | Phase | Key topics |
|---|---|---|---|
| 1 | [GPU Architecture & the CUDA Programming Model](./cuda_chapter1.md) | Foundation | SMs, warps, SIMT, warp divergence, memory hierarchy mental map |
| 2 | [Environment Setup & First Kernel](./cuda_chapter2.md) | Foundation | Host/device model, vector addition (Numba/CuPy/RawKernel), CUDA events for timing |
| 3 | [Thread Indexing & Memory Spaces](./cuda_chapter3.md) | Foundation | 1D/2D/3D indexing, all memory spaces with latencies, shared memory, syncthreads, bank conflicts |
| 4 | [CUDA from Python: Numba & CuPy](./cuda_chapter4.md) | Foundation | JIT internals, device functions, memory pool, ElementwiseKernel, Numba↔CuPy zero-copy interop |
| 5 | [Shared Memory & Synchronization](./cuda_chapter5.md) | Core CUDA | Tiled matrix multiply, two syncthreads calls, bank conflict padding, naive vs tiled vs cuBLAS |
| 6 | [Memory Access Patterns & Coalescing](./cuda_chapter6.md) | Core CUDA | Cache line transactions, coalescing rules, matrix transpose, `__ldg()`, x=fast-index rule |
| 7 | [Parallel Reductions & Scan](./cuda_chapter7.md) | Core CUDA | Tree reduction, warp shuffle `__shfl_down_sync`, multi-block reduction, Blelloch prefix scan, stream compaction |
| 8 | [Streams, Concurrency & Async Execution](./cuda_chapter8.md) | Core CUDA | Streams, pinned memory, double buffering pipeline, CUDA events, null stream trap |
| 9 | [Profiling with Nsight Systems & Compute](./cuda_chapter9.md) | Performance | Roofline model, arithmetic intensity, top-10 metrics, structured optimisation workflow |
| 10 | [Occupancy, Warp Efficiency & Launch Config](./cuda_chapter10.md) | Performance | Three occupancy constraints, register pressure, `__launch_bounds__`, warp divergence quantified, thread coarsening |
| 11 | [Atomic Operations & Lock-Free Algorithms](./cuda_chapter11.md) | Performance | All atomics, contention penalty, privatisation, `__ballot_sync`, warp compaction, `__threadfence()` |
| 12 | [cuBLAS, cuDNN & CUDA Libraries](./cuda_chapter12.md) | Performance | BLAS levels, GEMM α/β, cuDNN algorithm search, Winograd, cuFFT, cuSPARSE, fusion decision framework |
| 13 | [Custom CUDA Extensions for PyTorch](./cuda_chapter13.md) | Advanced ML | `.cu`+pybind11+`setup.py`, autograd Function, `AT_DISPATCH`, `gradcheck`, fused LayerNorm |
| 14 | [Tensor Cores & Mixed Precision](./cuda_chapter14.md) | Advanced ML | WMMA API, FP16/BF16/TF32/INT8 formats, AMP training loop, GradScaler, Flash Attention tile alignment |
| 15 | [Unified Memory, Pinned Memory & Multi-GPU](./cuda_chapter15.md) | Advanced ML | Pinned vs pageable vs UM, NCCL collectives, DDP bucketing, FSDP sharding, scaling efficiency |
| 16 | [Kernel Fusion & Flash Attention Internals](./cuda_chapter16.md) | Advanced ML | Online softmax trick, O(seq_len) memory, tiled fused attention kernel |
| 17 | [OpenAI Triton — Python GPU Kernels](./cuda_chapter17_triton.md) | Expert | Tile-based programming, `@triton.jit`, tl.load/store, auto-tuning, GEMM and softmax in Triton |
| 18 | [PTX, SASS & Compiler Internals](./cuda_chapter18_ptx_sass_compiler.md) | Expert | NVCC pipeline, reading PTX, inline PTX with `asm()`, `cuobjdump`, register allocation |
| 19 | [CUDA Graphs & Dynamic Parallelism](./cuda_chapter19_cuda_graphs_dynamic_parallelism.md) | Expert | Graph capture & replay, dynamic parallelism, cooperative groups, low-latency inference |
| 20 | [Capstone — Mini GPU-Accelerated ML Library](./cuda_chapter20_capstone.md) | Expert | Custom GEMM, fused attention, multi-stream pipeline, Triton softmax, PyTorch extension wrapper |


**Why this order matters**

Most CUDA tutorials throw you into C syntax immediately. This plan starts with the hardware mental model (Ch 1) before writing any code, because without understanding warps, SMs, and the memory hierarchy, you'll write technically correct kernels that are 10× slower than they should be — and have no idea why.

**How we'll work through each chapter**

Each session will follow a consistent pattern: concept explanation → visual intuition → working code → a hands-on exercise you run yourself → a mini-benchmark so you can *feel* the difference your changes make.

**Your Python background is a genuine advantage**

Numba and CuPy let you write real CUDA kernels without leaving Python. We'll use these heavily in the early chapters (2–8) so you can focus on GPU thinking rather than C syntax. C/CUDA C syntax gets introduced gradually in phases 2–3 when it becomes necessary for low-level control.

**Key milestones**

- After Ch 4 — you can write and run GPU kernels from Python
- After Ch 8 — you can write correct parallel code with proper synchronization
- After Ch 12 — you can use CUDA libraries to accelerate real workloads
- After Ch 16 — you can write kernels at the level of production ML infrastructure
- After Ch 20 — you have a portfolio project showcasing all of it

Click any chapter in the roadmap above to jump straight into it, or just say "let's start Chapter 1" and we'll go in order.