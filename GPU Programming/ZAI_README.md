# CUDA Learning Plan: Beginner to Expert (for Python Developers)

Welcome to GPU programming! I've put together a comprehensive 20-chapter plan.

## Detailed Chapter-by-Chapter Plan

### **Phase 1: Foundations (Chapters 1–3)**

**Chapter 1: Introduction to GPU Computing & Environment Setup**
- CPU vs GPU architecture (cores, latency vs throughput, SIMD/SIMT)
- The CUDA ecosystem: runtime, driver, toolkit, libraries
- Hardware overview: SMs, warps, lanes; memory hierarchy at a glance
- Setup: CUDA toolkit, driver, `nvidia-smi`, Nsight tools, CuPy, Numba, PyTorch with CUDA
- Hello World via CuPy and Numba CUDA
- *Hands-on*: Run `nvidia-smi`, query device properties, vector add in CuPy

**Chapter 2: CUDA Programming Model — Threads, Blocks, Grids**
- Host vs device; kernel launch syntax `<<<grid, block>>>`
- Built-in variables: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- Mapping data to threads (1D/2D/3D indexing)
- Writing kernels in Numba CUDA (`@cuda.jit`)
- Comparison with CuPy element-wise kernels and raw CUDA C
- *Hands-on*: vector add, saxpy, matrix element-wise ops, image brightness adjustment

**Chapter 3: Memory Hierarchy Fundamentals**
- Global, shared, local, constant, texture memory, registers
- L1/L2 cache; unified memory intro
- Memory transfer cost (`cudaMemcpy` patterns)
- Device query: shared memory per SM, registers, warp size
- *Hands-on*: query device limits; observe transfer overhead with different array sizes

---

### **Phase 2: Core CUDA (Chapters 4–8)**

**Chapter 4: Memory Coalescing & Access Patterns**
- What coalesced access means; 128-byte transactions
- Strided, scattered, AoS vs SoA
- Profiling memory throughput
- *Hands-on*: rewrite SoA kernels; benchmark coalesced vs strided

**Chapter 5: Shared Memory & Tiling**
- `__shared__` memory; bank conflicts; padding
- `__syncthreads()` and synchronization pitfalls
- Tiled matrix multiplication (classic case study)
- 1D/2D stencil with halo cells
- *Hands-on*: tiled matmul vs naïve matmul; benchmark speedups

**Chapter 6: Warp-Level Programming & Divergence**
- Warp divergence; cost of branches
- Warp-level primitives: `__shfl_sync`, `__ballot_sync`, `__any_sync`, `__all_sync`
- Lane masks, broadcasts, reductions within a warp
- *Hands-on*: warp-level reduction; prefix sum in a warp

**Chapter 7: Atomics, Reductions & Scan**
- Atomic operations and their costs
- Race conditions and how to debug
- Parallel reduction: pair-wise, tree, two-stage
- Work-efficient scan (Blelloch); Hillis-Steele
- *Hands-on*: sum-reduction kernel; exclusive scan; benchmark against `cupy.cumsum`

**Chapter 8: Occupancy & Launch Configuration**
- Registers, shared memory, and thread block limits
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- Choosing block sizes; auto-tuning
- *Hands-on*: occupancy calculator; sweep block sizes for reduction and matmul

---

### **Phase 3: Performance & Tooling (Chapters 9–11)**

**Chapter 9: Profiling & Debugging**
- Nsight Systems (timeline, bottlenecks)
- Nsight Compute (kernel-level metrics, roofline)
- Compute Sanitizer (race, memory, sync checks)
- Warp/occupancy/launch statistics
- *Hands-on*: profile your tiled matmul; identify memory vs compute bound

**Chapter 10: Streams, Events & Concurrency**
- Streams and asynchronous execution
- `cudaMemcpyAsync`, pinned memory
- Events for timing and dependencies
- Multi-stream overlap of compute and copy
- CUDA Graphs for static workflows
- *Hands-on*: pipelined data transfer + compute; build a CUDA graph

**Chapter 11: Advanced Memory Techniques**
- Unified Memory and page faults
- Memory pools, async allocators
- Zero-copy memory
- Double-buffering and software pipelining
- *Hands-on*: UM-based matrix ops; double-buffered tiled kernel

---

### **Phase 4: Libraries, Tensor Cores & Modern Hardware (Chapters 12–14)**

**Chapter 12: CUDA Libraries Ecosystem**
- cuBLAS, cuDNN, cuFFT, cuRAND, cuSPARSE
- Thrust (algorithms), CUB (low-level primitives)
- CUTLASS intro
- When to use libraries vs custom kernels
- *Hands-on*: cuBLAS gemm benchmark; Thrust sort vs custom

**Chapter 13: Tensor Cores & Mixed Precision**
- Tensor Core basics; FP16/BF16/TF32/FP8
- WMMA (warp matrix multiply accumulate) API
- mma.sync PTX intrinsics
- Hopper WGMMA overview
- *Hands-on*: WMMA-based GEMM in FP16; compare with cuBLAS

**Chapter 14: Dynamic Parallelism, Cooperative Groups & Clusters**
- Dynamic parallelism (kernel-launching kernels)
- Cooperative groups API
- Grid-wide synchronization
- Thread block clusters (Hopper) and distributed shared memory
- *Hands-on*: dynamic-parallelism histogram; cooperative-groups reduction

---

### **Phase 5: Production CUDA (Chapters 15–17)**

**Chapter 15: Multi-GPU Programming**
- Multi-GPU memory models
- NCCL for collectives (all-reduce, all-gather)
- Peer-to-peer and NVLink
- MPS (Multi-Process Service)
- *Hands-on*: multi-GPU all-reduce; benchmark NCCL vs naïve

**Chapter 16: Writing PyTorch CUDA Extensions**
- PyTorch C++ extension scaffold
- Custom autograd `Function` with `forward`/`backward`
- Dispatch macros (`AT_DISPATCH_FLOATING_TYPES`)
- Memory sharing between PyTorch tensors and raw CUDA
- Building with `setuptools`, JIT loading
- *Hands-on*: custom fused activation + bias kernel with autograd

**Chapter 17: Triton — High-Level GPU Programming**
- Triton language and block-level model
- Comparison with raw CUDA
- Autotuning
- Implementing fused attention in Triton
- When to use Triton vs CUDA vs libraries
- *Hands-on*: Triton matmul, softmax, and a simplified FlashAttention

---

### **Phase 6: Expert-Level Topics (Chapters 18–20)**

**Chapter 18: Real-World Kernel Case Studies**
- FlashAttention v2/v3 deep dive
- Fused LayerNorm, RMSNorm, Softmax, GeLU
- Convolution via im2col vs winograd vs direct
- Sort, top-k, gather/scatter
- *Hands-on*: implement a fused RMSNorm + residual kernel for a transformer block

**Chapter 19: Performance Engineering Methodology**
- Roofline model and bottleneck analysis
- Memory-bound vs compute-bound identification
- Latency hiding via ILP
- Instruction scheduling, software pipelining
- SASS inspection with `cuobjdump`
- *Hands-on*: take a slow kernel to within 80% of cuBLAS

**Chapter 20: Modern Architectures & the Frontier**
- Hopper features: TMA (Tensor Memory Accelerator), DSMEM, clusters
- Blackwell features and FP4/FP8 paths
- CUTLASS deep dive ( Cute_Tensor layouts, kernels)
- Asynchronous pipelines (`cp.async`, `mbarrier`)
- Future: Triton 3.0, MLIR, future of GPU programming models
- *Hands-on*: TMA-based copy kernel; CUTLASS GEMM customization

---

## Suggested Prerequisites
- Solid Python (✓)
- Basic linear algebra and algorithms
- Willingness to learn some C/C++ syntax (we'll introduce as needed)
- Familiarity with NumPy

## Tools We'll Install
- CUDA Toolkit (12.x) + driver
- `nvidia-smi`, Nsight Systems, Nsight Compute, Compute Sanitizer
- Python: `cupy`, `numba`, `torch`, `triton`, `nvtx`
- C++ side: `nvcc`, CMake, optionally `cutlass`

## How We'll Proceed
Each chapter will include:
1. **Concept explanation** with diagrams/analogies
2. **Minimal code example** (Python-first, then C++ where relevant)
3. **Hands-on exercise** for you to code
4. **Common pitfalls & profiling tips**
5. **Checkpoint**: small quiz or optimization challenge