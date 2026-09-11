# Triton GPU Programming — Beginner to Advanced Curriculum

Sources: official Triton docs/tutorials, the Triton paper (Tillet, Kung & Cox, MAPL 2019), GPU MODE's "Practitioner's Guide to Triton" and Triton-Puzzles (Sasha Rush et al.), PyTorch's warp-specialization/Gluon blog series, and current production practice (FlashAttention, Liger Kernels, TorchInductor).

## Part I — Orientation
1. **What Is Triton, and Why It Exists**
   Tiled-program compilation model vs. raw CUDA vs. polyhedral/scheduling DSLs (Halide, TVM, Tiramisu). Where Triton sits relative to cuBLAS/cuDNN, CUTLASS, torch.compile, TileLang, and ThunderKittens. When to reach for Triton vs. hand-written CUDA.
2. **Environment, Toolchain & the JIT Lifecycle**
   Install (NVIDIA/AMD/Intel backends), `@triton.jit` compilation flow at a glance, kernel caching, versioning notes (Triton 3.x). *Hands-on:* run and inspect the official vector-add example end to end.

## Part II — The Programming Model
3. **SPMD Kernel Anatomy**
   `program_id`, grids, launch semantics, `constexpr`, how a Triton "program" relates to a CUDA thread block/warp group.
4. **Pointers, Loads, Stores & Masking**
   Address arithmetic, boundary handling, `tl.load`/`tl.store` with masks.
   *Hands-on:* Vector Addition tutorial + Triton-Puzzles 1–3.
5. **Tensors, Shapes, Strides & Block Pointers**
   Multi-dimensional indexing, `tl.arange`, strided access, `make_block_ptr` vs. manual pointer math.
6. **Compile-Time Specialization & Control Flow**
   `constexpr` branching, loops, `tl.static_assert`, why specialization trades compile time for runtime speed.

## Part III — Core Kernels (Tutorial-Driven)
7. **Reductions & Fused Softmax**
   Row-wise reductions, why fusion beats bandwidth-bound PyTorch ops, SRAM-residency reasoning.
   *Hands-on:* Fused Softmax tutorial; benchmark vs. `torch.softmax`.
8. **Autotuning & Heuristics**
   `@triton.autotune`, `Config` spaces, tuning keys, `@triton.heuristics`, cache persistence (`triton-deja-vu` pattern).
9. **Matrix Multiplication I — Tiled GEMM Fundamentals**
   Blocked matmul, L2-cache-aware tile ordering ("group" scheduling), grid design.
   *Hands-on:* Matrix Multiplication tutorial, unautotuned → autotuned.
10. **Matrix Multiplication II — Precision & Tensor Cores**
    `tl.dot`, fp16/bf16 accumulation, numerical accuracy tradeoffs, intro to tensor-core mapping.
11. **Randomness & Low-Memory Dropout**
    Stateless Philox-style RNG in-kernel, seeding strategy.
    *Hands-on:* Low-Memory Dropout tutorial + its extension challenges (strided matrix version, sparse JL transform).
12. **Layer Normalization — Fused Forward & Backward**
    Welford's algorithm, backward-pass reduction patterns, atomics for gradient accumulation.
13. **libdevice & External Functions**
    Calling CUDA/HIP math libraries from Triton, extending the language safely.

## Part IV — Mapping Triton to the GPU (Compiler & Architecture)
14. **Compiler Pipeline Deep Dive**
    Triton IR → TTGIR (Triton GPU IR, MLIR-based) → LLVM IR → PTX/SASS or AMDGCN. Reading `MLIR_ENABLE_DUMP` output.
15. **Memory Hierarchy, Revisited for Triton**
    How `tl.load`/`tl.store` become coalesced global accesses; compiler-managed shared memory; register allocation and spills.
16. **Layouts & Data Movement**
    Blocked/shared/MMA layouts, why layout choice — not just tile size — determines performance.

## Part V — Attention & Advanced Fused Kernels
17. **Flash Attention I — Forward Pass**
    Online softmax, Q/K/V tiling, causal masking.
    *Hands-on:* Fused Attention tutorial (FlashAttention-2 style).
18. **Flash Attention II — Backward Pass & Numerical Stability**
    Recomputation strategy, gradient kernels, precision pitfalls in the backward reduction.
19. **Group GEMM & Persistent Kernels**
    Variable-size batched matmul, the persistent-kernel pattern, grid-size reduction for launch overhead.
20. **Low-Precision & Block-Scaled Matmul**
    FP8, MXFP4/NVFP4 quantized GEMM, per-block scaling factors — current frontier for LLM inference kernels.

## Part VI — Debugging, Testing & Performance Engineering
21. **Correctness First: Interpreter Mode & Testing**
    `TRITON_INTERPRET=1`, `triton-viz` visualization, systematic numerical validation against PyTorch references, edge-case (irregular shape) testing.
22. **Profiling & Performance Tuning**
    Proton profiler (lightweight, cross-vendor) vs. Nsight Compute/Systems; reading occupancy, register-spill, and memory-workload metrics; a repeatable benchmarking methodology.
23. **Autograd & PyTorch Integration**
    Wrapping kernels in `torch.autograd.Function`, `torch.library` custom-op registration, wiring forward/backward Triton kernels into a trainable `nn.Module`.

## Part VII — Frontier Topics
24. **torch.compile & TorchInductor Internals**
    How PyTorch auto-generates Triton kernels from FX graphs, `max-autotune`, and deciding when a hand-written kernel is still worth it.
25. **Warp Specialization, TMA & Gluon**
    Hopper/Blackwell Tensor Memory Accelerator, tensor memory (tmem), automatic warp specialization in the Triton compiler, and dropping to the low-level **Gluon** dialect for hand-scheduled warp-specialized kernels.
26. **Cross-Platform Portability**
    AMD ROCm and Intel XPU backends, writing kernels that stay fast across vendors, portability limits of autotuning.

## Part VIII — Production & Capstone
27. **Kernel Libraries & Production Case Studies**
    How FlashAttention, Liger Kernels, and Unsloth-style fine-tuning kernels are structured in the wild; TritonBench as an evaluation harness; packaging kernels for reuse.
28. **Capstone — A Small Fused-Kernel Library**
    Design, implement, autotune, test, and profile a cohesive set of kernels (e.g., fused RMSNorm + RoPE + attention) end-to-end, benchmarked against PyTorch/cuBLAS baselines, with CI-style correctness tests.

---
### Appendices
- **A. Triton Puzzles Reference** — mapping puzzles 1–21 (gpu-mode/Triton-Puzzles) to the chapters above for extra reps.
- **B. Triton vs. CUDA Cheat Sheet** — concept-to-concept mapping.
- **C. Glossary & Further Reading** — Triton paper, PyTorch warp-specialization blog series, GPU MODE Lecture 14 notebook, `triton-resources` curated list.
