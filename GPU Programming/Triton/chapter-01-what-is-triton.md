# Chapter 1 — What Is Triton, and Why It Exists

## 1.1 The Problem

By the mid-2010s, deep learning workloads had a mismatch problem. Neural networks are built from a small set of computational patterns — matmuls, elementwise ops, reductions, normalizations — but getting *peak* hardware performance out of any one of these on a GPU requires expert-level CUDA: manual thread-block layout, shared-memory tiling, bank-conflict avoidance, warp-level synchronization, and tensor-core scheduling. That expertise is scarce, and hand-tuned kernels don't compose — fusing two operators (say, a matmul followed by a bias-add and ReLU) usually means writing a brand-new kernel from scratch.

Two prior approaches existed:

- **Vendor libraries** (cuBLAS, cuDNN) — extremely fast, but fixed: you get the operators NVIDIA decided to ship, in the shapes and fusions they decided to support.
- **Polyhedral / scheduling compilers** (Halide, TVM, Tiramisu, Tensor Comprehensions) — flexible and automatic, but historically left significant performance on the table versus hand-written kernels because their scheduling search spaces are decoupled from how the hardware actually executes tiled workloads.

Triton, introduced by Philippe Tillet, H.T. Kung, and David Cox (MAPL 2019), took a third position: give programmers a language expressive enough to write *arbitrary* fused kernels, but raise the abstraction level just enough that the compiler — not the programmer — handles the parts of CUDA programming that are mechanical and error-prone (memory coalescing, shared-memory allocation and synchronization, and, on modern GPUs, tensor-core/warp scheduling). The bet was that **blocked algorithms** — expressing computation over tiles rather than individual scalars or threads — are the right unit of abstraction to make this tractable. OpenAI's own framing of the 1.0 release captures the goal directly: <cite index="4-1">an open-source Python-like programming language which enables researchers with no CUDA experience to write highly efficient GPU code, most of the time on par with what an expert would be able to produce</cite>, with FP16 matmul kernels matching cuBLAS in <cite index="4-1">under 25 lines of code</cite>.

## 1.2 The Mental Model: Tiled Programs, Not Threads

This is the single most important reframing to internalize before writing any Triton code, so sit with it before Chapter 3 gets concrete.

In CUDA, you write code from the perspective of **one thread**. You reason about thread indices, warps of 32 threads executing in lockstep, and blocks of warps sharing memory. Parallelism is expressed by *you* deciding how work is divided across that hierarchy.

In Triton, you write code from the perspective of **one program instance operating on a tile (block) of data**. A "program" is conceptually similar to a CUDA thread block, but you never see individual threads — you write vectorized operations over blocks (e.g., "load these 128 contiguous elements," "compute their softmax," "store the result"), and the Triton compiler decides how to map that block-level computation onto actual threads, warps, shared memory, and (on Hopper/Blackwell) asynchronous tensor-core and TMA units.

Concretely, Triton <cite index="2-1">uses a Single-Program Multiple-Data (SPMD) model similar to CUDA's thread blocks, but expressed at a higher level</cite>. <cite index="2-1">Kernels are defined as Python functions decorated with @triton.jit, and they use the triton.language API for operations on GPU data. Each kernel launch spawns many parallel program instances, and within each instance, you can perform vectorized operations on small arrays called blocks — Triton handles mapping these to the actual GPU threads and warps</cite>.

The practical consequence: the things you'd spend hours debugging in raw CUDA — shared-memory bank conflicts, warp divergence from misaligned accesses, manual double-buffering for pipelining — are compiler responsibilities in Triton. You still need to *understand* them (Part IV of this curriculum goes back into that machinery), but you don't hand-write them for every kernel.

## 1.3 Triton vs. Raw CUDA

Given your CUDA background, the clearest way to place Triton is as a trade of *some* control for a large reduction in boilerplate and bug surface:

| Concern | CUDA C++ | Triton |
|---|---|---|
| Unit of reasoning | Single thread | Tile / block of data |
| Memory coalescing | Manual (you compute addresses per thread) | Compiler-inferred from block-level load/store |
| Shared memory | Explicit `__shared__` allocation, manual staging | Compiler-allocated and managed |
| Synchronization | Explicit `__syncthreads()`, barriers | Implicit within a program instance |
| Tensor cores (WMMA/WGMMA) | Manual `mma` intrinsics or CUTLASS | `tl.dot`, compiler picks the right instruction |
| Warp-level primitives | Manual shuffles, reductions | Built-in reduction ops (`tl.sum`, `tl.max`, ...) |
| Language | C++, separate host/device compilation | Python, JIT-compiled per shape/dtype |
| Peak-performance ceiling | Highest (nothing between you and the hardware) | Very high, but bounded by what the compiler's scheduler can currently do (see Ch. 25 on warp specialization limits) |

Triton doesn't make CUDA knowledge obsolete — it changes *where* that knowledge is applied. You'll use it to reason about why a kernel is slow (Part IV–VI) rather than to write every load and store by hand. And when the compiler's automatic scheduling genuinely can't hit the performance you need on the newest hardware, Triton now exposes a lower-level escape hatch (**Gluon**, Chapter 25) that hands warp specialization back to you — CUDA-level control, still in Triton's Python-embedded syntax.

## 1.4 Triton vs. Polyhedral / Scheduling Compilers

It's worth understanding why Triton exists *in addition to* Halide, TVM, and Tiramisu, since they attack the same problem. Triton's own documentation is explicit about this lineage and the gap it identified: prior DSL/compiler systems — whether polyhedral (Tiramisu, Tensor Comprehensions) or scheduling-language-based (Halide, TVM) — <cite index="8-1">remain less flexible and, for the same algorithm, markedly slower than the best handwritten compute kernels available in libraries like cuBLAS, cuDNN, or TensorRT</cite>. Triton's premise is that <cite index="8-1">programming paradigms based on blocked algorithms can facilitate the construction of high-performance compute kernels for neural networks</cite> — i.e., tiling isn't just an optimization technique to be *discovered* by an autoscheduler, it should be the *primitive* the language is built around, so the compiler's job is narrower and more tractable.

The practical difference you'll feel: Halide/TVM separate *what* to compute from *how* to schedule it (a schedule is a separate, often-fragile specification). Triton doesn't ask you to write a schedule — you write the tiled algorithm directly, and scheduling decisions below the tile level (instruction selection, pipelining depth, warp assignment) are the compiler's problem.

## 1.5 Where Triton Sits in Today's Landscape (2026)

The ecosystem around Triton has grown considerably since 2019. It's useful to have a map before going deeper:

- **cuBLAS / cuDNN** — NVIDIA's hand-tuned, closed-source libraries. Still the ceiling for standard GEMM shapes; Triton targets the *long tail* these libraries don't cover well — fused, non-standard, or rapidly-iterating kernels (custom attention variants, quantized ops, novel normalization layers).
- **CUTLASS** — NVIDIA's open-source C++ template library for writing near-cuBLAS-performance GEMMs by hand. Lower-level and more verbose than Triton; used when you need the absolute last few percent of performance or hardware features Triton hasn't exposed yet.
- **torch.compile / TorchInductor** — PyTorch's compiler *automatically generates Triton kernels* from your model's FX graph. This is arguably Triton's largest deployment surface today: most PyTorch users benefit from Triton without writing a line of it. Chapter 24 covers when Inductor's output is good enough and when hand-written kernels still win.
- **TileLang, ThunderKittens** — newer tile-based kernel DSLs exploring similar territory to Triton, sometimes trading Python-embedding for different abstractions (e.g., ThunderKittens' warp-level tile primitives in CUDA C++). Good to know they exist; not required for this curriculum.
- **Gluon** — Triton's *own* lower-level dialect (Chapter 25), for when you need to hand-schedule warp specialization on Hopper/Blackwell because the automatic compiler passes aren't yet finding the optimal strategy.

The throughline: Triton has become the default "write a custom fused kernel" tool in the PyTorch ecosystem — either directly, or invisibly via `torch.compile`.

## 1.6 When to Reach for Triton vs. Hand-Written CUDA

A practical decision framework, which we'll revisit with more nuance once you've profiled real kernels in Part VI:

**Reach for Triton when:**
- You're fusing multiple ops (elementwise + reduction + elementwise) to cut memory round-trips — Triton's fusion story is its strongest use case.
- The operator is novel or research-stage (custom attention masks, new normalization/quantization schemes) and no vendor library covers it.
- You want performance portability across NVIDIA/AMD/Intel without maintaining separate codebases (Chapter 26).
- Iteration speed matters — a Triton kernel is a Python function; a CUDA kernel is a separate compilation unit with host/device glue.

**Reach for hand-written CUDA (or CUTLASS) when:**
- You need a hardware feature Triton/Gluon hasn't exposed yet, or a scheduling strategy the compiler provably can't find (this does happen on bleeding-edge hardware — see the Blackwell warp-specialization discussion in Chapter 25).
- You're implementing something *not* tile-shaped (irregular control flow, pointer-chasing data structures, graph algorithms) — Triton's abstraction is built for dense tiled math, not arbitrary parallel programs.
- You already have a highly-tuned CUDA/CUTLASS kernel hitting >95% of roofline and the maintenance cost of a rewrite isn't justified.

In practice, most teams today write Triton first, profile (Part VI), and drop to Gluon or CUDA only for the specific hot kernel that needs it — which is exactly the FlashAttention/Liger Kernels pattern we'll study in Chapter 27.

## 1.7 Check Your Understanding

1. In your own words: what does it mean that Triton programs are written "from the perspective of a tile" rather than a thread? What CUDA-level responsibilities does this shift onto the compiler?
2. Why does Triton's documentation position it as an alternative to *both* cuBLAS-style libraries *and* Halide/TVM-style scheduling compilers, rather than a replacement for just one?
3. Give one example of a kernel where Triton's fusion advantage would matter a lot, and one where it wouldn't (i.e., where a vendor library is already optimal).
4. Where does `torch.compile` fit relative to hand-written Triton kernels — are they in competition, or complementary? (You'll answer this more precisely after Chapter 24, but take a first pass now.)

## 1.8 What's Next

Chapter 2 gets you a working environment (NVIDIA/AMD/Intel install paths) and walks through the JIT compilation lifecycle of a Triton kernel end-to-end — from Python function to cached, executable GPU code — using the official vector-add example. From Chapter 3 onward we start writing kernels.
