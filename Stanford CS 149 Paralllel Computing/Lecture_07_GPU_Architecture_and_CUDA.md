# CS149 — Lecture 7: GPU Architecture and CUDA Programming

**Course:** Stanford CS149, Parallel Computing (Fall 2023)
**Instructors:** Prof. Kayvon Fatahalian & Prof. Kunle Olukotun
**Video:** https://www.youtube.com/watch?v=qQTDF0CBoxE
**Course site:** https://gfxcourses.stanford.edu/cs149/fall23/lecture/gpucuda/

*Notes combine the official slide deck with the lecture transcript (live Q&A, worked examples).*

---

## 1. Framing for This Lecture

The central promise up front: **there is no new conceptual material today.** Multi-core execution, SIMD, and multi-threading — the same three ideas covered in the first two weeks of this course — are simply deployed at a larger scale on a GPU. The goal of this lecture is to show how those familiar ideas map onto GPU hardware and onto **CUDA**, NVIDIA's C++-based programming language for GPUs, which turns out to closely resemble ISPC in spirit (unsurprising, since ISPC's own designers were directly inspired by CUDA's programming model, wanting CPU programmers to have access to a similarly clean, low-level SPMD abstraction).

---

## 2. History: From "Draw Triangles" to General-Purpose Computing

### What GPUs were originally built to do
Borrowing a quick summary from computer graphics: given a mathematical description of a 3D scene (triangle-mesh geometry, materials, lights, and a virtual camera position), compute the image a real camera at that position would have captured. Concretely, a GPU's original graphics pipeline: take a list of 3D triangle vertices, project them onto a 2D screen based on camera position/orientation, and then — for every resulting pixel — run a small program (a "shader") that computes that pixel's final color based on its surface material, its orientation, and texture data. Materials vary enormously across real-world surfaces (skin, metal, fabric, glass), which is precisely why, starting in the early 2000s, GPUs moved from fixed, built-in coloring formulas toward letting programmers write small custom programs that get run **once per pixel, entirely independently** — a textbook data-parallel workload: given per-pixel inputs, run one function, get one color out, for every pixel in the image.

Rendering millions of pixels per frame, dozens of times per second, meant GPU designers kept adding more cores and more SIMD width to keep up with growing image resolutions and frame rates — precisely the multi-core-plus-SIMD trajectory already discussed in this course's second lecture. Notably, this happened during the same window (roughly two decades before this lecture) when CPU single-thread performance growth was stalling out (the same "green line keeps growing, but ILP and clock frequency both ran out of room" story from Lecture 1) — except GPUs had already committed fully to a parallel, multi-core design, and simply kept scaling core count as transistor budgets grew.

### The GPGPU hack era
Some researchers noticed that GPUs, built purely to compute pixel colors, were becoming extremely fast, extremely parallel processors that happened to be able to run arbitrary small programs — so people started **repurposing** them for non-graphics computation via a genuine hack: draw two triangles that exactly cover the screen (forcing the GPU to invoke its per-pixel shader program once for every pixel on that screen, e.g., 512×512 times), and then write a "color" shader program that actually does something else entirely — like advancing one timestep of a physics or fluid simulation — treating the shader's RGBA color output as some other kind of numeric result entirely (e.g., an XYZ position). This let people exploit GPU parallelism for general computation years before GPUs officially supported it, at the cost of contorting the program into the shape of a graphics workload.

### A proper abstraction: the Brook / streaming programming model
A Stanford research project (around 2004) recognized this hack for what it was and built a real **data-parallel / streaming** programming language (Brook) on top of it: a programmer could write ordinary-looking code applying a function across an entire collection ("scale a collection of values by some amount"), and the compiler would translate that, source-to-source, into the same underlying triangle-drawing hack — hiding the ugliness from the *user*, even though the *implementation* was still, fundamentally, the same hack under the hood.

### NVIDIA's response: CUDA and "compute mode"
Seeing this trend, and already producing chips capable of running general code, NVIDIA introduced (2007) a proper hardware/software interface for general-purpose computation, alongside — not replacing — the traditional graphics pipeline: **compute mode**. Contrast the two very different ways of "starting a computation" on a chip:
- **How an operating system runs code on a CPU**: initialize a thread's register state, point its program counter at the start of some compiled binary, and tell one specific core to go — repeated once per thread, one at a time.
- **How the old graphics interface ran code on a GPU**: hand the GPU a list of triangles, a camera, and a shader program; the fixed graphics pipeline figures out which pixels need shading and invokes the shader for each one — there's no concept of directly "starting a thread" at all.

NVIDIA's compute-mode interface introduced a third option, deliberately in between: the programmer writes an ordinary-looking function (a **kernel**) and simply tells the hardware "run N logical copies of this" — the GPU itself figures out how to actually parallelize and schedule that. This is precisely the SPMD programming model already familiar from ISPC — a single program, launched many times, each instance aware of its own ID — just now expressed as the *primary* interface to the hardware, rather than something hidden inside a compiler.

---

## 3. The CUDA Programming Model

### Terminology mapping from ISPC
What ISPC calls a **program instance** (logically a thread, though ultimately implemented as a SIMD vector lane), CUDA calls a **CUDA thread** — a term defined entirely by however NVIDIA's hardware happens to implement it, not necessarily a literal, independent hardware execution context. Whenever "thread" is used loosely in this lecture, it means a CUDA thread in this SPMD sense, unless explicitly qualified as a "hardware execution context."

### Launching a kernel: multi-dimensional thread IDs and thread blocks
Just as an ISPC `task` launch says "run this function N times, however you want to schedule it," a CUDA kernel launch says the same thing — but with two additional wrinkles:
1. The count of instances to launch can be expressed as a **multi-dimensional** value (e.g., a 2D or 3D grid), which is often convenient for image, tensor, and graphics workloads — avoiding a lot of otherwise-necessary integer division to recover a multi-dimensional index from a flat 1D thread ID.
2. Instances are explicitly grouped into **thread blocks**. A launch like "create thread blocks of size 4×3 threads, with enough blocks to cover a 12×6 matrix" creates 12 threads per block × 6 total blocks = 72 CUDA threads overall — one per matrix element, in this example — even though nothing *requires* one thread per element; that's just how this particular kernel happens to be structured.

Within a kernel, built-in variables `threadIdx` (a thread's ID *within* its block) and `blockIdx` (which block it's in), combined with `blockDim` (the size of a block), let each thread compute its own unique global position — directly analogous to ISPC's `programIndex` (≈ `threadIdx`) and, one level up, `programCount`/task-index (≈ `blockDim`/`blockIdx`). CUDA simply adds one extra layer of hierarchy — grouping instances into blocks — that ISPC's task/gang split only loosely mirrors.

### The CUDA memory model: separate address spaces
Host (CPU) code and device (GPU) code operate in **distinct address spaces**. Ordinary heap allocation (`malloc`/`new`) creates CPU-side memory; `cudaMalloc` allocates GPU-side ("device") memory; `cudaMemcpy` explicitly copies data between the two. Dereferencing a device pointer directly from host code (or vice versa) is simply invalid and will crash — the two are not interchangeable, conceptually identical to two separate nodes in a message-passing system from the previous lecture: `cudaMemcpy` genuinely *is*, in effect, a message send between two separate address spaces, and — as raised in Q&A — can indeed be done asynchronously to hide its latency, exactly like an async message send. (Modern CUDA does allow more transparent pointer usage in some configurations, but under the hood, a physically discrete GPU card still has to move that data over a PCIe bus into its own separate DRAM — the separation is real, even when partially hidden by convenience APIs.)

### A must-understand detail: CUDA threads don't automatically map one-to-one onto "your" data
Consider computing a convolution or matrix operation over an array whose size isn't a clean multiple of the chosen block size (e.g., an 11×5 matrix with a 4×3 thread block) — the number of blocks launched has to be rounded *up*, meaning some threads end up with a computed index that falls **outside the actual array bounds**. CUDA does not automatically prevent or route around this — it is the programmer's job to include an explicit bounds check (`if` statement) inside the kernel so out-of-range threads simply do nothing. Omitting this check risks out-of-bounds reads/writes and a crash. This is a direct, concrete reminder that **CUDA's programming model is "launch N copies of this program, and the program itself decides what work it does based on its ID"** — not an implicit "one thread automatically maps to one array element" guarantee.

### Nested address spaces: per-thread, per-block, and global
CUDA actually exposes a small hierarchy of storage scopes: ordinary local variables are private to a single thread (like a thread's own local stack — no other thread can see them); variables explicitly declared **`shared`** are private to a *thread block* — visible to, and shared by, every thread within that one block only (conceptually similar to a `uniform` variable in ISPC, but writable and block-scoped rather than gang-scoped); and ordinary heap allocations (via `cudaMalloc`) live in **global device memory**, accessible by *any* thread in the entire kernel launch via ordinary loads and stores. This hierarchy is a strong hint about how the hardware is expected to physically group and co-locate threads for locality.

---

## 4. Worked Example: 1D Convolution

**The task**: given an input array, produce an output array where each output element is the (say, averaged) combination of itself and its two immediate neighbors — the 1D analogue of a convolutional layer in a neural network, or a basic signal/image-processing filter.

### A naive version
Launch enough thread blocks to cover the output array; each thread computes its own global index, loads the three needed input values directly from global memory, computes the result, and writes it out. This is correct — but neighboring threads redundantly re-load largely the *same* input data straight from (comparatively slow) global memory, over and over, with no explicit reuse.

### An optimized version, using shared memory
Instead, have each thread in a block of (say) 128 threads cooperatively load **one** input value into a `shared`-memory array sized to exactly what the whole block needs (130 values, for 128 outputs needing one extra element of "halo" data on each side) — with the two edge threads in the block doing a small amount of extra work to also grab those two additional boundary elements. Only *after* every thread in the block has finished this cooperative loading phase does each thread go on to actually compute its own output — now reading exclusively from the much faster, on-chip `shared` memory rather than from global memory at all.

**Why does this need an explicit barrier?** Between the cooperative-load phase and the compute phase, the kernel calls `__syncthreads()` — a barrier scoped to just the current thread block. Without it, nothing in the SPMD execution model guarantees any particular ordering between threads: a "fast" thread could reach the compute phase and start reading shared memory *before* every other thread in the block has actually finished writing its own contribution into it, silently reading stale or uninitialized data. The barrier guarantees that every thread's write has landed before any thread is allowed to proceed to reads that depend on the *whole* shared buffer being complete.

**Why bother with all this?** `shared` memory is backed by fast, genuinely on-chip storage — comparable in spirit to a high-performance, explicitly-managed L1 cache. This example is really a hand-orchestrated instance of the exact idea from the previous lecture (cache blocking / reducing artifactual communication): no single thread ever reuses the same piece of data twice, but *neighboring* threads within a block frequently need overlapping input data — so cooperatively loading it into fast shared storage once, and having everyone read from there, meaningfully cuts down on redundant trips to slower global memory.

---

## 5. Warps: How SPMD Code Actually Runs as SIMD on a GPU

### Implicit SIMD, revisited
Recall from Lecture 3 the distinction between **explicit SIMD** (a CPU compiler statically emits genuine vector instructions ahead of time) and **implicit SIMD** (the hardware itself notices, at runtime, that several logically-separate threads happen to be executing the exact same instruction, and runs them together on shared SIMD execution units). GPUs use the implicit approach: **every single CUDA thread has its own program counter** — nothing in the hardware assumes threads move in lockstep by default. But when a hardware unit notices that a contiguous group of 32 threads all currently share the *same* program counter, it dispatches that shared instruction to run across all 32 of them at once on a shared SIMD unit — behaving, in that moment, exactly like a single 32-wide vector instruction, purely because the hardware happened to discover 32-way coherence dynamically rather than because a compiler asserted it in advance.

This hardware-managed group of (historically) 32 threads is called a **warp** — a hardware-level implementation detail that never appears directly in the CUDA programming model itself (a program only ever creates thread blocks and threads; a warp is simply how the hardware happens to group execution contexts internally for SIMD purposes). Threads within a warp execute together, in SIMD fashion, whenever — and only for as long as — they remain at the same program counter; if they diverge (e.g., take different branches of an `if`), the hardware handles it via the same kind of masking mechanism covered back in Lecture 2's SIMD-divergence discussion, just now happening dynamically in hardware rather than being something the compiler statically arranged.

### Why hide warps behind the abstraction at all?
This design is deliberate, and mirrors ISPC's own `foreach` philosophy: by *not* exposing warp width as a fixed, binding part of the programming model (the way, say, an AVX instruction's fixed vector width is baked directly into a compiled CPU binary), NVIDIA keeps the freedom to change warp width, or how aggressively threads are grouped for coherence, in **future hardware generations** without breaking previously-compiled CUDA programs — since a CUDA binary only ever expresses ordinary scalar-looking instructions per thread, with the actual SIMD grouping entirely a runtime hardware decision, invisible to (and unassumed by) the compiled code itself.

### A concrete SM "subcore" microarchitecture example
One building block of a real GPU core (an NVIDIA "SM subcore," using a Volta V100-era design as the running example): one fetch/decode unit, paired with several different banks of 16-wide SIMD execution units (e.g., one bank for 32-bit floating point, another for 32-bit integer, another for transcendental/special functions like trig, another for loads/stores) — plus scalar (not vector!) register storage for many individual threads' worth of execution context (e.g., 128 threads' worth, in this simplified example — a heavily multi-threaded design). Since a warp is 32 threads wide but the actual floating-point ALU bank here is only 16-wide, a single warp instruction actually takes **two clock cycles** to fully issue across the whole warp — but this isn't wasted time: the freed-up cycle in between lets that same fetch/decode unit interleave dispatching a *different* kind of instruction (e.g., an integer operation) to a different functional-unit bank, effectively getting reasonably sophisticated instruction-level scheduling "for free" out of a fairly simple dispatch mechanism, without anything nearly as complex as a full CPU-grade out-of-order scheduler.

### Scaling up to a full SM (Streaming Multiprocessor)
A full SM core (again, V100-era) is built from **four** of these subcores, sharing one pool of on-chip `shared` memory storage across all four. Combined, one SM supports up to **64 resident warps** (64 × 32 = **2,048** concurrently-resident CUDA threads) at once — an enormous amount of available latency-hiding capacity — with each of the four subcores independently able to make progress on one warp instruction per clock, effectively behaving like a "4-way superscalar" core selecting, each clock, among a pool of up to 64 available warps (though each subcore can only actually select from its own quarter of that pool, in practice). None of this involves any genuinely new concept — it's simultaneous multi-threading, SIMD, and (loosely) superscalar issue, all composed together, at a scale considerably larger than a typical CPU core.

### Putting the whole chip together (V100 example)
Multiplying out: 80 SMs per chip × 4 subcores per SM × 16-wide FP32 SIMD × a ~1.2–1.6 GHz clock works out to roughly **12.7 TFLOPs** of peak single-precision throughput; and 80 SMs × 64 resident warps/SM × 32 threads/warp comes out to roughly **163,840 concurrently resident CUDA threads live on the chip at once** — vastly more than could ever all be actively executing in the very same clock cycle, but all genuinely present and available for the hardware to interleave between, for extreme latency-hiding capacity. (For comparison: a typical Intel CPU core of this era supports only 2 hyper-threads; this is the same underlying idea, just at a dramatically different scale.)

---

## 6. Mapping Thread Blocks Onto Real Hardware: Scheduling, Revisited

### A kernel launch looks exactly like a dynamic task queue
When compiled, a CUDA kernel carries along a small amount of metadata alongside its instructions: how many threads each block requires, and how many bytes of `shared` memory each block needs. A kernel launch like "run 8,000 thread blocks of this convolution kernel" is, at its heart, precisely the same **dynamic assignment / task-queue** pattern already covered for ISPC `task` launches and Cilk's fork-join scheduler — except here, the "worker pool" doing the scheduling isn't a software thread pool at all; **it's dedicated hardware, physically built into the GPU itself.**

The GPU's onboard work distributor greedily assigns thread blocks to available SM cores: given a block's known resource needs (execution contexts for its threads, plus its required shared-memory bytes), it packs blocks onto SMs for as long as both types of resource remain available, holds any remaining blocks back once an SM is full, and — as soon as a running block finishes and frees up its resources — immediately schedules the next waiting block into that SM. Because every block from the same kernel launch has *identical*, known resource requirements, this scheduling problem is unusually simple compared to a general bin-packing problem — there's no need to reason about differently-shaped resource requests competing for the same space.

### An important correctness rule: no partial-block execution, and no preemption
A thread block requesting *more* threads than a single SM can physically host at once (e.g., 256 threads on a hypothetical SM that only supports 128 concurrent execution contexts) is **rejected outright** — CUDA will not attempt to run "the first half now, then the second half later." Why not simply run the block in waves, as a naive read of dynamic scheduling might suggest? Because the programming model explicitly promises that all threads *within* a block are genuinely, simultaneously **co-resident and concurrently live** — a promise necessary to make constructs like `__syncthreads()` and shared-memory cooperation actually work correctly. If only half a block's threads were ever physically resident at once, and the kernel contained a barrier, the resident half would stall forever waiting for the other half to arrive at the same barrier — a **guaranteed deadlock**, since the hardware deliberately does **not** preempt already-running warps to swap in the other half (doing so would be both complex and detrimental to predictable performance). Rather than silently allow a program to deadlock, CUDA simply refuses to launch it in the first place if a single block's resource requirements exceed what one SM can provide.

### Inter-block communication: possible, but never assume ordering
Threads *within* the same block can safely use barriers, `shared` memory, and other tight cooperative constructs, precisely because they're guaranteed to be co-resident. Threads in **different** blocks have no such guarantee — the hardware might run them on different SMs at arbitrary, overlapping or non-overlapping times, in any order.

- **Safe pattern**: many threads, potentially from many different blocks, independently performing an **atomic** update into a single shared counter (e.g., every thread computing some value and atomically incrementing the appropriate bin of a shared histogram in global memory) — this is fine, because correctness here never depends on any particular *order* of updates, only on the update itself being atomic.
- **Unsafe (deadlock-prone) pattern**: one thread block writes a "signal" value that a *different* thread block spins waiting to observe before proceeding. If the GPU happens to schedule these two blocks such that the waiting block runs (or is scheduled) before the signaling block ever gets a chance to run, the waiting block spins forever, and (on hardware that can only run one block at a time) the signaling block never gets scheduled at all to unblock it — another genuine deadlock, this time arising purely from an unjustified assumption about inter-block scheduling order.

**The general rule**: cooperation *within* a thread block can safely rely on tight synchronization and assumed concurrency; cooperation *across* different thread blocks must never assume anything about relative scheduling order — only about the atomicity of individual shared operations, if used.

### Forward compatibility: another reason to let the hardware decide
Because the CUDA programming model deliberately doesn't let a compiled program bake in assumptions about exact warp width or the precise SIMD scheduling strategy, code written against an older GPU generation's smaller per-SM resource limits (e.g., a maximum of 256 concurrent threads per SM on some earlier chip) continues to run correctly, and can automatically take advantage of new hardware's larger resource limits (e.g., running more concurrent blocks per SM), without needing to be rewritten — a direct, practical payoff of NVIDIA's choice to have the hardware (rather than the programmer or compiler) own decisions like warp grouping and per-SM scheduling.

---

## 7. Summary

- Modern GPU compute is a direct evolutionary descendant of graphics hardware, arrived at via genuine historical hacks (repurposing pixel shaders for general computation) that eventually motivated a proper, dedicated general-purpose interface — CUDA's "compute mode."
- CUDA is fundamentally an **SPMD** programming model, structurally very similar to ISPC: a single kernel function is launched in bulk across a (potentially multi-dimensional) grid of thread blocks, each containing many CUDA threads, with `threadIdx`/`blockIdx`/`blockDim` playing the same role ISPC's `programIndex`/`programCount` and task indices play.
- The CUDA memory model separates host and device address spaces (data movement between them is explicit, much like message passing), and further exposes a hierarchy of per-thread, per-block (`shared`), and global memory scopes — with `shared` memory functioning as fast, programmer-managed on-chip storage exploitable for cooperative data reuse within a block.
- Underneath the SPMD abstraction, GPU hardware implements it via **implicit SIMD**: groups of (historically 32) threads, called **warps**, are dynamically detected to share a program counter and executed together on shared SIMD units — a hardware-level implementation detail deliberately hidden from the programming model itself, both for simplicity and to preserve forward compatibility across hardware generations.
- Thread blocks are scheduled onto SM cores via essentially the same **dynamic task-queue** pattern already seen in ISPC tasks and Cilk — just implemented directly in hardware — and the guarantee that all threads within one block are simultaneously co-resident (never partially, never preempted) is what makes intra-block barriers and shared-memory cooperation safe to rely on, while cross-block cooperation must never assume anything about relative execution order.

---

*Notes synthesized and paraphrased from the CS149 Fall 2023 Lecture 7 slide deck and lecture transcript, for study purposes — not a verbatim transcript.*
