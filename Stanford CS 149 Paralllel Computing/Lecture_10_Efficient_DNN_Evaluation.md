# CS149: Parallel Computing — Lecture 10
## Efficiently Evaluating Deep Neural Networks

> Course: Stanford CS149 (Parallel Computing)
> Topic: Mapping DNNs onto GPUs and CPUs
> Related: Assignment 3 (circle renderer), sets up Assignment 4

---

## 1. Housekeeping: Where We Are in the Course

- The course is entering the phase where new-concept density per lecture is slowing down, but workload (assignments) is ramping up.
- Focus for students right now: Assignment 3 practice + shoring up understanding of prior material.
- This lecture is placed here mostly for scheduling reasons (guest lecture slot), but is directly useful preparation for **Assignment 4**.
- Specialized DNN accelerator hardware (TPUs, etc.) is deliberately **not** covered today — that's a separate lecture (around Thanksgiving).

---

## 2. Assignment 3 Preview: The Circle Renderer Problem

Quick framing of the "real" part of Assignment 3 (parts 1–2 are warmups on CUDA mechanics; part 3 is the substantial problem):

- **Task:** render an image made entirely of circles, given each circle's center `(x, y)`, radius, and color.
- **Naive sequential algorithm:**
  1. For each circle (in a given order):
  2. Compute its bounding box.
  3. For each pixel in the bounding box, if the pixel center is inside the circle, blend its color into the pixel.
- **The twist:** circles are **semi-transparent**, so compositing is **order-dependent** (like layered stained glass). The array of circles is guaranteed to be in back-to-front order, and you must preserve that ordering when circles overlap.
- **Naive parallelization** (one thread per circle, all running concurrently) breaks correctness — race conditions on overlapping circles processed out of order.
- **Key insight:** ordering only matters where circles *overlap*. If two circles don't overlap, they can be processed in any order relative to each other.
- **Reframing as a data-parallel problem:** if, for every *pixel*, you knew the list of circles that could possibly overlap it (in the correct order), you could process all pixels in parallel safely. This is conceptually similar to "for every particle, which bin is it in?" problems discussed earlier in the course.
- **Suggested strategy for the assignment:**
  1. First get a **correct** parallel implementation (likely slow).
  2. Then optimize for performance while preserving correctness.

---

## 3. The DNN Workload, From Scratch

### 3.1 A Neuron Is Just a Dot Product + Nonlinearity

- Take an expression like a weighted sum of inputs passed through a nonlinearity — this is exactly a **dot product followed by a max operation**.
- In ML terminology this is a "neuron": a set of weights (a vector) applied to an input vector, optionally with a bias, passed through a nonlinear function.
- The nonlinearity used throughout the lecture is **max(0, x)** — i.e., ReLU (rectified linear unit), stripped of the fancy name.
- You can think of a single neuron as a simple binary classifier (output above/below zero).

### 3.2 Wiring Neurons Into Layers

- **Fully connected layer:** every output of layer *i* feeds into every neuron of layer *i+1*.
- **Convolutional layer (1D example):** each neuron only looks at a sliding window (e.g., every 3 inputs) — a *local* connectivity pattern instead of full connectivity.

### 3.3 From Diagrams to Linear Algebra

- A fully connected layer with multiple neurons, each taking the same input vector, is just a **matrix-vector product**: weights form a matrix, the input is a vector, output = `W · x`, then apply the nonlinearity elementwise.

### 3.4 Convolution as a Local Weighted Average

- 2D convolution example: for a `width × height` image, each output pixel is a weighted combination of surrounding input pixels (e.g., a 3×3 neighborhood).
- **Uniform weights (1/9 each)** → blurring (averaging neighborhood values).
- **Signed weights** (some positive, some negative) → gradient/edge detection (horizontal or vertical), i.e., a finite-difference/derivative operation.
- Real networks **learn** these weights rather than having them hand-set; classic visualizations (e.g., early ImageNet filters) show large banks of learned filters (e.g., 96 different 11×11 filters), each responding to different image features.

### 3.5 Stacking Convolutions: Tensors

- Input image: `W × H × 1` (or `× 3` for RGB).
- A bank of `numFilters` filters, each `k × k × inputChannels`, convolved against the input produces an output tensor of shape `W × H × numFilters`.
- Stack many such conv layers (each followed by a nonlinearity, and often downsampling like max-pooling) to build deep networks.
- Famous architectures (ResNet, U-Net, Inception, etc.) are all, computationally, sequences of these convolution blocks wired together in different topologies.

**Takeaway:** The core computational workload of DNNs (at least for image-style/conv networks) is **repeated convolution operations** — this is what we want to make fast.

---

## 4. Three Levers for Making DNNs Faster

1. **Better algorithms / architectures** (ML research territory)
   - Redesigning network topology to need fewer FLOPs/parameters for the same accuracy (e.g., ResNet/Inception vs. earlier "big" CNNs; MobileNet designed for phone deployment).
   - Example given: ~25x reduction in weights/compute over about 4 years for similar accuracy — far faster improvement than hardware alone provides.
   - Relevant to today's LLM landscape too: after a "scale is all that matters" phase, engineers now aggressively shrink models while preserving quality.
   - **Not** the focus of this course, but systems engineers need to track it since it changes what "the right system design" even means.

2. **Systems-level optimization for a fixed architecture** ← *this is the CS149-relevant lever*, covered in depth below (matrix multiply optimization, fusion, memory hierarchy usage on CPUs/GPUs).

3. **Specialized hardware** (deferred to a later lecture, briefly foreshadowed with tensor cores at the end of this lecture).

---

## 5. Convolution as Matrix Multiplication

### 5.1 The Naive Direct-Convolution Loop Nest

A direct (non-matrix) convolution implementation has **7 nested loops**:

```
for batch in batches:            # loop over images in the batch
  for filter in filters:         # loop over output channels
    for i, j in output pixels:   # loop over spatial output positions
      for input pixel in receptive field:  # e.g. 3x3 window
        for channel in input channels:      # e.g. RGB or 512 channels
          accumulate weight * input pixel
```

- Correct, but not fast — no attention to memory hierarchy or parallel hardware structure.

### 5.2 im2col: Turning Convolution into GEMM

Key trick (used since early convnet systems like Caffe): **reshape the convolution into a plain matrix-matrix multiply**, because heavily-optimized GEMM (General Matrix Multiply) libraries already exist.

- For a single filter: copy each receptive-field neighborhood of the input into a **row** of a new matrix (`numOutputPixels × filterSize`). The convolution becomes a **matrix-vector product** against the filter's flattened weight vector.
- For **multiple filters**: stack filter weight vectors as **columns** → now it's a full **matrix-matrix product**.
- For **multi-channel input** (e.g., 512 channels): the "row" per output pixel grows to `kernelH × kernelW × inputChannels`, and correspondingly the weight matrix grows.
- This matches NVIDIA's standard notation for conv-as-GEMM: given input tensor `X` (`W×H×C×N`) and `K` filters of size `R×S×C`, im2col builds matrices `A`, `B` such that `A × B = C`, where `C` holds the convolution output (`P×Q×N` per filter `K`).

**Cost of this approach:** massive data duplication — the same input pixel is copied into many rows/positions of the constructed matrix, which can blow up memory footprint (e.g., turning hundreds of MB of input into many GB of matrix data), which becomes a serious problem with larger batch sizes and during backpropagation (needing to retain intermediate activations).

---

## 6. Optimizing Matrix Multiplication (The Core Kernel)

### 6.1 Naive Triple-Loop GEMM Is Bandwidth-Bound

```
for i in 0..n:
  for j in 0..n:
    for k in 0..n:
      C[i][j] += A[i][k] * B[k][j]
```

- For `n × n` matrices: **O(n³) work**, but only **O(n²) data**. In principle, arithmetic intensity should scale as O(n) — great for compute-bound execution.
- But the *naive* loop order accesses:
  - `A`: row-major, sequential — fine.
  - `B`: column access on a row-major matrix — poor locality, effectively a new cache line per element.
  - `C`: reused correctly, stays in cache.
- Net effect: **≈1 useful math op per problematic memory access** → this naive implementation is bandwidth-bound, i.e., its *effective* arithmetic intensity is O(1), even though the algorithm's *theoretical* intensity is O(n). This is analogous to the bandwidth-bound problems from Assignment 1.

### 6.2 Fix: Blocking / Tiling

- Rewrite the computation as **block matrix multiplication**: partition `A`, `B`, `C` into submatrices ("blocks") of size `b × b`.
- Load one block of `A` and one block of `B` into cache/shared memory, multiply them, accumulate into the corresponding block of `C`, then move to the next blocks.
- **Arithmetic intensity of blocked computation:** work per block-step is `O(b³)`, data touched is `O(b²)` → arithmetic intensity is `O(b)`.
  - As `b → n`, arithmetic intensity approaches `O(n)` (ideal), but you can't fit the whole matrix in cache.
  - As `b → 1`, you're back to the naive, bandwidth-bound case.
  - **Design goal:** pick the largest block size `b` that still fits in the target cache level without causing capacity misses.
- In practice, this blocking can be applied **hierarchically** across multiple cache levels (L1, L2, L3, and even registers) — one level of blocking captures most of the benefit; further levels add smaller (e.g., ~2x) improvements.
- Empirically: blocking a large (multi-MB to GB scale) matrix multiply vs. the naive version can yield roughly a **10x speedup**.

### 6.3 Cache (CPU) vs. Shared Memory (GPU / Scratchpad)

- On a **CPU**, the cache is hardware-managed: the programmer issues normal loads, and the cache controller decides which cache line goes where. You don't control placement, and you can occasionally hit pathological cases (e.g., address-mapping collisions) that hurt performance unexpectedly.
- On a **GPU**, CUDA shared memory is a **scratchpad**: a distinct, software-managed address space. The programmer explicitly loads a block into a contiguous shared memory allocation. This is the key architectural difference between a cache and a scratchpad.

### 6.4 SIMD-Level Optimization

- Even within a block, how you vectorize matters a lot:
  - Naive SIMD dot-product approach requires "splatting" (broadcasting) scalar values from `A`, which wastes instruction slots, and creates a chain of dependent instructions (hurting instruction-level parallelism).
  - Alternative strategies: pre-transpose `B` (or `A`) so SIMD lanes access contiguous data cleanly; may require re-transposing the output `C` at the end.
  - The best micro-strategy is often a function of the target SIMD width, machine, and even the block's dimensions.
- Because different layers in a real network have different matrix shapes (e.g., very "thin" vs. "square" matrices), a **single fixed GEMM strategy is not optimal for every layer** — production libraries often select among several strategies/kernels per shape.

### 6.5 Implicit GEMM

- Problem with im2col: it explicitly materializes large duplicated matrices in memory (expensive in both memory footprint and bandwidth).
- **Implicit GEMM**: keep the loop structure of a blocked matrix multiply, but replace direct array indexing with an **accessor function** that computes, on the fly, the corresponding location in the *original* (unduplicated) input tensor.
  - This trades extra **address-computation math** for **avoiding the duplication of data in DRAM**.
  - Data is still "scattered" when read from DRAM into cache/shared memory, but once inside the fast on-chip memory, it's laid out densely and the actual block matrix-multiply runs on that dense, cache-resident data.
  - High-performance implementations often **precompute address-calculation math into lookup tables** to avoid repeating non-trivial index arithmetic per element.
- NVIDIA's **CUTLASS** library provides building blocks for this style of fast, flexible matrix multiplication — a middle ground between hand-written low-level CUDA/PTX and a full framework like TensorFlow.

### 6.6 Need for Enough Work to Saturate the GPU

- Even with a good implementation, the GPU needs sufficiently large matrices/output sizes to have enough independent work to stay busy.
- Small batch sizes (e.g., batch size 1–3, often used just to fit in memory) reduce achievable throughput because there isn't enough parallel work to fill the machine — a real, measurable cost of memory-constrained small-batch inference.

### 6.7 cuDNN: Multiple Algorithms, Selectable

- NVIDIA's cuDNN convolution API (`cudnnConvolutionForward`, etc.) exposes **multiple algorithm choices**, matching the concepts above:
  - **Implicit GEMM** (default): loop-index into the original tensors, as described in 6.5.
  - **Direct** (blocked direct convolution, no reshaping to GEMM).
  - **GEMM (explicit)**: materialize im2col matrices and call a normal GEMM.
  - **Winograd** and **FFT-based** algorithms (see §8 below for the algebraic idea behind Winograd-style savings).
  - The caller can pick which algorithm/parameters to use based on tensor shapes and hardware.

---

## 7. Operator Fusion: Fighting Bandwidth-Boundedness Between Layers

- Real networks chain many operations back-to-back: conv → bias-add → scale → max-pool → next conv, etc.
- Naively, each op reads its full input tensor from memory and writes its full output tensor back to memory — even though many of these ops (bias add, scaling, max-pool) do very little arithmetic per element. These are **severely bandwidth-bound**.
- **Fusion idea:** while a block of output is still resident in cache/shared memory (right after the matrix multiply produces it), immediately apply the next cheap operations (bias, scale, max-pool) **before** writing back to memory.
  - Example: fusing a 2×2 max-pool immediately after computing a block avoids both an extra store+load *and* reduces the data actually written out by 4x (since max-pool downsamples).
- Historically, this led to an explosion of special-cased fused API entry points in frameworks (e.g., "Conv2D fused with BatchNorm") because compilers couldn't automatically discover these fusions.
- Newer compiler frameworks (e.g., **JAX**, **Triton**) aim to automatically detect fusable sequences of tensor operations and generate fused code, though this remains an evolving/imperfect area.

**Bottom line:** you need *both* (a) a fast core matmul/conv kernel, and (b) smart fusion of the surrounding elementwise/reduction ops — otherwise the "glue" operations between layers dominate runtime due to bandwidth limits.

---

## 8. Case Study: Flash Attention (Fusing Through Softmax)

### 8.1 The Attention Workload

- Sequence-to-sequence Transformers: input is a token sequence, output predicts the next token autoregressively.
- Core computational block: **attention**, operating on three tensors `Q`, `K`, `V` (query, key, value), each roughly `N × D` (N = sequence length, D = embedding dimension).
- Computation:
  1. `S = Q · Kᵀ` → an `N × N` matrix (outer-product-style matmul). For long sequences (e.g., N = 10,000+), this is a huge, multi-gigabyte matrix.
  2. **Softmax** applied row-wise to `S`.
  3. `O = softmax(S) · V` → another matrix-vector/matrix-matrix product.

### 8.2 Why Softmax Blocks Naive Fusion

- Softmax on a row `x` requires: `softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`.
- This appears to require knowing the **max of the entire row** (and the full sum) before you can finish computing any single output — seemingly incompatible with processing the row in small blocks.
- Naive implementation: compute the full `N×N` matrix, store it to memory, reload it row-by-row for softmax, reload again for the final matrix-vector product. Very bandwidth-heavy, and infeasible to keep the full matrix on-chip for large N.

### 8.3 The Flash Attention Trick: Online/Chunked Softmax

- Key mathematical fact: softmax **can be computed incrementally in chunks**, by maintaining a **running max** and a **running (rescaled) sum** as you process the row in pieces.
  - If you split a row `X` into chunks `X1`, `X2`, the max of `X` is `max(max(X1), max(X2))`.
  - If you initially compute a partial softmax numerator using a chunk's local max, you can later **rescale** it once you learn the true global max — the algebra works out because of how `exp` and subtraction interact (a short derivation, offloaded to office hours/notes in the actual course).
- This enables processing **Q, K, V in blocks**: load a block of Q, K, V; compute the sub-matrix multiply; compute a chunked/rescaled softmax on that sub-block; do the partial matrix-vector product; accumulate into the output `O` — all without ever materializing the full `N×N` matrix in memory.
- **Result:**
  - Memory requirement drops from **O(N²)** to **O(block_size²)**.
  - This allowed practical sequence lengths to jump (roughly from ~8K to ~32K tokens in the models discussed), since the bottleneck was memory footprint, not raw compute.
  - Speed improvement was described as a modest constant factor, but the memory-footprint reduction was the transformative effect (enabling much longer context windows). This class of optimization is understood to be part of what enabled models like GPT-4-era context lengths.

**Conceptually:** this is the same "fuse across a chunk, keep partial state resident on-chip, rescale/accumulate at the end" idea as fusing max-pool into a conv block (§7) — just applied to a trickier, seemingly-global reduction (softmax).

---

## 9. Beyond Kernel/Fusion Optimization: Other Techniques (Brief)

- **Lower precision arithmetic:** moving from full precision down to 16-bit, 8-bit, and even ~4-bit representations to trade accuracy for throughput/memory savings. Actively being pushed further by current GPU hardware generations.
- **Sparsity:** exploiting zero/near-zero weights or activations to skip work (mentioned but not detailed in this lecture).
- **Algorithmic identities that reduce multiply count**, e.g. **Winograd convolution**: by identifying common sub-expressions across overlapping output computations, you can trade some multiplications for additional additions (fewer multiplies, more adds) — conceptually related to FFT-based convolution, which also exploits algebraic structure to reduce redundant work. Whether this trade is a net win is hardware-dependent.

---

## 10. Why Are GPUs a Good (and Imperfect) Fit for DNNs?

### 10.1 Why GPUs Are Good

- **Abundant parallelism** — DNN workloads are dominated by large matrix multiplications with tons of independent work.
- **High arithmetic intensity achievable** — if you use good blocking/data-reuse strategies (§6), you can actually exploit a GPU's compute capability instead of being memory-bound.
- **Highly SIMD-friendly** — the same weights are applied across many inputs, meaning the same instruction stream can be amortized across large data.
- **Historical fit** — GPUs already had massive FLOPs available for graphics workloads and happened to be well-positioned when deep learning took off roughly a decade ago.

### 10.2 Why GPUs Are Imperfect

- A GPU is a **general-purpose** parallel processor, while most of this workload boils down to simple matrix/matrix-vector multiply operations — a lot of GPU flexibility goes unused for this narrow pattern.
- **Instruction/control overhead amortization** is the classic motivation for SIMD (and beyond): the more math you can pack behind a single instruction's overhead (control, data access, etc.), the more efficient you are.
- **NVIDIA's own comparisons (their estimates, so read with some skepticism given incentives to make custom accelerators look necessary):**
  - Doing adds/multiplies individually on a general GPU core: **~2000x** less energy-efficient than dedicated matrix-multiply silicon.
  - Using a small fixed op like a 4-element dot product instruction: **~500x** less efficient.
  - Using a small (e.g., 4×4) matrix-multiply instruction (bundling ~64 ops into one instruction): closer to only **~30%** less efficient than dedicated silicon.
- **Tensor Cores** are NVIDIA's answer: dedicated hardware support for small fixed-size matrix-multiply instructions (e.g., an 8×4 by 4×8 matrix multiply — roughly 128 ops in one instruction).
  - If your workload is phrased in terms of these small matrix multiplies, tensor cores unlock dramatically more throughput than regular CUDA cores:
    - ~19.5 TFLOPS via standard CUDA cores at 32-bit precision.
    - ~300 TFLOPS via tensor cores at 16-bit precision — over an order of magnitude more compute, if your problem can be expressed as matrix multiplies.

**Framing for later lectures:** this tension (general-purpose flexibility vs. narrow, energy-efficient specialization) is exactly what motivates the dedicated DNN accelerator hardware discussed in a future lecture.

---

## 11. Summary / Mental Model

The lecture frames DNN performance work as spanning three interacting layers, with this course (149) focused squarely on the middle one:

| Lever | Who owns it | What it does |
|---|---|---|
| Algorithm/architecture design | ML researchers | Reduce FLOPs/params needed for a given accuracy (e.g., ResNet vs. earlier CNNs, MobileNet) |
| **Systems optimization (this lecture)** | **Systems/compiler engineers** | **Map a fixed architecture efficiently onto real hardware: blocking, implicit GEMM, fusion, precision** |
| Hardware specialization | Hardware architects | Build instructions/silicon (tensor cores, TPUs) matched to the dominant operation (matrix multiply) |

**Core recurring theme across the whole lecture:** identify operations that are naturally **bandwidth-bound** (naive matmul, elementwise ops between layers, softmax-in-attention) and restructure the computation — via **blocking/tiling** and **fusion** — so that data stays resident in fast on-chip memory (cache, shared memory, registers) long enough to do much more work per byte moved, i.e., **raise effective arithmetic intensity**.

### Key terms introduced
- Arithmetic intensity, bandwidth-bound vs. compute-bound
- Blocking / tiling (cache-level and register-level)
- Scratchpad memory (GPU shared memory) vs. hardware-managed cache
- im2col / explicit GEMM vs. implicit GEMM
- Operator fusion
- Winograd convolution, FFT-based convolution
- Flash Attention / online (chunked) softmax
- Tensor cores
