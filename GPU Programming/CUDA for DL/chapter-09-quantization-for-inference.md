# Chapter 9 — Quantization for Inference

*Part 8: Quantization for Inference. Confirmed directly from the book's companion repo: `book.cu/7_quant/README.md` — its own header literally reads "Chapter 8: Quantization," a clean match with this course's numbering, no drift this time. Nine self-contained "ultra-minimal" `.cu` files, each a single standalone program with its own kernels, data generation, and accuracy report — plus a tenth, more advanced bonus file (`awq.cu`) not counted in the README's "9 fundamentals" list.*

Every file in this chapter follows the same shape: generate synthetic FP32 data, quantize it, dequantize it, and print real error metrics (MSE, MAE, max error) comparing the round-tripped result against the original. That "measure the damage, every time" discipline is worth noticing on its own — it's Chapter 3's CPU-first verification habit, applied to a setting where the "bug" isn't correctness in the pass/fail sense, it's *how much precision you're willing to trade for memory*.

---

## 9.1 The Quantization Formula, Confirmed Once, Reused Everywhere

`fp32_int8.cu` is the foundation every other file in this chapter builds on:

```cuda
__global__ void quantize_fp32_to_int8(float* input, signed char* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);   // clamp to signed INT8 range, using 127 not 128 for symmetry
    output[idx] = (signed char)roundf(scaled);
}

__global__ void dequantize_int8_to_fp32(signed char* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = (float)input[idx] * scale;
}
```

The scale itself: `scale = max(abs(data)) / 127.0f` — the largest-magnitude value in the tensor gets mapped to the edge of the representable INT8 range, and everything else scales proportionally. The `main()` function computes this from real generated data (normally distributed, mean 0, std 2.0, 1M elements) and reports real numbers:

```
Scale: 0.052180
Memory reduction: 4x (FP32 -> INT8)
Accuracy Results:
  MSE: 0.000227
  MAE: 0.011234
  Max Error: 0.026090
```

*(Illustrative — your own run will print slightly different numbers depending on the random seed, but the shape holds.)* Keep this formula in view as you read the rest of the chapter: **every subsequent file reuses this exact `scale → divide → clamp → round` sequence.** The only thing that ever changes from file to file is *which scale value gets looked up* for a given element — that single variation is this entire chapter's real subject.

**Deep dive: the book's own reported MSE matches quantization-noise theory almost exactly.** There's a standard, well-established result from quantization theory worth checking the reported numbers against: for a uniform quantizer with step size `Δ` (here, `Δ = scale = 0.052180`), the rounding error is well-approximated as uniformly distributed over `[-Δ/2, Δ/2]`, giving an expected mean-squared error of `Δ²/12`:

```
Δ²/12 = (0.052180)² / 12 = 0.0027228 / 12 ≈ 0.0002269
```

The book's own reported figure: **MSE: 0.000227.** That's agreement to three significant figures, from a formula that uses nothing but the scale itself — real confirmation that this kernel's error behaves exactly the way textbook quantization theory predicts it should, not some looser approximation. It's a useful formula to keep on hand for the rest of this chapter, too: every later section's error can be sanity-checked against `Δ²/12` for whatever `Δ` that section's scheme actually uses.

## 9.2 INT4 and the Bit-Packing Trick

INT8 compresses 4×; getting to INT4's 8× requires an extra step ordinary types don't give you for free — **two 4-bit values packed into one 8-bit byte**, since there's no native 4-bit integer type in CUDA C++:

```cuda
__global__ void quantize_fp32_to_int4_packed(float* input, uint8_t* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size + 1) / 2) return;                 // each thread handles TWO input elements
    int elem1_idx = idx * 2, elem2_idx = idx * 2 + 1;

    float scaled1 = input[elem1_idx] / scale;
    scaled1 = fmaxf(fminf(scaled1, 7.0f), -7.0f);       // INT4 signed range: [-7, 7]
    int8_t quant1 = (int8_t)roundf(scaled1);

    int8_t quant2 = 0;
    if (elem2_idx < size) {
        float scaled2 = input[elem2_idx] / scale;
        scaled2 = fmaxf(fminf(scaled2, 7.0f), -7.0f);
        quant2 = (int8_t)roundf(scaled2);
    }

    // Pack: first value in the upper 4 bits, second in the lower 4 bits
    uint8_t packed = ((quant1 & 0x0F) << 4) | (quant2 & 0x0F);
    output[idx] = packed;
}
```

Unpacking needs one extra idea: **sign extension.** A 4-bit two's-complement value's sign bit is bit 3 (the `0x08` bit within the nibble), not bit 7 the way `int8_t` expects — so after extracting a nibble, you have to manually propagate that sign bit upward before the value means what it should as an `int8_t`:

```cuda
int8_t quant1 = (packed >> 4);          // upper nibble
int8_t quant2 = packed & 0x0F;           // lower nibble
if (quant1 & 0x08) quant1 |= 0xF0;       // sign bit set -> extend it: 0b1001 (=-7 in 4 bits) becomes 0b11111001 (-7 as int8_t)
if (quant2 & 0x08) quant2 |= 0xF0;
output[elem1_idx] = (float)quant1 * scale;
```

Skip that `if` check and every negative INT4 value silently decodes as a large *positive* number instead — a real, easy-to-miss bug class specific to sub-byte packed formats, worth remembering the next time you touch a packed weight format (this exact pattern appears throughout real quantized-LLM inference code, e.g. GGUF's Q4 formats).

**Deep dive: predicting INT4's error penalty before you even run it.** §9.1's `Δ²/12` formula lets you predict, not just measure, how much worse INT4 should be. For the same data range, INT8's step size is `range / (2×127)` while INT4's is `range / (2×7)` — INT4's step is `127/7 ≈ 18.14×` larger. Since error scales with `Δ²`, that predicts an MSE roughly `18.14² ≈ 329×` higher for INT4 than INT8, on the same data. Keep that number in hand for the Hands-On Lab below (Exercise 1 asks you to build a comparison table across every kernel in this chapter) — if your own measured ratio lands anywhere near 329×, that's the formula from §9.1 confirming itself a second time, on a completely different scheme.

## 9.3 Symmetric vs. Asymmetric: Signed vs. Unsigned

`int8_uint8.cu` puts the two schemes for a single element side by side:

```cuda
// Symmetric (int8_t): zero-point is implicitly 0, one scale parameter
__global__ void quantize_symmetric(float* input, int8_t* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

// Asymmetric (uint8_t): scale AND zero-point, uses the full [0,255] range
__global__ void quantize_asymmetric(float* input, uint8_t* output, float scale, float zero_point, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // Note: a more precise formula is round(x/scale + zero_point_int). This is a simplification.
    float scaled = roundf(input[idx] / scale) + zero_point;
    scaled = fmaxf(fminf(scaled, 255.0f), 0.0f);
    output[idx] = (uint8_t)scaled;
}
```

That inline comment is the book's own — a real, self-acknowledged simplification (rounding before adding the zero-point instead of after, which can shift results by up to half an integer step versus the textbook-correct order). Worth reading, not skipping past.

The theoretical distinction the file's docstring draws is the one that matters in practice: **symmetric quantization is ideal for data naturally centered around zero** (weights, typically) — it wastes no representational range on an offset it doesn't need. **Asymmetric quantization is better for data that isn't zero-centered** (activations after a ReLU, which are strictly non-negative) — without a zero-point, a symmetric scheme would waste half its INT8 range representing negative values that never actually occur.

**Deep dive: turning "wastes half the range" into a predicted error number.** §9.1's formula makes this concrete instead of just qualitative. Quantizing genuinely non-negative data (say, post-ReLU activations, range `[0, R]`) symmetrically still allocates the full `±127` code range, but half of it (`[-127, 0)`) represents values that never occur — so the *effective* step size for the data that actually exists is `Δ_sym = R / 127`. An asymmetric scheme uses the *entire* `[0, 255]` range for that same `[0, R]` data: `Δ_asym = R / 255`. That's roughly a `2×` finer step size for asymmetric — and by `Δ²/12`, a **≈4× worse MSE** from using symmetric quantization on data that's actually non-negative, for no reason other than picking the wrong scheme. This is exactly why the file's docstring insists on matching the scheme to the data's actual distribution rather than defaulting to one — the cost of getting it wrong is a clean, predictable 4×, not a vague "somewhat worse."

## 9.4 The Granularity Ladder: Tensor → Group → Block → Channel

This is the chapter's own version of Chapter 6's optimization ladder — except instead of rung after rung buying more *speed*, each rung here buys more *accuracy*, at the cost of storing more scale values. Read the four kernels' core index computations side by side; the `scale → divide → clamp → round` body is **identical** in every one — only the scale *lookup* changes:

```cuda
// tensorwise.cu — ONE scale for the entire tensor
float scale = /* single value, passed as a scalar argument */;

// groupwise.cu — one scale per contiguous GROUP of elements
int group_idx = idx / group_size;
float scale = group_scales[group_idx];

// blockwise.cu — one scale per 2D spatial BLOCK (for image/feature-map-shaped tensors)
int row = idx / tensor_width, col = idx % tensor_width;
int block_row = row / block_height, block_col = col / block_width;
int blocks_per_row = (tensor_width + block_width - 1) / block_width;
int block_idx = block_row * blocks_per_row + block_col;
float scale = block_scales[block_idx];

// channelwise.cu — one scale per CHANNEL (the fastest-varying / last dimension)
int channel_idx = idx % num_channels;
float scale = channel_scales[channel_idx];
```

The README frames this concretely against a real ML operation, the linear projection `(B, T, C) @ (C, H) → (B, T, H)` that underlies every linear layer in a transformer: a weight matrix's different **output channels** can have wildly different natural value ranges (one output neuron's weights might all be small, another's large), and a single tensor-wide scale forces every channel through the *same* resolution — wasting precision on small-range channels to accommodate the largest one. The book's own stated recommendation, direct from its "Performance Insights" section: **per-channel (specifically per-output-channel) quantization is critical for weight matrices** for exactly this reason, while a single tensor-wise scale is "simplest" but leaves real accuracy on the table whenever channels' ranges genuinely differ.

**Deep dive: how total the resolution loss gets, with a deliberately lopsided two-channel example.** Take a tensor with two channels — channel A's values span `[-1, 1]`, channel B's span `[-100, 100]`. A single tensor-wide scale must accommodate the *largest* magnitude anywhere: `scale = 100/127 ≈ 0.787`. Applied to channel A's data, that scale means each representable INT8 step corresponds to a jump of `0.787` in the original value — but channel A's *entire range* is only `2.0` wide. Channel A ends up using only `2.0/0.787 ≈ 2.5` distinct quantization levels out of the 254 available — its data collapses to something like 2 or 3 possible values, an almost total loss of resolution, purely because it happened to share a tensor with a much larger-range channel. A per-channel scheme gives channel A its *own* scale, `1/127 ≈ 0.00787`, restoring its full 254-level resolution regardless of what channel B looks like. This is the concrete mechanism behind the qualitative claim above — not "somewhat worse," but a specific channel losing *nearly all* of its representable precision to a completely unrelated channel's larger scale.

## 9.5 Calibration: Getting the Scale Right — and Two Real Correctness Subtleties

`calibrator.cu` implements and compares two ways to *compute* a scale from representative data, rather than assuming you already have one:

```cuda
// Min-max calibration: exact range, but sensitive to outliers
__global__ void calibrate_min_max(float* calibration_data, float* scale_output,
                                  float* zero_point_output, int size) {
    extern __shared__ float sdata_min_max[];
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < size) ? calibration_data[idx] : 0.0f;
    if (tid == 0) { sdata_min_max[0] = val; sdata_min_max[1] = val; }
    __syncthreads();
    // atomicMin/Max don't support float directly on all architectures, so bits are
    // reinterpreted as int for the atomic operation.
    atomicMin((int*)&sdata_min_max[0], __float_as_int(val));
    atomicMax((int*)&sdata_min_max[1], __float_as_int(val));
    __syncthreads();
    // Note: This is a simplification; a two-level reduction would be needed for multiple blocks.
    if (tid == 0 && blockIdx.x == 0) {
        float min_val = sdata_min_max[0], max_val = sdata_min_max[1];
        float range = max_val - min_val;
        if (range > 1e-6) { *scale_output = range / 255.0f; *zero_point_output = -min_val / *scale_output; }
        else { *scale_output = 1.0f; *zero_point_output = 0.0f; }
    }
}
```

Two real issues live in this kernel, and they're worth treating as a genuine critical-reading exercise rather than taking on faith:

**The self-acknowledged one:** `extern __shared__ float sdata_min_max[]` is scoped **per block**, so `atomicMin`/`atomicMax` here only ever reduce within a single block's 256-or-so elements. The final scale/zero-point computation only runs `if (tid==0 && blockIdx.x==0)` — meaning if this kernel is ever launched with more than one block (which it would be, for any dataset larger than one block's worth of threads), **every block after block 0 computes a min/max that's simply thrown away.** The comment calls this out honestly: *"a two-level reduction would be needed for multiple blocks."* As written, this kernel only gives a correct answer for datasets small enough to fit in one launch.

**A subtler one, worth working out yourself:** `atomicMin((int*)&x, __float_as_int(val))` — reinterpreting a float's bits as a signed integer for an atomic comparison — is only an *exact* substitute for real floating-point min/max when every value involved is **non-negative**. IEEE-754's bit layout happens to preserve ordering under a signed-integer reinterpretation for positive floats, but *not* for negative ones (more-negative floats have *smaller* magnitude bit patterns near the top of their exponent range in a way that doesn't monotonically match signed-integer comparison). Compare this directly against `dynamic_static.cu`'s own `reduce_absmax` kernel a few files over in the same chapter, whose comment explicitly justifies why *its* use of the identical trick is safe: *"same order as the values, so atomicMax on the bits is exact"* — because it calls `fabsf()` **before** the atomic, guaranteeing every value is non-negative first. `calibrate_min_max` applies the identical bit-reinterpretation trick to **raw, possibly-negative** calibration data, with no such guarantee. Given `fp32_int8.cu`'s own test data generator produces a mean-0 normal distribution (i.e., routinely negative values), this isn't a hypothetical edge case — it's exactly the kind of input this chapter's own examples generate everywhere else. Reading two files in the same chapter against each other like this is exactly the habit worth building; it's how the tie-breaking bug in Chapter 5 and the benchmark-timing bug in Chapter 7 both actually got caught.

The **percentile calibration** kernel sidesteps outlier-sensitivity differently — rather than the exact max, it approximates a 95th-percentile-like robust bound as 95% of the reduced maximum absolute value, via a standard shared-memory tree reduction:

```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
}
```

## 9.6 Dynamic vs. Static Scales

`dynamic_static.cu` contrasts a **precomputed, fixed scale** (`quantize_static`, taking `static_scale` as a plain argument — cheapest at inference time, since there's no reduction to run) against a scale **computed fresh from the actual data being quantized**, via the correctly-implemented (abs-first) reduction kernel from §9.5:

```cuda
__global__ void reduce_absmax(const float* input, unsigned int* max_abs_bits, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < size) ? fabsf(input[idx]) : 0.0f;      // abs() first — this is why the atomic trick is exact here
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMax(max_abs_bits, __float_as_uint(sdata[0]));   // folds EVERY block's result — no single-block bug here
}
```

Notice this version's final `atomicMax` runs unconditionally for every block (not gated behind `blockIdx.x==0`), which is exactly the fix §9.5's exercise asks you to apply to `calibrate_min_max`. The tradeoff the book frames directly: **static is faster** (no reduction kernel needed at inference time at all) **but risks drift** if the real runtime data distribution shifts away from whatever it was calibrated on; **dynamic is more accurate** (always matches the actual current data) **but costs a full reduction kernel's worth of extra latency on every single call** — use it, per the README's own guidance, "for critical paths only."

**Deep dive: when does dynamic quantization's overhead actually stop mattering?** `reduce_absmax` is a single memory-bound pass over the tensor — it reads every element once and does essentially no arithmetic per element (a running max, nothing more), so its cost is dominated entirely by `N × 4 bytes` of memory traffic at whatever bandwidth the GPU delivers. The operation that scale then feeds into — quantizing a weight matrix before a GEMM, say — is, per Chapter 3 §3.3.2, potentially **compute-bound**, with a runtime governed by FLOPs rather than bytes. As the matrix gets larger, the GEMM's compute time grows roughly with `M×N×K`, while the reduction's cost grows only linearly with the number of elements being calibrated — meaning the reduction's *relative* overhead shrinks the larger the operation it precedes gets. Put differently: dynamic quantization is expensive relative to a *tiny* op (where the extra memory pass might double your total time) and nearly free relative to a *large, compute-bound* op (where it's a rounding error on top of a much longer GEMM). This is Chapter 1's roofline framing again, this time explaining not whether one op is memory- or compute-bound, but whether an extra *fixed-cost preprocessing step* is worth paying for a given downstream operation's size.

## 9.7 AWQ: Activation-Aware Weight Quantization

The bonus file, and the most advanced idea in the chapter: instead of choosing a quantization scale purely from a weight tensor's own value distribution, **choose it using the *activations* that weight will be multiplied against.**

```cuda
// Naive: quantize weights with no regard for which activations multiply them
__global__ void matmul_naive_quant_kernel(const float* x, const float* W, float* out, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float quant_val = roundf(W[k * N + col] * 7.0f);    // simulated INT4, range [-7,7]
        sum += x[k] * (quant_val / 7.0f);
    }
    out[col] = sum;
}

// AWQ: scale activation UP and weight DOWN by the same per-channel factor before quantizing
__global__ void matmul_awq_quant_kernel(const float* x, const float* W, float* out, const float* scales, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float activation_to_use = x[k] * scales[k];
        float scaled_weight = W[k * N + col] / scales[k];
        float quant_val = roundf(scaled_weight * 7.0f);
        sum += activation_to_use * (quant_val / 7.0f);
    }
    out[col] = sum;
}
```

The scale itself is derived directly from activation magnitude:

```cuda
for (int k = 0; k < K; ++k) h_scales[k] = 1.0f / (fabsf(h_x[k]) + 0.1f);   // small |activation| -> LARGE scale
// ...normalized by the average scale across all K channels...
```

Trace the effect through: a **large-magnitude activation** gets a **small** scale (since scale ∝ 1/|activation|) → its paired weight is divided by that small number → the weight's value going *into* the rounding step is **larger in magnitude** → it occupies more of the [-7, 7] quantization range → it gets **more absolute precision**, relatively speaking. The weight that gets multiplied by the *loudest* activation — and therefore contributes the most to the output sum, and whose quantization error matters most — is exactly the one AWQ protects. A weight paired with a tiny, barely-contributing activation is allowed to lose more relative precision, because it barely matters to the final sum either way. This is the real, published idea behind AWQ (Activation-aware Weight Quantization, MIT/NVIDIA), reduced to a toy example small enough to trace by hand: **M=1, K=6, N=4** — six input activations, a 6×4 weight matrix, one output row. Exercise 4 below asks you to do exactly that trace yourself.

**Deep dive: the specific failure mode this scheme exists to prevent.** `matmul_naive_quant_kernel` rounds `W×7` directly, with no scale factor at all — meaning any weight smaller in magnitude than `1/14 ≈ 0.071` rounds to exactly **zero**, regardless of what it's multiplied against. A weight of `0.05` paired with an activation of `25.0` (this file's own largest input value) contributes `0.05 × 25.0 = 1.25` to the true output — genuinely significant — but vanishes to `0 × 25.0 = 0` under naive quantization, a **100% loss** of that term. AWQ's per-channel rescaling directly targets this: by dividing that same `0.05` weight by a *small* scale (since it's paired with a large activation), the value going into the rounding step is pushed well away from zero before rounding ever happens, so it survives instead of disappearing. This is the concrete, worst-case version of the "protects large-activation weights" mechanism described above — it's not just that those weights get *somewhat* better relative precision, it's that without this scheme, some of them would be thrown away entirely.

## 9.8 Verifying the Deep-Dive Predictions from Python

The deep dives above made several specific, falsifiable numeric predictions — the `Δ²/12` match, the ~329× INT4-vs-INT8 ratio, the ~4× symmetric-vs-asymmetric ratio. This section wraps the confirmed kernels via `load_inline` so you can check every one of them yourself, plus **run the two real bugs from §9.5 live** rather than just reading about them.

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ========= §9.1: FP32 <-> INT8, confirmed verbatim =========
__global__ void quantize_int8_kernel(const float* input, int8_t* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float s = fmaxf(fminf(input[idx] / scale, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(s);
}
__global__ void dequantize_int8_kernel(const int8_t* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = (float)input[idx] * scale;
}

// ========= §9.2: FP32 <-> packed INT4, confirmed verbatim =========
__global__ void quantize_int4_packed_kernel(const float* input, uint8_t* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size + 1) / 2) return;
    int e1 = idx * 2, e2 = idx * 2 + 1;
    int8_t q1 = (int8_t)roundf(fmaxf(fminf(input[e1] / scale, 7.0f), -7.0f));
    int8_t q2 = 0;
    if (e2 < size) q2 = (int8_t)roundf(fmaxf(fminf(input[e2] / scale, 7.0f), -7.0f));
    output[idx] = ((q1 & 0x0F) << 4) | (q2 & 0x0F);
}
__global__ void dequantize_int4_packed_kernel(const uint8_t* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size + 1) / 2) return;
    uint8_t packed = input[idx];
    int8_t q1 = (packed >> 4), q2 = packed & 0x0F;
    if (q1 & 0x08) q1 |= 0xF0;                    // sign extension -- §9.2's real bug class if omitted
    if (q2 & 0x08) q2 |= 0xF0;
    int e1 = idx * 2, e2 = idx * 2 + 1;
    output[e1] = (float)q1 * scale;
    if (e2 < size) output[e2] = (float)q2 * scale;
}

// ========= §9.3: symmetric vs. asymmetric, confirmed verbatim =========
__global__ void quantize_asymmetric_kernel(const float* input, uint8_t* output, float scale, float zp, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float s = roundf(input[idx] / scale) + zp;
    output[idx] = (uint8_t)fmaxf(fminf(s, 255.0f), 0.0f);
}
__global__ void dequantize_asymmetric_kernel(const uint8_t* input, float* output, float scale, float zp, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = ((float)input[idx] - zp) * scale;
}

// ========= §9.5: the two calibration kernels, bugs included, confirmed verbatim =========
__global__ void calibrate_min_max_kernel(const float* data, float* scale_out, float* zp_out, int size) {
    extern __shared__ float sdata_min_max[];
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < size) ? data[idx] : 0.0f;
    if (tid == 0) { sdata_min_max[0] = val; sdata_min_max[1] = val; }
    __syncthreads();
    atomicMin((int*)&sdata_min_max[0], __float_as_int(val));
    atomicMax((int*)&sdata_min_max[1], __float_as_int(val));
    __syncthreads();
    if (tid == 0 && blockIdx.x == 0) {              // <-- confirmed bug: only block 0's result is ever used
        float min_val = sdata_min_max[0], max_val = sdata_min_max[1];
        float range = max_val - min_val;
        if (range > 1e-6f) { *scale_out = range / 255.0f; *zp_out = -min_val / *scale_out; }
        else { *scale_out = 1.0f; *zp_out = 0.0f; }
    }
}

// ========= Launchers =========
torch::Tensor quantize_int8(torch::Tensor input, double scale) {
    int n = input.numel(); auto out = torch::empty({n}, input.options().dtype(torch::kInt8));
    int t=256, b=(n+t-1)/t;
    quantize_int8_kernel<<<b,t>>>(input.data_ptr<float>(), out.data_ptr<int8_t>(), (float)scale, n);
    return out;
}
torch::Tensor dequantize_int8(torch::Tensor input, double scale) {
    int n = input.numel(); auto out = torch::empty({n}, input.options().dtype(torch::kFloat32));
    int t=256, b=(n+t-1)/t;
    dequantize_int8_kernel<<<b,t>>>(input.data_ptr<int8_t>(), out.data_ptr<float>(), (float)scale, n);
    return out;
}
torch::Tensor quantize_int4(torch::Tensor input, double scale) {
    int n = input.numel(), p = (n+1)/2; auto out = torch::empty({p}, input.options().dtype(torch::kUInt8));
    int t=256, b=(p+t-1)/t;
    quantize_int4_packed_kernel<<<b,t>>>(input.data_ptr<float>(), out.data_ptr<uint8_t>(), (float)scale, n);
    return out;
}
torch::Tensor dequantize_int4(torch::Tensor input, int64_t n, double scale) {
    auto out = torch::empty({n}, input.options().dtype(torch::kFloat32));
    int p = (n+1)/2, t=256, b=(p+t-1)/t;
    dequantize_int4_packed_kernel<<<b,t>>>(input.data_ptr<uint8_t>(), out.data_ptr<float>(), (float)scale, (int)n);
    return out;
}
torch::Tensor quantize_asym(torch::Tensor input, double scale, double zp) {
    int n = input.numel(); auto out = torch::empty({n}, input.options().dtype(torch::kUInt8));
    int t=256, b=(n+t-1)/t;
    quantize_asymmetric_kernel<<<b,t>>>(input.data_ptr<float>(), out.data_ptr<uint8_t>(), (float)scale, (float)zp, n);
    return out;
}
torch::Tensor dequantize_asym(torch::Tensor input, double scale, double zp) {
    int n = input.numel(); auto out = torch::empty({n}, input.options().dtype(torch::kFloat32));
    int t=256, b=(n+t-1)/t;
    dequantize_asymmetric_kernel<<<b,t>>>(input.data_ptr<uint8_t>(), out.data_ptr<float>(), (float)scale, (float)zp, n);
    return out;
}
std::vector<torch::Tensor> calibrate_min_max(torch::Tensor data, int64_t threads_per_block) {
    int n = data.numel();
    auto scale_out = torch::zeros({1}, data.options()), zp_out = torch::zeros({1}, data.options());
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    calibrate_min_max_kernel<<<blocks, threads_per_block, 2*sizeof(float)>>>(
        data.data_ptr<float>(), scale_out.data_ptr<float>(), zp_out.data_ptr<float>(), n);
    return {scale_out, zp_out};
}
"""

cpp_source = r"""
torch::Tensor quantize_int8(torch::Tensor input, double scale);
torch::Tensor dequantize_int8(torch::Tensor input, double scale);
torch::Tensor quantize_int4(torch::Tensor input, double scale);
torch::Tensor dequantize_int4(torch::Tensor input, int64_t n, double scale);
torch::Tensor quantize_asym(torch::Tensor input, double scale, double zp);
torch::Tensor dequantize_asym(torch::Tensor input, double scale, double zp);
std::vector<torch::Tensor> calibrate_min_max(torch::Tensor data, int64_t threads_per_block);
"""

ch9 = load_inline(
    name="ch9_quant_kernels", cpp_sources=cpp_source, cuda_sources=cuda_source,
    functions=["quantize_int8", "dequantize_int8", "quantize_int4", "dequantize_int4",
               "quantize_asym", "dequantize_asym", "calibrate_min_max"],
    verbose=True,
)
```

**Verifying `Δ²/12` (§9.1) and the ~329× INT4 ratio (§9.2) in one script:**

```python
torch.manual_seed(0)
n = 1_000_000
data = torch.randn(n, device="cuda") * 2.0        # matches this chapter's own mean-0, std-2 test data

scale8 = data.abs().max().item() / 127.0
dq8 = ch9.dequantize_int8(ch9.quantize_int8(data, scale8), scale8)
mse8 = ((data - dq8) ** 2).mean().item()
print(f"INT8:  scale={scale8:.6f}  measured MSE={mse8:.6e}  predicted Δ²/12={scale8**2/12:.6e}")

scale4 = data.abs().max().item() / 7.0
dq4 = ch9.dequantize_int4(ch9.quantize_int4(data, scale4), n, scale4)
mse4 = ((data - dq4) ** 2).mean().item()
print(f"INT4:  scale={scale4:.6f}  measured MSE={mse4:.6e}  predicted Δ²/12={scale4**2/12:.6e}")
print(f"INT4/INT8 MSE ratio: {mse4/mse8:.1f}x  (§9.2 predicted ≈329x)")
```

**Verifying the ~4× symmetric-vs-asymmetric ratio on genuinely non-negative data (§9.3):**

```python
relu_data = torch.relu(torch.randn(n, device="cuda") * 2.0)     # non-negative, like post-ReLU activations

scale_sym = relu_data.abs().max().item() / 127.0
dq_sym = ch9.dequantize_int8(ch9.quantize_int8(relu_data, scale_sym), scale_sym)
mse_sym = ((relu_data - dq_sym) ** 2).mean().item()

r_min, r_max = relu_data.min().item(), relu_data.max().item()
scale_asym = (r_max - r_min) / 255.0
zp_asym = -r_min / scale_asym
dq_asym = ch9.dequantize_asym(ch9.quantize_asym(relu_data, scale_asym, zp_asym), scale_asym, zp_asym)
mse_asym = ((relu_data - dq_asym) ** 2).mean().item()

print(f"symmetric MSE={mse_sym:.6e}  asymmetric MSE={mse_asym:.6e}  ratio={mse_sym/mse_asym:.2f}x  (§9.3 predicted ≈4x)")
```

**Running the multi-block calibration bug live (§9.5):**

```python
data2 = torch.randn(100_000, device="cuda")
true_min, true_max = data2.min().item(), data2.max().item()

scale_1block, _ = ch9.calibrate_min_max(data2, 100_000)   # one block covers everything -- no bug triggered
scale_multi, _  = ch9.calibrate_min_max(data2, 256)        # forces ~400 blocks -- triggers the bug

print(f"true data range: [{true_min:.4f}, {true_max:.4f}]")
print(f"single-block calibration: scale={scale_1block.item():.6f}")
print(f"multi-block  calibration: scale={scale_multi.item():.6f}   <- based on ~256 of 100,000 elements only")
```

`scale_multi` should come out visibly smaller than `scale_1block` — with only 256 of 100,000 elements ever actually considered (whichever ones landed in block 0), the true tail values almost certainly aren't among them, so the computed range understates the real one. This is the self-acknowledged bug from §9.5, triggered on demand rather than taken on faith.

**Testing for the signed-float atomic issue (§9.5's second catch) — honestly, without assuming the outcome:**

```python
data3 = torch.tensor([-5.0, -1.0, 0.5, 2.0], device="cuda")
scale3, zp3 = ch9.calibrate_min_max(data3, 4)     # one block -- isolates this from the multi-block bug above
implied_min = -zp3.item() * scale3.item()
print(f"true min: -5.0    kernel's implied min: {implied_min:.4f}")
```

Unlike the predictions above, this one I genuinely can't tell you the outcome of without running it — whether `atomicMin` on signed-float bit patterns misorders *this specific* set of values depends on the exact bits involved, not just on "there are negative numbers present." If `implied_min` disagrees with `-5.0`, you've directly observed the ordering issue described in §9.5; if this particular array happens not to trigger it, that's consistent with the bug being real but data-dependent — try a few other negative-heavy arrays and see whether one does.

---

## Hands-On Lab

```bash
cd book.cu/7_quant
nvcc -O3 -o fp32_int8 fp32_int8.cu && ./fp32_int8
nvcc -O3 -o fp32_int4 fp32_int4.cu && ./fp32_int4
nvcc -O3 -o dynamic_static dynamic_static.cu && ./dynamic_static
nvcc -O3 -o int8_uint8 int8_uint8.cu && ./int8_uint8
nvcc -O3 -o calibrator calibrator.cu && ./calibrator
nvcc -O3 -o tensorwise tensorwise.cu && ./tensorwise
nvcc -O3 -o groupwise groupwise.cu && ./groupwise
nvcc -O3 -o blockwise blockwise.cu && ./blockwise
nvcc -O3 -o channelwise channelwise.cu && ./channelwise
nvcc -O3 -o awq awq.cu && ./awq
```

1. **Build a comparison table** from every printed MSE/MAE/Max-Error, across all nine core examples. Confirm INT4 (`fp32_int4`) shows meaningfully higher error than INT8 (`fp32_int8`) on the same kind of data — that gap *is* the price of the extra 2× compression.
2. **Stress-test `calibrator.cu`'s multi-block bug directly.** Modify its `SIZE` constant to something large enough to require multiple blocks (e.g. match `fp32_int8.cu`'s `1024*1024`), rebuild, and check whether the printed scale/zero-point still look sane — or whether they're silently wrong because only block 0's data was ever considered.
3. **Construct a deliberately lopsided-range tensor** (e.g., channel 0 drawn from a tiny range, channel 1 from a huge one) and compare `tensorwise.cu` vs. `channelwise.cu`'s reported MSE on it. This is exactly the scenario channel-wise quantization is supposed to fix — you should see a large, real gap.
4. **Run `compute-sanitizer` on `calibrator`**, specifically — atomics, shared memory, and a self-documented "simplification" together are exactly the combination worth double-checking with tooling, not just reading.

## Exercises

1. **Fix `calibrate_min_max` for real.** Implement a proper two-level reduction: have each block write its own partial min/max to a small per-block array (sized `gridDim.x`), then launch a second, small kernel (or do it on the host) to reduce that array into the final global min/max.
2. **Prove the signed-float atomic issue to yourself.** Construct a small test array containing at least one large-magnitude negative value alongside small positive values. Run it through `calibrate_min_max`'s bit-reinterpretation trick by hand (or in code) and compare against the true min. Then explain, using IEEE-754's sign-magnitude bit layout, exactly why the reinterpret-as-int comparison breaks for negative values but not positive ones.
3. **Combine the two axes this chapter kept separate.** Every granularity example (§9.4) quantizes to INT8; every bit-width example (§9.2) uses tensor-wise granularity. Extend `groupwise.cu` (or `blockwise.cu`) to pack its output as INT4, combining §9.2's packing logic with §9.4's per-group scale lookup.
4. **Hand-trace the AWQ toy example.** Using `awq.cu`'s actual input values (`h_x = {0.1, 25.0, -0.2, 0.05, -18.0, 0.3}`), compute all six `h_scales[k]` values by hand (before the averaging-normalization step), and confirm the largest-magnitude activations (25.0, -18.0) get the *smallest* scales. Explain in your own words why that's the correct direction for minimizing overall output error.
5. **Finish the dynamic-quantization pipeline.** `reduce_absmax` produces an atomic-friendly `unsigned int` bit pattern, not a usable float scale. Write the missing conversion step (`__uint_as_float(max_abs_bits) / 127.0f`, on the host or in a tiny follow-up kernel) and wire it into a complete dynamic quantize-then-dequantize call, verified against `dynamic_static.cu`'s static path on the same data.

---

**Next:** Chapter 10 — Profiling, Debugging & Performance Engineering (Part 9). Every optimization since Chapter 6 has been justified by measured numbers — GFLOPS tables, TFLOPS tables, MSE tables. This chapter covers the actual tools (Nsight Systems, Nsight Compute, `cuda-gdb`, `compute-sanitizer`) that produced numbers like these, properly, instead of a hand-rolled timer that might be quietly measuring the wrong thing — exactly as Chapter 7's benchmarking story warned.
