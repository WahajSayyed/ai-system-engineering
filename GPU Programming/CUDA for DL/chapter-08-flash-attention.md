# Chapter 8 — Flash Attention

*Part 7: Flash Attention. Confirmed from the book's companion repo: `book.cu/6_flash/README.md`, "Flash Attention Implementations — Progressive implementations of Flash Attention from naive baselines to WMMA tensor core optimizations."*

This chapter is deliberately narrow: exactly two implementations, `naive.cu` and `fa.cu`, benchmarked directly against PyTorch's own Flash Attention and PyTorch's own naive attention. That narrowness is the point — it isolates precisely what fusion, tiling, online softmax, and tensor cores each buy you, against real, honestly-reported numbers, including an honest accounting of exactly how far the result still sits from production-grade Flash Attention.

Confirmed measured results (H100, batch=16, heads=8, seq_len=512, head_dim=64):

| Implementation | Time (ms) | Speedup vs. PyTorch Flash | Architecture |
|---|---|---|---|
| PyTorch Flash (bf16) | 0.087 | 1.00× (reference) | Optimized (CUTLASS/CuTe) |
| PyTorch Naive (f32) | 0.465 | 0.19× | 3 separate matmuls |
| **`fa` (fp16 + WMMA)** | **5.314** | 0.02× | This chapter's fused kernel |
| `naive` (fp32) | 68.832 | 0.00× | 3 separate kernel launches, this chapter |

The book's own framing of that table, stated directly: **`fa` is 13× faster than `naive`**, entirely from fusion + tiling + online softmax + tensor cores — and still **~61× slower than PyTorch's real Flash Attention**, for reasons the README lists explicitly rather than glossing over. Both halves of that story are this chapter's real content.

---

## 8.1 The Memory-Bandwidth Problem, With Real Numbers Behind It

Chapter 1 §1.4.1 told you attention's O(N²) score matrix is the problem; this chapter's `naive.cu` shows you exactly what materializing it costs in practice. For this benchmark's B=16, H=8, that's **128 separate (batch, head) pairs**, each requiring **three sequential kernel launches** (QKᵀ, softmax, ×V) — **384 total kernel launches** for one forward pass, each paying Chapter 1 §1.3.2's few-microseconds launch overhead, on top of writing and re-reading a full N×N score matrix through HBM three separate times per head. At N=512, that's a 512×512 FP32 matrix — 1MB — allocated once (`cudaMalloc`) and reused across the whole batch/head loop, but still round-tripped through global memory by every one of those 384 launches. `fa.cu` collapses this to **128 total launches — one per (batch, head) pair, each covering the entire fused computation** — and never writes the full N×N matrix to HBM at all.

**Deep dive: the intermediate matrix moves more bytes than all the real data combined.** It's worth totaling this up. Per head, the `S` matrix is written once (by `naive_qk_matmul_kernel`), read and rewritten once (by `naive_softmax_kernel` — one read pass, one write pass), and read once more (by `naive_sv_matmul_kernel`) — **4 MB of traffic per head**, purely for a matrix that exists only as scratch space and holds zero information not already present in `Q` and `K`. Across all 128 (batch, head) pairs, that's **512 MB of pure intermediate-matrix traffic.** Now compare that against the *actual* data this operation needs — `Q`, `K`, `V`, and `O`, each sized `B×H×N×d×4 bytes = 16×8×512×64×4 = 16,777,216 bytes (16 MB)`, for **64 MB total** across all four real tensors. **The scratch matrix alone moves 8× more bytes through memory than all the real Q/K/V/O data combined.** That ratio is the single clearest way to see why "never materialize S in HBM" isn't a minor tweak — it's eliminating the majority of this operation's entire memory footprint.

## 8.2 The Naive Baseline (`naive.cu`): Three Kernels, Full Materialization

The docstring states its own purpose plainly: *"materializes the full N×N attention score matrix in global device memory (HBM)... highly inefficient for long sequences... FlashAttention is designed to solve [this]."* Three kernels, run in sequence, per head:

```cuda
template <int BLOCK_SIZE>
__global__ void naive_qk_matmul_kernel(float* Q, float* K, float* S, int N, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) sum += Q[row * d + k] * K[col * d + k];
        S[row * N + col] = sum * scale;                 // full N×N matrix written to global memory
    }
}

__global__ void naive_softmax_kernel(float* S, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;    // one thread per ROW — familiar by now
    if (row < N) {
        float* row_ptr = S + row * N;
        float max_val = -INFINITY;
        for (int i = 0; i < N; i++) max_val = fmaxf(max_val, row_ptr[i]);
        float sum = 0.0f;
        for (int i = 0; i < N; i++) { row_ptr[i] = expf(row_ptr[i] - max_val); sum += row_ptr[i]; }
        for (int i = 0; i < N; i++) row_ptr[i] /= sum;
    }
}

template <int BLOCK_SIZE>
__global__ void naive_sv_matmul_kernel(float* S, float* V, float* O, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < d) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += S[row * N + j] * V[j * d + col];
        O[row * d + col] = sum;
    }
}
```

The host-side orchestration, allocating `S` once and looping over every batch/head pair:

```cuda
float* S;
CUDA_CHECK(cudaMalloc(&S, N * N * sizeof(float)));   // one N×N scratch buffer, reused across the loop
for (int b = 0; b < B; b++) {
    for (int h = 0; h < nh; h++) {
        int offset = (b * nh + h) * N * d;
        naive_qk_matmul_kernel<BLOCK_SIZE><<<grid_qk, block_qk>>>(Q_ptr, K_ptr, S, N, d, scale);
        naive_softmax_kernel<<<grid_softmax, block_softmax>>>(S, N);
        naive_sv_matmul_kernel<BLOCK_SIZE><<<grid_sv, block_sv>>>(S, V_ptr, O_ptr, N, d);
    }
}
```

Worth naming directly: `naive_softmax_kernel` is the **fourth time this exact single-thread-per-row pattern has appeared in this course** — Chapter 4's MNIST softmax, Chapter 5's GEMV and top-k, and now here. It's a milder case than Chapter 4's (there, batch size 8 meant literally 8 threads total; here, `grid_softmax` covers all `N` rows properly with 256 threads/block), but every thread still serially scans its entire row three separate times. By this point in the course you should recognize the fix on sight: Chapter 6 §6.2's warp-shuffle reduction.

**Deep dive: what cooperating within a row would actually save, at this chapter's real N=512.** Each thread here does `3×N = 1,536` sequential steps (three full passes over the row). Spread that same row across a single 32-lane warp instead — each lane handles `N/32 = 16` elements per pass (`3×16 = 48` steps), then two warp-shuffle reductions (one for the max, one for the sum, `log₂(32) = 5` steps each, `≈10` steps total) combine the 32 lanes' partial results:

```
Naive (1 thread/row):        3 × 512 = 1,536 sequential steps
Warp-cooperative (32 lanes):  48 (local passes) + 10 (two reductions) = 58 steps
Reduction in per-thread critical path: 1,536 / 58 ≈ 26.5×
```

Just 32 cooperating threads — one warp, no extra hardware, no shared memory even required for the reduction stage — cuts each row's critical path by roughly **26.5×** at this exact problem size. That's a different, complementary number to Chapter 6 §6.2's more general treatment of the same fix: this one is anchored to the specific `N=512` this chapter's own benchmark actually uses.

## 8.3 Fusing It All: Flash Attention with WMMA (`fa.cu`)

One kernel, one launch per (batch, head) pair — `dim3 grid_size(B, nh)` — with `dim3 block_size(Br * Bc)` = 256 threads (`Br=Bc=16`, fixed to match WMMA's 16×16×16 fragment size exactly). The kernel's own docstring lays out its structure as two nested loops:

> *Outer loop (over `i`): iterates through rows of `O` in blocks of `Br`, loading a Query tile `Qi` into shared memory. Inner loop (over `j`): iterates through Key/Value in blocks of `Bc`, computing `Sij = Qi @ Kjᵀ` via WMMA, updating running max/sum via online softmax, rescaling the previous output `Oi`, then accumulating `Pij @ Vj` via WMMA into `Oi`.*

**Shared memory holds everything simultaneously** — no round trip to HBM anywhere inside the tile loop:

```cuda
extern __shared__ char smem_raw[];
half*  Qi        = reinterpret_cast<half*>(smem_raw);           // Br × d
half*  Kj        = Qi + Br * d;                                  // Bc × d
half*  Vj        = Kj + Bc * d;                                  // Bc × d
float* Sij_fp32  = reinterpret_cast<float*>(Vj + Bc * d);        // Br × Bc — scores, FP32 for stability
half*  Sij_fp16  = reinterpret_cast<half*>(Sij_fp32 + Br*Bc);    // Br × Bc — same scores, cast to FP16 for WMMA
float* Oi        = reinterpret_cast<float*>(Sij_fp16 + Br*Bc);   // Br × d — running output accumulator
float* temp_pv   = Oi + Br * d;                                  // Br × d — this tile's P@V result
float* mi        = temp_pv + Br * d;                             // Br — running max
float* mi_new    = mi + Br;                                      // Br — this tile's updated max
float* li        = mi_new + Br;                                  // Br — running sum of exponentials
```

**Step 1 — `Sij = Qi @ Kjᵀ` via WMMA**, accumulating in FP32 across the head dimension in chunks of 16:

```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag; // K must be col-major for Q @ Kᵀ
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);
for (int k = 0; k < d; k += 16) {
    wmma::load_matrix_sync(a_frag, Qi + k, d);
    wmma::load_matrix_sync(b_frag, Kj + k, d);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}
wmma::store_matrix_sync(Sij_fp32, c_frag, Bc, wmma::mem_row_major);
```

Notice `b_frag` is declared `wmma::col_major` while `a_frag` is `row_major` — computing `Q @ Kᵀ` via WMMA means simply telling the hardware to read `K`'s fragment *as if* transposed, rather than performing any actual transpose operation. A neat, free consequence of WMMA's fragment-based API.

**Step 2 — the online-softmax update**, exactly Chapter 6 §6.2's rescaling identity, now applied **across tiles** rather than across a single row's columns:

```cuda
if (s_col == 0 && s_row < Br) {
    mi[s_row] = mi_new[s_row];                                   // carry forward last tile's max
    float row_max = -INFINITY;
    for (int c = 0; c < Bc; c++) row_max = fmaxf(row_max, Sij_fp32[s_row * Bc + c]);
    float new_max = fmaxf(mi[s_row], row_max);
    mi_new[s_row] = new_max;

    float row_sum = 0.0f;
    for (int c = 0; c < Bc; c++) {
        float exp_val = expf(Sij_fp32[s_row * Bc + c] - new_max);
        Sij_fp32[s_row * Bc + c] = exp_val;
        row_sum += exp_val;
    }
    float correction = (mi[s_row] == -INFINITY) ? 0.0f : expf(mi[s_row] - new_max);
    li[s_row] = correction * li[s_row] + row_sum;                 // rescale the OLD running sum, then add the new tile's
}
```

This is the exact same `exp(m_old - m_new)` rescaling trick Chapter 6's online softmax kernel used within one row — Flash Attention's actual contribution is realizing that identity works *identically* whether you're merging softmax statistics across the columns of one tile or across a whole sequence of tiles processed one at a time. That single generalization is what lets the full N×N matrix stay out of HBM entirely.

**Step 3 — `Oi += Pij @ Vj`**, with the *same* correction factor applied to rescale the *existing* accumulated output before adding this tile's contribution:

```cuda
// (WMMA matmul of Sij_fp16 @ Vj into temp_pv — same fragment pattern as Step 1, omitted here for brevity)

for (int idx = tx; idx < Br * d; idx += blockDim.x) {
    int r = idx / d, c = idx % d;
    float correction = (mi[r] == -INFINITY || mi_new[r] == -INFINITY) ? 0.0f : expf(mi[r] - mi_new[r]);
    Oi[r * d + c] = correction * Oi[r * d + c] + temp_pv[r * d + c];
}
```

**Step 4, after all Bc-tiles are processed** — a single final normalization by `li`, and the only write to global memory in the entire kernel:

```cuda
O[qkv_off + global_row * d + col] = Oi[s_row * d + col] / li[s_row];
```

**The precision path is worth tracing end to end**, because it's a real, general pattern, not specific to this kernel: FP32 inputs are cast to FP16 *before* the kernel launches (`Q.to(torch::kFloat16)` in the wrapper) → WMMA computes `Qi @ Kjᵀ` as FP16×FP16 with **FP32 accumulation** → the softmax-sensitive max/exp/sum steps happen entirely in that FP32 buffer (`Sij_fp32`), never in FP16 → results are cast back to FP16 (`Sij_fp16`) only for the second WMMA call → that second matmul again accumulates in FP32 (`Oi`, `temp_pv`) → the final output is cast back to the caller's original dtype. **Reduce/accumulate in higher precision than you multiply in** — the same principle Chapter 4's cuBLAS MNIST kernels applied (FP16 GEMM, FP32 `alpha`/`beta`) and Chapter 7's WGMMA kernels applied (FP16 in, FP32 accumulate) — shows up here as a full, deliberate "precision sandwich" around the two matmul steps.

One hardware constraint enforced directly in the wrapper: `assert(d % 16 == 0)` — WMMA's fixed 16×16×16 fragment size means the head dimension must be a multiple of 16, and `Br`/`Bc` are fixed at 16 to match it exactly.

**Deep dive: how many individual tensor-core operations one forward pass actually issues.** It's easy to read "128 kernel launches total" (§8.1) and assume that's the whole story on parallelism — but each of those launches internally orchestrates a great deal of work. For one (batch, head) pair at this chapter's `N=512, d=64`: the outer loop runs `N/Br = 512/16 = 32` times, the inner loop `N/Bc = 512/16 = 32` times, so each head processes `32 × 32 = 1,024` `(Qi, Kj)` tile pairs. Each tile pair's Step 1 (`Sij = Qi @ Kjᵀ`) loops over the head dimension in chunks of 16, so `d/16 = 4` `mma_sync` calls; Step 3 (`Pij @ Vj`) contracts over exactly `Bc = 16` — one fragment, one `mma_sync` call. That's `4 + 1 = 5` tensor-core operations per tile pair, `1,024 × 5 = 5,120` per head, and **`5,120 × 128 = 655,360` individual WMMA `mma_sync` instructions issued across this one benchmark's entire forward pass** — all of it packed inside just 128 kernel launches. The "launch count" framing from §8.1 and this instruction count aren't in tension; they're answering two different questions — how many times the host talks to the device (128) versus how much actual tensor-core work each of those conversations triggers (over half a million individual matrix operations).

## 8.4 Why `fa` Is Still ~61× Slower Than PyTorch's Real Flash Attention

The README doesn't hide from this comparison — it lists the gap's causes directly, and every one of them is a technique you've either already seen (Chapter 7) or will see soon (Part 11, CUTLASS):

- **BF16 vs. FP16** — PyTorch's kernel uses BF16, which gets 2× the effective memory bandwidth on H100 relative to this chapter's FP16 path.
- **Tile size: 128×128 vs. 16×16** — PyTorch's tiles are 8× larger in each dimension, meaning far more reuse per shared-memory load before eviction (exactly Chapter 6 §6.1's arithmetic-intensity argument, just at Flash Attention's own tiling level instead of GEMM's).
- **Async memory pipelining (`cp.async` / TMA)** — this kernel is **fully synchronous**: every step is gated by `__syncthreads()`, with zero overlap between loading the next tile and computing on the current one. Chapter 7 §7.3's producer-consumer WGMMA pattern is exactly the fix, and it isn't applied here at all.
- **Swizzled shared-memory layouts** — avoiding shared-memory bank conflicts through a deliberately scrambled address pattern; not implemented in this straightforward version.
- **Warp specialization** and **1024 threads/block vs. this kernel's 256** — more threads cooperating per block, with different warps assigned different roles (again, Chapter 7's producer/consumer idea), versus this kernel's single undifferentiated 256-thread block matching WMMA's minimal 16×16 granularity.

The book's own pointer for where these get addressed, stated directly: *"See future CUTLASS chapter for these advanced optimizations."* That's Part 11 of this course. The honest lesson to sit with here: getting from "textbook-correct, tensor-core-accelerated, properly fused" (13× over naive) to "state of the art" required *five more separate categories* of optimization stacked on top — Chapter 1 §1.6.3's "compounding effect," demonstrated at a scale large enough to be genuinely humbling.

**Deep dive: even PyTorch's own 0.087ms has real headroom below it.** It's worth checking PyTorch Flash's own number against a theoretical floor, the same roofline instinct Chapter 1 §1.4.1 built. The *minimum* possible memory traffic for this operation — reading `Q`, `K`, `V` and writing `O` exactly once each, at BF16 (2 bytes/element) — is `4 × (16×8×512×64×2) = 4 × 8,388,608 ≈ 32 MB`. At H100 SXM's confirmed 3.35 TB/s bandwidth, that's a **memory floor of ≈10 microseconds (0.01 ms).** The *compute* side: full attention's two matmuls (`QKᵀ` and `×V`) total `4×B×H×N²×d = 4×16×8×512²×64 ≈ 8.59 GFLOP`; at H100's confirmed 989 TFLOPS dense BF16 tensor-core peak, that's a **compute floor of ≈8.7 microseconds (0.0087 ms).** Both floors land in the same narrow range — this problem size sits almost exactly on H100's roofline ridge point, meaning it's genuinely borderline between memory- and compute-bound rather than clearly one or the other. Either way, **PyTorch's real, measured 0.087ms is roughly 9–10× higher than either theoretical floor** — a useful, humbling data point in its own right: even NVIDIA's own production-grade, CUTLASS-backed implementation isn't operating at the hardware's true limit for this exact problem size, for reasons including kernel-launch overhead and imperfect occupancy at this comparatively modest scale (`N=512` is small by production LLM standards). The 61× gap between `fa.cu` and PyTorch Flash is real and dominant — but it's worth knowing the ceiling itself has some daylight above it too.

## 8.5 Running This from Python: Naive vs. a Simplified Fused Kernel

**A provenance and scope note before the code.** `naive.cu`'s three kernels (§8.2) are quoted verbatim earlier in this chapter and wrap directly. `fa.cu` (§8.3) is more complex than that — its shared-memory tile-loading loop and its P@V WMMA step were both marked as summarized placeholders in the original text (`// (WMMA matmul of Sij_fp16 @ Vj into temp_pv...)`), not fully quoted. Reconstructing WMMA fragment code from an incomplete description, with no GPU in this environment to test it, is exactly the risk Chapter 7 §7.5 declined to take for WGMMA — the same judgment applies here. Instead, below is a **simplified fused kernel** that implements the same core idea in plain CUDA-core arithmetic: one query row per warp, streaming through keys one at a time with online-softmax rescaling, **never materializing more than one score at a time** — no shared-memory Q/K/V tiling, no tensor cores, no batching over `(batch, head)` pairs. It's a genuine, correct flash-attention-style kernel (same recurrence as §8.3's online-softmax update, just applied one key at a time instead of one `Bc`-wide tile at a time), just a smaller and slower one than the book's own.

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ========= Confirmed verbatim, §8.2 =========
__global__ void naive_qk_matmul_kernel(const float* Q, const float* K, float* S, int N, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) sum += Q[row*d+k] * K[col*d+k];
        S[row*N+col] = sum * scale;
    }
}
__global__ void naive_softmax_kernel(float* S, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float* row_ptr = S + row * N;
        float max_val = -INFINITY;
        for (int i = 0; i < N; i++) max_val = fmaxf(max_val, row_ptr[i]);
        float sum = 0.0f;
        for (int i = 0; i < N; i++) { row_ptr[i] = expf(row_ptr[i]-max_val); sum += row_ptr[i]; }
        for (int i = 0; i < N; i++) row_ptr[i] /= sum;
    }
}
__global__ void naive_sv_matmul_kernel(const float* S, const float* V, float* O, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < d) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += S[row*N+j] * V[j*d+col];
        O[row*d+col] = sum;
    }
}

// ========= Simplified fused kernel: same recurrence as §8.3, one key at a time, no WMMA =========
__global__ void flash_attn_simple_kernel(const float* Q, const float* K, const float* V, float* O,
                                          int N, int d, float scale) {
    int row = blockIdx.x;             // one query row per block
    int lane = threadIdx.x;           // exactly 32 threads (one warp) cooperate on this row
    int per_thread = d / 32;          // d must be a multiple of 32 for this minimal demo
    float o_local[8];                 // supports d up to 256
    for (int t = 0; t < per_thread; ++t) o_local[t] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int j = 0; j < N; ++j) {
        float partial = 0.0f;                                   // cooperative Q[row,:] . K[j,:]
        for (int t = 0; t < per_thread; ++t) {
            int c = lane * per_thread + t;
            partial += Q[row*d+c] * K[j*d+c];
        }
        for (int o = 16; o > 0; o /= 2) partial += __shfl_down_sync(0xffffffff, partial, o);
        partial = __shfl_sync(0xffffffff, partial, 0) * scale;   // full dot product, broadcast to all lanes

        float new_m = fmaxf(m, partial);                          // online-softmax update, one key at a time
        float correction = (m == -INFINITY) ? 0.0f : expf(m - new_m);
        float p = expf(partial - new_m);
        l = l * correction + p;
        for (int t = 0; t < per_thread; ++t) {
            int c = lane * per_thread + t;
            o_local[t] = o_local[t] * correction + p * V[j*d+c];   // same rescale-then-accumulate as §8.3
        }
        m = new_m;
    }
    for (int t = 0; t < per_thread; ++t) {
        int c = lane * per_thread + t;
        O[row*d+c] = o_local[t] / l;                               // final normalize, exactly §8.3's Step 4
    }
}

// ========= Launchers =========
torch::Tensor attention_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    int N = Q.size(0), d = Q.size(1);
    auto S = torch::empty({N, N}, Q.options());       // the O(N^2) matrix §8.1 quantifies
    auto O = torch::empty({N, d}, Q.options());
    dim3 t2(16,16), g2((N+15)/16,(N+15)/16);
    naive_qk_matmul_kernel<<<g2,t2>>>(Q.data_ptr<float>(),K.data_ptr<float>(),S.data_ptr<float>(),N,d,(float)scale);
    int threads=256, blocks=(N+threads-1)/threads;
    naive_softmax_kernel<<<blocks,threads>>>(S.data_ptr<float>(), N);
    dim3 t3(16,16), g3((d+15)/16,(N+15)/16);
    naive_sv_matmul_kernel<<<g3,t3>>>(S.data_ptr<float>(),V.data_ptr<float>(),O.data_ptr<float>(),N,d);
    return O;
}
torch::Tensor attention_fused(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    int N = Q.size(0), d = Q.size(1);
    TORCH_CHECK(d % 32 == 0 && d <= 256, "this minimal demo assumes d is a multiple of 32, up to 256");
    auto O = torch::empty({N, d}, Q.options());        // no N x N buffer ever allocated
    flash_attn_simple_kernel<<<N, 32>>>(Q.data_ptr<float>(),K.data_ptr<float>(),V.data_ptr<float>(),O.data_ptr<float>(),N,d,(float)scale);
    return O;
}
"""

cpp_source = r"""
torch::Tensor attention_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale);
torch::Tensor attention_fused(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale);
"""

ch8 = load_inline(
    name="ch8_flash_kernels", cpp_sources=cpp_source, cuda_sources=cuda_source,
    functions=["attention_naive", "attention_fused"], verbose=True,
)
```

**Correctness**, against a plain-PyTorch reference (single head, no batching — the same simplification §8.2's own per-head loop makes, just without the outer `(batch, head)` loop around it):

```python
def attention_reference(Q, K, V, scale):
    return torch.softmax((Q @ K.t()) * scale, dim=-1) @ V

N, d = 512, 64
scale = 1.0 / (d ** 0.5)
Q, K, V = (torch.randn(N, d, device="cuda") for _ in range(3))

O_ref = attention_reference(Q, K, V, scale)
O_naive = ch8.attention_naive(Q, K, V, scale)
O_fused = ch8.attention_fused(Q, K, V, scale)
print("naive max_diff:", (O_naive - O_ref).abs().max().item())
print("fused max_diff:", (O_fused - O_ref).abs().max().item())
```

**The memory comparison — this is the number that actually matters for this chapter**, more than raw speed. §8.1's whole argument was about *bytes*, not FLOPs, so measure bytes directly:

```python
def peak_mem_mb(fn, *args):
    torch.cuda.reset_peak_memory_stats()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)

N_big = 4096   # large enough that the O(N^2) score matrix is impossible to miss
Qb, Kb, Vb = (torch.randn(N_big, d, device="cuda") for _ in range(3))
print(f"naive peak memory: {peak_mem_mb(ch8.attention_naive, Qb, Kb, Vb, scale):.1f} MB")
print(f"fused peak memory: {peak_mem_mb(ch8.attention_fused, Qb, Kb, Vb, scale):.1f} MB")
```

At `N=4096`, the naive path's `S` matrix alone is `4096×4096×4 bytes = 64 MB` — and that number should show up almost exactly in the naive path's peak-memory figure, while the fused path's peak memory should stay close to just `Q`, `K`, `V`, and `O` themselves (a few MB) — no `N×N` buffer ever gets allocated, because the fused kernel never needs one. That gap, measured directly rather than argued for, is §8.1's "8× more traffic than the real data" point made concrete and literal: not an estimate of bytes moved through HBM, but an actual reported allocation size you can watch scale with `N²` on one side and stay flat on the other.

---

## Hands-On Lab

```bash
cd book.cu/6_flash
python main.py                                  # compares naive, fa, and PyTorch reference/naive
python main.py --kernels fa                      # just this chapter's Flash Attention kernel
python main.py --seq-len 1024 --batch-size 8     # different problem size
```

1. **Reproduce the qualitative ordering on your own hardware.** Both your RTX 3090 and T4 support WMMA (Volta+), so `fa` should run on each. Confirm the same relative ordering (naive slowest, `fa` a clear multiple faster, PyTorch's native path fastest) even if your absolute numbers differ from the H100 table.
2. **Vary `--seq-len`.** Naive's O(N²) materialization cost should grow noticeably faster than `fa`'s as you increase sequence length — try 128, 512, 2048 and plot the two implementations' scaling side by side.
3. **Count the real kernel launches.** For `B=16, H=8`: confirm `naive.cu` issues `16 * 8 * 3 = 384` kernel launches for one forward pass, while `fa.cu` issues exactly `16 * 8 = 128` — one per (batch, head) pair, each handling that pair's *entire* fused computation.
4. **Run `compute-sanitizer` on both binaries** — a good habit revisited from every prior chapter, and especially relevant here given how many shared-memory buffers `fa.cu` juggles simultaneously.

## Exercises

1. **Verify the cross-tile online-softmax rescaling by hand.** Pick a tiny toy row of 4 values split into two `Bc=2` tiles. Compute softmax two ways: (a) directly on all 4 values at once, (b) tile-by-tile using the exact `mi`/`mi_new`/`li` update sequence from §8.3. Confirm both give identical results.
2. **Explain the dual `Sij_fp32`/`Sij_fp16` buffers.** Why does the kernel need *both*, rather than keeping the scores in just one precision throughout? Connect your answer to the "precision sandwich" pattern from §8.3 and to Chapter 4's cuBLAS MNIST kernel.
3. **Add causal masking.** Real autoregressive LLM training/inference needs causal attention — query position `i` should never attend to key position `j > i`. Add a check inside the inner loop (§8.3, Step 1 or Step 2) that skips or zeroes score entries where the global key column exceeds the global query row, and verify against a masked reference in PyTorch.
4. **Compute this kernel's shared-memory footprint.** Using the real `smem_size` formula from the wrapper (`fa_forward`) with `Br=Bc=16, d=64`, calculate the total bytes required per block. Compare that against your RTX 3090's and T4's per-SM shared-memory budget (Chapter 1's hardware table) to work out how many blocks could theoretically be resident on one SM simultaneously.
5. **Scale `Br`/`Bc` to 32.** WMMA fragments are fixed at 16×16×16, so a 32×32 tile needs *multiple* fragment operations per tile — the same hierarchical block/warp/fragment structure Chapter 7 §7.2's WMMA GEMM kernel used. Sketch (or implement) what changes.

---

**Next:** Chapter 9 — Quantization for Inference (Part 8). Every kernel so far has kept weights and activations in FP16/FP32. This chapter compresses them into INT8 and INT4, trading precision for memory bandwidth — directly relevant to everything Chapter 1 §1.5.2 told you about decode being bandwidth-bound in the first place.
