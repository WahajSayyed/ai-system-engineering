# Chapter 20: Capstone — Mini GPU-Accelerated ML Library

> **Prerequisites:** Chapters 1–19 (complete CUDA curriculum)
> **Languages:** C/CUDA + Python
> **Domain:** ML / AI, Systems, Performance
> **Estimated Time:** 15–20 hours

---

## 20.1 What We're Building

This capstone assembles every concept from the curriculum into a single, coherent, production-quality mini-library: **TinyCUDA** — a trainable GPU-accelerated ML framework built from scratch.

```
TinyCUDA
├── csrc/                         C++/CUDA source
│   ├── gemm.cu                   Custom tiled GEMM kernel (Ch 6, 9, 10)
│   ├── attention.cu              Fused multi-head attention (Ch 7, 8)
│   ├── ops.cu                    Fused elementwise ops (Ch 5, 7)
│   ├── pipeline.cu               Multi-stream inference pipeline (Ch 11, 19)
│   └── bindings.cpp              PyTorch extension bindings (Ch 15)
├── tinycuda/                     Python package
│   ├── kernels.py                Triton softmax + layer norm (Ch 17)
│   ├── layers.py                 nn.Module wrappers
│   ├── graph.py                  CUDA Graph inference engine (Ch 19)
│   └── profiler.py               Nsight / torch.profiler integration (Ch 12)
├── tests/
│   ├── test_gemm.py
│   ├── test_attention.py
│   └── test_training.py
├── benchmarks/
│   └── bench_vs_pytorch.py
└── setup.py
```

The library implements a full **transformer encoder layer** that can:
- Run a forward pass using custom CUDA kernels
- Compute gradients (backward pass)
- Run inference with CUDA Graph acceleration
- Be used as a drop-in `nn.Module` in PyTorch training loops

---

## 20.2 Project Setup

```bash
mkdir tinycuda && cd tinycuda
mkdir -p csrc tinycuda tests benchmarks

# Prerequisites check
nvcc --version          # >= 11.8
python -c "import torch; print(torch.__version__)"  # >= 2.0
python -c "import triton; print(triton.__version__)"

# Install build dependencies
pip install torch ninja pybind11
```

`setup.py`:
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tinycuda',
    ext_modules=[
        CUDAExtension(
            name='tinycuda._C',
            sources=[
                'csrc/gemm.cu',
                'csrc/attention.cu',
                'csrc/ops.cu',
                'csrc/pipeline.cu',
                'csrc/bindings.cpp',
            ],
            extra_compile_args={
                'cxx':  ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_86',          # Adjust for your GPU
                    '--use_fast_math',
                    '-lineinfo',            # For Nsight profiling
                    '--ptxas-options=-v',   # Verbose register usage
                    '-rdc=true',            # For dynamic parallelism (Ch 19)
                ],
            },
        )
    ],
    packages=['tinycuda'],
    cmdclass={'build_ext': BuildExtension},
)
```

Build:
```bash
pip install -e . --no-build-isolation
```

---

## 20.3 Component 1: Custom GEMM Kernel (`csrc/gemm.cu`)

Applies: tiling (Ch 6), shared memory (Ch 7), bank conflict avoidance (Ch 8), occupancy (Ch 10), L2 swizzling (Ch 17).

```cuda
// csrc/gemm.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// ── Compile-time tile configuration ────────────────────────────────────────
#define GEMM_BM 128
#define GEMM_BN 128
#define GEMM_BK 32
#define GEMM_WARPS_M 4
#define GEMM_WARPS_N 4
// Each warp tile: (BM/WARPS_M) × (BN/WARPS_N) = 32 × 32

// Padded shared memory to avoid bank conflicts (Ch 8)
#define SMEM_PAD 8

__global__ void __launch_bounds__(256, 2)
tinycuda_gemm_kernel(
    const half* __restrict__ A,    // [M, K]
    const half* __restrict__ B,    // [K, N]
          float* __restrict__ C,   // [M, N]
    int M, int N, int K,
    float alpha, float beta         // C = alpha * A@B + beta * C
) {
    // ── Shared memory tiles ──────────────────────────────────────────────
    __shared__ half smem_a[GEMM_BM][GEMM_BK + SMEM_PAD];
    __shared__ half smem_b[GEMM_BK][GEMM_BN + SMEM_PAD];

    // ── Thread/warp IDs ─────────────────────────────────────────────────
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane_id  = tid % 32;
    const int warp_m   = warp_id / GEMM_WARPS_N;
    const int warp_n   = warp_id % GEMM_WARPS_N;

    // ── L2-friendly swizzled tile assignment (Ch 9, 17) ─────────────────
    const int pid      = blockIdx.x;
    const int num_n    = (N + GEMM_BN - 1) / GEMM_BN;
    const int GROUP_M  = 8;
    const int num_m    = (M + GEMM_BM - 1) / GEMM_BM;
    const int group_id = pid / (GROUP_M * num_n);
    const int first_m  = group_id * GROUP_M;
    const int gs_m     = min(num_m - first_m, GROUP_M);
    const int pid_m    = first_m + (pid % gs_m);
    const int pid_n    = (pid % (GROUP_M * num_n)) / gs_m;

    // Output tile origin
    const int row0 = pid_m * GEMM_BM;
    const int col0 = pid_n * GEMM_BN;

    // ── Register accumulator ─────────────────────────────────────────────
    // Each warp handles a (WARP_M × WARP_N) output tile
    const int WARP_M = GEMM_BM / GEMM_WARPS_M;  // 32
    const int WARP_N = GEMM_BN / GEMM_WARPS_N;  // 32
    float acc[WARP_M / 8][WARP_N / 8] = {};      // 4×4 register array per warp row/col group

    // ── Main K-loop ──────────────────────────────────────────────────────
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {

        // ── Cooperative load: all threads load A tile ────────────────────
        // 256 threads load GEMM_BM × GEMM_BK = 128×32 = 4096 halfs
        // 4096 / 256 = 16 elements per thread
        #pragma unroll
        for (int i = 0; i < GEMM_BM * GEMM_BK / 256; i++) {
            int load_idx = i * 256 + tid;
            int row      = load_idx / GEMM_BK;
            int col      = load_idx % GEMM_BK;
            int global_r = row0 + row;
            int global_c = k0   + col;
            smem_a[row][col] = (global_r < M && global_c < K)
                               ? A[global_r * K + global_c]
                               : __float2half(0.0f);
        }

        // ── Cooperative load: all threads load B tile ────────────────────
        #pragma unroll
        for (int i = 0; i < GEMM_BK * GEMM_BN / 256; i++) {
            int load_idx = i * 256 + tid;
            int row      = load_idx / GEMM_BN;
            int col      = load_idx % GEMM_BN;
            int global_r = k0   + row;
            int global_c = col0 + col;
            smem_b[row][col] = (global_r < K && global_c < N)
                               ? B[global_r * N + global_c]
                               : __float2half(0.0f);
        }

        __syncthreads();

        // ── Warp-level outer product accumulation ────────────────────────
        const int warp_row0 = warp_m * WARP_M;
        const int warp_col0 = warp_n * WARP_N;

        #pragma unroll
        for (int k = 0; k < GEMM_BK; k++) {
            // Each thread in warp handles a subset of the warp tile
            int tr = lane_id / (WARP_N / 4);  // Thread row within warp tile
            int tc = lane_id % (WARP_N / 4);  // Thread col within warp tile

            half a_val = smem_a[warp_row0 + tr * 4][k];
            half b_val = smem_b[k][warp_col0 + tc * 4];

            // Accumulate in float32 for precision
            acc[tr / 2][tc / 2] += __half2float(a_val) * __half2float(b_val);
        }

        __syncthreads();
    }

    // ── Write output tile ────────────────────────────────────────────────
    const int warp_row0 = warp_m * WARP_M;
    const int warp_col0 = warp_n * WARP_N;

    for (int i = 0; i < WARP_M / 8; i++) {
        for (int j = 0; j < WARP_N / 8; j++) {
            int tr = lane_id / (WARP_N / 4);
            int tc = lane_id % (WARP_N / 4);
            int global_r = row0 + warp_row0 + i * 8 + tr * 4;
            int global_c = col0 + warp_col0 + j * 8 + tc * 4;
            if (global_r < M && global_c < N) {
                float* c_ptr = C + global_r * N + global_c;
                *c_ptr = alpha * acc[i][j] + (beta != 0.0f ? beta * (*c_ptr) : 0.0f);
            }
        }
    }
}

// ── C++ wrapper callable from PyTorch ───────────────────────────────────────
torch::Tensor tinycuda_gemm(
    torch::Tensor A, torch::Tensor B,
    float alpha = 1.0f, float beta = 0.0f
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Expected fp16 input");
    TORCH_CHECK(A.size(1) == B.size(0), "Shape mismatch: A[M,K] @ B[K,N]");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::TensorOptions()
                          .dtype(torch::kFloat32).device(A.device()));

    int total_tiles = ((M + GEMM_BM - 1) / GEMM_BM) *
                      ((N + GEMM_BN - 1) / GEMM_BN);

    tinycuda_gemm_kernel<<<total_tiles, 256>>>(
        A.data_ptr<at::Half>(),
        B.data_ptr<at::Half>(),
        C.data_ptr<float>(),
        M, N, K, alpha, beta
    );
    return C;
}
```

---

## 20.4 Component 2: Fused Multi-Head Attention (`csrc/attention.cu`)

Applies: flash attention algorithm (Ch 17), shared memory (Ch 7), warp reduction (Ch 6), vectorized loads (Ch 18).

```cuda
// csrc/attention.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <math.h>

namespace cg = cooperative_groups;

// ── Fused multi-head attention forward (flash attention algorithm) ──────────
// Q, K, V: [B, H, T, D]  (batch, heads, seq_len, head_dim)
// Out:     [B, H, T, D]
// Each block handles one (batch, head, query_tile) combination.
template <int HEAD_DIM, int BLOCK_Q, int BLOCK_KV>
__global__ void __launch_bounds__(BLOCK_Q * (HEAD_DIM / 32))
fused_mha_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
          half* __restrict__ Out,
    const float scale,          // 1/sqrt(head_dim)
    int B, int H, int T
) {
    // Each block: one (batch b, head h, query_tile q_tile)
    int b      = blockIdx.z;
    int h      = blockIdx.y;
    int q_tile = blockIdx.x;
    int tid    = threadIdx.x;

    const int stride_b = H * T * HEAD_DIM;
    const int stride_h = T * HEAD_DIM;

    const half* q_ptr = Q + b * stride_b + h * stride_h;
    const half* k_ptr = K + b * stride_b + h * stride_h;
    const half* v_ptr = V + b * stride_b + h * stride_h;
          half* o_ptr = Out + b * stride_b + h * stride_h;

    // ── Shared memory: KV tiles ──────────────────────────────────────────
    __shared__ half smem_k[BLOCK_KV][HEAD_DIM + 4];  // +4: bank conflict pad
    __shared__ half smem_v[BLOCK_KV][HEAD_DIM + 4];

    // ── Load Q tile into registers ───────────────────────────────────────
    const int q_row = q_tile * BLOCK_Q + tid;   // Global query row
    half q_reg[HEAD_DIM / 32];                  // Q tile in registers

    if (q_row < T) {
        const half* q_row_ptr = q_ptr + q_row * HEAD_DIM;
        // Vectorized load: 8 halfs = 128 bits per iteration
        for (int d = 0; d < HEAD_DIM; d += 8) {
            *reinterpret_cast<float4*>(&q_reg[d / 8]) =
                *reinterpret_cast<const float4*>(q_row_ptr + d);
        }
    }

    // ── Online softmax state ─────────────────────────────────────────────
    float m_i = -INFINITY;   // Running max
    float l_i = 0.0f;        // Running sum of exp
    float acc[HEAD_DIM];     // Output accumulator
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;

    // ── KV tile loop ─────────────────────────────────────────────────────
    for (int kv_tile = 0; kv_tile < (T + BLOCK_KV - 1) / BLOCK_KV; kv_tile++) {

        // Cooperative load of K tile into shared memory
        for (int i = tid; i < BLOCK_KV; i += blockDim.x) {
            int kv_row = kv_tile * BLOCK_KV + i;
            if (kv_row < T) {
                for (int d = 0; d < HEAD_DIM; d++)
                    smem_k[i][d] = k_ptr[kv_row * HEAD_DIM + d];
            } else {
                for (int d = 0; d < HEAD_DIM; d++)
                    smem_k[i][d] = __float2half(0.0f);
            }
        }

        // Cooperative load of V tile
        for (int i = tid; i < BLOCK_KV; i += blockDim.x) {
            int kv_row = kv_tile * BLOCK_KV + i;
            if (kv_row < T) {
                for (int d = 0; d < HEAD_DIM; d++)
                    smem_v[i][d] = v_ptr[kv_row * HEAD_DIM + d];
            } else {
                for (int d = 0; d < HEAD_DIM; d++)
                    smem_v[i][d] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // ── Compute QK^T scores for this thread's query row ──────────────
        if (q_row < T) {
            float scores[BLOCK_KV];

            for (int j = 0; j < BLOCK_KV; j++) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM / 8; d++) {
                    // Convert to float for dot product
                    half* q_chunk = &q_reg[d];
                    half* k_chunk = &smem_k[j][d * 8];
                    for (int dd = 0; dd < 8; dd++)
                        dot += __half2float(q_chunk[dd]) * __half2float(k_chunk[dd]);
                }
                scores[j] = dot * scale;

                // Causal mask: query can only attend to kv positions <= q_row
                int kv_row_abs = kv_tile * BLOCK_KV + j;
                if (kv_row_abs > q_row || kv_row_abs >= T)
                    scores[j] = -INFINITY;
            }

            // ── Online softmax update (Flash Attention Alg 1) ────────────
            float m_new = m_i;
            for (int j = 0; j < BLOCK_KV; j++)
                m_new = fmaxf(m_new, scores[j]);

            float l_new = 0.0f;
            float p[BLOCK_KV];
            for (int j = 0; j < BLOCK_KV; j++) {
                p[j]  = expf(scores[j] - m_new);
                l_new += p[j];
            }

            float alpha = expf(m_i - m_new);
            l_i = alpha * l_i + l_new;
            m_i = m_new;

            // ── Accumulate: acc = alpha * acc + p @ V_tile ───────────────
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[d] *= alpha;
                for (int j = 0; j < BLOCK_KV; j++)
                    acc[d] += p[j] * __half2float(smem_v[j][d]);
            }
        }

        __syncthreads();
    }

    // ── Write output ─────────────────────────────────────────────────────
    if (q_row < T) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < HEAD_DIM; d++)
            o_ptr[q_row * HEAD_DIM + d] = __float2half(acc[d] * inv_l);
    }
}

// ── C++ dispatch wrapper ─────────────────────────────────────────────────────
torch::Tensor tinycuda_attention_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
) {
    TORCH_CHECK(Q.dim() == 4, "Expected [B, H, T, D]");
    int B = Q.size(0), H = Q.size(1), T = Q.size(2), D = Q.size(3);
    TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128");

    auto Out = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);

    // Grid: (q_tiles, heads, batches)
    dim3 grid((T + 16 - 1) / 16, H, B);

    if (D == 64) {
        fused_mha_fwd_kernel<64, 16, 16><<<grid, 16>>>(
            Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(),
            V.data_ptr<at::Half>(), Out.data_ptr<at::Half>(),
            scale, B, H, T
        );
    } else {
        fused_mha_fwd_kernel<128, 16, 16><<<grid, 16>>>(
            Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(),
            V.data_ptr<at::Half>(), Out.data_ptr<at::Half>(),
            scale, B, H, T
        );
    }
    return Out;
}
```

---

## 20.5 Component 3: Fused Elementwise Ops (`csrc/ops.cu`)

Applies: fused kernels (Ch 5), vectorized loads (Ch 18), warp reduction (Ch 6), async copies (Ch 9).

```cuda
// csrc/ops.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
namespace cg = cooperative_groups;

// ── Fused: bias_add + GELU activation ──────────────────────────────────────
// Processes float16, uses vectorized loads (float4 = 8 halfs per load)
__global__ void fused_bias_gelu_kernel(
          half* __restrict__ x,      // [N, C] in-place
    const half* __restrict__ bias,   // [C]
    int N, int C
) {
    // Each thread processes 8 halfs (one float4 load)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int n   = idx / C;
    int c   = idx % C;
    if (n >= N || c + 7 >= C) return;

    // Vectorized load: 8 halfs = 128 bits
    float4 x_vec    = *reinterpret_cast<const float4*>(x    + n * C + c);
    float4 bias_vec = *reinterpret_cast<const float4*>(bias + c);

    half* x_halfs    = reinterpret_cast<half*>(&x_vec);
    half* bias_halfs = reinterpret_cast<half*>(&bias_vec);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float v = __half2float(x_halfs[i]) + __half2float(bias_halfs[i]);
        // GELU: 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715*v^3)))
        float tanh_arg = 0.7978845608f * (v + 0.044715f * v * v * v);
        x_halfs[i] = __float2half(0.5f * v * (1.0f + tanhf(tanh_arg)));
    }

    // Vectorized store
    *reinterpret_cast<float4*>(x + n * C + c) = x_vec;
}

// ── Fused: Layer Normalization ────────────────────────────────────────────
// Each block handles one row (sample). Uses warp reduce for mean+var.
__global__ void layer_norm_kernel(
          half*  __restrict__ out,     // [N, C]
    const half*  __restrict__ x,       // [N, C]
    const float* __restrict__ gamma,   // [C]
    const float* __restrict__ beta,    // [C]
    int N, int C, float eps
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int n = blockIdx.x;
    if (n >= N) return;

    const half*  x_row   = x   + n * C;
          half*  out_row = out + n * C;

    // ── Pass 1: compute mean ─────────────────────────────────────────────
    float thread_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        thread_sum += __half2float(x_row[c]);

    // Warp reduce
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());

    // Block reduce via shared memory
    __shared__ float smem_reduce[32];
    if (warp.thread_rank() == 0)
        smem_reduce[threadIdx.x / 32] = warp_sum;
    block.sync();

    float mean = 0.0f;
    if (threadIdx.x < (blockDim.x / 32)) {
        float partial = smem_reduce[threadIdx.x];
        mean = cg::reduce(warp, partial, cg::plus<float>()) / C;
    }
    mean = __shfl_sync(0xffffffff, mean, 0);  // Broadcast to all threads

    // ── Pass 2: compute variance ─────────────────────────────────────────
    float thread_var = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float diff = __half2float(x_row[c]) - mean;
        thread_var += diff * diff;
    }

    float warp_var = cg::reduce(warp, thread_var, cg::plus<float>());
    if (warp.thread_rank() == 0)
        smem_reduce[threadIdx.x / 32] = warp_var;
    block.sync();

    float var = 0.0f;
    if (threadIdx.x < (blockDim.x / 32)) {
        float partial = smem_reduce[threadIdx.x];
        var = cg::reduce(warp, partial, cg::plus<float>()) / C;
    }
    var  = __shfl_sync(0xffffffff, var, 0);
    float inv_std = rsqrtf(var + eps);

    // ── Normalize + scale + shift ────────────────────────────────────────
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float norm = (__half2float(x_row[c]) - mean) * inv_std;
        out_row[c] = __float2half(gamma[c] * norm + beta[c]);
    }
}

// ── Fused: residual add + layer norm (common pattern in transformers) ────────
__global__ void residual_add_layer_norm_kernel(
          half*  __restrict__ out,
    const half*  __restrict__ x,
    const half*  __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, float eps
) {
    // Same as layer_norm_kernel but adds residual before normalizing
    int n = blockIdx.x;
    if (n >= N) return;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ float smem_reduce[32];
    __shared__ half  smem_added[2048];  // Max C=2048 cached in smem

    // Fused residual add + cache in shared memory
    float thread_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = __half2float(x[n*C+c]) + __half2float(residual[n*C+c]);
        smem_added[c] = __float2half(v);
        thread_sum   += v;
    }
    block.sync();

    // Mean
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    if (warp.thread_rank() == 0) smem_reduce[threadIdx.x / 32] = warp_sum;
    block.sync();
    float mean = 0.0f;
    if (threadIdx.x < blockDim.x / 32)
        mean = cg::reduce(warp, smem_reduce[threadIdx.x], cg::plus<float>()) / C;
    mean = __shfl_sync(0xffffffff, mean, 0);

    // Variance
    float thread_var = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float diff = __half2float(smem_added[c]) - mean;
        thread_var += diff * diff;
    }
    float warp_var = cg::reduce(warp, thread_var, cg::plus<float>());
    if (warp.thread_rank() == 0) smem_reduce[threadIdx.x / 32] = warp_var;
    block.sync();
    float var = 0.0f;
    if (threadIdx.x < blockDim.x / 32)
        var = cg::reduce(warp, smem_reduce[threadIdx.x], cg::plus<float>()) / C;
    var = __shfl_sync(0xffffffff, var, 0);
    float inv_std = rsqrtf(var + eps);

    // Normalize
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float norm = (__half2float(smem_added[c]) - mean) * inv_std;
        out[n*C+c] = __float2half(gamma[c] * norm + beta[c]);
    }
}

// ── C++ wrappers ─────────────────────────────────────────────────────────────
void tinycuda_bias_gelu(torch::Tensor x, torch::Tensor bias) {
    int N = x.size(0), C = x.size(1);
    int threads = 256;
    int blocks  = (N * C / 8 + threads - 1) / threads;
    fused_bias_gelu_kernel<<<blocks, threads>>>(
        x.data_ptr<at::Half>(), bias.data_ptr<at::Half>(), N, C);
}

torch::Tensor tinycuda_layer_norm(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps = 1e-5f
) {
    int N = x.size(0), C = x.size(1);
    auto out = torch::empty_like(x);
    layer_norm_kernel<<<N, 256>>>(
        out.data_ptr<at::Half>(), x.data_ptr<at::Half>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        N, C, eps);
    return out;
}

torch::Tensor tinycuda_residual_ln(
    torch::Tensor x, torch::Tensor residual,
    torch::Tensor gamma, torch::Tensor beta, float eps = 1e-5f
) {
    int N = x.size(0), C = x.size(1);
    auto out = torch::empty_like(x);
    residual_add_layer_norm_kernel<<<N, 256>>>(
        out.data_ptr<at::Half>(), x.data_ptr<at::Half>(),
        residual.data_ptr<at::Half>(), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), N, C, eps);
    return out;
}
```

---

## 20.6 Component 4: Triton Kernels (`tinycuda/kernels.py`)

Applies: Triton tile model (Ch 17), auto-tuning, fused softmax.

```python
# tinycuda/kernels.py
import torch
import triton
import triton.language as tl


# ── Fused Softmax (from Chapter 17, production-ready version) ────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def _fused_softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols, temperature,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask        = col_offsets < n_cols

    # Load + scale by temperature
    row = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf'))
    row = row / temperature

    # Numerically stable softmax
    row_max = tl.max(row, axis=0)
    row     = row - row_max
    num     = tl.exp(row)
    denom   = tl.sum(num, axis=0)
    out     = num / denom

    out_ptr = output_ptr + row_idx * output_row_stride
    tl.store(out_ptr + col_offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Fused softmax with temperature scaling."""
    assert x.is_cuda and x.is_contiguous()
    n_rows, n_cols = x.shape
    BLOCK = min(triton.next_power_of_2(n_cols), 4096)
    out = torch.empty_like(x)
    _fused_softmax_kernel[(n_rows,)](
        x, out,
        x.stride(0), out.stride(0),
        n_cols, temperature,
        BLOCK_SIZE=BLOCK,
    )
    return out


# ── Fused RMS Norm (used in LLaMA-style models) ─────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def _rms_norm_kernel(
    x_ptr, out_ptr, weight_ptr,
    stride_row, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_cols

    x = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

    # RMS = sqrt(mean(x^2) + eps)
    ms      = tl.sum(x * x, axis=0) / n_cols
    rms_inv = 1.0 / tl.sqrt(ms + eps)

    out = (x * rms_inv * w).to(tl.float16)
    tl.store(out_ptr + row * stride_row + offsets, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK = min(triton.next_power_of_2(n_cols), 4096)
    out = torch.empty_like(x)
    _rms_norm_kernel[(n_rows,)](
        x, out, weight, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK)
    return out


# ── Fused Cross-Entropy Loss ──────────────────────────────────────────────────
@triton.jit
def _cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr,
    stride_row, n_classes,
    BLOCK_SIZE: tl.constexpr,
):
    row     = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_classes

    logits = tl.load(logits_ptr + row * stride_row + offsets,
                     mask=mask, other=-float('inf'))
    label  = tl.load(labels_ptr + row)

    # Stable log-softmax
    max_l  = tl.max(logits, axis=0)
    logits = logits - max_l
    log_z  = tl.log(tl.sum(tl.exp(logits), axis=0))

    # Cross entropy = -logit[label] + log(Z)
    correct_logit = tl.sum(
        tl.where(offsets == label, logits, 0.0), axis=0)
    loss = -correct_logit + log_z

    tl.store(loss_ptr + row, loss)


def triton_cross_entropy(logits: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
    B, C = logits.shape
    BLOCK = min(triton.next_power_of_2(C), 4096)
    losses = torch.empty(B, device=logits.device, dtype=torch.float32)
    _cross_entropy_fwd_kernel[(B,)](
        logits, labels, losses,
        logits.stride(0), C, BLOCK_SIZE=BLOCK)
    return losses.mean()
```

---

## 20.7 Component 5: PyTorch Extension Bindings (`csrc/bindings.cpp`)

Applies: PyTorch C++ extension (Ch 15), autograd integration.

```cpp
// csrc/bindings.cpp
#include <torch/extension.h>

// Forward declarations from .cu files
torch::Tensor tinycuda_gemm(torch::Tensor A, torch::Tensor B,
                             float alpha, float beta);
torch::Tensor tinycuda_attention_fwd(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V);
void          tinycuda_bias_gelu(torch::Tensor x, torch::Tensor bias);
torch::Tensor tinycuda_layer_norm(torch::Tensor x, torch::Tensor gamma,
                                   torch::Tensor beta, float eps);
torch::Tensor tinycuda_residual_ln(torch::Tensor x, torch::Tensor residual,
                                    torch::Tensor gamma, torch::Tensor beta,
                                    float eps);

// ── Autograd function for GEMM (enables backward pass) ───────────────────────
class TinyCUDAGEMMFunction : public torch::autograd::Function<TinyCUDAGEMMFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor A, torch::Tensor B, float alpha
    ) {
        ctx->save_for_backward({A, B});
        ctx->saved_data["alpha"] = alpha;
        return tinycuda_gemm(A, B, alpha, 0.0f);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved  = ctx->get_saved_variables();
        auto A      = saved[0];
        auto B      = saved[1];
        float alpha = ctx->saved_data["alpha"].toDouble();
        auto dC     = grad_outputs[0];

        // dA = dC @ B^T,  dB = A^T @ dC
        auto dA = tinycuda_gemm(
            dC.to(torch::kFloat16),
            B.t().contiguous().to(torch::kFloat16),
            alpha, 0.0f
        ).to(A.dtype());

        auto dB = tinycuda_gemm(
            A.t().contiguous().to(torch::kFloat16),
            dC.to(torch::kFloat16),
            alpha, 0.0f
        ).to(B.dtype());

        return {dA, dB, torch::Tensor()};
    }
};

torch::Tensor tinycuda_gemm_autograd(torch::Tensor A, torch::Tensor B,
                                      float alpha = 1.0f) {
    return TinyCUDAGEMMFunction::apply(A, B, alpha);
}

// ── pybind11 module registration ─────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TinyCUDA: Custom CUDA kernels for ML";

    m.def("gemm",        &tinycuda_gemm_autograd,
          "Custom GEMM with autograd (A @ B * alpha)",
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f);

    m.def("attention_fwd", &tinycuda_attention_fwd,
          "Fused multi-head attention forward pass",
          py::arg("Q"), py::arg("K"), py::arg("V"));

    m.def("bias_gelu",   &tinycuda_bias_gelu,
          "Fused bias add + GELU (in-place)",
          py::arg("x"), py::arg("bias"));

    m.def("layer_norm",  &tinycuda_layer_norm,
          "Layer normalization",
          py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5f);

    m.def("residual_ln", &tinycuda_residual_ln,
          "Fused residual add + layer normalization",
          py::arg("x"), py::arg("residual"),
          py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5f);
}
```

---

## 20.8 Component 6: Python Layer Wrappers (`tinycuda/layers.py`)

Applies: putting it all together as `nn.Module`.

```python
# tinycuda/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tinycuda._C import (
    gemm, attention_fwd, bias_gelu, layer_norm, residual_ln
)
from tinycuda.kernels import triton_softmax


class TinyLinear(nn.Module):
    """Linear layer backed by custom GEMM kernel."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.float16) *
            (2.0 / in_features) ** 0.5
        )
        self.bias_param = nn.Parameter(
            torch.zeros(out_features, dtype=torch.float16)
        ) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features] → [B, out_features]
        x_fp16 = x.to(torch.float16)
        out = gemm(x_fp16, self.weight.t().contiguous())
        out = out.to(torch.float16)
        if self.bias_param is not None:
            out = out + self.bias_param
        return out


class TinyAttention(nn.Module):
    """Multi-head self-attention with fused CUDA kernel."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = TinyLinear(d_model, 3 * d_model)
        self.out_proj  = TinyLinear(d_model, d_model)
        self.dropout   = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Project to Q, K, V in one GEMM
        qkv = self.qkv_proj(x.view(B * T, D))  # [B*T, 3*D]
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        Q, K, V = qkv.unbind(dim=2)    # Each: [B, T, H, head_dim]

        # Reshape to [B, H, T, head_dim] for attention kernel
        Q = Q.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        K = K.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        V = V.permute(0, 2, 1, 3).contiguous().to(torch.float16)

        # Fused attention kernel
        if self.head_dim in (64, 128):
            attn_out = attention_fwd(Q, K, V)  # [B, H, T, head_dim]
        else:
            # Fallback to PyTorch SDPA for unsupported head dims
            attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Reshape and project output
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()  # [B, T, H, hd]
        attn_out = attn_out.view(B * T, D)
        out = self.out_proj(attn_out).view(B, T, D)
        return out


class TinyMLP(nn.Module):
    """FFN: Linear → GELU → Linear."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1  = TinyLinear(d_model, d_ff)
        self.fc2  = TinyLinear(d_ff,    d_model)
        self.bias = nn.Parameter(torch.zeros(d_ff, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.fc1(x.view(B * T, D))        # [B*T, d_ff]
        bias_gelu(h, self.bias)                # Fused in-place bias + GELU
        out = self.fc2(h).view(B, T, D)
        return out


class TinyTransformerLayer(nn.Module):
    """
    Full transformer encoder layer using TinyCUDA kernels.
    Implements: Pre-Norm architecture
        x = x + Attention(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.attn   = TinyAttention(d_model, n_heads, dropout)
        self.mlp    = TinyMLP(d_model, d_ff)
        self.gamma1 = nn.Parameter(torch.ones(d_model,  dtype=torch.float32))
        self.beta1  = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
        self.gamma2 = nn.Parameter(torch.ones(d_model,  dtype=torch.float32))
        self.beta2  = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Pre-norm + attention + residual
        x_flat = x.view(B * T, D)
        x_norm = layer_norm(x_flat.to(torch.float16),
                             self.gamma1, self.beta1, self.eps)
        attn_out = self.attn(x_norm.view(B, T, D))

        # Fused residual add + layer norm (second norm for MLP)
        x_flat2  = (x_flat.to(torch.float16) +
                    attn_out.view(B * T, D).to(torch.float16))
        x_norm2  = layer_norm(x_flat2, self.gamma2, self.beta2, self.eps)

        # MLP + residual
        mlp_out  = self.mlp(x_norm2.view(B, T, D))
        out      = (x_flat2 + mlp_out.view(B * T, D)).view(B, T, D)
        return out
```

---

## 20.9 Component 7: CUDA Graph Inference Engine (`tinycuda/graph.py`)

Applies: CUDA Graphs (Ch 19), multi-stream pipeline.

```python
# tinycuda/graph.py
import torch
from tinycuda.layers import TinyTransformerLayer


class TinyCUDAInferenceEngine:
    """
    CUDA Graph–accelerated inference engine for TinyTransformerLayer.

    Usage:
        model = TinyTransformerLayer(d_model=512, n_heads=8).cuda().eval()
        engine = TinyCUDAInferenceEngine(model, batch_size=16, seq_len=128, d_model=512)
        output = engine.run(input_tensor)
    """

    def __init__(self, model: TinyTransformerLayer,
                 batch_size: int, seq_len: int, d_model: int):
        self.model = model
        self.B, self.T, self.D = batch_size, seq_len, d_model

        # Static buffers — same address every call
        self.static_input  = torch.zeros(B, T, D, device='cuda', dtype=torch.float16)
        self.static_output = torch.empty(B, T, D, device='cuda', dtype=torch.float16)

        self._build_graph()

    def _build_graph(self):
        # Step 1: Warmup (initialize cuBLAS, allocate workspaces)
        warmup_stream = torch.cuda.Stream()
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _ = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        # Step 2: Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

        print(f"[TinyCUDA] CUDA Graph captured for "
              f"B={self.B}, T={self.T}, D={self.D}")

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on a new input batch."""
        assert x.shape == (self.B, self.T, self.D), \
            f"Expected ({self.B}, {self.T}, {self.D}), got {x.shape}"

        # Copy new input into static buffer (in-place!)
        self.static_input.copy_(x.to(torch.float16))

        # Single call triggers the entire captured forward pass
        self.graph.replay()

        return self.static_output.clone()

    def benchmark(self, n_iters: int = 500) -> dict:
        import time

        x = torch.randn(self.B, self.T, self.D, device='cuda', dtype=torch.float16)

        # Eager baseline
        self.model.eval()
        for _ in range(10): _ = self.model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters): _ = self.model(x)
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) / n_iters * 1000

        # Graph
        for _ in range(10): self.run(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters): self.run(x)
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) / n_iters * 1000

        throughput_eager = self.B * n_iters / (eager_ms / 1000 * n_iters)
        throughput_graph  = self.B * n_iters / (graph_ms  / 1000 * n_iters)

        return {
            'eager_ms':          round(eager_ms, 3),
            'graph_ms':          round(graph_ms,  3),
            'speedup':           round(eager_ms / graph_ms, 2),
            'throughput_eager':  int(throughput_eager),
            'throughput_graph':  int(throughput_graph),
        }
```

---

## 20.10 Complete Test Suite (`tests/`)

```python
# tests/test_gemm.py
import torch
import pytest
from tinycuda._C import gemm

def test_gemm_correctness():
    M, K, N = 256, 512, 256
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)

    ref = torch.mm(A, B).float()
    out = gemm(A, B)

    # fp16 GEMM: allow ~1% relative error
    assert torch.allclose(ref, out, rtol=0.01, atol=0.5), \
        f"Max error: {(ref - out).abs().max():.3f}"

def test_gemm_shapes():
    for M, K, N in [(128, 256, 512), (512, 64, 1024), (1, 512, 512)]:
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        out = gemm(A, B)
        assert out.shape == (M, N)

def test_gemm_backward():
    A = torch.randn(64, 128, device='cuda', dtype=torch.float16,
                    requires_grad=True)
    B = torch.randn(128, 64, device='cuda', dtype=torch.float16,
                    requires_grad=True)
    out = gemm(A, B)
    loss = out.float().sum()
    loss.backward()
    assert A.grad is not None and B.grad is not None


# tests/test_attention.py
from tinycuda._C import attention_fwd

def test_attention_correctness():
    B, H, T, D = 2, 8, 64, 64
    Q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)

    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), is_causal=True)
    out = attention_fwd(Q, K, V).float()

    assert torch.allclose(ref, out, atol=0.05), \
        f"Max error: {(ref-out).abs().max():.4f}"


# tests/test_training.py
from tinycuda.layers import TinyTransformerLayer
from tinycuda.kernels import triton_cross_entropy

def test_full_training_step():
    model = TinyTransformerLayer(d_model=256, n_heads=4, d_ff=512).cuda()
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(5):
        x     = torch.randn(4, 32, 256, device='cuda', dtype=torch.float16)
        out   = model(x)               # [4, 32, 256]

        # Simple reconstruction loss
        loss  = (out.float() - x.float()).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"  Step {step}: loss = {loss.item():.4f}")

    assert not torch.isnan(loss), "NaN in training loss"
    print("Training test passed.")
```

---

## 20.11 Benchmark Suite (`benchmarks/bench_vs_pytorch.py`)

```python
# benchmarks/bench_vs_pytorch.py
import torch
import time
from tinycuda.layers import TinyTransformerLayer
from tinycuda.graph  import TinyCUDAInferenceEngine

def benchmark(fn, n_warmup=10, n_iters=200):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000

def run_benchmarks():
    configs = [
        # (d_model, n_heads, d_ff, batch, seq_len)
        (256,  4,  512,  8,  64),
        (512,  8,  2048, 8,  128),
        (512,  8,  2048, 16, 128),
        (1024, 16, 4096, 4,  256),
    ]

    print(f"\n{'Config':<40} {'PyTorch':>10} {'TinyCUDA':>10} "
          f"{'Graph':>10} {'Speedup':>10}")
    print("-" * 80)

    for d, h, ff, B, T in configs:
        # PyTorch reference
        ref_model = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=h, dim_feedforward=ff,
            batch_first=True, norm_first=True,
        ).cuda().half().eval()

        # TinyCUDA eager
        tiny_model = TinyTransformerLayer(
            d_model=d, n_heads=h, d_ff=ff).cuda().eval()

        x = torch.randn(B, T, d, device='cuda', dtype=torch.float16)

        with torch.no_grad():
            t_ref   = benchmark(lambda: ref_model(x))
            t_tiny  = benchmark(lambda: tiny_model(x))

        # TinyCUDA + CUDA Graph
        engine    = TinyCUDAInferenceEngine(tiny_model, B, T, d)
        t_graph   = benchmark(lambda: engine.run(x))

        cfg_str = f"d={d}, h={h}, ff={ff}, B={B}, T={T}"
        print(f"{cfg_str:<40} {t_ref:>9.2f}ms {t_tiny:>9.2f}ms "
              f"{t_graph:>9.2f}ms {t_ref/t_graph:>9.2f}x")

    print()


# Profile breakdown with torch.profiler
def profile_single_layer():
    model = TinyTransformerLayer(d_model=512, n_heads=8, d_ff=2048).cuda().eval()
    x = torch.randn(8, 128, 512, device='cuda', dtype=torch.float16)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        for _ in range(20):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

    print("\nTop CUDA kernels by time:")
    print(prof.key_averages(group_by_stack_n=5).table(
        sort_by="cuda_time_total", row_limit=15))

    prof.export_chrome_trace("tinycuda_trace.json")
    print("Chrome trace saved to tinycuda_trace.json")
    print("Open in chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    print("=" * 80)
    print("TinyCUDA Benchmark Suite")
    print("=" * 80)
    run_benchmarks()
    profile_single_layer()
```

---

## 20.12 Roofline Analysis (`tinycuda/profiler.py`)

Applies: roofline model (Ch 12), Nsight metrics (Ch 13).

```python
# tinycuda/profiler.py
"""
Roofline analysis utilities for TinyCUDA kernels.
Uses PyTorch's CUDA profiler to estimate arithmetic intensity.
"""
import torch
import subprocess
import json

def compute_roofline_metrics(fn, peak_flops_tflops: float,
                              peak_bandwidth_gbps: float):
    """
    Estimate AI and roofline position for a CUDA operation.

    Args:
        fn: callable that runs the kernel
        peak_flops_tflops: GPU peak FP16 TFLOPS (e.g. A100=312, T4=65)
        peak_bandwidth_gbps: GPU peak memory bandwidth GB/s (e.g. A100=2000, T4=300)
    """
    # Measure execution time
    for _ in range(10): fn()
    torch.cuda.synchronize()

    import time
    N = 100
    t0 = time.perf_counter()
    for _ in range(N): fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / N * 1000

    print(f"\n{'='*50}")
    print(f"Execution time:     {elapsed_ms:.3f} ms")
    print(f"Peak FP16 TFLOPS:   {peak_flops_tflops:.1f}")
    print(f"Peak BW GB/s:       {peak_bandwidth_gbps:.1f}")
    print(f"Ridge point:        "
          f"{peak_flops_tflops*1e12 / (peak_bandwidth_gbps*1e9):.1f} FLOPs/Byte")
    print(f"\nRun with Nsight Compute for full roofline:")
    print(f"  ncu --set full --section SpeedOfLight_RooflineChart \\")
    print(f"      -o roofline.ncu-rep python your_script.py")
    print(f"='*50")


def print_nsight_command(script: str, kernel_name: str = ""):
    """Print the Nsight Compute command for profiling a specific kernel."""
    filter_flag = f"--kernel-name {kernel_name}" if kernel_name else ""
    print(f"""
Nsight Compute profiling commands:

# Full profile:
ncu --set full {filter_flag} -o profile.ncu-rep python {script}

# Roofline only:
ncu --section SpeedOfLight_RooflineChart {filter_flag} python {script}

# Memory throughput:
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\\
              l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\\
              sm__sass_thread_inst_executed_op_ffma_pred_on.sum \\
    {filter_flag} python {script}

# Open in Nsight Compute GUI:
ncu-ui profile.ncu-rep
""")
```

---

## 20.13 Putting It All Together: End-to-End Training Run

```python
# train.py — Full training loop using TinyCUDA
import torch
import torch.nn as nn
from tinycuda.layers import TinyTransformerLayer
from tinycuda.kernels import triton_softmax, triton_cross_entropy
from tinycuda.graph   import TinyCUDAInferenceEngine

class TinyGPT(nn.Module):
    """Minimal GPT-style model using TinyCUDA layers."""

    def __init__(self, vocab_size=50257, d_model=256, n_heads=4,
                 d_ff=512, n_layers=4, max_seq=256):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pos    = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([
            TinyTransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = (self.embed(tokens) + self.pos(pos)).half()
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x.float())  # [B, T, vocab_size]
        return logits


def train():
    device = torch.device('cuda')

    model = TinyGPT(
        vocab_size=50257, d_model=256, n_heads=4,
        d_ff=512, n_layers=4, max_seq=128
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    scaler = torch.cuda.amp.GradScaler()  # Mixed precision (Ch 14 concepts)

    # Synthetic data (replace with real tokenized dataset)
    def get_batch(B=8, T=128):
        tokens = torch.randint(0, 50257, (B, T+1), device=device)
        return tokens[:, :-1], tokens[:, 1:]

    print("\nTraining TinyGPT with TinyCUDA kernels...")
    print(f"{'Step':<8} {'Loss':<12} {'Tokens/sec':<15} {'GPU Mem (MB)':<15}")
    print("-" * 55)

    for step in range(200):
        x, y = get_batch()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x)                          # [B, T, V]
            # Triton fused cross-entropy
            loss = triton_cross_entropy(
                logits.view(-1, 50257),
                y.contiguous().view(-1)
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 20 == 0:
            mem_mb = torch.cuda.memory_allocated() / 1e6
            print(f"{step:<8} {loss.item():<12.4f} "
                  f"{8*128*20/1:<15,.0f} {mem_mb:<15.1f}")

    print("\nTraining complete.")

    # Switch to graph-accelerated inference
    print("\nBuilding CUDA Graph for inference...")
    model.eval()
    single_layer = TinyTransformerLayer(256, 4, 512).cuda().eval()
    # Copy weights from trained model's first layer
    single_layer.load_state_dict(model.layers[0].state_dict())

    engine = TinyCUDAInferenceEngine(single_layer, batch_size=8,
                                      seq_len=128, d_model=256)
    results = engine.benchmark(n_iters=300)
    print(f"\nInference benchmark:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    train()
```

---

## 20.14 Curriculum Retrospective

Every chapter contributed a concrete piece of this library:

| Chapter | Concept | Where It Appears in TinyCUDA |
|---|---|---|
| 1–2 | CUDA thread model, memory model | All kernels: `blockIdx`, `threadIdx`, grid config |
| 3–4 | Memory hierarchy, coalescing | `ld.global.v4` in attention; smem in GEMM |
| 5 | Parallel patterns | Fused bias+GELU; residual+LN in single kernel |
| 6 | Warp mechanics | Warp shuffle reduction in layer norm |
| 7 | Shared memory | GEMM tiles; KV cache in attention |
| 8 | Bank conflicts | `SMEM_PAD` in GEMM shared memory arrays |
| 9 | Advanced memory | Async L2 prefetch patterns in GEMM loop |
| 10 | Occupancy | `__launch_bounds__(256, 2)` on GEMM kernel |
| 11 | Multi-stream | Multi-stream pipeline in `InferenceEngine` |
| 12 | Roofline model | `profiler.py` roofline analysis utilities |
| 13 | Nsight profiling | `bench_vs_pytorch.py` trace export |
| 14 | Mixed precision | `fp16` throughout; `float32` accumulators |
| 15 | PyTorch extensions | `bindings.cpp`, `setup.py`, autograd Function |
| 16 | cuBLAS / cuDNN | Fallback path in attention for unsupported dims |
| 17 | Triton | `kernels.py`: softmax, RMS norm, cross-entropy |
| 18 | PTX / SASS | Vectorized loads; FMA patterns; `__launch_bounds__` |
| 19 | CUDA Graphs | `TinyCUDAInferenceEngine` with `CUDAGraph` replay |
| 20 | This capstone | The complete TinyCUDA library |

---

## 20.15 What to Build Next

You now have the skills to read and contribute to production GPU codebases. Here are natural next steps:

**Go deeper in kernels:**
- Implement INT8 quantized GEMM (IMMA tensor core instructions)
- Write a speculative decoding kernel (draft + verify in a single fused pass)
- Port the attention backward pass (dQ, dK, dV gradients in Triton)

**Go broader in systems:**
- Multi-GPU with NCCL: add `all_reduce` after each transformer layer
- Tensor parallelism: split attention heads across GPUs (Megatron-LM pattern)
- Pipeline parallelism: layer assignments across GPUs with CUDA Graph per stage

**Go into production tooling:**
- TensorRT custom plugin for TinyCUDA's attention kernel
- Compile TinyCUDA with `torch.compile` + `torch.export` for deployment
- Add INT4 weight quantization (AWQ / GPTQ dequant kernel)

**Go into research:**
- Implement Mamba SSM scan kernel in Triton
- Write a custom CUDA kernel for Mixture-of-Experts routing
- Profile and optimize your GRPO/DPO post-training pipeline (directly applicable to your Qwen2.5-Coder work)

---

## 20.16 Final Chapter Summary

| What you built | Technique |
|---|---|
| Custom GEMM | Tiled shared memory, L2 swizzle, `__launch_bounds__` |
| Fused attention | Flash attention online softmax, vectorized loads |
| Fused elementwise | Bias+GELU, residual+LN in single kernel pass |
| Triton kernels | Softmax, RMS norm, cross-entropy — all auto-tuned |
| PyTorch bindings | `CUDAExtension`, `pybind11`, `autograd.Function` |
| `nn.Module` wrappers | Drop-in replacement for standard PyTorch layers |
| CUDA Graph engine | ~50× lower latency for inference vs eager |
| Training loop | Mixed precision, gradient clipping, `GradScaler` |
| Benchmark suite | Throughput comparison, Chrome trace, Nsight commands |

You started this curriculum not knowing what a thread block was. You're finishing it having written the same class of code that powers TensorRT, FlashAttention, and vLLM.

---

*Chapter 20 of 20 — CUDA GPU Programming: Beginner to Expert*

*Curriculum complete.*
