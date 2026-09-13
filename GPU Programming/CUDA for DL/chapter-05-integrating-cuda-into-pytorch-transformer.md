# Chapter 5 — Integrating CUDA into PyTorch: The Transformer

*Part 4: Integrating CUDA into PyTorch — The Transformer. Confirmed title and scope from the book's companion repo: `book.cu/3_transformer/README.md`, "Character-level Transformer: Training & Inference with Custom CUDA Kernels" — MIT licensed, so every code excerpt below is used freely and directly from the real source.*

This chapter covers a lot of ground — the full pipeline for wiring a raw `.cu` kernel into PyTorch's autograd, a character-level GPT trained with custom training kernels, and a separate inference path built around KV-caching, GEMV, and MoE top-k routing — because that's genuinely what the book's own Chapter 5 does in one repository. It also hands you the single richest piece of real, documented debugging material in the whole book: an actual numerical-precision bug the authors traced end-to-end, which we'll dig into with the exact line of code responsible.

---

## 5.1 The Four-Layer Pipeline: Kernel → Binding → Wrapper → Module

Every custom operation in this chapter, training or inference, is built the same four-layer way:

```
your_kernel.cu        (raw __global__ CUDA kernel + a thin C++ wrapper function)
        │
binding.cpp           (TORCH_CHECK validation, tensor → raw pointer, PYBIND11_MODULE)
        │
wrapper/*.py           (torch.autograd.Function: forward()/backward() call into the compiled extension)
        │
nn.Module               (a normal-looking PyTorch layer, usable anywhere)
```

**Layer 1 — the build.** `setup.py` compiles two *separate* extensions — one for training, one for inference — each from its own list of `.cu`/`.cpp` sources:

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='naive-cu-extensions',
    ext_modules=[
        CUDAExtension(
            name='custom_training_extension',
            sources=[..., 'kernels/matmul.cu', 'kernels/elementwise.cu', 'kernels/activation.cu',
                      'kernels/softmax.cu', 'kernels/layernorm.cu', 'kernels/embedding.cu'],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            name='custom_inference_extension',
            sources=[..., 'kernels/matmul_fwd.cu', 'kernels/gemv_fwd.cu', 'kernels/elementwise_fwd.cu',
                      'kernels/activation_fwd.cu', 'kernels/softmax_fwd.cu', 'kernels/layernorm_fwd.cu',
                      'kernels/topk_fwd.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--expt-extended-lambda']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

Worth noticing: the training extension compiles with `-g` (debug symbols, no optimization flag on the C++ side) while inference compiles with `-O3` everywhere. That's a real, deliberate difference — training kernels are still being iterated on and need to be debuggable; inference kernels are the "shipped" path and are built for speed.

**Layer 2 — the binding.** Every function follows an identical shape: validate every tensor (`is_cuda`, correct `dim()`, matching sizes), extract raw dimensions as plain `int`s, call the `.cu` file's wrapper function with raw pointers via `.data_ptr<float>()`, and expose it with `PYBIND11_MODULE`:

```cpp
void matmul_fwd(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "tensors must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0) && A.size(0) == C.size(0) && B.size(1) == C.size(1),
                "matrix dimensions must be compatible");

    int M = A.size(0), N = B.size(1), K = A.size(1);
    matmul_fwd_cuda(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

// ...one such function per op, then, at the bottom of the file:
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_fwd", &matmul_fwd, "MatMul forward");
    m.def("matmul_bwd", &matmul_bwd, "MatMul backward");
    m.def("embedding_fwd", &embedding_fwd, "Embedding forward");
    m.def("embedding_bwd", &embedding_bwd, "Embedding backward");
    // ...add_fwd/bwd, mul_fwd/bwd, gelu_fwd/bwd, softmax_fwd/bwd, layernorm_fwd/bwd, batched_matmul_fwd/bwd
}
```

**Layer 3 — the wrapper.** This is where autograd actually gets wired in, via `torch.autograd.Function`:

```python
class MatMulFunction(Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)              # cache for backward — same role as Ch.4's forward cache
        M, K = A.shape
        N = B.shape[1]
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)
        import custom_training_extension as cte
        cte.matmul_fwd(A, B, C)                   # calls straight into the compiled extension
        return C

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(B)
        import custom_training_extension as cte
        cte.matmul_bwd(A, B, grad_C, grad_A, grad_B)   # grad_A = grad_C@Bᵀ, grad_B = Aᵀ@grad_C
        return grad_A, grad_B

class MatMul(torch.nn.Module):
    def forward(self, A, B):
        return MatMulFunction.apply(A, B)
```

Read that `backward()` and Chapter 4 §4.1's linear-layer gradient formulas side by side — `grad_A = grad_C @ Bᵀ` and `grad_B = Aᵀ @ grad_C` are exactly the same two formulas, just hidden behind `MatMulFunction.apply()` now instead of a hand-called C function. **This is the entire point of this section**: `torch.autograd.Function` is a formal contract — implement `forward` and `backward` as static methods, call `ctx.save_for_backward` for anything the backward pass needs, and PyTorch's autograd graph treats your raw CUDA kernel exactly like any built-in op, including composing correctly with everything else in a larger network.

**Deep dive: the contract's actual rules, not just this one example.** Two things about `torch.autograd.Function` are worth stating explicitly, since they generalize to every custom op in this chapter (and every one you'll ever write): **`backward()` must return exactly one value per argument `forward()` took** — `MatMulFunction.forward(ctx, A, B)` takes two tensor arguments, so `backward` must return a two-element tuple, one gradient per input, in the same order. `EmbeddingFunction` further down (§5.3) is the sharper illustration: it takes `(weight, indices)` and returns `(grad_weight, None)` — the `None` isn't a placeholder for "not implemented," it's the formally correct answer for a non-differentiable input (`indices` are discrete integers; there's no continuous quantity to take a gradient of). Get the tuple length or order wrong and PyTorch doesn't silently do the wrong thing — it raises immediately, because autograd relies on this shape contract to route gradients backward through the rest of the graph correctly. Second: `ctx.save_for_backward` exists specifically for *tensors* — it plugs into PyTorch's own reference-counting and memory-reuse machinery so saved activations aren't freed prematurely, which is why the pattern is `ctx.save_for_backward(A, B)` rather than just `ctx.A = A` (plain attribute assignment works for saving non-tensor values like shapes or flags, but bypasses this bookkeeping for actual tensors).

## 5.2 Two-Stage Development: PyTorch Baseline → Custom CUDA

The book states its own methodology directly, and it's worth adopting verbatim: **always get a working PyTorch-only baseline first**, then swap in custom CUDA operations **one at a time**, verifying each swap doesn't change the loss curve before moving to the next. This is Chapter 1 §1.6.1's "naive-then-optimize" discipline, and Chapter 3's "CPU-first development," applied one more level up — now the trusted reference isn't a CPU loop, it's a framework you already know is correct.

The book's own measured result for this exact character-level GPT (architecture below): **PyTorch baseline ≈ 20 s, custom CUDA ≈ 15 s, for 1000 iterations to the same loss curve.** A real, modest win — and notice it required getting *ten-plus* custom ops (matmul, batched matmul, add, mul, GELU, softmax, layernorm, embedding — forward *and* backward for each) all correct before it paid off at all.

The single most common failure mode, per the book's own troubleshooting notes: **passing a non-contiguous tensor into a custom kernel.** PyTorch operations like `.transpose()` or `.permute()` return *views* — the underlying memory layout no longer matches the tensor's logical shape — and every custom kernel in this chapter assumes standard contiguous row-major layout, with no stride-aware indexing. The fix is always the same:

```python
assert tensor.is_contiguous(), f"Tensor not contiguous: {tensor.shape}"
# or, unconditionally: tensor = tensor.contiguous()
```

Skipping this doesn't raise a clean error — the book's own words are direct: *"training works initially but crashes randomly or produces NaN gradients."* That's a symptom you should now recognize instantly.

**Deep dive: why "modest" is exactly what you'd expect here, by the same arithmetic as Chapter 4.** A 25% wall-clock reduction (20s → 15s) might look underwhelming next to Chapter 6–7's later 100×+ speedups, but it makes sense the moment you count kernel launches the way Chapter 4 §4.6 did. A single transformer block in this architecture needs roughly: LayerNorm, a QKV projection matmul, a batched matmul for `Q@Kᵀ`, softmax, a batched matmul for `softmax@V`, an output projection matmul, a residual add, a second LayerNorm, an MLP up-projection matmul, GELU, an MLP down-projection matmul, and a second residual add — **on the order of 11 forward-pass kernel launches per layer.** Across 8 layers, plus the embedding lookup, a final LayerNorm, and an output projection, that's roughly **90 forward launches per training step** (a reasoned estimate from this chapter's own listed op roster, not a literally-counted figure from `train.py`) — and, following the same "each forward op needs a matching backward op" logic from Chapter 4, likely close to **double that including backward, before optimizer updates**. Over 1000 training iterations, that puts total kernel launches for this whole run in the same rough **tens-of-thousands to low-hundred-thousands** range as Chapter 4's MNIST training run (212,500 launches, precisely counted there). Fixed per-launch overhead eating into the win is a recurring cost of this architecture — custom CUDA replacing PyTorch's own (already reasonably optimized) kernels one-for-one, without any *fusion* across ops, was never going to buy more than a modest constant-factor improvement. Fusion — collapsing several of those ~11 per-layer launches into one kernel — is exactly what Chapter 8's Flash Attention (fusing 3 attention launches into 1) demonstrates is possible, and exactly what this chapter's own kernels don't attempt.

## 5.3 Building the Character-Level GPT: Training Kernels

Confirmed architecture and training setup, straight from the README:

| Hyperparameter | Value |
|---|---|
| Batch size | 16 |
| Sequence length | 64 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 8 |
| Vocabulary | ~80 (character-level) |
| Learning rate | 3e-4 |
| Training iterations | 1000 |
| Parameters | ~1.6M |
| Dataset | *The Wonderful Wizard of Oz* (public domain) |

The full training-kernel roster (from `binding.cpp`'s forward declarations): element-wise add/multiply, GELU activation, 2D matmul, **batched** matmul (for per-head, per-batch attention scores and attention-weighted values — a 3D tensor operation, `(batch, M, N) @ (batch, N, K)`), softmax, layernorm, and embedding.

The embedding op is worth walking through fully, because its backward pass is the first *new* CUDA technique in this chapter (though you built the theory for it already, in Chapter 4 Exercise 5):

```cuda
__global__ void embedding_fwd_kernel(const float* weight, const int* indices,
                                   float* out, int num_indices, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_indices * n_embd;
    if (idx < total_elements) {
        int token_idx = idx / n_embd;
        int emb_idx = idx % n_embd;
        int weight_idx = indices[token_idx] * n_embd + emb_idx;
        out[idx] = weight[weight_idx];      // pure lookup — one thread per output scalar
    }
}

__global__ void embedding_bwd_kernel(const float* grad_out, const int* indices,
                                   float* grad_weight, int num_indices, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_indices * n_embd;
    if (idx < total_elements) {
        int token_idx = idx / n_embd;
        int emb_idx = idx % n_embd;
        int weight_idx = indices[token_idx] * n_embd + emb_idx;
        // Use atomicAdd because multiple tokens may share the same embedding
        atomicAdd(&grad_weight[weight_idx], grad_out[idx]);
    }
}
```

Why the backward pass *needs* `atomicAdd` and the forward pass doesn't: forward is a pure gather (many threads read, no two threads ever write the same output location). Backward is a **scatter-add** — if the same vocabulary index appears twice in one training batch (entirely normal; common characters repeat constantly), two different threads will compute a gradient contribution for the *same* `grad_weight[weight_idx]` simultaneously. Without `atomicAdd`, that's a data race — whichever thread's write lands last silently overwrites the other's contribution instead of summing them. This is the exact scatter-reduction pattern Chapter 4 Exercise 5 asked you to design a privatized (shared-memory-first, single-atomic-per-block) improvement for — here it is in the wild, in its un-optimized form.

The Python side wires it into a familiar-looking module:

```python
class EmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, weight, indices):
        ctx.save_for_backward(weight, indices)
        batch_size, seq_len = indices.shape
        out = torch.empty(batch_size, seq_len, weight.shape[1], dtype=weight.dtype, device=weight.device)
        import custom_training_extension as cte
        cte.embedding_fwd(weight, indices, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight, indices = ctx.saved_tensors
        grad_weight = torch.zeros_like(weight)   # must start at zero — atomicAdd accumulates onto it
        import custom_training_extension as cte
        cte.embedding_bwd(grad_out, indices.to(torch.int32), grad_weight)
        return grad_weight, None                  # None: indices are discrete, no gradient flows to them

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, indices):
        return EmbeddingFunction.apply(self.weight, indices.to(torch.int32))
```

Notice `backward` returns a two-element tuple, `(grad_weight, None)` — one entry per argument `forward` took (`weight, indices`). Returning `None` for `indices` is how you tell autograd "this input isn't differentiable, don't propagate anything to it" — token indices are discrete integers, there's no meaningful gradient with respect to "which token."

**Deep dive: how often the atomic-contention path actually gets exercised, at this chapter's scale.** With a batch size of 16 and sequence length of 64, one training batch touches `16 × 64 = 1,024` token positions. Against a character-level vocabulary of only ≈80 possible tokens, simple pigeonhole reasoning says the *average* vocabulary entry is looked up `1,024 / 80 ≈ 12.8` times in a single batch — meaning `atomicAdd` contention in `embedding_bwd_kernel` isn't an edge case here, it's the *typical* case, on essentially every training step. Worth contrasting against scale: this entire embedding table is `80 × 128 × 4 bytes = 40,960 bytes` — comfortably smaller than either of your GPUs' L2 cache, so even naive atomic traffic here stays cheap in absolute terms. A real subword-tokenized LLM's embedding table (50,000+ vocabulary entries, hidden sizes in the thousands) is a different story: `50,000 × 4096 × 4 bytes ≈ 800 MB`, far larger than any GPU's L2 cache, and with a batch's tokens spread across a much larger vocabulary, contention concentrates on whichever tokens happen to be common (articles, common subwords) rather than being spread almost uniformly the way it is here. Chapter 5 Exercise 3's privatization fix matters more, not less, as vocabulary size grows — this toy example is exactly where you'd want to *learn* the technique, precisely because the stakes of getting it wrong are low while you're still practicing.

## 5.4 Transformer Inference: KV-Cache, GEMV & Top-K

Inference gets an entirely separate kernel set, and the reason is a direct, concrete application of Chapter 1 §1.5.2's claim that **autoregressive decode is memory-bandwidth-bound, not FLOP-bound**. Confirmed inference configuration:

| Hyperparameter | Value |
|---|---|
| Batch size | 1 (autoregressive) |
| Sequence length | 64 |
| Embedding dimension | 768 |
| Attention heads | 8 |
| Transformer layers | 24 |
| Vocabulary | 95 |
| MoE experts | 8 (top-2 routing) |
| Max new tokens | 200 |
| Parameters | Dense ~177M, MoE ~708M |

**Why GEMV instead of GEMM at inference:** during autoregressive generation, each step processes exactly *one new token*. Every weight matrix in the network — hundreds of millions of parameters for the dense 177M model — multiplies against a single-token activation *vector*, not a batch of vectors. That's matrix-times-vector (GEMV), not matrix-times-matrix (GEMM), and GEMV's arithmetic intensity is far lower: you load an entire `M×N` weight matrix from device memory but perform only `2MN` FLOPs on it, versus GEMM reusing that same matrix across many output columns. This is precisely why `gemv_fwd.cu` exists as a separate kernel file from `matmul_fwd.cu`, and it's exactly the mechanism behind Chapter 1's claim that single-sequence LLM decode spends most of its time simply *reading weights from memory*, not computing on them.

```cuda
__global__ void gemv_kernel(const float* A, const float* x, float* y, int batch, int M, int N) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx < batch && row < M) {
        float sum = 0.0f;
        const float* A_batch = A + batch_idx * M * N;
        const float* x_batch = x + batch_idx * N;
        for (int col = 0; col < N; col++)
            sum += A_batch[row * N + col] * x_batch[col];   // one thread does the FULL length-N dot product
        y[batch_idx * M + row] = sum;
    }
}
```

For the single (non-batched) case, the launch config is `dim3 threadsPerBlock(1, 256, 1)` — note the **x-dimension is fixed at 1**. Every thread independently walks an entire `N`-length row with no cooperation at all: no shared-memory reduction, no warp-shuffle partial sums, nothing. It's correct, and — same shape of problem as Chapter 4's `softmax_kernel<<<batch_size,1>>>` and this chapter's own top-k kernel below — it leaves an enormous amount of available parallelism on the table by design, since a single dot product over hundreds or thousands of elements is exactly the kind of reduction that benefits hugely from cooperating threads. Fixing this (parallel reduction across threads within a row) is Part 5, Chapter 17's subject.

**Deep dive: GEMV's arithmetic intensity has a hard ceiling, and it's low regardless of model size.** For a GEMV `y = Ax` with `A` sized `M×N`: FLOPs are `2MN`, and bytes moved are dominated by reading `A` exactly once — `4MN` bytes, since the `4N`-byte input and `4M`-byte output are negligible next to `A` for any reasonably-sized layer. That gives:

```
Arithmetic intensity = 2MN / 4MN = 0.5 FLOP/byte — independent of M and N entirely
```

This is a genuinely different, and worse, situation than GEMM's. Chapter 3 §3.3.2 showed GEMM's *ideal* arithmetic intensity for a 512×512×256 matmul working out to a clean 64.0 FLOP/byte — and, critically, that number **grows** as the matrices get bigger, because a GEMM reuses each loaded value across an entire additional output dimension (the batch, or the other matrix's rows/columns). A GEMV has no such second dimension to reuse across — batch size is 1 by construction at single-sequence decode time — so its arithmetic intensity is capped at roughly 0.5 FLOP/byte **no matter how large the weight matrix grows.** Set that 0.5 against your own GPUs' confirmed ridge points from Chapter 3 (≈38.0 FLOP/byte for the RTX 3090, ≈25.3 for the T4) and the conclusion is unambiguous: single-sequence decode is memory-bound by a margin of roughly **50–75×**, and scaling the model up doesn't change that ratio at all — it's a structural property of the operation, not a symptom of an under-optimized kernel. This is Chapter 1 §1.5.2's claim, now with the exact math behind it.

**KV-caching**, conceptually: without it, generating token *T* would require recomputing Key and Value projections for *all* T-1 previous tokens, every single step — quadratic total work across a generation. The fix is to compute K and V for each token exactly once, as it's generated, and cache them; each new decode step only computes Q, K, V for the *new* token, appends the new K/V to the cache, and attends the new query against the full cached K/V history. This is what makes the "GEMV, not GEMM" framing above hold at every layer, every step, for the entire 200-token generation in this chapter's inference benchmark.

**Deep dive: quantifying "quadratic vs. linear" for this exact benchmark.** Without caching, generating up to `T=264` total positions (64 prompt tokens + 200 new ones, this chapter's own confirmed figures) means step `t` redoes K/V projections for all `t` prior positions — total work across the whole generation is `Σ_{t=1}^{264} t = 264×265/2 ≈ 34,980` "token-projections" worth of redundant computation. With caching, it's exactly `264` — one projection per token, ever. That's a **≈132× reduction** in total K/V-projection work for this specific, fairly modest generation length, and the ratio only gets worse (favoring caching more) for longer generations, since it scales with `T` versus `T²/2`.

**Top-K for MoE expert routing:**

```cuda
__global__ void topk_kernel(const float* input, float* values, int* indices,
                           int batch_size, int n, int k) {
    int batch_idx = blockIdx.x;      // one BLOCK per batch sample...
    if (batch_idx < batch_size) {
        const float* input_row = input + batch_idx * n;
        float* values_row = values + batch_idx * k;
        int* indices_row = indices + batch_idx * k;

        for (int i = 0; i < k; i++) { values_row[i] = -INFINITY; indices_row[i] = -1; }

        for (int i = 0; i < n; i++) {                 // ...but ONE thread does all n elements, serially
            float val = input_row[i];
            for (int j = 0; j < k; j++) {
                if (val > values_row[j]) {              // <-- strict inequality, no tie-break. See §5.5.
                    for (int m = k - 1; m > j; m--) { values_row[m] = values_row[m-1]; indices_row[m] = indices_row[m-1]; }
                    values_row[j] = val;
                    indices_row[j] = i;
                    break;
                }
            }
        }
    }
}
```

Launched with `threads_per_block = 1, num_blocks = batch_size` — the identical single-thread-per-row anti-pattern you've now seen three times in this course (Chapter 4's naive softmax, this chapter's own single-query GEMV, and now top-k). It's a naive O(n·k) insertion sort per row, run on exactly one thread. The book keeps it this way *on purpose*, because — as you're about to see — this kernel's naivety isn't just a performance problem.

## 5.5 The MoE Numerical Precision Case Study

This is a real, documented investigation the book's authors ran, and it's worth reading as a worked example of exactly the debugging discipline you'll need for the rest of this course. The observed problem: **combining the custom CUDA softmax with the custom CUDA top-k for MoE expert routing produced 50% token divergence** from the PyTorch reference over a 200-token generation — but each custom op, tested *individually* against its PyTorch equivalent, matched exactly.

Their own ablation table, run by toggling which op (custom vs. PyTorch) fed into which:

| Test | Softmax | Top-K | Token divergence |
|---|---|---|---|
| Baseline | Custom CUDA | Custom CUDA | **50%** |
| Test A | PyTorch | Custom CUDA | 0% |
| Test B | Custom CUDA | PyTorch | 0% |
| Test C | PyTorch | PyTorch | 0% |

Only the *combination* of both custom kernels fails. Here's the mechanism, traceable to an exact line of the real code above: `topk_kernel`'s comparison is `if (val > values_row[j])` — **strictly greater-than, with no explicit tie-breaking rule.** When two expert-routing logits are extremely close (within normal floating-point noise, ~1e-7), whichever one is nominally "larger" determines which expert gets selected. The custom CUDA softmax and PyTorch's own softmax kernel don't compute bit-identical results — different reduction order, different intermediate rounding — so a pair of logits that's `[0.400000, 0.400000]` under one softmax implementation might come out `[0.400001, 0.399999]` under the other. That tiny, individually harmless difference is exactly the kind of input where a *strict, order-dependent* comparison like `val > values_row[j]` can flip which index "wins" — and once a different expert is selected, that expert's entirely different feed-forward weights produce a different hidden state, which produces a different next-token prediction, which (autoregressively) changes every token generated after it. Over 200 generated tokens, that single routing flip early in the sequence cascades into roughly half the output disagreeing with the PyTorch reference.

The book's own framing of what this teaches, and it's worth taking seriously rather than treating as a curiosity: **custom CUDA is excellent at raw compute (matmul, elementwise math) and requires real care wherever a kernel makes a discrete decision from continuous, imprecise inputs (sorting, comparisons, routing).** Their stated account of real production practice: teams building MoE inference systems at scale keep routing/sorting logic in the framework's own battle-tested ops, and reserve custom CUDA for the compute-heavy, decision-free parts (fused kernels, expert-parallel communication) — optimizing *around* routing via kernel fusion, not by reimplementing the routing decision itself. This chapter's naive top-k stays naive on purpose, precisely so this lesson has something real to point at. Fixing it properly — deterministic tie-breaking, a real parallel selection algorithm — is explicit, unfinished business the book earmarks for a later chapter, and it's Exercise 1 below.

**Deep dive: why a 50% divergence rate isn't a fluke — there are thousands of chances for it to happen.** Each of this chapter's 24 layers makes its own independent top-2-of-8 routing decision, for every one of the 200 newly generated tokens (the routing decision for prompt tokens doesn't compound autoregressively the same way, since they're not conditioned on the model's own prior output). That's `24 × 200 = 4,800` independent routing decisions over the course of one generation, each one a fresh opportunity for two of the 8 gating logits to land within floating-point noise of each other and flip which expert wins under the strict `>` comparison. You don't need a *high* per-decision probability of a near-tie for a flip to be near-certain *somewhere* across 4,800 independent rolls — and the very first flip, however far into the 200-token generation it happens, changes every token after it (a wrong expert selected early cascades through the rest of the autoregressive chain), which is exactly why the book's own measured outcome lands at a dramatic 50%, not some small single-digit percentage you might have guessed from "logits are only occasionally that close."

## 5.6 The CUDA Kernel Debugging Methodology

The book's own three-step process for any custom-kernel bug, stated directly and worth adopting as-is:

1. **Establish a PyTorch baseline.** Temporarily swap the suspect custom op back to its `torch.nn.functional` equivalent. If the bug disappears, it's in your kernel, not your model logic.
2. **Replace incrementally, one operation at a time.** Never swap more than one op from PyTorch to custom CUDA before re-testing — otherwise you can't tell *which* swap introduced a regression.
3. **Isolate and fix**, checking in this order: tensor shapes, memory contiguity (`.contiguous()`), kernel launch grid/block dimensions, and — specifically for training bugs — the backward pass, since gradient bugs often don't show up until `loss.backward()`.

Concrete checks the book gives for step 3, each a real, minimal snippet:

```python
# Contiguity — the single most common cause of "works sometimes" bugs
assert tensor.is_contiguous(), f"Tensor not contiguous: {tensor.shape}"

# Gradient sanity after backward()
loss.backward()
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}")
    elif torch.isnan(param.grad).any():
        print(f"NaN gradients in {name}")

# Numerical accuracy against the PyTorch reference — the book's own bar is 1e-4
max_diff = torch.abs(torch_result - custom_result).max()
assert max_diff < 1e-4, f"Accuracy test failed: {max_diff}"
```

And the tooling escalation path, which previews Part 9 properly:

```bash
export CUDA_LAUNCH_BLOCKING=1     # forces synchronous kernel launches — turns async errors into sync ones you can localize
export TORCH_USE_CUDA_DSA=1       # device-side assertions with better error messages
python your_script.py

cuda-memcheck python your_script.py     # (compute-sanitizer on newer toolkits)
nsys profile --stats=true python train.py    # Nsight Systems — timeline view
ncu --set full python train.py               # Nsight Compute — per-kernel deep dive
```

You've already used `compute-sanitizer` (Chapters 2–4); `CUDA_LAUNCH_BLOCKING=1` is the tool you reach for specifically when an error is reported on the *wrong* line, because the actual failing kernel launched asynchronously several lines earlier.

---

## Hands-On Lab

```bash
git clone <repo-url> naive.cu && cd naive.cu/3_transformer
uv venv && source .venv/bin/activate
uv pip install torch torchvision torchaudio pybind11 requests

python setup.py build_ext --inplace     # builds BOTH extensions

python train.py       # compare PyTorch-baseline vs custom-CUDA loss curves and wall-clock time
python inference.py   # dense should match PyTorch exactly; MoE should show the documented divergence
```

1. **Reproduce the MoE case study yourself.** In `inference.py`, toggle `USE_PYTORCH_SOFTMAX` and `USE_PYTORCH_TOPK` through all four combinations from §5.5's table and confirm you get the same qualitative pattern (only the all-custom combination diverges significantly).
2. **Break contiguity on purpose.** Feed a `.transpose(0, 1)`'d (non-contiguous) tensor into any custom op's wrapper without calling `.contiguous()` first. Observe the failure — is it a clean error, a silent wrong-answer, or a crash? Compare to what the README predicts.
3. **Run the tooling escalation path** on `train.py`: `CUDA_LAUNCH_BLOCKING=1 python train.py`, then (if available) `nsys profile --stats=true python train.py`. You don't need to interpret every metric yet — Part 9 covers that — just confirm the tools run and produce output.

## Exercises

1. **Fix top-k's tie-breaking, for real.** Modify `topk_kernel`'s insertion condition so ties are broken deterministically (e.g., prefer the lower index on an exact or near-tie, using a small epsilon comparison instead of strict `>`). Re-run the §5.5 ablation and confirm the baseline (custom softmax + your fixed custom top-k) now shows ~0% divergence.
2. **Write a correctness test for attention itself.** Using the training kernels (`batched_matmul`, `softmax`), build a minimal single-head attention block and compare its output against `torch.nn.functional.scaled_dot_product_attention` for the same Q/K/V, within the book's own 1e-4 tolerance. Deliberately check two classic bug sources the README calls out: the scale factor (`head_size ** -0.5`) and the transpose direction for `Kᵀ` (`.transpose(-2, -1)`).
3. **Privatize the embedding backward.** Apply Chapter 4 Exercise 5's shared-memory-first, one-atomic-per-block pattern to `embedding_bwd_kernel`. Vocabulary entries that appear often in a batch (common characters, in this dataset) will see the heaviest atomic contention — that's exactly where privatization pays off most.
4. **Vectorize single-query GEMV.** `gemv_kernel`'s non-batched path uses `threadsPerBlock.x = 1`, meaning one thread computes an entire `N`-length dot product alone. Rewrite it so multiple threads cooperate per row (shared-memory or warp-shuffle partial-sum reduction), and measure achieved memory bandwidth against your GPU's theoretical peak from Chapter 1's hardware table.
5. **Fix a real "no kernel image available" risk.** `setup.py` doesn't pin a target GPU architecture. Add `'-arch=sm_86'` (RTX 3090) or the correct flag for your own GPU's compute capability (Chapter 1 §1.1's table) to the `nvcc` compile args, rebuild, and confirm both extensions still build cleanly.

---

**Next:** Chapter 6 — Optimizing GEMM I: Memory Coalescing & Shared-Memory Tiling (Part 5). You now have several honestly-naive kernels with named, understood inefficiencies — the single-thread-per-row softmax and top-k, the non-cooperative GEMV — collected across three chapters. Part 5 is where you finally fix them, rung by rung, starting with the GEMM optimization ladder.
