# Chapter 2 — Environment, Toolchain & the JIT Lifecycle

## 2.1 Installation Paths

Triton ships as Linux-only wheels (x86_64 and aarch64) — there are no official macOS or Windows builds, since the compiler pipeline targets PTX/AMDGCN codegen that assumes a Linux CUDA/ROCm toolchain underneath. If you're on Windows, the community-maintained `triton-windows` fork exists, but for this curriculum assume a Linux box or WSL2 — which matches the setup you've already been using for your GPU infrastructure and fine-tuning work.

**Path A — you already have PyTorch installed (most common).**
PyTorch declares Triton as a dependency on Linux and installs a matching version automatically — you don't need a separate install step:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import triton; print(triton.__version__)"
```

There is no separate "pytorch-triton" package to worry about — the standalone `triton` package on PyPI is the same one PyTorch depends on. As of mid-2026, stable is Triton 3.7.x, and each PyTorch minor version pins a specific compatible Triton minor version. If you ever see a dependency-resolution error mentioning `triton==`, it means something in your environment is pinning a conflicting version — resolve it by either removing your explicit pin or upgrading PyTorch to match, not by fighting pip.

**Path B — standalone Triton (kernel development without a fixed PyTorch pin).**

```bash
pip install triton
```

**Path C — AMD ROCm backend.** Triton's ROCm support is upstream and mainstream at this point (both CUDA and ROCm backends get continued optimization work for flash-attention and linear-attention patterns, including on CDNA3 and Blackwell). Install the ROCm-built PyTorch wheel and Triton comes along compatible automatically:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0/
```

Requires ROCm 6.x+ installed and a compatible AMD Instinct or Radeon GPU; verify with `rocm-smi` / `amd-smi`.

**Path D — Intel XPU backend.** The Intel backend lives in a separate repo (`intel-xpu-backend-for-triton`) targeting Intel Data Center GPU Max series; install instructions there follow the same PyTorch-bundles-Triton pattern. We'll return to backend portability in Chapter 26 — for now, the NVIDIA path is what the rest of this curriculum assumes unless noted.

**Version/runtime requirements to sanity-check:** Python 3.10–3.14, glibc ≥ 2.27 (Ubuntu 18.04+/Debian 10+/RHEL 8+), and — obviously — a working GPU driver. Note that Triton itself *installs* fine without a GPU present (useful for CI or writing code on a laptop), but kernels can't execute without one. This is also where **interpreter mode** (Chapter 21) becomes genuinely useful: you can develop and debug kernel logic on a CPU-only machine before ever touching a GPU.

## 2.2 Verifying Your Setup

```python
import torch, triton
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```

If `torch.cuda.is_available()` is `False` on a machine you expect to have a GPU, that's a driver/CUDA-toolkit problem to fix *before* touching Triton — Triton inherits whatever device visibility PyTorch has.

## 2.3 The `@triton.jit` Compilation Lifecycle (Preview)

You'll get the full compiler-internals treatment in Chapter 14; here's the map you need to write and reason about your first kernels.

When you call a `@triton.jit`-decorated function, nothing compiles at *decoration* time — it compiles **lazily, on first call**, specialized to the actual arguments you pass:

```
Python function (your kernel source)
        │  parsed via Python's ast module
        ▼
Triton IR              — hardware-agnostic, tile-level operations
        ▼
TTGIR (Triton GPU IR)  — MLIR dialect; layouts, warps, shared memory assigned
        ▼
LLVM IR
        ▼
PTX  (NVIDIA)  /  AMDGCN (AMD)
        ▼
cubin / hsaco — the actual binary loaded onto the GPU
```

Two things matter about this pipeline right now:

1. **Compilation is per-specialization, not per-function.** The compiler generates different machine code depending on tensor dtypes, `tl.constexpr` values (like `BLOCK_SIZE`), and — for pointer arguments — whether shapes are divisible by the block size (this affects whether masking can be skipped). Call the same Python kernel function with `float16` tensors and then `float32` tensors, and you'll trigger *two separate compilations*, cached separately.
2. **The first call is slow; subsequent calls are fast.** JIT compilation takes anywhere from tens of milliseconds to a few seconds depending on kernel complexity. This is why benchmarking code always includes a warm-up call before timing (you'll see this pattern in every tutorial from Chapter 4 onward) — otherwise you're measuring compile time, not kernel performance.

## 2.4 Kernel Caching

Triton persists compiled kernels to disk so you don't pay the JIT cost on every process restart. The cache lives at `~/.triton/cache` by default (override with the `TRITON_CACHE_DIR` environment variable — useful in containerized or ephemeral-disk environments, and worth setting explicitly on shared GPU infrastructure so builds don't silently collide across users).

The cache key is derived from: the kernel's source (a hash of it — edit the function body and it *will* recompile, even if the file path is unchanged), the specialization (dtypes, `constexpr` values, divisibility properties of shapes), and the target device's compute capability. This means:

- Moving a cached kernel from an H100 to an A100 will *not* reuse the cache — different compute capability, different generated PTX/cubin.
- `@triton.autotune` (Chapter 8) layers its own cache on top of this: once the best `Config` is found for a given problem size, it's remembered so you don't re-run the full autotuning search on every launch.

You can inspect what's actually in the cache:

```bash
ls -la ~/.triton/cache
```

Each subdirectory corresponds to a distinct kernel specialization and typically contains the generated IR at each pipeline stage alongside the final binary — handy for the debugging techniques in Part IV and Part VI. If a kernel is behaving unexpectedly after you *think* you've changed its source, and you suspect a stale cache, `rm -rf ~/.triton/cache` is a legitimate first troubleshooting step.

## 2.5 Reading a Kernel Launch

Every tutorial kernel you touch from Chapter 4 onward is launched with syntax like this — worth being able to read before you see it a hundred times:

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
kernel[grid](x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE=1024, num_warps=4)
```

- **`grid`** — the number of program instances to launch, analogous to a CUDA grid of thread blocks. It's expressed as a function of the `meta` dict so it can depend on `constexpr` launch parameters (like `BLOCK_SIZE`) that might themselves be chosen by autotuning.
- **`triton.cdiv(a, b)`** — ceiling division; you'll use this constantly to compute "how many blocks do I need to cover `n` elements."
- **`num_warps`** — how many warps (32 threads each) the compiler should use per program instance; affects occupancy and register pressure (Chapter 15, Chapter 22).
- **`num_stages`** — software-pipelining depth for overlapping memory loads with compute (you'll meet this properly in Chapter 9's matmul tutorial).

## 2.6 Hands-On: Vector Addition, End to End

This is the official Triton "hello world." Run it, then go back and *inspect* what happened rather than just checking the output is correct.

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)

output_torch = x + y
output_triton = add(x, y)
print(f"Max difference: {torch.max(torch.abs(output_torch - output_triton))}")
assert torch.allclose(output_torch, output_triton)
print("OK — Triton and PyTorch agree.")
```

Now go beyond "it ran":

1. **Trigger a recompile and watch the cache grow.**
   ```bash
   rm -rf ~/.triton/cache
   python vector_add.py
   ls ~/.triton/cache   # one new entry — the fp32, BLOCK_SIZE=1024 specialization
   ```
   Change the tensor dtype to `torch.float16` and rerun — a *second* cache entry appears, confirming that specialization, not just the source file, determines the cache key.

2. **Time the JIT-compile tax.** Wrap the *first* call vs. a *subsequent* call in `time.perf_counter()` (with `torch.cuda.synchronize()` before and after each) and compare. You should see the first call take orders of magnitude longer — this is the compile-then-cache behavior from §2.3, and it's why every benchmarking utility you'll use starting in Chapter 7 always warms up before timing.

3. **Confirm device-specificity of the cache** (skip if you only have access to one GPU type): run the same script on two different GPU architectures pointed at the same `TRITON_CACHE_DIR`, and confirm two separate entries appear.

## 2.7 Common Environment Pitfalls

- **Triton/PyTorch version mismatch** — a pinned `triton==` in your project's requirements conflicting with what your PyTorch version expects. Fix by aligning versions, not by forcing an install.
- **"No CUDA GPUs are available" despite `nvidia-smi` working** — usually a container/venv not seeing the driver correctly, or a CUDA-toolkit/driver version mismatch. This is a PyTorch/CUDA problem upstream of Triton; verify `torch.cuda.is_available()` first.
- **Stale cache after editing a kernel inside a Jupyter notebook** — notebooks sometimes keep an old function object bound even after you re-run a cell; if behavior seems "stuck," restart the kernel (the Python one, not the Triton one) rather than debugging a ghost.
- **Mixing `conda` and `venv` Python environments** — a classic source of "wrong wheel installed" bugs (e.g., a `cp311` wheel under a `cp312` interpreter). If `import triton` succeeds but behaves strangely, check `which python` and `pip show triton` agree on the environment you think you're in.

## 2.8 Check Your Understanding

1. Why does calling the same `@triton.jit` function with `float16` and then `float32` tensors result in two separate compiled binaries rather than one?
2. What's the practical consequence of the cache key including compute capability, for someone who develops on one GPU and deploys on another?
3. In the launch line `kernel[grid](..., BLOCK_SIZE=1024, num_warps=4)`, which of these values are baked into the compiled binary (`constexpr`-like) and which are runtime arguments?
4. Why does every Triton benchmarking pattern you'll see include a warm-up call before timing?

## 2.9 What's Next

Chapter 3 stays with the vector-add kernel you just ran, but slows down on *why* it's written the way it is: `program_id`, the SPMD grid model, and how a Triton "program instance" maps conceptually onto the CUDA thread-block model you already know. From there we build up the full toolkit — pointers, masking, multi-dimensional indexing — needed to write kernels beyond a single flat vector.
