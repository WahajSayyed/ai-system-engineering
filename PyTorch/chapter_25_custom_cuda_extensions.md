# Chapter 25: Custom CUDA Extensions in PyTorch

## Introduction

Every operation used so far — from a basic tensor add in Chapter 1 to a full transformer block in Chapter 13 — has relied on PyTorch's built-in kernels. Those kernels cover the overwhelming majority of what deep learning needs, and `torch.compile` (Chapter 21) already fuses and optimizes combinations of them automatically. But occasionally you need something genuinely custom: a novel operation with no standard PyTorch equivalent, or an existing operation that a hand-tuned kernel could meaningfully outperform for your specific shapes and access patterns. This chapter connects your PyTorch and CUDA curricula directly, showing how to write a custom CUDA kernel and wire it into PyTorch as a normal, differentiable operation.

By the end of this chapter you'll be able to:
- Explain when writing a custom CUDA extension is actually justified
- Build a PyTorch C++/CUDA extension using `torch.utils.cpp_extension`
- Write both the CUDA kernel and the PyTorch-facing wrapper correctly
- Wire a custom kernel into autograd using `torch.autograd.Function` (Chapter 16)
- Benchmark a custom kernel fairly against the built-in PyTorch equivalent

---

## 25.1 When Is This Actually Justified?

This deserves the same upfront honesty as Chapter 16's opening on custom autograd functions, because the bar here is even higher: **writing and maintaining a custom CUDA extension is a real ongoing cost** — it needs to be compiled for each target GPU architecture, it bypasses PyTorch's extensive built-in testing and optimization work, and it's one more thing to debug when something goes wrong. Before writing one, it's worth confirming, using the profiling tools from Chapter 20, that:

1. The operation you need is actually a measured bottleneck, not just something that *feels* like it should be slow.
2. `torch.compile` (Chapter 21) doesn't already fuse the relevant operations into something sufficiently fast — checking this first often eliminates the need for a custom kernel entirely.
3. No existing PyTorch operation or combination of operations already implements what you need efficiently — reimplementing something PyTorch already does well, just because you *can* write CUDA, rarely pays off.

Given your CUDA curriculum's progression through Triton kernels and the TinyCUDA transformer encoder capstone, you're well-positioned to recognize genuinely justified cases: a novel fused operation specific to a research idea, an operation with an access pattern PyTorch's generic kernels handle suboptimally, or — directly relevant to your conveyor sorting system — a custom preprocessing or post-processing kernel tailored to a specific sensor data format that doesn't map cleanly onto standard tensor operations.

---

## 25.2 The Two Pieces: CUDA Kernel and C++ Binding

A PyTorch CUDA extension has two files: a `.cu` file containing the actual CUDA kernel (the GPU-side code, following the same CUDA C++ conventions from your CUDA curriculum) and a C++ file that binds that kernel into a Python-callable function using `pybind11` (which `torch.utils.cpp_extension` sets up for you automatically).

### The CUDA kernel: `muladd_kernel.cu`

A minimal, illustrative example — a fused multiply-add operation (`out = a * b + c`, elementwise) — chosen for its simplicity, so the PyTorch integration mechanics are clear without a complex kernel obscuring them:

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void muladd_kernel(
    const float* a, const float* b, const float* c,
    float* out, int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx] + c[idx];
    }
}

torch::Tensor muladd_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    auto out = torch::empty_like(a);
    int64_t n = a.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    muladd_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        out.data_ptr<float>(), n
    );

    return out;
}
```

If you've worked through the CUDA curriculum's kernel-writing chapters, `muladd_kernel` itself should look entirely familiar — standard thread-index computation, bounds checking, elementwise work. What's new here is `torch::Tensor` (PyTorch's C++ tensor type, mirroring the Python `torch.Tensor` API), `data_ptr<float>()` (extracting a raw pointer to the tensor's underlying storage — directly connecting to Chapter 2's discussion of tensors as metadata wrapped around a flat storage buffer, now accessed from the C++ side), and `torch::empty_like(a)` (the C++ equivalent of Chapter 1's `torch.empty_like`).

### The C++ binding: registers the function for Python

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("muladd", &muladd_cuda, "Fused multiply-add (CUDA)");
}
```

This macro (handled largely automatically by `torch.utils.cpp_extension`'s build process) exposes `muladd_cuda` as a Python-callable function named `muladd` inside the compiled extension module — the bridge that makes a C++/CUDA function callable from ordinary Python code, exactly like any other Python function.

---

## 25.3 Building the Extension with `torch.utils.cpp_extension`

PyTorch provides two ways to build an extension: ahead-of-time (via a `setup.py`, producing an installable package) and just-in-time (JIT), which compiles on first import — much faster to iterate with during development, and the approach used here.

```python
# build_and_load.py
from torch.utils.cpp_extension import load

muladd_extension = load(
    name="muladd_extension",
    sources=["muladd_kernel.cu"],
    verbose=True,   # prints compilation commands -- useful the first time, for debugging build issues
)

import torch

a = torch.rand(1000, device="cuda")
b = torch.rand(1000, device="cuda")
c = torch.rand(1000, device="cuda")

result = muladd_extension.muladd(a, b, c)
expected = a * b + c
print(torch.allclose(result, expected))   # True
```

`load()` compiles `muladd_kernel.cu` using `nvcc` (the CUDA compiler — the same one used throughout your CUDA curriculum) the first time this script runs, caching the compiled result for subsequent runs unless the source changes. This JIT approach is the right choice for development and experimentation; a `setup.py`-based ahead-of-time build (using `torch.utils.cpp_extension.CUDAExtension` and `BuildExtension` instead of `load()`) is more appropriate once an extension is stable and needs to be distributed or installed as a proper package — worth knowing exists, though the JIT workflow covers essentially all of this chapter's learning goals.

---

## 25.4 Wiring the Kernel into Autograd

The extension above works, but only for the forward computation — calling `muladd_extension.muladd(a, b, c)` directly on tensors with `requires_grad=True` won't produce a `grad_fn` (Chapter 3, Section 3.1), since PyTorch's autograd engine has no way to know how to differentiate through an opaque, externally-compiled function unless explicitly told. This is precisely the situation Chapter 16's `torch.autograd.Function` exists for, and it's the single most direct connection between this chapter and that one.

```python
class MulAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c):
        ctx.save_for_backward(a, b)   # Chapter 16, Section 16.2 -- needed for the backward pass
        return muladd_extension.muladd(a, b, c)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # out = a*b + c  ->  d(out)/da = b, d(out)/db = a, d(out)/dc = 1
        grad_a = grad_output * b
        grad_b = grad_output * a
        grad_c = grad_output
        return grad_a, grad_b, grad_c

muladd = MulAdd.apply   # Chapter 16, Section 16.2 -- always call via .apply

a = torch.rand(1000, device="cuda", requires_grad=True)
b = torch.rand(1000, device="cuda", requires_grad=True)
c = torch.rand(1000, device="cuda", requires_grad=True)

out = muladd(a, b, c)
loss = out.sum()
loss.backward()

print(a.grad is not None, b.grad is not None, c.grad is not None)   # True True True
```

Note the backward pass here is written directly in Python/PyTorch (`grad_output * b`, etc.), not as a second custom CUDA kernel — for this simple, elementwise example, PyTorch's own built-in multiplication is already an efficient CUDA kernel, so there's no benefit to hand-writing a separate backward kernel. For a genuinely performance-critical operation, you'd typically write a second `.cu` kernel implementing the backward pass as well, following the exact same pattern as `muladd_kernel` above, and call it from `backward()` the same way `forward()` calls the forward kernel — the mechanics are identical in both directions, just applied to different mathematical operations.

This is the complete pattern: **a custom CUDA kernel handles the actual computation; a `torch.autograd.Function` wraps it to make it differentiable; and the model code that uses it looks exactly like every other differentiable operation used throughout this book**, composable with `nn.Module` (Chapter 5), included transparently in a training loop (Chapter 8), and compatible with everything else in the PyTorch ecosystem covered so far.

---

## 25.5 Verifying Correctness and Benchmarking Fairly

Two checks worth running on any custom kernel before trusting it, both directly reusing tools from earlier chapters:

### Correctness: `gradcheck`

```python
from torch.autograd import gradcheck

a = torch.rand(100, device="cuda", dtype=torch.float64, requires_grad=True)
b = torch.rand(100, device="cuda", dtype=torch.float64, requires_grad=True)
c = torch.rand(100, device="cuda", dtype=torch.float64, requires_grad=True)

# Note: this requires the kernel to also support float64 inputs -- worth handling
# via a templated kernel or a dtype dispatch, briefly discussed in Section 25.6.
test_passed = gradcheck(MulAdd.apply, (a, b, c))
print(test_passed)
```

This is exactly the `gradcheck` verification from Chapter 16, Section 16.5, applied here to confirm the hand-written `backward()` correctly matches the kernel's actual forward computation — just as essential here as for a pure-Python custom autograd function, arguably more so, since a bug in a CUDA kernel can be considerably harder to spot by inspection than a bug in equivalent Python code.

### Fair benchmarking

```python
import time

def benchmark(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):   # Chapter 21, Section 21.1 -- warm-up before timing
        fn(*args)
    torch.cuda.synchronize()   # Chapter 20, Section 20.4 -- required for accurate GPU timing

    start = time.time()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000   # ms per call

a = torch.rand(10_000_000, device="cuda")
b = torch.rand(10_000_000, device="cuda")
c = torch.rand(10_000_000, device="cuda")

custom_time = benchmark(lambda: muladd_extension.muladd(a, b, c))
builtin_time = benchmark(lambda: a * b + c)

print(f"Custom kernel: {custom_time:.4f}ms")
print(f"Built-in PyTorch: {builtin_time:.4f}ms")
```

This benchmark deliberately reuses both the warm-up principle from Chapter 21 (compilation/first-call overhead shouldn't be included in a steady-state measurement) and the synchronization requirement from Chapter 20 (CUDA operations launch asynchronously, so unsynchronized CPU-side timing is meaningless). **For this specific toy example, don't expect the custom kernel to win** — a simple fused multiply-add is exactly the kind of operation PyTorch's built-in elementwise kernels already handle about as efficiently as a straightforward custom implementation can; the value of writing it yourself here is purely pedagogical, establishing the full pattern before applying it to an operation where a genuine, measured speedup is actually achievable.

---

## 25.6 Handling Multiple Data Types

The kernel in Section 25.2 hardcodes `float`, meaning it silently produces wrong results (or crashes) if called with a `float64` or `float16` tensor. Real extensions use `AT_DISPATCH_FLOATING_TYPES` (or similar dispatch macros) to generate and select the correct kernel instantiation based on the input tensor's actual dtype, connecting directly back to Chapter 1's emphasis on always being deliberate about tensor dtype:

```cpp
#include <ATen/Dispatch.h>

torch::Tensor muladd_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    auto out = torch::empty_like(a);
    int64_t n = a.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "muladd_cuda", ([&] {
        muladd_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), n
        );
    }));

    return out;
}
```

This requires the kernel itself to become a C++ template over `scalar_t` rather than hardcoding `float` — a direct application of standard C++ templating, layered on top of the CUDA kernel-writing skills from your CUDA curriculum, to make an extension robust across the dtype considerations that matter throughout PyTorch (Chapter 1's dtype discussion, Chapter 15's mixed precision, and so on).

---

## Summary

- Custom CUDA extensions are justified when profiling (Chapter 20) confirms a genuine bottleneck that neither built-in PyTorch operations nor `torch.compile` (Chapter 21) already address efficiently — not as a default choice simply because it's possible.
- A PyTorch CUDA extension pairs a `.cu` kernel (standard CUDA C++, as in your CUDA curriculum) with a `pybind11` binding, built via `torch.utils.cpp_extension.load()` for fast JIT iteration during development.
- A raw CUDA kernel has no autograd awareness on its own — wrap it in a `torch.autograd.Function` (Chapter 16) to make it a proper, differentiable PyTorch operation, with a hand-written or kernel-backed `backward()`.
- Verify correctness with `gradcheck` and benchmark fairly using warm-up and `torch.cuda.synchronize()` (Chapters 16, 20, 21) — a custom kernel isn't guaranteed to beat PyTorch's built-in operations, especially for simple elementwise cases.
- `AT_DISPATCH_FLOATING_TYPES` and C++ templating make an extension robust across multiple dtypes, rather than hardcoded to one.

## Exercises

1. Build and load the `muladd` extension from this chapter, verify its output against `a * b + c` using `torch.allclose`, and confirm `gradcheck` passes on the wrapped `MulAdd.apply` autograd function.
2. Extend `muladd_kernel` to support `float64` using `AT_DISPATCH_FLOATING_TYPES` and C++ templating (Section 25.6), and confirm `gradcheck` (which requires `float64` inputs) now passes without needing a separate workaround.
3. Write a second, genuinely non-trivial CUDA kernel — for example, a fused "leaky ReLU with a learnable negative slope" operation — including both forward and backward kernels, wired into autograd via `torch.autograd.Function`, and verified with `gradcheck`.
4. Benchmark your kernel from Exercise 3 against an equivalent implemented using standard PyTorch operations composed together, using the fair-benchmarking methodology from Section 25.5, and report whether (and by how much) the custom kernel is faster, slower, or comparable.

**Next:** Chapter 26 covers quantization — post-training quantization, quantization-aware training, and `torch.ao` — reducing model size and inference cost by representing weights and activations with fewer bits, building directly on this chapter's low-level tensor manipulation and Chapter 16's straight-through estimator.
