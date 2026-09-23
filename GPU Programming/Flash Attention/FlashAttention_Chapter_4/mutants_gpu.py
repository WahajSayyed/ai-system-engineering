"""Chapter 4: do the GPU tests have teeth?  Break the kernels / host code on purpose, run the
emulated suite, and see whether it notices.  Run: python mutants_gpu.py"""
import os

from emulate_cuda import build, run

K, H = "attention_kernels.cuh", "attention_gpu.cu"

MAIN = "main_gpu.cu"

MUTANTS = [
    ("scores_kernel: drop the bounds check (built with AddressSanitizer)",
     {K: [("    if (i >= N || j >= N) return;                          // the grid is rounded UP: extra threads exit\n", "")]}, True),
    ("scores_kernel: drop the bounds check (plain build, no sanitizer)",
     {K: [("    if (i >= N || j >= N) return;                          // the grid is rounded UP: extra threads exit\n", "")]}, False),
    ("scores_kernel: swap the roles of x and y (i from x, j from y)",
     {K: [("const int j = blockIdx.x * blockDim.x + threadIdx.x;   // column of S: which key",
           "const int j = blockIdx.y * blockDim.y + threadIdx.y;   // column of S: which key"),
          ("const int i = blockIdx.y * blockDim.y + threadIdx.y;   // row of S:    which query",
           "const int i = blockIdx.x * blockDim.x + threadIdx.x;   // row of S:    which query")]}, False),
    ("pv_kernel: swap the roles of x and y (c from y, i from x)",
     {K: [("const int c = blockIdx.x * blockDim.x + threadIdx.x;   // output column, 0 .. d-1",
           "const int c = blockIdx.y * blockDim.y + threadIdx.y;   // output column, 0 .. d-1"),
          ("const int i = blockIdx.y * blockDim.y + threadIdx.y;   // output row (query)",
           "const int i = blockIdx.x * blockDim.x + threadIdx.x;   // output row (query)")]}, False),
    ("host: ceil_div rounds DOWN (a / b)",
     {H: [("return (a + b - 1) / b;", "return a / b;")]}, False),
    ("host: ceil_div rounds DOWN, and the test grid has only N >= 130, d >= 17",
     {H: [("return (a + b - 1) / b;", "return a / b;")],
      MAIN: [("const int Ns[] = {1, 5, 37, 64, 100, 257};", "const int Ns[] = {130, 257};"),
             ("const int ds[] = {1, 8, 33, 64};", "const int ds[] = {17, 33, 64};")]}, False),
    ("host: 64 x 32 = 2048 threads per block",
     {H: [("const dim3 block2d(16, 16);", "const dim3 block2d(64, 32);")]}, False),
    ("scores_kernel: forget the causal mask",
     {K: [("if (causal && j > i) {", "if (false) {")]}, False),
    ("softmax_rows_kernel: exp(s) instead of exp(s - m), L left as it was",
     {K: [("s[j] = expf(s[j] - m);", "s[j] = expf(s[j]);")]}, False),
    ("softmax_rows_kernel: no max at all (m = 0), so P = exp(s)/sum exp(s) and L stay correct in exact arithmetic",
     {K: [("    float m = -INFINITY;\n    for (int j = 0; j < N; ++j) m = fmaxf(m, s[j]);        // pass 1: max\n",
           "    float m = 0.0f;                                        // (bug) no max subtraction\n")]}, False),
    ("host: copy back only L's byte count for O",
     {H: [("CUDA_CHECK(cudaMemcpy(hO, dO, bytes_nd, cudaMemcpyDeviceToHost));",
           "CUDA_CHECK(cudaMemcpy(hO, dO, bytes_n, cudaMemcpyDeviceToHost));")]}, False),
]

def summarise(res):
    lines = [l for l in (res.stdout + res.stderr).splitlines() if l.strip()]
    keep = [l.strip() for l in lines if "configurations" in l or "FAIL" in l or "CUDA error" in l or "error:" in l
            or "ERROR: AddressSanitizer" in l or "call :" in l]
    nfail = sum(1 for l in lines if l.strip().startswith("FAIL"))
    out = "; ".join(keep[:3]) if keep else "(no summary line)"
    return out[:300] + (f"   [{nfail} FAIL lines shown by the test]" if nfail else "") + f"   [exit {res.returncode}]"

exe, _, _ = build()
print("unmodified:", summarise(run(exe)), "\n")
for name, mutate, sanitize in MUTANTS:
    exe, _, _ = build(mutate=mutate, sanitize=sanitize)
    print(name, "\n   ->", summarise(run(exe)), "\n")
