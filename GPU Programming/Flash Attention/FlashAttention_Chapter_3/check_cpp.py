"""Chapter 3: build the C++ tests, run them, and compare one dumped case against
the float64 NumPy reference on bit-identical inputs.  Run from this folder: python check_cpp.py"""
import os
import subprocess
import tempfile

import numpy as np

from tiled_attention import flash_attention_forward, naive_attention_np


def lcg(n, seed, scale):
    state, out = seed & 0xFFFFFFFF, []
    for _ in range(n):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        out.append(((state >> 8) / 16777216.0 * 2.0 - 1.0) * scale)
    return np.array(out)


here = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
flags = ["-std=c++17", "-O2", "-Wall", "-Wextra"]
for src in ("tiled_attention", "main_test"):
    subprocess.run(["g++", *flags, "-c", os.path.join(here, src + ".cpp"), "-I", here,
                    "-o", os.path.join(tmp, src + ".o")], check=True)
exe = os.path.join(tmp, "main_test")
subprocess.run(["g++", os.path.join(tmp, "tiled_attention.o"), os.path.join(tmp, "main_test.o"), "-o", exe], check=True)
res = subprocess.run([exe], capture_output=True, text=True)
lines = res.stdout.splitlines()
print(lines[0], f"(exit code {res.returncode})")

N, d = 37, 8
Q, K, V = (lcg(N * d, s, 1.0).reshape(N, d) for s in (1, 2, 3))
O_ref, L_ref, _ = naive_attention_np(Q, K, V, causal=True)
O_py, L_py = flash_attention_forward(Q, K, V, 8, 16, causal=True)
O_cpp = np.zeros((N, d)); L_cpp = np.zeros(N)
for line in lines[1:]:
    parts = line.split()
    if parts[0] == "O":
        O_cpp[int(parts[1]), int(parts[2])] = float(parts[3])
    elif parts[0] == "L":
        L_cpp[int(parts[1])] = float(parts[2])
print(f"C++ float32 tiled vs float64 naive reference : max|dO| = {np.abs(O_cpp - O_ref).max():.2e}  max|dL| = {np.abs(L_cpp - L_ref).max():.2e}")
print(f"Python float64 tiled vs float64 naive        : max|dO| = {np.abs(O_py - O_ref).max():.2e}  max|dL| = {np.abs(L_py - L_ref).max():.2e}")
