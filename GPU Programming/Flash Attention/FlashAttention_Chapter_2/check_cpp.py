"""Chapter 2: compile online_softmax.cpp, run it, and compare its float32 output
with the float64 Python implementations on bit-identical inputs.

Run from this folder:  python check_cpp.py      (needs g++)
"""
import os
import re
import subprocess
import tempfile

import numpy as np

from online_softmax import NEG_INF, online_stats, softmax_3pass, stats_blocked, weighted_sum_online


def lcg(n, seed, scale):
    """Same generator as lcg_fill() in the C++ file. The scales used are powers
    of two, so every value is exactly representable in float32: identical inputs."""
    state, out = seed & 0xFFFFFFFF, []
    for _ in range(n):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF      # wraps modulo 2**32
        u = (state >> 8) / 16777216.0
        out.append((2.0 * u - 1.0) * scale)
    return out


here = os.path.dirname(os.path.abspath(__file__))
exe = os.path.join(tempfile.mkdtemp(), "online_softmax")
subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-o", exe,
                os.path.join(here, "online_softmax.cpp")], check=True)
out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
print(out.rstrip(), "\n")

x = lcg(16, 12345, 4.0)
v = lcg(16, 777, 2.0)

y3 = np.array([float(l.split()[2]) for l in out.splitlines() if l.startswith("y3 ")])
y2 = np.array([float(l.split()[2]) for l in out.splitlines() if l.startswith("y2 ")])
ref = np.array(softmax_3pass(x))
print("C++ float32 vs Python float64 (same inputs)")
print(f"  softmax, 3-pass : max abs diff = {np.abs(y3 - ref).max():.2e}")
print(f"  softmax, online : max abs diff = {np.abs(y2 - ref).max():.2e}")

m64, d64 = online_stats(x)
for line in out.splitlines():
    mt = re.match(r"blocked block=\s*(\d+)\s+m=(\S+)\s+d=(\S+)", line)
    if mt:
        block, m, d = int(mt.group(1)), float(mt.group(2)), float(mt.group(3))
        mb, db = stats_blocked(x, block)
        print(f"  blocked block={block:>2}: |m - m64| = {abs(m - mb):.1e}, |d - d64| = {abs(d - db):.1e}")

wa = float(re.search(r"one-pass = (\S+)", out).group(1))
print(f"  weighted average: |C++ - Python| = {abs(wa - weighted_sum_online(x, [np.array(t) for t in v])):.2e}")
xm = [NEG_INF] * 3 + x[3:]
wm = float(re.search(r"masked start:\s+one-pass = (\S+)", out).group(1))
print(f"  masked start    : |C++ - Python| = {abs(wm - weighted_sum_online(xm, [np.array(t) for t in v])):.2e}")
