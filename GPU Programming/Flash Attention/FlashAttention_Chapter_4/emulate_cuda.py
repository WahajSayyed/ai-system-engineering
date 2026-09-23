"""Chapter 4: run the CUDA host code and kernels on the CPU through a small emulation layer.

This is a TEST HARNESS for machines without a GPU. It compiles the *unchanged* kernel file
(attention_kernels.cuh) and host code (attention_gpu.cu, main_gpu.cu) with g++, after one
mechanical rewrite: every  name<<<grid, block>>>(args);  becomes  EMU_LAUNCH(name, grid, block, args);
(g++ cannot parse the triple-chevron syntax).  See emulation/cuda_runtime.h for what is emulated
and, importantly, what is not.

Usage:  python emulate_cuda.py                run the test-suite
        python emulate_cuda.py stats          print launch geometry for a few sizes
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CH03 = os.path.join(HERE, "..", "ch03_code")
LAUNCH = re.compile(r"^(\s*)(\w+)<<<(.*?)>>>\((.*)\);\s*$", re.M)


def rewrite_launches(src):
    new, n = LAUNCH.subn(lambda m: f"{m.group(1)}EMU_LAUNCH({m.group(2)}, {m.group(3)}, {m.group(4)});", src)
    return new, n


def build(mutate=None, sanitize=False, main="main_gpu.cu"):
    """Copy sources to a temp dir (optionally mutating them), rewrite launches, compile with g++.
    mutate: dict {filename: [(old, new), ...]}. Returns (exe_path, n_launches_rewritten, compile_stderr)."""
    tmp = tempfile.mkdtemp()
    files = ["attention_kernels.cuh", "attention_gpu.cu", "attention_gpu.h", "cuda_check.h", "main_gpu.cu"]
    n_launch = 0
    for f in files:
        src = open(os.path.join(HERE, f)).read()
        for old, new in (mutate or {}).get(f, []):
            assert src.count(old) == 1, f"mutation target not found exactly once in {f}: {old!r}"
            src = src.replace(old, new)
        if f.endswith(".cu"):
            src, n = rewrite_launches(src)
            n_launch += n
            f = f[:-3] + ".cpp"
        open(os.path.join(tmp, f), "w").write(src)
    extra = []
    if main == "stats":
        extra = [os.path.join(HERE, "emulation", "launch_stats_main.cpp")]
        srcs = [os.path.join(tmp, "attention_gpu.cpp")] + extra
    else:
        srcs = [os.path.join(tmp, "main_gpu.cpp"), os.path.join(tmp, "attention_gpu.cpp"),
                os.path.join(CH03, "tiled_attention.cpp")]
    exe = os.path.join(tmp, "emu")
    flags = ["-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-DCPU_EMULATION"]
    if sanitize:
        flags += ["-fsanitize=address", "-fno-omit-frame-pointer"]
    cmd = ["g++", *flags, "-I", os.path.join(HERE, "emulation"), "-I", tmp, "-I", CH03, "-I", HERE, *srcs, "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("compile failed:\n" + r.stderr[:2000])
    return exe, n_launch, r.stderr


def run(exe):
    return subprocess.run([exe], capture_output=True, text=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    exe, n, warn = build(main="stats" if mode == "stats" else "main_gpu.cu")
    print(f"launch statements rewritten: {n}   compiler warnings: {len(warn.strip().splitlines())}")
    res = run(exe)
    print(res.stdout.rstrip())
    if res.stderr:
        print(res.stderr.rstrip())
    sys.exit(res.returncode)
