"""Chapter 4: compile-check the CUDA code on a machine WITHOUT nvcc or a GPU.

This is a verification aid for the sandbox this chapter was written in, not something you need
if you have the CUDA toolkit (there, just run the nvcc commands in the chapter).  It uses:

  1. clang's CUDA front end (host-only pass) on attention_gpu.cu and main_gpu.cu;
  2. clang's CUDA front end (device-only pass) -> PTX, then NVIDIA's ptxas -> machine code;
  3. NVRTC (NVIDIA's own runtime compiler) on the kernel file -> PTX, then ptxas.

Paths are taken from environment variables (defaults are the sandbox's):
  CUDA_CHECK_ROOT  a directory with bin/ptxas, nvvm/libdevice, include/ (assembled from pip wheels)
  NVRTC_LIB        path to libnvrtc.so
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CH03 = os.path.join(HERE, "..", "ch03_code")
ROOT = os.environ.get("CUDA_CHECK_ROOT", "/home/claude/cuda_tc/cuda_root")
NVRTC = os.environ.get("NVRTC_LIB", "/home/claude/cuda_tc/root/nvidia/cuda_nvrtc/lib/libnvrtc.so.12")
PTXAS = os.path.join(ROOT, "bin", "ptxas")
CLANG = ["clang++-18", "-x", "cuda", f"--cuda-path={ROOT}", "-Wno-unknown-cuda-version", "-std=c++17",
         "-Wall", "-Wextra", "-I", HERE, "-I", CH03]
tmp = tempfile.mkdtemp()


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def ptxas_usage(ptx_path, arch):
    """Return {kernel_name: (registers, spill_stores, spill_loads)} from `ptxas -v`."""
    r = sh([PTXAS, "-v", f"-arch={arch}", ptx_path, "-o", os.devnull])
    out, cur, res = r.stderr, None, {}
    for line in out.splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", line)
        if m:
            cur = m.group(1)
            res[cur] = [None, None, None]
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur:
            res[cur][1], res[cur][2] = int(m.group(1)), int(m.group(2))
        m = re.search(r"Used (\d+) registers", line)
        if m and cur:
            res[cur][0] = int(m.group(1))
    if r.returncode:
        print("ptxas failed:", out[:500])
    return res


def short_name(mangled):
    """_Z13scores_kernelPKfS0_Pfiifb -> scores_kernel  (the digits after _Z give the name length)"""
    m = re.match(r"_Z(\d+)(.*)", mangled)
    return m.group(2)[: int(m.group(1))]


def main():
    print("== 1. clang CUDA front end, host-only pass (syntax + types of the host code) ==")
    for f in ("attention_gpu.cu", "main_gpu.cu"):
        r = sh(CLANG + ["--cuda-host-only", "--cuda-gpu-arch=sm_86", "-c", os.path.join(HERE, f), "-o", os.path.join(tmp, f + ".o")])
        warn = [l for l in r.stderr.splitlines() if "warning:" in l or "error:" in l]
        print(f"  {f:<18} exit {r.returncode}   diagnostics: {len(warn)}")
        for l in warn[:5]:
            print("     ", l)

    print("\n== 2. clang CUDA front end, device-only pass -> PTX -> ptxas (real NVIDIA assembler) ==")
    results = {}
    for arch in ("sm_75", "sm_86"):
        ptx = os.path.join(tmp, f"clang_{arch}.ptx")
        r = sh(CLANG + ["--cuda-device-only", f"--cuda-gpu-arch={arch}", "-S", os.path.join(HERE, "attention_gpu.cu"), "-o", ptx])
        print(f"  {arch}: clang exit {r.returncode}, diagnostics {len([l for l in r.stderr.splitlines() if 'warning:' in l or 'error:' in l])}")
        results[("clang", arch)] = ptxas_usage(ptx, arch)

    print("\n== 3. NVRTC (NVIDIA's runtime compiler) on attention_kernels.cuh -> PTX -> ptxas ==")
    lib = ctypes.CDLL(NVRTC)
    major, minor = ctypes.c_int(), ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    print(f"  NVRTC version {major.value}.{minor.value}")
    src = open(os.path.join(HERE, "attention_kernels.cuh")).read()
    src = re.sub(r"^#include.*$", "", src, flags=re.M)              # NVRTC has no <cmath>; math builtins are predefined
    for arch in ("75", "86"):
        prog = ctypes.c_void_p()
        assert lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"kernels.cu", 0, None, None) == 0
        opts = [f"--gpu-architecture=compute_{arch}".encode(), b"--std=c++17", b"-DINFINITY=__int_as_float(0x7f800000)"]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
        logsz = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(logsz))
        log = ctypes.create_string_buffer(logsz.value)
        lib.nvrtcGetProgramLog(prog, log)
        print(f"  compute_{arch}: nvrtcCompileProgram returned {rc}; log: {log.value.decode().strip() or '(empty)'}")
        if rc:
            continue
        psz = ctypes.c_size_t()
        lib.nvrtcGetPTXSize(prog, ctypes.byref(psz))
        ptx = ctypes.create_string_buffer(psz.value)
        lib.nvrtcGetPTX(prog, ptx)
        path = os.path.join(tmp, f"nvrtc_{arch}.ptx")
        open(path, "wb").write(ptx.value)
        results[("nvrtc", f"sm_{arch}")] = ptxas_usage(path, f"sm_{arch}")

    print("\n== resource usage per kernel, from ptxas -v (registers per thread; spill bytes) ==")
    print(f"  {'source of PTX':<8} {'arch':<6} {'kernel':<22} {'regs':>5} {'spill st/ld':>12}")
    for (who, arch), table in sorted(results.items()):
        for name, (regs, ss, sl) in sorted(table.items()):
            print(f"  {who:<8} {arch:<6} {short_name(name):<22} {regs:>5} {ss:>5}/{sl:<5}")


if __name__ == "__main__":
    main()
