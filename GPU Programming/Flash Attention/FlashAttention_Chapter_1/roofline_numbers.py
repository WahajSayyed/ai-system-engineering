"""Chapter 1 numbers: score-matrix size, HBM/DRAM traffic, arithmetic intensity, ridge points.
Everything is derived from formulas + datasheet peaks; nothing here is a measurement."""
GiB = 1024**3
MiB = 1024**2

# ---- datasheet peaks (see chapter for sources) ---------------------------------
GPUS = {
    #            mem GiB   DRAM GB/s   fp32 TFLOPS   fp16-tensor(fp32 acc) TFLOPS
    "RTX 3090": dict(mem=24, bw=936, fp32=35.6, tc=71),   # 3090 tc: dense, fp32 accumulate (GA102 whitepaper figures)
    "Tesla T4": dict(mem=16, bw=300, fp32=8.1,  tc=65),   # NVIDIA T4 datasheet (300 GB/s; product page says 320+)
}

# ---- 1. one N x N score matrix per (batch, head) --------------------------------
print("== one S (or P) matrix per (batch, head) ==")
print(f"{'N':>7} {'fp16':>10} {'fp32':>10}")
for n in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    print(f"{n:>7} {n*n*2/MiB:>8.1f}Mi {n*n*4/MiB:>8.1f}Mi")

# ---- 2. does it fit? B=1, H=32 heads, fp16, ~2 live N x N tensors ----------------
print("\n== B=1, H=32, fp16: S alone / ~2 live N x N tensors, vs. GPU memory ==")
H = 32
for n in [2048, 4096, 8192, 12288, 16384]:
    s = H*n*n*2/GiB
    row = f"N={n:>6}: S={s:6.2f} GiB, 2xS={2*s:6.2f} GiB"
    for name, g in GPUS.items():
        row += f" | {name}: {'fits' if 2*s < g['mem'] else 'OOM '}"
    print(row)

# ---- 3. traffic + intensity for the paper's GPT-2-medium shape ---------------------
B, H, N, d, bytes_el = 64, 16, 1024, 64, 2
bh = B*H
flops = 4*N*N*d*bh                                   # two matmuls, 2*N*N*d each
alg0_el   = 4*N*N + 4*N*d                            # paper's Algorithm 0 (minimal standard attention)
eager_el  = 6*N*N + 4*N*d                            # our eager code, no mask: matmul, scale, softmax, matmul
floor_el  = 4*N*d                                    # read Q,K,V + write O once (hard floor)
print(f"\n== forward only, B={B} H={H} N={N} d={d} fp16 ==")
print(f"FLOPs (2 matmuls)          : {flops/1e9:8.1f} GFLOP")
for label, el in [("Algorithm 0 (paper)", alg0_el), ("our eager code", eager_el), ("floor: Q,K,V in, O out", floor_el)]:
    by = el*bytes_el*bh
    print(f"{label:<26}: {by/1e9:8.2f} GB moved  -> intensity {flops/by:7.1f} FLOP/B")
print(f"ratio Algorithm 0 / floor  : {alg0_el/floor_el:5.1f}x")

# ---- 4. ridge points and lower-bound times -----------------------------------------
print("\n== ridge point = peak FLOP/s / peak B/s ==")
for name, g in GPUS.items():
    print(f"{name}: fp32 ridge = {g['fp32']*1e12/(g['bw']*1e9):6.1f} FLOP/B,  tensor-core ridge = {g['tc']*1e12/(g['bw']*1e9):6.1f} FLOP/B")

print("\n== lower-bound time for the same forward pass (datasheet peaks; real kernels are slower) ==")
for name, g in GPUS.items():
    t_mem_alg0  = alg0_el*bytes_el*bh/(g['bw']*1e9)*1e3
    t_mem_eager = eager_el*bytes_el*bh/(g['bw']*1e9)*1e3
    t_cmp_fp32  = flops/(g['fp32']*1e12)*1e3
    t_cmp_tc    = flops/(g['tc']*1e12)*1e3
    print(f"{name}: memory floor (Alg 0) {t_mem_alg0:6.2f} ms | (eager) {t_mem_eager:6.2f} ms | compute @fp32 {t_cmp_fp32:5.2f} ms | compute @tensor {t_cmp_tc:5.2f} ms")

# ---- 5. intensity vs N: naive ~ d/2 (flat), floor = N/2 (grows) ---------------------
print("\n== intensity vs N (fp16, d=64) ==")
print(f"{'N':>6} {'naive (Alg 0)':>14} {'floor':>8}")
for n in [128, 256, 512, 1024, 2048, 4096]:
    fl = 4*n*n*d
    print(f"{n:>6} {fl/((4*n*n+4*n*d)*2):>14.1f} {fl/(4*n*d*2):>8.1f}")
