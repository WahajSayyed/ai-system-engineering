"""Chapter 5 numbers: how much HBM traffic tiling actually removes, and
whether a given TILE_WIDTH fits the shared-memory budget on the T4 / RTX 3090.
Pure arithmetic from the formulas in the chapter (SS2.2-2.3); nothing here is
measured -- see measure_memory.py-style profiling for that, later in the series.
"""

BYTES_PER_EL = 4  # fp32

# ---- 1. HBM reads to produce one TILE_WIDTH x TILE_WIDTH output tile -----------
# naive: every output element re-reads its own K-long row of A and column of B
#        from HBM independently        -> 2 * tile^2 * K element-reads
# tiled: each tile of A (and of B) is loaded from HBM once per K-tile step,
#        by tile^2 threads doing one read each, over K/tile steps
#                                       -> 2 * tile * K element-reads
print("== HBM element-reads to produce one TILE_WIDTH x TILE_WIDTH output tile ==")
print(f"{'TILE_WIDTH':>10} {'K':>6} {'naive reads':>14} {'tiled reads':>14} {'reduction':>10}")
for tile in (8, 16, 32):
    for K in (128, 1024):
        naive = 2 * tile * tile * K
        tiled = 2 * tile * K
        print(f"{tile:>10} {K:>6} {naive:>14,} {tiled:>14,} {naive / tiled:>9.1f}x")

# ---- 2. arithmetic intensity (FLOPs per byte moved from HBM) --------------------
# FLOPs for the whole tile (all K steps): 2 * tile^2 * K  (one mul + one add per
# output element per K step). Divide by bytes moved, from section 1 above.
print("\n== arithmetic intensity, fp32 (FLOPs per byte moved from HBM) ==")
print(f"{'TILE_WIDTH':>10} {'K':>6} {'naive FLOP/B':>14} {'tiled FLOP/B':>14} {'ratio':>8}")
for tile in (8, 16, 32):
    for K in (128, 1024):
        flops = 2 * tile * tile * K
        naive_bytes = 2 * tile * tile * K * BYTES_PER_EL
        tiled_bytes = 2 * tile * K * BYTES_PER_EL
        naive_intensity = flops / naive_bytes
        tiled_intensity = flops / tiled_bytes
        print(f"{tile:>10} {K:>6} {naive_intensity:>14.3f} {tiled_intensity:>14.2f} "
              f"{tiled_intensity / naive_intensity:>7.1f}x")
print("(naive intensity is constant at 0.25 FLOP/B regardless of tile or K --")
print(" there's no reuse to speak of; tiled intensity scales linearly with TILE_WIDTH.)")

# ---- 3. shared memory budget: does a tile fit, and how many blocks per SM? -----
print("\n== shared memory per block (As + Bs, fp32) vs SM budget ==")
GPUS = {
    #                    default/block (no opt-in)   max opt-in/SM
    "Tesla T4 (sm_75)":  dict(default_per_block=48 * 1024, max_per_sm=64 * 1024),
    "RTX 3090 (sm_86)":  dict(default_per_block=48 * 1024, max_per_sm=100 * 1024),
}
for tile in (8, 16, 32, 64, 128):
    bytes_used = 2 * tile * tile * BYTES_PER_EL   # As + Bs
    print(f"\nTILE_WIDTH={tile}: {bytes_used:,} bytes/block ({bytes_used / 1024:.2f} KiB)")
    for name, g in GPUS.items():
        fits_default = bytes_used <= g["default_per_block"]
        concurrent = g["max_per_sm"] // bytes_used
        print(f"  {name}: fits 48 KiB default = {fits_default!s:<5}  "
              f"max concurrent blocks/SM (shared-mem limited) = {concurrent}")
