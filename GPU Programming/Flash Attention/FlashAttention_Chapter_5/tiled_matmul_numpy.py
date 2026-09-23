"""Chapter 5: numpy twin of the tiled matmul in tiled_matmul.cu.

Same blocking pattern as the CUDA kernel -- outer loop over output tiles,
inner loop over K tiles, load-then-accumulate -- written with plain numpy
slicing so the *algorithm* can be checked before any CUDA syntax gets in
the way. This sandbox has no torch / GPU, same as Chapter 1's verify_numpy.py.
"""
import numpy as np


def tiled_matmul_np(A, B, tile=16):
    """C[M,N] = A[M,K] @ B[K,N], computed tile by tile.

    Mirrors the CUDA kernel's loop structure exactly:
    for each (row_tile, col_tile) of the output:
        acc = 0
        for each k_tile along the shared K dimension:
            load A[row_tile, k_tile] and B[k_tile, col_tile]   (the __shared__ load)
            acc += that A tile @ that B tile                    (the inner k-loop)
        write acc into C[row_tile, col_tile]
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"inner dimensions must match, got {K} and {K2}"
    C = np.zeros((M, N), dtype=A.dtype)

    for row0 in range(0, M, tile):
        row1 = min(row0 + tile, M)
        for col0 in range(0, N, tile):
            col1 = min(col0 + tile, N)
            acc = np.zeros((row1 - row0, col1 - col0), dtype=A.dtype)
            for k0 in range(0, K, tile):
                k1 = min(k0 + tile, K)
                a_tile = A[row0:row1, k0:k1]     # the "As" load
                b_tile = B[k0:k1, col0:col1]     # the "Bs" load
                acc += a_tile @ b_tile           # the inner k-loop, done as one matmul
            C[row0:row1, col0:col1] = acc
    return C


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # small worked example, easy to eyeball
    print("== tiny worked example (tile=2, smaller than the matrices) ==")
    A = rng.standard_normal((5, 3)).round(2)
    B = rng.standard_normal((3, 4)).round(2)
    C = tiled_matmul_np(A, B, tile=2)
    ref = A @ B
    print("A @ B (numpy):\n", ref)
    print("tiled_matmul_np:\n", C)
    print("max abs diff:", np.abs(C - ref).max())

    # larger, non-multiple-of-tile sizes -- same shape the CUDA test uses
    print("\n== larger, non-multiple-of-tile sizes (matches test_tiled_matmul.cpp) ==")
    M, K, N, tile = 250, 130, 300, 16
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C = tiled_matmul_np(A, B, tile=tile)
    ref = A @ B
    print(f"M={M} K={K} N={N} tile={tile}")
    print("max abs diff vs np.matmul:", np.abs(C - ref).max())

    # tile size doesn't change the answer, only how the work is grouped
    print("\n== tile size shouldn't change the result, only the traffic pattern ==")
    for t in (1, 3, 16, 64, 300):
        C_t = tiled_matmul_np(A, B, tile=t)
        print(f"tile={t:>4}: max abs diff vs np.matmul = {np.abs(C_t - ref).max():.2e}")
