"""Chapter 8: numpy twin of tiled_score.cu's A @ B^T tiling pattern,
parameterized the way Python naturally would be -- by ordinary function
arguments, not by generating a new function per shape.

This is the point of the chapter, stated in code: changing Br/Bc/D here
costs nothing but a different function call. Doing the same in tiled_score.cu
means the compiler generates an entirely separate, independently-optimized
kernel per combination -- see SS2.1.
"""
import numpy as np


def tiled_score_np(A, B, Br, Bc):
    """C[M,N] = A[M,D] @ B[N,D].T, tiled Br rows x Bc cols at a time.
    D is just A.shape[1] / B.shape[1] -- nothing about it needs to be
    fixed ahead of time, because numpy dispatches on shape at every call,
    not at "compile" time the way tiled_score.cu's template does.
    """
    M, D = A.shape
    N, D2 = B.shape
    assert D == D2
    C = np.zeros((M, N), dtype=A.dtype)
    for r0 in range(0, M, Br):
        r1 = min(r0 + Br, M)
        for c0 in range(0, N, Bc):
            c1 = min(c0 + Bc, N)
            C[r0:r1, c0:c1] = A[r0:r1, :] @ B[c0:c1, :].T
    return C


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("== same function, four different (Br, Bc, D) shapes -- ==")
    print("== in tiled_score.cu each of these is a separate compiled kernel ==")
    for D in (32, 64, 128, 63):
        M, N = 200, 150
        A = rng.standard_normal((M, D)).astype(np.float32)
        B = rng.standard_normal((N, D)).astype(np.float32)
        Br, Bc = (64, 64) if D != 128 else (32, 32)
        C = tiled_score_np(A, B, Br, Bc)
        ref = A @ B.T
        print(f"D={D:>3} Br={Br:>2} Bc={Bc:>2}: max abs diff vs A @ B.T = "
              f"{np.abs(C - ref).max():.2e}")

    print("\n== the shared-memory budget check from tiled_score.cu's static_assert ==")
    print("== (SS2.3): same arithmetic, just done here instead of at compile time ==")
    for D, Br, Bc in [(32, 64, 64), (64, 64, 64), (128, 32, 32), (63, 64, 64), (128, 64, 64)]:
        shared_bytes = (Br * D + Bc * D) * 4
        fits = shared_bytes <= 48 * 1024
        print(f"D={D:>3} Br={Br:>2} Bc={Bc:>2}: {shared_bytes:>6} bytes "
              f"({shared_bytes / 1024:.1f} KiB) -- fits 48KiB default: {fits}")
    print("-> the last row (D=128, Br=Bc=64) is exactly the combination Exercise 1")
    print("   asks you to try instantiating in tiled_score.cu -- it should fail to")
    print("   compile there for the same reason it prints False here.")
