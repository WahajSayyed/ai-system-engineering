"""Chapter 9: numpy grounding for the [B,H,N,D] stride arithmetic (SS2.2) and
the shared-memory budget (SS2.3), plus the int32-overflow example from
Exercise 4 -- all checkable with no GPU. The offset formula is checked
against NumPy's real memory layout, not just hand algebra.
"""
import numpy as np

INT32_MAX = 2**31 - 1


def bh_offset(batch, head, H, N, D):
    """Flat element offset of Q[batch, head, 0, 0] in a [B,H,N,D] row-major
    tensor -- mirrors flash_skeleton.cu's bhOffset computation exactly."""
    return (batch * H + head) * N * D


def verify_offset_formula(B, H, N, D):
    """Build a real [B,H,N,D] array and confirm bh_offset matches NumPy's
    own flat layout for every (batch, head) pair."""
    arr = np.arange(B * H * N * D).reshape(B, H, N, D)
    for b in range(B):
        for h in range(H):
            expected = int(arr[b, h, 0, 0])
            got = bh_offset(b, h, H, N, D)
            assert got == expected, (b, h, got, expected)
    return True


def shared_mem_bytes(Br, Bc, D):
    """Q tile + K tile + V tile + running-max (m) + running-sum (l), fp32 --
    mirrors flash_skeleton.cu's launchFlashSkeletonImpl sharedBytes exactly."""
    return (Br * D + 2 * Bc * D + 2 * Br) * 4


if __name__ == "__main__":
    print("== SS2.2: [B,H,N,D] offset formula, checked against real NumPy layout ==")
    B, H, N, D = 2, 4, 200, 64
    verify_offset_formula(B, H, N, D)
    print(f"verified for every (batch, head) in B={B}, H={H}, N={N}, D={D}")

    print("\n== SS2.1: grid layout / qStart values for N=200, Br=64 ==")
    Br = 64
    num_q_tiles = -(-N // Br)   # ceil division
    q_starts = [t * Br for t in range(num_q_tiles)]
    last_tile_valid_rows = N - q_starts[-1]
    print(f"numQTiles={num_q_tiles}, qStarts={q_starts}")
    print(f"last tile starts at {q_starts[-1]}, only {last_tile_valid_rows} valid rows "
          f"(Chapter 10 handles this boundary; this chapter just needs qStart right)")

    print("\n== SS2.3: shared-memory budget for a few (Br, Bc, D) configs ==")
    configs = [
        (64, 64, 64),    # the chapter's own config -- just barely over 48KB
        (64, 64, 32),
        (32, 32, 128),
        (128, 128, 64),  # Exercise 2
    ]
    for Br, Bc, D in configs:
        b = shared_mem_bytes(Br, Bc, D)
        print(f"Br={Br:>3} Bc={Bc:>3} D={D:>3}: {b:>6} bytes ({b / 1024:.2f} KiB) -- "
              f"fits 48KiB default: {b <= 48*1024}, "
              f"fits 64KiB (T4 opt-in): {b <= 64*1024}, "
              f"fits ~100KiB (RTX 3090 opt-in): {b <= 100*1024}")

    print("\n== Exercise 4: int32 overflow for a large-model shape ==")
    B, H, N, D = 32, 64, 8192, 128
    max_elem_offset = (B * H - 1) * N * D   # Python ints are arbitrary precision
    max_byte_offset = max_elem_offset * 4
    print(f"B={B}, H={H}, N={N}, D={D}")
    print(f"max element offset: {max_elem_offset:,}  (INT32_MAX = {INT32_MAX:,})")
    print(f"  -> overflows int32: {max_elem_offset > INT32_MAX}  "
          f"(margin: {INT32_MAX - max_elem_offset:,} elements -- uncomfortably thin)")
    print(f"max byte offset:    {max_byte_offset:,}")
    print(f"  -> overflows int32: {max_byte_offset > INT32_MAX}")
