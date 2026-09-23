"""Chapter 6 numbers: simulate which 32-byte DRAM sectors a warp's load
touches, for different access patterns. This is a model of the hardware's
coalescing behavior (Turing/Ampere service global loads in 32-byte sectors),
not a live profile -- see Chapter 24 for reading the real thing off Nsight
Compute.
"""

SECTOR_BYTES = 32
WARP_SIZE = 32
FLOAT_BYTES = 4


def touched_sectors(addresses_bytes):
    """Given a list of byte addresses touched by a warp, return the set of
    32-byte-aligned sector indices those addresses fall into."""
    return {addr // SECTOR_BYTES for addr in addresses_bytes}


def simulate_warp_load(base_element, stride_elements, dtype_bytes=FLOAT_BYTES,
                        warp_size=WARP_SIZE, vector_width_elements=1):
    """One warp, one load instruction: thread t reads `vector_width_elements`
    contiguous elements starting at (base_element + t * stride_elements),
    each element `dtype_bytes` wide.
    Returns (sectors touched, bytes requested, bytes moved)."""
    addresses = []
    for t in range(warp_size):
        start = (base_element + t * stride_elements) * dtype_bytes
        for v in range(vector_width_elements):
            addresses.append(start + v * dtype_bytes)
    sectors = touched_sectors(addresses)
    bytes_requested = len(set(addresses)) * dtype_bytes
    bytes_moved = len(sectors) * SECTOR_BYTES
    return sectors, bytes_requested, bytes_moved


if __name__ == "__main__":
    print("== scalar float loads (4 bytes/thread), base aligned to a sector ==")
    print(f"{'stride':>8} {'sectors':>8} {'bytes moved':>12} {'bytes used':>11} {'waste':>8}")
    for stride in (1, 2, 4, 8, 32, 33, 128):
        sectors, used, moved = simulate_warp_load(base_element=0, stride_elements=stride)
        print(f"{stride:>8} {len(sectors):>8} {moved:>12} {used:>11} {moved / used:>7.2f}x")
    print("(waste caps at 8x once stride >= 8 elements -- each thread already")
    print(" owns a full 32-byte sector to itself; going wider can't cost more")
    print(" than one sector per thread for a 32-thread warp.)")

    print("\n== float4 vectorized loads (16 bytes/thread, contiguous groups of 4) ==")
    sectors, used, moved = simulate_warp_load(base_element=0, stride_elements=4,
                                               vector_width_elements=4)
    print(f"stride=4 elements (contiguous float4 groups): {len(sectors)} sectors, "
          f"{moved} bytes moved for {used} bytes used ({moved / used:.2f}x)")
    print("-> same 0-waste result as the scalar stride=1 case, just 4x the bytes")
    print("   per instruction and a quarter of the instructions issued.")

    print("\n== misaligned base address (start a few bytes off a sector boundary) ==")
    for base in (0, 1, 7, 8):
        sectors, used, moved = simulate_warp_load(base_element=base, stride_elements=1)
        print(f"base_element={base}: {len(sectors)} sectors, {moved} bytes moved "
              f"for {used} bytes used ({moved / used:.2f}x)")
