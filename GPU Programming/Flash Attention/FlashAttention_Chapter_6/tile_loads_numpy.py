"""Chapter 6: numpy grounding for alignment and vectorized-load grouping.
No CUDA involved -- this is the arithmetic behind SS2.3 and Exercises 1-2,
checkable without a GPU.
"""
import numpy as np

FLOAT_BYTES = 4
VECTOR_BYTES = 16   # float4 = 4 floats x 4 bytes


def row_start_offsets_bytes(N, d):
    """Byte offset of the start of each row in a flat, row-major [N, d] fp32 buffer."""
    return np.arange(N) * d * FLOAT_BYTES


def is_vector_aligned(byte_offsets, vector_bytes=VECTOR_BYTES):
    """Which offsets are safe base addresses for a float4 (16-byte) load,
    assuming the buffer itself starts 16-byte aligned (guaranteed by cudaMalloc)."""
    return (byte_offsets % vector_bytes) == 0


if __name__ == "__main__":
    print("== row alignment, d=64 (head dim used in tile_loads.cu) ==")
    offsets = row_start_offsets_bytes(N=6, d=64)
    aligned = is_vector_aligned(offsets)
    for i, (off, ok) in enumerate(zip(offsets, aligned)):
        print(f"row {i}: byte offset {off:>6}  16B-aligned: {bool(ok)}")

    print("\n== row alignment, d=63 (not a multiple of 4 -- Exercise 1/2) ==")
    offsets = row_start_offsets_bytes(N=6, d=63)
    aligned = is_vector_aligned(offsets)
    for i, (off, ok) in enumerate(zip(offsets, aligned)):
        print(f"row {i}: byte offset {off:>6}  16B-aligned: {bool(ok)}")
    print("-> only every 4th row is aligned; a float4 load applied uniformly")
    print("   to every row here reads the wrong bytes on 3 out of 4 rows, or")
    print("   faults outright on the misaligned ones, depending on the platform.")

    print("\n== vectorization changes only the grouping, not the data ==")
    d = 64
    row = np.arange(d, dtype=np.float32)             # a fake Q row: [0, 1, 2, ..., 63]
    scalar_view = row.copy()                          # what the scalar kernel reads, one at a time
    vector_view = row.reshape(-1, 4)                   # what the vectorized kernel reads, 4 at a time
    print("scalar order (first 8):", scalar_view[:8])
    print("vector groups (first 2):", vector_view[:2])
    print("same bytes, same values -- reassembled scalar equals original:",
          np.array_equal(vector_view.reshape(-1), row))
