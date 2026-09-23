"""Chapter 7: numpy trace of the warp-level butterfly (XOR) and down-shift
shuffle reductions, so the *algorithm* -- not the CUDA syntax -- can be
checked by hand or in this sandbox with no GPU. Mirrors warpReduceSumXor /
warpReduceSumDown in warp_reduce.cu, lane for lane, including the real
__shfl_down_sync semantics: a lane with no valid partner reads back its own
value instead of zero.
"""
import numpy as np


def xor_reduce_trace(values, verbose=True):
    """Butterfly (XOR) reduction: mirrors __shfl_xor_sync in a loop of
    halving offsets. Every lane ends up holding the full reduction."""
    lanes = np.array(values, dtype=np.float64)
    width = len(lanes)
    assert width & (width - 1) == 0, "width must be a power of 2"
    offset = width // 2
    step = 0
    while offset > 0:
        partner = np.arange(width) ^ offset          # each lane's XOR partner
        lanes = lanes + lanes[partner]                # every lane adds its partner's value
        step += 1
        if verbose:
            print(f"  step {step} (offset={offset:>2}): {lanes}")
        offset //= 2
    return lanes


def down_reduce_trace(values, verbose=True):
    """Down-shift reduction: mirrors __shfl_down_sync in a loop of halving
    offsets. Only lane 0 ends up holding the full reduction -- every other
    lane holds a partial sum, not the answer. Matches real hardware: a lane
    with no valid partner (i + offset >= width) reads back its OWN value,
    not zero."""
    lanes = np.array(values, dtype=np.float64)
    width = len(lanes)
    assert width & (width - 1) == 0, "width must be a power of 2"
    offset = width // 2
    step = 0
    while offset > 0:
        idx = np.arange(width)
        partner = np.where(idx + offset < width, idx + offset, idx)  # self-fallback
        lanes = lanes + lanes[partner]
        step += 1
        if verbose:
            print(f"  step {step} (offset={offset:>2}): {lanes}")
        offset //= 2
    return lanes


if __name__ == "__main__":
    values = list(range(1, 9))   # Exercise 1: an 8-lane toy warp, values 1..8
    total = sum(values)
    print(f"== toy 8-lane warp, values={values}, true sum={total} ==")

    print("\n-- XOR (butterfly) reduction --")
    xor_result = xor_reduce_trace(values)
    print(f"every lane after reduction: {xor_result}")
    print(f"all lanes correct: {np.allclose(xor_result, total)}")

    print("\n-- DOWN (shift) reduction --")
    down_result = down_reduce_trace(values)
    print(f"every lane after reduction: {down_result}")
    print(f"lane 0 correct: {np.isclose(down_result[0], total)}")
    print(f"lanes 1..7 also correct (they're NOT supposed to be): "
          f"{np.allclose(down_result[1:], total)}")

    print("\n== full 32-lane warp, random values (matches warp_reduce.cu) ==")
    rng = np.random.default_rng(0)
    values32 = rng.standard_normal(32)
    total32 = values32.sum()
    xor32 = xor_reduce_trace(values32, verbose=False)
    down32 = down_reduce_trace(values32, verbose=False)
    print(f"true sum: {total32:.6f}")
    print(f"XOR:  every lane matches sum: {np.allclose(xor32, total32)}")
    print(f"DOWN: lane 0 matches sum: {np.isclose(down32[0], total32)}, "
          f"lanes 1..31 also match (they shouldn't): {np.allclose(down32[1:], total32)}")
