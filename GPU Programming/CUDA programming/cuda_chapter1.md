Let's build a thorough intuition for GPU architecture before we touch a single line of code. This chapter has three parts: the mental model, a visual of the hardware, then an interactive simulator to make warp execution tangible.

## Part 1 — Why GPUs exist: the fundamental difference from CPUs

A CPU is built for *latency* — it wants to finish a single task as fast as possible. It has a few powerful cores (4–32), large caches, branch prediction, out-of-order execution — all to minimize the time any one instruction takes.

A GPU is built for *throughput* — it wants to process as many tasks simultaneously as possible, even if each one takes longer. A modern GPU has thousands of smaller, simpler cores. Where a CPU hides latency with clever hardware tricks, a GPU hides latency by *switching to another task* — if one group of threads is waiting on memory, it instantly runs another group that's ready.

This makes GPUs transformative for problems that are naturally parallel: matrix multiplication, image processing, neural networks, physics simulation — anywhere you're doing the same operation on millions of data points.

## Part 2 — The hardware hierarchy

The GPU is organized in three nested levels. Understanding this hierarchy is the foundation of everything else in CUDA.The three levels to remember:

- A **GPU** contains many **Streaming Multiprocessors (SMs)**. A modern A100 has 108 SMs; an H100 has 132. SMs are independent execution engines — they each have their own registers, shared memory, and L1 cache.
- Each **SM** runs many **warps** concurrently. A warp is a group of exactly 32 threads. This number — 32 — is fundamental and won't change. It's baked into the hardware.
- Each **warp** contains 32 **CUDA cores** executing in lockstep. All 32 threads in a warp execute the same instruction at the same cycle. This is called **SIMT** (Single Instruction, Multiple Threads).

## Part 3 — The CUDA programming model

The hardware hierarchy maps directly to software concepts you control in your code.

| Software (what you write) | Hardware (what executes it) |
|---|---|
| Grid | Entire GPU |
| Thread Block | One SM |
| Warp (auto, 32 threads) | One set of 32 cores |
| Thread | One CUDA core |

When you launch a kernel, you specify a **grid of blocks**. CUDA distributes those blocks across SMs. Each SM takes some blocks, breaks them into warps of 32, and schedules them onto its cores.

## Part 4 — The SIMT execution model & warp divergence

The most important behavior to understand before writing any kernel is how warps actually execute. The interactive below lets you feel it directly.Try all four scenarios. The key insight: **a warp can only execute one path at a time**. When threads disagree on which branch to take, the hardware serializes them — running one group while masking the others off. This is called **warp divergence**, and it's the single biggest performance pitfall for new CUDA programmers.

## Part 5 — Memory hierarchy (quick mental map)

Before the exercises, one more concept you'll need: CUDA has several memory spaces with very different speeds.

| Memory type | Where | Latency | Size | Who can access |
|---|---|---|---|---|
| Registers | On-chip (per thread) | ~1 cycle | ~255 per thread | That thread only |
| Shared memory | On-chip (per SM) | ~5 cycles | 48–228 KB per SM | All threads in a block |
| L1/L2 cache | On-chip | ~20–40 cycles | automatic | Automatic |
| Global memory (HBM) | Off-chip (VRAM) | ~400–800 cycles | GBs | All threads |
| Host (CPU RAM) | CPU side | ~ms | GBs | CPU only (use cudaMemcpy) |

The practical rule: **global memory is 100–200× slower than registers**. Most CUDA optimization is about moving data from global memory into shared memory or registers before doing compute on it.

---

## Hands-on exercises

These don't require any code yet. They build the mental model that makes the code make sense.

**Exercise 1 — Thread counting**

A kernel is launched with `gridDim = (4, 2)` blocks and `blockDim = (8, 4)` threads per block.

- How many total threads are launched?
- How many warps per block?
- How many total warps across the entire grid?

Work it out, then reply with your answers and I'll verify.

**Exercise 2 — Warp divergence prediction**

Look at this pseudocode running inside a CUDA kernel:

```c
int tid = threadIdx.x;  // 0 to 31 (one warp)

if (tid % 2 == 0) {
    do_work_A();
} else {
    do_work_B();
}
```

- Does this cause warp divergence? Why or why not?
- What about this version instead: `if (tid < 16)`?
- Which version is faster, and why?

**Exercise 3 — Memory intuition**

You're adding two vectors of 1 million floats each. Each element is independent. Describe in plain English how you'd use the GPU memory hierarchy to do this efficiently — which memory would the input live in, where would you do the computation, where would the output go?

---

When you've thought through these (or want to discuss any of them), say "check my answers" or "explain exercise N" and we'll go through them together. Once the mental model feels solid, we move to Chapter 2 where we install the toolkit and write the actual kernel.