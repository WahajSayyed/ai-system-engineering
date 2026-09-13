# Chapter 10 — Profiling, Debugging & Performance Engineering

*Part 9: Profiling, Debugging & Performance Engineering.*

**A transparency note, upfront, because it matters for how this chapter is grounded:** unlike every other chapter so far, there is **no dedicated folder** for this material in `book.cu`. The repo runs `0_vecadd` → `1_naive` → `2_mnist` → `3_transformer` → `4_optim` → `5_tensor_cores` → `6_flash` → `7_quant` → `8_distributed` → `9_cutlass` — and `8_distributed`'s own files explicitly label themselves "Chapter 10," which places this course's Chapter 10 (the book's true Chapter 9) in a gap the companion repo simply doesn't fill with code. The book's back-cover blurb explicitly promises Nsight Compute coverage, so the content clearly exists in the book itself — it just isn't backed by a runnable example folder in this snapshot the way every other chapter has been.

Given that, this chapter is grounded two other ways instead: **current, verified NVIDIA documentation** for Nsight Systems and Nsight Compute (checked directly against the docs rather than recalled from memory, since tool CLIs and metric names do shift version to version), and — more usefully — **retroactive application of these tools to real code you've already read** in Chapters 3 through 9 of this exact course. You already know, by reasoning about source code, that Chapter 3's naive GEMM is memory-bound, that Chapter 4's `softmax_kernel<<<8,1>>>` wastes almost the entire GPU, and that Chapter 7's benchmark once mismeasured a kernel by 5×. This chapter's job is to show you the *tool output* that would have told you the same things empirically, without reading a single line of source.

---

## 10.1 Two Tools, Two Questions

**Nsight Systems (`nsys`)** answers *"where does my program's time actually go?"* — a timeline across your whole application: every kernel launch, every memory copy, CPU activity, and how they overlap (or don't) across streams. It does **not** lock GPU clocks while profiling, so its timing reflects normal running conditions.

**Nsight Compute (`ncu`)** answers *"why is this one specific kernel slow?"* — deep, hardware-counter-level analysis of a single kernel's execution: occupancy, memory throughput, compute throughput, warp stall reasons. Collecting the full set of counters often requires **replaying the kernel multiple times** (not every counter can be read from hardware in a single pass), and — confirmed directly from NVIDIA's own Profiling Guide — **Nsight Compute locks SM clocks by default specifically so repeated runs are comparable**, which Nsight Systems deliberately does not do.

Basic invocations, confirmed against current Nsight documentation:

```bash
# Nsight Systems: whole-program timeline, with a text summary table
nsys profile --stats=true -t nvtx,cuda ./your_binary

# Nsight Compute: full metric set for every kernel, saved to a report file
ncu --set full -o report ./your_binary

# Nsight Compute: just the metrics needed for a roofline (compute-vs-memory) analysis
ncu --set roofline ./your_binary

# Re-open a saved report without re-running the program
ncu --import report.ncu-rep --page details

# Profile only kernels matching a name pattern — essential once a program launches many kernels
ncu --kernel-name regex:my_kernel_.* --set full -o targeted ./your_binary
```

## 10.2 Nsight Systems: Finding Chapter 8's 384 Launches

Chapter 8's `naive.cu` launches three kernels per (batch, head) pair, for 128 pairs — 384 total kernel launches, reasoned out by literally counting loop iterations in the source. `nsys --stats=true` gives you that same number **without reading any code**: its post-run summary table groups every launch by kernel name and reports call count, total time, and average time per call directly. Run it against `naive.cu` and you'd see three kernel names, each with a count of 128; run it against `fa.cu` and you'd see one kernel name with a count of 128 — an immediate, visual confirmation of exactly the launch-count argument Chapter 8 §8.1 made by counting loops.

The timeline view adds something counting loops can't show you at all: **the actual gaps between launches.** Chapter 1 §1.3.2 told you kernel launches cost a few microseconds of fixed overhead; `nsys`'s timeline is where you'd actually *see* those gaps as visible dead space between one kernel's end and the next one's start — and see immediately whether your program is launching kernels back-to-back efficiently or leaving the GPU idle between them waiting on host-side work.

## 10.3 Nsight Compute: The Speed-of-Light Page, Confirming Chapter 3 Empirically

Nsight Compute's **Speed-of-Light (SOL)** section reports exactly the two numbers Chapter 1 §1.4.1's roofline model needs, computed for you: **Compute (SM) Throughput** and **Memory Throughput**, each as a percentage of the GPU's peak — confirmed metric names `sm__throughput.avg.pct_of_peak_sustained_elapsed` and `dram__throughput.avg.pct_of_peak_sustained_elapsed`. Point this at Chapter 3's naive GEMM kernel and you'd expect to see **memory throughput sitting close to its ceiling while compute throughput sits far below its own** — the tool's own empirical confirmation of the "compute-bound in theory, memory-bound in practice" diagnosis Chapter 3 §3.3.2 reached by reasoning about data reuse, now backed by actual hardware counters instead of an argument.

The **Occupancy** section reports both a *theoretical* occupancy (the maximum possible, given your kernel's register and shared-memory usage) and an **achieved** occupancy (`sm__warps_active.avg.pct_of_peak_sustained_active`) — what actually happened at runtime. This is where Chapter 4's `softmax_kernel<<<batch_size, 1>>>` would show its damage most starkly: a grid of only 8 blocks, each with a single thread, on a GPU with dozens of SMs — most SMs receive zero work at all, and achieved occupancy would read as a tiny fraction of a percent. You already knew this kernel was wasteful from reading its launch configuration; `ncu`'s occupancy report is what makes that waste an unambiguous, quantified number instead of an inference.

**Warp stall reasons** (the `smsp__warp_issue_stalled_*` family of metrics) categorize, cycle by cycle, *why* warps aren't issuing an instruction — waiting on a global memory load (`long_scoreboard`), waiting on shared memory (`short_scoreboard`), waiting at a `__syncthreads()` barrier, and several other categories. Chapter 3's naive GEMM would show heavy `long_scoreboard` stalls (exactly what "no shared-memory reuse" predicts); Chapter 6's tiled GEMM kernel, with its two `__syncthreads()` calls per K-tile, would show a real, expected amount of `barrier` stalls — the cost of correctness, visible directly as a stall category instead of an invisible tradeoff.

## 10.4 Applying `ncu` Retroactively to Chapter 7's Benchmarking Bug

This is the clearest possible illustration of why these tools exist. Recall Chapter 7 §7.4: an early measurement folded a ~30ms row-major-to-column-major layout conversion into what was reported as a WGMMA kernel's runtime, making a genuinely fast kernel (~550 TFLOPS) look mediocre (~106 TFLOPS) — a real, documented mistake the book's own authors made and caught.

Here's precisely how `nsys` (or `ncu` with `--kernel-name regex:` targeting just the WGMMA kernel) would have prevented that mistake from ever shipping: the layout-conversion routine and the actual `wgmma_max_tiles` kernel are **two separate entries in the timeline**, with two separate durations. A `--stats=true` summary table lists them as two distinct rows — one bucketed at roughly 30ms, one at a small fraction of a millisecond. The mistake was possible specifically because the original benchmark used a **hand-rolled wall-clock timer wrapped around a block of code wider than the kernel itself** — exactly the failure mode Chapter 6 §6.1's benchmark harness (and its careful `Timer` class scoping) was designed to avoid, and exactly why "what's inside your timer" (Chapter 8 §8.4's framing, applied one chapter early) matters as much as the kernel's own code.

The general, durable habit this earns you: **profile the specific kernel by name**, never "the whole program's wall clock," whenever you want a number you'd be willing to publish.

## 10.5 `cuda-gdb` and `compute-sanitizer`: Correctness Tools, Not Speed Tools

You've used `compute-sanitizer` in nearly every chapter since Chapter 2, on the assumption it was a good habit; here's the fuller picture of what it actually checks. Its default tool, **`memcheck`**, catches out-of-bounds and misaligned memory accesses — exactly what Chapter 2's deliberately-broken bounds-check exercise was designed to trigger. A second tool, **`racecheck`** (`compute-sanitizer --tool racecheck`), specifically detects shared-memory race conditions — precisely the bug class Chapter 6 Exercise 4 asked you to reproduce, by deliberately removing one of GEMM kernel 3's two `__syncthreads()` calls.

**Here's the honest limitation worth naming directly, using a bug you've already met:** Chapter 9's `calibrate_min_max` kernel has a real, self-acknowledged bug — every block after block 0 computes a correct local min/max, but that result is silently discarded because only `blockIdx.x == 0` ever writes the final answer. **Neither `memcheck` nor `racecheck` would catch this.** There's no out-of-bounds access, and there's no data race — every thread's atomic operations on its own block's shared memory are perfectly well-defined. It's a pure **logic bug**: the algorithm itself is wrong, not the memory safety of its implementation. Sanitizers check *how* memory is accessed, not *whether the resulting answer is the one you intended*. Catching this class of bug needs either a targeted correctness test (exactly what Chapter 9 Exercise 1 asked you to build), a debugger, or — as it turned out here — just reading the code's own comments critically.

**`cuda-gdb`** is where you go once you need to inspect kernel state directly rather than just detect that something's wrong:

```bash
cuda-gdb ./your_binary
(cuda-gdb) break my_kernel               # breakpoint inside a __global__ function
(cuda-gdb) run
(cuda-gdb) info cuda kernels              # list all kernels currently running on the device
(cuda-gdb) info cuda blocks               # list active thread blocks
(cuda-gdb) info cuda threads              # list active threads, grouped by state
(cuda-gdb) cuda block 3 thread 12         # switch debugger focus to a specific thread
(cuda-gdb) print As[threadRow * BLOCKSIZE + threadCol]   # inspect a variable for the focused thread
```

And the two environment variables introduced back in Chapter 5 §5.6 now have their full context: `CUDA_LAUNCH_BLOCKING=1` forces every kernel launch to run synchronously, which matters because CUDA errors are often reported **several lines after** the launch that actually caused them (async execution means the host has moved on); forcing synchronous launches pins an error report to the correct line. `TORCH_USE_CUDA_DSA=1` enables device-side assertions with human-readable messages instead of a bare error code, when you're inside a PyTorch extension.

## 10.6 A Practical Profiling Checklist

Putting this chapter's tools in the order you'd actually reach for them, synthesizing the whole course:

1. **`nsys` for the big picture.** Which kernels dominate total wall-clock time? Are there suspiciously high launch counts (Chapter 8's 384) or visible gaps between launches?
2. **`ncu --set roofline` on the top 1–2 kernels by total time.** Memory-bound or compute-bound? (Chapter 1 §1.4.1's question, now answered by hardware counters instead of arithmetic.)
3. **`ncu`'s Occupancy section.** Is the grid even large enough to use the GPU at all? (Chapter 4's softmax kernel fails this check catastrophically.)
4. **`ncu`'s warp stall reasons.** What's actually blocking instruction issue — memory latency, a barrier, something else?
5. **`compute-sanitizer` (`memcheck` and `racecheck`) as a correctness gate *before* trusting any performance number at all.** A wrong-but-fast kernel is worthless, and neither speed metric above tells you whether the answer is correct.

---

## Hands-On Lab

1. **Confirm Chapter 6's ladder empirically.** Run `ncu --set roofline` against Chapter 6's GEMM kernel 1 (naive) and kernel 6 (vectorized). Compare the two Memory/Compute SOL percentages directly — confirm the optimization ladder really did move the kernel measurably along the roofline, not just "get faster" for reasons you're taking on faith.
2. **Reproduce the launch-count comparison.** Run `nsys profile --stats=true` against Chapter 8's `naive.cu` and `fa.cu`. Confirm the per-kernel-name call counts match the 384-vs-128 figures reasoned out in Chapter 8 §8.1.
3. **Catch Chapter 6's sync-removal bug with a real tool.** Reintroduce the missing `__syncthreads()` from Chapter 6 Exercise 4 (GEMM kernel 3) and run `compute-sanitizer --tool racecheck` against it. Confirm it's flagged, and read what the tool reports about *which* memory location the race involves.
4. **Measure Chapter 4's softmax anti-pattern directly.** Run `ncu`'s Occupancy section against `v4.cu`'s `softmax_kernel<<<batch_size, 1>>>` and record the achieved-occupancy percentage. If you completed Chapter 6 Exercise 4 (the warp-shuffle rewrite), compare its occupancy at the same problem size.
5. **Step through a kernel with `cuda-gdb`.** Break inside Chapter 6's shared-memory-tiled GEMM kernel (kernel 3), and inspect the values in `As`/`Bs` for one specific thread, both before and after a `__syncthreads()` call, to confirm you understand exactly what that synchronization point is protecting.

## Exercises

1. **Blind diagnosis.** Pick any kernel you've already built from Chapters 3, 6, 7, or 9. Run the full `ncu --set full` report and, using *only* the Speed-of-Light, Occupancy, and Warp-Stall sections — not the source code — write down your best guess at whether it's memory-bound or compute-bound, and why. Then check your guess against what you already know from having read the source.
2. **Reproduce Chapter 7's benchmarking lesson properly.** Wrap both a deliberate layout-conversion step and a real kernel launch in the same `nsys` profiling run, and confirm the timeline attributes time to each *separately* — the exact visibility a hand-rolled wall-clock wrapper would have hidden.
3. **Explain why sanitizers miss logic bugs.** Using §10.5's distinction between memory/synchronization errors and pure logic errors, explain in your own words why `compute-sanitizer` — with any of its available tools — would not have caught Chapter 9's multi-block `calibrate_min_max` bug.
4. **Occupancy math, verified.** For any kernel you profile, compare `ncu`'s reported *theoretical* occupancy against its *achieved* occupancy. If there's a large gap between the two, use the kernel's register and shared-memory usage (also reported by `ncu`) to reason about what's capping the theoretical number below 100%.
5. **Read one stall-reason breakdown end to end.** For a kernel of your choice, list its top three warp stall reasons by percentage, and explain each one in plain language — what is a warp in that state actually waiting for?

---

**Next:** Chapter 11 — Multi-GPU & Distributed Training (Part 10). Every chapter so far has optimized work on *one* GPU. This chapter is where you finally have the tools (this chapter) and the single-GPU efficiency (Chapters 6–9) that make scaling to many GPUs worth doing at all — recall Chapter 1 §1.6.4's warning that multiplying a slow kernel across 8 GPUs just produces a slow result 8× more expensively.
