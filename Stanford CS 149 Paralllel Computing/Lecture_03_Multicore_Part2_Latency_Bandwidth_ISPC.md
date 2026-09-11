# CS149 — Lecture 3: Multi-Core Architecture Part II (Latency/Bandwidth) + Parallel Programming Abstractions

**Course:** Stanford CS149, Parallel Computing (Fall 2023)
**Instructors:** Prof. Kayvon Fatahalian & Prof. Kunle Olukotun
**Video:** https://www.youtube.com/watch?v=F4bVSyz_jxo
**Course site:** https://gfxcourses.stanford.edu/cs149/fall23/lecture/multicore2-ispc/

*Notes combine the official slide deck with the lecture transcript (live Q&A, worked examples, analogies).*

---

## 1. Framing for This Lecture

Two halves today:
1. **Finish the hardware picture** — review hardware multi-threading in depth, then combine it with the other three throughput-computing ideas (superscalar, SIMD, multi-core) into one coherent mental model, and introduce **latency vs. bandwidth**, a distinction the course hasn't made explicit yet.
2. **Start the software picture** — introduce **ISPC**, a low-level parallel programming language used in Assignment 1, chosen specifically because its low-level nature makes the abstraction-vs-implementation distinction unusually clear.

---

## 2. Reviewing Hardware Multi-Threading

### Recap
The single idea from the end of Lecture 2: *if you're waiting on something, go do something else instead of idling.* A core that supports multi-threading duplicates only the **execution context** (register state) for multiple threads — not the fetch/decode logic or ALUs, which stay shared. When one thread stalls (most commonly on a long-latency memory request, though it could be any long-latency operation), the hardware simply switches to running an instruction from a different thread's context instead — "switch on stall" — and switches back once the original thread's data is ready.

### A relatable analogy: office hours
Picture holding office hours for 15 students. If you help one student, then sit and wait while they think through a follow-up before helping the next, you're **fully "utilized"** working with whoever's in front of you at every moment, but any individual student might wait a long time to get back to you. If instead you give a student something to think about, move to the next student, and cycle back around, you're using your own time just as fully, but now you're helping *more people* per unit of your fixed time — at the cost of any one student's total time-to-resolution going up. This is exactly the trade-off multi-threading makes inside a core: full utilization of the shared execution hardware, at the cost of increasing any single thread's individual completion latency.

### Worked utilization examples
Consider a thread pattern of **3 arithmetic instructions followed by a 12-cycle memory stall** (repeating):
- 1 thread → busy 3 of every 15 cycles → 20% utilization
- 2 threads → 40% utilization
- **5 threads needed for 100% utilization** — reasoning: one thread alone leaves a 12-cycle gap to fill; each *additional* thread contributes 3 cycles of useful work to fill that gap, so 4 more threads (5 total) exactly cover it.
- More than 5 threads provides no further benefit — the core is already saturated, and would just cost extra chip area for no gain (plus slightly worse latency per thread).

Now change the *program*, not the hardware: **6 arithmetic instructions followed by the same 12-cycle stall** — a program with a higher ratio of computation to memory latency:
- 1 thread → 33% utilization (6 of 18 cycles)
- Only **3 threads total needed for 100% utilization** this time (fewer threads, since each thread now covers more of the stall gap on its own).

**Key relationships this reveals:**
1. Multi-threading adds no new peak execution throughput — it only improves how efficiently *existing* execution resources get used.
2. The ratio of math-to-memory-latency in a program determines how many threads are needed to fully hide stalls — more math per memory access means fewer threads needed. Equivalently, a bigger data cache (which reduces effective memory latency by absorbing misses) also reduces the number of threads needed; a smaller cache would need more.
3. To estimate the *theoretical* speedup multi-threading could offer for a given program, you need two numbers: the achieved rate running only one thread at a time, and the processor's known peak per-clock capability — multi-threading's job is to close the gap between those two numbers, up to (but not beyond) that peak.

(An aside from Q&A: switching between threads on a stall is treated in this class as effectively free — conceptually just swapping which program counter the hardware is currently pointing at.)

---

## 3. Putting All Four Ideas Together: Superscalar + SIMD + Multi-Core + Multi-Threading

### A "fake chip": 16 cores, 4-way multi-threaded, 8-wide SIMD
Each of 16 cores can hold 4 threads' worth of context; each executed instruction is an 8-wide vector operation. A program that spawns 64 threads would have those threads distributed one-to-one across the chip's 64 available hardware thread slots (4 per core × 16 cores), and every core, every clock, picks one of its resident threads' next instruction to run — which happens to operate on 8 pieces of data at once.

Peak throughput for this chip is 16 cores × 8-wide = **128 pieces of data processed in parallel per clock**. But to actually *hide latency* and keep that throughput sustained, the chip needs enough independent work to fill all its thread slots too: 64 threads × 8 data items per thread = **512 independent pieces of data** needed simultaneously in flight to run at full, latency-hidden peak rate.

### A rough model of a real CPU core ("myth machine" style)
Stanford's teaching cluster ("myth") machines use a two-way hyper-threaded, superscalar Intel core. A simplified picture: 2 threads' worth of context, at least three 8-wide vector ALUs plus some scalar ALUs, and enough fetch/decode capacity to look across both threads simultaneously for independent instructions to issue.

- With no vector instructions, this core can issue roughly 3 scalar operations per clock; with vector instructions, roughly 3 *vector* operations per clock — meaning using vector instructions over scalar ones can be worth up to an 8x improvement on top of that same instruction-issue rate.
- A single thread often doesn't contain enough independent instruction-level parallelism (ILP) on its own to keep all 3 vector execution units fed every clock. Adding a *second* thread gives the hardware another, entirely independent source of instructions to draw from — instructions from different threads are automatically guaranteed to be independent of each other, since they're different, unrelated instruction streams. This is exactly why running two threads on a hyper-threaded core often meaningfully outperforms running just one, even with no other code changes.
- Superscalar execution (finding independent instructions to co-issue) and simultaneous multi-threading (SMT — finding independent instructions to co-issue *across different threads*) are, in this sense, the same underlying mechanism: a scheduler filling a fixed set of execution-unit "slots" each clock, just with a wider pool of candidate instructions to draw from once multiple threads are available. Historically, this is presented as basically why hyper-threading exists: once out-of-order logic already exists to find independent instructions within one thread, treating a second thread's instructions as another source of "obviously independent" work is a comparatively small additional step, that pays off specifically when a single thread doesn't have enough exploitable ILP on its own.
- Composing everything: a chip built from two of these cores, each 4-way multi-threaded, would need **8 threads total** (4 per core) to fully occupy its thread slots, and each core could still be picking, say, one scalar and one vector instruction from a mix of its resident threads each clock — the same rules just get applied independently, core by core.

**Caveat raised in Q&A:** it's a reasonable rule of thumb that a core generally needs roughly as many fetch/decode ("orange box") units as it has execution ("yellow box") units to keep those units fed — but real chips can deviate from this in low-level ways (e.g., instructions with long enough latency that a fetch unit doesn't need to refill its unit every single cycle), which gets into microarchitectural detail beyond this course's scope.

### Scaling further: GPU-style cores
The same ideas, just at larger scale: one core ("SM") of a modern NVIDIA GPU might support roughly 32-wide vector instructions, drawing from up to 64 threads' worth of context. Multiplied across ~80–144 such cores on a chip, the amount of independent work needed to reach full, latency-hidden peak throughput easily reaches into the hundreds of thousands — which is exactly why a small neural network (with comparatively little available parallelism) doesn't run efficiently on a large GPU: there simply isn't enough independent work available to fill the machine.

### A detail on how GPUs implement SIMD differently from CPUs
On CPUs (and some Intel GPUs), the compiler explicitly emits vector instructions — if you inspect the compiled binary, the vector operations are visibly there. NVIDIA and AMD GPUs generally work differently: **the compiler only ever emits scalar instructions**, and it's the hardware itself that recognizes when multiple resident hardware threads happen to be at the same program counter, and executes their (identical) instruction together across a shared SIMD ALU — achieving the same effect as an explicit vector instruction, but entirely as a runtime hardware decision rather than a compile-time one. (This distinction — sometimes called explicit vs. implicit SIMD — gets its own treatment when the course covers GPU/CUDA programming directly.)

### Who decides what runs where?
A clarifying Q&A point: when a C++ program spawns threads, it's the **operating system** that decides which hardware thread/execution context each software thread gets mapped to (e.g., you'd generally want two threads placed on two separate cores rather than doubled up on one, unless they specifically benefit from sharing a cache). Once threads are assigned to hardware contexts, however, the moment-to-moment decision of *which* resident thread's instruction to actually execute on a given clock is entirely a chip-level (hardware) decision, made billions of times a second — this is not something the OS is involved in at that granularity. The OS only intervenes rarely, via a full context switch (itself costing on the order of hundreds of thousands of cycles), to change which threads are resident on the chip at all.

---

## 4. Understanding Latency and Bandwidth

### Opening thought experiment
Task: element-wise multiply two very large vectors, `C[i] = A[i] * B[i]`. Is this a good fit for a modern throughput-oriented parallel processor?

First reaction: it seems ideal — there's essentially unlimited parallelism (millions of independent elements), and the operation vectorizes trivially onto SIMD hardware, so there's far more parallel work available than there are cores or ALUs, which should also make it easy to hide any latency. But (as the lecture goes on to show) this initial impression turns out to be wrong, for a reason that has nothing to do with parallelism at all.

### The highway analogy
- Driving from San Francisco to Stanford (~50 km) at 100 km/hr takes 30 minutes — that's the **latency** of one trip.
- With a rule of only one car on the highway at a time, a new car can only enter once the previous one arrives — giving a **throughput** of just 2 cars/hour, even though any single car's trip latency is fixed at 30 minutes.
- **Approach 1 — drive faster:** doubling speed to 200 km/hr cuts latency to 15 minutes *and* doubles throughput to 4 cars/hour. But this has real-world limits (safety, diminishing efficiency, etc.).
- **Approach 2 — add more lanes:** keeping speed at 100 km/hr but doubling the number of lanes doubles throughput to 8 cars/hour, while each individual car's latency is completely unchanged (still 30 minutes).
- **Approach 3 — use the road more efficiently (closer spacing):** keeping one lane but spacing cars 1 km apart (instead of one-at-a-time) raises throughput to 100 cars/hour, again without changing any single car's 30-minute latency.

**The core lesson: latency and throughput (bandwidth) are related but genuinely separate quantities.** You can improve one without improving the other, and a system can be made to sustain a high rate of completions even while any individual item still takes just as long to complete as before.

### Terminology
- **Bandwidth**: the rate at which a system delivers data/completions per unit time (e.g., GB/s, or "8 items/sec").
- **Latency**: how long any *one* item takes to make the full trip.

A minimal illustration: if items are sent one at a time, each taking 2 seconds to arrive, bandwidth might be ~4 items/sec; if instead 2 items are sent together each "trip," bandwidth doubles to ~8 items/sec — while the latency of any individual item is still 2 seconds either way. Bandwidth and latency are decoupled the moment you allow more than one thing to be "in flight" at once.

### Laundry, pipelined
Doing laundry has three sequential stages with fixed durations: wash (45 min) → dry (60 min) → fold (15 min). One load, start to finish, takes **2 hours** — that's the latency of a single load, and it can't be shortened without changing the underlying stages.

- **Duplicating resources**: getting a second washer, dryer, and a friend to help lets you finish *2 loads* in the same 2 hours — 2x the throughput, but it cost 2x the resources (two full sets of washer/dryer).
- **Pipelining instead**: with just *one* washer and *one* dryer, start a new load's wash cycle as soon as the previous load moves on to drying. Any single load still takes 2 hours start-to-finish (latency unchanged), but once the pipeline is full, a new load finishes roughly every hour — **1 load/hour throughput**, achieved using only the original, non-duplicated resources.

This is the general shape of pipelining: overlapping the different stages of multiple, independent units of work so that all stages stay busy simultaneously, improving throughput without needing to duplicate hardware or speed up any individual stage.

### A connected-pipe analogy
If you connect a pipe that can carry 100 liters/sec to a narrower pipe that can only carry 50 liters/sec, the combined system can never move data faster than **50 liters/sec — the throughput of the whole chain is capped by its slowest stage.** This maps directly onto the laundry example (the dryer, at one load/hour, is the bottleneck stage limiting overall throughput even though the washer alone could go faster) and previews the memory-bandwidth discussion below.

### Applying this to a real processor
Consider a simple repeating instruction sequence run across many threads on a multi-threaded core (assume enough threads exist to fully hide latency, so this isn't a latency problem):
```
1. X = load 64 bytes
2. Y = X + X
3. Z = X + Y
```
Machine assumptions: one math operation completed per clock, loads can be issued in parallel with math, and memory can deliver 8 bytes/clock, with a fixed cache-line-sized load taking 8 clocks to fully arrive, and only 3 loads allowed to be outstanding at once.

Tracing this over time: at first, math instructions and load requests can be issued back-to-back, since there's slack in the outstanding-load budget. But once 3 loads are already in flight, no *new* load can be issued until an earlier one completes and frees up a slot — so both the arithmetic and the ability to issue new loads eventually stall, waiting on data that's still arriving. In steady state, memory ends up **continuously transferring data at its maximum rate (100% of the time)** — it genuinely cannot go any faster — while the processor's math units sit idle a large fraction of the time, simply waiting for operands to arrive.

**This is memory-bandwidth-bound execution**: the rate of completing instructions is capped by how fast memory can supply data, not by how many outstanding requests are allowed or by the latency of any individual request. (Worth confirming for yourself: in steady state, the fraction of time the core sits idle here depends only on the *ratio* of instruction throughput to memory bandwidth — not on memory latency itself, and not on how many loads are allowed to be in flight at once. A "magical" zero-latency memory system with the *same* 8 bytes/clock bandwidth limit would show exactly the same steady-state stall behavior — more on why below.)

### Revisiting the thought experiment: why vector multiply is actually a poor fit
Back to `C[i] = A[i] * B[i]`. Each multiply requires reading `A[i]` and `B[i]` and writing `C[i]` — **12 bytes of memory traffic for every single multiply** (three 4-byte floats). An NVIDIA V100 GPU can perform 5,120 fp32 multiplies per clock (80 SMs × 64 ALUs) at ~1.6 GHz — around 8 trillion multiplies/second at peak. Feeding that rate would require roughly **98 TB/sec of memory bandwidth**; the V100's actual HBM2 memory delivers about 900 GB/sec — roughly 100x too little.

The result: this computation runs at **under 1% of the GPU's peak efficiency** — not because of insufficient parallelism (there's effectively infinite parallelism here) and not because of memory *latency* (a hypothetical zero-latency memory system with the same bandwidth ceiling would show the identical bottleneck) — purely because the ratio of memory traffic to actual computation is far too high for the available bandwidth to sustain. (For context: even an eight-core CPU with ~76 GB/sec of memory bandwidth only reaches around 3% efficiency on this same computation — still bandwidth-bound, just against a smaller peak-compute number, so the percentage looks a little less dismal.) No amount of prefetching, more outstanding requests, or additional parallel work changes this outcome — the fundamental constraint is the ratio of bytes moved to useful math performed, and a faster/smarter memory system is the only thing that would help, short of restructuring the computation itself.

### Bandwidth: the critical resource in modern computing
Because this problem generalizes far beyond one toy example, the lecture draws out a broader principle: on modern throughput-oriented hardware, **bandwidth — not raw compute, and not latency — is usually the binding constraint.** Performant parallel programs therefore need to:
- Fetch data from memory as infrequently as possible: reuse data already loaded by the same thread (temporal locality), and share/reuse data across cooperating threads where possible.
- Favor doing *more arithmetic* on data already in registers over re-loading or re-storing values — extra math is comparatively "free" next to the cost of moving data.
- Recognize that essentially every program on modern hardware will be bandwidth-bound to some degree unless it deliberately manages this ratio — parallelism alone does not fix a bandwidth problem.

(A caveat implicit in this framing: not every workload has a punishing 1-math-op-per-12-bytes ratio like the vector-multiply example. Workloads like computer graphics and machine learning typically involve far more arithmetic per byte moved — on the order of 10–20 operations per byte in many cases — which is a large part of why those workloads are able to approach a much higher fraction of peak compute efficiency on the same hardware. A modern chip architect's design process, in fact, often starts from "how much memory bandwidth can I realistically afford," and only then decides how many ALUs are worth building to match that bandwidth, given the target workloads.)

### Aside: instruction pipelining (throughput vs. latency, one more time)
The same latency/throughput distinction that applies to memory access also applies to instruction execution itself. A 4-stage instruction pipeline (fetch → decode → execute → write-back) might give any single instruction a **latency of 4 cycles** to fully complete, while still achieving a **throughput of 1 instruction per cycle** in steady state, by having multiple instructions in different pipeline stages simultaneously. This resolves a common point of confusion: when this course says a core "does one operation per clock," that's a statement about **throughput**, not about how many cycles any individual instruction actually takes internally to complete (real instruction pipelines can run considerably deeper than 4 stages — up to roughly 20 stages on some modern CPUs). Correctness with back-to-back dependent instructions in a pipelined design requires care (e.g., forwarding results between stages), but the mechanism is a natural extension of the same pipelining idea used throughout this lecture.

---

## 5. Part 2: Abstraction vs. Implementation

A theme the instructors flag as a common, recurring source of confusion in this course: conflating the **semantics** (meaning) of a parallel programming abstraction with the **details of how it happens to be implemented**.

- **Semantics**: given a program and the meaning of the operations it uses, what answer will it produce?
- **Implementation / scheduling**: given a parallel machine, in what actual (possibly parallel) order do those operations get carried out — which operations run on which thread, which execution unit, which lane of a vector instruction?

The skill this course is trying to build: given a parallel program and an understanding of how its programming model is implemented, be able to mentally "trace" through what every part of the machine is doing at each step.

---

## 6. Programming with ISPC

**ISPC** = Intel SPMD Program Compiler (SPMD = "single program, multiple data"). It's introduced not primarily to teach the language itself, but because its unusually low-level nature makes the abstraction-vs-implementation distinction concrete and hard to gloss over — a large fraction of office-hours confusion in past years has come from students conflating what an ISPC program *means* with how it happens to get executed underneath.

### Calling into ISPC from C++
Ordinary C++ code has one thread of control: calling a function transfers control to it, runs it sequentially, and returns. Rewriting the familiar `sinx` Taylor-series function (from Lecture 2) as an ISPC function changes this: a call from C++ into an ISPC function spawns a **"gang" of program instances**, which all run the function's logic concurrently; execution only returns to the calling C++ code once every instance in the gang has finished.

```c
export void ispc_sinx(
    uniform int N, uniform int terms,
    uniform float* x, uniform float* result)
{
    // assume N % programCount == 0
    for (uniform int i = 0; i < N; i += programCount)
    {
        int idx = i + programIndex;
        float value = x[idx];
        float numer = x[idx] * x[idx] * x[idx];
        uniform int denom = 6;
        uniform int sign = -1;
        for (uniform int j = 1; j <= terms; j++)
        {
            value += sign * numer / denom;
            numer *= x[idx] * x[idx];
            denom *= (2*j+2) * (2*j+3);
            sign *= -1;
        }
        result[idx] = value;
    }
}
```

Two new built-in variables carry the SPMD abstraction:
- **`programCount`** — the total number of simultaneously executing instances in the gang (a `uniform` value: every instance sees the same number).
- **`programIndex`** — this particular instance's ID within the gang (a *varying* value: every instance sees a different number, 0 through `programCount - 1`).

The `uniform` keyword is a type modifier meaning "every instance has the same value for this variable" — it's purely an optimization hint for the compiler, not required for program correctness.

### The key exercise: what does each program instance actually compute?
Given the code above, with (say) 8 program instances: instance 0 handles array indices 0, 8, 16, 24, ...; instance 1 handles 1, 9, 17, 25, ...; and so on — an **interleaved** assignment of array elements to instances, purely because of how the loop was written (`i += programCount`, `idx = i + programIndex`). This is entirely a property of the *program's logic*, not of the underlying hardware.

**Important abstraction-level point:** the *meaning* of this program (every element gets the correct sine value) is completely satisfied whether the 8 instances are implemented as 8 real operating-system threads spawned on 8 different cores, or as one thread running all 8 instances one after another in a plain sequential loop, or (what real ISPC actually does) as a single thread issuing 8-wide SIMD vector instructions. All of these are valid implementations of the same abstraction.

### Interleaved vs. blocked assignment
A second, rewritten version of `sinx` assigns each instance a **contiguous block** of the array instead (instance 0 gets indices 0..N/8-1, instance 1 gets N/8..2N/8-1, etc.) — same final answer, different assignment of work to instances.

This distinction turns out to matter a lot for the *real* SIMD implementation ISPC generates:
- **Interleaved** assignment means the 8 values `x[idx]` needed by the 8 instances on a given loop iteration are contiguous in memory — loadable with a single, efficient packed vector load instruction.
- **Blocked** assignment means those same 8 values are scattered far apart in memory on any given iteration — requiring a much more expensive "gather" instruction instead of a simple vector load.

This is a very concrete illustration of why the same *abstract* parallel program can have meaningfully different real performance depending on implementation-level choices the programmer made — even though both versions are equally "correct."

### `foreach`: raising the abstraction level
Rather than manually computing interleaved or blocked index assignments, ISPC provides `foreach`, which just declares "here are the loop iterations the whole gang needs to perform collectively" and leaves the actual assignment of iterations to instances up to the ISPC implementation:

```c
foreach (i = 0 ... N)
{
    float value = x[i];
    // ...
    result[i] = value;
}
```

Several different underlying implementations would all be valid ways to realize this same `foreach` construct: running everything on a single instance sequentially, interleaving iterations across instances, blocking iterations across instances, or even dynamically assigning the next available iteration to whichever instance becomes free next (a form of work-stealing/dynamic scheduling). The `foreach` abstraction is compatible with any of them — the programmer's job is only to correctly declare that the iterations are independent; deciding *how* to schedule them is left entirely to the implementation.

### Gotchas: undefined and incorrect programs
Because ISPC is low-level enough to expose `programIndex` and `programCount` directly, it's entirely possible to write `foreach` code whose result depends on implementation details, or that's simply broken:
- A program where two different loop iterations can end up writing to the *same* output memory location (e.g., writing to `y[i-1]` under some condition) has **undefined behavior** — different valid implementations (interleaved vs. blocked vs. sequential) could genuinely produce different final results, because the ordering/interleaving of writes to that shared location isn't specified by the abstraction.
- Attempting `sum += x[i]` inside a `foreach` when `sum` is declared `uniform` is a compile-time type error: `x[i]` is a different value per instance, so it's not obviously clear what a single shared `uniform` variable should even accumulate. Declaring `sum` as an ordinary (non-uniform, "varying") `float` instead just creates a *separate* private copy of `sum` per instance — which is also wrong, since the calling C++ code expects a single scalar float to be returned, not one copy per instance.

### Doing this correctly: private accumulation + explicit cross-instance reduction
```c
export uniform float sum_array(uniform int N, uniform float* x)
{
    uniform float sum;
    float partial = 0.0f;
    foreach (i = 0 ... N)
    {
        partial += x[i];
    }
    sum = reduce_add(partial);   // combines all instances' partial sums
    return sum;
}
```
Each instance privately accumulates its own `partial` sum with zero communication between instances during the loop; only at the end does `reduce_add` — one of several built-in **cross-program-instance operations** — explicitly combine every instance's private value into a single shared result. Other similar built-ins mentioned: `reduce_min` (combine via minimum), `broadcast` (send one instance's value to all instances), and `rotate` (pass each instance's value to another instance some fixed offset away). These map directly onto real SIMD hardware instructions — the equivalent hand-written AVX intrinsics version accumulates a vector register across the loop and then manually sums its 8 lanes together at the end, which is essentially what `reduce_add` compiles down to.

A more advanced example shown briefly: computing the product of all 8 elements of a gang-sized array in just **log₂(8) = 3 steps**, using `shift` (a variant of cross-instance data movement) combined with conditionals based on `programIndex` — demonstrating that ISPC's low-level access to `programIndex`/`programCount` allows expressing genuinely sophisticated inter-instance cooperation patterns well beyond simple independent, embarrassingly-parallel loops.

### ISPC's `gang` vs. `task` abstractions
Everything covered so far — the "gang" of program instances — is implemented purely with SIMD instructions running on a *single* CPU core/thread. On its own, this means ISPC code as shown would only ever use one of, say, four cores on a machine. ISPC provides a separate abstraction, **`task`**, specifically to spread work across multiple *cores* (left as further reading for the assignment) — `gang` gives you SIMD width; `task` gives you multi-core.

### Where a language could go from here
The lecture closes by sketching design alternatives further up the abstraction ladder than ISPC's deliberately low-level, `programIndex`-exposing design:
- A version of the language that hides `programIndex`/`programCount` entirely and only allows `foreach`, so the programmer essentially never has to think in terms of individual program instances at all.
- Going further still: no explicit indexing at all — just applying a function to every element of a collection (`map(doWork, x)`), a model that should feel very familiar to anyone who's used NumPy's vectorized/broadcasted operations or PyTorch tensor ops, which the course revisits in more depth later.

---

## 7. Summary

- Hardware multi-threading adds no new peak throughput — it only improves *utilization* of existing execution resources by hiding stalls, at the cost of increasing any individual thread's own completion latency.
- Superscalar, SIMD, multi-core, and multi-threading all compose together on real chips, and the amount of independent work needed to reach full, latency-hidden peak throughput multiplies across all four dimensions.
- **Latency and bandwidth (throughput) are distinct, decoupled quantities** — a system can be pipelined or parallelized to sustain high throughput without changing the latency of any individual operation, and vice versa.
- On modern hardware, **bandwidth is very often the real bottleneck**, independent of how much raw compute or parallelism is theoretically available — programs must be structured to minimize memory traffic relative to useful computation.
- Programming models (like ISPC's SPMD abstraction) define *meaning*, and separately admit multiple valid *implementations* — being able to reason clearly about that distinction, and to mentally trace what each part of a parallel machine is doing at each step, is a core skill this course is building toward.

---

*Notes synthesized and paraphrased from the CS149 Fall 2023 Lecture 3 slide deck and lecture transcript, for study purposes — not a verbatim transcript.*
