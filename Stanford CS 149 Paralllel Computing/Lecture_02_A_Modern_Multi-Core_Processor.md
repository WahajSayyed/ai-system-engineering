# CS149 — Lecture 2: A Modern Multi-Core Processor

**Course:** Stanford CS149, Parallel Computing (Fall 2023)
**Instructors:** Prof. Kayvon Fatahalian & Prof. Kunle Olukotun
**Video:** https://www.youtube.com/watch?v=CKmNpAO5rS4
**Course site:** https://gfxcourses.stanford.edu/cs149/fall23/lecture/multicore/

*Revised from the lecture transcript/audio — incorporates the live classroom discussion (Q&A, worked examples, analogies) in addition to the official slide deck.*

---

## 1. Framing for This Lecture

This lecture looks at computer architecture from a software engineer's point of view — not to design chips, but to understand *why* modern parallel processors are built the way they are, so you can:
- Reason about and optimize the performance of your own parallel programs
- Build intuition for which workloads actually benefit from parallel hardware

The lecture builds toward three distinct mechanisms modern processors use to go fast — **multi-core**, **SIMD**, and **multi-threading** — and frames the last of these explicitly as a way of fighting the same memory-latency problem that motivated caches in the first place.

---

## 2. Review & Deep Dive: Programs, Processors, and Memory

### The abstraction/implementation split (a recurring theme)
A useful habit reinforced throughout this course: separate what something *means* (its semantics/abstraction) from *how it's actually built* (its implementation).

- **A program** = a list of instructions (commands telling the machine what to do). Abstractly, that's all it is.
- **A processor** executes those instructions, and the only observable effect of execution is a change in **state** — held either in **registers** or in **memory**.
- **Memory**, abstractly, is just an array of bytes: every address has a value; asking for an address gets you a value back, writing to an address stores a value there. That's the entire abstraction — it says nothing about *how* that storage is actually built.
- **DRAM** is one concrete implementation of the memory abstraction — physical, off-chip storage. It's "dynamic" in that it needs power to hold its values (and, in fact, reading a DRAM cell is itself destructive — the value has to be written back after being read). DRAM internals get a dedicated deep dive later in the course.
- **Caches** are another implementation detail layered in front of DRAM — on-chip storage holding a copy of a subset of memory's values, purely to make common accesses faster. Critically: **a cache changes nothing about program correctness.** If you removed every cache from a computer, programs would produce identical results — they'd just run slower. Caches exist solely to reduce the latency of memory access.

### Why caches operate on lines, not single bytes
Even though the memory abstraction is "one address in, one value out," real cache hardware always moves data between memory and cache in **fixed-size chunks called cache lines** (e.g., 64 bytes on real Intel chips; 4 bytes in the simplified in-class examples). A cache doesn't really "think" in terms of individual addresses — it thinks in terms of which line an address falls into (address mod line-size). So asking for address 5 (in a 4-byte-line cache) actually triggers a fetch of the whole line covering addresses 4–7, not just byte 5.

This has a direct consequence: if your access pattern skips around within a line (e.g., touching 0, 1, 2, 3, then jumping to 5, 7, 8, 10, skipping 4, 6, 9), you still pay for moving the full lines those addresses belong to, including bytes you never actually use — i.e., you can waste memory traffic by *reducing* the spatial locality your access pattern could otherwise exploit.

### Cache hit/miss mechanics, walked through step by step
Using a toy setup (8-byte total cache capacity, 4-byte lines → 2 lines fit, LRU eviction policy):

- The **first** time any address is touched, the cache has nothing for it → **miss**. This particular kind of miss — the data was simply never in the cache before — is called a **cold miss** (a.k.a. compulsory miss). It would occur even with an infinitely large cache, because it's about *recency*, not capacity.
- Once a line has been loaded, further accesses to *any* address within that line are **hits** — fast, no trip to memory required. This is **temporal locality** at work when you re-touch the same address, and **spatial locality** at work when you touch a different address that happens to share a line with something recently loaded (the line load effectively "prefetches" its neighboring bytes).
- If you keep accessing new lines beyond what the cache can hold, the cache must **evict** something to make room. Under a least-recently-used (LRU) policy, whichever line hasn't been touched in the longest time gets kicked out. If that evicted line had only been read (never written), it can simply be discarded; if it had been written to, its updated contents must first be written back out to memory so the change isn't lost.
- A miss caused specifically because the cache's *capacity* is too small to hold everything you've touched (even though it wasn't literally your first time touching that particular address) is called a **capacity miss**. A third category, the **conflict miss** — a miss that happens because of *how* addresses are mapped to specific slots in the cache, even when there'd technically be room elsewhere — was mentioned but deliberately not covered in depth, since it depends on cache-implementation details (set-associativity) outside this course's scope. The instructor's practical takeaway: for this class, the *cause* of a miss matters less than the fact that a miss occurred and the eviction policy had to kick in — the underlying processor doesn't know or care *why* it missed, only that it has to go fetch data and evict something.

### Why a hierarchy exists
The instructor's analogy: think of storage tiers the way you'd think about where you keep physical things in your life — some papers stay right on your desk (fastest access, smallest capacity), some go in a nearby filing cabinet, some in a hallway closet, and some out in the garage (slowest to retrieve, but much larger capacity). Modern chips mirror this: **L1** (smallest, closest, fastest) → **L2** → **L3** (larger, farther, slower) → **DRAM** (largest, farthest, slowest). The general rule of thumb: bigger storage tends to mean farther away, which means higher latency to access (and, as later lectures cover, typically higher energy cost per access too).

Representative latency figures (in cycles, on a reference ~4 GHz Kaby Lake-era CPU):

| Location | Latency (cycles) |
|---|---|
| L1 cache | ~4 |
| L2 cache | ~12 |
| L3 cache | ~38 |
| DRAM (best case) | ~248 |

The scale of this gap is the whole motivation for caches: a program alternating between a bit of math and a memory load, if it consistently misses all the way to DRAM, can lose roughly two orders of magnitude in effective speed compared to hitting in L1 — which would completely swamp, say, a 4x speedup you might otherwise get from parallelizing across four cores.

---

## 3. The Running Example: Computing `sin(x)`

The lecture uses one recurring piece of code — a Taylor-series approximation of sine applied element-wise across an array — to walk through each new hardware idea:

```c
void sinx(int N, int terms, float* x, float* y)
{
    for (int i = 0; i < N; i++)
    {
        float value = x[i];
        float numer = x[i] * x[i] * x[i];
        int denom = 6;   // 3!
        int sign = -1;
        for (int j = 1; j <= terms; j++)
        {
            value += sign * numer / denom;
            numer *= x[i] * x[i];
            denom *= (2*j+2) * (2*j+3);
            sign *= -1;
        }
        y[i] = value;
    }
}
```

Compiled naively, this becomes one scalar instruction stream that processes one array element at a time on one core. The rest of the lecture is essentially: *how do we go from "one core, one element at a time" to using the full width of a modern chip?*

---

## 4. Instruction-Level Parallelism, Revisited: Why Hardware Alone Isn't Enough

Recall superscalar execution from Lecture 1: a core can automatically find independent instructions within a *single* instruction stream and run more than one per clock (the term literally breaks down as "more than scalar" — scalar meaning one-at-a-time).

Applying that lens to the compiled inner-loop body of `sinx`: there's a little bit of independence (e.g., computing the numerator update and the denominator update don't depend on each other), so a two-wide superscalar core could, in principle, execute a couple of these instructions simultaneously. But most of the instructions in this tight loop body form a dependency chain — each step needs the result of the last — so the *amount* of exploitable ILP within one loop iteration is small.

**The real parallelism in this program is at a completely different scale**: every iteration of the outer loop (i.e., every array element) is independent of every other iteration. That is enormous, obvious parallelism — but it's invisible to superscalar hardware, which only ever examines a narrow, local window of nearby instructions at each clock tick. Recognizing that an entire `for` loop's iterations are mutually independent would require reasoning across the whole loop structure — a kind of global, whole-program analysis that's far too expensive to do dynamically in hardware at clock-tick timescales.

This is why the burden shifts to software: rather than have hardware guess, the **programmer (or a compiler operating on very regular, tensor-like code) explicitly declares** which pieces of work are independent, and hardware/runtime systems take it from there. This reframing — hardware finds *micro* parallelism automatically; software must expose *macro* parallelism explicitly — motivates everything that follows in the lecture.

---

## 5. Idea #1 — Multi-Core: Trade Single-Thread Sophistication for More, Simpler Cores

### The transistor budget trade-off
A "pre-multi-core era" processor spends the bulk of its transistors making a *single* instruction stream run fast: big data caches, aggressive out-of-order execution logic, sophisticated branch predictors, hardware prefetchers.

**Idea #1** is to spend that same transistor budget differently: strip out much of that single-thread-acceleration machinery (smaller caches, less out-of-order logic, etc.) and instead **duplicate the entire core** — fetch/decode unit, execution unit(s), *and* the register file/execution context — as many times as will fit. This is a genuinely different structural choice than superscalar execution: a superscalar core duplicates *execution units* but keeps one shared register file and one instruction stream; a multi-core chip duplicates the *whole* core, so each core has its own independent registers and can run a fully separate instruction stream.

Each individual simplified core might run a single instruction stream somewhat slower than the original "fancy" core (the lecture uses an illustrative ~25% slowdown), but with two of them running two different instruction streams in parallel, you can come out ahead — e.g., two cores at 75% of the original single-core speed still nets roughly 1.5x total throughput, and that advantage compounds as more cores are added and kept busy.

### The catch: your code has to actually express that parallelism
Compiled as ordinary, unmodified C, `sinx` is still just **one** instruction stream. Dropped onto a multi-core chip built from "simpler, slightly slower" cores, it would only ever occupy one core — and since that core is individually weaker, the *unparallelized* program would actually run **slower** than before. Multi-core hardware only helps once software is restructured to create multiple independent instruction streams.

A natural question: could the compiler just find this parallelism automatically and do the restructuring for you? Sometimes, for very regular, tensor-like code, advanced compilers can. But in general the compiler can't safely prove that loop iterations are independent once code gets even a little more complex (aliasing, unpredictable indexing, etc.) — so in practice this responsibility is pushed up to the programmer or to higher-level parallel-language constructs, rather than relied upon from hardware or the compiler.

### Two ways to expose the parallelism

**1. Explicit threading**, e.g. with C++ `std::thread`: manually split the array's index range, hand half to a spawned thread, do the other half on the main thread, then join. From the operating system's perspective, this just produces two independent instruction streams, which it's free to schedule onto two different cores. Note this only happens because the *programmer* created two threads — nothing about the hardware or a "smart" compiler does this automatically.

**2. Data-parallel language constructs** — a hypothetical `forall` loop where the programmer simply *declares* that loop iterations are independent, leaving it to the compiler/runtime to decide how many actual threads to spawn based on the machine it's running on (2 threads on a dual-core chip, 16 threads on a 16-core chip, etc.), without the programmer having to hand-write and hard-code that thread-splitting logic themselves. This mirrors real constructs in modern parallel languages (e.g., PyTorch's implicit parallel ops, OpenMP's parallel-for, ISPC's `foreach`) and is the abstraction used throughout the rest of the course.

The key property that makes `sinx` a great fit for this: every loop iteration runs the *exact same sequence of instructions*, just on different data (`x[i]`) — that "same instructions, different data" shape is exactly what unlocks the next idea.

### Real hardware examples
- A 10-core Intel CPU, visibly divided into per-core regions on a chip die photo.
- NVIDIA RTX 4090: ~144 replicated processing blocks ("SMs" in NVIDIA's terminology) at that same granularity of duplication.
- Apple's A15 Bionic: 6 CPU cores, but *heterogeneous* — 2 "big" cores retaining more of the fancy out-of-order/large-cache machinery for fast single-thread performance, plus 4 "small" cores with much of that sophistication stripped out, aimed at workloads with more available parallelism.

---

## 6. Idea #2 — SIMD: Amortize Instruction Control Over Many ALUs

Given code shaped like `forall`-`sinx` — same instructions, independent data — there's a second lever besides spinning up more full cores: keep **one** fetch/decode unit, but attach it to **multiple** ALUs (e.g., 8), and widen the registers to match (turning scalar 32-bit registers into 256-bit, 8-wide vector registers). A single fetched instruction (e.g., "multiply") is then broadcast across all 8 ALUs simultaneously, each operating on its own data lane.

Concretely, `sinx` can be rewritten using AVX vector intrinsics (`__m256`, `_mm256_mul_ps`, etc.), turning each scalar variable into an 8-wide vector type and changing the loop increment from `i++` to `i += 8`. This compiles down to genuine vector instructions (`vmulps`, `vloadps`, etc.) operating on wide vector registers instead of scalar ones — the instruction stream is still singular, but each instruction now does 8x the work.

**Why this is a cheap way to get more throughput**: building extra arithmetic units (ALUs) is relatively inexpensive compared to the control logic (fetch/decode, out-of-order scheduling, etc.) needed to manage an independent instruction stream. SIMD lets you amortize that one, comparatively expensive control cost across many comparatively cheap ALUs — which is a big part of why chip designers favor it over just building 8x more full cores, when the workload allows it.

### Combining SIMD with multi-core
Stack the two ideas: 16 cores, each with an 8-wide SIMD unit, gives 16 independent instruction streams × 8 data lanes each = **128 execution units total**, in principle offering up to ~128x the throughput of the single-core, single-lane starting point (assuming everything is perfectly parallelized and there's no other bottleneck, like memory access, involved — a caveat the lecture explicitly flags and defers).

---

## 7. Handling Conditional Execution in SIMD (Divergence & Masking)

SIMD assumes all lanes want to do the *same* thing each instruction. An `if`/`else` inside a `forall` loop breaks that assumption, since different data elements may take different branches.

**How hardware copes:** it executes *both* branches, across all lanes, and simply **masks off** (discards) the results for lanes that shouldn't have taken that particular path — e.g., run the `if`-body across all 8 lanes, then throw away the results for lanes that evaluated false; then run the `else`-body across all 8 lanes and keep only the results for lanes that evaluated true. Once the branch fully resolves and execution reconverges, the SIMD unit returns to full-width efficiency.

- Worst case: only 1 of 8 lanes actually needed the work being done in a given instruction → as low as **1/8 peak utilization** on that portion of the computation (or 1/32 on a 32-wide GPU SIMD unit).

### Classroom exercise: constructing the true worst case
The instructor posed a challenge: using only a single `if` statement, construct code that runs at the worst possible 1/8 utilization for the *entire* execution (not just part of it).

- A first (reasonable) student proposal was **predication**: eliminate the branch entirely by multiplying each branch's result by a boolean mask (1 or 0) and summing them — a real, valid compiler technique for lowering branches into branch-free code. But this doesn't actually fix the *utilization* problem: it just guarantees you always execute both paths' full cost with no branch at all — same total work, same efficiency loss, just implemented differently.
- The actual answer requires shaping *both* the branch taken **and** its relative cost: e.g., data arranged so exactly 1 of every 8 elements takes the (cheap) `if` path and the other 7 take a much more expensive `else` path (or vice versa) — so that, averaged over time, the SIMD unit is almost always burning cycles on an expensive branch that only a small minority of lanes actually needed, dragging sustained utilization down toward 1/8. Simply having *some* imbalance (say, an even/odd split) only costs you 50% utilization on its own — true 1/8 worst-case requires the specific combination of an 8-way split *and* a proportionally larger cost on the branch most lanes don't take.

---

## 8. Terminology & Real SIMD Widths

- **Coherent (instruction stream) execution** — different data elements are all following the same instruction sequence. Coherence is *required* for SIMD hardware to run efficiently; it is *not* required for efficient multi-core parallelism, since each core independently fetches/decodes its own instruction stream regardless of what other cores are doing.
- **Divergent execution** — the opposite: a lack of coherence (e.g., from data-dependent branching), which is exactly what erodes SIMD efficiency.

Real-world SIMD widths range from around 4-wide on some mobile/ARM SIMD instruction sets, to 8-wide on mainstream CPU AVX2 instructions, up to roughly 32-wide on modern high-end GPU hardware — meaning divergence-heavy GPU code can, in the worst case, run at a small fraction of peak throughput. (Historically, x86 SIMD started narrower still — early 2000s SSE-era instructions operated on 4-wide vectors, aimed initially at simple graphics operations.)

---

## 9. Checkpoint — Three Orthogonal Forms of Parallel Execution

| Mechanism | What it parallelizes | Who finds/exposes the parallelism | Structural cost |
|---|---|---|---|
| **Superscalar** | Different instructions *within one instruction stream* (one core, one register file) | Discovered automatically by hardware at runtime, from a narrow local window | Duplicates execution units only |
| **SIMD** | One instruction applied across many ALUs/data lanes (still one instruction stream) | Exposed by the program/compiler (vector code, `forall`, or auto-vectorization) | Duplicates ALUs, amortizes one shared fetch/decode unit across them |
| **Multi-core** | Completely independent instruction streams | Exposed by software via threads | Duplicates the *entire* core: fetch/decode, execution units, and register file |

These combine freely — real chips mix and match them, and peak throughput is simply the product of the relevant factors. For example: a core that's 3-way superscalar and uses 8-wide vector instructions has 24 execution units; four such cores gives a chip with 96 execution units total, i.e. up to 96 operations per clock. An NVIDIA GPU with ~80 SM cores, each with ~128 ALUs (organized internally as 32-wide SIMD), multiplies out to a very large peak-operations-per-clock figure the same way.

---

## 10. Idea #3 — Multi-Threading: Hiding Memory Latency, Not Just Computing More

### Motivation
Even with caches and hardware prefetching (speculatively loading data the processor guesses will be needed soon), some access patterns are inherently hard to predict — e.g., using a just-computed value as an array index (`y = A[x]`), or walking a linked structure. And packing a chip full of ALUs (via multi-core and SIMD) only makes the underlying memory-latency problem *more* pressing: more execution units means more memory requests need to be satisfied to keep them all fed, often with smaller per-core caches than before (since some of that die area went toward extra cores/ALUs instead).

### The everyday-life analogy
The instructor's framing: if you're doing laundry, you don't put a load in the washer and then sit and stare at it until it's done — you go do something else useful (cook dinner, homework, etc.) and come back to it once it's ready. The same logic applies to boiling water while prepping other parts of a meal. The general principle: **if something you need isn't ready yet, and you have other useful work available, go do that other work instead of idling.**

### Applying this inside a processor core
Take a core and, this time, duplicate **only the execution context** (registers / thread state) — not the fetch/decode logic or the ALU, which stay shared. This is the key structural difference from multi-core: multi-threading (as covered here) shares the actual execution hardware across multiple threads' worth of *state*, whereas multi-core duplicates the execution hardware itself.

With, say, 4 duplicated hardware thread contexts on one core: the processor runs instructions from thread 1 until it hits a long-latency memory operation (e.g., a ~250-cycle DRAM miss) that stalls it. Rather than idle, the core immediately switches to thread 2's instructions, continuing until *it* stalls, then thread 3, then thread 4 — and by the time thread 4 also stalls, enough time may well have passed that thread 1's original memory request has already come back, so the core can resume thread 1 right where it left off. Net effect: the core's ALU stays continuously busy, even though any *individual* thread's own wall-clock completion time is now longer than it would have been running alone (since it's periodically paused in favor of others).

### Worked utilization example
Given a repeating thread pattern of 3 arithmetic instructions followed by a 12-cycle memory stall:
- 1 thread → core is busy only 3 of every 15 cycles → **20% utilization**
- 2 threads (interleaved) → **40% utilization**
- 5 threads → enough concurrent work to fill every stall gap → **100% utilization**
- Adding more threads beyond that point provides no further benefit — the core is already saturated.

If the ratio of arithmetic-to-memory-latency improves (e.g., 6 arithmetic instructions per stall instead of 3), *fewer* threads are needed to reach full utilization, since each thread now has more useful work to fill the gap on its own.

**Takeaways:**
1. Multi-threading doesn't reduce memory latency itself — it hides its cost by keeping the core continuously busy with other threads' work during the wait. (The latency of any single memory operation is unchanged; only overall core utilization improves.)
2. Programs with a higher ratio of computation to memory access need fewer threads to fully hide that latency.

### Two flavors of hardware multi-threading
- **Interleaved (temporal) multi-threading** — each clock cycle, the core picks one thread and runs an instruction from it (what's described above).
- **Simultaneous multi-threading (SMT)** — each clock cycle, the core can issue instructions drawn from *multiple* threads at once (e.g., Intel Hyper-Threading, 2 threads/core).

---

## 11. Pulling It All Together

To use a modern parallel processor efficiently, a program needs to satisfy three conditions simultaneously:
1. **Enough parallel work** to occupy every execution unit across every core.
2. **Coherent groups of that work** — large groups doing the same sequence of instructions — to make SIMD execution efficient.
3. **More parallel work than there are ALUs**, so extra threads can be interleaved to hide memory stalls.

---

## 12. Closing Thought Experiment (Sets Up Next Lecture)

Consider element-wise multiplication of two huge vectors: for each `i`, load `A[i]`, load `B[i]`, multiply, store into `C[i]`. Is this a good fit for a modern throughput-oriented parallel processor?

The lecture leaves this open, noting that answering it properly requires understanding the difference between **latency** and **bandwidth** — picked up at the start of the next lecture.

---

## 13. Terms Worth Knowing From This Lecture
- Cache line, cold/compulsory miss, capacity miss, conflict miss, LRU eviction
- Spatial locality, temporal locality
- Instruction stream
- Multi-core processor
- SIMD execution (explicit vs. implicit), predication
- Coherent vs. divergent control flow
- Hardware multi-threading (interleaved vs. simultaneous/SMT)

---

*Notes synthesized and paraphrased from the CS149 Fall 2023 Lecture 2 slide deck and lecture transcript, for study purposes — not a verbatim transcript.*
