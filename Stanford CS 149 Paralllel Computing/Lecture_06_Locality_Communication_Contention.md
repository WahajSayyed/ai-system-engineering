# CS149 — Lecture 6: Performance Optimization Part 2: Locality, Communication, and Contention

**Course:** Stanford CS149, Parallel Computing (Fall 2023)
**Instructors:** Prof. Kayvon Fatahalian & Prof. Kunle Olukotun
**Video:** https://www.youtube.com/watch?v=Mhdny2JNhmc
**Course site:** https://gfxcourses.stanford.edu/cs149/fall23/lecture/perfopt2/

*Notes combine the official slide deck with the lecture transcript (live Q&A, worked examples).*

---

## 1. Framing for This Lecture

Last time was about keeping workers busy (good workload balance with minimal overhead). Today adds the other major axis: **reducing the cost of communication and synchronization** between processors, and between a processor and memory. Time permitting, the lecture closes with some general program-optimization advice. This is also the last of the software/performance-focused lectures before the course shifts toward more hardware-centric material.

---

## 2. The Hidden Complexity Behind "Shared Memory"

Every program and assignment so far has assumed a single, simple abstraction: all processors connected to one shared address space, where any thread can read or write any variable. That abstraction is simple — its real hardware implementation is not. On a real multi-core CPU, the data nominally living in one shared address space is actually scattered across per-core caches, with multiple cores potentially holding their own copies of the same address (setting up the cache-coherence problem, covered in an upcoming lecture on hardware).

Real chips connect all their cores, caches, and memory controllers together via genuinely complex on-chip networks:
- **Ring interconnect** (e.g., Intel Core i7-era CPUs): cores, slices of shared L3 cache, and other on-chip agents (like an integrated GPU) sit as nodes on one or more physical rings. A load or store to some address isn't a magic direct connection — it's a message that has to be routed around the ring to wherever the relevant data (or memory controller) actually lives. Each L3 cache "slice" is deliberately connected to the ring in **two** places rather than one — this simplifies traffic direction (helping avoid deadlock, since messages can be kept flowing consistently in one rotational direction) and reduces average latency, since a request never has to travel more than halfway around the ring to reach its nearest connection point.
- **Crossbar interconnect** (e.g., Sun/Oracle's UltraSPARC T2 "Niagara," an early mainstream multi-threaded chip design): every core is wired directly to every other node, giving full point-to-point connectivity at the cost of roughly *N²* wiring complexity for *N* cores. On a real chip die photo, the physical area devoted to this network can be comparable to the area devoted to the processing cores themselves — high-bandwidth, low-latency interconnects between many cores are genuinely expensive to build.

**The upshot**: on any real machine, different memory addresses are not all equally "close." A cache is sharded into slices reachable at different costs depending on which core is asking; a dual-socket motherboard has cores on one physical chip that can reach their own local memory much faster than memory attached to the *other* socket, independent of caching behavior entirely. Most of the time, programmers can't afford to think about this level of detail — but genuinely performance-critical code sometimes has to reason about exactly *where* in the machine a given piece of data physically lives, not just whether it's "in cache" or "in memory."

---

## 3. A Different Communication Model: Message Passing

Rather than reasoning about all this hidden hardware complexity underneath a shared address space, it can be clarifying to consider an entirely different **abstraction** for communication, where the cost and existence of communication is explicit in the program itself: **message passing** — the same basic model behind ordinary distributed/networked programming (e.g., an HTTP request/response is conceptually a message).

### The core idea
Each thread operates in its **own, private address space** — address `X` in thread 1's memory has nothing to do with address `X` in thread 2's memory; they are simply different pieces of storage. The *only* way for one thread to get data that lives in another thread's address space is to explicitly **send** it a message and have the other thread **receive** it. A useful metaphor: a shared address space is like a public bulletin board anyone can read or write freely; message passing is more like conventional mail — you package data up, address it explicitly to a destination, and someone has to actively receive it.

### Reworking the grid solver for message passing
Imagine running the earlier red-black grid solver not on a shared-memory multi-core machine, but on a small cluster of separate computers connected only by a network (Ethernet, or conceptually even "carrier pigeon" — the transport mechanism doesn't matter to the abstraction). Each node now has its **own private memory**, holding only its own slice of the grid — there are now genuinely separate allocations across separate address spaces, rather than one shared array with different threads simply agreeing to touch different parts of it.

To update a boundary cell, a node needs neighboring values that live in a *different* node's private memory — data it fundamentally cannot reach with a plain load instruction. The standard solution: **over-allocate** a small border region in each node's local array (one extra row above and below, in this row-partitioned example) to hold a **local copy** of the neighboring data it needs — commonly called **ghost cells** (or ghost rows) in scientific computing. Each node explicitly sends its own boundary values to its neighbors, and receives their boundary values into its own ghost rows, before proceeding with the actual computation — after which the rest of the update logic can be written exactly as if all the needed data had always been local (by copying incoming messages directly into the appropriate ghost-row storage), keeping the core computational code simple.

A full message-passing iteration of the solver, then, naturally breaks into phases: **(1)** exchange boundary data with neighbors, **(2)** perform the local update using now-complete local (+ghost) data, **(3)** send updated boundary data back out to neighbors for their next iteration, and **(4)** determine collectively whether the whole computation has converged.

### Notice: no locks, no barriers
Unlike the shared-address-space version of this same solver, this message-passing version has **no locks and no barriers anywhere** — and structurally, it *can't*, because there's no shared state to protect in the first place. Synchronization instead emerges naturally from the communication pattern itself: e.g., having every non-coordinator thread **send** its local partial convergence value to one designated thread, which **receives** all of them, computes the aggregate result, and **sends** the final "are we done?" decision back out to everyone — the same overall effect as a barrier plus a shared reduction variable, but achieved entirely through the structure of who sends what to whom, and when.

### Blocking sends and receives — and a subtle deadlock
The simplest version of send/receive is **blocking**: a `send` call doesn't return until the receiver has actually received the data (in the simplest model, potentially involving an underlying acknowledgment back to the sender); a `receive` call doesn't return until matching data has actually arrived — if nothing is ever sent, a blocking receive simply waits forever. (Failures — e.g., a lost network packet — are treated, for this course's purposes, the way you'd treat a failed memory access on a single chip: some underlying layer is assumed to retry/guarantee delivery, and the abstraction just doesn't return until that succeeds.)

**A live bug hunt**: given a first-draft version of the exchange phase where every node unconditionally calls `send` (to, say, its neighbor "behind" it) *before* calling `receive`, what happens? Every single node ends up blocked waiting for its own `send` to be acknowledged — but the neighbor it's sending to is, symmetrically, *also* stuck waiting on its own outbound `send`, rather than having reached its matching `receive` yet. **Nobody ever receives anything, and the whole system deadlocks permanently** — a direct illustration of how a small, easy-to-miss structural mistake in a message-passing program can hang the entire system with zero forward progress, rather than just producing a slightly wrong (but completed) answer.

**A fix using only blocking calls**: stagger the order in which different nodes send versus receive first — e.g., pair nodes up (by index parity), so one side of each pair sends first while the other receives first, breaking the circular wait.

### Asynchronous send/receive: a different way to avoid this class of bug
An alternative to being careful about ordering: make send and receive **asynchronous**. An async `send` returns immediately, handing back a **handle** the caller can later use to check whether the message has actually gone out (`check_send_status(handle)`); an async `receive` similarly returns immediately with a handle to poll for arrival later. This trades away the risk of the specific deadlock above, but introduces new responsibilities and new potential bugs of its own:
- The data being sent must not be modified by the calling thread until the send is *confirmed* complete — modifying it before that point (or, worse, deleting it) risks sending corrupted or already-freed data, since the underlying library might not have actually copied it out yet. (A useful metaphor: it's like leaving a package on your porch for a courier and then changing what's inside before they actually arrive to pick it up.)
- There's generally **no guaranteed ordering** between multiple asynchronous sends unless a specific messaging library explicitly promises it — two async sends from the same thread could, in principle, arrive at their destination in either order.
- Every message is conceptually tagged with some identifying information (an ID, a designated sender, etc.), and receivers can choose to wait for a message matching specific criteria, or just for "the next message, whatever it is" — the specifics are a detail of whatever messaging library/API is actually being used.
- Introducing asynchrony can make some problems easier (sidestepping certain deadlock patterns) but also strictly adds *more* concurrency to reason about, which can introduce entirely new classes of bugs that a purely synchronous, blocking design would never have allowed in the first place.

### Message passing isn't only for clusters
Even on a single, shared-memory multi-core machine with no separate address spaces at all, some developers deliberately choose to write code using an explicit message-passing style (e.g., sending messages between OS processes rather than sharing memory maps) specifically *because* locks and shared mutable state are hard to get right. Message passing forces communication to be fully explicit and structured up front — which can make it easier to reason about, debug, and performance-tune, at the cost of more upfront design discipline than just grabbing a shared variable and a lock.

---

## 4. Communication Is a General Concept, Not Just a Network Thing

A key reframing: "communication" isn't specific to clusters of separate machines — it's the same underlying concept whether it's happening between a processor core and its own registers, between a core and its L1/L2/L3 cache, between a core and local DRAM, between two cores on the same chip, or between two entirely separate computers over a network. This reframing helps explain, in retrospect, exactly *where* the memory latency discussed several lectures ago actually comes from — an L1 lookup, potentially an L2 lookup, potentially a TLB miss, and ultimately (on a full cache miss) an actual message sent out to memory requesting data, which memory then streams back at whatever bandwidth the interconnect supports.

Revisiting the earlier "math, math, load" memory-bandwidth-bound execution trace from a prior lecture: the key quantity that determines overall throughput isn't really the *latency* of any individual memory request (which can be hidden, given enough outstanding requests or multithreading) — it's how much of the time the memory system itself is kept continuously, fully busy (the width of the blue "memory transferring data" bar in that diagram) versus how much time the processor spends stalled waiting on it. That, in turn, comes down to a single governing ratio: **how much useful math you do for every unit of data moved** — a quantity given a name below.

---

## 5. Inherent vs. Artifactual Communication

It's useful to split communication costs into two categories:
- **Inherent communication**: communication that has to happen, fundamentally, because of the structure of the algorithm and how work has been assigned to processors — no matter how cleverly implemented, this data genuinely has to move for the computation to produce a correct answer.
- **Artifactual communication**: extra communication that arises purely from the *details of how real machines work* — e.g., the fact that memory is always moved in fixed-size chunks (cache lines, network packets of some minimum size) rather than exactly the individual bytes actually needed.

### Reducing inherent communication: reassigning work (tiling)
Partitioning an N×N grid solver's work into **row blocks** across P processors: each processor does roughly N²/P work, but must communicate roughly 2N boundary values (its top and bottom row) with neighbors — giving an **arithmetic intensity** (work done per unit of communication) of roughly N/P.

An **interleaved** row assignment (rather than contiguous blocks) is dramatically worse: since a processor's own rows are scattered throughout the grid rather than clustered together, it ends up needing to exchange data with many more distinct neighbors, and communicates roughly 2 elements' worth of data *per row* it owns — arithmetic intensity collapses to a small constant, independent of problem size, making this scheme far more likely to become communication-bound as the problem or processor count grows.

**Can the blocked (row) scheme be improved further?** Yes — by switching from 1D row-blocks to a **2D tiled** assignment (dividing the grid into roughly square blocks, e.g., splitting both dimensions across √P processors along each axis instead of only splitting rows). Since arithmetic intensity is essentially the ratio of a tile's *area* (proportional to its total work) to its *perimeter* (proportional to the boundary data it must communicate), and a square has the best area-to-perimeter ratio of any simple shape subdividing a plane this way, 2D tiling improves arithmetic intensity from roughly N/P (1D blocking) to roughly **N/√P** — a genuinely large improvement at high processor counts (e.g., a full **4x** reduction in required communication-to-computation ratio on a 16-core machine, compared to row-blocking, for the same problem). This is a concrete example of **reducing inherent communication by changing the assignment of work to processors** — the total useful computation is unchanged, but the shape of the partition itself directly determines how much data absolutely must move.

### Reducing artifactual communication: cache blocking (loop reordering)
Now consider the *same* grid solver running on a **single** thread/core (no parallelism at all — this is now about communication between a processor and its own local memory, not between separate processors). Walking row-by-row across the grid, the four cache lines needed to compute one cell mostly stay resident in cache while moving *along* a row (since the relevant data was just loaded moments ago) — but by the time execution wraps back around to the start of the *next* row, the data directly above (from the row before) has long since been evicted from a small, finite cache. The result: substantial numbers of avoidable cache misses purely from the *order* in which memory happens to be visited — a clear case of **artifactual communication**, since nothing about the algorithm itself required re-fetching that data; it's purely a consequence of finite cache capacity and a memory-access order that doesn't exploit it well.

**Cache blocking**: reorder the traversal — e.g., a "swept" or blocked pattern that processes a narrow band of a few rows at a time before moving on, rather than one full row across the entire width before returning — so that by the time execution needs data from a nearby row again, it's still likely to be sitting in cache rather than long evicted. Reordering this way (without changing what's actually computed, or introducing any parallelism at all) meaningfully improves the ratio of useful output produced per cache line actually loaded — a substantial real improvement in effective bandwidth usage, achieved purely through *when* data is touched rather than changing the underlying computation. This general technique — restructuring loop order to maximize reuse of data already sitting in cache — is presented as one of the single most important optimizations in any code involving matrices or tensors, and is a large part of why well-tuned matrix-multiplication and tensor libraries dramatically outperform naive implementations of the same mathematical operation.

### The same idea, applied to a chain of vector operations
A very common real-world pattern: composing a sequence of separate, simple vectorized library calls (e.g., add two arrays, then multiply the result by a third array, then subtract a fourth — exactly the style of code common in NumPy-like libraries), where each individual library call has some fixed, modest arithmetic intensity (e.g., loading two values and a writing one result per math operation). Composed naively, one full array-sized pass is made *per operation* — meaning the overall arithmetic intensity of the whole expression is no better than that of any single one of its component operations, since intermediate results are all written out to memory and re-read by the next stage.

**Fusing** these operations — restructuring the code to loop over the array just *once*, computing the *entire* expression for each element (using registers/cache for intermediate values, rather than writing each intermediate all the way back out to memory) before moving to the next element — substantially raises arithmetic intensity, simply by reducing how many total memory loads/stores are needed to get the same final answer. This exact optimization ("loop fusion" or "kernel fusion") is one of the techniques modern deep-learning compilers (e.g., a PyTorch/TensorFlow JIT compiler) apply automatically: code is *written* in the convenient, composable, operation-at-a-time style, but *executed* in the far more memory-efficient, fused style.

---

## 6. Arithmetic Intensity and the Roofline Model

**Arithmetic intensity** (sometimes called *operational intensity*, or its reciprocal referred to as the *communication-to-computation ratio*): the number of math operations performed per unit of data moved from memory — typically expressed as FLOPs per byte. Whenever a program is bandwidth-bound rather than latency-bound (i.e., latency can be adequately hidden via multithreading, prefetching, or similar techniques), this single ratio ends up being *the* quantity that determines achievable performance, since it directly governs how much useful work gets done per byte the memory system has to supply.

### The roofline model
A **roofline plot** shows achievable performance (operations/second, e.g., GFLOPs) on the y-axis against arithmetic intensity (FLOPs/byte) on the x-axis, for a specific piece of hardware — producing a characteristic "roof" shape:
- **The flat top ("compute roof")**: at sufficiently high arithmetic intensity (few enough memory accesses relative to computation), performance flattens out at the machine's absolute peak compute throughput — the program is **compute-bound**, and further increases in arithmetic intensity yield no further speedup, since the arithmetic units are already fully saturated.
- **The sloped left side ("memory roof")**: at lower arithmetic intensity, performance instead falls off in direct proportion to arithmetic intensity — the program is **memory-bandwidth-bound**, and its achievable throughput is set entirely by how fast the memory system can supply data (the slope of this region of the curve *is*, in effect, the machine's memory bandwidth).

**A given machine's roofline is a fixed, fundamental property of its hardware** — but any specific *program*, once profiled or reasoned about, lands at one particular point on the x-axis (its own arithmetic intensity) and can be compared against the curve: sitting right on the roofline (at either the flat top or the sloped side, whichever applies at that arithmetic intensity) means the program is achieving the best performance physically possible on that hardware for its current arithmetic intensity; sitting noticeably *below* the roofline signals genuine headroom — something other than raw compute or bandwidth (e.g., poor workload balance, unnecessary synchronization overhead, or contention) is holding performance back, and is worth investigating.

**Comparing machines**: a chip built with substantially more raw compute capability (e.g., roughly 4x more, in one comparison used in lecture) but the *same* memory system pushes the compute-roof plateau proportionally higher — but also pushes the "knee" of the roofline (where the flat compute roof meets the sloped memory roof) rightward, toward a higher required arithmetic intensity. **The practical implication**: simply adding more parallel compute capability to a chip only translates into real speedups for workloads whose arithmetic intensity is high enough to actually reach that chip's new, higher compute roof — an underlying reason why hardware and software co-design (matching available compute to workloads with sufficient arithmetic intensity) matters so much in practice.

### An important nuance: is moving down the curve ever a good idea?
A natural question: since sitting further right on the curve (higher arithmetic intensity, ideally at or near the compute roof) is generally better, is it ever acceptable to make a change that *reduces* arithmetic intensity? The answer hinges on what ultimately matters: **total wall-clock time**, which is (amount of work) × (time per unit of work) — i.e., a function of both throughput (position on the roofline) *and* the total amount of work actually being done. If an algorithmic change moves a program to a lower arithmetic intensity but *also* substantially reduces the total amount of work required (e.g., a smarter algorithm that's less arithmetically dense but does far fewer total operations), while still landing on (or near) the roofline at that new intensity, the net result can still be a real, meaningful speedup — because doing much less work, even somewhat less efficiently per byte, can still finish faster in absolute terms. Conversely, a change that reduces arithmetic intensity only modestly, in exchange for only a modest reduction in total work, can end up being a net wash, or even a regression — the roofline model captures achievable *throughput*, not overall *runtime*, and both matter.

---

## 7. A Wrinkle: Contention

Everything above reasoned about communication cost in terms of *averages* — bytes moved per operation, assuming requests are spread out reasonably evenly over time. In practice, if many requests for a shared resource all happen to arrive at (or near) the same moment, that resource can behave as if it were much slower than its true average-case capability, simply because requests end up queued behind one another.

**An office-hours analogy**: if a student's total time cost is (walk over) + (wait in line, if any) + (get question answered), the *first* student to arrive pays no queuing delay at all, while a student arriving right after several others — even though their own question takes exactly the same time to answer — ends up paying a real, avoidable cost purely from bad luck in *when* they happened to show up relative to everyone else. By contrast, a fixed appointment schedule guarantees every student the same, minimal total time, regardless of arrival order — because it deliberately avoids simultaneous contention for the same shared resource (the instructor's attention) in the first place.

This maps directly onto shared computing resources: if every processor happens to issue a memory request at the exact same moment, memory can behave far more like a queue with real waiting time than the clean, steady-bandwidth pipeline diagrams from earlier lectures would suggest — echoing the same underlying problem already seen with a single shared work-queue lock or a single shared accumulator variable, just now applied to memory access itself. One practical mitigation technique mentioned: deliberately **randomizing** the timing of requests from different sources (the lecture's own analogy: if everyone leaves for the highway at a randomized time rather than all at once, the highway is more likely to sustain something close to its true peak throughput) — reducing the odds that many requests cluster together and induce queuing delay that wouldn't show up in a simple average-case analysis.

---

## 8. General Program Optimization Tips

- **Always implement the simplest possible solution first.** A dumb, unoptimized, purely static solution is very often faster than a complicated one — measure before investing further effort, and if the simple version is already fast enough, stop there.
- **When performance isn't satisfying, ask: is there realistically room to do better?** Beyond the workload-balance measurements from the previous lecture, it's worth directly asking whether a program is **bandwidth-bound**. Dedicated profiling tools (e.g., Intel VTune or similar) can often report this directly — and even without such a tool, comparing a program's achieved operations/second against a machine's known peak operations/second gives a useful first estimate of headroom. Profilers can *tell you* that you're bandwidth-bound; they generally can't tell you *how* to fix it — that still requires understanding techniques like the ones covered in this lecture (reassignment to reduce inherent communication, reordering/blocking to reduce artifactual communication, fusion to raise arithmetic intensity, and so on).

---

## 9. Summary

- Real hardware implementations of a "shared address space" involve genuinely complex on-chip and inter-socket networks (rings, crossbars), meaning not all memory accesses are equally costly in practice, even though the programming abstraction hides this.
- **Message passing** is an alternative communication model, with private per-thread address spaces and explicit send/receive — making communication cost visible directly in the code, at the price of needing careful reasoning about blocking semantics (to avoid deadlock) or additional care with asynchrony.
- Splitting communication into **inherent** (unavoidably required by the algorithm/assignment) and **artifactual** (arising purely from real hardware behavior, like fixed-size cache lines) gives two distinct levers for optimization: **reassigning work** (e.g., 2D tiling instead of row-blocking) reduces inherent communication; **reordering memory access** (e.g., cache blocking, loop fusion) reduces artifactual communication.
- **Arithmetic intensity** — math operations performed per byte moved — is the single quantity that determines achievable performance for any bandwidth-bound program, and the **roofline model** is a compact way to visualize a machine's fundamental compute/bandwidth trade-off and see exactly where a given program's performance stands relative to what's physically achievable.
- **Contention** — many requests for a shared resource clustering together in time — can make real-world performance meaningfully worse than simple average-case bandwidth/latency reasoning would predict, and is worth remembering as a distinct concern from the "average bytes per operation" framing used throughout most of this lecture.

---

*Notes synthesized and paraphrased from the CS149 Fall 2023 Lecture 6 slide deck and lecture transcript, for study purposes — not a verbatim transcript.*
