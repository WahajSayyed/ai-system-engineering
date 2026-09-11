# CS149 — Lecture 4: Parallel Programming Basics

**Course:** Stanford CS149, Parallel Computing (Fall 2023)
**Instructors:** Prof. Kayvon Fatahalian & Prof. Kunle Olukotun
**Video:** https://www.youtube.com/watch?v=0-ztm8SKq70
**Course site:** https://gfxcourses.stanford.edu/cs149/fall23/lecture/progbasics/

*Notes combine the official slide deck with the lecture transcript (live Q&A, worked examples, demos).*

---

## 1. Framing for This Lecture

Two parts: first, wrapping up ISPC (review of the SPMD "gang" abstraction, plus the `task` construct for multi-core execution) in response to a lot of conceptual questions from Assignment 1 office hours; then a general framework for thinking about *any* parallel program — decomposition, assignment, orchestration, and mapping — worked through via a full case study (a 2D grid solver), written two different ways.

### A note on why this feels harder than it should
A recurring observation from office hours: each of the four hardware ideas covered so far (multi-core, SIMD, hardware multi-threading, superscalar) is simple to reason about *in isolation*, and the written homework tests exactly that. But running on real hardware means all four are composing together simultaneously, and real, messy interaction effects show up that don't obviously follow from any single concept alone. The reassurance offered: this doesn't mean your understanding is wrong — it usually means two things you separately understand correctly are just now interacting, which takes a little more work to untangle, and gets easier with practice. (A fun aside: one past final project had a student interview random people on the street with course exam questions rephrased as everyday-life scenarios — e.g., prioritizing tasks, working around a bottleneck — and found people did surprisingly well without ever having studied CS, since a lot of parallel-computing intuition is just formalized common sense.)

---

## 2. ISPC Review: the Gang Abstraction, Reinforced

### Restating the abstraction cleanly
Calling an ISPC function from C++ does **not** mean "run this function once." It means: spawn `programCount` (the **gang size**, fixed at compile time — e.g., 8) copies ("instances") of the function, each with a different value of `programIndex`, and don't return control to the caller until every instance has finished. Nothing about this description says anything about implementation or parallelism yet — it's purely a statement of meaning.

A useful sanity check: this meaning is **satisfied** even by the simplest possible (and very much *not* what ISPC actually does) implementation — literally wrapping the function body in a `for` loop that runs it `programCount` times sequentially, incrementing `programIndex` each time. That's a valid implementation of the abstraction; it's just not one anyone would want, since it throws away all the parallelism ISPC exists to provide.

### Why gang size usually equals SIMD width
Since all `programCount` instances need to run "at once" conceptually, the natural, efficient implementation is to run them literally simultaneously, one per lane of a vector instruction — which is why gang size is normally set to match the hardware's SIMD width. If you request a larger gang size than the hardware's native vector width (e.g., a gang of 16 on 8-wide hardware), the compiler simply issues **multiple** vector instructions per logical "step" — and this can actually *help* performance, since the two resulting vector instructions are mutually independent and give the hardware's instruction scheduler more exploitable ILP to work with (tying directly back to the superscalar discussion from Lecture 3).

### A useful way to visualize instance-to-work mapping: draw the table
For any given ISPC program, it helps to draw a table: columns are program instances, rows are loop iterations, and cells hold whatever array index that instance touches on that iteration. Two programs from Lecture 3 differ only in this table's pattern:
- **Interleaved** assignment: on a given iteration, the 8 instances touch 8 *contiguous* memory addresses — meaning the required data can be pulled in with a single, efficient packed vector load.
- **Blocked** assignment: on a given iteration, the 8 instances touch addresses spread far apart — meaning that same load requires each lane to fetch from a different location entirely, needing a costlier "gather" operation, and potentially touching many different cache lines (or even different memory pages) for a single logical vector load.

This is a concrete illustration of why the *interleaved* version tends to be the more cache-friendly (and thus faster) choice specifically for this kind of single-core, SIMD-vectorized ISPC code — a conclusion that, notably, **doesn't automatically carry over** to every parallel context (see the grid-solver case study below, where the opposite assignment turns out to be preferable, for entirely different reasons).

### `foreach`: letting the system decide
Rather than hand-writing an interleaved or blocked index mapping, `foreach` just declares "the whole gang needs to collectively perform these N iterations," and leaves the actual mapping of iterations to program instances entirely up to the ISPC implementation. Multiple different implementations of the *same* `foreach` loop would all be equally valid: run everything on a single instance sequentially, interleave iterations across instances, block iterations across instances, or dynamically hand out "the next available iteration" to whichever instance is free — the abstraction doesn't commit to any particular one, and the current ISPC compiler happens to pick a memory-friendly static scheme (similar to interleaving) for good reasons, but a future version could reasonably do something different.

This is presented as a hallmark of a well-designed programming system: provide a low-level mechanism (manual `programIndex`/`programCount` control) for when a programmer genuinely knows better than the system, *and* a higher-level mechanism (`foreach`) with an obvious, well-defined mapping down to the low-level one — so that most of the time, programmers can default to the higher-level construct and trust the system to do a good (often better) job, while still having an escape hatch available.

### Gotchas: undefined behavior and outright bugs
Because ISPC exposes low-level detail, it's possible to write `foreach` code whose correctness silently depends on implementation choices the abstraction never promised:
- A loop that (under some condition) writes to `y[i-1]` in addition to `y[i]` can have **two different iterations writing to the same output location** — since `foreach` makes no promise about execution order, the final result is genuinely undefined, and could legitimately differ between ISPC versions or even between runs. The current compiler makes no attempt to catch this, and in general it's a hard problem to catch reliably (e.g., an arbitrary, data-dependent index like `x[a[i]]` makes this kind of aliasing impossible to detect just by looking at the source).
- Trying to accumulate into a `sum` declared `uniform` (`sum += x[i]`) inside a `foreach` is a compile-time type error — `x[i]` differs per instance, so there's no well-defined single value to accumulate into a variable that's supposed to be identical across all instances.
- Declaring `sum` as an ordinary (non-uniform) variable instead compiles, but is *also* wrong: it silently creates a **separate private copy per instance**, and the calling C++ code expects one single return value, not several.

### The correct pattern: private accumulation + explicit reduction
```c
export uniform float sum_array(uniform int N, uniform float* x)
{
    uniform float sum;
    float partial = 0.0f;
    foreach (i = 0 ... N)
    {
        partial += x[i];
    }
    sum = reduce_add(partial);
    return sum;
}
```
Each instance privately accumulates its own `partial` with zero cross-instance communication during the loop; only `reduce_add` — a built-in cross-program-instance library function — combines every instance's private value into one shared answer at the end. Under the hood, this compiles to something close to hand-written AVX intrinsics: accumulate a running vector sum across the loop (one SIMD add per 8 elements), then, at the very end, sum the 8 individual lanes of that final vector together sequentially — a small, one-time cost compared to doing it this way for every element.

---

## 3. ISPC `task`: Getting Beyond a Single Core

Everything above — the "gang" — is implemented purely with SIMD instructions inside **one thread on one core**. It never uses more than one of a machine's cores on its own. ISPC's separate `task` construct exists specifically to spread work across multiple cores: creating many tasks (e.g., `launch[100] my_ispc_task(...)`) is conceptually the multi-core analogue of `foreach` — the programmer declares a large amount of independent work, and the ISPC runtime, invisibly, maintains a small pool of worker threads (sized to the machine, not to the number of tasks) and hands each one "the next available task" as it finishes its current one.

### Live demo: why a fixed-size thread pool wins
A quick benchmark compares three ways of running many tiny, essentially-free "do nothing" tasks:
1. **Sequential** — just call the task function directly, over and over.
2. **Thread-per-task** — spawn a brand-new OS thread for every single task, then join it.
3. **Fixed thread pool** — spawn exactly as many worker threads as the machine has hardware execution contexts (e.g., 8), and have each one repeatedly grab "the next task" from a shared counter.

Measured result: the sequential version was roughly **23x faster** than the fixed thread pool, and the fixed thread pool was itself roughly **300x faster** than spawning a thread per task. With work this tiny, the overhead of thread creation/teardown (and, to a lesser extent, of coordinating who grabs the next task) completely swamps the actual work being done. As task size grows, this overhead becomes proportionally less significant, and a real thread pool would eventually pull ahead of the sequential version — the crossover point is something worth reasoning about (or measuring) for any given workload.

**A clarifying distinction reinforced here**: this is about **operating-system thread creation/scheduling overhead** (hundreds of thousands of cycles for an OS context switch), which is a completely different order of magnitude from the **hardware multi-threading** covered in Lectures 2–3 (switching which resident hardware thread executes next costs on the order of a single cycle). It essentially never makes sense for an application to create more software threads than the machine has hardware execution contexts — doing so just forces the OS to spend expensive context switches shuffling threads on and off the processor, which is a poor way to try to hide something like a few-hundred-cycle memory stall.

---

## 4. A General Framework: Decomposition, Assignment, Orchestration, Mapping

*(Framework as presented in lecture, adapted from Culler, Singh & Gupta.)*

Turning a problem into a running parallel program involves a chain of transformations:

**Problem → (Decomposition) → Subproblems ("tasks") → (Assignment) → Parallel threads ("workers") → (Orchestration) → Communicating parallel program → (Mapping) → Execution on real hardware**

Each of these four responsibilities can be handled by the programmer, by the system (compiler/runtime/hardware), or some mix of both — and a recurring theme of the course is figuring out which is which for a given tool or language.

### Decomposition
Break the problem into pieces of work that *can* run concurrently — generally, create at least enough pieces to keep every execution unit on the target machine busy. **The central challenge is identifying dependencies** (or confirming their absence). In this course, decomposition is almost always the programmer's job — there is no reliable, general-purpose "magic parallelizing compiler" that can look at arbitrary sequential code and correctly extract this on its own. Automatic decomposition remains a genuinely hard research problem: the compiler would need to prove the absence of dependencies, which is especially difficult when a dependency (if any) depends on runtime data rather than being visible statically; research systems have had some real but modest success on simple, regular loop nests, but nothing close to a general solution exists.

### Assignment
Deciding which worker (thread, program instance, vector lane, etc.) does which piece of decomposed work. Goals: good load balance, and low communication cost. Assignment can be done **statically** (fully decided before the program runs) or **dynamically** (decided as the program executes, adapting to real conditions). While decomposition is usually the programmer's job, assignment is very often something languages/runtimes take responsibility for.

Concrete examples spanning this spectrum, all seen already in this course:
- **Manual static assignment**: the hand-written ISPC code that computed `idx = i + programIndex` — the programmer explicitly decided, by construction, exactly which instance handles which array index (interleaved, in that case).
- **System-managed assignment**: `foreach` — the programmer only declares which iterations are independent; ISPC's implementation decides the actual mapping.
- **Manual static assignment, threads**: the `std::thread`-based `sinx` example from Lecture 2, where the programmer explicitly split the array in half (a **blocked** assignment) between the spawned thread and the main thread.
- **System-managed dynamic assignment**: ISPC `task` — the runtime maintains a pool of worker threads, and after finishing its current task, each worker inspects a shared "next task" pointer and claims the next uncompleted task for itself, adapting automatically to any imbalance in how long individual tasks take.

### Orchestration
Once work is assigned, workers still need to actually cooperate correctly and efficiently: structuring communication, adding whatever synchronization is needed to preserve real dependencies, organizing data layout in memory, and scheduling tasks appropriately. Goals here are reducing communication/synchronization cost, preserving data locality, and generally minimizing overhead — and the right choices are often highly dependent on the specifics of the target machine (e.g., if synchronization is expensive on a given system, a programmer might deliberately use it more sparingly, accepting some extra bookkeeping elsewhere to avoid it).

### Mapping to hardware
Finally, the abstract notion of a "worker" (thread, program instance, etc.) has to be placed onto actual hardware execution resources. This, too, can happen at multiple levels:
- **By the operating system** — mapping a software thread to a hardware execution context on a CPU core.
- **By the compiler** — mapping ISPC program instances onto the lanes of a vector instruction.
- **By the hardware itself** — mapping GPU thread blocks onto GPU cores (a topic for a future lecture on CUDA).

Mapping decisions can matter a lot for performance: e.g., placing cooperating threads that share data on the *same* core (maximizing locality, minimizing communication/sync cost) versus, conversely, deliberately placing *unrelated* threads together on a core to better exploit the machine — for instance, pairing a memory-bandwidth-limited thread with a compute-limited one on the same core so they draw on different bottleneck resources simultaneously rather than competing for the same one.

---

## 5. Amdahl's Law: Dependencies Cap Your Maximum Speedup

Let **S** be the fraction of a sequential program's execution that is inherently serial — i.e., cannot be parallelized no matter how many processors are available, typically because of real dependencies. Then, however good your parallel hardware or however many processors you throw at the rest of the program, your **maximum possible speedup is bounded by 1/S** — no more, regardless of processor count.

### Worked example: brightness + average of an N×N image
Two sequential steps, each taking roughly N² time (so the whole thing takes ~2N² sequentially):
1. Multiply every pixel's brightness by 2 (fully independent per pixel).
2. Compute the average of all pixel values.

**First attempt**: parallelize only step 1 across P processors (time becomes N²/P), leave step 2 fully sequential (still N²). As N grows large relative to P, total time approaches N² (from step 1, now negligible) + N² (step 2, unchanged) → speedup approaches, at best, **2x — no matter how many processors are used**, because half the total original work remains stubbornly sequential.

**Better attempt**: also parallelize step 2, by having each processor compute a *partial* sum over its own share of the data (N²/P time), then combine the P partial sums together at the end (an additional cost of roughly P, for a naive sequential combination step). Total time becomes roughly N²/P + N²/P + P; as N grows much larger than P, this approaches **2·N²/P**, meaning speedup approaches **P itself** — essentially full, near-linear scaling, in stark contrast to the capped-at-2x result above. The only real remaining cost is the overhead of combining those P partial sums at the end.

### The general shape of the curve
Plotting maximum theoretical speedup against number of processors for a few different values of S makes the effect vivid: on a 64-processor machine, if just **1%** of a program is inherently sequential, the best achievable speedup is only around **40x** (not 64x); if **10%** is sequential, the ceiling drops to only around **8x** — a fairly small serial fraction has an outsized effect on achievable speedup, even on a modest number of cores.

### Why this matters even more at extreme scale
Consider a machine on the scale of a real modern supercomputer — on the order of tens of thousands of GPUs, each with thousands of ALUs, totaling on the order of **~150 million** independently parallel arithmetic units. If just **0.1%** of an application's execution is inherently sequential, the maximum possible speedup is capped at **1/0.001 = 1000x** — a spectacular-sounding number in isolation, but a vanishingly tiny fraction of the machine's ~150-million-way theoretical parallelism. **The takeaway: at extreme scale, driving the serial fraction of a program down to an almost absurdly small number becomes essential — otherwise the overwhelming majority of an enormous machine simply goes to waste.** Fortunately, most real workloads people run on such machines are, in practice, highly parallelizable — but Amdahl's Law is exactly why that property has to be actively engineered for, not assumed.

---

## 6. Case Study: A 2D Grid Solver

### The problem
A classic numerical computing pattern: solving a partial differential equation on an (N+2)×(N+2) grid using an iterative method (Gauss-Seidel-style sweeps), where each cell's new value is a weighted combination of its current value and its four immediate neighbors, repeated until the total change across the grid falls below some convergence threshold.

### Step 1: find the dependencies (as originally written)
Written in the most natural, straightforward order (row by row, left to right within each row), each cell's new value depends on its neighbor to the left and the neighbor directly above it — both of which, within the *same* sweep, have themselves *already* been updated earlier in that same pass. This creates a genuine sequential dependency chain running through the entire grid.

**A first (real, but awkward) source of parallelism**: cells lying along the same anti-diagonal of the grid are actually mutually independent, since none of them depend on each other, only on cells from earlier diagonals. In principle, you could process the grid diagonal-by-diagonal, updating everything on a diagonal in parallel and synchronizing between diagonals. In practice, this is a poor strategy: there's very little parallelism available at the very beginning and very end of each sweep (the corner diagonals are tiny), and it requires frequent synchronization — once per diagonal — which adds up to a lot of overhead relative to how little independent work is available on any single diagonal.

### Changing the algorithm itself
Rather than force awkward parallelism onto this particular ordering, the lecture takes a different approach entirely: **swap in a different iterative algorithm that reaches an equivalent (converged, within-tolerance) answer, but does so via an update order that's naturally far more parallel.** This specific move — recognizing that a numerically different-but-acceptable variant of an algorithm might be dramatically easier to parallelize — requires domain-specific knowledge (here, of the Gauss-Seidel method) to know that this substitution is mathematically valid; it's presented as a common and important technique in parallel programming generally, not a one-off trick.

The substitute here: **red-black (checkerboard) ordering.** Color the grid like a checkerboard; on one phase, update every "red" cell in parallel using only "black" neighbor values; on the next phase, update every "black" cell in parallel using the just-updated "red" values; repeat until convergence. This genuinely may take somewhat more iterations to converge than the original ordering (the floating-point trajectory to the answer is different), but each phase now has enormous, clean, uniform parallelism with just one synchronization point between the two phases per iteration — a far better trade for parallel hardware.

### Assignment: it depends on the machine
Given the red-black reformulation, there's still a choice of how to divide grid cells among processors — e.g., contiguous **blocks** of rows per processor, versus an **interleaved** row assignment. Considering the communication this implies: after each processor updates its red cells, it needs the newly-updated red values from neighboring rows before it can correctly update its own black cells. With a **blocked** assignment, a processor's row-block is contiguous, so it only ever needs to exchange boundary information with the (at most two) processors owning the adjacent blocks — comparatively little data movement. With an **interleaved** assignment, a processor's rows are scattered throughout the grid, so nearly every row it owns has neighbors owned by *different* processors — resulting in substantially more total communication.

**This is a deliberately pointed contrast with the earlier ISPC SIMD example**, where interleaved assignment was the *better* choice — because that decision was about efficient memory access from a single vector unit on one core, not about minimizing communication between separate processors. **There is no universally "correct" assignment strategy — the right choice depends on what you're actually optimizing for on the specific system you're targeting.**

### Two ways to express this solver

**1. Data-parallel style** (`forall`, implicit orchestration):
```
while (!done) {
    diff = 0.f;
    for_all (red cells (i,j)) {
        prev = A[i,j];
        A[i,j] = 0.2f * (A[i-1,j] + A[i,j-1] + A[i,j] + A[i+1,j] + A[i,j+1]);
        reduceAdd(diff, abs(A[i,j] - prev));
    }
    if (diff / (n*n) < TOLERANCE) done = true;
}
```
Here, decomposition (individual grid cells are independent work) is explicit in the code, but assignment and orchestration are both left entirely to the system: the end of a `for_all` block implicitly acts as a barrier (nothing after it runs until every worker has finished the block), and `reduceAdd` is a built-in communication primitive handling the cross-worker accumulation safely, without the programmer writing any locking code themselves.

**2. Shared address space (SPMD threads) style**, where the programmer takes on orchestration explicitly:
```
while (!done) {
    myDiff = 0.f;
    diff = 0.f;
    barrier(myBarrier, NUM_PROCESSORS);
    for (j = myMin to myMax) {
        for (i = red cells in this row) {
            prev = A[i,j];
            A[i,j] = 0.2f * (A[i-1,j] + A[i,j-1] + A[i,j] + A[i+1,j] + A[i,j+1]);
            myDiff += abs(A[i,j] - prev);
        }
    }
    lock(myLock);
    diff += myDiff;
    unlock(myLock);
    barrier(myBarrier, NUM_PROCESSORS);
    if (diff / (n*n) < TOLERANCE) done = true;
    barrier(myBarrier, NUM_PROCESSORS);
}
```
Every thread runs this exact same code (SPMD: single program, multiple data), using its own thread ID to compute which contiguous range of rows (`myMin` to `myMax`) it's responsible for — i.e., a manually-coded blocked assignment.

### Why the shared address space model needs explicit synchronization
In this model, threads communicate purely by reading and writing ordinary shared variables in one common address space (a useful mental image: a shared bulletin board everyone can read from and write to). This is powerful but dangerous, because something as simple as `x++` isn't actually one atomic step — it decomposes into **load x into a register, add, store the result back** — three separate operations. If two threads interleave these three steps badly (e.g., both load the same old value before either has stored their update), one thread's update can be silently lost. This is a classic **race condition**, and it's exactly why the grid solver's shared `diff` accumulator needs a **lock**: the lock enforces mutual exclusion, guaranteeing only one thread can be in the middle of updating the shared value at any given time.

### A real performance bug, and its fix
A first-draft version of this code might acquire the lock and update the shared `diff` variable directly, inside the innermost loop, once per grid cell — meaning every single one of potentially millions of cell updates pays for a lock acquisition. The fix mirrors the same idea used by ISPC's `reduce_add`: have each thread accumulate its *own*, private `myDiff` with zero locking throughout the entire inner loop, and only acquire the lock **once per thread, per iteration** — right at the end, to fold that one private partial sum into the shared global `diff`. This removes essentially all of the lock contention while computing an identical final answer.

### Barriers: coarse, phase-based synchronization
A **barrier** is a synchronization point where no participating thread is allowed to proceed past it until *every* thread has reached it — effectively declaring that everything before the barrier, across all threads, must complete before anything after it, in any thread, is allowed to begin. It's a conservative but simple way to express "these two phases of computation have a dependency on each other."

**Why does the solver above need three separate barriers**, when intuitively it might seem like only one (at the very end of each iteration) should be necessary? Walking through what would go wrong if any one were removed:
- **The barrier right before the convergence check** exists to guarantee that every thread's contribution has actually been folded into the shared `diff` *before* any thread reads it to decide whether to stop — without it, a thread could check `diff` before others have finished adding their own partial sums in, and reach the wrong conclusion about convergence.
- **The barrier at the very end of the loop** (before looping back to the top) exists to prevent a "fast" thread from racing ahead into the *next* iteration and resetting the shared `diff` back to zero before every other thread has actually finished *reading* the current iteration's value to make its own convergence decision — without it, the check could silently see an already-cleared value instead of the real one.
- **The barrier at the top of the loop**, by the same logic reflected onto the start of an iteration, prevents any thread from prematurely clearing state for a new iteration before every other thread is actually ready to begin it.

### An exercise: can this be done with just one barrier?
The lecture poses this as a self-check challenge, with a hint pointing toward the same technique used earlier to eliminate lock contention: instead of reusing a single, shared `diff` variable across successive loop iterations (which is what forces multiple barriers to exist, purely to protect that one variable's lifecycle across iteration boundaries), **trade a bit of extra memory for removing the dependency entirely** — maintain a small rotating set of separate `diff` variables (e.g., three of them, indexed by iteration number mod 3), so that "this iteration's diff" and "the next iteration's diff" are never actually the same memory location. The official course materials confirm this works: with each iteration writing to a distinct slot in a small rotating buffer, a thread starting the next iteration's work no longer has any reason to wait on other threads finishing their read of the *previous* iteration's value, since they're no longer touching the same variable — collapsing the three barriers down to just one, needed only to make sure all threads have contributed their update before the convergence check for the current iteration is read.

### Comparing the two models directly
| | Data-parallel (`forall`) | Shared address space (SPMD) |
|---|---|---|
| **Synchronization** | Implicit barrier at the end of each `forall`/`for_all` block | Explicit, programmer-placed barriers marking dependencies between phases |
| **Mutual exclusion** | Handled by built-in primitives (e.g., `reduceAdd`) | Programmer must explicitly use locks around shared-variable updates |
| **Communication** | Implicit in loads/stores, plus special collective primitives for more complex patterns | Implicit in ordinary loads/stores to shared variables |

---

## 7. Summary

- **Amdahl's Law**: the fraction of a program that's inherently sequential puts a hard ceiling on achievable speedup, regardless of how many processors are thrown at the parallel portion — and this ceiling becomes brutally restrictive at very large processor counts.
- Building a parallel program is usefully broken into **decomposition** (find independent work — almost always the programmer's job), **assignment** (map work to workers — sometimes manual, often delegated to the system), **orchestration** (coordinate workers correctly and efficiently — communication, synchronization, data layout), and **mapping** (place workers onto real hardware — handled by the OS, compiler, or hardware itself, depending on context).
- Today's focus was specifically **identifying dependencies** — real ones (which force serialization or careful synchronization) and *apparent* ones that domain knowledge can sometimes restructure away, as in the red-black grid-solver rewrite. Upcoming lectures shift focus toward locality and reducing synchronization overhead.

---

*Notes synthesized and paraphrased from the CS149 Fall 2023 Lecture 4 slide deck and lecture transcript, for study purposes — not a verbatim transcript.*
