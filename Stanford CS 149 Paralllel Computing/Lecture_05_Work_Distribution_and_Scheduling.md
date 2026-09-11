# CS149 — Lecture 5: Performance Optimization Part 1: Work Distribution and Scheduling

**Course:** Stanford CS149, Parallel Computing (Fall 2023)
**Instructors:** Prof. Kayvon Fatahalian & Prof. Kunle Olukotun
**Video:** https://www.youtube.com/watch?v=mmO2Ri_dJkk
**Course site:** https://gfxcourses.stanford.edu/cs149/fall23/lecture/perfopt1/

*Notes combine the official slide deck with the lecture transcript (live Q&A, worked examples).*

---

## 1. Framing for This Lecture

Two parts: first, wrapping up last lecture's unfinished puzzle (getting the grid solver down to a single barrier); then a full lecture on **work distribution and scheduling** — how to keep every worker busy while spending as little overhead as possible making that happen. This week splits cleanly: today is about balancing workload and synchronizing workers (largely ignoring memory); Thursday's lecture shifts to communication cost and memory access efficiency.

### A standing piece of advice
Before diving into increasingly sophisticated scheduling techniques, an explicit warning: **always implement the simplest possible working, parallelized solution first, and measure its actual performance before reaching for anything more advanced.** Every year, a meaningful fraction of students read ahead into the more sophisticated techniques, spend days designing an elaborate scheme on a whiteboard, and — after finally coding it up close to a deadline — find it performs *worse* than the simple approach would have. Let measurements, not anticipated cleverness, decide when (and whether) more sophisticated techniques are actually worth it.

---

## 2. Closing the Loop: One Barrier for the Grid Solver

Recall the shared-address-space grid solver from Lecture 4, which used three barriers per iteration, all protecting a single shared `diff` variable that was being reused across iterations. The reasoning that motivated each barrier: threads couldn't safely reset, accumulate into, or check `diff` until every other thread was at a compatible point in the *same* iteration — because they were all reading and writing that one, shared piece of state.

**The fix**: the dependency wasn't fundamental to the algorithm — it was an artifact of reusing a single variable for a purpose that actually differs, conceptually, from one iteration to the next. Replacing the single `diff` with a small set of separate copies (in practice, just **three** are needed — one for the iteration just finishing, one for the current iteration, and one for the next — cycling through them, e.g., via an index modulo 3) removes the false dependency entirely. Threads moving on to a new iteration are now reading and writing a *physically different* variable than threads still finishing up the previous one, so there's no longer any reason for them to wait on each other outside of the one barrier still needed to make sure every thread's contribution has landed before anyone checks convergence.

This is presented as the same underlying trick used earlier to eliminate lock contention in that same code (replacing one globally-shared accumulator with private per-thread partial sums, combined only once at the end): **trade a small amount of extra memory for removing an unnecessary dependency.**

---

## 3. Programming for High Performance: The Big Picture

Optimizing a parallel program's performance is fundamentally an **iterative process** of refining decomposition, assignment, and orchestration choices — and the goals involved are often directly in tension with one another:
- **Balance workload** across all available execution resources.
- **Reduce communication** (to avoid stalls waiting on other workers or on memory).
- **Reduce overhead** — the extra work spent on parallelization mechanics themselves (scheduling logic, synchronization, assignment bookkeeping).

The rest of this lecture works through a progression of techniques for the first and third of these goals in particular.

---

## 4. Balancing the Workload

**The ideal**: every processor computing continuously throughout execution, and all of them finishing their share of the work at the same moment. Even a fairly small amount of imbalance can meaningfully cap achievable speedup — e.g., if one of four processors is given twice as much work as the other three, that processor alone determines the total runtime, and effectively **half the parallel program's execution time ends up serialized** (running on just one of the four processors while the other three sit idle, having already finished). In Amdahl's Law terms, even though only about a fifth of the total work is "the extra work" causing the imbalance here, its structural effect on runtime is as if a full 20% of the program were inherently serial.

### Static assignment
Assignment that doesn't depend on the program's *dynamic runtime behavior* — though "static" here doesn't necessarily mean fixed at compile time; the assignment can still depend on runtime parameters known in advance, like input size or the number of available threads, as long as it doesn't change based on how execution actually unfolds.

Recall the earlier fractal/image-rendering assignment example from Assignment 1: an interleaved row assignment (each thread handles rows spread throughout the image, rather than one contiguous block) turned out to work well specifically because nearby rows of the image tend to have similar computational cost — the image has enough local *coherence* in cost that spreading rows evenly across threads naturally balances the total workload each thread ends up with, on average, even without knowing the exact cost of any individual row up front.

**Static assignment is a good fit when:**
- All units of work have the exact same, known cost (simplest case — just divide evenly).
- Costs vary but are *known* in advance (assign work to balance total known cost per worker).
- Costs vary *unpredictably* per individual item, but are predictable *on average* — as in the fractal example, where you can't know any one row's exact cost, but you can trust that a large, well-spread sample of rows per worker will average out fairly evenly.

**Its big advantages**: near-zero runtime overhead (the only "cost" is some simple indexing arithmetic), and — critically — **no synchronization needed between workers at all**, since each one already knows exactly what it's responsible for before execution even begins.

### Semi-static assignment
For situations where cost is predictable only over the *near-term* future (recent history is a decent predictor of what's coming next, but not of the whole program's lifetime): periodically re-profile the running application and re-adjust the assignment, treating it as "static" only for the interval between adjustments. Examples: an adaptive simulation mesh around an aircraft wing, where the mesh's shape (and thus the per-region workload) shifts gradually as airflow patterns change during a simulation; a particle simulation, where particles drift slowly enough that an assignment made now stays reasonably balanced for a while before needing to be redone; or a long-running machine learning training job, periodically checked for workload imbalance and rebalanced between longer stretches of steady execution.

### Dynamic assignment
Used when the cost or even the total *number* of tasks is genuinely unknown or unpredictable ahead of time. A representative example: testing primality of a large array of numbers, where any individual test's cost is hard to predict in advance. Converting a plain sequential loop over this into a parallel SPMD version: every thread runs the same loop, but instead of each being handed a fixed range up front, they all share a single counter (protected by a lock); each thread grabs the lock, reads and increments the counter to claim "the next index nobody has done yet," releases the lock, and does that unit of work — repeating until the counter exceeds the array size.

Conceptually, this *is* a shared work queue, even though there's no explicit queue data structure in sight — the array itself, combined with a monotonically increasing counter, functions as an efficiently-implemented queue, and claiming "the next item" reduces to a single atomic increment. More generally, dynamic assignment schemes have worker threads **pull** work from a shared queue and (in more advanced systems) **push** newly discovered work onto it as they go.

---

## 5. What Actually Constitutes "a Piece of Work"? Granularity Trade-offs

### Fine-grained tasks: good balance, but sync overhead
Making each unit of dynamically-assigned work as small as possible (e.g., one array element = one task, as in the primality example above) tends to give excellent workload balance — there's a lot of flexibility in how the many small pieces can end up distributed. But it also means paying the cost of the lock/counter synchronization **once per element** — and time spent inside that critical section is, in effect, serialized execution that doesn't exist at all in the original sequential program (a direct instance of Amdahl's Law biting from an overhead source, not an inherent algorithmic one).

**A practical debugging methodology, worked through live**: instrument the program with timers — one around the whole program's execution, and another wrapped tightly around just the actual useful work being done (e.g., every call to the primality test itself). Comparing the two numbers tells you where time is actually going:
- If the "useful work" timer accounts for nearly all of the total time, there's little left to gain — further optimization effort likely isn't worth it (unless the remaining sliver genuinely matters, e.g., in an extremely latency-sensitive application).
- If a large fraction of total time is *not* accounted for by useful work, that overhead is worth chasing down — and in a program this simple, it's usually straightforward to guess where it's hiding (here: the lock).

A first, simple fallback if synchronization overhead turns out to dominate: consider whether a static assignment could be used instead, sidestepping the need for any per-task synchronization at all.

### Coarser-grained tasks: less overhead, same idea
Alternatively, without abandoning dynamic assignment altogether, simply increase how much work each queue "pop" claims at once — e.g., increment the shared counter by 10 instead of 1, and have each thread process a contiguous batch of 10 elements per lock acquisition instead of just 1. This proportionally reduces how often the lock is touched (10x fewer critical-section entries, in this example), at the cost of somewhat coarser (though usually still perfectly fine) workload granularity.

### Choosing task granularity
Two competing pressures:
- Want **many more tasks than processors**, so dynamic assignment has enough flexibility to actually achieve good balance — this favors small tasks.
- Want **as few tasks as possible**, to minimize the overhead of managing task assignment (locking, bookkeeping) — this favors large tasks.

There's no universal right answer — the ideal granularity depends on the specific workload and the specific machine, reinforcing a theme that recurs throughout this course: **good performance work requires actually knowing your workload and your target hardware**, not applying a fixed recipe.

---

## 6. Smarter Task Scheduling: Order Can Matter Even With Dynamic Assignment

Consider 16 dynamically-scheduled tasks of varying size, assigned to workers in simple left-to-right order as they become available. If, purely by chance, one especially long task happens to be scheduled *last*, the worker that ends up claiming it will keep working long after every other worker has run dry — a **long tail** that leaves most of the machine idle while one straggler finishes.

**Two ways to address this:**
1. **Break work into smaller pieces**, hoping this shortens the "long pole" relative to total execution time — though this may not always be possible (if the long task is fundamentally, irreducibly sequential, it simply can't be subdivided further), and it does add proportionally more synchronization overhead.
2. **Schedule long tasks first, if their relative cost is known or predictable in advance** — deliberately handing out the biggest tasks earliest, so a worker unlucky enough to draw one still ends up finishing at roughly the same time as everyone else, having simply completed fewer total tasks but the same total amount of work. This requires some upfront knowledge of relative task cost, but needs no change to how tasks themselves are structured.

---

## 7. Reducing Synchronization Cost: Per-Thread (Distributed) Work Queues

A single, shared work queue means *every* worker synchronizes against the same lock every time it needs more work — which becomes a real bottleneck as the number of workers (and the frequency of queue access) grows, especially with small task granularity. The fix: give each worker its **own local queue**. Workers pull from (and push newly-discovered work onto) their own queue by default, with no synchronization needed against anyone else, and only fall back to looking at — and **stealing** from — another worker's queue once their own runs completely empty. (The mechanics and surprising subtlety of exactly *how* this stealing should work are the focus of the rest of this lecture, via a deep dive into a real system called Cilk.)

### Task systems can also have dependencies
Everything covered so far assumed pieces of work were fully independent and could run in any order. Real task-scheduling systems often need to support genuine **dependencies** between tasks — e.g., "run task bar, but only after task foo has completed." A task management system can accept these dependency declarations alongside newly submitted work, and hold a task back from being assigned to any worker until every task it depends on has finished (mirroring exactly the kind of task-graph scheduling problem that constitutes the final part of Assignment 2 in this course).

---

## 8. Common Parallel Programming Patterns (Setting Up Fork-Join)

Everything covered in this course so far falls into one of two broad patterns:

**1. Data parallelism** — apply the same operation across many independent data elements. This is the pattern behind nearly every example seen so far (ISPC's `foreach`, ISPC bulk `task` launches, an OpenMP `#pragma parallel for`, a functional `map`, and — previewing a future lecture — bulk CUDA kernel launches on a GPU) — all fundamentally saying "here is a large, flat collection of independent work; go do it."

**2. Explicit thread creation** — directly spawn exactly as many threads as desired units of concurrency (e.g., plain `std::thread` in C++), with the programmer responsible for deciding exactly what each thread does.

There's a **third** important pattern, poorly served by either of the above: **divide-and-conquer / recursive algorithms**, exemplified by quicksort. Quicksort's parallelism isn't handed to you upfront as one big flat collection — it's *progressively revealed*, level by level, as the recursion unfolds: partitioning the array is (for now) treated as an inherently sequential black box, but the two resulting recursive calls (sort the left half, sort the right half) are mutually independent and can run in parallel — and each of *those* calls, in turn, spawns two more independent recursive calls, and so on, down to some small base case. This produces a rapidly-branching *tree* of potential parallelism, rather than a flat list — and needs its own dedicated programming pattern to express well.

---

## 9. The Fork-Join Pattern: Introducing Cilk

**Fork-join** is the natural way to express the independent work inherent to divide-and-conquer algorithms. This lecture uses **Cilk Plus** (a C++ extension, originally from MIT, now supported in mainstream compilers like GCC and Intel's ICC) as a concrete example — not because the specific syntax matters much, but because it makes the underlying scheduling ideas unusually clear.

### The core primitives
- **`cilk_spawn foo(args)`**: invoke `foo` — but, unlike an ordinary function call, the calling code is now free to **continue executing asynchronously**, without waiting for `foo` to return.
- **`cilk_sync`**: block until every call spawned by the *current* function has completed. There is also an **implicit `cilk_sync` at the end of every function that contains a `cilk_spawn`** — meaning that once a Cilk function returns to its own caller, all work it ever spawned is guaranteed to be fully finished.

It's worth being precise: **`spawn` is not the same thing as creating a thread.** Spawning declares "here is a logically independent, asynchronous unit of work that needs to run at some point" — it says nothing about whether or when it's actually assigned to any particular thread of execution. That's entirely a scheduling/implementation decision, made later.

### Abstraction vs. implementation, revisited
`cilk_spawn` deliberately says nothing about *how* or *when* spawned calls actually get scheduled — only that they *may* run concurrently with the caller (and with each other). This means a valid — if maximally unhelpful — implementation of Cilk could simply strip every `cilk_spawn`/`cilk_sync` keyword out of the source and compile the result as ordinary sequential C++; the program would still be correct, just entirely unparallelized. An equally valid, but very differently *performing*, implementation could instead spawn a genuine OS thread for every single `cilk_spawn` and join on it at every `cilk_sync`. Both satisfy the abstraction's actual guarantees.

### A few basic examples
```c
cilk_spawn foo();
bar();
cilk_sync;
```
`foo()` and `bar()` may run concurrently — `bar()` runs directly on the calling thread, `foo()` is available to run elsewhere; `cilk_sync` waits for `foo()` to finish before continuing.

```c
cilk_spawn foo();
cilk_spawn bar();
cilk_sync;
```
Same net independent work as above, but expressed with two spawns instead of one, at the cost of somewhat more runtime bookkeeping overhead for no additional parallelism gained.

```c
cilk_spawn foo();
cilk_spawn bar();
cilk_spawn fizz();
buzz();
cilk_sync;
```
Four logically-independent pieces of work available concurrently — `foo`, `bar`, `fizz`, and `buzz` (running directly on the caller). The only real guarantee: all three spawned calls must have completed by the time `cilk_sync` returns; nothing else about their relative scheduling order is promised or should be relied upon.

### Quicksort, expressed in Cilk
```c
void quick_sort(int* begin, int* end) {
    if (begin >= end - PARALLEL_CUTOFF)
        std::sort(begin, end);   // sequential base case
    else {
        int* middle = partition(begin, end);
        cilk_spawn quick_sort(begin, middle);
        quick_sort(middle + 1, end);   // runs on the current thread
        // implicit cilk_sync here, since this function contains a spawn
    }
}
```
Below some small array-size cutoff, fall back to a plain sequential sort — for small enough inputs, the overhead of spawning would outweigh any benefit from parallelizing further. Above that cutoff, one recursive half is spawned off; the other runs directly; the implicit sync at function return guarantees both halves are fully sorted before this call itself is considered done — which is exactly what's needed for the *caller's* own subsequent step (if any) to proceed correctly.

### "Parallel slack": how much work should you actually spawn?
General rules of thumb for writing fork-join code well:
- Spawn **at least** as much independent work as the machine has parallel execution capacity, or there's nothing for extra cores to do.
- Spawn genuinely **more** independent work than that capacity, to give a scheduler real flexibility to achieve good load balance — this ratio of "available independent work" to "machine's parallel capacity" is called **parallel slack**; in practice, a slack of roughly **8x** tends to work well.
- But don't create *far* more independent tasks than needed — pushed too far, the per-task management overhead of tracking enormous numbers of very fine-grained spawns starts to dominate.

---

## 10. How Cilk Actually Schedules This: Work Stealing

### Why the "naive" implementation is a bad idea
Literally spawning a real OS thread (e.g., via `pthread_create`) at every `cilk_spawn`, and joining at every `cilk_sync`, would be correct but slow — for the same reasons demonstrated by the earlier thread-pool timing demo from Lecture 4: heavyweight thread creation/teardown cost, far more concurrently-live OS threads than the machine actually has execution contexts for (forcing expensive OS-level context switching), and a larger, less cache-friendly working set than necessary.

### The real approach: a fixed worker-thread pool
Cilk's actual runtime maintains a **pool of worker threads, exactly as many as the machine's hardware execution contexts** (e.g., 8 workers on an 8-context machine) — conceptually, all created once, up front, at program launch (real implementations are often lazier about this in practice, only spinning them up on the first spawn, but the mental model is the same). Each worker just loops: "while there's still work anywhere in the system, get the next piece and run it."

### The scheduling choice at every spawn point
At `cilk_spawn foo(); bar();`, there are two conceptually distinct pieces of work available: the **spawned child** (`foo()`) and the **continuation** (everything in the calling function *after* the spawn point, here `bar()` and beyond). The executing thread has to pick one of these two to run right now, and defer the other one for possibly-later, possibly-remote execution.

- **Run continuation first ("child stealing")**: the thread immediately moves on to `bar()`, leaving `foo()` behind as the thing available for another idle thread to come along and steal.
- **Run child first ("continuation stealing")**: the thread immediately starts running `foo()` itself (exactly like a normal, un-parallelized function call would), leaving the *continuation* — the rest of the calling function — behind as the thing available for stealing.

### Why Cilk specifically chooses "run child first"
The trade-off becomes clear with a simple loop:
```c
for (int i = 0; i < N; i++) {
    cilk_spawn foo(i);
}
cilk_sync;
```
- **If continuations are run first (child stealing)**: the calling thread races straight through the entire loop, spawning — and immediately setting aside for stealing — *every single* `foo(i)` call before doing any real work itself. This means up to **O(N) items** end up sitting in a queue at once (effectively a breadth-first sweep across the whole call graph), and — if no stealing happens to occur — the resulting execution order bears little resemblance to what the equivalent sequential program (with spawn/sync simply deleted) would have done.
- **If children are run first (continuation stealing)**: the thread immediately starts executing `foo(0)`, having only enqueued **one** thing: a continuation representing "the rest of this loop, starting from `i = 1`." If nothing steals that continuation, the thread will eventually pop it back off, bump `i`, run `foo(1)`, re-enqueue an updated continuation for `i = 2`, and so on — meaning, in the no-stealing case, execution proceeds in **exactly the same order** as the sequential program would have, using only a small, roughly constant amount of extra queue storage at any moment (a depth-first-style traversal of the underlying call graph). It can further be proven that, across a system with T worker threads, total work-queue storage never exceeds roughly T times what a single sequential thread's own call stack would have used for the same computation.

This is why Cilk's real scheduler always runs the spawned **child** first, immediately, on the calling thread — deferring the **continuation** as the thing available for other threads to steal.

### Watching work stealing unfold: quicksort on 200 elements
As the recursion proceeds (quicksort 0–200 → spawn quicksort of one half, continue on the other, recurse again, and again), the actively-running thread's local queue fills up with a series of continuations of *decreasing* size as recursion goes deeper — the biggest, least-subdivided remaining chunk of work sits at one end of the queue (having been pushed there earliest, before any further recursion happened), while the smallest, most-recently-generated pieces sit at the other end, right next to whatever the thread is actively working on right now.

**When another thread goes idle and needs to steal, which end should it take from?** Stealing the *biggest* available chunk is clearly better, for two compounding reasons: it minimizes how often that thread will need to go steal again in the near future (one big steal buys a lot of runway), and — since that big chunk will itself recursively decompose into further sub-work once the stealing thread starts working on it — it also sets the stealing thread up with good future locality of its own, rather than handing it one tiny scrap that gets exhausted almost immediately.

### Implementation: a deque (double-ended queue) per worker
Each worker's local work queue is implemented as a **deque**:
- The **owning thread** pushes and pops from the **tail** ("bottom") — the same end where its actively shrinking, currently-relevant work lives, requiring no coordination with anyone else in the common case.
- **Remote, idle threads** steal from the **head** ("top") — where the largest, oldest, most-recursion-rich pieces of work sit.

This deliberately keeps the owning thread and any potential thieves operating on physically opposite ends of the same structure, which (combined with the "steal the biggest piece" policy above) minimizes both how often stealing needs to happen and how much the two parties ever actually contend with each other — efficient, largely lock-free deque implementations exist specifically to exploit this access pattern.

### Choosing a victim to steal from
When idle, a thread simply picks another thread's queue **at random** to attempt a steal from. This might sound naive compared to, say, always targeting whichever thread appears to have the most outstanding work — but it turns out random victim selection is **provably within a constant factor of the theoretically optimal schedule**. More "clever" heuristics mostly only improve that constant factor, and can actually backfire: if many idle threads all simultaneously decide the same "obviously busiest" thread is the best target, they immediately create contention over that one queue, and by the time some of them get there, much of what made it attractive may already be gone.

### Recursive spawning fills the machine faster than a flat loop
A subtle but important point: a `recursive_for` helper that recursively **halves** its iteration range (spawning a call on one half, recursing on the other) reveals independent, stealable work **much faster** than a flat loop that spawns every iteration one at a time — because, with continuation-stealing, the flat loop only ever exposes *one* stealable continuation at a time, growing the pool of available work at a linear, one-item-per-iteration rate, while the recursive-halving version generates a rapidly branching tree of independent chunks almost immediately. This is exactly why Cilk's own scheduler design specifically **anticipates and rewards divide-and-conquer-style recursive parallelism** — and why real Cilk libraries commonly provide a `cilk_for` construct that implements a flat-looking loop internally using this same recursive-halving strategy, rather than a naive one-spawn-per-iteration translation.

### Implementing `cilk_sync`
When no stealing has occurred for a given spawn "block," `cilk_sync` is a complete no-op — the executing thread already ran every piece of that block's spawned work itself, sequentially, in the exact same order a plain sequential program would have.

When stealing *has* occurred, the runtime needs a small amount of bookkeeping to know when a block's work — now potentially scattered across several different threads — has genuinely all finished: a small per-block descriptor tracks a running **spawn count** (how many pieces of work have been spawned from this block so far, which can keep growing as further steals and recursive spawns happen) and a **done count** (how many of those have actually completed). A `cilk_sync` for that block simply waits until the done count catches up to the spawn count.

An interesting consequence: **whichever thread happens to finish the very last outstanding piece of a block's work is the one that gets to continue on with that function's post-sync continuation** — which is not necessarily the same thread that originally started the block in the first place. This is called **greedy join scheduling**: every thread is *always* trying to grab available work rather than ever sitting idle voluntarily, and a thread only truly goes idle once there's genuinely nothing left to steal, anywhere in the whole system.

---

## 11. Cilk Summary

- Fork-join is a natural, elegant way to express divide-and-conquer parallelism — Cilk is one concrete example, but the same basic spawn/sync pattern shows up in other systems too (e.g., OpenMP has similar constructs).
- Cilk's runtime pairs this simple abstraction with a scheduler that is both **locality-aware** and provably close to optimal: always run the spawned child immediately (continuation stealing), always steal the largest available chunk of work from the top of a randomly-chosen victim's deque, and behave greedily at every sync point — threads never wait around when there's stealable work elsewhere in the system.

---

## 12. Overall Summary

- Achieving good workload balance while minimizing overhead is a constant, central tension in parallel programming — every mechanism that helps balance load (locks, dynamic queues, work stealing) itself costs something, and the right trade-off depends on the workload and machine at hand.
- **Static vs. dynamic assignment is a continuum, not a binary choice.** The right move is to exploit whatever predictability actually exists about a workload, to minimize how much runtime cost is needed to achieve good balance — in the limiting case of perfect predictability, fully static assignment is both the simplest option and essentially free.
- The next lecture shifts focus away from "keeping every worker busy" and toward the other major axis of performance: **locality, communication cost, and contention** — i.e., how efficiently a program accesses memory and coordinates data movement between workers.

---

*Notes synthesized and paraphrased from the CS149 Fall 2023 Lecture 5 slide deck and lecture transcript, for study purposes — not a verbatim transcript.*
