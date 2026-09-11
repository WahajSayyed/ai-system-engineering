# CS149 — Lecture 8: Data-Parallel Thinking

**Course:** Stanford CS149, Parallel Computing (Fall 2023)
**Instructors:** Prof. Kayvon Fatahalian & Prof. Kunle Olukotun
**Video:** https://www.youtube.com/watch?v=Ba3TqxSgnTk
**Course site:** https://gfxcourses.stanford.edu/cs149/fall23/lecture/dataparallel/

*Notes combine the official slide deck framing with the lecture transcript (live Q&A, worked examples).*

---

## 1. Framing for This Lecture

A deliberate shift in perspective. Every prior lecture in this course framed parallel programming in terms of "workers" — threads, program instances, CUDA threads, and what each one individually does. Today reframes the exact same territory in terms of **operations on whole sequences of data** — expressing an algorithm as a pipeline of a small number of well-known, richly-studied parallel primitives (map, fold/reduce, scan, and others), rather than reasoning explicitly about threads and dependencies at all. This isn't a departure from anything covered so far — it's a different, often more productive, *lens* for the same underlying ideas, and one that shows up constantly in real-world tools (NumPy, PyTorch/TensorFlow, CUDA's Thrust library, Apache Spark).

### Why this matters at the scale modern hardware demands
Recall, from the GPU lecture, that a single V100 chip supports up to roughly 163,000 concurrently resident CUDA threads. Practically, this means: even working entirely within one chip, real performance requires *hundreds of thousands* of independent pieces of work to be available at once — this isn't an exotic, cluster-scale concern; it's the baseline requirement for keeping a single modern GPU busy. Everything in this lecture is motivated by that scale of parallelism being the normal, expected case, not an unusual one.

### The core idea
Step one of parallelizing *any* program, repeated throughout this course, is always: find the dependencies (or, equivalently, find where dependencies are *absent*, since that's exactly where parallelism lives). Today's approach sidesteps having to reason about dependencies by hand, program by program: instead, **express your program as calls to a small set of operations that are already known, by construction, to have well-defined, highly parallel implementations** — and trust that composing calls to those operations together yields an overall program that is, itself, highly parallel. This is exactly the mental model already implicit in writing NumPy or tensor code: allocate large arrays, call operations like `+` on them, and never think about threads or dependencies at all.

### Sequences, not arrays
To generalize this idea beyond any one specific language's array type, the lecture adopts the term **sequence**: an *ordered* collection of elements (distinct from an unordered set) — the same underlying concept behind C++'s sequence containers, Scala's lists, Python's data frames/NumPy arrays, or PyTorch/TensorFlow tensors. The key restriction that makes sequences special: **unlike a raw array, code operating on a sequence can only touch its elements through a small set of specific, well-defined operations — not arbitrary indexed access at any time.** This restriction is deliberate: unconstrained indexed access (`a[i]` from anywhere, at any time) is exactly the mechanism that creates hard-to-reason-about dependencies between loop iterations in the first place; removing that freedom removes the ability to accidentally create such dependencies.

---

## 2. `map`: The Primitive Already Familiar From This Whole Course

**`map`** takes a function `f : A → B` and a sequence of type A, and produces a new sequence of type B by applying `f` independently to every element. Nearly every piece of parallel code written so far in this course has, in essence, been a `map` — a `forall`/`foreach` loop applying the same logic to every array element is precisely this pattern, just not previously given this name.

### Why `map` is trivially parallel
By its very definition, every individual application of `f` only ever touches **one** input element — `f` has no visibility into, and therefore no way to accidentally create a dependency on, any other element of the sequence. The implementor of `f` doesn't even need to think about "the collection" at all — just "given one element, produce one output."

### A simple parallel implementation
Partition the input sequence into P roughly-equal pieces (say, one per available thread), have each thread sequentially apply `f` to its own piece, and concatenate the P partial output sequences back together — no synchronization needed between threads at all during the actual work, since there's no shared state or dependency to protect.

---

## 3. `fold` (a.k.a. `reduce`): Combining a Sequence Into One Value

**`fold`** takes a combining function `f : (A, B) → B`, a starting value of type B, and a sequence of type A, and produces a single value of type B by repeatedly applying `f`, "folding" each successive element into a running accumulated result. A concrete instance: `fold` with the `+` operator over a sequence of integers computes their sum.

### Can `fold` be parallelized? Only sometimes — and it depends on a property of `f`
An immediate, and useful, disagreement surfaces in discussion: this class has *already* computed sums in parallel (e.g., in the earlier human-summation demos and grid-solver reductions) — so isn't `fold` obviously parallelizable? The careful answer: **parallel summation specifically** is fine, but **`fold` for an arbitrary, unknown function `f`** is not safely parallelizable in general. A naive parallel implementation — split the sequence into chunks, `fold` each chunk independently (possibly in a different order than the sequential version would have), then combine the partial results — is only guaranteed to produce the *same* answer as the sequential version if `f` is **associative** (grouping doesn't affect the result). A function like exclusive-or over booleans, or more generally an arbitrary non-associative function, can legitimately produce a *different* answer depending on the order operations happen to be grouped in.

Notably, `f` does **not** need to be *commutative* — a correct parallel `fold` implementation still applies the combining steps in the *same relative order* the sequential version would have (it must, if the goal is to reuse the same associative operator on each subsequence); it just performs that combining using a tree-shaped, rather than strictly linear, application order. Some libraries expose a more general form of `fold` that separately accepts a combining function for merging partial per-worker results (which must map `B × B → B`), giving the implementer flexibility even when the primary `A × B → B` combining function alone doesn't cleanly suggest how partials should be merged — but when a single associative operator (like `+`) can serve both roles, that's the common, simple case.

### An aside: fusing `map` and `fold`
A common pattern — e.g., multiply every element by 10 (a `map`), then sum the results (a `fold`) — can, if a compiler or runtime understands the definitions of both `map` and `fold`, be automatically **fused** into a single pass that multiplies-and-immediately-accumulates each element, avoiding a full extra pass over the data to materialize the intermediate mapped sequence. This is exactly the kind of optimization modern JIT compilers for tensor libraries (e.g., PyTorch's JIT) perform automatically on code that, as *written*, looks like separate composed operations.

---

## 4. `scan`: Producing All the Running Partial Results

**`scan`** (a.k.a. prefix sum, when the operator is `+`) also takes an associative binary operator, but rather than collapsing a sequence down to one final value like `fold`, it produces a **new sequence of the same length**, where each output element is the repeated application of the operator to every input element up to (and, for an **inclusive** scan, including) that position. An **exclusive** scan instead stops just *before* the current element; converting one form to the other is straightforward (an inclusive scan's value at position i is its exclusive scan's value at position i, combined with the original input element at position i).

A plain sequential implementation is a simple one-pass loop, carrying forward a running accumulated value — but that formulation looks stubbornly serial (each output depends directly on the immediately preceding output), which sets up the real question: how, if at all, can this be parallelized?

### First attempts, discussed live
- **"Just compute the total sum in parallel first"** — a reasonable starting point (and something this class already knows how to do efficiently), but it only produces the *final* value, not all the intermediate partials `scan` actually needs to output.
- **Divide and conquer, recursively**: if the scan of the first half of the sequence were already known, its final (total) value could simply be added, in parallel, to every element of an independently-computed scan of the second half — correctly "rebasing" the second half's local results into the correct global values. This intuition is exactly the seed of the real algorithm below.

### The classic work-efficient parallel scan algorithm (Blelloch scan)
A naive divide-and-conquer scan can achieve a **span** (longest chain of sequential dependency, i.e., time on an infinite number of processors) of O(log n) — but at the cost of doing O(n log n) *total* work, asymptotically more than the O(n) a sequential scan requires. A well-known algorithm (commonly attributed to Guy Blelloch) achieves **both** O(log n) span **and** O(n) total work, via two phases:
- **Up-sweep (reduce) phase**: build a combining tree from the leaves up, computing (and retaining) a series of partial sums along the way, ending with the total sum at the root.
- **Down-sweep phase**: starting from the root, propagate the appropriate partial sums back *down* the tree and out to the correct positions — effectively "splatting" each subtree's correct running base value out to where it's needed, until every element holds its correct final scan value.

Each phase takes O(log n) steps, and the total work across all steps of each phase telescopes down to O(n) (the classic geometric-series argument: n + n/2 + n/4 + ... converges to O(n)) — with a modest constant-factor overhead (roughly 2x total work, since there are two full phases instead of one).

**Real-world caveats, raised directly in lecture**: even though this algorithm is asymptotically optimal in both work and span, it isn't automatically the *fastest* choice on real hardware. Two practical downsides: **not every processor stays busy at every step** (as the tree narrows near the root during up-sweep, and widens again during down-sweep, progressively fewer/more processors have anything to do at any given moment), and the memory access pattern **bounces around non-contiguously**, rather than moving smoothly through memory — exactly the kind of poor locality flagged as a real performance concern in the previous lecture on communication and cache behavior. (This algorithm forms the basis of one of the warm-up exercises in this course's CUDA assignment.)

### A simpler, often-better approach with just a couple of threads
Given only two threads (or a small number of cores), a much simpler strategy tends to work just as well in practice: split the sequence in half, have each thread compute a plain sequential scan of its own half (fast, and reads straight through memory with excellent locality), then have one thread take the final (total) value from the first half and add it, in parallel, to every element of the second half's already-computed local scan. On a shared-memory machine, the "communication" of that one total value across threads costs almost nothing — reinforcing, again, the *simplest thing first* principle: the elaborate O(n)-work tree algorithm is genuinely valuable at very large processor counts, but is very likely overkill (and possibly slower in practice, due to its poor locality) on a machine with only a handful of cores.

### An almost paradoxical result on SIMD hardware
Now consider implementing scan for a **32-wide SIMD warp** (as in CUDA) — a setting where "the same instruction, applied to all lanes" is the fundamental unit of execution. A direct, low-level implementation exploiting a warp's known width computes a full 32-element scan in exactly **5 steps** (since log₂32 = 5) — each step doing a SIMD add across all lanes still participating, with a shrinking subset of lanes contributing new work at each step (some lanes simply stop needing to update further as the algorithm proceeds, but the instruction still nominally executes across the full width in lockstep). Total work here is O(n log n) — the "asymptotically worse" naive divide-and-conquer approach from earlier, not the "better" O(n) Blelloch algorithm.

**Here's the surprising part**: implementing the *asymptotically superior* O(n)-work Blelloch algorithm for this same 32-element case takes **10 steps** (5 for up-sweep, 5 for down-sweep) — genuinely *slower* in wall-clock terms than the "worse" O(n log n) approach, despite doing asymptotically less total work! The resolution: on SIMD hardware, an instruction executes across the *entire* warp width regardless of how many lanes are actually doing something useful in that instruction — so an algorithm that leaves *some* lanes idle at certain steps (as Blelloch's algorithm increasingly does, especially near the top of its tree) is, in effect, wasting SIMD width for no benefit, while the "wasteful," fully-lockstep O(n log n) version keeps *every* lane productively busy at *every* one of its steps. **The right choice of scan algorithm genuinely depends on how the target hardware maps parallel work onto physical resources** — a machine with many genuinely independent processors favors the work-efficient O(n) algorithm; a machine that only offers efficiency when *all* lanes of a SIMD unit do the same thing at once favors the "wasteful-looking" O(n log n) version instead.

### Composing warp-scan to build larger scans
A real, practical GPU scan implementation combines both ideas hierarchically, using the fast 32-wide, 5-step warp-scan as a building block: to scan 128 elements, run four independent 32-wide warp-scans (5 steps each, all in parallel across different warps) to get four local results and four partial totals; scan those four totals themselves (trivially, or via another small warp-scan); then, in parallel, add the appropriate rebasing total back into each of the four blocks' local results. Extending this recursively (warp-scan blocks of 32 → combine into groups of 1,024 → combine those into still-larger groups spanning multiple CUDA thread blocks) lets you scan arrays of any size, always bottoming out in the cheap, hardware-appropriate 5-step primitive rather than ever falling back to a naive, poorly-hardware-matched approach — this hierarchical mixing of data-parallel primitives with more conventional sequential composition, chosen deliberately based on how much parallelism is actually available at each level, is presented as a genuine hallmark of real, well-engineered parallel libraries (this course's CUDA assignment includes both implementing the basic O(n) algorithm as a learning exercise, and optionally trying to get close to a real, highly-tuned vendor scan implementation's performance).

---

## 5. `segmented scan`: Scan Over a Sequence of Sequences

Many real problems naturally have **two levels of parallelism** nested inside each other: for every vertex in a graph, its list of edges; for every particle in a simulation, nearby particles; for every document in a collection, its words. If the *outer* level alone doesn't provide enough total parallelism (e.g., a modest-sized graph with only a few thousand vertices, each with a handful of edges, isn't nearly enough independent work to fill a GPU with 163,000-plus available thread slots), the *inner* level's parallelism has to be exploited too.

**Segmented scan** applies an ordinary scan operation independently across each "sub-sequence" of a flattened sequence-of-sequences, all in one combined parallel pass. A common, compact way to represent a sequence of sub-sequences: store all elements flattened into one long array, alongside a parallel array of boolean **flags** marking which positions are the *start* of a new sub-sequence — exactly the encoding needed to compactly represent, for example, a graph's full edge list (a common real representation for sparse graph and matrix data). The actual algorithm adapts the same up-sweep/down-sweep structure as ordinary work-efficient scan, simply modified so that whenever propagation would otherwise cross a sub-sequence boundary (as marked by a start flag), that propagation is deliberately skipped — preserving each sub-sequence's independence while still achieving the same overall O(n) work, O(log n) span characteristics.

### Case study: sparse matrix–vector multiplication
A very common real workload: multiply a sparse matrix (one where the overwhelming majority of entries are exactly zero — extremely common whenever data has sparse structure, e.g., a company's customer-by-product purchase matrix) by a dense vector. Storing a sparse matrix in the common **Compressed Sparse Row (CSR)** format: a flat array of all nonzero *values*, a parallel flat array of each nonzero's *column index*, and a per-row array of *starting offsets* into those flat arrays (equivalent to the "start of sub-sequence" flags described above, but expressed per-row for convenience). This representation is naturally a sequence-of-sequences: the outer sequence is the matrix's rows, and each row's inner sequence is that row's nonzero values.

Expressing the full sparse matrix–vector multiply purely in terms of the primitives covered so far:
1. **`gather`** (introduced properly below) the needed input-vector values, using each nonzero's stored column index, into a new, densely-packed array the same length as the nonzero-values array.
2. **`map`** a multiply operation across the nonzero values and this newly gathered array, producing one product per nonzero.
3. **`segmented scan`** (with `+`) across these products, using the per-row start flags — the *last* element of each row's segment is exactly that row's total dot-product result.
4. Extract those final per-row values out of the segmented-scan result.

The resulting parallelism scales with the total **number of nonzeros** in the matrix, not with the number of rows — a substantial advantage whenever nonzero count vastly exceeds row count, letting this run about as fast as the underlying `map` and `segmented scan` primitives allow, entirely without hand-written thread/dependency logic for the matrix's irregular row lengths.

---

## 6. `gather` and `scatter`: The Data-Movement Primitives

Two operations underlie a lot of the above without having been formally introduced yet:

- **`gather`**: given an index sequence and a source data sequence, produce a new output sequence by, for every position, using the corresponding index value to look up (and copy) the appropriate source element — effectively "densifying" scattered data into a compact new array (exactly what was used to pull the needed vector values in the sparse matrix-vector example above).
- **`scatter`**: the inverse — given a dense sequence of values and a parallel sequence of destination indices, write each value out to its (potentially sparse, scattered) destination location.

### Why these can be expensive
Modern CPUs support gather directly as a genuine SIMD instruction (e.g., AVX2's gather instruction): given a vector register full of indices and a base pointer, it dereferences `base[index]` for every lane and packs the results into a single vector register. This is exactly what's implicitly happening any time ISPC (or CUDA) code writes something like `a[some_index_expression]` where the index depends on `programIndex`/`threadIdx` in a non-trivial way. But because the indices involved are arbitrary and potentially wildly different across lanes, this can be a genuinely costly operation in practice: it's entirely possible for every single lane of one gather instruction to miss on a *different* cache line, or even trigger a *different* page fault — a sharp contrast with an ordinary, contiguous vector load (as used when threads/lanes access strictly adjacent memory addresses), which is comparatively cheap and predictable.

### Turning one into the other
If a scatter operation happens to be a true **permutation** (every destination index is unique, and together they cover every output position exactly once), it can be re-expressed as a **sort**: sorting the data according to its destination indices produces exactly the same result as scattering it there. This is a useful trick on hardware or libraries that provide an efficient gather but no direct scatter primitive.

### Building a "scatter with combine" (e.g., for histogram-style updates) from simpler pieces
A common need — scatter a value to a target location, but *combine* it with whatever's already there (e.g., incrementing a histogram bin) rather than overwriting it — can be built entirely out of primitives already covered, even without a native "atomic scatter" operation:
1. **`sort`** the (value, destination-index) pairs by destination index (grouping everything headed to the same location together).
2. **`map`** across the now-sorted index sequence to compute a boolean "am I the start of a new group?" flag for every position (simply: compare each element to its immediate predecessor) — fully, trivially parallel.
3. **`segmented scan`** (with the desired combining operator, e.g., `+`) across the sorted values, using those group-start flags — collapsing each group of same-destination values down to one combined result, positioned at the end of each segment.
4. **`scatter`** those final combined values out to their (now unique) target locations.

This is a clean demonstration of how a relatively small, well-understood set of primitives — `map`, `sort`, `segmented scan`, `gather`/`scatter` — composes to express surprisingly sophisticated, genuinely irregular parallel computations.

(Two further primitives mentioned only briefly: **`filter`** — given a predicate function, keep only the sequence elements that satisfy it — and a **`groupBy`**-style operation common in data-processing systems: given a sequence of key-value pairs, produce a sequence of sequences, each grouping all values sharing the same key, directly analogous to a database-style group-by or a MapReduce-style shuffle stage.)

---

## 7. Case Study: Building a Uniform Grid for Particle Simulation

**The problem**: given a large number of particles scattered arbitrarily across 2D space (e.g., stars in a galaxy simulation, or fluid particles), and a fixed grid overlaid on space (say, a 4×4 grid of 16 cells), build a data structure that, for every grid cell, lists the IDs of the particles currently located in it — i.e., another sequence-of-sequences, useful for efficiently finding "nearby particles" in physics simulations (a very common building block for N-body-style simulations, discussed as directly related to this course's upcoming assignment work).

### First (natural) attempt, and why it doesn't scale
For every particle (in parallel): compute which cell it belongs to, then acquire a lock and append the particle's ID to that cell's list. This exposes plenty of *nominal* parallelism over particles — but in practice, essentially all of that parallelism immediately collapses into serialized contention on a shared lock (or, if per-cell locks are used instead, contention scoped down to whichever of the 16 cells happens to be popular) — fine for a small number of threads, but hopeless at the scale of hundreds of thousands of GPU threads all fighting over just 16 locks.

### Other tempting fixes, and why each still falls short at massive scale
- **Give every worker thread its own private cell-list structure, merged at the end**: avoids contention entirely during the main pass, but requires allocating (and later merging) as many separate cell-list structures as there are threads — perfectly reasonable with a modest thread count, but likely intractable with tens or hundreds of thousands of GPU threads, where the allocation and merge overhead alone becomes a serious cost.
- **Swap the axis of parallelization — parallelize over the 16 *cells* instead of over the particles**, with each cell-owning thread scanning the *entire* particle list itself, checking which particles belong to it. This trivially eliminates contention (no two "cell threads" ever touch the same output list) — but has two serious flaws: it caps total available parallelism at just 16 (far too little for a GPU-scale machine), *and*, even setting that aside, every one of those 16 threads independently does a full linear pass over *every* particle, making the total work scale with (cells × particles) rather than just particles — a substantial waste even before considering parallelism at all.

None of these approaches genuinely scales to the hundreds of thousands of threads a modern GPU actually needs kept busy.

### The fully data-parallel solution
Re-expressed purely in terms of `map`, `sort`, and `map` again:
1. **`map`**: for every particle (fully independently, no communication needed at all), compute which grid cell it falls into, given its (x, y) position.
2. **`sort`**: sort the particle-index / cell-ID pairs by cell ID — this single, well-understood, highly-parallel primitive is exactly what accomplishes the "grouping" needed, without any explicit locking or per-thread private structures at all.
3. **`map`**: for every position in the now cell-sorted array (again, fully independently — each thread just needs to look at its own position and its immediate neighbor), determine whether it's the *start* of a new cell's group (its cell ID differs from the previous position's), and use that to record, per cell, where that cell's block of particles begins and ends within the sorted array.

The resulting data structure — a per-cell "start" and "end" index into the sorted particle array — is exactly the sequence-of-sequences representation needed, built with **parallelism proportional to the number of particles**, not the number of cells, entirely free of locks, and just as capable of scaling to hundreds of thousands of GPU threads as any of the other primitives already covered — because it's built entirely out of them. (A closely related "histogram" variant of this same problem needs one further step — a segmented sum over each cell's group, with some care taken for empty cells — but is otherwise the same underlying pattern.)

---

## 8. Where This Shows Up in Practice

This style of thinking — decomposing an algorithm into a small set of well-understood, independently-optimized parallel primitives, rather than hand-writing thread/dependency logic — underlies several real, widely-used systems:
- **NVIDIA's Thrust** library for CUDA provides exactly this kind of API (map-like transforms, sort, scan, segmented scan, and more) for GPU programming, letting programmers express fairly sophisticated parallel algorithms without writing raw CUDA kernels by hand.
- **Apache Spark**, for distributed cluster computing, is built entirely around this same idea: its core abstraction (RDDs — Resilient Distributed Datasets) *is* essentially "sequence," and Spark programs are expressed purely in terms of a bounded set of operations over RDDs — which is precisely what lets Spark provide automatic parallelism across a whole cluster, along with fault tolerance, without requiring programmers to manage distributed threading or failure recovery by hand. (Spark itself is the subject of the next lecture.)

---

## 9. Summary

- Today's shift: instead of reasoning about individual "workers" and their explicit dependencies, express algorithms as compositions of a small set of operations over **sequences** — `map`, `fold`/`reduce`, `scan`, `segmented scan`, `gather`/`scatter`, `sort`, `filter`, and `groupBy`-style operations — each of which is assumed to already have a well-understood, highly-parallel implementation.
- Not every one of these primitives is *automatically* safe to parallelize for an arbitrary user-supplied function — `fold` and `scan`, in particular, require the combining operator to be **associative** for a parallel (tree-shaped) implementation to reliably match the sequential result.
- The *best* way to implement even a well-understood primitive like `scan` genuinely depends on the target hardware: a small number of independent cores favors simple divide-and-conquer; a SIMD-width-constrained warp favors a "wasteful-looking" but fully-lockstep O(n log n) approach over the asymptotically superior but SIMD-unfriendly O(n) algorithm; real systems combine both, hierarchically, depending on how much genuine parallelism is available at each level of a problem.
- Composing these primitives — as shown in the sparse matrix–vector multiply and uniform-grid case studies — makes it possible to express deeply **irregular** parallel computations (ragged rows, scattered particles) as clean, fully data-parallel pipelines with no explicit locks, no hand-written thread assignment logic, and parallelism that naturally scales with the true amount of independent work available, rather than with some artificial, coarser structural unit (like "rows" or "grid cells") that might not provide nearly enough of it.

---

*Notes synthesized and paraphrased from the CS149 Fall 2023 Lecture 8 slide deck and lecture transcript, for study purposes — not a verbatim transcript.*
