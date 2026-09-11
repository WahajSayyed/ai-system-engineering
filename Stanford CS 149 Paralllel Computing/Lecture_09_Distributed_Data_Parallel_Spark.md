# CS149 — Lecture 9: Distributed Data-Parallel Computing Using Spark

**Course:** Stanford CS149, Parallel Computing (Fall 2023)
**Instructors:** Prof. Kayvon Fatahalian & Prof. Kunle Olukotun
**Video:** https://www.youtube.com/watch?v=jaMWmLq422U
**Course site:** https://gfxcourses.stanford.edu/cs149/fall23/lecture/spark/

*Notes combine the official slide deck framing with the lecture transcript (live Q&A, worked examples). This lecture runs long and is explicitly left unfinished — it resumes the following Tuesday before the course moves on to cache coherence.*

---

## 1. Framing for This Lecture

Every lecture so far has optimized computation on a *single* machine — one or more chips, sharing memory, connected by an on-chip or on-board network. This lecture shifts scale entirely: how do you apply the same underlying data-parallel programming ideas (map, reduce, and friends, from the previous lecture) to a **distributed computer** — many separate nodes, each running its own independent operating system instance, with no shared memory at all? The primary vehicle for this discussion is **Spark**, though the lecture builds up to it via its predecessor, **MapReduce**, and the infrastructure (distributed file systems, warehouse-scale computers) both depend on.

Three central concerns drive everything in this lecture: **scaling** a data-parallel programming model to hundreds of thousands of cores, **fault tolerance** (something breaks, routinely, at this scale — the system has to keep going), and **efficient use of memory**, since memory bandwidth ends up being the resource that ultimately limits performance once the other two concerns are addressed.

---

## 2. Motivation: Why Use a Cluster At All?

The core reason isn't just "more compute" — it's **I/O bandwidth**. Processing a genuinely huge dataset (hundreds of terabytes — e.g., the log data behind a large website) on a *single* machine is fundamentally bottlenecked by that one machine's storage bandwidth: at roughly 50 MB/sec, reading through 100 TB takes on the order of **23 days**. Spreading that same data across 1,000 nodes, each contributing its own local storage bandwidth in parallel, cuts that down to roughly **33 minutes** — a thousand-fold improvement in aggregate I/O bandwidth. This is a genuinely new resource constraint not really discussed in earlier lectures (which focused on compute and memory bandwidth) — I/O bandwidth to persistent storage becomes the dominant concern once data no longer fits comfortably on one machine.

But unlocking that bandwidth means programming across hundreds or thousands of independent machines — and at that scale, **something is always failing**. Even if a single server has a mean time between failures measured in decades, a cluster of a thousand such servers will see a failure on a roughly daily basis, purely from the numbers. Any programming model for this setting has to treat failure as a routine, expected event, not an exceptional one.

---

## 3. Warehouse-Scale Computers

The infrastructure behind large-scale web services (search engines, social networks, e-commerce) is often described as a **warehouse-scale computer** — an idea credited to computer architect Luiz Barroso, who proposed treating an entire warehouse of networked machines (spanning networking, power, cooling, and programming model together) as a single, holistically-designed computing system, rather than just a large pile of independent PCs.

### Origins and evolution
The idea began with connecting cheap, commodity PCs together over Ethernet — an inexpensive way to build a large, scalable computer out of components everyone already had access to, especially in the early 2000s. Early Ethernet networking was comparatively slow, but as organizations like Google, Yahoo, and Facebook gained experience running these systems, they found that **the network specifically** was what most differentiated a productive cluster from an unproductive one — a genuinely high-bandwidth network dramatically simplified application development. As a result, warehouse-scale computers increasingly adopted custom, high-performance networking (much like traditional supercomputers), pushing their overall cost and sophistication closer to that of dedicated high-performance-computing systems, even though they started from "cheap commodity parts."

### Physical organization
- **Racks**: a warehouse contains many racks, each with a **top-of-rack switch** connecting it to the rest of the system, and typically **20–40 servers** inside — the exact count constrained largely by how much power can be delivered to the rack (roughly 12–20 kW for CPU-heavy racks; substantially less if the rack is instead packed with power-hungry GPUs).
- **Bandwidth**: modern intra-rack node-to-node bandwidth is on the order of 1–2 GB/sec; inter-rack (between racks, via the top-of-rack switches) bandwidth has grown from roughly 0.1 GB/sec in early systems up to around 2 GB/sec today.
- **A single node**: typically a dual-socket machine (2 physical CPU chips), each socket holding 16–32 cores, connected to 128 GB–2 TB of DRAM at roughly 100–200 GB/sec of memory bandwidth, plus 10–30 TB of local SSD storage, and network interfaces out to the rest of the cluster.

### A key observation about the bandwidth hierarchy
Comparing these numbers directly: **memory bandwidth (100s of GB/sec) dwarfs both local-disk and network bandwidth by roughly two orders of magnitude.** Historically, network bandwidth between racks was noticeably *lower* than local disk bandwidth (e.g., 0.1 GB/sec network vs. faster local SSD) — meaning fetching data from a remote node's disk was actually slower than just reading your own local disk. But as inter-rack network bandwidth has climbed toward roughly 2 GB/sec — now comparable to local SSD bandwidth — remote data has, for the first time, become nearly as cheap to access as local data, changing what kinds of scheduling and data-placement decisions actually matter. (This shift becomes directly relevant later, motivating why some design decisions that made sense for early MapReduce-era clusters became less important over time.)

### Communication between nodes: message passing, not shared memory
Since every node runs its **own, separate operating system instance**, there is no shared address space at all between nodes — communication has to happen via explicit **message passing**: a thread on one node issues a `send`, naming a source variable, a destination thread/node, and an optional message tag; a thread on the receiving node issues a matching `receive`, and the data lands in a variable in *its own* local address space. (The same basic mechanism can be used for communication within a single node too, but the interesting case here is genuinely crossing separate machines over the network.)

**Does message passing need separate, explicit synchronization on top of send/receive?** Not really — the act of a matching send and receive completing *is* the synchronization point; there's no additional locking or barrier construct layered on top needed for basic point-to-point communication. (This doesn't mean message-passing programs are immune to concurrency bugs — a receive waiting on a message that never arrives can still **deadlock** — but there's no *extra* synchronization primitive required beyond send/receive themselves, unlike in a shared-memory setting where locks/barriers are a separate concern layered on top of ordinary loads and stores.)

---

## 4. Persistent, Fault-Tolerant Storage: The Distributed File System

Since this infrastructure exists largely for **data processing**, and components fail routinely, the first requirement is a storage layer that simply never loses data, regardless of individual node failures. The standard solution: a **distributed file system** (pioneered by Google's GFS; the widely-used open-source equivalent is HDFS, the Hadoop Distributed File System).

### Design, matched to the expected access pattern
These systems are built around very large files (potentially hundreds of terabytes) with a specific, dominant access pattern: **mostly appended to, and mostly read — very rarely updated in place** — a natural match for log-style data, where new entries are continually appended as events occur, and later read back for analysis.

- Large files are divided into fixed-size **blocks** (commonly 64–256 MB).
- Each block is **replicated** across multiple nodes — deliberately spread across **different racks** (not just different nodes within the same rack), so that losing an entire rack's top-of-rack switch doesn't take out every copy of a given block at once.
- A designated **master (or "name") node** holds the system's metadata — essentially a directory mapping each file/block to the specific nodes currently holding replicas of it, functioning much like a file allocation table, but for a distributed system.
- A client wanting to read a file first contacts the master to discover which nodes hold the relevant replicas, then contacts one of those replica-holding nodes directly to actually fetch the data — a two-step protocol (metadata lookup, then data fetch) rather than routing all data itself through the master.

Practical notes raised in Q&A: coordinating multiple concurrent writers is generally handled at the application level rather than by the file system itself; the master/name node, despite being a single logical point of coordination, isn't usually a major fault-tolerance risk on its own (a single well-provisioned node fails rarely), though it can be replicated for load or additional redundancy if needed; with a typical replication factor of two to three copies per block, roughly a third to a half of total raw cluster storage capacity ends up dedicated purely to redundancy, in exchange for durability against node and even rack-level failures.

---

## 5. A Motivating Example, and Why Plain Message Passing (MPI) Falls Short

Concrete scenario: given a huge log of website page views (e.g., a wildly popular course website), determine what device type — mobile vs. desktop — visitors are using. In principle, this could be implemented directly with message passing, using something like MPI (the Message Passing Interface, a real, still widely used low-level API for exactly this kind of programming) — but doing so directly would be painful to write correctly, **and** it wouldn't, on its own, solve fault tolerance for the *computation itself*. The distributed file system already guarantees that data at rest is never lost — but if a node fails partway through a computation, whatever intermediate results it was holding *only in memory* are gone, and MPI provides no built-in mechanism to recover from that. A better answer needs a programming model with fault tolerance for the *computation*, not just for storage.

---

## 6. MapReduce

### Why `map` and `reduce` are the right building blocks here
Recall from the previous lecture: `map` has no dependencies between elements (fully parallelizable by construction) **and** — the property specifically emphasized in this lecture — it is **side-effect-free**: it never mutates its input. Because the input to a `map` operation is never modified, it can safely be **re-run** as many times as needed and always produce the identical result — which turns out to be exactly the property that makes robust fault recovery possible: if a machine computing part of a `map` fails partway through, its work can simply be redone elsewhere from the same, still-valid, unmodified input data, with no risk of the input having been corrupted or already partially consumed.

### The MapReduce programming model
A programmer supplies two functions:
- A **mapper**, called once per input record (e.g., once per line of a log file), which examines that record and emits zero or more **key-value pairs** (e.g., for a page-view log line: if the entry came from a mobile client, emit a key for the device type with a value of 1).
- A **reducer**, called once per **unique key**, given the full collection of values emitted under that key across the entire dataset, which combines them down to a single result (e.g., summing all the 1's associated with a given device-type key to get a total count).

### "MapReduce" is really "map → group-by-key → reduce"
A key structural point, drawn out directly in discussion: for the reducer phase to be correct and parallelizable, **all key-value pairs sharing the same key must end up routed to the same reducer task** — otherwise a reducer would only ever see a partial slice of the values for "its" key, and couldn't compute a correct combined result. This means there is necessarily a substantial **shuffle** (or "sort") stage sitting between the map and reduce phases — grouping all emitted pairs by key and routing them to the appropriate reducer — which is where a great deal of the network communication in a MapReduce job actually happens. (As in the previous lecture's discussion of `fold`, this also requires the reducer's combining logic to be associative for a parallel implementation to be correct.)

### Scheduling mapper tasks
Two competing approaches for deciding which node runs which mapper task, over an input file already divided into blocks and replicated by the distributed file system:
- **Dynamic work-queue assignment**: any node grabs the next available block's mapper task whenever it's free — good load balancing, but requires actually moving the block's data over the network to wherever the task happens to run.
- **Data/task-locality-driven assignment**: run each block's mapper task specifically on a node that *already* holds a local replica of that block, avoiding a network transfer for the primary input data entirely.

Historically, when networks were the clear bottleneck (well below local disk bandwidth), the locality-driven approach was strongly preferred — it minimized network traffic in the era when that traffic was itself the scarce resource. As discussed above, this trade-off shifts somewhat as network bandwidth improves and starts approaching local-disk bandwidth.

### Scheduling reducer tasks
Two related questions: which nodes run reducer tasks, and how does each mapper know where to send the key-value pairs it produces? A common approach: use a **hash function** on the key to deterministically assign each possible key to one of the available reducer tasks/nodes (e.g., all "Safari" entries hash to reducer node 0) — meaning every mapper can compute, independently and without any coordination, exactly which reducer each of its emitted pairs needs to go to. There is necessarily a **barrier** between the map and reduce phases — no reducer can safely begin combining values for a key until *every* mapper has finished emitting (since any late-finishing mapper could still contribute more values under any key). A further refinement raised in discussion: a **locality-aware** reducer placement could bias which physical node actually runs a given reducer task toward wherever most of that key's incoming data will originate, to minimize the shuffle's network cost — though, again, how much this matters in practice depends heavily on how much network bandwidth is actually available.

### Fault tolerance in MapReduce
Nodes are monitored via a simple **heartbeat** mechanism: each worker node periodically signals "I'm still alive" to the master node running the job scheduler; if a heartbeat stops arriving, that node is declared dead.

- **A failed mapper node**: since mapper input data was never mutated (and is durably replicated in the distributed file system), the scheduler simply reassigns that mapper's task to a different node holding a replica of the same input block, and it re-executes from scratch — completely safe, precisely because of `map`'s side-effect-free, non-mutating nature.
- **A failed reducer node**: if it had already fully completed before failing, its result is already safely recorded and nothing further is needed; if it failed partway through, its task simply has to be restarted (potentially requiring the relevant key-value data to be re-fetched from wherever it originated).

### Handling "stragglers" (slow, but not failed, machines)
In a large, long-lived data center, hardware is rarely uniform — some nodes are older, with fewer cores or lower clock speeds, and will simply run slower than newer nodes even with identical work assigned. The scheduler's solution: if a task is taking unusually long relative to others, **speculatively launch a duplicate copy of that same task on a different, presumably faster, node** — effectively racing the original and the backup against each other. Whichever finishes first "wins": its result is used, and the other, now-redundant copy is simply killed. This is only safe to do casually, without any special bookkeeping, because — once again — the underlying map/reduce functions are side-effect-free; running the same task twice, concurrently, causes no correctness problems at all.

### Why MapReduce succeeded — and where it falls short
**Strengths**: the programming model is simple enough to explain to most CS students almost immediately (a sharp contrast with hand-written message passing); it automatically divides work into mapper/reducer tasks, load-balances across many of each, supports locality-aware scheduling, and transparently recovers from both outright failures and stragglers — largely as consequences of just two design choices: side-effect-free functions, and durable, replicated storage underneath. This combination made it genuinely practical for a broad range of programmers (not just distributed-systems specialists) to harness hundreds of thousands of cores for data processing, and its influence has extended well beyond its original setting (echoes of the same ideas appear in this course's own GPU/data-parallel programming discussions).

**Limitations**:
- **Only a rigid, linear pipeline of map-then-reduce stages** is directly supported — a program is essentially map → reduce → map → reduce → ... in sequence. (An academic follow-on project, DryadLINQ, extended this to a full directed-acyclic-graph of stages — an interesting idea with real academic influence, though it didn't see the same widespread industrial adoption as MapReduce itself.)
- **Iterative algorithms are inefficient.** A classic example: PageRank (an iterative algorithm for ranking web pages by importance, originally developed by Larry Page, one of Google's founders) requires many repeated rounds of computation over the same underlying data. Implemented naively in MapReduce, **every single iteration** requires a full round-trip through the distributed file system — reading input from disk-backed storage, and writing results back out to disk-backed storage — even though the underlying dataset conceptually hasn't changed shape between iterations. Given how much slower disk-backed storage is than memory, this becomes a serious, repeated inefficiency across potentially many iterations.
- **Interactive, ad hoc querying is also inefficient**, for the same underlying reason: every distinct query against a dataset means yet another full pass reading from (relatively slow) persistent storage, with no way to keep frequently-reused data cached anywhere faster.

---

## 7. Motivating Spark: Use Memory, Not Just Disk

### The opportunity
A widely cited 2011 paper ("Disk Locality is Irrelevant") observed that, given typical per-node memory sizes of the era (around 64 GB), the **working sets** of real big-data production workloads at major companies (Facebook, Microsoft, and Yahoo, in the cited data) were overwhelmingly small enough to fit **entirely within aggregate cluster memory** — on the order of 97–99.5% of working-set data, across those three companies' workloads. (As clarified in discussion, "working set" here refers to the data actually being actively used/reused by a computation at a given time — not literally *all* data a job might ever touch, but the meaningfully hot, frequently-accessed portion of it.) The clear implication: **the memory capacity to keep most active data resident in RAM across an entire cluster already existed** — but MapReduce's programming model forced data to be repeatedly round-tripped through comparatively slow persistent storage anyway, between every stage.

### The core challenge: fault tolerance for data that only lives in memory
The obvious next idea — "just keep data in memory instead of writing it back to disk between stages" — runs immediately into a serious problem: **memory is volatile.** If a node holding some intermediate data in RAM fails (or the whole system loses power), that data is simply gone, with no automatic recovery path — unlike data durably replicated on disk by the distributed file system. **Spark's central goal**: **in-memory, fault-tolerant distributed computing** — get the massive performance benefit of operating primarily out of memory, without giving up the reliability guarantees that made MapReduce dependable at scale.

### Rejected alternative approaches
- **Actively replicate all in-memory data across multiple nodes/racks**, the same way the file system replicates disk blocks — correct, but potentially very network-intensive, undermining much of the performance benefit of staying in memory in the first place.
- **Maintain an explicit log of every update**, replayable to reconstruct lost state — technically workable, but potentially high-overhead to maintain continuously.
- **Simply checkpoint to the distributed file system periodically** (essentially, MapReduce's own approach) — safe, but exactly the lower-performance behavior Spark is trying to avoid.

---

## 8. Spark's Core Abstraction: the RDD

Spark's answer is a data structure called the **Resilient Distributed Dataset (RDD)**: a **read-only, ordered, immutable** collection of records. RDDs can only ever be created in one of two ways: by reading from persistent storage, or by applying a **transformation** to one or more *existing* RDDs. This is, at its heart, a natural extension of the same functional, non-mutating philosophy that made MapReduce's fault tolerance work — generalized into a richer, first-class programming abstraction.

### A worked example: building up a chain of RDDs
Starting from a raw log file in the distributed file system: read its lines into an RDD called `lines`; apply a `filter` transformation (checking for a mobile-client signature) to produce a new RDD, `mobileViews`; apply another `filter` (checking for "Safari") to that, producing `safariViews`; finally, apply an **action** (as opposed to a transformation) — e.g., a count — to actually produce a concrete result (not itself an RDD, but an ordinary returned value).

The recorded sequence of transformations that produced a given RDD — `lines` → `mobileViews` → `safariViews`, in this example — is called its **lineage**, and turns out to be the central mechanism Spark uses for fault recovery (discussed further below).

### Transformations vs. actions
Spark exposes a range of **transformations** (which always produce a new RDD, lazily, without necessarily doing any actual work yet): `map`, `filter`, `flatMap`, `sample`, `reduceByKey`, `join`, `sortByKey`, `partitionBy`, and others — alongside **actions** (which actually trigger computation and produce a concrete, non-RDD result back to the calling program): `count`, `collect`, `reduce`, `lookup`, `save`, and similar.

### Immutability as a safety property, restated
Since transformations never mutate an RDD's input — they only ever *read* from existing RDDs to *produce* new ones — the same input RDD can safely be reused as the source for multiple different downstream transformations without any risk of one consumer's processing interfering with another's, or with the ability to safely re-derive lost data later (directly generalizing the same property that made MapReduce's mapper re-execution safe).

---

## 9. Optimizing Spark Programs: `persist`

Consider reusing one intermediate RDD (e.g., `mobileViews`) as the input to *two* separate downstream computations — say, filtering it once for Chrome-specific views and separately for Safari-specific views. Without any special handling, Spark's default behavior would mean `mobileViews` isn't necessarily kept around anywhere convenient after being produced — meaning it might effectively need to be **recomputed (or re-fetched) from scratch** for each of the two downstream uses, redoing the earlier `filter` work (and its underlying storage read) twice over.

Calling **`persist`** on an RDD explicitly instructs Spark to **keep that RDD resident in memory** for reuse, rather than letting it be discarded (or re-derived from source) after its first use — directly avoiding this kind of redundant recomputation. This is presented as something the programmer sometimes has to request explicitly, though — as raised in discussion — a sufficiently sophisticated runtime can, in principle, analyze a program's structure and make some of these persistence decisions automatically, without the programmer needing to intervene by hand (echoing the same "help the compiler, or let a smart-enough system figure it out" theme from earlier optimization discussions).

---

## 10. Implementing RDDs Efficiently: Fusion and Narrow Dependencies

### Avoiding needless memory duplication
A naive implementation of a chain of RDD transformations — e.g., `lines` → (lowercase) → `mobileViews`, each partitioned across the cluster — could simply materialize every single intermediate RDD as its own fully duplicated array in memory at every stage. This is clearly wasteful: once an RDD has been fully consumed by whatever comes after it, there's no fundamental need to keep the *earlier* stage's data resident anymore (an initial improvement suggested in discussion) — and a more sophisticated system can potentially do even better than that.

### The same optimizations as earlier lectures, applied here
Spark's runtime can apply exactly the kind of optimizations already covered for single-machine code in the locality/communication lecture:
- **Loop/kernel fusion**: rather than materializing an intermediate result fully before starting the next transformation, combine adjacent transformations into a single pass wherever safely possible — directly analogous to fusing a chain of vectorized array operations into one pass to raise arithmetic intensity, as covered earlier in the course.
- **Tiling/blocking**-style reasoning about how much intermediate data genuinely needs to be materialized at once, to reduce peak memory usage.

**The key enabler**: these kinds of automatic transformations are generally very hard (often practically infeasible) to apply reliably to arbitrary, low-level code (like raw C) — but become far more tractable when working from a **high-level, semantically rich representation** that exposes the actual structure and dependencies of a computation directly (much as a framework like PyTorch can apply automatic fusion to tensor operations, as mentioned in the previous lecture). Spark's RDD/lineage graph is exactly this kind of higher-level representation, which is what makes automatic optimization genuinely practical here.

### Narrow vs. wide dependencies
This optimization potential hinges on understanding the **dependency structure** between RDD partitions:
- **Narrow dependency**: each partition of a derived RDD depends on exactly **one** partition of its parent RDD (e.g., partition 0 of `mobileViews` depends only on partition 0 of `lines` after a lowercase-then-filter chain) — this is the case for simple, per-record transformations like `map` and `filter`. Chains of narrow dependencies are exactly the case where automatic fusion (avoiding materializing every intermediate stage) is straightforwardly safe and beneficial.
- **Wide dependency**: a derived RDD's partition may depend on data spread across **multiple** partitions of its parent (e.g., a `groupByKey`-style operation, which — like MapReduce's shuffle stage — needs to gather same-keyed data that could originate from anywhere). Wide dependencies inherently require real data movement across the cluster, and can't be transparently fused away the same way narrow-dependency chains can.

---

## 11. To Be Continued

This lecture runs out of time partway through the fusion/dependency-graph discussion. The plan, as stated: finish the remaining Spark material (fault tolerance mechanics for RDD lineage recovery, and further scheduling/optimization details) the following Tuesday, before the course moves on to **cache coherence** — the first of the more hardware-focused lectures in the back half of the quarter.

---

## 12. Summary (of material covered so far)

- Distributed, cluster-scale computing is motivated primarily by **I/O bandwidth** — processing datasets far larger than any single machine can read through in reasonable time — and requires treating failure as a routine, expected condition rather than an exception.
- **Warehouse-scale computers** are built from racks of independent, separately-OS'd nodes connected by increasingly high-bandwidth networking; nodes communicate purely via **message passing**, since there's no shared address space across machines.
- A **distributed file system** (GFS/HDFS-style) provides durable, replicated storage as the foundation for fault-tolerant data processing, matched to a large-file, mostly-append/read access pattern.
- **MapReduce** builds a genuinely simple, robust distributed programming model on top of two properties of `map`/`reduce`: no inter-element dependencies, and (critically, for fault tolerance) **no mutation of input data** — which is exactly what makes re-executing failed or straggling tasks safe and correct, and what enables automatic task division, locality-aware scheduling, and load balancing. Its major limitations are a rigid linear map-then-reduce structure and poor efficiency for iterative algorithms or ad hoc, interactive querying, both stemming from forcing every stage through comparatively slow persistent storage.
- **Spark**'s central goal is preserving MapReduce's fault-tolerance guarantees while primarily operating out of **memory** rather than disk, motivated by the observation that most real production working sets already fit comfortably in aggregate cluster RAM.
- Spark's core abstraction, the **RDD**, generalizes MapReduce's "never mutate your input" philosophy into a full, immutable, lineage-tracked data structure — enabling both fault recovery (via lineage) and further automatic optimizations, like fusing chains of **narrow-dependency** transformations, made practical specifically because Spark programs are expressed in a high-level, semantically-rich form rather than arbitrary low-level code.

---

*Notes synthesized and paraphrased from the CS149 Fall 2023 Lecture 9 slide deck and lecture transcript, for study purposes — not a verbatim transcript. This lecture continues into the following session; a follow-up set of notes may be warranted once that continuation is covered.*
