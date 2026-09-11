# CS149: Parallel Computing — Lecture 11
## Wrapping Up Spark, and an Introduction to Cache Coherence

> Course: Stanford CS149 (Parallel Computing)
> Topics: (1) Spark/RDD fusion, fault tolerance, scale-up vs. scale-out; (2) Cache review + Cache Coherence (through the MSI protocol setup — continued next lecture)

---

## 0. Housekeeping

- Lecture ordering got shuffled: DNN-optimization content (Lecture 10) was originally scheduled later, but Assignment 3/4 timing shifted, so it came earlier than planned. No major content impact — just an FYI on why the sequence feels slightly out of order.

---

## PART 1: Finishing Spark

### 1.1 Recap: Why Spark Exists

- **Cluster computing recap:** independent nodes, each with its own OS and independent memory, communicating via **message passing** (not shared memory).
- **Motivating problem:** applications that make heavy use of **intermediate data** — e.g., iterative algorithms, or repeated ad-hoc queries over the same dataset.
- **MapReduce's fault-tolerance mechanism:** write intermediate data to **HDFS** (Hadoop Distributed File System), which is fault-tolerant via replication — but disk is roughly **100x slower** than memory, so this is a poor fit for iterative/interactive workloads.
- **Spark's key abstraction: RDD (Resilient Distributed Dataset)** — a read-only, ordered collection of records, built via a chain of **transformations** from data originally loaded from HDFS (e.g., extract lines → filter mobile views → filter Safari views → count). An **action** (like `count`) produces a final result (e.g., a scalar) rather than another RDD.

### 1.2 Locality Optimization #1: Fusion

- Two major mechanisms for optimizing locality (both already seen in the DNN lecture): **fusion** and **tiling**.
  - **Fusion** minimizes external memory accesses and increases arithmetic intensity by combining operations so intermediate results never leave fast memory. (Directly analogous to the Flash Attention example — a combination of fusion and tiling.)
- **Why RDDs enable fusion:** Spark transformations are bulk, declarative operations, so the Spark runtime can analyze the full chain and decide how to execute it efficiently — e.g., fusing "load lines → lowercase → filter mobile views → count" into a single pass that reads one line at a time and keeps everything else resident in memory, instead of materializing each intermediate RDD.

### 1.3 Narrow vs. Wide Dependencies

- **Narrow dependency:** each partition of a derived RDD depends on only a **single** partition of its parent RDD (e.g., `lower` partition 0 depends only on `lines` partition 0). This is what allows fusion — the whole chain can run locally on one node with **no cross-node communication**.
- **Wide dependency:** a partition of the derived RDD depends on **multiple** partitions across different nodes (e.g., `groupByKey`, which needs data from every partition to form groups). Wide dependencies require communication and **block fusion**.

#### Example: Avoiding Communication on a Join

- A naive `join` between two RDDs generally requires all-to-all communication, since any key could exist in any partition of either RDD.
- **Optimization:** if both RDDs are partitioned using the **same hash partitioner** (so a given key always lands in the same partition number in both RDDs), the join becomes a **narrow dependency** — the Spark runtime detects the matching partitioners and can fuse the join without cross-node communication.
- **Caveat raised in discussion:** this doesn't address potential **load imbalance** (a key with far more entries in one RDD than the other) — communication avoidance and load balance are separate concerns; communication cost is usually the bigger first-order issue.

### 1.4 Fault Tolerance via Lineage

- **Lineage** = the log of deterministic, functional transformations that describe how to (re)construct any RDD, starting from the original HDFS-backed data.
- Because RDDs are **read-only** and transformations are **functional** (never mutate inputs), any RDD can always be **recomputed** from the persisted (replicated, fault-tolerant) source data plus its lineage.
- **Why not materialize every intermediate RDD?** Only materialize what the user actually asked for; other intermediates can be recomputed on demand — you fuse as much as possible and avoid unnecessarily persisting data the user didn't request.
- **Recovery example:** if a node crashes mid-computation and some partitions are lost, Spark **replays the lineage** (the coarse-grained log of transformations) starting from the replicated source data to regenerate exactly the lost partitions — no need to re-run the entire job from scratch.
  - The lineage/log itself is tracked by a (presumably reliable/replicated) master node.
- **Boundary of the abstraction:** if you write code that breaks Spark's functional/immutable model (e.g., mutating inputs directly instead of using RDD transformations), you lose Spark's fault-tolerance guarantees — "you're on your own."

### 1.5 Performance: Spark vs. Hadoop

- Benchmark examples from the original Spark paper (~2012): **logistic regression** and **k-means**.
- **Hadoop:** first iteration ~80 seconds; subsequent iterations don't improve much, since each iteration requires an HDFS read *and* write. A binary (vs. text) in-memory representation helps marginally but disk access is still required.
- **Spark:** only needs the initial HDFS read; results are written straight to memory, so subsequent iterations are dramatically faster — **roughly 1–2 orders of magnitude** speedup, consistent with the memory-vs-disk bandwidth gap discussed earlier in the course.
- **Ecosystem:** Spark's RDD abstraction underlies several higher-level, domain-specific frameworks:
  - **Spark SQL** — database-style query processing.
  - **MLlib** — distributed machine learning.
  - **GraphX** — graph analytics (e.g., BFS-style operations, used in earlier iterations of this course's assignments).

### 1.6 Scale Up vs. Scale Out — And a Word of Caution

- **Scale out:** connecting independent nodes/servers over a network (no shared memory) — this is what Spark/distributed systems target.
- **Scale up:** connecting multiple cores within a **shared-memory** system (the model used throughout most of the rest of this course).
- **Motivation for scale-out systems:** handling datasets too large to fit in a single server's memory.
  - A single modern server might have on the order of **0.5–2 TB** of main memory.
- **Caution — don't over-apply distributed systems:** studies sometimes used datasets that easily fit in memory on one machine (e.g., ~5.7 GB Twitter graph, ~14.7 GB synthetic graph), yet still ran them on distributed frameworks.
  - Example data point: 20 iterations of PageRank on an in-memory-sized graph ran on **128 cores** via Spark but was still **~2x slower** than a single-threaded run — the distributed-system overhead outweighed the benefit.
- **Quote (paraphrased):** researcher Frank McSherry has criticized the big-data-systems community for prioritizing scalability as a goal in itself, sometimes creating overheads and then building elaborate mechanisms just to reduce those self-inflicted overheads — arguing that raw performance, not scalability alone, should be the primary metric.
- **Practical takeaway:** if your data doesn't actually require a distributed system (i.e., it fits comfortably in a single machine's memory), a single well-optimized machine will typically outperform a distributed cluster for that workload. Distributed systems make sense once you're dealing with hundreds of terabytes or more.

---

## PART 2: Introduction to Cache Coherence

### 2.1 Why This Matters

- Cache coherence has both **performance** and **correctness** implications, and directly affects how software developers must reason about shared-memory parallel programs.
- On modern chips, a large fraction of die area (30%+) is dedicated to cache — reflecting how critical locality/caching is to performance (off-chip memory access can cost hundreds of cycles of stalled compute).

### 2.2 Cache Review: The Three C's of Cache Misses

Quick review building on earlier lectures, using a 16-value array example and a generic cache-line-based cache:

- **Cache line:** a block of contiguous bytes/words moved as a unit. Reasons for cache lines:
  1. Exploit **spatial locality**.
  2. More efficient/cheaper to implement cache coherence and to move data in bulk (matches DRAM's bulk data-path granularity) than moving individual words.
- **Cold miss:** the very first access to an address — the cache has never seen it, so it can't possibly be present ("the cache is cold" with respect to that address).
- **Spatial locality example patterns:** sequential data access, and also **instruction fetch streams** (not just data).
- **Temporal locality example patterns:** repeatedly accessing the same address — e.g., a loop counter variable, or repeated stack accesses during recursive calls.
- **Capacity miss:** when the working set exceeds what the cache can hold, forcing eviction/replacement of a line that will be needed again later.
- **Conflict miss (the "third C"):** from the **three C's model** (cold, capacity, conflict), attributed to researcher **Mark Hill**.
  - Caches limit **associativity** — the number of possible locations ("ways") a given line can occupy — to make lookups cheap.
  - **Worked example (Intel Skylake, as in the "myth" cluster machines):**
    - L1 data cache: 32 KB, **8-way set associative**, per core.
    - L2 cache: per core, connected via a **ring interconnect** across cores.
    - L3 cache: shared, 8 MB.
    - Cache-line size: 64 bytes → a 32 KB L1 cache holds **512 lines**.
    - Fully associative lookup would mean checking all 512 possible locations for a given line — expensive. Limiting to **8 ways** means only 8 locations need to be checked, but this causes extra misses (conflict misses) that wouldn't occur with full associativity, when multiple lines competing for the same limited "buckets" evict each other even though the cache isn't globally full.
  - **Trade-off:** higher set-associativity → lower conflict-miss rate, but more expensive/complex lookup hardware. (Discussion noted this doesn't necessarily scale monotonically with cache size across a real design — e.g., set-associativity choices for L1/L2/L3 aren't simply "bigger cache → higher associativity.")

### 2.3 Cache Line Anatomy and Write Policies

- A cache line has two parts:
  - **Data** — the actual cached bytes.
  - **Metadata** — including:
    - **Tag**: encodes the memory address (line) that this cached data corresponds to. Caches are best thought of as **content-addressable**: given an address from the processor, the cache compares it against all stored tags to determine hit/miss — the internal array index/location doesn't matter, only whether some line's tag matches.
    - **Dirty bit**: indicates whether the cached copy has been modified relative to main memory.

- **Write-back vs. write-through:**
  - **Write-through:** every write to the cache is also immediately written to main memory. No dirty bit needed, since memory is always up to date.
  - **Write-back:** writes only update the cache (setting the dirty bit); the update is propagated to main memory later, typically when the line is evicted/replaced.

- **Write-allocate vs. no-write-allocate:**
  - On a **write miss**, write-allocate says: fetch the full line from memory (like a read miss) and then apply the write on top of it, so the rest of the line's bytes stay valid/coherent — as opposed to writing only to main memory and not bringing the line into cache.

- **Worked example — write-allocate, write-back cache, on a write miss (writing value `1` to address X):**
  1. Processor issues the write; it misses in the cache.
  2. The cache selects a location for the new line. If that location currently holds a **dirty** line, that dirty data is the *only* up-to-date copy in the system — it must be **written back to main memory** before being evicted (never simply dropped).
  3. Since this is write-allocate, the cache fetches the rest of the line's data from main memory (a read-miss-style fill).
  4. The write value is applied to the appropriate word in the now-resident line.
  5. The line's **dirty bit is set to 1**.

### 2.4 The Shared-Memory Correctness Problem

- **Intuitive expectation:** a load of address X should return the value from the **most recent** write to X — across all threads/processors.
- **The problem caches introduce:** once each processor has its **own private cache**, the same address can exist as multiple, independently-updatable copies simultaneously (in various caches and in main memory).

#### Walkthrough Example (from the lecture)

A shared variable `foo` at address X, with three processors (P1, P2, P3), each with a private cache, connected to main memory:

1. P1 loads X → cache miss → fetches `0` from memory.
2. P2 loads X → cache miss → also gets `0`.
3. P1 stores `1` to X → cache hit (write-back) → P1's cache now holds `1` (memory still has `0`).
4. P3 loads X → cache miss → gets `0` from memory (stale, since P1's update hasn't propagated).
5. P3 stores `2` to X → cache hit → P3's cache now holds `2`.
6. P2 loads X → cache hit in its own cache → gets `0` (stale).
7. P1 loads some other address Y, causing a **capacity miss** that evicts its cached copy of X (value `1`) back to memory.

**End state:** memory has `X = 1`; P3's cache has `X = 2`; P2's cache has `X = 0`; P1 no longer has X cached at all. Four different "truths" about the value of X exist simultaneously — clearly broken.

- **Important clarification:** this is **not** a problem that locks alone can fix. Even with perfectly synchronized (non-overlapping) writes, each processor could still write only to its *own* private cache, leaving the system globally inconsistent — synchronization of access ordering and coherence of cached copies are **separate, independent issues**.
- **Not unique to multiprocessors:** even a single-CPU system can see incoherence — e.g., a DMA-based I/O device writing directly into a memory buffer that's also cached can leave a stale cached copy. This occurs rarely enough that it can sometimes be patched with software (e.g., explicitly flushing relevant cache lines), but that approach doesn't scale to actively-shared, frequently-modified memory across many processors.

### 2.5 Coherence Invariants

- The core semantic goal: reads of address X should return the value of the **last write**, per the program order in which each thread issued its accesses — with all threads' writes interleaved into a single consistent global order for that address.
- Coherence is defined **per memory location** (or per cache line) — you serialize the sequence of accesses to *that* address such that every subsequent read sees the most recent write in the serialization, until the next write occurs.

**Two key invariants for a coherent system:**

1. **Single-Writer, Multiple-Reader (SWMR) invariant:** for any given address/cache line, at any point in time the system is in exactly one of two kinds of "epochs":
   - A **read-write epoch**: exactly **one** processor may hold/modify that line.
   - A **read-only epoch**: **any number** of processors may hold read-only copies simultaneously.
2. **Data-value invariant:** the value seen by all readers in a read-only epoch must be exactly the value produced by the **most recent write** of the preceding read-write epoch.

- There must be some explicit **mechanism for switching** between read-write and read-only epochs (e.g., something akin to a flush, though more efficient mechanisms exist — this is the subject of coherence protocols).

### 2.6 Implementing Coherence: Design Space Overview

- **Software-based approaches** (e.g., using OS-managed virtual memory page granularity) are possible but slow, and introduce a separate problem called **false sharing** (to be covered later).
- **Goal:** a **fine-grained, hardware-based** solution operating at cache-line granularity.
- **Two classic hardware approaches**, to be covered starting with the first:
  1. **Snooping** — the classic/historically first approach (covered first here — "more interesting and easier to understand").
  2. **Directory-based coherence** — the approach most common in large modern systems (covered in a later lecture).

#### Why Not Just Use a Single Shared Cache?

- Eliminates the multiple-copies problem by construction, but:
  - **Bandwidth bottleneck:** all processors contend for the same cache, which doesn't scale to many cores.
  - **Destructive interference:** one processor's accesses can evict another (unrelated) processor's data, causing extra capacity/conflict-style misses.
  - **Constructive interference is also possible:** e.g., a parallel `for` loop with interleaved iterations across processors accessing nearby data can benefit from another processor's fetch (this cuts both ways).
- Shared caches are impractical at **L1** but are used at **L2** in some designs — example given: the **Sun Niagara 2** processor, which used a shared L2 cache connected via a **crossbar** interconnect for bandwidth. Crossbars don't scale well (wiring cost grows **quadratically** with core count — an all-to-all network), which capped this design around 8 cores.

### 2.7 Snooping-Based Cache Coherence — Setting Up the Idea

- **Basic structure:** each processor has a private cache; caches are connected via a shared **interconnect** (a set of wires enabling communication among caches and memory). Each processor issues loads/stores to its own cache; each cache also **listens ("snoops")** on the interconnect for coherence-related messages triggered by other processors' actions.

#### A Naive Write-Through + Invalidate Scheme

- On a write, the writing cache **broadcasts an invalidation message** for that address: any other cache holding that line discards (invalidates) its copy.
- **Problem:** since this is write-through, **every write** must also appear on the interconnect — this quickly exhausts available bandwidth, making it a low-performance solution.

#### Moving to a Bus + Write-Back Design

- Switch the interconnect to a **bus**, which has two properties crucial for coherence:
  1. **Serialization:** a bus allows only **one transaction at a time** — this is *not* generally true for a ring or an arbitrary network, but is inherent to a bus. This gives a natural, built-in serialization point for coherence.
  2. **Broadcast:** every transaction placed on the bus is visible to all connected caches simultaneously ("like the air in the room").
- **Goal for a write-back coherence scheme:** ensure that whenever a processor writes a line, it is the **only** processor in the system currently allowed to do so — i.e., enforce the single-writer part of the SWMR invariant.
  - One way to represent this: exclusive ownership indicated via the line's **dirty bit** — but the coherence protocol must guarantee that **at most one cache** in the whole system can have a given line's dirty bit set at any time.
- This enforcement mechanism is a **cache coherence protocol**: hardware logic that watches both (a) the local processor's loads/stores and (b) messages from other caches arriving via the bus, and manages **per-line coherence state** accordingly.

### 2.8 Introducing MSI (Setup Only — Continued Next Lecture)

- **Protocol goals:**
  1. Ensure a processor can obtain **exclusive access** before writing.
  2. Correctly **locate the most recent copy** of a cache line on a miss (since main memory may be stale if another cache holds a dirty/modified copy).

- **MSI protocol — three per-line states** (name = the three states):
  - **M (Modified):** valid in exactly **one** cache; that cache's copy is the only up-to-date one (equivalent to "dirty"/exclusive).
  - **S (Shared):** the line may be present, **read-only**, in **multiple** caches simultaneously.
  - **I (Invalid):** the line is not present/not valid in this cache.

- **Two categories of events the protocol must react to:**
  - **Processor-initiated operations** (local, from this core): `PrRead`, `PrWrite`.
  - **Bus transactions** (from the interconnect, triggered by *other* processors' actions):
    - `BusRd` — "give me a copy of this line, I want to **read** it."
    - `BusRdX` (bus read exclusive) — "give me a copy of this line, I want to **write** it" (need the full line's data before modifying part of it, plus exclusive ownership).
    - `BusWB` (bus write-back) — "I'm writing a dirty line back to memory" (e.g., because it's being evicted).

**To be continued Thursday:** working through exactly how these three states transition in response to the two processor operations and three bus transactions to maintain coherence.

---

## Summary / Mental Model

| Section | Core idea |
|---|---|
| Spark fusion | Bulk, declarative transformations let the runtime fuse narrow-dependency chains, avoiding materialization of intermediate data — same fusion/tiling locality principles as GPU kernel optimization, applied at cluster scale |
| Spark fault tolerance | RDDs are immutable + transformations are deterministic/functional → a lineage log (not the data itself) is enough to reconstruct any lost partition after a failure |
| Scale up vs. scale out | Shared-memory multicore (scale up) vs. networked independent nodes (scale out); don't reach for distributed systems if your data fits comfortably on one machine — overhead often dominates |
| Cache misses (3 C's) | Cold (never seen this address), capacity (working set > cache size), conflict (limited associativity causes avoidable evictions) |
| Cache line metadata | Tag (identifies which memory line is cached — caches are content-addressable) + dirty bit (write-back tracking) |
| Write policies | write-back vs. write-through (when memory gets updated); write-allocate vs. no-write-allocate (whether a write-miss pulls in the full line) |
| Coherence problem | Private caches create multiple, independently-mutable copies of the same address — synchronization/locking alone does **not** solve this; it's an orthogonal issue |
| Coherence invariants | Single-Writer-Multiple-Reader (SWMR) + Data-Value invariant (readers see the last write) |
| Coherence approaches | Software (page-granularity, slow) vs. hardware (fine-grained, cache-line-based): **snooping** (bus-based, broadcast + serialization) vs. **directory-based** (later lecture) |
| MSI protocol (intro) | States: Modified (exclusive/dirty, 1 cache), Shared (read-only, many caches), Invalid (not present). Transitions driven by local `PrRead`/`PrWrite` and bus `BusRd`/`BusRdX`/`BusWB` — details next lecture |

### Key terms introduced
- RDD, transformation, action, lineage, narrow vs. wide dependency
- Scale up vs. scale out
- Three C's of cache misses: cold, capacity, conflict; set associativity
- Tag, dirty bit; write-back/write-through; write-allocate/no-write-allocate
- Cache coherence, SWMR invariant, data-value invariant
- Snooping, bus serialization + broadcast
- MSI protocol: Modified / Shared / Invalid; `PrRead`/`PrWrite`; `BusRd`/`BusRdX`/`BusWB`
