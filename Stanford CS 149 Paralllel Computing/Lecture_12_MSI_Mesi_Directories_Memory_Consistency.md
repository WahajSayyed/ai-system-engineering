# CS149: Parallel Computing — Lecture 12
## Finishing Cache Coherence (MSI/MESI, Directories, False Sharing) and Introducing Memory Consistency

> Course: Stanford CS149 (Parallel Computing)
> Topics: (1) MSI protocol details + worked example, MESI optimization, directory-based coherence, NUMA effects, false sharing; (2) Introduction to memory consistency (sequential consistency, write buffers, TSO/PSO, relaxed models, fences, data races) — continued next week

---

## 0. Recap From Last Lecture

**Definition of coherence**, restated precisely:
1. For any single address, all reads and writes to it by all processors can be placed into **some sequential order** consistent with each thread's own program order.
2. In that sequential order, a read returns the value of the **most recently written** value.

**Two invariants needed to implement this:**
- **Single-Writer, Multiple-Reader (SWMR):** at any time, an address is either in a **read-write epoch** (exactly one processor has access) or a **read-only epoch** (any number of processors may share read access).
- **Data-value invariant:** the value observed during a read-only epoch is exactly the value produced by the most recent write of the preceding read-write epoch.

**Why not write-through?** Every write would need to appear on the bus, exhausting bandwidth — a performance bottleneck. This motivates **write-back** caches, which need an actual protocol to stay coherent, since each processor can silently accumulate writes in its own private cache.

**Two questions a write-back coherence protocol must answer:**
1. When may a processor actually perform a write? → Indicated by a **modified/dirty** state; only one cache may be in this state for a given line at any time.
2. If another processor wants to read/write a line currently modified elsewhere, who supplies the data? → The **owner** — the cache currently holding the line in modified state.

---

## 1. The MSI Protocol in Detail

### 1.1 Recap of the Three States

- **Invalid (I):** the line is not present in this cache (accessing it is a miss).
- **Shared (S):** the line is valid, **read-only**, and may be present in one or more caches simultaneously. **Memory is always up to date** whenever a line is in Shared state.
- **Modified (M):** the line is valid in **exactly one** cache; the dirty bit is set. Equivalent to **exclusive** ownership — no other cache may have this line in any state, and memory is stale.

### 1.2 Processor Operations and Bus Transactions

- **Processor-initiated operations:** `PrRead`, `PrWrite`.
- **Bus transactions** (triggered by other processors, visible to all caches via snooping):
  - **BusRd** — "give me this line, I want to read it."
  - **BusRdX** (bus read exclusive) — "give me this line, I want to write/modify it."
  - **BusWB** (bus write-back) — "I'm writing a dirty line back to memory" (e.g., on eviction, or when supplying data to another requester).

### 1.3 State Transition Diagram — How to Read It

Each transition arc is labeled by an **action/transaction pair**: the initiating action, and the resulting behavior taken by the cache controller.
- **Green-labeled transitions** = initiated locally by *this* processor (`PrRead`/`PrWrite`).
- **Red-labeled transitions** = initiated by *another* processor's activity, observed via snooping on the bus.

### 1.4 Local (Processor-Initiated) Transitions

| From state | Action | Bus transaction | To state |
|---|---|---|---|
| Invalid | `PrRead` | `BusRd` (miss — fetch the line to read) | Shared |
| Shared | `PrWrite` | `BusRdX` (upgrade to get exclusive access) | Modified |
| Invalid | `PrWrite` | `BusRdX` (miss — fetch + go straight to exclusive) | Modified |
| Modified | `PrRead` / `PrWrite` | *none* (hit — do whatever you like) | Modified |
| Shared | `PrRead` | *none* (hit) | Shared |

- Any processor action that causes a bus transaction is, by definition, a **miss**. Once in Modified state, both reads and writes are pure hits with no bus traffic.

### 1.5 Remote (Bus-Snooped) Transitions

Every cache snoops all bus transactions; if a snooped transaction concerns an address the cache currently holds, it must react:

| From state | Snooped bus transaction | Action taken | To state |
|---|---|---|---|
| Shared | `BusRd` (another processor reading) | *nothing* — data is already consistent, and multiple readers are fine | Shared |
| Shared | `BusRdX` (another processor wants to write) | give up read access | Invalid |
| Modified | `BusRd` (another processor wants to read) | supply data via `BusWB`; write back to memory (since memory is stale) | Shared |
| Modified | `BusRdX` (another processor wants to write) | supply data via `BusWB` (the requester becomes the new sole owner) | Invalid |

- **Key point:** whenever a line transitions **out of Modified**, the owning cache must supply the up-to-date data (and, in the `BusRd` case, update memory too) — this is what a `BusWB` accomplishes.
- **On contention:** if multiple processors repeatedly write the same line, ownership can "bounce" back and forth between caches continuously, causing repeated bus transactions and misses — this directly hurts performance (illustrated in the worked example below).

### 1.6 Worked Example: MSI in Action

Setup: three processors (P1, P2, P3) sharing one tracked address X, each with a private cache.

| Step | Action | Bus transaction | State (P1 / P2 / P3) | Data comes from |
|---|---|---|---|---|
| 1 | P1 reads X | `BusRd` | S / – / – | Memory |
| 2 | P3 reads X | `BusRd` | S / – / S | Memory |
| 3 | P3 writes X | `BusRdX` | I / – / M | (write completes locally; P1 invalidated) |
| 4 | P1 reads X | `BusRd` | S / – / S | **P3's cache** (owner supplies data via `BusWB`, updates memory, both move to Shared) |
| 5 | P1 reads X again | *none — hit* | S / – / S | P1's own cache (no bus transaction; it's already Shared) |
| 6 | P1 writes X | `BusRdX` | M / – / I | P3 invalidated; P1 becomes sole owner |
| 7 | P3 writes X | `BusRdX` | I / – / M | Data supplied by P1 via `BusWB`; P1 invalidated |

- **Discussion point (invalid vs. "never accessed"):** functionally identical from the point of view of a subsequent access (both are misses), but conceptually different — a never-accessed line is a **cold miss**, while an explicitly invalidated line was evicted due to coherence activity.
- **Why is the Modified → Shared transition needed at all?** Because the SWMR invariant must be actively tracked: when a formerly-exclusive owner starts allowing other readers, the system needs to know it's no longer the sole writer.
- **Performance implication — communication increases memory latency:** in the example, P1's access after P3's write requires a bus transaction (much slower than a cache hit) purely because of the intervening remote write. This is the direct performance cost of shared-memory communication under coherence.

### 1.7 How MSI Maintains the Two Invariants

- **SWMR — single writer:** enforced because only one cache can ever be in Modified state (all remote caches must invalidate before a new writer gets `BusRdX`).
- **SWMR — multiple readers:** enforced via the Shared state, which any number of caches may hold simultaneously.
- **Serialization:** the bus itself provides serialization — only one transaction is in flight at a time, so all processors observe transactions (and thus state changes) in the same global order.
- **Data-value invariant:** maintained via `BusWB` — whenever a line leaves Modified state, the current owner is the definitive source of the latest value, and it supplies that value (to the requester, and to memory when going to Shared).
- **What happens under write contention?** The bus serializes all requests; each processor's write must complete (in Modified state) before the next requester's `BusRdX` can be serviced — writes to a highly-contended line effectively execute one at a time, in bus order, with the line "bouncing" between owners.

---

## 2. MESI: Adding an Exclusive State

### 2.1 The Problem MSI Leaves on the Table

- Going from **read** to **write** on a line **not shared by anyone else** still costs **two bus transactions** under MSI:
  1. `BusRd` (miss) to bring it in as Shared.
  2. `BusRdX` (miss) to **upgrade** Shared → Modified.
- This "upgrade" is wasteful when no other cache actually has the line.

### 2.2 The Fix: An Exclusive (E) State

- **MESI** adds a fourth state, **Exclusive (E)**: the line is valid in exactly one cache, and it is **clean** (not dirty) — i.e., it decouples "only one cache has it" (ownership) from "it has been modified" (dirtiness).
- **On a read miss where no other cache holds the line:** go directly to **Exclusive** instead of Shared (since we know we're the only one).
- **Benefit:** if that same processor later writes to the line, the Exclusive → Modified transition is a **local hit — no bus transaction needed**, since we already know no one else has a copy. This eliminates the wasted upgrade miss for the common case of unshared data followed by a write.
- Remaining transitions can be derived by extending the MSI diagram with this decoupling of "clean-but-exclusive" vs. "shared" vs. "dirty" — left as an exercise to work through.

---

## 3. Directory-Based Coherence (Scaling Beyond a Bus)

### 3.1 Why Buses Don't Scale

1. **Bandwidth:** all processors share one interconnect; every transaction for every line is visible to (and contends for the attention of) every processor, even ones that don't hold that line.
2. **Over-serialization:** a bus serializes **all** transactions system-wide, when coherence really only needs to serialize transactions **per cache line**.

### 3.2 The Directory Idea

- A **directory** tracks **which caches currently hold a given line**, so coherence messages (e.g., invalidates) can be sent **only to the processors that actually need them**, instead of broadcasting to everyone.
- This enables **more scalable interconnects** (point-to-point networks, rings) that don't require system-wide broadcast/serialization — snooping fundamentally requires a broadcast medium, which a ring or general network does not provide.

### 3.3 Example Implementation: Directory at a Shared L3

- Real-world example: a multicore chip (e.g., the i7 used in the course's "myth" machines) uses a **ring interconnect** (not a bus), so snooping isn't viable. Instead, a **directory is associated with the shared L3 cache**.
- **Requires the inclusion property:** every line present in any core's L2 must also be present in L3 — this guarantees the L3 directory has complete visibility into what's cached where.
- **Directory entry format (example):** a few bits per cache line — one **presence bit per processor/core** (e.g., 4 bits for a 4-core system) plus a **dirty bit**.
  - **Shared state:** multiple presence bits set (indicating which cores have a read-only copy); dirty bit clear.
  - **Modified state:** exactly **one** presence bit set (the owner) and the dirty bit set.
- On a write, the directory consults its bits to know exactly which cores to invalidate — no broadcast required.
- **Trade-off:** requires extra storage (the directory itself) somewhere in the memory system, but scales to **tens or even hundreds** of processors, unlike bus-based snooping.
- **Note:** the choice of directory vs. snooping is largely **orthogonal to the state machine** (MSI/MESI) itself — the states (Modified/Shared/Invalid/Exclusive) stay the same; only the *mechanism* for notifying/serializing per-line transactions changes (broadcast+snoop vs. targeted point-to-point directory messages).

---

## 4. Practical Implications for Programmers

### 4.1 Coherence Changes the Miss Distribution

- Moving from single-threaded to multi-threaded/multi-process execution **increases cache misses** at various levels of the memory hierarchy, purely due to coherence traffic.
- **NUMA (Non-Uniform Memory Access) systems:** multiple sockets, each with its own locally-attached DRAM, connected via an interconnect. A CPU has **much higher bandwidth/lower latency to its own local memory** than to a remote socket's memory.
  - Application programmers can often ignore this, but achieving peak performance may require OS-level allocation policies that place data near the processor that accesses it most.
- **Example latencies (Core i7-class chip):**
  - L3 hit, unshared: **40 cycles**
  - L3 hit, shared (clean, read by multiple cores): **65 cycles**
  - L3 hit, modified in a different core: **~75 cycles** (roughly double the unshared case)
- **Net effect on average memory access time:** since parallel execution shifts access patterns toward higher-latency paths (coherence traffic, remote sockets), a parallel program's average memory access time is typically **higher** than an equivalent sequential program's — this is the direct performance tax of communication.
- **Profiling tools:** Intel VTune (uses hardware performance counters to report cache misses and communication) on Intel systems; Apple Xcode Instruments for similar insight on Apple Silicon (M-series) — both can help identify which data structures are causing coherence-related misses.

### 4.2 False Sharing

- **Definition:** unintended communication/coherence traffic that occurs because **independent** data happens to land on the **same cache line** — even though no logical sharing is intended.
- **Root cause:** coherence operates at **cache-line granularity**, not per-word/per-byte — fine-grained coherence would help application programmers but would be a nightmare for hardware designers to implement efficiently.

#### Worked Example

- **Bad layout:** an array of per-thread counters packed tightly (`int counters[NUM_THREADS]`), where multiple threads' counters land on the same cache line.
- **Good layout:** pad each thread's counter into its own struct/cache-line-sized region so each thread's data occupies an exclusive line.
- **Measured impact (4-core system, simple increment workload):** ~**14.2 seconds** with the naive packed layout vs. ~**4.7 seconds** with padding — roughly a 3x difference from a purely artificial (no real logical sharing) coherence cost.
- **Also occurs in numerical/grid applications:** if a grid is partitioned across processors and a cache line happens to straddle a boundary between two processors' assigned regions, false sharing results.

#### Miss-Rate Data (Historical Stanford Benchmarks, ~30 years old but still illustrative)

- Categorizes misses into: **cold**, **capacity**, **conflict**, **true sharing** (genuinely shared data), and **false sharing** (only shared because of line placement).
- **True sharing misses decrease** as cache line size increases — larger lines better exploit spatial locality in genuinely shared data.
- **False sharing misses can increase** with cache line size (counter-intuitively) — e.g., in the "radiosity" benchmark, false sharing misses **rise** with larger lines; in an extreme case ("raytrace" or similar), the increase is dramatic. Larger lines mean more unrelated data gets swept into the same coherence unit.
- **Net miss-rate curve vs. line size** is often U-shaped: initially decreasing (spatial locality benefit dominates) then increasing (false sharing cost dominates) as lines get larger.
- **Implication:** if coherence were done at the (very large) granularity of a virtual memory page, false sharing would be far worse — reinforcing why page-granularity software coherence schemes are impractical.
- **Mitigation:** padding/alignment can reduce false sharing, but at the cost of wasted memory, and isn't always practical (e.g., if the application has phases requiring different data layouts).
- Typical cache line sizes today: **64 or 128 bytes**, and they have generally grown over time.

---

## 5. Transition: From Coherence to Consistency

- **Coherence** only defines behavior for accesses to a **single** memory address.
- **Memory consistency** addresses the harder question: what is the allowed/observed **ordering of accesses to *different* addresses**, as seen by different processors? It defines what a shared-memory parallel program is even allowed to mean.
- Framing: coherence effectively makes the memory system **behave as if there were no caches**; consistency matters **even in a system with no caches at all**, because it's fundamentally about defining legal reordering behavior for the memory system (and compiler).
- **Who needs to care:**
  - Synchronization library implementers, compiler writers, low-level OS/systems developers → **must** understand consistency models deeply.
  - Typical high-level application programmers using existing libraries/locks correctly → largely insulated from these details, **provided their programs are properly synchronized**.

---

## 6. Four Types of Memory Ordering

For two different addresses X and Y, four ordering relationships are possible in principle:

1. **Write → Read** ordering
2. **Read → Read** ordering
3. **Read → Write** ordering
4. **Write → Write** ordering

The central question of a consistency model: **which of these orderings does the memory system guarantee to preserve** across different threads?

---

## 7. Sequential Consistency (SC)

### 7.1 Motivating Example

```
Initially: A = 0, B = 0

P0:  A = 1;        P1:  B = 1;
     print(B);            print(A);
```

- Class discussion narrowed down the possible printed outputs: **(1,1)**, **(0,1)**, and **(1,0)** are all achievable under different interleavings, but **(0,0) is impossible**.
- **Reasoning via a "happens-before" graph:** to get `(0,0)`, you'd need `print(B)` to happen before `B=1` **and** `print(A)` to happen before `A=1`. Combined with each thread's own program order (`A=1` before `print(B)` on P0; `B=1` before `print(A)` on P1), this creates a **cycle** in the happens-before graph — an event would have to happen before itself, which is impossible. Hence `(0,0)` is unreachable under normal (non-reordering) execution.
- (A follow-up question about store buffers/out-of-order execution potentially reintroducing `(0,0)` was flagged for later discussion — see §8.)

### 7.2 Formal Definition

- **Sequential Consistency**, defined by **Leslie Lamport (1976)** (work for which he later shared the **2013 Turing Award**): the result of any execution is the same as if the operations of all processors were executed in **some sequential order**, and the operations of each individual processor appear in this sequence **in the order specified by its program**.
- Equivalently: SC preserves **all four** of the ordering types from §6 (write→read, read→read, read→write, write→write) — nothing gets reordered across threads relative to a single, globally agreed-upon interleaving.

### 7.3 The "Switch" Metaphor

- Imagine a single shared memory with one imaginary switch that randomly connects to one processor at a time, accepts a memory operation from it, then may switch to another processor and take a run of its operations, etc.
- Any such switching schedule preserves each processor's program order while interleaving across processors arbitrarily — this defines exactly the set of legal SC outcomes. Working through the example program's interleavings this way confirms `(0,0)` and (symmetrically) certain other combinations are excluded, while `(1,1)`, `(0,1)`, `(1,0)` remain legal.

---

## 8. Why Relax Sequential Consistency? (Performance)

- SC is the most intuitive model for programmers, but it's restrictive for hardware implementers, because it forbids reordering that could otherwise hide latency.
- **Motivating scenario:** a **write** that misses in cache (slow) followed by an unrelated **read** to a different address that would **hit** in cache (fast). Under strict SC, the read must wait for the write to fully complete — but since the addresses are unrelated, there's no correctness reason to stall the fast read behind the slow write.
- **Goal:** overlap/reorder unrelated operations to hide latency and improve throughput — this is purely a performance-driven relaxation.

### 8.1 Write Buffers Break SC

- **Write buffer:** writes are placed in a buffer and drained to the memory system asynchronously, while the processor is free to issue subsequent (unrelated) reads immediately, without waiting for the write to actually complete.
- **Consequence for the earlier example:** with write buffers, `(0,0)` **becomes possible** — e.g., both `A=1` and `B=1` sit in their respective write buffers while both `print(B)` and `print(A)` execute and observe the old (pre-write) values. This is **not** sequentially consistent.
- Write buffers can be placed between the processor and cache, or between the cache and the bus — either way, they break the write→read ordering guarantee.
- **Every major modern ISA implements write buffers**, because the performance benefit is large — but this means **no major modern ISA is purely sequentially consistent** by default; a *weaker* consistency model is required to describe legal behavior when write buffers are present.

### 8.2 Weaker Models: TSO, PSO, Processor Consistency

- **Total Store Order (TSO)** and related models relax the **write→read** ordering specifically: a read to Y is allowed to complete before an earlier write to X (different address) has become globally visible — this is exactly what a write buffer enables.
  - Under TSO, all processors still agree on a single order in which writes become visible (writes are seen by everyone in the same order).
- **Processor Consistency (PC)** is closely related but allows for the possibility that different processors observe writes becoming visible at (slightly) different times/orders — the lecturer noted some uncertainty about the precise distinctions between these named models in the moment, and pointed to further reading (the naming/definitions across TSO/PC/PSO are notoriously tricky — echoing the classic joke that the two hardest problems in computer science are naming things, cache invalidation, and off-by-one errors, humorously reframed here as "naming things and memory consistency").
- **Partial Store Order (PSO)** goes further and relaxes **write→write** ordering too — i.e., even writes from the *same* processor can become visible to others out of program order. This is more dangerous: e.g., if a flag variable is meant to signal that associated data has been written, PSO could let the flag update become visible before the data it's supposedly guarding, breaking naive synchronization idioms.
- **ARM** (as in most smartphone processors) uses a **relaxed consistency model** that does not guarantee any of the four orderings by default — this yields the most reordering freedom (and thus most performance headroom) but also the most programmer burden.

### 8.3 Fences (Memory Barriers)

- **Fence / memory barrier:** a hardware mechanism instructing the processor to **wait until all prior memory operations have completed/become globally visible** before allowing any subsequent memory operation to proceed.
- Effectively "slows things down" deliberately — much like locking is a mechanism that trades some performance for correctness guarantees, fencing is a heavier-weight version of the same idea, applied to memory ordering.
- Variants include **store fences** and **load fences**, restricting reordering around the fence in specific directions.
- Using fences correctly is nontrivial; low-level systems programmers rely on architecture-specific documentation for exact semantics.

---

## 9. Data Races and Programming Under Relaxed Consistency

- **Data race:** two accesses to the same address, at least one of which is a write, with **no synchronization** ordering them relative to each other.
- **Consequence:** in a program with data races, behavior is effectively **undefined/unintended** under relaxed consistency models — the reordering freedoms discussed above can surface in ways that violate naive intuition.
- **The practical solution: write data-race-free (properly synchronized) programs.**
  - Whenever data is shared, use appropriate synchronization (locks, etc.) around accesses.
  - The people who implement **synchronization libraries** and **compilers** must deeply understand the underlying hardware's consistency model.
  - **Application programmers**, in turn, can rely on correctly-synchronized programs behaving intuitively (as if sequentially consistent) **without needing to reason about the underlying relaxed hardware model themselves** — this is the practical payoff of the layered approach.

---

## 10. What's Next

- Memory consistency needs to be understood at **two levels**: the **hardware level** (what the processor/ISA actually guarantees) and the **language level** (what a high-level language's memory model promises, e.g., for properly synchronized code) — the language-level discussion is deferred to the following lecture.

---

## Summary / Mental Model

| Topic | Core idea |
|---|---|
| MSI protocol | 3 states (I/S/M); local `PrRead`/`PrWrite` trigger bus misses (`BusRd`/`BusRdX`) except when already in M; remote snooped transactions force S→I or M→(S or I) with a `BusWB` supplying the up-to-date data |
| MESI | Adds Exclusive (E): clean + sole owner, so a later local write is a free E→M hit instead of a costly S→M upgrade miss |
| Directory coherence | Tracks per-line presence (which caches hold it) so invalidations go only to relevant caches — enables scalable non-broadcast interconnects (rings, networks) instead of a bus; same states, different notification mechanism |
| NUMA | Local-socket memory access is faster than remote-socket; coherence + remote traffic raises average memory access time under parallel execution |
| False sharing | Unrelated data sharing one cache line causes unintended coherence traffic; padding/alignment can fix it at the cost of memory; miss rate vs. line size is often U-shaped (spatial locality helps, then false sharing hurts) |
| Coherence vs. consistency | Coherence = ordering guarantees for a **single address**; consistency = ordering guarantees **across different addresses**, and matters even without caches |
| Sequential Consistency (Lamport) | All operations appear to execute in one global sequential order consistent with each thread's program order; preserves all 4 ordering types; most intuitive but most restrictive for hardware |
| Write buffers | Break write→read ordering for performance, making some non-SC outcomes (e.g., "(0,0)") observable; motivates weaker models |
| TSO / PC / PSO | Progressively relax more ordering types (write→read, then also write→write) in exchange for more reordering freedom / performance |
| ARM relaxed model | Guarantees none of the four orderings by default — max performance freedom, max programmer responsibility |
| Fences | Explicit hardware mechanism to force completion/visibility of prior memory ops before proceeding — restores necessary ordering at a performance cost |
| Data-race-free programming | Properly synchronized programs can be reasoned about intuitively even on relaxed hardware; the burden of understanding consistency models falls on library/compiler/systems implementers, not typical application programmers |

### Key terms introduced
- MSI states (Invalid / Shared / Modified); `PrRead`, `PrWrite`, `BusRd`, `BusRdX`, `BusWB`
- MESI (adds Exclusive)
- Directory-based coherence, inclusion property, presence bits
- NUMA (Non-Uniform Memory Access)
- False sharing vs. true sharing
- Memory consistency, four ordering types (WR/RR/RW/WW)
- Sequential Consistency (Lamport), switch metaphor
- Write buffer, Total Store Order (TSO), Processor Consistency (PC), Partial Store Order (PSO), relaxed consistency (e.g., ARM)
- Fence / memory barrier (store fence, load fence)
- Data race, data-race-free programming
