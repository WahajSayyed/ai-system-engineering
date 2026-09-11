# CS149 — Lecture 1: Why Parallelism? Why Efficiency?

**Course:** Stanford CS149, Parallel Computing (Fall 2023)
**Instructors:** Prof. Kayvon Fatahalian & Prof. Kunle Olukotun
**Video:** https://www.youtube.com/watch?v=V1tINV2-9p4
**Course site:** https://gfxcourses.stanford.edu/cs149/fall23/

*Revised from the lecture transcript/audio — incorporates the live classroom discussion (in-class demos, Q&A, thought experiments) in addition to the official slide deck.*

---

## 1. Course Introduction

- Two instructors split responsibilities: Kayvon is the "software half," teaching most of the front half of the course; Kunle is the "hardware half," picking up the back half of the quarter to go deeper into how the hardware itself works.
- With around 270 students — the largest the course has had — the class draws a deliberately mixed audience: hardware architects, software engineers, machine-learning practitioners tired of waiting on training runs, and computer-graphics people. The opening icebreaker has students introduce themselves and share why they're taking the class, and a recurring theme in the answers is interest in the hardware/software boundary — understanding *why* code has to be structured a certain way to run fast.
- A framing the instructors return to throughout the course: modern computers are far more capable than most people intuitively expect. Developing a feel for what a given piece of hardware *should* be able to do (e.g., recognizing that a computation someone says is "taking all day" ought to really take a few seconds on ten cores) is treated as a genuinely useful professional skill, not just academic trivia.
- A quick poll early on (who has spawned a thread before, and in what context) surfaces that many students have already built simple thread pools in prior systems classes — usually for exactly the reason motivating this course: using multiple cores to get work done faster. The instructors note that threads get used for more than just raw parallelism, though — sometimes they're used purely to hide latency (e.g., in web servers) — and that distinction (parallelism vs. efficiency/latency-hiding) becomes a running theme of the course.

### Logistics Highlights
- **No textbook** — lecture slides are the primary reference material; students are encouraged to link out to good external explanations (blog posts, Stack Overflow answers, etc.) in the per-slide comments if something else explains a concept better.
- **Commenting system**: the course website supports commenting directly on individual slides. Conceptual questions ("I didn't understand X, can someone explain?") are meant to go there rather than on the class Q&A forum, so the discussion stays attached to the relevant material and is visible to everyone.
- **Four programming assignments**, forming the bulk of the grade:
  1. Multi-core / ISPC programming (a more elaborate thread-pool assignment involving task dependencies)
  2. Scheduling a task graph
  3. Writing a CUDA renderer on NVIDIA GPUs
  4. Implementing the Transformer module of a deep neural network as fast as possible (new this year) — the eventual target being fast token generation for a chatbot-style application
  5. *(Optional, released after Thanksgiving)* — can be used either to boost one of the four required assignments by roughly 10–15 points, or just done for fun
- **Written assignments** roughly every two weeks, built from previous years' exam questions, graded on making a reasonable effort — treated as exam practice and folded into the participation portion of the grade.
- **Per-lecture participation**: students are expected to post one substantive comment per lecture, in roughly the same week the lecture happens (not saved up and dumped right before exams) — averaging around two comments per week is considered reasonable. The instructors' rationale: writing a clear explanation is one of the best ways to actually test whether you understood something, and clear technical writing is a skill practicing engineers use constantly.
- **Grading breakdown:** ~58% programming assignments, roughly 30% combined across midterm and final exams, and the remaining portion split between written assignments and lecture-comment participation.
- **Final exam** is in-person, at the university-scheduled slot — this is treated as a firm requirement (no accommodation for pre-booked early travel, etc., without discussing it well in advance).
- **8 late days** for the quarter. Programming assignments can generally be turned in a few days late using these; written assignments are limited to at most one day late, since solutions get released and grading can't stay open indefinitely. Late days are intended to comfortably absorb ordinary life disruptions (minor illness, a busy week, travel), not to be requested case-by-case.

---

## 2. What Is Parallel Computing?

**Working definition:** A parallel computer is a collection of processing elements that cooperate to solve a problem quickly.

Two central concerns drive the whole course:
- **Performance** — how much faster can we get results?
- **Efficiency** — how well are we using the hardware to get that performance?

The instructors are explicit that although the course is titled "Parallel Computing," it is just as much a course about *efficiency* — and that the two are not the same thing. A solution can be highly parallel and still be a poor use of resources, and conversely, sometimes the simplest sequential approach is the genuinely efficient choice.

---

## 3. In-Class Demos: Humans as Processors

To build intuition before touching any real hardware, the lecture runs a series of live demonstrations where student volunteers act as "processors" tasked with summing numbers. Each demo is designed to surface a specific lesson about what actually limits real-world speedup.

### Demo 1 — Sequential baseline
One volunteer adds up 16 numbers alone, under mild "pressure" (being timed in front of the class). This takes roughly 40 seconds and serves as the baseline against which every subsequent demo is measured.

### Demo 2 — Two processors, unconstrained problem, real communication cost
Two volunteers each get 8 of a *new* set of 16 numbers and must produce the total sum — but they aren't allowed to talk to each other (they can pass a note). Naive expectation, stated by the class beforehand: two workers with twice the resources should finish in about half the time.

**What actually happened:** the two volunteers finished their individual halves quickly (in well under half the original time), but then had to relay one partial sum to the other so it could be combined — and that single hand-off ate up almost the entire time budget. Total time: essentially no improvement over the one-person baseline.

**Discussion, and the core lesson:** even a task that looks trivially parallelizable can have its entire speedup wiped out by the cost of moving a small amount of data between workers. When asked how they'd do better, students suggested: start walking toward each other while still computing (overlapping communication with computation), or use a much lower-latency communication channel (shouting, or a messaging app) instead of physically relaying paper. This is the class's first hands-on encounter with **communication overhead** as a limiter of parallel speedup.

### Demo 3 — Four processors, uneven work
Four volunteers are each given a share of the work, but (deliberately, as a setup) the shares are unequal — one volunteer gets more numbers, and harder ones to add. The result: most of the group finishes quickly and then simply waits, idle, for the one volunteer stuck with the disproportionate share.

**Lesson: load imbalance.** Even with correct, low-overhead work division, if the work isn't distributed evenly, the overall completion time is dictated by the slowest / most-loaded worker — everyone else's idle time is wasted parallel capacity.

### Demo 4 — Four processors, equal work, student-designed strategy
This time the total work is split evenly. Before running the demo, the four volunteers (and the class) discuss a strategy. The approach the volunteers land on: dump all the individual numbers into one shared pool and have each person repeatedly grab the next available number and add it to their own running total — a simple, informal form of **dynamic, on-demand work assignment** (conceptually similar to work-stealing).

**Result:** ~19 seconds total — individual summing was actually finished in about 12 seconds, but combining the four partial sums at the end (which happened serially, one at a time) added another ~7 seconds.

**Class discussion of alternative strategies:**
- Pre-splitting the numbers evenly up front (rather than dynamically grabbing from a shared pool) was raised as an alternative — avoids any risk of collision over the shared pool, though it doesn't help if the *difficulty* of items (not just the count) is uneven.
- Combining partial sums pairwise/hierarchically instead of serially at the end (e.g., 2+2 first, then combine those two results) was identified, after the fact, as an easy win that likely would have shaved a few seconds off the finish, since it reduces the length of the dependency chain at the combination step.
- A "dispatcher" idea was proposed: have one volunteer stand aside from the start, do no summation themselves, and simply wait to combine the other three volunteers' partial sums as each becomes available. The trade-off discussed: this means one person is idle for a while at the *start*, versus multiple people being idle for a shorter time at the *end* under the shared-pool approach — a real design trade-off between dedicating a coordinator role versus have every worker do symmetric work.

### Demo 5 — Estimating the size of the whole classroom (~150–160 people)
The class collectively designs and executes a strategy to count everyone in the room in parallel. Several strategies are proposed and discussed live:
- Split the room into sections (e.g., by seating block), have each row or section sum its own count, then combine those partial counts — though this still leaves whoever combines the sums waiting on the slowest section, and rows farther from the aggregation point wait longer.
- Start counting from the front of the room and the back of the room simultaneously, meeting somewhere in the middle.
- Physically reorganize: have everyone regroup into clusters of exactly 10 people, then just count the number of clusters — cleverly turning a big counting problem into a small one, but at the cost of significant **physical data movement** (getting ~160 people rearranged into new seats), directly analogous to the real cost of moving/reorganizing data in a computer system.
- A "speculative execution" suggestion: run two different counting strategies concurrently and just take whichever one finishes first, discarding the other's result — a nod to a real hardware/systems technique.

**What happened:** the class picked the "reorganize into clusters of 10" approach. It took roughly two minutes — far worse than a naive extrapolation from the 16-number, ~40-second baseline would suggest (roughly 10x the data should ideally cost only a little more than 10x the time for a scalable algorithm, and with real parallelism it should have been much faster than a purely serial approach). The overhead came almost entirely from coordination and physical movement: getting everyone into the right new seats, communicating who still needed to move, and so on.

**Postmortem, and the main takeaway of the whole exercise:** in hindsight, dedicating even a single additional person purely to counting while the reorganization happened — or, more provocatively, simply having *one* person serially scan and count every row without any parallel scheme at all — likely would have been competitive with, or faster than, the elaborate coordinated approach. **The point isn't that parallelism is bad — it's that communication, synchronization, and data movement are usually the real bottleneck**, and a solution that ignores those costs can lose to something much simpler. This is presented as the single biggest idea the instructors want carried out of the first lecture.

---

## 4. Three Course Themes

### Theme 1: Designing and writing parallel programs *that scale*
"Parallel thinking" involves:
1. **Decomposing** a problem into pieces that can safely run concurrently
2. **Assigning** those pieces of work to processors
3. **Managing communication/synchronization** between processors so it doesn't become the bottleneck (exactly what the demos above kept surfacing)

### Theme 2: How parallel hardware actually works
Understanding hardware matters because machine characteristics directly determine what's achievable — you can't reason about how to make code fast if you don't know what's actually happening underneath it.

### Theme 3: Efficiency is not the same as speed
**Key idea: FAST ≠ EFFICIENT.**

**Thought experiment posed to the class:** Imagine you're asked to speed up a program on a 10-core processor. A month later you report back with a 2x speedup. Do you get fired, or do you get a raise?

The class's answers split, and both sides have a point:
- **Case for "fired":** you had access to 10 processors and only achieved 2x — that's a very inefficient use of the available hardware; a more careful implementation should plausibly do much better.
- **Case for "raise":** not every program is equally parallelizable (some genuinely have limited available parallelism, like the human-summation demos with heavy communication costs); the *value* of a speedup depends on context, not just the raw multiplier — a 2x reduction in a web service's response time, or in a database query, can translate directly into real business value regardless of how many cores it "should" have used. A concrete example raised: a game running at 15 frames per second versus 30 — that 2x can be the literal difference between a game being shippable and not, independent of whether it used the hardware "efficiently" in a purist sense.

The upshot: raw performance and efficient hardware utilization are related but distinct goals, and reasoning about which one actually matters for a given situation is a core skill this course develops. Chip designers, in particular, care enormously about efficiency, since every bit of additional hardware capability put on a chip adds real manufacturing cost — the goal is the minimum hardware that reliably meets the performance target, not the maximum hardware you can fit.

---

## 5. Historical Context: Why Parallelism Matters Now

For decades, software developers could largely ignore parallelism because single-threaded CPU performance improved dramatically every year — one instructor recalls being told, as a student interested in parallel computing, to simply wait a year for processors to get faster rather than bother with parallel programming. That advice made sense for a long time, because two mechanisms reliably delivered more performance to *unmodified* sequential programs, year over year:

1. **Automatically extracting parallelism from your code without you ever knowing it** — i.e., hardware-level instruction-level parallelism (ILP) via superscalar execution (covered below).
2. **Increasing clock frequency.**

A useful way to see why both stopped working: picture three trend lines over time — **transistors per chip** (which kept climbing, tracking Moore's Law, with modern high-end GPUs now packing on the order of 80–100 billion transistors), **operations issued per clock** (which climbed for a while but then flattened once superscalar hardware ran out of exploitable ILP to find automatically — more on this below), and **clock frequency** (which climbed steadily for years and then flattened hard, roughly 15 years before this lecture, for reasons of power, not lack of manufacturing capability).

Both of the old free-performance levers ran dry at roughly the same point — which is the central reason parallel programming shifted from a specialty to a required skill: transistor counts kept growing (there's no shortage of hardware budget), but that budget could no longer be spent on making a single instruction stream faster, so it had to be spent on adding more parallel execution capability instead — which only pays off if software is written to use it.

---

## 6. Processor Fundamentals (Refresher)

### What is a program?
From the processor's point of view, a program — regardless of what source language it was written in — compiles down to a flat list of machine instructions (loads, stores, arithmetic, branches, etc.). The *meaning* of a program is that its output must be as if those instructions executed in exactly that order — but, critically, if a processor can produce an identical result by executing things in some other order (or in parallel), the program's meaning hasn't been violated, and it's free to do so. This distinction — the specified *order* of a program versus what a processor is actually allowed to do internally as long as results match — is foundational to the rest of the lecture.

### What does a processor do?
A simplified processor model has:
- **Fetch/Decode logic** — figures out which instruction to run next (including handling branches)
- **Execution unit (ALU)** — performs the operation an instruction specifies
- **Execution context** — the current state of the program: values held in registers, plus (not pictured in the simple diagram, but very much part of the state) values held in memory

At each clock tick, the simplest possible processor grabs the next instruction, executes it, and updates registers or memory accordingly — nothing more complicated than that as a starting mental model.

### Program state
"State" = the current values of all program data, held either in registers or in memory. Every instruction's entire effect, ultimately, boils down to changing this state.

---

## 7. Instruction-Level Parallelism (ILP) & Superscalar Execution

Using a small 5-instruction example — computing a dot product of two 3-element vectors (`a = x*x + y*y + z*z`), a genuinely common machine-learning operation — broken into 5 instructions (3 multiplies feeding into 2 sequential adds):

- A single, one-instruction-per-clock processor takes exactly 5 clock cycles to run this, since the program as written specifies a strict order.
- But looking at the actual **dependencies**: the three multiplies don't depend on each other at all — they could, in principle, all happen simultaneously if there were enough execution resources. The two adds, however, form a genuine dependency chain: the first add needs two of the multiply results, and the second add needs the first add's result.
- With **2 processors**, some improvement is possible.
- With **3 processors**, the three independent multiplies can all run in the same cycle — but no further speedup is available beyond that, because the two adds are stuck waiting on each other and on the multiply results in sequence. A 4th or 5th processor would sit idle; the dependency chain caps the achievable parallelism at 3 cycles no matter how many extra execution units are thrown at it.

This is the core intuition behind **superscalar execution**: hardware automatically scans a (necessarily narrow, local) window of upcoming instructions, finds ones that are mutually independent, and issues more than one per clock across multiple execution units — all without the program ever being rewritten, and without the processor "telling" the programmer it reordered anything.

### Diminishing returns
Early research (including work done at Stanford in the early days of parallel computing) studied how much ILP could be automatically extracted from ordinary, unmodified programs, without any help from the programmer. The finding: real programs' dependency structure only supports issuing somewhere around 3–4 independent instructions per clock, on average — building hardware to look for and issue *more* than that yields essentially no additional benefit, because the parallelism simply isn't there to find. This is exactly why the "operations per clock" trend line (mentioned above) flattened out even while transistor budgets kept growing: architects could no longer buy more automatic speedup this way, no matter how many more transistors they had to spend. (A real reference point mentioned: even a well-equipped early-2000s-era Pentium 4-class processor could already issue on the order of 3–4 instructions per clock using this kind of out-of-order superscalar logic.)

---

## 8. The Power Wall

With automatic ILP extraction maxed out, the other historical lever — raising clock frequency — also hit a hard limit: **power**. A useful rule of thumb given in lecture: power scales roughly with the *square* of clock frequency, making frequency a very costly way to buy more speed.

A vivid comparison offered in lecture: a high-end consumer GPU (like an RTX 4090) running a machine-learning workload at full tilt draws power in roughly the same ballpark — within about a factor of two — as a household microwave oven running at full power. That power becomes heat, and the practical limits of removing that heat are what ultimately capped how far clock frequencies could be pushed, roughly 15 years before this lecture.

**Net effect:** with both major single-thread performance levers stalled, chip architects redirected their still-growing transistor budgets toward **more parallel execution units** — multiple cores, and increasingly specialized cores for specific workloads — rather than trying to make one instruction stream faster.

**Consequence for developers:** the "free lunch" of automatically-faster sequential code is over. Illustrating just how large the gap has become: properly parallelized, well-optimized C++ code can run roughly **30–40x faster** than compiled-but-unparallelized C++ on an ordinary quad-core laptop — a preview of what the first assignment explores directly.

---

## 9. From Multi-Core to Specialized, Massively Parallel Hardware

Concrete points of scale mentioned in lecture:
- Consumer AMD chips are now available with **64 processor cores**.
- The RTX 4090 GPU contains on the order of **18,000 floating-point multiplier units**.
- The world's largest supercomputers today run into the **hundreds of thousands of CPU cores**, drawing power on the order of megawatts — comparable to the draw of a small town.
- This isn't limited to "big iron" — cracking open a modern smartphone reveals multi-core CPU *and* GPU processing already, alongside a variety of other specialized processing units.

### Beyond "just parallel" — specialization for efficiency
A recurring architectural response to the demand for efficiency: rather than build ever-more general-purpose cores, provide many *specialized* cores, each tuned for a specific task. Apple's A15 Bionic (used as the running example) has 6 general CPU cores split into 2 "big" cores (optimized for fast single-thread performance) and 4 "small," lower-power cores (better suited to background/parallel work) — plus additional dedicated silicon for the camera pipeline, a neural-network accelerator, various sensor processing, and more, none of which ever runs on the general-purpose CPU cores at all.

This pattern shows up at large scale too: Google's TPUs and Meta's (Facebook's) custom neural-network accelerator silicon are both mentioned as examples of large companies building their own specialized ML hardware, alongside a broader industry-wide trend of custom silicon built specifically for machine-learning workloads.

---

## 10. Memory & Caches (Introduced)

Memory, as an **abstraction**, is simple: it's just an array of addressable byte values. If you ask memory for the value at a given address, it gives you that value; if you tell it to store a value at an address, it does. Crucially, this abstraction says nothing at all about *how* that storage is physically built — that's an implementation question.

- **DRAM** is one common implementation of that abstraction — the actual off-chip physical storage most people picture when they hear "memory." It has relatively high capacity but comparatively high latency to access: a request to DRAM can take on the order of hundreds of processor clock cycles to come back.
- A **cache** is a second, on-chip storage mechanism that sits between the processor and DRAM. It holds much less data than DRAM, but is dramatically faster to access. A helpful analogy offered in lecture: if DRAM is like a garage — high capacity, but a walk away every time you need something — a cache is like the desktop right in front of you: much less room, but nearly instant to reach for.

### Illustrating cache behavior
Working through a simple example (a 16-byte memory address space, a cache with room for two 4-byte cache lines, and a sequence of accesses to addresses like 0, 1, 2, 3, 2, 1, 4, 1, ...):
- The very first access to a given cache line (e.g., address 0) has to go all the way out to memory — a full cache line's worth of data (e.g., addresses 0–3) is pulled in at once.
- Every subsequent access to an address that falls within an already-loaded line (1, 2, 3, and repeats of any of these) is served instantly from the cache — no trip to memory needed.

This single example already reveals two distinct reasons caches help: loading a whole line ahead of time effectively "pre-loads" the data for subsequent nearby accesses (benefiting programs that scan through memory in order), and simply holding onto recently-touched data means repeated accesses to the same address are fast. (The precise vocabulary for these two effects — spatial and temporal locality — along with a fuller treatment of cache miss types and eviction policy, is picked up in the next lecture.)

### Why this matters, visually
The lecture closes by comparing, to scale, how long a load instruction takes depending on where the data is actually found: a hit in a small, nearby cache is fast; a hit in a larger, more distant cache takes a bit longer; and a full miss all the way out to DRAM takes dramatically longer than either — visually driving home just how much performance is riding on whether or not data happens to already be in a cache when it's needed.

---

## 11. Summary

- Single-thread-of-control performance has essentially stopped improving on its own; meaningful speedups now require using multiple processing elements or specialized hardware.
- This means every programmer increasingly needs to know how to **reason about and write parallel, efficient code** — it's no longer an optional specialty.
- The in-class demos are the clearest illustration of the course's real throughline: writing a good parallel program is hard not because splitting up the *work* is hard, but because **communication, synchronization, and data movement** are usually where the real cost hides — and a solution that ignores those costs can easily lose to something much simpler.
- Encouraging note from the lecture: modern computers have far more raw processing power available than most people realize — the challenge (and the point of this course) is learning to use it efficiently.

---

*Notes synthesized and paraphrased from the CS149 Fall 2023 Lecture 1 slide deck and lecture transcript, for study purposes — not a verbatim transcript.*
