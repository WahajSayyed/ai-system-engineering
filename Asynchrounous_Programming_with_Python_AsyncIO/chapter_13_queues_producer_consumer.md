# Chapter 13: Queues and Producer-Consumer Patterns

> Part 3 — Synchronization and Communication · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 13.1 From `Condition` to `Queue`

Chapter 12 built a bounded producer-consumer buffer from scratch using `asyncio.Condition`. `asyncio.Queue` is the pre-built, battle-tested version of exactly that pattern — same underlying idea (a shared buffer, waiting-until-space-available, waiting-until-item-available), dramatically less code, and it's what you should reach for by default whenever your coordination need fits a "producers put things in, consumers take things out" shape.

```python
import asyncio

async def producer(queue):
    for i in range(6):
        await queue.put(i)
        print(f"Produced {i}")
        await asyncio.sleep(0.2)

async def consumer(queue):
    for _ in range(6):
        item = await queue.get()
        print(f"Consumed {item}")
        await asyncio.sleep(0.5)

async def main():
    queue = asyncio.Queue(maxsize=3)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(queue))
        tg.create_task(consumer(queue))

asyncio.run(main())
```

Compare this to Chapter 12's `Condition`-based bounded buffer: the same behavior (blocking the producer when full, blocking the consumer when empty) with none of the manual `wait_for(predicate)` / `notify_all()` bookkeeping — `Queue` handles all of that internally.

---

## 13.2 Core `Queue` API

| Method | Behavior |
|---|---|
| `await queue.put(item)` | Adds an item; **suspends** if the queue is full (`maxsize` reached), until space frees up |
| `await queue.get()` | Removes and returns an item; **suspends** if the queue is empty, until something is put |
| `queue.put_nowait(item)` | Non-blocking put; raises `asyncio.QueueFull` if full instead of waiting |
| `queue.get_nowait()` | Non-blocking get; raises `asyncio.QueueEmpty` if empty instead of waiting |
| `queue.qsize()` | Current number of items (approximate under concurrency — use for monitoring, not exact logic) |
| `queue.empty()` / `queue.full()` | Convenience boolean checks |
| `queue.task_done()` | Called by a consumer after **finishing work on** an item it `get()`-retrieved — signals "this one's done" |
| `await queue.join()` | Suspends until every item that has been `put()` has also had a matching `task_done()` called |

That last pair — `task_done()` / `join()` — is the piece that's easy to forget but essential for a common real pattern: *"fill the queue, then wait until everything in it has actually been fully processed"* (not merely removed from the queue, but processed to completion). We'll use this heavily in the worker pool pattern below.

---

## 13.3 Queue Variants: `PriorityQueue` and `LifoQueue`

`asyncio.Queue` is FIFO (first in, first out) by default. Two variants change the retrieval order:

- **`asyncio.PriorityQueue`** — retrieves the *smallest* item first (using `heapq` internally), useful for "process urgent items before routine ones."
- **`asyncio.LifoQueue`** — a stack: last in, first out.

```python
import asyncio

async def main():
    pq = asyncio.PriorityQueue()

    # Items must be comparable. Since dicts/objects often aren't directly
    # orderable, use (priority, tiebreaker, data) tuples to avoid TypeErrors
    # when two priorities are equal and Python tries to compare the data itself.
    await pq.put((2, 0, "routine cleanup"))
    await pq.put((0, 1, "critical alert"))
    await pq.put((1, 2, "normal request"))

    while not pq.empty():
        priority, _, item = await pq.get()
        print(f"[priority {priority}] {item}")
    # Output order: critical alert, normal request, routine cleanup

asyncio.run(main())
```

The `tiebreaker` (a monotonically increasing counter) is a standard trick: if two items have the same priority, `heapq` needs a way to break the tie without falling back to comparing the actual payload — which fails with a `TypeError` the moment that payload isn't itself orderable (e.g., two dicts).

---

## 13.4 Backpressure: The Real Point of a Bounded Queue

Here's the production-critical concept a `Queue` gives you almost for free: **backpressure.**

If a producer is faster than its consumer(s) and you use an **unbounded** queue, the queue simply grows — unboundedly — consuming more and more memory the longer the mismatch persists. This is a common, sneaky source of memory leaks and eventual crashes in real systems (a slow downstream API, a queue silently ballooning until the process is OOM-killed).

Setting `maxsize` on the queue fixes this structurally: once the queue is full, `put()` **suspends the producer** until the consumer catches up and frees a slot. The producer is naturally, automatically throttled to match the consumer's actual processing rate — no manual rate-limiting logic required.

```python
import asyncio
import time

async def fast_producer(queue):
    start = time.perf_counter()
    for i in range(6):
        await queue.put(i)
        print(f"[{time.perf_counter() - start:.1f}s] put {i}")

async def slow_consumer(queue):
    start = time.perf_counter()
    for _ in range(6):
        item = await queue.get()
        print(f"[{time.perf_counter() - start:.1f}s] got {item}")
        await asyncio.sleep(1)   # deliberately slow processing

async def main():
    queue = asyncio.Queue(maxsize=2)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(fast_producer(queue))
        tg.create_task(slow_consumer(queue))

asyncio.run(main())
```

Watch the timestamps: the producer's `put()` calls visibly **stall**, spaced out to roughly match the consumer's 1-second processing rate, instead of dumping all 6 items instantly. That stalling *is* backpressure working as intended.

---

## 13.5 The Worker Pool Pattern

The single most common real-world use of `Queue`: a fixed pool of **N worker coroutines**, all pulling from one shared queue, processing items concurrently, with a coordinator that fills the queue and then waits for everything to actually finish.

```python
import asyncio
import random

async def worker(name, queue):
    while True:
        job = await queue.get()
        try:
            print(f"{name} processing job {job}")
            await asyncio.sleep(random.uniform(0.3, 0.8))  # simulate work
        finally:
            queue.task_done()   # ALWAYS mark done, even if processing raised

async def main():
    queue = asyncio.Queue(maxsize=5)

    async with asyncio.TaskGroup() as tg:
        workers = [tg.create_task(worker(f"worker-{i}", queue)) for i in range(3)]

        for job in range(10):
            await queue.put(job)

        await queue.join()   # blocks until all 10 jobs have been task_done()'d

        for w in workers:
            w.cancel()        # worker loops run forever by design — stop them explicitly
    # The TaskGroup block doesn't exit until the cancellation above has settled.

asyncio.run(main())
```

Walking through the shape of this: 3 workers start immediately, each looping forever (`while True: job = await queue.get()`). The coordinator puts 10 jobs into the (bounded) queue, then calls `await queue.join()`, which suspends until all 10 jobs have had `task_done()` called on them — meaning all of them have actually been processed, not merely dequeued. Only *then* does the coordinator cancel the worker loops, since they'd otherwise sit forever waiting on an empty queue with no more work coming.

---

## 13.6 Don't Let One Bad Job Kill the Whole Pool

There's a subtle danger in the worker above: if `process(job)` raises an exception that **isn't caught inside the loop**, it propagates out of the `worker()` coroutine entirely — and since these workers are running inside a `TaskGroup`, that failure will cancel every sibling (the other workers, and the coordinator) immediately, per Chapter 8. One bad job tearing down your entire worker pool is rarely what you want.

**Fix: catch exceptions per-item, inside the loop, and keep going.**

```python
async def worker(name, queue):
    while True:
        job = await queue.get()
        try:
            await process(job)   # might raise for THIS job specifically
        except Exception as e:
            print(f"{name}: job {job} failed: {e!r} — continuing with next job")
        finally:
            queue.task_done()   # mark done regardless of success or failure
```

Note the `finally: queue.task_done()` is unconditional — whether `process(job)` succeeds, raises a normal exception (caught above), or the worker itself is later cancelled (in which case `CancelledError` propagates through the `finally` too, and `task_done()` still runs before the cancellation continues unwinding), the job is always accounted for. Forgetting this is one of the most common `Queue` bugs (see pitfalls below).

---

## 13.7 Hands-On Exercises

**Exercise 1 — Compare `Queue` to `Condition`.**
Rewrite Chapter 12's `Condition`-based bounded buffer (Exercise 3 there) using `asyncio.Queue(maxsize=3)` instead. Count how many lines of code you removed, and note which parts of the manual `wait_for`/`notify_all` logic `Queue` replaced.

**Exercise 2 — `PriorityQueue` in action.**
Create 6 jobs with random priorities (0–3) and put them into a `PriorityQueue` in a deliberately scrambled order, using `(priority, counter, job_name)` tuples. Drain the queue and confirm items come out in strict priority order, with ties broken by insertion order via the counter.

**Exercise 3 — Observe backpressure directly.**
Recreate the §13.4 example. Then remove `maxsize` entirely (unbounded queue) and observe the producer's timestamps now show it dumping all 6 items almost instantly, regardless of the consumer's pace. Explain in 2–3 sentences why this could be dangerous with, say, 10 million items instead of 6.

**Exercise 4 — Build the full worker pool.**
Implement the §13.5 example yourself with 3 workers and 10 jobs. Add a shared counter (protected by a `Lock`, per Chapter 12) tracking how many jobs each worker processed, and print a final summary showing the distribution across workers.

**Exercise 5 — Survive a bad job.**
Modify your Exercise 4 pool so that job `#7` specifically raises a `ValueError` during "processing." First run it **without** the per-item `try/except` from §13.6, and confirm the entire `TaskGroup` — all workers, the whole pool — gets torn down by that one failure. Then add the per-item exception handling and confirm the pool now finishes all 10 jobs, logging job 7's failure without disrupting anything else.

**Exercise 6 (stretch) — `LifoQueue` ordering.**
Put items `1, 2, 3, 4` into an `asyncio.LifoQueue` in that order, then retrieve all four and confirm they come back in reverse order (`4, 3, 2, 1`).

---

## 13.8 Common Pitfalls

- **Forgetting to call `task_done()`.** If a worker retrieves an item via `get()` but never calls `task_done()` (e.g., because it returned early on some code path), `queue.join()` will **hang forever** — its internal unfinished-item count never reaches zero.
- **Calling `task_done()` more times than items were retrieved.** This raises `ValueError`, similar in spirit to `BoundedSemaphore` catching a double-release — it's telling you your bookkeeping is off.
- **Letting one bad item tear down an entire worker pool.** Inside a `TaskGroup`, an uncaught exception in one worker cancels every sibling. Catch exceptions **per item**, inside the loop, unless you genuinely want a single failure to abort the whole pool.
- **Using `PriorityQueue` with non-orderable items and no tiebreaker.** Two equal-priority items will make `heapq` attempt to compare the payloads directly, raising `TypeError` if they're not comparable (most dicts and custom objects aren't, by default). Always include an incrementing tiebreaker in the tuple.
- **Using an unbounded queue and assuming that's "safer."** It removes backpressure entirely — a persistent producer/consumer speed mismatch becomes unbounded memory growth instead of a visible, controlled slowdown.

---

## 13.9 Further Reading

- [Python docs: `asyncio.Queue`](https://docs.python.org/3/library/asyncio-queue.html)

---

**Next: Chapter 14 — Async Context Managers** (`async with`, `__aenter__`/`__aexit__`, `contextlib.asynccontextmanager`, and resource lifecycle management for things like connections and file handles).
