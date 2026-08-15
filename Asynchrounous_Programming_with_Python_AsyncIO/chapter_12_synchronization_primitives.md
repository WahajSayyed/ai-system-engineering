# Chapter 12: Synchronization Primitives

> Part 3 — Synchronization and Communication · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 12.1 Do We Even Need Locks in Single-Threaded Code?

Recall Chapter 3's key rule: **a coroutine runs uninterrupted between `await` points.** No other coroutine can possibly run in the middle of a straight-line stretch of code with no `await` in it. So do we still need locks, the way multi-threaded code does?

**Yes — whenever your "critical section" itself spans an `await`.** The moment you `await` something in the middle of a check-then-act sequence, you've handed control back to the event loop, and *anything* else can run before you resume — including another coroutine doing the exact same check-then-act sequence on the same shared state.

```python
import asyncio

cache = {}

async def get_or_fetch(key):
    if key in cache:
        return cache[key]

    print(f"Cache miss for {key!r}, fetching...")
    await asyncio.sleep(1)   # simulates a slow fetch — an await in the middle!
    value = f"value-for-{key}"
    cache[key] = value
    return value

async def main():
    # Two coroutines check the cache for the same key "at the same time"
    results = await asyncio.gather(
        get_or_fetch("user:42"),
        get_or_fetch("user:42"),
    )
    print(results)

asyncio.run(main())
```

Run this, and you'll see **both** coroutines print `"Cache miss..."` and both perform the (wasteful, duplicate) fetch — because the first one hadn't gotten around to writing to `cache` yet by the time the second one checked `if key in cache`. This is exactly the same class of bug threaded code has always had to guard against with locks — it's just triggered by `await` points instead of OS-level preemption.

---

## 12.2 `asyncio.Lock`

`asyncio.Lock` serializes access to a critical section, exactly the way you'd expect, using the familiar `async with` pattern:

```python
import asyncio

cache = {}
cache_lock = asyncio.Lock()

async def get_or_fetch(key):
    async with cache_lock:
        if key in cache:
            return cache[key]

        print(f"Cache miss for {key!r}, fetching...")
        await asyncio.sleep(1)
        value = f"value-for-{key}"
        cache[key] = value
        return value

async def main():
    results = await asyncio.gather(
        get_or_fetch("user:42"),
        get_or_fetch("user:42"),
    )
    print(results)  # Now only ONE "Cache miss" print — the second waits, then hits the cache

asyncio.run(main())
```

Now only **one** coroutine ever prints `"Cache miss..."` — the second one blocks on `async with cache_lock:` until the first releases it, and by then the cache has already been populated, so it just returns the cached value instead of duplicating the fetch.

**Important framing:** an `asyncio.Lock` isn't protecting you from memory-visibility issues the way a threading lock does (there's only one thread here — that problem doesn't exist). It's purely about **serializing logical critical sections that span `await` points**, so that interleaved coroutines can't observe or mutate shared state halfway through each other's work.

**Two things worth knowing:**
- `asyncio.Lock` is **not reentrant** — if a coroutine tries to acquire a lock it already holds (e.g., through a recursive call), it will deadlock waiting on itself.
- If a coroutine holding a lock is cancelled while inside the `async with` block, the lock is still correctly released on the way out — `async with` guarantees `__aexit__` runs during the unwind, just like an ordinary `finally`.

---

## 12.3 `asyncio.Event`: One Signal, Many Waiters

An `Event` is a simple flag with a built-in "wait until it's set" mechanism — useful for "don't start until setup is complete" style coordination, where any number of coroutines might be waiting on the same signal.

```python
import asyncio

ready = asyncio.Event()

async def worker(name):
    print(f"{name}: waiting for setup to finish")
    await ready.wait()
    print(f"{name}: go!")

async def setup():
    print("Running setup...")
    await asyncio.sleep(2)
    print("Setup complete")
    ready.set()   # wakes up every coroutine currently waiting on ready.wait()

async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(worker("A"))
        tg.create_task(worker("B"))
        tg.create_task(worker("C"))
        tg.create_task(setup())

asyncio.run(main())
```

All three workers print `"waiting for setup to finish"` immediately, then all three print `"go!"` at the same moment, right after `setup()` finishes and calls `ready.set()`. Unlike a `Lock`, an `Event` doesn't serialize anything — it broadcasts to every waiter at once. Call `ready.clear()` to reset it back to unset if you need to reuse the same `Event` for repeated cycles.

---

## 12.4 `asyncio.Condition`: Coordinated Waiting on Shared State

A `Condition` combines a `Lock` with the ability to **wait for a notification while holding logical access to shared state** — useful when you need custom "wait until this specific condition about shared data becomes true" logic that a `Queue` (Chapter 13) doesn't directly express.

```python
import asyncio
import collections

buffer = collections.deque()
MAX_SIZE = 3
condition = asyncio.Condition()

async def producer():
    for i in range(6):
        async with condition:
            await condition.wait_for(lambda: len(buffer) < MAX_SIZE)
            buffer.append(i)
            print(f"Produced {i}, buffer={list(buffer)}")
            condition.notify_all()
        await asyncio.sleep(0.2)

async def consumer():
    for _ in range(6):
        async with condition:
            await condition.wait_for(lambda: len(buffer) > 0)
            item = buffer.popleft()
            print(f"Consumed {item}, buffer={list(buffer)}")
            condition.notify_all()
        await asyncio.sleep(0.5)

async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer())
        tg.create_task(consumer())

asyncio.run(main())
```

`condition.wait_for(predicate)` is a convenience wrapper that repeatedly calls `condition.wait()` until `predicate()` returns truthy — handling the classic "wait, then re-check the condition, because someone else's notification might have been about something else" pattern for you. Notice both producer and consumer must hold the `condition` lock (via `async with condition:`) while checking and mutating `buffer` — this is the same check-then-act protection from §12.1, applied to a more elaborate shared structure.

In practice, for straightforward producer-consumer pipelines, `asyncio.Queue` (next chapter) already builds exactly this pattern for you and is almost always what you want — reach for `Condition` directly only when your coordination logic doesn't fit a simple queue's shape.

---

## 12.5 `asyncio.Semaphore` and `BoundedSemaphore`: Limiting Concurrency

A `Semaphore` allows up to **N** holders at once, rather than just one (a `Lock` is really a semaphore with N=1). The most common real use: **capping concurrency** — e.g., "never have more than 5 requests to this API in flight at once," even if you've fanned out 100 of them.

```python
import asyncio
import random

MAX_CONCURRENT = 2
sem = asyncio.Semaphore(MAX_CONCURRENT)
current_in_flight = 0
max_observed = 0

async def download(n):
    global current_in_flight, max_observed
    async with sem:
        current_in_flight += 1
        max_observed = max(max_observed, current_in_flight)
        print(f"Download {n} starting ({current_in_flight} in flight)")
        await asyncio.sleep(random.uniform(0.5, 1.5))
        current_in_flight -= 1

async def main():
    async with asyncio.TaskGroup() as tg:
        for n in range(6):
            tg.create_task(download(n))
    print(f"Max concurrent downloads observed: {max_observed}")  # never exceeds 2

asyncio.run(main())
```

All 6 downloads are created immediately, but the semaphore ensures no more than `MAX_CONCURRENT` (2) are ever inside the `async with sem:` block simultaneously — the rest simply wait their turn.

**`BoundedSemaphore`** is a stricter variant: it raises `ValueError` if you call `.release()` more times than you've called `.acquire()`, catching a real bug (accidentally releasing twice, which would let more coroutines through concurrently than you intended). Prefer `BoundedSemaphore` over plain `Semaphore` in almost all application code — the extra safety check costs you nothing and catches a genuinely nasty class of bug.

```python
sem = asyncio.BoundedSemaphore(2)
await sem.acquire()
await sem.acquire()
sem.release()
sem.release()
sem.release()  # raises ValueError: BoundedSemaphore released too many times
```

---

## 12.6 Hands-On Exercises

**Exercise 1 — Reproduce and fix the cache race.**
Recreate the §12.1 unlocked cache example, confirming both coroutines print `"Cache miss"`. Then add an `asyncio.Lock` as shown in §12.2, and confirm only one does.

**Exercise 2 — Startup coordination with `Event`.**
Recreate the §12.3 example with 5 workers instead of 3. Add timing (`time.perf_counter()`) to confirm all 5 workers proceed at essentially the same moment, right after `setup()` completes — not staggered.

**Exercise 3 — Bounded buffer with `Condition`.**
Recreate the §12.4 producer/consumer example. Change `MAX_SIZE` to 1 and speed up the producer relative to the consumer (shorter sleep), then observe how often the producer has to wait (`wait_for` blocking) because the buffer is full.

**Exercise 4 — Limit concurrency with `Semaphore`.**
Recreate the §12.5 example with 10 total downloads and `MAX_CONCURRENT = 3`. Confirm `max_observed` never exceeds 3, and time the whole run — it should take roughly `(10 / 3) * average_delay`, not `10 * average_delay` (fully sequential) or a flat `average_delay` (fully concurrent).

**Exercise 5 (stretch) — `BoundedSemaphore` catches a real bug.**
Write a small buggy worker function that calls `sem.release()` **twice** by mistake (e.g., once in the `async with` block's implicit release, plus once more manually — you'll need to manage acquire/release manually rather than via `async with` to construct this bug). Run it once with `asyncio.Semaphore` (bug goes unnoticed, and explain why that's dangerous) and once with `asyncio.BoundedSemaphore` (raises `ValueError`, catching the bug immediately).

---

## 12.7 Common Pitfalls

- **Assuming asyncio locks solve memory-visibility problems like threading locks do.** There's only one thread; the concern here is purely about interleaving across `await` points within a logical critical section.
- **Trying to re-acquire an `asyncio.Lock` you already hold, from the same coroutine.** It's not reentrant — this deadlocks against yourself, not against another coroutine.
- **Reaching for `Condition` when a plain `Queue` (Chapter 13) would do.** `Condition` is a lower-level tool for custom coordination logic; most producer-consumer needs are better served by `asyncio.Queue` directly.
- **Using plain `Semaphore` instead of `BoundedSemaphore` in application code.** The bounded variant catches accidental double-releases for free — there's rarely a reason not to use it.

---

## 12.8 Further Reading

- [Python docs: `asyncio` Synchronization Primitives](https://docs.python.org/3/library/asyncio-sync.html)

---

**Next: Chapter 13 — Queues and Producer-Consumer Patterns** (`asyncio.Queue`, worker pools, and backpressure — the higher-level tool that replaces most hand-rolled `Condition` code).
