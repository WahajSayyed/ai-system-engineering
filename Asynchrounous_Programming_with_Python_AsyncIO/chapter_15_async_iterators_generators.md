# Chapter 15: Async Iterators and Generators

> Part 3 — Synchronization and Communication · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 15.1 The Async Iterator Protocol

Just as regular iteration is built on `__iter__`/`__next__`, async iteration is built on `__aiter__`/`__anext__`:

- **`__aiter__(self)`** returns the async iterator itself (almost always just `return self`)
- **`__anext__(self)`** is a **coroutine** that returns the next value, or raises `StopAsyncIteration` once exhausted

`async for` is the syntax that drives this protocol automatically:

```python
async for item in async_iterable:
    ...

# is roughly equivalent to:

iterator = async_iterable.__aiter__()
while True:
    try:
        item = await iterator.__anext__()
    except StopAsyncIteration:
        break
    ...
```

The key difference from regular iteration: **producing the next item can itself require an `await`** — a network read, a database round-trip, waiting for the next chunk of a stream. This is exactly the case regular `__iter__`/`__next__` can't handle, since `__next__` can't `await` anything.

---

## 15.2 A Manual Class-Based Async Iterator

```python
import asyncio

class Paginator:
    def __init__(self, total_pages):
        self.total_pages = total_pages
        self.current = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current >= self.total_pages:
            raise StopAsyncIteration
        self.current += 1
        await asyncio.sleep(0.3)   # simulates an async network fetch for this page
        return f"page-{self.current}"

async def main():
    async for page in Paginator(total_pages=5):
        print(f"Got {page}")

asyncio.run(main())
```

Each call to `__anext__` involves a real `await` — simulating, say, fetching one page of results from a paginated API. Forgetting to raise `StopAsyncIteration` once exhausted is the classic bug here: without it, the loop simply never ends.

---

## 15.3 The Easy Way: Async Generators

Writing a full class for this is usually unnecessary. An **async generator** — an `async def` function containing `yield` — gives you the same protocol automatically, with far less code:

```python
import asyncio

async def paginate(total_pages):
    for page_num in range(1, total_pages + 1):
        await asyncio.sleep(0.3)
        yield f"page-{page_num}"

async def main():
    async for page in paginate(5):
        print(f"Got {page}")

asyncio.run(main())
```

Identical behavior to the class-based `Paginator`, in a fraction of the code. This is almost always the right choice — reach for a hand-written `__aiter__`/`__anext__` class only when you need state or behavior a generator function genuinely can't express (e.g., supporting multiple independent iteration passes with reset logic).

**One thing async generators can't do:** like regular generators, they can't combine `yield` with a `return value` — `return` (with no value) is fine to stop early, but `return some_value` inside an async generator raises a `SyntaxError`.

---

## 15.4 Async Comprehensions

Python supports comprehensions over async iterables directly:

```python
async def numbers():
    for i in range(1, 11):
        await asyncio.sleep(0.05)
        yield i

async def main():
    evens = [n async for n in numbers() if n % 2 == 0]
    print(evens)   # [2, 4, 6, 8, 10]

asyncio.run(main())
```

**Don't confuse this with a different, unrelated feature:** you can also `await` an expression *inside* an ordinary comprehension over a regular (synchronous) iterable:

```python
async def fetch_one(x):
    await asyncio.sleep(0.1)
    return x * 2

async def main():
    # This iterates a plain sync list, but awaits INSIDE each iteration —
    # sequentially, one at a time (recall Chapter 4's sequential-await lesson!).
    results = [await fetch_one(x) for x in [1, 2, 3]]
    print(results)
```

The `async for` comprehension (§15.4's first example) is for iterating an **async iterable**. The `await expr for x in sync_iterable` pattern is for calling async functions **inside** a comprehension over ordinary data — the two look superficially similar but solve different problems, and neither one gives you concurrency by itself; both still run sequentially unless combined with `Task`s or `gather()`.

---

## 15.5 Streaming Pipelines: Chaining Async Generators

Async generators compose beautifully into multi-stage pipelines, where each stage is itself an `async for` over the previous stage — data flows through incrementally, without ever needing to be fully materialized in memory at once.

```python
import asyncio

async def raw_records():
    """Stage 1: simulate fetching raw records from a paginated source."""
    for i in range(1, 8):
        await asyncio.sleep(0.1)
        yield {"id": i, "raw": f"RECORD_{i}"}

async def parsed_records(source):
    """Stage 2: transform each raw record."""
    async for record in source:
        yield {"id": record["id"], "value": record["raw"].lower()}

async def valid_records(source):
    """Stage 3: filter out records we don't want."""
    async for record in source:
        if record["id"] % 2 == 0:   # keep only even ids, for example
            yield record

async def main():
    pipeline = valid_records(parsed_records(raw_records()))
    async for record in pipeline:
        print(record)

asyncio.run(main())
```

Each stage only pulls the next item from the stage before it exactly when it needs one — the whole pipeline processes one record at a time, end-to-end, rather than fetching everything, then parsing everything, then filtering everything. This is invaluable for large or unbounded data sources (paginated APIs, log tails, database cursors) where materializing the whole thing in memory upfront isn't practical.

---

## 15.6 Cleanup: `aclose()` and `contextlib.aclosing()`

Async generators can use `try`/`finally` around their `yield` for cleanup, exactly like the `@asynccontextmanager` pattern from Chapter 14:

```python
async def managed_resource():
    print("Opening resource")
    try:
        for i in range(5):
            yield i
    finally:
        print("Closing resource")

async def main():
    async for item in managed_resource():
        print(f"Got {item}")
        if item == 2:
            break   # we stop early, without exhausting the generator
```

**Run this, and `"Closing resource"` does *not* print right after the `break`.** Because the loop exited early rather than running the generator to exhaustion, the generator is left suspended mid-`yield` — its `finally` block only runs later, whenever the generator object is eventually garbage collected. In a short script this might happen almost immediately; in a long-running application, it might not happen for a long time, if ever, holding onto whatever resource was open the whole while.

**The fix:** explicitly close it, either directly...

```python
gen = managed_resource()
async for item in gen:
    print(f"Got {item}")
    if item == 2:
        await gen.aclose()   # runs the generator's finally block right now
        break
```

...or, more idiomatically, with `contextlib.aclosing()`, which guarantees `aclose()` is called on exit from the block regardless of how you leave it (early `break`, exception, or normal exhaustion) — directly analogous to how a regular `with` block guarantees cleanup:

```python
from contextlib import aclosing

async def main():
    async with aclosing(managed_resource()) as gen:
        async for item in gen:
            print(f"Got {item}")
            if item == 2:
                break
    # "Closing resource" has definitely printed by the time we get here
```

Prefer `aclosing()` any time you might exit an `async for` loop early over an async generator that holds a real resource — it removes the "did I remember to clean this up" risk entirely.

---

## 15.7 Cancellation Inside a Generator's `yield`

Since the "code between two `yield`s" inside an async generator is really just ordinary coroutine code, everything from Chapter 9 applies unchanged: if the task consuming the generator is cancelled while suspended at a `yield` (or at an `await` inside the generator), `CancelledError` propagates into the generator at that exact point, triggering its `finally` block just as you'd expect.

---

## 15.8 Hands-On Exercises

**Exercise 1 — Build the class-based `Paginator`.**
Implement `Paginator` from §15.2 yourself. Deliberately forget the `StopAsyncIteration` raise at first, run it, and confirm the loop never terminates (use a manual interrupt or a max-iteration guard to stop it safely). Then add the missing raise and confirm it now stops after 5 pages.

**Exercise 2 — Convert to an async generator.**
Rewrite `Paginator` as the `paginate()` async generator function from §15.3. Compare line counts.

**Exercise 3 — Async comprehension.**
Using the `numbers()` async generator from §15.4, write an async comprehension collecting only numbers divisible by 3 into a list, and a separate one collecting them into a set.

**Exercise 4 — Build the 3-stage pipeline.**
Implement `raw_records()`, `parsed_records()`, and `valid_records()` from §15.5 yourself, then add a **fourth** stage: `capped_records(source, limit)` that yields only the first `limit` records from whatever source it's given, stopping early with a plain `return`. Chain all four stages together.

**Exercise 5 — Observe (and fix) the delayed cleanup.**
Recreate the §15.6 `managed_resource()` example. Confirm breaking out early does *not* immediately print `"Closing resource"`. Then wrap the same loop in `contextlib.aclosing()` and confirm it now prints immediately at the point of the `break`.

**Exercise 6 (stretch) — Cancellation mid-generator.**
Wrap an `async for` loop over `managed_resource()` (with its `try`/`finally`) inside a `Task`. From the driving code, cancel that task while it's in the middle of iterating (after receiving 1–2 items but before exhaustion). Confirm `"Closing resource"` still prints, proving the generator's cleanup ran as part of the cancellation's unwind.

---

## 15.9 Common Pitfalls

- **Forgetting to raise `StopAsyncIteration` in a manual `__anext__`.** Without it, `async for` loops forever.
- **Assuming an early `break` out of `async for` closes the generator immediately.** It doesn't — cleanup is deferred until garbage collection unless you explicitly call `.aclose()` or use `contextlib.aclosing()`.
- **Confusing `async for` comprehensions with `await` inside a comprehension over a sync iterable.** They look similar but solve different problems — neither one gives you concurrency by itself.
- **Trying to `return a_value` from inside an async generator.** Like sync generators, only a bare `return` (or falling off the end) is allowed once `yield` is used in the same function.

---

## 15.10 Further Reading

- [PEP 525 – Asynchronous Generators](https://peps.python.org/pep-0525/)
- [PEP 530 – Asynchronous Comprehensions](https://peps.python.org/pep-0530/)
- [Python docs: `contextlib.aclosing()`](https://docs.python.org/3/library/contextlib.html#contextlib.aclosing)

---

**Next: Chapter 16 — Streams: TCP Clients and Servers** (moving from Part 3's coordination tools into Part 4's real networking — `asyncio.open_connection`, `start_server`, and building a simple chat server/client from scratch).
