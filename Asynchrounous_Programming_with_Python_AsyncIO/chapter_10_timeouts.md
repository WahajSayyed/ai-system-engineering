# Chapter 10: Timeouts

> Part 2 — Core AsyncIO Building Blocks · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 10.1 Why Timeouts Are Non-Negotiable in Real Systems

Every network call, every external dependency, can hang far longer than expected — a stalled connection, an overloaded downstream service, a firewall silently dropping packets. Without a timeout, your coroutine will simply `await` forever, tying up resources and, in the worst case, cascading into a system-wide stall as more and more callers pile up waiting on the same stuck dependency.

A timeout is really just **cancellation (Chapter 9), automatically triggered after a duration elapses.** Everything you learned about `CancelledError`, cleanup with `finally`, and the "cancellation only lands at the next `await`" rule applies exactly the same way here.

---

## 10.2 `asyncio.wait_for()`: Bounding a Single Awaitable

The older, still-common tool: wrap **one** awaitable with a deadline.

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(5)
    return "done"

async def main():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=1)
        print(result)
    except TimeoutError:
        print("Timed out after 1 second")

asyncio.run(main())
```

If `slow_operation()` doesn't finish within `timeout` seconds, `wait_for()` cancels it internally (using the exact `task.cancel()` mechanism from Chapter 9), waits for that cancellation to actually settle (respecting any `finally` cleanup inside it), and then raises `TimeoutError`.

**Version note:** as of Python 3.11, `asyncio.TimeoutError` was made an alias of the built-in `TimeoutError` — so `except TimeoutError:` and `except asyncio.TimeoutError:` are the exact same thing on 3.11+. If you need to support older Python versions, use `except asyncio.TimeoutError:` explicitly for compatibility.

---

## 10.3 The Problem with `wait_for()` for Multi-Step Operations

`wait_for()` bounds a *single* awaitable. What happens when you have several sequential steps and want **one shared time budget** across all of them?

```python
# The awkward way: manually tracking elapsed time across steps
import asyncio
import time

async def main():
    deadline = time.monotonic() + 5   # total budget: 5 seconds
    try:
        remaining = deadline - time.monotonic()
        await asyncio.wait_for(step_one(), timeout=remaining)

        remaining = deadline - time.monotonic()
        await asyncio.wait_for(step_two(), timeout=remaining)
    except TimeoutError:
        print("Exceeded the overall budget")
```

This works, but it's error-prone boilerplate you'd have to repeat before every single step — and it's easy to instead (incorrectly) give each step its own fixed timeout, which silently allows the *total* time to be a multiple of what you intended:

```python
# COMMON MISTAKE: this allows up to 10 seconds total, not 5!
await asyncio.wait_for(step_one(), timeout=5)
await asyncio.wait_for(step_two(), timeout=5)
```

---

## 10.4 `asyncio.timeout()`: A Shared Budget for a Whole Block (Python 3.11+)

`asyncio.timeout()` solves this cleanly: it's an **async context manager** wrapping an entire block of code — however many `await` statements it contains — under one deadline.

```python
import asyncio

async def main():
    try:
        async with asyncio.timeout(5):
            await step_one()
            await step_two()
            await step_three()
    except TimeoutError:
        print("The whole block took too long, combined")

asyncio.run(main())
```

If the **combined** time across all three steps exceeds 5 seconds, whichever step is currently running gets cancelled (again, via the same underlying mechanism from Chapter 9), and a clean `TimeoutError` is raised out of the `async with` block. This is almost always what you actually want for "this logical operation, start to finish, must not take longer than N seconds" — and it needs no manual elapsed-time bookkeeping at all.

### Rescheduling the Deadline

Sometimes you don't know the right deadline upfront — e.g., you want to extend the timeout every time you receive a keepalive signal from a long-running operation. `asyncio.timeout()` supports this via `.reschedule()`:

```python
import asyncio

async def main():
    async with asyncio.timeout(5) as cm:
        for i in range(3):
            await asyncio.sleep(2)
            print(f"Keepalive {i} received, extending deadline")
            loop = asyncio.get_running_loop()
            cm.reschedule(loop.time() + 5)   # push the deadline further out
        print("All done, well within the (repeatedly extended) budget")

asyncio.run(main())
```

Without the `reschedule()` calls, this loop (roughly 6 seconds of sleeping total) would exceed the original 5-second deadline and raise `TimeoutError` partway through. Each `reschedule()` call gives it fresh runway.

### Absolute Deadlines: `asyncio.timeout_at()`

If you already have an absolute point in time (rather than "N seconds from now"), use `asyncio.timeout_at(when)`, where `when` is a value from `loop.time()` (not `time.time()` — asyncio's internal clock, per Chapter 3):

```python
loop = asyncio.get_running_loop()
deadline = loop.time() + 10
async with asyncio.timeout_at(deadline):
    await do_work()
```

---

## 10.5 Composing Timeouts with `TaskGroup`

Because both `asyncio.timeout()` and `asyncio.TaskGroup` (Chapter 8) use the same `cancelling()`/`uncancel()` machinery from Chapter 9, they **compose cleanly**: wrapping a `TaskGroup` in a `timeout()` block produces exactly the behavior you'd hope for — a clean `TimeoutError`, not a confusing `ExceptionGroup` full of raw `CancelledError`s.

```python
import asyncio

async def worker(name, delay):
    await asyncio.sleep(delay)
    return name

async def main():
    try:
        async with asyncio.timeout(2):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(worker("fast", 1))
                tg.create_task(worker("slow", 5))   # will get cut off
    except TimeoutError:
        print("Group timed out — all children were cleanly cancelled")

asyncio.run(main())
```

Here, once the 2-second deadline hits, `timeout()` cancels the task it's watching (the whole `TaskGroup` block), which in turn cancels both children; the `TaskGroup` settles that cancellation internally, and `timeout()`'s own `cancelling()`/`uncancel()` bookkeeping correctly claims that cancellation as its own, converting it into a plain `TimeoutError` rather than letting a raw `CancelledError` (or an `ExceptionGroup` wrapping one) leak out to your calling code.

---

## 10.6 Hands-On Exercises

**Exercise 1 — Basic `wait_for` timeout.**
Write a coroutine that sleeps for 5 seconds with a `finally` block printing `"cleaned up"`. Wrap it in `asyncio.wait_for(..., timeout=1)`, catch `TimeoutError`, and confirm both the `TimeoutError` is raised **and** `"cleaned up"` prints — proving the underlying task was properly cancelled, not just abandoned.

**Exercise 2 — Multi-step budget, the hard way vs. the easy way.**
Implement the same three-step, 5-second-total-budget scenario twice: once manually tracking elapsed time and calling `wait_for()` before each step (as in §10.3), and once using a single `async with asyncio.timeout(5):` block (as in §10.4). Confirm both produce the same behavior, and write 2–3 sentences on which you'd rather maintain in a real codebase.

**Exercise 3 — Rescheduling.**
Recreate the §10.4 keepalive example. Then remove the `reschedule()` calls and confirm the operation now raises `TimeoutError` partway through, proving the extensions were doing real work.

**Exercise 4 — Timeout wrapping a TaskGroup.**
Recreate the §10.5 example. Add a `try/finally` inside the `"slow"` worker to print when it gets cancelled, and confirm: (a) you get a clean `TimeoutError` at the top level, not an `ExceptionGroup`; (b) the `"fast"` worker's result is simply discarded (never retrieved) because the whole group was cut short before you could read anything from it.

**Exercise 5 (stretch) — Timeout doesn't hijack unrelated cancellations.**
Write a coroutine `inner()` that wraps its own body in `async with asyncio.timeout(10):` (a long, generous internal budget) doing some work. Wrap a call to `inner()` in an outer `Task`, and — from the *caller* — cancel that outer task after just 1 second (unrelated to `inner()`'s own 10-second timeout). Confirm the exception that propagates is a genuine `CancelledError`, **not** an `inner()`-generated `TimeoutError` — proving `asyncio.timeout()` only claims cancellations it caused itself, and correctly lets externally-caused ones pass through untouched.

---

## 10.7 Common Pitfalls

- **Giving each step of a multi-step operation its own fixed `wait_for` timeout.** This silently allows the *total* time to be a multiple of what you intended — use `asyncio.timeout()` for one shared budget instead.
- **Assuming a timeout cancels instantly, at the exact moment the deadline passes.** Same rule as Chapter 9: cancellation only actually lands at the *next* `await` point — a long CPU-bound stretch delays it exactly as it would delay any other cancellation.
- **Not handling cleanup inside timed-out code.** Timeouts are cancellation — anything you learned about `try`/`finally` and `asyncio.shield()` for critical cleanup in Chapter 9 applies here without exception.
- **Confusing `asyncio.TimeoutError` and built-in `TimeoutError` on older Python.** They're the same class as of 3.11+, but if you support earlier versions, catch `asyncio.TimeoutError` explicitly for compatibility.

---

## 10.8 Further Reading

- [Python docs: `asyncio.timeout()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.timeout)
- [Python docs: `asyncio.wait_for()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for)
- [Python docs: `asyncio.timeout_at()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.timeout_at)

---

**Next: Chapter 11 — Exception Handling in Concurrent Code** (a focused look at `except*` patterns in practice, swallowed exceptions in fire-and-forget tasks, and `gather(return_exceptions=True)` versus `TaskGroup` for error aggregation — tying together everything from Chapters 6–10).
