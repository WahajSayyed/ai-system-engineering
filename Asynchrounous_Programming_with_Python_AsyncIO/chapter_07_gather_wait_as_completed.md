# Chapter 7: Running Things Concurrently — gather, wait, as_completed

> Part 2 — Core AsyncIO Building Blocks · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 6.1 Three Tools, Three Trade-offs

Chapter 5 showed you can manually `create_task()` several coroutines and `await` each one. In practice, asyncio gives you three higher-level helpers for exactly this "run several things concurrently, then do something with the results" pattern — and they differ in ways that matter:

| Tool | Result shape | Ordering | Default exception behavior |
|---|---|---|---|
| `asyncio.gather()` | A single list of results | Matches **input order**, regardless of completion order | First exception propagates immediately; **siblings keep running in the background** |
| `asyncio.wait()` | `(done, pending)` sets of Tasks | No inherent order (sets) | Never raises — you inspect `.exception()` on each task yourself |
| `asyncio.as_completed()` | An iterator | **Completion order** — first-finished yielded first | Exception raised at the point you `await` that particular result |

None of these is strictly "better" — they solve different shapes of problem. This chapter works through each, with special attention to the exception-handling gotchas, since that's where most real bugs hide.

---

## 6.2 `asyncio.gather()`: Simple, Ordered, but Watch the Exceptions

```python
results = await asyncio.gather(*aws, return_exceptions=False)
```

`gather()` is the tool you'll reach for most often: pass it several awaitables, get back a list of their results **in the same order you passed them in** — even if they finish in a completely different order.

```python
import asyncio

async def fetch(name, delay):
    await asyncio.sleep(delay)
    return f"{name} (took {delay}s)"

async def main():
    results = await asyncio.gather(
        fetch("slow", 2),
        fetch("fast", 0.5),
        fetch("medium", 1),
    )
    print(results)
    # ['slow (took 2s)', 'fast (took 0.5s)', 'medium (took 1s)']
    # "fast" finishes first internally, but its result still lands at index 1 —
    # matching the order it was PASSED IN, not the order it FINISHED.

asyncio.run(main())
```

### The exception gotcha

Here's the part that surprises people: by default (`return_exceptions=False`), if one awaitable raises, `gather()` immediately re-raises that exception to your calling code — **but the other still-running awaitables are not cancelled.** They keep executing in the background, and if you don't do anything else, you may never see their results *or* their own exceptions (which can later surface as an unrelated-looking "Task exception was never retrieved" warning).

```python
async def flaky(name, delay, fail=False):
    await asyncio.sleep(delay)
    if fail:
        raise ValueError(f"{name} failed")
    return name

async def main():
    try:
        await asyncio.gather(
            flaky("A", 1, fail=True),
            flaky("B", 3),   # keeps running even after A's exception propagates!
        )
    except ValueError as e:
        print(f"Caught: {e}")
    # At this point, "B" is still sleeping in the background, unmanaged.
    await asyncio.sleep(3)  # only here to let B finish quietly, avoiding the warning

asyncio.run(main())
```

If you want all sibling tasks to be **cancelled** the instant one fails — which is usually what people actually want — you need `asyncio.TaskGroup` (Chapter 8), not `gather()`. This is one of the main reasons `TaskGroup` is now the recommended default over `gather()` for new code.

### Collecting all results, errors included

If you'd rather get every result *and* every exception back as data, without any exception being raised, use `return_exceptions=True`:

```python
async def main():
    results = await asyncio.gather(
        flaky("A", 1, fail=True),
        flaky("B", 0.5),
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"Failed: {r}")
        else:
            print(f"Succeeded: {r}")

asyncio.run(main())
```

This waits for **all** of them to finish (success or failure) before returning, and hands you a list where each slot is either the real result or the exception object — you decide how to handle each one.

---

## 6.3 `asyncio.wait()`: Fine-Grained Control Over Partial Completion

```python
done, pending = await asyncio.wait(tasks, timeout=None, return_when=asyncio.ALL_COMPLETED)
```

`asyncio.wait()` doesn't raise exceptions and doesn't collect results for you — it just partitions your tasks into two **sets**: `done` and `pending`, once whatever condition you asked for (`return_when`) is met. You're then responsible for iterating `done` and calling `.result()` / `.exception()` yourself.

**Important:** `asyncio.wait()` requires actual `Task` (or `Future`) objects — passing bare coroutines directly was deprecated starting in Python 3.8 and has since been removed. Wrap coroutines in `asyncio.create_task()` first.

`return_when` accepts three values:

- `asyncio.ALL_COMPLETED` (default) — wait for everything
- `asyncio.FIRST_COMPLETED` — return as soon as *any one* finishes, leaving the rest in `pending`
- `asyncio.FIRST_EXCEPTION` — return as soon as any one raises (or once all complete successfully, if none do)

```python
import asyncio

async def worker(name, delay):
    await asyncio.sleep(delay)
    return name

async def main():
    tasks = [
        asyncio.create_task(worker("A", 3)),
        asyncio.create_task(worker("B", 1)),
        asyncio.create_task(worker("C", 2)),
    ]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for task in done:
        print(f"Finished first: {task.result()}")   # "B"

    print(f"Still pending: {len(pending)}")           # 2 (A and C)

    # wait() does NOT cancel pending tasks for you — that's your job if you want it:
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)  # let cancellation settle

asyncio.run(main())
```

This gives you the raw material to build patterns like "race several redundant requests and take whichever answers first, cancelling the losers" — useful for things like hedged requests to multiple mirrors/replicas.

---

## 6.4 `asyncio.as_completed()`: Streaming Results in Completion Order

```python
for coro in asyncio.as_completed(aws):
    result = await coro
```

Unlike `gather()`, `as_completed()` gives you results **as they finish** — first-done, first-yielded — which is ideal for progress reporting or "process each result the moment it's ready" pipelines, where you don't want to wait for the slowest one before doing anything with the fast ones.

```python
import asyncio
import time

async def fetch(name, delay):
    await asyncio.sleep(delay)
    return f"{name} done after {delay}s"

async def main():
    start = time.perf_counter()
    aws = [fetch("slow", 2), fetch("fast", 0.5), fetch("medium", 1)]

    for coro in asyncio.as_completed(aws):
        result = await coro
        elapsed = time.perf_counter() - start
        print(f"[{elapsed:.1f}s] {result}")
    # Output order: fast, then medium, then slow — completion order,
    # NOT the [slow, fast, medium] order they were listed in.

asyncio.run(main())
```

As of **Python 3.13**, the object returned by `as_completed()` works both as a plain iterator (as shown above) *and* as an async iterator, so `async for coro in asyncio.as_completed(aws):` also works, and — usefully — the coroutine you get back is tied to the original task/future you passed in, making it easier to know *which* input a given result corresponds to when you need that mapping.

If any awaited coroutine raises, that exception surfaces at the specific `await coro` call for that result — wrap each iteration in its own `try`/`except` if you want to keep processing the rest after one failure.

---

## 6.5 Choosing Between Them

- **Default choice for "run N things, get N results, in order":** `gather()` (or, better yet, `TaskGroup` from Chapter 8 if you also want automatic sibling cancellation on failure).
- **Need to react to whichever finishes first, or apply a timeout without cancelling everything automatically:** `wait()` with `FIRST_COMPLETED` or a `timeout=`.
- **Need to process/display results incrementally, in the order they actually complete:** `as_completed()`.

---

## 6.6 Hands-On Exercises

**Exercise 1 — Prove the ordering guarantee.**
Using `gather()`, launch three `fetch()` calls with delays `2`, `0.1`, `1` (in that order). Print the returned list and confirm the results come back in the `[2, 0.1, 1]` input order, not completion order.

**Exercise 2 — Reproduce the background-task gotcha.**
Recreate the §6.2 example where one of two gathered coroutines fails immediately and the other sleeps for several seconds. Catch the exception, then **without** the cleanup `await asyncio.sleep(...)` line, let `main()` return right away. Run it and see if you get a `Task exception was never retrieved`-style warning or an unfinished-task warning on exit. Then add proper cleanup and confirm the warning goes away.

**Exercise 3 — `return_exceptions=True` in practice.**
Write a list of 5 coroutines, 2 of which raise different exception types. Use `gather(..., return_exceptions=True)` and write code that separates the list into two lists: `successes` and `failures`, printing a summary count of each.

**Exercise 4 — Race and cancel.**
Using `asyncio.wait(..., return_when=FIRST_COMPLETED)`, launch 4 tasks simulating requests to 4 redundant mirrors with random delays between 0.5–3 seconds. Take the first result that completes, cancel the rest, and print which mirror "won."

**Exercise 5 — Streaming progress bar.**
Using `as_completed()`, launch 6 "downloads" with random delays. As each one completes, print `f"{n}/6 complete"` with an incrementing counter, so the user sees live progress rather than waiting for everything to finish before seeing any output.

---

## 6.7 Common Pitfalls

- **Assuming `gather()` returns results in completion order.** It doesn't — always input order. Use `as_completed()` if you need completion order.
- **Assuming `gather()` cancels siblings when one fails.** By default it does not; failed-but-still-running siblings keep executing in the background unless you handle them explicitly, or switch to `TaskGroup`.
- **Passing bare coroutines to `asyncio.wait()`.** This was deprecated starting in Python 3.8 and has since been removed — always wrap with `asyncio.create_task()` first.
- **Forgetting `wait()` doesn't cancel `pending` tasks for you.** If you only wanted the first result, you're responsible for cancelling (and awaiting the cancellation of) whatever's left in `pending`.

---

## 6.8 Further Reading

- [Python docs: `asyncio.gather()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
- [Python docs: `asyncio.wait()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.wait)
- [Python docs: `asyncio.as_completed()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.as_completed)

---

**Next: Chapter 8 — Structured Concurrency with TaskGroups** (the modern, safer alternative to `gather()` that fixes the sibling-cancellation gap you just saw in §6.2).
