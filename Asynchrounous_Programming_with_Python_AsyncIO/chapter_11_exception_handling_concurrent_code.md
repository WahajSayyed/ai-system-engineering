# Chapter 11: Exception Handling in Concurrent Code

> Part 2 — Core AsyncIO Building Blocks · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 11.1 Why Concurrent Error Handling Needs Its Own Chapter

In sequential code, an unhandled exception stops the program (or the current function call stack) immediately and loudly. In concurrent code, several independent things can be failing "at once," and the tools from Chapters 6–10 each make a different choice about what that means for you:

| Situation | What you get |
|---|---|
| `gather()`, default | First exception raised to you; siblings keep running unmanaged |
| `gather(return_exceptions=True)` | Nothing raised; every result/exception returned as data |
| `wait()` | Nothing raised; you inspect `.exception()` on each task yourself |
| `TaskGroup` | All non-cancellation exceptions collected into one `ExceptionGroup`, siblings cancelled |
| A bare `create_task()`, never awaited | **Nothing visible at all**, unless you go looking |

That last row is the most dangerous one, and where this chapter starts.

---

## 11.2 The Silent Failure: Swallowed Exceptions in Fire-and-Forget Tasks

If you `create_task()` a coroutine and never `await` it, never check `.result()`, and never attach a callback — and it raises — **nothing happens that you'll notice in normal program flow.** The exception doesn't crash your program, doesn't print to your console mid-run, and doesn't stop anything else. It just sits on the (now-finished) `Task` object.

```python
import asyncio

async def background_job():
    await asyncio.sleep(1)
    raise ValueError("something went wrong in the background")

async def main():
    asyncio.create_task(background_job())  # fire-and-forget — nobody's watching
    await asyncio.sleep(2)
    print("main() finished, seemingly without incident")

asyncio.run(main())
```

Run this, and `"main() finished, seemingly without incident"` genuinely does print — but eventually, once that `Task` object is garbage collected (often at process exit, sometimes earlier), asyncio's **default exception handler** logs something like:

```
Task exception was never retrieved
future: <Task finished name='Task-1' coro=<background_job() done, defined at ...> exception=ValueError('something went wrong in the background')>
Traceback (most recent call last):
  ...
ValueError: something went wrong in the background
```

This is logged, not raised — meaning if you don't have logging configured to actually surface it (or you're not watching stderr), **it disappears entirely.** This is one of the most common ways real asyncio applications silently lose errors: background tasks that fail without anyone structurally checking on them.

### Fixing it: always account for fire-and-forget tasks

**Option 1 — a done-callback that checks for exceptions:**

```python
def _log_task_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        print(f"Background task {task.get_name()} failed: {exc!r}")

async def main():
    task = asyncio.create_task(background_job(), name="background_job")
    task.add_done_callback(_log_task_exception)
    await asyncio.sleep(2)

asyncio.run(main())
```

**Option 2 — prefer `TaskGroup` (Chapter 8) whenever the task's lifetime can reasonably be scoped.** It structurally prevents this entire failure mode, since it won't let its `async with` block exit while children are still unaccounted for, and any failure is guaranteed to surface as an `ExceptionGroup`.

Fire-and-forget tasks with no natural enclosing scope (e.g., a genuinely long-lived background job started once at application startup) are the one case where neither `TaskGroup` nor simple `await` fits — for those, the done-callback pattern (or the custom exception handler below) is the right tool.

---

## 11.3 Customizing the Loop's Exception Handler

For applications with many background tasks — worker pools, long-running services — checking each task individually gets tedious. `asyncio` lets you install a **loop-wide exception handler** that catches anything the default handler would otherwise just log:

```python
import asyncio
import logging

def custom_exception_handler(loop, context):
    exception = context.get("exception")
    message = context.get("message")
    task = context.get("task")
    logging.error(
        "Unhandled exception in task %s: %s (%s)",
        task.get_name() if task else "unknown",
        message,
        exception,
    )
    # In a real application: also send to your error-tracking service here,
    # e.g. Sentry, Datadog, or an internal alerting pipeline.

async def background_job():
    await asyncio.sleep(1)
    raise ValueError("boom")

async def main():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(custom_exception_handler)

    asyncio.create_task(background_job(), name="job-1")
    await asyncio.sleep(2)

asyncio.run(main())
```

This is the closest thing asyncio has to a global "catch anything that fell through the cracks" safety net — genuinely worth setting up in any production service with fire-and-forget or long-lived background tasks, so failures land in your logs/metrics/alerting instead of silently vanishing.

---

## 11.4 A Resilient Fan-Out Helper with `gather(return_exceptions=True)`

A common, reusable pattern: fan out several independent operations, where you want to proceed with whatever succeeded and separately handle/log whatever failed, rather than letting one failure abort everything.

```python
import asyncio

async def resilient_gather(*coros):
    """Run all coroutines concurrently. Return (successes, failures) separately."""
    results = await asyncio.gather(*coros, return_exceptions=True)
    successes = [r for r in results if not isinstance(r, BaseException)]
    failures = [r for r in results if isinstance(r, BaseException)]
    return successes, failures

async def fetch(name, fail=False):
    await asyncio.sleep(0.2)
    if fail:
        raise RuntimeError(f"{name} failed")
    return name

async def main():
    successes, failures = await resilient_gather(
        fetch("A"),
        fetch("B", fail=True),
        fetch("C"),
        fetch("D", fail=True),
    )
    print(f"Succeeded: {successes}")
    for f in failures:
        print(f"Failed: {f}")

asyncio.run(main())
```

This is a genuinely useful production pattern: e.g., fetching data from 5 independent recommendation services for a web page, where you'd rather render with 4 successful results and log the 1 failure than show an error page because one non-critical service hiccuped.

---

## 11.5 `except*` in Practice: Handling Different Failure Categories Differently

Real systems usually distinguish between error categories that deserve different handling — for example, a "retryable" transient failure versus a "fatal" one that should abort everything immediately. `except*` (Chapter 8) is built for exactly this:

```python
import asyncio

class RetryableError(Exception):
    pass

class FatalError(Exception):
    pass

async def call_service(name, error_type=None):
    await asyncio.sleep(0.1)
    if error_type:
        raise error_type(f"{name} hit a problem")
    return name

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(call_service("A", RetryableError))
            tg.create_task(call_service("B", FatalError))
            tg.create_task(call_service("C"))
    except* FatalError as eg:
        for exc in eg.exceptions:
            print(f"FATAL, aborting: {exc}")
        raise  # a fatal error should still propagate up after logging
    except* RetryableError as eg:
        for exc in eg.exceptions:
            print(f"Would retry: {exc}")

asyncio.run(main())
```

Both `except*` clauses can fire in the same `try` (they're independent filters over the group's contents, not mutually exclusive branches), which is exactly the point — you can react differently to each category present in a single failure event, rather than treating "something failed" as one undifferentiated blob.

### Finer-grained filtering with `.split()` and `.subgroup()`

`except*` filters by exception *type*. If you need to filter by some other **condition** — e.g., "only errors with a specific error code attribute" — `ExceptionGroup` provides `.split(predicate)` and `.subgroup(predicate)`:

```python
try:
    async with asyncio.TaskGroup() as tg:
        ...
except* Exception as eg:
    retryable, rest = eg.split(lambda e: getattr(e, "retryable", False))
    if retryable:
        for exc in retryable.exceptions:
            print(f"Retryable: {exc}")
    if rest:
        raise rest  # re-raise only the non-retryable subset, preserving grouping
```

`.subgroup(predicate)` returns just the matching portion (or `None` if nothing matches); `.split(predicate)` returns both the matching and non-matching portions as a tuple, letting you selectively re-raise only what still needs to propagate.

---

## 11.6 Hands-On Exercises

**Exercise 1 — Watch a fire-and-forget failure disappear.**
Recreate the §11.2 example exactly. Run it and confirm `"main() finished, seemingly without incident"` prints before you see the `"Task exception was never retrieved"` warning (if you see it at all before the process exits — try adding `import gc; gc.collect()` at the end of `main()` to force it to surface).

**Exercise 2 — Fix it with a done-callback.**
Add the `_log_task_exception` done-callback pattern from §11.2 to your Exercise 1 code. Confirm the failure is now reported immediately when the task finishes, rather than silently at some later garbage-collection point.

**Exercise 3 — Install a custom loop exception handler.**
Recreate the §11.3 example. Trigger it with two different fire-and-forget failing tasks (no individual done-callbacks this time) and confirm both get caught by your global handler with their correct task names.

**Exercise 4 — Build the resilient fan-out helper.**
Implement `resilient_gather()` from §11.4 yourself. Test it with a mix of 5 coroutines where 2 fail with different exception types, and print a clean summary: `"3 succeeded, 2 failed"`, followed by the failure details.

**Exercise 5 — Differentiated `except*` handling.**
Recreate the §11.5 `RetryableError`/`FatalError` example. Then extend it: when a `FatalError` occurs, in addition to re-raising, also make sure any `RetryableError`s that happened in the *same* failure event still get logged first (both `except*` clauses should still run before the fatal one re-raises) — confirm the order of printed output makes sense to you.

**Exercise 6 (stretch) — Custom predicate filtering with `.split()`.**
Create 4 custom exceptions, 2 tagged with a `.retryable = True` attribute and 2 without. Trigger all 4 inside a `TaskGroup`, catch the resulting group with `except* Exception`, and use `.split()` to separate them by the `.retryable` attribute rather than by type. Re-raise only the non-retryable subset.

---

## 11.7 Common Pitfalls

- **Assuming a failing fire-and-forget task will make itself known.** It won't, beyond a log message you might never see — always attach a done-callback, use `TaskGroup`, or install a custom exception handler for anything long-lived.
- **Not configuring logging at all.** Even asyncio's default exception handler relies on Python's `logging` module being set up to actually surface output somewhere you'll look — an unconfigured logger can make even "handled" warnings invisible.
- **Reaching for `except* Exception` to catch everything and treating all failures identically.** This throws away exactly the category information (`RetryableError` vs. `FatalError`, etc.) that's usually the whole point of raising distinct exception types in the first place.
- **Leaving tasks unnamed.** `task.get_name()` defaults to a generic `Task-N` — in any application with more than a couple of concurrent tasks, descriptive names (`asyncio.create_task(coro, name="fetch-user-orders")`) make concurrent failure logs vastly easier to debug.

---

## 11.8 Further Reading

- [Python docs: `loop.set_exception_handler()`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.set_exception_handler)
- [Python docs: Exception Groups — `split()` and `subgroup()`](https://docs.python.org/3/library/exceptions.html#exception-groups)
- [PEP 654 – Exception Groups and `except*`](https://peps.python.org/pep-0654/)

---

**Next: Chapter 12 — Synchronization Primitives** (`Lock`, `Event`, `Condition`, `Semaphore` — when cooperative single-threaded code still needs coordination between coroutines).
