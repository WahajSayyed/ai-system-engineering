# Chapter 8: Structured Concurrency with TaskGroups

> Part 2 — Core AsyncIO Building Blocks · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 8.1 The Problem: Unstructured Concurrency

Across the last two chapters, two gaps kept surfacing:

- **Chapter 5:** a fire-and-forget `create_task()` can be silently garbage-collected if you don't keep a reference to it.
- **Chapter 7:** `gather()`'s default behavior lets sibling tasks **keep running in the background**, unmanaged, after one of them fails.

Both problems share a root cause: nothing enforces that concurrent work started inside a block of code actually finishes — or gets cleaned up — before that block of code considers itself "done." This is the core idea behind **structured concurrency**: every concurrent task should have a clear, bounded lifetime tied to the scope that created it. No task should be able to outlive the code block that spawned it, running invisibly in the background with nobody watching.

**Python 3.11** introduced `asyncio.TaskGroup` to enforce exactly this, and it's now the **recommended default** over bare `gather()` for most new code that fans out multiple tasks as a single logical unit.

---

## 8.2 Basic Usage

```python
import asyncio

async def fetch(name, delay):
    await asyncio.sleep(delay)
    return f"{name} done"

async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch("A", 1))
        task2 = tg.create_task(fetch("B", 2))
    # Execution only reaches here once BOTH tasks have finished.
    print(task1.result())
    print(task2.result())

asyncio.run(main())
```

The `async with asyncio.TaskGroup() as tg:` block **will not exit** until every task created with `tg.create_task()` inside it has finished — successfully, or otherwise. There's no need to separately `await` each task, and no need to manually keep a set of references the way we did with fire-and-forget tasks in Chapter 5 — the group itself holds strong references to everything created within it for as long as the block is open.

---

## 8.3 What Happens on Failure: Automatic Sibling Cancellation

This is the headline improvement over `gather()`. If **any** task inside a `TaskGroup` raises an exception (other than `CancelledError`), the group immediately:

1. **Cancels every other still-running task in the group.**
2. Waits for all of them to actually finish unwinding (respecting their `finally` blocks, cleanup code, etc.).
3. Raises once everything has settled — collecting every non-cancellation exception that occurred into a single `ExceptionGroup` (PEP 654), even if only one task actually failed.

```python
import asyncio

async def flaky(name, delay, fail=False):
    try:
        await asyncio.sleep(delay)
        if fail:
            raise ValueError(f"{name} failed")
        return name
    except asyncio.CancelledError:
        print(f"{name} was cancelled cleanly")
        raise  # always re-raise CancelledError after cleanup

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(flaky("A", 1, fail=True))
            tg.create_task(flaky("B", 3))   # this one gets cancelled once A fails
    except* ValueError as eg:
        for exc in eg.exceptions:
            print(f"Handled: {exc}")

asyncio.run(main())
```

Compare this directly to the `gather()` version of the same scenario from Chapter 7: there, "B" kept sleeping in the background, unmanaged, for its full 3 seconds even after "A"'s exception had already propagated. Here, the moment "A" fails, "B" is actively **cancelled** — you'll see `"B was cancelled cleanly"` print almost immediately, not after a 3-second wait.

---

## 8.4 Handling Exceptions: `except*` and `ExceptionGroup`

Because more than one task in a group can fail simultaneously, a `TaskGroup` doesn't just re-raise "the" exception the way `gather()`'s default mode does — it always raises an **`ExceptionGroup`** (or `BaseExceptionGroup`), a container that can hold **multiple** exceptions at once, even if they're of different types.

You handle this with **`except*`** (PEP 654, Python 3.11+), which filters by exception type *within* the group, potentially handling several categories separately in the same `try`:

```python
import asyncio

async def maybe_fail(name, exc_type=None):
    await asyncio.sleep(0.1)
    if exc_type:
        raise exc_type(f"{name} failed")
    return name

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(maybe_fail("A", ValueError))
            tg.create_task(maybe_fail("B", TypeError))
            tg.create_task(maybe_fail("C"))  # succeeds
    except* ValueError as eg:
        print(f"ValueErrors: {[str(e) for e in eg.exceptions]}")
    except* TypeError as eg:
        print(f"TypeErrors: {[str(e) for e in eg.exceptions]}")

asyncio.run(main())
```

Both `except*` clauses run here — one handling the `ValueError`, one handling the `TypeError` — because both were present in the same `ExceptionGroup`. This is different from ordinary `try`/`except`, which stops at the first matching clause; `except*` clauses are **not mutually exclusive** when the group contains a mix of exception types.

**A subtlety worth remembering:** even a single failure still comes wrapped in an `ExceptionGroup` of one. A plain `except ValueError:` will **not** catch it directly — you need `except* ValueError:`, or you need to unwrap `eg.exceptions` yourself.

---

## 8.5 Failure Inside the Block Itself

The same cancel-and-collect behavior applies if the exception originates in the `async with` block's own body, not just from a child task:

```python
async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(flaky("A", 5))
            await asyncio.sleep(1)
            raise RuntimeError("something in the block body went wrong")
            # "A" gets cancelled even though it never raised anything itself
    except* RuntimeError as eg:
        print(f"Caught: {eg.exceptions}")
```

`TaskGroup` also carefully handles **nested groups**: if an inner `TaskGroup` and its enclosing outer `TaskGroup` both encounter failures around the same time, the inner group resolves its own `ExceptionGroup` first; the outer group then separately receives its own cancellation and processes its own exceptions — they don't get tangled into a single flat mess.

---

## 8.6 Updated Decision Table

With `TaskGroup` in the picture, here's the full comparison, extending Chapter 7's table:

| Tool | Sibling cancellation on failure | Exception style | Best for |
|---|---|---|---|
| `asyncio.gather()` | No (by default) | First exception raised directly | Quick one-off fan-out where you don't need automatic cleanup on failure, or where `return_exceptions=True` suits you |
| `asyncio.wait()` | No (manual) | None raised — inspect `.exception()` yourself | Fine-grained partial-completion / racing patterns |
| `asyncio.as_completed()` | No | Raised per-item as you iterate | Streaming/progressive result processing |
| **`asyncio.TaskGroup`** | **Yes, automatic** | `ExceptionGroup`, handled with `except*` | **Default choice** for "these tasks are one unit of work" |

For new application code, prefer `TaskGroup` whenever a set of concurrent tasks conceptually belongs together and a failure in one should mean the whole unit gives up. Reach for `gather()` mainly when you deliberately want the more permissive "let everything finish regardless" behavior (or explicitly via `return_exceptions=True`), and `wait()`/`as_completed()` for the specific partial-completion or streaming needs they're suited to.

---

## 8.7 Hands-On Exercises

**Exercise 1 — Convert to TaskGroup.**
Take your Chapter 5, Exercise 1 solution (four concurrent `fetch()` calls via manual `create_task()`) and rewrite it using `asyncio.TaskGroup`. Confirm the timing is unchanged (~1 second total) and note how much manual bookkeeping (task list management) you got to delete.

**Exercise 2 — Watch the cancellation happen.**
Recreate the §8.3 `flaky()` example. Add a `print(f"{name}: still running at {i}s")` loop inside "B" that prints every 0.5 seconds while it "works," so you can visually confirm exactly when it gets cancelled relative to when "A" fails — rather than running to completion the way it did under `gather()` in Chapter 7.

**Exercise 3 — Multi-exception handling.**
Recreate the §8.4 example with three tasks raising `ValueError`, `TypeError`, and `KeyError` respectively (plus one that succeeds). Write `except*` clauses for all three exception types, and print a final summary: how many total tasks succeeded vs. failed, and which exception types were seen.

**Exercise 4 — Fail from the block body.**
Reproduce the §8.5 example: start a long-running child task, then raise an exception directly in the `async with` body (not from a task). Confirm the child task gets cancelled and its cancellation is visible (add a `try/except asyncio.CancelledError` with a print inside it).

**Exercise 5 (stretch) — Nested TaskGroups.**
Write an outer `TaskGroup` containing a task that itself opens an **inner** `TaskGroup` with two more tasks, one of which fails. Add print statements at each stage (inner group entering/exiting, outer group entering/exiting) to trace exactly how the inner group's `ExceptionGroup` gets handled before the outer group notices anything.

---

## 8.8 Common Pitfalls

- **Using a plain `except ValueError:` instead of `except* ValueError:`.** `TaskGroup` always raises an `ExceptionGroup`/`BaseExceptionGroup`, even for a single failure — ordinary `except` clauses won't match it.
- **Expecting `TaskGroup` to behave like permissive `gather()`.** The whole point of `TaskGroup` is that it does **not** let siblings keep running after a failure — if you genuinely want that more permissive behavior, use `gather()` (with `return_exceptions=True` if you also want to inspect failures without raising).
- **Not being on Python 3.11+.** `TaskGroup` and `except*` are both new in 3.11 — if you're stuck on an older version, you'll need `gather()` with manual cancellation logic, or a backport library.
- **Forgetting to re-raise `CancelledError` after cleanup inside a cancelled task.** Swallowing it silently (no bare `except asyncio.CancelledError: pass` without a `raise`) breaks the cancellation contract — we'll cover exactly why in Chapter 9.

---

## 8.9 Further Reading

- [Python docs: `asyncio.TaskGroup`](https://docs.python.org/3/library/asyncio-task.html#task-groups)
- [PEP 654 – Exception Groups and `except*`](https://peps.python.org/pep-0654/)
- [Python docs: Exception Groups](https://docs.python.org/3/library/exceptions.html#exception-groups)

---

**Next: Chapter 9 — Cancellation Deep Dive** (`Task.cancel()`, exactly how `CancelledError` gets thrown into a coroutine, `asyncio.shield`, and the cleanup patterns `TaskGroup` relies on internally).
