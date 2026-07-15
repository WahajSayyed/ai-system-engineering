# Chapter 5: Tasks — Scheduling Concurrent Work

> Part 2 — Core AsyncIO Building Blocks · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 5.1 From Coroutine to Task

In Chapter 4, we saw that `await coro()` runs a coroutine **inline** — sequentially, with no overlap with anything else. To get the real concurrency this whole course is about, you need `asyncio.Task`.

```python
task = asyncio.create_task(coro())
```

`asyncio.create_task()` does something fundamentally different from `await`: it wraps the coroutine in a `Task` object and **schedules it to start running independently**, without you awaiting it yet. The task begins executing on its own — interleaved with everything else — the moment the event loop gets a chance to run it (specifically, on the loop's *next* iteration; more on this "lazy start" timing in §5.4).

```python
import asyncio
import time

async def fetch(name, delay):
    print(f"{name}: starting")
    await asyncio.sleep(delay)
    print(f"{name}: done")
    return name

async def main():
    start = time.perf_counter()

    # Creating the tasks schedules all three to start running immediately —
    # we haven't awaited any of them yet.
    task_a = asyncio.create_task(fetch("A", 1))
    task_b = asyncio.create_task(fetch("B", 1))
    task_c = asyncio.create_task(fetch("C", 1))

    # Now we wait for each to finish.
    result_a = await task_a
    result_b = await task_b
    result_c = await task_c

    elapsed = time.perf_counter() - start
    print(f"Total: {elapsed:.2f}s")  # ~1.0s, not ~3.0s!

asyncio.run(main())
```

Notice the output: **all three "starting" messages print before any "done" message** — proof that all three are genuinely in flight at once. Compare this directly to Chapter 4's sequential-`await` example, which took ~2 seconds for two 1-second fetches; here, three 1-second fetches take only ~1 second total. This is the concurrency payoff the whole course has been building toward.

---

## 5.2 The Fire-and-Forget Gotcha: Keep Your References

`asyncio`'s official documentation carries a specific, easy-to-miss warning: **the event loop only holds a *weak* reference to tasks it's running.** If you create a task and don't keep a strong reference to it anywhere in your own code, Python's garbage collector is free to destroy it mid-execution — silently.

```python
async def background_worker():
    await asyncio.sleep(2)
    print("This might never print!")

async def main():
    # DANGEROUS: no reference to this task is kept anywhere
    asyncio.create_task(background_worker())
    await asyncio.sleep(3)  # meanwhile, the task above may get garbage-collected

asyncio.run(main())
```

In short scripts this often "happens to work" because nothing triggers garbage collection in time — which makes it a nasty bug to discover, since it can pass testing and then intermittently fail in production. The fix is simple: **keep a reference**, typically in a set, and discard it once done:

```python
background_tasks: set[asyncio.Task] = set()

async def main():
    task = asyncio.create_task(background_worker())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    await asyncio.sleep(3)

asyncio.run(main())
```

In Chapter 8 we'll see `asyncio.TaskGroup`, which solves this problem structurally (a task group holds strong references to everything created within it, and won't let the block exit until all tasks are accounted for) — for most application code, prefer `TaskGroup` over bare `create_task` fire-and-forget calls specifically to sidestep this entire class of bug.

---

## 5.3 Task Lifecycle and Introspection

A `Task` moves through a small set of states, and exposes methods to inspect them:

| Method | Meaning |
|---|---|
| `task.done()` | `True` once the task has finished — successfully, with an exception, or via cancellation |
| `task.result()` | The coroutine's return value. Raises the task's exception if it failed. Raises `asyncio.InvalidStateError` if called before the task is done. |
| `task.exception()` | The exception the task raised, or `None` if it completed normally. Also raises `InvalidStateError` if not yet done. |
| `task.cancelled()` | `True` if the task was cancelled |
| `task.get_name()` / `task.set_name(name)` | A human-readable label, shown in `repr(task)` and useful for debugging/logging (especially with the 3.14 `python -m asyncio pstree` tool from Chapter 2!) |

```python
async def flaky():
    await asyncio.sleep(0.5)
    raise ValueError("oops")

async def main():
    task = asyncio.create_task(flaky(), name="flaky-task")
    print(task.get_name())    # "flaky-task"
    print(task.done())        # False — hasn't run yet

    await asyncio.sleep(1)    # give it time to finish
    print(task.done())        # True
    print(task.exception())   # ValueError('oops')

    try:
        task.result()          # re-raises the ValueError
    except ValueError as e:
        print(f"Task failed with: {e}")

asyncio.run(main())
```

---

## 5.4 Lazy vs. Eager Task Start (Python 3.12+)

By default, `asyncio.create_task()` is **lazy**: calling it schedules the task, but the coroutine's body doesn't actually start executing until the event loop gets to it — typically the very next iteration, not the exact line `create_task()` was called on. This is a small delay, but it's a real one, and in performance-sensitive code with many short-lived tasks, that per-task scheduling overhead adds up.

**Python 3.12** added an **eager task factory**: `asyncio.eager_task_factory`, plus an `eager_start=True` keyword argument on task creation. With eager execution:

- The coroutine starts running **synchronously, immediately**, at the moment `create_task()` is called — up to its first `await`.
- If the coroutine finishes (returns or raises) *before* hitting any `await`, it never touches the event loop's scheduling machinery at all — this is the performance win, and it particularly benefits coroutines that often complete without real I/O (e.g., cache-hit lookups).
- If it does hit an `await`, it's scheduled onto the loop normally from that point forward, just like a lazy task.

```python
import asyncio

cache = {"a": 1, "b": 2}

async def cached_lookup(key):
    if key in cache:
        return cache[key]           # completes instantly, no await at all
    await asyncio.sleep(1)          # simulate a slow lookup
    return 42

async def main():
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)

    t1 = asyncio.create_task(cached_lookup("a"))  # eager: runs to completion immediately
    t2 = asyncio.create_task(cached_lookup("z"))  # blocks, scheduled onto the loop normally

    print("t1 already done?", t1.done())  # True — it never needed to suspend
    print("t2 already done?", t2.done())  # False — still waiting on sleep(1)

    print(await t1, await t2)

asyncio.run(main())
```

Real benchmarks from the CPython team have shown eager execution making certain task-heavy workloads noticeably faster — in some cases several times faster for coroutines that mostly complete synchronously.

**The catch:** eager execution is a genuine *semantic* change, not just a speed optimization, and it can surprise you:

- **Side effects run earlier than expected.** Any code before the first `await` in an eagerly-started coroutine executes *at the point `create_task()` is called*, not later — so logging, state mutation, or other side effects can happen before code that runs immediately after `create_task()` in the caller.
- **Execution order changes.** Code that implicitly relied on tasks not starting until the caller yields control can behave differently.
- **Deep eager-task chains without any `await` can hit Python's recursion limit**, since each eager task's synchronous portion runs on the current call stack rather than being handed off to the loop.

Because of these subtleties, most current guidance — including from the asyncio maintainers — is: **stick with the default lazy behavior for ordinary application code**, and reach for `eager_start=True` / `eager_task_factory` only in narrow, measured, performance-critical hot paths, resetting the factory afterward if you touch it at all.

```python
# Scoped, deliberate use — not a global default
loop = asyncio.get_running_loop()
original_factory = loop.get_task_factory()
loop.set_task_factory(asyncio.eager_task_factory)
try:
    ...  # performance-critical section
finally:
    loop.set_task_factory(original_factory)  # always restore it
```

---

## 5.5 Hands-On Exercises

**Exercise 1 — Convert sequential to concurrent.**
Take your Chapter 4, Exercise 3 code (four sequential `await fetch(name, 1)` calls) and rewrite it to use `asyncio.create_task()` for all four before awaiting any of them. Confirm the total time drops from ~4 seconds to ~1 second.

**Exercise 2 — Reproduce the fire-and-forget gotcha.**
Write a `background_worker()` coroutine that sleeps for 2 seconds and then prints a message. In `main()`, call `asyncio.create_task(background_worker())` **without** keeping a reference, immediately follow it with `import gc; gc.collect()`, then `await asyncio.sleep(3)`. Note whether the message printed. Then fix it using the strong-reference pattern from §5.2 and confirm the message reliably prints. (Behavior here can vary run to run — the point of the exercise is understanding *why* the bug is possible, not guaranteeing you reproduce it every single time.)

**Exercise 3 — Task introspection.**
Create a task from a coroutine that raises an exception partway through. Before it's done, try calling `.result()` and confirm you get `asyncio.InvalidStateError`. After it's done, call `.result()` again and confirm the original exception is re-raised. Print `.get_name()` and `repr(task)` at each stage.

**Exercise 4 — Eager vs. lazy, observed.**
Using the `cached_lookup` example from §5.4, add a `print("about to create tasks")` line immediately before the two `create_task()` calls and a `print("tasks created")` line immediately after. Run it once with the default (lazy) factory and once with `asyncio.eager_task_factory` set, and compare exactly where `"Non-blocking setup"`-style side-effect prints land relative to `"tasks created"`.

**Exercise 5 (stretch) — Find the recursion limit.**
Write a coroutine that, when run eagerly, immediately (with no `await` first) creates another eager task running itself with a decremented counter, bottoming out at 0. Set the eager task factory and run it with a large starting counter (try increasing values). At what depth do you hit `RecursionError`? Explain why lazy tasks don't have this problem.

---

## 5.6 Common Pitfalls

- **Not keeping a reference to a "fire and forget" task.** The event loop's own reference is weak; without your own strong reference, garbage collection can silently kill a running task. Prefer `TaskGroup` (Chapter 8) or the discard-on-done-callback pattern.
- **Calling `.result()` before checking `.done()`.** This raises `InvalidStateError` — always either `await` the task (which blocks until it's done) or check `.done()` first if you need a non-blocking peek.
- **Assuming `create_task()` starts the coroutine synchronously by default.** It doesn't — the default is lazy, starting on the *next* loop iteration. Only `eager_start=True` (3.12+) changes this, and it comes with real behavioral trade-offs, not just a speed boost.

---

## 5.7 Further Reading

- [Python docs: `asyncio.Task`](https://docs.python.org/3/library/asyncio-task.html#task-object)
- [Python docs: Eager Task Factory](https://docs.python.org/3/library/asyncio-task.html#asyncio.eager_task_factory)
- [Python discuss.python.org: "Make asyncio eager task factory default"](https://discuss.python.org/t/make-asyncio-eager-task-factory-default/75164) — a good look at the trade-offs from the people who maintain asyncio

---

**Next: Chapter 6 — Running Things Concurrently: gather, wait, as_completed** (the higher-level tools for fanning out and collecting results from many tasks at once, and how their error-handling behavior differs).
