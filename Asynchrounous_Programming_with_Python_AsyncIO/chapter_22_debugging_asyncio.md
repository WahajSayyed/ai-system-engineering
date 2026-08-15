# Chapter 22: Debugging AsyncIO Applications

> Part 5 — Internals and Production Concerns · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 22.1 Turning On Debug Mode

asyncio has a built-in debug mode that surfaces exactly the kinds of bugs this course has warned about throughout — blocking the loop, losing exceptions, thread-safety violations — as actual runtime warnings, rather than silent, hard-to-trace symptoms. Three ways to enable it:

```python
# 1. Directly on asyncio.run()
asyncio.run(main(), debug=True)

# 2. Via an environment variable (useful for enabling it without code changes)
# PYTHONASYNCIODEBUG=1 python my_script.py

# 3. Programmatically on a running loop
loop = asyncio.get_running_loop()
loop.set_debug(True)
```

Debug mode does several things at once: it logs a warning whenever a callback or task takes suspiciously long to run (§22.2), adds much more detail to "exception was never retrieved" warnings (§22.3), and checks that certain non-thread-safe APIs are only called from the correct thread. **It also adds real overhead** — extra bookkeeping like capturing source tracebacks for every `Task` — so it's a development/troubleshooting tool, not something to leave enabled in production.

---

## 22.2 Slow Callback Warnings: Catching Blocking Calls

This is debug mode's single most useful feature for the most common category of asyncio bug from this entire course: a blocking call hiding inside a coroutine (Chapter 3, Chapter 18).

```python
import asyncio
import time

async def sneaky_blocker():
    time.sleep(0.5)   # a blocking call, hiding where you might not expect it

async def main():
    await sneaky_blocker()

asyncio.run(main(), debug=True)
```

With debug mode on, once this callback's execution exceeds a threshold (`loop.slow_callback_duration`, defaulting to **0.1 seconds**), asyncio logs something like:

```
Executing <Task finished name='Task-1' coro=<main() done...> took 0.501 seconds
```

This is your primary, built-in tool for answering **"something in my program is blocking the loop — but what, exactly?"** without manually auditing every coroutine by hand. You can tune the sensitivity:

```python
loop = asyncio.get_running_loop()
loop.slow_callback_duration = 0.05   # flag anything over 50ms instead of the 100ms default
```

Set it too low, though, and you'll drown in false-positive noise from perfectly reasonable operations that happen to take a bit longer than your threshold — tune deliberately, and expect to adjust it per-application.

---

## 22.3 Never-Retrieved Exceptions, Revisited

Chapter 11 introduced the "Task exception was never retrieved" warning for fire-and-forget tasks that fail silently. Debug mode makes these warnings considerably more useful by attaching a full **source traceback** — showing not just where the exception was raised, but where the task was originally *created* — which is often the more useful piece of information when hunting down which fire-and-forget call is the culprit in a large codebase.

```python
import asyncio

async def background_job():
    await asyncio.sleep(0.1)
    raise ValueError("boom")

async def main():
    asyncio.create_task(background_job())   # fire-and-forget, no reference kept
    await asyncio.sleep(1)

asyncio.run(main(), debug=True)
```

Compare the traceback detail here to running the same script without `debug=True` — debug mode's version will typically include exactly where `create_task()` was called from, not just where the exception itself occurred.

---

## 22.4 A Field Guide: Anti-Patterns From This Entire Course

Since debugging is fundamentally about recognizing familiar failure shapes, here's a consolidated checklist of the bugs this course has covered, in the rough order you're likely to hit them:

| Symptom | Likely cause | Chapter |
|---|---|---|
| Program seems to "hang" or run much slower than expected | A blocking call inside a coroutine, freezing the whole loop | 3, 18 |
| Code you expected to run silently didn't | Forgot to `await` a coroutine call | 4 |
| Concurrent operations taking as long as sequential ones | Mistook sequential `await` calls for concurrency | 4 |
| A background task's failure never surfaced anywhere | Fire-and-forget `create_task()` with no reference/callback | 5, 11 |
| One failure didn't stop unrelated work that "should" have stopped too | Used `gather()`'s permissive default instead of `TaskGroup` | 7, 8 |
| A task appears "stuck," ignoring `.cancel()` | Caught `CancelledError` (or bare `except:`) without re-raising | 9 |
| Memory grows unboundedly under sustained load | An unbounded `Queue`, or missing `writer.drain()` | 13, 16 |
| `queue.join()` never returns | A worker forgot to call `task_done()` | 13 |
| A subprocess or thread outlived the task that seemed to own it | Cancellation doesn't kill OS processes/threads by default | 17, 18 |
| ORM relationship access raises a strange greenlet-related error | Missing eager loading in the async ORM | 20 |

Before diving into a debugger, it's often faster to simply scan this list — most asyncio bugs really do fall into one of these familiar shapes.

---

## 22.5 The 3.14 Introspection CLI, Finally Put to Use

Back in Chapter 2, we mentioned Python 3.14 added a zero-instrumentation way to inspect a **live**, running program's task tree: `python -m asyncio ps <PID>` and `python -m asyncio pstree <PID>`. This is the tool to reach for specifically when a long-running program seems stuck and you want to know *what it's currently awaiting*, without attaching a full debugger.

```python
# hanging_example.py
import asyncio

async def worker(name, seconds):
    await asyncio.sleep(seconds)
    print(f"{name} done")

async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(worker("fast", 3), name="fast-worker")
        tg.create_task(worker("slow", 60), name="slow-worker")

asyncio.run(main())
```

Run this script, find its process ID (`ps aux | grep hanging_example` on POSIX, Task Manager's "Details" tab on Windows), and — while it's still running — in a **second terminal**:

```
python -m asyncio pstree <PID>
```

You'll get a live tree view showing each task by name, its current state, and exactly what it's suspended on — instantly telling you, for example, that `"slow-worker"` is still sleeping while `"fast-worker"` already finished, without needing to add a single `print()` statement or attach `pdb`. `python -m asyncio ps <PID>` gives a flatter, simpler listing if you don't need the tree structure. This is exactly why Chapter 5's advice to always give your tasks descriptive `name=` values pays off — an introspection tree full of `Task-47`, `Task-48`, `Task-49` is far less useful than one showing `fetch-user-orders`, `slow-worker`, `db-write`.

---

## 22.6 Don't Forget: Logging Has to Be Configured

Echoing Chapter 11's lesson one more time, because it's easy to forget in the middle of a debugging session: asyncio's warnings — slow callbacks, never-retrieved exceptions, everything in this chapter — go through Python's standard `logging` module, under a logger named `"asyncio"`. If your application hasn't configured logging at all (no `logging.basicConfig()`, no handlers attached), these warnings may not be visible anywhere you're actually looking. A minimal, always-safe starting point for local debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 22.7 Beyond the Standard Library

For interactively debugging a **running** production-like asyncio application — attaching a live console to inspect tasks, the loop's state, and more — third-party tools like `aiomonitor` extend well beyond what `python -m asyncio ps/pstree` offers. Worth exploring once you outgrow the built-in tools for more involved production debugging needs, though the standard library's own tools cover the large majority of everyday debugging situations this course has focused on.

---

## 22.8 Hands-On Exercises

**Exercise 1 — Trigger a slow-callback warning.**
Run the §22.2 example with `debug=True` and confirm you see the `"took 0.5XX seconds"` warning. Then wrap the `time.sleep(0.5)` in `asyncio.to_thread()` (Chapter 18) instead, and confirm the warning disappears — proving the fix actually worked, not just hid the symptom.

**Exercise 2 — Tune the sensitivity.**
Lower `loop.slow_callback_duration` to `0.01` and re-run a version of your program doing several normal (non-blocking) operations. Observe how much noisier the warnings become, and discuss in 2–3 sentences the trade-off between catching real problems early and drowning in false positives.

**Exercise 3 — Compare traceback detail.**
Recreate the §22.3 fire-and-forget exception example, once with `debug=True` and once without. Compare the two tracebacks side by side and note what extra information debug mode provides.

**Exercise 4 — Use `pstree` on a real hang.**
Run the §22.5 example, find its PID, and use `python -m asyncio pstree <PID>` from a second terminal while it's still running. Confirm you can see both named tasks and tell which one is still pending.

**Exercise 5 (stretch) — Debug a deliberately buggy script.**
Write a ~30-line script containing at least 3 of the anti-patterns from §22.4 (e.g., a direct blocking call, a fire-and-forget task that fails, and a swallowed `CancelledError`). Enable debug mode and configure logging, then use the resulting warnings to find and fix each bug one at a time, confirming after each fix that the corresponding warning disappears.

---

## 22.9 Common Pitfalls

- **Leaving debug mode enabled in production.** The extra tracking (source tracebacks on every task, additional checks) has real overhead — use it for development and active troubleshooting, not as a permanent setting.
- **Not configuring `logging` at all.** Debug mode's warnings go through the standard logging system — without any configuration, they may never actually reach anywhere you're looking.
- **Setting `slow_callback_duration` too aggressively low.** This produces so much noise that genuinely useful warnings get lost in the flood.
- **Treating debug mode as a substitute for understanding the underlying mechanics.** It surfaces *symptoms* — slow callbacks, unretrieved exceptions — but the rest of this course is what tells you *why* they happen and how to actually fix them.

---

## 22.10 Further Reading

- [Python docs: Debug Mode](https://docs.python.org/3/library/asyncio-dev.html)
- [Python docs: `python -m asyncio` command-line tool](https://docs.python.org/3/library/asyncio-cli.html)
- [aiomonitor project](https://github.com/aio-libs/aiomonitor) — a live console debugger for running asyncio applications

---

**Next: Chapter 23 — Testing Async Code** (`pytest-asyncio`, mocking coroutines, and testing timeouts/cancellation properly).
