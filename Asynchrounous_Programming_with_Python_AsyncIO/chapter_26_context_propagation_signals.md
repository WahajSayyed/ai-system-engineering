# Chapter 26: Context Propagation and Signals

> Part 6 — Advanced Patterns and Design · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 26.1 `contextvars`: Per-Task Isolated State

Chapter 4 briefly mentioned `contextvars` alongside `asyncio.Runner`. Here's the full picture, and it solves a genuinely important problem: **how do you carry a value — like a request ID — through a whole chain of nested async calls, without explicitly passing it as a parameter to every single function along the way, while still keeping concurrent tasks fully isolated from each other?**

```python
import asyncio
import contextvars

request_id = contextvars.ContextVar("request_id", default="unknown")

async def log_something(message):
    print(f"[{request_id.get()}] {message}")

async def handle_request(rid):
    request_id.set(rid)
    await log_something("starting")
    await asyncio.sleep(0.1)
    await log_something("finishing")

async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(handle_request("req-A"))
        tg.create_task(handle_request("req-B"))

asyncio.run(main())
```

Output (interleaved, but correctly tagged):
```
[req-A] starting
[req-B] starting
[req-A] finishing
[req-B] finishing
```

**The key mechanism:** each `Task`, when created, gets its **own copy** of the current context at that moment — not a shared reference. So `request_id.set("req-A")` inside one task's coroutine tree has **zero effect** on the other task's view of `request_id`, even though both tasks are running concurrently, interleaved on the same single thread, referencing the exact same `ContextVar` object. This is precisely the isolation a naive module-level variable could never give you — a plain global would be clobbered by whichever task set it most recently, visible to everyone.

---

## 26.2 Practical Use: Structured Logging With Request IDs

This is the single most common real use of `contextvars` in production code — automatically tagging every log line within a request's task tree, without threading a `request_id` parameter through every function signature in your codebase:

```python
import logging
import contextvars

request_id_var = contextvars.ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

handler = logging.StreamHandler()
handler.addFilter(RequestIdFilter())
handler.setFormatter(logging.Formatter("[%(request_id)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])

logger = logging.getLogger(__name__)

async def handle_request(rid):
    request_id_var.set(rid)
    logger.info("handling request")
    await do_some_work()
    logger.info("request complete")

async def do_some_work():
    logger.info("doing work")   # automatically tagged with the CALLER's request_id,
                                 # with no explicit parameter passed at all
    await asyncio.sleep(0.1)
```

Every log call anywhere in `do_some_work()` (or anything it calls, however deeply nested) automatically picks up the correct `request_id` for whichever request's task tree it's currently running under — exactly the same benefit Chapter 11 gestured at when discussing tagging concurrent error logs.

---

## 26.3 The Gotcha: `run_in_executor` Doesn't Propagate Context Automatically

Chapter 18 mentioned that `asyncio.to_thread()` correctly propagates `contextvars` into the worker thread. **Plain `loop.run_in_executor()` does not do this automatically** — a real, easy-to-miss gap:

```python
def blocking_read_request_id():
    return request_id_var.get()   # will NOT see the caller's value via run_in_executor!

async def broken():
    request_id_var.set("req-X")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_read_request_id)
    print(result)   # prints the default, NOT "req-X"
```

**The fix**, if you must use `run_in_executor()` directly rather than `to_thread()`: explicitly capture and run within the current context using `contextvars.copy_context()`:

```python
async def fixed():
    request_id_var.set("req-X")
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    result = await loop.run_in_executor(None, ctx.run, blocking_read_request_id)
    print(result)   # correctly prints "req-X"
```

This is one more concrete reason to prefer `asyncio.to_thread()` (Chapter 18) over manually calling `run_in_executor(None, ...)` for the common case — it handles this correctly for you already.

---

## 26.4 Handling OS Signals: `SIGINT` and `SIGTERM`

Chapter 25 built the graceful shutdown *pattern*, triggered manually. In a real deployed service, that trigger needs to come from an actual **OS signal** — `SIGINT` (Ctrl+C) or `SIGTERM` (the standard "please shut down" signal sent by process managers, container orchestrators, `systemd`, etc.).

By default, Python's response to `SIGINT` is to raise `KeyboardInterrupt` — which technically does stop `asyncio.run()`, but abruptly, without giving your code a clean chance to run orderly async cleanup (finishing in-flight work, closing connection pools gracefully). `SIGTERM` by default does nothing special in Python at all (it terminates the process immediately, with no cleanup opportunity whatsoever).

**The asyncio-native fix: `loop.add_signal_handler()`** — registers a callback to run **on the event loop itself** when a signal arrives, letting you trigger genuinely async-aware shutdown logic instead of an abrupt unwind:

```python
import asyncio
import signal

async def main():
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    print("Running — press Ctrl+C or send SIGTERM to shut down gracefully")
    await shutdown_event.wait()
    print("Shutdown signal received, cleaning up...")

asyncio.run(main())
```

**Important platform limitation:** `loop.add_signal_handler()` is **POSIX-only** — it raises `NotImplementedError` on Windows' `ProactorEventLoop`. For cross-platform code, you have two realistic options: accept the simpler (if less graceful) default `KeyboardInterrupt`-based shutdown on Windows, or run a background thread using plain `signal.signal()` that communicates back to the loop via `loop.call_soon_threadsafe()` (Chapter 21) — genuinely more complex, and typically only worth it if graceful shutdown on Windows specifically is a hard requirement for your deployment target.

---

## 26.5 The Complete Picture: Real Signals + Graceful Shutdown

Combining Chapter 25's pattern with real signal handling gives you the full, deployable version:

```python
import asyncio
import signal

async def worker(name, shutdown_event):
    while not shutdown_event.is_set():
        print(f"{name}: working")
        await asyncio.sleep(1)
    print(f"{name}: noticed shutdown, winding down")

async def main():
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    async with asyncio.TaskGroup() as tg:
        workers = [
            tg.create_task(worker(f"worker-{i}", shutdown_event))
            for i in range(3)
        ]

        await shutdown_event.wait()
        print("Shutdown requested — waiting up to 5s for workers to finish")

        try:
            async with asyncio.timeout(5):
                await asyncio.gather(*workers)
        except TimeoutError:
            print("Grace period exceeded — force-cancelling stragglers")
            for w in workers:
                w.cancel()

asyncio.run(main())
```

This single example ties together `TaskGroup` (Chapter 8), cancellation (Chapter 9), `asyncio.timeout()` (Chapter 10), `Event` (Chapter 12), and real signal handling — a genuinely complete, production-shaped answer to **"how do I actually, correctly shut down a long-running asyncio service"**, which is one of the most common practical questions this entire course has been building toward.

---

## 26.6 Hands-On Exercises

**Exercise 1 — Prove context isolation.**
Recreate the §26.1 example with 2 concurrent tasks setting `request_id` to different values, each sleeping and re-reading the variable afterward. Confirm each task consistently sees its own value, despite running interleaved on the same thread and referencing the same `ContextVar` object.

**Exercise 2 — Structured logging with request IDs.**
Implement the §26.2 pattern. Simulate 3 concurrent "requests," each with a different ID, each making 2–3 log calls with small `asyncio.sleep()`s in between (so their log lines genuinely interleave in the console). Confirm every log line is tagged with the correct request ID for its own task tree.

**Exercise 3 — Reproduce and fix the `run_in_executor` gotcha.**
Implement `broken()` and `fixed()` from §26.3. Confirm `broken()` does NOT see the parent's context value, while `fixed()` (using `contextvars.copy_context()`) does. Then rewrite `broken()` using `asyncio.to_thread()` instead, and confirm it also now works correctly with no extra code needed.

**Exercise 4 — Real signal-triggered shutdown.**
Implement the §26.5 example. Run it, then send `SIGTERM` from another terminal (`kill -TERM <PID>` on POSIX) or press Ctrl+C, and confirm the workers notice and shut down gracefully within the 5-second grace period.

**Exercise 5 (stretch) — Force a straggler.**
Modify one worker in your Exercise 4 solution to ignore `shutdown_event` for 10 seconds regardless (simulating a worker stuck on something). Confirm the other workers shut down cleanly within the grace period, while the straggler gets force-cancelled once the 5-second bound is exceeded — and that the overall program still exits promptly rather than hanging on the stuck worker.

---

## 26.7 Common Pitfalls

- **Using a plain global variable instead of a `ContextVar` for per-request state.** A regular global is genuinely shared, mutable state — concurrent tasks would clobber each other's values. `ContextVar` gives you per-task isolation specifically because each `Task` gets its own copy of the context at creation time.
- **Assuming `run_in_executor()` propagates context automatically like `to_thread()` does.** It doesn't — use `contextvars.copy_context().run(...)` explicitly if you must use `run_in_executor()` directly, or simply prefer `to_thread()`.
- **Relying solely on default `KeyboardInterrupt` handling for a service needing orderly cleanup.** It works, but abruptly — use `loop.add_signal_handler()` for genuine async-aware graceful shutdown on POSIX systems.
- **Assuming `add_signal_handler()` works on Windows.** It's POSIX-only and raises `NotImplementedError` on the default Windows event loop — plan for this explicitly if cross-platform graceful shutdown matters for your deployment.
- **Not bounding the shutdown grace period.** Without a timeout wrapping the "wait for workers to finish" step, one stuck worker can prevent your entire process from ever exiting during shutdown.

---

## 26.8 Further Reading

- [Python docs: `contextvars`](https://docs.python.org/3/library/contextvars.html)
- [Python docs: `loop.add_signal_handler()`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.add_signal_handler)

---

**Next: Chapter 27 — AsyncIO vs. Alternatives** (comparing with `trio`'s structured concurrency model and `curio`, and what asyncio itself borrowed from them — `TaskGroup` and `timeout()` didn't appear in a vacuum).
