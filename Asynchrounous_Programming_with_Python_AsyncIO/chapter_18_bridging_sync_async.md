# Chapter 18: Bridging Sync and Async Code

> Part 4 — I/O, Networking, and Real Systems · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 18.1 The Problem: Blocking Calls Have No Business in a Coroutine

Recall Chapter 3's central rule: any blocking call inside a coroutine — `time.sleep()`, a synchronous `requests.get()`, a slow synchronous file read, a heavy pure-Python computation — **freezes the entire event loop**, not just the coroutine that made the call. Sometimes, though, you have no choice: a legacy library, a C extension, or a numeric library with no async equivalent. This chapter is about running that unavoidable blocking work **off to the side**, without stalling everything else.

---

## 18.2 `loop.run_in_executor()`

```python
import asyncio
import time

def blocking_call(seconds):
    time.sleep(seconds)   # a genuinely blocking call — no await available
    return f"slept {seconds}s"

async def main():
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_call, 2)
    print(result)

asyncio.run(main())
```

`run_in_executor(executor, func, *args)` runs `func(*args)` in a background **thread** (by default), returning an awaitable that resolves with `func`'s return value once it completes. Passing `None` as the executor uses the loop's built-in default `ThreadPoolExecutor` — you don't need to create one yourself for the common case.

Critically: while `blocking_call()` runs on its background thread, **the event loop keeps running everything else normally.** Other coroutines, other tasks, all continue uninterrupted — the blocking work has been moved off the loop's own thread entirely.

---

## 18.3 `asyncio.to_thread()`: The Modern Shortcut (Python 3.9+)

For the common "just run this blocking thing in a background thread" case, `asyncio.to_thread()` is a cleaner shorthand:

```python
import asyncio

async def main():
    result = await asyncio.to_thread(blocking_call, 2)
    print(result)

asyncio.run(main())
```

This is functionally similar to `run_in_executor(None, functools.partial(blocking_call, 2))`, but with one meaningful improvement: **`to_thread()` correctly propagates `contextvars`** into the worker thread, so context-local state set in your coroutine (e.g., a request ID for logging, per Chapter 26) is still visible inside the blocking function. Prefer `to_thread()` over manually calling `run_in_executor(None, ...)` unless you specifically need a custom executor (see below).

---

## 18.4 `ThreadPoolExecutor` vs. `ProcessPoolExecutor`

This distinction matters enormously, and it connects directly back to Chapter 1's GIL discussion:

- **`ThreadPoolExecutor`** (the default): good for **blocking I/O-bound** calls. On a standard (GIL-enabled) CPython build, the GIL is released during most blocking I/O waits implemented in C, so multiple threads can make I/O progress concurrently even though only one thread executes Python bytecode at a time.
- **`ProcessPoolExecutor`**: needed for **CPU-bound** blocking work. Since threads under a GIL-enabled interpreter can't execute Python bytecode in parallel, a CPU-heavy pure-Python function gains **nothing** from a thread pool — you need separate OS processes for genuine parallel computation, at the cost of serialization (pickling arguments and results) and higher memory overhead.

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
import time

def cpu_heavy(n):
    return sum(i * i for i in range(n))

async def main():
    loop = asyncio.get_running_loop()

    start = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        results = await asyncio.gather(
            *(loop.run_in_executor(pool, cpu_heavy, 20_000_000) for _ in range(4))
        )
    print(f"ProcessPoolExecutor: {time.perf_counter() - start:.2f}s")

asyncio.run(main())
```

Run the same 4 calls through `asyncio.to_thread()` (thread pool) instead, and — on a standard GIL-enabled Python — you'll see roughly the **same total time as running them one after another**, while the process-pool version genuinely parallelizes across CPU cores and finishes noticeably faster. This is the clearest, most concrete way to see the GIL's effect on CPU-bound work in practice.

---

## 18.5 2026 Update: Free-Threaded Python Changes the Threading Calculus

Tying back to Chapter 1: on a **free-threaded** Python build (3.13+ experimental, 3.14+ officially supported per PEP 779), the GIL isn't serializing bytecode execution — so `ThreadPoolExecutor` **can** actually deliver real parallel speedup for CPU-bound work too, something that's never been true on a standard GIL-enabled interpreter. This is a genuine, recent shift in the traditional "threads for I/O, processes for CPU" guidance — but as of 2026, **`ProcessPoolExecutor` still has its place** even on free-threaded builds: it gives you full OS-level isolation (useful for untrusted or crash-prone code) and sidesteps any remaining free-threading rough edges in less-updated C-extension dependencies. Since the default, non-free-threaded build remains the common case for most production deployments today, this chapter's core guidance (threads for I/O, processes for CPU) still applies as the safe default — treat free-threaded builds as a specific, deliberate opt-in for CPU-bound-heavy applications ready to test for it.

---

## 18.6 Sizing and Customizing Executors

The default thread pool has a bounded number of workers (`min(32, os.cpu_count() + 4)` as of recent Python versions). If you submit more concurrent blocking calls than available workers, the extras simply **queue up and wait** — a backpressure-like effect similar to Chapter 12's `Semaphore`.

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def main():
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2)

    async def timed_call(n):
        await loop.run_in_executor(executor, time.sleep, 1)
        print(f"call {n} done")

    start = time.perf_counter()
    await asyncio.gather(*(timed_call(i) for i in range(5)))
    print(f"Total: {time.perf_counter() - start:.1f}s")   # ~3s, not ~1s: 5 calls / 2 workers

asyncio.run(main())
```

With only 2 workers and 5 one-second blocking calls, the total time is roughly 3 seconds (5 calls, 2 at a time), not 1 second — exactly the same bounded-concurrency shape as a `Semaphore(2)`. Set `max_workers` deliberately based on your actual workload rather than relying on the default.

---

## 18.7 Real Pattern: Wrapping a Blocking Library Call

If a library genuinely has no async equivalent, `to_thread()` is your fallback:

```python
import asyncio
import requests   # a synchronous library, no async support

async def fetch(url):
    return await asyncio.to_thread(requests.get, url)

async def main():
    urls = ["https://example.com", "https://example.org", "https://example.net"]
    responses = await asyncio.gather(*(fetch(url) for url in urls))
    for r in responses:
        print(r.status_code)

asyncio.run(main())
```

This works, and it does let several blocking `requests.get()` calls happen "concurrently" from your coroutine's perspective. **But it doesn't scale the way a native async HTTP client does** — you're still bounded by the thread pool's worker count, and each blocking call ties up a whole OS thread for its entire duration. For genuinely high-concurrency HTTP work (hundreds or thousands of simultaneous requests), Chapter 19's async-native HTTP clients are a far better fit; reach for `to_thread()`-wrapped blocking libraries only when no async alternative exists at all.

---

## 18.8 Cancellation and Exceptions

Exceptions raised inside the executor function propagate normally through the awaited result, exactly as you'd expect:

```python
def will_fail():
    raise ValueError("boom")

async def main():
    try:
        await asyncio.to_thread(will_fail)
    except ValueError as e:
        print(f"Caught: {e}")
```

**Cancellation, however, has the same gotcha as Chapter 17's subprocesses:** cancelling the `Task` awaiting a `to_thread()`/`run_in_executor()` call does **not** stop the underlying thread. Python has no safe, general mechanism for forcibly killing a running thread — the blocking function keeps running to completion regardless of the cancellation; you simply stop waiting for its result.

```python
import asyncio
import time

def slow_blocking():
    time.sleep(5)
    print("blocking call finally finished (in the background)")

async def main():
    task = asyncio.create_task(asyncio.to_thread(slow_blocking))
    await asyncio.sleep(1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Task cancelled — but the thread is still running underneath!")

    await asyncio.sleep(5)   # only to let you observe the delayed print

asyncio.run(main())
```

For `ProcessPoolExecutor`, terminating the underlying OS process is at least theoretically possible (unlike a thread) — but the default `run_in_executor`/`to_thread` cancellation path still just detaches from waiting, rather than forcibly stopping anything. If you need genuinely interruptible background work, consider a design where the blocking function itself periodically checks a shared cancellation flag, or accept that the operation, once started, will run to completion.

---

## 18.9 Hands-On Exercises

**Exercise 1 — Prove the loop isn't frozen.**
Write a coroutine that calls `time.sleep(2)` **directly**, un-wrapped, alongside another coroutine that prints a message every 0.5 seconds. Confirm the second coroutine's prints are all delayed until the sleep finishes (the loop is frozen). Then wrap the sleep in `run_in_executor()` instead and confirm the second coroutine's prints now continue on schedule, unaffected.

**Exercise 2 — `to_thread()` vs. `run_in_executor()`.**
Rewrite Exercise 1's fix using `asyncio.to_thread()` instead of `run_in_executor(None, ...)`. Compare the two call sites for readability.

**Exercise 3 — Threads vs. processes for CPU-bound work.**
Implement `cpu_heavy(n)` from §18.4. Time 4 concurrent calls via `asyncio.to_thread()` (thread pool) and via `ProcessPoolExecutor` (as in §18.4). Confirm the thread-pool version takes roughly 4x as long as a single call (no real parallelism), while the process-pool version finishes much faster (genuine parallelism across cores).

**Exercise 4 — Bounded executor sizing.**
Implement the §18.6 example with `ThreadPoolExecutor(max_workers=2)` and 5 one-second blocking calls. Confirm the total time is ~3 seconds, then change `max_workers` to 5 and confirm it drops to ~1 second.

**Exercise 5 (stretch) — Watch the cancellation gotcha happen.**
Implement the §18.8 example exactly. Confirm the `CancelledError` is caught immediately (~1 second in), but `"blocking call finally finished"` still prints roughly 5 seconds after the script started — proving the background thread kept running the entire time, unaffected by the cancellation of the task that was waiting on it.

---

## 18.10 Common Pitfalls

- **Calling a blocking function directly inside a coroutine.** This freezes the entire event loop, not just that coroutine — always wrap it via `to_thread()`/`run_in_executor()` (or use a native async alternative if one exists).
- **Using `ThreadPoolExecutor` expecting a CPU-bound speedup on a standard interpreter.** It won't parallelize actual Python computation under the GIL — use `ProcessPoolExecutor` for that.
- **Using `ProcessPoolExecutor` without considering pickling costs.** Arguments, return values, and the target function itself must all be picklable — no local closures or lambdas as the target, and large arguments/results incur real serialization overhead.
- **Assuming cancellation stops the underlying thread or process.** It doesn't, by default — the blocking call runs to completion regardless; you're only detaching from waiting on its result.
- **Not sizing your executor deliberately for a workload with many concurrent blocking calls.** The default worker count may silently bottleneck a workload that assumes unlimited concurrency.

---

## 18.11 Further Reading

- [Python docs: `loop.run_in_executor()`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor)
- [Python docs: `asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread)
- [Python docs: `concurrent.futures` — `ThreadPoolExecutor` and `ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html)

---

**Next: Chapter 19 — HTTP Clients and Servers in the Async World** (`aiohttp` and `httpx`, and building a concurrent, rate-limited web scraper — the native-async alternative to the `to_thread()`-wrapped `requests` fallback from §18.7).
