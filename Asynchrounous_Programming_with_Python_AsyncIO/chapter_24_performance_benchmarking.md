# Chapter 24: Performance and Benchmarking

> Part 5 — Internals and Production Concerns · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 24.1 Measure, Don't Assume

Intuitions about async performance are frequently wrong — the only reliable approach is to measure. Two metrics matter, and they're not the same thing:

- **Throughput**: operations completed per unit time (e.g., requests/second)
- **Latency**: time taken per individual operation — and critically, not just the *average*, but the **distribution**, especially the tail (p95, p99)

A system can have great average latency and still feel terrible to use if its p99 latency is awful — a small fraction of very slow requests is often what users actually notice and complain about, even when the average looks fine on a dashboard.

---

## 24.2 Measuring Latency Correctly, With Percentiles

```python
import asyncio
import random
import statistics
import time

async def simulated_call():
    delay = random.uniform(0.01, 0.2)   # simulate variable network latency
    await asyncio.sleep(delay)

async def measure_latencies(n, concurrency=None):
    durations = []

    async def timed_call():
        start = time.perf_counter()
        await simulated_call()
        durations.append(time.perf_counter() - start)

    if concurrency is None:
        for _ in range(n):
            await timed_call()
    else:
        sem = asyncio.Semaphore(concurrency)
        async def bounded():
            async with sem:
                await timed_call()
        async with asyncio.TaskGroup() as tg:
            for _ in range(n):
                tg.create_task(bounded())

    durations.sort()
    p50 = statistics.median(durations)
    p95 = durations[int(len(durations) * 0.95)]
    p99 = durations[int(len(durations) * 0.99)]
    print(f"p50={p50*1000:.1f}ms  p95={p95*1000:.1f}ms  p99={p99*1000:.1f}ms  max={max(durations)*1000:.1f}ms")

asyncio.run(measure_latencies(200, concurrency=20))
```

Reporting p50/p95/p99 (rather than just an average) is standard practice for any real latency measurement — always prefer it, even for quick, informal benchmarks.

---

## 24.3 Measuring Throughput

Throughput is simplest measured under **realistic concurrent load**, not sequentially:

```python
start = time.perf_counter()
await measure_latencies(1000, concurrency=50)
elapsed = time.perf_counter() - start
print(f"Throughput: {1000 / elapsed:.1f} ops/sec")
```

**Be careful what you're actually measuring.** If your workload calls a real remote server, network variance and the server's own load become part of your numbers — which is sometimes exactly what you want (end-to-end, realistic numbers), but not if you're specifically trying to isolate **your own code's** overhead. For that, benchmark against a local test server, or replace real I/O with `asyncio.sleep()` calls of a known, fixed duration, so any measured overhead can only be coming from your own scheduling and coordination logic, not network noise.

---

## 24.4 Benchmarking AsyncIO's Own Scheduling Overhead

Before layering real work on top, it's useful to know the loop's own baseline overhead for scheduling many tasks:

```python
import asyncio
import time

async def noop():
    await asyncio.sleep(0)   # yields control once, does essentially nothing else

async def main():
    n = 100_000
    start = time.perf_counter()
    async with asyncio.TaskGroup() as tg:
        for _ in range(n):
            tg.create_task(noop())
    elapsed = time.perf_counter() - start
    print(f"{n} trivial tasks in {elapsed:.2f}s ({n/elapsed:,.0f} tasks/sec)")

asyncio.run(main())
```

This gives you a rough ceiling: however many "tasks per second" your machine can push through with essentially zero real work happening — a useful baseline to compare against once you add genuine I/O or computation, so you know how much of your measured overhead is the loop itself versus your actual workload.

---

## 24.5 `uvloop` Benchmarks in Practice

Reusing Chapter 16's streams code and Chapter 21's `loop_factory`, you can directly compare the default loop against `uvloop` on the exact same workload:

```python
import asyncio
import time
import uvloop

async def run_workload():
    # ... your concurrent TCP client workload from Chapter 16, e.g. N short requests ...
    pass

def benchmark(loop_factory, label):
    start = time.perf_counter()
    asyncio.run(run_workload(), loop_factory=loop_factory)
    print(f"{label}: {time.perf_counter() - start:.2f}s")

benchmark(None, "default SelectorEventLoop")
benchmark(uvloop.new_event_loop, "uvloop")
```

Publicly reported benchmarks commonly show `uvloop` delivering meaningful speedups — often cited in the range of 2–4x — for network-heavy workloads with many concurrent connections. **Treat any such number as a starting expectation, not a guarantee for your program specifically** — the actual gain depends heavily on what your workload is actually bottlenecked on (raw scheduling overhead vs. application logic vs. the remote server's own latency). Always benchmark your own real workload rather than assuming a published number will transfer directly.

---

## 24.6 When Threads/Processes Genuinely Beat AsyncIO

Chapter 18 established the guidance (threads for I/O, processes for CPU-bound work) — here it is with concrete numbers to actually measure yourself:

| Approach | 4x CPU-heavy calls, standard GIL-enabled Python |
|---|---|
| Sequential `await`, single coroutine | ~4x a single call's time (no overlap at all) |
| `asyncio.to_thread()` (thread pool) | ~4x a single call's time (GIL serializes the actual computation) |
| `ProcessPoolExecutor` | ~1x a single call's time (genuine parallel execution across cores) |

Run Chapter 18's `cpu_heavy()` benchmark yourself and fill in real numbers for your own machine — the *pattern* (threads not helping, processes genuinely parallelizing) should hold on any standard, non-free-threaded Python build.

**One more practical point, easy to over-look:** for a genuinely small number of one-off blocking calls with no complex coordination needs, plain `threading` (or even just accepting the blocking cost, if it's rare and short) can be simpler and just as effective as building out a full asyncio + executor setup. Not every blocking operation justifies the added complexity of async infrastructure — match the tool to the actual scale of the problem.

---

## 24.7 Guarding Against Accidental Blocking Regressions

Chapter 22's debug mode and slow-callback warnings aren't just for one-off debugging — they can be wired into your **test suite or CI pipeline** as an ongoing regression guard, catching a newly introduced blocking call before it reaches production:

```python
import asyncio
import logging

async def test_critical_path_has_no_blocking_calls(caplog):
    with caplog.at_level(logging.WARNING, logger="asyncio"):
        loop = asyncio.get_running_loop()
        loop.slow_callback_duration = 0.05
        await run_critical_path()   # your actual code under test

    slow_warnings = [r for r in caplog.records if "took" in r.message]
    assert not slow_warnings, f"Unexpected blocking detected: {slow_warnings}"
```

This is a lightweight, low-effort way to catch a regression like "someone added a synchronous `requests.get()` inside a hot coroutine" automatically, rather than only discovering it from a production latency spike.

**A caveat on profiling:** `cProfile` still runs against asyncio programs, but its wall-clock-time output can be misleading — since only one coroutine is "running" at any instant, but the loop itself may be idly waiting on I/O the rest of the time, straightforward profiler output can conflate "genuinely computing" time with "the loop happened to be here while waiting" time. Targeted latency/throughput measurements (§24.2–24.3), rather than a blind `cProfile` run, are usually more directly informative for async code specifically.

---

## 24.8 Hands-On Exercises

**Exercise 1 — Latency harness, sequential vs. concurrent.**
Implement `measure_latencies()` from §24.2. Run it sequentially (`concurrency=None`) and with `concurrency=20`, both for `n=200`. Compare the p50/p95/p99/max figures and the total wall-clock time between the two runs.

**Exercise 2 — Pure scheduling overhead.**
Implement the §24.4 benchmark. Run it with `n` values of 1,000, 10,000, and 100,000, and plot (or just tabulate) tasks/sec at each scale — does the rate stay roughly constant, or does it degrade as `n` grows?

**Exercise 3 — `uvloop` comparison.**
Using Chapter 16's TCP client/server code, benchmark handling 500 concurrent short requests with the default loop, then again with `loop_factory=uvloop.new_event_loop`. Report the timing difference you observe on your own machine.

**Exercise 4 — The threads-vs-processes table, with real numbers.**
Fill in the §24.6 table with actual measured times on your machine, using Chapter 18's `cpu_heavy()` function. Write 2–3 sentences interpreting what you see.

**Exercise 5 (stretch) — A CI-style blocking-call regression guard.**
Implement the §24.7 test pattern for a small piece of code you write yourself that deliberately calls `time.sleep()` directly (a "regression"). Confirm the test fails, catching the blocking call. Then fix the code (wrap it in `to_thread()`) and confirm the test now passes.

---

## 24.9 Common Pitfalls

- **Trusting average latency alone.** Always look at percentiles, especially p95/p99 — tail latency is frequently what actually affects user-perceived quality, even when the average looks fine.
- **Benchmarking against real remote servers and attributing all variance to your own code.** Isolate your code's overhead with local test servers or simulated delays when that's specifically what you want to measure.
- **Assuming a published `uvloop` (or any other) benchmark number transfers directly to your workload.** Gains vary substantially depending on what you're actually bottlenecked on — always benchmark your own real workload.
- **Using `cProfile` naively on asyncio code and misreading the results.** Wall-clock-heavy profiler output can conflate idle waiting time with genuine computation for whichever coroutine happens to be active during a sample.
- **Over-engineering a trivial, one-off blocking call into a full asyncio + executor setup.** Match the tool's complexity to the actual scale of the problem — sometimes plain threading, or just accepting a rare blocking cost, is the more sensible choice.

---

## 24.10 Further Reading

- [uvloop documentation and benchmarks](https://uvloop.readthedocs.io/)
- [Python docs: `cProfile` and Profiling](https://docs.python.org/3/library/profile.html)

---

**Next: Chapter 25 — Production Design Patterns** (rate limiting, retries with backoff, circuit breakers, graceful shutdown, and connection pooling patterns, moving into Part 6's advanced, production-facing material).
