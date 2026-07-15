# Chapter 21: Event Loop Internals

> Part 5 — Internals and Production Concerns · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 21.1 Revisiting Chapter 3, for Real This Time

Chapter 3's `TinyLoop` modeled the event loop's core idea with a single kind of waiting: `sleep`. The real event loop generalizes this to genuine **I/O readiness** — "tell me when this socket has data" — using operating-system-level mechanisms, and ships as one of two concrete implementations depending on your platform. This chapter fills in those details, plus the modern (and recently much-simplified) story around how loops get created and configured.

---

## 21.2 `selectors`: Watching File Descriptors for Readiness

On POSIX systems (Linux, macOS, BSD), asyncio's default loop is a `SelectorEventLoop`, built on Python's standard `selectors` module — a portable wrapper over the fastest OS-native "tell me when these file descriptors are ready" mechanism available: `epoll` on Linux, `kqueue` on macOS/BSD, falling back to plain `select` where neither is available.

```python
import selectors
import socket

selector = selectors.DefaultSelector()   # picks epoll/kqueue/select automatically

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect_ex(("example.com", 80))
sock.setblocking(False)

selector.register(sock, selectors.EVENT_WRITE)

events = selector.select(timeout=5)   # blocks here until ready, or timeout elapses
for key, mask in events:
    print(f"{key.fileobj} is ready for: {mask}")

selector.close()
```

This is conceptually exactly what `BaseEventLoop._run_once()` does internally, every single iteration: register interest in a set of file descriptors, call `select(timeout)` (blocking only until *something* is ready or a timer is due), then dispatch callbacks for whatever came back ready.

**Windows is different:** since Python 3.8, asyncio's default loop on Windows is a `ProactorEventLoop`, built on Windows' I/O Completion Ports (IOCP) rather than a selector-style readiness model — a different underlying OS mechanism, needed historically for full subprocess support on Windows. You generally don't need to think about this distinction in application code; it's handled transparently by the default loop selection for your platform.

---

## 21.3 The Scheduling Primitives: `call_soon`, `call_later`, `call_at`

These are the loop's raw scheduling API, underneath every `Task`, every `asyncio.sleep()`, everything:

| Method | Behavior |
|---|---|
| `loop.call_soon(callback, *args)` | Schedules `callback` to run on the **next** iteration of the loop — as soon as possible, but not synchronously right now |
| `loop.call_later(delay, callback, *args)` | Schedules `callback` to run after `delay` seconds have passed |
| `loop.call_at(when, callback, *args)` | Schedules `callback` to run at an absolute time, given in terms of `loop.time()` |
| `loop.call_soon_threadsafe(callback, *args)` | The **only safe way** to schedule a callback from a **different OS thread** — `call_soon()` itself is not thread-safe |

That last one matters more than it might look: if you ever have a callback (e.g., from a C extension, or a thread you spawned yourself) that needs to notify the event loop from outside its own thread, `call_soon_threadsafe()` is mandatory — calling `call_soon()` directly from another thread is a genuine, if often silent, bug. This is exactly what Chapter 6's callback-bridging pattern used under the hood when the callback could fire on a different thread.

Every one of these returns a `Handle` (or, for `call_later`/`call_at`, a `TimerHandle`), which has a `.cancel()` method — this is how, e.g., `asyncio.sleep()` and `asyncio.timeout()` clean up their own scheduled callbacks when they're no longer needed.

---

## 21.4 `loop.time()`: A Clock That Only Moves Forward

Internally, all of this scheduling is based on `loop.time()`, not `time.time()`. This is deliberate: `time.time()` reflects the **wall clock**, which can jump — backward or forward — due to NTP adjustments or manual changes to the system clock. If `call_later()` used the wall clock internally, a clock adjustment could cause scheduled callbacks to fire far too early, far too late, or (in a backward jump) seemingly "in the past." `loop.time()` is based on a **monotonic** clock (`time.monotonic()`-equivalent) that only ever moves forward, immune to these adjustments — exactly what you want underneath a scheduler.

---

## 21.5 Tying It Together: `_run_once`

Putting the pieces from this chapter and Chapter 3 together, one iteration of the real event loop does roughly this:

```
timeout = time until the next scheduled call is due (or None/0 if something's already ready)
ready_io_events = selector.select(timeout)     # blocks here, waiting on the OS
for each ready I/O event:
    queue its registered callback onto the ready list
move any scheduled calls that are now due onto the ready list
run every callback currently on the ready list, once each, in order
```

**Performance note:** `Task` and `Future` have C-accelerated implementations (the `_asyncio` module) for speed, but the `SelectorEventLoop` itself remains pure Python. This is exactly why **`uvloop`** — a drop-in event loop replacement built in Cython on top of `libuv` (the same C library Node.js uses) — can meaningfully outperform the default loop for I/O-heavy workloads: it replaces this whole `_run_once` cycle with a highly optimized C implementation, while keeping the exact same `asyncio` API you already write against.

```python
import asyncio
import uvloop

asyncio.run(main(), loop_factory=uvloop.new_event_loop)   # see §21.6 for loop_factory
```

---

## 21.6 The Policy System: Deprecated, and On Its Way Out

Historically, `asyncio` had a whole **event loop policy** system (`asyncio.get_event_loop_policy()`/`set_event_loop_policy()`) letting you customize how loops get created and retrieved — per-thread, or with custom behavior, and used by libraries like `uvloop` to install themselves as the process-wide default.

**As of Python 3.14, the entire policy system is deprecated, scheduled for removal in Python 3.16.** The CPython team's own reasoning: loops are always per-thread anyway, so a separate policy layer just added confusion and cross-library conflicts without serving a real remaining purpose. `asyncio.set_event_loop_policy()`, `get_event_loop_policy()`, and the policy classes themselves all now raise `DeprecationWarning`.

**The modern replacement: `loop_factory`.** Both `asyncio.run()` and `asyncio.Runner` (Chapter 4) now accept a `loop_factory` argument — a callable that returns the event loop instance you want used, replacing the entire policy indirection with a single, direct parameter:

```python
import asyncio
import uvloop

async def main():
    print(type(asyncio.get_running_loop()))

# The modern way to select a specific loop implementation:
asyncio.run(main(), loop_factory=uvloop.new_event_loop)

# Or, for more control (multiple run() calls sharing setup):
with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
    runner.run(main())
```

If you're reading older tutorials or library code using `set_event_loop_policy()` to install `uvloop` or a custom loop class, know that it still works today but is on a deprecation path — prefer `loop_factory` in any new code.

---

## 21.7 Hands-On Exercises

**Exercise 1 — Raw `selectors` demo.**
Run the §21.2 example against a real socket connection (adjust the host/port as needed). Confirm `selector.select()` blocks until the socket becomes writable (indicating the connection succeeded), then returns.

**Exercise 2 — Watch `call_later` ordering.**
Using `loop.call_later()` directly (no `asyncio.sleep()`), schedule three callbacks with delays `2`, `0.5`, and `1` seconds, each printing its own label. Run the loop with `loop.run_forever()` (stopping it via `loop.call_later(2.5, loop.stop)`), and confirm the callbacks fire in delay order, not registration order.

**Exercise 3 — Thread-safety in practice.**
Spawn a raw OS thread (via `threading.Thread`) that calls `loop.call_soon_threadsafe()` to schedule a callback back on your main event loop after a short delay, printing a message from within the loop's own thread. Confirm it works reliably. Then (carefully, in a throwaway script) try calling the non-thread-safe `loop.call_soon()` directly from that same background thread instead, and research/observe what goes wrong or becomes unreliable.

**Exercise 4 — Try `uvloop`.**
Install `uvloop` (`pip install uvloop` — POSIX only), and benchmark a program that opens many concurrent TCP connections (reuse Chapter 16's client code) both with the default loop and with `loop_factory=uvloop.new_event_loop` passed to `asyncio.run()`. Compare wall-clock time.

**Exercise 5 (stretch) — Modernize policy-based code.**
Find or write a small snippet using the older `asyncio.set_event_loop_policy()` pattern to install a custom loop policy. Rewrite it using `loop_factory` instead, and confirm it produces the same effective behavior with less code and no deprecation warnings.

---

## 21.8 Common Pitfalls

- **Calling `loop.call_soon()` from a different OS thread.** This isn't thread-safe — always use `loop.call_soon_threadsafe()` when scheduling from outside the loop's own thread.
- **Assuming the event loop uses the wall clock for scheduling.** It uses a monotonic clock via `loop.time()` specifically to stay immune to system clock adjustments — don't try to correlate `loop.time()` directly with `time.time()`.
- **Reaching for the deprecated policy system in new code.** `asyncio.set_event_loop_policy()` and related APIs are deprecated as of Python 3.14 and scheduled for removal in 3.16 — use `loop_factory` on `asyncio.run()`/`asyncio.Runner` instead.
- **Assuming `uvloop` changes your code.** It's a drop-in replacement for the loop implementation only — your `async`/`await` code, `Task`s, and everything else stay identical; only the underlying scheduling engine changes.

---

## 21.9 Further Reading

- [Python docs: Event Loop](https://docs.python.org/3/library/asyncio-eventloop.html)
- [Python docs: `selectors`](https://docs.python.org/3/library/selectors.html)
- [Python docs: Policies (deprecated)](https://docs.python.org/3/library/asyncio-policy.html)
- [CPython issue #127949: Deprecate asyncio policy system](https://github.com/python/cpython/issues/127949) — the design discussion behind the `loop_factory` shift
- [uvloop documentation](https://uvloop.readthedocs.io/)

---

**Next: Chapter 22 — Debugging AsyncIO Applications** (debug mode, detecting blocking calls in the loop, slow-callback warnings, and the Python 3.14 `python -m asyncio ps`/`pstree` introspection tools from Chapter 2, finally put to real use).
