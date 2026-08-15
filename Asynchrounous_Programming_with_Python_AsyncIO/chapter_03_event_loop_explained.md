# Chapter 3: The Event Loop Explained

> Part 1 — Foundations · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 3.1 What Is an Event Loop, Really?

Every `asyncio` program has exactly one thing at its center: **the event loop**. Strip away all the syntax sugar (`async`, `await`, `Task`, `Future`) and the event loop is just an object running a simple, repeating cycle:

1. Look at the queue of callbacks that are ready to run *right now* — run them, one at a time, in order.
2. Check whether any scheduled timers are due (things registered with "call this in N seconds") — if so, move them into the ready queue.
3. If nothing is ready, ask the operating system: *"which of these open sockets/file descriptors have data ready, and for how long should I wait if none do?"* — block until something is ready or a timer is due.
4. Repeat, forever, until there's no more work left to do.

That's it. There is no magic parallelism, no hidden thread pool spinning up tasks. It's a single-threaded `while True:` loop with a to-do list. Everything you'll learn in this course — coroutines, tasks, gather, queues — is built as a layer of abstraction *on top of* this simple mechanism.

### An analogy

Think of the event loop as a single waiter at a restaurant with many tables (your coroutines). The waiter doesn't stand at one table waiting for food to cook (that would be `time.sleep`-style blocking). Instead: take table 3's order, walk it to the kitchen, immediately go check if table 1's food is ready, deliver it, check if table 5 needs anything, and so on — continuously round-robining between tables, spending zero time standing idle unless *literally every* table is waiting on the kitchen. One waiter, many tables in flight, because the waiter never blocks on any single table.

---

## 3.2 Cooperative Multitasking vs. Preemptive Multitasking

This is the single most important mechanical fact about asyncio:

> **A coroutine only ever gives up control at an `await` point. Never anywhere else.**

Compare the two models:

- **Preemptive multitasking (OS threads):** the operating system can pause a thread *at any instruction* and switch to another thread, without the thread's cooperation or knowledge. This is powerful but dangerous — any shared mutable state needs locks, because you can be interrupted mid-operation.
- **Cooperative multitasking (asyncio):** a coroutine runs uninterrupted from one `await` to the next. Nobody preempts it. It voluntarily "yields" control back to the event loop only when it hits an `await` on something that isn't immediately ready.

This has a huge practical consequence, both a blessing and a curse:

- **Blessing:** code between two `await` statements executes atomically with respect to other coroutines. You generally don't need locks to protect simple in-memory state shared between coroutines (unlike with threads).
- **Curse:** if a coroutine goes a long time *without* hitting an `await` — say, a tight CPU-bound loop, or a call to a blocking library like `time.sleep()` or `requests.get()` — it **freezes the entire event loop**. Every other coroutine, no matter how ready, simply waits, because there's nobody else to run it.

We'll come back to this curse specifically in Chapter 22 (debugging blocking calls), but it's worth internalizing now: **the event loop's fairness is only as good as your coroutines' willingness to `await` regularly.**

---

## 3.3 The Loop's Building Blocks

Conceptually, a real event loop (like `asyncio`'s default `SelectorEventLoop`) tracks three kinds of pending work:

| Structure | Purpose | Real asyncio equivalent |
|---|---|---|
| **Ready queue** | Callbacks that should run on the very next loop iteration | `loop.call_soon(callback)` |
| **Scheduled (delayed) calls** | Callbacks that should run at or after a specific future time, kept sorted by due-time (typically a heap) | `loop.call_later(delay, callback)` / `call_at(when, callback)` |
| **I/O watchers** | File descriptors (sockets, pipes) the loop is watching for "readable" / "writable" readiness, delegated to the OS | `loop.add_reader(fd, callback)` / `add_writer(fd, callback)` — backed by the `selectors` module (epoll on Linux, kqueue on BSD/macOS, IOCP on Windows) |

Each iteration of the loop, in rough pseudocode:

```python
def run_forever(self):
    while True:
        timeout = self._compute_timeout()   # time until the next scheduled call is due
        io_events = self._selector.select(timeout)   # blocks here, waiting on the OS
        self._process_io_events(io_events)            # queues up ready callbacks for ready fds
        self._run_due_scheduled_calls()               # moves due timers into the ready queue
        self._run_ready_callbacks()                    # actually executes everything that's ready
```

The only place this loop *blocks* the whole process is inside `selector.select(timeout)` — and that's fine, because it's blocking on "is any of this I/O ready yet," which is exactly the waiting asyncio exists to make productive use of. While it waits there, the OS is free to do other things; the moment any watched socket has data (or the timeout expires because a timer is due), `select()` returns and the loop springs back into action.

---

## 3.4 Where Do Coroutines and `await` Fit In?

A `Task` is a thin wrapper that *drives* a coroutine by repeatedly calling `coro.send(value)` (or `coro.throw(exc)`), exactly like the generator-driving trick from Chapter 2. Here's the actual protocol:

1. `Task` calls `coro.send(None)` to start/resume the coroutine.
2. The coroutine runs until it hits an `await` on some object.
3. If that object is immediately done (e.g., an already-resolved `Future`), execution just continues without returning control to the loop.
4. If it's **not** ready, the awaited object's `__await__` yields *itself* out through every layer of `await`, all the way back to `Task`. The `Task` catches this yield, registers a callback on the awaited object ("call `task.__step__` again when you're done"), and returns control to the event loop.
5. The event loop moves on to other ready work.
6. Later, when the awaited thing completes (a socket becomes readable, a timer fires), it invokes that callback, which does `loop.call_soon(task.__step__)` — scheduling the task to resume on the *next* ready-queue pass.
7. `Task` calls `coro.send(result)` again, and the coroutine picks up exactly where it left off, with `await` "returning" that result.

This is precisely why `await` an `asyncio.Future` yields *itself*: `Future.__await__` is implemented (conceptually) as:

```python
def __await__(self):
    if not self.done():
        yield self          # hands control up to whoever is driving this coroutine
    return self.result()
```

Every `await` you write compiles down to this same yield-and-resume dance, whether you're awaiting `asyncio.sleep()`, a network read, or another coroutine.

---

## 3.5 Build a Tiny Event Loop From Scratch

The best way to make this mechanical, rather than just conceptual, is to build a miniature version yourself. Below is a real, runnable ~40-line event loop that drives actual `async def` coroutines — no `asyncio` import required.

The trick: we define our own awaitable whose `__await__` yields a plain tuple describing what it's waiting for. Our tiny loop's entire job is to interpret those tuples.

```python
import heapq
import time


class SleepAwaitable:
    """A minimal stand-in for asyncio.sleep()."""
    def __init__(self, seconds):
        self.seconds = seconds

    def __await__(self):
        # Yielding hands control all the way up to whoever is driving this
        # coroutine (our TinyLoop below). This is the same mechanism
        # asyncio.Future.__await__ uses internally.
        yield ("sleep", self.seconds)


def tiny_sleep(seconds):
    return SleepAwaitable(seconds)


class TinyLoop:
    def __init__(self):
        self.ready = []      # [(coro, value_to_send), ...] — runnable right now
        self.sleeping = []   # heap of (wake_time, tiebreaker, coro)
        self._counter = 0

    def create_task(self, coro):
        self.ready.append((coro, None))

    def run(self):
        while self.ready or self.sleeping:
            if not self.ready:
                # Nothing runnable — fast-forward time to the next sleeper.
                wake_time, _, coro = heapq.heappop(self.sleeping)
                delay = max(0, wake_time - time.time())
                time.sleep(delay)
                self.ready.append((coro, None))

            coro, value = self.ready.pop(0)
            try:
                yielded = coro.send(value)
            except StopIteration:
                continue  # this coroutine has finished

            kind, *args = yielded
            if kind == "sleep":
                seconds = args[0]
                self._counter += 1
                heapq.heappush(
                    self.sleeping, (time.time() + seconds, self._counter, coro)
                )
            else:
                raise RuntimeError(f"Unknown yield from coroutine: {yielded}")


# --- Using it, with completely ordinary async/await syntax ---

async def worker(name, delay):
    print(f"[{time.strftime('%X')}] {name}: starting, will sleep {delay}s")
    await tiny_sleep(delay)
    print(f"[{time.strftime('%X')}] {name}: done")


loop = TinyLoop()
loop.create_task(worker("A", 1.0))
loop.create_task(worker("B", 0.3))
loop.create_task(worker("C", 0.6))
loop.run()
```

Run this and watch the output: **A, B, and C all start immediately** (they're all in the ready queue), and then **finish in the order B → C → A** — matching their sleep durations, not the order they were created in. That interleaving is the entire value proposition of async programming, and you just built the mechanism that makes it happen, from scratch, in plain Python.

Compare that to real `asyncio`: `asyncio.Task` is exactly this `TinyLoop`'s job, `asyncio.Future.__await__` is exactly `SleepAwaitable.__await__`'s job, and the real event loop's `call_later` + `selectors.select()` combo is exactly our `sleeping` heap + `time.sleep(delay)` combo — just generalized to also watch sockets, pipes, subprocesses, and signals, and implemented partly in C for speed (`_asyncio` accelerator module) rather than pure Python.

---

## 3.6 Two Kinds of "Waiting" the Real Loop Handles

Our toy loop only understands `sleep`. The real event loop generalizes this to two categories of things worth waiting on:

- **Time-based waiting** — `asyncio.sleep()`, timeouts — handled by the scheduled-calls heap, exactly like our toy version.
- **I/O-based waiting** — "tell me when this socket has bytes to read" — handled by asking the OS via the `selectors` module, which wraps the fastest readiness-notification mechanism your OS offers (`epoll` on Linux, `kqueue` on macOS/BSD, IOCP-based proactor on Windows) instead of naively looping and checking.

This is also why third-party event loop implementations exist: **`uvloop`** replaces the pure-Python `SelectorEventLoop` with one built on `libuv` (the same battle-tested C library Node.js uses), commonly cutting loop overhead significantly for I/O-heavy workloads, while keeping the exact same `asyncio` API you already write against. We'll benchmark this properly in Chapter 24 — the point for now is just that "the event loop" is a *replaceable* implementation detail behind a stable interface, not a fixed piece of magic.

---

## 3.7 Hands-On Exercises

**Exercise 1 — Predict, then run.**
Before running the `TinyLoop` example above, write down on paper what order you *predict* the print statements will appear in. Run it and compare. Were you right about the interleaving of "starting" messages vs. "done" messages?

**Exercise 2 — Add a "yield now" primitive.**
Extend `TinyLoop` and `SleepAwaitable` to support a zero-delay yield — the equivalent of real `asyncio.sleep(0)`, which just gives other ready tasks a turn without actually waiting on a timer. Add a `tiny_yield()` function returning an awaitable whose `__await__` does `yield ("yield_now",)`, and handle that case in `TinyLoop.run()` by putting the coroutine straight back on the ready queue.

**Exercise 3 — Break it on purpose.**
Add a fourth worker to the example that does *not* await anything — instead it runs a tight CPU-bound loop for ~2 seconds (e.g., a busy `for` loop doing arithmetic) before finishing. Run all four workers together and observe: do B and C, which have shorter sleep times, actually finish before the busy worker, even though they're "due" earlier? Explain what this demonstrates about cooperative multitasking's weak point.

**Exercise 4 — Compare against real asyncio.**
Rewrite the same three-worker example using real `asyncio.sleep()` and `asyncio.create_task()` / `asyncio.gather()`. Confirm you get the same B → C → A completion order. Time both versions with `time.perf_counter()` — they should be close, since our toy loop isn't doing anything real asyncio doesn't also do, just less efficiently.

**Exercise 5 (stretch) — Read the source.**
Skim CPython's `Lib/asyncio/tasks.py` (`Task.__step` method) and `Lib/asyncio/base_events.py` (`BaseEventLoop._run_once`). Identify the lines that correspond to: (a) our `ready` list, (b) our `sleeping` heap, (c) the `coro.send()` call. You won't understand every detail yet — the goal is just pattern-matching your toy loop onto the real one.

---

## 3.8 Common Pitfalls

- **Thinking the event loop is a separate thread.** It isn't. It's an ordinary Python object running in whatever thread called `asyncio.run()` (typically the main thread). "Concurrency" here means *interleaving*, not simultaneous execution.
- **Blocking the loop without realizing it.** Any blocking call — `time.sleep()`, a synchronous `requests.get()`, a slow synchronous file read, a heavy CPU-bound pure-Python computation — freezes *every* task, not just the one that called it, because nothing preempts it. There's no `await`, so there's no opportunity to hand control back.
- **Assuming fairness across arbitrarily long stretches.** Cooperative scheduling means "fair" only at the granularity of your `await` points. A coroutine that awaits frequently plays nicely; one that doesn't can starve everyone else, even sleepers whose timers are already due (as Exercise 3 demonstrates).

---

## 3.9 Further Reading

- David Beazley, *"A Curious Course on Coroutines and Concurrency"* (the classic talk this chapter's toy loop is directly inspired by)
- A. Jesse Jiryu Davis & Guido van Rossum, *"A Web Crawler With asyncio"* — the "500 Lines or Less" essay building a real (if small) asyncio-style crawler from first principles
- [Python docs: `asyncio` — Event Loop](https://docs.python.org/3/library/asyncio-eventloop.html)
- [Python docs: `selectors` module](https://docs.python.org/3/library/selectors.html)
- CPython source: `Lib/asyncio/base_events.py`, `Lib/asyncio/tasks.py`

---

**Next: Chapter 4 — Coroutines 101** (`async def`, `await`, coroutine objects vs. coroutine functions, `asyncio.run()`, and the beginner mistakes that trip up almost everyone at first).
