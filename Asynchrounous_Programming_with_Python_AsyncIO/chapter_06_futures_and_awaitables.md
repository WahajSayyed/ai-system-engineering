# Chapter 6: Futures and Awaitables

> Part 2 — Core AsyncIO Building Blocks · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 6.1 Three Related, Distinct Things

By now you've used coroutines and Tasks, but there's a third concept underneath both of them that we've only glimpsed (in Chapter 3's toy event loop): the **Future**. Getting these three terms straight is essential, because they're used almost interchangeably in casual conversation but mean precisely different things:

| Concept | What it is | How you get one |
|---|---|---|
| **Coroutine** | A paused function body — the actual code and local state of an `async def` function | Calling a coroutine function: `coro = my_func()` |
| **Future** | A low-level placeholder object representing a value that *will exist eventually*, with no code attached to it — just a box waiting to be filled in | `loop.create_future()` |
| **Task** | A `Future` **subclass** that also drives a coroutine to completion, filling in its own "box" with that coroutine's eventual result | `asyncio.create_task(coro)` |

That middle row is the one most people never touch directly, but it's worth understanding because **`Task` literally inherits from `Future`** — every method you can call on a `Future` (`.done()`, `.result()`, `.exception()`, `.cancel()`, `.add_done_callback()`) also works on a `Task`, because a `Task` *is* a `Future`, plus the machinery to drive a coroutine and fill in its own result when that coroutine finishes.

```python
import asyncio

async def main():
    task = asyncio.create_task(asyncio.sleep(0.1, result="done"))
    print(isinstance(task, asyncio.Future))  # True — Task IS a Future

asyncio.run(main())
```

---

## 6.2 What a Bare Future Actually Is

A `Future` has no code of its own. It's just a state machine holding one of three states — **pending**, **done with a result**, or **done with an exception** — plus a list of callbacks to notify when it transitions out of "pending." Something else, external to the Future itself, is responsible for eventually calling `.set_result(value)` or `.set_exception(exc)` on it.

```python
import asyncio

async def set_after(fut: asyncio.Future, delay: float, value):
    await asyncio.sleep(delay)
    fut.set_result(value)   # someone else completes the Future

async def main():
    loop = asyncio.get_running_loop()
    fut = loop.create_future()               # starts out pending

    asyncio.create_task(set_after(fut, 1, "hello!"))

    print("Waiting on the future...")
    result = await fut       # suspends here until set_result() is called elsewhere
    print(f"Got: {result}")

asyncio.run(main())
```

Notice the shape of this: `main()` awaits `fut` directly — not a coroutine, not a Task — and it suspends exactly the way any `await` does, resuming only once something else (here, the `set_after` task) marks the Future done. This decoupling — "the thing waiting" doesn't need to know *what* will eventually complete the Future, only that something will — is the entire reason Futures exist as a separate concept from coroutines.

---

## 6.3 Why This Matters: Bridging Callback-Based APIs into `async`/`await`

This is the single most important practical use of raw Futures in everyday code: **wrapping a callback-style API so it becomes `await`-able.** Plenty of libraries — especially older ones, C extensions, or GUI toolkits — only offer a "register a callback, I'll call it when I'm done" interface, not a coroutine one. A `Future` is the standard bridge:

```python
import asyncio

def legacy_register_callback(on_done):
    """Imagine this is some external, non-async library."""
    # ... kicks off some operation, calls on_done(result) later ...
    ...

async def call_legacy_api():
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def callback(result):
        # If this callback fires on the SAME thread as the event loop:
        fut.set_result(result)
        # If it might fire from a DIFFERENT OS thread (common for C extensions),
        # use the thread-safe variant instead:
        # loop.call_soon_threadsafe(fut.set_result, result)

    legacy_register_callback(callback)
    return await fut

async def main():
    result = await call_legacy_api()
    print(result)
```

This pattern — create a Future, hand a closure that completes it into some non-async API, then `await` the Future — is exactly how libraries like `asyncpg`, GUI event loops, and low-level networking code bridge the callback world into the `async`/`await` world. You'll use this exact technique again in Chapter 18 (databases) and Chapter 17 (subprocesses) when wrapping lower-level APIs.

---

## 6.4 The Awaitable Protocol

"Awaitable" isn't a single class — it's a **protocol**: anything is awaitable if it implements `__await__()`, which must return an iterator. This is directly analogous to how "iterable" isn't one class either — anything with `__iter__()` qualifies.

Recall the `SleepAwaitable` we hand-built in Chapter 3:

```python
class SleepAwaitable:
    def __init__(self, seconds):
        self.seconds = seconds

    def __await__(self):
        yield ("sleep", self.seconds)
```

That `__await__` method is a minimal, from-scratch demonstration of the exact protocol that `Future.__await__` implements for real:

```python
# Conceptually, this is what Future.__await__ does:
def __await__(self):
    if not self.done():
        yield self             # hand control up to whatever is driving us
    return self.result()       # once resumed, "return" the completed value
```

Three categories of objects satisfy this protocol in practice, and all three are what the docs collectively call **"awaitables"**:

1. **Coroutines** (`async def` function results) — have `__await__` built in by the language itself.
2. **Tasks and Futures** — implement `__await__` as shown above.
3. **Any custom object** implementing `__await__` — like our `SleepAwaitable`, or library-provided awaitables you'll meet later (e.g., certain lock/semaphore acquisition objects).

---

## 6.5 Future Methods You'll Actually Use

| Method | Purpose |
|---|---|
| `fut.done()` | `True` once resolved (successfully, with exception, or cancelled) |
| `fut.result()` | The value, or re-raises the stored exception. Raises `InvalidStateError` if still pending. |
| `fut.exception()` | The stored exception, or `None` |
| `fut.set_result(value)` | Completes the Future successfully. **Raises `InvalidStateError` if called on an already-done Future** — a Future can only be completed once. |
| `fut.set_exception(exc)` | Completes the Future with a failure |
| `fut.cancel()` | Marks the Future as cancelled. Note: cancelling a *bare* Future just flips its state — there's no coroutine attached to actually interrupt, unlike cancelling a `Task` (Chapter 9). |
| `fut.add_done_callback(cb)` | Registers `cb(fut)` to run once the Future resolves — this is the literal low-level mechanism `Task` uses internally to know when to resume a coroutine waiting on something else. |

```python
import asyncio

async def main():
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    fut.add_done_callback(lambda f: print(f"Future resolved with: {f.result()}"))

    fut.set_result(42)
    try:
        fut.set_result(43)   # Futures can only be completed once
    except asyncio.InvalidStateError as e:
        print(f"Can't set again: {e}")

    await asyncio.sleep(0)   # give the done callback a chance to fire

asyncio.run(main())
```

---

## 6.6 Hands-On Exercises

**Exercise 1 — Manual Future, manual completion.**
Recreate the `set_after` example from §6.2 yourself. Then modify it so `set_after` calls `fut.set_exception(RuntimeError("failed"))` instead of `set_result`, and confirm that `await fut` in `main()` raises that exception exactly as if it had come from a normal coroutine call.

**Exercise 2 — Wrap a callback API.**
Write a small "legacy" function `legacy_register_callback(value, delay, on_done)` that uses `threading.Timer(delay, lambda: on_done(value)).start()` to call `on_done` after `delay` seconds from a **different thread**. Using the pattern from §6.3, write an `async def call_legacy_api(value, delay)` that bridges this into an `await`-able call — remembering to use `loop.call_soon_threadsafe()` since the callback fires from another thread. Confirm it works with several concurrent calls via `asyncio.gather()`.

**Exercise 3 — Prove Task IS a Future.**
Create a `Task`, and before awaiting it, call `.add_done_callback()` on it directly (instead of awaiting it) to print its result when it finishes. Confirm this callback-based approach works identically to how it would on a bare `Future`, and explain in a sentence why that's not a coincidence.

**Exercise 4 — Break it on purpose.**
Create a Future, call `.set_result(1)` on it, then try calling `.set_result(2)` again. Confirm you get `asyncio.InvalidStateError`, and write one sentence on why a Future is only allowed to be completed once (hint: think about what it would mean for `await fut` to be called from two different coroutines).

**Exercise 5 (stretch) — Build a "first one wins" combinator using Futures.**
Write a function `first_of(*coros)` that returns whichever of several coroutines finishes first, using a single shared `Future`: create tasks for each coroutine, have each one call `fut.set_result(value)` when it finishes (guard with `if not fut.done()`, since only the first caller should succeed), then `return await fut`. Compare your solution's behavior to `asyncio.wait(..., return_when=FIRST_COMPLETED)` from Chapter 7 — which is simpler to use correctly?

---

## 6.7 Common Pitfalls

- **Calling `set_result()` or `set_exception()` more than once on the same Future.** This raises `InvalidStateError` — a Future can only transition out of "pending" a single time.
- **Confusing cancelling a bare `Future` with cancelling a `Task`.** A bare Future has no coroutine to interrupt — cancelling it just flips a flag. Cancelling a `Task` (Chapter 9) actually throws `CancelledError` into the running coroutine at its next `await` point.
- **Forgetting `Task` methods and `Future` methods are the same methods.** If you've learned `.done()` / `.result()` / `.add_done_callback()` for one, you already know them for the other — `Task` doesn't reinvent this API, it inherits it.

---

## 6.8 Further Reading

- [Python docs: `asyncio.Future`](https://docs.python.org/3/library/asyncio-future.html)
- [Python docs: Awaitables](https://docs.python.org/3/library/asyncio-task.html#awaitables)

---

**Next: Chapter 7 — Running Things Concurrently: gather, wait, as_completed** (already written — see `chapter_07_gather_wait_as_completed.md`).
