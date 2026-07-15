# Chapter 14: Async Context Managers

> Part 3 — Synchronization and Communication · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 14.1 Why Regular Context Managers Aren't Enough

A regular context manager's `__enter__`/`__exit__` methods are ordinary synchronous functions — they can't `await` anything. That's a real problem for resources whose setup or teardown is inherently asynchronous: opening a network connection, acquiring a lock that itself needs to `await`, gracefully closing a database connection pool, flushing a network buffer before disconnecting.

`async with` solves this with `__aenter__` and `__aexit__` — **coroutine methods** that can `await` freely during both setup and teardown.

---

## 14.2 The Protocol

```python
class Connection:
    async def __aenter__(self):
        print("Opening connection...")
        await asyncio.sleep(0.5)   # simulate an async handshake
        print("Connected")
        return self   # this becomes the `as` variable

    async def __aexit__(self, exc_type, exc, tb):
        print("Closing connection...")
        await asyncio.sleep(0.2)   # simulate an async graceful close
        print("Closed")
        return False   # False (or None) means: don't suppress any exception

async def main():
    async with Connection() as conn:
        print("Using the connection")

asyncio.run(main())
```

Output:
```
Opening connection...
Connected
Using the connection
Closing connection...
Closed
```

`__aexit__` receives the exception type, value, and traceback if the block raised — exactly like sync `__exit__`. Returning a **truthy** value from `__aexit__` **suppresses** that exception (it won't propagate past the `async with` block); returning `False`/`None` lets it propagate normally. This is identical in spirit to the sync protocol, just with `await` available throughout.

---

## 14.3 You've Already Been Using This

Every synchronization primitive from Chapter 12, plus `asyncio.timeout()` and `asyncio.TaskGroup`, is an async context manager under the hood. `async with lock:` is, mechanically, just:

```python
await lock.__aenter__()
try:
    ...  # your block
finally:
    await lock.__aexit__(*sys.exc_info())
```

Recognizing this retroactively explains a lot: `TaskGroup`'s "wait for all children, cancel siblings on failure" behavior (Chapter 8), and `asyncio.timeout()`'s deadline enforcement (Chapter 10), are both implemented as `__aenter__`/`__aexit__` logic — the exact same protocol you're learning to write yourself in this chapter.

---

## 14.4 The Easy Way: `contextlib.asynccontextmanager`

Writing a full class with `__aenter__`/`__aexit__` is verbose for simple cases. `contextlib.asynccontextmanager` mirrors the familiar `@contextmanager` decorator: write an async **generator** function with a single `yield` marking the boundary — code before `yield` is setup, code after (wrapped in `try`/`finally`) is teardown.

```python
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def connection():
    print("Opening connection...")
    await asyncio.sleep(0.5)
    print("Connected")
    try:
        yield "connection-object"   # this is the `as` value
    finally:
        print("Closing connection...")
        await asyncio.sleep(0.2)
        print("Closed")

async def main():
    async with connection() as conn:
        print(f"Using {conn}")

asyncio.run(main())
```

Same behavior as the class-based version in §14.2, far less boilerplate. The `try`/`finally` around `yield` is important: it guarantees the cleanup code runs even if the `async with` block raises an exception — or is cancelled, since (per Chapter 9) cancellation is just another exception traveling through that same `finally`.

---

## 14.5 A Real Pattern: A Connection Pool's `acquire()`

This is where async context managers earn their keep in real code. Building directly on Chapter 13's `Queue`:

```python
import asyncio
from contextlib import asynccontextmanager

class ConnectionPool:
    def __init__(self, size):
        self._queue = asyncio.Queue(maxsize=size)
        for i in range(size):
            self._queue.put_nowait(f"connection-{i}")

    @asynccontextmanager
    async def acquire(self):
        conn = await self._queue.get()   # waits if the pool is fully checked out
        try:
            yield conn
        finally:
            await self._queue.put(conn)  # always return it, even on failure/cancellation

async def worker(name, pool):
    async with pool.acquire() as conn:
        print(f"{name} using {conn}")
        await asyncio.sleep(1)
    print(f"{name} released its connection")

async def main():
    pool = ConnectionPool(size=2)
    async with asyncio.TaskGroup() as tg:
        for i in range(5):
            tg.create_task(worker(f"worker-{i}", pool))

asyncio.run(main())
```

Five workers, a pool of only 2 connections: at most 2 workers can be "using" a connection at any moment, and the `finally` inside `acquire()` guarantees every checked-out connection makes it back to the pool — whether the worker finishes normally, raises, or is cancelled mid-use. This is the same "bounded concurrent access" idea as Chapter 12's `Semaphore`, but tied to an actual resource object being handed out and reclaimed, rather than just a counter.

---

## 14.6 Suppressing Exceptions in `__aexit__`

Occasionally you deliberately want a context manager to swallow a *specific* kind of exception — for example, an "ignorable" error that shouldn't interrupt the caller's flow:

```python
from contextlib import asynccontextmanager

class IgnorableError(Exception):
    pass

@asynccontextmanager
async def suppress_ignorable():
    try:
        yield
    except IgnorableError as e:
        print(f"Suppressing: {e}")
        # implicitly returning None here from a generator-based CM still
        # tells asynccontextmanager to suppress — because we caught it
        # and did NOT re-raise.

async def main():
    async with suppress_ignorable():
        raise IgnorableError("this is fine")
    print("Execution continues normally after the block")

    try:
        async with suppress_ignorable():
            raise ValueError("this is NOT suppressed")
    except ValueError:
        print("ValueError correctly propagated")

asyncio.run(main())
```

With a generator-based context manager (`@asynccontextmanager`), suppression works by **catching the exception inside the generator and not re-raising it** — rather than explicitly returning `True`, as you would from a class-based `__aexit__`. Use this sparingly and deliberately: silently swallowing exceptions, even "ignorable" ones, can hide real bugs if applied too broadly.

---

## 14.7 Interaction with Cancellation

Since `__aexit__`/the code after `yield` runs as part of normal exception unwinding, everything from Chapter 9 applies directly: if the `async with` block is cancelled, cleanup still runs via `finally`. But if that cleanup itself contains a slow `await`, a **second** cancellation request arriving during cleanup can interrupt *that*, same as any other `await`. For cleanup that absolutely must complete uninterrupted (e.g., releasing a connection back to a pool, as in §14.5), consider wrapping just that critical `await` in `asyncio.shield()` (Chapter 9) if repeated cancellation during teardown is a realistic concern in your application.

---

## 14.8 Managing a Variable Number of Resources: `AsyncExitStack`

Sometimes you don't know upfront how many async context managers you'll need to open — say, a list of connections whose length is determined at runtime. Manually nesting `async with` blocks doesn't scale to a dynamic count. `contextlib.AsyncExitStack` solves this:

```python
from contextlib import AsyncExitStack
import asyncio

@asynccontextmanager
async def named_connection(name):
    print(f"Opening {name}")
    try:
        yield name
    finally:
        print(f"Closing {name}")

async def main():
    names = ["db", "cache", "queue"]

    async with AsyncExitStack() as stack:
        connections = [
            await stack.enter_async_context(named_connection(name))
            for name in names
        ]
        print(f"All open: {connections}")
        # ... use them ...
    # All three are guaranteed closed here, in REVERSE order of opening,
    # even if one of them had failed to open partway through the list.

asyncio.run(main())
```

`stack.enter_async_context(cm)` is the programmatic equivalent of writing `async with cm as x:` — but you can call it in a loop, an arbitrary number of times, and `AsyncExitStack` guarantees everything already entered gets properly exited (in reverse order, like nested `with` blocks would) when the stack itself exits — including if a *later* resource fails to open, ensuring the earlier ones still get cleaned up rather than leaking.

---

## 14.9 Hands-On Exercises

**Exercise 1 — Class-based timer.**
Write a class `AsyncTimer` with `__aenter__` (recording a start time) and `__aexit__` (computing and printing elapsed time). Use it as `async with AsyncTimer(): await asyncio.sleep(1)` and confirm it prints roughly `1.0` seconds.

**Exercise 2 — The generator-based equivalent.**
Rewrite `AsyncTimer` using `@asynccontextmanager` instead of a class. Compare line counts and readability.

**Exercise 3 — Build the connection pool.**
Implement the §14.5 `ConnectionPool` yourself, with `size=2` and 5 concurrent workers. Add a shared counter (protected by a `Lock`, Chapter 12) tracking the maximum number of connections in simultaneous use, and confirm it never exceeds 2.

**Exercise 4 — Deliberate suppression.**
Implement `suppress_ignorable()` from §14.6. Confirm an `IgnorableError` raised inside the block is caught and execution continues normally afterward, while a `ValueError` raised inside the same block still propagates out as expected.

**Exercise 5 — `AsyncExitStack` with a partial failure.**
Recreate the §14.8 example, but make the *second* connection's `__aenter__`-equivalent (the code before `yield`) raise an exception. Confirm: (a) the first connection, which already opened successfully, still gets properly closed; (b) the third connection is never opened at all, since the failure happened before reaching it.

**Exercise 6 (stretch) — Shielded cleanup.**
Modify the `ConnectionPool.acquire()` from Exercise 3 so that returning a connection to the queue involves an artificial `await asyncio.sleep(0.1)` "cleanup" step. Simulate a scenario where a worker holding a connection gets cancelled, and — during that cleanup `sleep` — gets cancelled *again* (call `.cancel()` twice in quick succession from the test driver). Observe whether the connection still reliably makes it back to the pool. Then wrap just the cleanup `await` in `asyncio.shield()` and confirm the connection is now returned reliably even under repeated cancellation.

---

## 14.10 Common Pitfalls

- **Defining `__enter__`/`__exit__` instead of `__aenter__`/`__aexit__`.** Using `async with` on an object missing the async-specific methods raises `TypeError: object does not support the asynchronous context manager protocol` — the two protocols are distinct, not interchangeable.
- **Forgetting the `try`/`finally` around `yield` in a generator-based context manager.** Without it, an exception (or cancellation) inside the `async with` block will skip your cleanup code entirely.
- **Mixing up suppression semantics.** For class-based CMs, returning a truthy value from `__aexit__` suppresses the exception — an easy detail to get backwards. For `@asynccontextmanager` functions, suppression happens by catching-and-not-re-raising inside the generator, a different mechanism entirely.
- **Doing slow, uninterruptible-feeling cleanup in `__aexit__` without considering repeated cancellation.** A second cancellation during cleanup can still interrupt it — use `asyncio.shield()` for genuinely critical teardown steps.
- **Manually nesting `async with` blocks for a variable-length list of resources.** This doesn't scale and gets unreadable fast — use `AsyncExitStack` instead.

---

## 14.11 Further Reading

- [Python docs: `contextlib.asynccontextmanager`](https://docs.python.org/3/library/contextlib.html#contextlib.asynccontextmanager)
- [Python docs: `contextlib.AsyncExitStack`](https://docs.python.org/3/library/contextlib.html#contextlib.AsyncExitStack)
- [PEP 492 – Coroutines with async and await syntax (defines `__aenter__`/`__aexit__`)](https://peps.python.org/pep-0492/)

---

**Next: Chapter 15 — Async Iterators and Generators** (`__aiter__`/`__anext__`, `async for`, async generator functions, and streaming data pipelines).
