# Chapter 9: Cancellation Deep Dive

> Part 2 — Core AsyncIO Building Blocks · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 9.1 What `task.cancel()` Actually Does

`task.cancel()` does **not** immediately stop anything. It's a *request*, not a command. Mechanically, here's what happens, tying directly back to the Task-driving mechanism from Chapter 3:

1. `task.cancel()` arranges for `asyncio.CancelledError` to be **thrown into** the wrapped coroutine — via `coro.throw(CancelledError())` — the next time the task gets a chance to run (i.e., at its *next* suspension point, not necessarily this instant).
2. If the coroutine is currently suspended at an `await`, that's where the exception lands. If it's in the middle of a long CPU-bound stretch with no `await` in sight, cancellation has to **wait** until the next `await` is reached — cancellation cannot interrupt code mid-statement.
3. If the coroutine doesn't catch `CancelledError` (the normal, expected case), it propagates out, the task transitions to "cancelled," and `task.cancelled()` becomes `True`.
4. `task.cancel()` itself returns `True` if a cancellation request was successfully scheduled, `False` if the task was already done — this return value tells you whether the *request* was accepted, not whether the task has actually stopped yet.

```python
import asyncio

async def worker():
    print("Working...")
    await asyncio.sleep(10)
    print("This never prints")

async def main():
    task = asyncio.create_task(worker())
    await asyncio.sleep(0.5)   # let it start

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Confirmed: task was cancelled")

    print(task.cancelled())  # True

asyncio.run(main())
```

---

## 9.2 `CancelledError` Is a `BaseException`, Not an `Exception`

This is a deliberate, important design choice, in place since Python 3.8: `asyncio.CancelledError` inherits from `BaseException`, **not** `Exception` — the same category as `SystemExit` and `KeyboardInterrupt`.

**Why it matters:** a broad `except Exception:` clause will **not** accidentally swallow a cancellation:

```python
async def worker():
    try:
        await asyncio.sleep(10)
    except Exception as e:
        print(f"This won't catch CancelledError: {e}")
        # CancelledError propagates right past this clause, as intended
```

But a bare `except:` (or an explicit `except BaseException:`) **will** catch it — and this is one of the most common real cancellation bugs:

```python
async def broken_worker():
    try:
        await asyncio.sleep(10)
    except:  # DANGEROUS: silently swallows CancelledError too
        print("Ignoring... everything?")
    print("Still running, uncancelled!")
```

If you write a bare `except:` (or catch `BaseException`) around awaited code and don't re-raise, you've broken the cancellation contract: the caller who called `.cancel()` believes the task is stopping, `task.cancelled()` will end up `False`, and — worse — anything relying on that cancellation actually happening (a `TaskGroup` waiting for its children, an `asyncio.timeout()` block expecting to move on) can hang or misbehave.

**The golden rule:** if you catch `CancelledError` for cleanup purposes, **always re-raise it** (or simply don't catch it at all, and use `finally` instead — see §9.3).

---

## 9.3 Cleanup with `try`/`finally`

The correct way to guarantee cleanup runs on cancellation is `finally`, not `except`:

```python
import asyncio

async def worker():
    print("Acquiring resource")
    try:
        await asyncio.sleep(10)
        print("Never reached if cancelled")
    finally:
        print("Releasing resource")   # ALWAYS runs, cancelled or not

async def main():
    task = asyncio.create_task(worker())
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

asyncio.run(main())
```

Output:
```
Acquiring resource
Releasing resource
```

`finally` blocks run during cancellation exactly like they do during any other exception unwind — this is standard Python exception handling, and `CancelledError` is just another exception traveling up the stack.

**One subtlety to watch for:** if your `finally` block itself contains an `await` (e.g., an async cleanup call), and `.cancel()` is called *again* while that cleanup is running, the new cancellation can interrupt your cleanup too. For cleanup that absolutely must not be interrupted, see `asyncio.shield()` next.

---

## 9.4 `asyncio.shield()`: Protecting an Operation from the Caller's Cancellation

`asyncio.shield(aw)` protects the *inner* awaitable from being cancelled **as a side effect of the outer coroutine being cancelled** — while still letting the outer `await` raise `CancelledError` to whoever's waiting on it.

This is a subtle but important distinction: **`shield()` does not stop the outer coroutine from seeing the cancellation.** It only prevents that cancellation from propagating *into* the shielded inner operation, letting it keep running independently in the background.

```python
import asyncio

async def critical_write():
    print("Starting critical write...")
    await asyncio.sleep(2)
    print("Critical write complete!")   # we want this to happen no matter what

async def handle_request():
    try:
        await asyncio.shield(critical_write())
    except asyncio.CancelledError:
        print("handle_request was cancelled, but the write keeps going in the background")
        raise

async def main():
    task = asyncio.create_task(handle_request())
    await asyncio.sleep(0.5)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # The write is still running in the background — give it time to finish
    # so we can observe it complete despite the outer cancellation.
    await asyncio.sleep(2)

asyncio.run(main())
```

Output:
```
Starting critical write...
handle_request was cancelled, but the write keeps going in the background
Critical write complete!
```

Notice `handle_request()` itself still gets cancelled (that's expected and usually desired — the caller wanted to move on), but `critical_write()` runs to completion anyway. A common real-world use: a client disconnects (cancelling the request handler), but a database write already in flight should still finish rather than leave data in an inconsistent state. If you need to know when the shielded operation actually finishes, keep a reference to it separately — `shield()` returns a `Task`-like wrapper, but the *original* work continues independently of whatever awaits the shield.

---

## 9.5 Cancellation Scopes: `cancelling()` and `uncancel()` (Python 3.11+)

Here's a real problem: what if a task is cancelled **for two different reasons at once** — say, a `TaskGroup` cancelling it because a sibling failed, *and* an `asyncio.timeout()` cancelling it because time ran out? How does code catching `CancelledError` know *which* cancellation it's looking at, so it can decide whether to convert it to a `TimeoutError`, let it propagate as a real cancellation, or something else?

Python 3.11 added a small API specifically to answer this, used internally by both `TaskGroup` and `asyncio.timeout()`:

- **`task.cancelling()`** — returns the number of *pending* cancellation requests on this task (can be more than 1, if cancelled multiple times for different reasons).
- **`task.uncancel()`** — decrements that count by one, used to say "I'm handling this particular cancellation myself; don't treat the task as externally cancelled just for my internal purposes."

This is exactly how `asyncio.timeout()` (Chapter 10) converts an internal cancellation into a clean `TimeoutError` without confusing outer code into thinking the whole task was cancelled from outside:

```python
import asyncio

async def mini_timeout(coro, seconds):
    """A simplified illustration of what asyncio.timeout() does internally."""
    task = asyncio.ensure_future(coro)
    handle = asyncio.get_running_loop().call_later(seconds, task.cancel)
    try:
        return await task
    except asyncio.CancelledError:
        if task.cancelling() > 0:
            task.uncancel()  # this cancellation was ours — claim it
            raise TimeoutError(f"Timed out after {seconds}s") from None
        raise  # some other cancellation was responsible — let it propagate as-is
    finally:
        handle.cancel()

async def main():
    try:
        await mini_timeout(asyncio.sleep(5), seconds=1)
    except TimeoutError as e:
        print(f"Caught: {e}")

asyncio.run(main())
```

You won't often need `cancelling()`/`uncancel()` directly in application code — but understanding that it exists explains *why* `asyncio.timeout()` and `TaskGroup` can distinguish "my own internal cancellation machinery" from "a genuine cancellation from outside," rather than every layer of cancellation looking identical and untraceable.

---

## 9.6 Hands-On Exercises

**Exercise 1 — Confirm the basic mechanics.**
Recreate the §9.1 example. Add a print statement showing `task.cancel()`'s return value both when called on a still-running task and when called again immediately after (once it's already done). Confirm the second call returns `False`.

**Exercise 2 — Cleanup with `finally`.**
Write a worker that prints `"acquired"` at the start, sleeps for 5 seconds, and prints `"released"` in a `finally` block. Cancel it after 1 second and confirm `"released"` still prints, proving cleanup ran despite cancellation.

**Exercise 3 — Break it on purpose.**
Rewrite Exercise 2's worker to catch the sleep with a bare `except:` instead of `finally`, printing `"swallowed!"` and *not* re-raising. Cancel it the same way, then check `task.cancelled()` afterward — confirm it's `False`, and that the coroutine actually kept running past where it "should" have stopped. Write 2–3 sentences on why this is dangerous in real code (think about what a `TaskGroup` or `asyncio.timeout()` waiting on this task would experience).

**Exercise 4 — Shield a critical operation.**
Recreate the §9.4 example yourself. Then modify it so `main()` does **not** wait around afterward for the shielded write to finish — just let `main()` return right after the outer task is cancelled. Run it and observe: does `"Critical write complete!"` still print, or does the whole program exit before it gets the chance? Explain what this tells you about the shielded task's lifetime relative to the program's.

**Exercise 5 (stretch) — Build your own mini-timeout.**
Implement `mini_timeout()` from §9.5 yourself (or use the provided version), and test it two ways: (a) with a coroutine that finishes *before* the timeout — confirm it returns normally, no exception; (b) with a coroutine that finishes *after* the timeout — confirm you get your custom `TimeoutError`, not a bare `CancelledError`. Add a print inside the `except` block showing `task.cancelling()`'s value right before you call `uncancel()`.

---

## 9.7 Common Pitfalls

- **Catching `CancelledError` (or a bare `except:`) without re-raising it.** This silently breaks the cancellation contract — the task never actually reports itself as cancelled, and anything waiting on that cancellation (a `TaskGroup`, a timeout, a caller's own logic) can hang, misbehave, or produce confusing warnings.
- **Assuming cancellation is instantaneous.** It only takes effect at the *next* `await` point. A long CPU-bound stretch between awaits delays cancellation exactly as long as it delays everything else (recall Chapter 3's cooperative-multitasking "curse").
- **Misunderstanding what `shield()` protects.** It protects the *inner* awaitable's execution from being cancelled as a side effect — it does **not** prevent the *outer* `await asyncio.shield(...)` from raising `CancelledError` if the outer coroutine itself gets cancelled. Don't expect `shield()` to make your calling coroutine immune to cancellation; only the shielded work is immune.
- **Forgetting `CancelledError` is a `BaseException` since Python 3.8.** Relying on `except Exception:` to "catch everything" will correctly let cancellations through — that's the intended behavior, not a gap to work around.

---

## 9.8 Further Reading

- [Python docs: Task Cancellation](https://docs.python.org/3/library/asyncio-task.html#task-cancellation)
- [Python docs: `asyncio.shield()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.shield)
- [Python docs: `Task.cancelling()` / `Task.uncancel()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.uncancel)

---

**Next: Chapter 10 — Timeouts** (`asyncio.timeout()`, `wait_for()`, and building robust, timeout-bounded network code on top of everything you just learned about cancellation).
