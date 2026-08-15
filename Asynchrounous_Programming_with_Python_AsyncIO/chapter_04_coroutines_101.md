# Chapter 4: Coroutines 101

> Part 1 — Foundations · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 4.1 Coroutine Functions vs. Coroutine Objects

This is the single most common beginner trip-up in all of asyncio, so we start here.

```python
async def greet():
    print("Hello")
```

`greet` is a **coroutine function** — a special kind of function marked by `async def`. Just like calling a generator function doesn't run its body, **calling a coroutine function does not execute anything inside it.** It simply constructs and returns a **coroutine object**.

```python
>>> async def greet():
...     print("Hello")
...
>>> result = greet()
>>> print(result)
<coroutine object greet at 0x7f2a3c0b1ac0>
>>> print("Hello" in "already printed?")  # nothing was printed above!
```

Nothing was printed, because `greet()` only *built* a coroutine object — it didn't run it. To actually execute the body, that object needs to be **driven**: either directly `await`-ed inside another coroutine, or handed to something that drives coroutines for you, like `asyncio.run()` or `asyncio.create_task()`.

If you create a coroutine object and never do either of those things, Python will eventually garbage-collect it and print a warning:

```
RuntimeWarning: coroutine 'greet' was never awaited
```

This is one of the most common bugs beginners hit — and it fails *silently* in the sense that your program doesn't crash, it just quietly never runs that code. Always take this warning seriously; it means you wrote a coroutine call and forgot to do anything with it.

**Fixing it:**

```python
import asyncio

async def greet():
    print("Hello")

async def main():
    await greet()  # this is what actually runs the body

asyncio.run(main())
```

You can confirm the distinction with `inspect`:

```python
import inspect

print(inspect.iscoroutinefunction(greet))  # True — greet is a coroutine function
coro = greet()
print(inspect.iscoroutine(coro))           # True — coro is a coroutine object
coro.close()  # avoid the "never awaited" warning since we're not running it here
```

---

## 4.2 The `await` Expression

`await` does two things: it **runs** whatever's on its right-hand side, and it **suspends the current coroutine** if that thing isn't immediately done, handing control back to whatever is driving this coroutine (as we saw mechanically in Chapter 3).

A few rules:

- `await` is only legal **inside an `async def` body**. Using it at module level outside of a coroutine is a `SyntaxError` in ordinary scripts (though the `python -m asyncio` REPL, and Jupyter notebooks, special-case this and allow top-level `await` for convenience).
- The right-hand side of `await` must be an **awaitable**: a coroutine object, an `asyncio.Task`, an `asyncio.Future`, or any object implementing `__await__`.
- `await`-ing a coroutine that raises an exception causes that exception to propagate out of the `await` expression, exactly as if it were a normal function call — ordinary `try`/`except` works as you'd expect.

```python
async def might_fail():
    raise ValueError("something went wrong")

async def main():
    try:
        await might_fail()
    except ValueError as e:
        print(f"Caught it: {e}")

asyncio.run(main())
```

---

## 4.3 `asyncio.run()`: Your Program's Entry Point

`asyncio.run(coro)` is the standard way to kick off the *top level* of an async program. It does three things:

1. Creates a brand-new event loop.
2. Runs the given coroutine on it until it completes, returning its result (or raising its exception).
3. Cleans up: cancels any leftover tasks, shuts down async generators, and closes the loop.

```python
import asyncio

async def main():
    print("Running")
    return 42

result = asyncio.run(main())
print(result)  # 42
```

**Important constraint:** `asyncio.run()` is meant to be called **once**, from **synchronous** top-level code — not from inside another coroutine, and not while another event loop is already running in the current thread. Doing so raises:

```
RuntimeError: asyncio.run() cannot be called from a running event loop
```

```python
async def inner():
    return "done"

async def main():
    # WRONG: we're already inside a running loop here
    result = asyncio.run(inner())   # raises RuntimeError
    return result

asyncio.run(main())
```

If you find yourself wanting to call `asyncio.run()` again from within async code, that's a sign you actually just want `await inner()` instead — there's no need to spin up a second event loop.

**A note for notebook users:** Jupyter (7.0+) and IPython support top-level `await` directly in a cell, and manage the running loop for you, so you generally don't call `asyncio.run()` yourself there — you can just write `await main()` in a cell. If you ever see `RuntimeError: asyncio.run() cannot be called from a running event loop` in a notebook, that's why: the notebook already has a loop running, and `asyncio.run()` refuses to nest.

**For more advanced control** — running several independent top-level async operations in sequence while sharing loop-level state like `contextvars` — Python 3.11+ provides `asyncio.Runner` as a context manager:

```python
with asyncio.Runner() as runner:
    runner.run(operation_one())
    do_some_sync_work()
    runner.run(operation_two())
```

We'll use this pattern later when combining async code with blocking sections in more complex applications.

---

## 4.4 Sequential `await` Is Not Concurrency (Yet)

A subtle but critical point for this stage of the course: writing multiple `await` statements one after another runs them **one at a time, in order** — not concurrently.

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
    a = await fetch("A", 1)
    b = await fetch("B", 1)
    elapsed = time.perf_counter() - start
    print(f"Total: {elapsed:.2f}s")   # ~2.0s, NOT ~1.0s

asyncio.run(main())
```

Even though this is "async" code, `await fetch("B", 1)` doesn't even *start* until `await fetch("A", 1)` has fully finished — there's no overlap. `await` on its own just means "pause here until this one thing finishes"; it does not automatically run things in parallel with anything else. To actually get the overlap we demonstrated back in Chapter 1 (5 downloads finishing in ~1 second instead of ~5), you need to explicitly schedule multiple **Tasks** — which is exactly the subject of Chapter 5, immediately next.

This distinction — `await` for *sequencing*, versus `Task`/`gather` for *concurrency* — is worth sitting with before moving on, because conflating the two is the second most common beginner misunderstanding after the "forgot to await" issue in §4.1.

---

## 4.5 Hands-On Exercises

**Exercise 1 — Trigger and fix the warning.**
Write a coroutine function `say_hi()` that just prints a message. Call `say_hi()` at module level (not inside `asyncio.run`) without awaiting it, and run the script. Confirm you see the `RuntimeWarning: coroutine ... was never awaited`. Then fix it properly using `asyncio.run()`.

**Exercise 2 — Inspect before you run.**
Using `inspect.iscoroutinefunction()` and `inspect.iscoroutine()`, write a small script that proves to yourself: (a) an `async def` function is a coroutine function even before it's called, and (b) calling it produces a coroutine object, not a result — print the `type()` of each to see this directly.

**Exercise 3 — Measure the sequential-await trap.**
Using the `fetch()` example from §4.4, extend it to sequentially `await` **four** calls of `fetch(name, 1)`. Predict the total time before running it, then run it and confirm. Write one sentence explaining, in your own words, why sequential `await` doesn't overlap the waits.

**Exercise 4 — Reproduce the nested-loop error.**
Write a coroutine `main()` that calls `asyncio.run(some_other_coroutine())` from inside itself, and run the whole thing via `asyncio.run(main())`. Confirm you get `RuntimeError: asyncio.run() cannot be called from a running event loop`, and then fix it by replacing the inner `asyncio.run(...)` with a plain `await ...`.

**Exercise 5 — Exception propagation.**
Write a coroutine `risky(n)` that raises a `ZeroDivisionError` if `n == 0`, and otherwise returns `10 / n`. In `main()`, call it in a loop for `n` in `[5, 2, 0, 1]` inside a `try`/`except` around each `await risky(n)`, printing either the result or "caught error for n=0". Confirm the loop continues past the exception rather than crashing the whole program.

**Exercise 6 (stretch) — Try the REPL.**
Run `python -m asyncio` to open the built-in async REPL, and directly type `await asyncio.sleep(2)` followed by `print("back")` without wrapping anything in `asyncio.run()`. Note how the REPL lets you `await` at the top level — this is a special affordance of that REPL (and of Jupyter), not something available in ordinary `.py` scripts.

---

## 4.6 Common Pitfalls

- **Calling a coroutine function and forgetting to await/schedule it.** This produces a coroutine object that silently never runs, eventually surfacing only as a `RuntimeWarning` — easy to miss in noisy output. If code you expected to run clearly didn't, check for this first.
- **Believing sequential `await` calls run concurrently just because the function is `async def`.** `async` describes *how a function can pause*, not *how many things run at once*. Concurrency requires `Task`s (Chapter 5).
- **Calling `asyncio.run()` from inside already-running async code**, including inside notebooks that already manage a loop for you. If you're already inside a coroutine, you want `await`, not another `asyncio.run()`.

---

## 4.7 Further Reading

- [Python docs: Coroutines and Tasks](https://docs.python.org/3/library/asyncio-task.html)
- [Python docs: `asyncio.run()`](https://docs.python.org/3/library/asyncio-runner.html#asyncio.run)
- [PEP 492 – Coroutines with async and await syntax](https://peps.python.org/pep-0492/)

---

**Next: Chapter 5 — Tasks: Scheduling Concurrent Work** (`asyncio.create_task`, task lifecycle, and finally getting that real concurrent speedup).
