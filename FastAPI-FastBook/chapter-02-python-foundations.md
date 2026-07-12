# Chapter 2: Python Foundations FastAPI Assumes You Know

> Part I — Foundations · Chapter 2 of 28

Chapter 1 leaned on several things without fully unpacking them: that type hints "mean" something to FastAPI, that `async def` behaves differently from `def`, that `@app.get(...)` doesn't work like a typical decorator, and that Pydantic models aren't just fancy dataclasses. This chapter backfills all four, deliberately, before Chapter 3 goes deeper on routing. If you're already confident on these four topics, skim for the FastAPI-specific framing in each section and jump to the hands-on project.

## Learning Objectives

By the end of this chapter you will be able to:

- Read and write modern Python type hint syntax (`list[str]`, `dict[str, int]`, `X | None`) and explain what type hints do — and don't — enforce at runtime.
- Explain, mechanically, what happens when Python executes an `async def` function, what an `await` point actually does to program flow, and why the event loop is single-threaded.
- Explain what a decorator is at the AST/bytecode level (as sugar for a function call), and specifically how `@app.get(...)` differs from a "normal" decorator that wraps behavior.
- Explain why FastAPI chose Pydantic's `BaseModel` over the standard library's `@dataclass` for request/response modeling.

---

## 2.1 Type Hints: Metadata, Not Enforcement

The single most important thing to internalize about Python type hints, stated as plainly as possible: **Python itself does nothing with them at runtime.** They are metadata, stored on the function object, that some *other* tool — a type checker like `mypy`, an editor, or a framework like FastAPI — chooses to read and act on. The Python interpreter itself will happily ignore them:

```python
def add(a: int, b: int) -> int:
    return a + b

add("1", "2")   # runs without error, returns "11" (string concatenation)
```

Nothing crashed. `int` on `a` and `b` was never checked. This matters for FastAPI specifically because it clarifies what's actually going on when we say "FastAPI validates based on type hints" — FastAPI (via Pydantic) is *choosing* to read those hints and enforce them at the boundary of your application (parsing the request), not Python enforcing anything intrinsically. Once a value is inside your function body, ordinary Python rules apply again.

### Modern syntax you'll see throughout this curriculum

| Old style (pre-3.10, still valid) | Modern style (3.10+, used in this curriculum) | Meaning |
|---|---|---|
| `Optional[str]` | `str \| None` | a string, or `None` |
| `Union[int, str]` | `int \| str` | an integer or a string |
| `List[str]` | `list[str]` | a list of strings |
| `Dict[str, int]` | `dict[str, int]` | a dict of string keys to int values |

```python
def read_item(item_id: int, q: str | None = None) -> dict:
    ...
```

`q: str | None = None` reads as: *`q` is either a string or `None`, and if the caller omits it, default to `None`.* This exact pattern — an optional query parameter — is one you'll write dozens of times starting in Chapter 4, so it's worth being fluent in reading it now.

Generics (writing your own `class Paginated[T]`-style container) exist too, but they add real complexity for comparatively rare payoff at this stage — we'll pick them up properly in Chapter 16, once you have enough Pydantic experience for the payoff to be obvious rather than abstract.

## 2.2 async/await and the Event Loop, Mechanically

Chapter 1 introduced *why* ASGI matters (concurrency without one-worker-per-request). This section covers *how* that concurrency is actually implemented, because "the event loop" stops being mysterious the moment you see it as a simple, literal loop.

An `async def` function doesn't run when you call it — calling it produces a **coroutine object**, a paused, resumable unit of work. Nothing executes until something *drives* that coroutine forward, either `await`-ing it inside another coroutine, or handing it to the event loop via something like `asyncio.run()` or `asyncio.gather()`.

An `await` expression is a **yield point** — it says "I have nothing more to do until this finishes; the event loop may go do something else in the meantime." Critically, the event loop itself is single-threaded: at any given instant, only one piece of your Python code is actually executing. Concurrency here comes entirely from *voluntarily yielding at `await` points*, not from parallel execution.

```mermaid
sequenceDiagram
    participant Loop as Event Loop (single thread)
    participant A as Task A
    participant B as Task B

    Loop->>A: run until first await
    Note over A: A hits "await asyncio.sleep(1)"<br/>control returns to loop
    Loop->>B: run until first await
    Note over B: B hits "await asyncio.sleep(1)"<br/>control returns to loop
    Note over Loop: Both A and B are "sleeping"<br/>concurrently — loop is idle-ish,<br/>free to do other work
    Loop->>A: A's sleep is done, resume
    Loop->>B: B's sleep is done, resume
```

This is why `asyncio.sleep(1)` and `time.sleep(1)` look similar but behave completely differently in async code: `asyncio.sleep` is itself an `await`-able operation that yields control back to the loop, so *other* tasks can make progress during that second. `time.sleep`, by contrast, is a plain blocking call with no concept of the event loop — if you call it inside an `async def` function, you freeze the *entire* single thread, including every other task that was supposed to be running concurrently. There is no partial credit here: one `time.sleep()` call in the wrong place stalls everything, not just its own task. You'll reproduce exactly this failure on purpose in the hands-on project below, because reading about it is a poor substitute for watching your "concurrent" code take three times longer than expected.

**The rule this gives you, concretely:** inside an `async def` path operation, every I/O operation (database calls, HTTP calls to other services, file I/O) needs an async-native library and an `await` in front of it, or it needs to be pushed onto a thread (`asyncio.to_thread(...)`) so it doesn't block the loop. You'll apply this rule for real with a database driver starting in Chapter 9.

## 2.3 Decorators, and Why `@app.get(...)` Is Slightly Unusual

A decorator is syntax sugar. This:

```python
@my_decorator
def say_hello():
    print("hello")
```

is exactly equivalent to this:

```python
def say_hello():
    print("hello")

say_hello = my_decorator(say_hello)
```

A "normal" decorator typically *wraps* the function — it returns a new function that does something extra and then calls the original:

```python
import time

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.4f}s")
        return result
    return wrapper

@timed
def slow_add(a, b):
    time.sleep(0.5)
    return a + b

slow_add(2, 3)   # prints timing, then returns 5
```

Here's the part that trips people up about FastAPI specifically: `@app.get("/items/{item_id}")` is a **decorator factory** — `app.get("/items/{item_id}")` is a function call that *returns* a decorator, which is then applied to `read_item`. That part is normal. What's unusual is what the returned decorator *does*: unlike `timed` above, it does not wrap `read_item` in a new function that adds behavior on every call. Instead, it registers `read_item` (path, function, metadata) into Starlette's internal routing table, and then **returns the original function completely unchanged**.

You can see this for yourself by building a deliberately simplified stand-in:

```python
class TinyRouter:
    def __init__(self):
        self.routes = {}

    def get(self, path: str):
        def decorator(func):
            self.routes[("GET", path)] = func
            return func          # <-- unchanged, not wrapped!
        return decorator


router = TinyRouter()

@router.get("/hello")
def say_hello():
    return {"message": "hi"}

print(router.routes)     # {('GET', '/hello'): <function say_hello at 0x...>}
print(say_hello())       # {'message': 'hi'}  -- calling it directly works exactly as if undecorated
```

This is the actual shape of what `@app.get(...)` does (Starlette's real implementation is more involved, but the *decorator mechanics* are exactly this). The practical consequence: you can call your own path operation functions directly, in a test or from other code, and they'll behave like ordinary functions — because that's precisely what they still are. The decorator's job was purely to make FastAPI *aware* of them, not to change how they run.

## 2.4 Dataclasses vs Pydantic Models

The standard library's `@dataclass` and Pydantic's `BaseModel` look similar at a glance — both let you declare a class as a set of typed fields — but they solve different problems, and the difference is exactly why FastAPI is built on Pydantic rather than plain dataclasses.

```python
from dataclasses import dataclass

@dataclass
class ProductDC:
    name: str
    price: float

p = ProductDC(name="Widget", price="9.99")   # no error raised
print(p.price, type(p.price))                 # 9.99  <class 'str'>   -- never coerced!
```

`@dataclass` gives you a convenient `__init__`, `__repr__`, and `__eq__` generated from your field declarations — but the type hints on those fields are, per section 2.1, pure metadata. Nothing checks or converts `price` at construction time; you asked for a `float` and silently got a `str`.

```python
from pydantic import BaseModel

class ProductPD(BaseModel):
    name: str
    price: float

p = ProductPD(name="Widget", price="9.99")
print(p.price, type(p.price))                 # 9.99  <class 'float'>  -- coerced to float

ProductPD(name="Widget", price="not-a-number")
# raises pydantic.ValidationError:
#   price
#     Input should be a valid number, unable to parse string as a number
```

Pydantic actively *enforces* the type hints at construction time — coercing compatible values (a numeric string into a `float`) and rejecting incompatible ones with a structured, informative error. This single behavioral difference is the entire reason FastAPI's request bodies are Pydantic models rather than dataclasses: a dataclass would happily accept a malformed request body and hand your route function garbage; a Pydantic model rejects it before your code ever sees it, with a client-facing error explaining exactly what was wrong. We'll go deep on Pydantic's validation machinery in Chapter 5 — this section's only job was to establish *why* it exists at all, in contrast to the tool it's most often confused with.

---

## Hands-On Project: Feeling the Event Loop

This project has three parts, each a small standalone script. Run all three and actually watch the timings — the point is to feel the difference, not just read about it.

### Part A — Sequential blocking baseline

`sync_demo.py`:

```python
import time


def blocking_task(name: str, duration: float) -> None:
    print(f"[{name}] starting (blocking)")
    time.sleep(duration)
    print(f"[{name}] done (blocking)")


def main():
    start = time.perf_counter()
    blocking_task("A", 1)
    blocking_task("B", 1)
    blocking_task("C", 1)
    print(f"Sync total: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
```

Run it: `python sync_demo.py`. Expect roughly **3 seconds** total — each task fully blocks before the next starts.

### Part B — Real concurrency with `asyncio.gather`

`async_demo.py`:

```python
import asyncio
import time


async def async_task(name: str, duration: float) -> None:
    print(f"[{name}] starting (async)")
    await asyncio.sleep(duration)
    print(f"[{name}] done (async)")


async def main():
    start = time.perf_counter()
    await asyncio.gather(
        async_task("A", 1),
        async_task("B", 1),
        async_task("C", 1),
    )
    print(f"Async total: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it: `python async_demo.py`. Expect roughly **1 second** total — all three "waits" overlap, because `asyncio.sleep` yields control back to the loop instead of occupying it.

### Part C — The trap: blocking code inside `async def`

`broken_async_demo.py`:

```python
import asyncio
import time


async def bad_task(name: str, duration: float) -> None:
    print(f"[{name}] starting (blocking call inside async def!)")
    time.sleep(duration)          # BUG: this is NOT awaited, and blocks the whole loop
    print(f"[{name}] done")


async def main():
    start = time.perf_counter()
    await asyncio.gather(
        bad_task("A", 1),
        bad_task("B", 1),
        bad_task("C", 1),
    )
    print(f"Broken async total: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it: `python broken_async_demo.py`. Despite using `async def` and `asyncio.gather` exactly like Part B, expect roughly **3 seconds again** — matching the sequential baseline, not the concurrent version. This is the exact bug described in section 2.2 and previewed in Chapter 1: `time.sleep` has no idea an event loop exists, so it blocks the single thread outright, and every "concurrent" task behind it simply waits its turn. Keep this script — you'll want to remember what this failure mode looks like when we hit real (and less obvious) versions of it with database drivers in Chapter 9.

---

## Practice Exercises

**Exercise 2.1 — Predict before you run.**
Without running it, write down the printed order of lines for this program:

```python
import asyncio

async def worker(name: str, delay: float):
    print(f"{name} starting")
    await asyncio.sleep(delay)
    print(f"{name} finished")

async def main():
    await asyncio.gather(
        worker("A", 0.3),
        worker("B", 0.1),
        worker("C", 0.2),
    )

asyncio.run(main())
```

Then run it and check yourself. Explain any part of the ordering that surprised you.

**Exercise 2.2 — Fix a real blocking bug.**
This coroutine has the Part C bug, but disguised as a "real" HTTP call using the popular (synchronous) `requests` library:

```python
import asyncio
import requests   # synchronous library!

async def fetch_status(url: str) -> int:
    response = requests.get(url)
    return response.status_code

async def main():
    results = await asyncio.gather(
        fetch_status("https://example.com"),
        fetch_status("https://example.org"),
    )
    print(results)

asyncio.run(main())
```

Identify the bug, and fix it two different ways: (a) using `asyncio.to_thread(...)` to push the blocking call off the event loop without changing libraries, and (b) using an async-native HTTP client (`httpx.AsyncClient`) instead. Which fix would you actually reach for in a real FastAPI codebase, and why?

**Exercise 2.3 — `def` vs `async def` in a path operation.**
FastAPI lets you write a path operation as either `def` or `async def`, and runs plain `def` handlers in a thread pool behind the scenes so they don't block the event loop. Given that, answer: if you have a CPU-heavy (not I/O-heavy) route handler — say, one that does a large in-memory computation with no `await` in sight — is `def` or `async def` the safer default, and why? (Hint: think about what a thread pool buys you versus what it can't fix.)

**Exercise 2.4 — Write your own decorator.**
Write a decorator called `retry(times: int)` that re-runs a function up to `times` times if it raises an exception, then re-raises the last exception if all attempts fail. Apply it to a function that fails on its first two calls and succeeds on the third (you can simulate this with a counter in a closure or a mutable default argument for the exercise). This is unrelated to FastAPI's routing decorators specifically, but it's the same underlying mechanism you just took apart in section 2.3.

**Exercise 2.5 (stretch) — Dataclass to Pydantic.**
Take this dataclass:

```python
from dataclasses import dataclass

@dataclass
class SignupRequest:
    email: str
    age: int
    newsletter_opt_in: bool = False
```

Rewrite it as a Pydantic `BaseModel`. Then construct an instance with `age="25"` (a string) for both versions and compare what happens. Separately, try constructing both with `age="not-a-number"` and compare the failure modes.

---

## Solutions & Discussion

<details>
<summary>Exercise 2.1</summary>

Output:
```
A starting
B starting
C starting
B finished
C finished
A finished
```
All three `print("... starting")` lines fire immediately and in order, because `asyncio.gather` starts each coroutine before any of them hits their first `await`. Once all three are "sleeping," they resume in the order their delays expire — B (0.1s) first, then C (0.2s), then A (0.3s) — regardless of the order they were passed to `gather`. The common surprise: people expect A/B/C to finish in the order they were *listed*, when actually completion order tracks delay duration, not gather argument order.
</details>

<details>
<summary>Exercise 2.2</summary>

The bug: `requests.get(url)` is a synchronous, blocking network call sitting inside `async def fetch_status`. Even though it's `await`-ed as a coroutine at the call site (`await asyncio.gather(...)`), the body of `fetch_status` itself never actually yields to the event loop — it blocks the whole thread for the full duration of each HTTP request, exactly like Part C's `time.sleep`. The two requests will run one after another, not concurrently, despite `gather`.

**Fix (a) — offload to a thread:**
```python
async def fetch_status(url: str) -> int:
    response = await asyncio.to_thread(requests.get, url)
    return response.status_code
```
This keeps `requests` but runs the blocking call in a separate thread, freeing the event loop while it waits.

**Fix (b) — use an async-native client:**
```python
import httpx

async def fetch_status(url: str) -> int:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.status_code
```
In a real FastAPI codebase, (b) is generally preferred when an async-native library exists for the dependency you're calling (database driver, HTTP client, etc.) — it's more efficient than spinning up threads and integrates more naturally with the rest of an async codebase. `asyncio.to_thread` (fix a) is the right tool specifically when no async-native alternative exists, or when wrapping legacy/third-party sync code you don't control.
</details>

<details>
<summary>Exercise 2.3</summary>

For a CPU-heavy handler with no I/O and no `await`, `def` is the safer default. Here's why: FastAPI runs plain `def` path operations in a thread pool, which prevents them from blocking the *main* event loop — other concurrent requests handled by `async def` routes, or by the loop itself, keep making progress. If you instead wrote that same CPU-heavy logic as `async def` with no `await` anywhere in it, it would run directly on the event loop and monopolize it for the entire computation, stalling every other concurrent request on that worker — the exact Part C failure mode, just from CPU work instead of `time.sleep`.

The caveat worth internalizing: a thread pool doesn't make CPU-bound work *faster* — Python's GIL still limits true CPU parallelism across threads for pure-Python code. `def` protects the event loop from being blocked by that work; it doesn't parallelize the work itself. For genuinely CPU-bound heavy lifting at scale, the real fix is usually offloading to a process pool or a separate worker service — a topic that resurfaces in Chapter 22 (task queues) and Chapter 26 (microservices).
</details>

<details>
<summary>Exercise 2.4</summary>

```python
import functools


def retry(times: int):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    print(f"attempt {attempt} failed: {exc}")
            raise last_exc
        return wrapper
    return decorator


_calls = {"count": 0}

@retry(times=3)
def flaky():
    _calls["count"] += 1
    if _calls["count"] < 3:
        raise ValueError(f"failed on call {_calls['count']}")
    return "success"


print(flaky())
```

Note the shape: `retry(times: int)` is itself a decorator *factory*, exactly like `app.get("/path")` from section 2.3 — calling `retry(times=3)` returns the actual decorator, which is then applied to `flaky`. Unlike `@app.get`, though, this decorator *does* wrap the function's behavior (adding retry logic around each call) rather than just registering it somewhere and returning it unchanged — which is the more common decorator pattern outside of routing registration.
</details>

<details>
<summary>Exercise 2.5</summary>

```python
from pydantic import BaseModel

class SignupRequest(BaseModel):
    email: str
    age: int
    newsletter_opt_in: bool = False
```

With `age="25"`: the dataclass version stores the literal string `"25"` in `.age` with no complaint — `type(instance.age)` is `str`. The Pydantic version coerces it to the integer `25` — `type(instance.age)` is `int`. Downstream code that does `age + 1` will work correctly on the Pydantic version and crash (or silently misbehave, if it does string concatenation instead) on the dataclass version.

With `age="not-a-number"`: the dataclass version again accepts it silently, storing the unusable string. The Pydantic version raises a `ValidationError` immediately at construction, naming the `age` field and explaining that the input couldn't be parsed as an integer — the failure surfaces at the boundary, in a form a client can act on, instead of deep inside your business logic later.
</details>

---

## Chapter Summary

- Type hints are inert at the language level — Python enforces nothing on its own. FastAPI's entire value proposition rests on choosing to read and enforce them, via Pydantic, at the request boundary.
- `async`/`await` concurrency is cooperative and single-threaded: tasks only yield control at genuine `await` points. A blocking call inside `async def` (like `time.sleep` or a sync HTTP library) freezes the whole loop, not just its own task — and this is the single most common real-world FastAPI performance bug.
- `@app.get(...)` is a decorator factory whose returned decorator registers your function into a routing table and returns it unmodified — unlike most decorators, it doesn't change how the function behaves when called directly.
- Pydantic's `BaseModel` actively validates and coerces at construction time; `@dataclass` does not. That single difference is why FastAPI request/response models are Pydantic models, not dataclasses.

**Next:** Chapter 3 puts routing itself under the microscope — path operations, `APIRouter`, path-matching order, and the first real multi-route API you'll build in this curriculum.
