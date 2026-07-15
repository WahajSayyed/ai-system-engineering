# Chapter 25: Production Design Patterns

> Part 6 — Advanced Patterns and Design · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 25.1 From Building Blocks to Named Patterns

This chapter assembles primitives from across the course — `Semaphore`, retry loops, `TaskGroup`, `Queue` — into named, reusable production patterns you'll recognize from real systems: rate limiting, retries with backoff, circuit breakers, graceful shutdown, and connection pooling.

---

## 25.2 Rate Limiting: A Proper Token Bucket

Chapter 19 introduced a simple fixed-interval rate limiter. A **token bucket** is the more common production pattern, because it allows controlled **bursting**: a bucket holds up to `capacity` tokens, refilling continuously at `rate` tokens/second, and each call consumes one token — waiting only when the bucket is actually empty, rather than pacing every single call to a fixed interval regardless of whether headroom exists.

```python
import asyncio
import time

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate            # tokens added per second
        self.capacity = capacity    # maximum tokens the bucket can hold
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens < 1:
                wait = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0
            else:
                self.tokens -= 1
```

With `capacity=5, rate=2`, the first 5 calls go through immediately (burst), and only afterward does it settle into pacing at 2/second — a meaningfully different (and usually more desirable) behavior than a strict fixed-interval limiter, which would pace every single call regardless of built-up headroom.

---

## 25.3 Retries with Backoff and Jitter, as a Reusable Decorator

Rather than inlining retry logic per call site (as Chapter 19 did), wrap it once as a decorator applicable to any async function:

```python
import asyncio
import functools
import random

def retry(max_attempts=3, base_delay=0.5, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    # "Full jitter": a random delay between 0 and the exponential
                    # ceiling, rather than the exponential value itself — this
                    # spreads out retries from many clients instead of letting
                    # them all retry in lockstep after a shared outage.
                    delay = random.uniform(0, base_delay * (2 ** (attempt - 1)))
                    print(f"Attempt {attempt} failed ({e}); retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=4, base_delay=0.5, exceptions=(ConnectionError,))
async def flaky_call():
    if random.random() < 0.4:
        raise ConnectionError("transient failure")
    return "success"
```

The "full jitter" approach — a random delay *up to* the exponential ceiling, rather than the exponential value itself — is directly recommended by cloud providers' own architecture guidance specifically to avoid synchronized "retry storms," where many clients that failed at the same moment (e.g., during a shared outage) would otherwise all retry at the exact same intervals, hammering the recovering service in synchronized waves.

---

## 25.4 Circuit Breakers

A rate limiter and a retry decorator both still **attempt** every call. A **circuit breaker** goes a step further: after enough consecutive failures, it stops attempting calls entirely for a cooldown period, **failing fast** instead — protecting your own system from wasting time on calls that are very likely to fail anyway, and protecting the downstream system from being hammered further during an outage.

Three states:

- **CLOSED** — normal operation, calls go through
- **OPEN** — too many recent failures; calls fail immediately without even being attempted, until the cooldown expires
- **HALF_OPEN** — after cooldown, allow exactly one trial call through: success closes the circuit again, failure reopens it

```python
import asyncio
import time
import enum

class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitOpenError(Exception):
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown=10):
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.cooldown:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit is open — failing fast")

        try:
            result = await func(*args, **kwargs)
        except Exception:
            self.failure_count += 1
            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()
            raise
        else:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            return result

breaker = CircuitBreaker(failure_threshold=3, cooldown=5)

async def call_downstream():
    return await breaker.call(unreliable_api_call)
```

---

## 25.5 Graceful Shutdown: The Pattern (Mechanics in Chapter 26)

The general shape of a graceful shutdown, at the pattern level:

1. **Stop accepting new work** the moment shutdown is requested.
2. **Let in-flight work finish**, but only up to a **bounded** grace period — never wait indefinitely.
3. **Force-cancel** anything still running past that bound.
4. **Close shared resources last** — connection pools, HTTP sessions — only after all workers have actually stopped touching them.

```python
import asyncio

shutdown_event = asyncio.Event()

async def worker(name):
    while not shutdown_event.is_set():
        await asyncio.sleep(1)   # simulate doing a unit of work
        print(f"{name}: still working")
    print(f"{name}: shutdown noticed, winding down")

async def main():
    async with asyncio.TaskGroup() as tg:
        workers = [tg.create_task(worker(f"worker-{i}")) for i in range(3)]

        await asyncio.sleep(3)   # simulate the app running normally for a while
        print("Shutdown requested")
        shutdown_event.set()

        try:
            async with asyncio.timeout(5):   # bounded grace period
                await asyncio.gather(*workers)
        except TimeoutError:
            print("Grace period exceeded — force-cancelling stragglers")
            for w in workers:
                w.cancel()

asyncio.run(main())
```

This chapter deliberately stops at *triggering* `shutdown_event.set()` manually — Chapter 26 shows how to trigger it from a **real OS signal** (Ctrl+C, `SIGTERM`), which is what makes this pattern actually usable in a deployed service rather than just a demo.

---

## 25.6 Generalizing Connection Pooling

Chapter 14's `ConnectionPool` and Chapter 20's database pools share the same shape — a bounded resource, checked out and returned via an async context manager. If you're wrapping something that doesn't already provide pooling (most databases and HTTP libraries already do — prefer their built-in pooling over rolling your own), a generic version looks like this:

```python
from contextlib import asynccontextmanager
import asyncio

class ResourcePool:
    def __init__(self, factory, size):
        self._queue = asyncio.Queue(maxsize=size)
        for _ in range(size):
            self._queue.put_nowait(factory())

    @asynccontextmanager
    async def acquire(self):
        resource = await self._queue.get()
        try:
            yield resource
        finally:
            await self._queue.put(resource)
```

Reach for this only when no built-in pooling exists for the resource in question — for databases and HTTP clients specifically, the library's own pool (Chapters 19–20) is almost always the better-tested choice.

---

## 25.7 Combining Patterns: A Resilient Client

Real production clients typically layer several of these together:

```python
sem = asyncio.Semaphore(10)          # concurrency cap
bucket = TokenBucket(rate=5, capacity=10)   # requests/second budget
breaker = CircuitBreaker(failure_threshold=5, cooldown=15)

@retry(max_attempts=3, base_delay=0.5, exceptions=(ConnectionError,))
async def resilient_call(payload):
    async with sem:
        await bucket.acquire()
        return await breaker.call(flaky_downstream_call, payload)
```

Each layer solves a distinct problem: the semaphore bounds *how many* calls are in flight, the token bucket bounds *how fast* new calls start, the circuit breaker stops attempting calls entirely during a sustained outage, and the retry decorator absorbs isolated, transient failures — none of them substitute for each other.

---

## 25.8 Hands-On Exercises

**Exercise 1 — Token bucket bursting.**
Implement `TokenBucket(rate=2, capacity=5)`. Fire 10 calls to `acquire()` back to back and print timestamps. Confirm the first 5 go through immediately (the burst), and the remaining 5 are paced at roughly 0.5s intervals thereafter.

**Exercise 2 — Retry with jitter.**
Implement the `@retry` decorator and `flaky_call()` from §25.3. Run it 30 times, tracking overall success rate. Print the actual delay used on each retry across several runs and confirm they vary (proving jitter is working), rather than being identical fixed values each time.

**Exercise 3 — Circuit breaker lifecycle.**
Implement `CircuitBreaker`. Wrap a function that fails for its first 5 calls, then succeeds afterward. Call it repeatedly and confirm: the circuit opens after `failure_threshold` failures; calls during the cooldown raise `CircuitOpenError` immediately (fast-fail, not even attempting the underlying call); and after the cooldown, a successful half-open trial call closes the circuit again.

**Exercise 4 — Bounded graceful shutdown.**
Implement the §25.5 example with 5 workers instead of 3. Make one worker deliberately ignore the shutdown event for longer than the grace period (e.g., a hardcoded long sleep). Confirm the other workers shut down cleanly within the grace period, while the straggler gets force-cancelled once the bound is exceeded.

**Exercise 5 (stretch) — The fully combined resilient client.**
Implement §25.7's layered `resilient_call()`. Simulate 50 calls to a downstream function that's rate-sensitive (fails a lot if called too fast), occasionally flaky (transient failures), and occasionally down for a stretch (sustained failures). Report a final summary: successes, retried-then-succeeded, circuit-breaker-rejected, and permanent failures.

---

## 25.9 Common Pitfalls

- **Using fixed (non-jittered) exponential backoff across many clients.** Without jitter, clients that failed simultaneously tend to retry in synchronized waves, worsening exactly the kind of overload that caused the original failures.
- **Misconfiguring circuit breaker thresholds.** Too sensitive, and normal transient blips trip it unnecessarily; too lax, and it never actually protects anything. Tune based on your service's real observed failure patterns, not guesswork.
- **Building a custom connection pool when the library already provides one.** Reinventing pooling for a database or HTTP client that already has a well-tested built-in pool adds risk without benefit.
- **Treating graceful shutdown as an afterthought.** Retrofitting cooperative shutdown checks into worker loops that weren't designed for it is far harder than designing them in from the start.

---

## 25.10 Further Reading

- [AWS Architecture Blog: "Exponential Backoff and Jitter"](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)

---

**Next: Chapter 26 — Context Propagation and Signals** (`contextvars` across concurrent tasks, and hooking real OS signals — `SIGINT`/`SIGTERM` — into the graceful shutdown pattern from this chapter).
