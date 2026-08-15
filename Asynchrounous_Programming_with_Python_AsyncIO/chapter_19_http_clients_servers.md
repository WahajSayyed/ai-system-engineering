# Chapter 19: HTTP Clients and Servers in the Async World

> Part 4 — I/O, Networking, and Real Systems · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 19.1 Why Not Just `requests`?

Chapter 18 showed you can wrap the synchronous `requests` library with `to_thread()` to avoid freezing the event loop. That works, but it doesn't scale well: you're bounded by the thread pool's worker count, and each in-flight request ties up a whole OS thread for its entire duration. A **native async HTTP client** uses non-blocking sockets directly (the same streams/transport layer from Chapter 16), letting you have hundreds or thousands of requests genuinely in flight at once, limited mainly by memory and OS file descriptors — not thread count.

The two dominant async-native choices in the Python ecosystem are **`aiohttp`** and **`httpx`**. Both are mature, actively maintained, and commonly used in production — including by the OpenAI and Anthropic Python SDKs, which use `httpx` as their default HTTP backend specifically for its unified sync/async API. The rough guidance: `httpx` gives you one API that works both sync and async (handy if you're building a library or SDK that needs both), plus native HTTP/2 support; `aiohttp` is async-only but tends to edge out `httpx` at very high concurrency (some benchmarks report a meaningful edge starting somewhere around 200+ simultaneous requests), and also includes a full async web server, not just a client. For most application code, either is a perfectly good choice — start with whichever fits your broader stack, and only benchmark the difference if you're operating at real scale.

---

## 19.2 `aiohttp`: Async-Native Client

```python
import aiohttp
import asyncio

async def fetch(session, url):
    async with session.get(url) as response:
        return response.status, await response.text()

async def main():
    urls = ["https://example.com", "https://example.org", "https://example.net"]
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(fetch(session, url) for url in urls))
        for status, _ in results:
            print(status)

asyncio.run(main())
```

**Always create one `ClientSession` and reuse it** across requests — it manages connection pooling internally, and creating a new session per-request throws that benefit away (and, on Python 3.12+, creating one outside a running event loop now raises `RuntimeError` rather than just warning). The `async with session.get(url) as response:` pattern is itself an async context manager (Chapter 14), ensuring the response body is properly released back to the connection pool once you're done with it.

---

## 19.3 `httpx`: A Requests-Like API, Sync or Async

```python
import httpx
import asyncio

async def fetch(client, url):
    response = await client.get(url)
    return response.status_code, response.text

async def main():
    urls = ["https://example.com", "https://example.org", "https://example.net"]
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(fetch(client, url) for url in urls))
        for status, _ in results:
            print(status)

asyncio.run(main())
```

Notice the API is nearly identical in shape to `aiohttp` — `AsyncClient()` instead of `ClientSession()`, and `response.status_code`/`.text` instead of `response.status`/`await response.text()` (httpx reads the whole body eagerly by default, so no extra `await` is needed there). If you're already familiar with the synchronous `requests` library, `httpx`'s API will feel immediately familiar.

**One documented anti-pattern worth flagging directly:** never share a single `httpx.AsyncClient` across multiple threads — this has caused real concurrency bugs in production SDKs. Within a single event loop, sharing one client across many concurrent coroutines/tasks is exactly the intended, correct usage.

---

## 19.4 Rate Limiting: Concurrency Cap vs. Requests-Per-Second

These are **two different constraints**, easy to conflate:

- **Concurrency cap** — "never more than N requests in flight at once." A `Semaphore` (Chapter 12) handles this directly.
- **Rate limit** — "never more than N requests *per second*, regardless of how many are in flight." A semaphore alone doesn't enforce this — if your requests are fast, a `Semaphore(10)` could still let you burst 10 requests in the same millisecond, over and over, exceeding a strict per-second API limit.

**Concurrency-capped scraper (semaphore-based):**

```python
import asyncio
import aiohttp

sem = asyncio.Semaphore(5)   # never more than 5 requests in flight

async def fetch(session, url):
    async with sem:
        async with session.get(url) as response:
            return url, response.status

async def main():
    urls = [f"https://example.com/page/{i}" for i in range(20)]
    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(fetch(session, url)) for url in urls]

    for task in tasks:
        print(task.result())

asyncio.run(main())
```

**A simple token-bucket rate limiter, for a strict requests-per-second budget:**

```python
import asyncio
import time

class RateLimiter:
    def __init__(self, rate_per_second):
        self._interval = 1 / rate_per_second
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._last_call + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

async def fetch(session, url, limiter):
    await limiter.acquire()
    async with session.get(url) as response:
        return url, response.status

async def main():
    limiter = RateLimiter(rate_per_second=2)   # strictly 2 requests/second, paced
    urls = [f"https://example.com/page/{i}" for i in range(6)]
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(fetch(session, url, limiter) for url in urls))
        print(results)

asyncio.run(main())
```

Both patterns are frequently used **together** in real scrapers/API clients: a `Semaphore` to bound how many requests are simultaneously in flight (protecting your own resources), plus a rate limiter to respect a downstream API's strict per-second quota (protecting the API from being hammered) — neither one substitutes for the other.

---

## 19.5 Retries with Backoff

Transient failures — timeouts, connection resets, 5xx server errors — are normal at scale and usually worth retrying, but **never indefinitely and never without backoff**, or you risk hammering an already-struggling service harder (a "thundering herd" against your own downstream dependency).

```python
import asyncio
import random

class FlakyError(Exception):
    pass

async def flaky_fetch(url):
    if random.random() < 0.3:   # simulate a ~30% transient failure rate
        raise FlakyError(f"transient failure for {url}")
    await asyncio.sleep(0.1)
    return f"success: {url}"

async def fetch_with_retry(url, max_attempts=3, base_delay=0.5):
    for attempt in range(1, max_attempts + 1):
        try:
            return await flaky_fetch(url)
        except FlakyError as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"Attempt {attempt} failed ({e}); retrying in {delay:.1f}s")
            await asyncio.sleep(delay)

async def main():
    try:
        result = await fetch_with_retry("https://example.com")
        print(result)
    except FlakyError as e:
        print(f"Gave up: {e}")

asyncio.run(main())
```

This is exponential backoff (`0.5s`, `1s`, `2s`, ...) — in production code, it's also common to add **jitter** (a small random offset added to each delay) to avoid many clients retrying in lockstep after a shared outage. Combined with Chapter 11's `resilient_gather()` pattern, this gives you a scraper that retries individual failures gracefully while still returning whatever succeeded overall.

---

## 19.6 Timeouts for HTTP Clients

Both libraries have their own configurable timeout settings, layered on top of the general `asyncio.timeout()` concept from Chapter 10:

```python
# aiohttp
timeout = aiohttp.ClientTimeout(total=10, connect=3)
async with aiohttp.ClientSession(timeout=timeout) as session:
    ...

# httpx
timeout = httpx.Timeout(10.0, connect=3.0)
async with httpx.AsyncClient(timeout=timeout) as client:
    ...
```

These library-native settings are usually the right primary tool, since they distinguish finer-grained phases (connection establishment vs. waiting for a response) that a single blanket `asyncio.timeout()` can't. Still, wrapping a whole multi-step operation (e.g., "fetch, then parse, then store — all within 15 seconds total") in `asyncio.timeout()` remains useful as an overall backstop budget, exactly as in Chapter 10.

---

## 19.7 Hands-On Exercises

**Exercise 1 — Basic concurrent fetch with `aiohttp`.**
Install `aiohttp` (`pip install aiohttp`), and fetch 5 URLs concurrently using one shared `ClientSession` inside a `TaskGroup`. Print each URL's status code.

**Exercise 2 — The `httpx` equivalent.**
Repeat Exercise 1 using `httpx.AsyncClient` instead. Note the small API differences (`response.status_code` vs. `response.status`, no `await` needed for `.text`).

**Exercise 3 — Bounded concurrency.**
Fetch 20 URLs with a `Semaphore(5)`. Add a shared counter (protected by a `Lock`, Chapter 12) tracking the maximum number of simultaneously in-flight requests, and confirm it never exceeds 5.

**Exercise 4 — Strict rate limiting.**
Implement `RateLimiter` from §19.4 and use it to fetch 10 URLs at a strict 2 requests/second. Print timestamps for each request and confirm they're spaced roughly 0.5 seconds apart.

**Exercise 5 — Retry with backoff.**
Implement `fetch_with_retry()` from §19.5. Run it 20 times in a loop and count how often it ultimately succeeds vs. gives up after 3 attempts, given the ~30% simulated per-attempt failure rate.

**Exercise 6 (stretch) — Put it all together.**
Build a "resilient concurrent scraper" for 30 URLs combining: a `Semaphore` for concurrency capping, the `RateLimiter` for pacing, `fetch_with_retry` for transient failures, and Chapter 11's `resilient_gather()` pattern to separate final successes from permanent failures. Print a summary: total succeeded, total failed, and total time taken.

---

## 19.8 Common Pitfalls

- **Creating a new `ClientSession`/`AsyncClient` per request instead of reusing one.** This throws away connection pooling and is significantly slower under load — create one session/client and reuse it for the lifetime of your program (or a well-defined scope).
- **Wrapping a blocking library like `requests` instead of using a native async client at scale.** It works for small numbers of concurrent requests (Chapter 18) but doesn't scale to hundreds/thousands the way `aiohttp`/`httpx` do.
- **Using only a `Semaphore` when you actually need a requests-per-second limit.** A semaphore bounds concurrency, not throughput over time — fast requests can still burst well past a strict per-second quota without a dedicated rate limiter.
- **Not setting any timeout at all.** A hung request can wait forever without an explicit client-level timeout or an `asyncio.timeout()` backstop.
- **Retrying indefinitely, or without backoff.** Always cap retry attempts and use (ideally jittered) exponential backoff — unbounded, immediate retries can make a struggling downstream service's problem worse.

---

## 19.9 Further Reading

- [aiohttp documentation](https://docs.aiohttp.org/)
- [httpx documentation](https://www.python-httpx.org/)

---

**Next: Chapter 20 — Async Database Access** (`asyncpg`/`aiomysql`, SQLAlchemy 2.0's async ORM, and connection pooling considerations for real data-backed applications).
