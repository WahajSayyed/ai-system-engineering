# Chapter 28: Capstone Project — Concurrent Job Processing System

> Part 7 — Capstone · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 28.1 Project Overview

This capstone builds **AsyncJobRunner**: a small but realistic concurrent job-processing service, deliberately designed to touch nearly every concept from this course. Its architecture:

1. Jobs arrive on an internal **bounded `Queue`** (Chapter 13).
2. A **pool of worker coroutines** (Chapter 13) pulls jobs off the queue and processes each by calling a simulated external API — through a **rate limiter**, a **circuit breaker**, and **retries with jitter** (Chapter 25).
3. Every job gets a unique ID propagated via **`contextvars`** (Chapter 26), so every log line anywhere in that job's processing is automatically tagged.
4. The whole service supports **graceful shutdown** on `SIGINT`/`SIGTERM` (Chapter 26): stop pulling new jobs, let in-flight jobs finish within a bounded grace period, then force-cancel stragglers.
5. A small **HTTP monitoring endpoint** (Chapter 16/19) exposes live `/health` and `/stats`, running concurrently alongside the worker pool.

```
┌─────────────┐     ┌───────────────┐      ┌──────────────────────┐
│  Job Source  │────▶│  asyncio.Queue │────▶│   Worker Pool (N)    │
│ (seeded here)│     │   (bounded)    │      │  rate-limited, retried,│
└─────────────┘     └───────────────┘      │  circuit-breaker wrapped │
                                             └──────────┬───────────┘
                                                         │
                                             ┌───────────▼───────────┐
                                             │   Shared Stats object  │
                                             └───────────┬───────────┘
                                                         │
                                             ┌───────────▼───────────┐
                                             │ HTTP /health, /stats   │
                                             └────────────────────────┘
```

---

## 28.2 Building Blocks (Reused From Chapter 25)

```python
import asyncio
import contextvars
import functools
import logging
import random
import signal
import time
import enum
from dataclasses import dataclass, field

# --- Structured logging with per-job IDs (Chapter 26) ---

job_id_var = contextvars.ContextVar("job_id", default="-")

class JobIdFilter(logging.Filter):
    def filter(self, record):
        record.job_id = job_id_var.get()
        return True

handler = logging.StreamHandler()
handler.addFilter(JobIdFilter())
handler.setFormatter(logging.Formatter("%(asctime)s [job=%(job_id)s] %(message)s", "%H:%M:%S"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("asyncjobrunner")


# --- Token bucket rate limiter (Chapter 25) ---

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
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


# --- Circuit breaker (Chapter 25) ---

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
                raise CircuitOpenError("circuit open — failing fast")
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


# --- Retry decorator with full jitter (Chapter 25) ---

def retry(max_attempts=3, base_delay=0.3, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions:
                    if attempt == max_attempts:
                        raise
                    delay = random.uniform(0, base_delay * (2 ** (attempt - 1)))
                    logger.info(f"retry {attempt}/{max_attempts} in {delay:.2f}s")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator
```

---

## 28.3 The Job Model and Shared Stats

```python
@dataclass
class Job:
    job_id: str
    payload: dict

@dataclass
class Stats:
    processed: int = 0
    failed: int = 0
    rejected_by_breaker: int = 0
    in_flight: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record_start(self):
        async with self.lock:
            self.in_flight += 1

    async def record_success(self):
        async with self.lock:
            self.in_flight -= 1
            self.processed += 1

    async def record_failure(self, rejected=False):
        async with self.lock:
            self.in_flight -= 1
            self.failed += 1
            if rejected:
                self.rejected_by_breaker += 1

    def snapshot(self):
        return {
            "processed": self.processed,
            "failed": self.failed,
            "rejected_by_breaker": self.rejected_by_breaker,
            "in_flight": self.in_flight,
        }
```

---

## 28.4 The Simulated External API and the Resilient Wrapper

```python
async def flaky_external_api(payload):
    """Simulates a downstream API: usually fast, occasionally slow, sometimes fails."""
    await asyncio.sleep(random.uniform(0.05, 0.2))
    if random.random() < 0.25:
        raise ConnectionError(f"transient failure processing {payload}")
    return {"result": f"processed-{payload['item']}"}


class ResilientClient:
    def __init__(self):
        self.bucket = TokenBucket(rate=10, capacity=15)
        self.breaker = CircuitBreaker(failure_threshold=5, cooldown=8)

    @retry(max_attempts=3, base_delay=0.3, exceptions=(ConnectionError,))
    async def call(self, payload):
        await self.bucket.acquire()
        return await self.breaker.call(flaky_external_api, payload)
```

---

## 28.5 The Worker Pool

```python
async def worker(name, queue, client, stats, shutdown_event):
    while True:
        get_task = asyncio.create_task(queue.get())
        wait_task = asyncio.create_task(shutdown_event.wait())
        done, pending = await asyncio.wait(
            {get_task, wait_task}, return_when=asyncio.FIRST_COMPLETED
        )

        if wait_task in done and get_task not in done:
            get_task.cancel()
            logger.info(f"{name}: shutdown noticed, no job in flight — exiting")
            break

        job = get_task.result()
        wait_task.cancel()

        job_id_var.set(job.job_id)
        await stats.record_start()
        try:
            result = await client.call(job.payload)
            logger.info(f"{name}: succeeded -> {result}")
            await stats.record_success()
        except CircuitOpenError:
            logger.warning(f"{name}: rejected by circuit breaker")
            await stats.record_failure(rejected=True)
        except Exception as e:
            logger.warning(f"{name}: gave up after retries: {e}")
            await stats.record_failure()
        finally:
            queue.task_done()
```

The `asyncio.wait(..., return_when=FIRST_COMPLETED)` pattern here (Chapter 7) lets each worker respond to shutdown **even while blocked waiting for a job that hasn't arrived yet** — a plain `await queue.get()` alone would leave a worker stuck indefinitely if the queue is empty when shutdown is requested.

---

## 28.6 The Monitoring Endpoint

```python
from aiohttp import web

def build_monitor_app(stats):
    app = web.Application()

    async def health(request):
        return web.json_response({"status": "ok"})

    async def stats_view(request):
        return web.json_response(stats.snapshot())

    app.router.add_get("/health", health)
    app.router.add_get("/stats", stats_view)
    return app
```

---

## 28.7 Wiring It All Together

```python
async def job_producer(queue, num_jobs, shutdown_event):
    """Simulates jobs arriving over time. Stops enqueueing once shutdown is requested."""
    for i in range(num_jobs):
        if shutdown_event.is_set():
            logger.info("producer: shutdown requested, no longer enqueueing new jobs")
            break
        job = Job(job_id=f"job-{i}", payload={"item": i})
        await queue.put(job)
        await asyncio.sleep(0.05)

async def main():
    queue = asyncio.Queue(maxsize=20)
    stats = Stats()
    client = ResilientClient()
    shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    runner = web.AppRunner(build_monitor_app(stats))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8080)
    await site.start()
    logger.info("Monitoring endpoint live on http://127.0.0.1:8080/stats")

    try:
        async with asyncio.TaskGroup() as tg:
            workers = [
                tg.create_task(worker(f"worker-{i}", queue, client, stats, shutdown_event))
                for i in range(4)
            ]
            tg.create_task(job_producer(queue, num_jobs=200, shutdown_event=shutdown_event))

            await shutdown_event.wait()
            logger.info("Shutdown requested — waiting up to 10s for in-flight jobs")

            try:
                async with asyncio.timeout(10):
                    await queue.join()
            except TimeoutError:
                logger.warning("Grace period exceeded — remaining jobs will be abandoned")

            shutdown_event.set()   # ensure all workers definitely notice, even if already set
    finally:
        await runner.cleanup()
        logger.info(f"Final stats: {stats.snapshot()}")

if __name__ == "__main__":
    asyncio.run(main())
```

While this runs, `curl http://127.0.0.1:8080/stats` from another terminal shows live job counts — and pressing Ctrl+C (or sending `SIGTERM`) triggers the same graceful shutdown sequence built up across Chapters 25–26.

---

## 28.8 Testing the Capstone

A couple of representative tests, applying Chapter 23's patterns directly:

```python
import pytest
from unittest.mock import AsyncMock, patch

async def test_resilient_client_retries_then_succeeds():
    client = ResilientClient()
    with patch("__main__.flaky_external_api", new_callable=AsyncMock) as mock_api:
        mock_api.side_effect = [ConnectionError("fail once"), {"result": "ok"}]
        result = await client.call({"item": 1})
        assert result == {"result": "ok"}
        assert mock_api.await_count == 2

async def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=2, cooldown=100)
    always_fails = AsyncMock(side_effect=ConnectionError("down"))

    for _ in range(2):
        with pytest.raises(ConnectionError):
            await breaker.call(always_fails)

    with pytest.raises(CircuitOpenError):
        await breaker.call(always_fails)   # now fails fast, doesn't even call always_fails
```

---

## 28.9 Extension Ideas

If you'd like to keep building on this capstone as a personal reference project (a natural fit for a GitHub repo alongside your other completed curricula):

- Replace `flaky_external_api` with a real `httpx`/`aiohttp` call to an actual external service (Chapter 19).
- Persist jobs to a real Postgres-backed queue table instead of an in-memory `asyncio.Queue`, using `asyncpg` (Chapter 20).
- Swap the default event loop for `uvloop` via `loop_factory` (Chapter 21) and benchmark the difference (Chapter 24) under heavier concurrent load.
- Add a `/metrics` endpoint in Prometheus text-exposition format alongside `/stats`, for real observability tooling integration.
- Extend `contextvars`-based tracing (Chapter 26) into a proper distributed tracing format (e.g., OpenTelemetry) if the system ever spans multiple processes or services.

---

## 28.10 Course Wrap-Up

This capstone deliberately pulls a thread from nearly every chapter: the event loop mechanics from Part 1, the concurrency primitives and structured `TaskGroup` from Part 2, the `Queue`-based worker pool from Part 3, the networking layer from Part 4, the debugging, testing, and performance discipline from Part 5, and the production-grade resilience and shutdown patterns from Part 6.

The throughline across all 28 chapters has been the same handful of ideas, recombined at increasing scale: a single event loop, cooperatively multitasking; `await` points as the only place control changes hands; cancellation as an ordinary (if special) exception; and structured concurrency as the discipline that keeps all of it manageable as real systems grow. That's the whole of asyncio, really — everything else in this course has just been that same small set of ideas, applied to progressively more realistic problems.
