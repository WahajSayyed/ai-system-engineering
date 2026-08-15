# Chapter 23: Testing Async Code

> Part 5 — Internals and Production Concerns · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 23.1 Why Async Tests Need Special Handling

Here's a sneaky problem, directly echoing Chapter 4's "forgot to await" lesson — but this time hiding inside your test suite itself. Plain `pytest` has no idea what to do with an `async def test_something():` function. If you write one without the right plugin installed, calling it doesn't run its body — it just produces a coroutine object, exactly like calling any other coroutine function without awaiting it. Depending on your pytest version, this can silently "pass" without ever actually executing your assertions, which is far worse than an honest failure.

**`pytest-asyncio`** is the standard plugin that fixes this: it provides the machinery to actually drive async test functions (and async fixtures) inside a real event loop, the same way `asyncio.run()` would.

---

## 23.2 Installing and Configuring `pytest-asyncio`

```
pip install pytest-asyncio
```

The single most important configuration choice is the **mode**: `strict` (the default) or `auto`.

- **`strict`**: only tests explicitly decorated with `@pytest.mark.asyncio`, and fixtures decorated with `@pytest_asyncio.fixture`, are treated as async. Everything else is left alone. This matters if your project needs to coexist with other async frameworks (e.g., `trio`, via the `anyio` pytest plugin) in the same test suite, since it avoids pytest-asyncio grabbing tests that belong to a different backend.
- **`auto`**: pytest-asyncio automatically detects and runs **every** `async def test_*` function and `async def` fixture, with no decorators needed at all.

For a project that's entirely `asyncio`-based (which describes most application code), `auto` is the current recommended default — it removes a decorator you'd otherwise have to add to every single test.

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

---

## 23.3 A Basic Async Test

```python
async def add_async(a, b):
    await asyncio.sleep(0)   # simulate some async work
    return a + b

async def test_add_async():
    result = await add_async(2, 3)
    assert result == 5
```

With `asyncio_mode = "auto"`, that's the entire test — no decorator required. In `strict` mode, you'd add `@pytest.mark.asyncio` directly above the `async def test_add_async():` line.

---

## 23.4 Async Fixtures

Fixtures can be coroutines too, letting you `await` real async setup and teardown:

```python
import pytest_asyncio   # only needed if using strict mode's explicit decorator

@pytest_asyncio.fixture   # in auto mode, a plain @pytest.fixture works just as well
async def db_pool():
    pool = await create_pool("postgresql://localhost/test")
    yield pool
    await pool.close()   # async teardown, runs after the test using this fixture finishes

async def test_query(db_pool):
    result = await db_pool.fetch("SELECT 1")
    assert result
```

The `await pool.close()` after `yield` runs inside the same event loop as the test itself, so genuine async cleanup — closing connections, flushing buffers — works exactly as you'd want, directly analogous to Chapter 14's `@asynccontextmanager` pattern.

---

## 23.5 Event Loop Scope: Isolation vs. Sharing

By default, **each test gets its own, fresh event loop** (function scope) — maximizing isolation between tests, so one test's leftover state can never leak into another's. This has a real, easy-to-miss consequence: **async tests run sequentially, not concurrently**, by design:

```python
async def test_first():
    await asyncio.sleep(2)

async def test_second():
    await asyncio.sleep(2)

# Total suite time: ~4 seconds, not ~2 — pytest runs each test in its own
# loop, one at a time, exactly like it runs synchronous tests one at a time.
```

If you need to **share** an expensive resource (say, one database connection pool) across multiple tests rather than recreating it for each, you need a fixture with a **broader** loop scope, matched explicitly:

```python
import pytest
import pytest_asyncio

@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def shared_pool():
    pool = await create_pool(...)
    yield pool
    await pool.close()

@pytest.mark.asyncio(loop_scope="module")
async def test_one(shared_pool):
    ...

@pytest.mark.asyncio(loop_scope="module")
async def test_two(shared_pool):
    ...
```

**A common error this mismatch produces:** `RuntimeError: Event loop is closed`. This happens when a broader-scoped fixture (e.g., `session`-scoped) tries to use a narrower-scoped event loop (the default `function` scope) that's already been closed by the time a later test tries to reuse the fixture. The fix is always the same: make the fixture's scope and its `loop_scope` match the breadth you actually need, and make sure any test using it declares the same `loop_scope`.

---

## 23.6 Mocking Coroutines: `unittest.mock.AsyncMock`

A plain `MagicMock` doesn't work for mocking an async dependency — calling it returns another `MagicMock`, not something you can `await`, producing `TypeError: object MagicMock can't be used in 'await' expression`. **`unittest.mock.AsyncMock`** (standard library since Python 3.8) fixes this: calling an `AsyncMock` returns something awaitable, resolving to its configured `return_value`.

Conveniently, `unittest.mock.patch()` **auto-detects** when the thing being patched is an async function, and uses `AsyncMock` automatically — you usually don't need to construct one manually.

```python
from unittest.mock import patch

async def process_order(order_id, api_client):
    result = await api_client.fetch_order(order_id)
    return result["total"] * 1.1   # add tax

async def test_process_order_applies_tax():
    with patch("myapp.api_client.fetch_order") as mock_fetch:
        mock_fetch.return_value = {"total": 100}
        result = await process_order(42, api_client)
        assert result == 110
        mock_fetch.assert_awaited_once_with(42)

async def test_process_order_propagates_api_errors():
    with patch("myapp.api_client.fetch_order") as mock_fetch:
        mock_fetch.side_effect = ConnectionError("API down")
        with pytest.raises(ConnectionError):
            await process_order(42, api_client)
```

Note `assert_awaited_once_with(...)` — `AsyncMock` provides await-aware assertion methods (`assert_awaited`, `assert_awaited_once`, `assert_awaited_with`, etc.) distinct from the regular `assert_called*` family, specifically so you can confirm the mock was actually **awaited**, not just called.

---

## 23.7 Testing Timeouts and Cancellation

**Testing that a timeout actually fires:**

```python
import pytest
import asyncio

async def slow_operation():
    await asyncio.sleep(5)

async def test_operation_times_out():
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            await slow_operation()
```

**Testing that cancellation triggers proper cleanup** (tying directly back to Chapter 9):

```python
import asyncio

class FakeResource:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True

async def worker(resource):
    try:
        await asyncio.sleep(10)
    finally:
        await resource.close()

async def test_worker_cleans_up_on_cancellation():
    resource = FakeResource()
    task = asyncio.create_task(worker(resource))
    await asyncio.sleep(0.1)   # let it start

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert resource.closed is True   # confirm the finally block actually ran
```

This is a genuinely valuable test to have for any code with cleanup logic — it directly verifies the contract from Chapter 9 (cleanup runs on cancellation) rather than just hoping it does.

---

## 23.8 A Design Tip: Keep Test Suites Fast

Real `asyncio.sleep()` calls in tests add up — a suite with dozens of tests each sleeping even half a second adds real minutes to your CI pipeline. Rather than reaching for clock-mocking tricks (which are considerably trickier with asyncio's internal `loop.time()`-based scheduling than with simple synchronous code), the more robust habit is: **design your code so delays and timeouts are injectable parameters**, not hardcoded constants. Production code calls `my_operation(timeout=30)`; tests call `my_operation(timeout=0.01)` — same code path, dramatically faster test runs, no clock-mocking fragility involved at all.

---

## 23.9 Hands-On Exercises

**Exercise 1 — Basic setup.**
Install `pytest` and `pytest-asyncio`. Configure `asyncio_mode = "auto"` in `pyproject.toml`. Write and run a basic async test for a simple `async def add_async(a, b)` function, as in §23.3.

**Exercise 2 — Async fixture with verified teardown.**
Write an async fixture providing a fake "connection" object with an `async def close(self)` method that sets a flag. Write a test using the fixture, and — outside the test function itself — confirm the flag was set after the test completed (e.g., by holding a reference to the fixture's object at module scope, or checking via a second fixture).

**Exercise 3 — Reproduce and fix "Event loop is closed."**
Create a `session`-scoped async fixture **without** matching `loop_scope`, use it from two different tests, and reproduce the `RuntimeError: Event loop is closed`. Fix it by adding the matching `loop_scope="session"` to both the fixture and the tests using it.

**Exercise 4 — Mock an async dependency.**
Using `unittest.mock.patch` and `AsyncMock`, write two tests for `process_order()` from §23.6: one confirming correct behavior on a successful mocked response, and one confirming a mocked failure (`side_effect`) propagates correctly. Use `assert_awaited_once_with()` in the first test.

**Exercise 5 — Test a timeout.**
Implement and run the §23.7 timeout test. Then adjust the timeout to be longer than the operation and confirm the test now passes without raising.

**Exercise 6 (stretch) — Test cleanup-on-cancellation for a real worker.**
Take a worker function from Chapter 13's worker-pool pattern (or write a simplified version), and write a test confirming that cancelling it mid-execution still calls `queue.task_done()` in its `finally` block — using a real (or fake) `asyncio.Queue` and checking `queue.join()` behavior, or a mock tracking whether `task_done()` was called.

---

## 23.10 Common Pitfalls

- **Writing `async def test_*` functions without `pytest-asyncio` installed/configured.** The test can silently "pass" without its body ever actually running — always confirm the plugin is installed and `asyncio_mode` is set appropriately.
- **Using `MagicMock` instead of `AsyncMock` for an async dependency.** This raises `TypeError` the moment you try to `await` the mock — though `patch()`'s auto-detection since Python 3.8 handles the common case for you automatically.
- **Mismatched fixture and test `loop_scope`.** This is the direct cause of the common `"Event loop is closed"` error — always align them deliberately when sharing resources across tests.
- **Assuming async tests run concurrently with each other.** They don't, by design — pytest-asyncio runs them sequentially, exactly like synchronous tests, for isolation guarantees.
- **Writing tests with long, real `asyncio.sleep()` calls.** This bloats your test suite's runtime for no real benefit — design delays/timeouts as injectable parameters instead.

---

## 23.11 Further Reading

- [pytest-asyncio documentation: Concepts](https://pytest-asyncio.readthedocs.io/en/latest/concepts.html)
- [pytest-asyncio documentation: Configuration](https://pytest-asyncio.readthedocs.io/en/latest/reference/configuration.html)
- [Python docs: `unittest.mock.AsyncMock`](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.AsyncMock)

---

**Next: Chapter 24 — Performance and Benchmarking** (measuring throughput/latency, `uvloop` benchmarks in practice, and knowing when threads/processes genuinely beat asyncio).
