# Mastering Asynchronous Programming in Python — AsyncIO

A beginner-to-advanced course on Python's `asyncio`, grounded in the current state of the language (Python 3.13/3.14) and inspired by the official docs, *Using Asyncio in Python* (Caleb Hattingh), *Python Concurrency with asyncio* (Matthew Fowler), David Beazley's coroutine talks, Lynn Root's "asyncio: We Did It Wrong," and production patterns from the FastAPI/aiohttp ecosystem.

Every chapter includes theory, annotated runnable code, hands-on exercises (with stretch goals), a common-pitfalls section, and further reading.

---

## Part 1 — Foundations

1. [Why Asynchronous Programming?](chapter_01_why_asynchronous_programming.md) — concurrency vs. parallelism, the GIL, and where free-threaded Python (3.13/3.14) does and doesn't change the picture
2. [A Brief History of Async in Python](chapter_02_brief_history_of_async.md) — callbacks → generator coroutines → native `async`/`await`, through Python 3.14
3. [The Event Loop Explained](chapter_03_event_loop_explained.md) — build a real ~40-line event loop from scratch
4. [Coroutines 101](chapter_04_coroutines_101.md) — `async def`, `await`, `asyncio.run()`, and the two most common beginner bugs

## Part 2 — Core AsyncIO Building Blocks

5. [Tasks: Scheduling Concurrent Work](chapter_05_tasks_scheduling_concurrent_work.md) — real concurrency, the fire-and-forget gotcha, eager task execution (3.12+)
6. [Futures and Awaitables](chapter_06_futures_and_awaitables.md) — what `Task` inherits from, and bridging callback APIs into `async`/`await`
7. [Running Things Concurrently: gather, wait, as_completed](chapter_07_gather_wait_as_completed.md)
8. [Structured Concurrency with TaskGroups](chapter_08_structured_concurrency_taskgroups.md) — `except*`, `ExceptionGroup`, automatic sibling cancellation
9. [Cancellation Deep Dive](chapter_09_cancellation_deep_dive.md) — `CancelledError`, `shield()`, the 3.11 `cancelling()`/`uncancel()` API
10. [Timeouts](chapter_10_timeouts.md) — `wait_for`, `asyncio.timeout()`, composing timeouts with `TaskGroup`
11. [Exception Handling in Concurrent Code](chapter_11_exception_handling_concurrent_code.md) — silent fire-and-forget failures, custom exception handlers

## Part 3 — Synchronization and Communication

12. [Synchronization Primitives](chapter_12_synchronization_primitives.md) — `Lock`, `Event`, `Condition`, `Semaphore`/`BoundedSemaphore`
13. [Queues and Producer-Consumer Patterns](chapter_13_queues_producer_consumer.md) — backpressure, worker pools
14. [Async Context Managers](chapter_14_async_context_managers.md) — `@asynccontextmanager`, `AsyncExitStack`
15. [Async Iterators and Generators](chapter_15_async_iterators_generators.md) — streaming pipelines, `aclose()`

## Part 4 — I/O, Networking, and Real Systems

16. [Streams: TCP Clients and Servers](chapter_16_streams_tcp.md) — a broadcast chat server from scratch
17. [Subprocesses with AsyncIO](chapter_17_subprocesses.md) — avoiding stdin/stdout deadlock
18. [Bridging Sync and Async Code](chapter_18_bridging_sync_async.md) — `to_thread`, `ThreadPoolExecutor` vs. `ProcessPoolExecutor`
19. [HTTP Clients in the Async World](chapter_19_http_clients_servers.md) — `aiohttp`/`httpx`, rate limiting vs. concurrency capping
20. [Async Database Access](chapter_20_async_database_access.md) — `asyncpg`, `aiomysql`, SQLAlchemy 2.0 async ORM

## Part 5 — Internals and Production Concerns

21. [Event Loop Internals](chapter_21_event_loop_internals.md) — `selectors`, `call_soon`, the deprecated policy system, `loop_factory`
22. [Debugging AsyncIO Applications](chapter_22_debugging_asyncio.md) — debug mode, and the Python 3.14 `pstree` CLI
23. [Testing Async Code](chapter_23_testing_async_code.md) — `pytest-asyncio`, `AsyncMock`, testing cancellation
24. [Performance and Benchmarking](chapter_24_performance_benchmarking.md) — percentile latency, `uvloop`, threads vs. processes

## Part 6 — Advanced Patterns and Design

25. [Production Design Patterns](chapter_25_production_design_patterns.md) — token buckets, jittered retries, circuit breakers
26. [Context Propagation and Signals](chapter_26_context_propagation_signals.md) — `contextvars`, `SIGINT`/`SIGTERM`, graceful shutdown
27. [AsyncIO vs. Alternatives](chapter_27_asyncio_vs_alternatives.md) — `trio`, `curio`, `anyio`

## Part 7 — Capstone

28. [Capstone Project: Concurrent Job Processing System](chapter_28_capstone_project.md) — an end-to-end resilient job runner tying the whole course together

---

## Prerequisites

- Comfortable with Python fundamentals (functions, classes, exceptions, generators)
- Python 3.13 or 3.14 recommended for following along with the newest material (free-threading notes, the `python -m asyncio pstree` tool, `TaskGroup`, `asyncio.timeout()`)

## How to Use This Course

Work through the parts in order — later chapters assume concepts from earlier ones (e.g., Chapter 8's `TaskGroup` builds directly on Chapters 5–7). Each chapter's exercises are meant to be run, not just read; several deliberately ask you to reproduce a bug first, then fix it, so the fix actually means something.
