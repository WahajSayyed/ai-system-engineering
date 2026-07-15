# Mastering Asynchronous Programming in Python — AsyncIO Curriculum

**Format:** Beginner → Advanced, theory + hands-on practice exercises in every chapter
**Inspiration:** Official `asyncio` docs, *Using Asyncio in Python* (Caleb Hattingh), *Python Concurrency with asyncio* (Matthew Fowler), David Beazley's coroutine talks, Lynn Root's "asyncio: We Did It Wrong", Real Python, and production patterns from FastAPI/aiohttp ecosystems.

---

## Part 1 — Foundations (Why Async, How It Works)

1. **Why Asynchronous Programming?**
   Concurrency vs. parallelism, I/O-bound vs. CPU-bound work, the GIL's role, and where async shines vs. threads/multiprocessing.

2. **A Brief History of Async in Python**
   Callbacks → generators-as-coroutines (`yield from`) → `async`/`await`. Why the syntax evolved this way.

3. **The Event Loop Explained**
   What an event loop actually is, cooperative multitasking, single-threaded concurrency, and a from-scratch mini event loop built with generators.

4. **Coroutines 101**
   `async def`, `await`, coroutine objects vs. coroutine functions, `asyncio.run()`, common beginner mistakes (forgetting to await, calling vs. scheduling).

## Part 2 — Core AsyncIO Building Blocks

5. **Tasks: Scheduling Concurrent Work**
   `asyncio.create_task`, task lifecycle, eager vs. lazy scheduling, task naming and introspection.

6. **Futures and Awaitables**
   `Future` vs. `Task` vs. `Coroutine`, the `Awaitable` protocol, how `await` really works under the hood.

7. **Running Things Concurrently: gather, wait, as_completed**
   Fan-out/fan-in patterns, ordering guarantees, exception aggregation differences between the three.

8. **Structured Concurrency with TaskGroups (3.11+)**
   Why unstructured `create_task` leaks tasks, `asyncio.TaskGroup`, automatic cancellation propagation.

9. **Cancellation Deep Dive**
   `Task.cancel()`, `CancelledError` semantics, shielding with `asyncio.shield`, cleanup with `try/finally`, cancellation pitfalls.

10. **Timeouts**
    `asyncio.timeout()` (3.11+), `wait_for`, timeout patterns for robust network code.

11. **Exception Handling in Concurrent Code**
    Exception groups (`except*`, 3.11+), swallowed exceptions in fire-and-forget tasks, `asyncio.gather(return_exceptions=True)`.

## Part 3 — Synchronization and Communication

12. **Synchronization Primitives**
    `Lock`, `Event`, `Condition`, `Semaphore`, `BoundedSemaphore` — when cooperative code still needs coordination.

13. **Queues and Producer-Consumer Patterns**
    `asyncio.Queue`, `PriorityQueue`, `LifoQueue`, worker pool patterns, backpressure.

14. **Async Context Managers**
    `async with`, `__aenter__`/`__aexit__`, `contextlib.asynccontextmanager`, resource lifecycle management.

15. **Async Iterators and Generators**
    `__aiter__`/`__anext__`, `async for`, async generator functions, streaming data pipelines.

## Part 4 — I/O, Networking, and Real Systems

16. **Streams: TCP Clients and Servers**
    `asyncio.open_connection`, `start_server`, building a simple chat server/client from scratch.

17. **Subprocesses with AsyncIO**
    `asyncio.create_subprocess_exec/shell`, piping stdin/stdout/stderr concurrently.

18. **Bridging Sync and Async Code**
    `loop.run_in_executor`, `ThreadPoolExecutor` vs. `ProcessPoolExecutor`, `asyncio.to_thread`, calling blocking libraries safely.

19. **HTTP Clients and Servers in the Async World**
    `aiohttp` and `httpx` async clients, building a concurrent web scraper/API caller with rate limiting.

20. **Async Database Access**
    `asyncpg`/`aiomysql`, SQLAlchemy 2.0 async ORM, connection pooling considerations.

## Part 5 — Internals and Production Concerns

21. **Event Loop Internals**
    Selectors (`select`/`epoll`/`kqueue`), the `loop.call_soon`/`call_later` scheduling model, event loop policies, `uvloop`.

22. **Debugging AsyncIO Applications**
    Debug mode, `PYTHONASYNCIODEBUG`, detecting blocking calls in the loop, slow-callback warnings, common anti-patterns.

23. **Testing Async Code**
    `pytest-asyncio`, mocking coroutines, testing timeouts/cancellation, fixtures for event loops.

24. **Performance and Benchmarking**
    Measuring throughput/latency, `uvloop` benchmarks, when threads/processes beat asyncio, avoiding accidental blocking.

## Part 6 — Advanced Patterns and Design

25. **Production Design Patterns**
    Rate limiting, retries with backoff, circuit breakers, graceful shutdown, connection pooling patterns.

26. **Context Propagation and Signals**
    `contextvars` across tasks, handling OS signals (`SIGINT`/`SIGTERM`) for graceful shutdown in async apps.

27. **AsyncIO vs. Alternatives**
    Comparing with `trio`'s structured concurrency model and `curio`; what asyncio borrowed from them (TaskGroups, timeouts).

## Part 7 — Capstone

28. **Capstone Project: Concurrent Job Processing System**
    Build an end-to-end async pipeline (e.g., a job queue with worker pool, rate-limited external API calls, graceful shutdown, and a small monitoring endpoint) tying together every concept from the course.

---

### Structure per chapter
- **Theory**: concepts, mental models, diagrams where useful
- **Code walkthroughs**: annotated, runnable examples
- **Hands-on exercises**: 3–5 practice problems per chapter, increasing in difficulty
- **Common pitfalls**: mistakes specific to that topic
- **Further reading**: pointers to source docs/PEPs where relevant

---

Ready to start with **Chapter 1: Why Asynchronous Programming?** whenever you are — just say the word.
