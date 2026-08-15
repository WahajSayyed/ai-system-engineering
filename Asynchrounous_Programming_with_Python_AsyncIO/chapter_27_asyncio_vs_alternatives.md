# Chapter 27: AsyncIO vs. Alternatives

> Part 6 — Advanced Patterns and Design · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 27.1 Why Alternatives Exist

`asyncio` wasn't the first async framework in Python, and it isn't the only one in active use today. Two others are worth understanding — not because you'll necessarily use them, but because they directly shaped features you've already learned in this course, and understanding *why* they exist retroactively explains some of asyncio's own design choices.

- **`curio`** (David Beazley, 2015) — a minimal, "clean room" async framework built specifically to demonstrate what async Python could look like designed from scratch, without any legacy baggage. It's the direct spiritual ancestor of Chapter 3's `TinyLoop` and Chapter 6's `SleepAwaitable` — Beazley's own talks on coroutines and concurrency (cited back in Chapter 3) are essentially a guided tour of `curio`'s internals.
- **`trio`** (Nathaniel J. Smith, 2017) — built on many of `curio`'s ideas, with **structured concurrency** as a first-class design principle from day one, and unusually rigorous attention to cancellation correctness.

---

## 27.2 Structured Concurrency: Where `TaskGroup` Actually Came From

Here's the direct lineage worth knowing: `trio`'s core primitive is the **nursery** — a scoped construct where you spawn child tasks, and the block cannot exit until every child has finished, with automatic cancellation of siblings on failure.

```python
# trio
async with trio.open_nursery() as nursery:
    nursery.start_soon(worker, "A")
    nursery.start_soon(worker, "B")

# asyncio (3.11+)
async with asyncio.TaskGroup() as tg:
    tg.create_task(worker("A"))
    tg.create_task(worker("B"))
```

This similarity is not a coincidence. `asyncio.TaskGroup` (Chapter 8) was **explicitly modeled after** `trio`'s nursery, brought into asyncio specifically because `trio` had already demonstrated, in production use, how much safer structured concurrency is compared to the unstructured `create_task()`/`gather()` patterns this course covered first (Chapters 5–7) — precisely because those patterns came *before* `TaskGroup` existed in asyncio's own history, and remain in the standard library for backward compatibility even now that the structured alternative exists.

---

## 27.3 Cancellation Philosophy: Retrofitted vs. Designed-In

`trio`'s cancellation model centers on **cancel scopes** — context managers like `trio.move_on_after(seconds)` and `trio.fail_after(seconds)` — conceptually the direct ancestor of `asyncio.timeout()` (Chapter 10), which was itself directly inspired by this exact pattern.

```python
# trio
with trio.move_on_after(5):
    await slow_operation()

# asyncio (3.11+)
async with asyncio.timeout(5):
    await slow_operation()
```

The deeper philosophical difference: **`trio` was designed structured-concurrency-first, from its very first release** — there's no unstructured "bare task, no enclosing scope" escape hatch encouraged anywhere in idiomatic `trio` code. **`asyncio` evolved structured concurrency features on top of an already-existing unstructured foundation** — `TaskGroup` and `timeout()` arrived in 3.11, years after `create_task()` and `gather()` were already deeply embedded in the ecosystem. This is exactly why this course spent Chapters 5–7 on the unstructured tools *before* introducing `TaskGroup` in Chapter 8: that's genuinely the order in which asyncio itself evolved, and unstructured patterns remain extremely common in real-world asyncio code you'll encounter, even though `TaskGroup` is now the recommended default for new code.

---

## 27.4 `curio`: A Teaching-First Reference Implementation

`curio`'s explicit design goal was extreme minimalism and clarity — implemented with very little "magic," specifically so its entire internals could be read and understood end to end. It's seen relatively little production adoption compared to `trio`, but remains genuinely valuable as a **reference implementation** for understanding async fundamentals with nothing hidden. Many of this course's "build it yourself" exercises — Chapter 3's toy event loop, Chapter 6's hand-rolled awaitable — are directly in that same spirit: understanding a simplified, transparent version first makes the real, more complex implementation far less mysterious.

---

## 27.5 Bridging the Ecosystems: `anyio`

Since `asyncio`, `trio`, and `curio` all have incompatible event loops and task models, you generally **can't** mix them directly in one program — an `asyncio.Task` and a `trio` nursery don't know how to talk to each other. **`anyio`** exists specifically to solve this: it provides **one API**, modeled heavily on `trio`'s nursery/cancel-scope design, that runs on top of either `asyncio` or `trio` as a pluggable backend — letting library authors write their async code once and support users of either ecosystem.

```python
import anyio

async def worker(name):
    await anyio.sleep(1)
    print(f"{name} done")

async def main():
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker, "A")
        tg.start_soon(worker, "B")

anyio.run(main)               # runs on asyncio by default
anyio.run(main, backend="trio")   # or explicitly on trio
```

This is also exactly why Chapter 23 mentioned the `anyio` **pytest plugin** as an alternative to `pytest-asyncio` — for a codebase built on `anyio` rather than directly on `asyncio`, it lets the same test suite run against either backend.

---

## 27.6 When Would You Actually Choose `trio` Over `asyncio`?

For nearly all new Python async projects today, **`asyncio` remains the practical default** — it's in the standard library, and virtually the entire practical ecosystem this course has covered (`aiohttp`, `httpx`, `asyncpg`, SQLAlchemy's async ORM, FastAPI) is built directly on it. Choosing an alternative means giving up a meaningfully larger, more battle-tested library ecosystem.

`trio` is a compelling choice specifically when:
- You're starting a genuinely new project with no legacy `asyncio` dependencies to integrate with.
- Rigorous, hard-to-misuse structured concurrency and cancellation semantics are an unusually high priority for your specific application.
- You're comfortable with (or specifically want) a smaller, more curated ecosystem in exchange for those guarantees.

In practice, most production async Python you'll encounter is `asyncio`-based, largely due to sheer ecosystem gravity — `trio` tends to show up in projects whose maintainers specifically value its design philosophy enough to accept a smaller surrounding ecosystem, or who use `anyio` to get a foot in both worlds at once.

---

## 27.7 What This Means for Everything You've Learned

The concepts filling this entire course — event loops, coroutines, cancellation, structured concurrency, timeouts, backpressure — are **not asyncio-specific ideas**. They're general async programming concepts that `trio` and `curio` also implement, just with different APIs and different philosophical emphases (and, in `trio`'s case, more rigorously enforced guarantees baked in from the start rather than retrofitted). Having built a deep, mechanical understanding of *why* asyncio works the way it does — not just its API surface — transfers directly to reading and understanding `trio` or `curio` code, should you ever need to.

---

## 27.8 Hands-On Exercises

**Exercise 1 — `trio` nursery vs. `TaskGroup`.**
Install `trio` (`pip install trio`). Rewrite one of your Chapter 8 `TaskGroup` exercises using `trio.open_nursery()` instead. Compare the two side by side and note the similarities and differences in exception handling.

**Exercise 2 — `move_on_after`/`fail_after` vs. `asyncio.timeout()`.**
Rewrite one of your Chapter 10 `asyncio.timeout()` exercises using `trio.move_on_after()` (which silently exits the block rather than raising) and separately with `trio.fail_after()` (which raises, like `asyncio.timeout()` does). Compare the API shapes and behaviors.

**Exercise 3 — One codebase, two backends with `anyio`.**
Install `anyio`. Write a single async function using only `anyio`'s API (task groups, cancellation scopes). Run it once with the default `asyncio` backend and once explicitly with `backend="trio"`. Confirm identical behavior under both.

**Exercise 4 — Read `curio`'s source.**
Read a short section of `curio`'s core `Task`/`Kernel` implementation (or its documentation) and compare it conceptually to Chapter 3's `TinyLoop`. Identify at least two direct similarities in approach.

**Exercise 5 (stretch) — Write your own comparison notes.**
Write a short (1–2 paragraph) personal reference note: for a new project you might realistically start, would you reach for `asyncio` or `trio`, and why? Consider the libraries you'd actually need, your team's familiarity, and how much structured-concurrency rigor genuinely matters for that specific project.

---

## 27.9 Common Pitfalls

- **Assuming you can mix `trio` and `asyncio` code directly in one program.** You generally can't, without `anyio` or deliberate, careful bridging — their event loops and task models are fundamentally incompatible.
- **Treating `trio` as objectively "better" rather than a different set of trade-offs.** It offers a smaller ecosystem in exchange for more rigorously enforced structured-concurrency guarantees — neither framework is a strict upgrade over the other.
- **Overlooking how much of asyncio's recent evolution came directly from `trio`.** `TaskGroup` and `timeout()` — arguably asyncio's two most important recent additions — are both directly inspired by `trio`'s nursery and cancel-scope design.
- **Assuming `curio` is commonly used in production today.** It's primarily valuable now as a clear, readable reference implementation for learning async fundamentals, rather than a common production choice.

---

## 27.10 Further Reading

- [trio documentation](https://trio.readthedocs.io/)
- Nathaniel J. Smith, ["Notes on structured concurrency, or: Go statement considered harmful"](https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/) — the canonical essay behind structured concurrency's rise across the whole industry, not just Python
- [curio GitHub repository](https://github.com/dabeaz/curio)
- [anyio documentation](https://anyio.readthedocs.io/)

---

**Next: Chapter 28 — Capstone Project: Concurrent Job Processing System** (the final chapter — building an end-to-end async pipeline that ties together nearly every concept from this entire course).
