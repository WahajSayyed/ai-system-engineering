# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 14 — The Functional API: `@entrypoint` and `@task`

> *"Unlike many data orchestration frameworks that require restructuring code into an explicit pipeline or DAG, the Functional API allows you to incorporate these capabilities without enforcing a rigid execution model."*
> — LangChain official documentation

---

### What This Chapter Covers

Chapters 1 through 13 used exactly one API: `StateGraph` — nodes, edges, and an explicit state schema. That's the Graph API, and it's the right default for the reasons Chapter 1 laid out (the Pregel/superstep model benefits from an explicit structure you can visualize and reason about). It is not, however, the *only* API LangGraph offers. This chapter covers the second one: the **Functional API**, built on `@entrypoint` and `@task` — plain Python functions, `if` statements, and `for` loops, running on the exact same underlying runtime as everything in this book, with persistence, human-in-the-loop, streaming, and memory all still available.

By the end you will understand:

1. Why the Functional API exists alongside the Graph API rather than replacing it, and what "same underlying runtime" actually means in practice
2. `@entrypoint` and `@task`: the two building blocks, and the injectable parameters (`previous`, `store`, `writer`, `config`) each gives you access to
3. The determinism and replay model — how resuming a Functional API workflow differs subtly but importantly from resuming a graph, and the specific rule that keeps it correct
4. Idempotency and non-deterministic control flow: two documented pitfalls with wrong/right code, both consequences of the replay model
5. `entrypoint.final`: decoupling what a workflow returns from what it persists for its next invocation
6. Short-term memory via `previous`, long-term memory via the same `BaseStore` from Chapter 6
7. Parallel task execution, retry policies, and streaming — all familiar concepts from Chapters 4, 9, and 8, expressed in this paradigm instead
8. Human-in-the-loop with `interrupt()` — the same primitive from Chapter 7, with one twist specific to how the Functional API tracks multiple interrupts
9. The official decision criteria for choosing between the Graph API and the Functional API
10. MARRS checkpoint: a Functional API rewrite of a piece of the capstone, side by side with its Graph API equivalent

---

### 14.1 Why a Second API Exists

The official framing is direct: the Functional API lets you add LangGraph's key features — persistence, memory, human-in-the-loop, and streaming — to code that already uses ordinary Python control flow, with minimal restructuring. Where the Graph API asks you to think in nodes, edges, and an explicit state schema up front, the Functional API asks you to write a normal function with `if`/`for`/`while` and calls to other functions, and get checkpointing, resumability, and streaming underneath it without changing that shape.

**Both APIs share the same underlying runtime** — the same Pregel execution model, the same checkpointer interface from Chapter 5, the same `Store` from Chapter 6, the same `interrupt()`/`Command` mechanism from Chapter 7. This isn't marketing language; it has a concrete, useful consequence: you can mix the two paradigms in one application, calling a `StateGraph` from inside an `@entrypoint`, or a `@task` from inside a graph node. Neither API is the "real" one underneath — they're two different ergonomics on top of one execution engine.

**What genuinely differs between them** (the official comparison, worth internalizing before you pick one for a new project):

| Aspect | Graph API | Functional API |
|---|---|---|
| Control flow | Explicit nodes and edges | Ordinary Python `if`/`for`/function calls |
| Short-term memory | Requires an explicit `State` schema, often with reducers (Chapter 3) | Scoped to the function; no shared state object to declare |
| Checkpointing granularity | A new checkpoint after every superstep | Task results save into the *existing* checkpoint for that entrypoint invocation, rather than creating a new one per step |
| Visualization | `get_graph().draw_mermaid()` and friends work, because the structure is static and known at compile time | Not supported — the execution shape is generated dynamically at runtime, so there's no fixed graph to draw |

---

### 14.2 `@entrypoint`: The Workflow's Starting Point

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

@entrypoint(checkpointer=InMemorySaver())
def my_workflow(some_input: dict) -> int:
    # ordinary Python — logic, API calls, branching — with checkpointing
    # and interrupt support available underneath
    ...
    return result
```

An entrypoint's decorated function **must accept exactly one positional argument** — the workflow's input. If you need several pieces of data in, pass a dictionary as that one argument, the same way Chapter 2's node functions all took a single `state` parameter.

Decorating a function with `@entrypoint` produces a `Pregel` instance — the same underlying object type a compiled `StateGraph` produces — which is what gives you `.invoke()`, `.ainvoke()`, `.stream()`, and `.astream()`, exactly as you've used throughout this book:

```python
config = {"configurable": {"thread_id": "some_thread_id"}}

my_workflow.invoke(some_input, config)             # synchronous
await my_workflow.ainvoke(some_input, config)       # asynchronous
for chunk in my_workflow.stream(some_input, config):
    print(chunk)
async for chunk in my_workflow.astream(some_input, config):
    print(chunk)
```

**You will almost always want a checkpointer.** Without one, you lose persistence, `previous` (Section 14.6), and `interrupt()` support entirely — the same dependency Chapter 5 established for the Graph API applies here.

#### 14.2.1 Injectable Parameters

Just as Chapter 9's tools could request `InjectedState`/`InjectedStore`, an entrypoint function can request additional parameters the runtime injects automatically — declared by name and type, as keyword-only arguments:

| Parameter | Purpose |
|---|---|
| `previous` | The return value of the entrypoint's previous invocation on this `thread_id` — short-term memory (Section 14.6) |
| `store` | A `BaseStore` instance — the exact interface from Chapter 6, for cross-thread long-term memory |
| `writer` | A `StreamWriter` for emitting custom streamed data — needed specifically on async Python < 3.11; Section 14.9 covers streaming in full |
| `config` | The run's `RunnableConfig` — the same object Chapter 4's routing functions and Chapter 12's `langgraph_auth_user` lookups use |

```python
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter

@entrypoint(checkpointer=in_memory_checkpointer, store=in_memory_store)
def my_workflow(
    some_input: dict,
    *,
    previous: Any = None,
    store: BaseStore,
    writer: StreamWriter,
    config: RunnableConfig,
) -> ...:
    ...
```

---

### 14.3 `@task`: A Discrete, Checkpointed Unit of Work

```python
from langgraph.func import task

@task
def slow_computation(input_value):
    # a long-running operation — an API call, a document parse, an LLM call
    ...
    return result
```

A task has two defining properties:

1. **Asynchronous execution.** Calling a task returns immediately with a future-like object, not the result itself — this is what makes parallel execution (Section 14.7) natural.
2. **Checkpointing.** A task's result is saved to the checkpoint. On replay (Section 14.4), a completed task's result is *restored from the checkpoint*, not recomputed.

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(some_input: int) -> int:
    future = slow_computation(some_input)
    return future.result()          # block synchronously for the result

@entrypoint(checkpointer=checkpointer)
async def my_workflow_async(some_input: int) -> int:
    return await slow_computation(some_input)   # await asynchronously
```

**Tasks can only be called from inside an entrypoint, another task, or a Graph API node** — never directly from your top-level application code. This mirrors a rule you've seen before in a different guise: Chapter 9's tools only execute inside `ToolNode`; a task only executes inside the Pregel runtime an entrypoint (or graph) provides.

#### 14.3.1 When You Need a Task, Specifically

The official guidance names five concrete situations, and it's worth treating this as a checklist rather than intuition, since some of these are non-obvious until you've hit the failure mode:

- **Checkpointing a long-running operation**, so resuming doesn't recompute it
- **Human-in-the-loop workflows** — the official docs state this as a hard requirement: any randomness or side effect (an API call) in a workflow using `interrupt()` *must* be wrapped in a task, for reasons Section 14.5 makes concrete
- **Parallel execution** of I/O-bound work — multiple API calls that don't need to block each other
- **Observability** — a task gives LangSmith (Chapter 11) a discrete span to trace, the same way a graph node does
- **Retryable work** — attaching a `RetryPolicy` (Section 14.8, the same primitive from Chapter 9) to a specific unit of work

---

### 14.4 Determinism and the Replay Model

This section is the Functional API's version of Chapter 7's single most important fact about `interrupt()` — and it generalizes that fact rather than replacing it.

**When a Functional API workflow resumes, execution does not continue from the line where it paused.** It restarts from the very beginning of the `@entrypoint` function. What makes this safe rather than destructive is that **completed task results are restored from the checkpoint instead of being recomputed** — so the replay races forward through everything already done in essentially zero time, and only genuinely new work (or the just-resumed interrupt) actually executes.

```python
import time
from langgraph.func import entrypoint, task
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

@task
def write_essay(topic: str) -> str:
    time.sleep(1)   # a stand-in for a real long-running operation
    return f"An essay about topic: {topic}"

@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> dict:
    essay = write_essay(topic).result()
    is_approved = interrupt({"essay": essay, "action": "Please approve/reject the essay"})
    return {"essay": essay, "is_approved": is_approved}
```

```python
config = {"configurable": {"thread_id": "essay-1"}}
for item in workflow.stream("cat", config):
    print(item)
# {'write_essay': 'An essay about topic: cat'}
# {'__interrupt__': (Interrupt(value={'essay': '...', 'action': '...'}, id='...'),)}
```

```python
from langgraph.types import Command

for item in workflow.stream(Command(resume=True), config):
    print(item)
# {'workflow': {'essay': 'An essay about topic: cat', 'is_approved': True}}
```

On resume, `workflow` runs from its first line again. But `write_essay("cat")` is **not** re-executed — its result was checkpointed the first time through, so it's loaded instantly, and execution proceeds straight to `interrupt()`, which now has a resume value waiting for it. This is precisely Chapter 7's `interrupt()` re-execution rule, generalized from "the containing node re-runs" to "the containing entrypoint re-runs, with every already-completed task short-circuited by its checkpoint."

**The corollary rule, stated directly by the official docs:** to use human-in-the-loop features correctly, *all* non-deterministic work and side effects — API calls, file writes, random numbers, wall-clock reads — must live inside a `@task`, not directly in the entrypoint body. If they don't, they re-run on every replay, which is exactly the double-execution failure mode Chapter 7 warned about for graph nodes, now at the granularity of "anything not wrapped in a task."

---

### 14.5 Two Documented Pitfalls, With Wrong/Right Code

#### 14.5.1 Side Effects Outside a Task

```python
# WRONG — the file write executes again on every resume
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    with open("output.txt", "w") as f:
        f.write("Side effect executed")   # re-runs on replay — not what you want
    value = interrupt("question")
    return value

# CORRECT — wrapped in a task, so its result (completion) is checkpointed
@task
def write_to_file():
    with open("output.txt", "w") as f:
        f.write("Side effect executed")

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    write_to_file().result()
    value = interrupt("question")
    return value
```

#### 14.5.2 Non-Deterministic Control Flow

This is a subtler and more damaging version of the same problem — not just a side effect repeating, but the *shape of execution itself* changing between the original run and the replay:

```python
# WRONG — which task runs depends on wall-clock time, which differs on replay
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    t0 = inputs["t0"]
    t1 = time.time()          # different value on replay than on the original run
    if (t1 - t0) > 1:
        result = slow_task(1).result()
        value = interrupt("question")
    else:
        result = slow_task(2).result()
        value = interrupt("question")
    return {"result": result, "value": value}

# CORRECT — the non-deterministic read lives in a task, so replay reuses
# the ORIGINAL time value from the checkpoint instead of reading the clock again
@task
def get_time() -> float:
    return time.time()

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    t0 = inputs["t0"]
    t1 = get_time().result()   # checkpointed — same value every replay
    if (t1 - t0) > 1:
        result = slow_task(1).result()
        value = interrupt("question")
    else:
        result = slow_task(2).result()
        value = interrupt("question")
    return {"result": result, "value": value}
```

**Why this matters more than it looks:** if the branch taken changes between the original execution and a replay, the *sequence of `interrupt()` calls* can change too. Chapter 7 established that multiple interrupts are matched to resume values by strict call order. If non-deterministic control flow changes which branch — and therefore which `interrupt()` — executes on replay, a resume value meant for one question can be silently applied to a different one. Keeping non-deterministic reads inside tasks isn't just tidiness; it's what keeps the interrupt-matching guarantee from Chapter 7 intact in a paradigm where "the containing function" is the whole entrypoint rather than one graph node.

**Idempotency, as a closely related concern:** because a task that started but didn't finish may run again on resume, design task side effects — especially data writes — to be idempotent, using an idempotency key or checking for an existing result before writing, exactly the same discipline Chapter 5 and Chapter 12 asked for around checkpointed writes and tool calls generally.

---

### 14.6 Short-Term Memory: the `previous` Parameter

With a checkpointer attached, an entrypoint can access the return value of its own previous invocation on the same thread via `previous`:

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> int:
    previous = previous or 0
    return number + previous

config = {"configurable": {"thread_id": "accumulator-1"}}
my_workflow.invoke(1, config)   # 1   (previous was None)
my_workflow.invoke(2, config)   # 3   (previous was 1, from the prior call)
```

By default, `previous` is exactly what the entrypoint returned last time. **`entrypoint.final`** lets you decouple that — return one value to the caller while checkpointing a *different* value for the next invocation's `previous`:

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    # Return `previous` to the CALLER, but save `2 * number` for the NEXT
    # invocation's `previous` value — these no longer have to match.
    return entrypoint.final(value=previous, save=2 * number)

config = {"configurable": {"thread_id": "1"}}
my_workflow.invoke(3, config)   # 0  (previous was None; saves 2*3=6)
my_workflow.invoke(1, config)   # 6  (previous was 6 from the prior call; saves 2*1=2)
```

This is the Functional API's answer to a need Chapter 3's reducers solved differently in the Graph API — controlling exactly what accumulates across turns — expressed here as a single, explicit split between "what the caller sees" and "what persists."

---

### 14.7 Parallel Task Execution

Because calling a task returns a future immediately, running several in parallel is ordinary Python — no `Send` API (Chapter 4), no special fan-out syntax:

```python
from langgraph.func import entrypoint, task
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-3.5-turbo")

@task
def generate_paragraph(topic: str) -> str:
    response = model.invoke([
        {"role": "system", "content": "You are a helpful assistant that writes educational paragraphs."},
        {"role": "user", "content": f"Write a paragraph about {topic}."},
    ])
    return response.content

@entrypoint(checkpointer=InMemorySaver())
def workflow(topics: list[str]) -> str:
    """Generates multiple paragraphs in parallel and combines them."""
    futures = [generate_paragraph(topic) for topic in topics]
    paragraphs = [f.result() for f in futures]
    return "\n\n".join(paragraphs)
```

Compare this to Chapter 4's `Send`-based dynamic fan-out, which achieves the same parallel-dispatch outcome in the Graph API: there, you return a list of `Send` objects from a routing function and the framework handles parallel scheduling explicitly as part of the graph's structure. Here, the parallelism is just a list comprehension over task calls, followed by collecting `.result()` from each future — genuinely closer to how you'd write concurrent code in plain Python, at the cost of that parallel structure not being visible in any static graph diagram (per the Section 14.1 comparison table's visualization row).

---

### 14.8 Retry Policy

`@task` accepts the exact same `RetryPolicy` object introduced in Chapter 9 for graph nodes — it isn't a different mechanism reimplemented for this API, just the same one attachable in a different place:

```python
from langgraph.types import RetryPolicy

api_retry = RetryPolicy(max_attempts=5, initial_interval=1.0, backoff_factor=2.0, jitter=True)

@task(retry_policy=api_retry)
def call_flaky_api(payload: dict) -> dict:
    ...
```

Both `StateGraph.add_node(...)` and `@task(...)` accept either a single `RetryPolicy` or a sequence of them — when you pass a sequence, the runtime applies the first policy in the list whose `retry_on` filter matches the exception that was actually raised, letting one task have different backoff behavior for, say, a rate-limit error versus a validation error, without hand-rolling that branching yourself.

---

### 14.9 Streaming

The Functional API streams the same three categories of information Chapter 8 covers for the Graph API — workflow progress, LLM tokens, and custom updates — through the same `.stream()`/`.astream()` methods an entrypoint's `Pregel` object provides:

```python
config = {"configurable": {"thread_id": "stream-1"}}
for chunk in workflow.stream(topics, config):
    print(chunk)
# {'generate_paragraph': '...'}   ← one such chunk per completed task
# {'workflow': '...'}             ← the entrypoint's own final return value
```

For custom streamed data from inside a task or entrypoint, request the injectable `writer` parameter (Section 14.2.1) — needed explicitly on async Python versions below 3.11, where the ambient stream-writer context variable isn't reliably available:

```python
@entrypoint(checkpointer=checkpointer)
def workflow(inputs: dict, *, writer: StreamWriter) -> dict:
    writer({"status": "starting research phase"})
    ...
```

This is the same `get_stream_writer()`-based instrumentation from Chapter 8's `custom` stream mode, just obtained via parameter injection here rather than a module-level function call.

---

### 14.10 Human-in-the-Loop

`interrupt()` and `Command(resume=...)` are, again, not a separate mechanism — they're exactly Chapter 7's primitives, used identically:

```python
config = {"configurable": {"thread_id": "review-1"}}

# Run until interrupt
for item in workflow.stream(input_data, config):
    print(item)

# Resume with the human's decision
from langgraph.types import Command
for item in workflow.stream(Command(resume=human_decision), config):
    print(item)
```

**Resuming after an error** (rather than after an `interrupt()`) uses a related but distinct pattern — invoke with `None` and the same `thread_id`, once you've fixed whatever caused the failure:

```python
config = {"configurable": {"thread_id": "some_thread_id"}}
my_workflow.invoke(None, config)   # resumes from the last good checkpoint
```

**The multi-interrupt rule from Chapter 7 applies here at the entrypoint level, not just within a single node.** LangGraph keeps a list of resume values per entrypoint/task, matched to `interrupt()` calls strictly by index/call order. Section 14.5.2's non-deterministic-control-flow pitfall is precisely what threatens this: if the path through your entrypoint changes between the original run and a replay, the *order* in which `interrupt()` calls are encountered can change, and a resume value meant for one question gets matched to a different one. The fix is identical in spirit to Chapter 7's guidance — keep the sequence of interrupt calls deterministic — but here the scope of "keep it consistent" is the whole entrypoint function, not one node.

---

### 14.11 Long-Term Memory

Long-term, cross-thread memory works through the exact same `BaseStore` interface Chapter 6 introduced, injected via the `store` parameter (Section 14.2.1) rather than the `*, store: BaseStore` node-level injection Chapter 6 used in the Graph API:

```python
@entrypoint(checkpointer=checkpointer, store=research_store)
def workflow(topic: str, *, store: BaseStore) -> dict:
    related = store.search(("research", "topic_facts"), query=topic, limit=5)
    # ... use related facts, same as Chapter 6's memory-enhanced planner ...
    store.put(("research", "topic_facts"), str(uuid.uuid4()), {"text": "a new finding"})
    return {...}
```

Everything from Chapter 6 about namespace design, semantic search configuration, and the three cognitive memory types applies unchanged — only the mechanism for getting a `store` reference into your function differs.

---

### 14.12 Choosing Between the Graph API and the Functional API

The official guidance, restated as a decision framework:

| Choose the Graph API when... | Choose the Functional API when... |
|---|---|
| You want a visualizable, inspectable structure (Chapter 4's `draw_mermaid`) | The workflow is more naturally expressed as ordinary branching Python code |
| Multiple nodes need to read and write a shared, explicit state schema (Chapter 3) | State is naturally scoped to the function and doesn't need a shared schema across many independent pieces |
| You're building a complex multi-agent architecture with explicit routing (Chapters 4, 10) | You're retrofitting LangGraph's persistence/HITL/streaming onto an existing, already-working piece of Python code with minimal restructuring |
| The team benefits from the graph's structure as living documentation | The team is more comfortable reasoning in terms of function calls than graph topology |

**In practice, most of this book's content — reducers, `Command`, `Send`, subgraphs, multi-agent patterns — is Graph-API-specific vocabulary**, which is why the curriculum taught the Graph API first and in depth: it's the richer, more structured tool, and understanding supersteps and channels (Chapter 1) pays off whichever API you eventually write in, since the Functional API runs on that same substrate underneath. Reach for the Functional API specifically when a piece of your system is better expressed as a function than as a graph — and remember that Section 14.1's "same underlying runtime" point means you can call one from the other, rather than needing to pick exactly one paradigm for an entire application.

---

### 14.13 MARRS Checkpoint: A Functional API Rewrite

To make the comparison concrete, here is a simplified slice of MARRS — plan, research, write, human-review — expressed as a single `@entrypoint`, next to a reminder of its Graph API shape from earlier chapters.

```python
"""
marrs/functional_variant.py

The same responsibilities as Chapter 4's hand-built graph and Chapter 10's
Subagents architecture, expressed as ordinary Python control flow instead.
"""
import uuid
from typing import Any
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore

from .memory import research_store, retrieve_similar_episodes
from .tools import web_search


llm_retry = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0)


@task(retry_policy=llm_retry)
def plan_research(topic: str, few_shot_context: str) -> dict:
    """Equivalent to Chapter 6's planner_with_memory node."""
    response = llm.invoke(f"Plan research for {topic}.{few_shot_context}")
    return {"plan": response.content, "search_queries": [f"{topic} overview", f"{topic} recent advances"]}


@task(retry_policy=llm_retry)
def run_search(query: str) -> str:
    """Equivalent to one parallel search worker from Chapter 4's Send-based fan-out —
    here, just one task among several called concurrently (Section 14.7)."""
    return web_search.invoke({"query": query})


@task(retry_policy=llm_retry)
def write_draft(topic: str, findings: list[str]) -> str:
    """Equivalent to Chapter 2's writer_node."""
    response = llm.invoke(f"Write a report on {topic} based on:\n" + "\n".join(findings))
    return response.content


@entrypoint(checkpointer=InMemorySaver(), store=research_store)
def marrs_functional(topic: str, *, store: BaseStore, previous: Any = None) -> dict:
    """
    The whole MARRS pipeline as one entrypoint. Compare this control flow —
    ordinary sequential calls, a list comprehension for parallel search, and
    a single interrupt() — against the explicit nodes/edges of Chapter 4's
    build_marrs_v3() or Chapter 10's supervisor graph.
    """
    # Long-term memory read — identical BaseStore usage to Chapter 6 (§14.11)
    episodes = retrieve_similar_episodes(topic, store)
    few_shot = "\n" + "\n".join(f"- {e.get('input')}: {e.get('plan', '')[:100]}" for e in episodes) if episodes else ""

    plan_result = plan_research(topic, few_shot).result()

    # Parallel search — Section 14.7's pattern, no Send API needed
    search_futures = [run_search(q) for q in plan_result["search_queries"]]
    findings = [f.result() for f in search_futures]

    draft = write_draft(topic, findings).result()

    # Human-in-the-loop — Chapter 7's exact interrupt()/Command(resume=...) mechanism,
    # with the multi-interrupt ordering rule from §14.5.2/§14.10 in mind: this is the
    # ONLY interrupt() call in this entrypoint, so there is no ordering ambiguity to guard.
    decision = interrupt({
        "message": "Please review this draft.",
        "draft": draft,
        "instructions": "Reply 'approve' or 'revise: <feedback>'",
    })

    if decision == "approve":
        # Long-term memory write — same store.put() calls as Chapter 6's write_research_memory
        store.put(("research", "episodes"), str(uuid.uuid4()),
                   {"input": topic, "plan": plan_result["plan"], "text": f"Research on: {topic}"})
        return {"final_report": f"# {topic}\n\n{draft}", "status": "complete"}

    return {"final_report": None, "status": "needs_revision", "feedback": decision}
```

```python
# Running it — identical invocation pattern to every graph in this book
config = {"configurable": {"thread_id": "marrs-functional-1"}}

for chunk in marrs_functional.stream("quantum error correction", config):
    print(chunk)
# {'plan_research': {...}}
# {'run_search': '...'}   (one per parallel search task)
# {'write_draft': '...'}
# {'__interrupt__': (Interrupt(value={'message': 'Please review this draft.', ...}, id='...'),)}

result = marrs_functional.invoke(Command(resume="approve"), config)
print(result["status"])   # "complete"
```

**What stayed identical between this and the Graph API versions in earlier chapters:** the checkpointer, the `Store`, `interrupt()`/`Command`, `RetryPolicy`, and the `.stream()`/`.invoke()` calling convention. **What changed:** there's no `StateGraph`, no explicit state schema, no `add_edge`/`add_conditional_edges` — control flow is just the order statements execute in, and parallel search is a list comprehension instead of a `Send`-based routing function. Both are complete, correct implementations of the same responsibilities; which one is more legible depends on whether you find a graph diagram or a function body easier to hold in your head for this particular workflow.

---

### 14.14 Chapter Summary

**The Functional API** (`@entrypoint`, `@task`) is a second, complementary way to build on LangGraph's runtime — ordinary Python control flow instead of an explicit graph — sharing the same Pregel execution engine, checkpointer interface, `Store`, and `interrupt()`/`Command` mechanism as the Graph API used throughout the rest of this book.

**`@entrypoint`** wraps a single-argument function into a `Pregel` object with the familiar `.invoke()`/`.stream()` family, and can request injected `previous` (short-term memory), `store` (long-term memory), `writer` (custom streaming), and `config` parameters. **`@task`** wraps a discrete, checkpointed unit of work, callable only from inside an entrypoint, another task, or a graph node — calling one returns a future immediately, which is what makes parallel execution a plain list comprehension rather than a `Send`-based routing function.

**The replay model generalizes Chapter 7's re-execution rule**: on resume, the entire entrypoint restarts from its first line, but completed task results are restored from the checkpoint rather than recomputed. This makes two things mandatory: side effects and non-deterministic reads (API calls, file writes, `time.time()`, random numbers) must live inside a `@task`, or they silently re-execute — and worse, if non-deterministic logic changes which branch of an `if` runs, the resulting shift in `interrupt()` call order can misroute a resume value to the wrong question, since matching is strictly index-based across the whole entrypoint.

**`entrypoint.final`** decouples what a workflow returns to its caller from what gets saved as `previous` for its next invocation — a different mechanism from Chapter 3's reducers, solving a similar "control what persists across turns" problem.

**Retry policies, streaming, and human-in-the-loop** are not reimplemented for this API — they're the identical `RetryPolicy`, stream modes, and `interrupt()`/`Command` primitives from Chapters 9, 8, and 7 respectively, attached or accessed slightly differently.

**Choosing between the two APIs** comes down to whether a workflow is more naturally a graph (explicit routing, multi-agent architectures, a structure worth visualizing) or a function (existing code you're retrofitting LangGraph features onto, logic that's more naturally sequential Python than a topology) — and nothing prevents calling one from inside the other in the same application.

---

### Further Reading

- **Official Functional API overview**: `docs.langchain.com/oss/python/langgraph/functional-api` — the primary source for this chapter: `@entrypoint`/`@task` definitions, injectable parameters, determinism, idempotency, and the documented pitfalls with wrong/right examples
- **Official "Use the Functional API" how-to**: `docs.langchain.com/oss/python/langgraph/use-functional-api` — parallel execution, calling graphs from entrypoints, retry policy, caching tasks, and a full chatbot example
- **Official "Choosing between Graph API and Functional API"**: `docs.langchain.com/oss/python/langgraph/choosing-apis` — the decision criteria summarized in Section 14.12
- **"Introducing the LangGraph Functional API"**: `blog.langchain.com/introducing-the-langgraph-functional-api` — the original announcement, with the essay-review example this chapter's Section 14.4 is grounded in
- **`langgraph.func` reference**: `reference.langchain.com/python/langgraph/func` — the complete `entrypoint`/`task` API signatures, including `entrypoint.final`

---

*End of Chapter 14.*
