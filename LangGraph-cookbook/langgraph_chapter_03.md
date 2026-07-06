# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 3 — Reducers, Annotations, and Advanced State Design

> *"The State consists of the schema of the graph as well as reducer functions which specify how to apply updates to the state... All Nodes will emit updates to the State which are then applied using the specified reducer function."*
> — LangGraph official documentation

---

### What This Chapter Covers

Chapter 2 introduced the three primitives and built MARRS's skeleton. You wrote state fields like `findings: Annotated[list[str], operator.add]` and used `add_messages` without fully understanding what they do or why they are necessary. This chapter closes that gap entirely.

By the end you will understand:

1. The exact mechanics of how LangGraph applies state updates — the update pipeline step-by-step
2. The default reducer (overwrite) and precisely when it is and is not appropriate
3. `Annotated`: what it actually is in Python, and what it means to LangGraph
4. Every significant built-in and user-defined reducer pattern, with working code and output traces
5. `add_messages` in complete detail: deduplication by ID, coercion, `RemoveMessage`, and when NOT to use it
6. The `Overwrite` type: bypassing a reducer when you need to reset accumulated state
7. How to design state schemas that don't create subtle bugs under parallel execution
8. Input/output schema separation: exposing a clean API while keeping rich internal state
9. A catalogue of real production reducer patterns for MARRS and beyond

---

### 3.1 The Update Pipeline: How LangGraph Actually Applies Changes

Before learning the specific reducers, you need to understand the pipeline they live inside. This is the mechanism that runs after every node completes, in the Update phase of each superstep (from Chapter 1).

When a node returns a dict like `{"findings": ["new item"], "status": "searching"}`, LangGraph does not simply assign these values. It runs the following procedure for each key in the returned dict:

```
For each (key, new_value) in the node's return dict:
    1. Look up the channel for 'key' in the compiled graph
    2. Check whether that channel has a reducer function attached
    3a. If YES: call reducer(existing_value, new_value) → merged_value
        Store merged_value as the new channel value
    3b. If NO:  store new_value directly, overwriting existing_value
    4. The merged_value becomes the state that all subsequent nodes see
```

This happens atomically after all nodes in a superstep finish — never during execution. The state a node reads is always the state from the end of the previous superstep.

**Why this matters:** The reducer is the contract between the node and the channel. A node that returns `{"findings": ["item"]}` is saying "here is my contribution." Whether that contribution overwrites everything or appends to a growing list depends entirely on the reducer attached to `findings`. Writing the wrong reducer is one of the most common and damaging bugs in LangGraph code.

---

### 3.2 The Default Reducer: Overwrite

When you declare a state field without `Annotated`, no reducer is attached. Any update to that field replaces the current value:

```python
from typing import TypedDict

class State(TypedDict):
    status: str       # No Annotated → default overwrite
    draft: str        # No Annotated → default overwrite
    count: int        # No Annotated → default overwrite
```

```python
# Trace of the overwrite behavior:

# Starting state
state = {"status": "idle", "draft": "", "count": 0}

# Node A returns:
node_a_output = {"status": "running", "count": 1}
# After Update phase: {"status": "running", "draft": "", "count": 1}

# Node B returns:
node_b_output = {"status": "done", "draft": "Final report text"}
# After Update phase: {"status": "done", "draft": "Final report text", "count": 1}
```

This is correct for fields representing the **current state** of something — a status flag, the latest draft, the current score, the active task name. These are semantically "what is it right now?", not "what has accumulated over time?".

**Use the default overwrite for:**
- Status strings (`"idle"`, `"running"`, `"done"`)
- Latest computed values (current quality score, current draft, latest critique)
- Boolean flags (`is_done`, `human_approved`)
- Scalar counters that should replace, not accumulate (`revision_count`)

**Do not use the default overwrite for:**
- Lists that accumulate over time (message history, search findings, sources)
- Any field that multiple parallel nodes will write to in the same superstep

#### 3.2.1 The Parallel Write Conflict

The overwrite behavior creates a silent bug when two nodes run in the same superstep (i.e., in parallel) and both write to the same `LastValue` channel. Consider:

```python
class State(TypedDict):
    results: list[str]   # No reducer — default overwrite

# Two nodes that run in parallel:
def search_a(state: State) -> dict:
    return {"results": ["result from A"]}

def search_b(state: State) -> dict:
    return {"results": ["result from B"]}

# Both are in the same superstep:
builder.add_edge(START, "search_a")
builder.add_edge(START, "search_b")
builder.add_edge("search_a", "merge")
builder.add_edge("search_b", "merge")
```

When both `search_a` and `search_b` write to `results` in the same superstep, only one value survives — LangGraph writes them both, and the last one applied "wins." Which one is last? The behavior is non-deterministic. You lose one result silently every time.

**This is why reducer design matters:** the solution is `Annotated[list[str], operator.add]`. We will work through this in detail in Section 3.5.

---

### 3.3 `Annotated`: What It Is and What LangGraph Does With It

`Annotated` is a standard Python typing construct from the `typing` module (Python 3.9+, or `typing_extensions` for earlier versions). Its general purpose is to attach metadata to a type hint:

```python
from typing import Annotated

# The general form:
# Annotated[ActualType, *metadata]

# Examples from the standard library and LangGraph:
x: Annotated[int, "this is a docstring annotation"]
y: Annotated[str, "min_length=3", "max_length=50"]
z: Annotated[list[str], operator.add]   # LangGraph reducer pattern
```

Python's runtime does not act on `Annotated` metadata by default. Any framework that wants to use metadata in `Annotated` must explicitly inspect the type hints via `typing.get_type_hints()` or `__annotations__` and process the metadata itself.

**What LangGraph does:** When you compile a `StateGraph`, LangGraph inspects every field in your state schema using `typing.get_type_hints(State, include_extras=True)`. For each field, it checks whether the type hint is `Annotated`. If it is, it extracts the metadata (the second argument) and treats it as the reducer function for that channel. It then creates the appropriate `BinaryOperatorAggregate` channel (from Chapter 1) with that function.

The full chain looks like this:

```
TypedDict field:
    findings: Annotated[list[str], operator.add]
                  ↓
LangGraph compile() inspects the type hint:
    base_type = list[str]
    reducer_fn = operator.add
                  ↓
Creates a BinaryOperatorAggregate channel:
    BinaryOperatorAggregate(list[str], operator=operator.add)
                  ↓
At runtime, when node returns {"findings": ["item"]}:
    merged = operator.add(existing_list, ["item"])
           = existing_list + ["item"]
```

This is the complete picture. `Annotated[T, reducer]` is not magic — it is Python's standard metadata annotation system, used by LangGraph's compiler to configure channels.

**The metadata must be a callable.** LangGraph expects the second `Annotated` argument to be a function with signature `(existing_value, new_value) -> merged_value`. If it is not a callable, the field behaves as a `LastValue` (default overwrite) channel.

---

### 3.4 `operator.add` as a Reducer: The Standard List Accumulator

`operator.add` is Python's built-in addition operator exposed as a function. It calls `__add__` on the left operand:

```python
import operator

operator.add(1, 2)            # → 3       (int addition)
operator.add("a", "b")        # → "ab"    (string concatenation)
operator.add([1, 2], [3, 4])  # → [1, 2, 3, 4]  (list concatenation)
```

For LangGraph state, the list concatenation behavior is by far the most useful. When used as a reducer on a list field, `operator.add` accumulates items from every node that writes to that field:

```python
import operator
from typing import TypedDict, Annotated

class ResearchState(TypedDict):
    findings: Annotated[list[str], operator.add]

# Trace: three sequential nodes each write one finding
# Starting state: {"findings": []}
#
# Node 1 returns: {"findings": ["Finding about attention mechanisms"]}
# Reducer: [] + ["Finding about attention mechanisms"]
# State after step 1: {"findings": ["Finding about attention mechanisms"]}
#
# Node 2 returns: {"findings": ["BERT achieves state-of-the-art on GLUE"]}
# Reducer: ["Finding..."] + ["BERT achieves..."]
# State after step 2: {"findings": ["Finding...", "BERT achieves..."]}
#
# Node 3 returns: {"findings": ["GPT-3 demonstrates few-shot learning"]}
# Reducer: [...] + ["GPT-3 demonstrates..."]
# State after step 3: {"findings": ["Finding...", "BERT...", "GPT-3..."]}
```

The findings list grows monotonically. No node needs to know what other nodes have written — it just returns its own contribution, and `operator.add` handles the merge.

#### 3.4.1 Parallel Safety: The Key Property

The real power of `operator.add` (and reducers in general) is parallel safety. When two nodes run in the same superstep and both write to an `operator.add` channel, the results are deterministically merged:

```python
# Both search_a and search_b run in parallel (same superstep)
def search_a(state: State) -> dict:
    return {"findings": ["Result from source A"]}

def search_b(state: State) -> dict:
    return {"findings": ["Result from source B"]}

# After both complete, the Update phase calls:
# operator.add(existing_findings, ["Result from source A"])
# operator.add(result_above, ["Result from source B"])
# OR in some orderings:
# operator.add(existing_findings, ["Result from source B"])
# operator.add(result_above, ["Result from source A"])
#
# Either way, BOTH results end up in the list.
# Order may vary, but nothing is lost.
```

Without a reducer, one of those results would be silently discarded. This is the critical reason why parallel branches must use reducers on any shared list field.

#### 3.4.2 `operator.add` on Integers: Summation

For `int` or `float` fields, `operator.add` becomes summation. This is less commonly used than list accumulation, but is useful for counters where you want to increment by a specific amount, rather than just overwriting:

```python
class CostTracker(TypedDict):
    total_tokens: Annotated[int, operator.add]   # Accumulates token usage
    api_cost_cents: Annotated[float, operator.add]

# Node A returns {"total_tokens": 1200}  → adds 1200 to existing total
# Node B returns {"total_tokens": 800}   → adds 800 to existing total
# Final: total_tokens = 0 + 1200 + 800 = 2000
```

However, there is an important gotcha: **`operator.add` on integers cannot handle `None`**. If the initial state has `total_tokens: None` (because you used `total=False`) and a node returns `{"total_tokens": 1200}`, Python will raise `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'`. The fix is a custom reducer (covered in Section 3.7).

---

### 3.5 `add_messages`: The Production Message Reducer

`add_messages` is LangGraph's purpose-built reducer for conversational agents. It is available from `langgraph.graph.message`:

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

`add_messages` does considerably more than `operator.add`. Understanding exactly what it does — and what it does not — is critical for production agents.

#### 3.5.1 What `add_messages` Does

**1. Standard appending (same as `operator.add` for distinct messages):**

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

existing = [HumanMessage(content="Hello", id="msg-1")]
new = [AIMessage(content="Hi there!", id="msg-2")]

result = add_messages(existing, new)
# [HumanMessage(id="msg-1", content="Hello"),
#  AIMessage(id="msg-2", content="Hi there!")]
```

**2. Deduplication by message ID (update in place, not duplicate):**

Every LangChain message has an `id` field. If a new message has the same `id` as an existing message, `add_messages` updates the existing message in place rather than appending a second copy:

```python
existing = [
    HumanMessage(content="Hello", id="msg-1"),
    AIMessage(content="Hi there!", id="msg-2")
]

# Same ID as msg-2, but different content
update = [AIMessage(content="Hi there! How can I help you?", id="msg-2")]

result = add_messages(existing, update)
# [HumanMessage(id="msg-1", content="Hello"),
#  AIMessage(id="msg-2", content="Hi there! How can I help you?")]
# ↑ msg-2 was updated, not duplicated
```

This is critical for tool calls. When an LLM calls a tool, it produces an `AIMessage` with `tool_calls`. When the tool returns, the result is a `ToolMessage` with a matching `tool_call_id`. Both need to be in the message history. Without deduplication by ID, certain graph patterns would add the same `AIMessage` multiple times as the graph loops.

**3. Type coercion (accepts dicts and tuples):**

`add_messages` accepts not just `BaseMessage` objects but also raw dicts and `(role, content)` tuples, normalizing them to the appropriate `BaseMessage` subclass:

```python
existing = []

# Dict format
new1 = [{"role": "user", "content": "What is LangGraph?"}]
result1 = add_messages(existing, new1)
# [HumanMessage(content="What is LangGraph?", id="auto-generated-id")]

# Tuple format
new2 = [("assistant", "LangGraph is a graph-based agent framework.")]
result2 = add_messages(result1, new2)
# [HumanMessage(...), AIMessage(content="LangGraph is...")]
```

This makes it easy to initialize state with raw strings rather than constructing `BaseMessage` objects manually:

```python
result = graph.invoke({
    "messages": [("user", "Research transformer architectures")]
    #              ↑ tuple format — add_messages coerces to HumanMessage
})
```

#### 3.5.2 Why NOT to Use `operator.add` for Messages

Given that `add_messages` exists, you might wonder: can I just use `operator.add` for messages? Technically yes, but it creates problems:

- **No deduplication:** If you manually update state with `update_state()` (a feature covered in Chapter 5), the update appends a new copy of every message rather than updating existing ones in place. This is especially problematic for human-in-the-loop workflows where you want to edit a specific message.
- **No type coercion:** You must always pass `BaseMessage` objects; raw dicts and tuples are not coerced.
- **No `RemoveMessage` support:** The deletion mechanism (Section 3.5.3 below) only works with `add_messages`.

Always use `add_messages` for fields containing `BaseMessage` objects.

#### 3.5.3 `RemoveMessage`: Deleting Specific Messages

`add_messages` supports a special sentinel type: `RemoveMessage`. When you include a `RemoveMessage(id="some-id")` in an update, `add_messages` removes the message with that ID from the list:

```python
from langchain_core.messages import RemoveMessage

# Current messages:
# [HumanMessage(id="m1"), AIMessage(id="m2"), HumanMessage(id="m3")]

# Remove message m1:
update = [RemoveMessage(id="m1")]
result = add_messages(existing_messages, update)
# [AIMessage(id="m2"), HumanMessage(id="m3")]
```

**Practical use: Keeping the last N messages**

This is the standard pattern for managing context window size without losing the most recent messages:

```python
from langchain_core.messages import RemoveMessage

def trim_messages_node(state: ChatState) -> dict:
    """Keep only the last 10 messages in history."""
    messages = state["messages"]
    MAX_MESSAGES = 10
    
    if len(messages) > MAX_MESSAGES:
        # Messages to remove: everything except the last MAX_MESSAGES
        to_remove = messages[:-MAX_MESSAGES]
        return {
            "messages": [RemoveMessage(id=m.id) for m in to_remove]
        }
    return {}   # No trimming needed

# Wire it in:
builder.add_node("trim", trim_messages_node)
builder.add_edge("model", "trim")
builder.add_edge("trim", "model")   # Continue the loop
```

**Important constraint:** When removing messages, you must be careful not to violate the message format rules of the LLM you are using. Most LLMs require:
- The history to start with a `HumanMessage` (not a system or AI message)
- Every `AIMessage` with `tool_calls` to be immediately followed by the corresponding `ToolMessage` results
- No orphaned tool results without a preceding tool call

Removing messages carelessly can break the conversation structure and cause LLM API errors.

#### 3.5.4 `add_messages` and the CRDT Property

`add_messages` implements a CRDT-like (Conflict-Free Replicated Data Type) merge strategy. The deduplication by ID means it is both commutative and idempotent for the same set of messages: `add_messages(add_messages(a, b), b) == add_messages(a, b)`. Applying the same update twice does not create duplicates.

However, the retention policy is **append-only by default**. Messages are never deleted unless you explicitly send `RemoveMessage` signals. In long-running agentic loops, this means the message history grows unboundedly, eventually exceeding context windows. This is the "context leak" problem in production agents. Managing it with `RemoveMessage`, summarization nodes, or trimming (covered in Chapter 6) is a production requirement, not an optional optimization.

---

### 3.6 Custom Reducers: Any Binary Function

LangGraph accepts any binary callable as a reducer. A custom reducer is a function `(existing: T, new: T) -> T`. This is where you implement arbitrary merge semantics.

#### 3.6.1 The `None`-Safe List Accumulator

The most common custom reducer in practice is a `None`-safe version of `operator.add` for lists. The problem: `operator.add` raises `TypeError` when `existing` is `None`, which happens when the field is not included in the initial state (using `total=False`):

```python
import operator

# This crashes if existing is None:
operator.add(None, ["item"])   # TypeError: can only concatenate list to list

# Fix: a None-safe version
def reduce_list(existing: list | None, new: list | None) -> list:
    """
    Safely concatenate two lists, treating None as an empty list.
    
    Args:
        existing: The current accumulated list (may be None on first update)
        new: The new items to add (may be None if node returns explicitly)
    
    Returns:
        Concatenated list, never None
    """
    left = existing or []
    right = new or []
    return left + right

# Usage:
class State(TypedDict, total=False):
    findings: Annotated[list[str], reduce_list]   # Safe even when uninitialized
```

This is the pattern to use in any `total=False` TypedDict where list fields are not present in the initial input.

#### 3.6.2 Set-Based Deduplication

When you want to accumulate unique values only (no duplicates), use a set-based reducer:

```python
def reduce_unique(existing: list | None, new: list | None) -> list:
    """
    Accumulate unique values only. Maintains insertion order (Python 3.7+).
    
    Useful for: source URLs, unique categories, deduplicated search terms.
    """
    existing_set = set(existing or [])
    new_items = new or []
    
    result = list(existing or [])
    for item in new_items:
        if item not in existing_set:
            result.append(item)
            existing_set.add(item)
    return result

class State(TypedDict, total=False):
    source_urls: Annotated[list[str], reduce_unique]
    # Will never contain duplicate URLs, even if multiple nodes return the same URL
```

#### 3.6.3 Dict Merge Reducer

For accumulating key-value data across nodes:

```python
def merge_dicts(existing: dict | None, new: dict | None) -> dict:
    """
    Merge two dicts. New keys are added; conflicting keys are overwritten by new.
    
    Useful for: metadata accumulation, tool result caches, per-node statistics.
    """
    result = dict(existing or {})
    result.update(new or {})
    return result

class State(TypedDict, total=False):
    tool_results: Annotated[dict[str, str], merge_dicts]
    # Each node can add entries; no node loses data to another
```

#### 3.6.4 "Last Non-None Wins" Reducer

Sometimes a field should only be updated when a node actually produces a value — not when it is explicitly set to `None`:

```python
def last_non_none(existing, new):
    """
    Keep the new value only if it is not None. Otherwise keep existing.
    
    Useful for: optional computed values, fields that are only set on certain branches.
    """
    return new if new is not None else existing

class State(TypedDict, total=False):
    quality_score: Annotated[float, last_non_none]
    # A node that doesn't set quality_score returns None for that field
    # This reducer ensures that doesn't wipe out a previously computed score
```

#### 3.6.5 Windowed / Bounded Accumulator

For fields that should only keep the last N items (e.g., a rolling context window of tool calls):

```python
def keep_last_n(n: int):
    """
    Factory that creates a reducer keeping only the last N items.
    Useful for bounded-memory accumulation.
    """
    def reducer(existing: list | None, new: list | None) -> list:
        combined = (existing or []) + (new or [])
        return combined[-n:]   # Trim to last N
    return reducer

class State(TypedDict, total=False):
    # Keep only the last 5 tool call results in state
    recent_tool_results: Annotated[list[dict], keep_last_n(5)]
```

#### 3.6.6 Composing with `add_messages`

The LangGraph source shows a clean pattern for extending `add_messages` with additional logic:

```python
from langgraph.graph.message import add_messages

def custom_messages_reducer(existing, new):
    """
    Extend add_messages with preprocessing — e.g., strip empty content.
    """
    # Filter out empty messages before passing to add_messages
    if isinstance(new, list):
        new = [m for m in new if (hasattr(m, 'content') and m.content) or not hasattr(m, 'content')]
    
    # Delegate to add_messages for deduplication and coercion
    return add_messages(existing, new)
```

---

### 3.7 The `Overwrite` Type: Escaping a Reducer

Sometimes you have a field with a reducer — say `operator.add` for accumulating findings — but in a specific situation you want to **reset** it entirely rather than append to it. LangGraph provides the `Overwrite` wrapper for exactly this case.

```python
from langgraph.types import Overwrite

class State(TypedDict):
    findings: Annotated[list[str], operator.add]   # Normally accumulates

def reset_findings_node(state: State) -> dict:
    """Reset the findings list to start a new research phase."""
    return {
        # Without Overwrite: operator.add([existing], []) = [existing] (no change)
        # With Overwrite: bypasses operator.add entirely, sets to []
        "findings": Overwrite([])
    }
```

`Overwrite(value)` is a wrapper that tells LangGraph: "bypass the reducer for this key and set the channel directly to `value`." The channel's reducer is not called.

**When to use `Overwrite`:**
- Resetting an accumulator at the start of a new phase (e.g., new research round)
- Replacing an entire conversation history with a summarized version
- Implementing "undo" logic in human-in-the-loop workflows

**Important:** `Overwrite` only bypasses the reducer for the specific update where it is used. Subsequent updates still go through the reducer normally.

---

### 3.8 Mechanics Under Parallel Execution: The Full Picture

Now we can put all of this together into the complete picture of how reducers protect you under parallel execution. This is the scenario from Section 3.2.1, solved correctly:

```python
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

class ParallelResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]   # Reducer: safe to write in parallel
    sources: Annotated[list[str], operator.add]    # Reducer: safe to write in parallel
    status: str                                    # No reducer: last-write-wins (OK here
                                                   #   because only one branch writes this)

def search_academic(state: ParallelResearchState) -> dict:
    """Searches academic papers. Runs in parallel with search_web."""
    print(f"  [Academic] Searching for {state['topic']}...")
    return {
        "findings": [f"Academic finding about {state['topic']}: [paper abstract]"],
        "sources": ["https://arxiv.org/paper1", "https://arxiv.org/paper2"],
    }

def search_web(state: ParallelResearchState) -> dict:
    """Searches the web. Runs in parallel with search_academic."""
    print(f"  [Web] Searching for {state['topic']}...")
    return {
        "findings": [f"Web finding about {state['topic']}: [article summary]"],
        "sources": ["https://example.com/article1"],
    }

def synthesize(state: ParallelResearchState) -> dict:
    """Runs after both searches complete (fan-in). Sees all findings."""
    print(f"  [Synthesize] Working with {len(state['findings'])} findings")
    return {"status": "synthesized"}

builder = StateGraph(ParallelResearchState)
builder.add_node("academic", search_academic)
builder.add_node("web", search_web)
builder.add_node("synthesize", synthesize)

# Fan-out: both run in the same superstep
builder.add_edge(START, "academic")
builder.add_edge(START, "web")

# Fan-in: synthesize waits for both to finish
builder.add_edge("academic", "synthesize")
builder.add_edge("web", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()
result = graph.invoke({"topic": "transformer architectures", "findings": [], "sources": []})

print(f"\nFindings collected: {len(result['findings'])}")
print(f"Sources collected: {len(result['sources'])}")
# Findings collected: 2  ← Both findings preserved
# Sources collected: 3  ← All three sources preserved
```

Trace of the superstep execution:

```
Superstep 0: START seeds state
  State: {topic: "transformer...", findings: [], sources: [], status: ""}

Superstep 1: academic AND web run IN PARALLEL (same superstep)
  academic returns: {findings: ["Academic..."], sources: ["arxiv/1", "arxiv/2"]}
  web returns:      {findings: ["Web..."],      sources: ["example.com/1"]}

  Update phase (atomic):
    findings channel (operator.add):
      [] + ["Academic..."] = ["Academic..."]
      ["Academic..."] + ["Web..."] = ["Academic...", "Web..."]
    sources channel (operator.add):
      [] + ["arxiv/1", "arxiv/2"] = ["arxiv/1", "arxiv/2"]
      ["arxiv/1", "arxiv/2"] + ["example.com/1"] = ["arxiv/1", "arxiv/2", "example.com/1"]
    
  State: {findings: ["Academic...", "Web..."], sources: ["arxiv/1", "arxiv/2", "example.com/1"]}

Superstep 2: synthesize runs
  Receives state with BOTH findings from both parallel branches
  Returns: {status: "synthesized"}
  State: {... status: "synthesized"}
```

Without reducers, `findings` and `sources` would contain only the results from whichever node happened to write last.

---

### 3.9 Input and Output Schemas: Clean External APIs

By default, a `StateGraph` uses the same schema for three things: what the caller provides as input, what nodes pass between themselves, and what the graph returns as output. For simple graphs this is fine. For production systems exposed as APIs, you typically want to:

1. Accept a clean, minimal input from callers (just the user's question)
2. Maintain a rich internal state with all intermediate computations
3. Return a clean, minimal output (just the answer and metadata)

LangGraph supports this with explicit input and output schemas:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# What the caller provides (minimal public API)
class InputSchema(TypedDict):
    question: str

# What the graph returns (minimal public API)
class OutputSchema(TypedDict):
    answer: str
    sources: list[str]
    confidence: float

# The full internal state (rich, private)
class InternalState(InputSchema, OutputSchema):
    # Inherits question, answer, sources, confidence
    # Plus private internal fields:
    search_results: list[str]
    candidate_answers: list[str]
    reasoning_trace: str
    llm_calls_made: int

# Build with separate schemas
builder = StateGraph(
    InternalState,
    input=InputSchema,
    output=OutputSchema
)
```

With this configuration:
- `graph.invoke({"question": "..."})` — input only needs `question`; providing `search_results` would be accepted but is not required
- The graph returns only `answer`, `sources`, `confidence` — the internal fields (`search_results`, `reasoning_trace`, etc.) are filtered out
- Nodes communicate via `InternalState`, which includes all fields

**Practical example for MARRS:**

```python
class MARRSInput(TypedDict):
    topic: str   # Caller only needs to provide the research topic

class MARRSOutput(TypedDict):
    final_report: str
    quality_score: float
    revision_count: int
    status: str

class MARRSState(MARRSInput, MARRSOutput):
    # Internal state fields (not exposed externally):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    search_queries: list[str]
    findings: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]
    draft: str
    critique: str
    human_approved: bool
    human_feedback: str

marrs_graph = build_marrs().compile(
    checkpointer=checkpointer,
    # StateGraph(MARRSState, input=MARRSInput, output=MARRSOutput)
)

# Clean caller interface:
result = marrs_graph.invoke({"topic": "transformer architectures"})
# Returns only: {final_report: "...", quality_score: 0.87, revision_count: 2, status: "complete"}
# Internal fields (plan, messages, findings, etc.) are not exposed
```

**The mechanism:** When an output schema is defined, the compiled graph applies a final filter step before returning. It extracts only the keys present in the output schema from the final internal state. The internal state itself remains unchanged; the filtering is only applied to the return value.

---

### 3.10 Reducer Design Patterns: A Catalogue for Production

This section catalogues the reducer patterns you will reach for in real projects, with the appropriate use case for each.

#### 3.10.1 The Full State Pattern (Reference)

```python
import operator
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ProductionState(TypedDict, total=False):
    # ── Conversation history ────────────────────────────────────────────
    # Always use add_messages for BaseMessage lists.
    # Provides deduplication, coercion, and RemoveMessage support.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # ── Accumulated evidence ────────────────────────────────────────────
    # operator.add for lists that accumulate over time.
    # Use reduce_list (None-safe version) when using total=False.
    findings: Annotated[list[str], reduce_list]
    sources: Annotated[list[str], reduce_list]
    tool_outputs: Annotated[list[dict], reduce_list]
    
    # ── Current values (overwrite) ──────────────────────────────────────
    # No reducer needed. Latest value is always correct.
    topic: str
    draft: str
    critique: str
    quality_score: float
    revision_count: int
    status: str
    
    # ── Optional with None-safe last-value semantics ────────────────────
    # Only update when a node actually computes a value.
    human_feedback: Annotated[Optional[str], last_non_none]
    
    # ── Unique collection ───────────────────────────────────────────────
    # Deduplicated accumulation (e.g., visited URLs to avoid re-fetching).
    visited_urls: Annotated[list[str], reduce_unique]
    
    # ── Cost / usage tracking ───────────────────────────────────────────
    # Additive accumulation across all nodes.
    total_tokens_used: Annotated[int, safe_add]   # None-safe operator.add
```

#### 3.10.2 The `None`-Safe Numeric Accumulator

The most frequently needed custom reducer when using `total=False`:

```python
def safe_add(existing, new):
    """None-safe numeric addition. Essential with total=False state schemas."""
    return (existing or 0) + (new or 0)

class State(TypedDict, total=False):
    token_count: Annotated[int, safe_add]
    cost_cents: Annotated[float, safe_add]
```

#### 3.10.3 The Bounded History Reducer

For capping accumulated data at a maximum size, preventing unbounded growth:

```python
from functools import partial

def bounded_list(existing: list | None, new: list | None, max_size: int = 100) -> list:
    """Keep only the most recent max_size items."""
    combined = (existing or []) + (new or [])
    return combined[-max_size:]

# Create a version bound to max_size=50
bounded_50 = partial(bounded_list, max_size=50)

class State(TypedDict, total=False):
    event_log: Annotated[list[str], bounded_50]
    # Never grows beyond 50 items; older events are automatically evicted
```

#### 3.10.4 The Idempotent Status Reducer

For status fields that can only move "forward" through a defined progression:

```python
STATUS_ORDER = {"idle": 0, "planning": 1, "researching": 2, 
                "writing": 3, "reviewing": 4, "complete": 5, "failed": 5}

def forward_only_status(existing: str | None, new: str | None) -> str:
    """
    Status can only advance, never regress.
    Prevents a node from accidentally rolling back to an earlier state.
    """
    if existing is None:
        return new or "idle"
    if new is None:
        return existing
    
    existing_order = STATUS_ORDER.get(existing, -1)
    new_order = STATUS_ORDER.get(new, -1)
    
    return new if new_order >= existing_order else existing

class State(TypedDict, total=False):
    workflow_status: Annotated[str, forward_only_status]
    # Once "reviewing", can never go back to "planning" by accident
```

---

### 3.11 Testing Reducers in Isolation

A reducer is a pure function. It requires no LangGraph imports to test. Testing reducers in isolation catches bugs before they manifest in graph execution:

```python
import pytest
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

# ── Test operator.add ─────────────────────────────────────────────────────────

def test_operator_add_lists():
    """operator.add concatenates two lists."""
    assert operator.add([1, 2], [3, 4]) == [1, 2, 3, 4]

def test_operator_add_empty():
    """operator.add with empty lists."""
    assert operator.add([], ["item"]) == ["item"]
    assert operator.add(["item"], []) == ["item"]

def test_operator_add_none_raises():
    """operator.add raises TypeError on None — use reduce_list instead."""
    with pytest.raises(TypeError):
        operator.add(None, ["item"])

# ── Test reduce_list (None-safe) ──────────────────────────────────────────────

def test_reduce_list_normal():
    result = reduce_list(["a"], ["b"])
    assert result == ["a", "b"]

def test_reduce_list_none_existing():
    result = reduce_list(None, ["b"])
    assert result == ["b"]

def test_reduce_list_none_new():
    result = reduce_list(["a"], None)
    assert result == ["a"]

def test_reduce_list_both_none():
    result = reduce_list(None, None)
    assert result == []

# ── Test add_messages ─────────────────────────────────────────────────────────

def test_add_messages_appends():
    existing = [HumanMessage(content="Hello", id="1")]
    new = [AIMessage(content="Hi!", id="2")]
    result = add_messages(existing, new)
    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"

def test_add_messages_deduplicates_by_id():
    existing = [AIMessage(content="Original", id="msg-1")]
    update = [AIMessage(content="Updated", id="msg-1")]
    result = add_messages(existing, update)
    assert len(result) == 1         # No duplication
    assert result[0].content == "Updated"  # Content was updated

def test_add_messages_remove():
    existing = [
        HumanMessage(content="Hello", id="m1"),
        AIMessage(content="Hi", id="m2"),
    ]
    result = add_messages(existing, [RemoveMessage(id="m1")])
    assert len(result) == 1
    assert result[0].id == "m2"

def test_add_messages_coerces_tuple():
    result = add_messages([], [("user", "Hello")])
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"

def test_add_messages_coerces_dict():
    result = add_messages([], [{"role": "assistant", "content": "Hi"}])
    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
```

Running these tests gives you confidence that your reducer logic is correct *before* wiring it into a graph.

---

### 3.12 MARRS Checkpoint: Upgrading the State Schema

Now we can apply everything in this chapter to upgrade the MARRS `MARRSState` from Chapter 2. The key changes:

1. Replace `operator.add` with `reduce_list` for `None`-safety (since we use `total=False`)
2. Add `safe_add` for numeric accumulators
3. Add `reduce_unique` for source URLs
4. Separate input and output schemas
5. Add the bounded history reducer for tool outputs
6. Add proper documentation linking each field to its reducer choice

```python
import operator
from typing import TypedDict, Annotated, Optional
from functools import partial
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM REDUCERS (defined here, reusable across any project)
# ─────────────────────────────────────────────────────────────────────────────

def reduce_list(existing: list | None, new: list | None) -> list:
    """
    None-safe list concatenation.
    Required when using TypedDict with total=False, since unset fields are None.
    Equivalent to operator.add but handles None inputs gracefully.
    """
    return (existing or []) + (new or [])


def reduce_unique(existing: list | None, new: list | None) -> list:
    """
    Accumulate unique values only (no duplicates).
    Maintains insertion order. Use for URLs, IDs, tags.
    """
    existing_list = existing or []
    seen = set(existing_list)
    result = list(existing_list)
    for item in (new or []):
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def safe_add(existing, new):
    """
    None-safe numeric addition.
    Required with total=False schemas for int/float accumulators.
    """
    return (existing or 0) + (new or 0)


def last_non_none(existing, new):
    """
    Only update the field if the new value is not None.
    Prevents a node that doesn't compute this field from wiping it out.
    """
    return new if new is not None else existing


def bounded_list(existing: list | None, new: list | None, max_size: int = 50) -> list:
    """Keep only the most recent max_size items."""
    combined = (existing or []) + (new or [])
    return combined[-max_size:]


# ─────────────────────────────────────────────────────────────────────────────
# INPUT / OUTPUT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class MARRSInput(TypedDict):
    """
    Public input schema — what the caller must provide.
    All fields here are required (total=True by default).
    """
    topic: str


class MARRSOutput(TypedDict):
    """
    Public output schema — what the graph returns to the caller.
    Internal state fields (messages, plan, findings, etc.) are filtered out.
    """
    final_report: str
    quality_score: float
    revision_count: int
    status: str


# ─────────────────────────────────────────────────────────────────────────────
# FULL INTERNAL STATE
# ─────────────────────────────────────────────────────────────────────────────

class MARRSState(MARRSInput, MARRSOutput, total=False):
    """
    Internal state for the Multi-Agent Research & Report Writing System.
    
    Inherits from MARRSInput and MARRSOutput for schema separation.
    All additional fields use total=False (optional / populated incrementally).
    
    REDUCER KEY:
    ─────────────────────────────────────────────────────────────────────────
    Field                  │ Reducer         │ Why
    ─────────────────────────────────────────────────────────────────────────
    messages               │ add_messages    │ Full LangChain message history with
                           │                 │ deduplication, coercion, RemoveMessage
    findings               │ reduce_list     │ Accumulates across research rounds;
                           │                 │ None-safe for total=False init
    sources                │ reduce_unique   │ Deduplicated URL list;
                           │                 │ prevents duplicate citations
    tool_call_log          │ bounded_list    │ Recent tool calls for debugging;
                           │                 │ capped at 20 to prevent bloat
    total_tokens           │ safe_add        │ Running token counter across all
                           │                 │ LLM calls; None-safe
    plan                   │ overwrite       │ Latest plan replaces previous
    search_queries         │ overwrite       │ Latest query set replaces previous
    draft                  │ overwrite       │ Current draft replaces previous
    critique               │ overwrite       │ Latest critique replaces previous
    quality_score          │ last_non_none   │ Preserve score if node doesn't update
    revision_count         │ overwrite       │ Explicit counter set each revision
    human_approved         │ overwrite       │ Latest approval decision
    human_feedback         │ last_non_none   │ Preserve feedback if not re-set
    status                 │ overwrite       │ Latest workflow status
    ─────────────────────────────────────────────────────────────────────────
    """
    
    # ── Conversation / LLM interaction history ────────────────────────────────
    # add_messages: deduplication by ID, coercion, RemoveMessage support
    messages: Annotated[list[BaseMessage], add_messages]
    
    # ── Planner outputs (overwrite — latest plan is always current) ───────────
    plan: str
    search_queries: list[str]
    
    # ── Researcher outputs (accumulate across search rounds) ──────────────────
    # reduce_list: None-safe concatenation (needed with total=False)
    findings: Annotated[list[str], reduce_list]
    
    # reduce_unique: deduplicated URLs — no duplicate citations
    sources: Annotated[list[str], reduce_unique]
    
    # bounded_list: keep recent tool interactions for debugging; cap at 20
    tool_call_log: Annotated[list[dict], partial(bounded_list, max_size=20)]
    
    # ── Usage tracking ────────────────────────────────────────────────────────
    # safe_add: None-safe numeric accumulation across all LLM calls
    total_tokens: Annotated[int, safe_add]
    
    # ── Writer / Critic cycle (all overwrite) ─────────────────────────────────
    draft: str
    critique: str
    revision_count: int
    
    # last_non_none: only update quality_score when critic actually runs
    quality_score: Annotated[float, last_non_none]
    
    # ── Human review (overwrite / last_non_none) ──────────────────────────────
    human_approved: bool
    
    # last_non_none: preserve feedback from a prior round if not re-set
    human_feedback: Annotated[Optional[str], last_non_none]
    
    # ── Workflow status (overwrite — latest status is always current) ─────────
    # Note: MARRSOutput already declares status as str. This entry is here
    # only as documentation; it is inherited from MARRSOutput.
```

#### 3.12.1 Updating the `researcher_node` to Track Token Usage

With `total_tokens` now using `safe_add`, we can accumulate token counts:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

def researcher_node(state: MARRSState) -> dict:
    response = llm.invoke(state["messages"])
    
    # Extract token usage from the response metadata
    usage = response.usage_metadata if hasattr(response, 'usage_metadata') else {}
    tokens_used = usage.get("total_tokens", 0)
    
    return {
        "messages": [response],
        "total_tokens": tokens_used,   # safe_add will accumulate this
        "tool_call_log": [{             # bounded_list keeps last 20
            "node": "researcher",
            "tokens": tokens_used,
        }]
    }
```

#### 3.12.2 Updating the Graph to Use Separate Schemas

```python
from langgraph.graph import StateGraph, START, END

def build_marrs_graph_v2() -> StateGraph:
    """
    MARRS graph with separate input/output schemas.
    Callers only see topic → {final_report, quality_score, ...}.
    All internal state fields are hidden from the public API.
    """
    builder = StateGraph(
        MARRSState,
        input=MARRSInput,     # Caller provides only 'topic'
        output=MARRSOutput    # Caller receives only the output fields
    )
    
    # ... same nodes and edges as Chapter 2 ...
    # (We add the full implementation in Chapter 9)
    
    return builder

marrs_graph_v2 = build_marrs_graph_v2().compile()

# The external API is now clean:
result = marrs_graph_v2.invoke({"topic": "transformer architectures"})
# result = {"final_report": "...", "quality_score": 0.87,
#           "revision_count": 2, "status": "complete"}
# Internal fields (messages, plan, findings, etc.) are not included
```

---

### 3.13 Common Reducer Bugs and How to Diagnose Them

This section catalogues the bugs that actually appear in production code — not theoretical concerns.

#### 3.13.1 The Exponential Duplication Bug

**Symptom:** A list field that should have 5 items ends up with 5, then 10, then 20 items, growing exponentially.

**Cause:** Returning the full state instead of just updates, combined with a list reducer:

```python
# BUG: Returns the full existing state PLUS new items
def bad_researcher(state: MARRSState) -> dict:
    new_finding = "new research result"
    return {
        "findings": state["findings"] + [new_finding],  # WRONG
        #            ↑ current_list + [new_item]
        #            When operator.add runs: current_list + (current_list + [new_item])
        #            = current_list duplicated + [new_item]
    }

# CORRECT: Return only the new items
def good_researcher(state: MARRSState) -> dict:
    new_finding = "new research result"
    return {
        "findings": [new_finding],   # Just the new item; reducer handles the merge
    }
```

The reducer receives `(existing_findings, [new_finding])` and merges them. If you pre-concatenate the existing findings into your return value, the reducer then merges `(existing_findings, existing_findings + [new_finding])`, effectively doubling everything.

**Diagnosis:** Print the length of the field after each node. If it doubles at any node, that node is doing the concatenation manually.

#### 3.13.2 The `None` Crash

**Symptom:** `TypeError: can only concatenate list (not "NoneType") to list` on the first invocation.

**Cause:** Using `operator.add` on a list field in a `total=False` TypedDict, where the field is not present in the initial state. The field starts as `None`, and `operator.add(None, ["item"])` raises.

**Fix:** Use `reduce_list` instead of `operator.add` whenever using `total=False`.

#### 3.13.3 The Silent Last-Write-Wins

**Symptom:** In a parallel execution, results from one branch are always missing. Hard to reproduce because ordering is non-deterministic.

**Cause:** Using default overwrite on a field that two parallel nodes both write to.

**Diagnosis:** Check whether the missing field is written by multiple nodes that share a superstep (both connected to the same upstream source). If yes, the field needs a reducer.

#### 3.13.4 The Stale Value from Unused Field

**Symptom:** A field holds a value from a previous iteration, even though no node in the current iteration updated it.

**Cause/Non-Issue:** This is expected behavior. LangGraph does not reset fields between supersteps. Fields only change when a node explicitly updates them. If node A sets `quality_score = 0.75` in iteration 1, and no node updates `quality_score` in iteration 2, it remains `0.75` in iteration 2.

**When it is a bug:** If a node should be updating the field but isn't, check the node's return dict — it may be missing the key.

**When it is correct:** Fields that represent "the last known value of X" (quality score, current draft, latest critique) are supposed to persist from one iteration to the next.

---

### 3.14 Chapter Summary

**Reducers** are binary functions `(existing, new) → merged` that control how node updates are applied to state channels. They are the mechanism that makes LangGraph state both correct and safe under parallel execution.

**The default reducer** (no `Annotated`) is overwrite: the new value replaces the old. This is correct for scalars and "current value" fields. It is wrong for any field that two parallel nodes write to.

**`Annotated[T, reducer_fn]`** attaches a reducer to a TypedDict field. LangGraph's compiler reads this annotation and configures the corresponding channel (`BinaryOperatorAggregate`) with the provided function.

**`operator.add`** concatenates lists (or sums numbers). It is the standard list accumulator but crashes on `None` inputs — use `reduce_list` (a `None`-safe wrapper) in `total=False` schemas.

**`add_messages`** is the purpose-built reducer for `BaseMessage` lists. It provides: ID-based deduplication (update in place, not duplicate), type coercion (accepts dicts and tuples), and `RemoveMessage` support for targeted deletion. Always use it for message history fields.

**Custom reducers** are ordinary Python functions. Write them for `None`-safe accumulation, deduplication, dict merging, bounded histories, idempotent status progression, and any other merge semantics your domain requires.

**`Overwrite(value)`** bypasses a reducer for a single update, directly setting the channel value. Use it to reset accumulated state.

**Input/Output schemas** separate the public API from internal implementation. The caller provides only `InputSchema` fields; the graph returns only `OutputSchema` fields; nodes communicate via the full internal schema.

**Testing reducers in isolation** is fast, dependency-free, and catches the most common bugs before they enter graph code.

---

### Further Reading

- **Official "Process state updates with reducers" guide**: `docs.langchain.com/oss/python/langgraph/use-graph-api` — the canonical reference, with live worked examples
- **Official "Define input/output schemas" guide**: `docs.langchain.com/oss/python/langgraph/use-graph-api#define-input-and-output-schemas`
- **Official "Delete messages" how-to**: covers `RemoveMessage` patterns in depth
- **LangGraph source: `langgraph/graph/message.py`**: The `add_messages` implementation — reading it directly reveals exactly what the deduplication logic does
- **LangGraph source: `langgraph/channels/`**: The `BinaryOperatorAggregate` and `LastValue` channel implementations — shows how reducer functions are actually called during the Update phase

---

*End of Chapter 3. Chapter 4: Control Flow — Conditional Edges, Routing, and Cycles.*
