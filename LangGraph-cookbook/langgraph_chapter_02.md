# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 2 — Core Architecture: State, Nodes, and Edges

> *"At its core, LangGraph models agent workflows as graphs. You define the behavior of your agents using three key components: State, Nodes, and Edges... Nodes and Edges are nothing more than functions — they can contain an LLM or just good ol' code."*
> — LangGraph official documentation

---

### What This Chapter Covers

Chapter 1 built the theoretical foundation: why cyclic graphs, why Pregel, what a superstep means. Now we turn that theory into working code. This chapter is the hands-on core of the entire course — everything in every subsequent chapter is an elaboration of what you learn here.

By the end, you will:

1. Understand `StateGraph`, `TypedDict`, and `Pydantic` state schemas in complete, technical detail
2. Know every valid form a node function can take — sync, async, callable objects, lambdas, and nodes that accept `RunnableConfig`
3. Understand normal edges, conditional edges, the `path_map` parameter, the `then` parameter, and the `add_sequence` shorthand
4. Know exactly what `.compile()` does and what it validates
5. Know how to invoke and stream a graph, and what the config dict controls
6. Be able to use `get_graph()`, `draw_ascii()`, `draw_mermaid()` for visualization
7. Have built the first skeleton of MARRS — the capstone project

---

### 2.1 The Three Primitives: A Precise Definition

Before writing a line of code, let's establish precise definitions for each primitive. These are taken directly from the official LangGraph documentation and are worth memorizing:

**State** — *"A shared data structure that represents the current snapshot of your application."* It is the argument every node receives, and the source of every node's return value. In Pregel terms: the collection of channels.

**Nodes** — *"Functions that encode the logic of your agents. They receive the current state as input, perform some computation or side-effect, and return an updated state."* In Pregel terms: vertex compute functions.

**Edges** — *"Functions that determine which Node to execute next based on the current state. They can be conditional branches or fixed transitions."* In Pregel terms: the subscription relationships between channels.

The official documentation also makes an important point worth emphasizing: **"Nodes and Edges are nothing more than functions — they can contain an LLM or just good ol' code."** LangGraph imposes no constraint on what computation a node performs. It can call OpenAI. It can query a database. It can sort a list. The framework is indifferent.

---

### 2.2 State: The Shared Memory of Your Graph

#### 2.2.1 TypedDict: The Standard Approach

The most common way to define state is with Python's `TypedDict`. This gives you static type hints that editors and type checkers can use, without any runtime overhead or validation.

```python
from typing import TypedDict

class SimpleState(TypedDict):
    name: str
    count: int
    items: list[str]
    is_done: bool
```

`TypedDict` behaves like a regular Python dictionary at runtime — `state["name"]` works exactly as expected. But unlike a plain `dict`, it documents the expected shape and enables IDE autocompletion and `mypy` checking.

**What TypedDict does NOT do:**
- It does not validate values at runtime. If you pass `{"name": 123}`, no error is raised — `name` just holds `123` despite the `str` hint.
- It does not provide default values. Every key in the state you pass to `invoke()` must be present (unless you mark fields optional with `total=False` — covered below).
- It does not prevent extra keys. You can add keys not in the schema; they just won't be type-checked.

#### 2.2.2 `total=False`: Optional Fields

By default, `TypedDict` marks all fields as required (`total=True`). To make some fields optional — which is extremely common in real agents where many fields start empty:

```python
from typing import TypedDict

class AgentState(TypedDict, total=False):
    # All fields are optional — can be absent from the initial dict
    topic: str
    plan: str
    draft: str
    critique: str
    final: str
    revision_count: int

# This is valid even though most fields are absent
graph.invoke({"topic": "quantum computing"})
```

You can mix required and optional fields using inheritance:

```python
class RequiredFields(TypedDict):
    topic: str           # Always required

class AgentState(RequiredFields, total=False):
    plan: str            # Optional — generated later
    draft: str           # Optional — generated later
    revision_count: int  # Optional — starts at 0
```

This pattern is extremely useful in practice: `topic` must always be provided by the caller, but the rest of the state is built up incrementally as the graph executes.

#### 2.2.3 Pydantic: Runtime Validation

For production systems where you want to catch malformed inputs at the boundary, use Pydantic:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class AgentState(BaseModel):
    topic: str = Field(description="The research topic")
    plan: Optional[str] = None
    draft: Optional[str] = None
    revision_count: int = Field(default=0, ge=0, le=10)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("topic")
    @classmethod
    def topic_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Topic cannot be empty or whitespace")
        return v.strip()
```

**Important caveats about Pydantic in LangGraph (from the official docs):**

1. Runtime validation only occurs on inputs to the **first node** in the graph, not on subsequent node outputs. If a node returns `{"revision_count": -5}`, Pydantic will not catch this.
2. The output of the graph is **not** a Pydantic instance — it is a plain dict. If you need a Pydantic object at the end, you must call `AgentState.model_validate(result)` yourself.
3. Pydantic's recursive validation can be slow for complex schemas. For high-throughput systems, consider `TypedDict` with manual validation in nodes instead.

#### 2.2.4 Dataclasses: A Third Option

Python dataclasses also work as state schemas. They are rarely used in LangGraph code in the wild, but they are supported:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AgentState:
    topic: str
    plan: Optional[str] = None
    revision_count: int = 0
```

The main advantage over TypedDict: default values are easy to declare. The main disadvantage: reduced Pydantic-style validation, and some LangGraph features (like `MessagesState` inheritance) require TypedDict or Pydantic.

#### 2.2.5 `MessagesState`: The Prebuilt Conversation Schema

LangGraph ships with one prebuilt state schema: `MessagesState`. It is simply a TypedDict with a single field, `messages`, already annotated with the `add_messages` reducer:

```python
# What MessagesState is internally (from langgraph/graph/state.py)
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

To use it and add your own fields, simply subclass it:

```python
from langgraph.graph import MessagesState

class MyState(MessagesState):
    # Inherited: messages: Annotated[list[BaseMessage], add_messages]
    # Add your own fields:
    topic: str
    draft: str
    revision_count: int
```

`MessagesState` is the right starting point for any graph that uses an LLM with a conversation history. For MARRS, we will use it as our base.

#### 2.2.6 The `total=False` Initialization Problem

One important practical issue: when you initialize a `TypedDict` without `total=False`, you must provide all keys in the initial input to `invoke()`. This is a common source of confusion:

```python
class State(TypedDict):
    a: str
    b: str
    c: str

# This will fail — 'b' and 'c' are missing
graph.invoke({"a": "hello"})   # KeyError or validation error

# The fix: use total=False
class State(TypedDict, total=False):
    a: str
    b: str
    c: str

# Now this is valid
graph.invoke({"a": "hello"})   # b and c will be absent until a node sets them
```

---

### 2.3 Nodes: The Units of Computation

#### 2.3.1 The Node Contract

A node is any Python callable that:
1. Takes the current state as its first argument
2. Returns a dict of updates (not the full state — just the fields to change)

That's it. The node contract is intentionally minimal. LangGraph imposes no other requirements.

```python
def my_node(state: AgentState) -> dict:
    # Read from state
    current_count = state["count"]
    
    # Do any computation
    new_count = current_count + 1
    
    # Return ONLY the fields you want to change
    return {"count": new_count}
    # Fields not mentioned here are left unchanged
```

The return value is a partial update, not the full state. If your state has 10 fields and your node only changes 1, return a dict with just that 1 field. The other 9 remain exactly as they were. This is the message-passing model from Chapter 1 in action.

#### 2.3.2 What Nodes Can Return

Nodes can return several things. Understanding all of them is important:

**Return a dict** (most common):
```python
def my_node(state: AgentState) -> dict:
    return {"field": new_value}
```

**Return `None` or an empty dict** (no state update):
```python
def logging_node(state: AgentState) -> dict:
    print(f"Current state: {state}")
    return {}      # No state changes — this node is a side-effect only
    # Or equivalently: return None
```

This is a useful pattern for nodes that perform side effects (logging, metrics, notifications) without changing state.

**Return a `Command` object** (covered in depth in Chapter 4):
```python
from langgraph.types import Command

def routing_node(state: AgentState) -> Command:
    if state["quality_score"] > 0.8:
        return Command(goto="finalize", update={"status": "approved"})
    else:
        return Command(goto="revise", update={"status": "needs_work"})
```

`Command` lets a node combine a state update with a routing decision in a single return value.

#### 2.3.3 The Optional Second Parameter: `RunnableConfig`

Nodes can optionally accept a second parameter — a `RunnableConfig` object that carries runtime configuration:

```python
from langchain_core.runnables import RunnableConfig

def my_node(state: AgentState, config: RunnableConfig) -> dict:
    # Access runtime configuration
    thread_id = config["configurable"].get("thread_id")
    user_id = config["configurable"].get("user_id")
    
    # Access LangSmith tracing metadata
    run_id = config.get("run_id")
    
    # Access the current node name (useful when same function registered under different names)
    node_name = config["metadata"].get("langgraph_node")
    
    print(f"Node '{node_name}' running for thread '{thread_id}'")
    
    return {"processed_by": node_name}
```

**When to use `config`:**
- When a node needs to behave differently for different users, tenants, or sessions
- When a node needs to pass runtime values (API keys, user preferences) to sub-calls
- When a node needs access to LangSmith run metadata for tracing

**LangGraph auto-detects** whether your node accepts one argument or two. If your function signature is `def node(state)`, it is called with just state. If it is `def node(state, config)`, it is called with both. This introspection happens at compile time.

#### 2.3.4 Async Nodes

For I/O-bound work (LLM calls, database queries, HTTP requests), async nodes dramatically improve throughput. Write them as standard Python async functions:

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

async def async_writer_node(state: AgentState) -> dict:
    """Async node — returns a coroutine that LangGraph awaits."""
    
    # Multiple concurrent API calls in a single node
    research_task = llm.ainvoke(f"Research: {state['topic']}")
    outline_task = llm.ainvoke(f"Outline: {state['topic']}")
    
    # Both calls run concurrently — not sequentially
    research, outline = await asyncio.gather(research_task, outline_task)
    
    return {
        "research": research.content,
        "outline": outline.content
    }
```

To use async nodes, you must use `graph.ainvoke()` or `graph.astream()` at the invocation level. Mixing sync and async nodes in the same graph is allowed — LangGraph handles the coordination.

#### 2.3.5 Callable Objects as Nodes

Any callable works as a node — not just functions. This enables class-based nodes with initialization logic:

```python
class LLMNode:
    """A reusable node that wraps an LLM with a system prompt."""
    
    def __init__(self, model_name: str, system_prompt: str):
        self.llm = ChatOpenAI(model=model_name)
        self.system_prompt = system_prompt
    
    def __call__(self, state: AgentState) -> dict:
        from langchain_core.messages import SystemMessage
        
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

# Create specialized instances
researcher_node = LLMNode(
    model_name="gpt-4o",
    system_prompt="You are a thorough research assistant. Search carefully."
)
writer_node = LLMNode(
    model_name="gpt-4o",
    system_prompt="You are an expert technical writer. Write clearly and precisely."
)

# Register as nodes
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
```

#### 2.3.6 Lambda Nodes

For very simple transformations, lambdas work:

```python
# A node that just increments a counter
builder.add_node("increment", lambda state: {"count": state["count"] + 1})

# A node that clears a field
builder.add_node("reset_draft", lambda state: {"draft": "", "critique": ""})
```

Use lambdas sparingly. They make visualization output cryptic (node names show as `<lambda>` in traces) and they are hard to test in isolation. Prefer named functions for anything non-trivial.

#### 2.3.7 Private State: Nodes with Extra Channels

This is an advanced feature that Chapter 1's channel theory explains naturally. A node can declare state fields that only it uses — fields not in the main graph state schema. These are called **private channels**:

```python
from typing import TypedDict

# The main graph state — shared by all nodes
class GraphState(TypedDict):
    topic: str
    final_report: str

# Private state for the research node — only this node uses these fields
class ResearchPrivateState(TypedDict):
    search_queries: list[str]
    raw_results: list[str]
    source_urls: list[str]

def research_node(state: ResearchPrivateState) -> dict:
    # This node reads from the private schema, not the main GraphState
    queries = state.get("search_queries", [])
    # ... do research ...
    return {"final_report": "compiled research"}

# When registering, tell LangGraph which input schema this node uses
builder.add_node("research", research_node)
```

Private state is a powerful tool for keeping the main graph state clean and lean, while allowing individual nodes to work with richer internal data structures. We will use this pattern in MARRS.

---

### 2.4 Edges: Defining Control Flow

#### 2.4.1 `START` and `END`: The Special Sentinels

LangGraph uses two special sentinel objects to mark the entry and exit of the graph:

```python
from langgraph.graph import START, END
```

**`START`** — the virtual node that represents the caller's input. Adding an edge `START → "my_node"` tells LangGraph that `my_node` is the first node to execute. You can have multiple nodes connected to START, which causes them all to execute in the first superstep (parallel fan-out — covered in Chapter 15).

**`END`** — the virtual node that signals the graph has finished. A node connected to END with `add_edge("my_node", END)` will terminate the graph after it runs. You can connect multiple nodes to END, creating multiple possible exit points.

Internally, `START = "__start__"` and `END = "__end__"` — they are just string constants. You will see these strings in graph visualizations and trace logs.

#### 2.4.2 Normal Edges: `add_edge(source, destination)`

A normal edge creates an unconditional, always-taken connection:

```python
builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", END)
```

This is the equivalent of hardcoding a path through the graph. After `planner` runs, `researcher` *always* runs next — there is no decision logic.

**Normal edges are appropriate when:**
- The next step is always the same regardless of what the current step produced
- You are designing the fixed backbone of a workflow, with conditional logic added only at branch points

#### 2.4.3 Conditional Edges: `add_conditional_edges(source, path_fn, path_map)`

A conditional edge routes dynamically based on the current state:

```python
def route_after_critic(state: AgentState) -> str:
    """
    Decision function: inspect state, return a string key
    that maps to the next node.
    """
    if state.get("quality_score", 0.0) >= 0.80:
        return "approve"
    elif state.get("revision_count", 0) >= 3:
        return "force_approve"   # Max revisions reached
    else:
        return "revise"

builder.add_conditional_edges(
    "critic",           # Source node
    route_after_critic, # Routing function: state → string key
    {                   # path_map: key → destination node name
        "approve": "finalize",
        "force_approve": "finalize",
        "revise": "writer",
    }
)
```

The routing function receives the current state and returns a string key. The `path_map` dict maps those keys to actual node names (or `END`). This separation is valuable: the routing logic is expressed in natural terms ("approve", "revise"), while the mapping to actual node names is explicit and separately maintained.

**The `path_map` is optional but strongly recommended:**

```python
# Without path_map: routing function must return exact node names
def route_after_critic(state: AgentState) -> str:
    if state["quality_score"] >= 0.80:
        return "finalize"    # Must match exact node name
    else:
        return "writer"      # Must match exact node name

builder.add_conditional_edges("critic", route_after_critic)
```

Without `path_map`, the routing function must return exact node names (or the `END` sentinel). This is less readable and harder to maintain when node names change. Always prefer the `path_map` form in production code.

**Routing to END from a conditional edge:**

```python
from langgraph.graph import END

builder.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "approved": "finalize",
        "rejected": END,       # Map to END sentinel directly
        "revised": "writer",
    }
)
```

#### 2.4.4 The `then` Parameter: Shared Continuation

Sometimes multiple branches of a conditional edge converge on the same downstream node. Instead of repeating `add_edge` calls for each branch, use the `then` parameter:

```python
builder.add_conditional_edges(
    "validator",
    check_validity,
    {
        "valid_path_a": "process_a",
        "valid_path_b": "process_b",
    },
    then="merge"   # After EITHER process_a or process_b, always go to "merge"
)
# Equivalent to:
# builder.add_edge("process_a", "merge")
# builder.add_edge("process_b", "merge")
```

This is a pure convenience — it eliminates the need for explicit convergence edges. Use it whenever multiple conditional branches funnel into the same next step.

#### 2.4.5 `add_sequence`: The Convenience Shorthand (2025)

For simple linear sequences of nodes, `add_sequence` was added in LangGraph's 2025 release week to reduce boilerplate:

```python
# Before: explicit edge chaining
builder.add_edge("extract", "clean")
builder.add_edge("clean", "validate")
builder.add_edge("validate", "store")

# After: add_sequence handles all edges in one call
builder.add_sequence(["extract", "clean", "validate", "store"])
# Creates: extract → clean → clean → validate → validate → store
```

This is syntactic sugar only. Internally it calls `add_edge` for each consecutive pair. Use it for the straightforward portions of your graph; use explicit `add_edge` calls where sequence logic is non-obvious.

---

### 2.5 Building and Compiling a Graph

#### 2.5.1 The Builder Pattern

`StateGraph` follows the builder pattern: you create a `StateGraph` instance, call methods to add nodes and edges, then call `.compile()` to produce the runnable graph:

```python
from langgraph.graph import StateGraph, START, END

# Step 1: Create the builder, parameterized by your state schema
builder = StateGraph(AgentState)

# Step 2: Register nodes
builder.add_node("planner", planner_fn)
builder.add_node("researcher", researcher_fn)
builder.add_node("writer", writer_fn)
builder.add_node("critic", critic_fn)

# Step 3: Define edges
builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "critic")
builder.add_conditional_edges("critic", route_after_critic, {...})

# Step 4: Compile
graph = builder.compile()
```

#### 2.5.2 What `.compile()` Does — In Detail

`.compile()` is not just a formality. It performs several important operations:

**1. Structural validation:**
- Checks that every node registered with `add_node` is reachable from START (no orphaned nodes)
- Checks that all node names referenced in edges actually exist
- Checks that at least one path from START to END exists
- Raises a `ValueError` with a clear message if any of these fail

```python
# This will raise ValueError at compile time:
builder = StateGraph(State)
builder.add_node("orphan", some_fn)    # Added but never connected
builder.add_node("connected", other_fn)
builder.add_edge(START, "connected")
builder.add_edge("connected", END)
graph = builder.compile()
# ValueError: Node 'orphan' is not reachable from START
```

**2. Pregel instance construction:**
The `StateGraph` builder is a description layer. `.compile()` translates that description into an actual `langgraph.pregel.Pregel` instance — the runtime object. After this call, `graph` IS a `Pregel` object. All the channel subscriptions, node wrapping, and execution logic is assembled here.

**3. Runtime argument attachment:**
This is where you attach optional runtime components that control execution behavior:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer,    # Enables persistence and HITL
    interrupt_before=["critic"],  # Static breakpoint: pause before 'critic' runs
    interrupt_after=["writer"],   # Static breakpoint: pause after 'writer' runs
)
```

None of these can be attached after compilation — you must pass them here. (You can recompile the same builder with different runtime args without re-defining nodes and edges.)

**4. Recursion limit setup:**
The default `recursion_limit` is 25 supersteps. This is a safety net against infinite loops. Complex graphs may need more:

```python
# Override per-invocation (does not require recompilation)
result = graph.invoke(input, {"recursion_limit": 100})
```

If the limit is reached, LangGraph raises `langgraph.errors.GraphRecursionError`. This is not a compile-time setting — it is a per-invocation config value.

#### 2.5.3 Reusing the Builder

A `StateGraph` builder can be compiled multiple times with different runtime arguments. This is useful for creating different "modes" of the same graph:

```python
builder = StateGraph(AgentState)
# ... add nodes and edges ...

# Development mode: persistent memory, breakpoints for debugging
dev_graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["writer", "critic"]
)

# Production mode: PostgreSQL persistence, no breakpoints
prod_graph = builder.compile(
    checkpointer=PostgresSaver(conn),
)

# Testing mode: no persistence, no breakpoints
test_graph = builder.compile()
```

---

### 2.6 Invoking and Streaming a Graph

Once compiled, a `CompiledStateGraph` (which is a `Pregel` instance) exposes a standard interface for running graphs:

#### 2.6.1 `invoke()`: Synchronous Execution

`invoke()` runs the graph synchronously and returns the **final state** (all channel values after the last superstep):

```python
# Basic invocation
result = graph.invoke({"topic": "transformers"})
print(result)  # {'topic': 'transformers', 'plan': '...', 'draft': '...', ...}

# With config (thread_id enables persistence, recursion_limit adjusts safety net)
config = {
    "configurable": {"thread_id": "session-42"},
    "recursion_limit": 50
}
result = graph.invoke({"topic": "transformers"}, config)
```

The input to `invoke()` is the initial state. It must match the graph's input schema (or the full state schema if no separate input schema is defined).

#### 2.6.2 `stream()`: Incremental Output

`stream()` runs the graph and yields events as they happen. The `stream_mode` parameter controls what you receive:

```python
# "values" mode — yield the FULL state after each superstep
for state_snapshot in graph.stream({"topic": "transformers"}, stream_mode="values"):
    print(f"After step: {list(state_snapshot.keys())}")
    # {'topic': 'transformers'}
    # {'topic': 'transformers', 'plan': '...'}
    # {'topic': 'transformers', 'plan': '...', 'draft': '...'}
    # ...

# "updates" mode — yield (node_name, update_dict) for each node that ran
for event in graph.stream({"topic": "transformers"}, stream_mode="updates"):
    for node_name, updates in event.items():
        print(f"Node '{node_name}' changed: {list(updates.keys())}")
    # Node 'planner' changed: ['plan', 'search_queries']
    # Node 'researcher' changed: ['findings']
    # ...

# "messages" mode — yield (token_chunk, metadata) for LLM token streams
for chunk, metadata in graph.stream({"topic": "transformers"}, stream_mode="messages"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

**Choosing the right stream mode:**

| Use Case | `stream_mode` |
|----------|---------------|
| Debugging: see full state at each step | `"values"` |
| Efficient progress UI: show which agents are active | `"updates"` |
| Chatbot: stream tokens to the user | `"messages"` |
| Deep debugging: every internal event | `"debug"` |

Multiple modes can be requested simultaneously:

```python
for event in graph.stream(input, stream_mode=["updates", "messages"]):
    mode, data = event
    if mode == "updates":
        node_name, updates = data
        print(f"\n[{node_name}] updated: {list(updates.keys())}")
    elif mode == "messages":
        chunk, meta = data
        if chunk.content:
            print(chunk.content, end="", flush=True)
```

#### 2.6.3 `ainvoke()` and `astream()`: Async Execution

For concurrent production deployments (e.g., handling multiple users simultaneously in a FastAPI application), use the async variants:

```python
import asyncio

async def run_graph():
    result = await graph.ainvoke({"topic": "transformers"}, config)
    return result

async def stream_graph():
    async for event in graph.astream({"topic": "transformers"}, stream_mode="updates"):
        node_name, updates = event
        print(f"[{node_name}] {list(updates.keys())}")
```

Use `ainvoke`/`astream` whenever:
- Your nodes are async (using `await` inside them)
- You are handling multiple concurrent users
- You are building a web API endpoint

#### 2.6.4 The Config Dict: Runtime Control

The second argument to `invoke()` and `stream()` is a `RunnableConfig`-compatible dict. The most important keys:

```python
config = {
    # Required for persistence (memory across calls)
    "configurable": {
        "thread_id": "user-alice-session-1",   # Unique conversation identifier
        "user_id": "alice",                    # Your own custom runtime values
        "model_override": "gpt-4o-mini",       # Pass to nodes via config param
    },
    
    # Safety net against infinite loops (default: 25)
    "recursion_limit": 100,
    
    # LangSmith tagging and filtering
    "tags": ["production", "v2"],
    "metadata": {"environment": "prod", "customer_tier": "enterprise"},
    
    # Callback handlers for tracing/logging
    "callbacks": [...],
}
```

**The `configurable` sub-dict** is the escape hatch for passing runtime values into nodes without putting them in state. Anything you put there is accessible inside any node via `config["configurable"]["key"]`. This is the standard pattern for user IDs, tenant IDs, feature flags, and similar runtime parameters.

---

### 2.7 Graph Visualization

LangGraph builds a runtime graph representation that you can inspect and render. This is invaluable for debugging, documentation, and communication.

#### 2.7.1 `get_graph()`: The Visualization Object

`get_graph()` returns a `DrawableGraph` object — a lightweight representation of nodes and edges suitable for rendering:

```python
# Get the visualization object
viz = graph.get_graph()

# Render as ASCII (works anywhere, no dependencies)
print(viz.draw_ascii())

# Render as Mermaid source code (paste into mermaid.live)
print(viz.draw_mermaid())

# Render as PNG (requires internet access to mermaid.ink API, or pyppeteer)
from IPython.display import Image
Image(viz.draw_mermaid_png())

# For subgraph expansion: xray=True shows nested subgraph internals
viz = graph.get_graph(xray=True)
```

#### 2.7.2 Reading the ASCII Output

For the MARRS planner → researcher → writer → critic loop:

```
        +-----------+        
        | __start__ |        
        +-----------+        
               *             
               *             
               *             
         +---------+         
         | planner |         
         +---------+         
               *             
               *             
               *             
        +------------+       
        | researcher |       
        +------------+       
               *             
               *             
               *             
          +--------+         
          | writer |         
          +--------+         
               *             
               *             
               *             
          +--------+         
          | critic |         
          +--------+         
         **        **        
        *            *       
       *              *      
  +--------+     +----------+
  | writer |     | finalize |
  +--------+     +----------+
                      *      
                      *      
                      *      
                 +---------+ 
                 | __end__ | 
                 +---------+ 
```

The `**` symbols indicate conditional edges with multiple paths. The cycle back to `writer` is visible: `critic` has two outgoing paths.

#### 2.7.3 The Mermaid Output

The Mermaid output can be pasted into [mermaid.live](https://mermaid.live) for a rendered diagram:

```
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
	__start__([<p>__start__</p>]):::first
	planner(planner)
	researcher(researcher)
	writer(writer)
	critic(critic)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> planner;
	planner --> researcher;
	researcher --> writer;
	writer --> critic;
	critic -. revise .-> writer;
	critic -. approve .-> finalize;
	finalize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Conditional edges are shown with dashed arrows (`.->`) labeled with the routing key. This makes the graph's decision logic immediately visible.

---

### 2.8 Common Patterns and Anti-Patterns

Before building MARRS, let's catalogue the most important patterns and mistakes.

#### 2.8.1 Pattern: The Node Template

Every production node should follow this structure:

```python
import logging
from typing import Any
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

def my_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    One-line description of what this node does.
    
    Reads: state["field_a"], state["field_b"]
    Writes: state["field_c"]
    """
    logger.info(f"[my_node] Starting. Current: {state.get('field_a')}")
    
    # Do the work
    result = compute_something(state["field_a"], state["field_b"])
    
    logger.info(f"[my_node] Complete. Produced: {result[:50]}")
    
    # Return ONLY what changed
    return {"field_c": result}
```

The docstring convention matters: documenting which fields a node reads and writes serves as a lightweight data flow diagram that is always in sync with the code.

#### 2.8.2 Anti-Pattern: Mutating State In-Place

Do **not** mutate the state dict in place:

```python
# WRONG — mutating state in place
def bad_node(state: AgentState) -> dict:
    state["count"] += 1   # This mutates the live state object!
    return {}             # Returns empty update — but damage is done

# CORRECT — read, compute, return update
def good_node(state: AgentState) -> dict:
    return {"count": state["count"] + 1}
```

In-place mutation bypasses the reducer system entirely. If two nodes mutate the same field in the same superstep, the result is non-deterministic. Always treat the state you receive as read-only, and return updates.

#### 2.8.3 Anti-Pattern: Returning the Full State

Do **not** return a copy of the full state with one field changed:

```python
# WRONG — returns full state copy (wasteful, breaks reducers)
def bad_node(state: AgentState) -> dict:
    return {**state, "count": state["count"] + 1}

# CORRECT — return only the changed fields
def good_node(state: AgentState) -> dict:
    return {"count": state["count"] + 1}
```

Returning the full state passes every field through every reducer. For a field annotated with `operator.add`, this means the current list gets concatenated with itself — a subtle and hard-to-debug bug.

#### 2.8.4 Anti-Pattern: Missing Termination Condition

Every cycle needs a reliable termination condition. The most common mistake is a conditional edge that can return a value not in the `path_map`:

```python
def route_after_critic(state: AgentState) -> str:
    if state["quality_score"] >= 0.80:
        return "approve"
    else:
        return "revise"
    # Problem: if quality_score is None (not set yet), this crashes

builder.add_conditional_edges("critic", route_after_critic, {
    "approve": "finalize",
    "revise": "writer",
    # No safety fallback!
})
```

Always use `.get()` with defaults, and always have a fallback route:

```python
def route_after_critic(state: AgentState) -> str:
    score = state.get("quality_score", 0.0)
    revision_count = state.get("revision_count", 0)
    
    if score >= 0.80:
        return "approve"
    elif revision_count >= 3:     # Safety: never loop forever
        return "force_approve"
    else:
        return "revise"

builder.add_conditional_edges("critic", route_after_critic, {
    "approve": "finalize",
    "force_approve": "finalize",  # Covers the safety fallback
    "revise": "writer",
})
```

---

### 2.9 MARRS Checkpoint: Building the Skeleton

Let's now apply everything in this chapter to build the complete skeleton of MARRS. This version is functional — it runs end-to-end with placeholder implementations — so you can see the full graph structure before we make each component sophisticated in later chapters.

```python
# ── Installation ──────────────────────────────────────────────────────────────
# pip install langgraph langchain-openai

# ── Imports ───────────────────────────────────────────────────────────────────
import operator
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: STATE
# ─────────────────────────────────────────────────────────────────────────────

class MARRSState(TypedDict, total=False):
    """
    Central state for the Multi-Agent Research & Report Writing System.
    
    Using total=False so the initial invoke() only needs to provide 'topic'.
    All other fields are populated incrementally as the graph runs.
    
    Field annotations:
        messages      — append-only via add_messages (deduplicates by ID)
        findings      — append-only via operator.add (accumulates evidence)
        sources       — append-only via operator.add (accumulates citations)
        All others    — LastValue (overwrite on each update)
    
    Reads/Writes per node:
        planner       → writes: plan, search_queries
        researcher    → writes: findings, sources (via append reducers)
        writer        → writes: draft
        critic        → writes: critique, quality_score, revision_count
        human_review  → writes: human_approved, human_feedback
        finalize      → writes: final_report, status
    """
    # Required field — must always be provided
    topic: str
    
    # Accumulated conversation history (append-only via add_messages)
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Planner outputs
    plan: str
    search_queries: list[str]
    
    # Researcher outputs (accumulate across multiple search rounds)
    findings: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]
    
    # Writer / Critic cycle
    draft: str
    critique: str
    quality_score: float
    revision_count: int
    
    # Human review gate
    human_approved: bool
    human_feedback: str
    
    # Final output
    final_report: str
    status: str  # "in_progress" | "approved" | "rejected" | "complete"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: NODES
# Each node is a placeholder — we will replace these with full implementations
# in Chapters 9, 10, and 14. The structure is complete and correct now.
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: MARRSState, config: RunnableConfig) -> dict:
    """
    Reads: state['topic']
    Writes: plan, search_queries
    
    Creates a structured research plan for the given topic.
    In Chapter 9: uses LLM with structured output (Pydantic ResearchPlan).
    """
    print(f"  [Planner] Planning research for: {state['topic']}")
    
    # PLACEHOLDER: In Chapter 9, this calls an LLM with structured output
    plan = f"Research plan for: {state['topic']}"
    queries = [
        f"What is {state['topic']}?",
        f"Recent advances in {state['topic']}",
        f"Applications of {state['topic']}",
    ]
    
    return {
        "plan": plan,
        "search_queries": queries,
        "messages": [AIMessage(content=f"Created research plan: {plan}")]
    }


def researcher_node(state: MARRSState) -> dict:
    """
    Reads: state['topic'], state['plan'], state['search_queries']
    Writes: findings (appended), sources (appended)
    
    Runs a ReAct agent with search tools to gather evidence.
    In Chapter 9: full ReAct loop with Tavily search tools.
    """
    print(f"  [Researcher] Searching for: {state.get('search_queries', [])}")
    
    # PLACEHOLDER: In Chapter 9, this runs a ReAct agent with real search tools
    findings = [
        f"Finding 1 about {state['topic']}: key insight from search",
        f"Finding 2 about {state['topic']}: statistical data from research",
        f"Finding 3 about {state['topic']}: expert perspective from papers",
    ]
    sources = [
        "https://example.com/source1",
        "https://arxiv.org/source2",
    ]
    
    return {
        "findings": findings,    # Appended to existing list via operator.add
        "sources": sources,       # Appended to existing list via operator.add
    }


def writer_node(state: MARRSState) -> dict:
    """
    Reads: state['topic'], state['findings'], state['critique'] (if revision)
    Writes: draft
    
    Drafts or revises the research report.
    In Chapter 9: full LLM call with system prompt and critique context.
    """
    revision = state.get("revision_count", 0)
    critique = state.get("critique", "")
    
    if revision > 0 and critique:
        print(f"  [Writer] Revising draft (revision {revision}). Addressing: {critique[:60]}...")
    else:
        print(f"  [Writer] Writing initial draft on: {state['topic']}")
    
    # PLACEHOLDER: In Chapter 9, this calls an LLM with findings and critique context
    findings_summary = "; ".join(state.get("findings", [])[:2])
    draft = (
        f"# Report: {state['topic']}\n\n"
        f"## Overview\n{findings_summary}\n\n"
        f"## Analysis\n[Full analysis based on {len(state.get('findings', []))} findings]\n\n"
        f"## Conclusion\n[Synthesized conclusion]"
    )
    
    return {
        "draft": draft,
        "messages": [AIMessage(content=f"Draft written ({len(draft)} chars)")]
    }


def critic_node(state: MARRSState) -> dict:
    """
    Reads: state['draft'], state['topic']
    Writes: critique, quality_score, revision_count
    
    Evaluates the draft for quality, completeness, and accuracy.
    In Chapter 9: LLM with structured output (CritiqueResult Pydantic model).
    """
    draft = state.get("draft", "")
    current_revisions = state.get("revision_count", 0)
    
    print(f"  [Critic] Evaluating draft ({len(draft)} chars)...")
    
    # PLACEHOLDER: In Chapter 9, this calls an LLM to score and critique
    # For now, simulate increasing quality with each revision
    score = min(0.60 + (current_revisions * 0.15), 0.95)
    critique = (
        f"The draft covers the basic facts but lacks depth in key areas. "
        f"Revision {current_revisions + 1}: Add more specific data and citations."
        if score < 0.80 else "Good quality. Ready for human review."
    )
    
    print(f"  [Critic] Score: {score:.2f} — {'needs revision' if score < 0.80 else 'approved'}")
    
    return {
        "critique": critique,
        "quality_score": score,
        "revision_count": current_revisions + 1,
        "messages": [AIMessage(content=f"Critique complete. Score: {score:.2f}")]
    }


def human_review_node(state: MARRSState) -> dict:
    """
    Reads: state['draft'], state['quality_score']
    Writes: human_approved, human_feedback, status
    
    Pauses for human approval of the final draft.
    In Chapter 7: uses interrupt() for real async human-in-the-loop.
    For now: simulates automatic approval.
    """
    print(f"  [HumanReview] Simulating human review of draft...")
    print(f"  [HumanReview] Quality score: {state.get('quality_score', 0):.2f}")
    
    # PLACEHOLDER: In Chapter 7, this uses interrupt() to pause and await human input
    # For now, auto-approve if quality is sufficient
    if state.get("quality_score", 0) >= 0.75:
        print("  [HumanReview] Auto-approved (placeholder)")
        return {
            "human_approved": True,
            "human_feedback": "Approved (simulated)",
            "status": "approved"
        }
    else:
        print("  [HumanReview] Auto-rejected (placeholder) — quality too low")
        return {
            "human_approved": False,
            "human_feedback": "Rejected: quality insufficient",
            "status": "rejected"
        }


def finalize_node(state: MARRSState) -> dict:
    """
    Reads: state['draft'], state['topic'], state['quality_score']
    Writes: final_report, status
    
    Formats and packages the final report for delivery.
    """
    print("  [Finalize] Packaging final report...")
    
    final = (
        f"# {state['topic']}\n"
        f"*Research Report | MARRS System*\n\n"
        f"---\n\n"
        f"{state.get('draft', '[No draft available]')}\n\n"
        f"---\n"
        f"*Quality Score: {state.get('quality_score', 0):.2f} | "
        f"Revisions: {state.get('revision_count', 0)}*"
    )
    
    return {
        "final_report": final,
        "status": "complete"
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ROUTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

MAX_REVISIONS = 3

def route_after_critic(state: MARRSState) -> str:
    """
    After critic evaluation, decide: revise again OR proceed to human review.
    
    Termination guarantee: MAX_REVISIONS ensures the loop always terminates.
    This is a hard constraint independent of quality score — production safety.
    """
    score = state.get("quality_score", 0.0)
    revisions = state.get("revision_count", 0)
    
    if score >= 0.80:
        return "good_enough"
    elif revisions >= MAX_REVISIONS:
        print(f"  [Router] Max revisions ({MAX_REVISIONS}) reached. Forcing review.")
        return "max_reached"
    else:
        return "needs_revision"


def route_after_human_review(state: MARRSState) -> str:
    """
    After human review, decide: finalize OR reject OR send back to writer.
    """
    if state.get("human_approved"):
        return "finalize"
    elif state.get("status") == "rejected":
        return "reject"
    else:
        # Human provided edit instructions — treat as new critique
        return "revise"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_marrs_graph() -> StateGraph:
    """
    Build the MARRS StateGraph.
    Returns the builder (not compiled) so tests and different runtime
    configurations can compile it independently.
    """
    builder = StateGraph(MARRSState)
    
    # ── Register Nodes ────────────────────────────────────────────────────
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)
    
    # ── Define Edges ──────────────────────────────────────────────────────
    
    # Entry: planner is always first
    builder.add_edge(START, "planner")
    
    # Fixed sequence: planner → researcher → writer
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    
    # writer always submits to critic
    builder.add_edge("writer", "critic")
    
    # Critic branches: revise, or proceed to human review
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "needs_revision": "writer",    # Loop back for revision
            "good_enough":    "human_review",
            "max_reached":    "human_review",  # Force review after max revisions
        }
    )
    
    # Human review branches: approve/reject/revise
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "finalize": "finalize",
            "reject":   END,
            "revise":   "writer",
        }
    )
    
    # Finalize is the last step before END
    builder.add_edge("finalize", END)
    
    return builder


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: COMPILE AND RUN
# ─────────────────────────────────────────────────────────────────────────────

# Build the graph
marrs_builder = build_marrs_graph()

# Compile for development (no persistence yet — we add that in Chapter 5)
marrs_graph = marrs_builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def show_graph_structure():
    """Display the graph structure in multiple formats."""
    
    print("=" * 60)
    print("MARRS GRAPH STRUCTURE")
    print("=" * 60)
    
    # ASCII visualization
    print("\n--- ASCII Representation ---")
    print(marrs_graph.get_graph().draw_ascii())
    
    # Mermaid source (paste into mermaid.live)
    print("\n--- Mermaid Source ---")
    print(marrs_graph.get_graph().draw_mermaid())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: DEMONSTRATION RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_marrs_demo(topic: str) -> None:
    """
    Run MARRS with streaming output showing each node as it executes.
    """
    print(f"\n{'='*60}")
    print(f"MARRS: Starting research on '{topic}'")
    print(f"{'='*60}\n")
    
    # Use stream_mode="updates" for a clean progress display
    # Each iteration gives us {node_name: {updated_fields}} 
    final_state = None
    
    for event in marrs_graph.stream(
        {"topic": topic},
        stream_mode="updates"
    ):
        for node_name, updates in event.items():
            if node_name == "__start__":
                continue
            updated_keys = [k for k, v in updates.items() if v]
            print(f"✓ [{node_name}] → updated: {updated_keys}")
    
    # Get the final state after all nodes have run
    # (When no checkpointer is attached, we use invoke() to get the final state)
    result = marrs_graph.invoke({"topic": topic})
    
    print(f"\n{'='*60}")
    print("RESEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Status:          {result.get('status', 'unknown')}")
    print(f"Quality Score:   {result.get('quality_score', 0):.2f}")
    print(f"Revisions:       {result.get('revision_count', 0)}")
    print(f"Report length:   {len(result.get('final_report', ''))} characters")
    print(f"\n--- Final Report Preview ---")
    print(result.get("final_report", "No report generated")[:500])
    print("...")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: TEST HARNESS
# ─────────────────────────────────────────────────────────────────────────────

def test_marrs_structure():
    """
    Structural tests for the MARRS graph.
    Run these before touching any node implementation to catch wiring errors.
    """
    print("Running structural tests...")
    
    # Test 1: Graph compiled without errors
    assert marrs_graph is not None, "Graph failed to compile"
    print("  ✓ Graph compiled successfully")
    
    # Test 2: All expected nodes are registered
    node_names = set(marrs_graph.get_graph().nodes.keys())
    expected_nodes = {"__start__", "__end__", "planner", "researcher", "writer", 
                      "critic", "human_review", "finalize"}
    assert expected_nodes == node_names, f"Missing nodes: {expected_nodes - node_names}"
    print(f"  ✓ All {len(expected_nodes)} expected nodes present")
    
    # Test 3: Graph can be invoked without crashing
    result = marrs_graph.invoke({"topic": "test topic"})
    assert isinstance(result, dict), "Graph did not return a dict"
    print("  ✓ Graph invokes without error")
    
    # Test 4: Required output fields are present
    assert "final_report" in result or "status" in result, \
        "Graph output missing expected fields"
    print(f"  ✓ Output contains expected fields: {list(result.keys())}")
    
    # Test 5: The cycle terminates (no infinite loop with default topic)
    assert result.get("status") in ("complete", "rejected", "approved"), \
        f"Unexpected status: {result.get('status')}"
    print(f"  ✓ Graph terminated with status: {result.get('status')}")
    
    print("\nAll structural tests passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Show the graph structure first
    show_graph_structure()
    
    # Run the structural tests
    test_marrs_structure()
    
    # Demo run
    run_marrs_demo("The impact of transformer architectures on natural language processing")
```

#### 2.9.1 Running the Skeleton

When you run this file, you should see output like:

```
============================================================
MARRS GRAPH STRUCTURE
============================================================

--- ASCII Representation ---
        +-----------+        
        | __start__ |        
        +-----------+        
               *             
...
         +---------+         
         | planner |         
         +---------+         
...

Running structural tests...
  ✓ Graph compiled successfully
  ✓ All 8 expected nodes present
  ✓ Graph invokes without error
  ✓ Output contains expected fields: ['topic', 'messages', ...]
  ✓ Graph terminated with status: complete

All structural tests passed ✓

============================================================
MARRS: Starting research on 'The impact of...'
============================================================

  [Planner] Planning research for: The impact of...
✓ [planner] → updated: ['plan', 'search_queries', 'messages']

  [Researcher] Searching for: ['What is...', ...]
✓ [researcher] → updated: ['findings', 'sources']

  [Writer] Writing initial draft on: The impact of...
✓ [writer] → updated: ['draft', 'messages']

  [Critic] Evaluating draft (412 chars)...
  [Critic] Score: 0.60 — needs revision
✓ [critic] → updated: ['critique', 'quality_score', 'revision_count']

  [Writer] Revising draft (revision 1). Addressing: The draft covers...
✓ [writer] → updated: ['draft', 'messages']
...
```

The revision cycle will run until either the quality threshold is met or `MAX_REVISIONS` is reached, then human review runs (auto-approved in the placeholder), then finalize.

#### 2.9.2 What We Have Not Yet Implemented

This skeleton is structurally correct, but several placeholders remain for later chapters:

| Component | Status | Implemented In |
|-----------|--------|---------------|
| Real LLM calls in nodes | Placeholder | Chapter 9 |
| Actual web search tools | Placeholder | Chapter 9 |
| Persistence (checkpointer) | Not added | Chapter 5 |
| Real human-in-the-loop | Simulated | Chapter 7 |
| Streaming token output | Not used | Chapter 8 |
| Full multi-agent subgraph structure | Single flat graph | Chapter 11 |

This is intentional. You now understand the complete structural skeleton of a non-trivial multi-agent system, built entirely from the primitives covered in this chapter. Each subsequent chapter will replace one placeholder with the real implementation.

---

### 2.10 The Standard Build Sequence: A Checklist

Every LangGraph project follows the same build sequence. Memorize this:

```
1. Define State (TypedDict / Pydantic / dataclass)
   ├── Identify which fields need reducers (lists, accumulators)
   ├── Use Annotated[type, reducer_fn] for those fields
   └── Use total=False if not all fields need to be in the initial input

2. Define Nodes (Python functions)
   ├── Signature: def node(state: State) -> dict
   ├── Optional second param: config: RunnableConfig
   ├── Return dict with ONLY the fields being updated
   └── Never mutate state in-place

3. Define Routing Functions (for conditional edges)
   ├── Signature: def route(state: State) -> str
   ├── Always handle the None/missing case with .get() and defaults
   └── Always have a max-iterations safety fallback

4. Build the Graph
   ├── builder = StateGraph(State)
   ├── builder.add_node(name, function) for each node
   ├── builder.add_edge(START, first_node)
   ├── builder.add_edge / add_conditional_edges for all transitions
   └── builder.add_edge(last_node, END)

5. Compile
   └── graph = builder.compile(checkpointer=..., interrupt_before=...)

6. Test Structure
   ├── graph.get_graph().draw_ascii()  — confirm topology
   └── graph.invoke(minimal_input)     — confirm it runs

7. Iterate on Node Implementations
   └── The structure is locked; improve nodes independently
```

---

### 2.11 Chapter Summary

**State** is a TypedDict (or Pydantic model, or dataclass) that defines the shape of data flowing through your graph. It is the Pregel channels collection. Fields annotated with `Annotated[type, reducer_fn]` merge updates rather than overwrite. `MessagesState` is the prebuilt schema for conversation agents.

**Nodes** are Python callables: functions, async functions, lambda functions, or callable objects. They take the current state as input and return a dict of partial updates. They never mutate state in-place. They can optionally accept a `RunnableConfig` second argument for runtime configuration access.

**Edges** define the control flow. Normal edges are unconditional. Conditional edges call a routing function and map the result to a destination node. The `path_map` argument documents and validates possible routes. The `then` parameter handles shared downstream continuations. `add_sequence` reduces boilerplate for linear chains.

**`.compile()`** validates structure, constructs the Pregel runtime, and attaches checkpointers and breakpoints. It must be called before invocation.

**Invocation** uses `invoke()` for synchronous execution returning the final state, `stream()` for incremental events, and their async counterparts `ainvoke()` / `astream()`. The config dict passes `thread_id`, `recursion_limit`, and custom runtime values.

**Visualization** via `get_graph().draw_ascii()` and `get_graph().draw_mermaid()` produces renderable graph diagrams that match the actual execution topology.

---

### Further Reading

- **Official "Use the Graph API" guide**: `docs.langchain.com/oss/python/langgraph/use-graph-api` — the canonical reference for the topics in this chapter, with live code examples
- **LangGraph source: `langgraph/graph/state.py`**: The `StateGraph` implementation, where you can see exactly how `add_node`, `add_edge`, and `compile` work internally
- **LangGraph source: `langgraph/errors.py`**: `GraphRecursionError` and other runtime errors, with their causes and remedies documented inline
- **LangGraph troubleshooting guide**: `docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT` — common mistakes with cycles and how to fix them

---

*End of Chapter 2. Chapter 3: Reducers, Annotations, and Advanced State Design.*
