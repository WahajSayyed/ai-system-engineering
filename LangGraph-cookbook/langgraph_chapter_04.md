# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 4 — Control Flow: Conditional Edges, Routing, Cycles, Command, and the Send API

> *"If you want to optionally route to one or more edges (or optionally terminate), you can use the `add_conditional_edges` method... Use `Command` instead of conditional edges if you want to combine state updates and routing in a single function."*
> — LangGraph official documentation

---

### What This Chapter Covers

Chapters 2 and 3 built the data layer: state schemas, reducers, and how updates are applied. This chapter builds the **control layer**: the mechanisms that determine execution order, branching logic, cycles, and dynamic routing. Control flow is what transforms a static pipeline into an agent.

By the end you will understand:

1. The complete API of `add_conditional_edges` — every parameter, including the ones the docs obscure
2. Routing functions: typing best practices, the `path_map`, how visualization depends on them
3. How to build the ReAct loop — the fundamental cycle every tool-calling agent uses
4. The `Command` object: what it is, when to use it instead of conditional edges, its interaction with static edges, and the `Command.PARENT` pattern for subgraphs
5. The `Send` API: dynamic fan-out (map-reduce), what makes it different from static parallelism, and when to reach for each
6. `RemainingSteps`: LangGraph's managed value for graceful recursion limit handling
7. `tools_condition`: the prebuilt routing function every ReAct agent uses
8. Every common cycle termination pattern with worked code
9. MARRS checkpoint: upgrading the routing functions using all of these tools

---

### 4.1 The Taxonomy of Control Flow in LangGraph

Before diving into any specific mechanism, it helps to have a map. LangGraph provides five distinct control flow instruments:

| Instrument | When to Use | Return Type |
|---|---|---|
| `add_edge(a, b)` | Always go from A to B | N/A |
| `add_conditional_edges(a, fn, map)` | Route to different nodes based on state | Router fn returns a string key |
| `Command(goto=..., update=...)` | Route AND update state in one step | Node returns `Command` |
| `Send(node, state)` (via conditional edge) | Dynamic fan-out to N parallel instances | Router fn returns `list[Send]` |
| `add_conditional_edges(START, fn)` | Choose the first node at runtime | Router fn returns string or `list[Send]` |

Each instrument exists to solve a specific structural problem. Choosing the right one is as important as the implementation.

---

### 4.2 `add_conditional_edges` in Complete Detail

#### 4.2.1 The Full Signature

```python
builder.add_conditional_edges(
    source,        # str: The node whose output triggers routing
    path,          # Callable: The routing function (state → str or list[Send])
    path_map,      # dict | list | None: Maps return values to destination nodes
    then,          # str | None: Always-run node after conditional branches complete
)
```

Only `source` and `path` are required. Understanding the optional parameters in depth is the difference between readable, maintainable graphs and fragile ones.

#### 4.2.2 The Routing Function: Contract and Type Annotations

The routing function is a regular Python function with a strict contract:

- **Input:** the current state (same schema as all other graph functions)
- **Output:** a string (node name or key), a list of strings, or a list of `Send` objects

```python
from typing import Literal

# Minimal form — returns a string key
def route_after_critic(state: MARRSState) -> str:
    score = state.get("quality_score", 0.0)
    if score >= 0.80:
        return "approve"
    return "revise"

# Typed form — Literal annotation enables graph visualization
def route_after_critic(state: MARRSState) -> Literal["approve", "revise"]:
    score = state.get("quality_score", 0.0)
    if score >= 0.80:
        return "approve"
    return "revise"
```

**Why `Literal` matters:** LangGraph Studio and `get_graph().draw_mermaid()` need to know which nodes a conditional edge can route to in order to draw the correct connections in the visualization. There are two ways to provide this information:

1. A `path_map` dict (works in Python and JavaScript)
2. A `Literal[...]` return type annotation on the routing function (Python only)

If you provide neither, LangGraph draws edges from the conditional node to *every* node in the graph — a confusing and misleading visualization that does not reflect actual runtime behavior. This was [a confirmed bug/quirk](https://github.com/langchain-ai/langgraph/issues/987) in early versions that the path_map was added to fix.

**Always provide either a `path_map` or `Literal` return annotations. Never leave a conditional edge underspecified.**

#### 4.2.3 The `path_map`: Decoupling Logic from Names

The `path_map` dict maps the routing function's string output to actual node names:

```python
def route_after_critic(state: MARRSState) -> str:
    """Returns semantic labels, not node names."""
    score = state.get("quality_score", 0.0)
    revisions = state.get("revision_count", 0)
    
    if score >= 0.80:
        return "good_enough"      # Semantic — not a node name
    elif revisions >= 3:
        return "max_reached"      # Semantic — not a node name
    else:
        return "needs_revision"   # Semantic — not a node name

builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "good_enough":   "human_review",   # semantic → node name
        "max_reached":   "human_review",   # two keys → same destination
        "needs_revision": "writer",
    }
)
```

This design has three important benefits:

1. **Readability:** The routing function uses domain language ("good_enough", "needs_revision") rather than implementation details ("human_review", "writer")
2. **Maintainability:** If you rename the `human_review` node, you only change the `path_map`, not the routing function logic
3. **Safety:** LangGraph validates that every possible return value from the routing function has a corresponding entry in the `path_map`. If the routing function returns a value not in the map, a `ValueError` is raised at runtime, not silently ignored

The `path_map` also accepts `END` as a value:

```python
builder.add_conditional_edges(
    "human_review",
    route_after_human_review,
    {
        "finalize":  "finalize",
        "rejected":  END,        # Terminate the graph from this branch
        "revise":    "writer",
    }
)
```

#### 4.2.4 The `path_map` as a List

When routing function return values and destination node names are identical, you can pass a list instead of a dict:

```python
# These two are equivalent:
builder.add_conditional_edges("critic", route, {"writer": "writer", "finalize": "finalize"})
builder.add_conditional_edges("critic", route, ["writer", "finalize"])
```

The list form is less explicit but occasionally useful for brevity when names are already self-documenting.

#### 4.2.5 The `then` Parameter: Shared Continuation After Branching

When two or more branches of a conditional edge all converge on the same downstream node, the `then` parameter eliminates redundant edge declarations:

```python
# Without `then`: explicit convergence edges
builder.add_conditional_edges("validator", check_validity, {
    "path_a": "process_a",
    "path_b": "process_b",
})
builder.add_edge("process_a", "merge")   # Both paths converge on 'merge'
builder.add_edge("process_b", "merge")

# With `then`: the convergence is declared once
builder.add_conditional_edges("validator", check_validity, {
    "path_a": "process_a",
    "path_b": "process_b",
}, then="merge")
# After EITHER process_a or process_b, 'merge' always runs
```

`then` is syntactic sugar. Under the hood it adds the corresponding `add_edge` calls automatically. Use it when the convergence pattern is clear and you want the structure to be self-documenting.

#### 4.2.6 Routing to Multiple Nodes in One Step

A routing function can return a list of strings, causing multiple nodes to execute in parallel:

```python
def route_to_multiple(state: State) -> list[str]:
    """Route to multiple nodes simultaneously."""
    return ["process_a", "process_b"]  # Both run in the same superstep

builder.add_conditional_edges("fan_out_node", route_to_multiple)
```

This is equivalent to adding multiple normal edges:
```python
builder.add_edge("fan_out_node", "process_a")
builder.add_edge("fan_out_node", "process_b")
```

Use the dynamic list form when the set of nodes to activate depends on runtime state. Use static multiple edges when it is always the same set.

#### 4.2.7 A Conditional Entry Point

You can attach conditional edges to `START` to dynamically choose the first node:

```python
from langgraph.graph import START

def choose_entry(state: MARRSState) -> str:
    """Route to different starting nodes based on request type."""
    if state.get("quick_mode"):
        return "fast_path"
    return "full_pipeline"

builder.add_conditional_edges(START, choose_entry, {
    "fast_path":    "writer",   # Skip planning and research
    "full_pipeline": "planner",
})
```

---

### 4.3 The Routing Function: Design Principles

A routing function looks simple but has several important design requirements in production code.

#### 4.3.1 The Three Rules of Routing Functions

**Rule 1: Always use `.get()` with defaults.** State fields declared with `total=False` may be `None` on first pass. A routing function that does `state["quality_score"] >= 0.80` raises `TypeError` if `quality_score` has not been set yet.

```python
# WRONG — crashes if quality_score is None
def bad_router(state: MARRSState) -> str:
    if state["quality_score"] >= 0.80:
        return "approve"
    return "revise"

# CORRECT — None-safe with a sensible default
def good_router(state: MARRSState) -> str:
    score = state.get("quality_score", 0.0)
    if score >= 0.80:
        return "approve"
    return "revise"
```

**Rule 2: Every cycle must have a guaranteed termination path.** The routing function must eventually return a path that leads to `END` or a non-cycling node. The typical mechanism is a maximum iterations counter checked in the router:

```python
MAX_REVISIONS = 3

def route_after_critic(state: MARRSState) -> str:
    score = state.get("quality_score", 0.0)
    revisions = state.get("revision_count", 0)
    
    # Termination guarantee: this branch always exits the loop
    if revisions >= MAX_REVISIONS:
        print(f"[Router] Max revisions ({MAX_REVISIONS}) reached, forcing exit.")
        return "max_reached"
    
    if score >= 0.80:
        return "approve"
    
    return "revise"
```

Without this, a graph with a poor-quality LLM can loop forever, burning tokens and eventually hitting `GraphRecursionError` with no graceful output.

**Rule 3: The routing function must be a pure read — no side effects.** Routing functions are called after node execution during the Update phase's edge resolution. If your routing function has side effects (API calls, writes to external systems), those effects happen during routing, not node execution — which makes them invisible to checkpointing and tracing. Routing functions should only read state and return a destination.

#### 4.3.2 Testing Routing Functions

Routing functions are pure Python functions. They are trivially testable without any LangGraph imports:

```python
import pytest

def test_route_approve():
    state = {"quality_score": 0.85, "revision_count": 1}
    assert route_after_critic(state) == "approve"

def test_route_revise():
    state = {"quality_score": 0.65, "revision_count": 1}
    assert route_after_critic(state) == "revise"

def test_route_max_revisions():
    # Even with low quality, max revisions forces exit
    state = {"quality_score": 0.45, "revision_count": 3}
    assert route_after_critic(state) == "max_reached"

def test_route_none_safe():
    # Initial state — quality_score not yet set
    state = {}
    # Should not crash; should default to "revise"
    assert route_after_critic(state) in ("revise", "max_reached")
```

Testing routing functions in isolation catches the most common control flow bugs before they require a full graph execution to surface.

---

### 4.4 The ReAct Loop: The Fundamental Agent Cycle

The ReAct (Reason + Act) pattern is the basic cycle underlying most tool-calling agents. Understanding it deeply is essential because it is the universal template for single-agent behavior in LangGraph.

#### 4.4.1 The Loop Structure

```
START
  ↓
agent_node (LLM reasons, decides whether to call a tool)
  ↓ ← conditional edge
  ├─── (if tool_calls present) → tool_node → back to agent_node
  └─── (if no tool_calls)     → END
```

The LLM produces an `AIMessage`. If that message contains `tool_calls`, the tools execute and return `ToolMessage` results. The full history (including tool results) feeds back into the LLM, which reasons about whether more tool calls are needed. This continues until the LLM produces a response with no `tool_calls`.

#### 4.4.2 Building a ReAct Loop from Scratch

```python
import operator
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # In production: call Tavily, SerpAPI, etc.
    return f"[Search results for '{query}': ...]"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

tools = [search_web, calculate]

# ── Model ─────────────────────────────────────────────────────────────────────
# bind_tools registers tool schemas with the model, enabling tool_calls in output
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# ── Nodes ─────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """
    The 'reason' half of ReAct.
    Calls the LLM with the full message history (including any tool results).
    Returns the LLM's response as a new message.
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
    # If response.tool_calls is non-empty, routing will send to tool_node.
    # If response.tool_calls is empty, routing will send to END.

# ToolNode is a prebuilt executor that:
# 1. Reads tool_calls from the last AIMessage
# 2. Executes each tool (potentially in parallel)
# 3. Wraps results as ToolMessage objects with matching tool_call_id
# 4. Returns {"messages": [tool_messages]}
tool_executor = ToolNode(tools)

# ── Routing ───────────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    The heart of the ReAct cycle.
    
    After the agent node runs, inspect the last message:
    - If the LLM decided to call tools → go to tool_node
    - If the LLM is done → terminate
    """
    last_message = state["messages"][-1]
    
    # AIMessage has a tool_calls attribute (list of dicts)
    # It is non-empty only when the model wants to call a tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "__end__"

# ── Graph Assembly ────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_executor)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools":    "tools",   # Route to tool execution
        "__end__":  END,       # Terminate the graph
    }
)

# After tools run, ALWAYS return to agent for the next reasoning step
builder.add_edge("tools", "agent")

react_graph = builder.compile()
```

The shape this creates is exactly the Pregel cycle from Chapter 1:

```
START → agent ──(tool_calls?)──→ tools ──→ agent ──(tool_calls?)──→ ...
                    ↓ (no tool_calls)
                   END
```

#### 4.4.3 `tools_condition`: The Prebuilt Routing Function

Because the `should_continue` function above (checking `last_message.tool_calls`) is identical in nearly every ReAct agent, LangGraph provides it as a prebuilt:

```python
from langgraph.prebuilt import tools_condition

# This is equivalent to the should_continue function above
builder.add_conditional_edges("agent", tools_condition)
```

`tools_condition` returns `"tools"` if the last message has tool calls, and `END` otherwise. It handles the edge cases (non-AIMessage as last message, etc.) more robustly than a hand-written version.

**When to use `tools_condition` vs. writing your own:**
- Use `tools_condition` for standard ReAct loops where the only decision is "call tools or stop"
- Write your own when you need additional routing logic (e.g., route to different tool nodes based on which tool was called, or route to a "reflection" node after N iterations)

#### 4.4.4 Error Handling in the ReAct Loop

`ToolNode` has built-in error handling that converts tool exceptions into `ToolMessage` objects with error content, so the LLM can reason about failures and potentially retry:

```python
from langgraph.prebuilt import ToolNode

# handle_tool_errors=True (default): tool exceptions → ToolMessage(status="error")
# The LLM receives the error message and can decide to retry or give up
tool_executor = ToolNode(tools, handle_tool_errors=True)

# Custom error formatter
def format_error(error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return "The search timed out. Try a more specific query."
    return f"Tool error: {str(error)[:200]}"

tool_executor = ToolNode(tools, handle_tool_errors=format_error)
```

Without error handling, a tool exception propagates up and crashes the graph. With it, the LLM receives feedback and can adapt.

---

### 4.5 The `Command` Object: Routing With State Updates

`Command` was introduced in December 2024 as a tool for combining state updates and routing decisions in a single return value from a node. It is now a core mechanism for multi-agent communication.

#### 4.5.1 What `Command` Solves

Consider a critic node that needs to:
1. Compute a quality score and critique
2. Based on that score, decide whether to route to `writer` or `finalize`

Without `Command`, you need two separate steps: the node computes and returns the score, then a separate routing function reads the score and decides. But the routing function re-reads state that was just computed — and worse, if you want to route based on a value computed *during* the node's execution (not persisted to state), you cannot do it with conditional edges alone.

With `Command`, the node can return both the state update and the routing decision atomically:

```python
from langgraph.types import Command
from typing import Literal

def critic_node(state: MARRSState) -> Command[Literal["writer", "human_review"]]:
    """
    Evaluates the draft and returns both:
    1. The updated state (critique, quality_score)
    2. The routing decision (where to go next)
    
    The Literal type annotation is required for correct graph visualization.
    """
    draft = state.get("draft", "")
    revisions = state.get("revision_count", 0)
    
    # Compute score (this value drives routing — done in the node itself)
    score = evaluate_draft_quality(draft)
    critique = generate_critique(draft, score)
    new_revision_count = revisions + 1
    
    # Decide destination based on just-computed score
    if score >= 0.80:
        destination = "human_review"
    elif new_revision_count >= MAX_REVISIONS:
        destination = "human_review"
    else:
        destination = "writer"
    
    return Command(
        update={
            "critique": critique,
            "quality_score": score,
            "revision_count": new_revision_count,
        },
        goto=destination
    )
```

When using `Command` for routing, you do not call `add_conditional_edges` from this node. The `Command` itself carries the routing. However, you still need to register valid destinations so the graph can be visualized:

```python
# No add_conditional_edges needed for critic_node — Command handles routing
# But you MUST register nodes that Command can route to:
builder.add_node("critic", critic_node)   # critic_node returns Command
builder.add_node("writer", writer_node)
builder.add_node("human_review", human_review_node)

# Normal edges from critic are not needed
# The Literal["writer", "human_review"] annotation on the return type
# tells LangGraph Studio to draw the correct conditional edges
```

#### 4.5.2 `Command` and Static Edges: They Compose

An important, non-obvious fact from the official documentation: **`Command` only adds dynamic edges. Static edges defined with `add_edge` still execute.** If a node returns `Command(goto="node_b")` AND you have `builder.add_edge("node_a", "node_c")`, both `node_b` AND `node_c` will run in the next superstep.

```python
builder.add_node("node_a", lambda state: Command(goto="node_b", update={"x": 1}))
builder.add_edge("node_a", "node_c")  # This STILL runs even when Command routes to node_b

# After node_a: BOTH node_b (from Command) AND node_c (from static edge) execute
```

This is intentional — it allows `Command` to add *additional* routing on top of the static graph structure. In practice, avoid this ambiguity: use `Command` for pure dynamic routing without any static edges on the same node, or use static edges for all routing.

#### 4.5.3 `Command` for Multi-Agent Handoffs: `goto` as a List

`Command` can route to multiple nodes simultaneously:

```python
def supervisor_node(state: MARRSState) -> Command[Literal["researcher", "writer"]]:
    """Dispatch two agents in parallel."""
    return Command(
        goto=["researcher", "writer"],  # Both activate in next superstep
        update={"status": "dispatching"}
    )
```

#### 4.5.4 `Command.PARENT`: Escaping a Subgraph

When a node is inside a subgraph and needs to route to a node in the *parent* graph, use `Command.PARENT`:

```python
def node_inside_subgraph(state: SubgraphState) -> Command:
    """
    This node is inside a subgraph but wants to route to 'final_report'
    in the parent graph, bypassing the rest of the subgraph.
    """
    return Command(
        update={"result": "done"},
        goto="final_report",
        graph=Command.PARENT   # Navigate to the parent graph's node
    )
```

`Command.PARENT` navigates to the **closest parent graph**. For deeply nested subgraphs, you may need to chain `Command.PARENT` calls or restructure the graph to avoid the need for deep parent navigation.

#### 4.5.5 When to Use `Command` vs. `add_conditional_edges`

The official documentation provides a clear rule: **"Use `Command` when you need to both update state and route to a different node. If you only need to route without updating state, use conditional edges instead."**

In practice:

| Situation | Use |
|---|---|
| Routing depends on a value computed by the node itself | `Command` |
| Routing depends on pre-existing state fields | `add_conditional_edges` |
| Multi-agent handoff where the sending node updates state | `Command` |
| Simple branching on a flag in state | `add_conditional_edges` |
| Supervisor pattern where the supervisor LLM picks the next worker | `Command` with structured output |

---

### 4.6 The `Send` API: Dynamic Parallelism (Map-Reduce)

The `Send` API enables the most powerful form of parallelism in LangGraph: **dynamic fan-out**, where the number of parallel branches is determined at runtime based on data in the state, not at graph-compile time.

#### 4.6.1 The Problem Send Solves

Static fan-out (multiple edges from one node) works when you always want the same N parallel branches. But what if you need to process a list of N items in parallel, where N is only known when the graph is running?

Example: Your planner generates 7 search queries. You want to run all 7 in parallel. You cannot add 7 static edges at compile time because you don't know N will be 7. `Send` solves this.

#### 4.6.2 The `Send` Object

```python
from langgraph.types import Send

# Send(node_name, state_for_that_invocation)
send = Send("search_node", {"query": "transformer architectures", "depth": 3})
```

`Send` takes exactly two arguments:
1. The name of the target node (must be registered in the graph)
2. A state dict that will be passed **exclusively** to that node invocation — it does not need to match the parent graph's state schema

The target node can have a completely different input schema from the parent graph. This is the key feature: each parallel invocation gets its own private state payload.

#### 4.6.3 How `Send` Works With Conditional Edges

The `Send` API is invoked by returning a list of `Send` objects from a conditional edge routing function:

```python
from langgraph.types import Send

def dispatch_searches(state: MARRSState) -> list[Send]:
    """
    Fan out to one search instance per query.
    Returns a list of Send objects — LangGraph recognizes this pattern
    and executes all of them in parallel in the next superstep.
    """
    queries = state.get("search_queries", [])
    
    return [
        Send(
            "search_one",              # Target node name
            {                          # Private state for this invocation
                "query": query,
                "topic": state["topic"],
            }
        )
        for query in queries
    ]

# Wire it in:
builder.add_conditional_edges(
    "planner",
    dispatch_searches,
    ["search_one"]  # path_map as list: tells LangGraph the only destination
)
```

When `dispatch_searches` returns `[Send("search_one", {...}), Send("search_one", {...}), ...]`, LangGraph creates one independent invocation of `search_one` for each `Send` object, all running in the same superstep.

#### 4.6.4 Full Map-Reduce Example

```python
import operator
from typing import TypedDict, Annotated
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

# ── Parent state ──────────────────────────────────────────────────────────────
class MARRSSearchState(TypedDict, total=False):
    topic: str
    search_queries: list[str]
    # Accumulates results from all parallel invocations:
    findings: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]

# ── Worker state (per-invocation, private) ────────────────────────────────────
class SingleSearchState(TypedDict):
    query: str
    topic: str

# ── Worker node ───────────────────────────────────────────────────────────────
def search_one(state: SingleSearchState) -> dict:
    """
    Processes a single search query.
    Runs in parallel with all other search_one invocations.
    Writes back to the PARENT state schema (findings, sources).
    """
    query = state["query"]
    topic = state["topic"]
    
    print(f"  [Search] Processing: {query}")
    
    # Simulate search — in production, call Tavily/SerpAPI/arXiv
    result = f"Finding for '{query}': [relevant information about {topic}]"
    source = f"https://example.com/search?q={query.replace(' ', '+')}"
    
    # Returns fields from the PARENT state schema — reducers merge them
    return {
        "findings": [result],    # operator.add accumulates across all invocations
        "sources": [source],
    }

# ── Planner node ──────────────────────────────────────────────────────────────
def planner_node(state: MARRSSearchState) -> dict:
    """Generate search queries based on the topic."""
    topic = state["topic"]
    
    # In production: use LLM with structured output
    queries = [
        f"{topic} overview",
        f"{topic} recent advances",
        f"{topic} applications",
        f"{topic} limitations",
        f"{topic} future directions",
    ]
    return {"search_queries": queries}

# ── Dispatch function ─────────────────────────────────────────────────────────
def dispatch_to_workers(state: MARRSSearchState) -> list[Send]:
    """
    Create one Send per query — N parallel workers for N queries.
    The number of workers is determined at runtime by the planner's output.
    """
    queries = state.get("search_queries", [])
    
    return [
        Send("search_one", {"query": q, "topic": state["topic"]})
        for q in queries
    ]

# ── Synthesizer node ──────────────────────────────────────────────────────────
def synthesize(state: MARRSSearchState) -> dict:
    """
    Runs AFTER all parallel searches complete (fan-in).
    Has access to all accumulated findings.
    """
    all_findings = state.get("findings", [])
    print(f"\n  [Synthesize] Combining {len(all_findings)} findings from parallel searches")
    return {}

# ── Graph ─────────────────────────────────────────────────────────────────────
builder = StateGraph(MARRSSearchState)
builder.add_node("planner", planner_node)
builder.add_node("search_one", search_one)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    dispatch_to_workers,
    ["search_one"]   # path_map as list: only destination is "search_one"
)
builder.add_edge("search_one", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()

# Run it
result = graph.invoke({"topic": "transformer architectures", "findings": [], "sources": []})
print(f"\nTotal findings collected: {len(result['findings'])}")
print(f"Total sources collected:  {len(result['sources'])}")
# Total findings collected: 5   ← One per query, all merged by operator.add
# Total sources collected:  5
```

**Superstep trace:**
```
Superstep 0: planner runs
  → Produces: search_queries = ["...overview", "...advances", "...applications", ...]

Superstep 1: dispatch_to_workers routing function runs
  → Returns: [Send("search_one", {...}), Send("search_one", {...}), × 5]
  → LangGraph creates 5 independent invocations of search_one

Superstep 2: ALL 5 search_one invocations run IN PARALLEL
  → Each returns {"findings": [one_result], "sources": [one_source]}
  → operator.add reducer merges all 5 results:
     findings = [result_1, result_2, result_3, result_4, result_5]
     sources  = [url_1, url_2, url_3, url_4, url_5]

Superstep 3: synthesize runs
  → Receives state with ALL 5 findings and sources merged
```

#### 4.6.5 Static Parallelism vs. `Send` API: When to Use Each

| Characteristic | Static Fan-Out | `Send` API |
|---|---|---|
| Number of branches | Fixed at compile time | Determined at runtime |
| Input to each branch | Same parent state | Custom per-invocation |
| Implementation | `add_edge(src, dest_1)` + `add_edge(src, dest_2)` | Routing fn returns `list[Send]` |
| Visualization | Clear in graph diagram | Shows as one conditional edge |
| Best for | Always-parallel steps (search A AND search B) | Process a dynamic list (search for each of N queries) |

---

### 4.7 Cycle Termination: The Complete Toolkit

Every cycle needs a termination mechanism. LangGraph provides several, and choosing the right one is an architectural decision.

#### 4.7.1 State-Based Termination: Explicit Counters

The most explicit and debuggable approach: track iteration count in state and check it in the routing function.

```python
class LoopState(TypedDict, total=False):
    iteration: int       # Incremented by each cycle pass
    result: str
    done: bool

def some_node(state: LoopState) -> dict:
    current = state.get("iteration", 0)
    result = do_work(state)
    return {
        "result": result,
        "iteration": current + 1,
        "done": is_good_enough(result)
    }

def should_continue(state: LoopState) -> Literal["continue", "done"]:
    MAX_ITERATIONS = 5
    
    if state.get("done"):
        return "done"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "done"
    return "continue"
```

**Advantage:** Transparent in state, checkpointable, visible in traces
**Disadvantage:** Requires the counter field in state schema

#### 4.7.2 `RemainingSteps`: LangGraph's Managed Value

LangGraph 1.0 introduced `RemainingSteps` — a **managed value** that automatically tracks how many supersteps remain before the `recursion_limit` is reached. Unlike state fields you manage yourself, `RemainingSteps` is populated by LangGraph's runtime without you writing to it.

```python
from langgraph.managed import RemainingSteps
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps  # LangGraph fills this automatically

def agent_node(state: State) -> dict:
    remaining = state["remaining_steps"]
    
    if remaining <= 3:
        # Approaching the limit — wrap up gracefully
        return {"messages": [AIMessage(
            content="I'm running low on processing steps. Here's my best answer so far..."
        )]}
    
    # Normal processing
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> Literal["agent", "__end__"]:
    last = state["messages"][-1]
    
    # Normal termination
    if not (hasattr(last, "tool_calls") and last.tool_calls):
        return "__end__"
    
    # Proactive termination as limit approaches
    if state["remaining_steps"] <= 2:
        return "__end__"
    
    return "agent"
```

`RemainingSteps` is declared in state but never updated by nodes — LangGraph manages it. It decounts by 1 each superstep. When it reaches 0, `GraphRecursionError` would normally be raised; checking it proactively lets you exit gracefully before that happens.

**When to use `RemainingSteps` vs. an explicit counter:**
- Use an explicit counter when your termination condition is domain-specific ("no more than 3 revisions")
- Use `RemainingSteps` when you want a safety net against the runtime limit being hit in complex graphs, or when you want to degrade gracefully (return partial results) instead of crashing

#### 4.7.3 Quality-Based Termination

For quality-improvement loops, terminate when quality crosses a threshold:

```python
QUALITY_THRESHOLD = 0.80

def route_after_evaluation(state: MARRSState) -> str:
    score = state.get("quality_score", 0.0)
    iterations = state.get("revision_count", 0)
    
    # Priority 1: Maximum iterations (hard limit)
    if iterations >= MAX_REVISIONS:
        return "force_exit"
    
    # Priority 2: Quality threshold met
    if score >= QUALITY_THRESHOLD:
        return "approved"
    
    # Default: continue improving
    return "revise"
```

The ordering of checks is important: the hard limit check must come before the quality check, otherwise a graph that never reaches quality threshold loops forever.

#### 4.7.4 Content-Based Termination: LLM Signals FINISH

For multi-agent systems where an LLM decides when to stop:

```python
from pydantic import BaseModel

class SupervisorDecision(BaseModel):
    next_agent: Literal["researcher", "writer", "critic", "FINISH"]
    reasoning: str

supervisor_llm = ChatOpenAI(model="gpt-4o").with_structured_output(SupervisorDecision)

def supervisor_node(state: MARRSState) -> Command:
    decision = supervisor_llm.invoke(state["messages"])
    
    if decision.next_agent == "FINISH":
        return Command(goto=END, update={"status": "complete"})
    
    return Command(goto=decision.next_agent)
```

This is the most flexible termination pattern but the least predictable — it relies entirely on the LLM's judgment about when work is done. Always pair it with a hard iteration limit as a fallback.

---

### 4.8 The Execution Order of Edges: What Runs When

Understanding precisely when routing functions, edges, and nodes execute prevents subtle bugs.

#### 4.8.1 Within a Superstep

```
1. All active nodes EXECUTE (in parallel if possible, sequentially if dependent)
   → Each node runs its function, produces a return value
   
2. Return values are BUFFERED (not yet applied to state)

3. ROUTING FUNCTIONS for all just-completed nodes are called
   → Each routing function reads the (still-old) state
   → Returns a destination node name
   
4. UPDATE PHASE: All buffered state updates are applied via reducers
   → State now reflects the combined output of all nodes in this superstep
   
5. CHECKPOINT written (if checkpointer attached)

6. Next superstep begins: activate all nodes that received routing destinations
```

**The key implication:** A routing function reads the state from *before* the current superstep's updates are applied. If a node sets `quality_score = 0.85` and your routing function checks `state["quality_score"]`, the routing function sees the *previous* quality score, not 0.85. The 0.85 will be available in the *next* superstep.

This is the standard BSP barrier synchronization from Chapter 1, applied to routing. It prevents a node's output from influencing routing in the same superstep.

**The practical consequence:** Use `Command` (not a conditional edge) when you need to route based on a value *just computed by the same node*, because `Command` carries both the state update and the routing decision together, making the value available for routing in the same step.

#### 4.8.2 Normal Edges vs. Conditional Edges: Ordering Guarantee

Normal edges always activate the destination node in the next superstep — there is no computation delay. Conditional edges also activate in the next superstep. Neither type introduces latency between steps.

---

### 4.9 Common Control Flow Patterns Catalogue

These are the patterns you will reach for repeatedly in production:

#### 4.9.1 The Retry Loop

```python
class RetryState(TypedDict, total=False):
    attempts: int
    result: str
    error: str
    success: bool

MAX_RETRIES = 3

def attempt_node(state: RetryState) -> dict:
    attempts = state.get("attempts", 0)
    try:
        result = do_risky_operation(state)
        return {"result": result, "success": True, "attempts": attempts + 1}
    except Exception as e:
        return {"error": str(e), "success": False, "attempts": attempts + 1}

def route_retry(state: RetryState) -> Literal["retry", "success", "failed"]:
    if state.get("success"):
        return "success"
    if state.get("attempts", 0) >= MAX_RETRIES:
        return "failed"
    return "retry"

builder.add_conditional_edges(
    "attempt",
    route_retry,
    {"retry": "attempt", "success": "finalize", "failed": "error_handler"}
)
```

#### 4.9.2 The Quality Gate

```python
def quality_gate(state: State) -> Literal["pass", "fail", "retry"]:
    """Route based on multiple quality criteria."""
    score = state.get("quality_score", 0.0)
    errors = state.get("validation_errors", [])
    revisions = state.get("revision_count", 0)
    
    if errors:           return "fail"      # Hard failure: validation errors
    if score >= 0.85:    return "pass"      # Pass: quality threshold met
    if revisions >= 3:   return "fail"      # Hard failure: too many retries
    return "retry"                          # Soft failure: try again
```

#### 4.9.3 The Conditional Entry Point

```python
def choose_workflow(state: State) -> Literal["fast_path", "full_pipeline", "cached"]:
    """Select the workflow variant based on request properties."""
    if state.get("use_cache") and has_cached_result(state["topic"]):
        return "cached"
    if state.get("quick_mode"):
        return "fast_path"
    return "full_pipeline"

builder.add_conditional_edges(START, choose_workflow, {
    "fast_path":     "writer",    # Skip planning and research
    "full_pipeline": "planner",   # Full MARRS pipeline
    "cached":        "finalize",  # Use cached result directly
})
```

#### 4.9.4 The Human-in-the-Loop Gate

```python
def route_after_review(state: State) -> Literal["approved", "rejected", "needs_edit"]:
    """Route based on human decision stored in state."""
    if state.get("human_approved") is True:
        return "approved"
    feedback = state.get("human_feedback", "").lower()
    if feedback == "reject":
        return "rejected"
    if feedback:  # Non-empty feedback means edit request
        return "needs_edit"
    return "rejected"  # Default: reject if no clear signal

builder.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "approved":   "finalize",
        "rejected":   END,
        "needs_edit": "writer",
    }
)
```

---

### 4.10 MARRS Checkpoint: Upgrading All Control Flow

Let's apply everything from this chapter to finalize MARRS's control flow layer — the routing functions, the `Command`-based critic, and the dynamic search fan-out.

```python
import operator
from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, tools_condition

# ─────────────────────────────────────────────────────────────────────────────
# STATE (importing from Chapter 3's upgraded schema)
# ─────────────────────────────────────────────────────────────────────────────

class MARRSState(TypedDict, total=False):
    topic: str
    plan: str
    search_queries: list[str]
    findings: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]
    draft: str
    critique: str
    quality_score: float
    revision_count: int
    human_approved: bool
    human_feedback: str
    final_report: str
    status: str
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: RemainingSteps   # Managed: LangGraph fills this


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS (all three rules applied)
# ─────────────────────────────────────────────────────────────────────────────

MAX_REVISIONS = 3

def route_after_critic(state: MARRSState) -> Literal["needs_revision", "good_enough", "max_reached"]:
    """
    Route after the Critic node evaluates the draft.
    
    Three possible outcomes:
    - needs_revision: quality insufficient, send back to Writer
    - good_enough: quality threshold met, proceed to human review
    - max_reached: hard limit hit, force human review regardless of quality
    
    Rule 1: All accesses use .get() with defaults (total=False state)
    Rule 2: max_reached is the guaranteed termination path for the loop
    Rule 3: Pure read — no side effects
    """
    score = state.get("quality_score", 0.0)
    revisions = state.get("revision_count", 0)
    
    # Check hard limit FIRST — this is the termination guarantee
    if revisions >= MAX_REVISIONS:
        return "max_reached"
    
    if score >= 0.80:
        return "good_enough"
    
    return "needs_revision"


def route_after_human_review(state: MARRSState) -> Literal["finalize", "reject", "revise"]:
    """
    Route based on the human reviewer's decision.
    
    Three outcomes:
    - finalize: approved, proceed to final formatting
    - reject: hard rejection, terminate graph
    - revise: human provided feedback, return to writer
    """
    if state.get("human_approved") is True:
        return "finalize"
    
    feedback = state.get("human_feedback", "").lower().strip()
    
    if feedback in ("reject", "rejected"):
        return "reject"
    
    if feedback:
        # Non-empty non-rejection feedback = edit instructions
        return "revise"
    
    # Default: reject if no clear signal
    return "reject"


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH FUNCTION (Send API for dynamic parallelism)
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_searches(state: MARRSState) -> list[Send]:
    """
    Fan out to one search worker per query from the planner.
    
    Returns a list of Send objects — LangGraph executes all in parallel.
    Each Send carries exactly the data its target node needs.
    """
    queries = state.get("search_queries", [])
    topic = state.get("topic", "")
    
    if not queries:
        # Fallback: if no queries, do a single broad search
        queries = [f"{topic} overview"]
    
    return [
        Send(
            "search_one",
            {"query": q, "topic": topic}
        )
        for q in queries
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CRITIC NODE USING COMMAND (routing + state update in one step)
# ─────────────────────────────────────────────────────────────────────────────

def critic_node_command(state: MARRSState) -> Command[Literal["writer", "human_review"]]:
    """
    Critic implemented with Command — combines state update and routing.
    
    This avoids the double-read problem: the score computed here
    drives routing immediately, without needing a separate routing function
    that reads from a previously persisted state field.
    """
    draft = state.get("draft", "")
    revisions = state.get("revision_count", 0)
    
    # ── Compute quality (placeholder — Chapter 9 will use a real LLM) ─────────
    score = min(0.60 + (revisions * 0.15), 0.95)
    critique_text = (
        "Needs more specific evidence and cleaner structure."
        if score < 0.80 else
        "Good quality. Proceed to human review."
    )
    new_revision_count = revisions + 1
    
    # ── Route based on JUST-COMPUTED score ────────────────────────────────────
    # This is why Command is appropriate here: we're routing on a value
    # computed in this exact function, not a pre-existing state field
    if score >= 0.80:
        destination = "human_review"
    elif new_revision_count >= MAX_REVISIONS:
        destination = "human_review"  # Force exit at max revisions
    else:
        destination = "writer"
    
    return Command(
        update={
            "critique": critique_text,
            "quality_score": score,
            "revision_count": new_revision_count,
        },
        goto=destination
    )


# ─────────────────────────────────────────────────────────────────────────────
# REMAINING STEPS: GRACEFUL DEGRADATION
# ─────────────────────────────────────────────────────────────────────────────

def should_continue_research(state: MARRSState) -> Literal["search_one", "writer", "emergency_exit"]:
    """
    Route after researcher's internal ReAct loop or after dispatching searches.
    Includes emergency exit when approaching recursion limit.
    """
    # Emergency exit if we're running very low on steps
    remaining = state.get("remaining_steps")
    if remaining is not None and remaining <= 3:
        return "emergency_exit"
    
    # Normal routing: did we accumulate enough findings?
    findings = state.get("findings", [])
    if len(findings) >= 5:
        return "writer"
    
    return "writer"  # Simplified for now — Chapter 9 adds the ReAct loop


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH WORKER NODE (receives per-invocation state from Send)
# ─────────────────────────────────────────────────────────────────────────────

class SingleSearchState(TypedDict):
    """Private state for individual search workers."""
    query: str
    topic: str

def search_one_node(state: SingleSearchState) -> dict:
    """
    Single search worker — runs in parallel with siblings from the same Send dispatch.
    Writes to parent state schema fields (findings, sources).
    """
    query = state["query"]
    topic = state["topic"]
    
    print(f"  [Search] {query}")
    
    # Placeholder — Chapter 9 will call real search APIs
    finding = f"Finding: '{query}' relates to {topic} in the following ways..."
    source = f"https://search.example.com/?q={query.replace(' ', '+')}"
    
    return {
        "findings": [finding],  # operator.add in parent state merges all
        "sources": [source],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE PLACEHOLDER NODES (to be replaced in Chapter 9)
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: MARRSState) -> dict:
    print(f"  [Planner] Planning research for: {state.get('topic', '?')}")
    topic = state.get("topic", "")
    return {
        "plan": f"Research plan for {topic}",
        "search_queries": [
            f"{topic} overview",
            f"{topic} recent advances",
            f"{topic} applications",
            f"limitations of {topic}",
        ]
    }

def writer_node(state: MARRSState) -> dict:
    revision = state.get("revision_count", 0)
    critique = state.get("critique", "")
    print(f"  [Writer] {'Revising' if revision > 0 else 'Writing'} draft...")
    return {
        "draft": f"[Draft #{revision + 1} on {state.get('topic', '?')}]\n\nBased on {len(state.get('findings', []))} findings..."
    }

def human_review_node(state: MARRSState) -> dict:
    print(f"  [HumanReview] Simulating approval (score: {state.get('quality_score', 0):.2f})")
    return {
        "human_approved": True,
        "human_feedback": "Approved",
        "status": "approved"
    }

def finalize_node(state: MARRSState) -> dict:
    print("  [Finalize] Preparing final report...")
    return {
        "final_report": f"# {state.get('topic', 'Report')}\n\n{state.get('draft', '')}",
        "status": "complete"
    }

def emergency_exit_node(state: MARRSState) -> dict:
    print("  [EmergencyExit] Recursion limit approaching — returning best effort result.")
    return {
        "final_report": f"Partial result (processing limit reached): {state.get('draft', '[No draft]')}",
        "status": "partial"
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_marrs_v3() -> StateGraph:
    """
    MARRS with upgraded control flow:
    - Command-based critic (routing + state update together)
    - Send-based dynamic search parallelism
    - RemainingSteps emergency exit
    - Typed routing functions (Literal annotations)
    - Fully tested routing logic
    """
    builder = StateGraph(MARRSState)
    
    # Register all nodes
    builder.add_node("planner", planner_node)
    builder.add_node("search_one", search_one_node)    # Parallel worker
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node_command)    # Uses Command
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("emergency_exit", emergency_exit_node)
    
    # Entry: always start at planner
    builder.add_edge(START, "planner")
    
    # Dynamic fan-out: planner → N parallel search workers via Send
    builder.add_conditional_edges(
        "planner",
        dispatch_searches,
        ["search_one"]       # path_map as list: search_one is the only destination
    )
    
    # Fan-in: all search workers converge on writer
    builder.add_edge("search_one", "writer")
    
    # writer → critic (critic uses Command, no add_conditional_edges needed)
    builder.add_edge("writer", "critic")
    
    # critic routes via Command internally to either "writer" or "human_review"
    # No add_conditional_edges call needed here — Command handles it
    
    # human_review → conditional routing
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "finalize":  "finalize",
            "reject":    END,
            "revise":    "writer",
        }
    )
    
    # Emergency exit and finalize both terminate
    builder.add_edge("finalize", END)
    builder.add_edge("emergency_exit", END)
    
    return builder


# Compile
marrs_v3 = build_marrs_v3().compile()

# Visualize
print(marrs_v3.get_graph().draw_ascii())
```

#### 4.10.1 Testing the Control Flow

```python
import pytest

# ── Unit tests for routing functions ──────────────────────────────────────────

def test_route_after_critic_good_quality():
    state = {"quality_score": 0.85, "revision_count": 1}
    assert route_after_critic(state) == "good_enough"

def test_route_after_critic_low_quality():
    state = {"quality_score": 0.55, "revision_count": 1}
    assert route_after_critic(state) == "needs_revision"

def test_route_after_critic_max_revisions():
    # Hard limit takes priority over quality
    state = {"quality_score": 0.45, "revision_count": 3}
    assert route_after_critic(state) == "max_reached"

def test_route_after_critic_none_safe():
    # total=False state — no fields set yet
    assert route_after_critic({}) == "needs_revision"

def test_route_after_human_review_approved():
    state = {"human_approved": True, "human_feedback": "Approved"}
    assert route_after_human_review(state) == "finalize"

def test_route_after_human_review_rejected():
    state = {"human_approved": False, "human_feedback": "reject"}
    assert route_after_human_review(state) == "reject"

def test_route_after_human_review_edit_requested():
    state = {"human_approved": False, "human_feedback": "Please add more citations"}
    assert route_after_human_review(state) == "revise"

# ── Unit test for Send dispatch ───────────────────────────────────────────────

def test_dispatch_searches():
    from langgraph.types import Send
    state = {
        "topic": "transformers",
        "search_queries": ["query A", "query B", "query C"]
    }
    result = dispatch_searches(state)
    assert len(result) == 3
    assert all(isinstance(s, Send) for s in result)
    assert result[0].node == "search_one"
    assert result[0].arg["query"] == "query A"
    assert result[0].arg["topic"] == "transformers"

def test_dispatch_searches_empty_fallback():
    # If no queries, should produce at least one fallback search
    state = {"topic": "transformers", "search_queries": []}
    result = dispatch_searches(state)
    assert len(result) >= 1

# ── Integration test ──────────────────────────────────────────────────────────

def test_marrs_completes():
    result = marrs_v3.invoke({
        "topic": "test topic",
        "findings": [],
        "sources": [],
    })
    assert result.get("status") in ("complete", "partial")
    assert "final_report" in result
```

---

### 4.11 Chapter Summary

**Conditional edges** (`add_conditional_edges`) are the primary branching mechanism. They call a routing function after a node, mapping the return value to a destination node. The `path_map` (or `Literal` return annotations) documents possible destinations and enables correct visualization. The `then` parameter handles shared convergence points. Always use `.get()` with defaults in routing functions, always include a hard termination path, and always keep routing functions free of side effects.

**The ReAct loop** (`agent → tools → agent → ...`) is the universal single-agent cycle. `tools_condition` is the prebuilt routing function. `ToolNode` is the prebuilt tool executor with error handling. The loop terminates when the LLM produces a response with no `tool_calls`.

**`Command`** combines state updates and routing in a single node return value. Use it when routing depends on a value computed by the node itself. `Command` only adds dynamic edges — static edges still execute. The `Literal` return type annotation on the node function is required for visualization. `Command.PARENT` navigates from a subgraph to the parent graph.

**The `Send` API** enables dynamic fan-out: a routing function returns a list of `Send(node, state)` objects, each creating an independent parallel invocation with its own state payload. This is the map-reduce pattern. Parallel results merge via reducers in the parent state.

**`RemainingSteps`** is a managed value that tracks supersteps remaining before the recursion limit. Use it for graceful degradation: route to a fallback node when the count is low, instead of raising `GraphRecursionError`.

**Cycle termination** has four tools: explicit iteration counters in state, `RemainingSteps`, quality thresholds, and LLM-signaled completion. Always pair LLM-driven termination with a hard counter limit as a safety net.

---

### Further Reading

- **Official Graph API docs**: `docs.langchain.com/oss/python/langgraph/graph-api` — covers all edge types, `Command`, `Send`, `RemainingSteps`, and conditional entry points
- **Official `Command` tutorial**: `docs.langchain.com/oss/python/langgraph/how-to/command` — end-to-end example of `Command` in multi-agent handoffs
- **Official map-reduce how-to**: `langchain-ai.github.io/langgraph/how-tos/map-reduce/` — the canonical Send API example
- **LangChain blog: "Command: A new tool for building multi-agent architectures"**: The announcement post explaining the motivation for `Command`
- **`langgraph.prebuilt.tools_condition` source**: The 10-line implementation of the ReAct routing function — shows exactly how tool call detection works

---

*End of Chapter 4. Chapter 5: Persistence and Checkpointers.*
