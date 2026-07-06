# LangGraph: From Zero to Production-Grade Multi-Agent Systems
### A Comprehensive, Hands-On Curriculum — Beginner to Advanced

> **Style note:** This curriculum is written in the spirit of Sebastian Raschka's educational work — dense theory, clear mental models, lots of working code, and honest discussion of *why* things work the way they do. Nothing is hand-waved away.

---

## Preface

**What is this?**
A complete, chapter-by-chapter curriculum for mastering LangGraph — the low-level orchestration framework for building stateful, long-running AI agents. By the end, you will have built a full **Multi-Agent Research & Report Writing System** (MARRS) from scratch, incrementally adding features as each new concept is introduced.

**Who is this for?**
- Intermediate Python developers who know the basics of LLM APIs (OpenAI, Anthropic, etc.)
- ML practitioners who want to build production-grade agentic pipelines
- Engineers who have used LangChain but want to go deeper into stateful workflow orchestration

**What you'll need:**
- Python 3.10+
- `pip install langgraph langchain-openai langchain-anthropic langsmith`
- An OpenAI or Anthropic API key
- A working understanding of Python type hints and `async`/`await`

**The capstone project — MARRS:**
> Given a research topic, MARRS coordinates a **Planner agent**, a **Researcher agent** (with web search tools), a **Critic agent** (quality reviewer), and a **Writer agent** to produce a polished, structured report. It supports human-in-the-loop approval gates, persists state across sessions, and streams outputs in real time.

---

## Table of Contents

1. [Chapter 1 — The Mental Model: Why Graphs?](#chapter-1)
2. [Chapter 2 — LangGraph Core Architecture: State, Nodes, and Edges](#chapter-2)
3. [Chapter 3 — Reducers, Annotations, and Advanced State Design](#chapter-3)
4. [Chapter 4 — Control Flow: Conditional Edges, Routing, and Cycles](#chapter-4)
5. [Chapter 5 — Persistence and Checkpointers](#chapter-5)
6. [Chapter 6 — Memory: Short-Term and Long-Term](#chapter-6)
7. [Chapter 7 — Human-in-the-Loop: Interrupts, Breakpoints, and Time Travel](#chapter-7)
8. [Chapter 8 — Streaming: Token-Level to Node-Level](#chapter-8)
9. [Chapter 9 — Tool Integration and the ReAct Pattern](#chapter-9)
10. [Chapter 10 — Multi-Agent Systems: Patterns and Architectures](#chapter-10)
11. [Chapter 11 — Subgraphs and Hierarchical Teams](#chapter-11)
12. [Chapter 12 — The Functional API: `@entrypoint` and `@task`](#chapter-12)
13. [Chapter 13 — Production: Observability, LangSmith, and Deployment](#chapter-13)
14. [Chapter 14 — MARRS: Full Capstone Build](#chapter-14)
15. [Chapter 15 — Advanced Topics: Parallelism, Caching, and Beyond](#chapter-15)

---

<a name="chapter-1"></a>
## Chapter 1 — The Mental Model: Why Graphs?

### 1.1 The Problem With Linear Chains

Before LangGraph existed, the dominant paradigm for LLM applications was **chains** — a sequence of steps, each taking the output of the previous step as input. LangChain's LCEL (LangChain Expression Language) is the canonical example:

```python
chain = prompt | llm | output_parser
result = chain.invoke({"question": "What is ML?"})
```

This is elegant for simple, **Directed Acyclic Graph (DAG)** workflows. But real-world agents need something chains cannot express:

1. **Loops** — an agent must be able to try a tool, get a result, evaluate it, and try again
2. **Conditional branching** — "if the model is confident, finish; if not, search again"
3. **Parallel execution** — run two independent searches simultaneously, merge results
4. **Durable execution** — pause, wait for human input, resume days later
5. **Stateful memory** — remember what happened in step 3 when you're at step 10

These requirements demand **cyclic graphs**, not DAGs. This is precisely why LangGraph was built.

### 1.2 Pregel and Apache Beam: The Academic Foundation

LangGraph's execution model is explicitly inspired by **Google's Pregel** system (2010), which introduced the concept of "think like a vertex" for distributed graph computation, and **Apache Beam**'s unified batch/stream model.

The key insight from Pregel: computation proceeds in discrete **"super-steps."** In each super-step:
1. Every active vertex (node) receives messages from its incoming edges
2. Each vertex runs its function and produces output messages
3. Messages are sent along outgoing edges to activate the next set of vertices
4. Vertices with no incoming messages become inactive

LangGraph maps this directly:
- **Vertices → Nodes** (Python functions)
- **Messages → State updates** (Python dicts or TypedDicts)
- **Super-steps → Graph execution rounds**

This means nodes that are not dependent on each other can run **in parallel within the same super-step**, which is a key performance feature we'll explore in Chapter 15.

### 1.3 The State Machine Perspective

Another useful lens is the **finite state machine (FSM)**. In a traditional FSM:
- You have a set of states
- A set of transitions between states (triggered by conditions)
- An alphabet of possible inputs

An LLM agent maps naturally to an FSM with one extra dimension: the state is not just a label ("idle", "searching", "done") but an **arbitrary data structure** — the full context of the agent's work so far.

In LangGraph:
- **States** → TypedDict or Pydantic schemas (the full agent memory snapshot)
- **Transitions** → Edges (deterministic) or Conditional Edges (LLM-driven)
- **States visited** → The graph execution trace

This is not just an analogy. The official LangGraph documentation describes multi-agent architectures in terms of state machines, and the framework's design choices flow directly from this model.

### 1.4 LangGraph's Positioning in the Ecosystem

```
High-level abstractions (less control)
│
│   LangChain AgentExecutor (prebuilt, opinionated)
│   ↕ built on top of
│   LangGraph (low-level, explicit control)
│   ↕ integrates with
│   LangChain (tools, models, prompts — optional)
│   ↕ observability via
│   LangSmith (tracing, evaluation, monitoring)
│
Low-level primitives (more control)
```

Key fact: **LangGraph reached v1.0 (generally available) in October 2025**, after more than a year of powering agents at Uber, LinkedIn, Klarna, and JP Morgan. It is MIT-licensed and free to use. You do not need LangChain to use LangGraph — any LLM API will work.

### 1.5 Chapter Summary

- Linear chains cannot express cycles, parallelism, or durable execution
- LangGraph models workflows as cyclic graphs, inspired by Pregel
- The execution model is a sequence of super-steps
- Nodes are functions; edges are routing logic; state is the shared memory
- LangGraph is a low-level framework: it gives you full control and no magic

---

<a name="chapter-2"></a>
## Chapter 2 — LangGraph Core Architecture: State, Nodes, and Edges

### 2.1 The Three Pillars

Every LangGraph application is built from exactly three primitives:

| Primitive | What It Is | In Python |
|-----------|-----------|-----------|
| **State** | Shared data structure flowing through the graph | `TypedDict` or Pydantic `BaseModel` |
| **Nodes** | Functions that read state and emit updates | Python callables |
| **Edges** | Rules for which node runs next | Functions or direct `add_edge()` calls |

Let's build from first principles.

### 2.2 Defining State

The state is the "working memory" of your graph. Every node receives it as input and returns a (partial) update to it.

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

# The simplest possible state: a single field
class SimpleState(TypedDict):
    value: int
```

**Design principle:** State should contain everything a node needs to do its job, and everything downstream nodes will need. Think of it as the "baton" passed between runners in a relay race.

### 2.3 Defining Nodes

A node is any Python function that:
1. Accepts the current state as its first argument
2. Returns a dictionary of key-value updates to the state

```python
def add_one(state: SimpleState) -> dict:
    """Increment the value by 1."""
    return {"value": state["value"] + 1}

def multiply_by_two(state: SimpleState) -> dict:
    """Double the value."""
    return {"value": state["value"] * 2}
```

**Critical insight:** Nodes do NOT return the full state. They return only the **fields they want to update**. Unmentioned fields are left unchanged. This is the "partial update" model.

### 2.4 Defining Edges and Compiling the Graph

```python
# Step 1: Create the graph, parameterized by your state schema
builder = StateGraph(SimpleState)

# Step 2: Register nodes
builder.add_node("add_one", add_one)
builder.add_node("multiply_by_two", multiply_by_two)

# Step 3: Define the flow with edges
builder.add_edge(START, "add_one")         # Always start here
builder.add_edge("add_one", "multiply_by_two")
builder.add_edge("multiply_by_two", END)   # Always end here

# Step 4: Compile — validates structure and creates executable graph
graph = builder.compile()
```

**What does `.compile()` do?**
- Validates that there are no orphaned nodes (nodes with no edges in/out)
- Validates that START and END are reachable
- Optionally attaches a checkpointer (persistence layer)
- Returns a `CompiledStateGraph` — the executable object

### 2.5 Invoking the Graph

```python
# Invoke: runs synchronously, returns final state
result = graph.invoke({"value": 5})
# add_one: 5 → 6
# multiply_by_two: 6 → 12
print(result)  # {"value": 12}
```

### 2.6 A More Realistic Example: The Chat Agent Skeleton

Let's build something closer to a real use case — the skeleton of a conversational agent:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# State: a list of messages
# add_messages is a reducer — we explain reducers fully in Chapter 3
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Model
llm = ChatOpenAI(model="gpt-4o-mini")

# Node: call the LLM
def call_model(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages reducer appends, not replaces

# Build the graph
builder = StateGraph(ChatState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
graph = builder.compile()

# Run it
result = graph.invoke({
    "messages": [HumanMessage(content="What is a transformer architecture?")]
})
print(result["messages"][-1].content)
```

### 2.7 Visualizing the Graph

LangGraph can render its own structure as ASCII or Mermaid diagrams. This is enormously useful for debugging and documentation:

```python
# ASCII visualization
print(graph.get_graph().draw_ascii())

# Mermaid (render at mermaid.live)
print(graph.get_graph().draw_mermaid())
```

Example output for the chat agent:
```
        +-----------+      
        | __start__ |      
        +-----------+      
               *           
               *           
               *           
           +-------+       
           | model |       
           +-------+       
               *           
               *           
               *           
         +---------+       
         | __end__ |       
         +---------+       
```

### 2.8 Chapter Summary

- State is a TypedDict or Pydantic model: the graph's shared memory
- Nodes are Python functions: they read state and return partial updates
- Edges define flow: from START through nodes to END
- `.compile()` validates and locks the graph structure
- `.invoke()` runs the graph synchronously and returns the final state

---

<a name="chapter-3"></a>
## Chapter 3 — Reducers, Annotations, and Advanced State Design

### 3.1 The Default Reducer: Overwrite

When a node returns `{"key": new_value}`, the default behavior is to **overwrite** the existing value of that key. This is fine for most scalar fields (strings, ints, bools).

```python
class State(TypedDict):
    count: int      # overwrite default
    status: str     # overwrite default

def update(state: State) -> dict:
    return {"count": state["count"] + 1}
    # The existing count is REPLACED by the new value
```

But what about lists? If you just overwrite a list, you lose history:

```python
# WRONG for message history
def add_message(state: State) -> dict:
    return {"messages": [new_message]}  # This REPLACES the list! History lost.
```

This is where **reducers** come in.

### 3.2 Reducers: Defining How State Updates Are Applied

A reducer is a function with signature `(existing_value, new_value) -> merged_value`. You attach it to a state field using Python's `Annotated` type hint:

```python
from typing import Annotated
import operator

class State(TypedDict):
    # operator.add for lists: extends instead of replacing
    messages: Annotated[list, operator.add]
    
    # Custom reducer: always keep the maximum value
    max_score: Annotated[float, lambda old, new: max(old or 0.0, new)]
    
    # Standard overwrite (no Annotated needed, but explicit for clarity):
    step: int
```

When a node returns `{"messages": [new_msg]}`, the reducer is called:
```python
merged = operator.add(current_messages, [new_msg])
# = current_messages + [new_msg]
# The history is preserved!
```

### 3.3 The `add_messages` Reducer

For conversational agents, LangGraph provides a purpose-built reducer: `add_messages` from `langgraph.graph.message`. It does more than just append:

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

`add_messages` handles:
1. **Appending** new messages to the history
2. **Deduplication by ID** — if a message with the same `id` already exists, it's updated in place rather than duplicated. This is critical for tool call/result pairs.
3. **Type coercion** — accepts both `BaseMessage` objects and raw dicts, normalizing them

```python
# Example: add_messages in action
from langgraph.graph.message import add_messages

existing = [HumanMessage(content="Hello", id="msg-1")]
new = [AIMessage(content="Hi there!", id="msg-2")]

result = add_messages(existing, new)
# [HumanMessage(id="msg-1"), AIMessage(id="msg-2")]

# Now update an existing message (same ID)
updated = [AIMessage(content="Hi there! How are you?", id="msg-2")]
result = add_messages(result, updated)
# [HumanMessage(id="msg-1"), AIMessage(id="msg-2")]  ← content updated, no duplicate
```

### 3.4 Using Pydantic for State Validation

For production systems where you want runtime type validation, use Pydantic instead of TypedDict:

```python
from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph.message import add_messages

class ResearchState(BaseModel):
    """State for the MARRS research agent."""
    
    # The conversation + tool call history
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    
    # The research topic
    topic: str = ""
    
    # Accumulated research findings
    findings: list[str] = Field(default_factory=list)
    
    # Draft report (overwrite, not append)
    draft: str = ""
    
    # Number of revision cycles
    revision_count: int = 0
    
    # Flag to control the loop
    is_complete: bool = False
    
    # Quality score from the critic
    quality_score: float = 0.0

    class Config:
        arbitrary_types_allowed = True  # Required for BaseMessage subclasses
```

**TypedDict vs Pydantic — when to use which:**

| Situation | Use |
|-----------|-----|
| Simple prototyping | `TypedDict` |
| Need default values | `TypedDict` with `total=False` or Pydantic |
| Need runtime validation | Pydantic |
| Need `.model_dump()`, `.model_validate()` | Pydantic |
| Maximum performance (no validation overhead) | `TypedDict` |

### 3.5 Input and Output Schemas

By default, a LangGraph graph uses the same schema for input, output, and internal state. For production systems, you often want to expose a cleaner API: accept only the user's message as input, return only the final answer as output, while keeping the full rich state internally.

```python
from typing import TypedDict
from langgraph.graph import StateGraph

# Full internal state (rich, complex)
class InternalState(TypedDict):
    messages: list
    scratchpad: str
    tool_results: list
    retry_count: int

# Minimal input schema (what the caller provides)
class InputSchema(TypedDict):
    question: str

# Clean output schema (what the caller receives)
class OutputSchema(TypedDict):
    answer: str
    sources: list[str]

builder = StateGraph(
    InternalState,
    input=InputSchema,
    output=OutputSchema
)
```

### 3.6 MARRS Checkpoint: State Design

Here's the state we'll build toward for MARRS:

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class MARRSState(TypedDict):
    """
    Central state for the Multi-Agent Research & Report Writing System.
    
    Fields:
        messages     - Full message history (append-only via add_messages)
        topic        - The research question/topic
        plan         - The structured research plan from the Planner
        findings     - Accumulated search results (append-only)
        draft        - Current draft of the report (overwrite)
        critique     - Latest critique from the Critic agent (overwrite)
        revision_num - How many times we've revised (overwrite)
        final_report - Finalized output (overwrite)
    """
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    plan: str
    findings: Annotated[list[str], operator.add]
    draft: str
    critique: str
    revision_num: int
    final_report: str
```

### 3.7 Chapter Summary

- Default reducer = overwrite: new value replaces old
- Annotated[type, reducer_fn] lets you define custom merge behavior
- `add_messages` is a production-grade reducer for message history with deduplication
- Pydantic gives runtime validation; TypedDict is simpler but un-validated
- Input/output schemas let you expose a clean API while keeping rich internal state

---

<a name="chapter-4"></a>
## Chapter 4 — Control Flow: Conditional Edges, Routing, and Cycles

### 4.1 Normal Edges

Normal edges are unconditional transitions. Node A **always** goes to Node B:

```python
builder.add_edge("node_a", "node_b")
```

Simple, but not powerful enough for agents. Agents need to make decisions.

### 4.2 Conditional Edges

A conditional edge is a Python function that looks at the current state and returns the name of the next node to visit:

```python
def route_after_model(state: AgentState) -> str:
    """
    After the LLM responds, decide what to do next:
    - If the LLM called a tool, go to 'tools' node
    - Otherwise, we're done — go to END
    """
    last_message = state["messages"][-1]
    
    # AIMessage has tool_calls attribute if the model wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "__end__"

builder.add_conditional_edges(
    "model",            # Source node
    route_after_model,  # Routing function
    {
        "tools": "tools",   # String output "tools" → go to "tools" node
        "__end__": END,     # String output "__end__" → go to END
    }
)
```

**The third argument (path_map)** is optional but highly recommended. It:
1. Documents the possible destinations explicitly
2. Allows LangGraph to validate that all possible return values are mapped
3. Lets you use different internal names and node names

Without path_map:
```python
# This also works, but the router must return exact node names
builder.add_conditional_edges("model", route_after_model)
# route_after_model must return "tools" or END directly
```

### 4.3 Building the ReAct Loop

The ReAct (Reason + Act) pattern is the bedrock of most agent systems. The agent loops between:
1. **Reason** — the LLM thinks about what to do (possibly calling tools)
2. **Act** — tools are executed
3. Back to **Reason** with tool results in context

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage

# Define a tool
@tool
def search_web(query: str) -> str:
    """Search the web for information about the query."""
    # In reality, call Tavily or SerpAPI here
    return f"Search results for '{query}': [simulated results]"

tools = [search_web]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node 1: The "brain" — calls the LLM
def agent_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# The routing function
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return "__end__"

# Build the graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))  # ToolNode handles execution + error handling

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")  # Always return to agent after tools

graph = builder.compile()
```

This creates the classic loop:
```
START → agent → (if tool calls) → tools → agent → ...
                ↓ (if no tool calls)
               END
```

### 4.4 The `Command` Object: Combining State Update and Routing

Normally, nodes update state by returning a dict, and routing is handled by conditional edges. But sometimes you want a node to both update state AND decide where to go next. The `Command` object enables this:

```python
from langgraph.types import Command
from typing import Literal

def critic_node(state: MARRSState) -> Command[Literal["writer", "finalize"]]:
    """
    Evaluate the draft. If quality is high enough, finalize.
    Otherwise, send back to the writer for revision.
    """
    score = evaluate_draft(state["draft"])
    
    if score > 0.85 or state["revision_num"] >= 3:
        return Command(
            update={"critique": "Approved", "quality_score": score},
            goto="finalize"
        )
    else:
        critique = generate_critique(state["draft"])
        return Command(
            update={
                "critique": critique,
                "quality_score": score,
                "revision_num": state["revision_num"] + 1
            },
            goto="writer"
        )
```

**When to use `Command` vs. `add_conditional_edges`:**

| Scenario | Use |
|----------|-----|
| Routing logic depends on the output you just computed | `Command` |
| Routing logic depends on pre-existing state fields | `add_conditional_edges` |
| You want to avoid redundant computation (compute result, then inspect same result for routing) | `Command` |
| Routing logic is shared by multiple nodes | `add_conditional_edges` with shared function |

### 4.5 `Command` for Multi-Agent Handoffs

In multi-agent systems (covered in depth in Chapter 10), `Command` is the primary mechanism for one agent to hand off work to another:

```python
def researcher_node(state: MARRSState) -> Command[Literal["critic"]]:
    # Do research...
    findings = do_research(state["topic"])
    
    # Hand off to critic with updated state
    return Command(
        update={"findings": [findings]},
        goto="critic"
    )
```

### 4.6 MARRS Checkpoint: The Revision Loop

Here's the core control flow logic for MARRS:

```python
def route_after_critic(state: MARRSState) -> str:
    """
    After the Critic evaluates the draft:
    - Low quality OR first revision → send back to Writer
    - High quality OR max revisions reached → finalize
    """
    MAX_REVISIONS = 3
    
    if state["quality_score"] >= 0.85:
        return "finalize"
    elif state["revision_num"] >= MAX_REVISIONS:
        print(f"Max revisions ({MAX_REVISIONS}) reached. Finalizing.")
        return "finalize"
    else:
        return "writer"

builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {"finalize": "finalize", "writer": "writer"}
)
builder.add_edge("writer", "critic")  # Writer always goes back to Critic
```

### 4.7 Chapter Summary

- Normal edges create unconditional, always-taken transitions
- Conditional edges route dynamically by calling a Python function on the current state
- `add_conditional_edges` with a path_map is the standard approach
- The `Command` object lets a node control both state updates and routing in one return value
- The ReAct loop (agent ↔ tools) is the fundamental cycle in most agents

---

<a name="chapter-5"></a>
## Chapter 5 — Persistence and Checkpointers

### 5.1 Why Persistence Matters

Without persistence, every graph invocation starts fresh. This means:
- No conversation history across sessions
- No ability to pause and resume
- No human-in-the-loop (impossible to resume after human input)
- No fault tolerance (failure = start over)

LangGraph's persistence system solves all of these by writing a **checkpoint** after every super-step.

### 5.2 The Checkpoint System: How It Works

A checkpoint is a complete snapshot of the graph state at a point in time. It contains:
- The current values of all state fields
- Which node runs next
- The execution configuration (thread ID, etc.)
- Metadata about the step number and timing

The core mechanism:

```
User sends message (thread_id="conversation-42")
    ↓
LangGraph looks up latest checkpoint for thread "conversation-42"
    ↓
Loads state from checkpoint (or starts fresh if first time)
    ↓
Executes graph nodes (one super-step at a time)
    ↓ (after every super-step)
Writes new checkpoint to storage backend
    ↓
Returns final state to caller
```

### 5.3 The `thread_id`: Your Conversational Identity

Every persistent conversation requires a unique `thread_id` in the config:

```python
config = {"configurable": {"thread_id": "user-alice-session-1"}}

# First message — starts fresh or loads existing
result1 = graph.invoke(
    {"messages": [HumanMessage("What is quantum computing?")]},
    config=config
)

# Second message — automatically has full history from the checkpoint
result2 = graph.invoke(
    {"messages": [HumanMessage("Explain superposition in more detail.")]},
    config=config
)
# The graph "remembers" the first message because the checkpoint stored it!
```

### 5.4 Checkpointer Backends

**In-Memory (development only):**
```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```
Data lives in RAM. Disappears on process restart. Good for testing.

**SQLite (single-process production or development):**
```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory SQLite (temporary)
memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
checkpointer = SqliteSaver(memory_conn)

# On-disk SQLite (persistent)
disk_conn = sqlite3.connect("./marrs_checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(disk_conn)

graph = builder.compile(checkpointer=checkpointer)
```

**PostgreSQL (production, multi-process):**
```bash
pip install langgraph-checkpoint-postgres
```
```python
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

conn_string = "postgresql://user:password@localhost:5432/marrs"
with psycopg.connect(conn_string) as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # Creates tables on first run
    graph = builder.compile(checkpointer=checkpointer)
```

### 5.5 Inspecting Checkpoints

You can inspect the full checkpoint history for any thread:

```python
config = {"configurable": {"thread_id": "my-thread"}}

# Get current state
current_state = graph.get_state(config)
print(current_state.values)    # The state dict
print(current_state.next)      # Which nodes run next (empty if finished)
print(current_state.metadata)  # Step number, timing

# Get full history (most recent first)
for snapshot in graph.get_state_history(config):
    print(f"Step {snapshot.metadata['step']}: next={snapshot.next}")
    print(f"  State: {snapshot.values}")
```

### 5.6 State Modification: Correcting Agent Behavior

You can modify the current state of a paused graph — extremely powerful for debugging and human-in-the-loop workflows:

```python
# The agent produced a bad draft. Let's correct it before continuing.
graph.update_state(
    config,
    {"draft": "A better draft that the human wrote manually."},
    as_node="writer"   # Pretend this update came from the 'writer' node
)

# Now continue execution from the updated state
result = graph.invoke(None, config)
```

### 5.7 State Bloat: A Production Warning

A critical production concern: LangGraph writes a new checkpoint at **every super-step**. If your state contains large binary data (PDFs, images, large embeddings), you will rapidly accumulate enormous amounts of data.

**Anti-pattern:**
```python
class BadState(TypedDict):
    pdf_content: bytes  # 5MB PDF — stored EVERY STEP!
```

**Correct pattern:**
```python
class GoodState(TypedDict):
    pdf_url: str        # Just store a reference to external storage
    pdf_summary: str    # Store the extracted/summarized content
```

Store large artifacts in S3, GCS, or a database. Keep only references and derived metadata in LangGraph state.

### 5.8 MARRS Checkpoint: Adding Persistence

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Create persistent checkpointer
conn = sqlite3.connect("marrs.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Compile with checkpointer
marrs_graph = builder.compile(checkpointer=checkpointer)

# Run with a user-specific thread ID
user_config = {"configurable": {"thread_id": "user-bob-report-1"}}
result = marrs_graph.invoke(
    {"topic": "The impact of transformer architectures on NLP"},
    config=user_config
)

# Bob can resume this later — state is fully persisted
resumed = marrs_graph.invoke(
    {"messages": [HumanMessage("Actually, focus more on BERT specifically.")]},
    config=user_config
)
```

### 5.9 Chapter Summary

- Persistence = checkpointing: a full state snapshot is written after every super-step
- `thread_id` is the key that isolates different conversations/sessions
- `InMemorySaver` → dev only; `SqliteSaver` → local production; `PostgresSaver` → distributed production
- `get_state()` and `get_state_history()` let you inspect any checkpoint
- `update_state()` lets you modify agent state before resuming
- Keep state lean: store references to large objects, not the objects themselves

---

<a name="chapter-6"></a>
## Chapter 6 — Memory: Short-Term and Long-Term

### 6.1 Two Types of Memory

LangGraph distinguishes between two fundamentally different memory systems:

| Type | Scope | Storage | Use Case |
|------|-------|---------|----------|
| **Short-term** | One conversation thread | Checkpointer | Message history, current task context |
| **Long-term** | Across all threads/sessions | Store (key-value) | User preferences, past decisions, learned facts |

Short-term memory is what we've been building with the checkpointer. Long-term memory requires the `Store` interface.

### 6.2 Short-Term Memory: Managing Message History

In real applications, message history grows unboundedly. A conversation with 100 turns might have 50,000 tokens of history — exceeding model context windows and inflating costs.

**Trimming messages:**
```python
from langchain_core.messages import trim_messages

def call_model_with_trimming(state: AgentState) -> dict:
    # Keep only the last 10 messages (adjust as needed)
    trimmed = trim_messages(
        state["messages"],
        max_tokens=4096,
        strategy="last",
        token_counter=llm,  # Use the actual model's tokenizer
        include_system=True,
        allow_partial=False,
        start_on="human"
    )
    response = llm.invoke(trimmed)
    return {"messages": [response]}
```

**Summarizing old messages:**
```python
def summarize_conversation(state: AgentState) -> dict:
    """Summarize old messages to compress history."""
    if len(state["messages"]) > 20:
        # Get the old messages
        old_messages = state["messages"][:-10]  # Everything except last 10
        recent_messages = state["messages"][-10:]  # Keep last 10 verbatim
        
        # Summarize the old ones
        summary_prompt = f"""Summarize the following conversation history 
        concisely, preserving key facts and decisions:
        
        {[m.content for m in old_messages]}
        """
        summary = llm.invoke(summary_prompt).content
        
        # Replace old messages with summary
        from langchain_core.messages import SystemMessage
        summary_msg = SystemMessage(content=f"Previous conversation summary: {summary}")
        return {"messages": [summary_msg] + recent_messages}
    
    return {}  # No update needed if history is short
```

### 6.3 Long-Term Memory: The `Store` Interface

The `Store` is a key-value store that persists data **across threads**. This is where your agent can remember things about a specific user, store learned facts, or cache expensive computations.

```python
from langgraph.store.memory import InMemoryStore

# Create the store
store = InMemoryStore()

# Compile the graph with BOTH a checkpointer and a store
graph = builder.compile(
    checkpointer=InMemorySaver(),
    store=store
)
```

Nodes access the store via a special parameter:

```python
from langgraph.store.base import BaseStore

def personalized_agent(state: AgentState, store: BaseStore) -> dict:
    """Agent that remembers user preferences across sessions."""
    
    user_id = "user-alice"
    
    # Retrieve stored memories for this user
    # Namespace = (user_id, category) for organization
    memories = store.search(namespace=(user_id, "preferences"))
    
    preference_context = ""
    if memories:
        preference_context = "\n".join([m.value["text"] for m in memories])
    
    # Build prompt with memory context
    system_prompt = f"""You are a helpful assistant.
    
    Known user preferences:
    {preference_context}
    
    Use these preferences to personalize your response."""
    
    from langchain_core.messages import SystemMessage
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    
    # Store any new preferences learned from this conversation
    if "prefers" in response.content.lower():
        store.put(
            namespace=(user_id, "preferences"),
            key=f"pref_{len(memories)}",
            value={"text": f"User mentioned: {response.content[:200]}"}
        )
    
    return {"messages": [response]}
```

### 6.4 Semantic Memory Search (2025 Feature)

LangGraph's `InMemoryStore` (and production backends) support **semantic search** — finding memories by meaning, not just exact key lookup:

```python
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

# Store with semantic search enabled
store = InMemoryStore(
    index={
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "dims": 1536,
    }
)

# Store a memory
store.put(
    namespace=("user-alice", "facts"),
    key="fact-1",
    value={"text": "Alice is a machine learning engineer who prefers Python over R"}
)

# Search by meaning — no exact key needed
results = store.search(
    namespace=("user-alice", "facts"),
    query="programming language preference",  # Semantic search
    limit=3
)
# Returns: the "Python over R" fact, even though the query didn't mention it
```

### 6.5 Chapter Summary

- Short-term memory = message history, scoped to a thread, stored in the checkpointer
- Trim or summarize messages to prevent context window overflow
- Long-term memory = the `Store` interface, a key-value DB persisting across threads
- `store.put(namespace, key, value)` to write; `store.search(namespace)` to read
- Semantic search allows meaning-based retrieval in production stores

---

<a name="chapter-7"></a>
## Chapter 7 — Human-in-the-Loop: Interrupts, Breakpoints, and Time Travel

### 7.1 Why Human-in-the-Loop?

Fully autonomous agents make mistakes. For high-stakes actions — sending emails, modifying databases, publishing content, approving transactions — you need a human to review before the agent proceeds.

LangGraph's human-in-the-loop (HITL) system is built on three mechanisms:
1. **Interrupts** — dynamic pauses anywhere in node code
2. **Breakpoints** — static pauses before/after specific nodes
3. **Time Travel** — rewind to any prior checkpoint and re-execute

All three require a **checkpointer** (Chapter 5) to be attached to the graph.

### 7.2 Static Breakpoints: Always Pause Here

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["send_email"],   # Pause BEFORE this node runs
    interrupt_after=["draft_writer"],  # Pause AFTER this node runs
)
```

When execution reaches "send_email", it pauses. The caller receives a partial result:

```python
# Start execution — runs until interrupt
config = {"configurable": {"thread_id": "task-1"}}
for event in graph.stream({"task": "Email Alice about the meeting"}, config):
    print(event)

# Execution is now paused. Inspect the state:
state = graph.get_state(config)
print("Pending action:", state.values["pending_email"])
print("Next node:", state.next)  # ("send_email",)

# Human approves — resume by invoking with None input
for event in graph.stream(None, config):
    print(event)
```

### 7.3 Dynamic Interrupts: The `interrupt()` Function

Static breakpoints always pause. For more nuanced control, use the `interrupt()` function directly in your node code — pause only when certain conditions are met:

```python
from langgraph.types import interrupt

def send_email_node(state: AgentState) -> dict:
    email_draft = state["pending_email"]
    
    # Dynamically decide whether to pause based on context
    if email_draft.get("is_external"):  # Only pause for external emails
        # Pause and surface data to the human
        human_decision = interrupt({
            "question": "Please review this outgoing email before sending.",
            "email": email_draft,
            "recipient": email_draft["to"],
            "action": "approve_or_reject"
        })
        
        if human_decision.get("approved") is not True:
            return {"status": "email_rejected", "rejection_reason": human_decision.get("reason")}
    
    # If approved (or internal email), proceed with sending
    result = actually_send_email(email_draft)
    return {"status": "email_sent", "message_id": result["id"]}
```

**Resuming after `interrupt()`:**
```python
from langgraph.types import Command

# Human reviews and approves
resume_payload = {"approved": True}

for event in graph.stream(
    Command(resume=resume_payload),  # Pass the human's decision
    config
):
    print(event)
```

The `interrupt()` call inside the node receives the `resume_payload` as its return value and execution continues from exactly that point.

### 7.4 Input Validation with `interrupt()`

You can loop on `interrupt()` to validate user input before continuing:

```python
from langgraph.types import interrupt

def collect_budget_node(state: AgentState) -> dict:
    """Collect and validate a budget from the user."""
    prompt = "Please enter the maximum budget for this project (in USD):"
    
    while True:
        user_input = interrupt({"question": prompt})
        
        try:
            budget = float(str(user_input).replace("$", "").replace(",", ""))
            if budget > 0:
                return {"budget": budget}
            else:
                prompt = "Budget must be positive. Please enter a valid amount:"
        except (ValueError, TypeError):
            prompt = f"'{user_input}' is not a valid number. Please enter a numeric budget:"
```

### 7.5 Time Travel: Checkpoints as a Superpower

Because LangGraph checkpoints every super-step, you have a complete history of every state your agent has been in. This enables:

**1. Inspecting past states:**
```python
# Get all checkpoints for a thread
history = list(graph.get_state_history(config))
for i, snapshot in enumerate(history):
    print(f"Checkpoint {i}: step={snapshot.metadata.get('step')}, next={snapshot.next}")
```

**2. Forking from a past checkpoint (Time Travel):**
```python
# Say checkpoint at step 3 had a bug. Let's fork from step 2 instead.
old_config = {
    "configurable": {
        "thread_id": "task-1",
        "checkpoint_id": history[2].config["configurable"]["checkpoint_id"]
    }
}

# This creates a NEW branch from checkpoint 2 — original history is preserved
for event in graph.stream(
    {"topic": "Let's try a different approach"},  # New input at the fork point
    old_config
):
    print(event)
```

**3. Replaying for debugging:**
```python
# Re-run a specific checkpoint to reproduce a bug
replay_config = {
    "configurable": {
        "thread_id": "task-1",
        "checkpoint_id": "the_bad_checkpoint_id"
    }
}
result = graph.invoke(None, replay_config)
```

### 7.6 MARRS Checkpoint: Human Approval Gate

```python
from langgraph.types import interrupt

def human_approval_node(state: MARRSState) -> dict:
    """
    Pause for human review of the final draft before publication.
    Allow the human to approve, reject, or request specific edits.
    """
    decision = interrupt({
        "prompt": "Please review the following report draft:",
        "draft": state["draft"],
        "quality_score": state["quality_score"],
        "topic": state["topic"],
        "instructions": "Reply with: 'approve', 'reject', or a string of edit instructions"
    })
    
    if decision == "approve":
        return {"final_report": state["draft"], "status": "approved"}
    elif decision == "reject":
        return {"final_report": "", "status": "rejected"}
    else:
        # Human provided edit instructions — treat as new critique
        return {
            "critique": f"Human feedback: {decision}",
            "status": "needs_revision"
        }
```

### 7.7 Chapter Summary

- HITL requires a checkpointer (state is saved across the pause)
- `interrupt_before/interrupt_after` at compile time = static breakpoints
- `interrupt()` inside a node = dynamic, conditional pauses
- `Command(resume=payload)` sends human input back to the waiting node
- Time travel = invoking a specific `checkpoint_id` to replay or fork history

---

<a name="chapter-8"></a>
## Chapter 8 — Streaming: Token-Level to Node-Level

### 8.1 Why Streaming Matters

For a 30-second research task with no streaming, the user sees nothing for 30 seconds, then a wall of text. This is a terrible UX. Streaming lets users see:
- Each token as the LLM generates it
- Which agent is currently active
- Intermediate results as they accumulate
- Tool calls being made in real time

### 8.2 Stream Modes

LangGraph's `.stream()` method accepts a `stream_mode` parameter:

| Mode | What You Get | Use Case |
|------|-------------|----------|
| `"values"` | Full state after each super-step | Inspecting state at each stage |
| `"updates"` | Only the delta (what changed) per node | Efficient progress updates |
| `"messages"` | Token-by-token from LLM nodes | Real-time chat UX |
| `"debug"` | Verbose debug events | Development/debugging |

```python
# "values" mode: full state snapshot after each super-step
for state in graph.stream(input, config, stream_mode="values"):
    print("Current state:", state)

# "updates" mode: (node_name, update_dict) after each node
for node_name, update in graph.stream(input, config, stream_mode="updates"):
    print(f"Node '{node_name}' updated: {update}")

# "messages" mode: (message_chunk, metadata) tokens as they're generated
for chunk, metadata in graph.stream(input, config, stream_mode="messages"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if metadata.get("langgraph_node") == "tools":
        print(f"\n[Tool called by: {metadata['langgraph_node']}]")
```

### 8.3 Multiple Stream Modes Simultaneously

```python
# Get both message chunks AND state updates
for event in graph.stream(
    input,
    config,
    stream_mode=["messages", "updates"]
):
    # event is a tuple: (stream_mode, data)
    mode, data = event
    if mode == "messages":
        chunk, metadata = data
        print(chunk.content, end="", flush=True)
    elif mode == "updates":
        node_name, updates = data
        if "findings" in updates:
            print(f"\n[New finding added by {node_name}]")
```

### 8.4 Streaming from LLM Nodes: The Detail

For token-level streaming to work, your LLM node needs to use a streaming-capable model. LangChain's chat models all support streaming natively when used in LangGraph:

```python
from langchain_openai import ChatOpenAI

# This model streams tokens automatically in LangGraph
llm = ChatOpenAI(model="gpt-4o", streaming=True)

def writer_node(state: MARRSState) -> dict:
    response = llm.invoke(state["messages"])  # Tokens stream as they're generated
    return {"draft": response.content}
```

When you use `stream_mode="messages"`, LangGraph intercepts the token stream from the LLM and relays it to the caller — no extra code needed in your nodes.

### 8.5 Async Streaming for Production

For web applications serving multiple concurrent users, use `astream()`:

```python
import asyncio

async def run_marrs(topic: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    async for event in graph.astream(
        {"topic": topic},
        config,
        stream_mode="messages"
    ):
        if isinstance(event, tuple):
            chunk, metadata = event
            if chunk.content:
                yield chunk.content  # Yield tokens to the caller (e.g., FastAPI SSE)

# FastAPI endpoint example
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/research")
async def research_endpoint(topic: str, session_id: str):
    return StreamingResponse(
        run_marrs(topic, session_id),
        media_type="text/event-stream"
    )
```

### 8.6 Chapter Summary

- `stream_mode="values"` = full state per super-step
- `stream_mode="updates"` = delta per node (more efficient)
- `stream_mode="messages"` = token-by-token from LLM nodes
- Multiple modes can be combined
- Use `.astream()` for concurrent production deployments

---

<a name="chapter-9"></a>
## Chapter 9 — Tool Integration and the ReAct Pattern

### 9.1 What Is a Tool?

In LangGraph's world, a **tool** is any Python function that:
1. An LLM can decide to call (via function/tool calling APIs)
2. Takes structured inputs (defined by its signature and docstring)
3. Returns a result (typically a string or JSON-serializable object)

```python
from langchain_core.tools import tool

@tool
def calculate_compound_interest(
    principal: float,
    annual_rate: float,
    years: int,
    compounds_per_year: int = 12
) -> float:
    """
    Calculate compound interest.
    
    Args:
        principal: Initial investment amount in USD
        annual_rate: Annual interest rate as a decimal (e.g., 0.05 for 5%)
        years: Number of years to compound
        compounds_per_year: How many times per year to compound (default: 12 for monthly)
    
    Returns:
        Final amount after compounding
    """
    return principal * (1 + annual_rate / compounds_per_year) ** (compounds_per_year * years)
```

The docstring is crucial — it's what the LLM uses to decide whether to call this tool and what arguments to pass.

### 9.2 Binding Tools to a Model

To let an LLM call tools, you "bind" them to the model. This adds the tool schemas to the model's request:

```python
from langchain_openai import ChatOpenAI

# Tools available to the agent
tools = [search_web, calculate_compound_interest, read_file]

# Bind tools to the model
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# When invoked, the model can now return AIMessage with tool_calls
response = llm_with_tools.invoke("What is 1000 invested at 7% for 20 years?")
print(response.tool_calls)
# [{"name": "calculate_compound_interest", "args": {"principal": 1000, "annual_rate": 0.07, "years": 20}, "id": "..."}]
```

### 9.3 `ToolNode`: The Standard Tool Executor

`ToolNode` is a prebuilt LangGraph node that handles tool execution. It:
1. Reads tool calls from the last `AIMessage` in state
2. Executes each tool call in parallel (if multiple)
3. Wraps results in `ToolMessage` objects with matching IDs
4. Returns `{"messages": [tool_messages]}`

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

builder.add_node("tools", tool_node)
```

### 9.4 Error Handling in Tools

`ToolNode` has built-in error handling. By default, tool errors are caught and returned as error `ToolMessage`s so the LLM can recover:

```python
tool_node = ToolNode(tools, handle_tool_errors=True)
# On tool failure: ToolMessage(content="Error: ...", status="error")
# The LLM receives this and can decide how to recover
```

For custom error handling:
```python
def custom_error_handler(error: Exception) -> str:
    """Convert tool errors to user-friendly messages."""
    if isinstance(error, TimeoutError):
        return "The search timed out. Try a more specific query."
    elif isinstance(error, ValueError):
        return f"Invalid input: {str(error)}. Please check your parameters."
    else:
        return f"An unexpected error occurred: {str(error)[:200]}"

tool_node = ToolNode(tools, handle_tool_errors=custom_error_handler)
```

### 9.5 Building the Full ReAct Agent

Let's build a complete, production-ready ReAct agent for MARRS's Researcher component:

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage

# --- Tools ---

@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for academic papers related to the query.
    Use for finding scientific research and technical papers.
    
    Args:
        query: Search terms related to the research topic
        max_results: Maximum number of papers to return (1-10)
    
    Returns:
        Formatted list of paper titles, authors, abstracts, and URLs
    """
    # In production, call the arxiv API
    return f"[arXiv results for '{query}': paper1, paper2, paper3...]"

@tool
def search_web(query: str) -> str:
    """
    Search the web for current information, news, and general knowledge.
    Use for finding recent events, statistics, and non-academic sources.
    
    Args:
        query: Search query string
    
    Returns:
        Formatted search results with titles, snippets, and URLs
    """
    # In production, call Tavily, SerpAPI, etc.
    return f"[Web results for '{query}': result1, result2, result3...]"

@tool
def extract_key_facts(text: str, topic: str) -> str:
    """
    Extract key facts, statistics, and claims from a piece of text.
    Use to distill important information from long sources.
    
    Args:
        text: The source text to analyze
        topic: The research topic to focus on
    
    Returns:
        Bullet-point list of key facts related to the topic
    """
    # In production, use an LLM for extraction
    return f"[Key facts extracted from text about '{topic}']"

tools = [search_arxiv, search_web, extract_key_facts]

# --- State ---

class ResearcherState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    findings: Annotated[list[str], operator.add]

# --- Model ---

RESEARCHER_SYSTEM_PROMPT = """You are an expert research assistant. Your job is to 
gather comprehensive, accurate information about a given topic using the tools 
available to you.

Strategy:
1. Start with a broad web search to understand the landscape
2. Search arXiv for relevant academic work
3. Extract key facts from the most relevant results
4. Compile at least 5-7 distinct, well-sourced findings
5. When you have sufficient information, stop and report your findings

Always cite your sources. Be thorough but efficient."""

llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

# --- Nodes ---

def researcher_agent(state: ResearcherState) -> dict:
    messages = [SystemMessage(content=RESEARCHER_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: ResearcherState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"

def extract_findings(state: ResearcherState) -> dict:
    """Post-process: extract findings from the final message."""
    last_message = state["messages"][-1]
    # In production, parse structured output from the final response
    findings = [last_message.content[:500]]  # Simplified
    return {"findings": findings}

# --- Graph ---

researcher_builder = StateGraph(ResearcherState)
researcher_builder.add_node("agent", researcher_agent)
researcher_builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
researcher_builder.add_node("extract", extract_findings)

researcher_builder.add_edge(START, "agent")
researcher_builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "__end__": "extract"}
)
researcher_builder.add_edge("tools", "agent")
researcher_builder.add_edge("extract", END)

researcher_graph = researcher_builder.compile()
```

### 9.6 Forcing Tool Use and Structured Outputs

Sometimes you want the LLM to always return structured data, not freeform text. Use `tool_choice` and Pydantic schemas:

```python
from pydantic import BaseModel, Field

class ResearchPlan(BaseModel):
    """A structured research plan."""
    main_questions: list[str] = Field(description="Key questions to answer")
    search_queries: list[str] = Field(description="Specific search queries to run")
    expected_sources: list[str] = Field(description="Types of sources to prioritize")

# Force the model to always return a ResearchPlan
planner_llm = ChatOpenAI(model="gpt-4o").with_structured_output(ResearchPlan)

def planner_node(state: MARRSState) -> dict:
    plan: ResearchPlan = planner_llm.invoke(
        f"Create a research plan for: {state['topic']}"
    )
    # plan is a Pydantic object — serialize for storage
    return {"plan": plan.model_dump_json()}
```

### 9.7 Chapter Summary

- Tools are Python functions decorated with `@tool`; docstrings define the LLM's understanding
- `.bind_tools(tools)` attaches tool schemas to a model
- `ToolNode` is the standard tool executor — handles parallel execution and error handling
- The ReAct loop (agent ↔ tools) is the standard architecture
- `with_structured_output(PydanticModel)` forces JSON output matching a schema

---

<a name="chapter-10"></a>
## Chapter 10 — Multi-Agent Systems: Patterns and Architectures

### 10.1 When Do You Need Multiple Agents?

A single agent with many tools works well for simple tasks. But as complexity grows, you hit problems:

- **Too many tools**: An agent choosing from 30+ tools makes poor decisions
- **Cognitive overload**: A single context window for a multi-hour task fills up
- **Specialization**: Different subtasks require different system prompts, tools, and models
- **Parallelism**: Independent subtasks should run concurrently

The solution: break the problem into **specialized agents**, each focused on a narrow domain, coordinated by a workflow.

### 10.2 Architecture 1: Network (Peer-to-Peer)

Agents communicate as peers, any agent can talk to any other agent. No central coordinator.

```
Agent A ↔ Agent B ↔ Agent C
         ↕
      Agent D
```

**When to use:** Creative collaboration, brainstorming, emergent workflows

**LangGraph implementation:** Each agent is a node; edges or `Command` objects route between them based on conversation content.

### 10.3 Architecture 2: Supervisor

A central supervisor agent manages worker agents. Workers report back after completing tasks; the supervisor decides what to do next.

```
         Supervisor
        /     |     \
  Worker1  Worker2  Worker3
```

**When to use:** Well-defined task decomposition, when you need consistent coordination, compliance workflows

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

SUPERVISOR_PROMPT = """You are a research supervisor coordinating a team of specialists.
Your team:
- 'researcher': Gathers information and finds sources
- 'writer': Drafts and edits written content
- 'critic': Reviews quality and identifies gaps

Assign tasks to team members based on what's needed next.
When all tasks are complete, respond with 'FINISH'."""

supervisor_llm = ChatOpenAI(model="gpt-4o")

class SupervisorState(MessagesState):
    next: str  # Which agent runs next

def make_supervisor(members: list[str]):
    def supervisor_node(state: SupervisorState) -> Command:
        options = members + ["FINISH"]
        
        # LLM decides who works next
        response = supervisor_llm.invoke([
            {"role": "system", "content": SUPERVISOR_PROMPT},
            *state["messages"],
            {"role": "user", "content": f"Given the above, who should act next? Options: {options}"}
        ])
        
        next_worker = response.content.strip()
        
        if next_worker == "FINISH":
            return Command(goto=END)
        return Command(goto=next_worker, update={"next": next_worker})
    
    return supervisor_node
```

### 10.4 Architecture 3: Hierarchical Teams

Multi-level supervision: a top-level supervisor manages team supervisors, who each manage specialized workers.

```
        Top Supervisor
       /               \
Research Supervisor   Writing Supervisor
  /        \              /         \
  Searcher  Analyzer   Drafter   Editor
```

**When to use:** Complex, multi-phase projects with distinct domains; large-scale automation

```python
# Each team is a compiled subgraph
# The top supervisor treats them as opaque units

# Research team
research_graph = build_research_team()  # Returns compiled graph
writing_graph = build_writing_team()

# Top-level supervisor
top_builder = StateGraph(TopLevelState)
top_builder.add_node("research_team", research_graph)
top_builder.add_node("writing_team", writing_graph)
top_builder.add_node("top_supervisor", make_supervisor(["research_team", "writing_team"]))
# ... edges
top_graph = top_builder.compile()
```

### 10.5 Architecture 4: Plan-and-Execute

A planner creates a structured plan upfront; an executor (or multiple executors) carries it out step by step.

```python
class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]          # Steps to execute
    past_steps: list[tuple]  # (step, result) pairs
    response: str            # Final answer

def planner_node(state: PlanExecuteState) -> dict:
    """Create a step-by-step plan to answer the question."""
    plan_template = ChatPromptTemplate.from_messages([
        ("system", "Create a step-by-step plan to answer this question."),
        ("human", "{question}")
    ])
    planner = plan_template | ChatOpenAI(model="gpt-4o") | PydanticOutputParser(pydantic_object=Plan)
    plan = planner.invoke({"question": state["messages"][0].content})
    return {"plan": plan.steps}

def executor_node(state: PlanExecuteState) -> dict:
    """Execute the next step in the plan."""
    current_step = state["plan"][0]
    result = executor_agent.invoke({"task": current_step})
    return {
        "past_steps": [(current_step, result)],
        "plan": state["plan"][1:]  # Remove completed step
    }

def replanner_node(state: PlanExecuteState) -> Command:
    """After executing a step, re-evaluate the plan."""
    if not state["plan"] or is_objective_met(state):
        return Command(goto="respond", update={"response": synthesize(state)})
    else:
        # Optionally update the remaining plan based on new information
        updated_plan = replan(state)
        return Command(goto="executor", update={"plan": updated_plan})
```

### 10.6 Choosing an Architecture

| Architecture | Pros | Cons | Best For |
|-------------|------|------|----------|
| Network | Flexible, emergent | Hard to debug, non-deterministic | Creative tasks |
| Supervisor | Clear control, debuggable | Single point of failure, bottleneck | Structured workflows |
| Hierarchical | Scalable, modular | Complex to build | Large systems |
| Plan-Execute | Transparent, auditable | Brittle to plan errors | Well-defined tasks |

### 10.7 Chapter Summary

- Multi-agent systems solve single-agent limitations: too many tools, context length, specialization, parallelism
- Four core patterns: Network, Supervisor, Hierarchical, Plan-Execute
- `Command(goto=agent_name)` is the primary mechanism for agent-to-agent handoff
- Choose architecture based on task structure and control requirements

---

<a name="chapter-11"></a>
## Chapter 11 — Subgraphs and Hierarchical Teams

### 11.1 What Is a Subgraph?

A subgraph is a compiled LangGraph graph used as a **node inside another graph**. This is the mechanism for hierarchical decomposition:

```python
# Build and compile a specialized subgraph
research_team_graph = build_research_team().compile()

# Use it as a node in the parent graph
parent_builder = StateGraph(ParentState)
parent_builder.add_node("research", research_team_graph)
```

When the parent graph visits the "research" node, it invokes the entire subgraph, waits for it to finish, and receives its output state.

### 11.2 State Communication: Parent ↔ Subgraph

The key challenge: parent and subgraph may have different state schemas. LangGraph handles this automatically **if they share at least one key** (usually `messages`).

**Option A: Shared state key (simplest)**
```python
# Parent state
class ParentState(TypedDict):
    messages: Annotated[list, add_messages]  # Shared key
    topic: str

# Subgraph state — also has 'messages'
class SubgraphState(TypedDict):
    messages: Annotated[list, add_messages]  # Same key name
    internal_scratchpad: str                # Private to subgraph
```

The subgraph receives the parent's `messages` and its updates to `messages` flow back up.

**Option B: Different schemas with transform functions**
```python
def research_input_transform(parent_state: ParentState) -> SubgraphState:
    """Convert parent state to subgraph input format."""
    return {
        "messages": [HumanMessage(content=f"Research this topic: {parent_state['topic']}")],
        "internal_scratchpad": ""
    }

def research_output_transform(subgraph_output: SubgraphState) -> dict:
    """Convert subgraph output back to parent state format."""
    return {
        "findings": extract_findings(subgraph_output["messages"]),
        "messages": subgraph_output["messages"][-1:]  # Only the final message
    }
```

### 11.3 Building MARRS with Subgraphs

```python
# ---- Research Subgraph ----
def build_research_subgraph():
    builder = StateGraph(ResearchSubgraphState)
    builder.add_node("search_agent", search_react_agent)
    builder.add_node("tools", ToolNode([search_arxiv, search_web]))
    builder.add_node("synthesize", synthesize_findings_node)
    
    builder.add_edge(START, "search_agent")
    builder.add_conditional_edges("search_agent", should_continue_search)
    builder.add_edge("tools", "search_agent")
    builder.add_edge("search_agent", "synthesize")
    builder.add_edge("synthesize", END)
    
    return builder.compile()

# ---- Writing Subgraph ----
def build_writing_subgraph():
    builder = StateGraph(WritingSubgraphState)
    builder.add_node("drafter", draft_report_node)
    builder.add_node("formatter", format_report_node)
    
    builder.add_edge(START, "drafter")
    builder.add_edge("drafter", "formatter")
    builder.add_edge("formatter", END)
    
    return builder.compile()

# ---- Top-Level MARRS Graph ----
def build_marrs():
    research_sg = build_research_subgraph()
    writing_sg = build_writing_subgraph()
    
    builder = StateGraph(MARRSState)
    builder.add_node("planner", planner_node)
    builder.add_node("research", research_sg)  # Subgraph as node!
    builder.add_node("critic", critic_node)
    builder.add_node("writer", writing_sg)    # Another subgraph!
    builder.add_node("human_review", human_approval_node)
    builder.add_node("finalize", finalize_node)
    
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "research")
    builder.add_edge("research", "writer")
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges("critic", route_after_critic)
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)
    
    return builder

marrs_graph = build_marrs().compile(checkpointer=checkpointer)
```

### 11.4 `compile(name=...)` for Observability

When a subgraph is compiled with a name, it appears clearly in LangSmith traces:

```python
research_sg = build_research_subgraph().compile(name="ResearchTeam")
writing_sg = build_writing_subgraph().compile(name="WritingTeam")
```

### 11.5 Chapter Summary

- Subgraphs = compiled graphs used as nodes in parent graphs
- Shared state keys enable automatic parent ↔ subgraph communication
- Different state schemas require explicit transform functions
- `compile(name=...)` improves observability in traces
- Hierarchical composition is the primary mechanism for scalable, modular multi-agent systems

---

<a name="chapter-12"></a>
## Chapter 12 — The Functional API: `@entrypoint` and `@task`

### 12.1 Graph API vs. Functional API

LangGraph provides two complementary ways to build workflows:

| | Graph API | Functional API |
|---|-----------|----------------|
| Style | Explicit nodes/edges | Python functions with decorators |
| Visualization | Full graph diagram | Not available (dynamic) |
| Time Travel | Full support | Limited (no per-task checkpoints) |
| Best For | Complex multi-agent systems, when visualization matters | Simple workflows, tasks with natural Python control flow |
| State Management | TypedDict/Pydantic state schema | `previous` parameter for history |

Both APIs use the same underlying runtime and can be mixed freely.

### 12.2 `@task`: Asynchronous Work Units

A `@task` is like an async function with built-in durable execution. When called, it returns a future. The runtime can schedule, retry, and checkpoint tasks:

```python
from langgraph.func import task
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

@task
def generate_outline(topic: str) -> str:
    """Generate a research outline for the given topic."""
    response = llm.invoke(f"Create a detailed research outline for: {topic}")
    return response.content

@task
def write_section(outline: str, section_title: str) -> str:
    """Write a specific section of the report."""
    response = llm.invoke(
        f"Based on this outline:\n{outline}\n\nWrite the section: {section_title}"
    )
    return response.content

@task
def search_for_section(section_title: str) -> str:
    """Search for relevant information for a section."""
    # Call actual search API
    return f"Search results for {section_title}..."
```

### 12.3 `@entrypoint`: The Workflow Container

An `@entrypoint` wraps your workflow logic and provides:
- A callable interface (invoke/stream)
- Checkpointing (if a checkpointer is attached)
- Human-in-the-loop support via `interrupt()`

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
import asyncio

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def research_workflow(topic: str) -> dict:
    """
    Full research workflow using the Functional API.
    Generates an outline, searches for each section, writes each section.
    """
    # Step 1: Generate outline
    outline_future = generate_outline(topic)
    outline = outline_future.result()
    
    # Step 2: Parse sections from outline (simplified)
    sections = ["Introduction", "Background", "Findings", "Conclusion"]
    
    # Step 3: Search for each section IN PARALLEL
    search_futures = [search_for_section(section) for section in sections]
    search_results = [f.result() for f in search_futures]
    
    # Step 4: Write each section (can also be parallelized)
    section_futures = [
        write_section(outline, section) 
        for section in sections
    ]
    written_sections = [f.result() for f in section_futures]
    
    # Step 5: Human review gate
    draft = "\n\n".join(written_sections)
    approved = interrupt({
        "message": "Please review the draft report",
        "draft": draft
    })
    
    if not approved:
        return {"status": "rejected", "draft": draft}
    
    return {"status": "complete", "report": draft}

# Use it
config = {"configurable": {"thread_id": "workflow-1"}}
result = research_workflow.invoke("transformer architectures", config=config)
```

### 12.4 Parallelism in the Functional API

When you call multiple `@task`s without `.result()` immediately, they run concurrently:

```python
@entrypoint(checkpointer=checkpointer)
def parallel_research(topics: list[str]) -> list[str]:
    """Research multiple topics in parallel."""
    
    # Launch all searches concurrently
    futures = [search_for_section(topic) for topic in topics]
    
    # Now collect results (blocking until all complete)
    results = [f.result() for f in futures]
    
    return results
```

This is the Functional API's killer feature for parallelism — it's much simpler than managing parallel nodes in the Graph API.

### 12.5 Mixing Graph and Functional APIs

```python
# A task can call a compiled graph
from langgraph.func import task

@task
def run_research_subgraph(topic: str) -> dict:
    """Use the full research graph as a task."""
    return researcher_graph.invoke({"topic": topic, "messages": []})

@entrypoint(checkpointer=checkpointer)
def top_level_workflow(query: str) -> str:
    # Use the graph-based subgraph from within the Functional API
    research_result = run_research_subgraph(query).result()
    draft = write_report(research_result["findings"]).result()
    return draft
```

### 12.6 Chapter Summary

- The Functional API uses `@entrypoint` + `@task` decorators instead of explicit nodes/edges
- `@task` creates durable, potentially parallel work units
- `@entrypoint` wraps workflow logic, provides checkpointing and HITL support
- Parallelism: launch multiple tasks before calling `.result()` to execute concurrently
- Graph and Functional APIs can be mixed in the same project

---

<a name="chapter-13"></a>
## Chapter 13 — Production: Observability, LangSmith, and Deployment

### 13.1 Why Observability Is Non-Negotiable for Agents

Unlike deterministic software, agent behavior is probabilistic. The LLM makes different choices on every run. Without observability, you cannot:
- Debug why an agent made a specific decision
- Measure agent performance across thousands of runs
- Catch regressions when you update prompts or models
- Understand cost and latency breakdowns

### 13.2 LangSmith Integration

LangSmith is the observability platform built specifically for LLM applications. It integrates with LangGraph automatically — no code changes needed beyond setting environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="your-langsmith-api-key"
export LANGCHAIN_PROJECT="MARRS-Production"
```

With these set, every graph invocation automatically creates a detailed trace in LangSmith showing:
- Every node that executed and in what order
- Input and output state at each node
- Every LLM call: prompt, response, token counts, latency
- Every tool call: function name, arguments, result
- Total cost and duration per run

### 13.3 Custom Metadata and Tags

```python
# Add metadata to runs for filtering in LangSmith
config = {
    "configurable": {"thread_id": "user-alice-1"},
    "metadata": {
        "user_id": "alice",
        "environment": "production",
        "report_type": "technical"
    },
    "tags": ["marrs", "v2", "gpt-4o"]
}

result = marrs_graph.invoke(input, config=config)
```

### 13.4 Structured Logging in Nodes

```python
import logging
from langchain_core.callbacks import get_callback_manager

logger = logging.getLogger(__name__)

def researcher_node(state: MARRSState) -> dict:
    logger.info(f"Starting research for topic: {state['topic']}")
    logger.info(f"Current revision: {state.get('revision_num', 0)}")
    
    result = do_research(state)
    
    logger.info(f"Research complete. Found {len(result['findings'])} findings.")
    return result
```

### 13.5 Node-Level Caching (LangGraph 2025 Feature)

LangGraph's June 2025 release introduced **node-level caching** — cache expensive node results to avoid re-computation:

```python
from langgraph.cache import InMemoryCache

cache = InMemoryCache()

graph = builder.compile(
    checkpointer=checkpointer,
    cache=cache  # Attach cache at compile time
)

# In your node, mark it as cacheable:
def expensive_search_node(state: MARRSState) -> dict:
    # Results for the same input will be returned from cache on subsequent calls
    return do_expensive_search(state["topic"])

builder.add_node("search", expensive_search_node, cache=True)
```

This is particularly valuable in development: re-running the same graph with the same inputs returns cached results instantly, saving API costs and time.

### 13.6 Deploying with LangGraph Platform

For production deployment, LangChain offers the **LangGraph Platform** — a purpose-built runtime for stateful, long-running agents:

```
Local LangGraph (dev) → LangGraph Platform (production)
       ↓                          ↓
  InMemorySaver              PostgresSaver (managed)
  Single process             Auto-scaling workers
  No monitoring              LangSmith integration
  Manual restarts            Automatic fault recovery
```

**LangGraph Cloud configuration (`langgraph.json`):**
```json
{
  "dependencies": ["./"],
  "graphs": {
    "marrs": "./marrs/graph.py:marrs_graph"
  },
  "env": ".env"
}
```

Deploy:
```bash
pip install "langgraph-cli[inmem]"
langgraph dev          # Local dev server
langgraph deploy       # Deploy to LangGraph Platform
```

### 13.7 Self-Hosted Deployment with FastAPI

For teams that need full control:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

app = FastAPI(title="MARRS API")

class ResearchRequest(BaseModel):
    topic: str
    thread_id: str

@app.post("/research")
async def start_research(request: ResearchRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    async def event_stream():
        async for event in marrs_graph.astream(
            {"topic": request.topic},
            config,
            stream_mode="updates"
        ):
            node_name, updates = event
            yield f"data: {json.dumps({'node': node_name, 'updates': str(updates)})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = marrs_graph.get_state(config)
    return {
        "status": "complete" if not state.next else "running",
        "next_node": list(state.next),
        "revision_count": state.values.get("revision_num", 0)
    }
```

### 13.8 Chapter Summary

- Set `LANGCHAIN_TRACING_V2=true` for automatic LangSmith tracing with zero code changes
- Add metadata and tags to runs for filtering and analysis
- Node-level caching (2025) reduces dev costs and speeds up iteration
- LangGraph Platform provides managed deployment with auto-scaling
- Self-hosted: FastAPI + async streaming is the standard pattern

---

<a name="chapter-14"></a>
## Chapter 14 — MARRS: Full Capstone Build

This chapter assembles everything we've learned into the complete Multi-Agent Research & Report Writing System. Every piece of code here builds directly on concepts from previous chapters.

### 14.1 Architecture Overview

```
User Query
    ↓
[Planner Agent]          — creates a structured research plan (Chapter 9)
    ↓
[Research Agent]         — ReAct loop with search tools (Chapter 9)
    ↓
[Writer Agent]           — drafts the report (Chapter 9)
    ↓
[Critic Agent]           — scores and critiques (Chapter 4)
    ↓          ↑
   (loop back to Writer if quality < threshold) (Chapter 4)
    ↓
[Human Review]           — interrupt() for human approval (Chapter 7)
    ↓
[Finalizer]              — formats and returns final report
    ↓
Final Report
```

### 14.2 Complete State Definition

```python
import operator
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class MARRSState(TypedDict):
    # Core conversation history
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Research inputs
    topic: str
    
    # Planner output
    research_plan: str
    search_queries: list[str]
    
    # Research findings (accumulate across multiple search rounds)
    findings: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]
    
    # Writing/revision cycle
    draft: str
    critique: str
    revision_num: int
    quality_score: float
    
    # Human review
    human_approved: bool
    human_feedback: str
    
    # Final output
    final_report: str
    status: str  # "in_progress", "approved", "rejected", "complete"
```

### 14.3 All Agents

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
import operator

# --- Models ---
gpt4o = ChatOpenAI(model="gpt-4o", temperature=0.1)
gpt4o_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# --- Tools ---

@tool
def search_web(query: str) -> str:
    """Search the web for current information, news, and general knowledge."""
    # Replace with Tavily: from langchain_community.tools.tavily_search import TavilySearchResults
    return f"[Web results for: {query}]"

@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for scientific papers and technical research."""
    return f"[arXiv results for: {query}]"

search_tools = [search_web, search_arxiv]
search_tool_node = ToolNode(search_tools, handle_tool_errors=True)

# Bind tools to the researcher
researcher_llm = gpt4o.bind_tools(search_tools)

# --- Pydantic Schemas for Structured Outputs ---

class ResearchPlan(BaseModel):
    summary: str = Field(description="One-sentence topic overview")
    main_questions: list[str] = Field(description="3-5 key questions to answer")
    search_queries: list[str] = Field(description="5-8 specific search queries")

class CritiqueResult(BaseModel):
    score: float = Field(description="Quality score 0.0 to 1.0", ge=0.0, le=1.0)
    strengths: list[str] = Field(description="What the draft does well")
    weaknesses: list[str] = Field(description="Specific gaps or problems")
    suggestions: list[str] = Field(description="Concrete improvement suggestions")
    verdict: str = Field(description="'approve' or 'revise'")

# --- Node: Planner ---

PLANNER_SYSTEM = """You are a research planning expert. Given a topic, create a 
structured research plan with specific search queries to gather comprehensive information."""

planner_llm = gpt4o.with_structured_output(ResearchPlan)

def planner_node(state: MARRSState) -> dict:
    print(f"[Planner] Creating research plan for: {state['topic']}")
    
    plan: ResearchPlan = planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Create a research plan for: {state['topic']}")
    ])
    
    return {
        "research_plan": plan.summary,
        "search_queries": plan.main_questions,  # Using questions as research guide
        "messages": [HumanMessage(
            content=f"Research Plan: {plan.model_dump_json(indent=2)}"
        )]
    }

# --- Node: Researcher (ReAct agent) ---

RESEARCHER_SYSTEM = """You are a thorough research assistant. Use your search tools 
to gather comprehensive information. Aim for at least 6-8 distinct, well-sourced 
findings. Each finding should be specific and citable."""

def researcher_node(state: MARRSState) -> dict:
    print(f"[Researcher] Starting research...")
    
    context_msg = f"""Research topic: {state['topic']}
    
Research plan:
{state['research_plan']}

Suggested questions to answer:
{chr(10).join(f"- {q}" for q in state.get('search_queries', []))}

Please search for information to answer these questions thoroughly."""

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=context_msg)
    ] + state["messages"][-5:]  # Include recent context, avoid bloat
    
    response = researcher_llm.invoke(messages)
    return {"messages": [response]}

def should_continue_research(state: MARRSState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "search_tools"
    return "writer"  # Research done, move to writing

def extract_research_results(state: MARRSState) -> dict:
    """After research loop ends, extract findings from the conversation."""
    # Get recent AI messages (tool responses + summaries)
    recent_ai_messages = [
        m for m in state["messages"][-10:]
        if hasattr(m, "content") and m.content
    ]
    
    findings_text = "\n\n".join([m.content[:500] for m in recent_ai_messages[-3:]])
    
    return {
        "findings": [findings_text],
        "sources": ["[Sources from search results]"]
    }

# --- Node: Writer ---

WRITER_SYSTEM = """You are an expert technical writer. Write clear, well-structured, 
accurate reports. Use the research findings provided. Include:
- An executive summary
- Properly organized sections
- Specific facts and data from the research
- Clear conclusions

If this is a revision, carefully address the critique provided."""

def writer_node(state: MARRSState) -> dict:
    print(f"[Writer] Drafting report (revision {state.get('revision_num', 0)})...")
    
    prompt_parts = [
        f"Topic: {state['topic']}",
        f"\nResearch Findings:\n{chr(10).join(state.get('findings', []))[:3000]}",
    ]
    
    if state.get("critique"):
        prompt_parts.append(f"\nPrevious critique to address:\n{state['critique']}")
        prompt_parts.append("\nPlease revise the report to address these issues.")
    else:
        prompt_parts.append("\nPlease write a comprehensive report on this topic.")
    
    response = gpt4o.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content="\n".join(prompt_parts))
    ])
    
    return {
        "draft": response.content,
        "messages": [response]
    }

# --- Node: Critic ---

CRITIC_SYSTEM = """You are a rigorous quality reviewer for research reports.
Evaluate the draft on: accuracy, completeness, clarity, structure, and evidence quality.
Be specific and constructive. Score 0.0 (terrible) to 1.0 (publication-ready)."""

critic_llm = gpt4o.with_structured_output(CritiqueResult)

def critic_node(state: MARRSState) -> dict:
    print(f"[Critic] Evaluating draft...")
    
    critique: CritiqueResult = critic_llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=f"""Please evaluate this report draft:

Topic: {state['topic']}

Draft:
{state['draft'][:4000]}

Provide your detailed critique.""")
    ])
    
    critique_summary = f"""Score: {critique.score:.2f}/1.0
Verdict: {critique.verdict}
Strengths: {'; '.join(critique.strengths[:2])}
Weaknesses: {'; '.join(critique.weaknesses[:2])}
Suggestions: {'; '.join(critique.suggestions[:3])}"""
    
    print(f"[Critic] Score: {critique.score:.2f} — {critique.verdict}")
    
    return {
        "critique": critique_summary,
        "quality_score": critique.score,
        "revision_num": state.get("revision_num", 0) + 1,
        "messages": [HumanMessage(content=f"Critique: {critique_summary}")]
    }

# --- Routing Function ---

MAX_REVISIONS = 3

def route_after_critic(state: MARRSState) -> str:
    score = state.get("quality_score", 0.0)
    revisions = state.get("revision_num", 0)
    
    if score >= 0.80:
        print(f"[Router] Quality sufficient ({score:.2f}). Moving to human review.")
        return "human_review"
    elif revisions >= MAX_REVISIONS:
        print(f"[Router] Max revisions ({MAX_REVISIONS}) reached. Forcing review.")
        return "human_review"
    else:
        print(f"[Router] Quality insufficient ({score:.2f}). Sending back to writer.")
        return "writer"

# --- Node: Human Review ---

def human_review_node(state: MARRSState) -> dict:
    print(f"[HumanReview] Pausing for human approval...")
    
    decision = interrupt({
        "prompt": "Please review the research report and provide your decision.",
        "topic": state["topic"],
        "quality_score": state.get("quality_score", 0.0),
        "revision_count": state.get("revision_num", 0),
        "draft_preview": state["draft"][:1000] + "...",
        "instructions": (
            "Reply with:\n"
            "  - 'approve' to finalize\n"
            "  - 'reject' to cancel\n"
            "  - Any text to request specific changes"
        )
    })
    
    if str(decision).lower() == "approve":
        return {"human_approved": True, "human_feedback": "Approved", "status": "approved"}
    elif str(decision).lower() == "reject":
        return {"human_approved": False, "human_feedback": "Rejected by human", "status": "rejected"}
    else:
        return {
            "human_approved": False,
            "human_feedback": str(decision),
            "critique": f"Human feedback: {decision}",
            "status": "needs_revision"
        }

def route_after_human_review(state: MARRSState) -> str:
    if state.get("human_approved"):
        return "finalize"
    elif state.get("status") == "rejected":
        return "__end__"
    else:
        return "writer"  # Human requested changes

# --- Node: Finalizer ---

def finalize_node(state: MARRSState) -> dict:
    print(f"[Finalizer] Preparing final report...")
    
    final = f"""# {state['topic']}
*Research Report | Generated by MARRS*

---

{state['draft']}

---
*Quality Score: {state.get('quality_score', 0.0):.2f} | Revisions: {state.get('revision_num', 0)}*
"""
    
    return {
        "final_report": final,
        "status": "complete"
    }
```

### 14.4 Graph Assembly

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

def build_marrs_graph():
    builder = StateGraph(MARRSState)
    
    # Register all nodes
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("search_tools", search_tool_node)
    builder.add_node("extract_research", extract_research_results)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)
    
    # Define the flow
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    
    # ReAct research loop
    builder.add_conditional_edges(
        "researcher",
        should_continue_research,
        {"search_tools": "search_tools", "writer": "extract_research"}
    )
    builder.add_edge("search_tools", "researcher")
    builder.add_edge("extract_research", "writer")
    
    # Writing/critique loop
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", "human_review": "human_review"}
    )
    
    # Human review and finalization
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"finalize": "finalize", "writer": "writer", "__end__": END}
    )
    builder.add_edge("finalize", END)
    
    return builder

# Compile with persistence
conn = sqlite3.connect("marrs.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

marrs_graph = build_marrs_graph().compile(checkpointer=checkpointer)

# Visualize
print(marrs_graph.get_graph(xray=True).draw_ascii())
```

### 14.5 Running MARRS

```python
from langchain_core.messages import HumanMessage
from langgraph.types import Command

def run_marrs(topic: str, user_id: str = "default"):
    """
    Run the MARRS workflow with streaming output and human-in-the-loop.
    """
    config = {"configurable": {"thread_id": f"{user_id}-{topic[:20]}"}}
    
    initial_state = {
        "topic": topic,
        "messages": [],
        "findings": [],
        "sources": [],
        "revision_num": 0,
        "quality_score": 0.0,
        "draft": "",
        "critique": "",
        "final_report": "",
        "human_approved": False,
        "status": "in_progress"
    }
    
    print(f"\n{'='*60}")
    print(f"MARRS Research System")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")
    
    # Stream execution with progress updates
    for event in marrs_graph.stream(initial_state, config, stream_mode="updates"):
        for node_name, updates in event.items():
            if node_name == "__interrupt__":
                # Human review is needed
                interrupt_data = updates[0].value
                print(f"\n{'='*40}")
                print("⏸️  HUMAN REVIEW REQUIRED")
                print(f"{'='*40}")
                print(f"Topic: {interrupt_data['topic']}")
                print(f"Quality Score: {interrupt_data['quality_score']:.2f}")
                print(f"\nDraft preview:\n{interrupt_data['draft_preview']}")
                print(f"\n{interrupt_data['instructions']}")
                
                decision = input("\nYour decision: ").strip()
                
                # Resume the graph with the human's decision
                for resume_event in marrs_graph.stream(
                    Command(resume=decision),
                    config,
                    stream_mode="updates"
                ):
                    for rnode, rupdates in resume_event.items():
                        print(f"  [{rnode}] {list(rupdates.keys())}")
                break
            else:
                update_keys = [k for k, v in updates.items() if v]
                if update_keys:
                    print(f"  ✓ [{node_name}] Updated: {update_keys}")
    
    # Get final state
    final_state = marrs_graph.get_state(config)
    
    print(f"\n{'='*60}")
    print("RESEARCH COMPLETE")
    print(f"Status: {final_state.values.get('status', 'unknown')}")
    print(f"Quality Score: {final_state.values.get('quality_score', 0):.2f}")
    print(f"Revisions: {final_state.values.get('revision_num', 0)}")
    print(f"{'='*60}")
    
    return final_state.values.get("final_report", "")

# Run it!
if __name__ == "__main__":
    report = run_marrs("The impact of attention mechanisms on large language models")
    print("\n" + report)
```

### 14.6 Full Graph Visualization

```
        +-------------+
        | __start__   |
        +-------------+
               ↓
        +-------------+
        |   planner   |
        +-------------+
               ↓
        +-------------+
        | researcher  | ←──────────┐
        +-------------+            │
          ↓         ↓              │
    [tool calls] [done]            │
          ↓         ↓              │
  +-----------+ +------------------+--+
  |search_    | |  extract_research   |
  |tools      | +---------------------+
  +-----------+         ↓
          │      +-------------+
          │      |    writer   | ←──────────────┐
          │      +-------------+                │
          │             ↓                       │
          │      +-------------+                │
          │      |    critic   |                │
          │      +-------------+                │
          │        ↓         ↓                  │
          │   [score<0.8] [score≥0.8]           │
          │        ↓         ↓                  │
          │   (back to)  +------------------+   │
          │    writer    |  human_review    |   │
          │              +------------------+   │
          │               ↓       ↓      ↓     │
          │          [approve] [reject] [edit] ─┘
          │               ↓       ↓
          │         +----------+  END
          │         | finalize |
          │         +----------+
          │               ↓
          └──────────────END
```

---

<a name="chapter-15"></a>
## Chapter 15 — Advanced Topics: Parallelism, Caching, and Beyond

### 15.1 True Parallel Execution with Fan-Out / Fan-In

LangGraph executes nodes in the same super-step concurrently. To run nodes in parallel, add edges from a single source to multiple targets:

```python
# Fan-out: both nodes run in the SAME super-step (parallel)
builder.add_edge("planner", "search_web_agent")
builder.add_edge("planner", "search_arxiv_agent")

# Fan-in: both must complete before 'synthesizer' runs
builder.add_edge("search_web_agent", "synthesizer")
builder.add_edge("search_arxiv_agent", "synthesizer")
```

When the planner finishes, both search agents start simultaneously. The `synthesizer` waits until both have completed. This is a **scatter-gather** pattern.

**Important:** When multiple nodes update the same state key in parallel, you need a reducer that can merge concurrent updates:

```python
class ParallelResearchState(TypedDict):
    # operator.add safely merges concurrent list updates
    findings: Annotated[list[str], operator.add]
    
    # Last-write-wins (OK if only one node writes this)
    status: str
```

### 15.2 Send API: Dynamic Parallelism

What if you don't know how many parallel branches you need at graph compile time? The `Send` API handles this:

```python
from langgraph.types import Send

def distribute_searches(state: MARRSState) -> list[Send]:
    """
    Dynamically create parallel search tasks based on the plan.
    Returns a list of Send objects — one per search query.
    """
    search_queries = state["search_queries"]
    
    # Each Send creates an independent invocation of the 'search_one' node
    return [
        Send("search_one", {"query": q, "topic": state["topic"]})
        for q in search_queries
    ]

def search_one_node(state: dict) -> dict:
    """Executes a single search query."""
    result = search_web.invoke(state["query"])
    return {"findings": [f"Q: {state['query']}\nA: {result}"]}

builder.add_node("search_one", search_one_node)
builder.add_conditional_edges(
    "planner",
    distribute_searches,  # Returns list[Send] for dynamic fan-out
)
builder.add_edge("search_one", "synthesizer")
```

This creates as many parallel `search_one` invocations as there are queries in the plan — 3, 8, or 20 — all running concurrently.

### 15.3 Node Caching (2025)

As covered in Chapter 13, node-level caching prevents redundant computation:

```python
from langgraph.cache import InMemoryCache, SqliteCache

# For development
cache = InMemoryCache()

# For production (persistent cache)
cache = SqliteCache("cache.db")

graph = builder.compile(checkpointer=checkpointer, cache=cache)

# Mark nodes as cacheable
builder.add_node("expensive_embedder", embed_documents_node, cache=True)
# Cache key is automatically computed from inputs
# Same inputs → cached output immediately, no API calls
```

### 15.4 Deferred Nodes (2025)

Deferred nodes run only after **all upstream parallel paths have completed** — the join point in a fan-out/fan-in pattern. Introduced in LangGraph's June 2025 release:

```python
# The synthesizer is automatically "deferred" — it waits for all fan-out branches
builder.add_node("synthesizer", synthesizer_node, defer=True)
```

Without `defer=True`, the synthesizer might run before all parallel searches complete. With it, LangGraph guarantees all upstream nodes have finished.

### 15.5 Async Nodes for I/O-Bound Workloads

For nodes that do network I/O (API calls, database queries), async nodes are significantly more efficient:

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

async def async_researcher(state: MARRSState) -> dict:
    """Async node — non-blocking I/O."""
    # Multiple concurrent API calls with asyncio.gather
    results = await asyncio.gather(
        search_web_async(state["topic"]),
        search_arxiv_async(state["topic"]),
        llm.ainvoke(state["messages"])
    )
    
    web_result, arxiv_result, llm_response = results
    return {
        "findings": [web_result, arxiv_result],
        "messages": [llm_response]
    }

# Use async graph execution
async def run():
    result = await marrs_graph.ainvoke(initial_state, config)
```

### 15.6 Configuration: Runtime Graph Parameterization

The `config["configurable"]` dict allows passing runtime parameters to nodes without putting them in state:

```python
from langchain_core.runnables import RunnableConfig

def writer_node(state: MARRSState, config: RunnableConfig) -> dict:
    # Access runtime config parameters
    max_length = config["configurable"].get("max_report_length", 2000)
    style = config["configurable"].get("writing_style", "technical")
    
    prompt = f"Write a {style} report of approximately {max_length} words about {state['topic']}"
    response = gpt4o.invoke(prompt)
    return {"draft": response.content}

# Caller specifies runtime parameters
config = {
    "configurable": {
        "thread_id": "session-1",
        "max_report_length": 5000,
        "writing_style": "executive summary"
    }
}
result = marrs_graph.invoke(initial_state, config)
```

### 15.7 Recursive Graphs: Agents That Call Themselves

A graph node can invoke the same compiled graph recursively (with depth limits):

```python
def meta_researcher(state: MARRSState) -> dict:
    """
    If the topic is broad, decompose into subtopics and
    run MARRS recursively on each.
    """
    subtopics = decompose_topic(state["topic"])
    
    if len(subtopics) > 1 and state.get("depth", 0) < 2:
        # Run MARRS on each subtopic
        sub_reports = []
        for subtopic in subtopics[:3]:  # Limit parallelism
            sub_config = {"configurable": {"thread_id": f"sub-{subtopic[:20]}"}}
            sub_result = marrs_graph.invoke(
                {**initial_state_template, "topic": subtopic, "depth": state.get("depth", 0) + 1},
                sub_config
            )
            sub_reports.append(sub_result.get("final_report", ""))
        
        return {"findings": sub_reports}
    
    return {}  # Non-recursive path
```

### 15.8 What's Next in the Ecosystem

As of 2025-2026, the LangGraph ecosystem is actively developing:

- **LangGraph Studio**: Visual IDE for building and debugging graphs interactively
- **Deep Agents**: Higher-level abstractions for complex planning agents built on LangGraph
- **A2A Protocol**: Cross-framework agent interoperability — agents built with different frameworks can communicate
- **MCP Integration**: Model Context Protocol support for tool standardization
- **Enhanced observability**: Per-token cost tracking, agent replay in LangSmith

### 15.9 Final Summary: Key Principles

After 15 chapters, here are the principles that will serve you best in production:

**1. Start simple, grow incrementally.**
Build the simplest graph that works. Add persistence, then memory, then HITL, then multi-agent only when you need them.

**2. State is your API.**
Design state carefully. It's the contract between all nodes. Changing it later in a checkpointed system has consequences.

**3. Reducers are critical.**
Wrong reducer = lost data or bloated state. Use `operator.add` for accumulating lists, custom reducers for merging logic.

**4. Observability first.**
Set up LangSmith tracing before writing production code. You cannot debug what you cannot see.

**5. Lean state, external storage.**
Never store large binary data in LangGraph state. Reference URLs/IDs to external storage.

**6. Thread IDs are identity.**
Choose thread ID schemes carefully. They are the primary key for all persistent state.

**7. HITL is a feature, not an afterthought.**
Design your graph with `interrupt()` points from the beginning. They are nearly free to add and invaluable in production.

---

## Appendix A: Quick Reference

### Core Imports
```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import Command, Send, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.func import entrypoint, task
```

### Graph Building Cheatsheet
```python
# 1. Define state
class State(TypedDict): ...

# 2. Build graph
builder = StateGraph(State)
builder.add_node("name", function)
builder.add_edge("a", "b")
builder.add_conditional_edges("a", routing_fn, {"route1": "b", "route2": "c"})
builder.add_edge(START, "first_node")

# 3. Compile
graph = builder.compile(checkpointer=..., store=..., interrupt_before=[...])

# 4. Run
result = graph.invoke(input, config)
for event in graph.stream(input, config, stream_mode="updates"): ...
state = graph.get_state(config)
graph.update_state(config, updates, as_node="node_name")
```

### Common Patterns
```python
# Message history with reducer
messages: Annotated[list[BaseMessage], add_messages]

# Accumulating list
items: Annotated[list[str], operator.add]

# ReAct routing
def should_continue(state) -> str:
    return "tools" if state["messages"][-1].tool_calls else "__end__"

# Human interrupt
decision = interrupt({"question": "...", "data": ...})

# Parallel fan-out via Send
return [Send("node", {"input": item}) for item in items]

# Thread-scoped config
config = {"configurable": {"thread_id": "unique-id"}}
```

---

## Appendix B: Further Reading

- **Official LangGraph Documentation**: `docs.langchain.com/oss/python/langgraph`
- **LangChain Academy**: Free LangGraph course at `academy.langchain.com`
- **GitHub**: `github.com/langchain-ai/langgraph` — source code and examples
- **Pregel Paper**: "Pregel: A System for Large-Scale Graph Processing" (Malewicz et al., 2010)
- **ReAct Paper**: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- **LangSmith**: `smith.langchain.com` — tracing and evaluation platform
- **LangGraph v1.0 Blog Post**: `blog.langchain.com/langgraph-1-0`

---

*This curriculum reflects the LangGraph ecosystem as of April 2026 (LangGraph v1.x). The framework is actively developed; always check the official documentation for the most current API.*
