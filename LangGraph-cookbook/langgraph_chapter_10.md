# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 10 — Subgraphs and Multi-Agent Architectures

> *"Multi-agent systems coordinate specialized components to tackle complex workflows. However, not every complex task requires this approach — a single agent with the right tools and prompt can often achieve similar results."*
> — LangChain official documentation

---

### What This Chapter Covers

Every graph so far has been one flat set of nodes. This chapter covers what happens when a node is itself a graph — subgraphs — and the architectural patterns built on top of that mechanism for coordinating multiple agents: a main agent delegating to specialists as tools, and agents handing control directly to one another.

By the end you will understand:

1. The two ways to compose a subgraph into a parent graph, and exactly when each applies
2. Multi-level subgraph nesting and what state is visible at each level
3. The three subgraph persistence modes — per-invocation, per-thread, stateless — and the checkpoint-namespace conflict that per-thread mode can create
4. Namespace isolation: why call order matters, and the stable-name fix
5. Why subgraph state becomes uninspectable the moment it's called from inside a tool function — and why this doesn't affect interrupts
6. The current official multi-agent pattern taxonomy — Subagents, Handoffs, Skills, Router, Custom Workflow — with real performance numbers for choosing between them
7. The Subagents pattern in full: a main agent coordinating specialists as tools (what the ecosystem broadly calls "supervisor")
8. The Handoffs pattern in full: `Command.PARENT`-based control transfer between agent nodes (what the ecosystem broadly calls "swarm"), and the context-engineering rules that keep the message history valid across a handoff
9. Where the standalone `langgraph-supervisor` / `langgraph-swarm` packages fit, and a version gotcha in their published examples
10. MARRS checkpoint: decomposing the capstone into a supervisor coordinating a researcher subagent and a writer subagent

---

### 10.1 Subgraph Mechanics: Two Composition Patterns

A subgraph is simply a compiled `StateGraph` used as a node in another graph. There are exactly two ways to wire one in, and the choice is dictated entirely by whether the parent and subgraph state schemas overlap.

| Pattern | When to use | How |
|---|---|---|
| **Add a subgraph as a node** | Parent and subgraph **share state keys** | Pass the compiled subgraph directly to `add_node` — no wrapper needed |
| **Call a subgraph inside a node** | Parent and subgraph have **different schemas** (no shared keys) | Write a node function that transforms parent state → subgraph input, invokes the subgraph, transforms the output back |

#### 10.1.1 Shared Schema: Add as a Node Directly

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class SubgraphState(TypedDict):
    foo: str  # shared with parent graph state
    bar: str  # private to the subgraph

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    # Reads a subgraph-private key ('bar'), writes to the shared key ('foo')
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# Parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)          # ← the compiled subgraph IS the node
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream({"foo": "foo"}, version="v2"):
    if chunk["type"] == "updates":
        print(chunk["data"])
# {'node_1': {'foo': 'hi! foo'}}
# {'node_2': {'foo': 'hi! foobar'}}
```

The subgraph reads from and writes to the parent's `foo` channel automatically, because the channel name is identical on both sides. `bar` never leaves the subgraph.

#### 10.1.2 Different Schemas: Call Inside a Node

```python
class SubgraphState(TypedDict):
    # None of these keys are shared with the parent graph state
    bar: str
    baz: str

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

class ParentState(TypedDict):
    foo: str

def node_2(state: ParentState):
    # Manually transform state in both directions — this is the entire point
    # of this pattern: nothing is shared automatically.
    response = subgraph.invoke({"bar": state["foo"]})
    return {"foo": response["bar"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", lambda state: {"foo": "hi! " + state["foo"]})
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()
```

**Rule of thumb:** if you're reaching for this pattern only to avoid naming a shared key, reconsider — sharing the key and using the direct-node pattern is simpler. Use the wrapper-function pattern when you genuinely want isolation: a subagent's internal reasoning messages, working state, or scratch variables that the parent has no business seeing.

#### 10.1.3 Multi-Level Nesting

Subgraphs nest arbitrarily deep. Each level only sees its own state — a grandchild cannot see the parent's keys, and the parent cannot see the grandchild's:

```python
# Grandchild
class GrandChildState(TypedDict):
    my_grandchild_key: str

def grandchild_1(state: GrandChildState) -> GrandChildState:
    # parent or child keys are NOT accessible here
    return {"my_grandchild_key": state["my_grandchild_key"] + ", how are you"}

grandchild_graph = (
    StateGraph(GrandChildState)
    .add_node("grandchild_1", grandchild_1)
    .add_edge(START, "grandchild_1")
    .compile()
)

# Child
class ChildState(TypedDict):
    my_child_key: str

def call_grandchild_graph(state: ChildState) -> ChildState:
    # parent or grandchild keys are NOT accessible here
    output = grandchild_graph.invoke({"my_grandchild_key": state["my_child_key"]})
    return {"my_child_key": output["my_grandchild_key"] + " today?"}

child_graph = (
    StateGraph(ChildState)
    .add_node("child_1", call_grandchild_graph)
    .add_edge(START, "child_1")
    .compile()
)

# Parent
class ParentState(TypedDict):
    my_key: str

def call_child_graph(state: ParentState) -> ParentState:
    output = child_graph.invoke({"my_child_key": state["my_key"]})
    return {"my_key": output["my_child_key"]}

parent_graph = (
    StateGraph(ParentState)
    .add_node("parent_1", lambda s: {"my_key": "hi " + s["my_key"]})
    .add_node("child", call_child_graph)
    .add_node("parent_2", lambda s: {"my_key": s["my_key"] + " bye!"})
    .add_edge(START, "parent_1")
    .add_edge("parent_1", "child")
    .add_edge("child", "parent_2")
    .compile()
)

for chunk in parent_graph.stream({"my_key": "Bob"}, subgraphs=True, version="v2"):
    if chunk["type"] == "updates":
        print(chunk["ns"], chunk["data"])
# () {'parent_1': {'my_key': 'hi Bob'}}
# ('child:2e26...', 'child_1:781b...') {'grandchild_1': {'my_grandchild_key': 'hi Bob, how are you'}}
# ('child:2e26...',) {'child_1': {'my_child_key': 'hi Bob, how are you today?'}}
# () {'child': {'my_key': 'hi Bob, how are you today?'}}
# () {'parent_2': {'my_key': 'hi Bob, how are you today? bye!'}}
```

Notice the `ns` (namespace) tuple grows with nesting depth — a two-level-deep event carries a two-element namespace tuple, each element identifying one level's node and a unique invocation ID.

---

### 10.2 Subgraph Persistence: Three Modes

When a subgraph is invoked repeatedly — most commonly because it's wrapped as a tool a supervisor calls — you must decide what happens to its internal state between calls. This is controlled by the `checkpointer` parameter on the *subgraph's own* `.compile()` call, independent of the parent's checkpointer:

| Mode | `checkpointer=` | Behavior |
|---|---|---|
| **Per-invocation** (default) | `None` | Each call starts fresh; inherits the parent's checkpointer for interrupts and durable execution *within* that one call |
| **Per-thread** | `True` | State accumulates across calls on the same thread — each call picks up where the last left off |
| **Stateless** | `False` | No checkpointing at all — a plain function call; no interrupts, no durable execution |

The parent graph must itself be compiled with a checkpointer for any of this to work — subgraph persistence features (interrupts, state inspection, per-thread memory) all depend on it.

#### 10.2.1 Per-Invocation (Default): The Right Choice for Most Multi-Agent Systems

Use this when each call to a subagent is an independent, one-off request — "look up this order," "summarize this document" — and the subagent has no need to remember previous calls.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

@tool
def fruit_info(fruit_name: str) -> str:
    """Look up fruit info."""
    return f"Info about {fruit_name}"

# No checkpointer set on the subagent — it inherits the parent's
fruit_agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[fruit_info],
    system_prompt="You are a fruit expert. Respond in one sentence.",
)

@tool
def ask_fruit_expert(question: str) -> str:
    """Ask the fruit expert. Use for ALL fruit questions."""
    response = fruit_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[ask_fruit_expert],
    system_prompt="Delegate all fruit questions to ask_fruit_expert.",
    checkpointer=MemorySaver(),
)
```

Each call to `ask_fruit_expert` starts with fresh subagent state — asking about apples, then bananas, produces two independent 4-message subagent runs with no memory of each other. Within a single call, the subagent can still use `interrupt()` (e.g., inside `fruit_info` to require approval) and correctly pause/resume, because it inherits the parent's checkpointer for that one invocation.

**Multiple calls in the same turn don't conflict** under per-invocation persistence — if the supervisor calls `ask_fruit_expert` for both "apples" and "bananas" in parallel, each gets its own isolated checkpoint namespace.

#### 10.2.2 Per-Thread: When a Subagent Needs Multi-Turn Memory

Use this when a subagent genuinely needs to remember earlier calls — a research assistant building context over several exchanges, a coding assistant tracking which files it already edited.

```python
fruit_agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[fruit_info],
    system_prompt="You are a fruit expert.",
    checkpointer=True,   # ← per-thread: state accumulates across calls
)
```

```python
# First call: 4 messages
response = agent.invoke({"messages": [{"role": "user", "content": "Tell me about apples"}]}, config)
# Second call: 8 messages — the subagent REMEMBERS the apples conversation
response = agent.invoke({"messages": [{"role": "user", "content": "Now tell me about bananas"}]}, config)
```

**The parallel-call trap:** per-thread subgraphs do **not** support parallel tool calls. If the LLM tries to call the same per-thread subagent twice in parallel (asking about apples and bananas simultaneously), both calls write to the *same* checkpoint namespace and conflict. The documented mitigation is to cap parallel calls to that tool:

```python
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[ask_fruit_expert],
    middleware=[ToolCallLimitMiddleware(tool_name="ask_fruit_expert", run_limit=1)],
    checkpointer=MemorySaver(),
)
```

If you're building with raw `StateGraph` rather than `create_agent`, you're responsible for preventing this yourself — either disable parallel tool calling on the model, or add logic ensuring the same subgraph isn't invoked twice in the same turn.

**The namespace isolation gotcha:** when you have multiple *different* per-thread subagents (a fruit expert and a veggie expert), each needs its own storage so their checkpoints don't collide. If subgraphs are **called inside a node** (Section 10.1.2's pattern), LangGraph assigns namespaces by **call order** — first call, second call. Reordering your calls silently mixes up which subgraph loads which state. The fix is to wrap each subagent in its own `StateGraph` with a unique node name, which gives it a stable namespace independent of call order:

```python
from langgraph.graph import MessagesState, StateGraph

def create_sub_agent(model, *, name, **kwargs):
    """Wrap an agent with a unique node name for a stable, order-independent namespace."""
    agent = create_agent(model=model, name=name, **kwargs)
    return (
        StateGraph(MessagesState)
        .add_node(name, agent)          # unique name → stable namespace
        .add_edge("__start__", name)
        .compile()
    )

fruit_agent = create_sub_agent("anthropic:claude-sonnet-5", name="fruit_agent",
                                tools=[fruit_info], system_prompt="...", checkpointer=True)
veggie_agent = create_sub_agent("anthropic:claude-sonnet-5", name="veggie_agent",
                                 tools=[veggie_info], system_prompt="...", checkpointer=True)
```

Subgraphs **added as nodes** (Section 10.1.1's pattern) already get automatic name-based namespaces — this wrapper is only needed for the call-inside-a-node pattern.

#### 10.2.3 Stateless: No Checkpointing Overhead

```python
subgraph = subgraph_builder.compile(checkpointer=False)
```

Runs like a plain function call — no durable execution, no interrupts, no resumability. If the process crashes mid-run, the subgraph cannot recover and must restart from the beginning. Use this only when a subgraph's work is quick, side-effect-free, and cheap to simply re-run on failure.

#### 10.2.4 Feature Comparison

| Feature | Per-invocation (default) | Per-thread | Stateless |
|---|---|---|---|
| Interrupts (HITL) | ✅ | ✅ | ❌ |
| Multi-turn memory | ❌ | ✅ | ❌ |
| Multiple calls, different subgraphs | ✅ | ⚠️ (needs stable names) | ✅ |
| Multiple calls, same subgraph | ✅ | ❌ (checkpoint conflict) | ✅ |
| State inspection | ⚠️ | ✅ | ❌ |

---

### 10.3 Viewing Subgraph State — and Where It Silently Stops Working

```python
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state
```

**This requires LangGraph to statically discover the subgraph** — meaning it was added as a node (10.1.1) or called inside a node (10.1.2). It does **not** work when a subgraph is invoked inside a **tool function** — which is exactly the shape of the subagents-as-tools pattern in Sections 10.2.1–10.2.2 and Section 10.5. If your multi-agent architecture wraps subagents as tools (the most common production shape), you lose `get_state(subgraphs=True)` visibility into them, even though everything else about them — persistence, interrupts — works normally.

**Interrupts still propagate to the top-level graph regardless of nesting depth or invocation style.** This is the important asymmetry to remember: you can build and resume a paused subagent-as-tool exactly as shown in Chapter 7, you simply can't introspect its internal state snapshot through `get_state(subgraphs=True)` the way you could with a subgraph added as a node. If deep state inspection matters for your use case, prefer the add-as-node or call-inside-a-node patterns over subagents-as-tools.

---

### 10.4 The Current Multi-Agent Pattern Taxonomy

The ecosystem has largely settled on five named patterns for structuring multi-agent systems. "Supervisor" and "swarm" — terms you'll see constantly in blog posts, talks, and older tutorials — map onto two of these five, and this chapter uses both vocabularies side by side since both are in active use.

| Pattern | How it works | Common alternate name |
|---|---|---|
| **Subagents** | A main agent coordinates subagents as tools. All routing passes through the main agent. | "Supervisor" |
| **Handoffs** | Agents transfer control to each other directly via tool calls that update state. | "Swarm" |
| **Skills** | A single agent loads specialized prompts/knowledge on demand, staying in control throughout. | — |
| **Router** | A routing step classifies input and dispatches to one or more specialized agents; results are synthesized. | "Router" / classic dispatcher |
| **Custom workflow** | Bespoke `StateGraph` mixing deterministic logic and agentic behavior, embedding other patterns as nodes. | — |

You can mix these — a Subagents architecture can invoke a tool that runs a Custom Workflow or a Router internally.

#### 10.4.1 Choosing a Pattern: Real Numbers, Not Vibes

The official documentation quantifies the tradeoff across three scenarios, measured in model calls and tokens processed — worth internalizing because the "obviously correct" pattern changes depending on which scenario dominates your traffic:

**One-shot request** ("Buy coffee," asked once):

| Pattern | Model calls |
|---|---|
| Subagents | 4 |
| Handoffs | 3 |
| Skills | 3 |
| Router | 3 |

Subagents costs one extra call because results flow back through the main agent before reaching the user — that round-trip is the price of centralized control.

**Repeat request** (same request, asked again in the same conversation):

| Pattern | Turn 2 calls | Total (both turns) |
|---|---|---|
| Subagents | 4 | 8 |
| Handoffs | 2 | 5 |
| Skills | 2 | 5 |
| Router | 3 | 6 |

Subagents are stateless by design — every call repeats the full flow, which gives strong context isolation but no savings on repetition. Handoffs and Skills are stateful: once the right agent/context is active, later turns skip the re-routing step entirely, saving roughly 40–50% of calls.

**Multi-domain request** ("Compare Python, JavaScript, and Rust," three specialist domains):

| Pattern | Model calls | Total tokens |
|---|---|---|
| Subagents | 5 | ~9K |
| Handoffs | 7+ | ~14K+ |
| Skills | 3 | ~15K |
| Router | 5 | ~9K |

Here Handoffs loses badly — it's inherently sequential (one agent hands off to the next), so it can't parallelize across the three domains, and the growing conversation history adds token overhead with each hop. Subagents and Router both dispatch specialists in parallel and stay far cheaper. Skills has the fewest calls but the most tokens, because once a skill's ~2K-token documentation is loaded, every subsequent call in the conversation carries that full context forward.

**The takeaway pattern:** Subagents and Router win when you need parallel dispatch across genuinely separate domains or strong context isolation between them. Handoffs and Skills win when the same specialist is likely to keep handling several turns in a row. Neither is universally "the multi-agent pattern" — the traffic shape decides.

---

### 10.5 The Subagents Pattern ("Supervisor")

A main agent treats each specialist as a tool. It decides when to call which specialist, and every response flows back through the main agent before reaching the user — the main agent is always in control, and the user only ever talks to it directly.

```python
from langchain.agents import create_agent
from langchain.tools import tool

# ── Specialist subagents ──────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return f"[results for '{query}']"

@tool
def fetch_paper(arxiv_id: str) -> str:
    """Fetch an academic paper abstract by arXiv ID."""
    return f"[abstract for {arxiv_id}]"

researcher = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[web_search, fetch_paper],
    system_prompt="You are a research specialist. Investigate topics thoroughly and cite sources.",
)

@tool
def save_draft(content: str) -> str:
    """Save a draft to the working document."""
    return "Draft saved."

writer = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[save_draft],
    system_prompt="You are a writing specialist. Produce clear, well-structured prose.",
)

# ── Wrap each subagent as a tool for the main (supervisor) agent ──────────────
@tool
def delegate_to_researcher(task: str) -> str:
    """Delegate a research task to the research specialist. Use for ANY fact-finding."""
    response = researcher.invoke({"messages": [{"role": "user", "content": task}]})
    return response["messages"][-1].content

@tool
def delegate_to_writer(task: str) -> str:
    """Delegate a writing task to the writing specialist. Use for ANY drafting."""
    response = writer.invoke({"messages": [{"role": "user", "content": task}]})
    return response["messages"][-1].content

# ── The supervisor ────────────────────────────────────────────────────────────
from langgraph.checkpoint.memory import MemorySaver

supervisor = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[delegate_to_researcher, delegate_to_writer],
    system_prompt=(
        "You coordinate a research specialist and a writing specialist. "
        "ALWAYS delegate research questions to delegate_to_researcher and "
        "drafting tasks to delegate_to_writer. Never do the work yourself."
    ),
    checkpointer=MemorySaver(),
)

result = supervisor.invoke(
    {"messages": [{"role": "user", "content": "Research transformer architectures and draft a summary."}]},
    config={"configurable": {"thread_id": "session-1"}},
)
```

This is exactly the subgraph-persistence machinery from Section 10.2 in action: `delegate_to_researcher` and `delegate_to_writer` each invoke a subagent from inside a tool, using per-invocation persistence by default. If either specialist needs multi-turn memory of its own, compile it with `checkpointer=True` and apply the `ToolCallLimitMiddleware` guard from Section 10.2.2.

**Why "4 model calls" for a one-shot request (Section 10.4.1):** the supervisor reasons about which specialist to call (1), the specialist may call its own tools and respond internally (not counted separately here), the specialist's answer returns to the supervisor which must synthesize a final reply (2), and so on — the exact count depends on how many specialists get involved, but the structural reason it costs more than Handoffs is always the same: everything routes back through the coordinating agent.

---

### 10.6 The Handoffs Pattern ("Swarm")

Instead of a central coordinator, agents transfer control to each other directly. A tool call updates a state variable (`active_agent`, `current_step`) that determines which configuration or which agent node handles the next turn — and unlike Subagents, an agent can respond to the user directly without routing back through anything.

The term comes from OpenAI's Agents SDK, where tools like `transfer_to_sales_agent` move control between agents. There are two implementation approaches.

#### 10.6.1 Single Agent With Middleware (Simpler — Prefer This by Default)

One agent, whose system prompt and available tools change dynamically based on a state variable. Middleware intercepts each model call and reconfigures the agent before it runs:

```python
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable

class SupportState(AgentState):
    current_step: str = "triage"
    warranty_status: str | None = None

@tool
def record_warranty_status(status: str, runtime: ToolRuntime[None, SupportState]) -> Command:
    """Record warranty status and transition to the next step."""
    return Command(update={
        "messages": [ToolMessage(content=f"Warranty status: {status}", tool_call_id=runtime.tool_call_id)],
        "warranty_status": status,
        "current_step": "specialist",   # ← this is the entire handoff mechanism
    })

@wrap_model_call
def apply_step_config(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Reconfigure the agent's prompt and tools based on current_step."""
    step = request.state.get("current_step", "triage")
    configs = {
        "triage": {"prompt": "Collect warranty information from the customer.",
                   "tools": [record_warranty_status]},
        "specialist": {"prompt": "Provide a solution based on warranty status: {warranty_status}",
                       "tools": [provide_solution, escalate]},
    }
    config = configs[step]
    request = request.override(system_prompt=config["prompt"].format(**request.state), tools=config["tools"])
    return handler(request)

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[record_warranty_status, provide_solution, escalate],
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),   # required — current_step must persist across turns
)
```

Nothing here is a subgraph at all — it's one agent whose behavior mutates. This is the **default recommendation** for handoffs: simpler to reason about, one message history throughout, no context-engineering decisions to get wrong.

#### 10.6.2 Multiple Agent Subgraphs (Only When You Need Bespoke Agents)

When each "specialist" genuinely needs its own complex internal graph (its own reflection loop, its own retrieval step) rather than just a different prompt, use distinct agent nodes and transfer between them with `Command(goto=..., graph=Command.PARENT)` — the exact mechanism introduced in Chapter 4:

```python
from typing import Literal
from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import NotRequired

class MultiAgentState(AgentState):
    active_agent: NotRequired[str]

@tool
def transfer_to_sales(runtime: ToolRuntime) -> Command:
    """Transfer to the sales agent."""
    last_ai_message = next(m for m in reversed(runtime.state["messages"]) if isinstance(m, AIMessage))
    transfer_message = ToolMessage(content="Transferred to sales agent", tool_call_id=runtime.tool_call_id)
    return Command(
        goto="sales_agent",
        update={"active_agent": "sales_agent", "messages": [last_ai_message, transfer_message]},
        graph=Command.PARENT,   # ← escape the current agent's subgraph, land in the parent
    )

@tool
def transfer_to_support(runtime: ToolRuntime) -> Command:
    """Transfer to the support agent."""
    last_ai_message = next(m for m in reversed(runtime.state["messages"]) if isinstance(m, AIMessage))
    transfer_message = ToolMessage(content="Transferred to support agent", tool_call_id=runtime.tool_call_id)
    return Command(
        goto="support_agent",
        update={"active_agent": "support_agent", "messages": [last_ai_message, transfer_message]},
        graph=Command.PARENT,
    )

sales_agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[transfer_to_support],
    system_prompt="You are a sales agent. Transfer technical questions to support.",
)
support_agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[transfer_to_sales],
    system_prompt="You are a support agent. Transfer pricing questions to sales.",
)

def call_sales_agent(state: MultiAgentState) -> Command:
    return sales_agent.invoke(state)

def call_support_agent(state: MultiAgentState) -> Command:
    return support_agent.invoke(state)

def route_after_agent(state: MultiAgentState) -> Literal["sales_agent", "support_agent", "__end__"]:
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        return "__end__"   # agent answered without handing off — done
    return state.get("active_agent") or "sales_agent"

builder = StateGraph(MultiAgentState)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("support_agent", call_support_agent)
builder.add_conditional_edges(START, lambda s: s.get("active_agent") or "sales_agent",
                               ["sales_agent", "support_agent"])
builder.add_conditional_edges("sales_agent", route_after_agent,
                               ["sales_agent", "support_agent", END])
builder.add_conditional_edges("support_agent", route_after_agent,
                               ["sales_agent", "support_agent", END])
graph = builder.compile()
```

#### 10.6.3 Context Engineering: The Rule That Keeps Handoffs From Breaking

This is the single most important detail in the multi-agent-subgraphs approach, and it's easy to get wrong in a way that produces confusing, hard-to-debug errors. Every LLM API expects each tool call to be paired with exactly one tool response. When `Command.PARENT` hands off to a different agent node, you must pass **both**:

1. The `AIMessage` that contained the tool call which triggered the handoff
2. A synthetic `ToolMessage` acknowledging that call

```python
@tool
def transfer_to_sales(runtime: ToolRuntime) -> Command:
    last_ai_message = runtime.state["messages"][-1]
    transfer_message = ToolMessage(content="Transferred to sales agent", tool_call_id=runtime.tool_call_id)
    return Command(
        goto="sales_agent",
        update={
            "active_agent": "sales_agent",
            "messages": [last_ai_message, transfer_message],   # both, always
        },
        graph=Command.PARENT,
    )
```

Omit either half of the pair and the receiving agent sees a malformed conversation — a tool call with no matching response, or vice versa — which most providers reject outright or handle unpredictably.

**Why not just pass the entire subagent conversation history to the next agent?** You could, but it usually backfires: the receiving agent gets confused by the previous agent's internal reasoning that has nothing to do with its job, and token costs climb for no benefit. The documented guidance is to pass *only* the handoff pair, keeping the parent graph's context focused on high-level coordination. If the receiving agent genuinely needs more context than that, summarize the outgoing agent's work into the `ToolMessage`'s content rather than forwarding raw messages.

**Returning control to the user:** when an agent's turn ends without a further handoff, make sure the last message is a plain `AIMessage` with no pending tool calls — this is what `route_after_agent` above checks for, and it's what signals to both the LLM API and any chat UI that the turn genuinely finished.

---

### 10.7 Standalone Packages: `langgraph-supervisor` and `langgraph-swarm`

Two separate, official `langchain-ai` GitHub packages implement ready-made versions of these two patterns: `langgraph-supervisor` (`create_supervisor`) and `langgraph-swarm` (`create_swarm`). Both come with out-of-the-box streaming, memory, and HITL support, and remain actively referenced in the ecosystem.

```python
# pip install langgraph-supervisor
from langgraph_supervisor import create_supervisor
```

**A version gotcha worth knowing before you reach for either package:** as of this writing, published examples for both packages still show `from langgraph.prebuilt import create_react_agent` for constructing the specialist agents — which, per Chapter 9, is deprecated as of LangGraph v1.0. The packages' own internal supervisor/swarm orchestration logic is unaffected by this deprecation, but if you follow a tutorial for either package verbatim, swap the specialist-construction line for `from langchain.agents import create_agent` (with `prompt=` renamed to `system_prompt=`) rather than copying the deprecated import.

**When to reach for the packages versus hand-rolling the patterns from Sections 10.5–10.6:** the packages save boilerplate for the common case and are worth trying first. Hand-roll when you need custom routing logic the package doesn't expose, or when — as this chapter's MARRS integration does next — the "supervisor" needs to be more than a thin dispatcher and carries its own domain-specific state and routing rules.

---

### 10.8 MARRS Checkpoint: A Supervisor Coordinating Research and Writing Subagents

We now restructure MARRS as a Subagents ("supervisor") architecture: a coordinating agent delegates to a researcher subagent and a writer subagent, each wrapped as tools, with per-invocation persistence for both and a shared checkpointer at the supervisor level for the whole session.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH SUBAGENT (per-invocation persistence — stateless across calls)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information on a topic."""
    return f"[{max_results} results for '{query}']"

@tool
def fetch_academic_paper(arxiv_id: str) -> str:
    """Fetch an academic paper's abstract by arXiv ID."""
    return f"[abstract for arXiv:{arxiv_id}]"

research_subagent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[web_search, fetch_academic_paper],
    system_prompt=(
        "You are MARRS's research specialist. Investigate the given topic thoroughly, "
        "using both web search and academic sources where relevant. Return a structured "
        "summary of findings with source attributions."
    ),
    # No checkpointer set — per-invocation, inherits the supervisor's checkpointer
    # for the duration of a single call (Section 10.2.1)
)


# ─────────────────────────────────────────────────────────────────────────────
# WRITER SUBAGENT (per-invocation persistence)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def format_citation(source: str) -> str:
    """Format a source into a citation string."""
    return f"[Source: {source}]"

writer_subagent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[format_citation],
    system_prompt=(
        "You are MARRS's writing specialist. Given research findings, produce a clear, "
        "well-structured report with proper citations. Match the requested tone and depth."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# DELEGATION TOOLS — wrap each subagent for the supervisor
# ─────────────────────────────────────────────────────────────────────────────

@tool
def delegate_research(topic: str, focus_areas: str = "") -> str:
    """Delegate a research task to the research specialist.

    Args:
        topic: The subject to research.
        focus_areas: Optional specific angles to emphasize.
    """
    task = f"Research: {topic}" + (f"\nFocus on: {focus_areas}" if focus_areas else "")
    response = research_subagent.invoke({"messages": [{"role": "user", "content": task}]})
    return response["messages"][-1].content

@tool
def delegate_writing(findings: str, tone: str = "professional") -> str:
    """Delegate report writing to the writing specialist.

    Args:
        findings: Research findings to base the report on.
        tone: Desired tone — 'professional', 'casual', or 'academic'.
    """
    task = f"Write a {tone} report based on these findings:\n{findings}"
    response = writer_subagent.invoke({"messages": [{"role": "user", "content": task}]})
    return response["messages"][-1].content


# ─────────────────────────────────────────────────────────────────────────────
# THE SUPERVISOR — persistent at the session level
# ─────────────────────────────────────────────────────────────────────────────

def build_marrs_supervisor(use_sqlite: bool = False):
    if use_sqlite:
        conn = sqlite3.connect("marrs_supervisor.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    else:
        checkpointer = MemorySaver()

    supervisor = create_agent(
        model="anthropic:claude-sonnet-5",
        tools=[delegate_research, delegate_writing],
        system_prompt=(
            "You coordinate MARRS research reports. For any research request:\n"
            "1. ALWAYS delegate fact-finding to delegate_research first.\n"
            "2. ALWAYS delegate drafting to delegate_writing once research is complete.\n"
            "3. Never research or write content yourself — delegate everything.\n"
            "4. Present the final report to the user once both steps are complete."
        ),
        checkpointer=checkpointer,
    )
    return supervisor

marrs_supervisor = build_marrs_supervisor()


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_marrs_supervisor_demo(topic: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = marrs_supervisor.invoke(
        {"messages": [{"role": "user", "content": f"Produce a research report on: {topic}"}]},
        config,
    )
    return result["messages"][-1].content


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_supervisor_delegates_to_research():
    """The supervisor should call delegate_research, not answer directly."""
    config = {"configurable": {"thread_id": "test-delegate-research"}}
    result = marrs_supervisor.invoke(
        {"messages": [{"role": "user", "content": "Produce a research report on quantum computing"}]},
        config,
    )
    tool_names_called = {
        tc["name"]
        for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
        for tc in m.tool_calls
    }
    assert "delegate_research" in tool_names_called
    assert "delegate_writing" in tool_names_called

def test_research_subagent_independent_of_supervisor_state():
    """Per-invocation persistence: two calls to the research subagent don't share state."""
    r1 = research_subagent.invoke({"messages": [{"role": "user", "content": "Research: AI safety"}]})
    r2 = research_subagent.invoke({"messages": [{"role": "user", "content": "Research: quantum computing"}]})
    # Each call starts fresh — 2 messages in, roughly 2-4 out, no leakage between calls
    assert len(r1["messages"]) < 10
    assert len(r2["messages"]) < 10
```

---

### 10.9 Chapter Summary

**Subgraphs** compose two ways: add a compiled subgraph directly as a node when schemas share keys (automatic channel communication, no wrapper), or call it inside a node function with manual state transformation when schemas differ (full isolation). Nesting is arbitrarily deep, with each level's state invisible to its neighbors.

**Subgraph persistence** has three modes controlled by the subgraph's own `checkpointer` parameter: per-invocation (default — fresh state each call, the right choice for most subagents-as-tools architectures), per-thread (state accumulates — needed for subagents with genuine multi-turn memory, but incompatible with parallel calls to the same subgraph without a `ToolCallLimitMiddleware` guard), and stateless (no persistence at all).

**Namespace isolation** matters once you have multiple different per-thread subgraphs: calling them inside a node assigns namespaces by call order, which breaks if you ever reorder the calls — wrap each in its own named `StateGraph` node for a stable namespace instead.

**Subgraph state inspection** via `get_state(subgraphs=True)` requires static discoverability and silently stops working the moment a subgraph is invoked from inside a tool function — the shape nearly every subagents-as-tools architecture takes. Interrupts are unaffected by this and propagate to the top level regardless.

**The current multi-agent taxonomy** names five patterns — Subagents, Handoffs, Skills, Router, Custom Workflow — where "Subagents" and "Handoffs" map onto the widely-used "Supervisor" and "Swarm" vocabulary respectively. Real performance numbers show Subagents and Router winning on parallel, multi-domain work; Handoffs and Skills winning on repeated, single-domain conversation; and Handoffs specifically losing on multi-domain work because it's inherently sequential.

**Handoffs** has two implementations: single-agent-with-middleware (simpler, one message history, the default recommendation) and multiple-agent-subgraphs with `Command(goto=..., graph=Command.PARENT)` (for when specialists genuinely need distinct internal graphs). The context-engineering rule that keeps subgraph handoffs valid — pass exactly the triggering `AIMessage` plus a synthetic `ToolMessage`, not the full subagent history — is the single most common source of malformed-conversation errors in hand-rolled multi-agent systems.

**Standalone packages** (`langgraph-supervisor`, `langgraph-swarm`) implement these patterns ready-made, though their published examples still reference the deprecated `create_react_agent` — substitute `langchain.agents.create_agent` when following their tutorials.

---

### Further Reading

- **Official Subgraphs docs**: `docs.langchain.com/oss/python/langgraph/use-subgraphs` — the primary source for this chapter's mechanics: composition patterns, all three persistence modes, namespace isolation, and state inspection limits
- **Official Multi-agent overview**: `docs.langchain.com/oss/python/langchain/multi-agent` — the five-pattern taxonomy and the quantified performance comparison across one-shot, repeat, and multi-domain scenarios
- **Official Subagents docs**: `docs.langchain.com/oss/python/langchain/multi-agent/subagents` — the "supervisor" pattern in current terminology
- **Official Handoffs docs**: `docs.langchain.com/oss/python/langchain/multi-agent/handoffs` — the "swarm" pattern, both implementation approaches, and the context-engineering rules for valid tool-call pairing
- **`langgraph-supervisor-py`**: `github.com/langchain-ai/langgraph-supervisor-py` — the standalone supervisor package
- **`langgraph-swarm-py`**: `github.com/langchain-ai/langgraph-swarm-py` — the standalone swarm package
- **"Benchmarking Multi-Agent Architectures"**: `langchain.com/blog/benchmarking-multi-agent-architectures` — empirical comparison of supervisor vs. swarm-style implementations across task types

---

*End of Chapter 10. Chapter 11: Evaluation and Testing — Tracing, LangSmith, and Regression Suites for Agentic Systems.*
