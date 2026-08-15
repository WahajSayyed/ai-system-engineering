# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 9 — Tools and Tool-Calling: Structured Outputs, Parallel Execution, and Error Handling

> *"The fix isn't 'add a try/except.' The fix is classifying errors by who can fix them and routing each class to the right handler."*
> — Production LangGraph error-handling guidance, 2026

---

### What This Chapter Covers

Chapter 4 introduced the ReAct loop and used `ToolNode`/`tools_condition` as prebuilt pieces without examining what they actually do. This chapter opens both up completely, and adds the three capabilities every production tool-calling agent needs beyond the basic loop: structured output the rest of your system can parse without regex, parallel tool execution that doesn't silently corrupt state, and error handling that routes each failure to whoever can actually fix it.

By the end you will understand:

1. The `@tool` decorator in full: docstrings, type hints, `args_schema`, `return_direct`, and a real gotcha in how docstrings get parsed
2. `bind_tools()`: what it actually injects into a model call
3. `ToolNode` internals: parallel dispatch via `ThreadPoolExecutor`, state/store injection, and the newer `Send`-based per-call dispatch used by `create_agent`
4. A version-sensitive gotcha in `handle_tool_errors` worth checking against your installed version
5. Structured output: `with_structured_output()` on a bare model vs. `response_format` on a full agent, and the two strategies (`ToolStrategy`, `ProviderStrategy`) LangChain chooses between
6. `create_agent`: the current prebuilt agent factory (replacing the deprecated `create_react_agent`), and its middleware system
7. The four-class error classification framework: transient, LLM-recoverable, user-fixable, and unexpected errors — and the LangGraph primitive that matches each
8. `RetryPolicy`: every parameter, and the superstep transaction rule that makes per-node retry necessary
9. Testing tool-calling nodes in isolation
10. MARRS checkpoint: wiring real tools with structured critic output, retry policies, and classified error handling into the capstone project

---

### 9.1 The Tool-Calling Loop, Revisited

Chapter 4 built this loop from scratch:

```
agent (LLM reasons, may emit tool_calls) → tools (ToolNode executes them) → agent → ... → END
```

That chapter treated `ToolNode` and `tools_condition` as black boxes. This chapter opens them. Three questions this chapter answers that Chapter 4 didn't:

- What exactly does `@tool` turn a function into, and what can go wrong in that conversion?
- When the LLM requests three tool calls in one turn, what actually executes them, in what order, and what happens if one fails?
- How do you get the agent to return `{"summary": "...", "confidence": 0.87}` as a validated object instead of a string you have to parse?

---

### 9.2 Defining Tools: The `@tool` Decorator in Full

#### 9.2.1 Basic Usage

```python
from langchain_core.tools import tool
# Equivalently, current docs also show: from langchain.tools import tool
# Both import paths work — langchain.tools re-exports the langchain_core implementation.

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

The decorator does three things, mechanically:
1. **Wraps the function** in a `StructuredTool` object with a `.name`, `.description`, and `.args_schema`
2. **Uses the docstring** as the tool's `description` — this is literally what the LLM reads to decide whether and how to call the tool. Treat it as a prompt, not internal documentation.
3. **Derives a JSON schema from type hints** — every parameter needs a type annotation, because that's what builds the schema the LLM's tool-call arguments are validated against

**Type hints are not optional.** A parameter with no type hint gets no schema entry, which means the LLM has no idea what shape of value to send for it.

#### 9.2.2 The Docstring Parsing Gotcha

This is a real, currently open point of confusion (tracked as a LangChain GitHub issue). By default, `@tool` does **not** parse the `Args:` section of a Google-style docstring into per-parameter descriptions:

```python
@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the knowledge base for relevant documents.

    Args:
        query: Search query for the knowledge base
        top_k: Number of results to return (1-10)
    """
    return f"Results for: {query}"

print(search_knowledge_base.args_schema.model_json_schema())
# {
#   "properties": {
#     "query": {"title": "Query", "type": "string"},
#     "top_k": {"default": 5, "title": "Top K", "type": "integer"}
#   },
#   "required": ["query"], ...
# }
# Notice: NO "description" field on either parameter, despite the docstring
# spelling them out. The LLM only sees the parameter names and types.
```

**The fix — opt in to docstring parsing explicitly:**

```python
@tool(parse_docstring=True)
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the knowledge base for relevant documents.

    Args:
        query: Search query for the knowledge base
        top_k: Number of results to return (1-10)
    """
    return f"Results for: {query}"

# Now the schema includes:
# "query": {..., "description": "Search query for the knowledge base"}
# "top_k": {..., "description": "Number of results to return (1-10)"}
```

**Or bypass the ambiguity entirely with `args_schema` and Pydantic `Field`:**

```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Athens'")
    units: str = Field(default="celsius", description="celsius or fahrenheit")

@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """Fetch current weather for a given city."""
    return f"22°{units[0].upper()} and sunny in {city}"
```

**Practical guidance:** for any tool with more than one or two simple parameters, use `args_schema` with explicit `Field(description=...)` rather than relying on docstring parsing. It's unambiguous, validated by Pydantic, and doesn't depend on a flag you have to remember to set.

#### 9.2.3 `return_direct`: Skipping the Final LLM Pass

```python
@tool("product_search", return_direct=True)
def search_products(query: str) -> str:
    """Search for products by name or category."""
    return do_search(query)
```

`return_direct=True` means the tool's raw output becomes the graph's final answer without another LLM call to synthesize a response around it. Use this when the tool output is already the answer (e.g., a lookup that returns exactly what the user asked for) and an extra LLM pass would only add latency without adding value.

#### 9.2.4 Async Tools

```python
@tool
async def async_web_search(query: str) -> str:
    """Search the web asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.search.example/?q={query}")
        return response.text
```

`ToolNode` detects async tool functions automatically and awaits them correctly inside `.ainvoke()`/`.astream()`. One caveat carried over from Chapter 8: as of recent LangGraph versions, `get_stream_writer()` calls made from inside an **async** tool do not reliably surface via `stream_mode="custom"` — a known limitation, not something wrong with your code. Synchronous tools stream custom events correctly.

---

### 9.3 `bind_tools()`: What Actually Happens

```python
from langchain_openai import ChatOpenAI

tools = [search_database, get_weather]
llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
```

`bind_tools()` serializes each tool's `args_schema` into the JSON-schema format the model provider's API expects, and attaches that schema list to every subsequent call made through `llm_with_tools`. The model itself doesn't execute anything — it can only *emit* a structured `tool_calls` list in its response, naming which tool it wants and with what arguments. Execution is entirely your graph's responsibility, which is exactly what `ToolNode` is for.

```python
response = llm_with_tools.invoke("What's the weather in Athens?")
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Athens'}, 'id': 'call_abc123', 'type': 'tool_call'}]
```

Not every model supports tool calling — `bind_tools()` is a no-op error if the underlying provider doesn't implement it.

---

### 9.4 `ToolNode` Internals

#### 9.4.1 Dispatch and Parallel Execution

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([search_database, get_weather])
```

When the last `AIMessage` in state contains multiple `tool_calls`, `ToolNode` dispatches each to the matching function by name (matched against the tools list it was constructed with) and **executes them concurrently** — historically via a `ThreadPoolExecutor`, with the same effective behavior in current versions: if the LLM requests `get_weather("Athens")` and `search_database("hotels")` in the same turn, both run at the same time, not sequentially. Each result is wrapped in a `ToolMessage` carrying the original `tool_call_id`, so the LLM can correlate each result back to the call that produced it — this is also what lets `add_messages` deduplicate and order things correctly when they land back in state.

```python
def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
```

**Design implication:** because tool calls in the same turn run concurrently, tools that share mutable external state (a shared file, a non-thread-safe client) need their own locking. Tools that only read, or write to independent resources, need nothing extra.

#### 9.4.2 The `Send`-Based Dispatch Used by `create_agent`

Chapter 4 introduced the `Send` API for dynamic fan-out. `ToolNode`'s internals use exactly this mechanism when invoked from `create_agent` (Section 9.6): each tool call is distributed as its own `Send`-dispatched task (internally represented as a `ToolCallWithContext`), rather than looping over calls in a single node invocation. This is what allows a `create_agent`-built agent to correctly pause for human review (Chapter 7's HITL patterns) on an *individual* tool call while other parallel calls in the same turn proceed independently — the graph's execution model treats each tool call as its own addressable unit of work rather than an opaque batch.

You don't need to construct `Send` objects yourself to get this — it's internal to how `create_agent` and modern `ToolNode` usage compose. It's worth knowing about because it explains *why* per-tool-call human approval (Section 9.6.2) works correctly even with several tool calls in flight at once.

#### 9.4.3 State and Store Injection

Tools sometimes need more than the LLM-supplied arguments — they need to read graph state or the long-term store (Chapter 6) without the LLM having to know those exist.

```python
from typing import Annotated
from langgraph.prebuilt import InjectedState
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore

@tool
def get_user_preference(
    key: str,
    state: Annotated[dict, InjectedState],       # Injected — not part of the LLM-visible schema
    store: Annotated[BaseStore, InjectedStore],   # Injected — not part of the LLM-visible schema
) -> str:
    """Look up a stored user preference by key."""
    user_id = state.get("user_id", "anonymous")
    item = store.get(("users", user_id, "preferences"), key)
    return item.value["fact"] if item else "No preference found."
```

`Annotated[..., InjectedState]` and `Annotated[..., InjectedStore]` tell `ToolNode` to fill these parameters itself from the graph's runtime — they are stripped from the schema the LLM sees entirely, so the model never has to (and never can) supply them.

#### 9.4.4 `handle_tool_errors`: A Version-Sensitive Gotcha

```python
tool_node = ToolNode(tools, handle_tool_errors=True)
```

With `handle_tool_errors=True`, a tool exception is caught and converted into a `ToolMessage` containing the error text, which goes back to the LLM instead of crashing the graph. Without it, tool exceptions propagate and crash the run.

**Check your installed version before assuming a default.** A confirmed regression (filed against `langgraph` shortly after a `1.0.1` patch) showed `handle_tool_errors` behaving as though it were disabled by default in some versions, even when nothing in the calling code changed — a tool that always raised was allowed to crash the graph instead of being caught. Given this history, **always pass `handle_tool_errors` explicitly** rather than relying on whatever the default happens to be in your installed version:

```python
# Always explicit — don't rely on the default
tool_node = ToolNode(tools, handle_tool_errors=True)
```

You can also pass a custom formatter instead of `True`, to shape what the LLM sees when a tool fails:

```python
def format_tool_error(error: Exception) -> str:
    return (
        f"Tool failed with: {error}\n"
        "Review your arguments and try again. "
        "Check the tool's docstring for valid parameter values."
    )

tool_node = ToolNode(tools, handle_tool_errors=format_tool_error)
```

---

### 9.5 Structured Output

Free-text LLM output is fine for a chat reply. It is not fine when a downstream node needs `quality_score: float` or `structured_response.confidence` as an actual typed value rather than something regex'd out of prose.

#### 9.5.1 Model-Level: `with_structured_output()`

For a single, standalone LLM call with no tool loop involved:

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class Answer(BaseModel):
    summary: str
    confidence: float

model = ChatOpenAI(model="gpt-4.1")
structured_model = model.with_structured_output(Answer)

response = structured_model.invoke("Extract: John, [email protected], 555-1234")
# Answer(summary="...", confidence=0.9) — a validated Pydantic instance, not a string
```

#### 9.5.2 Agent-Level: `response_format`

When the structured output needs to come *after* a full tool-calling loop — the agent researches, calls tools, reasons, and only then needs to emit a validated final object — use `response_format` on the agent factory rather than trying to bolt `with_structured_output` onto an already-looping graph:

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class Answer(BaseModel):
    summary: str
    confidence: float

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search_database, get_weather],
    response_format=Answer,
)

result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
result["structured_response"]   # Answer(summary=..., confidence=...)
```

Two things happen mechanically that are worth understanding rather than treating as magic:
- **`ToolStrategy`**: LangChain wraps your schema as an additional callable "tool" the model can invoke to submit its final structured answer, with configurable retry if the model's first attempt doesn't validate. The retry mechanism creates a synthetic `ToolMessage` describing the validation error, letting the model fix its own mistake on the next turn — this is the same "let the LLM see the error and adjust" pattern used for tool-calling errors generally (Section 9.7).
- **`ProviderStrategy`**: for model providers with native structured-output support (e.g., an API-level `response_format` parameter), LangChain converts your schema directly into that provider's own mechanism instead of the tool-wrapping trick.

You do not choose between these yourself in the common case — passing a raw Pydantic class to `response_format` lets LangChain pick the best strategy for the bound model automatically. You can specify a strategy explicitly (`response_format=ToolStrategy(Answer)`) if you need to force one, e.g., to guarantee the retry-on-validation-failure behavior regardless of provider.

**A note on the older `response_format` behavior:** if you're reading code from before LangChain's v1 middleware system, you may see `response_format` used to request loosely-prompted JSON without a schema. That prompted-JSON mode is gone in the current system — `response_format` now requires an actual schema (Pydantic, `TypedDict`, or JSON Schema), which is what makes `result["structured_response"]` a validated object rather than "JSON that's usually well-formed."

---

### 9.6 `create_agent`: The Current Prebuilt Agent Factory

Chapter 4 built the ReAct loop by hand — `agent` node, `tools` node, conditional edge, back-edge. That remains the right way to *learn* the mechanics, and the right approach when you need custom routing the prebuilt doesn't support. For the common case — a standard tool-calling loop with well-understood extension points — `create_agent` from `langchain.agents` is the current, non-deprecated factory:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search_database, get_weather],
    system_prompt="You are a helpful research assistant.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Athens?"}]})
```

> **Migration note:** `create_agent` replaces `create_react_agent` (formerly imported from `langgraph.prebuilt`), which is deprecated as of LangGraph v1.0. The parameter renamed from `prompt=` to `system_prompt=`; everything else carries over directly. `ToolNode` and `tools_condition`, used when building the loop by hand as in Chapter 4, are unaffected by this deprecation and remain current.

#### 9.6.1 The Middleware System

`create_agent`'s actual extensibility model is middleware — interception points at specific stages of the agent's execution: before/after the whole agent run, before/after each model call, and wrapping each tool call. This is where you attach cross-cutting behavior without hand-writing new graph nodes for it.

```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def log_tool_calls(request, handler):
    """Middleware wrapping every tool call for logging."""
    print(f"[tool call] {request.tool_call['name']}({request.tool_call['args']})")
    result = handler(request)
    print(f"[tool result] {result}")
    return result

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search_database, get_weather],
    middleware=[log_tool_calls],
)
```

#### 9.6.2 `HumanInTheLoopMiddleware`: Per-Tool-Call Approval, the Prebuilt Way

Section 7.6.3 built a tool-call approval gate by hand with `interrupt()` inside a custom node. `create_agent` ships an equivalent as prebuilt middleware, wiring the same underlying `interrupt()`/`Command(resume=...)` mechanism from Chapter 7 without you writing the node yourself:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[send_email, charge_payment, search_database],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True, "charge_payment": True},
            # search_database is omitted — it executes without approval
        )
    ],
    checkpointer=checkpointer,  # Required — interrupt() needs persistence, as in Chapter 7
)
```

Because of the `Send`-based per-tool-call dispatch described in Section 9.4.2, this correctly pauses only for the flagged high-risk tools — if the model calls `search_database` and `send_email` in the same turn, `search_database` executes immediately while `send_email` pauses for approval, rather than the whole turn blocking on the riskiest call.

#### 9.6.3 When to Hand-Build vs. Use `create_agent`

| Situation | Use |
|---|---|
| Standard tool-calling loop, structured output, standard HITL gates | `create_agent` |
| Custom routing beyond "call tools or stop" (Chapter 4's routing patterns) | Hand-built `StateGraph` |
| Multi-agent architectures with custom handoff logic (`Command`, `Send` fan-out) | Hand-built `StateGraph` |
| You need to understand exactly what's happening, for teaching or debugging | Hand-built `StateGraph` (this is why Chapter 4 built it manually first) |

---

### 9.7 Production Error Handling: The Four-Class Framework

The single most important question in error handling is not "how do I catch this?" — it's **"who is capable of fixing this?"** A useful classification, grounded in production LangGraph deployments, sorts every failure mode into four classes, each paired with the LangGraph primitive built for it:

| Error Class | Who Fixes It | LangGraph Primitive | Example |
|---|---|---|---|
| Transient | The system, automatically | `RetryPolicy` | API 429, network timeout, DNS blip |
| LLM-recoverable | The LLM, given the error | `ToolNode(handle_tool_errors=...)` | Tool returned bad JSON, wrong tool chosen |
| User-fixable | The human | `interrupt()` | Missing required field, ambiguous input |
| Unexpected | The developer (later) | Let it bubble up | `TypeError`, schema mismatch, logic bug |

Getting the classification wrong is expensive in a specific, measurable way: retrying a user-fixable error burns several attempts and seconds before failing anyway with the identical error; interrupting for a transient error pages a human to click "retry" on something that would have resolved itself; swallowing an unexpected error in a blanket `try/except` hides a real bug behind a generic fallback message, and it stays hidden until someone notices degraded output quality weeks later.

#### 9.7.1 Transient Errors: `RetryPolicy`

Rate limits, network blips, transient DNS failures — these resolve themselves. Configure retry behavior at the node level; don't write bespoke retry code for it.

```python
from langgraph.types import RetryPolicy

# Aggressive retry for flaky external APIs — cheap to retry
api_retry = RetryPolicy(
    max_attempts=5,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=True,
)

# Conservative retry for LLM calls — expensive, don't over-retry
llm_retry = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=5.0,
    jitter=True,
)

builder.add_node("agent", agent_node, retry=llm_retry)
builder.add_node("tools", tool_node, retry=api_retry)
```

**`RetryPolicy` parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `max_attempts` | 3 | Total attempts including the first |
| `initial_interval` | 0.5 | Seconds before the first retry |
| `backoff_factor` | 2.0 | Multiplier per retry (exponential backoff) |
| `max_interval` | 128.0 | Cap on wait time between retries |
| `jitter` | `True` | Randomizes wait to avoid thundering-herd retries |
| `retry_on` | default exception set | Exception types, or a callable, to filter what qualifies for retry |

`retry_on` is where most people get it wrong — the default set covers common transient exceptions, but a custom exception type (e.g., an HTTP client's status-code-specific error) needs an explicit filter:

```python
from httpx import HTTPStatusError

def should_retry(error: Exception) -> bool:
    if isinstance(error, HTTPStatusError):
        return error.response.status_code in (429, 502, 503)
    return False

selective_retry = RetryPolicy(max_attempts=5, initial_interval=1.0, retry_on=should_retry)
```

**The superstep transaction rule — why per-node retry matters more than it looks:** LangGraph executes parallel branches within a superstep (Chapter 1's Pregel model). If *any* branch in that superstep raises, **none of the state updates from that superstep apply** — the checkpoint rolls back to before the superstep started. Individually successful nodes within the failed superstep are checkpointed via pending writes (Chapter 5) and won't re-execute on retry, but the *state update itself* is atomic across the whole superstep. A transient failure in one parallel branch can therefore block an unrelated branch's already-successful state update from landing. `RetryPolicy` on every node that touches the network is what prevents one flaky branch from poisoning its siblings.

#### 9.7.2 LLM-Recoverable Errors: `ToolNode` + Error-as-Data

The LLM picked the wrong tool, sent malformed arguments, or the tool returned something unparseable. The right fix is not retrying the identical call — it's letting the LLM *see* what went wrong and adjust:

```python
@tool
def extract_clause(text: str, clause_type: str) -> dict:
    """Extract a specific clause from contract text.

    Args:
        text: The contract text to search.
        clause_type: One of 'termination', 'liability', 'indemnification', 'payment'.
    """
    valid_types = {"termination", "liability", "indemnification", "payment"}
    if clause_type not in valid_types:
        raise ValueError(f"Invalid clause_type '{clause_type}'. Must be one of: {valid_types}")
    return {"clause_type": clause_type, "text": f"Extracted {clause_type} clause.", "confidence": 0.92}

tool_node = ToolNode([extract_clause], handle_tool_errors=True)
```

When `extract_clause` raises because the LLM sent `clause_type="termnation"` (typo), `handle_tool_errors=True` converts that exception into a `ToolMessage` describing the error, which flows back into the message history. On its next turn, the LLM reads the error text and — because the tool's docstring names the exact valid values — usually self-corrects. This is the same error-as-data principle from `response_format`'s validation-retry loop in Section 9.5.2: don't hide the failure, feed it back to whoever can act on it.

**A design principle:** store errors as data in state, not just as exceptions that happen and vanish. If your state schema tracks `validation_errors: list[str]` or similar, downstream nodes and the LLM itself get visibility into what's gone wrong across the whole run, not just the most recent attempt.

#### 9.7.3 User-Fixable Errors: `interrupt()`

Some failures genuinely cannot be resolved by retrying or by the LLM trying again — the input itself is incomplete or ambiguous. This is Chapter 7's territory, applied specifically to error recovery:

```python
from langgraph.types import interrupt

def validate_node(state: PipelineState) -> dict:
    clauses = state.get("extracted_clauses", [])
    errors = []

    required_types = {"termination", "payment"}
    found_types = {c["clause_type"] for c in clauses}
    missing = required_types - found_types
    if missing:
        errors.append(f"Missing required clause types: {missing}")

    if errors:
        human_input = interrupt({
            "type": "validation_errors",
            "errors": errors,
            "message": "Document validation failed. Please review and provide corrections.",
        })
        return {"extracted_clauses": human_input.get("corrected_clauses", clauses)}

    return {}
```

**The failure mode to watch for (from Chapter 5 and 7, worth repeating here because it's an easy trap in error-handling code specifically):** `interrupt()` requires a checkpointer. Adding validation-triggered interrupts, testing locally with a working checkpointer, then deploying without one (or with `InMemorySaver` behind a load-balanced multi-instance deployment) means the interrupt pauses correctly, the human submits corrections, and the graph has no memory of where it was — a different server instance picks up the resume with cold state. This is a silent failure: everything works in every test until the exact moment it needs to resume in production.

#### 9.7.4 Unexpected Errors: Let Them Bubble

`TypeError`, `KeyError`, a schema mismatch, a genuine logic bug. **Do not catch these. Do not retry them. Do not interrupt for them.** A retry wastes time on something that will never self-resolve by repetition. An interrupt pages a human to look at what should be a bug-tracker ticket, not a "click approve" decision.

```python
# WRONG — swallows everything, including real bugs
def bad_node(state: State) -> dict:
    try:
        return risky_operation(state)
    except Exception:
        return {"error": "Something went wrong."}
    # Every TypeError, every KeyError, every schema mismatch now disappears
    # into a generic message. The trace shows the node "succeeded" — it
    # returned a value. The bug lives in production until output quality
    # degrades enough for someone to notice.

# CORRECT — only catch what you know how to handle; let the rest surface
def good_node(state: State) -> dict:
    return risky_operation(state)
    # An unexpected exception here crashes the run, appears in your tracing
    # tool with full state context, and gets fixed as an actual bug.
```

The only appropriate action for this class is making failures *observable* — attaching tracing metadata so that when something does crash, you get full context (state, node, inputs) rather than a bare stack trace.

#### 9.7.5 Assembling the Four Classes in One Graph

```python
def should_continue(state: PipelineState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "validate"

builder = StateGraph(PipelineState)

# Transient errors: RetryPolicy, tuned per node's actual failure profile
builder.add_node("agent", agent_node, retry=llm_retry)
builder.add_node("tools", tool_node, retry=api_retry)          # + handle_tool_errors=True inside tool_node

# LLM-recoverable errors: handled inside the tool_node itself (Section 9.7.2)
builder.add_node("validate", validate_node)                    # User-fixable errors: interrupt() inside
builder.add_node("summarize", summarize_node, retry=llm_retry)

# NOTE: 'validate' deliberately has NO retry policy — a missing clause
# won't appear no matter how many times you retry extracting it. That's
# a user-fixable problem, not a transient one, and gets interrupt() instead.

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "validate": "validate"})
builder.add_edge("tools", "agent")
builder.add_edge("validate", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile(checkpointer=checkpointer)   # Required for validate_node's interrupt()
```

Notice the *absence* of a retry policy on `validate` is a deliberate decision, not an oversight — it's the tell that distinguishes "this node's errors are transient" from "this node's errors need a human."

#### 9.7.6 Common Production Failures From Misclassification

- **Retrying a user-fixable error.** Three attempts and several seconds burned re-extracting a clause that genuinely isn't in the document, before failing with the identical error a fourth time. Fix: classify before choosing a handler — if content is missing, `interrupt()` immediately rather than retrying.
- **An error-recovery loop that never converges.** The LLM sends malformed tool arguments, the error goes back, the LLM tries again with different-but-still-wrong arguments, indefinitely — until the recursion limit (Chapter 4's `RemainingSteps`) is hit, having burned real API cost along the way. Fix: track an explicit `retry_count` in state; after a small fixed number of LLM-recovery attempts, escalate to `interrupt()` or fail with a clear message rather than looping indefinitely.
- **Superstep transaction surprise** (Section 9.7.1): expecting one successful parallel branch's state update to "count" even though a sibling branch failed in the same superstep. It doesn't — the whole superstep's update is atomic. `RetryPolicy` on every network-touching node is the mitigation, not a workaround after the fact.

---

### 9.8 Testing Tool-Calling Nodes

```python
import pytest
from unittest.mock import patch

def test_tool_schema_has_descriptions():
    """Catches the docstring-parsing gotcha before it ships."""
    schema = get_weather.args_schema.model_json_schema()
    assert "description" in schema["properties"]["city"], (
        "Tool parameter is missing a description — the LLM will see "
        "only a name and type, not what the parameter means."
    )

def test_tool_raises_on_invalid_input():
    with pytest.raises(ValueError, match="Invalid clause_type"):
        extract_clause.invoke({"text": "...", "clause_type": "not_a_real_type"})

def test_tool_node_catches_errors_as_messages():
    """handle_tool_errors=True should produce a ToolMessage, not a crash."""
    tool_node = ToolNode([extract_clause], handle_tool_errors=True)
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{
                "name": "extract_clause",
                "args": {"text": "...", "clause_type": "bad_type"},
                "id": "call_1",
            }])
        ]
    }
    result = tool_node.invoke(state)
    tool_message = result["messages"][0]
    assert "Invalid clause_type" in tool_message.content
    assert tool_message.tool_call_id == "call_1"

def test_parallel_tool_calls_both_execute():
    """Verify ToolNode actually runs multiple calls, not just the first."""
    tool_node = ToolNode([get_weather, search_database])
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[
                {"name": "get_weather", "args": {"city": "Athens"}, "id": "call_1"},
                {"name": "search_database", "args": {"query": "hotels"}, "id": "call_2"},
            ])
        ]
    }
    result = tool_node.invoke(state)
    assert len(result["messages"]) == 2
    ids = {m.tool_call_id for m in result["messages"]}
    assert ids == {"call_1", "call_2"}

def test_structured_output_validates():
    """response_format should reject malformed structured responses upstream."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Answer(summary="ok", confidence="not_a_float")   # Wrong type
```

---

### 9.9 MARRS Checkpoint: Real Tools, Structured Critic Output, Classified Errors

We now replace MARRS's placeholder search/critic functions with real tools, wire in structured output for the critic's evaluation, and apply the four-class error framework across the pipeline.

```python
import operator
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy, Command, interrupt


# ─────────────────────────────────────────────────────────────────────────────
# REAL TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information on a topic.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (1-10).
    """
    if not query.strip():
        raise ValueError("Search query cannot be empty. Provide a specific topic to search.")
    # Placeholder for a real search API call (Tavily, SerpAPI, etc.)
    return f"[{max_results} results for '{query}']: relevant findings about {query}..."


@tool
def fetch_academic_paper(arxiv_id: str) -> str:
    """Fetch the abstract of an academic paper by its arXiv ID.

    Args:
        arxiv_id: The arXiv identifier, e.g. '2301.00234'.
    """
    import re
    if not re.match(r"^\d{4}\.\d{4,5}$", arxiv_id):
        raise ValueError(f"'{arxiv_id}' is not a valid arXiv ID format (expected e.g. '2301.00234').")
    return f"[Abstract for arXiv:{arxiv_id}]: This paper investigates..."


research_tools = [web_search, fetch_academic_paper]

# handle_tool_errors set EXPLICITLY — never rely on the version default (Section 9.4.4)
research_tool_node = ToolNode(research_tools, handle_tool_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED CRITIC OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class CriticEvaluation(BaseModel):
    """Structured evaluation of a research draft."""
    quality_score: float = Field(description="Overall quality from 0.0 to 1.0")
    strengths: list[str] = Field(description="What the draft does well")
    weaknesses: list[str] = Field(description="Specific issues that need fixing")
    recommendation: Literal["approve", "revise"] = Field(
        description="Whether the draft should proceed to human review or be revised"
    )

critic_llm = llm.with_structured_output(CriticEvaluation)   # Model-level structured output (Section 9.5.1)


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class MARRSStateV4(TypedDict, total=False):
    topic: str
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    findings: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]
    draft: str
    critique: str
    quality_score: float
    revision_count: int
    validation_errors: list[str]
    final_report: str
    status: str


# ─────────────────────────────────────────────────────────────────────────────
# NODES WITH CLASSIFIED ERROR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def researcher_agent_node(state: MARRSStateV4) -> dict:
    """
    LLM decides which research tools to call. Transient failures inside the
    tools (network errors) are handled by RetryPolicy on the 'tools' node.
    Malformed tool arguments are handled by handle_tool_errors inside ToolNode.
    """
    llm_with_tools = llm.bind_tools(research_tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def critic_node_structured(state: MARRSStateV4) -> Command[Literal["writer", "human_review"]]:
    """
    Critic using structured output instead of free-text parsing.
    quality_score, strengths, weaknesses all arrive as validated fields —
    no regex, no "hope the LLM formatted it correctly."
    """
    draft = state.get("draft", "")
    revisions = state.get("revision_count", 0)

    evaluation = critic_llm.invoke(
        f"Evaluate this research draft on {state.get('topic', '')}:\n\n{draft}"
    )
    # evaluation is a validated CriticEvaluation instance — attribute access, not parsing
    critique_text = (
        "Strengths: " + "; ".join(evaluation.strengths)
        + "\nWeaknesses: " + "; ".join(evaluation.weaknesses)
    )

    new_revision_count = revisions + 1
    if evaluation.recommendation == "approve" or new_revision_count >= 3:
        destination = "human_review"
    else:
        destination = "writer"

    return Command(
        update={
            "critique": critique_text,
            "quality_score": evaluation.quality_score,
            "revision_count": new_revision_count,
        },
        goto=destination,
    )


def validate_sources_node(state: MARRSStateV4) -> dict:
    """
    User-fixable error class: if the research produced too few sources,
    that's not something a retry fixes — it needs a human decision about
    whether to proceed with thin evidence or provide additional direction.
    """
    sources = state.get("sources", [])
    MIN_SOURCES = 2

    if len(sources) < MIN_SOURCES:
        human_input = interrupt({
            "type": "insufficient_sources",
            "message": f"Only found {len(sources)} source(s) (minimum {MIN_SOURCES}). "
                       "Proceed anyway, or provide additional search direction?",
            "current_sources": sources,
        })
        if isinstance(human_input, dict) and human_input.get("additional_query"):
            # Human wants another search attempt with their suggested angle
            return {"validation_errors": [], "plan": human_input["additional_query"]}
        # Human said proceed anyway
        return {"validation_errors": []}

    return {"validation_errors": []}


def finalize_node(state: MARRSStateV4) -> dict:
    """No retry needed — this only assembles already-validated state."""
    return {
        "final_report": f"# {state.get('topic', 'Report')}\n\n{state.get('draft', '')}",
        "status": "complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
# RETRY POLICIES, MATCHED TO EACH NODE'S ACTUAL ERROR PROFILE
# ─────────────────────────────────────────────────────────────────────────────

llm_retry = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0, jitter=True)
api_retry = RetryPolicy(max_attempts=5, initial_interval=1.0, backoff_factor=2.0, jitter=True)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_marrs_with_tools():
    builder = StateGraph(MARRSStateV4)

    builder.add_node("researcher", researcher_agent_node, retry=llm_retry)
    builder.add_node("tools", research_tool_node, retry=api_retry)   # Transient errors: RetryPolicy
                                                                       # LLM-recoverable errors: handle_tool_errors
    builder.add_node("validate_sources", validate_sources_node)      # User-fixable: interrupt() inside, NO retry
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node_structured, retry=llm_retry)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "researcher")
    builder.add_conditional_edges("researcher", tools_condition, {"tools": "tools", "__end__": "validate_sources"})
    builder.add_edge("tools", "researcher")
    builder.add_edge("validate_sources", "writer")
    builder.add_edge("writer", "critic")
    # critic routes via Command to "writer" or "human_review"
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)   # Required — validate_sources uses interrupt()

marrs_with_tools = build_marrs_with_tools()
```

---

### 9.10 Chapter Summary

**The `@tool` decorator** wraps a function into a `StructuredTool`, using the docstring as the LLM-facing description and type hints as the argument schema. Docstring `Args:` sections are **not** parsed into per-parameter descriptions by default — either pass `parse_docstring=True` or, for anything beyond trivial tools, use `args_schema` with Pydantic `Field(description=...)` for unambiguous, validated schemas.

**`bind_tools()`** attaches tool schemas to a model so it can *emit* structured tool-call requests; it never executes anything itself. Execution is `ToolNode`'s job.

**`ToolNode`** dispatches multiple tool calls from one turn concurrently and wraps each result in a `ToolMessage` carrying the matching `tool_call_id`. Modern `create_agent`-based usage dispatches each call as its own `Send`-based unit, which is what allows per-tool-call human approval to work correctly alongside other calls proceeding in parallel. `Annotated[..., InjectedState]`/`InjectedStore` let tools read graph state or the long-term store without exposing those as LLM-visible parameters. Always pass `handle_tool_errors` explicitly rather than trusting the version default — this has changed unexpectedly before.

**Structured output** comes in two forms: `model.with_structured_output(Schema)` for a single standalone call, and `response_format=Schema` on `create_agent` when structured output needs to follow a full tool-calling loop. The latter uses either a tool-wrapping strategy (`ToolStrategy`, with automatic retry-on-validation-failure) or the provider's native structured-output mechanism (`ProviderStrategy`), chosen automatically unless you specify one explicitly.

**`create_agent`** (from `langchain.agents`) is the current prebuilt agent factory, replacing the deprecated `create_react_agent`. Its middleware system (`wrap_tool_call`, `HumanInTheLoopMiddleware`, and others) is the extension point for cross-cutting behavior like logging and per-tool approval gates, without hand-writing new graph nodes for each.

**The four-class error framework** is the chapter's central production idea: classify every failure by *who can fix it* — transient errors get `RetryPolicy`, LLM-recoverable errors get caught and fed back as data via `handle_tool_errors`, user-fixable errors get `interrupt()`, and unexpected errors are left to crash loudly and visibly rather than being swallowed. The superstep transaction rule — a failure anywhere in a superstep rolls back the whole superstep's state update — is why `RetryPolicy` belongs on every node that touches the network, not just the ones you've personally seen fail.

---

### Further Reading

- **Official Tools docs**: `docs.langchain.com/oss/python/langchain/tools` — `@tool`, `args_schema`, `parse_docstring`, and injected arguments
- **`langchain_core.tools` reference**: `reference.langchain.com/python/langchain-core/tools` — the full `StructuredTool` / `BaseTool` API
- **Official Agents docs**: `docs.langchain.com/oss/python/langchain/agents` — `create_agent`, `response_format`, structured output strategies
- **`create_agent` reference**: `reference.langchain.com/python/langchain/agents/factory/create_agent` — every parameter, including `response_format` strategy types
- **Middleware reference**: `reference.langchain.com/python/langchain/agents/middleware` — `HumanInTheLoopMiddleware`, `wrap_tool_call`, and the full hook system
- **`ToolNode` reference**: `reference.langchain.com/python/langgraph.prebuilt/tool_node` — parallel dispatch, `Send`-based `ToolCallWithContext`, `InvalidToolCall` handling
- **GitHub issue #34292**: `github.com/langchain-ai/langchain/issues/34292` — the `parse_docstring` default behavior and parameter description gap
- **GitHub issue #6486**: `github.com/langchain-ai/langgraph/issues/6486` — the `handle_tool_errors` default-behavior regression; check this against your installed version
- **"LangGraph Error Handling Patterns for Production AI Agents"**: the four-class error classification matrix, `RetryPolicy` parameter reference, and the superstep transaction rule, with a full worked contract-processing pipeline example

---

*End of Chapter 9. Chapter 10: Subgraphs and Multi-Agent Architectures — Supervisor, Swarm, and Hierarchical Patterns.*
