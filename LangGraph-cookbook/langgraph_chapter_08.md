# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 8 — Streaming: Real-Time Output, Token-Level Feedback, and Progress Monitoring

> *"LangGraph is built with first class support for streaming... Pass one or more stream modes to control what data you receive."*
> — LangGraph official documentation

> **Chapter revision note:** Since this chapter was first written, LangGraph shipped a new `version="v3"` event-streaming API as a beta, opt-in addition. The official position is explicit: *"v1 and v2 are unchanged."* Everything below using `stream_mode` and `astream_events(version="v2")` remains current and is still the right default for most use cases. Section 8.5.2 introduces v3 as a forward-looking option for anyone hitting the specific pain points it addresses.

---

### What This Chapter Covers

Every graph in this curriculum so far has been called with `.invoke()` — run to completion, return the final state. For a nine-second research pipeline, that means nine seconds of a blank screen before the user sees anything. This chapter replaces that with streaming: showing the user what the agent is doing, token by token and step by step, while it is doing it.

By the end you will understand:

1. Why streaming is a perception fix, not a performance fix — and when to reach for it
2. The five stream modes (`values`, `updates`, `messages`, `custom`, `debug`) and exactly what each yields
3. `get_stream_writer()`: emitting arbitrary progress data from inside a node
4. Combining multiple stream modes in one call and how the output shape changes
5. Filtering token streams by node (`langgraph_node`) and by tag
6. `astream_events()`: the fine-grained event API, when to reach for it instead of `stream_mode`
7. Streaming through subgraphs: the namespace tuple and the `subgraphs=True` flag
8. Known limitations: async tools and `custom` mode, subgraph token capture
9. Production wiring: FastAPI + Server-Sent Events, backpressure, reverse proxy buffering
10. Testing streaming code: verifying stream completeness against `invoke()`
11. MARRS checkpoint: full streaming integration — node progress, custom status events, and token-level synthesis output

---

### 8.1 Why Streaming Is a Perception Fix, Not a Performance Fix

A useful way to frame this chapter: streaming does not make your graph faster. The same computation, the same tokens, the same tool calls all still happen. What changes is *when the user sees output*.

Without streaming: the user submits a query, watches a blank screen or spinner for the full duration of the run, then receives the complete answer all at once. Empirically, users perceive this wait as much longer than it is, and abandonment climbs sharply once the wait crosses a few seconds.

With streaming: the first token, or the first progress update, appears within a few hundred milliseconds. The total wall-clock time to the final answer is unchanged, but the user's *subjective* experience is that the system is already working, thinking, responding. This is a well-documented perception effect specific to token-by-token and step-by-step output — not a claim about actual latency.

**When to use streaming:**
- Any user-facing agent interaction expected to take more than roughly two seconds
- Multi-step agents where progress indicators (searching, writing, reviewing) reduce perceived wait
- Chat interfaces where token-by-token display is the expected interaction pattern

**When to skip it:**
- Background jobs with no user waiting on the result
- Latency already under a second — streaming adds complexity for no perceptible gain
- Output is structured data (a JSON object, a table) rather than natural language prose, where a "typing" effect makes little sense

---

### 8.2 The Five Stream Modes

LangGraph's `.stream()` (sync) and `.astream()` (async) methods accept a `stream_mode` parameter. There are five modes, each yielding a different view of graph execution.

```python
# Single mode
for chunk in graph.stream(input, config, stream_mode="updates"):
    ...

# Multiple modes — see Section 8.4 for the output shape when combining
for mode, chunk in graph.stream(input, config, stream_mode=["updates", "custom"]):
    ...
```

#### 8.2.1 `values`: Full State Snapshot After Each Step

Yields the complete graph state after every superstep.

```python
for state in graph.stream({"topic": "AI"}, config, stream_mode="values"):
    print(state)
# {'topic': 'AI'}                                    ← after START
# {'topic': 'AI', 'plan': '...'}                      ← after planner
# {'topic': 'AI', 'plan': '...', 'findings': [...]}   ← after researcher
# ...
```

**Characteristic:** Each yield contains the *entire* state, not just what changed. If your state carries large accumulated lists (search results, message history), every single yield repeats all of it — the payload is large and grows across the run. This is useful when a downstream consumer needs the complete picture at every step (e.g., feeding a separate orchestrator that expects full snapshots), but it is expensive for long-running graphs with big state.

#### 8.2.2 `updates`: Only What Changed

Yields only the incremental state changes produced by each node, keyed by node name.

```python
for delta in graph.stream({"topic": "AI"}, config, stream_mode="updates"):
    print(delta)
# {'planner': {'plan': '...', 'search_queries': [...]}}
# {'researcher': {'findings': ['...'], 'sources': ['...']}}
# {'writer': {'draft': '...'}}
# {'critic': {'quality_score': 0.85, 'critique': '...'}}
```

**Characteristic:** This is the natural choice for a "Node X completed" progress indicator. The dict has exactly one key — the node name — mapping to that node's return value. This is far more bandwidth-efficient than `values` mode for graphs with large accumulated state, and it directly answers "what just happened?"

#### 8.2.3 `messages`: Token-by-Token LLM Output

Yields individual LLM tokens as they are generated, along with metadata about which node and model produced them. This is only meaningful for graphs whose nodes call chat models.

```python
for msg_chunk, metadata in graph.stream({"topic": "AI"}, config, stream_mode="messages"):
    if msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)
```

Each yield is a `(message_chunk, metadata)` tuple:
- `message_chunk`: an `AIMessageChunk` with a `.content` attribute (may be empty for tool-call argument deltas)
- `metadata`: a dict including `langgraph_node` (which node produced this token) and `tags` (any tags attached to the model)

**Filtering by node** — critical in multi-node graphs where you only want the final answer streamed, not intermediate reasoning:

```python
for msg_chunk, metadata in graph.stream(input, config, stream_mode="messages"):
    if metadata["langgraph_node"] == "writer" and msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)
    # Tokens from "planner" or "critic" nodes are silently skipped
```

**Filtering by tag** — useful when a single node calls multiple differently-configured models:

```python
from langchain.chat_models import init_chat_model

planning_llm = init_chat_model("openai:gpt-4o-mini", tags=["planning"])
writing_llm = init_chat_model("openai:gpt-4o", tags=["writing"])

for msg_chunk, metadata in graph.stream(input, config, stream_mode="messages"):
    if metadata.get("tags") == ["writing"] and msg_chunk.content:
        print(msg_chunk.content, end="|", flush=True)
```

**Important note (from the official docs):** `stream_mode="messages"` assumes your graph state has a `messages` key. This is a specialized mode for message-based state (the common `MessagesState` pattern), not a general-purpose token stream for arbitrary state shapes.

#### 8.2.4 `custom`: Arbitrary Progress Data From Inside Nodes

Yields anything you explicitly emit from within a node using `get_stream_writer()`. This is the mechanism for surfacing progress messages that are not part of the graph's state schema at all — pure UI signaling.

```python
from langgraph.config import get_stream_writer

def researcher_node(state: State) -> dict:
    writer = get_stream_writer()
    
    writer({"status": "Searching academic sources..."})
    academic_results = search_academic(state["topic"])
    
    writer({"status": "Searching web sources...", "progress": 0.5})
    web_results = search_web(state["topic"])
    
    writer({"status": "Consolidating findings...", "progress": 0.9})
    
    return {"findings": academic_results + web_results}
```

```python
for chunk in graph.stream(input, config, stream_mode="custom"):
    print(f"Status: {chunk['status']}")
# Status: Searching academic sources...
# Status: Searching web sources...
# Status: Consolidating findings...
```

**Key property:** `get_stream_writer()` requires no special node signature changes — it is called from inside the node body, and works whether or not you have also declared a `store` or `config` parameter. Any JSON-serializable value can be passed to the writer.

**Known limitation (GitHub issue #6447, filed November 2025):** As of LangGraph versions tested at the time of writing, **async tools do not support `custom` events streaming via `get_stream_writer()`** — the emitted message is silently dropped. This bug is specific to async tool functions (`@tool async def ...`); synchronous tools stream custom messages correctly. If you rely on `custom` mode inside an async tool and events aren't appearing, this known issue is the likely cause — check the issue tracker for the current fix status before assuming your code is wrong.

#### 8.2.5 `debug`: Maximum Verbosity for Development

Emits the same information as all other modes combined, formatted for console inspection. Per the official reference documentation, `debug` accepts the same values as the other stream modes but is intended purely for debugging — it does not affect the output of the graph in any way and should not be used in production UI code.

```python
for chunk in graph.stream(input, config, stream_mode="debug"):
    print(chunk)  # Verbose internal execution detail
```

---

### 8.3 The Async Path: `.astream()`

Every stream mode above works identically with `.astream()` for async graphs. The choice between `.stream()` and `.astream()` is about your runtime, not about which stream modes are available:

```python
# Sync — scripts, notebooks, simple CLI tools
for chunk in graph.stream(input, config, stream_mode="messages"):
    ...

# Async — web servers (FastAPI, etc.), concurrent request handling
async for chunk in graph.astream(input, config, stream_mode="messages"):
    ...
```

**Guidance:** For scripts and Jupyter notebooks, `.stream()` is the simpler path — no async machinery needed. For web servers, use `.astream()` — it integrates with async frameworks like FastAPI and allows the event loop to serve many concurrent users without blocking on any single graph's execution.

---

### 8.4 Combining Multiple Stream Modes

Production systems almost always need more than one mode simultaneously: node-level progress (`updates`), custom status text (`custom`), and token-level output (`messages`) all at once.

```python
async for mode, chunk in graph.astream(
    {"topic": "AI safety"},
    config,
    stream_mode=["updates", "custom", "messages"],
):
    if mode == "messages":
        message_chunk, metadata = chunk
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    elif mode == "custom":
        print(f"\n[status] {chunk}")
    elif mode == "updates":
        for node_name, state_delta in chunk.items():
            print(f"\n[node complete] {node_name}: {list(state_delta.keys())}")
```

**The critical shape change:** When you pass a *single* string to `stream_mode`, each yielded item is just the chunk (a dict for `updates`, a `(chunk, metadata)` tuple for `messages`, etc.). When you pass a *list* of modes, each yielded item becomes a `(mode_name, chunk)` tuple — you must unpack the mode name first, then branch on it. This is the single most common bug when adding a second stream mode to existing code: forgetting that the output shape has changed from `chunk` to `(mode, chunk)`.

```python
# WRONG — worked with single mode, breaks silently with multiple
for chunk in graph.stream(input, config, stream_mode=["updates", "custom"]):
    print(chunk["some_key"])  # chunk is now a tuple, not a dict!

# CORRECT
for mode, chunk in graph.stream(input, config, stream_mode=["updates", "custom"]):
    if mode == "updates":
        print(chunk["some_key"])
```

---

### 8.5 `astream_events()`: Fine-Grained Lifecycle Events

`stream_mode` (Sections 8.2–8.4) covers the vast majority of production needs. `astream_events()` is a lower-level API for cases requiring deep observability — full lifecycle hooks for every chain, tool, and model invocation.

```python
async for event in graph.astream_events(input, version="v2"):
    print(event)
```

Each event is a dict with a standard schema (from the official `langchain_core` reference):

```python
{
    "event": "on_chat_model_stream",   # Event name: on_[runnable_type]_(start|stream|end)
    "data": {"chunk": AIMessageChunk(content="Hello")},
    "name": "ChatOpenAI",              # Name of the Runnable that emitted this event
    "run_id": "cdc9524f-...",          # Unique ID for this specific execution
    "tags": ["graph_node:writer"],
    "metadata": {"langgraph_node": "writer", ...},
    "parent_ids": [...],               # Ancestor run IDs — traces nesting depth
}
```

**Common event types you'll filter on:**

| Event | Fires When |
|---|---|
| `on_chain_start` / `on_chain_end` | A graph, node, or chain begins/completes |
| `on_chat_model_start` | An LLM call begins |
| `on_chat_model_stream` | An LLM emits a token chunk |
| `on_chat_model_end` | An LLM call completes (full message available) |
| `on_tool_start` / `on_tool_end` | A tool invocation begins/completes |

**Token streaming with `astream_events`:**

```python
async for event in graph.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        token = chunk.content
        if token:
            yield token  # push to SSE, websocket, etc.
```

**Filtering to a specific node in a multi-node graph:**

```python
async for event in graph.astream_events(input, version="v2"):
    node_name = event.get("metadata", {}).get("langgraph_node", "")
    if event["event"] == "on_chat_model_stream" and node_name == "writer":
        token = event["data"]["chunk"].content
        if token:
            yield token
```

**Always pass `version="v2"`.** This is required — the v2 schema is the current standard with consistent event structure across all Runnable types. Custom events (from `get_stream_writer` or `dispatch_custom_event`) are only surfaced in v2.

#### 8.5.1 When to Reach for `astream_events` Instead of `stream_mode`

| Need | Use |
|---|---|
| Node progress + token streaming + custom status | `stream_mode=["updates", "custom", "messages"]` |
| Per-tool timing, tool-call argument deltas | `astream_events()` |
| Building a custom tracing dashboard | `astream_events()` |
| Filtering by exact model instance (`include_names`) | `astream_events()` |
| Subgraph nesting depth via full tag path | `astream_events()` (tags show full path; `messages` mode only shows the outer node) |
| Standard chat UI, simplest implementation | `stream_mode="messages"` |

The general guidance from production practitioners: skip `astream_events` unless you specifically need it. Plain `messages` mode delivers the token-by-token "typing" effect through both `.stream()` and `.astream()` with substantially less code. Reserve `astream_events` for lifecycle hooks, per-model filtering, or subgraph depth tracing.

`astream_events` also accepts filter parameters directly:

```python
async for event in graph.astream_events(
    input,
    version="v2",
    include_names=["writer_llm"],     # Only events from Runnables named this
    include_types=["chat_model"],     # Only chat model events
    include_tags=["synthesis"],       # Only events tagged this way
):
    ...
```

#### 8.5.2 Looking Ahead: The `version="v3"` Event Streaming API (Beta)

Everything in Sections 8.2–8.5.1 — `stream_mode` lists and `astream_events(version="v2")` — remains the current, recommended default. LangChain's own changelog states plainly that "v1 and v2 are unchanged." This section is a forward-pointer for a newer, opt-in addition, not a replacement for anything above.

**The problem v3 targets:** as graphs grow — multiple LLM calls, subgraphs, tool calls, reasoning models with distinct reasoning/answer phases — the `if event["event"] == ...` branching from Section 8.5 tends to grow into a long, fragile chain of conditionals, along with a hand-rolled accumulator for reassembling tokens across node boundaries. This is exactly the pattern shown in Section 8.5's filtering examples, and it is manageable for a two-node graph but gets noisy fast with more structure.

**What v3 changes:** instead of one undifferentiated event stream you branch on, `version="v3"` gives you a content-block-centric protocol with typed, per-channel projections — separate, purpose-built views for values, messages, lifecycle events, and subgraphs, rather than one firehose:

```python
stream = graph.stream_events(input_data, config=config, version="v3")

# Typed projections, each independently iterable:
for message in stream.messages:        # Token-by-token chat output, pre-separated from other events
    for token in message.text:
        print(token, end="", flush=True)

final_state = stream.output            # The completed (or paused) state
```

This is the same `stream.messages` / `stream.output` / `stream.interrupted` / `stream.interrupts` object introduced in Chapter 7 (Section 7.3.6) for driving HITL loops — v3 is one unified interface covering both token streaming and interrupt handling, rather than two separate mechanisms (`stream_mode` for tokens, `"__interrupt__"` dict-key checking for pauses).

**Current status and guidance:**
- `version="v3"` is beta and "may change" per the official reference — it is currently supported specifically on `BaseChatModel` and `langgraph.CompiledGraph`, not on arbitrary Runnables
- Custom events (`get_stream_writer()`) are surfaced in both v2 and v3
- For this curriculum, and for most production code today, `stream_mode` lists (Section 8.4) and `astream_events(version="v2")` (Section 8.5) remain the right teaching foundation and the right production default — they are stable, widely documented, and not going anywhere
- Revisit v3 once it exits beta, or sooner if your graph's event-handling code has genuinely become the tangle of conditionals this section describes

---

### 8.6 Streaming Through Subgraphs

When a graph contains subgraphs (covered fully in a later chapter on multi-agent architectures), streaming requires an additional flag to see inside them.

```python
# Without subgraphs=True: only top-level graph events are streamed
for chunk in graph.stream(input, config, stream_mode="updates"):
    ...

# With subgraphs=True: events from inside subgraphs are also streamed
for chunk in graph.stream(input, config, stream_mode="updates", subgraphs=True):
    ...
```

With `subgraphs=True`, each yielded item becomes a `(namespace, data)` tuple (or `(namespace, mode, data)` if `stream_mode` is a list). The `namespace` is a tuple representing the node path — for example, `('researcher_subgraph:abc123',)` indicates the event originated from inside the `researcher_subgraph` node's internal subgraph.

```python
for namespace, chunk in graph.stream(input, config, stream_mode="updates", subgraphs=True):
    if namespace:
        print(f"[subgraph {namespace}] {chunk}")
    else:
        print(f"[top level] {chunk}")
```

**A known limitation (GitHub issue #4718):** capturing LLM tokens streamed from *inside* a subgraph node and re-emitting them as a `custom` event via `get_stream_writer` from the parent does not currently work as of recent LangGraph versions — the tokens continue to surface as `messages`-mode events rather than being redirected through `custom`. If your architecture nests subgraphs and needs token capture, test this behavior directly against your installed version before building on it, and prefer streaming subgraph tokens through the same `messages` mode used at the top level rather than trying to intercept and re-wrap them.

**`messages` mode metadata vs. `astream_events` for subgraphs:** `messages` mode's metadata gives you the outer node name only (`metadata["langgraph_node"]`). `astream_events()` gives you the full nesting path via `tags` and `parent_ids`. If you need to distinguish "which specific sub-node inside the subgraph" produced a token, `astream_events` is the correct tool, not `messages` mode.

---

### 8.7 Production Wiring: FastAPI + Server-Sent Events

The standard production pattern is a FastAPI endpoint that converts a LangGraph stream into Server-Sent Events (SSE) — a simple, one-directional, auto-reconnecting protocol well suited to streaming chat responses.

```python
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

app = FastAPI()

async def generate_sse(question: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    async for mode, chunk in graph.astream(
        {"messages": [HumanMessage(content=question)]},
        config,
        stream_mode=["updates", "custom", "messages"],
    ):
        if mode == "messages":
            message_chunk, metadata = chunk
            if metadata.get("langgraph_node") == "writer" and message_chunk.content:
                data = json.dumps({"type": "token", "content": message_chunk.content})
                yield f"data: {data}\n\n"
        
        elif mode == "custom":
            data = json.dumps({"type": "status", "content": chunk})
            yield f"data: {data}\n\n"
        
        elif mode == "updates":
            for node_name in chunk.keys():
                data = json.dumps({"type": "node_complete", "node": node_name})
                yield f"data: {data}\n\n"
    
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.get("/stream")
async def stream_endpoint(question: str, thread_id: str):
    return StreamingResponse(
        generate_sse(question, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Critical for nginx — see below
        },
    )
```

**Client side (JavaScript `EventSource`):**

```javascript
const evtSource = new EventSource(`/stream?question=${q}&thread_id=${threadId}`);

evtSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "token") {
        appendToOutput(data.content);
    } else if (data.type === "status") {
        updateStatusIndicator(data.content);
    } else if (data.type === "done") {
        evtSource.close();
    }
};
```

#### 8.7.1 Production Gotchas

**Reverse proxy buffering.** Nginx and similar reverse proxies buffer responses by default, which defeats streaming entirely — the client receives everything at once when the buffer flushes, not incrementally. The `X-Accel-Buffering: no` header (shown above) disables this for nginx specifically. Confirm your specific proxy/load balancer configuration passes streaming responses through without buffering; different infrastructure (Cloudflare, AWS ALB, various CDNs) has its own buffering behavior to check.

**Backpressure on slow clients.** If a client's connection is slow, tokens can accumulate faster than they're consumed. Use bounded queues or timeouts in production streaming code rather than assuming the client will always keep up.

**SSE is one-directional.** Server-to-client only. For bidirectional needs (e.g., allowing the client to send a mid-stream cancellation), use FastAPI's WebSocket support instead. For most chat UIs, SSE is simpler and sufficient.

**Resumable streaming.** A production pattern for handling network interruptions: stream tokens into a durable store (e.g., Redis Streams) as they're generated, and have the client's SSE endpoint read from that store rather than directly from the graph. If the client disconnects and reconnects, it resumes reading from where it left off in the durable stream rather than losing the in-flight response. This decouples token generation (which continues on the server regardless of client connectivity) from token delivery.

---

### 8.8 Testing Streaming Code

A useful invariant when testing streaming implementations: **stream completeness** — verifying that streaming a graph produces output equivalent to what `.invoke()` would produce. This catches bugs where the streaming path silently drops or truncates content (a common failure mode: an SSE serializer that truncates chunks exceeding a size limit).

```python
def test_stream_completeness():
    """Reconstructing the full output from a stream should match invoke()."""
    config_a = {"configurable": {"thread_id": "test-invoke"}}
    config_b = {"configurable": {"thread_id": "test-stream"}}
    
    input_data = {"topic": "test topic", "findings": [], "sources": []}
    
    # Get the ground-truth result via invoke
    invoke_result = graph.invoke(input_data, config_a)
    
    # Reconstruct the final draft by accumulating streamed tokens
    accumulated_tokens = []
    for msg_chunk, metadata in graph.stream(input_data, config_b, stream_mode="messages"):
        if metadata.get("langgraph_node") == "writer" and msg_chunk.content:
            accumulated_tokens.append(msg_chunk.content)
    
    streamed_draft = "".join(accumulated_tokens)
    
    # The streamed reconstruction should match the invoke() result
    # (allowing for minor whitespace differences from token boundaries)
    assert streamed_draft.strip() == invoke_result["draft"].strip()

def test_updates_mode_covers_all_nodes():
    """Every node that runs should appear at least once in 'updates' mode."""
    config = {"configurable": {"thread_id": "test-updates"}}
    input_data = {"topic": "test", "findings": [], "sources": []}
    
    nodes_seen = set()
    for delta in graph.stream(input_data, config, stream_mode="updates"):
        nodes_seen.update(delta.keys())
    
    expected_nodes = {"planner", "researcher", "writer", "critic", "finalize"}
    assert expected_nodes.issubset(nodes_seen)

def test_custom_events_emitted():
    """Verify get_stream_writer() calls in nodes actually surface via custom mode."""
    config = {"configurable": {"thread_id": "test-custom"}}
    input_data = {"topic": "test", "findings": [], "sources": []}
    
    custom_events = list(graph.stream(input_data, config, stream_mode="custom"))
    assert len(custom_events) > 0, "Expected at least one custom status event"
```

---

### 8.9 MARRS Checkpoint: Full Streaming Integration

We now wire streaming into MARRS end-to-end: node-level progress via `updates`, human-readable status via `custom`, and token-level output for the final synthesis via `messages`.

```python
import json
import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessageChunk
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import operator


# ─────────────────────────────────────────────────────────────────────────────
# NODES INSTRUMENTED WITH get_stream_writer FOR CUSTOM PROGRESS
# ─────────────────────────────────────────────────────────────────────────────

def planner_node_streaming(state: MARRSStateV3) -> dict:
    """Planner with custom progress events."""
    writer = get_stream_writer()
    writer({"status": f"Planning research for '{state['topic']}'...", "progress": 0.05})
    
    topic = state["topic"]
    plan = f"Research plan for {topic}"
    queries = [f"{topic} overview", f"{topic} recent advances", f"{topic} applications"]
    
    writer({"status": f"Generated {len(queries)} search queries", "progress": 0.15})
    
    return {"plan": plan, "search_queries": queries}


def researcher_node_streaming(state: MARRSStateV3) -> dict:
    """Researcher with per-query progress events."""
    writer = get_stream_writer()
    queries = state.get("search_queries", [])
    
    all_findings = []
    all_sources = []
    
    for i, query in enumerate(queries):
        writer({
            "status": f"Searching: {query}",
            "progress": 0.15 + (0.5 * (i + 1) / len(queries)),
        })
        # Placeholder — Chapter 4's Send-based parallel search would replace this
        finding = f"Finding for '{query}'"
        source = f"https://example.com/search?q={query}"
        all_findings.append(finding)
        all_sources.append(source)
    
    writer({"status": f"Collected {len(all_findings)} findings", "progress": 0.65})
    
    return {"findings": all_findings, "sources": all_sources}


def writer_node_streaming(state: MARRSStateV3) -> dict:
    """
    Writer node — this is where token-level streaming happens naturally.
    Because this node calls an LLM, its output tokens are automatically
    captured by stream_mode="messages" without any extra code here —
    LangGraph observes the underlying chat model's streaming behavior.
    """
    writer = get_stream_writer()
    writer({"status": "Writing draft...", "progress": 0.70})
    
    findings_text = "\n".join(state.get("findings", []))
    prompt = f"Write a research report on {state['topic']} based on:\n{findings_text}"
    
    # This LLM call's tokens are automatically streamable via stream_mode="messages"
    # when this node is invoked as part of graph.astream(..., stream_mode=["messages"])
    response = llm.invoke(prompt)  # In production: llm is tagged/named for filtering
    
    writer({"status": "Draft complete", "progress": 0.85})
    
    return {"draft": response.content}


def critic_node_streaming(state: MARRSStateV3) -> dict:
    """Critic with progress event."""
    writer = get_stream_writer()
    writer({"status": "Evaluating draft quality...", "progress": 0.90})
    
    revisions = state.get("revision_count", 0)
    score = min(0.60 + (revisions * 0.15), 0.95)
    
    writer({"status": f"Quality score: {score:.2f}", "progress": 0.95})
    
    return {"quality_score": score, "revision_count": revisions + 1}


def finalize_node_streaming(state: MARRSStateV3) -> dict:
    """Finalize with completion event."""
    writer = get_stream_writer()
    writer({"status": "Finalizing report...", "progress": 1.0})
    
    return {
        "final_report": f"# {state.get('topic', 'Report')}\n\n{state.get('draft', '')}",
        "status": "complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_marrs_streaming():
    builder = StateGraph(MARRSStateV3)
    
    builder.add_node("planner", planner_node_streaming)
    builder.add_node("researcher", researcher_node_streaming)
    builder.add_node("writer", writer_node_streaming)
    builder.add_node("critic", critic_node_streaming)
    builder.add_node("finalize", finalize_node_streaming)
    
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "critic")
    builder.add_edge("critic", "finalize")
    builder.add_edge("finalize", END)
    
    return builder.compile(checkpointer=checkpointer)

marrs_streaming = build_marrs_streaming()


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE DEMO: All three modes combined
# ─────────────────────────────────────────────────────────────────────────────

async def run_marrs_streaming_demo(topic: str, thread_id: str):
    """
    Demonstrates the full production streaming pattern:
    - 'custom' events show human-readable progress ("Searching...", "Writing...")
    - 'updates' events show which node just completed
    - 'messages' events show the writer's LLM output token by token
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"topic": topic, "findings": [], "sources": []}
    
    print(f"\n{'='*60}")
    print(f"Streaming MARRS research: {topic}")
    print(f"{'='*60}\n")
    
    async for mode, chunk in marrs_streaming.astream(
        input_data,
        config,
        stream_mode=["updates", "custom", "messages"],
    ):
        if mode == "custom":
            progress = chunk.get("progress", 0)
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r[{bar}] {chunk['status']:<50}", end="", flush=True)
        
        elif mode == "messages":
            message_chunk, metadata = chunk
            if metadata.get("langgraph_node") == "writer" and message_chunk.content:
                # First token of the writer's output — print a newline to
                # separate from the progress bar
                if not hasattr(run_marrs_streaming_demo, "_writer_started"):
                    print("\n\n--- Draft (streaming) ---")
                    run_marrs_streaming_demo._writer_started = True
                print(message_chunk.content, end="", flush=True)
        
        elif mode == "updates":
            for node_name in chunk.keys():
                pass  # Node-complete events already reflected via custom progress bar
    
    print("\n\n✅ Research complete.")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI SSE ENDPOINT FOR MARRS
# ─────────────────────────────────────────────────────────────────────────────

def create_marrs_streaming_api():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    
    app = FastAPI(title="MARRS Streaming API")
    
    async def marrs_sse_stream(topic: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        input_data = {"topic": topic, "findings": [], "sources": []}
        
        async for mode, chunk in marrs_streaming.astream(
            input_data,
            config,
            stream_mode=["updates", "custom", "messages"],
        ):
            if mode == "custom":
                payload = {"type": "status", **chunk}
                yield f"data: {json.dumps(payload)}\n\n"
            
            elif mode == "messages":
                message_chunk, metadata = chunk
                if metadata.get("langgraph_node") == "writer" and message_chunk.content:
                    payload = {"type": "token", "content": message_chunk.content}
                    yield f"data: {json.dumps(payload)}\n\n"
            
            elif mode == "updates":
                for node_name, delta in chunk.items():
                    payload = {"type": "node_complete", "node": node_name}
                    yield f"data: {json.dumps(payload)}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    @app.get("/research/stream")
    async def stream_research(topic: str, thread_id: str):
        return StreamingResponse(
            marrs_sse_stream(topic, thread_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    return app


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_marrs_stream_completeness():
    """The streamed draft should match the invoke() draft."""
    config_invoke = {"configurable": {"thread_id": "stream-test-invoke"}}
    config_stream = {"configurable": {"thread_id": "stream-test-stream"}}
    input_data = {"topic": "streaming test", "findings": [], "sources": []}
    
    invoke_result = marrs_streaming.invoke(input_data, config_invoke)
    
    accumulated = []
    for msg_chunk, metadata in marrs_streaming.stream(input_data, config_stream, stream_mode="messages"):
        if metadata.get("langgraph_node") == "writer" and msg_chunk.content:
            accumulated.append(msg_chunk.content)
    
    streamed_draft = "".join(accumulated)
    assert streamed_draft.strip() == invoke_result["draft"].strip()
    print("✅ test_marrs_stream_completeness passed")

def test_marrs_custom_progress_events():
    """Every stage should emit at least one custom progress event."""
    config = {"configurable": {"thread_id": "stream-test-progress"}}
    input_data = {"topic": "progress test", "findings": [], "sources": []}
    
    statuses = []
    for chunk in marrs_streaming.stream(input_data, config, stream_mode="custom"):
        statuses.append(chunk["status"])
    
    assert any("Planning" in s for s in statuses)
    assert any("Searching" in s or "Collected" in s for s in statuses)
    assert any("Writing" in s or "Draft" in s for s in statuses)
    assert any("Quality" in s for s in statuses)
    assert any("Finalizing" in s for s in statuses)
    print(f"✅ test_marrs_custom_progress_events passed ({len(statuses)} events)")

def test_marrs_updates_mode_node_coverage():
    """All five nodes should report an update."""
    config = {"configurable": {"thread_id": "stream-test-nodes"}}
    input_data = {"topic": "node coverage test", "findings": [], "sources": []}
    
    nodes_seen = set()
    for delta in marrs_streaming.stream(input_data, config, stream_mode="updates"):
        nodes_seen.update(delta.keys())
    
    assert nodes_seen == {"planner", "researcher", "writer", "critic", "finalize"}
    print("✅ test_marrs_updates_mode_node_coverage passed")
```

---

### 8.10 Chapter Summary

**Streaming is a perception fix.** The total compute time is unchanged; what changes is when the user sees the first sign of progress. Use it for any user-facing interaction over roughly two seconds; skip it for background jobs or sub-second latencies.

**Five stream modes** cover distinct needs: `values` (full state snapshot each step — expensive but complete), `updates` (only what changed — the natural progress-bar mode), `messages` (token-by-token LLM output, filterable by `langgraph_node` or tag), `custom` (arbitrary progress data via `get_stream_writer()`), and `debug` (maximum verbosity, development only).

**`get_stream_writer()`** lets any node emit JSON-serializable progress data with no changes to the node's signature. A known limitation: async tools currently do not support `custom` streaming (GitHub #6447) — synchronous tools work correctly.

**Combining multiple stream modes** changes the yielded shape from `chunk` to `(mode, chunk)` — the most common bug when adding a second mode to existing single-mode code.

**`astream_events()`** is the fine-grained lifecycle API — reach for it only when you need per-tool timing, per-model filtering via `include_names`/`include_tags`, or full subgraph nesting depth. For standard chat UIs, `stream_mode="messages"` is simpler and sufficient.

**A newer `version="v3"` event streaming API exists in beta**, offering typed per-channel projections (`stream.messages`, `stream.values`, `stream.interrupted`) instead of one branch-on-everything event firehose, and it's the same interface used to drive HITL loops in Chapter 7. It doesn't replace `stream_mode` or v2 — both are explicitly "unchanged" — but it's worth knowing about once your event-handling code outgrows simple conditionals.

**Subgraphs require `subgraphs=True`** to surface their internal events, which changes the yielded shape to include a namespace tuple. A known limitation (GitHub #4718): redirecting subgraph LLM tokens through `custom` mode from the parent does not currently work reliably — prefer letting subgraph tokens surface through the same `messages` mode used at the top level.

**Production wiring** is FastAPI + SSE, with `X-Accel-Buffering: no` to defeat reverse-proxy buffering, and consideration for backpressure on slow clients. For resilience against disconnects, route tokens through a durable intermediate store (e.g., Redis Streams) so clients can resume.

**Testing streaming code** should verify stream completeness — that accumulating streamed output reproduces the equivalent `invoke()` result — to catch silent truncation bugs in the serialization path.

---

### Further Reading

- **Official LangGraph Streaming docs**: `docs.langchain.com/oss/python/langgraph/streaming` — the canonical reference for all five stream modes, `get_stream_writer`, and combining modes
- **`astream_events` reference (langchain_core)**: `reference.langchain.com/python/langchain-core/runnables/base/Runnable/astream_events` — the full `StreamEvent` schema and version parameter
- **LangGraph `stream` reference (v3 streaming infrastructure)**: `reference.langchain.com/python/langgraph/stream` — the `StreamMux`, typed projections, and transformer pipeline behind `version="v3"`
- **GitHub issue #6447**: `github.com/langchain-ai/langgraph/issues/6447` — async tools and `custom` stream mode limitation
- **GitHub issue #4718**: `github.com/langchain-ai/langgraph/issues/4718` — subgraph LLM token capture limitation
- **"Streaming LangGraph Agents: Real-Time Progress, Token Streaming, and Production Patterns"**: production patterns for combining `updates`/`custom`/`messages` and deploying behind FastAPI + SSE
- **"Streaming Agent Responses in LangGraph: Tokens, Events, and Real-Time UI Integration"**: a complete worked FastAPI SSE endpoint with a JavaScript `EventSource` client

---

*End of Chapter 8. Chapter 9: Tools and Tool-Calling — Structured Outputs, Parallel Execution, and Error Handling.*
