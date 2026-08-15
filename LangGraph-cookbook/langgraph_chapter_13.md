# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 13 — MCP and A2A: Connecting Agents to Tools and to Each Other

> *"MCP servers run as separate processes—they can't access LangGraph runtime information like the store, context, or agent state. Interceptors bridge this gap."*
> — LangChain official documentation

---

### What This Chapter Covers

Chapter 9 built tools with `@tool` — Python functions living inside your own codebase. That works well when you own the tool's implementation. It works poorly when the tool is "read my company's Slack," "query a partner's database," or "call a specialist agent someone else built in a different framework entirely" — cases where the capability lives outside your code and you don't want to hand-write a client for every different API's shape. This chapter covers the two open protocols that solve that: **MCP** (Model Context Protocol) for connecting agents to external tools and data, and **A2A** (Agent2Agent) for connecting agents to *other agents*, regardless of what framework built them.

By the end you will understand:

1. What MCP actually standardizes, and why it exists alongside `@tool` rather than replacing it
2. `MultiServerMCPClient`: connecting to multiple MCP servers over `stdio` and `streamable_http`, and pulling their tools into any LangGraph agent — hand-built (Chapter 4) or prebuilt (Chapter 9)
3. An honest, documented caution about when MCP is the wrong choice — including a specific transport you should not use in a deployed web service
4. Statelessness by default: what a fresh `ClientSession` per call means for your tools, and when to reach for a stateful session instead
5. Interceptors: how to bridge the gap between an MCP server (a separate process with no access to your graph's state or store) and your running graph — using the same `Command` mechanism from Chapter 4
6. Authenticating to MCP servers with custom headers
7. The flip side: exposing your own LangGraph agent as an MCP tool via the Agent Server's `/mcp` endpoint (Chapter 12), and the schema-design implications of doing so
8. A2A: what it standardizes that MCP doesn't, the Agent Card, and how it relates to Chapter 10's Handoffs pattern once the "other agent" is built in a different framework
9. A decision framework: local `@tool`, MCP, Chapter 10's in-process multi-agent patterns, or A2A — for a given situation, which is the right layer
10. MARRS checkpoint: giving the research subagent real MCP-backed tools, and exposing MARRS itself as both an MCP tool and an A2A-discoverable agent

---

### 13.1 What MCP Standardizes, and Why It's Not Just `@tool` Again

MCP, released by Anthropic in late 2024, is an open protocol for describing tools and data sources in a model-agnostic format, so an LLM application can discover and call them through one structured, transport-agnostic API — commonly described as a standardized connector for AI applications, the same way USB-C standardized device connectivity.

The problem it solves is specifically the *N×M integration problem*: without a shared protocol, every agent framework needs its own bespoke client for every external service (Slack, GitHub, a company's internal database, a partner's API), and every service needs to be wrapped separately for every framework that wants to call it. MCP collapses this to N+M: a service exposes one MCP server, and any MCP-compliant client — LangGraph, another framework, an entirely different vendor's agent — can call it without custom integration code.

**This does not replace `@tool` from Chapter 9.** A tool you own, whose implementation lives in your own codebase and has no reason to be shared outside it, is simpler as a plain `@tool` — no separate process, no protocol overhead, no network hop. MCP earns its complexity specifically when the tool's implementation lives *outside* your codebase: someone else's service, a capability you want to swap or version independently of your agent's deployment, or a tool you want reachable from more than one agent or framework. Section 13.2's official guidance makes this tradeoff explicit rather than treating MCP as a default.

---

### 13.2 `MultiServerMCPClient`: Connecting to MCP Servers

```bash
pip install langchain-mcp-adapters
```

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "math": {
        "command": "python",
        "args": ["./servers/math_server.py"],
        "transport": "stdio",       # a local subprocess, communicating over stdin/stdout
    },
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",   # the current standard transport for networked MCP servers
    },
})

tools = await client.get_tools()   # every tool from every configured server, wrapped as LangChain tools
```

`get_tools()` fetches tool schemas from each configured server and returns them as ordinary LangChain tools — indistinguishable, from your graph's point of view, from a tool you wrote with `@tool` in Chapter 9. This is what makes the integration essentially free: everything from Chapters 4 and 9 about tools, `ToolNode`, and error handling applies unchanged.

#### 13.2.1 With a Hand-Built Graph (Chapter 4's Pattern)

```python
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4.1")

def call_model(state: MessagesState):
    response = model.bind_tools(tools).invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_node(ToolNode(tools))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")
graph = builder.compile()
```

This is the exact ReAct loop from Chapter 4 — the only difference is that `tools` came from `client.get_tools()` instead of a list of `@tool`-decorated functions.

#### 13.2.2 With `create_agent` (Chapter 9's Pattern)

```python
from langchain.agents import create_agent

agent = create_agent("openai:gpt-4.1", tools)
response = await agent.ainvoke({"messages": [{"role": "user", "content": "What's the weather in SF?"}]})
```

Equally direct — `create_agent` doesn't distinguish between an MCP-sourced tool and a local `@tool`.

#### 13.2.3 Transports: `stdio` vs. `streamable_http`

| Transport | What it is | Use for |
|---|---|---|
| `stdio` | The MCP server runs as a local subprocess of your Python process, communicating over stdin/stdout | Local development, tools genuinely meant to run on the same machine as the calling process |
| `streamable_http` | The MCP server is a networked HTTP service (with automatic fallback to SSE for older servers that don't yet support Streamable HTTP) | Any deployed, multi-process, or remote scenario — this is the transport for production |

**The caution worth taking seriously, directly from the official documentation:** *"MCP's stdio transport was designed primarily to support applications running on a user's machine. Before using stdio in a web server context, evaluate whether there's a more appropriate solution. For example, do you actually need MCP? Or can you get away with a simple `@tool`?"* This is a real, documented warning, not hedging — `stdio` ties an MCP server's lifecycle to a subprocess of your web server process, which is a poor fit for anything horizontally scaled or restarted independently of that subprocess. If you're deploying (Chapter 12) and reaching for `stdio` because that's what the tutorial you copied used locally, stop and ask whether `streamable_http` (or, per the quoted guidance, no MCP at all) is the better fit before shipping it.

#### 13.2.4 Authentication: Custom Headers

```python
client = MultiServerMCPClient({
    "internal_api": {
        "transport": "streamable_http",
        "url": "https://internal-tools.example.com/mcp",
        "headers": {
            "Authorization": "Bearer YOUR_TOKEN",
            "X-Custom-Header": "custom-value",
        },
    }
})
```

For per-user credentials rather than one static token, combine this with Chapter 12's custom auth: read the authenticated user's identity from `config["configurable"]["langgraph_auth_user"]` inside your node, fetch that user's token from a secret store, and construct the `MultiServerMCPClient` config per-request rather than once at import time.

---

### 13.3 Statelessness by Default, and When You Need More

`MultiServerMCPClient` is stateless by default: **each tool invocation opens a fresh `ClientSession`, executes the call, and tears the session down.** For most tools — a search, a lookup, a calculation — this is exactly right and requires no thought.

Some MCP servers expose more than tools — **prompts**, for instance — which require an explicit session rather than the one-shot-per-call default:

```python
from langchain_mcp_adapters.prompts import load_mcp_prompt

async with client.session("server_name") as session:
    messages = await load_mcp_prompt(session, "summarize")
    # or with arguments:
    messages = await load_mcp_prompt(session, "code_review", arguments={"language": "python", "focus": "security"})
```

Reach for an explicit `client.session(...)` block when you need multiple related calls against the same server to share context (a stateful multi-step negotiation with that server), or when working with MCP capabilities beyond simple tool invocation, like prompts. For ordinary tool calls, the default stateless behavior is what you want — it means a slow or failing server can't leave a dangling connection your process has to manage.

---

### 13.4 Interceptors: Bridging MCP's Process Isolation

This is the single most important structural fact about MCP tools, and it's easy to miss until it bites you: **MCP servers run as separate processes.** They cannot see your LangGraph runtime's `store` (Chapter 6), the run's `config`, or the graph's `state` — none of the context-injection mechanisms from Chapter 9's `InjectedState`/`InjectedStore` are available to code running on the other side of an MCP connection, because that code isn't running inside your graph's process at all.

**Interceptors** bridge this gap. They wrap MCP tool calls with the same kind of middleware-style control Chapter 9 introduced for local tools (`wrap_tool_call`) — letting you inject runtime context into the request, modify it, retry it, add headers dynamically, or short-circuit it entirely:

```python
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

async def handle_task_completion(request: MCPToolCallRequest, handler):
    """Mark a task complete and hand off to a summary agent — using Command,
    exactly as Chapter 4 introduced it, but triggered from an MCP tool call."""
    result = await handler(request)
    if request.name == "submit_order":
        return Command(
            update={"messages": [result] if isinstance(result, ToolMessage) else [ToolMessage(content=str(result), tool_call_id=request.tool_call_id)]},
            goto="summary_agent",
        )
    return result
```

The pattern is directly analogous to Chapter 9's tool middleware — the difference is *where* the interception needs to happen. A local `@tool` can read `InjectedState` directly because it executes inside your process. An MCP tool cannot, so the interceptor is where you supply whatever runtime context the call needs (a user ID for scoping, an API key looked up from your own store, a retry policy) before the request ever leaves your process for the MCP server.

---

### 13.5 The Flip Side: Exposing Your LangGraph Agent as an MCP Tool

Everything so far treats your graph as an MCP *client*. The Agent Server from Chapter 12 can also make your deployed graph an MCP *server* — reachable by any MCP-compliant client, including agents built in entirely different frameworks.

```json
// langgraph.json
{
  "graphs": {
    "my_agent": {
      "path": "./my_agent/agent.py:graph",
      "description": "Researches a topic and produces a cited summary report."
    }
  },
  "env": ".env"
}
```

With `langgraph-api >= 0.2.3`, every deployed graph is automatically exposed as an MCP tool at the `/mcp` endpoint — no additional code required. If you're deploying to LangSmith Deployment's Cloud environment (Chapter 12), this happens automatically on each new revision. The endpoint uses Streamable HTTP and the same authentication as the rest of the Agent Server API (Chapter 12's `@auth.authenticate`/`@auth.on`), including the ability to authenticate a specific user to get access to that user's scoped tools within the deployment.

**A schema-design point worth taking seriously before you expose an agent this way:** the default `MessagesState` uses `AnyMessage` — a deliberately permissive type that's exactly right for a chat interface but, per the official guidance, "too general for direct LLM exposure" as an MCP tool. A calling agent needs a tightly-scoped, explicit input/output schema to reliably use your agent as a tool — not a type that accepts arbitrary message-list shapes. Define custom, narrowly-typed input and output schemas (Chapter 3's input/output schema separation is the exact mechanism) for any graph you intend to expose over MCP, rather than exposing your internal `MessagesState` directly.

---

### 13.6 A2A: When the "Tool" Is Another Agent

MCP standardizes how an agent talks to tools and data sources. **A2A (Agent2Agent) standardizes how one agent talks to another agent** — including one built in a completely different framework. The two protocols are explicitly complementary, not competing: a single system commonly uses MCP for each agent's tool access and A2A for delegation between agents.

#### 13.6.1 The Agent Card

An A2A-compliant agent publishes an **Agent Card** — a JSON document at a well-known path (`/.well-known/agent-card.json`) describing what it can do, so other agents can discover its capabilities without out-of-band documentation:

```python
# Sketch of exposing a LangGraph agent as an A2A server
from a2a.server import A2AServer
from a2a.types import AgentCard

card = AgentCard(
    name="marrs-research-agent",
    description="Researches a topic and produces a cited report.",
    version="1.0.0",
    capabilities={"streaming": True},
)

# The A2A server serves the card at GET /.well-known/agent-card.json
# and handles JSON-RPC calls that invoke the underlying LangGraph agent
app = A2AServer(agent_card=card, agent_executor=my_marrs_executor)
```

A calling agent — regardless of whether it's built with LangGraph, Google's ADK, CrewAI, or Microsoft's Foundry — fetches the Agent Card, learns what the agent can do, and sends it a task over a standard JSON-RPC interface. This is sometimes called the **opacity principle**: the calling agent doesn't need to know the callee is built with LangGraph internally — it just sends a task and gets a result.

#### 13.6.2 How This Relates to Chapter 10's Handoffs

Chapter 10's Handoffs pattern (`Command(goto=..., graph=Command.PARENT)`) is control transfer *within one codebase* — both agents are LangGraph nodes in the same process, sharing the same graph and the same deployment. A2A is the same conceptual handoff — one agent recognizing another is better suited to continue the task — but **across a process and framework boundary**. You reach for A2A specifically when the other agent:

- Is maintained by a different team, with its own independent deployment lifecycle
- Is built in a different framework entirely (a CrewAI agent, an ADK agent, an agent built on Microsoft Foundry)
- Needs to be called by multiple different orchestrators, not just your one LangGraph application

If both agents are yours, in the same codebase, and deployed together, Chapter 10's in-process Handoffs or Subagents patterns remain simpler and lower-latency — there's no protocol overhead, no separate service to run, no Agent Card to maintain. A2A earns its overhead at genuine organizational or framework boundaries, the same way MCP earns its overhead specifically when the tool implementation lives outside your codebase.

---

### 13.7 Choosing the Right Layer: A Decision Framework

| Situation | Use |
|---|---|
| You own the tool's implementation, it's used only by this agent | `@tool` (Chapter 9) — simplest, no process/protocol overhead |
| The tool's implementation lives outside your codebase, or should be reusable across agents/frameworks | MCP — `MultiServerMCPClient` as a client, or expose your own graph via `/mcp` as a server |
| A specialist "agent" is really just your own graph logic, in the same codebase | Chapter 10's Subagents or Handoffs — no protocol needed at all |
| The other agent is a different team's or a different framework's deployment | A2A — Agent Card discovery, JSON-RPC delegation across the boundary |
| You're tempted to reach for MCP's `stdio` transport inside a deployed web service | Stop and reconsider — Section 13.2.3's caution applies; `streamable_http` or a plain `@tool` is very likely the better fit |

A realistic production system layers all of these rather than picking exactly one: a LangGraph supervisor (Chapter 10) coordinating in-process specialists, some of which call out to MCP servers for tools they don't own, occasionally delegating a whole sub-task to an entirely different team's agent over A2A — with LangSmith (Chapter 11) tracing across all of it.

---

### 13.8 MARRS Checkpoint: Real MCP Tools, and MARRS as an MCP/A2A Endpoint

We now give MARRS's research subagent (Chapter 9's `RESEARCH_TOOLS`) real, externally-hosted tools via MCP instead of the illustrative local functions, and expose MARRS itself both as an MCP tool and as an A2A-discoverable agent.

```python
"""
marrs/mcp_tools.py

Replaces App. A.2's illustrative @tool functions with real MCP-backed tools —
a web-search MCP server and an internal document-store MCP server, both
reachable over streamable_http since this runs inside a deployed Agent Server
(Chapter 12), where stdio would be the wrong transport (§13.2.3).
"""
from langchain_mcp_adapters.client import MultiServerMCPClient

async def build_research_tools(user_token: str) -> list:
    client = MultiServerMCPClient({
        "web_search": {
            "transport": "streamable_http",
            "url": "https://search-mcp.example.com/mcp",
            "headers": {"Authorization": f"Bearer {user_token}"},   # §13.2.4
        },
        "arxiv": {
            "transport": "streamable_http",
            "url": "https://arxiv-mcp.example.com/mcp",
        },
    })
    return await client.get_tools()


async def researcher_agent_node(state, config):
    """
    Ch9's researcher_agent_node, now backed by real MCP tools instead of
    illustrative placeholders. The user's token is drawn from the
    authenticated identity Chapter 12's custom auth attaches to every run.
    """
    user_info = config["configurable"].get("langgraph_auth_user", {})
    tools = await build_research_tools(user_info.get("token", ""))
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}
```

```json
// langgraph.json — expose MARRS itself as an MCP tool other agents can call
{
  "graphs": {
    "marrs": {
      "path": "./marrs/graph.py:graph",
      "description": "Researches a topic thoroughly and produces a structured, cited report. Input: a topic string. Output: a final_report field containing the completed report."
    }
  },
  "env": ".env",
  "auth": {
    "path": "src/security/auth.py:auth"
  }
}
```

```python
"""
marrs/schemas.py

Custom, narrowly-typed input/output schemas for MCP exposure (§13.5) —
NOT the raw MessagesState/AnyMessage shape used for MARRS's own chat interface.
"""
from typing import TypedDict

class MARRSMCPInput(TypedDict):
    topic: str

class MARRSMCPOutput(TypedDict):
    final_report: str
    quality_score: float
    status: str
```

```python
"""
marrs/a2a_server.py

Expose MARRS as an A2A-discoverable agent, so a supervisor built in a
different framework entirely (Chapter 10 §10.6's "opacity principle") can
delegate a research task to MARRS without knowing it's built on LangGraph.
"""
from a2a.server import A2AServer
from a2a.types import AgentCard

marrs_card = AgentCard(
    name="marrs-research-agent",
    description="Multi-agent research and report writing system. "
                 "Give it a topic; it returns a cited research report.",
    version="1.0.0",
    capabilities={"streaming": True, "human_in_the_loop": True},   # Ch7's interrupt()-based review gate
)

marrs_a2a_app = A2AServer(agent_card=marrs_card, agent_executor=marrs_executor)
```

With this in place, MARRS is simultaneously: a chat-facing LangGraph application (Chapters 2–9), an MCP tool other LangGraph agents (or agents in any other framework) can call directly, and an A2A-discoverable agent a cross-framework orchestrator can delegate to — all backed by the same underlying graph and deployment from Chapter 12.

---

### 13.9 Chapter Summary

**MCP** standardizes how an agent discovers and calls external tools and data sources, solving the N×M integration problem — one MCP server, callable by any MCP-compliant client, instead of a bespoke integration per framework per service. It doesn't replace `@tool` (Chapter 9); reach for it specifically when the tool's implementation lives outside your codebase or needs to be shared across agents and frameworks.

**`MultiServerMCPClient`** connects to multiple MCP servers over `stdio` (local subprocess) or `streamable_http` (networked, the correct choice for anything deployed) and returns ordinary LangChain tools via `get_tools()` — usable identically in a hand-built `ToolNode`/`tools_condition` graph (Chapter 4) or a `create_agent` (Chapter 9). The client is stateless by default (a fresh session per call); use `client.session(...)` explicitly for multi-step interactions or non-tool MCP capabilities like prompts.

**MCP servers run as separate processes** with no access to your graph's state, config, or store — Chapter 9's `InjectedState`/`InjectedStore` don't reach across that boundary. **Interceptors** (`MCPToolCallRequest`) bridge this gap, giving you the same `Command`-based control over an MCP call that Chapter 9's tool middleware gives you over a local one.

**Exposing your own agent as an MCP tool** is largely automatic on the Agent Server (Chapter 12) via the `/mcp` endpoint, but requires deliberately narrow, typed input/output schemas (Chapter 3's schema-separation mechanism) rather than the permissive `MessagesState`/`AnyMessage` shape suited to a chat interface.

**A2A** standardizes agent-to-agent delegation across process and framework boundaries, via a discoverable Agent Card and JSON-RPC calls — the same conceptual handoff as Chapter 10's in-process patterns, but for when the other agent is a different team's or a different framework's deployment rather than another node in your own graph.

**Choosing the right layer** — `@tool`, MCP, Chapter 10's in-process multi-agent patterns, or A2A — comes down to where the boundary actually is: inside your codebase, outside your codebase but still tool-shaped, another specialist within your own graph, or genuinely another team's or framework's agent.

---

### Further Reading

- **Official MCP integration guide**: `docs.langchain.com/oss/python/langchain/mcp` — `MultiServerMCPClient`, interceptors, stateful sessions, and prompt loading, with the stdio-in-production caution
- **`langchain-mcp-adapters` repository**: `github.com/langchain-ai/langchain-mcp-adapters` — full transport configuration reference and working examples
- **Official MCP endpoint in Agent Server**: `docs.langchain.com/langsmith/server-mcp` — exposing a deployed graph as an MCP tool, schema design guidance, and authentication
- **DeepLearning.AI: "A2A: The Agent2Agent Protocol"**: `deeplearning.ai/courses/a2a-the-agent2agent-protocol` — built with Google Cloud and IBM Research; covers the Agent Card, cross-framework delegation, and combining A2A with MCP in one system
- **"The Agent Protocol Stack: MCP vs A2A vs AG-UI"**: a clear layer-by-layer breakdown of which protocol solves which problem, including the opacity principle
- **Model Context Protocol specification**: `modelcontextprotocol.io` — the protocol itself, independent of any single framework's implementation

---

*End of Chapter 13.*
