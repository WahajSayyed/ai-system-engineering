# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 12 — Deployment: LangSmith Deployment, Self-Hosting, and Production Operations

> *"Every agent in production faces the same needs: durable execution, streaming, state and task management. LangSmith Deployment ships with all of it, so you spend your time on your agent logic instead of rebuilding the wheel."*
> — LangChain official documentation

---

### What This Chapter Covers

Every graph in this curriculum has run inside a script, a notebook, or a test. This chapter covers what it takes to run one as a real service — a persistent HTTP API with durable execution, streaming, concurrent-request handling, and authentication, deployable to infrastructure you either fully manage or hand off entirely.

By the end you will understand:

1. A naming change worth knowing before you search for anything else: **LangGraph Platform is now called LangSmith Deployment**
2. The Agent Server execution model — assistants, threads, runs — that every deployment option shares
3. `langgraph.json`: the configuration file every deployment path builds from
4. The LangGraph CLI in full: `new`, `dev`, `build`, `up`, `dockerfile`, and the new one-step `deploy` command
5. The three deployment environments — Cloud, Hybrid/self-hosted-with-control-plane, and Standalone servers — and which to choose
6. Standalone server infrastructure requirements: Postgres and Redis, what each is for, and a hard constraint on where you can run it
7. Double-texting: the four multitask strategies for handling concurrent messages on the same thread, and why this is a deployment-layer feature, not something open-source LangGraph provides
8. Custom authentication: the `Auth` object, `@auth.authenticate`, `@auth.on`, and resource-level access control
9. MARRS checkpoint: packaging the capstone project for deployment

---

### 12.1 A Naming Change Worth Knowing First

Everything this curriculum and most existing tutorials call **"LangGraph Platform"** or **"LangGraph Cloud"** was renamed **LangSmith Deployment** as of October 2025. The rename reflects that this is a deployment *service* — part of LangSmith, the observability and evaluation product from Chapter 11 — sitting on top of the open-source LangGraph *framework*, not a different framework. The distinction that matters: **LangGraph** is the open-source library you've been using throughout this book; **LangSmith Deployment** is the managed service for running LangGraph (or other framework) applications at scale in production. You do not need LangSmith Deployment to use LangGraph — every capability in Chapters 1–11 works standalone — but you do need *something* like it (or the self-hosted equivalent this chapter covers) to run a graph as a durable, scalable, streamable HTTP service rather than a script you invoke locally.

**It's also framework-agnostic.** LangSmith Deployment can run LangGraph applications natively, and it can run agents built with Google's Agent Development Kit, the Claude Agent SDK, Strands, CrewAI, or AutoGen through a wrapping layer (`deployments-wrap-sdk`) or the Functional API. This chapter focuses on the native LangGraph path, which needs no wrapping at all.

---

### 12.2 The Agent Server Execution Model

Whichever deployment environment you choose (Section 12.4), the running service is called an **Agent Server**, and every one of them exposes the same three primitives:

| Primitive | What it is |
|---|---|
| **Assistant** | A configured, named instance of a graph — a specific combination of graph code plus configuration (model, prompt, tools). One graph can have many assistants, each a different configuration. |
| **Thread** | Persisted conversation state, identified by `thread_id` — exactly the same concept from Chapter 5's checkpointer, now exposed over HTTP. |
| **Run** | A single execution of an assistant against a thread — one invocation, trackable, cancellable, resumable. |

This should feel entirely familiar: it's the checkpointer/thread model from Chapter 5, wrapped in an HTTP API with additional server-level concerns — queuing, concurrency control, authentication — layered on top. A deployed graph is called from client code exactly like a local compiled graph, via the LangGraph SDK:

```python
from langgraph_sdk import get_client

client = get_client(url="<DEPLOYMENT_URL>")

thread = await client.threads.create()
run = await client.runs.create(
    thread["thread_id"],
    assistant_id="agent",   # the name given to your graph at deploy time
    input={"messages": [{"role": "user", "content": "Research transformer architectures"}]},
)
await client.runs.join(thread["thread_id"], run["run_id"])
```

---

### 12.3 `langgraph.json`: The Deployment Configuration File

Every deployment path — Cloud, self-hosted, or standalone — starts from the same configuration file at the root of your project:

```json
{
  "dependencies": ["langchain_openai", "."],
  "graphs": {
    "agent": "./src/marrs/graph.py:graph"
  },
  "env": ".env",
  "auth": {
    "path": "src/security/auth.py:auth"
  }
}
```

| Field | Purpose |
|---|---|
| `dependencies` | Python packages and local paths needed to run the graph — mirrors what you'd otherwise put in `pyproject.toml`/`requirements.txt` |
| `graphs` | Maps an assistant name (`"agent"`) to the importable path of a compiled graph object |
| `env` | Path to an `.env` file (or an explicit key-value map) for environment variables the server needs at runtime |
| `auth` | Optional — points to your custom authentication handler (Section 12.8) |

This file is what every CLI command in Section 12.4 reads by default (`-c`/`--config`, defaulting to `langgraph.json` in the current directory).

---

### 12.4 The LangGraph CLI

```bash
pip install langgraph-cli
# For local development with hot reload:
pip install "langgraph-cli[inmem]"
```

| Command | Purpose |
|---|---|
| `langgraph new [PATH] --template TEMPLATE_NAME` | Scaffold a new project from an official template |
| `langgraph dev` | Run the Agent Server locally with hot reload — no Docker required, fastest local dev loop |
| `langgraph build -t IMAGE_TAG` | Build a production Docker image from your `langgraph.json` |
| `langgraph up` | Launch the Agent Server (and its backing services) locally via Docker Compose |
| `langgraph dockerfile SAVE_PATH` | Generate a Dockerfile you can customize further, for deployments needing manual control over the image |
| `langgraph deploy` | **New**: build and deploy to LangSmith Deployment's Cloud environment in one step |

#### 12.4.1 Local Development: `langgraph dev`

```bash
langgraph dev --port 2024
```

This starts a local Agent Server with hot reload directly from your Python environment — no Docker layer, no Postgres/Redis to stand up, the fastest iteration loop while you're actively writing graph code. It opens LangGraph Studio in your browser by default (`--no-browser` to skip this), giving you the visual graph inspector and thread/run debugger from earlier chapters connected to your live local server.

#### 12.4.2 `langgraph up`: Full Local Stack via Docker Compose

```bash
langgraph up --port 8123 --wait
```

This runs the full Agent Server stack — including Postgres and Redis — via Docker Compose, giving you a local environment that matches production topology far more closely than `langgraph dev` does. Use this when you need to test against the same persistence and streaming infrastructure production will use, not just the graph logic itself.

#### 12.4.3 `langgraph deploy`: One-Step Cloud Deployment (New, April 2026)

```bash
langgraph deploy
```

Introduced as a direct addition to the CLI, this single command builds a Docker image from your local project and provisions everything needed to run it in LangSmith Deployment's Cloud environment — including the supporting Postgres and Redis services — without any manual infrastructure setup. It's built specifically to slot into existing CI/CD pipelines (GitHub Actions, GitLab CI, Bitbucket Pipelines), turning "deploy this agent" into one command a pipeline step can call directly, mirroring the CI/CD quality-gate pattern from Chapter 11's Section 11.7.

---

### 12.5 The Three Deployment Environments

All infrastructure types run the identical Agent Server runtime — the difference between them is entirely about *who manages what*, not different feature sets.

| Environment | Control plane | Data plane (Agent Servers + DBs) | Best for |
|---|---|---|---|
| **Cloud** | LangChain (AWS/GCP) | LangChain (AWS/GCP) | Fastest path to production; push from GitHub or `langgraph deploy`; requires a Plus plan or above |
| **Hybrid / Self-hosted with control plane** | LangChain-hosted or self-hosted control plane | **You** (Kubernetes, alongside self-hosted LangSmith) | Teams needing data to stay in their own infrastructure while still getting a managed deployment UI |
| **Standalone servers** | None | **You** (Docker, Docker Compose, or Kubernetes) | Maximum control; no control-plane dependency at all; the lightest-weight production option |

#### 12.5.1 Cloud: The Managed Default

```bash
langgraph deploy
# or: create a deployment from a connected GitHub repo directly in the LangSmith UI
```

Fully managed by LangChain on AWS or GCP. You push code; the build (Docker image creation, provisioning) is handled internally. This is the right default when you don't have a specific data-residency or infrastructure-ownership requirement, and it's the fastest path from a working local graph to a production URL.

#### 12.5.2 Hybrid / Self-Hosted With Control Plane: Data Stays With You

```bash
# Recommended path: run the control plane and Agent Servers in your own Kubernetes
# cluster using the LangSmith Helm chart, alongside self-hosted LangSmith
```

The control plane (deployment management, the UI you use to create and version deployments) can be either LangChain-hosted or self-hosted, but the data plane — the actual Agent Server containers, Postgres, and Redis — runs in your own cloud. This is the option teams reach for when compliance or data-residency requirements mean production data cannot leave infrastructure they control, while they still want the convenience of a managed deployment UI for builds and revisions.

#### 12.5.3 Standalone Servers: No Control Plane at All

```bash
langgraph build -t marrs-agent:latest
# then deploy the resulting image via Docker, Docker Compose, or Kubernetes yourself,
# with no LangSmith control plane involved
```

This is the most lightweight, least restrictive deployment model — you run the Agent Server container directly, with your own backing Postgres and Redis, and no dependency on any LangChain-managed control plane whatsoever. You can still optionally point the server at LangSmith (Cloud or self-hosted) purely for tracing and evaluation, entirely decoupled from deployment management.

**Required environment variables for a standalone deployment:**

| Variable | Purpose |
|---|---|
| `DATABASE_URI` | PostgreSQL connection string — stores assistants, threads, and runs; persists thread state and long-term memory (Chapters 5–6); manages the background task queue's state |
| `REDIS_URI` | Redis connection string — used as a pub-sub broker to stream real-time output from background runs |

**A hard constraint:** standalone servers should not be run in serverless environments. The durable execution, background task queue, and streaming model all assume a long-lived process — a serverless function's lifecycle (spin up, handle one request, tear down) is fundamentally incompatible with how the Agent Server manages runs.

**Sharing a Redis instance across deployments:** multiple standalone deployments can share one Redis instance as long as each uses a distinct database number within it — `redis://<host>:<port>/1` for deployment A, `redis://<host>:<port>/2` for deployment B. The same database number must never be reused across separate deployments, since Redis is acting as the streaming pub-sub layer and cross-deployment message bleed would follow.

#### 12.5.4 Choosing Between Them

| If you need... | Choose |
|---|---|
| Fastest path to production, no infrastructure to manage | Cloud |
| Data-residency guarantees, but still want a managed UI | Hybrid / self-hosted with control plane |
| Total infrastructure control, minimal dependency footprint | Standalone |
| Kubernetes as your production standard already | Hybrid (Helm chart) or Standalone (Kubernetes) — both work |

---

### 12.6 Double-Texting: Handling Concurrent Messages on the Same Thread

Users don't always wait for a response before sending another message — someone asks a question, then immediately follows up before the first answer finishes. LangSmith Deployment calls this **double-texting**, and it's explicitly a feature of the Agent Server, not something the open-source LangGraph framework handles on its own — you get this behavior only once you're running behind a deployed Agent Server.

Four `multitask_strategy` options control what happens when a new run arrives on a thread that already has one in progress:

| Strategy | Behavior |
|---|---|
| **Enqueue** (default) | The current run finishes first; the new run is queued and executes afterward, in order received |
| **Interrupt** | The current run is stopped; a new run starts immediately with the new input; the interrupted run's partial state up to that point is preserved |
| **Rollback** | The current run is stopped **and deleted from the database entirely** — it cannot be resumed or restarted, unlike Interrupt |
| **Reject** | The new run is rejected outright (the client gets an error); the original run continues uninterrupted |

```python
from langgraph_sdk import get_client

client = get_client(url="<DEPLOYMENT_URL>")
thread = await client.threads.create()

# First run
first_run = await client.runs.create(
    thread["thread_id"], "agent",
    input={"messages": [{"role": "user", "content": "What's the weather in SF?"}]},
)

import asyncio
await asyncio.sleep(2)  # let it get partway through

# Second run arrives before the first finishes — choose a strategy explicitly
second_run = await client.runs.create(
    thread["thread_id"], "agent",
    input={"messages": [{"role": "user", "content": "What's the weather in NYC?"}]},
    multitask_strategy="interrupt",   # or "rollback", "reject", "enqueue"
)
await client.runs.join(thread["thread_id"], second_run["run_id"])
```

**Interrupt vs. Rollback — the distinction that matters:** both stop the in-progress run immediately, but Interrupt preserves the interrupted run's record (it remains inspectable, even if not resumable as-is), while Rollback deletes it from the database outright, permanently. Choose Rollback only when the interrupted run's partial state genuinely has no value — for instance, a stale query that's been fully superseded by the user's follow-up.

**A production gotcha worth knowing before you pick Interrupt or Rollback for a tool-calling agent:** if a run is stopped mid-tool-call — after the model emitted an `AIMessage` with `tool_calls` but before the corresponding `ToolMessage` results were produced — the thread's message history can end up with an orphaned tool call. Most model providers (including Anthropic's API) reject a message history containing a tool call with no matching result, which means every subsequent message on that thread fails until the orphaned call is repaired. If your agent mixes long-running tool calls with double-texting via Interrupt or Rollback, plan for this: either keep tool calls short enough that mid-call interruption is rare, or add repair logic that detects and patches an orphaned tool call before the next run on that thread proceeds.

**Choosing a default:** Enqueue is the safest default for most applications — nothing is lost, nothing needs repair logic, the tradeoff is purely that the user waits slightly longer for their follow-up to be addressed. Reach for Interrupt or Reject only when immediate responsiveness to the newest message matters more than guaranteed delivery of the first one, and be deliberate about the orphaned-tool-call risk above.

---

### 12.7 Assistants: Versioned Configuration

An assistant is not just "a name pointing at a graph" — it's a specific, versioned configuration of that graph. The same `graph` entry in `langgraph.json` can back many assistants, each with a different model, prompt, or tool configuration:

```python
research_assistant = await client.assistants.create(
    graph_id="agent",
    config={"configurable": {"model": "anthropic:claude-sonnet-5", "system_prompt": "Be thorough and cite sources."}},
    name="marrs-thorough",
)

quick_assistant = await client.assistants.create(
    graph_id="agent",
    config={"configurable": {"model": "anthropic:claude-haiku-4-5", "system_prompt": "Be brief."}},
    name="marrs-quick",
)
```

This is what makes A/B testing prompts or models in production a first-class operation rather than a code branch — two assistants, same graph, different configuration, both addressable by their own `assistant_id` from client code.

---

### 12.8 Custom Authentication and Resource-Level Access Control

Every deployment path ships with built-in API-key authentication, but production systems almost always need to integrate with an existing identity provider (Auth0, Okta, Supabase Auth, or a homegrown auth server) and enforce that users can only access their own threads. LangSmith Deployment provides this through a low-level `Auth` object with two kinds of handler, and it's supported on every plan, including self-hosted.

**The distinction that matters:** authentication ("AuthN") verifies *who* is making the request; authorization ("AuthZ") determines *what* they're allowed to do once identified. LangSmith separates these explicitly: `@auth.authenticate` handles the former, `@auth.on` handlers handle the latter.

#### 12.8.1 `@auth.authenticate`: Verifying Identity

```python
# src/security/auth.py
from langgraph_sdk import Auth

auth = Auth()

# Toy example — validate against a real identity provider in production
VALID_TOKENS = {
    "user1-token": {"id": "user1", "name": "Alice"},
    "user2-token": {"id": "user2", "name": "Bob"},
}

@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    token = headers.get("authorization", "").removeprefix("Bearer ")
    user = VALID_TOKENS.get(token)
    if not user:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")
    return {"identity": user["id"], "name": user["name"]}
```

This function runs as middleware on **every** request to the deployed server. Whatever dict you return becomes part of a special configuration object the platform attaches to the run's `config["configurable"]` — available to every node in your graph under the `langgraph_auth_user` key:

```python
def my_node(state: MARRSState, config: RunnableConfig) -> dict:
    user_info = config["configurable"].get("langgraph_auth_user")
    user_id = user_info.get("identity")
    # Use this to scope long-term memory (Chapter 6) to the authenticated user,
    # or to fetch a user-specific credential for an on-behalf-of API call
    ...
```

This is exactly how you'd wire the `user_id` distinction from Chapter 6's memory namespaces to a real authenticated identity rather than a value the client claims in the request body — the server, not the client, determines who the user actually is.

#### 12.8.2 `@auth.on`: Resource-Level Authorization

```python
from langgraph_sdk.auth import Auth, is_studio_user

@auth.on
async def add_owner(ctx: Auth.types.AuthContext, value: dict) -> dict:
    """
    Runs on every create/read/update/delete against a resource (threads, runs,
    assistants). Returns a filter dict enforced at the database level —
    not just hidden in the UI.
    """
    if is_studio_user(ctx.user):
        return {}   # allow LangGraph Studio's own logged-in access unrestricted

    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
```

The returned filter dict is enforced at the database level on every operation — meaning even a user holding a technically-valid token cannot read, modify, or delete another user's thread simply by guessing or obtaining its `thread_id`. This closes a real gap that a purely client-side or UI-level check would leave open: `@auth.on` makes the constraint impossible to bypass by talking to the API directly, since the API itself is where the filter is applied.

#### 12.8.3 Wiring Auth Into `langgraph.json`

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/marrs/graph.py:graph"
  },
  "env": ".env",
  "auth": {
    "path": "src/security/auth.py:auth"
  }
}
```

**A note on secrets:** fetch user-specific credentials (an OAuth token, an API key scoped to that user) from a secure secret store inside your node, using the `langgraph_auth_user` identity to look it up — never store secrets directly in graph state, since state is checkpointed to your database in plain form unless you've layered encryption on top (Chapter 5's `EncryptedSerializer`).

---

### 12.9 MARRS Checkpoint: Packaging the Capstone for Deployment

We now package MARRS — the supervisor architecture from Chapter 10, with the persistence, memory, and HITL layers from Chapters 5–7 — for deployment.

```python
# src/marrs/graph.py — the deployable entry point
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information on a topic."""
    return f"[{max_results} results for '{query}']"

research_subagent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[web_search],
    system_prompt="You are MARRS's research specialist.",
)

@tool
def delegate_research(topic: str) -> str:
    """Delegate a research task to the research specialist."""
    response = research_subagent.invoke({"messages": [{"role": "user", "content": f"Research: {topic}"}]})
    return response["messages"][-1].content

# NOTE: no checkpointer is set here — the deployed Agent Server provides its
# own Postgres-backed checkpointer automatically at deploy time. Setting one
# explicitly in code would conflict with the platform-managed persistence layer.
graph = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[delegate_research],
    system_prompt="You coordinate MARRS research reports. Delegate all research to delegate_research.",
)
```

```json
// langgraph.json
{
  "dependencies": ["langchain", "langchain_anthropic", "."],
  "graphs": {
    "agent": "./src/marrs/graph.py:graph"
  },
  "env": ".env",
  "auth": {
    "path": "src/security/auth.py:auth"
  }
}
```

```python
# src/security/auth.py — scope every thread to its authenticated owner
from langgraph_sdk import Auth
from langgraph_sdk.auth import is_studio_user

auth = Auth()

@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    token = headers.get("authorization", "").removeprefix("Bearer ")
    user = validate_against_identity_provider(token)   # your real auth provider call
    if not user:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")
    return {"identity": user["id"]}

@auth.on
async def scope_to_owner(ctx: Auth.types.AuthContext, value: dict) -> dict:
    if is_studio_user(ctx.user):
        return {}
    filters = {"owner": ctx.user.identity}
    value.setdefault("metadata", {}).update(filters)
    return filters
```

```bash
# Local development — fast iteration, hot reload, LangGraph Studio attached
langgraph dev

# Local full-stack test — matches production topology (Postgres + Redis)
langgraph up --wait

# Build a production image for a standalone or hybrid deployment
langgraph build -t marrs-agent:latest

# Or, for LangSmith Deployment's Cloud environment, one step:
langgraph deploy
```

```python
# Client-side usage against the deployed MARRS agent
from langgraph_sdk import get_client

client = get_client(url="https://marrs-agent.deployments.langchain.com")

thread = await client.threads.create()

run = await client.runs.create(
    thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Research quantum error correction"}]},
    multitask_strategy="enqueue",   # safe default: no follow-up message is ever lost
)
result = await client.runs.join(thread["thread_id"], run["run_id"])
```

**Deployment choice for MARRS specifically:** given that MARRS involves per-user research threads with long-term memory (Chapter 6) that must stay isolated per user, and HITL review gates (Chapter 7) that can leave a thread paused for extended periods, Cloud is the pragmatic default unless there's a specific data-residency requirement — the managed Postgres/Redis backing and the automatic durable-execution handling mean the interrupted-thread and long-lived-thread scenarios MARRS specifically relies on are handled without you standing up and operating that infrastructure yourself.

---

### 12.10 Chapter Summary

**LangGraph Platform was renamed LangSmith Deployment in October 2025.** LangGraph remains the open-source framework; LangSmith Deployment is the managed service for running it (or other agent frameworks) in production, built on the same open-source runtime throughout.

**Every deployment shares the Agent Server execution model** — assistants (configuration), threads (state, the same checkpointer concept from Chapter 5 exposed over HTTP), and runs (individual executions) — regardless of which of the three environments you choose.

**`langgraph.json`** is the configuration file every deployment path reads: dependencies, the graph-name-to-object mapping, environment variables, and an optional custom auth handler.

**The CLI** covers the full lifecycle: `new` to scaffold, `dev` for hot-reload local iteration, `up` for a full local stack matching production topology, `build`/`dockerfile` for image creation, and the new `deploy` for one-step Cloud deployment suited to CI/CD pipelines.

**Three deployment environments** — Cloud (fully managed), Hybrid/self-hosted-with-control-plane (data stays with you, managed UI), and Standalone (no control plane at all, maximum control) — all run the identical Agent Server runtime and differ only in who manages the control plane and data plane. Standalone servers need `DATABASE_URI` (Postgres) and `REDIS_URI` (streaming pub-sub) and must never run in a serverless environment.

**Double-texting** — concurrent messages on the same thread — is handled by four `multitask_strategy` options (Enqueue default, Interrupt, Rollback, Reject), a feature specific to the deployed Agent Server rather than open-source LangGraph. Interrupt and Rollback both stop an in-progress run immediately but differ in whether the interrupted run's record is preserved; both carry a real risk of orphaned tool calls breaking a thread's message history if a tool-calling run is stopped mid-call.

**Custom authentication** separates verifying identity (`@auth.authenticate`, populating `config["configurable"]["langgraph_auth_user"]` for every node to use) from enforcing resource-level access (`@auth.on`, returning a filter dict enforced at the database level on every thread/run/assistant operation) — closing the gap a UI-only access check would leave open.

---

### Further Reading

- **Official LangSmith Deployment overview**: `docs.langchain.com/langsmith/deployment` — the current entry point covering framework-agnostic deployment, the three environments, and the Agent Server execution model
- **Official Standalone servers guide**: `docs.langchain.com/langsmith/deploy-standalone-server` — full environment variable reference, Kubernetes/Docker specifics, Redis-sharing pattern
- **Official Double-texting concept guide**: `docs.langchain.com/langsmith/double-texting` — all four multitask strategies with full examples
- **Official Authentication & access control guide**: `docs.langchain.com/langsmith/auth` — the complete `Auth` object reference, `@auth.authenticate`/`@auth.on`, and MCP/OAuth2 identity provider integration
- **Official Custom auth setup tutorial**: `docs.langchain.com/langsmith/custom-auth` — the three-part progression from toy auth to production-ready
- **`langgraph-cli` reference**: `pypi.org/project/langgraph-cli` — every command and flag, and the full `langgraph.json` schema
- **"Introducing `langgraph deploy`"**: `langchain.com/blog/introducing-deploy-cli` — the one-step CI/CD deployment command
- **"LangGraph Platform in beta: New deployment options"**: `blog.langchain.com/langgraph-platform-announce` — historical context for the Cloud SaaS / BYOC / Self-Hosted naming evolution into the current three-environment model

---

*End of Chapter 12. This concludes the core curriculum — Chapters 1 through 12 take LangGraph from the Pregel mental model through production deployment. Appendices to follow: a math and CUDA cross-reference, a full MARRS source listing, and a troubleshooting index organized by symptom.*
