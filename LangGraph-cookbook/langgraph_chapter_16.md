# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 16 — Sandboxing Infrastructure: Isolating Untrusted Code Execution

> *"An agent that can generate Python is only useful if it can actually run it safely."*
> — Production sandboxing guidance, 2026

---

### What This Chapter Covers

Chapter 15's guardrails operate at the *decision* layer — filtering what an agent says, blocking what a tool call is allowed to attempt, requiring human approval before it proceeds. This chapter operates one layer down, at the *execution* layer: when a guardrail misses something, or when the task genuinely requires running LLM-generated code, sandboxing is what limits the actual damage a compromised or malicious execution can do. The two are complementary, not redundant — Section 16.1 makes this distinction concrete before anything else.

By the end you will understand:

1. Why sandboxing is architecturally distinct from Chapter 15's guardrails, and why production systems need both
2. The three isolation primitives underneath every sandbox provider — containers, gVisor, and Firecracker microVMs — and the concrete security/performance tradeoff each makes
3. The current managed sandbox providers (E2B, Modal, Daytona, and others), which isolation primitive each uses, and when to reach for which
4. Wiring a sandboxed code-execution tool into a LangGraph agent as an ordinary `@tool` (Chapter 9)
5. Network egress control — why it's a necessary complement to Chapter 15's content-based defenses against data exfiltration, not a redundant one
6. Filesystem isolation and resource limits — the remaining two legs of a hardened sandbox
7. Session lifecycle: ephemeral vs. persistent sandboxes, mapped onto Chapter 10's per-invocation vs. per-thread persistence framing
8. Sandboxing MCP servers themselves, not just the code an agent generates
9. MARRS checkpoint: a fully hardened code-execution tool for the capstone project

---

### 16.1 Guardrails vs. Sandboxing: Two Different Layers of Defense

Chapter 15 asked: *should this action be allowed to happen at all?* — content filtering, PII redaction, human approval gates. Sandboxing asks a different question: *if a malicious or buggy action executes anyway, how much can it actually damage?*

This distinction matters because Chapter 15's defenses can fail. A sufficiently novel prompt injection can slip past a content filter; a model-based judge can be wrong; a human reviewer, faced with dozens of routine approvals a day, can click "approve" on autopilot. Sandboxing doesn't try to prevent the bad decision — it constrains the blast radius of one that gets through. A tool that executes arbitrary Python is the sharpest version of this problem: even a fully "safe" instruction can contain a bug that deletes the wrong file or exhausts memory, with no adversary involved at all. **You need both layers.** Chapter 15's guardrails reduce how often something goes wrong; this chapter's sandboxing bounds how bad it is when something does.

---

### 16.2 Three Isolation Primitives

Every sandbox provider in production today is built on one of three underlying isolation mechanisms, each making a different tradeoff between security strength and performance:

| Primitive | How it isolates | Boot time | Isolation strength | GPU support |
|---|---|---|---|---|
| **OCI/Docker containers** | Shared host kernel, namespace/cgroup isolation | Fastest (sub-90ms achievable) | Weakest — a kernel exploit in the container reaches the host | Straightforward |
| **gVisor** | A user-space "application kernel" reimplementing roughly 200 Linux syscalls, intercepting them before they reach the real host kernel | Sub-second | Stronger than containers — the host kernel's attack surface is never directly exposed to sandboxed code | Straightforward, since it's software-based rather than hardware virtualization |
| **Firecracker microVMs** | KVM-based hardware virtualization; each sandbox gets its own dedicated kernel | ~150ms | Strongest — a full VM boundary, the same isolation model AWS Lambda runs at massive scale | Possible but more involved than the other two |

**The practical read:** containers are fast but share a kernel with the host, meaning a sufficiently severe container-escape vulnerability reaches your actual infrastructure. gVisor closes most of that gap in software, at a small performance cost, by never letting sandboxed code talk to the real kernel directly. Firecracker closes it in hardware — a genuinely separate kernel per sandbox — at the cost of a slightly slower boot. None of these is "wrong"; they represent a real, current spectrum, and the right choice depends on what you're actually running (Section 16.4).

---

### 16.3 Managed Sandbox Providers

Building and operating Firecracker or gVisor infrastructure yourself is a legitimate option at high volume, but most teams start with a managed provider. The current landscape, as of this writing:

| Provider | Isolation primitive | Notable characteristics |
|---|---|---|
| **E2B** | Firecracker microVMs | Purpose-built SDK for AI agent workflows; strongest default isolation; 24-hour maximum session length; no GPU support on the managed service |
| **Modal** | gVisor | Strong GPU support (T4 through H100/H200/B200) for agents that need inference or fine-tuning alongside code execution; Python-first design; no session time limit |
| **Daytona** | Docker containers | Fastest cold starts (sub-90ms); weakest isolation of the three, since it shares the host kernel; good fit for lower-stakes, high-throughput code-gen workloads |
| **Northflank** | Configurable (Kata/gVisor microVMs) | Bring-your-own-cloud across AWS, GCP, Azure, and others, including on-premises — the option for compliance requirements that keep sandboxed execution inside infrastructure you already control |

**Choosing among them, as a decision:**

- **Maximum security isolation, no GPU needed:** E2B — Firecracker's hardware-level separation is the strongest default available as a managed service.
- **GPU-accelerated code execution** (an agent running inference or a training step inside its sandbox): Modal — gVisor's software-based isolation doesn't block GPU passthrough the way some hypervisor-based approaches do.
- **High-throughput, lower-stakes code generation** where cold-start latency dominates the user experience: Daytona, accepting the weaker isolation tradeoff deliberately.
- **Data residency or compliance requirements** that mean sandboxed execution cannot leave infrastructure you control: Northflank or a self-hosted Firecracker/gVisor deployment.

---

### 16.4 Wiring a Sandbox Into a LangGraph Tool

The integration point is exactly Chapter 9's `@tool` — a sandbox provider's SDK becomes the tool's implementation, and everything about `ToolNode`, `handle_tool_errors`, and `RetryPolicy` from Chapter 9 applies unchanged.

```python
"""
A code-execution tool backed by a real sandbox, following Chapter 9's
@tool pattern exactly — the sandbox is an implementation detail inside
the tool function, invisible to the rest of the graph.
"""
from langchain_core.tools import tool
from e2b_code_interpreter import Sandbox

@tool
def execute_python(code: str) -> str:
    """Execute Python code in a secure, isolated sandbox and return the output.

    Args:
        code: The Python code to execute. Has access to common data science
            libraries. Does NOT have access to the host filesystem or credentials.
    """
    with Sandbox() as sandbox:   # a fresh Firecracker microVM, torn down on exit
        execution = sandbox.run_code(code)
        if execution.error:
            # Feed the error back as data, not a crash — Ch9 §9.7.2's
            # LLM-recoverable error class applies here directly: a syntax
            # error or a bad assumption is something the model can often
            # fix on its next turn, given the actual error text.
            raise ValueError(f"Execution error: {execution.error.name}: {execution.error.value}")
        return "\n".join(str(r) for r in execution.results) + (execution.logs.stdout or "")
```

```python
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

# Chapter 9's patterns apply directly: explicit handle_tool_errors, and a
# RetryPolicy tuned for a sandbox provider's own transient failures
# (cold-start timeouts, provider-side rate limits) rather than the LLM's.
sandbox_tool_node = ToolNode(
    [execute_python],
    handle_tool_errors=True,
)
sandbox_retry = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)

builder.add_node("execute_code", sandbox_tool_node, retry=sandbox_retry)
```

**The important architectural point:** from the graph's perspective, nothing changed from Chapter 9. The sandbox is entirely inside the tool's implementation. This is exactly why sandboxing composes cleanly with everything else in this book — it's a hardening decision made *inside* a tool, not a new orchestration concept requiring its own graph structure.

---

### 16.5 Network Egress Control: The Complement to Chapter 15's Content Filtering

Recall Chapter 15's indirect prompt injection scenario (§15.7): a tool result contains a hidden instruction telling the agent to exfiltrate data to an attacker-controlled endpoint. `PIIMiddleware` and content filtering try to catch this at the *content* level — recognizing that what's about to be sent looks like sensitive data. Network egress control is the complementary defense at the *infrastructure* level: even if the content check is fooled, the sandbox's network configuration simply refuses to let a connection to an unrecognized destination succeed at all.

```python
# Conceptual configuration — exact syntax varies by provider, but the
# principle is universal: default-deny network egress, then allowlist
# only what the specific tool genuinely needs to reach.
sandbox_network_policy = {
    "default": "deny",
    "allow": [
        "api.openai.com",        # if the sandboxed code itself calls an LLM
        "pypi.org",              # if package installation is needed
        "files.pythonhosted.org",
    ],
    # Notably absent: arbitrary outbound HTTP to any domain the generated
    # code decides to construct at runtime — this is exactly the channel
    # an indirect-injection-driven exfiltration attempt would try to use.
}
```

**Why this is a distinct control, not a duplicate of Chapter 15's guardrails:** a content-based guardrail has to *recognize* that something looks like an exfiltration attempt — which means it can be fooled by obfuscation, encoding, or simply not anticipating the specific pattern. A network egress allowlist doesn't need to recognize anything; it simply doesn't route traffic to destinations that were never approved, regardless of what the payload looks like or how it's encoded. This is the sandboxing-layer equivalent of Section 15.8's "excessive agency" principle: don't rely on a tool to behave — constrain what it's structurally capable of reaching.

---

### 16.6 Filesystem Isolation and Resource Limits

**Filesystem isolation:** a sandboxed code-execution tool should never have access to the host's real filesystem, credentials, or configuration files. Managed providers handle this by default — a sandbox's filesystem is its own, freshly provisioned, and torn down with the sandbox. When mounting external data *is* required (a document the agent needs to process), mount it read-only and scoped to exactly the directory needed, never the whole filesystem, following the same least-privilege principle Chapter 15 applied to tool scoping generally.

**Resource limits:** wall-clock timeouts, memory ceilings, and CPU limits prevent a runaway or maliciously-crafted execution from consuming unbounded resources — an infinite loop, a fork bomb, or a memory-exhausting allocation. Every managed provider exposes these as configuration:

```python
with Sandbox(timeout=30) as sandbox:   # hard wall-clock limit, in seconds
    execution = sandbox.run_code(code)
```

A timeout that fires mid-execution should be treated the same way Chapter 9's transient-error class is handled — caught, reported back as data (`"Execution exceeded the time limit"`), not allowed to hang the graph indefinitely. Pair a sandbox-level timeout with a `RetryPolicy` that does *not* blindly retry a timeout (Chapter 9's `retry_on` filter): a computation that timed out once at the resource limit will very likely time out again identically, and mechanically retrying it wastes the retry budget on something an automatic retry cannot fix.

---

### 16.7 Session Lifecycle: Ephemeral vs. Persistent Sandboxes

This maps directly onto Chapter 10's per-invocation vs. per-thread persistence distinction for subagents, applied to sandbox sessions instead of graph checkpoints:

| Lifecycle | When to use | Tradeoff |
|---|---|---|
| **Ephemeral** — a fresh sandbox per tool call, torn down immediately after | Independent, one-off computations — most data analysis, most "run this snippet" requests | Simplest and safest; no state to accidentally leak between calls; small latency cost to provision a fresh sandbox each time |
| **Persistent** — one long-lived sandbox reused across multiple calls in a session | A multi-step coding session where later steps genuinely depend on files or variables an earlier step created | Avoids re-provisioning latency, but requires the same session-scoping discipline Chapter 10 §10.2.2 required for per-thread subagents — one sandbox per user/thread, never shared across users |

**The security implication of choosing persistent sessions:** a long-lived sandbox accumulates state — files written, packages installed, environment variables set — across calls. If that sandbox is ever reused across different users or threads (the exact namespace-isolation mistake Chapter 10 warned against for subgraphs), one user's data or malicious code can leak into another's session. Scope persistent sandboxes to a `thread_id` with the same discipline applied to per-thread checkpointers, and default to ephemeral sandboxes unless a specific workflow genuinely needs continuity.

---

### 16.8 Sandboxing MCP Servers Themselves

Chapter 13 covered MCP as a way to connect to external tool servers — and noted that those servers run as separate processes you don't control. That cuts both ways: **an MCP server itself can be a source of risk**, not just a source of useful tools. A compromised, misconfigured, or simply buggy MCP server is functionally similar to an untrusted code-execution tool, and the same sandboxing principles apply:

- Run self-hosted MCP servers (ones you operate, rather than a well-known managed one) inside their own sandboxed environment, isolated from your agent's own execution environment and credentials.
- Apply network egress controls to what an MCP server process itself can reach, not just what your agent's generated code can reach.
- Treat an MCP server's *responses* with the same content-level scrutiny Chapter 15's `PIIMiddleware(apply_to_tool_results=True)` and Chapter 13's interceptors already apply — sandboxing constrains what a compromised server's *process* can do; content guardrails constrain what its *responses* can influence.

---

### 16.9 MARRS Checkpoint: A Hardened Code-Execution Tool

MARRS's research subagent (Chapter 10) occasionally needs to run computations — parsing a downloaded dataset, computing a statistic cited in a report. We give it a fully hardened sandboxed execution tool.

```python
"""
marrs/sandbox_tool.py

A code-execution tool for MARRS's research subagent, combining every
control from this chapter: Firecracker-level isolation, network egress
allowlisting, a wall-clock timeout, and ephemeral (not persistent) sessions
since MARRS's computations are one-off, not multi-step coding sessions.
"""
from langchain_core.tools import tool
from langgraph.types import RetryPolicy
from e2b_code_interpreter import Sandbox

# Only the data-fetching and computation packages MARRS's research
# workflow genuinely needs — no arbitrary outbound network access
ALLOWED_DOMAINS = ["pypi.org", "files.pythonhosted.org"]

@tool
def run_analysis_code(code: str) -> str:
    """Execute Python data-analysis code in a secure sandbox. Has access to
    pandas, numpy, and matplotlib. Has NO access to the host filesystem,
    credentials, or arbitrary network destinations.

    Args:
        code: Python code to execute. Assign a final result to a variable
            named `result` to have it returned.
    """
    with Sandbox(timeout=30) as sandbox:   # ephemeral — Section 16.7; 30s hard limit
        # Real provider configuration would set network policy at sandbox
        # creation; shown conceptually here per Section 16.5.
        execution = sandbox.run_code(code)
        if execution.error:
            raise ValueError(
                f"Execution error: {execution.error.name}: {execution.error.value}. "
                f"Review the code and try again with corrected logic."
            )
        return execution.logs.stdout or "Code executed with no output."


# handle_tool_errors=True (Ch9 §9.4.4) so a bad snippet becomes feedback
# the model can act on, not a crashed graph run.
from langgraph.prebuilt import ToolNode
sandbox_tool_node = ToolNode([run_analysis_code], handle_tool_errors=True)

# retry_on deliberately EXCLUDES timeout errors — a computation that hit
# the 30s wall clock once will hit it again; retrying wastes the budget.
def is_transient(error: Exception) -> bool:
    return "timeout" not in str(error).lower()

sandbox_retry = RetryPolicy(max_attempts=2, initial_interval=1.0, retry_on=is_transient)
```

```python
# Wired into the research subagent from Chapter 10, alongside its existing
# web_search and fetch_academic_paper tools (Chapter 9's RESEARCH_TOOLS)
research_subagent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[web_search, fetch_academic_paper, run_analysis_code],
    system_prompt=(
        "You are MARRS's research specialist. For any computation or data "
        "analysis, use run_analysis_code rather than reasoning through "
        "arithmetic yourself — it runs in a secure sandbox with no access "
        "to sensitive systems, so it is safe to use freely for this purpose."
    ),
)
```

---

### 16.10 Chapter Summary

**Sandboxing and guardrails are complementary, not redundant.** Chapter 15's guardrails try to prevent a bad decision from being made; sandboxing bounds the damage when one gets through anyway, or when a bug — with no adversary involved — causes unintended behavior.

**Three isolation primitives** underlie every sandbox provider: Docker/OCI containers (fastest, weakest — shared kernel), gVisor (a user-space kernel reimplementing syscalls, stronger isolation with good GPU support), and Firecracker microVMs (strongest — a genuinely separate kernel per sandbox, at a small boot-time cost). **Managed providers** map onto this spectrum — E2B (Firecracker, strongest default isolation), Modal (gVisor, strong GPU support), Daytona (containers, fastest cold starts), Northflank (configurable, bring-your-own-cloud) — and the right choice depends on your GPU needs, isolation requirements, and data-residency constraints.

**Wiring a sandbox into LangGraph** is just Chapter 9's `@tool` pattern — the sandbox lives entirely inside the tool's implementation, with `handle_tool_errors` and a tuned `RetryPolicy` (explicitly excluding timeouts from the retry-eligible set) applying exactly as Chapter 9 taught.

**Network egress control** is the infrastructure-level complement to Chapter 15's content-based guardrails against exfiltration — a default-deny allowlist doesn't need to *recognize* an exfiltration attempt to stop it, unlike a content filter that can be fooled by obfuscation.

**Session lifecycle** (ephemeral vs. persistent) mirrors Chapter 10's per-invocation vs. per-thread subagent persistence distinction, with the same namespace-isolation discipline required if you choose persistent sessions.

**MCP servers themselves** (Chapter 13) deserve the same sandboxing scrutiny as any other untrusted code path, since a self-hosted MCP server is, from a security standpoint, functionally similar to any other process you don't fully control.

---

### Further Reading

- **E2B documentation and LangChain integration guide**: `docs.langchain.com/oss/python/integrations/providers/e2b` and `e2b.dev/blog/langgraph-with-code-interpreter-guide-with-code` — the exact `@tool` wiring pattern this chapter builds on
- **"Best Code Execution Sandbox for AI Agents in 2026"** — comparative analysis of E2B, Modal, Daytona, and Northflank's isolation primitives, session limits, and GPU support
- **"AI Agent Sandboxing in 2026: Docker, E2B, Firecracker, gVisor, Modal & Daytona Compared"** — a detailed technical comparison of the three underlying isolation mechanisms
- **Firecracker documentation**: `firecracker-microvm.github.io` — the microVM technology underlying E2B and AWS Lambda's own isolation model
- **gVisor documentation**: `gvisor.dev` — Google's user-space kernel, underlying Modal's isolation approach
- **E2B Cookbook**: `github.com/e2b-dev/e2b-cookbook` — worked examples across LangGraph, AutoGen, and MCP-connected sandboxes

---

*End of Chapter 16.*
