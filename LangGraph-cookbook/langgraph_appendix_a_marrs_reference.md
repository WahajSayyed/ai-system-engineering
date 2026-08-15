# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Appendix A — The Complete MARRS Reference Implementation

### Purpose of This Appendix

MARRS (Multi-Agent Research & Report Writing System) was built incrementally across Chapters 2 through 12, one mechanism at a time — state schemas in Chapter 3, control flow in Chapter 4, persistence in Chapter 5, and so on through deployment in Chapter 12. That's the right way to *learn* the mechanisms. It's the wrong way to *run* the system, since the working code is scattered across eleven chapters with earlier versions superseded by later ones.

This appendix assembles the final, coherent version into one place. Every section below is labeled with the chapter that introduced it, so you can trace any piece of code back to the explanation of *why* it works that way. Two design decisions evolved over the course of the book and are worth naming up front rather than silently papering over:

1. **The researcher evolved from a placeholder to a real tool-calling agent.** Chapters 2–4 used a hard-coded placeholder that returned canned strings. Chapter 4 additionally demonstrated a `Send`-based fan-out to N parallel search workers as an alternative dispatch strategy. Chapter 9 replaced the placeholder with a genuine tool-calling `researcher_agent_node` + `ToolNode` loop, calling real (if illustrative) search tools. **This appendix uses the Chapter 9 tool-calling version as canonical**, since it's what the book explicitly describes as superseding the earlier placeholder. The `Send`-based fan-out pattern remains a valid alternative — Section A.9 shows where it would slot in if your research step is better modeled as "run N independent searches in parallel" rather than "let an agent decide which tools to call."

2. **The overall architecture branched in two directions in Chapter 10.** Chapters 2–9 build MARRS as one hand-assembled `StateGraph` with explicit nodes for planning, research, writing, critique, and human review. Chapter 10 then restructured the same problem as a **Subagents ("supervisor") architecture** — a coordinating `create_agent` delegating to a researcher subagent and a writer subagent, each wrapped as tools. **Both are legitimate, complete implementations of MARRS.** This appendix presents the hand-built `StateGraph` version in full (Sections A.1–A.8) because it's the version that exercises every mechanism the book teaches chapter by chapter — reducers, `Command`, `Send`, `interrupt()`, streaming instrumentation, structured output, and classified error handling all appear in one graph. Section A.10 gives the condensed supervisor-architecture equivalent from Chapter 10 side by side, so you can see the same responsibilities expressed both ways and choose deliberately for your own projects, guided by Chapter 10's Section 10.4.1 performance comparison.

---

### A.1 State Schema (Chapters 3, 6, 7, 9)

```python
"""
marrs/state.py

The MARRS state schema, final consolidated form.

Field provenance:
  Chapter 2 — topic, plan, draft, status (skeleton)
  Chapter 3 — findings/sources as operator.add reducers, input/output schema split
  Chapter 4 — search_queries, quality_score, revision_count (Command-based critic)
  Chapter 6 — summary (rolling conversation compression)
  Chapter 7 — human_decision, human_feedback, review_count (HITL gate)
  Chapter 9 — validation_errors (classified user-fixable errors)
"""
import operator
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MARRSState(TypedDict, total=False):
    # Core research task (Ch2)
    topic: str
    plan: str
    search_queries: list[str]                                  # Ch4

    # Accumulated research output — reducers merge parallel/repeated writes (Ch3)
    findings: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]

    # Draft and critique cycle (Ch2, Ch4, Ch9)
    draft: str
    critique: str
    quality_score: float
    revision_count: int

    # Conversation history for the tool-calling researcher agent (Ch9)
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str                                                # Ch6 — rolling compression

    # Human-in-the-loop gate (Ch7)
    human_decision: str        # "approve" | "revise" | "auto_approved"
    human_feedback: str
    review_count: int

    # Classified error tracking (Ch9)
    validation_errors: list[str]

    # Final output
    final_report: str
    status: str
```

---

### A.2 Tools (Chapter 9)

```python
"""
marrs/tools.py

Real tools for the research agent, with input validation that raises on bad
arguments — this is what makes the LLM-recoverable error class (Ch9 §9.7.2)
meaningful: a malformed argument produces a ValueError the LLM can see and
correct on its next turn, rather than crashing the graph.
"""
import re
from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information on a topic.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (1-10).
    """
    if not query.strip():
        raise ValueError("Search query cannot be empty. Provide a specific topic to search.")
    # Placeholder for a real search API (Tavily, SerpAPI, etc.) — swap in production.
    return f"[{max_results} results for '{query}']: relevant findings about {query}..."


@tool
def fetch_academic_paper(arxiv_id: str) -> str:
    """Fetch the abstract of an academic paper by its arXiv ID.

    Args:
        arxiv_id: The arXiv identifier, e.g. '2301.00234'.
    """
    if not re.match(r"^\d{4}\.\d{4,5}$", arxiv_id):
        raise ValueError(f"'{arxiv_id}' is not a valid arXiv ID format (expected e.g. '2301.00234').")
    return f"[Abstract for arXiv:{arxiv_id}]: This paper investigates..."


RESEARCH_TOOLS = [web_search, fetch_academic_paper]
```

---

### A.3 Long-Term Memory Store (Chapter 6)

```python
"""
marrs/memory.py

Long-term, cross-thread memory: episodic (past successful research approaches,
used as few-shot guidance for the planner) and semantic (individual findings,
retrievable by future runs on related topics).
"""
import json
import uuid
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

research_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text"],
    }
)


def retrieve_similar_episodes(topic: str, store, limit: int = 3) -> list[dict]:
    """Find past successful research episodes similar to the current topic (Ch6 §6.7.2)."""
    results = store.search(("research", "episodes"), query=f"Research on: {topic}", limit=limit)
    return [r.value for r in results if r.score and r.score > 0.75]


def write_research_memory(state: MARRSState, config, *, store) -> dict:
    """
    Write to long-term memory after a successful run: individual findings as
    semantic memory (one fact per item, per Ch6 §6.7.1's design principle),
    and the overall approach as an episodic memory for future few-shot use.
    Only runs when quality is high enough to be worth remembering (Ch6 §6.8).
    """
    score = state.get("quality_score", 0.0)
    topic = state.get("topic", "")
    if score < 0.80 or not topic:
        return {}

    for finding in state.get("findings", [])[:5]:
        if finding and len(finding) > 30:
            store.put(
                ("research", "topic_facts"),
                str(uuid.uuid4()),
                {"text": finding[:500], "topic": topic},
                index=["text"],
            )

    store.put(
        ("research", "episodes"),
        str(uuid.uuid4()),
        {
            "input": topic,
            "plan": state.get("plan", ""),
            "quality_score": score,
            "text": f"Research on: {topic}. Plan: {state.get('plan', '')}",
        },
        index=["text"],
    )
    return {}
```

---

### A.4 Nodes (Chapters 4, 6, 7, 8, 9)

```python
"""
marrs/nodes.py

Every node function, each annotated with the chapter(s) that shaped it.
get_stream_writer() calls throughout are Chapter 8's progress-streaming
instrumentation — they cost nothing when no one is listening on
stream_mode="custom", and give real-time status when someone is.
"""
import json
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from .memory import retrieve_similar_episodes


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER (Ch6 — memory-enhanced with episodic few-shot retrieval)
# ─────────────────────────────────────────────────────────────────────────────

def planner_with_memory(state: MARRSState, config, *, store) -> dict:
    writer = get_stream_writer()                                        # Ch8
    writer({"status": f"Planning research for '{state['topic']}'...", "progress": 0.05})

    topic = state.get("topic", "")
    past_episodes = retrieve_similar_episodes(topic, store)              # Ch6

    few_shot_context = ""
    if past_episodes:
        few_shot_context = "\n\nSuccessful past research approaches:\n" + "\n".join(
            f"- {ep.get('input', 'N/A')}: {ep.get('plan', 'N/A')[:150]} (quality {ep.get('quality_score', 0):.2f})"
            for ep in past_episodes
        )

    summary = state.get("summary", "")
    summary_context = f"\nConversation summary:\n{summary}\n" if summary else ""

    plan_prompt = (
        f"Create a research plan for: {topic}{summary_context}{few_shot_context}\n\n"
        f'Output JSON: {{"summary": "one-line overview", "search_queries": ["q1", "q2", ...]}}'
    )
    response = llm.invoke([SystemMessage(content=plan_prompt)])
    try:
        plan_data = json.loads(response.content)
        plan, queries = plan_data.get("summary", ""), plan_data.get("search_queries", [])
    except (json.JSONDecodeError, AttributeError):
        plan, queries = f"Research plan for {topic}", [f"{topic} overview"]

    writer({"status": f"Generated {len(queries)} search queries", "progress": 0.15})
    return {"plan": plan, "search_queries": queries}


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCHER — tool-calling agent loop (Ch9, superseding the Ch2-4 placeholder)
# ─────────────────────────────────────────────────────────────────────────────

def researcher_agent_node(state: MARRSState) -> dict:
    """
    LLM decides which research tools to call from RESEARCH_TOOLS (App. A.2).
    Transient tool failures are handled by RetryPolicy on the 'tools' node
    (Ch9 §9.7.1); malformed arguments are handled by handle_tool_errors
    inside ToolNode (Ch9 §9.7.2). This node itself just binds tools and calls
    the model — tools_condition (Ch4, Ch9) does the routing.
    """
    writer = get_stream_writer()
    writer({"status": "Researching...", "progress": 0.30})
    llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def collect_findings_node(state: MARRSState) -> dict:
    """
    After the researcher's tool loop exits (tools_condition routes to here),
    fold ToolMessage contents into the accumulated findings/sources fields
    that the writer and critic actually read.
    """
    writer = get_stream_writer()
    tool_messages = [m for m in state.get("messages", []) if m.__class__.__name__ == "ToolMessage"]
    findings = [m.content for m in tool_messages]
    writer({"status": f"Collected {len(findings)} findings", "progress": 0.55})
    return {"findings": findings, "sources": [f"tool-result-{i}" for i in range(len(findings))]}


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZATION (Ch6 — bounding message-history growth across long sessions)
# ─────────────────────────────────────────────────────────────────────────────

SUMMARIZE_AFTER_N_MESSAGES = 20

def maybe_summarize_messages(state: MARRSState) -> dict:
    messages = state.get("messages", [])
    if len(messages) < SUMMARIZE_AFTER_N_MESSAGES:
        return {}

    existing_summary = state.get("summary", "")
    to_summarize, keep = messages[:-6], messages[-6:]
    if not to_summarize:
        return {}

    history_text = "\n".join(f"{m.__class__.__name__}: {str(m.content)[:300]}" for m in to_summarize)
    prompt = (
        f"Existing summary:\n{existing_summary}\n\nNew messages:\n{history_text}\n\nUpdate the summary concisely."
        if existing_summary else
        f"Summarize this research conversation concisely:\n{history_text}"
    )
    new_summary = llm.invoke([SystemMessage(content=prompt)]).content
    return {"summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in to_summarize]}


# ─────────────────────────────────────────────────────────────────────────────
# WRITER (Ch2, streamed token-by-token per Ch8)
# ─────────────────────────────────────────────────────────────────────────────

def writer_node(state: MARRSState) -> dict:
    writer = get_stream_writer()
    writer({"status": "Writing draft...", "progress": 0.70})
    findings_text = "\n".join(state.get("findings", []))
    prompt = f"Write a research report on {state['topic']} based on:\n{findings_text}"
    # This LLM call's tokens are automatically captured by stream_mode="messages"
    # when the graph is invoked via .astream(..., stream_mode=["messages", ...]) — Ch8 §8.2.3
    response = llm.invoke(prompt)
    writer({"status": "Draft complete", "progress": 0.85})
    return {"draft": response.content}


# ─────────────────────────────────────────────────────────────────────────────
# CRITIC — structured output + Command-based routing (Ch4, Ch9)
# ─────────────────────────────────────────────────────────────────────────────

class CriticEvaluation(BaseModel):
    """Structured evaluation of a research draft (Ch9 §9.5.1)."""
    quality_score: float = Field(description="Overall quality from 0.0 to 1.0")
    strengths: list[str] = Field(description="What the draft does well")
    weaknesses: list[str] = Field(description="Specific issues that need fixing")
    recommendation: Literal["approve", "revise"] = Field(description="Next step")


MAX_REVISIONS = 3

def critic_node(state: MARRSState) -> Command[Literal["writer", "human_review"]]:
    """
    Command combines the structured evaluation (Ch9) with routing (Ch4 §4.5) —
    the score computed here drives the very next superstep's destination
    without a second, separate conditional-edge read of stale state.
    """
    writer = get_stream_writer()
    writer({"status": "Evaluating draft quality...", "progress": 0.90})

    draft, revisions = state.get("draft", ""), state.get("revision_count", 0)
    critic_llm = llm.with_structured_output(CriticEvaluation)
    evaluation = critic_llm.invoke(f"Evaluate this research draft on {state.get('topic', '')}:\n\n{draft}")

    new_revision_count = revisions + 1
    destination = (
        "human_review" if evaluation.recommendation == "approve" or new_revision_count >= MAX_REVISIONS
        else "writer"
    )
    writer({"status": f"Quality score: {evaluation.quality_score:.2f}", "progress": 0.95})

    return Command(
        update={
            "critique": "Strengths: " + "; ".join(evaluation.strengths)
                        + "\nWeaknesses: " + "; ".join(evaluation.weaknesses),
            "quality_score": evaluation.quality_score,
            "revision_count": new_revision_count,
        },
        goto=destination,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN REVIEW — conditional interrupt() gate (Ch7)
# ─────────────────────────────────────────────────────────────────────────────

AUTO_APPROVE_THRESHOLD = 0.92

def human_review_node(state: MARRSState) -> Command[Literal["finalize", "writer"]]:
    """
    Conditional HITL (Ch7 §7.6.6): skip the interrupt entirely for
    high-confidence drafts; only pause a human for borderline quality.
    interrupt() is the very first side-effecting statement reached in the
    borderline branch — nothing before it can double-execute on resume
    (Ch7 §7.3.4's idempotency rule).
    """
    quality = state.get("quality_score", 0.0)
    review_count = state.get("review_count", 0) + 1

    if quality >= AUTO_APPROVE_THRESHOLD:
        return Command(goto="finalize", update={"human_decision": "auto_approved",
                                                  "review_count": review_count, "status": "approved"})

    decision = interrupt({
        "message": f"Draft quality is {quality:.2f} (borderline). Please review.",
        "draft": state.get("draft", ""),
        "quality_score": quality,
        "sources": state.get("sources", [])[:5],
        "instructions": "Reply 'approve' or 'revise: <feedback>'",
    })

    if decision == "approve":
        return Command(goto="finalize",
                        update={"human_decision": "approve", "review_count": review_count, "status": "approved"})

    feedback = decision[len("revise:"):].strip() if isinstance(decision, str) and decision.startswith("revise:") else str(decision)
    return Command(goto="writer", update={
        "human_decision": "revise", "human_feedback": feedback,
        "critique": f"Human reviewer: {feedback}", "review_count": review_count, "status": "in_revision",
    })


# ─────────────────────────────────────────────────────────────────────────────
# FINALIZE
# ─────────────────────────────────────────────────────────────────────────────

def finalize_node(state: MARRSState) -> dict:
    writer = get_stream_writer()
    writer({"status": "Finalizing report...", "progress": 1.0})
    return {
        "final_report": f"# {state.get('topic', 'Report')}\n\n{state.get('draft', '')}",
        "status": "complete",
    }
```

---

### A.5 Routing Functions (Chapters 4, 9)

```python
"""
marrs/routing.py

Every routing function here follows the three rules from Ch4 §4.3:
.get() with defaults, a guaranteed termination path, and no side effects.
"""
from typing import Literal


def route_after_researcher(state: MARRSState) -> Literal["tools", "collect_findings"]:
    """tools_condition-equivalent, but routes to collect_findings instead of END (Ch4, Ch9)."""
    last = state.get("messages", [None])[-1]
    if last is not None and getattr(last, "tool_calls", None):
        return "tools"
    return "collect_findings"
```

---

### A.6 Graph Assembly (Chapters 2, 4, 5, 6, 9)

```python
"""
marrs/graph.py

Final graph assembly. RetryPolicy placement follows the four-class error
framework from Ch9 §9.7: the 'researcher' and 'tools' nodes get RetryPolicy
(transient failures); 'human_review' deliberately gets none, since a
borderline-quality draft won't become non-borderline no matter how many
times you retry evaluating it — that's a user-fixable problem (interrupt()),
not a transient one.
"""
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import MARRSState
from .tools import RESEARCH_TOOLS
from .nodes import (
    planner_with_memory, researcher_agent_node, collect_findings_node,
    maybe_summarize_messages, writer_node, critic_node, human_review_node, finalize_node,
)
from .memory import research_store, write_research_memory
from .routing import route_after_researcher


llm_retry = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0, jitter=True)   # Ch9 §9.7.1
api_retry = RetryPolicy(max_attempts=5, initial_interval=1.0, backoff_factor=2.0, jitter=True)


def build_marrs(checkpointer=None, store=None):
    builder = StateGraph(MARRSState)

    builder.add_node("planner", planner_with_memory, retry=llm_retry)
    builder.add_node("researcher", researcher_agent_node, retry=llm_retry)
    builder.add_node("tools", ToolNode(RESEARCH_TOOLS, handle_tool_errors=True), retry=api_retry)  # Ch9 §9.4.4
    builder.add_node("collect_findings", collect_findings_node)
    builder.add_node("summarize", maybe_summarize_messages)
    builder.add_node("writer", writer_node, retry=llm_retry)
    builder.add_node("critic", critic_node, retry=llm_retry)
    builder.add_node("human_review", human_review_node)              # NO retry — see module docstring
    builder.add_node("finalize", finalize_node)
    builder.add_node("write_memory", write_research_memory)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_conditional_edges("researcher", route_after_researcher,
                                   {"tools": "tools", "collect_findings": "collect_findings"})
    builder.add_edge("tools", "researcher")                          # ReAct loop back-edge (Ch4)
    builder.add_edge("collect_findings", "summarize")
    builder.add_edge("summarize", "writer")
    builder.add_edge("writer", "critic")
    # critic and human_review route via Command — no add_conditional_edges needed for them (Ch4 §4.5)
    builder.add_edge("finalize", "write_memory")
    builder.add_edge("write_memory", END)

    return builder.compile(
        checkpointer=checkpointer or SqliteSaver(sqlite3.connect("marrs.db", check_same_thread=False)),
        store=store or research_store,
    )


marrs = build_marrs()
```

---

### A.7 Running MARRS

```python
"""
marrs/run.py

A complete session: dispatch, detect the pause, resume with a decision,
and stream progress the whole way — every mechanism from Ch5-Ch8 in one call.
"""
from langgraph.types import Command


async def run_marrs_session(topic: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"topic": topic, "messages": [], "findings": [], "sources": []}

    async for mode, chunk in marrs.astream(input_data, config, stream_mode=["custom", "updates"]):
        if mode == "custom":
            print(f"[{chunk.get('progress', 0)*100:5.1f}%] {chunk['status']}")
        elif mode == "updates" and "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
            print(f"\n⏸  PAUSED FOR REVIEW: {payload['message']}")
            print(f"   Draft preview: {payload['draft'][:200]}...")

            decision = input("Your decision (approve / revise: <feedback>): ")   # Ch7 §7.9 — swap for a real UI/webhook
            async for mode2, chunk2 in marrs.astream(Command(resume=decision), config, stream_mode=["custom", "updates"]):
                if mode2 == "custom":
                    print(f"[{chunk2.get('progress', 0)*100:5.1f}%] {chunk2['status']}")

    final_state = marrs.get_state(config).values
    print(f"\n✅ Status: {final_state.get('status')}")
    return final_state.get("final_report")
```

---

### A.8 Test Suite (Chapter 11)

```python
"""
marrs/test_marrs.py

The three-tier strategy from Ch11: fast unit tests with a fake model,
integration tests against the real model, and trajectory evals.
"""
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langgraph.checkpoint.memory import InMemorySaver


def test_researcher_calls_tool():
    """Unit test (Ch11 §11.2): scripted fake model, no network."""
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[ToolCall(name="web_search", args={"query": "AI safety"}, id="c1")]),
        AIMessage(content="Based on the results, AI safety research focuses on..."),
    ]))
    test_graph = build_marrs(checkpointer=InMemorySaver())   # inject fake model via your test factory
    config = {"configurable": {"thread_id": "unit-1"}}
    result = test_graph.invoke({"topic": "AI safety", "messages": [], "findings": [], "sources": []}, config)
    assert result.get("status") in ("complete", "in_revision", "approved")


def test_critic_routing_pure_logic():
    """Unit test: routing decisions, no model or graph needed at all."""
    from marrs.nodes import AUTO_APPROVE_THRESHOLD
    assert 0.85 < AUTO_APPROVE_THRESHOLD


@pytest.mark.integration
def test_marrs_end_to_end():
    """Integration test (Ch11 §11.3): real model, structural assertions only."""
    config = {"configurable": {"thread_id": "integration-1"}}
    result = marrs.invoke({"topic": "quantum error correction", "messages": [], "findings": [], "sources": []}, config)
    assert "final_report" in result or "__interrupt__" in result
```

---

### A.9 Where the `Send`-Based Fan-Out Would Slot In (Chapter 4)

If your research step is better modeled as "run N independent searches in parallel" rather than "let one agent decide which tools to call across several turns," Chapter 4's `Send`-based dispatch (§4.6) is a drop-in alternative to `researcher_agent_node` + `tools` + `collect_findings`:

```python
from langgraph.types import Send

def dispatch_searches(state: MARRSState) -> list[Send]:
    return [Send("search_one", {"query": q, "topic": state["topic"]}) for q in state.get("search_queries", [])]

def search_one_node(state: dict) -> dict:
    # Runs once per query, in parallel, with its own private state (Ch4 §4.6.2)
    return {"findings": [f"Finding for '{state['query']}'"], "sources": [f"src:{state['query']}"]}
```

Swap the `add_conditional_edges("researcher", route_after_researcher, ...)` wiring in Section A.6 for `builder.add_conditional_edges("planner", dispatch_searches, ["search_one"])` and route `search_one` directly to `summarize`. This trades the tool-calling agent's flexibility (it can decide *which* tools to call and adapt mid-research) for the `Send` fan-out's simplicity and guaranteed parallelism when the query list is already fully known up front.

---

### A.10 The Alternative: Subagents Architecture (Chapter 10)

Chapter 10 restructures the same problem entirely differently — a coordinating agent delegating to specialist subagents wrapped as tools, rather than one hand-assembled graph:

```python
from langchain.agents import create_agent
from langchain.tools import tool

research_subagent = create_agent(
    model="anthropic:claude-sonnet-5", tools=RESEARCH_TOOLS,
    system_prompt="You are MARRS's research specialist.",
)
writer_subagent = create_agent(
    model="anthropic:claude-sonnet-5", tools=[],
    system_prompt="You are MARRS's writing specialist.",
)

@tool
def delegate_research(topic: str) -> str:
    """Delegate a research task to the research specialist."""
    return research_subagent.invoke({"messages": [{"role": "user", "content": f"Research: {topic}"}]})["messages"][-1].content

@tool
def delegate_writing(findings: str) -> str:
    """Delegate report writing to the writing specialist."""
    return writer_subagent.invoke({"messages": [{"role": "user", "content": f"Write a report based on: {findings}"}]})["messages"][-1].content

marrs_supervisor = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[delegate_research, delegate_writing],
    system_prompt="Coordinate MARRS reports. Delegate research and writing; never do either yourself.",
    checkpointer=SqliteSaver(sqlite3.connect("marrs_supervisor.db", check_same_thread=False)),
)
```

**Which to choose:** the hand-built `StateGraph` (Sections A.1–A.8) gives you explicit control over every routing decision, the exact `Command`/`Send`/`interrupt()` mechanics from this book, and — per Chapter 10's §10.4.1 measurements — better performance on genuinely parallel, multi-domain work. The Subagents version is less code, easier to extend with new specialists, and stateless-by-default isolation between delegated tasks, at the cost of an extra model round-trip per delegation and no built-in `Send`-style parallel fan-out. Neither is strictly better; Chapter 10 exists to help you make that call deliberately rather than by default.

---

### Further Reading

This appendix consolidates code already grounded and cited in the chapters that introduced it. Refer back to:
- Chapter 3 (state/reducers), Chapter 4 (control flow), Chapter 5 (persistence), Chapter 6 (memory) for the mechanisms behind Sections A.1–A.3
- Chapter 7 (HITL), Chapter 8 (streaming), Chapter 9 (tools, structured output, error handling) for Sections A.4–A.5
- Chapter 10 (multi-agent architectures) for Section A.10
- Chapter 11 (testing) for Section A.8
- Chapter 12 (deployment) for packaging any of the above as a deployable `langgraph.json` project

---

*End of Appendix A.*
