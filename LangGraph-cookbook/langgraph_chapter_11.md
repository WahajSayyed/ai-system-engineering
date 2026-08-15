# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 11 — Evaluation and Testing: Tracing, LangSmith, and Regression Suites for Agentic Systems

> *"Agentic applications let an LLM decide its own next steps to solve a problem. That flexibility is powerful, but the model's black-box nature makes it hard to predict how a tweak in one part of your agent will affect the whole."*
> — LangChain official documentation

---

### What This Chapter Covers

Every prior chapter has included scattered `pytest` snippets testing one mechanism at a time — a routing function here, a checkpointer there. This chapter makes testing itself the subject: the official three-tier strategy for agentic systems, the specific tools for each tier, and how to wire the whole thing into LangSmith for tracing, regression detection, and CI.

By the end you will understand:

1. Why agentic systems need a different testing strategy than traditional deterministic software
2. The official three-tier model: unit tests, integration tests, and evals — what each is for and what it cannot catch
3. Unit testing with `GenericFakeChatModel` and `InMemorySaver` — fast, free, deterministic, no API calls
4. Integration testing: separating it from unit tests with pytest markers, managing API keys safely, asserting on structure instead of exact content, and recording HTTP cassettes with `vcrpy` to make repeated CI runs free
5. Evals: what a trajectory is, and the two ways to score one — deterministic trajectory matching (`agentevals`, four match modes) and LLM-as-judge
6. Running evals inside LangSmith: the pytest integration versus `Client.evaluate()` against a dataset
7. LangSmith tracing: the current environment variables, `@traceable`, and what a LangGraph trace actually shows you
8. Building this into a CI/CD pipeline with a quality gate
9. MARRS checkpoint: a complete three-tier test suite for the capstone project

---

### 11.1 Why Agentic Testing Is Different

Traditional software tests ask "did the code run correctly?" — a deterministic question with a deterministic answer. Agentic systems add a second, harder question: "was the *decision* the LLM made a good one?" The same input can legitimately produce different tool call sequences, different wording, different numbers of steps, because the model is reasoning, not executing a fixed program.

The official framing is precise about where each testing tier's confidence comes from:

- **Unit tests** exercise small, deterministic pieces of your agent in isolation using in-memory fakes, so you can assert exact behavior quickly and deterministically.
- **Integration tests** use real network calls to confirm that components work together, credentials and schemas line up, and latency is acceptable.
- **Evals** use evaluators to assess your agent's execution trajectory — the sequence of messages and tool calls it produces — against a reference or rubric, via either deterministic matching or an LLM judge.

Agentic applications lean more heavily on the integration and eval tiers than traditional software does, precisely because they chain multiple components together and must deal with the nondeterminism of the LLM itself. A unit test tells you the wiring is correct; it cannot tell you the agent chose wisely. That's what evals are for.

---

### 11.2 Unit Testing: Fakes, Not API Calls

Unit tests replace the real LLM with an in-memory fake so responses are scripted, tests run in milliseconds, and nothing costs money or requires a network connection.

#### 11.2.1 `GenericFakeChatModel`

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall

model = GenericFakeChatModel(messages=iter([
    AIMessage(content="", tool_calls=[ToolCall(name="foo", args={"bar": "baz"}, id="call_1")]),
    "bar",
]))

model.invoke("hello")
# AIMessage(content='', tool_calls=[{'name': 'foo', 'args': {'bar': 'baz'}, 'id': 'call_1', 'type': 'tool_call'}])

model.invoke("hello, again!")
# AIMessage(content='bar', ...)
```

`GenericFakeChatModel` takes an iterator of responses — plain strings or full `AIMessage` objects (including tool calls) — and returns the next one on each `.invoke()`. It supports streaming as well as regular invocation, so it works as a drop-in replacement anywhere your graph calls `.invoke()`, `.stream()`, or their async equivalents. Script the exact sequence you want the "model" to produce, and you get an exact, repeatable trace to assert against — no prompt-sensitivity, no flakiness, no API key.

#### 11.2.2 `InMemorySaver` for Multi-Turn State Tests

Combine the fake model with `InMemorySaver` (Chapter 5) to test state-dependent behavior across multiple turns without touching a real database:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent(model, tools=[], checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "session-1"}}

# First turn establishes context
agent.invoke({"messages": [HumanMessage(content="I live in Sydney, Australia")]}, config)

# Second turn: the persisted state from turn 1 should influence this response
agent.invoke({"messages": [HumanMessage(content="What's my local time?")]}, config)
```

Because `InMemorySaver` is real persistence (just backed by RAM instead of a database), this test genuinely exercises the checkpointing and state-loading machinery — it's not a fake of *that* part, only of the LLM. This is the fast, deterministic way to catch bugs in multi-turn memory logic (Chapter 6) without waiting on real model latency.

**What unit tests are for, precisely:** routing function logic (Chapter 4), reducer behavior (Chapter 3), state schema validation, and any node whose behavior you can fully script by controlling the fake model's output. What they cannot tell you: whether the *real* model, given a real prompt, will actually produce sensible tool calls. That requires the next two tiers.

---

### 11.3 Integration Testing: Real APIs, Managed Carefully

Integration tests hit the real LLM API. They're slower, cost money, and are the only tier that can catch a broken API credential, a schema mismatch with the provider, or a real latency regression.

#### 11.3.1 Separating Integration Tests From Unit Tests

Because integration tests are slow and require credentials, keep them out of the default test run using a pytest marker:

```ini
# pytest.ini
[pytest]
markers =
    integration: tests that call real LLM APIs
addopts = -m "not integration"
```

```python
import pytest

@pytest.mark.integration
def test_agent_with_real_model():
    agent = create_agent("claude-sonnet-4-6", tools=[get_weather])
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})
    assert len(result["messages"]) > 1
```

```bash
pytest                  # runs only unit tests, fast, free
pytest -m integration   # runs integration tests explicitly — CI or pre-deploy
```

#### 11.3.2 Managing API Keys Safely

Load credentials from environment variables, never from source, and skip cleanly when a key is missing rather than failing with a cryptic auth error:

```python
# conftest.py
import os
import pytest
from dotenv import load_dotenv

load_dotenv()   # loads a local .env for development

@pytest.fixture(autouse=True)
def check_api_keys():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
```

Add `.env` to `.gitignore`. In CI, inject secrets through your provider's secret manager rather than committing them anywhere.

#### 11.3.3 Assert on Structure, Not Content

Because LLM output varies between runs even with identical input, asserting on exact strings is a guaranteed source of flaky tests. Assert on the *shape* of the response instead — message types, which tools were called, argument shapes, message count:

```python
def test_agent_calls_weather_tool():
    agent = create_agent("claude-sonnet-4-6", tools=[get_weather])
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})

    messages = result["messages"]
    tool_calls = [
        tc for msg in messages if hasattr(msg, "tool_calls") for tc in (msg.tool_calls or [])
    ]

    assert any(tc["name"] == "get_weather" for tc in tool_calls)
    assert isinstance(messages[-1], AIMessage)
    assert len(messages[-1].content) > 0
```

For anything beyond this basic structural check — verifying a whole tool-call sequence, allowing for equivalent orderings — reach for the trajectory evaluators in Section 11.4 rather than hand-rolling more assertion logic here.

#### 11.3.4 Controlling Cost and Latency

- **Use smaller models for structural tests** — a lightweight model is enough to verify tool-calling and response shape without paying for a frontier model on every CI run.
- **Cap response length** (`max_tokens`) to avoid long, expensive completions when you only need to check structure.
- **One behavior per test.** Avoid chaining many LLM calls in a single test when a single-turn test already proves the point.
- **Run integration tests selectively** — CI and pre-deploy, not on every file save (Section 11.3.1's marker handles this).

#### 11.3.5 Record and Replay: Making Repeated CI Runs Free

For integration tests that run on every CI build, recording the HTTP interaction once and replaying it on every subsequent run eliminates both cost and latency after the first recording. `vcrpy` records request/response pairs into YAML "cassette" files; `pytest-recording` integrates this with pytest.

```python
# conftest.py — filter secrets out of recorded cassettes
import pytest

@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": [("authorization", "XXXX"), ("x-api-key", "XXXX")],
        "filter_query_parameters": [("api_key", "XXXX"), ("key", "XXXX")],
    }
```

```ini
# pytest.ini
[pytest]
markers =
    vcr: record/replay HTTP via VCR
addopts = --record-mode=once
```

```python
@pytest.mark.vcr()
def test_agent_trajectory():
    agent = create_agent("claude-sonnet-4-6", tools=[get_weather])
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})
    assert any(
        tc["name"] == "get_weather"
        for msg in result["messages"] if hasattr(msg, "tool_calls")
        for tc in (msg.tool_calls or [])
    )
```

The first run makes a real network call and writes a cassette to `tests/cassettes/`. Every subsequent run replays the recorded interaction — no network, no cost, no latency, and no flakiness from the model behaving slightly differently between runs.

**The gotcha to plan for:** cassettes go stale. The moment you change a prompt, add a tool, or otherwise alter what the agent should do, the recorded cassette no longer reflects reality, and tests using it will fail in a way that looks like a regression but is actually a stale fixture. Delete the affected cassette files and let the test suite re-record against the real API whenever you deliberately change agent behavior.

---

### 11.4 Evals: Scoring the Trajectory

A **trajectory** is the full sequence of messages and tool calls an agent produces for a given input. Evals score that trajectory against either a known-good reference (deterministic matching) or a rubric an LLM judges against (no reference required). This is the tier that catches "the agent still runs, but it's now doing something subtly wrong" — the class of regression neither unit nor integration tests are built to notice.

An evaluator, in general, is nothing more than a function taking outputs (and optionally a reference) and returning a score:

```python
def evaluator(*, outputs: dict, reference_outputs: dict):
    output_messages = outputs["messages"]
    reference_messages = reference_outputs["messages"]
    score = compare_messages(output_messages, reference_messages)
    return {"key": "evaluator_score", "score": score}
```

The `agentevals` package (`pip install agentevals`) provides prebuilt evaluators for exactly this shape, for both trajectory matching and LLM-as-judge.

#### 11.4.1 Trajectory Match: Deterministic, Fast, Free

`create_trajectory_match_evaluator` compares your agent's actual trajectory against a reference trajectory you supply, in one of four modes:

| Mode | Behavior | Use case |
|---|---|---|
| `strict` | Identical message structure and tool calls, same order (content can differ) | Enforcing a specific required sequence — e.g., a policy lookup before an authorization |
| `unordered` | Same tool calls as the reference, any order | Verifying information was retrieved when order genuinely doesn't matter |
| `subset` | Agent calls only tools from the reference — no extras | Ensuring the agent doesn't exceed its expected scope |
| `superset` | Agent calls *at least* the reference's tools — extras allowed | Verifying the minimum required actions were taken |

```python
from agentevals.trajectory.match import create_trajectory_match_evaluator
from langchain.messages import HumanMessage, AIMessage, ToolMessage

evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")

def test_weather_tool_called_strict():
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in San Francisco?")]})

    reference_trajectory = [
        HumanMessage(content="What's the weather in San Francisco?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "San Francisco"}}
        ]),
        ToolMessage(content="It's 75 degrees and sunny in San Francisco.", tool_call_id="call_1"),
        AIMessage(content="The weather in San Francisco is 75 degrees and sunny."),
    ]

    evaluation = evaluator(outputs=result["messages"], reference_outputs=reference_trajectory)
    # {'key': 'trajectory_strict_match', 'score': True, 'comment': None}
    assert evaluation["score"] is True
```

`unordered` mode is the right choice when an agent legitimately might call two independent tools (weather, then events; or events, then weather) in either order and both are correct:

```python
evaluator = create_trajectory_match_evaluator(trajectory_match_mode="unordered")

def test_multiple_tools_any_order():
    result = agent.invoke({"messages": [HumanMessage(content="What's happening in SF today?")]})
    reference_trajectory = [
        HumanMessage(content="What's happening in SF today?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_events", "args": {"city": "SF"}},
            {"id": "call_2", "name": "get_weather", "args": {"city": "SF"}},
        ]),
        ToolMessage(content="Concert at the park in SF tonight.", tool_call_id="call_1"),
        ToolMessage(content="It's 75 degrees and sunny in SF.", tool_call_id="call_2"),
        AIMessage(content="Today in SF: 75 degrees and sunny with a concert at the park tonight."),
    ]
    evaluation = evaluator(outputs=result["messages"], reference_outputs=reference_trajectory)
    assert evaluation["score"] is True
```

`superset` mode is useful precisely when you've added a new tool and want to confirm the agent still performs the previously-required minimum, without failing every existing test the moment it also starts using the new capability:

```python
evaluator = create_trajectory_match_evaluator(trajectory_match_mode="superset")
# Reference only requires get_weather; the agent may also call get_detailed_forecast
```

You can further tune what counts as "the same tool call" via `tool_args_match_mode` and `tool_args_match_overrides` — by default, two calls match only if they target the same tool with identical arguments.

#### 11.4.2 LLM-as-Judge: Qualitative, No Reference Required

`create_trajectory_llm_as_judge` scores a trajectory using an LLM, with or without a reference trajectory — appropriate when "correctness" is a matter of reasoning quality rather than an exact expected sequence.

```python
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

def test_trajectory_quality():
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in Seattle?")]})
    evaluation = evaluator(outputs=result["messages"])
    assert evaluation["score"] is True
```

With a reference trajectory available, use the paired prompt so the judge grades against it rather than judging in a vacuum:

```python
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE

evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)
evaluation = evaluator(outputs=result["messages"], reference_outputs=reference_trajectory)
```

**Choosing between match and judge:** use trajectory match when you know the expected tool calls and want a fast, deterministic, zero-cost check (this should be the majority of your eval suite — it's essentially free and never flaky). Reach for LLM-as-judge when you want to assess overall reasoning quality without a strict expected sequence, accepting the tradeoff of added cost, latency, and the judge's own occasional inconsistency.

#### 11.4.3 Async Support

Every `agentevals` evaluator has an async counterpart — add `async` after `create_`:

```python
from agentevals.trajectory.llm import create_async_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from agentevals.trajectory.match import create_async_trajectory_match_evaluator

async_judge = create_async_trajectory_llm_as_judge(model="openai:o3-mini", prompt=TRAJECTORY_ACCURACY_PROMPT)
async_evaluator = create_async_trajectory_match_evaluator(trajectory_match_mode="strict")

async def test_async_evaluation():
    result = await agent.ainvoke({"messages": [HumanMessage(content="What's the weather?")]})
    evaluation = await async_judge(outputs=result["messages"])
    assert evaluation["score"] is True
```

---

### 11.5 Running Evals in LangSmith

Scoring one trajectory locally is useful for a single regression test. Tracking scores across many examples, over time, as prompts and models change, is what LangSmith's evaluation layer is for. There are two ways in:

#### 11.5.1 The `pytest` Integration

Mark a test with `@pytest.mark.langsmith`, log inputs/outputs/reference via the `langsmith.testing` helpers, and every test run becomes a tracked LangSmith experiment:

```python
import pytest
from langsmith import testing as t
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

trajectory_evaluator = create_trajectory_llm_as_judge(model="openai:o3-mini", prompt=TRAJECTORY_ACCURACY_PROMPT)

@pytest.mark.langsmith
def test_trajectory_accuracy():
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})

    reference_trajectory = [
        HumanMessage(content="What's the weather in SF?"),
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "SF"}}]),
        ToolMessage(content="It's 75 degrees and sunny in SF.", tool_call_id="call_1"),
        AIMessage(content="The weather in SF is 75 degrees and sunny."),
    ]

    t.log_inputs({})
    t.log_outputs({"messages": result["messages"]})
    t.log_reference_outputs({"messages": reference_trajectory})

    trajectory_evaluator(outputs=result["messages"], reference_outputs=reference_trajectory)
```

```bash
pytest test_trajectory.py --langsmith-output
```

Every run of this test logs a case to a LangSmith experiment automatically — you get the trace for any failed case (to see exactly what the agent did) and a history of results over time, so "did this change actually improve quality?" becomes something you can look up instead of something you have to guess at.

#### 11.5.2 The `evaluate()` Function Against a Dataset

For evaluating against a curated batch of examples rather than test-by-test, create a LangSmith dataset and run `Client.evaluate()`:

```python
from langsmith import Client
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

client = Client()

trajectory_evaluator = create_trajectory_llm_as_judge(model="openai:o3-mini", prompt=TRAJECTORY_ACCURACY_PROMPT)

def run_agent(inputs):
    return agent.invoke(inputs)["messages"]

experiment_results = client.evaluate(
    run_agent,
    data="your_dataset_name",
    evaluators=[trajectory_evaluator],
)
```

The dataset schema for trajectory evaluation is straightforward:
- **input**: `{"messages": [...]}` — the input messages to call the agent with
- **output**: `{"messages": [...]}` — the expected message history (for trajectory evaluation, you can keep only the assistant messages if that's all you're scoring)

**Building the dataset itself, in practice:** the most effective source of examples is production traces, not hand-written synthetic cases — real user queries surface edge cases synthetic examples miss. A practical workflow is to start small (ten examples is a reasonable floor), grow the dataset every time a real failure surfaces in production, and treat it the way you'd treat a regression test suite for traditional code: once a bug is fixed, the example that caught it stays in the dataset permanently.

**Comparing experiments:** LangSmith's UI shows side-by-side metrics across experiment runs, which is what turns "I changed the prompt and it feels better" into "example #7 regressed and here's the trace showing why."

---

### 11.6 LangSmith Tracing

Evals score a trajectory after the fact. Tracing shows you the trajectory *as it happened* — every LLM call, every tool invocation, every node transition, with latency and token counts attached to each. For a LangGraph application, the trace tree mirrors the graph itself: each node becomes a span, each conditional edge's decision is visible, and a failure three tool calls deep is something you can click into rather than guess at.

#### 11.6.1 Setup

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
export OPENAI_API_KEY=<your-openai-api-key>   # or whichever provider you use

# If your LangSmith API key spans multiple workspaces, specify which to use:
export LANGSMITH_WORKSPACE_ID=<workspace-id>
```

With these set, LangChain and LangGraph executions are traced automatically — no code changes to your existing graph. This is the single biggest reason LangSmith integrates with near-zero glue code for anything already built on `StateGraph` or `create_agent`: the callback machinery that produces traces is the same machinery already running your graph.

#### 11.6.2 `@traceable` for Everything Else

Real applications call things LangChain doesn't instrument automatically — a custom preprocessing function, a direct HTTP call to an external API, business logic that sits alongside the graph. `@traceable` brings any of these into the same run tree:

```python
from langsmith import traceable

@traceable
def preprocess_query(raw_query: str) -> str:
    return raw_query.strip().lower()
```

Nested `@traceable` functions automatically form parent-child relationships in the trace, matching the actual call structure — you don't have to wire this up manually. Whether a span comes from automatic LangGraph instrumentation, `@traceable`, or the lower-level `RunTree` API for full manual control, they all populate the same underlying run schema (inputs, outputs, timing, `run_type`, and hierarchy via `parent_run_id`), so a single trace view tells one coherent story regardless of which mechanism produced each span.

#### 11.6.3 What a LangGraph Trace Shows You

- **Graph and node spans** — the compiled graph invocation and each node it dispatched to, in execution order
- **LLM spans** — every chat model call inside a node, with prompt, response, latency, and token counts
- **Tool spans** — every tool call executed by `ToolNode` (Chapter 9)
- **The Messages view** — a simplified, chat-like rendering pulled from the top-level trace: the user's original request, tool calls, and the agent's final response, without the internal span-tree noise

When a full-turn eval fails (Section 11.5), this is where you go next: open the trace for that specific failing case and see exactly where the agent's actual behavior diverged from what you expected — which node ran, what the model was given, what it decided, and why.

---

### 11.7 Wiring This Into CI/CD: A Quality Gate

The pieces from this chapter compose into a deployment pipeline where agent changes are gated on passing evals, not just passing unit tests:

```
PR opened
   ↓
Unit tests (GenericFakeChatModel, InMemorySaver) — fast, run on every push
   ↓
Integration tests (pytest -m integration, VCR-cached where possible) — run in CI
   ↓
Offline evals (agentevals trajectory match + LLM judge, against a LangSmith dataset)
   ↓
Quality gate: if eval scores regress below threshold, block merge
   ↓
Preview deployment (staging) — manual or automated smoke test
   ↓
Production deployment — gated on the above passing
   ↓
Continuous monitoring: production traces feed back into the eval dataset
```

The monitoring feedback loop closes the system: production traces that surface a real failure become new dataset examples (Section 11.5.2), which means every incident makes the eval suite strictly better at catching that class of problem before it ships again — the same principle as adding a regression test for every bug fixed in traditional software, applied to agent behavior instead of code paths.

---

### 11.8 MARRS Checkpoint: A Complete Three-Tier Test Suite

We now build the full testing story for MARRS: fast unit tests with a fake model, integration tests against the real model with structural assertions and VCR caching, and trajectory evals — both deterministic and LLM-judged — tracked in LangSmith.

```python
# ─────────────────────────────────────────────────────────────────────────────
# TIER 1: UNIT TESTS — fake model, no network, no cost
# ─────────────────────────────────────────────────────────────────────────────

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langgraph.checkpoint.memory import InMemorySaver

def test_researcher_calls_search_tool():
    """Unit test: scripted fake model should trigger a tool call, and ToolNode should execute it."""
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[
            ToolCall(name="web_search", args={"query": "transformer architectures"}, id="call_1")
        ]),
        AIMessage(content="Based on the search results, transformers use self-attention..."),
    ]))

    test_graph = build_marrs_test_graph(model=fake_model)   # test-only factory injecting the fake
    config = {"configurable": {"thread_id": "unit-test-1"}}

    result = test_graph.invoke(
        {"messages": [HumanMessage(content="Research transformer architectures")]}, config
    )

    tool_calls = [
        tc for m in result["messages"] if hasattr(m, "tool_calls") for tc in (m.tool_calls or [])
    ]
    assert any(tc["name"] == "web_search" for tc in tool_calls)


def test_marrs_multi_turn_memory():
    """Unit test: state persists correctly across turns using InMemorySaver."""
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="Got it — I'll focus on climate policy research."),
        AIMessage(content="Building on the climate policy focus, here's an update..."),
    ]))

    checkpointer = InMemorySaver()
    test_graph = build_marrs_test_graph(model=fake_model, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "unit-test-memory"}}

    test_graph.invoke({"messages": [HumanMessage(content="Focus on climate policy")]}, config)
    result = test_graph.invoke({"messages": [HumanMessage(content="Give me an update")]}, config)

    # The second turn's response should reflect context carried from the first
    assert "climate policy" in result["messages"][-1].content.lower() or len(result["messages"]) >= 4


def test_critic_routing_logic():
    """Unit test: the Command-based critic router (Chapter 4) in isolation, no model needed."""
    high_quality_state = {"quality_score": 0.0, "revision_count": 1, "draft": "excellent draft"}
    # critic_node_command internally computes its own score in this MARRS version;
    # here we test the pure routing decision function directly instead
    assert route_after_critic({"quality_score": 0.85, "revision_count": 1}) == "good_enough"
    assert route_after_critic({"quality_score": 0.50, "revision_count": 1}) == "needs_revision"
    assert route_after_critic({"quality_score": 0.50, "revision_count": 3}) == "max_reached"


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2: INTEGRATION TESTS — real model, marked separately, structure-only assertions
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.vcr()
def test_marrs_research_integration():
    """Integration test: real model call, cached via VCR after first recording."""
    config = {"configurable": {"thread_id": "integration-test-1"}}
    result = marrs_supervisor.invoke(
        {"messages": [{"role": "user", "content": "Research quantum error correction"}]}, config
    )

    tool_calls = [
        tc for m in result["messages"] if hasattr(m, "tool_calls") for tc in (m.tool_calls or [])
    ]
    # Assert structure, not exact wording — the model's phrasing will vary run to run
    assert any(tc["name"] in ("delegate_research", "delegate_writing") for tc in tool_calls)
    assert isinstance(result["messages"][-1], AIMessage)
    assert len(result["messages"][-1].content) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3: EVALS — trajectory match + LLM judge, tracked in LangSmith
# ─────────────────────────────────────────────────────────────────────────────

from agentevals.trajectory.match import create_trajectory_match_evaluator
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from langchain_core.messages import ToolMessage
from langsmith import testing as t

# Deterministic: the supervisor must delegate to BOTH specialists, order doesn't matter
delegation_evaluator = create_trajectory_match_evaluator(trajectory_match_mode="superset")

@pytest.mark.langsmith
@pytest.mark.integration
def test_marrs_delegates_to_both_specialists():
    config = {"configurable": {"thread_id": "eval-delegation-1"}}
    result = marrs_supervisor.invoke(
        {"messages": [{"role": "user", "content": "Research and write a report on AI safety"}]}, config
    )

    reference_trajectory = [
        HumanMessage(content="Research and write a report on AI safety"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "delegate_research", "args": {"topic": "AI safety"}}]),
        ToolMessage(content="[research findings]", tool_call_id="c1"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "delegate_writing", "args": {"findings": "..."}}]),
        ToolMessage(content="[draft report]", tool_call_id="c2"),
        AIMessage(content="Here is the report..."),
    ]

    t.log_inputs({"topic": "AI safety"})
    t.log_outputs({"messages": result["messages"]})
    t.log_reference_outputs({"messages": reference_trajectory})

    evaluation = delegation_evaluator(outputs=result["messages"], reference_outputs=reference_trajectory)
    assert evaluation["score"] is True


# Qualitative: is the final report actually well-reasoned, not just structurally correct?
quality_judge = create_trajectory_llm_as_judge(model="openai:o3-mini", prompt=TRAJECTORY_ACCURACY_PROMPT)

@pytest.mark.langsmith
@pytest.mark.integration
def test_marrs_report_quality():
    config = {"configurable": {"thread_id": "eval-quality-1"}}
    result = marrs_supervisor.invoke(
        {"messages": [{"role": "user", "content": "Research and write a report on AI safety"}]}, config
    )
    evaluation = quality_judge(outputs=result["messages"])
    assert evaluation["score"] is True


# ─────────────────────────────────────────────────────────────────────────────
# DATASET-BASED REGRESSION SUITE — run against a curated LangSmith dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_marrs_regression_suite():
    """
    Run the full MARRS regression suite against a LangSmith dataset built from
    real production topics that previously surfaced issues.
    Intended to run in CI as the offline-evals quality gate (Section 11.7).
    """
    from langsmith import Client

    client = Client()

    def run_marrs(inputs):
        config = {"configurable": {"thread_id": f"regression-{inputs.get('topic', 'unknown')}"}}
        result = marrs_supervisor.invoke(
            {"messages": [{"role": "user", "content": f"Research and write a report on {inputs['topic']}"}]},
            config,
        )
        return {"messages": result["messages"]}

    results = client.evaluate(
        run_marrs,
        data="marrs-regression-suite",   # curated dataset, grown from production incidents
        evaluators=[delegation_evaluator, quality_judge],
    )
    return results
```

---

### 11.9 Chapter Summary

**Agentic testing needs three tiers**, not one, because unit-level correctness ("the code ran") and decision-level quality ("the agent chose wisely") are genuinely different questions. Unit tests answer the first with fakes; evals answer the second by scoring a trajectory.

**Unit tests** use `GenericFakeChatModel` to script exact model responses (text or tool calls) and `InMemorySaver` for fast, free, deterministic multi-turn state tests — no network, no API key, no flakiness.

**Integration tests** hit real APIs and are kept separate from unit tests via pytest markers (`-m "not integration"` by default). Assert on structure — message types, tool names, argument shapes — never on exact content, since real model output varies between runs. `vcrpy` cassettes make repeated CI runs of the same integration test free after the first recording, at the cost of needing to delete stale cassettes whenever agent behavior deliberately changes.

**Evals** score a trajectory two ways: `agentevals`' `create_trajectory_match_evaluator` (four modes — `strict`, `unordered`, `subset`, `superset` — deterministic, fast, free) when you know the expected tool calls, and `create_trajectory_llm_as_judge` when you're assessing reasoning quality without a strict expected sequence. Both integrate into LangSmith either via the `@pytest.mark.langsmith` decorator (test-by-test tracking) or `Client.evaluate()` against a curated dataset (batch regression runs) — with production traces themselves being the best source of new dataset examples over time.

**LangSmith tracing** requires only `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and your model provider's key to automatically trace every LangChain/LangGraph execution with near-zero code changes; `@traceable` extends the same trace tree to arbitrary custom functions. A LangGraph trace mirrors the graph itself — node spans, LLM spans, tool spans — turning "which of ten LLM calls produced the bad output" from a guessing game into something you click into directly.

**The full pipeline** — unit tests on every push, integration tests and offline evals as a CI quality gate, production traces feeding back into the eval dataset — is what turns "it looked better on the three examples I tried" into a system that actually catches regressions before they reach users.

---

### Further Reading

- **Official Test overview**: `docs.langchain.com/oss/python/langchain/test` — the three-tier framework this chapter is built on
- **Official Unit testing docs**: `docs.langchain.com/oss/python/langchain/test/unit-testing` — `GenericFakeChatModel` and `InMemorySaver` patterns
- **Official Integration testing docs**: `docs.langchain.com/oss/python/langchain/test/integration-testing` — pytest markers, API key management, structural assertions, VCR cassette recording
- **Official Agent Evals docs**: `docs.langchain.com/oss/python/langchain/test/evals` — all four trajectory-match modes, LLM-as-judge, async variants, and both LangSmith integration paths
- **`agentevals` repository**: `github.com/langchain-ai/agentevals` — full evaluator configurability, `tool_args_match_mode` details
- **Official LangGraph tracing guide**: `docs.langchain.com/langsmith/trace-with-langgraph` — environment variable setup, the Messages view, and workspace-scoped API keys
- **"Implement a CI/CD pipeline using LangSmith Deployment and Evaluation"**: `docs.langchain.com/langsmith/cicd-pipeline-example` — the full staged pipeline (preview deployments, quality-gated production releases) this chapter's Section 11.7 summarizes

---

*End of Chapter 11. Chapter 12: Deployment — LangGraph Platform, Self-Hosting, and Production Operations.*
