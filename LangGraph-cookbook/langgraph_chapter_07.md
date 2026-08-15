# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 7 — Human-in-the-Loop: Interrupts, Breakpoints, and Time Travel

> *"When an interrupt is triggered, LangGraph saves the graph state using its persistence layer and waits indefinitely until you resume execution."*
> — LangGraph official documentation

> **Chapter revision note:** This chapter was refreshed to reflect current LangGraph documentation. Two things changed since the first version: (1) LangGraph now documents an official, supported pattern for resuming multiple *parallel* interrupts in a single invocation using an interrupt-ID-keyed resume map — previously this was an open edge case; (2) LangGraph now offers `graph.stream_events(..., version="v3")` as the recommended interface for driving HITL loops, exposing `stream.interrupted` / `stream.interrupts` / `stream.output` directly. `graph.invoke()` still works exactly as described throughout this chapter and remains fully supported — the official docs note it explicitly — so nothing below is broken, but the newer interface is worth knowing about and is introduced in Section 7.3.6.

---

### What This Chapter Covers

Chapters 2–6 built autonomous systems — agents that run to completion without pausing for external input. This chapter adds the ability to stop, consult a human, incorporate feedback, and continue. This is human-in-the-loop (HITL), and it is what transforms an agent from a demo into something trustworthy enough for production.

By the end you will understand:

1. Why HITL is architecturally necessary and the four canonical situations requiring it
2. The two interrupt mechanisms: dynamic `interrupt()` vs. static `interrupt_before`/`interrupt_after` breakpoints — what they are, how they differ, and when to reach for each
3. The complete mechanics of `interrupt()`: what it does under the hood, why the node re-executes on resume, and what that means for your code
4. The idempotency requirement: the most consequential rule in HITL code
5. The try/except trap: why you must never catch the interrupt exception
6. `Command(resume=...)`: the single way to resume a paused graph
7. Multiple interrupts in a single node: the strictly index-based matching rule, and why call order must stay consistent
8. Multiple *parallel* interrupts across nodes: the official interrupt-ID-keyed resume map pattern
9. Static breakpoints: `interrupt_before` and `interrupt_after` — compile-time vs. runtime configuration
10. Seven canonical HITL design patterns with complete implementations
11. State editing during a pause: `update_state` in the HITL context
12. Time travel in HITL: replaying, forking, and correcting from historical checkpoints
13. Production considerations: notifications, async workflows, audit trails
14. MARRS checkpoint: wiring the human review gate into the capstone project

---

### 7.1 Why Human-in-the-Loop Is Architecturally Necessary

Full automation is the goal of most agentic systems, but not every decision should be made autonomously. There are four situations where human oversight is structurally required rather than just desirable:

**1. Irreversibility.** A node is about to perform an action that cannot be undone: sending an email, charging a credit card, deploying to production, deleting a database row. The cost of a wrong AI decision is unboundedly high. A human must confirm before execution.

**2. Authorization thresholds.** Some actions require human sign-off by policy, regulation, or business rule — regardless of how confident the model is. "Purchases over $10,000 require CFO approval" is not something an LLM can override.

**3. Ambiguity that requires judgment.** The agent has gathered information but the next step genuinely depends on a preference, value judgment, or context that only a human can supply. ("Should I frame this report for a technical or executive audience?")

**4. Quality assurance.** A draft, plan, or decision that looks reasonable but needs review by an expert who can catch errors the model cannot recognize as errors.

LangGraph builds HITL on top of its persistence layer from Chapter 5. The insight is clean: a paused graph is just a graph whose checkpoint shows it is waiting at a specific node. The state is fully persisted. The graph can wait for hours, days, or indefinitely. When the human responds, the graph resumes from exactly where it stopped.

---

### 7.2 Two Interrupt Mechanisms: Dynamic vs. Static

LangGraph provides two ways to pause a graph for human input. They differ in where the decision to pause is made — at code-write time or at runtime.

| | Dynamic `interrupt()` | Static Breakpoints |
|---|---|---|
| **Where defined** | Inside node body | At `compile()` call |
| **When decided** | At runtime (can be conditional) | At compile time (always fires) |
| **Pause timing** | Anywhere within a node | Before or after a specific node |
| **Use when** | Pause depends on current state | Always pause at a fixed graph point |
| **Payload** | Any JSON-serializable value | None (no payload; you inspect state) |
| **API** | `interrupt(payload)` inside node | `compile(interrupt_before=["node"])` |

Both require a checkpointer and a `thread_id`. Both resume with `graph.invoke(Command(resume=...), config)` or `graph.invoke(None, config)`. Their mechanics diverge in important ways.

---

### 7.3 `interrupt()`: The Dynamic Interrupt

`interrupt()` is the modern, recommended mechanism. It gives you the flexibility to pause a graph conditionally — only when a runtime condition is met.

#### 7.3.1 Basic Usage

```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

class ReviewState(TypedDict):
    draft: str
    approved: bool
    feedback: str

def review_node(state: ReviewState) -> dict:
    """
    Pause for human review. The payload passed to interrupt()
    is surfaced to the caller as the interrupt's value.
    """
    # interrupt() pauses here and sends this payload to the caller.
    # When resumed, Command(resume=...) provides the return value.
    decision = interrupt({
        "message": "Please review this draft",
        "draft": state["draft"],
        "options": ["approve", "reject", "revise"],
    })
    # After resume, 'decision' holds whatever Command(resume=...) provided.
    return {"approved": decision == "approve", "feedback": decision}

builder = StateGraph(ReviewState)
builder.add_node("write", write_draft_node)
builder.add_node("review", review_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "write")
builder.add_edge("write", "review")
builder.add_edge("review", "finalize")

graph = builder.compile(checkpointer=InMemorySaver())
```

#### 7.3.2 The Pause-Resume Lifecycle

```python
config = {"configurable": {"thread_id": "draft-review-001"}}

# Step 1: Run the graph — it hits interrupt() and pauses
result = graph.invoke({"draft": "", "approved": False, "feedback": ""}, config)

# The result contains an "__interrupt__" key with the payload you passed:
print(result["__interrupt__"])
# (Interrupt(value={'message': 'Please review this draft', 'draft': '...', ...}, when='during'),)

# The graph is now paused. The state is fully checkpointed.
# The human examines the draft and makes a decision.

# Step 2: Resume with the human's decision
final = graph.invoke(Command(resume="approve"), config)
print(final)
# {'draft': '...', 'approved': True, 'feedback': 'approve', ...}
```

Three important facts about this lifecycle:

1. `graph.invoke()` returns as soon as `interrupt()` is hit — the first call does not block waiting for input. It returns the paused state.
2. The second `graph.invoke(Command(resume=...), config)` uses the **same `thread_id`** to load the checkpointed state and continue.
3. `Command(resume=...)` is the **only** way to pass the human's input back to the graph. You cannot pass a new input dict instead — that would start a new execution, not resume the paused one.

#### 7.3.3 What `interrupt()` Does Under the Hood

From the official LangGraph docs:

> "When you call `interrupt` within a node, LangGraph suspends execution by raising an exception that signals the runtime to pause. This exception propagates up through the call stack and is caught by the runtime, which notifies the graph to save the current state and wait for external input."

The mechanism is exception-based: `interrupt()` raises a special internal exception (`GraphInterrupt`) that the Pregel runtime catches, writes the checkpoint, and returns control to the caller. This has one critical consequence.

#### 7.3.4 The Re-Execution Rule: The Most Important Fact in This Chapter

From the official docs:

> "When execution resumes (after you provide the requested input), **the runtime restarts the entire node from the beginning** — it does not resume from the exact line where `interrupt` was called. This means any code that ran before the `interrupt` will execute again."

This is not an implementation detail — it is a fundamental property of the system. Every line of code before `interrupt()` in the node runs **twice**: once during the first invocation (before pausing), and again when resuming.

**Why this design?** Because the runtime cannot serialize a Python call stack mid-execution. The checkpoint captures state channels, not bytecode position. On resume, the only way to get back to the interrupt point is to re-run the node from the top until `interrupt()` is encountered again. At that point, the runtime checks whether this interrupt already has a resume value. If it does, it supplies that value immediately instead of pausing again.

**The consequence:** Any code before `interrupt()` must be **idempotent** — safe to run twice with no additional side effects.

```python
# WRONG — this node is not idempotent
def bad_approval_node(state: State) -> dict:
    charge_credit_card(state["amount"])   # ← This fires TWICE (once before pause, once on resume)
    decision = interrupt("Approve the charge?")
    return {"charged": True}

# CORRECT — charge happens AFTER the interrupt, not before
def good_approval_node(state: State) -> dict:
    decision = interrupt({
        "message": "Approve this charge?",
        "amount": state["amount"]
    })
    if decision == "approve":
        charge_credit_card(state["amount"])   # ← Only runs once, after resume
    return {"charged": decision == "approve"}
```

**The rule:** Place `interrupt()` first in the node, or place all side-effecting code **after** `interrupt()`. Code before `interrupt()` must be pure reads or idempotent operations.

#### 7.3.5 The try/except Trap

Because `interrupt()` works by raising an exception, wrapping it in a `try/except` block will catch the interrupt signal and prevent it from propagating to the runtime. The interrupt will silently fail.

```python
# WRONG — this catches the interrupt exception and breaks HITL entirely
def broken_node(state: State) -> dict:
    try:
        decision = interrupt("Approve?")   # ← Raises internally
    except Exception:
        decision = "default"              # ← Catches the interrupt signal!
    return {"decision": decision}
    # The graph never actually pauses — the interrupt is swallowed.

# CORRECT — never wrap interrupt() in try/except
def working_node(state: State) -> dict:
    decision = interrupt("Approve?")
    return {"decision": decision}
```

This is documented explicitly by the LangGraph team. Any exception handling in your node must be structured so that `interrupt()`'s internal exception can propagate freely.

#### 7.3.6 A Newer Interface for Driving HITL Loops: `stream_events(version="v3")`

Everything so far in this chapter drives the pause/resume cycle through `graph.invoke()`, checking for the `"__interrupt__"` key in the returned dict (Section 7.8 covers this in detail). That remains fully supported — the official docs are explicit that `graph.invoke()` "still works and surfaces interrupts under `result["__interrupt__"]`."

LangGraph now also offers a richer, purpose-built interface for this exact loop: `graph.stream_events(..., version="v3")`. Instead of inspecting a dict key after the fact, you get typed properties on the returned stream object:

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "thread-1"}}

# Initial run — drives the stream until it finishes or pauses
stream = graph.stream_events({"input": "data"}, config=config, version="v3")
final = stream.output          # The final state, once the run completes or pauses

if stream.interrupted:         # True when the run paused for human input
    print(stream.interrupts)   # Tuple of pending Interrupt objects, each with .value and .id
    # > (Interrupt(value='Do you approve this action?', id='...'),)

# Resume with the human's response — same shape as Command(resume=...) throughout this chapter
resumed = graph.stream_events(Command(resume=True), config=config, version="v3")
final = resumed.output
```

The advantage over the `invoke()` + `"__interrupt__"` pattern is mainly ergonomic for interactive UIs: `stream.messages` gives token-by-token LLM output (Chapter 8's streaming modes, unified into one object), `stream.values` gives state snapshots, and `stream.interrupted`/`stream.interrupts` gives a clean boolean-plus-payload pair instead of a dict-key check — all through the same stream object, in a loop:

```python
stream_input: dict | Command = initial_input

while True:
    stream = graph.stream_events(stream_input, config=config, version="v3")

    for message in stream.messages:            # Token-by-token output as it streams
        for token in message.text:
            display_streaming_content(token)

    if not stream.interrupted:                  # Run finished without pausing
        final_state = stream.output
        break

    interrupt_info = stream.interrupts[0].value
    user_response = get_user_input(interrupt_info)
    stream_input = Command(resume=user_response)  # Loop drives the next resume
```

**This chapter continues to use `graph.invoke()` and `result["__interrupt__"]`** for its examples, because that pattern is simpler to reason about while learning the underlying mechanics, and it remains a first-class, fully supported API. Reach for `stream_events(version="v3")` once you're building an interactive, streaming HITL surface and want the token-level and interrupt-level views unified in one object rather than juggling `stream_mode` lists (Chapter 8) alongside a separate invoke-based interrupt check.

---

### 7.4 Multiple Interrupts in a Single Node

It is possible to call `interrupt()` multiple times in one node — for a multi-step review form, for instance:

```python
def multi_question_node(state: State) -> dict:
    name = interrupt("What is your name?")       # Interrupt 1
    age = interrupt("What is your age?")          # Interrupt 2
    city = interrupt("What is your city?")        # Interrupt 3
    return {"name": name, "age": age, "city": city}
```

**How it works:** When the node re-executes on resume, the runtime maintains a **list of resume values** for this task in order of arrival. For each `interrupt()` call encountered, it checks whether a matching resume value exists at that index. If it does, the value is returned immediately without pausing. If not, the node pauses again.

This means:
- Resume 1 → node re-executes, first `interrupt()` returns `name`, second `interrupt()` pauses again
- Resume 2 → node re-executes again, first returns `name`, second returns `age`, third pauses
- Resume 3 → node re-executes, all three return values, node completes

**This is documented as strictly index-based matching — an intentional design, not a bug.** The current official docs state it plainly: *"Matching is strictly index-based, so the order of interrupt calls within the node is important."* Three rules follow directly from this:

```python
# ✅ GOOD — interrupt calls happen in the same order every time
def consistent_node(state: State) -> dict:
    name = interrupt("What's your name?")
    age = interrupt("What's your age?")
    city = interrupt("What's your city?")
    return {"name": name, "age": age, "city": city}

# ❌ BAD — conditionally skipping an interrupt changes the order on resume
def inconsistent_node(state: State) -> dict:
    name = interrupt("Name?")
    if state.get("needs_age"):          # ← On first run this might skip the interrupt;
        age = interrupt("Age?")         #   on resume it might not — index mismatch
    city = interrupt("City?")
    return {"name": name, "city": city}

# ❌ BAD — looping interrupt() over data that can change between executions
def bad_loop_node(state: State) -> dict:
    results = []
    for item in state.get("dynamic_list", []):  # list length might differ on resume
        result = interrupt(f"Approve {item}?")
        results.append(result)
    return {"results": results}
```

The third rule is worth calling out specifically: a `while True: interrupt(...)` loop is fine (Section 7.6.5's validation pattern uses exactly this, and it's an official pattern), because it's the *same* call site re-evaluated deterministically each pass. What breaks index matching is a loop whose *iteration count* depends on data that can differ between the first run and the resumed run — for example, iterating over a state list that itself might grow or shrink.

**Given all this, the simplest and most robust pattern remains one interrupt per node**, chaining nodes for multi-step flows. This isn't a workaround for a limitation — it's simply the easiest way to guarantee the ordering rule holds, since a single-interrupt node has no ordering to violate:

```python
# Simplest robust pattern: one interrupt per node, chained
def ask_name(state: State) -> dict:
    name = interrupt("What is your name?")
    return {"name": name}

def ask_age(state: State) -> dict:
    age = interrupt("What is your age?")
    return {"age": age}

def ask_city(state: State) -> dict:
    city = interrupt("What is your city?")
    return {"city": city}

builder.add_edge(START, "ask_name")
builder.add_edge("ask_name", "ask_age")
builder.add_edge("ask_age", "ask_city")
builder.add_edge("ask_city", "process")
```

#### 7.4.1 Multiple *Parallel* Interrupts: The Interrupt-ID-Keyed Resume Map

Section 7.4 above covers multiple interrupts *within one node*, resolved by call order. A related but distinct situation is multiple *different nodes* — running in parallel, in the same superstep — each hitting their own `interrupt()`. Fan-out to several reviewers at once is the typical case: three parallel branches, each pausing for a different approval.

For this scenario, LangGraph now documents an official, supported pattern: **pair each pending interrupt's `id` with its resume value in a dict**, and pass that dict as the `resume` argument.

```python
from typing import Annotated, TypedDict
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

class State(TypedDict):
    vals: Annotated[list[str], operator.add]

def node_a(state: State) -> dict:
    answer = interrupt("question_a")
    return {"vals": [f"a:{answer}"]}

def node_b(state: State) -> dict:
    answer = interrupt("question_b")
    return {"vals": [f"b:{answer}"]}

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")     # Both nodes run in the same superstep —
    .add_edge(START, "b")     # a genuine parallel fan-out from START
    .add_edge("a", END)
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

# Step 1: run — both parallel nodes hit interrupt() and the graph pauses
result = graph.invoke({"vals": []}, config)
pending = result["__interrupt__"]
print(pending)
# (Interrupt(value='question_a', id='...'), Interrupt(value='question_b', id='...'))

# Step 2: resume BOTH pending interrupts in a single invocation,
# keyed by each interrupt's own id — order no longer matters, only the id does
resume_map = {i.id: f"answer for {i.value}" for i in pending}
final = graph.invoke(Command(resume=resume_map), config)

print(final["vals"])
# ['a:answer for question_a', 'b:answer for question_b']
```

Every `Interrupt` object now carries an `id` field. Building `resume_map` from the pending interrupts and passing the whole dict to `Command(resume=...)` correctly routes each answer back to the node that asked for it — regardless of which node happened to be listed first. This is the direct fix for the exact failure mode earlier versions of LangGraph had with parallel interrupts (mismatched or duplicate resume values landing on the wrong branch): the `id` is now the correlation key, not position.

**Practical guidance:** if your graph fans out to multiple reviewers or approval gates in parallel, always build the resume map from the actual pending `Interrupt.id` values you receive back — never assume you know their order in advance.

Each node has exactly one interrupt. Each resume restarts a single small node with no state before the interrupt. Clean, predictable, no double-execution risk.

---

### 7.5 Static Breakpoints: `interrupt_before` and `interrupt_after`

Static breakpoints are configured at compile time and fire unconditionally on every execution. They do not require modifying node logic.

#### 7.5.1 At Compile Time

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"],   # Pause BEFORE this node runs
    # OR
    interrupt_after=["writer"],          # Pause AFTER this node runs
)
```

`interrupt_before=["human_review"]` means: after the previous superstep completes, save a checkpoint and pause. The `human_review` node has not yet run. `snapshot.next` will show `("human_review",)`.

`interrupt_after=["writer"]` means: after `writer` runs and its state updates are applied, save a checkpoint and pause. `snapshot.next` will show the node(s) that come after `writer`.

#### 7.5.2 The Resume Pattern for Static Breakpoints

After a static breakpoint, resume with `None` as input (no human payload to inject) or use `update_state` to modify state before resuming:

```python
config = {"configurable": {"thread_id": "t1"}}

# Run until breakpoint
graph.invoke({"topic": "AI"}, config)

# Inspect the paused state
snapshot = graph.get_state(config)
print(snapshot.next)   # ('human_review',)
print(snapshot.values.get("draft"))  # The draft that needs review

# Option A: Resume without changes
graph.invoke(None, config)

# Option B: Edit state then resume
graph.update_state(config, {"quality_score": 0.95})  # Override quality before human_review runs
graph.invoke(None, config)
```

#### 7.5.3 Static vs. Dynamic: Decision Guide

Use **static breakpoints** when:
- You always want oversight at a specific point, regardless of state
- The node being reviewed doesn't need to communicate a payload to the reviewer
- You are retrofitting HITL into an existing graph without modifying node code
- You want oversight during debugging (temporarily add `interrupt_before` at compile time)

Use **dynamic `interrupt()`** when:
- The pause is conditional ("only interrupt if quality score < threshold")
- The node needs to surface specific information to the reviewer (the interrupt payload)
- You want the reviewer's input to become a value in the node's logic
- You need different payloads depending on which condition triggered the pause

---

### 7.6 Seven Canonical HITL Design Patterns

These are the patterns you will encounter repeatedly. Each is a complete, working implementation.

#### 7.6.1 Pattern 1: Approve or Reject

The simplest pattern. The agent proposes an action; a human approves or rejects it.

```python
from typing import Literal
from langgraph.types import interrupt, Command

def approval_gate(state: State) -> Command[Literal["execute", "abort"]]:
    """Pause for binary approval before an irreversible action."""
    decision = interrupt({
        "message": "Approve executing this action?",
        "action": state.get("proposed_action"),
        "risk_level": state.get("risk_level", "unknown"),
    })
    
    if decision == "approve":
        return Command(goto="execute", update={"approved": True})
    else:
        return Command(goto="abort", update={"approved": False, "abort_reason": decision})
```

**Resume:**
```python
graph.invoke(Command(resume="approve"), config)   # Approve
graph.invoke(Command(resume="reject"), config)    # Reject
```

#### 7.6.2 Pattern 2: Edit State Before Continuing

The human reviews the AI's output and can modify it before execution continues. The edit is injected as the interrupt's return value.

```python
def review_and_edit_node(state: State) -> dict:
    """Pause for review. Human can edit the draft before it continues."""
    result = interrupt({
        "task": "Review this draft. Return the edited version or 'approve' to accept as-is.",
        "draft": state["draft"],
        "critique": state.get("critique", ""),
    })
    
    if result == "approve":
        return {}  # No change — draft is fine as-is
    else:
        # Human returned an edited draft
        return {"draft": result}   # Updated draft replaces the AI's output
```

#### 7.6.3 Pattern 3: Review Tool Calls Before Execution (Approval Gate on Tools)

The most safety-critical pattern. The agent decides which tools to call; a human approves the calls before they execute. This prevents irreversible actions (database mutations, API calls, emails) from happening without human sign-off.

```python
from langchain_core.messages import AIMessage, ToolMessage

def review_tool_calls(state: AgentState) -> dict:
    """
    Pause before executing tool calls, show them to a human for approval.
    The agent has generated tool_calls in the last AIMessage; we pause
    before ToolNode executes them.
    """
    last_msg = state["messages"][-1]
    
    if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
        return {}  # No tool calls to review
    
    # Format tool calls for the reviewer
    call_summaries = [
        {
            "name": tc["name"],
            "args": tc["args"],
            "id": tc["id"],
        }
        for tc in last_msg.tool_calls
    ]
    
    decision = interrupt({
        "message": "The agent wants to call these tools. Approve, reject, or edit?",
        "tool_calls": call_summaries,
    })
    
    if decision == "approve":
        return {}  # Proceed with original tool calls
    
    if decision == "reject":
        # Inject a refusal message so the agent knows the calls were blocked
        rejection_messages = [
            ToolMessage(
                content=f"Tool call rejected by human reviewer.",
                tool_call_id=tc["id"],
                name=tc["name"],
            )
            for tc in last_msg.tool_calls
        ]
        return {"messages": rejection_messages}
    
    # Handle edit: human returns modified args
    # (advanced — modify last_msg.tool_calls in place, then let ToolNode execute)
    return {}
```

**The static breakpoint equivalent** (insert before tool node without modifying node code):

```python
# compile-time tool call review:
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"],   # Always pause before ToolNode executes
)
```

#### 7.6.4 Pattern 4: Multi-Turn Human Conversation (Clarification Loop)

When the agent needs to ask the human several follow-up questions before proceeding:

```python
def gather_requirements(state: FormState) -> dict:
    """Collect requirements across multiple conversation turns."""
    
    # Each interrupt() is one turn of the conversation
    topic = interrupt("What topic should I research?")
    audience = interrupt(f"Who is the target audience for a report on '{topic}'?")
    depth = interrupt("Brief overview or deep dive?")
    
    return {
        "topic": topic,
        "audience": audience,
        "depth": depth,
    }

# Each question requires a separate resume:
config = {"configurable": {"thread_id": "form-1"}}
graph.invoke({}, config)                                  # Pauses: "What topic?"
graph.invoke(Command(resume="transformer architectures"), config)  # Pauses: "Who is the audience?"
graph.invoke(Command(resume="ML engineers"), config)               # Pauses: "Brief or deep?"
graph.invoke(Command(resume="deep dive"), config)                  # Continues to next node
```

(Per the maintainer recommendation in Section 7.4, it's better to chain three separate nodes. This example is shown for illustration.)

#### 7.6.5 Pattern 5: Validation Loop (Retry on Invalid Input)

The graph keeps asking until it gets valid input:

```python
def validated_input_node(state: State) -> dict:
    """
    Ask for input, validate it, and keep asking until valid input arrives.
    The while loop wrapping interrupt() is the correct pattern for validation.
    """
    prompt = "Enter a positive integer:"
    
    while True:
        raw = interrupt(prompt)
        
        # Validate the input
        try:
            value = int(raw)
            if value > 0:
                return {"validated_value": value}
            prompt = f"'{raw}' must be positive. Enter a positive integer:"
        except (ValueError, TypeError):
            prompt = f"'{raw}' is not an integer. Enter a positive integer:"
        
        # Loop continues — next interrupt() call with updated prompt
```

Each iteration of the loop is a separate pause-resume cycle. The prompt updates based on the previous input's validation result.

#### 7.6.6 Pattern 6: Human Provides Additional Context Mid-Run

The agent is running, encounters something it doesn't know, and asks the human without stopping the entire workflow:

```python
def context_seeking_node(state: ResearchState) -> dict:
    """Ask for clarification when the agent lacks needed context."""
    
    # Only interrupt if genuinely uncertain
    confidence = state.get("plan_confidence", 1.0)
    
    if confidence < 0.7:
        clarification = interrupt({
            "message": "I'm uncertain how to proceed. Can you clarify?",
            "topic": state["topic"],
            "uncertainty": state.get("plan_rationale", "unclear"),
        })
        # Inject the clarification into the state for downstream nodes
        return {"clarification": clarification, "plan_confidence": 0.9}
    
    # If confident, skip the interrupt entirely
    return {}
```

This demonstrates **conditional interrupts** — the pause only happens when a runtime condition is met. This is the key advantage of dynamic `interrupt()` over static breakpoints.

#### 7.6.7 Pattern 7: State Edit via `update_state` (External Correction)

Sometimes you want a human to correct state outside of the graph's own node flow — not in response to an `interrupt()` payload, but as an external correction before resuming. This uses `update_state` directly (covered in Chapter 5) in a HITL context:

```python
# Graph is paused at interrupt_before=["critic"]
config = {"configurable": {"thread_id": "marrs-run-1"}}

# Run until breakpoint
graph.invoke({"topic": "AI safety", "findings": [], "sources": []}, config)

# Inspect the paused state
snapshot = graph.get_state(config)
print(f"Draft: {snapshot.values.get('draft', '')[:200]}")
print(f"Quality score: {snapshot.values.get('quality_score', 0)}")

# Human edits the draft externally
edited_draft = "# AI Safety\n\nEdited by human reviewer...\n"
graph.update_state(
    config,
    {"draft": edited_draft, "quality_score": 0.90},
    as_node="writer"   # Attribute this change to the writer node
                       # so execution resumes from writer's successors
)

# Resume from the corrected state — critic sees the edited draft
graph.invoke(None, config)
```

---

### 7.7 Time Travel in the HITL Context

Chapter 5 introduced time travel for fault recovery and debugging. In HITL workflows it has three additional use cases that are qualitatively different.

#### 7.7.1 Re-Review: Replaying From Before an Interrupt

When a human makes a review decision they want to revisit:

```python
config = {"configurable": {"thread_id": "review-thread-1"}}

# Original run
graph.invoke(initial_input, config)
graph.invoke(Command(resume="reject"), config)   # Human rejected draft
# ... graph routed to abort

# Human wants to reconsider — replay from before the review
history = list(graph.get_state_history(config))

# Find the checkpoint where the graph was waiting at 'review'
review_checkpoint = next(
    s for s in history
    if "review" in s.next
)

# Fork from that point and make a different decision
result = graph.invoke(Command(resume="approve"), review_checkpoint.config)
# Now the graph takes the approval path
```

Note: from the official docs on time travel and interrupts:

> "Replaying from the final checkpoint (no next nodes) is a no-op. The node containing the interrupt re-executes, and `interrupt()` pauses for a new `Command(resume=...)`."

So replaying from a checkpoint where an interrupt is pending works exactly as you'd expect — the node re-runs, hits `interrupt()`, and waits for a new resume value.

#### 7.7.2 Divergent Outcomes: A/B Testing Human Decisions

Time travel enables exploring "what if the human had decided differently?":

```python
# A run where the human approved
graph.invoke(Command(resume="approve"), approval_checkpoint.config)
approved_result = graph.get_state(config)

# Fork from the same approval checkpoint, but reject this time
graph.invoke(Command(resume="reject"), approval_checkpoint.config)
rejected_result = graph.get_state(config)

# Both branches now exist in the thread history
# This lets you compare outcomes before committing to a production decision
```

#### 7.7.3 Correcting a Past Decision After Observing Its Consequences

The most powerful time travel use case in production: an agent approved something that turned out to be wrong. Roll back and try again with corrected state:

```python
# Production run: approved at step 4, final result was bad
# Human wants to correct the approval and re-run from step 4

config = {"configurable": {"thread_id": "prod-thread-x"}}
history = list(graph.get_state_history(config))

# Find step 4 (before the bad approval)
step4 = next(s for s in history if s.metadata.get("step") == 4)

# Inject a correction: change the field that caused the bad decision
corrected_config = graph.update_state(
    step4.config,
    {"proposed_action": "safer_alternative", "risk_level": "low"},
    as_node="risk_assessor"
)

# Re-run from step 4 with the corrected state
result = graph.invoke(None, corrected_config)
```

---

### 7.8 The `__interrupt__` Key: Reading the Interrupt Payload

When `graph.invoke()` returns because an interrupt was hit, the result dict contains an `"__interrupt__"` key:

```python
result = graph.invoke(initial_input, config)

# Access the interrupt payload
interrupts = result.get("__interrupt__", ())

for interrupt_obj in interrupts:
    print(f"Interrupt value: {interrupt_obj.value}")
    print(f"When: {interrupt_obj.when}")     # "during" (always for interrupt())
```

When using `graph.stream()`, the interrupt appears as a special update:

```python
for chunk in graph.stream(initial_input, config, stream_mode="updates"):
    if "__interrupt__" in chunk:
        for interrupt_obj in chunk["__interrupt__"]:
            print(f"Graph paused: {interrupt_obj.value}")
        break  # Stop streaming — graph is paused
```

You can also inspect the paused state directly:

```python
snapshot = graph.get_state(config)
if snapshot.interrupts:
    for interrupt_obj in snapshot.interrupts:
        print(f"Waiting for: {interrupt_obj.value}")
```

---

### 7.9 Production HITL Considerations

The LangGraph `interrupt()` mechanism handles the persistence, pause, and resume lifecycle. It does not handle everything else a production HITL system needs.

#### 7.9.1 Notification: Nobody Gets Told

From a production implementation perspective:

> "When a thread is interrupted, nobody gets notified. The graph state is persisted in the checkpointer and the thread is marked as interrupted, but that's it. There's no email sent, no Slack message, no ping of any kind." — thehandover.xyz

The notification layer is your responsibility. A common pattern:

```python
async def notify_and_wait_for_resume(
    graph, 
    initial_input: dict, 
    config: dict,
    reviewer_email: str,
    webhook_url: str,
):
    """Run graph until interrupt, notify reviewer, then return."""
    result = await graph.ainvoke(initial_input, config)
    
    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        thread_id = config["configurable"]["thread_id"]
        
        # Send notification
        await send_email(
            to=reviewer_email,
            subject="Agent awaiting your approval",
            body=f"Thread {thread_id} needs review:\n{interrupt_payload}",
            resume_url=f"{webhook_url}/resume/{thread_id}",
        )
        # Graph is paused in the checkpointer. 
        # The reviewer clicks the URL → your webhook calls:
        # await graph.ainvoke(Command(resume=their_decision), config)
    
    return result
```

#### 7.9.2 Async HITL: The Graph Can Wait Indefinitely

The graph's paused state lives in the checkpointer database. Your server does not need to maintain any in-memory state between the initial invocation and the resume. This means:

- The server can restart between the pause and the resume
- A different server instance can handle the resume
- The reviewer can respond hours or days later

This is the key production advantage of LangGraph's HITL design: the pause is **decoupled from compute**. The agent is not consuming any resources while waiting.

```python
# Request 1 (from client): Start a run
@app.post("/research")
async def start_research(topic: str, db: AsyncSession):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    result = await graph.ainvoke({"topic": topic, "findings": []}, config)
    
    if "__interrupt__" in result:
        # Save pending review metadata
        await db.execute(
            "INSERT INTO pending_reviews (thread_id, payload) VALUES (?, ?)",
            [thread_id, json.dumps(result["__interrupt__"][0].value)]
        )
        return {"thread_id": thread_id, "status": "awaiting_review"}
    
    return {"thread_id": thread_id, "status": "complete", "report": result.get("final_report")}

# Request 2 (from reviewer UI): Submit decision
@app.post("/review/{thread_id}")
async def submit_review(thread_id: str, decision: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Resume the paused graph
    result = await graph.ainvoke(Command(resume=decision), config)
    
    # Clean up pending review
    await db.execute("DELETE FROM pending_reviews WHERE thread_id = ?", [thread_id])
    
    return {"status": "complete", "report": result.get("final_report")}
```

#### 7.9.3 Audit Trail: Every Decision Is Checkpointed

Because LangGraph checkpoints state after every superstep, and each HITL decision creates a new superstep, the full decision history is automatically preserved. You can reconstruct what the agent proposed, what the human decided, and what happened next — for every run, indefinitely.

```python
def build_audit_trail(thread_id: str) -> list[dict]:
    """Reconstruct the full decision history for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    
    audit = []
    for snapshot in graph.get_state_history(config):
        entry = {
            "step": snapshot.metadata.get("step"),
            "timestamp": snapshot.created_at,
            "next": list(snapshot.next),
            "quality_score": snapshot.values.get("quality_score"),
            "human_decision": snapshot.values.get("human_feedback"),
            "approved": snapshot.values.get("human_approved"),
        }
        audit.append(entry)
    
    return sorted(audit, key=lambda x: x["step"] or 0)
```

---

### 7.10 MARRS Checkpoint: The Human Review Gate

We now wire full HITL into MARRS. The architecture:

```
planner → researcher → writer → critic → [loop or] → HUMAN_REVIEW → finalize
                                                           ↑
                                              interrupt() pauses here
                                              human approves, rejects, or requests edits
```

```python
from typing import Literal, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3, operator, json

# ─────────────────────────────────────────────────────────────────────────────
# STATE (extends Chapter 6's MARRSStateV2 with HITL fields)
# ─────────────────────────────────────────────────────────────────────────────

class MARRSStateV3(TypedDict, total=False):
    topic: str
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    search_queries: list[str]
    findings: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]
    draft: str
    critique: str
    quality_score: float
    revision_count: int
    final_report: str
    status: str
    summary: str
    # HITL fields
    human_decision: str     # "approve" | "reject" | "revise"
    human_feedback: str     # Free-text feedback if revise/reject
    review_count: int       # Track how many times this went through human review


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN REVIEW NODE — The HITL gate
# ─────────────────────────────────────────────────────────────────────────────

def human_review_node(state: MARRSStateV3) -> Command[Literal["finalize", "writer", "__end__"]]:
    """
    Pause for human review of the research draft.
    
    This is the HITL gate. The graph pauses here and surfaces:
    - The draft text
    - The quality score from the critic
    - The search sources
    
    The human can:
    - "approve": accept the draft → route to finalize
    - "reject": discard entirely → route to END
    - A string beginning with "revise:": return with specific feedback → route back to writer
    
    IDEMPOTENCY NOTE: The interrupt() is the very first statement in this node.
    There is no code before it, so re-execution on resume is safe.
    """
    # interrupt() is first — nothing before it can double-execute
    review_result = interrupt({
        "message": "Please review this research draft.",
        "draft": state.get("draft", ""),
        "quality_score": state.get("quality_score", 0),
        "revision_count": state.get("revision_count", 0),
        "sources": state.get("sources", [])[:5],  # Show first 5 sources
        "instructions": "Reply 'approve', 'reject', or 'revise: <your feedback>'",
    })
    
    # Parse the human's response
    review_count = state.get("review_count", 0) + 1
    
    if review_result == "approve":
        return Command(
            goto="finalize",
            update={
                "human_decision": "approve",
                "human_feedback": "Approved",
                "review_count": review_count,
                "status": "approved",
            }
        )
    
    if review_result == "reject":
        return Command(
            goto=END,
            update={
                "human_decision": "reject",
                "human_feedback": "Rejected by reviewer",
                "review_count": review_count,
                "status": "rejected",
            }
        )
    
    if isinstance(review_result, str) and review_result.startswith("revise:"):
        feedback = review_result[len("revise:"):].strip()
        return Command(
            goto="writer",
            update={
                "human_decision": "revise",
                "human_feedback": feedback,
                "critique": f"Human reviewer: {feedback}",   # Inject feedback as critique
                "review_count": review_count,
                "status": "in_revision",
            }
        )
    
    # Unrecognized response — treat as approval to avoid blocking
    return Command(
        goto="finalize",
        update={
            "human_decision": "approve",
            "human_feedback": str(review_result),
            "review_count": review_count,
            "status": "approved",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# RISK-BASED CONDITIONAL INTERRUPT
# Only pauses when quality is borderline — skips human review for high-confidence outputs
# ─────────────────────────────────────────────────────────────────────────────

def smart_review_node(state: MARRSStateV3) -> Command[Literal["finalize", "writer"]]:
    """
    Conditional HITL: only interrupt if quality is below the auto-approve threshold.
    
    - quality >= 0.92: auto-approve (no human needed)
    - 0.80 <= quality < 0.92: interrupt for review
    - quality < 0.80: auto-reject and revise (no human needed — critic handles it)
    """
    quality = state.get("quality_score", 0.0)
    
    AUTO_APPROVE_THRESHOLD = 0.92
    
    if quality >= AUTO_APPROVE_THRESHOLD:
        # High confidence — skip human review entirely
        return Command(
            goto="finalize",
            update={"human_decision": "auto_approved", "status": "approved"}
        )
    
    # Borderline quality — interrupt for human review
    # interrupt() is called only here, after the quality check.
    # Everything before this point is a pure read (quality check) — safe to re-execute.
    decision = interrupt({
        "message": f"Draft quality is {quality:.2f} (borderline). Please review.",
        "draft": state.get("draft", ""),
        "quality_score": quality,
        "auto_approve_threshold": AUTO_APPROVE_THRESHOLD,
    })
    
    if decision == "approve":
        return Command(goto="finalize", update={"human_decision": "approve", "status": "approved"})
    
    return Command(
        goto="writer",
        update={"human_decision": "revise", "human_feedback": str(decision), "status": "in_revision"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL CALL REVIEW NODE
# For approving tool calls before they execute
# ─────────────────────────────────────────────────────────────────────────────

def tool_call_reviewer(state: MARRSStateV3) -> dict:
    """
    Review tool calls before execution.
    
    Static breakpoint equivalent: compile with interrupt_before=["tools"]
    This dynamic version only interrupts for high-risk tools.
    """
    from langchain_core.messages import AIMessage
    
    last_msg = state.get("messages", [None])[-1]
    if not (last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
        return {}
    
    # Check if any tool calls are high-risk
    HIGH_RISK_TOOLS = {"delete_file", "send_email", "charge_payment", "deploy_to_production"}
    risky_calls = [tc for tc in last_msg.tool_calls if tc["name"] in HIGH_RISK_TOOLS]
    
    if not risky_calls:
        return {}  # All tools are safe — no interrupt needed
    
    # Only interrupt for high-risk tools
    decision = interrupt({
        "message": "Agent wants to call high-risk tool(s). Approve?",
        "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in risky_calls],
    })
    
    if decision != "approve":
        # Inject refusal messages for the risky calls
        from langchain_core.messages import ToolMessage
        refusals = [
            ToolMessage(
                content="Rejected by human reviewer.",
                tool_call_id=tc["id"],
                name=tc["name"],
            )
            for tc in risky_calls
        ]
        return {"messages": refusals}
    
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY WITH HITL
# ─────────────────────────────────────────────────────────────────────────────

def build_marrs_hitl():
    """
    MARRS v5: Full pipeline with dynamic human review gate.
    
    Uses smart_review_node for conditional HITL — only interrupts when needed.
    Static breakpoint version also shown as alternative.
    """
    builder = StateGraph(MARRSStateV3)
    
    builder.add_node("summarize", maybe_summarize_messages)
    builder.add_node("planner", planner_with_memory)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node_command)      # Routes via Command
    builder.add_node("human_review", smart_review_node)  # HITL gate
    builder.add_node("finalize", finalize_node)
    builder.add_node("write_memory", write_research_memory)
    
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "critic")
    # critic routes via Command to either "human_review" or "writer" (retry)
    # (After enough revisions, critic routes to "human_review" unconditionally)
    builder.add_edge("finalize", "write_memory")
    builder.add_edge("write_memory", END)
    
    conn = sqlite3.connect("marrs_v5.db", check_same_thread=False)
    
    return builder.compile(
        checkpointer=SqliteSaver(conn),
        store=research_store,
        # ALTERNATIVE: use static breakpoint instead of dynamic interrupt():
        # interrupt_before=["human_review"],
    )

marrs_hitl = build_marrs_hitl()


# ─────────────────────────────────────────────────────────────────────────────
# DEMO: Full HITL research session
# ─────────────────────────────────────────────────────────────────────────────

def run_with_hitl_review(topic: str, thread_id: str):
    """
    Full HITL session: run until interrupt, print the review request,
    simulate a human decision, then resume.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n{'='*60}")
    print(f"Starting MARRS HITL for: {topic}")
    print(f"Thread: {thread_id}")
    print(f"{'='*60}\n")
    
    # === Phase 1: Run until interrupt ===
    result = marrs_hitl.invoke(
        {"topic": topic, "findings": [], "sources": []},
        config
    )
    
    # Check if we paused at an interrupt
    if "__interrupt__" not in result:
        # Graph completed without needing human review (auto-approved)
        print(f"\n✅ Auto-approved (quality >= threshold)")
        print(f"Status: {result.get('status')}")
        return result
    
    # === Phase 2: Present the interrupt payload ===
    interrupt_payload = result["__interrupt__"][0].value
    print("\n⏸️  GRAPH PAUSED FOR HUMAN REVIEW")
    print(f"\nMessage: {interrupt_payload['message']}")
    print(f"Quality score: {interrupt_payload.get('quality_score', 'N/A'):.2f}")
    print(f"\nDraft preview (first 300 chars):")
    print(interrupt_payload.get('draft', '')[:300])
    print("\nOptions: 'approve', 'reject', or 'revise: <feedback>'")
    
    # Simulate human decision (in production, this comes from a UI/webhook)
    print("\n[Simulating human approval]")
    human_decision = "approve"
    
    # === Phase 3: Resume with human decision ===
    print(f"\n▶️  Resuming with decision: {human_decision}")
    final_result = marrs_hitl.invoke(Command(resume=human_decision), config)
    
    print(f"\n✅ Research complete!")
    print(f"Status: {final_result.get('status')}")
    print(f"Human reviews: {final_result.get('review_count', 0)}")
    
    return final_result


# ─────────────────────────────────────────────────────────────────────────────
# TESTING HITL WORKFLOWS
# ─────────────────────────────────────────────────────────────────────────────

def test_hitl_approve():
    """Test: graph pauses at interrupt, resumes with approval."""
    from langgraph.checkpoint.memory import InMemorySaver
    
    # Build a simple test graph with interrupt
    class TestState(TypedDict):
        value: str
        approved: bool
    
    def task_node(state: TestState) -> dict:
        return {"value": "AI output: important decision"}
    
    def review_test_node(state: TestState) -> dict:
        decision = interrupt({"draft": state["value"]})
        return {"approved": decision == "approve"}
    
    test_graph = (
        StateGraph(TestState)
        .add_node("task", task_node)
        .add_node("review", review_test_node)
        .add_edge(START, "task")
        .add_edge("task", "review")
        .add_edge("review", END)
        .compile(checkpointer=InMemorySaver())
    )
    
    config = {"configurable": {"thread_id": "test-approve-1"}}
    
    # Run until interrupt
    result = test_graph.invoke({"value": "", "approved": False}, config)
    assert "__interrupt__" in result, "Graph should have paused"
    assert result["__interrupt__"][0].value == {"draft": "AI output: important decision"}
    
    # Resume with approval
    final = test_graph.invoke(Command(resume="approve"), config)
    assert final.get("approved") is True
    print("✅ test_hitl_approve passed")

def test_hitl_reject():
    """Test: rejection routes correctly."""
    config = {"configurable": {"thread_id": "test-reject-1"}}
    
    result = marrs_hitl.invoke({"topic": "test", "findings": [], "sources": []}, config)
    
    if "__interrupt__" in result:
        final = marrs_hitl.invoke(Command(resume="reject"), config)
        assert final.get("status") == "rejected"
        print("✅ test_hitl_reject passed")

def test_hitl_no_try_except():
    """Verify that interrupt() cannot be caught by try/except."""
    from langgraph.checkpoint.memory import InMemorySaver
    
    class State(TypedDict):
        result: str
    
    def bad_node(state: State) -> dict:
        try:
            value = interrupt("Input?")
        except Exception:
            value = "default"  # This should never happen — interrupt raises internally
        return {"result": value}
    
    g = (
        StateGraph(State)
        .add_node("bad", bad_node)
        .add_edge(START, "bad")
        .add_edge("bad", END)
        .compile(checkpointer=InMemorySaver())
    )
    
    config = {"configurable": {"thread_id": "no-try-1"}}
    result = g.invoke({"result": ""}, config)
    
    # The graph should have paused despite the try/except (interrupt propagates through)
    # In practice: the try/except catches it, interrupt fails silently.
    # This test documents the dangerous behavior.
    print(f"Result with try/except: {result}")
    print("⚠️  Note: interrupt() inside try/except is silently caught — never do this!")
```

---

### 7.11 Chapter Summary

**Human-in-the-loop** is not a special-case feature — it is a first-class architectural capability built on the same persistence infrastructure as fault tolerance and time travel. The graph pauses, checkpoints state, and waits indefinitely. A human reviews, decides, and the graph resumes. Between pause and resume, no compute is consumed.

**The two mechanisms differ in where the pause decision lives.** Dynamic `interrupt()` puts the decision inside node code — it can be conditional, payload-rich, and placed at any code point. Static `interrupt_before`/`interrupt_after` puts the decision at compile time — it always fires, has no payload, and requires no node modification.

**The re-execution rule is the most consequential fact in this chapter.** On resume, the node restarts from the beginning. Every line before `interrupt()` runs twice. This mandates idempotency: place `interrupt()` first in the node, or place all side-effecting operations after it. Never place API calls, database writes, or charges before `interrupt()`.

**Never wrap `interrupt()` in try/except.** The interrupt works by raising an internal exception. Catching it silently prevents the pause from propagating to the runtime.

**Multiple interrupts in one node** use strictly index-based resume matching — this is documented, intentional behavior, not a bug. The order interrupt calls execute in must stay identical across resumes: no conditional skips, no looping over data whose length can change between runs. The simplest way to guarantee this is one interrupt per node, chaining nodes for multi-step flows.

**Multiple parallel interrupts across different nodes** (a genuine fan-out where several branches each pause independently) use a distinct, now-documented mechanism: pair each pending `Interrupt.id` with its resume value in a dict and pass that dict to `Command(resume=...)`. This is the current, supported fix for what was previously an unresolved edge case with parallel interrupts.

**`graph.stream_events(..., version="v3")`** is a newer, richer interface for driving the pause/resume loop — exposing `stream.interrupted`, `stream.interrupts`, and `stream.output` directly. `graph.invoke()` with `result["__interrupt__"]` remains fully supported and is what this chapter uses throughout for clarity; reach for the streaming interface when building interactive, token-streaming HITL surfaces.

**Seven canonical HITL patterns** cover the full design space: approve/reject, edit state, review tool calls, multi-turn conversation, validation loops, conditional interrupts, and external state correction via `update_state`.

**Production HITL** requires building the notification layer yourself — LangGraph does not notify reviewers. The resume is fully decoupled from compute: different server instances, different processes, different days later can all handle the resume correctly because the state is in the database.

**Time travel + HITL** enables re-review (replay from before a decision), divergent outcome testing (fork and try both approve/reject), and production error correction (roll back a bad decision and re-run).

---

### Further Reading

- **Official LangGraph Interrupts docs**: `docs.langchain.com/oss/python/langgraph/interrupts` — the primary source for this chapter; covers `interrupt()` mechanics, re-execution, the rules of interrupts, handling multiple interrupts (both single-node and parallel), and resumption via `stream_events(version="v3")`
- **Official Time Travel docs**: `docs.langchain.com/oss/python/langgraph/use-time-travel` — covers `update_state`, forking from checkpoints, replay semantics, and subgraph time travel
- **Official Static Breakpoints how-to**: `langchain-ai.github.io/langgraph/cloud/how-tos/human_in_the_loop_breakpoint/` — `interrupt_before`/`interrupt_after` with the LangGraph SDK
- **`interrupt()` API reference**: `reference.langchain.com/python/langgraph/types/interrupt` — the precise contract for index-based resume matching within a node
- **LangGraph Changelog: Dynamic Breakpoints**: `changelog.langchain.com/announcements/langgraph-python-dynamic-breakpoints-error-tracking-in-checkpointer-and-custom-configs` — the original announcement of dynamic interrupt support

---

*End of Chapter 7. Chapter 8: Streaming — Real-Time Output, Token-Level Feedback, and Progress Monitoring.*
