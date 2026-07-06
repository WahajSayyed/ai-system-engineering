# Chapter 13: The Claude Agent SDK

> **CCA Exam Note:** This chapter covers Domain 1 (Agentic Architecture & Orchestration, 27%) and Domain 2 (Tool Design & MCP Integration, 18%). The exam tests the agent loop mechanics, result handling subtypes, permission modes, hooks, tool categories, and when to choose the SDK vs the CLI. This chapter is sourced directly from official documentation at platform.claude.com/docs/en/agent-sdk.

---

## 1. What the Agent SDK Is

The Claude Code CLI gives you an interactive terminal experience. The Claude Agent SDK gives you **that same agent loop as a programmable library** — embed it directly in your own applications.

> 🔑 **Important naming:** The SDK was previously called the "Claude Code SDK." It was renamed to the **Claude Agent SDK**. The old name still appears in some resources — they refer to the same product.

```
Claude Code CLI         → You interact with Claude in a terminal
Claude Agent SDK        → Your application embeds Claude as an agent
```

What you get with the SDK:
- The **same execution loop** that powers Claude Code
- The **same built-in tools** (Read, Edit, Write, Bash, Glob, Grep, WebSearch, etc.)
- The **same CLAUDE.md, skills, and hooks** support
- Programmatic control over tools, permissions, cost limits, and output
- Python and TypeScript implementations

You do **not** need the Claude Code CLI installed to use the SDK.

---

## 2. The Agent Loop — How It Works

Every SDK session follows the same cycle:

```
Your prompt arrives
       ↓
SDK yields SystemMessage (subtype: "init") — session metadata
       ↓
┌─────────────────────────────────────────────────────┐
│  TURN                                               │
│  Claude evaluates → produces AssistantMessage       │
│  (contains: text + tool call requests)              │
│         ↓                                           │
│  SDK executes tools                                 │
│  SDK yields UserMessage (tool results)              │
│         ↓                                           │
│  Results feed back to Claude → next turn            │
└─────────────────────────────────────────────────────┘
       ↓ (repeat until no tool calls)
Final AssistantMessage (text only, no tool calls)
       ↓
ResultMessage — final output, cost, session ID, stop reason
```

**A turn** is one round trip: Claude produces tool calls → SDK executes them → results feed back. This repeats without yielding control to your code. The loop ends when Claude produces output with **no tool calls**.

### Concrete Example — "Fix the failing tests in auth.ts"

```
Turn 1: Claude calls Bash → runs npm test → 3 failures
Turn 2: Claude calls Read → reads auth.ts and auth.test.ts
Turn 3: Claude calls Edit → fixes auth.ts → calls Bash → all pass
Turn 4: Claude produces text only → "Fixed the auth bug, all three tests pass"
        → Loop ends → ResultMessage emitted
```

That was 4 turns: 3 with tool calls, 1 final text response. `max_turns=2` would have stopped before Turn 3.

---

## 3. Message Types

The SDK yields a stream of typed messages as the loop runs. Understanding them is essential for building SDK applications.

| Message Type | When emitted | Key content |
|---|---|---|
| `SystemMessage` | Loop start (`subtype: "init"`) and after compaction (`subtype: "compact_boundary"`) | Session metadata, session ID |
| `AssistantMessage` | After each Claude response | Text content + tool call requests |
| `UserMessage` | After each tool execution | Tool results fed back to Claude |
| `StreamEvent` | Only when partial messages enabled | Raw API streaming events (text deltas, tool input chunks) |
| `ResultMessage` | Always last — exactly once | Final text, cost, token usage, session ID, `subtype`, `stop_reason` |

### Checking Message Types

**Python:** use `isinstance()`:
```python
from claude_agent_sdk import ResultMessage, AssistantMessage

async for message in query(prompt="...", options=options):
    if isinstance(message, ResultMessage):
        # handle result
    elif isinstance(message, AssistantMessage):
        # handle assistant turn
```

**TypeScript:** check the `type` string field:
```typescript
for await (const message of query({ prompt: "...", options })) {
    if (message.type === "result") {
        // handle result
    } else if (message.type === "assistant") {
        // note: content blocks at message.message.content, not message.content
    }
}
```

> ⚠️ **TypeScript gotcha:** `AssistantMessage` and `UserMessage` wrap the raw API message in a `.message` field. Content blocks are at `message.message.content`, not `message.content`.

---

## 4. The ResultMessage — Handling Outcomes

The `ResultMessage` is the most important message. Always check its `subtype` before reading the result.

| `subtype` | What happened | `result` field available? |
|---|---|---|
| `success` | Claude finished normally | ✅ Yes |
| `error_max_turns` | Hit `maxTurns` limit | ❌ No |
| `error_max_budget_usd` | Hit `maxBudgetUsd` cost limit | ❌ No |
| `error_during_execution` | API failure or cancelled request | ❌ No |
| `error_max_structured_output_retries` | Structured output validation failed | ❌ No |

```python
if isinstance(message, ResultMessage):
    if message.subtype == "success":
        print(f"Done: {message.result}")
    elif message.subtype == "error_max_turns":
        print(f"Hit turn limit. Resume session {message.session_id}")
    elif message.subtype == "error_max_budget_usd":
        print("Hit budget. Reduce scope or increase budget.")
    else:
        print(f"Failed: {message.subtype}")

    # Always available regardless of subtype
    if message.total_cost_usd is not None:
        print(f"Cost: ${message.total_cost_usd:.4f}")
    print(f"Turns: {message.num_turns}")
    print(f"Session ID: {message.session_id}")  # Save for resumption
```

The `stop_reason` field tells you why Claude stopped on its final turn:
- `end_turn` — finished normally
- `max_tokens` — hit output token limit
- `refusal` — model declined the request

---

## 5. Built-in Tools

The SDK includes every tool that powers Claude Code:

| Category | Tools | What they do |
|---|---|---|
| **File operations** | `Read`, `Edit`, `Write` | Read, modify, and create files |
| **Search** | `Glob`, `Grep` | Find files by pattern, search content with regex |
| **Execution** | `Bash` | Run shell commands, scripts, git operations |
| **Web** | `WebSearch`, `WebFetch` | Search the web, fetch and parse pages |
| **Discovery** | `ToolSearch` | Load MCP tools on-demand instead of preloading all |
| **Orchestration** | `Task`, `Skill`, `AskUserQuestion`, `TodoWrite` | Spawn subagents, invoke skills, ask the user, track tasks |

Beyond built-in tools:
- **MCP servers** — connect databases, browsers, external APIs
- **Custom tool handlers** — define your own tools with custom execution logic

### Tool Permissions — Three Mechanisms

```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Glob", "Grep"],     # Auto-approved, no prompting
    disallowed_tools=["Bash", "Edit"],           # Blocked regardless of other settings
    permission_mode="acceptEdits",               # Mode for everything else
)
```

**`allowed_tools`** — tools listed here execute automatically without prompting.
**`disallowed_tools`** — tools listed here are always blocked, regardless of other settings.
**`permission_mode`** — controls what happens to tools not covered by the above two.

You can scope Bash with specific command patterns:
```python
allowed_tools=["Read", "Glob", "Bash(npm:*)"]  # Only npm commands allowed
```

### Parallel Tool Execution

When Claude requests multiple tools in one turn:
- **Read-only tools** (`Read`, `Glob`, `Grep`, read-only MCP tools) → run concurrently
- **State-modifying tools** (`Edit`, `Write`, `Bash`) → run sequentially to avoid conflicts
- **Custom tools** → sequential by default; mark as `readOnly` to enable parallel execution

---

## 6. Controlling the Loop

### Turns and Budget

```python
options = ClaudeAgentOptions(
    max_turns=30,           # Stop after 30 tool-use turns
    max_budget_usd=0.50,    # Stop after $0.50 spent
)
```

`max_turns` counts **tool-use turns only** — the final text-only response doesn't count.

When either limit is hit, the loop stops and `ResultMessage` has the corresponding error subtype. Always set at least one of these in production — otherwise an open-ended prompt can run indefinitely.

### Effort Level

```python
options = ClaudeAgentOptions(effort="high")
```

| Level | Reasoning | Best for |
|---|---|---|
| `"low"` | Minimal | File lookups, directory listings |
| `"medium"` | Balanced | Routine edits, standard tasks |
| `"high"` | Thorough | Refactors, debugging (TypeScript SDK default) |
| `"max"` | Maximum | Multi-step problems, deep analysis |

> 🔑 `effort` trades latency and token cost for reasoning depth. It is separate from extended thinking — they are independent features.

### Permission Modes

| Mode | Behaviour |
|---|---|
| `"default"` | Tools not in `allowed_tools` trigger your approval callback; no callback = deny |
| `"acceptEdits"` | Auto-approves file edits; other tools follow default rules |
| `"plan"` | Read-only exploration only; Claude produces a plan, no execution |
| `"dontAsk"` (TypeScript only) | Never prompts; pre-approved tools run, everything else denied |
| `"bypassPermissions"` | All allowed tools run without asking. **Only for isolated environments.** Cannot run as root on Unix. |

### Model Selection

```python
options = ClaudeAgentOptions(model="claude-sonnet-4-6")
```

If not set, the SDK uses Claude Code's default based on your auth method. Always pin explicitly in production for consistent behaviour.

---

## 7. A Complete Example

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

async def run_agent():
    session_id = None

    async for message in query(
        prompt="Find and fix the bug causing test failures in the auth module",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
            setting_sources=["project"],  # Load CLAUDE.md, skills, hooks
            max_turns=30,
            max_budget_usd=1.00,
            effort="high",
            permission_mode="acceptEdits",
        ),
    ):
        if isinstance(message, ResultMessage):
            session_id = message.session_id

            if message.subtype == "success":
                print(f"Done: {message.result}")
            elif message.subtype == "error_max_turns":
                print(f"Hit turn limit. Resume: {session_id}")
            elif message.subtype == "error_max_budget_usd":
                print("Hit budget limit.")
            else:
                print(f"Failed: {message.subtype}")

            if message.total_cost_usd is not None:
                print(f"Cost: ${message.total_cost_usd:.4f}")

asyncio.run(run_agent())
```

---

## 8. Context Window in the SDK

The context window accumulates across turns within a session. It does not reset between turns.

### What Consumes Context

| Source | When it loads | Impact |
|---|---|---|
| System prompt | Every request | Fixed, always present |
| CLAUDE.md files | Session start (with `setting_sources=["project"]`) | Full content but prompt-cached after first request |
| Tool definitions | Every request | Each tool adds its schema |
| Conversation history | Accumulates over turns | Grows with every prompt, response, tool input, output |
| Skill descriptions | Session start | Short summaries; full content loads only on invocation |

**Prompt caching:** Content that stays the same across turns (system prompt, tool definitions, CLAUDE.md) is automatically cached. Only the first request in a session pays the full cost for that content.

### Automatic Compaction

When context approaches its limit, the SDK automatically summarises older history. The SDK emits `SystemMessage` with `subtype: "compact_boundary"` when this happens.

After compaction, specific instructions from early in the conversation may be lost. **Put persistent rules in CLAUDE.md** (loaded via `setting_sources`) rather than the initial prompt — CLAUDE.md is re-injected on every request and survives compaction.

### Customising Compaction

```python
# Trigger compaction manually
async for message in query(prompt="/compact", options=options):
    pass

# Add summarisation instructions to CLAUDE.md
# (compactor reads CLAUDE.md like any other context)
"""
## When Compacting
Always preserve:
- The list of files modified so far
- Any test failure messages
- The agreed approach for the current task
"""
```

### Context Efficiency Strategies

- **Subagents for subtasks** — each starts with fresh context; only the summary returns to the parent
- **Scope tool lists** — every tool definition consumes context space
- **MCP tool search** — loads tool definitions on demand instead of preloading all
- **Lower effort for simple tasks** — reduces token usage per turn

---

## 9. SDK vs CLI vs Messages API

A question frequently tested on the CCA exam. Three surfaces for building with Claude:

| | Messages API | Agent SDK | Claude Code CLI |
|---|---|---|---|
| **What it is** | Raw HTTP API | Agent loop as a library | Interactive terminal tool |
| **Who manages the loop** | You | SDK | Claude Code |
| **Built-in tools** | None | Full (Read, Edit, Bash, etc.) | Full |
| **CLAUDE.md support** | No | Yes (via `setting_sources`) | Yes |
| **Best for** | Custom integrations, chat apps | Embedding autonomous agents in apps | Developer coding sessions |
| **Output** | API response object | Typed message stream | Terminal output |

**Use the Messages API when:** you need fine-grained control over every API call, you're building a custom tool loop, or you're integrating Claude into an existing chat system.

**Use the Agent SDK when:** you want an autonomous agent embedded in your application without building the tool loop yourself.

**Use the CLI when:** you're a developer coding interactively at the terminal.

---

## 10. Sessions and Continuity

Capture the session ID from `ResultMessage.session_id` to resume later:

```python
# First run
async for message in query(prompt="Analyse the auth module", options=options):
    if isinstance(message, ResultMessage):
        saved_session_id = message.session_id

# Resume — full context from previous run is restored
async for message in query(
    prompt="Now fix the issues you found",
    options=ClaudeAgentOptions(session_id=saved_session_id)
):
    pass

# Fork — branch into a different approach, original session untouched
async for message in query(
    prompt="Try a different approach using Redis instead",
    options=ClaudeAgentOptions(
        session_id=saved_session_id,
        fork_session=True
    )
):
    pass
```

When resuming, the full context is restored: files read, analysis performed, actions taken.

---

## 11. Hooks in the SDK

Hooks fire at specific points in the loop without consuming context (they run in your application process):

| Hook | When it fires | Common use |
|---|---|---|
| `PreToolUse` | Before a tool executes | Validate inputs, block dangerous commands |
| `PostToolUse` | After a tool returns | Audit outputs, trigger side effects |
| `UserPromptSubmit` | When a prompt is sent | Inject additional context |
| `Stop` | When the agent finishes | Validate result, save session state |
| `SubagentStart/Stop` | Subagent spawns/completes | Track parallel task results |
| `PreCompact` | Before context compaction | Archive full transcript |

A `PreToolUse` hook that rejects a tool call prevents it from executing — Claude receives the rejection and tries a different approach.

---

## 12. Loading Claude Code Features

To give your SDK agent access to CLAUDE.md, skills, and hooks from the project:

```python
options = ClaudeAgentOptions(
    setting_sources=["project"]  # Python
)
```

```typescript
const options: Options = {
    settingSources: ['project']  // TypeScript
}
```

This loads from the current directory:
- `CLAUDE.md` / `.claude/CLAUDE.md` — project context and rules
- `.claude/skills/` — reusable workflows
- `.claude/agents/` — custom subagent definitions
- Hooks from `.claude/settings.json`

Without `setting_sources`, the SDK runs without any filesystem-based configuration.

---

## 13. Internal Mechanics — The Loop Under the Hood

```
query(prompt, options) called
       ↓
SDK constructs system prompt:
  + Base agent instructions
  + Tool definitions (from allowed_tools)
  + CLAUDE.md content (if setting_sources enabled)
  + Skill descriptions
       ↓
API call: send prompt + system prompt + tool definitions
       ↓
Claude responds with AssistantMessage:
  → Text content
  → Tool call requests (zero or more)
       ↓
SDK yields AssistantMessage to your code
       ↓
For each tool call:
  → Check against allowed_tools, disallowed_tools, permission_mode
  → Run PreToolUse hooks
  → Execute tool (concurrent for read-only, sequential for writes)
  → Run PostToolUse hooks
  → Collect result
       ↓
SDK yields UserMessage (tool results) to your code
       ↓
API call: append tool results to conversation history
       ↓
[Loop repeats]
       ↓
Claude responds with text only (no tool calls)
       ↓
SDK yields final AssistantMessage
SDK yields ResultMessage
       ↓
query() iterator ends
```

---

## 14. CCA Exam — Practice Questions

**Q1:** An SDK agent hits `error_max_turns` on a complex refactoring task. The developer wants to continue from where it left off. What is the correct approach?

- A) Re-run the agent with the same prompt from scratch
- B) Increase `max_turns` and re-run from scratch
- C) Save `message.session_id` from the `ResultMessage` and resume that session ✅
- D) Use `--fork-session` to branch from the failure point

**Q2:** An agent needs to run `npm test` and `git commit`, but should not be able to run arbitrary bash commands. What is the correct tool configuration?

- A) `allowed_tools=["Bash"]`
- B) `allowed_tools=["Bash(npm:*)", "Bash(git:*)"]` ✅
- C) `disallowed_tools=["Bash"]` and add npm and git separately
- D) `permission_mode="acceptEdits"` with no tool restrictions

**Q3:** A developer wants an SDK agent to load their project's CLAUDE.md instructions and follow them. What configuration is required?

- A) Pass CLAUDE.md contents as the system prompt
- B) Set `setting_sources=["project"]` in the options ✅
- C) Use `--append-system-prompt` flag
- D) Read CLAUDE.md manually and inject into the prompt

**Q4:** What is the difference between setting `effort="max"` and enabling extended thinking?

- A) They are identical — both increase reasoning depth
- B) `effort` controls reasoning depth within each response; extended thinking produces visible chain-of-thought blocks — they are independent features ✅
- C) `effort="max"` replaces extended thinking
- D) Extended thinking requires `effort="high"` as a prerequisite

**Q5:** An SDK agent reads many large files and exhausts the context window mid-task. What is the most effective architectural fix?

- A) Increase the model's context window size
- B) Use `/compact` at the start of each session
- C) Offload subtasks to subagents — each starts with fresh context and only returns a summary to the parent ✅
- D) Reduce `max_turns` to prevent context accumulation

**Q6:** A `ResultMessage` arrives with `subtype: "error_max_budget_usd"`. What is true about this message?

- A) The `result` field contains a partial result up to the budget limit
- B) The `result` field is not available; `total_cost_usd`, `num_turns`, and `session_id` are still present ✅
- C) The session cannot be resumed
- D) The agent made no progress and no tokens were consumed

---

## 15. Practical Exercise — Build an Agent

**Exercise A — First agent:**

```bash
# Python
pip install claude-agent-sdk
export ANTHROPIC_API_KEY="your-key"
```

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

async def main():
    async for message in query(
        prompt="List all Python files in this directory and summarise what each does",
        options=ClaudeAgentOptions(
            allowed_tools=["Glob", "Read"],
            max_turns=10,
            max_budget_usd=0.10,
        ),
    ):
        if isinstance(message, ResultMessage):
            if message.subtype == "success":
                print(message.result)
            print(f"Cost: ${message.total_cost_usd:.4f}")

asyncio.run(main())
```

**Exercise B — Bug-fixing agent with CLAUDE.md:**

Create a `CLAUDE.md` with coding rules, then:

```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
    setting_sources=["project"],  # Load your CLAUDE.md
    max_turns=20,
    effort="high",
)
```

Run it on a project with a known bug. Verify it follows your CLAUDE.md conventions.

**Exercise C — Session resume:**

```python
# Run 1: analyse
session_id = None
async for message in query(prompt="Analyse src/auth/ for issues", options=options):
    if isinstance(message, ResultMessage):
        session_id = message.session_id
        print(f"Session: {session_id}")

# Run 2: fix using same session context
options_resume = ClaudeAgentOptions(session_id=session_id, ...)
async for message in query(prompt="Fix the issues you found", options=options_resume):
    pass
```

**Exercise D — Handle all result subtypes:**

Deliberately set `max_turns=1` on a complex task and verify you get `error_max_turns`. Then handle it by resuming.

---

## Chapter 13 Summary

| Concept | Key Takeaway | CCA Domain |
|---|---|---|
| SDK = embedded agent loop | Same tools and loop as Claude Code CLI, in your application | Domain 1 |
| Rename | "Claude Code SDK" → "Claude Agent SDK" | Domain 1 |
| Turn definition | One tool-call round trip; final text response doesn't count | Domain 1 |
| Five message types | SystemMessage, AssistantMessage, UserMessage, StreamEvent, ResultMessage | Domain 1 |
| ResultMessage subtypes | Always check `subtype` before reading `result` — only `success` has `result` | Domain 1 |
| allowed_tools | Auto-approves tools — no prompting in production | Domain 2 |
| Bash scoping | `"Bash(npm:*)"` restricts to specific command patterns | Domain 2 |
| Parallel execution | Read-only tools concurrent; state-modifying sequential | Domain 2 |
| effort vs extended thinking | Independent features — effort = reasoning depth per response | Domain 1 |
| setting_sources | Required to load CLAUDE.md, skills, hooks from filesystem | Domain 3 |
| Compaction | CLAUDE.md survives (re-injected every request); conversation instructions may not | Domain 5 |
| Session resume | Save `session_id` from ResultMessage; pass back to resume | Domain 1 |

### CCA Domain Coverage

| CCA Domain | Concepts Covered |
|---|---|
| **Domain 1: Agentic Architecture & Orchestration (27%)** | Agent loop mechanics, message types, result handling, session management, subagents for context |
| **Domain 2: Tool Design & MCP Integration (18%)** | Built-in tools, permission model, Bash scoping, parallel execution, MCP in SDK |
| **Domain 3: Claude Code Config & Workflows (20%)** | `setting_sources`, CLAUDE.md in SDK, skills loading |
| **Domain 5: Context Management & Reliability (15%)** | Context accumulation, compaction, subagents for fresh context |

---

## ✅ Before Chapter 14

- [ ] Installed the Agent SDK and run the first example
- [ ] Built an agent with `allowed_tools` and `max_turns`
- [ ] Loaded `setting_sources=["project"]` and verified CLAUDE.md was followed
- [ ] Handled all five `ResultMessage.subtype` values
- [ ] Saved a session ID and successfully resumed a session
- [ ] Used Bash scoping to restrict to specific commands
- [ ] Answered all six practice questions correctly

---

*Previous: [Chapter 12 — CI/CD Automation](./chapter-12-cicd-automation.md)*
*Next: Chapter 14 — Security, Trust & Safety (coming soon)*
