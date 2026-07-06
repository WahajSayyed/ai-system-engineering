# Chapter 10: Multi-Agent & Parallel Workflows

> **CCA Exam Note:** This chapter covers Domain 1 (Agentic Architecture & Orchestration, 27%) — the heaviest domain on the exam. Multi-agent patterns, task decomposition, hub-and-spoke orchestration, and context isolation are all directly tested. This chapter is built from official Claude Code documentation at code.claude.com/docs/en/sub-agents and code.claude.com/docs/en/agent-teams.

---

## 1. The Scaling Problem — Why One Claude Isn't Always Enough

A single Claude Code session has three fundamental limits:

1. **One context window** — everything you read, everything you write, every command output, accumulates in one finite bucket. Long tasks degrade as context fills.
2. **Sequential execution** — Claude can't do two things at once. Every file read, every command, happens one after another.
3. **No specialisation** — the same Claude handles codebase exploration, code review, database queries, and documentation. It has to context-switch between all of them.

Multi-agent workflows solve all three:

| Problem | Solution |
|---|---|
| Context overflow | Each agent gets its own fresh window |
| Sequential bottleneck | Multiple agents run in parallel |
| No specialisation | Each agent has a custom system prompt, tools, and model |

The tradeoff is coordination overhead and higher token cost. Understanding when that tradeoff is worth it — and when it isn't — is what the CCA exam tests.

---

## 2. The Two Multi-Agent Primitives

Claude Code offers two distinct multi-agent systems. They are fundamentally different and not interchangeable.

### Subagents — Delegation Within a Session

Subagents run **within your main session**. The main agent spawns them, they do work in their own context window, and they return a summary when done.

```
Main session
    ↓ spawns
    Subagent (own context window)
        ↓ does work
        ↓ returns summary
    Main session receives result
```

Key characteristics:
- Subagents **cannot spawn other subagents** (no nesting)
- They **only report back to the main agent** — no direct communication between subagents
- Context isolation: their file reads stay out of your main session
- Lower coordination overhead

### Agent Teams — Coordinated Independent Sessions

Agent teams coordinate **multiple full Claude Code sessions**. Each teammate is a complete, independent session with its own context, tools, and permissions.

```
Lead session (your main terminal)
    ↓ spawns via shared task list
    Teammate A (full independent session)
    Teammate B (full independent session)  ← all running simultaneously
    Teammate C (full independent session)
    ↓ teammates message each other directly
    ↓ lead synthesises results
```

Key characteristics:
- Teammates **communicate directly with each other** — not just through the lead
- Each teammate is a **full independent Claude Code session**
- Shared task list coordinates work; teammates self-claim tasks
- Significantly **higher token cost** — each teammate has its own full context
- **Experimental** — must be explicitly enabled

> ⚠️ Agent teams require Claude Code v2.1.32+ and must be enabled: set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings.json or environment.

### Which to Use — The Decision Framework

| | Subagents | Agent Teams |
|---|---|---|
| **Context** | Own window; results return to main | Own window; fully independent |
| **Communication** | Report back to main agent only | Teammates message each other directly |
| **Coordination** | Main agent manages all work | Shared task list + self-coordination |
| **Best for** | Focused tasks where only result matters | Complex work requiring discussion and collaboration |
| **Token cost** | Lower — results summarised back | Higher — each teammate is a full Claude instance |
| **Nesting** | Cannot spawn subagents | Cannot spawn teams (no nested teams) |
| **Status** | Stable | Experimental |

**Use subagents when:** workers need to do focused work and report back.
**Use agent teams when:** workers need to share findings, challenge each other, and coordinate on their own.

---

## 3. Built-in Subagents

Claude Code includes built-in subagents that activate automatically. Understanding them matters because they directly explain why Claude reads files without being told to.

### Explore
- **Model:** Haiku (fast, low-latency)
- **Tools:** Read-only (Write and Edit denied)
- **Purpose:** Searching and analysing codebases without making changes
- **Thoroughness levels:** Claude specifies `quick`, `medium`, or `very thorough` when invoking it

When you ask "explain how this codebase works", Claude often delegates to Explore. The file reads happen in Explore's context window — not yours. This is why `/context` shows lower usage than you might expect after exploration tasks.

### Plan
- **Model:** Inherits from main conversation
- **Tools:** Read-only
- **Purpose:** Codebase research during plan mode, before presenting a plan

When you enter plan mode and Claude needs to research your codebase, it delegates to Plan. This prevents infinite nesting (subagents can't spawn subagents) while still gathering necessary context.

### General-purpose
- **Model:** Inherits from main conversation
- **Tools:** All tools
- **Purpose:** Complex multi-step tasks that require both exploration and action

Delegated when the task requires both exploration and modification, complex reasoning, or multiple dependent steps.

---

## 4. Creating Custom Subagents

### File Format

Subagents are Markdown files with YAML frontmatter:

```markdown
---
name: code-reviewer
description: Reviews code for quality and best practices. Use proactively after code changes.
tools: Read, Grep, Glob, Bash
model: sonnet
---

You are a senior code reviewer. When invoked:
1. Run git diff to see recent changes
2. Focus on modified files

Review checklist:
- Code clarity and naming
- Error handling
- No exposed secrets
- Test coverage

Provide feedback by priority: Critical / Warnings / Suggestions.
```

The YAML frontmatter defines configuration. The markdown body becomes the subagent's system prompt. Subagents receive **only this system prompt** — not the full Claude Code system prompt and not the parent conversation history.

### Key Frontmatter Fields

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Unique ID — lowercase letters and hyphens |
| `description` | ✅ | **When Claude should delegate** — write this carefully |
| `tools` | No | Allowlist. Inherits all tools if omitted |
| `disallowedTools` | No | Denylist — removed from inherited or specified list |
| `model` | No | `sonnet`, `opus`, `haiku`, full model ID, or `inherit` (default) |
| `permissionMode` | No | `default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, `plan` |
| `maxTurns` | No | Max agentic turns before stopping |
| `skills` | No | Skills injected at startup — full content, not just descriptions |
| `mcpServers` | No | MCP servers scoped to this subagent only |
| `memory` | No | `user`, `project`, or `local` for persistent cross-session memory |
| `background` | No | `true` to always run as a background task |
| `isolation` | No | `worktree` to run in a temporary git worktree |
| `color` | No | UI display colour |

> 🔑 **The description field is critical.** Claude uses it to decide when to delegate. Write it like a trigger: "Use proactively when...", "Delegate when the task involves...", "Invoke immediately after code changes."

### Tool Control — Allowlist vs Denylist

```yaml
# Allowlist: only these tools (nothing else)
tools: Read, Grep, Glob, Bash

# Denylist: everything except these
disallowedTools: Write, Edit

# If both set: denylist applied first, then allowlist resolved
# A tool in both lists is removed
```

### Model Selection

Model resolution order when Claude invokes a subagent:
1. `CLAUDE_CODE_SUBAGENT_MODEL` environment variable (if set)
2. Per-invocation `model` parameter from Claude
3. Subagent definition's `model` frontmatter
4. Main conversation's model

Use `haiku` for fast exploration (low cost), `sonnet` for balanced tasks, `opus` for complex reasoning.

### Storage Locations and Scope

| Location | Scope | Priority |
|---|---|---|
| Managed settings | Organisation-wide | 1 (highest) |
| `--agents` CLI flag | Current session only | 2 |
| `.claude/agents/` | Current project | 3 |
| `~/.claude/agents/` | All your projects | 4 |
| Plugin's `agents/` | Where plugin enabled | 5 (lowest) |

When multiple subagents share the same name, higher priority wins.

> 💡 **Check `.claude/agents/` into version control** so your team can use and improve project subagents collaboratively.

---

## 5. Working with Subagents

### Automatic Delegation

Claude automatically delegates based on the task description and the subagent's `description` field. To encourage proactive delegation, include phrases like "use proactively" in the description.

### Invoking Subagents Explicitly

Three escalating levels of explicitness:

**Natural language** — suggest it, Claude decides:
```
Use the code-reviewer subagent to look at my recent changes
```

**@-mention** — guarantees that specific subagent runs for one task:
```
@"code-reviewer (agent)" look at the auth module changes
```

**Session-wide** — the whole session uses that subagent's system prompt:
```bash
claude --agent code-reviewer
```

Or set as default for a project in `.claude/settings.json`:
```json
{ "agent": "code-reviewer" }
```

### Foreground vs Background

**Foreground subagents** block the main conversation until complete. Permission prompts pass through to you.

**Background subagents** run concurrently while you continue working. Before launching, Claude Code prompts upfront for all tool permissions the subagent will need. Once running, the subagent auto-denies anything not pre-approved.

```
# Ask Claude to background a task
"Run the test suite in the background and report back when done"

# Or press Ctrl+B to background a running task
```

> ⚠️ If a background subagent fails due to missing permissions, start a new foreground subagent with the same task — it can request permissions interactively.

### Persistent Memory

Give a subagent cross-session memory with the `memory` field:

```yaml
---
name: code-reviewer
description: Reviews code for quality
memory: user
---

As you review code, update your agent memory with patterns,
conventions, and recurring issues you discover.
```

| Scope | Location | Use when |
|---|---|---|
| `user` | `~/.claude/agent-memory/<name>/` | Knowledge applies across all projects |
| `project` | `.claude/agent-memory/<name>/` | Project-specific, shareable via version control |
| `local` | `.claude/agent-memory-local/<name>/` | Project-specific, not in version control |

Memory works like auto memory: first 200 lines or 25KB of `MEMORY.md` loaded at start. Topic files loaded on demand.

---

## 6. Common Subagent Patterns

These patterns are tested in the CCA exam's Domain 1.

### Pattern 1: Isolate High-Volume Operations

Keep verbose output out of your main context:

```
Use a subagent to run the full test suite and report only the failing tests
with their error messages.
```

The test output — potentially thousands of lines — stays in the subagent's context. You get a clean summary.

### Pattern 2: Parallel Research

For independent investigations:

```
Research the authentication, database, and API modules in parallel
using separate subagents.
```

Each subagent explores its module independently. Claude synthesises findings. Works best when the research paths don't depend on each other.

> ⚠️ Running many subagents that each return detailed results can still consume significant main context when results come back. If this is a concern, use agent teams instead.

### Pattern 3: Chain Subagents

For sequential workflows where each step feeds the next:

```
Use the code-reviewer subagent to find performance issues, then use the
optimizer subagent to fix them.
```

Claude passes relevant context from the first subagent to the second.

### Pattern 4: MCP-Scoped Subagents

Scope expensive MCP server tools to specific subagents — keeps them out of main context:

```yaml
---
name: browser-tester
description: Tests features in a browser using Playwright
mcpServers:
  - playwright:
      type: stdio
      command: npx
      args: ["-y", "@playwright/mcp@latest"]
---
```

The Playwright tools are available only to this subagent. The main conversation never loads them.

---

## 7. Subagent Hooks

Hooks let you automate actions at subagent lifecycle points.

### In Subagent Frontmatter (Scoped to That Subagent)

```yaml
---
name: code-reviewer
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-command.sh"
  PostToolUse:
    - matcher: "Edit|Write"
      hooks:
        - type: command
          command: "./scripts/run-linter.sh"
---
```

These hooks only run while that subagent is active.

### In `settings.json` (Main Session Hooks for Subagent Events)

```json
{
  "hooks": {
    "SubagentStart": [
      {
        "matcher": "db-agent",
        "hooks": [{ "type": "command", "command": "./scripts/setup-db.sh" }]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [{ "type": "command", "command": "./scripts/cleanup.sh" }]
      }
    ]
  }
}
```

`SubagentStart` and `SubagentStop` fire in the main session when subagents begin and end.

---

## 8. Agent Teams — Deep Dive

### Enabling Agent Teams

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

### Architecture Components

| Component | Role |
|---|---|
| **Team lead** | Your main Claude Code session — creates team, spawns teammates, coordinates work |
| **Teammates** | Separate full Claude Code instances — each owns assigned tasks |
| **Task list** | Shared list of work items — teammates self-claim available tasks |
| **Mailbox** | Messaging system — teammates communicate directly with each other |

Storage locations (auto-managed — do not edit by hand):
- Team config: `~/.claude/teams/{team-name}/config.json`
- Task list: `~/.claude/tasks/{team-name}/`

### Starting a Team

Simply describe the task and structure:

```
I'm designing a CLI tool for tracking TODO comments. Create an agent team
to explore this from different angles: one teammate on UX, one on technical
architecture, one playing devil's advocate.
```

Claude creates the team, spawns teammates, and begins coordination.

### Display Modes

**In-process** (default): all teammates run in your main terminal. Use `Shift+Down` to cycle through teammates and interact.

**Split panes**: each teammate gets its own pane (requires tmux or iTerm2). See everyone's output simultaneously.

```bash
# Force in-process mode
claude --teammate-mode in-process
```

Set default in `~/.claude.json`:
```json
{ "teammateMode": "in-process" }
```

### Controlling a Team

**Talk to teammates directly:** `Shift+Down` to cycle, then type to send a message. Press `Ctrl+T` to toggle the task list.

**Specify teammates and models:**
```
Create a team with 4 teammates. Use Sonnet for each teammate.
```

**Require plan approval before implementation:**
```
Spawn an architect teammate to refactor auth. Require plan approval before changes.
```

**Shut down a teammate:**
```
Ask the researcher teammate to shut down.
```

**Clean up:**
```
Clean up the team.
```
> ⚠️ Always use the **lead** to clean up. Teammates should not run cleanup.

### Agent Team Quality Hooks

Three hook events specific to agent teams:

| Hook | When it fires | Exit code 2 effect |
|---|---|---|
| `TeammateIdle` | Teammate about to go idle | Sends feedback, keeps teammate working |
| `TaskCreated` | Task being created | Prevents creation, sends feedback |
| `TaskCompleted` | Task being marked complete | Prevents completion, sends feedback |

---

## 9. Internal Mechanics — How Multi-Agent Context Works

### Subagent Context Isolation

```
Main session context window (200K tokens):
  CLAUDE.md + auto memory + conversation  →  ~30,000 tokens used

Subagent spawned for codebase exploration:
  Fresh context window starts at zero
  Reads 50 files × 3,000 tokens = 150,000 tokens
  Only its SUMMARY returns to main session: ~2,000 tokens

Net impact on main session: +2,000 tokens (the summary)
Without subagent: +150,000 tokens
```

This is the core value of subagent delegation for high-volume operations. The expensive work happens in an isolated context; you receive only the result.

### Teammate Context in Agent Teams

Each teammate starts like a fresh Claude Code session:
- Loads CLAUDE.md, MCP servers, and skills normally
- Does **not** inherit the lead's conversation history
- Receives only the spawn prompt from the lead

```
Lead spawns Teammate A with:
  "Review src/auth/ for security issues. Focus on token handling,
  session management, and input validation. JWT tokens in httpOnly cookies."

Teammate A starts fresh:
  ✅ Loads CLAUDE.md
  ✅ Loads MCP servers
  ❌ Does NOT see lead's conversation history
  ✅ Receives the spawn prompt above
```

### The Shared Task List

Teammates coordinate via a shared task list:

```
Task List:
  [PENDING]      Review authentication module
  [IN_PROGRESS]  Review payment module  ← Teammate B claimed this
  [COMPLETED]    Review user module     ← Teammate A finished

Dependencies are tracked automatically:
  "Generate final report" → depends on all review tasks
  → Stays PENDING until all reviews complete
  → Automatically unblocks when dependencies resolve
```

Task claiming uses file locking to prevent race conditions.

---

## 10. When NOT to Use Multi-Agent Workflows

This is as important as knowing when to use them — and directly tested on the CCA exam.

### Don't Use Subagents When:
- The task needs frequent back-and-forth or iterative refinement
- Multiple phases share significant context (planning → implementation → testing)
- You're making a quick targeted change
- Latency matters — subagents start fresh and take time to gather context

### Don't Use Agent Teams When:
- Tasks are sequential — coordination overhead exceeds the benefit
- Workers need to edit the same files — this causes overwrites
- Tasks have many dependencies — self-coordination breaks down
- The work is routine — single session is more cost-effective
- You need stable, production-ready behaviour — teams are experimental

### Choosing Between All Options

```
Is the task quick and in context?
  → Main conversation directly

Is it a quick side question?
  → /btw (no context cost, no tools)

Is it a reusable prompt/workflow, not isolated work?
  → Skill (runs in main conversation)

Is it an isolated task with verbose output?
  → Subagent (fresh context, summary returns)

Do workers need to communicate with each other?
  → Agent team (experimental, higher cost)

Do you need parallel sessions with full isolation?
  → Git worktrees + multiple manual sessions
```

---

## 11. CCA Exam — Practice Questions

**Q1:** A developer needs Claude to explore 200 files across a large codebase without filling their main context window. What is the most appropriate approach?

- A) Use `/compact` before exploration begins
- B) Delegate to a subagent with read-only tools — it explores in its own context and returns a summary ✅
- C) Use agent teams to explore files in parallel
- D) Add all relevant files to CLAUDE.md using `@` references

**Q2:** You have three independent research tasks: reviewing the auth module, reviewing the payment module, and reviewing the API module. Workers need to compare findings and challenge each other's conclusions. Which approach is most appropriate?

- A) Three sequential subagents, each building on the last
- B) Three parallel subagents that all report back to main
- C) An agent team where teammates can message each other and debate findings ✅
- D) A single session doing all three reviews sequentially

**Q3:** A custom subagent has `tools: Read, Grep, Glob` and `disallowedTools: Bash`. What tools does it have access to?

- A) Read, Grep, Glob, and Bash
- B) Read, Grep, and Glob only — disallowedTools is applied first, then tools resolved ✅
- C) Only the tools not in disallowedTools
- D) All tools since disallowedTools takes precedence over tools

**Q4:** A subagent's `description` field says "Reviews code for quality". Claude rarely delegates to it automatically. What is the most likely cause and fix?

- A) The model field is not set — set it to `sonnet`
- B) The description is too vague — rewrite it to specify when to delegate, e.g. "Use proactively after any code changes" ✅
- C) The tools field is too restrictive
- D) Add `background: true` to make it run automatically

**Q5:** An agent team teammate needs access to a Playwright MCP server that should not be available to the main session or other teammates. What is the correct configuration?

- A) Add Playwright to the project `.mcp.json`
- B) Add Playwright to the teammate's subagent definition `mcpServers` field — it connects when the subagent starts and disconnects when it finishes ✅
- C) Add Playwright to the user scope MCP config
- D) Add Playwright to CLAUDE.md as an @ reference

**Q6:** An agent team lead starts implementing tasks itself instead of waiting for teammates. What is the correct intervention?

- A) Shut down the team and use subagents instead
- B) Reduce the number of teammates
- C) Tell the lead directly: "Wait for your teammates to complete their tasks before proceeding" ✅
- D) Use `TeammateIdle` hook to prevent the lead from working

**Q7:** Which component of the agent team architecture prevents two teammates from claiming the same task simultaneously?

- A) The lead assigns all tasks explicitly, preventing conflicts
- B) Teammates negotiate ownership through the mailbox
- C) File locking on the shared task list prevents race conditions ✅
- D) Tasks are queued in a sequential pipeline

---

## 12. Practical Exercise — Full Multi-Agent Workout

### Exercise A — Build a Custom Subagent

**Step 1:** Create a project-level subagent:
```bash
mkdir -p .claude/agents
```

Create `.claude/agents/security-reviewer.md`:
```markdown
---
name: security-reviewer
description: Security specialist. Use proactively after any changes to auth, payments, or data handling code.
tools: Read, Grep, Glob, Bash
model: sonnet
---

You are a senior security engineer. When invoked:
1. Run git diff to see recent changes
2. Focus on modified files

Review for:
- Injection vulnerabilities (SQL, command, template)
- Exposed secrets or API keys in code
- Authentication and authorisation gaps
- Input validation issues
- Insecure data storage or transmission

Report: Critical issues first, then warnings, then suggestions.
Include specific line references and suggested fixes.
```

**Step 2:** Test automatic delegation:
```
Make a small change to any auth-related file, then ask:
"Review my recent changes for security issues"
```
Watch whether Claude delegates to your subagent automatically.

**Step 3:** Test explicit invocation:
```
@"security-reviewer (agent)" review the changes in src/auth/
```

**Step 4:** Add persistent memory:
Add `memory: project` to the frontmatter and ask it to review code twice. Check `.claude/agent-memory/security-reviewer/` to see what it remembered.

**Step 5:** Test tool restriction:
Add `disallowedTools: Bash` and ask it to run a command. Verify it can't.

### Exercise B — Parallel Subagent Research

```
Research three parts of this codebase in parallel using separate subagents:
1. The authentication flow
2. The database layer
3. The API endpoints
Then synthesise findings into a single architectural summary.
```

Use `/context` before and after to see the context difference between doing this with and without subagents.

### Exercise C — Agent Team (if enabled)

```bash
# Enable in settings.json
echo '{"env": {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"}}' > .claude/settings.json
claude
```

```
Create an agent team to investigate a performance problem in this codebase.
Spawn three teammates: one investigating database queries, one looking at
API response times, one examining frontend rendering. Have them share findings
and identify the most likely bottleneck.
```

Try: `Shift+Down` to cycle through teammates and send one a direct message.

---

## Chapter 10 Summary

| Concept | Key Takeaway | CCA Domain |
|---|---|---|
| Two primitives | Subagents (within session) vs Agent Teams (independent sessions) | Domain 1 |
| Built-in subagents | Explore (Haiku, read-only), Plan (read-only), General-purpose (all tools) | Domain 1 |
| Description field | Critical — Claude uses this to decide when to delegate | Domain 1 |
| Tool control | `tools` = allowlist; `disallowedTools` = denylist; both = denylist first | Domain 1 |
| Context isolation | Subagent's expensive work stays in its window; only summary returns | Domain 5 |
| Background subagents | Permissions pre-approved upfront; auto-deny anything not pre-approved | Domain 1 |
| Persistent memory | `memory: user/project/local` — same 200-line/25KB limit as auto memory | Domain 5 |
| Agent teams | Experimental; teammates communicate directly; shared task list | Domain 1 |
| Task claiming | File locking prevents race conditions when multiple teammates claim same task | Domain 1 |
| When NOT to use | Sequential tasks, same-file edits, high-dependency work → single session | Domain 1 |

### CCA Domain Coverage

| CCA Domain | Concepts Covered |
|---|---|
| **Domain 1: Agentic Architecture & Orchestration (27%)** | Subagent design, tool boundaries, delegation patterns, agent teams, task decomposition, hub-and-spoke |
| **Domain 5: Context Management & Reliability (15%)** | Context isolation via subagents, persistent memory limits |

---

## ✅ Before Chapter 11

- [ ] Created a custom project subagent in `.claude/agents/`
- [ ] Tested automatic vs explicit delegation
- [ ] Understood the difference between `tools` (allowlist) and `disallowedTools` (denylist)
- [ ] Used a subagent for parallel research and compared context usage
- [ ] Understood why subagents can't spawn other subagents
- [ ] Can explain the subagents vs agent teams decision framework
- [ ] Answered all seven practice questions correctly

---

*Previous: [Chapter 9 — MCP: Model Context Protocol](./chapter-09-mcp-model-context-protocol.md)*
*Next: Chapter 11 — Checkpoints & Session Recovery (coming soon)*
