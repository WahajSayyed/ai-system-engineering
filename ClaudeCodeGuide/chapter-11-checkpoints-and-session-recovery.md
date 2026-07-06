# Chapter 11: Checkpoints & Session Recovery

> **CCA Exam Note:** This chapter covers Domain 3 (Claude Code Configuration & Workflows, 20%) and Domain 5 (Context Management & Reliability, 15%). The exam tests understanding of what checkpoints track, their limitations, the five rewind actions, session forking, and how these tools fit into a reliable agentic workflow.

---

## 1. The Safety Net Model

Every edit Claude makes carries risk. A refactor might break something subtle. An automated fix might touch more than you intended. A long agentic run might drift in an unexpected direction.

Claude Code addresses this with two complementary safety systems:

```
Checkpoints          → undo file changes (session-level)
Session persistence  → resume, fork, and navigate conversations
```

Together they give you the ability to:
- Roll back file changes to any prior state in the session
- Compress verbose history while keeping key instructions
- Branch a session to explore an alternative approach
- Resume exactly where you left off after closing the terminal

Understanding these systems precisely — what they cover, and crucially what they do **not** cover — is directly tested on the CCA exam.

---

## 2. How Checkpoints Work

### What Gets Tracked

Claude Code tracks all changes made by its **file editing tools**:

- Every user prompt creates a new checkpoint automatically
- The snapshot is taken **before** the edit, so you can always revert
- Checkpoints persist across sessions — you can rewind in a resumed conversation
- Automatically cleaned up after 30 days (configurable via `cleanupPeriodDays`)

### What Does NOT Get Tracked

This is exam-critical. Checkpoints track **only file edits made through Claude's editing tools**.

They do **not** track:

| What | Why |
|---|---|
| Files modified by bash commands | `rm`, `mv`, `cp`, shell scripts — these bypass the file editing tools |
| Your own manual edits outside Claude Code | Not in the session's tracking scope |
| Edits from other concurrent sessions | Only the current session's tool calls are tracked |
| External side effects | Database writes, API calls, deployments — cannot be undone |

```bash
# These are NOT tracked by checkpoints:
rm important.py
mv old_name.py new_name.py
cp template.py new_module.py
echo "something" >> config.py   # shell redirect, not Claude's Edit tool

# These ARE tracked:
# Any file Claude modifies via its Read/Write/Edit tools
```

> ⚠️ **This is the most commonly tested limitation.** If Claude runs a bash command that modifies files, those changes cannot be rewound. This is why Claude asks for confirmation before running commands with side effects.

---

## 3. The `/rewind` Menu — Five Actions

Press `Esc + Esc` or run `/rewind` to open a scrollable list of your session's prompts. Select any point, then choose one of five actions:

### Action 1: Restore code and conversation
Reverts **both** file changes and conversation history to the selected point. The selected prompt is restored into your input field so you can re-send or edit it before trying again.

**Use when:** Claude went down the wrong path entirely — both the approach and the files need to go back.

### Action 2: Restore conversation only
Rewinds to that message while **keeping the current code**. Conversation history rolls back; files on disk stay as they are now.

**Use when:** You want to re-approach a task from a different angle but keep the code changes that are already good.

### Action 3: Restore code only
Reverts file changes while **keeping the conversation**. All the discussion, context, and decisions stay; only the file edits are undone.

**Use when:** Claude made a bad edit but the conversation context leading to it is still useful.

### Action 4: Summarise from here
Compresses messages **from the selected point forward** into an AI-generated summary. Messages **before** the selected point stay intact and unchanged. No files are modified.

**Use when:** The middle of a long session has become noisy (verbose debugging, failed attempts) but your initial instructions and recent work are both valuable. Keep early context, compress the messy middle.

This is different from `/compact`:
- `/compact` — summarises the **entire** conversation
- **Summarise from here** — keeps early context intact, compresses only from the selected point forward

You can pass optional instructions to guide what the summary focuses on.

### Action 5: Never mind
Returns to the message list without making any changes.

---

## 4. `Esc + Esc` vs `/rewind` — Clarified

These both open the same rewind menu. The difference is input method only:

- `Esc + Esc` — keyboard shortcut, faster during active work
- `/rewind` — slash command, same result

Both open the scrollable prompt list and present the same five actions. There is no functional difference.

> 🔑 **Correction from earlier chapters:** We previously described `Esc × 2` as an "instant undo of the last action." The official docs clarify it opens the full rewind menu — it is not a one-step instant undo. The menu lets you choose scope and action.

---

## 5. Session Forking — Branching Your Approach

Checkpoints handle undoing within a session. Forking handles exploring **alternative approaches** without losing the original.

```bash
# Resume the most recent session
claude --continue

# Resume AND create a branch — original session preserved
claude --continue --fork-session
```

Forking creates a new session ID while preserving the full conversation history up to the fork point. The original session is completely untouched. You now have two independent sessions from the same starting point — like git branches for conversations.

### When to Fork vs Rewind

| Scenario | Use |
|---|---|
| Went wrong, want to try again from the same point | `/rewind` — restore code and conversation |
| Want to explore two approaches simultaneously | `--fork-session` — two independent sessions |
| Messy context, want to compress and continue | `/rewind` → Summarise from here |
| Finished, want to start completely fresh | `/clear` — new session |

---

## 6. Session Resume — Picking Up Where You Left Off

Claude Code saves every conversation locally. Sessions are tied to your current working directory.

```bash
claude --continue    # Resume the most recent session in this directory
claude --resume      # Interactive picker: choose from recent sessions
```

When you resume:
- Full conversation history is restored
- File edits from the session are visible in the context
- **Session-scoped permissions are NOT restored** — you must re-approve those

### Naming Sessions for Easy Navigation

```
/rename oauth-migration
/rename debugging-memory-leak
/rename api-v2-design
```

Descriptive names make `claude --resume` useful on projects with multiple active workstreams.

### Sessions as Branches

Treat sessions like branches: different workstreams get separate, persistent contexts. You can maintain a `feature/auth` session and a `bugfix/payment` session independently, resuming each when needed.

### Session Storage

Transcripts stored at: `~/.claude/projects/{project}/{sessionId}/`

Subagent transcripts stored separately at: `~/.claude/projects/{project}/{sessionId}/subagents/agent-{agentId}.jsonl`

Cleaned up after `cleanupPeriodDays` (default: 30 days).

---

## 7. Task List — Tracking Progress Across Long Sessions

For complex multi-step work, Claude creates and maintains a task list:

```
Ctrl+T    — toggle task list display (shows up to 10 tasks)
```

Tasks have three states: pending, in progress, completed. They persist across context compactions — when you `/compact`, the task list survives.

### Sharing Task Lists Across Sessions

```bash
# Use a named task list, shared across sessions
CLAUDE_CODE_TASK_LIST_ID=my-feature claude
```

This stores the task list at `~/.claude/tasks/my-feature/`, accessible from any session that uses the same ID. Useful for multi-day work where you resume across multiple Claude sessions.

---

## 8. The Full Keyboard Reference for Session Control

From the official interactive mode documentation, these are the session management shortcuts:

| Shortcut | Action |
|---|---|
| `Esc + Esc` | Open rewind menu — restore code/conversation or summarise |
| `Ctrl+C` | Cancel current input or generation |
| `Ctrl+D` | Exit Claude Code session |
| `Ctrl+R` | Reverse search through command history |
| `Ctrl+B` | Background a running task |
| `Ctrl+T` | Toggle task list display |
| `Ctrl+L` | Redraw the screen |
| `Ctrl+O` | Toggle verbose output (shows detailed tool usage) |
| `Shift+Tab` | Cycle permission modes |
| `Ctrl+G` or `Ctrl+X Ctrl+E` | Open current prompt in your default text editor |

### The `!` Prefix — Direct Bash Mode

Type `!` at the start of a prompt to run a shell command directly without involving Claude:

```bash
! npm test
! git status
! ls -la
```

This adds the command and its output to the conversation context. Output can be backgrounded with `Ctrl+B`. Useful for quick shell operations without asking Claude to do them.

> ⚠️ Commands run with `!` are **not tracked by checkpoints**. They go through the shell directly, not through Claude's file editing tools.

---

## 9. Internal Mechanics — What Checkpoints Actually Store

```
User sends prompt
       ↓
Checkpoint created:
  - Current state of every file that Claude has edited in this session
  - Stored as snapshots in session transcript
  - Linked to the prompt that triggered them
       ↓
Claude edits files via Edit/Write tools
       ↓
Next user prompt → new checkpoint → repeat
```

### What the Rewind Restoration Does

When you select "Restore code" from the rewind menu:

```
1. Claude reads the checkpoint snapshot for the selected prompt
2. Writes the snapshot contents back to disk for each tracked file
3. Any files created after that point are left as-is
   (checkpoints track content, not file existence)
4. Conversation is NOT changed (for code-only restore)
```

### Checkpoints vs Git — The Right Mental Model

| | Checkpoints | Git |
|---|---|---|
| **Scope** | Session-level undo | Permanent project history |
| **Granularity** | Every prompt | Every commit |
| **Duration** | 30 days | Permanent |
| **Sharing** | Local only | Team-wide |
| **Covers bash changes** | ❌ No | ✅ If committed |
| **Best for** | Quick undo during a session | Code history and collaboration |

The official docs are explicit: **"Think of checkpoints as 'local undo' and Git as 'permanent history'."**

Checkpoints complement git — they do not replace it. For any significant work, commit regularly so that git captures what checkpoints cannot.

---

## 10. PR Review Status — A Bonus Interactive Mode Feature

When working on a branch with an open pull request, Claude Code displays a clickable PR link in the terminal footer:

```
PR #446  ← shown in footer, colour-coded by review state
```

Colour coding:
- 🟢 Green — approved
- 🟡 Yellow — pending review
- 🔴 Red — changes requested
- ⬜ Grey — draft
- 🟣 Purple — merged

`Cmd+click` (Mac) or `Ctrl+click` (Windows/Linux) opens the PR in your browser. Status updates automatically every 60 seconds. Requires `gh` CLI installed and authenticated.

---

## 11. Practical Workflow — Safe Agentic Runs

Combining everything in this chapter into a reliable pattern:

### For Any Significant Task

```
1. Name your session before starting
   /rename feature-x-implementation

2. Start with a plan (plan mode)
   /plan add OAuth2 to the auth module

3. Review Claude's plan, approve

4. Work proceeds — checkpoints created automatically

5. If Claude goes wrong:
   Esc + Esc → choose restore scope → re-approach

6. Commit at logical milestones
   "Commit the model changes before we move to the API layer"

7. At session end, verify
   /diff
   /security-review
```

### For Exploratory or Risky Work

```
# Start the session normally, do initial setup

# Before a risky operation, note where you are
# (checkpoint is automatic on every prompt)

# If you want to try TWO different approaches:
# Open second terminal:
claude --continue --fork-session
# Now you have two independent sessions from the same starting point
```

### Recovering from a Bad Agentic Run

```
Esc + Esc           → opens rewind menu
↑↓ to select        → find the last good prompt
Choose: "Restore code and conversation"
                    → both code and conversation go back

Then: reframe your prompt more precisely and try again
```

---

## 12. CCA Exam — Practice Questions

**Q1:** A developer asks Claude to refactor a module. Claude runs a bash command `mv utils.py helpers.py` as part of the refactor, then makes several more file edits. The developer wants to undo everything. What will happen when they use `/rewind` → Restore code?

- A) Everything reverts including the `mv` operation
- B) The file edits via Claude's tools revert, but the `mv` operation is not tracked by checkpoints and cannot be undone through rewind ✅
- C) Nothing reverts because a bash command was involved
- D) Only the bash command reverts

**Q2:** A developer is deep in a debugging session. The first 10 messages contain critical setup instructions; the last 20 messages are a messy back-and-forth of failed attempts. They want to keep the early instructions but compress the noisy middle. What is the correct tool?

- A) `/compact` — summarises the entire conversation
- B) `/rewind` → select message 10 → "Summarise from here" — keeps early messages intact, compresses from message 10 forward ✅
- C) `/clear` — starts fresh
- D) `--fork-session` — creates a branch

**Q3:** A developer wants to explore two different architectural approaches for the same feature without losing either attempt. What is the correct approach?

- A) Use `/rewind` to go back after each attempt
- B) Use two separate `/clear` sessions
- C) Use `claude --continue --fork-session` to create a second independent session from the same starting point ✅
- D) Use subagents for each approach

**Q4:** When a developer runs `claude --continue` to resume a session, what is NOT restored?

- A) Conversation history
- B) File edit context
- C) Session-scoped permissions — these must be re-approved ✅
- D) The session's checkpoint history

**Q5:** What is the key difference between `/compact` and the "Summarise from here" option in the rewind menu?

- A) `/compact` is faster; rewind summarisation is more thorough
- B) `/compact` summarises the entire conversation; "Summarise from here" keeps messages before the selected point intact and compresses only from that point forward ✅
- C) They are identical in behaviour
- D) `/compact` preserves CLAUDE.md; rewind summarisation does not

**Q6:** A developer wants to track their task list across multiple Claude Code sessions on the same feature over several days. What is the correct approach?

- A) Write task status to CLAUDE.md manually
- B) Use `/rename` on each session with the same name
- C) Set `CLAUDE_CODE_TASK_LIST_ID=feature-name` when starting each session ✅
- D) Task lists cannot persist across separate sessions

---

## 13. Practical Exercise — Session Recovery Workout

**Step 1 — Set up a test project:**
```bash
mkdir checkpoint-practice && cd checkpoint-practice && git init
echo "def greet(name):\n    return f'Hello {name}'" > greet.py
claude
```

**Step 2 — Create multiple checkpoints:**
```
Add a farewell function to greet.py
Add type hints to all functions
Add a docstring to the greet function
```

Each prompt creates a checkpoint automatically.

**Step 3 — Practice rewind:**
```
Esc + Esc
```
Select the first prompt ("Add a farewell function"). Try each restore option and observe the effect.

**Step 4 — Test the bash limitation:**
```
Run: mv greet.py greeting.py using a bash command
```
Then try to rewind — confirm the file rename is not undone.

**Step 5 — Practice "Summarise from here":**
After several back-and-forth messages, open the rewind menu, select an early message, and choose "Summarise from here". Verify early messages remain intact.

**Step 6 — Fork a session:**
```bash
# In a second terminal:
claude --continue --fork-session
```
Make a different change in each terminal. Confirm they are independent.

**Step 7 — Name and resume:**
```
/rename checkpoint-practice-session
```
Exit with `Ctrl+D`, then:
```bash
claude --resume
```
Find and resume the named session.

---

## Chapter 11 Summary

| Concept | Key Takeaway | CCA Domain |
|---|---|---|
| What checkpoints track | Only file edits via Claude's tools — NOT bash commands or external changes | Domain 3 |
| Five rewind actions | Restore code+conversation / code only / conversation only / Summarise from here / Never mind | Domain 3 |
| `Esc + Esc` = `/rewind` | Both open the same menu — no functional difference | Domain 3 |
| Summarise from here | Keeps early messages intact; compresses only from selected point forward | Domain 5 |
| `/compact` vs summarise | `/compact` = entire conversation; summarise = targeted partial compression | Domain 5 |
| Session fork | `--fork-session` creates independent session from same history; original untouched | Domain 3 |
| Session resume | `--continue` or `--resume`; session-scoped permissions NOT restored | Domain 3 |
| Task list persistence | `CLAUDE_CODE_TASK_LIST_ID` shares task list across sessions | Domain 3 |
| Checkpoints vs git | "Local undo" vs "permanent history" — complement each other | Domain 3 |
| Bash commands | NOT tracked — critical limitation for agentic workflows | Domain 5 |

---

## ✅ Before Chapter 12

- [ ] Practiced `/rewind` with all five action options
- [ ] Confirmed bash command changes are not tracked by checkpoints
- [ ] Used "Summarise from here" on a long session
- [ ] Forked a session with `--fork-session`
- [ ] Named a session and resumed it with `claude --resume`
- [ ] Understood the mental model: checkpoints = local undo, git = permanent history
- [ ] Answered all six practice questions correctly

---

*Previous: [Chapter 10 — Multi-Agent & Parallel Workflows](./chapter-10-multi-agent-and-parallel-workflows.md)*
*Next: Chapter 12 — CI/CD Automation (coming soon)*
