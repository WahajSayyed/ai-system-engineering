# Chapter 8: Context Window Management

> **CCA Exam Note:** This chapter covers Domain 5 (Context Management & Reliability, 15%). The official Anthropic documentation states directly: "The context window is the most important resource to manage." Domain 5 questions test your understanding of what fills context, how degradation manifests, and which strategies preserve reliable performance. This chapter is built directly from official Claude Code documentation at code.claude.com.

---

## 1. Why Context Is Your Primary Constraint

Most best practices are based on one constraint: Claude's context window fills up fast, and performance degrades as it fills.

This is not a minor technical detail. It is the foundational constraint that explains why every other best practice in this guide exists. Understanding it deeply changes how you work.

Claude's context window holds your entire conversation, including every message, every file Claude reads, and every command output. A single debugging session or codebase exploration might generate and consume tens of thousands of tokens. LLM performance degrades as context fills. When the context window is getting full, Claude may start "forgetting" earlier instructions or making more mistakes.

The mental model to hold throughout this chapter:

```
Context window = a finite bucket
Every file read   → fills the bucket
Every message     → fills the bucket
Every tool output → fills the bucket
Every command run → fills the bucket

When the bucket is full:
- Earlier contents get summarised or dropped
- Instructions set early in the session are forgotten
- Mistakes increase
- Precision degrades
```

---

## 2. What Loads Into Context — The Full Picture

Claude Code's context window holds everything Claude knows about your session: your instructions, the files it reads, its own responses, and content that never appears in your terminal.

Understanding exactly what fills context before you type a single character is essential.

### What Loads at Session Start (Before You Type Anything)

Before you type anything: CLAUDE.md, auto memory, MCP tool names, and skill descriptions all load into context. Your own setup may add more here, like an output style or text from `--append-system-prompt`, which both go into the system prompt the same way.

```
Session opens
       ↓
Automatically loaded into context:
  ├── CLAUDE.md contents (all levels: global, project, local, subdirectory)
  ├── Auto memory entries (MEMORY.md — first 200 lines or 25KB)
  ├── MCP tool names and descriptions
  ├── Skill descriptions (not skill content — just descriptions)
  └── Output style / --append-system-prompt content (if configured)
```

> 🔑 **Key insight from the docs:** The first 200 lines or 25KB of MEMORY.md, whichever comes first, load at the start of each session. This is a hard limit — auto memory beyond this threshold does not load.

### What Loads As Claude Works

As Claude works: each file read adds to context, path-scoped rules load automatically alongside matching files, and hooks fire after each edit.

```
As session progresses:
  ├── Every file Claude reads → full contents added
  ├── Path-scoped CLAUDE.md rules → load when matching files are touched
  ├── Hook output → added after each tool use
  ├── Command stdout/stderr → added after each bash execution
  ├── Your prompts → added
  └── Claude's responses → added
```

### The Token Cost Breakdown (Representative Numbers)

```
Session start (typical setup):
  CLAUDE.md (concise)         ~500  tokens
  Auto memory                 ~300  tokens
  MCP tool names              ~200  tokens
  ─────────────────────────────────────────
  Baseline cost:            ~1,000  tokens

Each file read (500-line Python file):
  ~3,000 tokens per file

Each round trip (your prompt + Claude response):
  ~500–2,000 tokens depending on length

After 10 file reads + 20 messages:
  Baseline:                  1,000
  File reads (10 × 3,000):  30,000
  Messages (20 × 1,000):    20,000
  ─────────────────────────────────
  Total:                    ~51,000 tokens
  (of a 200,000 token window = 25% used)

After a long debugging session (50+ files, 60+ messages):
  Can easily reach 150,000–180,000+ tokens
```

---

## 3. The `/context` Command — Live Context Monitoring

To see your actual context usage at any point, run `/context` for a live breakdown by category with optimization suggestions.

```
/context
```

This is your primary tool for monitoring context health during a session. It shows:
- Usage broken down by category (system prompt, conversation, files, etc.)
- A visual grid showing how full the window is
- Optimisation suggestions when specific categories are bloated

Run `/context` when:
- You notice Claude getting less precise
- Before starting a large task to check available headroom
- When a session has been running for a long time
- After a large file exploration to see the impact

---

## 4. Managing Context — The Official Strategies

These strategies come directly from the official best practices documentation.

### Strategy 1: `/clear` Between Unrelated Tasks

Use `/clear` frequently between tasks to reset the context window entirely.

This is the most aggressive strategy — it throws away everything and starts fresh. Use it when:
- You've finished one task and are starting a completely different one
- The session has accumulated noise from failed attempts
- You've corrected Claude on the same issue more than twice

If you've corrected Claude more than twice on the same issue in one session, the context is cluttered with failed approaches. Run `/clear` and start fresh with a more specific prompt that incorporates what you learned. A clean session with a better prompt almost always outperforms a long session with accumulated corrections.

### Strategy 2: `/compact` With Focus Instructions

When auto compaction triggers, Claude summarises what matters most, including code patterns, file states, and key decisions. For more control, run `/compact <instructions>`, like `/compact Focus on the API changes`.

Unlike `/clear`, `/compact` preserves the important parts of the session. The optional instructions let you tell Claude what to prioritise in the summary:

```bash
/compact Focus on the auth changes and the failing tests
/compact Keep the full list of modified files and the agreed architecture
/compact Preserve the database schema decisions
```

Customise compaction behaviour in CLAUDE.md with instructions like "When compacting, always preserve the full list of modified files and any test commands" to ensure critical context survives summarisation.

> 🔑 **Official doc insight:** CLAUDE.md fully survives compaction. After `/compact`, Claude re-reads your CLAUDE.md from disk and re-injects it fresh into the session. If an instruction disappeared after compaction, it was given only in conversation, not written to CLAUDE.md. Add it to CLAUDE.md to make it persist.

### Strategy 3: Partial Rewind for Targeted Compaction

To compact only part of the conversation, use `Esc + Esc` or `/rewind`, select a message checkpoint, and choose **Summarise from here**. This condenses messages from that point forward while keeping earlier context intact.

This is the surgical option — you keep the early part of the session exactly as it is, and only compress the recent section that has grown noisy.

### Strategy 4: `/btw` for Side Questions

For quick questions that don't need to stay in context, use `/btw`. The answer appears in a dismissible overlay and never enters conversation history, so you can check a detail without growing context.

```
/btw what's the syntax for a Python dataclass with default values?
```

The answer pops up as an overlay. You read it and dismiss it. Zero context cost.

Use `/btw` when:
- You need a quick factual lookup
- You want to check a syntax detail
- You have a question unrelated to the current task

### Strategy 5: Subagents for Exploration

Since context is your fundamental constraint, subagents are one of the most powerful tools available. When Claude researches a codebase it reads lots of files, all of which consume your context. Subagents run in separate context windows and report back summaries.

```
Use subagents to investigate how our authentication system handles
token refresh, and whether we have any existing OAuth utilities
I should reuse.
```

A subagent handles the research in its own separate context window, so the large file reads stay out of yours. Only the summary and a small metadata trailer come back.

This is the most powerful context strategy for large codebase exploration — the file reads happen in a separate window and you only receive the summary.

---

## 5. Auto-Compaction — What Happens Automatically

Claude Code automatically compacts conversation history when you approach context limits, which preserves important code and decisions while freeing space.

When auto-compaction triggers:
1. Claude creates a structured summary of the conversation
2. The full history is replaced with that summary
3. CLAUDE.md is re-read fresh from disk and re-injected
4. The session continues with recovered context headroom

### What Survives Auto-Compaction

| Content | Survives? |
|---|---|
| CLAUDE.md contents | ✅ Always — re-read from disk in full |
| Auto memory (MEMORY.md) | ✅ Re-loaded (first 200 lines/25KB) |
| Key code decisions | ✅ Preserved in summary |
| Modified file list | ✅ If instructed in CLAUDE.md |
| Exact conversation history | ❌ Replaced by summary |
| Skill descriptions | ❌ Not re-loaded after compaction |
| Instructions given only in chat | ❌ May be lost — put them in CLAUDE.md |

> ⚠️ If an instruction disappeared after compaction, it was given only in conversation, not written to CLAUDE.md. This is the most common source of confusion after compaction.

---

## 6. Auto Memory — Context That Persists Across Sessions

Auto memory lets Claude accumulate knowledge across sessions without you writing anything. Claude saves notes for itself as it works: build commands, debugging insights, architecture notes, code style preferences, and workflow habits. Claude does not save something every session — it decides what is worth remembering based on whether the information would be useful in a future conversation.

### How to Manage Auto Memory

```
/memory
```

Run `/memory` and select the auto memory folder to browse what Claude has saved. Everything is plain markdown you can read, edit, or delete.

Use `/memory` to:
- See what Claude has learned about your project across sessions
- Delete entries that are outdated or incorrect
- Toggle auto memory on or off
- View and edit CLAUDE.md files

### The Auto Memory Size Limit

The 200-line / 25KB limit applies **only to `MEMORY.md`** (auto memory), not to CLAUDE.md. From the official docs:

> "This limit applies only to `MEMORY.md`. CLAUDE.md files are loaded in full regardless of length, though shorter files produce better adherence."

So the distinction is:
- **CLAUDE.md** → always loaded in full, no line limit. But longer files reduce adherence quality.
- **MEMORY.md** → hard limit: first 200 lines or 25KB, whichever comes first. Content beyond this is not loaded at session start.

For CLAUDE.md: target under 200 lines not because of a hard limit, but because longer files cause Claude to miss or deprioritise rules. For MEMORY.md: the 200-line limit is a technical hard cap.

### Auto Memory vs CLAUDE.md

| | CLAUDE.md | Auto Memory (MEMORY.md) |
|---|---|---|
| Who writes it | You | Claude (automatically) |
| Content type | Rules, conventions, architecture | Observations, patterns, learnings |
| Control | Full manual control | Claude decides what to save |
| Load amount | Always loaded **in full** | First 200 lines or 25KB only |
| Topic files | n/a | Created on demand, NOT loaded at startup |
| Best for | Team standards, fixed rules | Project-specific learnings |

---

## 7. The Context Window Visualisation

The official docs provide an interactive context window visualiser at `code.claude.com/docs/en/context-window`. The session walks through a realistic flow with representative token counts showing what loads at startup and what each file read costs.

Even without the interactive version, understanding the timeline is valuable:

```
T=0  Session opens
     CLAUDE.md loaded (~500 tokens)
     Auto memory loaded (~300 tokens)
     MCP tool names loaded (~200 tokens)
     ─────────────────────────────
     Baseline: ~1,000 tokens used

T=1  You ask: "Explain the auth system"
     Claude reads: src/auth/login.py     (+3,000)
     Claude reads: src/auth/tokens.py    (+2,500)
     Claude reads: src/auth/middleware.py(+2,000)
     Claude response                     (+1,000)
     ─────────────────────────────
     Total: ~9,500 tokens used

T=2  Path-scoped rule fires (src/auth/ CLAUDE.md loads)
                                         (+400)

T=3  You ask: "Add rate limiting"
     Claude reads: src/core/config.py    (+1,500)
     Claude edits: src/auth/middleware.py(+500 diff)
     PostToolUse hook fires              (+200)
     ─────────────────────────────
     Running total: ~12,100 tokens

... [session continues, accumulating rapidly] ...
```

To see how a session fills up in practice, watch an interactive walkthrough of what loads at startup and what each file read costs. Track context usage continuously with a custom status line.

---

## 8. Reducing Token Usage — Official Strategies

From the official documentation, specific techniques for keeping context lean:

### Write Lean CLAUDE.md Files

Keep it concise. For each line, ask: "Would removing this cause Claude to make mistakes?" If not, cut it. Bloated CLAUDE.md files cause Claude to ignore your actual instructions.

The official guidance on what to include vs exclude:

| ✅ Include | ❌ Exclude |
|---|---|
| Bash commands Claude can't guess | Anything Claude can figure out by reading code |
| Code style rules that differ from defaults | Standard language conventions Claude already knows |
| Testing instructions and preferred test runners | Detailed API documentation (link to docs instead) |
| Repository etiquette (branch naming, PR conventions) | Information that changes frequently |
| Architectural decisions specific to your project | Long explanations or tutorials |
| Developer environment quirks (required env vars) | File-by-file descriptions of the codebase |
| Common gotchas or non-obvious behaviours | Self-evident practices like "write clean code" |

### Use Skills Instead of CLAUDE.md for Domain Knowledge

CLAUDE.md is loaded every session, so only include things that apply broadly. For domain knowledge or workflows that are only relevant sometimes, use skills instead. Claude loads them on demand without bloating every conversation.

This is a critical architectural decision:
- **CLAUDE.md** → always loaded, always consumes context
- **Skills** → loaded on demand, only when the task needs them

### Use `@` Mentions Selectively

As covered in Chapter 3 and 4, `@` file mentions in prompts are efficient — they load exactly the file needed. But `@` mentions **inside CLAUDE.md** load on every request. Use CLAUDE.md `@` references only for small, universally-relevant files.

### Scope Investigations with Subagents

You can also use subagents for verification after Claude implements something: "use a subagent to review this code for edge cases." The subagent explores, reads relevant files, and reports back with findings — all without cluttering your main conversation.

---

## 9. Common Context Failure Patterns

The official docs name these explicitly. Recognising them early saves time:

### The Kitchen Sink Session
You start with one task, then ask Claude something unrelated, then go back to the first task. Context is full of irrelevant information. **Fix**: `/clear` between unrelated tasks.

### Correcting Over and Over
Claude does something wrong, you correct it, it is still wrong, you correct again. Context is polluted with failed approaches. **Fix**: After two failed corrections, `/clear` and write a better initial prompt incorporating what you learned.

### The Over-Specified CLAUDE.md
If your CLAUDE.md is too long, Claude ignores half of it because important rules get lost in the noise. **Fix**: Ruthlessly prune. If Claude already does something correctly without the instruction, delete it or convert it to a hook.

### The Infinite Exploration
You ask Claude to "investigate" something without scoping it. Claude reads hundreds of files, filling the context. **Fix**: Scope investigations narrowly or use subagents so the exploration doesn't consume your main context.

---

## 10. Session Management — Resume and Rename

Claude Code saves conversations locally. When a task spans multiple sessions, you don't have to re-explain the context:

```bash
claude --continue   # Resume the most recent conversation
claude --resume     # Select from recent conversations
```

Use `/rename` to give sessions descriptive names like "oauth-migration" or "debugging-memory-leak" so you can find them later. Treat sessions like branches: different workstreams can have separate, persistent contexts.

### The Session-as-Branch Mental Model

This is a powerful way to think about long tasks:

```
main-session          → daily development work
oauth-migration       → dedicated session for the OAuth refactor
debug-memory-leak     → dedicated session for the performance bug
api-v2-design         → dedicated session for architecture discussion
```

Each session maintains its own context, conversation history, and focus. You can switch between them without losing state — just like switching git branches.

---

## 11. Internal Mechanics — How Compaction Works

Understanding what happens under the hood during `/compact`:

```
/compact triggered (manually or automatically)
       ↓
Claude reads entire conversation history
       ↓
Generates structured summary preserving:
  - Key decisions made
  - Files modified and their current state
  - Architectural choices
  - Test results
  - Any explicit preservation instructions from CLAUDE.md
       ↓
Conversation history REPLACED with summary
       ↓
CLAUDE.md re-read from disk → re-injected fresh
Auto memory re-checked
       ↓
Session continues with reduced context usage
and fresh CLAUDE.md injection
```

### Why CLAUDE.md Surviving Compaction Matters

This is the key architectural insight: your CLAUDE.md instructions are **not** part of the conversation history that gets compacted. They live on disk and are re-injected fresh after every compaction. This is why:

1. Instructions in CLAUDE.md are reliable across long sessions
2. Instructions given only in chat are fragile — they can be lost in compaction
3. Any rule that must persist should be in CLAUDE.md, not just typed in conversation

---

## 12. The Effort Level and Context

A new command covered in Chapter 2's slash command table but worth understanding in context terms:

```
/effort [low|medium|high|max|auto]
```

The effort level controls thinking depth. `low`, `medium`, and `high` persist across sessions. `max` applies to the current session only and requires Opus 4.6.

Higher effort = more tokens used per response = faster context fill. For long exploration sessions, consider using `low` or `medium` effort to extend session length. Switch to `high` or `max` for the critical implementation steps.

---

## 13. CCA Exam — Practice Questions

**Q1:** A developer's Claude Code session has been running for 2 hours. Claude starts ignoring a constraint that was set in the first 10 messages. What is the most likely cause and correct fix?
- A) Claude has a bug — restart the CLI
- B) The constraint was given only in conversation and has been lost due to context filling — add it to CLAUDE.md and use `/compact` ✅
- C) Use `/clear` to reset context immediately
- D) Switch to a more powerful model

**Q2:** A developer wants to investigate an unfamiliar 200-file codebase without filling their main context window. What is the officially recommended approach?
- A) Use `@` to mention all relevant files
- B) Ask Claude to read files one at a time
- C) Delegate the investigation to a subagent, which runs in its own separate context window ✅
- D) Use `/compact` before and after the investigation

**Q3:** After running `/compact`, a developer notices that a coding style instruction they gave 30 messages ago has been lost. What is the root cause?
- A) `/compact` has a bug that deletes instructions
- B) The instruction was given only in conversation and was not preserved in CLAUDE.md — CLAUDE.md content always survives compaction because it is re-read from disk ✅
- C) The CLAUDE.md file was too large (note: CLAUDE.md has no hard line limit — only MEMORY.md does)
- D) The auto memory limit was exceeded

> 🔑 Note: The 200-line limit applies to MEMORY.md (auto memory), NOT CLAUDE.md. CLAUDE.md is always loaded in full. If instructions were in CLAUDE.md they would have survived compaction.

**Q4:** A developer wants to ask a quick syntax question without adding it to the conversation history. Which command should they use?
- A) `/context`
- B) `/compact`
- C) `/btw` ✅
- D) A new `/clear` session

**Q5:** A developer's CLAUDE.md is 800 lines long and Claude is frequently ignoring instructions in it. What is the official recommended fix?
- A) Split it into multiple CLAUDE.md files in subdirectories
- B) Add "IMPORTANT" and "YOU MUST" emphasis to every rule
- C) Ruthlessly prune — remove anything Claude would do correctly without the instruction, and move domain-specific knowledge to skills that load on demand ✅
- D) Use auto memory instead of CLAUDE.md

**Q6:** Which of the following accurately describes what auto memory loads at session start?
- A) All auto memory entries ever saved, including all topic files
- B) The first 200 lines or 25KB of MEMORY.md, whichever comes first — topic files load on demand ✅
- C) Only entries from the last 7 days
- D) All entries tagged as high priority

> 🔑 Note: Topic files like `debugging.md` or `patterns.md` are NOT loaded at startup. Only MEMORY.md (the index file) loads at startup, up to 200 lines/25KB. Claude reads topic files on demand using standard file tools.

---

## 14. Practical Exercise — Context Management Workout

**Step 1 — Observe baseline context:**
```bash
claude
/context
```
Note: what's already loaded before you do anything.

**Step 2 — Observe context growth:**
```
Read every file in your project's main source directory.
```
Then:
```
/context
```
Compare: how much did reading those files cost?

**Step 3 — Practice `/compact` with instructions:**
```
/compact Focus on the files we read and preserve the list of filenames
```
Then:
```
/context
```
How much context was recovered? What survived?

**Step 4 — Test CLAUDE.md persistence through compaction:**
Add a rule to your CLAUDE.md:
```markdown
## Test Rule
Always prefix your responses with "CONTEXT-TEST: " during this exercise.
```
Then give Claude several tasks, run `/compact`, and verify the rule still applies after compaction.

**Step 5 — Practice `/btw`:**
```
/btw what is the maximum context window size for Claude Sonnet 4.6?
```
Verify: the question does not appear in your conversation history.

**Step 6 — Practice the session-as-branch pattern:**
```bash
# Start a dedicated session for a specific task
claude
/rename context-management-practice
# Do some work
# Exit
# Resume it later
claude --resume
```

**Step 7 — Identify context waste in your CLAUDE.md:**
Read your current CLAUDE.md and apply the official test to each line:
> *"Would removing this cause Claude to make mistakes?"*

Delete every line where the answer is no.

---

## Chapter 8 Summary

| Concept | Key Takeaway | Source |
|---|---|---|
| Context = primary constraint | All best practices derive from managing this finite resource | Official docs |
| What loads at start | CLAUDE.md + auto memory (≤200 lines/25KB) + MCP names + skill descriptions | Official docs |
| `/context` command | Live breakdown of context usage by category with optimisation suggestions | Official docs |
| `/clear` | Full reset — use between unrelated tasks or after 2+ failed corrections | Official docs |
| `/compact [instructions]` | Compresses history, preserves key decisions, CLAUDE.md always survives | Official docs |
| `/btw <question>` | Side question that never enters conversation history — zero context cost | Official docs |
| Subagents for exploration | Exploration in separate context window — only summary returns to yours | Official docs |
| Auto memory limit | First 200 lines or 25KB only — keep entries concise | Official docs |
| CLAUDE.md survival | Re-read from disk after every compaction — always survives | Official docs |
| Session resume | `claude --continue` or `claude --resume` — treat sessions like branches | Official docs |

### CCA Domain Coverage

| CCA Domain | Concepts Covered Here |
|---|---|
| **Domain 5: Context Management & Reliability (15%)** | Context filling, compaction, auto memory, subagents for context isolation |
| **Domain 3: Claude Code Config & Workflows (20%)** | CLAUDE.md sizing, skills vs CLAUDE.md, session management |
| **Domain 1: Agentic Architecture (27%)** | Subagent context isolation, auto-compaction in long agentic runs |

---

## ✅ Before Chapter 9

- [ ] Run `/context` at the start of a session and after reading several files — compare
- [ ] Used `/compact` with focus instructions on a real session
- [ ] Verified CLAUDE.md content survives compaction
- [ ] Used `/btw` for a side question and confirmed it doesn't appear in history
- [ ] Audited your CLAUDE.md — removed anything Claude already does without being told
- [ ] Renamed a session with `/rename` and resumed it with `claude --resume`
- [ ] Answered all six practice questions correctly

---

*Previous: [Chapter 7 — Git Integration](./chapter-07-git-integration.md)*
*Next: Chapter 9 — MCP: Model Context Protocol (coming soon)*
