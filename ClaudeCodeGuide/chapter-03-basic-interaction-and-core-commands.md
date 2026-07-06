# Chapter 3: Basic Interaction & Core Commands

---

## 1. The Fundamental Mindset Shift

Before learning *how* to talk to Claude Code, internalize one idea:

**You are not prompting an AI. You are directing a developer.**

When you onboard a new developer, you don't say:
> *"Please utilize your programming expertise to potentially implement a solution for the authentication challenge we are experiencing."*

You say:
> *"The login endpoint is broken. It's returning 401 even with valid credentials. Fix it."*

Claude Code responds to the same directness. Vague, polite, over-qualified prompts produce vague results. Direct, specific, task-framed prompts produce precise action.

### Context Management Is Your Responsibility

When working with Claude on coding projects, context management is crucial. Your project might have dozens or hundreds of files, but Claude only needs the **right** information to help you effectively.

> ⚠️ **Too much irrelevant context actually decreases Claude's performance.** Loading every file in a large project into context is worse than loading only the relevant ones.

Your job as the developer is to guide Claude toward the files and information it needs for each specific task. The tools for doing this — `@` file mentions, CLAUDE.md references, and `/add-dir` — are covered in this chapter and Chapter 4. Learning to use them well is one of the highest-leverage skills in Claude Code.

---

## 2. The Four Interaction Modes

### Mode 1: Question / Exploration
You want to *understand* something. Claude reads and explains, makes no changes.

```
What does the PaymentProcessor class do?
How does authentication flow through this app?
Why is this function O(n²)?
What dependencies does this project have and are any outdated?
```

### Mode 2: Task Execution
You want Claude to *do* something. It plans, acts, and verifies.

```
Add rate limiting to the /api/login endpoint
Write tests for the UserService class
Refactor this function to use async/await
Fix the failing test in test_payments.py
```

### Mode 3: Iterative Collaboration
You work back and forth, refining. Each message builds on the last.

```
You:     Add a caching layer to the database queries
Claude:  [adds Redis caching]
You:     Good, but the TTL should be configurable via environment variable
Claude:  [updates implementation]
You:     Also add a cache invalidation method
Claude:  [adds the method]
```

### Mode 4: One-Shot CLI
No interactive session. One command, one result, done.

```bash
claude "add docstrings to every function in utils.py"
claude "what is the database schema for this project?"
claude --print "summarize main.py" > summary.txt
```

---

## 3. Prompt Structure — What Makes a Great Prompt

Every strong Claude Code prompt has up to four components:

```
[CONTEXT] + [TASK] + [CONSTRAINTS] + [OUTPUT FORMAT]
```

### Component 1: Context
What Claude needs to know that it can't infer:
```
The app uses soft deletes — records are never truly deleted,
just marked with deleted_at timestamp.
```

### Component 2: Task
What you actually want done — verb-first, specific:
```
Add a scope to the User model that filters out soft-deleted records
by default on all queries.
```

### Component 3: Constraints
Boundaries it must respect:
```
Don't modify the existing migration files.
Keep backward compatibility with the existing API.
Use the same pattern as the Product model's soft delete implementation.
```

### Component 4: Output Format
How you want the result:
```
Show me the diff before applying.
Add comments explaining the changes.
Also update the tests.
```

**Full example combined:**
```
The app uses soft deletes (deleted_at timestamp). Add a default scope
to the User model that filters out soft-deleted records on all queries.
Don't touch existing migrations. Follow the same pattern as the Product
model. Show the diff before applying and update the relevant tests.
```

One prompt. Everything Claude needs. Right first time.

---

## 4. Prompt Patterns — Your Repeatable Toolkit

### The "Explain First" Pattern
Forces Claude to demonstrate understanding before acting. Prevents confident wrong action.
```
Before making any changes, explain how the current authentication
system works and what you plan to do. Then wait for my confirmation.
```

### The "Follow Existing Patterns" Pattern
Critical for consistency in real codebases:
```
Add input validation to the registration endpoint.
Follow the exact same pattern used in the login endpoint.
```

Without this, Claude invents its own approach which may not match your codebase conventions.

### The "Scope Limiter" Pattern
Prevents Claude from helpfully doing too much:
```
Only change user_service.py. Don't touch any other files.
```

### The "Verification First" Pattern
Ask Claude to find the problem before fixing it:
```
Find all the places in the codebase where we're not handling
database connection errors. List them. Don't fix anything yet.
```

### The "Step by Step" Pattern
For complex multi-part tasks, break them into explicit stages:
```
Do this in three steps:
1. First, add the database migration
2. Then update the model
3. Finally update the API endpoint

Complete each step and show me before moving to the next.
```

### The "Reference Implementation" Pattern
Point Claude at something that already works:
```
Write a new EmailNotificationService using the exact same
structure as the existing SMSNotificationService in
src/notifications/sms.py
```

### The "Rubber Duck" Pattern
Use Claude to think through a problem without writing code yet:
```
I need to add real-time notifications to this app. Talk me through
the different approaches I could take given this tech stack.
Don't write any code yet.
```

---

## 5. File Mentions with `@` — Pinpoint Context

One of the most powerful and underused features in Claude Code. Instead of letting Claude search for relevant files, you tell it exactly which files to look at using the `@` symbol.

### Basic Usage
```
How does the auth system work? @src/auth/login.py
```

Claude immediately reads that exact file and includes it in context before answering. No searching, no guessing.

### Fuzzy File Selection
You don't need the full path. Type `@auth` and Claude shows you a list of matching files to choose from:
```
How does the auth system work? @auth

  Matching files:
  > src/auth/login.py
    src/auth/tokens.py
    src/auth/middleware.py
    tests/test_auth.py

  [Select with arrow keys, Enter to confirm]
```

This is faster than typing full paths and helps when you know roughly which module is relevant but not the exact file.

### Multiple Files in One Prompt
```
Why is there a mismatch between @src/schemas/user.py
and @src/models/user.py?
```

Both files are loaded into context and Claude can directly compare them.

### When to Use `@` vs Letting Claude Search

| Situation | Best Approach |
|---|---|
| You know exactly which file is relevant | `@` mention — faster and more token-efficient |
| You're not sure which file has the logic | Let Claude search — it uses grep internally |
| Comparing two specific files | `@` both explicitly |
| Asking about a feature that spans many files | Let Claude explore, then `@` specific ones for follow-up |

> 💡 **Context efficiency tip:** Every file you `@` mention is fully loaded into the context window. For large files, be selective — load only what's needed for the current question.

---

## 6. The `#` Command — Memory Mode

The `#` command puts Claude into **memory mode**, which lets you update your `CLAUDE.md` file directly from the chat prompt without manually editing the file.

### Usage
```
# Use comments sparingly. Only comment complex code.
```

Claude takes this instruction and **intelligently merges** it into your `CLAUDE.md` file — it doesn't just append, it understands where the instruction belongs and integrates it properly.

### More Examples
```
# Always use async/await for database calls
# Never use print() — use the logger from src/core/logging.py
# Run pytest after every change
```

### Why This Matters
Without `#` mode, you'd need to:
1. Stop your session
2. Open CLAUDE.md in an editor
3. Find the right section
4. Add your instruction
5. Return to Claude

With `#` mode, you update your project memory **without breaking your flow**. As you notice things during a session ("Claude keeps adding verbose comments"), you fix them instantly.

> 🔑 **Best practice:** When you find yourself correcting Claude for the same thing twice, use `#` to encode that correction into CLAUDE.md so it never happens again.

---

## 7. Reading Claude's Output — What to Watch For

### Tool Use Indicators
```
● Reading: src/auth/login.py          ← reading a file
● Searching: "def validate_token"     ← grepping the codebase
● Running: python -m pytest tests/    ← executing a command
● Writing: src/auth/login.py          ← making an edit
```

Watch these carefully. If Claude is reading files you didn't expect, it's either exploring smartly or going off course. If it's running commands you didn't authorize, press `Escape`.

### The Thinking Block
Sometimes Claude shows its reasoning before acting:
```
I need to understand how tokens are currently validated before
adding rate limiting. Let me read the auth module first...
```
This is Claude planning. A good sign — it means it's not blindly executing.

### Diff Display
Before applying any file edit:
```
  Editing: src/auth/login.py
  ┌──────────────────────────────────────┐
  │   def login(username, password):     │
  │ -     return authenticate(username,  │
  │ -                          password) │
  │ +     user = authenticate(username,  │
  │ +                          password) │
  │ +     if not user:                   │
  │ +         raise AuthError("Invalid") │
  │ +     return user                    │
  └──────────────────────────────────────┘
  Apply this change? [y/n/e]
```

`y` = apply, `n` = skip, `e` = open in editor to modify manually before applying.

> ⚠️ **Always read diffs.** This is your primary quality gate.

---

## 6. Slash Commands — Deep Practical Usage

### `/clear` — When to Use It
Clears conversation history. Files are untouched.

**Use when:**
- Finished one task, starting a completely unrelated one
- Conversation has gone in the wrong direction
- Claude seems confused by accumulated context

**Don't** use it mid-task — you'll lose the context Claude needs to complete what it started.

### `/compact` — The Context Saver
Compresses conversation history into a summary, freeing up token space while keeping important context.

**Use when:**
- Sessions run long
- Claude's responses are getting slower or less precise
- You want to continue a long session without starting over

Unlike `/clear`, `/compact` **keeps the project understanding** — it just shrinks how it's represented.

### `/memory` — Inspect What Claude Knows
```
/memory

Active context:
- Project: FastAPI invoicing backend
- CLAUDE.md: loaded (247 tokens)
- Files read this session: 8
- Conversation turns: 14
- Estimated tokens used: 18,400 / 200,000
```

Use to understand why Claude might be missing context, or to check how close you are to the context limit.

### `/tasks` — Background Task Tracking
Lists and manages background tasks Claude is running or has queued:
```
/tasks
```
Use this to monitor long-running operations and check what Claude has in progress.

### `/diff` — See All Changes This Session
Shows an interactive diff viewer of every uncommitted change Claude has made — per-turn and combined. Use left/right arrows to switch views, up/down to browse files.

### `/insights` — Session Analytics
```
/insights

Session statistics:
- Duration: 34 minutes
- Prompts sent: 12
- Tools called: 47
- Files read: 15
- Files modified: 4
- Tokens used: 42,800
- Estimated cost: $0.18
```



---

## 8. Full Slash Command Reference

### Session Management
| Command | What it does |
|---|---|
| `/help` | Lists all slash commands |
| `/exit` | Ends session. Alias: `/quit` |
| `/clear` | Clears history, keeps files. Aliases: `/reset`, `/new` |
| `/compact [instructions]` | Summarises conversation to free context space |
| `/branch [name]` | Forks the conversation at this point. Alias: `/fork` |
| `/resume [session]` | Resumes a previous session. Alias: `/continue` |

### Model & Configuration
| Command | What it does |
|---|---|
| `/model [model]` | Shows or switches current model |
| `/effort [low|medium|high|max|auto]` | Sets model reasoning depth |
| `/config` | Opens configuration settings. Alias: `/settings` |
| `/fast [on|off]` | Toggles fast mode |

### Memory & Context
| Command | What it does |
|---|---|
| `/memory` | Edit CLAUDE.md files and manage auto-memory |
| `/add-dir <path>` | Adds a directory to current session context |
| `/context` | Visualises context usage with optimisation suggestions |

### Workflow & Planning
| Command | What it does |
|---|---|
| `/plan [description]` | Enters plan mode — shows plan, waits for approval |
| `/agents` | Manages agent configurations |
| `/tasks` | Lists and manages background tasks |
| `/btw <question>` | Asks a side question without affecting main conversation |

### Recovery & Undo
| Command | What it does |
|---|---|
| `/rewind` | Reverts to a previous checkpoint. Alias: `/checkpoint` |
| `Escape × 2` | Quick undo of last action |

### Review & Quality
| Command | What it does |
|---|---|
| `/diff` | Interactive diff viewer — uncommitted changes and per-turn diffs |
| `/security-review` | Analyses current branch for security vulnerabilities |
| `/pr-comments [PR]` | Fetches GitHub PR comments (requires `gh` CLI) |

### Analytics & Cost
| Command | What it does |
|---|---|
| `/cost` | Shows token usage for the session |
| `/insights` | Generates a report on session patterns and history |
| `/stats` | Visualises daily usage, streaks, model preferences |
| `/usage` | Shows plan usage limits and rate limit status |

### Feedback & Debug
| Command | What it does |
|---|---|
| `/feedback [report]` | Submits feedback or bug report. Alias: `/bug` |
| `/doctor` | Diagnoses your Claude Code installation |
| `/status` | Opens Settings Status tab — version, model, account |

### Session Utilities
| Command | What it does |
|---|---|
| `/export [filename]` | Exports the conversation as plain text |
| `/rename [name]` | Renames the current session |
| `/copy [N]` | Copies last response to clipboard |

---

## 9. Handling Claude When It Goes Wrong

### When it misunderstands:
```
That's not what I meant. Stop and let me clarify.

I don't want you to rewrite the whole function. I just want
you to add a null check at line 12. Only that.
```

Explicitly say "stop" and redirect. Don't pile on more instructions — that compounds confusion.

### When the approach is wrong:
```
/rewind
```
Pick the checkpoint before it went wrong. Restart with a better-scoped prompt.

### When one edit is wrong:
`Escape × 2` — undo that one action and redirect.

### When you want to reject a diff:
At the `Apply this change? [y/n/e]` prompt:
- `n` — skip this edit, Claude continues planning
- `e` — open in editor, you manually fix it, then Claude continues

### When Claude is confidently wrong:
```
I don't think that's right. The function is called from three
places — read all three call sites before deciding how to
change the signature.
```

Push back directly with specific reasoning. Claude responds well to being challenged.

---

## 10. Multi-turn Conversation — Keeping Context Sharp

### Re-anchor When Switching Topics
```
We're done with the auth refactor. Now switching to a completely
different problem: the PDF generation in src/pdf/generator.py
is running out of memory on large documents.
```

### Reference Specific Files and Lines
```
In src/payments/processor.py, line 84, the retry logic
is missing exponential backoff. Add it.
```

### Confirm Before Long Tasks
```
Before you start, tell me which files you're going to modify.
```

This forces Claude to plan explicitly. You can catch wrong assumptions before any code is written.

---

## 11. Internal Mechanics — How Claude Processes Your Prompt

```
Your prompt arrives
       ↓
Tokenization: your words → tokens
       ↓
Context assembly:
  [System prompt]
  + [CLAUDE.md contents]
  + [Conversation history]
  + [Your new prompt]
  = Full context sent to model
       ↓
Model generates a response
(mix of: text to show you + tool calls to execute)
       ↓
Tool calls extracted and executed:
  - Read file → file contents returned
  - Run command → stdout/stderr returned
  - Edit file → diff shown to you
       ↓
Tool results appended to context
       ↓
Model generates next response
(loop continues until task done or model responds to you)
```

### Why This Matters Practically

**Everything is in the context window.** Every file Claude reads, every command output, every conversation turn accumulates in one large context. When that context fills up, Claude starts losing early information first (recency bias).

This is why:
- `/compact` matters on long sessions
- Specific prompts work better than vague ones (less back-and-forth = less context used)
- Pointing Claude at specific files is more efficient than letting it explore freely

**Tool calls are synchronous within the loop.** Claude can't do two things at once. It reads a file, gets the result, decides what to do next, reads another file — sequentially.

---

## 12. Practical Exercise — The Full Interaction Workout

Use the calculator project from Chapter 2 or any project you have.

**Exploration prompts:**
```
Explain the overall structure of this project
What's the most complex function here and why?
Are there any obvious code quality issues?
```

**Task prompts:**
```
Add type hints to all functions
Add a modulo operation function following the same pattern as the others
Write a comprehensive docstring for each function
```

**Pattern practice:**
```
Before changing anything, explain what changes you'd make to add
logging to every function. Then wait for my approval.
```

**`@` file mention practice:**
```
How does the divide function handle errors? @calculator.py
Compare the structure of @calculator.py with @test_calculator.py
```

**`#` memory mode practice:**
```
# Always add type hints to new functions
# Never add inline comments unless the logic is non-obvious
```

**Slash command workout — run each one:**
```
/diff
/cost
/context
/insights
/stats
/security-review
```

**Recovery practice:**
- Let Claude start editing something
- Press `Escape` to interrupt
- Use `Escape × 2` to undo
- Use `/rewind` to go back further

---

## Chapter 3 Summary

| Topic | Key Takeaway |
|---|---|
| Mindset | Direct a developer, don't prompt an AI |
| Context management | Too much irrelevant context hurts performance — guide Claude to the right files |
| Interaction modes | Question / Task / Iterative / One-shot |
| Prompt structure | Context + Task + Constraints + Output format |
| Key patterns | Explain first, follow existing, scope limiter, step by step |
| `@` file mentions | Pinpoint exactly which files Claude should read — faster and more token-efficient |
| `#` memory mode | Update CLAUDE.md on the fly without breaking your session flow |
| Reading output | Watch tool use indicators and always read diffs |
| Daily slash commands | `/clear`, `/compact`, `/memory`, `/diff`, `/cost`, `/context`, `/plan` |
| When things go wrong | Escape, Escape×2, /rewind, redirect with specific language |
| Internal mechanics | Everything is context; tools run sequentially in a loop |

---

*Previous: [Chapter 2 — Installation & Setup](./chapter-02-installation-and-setup.md)*
*Next: [Chapter 4 — CLAUDE.md: Persistent Context & Project Memory](./chapter-04-claude-md-persistent-context.md)*