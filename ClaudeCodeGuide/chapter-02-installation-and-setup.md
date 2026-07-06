# Chapter 2: Installation & Setup

---

## 1. Prerequisites — What You Need First

Before installing Claude Code, you need two things:

### Node.js
Claude Code is built on Node.js. You need **Node.js version 18 or higher**.

Check if you have it:
```bash
node --version   # Should show v18.0.0 or higher
npm --version    # Should show alongside Node
```

If not installed, get it from **nodejs.org** — download the LTS version.

### An Anthropic Account
You need either:
- A **Claude.ai account** (Pro, Team, or Enterprise plan) — for billing through claude.ai
- An **Anthropic API key** — for billing directly through the API

> ⚠️ Free Claude.ai accounts **do not** include Claude Code access.

---

## 2. Installation

Claude Code is installed as a global npm package:

```bash
npm install -g @anthropic-ai/claude-code
```

That's it. One command. This installs the `claude` binary globally so you can run it from any directory.

**Verify the installation:**
```bash
claude --version
```

### Platform-Specific Notes

| Platform | Notes |
|---|---|
| **macOS** | Works out of the box. No extra steps. |
| **Linux** | Works out of the box on most distributions. |
| **Windows** | Requires **Git for Windows** to be installed first (provides the bash shell layer). Install from gitforwindows.org, then run npm install inside Git Bash. |
| **WSL** | Fully supported and recommended Windows experience. Treat it like Linux. |

### Auto-Updates
Claude Code updates itself automatically when new versions are released. To force an update manually:
```bash
npm update -g @anthropic-ai/claude-code
```

---

## 3. Authentication

When you run `claude` for the first time, it prompts you to authenticate. Two paths:

### Path A — Claude.ai Account (Recommended for most)
```bash
claude
# First-run wizard launches
# Choose: "Login with Claude.ai"
# Browser opens → you authorize → done
```

This ties Claude Code to your Claude.ai subscription. Usage costs come out of your plan.

### Path B — API Key
```bash
claude config set apiKey YOUR_API_KEY
```

Or set it as an environment variable (useful for CI/CD):
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Claude Code automatically picks up the `ANTHROPIC_API_KEY` environment variable — no config command needed.

**Check your current auth status:**
```bash
claude config list
```

---

## 4. Your First `claude` Session

Navigate to any project directory (or create a test one):

```bash
mkdir my-test-project
cd my-test-project
echo "print('hello world')" > app.py
claude
```

You'll see something like:

```
╭─────────────────────────────────────╮
│ ✻ Welcome to Claude Code!           │
│                                     │
│ /help for commands, /exit to quit   │
╰─────────────────────────────────────╯

claude>
```

You're now in an **interactive session**. Claude Code has already scanned your directory. Try:

```
claude> What files are in this project and what do they do?
```

Claude will read your files and respond. Watch closely — you'll see it use the **Read** tool in real time, shown in the terminal output before its answer.

### The `/init` Command — Your Most Important First Step

When starting Claude Code in **any new project for the first time**, run this immediately:

```
/init
```

This is one of the most valuable commands in Claude Code and is commonly overlooked by beginners. Here's what it does:

Claude analyses your **entire codebase** and builds understanding of:
- The project's purpose and overall architecture
- Important commands (build, test, run, deploy)
- Critical files and their roles
- Coding patterns and conventions already in use

After analysis, Claude **automatically creates a `CLAUDE.md` file** for you — a fully populated project memory file written from what it observed in your code. You will be prompted to approve file writes:

```
Claude wants to create: CLAUDE.md
  [Enter] to approve   [Shift+Tab] to switch to auto-accept mode
```

- Press `Enter` to approve each write individually
- Press `Shift+Tab` to switch to auto-accept edits mode so Claude writes files freely for the rest of the session (faster for initial setup)

### Enhanced `/init` with `CLAUDE_CODE_NEW_INIT=1`

For a more thorough interactive setup, set the environment variable before running:

```bash
CLAUDE_CODE_NEW_INIT=1 claude
/init
```

This enables a multi-phase flow where `/init`:
1. Asks which artifacts to set up: CLAUDE.md files, skills, and hooks
2. Explores your codebase with a subagent
3. Fills gaps via follow-up questions
4. Presents a reviewable proposal before writing any files

If a CLAUDE.md already exists, `/init` suggests improvements rather than overwriting it.

> 💡 **This is the right order:** `/init` first → Claude generates CLAUDE.md → you refine it → future sessions start context-loaded. Without `/init`, you're writing CLAUDE.md from scratch by hand.

**What a `/init`-generated CLAUDE.md looks like:**
```markdown
# Project: my-fastapi-app

## Overview
FastAPI REST API with PostgreSQL database for user management.

## Key Commands
- `uvicorn main:app --reload` — start dev server
- `pytest tests/` — run tests
- `alembic upgrade head` — run migrations

## Architecture
- `main.py` — application entry point
- `routers/` — API route handlers
- `models/` — SQLAlchemy database models
- `schemas/` — Pydantic request/response schemas

## Patterns
- Dependency injection via FastAPI `Depends()`
- Repository pattern for database access
- Async route handlers throughout
```

Claude generated all of this by reading your code — you didn't write a word of it. You then refine and extend it (covered in Chapter 4).

---

## 5. The Terminal Interface — A Full Tour

### The Prompt Area
```
claude>
```
Accepts: natural language tasks, slash commands (start with `/`), and follow-up messages mid-task.

### The Tool Use Display
When Claude acts, it shows you what it's doing:

```
● Reading file: app.py
● Running: python app.py
● Editing: app.py
  ┌─ diff ──────────────────┐
  │ - print('hello world')  │
  │ + print('Hello, World!') │
  └──────────────────────────┘
  Apply this change? [y/n]
```

### Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Enter` | Send your message |
| `Shift + Enter` | New line without sending (multi-line input) |
| `Escape` | Interrupt Claude mid-task |
| `Escape × 2` | Undo last action / restore checkpoint |
| `Shift + Tab` | Cycle through permission modes (Default → Auto-accept edits → Plan mode → Auto mode) |
| `Ctrl + C` | Exit session |
| `↑ / ↓` arrows | Scroll through prompt history |

### Permission Modes — `Shift+Tab` Cycle

`Shift+Tab` cycles through four permission modes — understanding each one is essential:

| Mode | What it does |
|---|---|
| **Default** | Claude asks before file edits and shell commands — safest, most controlled |
| **Auto-accept edits** | Claude edits files without asking, but still asks before running commands |
| **Plan mode** | Read-only — Claude builds a complete plan first, you approve before any action |
| **Auto mode** | Background classifier evaluates all actions; only risky ones are blocked |

> 💡 **The Escape key is your safety net.** If Claude starts doing something unexpected mid-task, press Escape immediately. It stops the agent loop. You can then redirect, undo with `Escape × 2`, or start fresh with `/clear`.

---

## 6. All Core Slash Commands

### Session Management
| Command | What it does |
|---|---|
| `/help` | Lists all slash commands with descriptions |
| `/init` | Analyses codebase and generates CLAUDE.md — run first in every new project |
| `/exit` | Ends the current session cleanly. Alias: `/quit` |
| `/clear` | Clears conversation history and starts fresh (keeps files). Aliases: `/reset`, `/new` |
| `/compact [instructions]` | Summarises conversation to free up context window space |
| `/branch [name]` | Creates a branch/fork of the conversation at this point. Alias: `/fork` |

### Model & Configuration
| Command | What it does |
|---|---|
| `/model [model]` | Shows or changes the current model. Use arrow keys to adjust effort level |
| `/effort [low\|medium\|high\|max\|auto]` | Sets model effort level — affects reasoning depth |
| `/config` | Opens configuration settings. Alias: `/settings` |
| `/fast [on\|off]` | Toggles fast mode on or off |

### Memory & Context
| Command | What it does |
|---|---|
| `/memory` | Edit CLAUDE.md files, enable/disable auto-memory, view auto-memory entries |
| `/add-dir <path>` | Adds an additional directory to Claude's context for the session |
| `/context` | Visualises current context usage as a coloured grid with optimisation suggestions |

### Workflow & Tasks
| Command | What it does |
|---|---|
| `/plan [description]` | Enters plan mode — Claude shows its plan and waits for approval before acting |
| `/agents` | Manages agent configurations |
| `/tasks` | Lists and manages background tasks |
| `/btw <question>` | Asks a quick side question without adding it to the main conversation |

### Recovery & Undo
| Command | What it does |
|---|---|
| `/rewind` | Rewinds conversation and/or code to a previous checkpoint. Alias: `/checkpoint` |
| `Escape × 2` | Quick undo of the last action |

### Review & Quality
| Command | What it does |
|---|---|
| `/diff` | Opens interactive diff viewer showing uncommitted changes and per-turn diffs |
| `/security-review` | Analyses pending changes on current branch for security vulnerabilities |
| `/pr-comments [PR]` | Fetches and displays comments from a GitHub pull request (requires `gh` CLI) |

### Analytics & Cost
| Command | What it does |
|---|---|
| `/cost` | Shows token usage statistics for the session |
| `/insights` | Generates a report analysing your Claude Code sessions and interaction patterns |
| `/stats` | Visualises daily usage, session history, streaks, and model preferences |
| `/usage` | Shows plan usage limits and rate limit status |

### Feedback & Debug
| Command | What it does |
|---|---|
| `/feedback [report]` | Submits feedback or bug report to Anthropic. Alias: `/bug` |
| `/doctor` | Diagnoses and verifies your Claude Code installation and settings |
| `/status` | Opens the Settings Status tab — version, model, account, connectivity |

### Session Utilities
| Command | What it does |
|---|---|
| `/export [filename]` | Exports the current conversation as plain text |
| `/rename [name]` | Renames the current session |
| `/resume [session]` | Resumes a previous conversation by ID or name. Alias: `/continue` |
| `/copy [N]` | Copies the last assistant response to clipboard |

---

## 7. CLI Flags — Running Claude Without Interactive Mode

```bash
# One-shot task (no interactive session)
claude "write a function that validates email addresses"

# Run on a specific directory without cd-ing
claude --dir /path/to/project "explain this codebase"

# Use a specific model
claude --model claude-sonnet-4-6 "refactor my auth module"

# Run without any permission prompts (auto mode)
claude --permission-mode auto "fix all lint errors"

# Print output to stdout (good for piping)
claude --print "summarize main.py"

# Non-interactive mode for CI/CD pipelines
claude -p "run tests and report failures"

# Resume the most recent conversation
claude --continue

# Select from recent conversations to resume
claude --resume

# Fork a session — creates a new session from the current point
# without affecting the original (safe parallel experimentation)
claude --continue --fork-session

# Append extra instructions to the system prompt (every invocation)
claude --append-system-prompt "Always prefix responses with REVIEW:"
```

> ⚠️ **About `--fork-session`:** When you resume the same session in multiple terminals, both write to the same session file and messages get interleaved. Use `--fork-session` to give each terminal its own clean session branched from the same starting point. Session-scoped permissions are not restored on resume — you will need to re-approve those.

---

## 8. Configuration — `claude config`

Claude Code stores configuration in `~/.claude/` on your system:

```bash
claude config list          # View all config values
claude config set theme dark  # Set a value
claude config get apiKey    # Get a specific value
claude config reset         # Reset to defaults
```

### Key Configuration Options

| Config Key | What it controls |
|---|---|
| `apiKey` | Your Anthropic API key |
| `model` | Default model for all sessions |
| `theme` | Terminal color theme (`dark` / `light`) |
| `autoApprove` | Skip confirmation on file edits (default: false) |
| `maxTokens` | Default token limit per request |
| `editor` | Preferred editor for opening files |

---

## 9. Internal Mechanics — How Claude Code Bootstraps

Here's exactly what happens in the first few seconds after you type `claude`:

```
Step 1: DIRECTORY SCAN
   └── Claude reads your current directory tree
   └── Identifies file types, languages, frameworks
   └── Notes presence of: package.json, pyproject.toml,
       Makefile, .git, etc.

Step 2: CLAUDE.md CHECK
   └── Looks for CLAUDE.md in current dir
   └── Looks for global CLAUDE.md in ~/.claude/
   └── Loads instructions into system context if found

Step 3: GIT CONTEXT
   └── Runs `git status` and `git log --oneline -10`
   └── Understands what's changed, what branch you're on

Step 4: SYSTEM PROMPT CONSTRUCTION
   └── Combines: base instructions + project context +
       CLAUDE.md contents + git context
   └── This becomes the "world" Claude operates in

Step 5: SESSION OPENS
   └── Interactive prompt appears
   └── Claude is ready, already knowing your project's shape
```

### The `~/.claude/` Directory

```
~/.claude/
├── config.json          ← your settings
├── CLAUDE.md            ← your global instructions (applies everywhere)
├── sessions/            ← session history logs
├── checkpoints/         ← undo snapshots (30 day retention)
└── mcp/                 ← MCP server configs (Chapter 9)
```

---

## 10. The `.claudeignore` File

If your project has large auto-generated directories (like `node_modules`, `dist`, `.next`), Claude wastes time scanning them. Create a `.claudeignore`:

```
node_modules/
dist/
.next/
build/
*.log
*.lock
```

Same syntax as `.gitignore`. Claude skips these during its directory scan — significantly speeding up startup on large projects.

---

## 11. Undo System — `Escape × 2` vs `/rewind`

These two are different and commonly confused:

| | `Escape × 2` | `/rewind` |
|---|---|---|
| **Scope** | One action | Any checkpoint |
| **Speed** | Instant | Shows a menu |
| **Granularity** | Single tool call | Whole session snapshots |
| **Use when** | Claude just did one wrong thing | Claude went down a wrong path for a while |
| **Restore options** | File change only | Chat only / Code only / Both / Summarise from here |

**`Escape × 2`** = *"undo that one thing you just did"*
**`/rewind`** = *"take me back to before this whole approach"*

> ⚠️ **Important:** Checkpoints only cover file changes Claude made — not external side effects like database writes, API calls, or deployments. They are also local to your session and separate from git.

---

## 12. Session Resume and Forking

Claude Code saves every conversation locally. Sessions are tied to your current working directory.

```bash
claude --continue    # Resume the most recent session in this directory
claude --resume      # Pick from a list of recent sessions
```

Use `/rename` to give sessions descriptive names like `"oauth-refactor"` so you can find them easily.

### Forking a Session

To branch off and try a different approach without affecting the original session:

```bash
claude --continue --fork-session
```

This creates a new session ID while preserving the full conversation history up to that point. The original session remains unchanged. Use this when you want to explore two different solutions from the same starting point — like git branches for your conversations.

> ⚠️ **Note:** When you resume a session, your full conversation history is restored — but session-scoped permissions are **not**. You will need to re-approve those.

---

## 12. Common Setup Problems & Fixes

| Problem | Cause | Fix |
|---|---|---|
| `claude: command not found` | npm global bin not in PATH | `export PATH="$PATH:$(npm bin -g)"` |
| Auth loop keeps repeating | Browser didn't complete auth | `claude config set apiKey YOUR_KEY` instead |
| `EACCES` permission error on install | npm global directory permissions | `sudo npm install -g` or fix npm permissions |
| Windows: `bash not found` | Git for Windows not installed | Install Git for Windows first |
| Claude seems slow to start | Large directory being scanned | Add a `.claudeignore` file |

---

## 13. Practical Exercise

**Step 1:** Create a test project:
```bash
mkdir claude-practice && cd claude-practice && git init
```

Create `calculator.py`:
```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b  # Bug: no zero division check
```

**Step 2:** Start Claude Code and run `/init`:
```bash
claude
```
Once inside the session:
```
/init
```
Watch Claude read your files and generate a CLAUDE.md. At each file write prompt, press `Enter` to approve individually or `Shift+Tab` to switch to Auto-accept edits mode (so Claude writes files without asking for the rest of the session).

**Step 3:** Try each of these prompts and observe the tool usage:
```
What's in this project?
Is there anything wrong with calculator.py?
Fix the bug and add proper error handling
Add a test file for these functions
```

**Step 4:** Run the slash commands:
```
/diff
/cost
/insights
/context
```

**Step 5:** Practice interrupting:
```
Rewrite the entire calculator to use a class-based approach
```
→ While it's running, press `Escape`
→ Then try `Escape × 2` to undo

---

## Chapter 2 Summary

| Topic | Key Takeaway |
|---|---|
| Installation | `npm install -g @anthropic-ai/claude-code` — one command |
| Auth | claude.ai account or `ANTHROPIC_API_KEY` env variable |
| Starting | `cd your-project && claude` |
| Interface | Tool use is visible, diffs shown before applying |
| Escape key | Your interrupt and undo safety net |
| Key commands | `/help`, `/clear`, `/compact`, `/rewind`, `/cost`, `/context`, `/plan` |
| Bootstrap | Claude scans dir + git + CLAUDE.md before you type anything |
| Config | Lives in `~/.claude/config.json` |
| `.claudeignore` | Speeds up large projects by skipping irrelevant dirs |

---

*Previous: [Chapter 1 — What Is Claude Code?](./chapter-01-what-is-claude-code.md)*
*Next: [Chapter 3 — Basic Interaction & Core Commands](./chapter-03-basic-interaction-and-core-commands.md)*