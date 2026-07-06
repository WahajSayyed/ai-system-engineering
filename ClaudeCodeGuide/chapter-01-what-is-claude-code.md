# Chapter 1: What Is Claude Code?

---

## 1. Concept — What Problem Does It Solve?

Before Claude Code existed, a typical developer workflow looked like this:

- Open your editor → get stuck → **switch to a browser** → search Stack Overflow → copy paste → switch back → repeat.
- Or: open Claude.ai → **manually copy-paste** your code → ask a question → copy the answer back → repeat.

This constant context-switching is expensive. You lose your train of thought, you lose codebase context, and you waste time narrating your own code to an AI that can't see it.

**Claude Code solves this by living where you work — the terminal — and seeing exactly what you see.**

It's not a chatbot you paste code into. It's an **agent** that:
- Reads your actual files
- Runs real commands
- Makes real edits
- Works iteratively until the task is done

---

## 2. Claude Code vs Claude.ai vs Claude API

These three are often confused. Here's a clean mental model:

| | **Claude.ai** | **Claude API** | **Claude Code** |
|---|---|---|---|
| **What it is** | A chat web app | Raw API access | A coding agent CLI |
| **Who uses it** | General users | Developers building apps | Developers coding |
| **Interface** | Browser | HTTP calls / SDKs | Terminal |
| **Sees your code?** | Only what you paste | Only what you send | ✅ Yes, directly |
| **Can run commands?** | ❌ No | ❌ No | ✅ Yes |
| **Makes file edits?** | ❌ No | ❌ No | ✅ Yes |
| **Memory of project?** | ❌ Per session only | ❌ Per call only | ✅ Via CLAUDE.md |

Think of it this way:
- **Claude.ai** = talking to Claude through a window
- **Claude API** = Claude as a brain you wire into your own app
- **Claude Code** = Claude as a pair programmer sitting next to you in your terminal

---

## 3. What Claude Code Actually Does

Claude Code is built around **tools** — discrete capabilities it can invoke to interact with your system:

| Tool Category | Examples |
|---|---|
| **Read** | Read files, list directories, search code with grep |
| **Write** | Create files, edit files, delete files |
| **Execute** | Run shell commands, run tests, run build scripts |
| **Git** | Stage, commit, branch, diff, push, open PRs |
| **Web** | Fetch URLs, search the web (when enabled) |
| **MCP** | Connect to external services like Jira, Slack, databases |

When you say *"add input validation to my login function"*, Claude Code doesn't just write text back at you. It:

1. Reads your login file
2. Understands the existing structure
3. Plans the change
4. Writes the edit
5. Optionally runs your tests to verify

---

## 4. Internal Mechanics — The Agent Loop

This is the most important thing to understand about how Claude Code works internally. It is **not** a simple question-and-answer system. It runs what's called an **agentic loop**:

```
┌─────────────────────────────────────────────┐
│                                             │
│   You give a task                           │
│          ↓                                  │
│   Claude makes a PLAN                       │
│          ↓                                  │
│   Claude picks a TOOL to use                │
│   (read file / run command / edit file)     │
│          ↓                                  │
│   Tool runs → result comes back             │
│          ↓                                  │
│   Claude OBSERVES the result                │
│          ↓                                  │
│   Is the task done?                         │
│      YES → respond to you                   │
│      NO  → pick next tool → loop again      │
│                                             │
└─────────────────────────────────────────────┘
```

**Why this matters:** Claude Code can take 10, 20, 30+ autonomous steps on a complex task before coming back to you. It's not answering you — it's *working*.

### What triggers the loop to stop?

- Task is complete
- Claude hits something it needs **your permission** for (e.g., deleting files)
- Claude gets stuck and asks for clarification
- You press `Escape` to interrupt

### The Permission Layer

Claude Code operates on a **trust and permission model**. Some actions it takes freely, others it asks you first:

```
Low risk  ──────────────────────────────► High risk
   │                                          │
Read files    Edit files    Run commands    Delete files
(silent)      (shows diff)  (asks once)    (always asks)
```

---

## 5. Where Claude Code Runs

Claude Code is not just one interface. It runs in several environments:

- **Terminal / CLI** — the primary, most powerful interface (`claude` command)
- **VS Code extension** — embedded in your editor sidebar
- **JetBrains plugin** — same for IntelliJ, PyCharm, WebStorm etc.
- **Claude Desktop App** — GUI wrapper with terminal underneath
- **claude.ai/code** — browser-based version (lighter capabilities)

The **CLI is the canonical experience**. Everything in this guide is CLI-first, because that's where the full power lives.

---

## 6. Key Commands Introduced in This Chapter

Even before installation, understand these foundational CLI entry points:

| Command | What it does |
|---|---|
| `claude` | Starts an interactive session in your current directory |
| `claude "your task here"` | Runs a one-shot task without entering interactive mode |
| `claude --help` | Shows all available flags and options |
| `claude --version` | Shows the installed version |
| `claude --model` | Specifies which Claude model to use |

Inside an interactive session, your first slash commands:

| Command | What it does |
|---|---|
| `/help` | Lists all available slash commands |
| `/exit` or `Ctrl+C` | Ends the session |
| `/bug` | Reports a bug directly to Anthropic |
| `/model` | Switches the model mid-session |

---

## 7. Common Misconceptions — Cleared Up Now

**❌ "Claude Code just writes code for me"**
✅ It *works with* your code. It reads, edits, runs, tests, and iterates. It's a collaborator, not a code generator.

**❌ "It knows my whole codebase automatically"**
✅ It reads what's relevant for the task. For large projects, you'll learn to guide its context (Chapter 8).

**❌ "It's always running in the background"**
✅ Claude Code only runs when you invoke it. It's not a daemon or background process.

**❌ "I need to know how to prompt AI to use it"**
✅ You talk to it like you'd talk to a developer colleague. Natural, direct, task-focused.

**❌ "It will overwrite my files without asking"**
✅ By default it shows diffs and asks for confirmation on edits. You control the trust level.

---

## 8. Mental Model to Carry Forward

Think of Claude Code as a **junior-to-mid developer** who:
- Has read every programming book ever written
- Can type 1000x faster than any human
- But needs **you** to set direction, review decisions, and catch domain-specific mistakes
- Gets better the more context you give them (via CLAUDE.md, clear prompts, etc.)

Your job shifts from *writing code* to *directing and reviewing*. The better you get at that, the more powerful Claude Code becomes.

---

## Chapter 1 Summary

| Concept | Key Takeaway |
|---|---|
| What it is | A terminal-based coding agent, not a chatbot |
| How it works | An agentic loop: plan → tool → observe → repeat |
| vs Claude.ai | Claude Code sees and edits your actual files |
| vs Claude API | Claude Code is for developers coding, not building apps |
| Permission model | Low-risk actions are silent; high-risk always asks |
| Where it runs | CLI, VS Code, JetBrains, Desktop, Web |

---

## ✅ Before Chapter 2 — Check Your Understanding

1. Can you explain the agent loop in your own words?
2. Do you understand why Claude Code is different from copy-pasting into Claude.ai?
3. Are you clear on which interface (CLI/VS Code/etc.) you'll use?

---

*Next: [Chapter 2 — Installation & Setup](./chapter-02-installation-and-setup.md)*
