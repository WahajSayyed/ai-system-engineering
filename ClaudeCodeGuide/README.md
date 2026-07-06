# Developer Guide to Claude Code

A comprehensive, structured guide to mastering Claude Code — from first install to advanced agentic workflows and Claude Certified Architect (CCA) exam preparation.

---

## How to Use This Guide

Each chapter builds on the last. Read sequentially for the full learning experience, or jump to a specific topic using the index below. Every chapter covers:

- **Concepts** — what it is and why it matters
- **Internal Mechanics** — how Claude Code implements it under the hood
- **Hands-On Practice** — step-by-step exercises
- **Common Mistakes** — what to watch out for

---

## Table of Contents

### Part 1 — Foundations (Beginner)

| Chapter | Topic | CCA Domains |
|---|---|---|
| [Chapter 1](./chapter-01-what-is-claude-code.md) | What Is Claude Code? | Domain 1 |
| [Chapter 2](./chapter-02-installation-and-setup.md) | Installation & Setup | Domain 3 |
| [Chapter 3](./chapter-03-basic-interaction-and-core-commands.md) | Basic Interaction & Core Commands | Domain 1, 4 |

### Part 2 — Core Workflows (Intermediate)

| Chapter | Topic | CCA Domains |
|---|---|---|
| [Chapter 4](./chapter-04-claude-md-persistent-context.md) | CLAUDE.md — Persistent Context & Project Memory | Domain 3, 5 |
| [Chapter 5](./chapter-05-file-editing-and-multi-file-tasks.md) | File Editing & Multi-File Tasks | Domain 1, 4 |
| [Chapter 6](./chapter-06-debugging-and-bug-fixing.md) | Debugging & Bug Fixing | Domain 1, 4 |
| [Chapter 7](./chapter-07-git-integration.md) | Git Integration | Domain 3 |
| [Chapter 8](./chapter-08-context-window-management.md) | Context Window Management | Domain 5 |

### Part 3 — Advanced Usage

| Chapter | Topic | CCA Domains |
|---|---|---|
| [Chapter 8](./chapter-08-context-window-management.md) | Context Window Management | Domain 5 |
| [Chapter 9](./chapter-09-mcp-model-context-protocol.md) | MCP — Model Context Protocol | Domain 2 |
| [Chapter 10](./chapter-10-multi-agent-and-parallel-workflows.md) | Multi-Agent & Parallel Workflows | Domain 1, 2 |
| [Chapter 11](./chapter-11-checkpoints-and-session-recovery.md) | Checkpoints & Session Recovery | Domain 3 |
| [Chapter 12](./chapter-12-cicd-automation.md) | CI/CD Automation | Domain 3 |

### Part 4 — Expert & Internal Architecture

| Chapter | Topic | CCA Domains |
|---|---|---|
| [Chapter 13](./chapter-13-claude-agent-sdk.md) | The Claude Agent SDK | Domain 1, 2 |
| [Chapter 14](./chapter-14-security-trust-and-safety.md) | Security, Trust & Safety | Domain 6 |
| Chapter 15 *(coming soon)* | Output Quality & Verification | Domain 4 |
| Chapter 16 *(coming soon)* | Advanced CLAUDE.md & Custom Commands | Domain 3 |

---

## Claude Certified Architect (CCA) Exam Coverage

This guide is structured to cover all six CCA exam domains:

| Domain | Topic | Exam Weight |
|---|---|---|
| Domain 1 | Agentic Architecture & Tool Use | 27% |
| Domain 2 | Multi-Agent Systems & MCP | 18% |
| Domain 3 | Claude Code Configuration & Workflows | 20% |
| Domain 4 | Prompt Engineering & Structured Output | 15% |
| Domain 5 | Context & Memory Management | 15% |
| Domain 6 | Security, Safety & Responsible AI | 5% |

---

## Quick Reference — Commands

### CLI Entry Points
```bash
claude                          # Start interactive session
claude "your task here"         # One-shot task
claude --help                   # All flags and options
claude --version                # Installed version
claude --model <model-name>     # Specify model
claude --yes "task"             # Auto-approve all edits
claude --plan "task"            # Plan mode (shows plan before acting)
claude --print "task"           # Print output to stdout
claude --no-interactive "task"  # Non-interactive mode (CI/CD)
claude --dir /path "task"       # Run on specific directory
claude --max-tokens 4000 "task" # Limit token usage
```

### Slash Commands — Session
```
/help          List all commands
/exit          End session
/clear         Clear history (keep files)
/compact       Compress history to save context
/reset         Full reset
```

### Slash Commands — Context & Memory
```
/memory        Show active context and token usage
/add-dir       Add directory to context
/tasks         List and manage background tasks
```

### File & Memory Shortcuts
```
@filename      Mention a file to load it into context
               e.g: How does auth work? @src/auth/login.py

# instruction  Memory mode — update CLAUDE.md on the fly
               e.g: # Never use print(), use the logger instead
```

### Slash Commands — Review & Quality
```
/diff          Show all git changes this session
/insights      Session stats (tokens, tools, cost)
/status        Session health check
```

### Slash Commands — Recovery
```
/rewind        Restore to a previous checkpoint
Escape × 2     Quick undo of last action
```

### Slash Commands — Git
```
/pr-comments   Fetch GitHub PR comments (requires gh CLI)
/security-review  Analyse current branch for vulnerabilities
/diff          Show git diff
```

### Slash Commands — Model & Config
```
/model                    Show current model
/model claude-opus-4-5    Switch model mid-session
/config                   Open configuration
/bug                      Report a bug to Anthropic
```

---

## Key Concepts Quick Reference

| Concept | One-liner |
|---|---|
| Agent loop | Plan → Tool → Observe → Repeat |
| CLAUDE.md hierarchy | Global → Project → Local → Subdirectory (most specific wins) |
| CLAUDE.local.md | Personal, gitignored — team never sees it |
| `@` file mention | Load a specific file into context: `How does auth work? @src/auth/login.py` |
| `#` memory mode | Update CLAUDE.md on the fly: `# Always use async for DB calls` |
| `/init` command | Auto-generates CLAUDE.md by analysing your codebase — run first in every new project |
| Programmatic vs prompt | Hard constraints must be code, not instructions |
| Context window | Everything accumulates; use `/compact` on long sessions |
| Plan mode | Claude shows plan and waits for approval before acting |
| Checkpoints | Auto-saved after every prompt, retained 30 days |
| `.claudeignore` | Skip irrelevant dirs to speed up large project scanning |
| MCP | Open protocol for connecting Claude to external services |

---

## Contributing

This guide is a living document. As new chapters are added or Claude Code updates, this repo will be updated accordingly. Feel free to open issues for corrections or suggestions.

---

*Guide version: March 2026 | Claude Code version: latest*