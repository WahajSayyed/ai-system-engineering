# Chapter 7: Git Integration

> **CCA Exam Note:** This chapter covers Domain 3 (Claude Code Configuration & Workflows, 20%). The CI/CD scenario — "Claude Code for CI/CD" — is one of the six core exam scenario contexts. Git workflow, commit discipline, and branch management with Claude Code are directly tested.

---

## 1. Why Git Integration Matters

Git is where individual work becomes team work. Every edit Claude makes is worthless if it can't be safely committed, reviewed, and merged. Claude Code's git integration closes that loop — it doesn't just write code, it manages the full lifecycle from edit to pull request.

But there's a subtlety worth understanding upfront:

> **Claude Code has no dedicated git slash commands.** There is no `/commit`, `/pr`, or `/push`. Git operations happen one of two ways: you ask Claude in natural language, or Claude runs git commands itself via the bash tool.

This is actually more powerful than dedicated commands — you can describe complex git operations in plain English and Claude executes the right sequence of git commands to make them happen.

---

## 2. How Claude Code Interacts with Git

Claude Code interacts with git through its **bash tool** — it runs real terminal commands. When you say "commit these changes", Claude is literally running:

```bash
git add src/payments/processor.py
git add tests/test_payments.py
git status
git diff --staged
git commit -m "Add idempotency key support to execute_payment()"
```

It is not abstracting git — it is using git exactly as a developer would, but faster and with better commit message writing.

### What Claude Knows About Your Repo at Startup

From Chapter 2, you know Claude reads git context during bootstrap. Specifically:

```bash
git status          # what's changed, staged, untracked
git log --oneline -10  # last 10 commits
git branch          # current branch and recent branches
```

This means the moment you start a session, Claude already knows:
- What branch you are on
- What has been recently committed
- What is currently modified but uncommitted
- What files are untracked

You do not need to tell Claude any of this — it already knows.

---

## 3. Commit Operations

### Basic Commit Request

The simplest interaction:

```
Commit the changes you just made.
```

Claude will:
1. Run `git status` to see what has changed
2. Run `git diff` to review the changes
3. Stage the relevant files with `git add`
4. Write a commit message following conventional commit format
5. Run `git commit`
6. Report the commit hash back to you

### Writing Good Commit Messages

Claude writes commit messages significantly better than most developers do under time pressure. To get the best results:

```
Commit these changes. Write a detailed commit message that:
- Uses conventional commit format (feat/fix/refactor/test/docs)
- Summarises what changed in the subject line (under 72 chars)
- Explains WHY the change was made in the body
- Notes any breaking changes
```

Sample Claude-generated commit message:
```
feat(payments): add idempotency key support to execute_payment()

Previously, duplicate requests could result in double charges if
the client retried after a network timeout. Idempotency keys ensure
Stripe treats retries of the same logical operation as a single charge.

Key generated from order_id + request timestamp (milliseconds).
TTL matches Stripe's 24-hour idempotency window.

Breaking change: execute_payment() now requires order_id parameter.
Update all callers before deploying.
```

That is a production-quality commit message written in seconds.

### Selective Commits — Staging Specific Files

```
Commit only the changes to src/payments/processor.py.
Leave the test file unstaged for now.
```

Claude stages exactly what you specify:
```bash
git add src/payments/processor.py
git commit -m "..."
```

### Amending the Last Commit

```
The last commit message was wrong. Amend it to say:
"fix(auth): prevent session expiry for users in UTC+5 and beyond"
```

Claude runs:
```bash
git commit --amend -m "fix(auth): prevent session expiry for users in UTC+5 and beyond"
```

> ⚠️ Only amend commits that have not been pushed yet.

### Committing in Stages During Long Tasks

For long multi-file tasks, build commits in as you go:

```
We've finished the model changes. Commit just those before
we move to the API layer.
```

This is the atomic commit discipline from Chapter 5 — Claude makes it easy to commit at each logical milestone rather than one giant commit at the end.

---

## 4. Branch Operations

### Creating a Branch

```
Create a new branch called feature/user-notifications
and switch to it before we start making changes.
```

Claude runs:
```bash
git checkout -b feature/user-notifications
```

Or for more context-aware branching:

```
We're about to add the bulk discount feature.
Create an appropriately named branch for this work.
```

Claude will name it something like `feature/bulk-discount-system` based on the task context.

### Checking Current Branch and Status

```
What branch am I on and what's changed since the last commit?
```

Claude runs `git branch` and `git status` and gives you a clean summary.

### Switching Branches

```
Switch to the main branch.
```

Claude checks for uncommitted changes first and warns you before switching if they would be lost.

### Listing and Cleaning Up Branches

```
List all local branches and tell me which ones are more
than 2 weeks old and have already been merged into main.
```

Claude runs the appropriate git commands and gives you a clean list with recommendations on what is safe to delete.

---

## 5. The `/diff` Command — Your Primary Review Tool

`/diff` is the most important git-related slash command. It opens an **interactive diff viewer**:

```
/diff
```

### What It Shows

- **Current git diff** — all uncommitted changes across all files
- **Per-turn diffs** — what changed in each individual Claude action this session
- Navigation: left/right arrows switch between views, up/down browses files

### When to Use It

| Situation | Why |
|---|---|
| Before committing | Final review of everything that will be committed |
| After a multi-file task | See the full scope of what changed |
| When Claude surprises you | Check exactly what it changed vs what you asked for |
| Before creating a PR | Last sanity check before others see your code |

### The Pre-Commit Workflow

Make `/diff` a habit before every commit:

```
1. Claude finishes task
2. /diff → review all changes
3. Ask Claude about anything that looks wrong
4. Commit once satisfied
```

This two-minute habit catches scope creep, unintended side effects, and logic errors before they enter your git history.

---

## 6. Pull Requests

Claude Code has no `/pr` slash command. PRs are created through natural language — Claude uses the `gh` CLI (GitHub CLI) if it is installed.

### Creating a PR

```
Create a pull request for the current branch.
Title: "Add bulk discount system"
Target: main branch
Include a description of what changed and why.
```

Claude runs:
```bash
gh pr create \
  --title "Add bulk discount system" \
  --base main \
  --body "..."
```

And writes the PR description automatically from the commit history and changes.

### Checking PR Comments

```
/pr-comments
```

Fetches and displays comments from the GitHub PR associated with your current branch — requires the `gh` CLI.

```
/pr-comments 247
```

Fetches comments from PR #247 specifically. Useful when reviewing feedback without leaving the terminal.

### Responding to PR Feedback

```
The reviewer left a comment on the PR saying the error handling
in processor.py is too broad. Read their comment and fix the issue.

/pr-comments
```

Claude reads the PR comment, understands the feedback, and implements the fix — without you having to translate the review into a task manually.

---

## 7. Merge Conflict Resolution

Merge conflicts are one of Claude Code's most genuinely impressive capabilities. Reading conflict markers and reasoning about intent is exactly the kind of structured problem Claude handles well.

### Basic Conflict Resolution

```
I have merge conflicts. Resolve them, preserving the
intent of both branches.
```

Claude reads the conflict markers:

```python
<<<<<<< HEAD
def process_payment(amount: float, currency: str = "USD") -> dict:
    return stripe_client.charge(amount, currency)
=======
def process_payment(amount: int, idempotency_key: str) -> dict:
    return stripe_client.charge(amount, idempotency_key=idempotency_key)
>>>>>>> feature/idempotency
```

And reasons about the intent:
- HEAD changed `amount` to `float` and added `currency`
- Feature branch changed `amount` to `int` (cents) and added `idempotency_key`

Claude produces a merged version that preserves both intents:

```python
def process_payment(
    amount: int,
    currency: str = "USD",
    idempotency_key: str = None
) -> dict:
    return stripe_client.charge(
        amount,
        currency=currency,
        idempotency_key=idempotency_key
    )
```

### Conflict Resolution with Context

Give Claude the full context for better decisions:

```
We're merging feature/idempotency into main. The feature branch
changes payment amounts to integers (cents) for precision.
The main branch added multi-currency support.

Resolve all conflicts, preserving both the integer amount
type AND the currency parameter.
```

### When to Not Let Claude Resolve Conflicts

Claude is excellent at **structural conflicts** (same function modified in different ways). It is less reliable at **semantic conflicts** — where both changes are syntactically valid but logically incompatible in ways that require business knowledge.

```
This conflict involves our pricing logic and the new tax
calculation rules. Explain both sides of the conflict
before resolving — I need to verify the business logic.
```

For business-logic conflicts, always use the "explain first" pattern from Chapter 3.

---

## 8. Git History and Archaeology

Claude Code turns git history into a conversational interface. Instead of memorising git flags, just ask.

### Understanding Recent Changes

```
What's changed in the last 5 commits? Give me a plain
English summary of what each commit did.
```

Claude runs `git log` with appropriate flags and translates the history into readable prose.

### Finding When Something Broke

```
The bulk discount calculation started failing sometime in the
last 2 weeks. Look through the git log for commits that touched
src/pricing/ in that period and identify the most likely culprit.
```

Claude runs `git log --since`, reads the relevant diffs, and reasons through each commit to identify the regression.

### Blame Analysis

```
Who last modified the calculate_tax() function and what did
they change? I need to understand the history of that function.
```

Claude runs the appropriate `git log` and `git blame` commands and gives you a readable narrative of the function's evolution.

### Comparing Branches

```
What's different between main and feature/bulk-discounts?
Summarise the changes in plain English.
```

Claude runs `git diff` and `git log` between the branches and summarises both the file-level diff and the commit narrative.

---

## 9. The `/security-review` Command

A genuinely useful command for pre-commit review:

```
/security-review
```

Claude analyses all pending changes on the current branch for:
- Injection vulnerabilities (SQL, command, template)
- Authentication and authorisation gaps
- Sensitive data exposure (API keys, PII in logs)
- Insecure cryptography or hashing
- Dependency vulnerabilities in newly added packages

Run this before any PR that touches auth, payments, or data handling. It is not a replacement for a dedicated security audit but catches common issues automatically.

---

## 10. Git Workflows — Practical Patterns

### The Feature Branch Workflow

The standard pattern for using Claude Code with git:

```
Step 1: Create branch
"Create a branch for the new bulk discount feature"

Step 2: Implement in stages
[Claude implements, you review diffs]
"Commit the model changes"
[Claude commits]
[Claude continues to API layer]
"Commit the API changes"
[Claude commits]

Step 3: Review before PR
/diff
/security-review

Step 4: Create PR
"Create a PR targeting main with a description of what changed"

Step 5: Handle review feedback
/pr-comments
"Fix the issues the reviewer raised"
```

### The Hotfix Workflow

For urgent production fixes:

```
There's a critical bug in production — the payment processor
is double-charging customers on retry.

1. Create a hotfix branch from main: hotfix/prevent-double-charge
2. Read src/payments/processor.py and find the issue
3. Fix it with the minimal possible change
4. Run the payment tests
5. Commit with a message that explains the fix and impact
6. Create a PR marked as urgent
```

Claude executes the entire sequence — branch, diagnosis, fix, test, commit, PR — as one coordinated workflow.

### The Refactor Workflow

For large refactors across many commits:

```
We're refactoring UserRepository to use the new BaseRepository
pattern. Do it module by module. After each module passes tests,
commit it with a message like:
"refactor(users): migrate UserXxxRepository to BaseRepository"
Don't move to the next module until I confirm the commit.
```

---

## 11. CLAUDE.md — Encoding Your Git Conventions

Since git conventions vary by team, CLAUDE.md is where you encode yours so Claude follows them consistently without being told every session:

```markdown
## Git Conventions

Branch naming: feature/<ticket-id>-<short-description>
  e.g. feature/PAY-234-add-bulk-discounts

Commit format: conventional commits
  feat / fix / refactor / test / docs / chore
  Subject line: max 72 chars, imperative mood
  Always include ticket ID in subject

Before committing:
  1. Run pytest tests/ -x
  2. Run ruff check src/
  3. Only commit if both pass

Never commit directly to main or develop.
Always create a feature branch first.

PR description must include:
  - What changed and why
  - How to test
  - Any breaking changes
```

With this in CLAUDE.md, every commit, branch, and PR Claude creates follows your team's standards automatically — every session, without reminders.

---

## 12. Internal Mechanics — Git as a Tool Call

Understanding what happens under the hood when Claude does git operations:

```
You: "Commit these changes with a good message"
       ↓
Claude plans the sequence of git commands needed
       ↓
Tool call 1: bash("git status")
  → Returns: modified files, staged files, untracked files
       ↓
Tool call 2: bash("git diff")
  → Returns: full diff of unstaged changes
       ↓
Claude reasons about what to stage based on your request
       ↓
Tool call 3: bash("git add src/payments/processor.py")
  → Stages the file
       ↓
Claude generates a commit message from the diff content
       ↓
Tool call 4: bash('git commit -m "feat(payments): ..."')
  → Creates the commit
       ↓
Tool call 5: bash("git log --oneline -1")
  → Confirms commit was created, returns hash
       ↓
Reports the commit hash and message to you
```

### The Permission Model for Git Commands

Git operations fall into different risk tiers:

| Operation | Risk | Claude's Behaviour |
|---|---|---|
| `git status`, `git log`, `git diff` | None — read only | Runs silently |
| `git add`, `git branch`, `git checkout` | Low | Usually runs without asking |
| `git commit` | Medium | Shows the commit message first |
| `git push` | Medium-High | Usually asks for confirmation |
| `git reset --hard`, `git clean -fd` | High — destructive | Always asks explicitly |
| `git rebase -i`, `git push --force` | Very High | Requires explicit instruction + confirmation |

---

## 13. CCA Exam — Practice Questions

**Q1:** A developer wants Claude to commit only files in `src/payments/` from a session that modified files across multiple directories. What is the most reliable instruction?
- A) "Commit the changes"
- B) "Stage and commit only the files in src/payments/. Leave everything else unstaged." ✅
- C) Use `/diff` to stage files manually
- D) Run `git add` manually before asking Claude to commit

**Q2:** Your team uses conventional commits and requires a ticket ID in every message. What is the most reliable way to enforce this with Claude Code?
- A) Remind Claude before every commit
- B) Add git conventions to CLAUDE.md so Claude follows them automatically every session ✅
- C) Use a commit message template in `.git/commit_editmsg`
- D) Review every commit message and ask Claude to amend if wrong

**Q3:** Claude encounters a merge conflict in payment processing logic involving a business rule about discounts and tax. What is the best approach?
- A) Ask Claude to resolve it automatically
- B) Ask Claude to explain both sides and its intended resolution before applying, so you can verify the business logic ✅
- C) Resolve it manually without Claude
- D) Accept the incoming branch version and fix later

**Q4:** A developer wants to review all changes before creating a PR. Which command shows the full scope of uncommitted changes with per-turn navigation?
- A) Ask Claude to summarise changes
- B) `/diff` ✅
- C) `/security-review`
- D) Run `git diff` via a bash prompt

**Q5:** After Claude makes several commits during a long feature session, the developer wants to check for security issues before creating the PR. What is the correct command?
- A) Ask Claude to read each changed file for security issues
- B) `/security-review` ✅
- C) `/doctor`
- D) `/feedback`

**Q6:** A team member left comments on a GitHub PR. The developer wants Claude to read those comments and implement the fixes without leaving the terminal. What is the correct workflow?
- A) Copy-paste PR comments into the Claude Code session
- B) Run `/pr-comments` to have Claude fetch the comments, then ask Claude to implement the fixes ✅
- C) Use `/diff` to see what the reviewer saw
- D) Create a new branch and re-implement the feature

---

## 14. Practical Exercise — Full Git Workflow

Use the calculator project from earlier chapters or any git-initialised project.

**Step 1 — Baseline commit:**
```
What's the current git status of this project?
Commit any uncommitted changes with an appropriate message.
```

**Step 2 — Feature branch:**
```
Create a feature branch for adding a power function.
Name it appropriately.
```

**Step 3 — Implement and commit in stages:**
```
Add a power(base, exponent) function to calculator.py.
After adding it, commit just that change before we add tests.
```
```
Now add tests for the power function in test_calculator.py.
Run the tests, then commit if they pass.
```

**Step 4 — Review before PR:**
```
/diff
/security-review
```

**Step 5 — Encode git conventions in CLAUDE.md:**
```markdown
## Git Conventions
- Conventional commits: feat/fix/refactor/test/docs
- Subject line max 72 chars, imperative mood
- Run tests before every commit
- Branch names: feature/<description> or fix/<description>
```

Then test it:
```
Make a small improvement to the error message in the divide
function and commit it. Follow the git conventions in CLAUDE.md.
```

**Step 6 — History exploration:**
```
Show me the git log for this project in plain English.
What has changed across all commits?
```

**Step 7 — Simulate a conflict (optional):**
```
Create a second branch from main called feature/division-refactor.
Switch back to main. Make a different change to the divide function.
Commit it. Switch to feature/division-refactor and make a
conflicting change to the same function. Now merge main into
this branch and resolve the conflict.
```

---

## Chapter 7 Summary

| Concept | Key Takeaway | CCA Domain |
|---|---|---|
| No dedicated git slash commands | Git via natural language — Claude runs bash commands | Domain 3 |
| Git context at startup | Claude already knows branch, status, recent commits | Domain 3 |
| Commit discipline | Stage selectively, commit atomically, Claude writes messages | Domain 3 |
| `/diff` command | Primary review tool — use before every commit and PR | Domain 3 |
| PR workflow | Natural language + `gh` CLI — includes `/pr-comments` | Domain 3 |
| Merge conflicts | Excellent for structural; use "explain first" for business logic | Domain 1 |
| `/security-review` | Run before every PR touching auth, payments, or data | Domain 3 |
| CLAUDE.md git conventions | Encode branch naming, commit format, workflow rules | Domain 3 |
| Git archaeology | Plain English interface to log, blame, branch comparisons | Domain 3 |
| Permission tiers | Read ops silent, destructive ops always confirm | Domain 1 |

---

## ✅ Before Chapter 8

- [ ] Created a feature branch via natural language prompt
- [ ] Made Claude commit with a conventional commit message
- [ ] Used `/diff` before committing
- [ ] Used `/security-review` on a branch with changes
- [ ] Added git conventions to your CLAUDE.md
- [ ] Asked Claude to resolve a merge conflict
- [ ] Explored git history via natural language
- [ ] Answered all six practice questions correctly

---

*Previous: [Chapter 6 — Debugging & Bug Fixing](./chapter-06-debugging-and-bug-fixing.md)*
*Next: Chapter 8 — Context Window Management (coming soon)*
