# Chapter 12: CI/CD Automation

> **CCA Exam Note:** This chapter covers Domain 3 (Claude Code Configuration & Workflows, 20%). The CI/CD scenario — "Claude Code for CI/CD" — is one of the six core exam scenario contexts. Questions test the `-p` flag for non-interactive mode, permission modes in automation, the `@claude` trigger pattern, CLAUDE.md in CI, and security considerations for automated workflows.

---

## 1. The CI/CD Shift — From Interactive to Automated

Everything covered so far has been **interactive** — you sitting at a terminal, directing Claude, reviewing diffs. CI/CD flips the model:

```
Interactive:  You → Claude → You review → You approve → Done
CI/CD:        Event → Claude → Claude runs → Result committed → You review
```

The key change is **no human in the loop during execution**. Claude must operate with clearly scoped permissions, a well-defined task, and a defined stopping condition. Getting this right is what separates a useful automation from an unpredictable one.

Claude Code supports CI/CD through two complementary mechanisms:

1. **Non-interactive CLI (`-p` flag)** — run Claude as a one-shot command in any pipeline
2. **GitHub Actions / GitLab CI integrations** — purpose-built actions that connect Claude to your repo's events (PRs, issues, comments)

---

## 2. Non-Interactive Mode — The Foundation of All CI/CD

The `-p` flag is the single most important CI/CD primitive. It runs Claude with a prompt, waits for it to complete, and exits.

```bash
# Basic usage
claude -p "your prompt here"

# Real examples
claude -p "Review this PR for security issues and post findings as comments"
claude -p "Fix all failing tests in the test suite"
claude -p "Generate release notes from commits since the last tag"
claude -p "Run the linter and fix all auto-fixable issues"
```

### Key flags for CI/CD use

```bash
# Limit turns to prevent runaway execution
claude -p "task" --max-turns 10

# Set the model explicitly
claude -p "task" --model claude-sonnet-4-6

# Set permission mode (no interactive prompts)
claude -p "task" --permission-mode acceptEdits

# Restrict tools to only what's needed
claude -p "task" --allowedTools "Read,Edit,Bash(npm test)"

# Structured JSON output for scripting
claude -p "task" --output-format json

# Streaming JSON for real-time processing
claude -p "task" --output-format stream-json

# Append to system prompt (CI-specific instructions)
claude -p "task" --append-system-prompt "You are running in CI. Be concise."
```

> 🔑 **CCA Exam Critical:** In non-interactive mode with `-p`, if auto mode is enabled and the classifier blocks actions 3 times in a row, the session **aborts** — there is no user to fall back to. Design prompts and permissions to avoid triggering classifier blocks.

### Piping data into Claude

```bash
# Pipe test output for analysis
npm test 2>&1 | claude -p "Identify the root cause of these failures and suggest fixes"

# Pipe a diff for review
git diff main..feature | claude -p "Review this diff for issues"

# Pipe log files
cat error.log | claude -p "Summarise the errors and identify patterns"
```

---

## 3. GitHub Actions Integration

### Quick Setup

The fastest path is running `/install-github-app` inside a Claude Code session. This guides you through installing the GitHub App and configuring secrets. Requires repository admin access.

For manual setup:
1. Install the Claude GitHub App: `github.com/apps/claude`
2. Add `ANTHROPIC_API_KEY` to repository secrets
3. Copy the workflow file from the examples directory into `.github/workflows/`

### The Basic Workflow — `@claude` Trigger Pattern

```yaml
name: Claude Code
on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]

jobs:
  claude:
    runs-on: ubuntu-latest
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          # No prompt needed — responds to @claude mentions automatically
```

With this workflow, anyone can trigger Claude in any PR or issue comment:

```
@claude implement this feature based on the issue description
@claude fix the TypeError in the user dashboard component
@claude review this PR for security issues
@claude how should I implement caching for this endpoint?
```

Claude analyses the context, implements changes, and creates a PR — all without leaving GitHub.

### Action Parameters (v1.0)

| Parameter | Description | Required |
|---|---|---|
| `anthropic_api_key` | Claude API key | Yes (unless using Bedrock/Vertex) |
| `prompt` | Instructions for Claude | No — when omitted, responds to `@claude` |
| `claude_args` | Any Claude CLI argument | No |
| `github_token` | GitHub token for API access | No |
| `trigger_phrase` | Custom trigger (default: `@claude`) | No |
| `use_bedrock` | Route through AWS Bedrock | No |
| `use_vertex` | Route through Google Vertex AI | No |

All CLI arguments pass through `claude_args`:

```yaml
claude_args: |
  --max-turns 10
  --model claude-sonnet-4-6
  --allowedTools "Read,Edit,Bash(npm test)"
```

### Automated PR Review on Every Push

No trigger phrase needed — runs automatically on every PR:

```yaml
name: Automated Code Review
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            Review this pull request for code quality, correctness, and security.
            Analyse the diff and post your findings as review comments.
          claude_args: "--max-turns 5"
```

### Scheduled Automation

```yaml
name: Daily Report
on:
  schedule:
    - cron: "0 9 * * *"   # 9am UTC daily

jobs:
  report:
    runs-on: ubuntu-latest
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: "Generate a summary of yesterday's commits and open issues"
          claude_args: "--model opus"
```

### Auto-Detection of Mode

The v1 action automatically detects whether to run in:
- **Interactive mode** — when no `prompt` is provided, waits for `@claude` mentions
- **Automation mode** — when `prompt` is provided, runs immediately

No `mode` parameter needed (that was a beta-only field, now removed).

---

## 4. GitLab CI/CD Integration

> Note: The GitLab integration is currently in beta and maintained by GitLab.

### Basic `.gitlab-ci.yml`

```yaml
stages:
  - ai

claude:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "web"'
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  variables:
    GIT_STRATEGY: fetch
  before_script:
    - apk update
    - apk add --no-cache git curl bash
    - curl -fsSL https://claude.ai/install.sh | bash
  script:
    - /bin/gitlab-mcp-server || true
    - >
      claude
      -p "${AI_FLOW_INPUT:-'Review this MR and implement requested changes'}"
      --permission-mode acceptEdits
      --allowedTools "Bash Read Edit Write mcp__gitlab"
  # Claude Code reads ANTHROPIC_API_KEY from CI/CD variables
```

The `AI_FLOW_INPUT` variable carries the trigger context (comment content, issue description, etc.) when the pipeline is triggered by an event listener watching for `@claude` mentions.

### GitLab `@claude` Pattern

GitLab doesn't have native comment webhooks as simple as GitHub. The pattern is:
1. A webhook or note event listener detects `@claude` in a comment
2. It triggers the GitLab pipeline via API, passing the comment as `AI_FLOW_INPUT`
3. The pipeline runs Claude with that input

```
# In a GitLab issue or MR comment:
@claude implement this feature based on the issue description
@claude fix the failing tests in the auth module
@claude suggest an approach to optimise these database queries
```

---

## 5. Permission Modes in CI/CD

Choosing the right permission mode is critical for automated workflows. This is directly tested on the CCA exam.

### The Permission Mode Decision

```
Do you need interactive approval?
    YES → Default mode (not suitable for CI — blocks)
    NO  → Choose based on risk:

Low risk, trusted code, just file edits:
    → acceptEdits (Claude edits files without asking)

Need shell commands too:
    → auto mode (classifier reviews commands automatically)

Fully trusted, isolated container:
    → bypassPermissions (skip all checks — only in containers/VMs)

Pre-defined allowed tools only:
    → dontAsk (auto-deny anything not in allowlist)
```

### `acceptEdits` — Most Common for CI

```yaml
claude_args: "--permission-mode acceptEdits --allowedTools 'Read,Edit,Write,Bash(npm test),Bash(git commit)'"
```

Claude edits files without asking. You still need `allowedTools` to specify which bash commands are permitted.

### `auto` Mode — Safer for Complex Tasks

```bash
claude -p "fix all lint errors" --permission-mode auto
```

A background classifier reviews each action. Blocks scope escalation and risky operations automatically. Adds latency and token cost per action.

> ⚠️ In auto mode with `-p`, if the classifier blocks the same action 3 times consecutively or 20 times total, the session aborts. Design your prompts to stay within expected scope.

### `bypassPermissions` — Isolated Environments Only

```bash
# Only use in sandboxed containers
claude -p "refactor the auth module" --permission-mode bypassPermissions
```

No checks at all. Only safe in completely isolated environments (Docker containers, VMs, devcontainers without internet access).

### `dontAsk` — Locked-Down Pipelines

```yaml
# Pre-approve only safe commands, block everything else
claude_args: |
  --permission-mode dontAsk
  --allowedTools "Read,Bash(npm test),Bash(npm run lint)"
```

Auto-denies any tool not in the explicit allowlist. Good for read-only analysis or strictly controlled operations.

---

## 6. Enterprise Deployments — Bedrock and Vertex

For teams with data residency requirements or existing cloud agreements:

### AWS Bedrock (GitHub Actions)

```yaml
- name: Configure AWS Credentials (OIDC)
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
    aws-region: us-west-2

- uses: anthropics/claude-code-action@v1
  with:
    github_token: ${{ steps.app-token.outputs.token }}
    use_bedrock: "true"
    claude_args: '--model us.anthropic.claude-sonnet-4-6 --max-turns 10'
```

> 🔑 **Bedrock model IDs include a region prefix** — e.g., `us.anthropic.claude-sonnet-4-6`, not `claude-sonnet-4-6`. This is a common misconfiguration.

Use OIDC (not static keys) — credentials are temporary and auto-rotated.

### Google Vertex AI (GitHub Actions)

```yaml
- name: Authenticate to Google Cloud
  uses: google-github-actions/auth@v2
  with:
    workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
    service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

- uses: anthropics/claude-code-action@v1
  with:
    github_token: ${{ steps.app-token.outputs.token }}
    use_vertex: "true"
    claude_args: '--model claude-sonnet-4-5@20250929 --max-turns 10'
  env:
    ANTHROPIC_VERTEX_PROJECT_ID: ${{ steps.auth.outputs.project_id }}
    CLOUD_ML_REGION: us-east5
```

Use Workload Identity Federation (not service account keys) — no downloaded credentials needed.

---

## 7. CLAUDE.md in CI/CD

CLAUDE.md is as important in CI/CD as in interactive use. In a CI pipeline, Claude has no human to ask for clarification — your CLAUDE.md becomes the primary source of project knowledge.

### CI-Specific CLAUDE.md Section

```markdown
## CI/CD Behaviour

When running in CI (non-interactive mode):
- Be concise in commit messages and PR descriptions
- Run tests before committing: `npm test`
- Run linter before committing: `npm run lint`
- Only commit to feature branches, never to main
- PR titles must follow: `type(scope): description`
- If tests fail after a fix, stop and report — do not loop endlessly

## Code Review Standards
- Flag all TODO comments added in this PR
- Security: check for hardcoded secrets, SQL injection, XSS
- Test coverage: every new function must have a test
```

Without CI-specific guidance, Claude might create PRs that fail your linting, produce verbose commit messages, or attempt to push to protected branches.

---

## 8. Cost Management in CI/CD

Two cost categories apply:

**Runner costs:** Claude runs on GitHub Actions runners (GitHub-hosted = consume minutes) or your GitLab runners (compute time).

**API costs:** Each interaction consumes tokens based on prompt and response length. A complex PR review on a large codebase can consume tens of thousands of tokens.

### Cost Control Strategies

```yaml
# Limit turns — most important lever
claude_args: "--max-turns 5"

# Use Sonnet for most tasks, Opus only when needed
claude_args: "--model claude-sonnet-4-6"

# Restrict to specific events to avoid unnecessary runs
on:
  pull_request:
    types: [opened]          # Only on new PRs, not every push
    paths:
      - 'src/**'             # Only when source files change
      - '!**/*.md'           # Not for documentation changes

# Add a concurrency limit
concurrency:
  group: claude-${{ github.ref }}
  cancel-in-progress: true  # Cancel previous run if new one starts
```

---

## 9. Security in CI/CD Pipelines

Security is a distinct focus area in CI/CD — and on the CCA exam.

### The Golden Rules

**Never commit API keys.** Always use secrets:
```yaml
anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}  # ✅
anthropic_api_key: "sk-ant-..."                        # ❌ Never do this
```

**Use OIDC for cloud providers** — temporary credentials that rotate automatically, no static keys stored anywhere.

**Least-privilege permissions.** Grant only what the workflow needs:
```yaml
permissions:
  contents: write        # Only if Claude commits code
  pull-requests: write   # Only if Claude creates PRs
  issues: write          # Only if Claude comments on issues
  id-token: write        # Required for OIDC authentication
```

**Restrict tools explicitly:**
```yaml
claude_args: "--allowedTools 'Read,Bash(npm test),Bash(npm run lint)'"
# Claude cannot write files, run arbitrary commands, or access external services
```

**Review AI-generated code like any other.** Branch protection rules and required reviews should apply to Claude's PRs. Never auto-merge Claude's output without human review.

### The `--permission-mode acceptEdits` + `--allowedTools` Pattern

This is the recommended pattern for most CI workflows:

```yaml
claude_args: |
  --permission-mode acceptEdits
  --allowedTools "Read,Edit,Write,Bash(npm test),Bash(npm run lint),Bash(git add),Bash(git commit),Bash(git push)"
  --max-turns 10
```

- `acceptEdits` — no prompts for file edits
- Explicit `Bash(...)` allowlist — only named commands can run
- `--max-turns 10` — bounded execution

---

## 10. Internal Mechanics — How CI/CD Claude Runs

Understanding what happens when Claude runs in CI helps you design reliable workflows:

```
Pipeline triggered (PR opened, @claude comment, schedule, etc.)
       ↓
Job starts on runner
       ↓
Claude Code installed (if not pre-installed in image)
       ↓
Session starts with:
  - CLAUDE.md loaded (from repo root)
  - Repo checked out
  - No auto memory (fresh environment each run)
  - No interactive terminal
       ↓
claude -p "prompt" runs:
  - Claude reads relevant files
  - Executes within allowed tools
  - Cannot prompt user for approval
  - Makes changes or posts comments
       ↓
Claude exits (max-turns reached or task complete)
       ↓
Changes committed/pushed if permitted
PR created if configured
```

### Key Differences From Interactive Mode

| Aspect | Interactive | CI/CD |
|---|---|---|
| Context | Built up over session | Fresh every run |
| CLAUDE.md | Loaded at session start | Loaded from repo |
| Auto memory | Persists across sessions | Not available (fresh env) |
| Permission prompts | You respond | Must be pre-configured |
| Mistakes | `/rewind` to undo | Git revert or new PR |
| Cost visibility | `/cost` command | GitHub Actions logs + API billing |

---

## 11. CCA Exam — Practice Questions

**Q1:** A CI workflow uses `claude -p "fix failing tests"` without specifying a permission mode. Claude needs to run tests and edit files. What is the most likely problem?

- A) Claude will fail because `-p` requires a model flag
- B) Claude will prompt for approval on file edits, blocking the CI job indefinitely ✅
- C) Claude will refuse to run without CLAUDE.md
- D) Claude will run with no restrictions

**Q2:** A GitHub Actions workflow responds to `@claude` comments. A developer mentions `@claude implement the OAuth feature from issue #234`. What does Claude do?

- A) Opens a browser to view the issue
- B) Analyses the issue context and codebase, implements changes, and creates a PR ✅
- C) Posts a comment asking for more details
- D) Runs only if the comment is on a PR, not an issue

**Q3:** A team uses AWS Bedrock for CI/CD to meet data residency requirements. The workflow fails with model not found. What is the most likely cause?

- A) Bedrock requires a different action version
- B) The model ID uses a region prefix (e.g., `us.anthropic.claude-sonnet-4-6`) that was not specified ✅
- C) Bedrock does not support the `-p` flag
- D) OIDC credentials expired

**Q4:** A CI pipeline using auto mode keeps aborting after Claude runs for a few turns. What is the most likely cause and fix?

- A) The model is too slow — switch to Haiku
- B) The classifier is blocking actions, likely hitting the 3 consecutive block threshold — scope the prompt more precisely and add trusted actions to the allowlist ✅
- C) `--max-turns` is too low — increase it
- D) Auto mode is incompatible with `-p` flag

**Q5:** A developer wants Claude to automatically review every PR but only when Python files change. Which workflow configuration is correct?

- A) Use `@claude` trigger in PR description template
- B) Set `on: push` with no path filters
- C) Set `on: pull_request` with `paths: ['**/*.py']` and use the `prompt` parameter ✅
- D) Run on a schedule every hour

**Q6:** What is the most secure way to configure Claude Code GitHub Actions for an enterprise team using their own cloud provider?

- A) Store API keys as environment variables in the workflow file
- B) Use OIDC/Workload Identity Federation — temporary credentials, no static keys, least-privilege IAM roles ✅
- C) Store credentials in CLAUDE.md
- D) Use `bypassPermissions` mode to avoid authentication issues

---

## 12. Practical Exercise — CI/CD Setup

### Exercise A — Non-Interactive Mode

```bash
# Create a test repo
mkdir ci-practice && cd ci-practice && git init
echo "def divide(a, b): return a/b" > math.py
echo "def test_divide(): assert divide(10, 2) == 5" > test_math.py

# Run Claude non-interactively
export ANTHROPIC_API_KEY="your-key"
claude -p "Read math.py and test_math.py. Find the missing edge case in the tests and add it." \
  --permission-mode acceptEdits \
  --allowedTools "Read,Edit" \
  --max-turns 5
```

Verify Claude added a zero-division test without prompting for approval.

### Exercise B — GitHub Actions

Create `.github/workflows/claude.yml`:

```yaml
name: Claude Code
on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]

jobs:
  claude:
    if: contains(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
```

Push to a GitHub repo, create a PR, and comment `@claude review this PR for any issues`.

### Exercise C — Automated Review

```yaml
name: PR Review
on:
  pull_request:
    types: [opened]
    paths: ['src/**']

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            Review this PR for:
            1. Code quality issues
            2. Missing tests
            3. Security concerns
            Post a summary comment on the PR.
          claude_args: "--max-turns 3 --model claude-sonnet-4-6"
```

### Exercise D — CLAUDE.md for CI

Add a CI section to your project's CLAUDE.md:
```markdown
## CI/CD Behaviour
- Always run `npm test` before committing
- Use conventional commits: feat/fix/chore/docs
- Never commit to main branch
- If tests fail after changes, stop and report
```

Trigger a CI workflow and verify Claude follows these rules.

---

## Chapter 12 Summary

| Concept | Key Takeaway | CCA Domain |
|---|---|---|
| `-p` flag | Core primitive for all CI/CD — non-interactive, one-shot execution | Domain 3 |
| `@claude` trigger | GitHub/GitLab comment trigger — Claude analyses context and acts | Domain 3 |
| Permission modes in CI | `acceptEdits` most common; `auto` for complex; `bypassPermissions` containers only | Domain 3 |
| `--max-turns` | Essential guard — bounds execution in automated pipelines | Domain 3 |
| `--allowedTools` | Explicit tool allowlist — least-privilege for CI | Domain 3 |
| Bedrock model IDs | Include region prefix: `us.anthropic.claude-sonnet-4-6` | Domain 3 |
| OIDC over static keys | Temporary credentials, auto-rotated — required for enterprise | Domain 3 |
| CLAUDE.md in CI | Primary source of project knowledge — must include CI-specific rules | Domain 3 |
| Auto mode abort | `-p` + auto mode aborts if classifier blocks 3 consecutive or 20 total actions | Domain 5 |
| Security rules | Never commit keys; OIDC; least-privilege; review AI PRs like human PRs | Domain 6 |

### CCA Domain Coverage

| CCA Domain | Concepts Covered |
|---|---|
| **Domain 3: Claude Code Config & Workflows (20%)** | Non-interactive mode, GitHub Actions, GitLab CI, permission modes, CLAUDE.md in CI, cost management |
| **Domain 5: Context Management & Reliability (15%)** | Auto mode fallback behaviour, fresh context per run, no auto memory |
| **Domain 6: Security (5%)** | Secrets management, OIDC, least-privilege, code review of AI output |

---

## ✅ Before Chapter 13

- [ ] Run `claude -p` non-interactively on a local task with `--permission-mode acceptEdits`
- [ ] Set up the basic GitHub Actions `@claude` workflow
- [ ] Triggered Claude with an `@claude` comment on a PR
- [ ] Added CI-specific rules to your CLAUDE.md
- [ ] Understood the auto mode abort threshold for non-interactive runs
- [ ] Answered all six practice questions correctly

---

*Previous: [Chapter 11 — Checkpoints & Session Recovery](./chapter-11-checkpoints-and-session-recovery.md)*
*Next: Chapter 13 — The Claude Agent SDK (coming soon)*
