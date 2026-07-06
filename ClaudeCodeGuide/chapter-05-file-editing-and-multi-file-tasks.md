# Chapter 5: File Editing & Multi-File Tasks

> **CCA Exam Note:** This chapter covers Domain 1 (Agentic Architecture & Orchestration, 27%) and Domain 4 (Prompt Engineering & Structured Output, 20%) — the two heaviest domains on the exam. The read-edit-verify loop and task decomposition patterns tested here appear in multiple scenario contexts.

---

## 1. How Claude Code Approaches File Editing

There's a fundamental difference between how a human edits code and how Claude Code does it.

A human opens a file, finds the right place, makes a change, saves. Linear, manual, one file at a time.

Claude Code approaches editing more like a **surgeon reviewing scans before operating**:

```
1. UNDERSTAND — Read all relevant files before touching anything
2. PLAN       — Determine what needs to change and in what order
3. SEQUENCE   — Order changes to avoid breaking intermediate states
4. EXECUTE    — Make changes one file at a time, showing diffs
5. VERIFY     — Run tests/linters to confirm nothing broke
```

This is why Claude Code often reads more files than you expect before making a single edit. It's not being slow — it's building the complete picture so its edits are correct the first time.

---

## 2. The Read-Edit-Verify Loop — Internal Mechanics

This is the core engine behind every file editing task. Understanding it makes you a dramatically better director of Claude Code.

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Task arrives: "Add rate limiting to login"         │
│          ↓                                          │
│  READ PHASE                                         │
│  ├── Read: src/api/auth.py          (target file)  │
│  ├── Read: src/middleware/           (patterns)     │
│  ├── Read: src/core/config.py        (settings)     │
│  └── Search: "rate_limit" in codebase               │
│          ↓                                          │
│  PLAN PHASE                                         │
│  ├── Identify: where to insert middleware           │
│  ├── Check: existing rate limit utils?              │
│  └── Determine: which files need changes            │
│          ↓                                          │
│  EDIT PHASE (each edit shown as diff)               │
│  ├── Edit: src/middleware/rate_limit.py (new file)  │
│  ├── Edit: src/api/auth.py           (apply it)     │
│  └── Edit: src/core/config.py        (add settings) │
│          ↓                                          │
│  VERIFY PHASE                                       │
│  ├── Run: pytest tests/test_auth.py                 │
│  ├── Run: ruff check src/                           │
│  └── Report: all passing / failures found           │
│          ↓                                          │
│  Done — responds to you with summary                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### What Can Go Wrong At Each Phase

| Phase | Common Failure | Your Response |
|---|---|---|
| Read | Claude reads wrong files, misses key context | Use `@` to point at specific files |
| Plan | Claude over-engineers or under-scopes | Ask it to explain the plan before acting |
| Edit | Claude modifies a file you didn't intend | Reject the diff with `n` at the prompt |
| Verify | Tests fail, Claude attempts auto-fix in a loop | Press `Escape`, review the failure yourself |

---

## 3. Single File Editing — The Fundamentals

Before tackling multi-file tasks, master single-file editing. This is where you develop the judgment to read diffs correctly.

### Requesting a Targeted Edit

Always be as specific as possible about scope:

```
# Too vague — Claude might rewrite the whole file
"Improve the login function"

# Specific — Claude knows exactly what to change
"In src/auth/login.py, add a check at line 45 that raises
AuthError if the user account is_active is False.
Don't change anything else in the file."
```

### Reading the Diff — Your Quality Gate

Every edit Claude proposes shows a diff:

```
  Editing: src/auth/login.py
  ┌────────────────────────────────────────────────┐
  │  def authenticate(username, password):          │
  │      user = User.query.filter_by(               │
  │          username=username).first()             │
  │                                                 │
  │ -    if not user or not user.check_password(    │
  │ -        password):                             │
  │ -        raise AuthError("Invalid credentials") │
  │                                                 │
  │ +    if not user:                               │
  │ +        raise AuthError("User not found")      │
  │ +    if not user.is_active:                     │
  │ +        raise AuthError("Account disabled")    │
  │ +    if not user.check_password(password):      │
  │ +        raise AuthError("Invalid password")    │
  │                                                 │
  │      return generate_token(user)                │
  └────────────────────────────────────────────────┘
  Apply this change? [y/n/e]
```

**What to check in every diff:**
- Are the removed lines (`-`) exactly what you expected to be removed?
- Do the added lines (`+`) do exactly what you asked?
- Is any surrounding context (shown without `+/-`) correct and untouched?
- Did Claude change anything you didn't ask about?

**The `e` option** — open in editor before applying. Use this when:
- The change is mostly right but needs a small tweak
- You want to adjust variable names or add a comment
- You disagree with one line but want the rest

### Rejecting Part of a Multi-Edit Task

When Claude is making several changes and one diff is wrong:
- Press `n` on that specific diff
- Claude notes it was rejected and continues with the rest
- After the task completes, redirect: "The third edit was wrong — here's what I actually need..."

---

## 4. Multi-File Tasks — The Right Mental Model

Multi-file tasks are where Claude Code becomes genuinely powerful. But they're also where things most often go sideways. The key is understanding **task decomposition**.

### What Is Task Decomposition?

When you give Claude a large task, it must break it into an ordered sequence of subtasks. The quality of this decomposition determines whether the result is good or messy.

**Poor decomposition (Claude left to its own devices on a vague task):**
```
You: "Refactor the user system"

Claude's decomposition:
1. Rename all User methods (done)
2. Update API layer (done)
3. Realise it broke the auth module
4. Fix auth module
5. Realise it broke tests
6. Fix tests
7. Realise config references old method names
... [spirals]
```

**Good decomposition (you guide it explicitly):**
```
You: "Refactor the user system. Do it in this exact order:
1. First, audit all files that reference User — list them, don't change anything
2. Then update the User model
3. Then update each dependent file one at a time
4. Then run tests after each file change
5. Stop and report if any test fails"
```

Same task. Completely different outcome.

### The CCA Exam Mental Model

> **Task decomposition is explicitly tested in Domain 1.** The exam consistently rewards candidates who choose explicit, ordered decomposition over letting the agent decide its own execution sequence.

The principle: **the more irreversible or interconnected the changes, the more you should control the sequence.**

---

## 5. Multi-File Editing Patterns

These are the patterns that work reliably in production. Each one is tested in the CCA exam.

### Pattern 1: Audit First, Edit Second

For any task touching more than 3 files, always start with an audit:

```
Before making any changes, find every file in the codebase
that references the UserProfile class. List them with the
line numbers where they appear. Don't edit anything yet.
```

Claude returns a map of the blast radius. You review it, decide if the scope is right, then proceed. This prevents surprises mid-task.

### Pattern 2: One File at a Time with Confirmation

For high-risk changes:

```
Update the API to use the new UserProfile schema.
Do it one file at a time. After each file, show me the
diff and wait for my approval before moving to the next.
```

Slower but zero surprises. Every change is intentional.

### Pattern 3: Dependency Order Enforcement

Explicitly tell Claude the correct order when you know it:

```
We're renaming the `process_payment` method to `execute_payment`.
Do the changes in this exact order:
1. Update the method definition in src/payments/processor.py
2. Update the call in src/api/checkout.py
3. Update the call in src/workers/subscription.py
4. Update the tests in tests/test_payments.py

Do not proceed to the next file until the current one compiles.
```

This prevents the common failure mode of updating callers before updating the definition, creating a broken intermediate state.

### Pattern 4: Test-Driven Multi-File Editing

Write the test first, then let Claude implement to make it pass:

```
First, write a failing test in tests/test_notifications.py
that verifies email notifications are sent when an invoice
is marked as paid.

Then implement the feature across whatever files are needed
to make that test pass. Run the test after each change.
```

This gives Claude a clear, verifiable definition of "done" — it can't drift because the test is the target.

### Pattern 5: Parallel Read, Sequential Write

For exploration followed by editing:

```
Read these three files and understand how they connect:
@src/models/invoice.py
@src/services/invoice_service.py
@src/api/invoice_router.py

Then explain the data flow. Don't edit anything yet.
[Review Claude's explanation]
Now add a status_history field to the Invoice model and
propagate it through the service and API layers.
```

You use the read phase to verify Claude understands the system before it touches anything.

---

## 6. Controlling Scope — Keeping Claude Focused

The most common multi-file problem: Claude helpfully refactors things you didn't ask it to refactor.

### Hard Scope Boundaries

```
Only modify files in src/payments/.
Do not touch src/api/ or any test files.
If you need something from outside src/payments/,
tell me and I'll decide whether to include it.
```

### Soft Scope Guidance

```
Focus on the core task. If you notice other issues
while working, note them in a list at the end but
don't fix them unless I ask.
```

This pattern is excellent — Claude still surfaces problems it finds, but doesn't go rogue fixing everything. You get the information without the scope creep.

### The "Touch List" Technique

Ask Claude to declare its intended changes before making them:

```
Before starting, list every file you intend to modify
and what change you'll make to each. Wait for my
confirmation before proceeding.
```

Sample Claude response:
```
Planned changes:
1. src/models/user.py — add `last_login` field to User model
2. migrations/ — create new Alembic migration for the field
3. src/services/auth_service.py — update login() to set last_login
4. src/schemas/user.py — add last_login to UserResponse schema
5. tests/test_auth.py — add test for last_login being updated

Shall I proceed?
```

You can now spot if Claude is missing a file, or planning to touch something it shouldn't. This is the most reliable way to control multi-file scope before a single character is written.

---

## 7. Large-Scale Refactoring — Special Considerations

Refactoring is the highest-risk multi-file operation. Specific guidance:

### Break Into Atomic Commits

```
We're migrating from synchronous to async database calls.
Don't do this all at once. Do it one module at a time,
and after each module is complete and tests pass,
tell me so I can commit before we continue.
```

This gives you git checkpoints throughout the refactor. If something goes wrong, you revert one commit, not the entire refactor.

### The Strangler Fig Pattern with Claude

For large rewrites, don't replace everything at once:

```
We're moving from the old UserRepository pattern to the new
RepositoryBase pattern. Start with the least-used repository
first — UserPreferenceRepository. Migrate just that one.
Keep the old code alongside the new code until we verify
the new version works. Then we'll migrate the next one.
```

Claude can implement this pattern reliably if you describe it explicitly. Don't assume it will choose this safe approach on its own.

### Tracking Progress on Long Refactors

```
/tasks
```

Lists active background tasks. For tracking what's been completed in a long refactor, ask Claude directly: "What have we done so far and what remains?" then use `/compact` to compress history before continuing — this prevents context degradation from sabotaging the later stages of a long refactor.

---

## 8. File Creation vs File Editing

Knowing when Claude should create a new file vs edit an existing one matters:

### When Claude Should Create a New File

```
Create a new service class src/services/notification_service.py
following the same structure as src/services/email_service.py.
```

Explicitly saying "create a new file" prevents Claude from cramming new code into an existing file where it doesn't belong.

### When Claude Should Edit an Existing File

```
Add the send_invoice_notification method to the existing
NotificationService in src/services/notification_service.py.
Don't create a new file.
```

The "don't create a new file" constraint is worth adding when the task is an extension, not a new component.

### The New File Checklist

When Claude creates a new file, verify:
- [ ] Filename follows your project's naming convention
- [ ] File is in the right directory
- [ ] Imports follow your project's import style
- [ ] The file was added to any relevant `__init__.py` or module registry
- [ ] A corresponding test file was created (if your convention requires it)

---

## 9. Working With Generated Code — The Verification Mindset

> ⚠️ **CCA Exam Critical:** The exam tests whether you understand that Claude-generated code requires systematic verification, not just a quick read.

Claude Code can generate correct-looking code that has subtle logical errors. The verification mindset means you treat every generated file as a code review task, not a finished product.

### Three Levels of Verification

**Level 1: Diff Review (Always)**
Read every diff before approving. Check for:
- Logic errors in conditionals
- Off-by-one errors
- Missing error handling
- Hardcoded values that should be configurable

**Level 2: Test Execution (After Each Edit)**
```
After making the change, run:
pytest tests/test_payments.py -v
```

Tell Claude which tests to run. Don't let it decide — it may run a broader or narrower suite than you need.

**Level 3: Manual Walkthrough (Before Committing)**

Ask Claude directly to self-audit its work:
```
Walk me through every change you made this session.
Explain the logic of each edit and flag anything you're uncertain about.
```

Claude re-reads its own changes and often catches things it missed initially — it approaches the code fresh in the review pass.

### Asking Claude to Self-Audit After Multi-File Sessions

After a multi-file session, ask Claude to review its own work. It reads all changed files and gives a critical summary — often flagging edge cases or uncertainties it didn't surface during the original task:

```
Changes made this session:

1. src/payments/processor.py — Added idempotency key support
   to execute_payment(). Key generated from order_id + timestamp.
   ⚠️ Note: If the same order is submitted within 1 second,
   keys could collide. Consider using a UUID instead.

2. src/api/checkout.py — Updated to pass idempotency_key.
   Change looks correct.

3. tests/test_payments.py — Added 3 tests. All passing.
   Edge case not covered: concurrent requests.
```

That `⚠️ Note` is Claude catching something during its review pass that it missed initially. This self-audit step after every multi-file session is non-negotiable.

---

## 10. Internal Mechanics — How Multi-File Context Works

When Claude executes a multi-file task, here's exactly what's happening in the context window:

```
Initial context:
  [System prompt + CLAUDE.md]          ~2,000 tokens
  [Your task prompt]                     ~200 tokens

After reading 5 files (500 lines each):
  [System prompt + CLAUDE.md]          ~2,000 tokens
  [file 1 contents]                    ~3,000 tokens
  [file 2 contents]                    ~3,000 tokens
  [file 3 contents]                    ~3,000 tokens
  [file 4 contents]                    ~3,000 tokens
  [file 5 contents]                    ~3,000 tokens
  [Your task prompt]                     ~200 tokens
  Total so far:                       ~17,200 tokens

After 3 edits with diffs + test output:
  + [edit 1 diff + approval]           ~1,000 tokens
  + [edit 2 diff + approval]           ~1,000 tokens
  + [edit 3 diff + approval]           ~1,000 tokens
  + [pytest output]                    ~2,000 tokens
  Total:                              ~22,200 tokens
```

On a 200,000-token context window (Pro plan), this seems fine. But on long refactors spanning many files, it accumulates fast.

### Context Degradation in Long Tasks

As the context window fills, Claude's performance on the **later** parts of a long task degrades. It starts losing precision on files it read early in the session.

**Signs of context degradation:**
- Claude forgets a constraint you set 30 messages ago
- Claude repeats an edit it already made
- Claude references a file incorrectly
- Responses become noticeably less precise

**Your response:**
```
/compact
```

Then re-state the key constraints and continue. `/compact` compresses history — you lose some detail but gain back context headroom. Better to compact and continue cleanly than push through with a degraded context.

### The Optimal Session Structure for Large Tasks

```
Session 1: Audit + Plan
  - Read all relevant files
  - Generate the touch list
  - Commit the plan to CLAUDE.md or a task file
  /compact or end session

Session 2: Implement Module A
  - Reference the plan
  - Edit, verify, commit
  /compact or end session

Session 3: Implement Module B
  - Reference the plan
  - Edit, verify, commit
```

Breaking large tasks across sessions is not a limitation — it's professional practice. Each session starts with a clean context. The plan persists in CLAUDE.md.

---

## 11. CCA Exam — Practice Questions

**Q1:** You ask Claude to refactor a payment module that affects 12 files. Claude begins editing immediately without reading all the files first. What is the most effective intervention?
- A) Let Claude proceed — it has enough training to handle this
- B) Press Escape, then ask Claude to audit all affected files and list planned changes before editing ✅
- C) Use `/clear` and start over with a smaller task
- D) Switch to a more capable model

**Q2:** Claude is halfway through a 15-file refactor when you notice it modified a file outside the agreed scope. What is the correct response?
- A) `Escape × 2` to undo (if just happened) or `/rewind` to checkpoint before scope violation ✅
- B) Continue and fix it at the end
- C) Use `/clear` and restart the entire refactor
- D) Accept the change and update your scope

**Q3:** After a multi-file session, you want to ensure Claude didn't introduce subtle bugs. What is the most systematic approach?
- A) Read all changed files manually
- B) Run the test suite and trust the results
- C) Ask Claude to walk through every change it made and flag uncertainties, then run tests, then examine any flagged areas ✅
- D) Ask Claude if it made any mistakes

**Q4:** Which prompt structure is most likely to produce a correct multi-file result on the first attempt?
- A) "Add real-time notifications to the app"
- B) "Add real-time notifications. Use WebSockets. Update the backend and frontend."
- C) "Audit files that need changing for WebSocket notifications, list them, wait for approval, then implement one file at a time with tests after each change" ✅
- D) "Add WebSocket notifications following best practices"

**Q5:** During a multi-file task, Claude's responses become less precise and it forgets a constraint you set earlier. What is the most likely cause and correct fix?
- A) Claude is hallucinating — switch models
- B) Context window is filling up — use `/compact` then restate key constraints ✅
- C) CLAUDE.md is conflicting — run `/clear`
- D) The task is too complex — break into smaller tasks

---

## 12. Practical Exercise — Multi-File Feature Build

Build a complete feature across multiple files from scratch.

**Setup:**
```bash
mkdir multi-file-practice && cd multi-file-practice && git init
```

Create three starter files:

`models.py`:
```python
class Product:
    def __init__(self, name: str, price: float, stock: int):
        self.name = name
        self.price = price
        self.stock = stock
```

`service.py`:
```python
from models import Product

products = []

def add_product(name: str, price: float, stock: int):
    product = Product(name, price, stock)
    products.append(product)
    return product
```

`api.py`:
```python
from service import add_product

def handle_add_product(data: dict):
    return add_product(data['name'], data['price'], data['stock'])
```

**Step 1 — Audit first:**
```
Before making any changes, read models.py, service.py, and api.py.
Then list every change you'd need to make to add a "purchase product"
feature that:
- Reduces stock when a product is purchased
- Raises an error if stock is 0
- Returns the updated product

List the files and changes. Don't edit anything yet.
```

**Step 2 — Execute with approval gates:**
```
That plan looks correct. Implement it one file at a time.
Start with models.py, show me the diff, wait for my approval,
then move to service.py, then api.py.
```

**Step 3 — Test coverage:**
```
Now create tests/test_purchase.py with tests covering:
- Successful purchase reduces stock by 1
- Purchasing when stock is 0 raises an error
- Purchasing returns the updated product
Run the tests and report results.
```

**Step 4 — Self review:**
```
Walk me through every change you made. Explain the logic and flag anything you're uncertain about.
```

**Step 5 — Scope control practice:**
```
Add a discount system to the purchase flow.
Only modify service.py. Do not touch models.py or api.py.
If you need changes in other files to make this work,
tell me instead of making them.
```

---

## Chapter 5 Summary

| Concept | Key Takeaway | CCA Domain |
|---|---|---|
| Read-edit-verify loop | Claude reads first, plans, edits sequentially, then verifies | Domain 1 |
| Task decomposition | Explicit ordered sequences beat open-ended instructions | Domain 1 |
| Audit first pattern | Always map the blast radius before touching anything | Domain 1 |
| Touch list technique | Declare planned changes, get approval, then execute | Domain 1 |
| One-at-a-time pattern | Show diff per file, wait for approval on high-risk tasks | Domain 1 |
| Scope control | Hard and soft boundaries prevent scope creep | Domain 1 |
| Verification mindset | Diff review + test execution + self-audit prompt = complete verification | Domain 4 |
| Context degradation | Long tasks fill context window — use `/compact` and session breaks | Domain 5 |
| Self-audit prompt | Ask Claude to walk through its changes — it flags things it missed | Domain 4 |
| Session structure | Break large refactors across sessions for clean context | Domain 5 |

---

## ✅ Before Chapter 6

- [ ] Completed the multi-file feature build exercise
- [ ] Used the audit-first pattern on a real task
- [ ] Used the touch list technique — reviewed a plan before execution
- [ ] Rejected at least one diff with `n` and redirected
- [ ] Asked Claude to self-audit its changes after a multi-file session
- [ ] Triggered `/compact` on a long session and continued cleanly
- [ ] Answered all five practice questions correctly

---

*Previous: [Chapter 4 — CLAUDE.md: Persistent Context & Project Memory](./chapter-04-claude-md-persistent-context.md)*
*Next: Chapter 6 — Debugging & Bug Fixing (coming soon)*
