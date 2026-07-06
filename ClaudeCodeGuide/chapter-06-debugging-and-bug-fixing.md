# Chapter 6: Debugging & Bug Fixing

> **CCA Exam Note:** This chapter covers Domain 1 (Agentic Architecture & Orchestration, 27%) and Domain 4 (Prompt Engineering & Structured Output, 20%). The debugging scenario — "Code Generation with Claude Code" — is one of the six core scenario contexts the exam uses. Questions here test your ability to guide Claude through systematic diagnosis rather than jumping straight to fixes.

---

## 1. The Debugging Mindset — Diagnosis Before Fix

The single biggest mistake developers make when using Claude Code for debugging: **asking for a fix before asking for a diagnosis.**

```
❌ "Fix the bug in my payment processor"
✅ "Find the bug in my payment processor. Explain what's
    causing it and why before you change anything."
```

The first prompt gets you a change. The second gets you understanding — and a much higher chance the fix is correct.

Why does this matter? Because Claude can generate a plausible-looking fix to an incorrectly diagnosed bug. The fix compiles, passes a surface-level check, and appears correct. But it solves the wrong problem.

**The rule: diagnosis is not optional. It is phase one.**

```
Phase 1: DIAGNOSE — Understand root cause before touching code
Phase 2: VERIFY   — Confirm the diagnosis with evidence
Phase 3: FIX      — Implement the minimal correct fix
Phase 4: TEST     — Confirm the fix resolves the issue
Phase 5: REVIEW   — Check the fix didn't break anything else
```

This is the debugging loop you will apply to every bug in this chapter.

---

## 2. Types of Bugs Claude Code Handles

Understanding the bug category changes how you frame your prompt:

| Bug Type | Characteristics | Best Approach |
|---|---|---|
| **Syntax/Runtime Error** | Has a stack trace, clear error message | Paste the error, let Claude trace it |
| **Logic Error** | Code runs but produces wrong output | Describe expected vs actual behaviour |
| **Integration Bug** | Works in isolation, breaks when connected | Trace the boundary between components |
| **Performance Bug** | Correct output, too slow or memory-heavy | Ask Claude to profile, not fix |
| **Concurrency Bug** | Intermittent, hard to reproduce | Describe the conditions, ask for analysis |
| **Regression** | Worked before, broken now | Give Claude the git diff or commit range |

Each type needs a different diagnostic approach. The most dangerous is the **logic error** — Claude can generate a confident wrong answer because there is no error message to anchor its reasoning.

---

## 3. Debugging with Stack Traces

The most straightforward debugging scenario. You have an error — paste it.

### How to Frame It

```
This error occurs when I try to process a payment:

Traceback (most recent call last):
  File "src/api/checkout.py", line 47, in process_checkout
    result = payment_processor.charge(amount, card_token)
  File "src/payments/processor.py", line 23, in charge
    return self.client.create_charge(amount=amount, source=token)
  File "src/integrations/stripe_client.py", line 15, in create_charge
    response = stripe.Charge.create(**kwargs)
stripe.error.InvalidRequestError: No such token: 'tok_test_123'

Before fixing anything, trace through exactly what's happening
and explain why this error occurs.
```

### What Claude Does Internally

When given a stack trace, Claude:
1. Reads each file mentioned in the traceback
2. Follows the call chain from top to bottom
3. Identifies the exact line and variable state causing the failure
4. Reasons about why that state exists

You will see it use the Read tool on `checkout.py`, `processor.py`, and `stripe_client.py` in sequence — tracing the exact path the error takes.

### The Diagnosis-First Response

Claude should explain the root cause clearly and — if there is ambiguity — ask a clarifying question before fixing. A confident immediate fix on ambiguous information is a red flag.

---

## 4. Debugging Logic Errors — No Stack Trace

Logic errors are harder. The code runs. It just does the wrong thing.

### The "Expected vs Actual" Frame

```
The calculate_discount() function in src/pricing/discounts.py
is producing wrong results.

Expected: A 20% discount on a $100 order should return $80.00
Actual: It returns $120.00

Read the function and explain why it is returning the wrong value
before changing anything.
```

This frame gives Claude everything it needs:
- Which function
- What the correct behaviour should be
- What the wrong behaviour is
- An explicit instruction to diagnose first

### Providing a Minimal Reproduction

If you can give Claude a minimal reproduction, do it:

```
Here's a minimal example that shows the bug:

from src.pricing.discounts import calculate_discount

result = calculate_discount(price=100.00, discount_percent=20)
print(result)  # prints 120.0, should print 80.0

Read calculate_discount() and trace through this specific input
to find where the logic breaks.
```

Minimal reproductions dramatically speed up diagnosis because Claude has exact input/output evidence to reason from. There is no ambiguity about what "wrong" means.

### When You Don't Know Which Function Is Wrong

```
Orders with a discount code are calculating the wrong final price.
I don't know which function is responsible.

The relevant area is src/pricing/. Read the files in that directory
and trace how a discounted price is calculated from start to finish.
Identify every place where the calculation could go wrong.

Don't fix anything yet — give me a list of suspects with
your reasoning for each.
```

This is the **suspect list** pattern — useful when you have narrowed to a module but not a function. Claude explores and returns a ranked list of likely failure points.

---

## 5. Debugging Integration Bugs

Integration bugs live at the boundary between components. The classic: "works fine in unit tests, breaks in the real system."

### Trace the Boundary

```
The UserService works correctly in unit tests but fails
when called from the API layer in production.

@src/services/user_service.py
@src/api/user_router.py

Trace exactly how the API layer calls UserService. Focus on:
- What data is passed at the boundary
- Whether types/schemas match on both sides
- Whether the same database session is being used
- Whether any transformation happens to the data in transit

Explain what you find before making any changes.
```

### The Interface Contract Pattern

For integration bugs, ask Claude to explicitly document what each side expects:

```
Read @src/services/payment_service.py and @src/api/checkout_router.py

For the process_payment call specifically:
1. What does the API layer send? (types, field names, format)
2. What does the service expect to receive? (types, field names, format)
3. Are there any mismatches?

Show me the mismatch as a before/after table.
```

Claude will produce a precise table showing the exact schema mismatch — you did not have to read a single file yourself.

---

## 6. Debugging Regressions — "It Worked Before"

Regressions are bugs introduced by a recent change. The key diagnostic tool: the git diff.

### Give Claude the Diff

```
This worked in the last release but is now broken after
the recent auth refactor. Here's what changed:

[paste git diff or describe the commits]

Read the diff and identify what change could have caused
the login flow to fail for users with 2FA enabled.
```

Or let Claude get the diff itself:

```
Something broke the login flow for 2FA users in the last 3 commits.
Run `git log --oneline -5` and then read the diff for each commit.
Find the change that broke it.
```

Claude runs the git commands, reads each diff, and reasons about which change introduced the regression — dramatically faster than manual bisection.

### The Before/After Comparison Pattern

```
The old implementation of calculate_shipping() is in git at
commit abc1234. The new one is the current version.

Run: git show abc1234:src/shipping/calculator.py

Compare the old and new implementations side by side.
Explain what changed and why the new version might be
producing different results for orders over 10kg.
```

---

## 7. Debugging Performance Issues

Performance bugs need a different approach — you need profiling data, not a code rewrite.

### Profile Before Optimising

```
The generate_report() function in src/reports/generator.py
is taking 45 seconds for large datasets. It used to take 2 seconds.

Before changing anything:
1. Read the function and identify all database calls
2. Identify any N+1 query patterns
3. Identify any operations done in memory that could be
   done at the database level
4. Give me a ranked list of performance suspects

Don't optimise yet — I want to understand the problem first.
```

### The N+1 Query Pattern

One of the most common performance bugs in ORMs:

```
Read @src/services/invoice_service.py and check for N+1 query
patterns — places where we query the database inside a loop.
List every instance you find with line numbers and estimated
query count for a dataset of 1000 records.
```

Claude identifies these precisely through static analysis — it does not need to run the code.

### Performance Fix Sequence

Once diagnosed:
```
The N+1 query on line 67 is the main culprit.
Fix only that one issue. Use a single joined query
with eager loading instead of the loop.
Run the existing tests after fixing to confirm nothing broke.
```

Fix one thing at a time. Performance optimisation that touches multiple issues in one pass is hard to verify and easy to break.

---

## 8. The `/bug` Command — Reporting Claude Code Issues

Separate from debugging your own code — this is for when Claude Code itself is misbehaving:

```
/bug
```

This opens a bug report that sends feedback directly to Anthropic. Use it when:
- Claude Code crashes or hangs
- A slash command does not work as expected
- Tool use produces unexpected results
- You find a reproducible problem with Claude Code's own behaviour

This is different from Claude making a wrong diagnosis on your code — that is a prompting issue, not a bug. `/bug` is for Claude Code the tool itself.

---

## 9. Advanced Debugging Patterns

### The Rubber Duck Diagnosis

When you are stuck and not sure what is wrong:

```
I'm going to describe a bug and I want you to ask me questions
to help narrow down the cause. Don't read any files yet —
just ask me questions like a senior developer trying to
understand a bug report.

The bug: users are occasionally logged out unexpectedly
during active sessions. It doesn't happen every time.
It seems to happen more on mobile.
```

This forces structured thinking about the problem before diving into code. Often the questions Claude asks reveal information you knew but did not think was relevant.

### The Hypothesis Testing Pattern

```
I think the bug might be caused by the session token expiry
time being set incorrectly. My hypothesis:

The token TTL is being read from config in UTC but compared
against a timestamp stored in local time, causing sessions
to expire 5 hours early for US East Coast users.

Read @src/auth/session.py and @src/core/config.py and tell me
if this hypothesis is consistent with the code.
If not, what is actually happening?
```

Give Claude your hypothesis. Ask it to validate or refute with evidence from the code. This is much faster than asking Claude to diagnose from scratch when you already have a theory.

### The Binary Search Pattern

For bugs in large codebases where you have narrowed the problem to a data flow:

```
The incorrect value is coming from somewhere between the API
request and the database write. The input is correct (verified
at the API layer) and the stored value is wrong (verified in DB).

Trace the data flow from @src/api/order_router.py through to
the database write. At each transformation step, tell me what
the value should be and whether the code actually produces that.
Stop at the first point where you find a discrepancy.
```

### The Isolation Pattern

For bugs that only appear under specific conditions:

```
This bug only happens when:
- The user has more than 50 items in their cart
- AND they have a discount code applied
- AND they are using a non-USD currency

Read @src/cart/checkout.py and look specifically for code paths
triggered only by all three conditions simultaneously.
Focus on conditional logic that checks item count, discount
codes, and currency.
```

Describing the exact reproduction conditions turns a hard intermittent bug into a tractable code-reading exercise.

---

## 10. When Claude Gets the Diagnosis Wrong

This happens. Claude produces a confident, plausible diagnosis that turns out to be incorrect.

### Push Back With Evidence

```
I don't think that's the root cause. The issue also occurs
when the discount_code field is empty, so it's not specific
to the discount logic.

Here's a log entry from a failed request with no discount code:
[paste log]

Re-read your hypothesis with this new evidence and revise
your diagnosis.
```

Give Claude the contradicting evidence. It will revise — this is expected and healthy behaviour.

### The "What Evidence Would Prove This?" Test

```
You've diagnosed this as a race condition in the session update
logic. What specific evidence in the logs or code would confirm
this diagnosis?
```

Asking Claude to specify what evidence would prove its hypothesis forces a testable prediction. If that evidence is not present, the hypothesis is wrong.

### Starting Fresh on a Bad Diagnosis

```
/clear
```

If a debugging session has gone deeply down a wrong path, the cleanest move is to clear and start over with a better-framed prompt. Accumulated incorrect reasoning can bias Claude's subsequent responses even after you correct it.

---

## 11. Internal Mechanics — How Claude Traces Bugs

```
Bug report arrives
       ↓
STATIC ANALYSIS PHASE
  Claude reads relevant files — not running them,
  but building a model of:
  - Variable types at each point
  - Control flow paths
  - Data transformations
  - Call dependencies
       ↓
TRACE CONSTRUCTION
  Claude constructs the execution path for the
  specific failing case:
  - "Given input X, at line Y, variable Z has value W"
  - "This means the condition at line 47 is True"
  - "Which means we take the else branch..."
       ↓
FAILURE POINT IDENTIFICATION
  Claude identifies where the trace diverges from
  expected behaviour — the specific line and state
  where wrong output is first produced
       ↓
ROOT CAUSE REASONING
  Claude reasons backward from the failure point:
  - Why does that state exist?
  - What upstream code produced it?
  - Is it a logic error, data error, or assumption error?
       ↓
FIX GENERATION
  Only after the above — Claude generates a minimal
  fix targeting the identified root cause
```

### Why Claude Sometimes Gets It Wrong

Claude does **static analysis without execution**. It reasons about what the code *should* do, not what it *actually does* at runtime. This means:

- **Dynamic values** — Claude cannot know what a variable holds at runtime unless you tell it
- **External state** — database contents, environment variables, third-party API responses are invisible
- **Timing/concurrency** — Claude can identify patterns that *could* cause race conditions but cannot observe them happening
- **Platform-specific behaviour** — byte code optimisations, JIT quirks, OS-level differences

This is why giving Claude runtime evidence (logs, error messages, actual vs expected output) dramatically improves diagnostic accuracy. The more runtime information you provide, the less Claude has to guess.

---

## 12. CCA Exam — Practice Questions

**Q1:** A developer pastes a stack trace and asks Claude to "fix this error." Claude immediately produces a code change. What is the problem with this workflow?
- A) Claude should ask for more context before producing any response
- B) The developer skipped the diagnosis phase — Claude should have been asked to explain the root cause before fixing ✅
- C) Stack traces should not be pasted into Claude Code
- D) Claude should run the code before attempting a fix

**Q2:** A bug only occurs when three specific conditions are true simultaneously. What is the most effective debugging prompt structure?
- A) "Find the bug in the checkout system"
- B) "Read all files in the checkout module and find bugs"
- C) "The bug only occurs when conditions A, B, and C are all true. Find code paths triggered only by all three conditions together" ✅
- D) "Fix the intermittent checkout bug"

**Q3:** Claude produces a diagnosis for a logic error. The developer suspects the diagnosis is wrong. What is the most effective response?
- A) Accept Claude's diagnosis — it has read the code
- B) Use `/clear` and start a new session
- C) Provide contradicting evidence and ask Claude to revise its diagnosis ✅
- D) Ask Claude the same question again

**Q4:** A performance bug is causing a report to run 20x slower than expected. What is the correct first step with Claude Code?
- A) Ask Claude to optimise the function immediately
- B) Ask Claude to rewrite the function using a faster algorithm
- C) Ask Claude to read the function, identify all database calls and N+1 patterns, and rank performance suspects — without making any changes ✅
- D) Run a profiler and fix the top result

**Q5:** Claude has been debugging a regression for 20 messages and keeps arriving at incorrect diagnoses. What is the best course of action?
- A) Keep providing more context — Claude will eventually find it
- B) Switch to a more powerful model
- C) Use `/clear`, start fresh with a better-structured prompt including the git diff and a minimal reproduction ✅
- D) Ask Claude to read every file in the codebase

**Q6:** A bug only occurs in production but not locally. Claude has read the code but cannot identify the issue. What additional information should you provide?
- A) More code files for Claude to read
- B) Runtime evidence: production logs, environment variable differences, actual database state at time of failure ✅
- C) A description of expected behaviour
- D) The full codebase context

---

## 13. Practical Exercise — The Full Debugging Workflow

**Setup** — create these buggy files:

`discount.py`:
```python
def calculate_discount(price: float, discount_percent: int) -> float:
    """Apply a percentage discount to a price."""
    discount_amount = price * discount_percent  # Bug: missing /100
    return price + discount_amount              # Bug: should subtract

def apply_bulk_discount(items: list, threshold: int = 10) -> list:
    """Apply 15% discount if order has more than threshold items."""
    if len(items) > threshold:
        # Bug: discounts item count, not price
        return [calculate_discount(len(items), 15) for item in items]
    return items
```

`orders.py`:
```python
from discount import calculate_discount, apply_bulk_discount

def process_order(items: list, discount_code: str = None) -> dict:
    processed_items = apply_bulk_discount(items)
    subtotal = sum(item['price'] for item in items)  # Bug: uses original items
    
    if discount_code == "SAVE20":
        final_price = calculate_discount(subtotal, 20)
    else:
        final_price = subtotal
    
    return {"items": processed_items, "subtotal": subtotal, "final_price": final_price}
```

**Exercise 1 — Logic error with expected/actual frame:**
```
Expected: calculate_discount(100, 20) should return 80.0
Actual: returns something wrong.

Read discount.py and trace through calculate_discount(100, 20)
step by step. At each line tell me the value of each variable.
Find where the logic breaks. Don't fix yet.
```

**Exercise 2 — Suspect list:**
```
process_order() returns wrong totals for bulk orders with a
discount code. I don't know which bug to fix first.

Read discount.py and orders.py. List every bug you find,
ranked by impact on the final_price. Don't fix anything yet.
```

**Exercise 3 — Hypothesis testing:**
```
My hypothesis: apply_bulk_discount() discounts the item count
instead of the item prices.

Read apply_bulk_discount() and tell me if this is correct.
If so, what is the minimal fix?
```

**Exercise 4 — Fix with verification:**
```
Fix all bugs one at a time. After each fix, run:

python -c "
from discount import calculate_discount
assert calculate_discount(100, 20) == 80.0, f'Got {calculate_discount(100, 20)}'
print('PASS')
"

Report whether each test passes before moving to the next fix.
```

**Exercise 5 — Push back on wrong diagnosis:**
```
[After Claude gives any diagnosis]

I don't think that fully explains it. The wrong total also
appears for orders with no discount code applied.
Here is an example: process_order([{'price': 10}] * 15)
returns the wrong subtotal even with no discount.

Revise your diagnosis with this evidence.
```

---

## Chapter 6 Summary

| Concept | Key Takeaway | CCA Domain |
|---|---|---|
| Diagnosis before fix | Always understand root cause before changing code | Domain 1 |
| Bug type framing | Different bug types need different prompt structures | Domain 4 |
| Stack trace debugging | Paste the error, ask for diagnosis before fix | Domain 1 |
| Logic error debugging | Expected vs actual frame + minimal reproduction | Domain 4 |
| Integration debugging | Trace the boundary, use the interface contract pattern | Domain 1 |
| Regression debugging | Give Claude the git diff, ask it to find the culprit | Domain 1 |
| Performance debugging | Profile first, optimise one issue at a time | Domain 1 |
| Wrong diagnosis handling | Provide contradicting evidence, ask Claude to revise | Domain 4 |
| Static analysis limits | Claude cannot see runtime state — provide logs and evidence | Domain 1 |
| Fresh start | `/clear` when accumulated wrong reasoning biases the session | Domain 5 |

---

## ✅ Before Chapter 7

- [ ] Completed all five exercises in the practical section
- [ ] Used the diagnosis-first pattern on a real bug in your own code
- [ ] Gave Claude a stack trace and asked for diagnosis before fix
- [ ] Pushed back on a Claude diagnosis with contradicting evidence
- [ ] Used the hypothesis testing pattern on a logic error
- [ ] Answered all six practice questions correctly

---

*Previous: [Chapter 5 — File Editing & Multi-File Tasks](./chapter-05-file-editing-and-multi-file-tasks.md)*
*Next: Chapter 7 — Git Integration (coming soon)*
