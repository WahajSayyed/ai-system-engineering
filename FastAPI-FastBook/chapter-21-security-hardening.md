# Chapter 21: Security Hardening

> Part III — Advanced: Production Engineering · Chapter 21 of 28

This chapter audits decisions made several chapters ago rather than introducing an entirely new feature: the OWASP API Security Top 10 as it applies to code this curriculum has already written, security headers, secrets management done properly, and — the centerpiece — a real authorization vulnerability this curriculum's own Notes API has quietly had since Chapter 3, found and fixed.

A brief, current framing note: OWASP maintains two genuinely separate Top 10 projects — the general **OWASP Top 10** (web applications broadly, which received a major revision in November 2025) and the **OWASP API Security Top 10**, specifically about API-shaped risks, last revised in 2023. This chapter uses the API-specific list, since it's the one built around exactly the kind of application this curriculum has been constructing.

## Learning Objectives

By the end of this chapter you will be able to:

- Name the OWASP API Security Top 10 (2023) categories most relevant to a FastAPI application, and identify which ones this curriculum's own code has and hasn't already defended against.
- Find and fix a Broken Object Level Authorization (BOLA) vulnerability — the single most common, most exploited API vulnerability category — in code from earlier in this curriculum.
- Explain why a failed authorization check should return `404`, not `403`, and why that choice itself is a security decision.
- Add standard security headers via middleware, and explain what each one defends against.
- Manage secrets correctly across environments, and identify what belongs in a real secrets manager rather than a checked-in `.env` file.
- Identify and fix a SQL injection risk, and explain why SQLAlchemy's query builder prevents it structurally rather than through escaping.

---

## 21.1 The OWASP API Security Top 10 (2023): What This Curriculum Can Actually Demonstrate

The full list, briefly: **API1** Broken Object Level Authorization (BOLA), **API2** Broken Authentication, **API3** Broken Object Property Level Authorization (BOPLA), **API4** Unrestricted Resource Consumption, **API5** Broken Function Level Authorization (BFLA), **API6** Unrestricted Access to Sensitive Business Flows, **API7** Server-Side Request Forgery (SSRF), **API8** Security Misconfiguration, **API9** Improper Inventory Management, **API10** Unsafe Consumption of APIs.

Five of these map directly onto code and decisions already made in this curriculum, which is where this chapter spends its time: **BOLA** (API1) — this chapter's centerpiece, found live in the Notes API; **Broken Authentication** (API2) — an audit pass over Chapter 11; **BOPLA** (API3) — already substantially defended by Chapter 6's input/output model separation, worth naming explicitly now that you have the term for it; **Unrestricted Resource Consumption** (API4) — Chapter 19's rate limiting, extended here specifically to the login endpoint; **Security Misconfiguration** (API8) — security headers, secrets, and the `docs_url` toggle from Chapter 18's exercises.

## 21.2 BOLA: The Vulnerability This Curriculum Has Had Since Chapter 3

**Broken Object Level Authorization has held the #1 spot on this list since it was first published in 2019, and accounts for roughly 40% of real API attacks** — it's the single most important category to actually internalize, not just recognize by name. The pattern: an endpoint accepts an object ID, correctly verifies the caller is authenticated, but never checks whether *this specific object* actually belongs to *this specific caller*. Any authenticated user can then access any other user's data, simply by supplying a different ID.

Here's the uncomfortable reveal: the Notes API, dormant since Chapter 4, has exactly this bug, and it's easy to see why once you look — `get_note_or_raise` (Chapter 7) checks only that a note *exists*:

```python
# The shape this has had since Chapter 7 — checks existence, not ownership
async def get_note_or_raise(self, note_id: int) -> NoteTable:
    note = await self.session.get(NoteTable, note_id)
    if note is None:
        raise NotFoundError("Note", note_id)
    return note
```

If Notes were ever attached to a real user (as a genuinely private resource should be), this function has no way to stop User B from reading, editing, or deleting User A's note — it never asks "does this note belong to whoever is currently authenticated?" at all, only "does a note with this ID exist anywhere?" This isn't a contrived example — it's the literal, unmodified shape of code already written earlier in this curriculum, which is exactly the point: BOLA is easy to introduce by simply never thinking to add the check, not by writing something obviously wrong.

The fix, and a detail worth being deliberate about:

```python
async def get_owned_note_or_raise(self, note_id: int, owner_id: int) -> NoteTable:
    note = await self.session.get(NoteTable, note_id)
    if note is None or note.owner_id != owner_id:
        # Deliberately the SAME exception, the SAME 404, for "doesn't exist"
        # and "exists but belongs to someone else" — see the discussion below.
        raise NotFoundError("Note", note_id)
    return note
```

**Why `404`, not `403`?** Returning `403 Forbidden` for "this note exists but isn't yours" would *confirm to the caller* that a note with that ID exists at all — leaking information (the existence of another user's private resource) that a `404` correctly withholds. From the caller's point of view, "this note doesn't exist" and "this note exists but you can't have it" should be indistinguishable, precisely because distinguishing them hands an attacker a free way to enumerate which IDs correspond to real, private data belonging to other people, even without ever seeing that data's contents. This is a genuinely common real-world mistake — a well-intentioned `403` that leaks exactly the information it was trying not to.

## 21.3 BOPLA: Already Defended, Now Named

Chapter 6's `ProductCreate`/`ProductUpdate`/`ProductPublic` separation, and Chapter 11's `UserCreate`/`UserPublic` separation, were built *before* this chapter had a name for what they defend against. **Broken Object Property Level Authorization** covers exactly two failure modes: *excessive data exposure* (a response includes fields the caller shouldn't see — Chapter 6's `cost_price`, Chapter 11's `hashed_password`, both already structurally excluded from their respective `...Public` models) and *mass assignment* (an input accepts fields the caller shouldn't be able to set — Chapter 6's `ProductUpdate` deliberately omitting `cost_price` combined with `extra="forbid"`, and Chapter 11's `UserCreate` never declaring a `role` field at all, meaning a signup payload can't smuggle in `{"role": "admin"}` no matter what a malicious client sends).

Worth stating plainly, now that the vulnerability class has a name: **this defense was never a filter applied after the fact — it was a structural property of which fields each schema declares, from the moment those models were written.** A future engineer adding a new field to `ProductTable` doesn't automatically expose or make assignable that field anywhere — they'd have to deliberately add it to `ProductPublic` or `ProductCreate` for it to become visible or settable, which is precisely the safe default this design produces.

## 21.4 Broken Authentication: Auditing Chapter 11

A checklist, applied against what Chapter 11 actually built:

- ✅ Passwords hashed with Argon2 via `pwdlib` — not reversible, not plaintext.
- ✅ JWTs signed with a secret sourced from environment configuration (Chapter 16), not a hardcoded literal.
- ✅ Token expiry (`exp`) enforced on every verification.
- ✅ No sensitive data embedded in the JWT payload itself (Chapter 11.2's warning, followed).
- ❌ **Rate limiting specifically on the login endpoint — missing until this chapter.** Chapter 19's rate limiter protects the API broadly, but a login endpoint is a specifically attractive target for credential-stuffing and brute-force attacks, and deserves its own, much stricter limit than general API traffic — a handful of attempts per minute, not the hundred-per-minute general limit Chapter 19 configured for ordinary usage.

```python
# routers/v1/auth.py (addition)
import time
from fastapi import Request, HTTPException, Depends
from cache import redis_client

LOGIN_RATE_LIMIT = 5
LOGIN_WINDOW_SECONDS = 60

async def check_login_rate_limit(request: Request):
    key = f"login_ratelimit:{request.client.host}"
    now = time.time()
    await redis_client.zremrangebyscore(key, 0, now - LOGIN_WINDOW_SECONDS)
    if await redis_client.zcard(key) >= LOGIN_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")
    await redis_client.zadd(key, {str(now): now})
    await redis_client.expire(key, LOGIN_WINDOW_SECONDS)

@router.post("/login", response_model=Token, dependencies=[Depends(check_login_rate_limit)])
async def login(...):
    ...
```

This reuses Chapter 19's exact sliding-window sorted-set technique, applied at a much stricter threshold and scoped specifically to this one security-critical endpoint via `dependencies=[Depends(...)]` (Chapter 8.5's side-effect-only dependency pattern) — five attempts per minute per IP is enough for a legitimate user who mistypes their password once or twice, and a real obstacle to automated credential-stuffing against this endpoint specifically.

## 21.5 Security Headers

A small set of response headers that cost almost nothing to add and close off entire categories of browser-side attack:

| Header | Defends against |
|---|---|
| `X-Content-Type-Options: nosniff` | Browsers guessing a response's content type differently than declared, potentially executing something as script that was meant to be plain data |
| `X-Frame-Options: DENY` | Clickjacking — your site being embedded in an invisible iframe on an attacker's page, tricking users into clicking something they can't see |
| `Strict-Transport-Security` (HSTS) | Downgrade attacks — tells browsers to *only* ever connect over HTTPS to this domain, even if a link or bookmark says `http://` |
| `Content-Security-Policy` (CSP) | Limits what sources scripts/styles/etc. can load from, meaningfully reducing the damage a successful cross-site scripting injection can do |

```python
# middleware.py (addition)
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
```

## 21.6 Secrets Management, Beyond `.env`

Chapter 16's `.env` file is correct for **local development** and must never be treated as correct for anything beyond it: `.env` should be listed in `.gitignore` and never committed to version control — a secret checked into git history remains recoverable from that history indefinitely, even after being "removed" in a later commit. In production, the same `Settings` class reads its values from **real environment variables**, injected by your deployment platform (a container orchestrator's secret store, a cloud provider's secrets manager) rather than any file at all — the code never changes, only where the values actually come from, exactly Chapter 18.5's "same code, different values" principle applied specifically to secrets.

Two further practical points: **secrets should be rotated periodically** (Chapter 11's Exercise 11.5 already demonstrated the operational reality of what changing `SECRET_KEY` does — instant, total invalidation of every existing token — worth planning for deliberately rather than treating rotation as something that only happens during an incident response). And **secrets must never end up in logs** — Chapter 20's structured logging needs active care here: logging a full request body that happens to contain a password field, or logging the raw `Authorization` header, silently writes a live credential into your log storage, which is very often *less* protected and *more* widely accessible than your actual credential store or database.

## 21.7 Supply Chain, Briefly

Pin dependency versions (a lockfile — `uv.lock` — records exact resolved versions, not just the ranges in `pyproject.toml`), so what you tested is what actually deploys. Watch for security advisories against your dependencies rather than assuming "it hasn't been a problem yet" will remain true. Be deliberate about package names, especially when adding a new dependency under time pressure — a typo'd or unfamiliar-but-similar-sounding package name is exactly the shape a dependency-confusion or typosquatting attack takes. This curriculum treats supply-chain security as a brief, named concern rather than a deep topic — the tooling for it (SCA scanners, `pip-audit`, Dependabot-style automation) is genuinely worth adopting in a real project, beyond what one chapter can meaningfully implement and verify hands-on.

---

## Hands-On Project: Fixing BOLA, Hardening Auth, and Adding Security Headers

### Step 1 — Reproduce the BOLA vulnerability, then fix it

```python
# models.py — giving Notes a real owner, finally
class NoteTable(SQLModel, table=True):
    __tablename__ = "notes"
    id: int | None = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="users.id")
    title: str
    content: str
```

```python
# repositories/note.py
class NoteRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_owned_or_raise(self, note_id: int, owner_id: int) -> NoteTable:
        note = await self.session.get(NoteTable, note_id)
        if note is None or note.owner_id != owner_id:
            raise NotFoundError("Note", note_id)
        return note
```

```python
# routers/v1/notes.py
@router.get("/{note_id}", response_model=NotePublic)
async def read_note(note_id: int, current_user: CurrentUserDep, repo: NoteRepoDep):
    return await repo.get_owned_or_raise(note_id, owner_id=current_user.id)
```

Write the demonstration as a test, deliberately proving the fix rather than just trusting it:

```python
@pytest.mark.asyncio
async def test_user_cannot_read_another_users_note(client, session):
    # user A signs up and creates a note
    await client.post("/auth/signup", json={"username": "alice", "password": "s3cret123"})
    login_a = await client.post("/auth/login", data={"username": "alice", "password": "s3cret123"})
    token_a = login_a.json()["access_token"]
    create_response = await client.post(
        "/notes/", json={"title": "Private", "content": "secret stuff"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    note_id = create_response.json()["id"]

    # user B signs up separately and tries to read A's note by ID
    await client.post("/auth/signup", json={"username": "bob", "password": "s3cret456"})
    login_b = await client.post("/auth/login", data={"username": "bob", "password": "s3cret456"})
    token_b = login_b.json()["access_token"]
    response = await client.get(f"/notes/{note_id}", headers={"Authorization": f"Bearer {token_b}"})

    assert response.status_code == 404   # not 200 with Alice's private content, not 403 either
```

### Step 2 — Rate-limit `/auth/login` specifically (section 21.4)

### Step 3 — Add `SecurityHeadersMiddleware` (section 21.5), positioned outermost alongside CORS (Chapter 12's ordering rule)

### Step 4 — Confirm `.env` is git-ignored, and document (in your project's README, not in code) that production secrets come from real environment variables, never a checked-in file.

---

## Practice Exercises

**Exercise 21.1 — Find and fix a SQL injection risk.**
Here is a deliberately vulnerable search endpoint (never use this pattern for real — it's included specifically to be fixed):

```python
@router.get("/search")
async def search_products_vulnerable(q: str, session: SessionDep):
    query = f"SELECT * FROM products WHERE name LIKE '%{q}%'"
    result = await session.execute(text(query))
    return result.fetchall()
```

Explain what happens if a client sends `q = "' OR '1'='1"`, and rewrite the endpoint using SQLAlchemy's query builder (`select(...).where(...)`) so that `q`'s value is never directly interpolated into a SQL string at all. Explain, referencing Chapter 9's original decision to use SQLModel/SQLAlchemy throughout, why that earlier choice was already a meaningful security decision, not just a convenience one.

**Exercise 21.2 — Verify your security headers programmatically.**
Write a short script (or a pytest test) that makes a request to any endpoint and asserts that `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security`, and `Content-Security-Policy` are all present in the response headers, with the expected values. Run it before and after adding `SecurityHeadersMiddleware` to confirm it correctly fails before and passes after.

**Exercise 21.3 — Write a pre-merge security checklist for new endpoints.**
Drawing on Chapters 7, 11, 12, 18, and this chapter, write a concrete checklist (5–8 items) a developer should run through before merging any *new* endpoint into this application — covering at minimum: authorization (does it check ownership, not just existence, for any user-owned resource?), error handling (does a failed auth check return the right status code, without leaking information?), and anything else this curriculum has established as a standing expectation for every endpoint.

**Exercise 21.4 — Apply the BOLA fix to a second resource.**
If your project has Reviews (Chapter 10's exercise) or another user-owned resource, apply the exact same `get_owned_or_raise` pattern to it, and write a test proving one user cannot read, update, or delete another user's review by ID — the same shape as Step 1's test, applied independently, confirming the pattern generalizes rather than being a one-off fix specific to Notes.

**Exercise 21.5 (stretch) — Prove the 404-vs-403 information leak concretely.**
Temporarily change `get_owned_note_or_raise` to raise a `403 Forbidden` (with a distinct message, like `"You don't own this note"`) specifically when a note exists but belongs to someone else, while still raising `404` when a note genuinely doesn't exist at all. Write a test demonstrating that an attacker (without ever seeing the note's actual content) can now distinguish "this ID belongs to a real, private note owned by someone else" from "this ID doesn't correspond to anything," purely by checking whether the response is `403` or `404` — confirming section 21.2's claim concretely, then revert to the single, uniform `404` behavior.

---

## Solutions & Discussion

<details>
<summary>Exercise 21.1</summary>

With `q = "' OR '1'='1"`, the constructed string becomes:
```sql
SELECT * FROM products WHERE name LIKE '%' OR '1'='1%'
```
`'1'='1'` is always true, so the `WHERE` clause is effectively neutralized — every row in the table is returned, regardless of the intended name filter, completely bypassing the search. A more damaging payload (depending on the database and whether multi-statement execution is permitted) could go further than just bypassing a filter — the fundamental problem is that `q`'s value became part of the SQL *code* itself, rather than being treated as *data*.

```python
@router.get("/search")
async def search_products_fixed(q: str, repo: ProductRepoDep):
    return await repo.search_by_name(q)

# repositories/product.py
async def search_by_name(self, q: str) -> list[ProductTable]:
    result = await self.session.execute(
        select(ProductTable).where(ProductTable.name.ilike(f"%{q}%"))
    )
    return list(result.scalars().all())
```

SQLAlchemy's query builder never concatenates `q` into a raw SQL string at all — `q`'s value is sent to the database driver as a separate, bound parameter, alongside a parameterized query template. The database itself treats that bound value strictly as data, never as part of the SQL grammar, no matter what characters it contains — this is a structural guarantee, not a matter of successfully escaping every dangerous character (which is exactly the fragile, error-prone approach real injection defenses moved away from). This is precisely why Chapter 9's decision to route every query through SQLModel/SQLAlchemy rather than hand-built SQL strings mattered beyond convenience: it closed off this entire vulnerability class by construction, for every query written that way, without anyone needing to remember to sanitize input on a query-by-query basis.
</details>

<details>
<summary>Exercise 21.2</summary>

```python
def test_security_headers_present(client_sync):
    response = client_sync.get("/products/")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "max-age" in response.headers["strict-transport-security"]
    assert response.headers["content-security-policy"] == "default-src 'self'"
```

Before adding `SecurityHeadersMiddleware`, this test fails with a `KeyError`-style failure (the headers simply aren't present at all). After adding it, the test passes — a concrete, automated confirmation that these headers are actually being sent, rather than something a developer eyeballed once in a browser's network tab and then never verified again after later refactoring the middleware stack.
</details>

<details>
<summary>Exercise 21.3</summary>

A reasonable checklist:
1. Does this endpoint check *ownership*, not just *existence*, for any resource that belongs to a specific user? (Section 21.2 — BOLA.)
2. Does a failed authorization check return `404`, not `403`, unless there's a deliberate, considered reason existence-leakage is acceptable here? (Section 21.2.)
3. Are input and output modeled with separate schemas, with no field present in an output model that shouldn't be publicly visible, and no field acceptable as input that shouldn't be client-settable? (Section 21.3 — BOPLA.)
4. Does this endpoint raise domain exceptions (Chapter 7's `AppError` subclasses) rather than `HTTPException` directly, if it lives in a service or repository? (Chapter 18's layering rule.)
5. Is any new query built through SQLAlchemy's query builder, never through raw string interpolation of user input? (This chapter, Exercise 21.1.)
6. Does a genuinely expensive or abusable operation have its own rate limit, distinct from the general API-wide limit, if it's a plausible target (login, signup, password reset, anything resembling a "sensitive business flow")? (Section 21.4, Chapter 19.)
7. Are any new configuration values added via `Settings` (Chapter 16), with no new hardcoded secrets or environment-specific literals anywhere in the diff?
8. Does any new logging statement avoid including a password, token, or other credential-shaped value anywhere in its arguments? (Section 21.6.)
</details>

<details>
<summary>Exercise 21.4</summary>

```python
# repositories/review.py
async def get_owned_or_raise(self, review_id: int, owner_id: int) -> ReviewTable:
    review = await self.session.get(ReviewTable, review_id)
    if review is None or review.owner_id != owner_id:
        raise NotFoundError("Review", review_id)
    return review
```

```python
@pytest.mark.asyncio
async def test_user_cannot_modify_another_users_review(client, session):
    # (same two-user setup pattern as Step 1's Notes test, applied to Reviews)
    ...
    response = await client.patch(f"/reviews/{review_id}", json={"rating": 1}, headers={"Authorization": f"Bearer {token_b}"})
    assert response.status_code == 404
```

The identical pattern applies cleanly to a second resource with essentially no new ideas required — confirming that `get_owned_or_raise` is a genuine, reusable *pattern* for any user-owned resource, not a fix specific to Notes' particular circumstances.
</details>

<details>
<summary>Exercise 21.5</summary>

```python
async def get_owned_note_or_raise_leaky(self, note_id: int, owner_id: int) -> NoteTable:
    note = await self.session.get(NoteTable, note_id)
    if note is None:
        raise NotFoundError("Note", note_id)
    if note.owner_id != owner_id:
        raise HTTPException(status_code=403, detail="You don't own this note")   # the leak
    return note
```

```python
@pytest.mark.asyncio
async def test_403_leaks_existence(client, session):
    # Alice creates note (id known), Bob queries a genuinely nonexistent id and Alice's real id
    response_nonexistent = await client.get("/notes/999999", headers=bob_headers)
    response_real_but_not_owned = await client.get(f"/notes/{alice_note_id}", headers=bob_headers)

    assert response_nonexistent.status_code == 404
    assert response_real_but_not_owned.status_code == 403   # <-- distinguishable from 404!
```

Bob never sees Alice's note content in either case — but the *status code alone* tells him definitively that `alice_note_id` corresponds to a real note belonging to someone else, while `999999` does not. An attacker with nothing more than the ability to send requests and read status codes can now enumerate real, private resource IDs across the entire ID space, purely by checking which ones return `403` versus `404` — turning a single design choice into a systematic information-disclosure tool. Reverting to a uniform `404` for both cases (this chapter's original fix) makes the two situations genuinely indistinguishable from the outside, closing that enumeration path entirely.
</details>

---

## Chapter Summary

- BOLA (checking existence but not ownership) is the single most common, most exploited API vulnerability category, and this curriculum's own Notes API had exactly this bug from Chapter 3 onward — the fix is a repository-level ownership check, and the *status code* chosen for a failed check (404, never 403) is itself a security decision.
- BOPLA (excessive data exposure, mass assignment) was already substantially defended by Chapter 6's and Chapter 11's input/output model separation — a structural property of which fields each schema declares, not a filter applied after the fact.
- Login and other security-sensitive endpoints deserve their own, stricter rate limit beyond general API-wide limiting, reusing Chapter 19's sliding-window technique at a much tighter threshold.
- Security headers (`X-Content-Type-Options`, `X-Frame-Options`, HSTS, CSP) are cheap to add and close off entire categories of browser-side attack.
- `.env` is for local development only — production secrets come from real environment variables or a secrets manager, never a committed file, and secrets should never appear in logs.
- SQLAlchemy's query builder prevents SQL injection structurally, by sending user input as bound parameters rather than concatenated SQL text — a benefit of Chapter 9's original tooling choice that this chapter makes explicit.

**Next:** Chapter 22 covers task queues and distributed processing — replacing Chapter 13's `BackgroundTasks` with a real, durable, retrying queue (ARQ or Celery, backed by Redis), closing the gap this curriculum has flagged since Chapter 13 first introduced background work. This is the final chapter of Part III.
