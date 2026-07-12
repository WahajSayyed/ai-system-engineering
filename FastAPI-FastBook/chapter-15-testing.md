# Chapter 15: Testing FastAPI Applications

> Part II — Intermediate: Building Real APIs · Chapter 15 of 28

Every chapter in Part II has been verified by hand — `curl`, the `/docs` UI, manual restarts. This chapter replaces that with a real, repeatable pytest suite: `TestClient`, dependency overrides that swap out the real database and auth for test-only versions, fixtures for clean per-test isolation, and mocking for external dependencies like the email sender from Chapter 13. This closes out Part II — everything from here through Chapter 22 assumes you can verify behavior with a test, not just a manual check.

## Learning Objectives

By the end of this chapter you will be able to:

- Choose between `TestClient` and `httpx.AsyncClient` for testing a FastAPI application, and explain when each is the better fit.
- Use `app.dependency_overrides` to substitute a test database and a test user for the real dependencies, without touching route code at all.
- Write pytest fixtures that provide clean, isolated setup/teardown per test — the test-level analog of Chapter 8's `yield`-based dependencies.
- Design a test database strategy that never touches a real database, with a documented, real gotcha around in-memory SQLite and connection pooling.
- Mock an external dependency correctly — including patching it at the right location, a genuinely common source of silently-passing-for-the-wrong-reason tests.

---

## 15.1 `TestClient` vs `httpx.AsyncClient`

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
```

`TestClient` wraps your ASGI application in a synchronous interface — no real network socket, no real server process, just direct in-process calls into your app — which is why ordinary `def` test functions work with it, no `async`/`await` needed anywhere in the test itself, even though the application underneath is fully async. This is the right default for the large majority of route-behavior tests, and it's what Chapter 13's background-task testing already relied on (recall: `TestClient` runs the entire request cycle, background tasks included, to completion before returning control to your test).

```python
import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_read_root():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
```

`httpx.AsyncClient`, wired to your app via `ASGITransport` instead of a real network connection, is the async-native alternative — genuinely `async def` test functions, needed when your fixtures are already async (an async database session, say) and you'd rather not bridge between sync test code and async setup, or when you specifically want to test real concurrent behavior (multiple simulated clients hitting your app at once, à la Chapter 9's Exercise 9.3). This chapter uses `AsyncClient` throughout, specifically because the test database fixtures it needs are async themselves — bridging an async fixture into a sync `TestClient` test is possible but adds friction `AsyncClient` avoids entirely.

## 15.2 Dependency Overrides

Every dependency this curriculum has built — `get_session` (Chapter 9), `get_current_user` (Chapter 11) — is just a callable FastAPI resolves via `Depends(...)`. Tests need different behavior from all of them: a test database instead of the real one, a known test user instead of real JWT verification. `app.dependency_overrides` is a dict, keyed by the *original* dependency callable, mapping to a *replacement* callable used instead, for the lifetime of the override:

```python
from database import get_session

async def get_session_override():
    yield test_session

app.dependency_overrides[get_session] = get_session_override
# ... run tests ...
app.dependency_overrides.clear()   # always clean up, or overrides leak into unrelated tests
```

Nothing about your route code, or the dependency's own declaration, needs to change — `Depends(get_session)` in a route still refers to the *original* `get_session` function object; the override dict is consulted at resolution time and silently substitutes the replacement, transparently, for every route that would otherwise have used the real one. Forgetting to clear an override after a test is a genuine, easy-to-hit bug: it silently persists into every subsequent test in the same process, potentially making unrelated tests pass or fail for the wrong reasons — always pair setting an override with clearing it, ideally via fixture teardown rather than manual cleanup you might forget.

## 15.3 Fixtures — The Test-Level `yield`-Dependency Pattern

Pytest fixtures use the exact same shape Chapter 8.3 introduced for dependencies — code before `yield` is setup, the yielded value is what your test receives, code after `yield` is teardown, guaranteed to run whether the test passed or failed:

```python
import pytest

@pytest.fixture
def some_resource():
    resource = create_resource()
    yield resource
    resource.cleanup()
```

This isn't a coincidence or a forced analogy — it's the same underlying pattern (open something, hand it to the code that needs it, guarantee cleanup afterward) applied to a different context. If Chapter 8's `yield`-based dependencies made sense, pytest fixtures should feel immediately familiar rather than like new syntax to memorize. Fixtures also support **scope** (`function`, the default — fresh per test; `module`; `session` — shared across many tests) — this chapter uses function-scoped fixtures for the database, prioritizing isolation (every test gets a genuinely clean slate) over the speed a shared, session-scoped database might offer; that speed/isolation trade-off is worth knowing exists, even though this chapter's suite doesn't need to make it.

## 15.4 Test Database Strategy — and a Real SQLite Gotcha

Tests must never run against a real (or even shared development) database — a failing test that leaves bad data behind, or two tests racing against the same rows, defeats the entire point of testing. The straightforward approach: a fresh, in-memory SQLite database, created (schema and all) at the start of every single test function, and thrown away at the end.

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
```

**`poolclass=StaticPool` is not optional decoration here — it's fixing a real, specific gotcha.** SQLite's `:memory:` database lives only within the single connection that created it; by default, SQLAlchemy's connection pooling can open a *new* connection for a later operation, and that new connection gets its own, completely separate, empty in-memory database — meaning tables you created moments earlier on a different connection simply don't exist from the new connection's point of view. `StaticPool` forces the engine to reuse exactly one single connection for its entire lifetime, so every operation against this test engine sees the same in-memory database, tables included. Skipping this is a classic, confusing first encounter with in-memory SQLite testing — Exercise 15.5 has you reproduce the resulting failure on purpose, so the fix isn't just a line you copy without understanding why it's there.

## 15.5 Mocking External Dependencies — And Patching Them in the Right Place

Dependency overrides handle anything wired through FastAPI's `Depends(...)` system. Chapter 13's `send_welcome_email`, though, is called *directly* inside a route (via `background_tasks.add_task(send_welcome_email, ...)`) — it was never a `Depends(...)`-injected dependency, so `app.dependency_overrides` has nothing to do with it. For this, you reach for `unittest.mock.patch`:

```python
from unittest.mock import patch

def test_signup_sends_email():
    with patch("routers.auth.send_welcome_email") as mock_send:
        # ... call signup ...
        mock_send.assert_called_once()
```

**The exact string you pass to `patch(...)` matters more than it looks like it should, and getting it wrong is a genuinely common way to write a test that silently tests nothing.** `routers/auth.py` does `from tasks import send_welcome_email` — which binds the *name* `send_welcome_email` inside `routers.auth`'s own namespace, pointing at the same function object `tasks.send_welcome_email` also points to. Patching `tasks.send_welcome_email` replaces the attribute on the `tasks` module — but `routers.auth`'s own already-bound reference to the original function is untouched by that, since it's a separate name binding, not a live alias. The rule to memorize: **patch the name where it's looked up at call time, not where it was originally defined.** Here, that's `routers.auth.send_welcome_email`, because that's the namespace the route function actually reads `send_welcome_email` from when it calls it. Exercise 15.3 has you deliberately patch the wrong target first, watch the mock silently never get called (while the real function keeps running unmocked), and then fix it — because reading about this mistake is a weaker lesson than watching a test pass for the wrong reason.

---

## Hands-On Project: A Real Test Suite for the Products + Auth API

### Step 1 — `conftest.py`: an isolated test database per test, and two client fixtures

```python
# conftest.py
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from database import get_session
from auth_dependencies import get_current_active_user
from models import UserTable

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    test_session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with test_session_maker() as s:
        yield s

    await engine.dispose()


@pytest_asyncio.fixture
async def client(session):
    async def get_session_override():
        yield session

    app.dependency_overrides[get_session] = get_session_override

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def authenticated_client(client, session):
    """A client fixture for tests that need *a* logged-in user, without re-testing auth itself."""
    test_user = UserTable(username="testuser", hashed_password="unused", role="user", disabled=False)
    session.add(test_user)
    await session.commit()
    await session.refresh(test_user)

    async def override_current_user():
        return test_user

    app.dependency_overrides[get_current_active_user] = override_current_user
    yield client
    del app.dependency_overrides[get_current_active_user]
```

Every test gets a brand-new `engine`, a brand-new in-memory database with freshly created tables, and a brand-new session — genuine per-test isolation, at the cost of recreating the schema every single test (a completely reasonable trade-off at this curriculum's scale; larger test suites often switch to a shared engine with a per-test transaction rollback instead, prioritizing speed once isolation-via-full-recreation gets slow).

### Step 2 — Testing the real auth flow (happy path + failures)

```python
# test_auth.py
import pytest

@pytest.mark.asyncio
async def test_signup_success(client):
    response = await client.post("/auth/signup", json={"username": "alice", "password": "s3cret123"})
    assert response.status_code == 201
    assert response.json()["username"] == "alice"


@pytest.mark.asyncio
async def test_signup_duplicate_username_fails(client):
    await client.post("/auth/signup", json={"username": "alice", "password": "s3cret123"})
    response = await client.post("/auth/signup", json={"username": "alice", "password": "different"})
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_and_access_protected_route(client):
    await client.post("/auth/signup", json={"username": "bob", "password": "s3cret123"})
    # NOTE: the login endpoint expects FORM data (OAuth2PasswordRequestForm), not JSON — use `data=`, not `json=`
    login_response = await client.post("/auth/login", data={"username": "bob", "password": "s3cret123"})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    me_response = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    assert me_response.json()["username"] == "bob"


@pytest.mark.asyncio
async def test_login_wrong_password_fails(client):
    await client.post("/auth/signup", json={"username": "carol", "password": "s3cret123"})
    response = await client.post("/auth/login", data={"username": "carol", "password": "wrongpassword"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_without_token_fails(client):
    response = await client.get("/auth/me")
    assert response.status_code == 401
```

### Step 3 — Testing Products, using the "already logged in" fixture

```python
# test_products.py
import pytest

@pytest.mark.asyncio
async def test_create_and_read_product(authenticated_client):
    create_response = await authenticated_client.post(
        "/products/", json={"name": "Widget", "price": 9.99, "cost_price": 4.50}
    )
    assert create_response.status_code == 201
    product_id = create_response.json()["id"]

    read_response = await authenticated_client.get(f"/products/{product_id}")
    assert read_response.status_code == 200
    assert read_response.json()["name"] == "Widget"
    assert "cost_price" not in read_response.json()   # Chapter 6's input/output separation, now verified by a test
```

This test doesn't re-verify anything about *how* authentication works — it uses `authenticated_client`, which bypassed real token verification via a dependency override, precisely so this test can focus on what it's actually about: product creation and Chapter 6's `cost_price` hiding, verified concretely rather than just asserted in prose.

### Step 4 — Mocking the email side effect correctly

```python
# test_signup_email.py
import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_signup_triggers_welcome_email(client):
    with patch("routers.auth.send_welcome_email") as mock_send:
        response = await client.post("/auth/signup", json={"username": "dana", "password": "s3cret123"})
        assert response.status_code == 201
        mock_send.assert_called_once()
```

As in Chapter 13, both `TestClient` and `AsyncClient` (via `ASGITransport`) run the ASGI app's full request cycle — background tasks included — to completion before control returns to your test, so `mock_send.assert_called_once()` works immediately here with no waiting or polling needed, even though `send_welcome_email` is queued as a background task rather than awaited directly by the route.

---

## Practice Exercises

**Exercise 15.1 — Full coverage for role-gating: happy path plus two failures.**
Write three tests for the `/auth/admin/stats` route from Chapter 11: (a) a user with `role="admin"` succeeds; (b) a user with `role="user"` gets `403`; (c) a request with no `Authorization` header at all gets `401`. Use the `authenticated_client` fixture pattern (or a variant of it) to control the test user's role directly, rather than going through real signup for each case.

**Exercise 15.2 — Prove the per-test database isolation actually works.**
Write two separate test functions that each create a product named `"Isolation Test Widget"` and then assert `GET /products/` returns exactly one product with that name. If the `session`/`client` fixtures were *not* providing genuine per-test isolation (e.g., if they accidentally shared one database across tests), the second test would see two matching products instead of one. Run both tests and confirm each passes independently, proving isolation rather than merely asserting it in a docstring.

**Exercise 15.3 — Patch the wrong target first, on purpose.**
Rewrite Step 4's email test to instead patch `"tasks.send_welcome_email"` (the *original* definition location, not where `routers.auth` looks it up). Run it and observe: does `mock_send.assert_called_once()` pass or fail? Explain exactly why, using section 15.5's naming-vs-binding explanation, then fix it back to the correct target and confirm it passes.

**Exercise 15.4 — Turn Chapter 9's sync-vs-async benchmark into a real, repeatable test.**
Using `asyncio.gather` and the `client` fixture, write a test that fires 10 concurrent requests at Chapter 9 Exercise 9.3's `/products/slow-async` endpoint and asserts the total wall-clock time is under some generous threshold (e.g., under 2 seconds for 10 requests each with a 0.5s simulated delay — well below the ~5 seconds the *sync* version would take). Explain why this threshold needs to be generous rather than tight, given that test execution environments (especially CI runners) can have unpredictable, variable overhead.

**Exercise 15.5 (stretch) — Remove `StaticPool` and watch it fail.**
Temporarily remove `poolclass=StaticPool` from the `session` fixture's engine creation. Run the existing test suite and observe the failure — what specific error do you get, and on which operation? Explain, using section 15.4's explanation, precisely why creating the tables succeeds but a subsequent query fails.

---

## Solutions & Discussion

<details>
<summary>Exercise 15.1</summary>

```python
@pytest_asyncio.fixture
async def admin_client(client, session):
    admin_user = UserTable(username="admin1", hashed_password="unused", role="admin", disabled=False)
    session.add(admin_user)
    await session.commit()

    async def override():
        return admin_user

    app.dependency_overrides[get_current_active_user] = override
    yield client
    del app.dependency_overrides[get_current_active_user]


@pytest.mark.asyncio
async def test_admin_can_access_admin_stats(admin_client):
    response = await admin_client.get("/auth/admin/stats")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_regular_user_forbidden_from_admin_stats(authenticated_client):
    response = await authenticated_client.get("/auth/admin/stats")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_request_to_admin_stats(client):
    response = await client.get("/auth/admin/stats")
    assert response.status_code == 401
```

Three fixtures, three distinct authorization states (admin, authenticated-but-wrong-role, unauthenticated), each producing a different, specific status code — exactly mirroring Chapter 11's own manual verification (Exercise 11.4), now automated and repeatable on every test run rather than something you checked once by hand.
</details>

<details>
<summary>Exercise 15.2</summary>

```python
@pytest.mark.asyncio
async def test_isolation_a(authenticated_client):
    await authenticated_client.post("/products/", json={"name": "Isolation Test Widget", "price": 1, "cost_price": 1})
    response = await authenticated_client.get("/products/")
    matching = [p for p in response.json() if p["name"] == "Isolation Test Widget"]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_isolation_b(authenticated_client):
    await authenticated_client.post("/products/", json={"name": "Isolation Test Widget", "price": 2, "cost_price": 2})
    response = await authenticated_client.get("/products/")
    matching = [p for p in response.json() if p["name"] == "Isolation Test Widget"]
    assert len(matching) == 1
```

Both tests pass independently. If `session`/`client` were accidentally session-scoped (shared across the whole test run) rather than function-scoped, `test_isolation_b` would see **two** matching products — the one it just created, plus the leftover one from `test_isolation_a` — and fail its `len(matching) == 1` assertion. Passing here is direct evidence of isolation, not just a description of it.
</details>

<details>
<summary>Exercise 15.3</summary>

```python
with patch("tasks.send_welcome_email") as mock_send:
    response = await client.post("/auth/signup", json={"username": "erin", "password": "s3cret123"})
    assert response.status_code == 201
    mock_send.assert_called_once()   # FAILS — AssertionError: expected call not found
```

This fails with an `AssertionError` — the mock was never called. `routers/auth.py`'s `from tasks import send_welcome_email` already bound its own `send_welcome_email` name to the original function object at import time; patching `tasks.send_welcome_email` afterward only replaces the attribute on the `tasks` module itself, which `routers.auth` never looks at again after that initial import — `routers.auth`'s own reference is untouched, so the *real* `send_welcome_email` (with its real 2-second `time.sleep`) actually runs during this test, unmocked, while the mock object sits unused. Patching `"routers.auth.send_welcome_email"` instead fixes it, because that's the actual namespace the route code reads the name from at call time — exactly section 15.5's rule, now verified by watching the wrong version fail first.
</details>

<details>
<summary>Exercise 15.4</summary>

```python
import asyncio
import time
import pytest

@pytest.mark.asyncio
async def test_async_endpoint_handles_concurrency_well(client):
    start = time.perf_counter()
    await asyncio.gather(*[client.get("/products/slow-async") for _ in range(10)])
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0   # generous — sync version would take closer to 5s for the same 10 requests
```

The threshold is deliberately generous (2 seconds, when the "should" time is closer to 0.5–1s) rather than tight, because test environments — particularly shared CI runners — have real, variable overhead unrelated to your application's own logic: other processes competing for CPU, scheduling jitter, cold caches. A tight assertion (`elapsed < 0.6`) risks becoming a *flaky* test — one that fails intermittently for reasons that have nothing to do with an actual regression in your code, which trains people to distrust and eventually ignore test failures altogether. The generous threshold here is specifically calibrated to still clearly distinguish "the async version is working as intended" from "someone accidentally reintroduced Chapter 9's blocking-sync-driver bug," which would push elapsed time up toward 5 seconds — nowhere near the 2-second threshold, leaving a wide, reliable margin in both directions.
</details>

<details>
<summary>Exercise 15.5</summary>

Without `poolclass=StaticPool`, table creation (`await conn.run_sync(SQLModel.metadata.create_all)`) succeeds — it runs against whichever connection the pool happens to open for that operation, and creates tables on *that* connection's in-memory database. The failure shows up on the very next database operation that happens to acquire a *different* connection from the pool (which, for `:memory:` SQLite without `StaticPool`, is entirely possible even within what looks like "one test") — typically something like `sqlite3.OperationalError: no such table: products`, because that new connection is looking at its own separate, genuinely empty in-memory database, one that never had `create_all` run against it at all. This is precisely section 15.4's explanation, now reproduced directly: the tables aren't "corrupted" or "missing" in any normal sense, they simply were never created on *this particular* connection, because `:memory:` SQLite databases don't share state across connections the way a real file-based or networked database would.
</details>

---

## Chapter Summary

- `TestClient` (sync) and `httpx.AsyncClient` via `ASGITransport` (async) both test your app in-process, with no real network involved — choose `AsyncClient` when your fixtures are already async, as this chapter's database fixtures are.
- `app.dependency_overrides` substitutes test-only replacements for any `Depends(...)`-based dependency, transparently to route code — always clear overrides after each test, ideally via fixture teardown.
- Pytest fixtures are Chapter 8's `yield`-based dependency pattern, applied to tests instead of routes — the same mental model, a different context.
- In-memory SQLite requires `poolclass=StaticPool` to avoid each new connection silently getting its own separate, empty database — a real, common gotcha, not a rare edge case.
- `unittest.mock.patch` must target the name *where it's looked up at call time*, not where it was originally defined — patching the wrong location produces a test that silently exercises real, unmocked code while the mock assertion fails or (worse) never gets checked at all.

**This closes Part II.** You can now build, secure, and verify a real, database-backed, authenticated API with a genuine test suite. Part III begins in Chapter 16 with advanced Pydantic — generics, `pydantic-settings`, and discriminated unions — the first of seven chapters (through Chapter 22) covering what a codebase needs once it's headed toward production rather than just working correctly.
