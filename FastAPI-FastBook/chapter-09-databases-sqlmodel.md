# Chapter 9: Databases I — SQLModel and Async SQLAlchemy 2.0

> Part II — Intermediate: Building Real APIs · Chapter 9 of 28

Every route so far has stored data in a plain Python `dict` that vanishes the moment the process restarts. This chapter replaces it with a real, persistent, async database — using SQLModel for the table definitions and async SQLAlchemy 2.0 underneath, wired in through exactly the `yield`-based dependency pattern Chapter 8 just built.

## Learning Objectives

By the end of this chapter you will be able to:

- Explain what SQLModel actually is: a single class definition that serves simultaneously as a Pydantic model and a SQLAlchemy ORM table mapping.
- Explain the difference between sync and async database engines/sessions, and why using a sync driver inside an `async def` route reproduces exactly the blocking bug from Chapter 2.
- Manage a database session's lifecycle correctly per request, using a `yield`-based dependency.
- Explain the unit-of-work pattern (`add`/`commit`/`refresh`) and a session's identity map.
- Explain connection pooling at a basic level, and why the engine is created once, not per request.
- Wire a real SQLite (and, with one line changed, Postgres) database into the Products API from Chapters 6–8.

---

## 9.1 What SQLModel Actually Is

SQLModel — from the same author as FastAPI — solves a specific redundancy: without it, you'd typically write one Pydantic model for your API's request/response shape *and* a separate SQLAlchemy model for the database table, keeping their fields in sync by hand. SQLModel lets a single class be both, controlled by one flag:

```python
from sqlmodel import SQLModel, Field

class ProductTable(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    price: float
```

`table=True` is the entire mechanism: with it, `ProductTable` is a real, mapped SQLAlchemy table (it participates in queries, inserts, foreign keys); without it, a `SQLModel` subclass behaves like a plain Pydantic model (validation only, no table). This chapter uses `table=True` classes purely as the database layer — worth being explicit about, given Chapter 6 spent real effort separating `ProductCreate`/`ProductUpdate`/`ProductPublic` for API purposes: **those Pydantic schema models remain separate from `ProductTable`.** SQLModel makes it *possible* to use one class for both jobs, but doing so would re-couple exactly what Chapter 6 deliberately decoupled — your database's shape (auto-increment IDs, internal-only columns, storage-specific constraints) is not automatically the right shape for a public API contract. This curriculum keeps them distinct: `ProductTable` (SQLModel, `table=True`) for the database; `ProductCreate`/`ProductUpdate`/`ProductPublic` (plain Pydantic, from Chapter 6) for the API surface.

## 9.2 Sync vs Async Engines and Drivers

Chapter 2 established that a blocking call inside `async def` freezes the entire event loop, not just its own task. Database access is the single most common place this bug actually shows up in production FastAPI code, because it's easy to reach for a familiar synchronous driver without noticing the trap.

| | Sync | Async |
|---|---|---|
| Engine constructor | `create_engine(...)` | `create_async_engine(...)` |
| Session type | `Session` | `AsyncSession` |
| SQLite driver | `sqlite3` (built in) — DSN: `sqlite:///./app.db` | `aiosqlite` — DSN: `sqlite+aiosqlite:///./app.db` |
| Postgres driver | `psycopg2`/`psycopg` — DSN: `postgresql://...` | `asyncpg` — DSN: `postgresql+asyncpg://...` |

The DSN's driver segment (`+aiosqlite`, `+asyncpg`) is what actually selects the underlying driver — SQLAlchemy uses it to decide which library does the real talking to the database. Using the sync driver (`sqlite:///./app.db`, no `+aiosqlite`) inside an `async def` route doesn't raise an error — it silently works, and silently blocks the entire event loop on every query, exactly like Chapter 2's `time.sleep` inside `async def`. This curriculum uses async drivers throughout for this exact reason, and Exercise 9.3 has you reproduce the blocking version on purpose, under real concurrent load, so the cost isn't just theoretical.

## 9.3 Sessions and the Unit-of-Work Pattern

A `Session` (or `AsyncSession`) is not a raw connection — it's a **unit of work**: an object that tracks every change you make to the Python objects it knows about, and applies them to the database as a single, deliberate operation, only when you tell it to.

```python
db_product = ProductTable(name="Widget", price=9.99, cost_price=4.50)
session.add(db_product)        # staged, not yet in the database
await session.commit()          # now it's actually written
await session.refresh(db_product)  # reload db_product with anything the DB itself generated (like id)
```

`session.add(...)` merely *stages* the object — nothing reaches the database until `commit()`. This matters because a session can stage several related changes (creating a product, adjusting inventory, logging an audit entry) and commit them all atomically, as one transaction — either all of it lands, or none of it does, if something fails partway through. `session.refresh(...)` is necessary specifically because `id` (and anything else the database itself assigns, like a default timestamp) doesn't exist on your Python object until the database has actually generated it — `refresh` reloads the object's attributes from the database after the insert, so `db_product.id` is populated correctly afterward.

A session also maintains an **identity map**: within one session, asking for the same row twice returns the *same* Python object, not two separate copies that happen to have equal data:

```python
p1 = await session.get(ProductTable, 1)
p2 = await session.get(ProductTable, 1)
print(p1 is p2)   # True
```

This matters once relationships enter the picture (Exercise 9.1): if two different code paths within one request both load "product 1," you want them modifying the *same* in-memory object, not two independent copies that could silently diverge before either gets committed.

## 9.4 Connection Pooling, Briefly

Opening a fresh database connection is not free — it involves a TCP handshake, authentication, and (for some databases) session setup, all of which take real time. Doing that on *every single request* would make your API's latency dominated by connection overhead rather than actual query time. Instead, the **engine** — created exactly once, when your application starts — maintains a **pool** of already-open connections, handed out to sessions as needed and returned to the pool when a session closes, rather than being closed and reopened from scratch.

```mermaid
flowchart LR
    subgraph Engine["Engine — created once, at startup"]
        Pool["Connection Pool"]
    end
    Pool -->|"checked out"| S1["Session — request 1"]
    Pool -->|"checked out"| S2["Session — request 2"]
    S1 -->|"session closes,<br/>connection returned"| Pool
    S2 -->|"session closes,<br/>connection returned"| Pool
```

Practically, this is the reason the **engine is created once, at module level or in a startup hook — never per request** — while a **session is created fresh per request** (via the `yield`-based dependency you're about to build) and closed at the end of that request, returning its connection to the pool rather than destroying it. Getting this backwards (creating a new engine per request) would silently defeat pooling entirely, recreating the exact overhead it exists to avoid.

---

## Hands-On Project: A Real Database for the Products API

### Step 1 — Install and set up the engine and session dependency

```bash
uv pip install sqlmodel aiosqlite
```

```python
# database.py
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

DATABASE_URL = "sqlite+aiosqlite:///./app.db"

engine = create_async_engine(DATABASE_URL, echo=True)

# expire_on_commit=False: without this, accessing an attribute on a committed
# object triggers an implicit reload — which, in async SQLAlchemy, requires an
# explicit await and cannot happen silently. Setting it False avoids that trap
# entirely for the common case of "commit, then return the object as-is."
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def get_session():
    async with async_session_maker() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
```

`get_session` is precisely Chapter 8's `yield`-based dependency pattern: the `async with` block opens a session before your route runs, and closes it automatically afterward — success or failure — the moment the `async with` block exits.

### Step 2 — The table model

```python
# models.py
from datetime import datetime
from sqlmodel import SQLModel, Field

class ProductTable(SQLModel, table=True):
    __tablename__ = "products"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    price: float
    currency: str = "USD"
    in_stock: bool = True
    cost_price: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Step 3 — The Pydantic API schema, now bridging from an ORM object

```python
# schemas.py
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class ProductCreate(BaseModel):
    name: str
    price: float
    currency: str = "USD"
    in_stock: bool = True
    cost_price: float

class ProductUpdate(BaseModel):
    name: str | None = None
    price: float | None = None
    currency: str | None = None
    in_stock: bool | None = None

class ProductPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)   # <-- the bridge promised back in Chapter 5

    id: int
    name: str
    price: float
    currency: str
    in_stock: bool
    created_at: datetime
```

Recall Chapter 5.4 described `from_attributes=True` as "your bridge to ORMs" without yet having an ORM to bridge to. This is that moment: without it, `response_model=ProductPublic` would only know how to validate a *dict*; with it, FastAPI can validate directly against a `ProductTable` instance's attributes, which is exactly what every route below returns.

### Step 4 — Wire the database into app startup via `lifespan`

```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from database import init_db
from routers import products

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()   # runs once, before the app starts accepting requests
    yield              # (nothing needed on shutdown yet — more on this in Chapter 24)

app = FastAPI(title="Products API", lifespan=lifespan)
app.include_router(products.router)
```

`lifespan` is the same `yield`-based setup/teardown idea from Chapter 8, just scoped to the *entire application's* lifetime rather than a single request — code before `yield` runs once at startup, code after it (none needed yet here) would run once at shutdown.

### Step 5 — The routes, using a real async session

```python
# routers/products.py
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_session
from models import ProductTable
from schemas import ProductCreate, ProductUpdate, ProductPublic

router = APIRouter(prefix="/products", tags=["products"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.post("/", response_model=ProductPublic, status_code=status.HTTP_201_CREATED)
async def create_product(product: ProductCreate, session: SessionDep):
    db_product = ProductTable(**product.model_dump())
    session.add(db_product)
    await session.commit()
    await session.refresh(db_product)
    return db_product


@router.get("/{product_id}", response_model=ProductPublic)
async def read_product(product_id: int, session: SessionDep):
    db_product = await session.get(ProductTable, product_id)
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return db_product


@router.get("/", response_model=list[ProductPublic])
async def list_products(session: SessionDep, limit: int = 20, offset: int = 0):
    result = await session.execute(select(ProductTable).offset(offset).limit(limit))
    return result.scalars().all()


@router.patch("/{product_id}", response_model=ProductPublic)
async def update_product(product_id: int, update: ProductUpdate, session: SessionDep):
    db_product = await session.get(ProductTable, product_id)
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(db_product, field, value)
    session.add(db_product)
    await session.commit()
    await session.refresh(db_product)
    return db_product


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(product_id: int, session: SessionDep):
    db_product = await session.get(ProductTable, product_id)
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    await session.delete(db_product)
    await session.commit()
```

`Annotated[AsyncSession, Depends(get_session)]` aliased to `SessionDep` is a small but genuinely useful habit — every route needing the database repeats one short name instead of the full `Annotated[...]` expression, without losing any of Chapter 8's dependency machinery underneath.

Run `fastapi dev main.py`, exercise every route exactly as you did with the in-memory version in Chapters 6–7, then stop the server, restart it, and confirm your products are still there — the entire point of this chapter, made concrete.

### Step 6 — Swap to Postgres, changing exactly one line

```bash
docker run -d --name products-postgres \
  -e POSTGRES_PASSWORD=devpassword -e POSTGRES_DB=products \
  -p 5432:5432 postgres:16

uv pip install asyncpg
```

```python
# database.py
DATABASE_URL = "postgresql+asyncpg://postgres:devpassword@localhost:5432/products"
```

Every route, every session-handling line, every query — completely unchanged. This is the practical payoff of SQLAlchemy's engine abstraction: your application code talks to "an async database," not specifically to SQLite or specifically to Postgres, and switching which one backs it in development versus production is a configuration change, not a rewrite.

---

## Practice Exercises

**Exercise 9.1 — A one-to-many relationship: Product → Reviews.**
Add a `ReviewTable(SQLModel, table=True)` with `id`, `product_id: int = Field(foreign_key="products.id")`, `rating: int`, and `comment: str | None`. Add a `reviews: list["ReviewTable"] = Relationship(back_populates="product")` to `ProductTable`, and a matching `product: ProductTable | None = Relationship(back_populates="reviews")` on `ReviewTable`. Add a `POST /products/{product_id}/reviews` endpoint to create one, and extend `GET /products/{product_id}` to eagerly load and include its reviews using `selectinload` (`select(ProductTable).where(ProductTable.id == product_id).options(selectinload(ProductTable.reviews))`) rather than accessing `.reviews` directly off a plain `session.get(...)` result. Confirm a fresh product with no reviews returns an empty list rather than erroring.

**Exercise 9.2 — A migration, written by hand.**
Without Alembic (that's Chapter 10), add a `discount_pct: float = 0.0` field to `ProductTable`, then write a short standalone script using `engine.begin()` and `conn.execute(text("ALTER TABLE products ADD COLUMN discount_pct FLOAT DEFAULT 0"))` to bring your *existing* SQLite database's schema in line with the updated model, without losing any existing rows. Run it against a database that already has products in it, and confirm both that existing rows now show `discount_pct: 0.0` and that the application still starts and queries correctly afterward.

**Exercise 9.3 — Benchmark sync vs async DB drivers under real concurrent load.**
Build two versions of a deliberately slow endpoint: `GET /products/slow-sync`, using a *sync* engine/session (`create_engine`, `sqlite:///./app.db`, no `+aiosqlite`) inside an `async def` route, doing a query wrapped in an artificial delay; and `GET /products/slow-async`, using the real async session from this chapter, with an equivalent artificial delay via `await asyncio.sleep(...)`. Using `httpx.AsyncClient` and `asyncio.gather`, fire 10 concurrent requests at each endpoint and compare total wall-clock time. Explain the result using Chapter 2's event-loop model, not just "one was faster."

**Exercise 9.4 — Identity map, within and across requests.**
Inside a single route, call `await session.get(ProductTable, 1)` twice and confirm (with `is`) they're the same object. Then make two genuinely separate HTTP requests to `GET /products/1` and, using a debugger or a print statement inside the route, confirm the `ProductTable` instances involved are *not* the same Python object across those two requests. Explain why, tying your answer back to how `get_session` creates a fresh session (via `async with async_session_maker()`) on every single request.

**Exercise 9.5 (stretch) — What `expire_on_commit=False` is actually preventing.**
Temporarily remove `expire_on_commit=False` from `async_session_maker(...)` (i.e., let it default to `True`). Call `create_product`, then — still within the same route, after `await session.commit()` — try to access `db_product.name` again before returning it. Run it and read whatever error or behavior results. Explain, based on what you observe, why `expire_on_commit=False` was set deliberately in Step 1, rather than being an arbitrary default this curriculum happened to choose.

---

## Solutions & Discussion

<details>
<summary>Exercise 9.1</summary>

```python
# models.py
from sqlmodel import SQLModel, Field, Relationship

class ReviewTable(SQLModel, table=True):
    __tablename__ = "reviews"
    id: int | None = Field(default=None, primary_key=True)
    product_id: int = Field(foreign_key="products.id")
    rating: int
    comment: str | None = None
    product: "ProductTable" = Relationship(back_populates="reviews")


class ProductTable(SQLModel, table=True):
    __tablename__ = "products"
    id: int | None = Field(default=None, primary_key=True)
    name: str
    price: float
    currency: str = "USD"
    in_stock: bool = True
    cost_price: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviews: list[ReviewTable] = Relationship(back_populates="product")
```

```python
from sqlalchemy.orm import selectinload

@router.get("/{product_id}/with-reviews")
async def read_product_with_reviews(product_id: int, session: SessionDep):
    result = await session.execute(
        select(ProductTable)
        .where(ProductTable.id == product_id)
        .options(selectinload(ProductTable.reviews))
    )
    db_product = result.scalar_one_or_none()
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"id": db_product.id, "name": db_product.name, "reviews": db_product.reviews}
```

`selectinload` is deliberate here rather than accidental: in async SQLAlchemy, accessing a relationship attribute that *wasn't* eagerly loaded (e.g., calling `.reviews` on an object fetched via plain `session.get(...)`) triggers an implicit lazy-load — which, outside of an explicit `await`, raises a `MissingGreenlet`/`greenlet_spawn has not been called` error, a well-known async SQLAlchemy footgun. `selectinload` avoids it entirely by fetching the relationship as part of the *original* query, so `.reviews` is already populated by the time you touch it. Chapter 10 goes deeper on this exact category of problem (the N+1 query pattern), so treat this as the minimum you need to know for now: relationships you plan to access need to be eagerly loaded, explicitly, in async code.
</details>

<details>
<summary>Exercise 9.2</summary>

```python
# migrate_add_discount.py
import asyncio
from sqlalchemy import text
from database import engine

async def migrate():
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE products ADD COLUMN discount_pct FLOAT DEFAULT 0"))
    print("Migration applied.")

if __name__ == "__main__":
    asyncio.run(migrate())
```

Run once against your existing `app.db`. Existing rows pick up `discount_pct = 0.0` automatically (the column's `DEFAULT 0` applies retroactively to already-existing rows in SQLite's `ALTER TABLE ADD COLUMN`), and new inserts through the ORM now populate it via the updated `ProductTable` model. This exercise is deliberately tedious on purpose — hand-writing and manually running a `.py` script per schema change doesn't scale past a handful of changes, has no record of *which* migrations have already been applied to a given database, and has no clean rollback story. That gap is exactly what Chapter 10 (Alembic) exists to close.
</details>

<details>
<summary>Exercise 9.3</summary>

```python
# a deliberately blocking route
from sqlalchemy import create_engine, text
sync_engine = create_engine("sqlite:///./app.db")

@router.get("/slow-sync")
async def slow_sync():
    with sync_engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        import time; time.sleep(0.5)   # simulating slow I/O with a sync, blocking call
    return {"ok": True}

@router.get("/slow-async")
async def slow_async(session: SessionDep):
    await session.execute(select(1))
    await asyncio.sleep(0.5)            # simulating the same delay, but non-blocking
    return {"ok": True}
```

```python
import asyncio, time, httpx

async def hit(client, path):
    return await client.get(path)

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        for path in ("/products/slow-sync", "/products/slow-async"):
            start = time.perf_counter()
            await asyncio.gather(*[hit(client, path) for _ in range(10)])
            print(f"{path}: {time.perf_counter() - start:.2f}s")

asyncio.run(main())
```

Expect `/slow-sync` to take roughly **5 seconds** (10 × 0.5s, serialized) and `/slow-async` to take roughly **0.5–1 second** (all 10 overlapping). This is precisely Chapter 2's Part C bug, reproduced with a real (if simulated) database call instead of `time.sleep`: the sync engine's blocking call inside `async def` freezes the single-threaded event loop for its full duration, so every other concurrent request queues up behind it one at a time, exactly as if the whole server had only one worker regardless of how many requests arrived "concurrently."
</details>

<details>
<summary>Exercise 9.4</summary>

Within one route:
```python
p1 = await session.get(ProductTable, 1)
p2 = await session.get(ProductTable, 1)
print(p1 is p2)   # True
```

Across two separate HTTP requests, the objects are different Python objects (different `id()` values in the CPython sense, unrelated to the product's own `id` field), even though both represent "product 1." The reason is `get_session`'s `async with async_session_maker() as session: yield session` — this creates a **brand-new session, from scratch, for every single incoming request**, and a session's identity map only exists for the lifetime of that one session. Request A's session and Request B's session have no knowledge of each other whatsoever; each independently queries the database and constructs its own fresh Python object to represent the same underlying row. This is consistent with — and actually a specific instance of — Chapter 8's "dependency caching is per-request, not global" rule: `get_session` itself isn't cached across requests either, by design, since a fresh session per request is precisely the correct lifecycle for a database session.
</details>

<details>
<summary>Exercise 9.5</summary>

With `expire_on_commit` at its default (`True`), calling `await session.commit()` marks every attribute on every object the session is tracking as *expired* — meaning the next time you access one of those attributes, SQLAlchemy tries to transparently reload it from the database to guarantee you're seeing post-commit state. In synchronous SQLAlchemy, that reload happens invisibly, mid-attribute-access, with no `await` needed. In *async* SQLAlchemy, that same reload would require an actual database round trip — which fundamentally requires an `await` — but a plain attribute access (`db_product.name`) has no way to be `await`-ed; it's just Python attribute lookup syntax. The result is exactly the `MissingGreenlet`/`greenlet_spawn has not been called` error mentioned in Exercise 9.1's solution: SQLAlchemy needs to run async code to satisfy the access, but nothing in a bare `db_product.name` expression gives it the chance to do so.

`expire_on_commit=False` avoids this entirely by simply *not* expiring attributes after commit — you keep using the in-memory values you already have, which is exactly the behavior every route in this chapter's hands-on project relies on (`return db_product` immediately after `commit()`/`refresh()`, with no further awaited reload needed). This is why the setting is deliberate rather than incidental: it's close to a required setting for the common "commit, then return the object" pattern in async SQLAlchemy, not a stylistic preference.
</details>

---

## Chapter Summary

- SQLModel's `table=True` flag is the entire mechanism turning a class into both a Pydantic model and a mapped SQLAlchemy table — but API schema models (`ProductCreate`/`ProductPublic`/`ProductUpdate`) stay separate from table models, preserving Chapter 6's input/output separation rather than undoing it.
- A sync driver inside an `async def` route blocks the entire event loop exactly like Chapter 2's `time.sleep` — the DSN's driver segment (`+aiosqlite`, `+asyncpg`) is what actually determines whether you're safe.
- A session is a unit of work: `add()` stages, `commit()` persists, `refresh()` reloads database-generated values — and its identity map guarantees the same row maps to the same Python object *within one session*, but never across separate requests.
- The engine (and its connection pool) is created once, at startup; a fresh session is created per request via a `yield`-based dependency, closed at the end of that request.
- `ConfigDict(from_attributes=True)` — introduced without an ORM to use it on back in Chapter 5 — is exactly what lets `response_model` validate directly against a returned `ProductTable` instance's attributes, instead of requiring a dict.

**Next:** Chapter 10 addresses what this chapter deliberately left rough: hand-written migrations that don't scale, and the N+1 query problem previewed in Exercise 9.1's solution — introducing Alembic and a proper repository layer for separating query logic from route logic.
