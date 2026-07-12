# Chapter 10: Databases II — Migrations, Relationships, and Query Patterns

> Part II — Intermediate: Building Real APIs · Chapter 10 of 28

Chapter 9 got a real database working, with one hand-written migration script (Exercise 9.2) and a relationship loaded just carefully enough to avoid an async footgun (Exercise 9.1). This chapter replaces the hand-written migration with Alembic, names and fixes the N+1 query problem properly, covers transactions with real rollback, and introduces the repository pattern — the structural fix for routes that are starting to know too much about SQL.

## Learning Objectives

By the end of this chapter you will be able to:

- Set up Alembic for a SQLModel-based project and generate/apply versioned migrations, including ones Alembic can autogenerate from model changes.
- Explain the N+1 query problem precisely, demonstrate it happening, and fix it with `selectinload`.
- Wrap multi-step database operations in an explicit transaction that rolls back everything on failure, not just the step that failed.
- Explain what the repository pattern is, and why it's the natural next step once routes start containing real query logic.

---

## 10.1 Alembic — Migrations That Actually Scale

Exercise 9.2's hand-written migration script worked, but it has three real problems the moment more than one person or more than one schema change is involved: nothing records *which* migrations have already been applied to a given database, there's no systematic way to undo one, and two developers making schema changes independently have no mechanism to reconcile their changes into a single, ordered history. Alembic solves exactly this: versioned, ordered migration scripts, each capable of both an `upgrade()` and a `downgrade()`, with a table in your database (`alembic_version`) tracking exactly which migration that specific database is currently at.

```bash
uv pip install alembic
alembic init migrations
```

This creates a `migrations/` directory and an `alembic.ini` file. The key file to edit is `migrations/env.py`, which needs to know about your models' metadata:

```python
# migrations/env.py (relevant excerpt)
from sqlmodel import SQLModel
import models  # noqa: F401 — importing this registers every table class with SQLModel.metadata

target_metadata = SQLModel.metadata
```

One pragmatic choice worth calling out explicitly: **Alembic itself can use a plain synchronous connection, even though your running application uses an async one.** Migrations are administrative, one-off operations run during development or deployment — not per-request hot-path code — so there's no correctness or performance reason to fight with Alembic's (more involved) async configuration template. This curriculum points Alembic's `sqlalchemy.url` at the sync driver (`sqlite:///./app.db`, no `+aiosqlite`) purely for migration purposes, while the application itself continues using the async driver for everything request-related. (Alembic does support fully async migration environments, for teams with a reason to need one — worth knowing the option exists, out of scope for this curriculum.)

With models already defined (`ProductTable`, `ReviewTable` from Chapter 9), Alembic can generate a migration by comparing your models against the current database schema:

```bash
alembic revision --autogenerate -m "create products and reviews tables"
alembic upgrade head
```

`--autogenerate` inspects `target_metadata` and the actual database, and writes out the `upgrade()`/`downgrade()` operations needed to reconcile them — you should always read the generated script before applying it, since autogenerate is a strong assistant, not an infallible one (it can miss certain kinds of changes, like a column rename, which looks identical to "drop one column, add another" from its point of view).

## 10.2 The N+1 Query Problem

Chapter 9's Exercise 9.1 solution used `selectinload` without fully justifying why. Here's the justification, made concrete. Consider a route that lists 10 products and, for each one, separately queries its reviews:

```python
products = (await session.execute(select(ProductTable).limit(10))).scalars().all()

result = []
for product in products:
    reviews = (
        await session.execute(select(ReviewTable).where(ReviewTable.product_id == product.id))
    ).scalars().all()
    result.append({"product": product, "reviews": reviews})
```

Count the queries: **one** to fetch the 10 products, plus **one more per product** to fetch its reviews — 11 queries total for 10 products. This is the "N+1" pattern by name: 1 query for the parent collection, N additional queries (one per parent row) for the related data — and it scales linearly with the number of products, meaning a page of 100 products issues 101 queries instead of a small, fixed number. It's easy to write exactly this code without noticing the problem, because each individual line looks completely reasonable in isolation.

`selectinload` fixes it by issuing a **second** query — not one per product, but one covering *all* of them at once, using a single `WHERE product_id IN (...)`:

```python
from sqlalchemy.orm import selectinload

result = await session.execute(
    select(ProductTable).limit(10).options(selectinload(ProductTable.reviews))
)
products = result.scalars().all()
# product.reviews is now already populated for every product — no further queries needed
```

Now the total is **two** queries, regardless of whether you fetched 10 products or 10,000 — one for the products, one `IN (...)` query for every review belonging to any of them. You can verify this isn't just a claim by counting queries directly, using a SQLAlchemy event hook:

```python
from sqlalchemy import event

query_count = 0

@event.listens_for(engine.sync_engine, "before_cursor_execute")
def count_queries(conn, cursor, statement, parameters, context, executemany):
    global query_count
    query_count += 1
```

(`engine.sync_engine` — even for an async engine, SQLAlchemy's event system attaches to the underlying sync engine object it wraps; this is the documented way to hook into query execution for an async engine.) You'll use exactly this counter in the hands-on project to watch 11 queries become 2, not just take it on faith.

There's a second eager-loading strategy worth knowing by name, `joinedload`, which fetches everything in a **single** query using a SQL `JOIN` rather than a second, separate query:

```python
from sqlalchemy.orm import joinedload

result = await session.execute(
    select(ProductTable).options(joinedload(ProductTable.reviews))
)
```

The trade-off: a `JOIN` duplicates the parent row once per matching child row in the raw result set (a product with 3 reviews appears 3 times, each paired with a different review, before SQLAlchemy deduplicates it back into one object with a 3-item list on your side) — for a one-to-many relationship where the "many" side can be large, `selectinload`'s two-queries-total approach is usually the better default; `joinedload` earns its keep more clearly on one-to-one or small, bounded relationships, or when you specifically want everything in one round trip. You'll compare the two directly in Exercise 10.5.

## 10.3 Transactions and Rollback

A session's changes aren't really "safe" until `commit()` succeeds — and the value of that is easiest to see with a multi-step operation where the steps depend on each other. Consider transferring stock between two products: decrement one, increment the other. If the second half fails, the first half absolutely must not have taken effect either — otherwise stock has simply vanished.

```python
async def transfer_stock(session: AsyncSession, from_id: int, to_id: int, quantity: int):
    async with session.begin():
        source = await session.get(ProductTable, from_id)
        destination = await session.get(ProductTable, to_id)
        if source is None or destination is None:
            raise NotFoundError("Product", from_id if source is None else to_id)
        if source.stock_qty < quantity:
            raise InsufficientStockError(from_id, quantity, source.stock_qty)

        source.stock_qty -= quantity
        destination.stock_qty += quantity
        # if we reach the end of this block with no exception, everything commits together;
        # if ANY line above raised, everything staged in this block is rolled back together
```

`async with session.begin():` is the key piece of machinery here: it commits everything staged inside the block automatically if the block completes without an exception, and rolls back everything staged inside it automatically if any exception propagates out — you don't write `try`/`except`/`rollback()`/`commit()` by hand at all. This matters specifically because `source.stock_qty -= quantity` and `destination.stock_qty += quantity` are two separate in-memory mutations that only become durable together, at the same `commit()` — there's no window where only one side of the transfer has taken effect in the database, regardless of which line inside the block happens to raise.

`session.flush()` is worth distinguishing from `commit()` here too: `flush()` sends pending SQL to the database (so, for instance, an auto-generated ID becomes available on a freshly-added object) *without* ending the transaction — you can still roll back everything that's been flushed. `commit()` ends the transaction, making everything durable and no longer reversible. You'll use `flush()` in the hands-on project exactly where you need a newly-created row's ID before the transaction as a whole is ready to commit.

## 10.4 The Repository Pattern

Every route so far has mixed two genuinely different concerns in one function body: *how to query the database* (the `select(...)`, `.options(...)`, `session.get(...)` calls) and *how to behave as an HTTP endpoint* (status codes, response models, request validation). This is fine while routes are short. It stops being fine once a route's query logic grows complex enough that you'd want to reuse it elsewhere — a background job (Chapter 13) that needs to look up a product the same way a route does, or a test that wants to check "does this query return the right thing" without spinning up the whole HTTP stack.

The **repository pattern** pulls all of a resource's query logic into its own class, which knows about the database and your domain exceptions (Chapter 7's `NotFoundError` and friends) — and nothing about HTTP, FastAPI, or response models at all:

```python
# repositories/products.py
from sqlmodel import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from models import ProductTable
from exceptions import NotFoundError


class ProductRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_raise(self, product_id: int) -> ProductTable:
        result = await self.session.execute(
            select(ProductTable)
            .where(ProductTable.id == product_id)
            .options(selectinload(ProductTable.reviews), selectinload(ProductTable.tags))
        )
        product = result.scalar_one_or_none()
        if product is None:
            raise NotFoundError("Product", product_id)
        return product

    async def list(self, limit: int, offset: int) -> list[ProductTable]:
        result = await self.session.execute(
            select(ProductTable).options(selectinload(ProductTable.tags)).offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    async def create(self, data: dict) -> ProductTable:
        product = ProductTable(**data)
        self.session.add(product)
        await self.session.commit()
        await self.session.refresh(product)
        return product
```

A route using this repository no longer contains a single `select(...)` statement:

```python
@router.get("/{product_id}", response_model=ProductPublic)
async def read_product(product_id: int, repo: ProductRepoDep):
    return await repo.get_or_raise(product_id)
```

Notice this route doesn't `raise HTTPException(404, ...)` at all — `get_or_raise` raises `NotFoundError`, exactly the exception class Chapter 7 built a global handler for, back before a real database even existed. That handler still applies here, unmodified — the repository pattern didn't require touching Chapter 7's error-handling layer at all, because `NotFoundError` was already framework-agnostic by design. This is the concrete payoff of that earlier decision: a repository, a route, and even a future CLI script or background job can all raise and handle the exact same exception type consistently, because none of them were ever coupled to `HTTPException` in the first place.

---

## Hands-On Project: Alembic, a Many-to-Many Relationship, and a Repository Layer

### Step 1 — Set up Alembic (see section 10.1 for the commands), then generate and apply the first migration covering `ProductTable` and `ReviewTable` from Chapter 9.

### Step 2 — Add a many-to-many relationship: Products ↔ Tags

```python
# models.py (additions)
from sqlmodel import SQLModel, Field, Relationship

class ProductTagLink(SQLModel, table=True):
    __tablename__ = "product_tag_links"
    product_id: int | None = Field(default=None, foreign_key="products.id", primary_key=True)
    tag_id: int | None = Field(default=None, foreign_key="tags.id", primary_key=True)


class TagTable(SQLModel, table=True):
    __tablename__ = "tags"
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    products: list["ProductTable"] = Relationship(back_populates="tags", link_model=ProductTagLink)


class ProductTable(SQLModel, table=True):
    __tablename__ = "products"
    # ...existing fields from Chapter 9...
    tags: list[TagTable] = Relationship(back_populates="products", link_model=ProductTagLink)
```

Generate and apply the migration:

```bash
alembic revision --autogenerate -m "add tags and product_tag_links"
alembic upgrade head
```

### Step 3 — Watch the N+1 problem happen, then fix it, counting queries both times

```python
# using the query-counting event listener from section 10.2
query_count = 0
# ...event.listens_for(...) registered as shown above...

# BAD: N+1
query_count = 0
products = (await session.execute(select(ProductTable).limit(10))).scalars().all()
for p in products:
    _ = (await session.execute(select(ReviewTable).where(ReviewTable.product_id == p.id))).scalars().all()
print(f"N+1 version: {query_count} queries")   # 11, for 10 products

# GOOD: selectinload
query_count = 0
result = await session.execute(
    select(ProductTable).limit(10).options(selectinload(ProductTable.reviews))
)
products = result.scalars().all()
for p in products:
    _ = p.reviews   # already loaded — no additional query
print(f"selectinload version: {query_count} queries")   # 2, regardless of product count
```

### Step 4 — Build `ProductRepository` and refactor every route to use it

Using the repository from section 10.4, wire it in via a dependency:

```python
# dependencies.py (addition)
from typing import Annotated
from fastapi import Depends
from repositories.products import ProductRepository

def get_product_repository(session: SessionDep) -> ProductRepository:
    return ProductRepository(session)

ProductRepoDep = Annotated[ProductRepository, Depends(get_product_repository)]
```

```python
# routers/products.py — now genuinely thin
@router.post("/", response_model=ProductPublic, status_code=201)
async def create_product(product: ProductCreate, repo: ProductRepoDep):
    return await repo.create(product.model_dump())

@router.get("/{product_id}", response_model=ProductPublic)
async def read_product(product_id: int, repo: ProductRepoDep):
    return await repo.get_or_raise(product_id)

@router.get("/", response_model=list[ProductPublic])
async def list_products(repo: ProductRepoDep, limit: int = 20, offset: int = 0):
    return await repo.list(limit=limit, offset=offset)
```

Confirm every route still behaves identically to Chapter 9's version from the outside — the refactor should be invisible to a client, exactly like Chapter 3's `APIRouter` refactor was.

---

## Practice Exercises

**Exercise 10.1 — N+1 on a different relationship, from scratch.**
Without looking back at the hands-on project's code, write a fresh N+1 query against `ProductTable.tags` (not `.reviews`) — a loop that fetches 10 products, then separately queries each one's tags. Count the queries using the event-listener technique, confirm it's 11, then fix it with `selectinload(ProductTable.tags)` and confirm it drops to 2.

**Exercise 10.2 — A safe nullable-column migration.**
Add `notes: str | None = None` to `ProductTable`, then generate and apply the Alembic migration. Explain, in your own words, why declaring this new column as **nullable** (rather than a required `str`) is what makes the migration safe to run against a table that already has existing rows — what would happen if you instead added `notes: str` (required, no default) and ran `alembic revision --autogenerate` against existing data?

**Exercise 10.3 — A transaction that must roll back correctly.**
Implement `transfer_stock(session, from_id, to_id, quantity)` from section 10.3 for real (add a `stock_qty: int = 0` field to `ProductTable` first). Deliberately call it with a `to_id` that doesn't exist, and confirm — by re-querying `from_id` afterward — that its `stock_qty` was **not** decremented, even though the decrement line ran before the missing-destination check failed. Then repeat with valid IDs and confirm both sides update together.

**Exercise 10.4 — Extend the repository with a tag-attaching method.**
Add an `attach_tag(product_id: int, tag_name: str) -> ProductTable` method to `ProductRepository` that finds-or-creates a `TagTable` row by name and adds it to the given product's `tags` relationship, committing the change. Add a thin `POST /products/{product_id}/tags` route that calls it. Confirm the route contains no `select(...)` statements of its own.

**Exercise 10.5 (stretch) — `selectinload` vs `joinedload`, side by side.**
Run the same "10 products with reviews" query twice — once with `selectinload(ProductTable.reviews)`, once with `joinedload(ProductTable.reviews)` — and use the query-counting event listener plus `print(result)` (or `.all()` length) to compare: how many queries did each strategy issue, and how many raw rows came back before SQLAlchemy deduplicated them into your final Python objects? Write two sentences on when you'd pick one over the other for a real endpoint.

---

## Solutions & Discussion

<details>
<summary>Exercise 10.1</summary>

```python
query_count = 0
products = (await session.execute(select(ProductTable).limit(10))).scalars().all()
for p in products:
    _ = (await session.execute(select(TagTable).join(ProductTagLink).where(ProductTagLink.product_id == p.id))).scalars().all()
print(query_count)   # 11

query_count = 0
result = await session.execute(select(ProductTable).limit(10).options(selectinload(ProductTable.tags)))
products = result.scalars().all()
for p in products:
    _ = p.tags   # already loaded
print(query_count)   # 2
```

Same pattern, same fix, different relationship — confirming the N+1 problem and `selectinload`'s solution to it aren't specific to reviews, they apply to *any* one-to-many (or many-to-many, as here) relationship accessed in a loop.
</details>

<details>
<summary>Exercise 10.2</summary>

Declaring `notes: str | None = None` produces a migration that adds a **nullable** column with no `NOT NULL` constraint — every existing row simply gets `NULL` for `notes`, which is a valid value for a nullable column, so the migration succeeds instantly regardless of how many rows already exist.

Declaring `notes: str` (required, no default) instead would generate a migration adding a `NOT NULL` column with no default — which fails immediately against a table that already has rows, because the database has no value to put in that column for rows that existed before the migration ran, and `NOT NULL` forbids leaving it empty. Making a column required *and* backfilling existing rows safely is a genuinely multi-step migration in practice (add the column as nullable, backfill every existing row with a real value, *then* alter the column to add the `NOT NULL` constraint) — collapsing all three steps into one migration is exactly the kind of mistake that looks fine in a fresh development database with no data and then fails loudly the first time it's run against production.
</details>

<details>
<summary>Exercise 10.3</summary>

```python
async def transfer_stock(session: AsyncSession, from_id: int, to_id: int, quantity: int):
    async with session.begin():
        source = await session.get(ProductTable, from_id)
        destination = await session.get(ProductTable, to_id)
        if source is None or destination is None:
            raise NotFoundError("Product", from_id if source is None else to_id)
        source.stock_qty -= quantity
        destination.stock_qty += quantity
```

Calling this with a nonexistent `to_id`: `source.stock_qty -= quantity` never actually executes, because the `if source is None or destination is None` check runs first and raises before reaching that line in this ordering — but even if the mutation *had* been written before the check (a more realistic mistake), `async with session.begin()` guarantees that raising anywhere inside the block rolls back every change staged within it, so `source.stock_qty` would be unaffected in the database regardless of exactly which line inside the block failed or when. Re-querying `from_id` in a fresh call afterward confirms its `stock_qty` is unchanged from before the failed attempt. Calling it again with two valid IDs updates both rows together, visible on the next query for either product.
</details>

<details>
<summary>Exercise 10.4</summary>

```python
class ProductRepository:
    ...
    async def attach_tag(self, product_id: int, tag_name: str) -> ProductTable:
        product = await self.get_or_raise(product_id)
        result = await self.session.execute(select(TagTable).where(TagTable.name == tag_name))
        tag = result.scalar_one_or_none()
        if tag is None:
            tag = TagTable(name=tag_name)
        product.tags.append(tag)
        self.session.add(product)
        await self.session.commit()
        await self.session.refresh(product)
        return product
```

```python
@router.post("/{product_id}/tags", response_model=ProductPublic)
async def add_tag_to_product(product_id: int, tag_name: str, repo: ProductRepoDep):
    return await repo.attach_tag(product_id, tag_name)
```

The route is three lines and contains zero SQL-adjacent code — it's purely "receive input, call the repository, return the result," with every actual query concern (finding-or-creating the tag, managing the relationship, committing) living in `ProductRepository`, reusable from anywhere else that might need the same "attach a tag to a product" operation without going through HTTP at all.
</details>

<details>
<summary>Exercise 10.5</summary>

`selectinload` issues **2** queries total (one for products, one `IN (...)` for all their reviews combined) and returns exactly 10 product objects, each with its `reviews` list already populated — no duplication, because the second query is a separate round trip rather than a join. `joinedload` issues **1** query, but the raw row count returned by the database is the sum of every product-review pair — a product with 3 reviews contributes 3 raw rows to the join's result set, all sharing the same product columns repeated across them; SQLAlchemy then deduplicates these back down to 10 distinct product objects on your side, but the *data actually transferred* is larger than `selectinload`'s approach for anything with more than a handful of related rows.

For this kind of one-to-many relationship, where the "many" side (reviews per product) can genuinely vary and isn't tightly bounded, `selectinload`'s two-fixed-queries approach scales more predictably. `joinedload` is more attractive when the relationship is one-to-one or small-and-bounded (a product's single "primary image," say), where the single-round-trip benefit outweighs the row-duplication cost, or when you specifically need everything resolved in one database round trip for latency reasons and know the fan-out is small.
</details>

---

## Chapter Summary

- Alembic replaces hand-written migration scripts with versioned, trackable `upgrade()`/`downgrade()` history — and it's entirely reasonable to point it at a sync driver even in an otherwise-async application, since migrations are administrative, not per-request.
- The N+1 query problem is exactly what its name says: one query for a parent collection, plus one more per parent row for related data — `selectinload` collapses that into a fixed two queries regardless of collection size; `joinedload` collapses it into one query at the cost of duplicated rows in the raw result.
- `async with session.begin():` makes a multi-step operation atomic without manual `try`/`except`/`rollback` — everything staged inside commits together on success, or rolls back together on any exception.
- The repository pattern moves query logic out of routes and into a dedicated class that knows about the database and your domain exceptions (Chapter 7's `NotFoundError`) but nothing about HTTP — routes become thin translators between requests and repository calls, and the exception-handling layer built back in Chapter 7 needed zero changes to keep working with it.

**Next:** Chapter 11 covers authentication and authorization — OAuth2, JWTs, password hashing, and role-based access control — the last piece Part II needs before Chapter 12 tackles middleware and the full request lifecycle.
