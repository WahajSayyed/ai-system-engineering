# Chapter 20: Async Database Access

> Part 4 — I/O, Networking, and Real Systems · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 20.1 Why Not Just Wrap a Sync Driver in a Thread?

The same argument from Chapter 19 applies here: you *could* wrap a synchronous driver like `psycopg2` or `PyMySQL` with `asyncio.to_thread()` (Chapter 18), and for low-concurrency scripts, that's often fine. But for a service handling many concurrent requests — each needing a database round-trip — you're bounded by the thread pool's worker count, and every in-flight query ties up a whole OS thread. Native async database drivers use non-blocking sockets directly, the same principle as `aiohttp`/`httpx`, letting you have many queries genuinely in flight without a thread-per-query cost.

---

## 20.2 `asyncpg`: A Fast, Async-Native PostgreSQL Driver

`asyncpg` isn't DB-API-compliant — it has its own API, designed purely for performance and tight asyncio integration.

```python
import asyncpg
import asyncio

async def main():
    conn = await asyncpg.connect("postgresql://user:password@localhost/mydb")
    rows = await conn.fetch("SELECT id, name FROM users WHERE active = $1", True)
    for row in rows:
        print(row["id"], row["name"])
    await conn.close()

asyncio.run(main())
```

Note the `$1` placeholder style (PostgreSQL's native parameter syntax) rather than Python string formatting — **always use parameterized queries like this**, never build SQL strings via f-strings or `%` formatting with untrusted input, to avoid SQL injection.

**For any real application, use a connection pool rather than a single connection:**

```python
import asyncpg
import asyncio

async def main():
    pool = await asyncpg.create_pool(
        "postgresql://user:password@localhost/mydb",
        min_size=5,
        max_size=20,
    )

    async def query_user(user_id):
        async with pool.acquire() as conn:
            return await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(query_user(i)) for i in range(1, 11)]

    for task in tasks:
        print(task.result())

    await pool.close()

asyncio.run(main())
```

`pool.acquire()` is an async context manager (Chapter 14) — it checks out a connection from the pool for the duration of the block and returns it automatically afterward, exactly the same shape as Chapter 14's hand-built connection pool example.

---

## 20.3 `aiomysql`: The Equivalent for MySQL

`aiomysql` follows a more DB-API-like, cursor-based pattern:

```python
import aiomysql
import asyncio

async def main():
    pool = await aiomysql.create_pool(
        host="localhost", user="user", password="password", db="mydb",
        minsize=5, maxsize=20,
    )

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, name FROM users WHERE active = %s", (True,))
            rows = await cur.fetchall()
            for row in rows:
                print(row)

    pool.close()
    await pool.wait_closed()

asyncio.run(main())
```

Same underlying ideas as `asyncpg` — a pool, `acquire()` as an async context manager, parameterized queries (`%s` placeholders here, MySQL's convention) — just with an extra cursor layer, closer to the traditional DB-API shape.

---

## 20.4 SQLAlchemy 2.0's Async ORM

SQLAlchemy 2.0 added first-class async support: `create_async_engine()` plus `AsyncSession`, running on top of an async driver underneath (`asyncpg` for Postgres, `aiomysql` for MySQL, `aiosqlite` for SQLite — selected via the connection URL's dialect).

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# ... assume a User model is defined here with `id` and `name` columns ...

async def main():
    engine = create_async_engine("postgresql+asyncpg://user:password@localhost/mydb")
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with async_session() as session:
        result = await session.execute(select(User).where(User.active == True))
        users = result.scalars().all()
        for user in users:
            print(user.name)

    await engine.dispose()
```

### The Lazy-Loading Gotcha

This is the single most common trip-up with SQLAlchemy's async ORM: **implicit lazy loading of relationships doesn't work the way it does in sync code**, because lazy loading a relationship means running another synchronous-shaped query behind the scenes — something that can't happen implicitly inside an async context.

```python
async with async_session() as session:
    author = await session.get(Author, 1)

# Outside the session, or without eager loading, this raises an error:
print(author.books)   # MissingGreenlet or similar — relationship wasn't loaded
```

**The fix: eager-load relationships you'll need**, using `selectinload` or `joinedload`:

```python
from sqlalchemy.orm import selectinload

async with async_session() as session:
    result = await session.execute(
        select(Author).options(selectinload(Author.books)).where(Author.id == 1)
    )
    author = result.scalar_one()
    print(author.books)   # works — already loaded eagerly as part of the query
```

If you genuinely need to trigger a lazy load after the fact, `await session.refresh(author, attribute_names=["books"])` is another option — but planning your eager-loading strategy upfront, based on what your code will actually access, is the more robust pattern.

### Transactions

```python
async with async_session() as session:
    async with session.begin():   # commits on success, rolls back on exception
        session.add(User(name="new user"))
        # if anything raises here, the whole transaction rolls back automatically
```

---

## 20.5 Connection Pooling Considerations

Opening a new database connection is relatively expensive (TCP handshake, authentication) — pooling amortizes this cost across many queries. A few things worth tuning deliberately, rather than leaving at defaults:

- **`min_size`/`max_size` (or `minsize`/`maxsize`)**: too small, and requests queue up waiting for a free connection under load (the exact same bounded-concurrency behavior as a `Semaphore`, Chapter 12); too large, and you risk overwhelming the *database server's own* connection limit (`max_connections` in PostgreSQL/MySQL), which is often a much harder constraint than anything on the client side.
- **Pool size is not "one per concurrent request."** Since queries are usually fast, a modest pool (tens of connections) can typically serve a much larger number of concurrent application-level requests — size the pool based on real measured query latency and throughput, not a naive 1:1 assumption.
- **Set a timeout on `acquire()`** if your driver supports it. Without one, pool exhaustion under sustained overload can cause requests to hang indefinitely rather than failing fast with a clear error — generally the better failure mode for a production service.

---

## 20.6 Hands-On Exercises

*(These exercises assume access to a local PostgreSQL instance. If you don't have one handy, `aiosqlite` — usable directly or via SQLAlchemy's `sqlite+aiosqlite://` dialect — is a good zero-setup alternative for practicing the same concepts.)*

**Exercise 1 — Basic `asyncpg` usage.**
Connect to a local Postgres database, create a simple `users` table, insert a few rows using parameterized queries, then fetch and print them.

**Exercise 2 — Pool-bounded concurrent queries.**
Build an `asyncpg` pool with `max_size=5`. Run 15 concurrent queries via a `TaskGroup`, each acquiring a connection from the pool. Add a shared counter (Chapter 12's `Lock` pattern) tracking maximum simultaneous connections in use, and confirm it never exceeds 5.

**Exercise 3 — SQLAlchemy 2.0 async basics.**
Define a simple `User` ORM model, create an async engine and session, insert a few rows, then query them back with `select()`.

**Exercise 4 — Reproduce (and fix) the lazy-loading gotcha.**
Define two related models (e.g., `Author` and `Book` with a one-to-many relationship). Query an `Author` without eager loading, then try accessing `author.books` and observe the error. Fix it using `selectinload`.

**Exercise 5 — Transaction rollback.**
Write code that inserts two rows inside a transaction, where the second insert deliberately violates a constraint (e.g., a duplicate primary key) causing an exception. Confirm the transaction rolls back and neither row is actually committed.

**Exercise 6 (stretch) — Observe pool exhaustion.**
Set `max_size=2` on a pool. Launch 5 concurrent queries, each holding its connection for 1 second (simulate with `pg_sleep(1)` or an `asyncio.sleep()` inside the connection block). Observe the 3 extra queries waiting in line. Then add a timeout to `pool.acquire()` and confirm you get a clean timeout error instead of an indefinite hang if you shorten the wait budget below what's needed.

---

## 20.7 Common Pitfalls

- **Wrapping a sync driver in `to_thread()` for a high-concurrency service, instead of using a native async driver.** It works, but ties up thread pool workers and doesn't scale as gracefully as `asyncpg`/`aiomysql` at real concurrency.
- **Not using a connection pool at all.** Opening a fresh connection per request is slow and can exhaust the database server's own connection limit under load.
- **Oversizing the client-side pool.** A larger pool doesn't help if the database server's `max_connections` is the actual bottleneck — and many open connections consume real memory on the server regardless of how busy they are.
- **Forgetting eager loading in the async ORM.** Accessing a lazy-loaded relationship outside the session's active async context raises an error — implicit lazy loading requires synchronous I/O that can't happen automatically here.
- **Building SQL strings via f-strings/`%` formatting instead of parameterized queries.** This is a real SQL-injection risk — always use your driver's native placeholder syntax (`$1`, `%s`, etc.) with parameters passed separately.
- **Not setting a timeout on pool acquisition.** Without one, pool exhaustion can hang requests indefinitely instead of failing fast with an actionable error.

---

## 20.8 Further Reading

- [asyncpg documentation](https://magicstack.github.io/asyncpg/current/)
- [aiomysql documentation](https://aiomysql.readthedocs.io/)
- [SQLAlchemy 2.0: Asynchronous I/O](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)

---

**Next: Chapter 21 — Event Loop Internals** (selectors, the `call_soon`/`call_later` scheduling model in the real implementation, the deprecated policy system, `loop_factory`, and `uvloop`).
