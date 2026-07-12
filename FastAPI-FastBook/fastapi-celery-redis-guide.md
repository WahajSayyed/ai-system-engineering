# FastAPI + Celery + Redis: A Complete Production Guide

> A standalone deep-dive, companion to the FastAPI curriculum. Chapter 22 built background processing with ARQ; this guide covers the other major production stack — Celery, backed by Redis — including why you'd choose it, how to configure it for multiple distinct job types, and what changes when it runs in production rather than on your laptop.

---

## 1. What Each Piece Actually Does

- **FastAPI** — your API layer. It accepts requests, validates them, and needs to stay responsive. It should never block on slow work.
- **Celery** — a distributed task queue. Long-running or resource-intensive work (sending emails, generating reports, processing images, running ML inference) is handed off to Celery, executed by **worker processes that are completely separate from your FastAPI/Uvicorn process**.
- **Redis** — in this stack, playing two distinct roles simultaneously: the **broker** (where task messages are queued, waiting for a worker to pick them up) and, optionally, the **result backend** (where a task's return value or failure is stored, so something can later ask "is this done, and what did it return?").

**Current version note (early 2026):** Celery's latest stable release is the 5.6.x series, supporting Python 3.9 through 3.13, with native Pydantic model support for task arguments — worth knowing since older tutorials predate this. Celery pins compatibility to Redis client versions ≤5.2.1; check this against whatever `redis-py` version your project resolves to.

## 2. Architecture: Broker vs. Backend, Producer vs. Worker

```mermaid
flowchart LR
    A["FastAPI route<br/>(producer)"] -->|"task.delay(...)"| B["Redis — BROKER<br/>(queued task messages)"]
    B --> C["Celery worker process<br/>(consumer, separate process)"]
    C -->|"stores result"| D["Redis — BACKEND<br/>(task results/status)"]
    E["FastAPI route<br/>(status check)"] -->|"AsyncResult(task_id)"| D
```

The **broker** and **backend** are conceptually distinct even when (as here) they're the same Redis instance:

- **Broker**: "what work needs doing, and has anyone picked it up yet." Celery supports Redis or RabbitMQ as feature-complete broker options.
- **Backend**: "what happened to a specific task I already handed off." Optional — you only need it if something later needs to check status or fetch a return value, which in practice is almost always true for an API that wants to tell a client "your job is done."

A common, deliberate practice: point the broker and backend at **different Redis logical databases** (Redis supports numbered databases within one instance, `redis://host:6379/0` vs `redis://host:6379/1`) to avoid key collisions between broker bookkeeping and stored results, even though both live in the same physical Redis process.

```python
# celery_app.py
from celery import Celery

celery_app = Celery(
    "myapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=["tasks.email", "tasks.reports", "tasks.ml"],
)
```

`include=[...]` tells Celery which modules contain task definitions, so the worker process knows to import and register them at startup — this matters because the worker is a genuinely separate Python process from your FastAPI app, with its own import step, not something that automatically shares your API's already-loaded modules.

## 3. Project Structure for Multiple Job Types

A single flat `tasks.py` works for a toy example; a real system with several genuinely different kinds of background work benefits from splitting by domain, mirroring the layered structure this curriculum's Chapter 18 established for the API side:

```
app/
├── main.py                  # FastAPI app
├── celery_app.py             # Celery instance + configuration
├── tasks/
│   ├── __init__.py
│   ├── email.py               # send_welcome_email, send_password_reset, ...
│   ├── reports.py              # generate_report, export_analytics, ...
│   └── ml.py                    # run_inference, batch_embed, ...
├── routers/
│   └── ...                        # FastAPI routes that enqueue tasks and check status
└── docker-compose.yml
```

## 4. Configuring Celery for Real Use

```python
# celery_app.py
from celery import Celery
from kombu import Queue

celery_app = Celery(
    "myapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=["tasks.email", "tasks.reports", "tasks.ml"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],           # refuse to deserialize anything but JSON — a real security boundary, see §12
    timezone="UTC",
    task_acks_late=True,                # acknowledge a task only AFTER it succeeds — see below
    worker_prefetch_multiplier=1,        # don't let one worker hoard many tasks while others sit idle
    result_expires=3600,                  # stop accumulating results forever — expire after 1 hour
    task_routes={
        "tasks.email.*": {"queue": "email"},
        "tasks.reports.*": {"queue": "reports"},
        "tasks.ml.*": {"queue": "ml_inference"},
    },
)
```

Two settings worth understanding rather than copying blindly:

- **`task_acks_late=True`** — by default, Celery acknowledges (marks as "handled") a task the moment a worker *starts* it, not when it finishes. If that worker crashes mid-task, the task is gone — already acknowledged, never actually completed. `task_acks_late=True` delays acknowledgment until the task genuinely finishes (or fails in a way Celery is told about), so a crashed worker's in-flight task gets redelivered to another worker instead of silently vanishing. The trade-off: a task that crashes the *worker itself* (not just raises a normal exception) might run twice — which is exactly why **idempotency** (§8) matters as much here as it did with ARQ in Chapter 22.
- **`worker_prefetch_multiplier=1`** — controls how many unacknowledged tasks a single worker process will hold at once. The default (historically 4) can mean one slow worker sits on a batch of tasks other, faster or idle workers could have picked up instead. Setting it to `1` trades some throughput efficiency for fairer distribution — usually the right call for tasks with meaningfully variable duration (a mix of fast and slow reports, say), less critical for a queue of uniformly quick tasks.

## 5. Defining Tasks for Multiple, Genuinely Different Job Types

```python
# tasks/email.py
from celery_app import celery_app
import logging

logger = logging.getLogger("worker.email")

@celery_app.task(name="tasks.email.send_welcome_email", bind=True, max_retries=3)
def send_welcome_email(self, email: str, username: str) -> dict:
    logger.info(f"Sending welcome email to {email}")
    # ...real email provider call here...
    return {"email": email, "sent": True}
```

```python
# tasks/reports.py
from celery_app import celery_app
import time

@celery_app.task(name="tasks.reports.generate_report")
def generate_report(product_count: int) -> dict:
    time.sleep(5)   # simulating real report-generation work
    return {"summary": f"Report covering {product_count} products"}
```

```python
# tasks/ml.py
from celery_app import celery_app

@celery_app.task(name="tasks.ml.run_inference")
def run_inference(model_name: str, input_payload: dict) -> dict:
    # ...load model (or reuse a pre-loaded one — see §10), run inference...
    return {"model": model_name, "prediction": "..."}
```

**Naming tasks explicitly** (`name="tasks.email.send_welcome_email"`) rather than letting Celery infer a name from the function's module path is worth doing deliberately — it decouples the task's identity (used for routing, monitoring, and any already-queued messages referencing it) from your Python file layout, so refactoring which file a task lives in doesn't silently break `task_routes` patterns or in-flight messages referencing the old auto-generated name.

Each task module maps to its own queue via `task_routes` (§4) — `tasks.email.*` always goes to the `email` queue, `tasks.ml.*` to `ml_inference`, and so on. This is the mechanism that makes "multiple job types" genuinely useful rather than just organizational: **you can run separate worker processes, each dedicated to one queue, each independently sized and scaled**:

```bash
# A worker dedicated to email — many short, I/O-bound tasks, high concurrency is cheap
celery -A celery_app worker -Q email --concurrency=8 --loglevel=info

# A worker dedicated to reports — fewer, longer-running tasks
celery -A celery_app worker -Q reports --concurrency=2 --loglevel=info

# A worker dedicated to ML inference — likely CPU/GPU-bound, deliberately low concurrency
celery -A celery_app worker -Q ml_inference --concurrency=1 --loglevel=info
```

This is a genuine, practical benefit distinct implementations rarely get right by accident: a burst of report-generation traffic cannot starve email delivery, because they're served by entirely separate worker pools, each tuned to its own workload's actual resource profile — a `--concurrency=8` pool for lightweight I/O-bound email sending would be actively wrong for CPU-heavy ML inference, where high concurrency just causes contention rather than throughput.

## 6. Integrating with FastAPI

```python
# routers/reports.py
from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult
from celery_app import celery_app
from tasks.reports import generate_report

router = APIRouter(prefix="/reports", tags=["reports"])

@router.post("/", status_code=202)
async def create_report(product_count: int):
    task = generate_report.delay(product_count)
    return {"task_id": task.id, "status": "queued"}

@router.get("/{task_id}")
async def get_report_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    if result.state == "PENDING":
        return {"task_id": task_id, "status": "pending"}
    if result.state == "FAILURE":
        return {"task_id": task_id, "status": "failed", "error": str(result.result)}
    if result.ready():
        return {"task_id": task_id, "status": "complete", "result": result.result}
    return {"task_id": task_id, "status": result.state}
```

**The single most important rule for this integration, worth stating as plainly as possible: never call `result.get()` (which blocks until the task finishes) inside a FastAPI route.** In a synchronous framework this blocks one worker thread; in FastAPI's async event loop, it blocks the *entire* event loop — precisely Chapter 2's `time.sleep`-inside-`async def` bug, now wearing a Celery-shaped disguise. The whole point of a task queue is that the client gets an immediate response (a `task_id`) and polls a separate status endpoint — calling `.get()` synchronously defeats that purpose entirely and reintroduces the exact blocking problem this architecture exists to avoid.

## 7. The Sync/Async Boundary: Why Celery Tasks Need Their Own Database Session Setup

This is the detail most tutorials gloss over and most real projects get bitten by at least once: **Celery tasks are fundamentally synchronous functions**, executed by worker processes that are not running an asyncio event loop the way your FastAPI app is. If a task needs database access, reusing your FastAPI app's *async* SQLAlchemy engine (Chapter 9) inside a Celery task is the wrong tool — there's no running event loop in the worker process to support it correctly, and forcing it (via `asyncio.run(...)` called from inside a task) is a workable but easy-to-misuse escape hatch, not a clean fit.

The common, robust pattern: **a separate, genuinely synchronous database engine, used only by Celery tasks**, alongside the async engine your FastAPI routes already use:

```python
# db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# ASYNC — used by FastAPI routes (Chapter 9)
async_engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/mydb")
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

# SYNC — used ONLY by Celery tasks, a different driver for the same database
sync_engine = create_engine("postgresql+psycopg2://user:pass@localhost/mydb")
SyncSessionLocal = sessionmaker(bind=sync_engine)
```

```python
# tasks/reports.py
from db import SyncSessionLocal
from models import ProductTable
from sqlalchemy import select

@celery_app.task(name="tasks.reports.generate_report")
def generate_report(product_count: int) -> dict:
    with SyncSessionLocal() as session:
        products = session.execute(select(ProductTable).limit(product_count)).scalars().all()
        # ...build the report using plain, synchronous SQLAlchemy calls...
    return {"summary": f"Report covering {len(products)} products"}
```

Note the driver difference: `asyncpg` for the async engine, `psycopg2` (or `psycopg` v3) for the sync one — same database, same connection string structure, different driver segment, exactly Chapter 9.2's DSN table, now applied across a process boundary rather than within one app. This "dual session" pattern is genuinely common in real production FastAPI+Celery codebases specifically because it sidesteps fighting an event loop that doesn't exist in the worker process, rather than trying to force async code to run somewhere it doesn't naturally belong.

## 8. Retries, Backoff, and Idempotency

Celery's retry support is more built-in and declarative than hand-rolling it (as Chapter 22 did for ARQ):

```python
@celery_app.task(
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,       # exponential backoff, handled automatically
    retry_backoff_max=60,      # cap the maximum delay between retries
    retry_jitter=True,          # add randomness — avoids synchronized retry storms (Chapter 22.5's lesson)
    max_retries=5,
)
def send_welcome_email(self, email: str, username: str) -> dict:
    # ...email logic; ConnectionError or TimeoutError triggers an automatic retry
    # with exponential backoff and jitter, no manual Retry()-raising required...
    ...
```

`autoretry_for`, `retry_backoff`, and `retry_jitter` collapse what Chapter 22 built by hand for ARQ (manually computing `2 ** attempt`, manually raising a `Retry` exception) into declarative decorator arguments — a genuine maturity advantage of Celery's longer history as a project.

**Idempotency remains just as essential here as it was in Chapter 22 — arguably more so**, given `task_acks_late=True` (§4) explicitly accepts the possibility of a task running more than once after a worker crash. The practical pattern: design every task so that running it twice with identical arguments produces the same end state as running it once — a database `UPSERT` (insert-or-update) instead of a blind `INSERT`, a unique constraint that makes a duplicate write fail harmlessly rather than corrupt data, or an explicit Redis-based lock/deduplication check before doing real work.

```python
@celery_app.task(name="tasks.reports.generate_report_idempotent")
def generate_report_idempotent(job_id: str, product_count: int) -> dict:
    with SyncSessionLocal() as session:
        existing = session.execute(
            select(ReportResult).where(ReportResult.job_id == job_id)
        ).scalar_one_or_none()
        if existing is not None:
            return existing.result_json   # already done — a retry-safe no-op

        # ...do the real work...
        result = {"summary": f"Report covering {product_count} products"}
        session.add(ReportResult(job_id=job_id, result_json=result))
        session.commit()
        return result
```

## 9. Task Workflows: `chain`, `group`, and `chord`

Celery's built-in primitives for composing multiple tasks into a pipeline are a real strength over simpler queues:

```python
from celery import chain, group, chord

# chain: run tasks in sequence, each one's output feeding the next
pipeline = chain(fetch_raw_data.s(source_url), clean_data.s(), generate_report.s())
pipeline.apply_async()

# group: run several tasks in parallel, independently
job = group(run_inference.s(model, item) for item in batch)
job.apply_async()

# chord: run a group in parallel, then run one final task once ALL of them finish
workflow = chord(
    (run_inference.s(model, item) for item in batch),
    aggregate_predictions.s(),
)
workflow.apply_async()
```

`chord` in particular is worth knowing by name — "run N independent tasks in parallel, then do one thing with all their combined results" is an extremely common real shape (batch ML inference across many items, then aggregate; fan out a report across several data sources, then combine them into one final document), and building this correctly by hand (tracking completion of an arbitrary-sized group, then triggering exactly one follow-up) is considerably more error-prone than using Celery's built-in primitive for exactly this pattern.

## 10. Periodic Tasks with Celery Beat

For scheduled, recurring work (a nightly cleanup, a weekly digest email) rather than work triggered by a request:

```python
# celery_app.py (addition)
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "cleanup-old-reports-nightly": {
        "task": "tasks.reports.cleanup_old_reports",
        "schedule": crontab(hour=2, minute=0),   # 2:00 AM daily
    },
    "send-weekly-digest": {
        "task": "tasks.email.send_weekly_digest",
        "schedule": crontab(hour=9, minute=0, day_of_week=1),   # Monday 9 AM
    },
}
```

Run the scheduler as its own process, separate again from both your API and your regular workers:

```bash
celery -A celery_app beat --loglevel=info
```

`beat` only *schedules* — it enqueues these tasks onto the broker at the configured times; the actual execution still happens on whichever regular worker process is watching the relevant queue, exactly as if a FastAPI route had enqueued them.

## 11. Monitoring: Flower and Prometheus

**Flower** is a real-time web dashboard for a running Celery deployment — task states, worker status, throughput, all inspectable live:

```bash
celery -A celery_app flower --port=5555
```

For genuine production observability (tying back to Chapter 20), Flower's live dashboard isn't a substitute for time-series metrics you can alert on. The **`celery-exporter`** package exposes Celery's internal metrics in Prometheus format — queue depth, task latency, failure rate, worker uptime — which you can then wire into Grafana dashboards and real alerting, exactly the Chapter 20 pattern (Counters, Histograms, `/metrics`), applied to your task queue instead of your HTTP layer.

## 12. Production Deployment: Docker Compose

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    command: ["redis-server", "--appendonly", "yes", "--requirepass", "${REDIS_PASSWORD}"]
    volumes:
      - redis_data:/data

  api:
    build: .
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    depends_on:
      - redis
    env_file: .env

  worker-email:
    build: .
    command: celery -A celery_app worker -Q email --concurrency=8 --loglevel=info
    depends_on:
      - redis
    env_file: .env

  worker-reports:
    build: .
    command: celery -A celery_app worker -Q reports --concurrency=2 --loglevel=info
    depends_on:
      - redis
    env_file: .env

  worker-ml:
    build: .
    command: celery -A celery_app worker -Q ml_inference --concurrency=1 --loglevel=info
    depends_on:
      - redis
    env_file: .env

  beat:
    build: .
    command: celery -A celery_app beat --loglevel=info
    depends_on:
      - redis
    env_file: .env

  flower:
    build: .
    command: celery -A celery_app flower --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis

volumes:
  redis_data:
```

Notice the shape this takes: **one image, many different `command`s** — `api`, each `worker-*`, `beat`, and `flower` are all the *same* built container, differing only in what process actually runs inside it. This is what "scale workers independently of the API tier" means concretely: in a real orchestrator (Kubernetes, ECS), you'd scale `worker-email`'s replica count up during a marketing-email blast without touching the `api` service's replica count at all, and vice versa if API traffic spikes with steady background load — the two scale on entirely independent axes, because they're entirely independent processes sharing nothing but the Redis broker.

**Security, concretely:** `--requirepass` on Redis is not optional for anything beyond local development — an unauthenticated Redis instance reachable from outside a trusted private network is a real, commonly-exploited attack surface (unauthenticated Redis instances have been used for everything from data theft to remote code execution via misconfigured persistence settings). Combine a password with network isolation (Redis should not be reachable from the public internet at all — only from your API and worker containers, on a private network) and, in `celery_app.conf`, `accept_content=["json"]` explicitly (§4) — Celery historically defaulted to a more permissive serializer (`pickle`) capable of deserializing arbitrary Python objects, which is a genuine remote-code-execution risk if anything untrusted can ever reach your broker; restricting to `json` closes that off structurally, the same "structural, not just a filter" pattern Chapter 21 established for BOPLA.

**Graceful shutdown:** a worker receiving a shutdown signal (during a deploy, a scale-down) should finish its current task rather than being killed mid-execution — `task_acks_late=True` (§4) combined with Celery's default handling of `SIGTERM` (warm shutdown, finishing in-flight tasks before exiting) versus `SIGKILL` (immediate, tasks lost) is worth being deliberate about in your deployment platform's configured shutdown grace period — too short a grace period effectively forces a hard kill regardless of `acks_late`'s intent.

## 13. Common Pitfalls

- **Passing non-serializable objects as task arguments.** A SQLAlchemy model instance, an open file handle, a database session — none of these survive being serialized to JSON and sent through Redis to a separate process. Pass primitive types (IDs, strings, plain dicts) and re-fetch or reconstruct anything complex *inside* the task, using that task's own (sync) database session.
- **Calling `.get()` synchronously inside a FastAPI route** (§6) — blocks the event loop, defeating the entire architecture.
- **Skipping idempotency** (§8) — with `task_acks_late=True` explicitly accepting the possibility of duplicate execution after a crash, an idempotency-free task is a data-corruption risk waiting for an unlucky worker restart, not a hypothetical edge case.
- **Reusing the async engine inside a task** (§7) — works by accident sometimes, fails confusingly and intermittently other times, depending on event-loop state that doesn't reliably exist in a worker process.
- **Leaving the default (often permissive) serializer in place** (§12) — explicitly set `task_serializer`/`accept_content` to `json` rather than trusting defaults, especially in anything internet-adjacent.

## 14. Celery vs. ARQ — Revisited

| | ARQ (Chapter 22) | Celery (this guide) |
|---|---|---|
| Design | Async-native from the ground up | Sync-first; async support layered on |
| Broker support | Redis only | Redis, RabbitMQ, others |
| Workflow primitives | Basic | Rich — `chain`, `group`, `chord` |
| Scheduling | Basic cron-like support | Mature (`celery beat`) |
| Monitoring | Minimal built-in tooling | Flower, `celery-exporter` + Prometheus/Grafana |
| Maturity / ecosystem | Newer, smaller | Extremely mature, extremely widely deployed |
| Best fit | A project already fully async, wanting minimal added complexity | A project needing rich workflows, multiple queues with independent scaling, mature scheduling, and battle-tested production tooling |

Neither is objectively "better" — they solve the same core problem (durable, retryable, out-of-process task execution) with different design centers of gravity. A project that's already all-in on async everywhere (this curriculum's own running example) fits ARQ more naturally with less friction; a project needing complex multi-step workflows, several genuinely independent worker pools, or Celery's much larger surrounding tooling ecosystem often finds that maturity worth the sync/async boundary this guide spent §7 addressing directly.

## 15. Summary Checklist

- [ ] Broker and result backend configured (Redis, ideally on separate logical DB numbers).
- [ ] `task_serializer`/`accept_content` explicitly restricted to `json` — not left on a permissive default.
- [ ] `task_acks_late=True` set, and every task designed to be idempotent as a result.
- [ ] Different job types routed to different queues (`task_routes`), each served by an independently-sized, independently-scalable worker pool.
- [ ] A separate, genuinely synchronous database engine/session for use inside tasks — never the async engine your FastAPI routes use.
- [ ] Retries configured declaratively (`autoretry_for`, `retry_backoff`, `retry_jitter`) rather than hand-rolled.
- [ ] No route ever calls `.get()` synchronously — always return a task ID and poll a separate status endpoint.
- [ ] Flower (and ideally `celery-exporter` + Prometheus/Grafana) running for real visibility into queue depth, latency, and failure rate.
- [ ] Redis password-protected and network-isolated — never exposed to the open internet.
- [ ] Workers, beat, and the API deployed as separate, independently-scalable services (Docker Compose locally; separate deployments/services in any real orchestrator).
