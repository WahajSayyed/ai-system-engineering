# FastAPI: From Foundations to Production
### A Code-First Curriculum (Beginner → Advanced)

> Stack target: **FastAPI 0.136+**, **Python 3.12+**, **Pydantic v2.9+**, `fastapi[standard]` CLI, **SQLModel / SQLAlchemy 2.0 (async)**, **pytest + httpx**, **Docker**. Pydantic v1 is not covered — FastAPI dropped it, and so do we.

---

## How This Curriculum Is Organized

Each chapter follows the same three-part structure:

- **Theory** — the concepts, internals, and *why*, not just the *how*.
- **Hands-On Project** — a working piece of code you build during the chapter.
- **Practice Exercises** — 3–5 problems to solidify the chapter before moving on, ranging from quick drills to small extensions.

The curriculum is split into four parts, moving from raw fundamentals to a production-grade, deployed system. Later chapters assume everything before them.

**Prerequisites:** comfortable with Python (functions, classes, basic OOP), some exposure to HTTP (what a GET/POST is), command line basics. No prior async or web framework experience assumed — Chapter 2 covers that from scratch.

---

## Part I — Foundations

### Chapter 1: Why FastAPI — The Landscape and the Mental Model
- **Theory:** ASGI vs WSGI, where FastAPI sits relative to Flask/Django/Starlette, the "three pillars" (Starlette for the web parts, Pydantic for the data parts, type hints as the API contract), how OpenAPI/Swagger generation actually works under the hood.
- **Hands-On:** Install via `fastapi[standard]`, scaffold a project with `uv`, run `fastapi dev`, inspect the auto-generated `/docs` and `/redoc` and the raw `openapi.json`.
- **Exercises:** Compare a minimal Flask route vs FastAPI route side by side; identify what type hints buy you; break the auto-docs on purpose by removing type hints and observe the diff.

### Chapter 2: Python Foundations FastAPI Assumes You Know
- **Theory:** Type hints (`list[str]`, `dict`, `Optional`/`X | None`), `async`/`await` and the event loop, decorators refresher, dataclasses vs Pydantic models.
- **Hands-On:** Write a small async script with `asyncio.gather`, then a sync equivalent, and benchmark blocking vs non-blocking I/O with `time.sleep` vs `asyncio.sleep`.
- **Exercises:** Predict execution order of a set of `async def` functions before running them; convert a blocking function to async; explain (in your own words) why `def` vs `async def` matters for a path operation.

### Chapter 3: Your First API — Path Operations and Routing
- **Theory:** `@app.get/post/put/delete/patch`, path parameters, path ordering and matching rules, `APIRouter` basics, route tags.
- **Hands-On:** Build a small "Notes" API with in-memory storage covering all CRUD verbs across multiple routes.
- **Exercises:** Add a route that conflicts with an existing one and diagnose the ordering bug; add path parameter type coercion; split routes into an `APIRouter` and mount it.

### Chapter 4: Request Data — Path, Query, and Body Parameters
- **Theory:** How FastAPI decides path vs query vs body, `Query`/`Path`/`Body` metadata (validation constraints, aliasing, deprecation flags), optional vs required params, multiple body parameters.
- **Hands-On:** Extend the Notes API with filtering (`?tag=`, `?limit=`, `?offset=`), pagination, and a nested body payload.
- **Exercises:** Add a query parameter with regex validation; make a body field optional with a sensible default; deliberately trigger and read a 422 validation error response.

### Chapter 5: Pydantic v2 Deep Dive — Validation and Serialization
- **Theory:** `BaseModel` internals, `pydantic-core` (Rust) and why v2 is faster, `model_validate`/`model_dump`, `field_validator` vs `model_validator`, `ConfigDict`, strict mode, computed fields.
- **Hands-On:** Model a "Product" domain with nested models, custom validators (e.g., non-negative price), and a computed `display_price` field.
- **Exercises:** Write a validator that cross-checks two fields (e.g., `start_date < end_date`); convert a dict-based payload into a typed model with `TypeAdapter`; break strict mode intentionally and read the error.

### Chapter 6: Response Models, Status Codes, and Serialization Control
- **Theory:** `response_model`, `response_model_exclude`/`include`, `status_code` conventions, `response_model_by_alias`, why double-validation happens and how to avoid it (a real 2026 performance gotcha).
- **Hands-On:** Add input/output model separation (`ProductIn` vs `ProductOut`) to the Products API, hide sensitive fields, return proper status codes for create/update/delete.
- **Exercises:** Reproduce the "double validation" performance trap from a profiling article and fix it; add an endpoint returning `204 No Content` correctly; use `response_model_exclude_none`.

### Chapter 7: Error Handling and Custom Exceptions
- **Theory:** `HTTPException`, custom exception classes + `exception_handler`, `RequestValidationError` customization, consistent error envelope design for real APIs.
- **Hands-On:** Build a global error-handling layer with a consistent JSON error shape (`{"error": {"code", "message", "details"}}`) across the Products API.
- **Exercises:** Write a custom exception for "resource not found" reused across three endpoints; override the default 422 validation error format; add a catch-all 500 handler that never leaks stack traces.

---

## Part II — Intermediate: Building Real APIs

### Chapter 8: Dependency Injection — FastAPI's Core Superpower
- **Theory:** `Depends()`, dependency resolution order, `yield`-based dependencies (setup/teardown), sub-dependencies, dependency caching within a request, class-based dependencies.
- **Hands-On:** Refactor the Products API to use a shared "pagination params" dependency and a "current settings" dependency.
- **Exercises:** Build a dependency that measures and logs request duration; build a nested dependency chain (A depends on B depends on C) and trace resolution order; convert a function dependency to a callable class.

### Chapter 9: Databases I — SQLModel and Async SQLAlchemy 2.0
- **Theory:** SQLModel's relationship to Pydantic and SQLAlchemy, sync vs async engines, sessions and the unit-of-work pattern, connection pooling basics.
- **Hands-On:** Replace in-memory storage with a real SQLite (then Postgres) backend using async SQLModel sessions injected via `Depends`.
- **Exercises:** Add a one-to-many relationship (Product → Reviews) and query it; write a migration by hand; benchmark sync vs async DB calls under concurrent load.

### Chapter 10: Databases II — Migrations, Relationships, and Query Patterns
- **Theory:** Alembic migrations, N+1 query problems and eager loading, transactions and rollback, repository pattern vs "fat routes."
- **Hands-On:** Set up Alembic for the Products API, add a many-to-many relation (Products ↔ Tags), implement a repository layer separating DB logic from route logic.
- **Exercises:** Deliberately create and then fix an N+1 query; write a migration that adds a nullable column safely; wrap a multi-step operation in a transaction with rollback on failure.

### Chapter 11: Authentication and Authorization
- **Theory:** OAuth2 password flow, JWT structure and verification, `OAuth2PasswordBearer`, password hashing (passlib/argon2), scopes and role-based access control.
- **Hands-On:** Build a full auth system: signup, login (JWT issuance), protected routes, and a role-gated admin endpoint.
- **Exercises:** Add token refresh; add scope-based permission checks to two different roles; simulate and defend against a token replay scenario conceptually.

### Chapter 12: Middleware, CORS, and the Request Lifecycle
- **Theory:** ASGI middleware stack ordering, `BaseHTTPMiddleware`, CORS mechanics (preflight, origins, credentials), request/response lifecycle diagram end-to-end.
- **Hands-On:** Add custom middleware for request ID injection and structured request logging; configure CORS correctly for a separate frontend origin.
- **Exercises:** Diagnose a CORS failure from browser console output; write middleware that rejects requests over a certain payload size; explain middleware ordering with a diagram.

### Chapter 13: Background Tasks and Async Job Patterns
- **Theory:** `BackgroundTasks` vs a real task queue, when "fire and forget" is (and isn't) safe, idempotency concerns.
- **Hands-On:** Add an email-notification-on-signup background task and a "generate report" endpoint that returns immediately while processing continues.
- **Exercises:** Identify a scenario where `BackgroundTasks` is the wrong tool (ties into Ch. 22); add error handling/logging inside a background task; test a background task's side effect.

### Chapter 14: File Uploads, Static Files, and Streaming Responses
- **Theory:** `UploadFile` vs `bytes`, streaming large files without loading into memory, `StreamingResponse`, serving static assets.
- **Hands-On:** Add product image upload/storage and a CSV export endpoint that streams rather than buffers.
- **Exercises:** Add file-type and size validation on upload; stream a large generated file without OOM; benchmark memory usage streaming vs buffering.

### Chapter 15: Testing FastAPI Applications
- **Theory:** `TestClient`/`httpx.AsyncClient`, dependency overrides for testing, fixtures, testing async endpoints, test database strategy.
- **Hands-On:** Write a full pytest suite for the Products + Auth API including a test-only SQLite database and dependency overrides for auth.
- **Exercises:** Reach meaningful coverage on the auth flow (happy path + 2 failure paths); write a fixture that spins up and tears down a clean DB per test; mock an external dependency (e.g., email sender).

---

## Part III — Advanced: Production Engineering

### Chapter 16: Advanced Pydantic — Generics, Settings, and Custom Types
- **Theory:** Generic models (`BaseModel[T]`), `pydantic-settings` for environment-driven config, `pydantic-extra-types`, custom types with `Annotated`, discriminated unions.
- **Hands-On:** Build a generic `Paginated[T]` response wrapper and a typed `Settings` class loaded from `.env`.
- **Exercises:** Model a polymorphic payload (e.g., different notification types) with a discriminated union; add a custom validated type (e.g., a `PhoneNumber`); write settings that fail fast on missing required env vars.

### Chapter 17: WebSockets and Real-Time Communication
- **Theory:** WebSocket lifecycle in ASGI, connection managers, broadcasting patterns, comparing WebSockets vs Server-Sent Events vs polling.
- **Hands-On:** Build a live order-status/notification feed over WebSockets with a simple connection manager supporting multiple clients.
- **Exercises:** Handle disconnects gracefully; broadcast a message to all connected clients except the sender; add an SSE endpoint as an alternative and compare tradeoffs.

### Chapter 18: Application Architecture at Scale
- **Theory:** Structuring large FastAPI codebases (routers/services/repositories/schemas layering), `APIRouter` composition and versioning strategies (`/v1`, `/v2`), sub-applications and mounting, settings-per-environment.
- **Hands-On:** Refactor the whole project so far into a clean layered structure with versioned routers and a clear dependency graph.
- **Exercises:** Add a `/v2` of one endpoint without breaking `/v1`; mount a second mini-app under the main app; draw the dependency graph of your own project.

### Chapter 19: Performance — Caching, Rate Limiting, and Profiling
- **Theory:** Where FastAPI's time actually goes (validation, serialization, I/O — from the "hidden taxes" framing), Redis caching patterns, rate limiting strategies, `ORJSONResponse` and response class tradeoffs, avoiding double validation.
- **Hands-On:** Add Redis-backed response caching to a read-heavy endpoint and a sliding-window rate limiter middleware.
- **Exercises:** Profile an endpoint under load and identify the single biggest bottleneck; switch default response class to `ORJSONResponse` and measure the difference; implement per-user rate limiting vs global rate limiting.

### Chapter 20: Observability — Logging, Metrics, and Tracing
- **Theory:** Structured logging, correlation/request IDs across services, Prometheus metrics basics, OpenTelemetry tracing concepts, health-check vs readiness-check endpoints.
- **Hands-On:** Add structured JSON logging, a `/health` and `/ready` endpoint, and basic Prometheus metrics for request count/latency.
- **Exercises:** Trace a request ID through logs across two middleware layers; add a custom business metric (e.g., orders created); simulate a dependency failure and verify `/ready` reflects it.

### Chapter 21: Security Hardening
- **Theory:** OWASP API Top 10 relevant to FastAPI, security headers, secrets management, input sanitization beyond Pydantic validation, dependency-confusion and supply-chain basics.
- **Hands-On:** Add a security-headers middleware, audit the auth system from Ch. 11 against common attack patterns, integrate secret loading from environment/secret manager rather than hardcoded values.
- **Exercises:** Find and fix an injection risk introduced deliberately in sample code; add security headers and verify with a scanner tool; write a checklist for auditing any new endpoint before merge.

### Chapter 22: Task Queues and Distributed Processing
- **Theory:** Why `BackgroundTasks` breaks down at scale, message queue concepts, Celery vs ARQ vs a lightweight Redis-queue approach, task retries and dead-letter patterns.
- **Hands-On:** Replace the Ch. 13 background task with a proper task queue (ARQ or Celery + Redis) supporting retries.
- **Exercises:** Add a retry policy with backoff to a flaky task; add a dead-letter path for permanently failed tasks; compare latency/throughput of `BackgroundTasks` vs the queue under load.

### Chapter 23: Serving Machine Learning Models with FastAPI
- **Theory:** Sync vs async inference endpoints, model loading at startup (`lifespan` events) vs per-request, batching requests for throughput, GPU-bound vs CPU-bound serving considerations.
- **Hands-On:** Wrap a small model (or a stub inference function) behind a FastAPI endpoint using `lifespan` for model loading, with request batching for a queue of inputs.
- **Exercises:** Move model loading from "on first request" to `lifespan` startup and measure cold-start difference; add a request-batching layer with a max wait time; add a `/metrics` endpoint reporting inference latency.

### Chapter 24: Containerizing and Deploying FastAPI
- **Theory:** Writing a production Dockerfile (multi-stage builds, `uv` for fast installs), Uvicorn/Gunicorn worker configuration, environment-based config, an overview of FastAPI Cloud vs self-managed deploys (ECS/Cloud Run/AKS-GKE-EKS).
- **Hands-On:** Write a multi-stage Dockerfile for the full project, run it locally with `docker compose` alongside Postgres and Redis.
- **Exercises:** Reduce the final image size measurably via multi-stage build; configure worker count based on CPU cores; deploy the container to one cloud target of your choice and document the steps.

### Chapter 25: CI/CD for FastAPI Services
- **Theory:** Test/lint/build/deploy pipeline design, `ruff` for linting, running the pytest suite in CI, image build-and-push, basic blue/green or rolling deploy concepts.
- **Hands-On:** Build a GitHub Actions pipeline: lint → test → build image → push → (optionally) deploy.
- **Exercises:** Add a required status check that blocks merge on failing tests; cache dependencies in CI to speed up builds; add a smoke test that hits `/health` post-deploy.

### Chapter 26: Microservices and Multi-Service Architectures
- **Theory:** Service boundaries and shared-nothing design, service-to-service auth (API keys/mTLS/JWT propagation), API gateway patterns, sync (HTTP) vs async (event-driven) communication between services.
- **Hands-On:** Split the project into two services (e.g., "Orders" and "Inventory") communicating over HTTP with service-to-service auth, plus one async event via a message broker.
- **Exercises:** Add a circuit-breaker-style timeout/retry when calling the other service; replace one synchronous call with an async event; document the resulting architecture diagram.

### Chapter 27: GraphQL and Alternative API Styles with FastAPI
- **Theory:** When REST isn't the right fit, GraphQL integration via Strawberry, gRPC as an alternative for internal service-to-service calls, tradeoffs vs plain REST.
- **Hands-On:** Add a GraphQL endpoint (Strawberry) alongside the existing REST API for the same domain, and compare developer experience.
- **Exercises:** Write a GraphQL query solving an over-fetching problem that REST couldn't solve cleanly; identify one endpoint that should stay REST and justify why; sketch (no need to fully implement) how the Orders/Inventory split from Ch. 26 could use gRPC internally.

### Chapter 28: Capstone — Full Production System
- **Theory:** Bringing every prior chapter together into one deployable system; a design-review checklist covering architecture, security, testing, observability, and deployment.
- **Hands-On (choose one track, or propose your own):**
  1. **ML-Serving Track:** A production inference API (auth, rate limiting, batching, observability, Dockerized, CI/CD) serving a real model.
  2. **Device/Control Track:** A FastAPI service exposing REST + WebSocket endpoints to control and monitor a piece of hardware or simulated device in real time, with an async job queue for longer-running operations.
  3. **SaaS-Style Track:** A multi-tenant CRUD platform with auth, roles, background jobs, caching, and a public + admin API surface.
- **Exercises:** Run your own system through the Ch. 28 design-review checklist and fix at least three gaps; write a short architecture doc (1–2 pages) as if handing this off to another engineer; identify the single next bottleneck you'd address if traffic grew 10x.

---

## Appendix A — Suggested Tooling Baseline
- **Package/env management:** `uv`
- **Linting/formatting:** `ruff`
- **Testing:** `pytest`, `pytest-asyncio`, `httpx`
- **DB:** SQLModel + `asyncpg` (Postgres) or `aiosqlite` (SQLite) + Alembic
- **Cache/queue:** Redis, ARQ or Celery
- **Observability:** structlog or standard `logging` with JSON formatting, Prometheus client, OpenTelemetry (optional, advanced)
- **Containerization:** Docker + `docker compose`

## Appendix B — Reference Material Worth Reading Alongside This Curriculum
- Official FastAPI documentation (canonical source — updates fastest, especially release notes)
- Pydantic v2 migration guide
- Starlette documentation (for anything below FastAPI's abstraction layer)
- OWASP API Security Top 10

---

**Next step:** once you've reviewed this list, tell me which chapter to start with and how deep to go (quick reference vs. full walkthrough with runnable code), and we'll build it out.
