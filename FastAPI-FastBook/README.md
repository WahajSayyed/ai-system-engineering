# FastAPI: From Foundations to Production

A complete, code-first, 28-chapter curriculum — beginner through advanced — built around one running project (a Products API) that gains a database, auth, a task queue, observability, and a full deployment pipeline as the chapters progress. Every chapter follows the same shape: **Theory → Hands-On Project → Practice Exercises → Solutions & Discussion**.

**Stack:** FastAPI 0.136+ · Python 3.13 · Pydantic v2 (v1 no longer supported) · SQLModel / async SQLAlchemy 2.0 · `uv` · pytest + httpx · Docker · Redis · ARQ / Celery

---

## How to Use This Repo

Work through the chapters in order — each one assumes every prior chapter's code exists and builds directly on top of it. The running example (a Products API) accumulates functionality chapter by chapter: it's a plain in-memory CRUD API in Part I, gains a real database and auth in Part II, becomes layered, cached, observable, and hardened in Part III, and ends up containerized, tested in CI, and split across service boundaries in Part IV.

Each chapter's **Hands-On Project** is meant to be actually typed and run, not just read. The **Practice Exercises** at the end of each chapter have full solutions in collapsible `<details>` sections — GitHub renders these natively, so attempt an exercise before expanding its solution.

---

## Table of Contents

### [Curriculum Outline](./fastapi-curriculum.md)
The full 28-chapter syllabus with learning objectives per chapter, before diving into any individual one.

### Part I — Foundations
| Ch. | Title | Core Topic |
|---|---|---|
| 1 | [Why FastAPI](./chapter-01-why-fastapi.md) | ASGI vs WSGI, the three pillars, how `/docs` actually works |
| 2 | [Python Foundations](./chapter-02-python-foundations.md) | Type hints, `async`/`await` mechanics, decorators, dataclasses vs Pydantic |
| 3 | [Path Operations and Routing](./chapter-03-path-operations-and-routing.md) | HTTP methods, path parameters, the route-ordering bug, `APIRouter` |
| 4 | [Request Data](./chapter-04-request-data.md) | `Query`/`Path`/`Body`, validation constraints, the `422` error shape |
| 5 | [Pydantic Deep Dive](./chapter-05-pydantic-deep-dive.md) | `field_validator`/`model_validator`, `ConfigDict`, computed fields, `TypeAdapter` |
| 6 | [Response Models and Status Codes](./chapter-06-response-models-and-status-codes.md) | `response_model`, input/output separation, the double-validation trap |
| 7 | [Error Handling](./chapter-07-error-handling.md) | Custom exceptions, one consistent error envelope, the catch-all handler |

### Part II — Intermediate: Building Real APIs
| Ch. | Title | Core Topic |
|---|---|---|
| 8 | [Dependency Injection](./chapter-08-dependency-injection.md) | `Depends()`, dependency trees, `yield`-based setup/teardown, caching |
| 9 | [Databases I](./chapter-09-databases-sqlmodel.md) | SQLModel, async SQLAlchemy 2.0, sessions, connection pooling |
| 10 | [Databases II](./chapter-10-migrations-and-query-patterns.md) | Alembic, N+1 queries, transactions, the repository pattern |
| 11 | [Authentication and Authorization](./chapter-11-authentication-authorization.md) | OAuth2, JWTs, `pwdlib`/Argon2, role-based access control |
| 12 | [Middleware, CORS, and the Request Lifecycle](./chapter-12-middleware-cors.md) | Middleware ordering, CORS mechanics, the full request lifecycle |
| 13 | [Background Tasks](./chapter-13-background-tasks.md) | `BackgroundTasks`, its real limitations, idempotency |
| 14 | [File Uploads, Static Files, Streaming](./chapter-14-file-uploads-streaming.md) | Chunked uploads, `StreamingResponse`, `StaticFiles` |
| 15 | [Testing FastAPI Applications](./chapter-15-testing.md) | `TestClient`/`AsyncClient`, dependency overrides, fixtures, mocking |

### Part III — Advanced: Production Engineering
| Ch. | Title | Core Topic |
|---|---|---|
| 16 | [Advanced Pydantic](./chapter-16-advanced-pydantic.md) | Generic models, `pydantic-settings`, custom types, discriminated unions |
| 17 | [WebSockets](./chapter-17-websockets.md) | Connection lifecycle, broadcasting, WebSockets vs SSE vs polling |
| 18 | [Application Architecture at Scale](./chapter-18-application-architecture.md) | The service layer, directory structure, API versioning, sub-apps |
| 19 | [Performance](./chapter-19-performance.md) | `ORJSONResponse`, Redis caching, sliding-window rate limiting, profiling |
| 20 | [Observability](./chapter-20-observability.md) | Structured logging, `contextvars`, health vs readiness, Prometheus |
| 21 | [Security Hardening](./chapter-21-security-hardening.md) | OWASP API Top 10, BOLA/BOPLA, security headers, secrets management |
| 22 | [Task Queues](./chapter-22-task-queues.md) | ARQ, retries with backoff, dead-letter patterns |

### Part IV — Deployment & Production Systems
| Ch. | Title | Core Topic |
|---|---|---|
| 23 | [Serving ML Models](./chapter-23-serving-ml-models.md) | `lifespan` model loading, sync/async inference, request batching |
| 24 | [Containerizing and Deploying](./chapter-24-containerizing-deploying.md) | Multi-stage Docker, worker configuration, FastAPI Cloud vs self-managed |
| 25 | [CI/CD](./chapter-25-cicd.md) | GitHub Actions, `ruff`, image tagging, smoke tests |
| 26 | [Microservices](./chapter-26-microservices.md) | Service boundaries, service-to-service auth, sync vs async communication |
| 27 | [GraphQL and Alternative API Styles](./chapter-27-graphql-grpc.md) | Strawberry, gRPC, when REST isn't the right fit |
| 28 | [Capstone](./chapter-28-capstone.md) | The full design-review checklist and three project tracks |

### Bonus / Supplementary
| Title | Description |
|---|---|
| [FastAPI + Celery + Redis Guide](./fastapi-celery-redis-guide.md) | A standalone deep-dive on the Celery-based alternative to Chapter 22's ARQ-based task queue — architecture, multi-queue routing, retries, production Docker Compose setup |

---

## Recurring Threads Worth Tracking Across Chapters

A few things get introduced early and paid off much later — worth noticing as you go, not just treated as isolated facts:

- **The route-ordering bug** (Ch. 3) resurfaces in Ch. 14's `/export.csv`.
- **`from_attributes=True`** (Ch. 5, "your bridge to ORMs") is cashed in the moment a real database appears (Ch. 9).
- **`request_id`** is generated once, ad hoc, inside one exception handler (Ch. 7) — then properly propagated through every log line in the codebase via `contextvars` (Ch. 20).
- **The Notes API's BOLA vulnerability** exists, unnoticed, from Chapter 3 onward — found and fixed explicitly in Ch. 21.
- **`BackgroundTasks`' three named gaps** (durability, retries, multi-worker correctness — Ch. 13) are each resolved concretely in Ch. 22.
- **Chapter 18's layering rule** ("repositories never import `fastapi`") becomes an automated CI gate in Ch. 25, not just a convention.

---

## Prerequisites

Comfortable with Python (functions, classes, basic OOP). No prior async or web framework experience assumed — Chapter 2 covers async fundamentals from scratch before Chapter 3 needs them.
