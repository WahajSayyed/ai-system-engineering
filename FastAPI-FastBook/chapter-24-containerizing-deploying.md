# Chapter 24: Containerizing and Deploying FastAPI

> Part IV — Deployment & Production Systems · Chapter 24 of 28

Every chapter so far has run via `fastapi dev`. This chapter makes the application deployable: a real multi-stage Dockerfile, correct Uvicorn/Gunicorn worker configuration (directly informed by Chapter 23's CPU-vs-GPU-bound distinction), and a genuine comparison between FastAPI Cloud and self-managed container platforms — current as of FastAPI Cloud's public beta launch in June 2026.

## Learning Objectives

By the end of this chapter you will be able to:

- Write a multi-stage Dockerfile that keeps build tools out of the final image, using `uv` for fast, reproducible installs from a lockfile.
- Explain why a container's `CMD` must use exec form, not shell form, and what silently breaks if you get this wrong.
- Configure worker process count correctly for CPU-bound serving, and explain why Chapter 23's GPU-bound case is a deliberate exception to the usual formula.
- Run the full application — API, Postgres, Redis — locally via `docker compose`, configured entirely through environment variables.
- Compare FastAPI Cloud against self-managed container platforms (Cloud Run, GKE/EKS/AKS) and choose appropriately for a given project's needs.

---

## 24.1 Why Multi-Stage

A naive, single-stage Dockerfile installs everything — compilers, build headers, `uv` itself, pip's download cache — directly into the image you actually ship, because there's only one stage and nothing to separate "what I needed to *build* this" from "what I need to *run* this." A **multi-stage** build uses one stage (the "builder") to install dependencies and compile anything that needs compiling, then copies *only* the finished, runnable artifacts into a clean final stage — discarding the builder stage, and everything it accumulated, entirely.

## 24.2 A Real Multi-Stage Dockerfile

```dockerfile
# ---- Builder stage ----
FROM python:3.13-slim AS builder

WORKDIR /code

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# ---- Final stage ----
FROM python:3.13-slim

WORKDIR /code

COPY --from=builder /code/.venv /code/.venv
COPY --from=builder /code/app /code/app

ENV PATH="/code/.venv/bin:$PATH"

EXPOSE 80

CMD ["fastapi", "run", "app/main.py", "--port", "80", "--workers", "4"]
```

`uv sync --frozen` installs *exactly* what `uv.lock` specifies — no re-resolution, no "whatever the latest compatible version happens to be today" — reproducible builds, the same guarantee a lockfile gives you anywhere else. Splitting the dependency-install step (`COPY pyproject.toml uv.lock ./` then `uv sync`) from the code-copy step (`COPY . .`) is a deliberate Docker layer-caching optimization: as long as your dependencies haven't changed, Docker reuses the cached dependency-install layer even when your application code changes constantly — meaningfully faster rebuilds during active development.

**`CMD` must be written in exec form — a JSON array, `["fastapi", "run", ...]` — never shell form (`CMD fastapi run app/main.py ...`), and this is not a style preference.** Shell form runs your command as a child process *of a shell*, which means the container's stop signal (`SIGTERM`, sent during a graceful deploy or restart) is delivered to the shell, not directly to `fastapi`/Uvicorn — the shell typically does not forward it to the child process at all. The practical consequence: your `lifespan` shutdown code (Chapter 9's `engine.dispose()`, Chapter 22's ARQ pool cleanup) **never runs**, because the process that owns that shutdown logic never actually receives the signal telling it to run it — the container is simply killed outright once its grace period expires, having never had a real chance to shut down cleanly. Exec form delivers `SIGTERM` directly to the `fastapi`/Uvicorn process itself, which correctly triggers graceful shutdown and runs your `lifespan` code's post-`yield` half exactly as intended. Exercise 24.4 has you reproduce this failure on purpose.

## 24.3 Worker Configuration — Chapter 23's Distinction, Made Concrete

**A single Uvicorn process uses exactly one CPU core.** To use more of a multi-core machine, you run multiple **worker processes** — not threads, since Python's GIL (Chapter 2, Chapter 23) prevents true CPU parallelism within one process regardless of thread count. `fastapi run` (which wraps Uvicorn) supports this directly:

```dockerfile
CMD ["fastapi", "run", "app/main.py", "--port", "80", "--workers", "4"]
```

A common starting heuristic for CPU-bound, general-purpose services: **`(2 × CPU core count) + 1`** worker processes — a widely used rule of thumb, not a law, and one that assumes a fairly typical mix of I/O waiting and computation. **This is exactly where Chapter 23's GPU-bound exception matters concretely, not just theoretically**: applying this formula blindly to a GPU-inference service on an 8-core machine (suggesting 17 workers) would be actively harmful — 17 processes each attempting to load the same multi-gigabyte model onto one shared GPU will exhaust VRAM long before reaching anywhere near 17. For a GPU-bound service, worker count is governed by *GPU count*, not CPU core count — typically one worker process per available GPU, full stop, regardless of how many CPU cores the machine otherwise has to offer.

For more mature process supervision (automatically restarting a crashed worker, more configurable logging), Gunicorn as a process manager running Uvicorn workers underneath it remains a common, battle-tested pattern:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 app.main:app --bind 0.0.0.0:80
```

Both approaches solve the same problem — Uvicorn's own `--workers` flag has matured enough that it's a perfectly reasonable default choice; Gunicorn remains the more configurable, more battle-tested option for teams with existing operational tooling built around it.

## 24.4 Environment-Based Configuration, in Containers

Nothing changes conceptually from Chapter 16/18 — the same `Settings` class, reading environment variables, works identically whether those variables come from a local `.env` file, a `docker run -e KEY=value`, or a `docker-compose.yml`'s `environment:` section. **The image itself should be entirely environment-agnostic** — the same built image runs correctly in development, staging, and production, differing only in which environment variables are injected at container-start time, never in anything baked into the image at build time.

## 24.5 FastAPI Cloud vs. Self-Managed Deployment

**FastAPI Cloud**, built by the FastAPI team itself, entered public beta in June 2026. It's genuinely worth knowing about specifically because it's opinionated in a way generic cloud platforms aren't — built for FastAPI applications specifically, not adapted from a framework-agnostic container platform:

```bash
uv run fastapi deploy
```

That's the entire deploy command for an existing FastAPI project (the `fastapi-cloud-cli` package is bundled by default with `fastapi[standard]`). What it currently provides: zero-config deploys (detecting `pyproject.toml`/`uv.lock`/`requirements.txt` automatically), automatic HTTPS via TLS certificates, autoscaling, zero-downtime rolling deployments, **automatic deployment verification** — if a new deployment appears broken, it keeps your previous working version serving traffic rather than cutting over to something failing — CI/CD integration via `fastapi cloud setup-ci` (wiring up GitHub Actions automatically), built-in application metrics (request traffic, CPU, memory, replica count), and custom domains.

**Self-managed alternatives** remain the right choice under different circumstances: **Cloud Run** for a containerized app wanting autoscaling with less platform lock-in than a FastAPI-specific tool, at the cost of handling cold starts, concurrency settings, and timeouts yourself; **GKE/EKS/AKS** (managed Kubernetes) when your FastAPI application is one piece of a larger, multi-service platform and you need fine-grained control over networking, scaling policy, and release strategy across many services, not just this one — precisely the scenario Chapter 26's microservices architecture describes.

**The practical guidance:** FastAPI Cloud is the fastest path to production for a FastAPI-specific application that doesn't yet need complex multi-service orchestration — genuinely comparable to "no Dockerfile, no Nginx config, no fighting environment variables at 2 AM" for that specific case. Reach for self-managed Kubernetes or a cloud-native container service specifically once you need to orchestrate this application *alongside* several others as part of one coordinated system, or need infrastructure control FastAPI Cloud's opinionated defaults don't expose.

---

## Hands-On Project: The Full Stack, Containerized

### Step 1 — The Dockerfile (section 24.2, as shown above)

### Step 2 — `docker-compose.yml`: API, Postgres, and Redis together

```yaml
services:
  api:
    build: .
    ports:
      - "8000:80"
    environment:
      DATABASE_URL: postgresql+asyncpg://appuser:${POSTGRES_PASSWORD}@db:5432/appdb
      REDIS_URL: redis://cache:6379/0
      SECRET_KEY: ${SECRET_KEY}
    depends_on:
      - db
      - cache

  db:
    image: postgres:16
    environment:
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: appdb
    volumes:
      - pg_data:/var/lib/postgresql/data

  cache:
    image: redis:7-alpine
    command: ["redis-server", "--requirepass", "${REDIS_PASSWORD}"]

volumes:
  pg_data:
```

This is Chapter 9's Postgres swap and Chapter 19's Redis, finally running together in one place — with every credential supplied via environment variables (`${POSTGRES_PASSWORD}`, `${SECRET_KEY}`, read from a local `.env` file `docker compose` picks up automatically, itself git-ignored per Chapter 21.6), never hardcoded in the compose file itself.

### Step 3 — Build and run

```bash
docker compose up --build
```

Confirm the full application — API, database, cache — comes up correctly, and that `GET /ready` (Chapter 20) reports both dependencies healthy.

---

## Practice Exercises

**Exercise 24.1 — Measure the multi-stage size reduction directly.**
Write a naive, single-stage version of the Dockerfile (installing `uv`, syncing dependencies, and copying code all within one `FROM python:3.13-slim` stage, no `--from=builder` copying at all). Build both versions and compare `docker images` output for their sizes. Report the size difference in MB and as a percentage.

**Exercise 24.2 — Compute worker count from CPU cores at container startup.**
Replace the hardcoded `--workers 4` with a small entrypoint script that computes `(2 * os.cpu_count()) + 1` and passes it to `fastapi run` dynamically. Then explicitly write out, in a comment or a short paragraph, the specific circumstance under which you would *override* this formula and hardcode a much lower number instead — referencing Chapter 23 directly.

**Exercise 24.3 — Deploy to one real target and document every step.**
Choose either `fastapi deploy` (FastAPI Cloud) or a self-managed target (`gcloud run deploy`, or an equivalent for your preferred provider). Actually deploy this chapter's containerized application, and write a short, step-by-step document (a README section is fine) covering every command you ran, every configuration value you had to supply, and the final live URL. Note anything that surprised you or took longer than expected.

**Exercise 24.4 — Reproduce the shell-form `CMD` bug.**
Temporarily change your Dockerfile's `CMD` to shell form (`CMD fastapi run app/main.py --port 80`). Add a `print("Shutting down — cleaning up ARQ pool")` line to your `lifespan` function's post-`yield` section (Chapter 22). Run the container, then `docker stop` it, and check whether that print statement appears in the container's logs before it stops. Revert to exec form, repeat, and confirm the difference.

**Exercise 24.5 (stretch) — `--proxy-headers` and Chapter 19's rate limiter.**
Add an Nginx (or Traefik) reverse proxy in front of your API within `docker-compose.yml`, so all traffic passes through it before reaching the FastAPI container. Without `--proxy-headers` on your `fastapi run` command, make several requests from different (simulated, if needed) clients and check what `request.client.host` reports inside your application — is it each distinct client's real address, or is it the proxy's own internal address for every single request? Explain what this means for Chapter 19's per-IP sliding-window rate limiter if left unfixed, then add `--proxy-headers` and confirm real client IPs are correctly recovered from the `X-Forwarded-For` header the proxy adds.

---

## Solutions & Discussion

<details>
<summary>Exercise 24.1</summary>

A typical result: a naive single-stage image might land somewhere in the 300–500+ MB range (depending on exactly which build tools and caches accumulate), while the multi-stage version — shipping only the final `.venv` and application code, with `uv` itself, pip's cache, and any compilation toolchain entirely absent from the final image — commonly lands considerably smaller, often 40–60% smaller or more, depending on how many compiled dependencies your project has. The exact numbers vary by project, but the direction and rough magnitude should be clearly, measurably different — not a marginal difference you'd need to squint at.
</details>

<details>
<summary>Exercise 24.2</summary>

```dockerfile
CMD ["sh", "-c", "fastapi run app/main.py --port 80 --workers $(python -c 'import os; print(2 * os.cpu_count() + 1)')"]
```

(Note: this specific line uses shell form deliberately for the arithmetic substitution — worth flagging as a nuance: if graceful-shutdown correctness matters here, prefer computing the worker count in a tiny wrapper script invoked via exec form, rather than accepting shell form for the final process itself.)

The override circumstance, stated directly: **if this service is doing GPU-bound inference (Chapter 23), do not use this formula at all** — hardcode worker count to the number of available GPUs (commonly exactly 1 per GPU), regardless of CPU core count, because VRAM — not CPU cores — is the actual constraint governing how many worker processes can safely coexist, each attempting to load a model onto the same physical GPU.
</details>

<details>
<summary>Exercise 24.3</summary>

Answers will vary by chosen platform, but a complete answer should include: the exact deploy command(s) run, every environment variable/secret that had to be configured on the platform (database URL, `SECRET_KEY`, Redis URL), the resulting live URL, and total time from "container built locally" to "reachable in production." A `fastapi deploy` answer should be able to report this in a handful of steps and a couple of minutes; a Cloud Run or Kubernetes answer will legitimately involve more steps (registry push, service/deployment YAML, exposing a load balancer) — the point of this exercise is experiencing that difference directly, not just reading about it in section 24.5's comparison.
</details>

<details>
<summary>Exercise 24.4</summary>

With shell-form `CMD`: `docker stop` sends `SIGTERM` to the shell process wrapping `fastapi run`, which typically does not forward that signal to the actual Uvicorn/FastAPI process underneath it — the container's logs show no `"Shutting down..."` line at all before the container is forcibly killed once Docker's stop grace period (10 seconds by default) expires. Your `lifespan` function's shutdown code never ran.

With exec-form `CMD`: `docker stop` delivers `SIGTERM` directly to the `fastapi`/Uvicorn process, which correctly triggers graceful shutdown — your `lifespan` function's post-`yield` code runs, the `"Shutting down..."` line appears in the logs, and the container exits cleanly, generally well before the grace period would need to force-kill it. This is section 24.2's warning, made directly observable rather than taken on faith.
</details>

<details>
<summary>Exercise 24.5</summary>

Without `--proxy-headers`: `request.client.host` inside your application reports the reverse proxy's own internal address (e.g., the Docker network IP of the Nginx container) for **every single request**, regardless of which real external client actually made it — because, from the FastAPI process's point of view, every request genuinely does arrive as a direct connection from the proxy, which is the only thing it's actually connected to at the TCP level. Chapter 19's rate limiter, keying by `request.client.host`, would therefore lump every distinct real-world client together under one shared key — either applying the intended per-client limit against the *combined* traffic of every client at once (locking everyone out together once the aggregate crosses the threshold) or, depending on exact logic, never meaningfully distinguishing clients from each other at all. This is a genuinely common, easy-to-miss production bug: a rate limiter that worked correctly in local testing (no proxy in front of it) silently breaks the moment a reverse proxy or load balancer is introduced.

Adding `--proxy-headers` tells Uvicorn to trust and read `X-Forwarded-For` (and related headers) that a properly configured reverse proxy adds, recovering the real original client address into `request.client.host` correctly — restoring the rate limiter's actual per-client behavior. Worth noting as a security caveat: `--proxy-headers` should only be enabled when you genuinely trust whatever sits in front of your application to set these headers accurately — trusting `X-Forwarded-For` from an untrusted or misconfigured source would let a malicious client simply lie about its own IP by setting that header itself.
</details>

---

## Chapter Summary

- Multi-stage Dockerfiles keep build-time tools and caches out of the final image — a builder stage installs and compiles, a clean final stage copies only the finished, runnable artifacts.
- `CMD` must use exec form (a JSON array), never shell form — shell form prevents `SIGTERM` from reaching your application process directly, silently breaking graceful shutdown and skipping `lifespan` cleanup code entirely.
- Worker process count should be based on CPU cores for typical services (`(2 × cores) + 1` as a starting heuristic) — but GPU-bound services (Chapter 23) are a deliberate exception, governed by GPU count, not CPU count.
- The same environment-variable-driven `Settings` pattern from Chapters 16 and 18 applies unchanged to containers — the image itself stays environment-agnostic, configured entirely at runtime.
- FastAPI Cloud (public beta as of June 2026) offers a genuinely fast, FastAPI-specific path to production; self-managed platforms (Cloud Run, GKE/EKS/AKS) remain the right choice once you need to orchestrate this service alongside several others, or need infrastructure control a specific platform's opinionated defaults don't expose.

**Next:** Chapter 25 covers CI/CD — automating the lint/test/build/deploy pipeline this chapter's manual `docker compose up` and deploy commands have so far required running by hand.
