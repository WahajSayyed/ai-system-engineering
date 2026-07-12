# Chapter 23: Serving Machine Learning Models with FastAPI

> Part IV — Deployment & Production Systems · Chapter 23 of 28

Part IV moves from general production engineering to specific deployment scenarios. This chapter is the first: serving a model behind FastAPI, done correctly — which turns out to revisit Chapter 2's sync/async rule in a context where getting it wrong is expensive, Chapter 9's `lifespan` pattern applied to model weights instead of database tables, and a genuinely advanced pattern — request batching — that's the single biggest throughput lever for GPU-bound serving.

## Learning Objectives

By the end of this chapter you will be able to:

- Choose correctly between `def` and `async def` for an inference endpoint, and explain why ML inference changes Chapter 9's I/O-bound calculus.
- Load a model exactly once, at startup, via `lifespan` — and explain the concrete race condition a naive "lazy load on first request" pattern has.
- Build a request-batching layer that accumulates concurrent requests into batches, bounded by size or wait time, without blocking any individual request's own coroutine.
- Explain why GPU-bound serving has different scaling considerations than CPU-bound serving, particularly around worker process count and VRAM.

---

## 23.1 Sync vs Async Inference: Revisiting Chapter 2's Rule

Chapter 9 established the async database rule around **I/O-bound** work: a database call spends most of its time *waiting* (network round trip, disk seek), and an async driver lets the event loop do other useful work during that wait. ML inference is a genuinely different shape of work: it's typically **CPU-bound** (or GPU-bound) — actual computation happening, not waiting for something external. Most inference libraries (PyTorch, `transformers`, `onnxruntime`, scikit-learn) are fundamentally synchronous under the hood, even when wrapped in Python — calling `model.predict(...)` doesn't *await* anything, it computes, using the CPU or GPU, for however long that takes.

This means the safe default from Chapter 2/3 still applies, for a reason worth restating precisely: **a plain `def` inference route, letting FastAPI's thread pool absorb the blocking computation, is generally the safer choice than `async def`** — an `async def` route calling a synchronous inference function directly would block the *entire* event loop for the full duration of every single inference call, exactly Chapter 2's Part C bug, now with real computational cost behind it instead of a simulated `time.sleep`.

But the thread pool isn't a free lunch either, and this is where ML serving genuinely differs from the database case: **the GIL still applies to pure-Python/CPU-bound work running in threads**, meaning multiple concurrent inference calls in separate threads don't truly parallelize on CPU the way I/O-bound work does (a thread waiting on I/O releases the GIL; a thread doing tight numeric computation mostly doesn't). And for GPU-bound inference, the GPU itself is typically one shared physical resource per process — concurrent inference calls, even from different threads, effectively queue up at the GPU regardless of how many threads requested them. This is precisely why section 23.3's batching pattern isn't a nice-to-have optimization for ML serving — it's often *the* mechanism that actually lets a GPU serve meaningful concurrent throughput at all, rather than serializing requests one at a time behind the scenes regardless of how the web layer is structured.

## 23.2 Model Loading: `lifespan`, Not Per-Request, Not Lazy

Two anti-patterns worth naming explicitly, because both are easy to write without noticing the problem:

**Loading the model inside the route function** — reloading potentially gigabytes of weights from disk (and, for a GPU model, transferring them into VRAM) on *every single request* — is catastrophic for latency and is the most obviously wrong version, rarely written deliberately but occasionally arrived at by accident (a helper function that "just loads and returns a model," called from inside a route without anyone noticing it does real work every time).

**Lazy loading on first request** — a global variable, checked and populated inside the route (`if model is None: model = load_model()`) — looks reasonable and works most of the time, but has two real problems. First, whichever request happens to arrive first pays the *entire* cold-start cost, unluckily, with no way to know in advance which client that will be. Second, and more seriously: **under concurrent traffic at startup, multiple requests can each see `model is None` before either has finished loading it**, triggering the (expensive, possibly GPU-memory-allocating) load logic more than once, concurrently — a genuine race condition, not a hypothetical one. Exercise 23.4 has you reproduce this concretely.

The correct pattern, using exactly Chapter 9's `lifespan` mechanism, now loading model weights instead of creating database tables:

```python
# ml/model.py
import time
import numpy as np

class StubModel:
    """Stands in for a real model (PyTorch, onnxruntime, scikit-learn, ...).
    __init__ simulates real weight-loading cost; predict simulates real per-batch compute cost."""

    def __init__(self):
        time.sleep(2)   # simulating loading a multi-GB checkpoint from disk
        self.weights = np.random.rand(512, 10)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        time.sleep(0.05 + 0.002 * len(batch))   # cost grows sub-linearly with batch size — the whole point of batching
        return batch @ self.weights
```

```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from ml.model import StubModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app.state.model = StubModel()
    print("Model loaded — ready to serve.")
    yield
    # a real GPU model might explicitly release VRAM here on shutdown

app = FastAPI(lifespan=lifespan)
```

Loading happens exactly once, **before the application accepts any traffic at all** — there is no window, ever, where a request could observe a not-yet-loaded model, and no possibility of two requests racing to load it concurrently, because loading isn't triggered by requests in the first place.

## 23.3 Batching for Throughput

Here's the core insight batching exploits: a GPU (and, to a lesser degree, vectorized CPU operations) is dramatically more *efficient* processing many inputs at once than processing them one at a time in sequence — per-call overhead (kernel launch, memory transfer) is largely fixed regardless of batch size, so amortizing it across a batch of 16 inputs instead of paying it 16 separate times is a real, often large throughput win. A naive per-request inference endpoint — one input in, one prediction out, called independently for every concurrent request — leaves most of that efficiency on the table.

The pattern: incoming requests don't call the model directly. Instead, each one pushes its input onto a shared queue and *waits* for its own result; a single background consumer drains that queue, accumulates a batch bounded by **either** a maximum size **or** a maximum wait time (whichever limit is hit first), runs the model *once* against the whole batch, and routes each individual result back to the request that's waiting for it.

```python
# ml/batcher.py
import asyncio
import numpy as np

class BatchingInferenceQueue:
    def __init__(self, model, max_batch_size: int = 16, max_wait_ms: int = 20):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None

    def start(self):
        self._consumer_task = asyncio.create_task(self._consume())

    async def stop(self):
        if self._consumer_task:
            self._consumer_task.cancel()

    async def predict(self, features: list[float]) -> list[float]:
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self.queue.put((features, future))
        return await future   # yields control here — doesn't block anything else

    async def _consume(self):
        while True:
            batch_items = [await self.queue.get()]
            deadline = asyncio.get_event_loop().time() + (self.max_wait_ms / 1000)
            while len(batch_items) < self.max_batch_size:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    batch_items.append(await asyncio.wait_for(self.queue.get(), timeout=timeout))
                except asyncio.TimeoutError:
                    break

            features_batch = np.array([item[0] for item in batch_items])
            results = await asyncio.to_thread(self.model.predict, features_batch)

            for (_, future), result in zip(batch_items, results):
                if not future.done():
                    future.set_result(result.tolist())
```

Walking through why this is correct rather than just plausible: `predict(...)` is `async def`, and its only real work is `await self.queue.put(...)` and `await future` — both yield control back to the event loop, per Chapter 2's cooperative-multitasking model, so one request awaiting its result doesn't block any other concurrent request's own `predict(...)` call from proceeding. The single `_consume` background task is the *only* thing that ever calls the actual (blocking) model — and it does so via `asyncio.to_thread(...)`, offloading that blocking call to a thread so that even the consumer's own inference call doesn't freeze the event loop while it runs. Each request's `future` is resolved individually once the shared batch it happened to land in has been processed, letting that specific request's `await future` line resume and return, independent of every other request's own timing.

## 23.4 GPU-Bound vs. CPU-Bound Serving

A GPU is typically one shared physical resource per visible device — this has real, concrete consequences distinct from CPU serving. **VRAM is a hard ceiling**: running multiple worker *processes*, each independently loading the same model onto the same GPU (a completely reasonable-looking way to scale a CPU-bound service — more processes, more throughput), can trivially exhaust VRAM if the model is large enough that even two or three copies don't fit. GPU-bound services generally run a **low process count per GPU — often exactly one** — and lean on batching (section 23.3), not process count, as the primary throughput lever. CPU-bound serving has more headroom to scale via additional worker processes (Chapter 24's territory), bounded more gracefully by core count and system memory rather than a hard, unforgiving VRAM ceiling.

---

## Hands-On Project: Model Loading, Naive Inference, and a Real Batching Layer

### Step 1 — `StubModel` and `lifespan` loading (section 23.2, as shown above)

### Step 2 — A naive, unbatched endpoint, for comparison

```python
# routers/inference.py
from typing import Annotated
from fastapi import APIRouter, Depends, Request
import numpy as np
from ml.model import StubModel

router = APIRouter(prefix="/ml", tags=["ml"])

def get_model(request: Request) -> StubModel:
    return request.app.state.model

@router.post("/predict")
def predict(payload: list[float], model: Annotated[StubModel, Depends(get_model)]):
    batch = np.array([payload])
    result = model.predict(batch)
    return {"prediction": result[0].tolist()}
```

Deliberately `def`, not `async def` — per section 23.1, `model.predict` is a blocking call, and FastAPI's thread pool is the correct place for it to run without freezing the event loop.

### Step 3 — Wire the batching layer into `lifespan`, and add a batched endpoint

```python
# main.py (updated)
from ml.batcher import BatchingInferenceQueue

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = StubModel()
    app.state.batcher = BatchingInferenceQueue(app.state.model, max_batch_size=16, max_wait_ms=20)
    app.state.batcher.start()
    yield
    await app.state.batcher.stop()
```

```python
# routers/inference.py (addition)
def get_batcher(request: Request) -> BatchingInferenceQueue:
    return request.app.state.batcher

@router.post("/predict-batched")
async def predict_batched(payload: list[float], batcher: Annotated[BatchingInferenceQueue, Depends(get_batcher)]):
    result = await batcher.predict(payload)
    return {"prediction": result}
```

### Step 4 — Confirm batching actually happens

Fire 20 concurrent requests at `/ml/predict-batched` via `asyncio.gather` and add a `print(f"batch of {len(batch_items)}")` line inside `_consume`. You should see requests arriving close together genuinely grouped into batches larger than 1 — direct evidence the mechanism is working, not just trusted by inspection.

---

## Practice Exercises

**Exercise 23.1 — Measure the cold-start difference directly.**
Build a lazy-loading version (`model: StubModel | None = None` at module level; the route checks and loads if `None`). Time the *first* request against it (should include the full ~2s load) versus subsequent ones. Then time the *first* request against the `lifespan`-loaded version, and confirm it's already fast — the load cost was paid before your timer even started, during application startup.

**Exercise 23.2 — Tune `max_wait_ms` and observe the tradeoff.**
Run the same 20-concurrent-request test from Step 4 with `max_wait_ms=5`, then `max_wait_ms=50`. Report the average batch size and the average per-request latency for each setting. Explain, in your own words, why a smaller `max_wait_ms` produces smaller, less efficient batches with lower added latency, while a larger one produces bigger, more efficient batches at the cost of every request waiting longer, even under light load.

**Exercise 23.3 — Add an inference-latency metric.**
Using Chapter 20's Prometheus pattern, add `INFERENCE_DURATION = Histogram("ml_inference_duration_seconds", "Model inference duration")`, observing it around the `await asyncio.to_thread(self.model.predict, ...)` call inside `_consume`. Confirm `/metrics` shows this Histogram populated after a few batched requests, distinct from the generic `http_request_duration_seconds` every endpoint already gets.

**Exercise 23.4 — Reproduce the lazy-load race condition concretely.**
Add a class-level counter to `StubModel.__init__` (`StubModel.load_count += 1`) in the lazy-loading version from Exercise 23.1. Fire several genuinely concurrent requests (via `asyncio.gather`, all targeting the endpoint at the exact same moment, before any of them could have finished loading) against a *freshly restarted* server, and check `StubModel.load_count` afterward. Confirm it's greater than 1 — the model was loaded more than once, concurrently, exactly the race condition section 23.2 described.

**Exercise 23.5 (stretch) — Reason through the GPU case.**
Without necessarily running this against a real GPU, answer: if `StubModel` were a real multi-GB PyTorch model loaded onto a single GPU, how many Uvicorn/worker processes would be safe to run on one machine with one GPU, and why? If you do have access to a real GPU, load an actual small model onto it, run the batching load test from Step 4, and watch `nvidia-smi` (or equivalent) during the run — does GPU utilization look meaningfully different with `max_batch_size=1` (effectively disabling batching) versus `max_batch_size=16`?

---

## Solutions & Discussion

<details>
<summary>Exercise 23.1</summary>

The lazy-loading version's first request takes roughly 2+ seconds (the full simulated load, plus inference); every subsequent request is fast, just the ~50-100ms inference cost. The `lifespan`-loaded version's *first real request* is already fast — indistinguishable from its hundredth — because the 2-second cost was paid once, during startup, before the server logged "ready to serve" and before your client's timer even began. This is the concrete difference between "someone pays the cold-start cost, unluckily, at an unpredictable moment" and "the cost is paid once, deliberately, at a moment nobody's actual request is waiting on."
</details>

<details>
<summary>Exercise 23.2</summary>

With `max_wait_ms=5`: expect a smaller average batch size (the consumer doesn't wait long before giving up and processing whatever arrived), and lower added latency per request beyond the model's own inference time — but less amortization benefit, since fewer requests get bundled together. With `max_wait_ms=50`: expect closer to (or at) `max_batch_size` on average under this test's concurrent load, and a somewhat higher minimum latency per request (since even a request that arrives when the queue is otherwise empty may now wait up to 50ms hoping others join it) — but each batch does proportionally more useful work per model invocation. The tradeoff is fundamentally about who you're optimizing for: a smaller wait suits latency-sensitive traffic with modest concurrency; a larger wait suits high-throughput batch-style traffic willing to trade a little per-request latency for meaningfully better aggregate efficiency.
</details>

<details>
<summary>Exercise 23.3</summary>

```python
from prometheus_client import Histogram
INFERENCE_DURATION = Histogram("ml_inference_duration_seconds", "Model inference duration")

# inside _consume:
with INFERENCE_DURATION.time():
    results = await asyncio.to_thread(self.model.predict, features_batch)
```

`/metrics` now shows `ml_inference_duration_seconds` with real observed samples, distinct from `http_request_duration_seconds` — the HTTP-level metric measures the *entire* request/response cycle (queueing time plus inference time plus response serialization), while this metric isolates *just* the model's own compute time, letting you distinguish "requests are slow because inference itself is slow" from "requests are slow because they're waiting in the batching queue" — two different problems with two different fixes.
</details>

<details>
<summary>Exercise 23.4</summary>

```python
class StubModel:
    load_count = 0

    def __init__(self):
        StubModel.load_count += 1
        time.sleep(2)
        self.weights = np.random.rand(512, 10)
```

Firing several concurrent requests against a freshly restarted server (before any of them could have completed the ~2-second load) commonly results in `StubModel.load_count` being 2 or more — multiple requests each observed `model is None` before any of them had finished setting it, and each proceeded to construct their own separate `StubModel()` instance, each independently paying the full load cost, wastefully and redundantly, exactly the race condition section 23.2 named without yet proving. This is directly analogous to Chapter 2's async concurrency lessons: without a guard (a lock, or — far simpler — never letting this race be possible at all by loading before any request-handling concurrency exists, which is exactly what `lifespan` provides), concurrent access to shared, lazily-initialized state is a genuine bug, not a rare edge case.
</details>

<details>
<summary>Exercise 23.5</summary>

For a real multi-GB model on one GPU: typically **exactly one** worker process per GPU is the safe default. Each additional process attempting to load its own full copy of the model onto the same physical GPU directly competes for the same finite VRAM — two processes each loading a model that individually uses 60% of available VRAM will simply fail to both fit, with the second process's load either OOM-crashing outright or (worse) succeeding in a way that leaves too little VRAM headroom for actual inference batches, causing intermittent OOM failures under load rather than a clean, immediate startup failure. Multiple *requests*, by contrast, are meant to be handled by that single process's batching layer (section 23.3) — batching is the throughput lever for a GPU-bound service, not process count, precisely because process count is constrained by VRAM in a way CPU-bound serving's process count isn't nearly as tightly constrained by system RAM.

If you ran this against a real GPU: with `max_batch_size=1` (effectively one-at-a-time inference), `nvidia-smi`'s utilization percentage during the load test would typically show a "sawtooth" pattern — spikes during each individual inference call, followed by mostly-idle GPU time between them while the next request is fetched, queued, and dispatched individually. With `max_batch_size=16`, utilization during the same load test should look meaningfully higher and steadier — the GPU spends a much larger fraction of the test actually computing on non-trivial batches, rather than sitting comparatively idle between many small, individually-dispatched single-item calls. This observable utilization difference is the concrete, visible version of section 23.3's efficiency argument, not just a claim to take on faith.
</details>

---

## Chapter Summary

- ML inference is typically CPU/GPU-bound, not I/O-bound — a plain `def` route (thread pool) is generally the safer default for a synchronous inference library, exactly per Chapter 2/3's rule, now applied to computation rather than I/O.
- Model loading belongs in `lifespan`, loaded exactly once before any request-handling concurrency exists — lazy loading on first request has a genuine, reproducible race condition under concurrent startup traffic, not just a "unlucky first request" latency problem.
- Request batching — accumulating concurrent requests into a batch bounded by size or wait time, then running the model once per batch — is the core throughput lever for GPU-bound serving, implemented here with an `asyncio.Queue`, per-request `asyncio.Future`s, and `asyncio.to_thread` to keep the batching consumer itself from blocking the event loop.
- GPU-bound serving generally runs one process per GPU (VRAM is a hard ceiling that additional processes compete for directly), leaning on batching rather than process count as the primary scaling lever — a real difference from CPU-bound serving's more forgiving scaling characteristics.

**Next:** Chapter 24 covers containerizing and deploying FastAPI — multi-stage Dockerfiles, Uvicorn/Gunicorn worker configuration, and where this chapter's "one process per GPU" constraint concretely shows up in a real deployment's container and orchestration configuration.
