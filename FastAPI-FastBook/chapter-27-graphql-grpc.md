# Chapter 27: GraphQL and Alternative API Styles with FastAPI

> Part IV — Deployment & Production Systems · Chapter 27 of 28

Every endpoint this curriculum has built returns a fixed shape, declared in advance (Chapter 6). This chapter covers what happens when that stops fitting — GraphQL via Strawberry, FastAPI's own officially recommended integration, alongside gRPC as the right alternative specifically for internal service-to-service calls like Chapter 26's Orders/Inventory boundary.

## Learning Objectives

By the end of this chapter you will be able to:

- Explain over-fetching and under-fetching precisely, and why GraphQL addresses both at once.
- Integrate Strawberry with FastAPI via `GraphQLRouter`, reusing the exact repository layer (Chapter 10, 18) your REST routes already depend on.
- Explain a critical, verified difference between how FastAPI and Strawberry handle synchronous resolver functions — one that breaks an assumption Chapter 2/3 established.
- Recognize when a request is genuinely better served by REST than GraphQL, and justify that choice concretely.
- Explain, at a sketch level, why gRPC fits internal service-to-service calls better than it fits a public-facing API.

---

## 27.1 Over-Fetching and Under-Fetching

Every REST response this curriculum has built returns a fixed shape — `ProductPublic`, declared once, returned in full to every caller regardless of what that specific caller actually needs. A client that only wants a product's `name` and `price` still receives every field `ProductPublic` declares — **over-fetching**. A client that needs a product *and* its reviews *and* its tags needs three separate REST calls (Chapter 10's relationships, each behind its own endpoint) — **under-fetching**, requiring multiple round trips for what is conceptually one logical request.

**GraphQL's core idea directly addresses both**: the client specifies exactly which fields it wants, across however many related resources, in a single request:

```graphql
query {
  product(productId: 1) {
    name
    price
    reviews {
      rating
    }
  }
}
```

One round trip, exactly the fields asked for — no more, no less — spanning a relationship (`reviews`) that would have been a separate REST call entirely. This is a genuine capability REST's fixed-shape-per-endpoint model can't cleanly replicate without either a proliferation of narrowly-tailored endpoints (`/products/{id}/summary`, `/products/{id}/with-reviews`, ...) or accepting the over/under-fetching cost.

**This is not a strict upgrade, and FastAPI's own documentation is explicit about that**: GraphQL solves specific use cases, with real trade-offs against REST worth weighing rather than assuming away. Caching is genuinely harder — REST's cacheable-by-URL model (a `GET /products/1` response can be cached by its URL alone) doesn't translate cleanly to GraphQL, where every query can request a differently-shaped response from the same endpoint. Section 27.4 covers a real correctness risk (N+1, resurfacing at the resolver level) that's easier to introduce accidentally in GraphQL than in a well-structured REST API. And a single, sufficiently complex GraphQL query can be far more expensive to execute than any single REST call typically is, since the client controls the query's shape and depth, not just which endpoint to hit.

## 27.2 Strawberry: FastAPI's Officially Recommended GraphQL Library

FastAPI's own documentation recommends **Strawberry** specifically because its design mirrors FastAPI's own: schemas are defined via Python type annotations, not a separate schema language or a class-heavy alternative approach (Graphene, an older option). Strawberry provides a `GraphQLRouter` — mountable via `include_router`, exactly like every other router this curriculum has built:

```python
# graphql_schema.py
import strawberry
from typing import Optional

@strawberry.type
class ReviewType:
    id: int
    rating: int
    comment: Optional[str]

@strawberry.type
class ProductType:
    id: int
    name: str
    price: float
    currency: str

    @strawberry.field
    async def reviews(self, info: strawberry.types.Info) -> list[ReviewType]:
        repo = info.context["review_repo"]
        reviews = await repo.list_for_product(self.id)
        return [ReviewType(id=r.id, rating=r.rating, comment=r.comment) for r in reviews]

@strawberry.type
class Query:
    @strawberry.field
    async def product(self, info: strawberry.types.Info, product_id: int) -> Optional[ProductType]:
        repo = info.context["product_repo"]
        product = await repo.get_or_raise(product_id)
        return ProductType(id=product.id, name=product.name, price=product.price, currency=product.currency)

schema = strawberry.Schema(query=Query)
```

```python
# main.py
from strawberry.fastapi import GraphQLRouter
from graphql_schema import schema

async def get_graphql_context(
    product_repo: Annotated[ProductRepository, Depends(get_product_repository)],
    review_repo: Annotated[ReviewRepository, Depends(get_review_repository)],
):
    return {"product_repo": product_repo, "review_repo": review_repo}

graphql_router = GraphQLRouter(schema, context_getter=get_graphql_context)
app.include_router(graphql_router, prefix="/graphql")
```

**`context_getter` is itself a genuine FastAPI dependency** — it can declare its own `Depends(...)` parameters exactly like any route, which is what makes this integration architecturally clean rather than a bolt-on: `get_graphql_context` here injects the *exact same* `ProductRepository`/`ReviewRepository` instances (Chapter 10, wired through Chapter 18's layering) that your REST routes already use. This is a direct, concrete payoff of Chapter 18's architecture: the repository layer doesn't know or care whether it's being called from a REST route or a GraphQL resolver — both are just callers, and the actual data-access logic exists in exactly one place either way.

## 27.3 A Critical, Verified Gotcha: Strawberry Resolvers Don't Get FastAPI's Thread-Pool Protection

Here's a detail worth stating precisely, because assuming otherwise reproduces Chapter 2's blocking bug in a place you wouldn't expect it: **FastAPI's own routing gives a plain `def` route automatic protection — it runs in a thread pool, safely, without you doing anything extra (Chapter 2/3). Strawberry does not extend that same protection to its own resolvers.** Both synchronous *and* asynchronous Strawberry field resolvers run on the event loop — meaning a synchronous `def` resolver that does anything blocking will freeze the **entire worker**, exactly Chapter 2's Part C bug, except this time it happens despite the function looking like an ordinary, unremarkable `def`, the kind that would have been perfectly safe as a FastAPI route.

```python
# WRONG — blocks the entire event loop, despite looking like an ordinary sync function
@strawberry.field
def slow_resolver(self, info) -> str:
    time.sleep(2)   # or any blocking call — this freezes every concurrent request
    return "done"

# RIGHT — genuinely async, or explicitly offloaded
@strawberry.field
async def fast_resolver(self, info) -> str:
    await asyncio.sleep(2)
    return "done"

# If you must call blocking code from a resolver:
from starlette.concurrency import run_in_threadpool

@strawberry.field
async def wrapped_resolver(self, info) -> str:
    result = await run_in_threadpool(some_blocking_function)
    return result
```

The rule to carry forward: **write every Strawberry resolver as `async def`**, and if a resolver genuinely must call blocking code, wrap that specific call in `run_in_threadpool` explicitly — don't rely on the thread-pool safety net Chapter 2/3 established for ordinary FastAPI routes, because Strawberry simply doesn't provide the equivalent for its own resolvers. Exercise 27.4 has you reproduce this concretely, so the warning isn't just an assertion to take on faith.

## 27.4 N+1, Resurfacing at the Resolver Level

Chapter 10 fixed a very specific N+1 pattern with `selectinload`. GraphQL's flexible, client-controlled nested queries make the *same* underlying problem easier to reintroduce, and harder to notice in your own code — because whether it happens now depends on what shape of query a *client* happens to send, not on anything visible in a fixed REST route:

```graphql
query {
  products(limit: 50) {
    name
    reviews {   # if this resolver naively queries per-product, 50 products = 50 extra queries
      rating
    }
  }
}
```

If `ProductType.reviews`'s resolver (section 27.2) issues one query per product it's called for, a request for 50 products with their reviews reproduces Chapter 10.2's exact N+1 pattern — 1 query for the products, 50 more for their reviews — except now triggered by whatever query shape a client happened to send, not by anything a code reviewer would necessarily catch by reading your resolver in isolation. The standard fix, the **DataLoader** pattern, batches and caches resolver calls *within a single request* — conceptually similar to Chapter 23's request-batching idea, applied here to database lookups instead of ML inference: instead of each `reviews` resolver call querying independently, a DataLoader accumulates all the product IDs asked for during one GraphQL request and issues one batched query for all of them, then hands each resolver call its own slice of the batched result. Strawberry supports DataLoaders directly; a full implementation is beyond this chapter's scope, but recognizing the pattern's necessity — and that it's solving the *same* N+1 problem Chapter 10 named, just resurfacing in a new context — is the point.

## 27.5 gRPC: The Right Fit for Internal Service-to-Service Calls

Recall Chapter 26's Orders service calling Inventory's `/reserve` endpoint over plain HTTP/JSON. **gRPC** is a genuine alternative for exactly this kind of internal, service-to-service boundary — built on HTTP/2, using Protocol Buffers (`.proto` files) to define a strongly-typed contract, with client and server code *generated* from that single contract in whatever languages each service happens to be written in.

```protobuf
// inventory.proto (sketch)
service InventoryService {
  rpc ReserveStock(ReserveRequest) returns (ReserveResponse);
}

message ReserveRequest {
  int32 product_id = 1;
  int32 quantity = 2;
}

message ReserveResponse {
  bool reserved = 1;
}
```

From this one file, both Orders (the client) and Inventory (the server) generate strongly-typed stubs — no manually-written `httpx` calls, no manually-kept-in-sync request/response schemas duplicated across two codebases (Chapter 26's `InventoryServiceError`-wrapped `httpx.AsyncClient` call, hand-written and hand-maintained, is exactly what gRPC's generated stubs replace). Binary serialization over HTTP/2 is also meaningfully more efficient than JSON over HTTP/1.1 for high-throughput internal traffic.

**Why this doesn't generally replace your public-facing REST API**: browser support for gRPC is limited — a browser can't natively speak gRPC without a translation layer (`grpc-web` plus a compatible proxy), adding real infrastructure most public APIs have no reason to take on. And gRPC's code-generation-first model, ideal when every party involved can regenerate stubs from a shared `.proto` file on a coordinated release cadence, is a much bigger ask for arbitrary third-party API consumers than "send JSON to a documented HTTP endpoint" — exactly the accessibility Chapter 1 named as part of FastAPI's whole value proposition. gRPC earns its complexity specifically at internal boundaries, within one organization's control, where all parties can coordinate on contract changes together — precisely Chapter 26's Orders/Inventory boundary, and exactly why Exercise 27.3 asks you to sketch it there rather than anywhere public-facing.

---

## Hands-On Project: GraphQL Alongside REST, for the Same Domain

### Step 1 — The schema and router (section 27.2, as shown above), mounted at `/graphql`

### Step 2 — Confirm both APIs work, side by side, backed by the same repository layer

Visit `/graphql`'s built-in GraphiQL interface (Strawberry provides one automatically) and run:

```graphql
query {
  product(productId: 1) {
    name
    price
  }
}
```

Compare against `GET /products/1` (REST) — note the REST response includes every `ProductPublic` field regardless of whether you wanted them all, while the GraphQL query returns exactly `name` and `price`, nothing more.

### Step 3 — A query spanning a relationship, in one round trip

```graphql
query {
  product(productId: 1) {
    name
    reviews {
      rating
    }
  }
}
```

Compare this single request against the REST equivalent, which would require `GET /products/1` *and* a separate `GET /products/1/reviews`-style call — two round trips for what this one GraphQL query accomplishes in one.

---

## Practice Exercises

**Exercise 27.1 — Quantify an over-fetching fix.**
Using your actual `ProductPublic` schema (Chapter 6), count how many fields it declares. Write a GraphQL query requesting only 2 of them, plus each returned product's review ratings (not full review objects — just the `rating` field). Compare: how many total fields of data would the REST equivalent transfer (across however many calls it would take), versus this one GraphQL query? Report both counts.

**Exercise 27.2 — Identify one endpoint that should stay REST, and justify it.**
Pick one endpoint from earlier in this curriculum — Chapter 14's file upload, or Chapter 20's `/health`/`/ready` — and argue concretely why it should remain a plain REST endpoint rather than being reimplemented as a GraphQL mutation or query. Ground your argument in something specific from this chapter (GraphQL's core strengths — flexible field selection, spanning relationships in one round trip — and whether this particular endpoint's use case actually benefits from either).

**Exercise 27.3 — Sketch gRPC for the Orders/Inventory boundary.**
Write a complete `.proto` sketch (like section 27.5's) covering both Chapter 26's `reserve_stock` call and a second operation of your choosing (e.g., checking current stock without reserving). Explain what code would be generated from this file for each side of the boundary, and why this fits the Orders↔Inventory relationship specifically better than it would fit the public-facing Products REST API a browser-based frontend consumes directly.

**Exercise 27.4 — Reproduce the sync-resolver blocking gotcha, concretely.**
Add a deliberately synchronous `def` resolver (not `async def`) to your schema that calls `time.sleep(2)`. Fire several concurrent GraphQL queries at it (via `asyncio.gather` from a test client) and measure total elapsed time. Confirm it's roughly `2 × (number of concurrent requests)` — serialized, not concurrent — despite this looking like an ordinary function that would have been perfectly safe as a plain FastAPI route. Fix it (either make it genuinely `async def`, or wrap the blocking call in `run_in_threadpool`) and confirm the same concurrent-request test now completes in roughly 2 seconds total, not `2 × N`.

**Exercise 27.5 (stretch) — Reproduce GraphQL's own N+1, and describe the DataLoader fix.**
Using the query-counting technique from Chapter 10.2, confirm that requesting `reviews` for 20 different products in one GraphQL query issues 20 separate review-fetching queries (assuming `ProductType.reviews`'s resolver queries independently per product, as written in section 27.2). Without necessarily implementing a full DataLoader, describe in a paragraph how a DataLoader would change this: what would it batch, when would the batched query actually run, and how would each individual resolver call get its own correct slice of the result back — drawing the parallel to Chapter 23's request-batching pattern explicitly.

---

## Solutions & Discussion

<details>
<summary>Exercise 27.1</summary>

If `ProductPublic` declares, say, 8 fields (`id`, `name`, `price`, `currency`, `in_stock`, `tags`, `created_at`, `display_price`), the REST equivalent for 10 products would transfer 80 total field-values via `GET /products/` alone, plus a separate call per product for reviews (10 more round trips, each returning full review objects). The GraphQL query requesting just `name`, `price`, and `reviews { rating }` for those same 10 products transfers 20 field-values for the product data (2 fields × 10 products) plus only each review's `rating` (not the full review object) — a substantially smaller total payload, fetched in one round trip rather as many as 11. The exact numbers depend on your schema, but the *shape* of the difference — GraphQL transferring a small, precisely-requested fraction of what REST's fixed shape would have sent, in fewer round trips — should be clearly visible regardless of your exact field count.
</details>

<details>
<summary>Exercise 27.2</summary>

Chapter 14's file upload is a strong candidate: GraphQL's core spec doesn't natively handle binary file uploads well — multipart uploads are an explicit, opt-in extension in modern GraphQL server implementations, not a first-class part of the query language the way a REST `multipart/form-data` `POST` endpoint (Chapter 14) already handles cleanly and directly. There's also no meaningful "over-fetching" or "under-fetching" concern for a file upload — the client isn't selecting which *fields* of a response it wants; it's sending bytes and getting back a simple confirmation. GraphQL's actual strengths (flexible field selection, spanning relationships in one query) simply don't apply to this operation's shape, making the added complexity of a GraphQL mutation for it a cost with no corresponding benefit.

Chapter 20's `/health`/`/ready` are an even simpler case: these are infrastructure-level checks, consumed by orchestrators and load balancers expecting a plain HTTP status code and a minimal body — introducing a query language layer for "is this process alive" adds complexity serving no client need at all, since there's no field selection or relationship-spanning question being asked in the first place.
</details>

<details>
<summary>Exercise 27.3</summary>

```protobuf
service InventoryService {
  rpc ReserveStock(ReserveRequest) returns (ReserveResponse);
  rpc CheckStock(CheckStockRequest) returns (CheckStockResponse);
}

message ReserveRequest {
  int32 product_id = 1;
  int32 quantity = 2;
}
message ReserveResponse {
  bool reserved = 1;
}

message CheckStockRequest {
  int32 product_id = 1;
}
message CheckStockResponse {
  int32 available_quantity = 1;
}
```

From this file, Inventory (the server) generates a base class implementing both RPCs, and Orders (the client) generates a strongly-typed client stub exposing `reserve_stock(...)` and `check_stock(...)` as ordinary-looking method calls — no manually-written `httpx` request construction, no manually-kept-in-sync `ReserveRequest`/`ReserveResponse` shapes maintained independently on each side the way Chapter 26's hand-written client and route currently are. This fits the Orders↔Inventory boundary specifically because both services are part of one coordinated system — a schema change to this `.proto` file can be rolled out to both sides together, by the same team, on a coordinated release. The public-facing Products REST API has neither property: its consumers include arbitrary third-party clients and, plausibly, direct browser JavaScript (Chapter 12's CORS setup exists specifically for this), for whom "regenerate your gRPC stubs from our `.proto` file" is a dramatically higher-friction integration story than "send a GET request and read the JSON," and browsers can't natively speak gRPC at all without additional proxy infrastructure.
</details>

<details>
<summary>Exercise 27.4</summary>

```python
import time

@strawberry.type
class Query:
    @strawberry.field
    def slow_sync_field(self) -> str:   # deliberately NOT async def
        time.sleep(2)
        return "done"
```

Firing 5 concurrent queries against this field via `asyncio.gather` takes roughly **10 seconds** total (5 × 2s, serialized) — the blocking `time.sleep(2)` inside a sync resolver freezes Strawberry's event-loop-based execution for its full duration, and since Strawberry (unlike FastAPI's own route handling) doesn't run sync resolvers in a thread pool automatically, every other concurrent GraphQL request queues up behind it, one at a time.

```python
from starlette.concurrency import run_in_threadpool

@strawberry.field
async def fixed_field(self) -> str:
    await run_in_threadpool(time.sleep, 2)
    return "done"
```

The same 5-concurrent-request test against this fixed version completes in roughly **2 seconds** total — `run_in_threadpool` explicitly offloads the blocking call to a thread, freeing the event loop to make progress on the other 4 concurrent requests while any one of them waits, exactly restoring the concurrency Chapter 2's model predicts for correctly-offloaded blocking work.
</details>

<details>
<summary>Exercise 27.5</summary>

Using Chapter 10.2's query-counting event listener around a GraphQL request for `products(limit: 20) { name reviews { rating } }`: expect **21** queries — 1 for the products, 20 more (one per product) for their individually-resolved `reviews` fields, exactly Chapter 10's N+1 pattern, now triggered by a GraphQL query's shape rather than a hand-written loop.

A DataLoader would change this by **batching resolver calls within one request**: instead of each product's `reviews` resolver immediately querying the database the moment it's called, it registers "I need reviews for product X" with a shared DataLoader instance, and *returns a pending promise* rather than querying right away. The DataLoader, once all 20 individual registrations for this one GraphQL request have been collected (conceptually similar to Chapter 23's batching consumer accumulating requests before running one shared model call), issues **one** query — `SELECT * FROM reviews WHERE product_id IN (1, 2, ..., 20)` — and then resolves each individual product's pending promise with its own slice of that one batched result. The net effect: 20 individual "please fetch my reviews" requests collapse into 2 total queries (1 for products, 1 batched query for every product's reviews combined) — the same two-queries-total outcome Chapter 10.2's `selectinload` achieved for a fixed REST route, now achieved for an arbitrarily-shaped, client-controlled GraphQL query via the DataLoader pattern instead.
</details>

---

## Chapter Summary

- GraphQL solves over-fetching (fixed REST shapes returning more than a client needs) and under-fetching (relationships requiring separate round trips) simultaneously — a genuine capability, not a strict upgrade, with real costs around caching and query cost control that REST's simpler model doesn't share.
- Strawberry is FastAPI's officially recommended GraphQL library, integrating via `GraphQLRouter` and a `context_getter` that is itself a real FastAPI dependency — letting GraphQL resolvers reuse the exact repository layer (Chapter 10, 18) your REST routes already depend on.
- Strawberry resolvers do **not** get FastAPI's automatic thread-pool protection for sync functions — every resolver should be `async def`, with blocking calls explicitly wrapped in `run_in_threadpool` when unavoidable, or an ordinary-looking `def` resolver will silently freeze the entire worker.
- N+1 resurfaces at the GraphQL resolver level, driven by whatever query shape a client sends rather than anything visible in your own route code — the DataLoader pattern (batching resolver calls within one request) is the standard fix, directly analogous to Chapter 10's `selectinload` and Chapter 23's request-batching.
- gRPC fits internal, same-organization service-to-service boundaries (Chapter 26's Orders/Inventory) well — strongly-typed, code-generated contracts, efficient binary serialization over HTTP/2 — but doesn't generally replace a public-facing REST API, given limited native browser support and a much higher integration cost for arbitrary third-party consumers.

**Next:** Chapter 28 is the capstone — bringing every chapter in this curriculum together into one complete, deployable system, across your choice of three project tracks.
