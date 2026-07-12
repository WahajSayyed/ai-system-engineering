# Chapter 14: File Uploads, Static Files, and Streaming Responses

> Part II — Intermediate: Building Real APIs · Chapter 14 of 28

Every request and response body so far has been small enough that "just build it in memory" was never a real concern. This chapter is about the moment that stops being true — accepting a large uploaded file without loading it all into RAM at once, generating a large response the same way, and serving static assets efficiently.

## Learning Objectives

By the end of this chapter you will be able to:

- Choose between `bytes` and `UploadFile` for accepting file uploads, and explain the memory trade-off between them.
- Stream an upload to disk in fixed-size chunks, keeping peak memory bounded regardless of file size.
- Use `StreamingResponse` to generate a large response incrementally instead of building it entirely in memory first.
- Serve static files correctly via `StaticFiles`, and explain how `app.mount(...)` differs from `app.include_router(...)`.
- Validate an upload's actual type using its file signature, not just a client-supplied `content_type` header.

---

## 14.1 `bytes` vs `UploadFile`

FastAPI gives you two ways to accept an uploaded file, and the difference is entirely about memory:

```python
from fastapi import File, UploadFile

@app.post("/upload-bytes")
async def upload_bytes(file: bytes = File(...)):
    # `file` is the ENTIRE file content, already loaded into memory, as a bytes object
    ...

@app.post("/upload-file")
async def upload_file(file: UploadFile):
    # `file` is a wrapper around a spooled temporary file — small uploads stay in
    # memory, larger ones spill to disk automatically, and you read it incrementally
    ...
```

`file: bytes = File(...)` reads the *entire* upload into memory the moment FastAPI parses the request — simple, and fine for genuinely small files, but it means a 500 MB upload requires 500 MB of RAM just to receive it, before your route has done anything at all. `UploadFile` is backed by Starlette's `SpooledTemporaryFile` — small files stay in memory, but once an upload exceeds a size threshold, it automatically spills to a temporary file on disk, and you interact with it through async `read()`/`write()`/`seek()` methods rather than a single already-materialized blob. It also carries `filename` and `content_type` as attributes directly, without you needing extra form fields to capture them. For anything beyond a trivially small file, `UploadFile` is the correct default.

## 14.2 The Gotcha: `UploadFile` Doesn't Save You If You `.read()` It All at Once

Here's the part that's easy to miss: choosing `UploadFile` over `bytes` doesn't automatically protect you from loading everything into memory — it only *can* avoid that, if you actually read it in chunks.

```python
# Still loads the ENTIRE file into memory, despite using UploadFile!
contents = await file.read()
```

`await file.read()`, called with no size argument, reads everything, exactly like the `bytes` version would — `UploadFile`'s memory advantage only materializes if you actually read it incrementally, in a loop, processing (or writing to disk) each chunk before asking for the next one:

```python
CHUNK_SIZE = 1024 * 1024  # 1 MB

with destination.open("wb") as out_file:
    while chunk := await file.read(CHUNK_SIZE):
        out_file.write(chunk)
```

This keeps peak memory bounded to roughly one chunk's worth, regardless of whether the uploaded file is 1 MB or 1 GB — the difference is only in how many loop iterations it takes, not in how much memory any single iteration requires. This is the same principle Chapter 10's `selectinload` applied to database rows, applied here to bytes: process data incrementally instead of materializing all of it at once, and memory usage stops scaling with the size of what you're processing.

## 14.3 `StreamingResponse`: The Same Idea, on the Way Out

A normal route return value — a dict, a list, a Pydantic model — gets fully serialized into a complete response body in memory *before* any of it is sent to the client. For a large generated response (a big CSV export, say), this means holding the *entire* output in memory at once, exactly the same problem section 14.2 just solved for uploads, but on the response side instead.

`StreamingResponse` takes a generator (sync or async) and sends each chunk to the client as it's produced, rather than waiting for the whole thing to be ready first:

```python
from fastapi.responses import StreamingResponse

async def generate_rows():
    yield "id,name,price\n"
    for i in range(1_000_000):
        yield f"{i},Product {i},{i * 1.5:.2f}\n"

@app.get("/export.csv")
async def export_csv():
    return StreamingResponse(generate_rows(), media_type="text/csv")
```

Two real benefits, not just one: memory stays bounded (you're never holding all 1,000,000 rows as one string at once), *and* the client starts receiving data immediately, rather than waiting for the entire dataset to be generated server-side before the first byte goes out. For a genuinely large export, this can be the difference between a response that starts arriving in milliseconds and one where the client sits waiting with no data at all until the full thing is assembled.

## 14.4 Serving Static Files

```python
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="uploads"), name="static")
```

`app.mount(...)` is different from `app.include_router(...)` in a way worth being precise about: `include_router` adds routes to *your* FastAPI application's own routing table (Chapter 3); `mount` attaches an entirely separate, independent ASGI application at a URL prefix — `StaticFiles` is its own minimal ASGI app, not a FastAPI router, and it handles the details of serving files from disk itself (reading the requested path, setting correct content types, handling range requests for partial content, and etags for caching) without you writing any of that logic. A request to `/static/products/42.jpg` gets handed entirely to the mounted `StaticFiles` app, which resolves it against `directory="uploads"` and serves `uploads/products/42.jpg` directly from disk — conceptually the same "match a path that can contain slashes" problem Chapter 3's `:path` converter solved by hand, here handled for you by a purpose-built ASGI app instead.

---

## Hands-On Project: Product Image Uploads and a Streaming CSV Export

### Step 1 — Chunked, validated image upload

```python
# routers/products.py (additions)
from pathlib import Path
from fastapi import UploadFile, File, HTTPException, status

UPLOAD_DIR = Path("uploads/products")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
CHUNK_SIZE = 1024 * 1024


@router.post("/{product_id}/image")
async def upload_product_image(product_id: int, repo: ProductRepoDep, file: UploadFile = File(...)):
    await repo.get_or_raise(product_id)   # 404s if the product doesn't exist — Chapter 7's NotFoundError, as always

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    destination = UPLOAD_DIR / f"{product_id}{Path(file.filename).suffix}"
    size = 0

    with destination.open("wb") as out_file:
        while chunk := await file.read(CHUNK_SIZE):
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                out_file.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds the {MAX_UPLOAD_SIZE // (1024*1024)} MB limit",
                )
            out_file.write(chunk)

    return {"product_id": product_id, "image_path": str(destination)}
```

This is worth pausing on: the size check here counts **actual bytes read**, chunk by chunk, as the upload streams in — rejecting and cleaning up mid-upload the moment the running total crosses the limit, rather than only checking a client-supplied `Content-Length` header up front. Recall Chapter 12's Exercise 12.2 flagged exactly this gap in a header-only size check (a client can omit or misstate `Content-Length`, particularly with chunked transfer encoding) — this is the robust version that exercise was gesturing toward: the limit is enforced against bytes you've actually received, not a number the client merely claimed in advance.

### Step 2 — Serve uploaded images as static files

```python
# main.py (addition)
from fastapi.staticfiles import StaticFiles

app.mount("/static/products", StaticFiles(directory="uploads/products"), name="product_images")
```

An uploaded image for product 42 is now reachable directly at `/static/products/42.jpg` — no FastAPI route involved in serving it at all, just the mounted `StaticFiles` app resolving the path against disk.

### Step 3 — A streaming CSV export, querying in batches

```python
# routers/products.py (addition)
import csv
import io
from fastapi.responses import StreamingResponse


async def generate_csv_rows(repo: ProductRepository):
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["id", "name", "price", "currency", "in_stock"])
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    offset = 0
    batch_size = 100
    while True:
        products = await repo.list(limit=batch_size, offset=offset)
        if not products:
            break
        for p in products:
            writer.writerow([p.id, p.name, p.price, p.currency, p.in_stock])
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        offset += batch_size


@router.get("/export.csv")
async def export_products_csv(repo: ProductRepoDep):
    return StreamingResponse(
        generate_csv_rows(repo),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=products.csv"},
    )
```

**Route ordering, again:** `/export.csv` must be declared *before* `/{product_id}` in this router, for exactly the reason Chapter 3.3 established — `/products/export.csv` would otherwise match `/products/{product_id}` first, fail `int` coercion on `"export.csv"`, and produce a `422` instead of ever reaching this endpoint. The rule from Chapter 3 didn't stop applying just because six chapters have passed; it applies to every new fixed-string route you add for the rest of this curriculum.

This export does two things well simultaneously, worth noticing as one combined win rather than two separate tricks: it queries the database in **batches** (the same `limit`/`offset` pagination pattern from Chapters 4 and 9–10), never holding the entire product table in memory at once, *and* it streams the response out incrementally via `StreamingResponse`, never holding the entire generated CSV string in memory either. A product catalog of 10 or 10 million makes no difference to this endpoint's memory footprint — only to how long it takes and how many chunks get sent.

---

## Practice Exercises

**Exercise 14.1 — Don't trust the client-supplied content type.**
`file.content_type` in this chapter's upload handler is a value the *client* set in the multipart form data — nothing stops a malicious or buggy client from renaming a file and manually declaring any `content_type` it likes, regardless of the file's actual contents. Add a `verify_magic_bytes` check that reads the first several bytes of the upload and confirms they match the expected file signature for the claimed type (JPEG starts with `FF D8 FF`; PNG starts with `89 50 4E 47`), rejecting the upload with a `415` if they don't match — even if `content_type` claimed to be valid. Remember to `seek(0)` back to the start of the file after peeking at its header, so the rest of your chunked-read logic still works correctly.

**Exercise 14.2 — Confirm bounded memory on a genuinely large streamed export.**
Temporarily seed enough products (a script inserting, say, 500,000 rows directly via the repository or raw SQL is fine) to make `export.csv` a multi-hundred-thousand-row response. Using `tracemalloc` (or an external tool like watching the process's RSS memory via `psutil` or your OS's process monitor) around the `generate_csv_rows` generator's execution, confirm that memory usage stays roughly flat throughout the export rather than growing proportionally to the number of rows. Report the peak memory you observed.

**Exercise 14.3 — Build the "bad" buffered version and compare.**
Write a second export endpoint, `/export-buffered.csv`, that builds the *entire* CSV as one big Python string (looping through every product, accumulating rows into a single `io.StringIO()`, calling `.getvalue()` only once at the end) and returns a plain `Response` with that complete string, instead of using `StreamingResponse`. Using the same `tracemalloc` measurement technique from Exercise 14.2, compare peak memory between this version and the streaming version, for the same dataset size. Report both numbers and explain the difference in your own words.

**Exercise 14.4 — Reproduce and fix the route-ordering bug, in this new context.**
Temporarily move `/export.csv`'s route definition to *after* `/{product_id}` in the router. Call `GET /products/export.csv` and confirm you get a `422` rather than a CSV file, reproducing Chapter 3.3's bug in a new setting. Move it back above `/{product_id}` and confirm it works again.

**Exercise 14.5 (stretch) — Feel the difference `.read()` without chunking makes.**
Write a deliberately "bad" upload endpoint that does `contents = await file.read()` (no chunk size, reading everything at once) instead of this chapter's chunked loop. Upload a large file (as large as you can conveniently generate — a few hundred MB, if your disk and patience allow) to both this endpoint and the chunked one from Step 1, measuring peak memory for each the same way as Exercise 14.2. At what file size does the difference become impossible to ignore?

---

## Solutions & Discussion

<details>
<summary>Exercise 14.1</summary>

```python
FILE_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
}

async def verify_magic_bytes(file: UploadFile, expected_content_type: str) -> bool:
    header = await file.read(8)
    await file.seek(0)   # reset position — the chunked read loop still needs to start from byte 0
    for signature, content_type in FILE_SIGNATURES.items():
        if header.startswith(signature) and content_type == expected_content_type:
            return True
    return False
```

```python
if file.content_type not in ALLOWED_CONTENT_TYPES:
    raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

if not await verify_magic_bytes(file, file.content_type):
    raise HTTPException(status_code=415, detail="File contents don't match the declared file type")
```

`webp` is deliberately omitted from `FILE_SIGNATURES` above for brevity — its signature check is slightly more involved (`RIFF....WEBP`, checking two non-contiguous byte ranges) — worth noting the table isn't exhaustive, just illustrative of the technique. The core point: `file.content_type` alone is trivially spoofable (it's just a string the client's form data declared), while the file signature check reads actual bytes belonging to the file itself — still not a perfect guarantee against a sufficiently determined attacker, but a meaningfully stronger check than trusting client-declared metadata alone, at very low cost (8 bytes, one read).
</details>

<details>
<summary>Exercise 14.2</summary>

```python
import tracemalloc

tracemalloc.start()
# ...execute a request against /export.csv, consuming the streamed response fully...
current, peak = tracemalloc.get_traced_memory()
print(f"peak memory: {peak / 1024 / 1024:.2f} MB")
tracemalloc.stop()
```

For a 500,000-row export with `batch_size=100`, expect peak memory in the low single-digit megabytes — dominated by one batch of 100 `ProductTable` objects and one `io.StringIO()` buffer's worth of CSV text at any given moment, not by the size of the full dataset. The exact number will vary by machine and row width, but the key observation is that it does **not** scale meaningfully with total row count — running the same export against 50,000 vs 500,000 rows should show roughly the *same* peak memory, only a longer total duration.
</details>

<details>
<summary>Exercise 14.3</summary>

```python
@router.get("/export-buffered.csv")
async def export_products_csv_buffered(repo: ProductRepoDep):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "name", "price", "currency", "in_stock"])

    offset = 0
    while True:
        products = await repo.list(limit=100, offset=offset)
        if not products:
            break
        for p in products:
            writer.writerow([p.id, p.name, p.price, p.currency, p.in_stock])
        offset += 100

    return Response(content=buffer.getvalue(), media_type="text/csv")
```

For the same 500,000-row dataset, expect peak memory here to be dramatically higher — proportional to the total size of the generated CSV text, since the entire thing accumulates in one `io.StringIO()` buffer before `.getvalue()` is ever called, and that complete string then has to exist in memory again as the `Response` body. Where the streaming version's peak memory stayed roughly flat regardless of row count, this version's peak memory scales up directly with it — a `500,000`-row export might peak at tens or hundreds of MB here, against low single-digit MB for the streaming version handling the identical dataset.
</details>

<details>
<summary>Exercise 14.4</summary>

With `/export.csv` moved below `/{product_id}`: `GET /products/export.csv` matches `/{product_id}` first (as a string pattern — Chapter 3.3's rule, unchanged by six intervening chapters), FastAPI attempts to coerce `"export.csv"` to `int`, fails, and returns `422` — `export_products_csv` never runs. Moving the route back above `/{product_id}` restores the fixed-string-first ordering, and the export works again. This is the same bug, the same fix, and the same underlying rule as Chapter 3 — worth explicitly noticing that it wasn't a one-time lesson specific to the Notes API, but a permanent property of how Starlette matches routes, applicable to any fixed path you add for the rest of this curriculum.
</details>

<details>
<summary>Exercise 14.5</summary>

```python
@router.post("/{product_id}/image-unsafe")
async def upload_product_image_unsafe(product_id: int, file: UploadFile = File(...)):
    contents = await file.read()   # entire file, all at once
    destination = UPLOAD_DIR / f"{product_id}_unsafe{Path(file.filename).suffix}"
    destination.write_bytes(contents)
    return {"product_id": product_id}
```

For small test files (a few KB), the difference is invisible — both versions use negligible, roughly identical memory. The difference becomes impossible to ignore once the uploaded file's size becomes a meaningful fraction of available memory relative to how many concurrent uploads your server might handle: a single 500 MB upload via the unsafe version requires roughly 500 MB of RAM just to hold `contents`, in addition to whatever memory the rest of your application needs at that moment — and if two such uploads happen concurrently (a real possibility under genuine load, tying back to Chapter 2's concurrency model), that's roughly 1 GB, simultaneously, for uploads alone. The chunked version's peak memory stays bounded to roughly `CHUNK_SIZE` (1 MB in this chapter's example) regardless of the uploaded file's total size or how many uploads are happening concurrently — the exact crossover point where this matters depends on your server's available memory and expected concurrent upload volume, but the qualitative lesson holds well before you hit an actual out-of-memory crash: peak memory for the unsafe version is a direct function of upload size; peak memory for the chunked version is not.
</details>

---

## Chapter Summary

- `UploadFile` is generally the right default over `bytes` for file uploads, but its memory advantage only materializes if you actually read it in chunks — `await file.read()` with no size argument loads everything at once regardless of which type you chose.
- `StreamingResponse` applies the same "process incrementally, don't materialize everything at once" principle to responses — useful for large generated output like CSV exports, with the added benefit that the client starts receiving data before generation finishes.
- `app.mount(...)` (for `StaticFiles`) attaches a separate ASGI application at a path prefix, distinct from `app.include_router(...)`, which adds routes to your own FastAPI app's routing table.
- A client-supplied `content_type` on an upload is exactly as trustworthy as a client-supplied `Content-Length` header from Chapter 12 — not very; checking the file's actual signature bytes is a meaningfully stronger (if still imperfect) check.
- Chapter 3's route-ordering rule (fixed paths before parameterized ones) doesn't expire — it applies to every new fixed-string route added anywhere in a growing API, including this chapter's `/export.csv`.

**Next:** Chapter 15 covers testing FastAPI applications properly — `TestClient`, dependency overrides, and a real pytest suite covering the auth system, database layer, and error handling built across every chapter so far. This closes out Part II.
