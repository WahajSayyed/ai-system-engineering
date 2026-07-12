# Chapter 16: Advanced Pydantic — Generics, Settings, and Custom Types

> Part III — Advanced: Production Engineering · Chapter 16 of 28

Part II ended with a real test suite — the last thing needed before treating this codebase as something headed toward production rather than a learning exercise. Part III starts by finishing two things Part II deliberately left rough: Chapter 11's hardcoded `SECRET_KEY` (flagged then as "move to env config in Chapter 16" — this is that moment) and the repeated, slightly-different pagination shape every list endpoint has hand-rolled since Chapter 4. Generics, `pydantic-settings`, custom `Annotated` types, and discriminated unions close out the advanced Pydantic material this curriculum has been building toward since Chapter 5.

## Learning Objectives

By the end of this chapter you will be able to:

- Build a generic Pydantic model (`Paginated[T]`) reusable across every resource in your API, instead of a bespoke pagination shape per endpoint.
- Load configuration from environment variables and `.env` files using `pydantic-settings`, with validation and fail-fast behavior on missing required values.
- Build reusable custom types with `Annotated` plus `AfterValidator`, and know when `pydantic-extra-types` already has one you need.
- Model a polymorphic payload correctly using a discriminated union, and explain why it's both faster and clearer than a plain `Union`.

---

## 16.1 Generic Models

Every list endpoint since Chapter 4 has returned some ad hoc shape — `{"items": [...], "total": ..., "limit": ..., "offset": ...}` — redeclared slightly differently each time. A **generic model**, parametrized by a type variable, lets you define this shape exactly once and reuse it for any resource:

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class Paginated(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
```

`Paginated` by itself isn't a usable model — `T` is a placeholder. `Paginated[ProductPublic]` is the *concrete* type: a `Paginated` model whose `items` field is specifically `list[ProductPublic]`, with full validation and OpenAPI schema generation exactly as if you'd hand-written a dedicated `PaginatedProducts` model. FastAPI understands this directly as a `response_model`:

```python
@router.get("/", response_model=Paginated[ProductPublic])
async def list_products(repo: ProductRepoDep, limit: int = 20, offset: int = 0):
    ...
```

One model definition, reused for `Paginated[ProductPublic]`, `Paginated[UserPublic]`, `Paginated[ReviewPublic]` — anywhere a paginated list shows up — with the same guarantee every other Pydantic model gives you: if `items` doesn't actually validate against the parametrized type, you get a real validation error, not a silently-wrong response shape.

## 16.2 `pydantic-settings`: Configuration That Validates and Fails Fast

Chapter 8's `Settings` class was two hardcoded attributes; Chapter 11's `SECRET_KEY` was a literal string sitting in `security.py`. Neither belongs in source code for anything beyond a chapter-sized example — real configuration needs to come from the environment (so the same code runs correctly across development, staging, and production with different values) and it needs to fail loudly and immediately if something required is missing, rather than limping along with `None` and failing confusingly three layers deeper.

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    secret_key: str                       # no default → required
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    database_url: str = "sqlite+aiosqlite:///./app.db"
    default_currency: str = "USD"
    max_page_size: int = 100

settings = Settings()
```

```bash
# .env
SECRET_KEY=09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7
DATABASE_URL=sqlite+aiosqlite:///./app.db
```

`BaseSettings` behaves like an ordinary Pydantic model — the same validation, coercion, and required-vs-optional rules from Chapter 5 — except its values come from the environment (and, if present, an `.env` file) instead of a request body. Matching is case-insensitive by default: the `SECRET_KEY` environment variable populates the `secret_key` field automatically, no manual mapping needed. Precedence, highest to lowest: values passed directly to `Settings()`'s constructor, then real environment variables, then values from the `.env` file, then the field's own default (for fields that have one) — meaning an actual environment variable set in production always wins over whatever `.env` (typically a development-only file, not committed to version control) contains.

`secret_key: str` — no default — is deliberately **required**. If `SECRET_KEY` isn't set anywhere `Settings()` looks, constructing `Settings()` raises a `ValidationError` immediately, at import time, before your application finishes starting up at all. This is exactly the fail-fast behavior you want: a missing secret key should crash your application loudly on startup, not silently produce a broken JWT signing key that fails in some confusing way the first time someone tries to log in.

With `settings` now a real, validated configuration object, Chapter 11's hardcoded values get replaced directly:

```python
# security.py
from config import settings

# ...
token = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
```

```python
# database.py
from config import settings

engine = create_async_engine(settings.database_url, echo=True)
```

One design note worth being explicit about, since it's a genuine change from Chapter 8's pattern: Chapter 8's `get_settings()` was a *dependency*, returning a fresh `Settings()` instance per call, relying on per-request caching (section 8.4) to avoid redundant work within one request. Now that `Settings` reads real environment variables — which don't change during your application's lifetime — there's no benefit to reconstructing it per request at all. A single module-level `settings = Settings()`, read once at import time and imported wherever needed, is the more appropriate shape here; the dependency-injection version made sense when "settings" was a stand-in placeholder, less so now that it's genuine, static configuration.

## 16.3 Reusable Custom Types with `Annotated`

Chapter 5's `field_validator` attaches a validation rule to one field, on one model. The moment you need the *same* constraint — a valid phone number, say — on several different models, redeclaring that validator repeatedly is exactly the kind of duplication Pydantic's `Annotated`-based custom types exist to avoid:

```python
import re
from typing import Annotated
from pydantic import AfterValidator

def validate_phone(v: str) -> str:
    if not re.match(r"^\+?[1-9]\d{7,14}$", v):
        raise ValueError("must be a valid phone number, e.g. +14155552671")
    return v

PhoneNumber = Annotated[str, AfterValidator(validate_phone)]
```

```python
class Contact(BaseModel):
    name: str
    phone: PhoneNumber

class Business(BaseModel):
    name: str
    support_phone: PhoneNumber
```

`PhoneNumber` is now a genuine, reusable *type* — used exactly like `str` or `int` anywhere you need it, carrying its validation rule along with it, defined once. This is the direct generalization of Chapter 5's field-level validators: instead of attaching validation logic to a specific field on a specific model, you attach it to a type alias that any model can adopt.

For common cases, you may not need to write the validator yourself at all — **`pydantic-extra-types`** is a real, maintained companion package providing ready-made types for exactly this kind of need (a proper `PhoneNumber` type backed by the `phonenumbers` library, `Color`, `Currency`, and others), worth checking before hand-rolling a constrained type that a well-tested library already provides.

## 16.4 Discriminated Unions

Consider a notification payload that could be one of several genuinely different shapes:

```python
class EmailNotification(BaseModel):
    type: Literal["email"]
    to_address: str
    subject: str

class SMSNotification(BaseModel):
    type: Literal["sms"]
    to_phone: str
    body: str
```

A plain `Union[EmailNotification, SMSNotification]` *works*, but Pydantic validates it by trying each member in turn until one succeeds — slower than it needs to be, and prone to confusing error messages when validation fails (a malformed email notification might report errors from its attempted match against `SMSNotification` instead, if that happened to be tried and partially matched first, deep in unhelpful "here's why it doesn't match option 2" detail).

A **discriminated union** fixes both problems by naming a field — here, `type` — that unambiguously identifies which variant applies, *before* any real validation against that variant's other fields even begins:

```python
from typing import Literal, Union, Annotated
from pydantic import Field

Notification = Annotated[
    Union[EmailNotification, SMSNotification],
    Field(discriminator="type"),
]
```

Now Pydantic reads `type` first, and validates *only* against the one matching model — no trial-and-error across every variant, and a validation failure produces an error message scoped precisely to the variant that was actually being attempted, since there's no ambiguity about which one that was. This is both a real performance improvement (no wasted validation attempts against variants that were never going to match) and a real clarity improvement for anyone debugging a `422` — worth using any time a `Union` of models has a natural "tag" field to discriminate on, rather than defaulting to a plain `Union` out of habit.

---

## Hands-On Project: A Generic Pagination Wrapper and Real Settings

### Step 1 — `config.py`, replacing every hardcoded value from Chapters 8, 9, and 11

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    database_url: str = "sqlite+aiosqlite:///./app.db"
    default_currency: str = "USD"
    max_page_size: int = 100

settings = Settings()
```

Update `security.py` and `database.py` to import `settings` and read from it, exactly as shown in section 16.2 — removing every hardcoded literal those modules previously carried.

### Step 2 — `Paginated[T]`, applied to Products

```python
# schemas.py (addition)
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class Paginated(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
```

```python
# repositories/products.py (addition)
from sqlalchemy import func

class ProductRepository:
    ...
    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(ProductTable))
        return result.scalar_one()
```

```python
# routers/products.py
from schemas import Paginated, ProductPublic

@router.get("/", response_model=Paginated[ProductPublic])
async def list_products(repo: ProductRepoDep, limit: int = 20, offset: int = 0):
    products = await repo.list(limit=limit, offset=offset)
    total = await repo.count()
    return Paginated[ProductPublic](items=products, total=total, limit=limit, offset=offset)
```

Confirm `GET /products/` now returns `{"items": [...], "total": N, "limit": ..., "offset": ...}`, validated against a real, named, reusable model rather than an ad hoc dict — and check `/docs`, where `Paginated_ProductPublic_` (or similarly named) now appears as a proper schema component.

### Step 3 — Confirm fail-fast behavior directly

Temporarily rename or remove `SECRET_KEY` from your `.env` file and restart the application. Confirm it fails immediately, at startup, with a `ValidationError` naming `secret_key` as missing — rather than starting up "successfully" and only failing later, confusingly, the first time someone tries to log in.

---

## Practice Exercises

**Exercise 16.1 — A discriminated union for polymorphic notifications.**
Add `EmailNotification`, `SMSNotification`, and `PushNotification` models (each with a `Literal` `type` tag and its own distinct fields), combine them into a discriminated `Notification` union, and add a `POST /notifications` endpoint accepting one. Send three different valid payloads (one per type) and confirm each validates against the correct variant. Then send a payload with `"type": "sms"` but missing `body`, and confirm the resulting `422` error clearly references `SMSNotification`'s fields specifically, not a confusing blend of all three variants.

**Exercise 16.2 — A custom, reusable `PhoneNumber` type.**
Using section 16.3's pattern, add a `PhoneNumber` type and use it on two different models (e.g., a `Contact` model and a field on `UserTable`'s corresponding Pydantic schema). Confirm an invalid phone number is rejected identically on both models, and that the validation logic exists in exactly one place.

**Exercise 16.3 — Settings that fail fast, verified two ways.**
Add a new required setting, `SMTP_HOST: str` (no default), representing an email server this application will eventually need for Chapter 13's welcome email. Confirm the application fails to start with it unset. Then set it in `.env` and confirm the application starts normally, with `settings.smtp_host` correctly populated.

**Exercise 16.4 — Apply `Paginated[T]` to a second resource.**
Apply `Paginated[T]` to the Reviews or Users list endpoint (whichever your project has), reusing the *exact same* `Paginated` class from Step 2 with a different type parameter. Confirm both `Paginated[ProductPublic]` and `Paginated[ReviewPublic]` (or equivalent) appear as distinct, correctly-shaped schemas in `/docs`, generated from one shared model definition.

**Exercise 16.5 (stretch) — Swap in the real `pydantic-extra-types` `PhoneNumber`.**
Install `pydantic-extra-types` and `phonenumbers`, and replace your hand-rolled `PhoneNumber` type from Exercise 16.2 with the package's real `PhoneNumber` type. Compare behavior on a few edge cases (international formats, obviously invalid strings) between your regex-based version and the library's — where do they disagree, and what does that suggest about the risk of hand-rolling validation for something as genuinely complex as global phone number formats?

---

## Solutions & Discussion

<details>
<summary>Exercise 16.1</summary>

```python
from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field

class EmailNotification(BaseModel):
    type: Literal["email"]
    to_address: str
    subject: str

class SMSNotification(BaseModel):
    type: Literal["sms"]
    to_phone: str
    body: str

class PushNotification(BaseModel):
    type: Literal["push"]
    device_token: str
    title: str

Notification = Annotated[
    Union[EmailNotification, SMSNotification, PushNotification],
    Field(discriminator="type"),
]

@router.post("/notifications")
async def send_notification(notification: Notification):
    return {"received_type": notification.type}
```

Sending `{"type": "sms", "to_phone": "+14155552671"}` (missing `body`) produces a `422` whose `detail` references `SMSNotification`'s `body` field specifically as missing — Pydantic already knew, from the `type` discriminator, that `SMSNotification` was the *only* variant being attempted, so the error is scoped precisely, with no noise from `EmailNotification`'s or `PushNotification`'s unrelated fields appearing anywhere in the response.
</details>

<details>
<summary>Exercise 16.2</summary>

```python
class Contact(BaseModel):
    name: str
    phone: PhoneNumber

class UserPublicWithPhone(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    username: str
    phone: PhoneNumber
```

An invalid phone number (`"phone": "not-a-number"`) against either model produces the identical validation error message, because both fields ultimately point at the exact same `PhoneNumber` type alias and its one `validate_phone` function — there's no risk of the two models' phone validation silently drifting apart over time the way two independently-written `field_validator`s on two different models eventually could.
</details>

<details>
<summary>Exercise 16.3</summary>

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    secret_key: str
    smtp_host: str   # new required field, no default
    # ...existing fields...
```

With `SMTP_HOST` unset anywhere, `Settings()` raises a `ValidationError` at startup naming `smtp_host` as required and missing — the application never finishes starting, exactly like the `secret_key` case from the hands-on project. Adding `SMTP_HOST=smtp.example.com` to `.env` resolves it, and `settings.smtp_host` reads back correctly. This confirms the fail-fast behavior isn't specific to `secret_key` — it's a general property of any required (no-default) field on a `BaseSettings` subclass.
</details>

<details>
<summary>Exercise 16.4</summary>

```python
@router.get("/", response_model=Paginated[ReviewPublic])
async def list_reviews(repo: ReviewRepoDep, limit: int = 20, offset: int = 0):
    reviews = await repo.list(limit=limit, offset=offset)
    total = await repo.count()
    return Paginated[ReviewPublic](items=reviews, total=total, limit=limit, offset=offset)
```

`/docs` now shows both `Paginated[ProductPublic]` and `Paginated[ReviewPublic]` as distinct schema components (typically named something like `Paginated_ProductPublic_` and `Paginated_ReviewPublic_` in the raw OpenAPI schema), each correctly shaped for its respective `items` type — generated automatically from the single shared `Paginated` class definition in `schemas.py`, with zero duplicated pagination-wrapper code between the two endpoints.
</details>

<details>
<summary>Exercise 16.5</summary>

```python
from pydantic_extra_types.phone_numbers import PhoneNumber as RealPhoneNumber

class Contact(BaseModel):
    name: str
    phone: RealPhoneNumber
```

The library's version, backed by Google's `phonenumbers` library, correctly handles genuine international format variation (spacing, country-code conventions, valid-but-unusual national formats) that a simple regex like this chapter's hand-rolled version either rejects incorrectly (a valid number in an unfamiliar format) or, in the other direction, accepts incorrectly (a string that merely *looks* plausible to a loose regex but isn't a real, dialable number). This is a good concrete illustration of a more general principle: for genuinely complex, well-studied validation domains — phone numbers, email addresses, IBANs — a hand-rolled regex is a reasonable first pass for a chapter exercise, but a real, actively maintained library encodes edge cases (country-specific numbering plan quirks, for phone numbers specifically) that would take considerable, ongoing effort to replicate and keep correct by hand.
</details>

---

## Chapter Summary

- Generic models (`class Paginated(BaseModel, Generic[T])`) let you define a reusable shape once and parametrize it per-resource (`Paginated[ProductPublic]`), with full validation and OpenAPI schema generation, instead of hand-rolling a slightly different pagination shape per endpoint.
- `pydantic-settings`' `BaseSettings` loads configuration from environment variables and `.env` files, with the same validation Pydantic models always provide — a required field with no default fails the application fast, at startup, rather than producing a confusing failure deep inside a request later.
- `Annotated` plus `AfterValidator` builds genuinely reusable custom types, generalizing Chapter 5's per-model `field_validator` into something any model can adopt — and `pydantic-extra-types` may already provide the one you need.
- Discriminated unions (`Field(discriminator="...")`) validate a polymorphic payload against exactly the right variant, based on a tag field, producing both better performance and clearer error messages than a plain `Union` tried in sequence.

**Next:** Chapter 17 covers WebSockets and real-time communication — connection lifecycle, broadcasting to multiple clients, and how WebSockets compare to Server-Sent Events for the cases where a full duplex connection isn't actually necessary.
