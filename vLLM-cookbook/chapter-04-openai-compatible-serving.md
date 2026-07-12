# Chapter 4 — Serving with the OpenAI-Compatible API

> **Track:** Intermediate  
> **Prerequisite:** Chapters 1–3  
> **vLLM version:** 0.19.0 (April 2026)

---

## Table of Contents

1. [Offline vs online inference — the distinction](#1-offline-vs-online-inference--the-distinction)
2. [Starting the server: `vllm serve`](#2-starting-the-server-vllm-serve)
3. [Essential CLI flags](#3-essential-cli-flags)
4. [Available endpoints](#4-available-endpoints)
5. [Querying with `curl`](#5-querying-with-curl)
6. [Querying with the OpenAI Python SDK](#6-querying-with-the-openai-python-sdk)
7. [Streaming responses](#7-streaming-responses)
8. [vLLM-specific sampling parameters via `extra_body`](#8-vllm-specific-sampling-parameters-via-extra_body)
9. [Structured outputs (JSON mode)](#9-structured-outputs-json-mode)
10. [Tool calling / function calling](#10-tool-calling--function-calling)
11. [How a request flows through the server](#11-how-a-request-flows-through-the-server)
12. [LangChain and LlamaIndex integration](#12-langchain-and-llamaindex-integration)
13. [API key authentication](#13-api-key-authentication)
14. [Common server errors and fixes](#14-common-server-errors-and-fixes)
15. [Practice exercises](#15-practice-exercises)
16. [Chapter summary](#16-chapter-summary)

---

## 1. Offline vs online inference — the distinction

You already know the `LLM` class from Chapter 2. That is **offline inference**: a blocking Python function call, one batch at a time, inside your script.

**Online inference** (this chapter) is different:

| | Offline (`LLM` class) | Online (`vllm serve`) |
|---|---|---|
| Interface | Python function | HTTP REST API |
| Concurrency | Single thread, one batch | Async, many clients simultaneously |
| Use case | Scripts, batch jobs, experiments | Production APIs, chatbots, demos |
| Client | Your Python code only | Any language, any tool |
| Compatibility | vLLM-only | OpenAI-compatible — any existing OpenAI SDK client works |

The online server wraps the same `AsyncLLMEngine` (which uses the same PagedAttention machinery) behind an HTTP server built on **FastAPI** and **Uvicorn**. Every request that hits the server becomes a `Request` object queued into the engine's scheduler.

---

## 2. Starting the server: `vllm serve`

```bash
# Minimal — starts on localhost:8000
vllm serve facebook/opt-125m

# CPU laptop (recommended flags)
vllm serve facebook/opt-125m \
  --device cpu \
  --dtype bfloat16 \
  --max-model-len 512 \
  --port 8000

# GPU (e.g. A100, Qwen 7B)
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --port 8000
```

When the server is ready, you'll see:

```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> **Note:** `vllm serve` is equivalent to `python -m vllm.entrypoints.openai.api_server --model <model>`. Both do the same thing.

---

## 3. Essential CLI flags

These are the flags you'll use in almost every deployment:

```bash
vllm serve <model> \
  # ── Model ──────────────────────────────────────────────────
  --dtype bfloat16                 # Weight precision (bfloat16 for GPU/CPU)
  --max-model-len 4096             # Max context window (prompt + output)
  --tokenizer <path>               # Override tokenizer (defaults to model)
  --trust-remote-code              # Allow custom model code from HuggingFace

  # ── Memory ─────────────────────────────────────────────────
  --gpu-memory-utilization 0.90    # GPU only: fraction of VRAM for vLLM
  --device cpu                     # CPU only: force CPU backend

  # ── Server ─────────────────────────────────────────────────
  --host 0.0.0.0                   # Bind address (0.0.0.0 = all interfaces)
  --port 8000                      # Port number
  --api-key my-secret-key          # Require Bearer token authentication

  # ── Model name in API ───────────────────────────────────────
  --served-model-name my-model     # Alias shown in /v1/models and used in requests

  # ── Performance ─────────────────────────────────────────────
  --enable-prefix-caching          # Enable APC (Chapter 3)
  --max-num-seqs 256               # Max concurrent sequences in flight

  # ── Generation config ───────────────────────────────────────
  --generation-config vllm         # Don't override defaults from model's config file
```

**`--served-model-name` is important in production.** Without it, clients must pass the full HuggingFace model ID (e.g. `Qwen/Qwen2.5-7B-Instruct`) in every request. With it, clients just send `my-model`.

---

## 4. Available endpoints

vLLM's server implements the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List loaded models |
| `/v1/chat/completions` | POST | Chat (multi-turn, with roles) |
| `/v1/completions` | POST | Raw text completion (legacy) |
| `/v1/embeddings` | POST | Text embeddings (embedding models only) |
| `/health` | GET | Server health check |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Interactive API docs (FastAPI Swagger UI) |

The two you will use most are `/v1/chat/completions` (instruction/chat models) and `/v1/completions` (base/completion models).

**Key difference:**
- `/v1/completions` — sends raw text, gets raw text back. No chat template applied.
- `/v1/chat/completions` — sends a list of role/content messages. vLLM applies the model's chat template before feeding to the model.

---

## 5. Querying with `curl`

### Check the server is up

```bash
curl http://localhost:8000/health
# → 200 OK
```

### List models

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

### Chat completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 50
  }'
```

**Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1712345678,
  "model": "facebook/opt-125m",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 9,
    "total_tokens": 23
  }
}
```

### Raw text completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "temperature": 0
  }'
```

---

## 6. Querying with the OpenAI Python SDK

This is the key integration point. Because vLLM implements the OpenAI API specification, you point the OpenAI SDK at your local server and everything works — no other changes needed.

```bash
pip install openai
```

### Chat completions

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",           # Required by SDK — value doesn't matter without --api-key
)

response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": "What is the capital of France?"},
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Finish reason: {response.choices[0].finish_reason}")
```

### Multi-turn conversation

The server itself is **stateless** — it doesn't remember previous turns. You manage conversation history client-side and send the full history on each request.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Build conversation history manually
history = [{"role": "system", "content": "You are a helpful assistant."}]

def chat(user_message: str) -> str:
    history.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=history,
        temperature=0.7,
        max_tokens=150,
    )
    
    assistant_msg = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg

print(chat("What is photosynthesis?"))
print(chat("Give me a one-line summary of that."))
# Both requests include the full history — vLLM recomputes or prefix-caches it
```

### Text completions (base models)

```python
response = client.completions.create(
    model="facebook/opt-125m",
    prompt="Once upon a time in a distant galaxy,",
    max_tokens=80,
    temperature=0.9,
    stop=["\n\n"],
)
print(response.choices[0].text)
```

---

## 7. Streaming responses

For interactive applications (chatbots, terminals), streaming returns tokens as they are generated rather than waiting for the full response. This dramatically reduces **perceived latency** — users see the first token immediately.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# stream=True returns an iterator of chunks
stream = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{"role": "user", "content": "Tell me about the solar system."}],
    temperature=0.7,
    max_tokens=200,
    stream=True,            # ← enable streaming
)

# Print tokens as they arrive
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()  # newline at end
```

### Streaming with `curl` (SSE — Server-Sent Events)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [{"role": "user", "content": "Count to 5."}],
    "max_tokens": 30,
    "stream": true
  }'
```

You'll see a stream of `data: {...}` lines, each containing one token delta. The stream ends with `data: [DONE]`.

---

## 8. vLLM-specific sampling parameters via `extra_body`

The OpenAI SDK only exposes the parameters OpenAI supports. vLLM supports additional parameters (like `top_k`, `min_p`, `repetition_penalty`). Pass these through `extra_body`:

```python
response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{"role": "user", "content": "Write a haiku about the sea."}],
    temperature=0.9,
    max_tokens=60,
    extra_body={
        "top_k": 50,                    # Only sample from top-50 tokens
        "min_p": 0.05,                  # Min probability threshold
        "repetition_penalty": 1.1,      # Penalise repeated tokens
        "skip_special_tokens": True,    # Strip <eos>, <pad> etc from output
        "stop_token_ids": [2],          # Stop on token ID 2 (common EOS)
    }
)
```

### Full list of vLLM-specific fields available in `extra_body`

| Field | Type | Description |
|-------|------|-------------|
| `top_k` | int | Sample from top-k tokens only |
| `min_p` | float | Min probability relative to top token |
| `repetition_penalty` | float | >1.0 penalises repeating tokens |
| `length_penalty` | float | For beam search: penalise long outputs |
| `use_beam_search` | bool | Use beam search instead of sampling |
| `stop_token_ids` | list[int] | Token IDs that halt generation |
| `skip_special_tokens` | bool | Strip `<eos>` etc from output (default True) |
| `min_tokens` | int | Force at least N tokens before stopping |
| `ignore_eos` | bool | Don't stop at EOS token |
| `prompt_logprobs` | int | Return logprobs for prompt tokens too |

---

## 9. Structured outputs (JSON mode)

Structured outputs constrain the model to generate valid output matching a schema. vLLM enforces this at the **token level** — invalid tokens are masked out at each step, so the output is guaranteed to be valid.

### Option A — JSON object mode (any valid JSON)

```python
response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{
        "role": "user",
        "content": "Extract: name and age from 'Alice is 30 years old'. Return JSON."
    }],
    max_tokens=50,
    response_format={"type": "json_object"},  # Guarantees valid JSON output
)
import json
data = json.loads(response.choices[0].message.content)
print(data)  # {"name": "Alice", "age": 30}
```

### Option B — JSON schema mode (exact schema enforcement)

Use a Pydantic model to define the exact fields you want:

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

class Person(BaseModel):
    name: str
    age: int
    city: str

response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{
        "role": "user",
        "content": "Generate a JSON with name, age, and city for a fictional person."
    }],
    max_tokens=80,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": Person.model_json_schema(),  # Pydantic → JSON Schema
        }
    },
)

import json
person = Person(**json.loads(response.choices[0].message.content))
print(f"{person.name}, age {person.age}, from {person.city}")
```

### Option C — Choice constraint (classification)

```python
response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{
        "role": "user",
        "content": "Classify the sentiment: 'vLLM is incredibly fast!'"
    }],
    max_tokens=10,
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative", "neutral"]}
    },
)
print(response.choices[0].message.content)  # → "positive"
```

### Option D — Regex constraint

```python
response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{"role": "user", "content": "Give me a UK postcode."}],
    max_tokens=10,
    extra_body={
        "guided_regex": r"[A-Z]{1,2}[0-9][0-9A-Z]?\s?[0-9][A-Z]{2}"
    },
)
print(response.choices[0].message.content)  # → e.g. "SW1A 2AA"
```

---

## 10. Tool calling / function calling

Tool calling lets the model request execution of external functions during a conversation. vLLM supports this by using guided decoding to force the model to emit valid tool call JSON.

### Server setup (requires a model that supports tool calling)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype bfloat16
```

Available parsers (choose based on your model family):
`hermes`, `mistral`, `llama3_json`, `qwen3_coder`, `internlm`, and more.

### Client usage

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Define the tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        }
    }
]

# First call — model decides to use the tool
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message

if msg.tool_calls:
    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"Model wants to call: {tool_call.function.name}({args})")

    # Execute the function (your code here)
    weather_result = {"city": args["city"], "temp": "18°C", "condition": "sunny"}

    # Second call — feed result back to model
    messages = [
        {"role": "user",      "content": "What's the weather in Paris?"},
        msg,                                          # assistant's tool call
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(weather_result),
        }
    ]
    final = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages,
    )
    print(final.choices[0].message.content)
    # → "The weather in Paris is currently 18°C and sunny."
```

---

## 11. How a request flows through the server

Understanding this flow explains latency, concurrency, and error behaviour.

```
Client (curl / Python SDK / any HTTP client)
  │
  │  POST /v1/chat/completions  (JSON body)
  ▼
FastAPI + Uvicorn  (async HTTP layer)
  │
  ├─ Validate request schema
  ├─ Apply chat template (Jinja2) → converts messages to a single token sequence
  ├─ Tokenise the prompt
  ├─ Create a Request object (request_id, prompt_token_ids, sampling_params)
  │
  ▼
AsyncLLMEngine  (request queue)
  │
  ├─ Add request to WAITING queue
  │
  ▼
Scheduler  (runs every decode step)
  │
  ├─ Selects batch of WAITING + RUNNING requests
  ├─ Allocates KV blocks from free pool (PagedAttention)
  ├─ Runs prefill for new requests (parallel, all tokens at once)
  ├─ Runs one decode step for all RUNNING requests (one token each)
  ├─ Checks stop conditions (EOS, stop strings, max_tokens)
  ├─ Frees KV blocks for completed requests
  │
  ▼
AsyncLLMEngine  (output queue)
  │
  ├─ Detokenise generated token IDs → text
  ├─ Build OpenAI-compatible response JSON
  │
  ▼
FastAPI  (response or SSE stream)
  │
  ▼
Client
```

**Key implication:** Your request sits in the WAITING queue until the scheduler picks it up. At high load, TTFT (Time To First Token) grows because requests wait longer in the queue — not because the model is slower.

---

## 12. LangChain and LlamaIndex integration

Because vLLM exposes the OpenAI API, every framework that supports OpenAI works with zero additional code.

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="facebook/opt-125m",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="none",
    temperature=0.7,
    max_tokens=200,
)

response = llm.invoke("Explain what a transformer model is.")
print(response.content)
```

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI as LlamaOpenAI

llm = LlamaOpenAI(
    model="facebook/opt-125m",
    api_base="http://localhost:8000/v1",
    api_key="none",
)

response = llm.complete("The capital of France is")
print(response.text)
```

---

## 13. API key authentication

By default, vLLM's server accepts all requests with any API key (it ignores the key value). To enforce a real key:

```bash
vllm serve facebook/opt-125m --api-key my-secret-token-123
```

Requests without the correct key get a `401 Unauthorized`. With the key:

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-secret-token-123",   # Must match --api-key value
)
```

With `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-token-123" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## 14. Common server errors and fixes

### `404 Not Found` on `/v1/chat/completions`

Your model is a **base model** without a chat template. Use `/v1/completions` instead, or use an instruct/chat model.

```bash
# Base model — use completions
vllm serve facebook/opt-125m
curl .../v1/completions -d '{"prompt": "...", ...}'

# Chat/instruct model — use chat/completions
vllm serve Qwen/Qwen2.5-0.5B-Instruct
curl .../v1/chat/completions -d '{"messages": [...], ...}'
```

### `Model not found` / `404` on model name

The `model` field in your request must exactly match either the HuggingFace model ID or the value you set with `--served-model-name`.

```bash
# Started with:
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name my-model

# Request must use:
{"model": "my-model", ...}       # ✅ works
{"model": "Qwen/Qwen2.5-0.5B-Instruct", ...}  # ❌ 404
```

### `422 Unprocessable Entity`

A required field is missing or has wrong type. Check your JSON body — common culprits are `messages` (must be a list, not a string) or `max_tokens` (must be an integer, not a string).

### `Connection refused` on port 8000

The server hasn't finished starting yet, or it crashed. Check the terminal where you ran `vllm serve` for error messages. OOM errors during startup are the most common cause.

### Response contains `<|im_end|>` or other special tokens

Set `skip_special_tokens: true` in `extra_body`, or the model's tokenizer config doesn't have it set by default:

```python
extra_body={"skip_special_tokens": True}
```

---

## 15. Practice exercises

### Exercise 1 — Start the server and verify all endpoints

```bash
# Terminal 1: start server
export VLLM_CPU_KVCACHE_SPACE=4
vllm serve facebook/opt-125m \
  --device cpu --dtype bfloat16 --max-model-len 256 --port 8000

# Terminal 2: test all endpoints
curl -s http://localhost:8000/health && echo " ✓ health OK"
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "facebook/opt-125m", "prompt": "Hello world", "max_tokens": 5}'
```

### Exercise 2 — SDK client with proper error handling

```python
from openai import OpenAI, APIConnectionError, AuthenticationError

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def safe_chat(user_msg: str, model: str = "facebook/opt-125m") -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.7,
            max_tokens=100,
        )
        return resp.choices[0].message.content
    except APIConnectionError:
        return "ERROR: Could not connect to vLLM server. Is it running?"
    except Exception as e:
        return f"ERROR: {e}"

print(safe_chat("What is 2 + 2?"))
print(safe_chat("Name three planets.", model="nonexistent-model"))
```

### Exercise 3 — Streaming chatbot in terminal

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
history = []

print("Chat with vLLM (type 'quit' to exit)\n")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break
    
    history.append({"role": "user", "content": user_input})
    
    print("Bot: ", end="", flush=True)
    full_response = ""
    stream = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=history,
        max_tokens=100,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        print(token, end="", flush=True)
        full_response += token
    print()
    
    history.append({"role": "assistant", "content": full_response})
```

### Exercise 4 — Structured JSON output

```python
from openai import OpenAI
from pydantic import BaseModel
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

class Country(BaseModel):
    name: str
    capital: str
    population_millions: float
    continent: str

response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{
        "role": "user",
        "content": "Return JSON data about France."
    }],
    max_tokens=100,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "country",
            "schema": Country.model_json_schema(),
        }
    },
)

country = Country(**json.loads(response.choices[0].message.content))
print(f"{country.name}: capital={country.capital}, pop={country.population_millions}M")
```

### Exercise 5 — Measure TTFT vs total latency

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def measure_latency(prompt: str, max_tokens: int = 100):
    t0 = time.time()
    first_token_time = None
    
    stream = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    
    tokens = []
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        t = chunk.choices[0].delta.content or ""
        tokens.append(t)
    
    total_time = time.time() - t0
    ttft = first_token_time - t0
    
    print(f"TTFT:          {ttft*1000:.0f}ms")
    print(f"Total latency: {total_time*1000:.0f}ms")
    print(f"Tokens:        {len(tokens)}")
    print(f"Throughput:    {len(tokens)/total_time:.1f} tok/s")

measure_latency("Explain the water cycle in detail.")
```

---

## 16. Chapter summary

| Concept | Key point |
|---------|-----------|
| `vllm serve` | Starts the HTTP server; wraps `AsyncLLMEngine` behind FastAPI |
| `--served-model-name` | Alias clients use in the `model` field; avoids typing the full HF path |
| `/v1/chat/completions` | For chat/instruct models; applies chat template automatically |
| `/v1/completions` | For base models; raw text in, raw text out |
| OpenAI SDK | Just change `base_url` to `http://localhost:8000/v1` — everything else works |
| Server is stateless | Conversation history managed client-side; send full history each request |
| Streaming | `stream=True` returns SSE chunks; dramatically reduces perceived latency |
| `extra_body` | Pass vLLM-specific params (`top_k`, `min_p`, `repetition_penalty`) not in OpenAI spec |
| Structured outputs | `response_format` for JSON; `extra_body.structured_outputs.choice` for classification |
| Tool calling | Requires `--enable-auto-tool-choice --tool-call-parser <parser>` at server start |
| `/health` | Liveness check for monitoring; `/metrics` for Prometheus |
| `--api-key` | Enforce Bearer token auth on the server |
| Request flow | HTTP → FastAPI → tokenise → queue → scheduler → PagedAttention engine → response |

---

## What's next

Chapter 5 — **Continuous Batching & Scheduling**: now that you know how requests arrive, you'll see exactly how the scheduler decides which requests run each step, why one slow request doesn't block fast ones, and how to tune throughput vs latency with `max-num-seqs` and chunked prefill.

---

*vLLM docs: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/ | vLLM version: 0.19.0*
