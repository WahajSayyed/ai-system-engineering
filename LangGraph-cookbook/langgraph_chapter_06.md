# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 6 — Memory: Short-Term and Long-Term

> *"Short-term memory lets your application remember previous interactions within a single thread or conversation. Long-term memory stores user-specific or application-level data across sessions and is shared across conversational threads."*
> — LangGraph official documentation

---

### What This Chapter Covers

Chapter 5 established persistence — the checkpointer that saves state between supersteps. This chapter builds the **memory layer** on top of that foundation. Memory and persistence overlap in implementation, but they address different problems:

- **Persistence** = "don't lose data when things crash or pause"
- **Memory** = "know what to keep, compress, and retrieve intelligently"

By the end you will understand:

1. The two-layer memory architecture: short-term (thread-scoped) vs. long-term (cross-thread)
2. Why conversation history grows unboundedly and the three techniques to manage it: filtering, trimming, and summarization
3. The `trim_messages` function: every parameter, every strategy
4. The `BaseStore` interface: the four core operations, the `Item` object, namespaces
5. `InMemoryStore` and `PostgresStore`: when to use each, how to configure semantic search
6. The `index` configuration: embedding models, dims, field selection — what gets embedded
7. The three cognitive memory types (semantic, episodic, procedural) and how each maps to LangGraph patterns
8. Hot-path vs. background memory updates: the latency tradeoff and when each is appropriate
9. `LangMem`: the official LangGraph memory toolkit
10. The critical `thread_id` vs. `user_id` distinction
11. Privacy, deduplication, and TTL in production memory systems
12. MARRS checkpoint: adding researcher memory to the capstone project

---

### 6.1 The Two-Layer Memory Architecture

LangGraph formalizes a distinction that is natural to humans but often muddled in AI systems:

```
┌─────────────────────────────────────────────────────────────────┐
│ SHORT-TERM MEMORY (Thread-Scoped)                               │
│                                                                 │
│ • Backed by: Checkpointer (Chapter 5)                          │
│ • Scope: One conversation thread (identified by thread_id)     │
│ • Lifetime: Lives within a thread; lost when thread is deleted │
│ • Access: Any node that receives the graph state               │
│ • Contents: Message history, current task state, tool outputs  │
│ • Updated: Automatically, after every superstep               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ LONG-TERM MEMORY (Cross-Thread)                                 │
│                                                                 │
│ • Backed by: Store (BaseStore implementations)                 │
│ • Scope: Namespaced — can span all threads for a user          │
│ • Lifetime: Persists independently of any thread               │
│ • Access: Nodes that accept a store: BaseStore parameter       │
│ • Contents: User preferences, learned facts, agent instructions│
│ • Updated: Explicitly by nodes or tools that call store.put()  │
└─────────────────────────────────────────────────────────────────┘
```

**The critical architectural distinction** between these two layers is not just technical — it is conceptual:

- Short-term memory answers: "What happened in *this* conversation?"
- Long-term memory answers: "What do we know about *this user* from *all* conversations?"

With checkpointers alone, each new `thread_id` starts cold. A new thread for user Alice has no knowledge of what Alice told the agent last week in a different thread. The Store solves this: it holds user-scoped data that persists across all of Alice's threads indefinitely.

---

### 6.2 Short-Term Memory: Managing Conversation History

The most common form of short-term memory is the message list — the accumulated `AIMessage`, `HumanMessage`, and `ToolMessage` objects in `state["messages"]`. This is handled automatically by the `add_messages` reducer (Chapter 3) and the checkpointer (Chapter 5).

The problem is growth. Conversations accumulate messages indefinitely. The message list grows with every turn. Eventually:

- **Token costs**: Each LLM call includes the entire history in the prompt, costing proportionally more
- **Latency**: Processing time grows with token count
- **Context window limits**: Every LLM has a hard token limit. Exceeding it causes an irrecoverable API error
- **Performance degradation**: Most LLMs perform poorly over very long contexts even when within the limit — the "lost in the middle" phenomenon

LangGraph provides three complementary techniques to manage this. They are not mutually exclusive; production systems often combine all three.

#### 6.2.1 Technique 1: Filtering — Keep Only Recent Messages

The simplest approach: before sending the message list to the LLM, discard everything except the last N messages.

```python
from langchain_core.messages import BaseMessage

def filter_messages(messages: list[BaseMessage], k: int = 10) -> list[BaseMessage]:
    """Keep only the last k messages."""
    return messages[-k:]

def agent_node(state: AgentState) -> dict:
    # Only pass the last 10 messages to the LLM
    truncated = filter_messages(state["messages"], k=10)
    response = llm.invoke(truncated)
    return {"messages": [response]}
```

The full history is still stored in state (and checkpointed). You are only filtering what gets *passed to the LLM*, not what gets *stored*. This means earlier messages remain available for tools, debugging, or retrieval — they are just not included in the LLM's context for this particular call.

**When to use:** Simple cutoff is sufficient, message count (not tokens) is the relevant limit, and you do not need earlier context for recent replies.

**Limitation:** A blunt cutoff loses potentially important early context (e.g., a critical user requirement stated at turn 1 is invisible by turn 20).

#### 6.2.2 Technique 2: Trimming — Token-Aware Context Management

`trim_messages` from `langchain_core.messages` is a more sophisticated version of filtering. It is token-aware, respects message type ordering constraints, and supports multiple strategies.

```python
from langchain_core.messages import trim_messages, BaseMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def agent_node_with_trimming(state: AgentState) -> dict:
    trimmed = trim_messages(
        state["messages"],
        
        # Hard token limit (the context budget for history)
        max_tokens=4096,
        
        # "last": keep the most recent messages (trim from the front)
        # "first": keep the oldest messages (trim from the end) — rare
        strategy="last",
        
        # Use the actual model's tokenizer for accurate counting
        token_counter=llm,
        
        # Always include system messages regardless of token budget
        include_system=True,
        
        # Don't split a message if it straddles the token boundary;
        # instead, move the cut to the next clean message boundary
        allow_partial=False,
        
        # The trimmed list must start with a human message (required by most models)
        start_on="human",
    )
    
    response = llm.invoke(trimmed)
    return {"messages": [response]}
```

**Parameter deep-dive:**

| Parameter | Values | Effect |
|---|---|---|
| `max_tokens` | `int` | Maximum tokens in the returned message list |
| `strategy` | `"last"`, `"first"` | `"last"` keeps recent messages (most common). `"first"` keeps oldest. |
| `token_counter` | `Callable`, `BaseLanguageModel`, or `int` | How tokens are counted. An LLM uses the model's actual tokenizer. A plain `int` is used as a per-message constant (approximate). |
| `include_system` | `bool` | Whether `SystemMessage` objects are always preserved regardless of limit |
| `allow_partial` | `bool` | Whether a message can be truncated mid-content to fit within the limit |
| `start_on` | `"human"`, `"ai"`, message type | Ensures the trimmed list starts with the specified message type (preserves format requirements) |
| `end_on` | `"human"`, `"ai"`, message type | Ensures the trimmed list ends with the specified type |

**Important nuance — `start_on`:** Most LLM APIs require the conversation history to start with a human/user message. If you trim from the front and the oldest retained message happens to be an `AIMessage`, the API will reject the call. Setting `start_on="human"` ensures `trim_messages` adjusts the cutpoint if needed to satisfy this constraint.

```python
# Example of the constraint in action:
messages = [
    HumanMessage("Hello"),         # msg 1
    AIMessage("Hi there"),         # msg 2
    HumanMessage("What is AI?"),   # msg 3
    AIMessage("AI stands for..."), # msg 4 ← would be the trim boundary
    HumanMessage("Thanks"),        # msg 5
    AIMessage("You're welcome"),   # msg 6
]

# Without start_on: might cut to [AIMessage, HumanMessage, AIMessage]
# → API error: history must start with Human
# 
# With start_on="human": backs up to the next HumanMessage
# → [HumanMessage("What is AI?"), AIMessage, HumanMessage, AIMessage]
# → Valid for all major APIs
```

#### 6.2.3 Technique 3: Summarization — Compress Old Context

Rather than discarding old messages, summarize them into a compact representation and use that summary as context going forward. This preserves information from early conversation turns while keeping the token budget bounded.

```python
from typing import TypedDict, Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, RemoveMessage

class ConversationState(MessagesState):
    # Extra field for the rolling summary
    summary: str   # Default overwrite — always the latest summary

def summarize_if_needed(state: ConversationState) -> dict:
    """
    Node that checks if summarization is needed and performs it.
    Call this node periodically (e.g., every 10 messages) or
    when the message count exceeds a threshold.
    """
    messages = state["messages"]
    
    # Only summarize if history is getting long
    SUMMARIZE_AFTER = 12
    if len(messages) < SUMMARIZE_AFTER:
        return {}  # Nothing to do
    
    # Get the existing summary (may be empty on first pass)
    existing_summary = state.get("summary", "")
    
    # Decide which messages to summarize:
    # - Everything except the last 4 messages (keep those verbatim)
    messages_to_summarize = messages[:-4]
    keep_verbatim = messages[-4:]
    
    # Build the summarization prompt
    if existing_summary:
        summary_prompt = (
            f"This is a summary of the conversation so far:\n{existing_summary}\n\n"
            f"Now extend this summary to include the following new messages. "
            f"Be concise but preserve all important facts, decisions, and context:\n"
            + "\n".join(f"{m.__class__.__name__}: {m.content}" 
                       for m in messages_to_summarize)
        )
    else:
        summary_prompt = (
            f"Create a concise summary of the following conversation, "
            f"preserving all important facts, user preferences, and decisions:\n"
            + "\n".join(f"{m.__class__.__name__}: {m.content}"
                       for m in messages_to_summarize)
        )
    
    # Generate the summary
    new_summary = llm.invoke([SystemMessage(content=summary_prompt)]).content
    
    # Remove the summarized messages from the list using RemoveMessage
    # (add_messages reducer handles deletion when it sees RemoveMessage objects)
    deletions = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    return {
        "summary": new_summary,
        "messages": deletions,    # Removes old messages from state
    }

def call_model_with_summary(state: ConversationState) -> dict:
    """
    Use the summary as context when the full history isn't included.
    """
    summary = state.get("summary", "")
    
    messages = state["messages"]
    
    if summary:
        # Inject summary as the first message for context
        context_message = SystemMessage(
            content=f"Summary of earlier conversation:\n{summary}"
        )
        messages_for_llm = [context_message] + messages
    else:
        messages_for_llm = messages
    
    response = llm.invoke(messages_for_llm)
    return {"messages": [response]}
```

**The two-phase wiring:**

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(ConversationState)
builder.add_node("summarize", summarize_if_needed)
builder.add_node("agent", call_model_with_summary)

builder.add_edge(START, "summarize")
builder.add_edge("summarize", "agent")
builder.add_edge("agent", END)

graph = builder.compile(checkpointer=checkpointer)
```

**Tradeoffs of summarization:**

| Aspect | Pros | Cons |
|---|---|---|
| Context quality | Preserves semantic content across the full history | LLM summarization can drop important details |
| Token efficiency | Excellent — summary is much smaller than history | Adds one LLM call per summarization |
| Latency | Low (summarization only happens occasionally) | Adds latency on the turns where it runs |
| Complexity | Self-contained node | Requires tuning the threshold and keeping logic |

#### 6.2.4 Choosing Between the Three Techniques

```
Message count < threshold?     → Use as-is (no management needed)
         ↓ (threshold exceeded)
Token budget is the concern?   → Use trim_messages (precise)
         OR
Count budget is the concern?   → Use filter (simpler)
         OR
History has long-term value?   → Use summarization (preserves info)
         OR
All of the above?              → Summarize old + trim to fit remaining budget
```

In production MARRS, research agents accumulate `findings` and `messages` over many supersteps. A summarization node that compresses the `messages` list (but not `findings` — those are stored separately) is the right approach. The raw research findings are what matter; the intermediate reasoning messages are expendable.

---

### 6.3 Long-Term Memory: The `Store` Interface

Long-term memory in LangGraph is implemented through the `BaseStore` interface — a persistent key-value document store with optional semantic search. It is architecturally separate from the checkpointer:

- **Checkpointer**: Saves state *within* a thread. State is scoped to `thread_id`.
- **Store**: Saves memories *across* threads. Data is scoped to custom `namespace` tuples.

```python
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# The store is attached at compile time, same as the checkpointer
store = InMemoryStore()

graph = builder.compile(
    checkpointer=checkpointer,   # Short-term: per-thread state
    store=store,                  # Long-term: cross-thread memories
)
```

#### 6.3.1 The `Item` Object: What You Get Back

When you retrieve from the store, you get `Item` objects (or `SearchItem` objects from `search()`):

```python
from langgraph.store.base import Item

# Item fields:
item.namespace    # tuple: ('user-alice', 'preferences')
item.key          # str: 'food_pref_1'
item.value        # dict: {'fact': 'Prefers Python over R', 'category': 'tools'}
item.created_at   # str: ISO timestamp
item.updated_at   # str: ISO timestamp

# SearchItem (from store.search()) has one additional field:
search_item.score # float: semantic similarity score (0.0 to 1.0), or None if no query
```

#### 6.3.2 Namespaces: The Organizational Unit

Namespaces are tuples of strings that organize memories hierarchically — like folders in a file system. Every `put`, `get`, and `search` operation is scoped to a namespace.

```python
# Namespaces can represent any hierarchy meaningful to your application

# User-scoped: memories for a specific user
("user-alice", "preferences")
("user-alice", "facts")
("user-alice", "task_history")

# Application-level: shared across all users
("application", "system_instructions")
("application", "faq")

# Agent-scoped: memories for a specific agent role
("agent-researcher", "search_strategies")
("agent-writer", "style_guidelines")

# Multi-dimensional: user within a tenant
("tenant-acme", "user-bob", "preferences")
```

**The namespace is not automatically user-specific.** You are responsible for including `user_id` in the namespace if you want per-user isolation. Omitting it creates a global store shared by all users — fine for application-level facts, dangerous for user-specific data.

```python
def remember_user_fact(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    # Get user_id from the runtime config
    user_id = config["configurable"].get("user_id")
    if not user_id:
        return {}  # No user_id = no personalization
    
    # Namespace scoped to this user
    namespace = ("users", user_id, "facts")
    
    # ... extract and store a fact ...
    store.put(namespace, key, value)
    return {}
```

#### 6.3.3 The Four Core `BaseStore` Operations

```python
from langgraph.store.base import BaseStore

# 1. put: Write a memory
store.put(
    namespace=("users", "alice", "preferences"),
    key="food_pref",              # Unique key within the namespace
    value={"fact": "Prefers vegetarian food", "confidence": "high"},
    index=["fact"],               # Optional: specify which fields to embed (semantic search)
)

# 2. get: Read a specific memory by key
item = store.get(
    namespace=("users", "alice", "preferences"),
    key="food_pref"
)
if item:
    print(item.value)  # {"fact": "Prefers vegetarian food", "confidence": "high"}

# 3. search: Find memories by content
results = store.search(
    namespace=("users", "alice", "preferences"),
    query="what kind of food does Alice like?",  # Natural language query
    limit=3,                                      # Return at most 3 results
    filter={"confidence": "high"},               # Optional metadata filter
)
for item in results:
    print(f"[{item.score:.2f}] {item.value['fact']}")
# Output:
# [0.94] Prefers vegetarian food

# 4. delete: Remove a specific memory
store.delete(
    namespace=("users", "alice", "preferences"),
    key="food_pref"
)
```

**Search without a `query` (listing all items in a namespace):**
```python
# If you omit the query, search returns all items in the namespace (no ranking)
all_prefs = store.search(("users", "alice", "preferences"))
for item in all_prefs:
    print(item.key, item.value)
```

**Async variants:** All operations have async counterparts: `aput`, `aget`, `asearch`, `adelete`. Use these when your graph uses `ainvoke`/`astream` (Chapter 8).

---

### 6.4 Accessing the Store Inside Nodes

Nodes declare store access by accepting a special `store` parameter with type annotation `BaseStore`. LangGraph auto-injects the store at runtime — you do not pass it manually.

```python
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

def personalized_agent(
    state: AgentState,
    config: RunnableConfig,
    *,
    store: BaseStore,               # ← Injected automatically by LangGraph
) -> dict:
    """
    Agent that reads user preferences from the long-term store.
    
    The '*' makes 'store' keyword-only. This is the required pattern.
    """
    user_id = config["configurable"].get("user_id", "anonymous")
    namespace = ("users", user_id, "preferences")
    
    # Retrieve memories relevant to the current message
    query = state["messages"][-1].content if state["messages"] else ""
    memories = store.search(namespace, query=query, limit=5)
    
    # Format memories for the system prompt
    if memories:
        memory_context = "\n".join(
            f"- {item.value.get('fact', item.value)}" 
            for item in memories
        )
        system_content = f"Known facts about this user:\n{memory_context}"
    else:
        system_content = "No prior information about this user."
    
    from langchain_core.messages import SystemMessage
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

**The `*` keyword-only separator is mandatory.** Without it, LangGraph cannot distinguish the `store` parameter from a regular positional argument and will not inject it. This is specified in the official LangGraph docs and source.

---

### 6.5 Store Backends

#### 6.5.1 `InMemoryStore` — Development

```python
from langgraph.store.memory import InMemoryStore

# Basic (no semantic search)
store = InMemoryStore()

# With semantic search enabled
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "dims": 1536,                                           # Embedding dimensions
        "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding model
        "fields": ["fact", "content"],                          # Which dict keys to embed
    }
)
```

**Characteristics:**
- Zero dependencies beyond `langgraph`
- All data lives in RAM — lost on process restart
- The `index` parameter enables vector similarity search using cosine similarity on in-memory numpy arrays
- Performance degrades with very large stores (linear scan for semantic search without an index)
- Suitable for: development, unit tests, small demos

#### 6.5.2 `PostgresStore` — Production

```bash
pip install langgraph-checkpoint-postgres
```

```python
from langchain.embeddings import init_embeddings
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://user:password@localhost:5432/marrs_db"

# Synchronous (use with sync graph execution)
with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()   # Creates the 'store' table and enables pgvector
    graph = builder.compile(checkpointer=checkpointer, store=store)

# With semantic search (uses PostgreSQL pgvector extension)
store = PostgresStore(
    conn=pool,                    # Connection pool (same pool as checkpointer is fine)
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text", "summary"],   # Which fields to embed
    }
)
store.setup()   # Creates table + enables pgvector. Idempotent.
```

**Important:** `PostgresStore.setup()` enables the `pgvector` PostgreSQL extension. This requires your PostgreSQL instance to have `pgvector` installed. On managed cloud databases (AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL), pgvector is usually available as an optional extension. On self-hosted PostgreSQL, install it with:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Asynchronous variant:**
```python
from langgraph.store.postgres.aio import AsyncPostgresStore

async with AsyncPostgresStore.from_conn_string(DB_URI) as store:
    await store.asetup()
    graph = builder.compile(checkpointer=async_checkpointer, store=store)
```

#### 6.5.3 Sharing the Same Database for Checkpointer and Store

A common production pattern: use the same PostgreSQL database (and the same connection pool) for both the checkpointer and the store. They write to different tables (`checkpoints`/`checkpoint_blobs` vs. `store`) and do not interfere with each other:

```python
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain.embeddings import init_embeddings

DB_URI = "postgresql://user:password@postgres:5432/marrs_db"

async def create_production_graph():
    pool = AsyncConnectionPool(
        DB_URI,
        max_size=20,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=False,
    )
    await pool.open()
    
    # Both use the same pool — same DB, different tables
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()   # Creates checkpoints tables
    
    store = AsyncPostgresStore(
        pool,
        index={
            "dims": 1536,
            "embed": init_embeddings("openai:text-embedding-3-small"),
            "fields": ["text"],
        }
    )
    await store.asetup()         # Creates store table, enables pgvector
    
    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph, pool
```

---

### 6.6 Semantic Search: Finding Memories by Meaning

Without a `query` parameter, `store.search()` returns all items in the namespace (listing). With a `query`, it performs semantic search — returning items ranked by meaning similarity rather than exact key match.

#### 6.6.1 How Semantic Search Works

When you call `store.put(namespace, key, value, index=["field"])`:
1. LangGraph extracts the text from the specified fields of `value`
2. The embedding model encodes that text into a vector
3. The vector is stored alongside the item in the database

When you call `store.search(namespace, query="...")`:
1. The query is encoded into a vector by the same embedding model
2. Cosine similarity between the query vector and all stored item vectors is computed
3. Items are returned ranked by similarity, with a `score` attribute

```python
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text"],  # Embed the "text" field of each stored item
    }
)

# Store memories — the "text" field gets embedded
store.put(("alice", "facts"), "fact_1", {"text": "Alice is a machine learning engineer"})
store.put(("alice", "facts"), "fact_2", {"text": "Alice prefers Python over R for data analysis"})
store.put(("alice", "facts"), "fact_3", {"text": "Alice's company uses Kubernetes for deployment"})

# Semantic search — finds by meaning, not exact match
results = store.search(
    ("alice", "facts"),
    query="programming language preference",   # Doesn't mention Python explicitly
    limit=2
)

for r in results:
    print(f"[{r.score:.3f}] {r.value['text']}")

# Output (order may vary slightly):
# [0.912] Alice prefers Python over R for data analysis
# [0.843] Alice is a machine learning engineer
```

The query "programming language preference" does not contain the word "Python" — but semantic search finds the relevant memory anyway because their *meanings* are similar in the embedding space.

#### 6.6.2 The `index` Configuration in Detail

```python
InMemoryStore(
    index={
        # REQUIRED: Number of dimensions in the embedding vectors
        # OpenAI text-embedding-3-small: 1536
        # OpenAI text-embedding-3-large: 3072
        # OpenAI text-embedding-ada-002: 1536
        # HuggingFace all-MiniLM-L6-v2: 384
        "dims": 1536,
        
        # REQUIRED: The embedding function
        # Can be a LangChain Embeddings object or an init_embeddings string
        "embed": init_embeddings("openai:text-embedding-3-small"),
        
        # OPTIONAL: Which fields in the stored dict to embed
        # "$" means embed the entire document (all text-like fields)
        # ["field1", "field2"] embeds only the specified fields
        # If omitted, defaults to ["$"] (embed everything)
        "fields": ["text", "summary"],
    }
)
```

**Per-item field override:** You can override which fields to embed on individual `put` calls:

```python
# Default: embed the "text" field
store.put(ns, "mem1", {"text": "Alice prefers Python", "metadata": "from session 1"})

# Override: embed "metadata" instead for this specific item
store.put(ns, "mem2", {"text": "Bob prefers R", "metadata": "language preference"}, 
          index=["metadata"])

# Skip embedding entirely for this item
store.put(ns, "mem3", {"text": "Config value: 42"}, index=False)
```

---

### 6.7 The Three Cognitive Memory Types

The LangGraph official documentation frames long-term memory through the lens of cognitive psychology, categorizing it into three types that correspond to how humans remember. This is not academic abstraction — it directly informs namespace design and retrieval strategy.

#### 6.7.1 Semantic Memory: Facts About the World

> "Facts can be written to semantic memory." — LangGraph official docs

Semantic memory stores factual knowledge — what is true about a user, domain, or entity. It answers "what?" questions: What does Alice prefer? What are the company's policies? What tools does this team use?

```python
def store_semantic_memory(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """
    Extract and store factual knowledge about the user.
    Called after each conversation turn to update the user's fact profile.
    """
    user_id = config["configurable"].get("user_id")
    namespace = ("users", user_id, "facts")
    
    # Use an LLM to extract memorable facts from the conversation
    extraction_prompt = f"""
    Review this conversation turn and extract any important facts worth remembering
    about the user. Focus on: preferences, constraints, background, goals.
    Return JSON: [{{"fact": "...", "category": "..."}}]
    Only extract facts explicitly stated. Output [] if none.
    
    Human: {state['messages'][-2].content if len(state['messages']) >= 2 else ''}
    """
    
    try:
        raw = llm.invoke(extraction_prompt).content
        facts = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return {}
    
    # Store each fact as a separate item with a unique UUID key
    # (one fact per item, not multiple facts in one item — easier to search and delete)
    import uuid
    for fact_data in facts:
        store.put(
            namespace,
            key=str(uuid.uuid4()),
            value={"text": fact_data["fact"], "category": fact_data["category"]},
            index=["text"],
        )
    
    return {}
```

**Design principle for semantic memory:** Store one fact per item. Avoid packing multiple facts into a single `value` dict. Each item is the unit of retrieval — if you pack ten facts together, searching for one of them returns all ten (not just the relevant one), and deleting a stale fact requires rewriting the item.

#### 6.7.2 Episodic Memory: Records of Past Experiences

> "Experiences can be written to episodic memory." — LangGraph official docs

Episodic memory stores *how* things were done — past interactions, successful approaches, and examples of agent behavior that worked well. It answers "how did we handle this before?" questions. In LangGraph, episodic memory is typically implemented as few-shot examples: "here is a past interaction that is similar to the current one."

```python
def store_episodic_memory(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """
    After a successful task completion, store the interaction as a few-shot example.
    """
    user_id = config["configurable"].get("user_id", "global")
    thread_id = config["configurable"].get("thread_id")
    namespace = ("agents", "researcher", "successful_searches")
    
    # Only store if the last response was marked as high quality
    if state.get("quality_score", 0) < 0.85:
        return {}
    
    # Create a condensed example from this interaction
    episode = {
        "input": state.get("topic", ""),
        "approach": state.get("plan", ""),
        "outcome": state.get("draft", "")[:500],  # Truncate for storage
        "quality_score": state.get("quality_score", 0),
        "revision_count": state.get("revision_count", 0),
        "thread_id": thread_id,
        # text field for semantic search
        "text": f"Research topic: {state.get('topic', '')}. Approach: {state.get('plan', '')}",
    }
    
    import uuid
    store.put(namespace, str(uuid.uuid4()), episode, index=["text"])
    return {}

def retrieve_similar_episodes(topic: str, store: BaseStore, limit: int = 3) -> list[dict]:
    """
    Find past successful research interactions similar to the current topic.
    Used to initialize the planner with few-shot examples.
    """
    results = store.search(
        ("agents", "researcher", "successful_searches"),
        query=f"Research on: {topic}",
        limit=limit,
    )
    return [r.value for r in results if r.score and r.score > 0.75]
```

**Few-shot injection in the planner:**
```python
def planner_node_with_memory(state: MARRSState, *, store: BaseStore) -> dict:
    # Retrieve similar past episodes
    past_episodes = retrieve_similar_episodes(state["topic"], store)
    
    # Build few-shot context
    few_shot_context = ""
    if past_episodes:
        few_shot_context = "\n\nHere are examples of successful research approaches:\n"
        for ep in past_episodes:
            few_shot_context += (
                f"\nTopic: {ep.get('input', 'N/A')}\n"
                f"Approach: {ep.get('approach', 'N/A')[:200]}\n"
                f"Quality: {ep.get('quality_score', 0):.2f}\n"
            )
    
    plan_prompt = f"Create a research plan for: {state['topic']}{few_shot_context}"
    plan = planner_llm.invoke(plan_prompt)
    
    return {"plan": plan.summary, "search_queries": plan.search_queries}
```

#### 6.7.3 Procedural Memory: Rules and Instructions

> "Procedural memory is a combination of model weights, agent code, and agent's prompt." — LangGraph official docs

Procedural memory stores *how to behave* — the agent's operating instructions, learned style guides, and refined system prompts. Unlike semantic memory (facts about the world) and episodic memory (specific past experiences), procedural memory is about the agent's *operating procedure*.

In practice, procedural memory in LangGraph is often implemented as a system prompt that the agent can update based on feedback:

```python
# ── Store initial instructions ────────────────────────────────────────────────
DEFAULT_INSTRUCTIONS = """You are a research assistant for MARRS.
Your goal is to produce thorough, well-cited research reports.
Always search academic sources before web sources.
Quality threshold: 0.80 minimum before approving a draft."""

def initialize_agent_instructions(store: BaseStore):
    """Set up default instructions on first deployment."""
    namespace = ("agents", "researcher", "instructions")
    existing = store.get(namespace, "default")
    if not existing:
        store.put(namespace, "default", {"text": DEFAULT_INSTRUCTIONS})

# ── Update instructions based on feedback ─────────────────────────────────────
def update_instructions_node(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """
    Procedural memory update: refine agent instructions based on user feedback.
    This implements the 'reflection' pattern — the agent learns from critique.
    """
    user_feedback = state.get("human_feedback", "")
    
    if not user_feedback or user_feedback == "Approved":
        return {}  # No feedback to learn from
    
    namespace = ("agents", "researcher", "instructions")
    current_item = store.get(namespace, "default")
    current_instructions = current_item.value["text"] if current_item else DEFAULT_INSTRUCTIONS
    
    # Ask the LLM to refine the instructions based on feedback
    refinement_prompt = f"""
    Current agent instructions:
    {current_instructions}
    
    User feedback on the last research report:
    {user_feedback}
    
    Please refine the instructions to avoid this issue in future runs.
    Return ONLY the updated instructions text.
    """
    
    updated_instructions = llm.invoke(refinement_prompt).content
    store.put(namespace, "default", {"text": updated_instructions})
    
    return {}

# ── Use instructions in the agent ─────────────────────────────────────────────
def agent_with_procedural_memory(state: State, *, store: BaseStore) -> dict:
    namespace = ("agents", "researcher", "instructions")
    instructions_item = store.get(namespace, "default")
    instructions = instructions_item.value["text"] if instructions_item else DEFAULT_INSTRUCTIONS
    
    from langchain_core.messages import SystemMessage
    messages = [SystemMessage(content=instructions)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

---

### 6.8 When to Update Memory: Hot Path vs. Background

The LangGraph official documentation explicitly identifies this as a key architectural decision when designing memory systems:

> "When do you want to update memories? Memory can be updated as part of an agent's application logic ('on the hot path'). In this case, the agent typically decides to remember facts before responding to a user."

#### 6.8.1 Hot-Path Updates: In the Main Graph Execution

Memory is written during the same graph run that processes the user's message. The user's response is delayed by the time taken to extract and write memories.

```
User message → [memory_read] → [agent] → [memory_write] → Response
                                                  ↑ 
                                       Adds latency to every turn
```

```python
# Hot-path pattern: memory write in the graph
builder.add_node("read_memory", read_user_facts)
builder.add_node("agent", agent_node)
builder.add_node("write_memory", extract_and_store_facts)   # Runs before returning

builder.add_edge(START, "read_memory")
builder.add_edge("read_memory", "agent")
builder.add_edge("agent", "write_memory")
builder.add_edge("write_memory", END)
```

**When to use hot-path:**
- Memory updates are fast (simple extraction, no LLM call)
- Immediate feedback is important (the user said they hate tomatoes and you need to know before your very next response)
- The user expects the agent to acknowledge the update in the same turn

#### 6.8.2 Background Updates: Deferred Memory Processing

Memory extraction runs asynchronously after the response is returned to the user. The user is not kept waiting.

In LangGraph, this is implemented by triggering a background workflow (using a task queue or async subprocess) after the main graph completes, rather than including memory extraction as a graph node.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4)

def extract_memories_background(
    thread_id: str,
    user_id: str,
    conversation_snapshot: dict,
    store: BaseStore
):
    """Runs in a background thread after the main response is sent."""
    namespace = ("users", user_id, "facts")
    # Expensive extraction call — happens in background
    extracted = extract_facts_with_llm(conversation_snapshot)
    for fact in extracted:
        import uuid
        store.put(namespace, str(uuid.uuid4()), fact)

async def agent_with_background_memory(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Agent that triggers background memory extraction."""
    # ... main agent logic ...
    response = llm.invoke(state["messages"])
    
    user_id = config["configurable"].get("user_id")
    if user_id:
        # Fire and forget — doesn't block the response
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            _executor,
            extract_memories_background,
            config["configurable"]["thread_id"],
            user_id,
            {"messages": state["messages"]},
            store
        )
    
    return {"messages": [response]}
```

**LangMem's `ReflectionExecutor`** is the production-grade solution for background memory updates (covered in Section 6.10).

**When to use background:**
- Memory extraction is expensive (LLM call required)
- Low-latency response is more important than immediate memory availability
- Memory updates are not needed to personalize the *current* turn (they influence future turns)

#### 6.8.3 Decision Table

| Consideration | Hot Path | Background |
|---|---|---|
| Update latency added to response | Yes | No |
| Availability of update | Immediate (same session) | Next session (or later in same session) |
| Suitable for | Fast extraction, critical updates | LLM-based extraction, episodic summaries |
| Implementation complexity | Low | Higher (async, thread safety) |
| Standard LangGraph pattern | ✓ Node in graph | ✓ `ThreadPoolExecutor` or LangMem |

---

### 6.9 The `thread_id` vs. `user_id` Distinction

This is the most commonly confused concept in LangGraph memory design. They serve different purposes and should never be conflated:

```
thread_id:  "Which conversation is this?"
user_id:    "Who is having this conversation?"

One user can have many threads. One thread belongs to one user.
```

```python
# CORRECT: separate roles for each identifier
config = {
    "configurable": {
        "thread_id": "alice-research-session-20250115-001",  # Unique per conversation
        "user_id": "user-alice-12345",                        # Constant for Alice
    }
}

# The checkpointer uses thread_id to scope conversation history:
# → Checkpoint table: WHERE thread_id = "alice-research-session-..."

# The store uses user_id in namespaces to scope long-term memories:
# → Store: namespace = ("users", "user-alice-12345", "preferences")
#   → Available in ALL of Alice's threads

# WRONG: using user_id as thread_id
config = {
    "configurable": {
        "thread_id": "user-alice-12345",   # Now all of Alice's conversations
                                            # share the same checkpoint history!
                                            # This is almost never what you want.
    }
}
```

**A common architecture:** Each conversation gets a unique `thread_id` (often a UUID generated at session start). The `user_id` is stable across all of a user's sessions, typically coming from your authentication system.

```python
import uuid

def start_conversation(user_id: str) -> str:
    """Create a new thread_id for a user's new conversation."""
    thread_id = f"{user_id}-{uuid.uuid4()}"
    return thread_id

# At conversation start:
thread_id = start_conversation("user-alice")
config = {
    "configurable": {
        "thread_id": thread_id,
        "user_id": "user-alice",
    }
}
```

---

### 6.10 LangMem: The Official Memory Toolkit

LangMem (`pip install langmem`) is LangChain's higher-level toolkit for agent memory, built directly on top of LangGraph's `BaseStore`. It automates the three most tedious parts of memory engineering:

1. **Hot-path memory tools**: `create_manage_memory_tool` and `create_search_memory_tool` give the agent tools to explicitly create, update, and delete memories during a conversation
2. **Background memory manager**: `ReflectionExecutor` extracts and consolidates memories from completed conversations without adding latency
3. **Native LangGraph integration**: Works directly with `InMemoryStore` and `PostgresStore`

> **Note on imports:** Older LangMem examples show `from langgraph.prebuilt import create_react_agent`. As of LangGraph v1.0 (GA October 2025), `create_react_agent` is deprecated in favor of `langchain.agents.create_agent`, which runs on the same LangGraph runtime but adds a flexible middleware system. The function signature is nearly identical — the main change is `system_prompt=` instead of `prompt=`. The example below uses the current, non-deprecated import.

```python
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# Store with semantic search
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
    }
)

# Create an agent with explicit memory management tools
agent = create_agent(
    "openai:gpt-4o-mini",
    tools=[
        # Agent can call "manage_memory" to create/update/delete memories
        create_manage_memory_tool(namespace=("memories", "{user_id}")),
        
        # Agent can call "search_memory" to retrieve relevant memories
        create_search_memory_tool(namespace=("memories", "{user_id}")),
    ],
    system_prompt="You are a helpful assistant that remembers user preferences.",
    store=store,
)

# The agent now autonomously manages its own long-term memory
result = agent.invoke(
    {"messages": [("user", "I'm Alice. I prefer Python for all my projects.")]},
    config={"configurable": {"thread_id": "alice-1", "user_id": "alice"}}
)
```

**`ReflectionExecutor` for background memory extraction:**
```python
from langmem import ReflectionExecutor

# Set up background memory extraction
executor = ReflectionExecutor(store=store)

# After each conversation, trigger background extraction
# (this happens asynchronously — doesn't block the response)
async def on_conversation_end(thread_id: str, user_id: str):
    await executor.asubmit(
        {"configurable": {"thread_id": thread_id, "user_id": user_id}},
        namespace=("memories", user_id)
    )
```

LangMem is the recommended tool when:
- You want agents that autonomously decide what to remember (tool-calling approach)
- Background memory extraction is needed (ReflectionExecutor)
- You want extraction+consolidation logic without writing it from scratch

Build your own store nodes when:
- You need precise control over what gets stored and how
- Your memory structure is domain-specific and doesn't fit LangMem's general model
- You need custom deduplication logic

---

### 6.11 Production Memory Patterns

#### 6.11.1 Deduplication: Preventing Memory Bloat

As agents run many conversations, the store accumulates memories. Without deduplication, similar facts pile up:
```
"Alice prefers Python" (from session 1)
"Alice likes Python better than R" (from session 2)
"Alice uses Python for ML projects" (from session 3)
```

Three facts that say essentially the same thing. Deduplication strategies:

```python
def store_fact_with_dedup(
    namespace: tuple,
    new_fact: str,
    store: BaseStore,
    similarity_threshold: float = 0.92,
):
    """
    Store a fact only if no semantically similar fact already exists.
    """
    # Search for existing similar facts
    existing = store.search(namespace, query=new_fact, limit=1)
    
    if existing and existing[0].score and existing[0].score > similarity_threshold:
        # Too similar to an existing memory — update instead of adding
        # (preserves the most recent version)
        store.put(
            namespace,
            existing[0].key,   # Same key as the existing item
            {"text": new_fact, "updated": True}
        )
    else:
        # New, distinct fact — add it
        import uuid
        store.put(namespace, str(uuid.uuid4()), {"text": new_fact})
```

#### 6.11.2 Memory Privacy: Namespace Isolation

Every namespace that includes `user_id` is automatically isolated per user. But be explicit about this in your design:

```python
# SAFE: user-scoped namespace
user_namespace = ("users", user_id, "preferences")

# UNSAFE: shared namespace accidentally leaks data between users
shared_namespace = ("all_users", "preferences")   # Don't do this for user data

# Application-level knowledge (intentionally shared): fine
app_namespace = ("application", "faq")
```

For regulated industries (healthcare, finance), consider encrypting values before storing in the namespace:

```python
import json
from cryptography.fernet import Fernet

def encrypt_value(value: dict, key: bytes) -> dict:
    f = Fernet(key)
    encrypted = f.encrypt(json.dumps(value).encode())
    return {"__encrypted__": encrypted.decode()}

def decrypt_value(stored: dict, key: bytes) -> dict:
    if "__encrypted__" not in stored:
        return stored
    f = Fernet(key)
    return json.loads(f.decrypt(stored["__encrypted__"].encode()))
```

#### 6.11.3 TTL: Memory Expiration

Memories can become stale. A user's job title from two years ago may no longer be accurate. PostgresStore supports TTL at the database level:

```sql
-- Add TTL to items older than 90 days
-- (Run as a maintenance job)
DELETE FROM store 
WHERE namespace LIKE '%,preferences%' 
  AND updated_at < NOW() - INTERVAL '90 days';
```

In code, attach a timestamp to each stored item and filter during retrieval:

```python
from datetime import datetime, timedelta

def store_with_ttl(namespace, key, value, store: BaseStore, ttl_days: int = 90):
    """Store a value with an expiration timestamp."""
    store.put(
        namespace,
        key,
        {
            **value,
            "_expires_at": (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
        }
    )

def search_non_expired(namespace, query: str, store: BaseStore) -> list:
    """Search, filtering out expired memories."""
    now = datetime.utcnow().isoformat()
    results = store.search(namespace, query=query, limit=10)
    return [
        r for r in results
        if r.value.get("_expires_at", "9999") > now
    ]
```

---

### 6.12 MARRS Checkpoint: Adding Memory to the Research Agent

We will enhance MARRS with three forms of memory:

1. **Short-term**: `trim_messages` to prevent context window overflow during long research sessions
2. **Long-term semantic**: Store facts learned about research topics (avoids re-searching the same ground)
3. **Long-term episodic**: Store successful research patterns for few-shot guidance of the planner

```python
import json
import uuid
import operator
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STORE SETUP
# ─────────────────────────────────────────────────────────────────────────────

research_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text"],
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# STATE (extends Chapter 5's MARRSState with memory fields)
# ─────────────────────────────────────────────────────────────────────────────

class MARRSStateV2(TypedDict, total=False):
    topic: str
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    search_queries: list[str]
    findings: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]
    draft: str
    critique: str
    quality_score: float
    revision_count: int
    human_approved: bool
    human_feedback: str
    final_report: str
    status: str
    summary: str   # NEW: conversation summary for context compression


# ─────────────────────────────────────────────────────────────────────────────
# SHORT-TERM MEMORY: Summarization Node
# ─────────────────────────────────────────────────────────────────────────────

SUMMARIZE_AFTER_N_MESSAGES = 20

def maybe_summarize_messages(state: MARRSStateV2) -> dict:
    """
    Compresses message history when it grows too long.
    Keeps the last 6 messages verbatim; summarizes everything before that.
    Runs at the START of each planning cycle (before planner_node).
    """
    from langchain_core.messages import RemoveMessage
    
    messages = state.get("messages", [])
    if len(messages) < SUMMARIZE_AFTER_N_MESSAGES:
        return {}  # Not long enough to warrant summarization
    
    existing_summary = state.get("summary", "")
    messages_to_summarize = messages[:-6]  # Summarize everything except the last 6
    keep = messages[-6:]
    
    if not messages_to_summarize:
        return {}
    
    # Build summarization prompt
    history_text = "\n".join(
        f"{m.__class__.__name__}: {str(m.content)[:300]}"
        for m in messages_to_summarize
    )
    
    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            f"New messages to incorporate:\n{history_text}\n\n"
            f"Update the summary to include these new messages. Be concise."
        )
    else:
        prompt = (
            f"Summarize this research conversation concisely:\n{history_text}"
        )
    
    new_summary = llm.invoke([SystemMessage(content=prompt)]).content
    
    # Remove old messages — only keep the last 6
    removals = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    return {"summary": new_summary, "messages": removals}


# ─────────────────────────────────────────────────────────────────────────────
# LONG-TERM MEMORY: Memory-Enhanced Planner
# ─────────────────────────────────────────────────────────────────────────────

def planner_with_memory(
    state: MARRSStateV2,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """
    Planner that retrieves similar past research episodes from the store
    to use as few-shot examples.
    
    Reads from: ("research", "episodes") — episodic memory
    Reads from: ("research", "topic_facts") — semantic memory about past topics
    """
    topic = state.get("topic", "")
    
    # ── Retrieve episodic memory: past successful research approaches ─────────
    past_episodes = store.search(
        ("research", "episodes"),
        query=f"Research on: {topic}",
        limit=3,
        filter={"quality_score_gte": 0.80},  # Only high-quality episodes
    )
    
    # ── Retrieve semantic memory: known facts about related topics ────────────
    related_facts = store.search(
        ("research", "topic_facts"),
        query=topic,
        limit=5,
    )
    
    # ── Build context for the planner ─────────────────────────────────────────
    few_shot_context = ""
    if past_episodes:
        few_shot_context += "\n\nSuccessful past research approaches:\n"
        for ep in past_episodes:
            v = ep.value
            few_shot_context += (
                f"- Topic: {v.get('input', 'N/A')}\n"
                f"  Approach: {v.get('plan', 'N/A')[:200]}\n"
                f"  Quality: {v.get('quality_score', 0):.2f}\n\n"
            )
    
    prior_knowledge = ""
    if related_facts:
        prior_knowledge = "\n\nRelevant known facts from prior research:\n"
        prior_knowledge += "\n".join(f"- {r.value.get('text', '')}" for r in related_facts)
    
    # Inject conversation summary if available
    summary = state.get("summary", "")
    summary_context = f"\nConversation summary:\n{summary}\n" if summary else ""
    
    # Build planning prompt
    plan_prompt = f"""Create a research plan for: {topic}
    {summary_context}{prior_knowledge}{few_shot_context}
    
    Output JSON: {{"summary": "one-line overview", "search_queries": ["q1", "q2", ...]}}
    """
    
    try:
        response = llm.invoke([SystemMessage(content=plan_prompt)])
        plan_data = json.loads(response.content)
        plan = plan_data.get("summary", f"Research plan for {topic}")
        queries = plan_data.get("search_queries", [f"{topic} overview"])
    except (json.JSONDecodeError, AttributeError):
        plan = f"Research plan for {topic}"
        queries = [f"{topic} overview", f"{topic} recent developments"]
    
    return {"plan": plan, "search_queries": queries}


# ─────────────────────────────────────────────────────────────────────────────
# LONG-TERM MEMORY: Memory Writer Node
# Stores findings and successful episodes AFTER a good run
# ─────────────────────────────────────────────────────────────────────────────

def write_research_memory(
    state: MARRSStateV2,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """
    Writes to long-term memory AFTER a successful research run.
    Stores:
    1. Semantic memory: topic facts (key findings for future reference)
    2. Episodic memory: the successful research approach (for few-shot guidance)
    
    Only runs when quality_score is above threshold.
    """
    score = state.get("quality_score", 0.0)
    topic = state.get("topic", "")
    
    # Only store memory from high-quality outputs
    if score < 0.80 or not topic:
        return {}
    
    # ── Semantic memory: store key findings as facts ──────────────────────────
    findings = state.get("findings", [])
    for finding in findings[:5]:   # Limit to 5 findings per run
        if finding and len(finding) > 30:   # Skip trivially short findings
            store.put(
                ("research", "topic_facts"),
                str(uuid.uuid4()),
                {
                    "text": finding[:500],   # Truncate long findings
                    "topic": topic,
                    "source": "marrs_research",
                },
                index=["text"],
            )
    
    # ── Episodic memory: store the successful research approach ───────────────
    episode = {
        "input": topic,
        "plan": state.get("plan", ""),
        "quality_score": score,
        "revision_count": state.get("revision_count", 0),
        "search_queries": json.dumps(state.get("search_queries", [])),
        # Text for semantic search — what was this research episode about?
        "text": f"Research on: {topic}. Plan: {state.get('plan', '')}",
    }
    
    store.put(
        ("research", "episodes"),
        str(uuid.uuid4()),
        episode,
        index=["text"],
    )
    
    return {}   # No state updates — memory write is a pure side effect


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY WITH MEMORY
# ─────────────────────────────────────────────────────────────────────────────

def build_marrs_with_memory():
    """
    MARRS v4: Full persistence + both memory layers.
    
    Flow:
    START → maybe_summarize → planner_with_memory → researcher → writer
          → critic → [loop or] → human_review → finalize → write_research_memory → END
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    
    builder = StateGraph(MARRSStateV2)
    
    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("summarize", maybe_summarize_messages)
    builder.add_node("planner", planner_with_memory)
    builder.add_node("researcher", researcher_node)        # From Chapter 4
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node_command)        # Uses Command (Chapter 4)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("write_memory", write_research_memory)
    
    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "critic")           # critic uses Command for routing
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"finalize": "finalize", "reject": END, "revise": "writer"}
    )
    builder.add_edge("finalize", "write_memory")   # Always write memory after success
    builder.add_edge("write_memory", END)
    
    # ── Compile with both checkpointer AND store ───────────────────────────────
    conn = sqlite3.connect("marrs_v4.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    return builder.compile(
        checkpointer=checkpointer,
        store=research_store,     # Long-term memory store
    )


marrs_v4 = build_marrs_with_memory()


# ─────────────────────────────────────────────────────────────────────────────
# DEMONSTRATION: Memory Across Multiple Research Runs
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_memory():
    """
    Shows how memory improves subsequent research runs.
    Run 1: No memory (fresh start)
    Run 2: Has episodic memory from Run 1 — planner gets few-shot guidance
    """
    
    # Run 1: Research on transformers
    print("=== Run 1: Researching transformers (no memory) ===")
    config1 = {"configurable": {"thread_id": "run-1"}}
    result1 = marrs_v4.invoke(
        {"topic": "transformer architectures in NLP", "findings": [], "sources": []},
        config1
    )
    print(f"Status: {result1.get('status')}, Quality: {result1.get('quality_score', 0):.2f}")
    
    # Check what was stored in long-term memory
    episodes = list(research_store.search(("research", "episodes"), query="transformers"))
    facts = list(research_store.search(("research", "topic_facts"), query="transformers"))
    print(f"Stored {len(episodes)} episodes, {len(facts)} topic facts")
    
    # Run 2: Related topic — benefits from Run 1's episodic memory
    print("\n=== Run 2: Researching BERT (benefits from memory) ===")
    config2 = {"configurable": {"thread_id": "run-2"}}
    result2 = marrs_v4.invoke(
        {"topic": "BERT model fine-tuning techniques", "findings": [], "sources": []},
        config2
    )
    print(f"Status: {result2.get('status')}, Quality: {result2.get('quality_score', 0):.2f}")
    print("(Planner retrieved transformer research patterns as few-shot guidance)")
```

---

### 6.13 Chapter Summary

LangGraph provides a **two-layer memory architecture**. Short-term memory is thread-scoped, backed by the checkpointer, and contains the message history and current task state. Long-term memory is cross-thread, backed by the `BaseStore`, and contains user preferences, learned facts, and past experiences.

**Short-term memory management** requires active work as conversations grow: filtering (keep last N messages), trimming with `trim_messages` (token-aware, respects message format constraints), and summarization (compress old history into a rolling summary while preserving its content). Production systems typically combine all three.

**The `BaseStore` interface** provides four operations: `put` (write), `get` (read by key), `search` (find by content or meaning), and `delete`. All concrete backends implement this interface, making them interchangeable. `InMemoryStore` is for development; `PostgresStore` (with pgvector) is for production.

**Semantic search** — finding memories by meaning rather than exact key — is enabled by providing an `index` configuration with an embedding model and dimension count. It requires no changes to the `put`/`get` API.

**Three cognitive memory types** map to LangGraph patterns: semantic memory (facts about users and the world, stored as individual items, retrieved by query), episodic memory (past successful interactions, used as few-shot examples for the planner), and procedural memory (agent instructions, updated based on feedback — the reflection/meta-prompting pattern).

**Hot-path vs. background updates** trade latency against immediacy. Hot-path adds memory writes to the main graph execution; background uses async workers to extract memories after the response is sent.

**The `thread_id` vs. `user_id` distinction** is critical: `thread_id` scopes a conversation session; `user_id` scopes the user's identity across all sessions. Always include `user_id` in namespaces for user-scoped memories.

**LangMem** provides official higher-level tooling: memory management tools the agent can call explicitly, and `ReflectionExecutor` for background memory extraction.

---

### Further Reading

- **Official Memory overview**: `docs.langchain.com/oss/python/langgraph/memory` — the canonical reference; covers all three memory types, hot-path vs. background, and the Store interface
- **Official Short-term memory docs**: `docs.langchain.com/oss/python/langchain/short-term-memory` — `trim_messages` with middleware pattern
- **LangGraph memory launch blog post**: `blog.langchain.com/launching-long-term-memory-support-in-langgraph/` — the original announcement explaining the Store design philosophy
- **Semantic search launch**: `blog.langchain.com/semantic-search-for-langgraph-memory/` — how semantic search was added to `BaseStore`
- **LangMem GitHub**: `github.com/langchain-ai/langmem` — source code and documentation for the official memory toolkit
- **LangGraph source: `langgraph/store/memory/__init__.py`**: The `InMemoryStore` implementation — shows exactly how vector indexing and search work

---

*End of Chapter 6. Chapter 7: Human-in-the-Loop — Interrupts, Breakpoints, and Time Travel.*
