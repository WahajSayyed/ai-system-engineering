# LangGraph: From Zero to Production-Grade Multi-Agent Systems
## Chapter 5 — Persistence and Checkpointers

> *"LangGraph has a built-in persistence layer that saves graph state as checkpoints. When you compile a graph with a checkpointer, a snapshot of the graph state is saved at every step of execution, organized into threads. This enables human-in-the-loop workflows, conversational memory, time travel debugging, and fault-tolerant execution."*
> — LangGraph official documentation

---

### What This Chapter Covers

Every graph we built in Chapters 2–4 was ephemeral. The moment `.invoke()` returned, all state vanished. This chapter changes that permanently. Persistence is the capability that transforms an agent from a stateless script into a production system that can survive failures, resume interrupted work, maintain conversation history across sessions, and support human review workflows.

By the end you will understand:

1. Why persistence is architecturally necessary — not just a feature
2. What a checkpoint is in exact technical terms: the data structures, the versioning, the serialization
3. Threads: how LangGraph organizes checkpoints into isolated conversations
4. The full checkpointer ecosystem: `InMemorySaver`, `SqliteSaver`, `AsyncSqliteSaver`, `PostgresSaver`, `AsyncPostgresSaver` — what each is, when to use it, and how to configure it
5. The `BaseCheckpointSaver` interface: what four methods every checkpointer must implement
6. The `StateSnapshot` object: every field and what it tells you
7. `get_state()`, `get_state_history()`, and `update_state()`: the complete state management API
8. Time travel: forking from a historical checkpoint, replaying runs, debugging production failures
9. Fault tolerance and pending writes: what actually happens when a parallel node fails mid-superstep
10. Serialization: `JsonPlusSerializer`, what types it handles, the security advisory from late 2025
11. Production patterns: the Pointer State Pattern for large payloads, encrypted serialization, connection pooling
12. MARRS checkpoint: wiring full persistence into the capstone project

---

### 5.1 Why Persistence Is Architecturally Necessary

Consider what a graph without persistence can and cannot do:

**Without a checkpointer:**
- The graph runs to completion (or failure) — then all state disappears
- If a node fails, the entire run must restart from the beginning
- Multi-turn conversations are impossible: each call to `invoke()` is an isolated, stateless execution
- Human-in-the-loop cannot work: there is nowhere to park the paused graph while waiting for a human
- Time travel, replay, and debugging are impossible: no history was saved

**With a checkpointer:**
- After every superstep, the full state is serialized to a database
- A failed node can be retried from the last successful checkpoint
- A new call to `invoke()` with the same `thread_id` automatically loads prior context
- An interrupted graph can be resumed days later — the human reviews at their convenience
- The entire execution history is queryable: you can inspect any intermediate state

LangGraph's four documented use cases for persistence are, in the order they appear in the official docs: human-in-the-loop, conversational memory, time travel debugging, and fault-tolerant execution. All four require the same underlying mechanism: a checkpointer that saves state after every superstep.

---

### 5.2 What Is a Checkpoint? The Data Structure

A checkpoint is a snapshot of the complete graph state at one superstep boundary. It is a Python dict (internally a TypedDict) with the following fields, as defined in `langgraph.checkpoint.base`:

```python
# Simplified view of the Checkpoint TypedDict (from langgraph source)
class Checkpoint(TypedDict):
    v: int                    # Format version (currently v=2 for current serializer)
    ts: str                   # ISO 8601 timestamp when this checkpoint was created
    id: str                   # Unique checkpoint ID (monotonically increasing UUID-format string)
    channel_values: dict      # The actual state: {field_name: serialized_value}
    channel_versions: dict    # Version counter per channel: {channel: version_int}
    versions_seen: dict       # Which version of each channel each node has seen
                              # Used to determine what needs to run next
```

Let's look at a real checkpoint (from the PyPI langgraph-checkpoint documentation):

```python
{
    "v": 4,                                           # format version
    "ts": "2024-07-31T20:14:19.804150+00:00",        # when saved
    "id": "1ef4f797-8335-6428-8001-8a1503f9b875",    # unique checkpoint ID
    "channel_values": {                               # the actual state
        "my_key": "meow",
        "node": "node"
    },
    "channel_versions": {                             # version of each channel
        "__start__": 2,
        "my_key": 3,
        "start:node": 3,
        "node": 3
    },
    "versions_seen": {                                # what each node has processed
        "__input__": {},
        "__start__": {"__start__": 1},
        "node": {"start:node": 2}
    }
}
```

**The `channel_versions` and `versions_seen` fields** are the most technically interesting. They are not just metadata — they are the mechanism by which LangGraph determines which nodes to activate in the next superstep. When a node runs, it "sees" certain channel versions. After the superstep, channel versions increment. On the next superstep, a node is activated if there are channels it subscribes to whose version it has not yet seen. This is the direct translation of Pregel's "a vertex is activated by a new message" model.

**The `id` field** is a monotonically increasing string — a ULID-format identifier (sort lexicographically by time). This ordering property is what makes "get the most recent checkpoint" efficient: it is a simple `ORDER BY checkpoint_id DESC LIMIT 1` query.

---

### 5.3 Threads: Organizing Checkpoints Into Isolated Conversations

A **thread** is a named sequence of checkpoints, identified by a `thread_id`. Every checkpoint belongs to exactly one thread. Threads are the mechanism that allows LangGraph to serve multiple users or conversations simultaneously — each gets its own thread, and their states never collide.

```python
# Thread 1: User Alice's conversation
config_alice = {"configurable": {"thread_id": "user-alice-session-1"}}

# Thread 2: User Bob's conversation
config_bob = {"configurable": {"thread_id": "user-bob-session-1"}}

# These two invocations share the same compiled graph but have
# completely separate checkpoint histories:
result1 = graph.invoke({"topic": "quantum computing"}, config_alice)
result2 = graph.invoke({"topic": "transformer architectures"}, config_bob)
```

The `thread_id` is just a string. You are responsible for its uniqueness and meaning. Common patterns:

- **Per-user-session:** `f"user-{user_id}-session-{session_id}"` — a new thread per conversation
- **Per-user-task:** `f"user-{user_id}-task-{task_id}"` — one thread per long-running task
- **Per-user:** `f"user-{user_id}"` — one thread accumulates all of a user's history (risky: grows unboundedly)
- **Tenant-scoped:** `f"tenant-{tenant_id}-thread-{uuid4()}"` — safe isolation for multi-tenant SaaS

**Thread creation is implicit:** a thread comes into existence the first time you invoke with a new `thread_id`. There is no explicit "create thread" API call.

**Critical requirement:** When using a checkpointer, you *must* specify `thread_id` in the `configurable` portion of the config on every invocation. Without it, LangGraph cannot route to the correct checkpoint history.

```python
# WRONG — no thread_id, checkpointer cannot save/load correctly
graph.invoke({"topic": "AI"})

# CORRECT — always provide thread_id when a checkpointer is attached
graph.invoke({"topic": "AI"}, {"configurable": {"thread_id": "my-thread-1"}})
```

---

### 5.4 The Checkpointer Ecosystem

LangGraph's persistence system is modular. All checkpointers extend `BaseCheckpointSaver` from `langgraph-checkpoint` (the base package, included with `langgraph`). Concrete backends are separate packages.

#### 5.4.1 `InMemorySaver` — Development Only

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

**What it is:** Stores checkpoints in a Python dict in RAM. Zero dependencies beyond `langgraph` itself.

**Use for:**
- Local development and prototyping
- Unit tests (fast, no external services needed)
- Jupyter notebooks

**Do NOT use for:**
- Any multi-process deployment (each process has its own RAM dict; they do not share)
- Anything where state must survive process restart
- Production

**Note:** As of LangGraph 1.x, the canonical name is `InMemorySaver`. Older code may use `MemorySaver` — both exist in the codebase and are equivalent.

#### 5.4.2 `SqliteSaver` / `AsyncSqliteSaver` — Local Production or Single-Process

```bash
pip install langgraph-checkpoint-sqlite
```

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory SQLite (development — survives within process lifetime only)
conn = sqlite3.connect(":memory:", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)

# On-disk SQLite (persistent — survives process restarts)
conn = sqlite3.connect("./marrs_checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)
```

The `check_same_thread=False` parameter is required: Python's sqlite3 module by default raises an error if a connection is used from a thread other than the one that created it. LangGraph uses threads internally for parallel node execution, so you must disable this check.

**Async variant:**
```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

async def main():
    async with AsyncSqliteSaver.from_conn_string("./marrs.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        result = await graph.ainvoke({"topic": "AI"}, config)
```

**Use for:**
- Single-process applications (CLI tools, notebooks, small deployments)
- Local development with persistent state
- Demos and prototypes where you want a real file on disk

**Limitations:**
- Single-process only: SQLite's write locking prevents concurrent writes from multiple processes
- Not suitable for horizontally scaled deployments

#### 5.4.3 `PostgresSaver` / `AsyncPostgresSaver` — Production

```bash
pip install langgraph-checkpoint-postgres
```

**Synchronous (for sync graph execution):**
```python
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/marrs_db"

# Context manager form (recommended for scripts and CLI tools)
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()   # Creates tables on first run — idempotent
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke({"topic": "AI"}, config)

# Connection pool form (recommended for long-running servers)
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    DB_URI,
    max_size=10,
    kwargs={"autocommit": True, "row_factory": dict_row}
)
checkpointer = PostgresSaver(pool)
checkpointer.setup()
graph = builder.compile(checkpointer=checkpointer)
```

**Asynchronous (for async graph execution — recommended for web APIs):**
```python
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def setup_graph():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()   # Async setup
        graph = builder.compile(checkpointer=checkpointer)
        return graph

# Or with explicit connection pool (for FastAPI lifespan management):
pool = AsyncConnectionPool(DB_URI, max_size=10, open=False)

async def lifespan(app: FastAPI):
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    app.state.graph = builder.compile(checkpointer=checkpointer)
    yield
    await pool.close()
```

**The `setup()` / `asetup()` call:** This creates the necessary tables in the PostgreSQL database (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`). It is safe to call multiple times — it uses `IF NOT EXISTS` internally and is idempotent.

**Production best practice:** Call `setup()` once during application startup or as a database migration step. Do not call it in the hot path of request handling. For production deployments with CI/CD, run it as part of your migration pipeline.

**Use for:**
- Any multi-process or multi-instance deployment
- Horizontally scaled agents (multiple workers, Kubernetes)
- Long-running workflows where state must survive any infrastructure failure
- Applications where checkpoint history must be queryable and auditable

#### 5.4.4 Other Backends

LangGraph's modular design means anyone can implement a `BaseCheckpointSaver` for any database. Official and community backends include:

| Package | Backend | Use Case |
|---|---|---|
| `langgraph-checkpoint-sqlite` | SQLite | Single-process, local |
| `langgraph-checkpoint-postgres` | PostgreSQL | Multi-process production |
| `langgraph-checkpoint-cosmosdb` | Azure Cosmos DB | Production on Azure |
| `langgraph-checkpoint-mongodb` (JS) | MongoDB | Production on Atlas |
| `langgraph-checkpoint-redis` (JS) | Redis | High-throughput, short-lived state |
| Community | Snowflake, Couchbase, DynamoDB, etc. | Specific enterprise stacks |

---

### 5.5 The `BaseCheckpointSaver` Interface

Every checkpointer, regardless of backend, implements the same four core methods. Understanding them demystifies how checkpoints are saved and retrieved:

```python
from langgraph.checkpoint.base import BaseCheckpointSaver

class BaseCheckpointSaver:
    def put(
        self,
        config: RunnableConfig,       # Contains thread_id and checkpoint namespace
        checkpoint: Checkpoint,        # The state snapshot to persist
        metadata: CheckpointMetadata,  # Step number, source, parent checkpoint
        new_versions: ChannelVersions, # Updated channel version counters
    ) -> RunnableConfig:
        """
        Called after every superstep.
        Stores the checkpoint and returns a new config containing the
        generated checkpoint_id (used to retrieve this exact checkpoint later).
        """
        ...

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],  # (channel, value) pairs
        task_id: str,
    ) -> None:
        """
        Called when individual nodes complete within a superstep, BEFORE
        the superstep ends. Stores "pending writes" — the partial progress
        that enables fault-tolerant parallel execution.
        If a superstep fails, pending writes from successful nodes are
        preserved so they don't need to re-run on retry.
        """
        ...

    def get_tuple(
        self,
        config: RunnableConfig,
    ) -> CheckpointTuple | None:
        """
        Retrieves a specific checkpoint.
        If config contains checkpoint_id: fetches that exact checkpoint.
        If config contains only thread_id: fetches the LATEST checkpoint.
        Returns None if no checkpoint exists for this config.
        """
        ...

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """
        Lists checkpoints for a thread, most recent first.
        Supports filtering by metadata fields, pagination via 'before',
        and limiting the number of results.
        Used by graph.get_state_history().
        """
        ...
    
    def delete_thread(self, config: RunnableConfig) -> None:
        """Delete all checkpoints for a thread (cleanup)."""
        ...
```

For async execution, there are corresponding `aput`, `aput_writes`, `aget_tuple`, `alist`, and `adelete_thread` methods.

**When you call `graph.invoke()`, this is the sequence:**

```
1. Runtime calls checkpointer.get_tuple(config)
   → Loads the latest checkpoint for this thread_id
   → If no checkpoint: initializes fresh state

2. Graph executes superstep N
   → Nodes run, produce partial state updates

3. For each completed node (as it finishes):
   → Runtime calls checkpointer.put_writes(config, writes, task_id)
   → Pending writes are persisted mid-superstep (fault tolerance!)

4. Superstep N ends — all updates merged into new state

5. Runtime calls checkpointer.put(config, checkpoint, metadata, versions)
   → Full checkpoint (complete state snapshot) is persisted

6. Repeat from step 2 for the next superstep
```

---

### 5.6 The `StateSnapshot` Object: Reading a Checkpoint

When you call `graph.get_state(config)`, you get a `StateSnapshot` object. This is the public-facing wrapper around the raw checkpoint data, with all fields deserialized and ready to use:

```python
from langgraph.types import StateSnapshot

# Retrieve the current state
config = {"configurable": {"thread_id": "marrs-thread-1"}}
snapshot: StateSnapshot = graph.get_state(config)
```

The `StateSnapshot` has these fields:

```python
@dataclass
class StateSnapshot:
    values: dict                  # The current channel values (your actual state)
    next: tuple[str, ...]        # Tuple of node names that will run next
                                  # Empty tuple () = graph is finished
    config: RunnableConfig        # Config that identifies this exact checkpoint
                                  # Use this to "lock onto" this checkpoint for time travel
    metadata: dict                # Internal metadata: step number, source, parents
    created_at: str               # ISO timestamp when this checkpoint was created
    parent_config: RunnableConfig | None  # Config of the preceding checkpoint
    tasks: tuple[PregelTask, ...]  # Tasks scheduled for the next superstep
    interrupts: tuple             # Interrupt objects if graph is paused at an interrupt()
```

**Practical usage:**

```python
config = {"configurable": {"thread_id": "marrs-1"}}
snapshot = graph.get_state(config)

# Get the full state dict
state = snapshot.values
print(f"Current draft: {state.get('draft', '[not written yet]')[:100]}")
print(f"Quality score: {state.get('quality_score', 0):.2f}")

# Check if the graph is done
is_finished = len(snapshot.next) == 0
print(f"Graph complete: {is_finished}")

# If not finished, see what runs next
if snapshot.next:
    print(f"Next node(s): {snapshot.next}")

# Check if graph is paused at an interrupt()
if snapshot.interrupts:
    print(f"Waiting for human input: {snapshot.interrupts[0].value}")

# This config uniquely identifies this checkpoint for time travel
checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
print(f"Checkpoint ID: {checkpoint_id}")
```

---

### 5.7 The State Management API: `get_state`, `get_state_history`, `update_state`

#### 5.7.1 `get_state(config)`: Latest Snapshot

```python
# Get the latest state for a thread
latest = graph.get_state({"configurable": {"thread_id": "thread-1"}})

# Get a specific historical checkpoint
historical = graph.get_state({
    "configurable": {
        "thread_id": "thread-1",
        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"
    }
})
```

#### 5.7.2 `get_state_history(config)`: Full Execution History

```python
config = {"configurable": {"thread_id": "thread-1"}}

# Returns a generator of StateSnapshot objects, most recent first
for snapshot in graph.get_state_history(config):
    step = snapshot.metadata.get("step", "?")
    next_nodes = snapshot.next
    checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
    created = snapshot.created_at
    
    print(f"Step {step} | Next: {next_nodes} | ID: {checkpoint_id[:20]}... | {created}")

# Output (most recent first):
# Step 8 | Next: ()             | ID: 1ef4f797-8335-6428... | 2025-01-15T12:05:41...
# Step 7 | Next: ('finalize',)  | ID: 1ef4f797-7234-5317... | 2025-01-15T12:05:39...
# Step 6 | Next: ('critic',)    | ID: 1ef4f797-6123-4206... | 2025-01-15T12:05:37...
# ...
```

**Real output example from the LangGraph source/tests:**

```
StateSnapshot(
    values={'foo': 'b', 'bar': ['a', 'b']},
    next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '',
            'checkpoint_id': '1f070a87-33b5-66ae-8002-fd25026c289a'}},
    metadata={'source': 'loop', 'step': 2, 'parents': {}},
    created_at='2025-08-03T20:28:43.857059+00:00',
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '',
                   'checkpoint_id': '1f070a87-33b3-6d36-8001-50c306580336'}},
    tasks=(),
    interrupts=()
)
```

This is an actual `StateSnapshot` from a test. Notice:
- `next=()` → graph has finished (no more nodes to run)
- `metadata['step']=2` → this is checkpoint #2
- `parent_config` points to the preceding checkpoint
- `tasks=()` and `interrupts=()` → no pending work

#### 5.7.3 `update_state(config, values, as_node)`: Modifying State

`update_state` is one of LangGraph's most powerful and underused features. It lets you inject state changes into a thread's history — as if a node had run and produced those updates:

```python
config = {"configurable": {"thread_id": "thread-1"}}

# Scenario: The critic gave a bad score. We want to override it before resuming.
new_config = graph.update_state(
    config,
    {
        "quality_score": 0.90,        # Override the bad score
        "critique": "Human override: report is acceptable."
    },
    as_node="critic"   # Attribute this update to the 'critic' node
)
# new_config contains the checkpoint_id of the newly created checkpoint
# (update_state creates a new checkpoint that replaces the current state)

# Now resume from the corrected state
result = graph.invoke(None, new_config)
```

**What `update_state` does internally:**
1. Creates a new checkpoint with the provided values merged (via reducers) into the current state
2. Sets `next` to include whatever nodes would naturally run after `as_node` (based on the graph structure)
3. Returns a new config containing the `checkpoint_id` of this new checkpoint
4. The original checkpoint is preserved in history (non-destructive)

**`as_node` is important:** It determines which node's outgoing edges are used to compute `next`. If you set `as_node="critic"`, the new checkpoint's `next` will be whatever the critic normally routes to. If you omit `as_node`, LangGraph will try to infer it.

**Common use cases:**
- Correcting a bad LLM output before resuming
- Injecting human edits into the state
- Testing: seed state at a specific point without running prior nodes
- Debugging: replicate a production state locally and experiment

---

### 5.8 Time Travel: Forking From Historical Checkpoints

Time travel is the ability to invoke a graph starting from any historical checkpoint — not just the latest one. This creates a new execution branch without modifying the original history.

#### 5.8.1 The Mechanics

```python
config = {"configurable": {"thread_id": "thread-1"}}

# Get all historical checkpoints
history = list(graph.get_state_history(config))

# Identify the checkpoint you want to branch from
# (e.g., step 4 out of 8, before the quality went wrong)
target_snapshot = history[4]   # history is most-recent-first, so index 4 = step 4 from end
                                # But better to filter by step number explicitly:

target_snapshot = next(s for s in history if s.metadata.get("step") == 4)
target_checkpoint_id = target_snapshot.config["configurable"]["checkpoint_id"]

# Fork: invoke from the historical checkpoint
# A new thread_id is NOT required — LangGraph creates a new branch
# within the same thread using the checkpoint_id as the starting point
fork_config = {
    "configurable": {
        "thread_id": "thread-1",              # Same thread
        "checkpoint_id": target_checkpoint_id  # Start from step 4
    }
}

# This creates new checkpoints continuing from step 4, branching off the original
result = graph.invoke(
    None,           # None = resume from checkpoint, no new input
    fork_config
)
```

#### 5.8.2 Fork vs. Same-Thread Replay

**Fork (invoke from historical checkpoint, same thread):**
- Creates new checkpoints after the fork point in the same thread
- The original history (steps 5-8) is preserved alongside the new branch
- `get_state_history` returns both branches

**New thread with seeded state (cross-thread fork):**
- Copy state from an old thread into a new `thread_id`
- Cleaner isolation: the new thread has its own pristine history
```python
source_state = graph.get_state(old_config).values

new_config = {"configurable": {"thread_id": "thread-2-fork"}}
graph.update_state(new_config, source_state)
result = graph.invoke(None, new_config)
```

#### 5.8.3 Practical Use Cases for Time Travel

**Production debugging — investigating a bad decision:**
```python
# An agent made a wrong decision at step 5. Let's replay from step 4
# with improved state and see what would have happened differently.

history = list(graph.get_state_history(prod_config))
bad_step = next(s for s in history if s.metadata["step"] == 4)

# Fork from the step before the bad decision
fork_config = bad_step.config
# Inject a corrected state
graph.update_state(fork_config, {"better_context": "...corrected data..."})
# Now replay — what path does the agent take with the corrected context?
result = graph.invoke(None, fork_config)
```

**A/B testing different prompts:**
```python
# Run the graph to a branching point, then fork twice:
#   Branch A: writer with prompt template A
#   Branch B: writer with prompt template B

base_checkpoint_id = "the_checkpoint_before_writer"
base_config = {"configurable": {"thread_id": "test", "checkpoint_id": base_checkpoint_id}}

# Branch A
graph_a.invoke(None, base_config)  # graph_a has prompt template A in writer node

# Branch B — same starting point, different graph variant
graph_b.invoke(None, base_config)  # graph_b has prompt template B
```

**Error recovery without full restart:**
```python
# The graph crashed at step 7 (node "write_to_database" hit a transient error)
# Instead of starting over, resume from step 6:

history = list(graph.get_state_history(config))
last_successful = next(s for s in history if s.metadata["step"] == 6)

result = graph.invoke(None, last_successful.config)
# Only steps 7-END re-run — steps 1-6 are not repeated
```

---

### 5.9 Fault Tolerance and Pending Writes

A subtle but critical feature: LangGraph checkpoints **pending writes** during superstep execution, not just at superstep boundaries.

**The problem this solves:** Imagine a superstep with three parallel nodes: A, B, and C. After 30 seconds:
- Node A: ✅ completed
- Node B: ✅ completed
- Node C: ❌ crashed (database timeout)

Without pending writes: the entire superstep is lost. A, B, and C all re-run on retry — wasting time and API calls.

**With pending writes (LangGraph's actual behavior, from the official docs):**
> "When a graph node fails mid-execution at a given super-step, LangGraph stores pending checkpoint writes from any other nodes that completed successfully at that super-step. When you resume graph execution from that super-step you don't re-run the successful nodes."

So when the graph resumes:
- Node A: ⏭️ skipped (result already persisted in `checkpoint_writes` table)
- Node B: ⏭️ skipped (result already persisted)
- Node C: 🔄 retried (only this node re-runs)

This is implemented through the `put_writes()` checkpointer method, which is called immediately when each individual node completes — before the full superstep checkpoint is written. The `checkpoint_writes` table in PostgreSQL (or equivalent in other backends) stores these per-node partial results.

**What this means for your code:** You can design nodes to be **idempotent** (safe to re-run if necessary) without worrying about wasted work from partial failures in parallel steps. But you should still write nodes that are idempotent when possible, because if a node writes to an external system (database, email) and then crashes internally before returning, LangGraph may retry the node — and the external write has already happened.

---

### 5.10 Serialization: `JsonPlusSerializer` and What It Supports

Every value stored in a checkpoint must be serialized to bytes for storage. LangGraph uses `JsonPlusSerializer` by default, from `langgraph.checkpoint.serde.jsonplus`.

`JsonPlusSerializer` handles:
- All Python primitives: `str`, `int`, `float`, `bool`, `None`
- Collections: `list`, `dict`, `tuple`, `set`
- LangChain/LangGraph types: `BaseMessage` subclasses (`HumanMessage`, `AIMessage`, `ToolMessage`, etc.), `RunnableConfig`, LangGraph channels
- Common Python types: `datetime`, `date`, `timedelta`, `UUID`, `Enum`, `Decimal`
- Pydantic models: serialized via `.model_dump()` / reconstructed via `.model_validate()`

**How it works:** `JsonPlusSerializer` first tries to encode with `ormsgpack` (a fast MessagePack implementation). If ormsgpack fails (e.g., for LangChain types with custom Python objects), it falls back to an extended JSON format with type annotations.

**What it CANNOT serialize:**
- Python functions, lambdas, or closures
- File handles, sockets, or other OS resources
- Numpy arrays (use lists instead, or convert to lists before storing in state)
- Binary data larger than a few MB (causes database bloat — use the reference pattern instead)
- Circular references

**Practical implication:** State fields must contain JSON-serializable values (or types explicitly supported by `JsonPlusSerializer`). If you try to store an unsupported type in state and then checkpoint, you will get a serialization error. The fix is to convert to a supported type before returning from your node.

#### 5.10.1 The Security Advisory: `langgraph-checkpoint` < 3.0

In November 2025, a **critical remote code execution (RCE) vulnerability** was disclosed in `langgraph-checkpoint` versions prior to `3.0.0` (CVE-2025-64439):

> "Prior to version 3.0, LangGraph's `JsonPlusSerializer` contains an RCE vulnerability when deserializing payloads saved in 'json' mode. If an attacker can cause your application to persist a payload serialized in this mode, they may be able to execute arbitrary Python code during deserialization."

**The cause:** When `ormsgpack` serialization failed due to illegal Unicode surrogate values, the serializer fell back to a "json" mode that supported a constructor-style format for custom objects — essentially `eval()`-like behavior during deserialization.

**The fix:** Version 3.0.0 introduces an allowlist for constructor deserialization and deprecates the unsafe "json" fallback path.

**Action required:** Ensure you are running `langgraph-checkpoint >= 3.0.0`. This is included automatically with `langgraph >= 0.3`. If you use `langgraph-api`, version `0.5` or later includes the patched checkpointer.

```bash
# Check your version
pip show langgraph-checkpoint

# Upgrade
pip install --upgrade langgraph-checkpoint
```

#### 5.10.2 Encrypted Serialization

For sensitive state (PII, credentials, medical data), LangGraph provides an `EncryptedSerializer` that wraps any serializer with AES encryption:

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver

# AES encryption key (store securely — not in code)
encryption_key = b"your-32-byte-aes-key-goes-here!!"  # 32 bytes for AES-256

encrypted_serde = EncryptedSerializer.from_pycryptodome_aes(encryption_key)

checkpointer = PostgresSaver(conn, serde=encrypted_serde)
# All checkpoint data is encrypted before writing to the database
# and decrypted on read
```

```bash
pip install pycryptodome  # Required for EncryptedSerializer
```

Encryption adds per-read/write overhead. Use it when your threat model requires data-at-rest encryption (HIPAA, GDPR, financial data).

---

### 5.11 Production Patterns

#### 5.11.1 The Pointer State Pattern: Solving Checkpoint Bloat

The most important production pattern for checkpointing: **never store large binary payloads in state**.

**The problem** (from the official documentation and production reports):

> "LangGraph creates a new checkpoint at every step. If your agent state includes a 50MB PDF and the agent takes 10 steps, the checkpointer writes 500MB of data to PostgreSQL."

This causes:
- Massive database growth (PostgreSQL TOAST table bloat, WAL spikes)
- Slow checkpoint reads/writes as large blobs are serialized/deserialized on every step
- Query latency for all checkpoint operations

**The Pointer State Pattern:**

```python
# WRONG: Large payload stored directly in state
class BadState(TypedDict):
    document_content: bytes    # 5MB PDF — stored at EVERY checkpoint!
    images: list[bytes]        # Images — multiplied by N checkpoints

# CORRECT: Store only references; fetch content when needed
class GoodState(TypedDict):
    document_url: str          # "s3://my-bucket/docs/report.pdf"
    document_summary: str      # Extracted text (small, OK to checkpoint)
    image_urls: list[str]      # ["s3://my-bucket/img/1.png", ...]
    image_descriptions: list[str]  # Alt text / captions (small, OK)
```

**The full architecture:**
1. When a document/image arrives: upload to S3/GCS/Azure Blob
2. Store only the URL/key in state → checkpoint is tiny (bytes, not megabytes)
3. When a node needs the content: fetch it by URL using a helper function
4. Store derived representations (summaries, embeddings, metadata) in state — not the raw payload

The control-plane data (routing flags, counters, status, IDs, summaries) belongs in state. The data-plane data (raw documents, images, audio, large embedding vectors) belongs in external storage.

#### 5.11.2 Connection Pooling for PostgreSQL

For production applications handling concurrent users, use a connection pool rather than a single connection:

```python
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from contextlib import asynccontextmanager
from fastapi import FastAPI

DB_URI = "postgresql://user:password@postgres:5432/marrs_db"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open connection pool at startup
    pool = AsyncConnectionPool(
        DB_URI,
        max_size=20,              # Maximum 20 simultaneous connections
        max_idle=300.0,           # Close idle connections after 5 minutes
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=False,               # Don't open immediately
    )
    await pool.open()
    
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()   # Creates tables if not present
    
    app.state.graph = builder.compile(checkpointer=checkpointer)
    
    yield   # Application runs
    
    await pool.close()

app = FastAPI(lifespan=lifespan)
```

**Important note from the official docs:** PostgresSaver's default behavior holds a database connection for the entire duration of a graph run. For long-running workflows, this can cause connection timeout issues. The connection pool pattern above handles this correctly by borrowing and returning connections as needed.

#### 5.11.3 Thread Cleanup: Avoiding Unbounded Growth

Production systems create threads continuously. Without cleanup, the `checkpoints` table grows unboundedly:

```python
# Delete a specific thread's checkpoints (e.g., after workflow completes)
graph.checkpointer.delete_thread({"configurable": {"thread_id": "old-thread-id"}})

# For PostgreSQL, you can also use TTL-based cleanup:
# DELETE FROM checkpoints WHERE created_at < NOW() - INTERVAL '30 days';

# In production: run cleanup as a scheduled job
async def cleanup_old_threads(pool: AsyncConnectionPool, days_old: int = 30):
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM checkpoints WHERE created_at < NOW() - INTERVAL '%s days'",
            (days_old,)
        )
```

LangGraph Platform (the hosted service) handles TTL automatically. For self-hosted deployments, implement cleanup as a scheduled task.

---

### 5.12 Building and Testing Persistent Graphs

A practical challenge with persistence: unit tests that use a real SQLite/PostgreSQL checkpointer are slower and stateful. The solution is a two-tier test strategy:

```python
import pytest
import sqlite3
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# ── Fast unit tests: InMemorySaver ────────────────────────────────────────────

@pytest.fixture
def graph_with_memory():
    """Fast, stateless graph for unit tests."""
    checkpointer = InMemorySaver()
    return marrs_builder.compile(checkpointer=checkpointer)

def test_basic_flow_completes(graph_with_memory):
    config = {"configurable": {"thread_id": "test-1"}}
    result = graph_with_memory.invoke({"topic": "AI"}, config)
    assert result.get("status") in ("complete", "partial")

def test_conversation_persists_across_calls(graph_with_memory):
    """Second invocation should have context from the first."""
    config = {"configurable": {"thread_id": "conv-1"}}
    
    # First call
    graph_with_memory.invoke({"topic": "quantum"}, config)
    
    # Verify state was saved
    state = graph_with_memory.get_state(config)
    assert state.values.get("plan") is not None
    assert len(state.values.get("findings", [])) > 0

def test_multi_turn_isolation(graph_with_memory):
    """Different thread_ids should be completely isolated."""
    config_a = {"configurable": {"thread_id": "user-a"}}
    config_b = {"configurable": {"thread_id": "user-b"}}
    
    graph_with_memory.invoke({"topic": "topic A"}, config_a)
    graph_with_memory.invoke({"topic": "topic B"}, config_b)
    
    state_a = graph_with_memory.get_state(config_a)
    state_b = graph_with_memory.get_state(config_b)
    
    assert state_a.values["topic"] == "topic A"
    assert state_b.values["topic"] == "topic B"
    # States are completely isolated despite sharing the same graph/checkpointer

# ── Integration tests: SqliteSaver with temp file ─────────────────────────────

@pytest.fixture
def sqlite_graph(tmp_path):
    """SQLite-backed graph for integration tests."""
    db_path = tmp_path / "test_checkpoints.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = marrs_builder.compile(checkpointer=checkpointer)
    yield graph
    conn.close()

def test_state_survives_new_graph_instance(tmp_path):
    """
    True persistence test: state saved by one graph instance
    should be readable by a fresh instance using the same database.
    """
    db_path = str(tmp_path / "shared.db")
    thread_id = "persist-test"
    
    # First instance: save state
    conn1 = sqlite3.connect(db_path, check_same_thread=False)
    graph1 = marrs_builder.compile(checkpointer=SqliteSaver(conn1))
    graph1.invoke({"topic": "persistence test"}, {"configurable": {"thread_id": thread_id}})
    conn1.close()
    
    # Second instance: load from the same database
    conn2 = sqlite3.connect(db_path, check_same_thread=False)
    graph2 = marrs_builder.compile(checkpointer=SqliteSaver(conn2))
    state = graph2.get_state({"configurable": {"thread_id": thread_id}})
    
    # State should be present from the previous run
    assert state is not None
    assert state.values.get("topic") == "persistence test"
    conn2.close()
```

---

### 5.13 MARRS Checkpoint: Full Persistence Integration

Now we add persistence to MARRS. This transforms it from a stateless script into a production-grade system that can resume interrupted research, serve multiple concurrent users, and support human review workflows.

```python
import os
import sqlite3
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTER FACTORY
# Selects the appropriate backend based on environment
# ─────────────────────────────────────────────────────────────────────────────

def create_checkpointer(mode: str = "auto"):
    """
    Create the appropriate checkpointer for the current environment.
    
    mode="memory"   → InMemorySaver (testing, notebooks)
    mode="sqlite"   → SqliteSaver with on-disk DB (development, single-process)
    mode="postgres" → AsyncPostgresSaver (production, multi-process)
    mode="auto"     → Selects based on DATABASE_URL environment variable
    """
    if mode == "memory" or (mode == "auto" and not os.getenv("DATABASE_URL")):
        print("[Checkpointer] Using InMemorySaver (development mode)")
        return InMemorySaver()
    
    if mode == "sqlite" or (mode == "auto" and os.getenv("SQLITE_PATH")):
        path = os.getenv("SQLITE_PATH", "./marrs_checkpoints.db")
        print(f"[Checkpointer] Using SqliteSaver: {path}")
        conn = sqlite3.connect(path, check_same_thread=False)
        return SqliteSaver(conn)
    
    if mode == "postgres" or os.getenv("DATABASE_URL"):
        # Production: return async checkpointer
        # (requires psycopg_pool and langgraph-checkpoint-postgres)
        try:
            from psycopg_pool import AsyncConnectionPool
            from psycopg.rows import dict_row
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            
            db_uri = os.getenv("DATABASE_URL", "postgresql://localhost/marrs")
            print(f"[Checkpointer] Using AsyncPostgresSaver (production mode)")
            # Caller must manage pool lifecycle (see FastAPI lifespan below)
            return "postgres"  # Signal to caller to use async pattern
        except ImportError:
            print("[Checkpointer] psycopg_pool not installed, falling back to SQLite")
            return create_checkpointer("sqlite")
    
    return InMemorySaver()  # Fallback


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH WITH PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

# For development/testing
checkpointer = create_checkpointer("sqlite")

marrs_persistent = build_marrs_v3().compile(
    checkpointer=checkpointer,
    # interrupt_before=["human_review"]  # Add in Chapter 7
)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TURN CONVERSATION RUNNER
# Demonstrates how the same thread_id enables session memory
# ─────────────────────────────────────────────────────────────────────────────

def run_marrs_session(
    topic: str,
    thread_id: str,
    additional_context: Optional[str] = None,
) -> dict:
    """
    Run MARRS with persistence.
    
    If thread_id already has a checkpoint, the graph resumes from where
    it left off rather than starting fresh.
    
    This enables:
    1. Multi-turn refinement (call again with feedback)
    2. Session resumption (restart after failure)
    3. Human review and re-invocation
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if this thread already has state
    existing_state = marrs_persistent.get_state(config)
    
    if existing_state and existing_state.values:
        print(f"\n[Session] Resuming thread '{thread_id}'")
        print(f"[Session] Previous status: {existing_state.values.get('status', 'unknown')}")
        print(f"[Session] Previous quality score: {existing_state.values.get('quality_score', 0):.2f}")
        print(f"[Session] Next nodes: {existing_state.next}")
        
        if existing_state.next:
            # Graph was interrupted mid-run — resume it
            print("[Session] Graph was interrupted. Resuming...")
            result = marrs_persistent.invoke(None, config)
        else:
            # Graph completed — start a new run with the additional context
            print("[Session] Graph completed previous run. Starting refinement...")
            input_data = {
                "topic": topic,
                "findings": [],   # Reset findings for fresh research
                "sources": [],
            }
            if additional_context:
                input_data["messages"] = [
                    {"role": "user", "content": f"Previous run context: {additional_context}"}
                ]
            result = marrs_persistent.invoke(input_data, config)
    else:
        # New thread — fresh start
        print(f"\n[Session] Starting new thread '{thread_id}' for topic: {topic}")
        result = marrs_persistent.invoke(
            {"topic": topic, "findings": [], "sources": []},
            config
        )
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STATE INSPECTION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def inspect_thread_history(thread_id: str, max_snapshots: int = 10):
    """
    Print a summary of the checkpoint history for a thread.
    Useful for debugging and understanding what happened.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n{'='*60}")
    print(f"Checkpoint History: Thread '{thread_id}'")
    print(f"{'='*60}")
    
    snapshots = list(marrs_persistent.get_state_history(config))
    
    if not snapshots:
        print("  No checkpoints found for this thread.")
        return
    
    print(f"  Total checkpoints: {len(snapshots)}")
    print(f"  (Showing most recent {min(max_snapshots, len(snapshots))})\n")
    
    for i, snapshot in enumerate(snapshots[:max_snapshots]):
        step = snapshot.metadata.get("step", "?")
        next_nodes = snapshot.next if snapshot.next else ("COMPLETED",)
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        quality = snapshot.values.get("quality_score", 0.0)
        
        print(f"  Step {step:>3} │ Next: {str(next_nodes):<30} │ "
              f"Quality: {quality:.2f} │ ID: {checkpoint_id[:16]}...")


def fork_and_correct_state(
    thread_id: str,
    step_to_fork_from: int,
    corrections: dict,
) -> dict:
    """
    Time travel: fork from a historical checkpoint and inject corrections.
    
    Args:
        thread_id: The thread to fork
        step_to_fork_from: Which step number to branch from
        corrections: State updates to inject at the fork point
    
    Returns:
        The result of running the graph from the corrected fork point
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Find the checkpoint at the target step
    target = None
    for snapshot in marrs_persistent.get_state_history(config):
        if snapshot.metadata.get("step") == step_to_fork_from:
            target = snapshot
            break
    
    if target is None:
        raise ValueError(f"No checkpoint found at step {step_to_fork_from} in thread {thread_id}")
    
    print(f"[TimeTravel] Forking thread '{thread_id}' from step {step_to_fork_from}")
    print(f"[TimeTravel] Checkpoint ID: {target.config['configurable']['checkpoint_id'][:20]}...")
    print(f"[TimeTravel] Injecting corrections: {list(corrections.keys())}")
    
    # Inject corrections at the fork point
    fork_config = marrs_persistent.update_state(
        target.config,
        corrections,
    )
    
    # Resume execution from the corrected state
    result = marrs_persistent.invoke(None, fork_config)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION: FastAPI WITH POSTGRESQL (async)
# ─────────────────────────────────────────────────────────────────────────────

def create_fastapi_app():
    """
    Production FastAPI application with PostgreSQL persistence.
    
    Demonstrates proper lifecycle management for the connection pool
    and checkpointer setup.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        import json
        from psycopg_pool import AsyncConnectionPool
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:
        print("FastAPI/psycopg not installed. Skipping FastAPI app creation.")
        return None
    
    DB_URI = os.getenv("DATABASE_URL", "postgresql://localhost/marrs_db")
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage PostgreSQL connection pool lifecycle."""
        pool = AsyncConnectionPool(
            DB_URI,
            max_size=20,
            max_idle=300.0,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=False,
        )
        await pool.open()
        
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        
        # Store the compiled graph in app state for access by route handlers
        app.state.graph = build_marrs_v3().compile(checkpointer=checkpointer)
        
        print(f"[Server] Connected to PostgreSQL. MARRS ready.")
        yield
        
        await pool.close()
        print("[Server] PostgreSQL pool closed.")
    
    app = FastAPI(title="MARRS API", lifespan=lifespan)
    
    @app.post("/research")
    async def start_research(topic: str, thread_id: str):
        """Start a new research run."""
        config = {"configurable": {"thread_id": thread_id}}
        
        async def event_stream():
            async for event in app.state.graph.astream(
                {"topic": topic, "findings": [], "sources": []},
                config,
                stream_mode="updates"
            ):
                for node_name, updates in event.items():
                    yield f"data: {json.dumps({'node': node_name, 'keys': list(updates.keys())})}\n\n"
            yield "data: {\"done\": true}\n\n"
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    
    @app.get("/research/{thread_id}/state")
    async def get_research_state(thread_id: str):
        """Get current state of a research run."""
        config = {"configurable": {"thread_id": thread_id}}
        state = app.state.graph.get_state(config)
        
        if not state or not state.values:
            return {"status": "not_found"}
        
        return {
            "status": state.values.get("status", "unknown"),
            "quality_score": state.values.get("quality_score", 0.0),
            "revision_count": state.values.get("revision_count", 0),
            "is_complete": len(state.next) == 0,
            "next_nodes": list(state.next),
        }
    
    @app.get("/research/{thread_id}/history")
    async def get_research_history(thread_id: str):
        """Get checkpoint history for a research thread."""
        config = {"configurable": {"thread_id": thread_id}}
        history = []
        
        async for snapshot in app.state.graph.aget_state_history(config):
            history.append({
                "step": snapshot.metadata.get("step"),
                "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
                "created_at": snapshot.created_at,
                "next": list(snapshot.next),
                "quality_score": snapshot.values.get("quality_score", 0),
            })
            if len(history) >= 20:  # Limit response size
                break
        
        return {"thread_id": thread_id, "checkpoints": history}
    
    return app
```

---

### 5.14 Common Mistakes and How to Avoid Them

#### 5.14.1 Forgetting `thread_id`

```python
# WRONG — no thread_id
result = graph.invoke({"topic": "AI"})
# With a checkpointer attached, this raises a ValueError or silently
# creates a thread with a random/empty ID

# CORRECT
result = graph.invoke({"topic": "AI"}, {"configurable": {"thread_id": "my-thread"}})
```

#### 5.14.2 Calling `setup()` in the Hot Path

```python
# WRONG — setup() runs on every request
async def handle_request(topic: str):
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # Creates tables every time!
    graph = builder.compile(checkpointer=checkpointer)
    return await graph.ainvoke(...)

# CORRECT — setup() once at startup, graph shared across requests
# (see FastAPI lifespan pattern above)
```

#### 5.14.3 Storing Large Objects in State

```python
# WRONG
class State(TypedDict):
    pdf_bytes: bytes      # 5MB × 20 checkpoints = 100MB per thread

# CORRECT
class State(TypedDict):
    pdf_url: str          # "s3://bucket/doc.pdf" — a few bytes per checkpoint
    pdf_summary: str      # Extracted text (small)
```

#### 5.14.4 Using `InMemorySaver` in Multi-Process Deployments

```python
# WRONG for multi-process (uvicorn --workers 4, Kubernetes)
# Each process has its own RAM, so checkpoints are NOT shared
checkpointer = InMemorySaver()

# CORRECT for multi-process
checkpointer = PostgresSaver(conn)  # Shared external database
```

#### 5.14.5 Not Handling the `None` Input for Resumption

```python
# When resuming a paused or interrupted graph, pass None as input
# (not the original input — the state is already loaded from the checkpoint)

# WRONG — re-invokes with original input, which creates a new computation
# from the beginning of the thread's state (not resume)
result = graph.invoke({"topic": "AI"}, config)  # When you meant to resume

# CORRECT — None means "load from checkpoint and continue"
result = graph.invoke(None, config)
```

---

### 5.15 Chapter Summary

**Persistence** transforms ephemeral scripts into stateful production agents. Without it, every `invoke()` is isolated; with it, state accumulates and survives across calls, processes, and failures.

**Checkpoints** are complete serialized snapshots of graph state at superstep boundaries. They contain channel values, channel versions (for determining what to run next), and metadata. Each checkpoint has a unique, monotonically increasing ID.

**Threads** are named sequences of checkpoints identified by `thread_id`. Each thread is an isolated conversation or workflow run. Multi-tenancy is achieved by assigning each user/session a unique `thread_id`.

**The checkpointer backends** follow a clear progression: `InMemorySaver` for development, `SqliteSaver` for single-process production, `PostgresSaver`/`AsyncPostgresSaver` for multi-process production. All implement the same four-method `BaseCheckpointSaver` interface.

**Pending writes** provide intra-superstep fault tolerance: individual nodes' results are persisted as they complete, preventing re-execution of successful nodes when a parallel node fails.

**The state management API** — `get_state()`, `get_state_history()`, `update_state()` — gives full programmatic access to the checkpoint history. This is the foundation for time travel, debugging, and human-in-the-loop corrections.

**Time travel** allows forking from any historical checkpoint, enabling: production bug reproduction, A/B testing, error recovery without full restart, and "what-if" exploration.

**Serialization** uses `JsonPlusSerializer` (ormsgpack + extended JSON fallback). Keep `langgraph-checkpoint >= 3.0.0` to avoid the RCE vulnerability from the json fallback path. Use `EncryptedSerializer` for sensitive state.

**The Pointer State Pattern** — storing references to large objects rather than the objects themselves — is mandatory for any production application that processes documents, images, or other large payloads.

---

### Further Reading

- **Official Persistence docs**: `docs.langchain.com/oss/python/langgraph/persistence` — the canonical reference; covers threads, checkpoints, fault tolerance, and the Store interface
- **`langgraph-checkpoint` PyPI**: `pypi.org/project/langgraph-checkpoint/` — the base interface documentation with the raw `Checkpoint` TypedDict definition
- **CVE-2025-64439 security advisory**: `github.com/langchain-ai/langgraph/security/advisories/GHSA-wwqv-p2pp-99h5` — the RCE vulnerability in checkpoint serialization; upgrade guidance
- **LangGraph v0.2 blog post**: `blog.langchain.com/langgraph-v0-2/` — the announcement of the separate checkpointer packages and their design philosophy
- **"Understanding Checkpointers, Databases, API Memory and TTL"**: LangChain support documentation on managing PostgreSQL connection pools, TTL, and checkpoint bloat in production
- **LangGraph source: `libs/checkpoint/langgraph/checkpoint/base/__init__.py`**: The `BaseCheckpointSaver`, `Checkpoint`, `CheckpointTuple`, and `CheckpointMetadata` definitions — reading the source resolves any ambiguity about the data model

---

*End of Chapter 5. Chapter 6: Memory — Short-Term and Long-Term.*
