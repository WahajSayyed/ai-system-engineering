# LangGraph: From Zero to Production-Grade Multi-Agent Systems

A grounded, chapter-by-chapter curriculum covering LangGraph from first principles through production deployment, security, and red-teaming. Written in the style of a dense technical book: theory first, then annotated code, then honest production tradeoffs — no invented APIs, every non-obvious claim checked against current official documentation or source.

A single capstone project, **MARRS** (Multi-Agent Research & Report Writing System), is built incrementally across the chapters and serves as the running example throughout.

---

## Prerequisites

Comfortable Python (type hints, `async`/`await`, decorators). No prior LangChain/LangGraph experience assumed. Chapters build on each other in order — later chapters lean on vocabulary and mechanisms introduced earlier rather than re-explaining them.

---

## Table of Contents

| # | Chapter | What it covers |
|---|---|---|
| 1 | [The Mental Model: Why Graphs?](langgraph_chapter_01.md) | Why chains break down; the Pregel/Bulk Synchronous Parallel model; supersteps; how LangGraph maps onto Google's Pregel paper |
| 2 | [Core Architecture: State, Nodes, Edges](langgraph_chapter_02.md) | State schemas (TypedDict/Pydantic), node signatures, all edge types, `compile()` mechanics, invoke/stream basics |
| 3 | [Reducers, Annotations, and Advanced State Design](langgraph_chapter_03.md) | The update pipeline, `Annotated[T, fn]`, `add_messages` internals, custom reducers, input/output schema separation |
| 4 | [Control Flow: Conditional Edges, Cycles, Command, Send](langgraph_chapter_04.md) | `add_conditional_edges`, the ReAct loop, `Command` for routing+state updates, the `Send` API for dynamic fan-out, cycle termination patterns |
| 5 | [Persistence and Checkpointers](langgraph_chapter_05.md) | Checkpoint internals, threads, all checkpointer backends, `get_state`/`update_state`, time travel, the CVE-2025-64439 serialization advisory |
| 6 | [Memory: Short-Term and Long-Term](langgraph_chapter_06.md) | Message trimming/summarization, the `BaseStore` interface, semantic search, the three cognitive memory types, LangMem |
| 7 | [Human-in-the-Loop: Interrupts, Breakpoints, Time Travel](langgraph_chapter_07.md) | `interrupt()` mechanics and the re-execution rule, static breakpoints, seven HITL design patterns, the current interrupt-ID-keyed resume mechanism for parallel interrupts |
| 8 | [Streaming: Real-Time Output and Progress Monitoring](langgraph_chapter_08.md) | The five stream modes, `get_stream_writer()`, `astream_events`, streaming through subgraphs, production FastAPI/SSE wiring |
| 9 | [Tools and Tool-Calling](langgraph_chapter_09.md) | `@tool` internals, `ToolNode` parallel dispatch, structured output, `create_agent`, a four-class production error-handling framework |
| 10 | [Subgraphs and Multi-Agent Architectures](langgraph_chapter_10.md) | Subgraph composition and persistence modes, the current Subagents/Handoffs/Skills/Router taxonomy ("supervisor"/"swarm" in older vocabulary), quantified pattern tradeoffs |
| 11 | [Evaluation and Testing](langgraph_chapter_11.md) | The three-tier test strategy (unit/integration/evals), `agentevals` trajectory matching, LLM-as-judge, LangSmith tracing and CI/CD quality gates |
| 12 | [Deployment](langgraph_chapter_12.md) | LangSmith Deployment (formerly LangGraph Platform), the Agent Server model, the three deployment environments, double-texting, custom auth |
| 13 | [MCP and A2A](langgraph_chapter_13.md) | Connecting to external tools via `MultiServerMCPClient`, exposing agents as MCP tools, cross-framework delegation via A2A |
| 14 | [The Functional API: `@entrypoint` and `@task`](langgraph_chapter_14.md) | LangGraph's second, complementary paradigm — ordinary Python control flow with the same persistence/memory/HITL primitives underneath |
| 15 | [Guardrails and Agent Security](langgraph_chapter_15.md) | `PIIMiddleware`, `HumanInTheLoopMiddleware`, custom `before_agent`/`after_agent` guardrails, indirect prompt injection through tool output |
| 16 | [Sandboxing Infrastructure](langgraph_chapter_16.md) | Containers vs. gVisor vs. Firecracker, managed sandbox providers (E2B/Modal/Daytona), network egress control, resource limits |
| 17 | [Threat Modeling and Red Teaming](langgraph_chapter_17.md) | STRIDE and MAESTRO, cross-layer threat tracing, Garak/PyRIT/Promptfoo, continuous red-teaming wired into CI/CD |
| A | [Appendix: The Complete MARRS Reference Implementation](langgraph_appendix_a_marrs_reference.md) | Every chapter's MARRS contribution consolidated into one coherent, runnable codebase, with both the hand-built and Subagents architectures shown side by side |

---

## The MARRS Capstone Project

**MARRS** (Multi-Agent Research & Report Writing System) is a research-report-writing agent built incrementally: a skeleton graph in Chapter 2, production state design in Chapter 3, control flow in Chapter 4, persistence in Chapter 5, memory in Chapter 6, a human review gate in Chapter 7, streaming in Chapter 8, real tools in Chapter 9, a full architectural rethink as a supervisor-of-subagents in Chapter 10, a three-tier test suite in Chapter 11, a deployment package in Chapter 12, external tool/agent connectivity in Chapter 13, a Functional API rewrite in Chapter 14, and a layered security stack across Chapters 15–17.

Appendix A resolves the natural tension in following one project through an evolving book: it presents the complete, consolidated hand-built `StateGraph` version (which exercises every mechanism the book teaches) alongside the Chapter 10 Subagents version, so both are available as working references rather than scattered across chapters with earlier pieces superseded by later ones.

---

## A Note on Currency

This is a fast-moving ecosystem. Chapters 6–8 were revisited mid-curriculum against current documentation and corrected in place (each carries a "chapter revision note" where something changed — notably a deprecated import in Chapter 6's LangMem example and an updated multi-interrupt resume mechanism in Chapter 7). Every chapter is grounded in official documentation, source code, or changelogs current as of its writing date rather than training-data recall; where the ecosystem's own terminology shifted mid-book (LangGraph Platform → LangSmith Deployment, `create_react_agent` → `create_agent`), the chapter that introduced the older term is the one that explains the change.

---

## Suggested Use

Read in order once, since later chapters assume earlier vocabulary without re-explaining it (a routing function in Chapter 10 is written the way Chapter 4 taught, not re-derived). After that, each chapter stands alone as a reference — the chapter summaries and "Further Reading" sections at the end of every chapter are written to be useful without re-reading the whole thing.
