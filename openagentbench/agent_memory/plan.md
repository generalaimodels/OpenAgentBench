# Advanced Memory Management Sub-Module

## Execution Scheme for a Five-Layer, Multimodal, OpenAI-Compatible Memory System

---

## 1. Mission

This module is the stateful governance layer between the already-implemented retrieval stack and the downstream agent loop. It owns:

- working memory
- session memory summarization and checkpoints
- episodic memory capture and recall
- semantic memory write validation and conflict handling
- procedural memory extraction and matching
- cross-layer promotion
- cache, maintenance, and observability

This module does **not** duplicate:

- `agent_data` durable persistence ownership
- `agent_retrieval` retrieval classification, routing, and endpoint-compat ownership
- OpenAI or vLLM SDK client logic that already exists in shared compatibility surfaces

The output of this plan is an implementation sequence that can be executed without creating schema drift, enum drift, or endpoint drift relative to the existing repository.

---

## 2. Canonical Compatibility Decisions

### 2.1 Storage and Naming Canon

| Concept in Memory Design | Canonical Repo Surface | Required Decision |
|---|---|---|
| User partition key | `agent_data.*.user_id` | `user_id` is canonical on disk; `uu_id` remains a runtime alias only |
| Session metadata | `agent_data.sessions` | Session summary, budget metadata, checkpoint counters, and session-scoped state live here |
| Session turns | `agent_data.conversation_history` | Turn-level flags, modality markers, and compression state live here, not in `sessions` |
| Durable long-lived memory | `agent_data.memory_store` | L1-L4 durable memory rows are implemented by extending this table additively |
| Ephemeral working memory | `agent_data.memory_store_working` | Extend this table instead of creating a parallel durable design |
| Retrieval classification | `openagentbench.agent_retrieval.enums` and `models` | Reuse `QueryType`, `Modality`, `OutputStream`, `ModelRole`, `SignalTopology`, `LoopStrategy` |
| Context output | `openagentbench.agent_data.compiler` and `ChatMessage` | Memory compilation must emit OpenAI-compatible message payloads and preserve multimodal `content_parts` |
| Embedding / generation boundary | OpenAI-compatible client + vLLM | Keep provider injection external to memory-core logic |

### 2.2 Raw-Spec to Repo Mapping

The source architecture describes conceptual `session`, `history`, and `memory` tables. In this repository the correct mapping is:

- conceptual `session` turn rows -> `agent_data.conversation_history`
- conceptual session metadata row -> `agent_data.sessions`
- conceptual durable `memory` rows -> `agent_data.memory_store`
- conceptual working table -> `agent_data.memory_store_working`

The source architecture also models checkpoints as negative `turn_index` rows. In this repository, checkpoints should be stored in a dedicated companion table rather than overloading `conversation_history`.

### 2.3 Enum and Modality Normalization

The memory module must not introduce conflicting duplicate enum vocabularies. Normalize as follows:

- `structured_data` -> `Modality.STRUCTURED`
- `tool_trace` -> `Modality.TRACE` plus `OutputStream.TOOL_TRACE`
- session- and tool-runtime state -> `Modality.RUNTIME` when it is execution-state material rather than user content
- factual/procedural typing for durable memory -> new memory-specific enum on disk, adapter-mapped to retrieval `MemoryType`

### 2.4 Compatibility Contract

The module is considered compatible only if all of the following remain true:

- it can consume embeddings and chat/generation results from an OpenAI-compatible client
- it can consume local summarization, reranking, contradiction checks, and extraction jobs from vLLM via OpenAI-compatible payloads
- it can compile outputs into `List[ChatCompletionMessageParam]`-compatible shapes
- it preserves multimodal `content_parts` already supported by `agent_data`
- it does not break the current `agent_retrieval` compatibility tests

### 2.5 Above-SOTA Integration Matrix

| Integration Surface | Memory Responsibility | Above-SOTA Requirement |
|---|---|---|
| OpenAI-compatible module | embeddings, chat/responses generation, audio transcription, vision description, tool-call capture | batch embeddings, multimodal text projection, token-aware request shaping, payload compatibility for chat and responses surfaces |
| vLLM | summarization, reranking, contradiction checks, equivalence checks, pattern extraction, procedure synthesis | local low-latency secondary reasoning pipeline with strict fallback behavior when a specialist model is unavailable |
| `agent_data` | canonical persistence, multimodal `content_parts`, model API capture, protocol event capture, message compilation | memory writes must extend `agent_data` rather than bypass it; all durable state remains queryable through the same typed records and SQL ownership model |
| `agent_retrieval` | query classification, modality inference, topology, routing, selected model plan, retrieval contract | memory compilation must consume retrieval signals directly so normal, multimodal, thinking, and agentic-loop paths remain first-class and not downcast into a generic memory path |

Perfect integration in this module means:

- OpenAI-compatible and vLLM paths are both first-class, not bolt-on adapters
- `agent_data` remains the single durable data plane
- `agent_retrieval` remains the single retrieval and model-routing authority
- `agent_memory` becomes the governed lifecycle layer that binds them together without redefining their responsibilities

---

## 3. Non-Negotiable Engineering Rules

1. Every read and write path must filter by `user_id` before any other retrieval logic.
2. All schema work is additive to `agent_data`; no duplicate durable tables with alternate meanings.
3. No unbounded `O(n^2)` scans are allowed on per-user memory sets; dedup and conflict detection must use hash checks, ANN preselection, or bounded batches.
4. Background jobs must be idempotent, chunked, resumable, and safe under retry. Use lease-based work claiming or `FOR UPDATE SKIP LOCKED` semantics.
5. Multimodal inputs keep original references and `content_parts`; text projection is indexing material, not the source of truth.
6. Promotion into durable layers always follows a fixed gate: schema validation -> dedup -> conflict detection -> write -> audit -> cache invalidation.
7. All critical paths emit latency and decision metadata from the first executable version.
8. Cache writes, access-count bumps, and maintenance side effects must be async or batched so the hot path stays within the memory latency budget.
9. Cross-session and cross-user contamination checks fail closed and produce auditable events.
10. The memory module may depend on `agent_data` and `agent_retrieval`, but those existing modules must not become tightly coupled back to `agent_memory` internals.

---

## 4. Source-Spec Coverage Map

| Source Area | Execution Stage |
|---|---|
| Architectural boundary, protocols, MIMO scope | Stage 0 |
| Extended schema and indexes | Stage 1 |
| Working and session memory | Stage 2 |
| Episodic, semantic, and procedural memory | Stage 3 |
| Promotion pipeline | Stage 4 |
| Cache and prefill compilation | Stage 5 |
| Maintenance and observability | Stage 6 |
| MCP / function-tool exposure | Stage 7 |
| Innovation summary, latency target, rollout hardening | Stage 8 |

This keeps the advanced source design intact while converting it into an executable delivery order.

---

## 5. Target Package Layout

```text
openagentbench/agent_memory/
  __init__.py
  README.md
  plan.md
  runtime.py
  providers.py
  types.py
  enums.py
  models.py
  repository.py
  queries.py
  budgeting.py
  working.py
  session.py
  episodic.py
  semantic.py
  procedural.py
  promotion.py
  compiler.py
  cache.py
  maintenance.py
  observability.py
  endpoint_compat.py

agent_data/sql/
  002_agent_memory_schema.sql

tests/
  test_agent_memory.py
  test_agent_memory_api.py
  test_agent_memory_compat.py
  test_agent_memory_sql.py
```

Only memory-specific enums and models should live in `openagentbench/agent_memory`. Reuse existing `agent_data` and `agent_retrieval` types wherever possible.

---

## 6. Schema Rollout Plan

### 6.1 Extend Existing `agent_data` Tables

| Table | Additive Fields | Reason |
|---|---|---|
| `agent_data.sessions` | `summary_version`, `last_summarized_turn`, `checkpoint_seq`, `checkpoint_data`, `supported_modalities`, `active_task_metadata` | Session-wide summary and checkpoint governance |
| `agent_data.conversation_history` | `correction_flag`, `decision_flag`, `segment_boundary`, `message_modality`, `modality_ref`, `tool_trace_summary`, `compression_group_id` | Turn-local memory signals belong on turn rows |
| `agent_data.memory_store_working` | `agent_step_id`, `modality`, `binary_ref`, `utility_score`, `dependency_count`, `ttl_remaining`, `carry_forward`, `source_trace_id` | Working-memory lifecycle and overflow handling |
| `agent_data.memory_store` | `memory_type`, `authority_tier`, `modality`, `modality_ref`, `promotion_source`, `promotion_score`, `evaluation_record`, `action_trace`, `outcome_status`, `procedure_schema`, `procedure_status`, `procedure_version`, `test_suite_ref`, `success_rate`, `conflict_status`, `staleness_flags`, `promotion_chain` | Durable layer semantics for episodic, semantic, and procedural memory |

### 6.2 New Companion Tables

| Table | Shape | Purpose |
|---|---|---|
| `agent_data.session_checkpoints` | partitioned by `user_id` | Dedicated checkpoint snapshots; avoids overloading conversation turns |
| `agent_data.memory_promotion_log` | partitioned by `user_id` | Track promotions, rejections, merges, and deferrals |
| `agent_data.memory_audit_log` | partitioned by `user_id` | Operation-level audit, latency, token deltas, contamination checks |
| `agent_data.memory_cache` | `UNLOGGED` | L2 and L3 cache rows keyed by query hash / embedding bucket |
| `agent_data.memory_conflict_queue` | partitioned by `user_id` | Deferred canonical/curated conflict review queue |

### 6.3 Index Requirements

Minimum index set:

- partial B-tree indexes on `conversation_history` for `correction_flag`, `decision_flag`, `segment_boundary`
- B-tree on `memory_store(user_id, memory_tier, authority_tier, confidence)`
- partial B-tree on active procedures
- B-tree on `memory_store(user_id, conflict_status)` for unresolved conflicts
- B-tree on `memory_store(user_id, staleness_flags desc)`
- primary lookup on `memory_cache(cache_key)`
- HNSW on `memory_store.content_embedding` and `conversation_history.content_embedding` where embeddings are present

Every index strategy must preserve partition pruning by leading with `user_id`.

---

## 7. Stage-by-Stage Implementation Scheme

### Stage 0. Contract Alignment and Surface Freeze

**Goal**

Freeze the integration contract before code starts so the memory module extends the repo rather than forking its meanings.

**Code Deliverables**

- `openagentbench/agent_memory/README.md`
- `openagentbench/agent_memory/runtime.py`
- `openagentbench/agent_memory/providers.py`
- `openagentbench/agent_memory/types.py`
- `openagentbench/agent_memory/enums.py` for memory-only enums not already defined elsewhere

**Required Decisions**

- `user_id` is the canonical storage key; `uu_id` is an adapter alias
- turn-bound memory signals live on `conversation_history`
- `agent_data` remains the owner of SQL migrations
- `agent_retrieval` remains the owner of query classification and model-routing semantics

**Exit Gate**

- no duplicate table meaning remains in the plan
- import direction is one-way: `agent_memory` may import from existing modules, not vice versa
- memory terminology is normalized to existing repo types

### Stage 1. Schema Extension and Repository Base

**Goal**

Create the additive schema and the storage abstraction layer that all later phases build on.

**Code Deliverables**

- `agent_data/sql/002_agent_memory_schema.sql`
- `openagentbench/agent_memory/models.py`
- `openagentbench/agent_memory/repository.py`
- `openagentbench/agent_memory/queries.py`

**Implementation Notes**

- reuse partitioning strategy from `agent_data`
- use compact on-disk smallint enums where appropriate
- expose repository methods for checkpoints, audit events, cache, promotion logs, and memory writes
- keep query builders deterministic and easy to test, matching the style already used in `agent_data` and `agent_retrieval`

**Exit Gate**

- migration applies cleanly in an isolated schema
- no existing `agent_data` tests regress
- schema extensions remain additive and backward-compatible

### Stage 2. Working Memory and Session Memory

**Goal**

Implement the short-horizon layers: working memory, session turn ingestion, summarization, and checkpoints.

**Code Deliverables**

- `openagentbench/agent_memory/budgeting.py`
- `openagentbench/agent_memory/working.py`
- `openagentbench/agent_memory/session.py`

**Core Behavior**

- compute working-memory capacity from model context window, tool budget, output reserve, and layer claims
- carry forward at most `k = 0.30` of working-memory capacity between agent steps
- apply modality-aware overflow handling:
  - externalize expensive image/audio/video artifacts first
  - then utility-based pruning
  - then text compression
  - then large-item external references
  - then task decomposition escalation
- record turn-level correction, decision, and segment markers on `conversation_history`
- update `sessions.summary_text` incrementally and checkpoint state into `session_checkpoints`

**Compatibility Notes**

- preserve original multimodal `content_parts`
- text projections for image/audio are secondary index artifacts
- session compilation must still be consumable by `agent_data.compiler`

**Exit Gate**

- deterministic working-memory lifecycle behavior under repeated runs
- checkpoints are idempotent
- session summaries preserve corrections and decisions

### Stage 3. Durable Layer Implementation

**Goal**

Implement episodic, semantic, and procedural memory on top of the durable store.

**Code Deliverables**

- `openagentbench/agent_memory/episodic.py`
- `openagentbench/agent_memory/semantic.py`
- `openagentbench/agent_memory/procedural.py`

**Core Behavior**

- episodic:
  - record completed-task traces
  - support success-biased, failure-biased, correction-biased, and all-mode recall
  - consolidate similar episodes in bounded batches
- semantic:
  - exact-hash dedup first
  - semantic near-duplicate check second
  - contradiction detection third
  - authority-aware resolution last
- procedural:
  - mine frequent successful subsequences from episodic traces
  - synthesize candidate procedures
  - match procedures by embedding similarity plus precondition satisfaction

**Exit Gate**

- durable writes cannot bypass validation
- recall behavior is mode-aware and token-budget aware
- procedure matching can return direct invoke, guided planning, verification-only, or plan-from-scratch

### Stage 4. Promotion Pipeline and Conflict Governance

**Goal**

Centralize movement across layers so no item reaches durable memory without scoring, provenance, and audit.

**Code Deliverables**

- `openagentbench/agent_memory/promotion.py`

**Core Behavior**

- working -> session promotion
- session -> episodic promotion
- episodic -> semantic or procedural promotion
- conflict queue insertion for canonical or curated contradictions
- promotion log write-through
- audit log write-through

**Required Pipeline Order**

1. validate schema and provenance
2. compute promotion score
3. run exact dedup
4. run semantic dedup
5. run contradiction detection
6. resolve or defer conflict
7. write target row
8. append audit and promotion logs
9. invalidate affected cache keys

**Exit Gate**

- every durable write has auditable provenance
- deferred conflicts are queryable and recoverable
- no direct durable write path exists outside the promotion engine

### Stage 5. Prefill Compilation and Orchestration Integration

**Goal**

Compile memory into model-ready context while staying aligned with existing retrieval classification and compiler behavior.

**Code Deliverables**

- `openagentbench/agent_memory/compiler.py`

**Core Behavior**

- allocate per-layer budgets using task type, modality, and procedure-match state
- retrieve semantic, episodic, session, and procedural segments in parallel where possible
- include multimodal references only when supported by the chosen model capability profile
- emit OpenAI-compatible messages and optionally Responses-style segments without changing upstream payload meaning
- preserve provenance metadata for downstream citation and debugging

**Integration Contract**

- accept query-shape signals from `agent_retrieval.classify_query`
- respect routing decisions made by the retrieval module
- produce a context surface consumable by existing `agent_data` message assembly
- use OpenAI-compatible providers for embeddings, chat/responses generation, transcription, and multimodal description
- use vLLM-backed specialist providers for reranking, summarization, contradiction checks, equivalence checks, and procedure extraction
- persist all model-call and protocol metadata through `agent_data.model_api_calls` and `agent_data.protocol_events` when execution capture is enabled

**Exit Gate**

- normal single-model case stays cheap
- multimodal MIMO case preserves reference integrity
- thinking and agentic-loop cases remain compatible with current retrieval classifications
- provider routing between OpenAI-compatible surfaces and vLLM specialist surfaces is explicit, testable, and reversible

### Stage 6. Cache, Maintenance, and Observability

**Goal**

Add the performance and lifecycle systems needed for production use.

**Code Deliverables**

- `openagentbench/agent_memory/cache.py`
- `openagentbench/agent_memory/maintenance.py`
- `openagentbench/agent_memory/observability.py`

**Core Behavior**

- L1 in-process request cache
- L2 per-session `memory_cache`
- L3 per-user centroid or bucket cache
- TTL expiry, confidence decay, staleness archiving, dedup sweeps, capacity-cap enforcement
- health snapshots and alert thresholds:
  - utilization
  - hit rate
  - staleness ratio
  - conflict backlog
  - context pollution index

**Exit Gate**

- cache invalidates correctly on write and eviction
- maintenance jobs are safe under retry
- metrics are emitted for critical paths and background jobs

### Stage 7. Tool Surface and Control Plane Exposure

**Goal**

Expose memory as an agent-usable control plane without weakening safety boundaries.

**Code Deliverables**

- `openagentbench/agent_memory/endpoint_compat.py`

**Tool Surface**

- `memory_read`
- `memory_write`
- `memory_inspect`
- function tool schemas for:
  - `recall_similar_experiences`
  - `store_learned_fact`
  - `check_known_procedures`

**Control Rules**

- approvals or elevated policy gates for canonical and curated writes
- caller context always supplies `user_id`
- session-scoped reads must match caller session
- tool payloads remain OpenAI-compatible

**Exit Gate**

- tool schemas are deterministic and validated by tests
- privileged writes cannot occur without policy gates

### Stage 8. Performance, Soak, and Rollout Hardening

**Goal**

Validate latency, correctness, and rollback behavior before this module becomes a default dependency of the agent loop.

**Required Work**

- feature flags by layer and by operation
- shadow-mode promotion and audit-only mode
- benchmark harness for per-user memory sizes and multimodal payload mixes
- replay tests from captured `model_api_calls` and `protocol_events`
- migration backfill for summary versions, authority tiers, and modality defaults

**Exit Gate**

- memory subsystem contribution stays within the target hot-path budget
- backfill and rollback paths are tested
- existing retrieval and data compatibility suites remain green

---

## 8. Frozen Scoring and Policy Formulae

These equations are part of the contract and should be implemented consistently across code and tests.

### 8.1 Working-Memory Utility

```text
u_work(m, T) =
    0.40 * relevance(m, T)
  + 0.25 * recency(m)
  + 0.20 * dependency(m)
  + 0.15 * modality_value(m)
```

Default modality values:

- text = 1.0
- code = 1.2
- structured = 0.9
- trace = 0.8
- image/audio/video = 0.6 unless explicitly required by task

### 8.2 Episodic Recall Score

```text
S_recall(e, q) =
    0.40 * semantic_similarity(q, e)
  + 0.20 * recency_decay(e)
  + 0.25 * outcome_quality(e)
  + 0.15 * frequency(e)
```

### 8.3 Promotion Score

```text
S_promote(m) =
    0.30 * non_obviousness(m)
  + 0.40 * estimated_correctness_gain(m)
  + 0.30 * reusability(m)
```

### 8.4 Effective TTL

```text
tau_eff(m) = tau_base * (1 - eta * exp(-access_count / mu))
```

### 8.5 Capacity Constraint

```text
working_capacity =
    context_window
  - system_prompt
  - tool_budget
  - output_reserve
  - session_claim
  - episodic_claim
  - semantic_claim
  - procedural_claim
```

These values should be configurable, but their default weights must match the plan and tests.

---

## 9. Above-SOTA Acceptance Targets

The memory module is not considered above-SOTA unless the integrated system meets all of these targets together:

| Area | Target |
|---|---|
| OpenAI-compatible boundary | message, responses, embeddings, multimodal payloads, and tool payloads remain contract-valid |
| vLLM boundary | specialist memory tasks are locally executable with deterministic fallback behavior |
| `agent_data` integration | all durable writes, checkpoints, cache rows, audit rows, and execution captures land in the canonical schema family |
| `agent_retrieval` integration | memory honors query classification, modality routing, topology, and selected-model planning without semantic drift |
| Session summarization | corrections and decisions are preserved with zero silent loss on validation fixtures |
| Conflict governance | zero silent override of canonical knowledge |
| Procedure reuse | matched procedures show measurable token savings and lower planning overhead |
| Latency | memory hot path stays within the plan budget under warm cache |
| Isolation | zero tolerated cross-user or cross-session contamination events in test fixtures |

---

## 10. Operational Invariants

| Invariant | Enforcement |
|---|---|
| Every query is partition-pruned by `user_id` | repository API requires `user_id` |
| No cross-session session-memory reads | session-scoped methods require caller session match |
| No cross-user memory writes | repository and service layer validate caller `user_id` |
| No durable write without provenance | write validators reject missing provenance |
| No promotion without dedup and conflict checks | promotion engine is the only durable write path |
| No context assembly above budget | compiler asserts total selected tokens <= budget |
| Carry-forward remains bounded | working-memory GC enforces `k <= 0.30` |
| Canonical knowledge is never silently overridden | contradiction path defers to conflict queue |
| Multimodal artifacts remain recoverable | `content_parts` and `modality_ref` are preserved |
| Cache entries become stale on affected writes | write path triggers invalidation by user and layer |

Any invariant violation must emit an audit event with enough metadata to reconstruct the failure domain.

---

## 11. Test Plan

### 11.1 Unit and Contract Tests

| File | Coverage |
|---|---|
| `tests/test_agent_memory.py` | working-memory lifecycle, summary preservation, episodic recall, semantic conflict handling, procedure matching |
| `tests/test_agent_memory_sql.py` | migration shape, query builders, additive schema assertions, cache invalidation SQL, checkpoint persistence |
| `tests/test_agent_memory_compat.py` | OpenAI/vLLM payload compatibility, retrieval-classification interop, multimodal reference preservation |
| `tests/test_agent_memory_api.py` | MCP/function-tool schemas, service contract validation, admin inspection payloads |

### 11.2 Cross-Module Regression Gates

These suites must continue to pass unchanged:

- `tests/test_agent_data.py`
- `tests/test_agent_retrieval.py`
- `tests/test_agent_retrieval_compat.py`
- `tests/test_agent_retrieval_api.py`
- `tests/realtime/test_realtime_dry.py`

### 11.3 Performance Gates

Required benchmark checks:

- memory hot path target <= 30 ms added latency under warm cache
- cold cache still remains within the broader retrieval budget envelope
- no full-user duplicate scan path degenerates into quadratic behavior
- background maintenance jobs run in bounded batches
- OpenAI-compatible and vLLM specialist provider routes remain individually testable with deterministic fake providers

---

## 12. Definition of Done

The sub-module is complete only when all of the following are true:

1. The package layout exists with executable code, not only plan text.
2. The SQL migration is additive, reviewable, and validated in tests.
3. Session, episodic, semantic, and procedural memory all have typed models and repository support.
4. Promotion, checkpointing, cache, maintenance, and observability are implemented and tested.
5. OpenAI-compatible and vLLM-compatible payload paths are preserved and explicitly routed by role.
6. `agent_data` remains the canonical durable data plane for memory persistence and execution capture.
7. `agent_retrieval` remains the canonical classifier and routing authority and all of its existing test surfaces still pass.
8. The memory compiler can build model-ready context without violating token budgets or isolation invariants.
9. Rollout can be feature-flagged and reversed without data corruption.
10. The module satisfies the above-SOTA acceptance targets as an integrated system, not only as a local code unit.

---

## 13. Final Delivery Order

Execute in this exact order:

1. Stage 0: freeze contract and naming
2. Stage 1: add schema and repositories
3. Stage 2: implement working and session memory
4. Stage 3: implement episodic, semantic, and procedural layers
5. Stage 4: centralize promotion and conflict governance
6. Stage 5: wire prefill compilation into existing orchestration
7. Stage 6: add cache, maintenance, and observability
8. Stage 7: expose safe tool/control surfaces
9. Stage 8: run performance hardening and rollout

This ordering minimizes schema churn, avoids cyclic dependencies, keeps compatibility with `agent_data` and `agent_retrieval`, and converts the advanced source architecture into an implementation plan that can now be coded stage by stage.
