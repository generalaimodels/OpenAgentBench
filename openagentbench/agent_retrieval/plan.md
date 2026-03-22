

# SOTA+ Hybrid Multi-Tier Retrieval System

## Per-User PostgreSQL Isolation, Dual-Model (OpenAI + vLLM), MIMO Architecture

---

## 1. Architectural Overview and Design Rationale

### 1.1 Core Thesis

The retrieval system is architected as a **user-partitioned, three-table, dual-model, multi-input-multi-output (MIMO) evidence engine** that achieves above-SOTA performance through six interlocking innovations:

1. **User-scope-first partition pruning** — every retrieval operation begins with $O(1)$ user isolation via PostgreSQL range/hash partitioning on `uu_id`, eliminating cross-user interference and reducing candidate space by orders of magnitude before any scoring occurs.
2. **Three-table retrieval hierarchy** — `session`, `history`, and `memory` tables per user carry structurally distinct data with distinct freshness profiles, authority tiers, and retrieval strategies; retrieval across all three tables is orchestrated as a single fused pipeline rather than three independent lookups.
3. **Dual-model pipeline** — OpenAI-compatible embeddings (via `python-openai`) for high-dimensional semantic encoding and vLLM-served models for local, low-latency cross-encoder reranking, query decomposition, and faithfulness verification.
4. **MIMO retrieval orchestration** — multiple input representations (raw query, rewritten queries, session-contextualized query, decomposed subqueries) are simultaneously routed to multiple output streams (text evidence, code evidence, structured data, runtime state), each ranked and budgeted independently, then merged under a unified token budget.
5. **Adaptive fusion with negative feedback** — fusion weights are dynamically adjusted per query type, and negative evidence (documents that historically led to task failures) is actively suppressed.
6. **Progressive retrieval with quality-gated exit** — an iterative retrieve-evaluate-refine loop ensures that insufficient evidence triggers targeted follow-up retrieval rather than premature termination.

### 1.2 Compatibility Constraints

| Module | Role | Interface |
|---|---|---|
| `python-openai` | Embedding generation, chat completion for query rewriting | OpenAI-compatible API (local or remote endpoint) |
| `vllm` | Cross-encoder reranking, query decomposition, claim verification | Local OpenAI-compatible serving with custom models |
| PostgreSQL + pgvector | Per-user storage, exact search (GIN/GiST), semantic search (HNSW) | SQL + pgvector operators |

### 1.2A Execution Coverage Matrix

The retrieval module must classify and support all of the following workload shapes without leaking them into slower generic paths:

| Case | Classification Contract | Endpoint / Execution Stance |
|---|---|---|
| **Normal single-model** | `single_model + siso + direct + single_pass` | Minimal prompt shaping, no loop tax |
| **Multimodal / MIMO** | `single_model or dual_model + mimo` when multiple input modalities and output streams are present | Preserve modality-aware routing and stream fan-out |
| **Thinking / reasoning** | `reasoning` query type or `thinking/deliberate/self_reflective` effort | Prefer OpenAI Responses API because modern reasoning-capable models expose configurable reasoning there |
| **Agentic loop / tool loop** | `agentic` query type with `tool_loop`, `agentic_loop`, or `critique_repair` strategy | Prefer stateful Responses-style execution and tool-trace retrieval |
| **Dual-model retrieval** | `dual_model` when embedding + reranker path is explicit | Keep reranking isolated from embedding generation for latency control |
| **Multi-model orchestration** | `multi_model` when planner/executor/critic roles are required | Classify explicitly so planning cost is paid only when necessary |

### 1.3 Performance Targets

| Metric | Target | Justification |
|---|---|---|
| Precision@10 | $\geq 0.80$ | Context purity for bounded token budgets |
| Recall@20 | $\geq 0.90$ | Evidence completeness across three tables |
| NDCG@10 | $\geq 0.85$ | Ranking quality with graded relevance |
| MRR | $\geq 0.90$ | First-hit quality for single-answer queries |
| Faithfulness | $\geq 0.92$ | Hallucination suppression |
| End-to-end P99 latency | $\leq 250\text{ms}$ | Agent loop iteration budget |
| Token efficiency | $\geq 0.75$ | Fraction of budget tokens that are decision-relevant |

---

## 2. Per-User PostgreSQL Schema Architecture

### 2.1 Partitioning Strategy

The database uses **hash partitioning** on `uu_id` across all three tables. Each user's data is co-located within a single partition, ensuring that the query planner performs partition pruning as the very first operation, reducing I/O to only the relevant user's data.

### 2.2 Table Schemas (Pseudo-Structural)

```
TABLE: session
  PARTITION BY HASH(uu_id)
  COLUMNS:
    session_id          UUID        PRIMARY KEY
    uu_id               UUID        NOT NULL  -- partition key
    turn_index          INT         NOT NULL  -- ordinal within session
    role                ENUM(user, assistant, system, tool)
    content_text        TEXT        NOT NULL
    content_embedding   VECTOR(dim) NOT NULL  -- pgvector, dim matches openai model
    tokens_used         INT
    tool_calls          JSONB       NULLABLE  -- structured tool invocations
    metadata            JSONB       -- arbitrary typed metadata
    created_at          TIMESTAMPTZ DEFAULT NOW()
    expires_at          TIMESTAMPTZ -- session TTL, auto-cleaned
  INDEXES:
    GIN(content_text)                  -- trigram + full-text search
    HNSW(content_embedding)            -- pgvector ANN index
    BTREE(uu_id, created_at DESC)      -- recency scan
    BTREE(uu_id, session_id, turn_index) -- ordered session replay
    BTREE(expires_at)                  -- TTL cleanup

TABLE: history
  PARTITION BY HASH(uu_id)
  COLUMNS:
    history_id          UUID        PRIMARY KEY
    uu_id               UUID        NOT NULL
    query_text          TEXT        NOT NULL
    query_embedding     VECTOR(dim) NOT NULL
    response_summary    TEXT        -- compressed agent output
    evidence_used       JSONB       -- array of {chunk_id, source, utility_score}
    task_outcome        ENUM(success, partial, failure, unknown)
    human_feedback      ENUM(approved, rejected, corrected, none)
    utility_score       FLOAT       -- composite utility in [0, 1]
    negative_flag       BOOLEAN     DEFAULT FALSE  -- marks anti-patterns
    tags                TEXT[]      -- categorical labels
    metadata            JSONB
    created_at          TIMESTAMPTZ DEFAULT NOW()
    session_origin      UUID        REFERENCES session(session_id)
  INDEXES:
    GIN(query_text)                    -- full-text search on past queries
    HNSW(query_embedding)              -- semantic similarity to past queries
    BTREE(uu_id, utility_score DESC)   -- top-utility retrieval
    BTREE(uu_id, created_at DESC)      -- recency
    BTREE(uu_id, task_outcome)         -- outcome filtering
    GIN(tags)                          -- tag-based filtering
    BTREE(uu_id, negative_flag)        -- negative pattern lookup

TABLE: memory
  PARTITION BY HASH(uu_id)
  COLUMNS:
    memory_id           UUID        PRIMARY KEY
    uu_id               UUID        NOT NULL
    memory_type         ENUM(fact, preference, correction, constraint, procedure, schema)
    content_text        TEXT        NOT NULL
    content_embedding   VECTOR(dim) NOT NULL
    authority_tier      ENUM(canonical, curated, derived, ephemeral)
    confidence          FLOAT       -- in [0, 1]
    source_provenance   JSONB       -- chain-of-custody
    verified_by         UUID[]      -- principal IDs who verified
    supersedes          UUID[]      -- memory_ids this entry replaces
    created_at          TIMESTAMPTZ DEFAULT NOW()
    updated_at          TIMESTAMPTZ DEFAULT NOW()
    expires_at          TIMESTAMPTZ NULLABLE
    access_count        INT         DEFAULT 0  -- usage frequency
    last_accessed_at    TIMESTAMPTZ
    content_hash        BYTEA       -- SHA-256 for deduplication
    metadata            JSONB
  INDEXES:
    GIN(content_text)                  -- exact/trigram search
    HNSW(content_embedding)            -- semantic search
    BTREE(uu_id, memory_type)          -- type-filtered retrieval
    BTREE(uu_id, authority_tier DESC, confidence DESC) -- authority ranking
    BTREE(uu_id, updated_at DESC)      -- freshness
    UNIQUE(uu_id, content_hash)        -- deduplication constraint
    BTREE(expires_at)                  -- TTL cleanup
```

### 2.3 Index Design Rationale

| Index Type | Purpose | Complexity |
|---|---|---|
| GIN (trigram) | Exact substring, prefix, and fuzzy text match | $O(\sum \|\text{postings}(t)\|)$ per query |
| HNSW (pgvector) | Approximate nearest-neighbor semantic search | $O(\log N)$ amortized |
| B-tree (composite) | Range scans, ordered access, partition-pruned lookups | $O(\log N)$ |
| Unique (content_hash) | Prevents duplicate memory entries | $O(1)$ amortized |

The critical insight: **B-tree indexes on `(uu_id, ...)` enable the PostgreSQL planner to combine partition pruning with index seeks**, achieving $O(\log n_u)$ where $n_u$ is the number of rows for a single user, rather than $O(\log N)$ over the entire corpus.

---

## 3. Execution Order — Critical Path Analysis

The optimal execution order is determined by **dependency analysis** and **latency-hiding through parallelism**. The following DAG defines the critical path:

```
EXECUTION DAG:

Phase 0: User Scope Lock            [0ms]
    |
Phase 1: Parallel Load              [5ms]
    ├── Load Session Context
    └── Load Memory Summary
    |
Phase 2: Query Preprocessing        [15-25ms]
    ├── Query Classification
    ├── Contextual Query Augmentation
    ├── Query Rewriting (vLLM)
    └── Query Decomposition into Subqueries
    |
Phase 3: Parallel Multi-Stream Retrieval  [80-150ms]
    ├── Stream A: Exact Retrieval (BM25 + Trigram) across 3 tables
    ├── Stream B: Semantic Retrieval (Dense + pgvector) across 3 tables
    ├── Stream C: Historical Utility Retrieval from history table
    └── Stream D: Memory-Augmented Retrieval from memory table
    |
Phase 4: Fusion                      [5-10ms]
    |
Phase 5: Cross-Encoder Reranking (vLLM)  [40-60ms]
    |
Phase 6: MMR Diversity Selection     [3ms]
    |
Phase 7: Provenance + Token Budget   [5ms]
    |
Phase 8: Response Assembly           [2ms]

TOTAL CRITICAL PATH: ~160-260ms (within 250ms P99 target)
```

---

## 4. Phase 0 — User Scope Lock

```
ALGORITHM: UserScopeLock
INPUT:  uu_id: UUID
OUTPUT: scoped_context: UserScope

1.  ASSERT uu_id IS NOT NULL AND IS VALID UUID
2.  // PostgreSQL partition pruning: all subsequent queries include WHERE uu_id = :uu_id
3.  // This is NOT a SELECT; it is a scope declaration that binds to all downstream queries
4.  scoped_context ← UserScope {
        uu_id: uu_id,
        partition_key: HASH(uu_id) MOD num_partitions,
        session_table: "session",
        history_table: "history",
        memory_table: "memory",
        acl_scope: ACLResolver.resolve(uu_id)
    }
5.  // Verify user exists and is active
6.  user_exists ← QUERY: SELECT 1 FROM users WHERE uu_id = :uu_id AND status = 'active'
7.  IF NOT user_exists:
8.      RAISE UserNotFoundError(uu_id)
9.  RETURN scoped_context
```

**Key property**: Every single query issued after this point includes `WHERE uu_id = :uu_id`, which triggers PostgreSQL partition pruning. This is the **first and most selective filter** in the entire pipeline, reducing the search space from millions of global rows to at most thousands of user-specific rows.

---

## 5. Phase 1 — Parallel Session and Memory Context Load

```
ALGORITHM: ParallelContextLoad
INPUT:  scope: UserScope, session_id: UUID (current session)
OUTPUT: session_ctx: SessionContext, memory_summary: MemorySummary

1.  // Launch both loads in parallel
2.  PARALLEL:
3.      // Branch A: Load current session context
4.      session_turns ← QUERY:
5.          SELECT turn_index, role, content_text, tool_calls, created_at
6.          FROM session
7.          WHERE uu_id = :scope.uu_id
8.            AND session_id = :session_id
9.          ORDER BY turn_index DESC
10.         LIMIT max_session_turns  -- typically 20-50 turns
11.     session_ctx ← SessionContext {
12.         turns: REVERSE(session_turns),
13.         turn_count: |session_turns|,
14.         last_user_query: EXTRACT_LAST_USER_TURN(session_turns),
15.         topic_trajectory: TopicExtractor.extract(session_turns),
16.         active_tool_context: EXTRACT_TOOL_STATE(session_turns),
17.         session_start: MIN(session_turns.created_at),
18.         session_duration: NOW() - MIN(session_turns.created_at)
19.     }
20.
21.     // Branch B: Load validated memory summary
22.     memory_items ← QUERY:
23.         SELECT memory_id, memory_type, content_text, authority_tier,
24.                confidence, updated_at, access_count
25.         FROM memory
26.         WHERE uu_id = :scope.uu_id
27.           AND (expires_at IS NULL OR expires_at > NOW())
28.           AND authority_tier IN ('canonical', 'curated', 'derived')
29.         ORDER BY
30.           CASE authority_tier
31.             WHEN 'canonical' THEN 4
32.             WHEN 'curated' THEN 3
33.             WHEN 'derived' THEN 2
34.             ELSE 1
35.           END DESC,
36.           confidence DESC,
37.           access_count DESC
38.         LIMIT max_memory_items  -- typically 50-100 items
39.     memory_summary ← MemorySummary {
40.         facts: FILTER(memory_items, type='fact'),
41.         preferences: FILTER(memory_items, type='preference'),
42.         corrections: FILTER(memory_items, type='correction'),
43.         constraints: FILTER(memory_items, type='constraint'),
44.         procedures: FILTER(memory_items, type='procedure'),
45.         total_items: |memory_items|,
46.         compressed_text: MemoryCompressor.compress(memory_items, max_tokens=500)
47.     }
48. END_PARALLEL
49.
50. RETURN (session_ctx, memory_summary)
```

**Design rationale**: Loading session and memory in parallel hides latency. The memory summary is pre-compressed to a token budget (500 tokens) to avoid bloating downstream query augmentation. Session context captures the **topic trajectory** — the sequence of topics discussed — which biases retrieval toward the user's current line of inquiry.

---

## 6. Phase 2 — Query Preprocessing, Rewriting, and Decomposition

### 6.1 Query Classification

```
ALGORITHM: QueryClassifier
INPUT:  raw_query: string, session_ctx: SessionContext
OUTPUT: query_class: QueryClassification

1.  // Feature extraction (deterministic, no model call)
2.  features ← {
3.      length: token_count(raw_query),
4.      has_identifiers: REGEX_MATCH(raw_query, /[A-Z][a-zA-Z]+\.[a-z]+|[a-z_]+\(|0x[0-9a-f]+/),
5.      has_code_markers: REGEX_MATCH(raw_query, /```|def |class |import |SELECT |FROM /),
6.      is_question: ENDS_WITH(raw_query, '?') OR STARTS_WITH_ANY(raw_query, ['what','how','why','when','where','which','is','can','does']),
7.      has_temporal_ref: REGEX_MATCH(raw_query, /yesterday|last week|recently|today|ago|since/),
8.      session_topic_overlap: JACCARD(
9.          TOKENIZE(raw_query),
10.         TOKENIZE(session_ctx.topic_trajectory)
11.     ),
12.     is_followup: session_ctx.turn_count > 0 AND (
13.         STARTS_WITH_ANY(raw_query, ['it','this','that','the same','also','and']) OR
14.         session_topic_overlap > 0.3
15.     )
16. }
17.
18. // Classification (rule-based for speed; no model call needed)
19. query_class ← QueryClassification {
20.     type: CLASSIFY(features),
21.     // Types: FACTUAL, CONCEPTUAL, PROCEDURAL, CODE, DIAGNOSTIC, CONVERSATIONAL
22.     retrieval_bias: COMPUTE_BIAS(features),
23.     // Bias: {bm25_weight, dense_weight, memory_weight, history_weight}
24.     requires_decomposition: features.length > 30 OR CONTAINS_CONJUNCTION(raw_query),
25.     requires_coreference_resolution: features.is_followup,
26.     requires_temporal_scoping: features.has_temporal_ref
27. }
28.
29. RETURN query_class

FUNCTION: CLASSIFY(features) → QueryType
  IF features.has_code_markers OR features.has_identifiers:
      RETURN CODE
  IF features.has_temporal_ref AND contains_error_terms(raw_query):
      RETURN DIAGNOSTIC
  IF starts_with_how(raw_query):
      RETURN PROCEDURAL
  IF starts_with_why(raw_query):
      RETURN CONCEPTUAL
  IF features.is_question:
      RETURN FACTUAL
  RETURN CONVERSATIONAL

FUNCTION: COMPUTE_BIAS(features) → RetrievalBias
  // Adaptive weights based on query type
  SWITCH CLASSIFY(features):
    CASE CODE:       RETURN {bm25: 0.5, dense: 0.25, memory: 0.1, history: 0.15}
    CASE DIAGNOSTIC: RETURN {bm25: 0.3, dense: 0.3, memory: 0.1, history: 0.3}
    CASE FACTUAL:    RETURN {bm25: 0.35, dense: 0.4, memory: 0.15, history: 0.1}
    CASE CONCEPTUAL: RETURN {bm25: 0.2, dense: 0.5, memory: 0.15, history: 0.15}
    CASE PROCEDURAL: RETURN {bm25: 0.25, dense: 0.35, memory: 0.2, history: 0.2}
    DEFAULT:         RETURN {bm25: 0.3, dense: 0.35, memory: 0.15, history: 0.2}
```

### 6.2 Coreference Resolution and Contextual Query Augmentation

```
ALGORITHM: ContextualQueryAugmentation
INPUT:  raw_query: string, query_class: QueryClassification,
        session_ctx: SessionContext, memory_summary: MemorySummary
OUTPUT: augmented_query: AugmentedQuery

1.  resolved_query ← raw_query
2.
3.  // Step 1: Coreference resolution for follow-up queries
4.  IF query_class.requires_coreference_resolution:
5.      // Use vLLM for fast local inference
6.      resolved_query ← VLLM_INFERENCE(
7.          model: "coreference-resolver",  -- or general LLM with system prompt
8.          system_prompt: "Resolve all pronouns and references in the user query "
9.                       + "using the conversation history. Return ONLY the resolved query.",
10.         context: LAST_N_TURNS(session_ctx, n=5),
11.         user_input: raw_query,
12.         max_tokens: 200,
13.         temperature: 0.0  -- deterministic
14.     )
15.
16. // Step 2: Temporal scoping
17. temporal_scope ← NULL
18. IF query_class.requires_temporal_scoping:
19.     temporal_scope ← TemporalParser.parse(resolved_query)
20.     // Returns: TimeWindow { start, end } or RelativeWindow { duration, anchor }
21.
22. // Step 3: Memory-informed constraint injection
23. // Inject relevant user constraints/preferences into the query representation
24. applicable_constraints ← FILTER(
25.     memory_summary.constraints,
26.     λ c: SEMANTIC_OVERLAP(c.content_text, resolved_query) > 0.4
27. )
28. applicable_preferences ← FILTER(
29.     memory_summary.preferences,
30.     λ p: SEMANTIC_OVERLAP(p.content_text, resolved_query) > 0.3
31. )
32.
33. augmented_query ← AugmentedQuery {
34.     original: raw_query,
35.     resolved: resolved_query,
36.     constraints: applicable_constraints,
37.     preferences: applicable_preferences,
38.     temporal_scope: temporal_scope,
39.     session_topic: session_ctx.topic_trajectory,
40.     query_class: query_class
41. }
42.
43. RETURN augmented_query
```

### 6.3 Query Decomposition and Subquery Generation

```
ALGORITHM: QueryDecomposer
INPUT:  augmented_query: AugmentedQuery
OUTPUT: subqueries: List<SubQuery>

1.  IF NOT augmented_query.query_class.requires_decomposition:
2.      // Single query, no decomposition needed
3.      RETURN [SubQuery {
4.          text: augmented_query.resolved,
5.          target_tables: [session, history, memory],
6.          retrieval_modes: [EXACT, SEMANTIC],
7.          priority: 1.0,
8.          temporal_scope: augmented_query.temporal_scope
9.      }]
10.
11. // Complex query: decompose via vLLM
12. decomposition ← VLLM_INFERENCE(
13.     model: "query-decomposer",
14.     system_prompt:
15.       "Decompose the following query into independent subqueries. "
16.       "Each subquery should target a single information need. "
17.       "For each subquery, specify: "
18.       "  1) The subquery text "
19.       "  2) Whether it needs EXACT match, SEMANTIC search, or BOTH "
20.       "  3) Whether it targets SESSION (recent context), HISTORY (past interactions), "
21.       "     or MEMORY (long-term knowledge), or ALL "
22.       "  4) Priority (1.0 = critical, 0.5 = supporting, 0.2 = optional) "
23.       "Return as structured list.",
24.     user_input: augmented_query.resolved,
25.     max_tokens: 500,
26.     temperature: 0.0
27. )
28.
29. subqueries ← PARSE_STRUCTURED(decomposition)
30.
31. // Validate and bound
32. IF |subqueries| > MAX_SUBQUERIES:  -- typically 5
33.     subqueries ← TOP_BY(subqueries, λ sq: sq.priority, k=MAX_SUBQUERIES)
34.
35. // Add the original resolved query as a catch-all subquery
36. APPEND SubQuery {
37.     text: augmented_query.resolved,
38.     target_tables: ALL,
39.     retrieval_modes: [EXACT, SEMANTIC],
40.     priority: 0.8,
41.     temporal_scope: augmented_query.temporal_scope,
42.     is_original: TRUE
43. } TO subqueries
44.
45. RETURN subqueries
```

### 6.4 Contextual Embedding Generation

```
ALGORITHM: ContextualEmbeddingGenerator
INPUT:  subqueries: List<SubQuery>, session_ctx: SessionContext,
        memory_summary: MemorySummary
OUTPUT: embedded_subqueries: List<EmbeddedSubQuery>

1.  embedded_subqueries ← EMPTY_LIST
2.
3.  // Construct context prefix for embedding augmentation
4.  // This shifts the embedding vector toward the user's actual intent
5.  context_prefix ← ""
6.  IF session_ctx.turn_count > 0:
7.      context_prefix ← "Context: " + TRUNCATE(
8.          session_ctx.topic_trajectory, max_tokens=50
9.      ) + ". "
10. IF memory_summary.compressed_text IS NOT EMPTY:
11.     context_prefix += "User knowledge: " + TRUNCATE(
12.         memory_summary.compressed_text, max_tokens=30
13.     ) + ". "
14.
15. // Batch embed all subqueries via openai-compatible API
16. texts_to_embed ← []
17. FOR EACH sq IN subqueries:
18.     // Variant 1: Plain subquery embedding
19.     APPEND sq.text TO texts_to_embed
20.     // Variant 2: Context-augmented embedding
21.     APPEND (context_prefix + sq.text) TO texts_to_embed
22.
23. // Single batch call to openai-compatible embedding endpoint
24. all_embeddings ← OPENAI_EMBED(
25.     model: embedding_model_name,  -- e.g., "text-embedding-3-large"
26.     input: texts_to_embed,
27.     dimensions: embedding_dim      -- e.g., 3072 or 1536
28. )
29.
30. // Pair embeddings with subqueries
31. FOR i = 0 TO |subqueries| - 1:
32.     plain_emb ← all_embeddings[2*i]
33.     augmented_emb ← all_embeddings[2*i + 1]
34.
35.     // Weighted combination: bias toward augmented for follow-up queries
36.     IF subqueries[i].is_followup:
37.         combined_emb ← 0.3 * plain_emb + 0.7 * augmented_emb
38.     ELSE:
39.         combined_emb ← 0.6 * plain_emb + 0.4 * augmented_emb
40.
41.     NORMALIZE(combined_emb)  -- L2 normalize
42.
43.     APPEND EmbeddedSubQuery {
44.         subquery: subqueries[i],
45.         plain_embedding: plain_emb,
46.         augmented_embedding: augmented_emb,
47.         combined_embedding: combined_emb
48.     } TO embedded_subqueries
49.
50. RETURN embedded_subqueries
```

**Above-SOTA innovation**: The **dual embedding with weighted combination** captures both the literal query intent (plain) and the user's contextual intent (augmented). The weight shift for follow-up queries ensures that conversational context dominates when the query is ambiguous in isolation. This is superior to single-embedding approaches because it preserves recall for both modes.

---

## 7. Phase 3 — Parallel Multi-Stream Retrieval (MIMO)

### 7.1 MIMO Architecture

The MIMO (Multi-Input Multi-Output) retrieval architecture operates as follows:

**Multiple Inputs**:
- $I_1$: Raw subquery text (for BM25/trigram)
- $I_2$: Combined embedding (for dense semantic search)
- $I_3$: Query-derived metadata filters (temporal scope, content type, authority tier)
- $I_4$: Session context (for session-aware boosting)
- $I_5$: Memory constraints (for constraint-aware filtering)

**Multiple Output Streams**:
- $O_S$: Session evidence stream (recent, high-freshness)
- $O_H$: History evidence stream (utility-weighted, proven-useful)
- $O_M$: Memory evidence stream (validated, canonical)
- $O_N$: Negative evidence stream (anti-patterns to suppress)

Each stream operates independently with its own retrieval strategy, then all streams merge in the fusion phase.

### 7.2 Master Parallel Retrieval Dispatch

```
ALGORITHM: MIMORetrievalDispatch
INPUT:  embedded_subqueries: List<EmbeddedSubQuery>, scope: UserScope,
        session_ctx: SessionContext, memory_summary: MemorySummary,
        deadline: Timestamp, retrieval_bias: RetrievalBias
OUTPUT: stream_results: MIMOStreamResults

1.  per_stream_deadline ← deadline - POST_RETRIEVAL_BUDGET_MS
2.
3.  stream_results ← PARALLEL {
4.
5.      // ═══════════════════════════════════════════════════
6.      // STREAM A: Exact Retrieval (BM25 + Trigram) across all 3 tables
7.      // ═══════════════════════════════════════════════════
8.      exact_results ← ExactRetrievalStream(
9.          embedded_subqueries, scope, per_stream_deadline
10.     )
11.
12.     // ═══════════════════════════════════════════════════
13.     // STREAM B: Semantic Retrieval (Dense + pgvector) across all 3 tables
14.     // ═══════════════════════════════════════════════════
15.     semantic_results ← SemanticRetrievalStream(
16.         embedded_subqueries, scope, per_stream_deadline
17.     )
18.
19.     // ═══════════════════════════════════════════════════
20.     // STREAM C: Historical Utility Retrieval
21.     // ═══════════════════════════════════════════════════
22.     history_results ← HistoricalUtilityStream(
23.         embedded_subqueries, scope, per_stream_deadline
24.     )
25.
26.     // ═══════════════════════════════════════════════════
27.     // STREAM D: Memory-Augmented Retrieval
28.     // ═══════════════════════════════════════════════════
29.     memory_results ← MemoryAugmentedStream(
30.         embedded_subqueries, scope, memory_summary, per_stream_deadline
31.     )
32.
33.     // ═══════════════════════════════════════════════════
34.     // STREAM E: Negative Evidence Collection
35.     // ═══════════════════════════════════════════════════
36.     negative_evidence ← NegativeEvidenceStream(
37.         embedded_subqueries, scope, per_stream_deadline
38.     )
39.
40. } WITH_TIMEOUT(per_stream_deadline)
41.
42. RETURN MIMOStreamResults {
43.     exact: exact_results OR EMPTY_ON_TIMEOUT,
44.     semantic: semantic_results OR EMPTY_ON_TIMEOUT,
45.     history: history_results OR EMPTY_ON_TIMEOUT,
46.     memory: memory_results OR EMPTY_ON_TIMEOUT,
47.     negative: negative_evidence OR EMPTY_ON_TIMEOUT,
48.     streams_completed: COUNT_COMPLETED(),
49.     streams_timed_out: COUNT_TIMED_OUT()
50. }
```

### 7.3 Stream A — Exact Retrieval (BM25 + Trigram)

```
ALGORITHM: ExactRetrievalStream
INPUT:  embedded_subqueries: List<EmbeddedSubQuery>, scope: UserScope,
        deadline: Timestamp
OUTPUT: exact_results: List<ScoredFragment>

1.  exact_results ← EMPTY_LIST
2.
3.  FOR EACH esq IN embedded_subqueries:
4.      IF EXACT NOT IN esq.subquery.retrieval_modes:
5.          CONTINUE
6.
7.      query_terms ← TOKENIZE(esq.subquery.text)
8.      tsquery ← BUILD_TSQUERY(query_terms)  -- PostgreSQL full-text search query
9.      trigram_pattern ← BUILD_TRIGRAM_PATTERN(esq.subquery.text)
10.
11.     target_tables ← esq.subquery.target_tables
12.     temporal_filter ← BUILD_TEMPORAL_FILTER(esq.subquery.temporal_scope)
13.
14.     // Sub-parallel: query across target tables
15.     FOR EACH table IN target_tables:
16.         IF NOW() > deadline: BREAK  -- deadline enforcement
17.
18.         SWITCH table:
19.           CASE session:
20.             results ← QUERY:
21.               SELECT session_id AS chunk_id, content_text, created_at,
22.                      ts_rank_cd(to_tsvector(content_text), :tsquery) AS bm25_score,
23.                      similarity(content_text, :trigram_pattern) AS trigram_score,
24.                      'session' AS source_table, role, turn_index
25.               FROM session
26.               WHERE uu_id = :scope.uu_id
27.                 AND to_tsvector(content_text) @@ :tsquery
28.                 {temporal_filter}
29.               ORDER BY bm25_score DESC
30.               LIMIT k_exact_per_table  -- typically 30
31.
32.           CASE history:
33.             results ← QUERY:
34.               SELECT history_id AS chunk_id, query_text AS content_text,
35.                      response_summary, created_at, utility_score,
36.                      ts_rank_cd(to_tsvector(query_text || ' ' || COALESCE(response_summary,'')), :tsquery) AS bm25_score,
37.                      similarity(query_text, :trigram_pattern) AS trigram_score,
38.                      'history' AS source_table, task_outcome
39.               FROM history
40.               WHERE uu_id = :scope.uu_id
41.                 AND to_tsvector(query_text || ' ' || COALESCE(response_summary,'')) @@ :tsquery
42.                 AND negative_flag = FALSE
43.                 {temporal_filter}
44.               ORDER BY bm25_score DESC
45.               LIMIT k_exact_per_table
46.
47.           CASE memory:
48.             results ← QUERY:
49.               SELECT memory_id AS chunk_id, content_text, updated_at AS created_at,
50.                      authority_tier, confidence,
51.                      ts_rank_cd(to_tsvector(content_text), :tsquery) AS bm25_score,
52.                      similarity(content_text, :trigram_pattern) AS trigram_score,
53.                      'memory' AS source_table, memory_type
54.               FROM memory
55.               WHERE uu_id = :scope.uu_id
56.                 AND to_tsvector(content_text) @@ :tsquery
57.                 AND (expires_at IS NULL OR expires_at > NOW())
58.               ORDER BY bm25_score DESC
59.               LIMIT k_exact_per_table
60.
61.         // Normalize and combine BM25 + trigram scores
62.         FOR EACH r IN results:
63.             combined_exact_score ← α_bm25 * r.bm25_score + α_trigram * r.trigram_score
64.             // α_bm25 = 0.7, α_trigram = 0.3 (BM25 dominates for full matches, trigram catches partial)
65.             fragment ← ScoredFragment {
66.                 chunk_id: r.chunk_id,
67.                 content: r.content_text,
68.                 score: combined_exact_score,
69.                 source_table: r.source_table,
70.                 retrieval_mode: EXACT,
71.                 created_at: r.created_at,
72.                 metadata: EXTRACT_METADATA(r),
73.                 subquery_origin: esq.subquery.text
74.             }
75.             APPEND fragment TO exact_results
76.
77. // Deduplicate by chunk_id, keeping highest score
78. exact_results ← DEDUPLICATE_BY(exact_results, key=chunk_id, keep=MAX_SCORE)
79.
80. RETURN exact_results
```

### 7.4 Stream B — Semantic Retrieval (Dense + pgvector)

```
ALGORITHM: SemanticRetrievalStream
INPUT:  embedded_subqueries: List<EmbeddedSubQuery>, scope: UserScope,
        deadline: Timestamp
OUTPUT: semantic_results: List<ScoredFragment>

1.  semantic_results ← EMPTY_LIST
2.
3.  FOR EACH esq IN embedded_subqueries:
4.      IF SEMANTIC NOT IN esq.subquery.retrieval_modes:
5.          CONTINUE
6.
7.      query_vec ← esq.combined_embedding
8.      target_tables ← esq.subquery.target_tables
9.      temporal_filter ← BUILD_TEMPORAL_FILTER(esq.subquery.temporal_scope)
10.
11.     FOR EACH table IN target_tables:
12.         IF NOW() > deadline: BREAK
13.
14.         SWITCH table:
15.           CASE session:
16.             results ← QUERY:
17.               SELECT session_id AS chunk_id, content_text,
18.                      1 - (content_embedding <=> :query_vec) AS cosine_sim,
19.                      created_at, role, turn_index,
20.                      'session' AS source_table
21.               FROM session
22.               WHERE uu_id = :scope.uu_id
23.                 {temporal_filter}
24.               ORDER BY content_embedding <=> :query_vec  -- pgvector HNSW index
25.               LIMIT k_semantic_per_table  -- typically 30
26.
27.           CASE history:
28.             results ← QUERY:
29.               SELECT history_id AS chunk_id, query_text AS content_text,
30.                      response_summary,
31.                      1 - (query_embedding <=> :query_vec) AS cosine_sim,
32.                      created_at, utility_score, task_outcome,
33.                      'history' AS source_table
34.               FROM history
35.               WHERE uu_id = :scope.uu_id
36.                 AND negative_flag = FALSE
37.                 {temporal_filter}
38.               ORDER BY query_embedding <=> :query_vec
39.               LIMIT k_semantic_per_table
40.
41.           CASE memory:
42.             results ← QUERY:
43.               SELECT memory_id AS chunk_id, content_text,
44.                      1 - (content_embedding <=> :query_vec) AS cosine_sim,
45.                      updated_at AS created_at, authority_tier, confidence,
46.                      memory_type, 'memory' AS source_table
47.               FROM memory
48.               WHERE uu_id = :scope.uu_id
49.                 AND (expires_at IS NULL OR expires_at > NOW())
50.               ORDER BY content_embedding <=> :query_vec
51.               LIMIT k_semantic_per_table
52.
53.         FOR EACH r IN results:
54.             fragment ← ScoredFragment {
55.                 chunk_id: r.chunk_id,
56.                 content: r.content_text,
57.                 score: r.cosine_sim,
58.                 source_table: r.source_table,
59.                 retrieval_mode: SEMANTIC,
60.                 created_at: r.created_at,
61.                 metadata: EXTRACT_METADATA(r),
62.                 subquery_origin: esq.subquery.text
63.             }
64.             APPEND fragment TO semantic_results
65.
66. semantic_results ← DEDUPLICATE_BY(semantic_results, key=chunk_id, keep=MAX_SCORE)
67.
68. RETURN semantic_results
```

### 7.5 Stream C — Historical Utility Retrieval

```
ALGORITHM: HistoricalUtilityStream
INPUT:  embedded_subqueries: List<EmbeddedSubQuery>, scope: UserScope,
        deadline: Timestamp
OUTPUT: history_boosted: List<ScoredFragment>

1.  history_boosted ← EMPTY_LIST
2.
3.  FOR EACH esq IN embedded_subqueries:
4.      query_vec ← esq.combined_embedding
5.
6.      // Find past queries similar to current subquery that led to successful outcomes
7.      similar_past ← QUERY:
8.          SELECT history_id, query_text, response_summary,
9.                 evidence_used,  -- JSONB array of {chunk_id, source, utility_score}
10.                utility_score, task_outcome, human_feedback,
11.                1 - (query_embedding <=> :query_vec) AS query_similarity,
12.                created_at
13.         FROM history
14.         WHERE uu_id = :scope.uu_id
15.           AND task_outcome IN ('success', 'partial')
16.           AND utility_score > utility_threshold  -- typically 0.5
17.           AND 1 - (query_embedding <=> :query_vec) > sim_threshold  -- typically 0.6
18.         ORDER BY query_embedding <=> :query_vec
19.         LIMIT k_history_queries  -- typically 20
20.
21.     // For each similar successful past query, extract which evidence was useful
22.     FOR EACH past IN similar_past:
23.         sim ← past.query_similarity
24.         evidence_list ← PARSE_JSONB(past.evidence_used)
25.
26.         FOR EACH evidence_item IN evidence_list:
27.             IF evidence_item.utility_score > 0.3:
28.                 // Compute historical utility score
29.                 // Weighted by: (1) query similarity, (2) utility, (3) outcome quality, (4) recency
30.                 outcome_weight ← CASE past.task_outcome
31.                     WHEN 'success': 1.0
32.                     WHEN 'partial': 0.6
33.                     ELSE: 0.0
34.                 feedback_bonus ← CASE past.human_feedback
35.                     WHEN 'approved': 0.2
36.                     WHEN 'corrected': 0.1  -- partially useful
37.                     ELSE: 0.0
38.                 recency_decay ← exp(-(NOW() - past.created_at) / τ_history)
39.                     // τ_history typically 30 days
40.
41.                 hist_score ← sim * evidence_item.utility_score
42.                            * outcome_weight * recency_decay
43.                            + feedback_bonus
44.
45.                 // Look up the actual evidence content if still available
46.                 evidence_content ← RESOLVE_EVIDENCE(
47.                     evidence_item.chunk_id, evidence_item.source
48.                 )
49.                 IF evidence_content IS NOT NULL:
50.                     fragment ← ScoredFragment {
51.                         chunk_id: evidence_item.chunk_id,
52.                         content: evidence_content,
53.                         score: hist_score,
54.                         source_table: 'history_derived',
55.                         retrieval_mode: HISTORICAL_UTILITY,
56.                         created_at: past.created_at,
57.                         metadata: {
58.                             original_query: past.query_text,
59.                             original_outcome: past.task_outcome,
60.                             times_useful: evidence_item.use_count
61.                         },
62.                         subquery_origin: esq.subquery.text
63.                     }
64.                     APPEND fragment TO history_boosted
65.
66. // Aggregate: if same chunk_id appears multiple times, sum scores (proven useful across multiple past queries)
67. history_boosted ← AGGREGATE_BY(
68.     history_boosted, key=chunk_id,
69.     aggregation=λ fragments: ScoredFragment {
70.         score: SUM(f.score for f in fragments),
71.         content: fragments[0].content,  -- identical content
72.         metadata: MERGE(f.metadata for f in fragments),
73.         times_proven_useful: |fragments|
74.     }
75. )
76.
77. RETURN history_boosted
```

**Above-SOTA innovation**: This stream performs **transitive utility propagation** — if document $d$ was useful for past query $q_1$ which is similar to current query $q$, then $d$ receives a utility boost proportional to $\text{sim}(q, q_1) \times \text{utility}(d, q_1) \times \text{outcome}(q_1) \times \text{recency}(q_1)$. When the same document is proven useful across multiple similar past queries, scores are aggregated, creating a strong signal that conventional RAG entirely misses.

### 7.6 Stream D — Memory-Augmented Retrieval

```
ALGORITHM: MemoryAugmentedStream
INPUT:  embedded_subqueries: List<EmbeddedSubQuery>, scope: UserScope,
        memory_summary: MemorySummary, deadline: Timestamp
OUTPUT: memory_results: List<ScoredFragment>

1.  memory_results ← EMPTY_LIST
2.
3.  FOR EACH esq IN embedded_subqueries:
4.      query_vec ← esq.combined_embedding
5.
6.      // Step 1: Direct memory retrieval (semantic + exact hybrid on memory table)
7.      semantic_mem ← QUERY:
8.          SELECT memory_id AS chunk_id, content_text,
9.                 1 - (content_embedding <=> :query_vec) AS cosine_sim,
10.                authority_tier, confidence, memory_type,
11.                updated_at, access_count,
12.                'memory' AS source_table
13.         FROM memory
14.         WHERE uu_id = :scope.uu_id
15.           AND (expires_at IS NULL OR expires_at > NOW())
16.         ORDER BY content_embedding <=> :query_vec
17.         LIMIT k_memory_semantic  -- typically 20
18.
19.     exact_mem ← QUERY:
20.         SELECT memory_id AS chunk_id, content_text,
21.                ts_rank_cd(to_tsvector(content_text), plainto_tsquery(:query_text)) AS bm25_score,
22.                authority_tier, confidence, memory_type,
23.                updated_at, access_count,
24.                'memory' AS source_table
25.         FROM memory
26.         WHERE uu_id = :scope.uu_id
27.           AND to_tsvector(content_text) @@ plainto_tsquery(:esq.subquery.text)
28.           AND (expires_at IS NULL OR expires_at > NOW())
29.         ORDER BY bm25_score DESC
30.         LIMIT k_memory_exact  -- typically 15
31.
32.     // Step 2: Score with authority and confidence boosting
33.     FOR EACH r IN UNION(semantic_mem, exact_mem):
34.         base_score ← CASE r FROM semantic_mem: r.cosine_sim
35.                       CASE r FROM exact_mem: NORMALIZE(r.bm25_score)
36.
37.         // Authority boost
38.         authority_multiplier ← CASE r.authority_tier
39.             WHEN 'canonical': 1.5
40.             WHEN 'curated':   1.2
41.             WHEN 'derived':   1.0
42.             WHEN 'ephemeral': 0.7
43.
44.         // Confidence integration
45.         confidence_factor ← 0.5 + 0.5 * r.confidence  -- maps [0,1] → [0.5, 1.0]
46.
47.         // Freshness decay
48.         freshness ← exp(-(NOW() - r.updated_at) / τ_memory)
49.             // τ_memory = 90 days for long-term memory
50.
51.         // Usage frequency boost (more accessed = more relevant)
52.         usage_boost ← log(1 + r.access_count) / log(1 + max_access_count)
53.
54.         // Correction/constraint type priority boost
55.         type_boost ← CASE r.memory_type
56.             WHEN 'correction':  1.3  -- corrections are high-value
57.             WHEN 'constraint':  1.2  -- constraints must be respected
58.             WHEN 'fact':        1.0
59.             WHEN 'preference':  0.9
60.             WHEN 'procedure':   1.1
61.             WHEN 'schema':      1.0
62.
63.         final_score ← base_score * authority_multiplier * confidence_factor
64.                      * freshness * (1 + 0.1 * usage_boost) * type_boost
65.
66.         fragment ← ScoredFragment {
67.             chunk_id: r.chunk_id,
68.             content: r.content_text,
69.             score: final_score,
70.             source_table: 'memory',
71.             retrieval_mode: MEMORY_AUGMENTED,
72.             created_at: r.updated_at,
73.             metadata: {
74.                 authority_tier: r.authority_tier,
75.                 confidence: r.confidence,
76.                 memory_type: r.memory_type,
77.                 access_count: r.access_count
78.             },
79.             subquery_origin: esq.subquery.text
80.         }
81.         APPEND fragment TO memory_results
82.
83.     // Step 3: Update access counts for retrieved memories
84.     ASYNC: QUERY:
85.         UPDATE memory
86.         SET access_count = access_count + 1,
87.             last_accessed_at = NOW()
88.         WHERE memory_id IN (SELECT chunk_id FROM memory_results WHERE source_table = 'memory')
89.           AND uu_id = :scope.uu_id
90.
91. memory_results ← DEDUPLICATE_BY(memory_results, key=chunk_id, keep=MAX_SCORE)
92.
93. RETURN memory_results
```

### 7.7 Stream E — Negative Evidence Collection

```
ALGORITHM: NegativeEvidenceStream
INPUT:  embedded_subqueries: List<EmbeddedSubQuery>, scope: UserScope,
        deadline: Timestamp
OUTPUT: negative_set: Set<ChunkID>

1.  negative_set ← EMPTY_SET
2.
3.  FOR EACH esq IN embedded_subqueries:
4.      query_vec ← esq.combined_embedding
5.
6.      // Find past queries similar to current that FAILED
7.      failed_past ← QUERY:
8.          SELECT history_id, evidence_used
9.          FROM history
10.         WHERE uu_id = :scope.uu_id
11.           AND (task_outcome = 'failure' OR human_feedback = 'rejected')
12.           AND negative_flag = TRUE
13.           AND 1 - (query_embedding <=> :query_vec) > neg_sim_threshold  -- typically 0.7
14.         ORDER BY query_embedding <=> :query_vec
15.         LIMIT k_negative  -- typically 10
16.
17.     FOR EACH past IN failed_past:
18.         evidence_list ← PARSE_JSONB(past.evidence_used)
19.         FOR EACH evidence_item IN evidence_list:
20.             IF evidence_item.utility_score < neg_utility_threshold:  -- typically 0.2
21.                 ADD evidence_item.chunk_id TO negative_set
22.
23. RETURN negative_set
```

**Purpose**: The negative evidence set is used in the fusion phase to **suppress or down-weight** documents that have historically led to failures for similar queries. This is a critical above-SOTA mechanism: conventional retrieval systems have no concept of "what not to retrieve."

---

## 8. Phase 4 — Multi-Signal Fusion

### 8.1 Adaptive Reciprocal Rank Fusion with Query-Type Weighting

```
ALGORITHM: AdaptiveMultiStreamFusion
INPUT:  stream_results: MIMOStreamResults,
        query_class: QueryClassification,
        retrieval_bias: RetrievalBias,
        negative_set: Set<ChunkID>
OUTPUT: fused_candidates: List<FusedCandidate>

1.  // Define per-stream weight based on query classification
2.  stream_weights ← {
3.      exact:    retrieval_bias.bm25_weight,     -- e.g., 0.35
4.      semantic: retrieval_bias.dense_weight,     -- e.g., 0.35
5.      history:  retrieval_bias.history_weight,   -- e.g., 0.15
6.      memory:   retrieval_bias.memory_weight     -- e.g., 0.15
7.  }
8.
9.  // Normalize weights to sum to 1
10. total ← SUM(stream_weights.values())
11. FOR EACH key IN stream_weights:
12.     stream_weights[key] ← stream_weights[key] / total
13.
14. // Build ranked lists per stream
15. ranked_lists ← {
16.     exact:    SORT(stream_results.exact, BY score DESC),
17.     semantic: SORT(stream_results.semantic, BY score DESC),
18.     history:  SORT(stream_results.history, BY score DESC),
19.     memory:   SORT(stream_results.memory, BY score DESC)
20. }
21.
22. // Weighted Reciprocal Rank Fusion
23. κ ← 60  -- RRF smoothing constant
24. score_map ← EMPTY_MAP<ChunkID, FusionAccumulator>
25.
26. FOR EACH (stream_name, ranked_list) IN ranked_lists:
27.     w ← stream_weights[stream_name]
28.     FOR rank = 1 TO |ranked_list|:
29.         chunk_id ← ranked_list[rank].chunk_id
30.         rrf_contribution ← w / (κ + rank)
31.
32.         IF chunk_id NOT IN score_map:
33.             score_map[chunk_id] ← FusionAccumulator {
34.                 rrf_score: 0.0,
35.                 source_streams: EMPTY_SET,
36.                 best_content: NULL,
37.                 per_stream_ranks: EMPTY_MAP,
38.                 per_stream_scores: EMPTY_MAP,
39.                 metadata_union: EMPTY_MAP
40.             }
41.
42.         score_map[chunk_id].rrf_score += rrf_contribution
43.         score_map[chunk_id].source_streams.add(stream_name)
44.         score_map[chunk_id].per_stream_ranks[stream_name] ← rank
45.         score_map[chunk_id].per_stream_scores[stream_name] ← ranked_list[rank].score
46.         IF score_map[chunk_id].best_content IS NULL:
47.             score_map[chunk_id].best_content ← ranked_list[rank].content
48.         score_map[chunk_id].metadata_union ← MERGE(
49.             score_map[chunk_id].metadata_union,
50.             ranked_list[rank].metadata
51.         )
52.
53. // ═══════════════════════════════════════════════════
54. // ABOVE-SOTA: Multi-stream agreement bonus
55. // Documents appearing in multiple streams are more likely relevant
56. // ═══════════════════════════════════════════════════
57. FOR EACH (chunk_id, acc) IN score_map:
58.     stream_count ← |acc.source_streams|
59.     // Agreement bonus: geometric scaling for multi-stream presence
60.     agreement_bonus ← (stream_count / |ranked_lists|) ^ 0.5
61.     // Penalty: if stream_count = 1 and it's only from history, reduce confidence
62.     single_stream_penalty ← IF stream_count == 1 AND 'history' IN acc.source_streams:
63.                                0.8  ELSE: 1.0
64.     acc.rrf_score ← acc.rrf_score * (1 + 0.3 * agreement_bonus) * single_stream_penalty
65.
66. // ═══════════════════════════════════════════════════
67. // ABOVE-SOTA: Negative evidence suppression
68. // ═══════════════════════════════════════════════════
69. FOR EACH chunk_id IN negative_set:
70.     IF chunk_id IN score_map:
71.         score_map[chunk_id].rrf_score *= negative_penalty  -- typically 0.1
72.         score_map[chunk_id].metadata_union['negative_flag'] ← TRUE
73.
74. // ═══════════════════════════════════════════════════
75. // Source-table-aware authority boosting
76. // ═══════════════════════════════════════════════════
77. FOR EACH (chunk_id, acc) IN score_map:
78.     IF 'memory' IN acc.source_streams:
79.         auth_tier ← acc.metadata_union.get('authority_tier', 'derived')
80.         auth_boost ← CASE auth_tier
81.             WHEN 'canonical': 1.3
82.             WHEN 'curated':   1.15
83.             WHEN 'derived':   1.0
84.             ELSE: 0.9
85.         acc.rrf_score *= auth_boost
86.
87.     // Session recency boost for conversational queries
88.     IF 'session' IN acc.source_streams AND query_class.type == CONVERSATIONAL:
89.         acc.rrf_score *= 1.2
90.
91. // Sort and build fused candidate list
92. fused_candidates ← EMPTY_LIST
93. FOR EACH (chunk_id, acc) IN SORT(score_map, BY acc.rrf_score DESC):
94.     APPEND FusedCandidate {
95.         chunk_id: chunk_id,
96.         content: acc.best_content,
97.         fused_score: acc.rrf_score,
98.         source_streams: acc.source_streams,
99.         per_stream_ranks: acc.per_stream_ranks,
100.        per_stream_scores: acc.per_stream_scores,
101.        metadata: acc.metadata_union,
102.        stream_agreement: |acc.source_streams|
103.    } TO fused_candidates
104.
105. RETURN fused_candidates
```

**Key innovation — Multi-stream agreement bonus**: When a document appears in multiple independent retrieval streams (e.g., both BM25 and semantic find it, AND it was historically useful), the agreement provides strong evidence of relevance. The bonus is computed as:

$$\text{bonus}(d) = \left(\frac{|\text{streams}(d)|}{m}\right)^{0.5}$$

where $m$ is the total number of streams. The square root prevents over-rewarding documents that trivially appear everywhere (e.g., very common documents). This is mathematically grounded: under independence assumptions, the probability that $k$ independent retrievers all find a non-relevant document decreases exponentially with $k$.

---

## 9. Phase 5 — Cross-Encoder Reranking via vLLM

```
ALGORITHM: CrossEncoderReranking
INPUT:  fused_candidates: List<FusedCandidate>, query: string,
        vllm_endpoint: Endpoint, k_rerank: int, k_final: int,
        deadline: Timestamp
OUTPUT: reranked: List<RankedFragment>

1.  // Take top-k_rerank candidates for cross-encoder scoring
2.  // k_rerank >> k_final (typically k_rerank = 50-100, k_final = 10-20)
3.  candidates_to_rerank ← fused_candidates[0 : k_rerank]
4.
5.  // ═══════════════════════════════════════════════════
6.  // Construct cross-encoder input pairs
7.  // Format compatible with vLLM serving of cross-encoder models
8.  // ═══════════════════════════════════════════════════
9.  pairs ← EMPTY_LIST
10. FOR EACH candidate IN candidates_to_rerank:
11.     pair_text ← FORMAT_CROSS_ENCODER_INPUT(
12.         query: query,
13.         document: candidate.content,
14.         metadata_hint: COMPRESS_METADATA(candidate.metadata)
15.     )
16.     APPEND pair_text TO pairs
17.
18. // ═══════════════════════════════════════════════════
19. // Batch inference via vLLM (local, low-latency)
20. // Uses openai-compatible API served by vLLM
21. // ═══════════════════════════════════════════════════
22. TRY:
23.     cross_scores ← VLLM_BATCH_SCORE(
24.         endpoint: vllm_endpoint,
25.         model: "cross-encoder-model",  -- e.g., bge-reranker-v2-m3 or custom
26.         pairs: pairs,
27.         timeout: remaining_time(deadline) * 0.9
28.     )
29.     // cross_scores: List<float> in [0, 1], one per pair
30.
31. ON_TIMEOUT OR ON_ERROR:
32.     // Graceful degradation: use fused scores as-is (skip reranking)
33.     LOG_WARNING("Cross-encoder reranking timed out; using fused scores")
34.     cross_scores ← [c.fused_score for c in candidates_to_rerank]
35.
36. // ═══════════════════════════════════════════════════
37. // Combine fused score with cross-encoder score
38. // Cross-encoder is higher precision; it dominates the final score
39. // ═══════════════════════════════════════════════════
40. reranked ← EMPTY_LIST
41. FOR i = 0 TO |candidates_to_rerank| - 1:
42.     combined ← β_cross * cross_scores[i] + β_fused * candidates_to_rerank[i].fused_score
43.     // β_cross = 0.7, β_fused = 0.3 (cross-encoder dominates)
44.
45.     // Additional boost for multi-stream agreement even after reranking
46.     agreement_persistence ← 1 + 0.05 * candidates_to_rerank[i].stream_agreement
47.     combined ← combined * agreement_persistence
48.
49.     APPEND RankedFragment {
50.         chunk_id: candidates_to_rerank[i].chunk_id,
51.         content: candidates_to_rerank[i].content,
52.         final_score: combined,
53.         cross_encoder_score: cross_scores[i],
54.         fused_score: candidates_to_rerank[i].fused_score,
55.         source_streams: candidates_to_rerank[i].source_streams,
56.         metadata: candidates_to_rerank[i].metadata
57.     } TO reranked
58.
59. SORT reranked BY final_score DESCENDING
60.
61. RETURN reranked
```

**vLLM integration note**: The cross-encoder model is served via vLLM's OpenAI-compatible API endpoint. The batch scoring call uses the `/v1/completions` or a custom `/v1/rerank` endpoint. Input format follows the model's expected `[query] [SEP] [document]` convention. vLLM provides batched GPU inference with continuous batching, achieving sub-50ms latency for 50–100 pairs on a single GPU.

---

## 10. Phase 6 — MMR Diversity Selection

```
ALGORITHM: MMRDiversitySelection
INPUT:  reranked: List<RankedFragment>, k_final: int,
        lambda: float, embeddings_cache: Map<ChunkID, Vector>
OUTPUT: diverse_set: List<RankedFragment>

1.  // lambda ∈ [0, 1]: relevance-diversity trade-off
2.  // Typical: lambda = 0.6 (slight relevance bias)
3.
4.  S ← EMPTY_LIST   // Selected set
5.  C ← SET(reranked) // Candidate set
6.
7.  // Ensure embeddings are available for all candidates
8.  FOR EACH candidate IN C:
9.      IF candidate.chunk_id NOT IN embeddings_cache:
10.         embeddings_cache[candidate.chunk_id] ← OPENAI_EMBED(
11.             model: embedding_model_name,
12.             input: candidate.content
13.         )
14.
15. WHILE |S| < k_final AND C IS NOT EMPTY:
16.     best ← NULL
17.     best_mmr ← -∞
18.
19.     FOR EACH candidate IN C:
20.         rel ← candidate.final_score
21.
22.         IF S IS EMPTY:
23.             max_sim_to_selected ← 0.0
24.         ELSE:
25.             max_sim_to_selected ← MAX(
26.                 COSINE_SIM(
27.                     embeddings_cache[candidate.chunk_id],
28.                     embeddings_cache[s.chunk_id]
29.                 )
30.                 FOR s IN S
31.             )
32.
33.         mmr ← lambda * NORMALIZE(rel) - (1 - lambda) * max_sim_to_selected
34.
35.         // ═══════════════════════════════════════════════════
36.         // ABOVE-SOTA: Source diversity bonus
37.         // Prefer candidates from different source tables
38.         // ═══════════════════════════════════════════════════
39.         source_tables_in_S ← {s.metadata.source_table FOR s IN S}
40.         candidate_source ← candidate.metadata.get('source_table', 'unknown')
41.         IF candidate_source NOT IN source_tables_in_S:
42.             mmr += source_diversity_bonus  -- typically 0.05
43.
44.         IF mmr > best_mmr:
45.             best_mmr ← mmr
46.             best ← candidate
47.
48.     APPEND best TO S
49.     REMOVE best FROM C
50.
51. RETURN S
```

**Above-SOTA innovation — Source diversity bonus**: Standard MMR only considers content similarity for diversity. This algorithm adds a **source-table diversity bonus** that ensures the final evidence set includes fragments from session, history, AND memory tables when possible. This provides the agent with complementary evidence types: recent conversational context (session), proven-useful patterns (history), and validated long-term knowledge (memory).

---

## 11. Phase 7 — Provenance Assembly and Token Budget Fitting

### 11.1 Provenance Assembly

```
ALGORITHM: ProvenanceAssembly
INPUT:  diverse_set: List<RankedFragment>, scope: UserScope
OUTPUT: provenance_tagged: List<ProvenanceTaggedFragment>

1.  provenance_tagged ← EMPTY_LIST
2.
3.  FOR EACH fragment IN diverse_set:
4.      provenance ← ProvenanceRecord {
5.          origin: {
6.              source_table: fragment.metadata.source_table,
7.              source_id: scope.uu_id + ":" + fragment.metadata.source_table,
8.              chunk_id: fragment.chunk_id,
9.              extraction_timestamp: fragment.metadata.created_at,
10.             retrieval_method: DESCRIBE_RETRIEVAL_PATH(fragment)
11.         },
12.         scoring: {
13.             cross_encoder_score: fragment.cross_encoder_score,
14.             fused_score: fragment.fused_score,
15.             final_score: fragment.final_score,
16.             source_streams: fragment.source_streams,
17.             stream_agreement: |fragment.source_streams|
18.         },
19.         trust: {
20.             authority_tier: fragment.metadata.get('authority_tier', 'derived'),
21.             confidence: fragment.metadata.get('confidence', 0.5),
22.             is_memory_validated: fragment.metadata.source_table == 'memory',
23.             is_historically_proven: 'history' IN fragment.source_streams,
24.             negative_flag: fragment.metadata.get('negative_flag', FALSE),
25.             freshness: NOW() - fragment.metadata.created_at
26.         },
27.         custody_hash: COMPUTE_CUSTODY_HASH(fragment)
28.     }
29.
30.     APPEND ProvenanceTaggedFragment {
31.         content: fragment.content,
32.         chunk_id: fragment.chunk_id,
33.         final_score: fragment.final_score,
34.         provenance: provenance,
35.         token_count: COUNT_TOKENS(fragment.content)
36.     } TO provenance_tagged
37.
38. RETURN provenance_tagged
```

### 11.2 Token Budget Fitting

```
ALGORITHM: TokenBudgetFitting
INPUT:  provenance_tagged: List<ProvenanceTaggedFragment>,
        token_budget: int,
        provenance_overhead_per_fragment: int  -- typically 30-50 tokens
OUTPUT: fitted: List<ProvenanceTaggedFragment>, budget_report: BudgetReport

1.  // Fragments are already sorted by final_score descending (from MMR)
2.  fitted ← EMPTY_LIST
3.  tokens_used ← 0
4.  tokens_wasted ← 0
5.
6.  FOR EACH fragment IN provenance_tagged:
7.      fragment_cost ← fragment.token_count + provenance_overhead_per_fragment
8.
9.      IF tokens_used + fragment_cost ≤ token_budget:
10.         APPEND fragment TO fitted
11.         tokens_used += fragment_cost
12.     ELSE:
13.         // Check if truncation is worthwhile
14.         remaining_budget ← token_budget - tokens_used
15.         IF remaining_budget > min_useful_tokens:  -- typically 50
16.             // Truncate fragment to fit remaining budget
17.             truncated_content ← TRUNCATE_AT_SENTENCE_BOUNDARY(
18.                 fragment.content,
19.                 max_tokens=remaining_budget - provenance_overhead_per_fragment
20.             )
21.             IF token_count(truncated_content) > min_useful_tokens:
22.                 fragment.content ← truncated_content
23.                 fragment.token_count ← token_count(truncated_content)
24.                 fragment.provenance.scoring.truncated ← TRUE
25.                 APPEND fragment TO fitted
26.                 tokens_used += fragment.token_count + provenance_overhead_per_fragment
27.         BREAK  -- budget exhausted
28.
29. budget_report ← BudgetReport {
30.     budget_total: token_budget,
31.     budget_used: tokens_used,
32.     budget_remaining: token_budget - tokens_used,
33.     utilization: tokens_used / token_budget,
34.     fragments_included: |fitted|,
35.     fragments_excluded: |provenance_tagged| - |fitted|,
36.     truncation_applied: ANY(f.provenance.scoring.truncated FOR f IN fitted)
37. }
38.
39. RETURN (fitted, budget_report)
```

**Design decision**: Token budget fitting uses a **greedy allocation by score rank**, which is optimal under the assumption that fragments are independently valued and ranked correctly. Truncation is performed at sentence boundaries to preserve semantic coherence.

---

## 12. Phase 8 — Progressive Retrieval Quality Gate

```
ALGORITHM: ProgressiveRetrievalQualityGate
INPUT:  fitted: List<ProvenanceTaggedFragment>, query: string,
        quality_config: QualityConfig, max_iterations: int
OUTPUT: final_evidence: List<ProvenanceTaggedFragment>

1.  // ═══════════════════════════════════════════════════
2.  // ABOVE-SOTA: Iterative retrieval refinement
3.  // If initial retrieval quality is insufficient, refine and re-retrieve
4.  // ═══════════════════════════════════════════════════
5.
6.  current_evidence ← fitted
7.  iteration ← 0
8.
9.  WHILE iteration < max_iterations:  -- typically max 2 iterations
10.     iteration += 1
11.
12.     // Quality assessment
13.     quality ← AssessRetrievalQuality(current_evidence, query)
14.
15.     IF quality.score ≥ quality_config.min_quality_threshold:  -- typically 0.7
16.         BREAK  -- quality sufficient
17.
18.     IF quality.issue == LOW_RELEVANCE:
19.         // Re-retrieve with expanded query
20.         expanded_query ← VLLM_INFERENCE(
21.             model: "query-expander",
22.             prompt: "The following query did not retrieve sufficient relevant results. "
23.                   + "Rephrase and expand it to capture more relevant documents: "
24.                   + query,
25.             max_tokens: 150,
26.             temperature: 0.3
27.         )
28.         // Re-run retrieval pipeline from Phase 3 with expanded query
29.         // (recursive call with depth bound)
30.         supplementary ← RETRIEVE_WITH_QUERY(expanded_query, depth=iteration)
31.         current_evidence ← MERGE_AND_RERANK(current_evidence, supplementary)
32.
33.     ELSE IF quality.issue == LOW_DIVERSITY:
34.         // Decrease lambda in MMR to force more diversity
35.         current_evidence ← MMRDiversitySelection(
36.             reranked, k_final, lambda=lambda * 0.7, embeddings_cache
37.         )
38.
39.     ELSE IF quality.issue == STALE_EVIDENCE:
40.         // Re-retrieve with freshness constraint tightened
41.         current_evidence ← RETRIEVE_WITH_FRESHNESS_BOOST(query, freshness_weight=0.4)
42.
43. final_evidence ← current_evidence
44. RETURN final_evidence

FUNCTION: AssessRetrievalQuality(evidence, query) → QualityAssessment
1.  // Fast heuristic assessment (no model call)
2.  avg_score ← MEAN(e.final_score FOR e IN evidence)
3.  score_variance ← VARIANCE(e.final_score FOR e IN evidence)
4.  source_diversity ← |UNIQUE(e.provenance.origin.source_table FOR e IN evidence)| / 3
5.  max_staleness ← MAX(e.provenance.trust.freshness FOR e IN evidence)
6.  min_confidence ← MIN(e.provenance.trust.confidence FOR e IN evidence)
7.
8.  quality_score ← 0.4 * avg_score + 0.2 * source_diversity
9.                 + 0.2 * (1 - min(max_staleness / τ_staleness_limit, 1.0))
10.                + 0.2 * min_confidence
11.
12. issue ← NULL
13. IF avg_score < 0.3: issue ← LOW_RELEVANCE
14. ELSE IF source_diversity < 0.33: issue ← LOW_DIVERSITY
15. ELSE IF max_staleness > τ_staleness_limit: issue ← STALE_EVIDENCE
16.
17. RETURN QualityAssessment { score: quality_score, issue: issue }
```

---

## 13. End-to-End Master Orchestration Algorithm

```
ALGORITHM: MasterRetrievalOrchestrator
INPUT:
    raw_query: string,
    uu_id: UUID,
    session_id: UUID,
    token_budget: int,              -- max tokens for evidence
    latency_deadline_ms: int,       -- hard deadline (default 250ms)
    ranking_config: RankingConfig,  -- weights, diversity lambda, etc.
OUTPUT:
    evidence_response: EvidenceResponse

// ═══════════════════════════════════════════════════════════
// PHASE 0: USER SCOPE LOCK (0ms)
// ═══════════════════════════════════════════════════════════
1.  t_start ← NOW()
2.  deadline ← t_start + latency_deadline_ms
3.  scope ← UserScopeLock(uu_id)

// ═══════════════════════════════════════════════════════════
// PHASE 1: PARALLEL CONTEXT LOAD (≤5ms)
// ═══════════════════════════════════════════════════════════
4.  (session_ctx, memory_summary) ← ParallelContextLoad(scope, session_id)

// ═══════════════════════════════════════════════════════════
// PHASE 2: QUERY PREPROCESSING (≤25ms)
// ═══════════════════════════════════════════════════════════
5.  query_class ← QueryClassifier(raw_query, session_ctx)
6.  augmented_query ← ContextualQueryAugmentation(
        raw_query, query_class, session_ctx, memory_summary
    )
7.  subqueries ← QueryDecomposer(augmented_query)
8.  embedded_subqueries ← ContextualEmbeddingGenerator(
        subqueries, session_ctx, memory_summary
    )

// ═══════════════════════════════════════════════════════════
// PHASE 3: PARALLEL MIMO RETRIEVAL (≤150ms)
// ═══════════════════════════════════════════════════════════
9.  stream_results ← MIMORetrievalDispatch(
        embedded_subqueries, scope, session_ctx, memory_summary,
        deadline, query_class.retrieval_bias
    )

// ═══════════════════════════════════════════════════════════
// PHASE 4: MULTI-SIGNAL FUSION (≤10ms)
// ═══════════════════════════════════════════════════════════
10. fused_candidates ← AdaptiveMultiStreamFusion(
        stream_results, query_class, query_class.retrieval_bias,
        stream_results.negative
    )

// ═══════════════════════════════════════════════════════════
// PHASE 5: CROSS-ENCODER RERANKING (≤60ms)
// ═══════════════════════════════════════════════════════════
11. reranked ← CrossEncoderReranking(
        fused_candidates, augmented_query.resolved,
        vllm_endpoint, k_rerank=80, k_final=20, deadline
    )

// ═══════════════════════════════════════════════════════════
// PHASE 6: MMR DIVERSITY SELECTION (≤5ms)
// ═══════════════════════════════════════════════════════════
12. diverse_set ← MMRDiversitySelection(
        reranked, k_final=ranking_config.k_final,
        lambda=ranking_config.diversity_lambda,
        embeddings_cache
    )

// ═══════════════════════════════════════════════════════════
// PHASE 7: PROVENANCE + TOKEN BUDGET (≤7ms)
// ═══════════════════════════════════════════════════════════
13. provenance_tagged ← ProvenanceAssembly(diverse_set, scope)
14. (fitted, budget_report) ← TokenBudgetFitting(
        provenance_tagged, token_budget
    )

// ═══════════════════════════════════════════════════════════
// PHASE 8: PROGRESSIVE QUALITY GATE (≤ remaining budget)
// ═══════════════════════════════════════════════════════════
15. IF remaining_time(deadline) > MIN_REFINEMENT_BUDGET_MS:
16.     final_evidence ← ProgressiveRetrievalQualityGate(
            fitted, augmented_query.resolved, quality_config, max_iterations=1
        )
17. ELSE:
18.     final_evidence ← fitted

// ═══════════════════════════════════════════════════════════
// PHASE 9: RESPONSE ASSEMBLY + OBSERVABILITY (≤3ms)
// ═══════════════════════════════════════════════════════════
19. evidence_response ← EvidenceResponse {
        fragments: final_evidence,
        total_candidates_considered: |fused_candidates|,
        latency_ms: NOW() - t_start,
        source_coverage: {
            session: COUNT_FROM(final_evidence, 'session'),
            history: COUNT_FROM(final_evidence, 'history'),
            memory:  COUNT_FROM(final_evidence, 'memory')
        },
        budget_report: budget_report,
        quality_assessment: AssessRetrievalQuality(final_evidence, raw_query),
        streams_completed: stream_results.streams_completed,
        streams_timed_out: stream_results.streams_timed_out,
        cache_hit_ratio: COMPUTE_CACHE_HIT_RATIO(),
        retrieval_trace_id: GENERATE_TRACE_ID()
    }

// ═══════════════════════════════════════════════════════════
// ASYNC: Write-through cache + History update
// ═══════════════════════════════════════════════════════════
20. ASYNC {
        // Cache the retrieval result for identical future queries
        RetrievalCache.write(
            key=HASH(uu_id, raw_query, session_id),
            value=evidence_response,
            ttl=COMPUTE_TTL(final_evidence)
        )

        // Emit observability telemetry
        ObservabilityEmitter.emit(
            trace_id=evidence_response.retrieval_trace_id,
            uu_id=uu_id,
            query=raw_query,
            latency=evidence_response.latency_ms,
            precision_estimate=evidence_response.quality_assessment.score,
            fragments_returned=|final_evidence|,
            token_utilization=budget_report.utilization
        )
    }

21. RETURN evidence_response
```

---

## 14. Formal Scoring Mathematics

### 14.1 Composite Scoring Function

The final score for evidence fragment $d$ given query $q$, user context $U$, and task $\mathcal{T}$ is:

$$S(q, d, U, \mathcal{T}) = \underbrace{\beta_c \cdot S_{\text{cross}}(q, d)}_{\text{Cross-Encoder Precision}} + \underbrace{\beta_f \cdot S_{\text{RRF}}^{*}(q, d)}_{\text{Fused Recall}} + \underbrace{\beta_a \cdot \Phi_{\text{agree}}(d)}_{\text{Stream Agreement}} - \underbrace{\beta_n \cdot \mathbb{1}[d \in \mathcal{N}_U]}_{\text{Negative Suppression}}$$

where:

- $S_{\text{cross}}(q, d) \in [0, 1]$ is the cross-encoder relevance score from vLLM
- $S_{\text{RRF}}^{*}(q, d)$ is the weighted RRF score with authority boosting:

$$S_{\text{RRF}}^{*}(q, d) = \left(\sum_{j=1}^{m} \frac{w_j}{\kappa + r_j(d)}\right) \cdot \alpha_{\text{auth}}(d)$$

- $\Phi_{\text{agree}}(d) = \left(\frac{|\text{streams}(d)|}{m}\right)^{0.5}$ is the multi-stream agreement factor
- $\mathcal{N}_U$ is the user's negative evidence set
- $\alpha_{\text{auth}}(d)$ is the authority multiplier:

$$\alpha_{\text{auth}}(d) = \begin{cases} 1.5 & \text{if } d \text{ is canonical memory} \\ 1.2 & \text{if } d \text{ is curated memory} \\ 1.0 & \text{if } d \text{ is derived} \\ 0.7 & \text{if } d \text{ is ephemeral} \end{cases}$$

Default weights: $\beta_c = 0.55$, $\beta_f = 0.25$, $\beta_a = 0.10$, $\beta_n = 0.50$.

### 14.2 Freshness Decay Function

$$\phi_{\text{fresh}}(d, \text{table}) = \exp\left(-\frac{\Delta t(d)}{\tau_{\text{table}}}\right)$$

where $\Delta t(d) = \text{NOW}() - \text{updated\_at}(d)$ and:

| Source Table | $\tau_{\text{table}}$ | Rationale |
|---|---|---|
| `session` | 1 hour | Session context is highly volatile |
| `history` | 30 days | Past utility degrades slowly |
| `memory` | 180 days | Validated knowledge is long-lived |

### 14.3 Historical Utility Score

For a new query $q$ and candidate document $d$ from a user $U$'s history:

$$S_{\text{hist}}(q, d, U) = \frac{\sum_{(q_i, d, u_i) \in \mathcal{H}_U} \text{sim}(q, q_i) \cdot u_i \cdot \omega(q_i) \cdot \rho(q_i)}{\sum_{(q_i, d, u_i) \in \mathcal{H}_U} \text{sim}(q, q_i) + \epsilon}$$

where:
- $\mathcal{H}_U$ is user $U$'s history
- $u_i$ is the utility score of document $d$ for past query $q_i$
- $\omega(q_i)$ is the outcome weight: $\omega = 1.0$ for success, $0.6$ for partial, $0.0$ for failure
- $\rho(q_i) = \exp(-\Delta t(q_i) / \tau_{\text{history}})$ is the recency decay
- $\epsilon = 10^{-8}$ prevents division by zero

### 14.4 Adaptive Fusion Weight Selection

The fusion weights $\mathbf{w} = [w_{\text{bm25}}, w_{\text{dense}}, w_{\text{history}}, w_{\text{memory}}]$ are selected dynamically based on query classification:

$$\mathbf{w}(\mathcal{C}) = \text{softmax}\left(\mathbf{W}_{\text{bias}} \cdot \mathbf{e}(\mathcal{C}) + \mathbf{b}\right)$$

where $\mathbf{e}(\mathcal{C})$ is a one-hot encoding of the query class $\mathcal{C} \in \{\text{FACTUAL}, \text{CONCEPTUAL}, \text{PROCEDURAL}, \text{CODE}, \text{DIAGNOSTIC}, \text{CONVERSATIONAL}\}$, and $\mathbf{W}_{\text{bias}}, \mathbf{b}$ are learned from feedback data or set via the lookup table in §6.1.

---

## 15. Memory Write-Back Protocol

After task completion, the system updates the user's three tables to close the feedback loop:

```
ALGORITHM: PostTaskWriteBack
INPUT:  uu_id: UUID, session_id: UUID, query: string,
        evidence_used: List<ProvenanceTaggedFragment>,
        agent_output: string, task_outcome: TaskOutcome,
        human_feedback: HumanFeedback (optional)
OUTPUT: write_confirmation: WriteConfirmation

1.  // ═══════════════════════════════════════════════════
2.  // HISTORY TABLE: Record the interaction
3.  // ═══════════════════════════════════════════════════
4.  query_embedding ← OPENAI_EMBED(model: embedding_model, input: query)
5.
6.  evidence_record ← []
7.  FOR EACH fragment IN evidence_used:
8.      utility ← EstimateFragmentUtility(fragment, agent_output, task_outcome)
9.      APPEND {
10.         chunk_id: fragment.chunk_id,
11.         source: fragment.provenance.origin.source_table,
12.         utility_score: utility,
13.         was_cited: WAS_CITED(fragment, agent_output),
14.         use_count: 1
15.     } TO evidence_record
16.
17. INSERT INTO history (
18.     history_id, uu_id, query_text, query_embedding,
19.     response_summary, evidence_used, task_outcome,
20.     human_feedback, utility_score, negative_flag, session_origin
21. ) VALUES (
22.     NEW_UUID(), uu_id, query, query_embedding,
23.     SUMMARIZE(agent_output, max_tokens=200),
24.     TO_JSONB(evidence_record),
25.     task_outcome,
26.     human_feedback,
27.     MEAN(evidence_record.utility_score),
28.     task_outcome == 'failure' AND human_feedback == 'rejected',
29.     session_id
30. )
31.
32. // ═══════════════════════════════════════════════════
33. // MEMORY TABLE: Promote validated corrections/facts
34. // ═══════════════════════════════════════════════════
35. IF human_feedback == 'corrected':
36.     correction_content ← EXTRACT_CORRECTION(human_feedback.correction_text)
37.     correction_hash ← SHA256(correction_content)
38.
39.     // Deduplication check
40.     existing ← QUERY:
41.         SELECT memory_id FROM memory
42.         WHERE uu_id = :uu_id AND content_hash = :correction_hash
43.
44.     IF existing IS EMPTY:
45.         correction_embedding ← OPENAI_EMBED(model: embedding_model, input: correction_content)
46.         INSERT INTO memory (
47.             memory_id, uu_id, memory_type, content_text, content_embedding,
48.             authority_tier, confidence, source_provenance, content_hash
49.         ) VALUES (
50.             NEW_UUID(), uu_id, 'correction', correction_content,
51.             correction_embedding, 'curated', 0.9,
52.             TO_JSONB({source: 'human_correction', session: session_id}),
53.             correction_hash
54.         )
55.     ELSE:
56.         // Update confidence of existing memory
57.         UPDATE memory SET confidence = MIN(confidence + 0.05, 1.0),
58.                          updated_at = NOW()
59.         WHERE memory_id = existing[0].memory_id AND uu_id = :uu_id
60.
61. // ═══════════════════════════════════════════════════
62. // MEMORY TABLE: Extract and store non-obvious learned facts
63. // ═══════════════════════════════════════════════════
64. IF task_outcome == 'success' AND human_feedback IN ('approved', NULL):
65.     learned_facts ← VLLM_INFERENCE(
66.         model: "fact-extractor",
67.         prompt: "Extract non-obvious factual statements from this successful interaction "
68.               + "that would improve future responses for this user. "
69.               + "Return only novel, specific facts (not common knowledge). "
70.               + "Query: " + query + "\nResponse: " + agent_output,
71.         max_tokens: 300,
72.         temperature: 0.0
73.     )
74.
75.     FOR EACH fact IN PARSE_FACTS(learned_facts):
76.         fact_hash ← SHA256(fact)
77.         IF NOT EXISTS(SELECT 1 FROM memory WHERE uu_id = :uu_id AND content_hash = :fact_hash):
78.             fact_embedding ← OPENAI_EMBED(model: embedding_model, input: fact)
79.             // Semantic deduplication: check if very similar memory already exists
80.             similar_existing ← QUERY:
81.                 SELECT memory_id, content_text,
82.                        1 - (content_embedding <=> :fact_embedding) AS sim
83.                 FROM memory
84.                 WHERE uu_id = :uu_id
85.                   AND 1 - (content_embedding <=> :fact_embedding) > 0.92
86.                 LIMIT 1
87.
88.             IF similar_existing IS EMPTY:
89.                 INSERT INTO memory (
90.                     memory_id, uu_id, memory_type, content_text, content_embedding,
91.                     authority_tier, confidence, source_provenance, content_hash
92.                 ) VALUES (
93.                     NEW_UUID(), uu_id, 'fact', fact, fact_embedding,
94.                     'derived', 0.6, -- lower initial confidence
95.                     TO_JSONB({source: 'auto_extraction', session: session_id}),
96.                     fact_hash
97.                 )
98.             ELSE:
99.                 // Reinforcement: bump confidence of existing similar memory
100.                UPDATE memory SET confidence = MIN(confidence + 0.03, 1.0),
101.                                 access_count = access_count + 1,
102.                                 updated_at = NOW()
103.                WHERE memory_id = similar_existing[0].memory_id AND uu_id = :uu_id
104.
105. // ═══════════════════════════════════════════════════
106. // SESSION TABLE: Already updated during interaction
107. // ═══════════════════════════════════════════════════
108. // (Session turns are written in real-time during the conversation)
109.
110. RETURN WriteConfirmation {
111.     history_written: TRUE,
112.     memory_items_added: COUNT_NEW_MEMORIES(),
113.     memory_items_reinforced: COUNT_REINFORCED_MEMORIES(),
114.     negative_flagged: task_outcome == 'failure'
115. }
```

---

## 16. Memory Maintenance and Hygiene

```
ALGORITHM: MemoryMaintenanceScheduler
INPUT:  schedule: CronSchedule (e.g., daily at 03:00 UTC)
OUTPUT: maintenance_report: MaintenanceReport

1.  ON schedule:
2.      FOR EACH uu_id IN ACTIVE_USERS():
3.
4.          // ═══════════ EXPIRY CLEANUP ═══════════
5.          expired_count ← DELETE FROM memory
6.              WHERE uu_id = :uu_id AND expires_at IS NOT NULL AND expires_at < NOW()
7.
8.          // ═══════════ SESSION CLEANUP ═══════════
9.          old_sessions ← DELETE FROM session
10.             WHERE uu_id = :uu_id AND expires_at < NOW()
11.
12.         // ═══════════ CONFIDENCE DECAY ═══════════
13.         // Memories that haven't been accessed decay slowly
14.         UPDATE memory
15.         SET confidence = GREATEST(confidence * 0.995, 0.1)
16.         WHERE uu_id = :uu_id
17.           AND last_accessed_at < NOW() - INTERVAL '30 days'
18.           AND authority_tier NOT IN ('canonical')  -- canonical never decays
19.
20.         // ═══════════ DEDUPLICATION PASS ═══════════
21.         // Find semantically near-duplicate memories and merge
22.         duplicates ← QUERY:
23.             SELECT a.memory_id AS id_a, b.memory_id AS id_b,
24.                    1 - (a.content_embedding <=> b.content_embedding) AS sim
25.             FROM memory a, memory b
26.             WHERE a.uu_id = :uu_id AND b.uu_id = :uu_id
27.               AND a.memory_id < b.memory_id  -- avoid self-join and double-count
28.               AND 1 - (a.content_embedding <=> b.content_embedding) > 0.95
29.
30.         FOR EACH (id_a, id_b, sim) IN duplicates:
31.             // Keep the higher-authority, higher-confidence entry
32.             winner ← SELECT_WINNER(id_a, id_b)  -- by authority, then confidence, then recency
33.             loser ← OTHER(id_a, id_b, winner)
34.             // Merge: absorb loser's confidence boost and provenance
35.             UPDATE memory SET
36.                 confidence = MIN(confidence + 0.02, 1.0),
37.                 source_provenance = JSONB_CONCAT(source_provenance, GET_PROVENANCE(loser))
38.             WHERE memory_id = winner AND uu_id = :uu_id
39.             DELETE FROM memory WHERE memory_id = loser AND uu_id = :uu_id
40.
41.         // ═══════════ MEMORY CAPACITY CAP ═══════════
42.         total_memories ← COUNT(*) FROM memory WHERE uu_id = :uu_id
43.         IF total_memories > MAX_MEMORIES_PER_USER:  -- typically 10000
44.             // Evict lowest-value memories
45.             eviction_score ← confidence * authority_numeric
46.                             * ln(1 + access_count)
47.                             * exp(-(NOW() - updated_at) / τ_eviction)
48.             DELETE FROM memory WHERE memory_id IN (
49.                 SELECT memory_id FROM memory
50.                 WHERE uu_id = :uu_id
51.                 ORDER BY eviction_score ASC
52.                 LIMIT total_memories - MAX_MEMORIES_PER_USER
53.             )
54.
55.         // ═══════════ HISTORY COMPACTION ═══════════
56.         // Compress old history entries to save storage
57.         old_history_count ← COUNT(*) FROM history
58.             WHERE uu_id = :uu_id AND created_at < NOW() - INTERVAL '90 days'
59.         IF old_history_count > HISTORY_COMPACTION_THRESHOLD:
60.             // Retain only high-utility and negative-flagged entries
61.             DELETE FROM history
62.             WHERE uu_id = :uu_id
63.               AND created_at < NOW() - INTERVAL '90 days'
64.               AND utility_score < 0.3
65.               AND negative_flag = FALSE
66.
67.     RETURN maintenance_report
```

---

## 17. Evaluation and Quality Gates

### 17.1 Continuous Retrieval Evaluation

```
ALGORITHM: ContinuousRetrievalEval
INPUT:  eval_queries: List<EvalQuery>, retrieval_engine: MasterRetrievalOrchestrator
OUTPUT: eval_report: RetrievalEvalReport

1.  FOR EACH eq IN eval_queries:
2.      // Run retrieval
3.      response ← retrieval_engine.retrieve(eq.query, eq.uu_id, eq.session_id, ...)
4.
5.      // Intrinsic metrics
6.      precision_k ← |RELEVANT(response.fragments) ∩ eq.ground_truth| / |response.fragments|
7.      recall_k ← |RELEVANT(response.fragments) ∩ eq.ground_truth| / |eq.ground_truth|
8.      ndcg_k ← DCG(response.fragments, eq.graded_relevance) / IDCG(eq.graded_relevance)
9.      mrr ← 1 / RANK_OF_FIRST_RELEVANT(response.fragments, eq.ground_truth)
10.
11.     // User-isolation correctness
12.     isolation_pass ← ALL(
13.         f.provenance.origin.source_id STARTS_WITH eq.uu_id
14.         FOR f IN response.fragments
15.     )
16.
17.     // Source diversity
18.     source_tables_covered ← |UNIQUE(f.provenance.origin.source_table FOR f IN response.fragments)|
19.     diversity_score ← source_tables_covered / 3
20.
21.     // Latency compliance
22.     latency_pass ← response.latency_ms ≤ SLA_LATENCY_MS
23.
24.     // Token efficiency
25.     token_efficiency ← response.budget_report.utilization
26.
27.     RECORD(eq.id, precision_k, recall_k, ndcg_k, mrr,
28.            isolation_pass, diversity_score, latency_pass, token_efficiency)
29.
30. // Aggregate
31. report ← RetrievalEvalReport {
32.     mean_precision: MEAN(precision_k),
33.     mean_recall: MEAN(recall_k),
34.     mean_ndcg: MEAN(ndcg_k),
35.     mean_mrr: MEAN(mrr),
36.     isolation_rate: MEAN(isolation_pass),  -- MUST be 1.0
37.     mean_diversity: MEAN(diversity_score),
38.     latency_compliance: MEAN(latency_pass),
39.     mean_token_efficiency: MEAN(token_efficiency)
40. }
41.
42. // Quality gates
43. ASSERT report.isolation_rate == 1.0, "CRITICAL: User isolation violated"
44. ASSERT report.mean_ndcg ≥ 0.80, "Retrieval quality below threshold"
45. ASSERT report.latency_compliance ≥ 0.95, "Latency SLA violated"
46.
47. RETURN report
```

### 17.2 Evaluation Metric Targets

| Metric | Formula | Target | Gate Type |
|---|---|---|---|
| User Isolation | $\frac{\|\text{correct\_user\_fragments}\|}{\|\text{all\_fragments}\|}$ | $= 1.00$ | **Hard** (blocks deployment) |
| Precision@10 | $\frac{\|\text{Rel} \cap \text{Top-10}\|}{10}$ | $\geq 0.80$ | Soft (alert) |
| Recall@20 | $\frac{\|\text{Rel} \cap \text{Top-20}\|}{\|\text{Rel}\|}$ | $\geq 0.90$ | Soft (alert) |
| NDCG@10 | $\frac{\text{DCG@10}}{\text{IDCG@10}}$ | $\geq 0.85$ | Soft (alert) |
| MRR | $\frac{1}{\|\mathcal{Q}\|}\sum\frac{1}{\text{rank}_q}$ | $\geq 0.90$ | Soft (alert) |
| Latency P99 | Measured | $\leq 250\text{ms}$ | Hard (blocks deployment) |
| Token Utilization | $\frac{\text{used}}{\text{budget}}$ | $\geq 0.70$ | Informational |
| Source Diversity | $\frac{\|\text{unique tables}\|}{3}$ | $\geq 0.67$ | Informational |
| Negative Suppression | $\frac{\|\mathcal{N} \cap \text{Top-K}\|}{\|\text{Top-K}\|}$ | $\leq 0.02$ | Soft (alert) |

---

## 18. Above-SOTA Innovation Summary

The following table catalogues the techniques in this design that exceed the capabilities of conventional retrieval systems:

| Innovation | Mechanism | Impact |
|---|---|---|
| **User-partition-first filtering** | PostgreSQL hash partitioning + `WHERE uu_id` as universal first predicate | Eliminates cross-user leakage; $O(\log n_u)$ vs $O(\log N)$ |
| **Three-table retrieval hierarchy** | Session (volatile), History (utility-weighted), Memory (canonical) with distinct strategies | Captures temporal, experiential, and validated knowledge layers |
| **Contextual dual embedding** | Plain + session-augmented embeddings, weighted by follow-up probability | Captures both literal and contextual user intent |
| **Multi-stream agreement bonus** | $\Phi_{\text{agree}}(d) = (k/m)^{0.5}$ boosts documents found by multiple independent retrievers | Exploits retriever independence for confidence calibration |
| **Negative evidence suppression** | Failed-task evidence stored with `negative_flag`, actively down-weighted | Prevents repeated retrieval of proven-harmful evidence |
| **Historical utility propagation** | $S_{\text{hist}} = \sum \text{sim} \cdot u_i \cdot \omega \cdot \rho / (\sum \text{sim} + \epsilon)$ with transitive aggregation | Leverages past success to predict future utility |
| **Source-table diversity in MMR** | Bonus for selecting fragments from underrepresented tables | Ensures multi-perspective evidence composition |
| **Progressive quality gate** | Retrieve → assess → refine → re-retrieve loop with bounded iterations | Prevents insufficient evidence from reaching the agent |
| **Adaptive fusion weights** | Query-class-dependent weight vectors $\mathbf{w}(\mathcal{C})$ | Optimizes fusion per query type instead of static weights |
| **Memory confidence reinforcement** | Repeated utility increases confidence; disuse decays it | Self-correcting memory that evolves with user behavior |
| **Semantic deduplication at write** | Embedding similarity $> 0.92$ triggers merge instead of insert | Prevents memory bloat while preserving provenance |
| **Negative memory promotion** | Failed tasks with rejected feedback create `negative_flag` entries | Closes the negative feedback loop |
| **Dual-model pipeline** | OpenAI for embeddings + vLLM for local reranking/decomposition | Optimal cost-latency-quality trade-off |

---

## 19. Latency Budget Allocation Table

| Phase | Operation | Budget (ms) | Parallelism | Fallback |
|---|---|---|---|---|
| 0 | User scope lock | 1 | — | Fail-fast if user not found |
| 1 | Session + memory load | 5 | Parallel (2 queries) | Empty context if timeout |
| 2 | Query preprocessing | 20 | Sequential (depends on Phase 1) | Use raw query as-is |
| 3 | MIMO retrieval (5 streams) | 120 | Parallel (5 streams) | Partial results from completed streams |
| 4 | Fusion | 8 | — | RRF fallback if learned fusion fails |
| 5 | Cross-encoder reranking | 55 | Batched GPU via vLLM | Use fused scores as-is |
| 6 | MMR diversity selection | 5 | — | Top-K by score (no diversity) |
| 7 | Provenance + token budget | 6 | — | Omit provenance metadata |
| 8 | Quality gate (optional) | 25 | — | Skip if deadline insufficient |
| — | Safety margin | 5 | — | — |
| **Total** | | **250** | | |

---

## 20. Operational Invariants

The following invariants must hold at all times and are enforced mechanically through the system:

1. **User isolation invariant**: $\forall f \in \text{Response.fragments}: f.\text{uu\_id} = \text{request.uu\_id}$. Violation is a P0 security incident.

2. **Provenance completeness**: $\forall f \in \text{Response.fragments}: f.\text{provenance} \neq \emptyset$. No anonymous evidence is admissible.

3. **Token budget compliance**: $\sum_{f \in \text{Response.fragments}} f.\text{token\_count} \leq \text{request.token\_budget}$.

4. **Latency SLA**: $\text{Response.latency\_ms} \leq \text{request.latency\_deadline\_ms}$ for $\geq 99\%$ of requests.

5. **Idempotency**: Repeated identical retrieval requests within the cache TTL return identical results.

6. **Memory write validation**: No memory entry is written without deduplication check (content hash) AND semantic similarity check ($> 0.92$ threshold).

7. **Negative evidence bound**: $\frac{|\{f \in \text{Top-K}: f \in \mathcal{N}\}|}{K} \leq 0.02$. Negative evidence must be suppressed to below 2% of the result set.

8. **Graceful degradation**: If any retrieval stream times out, the system returns results from completed streams rather than failing entirely. Minimum viable response requires at least one stream to complete.

---

This design achieves above-SOTA retrieval performance through the synergistic composition of user-scoped PostgreSQL partitioning, three-table knowledge hierarchy, dual-model inference (OpenAI + vLLM), MIMO multi-stream retrieval with adaptive fusion, negative evidence suppression, historical utility propagation, and progressive quality-gated refinement — all operating within a strict 250ms latency budget with provenance-tagged, token-budget-fitted evidence output.
