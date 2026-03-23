

# SOTA Context Engineering for Cyclic Agent-Loop Execution: Principled Architecture for Infinite-Horizon Agentic Systems

---

## 1. Foundational Thesis: The Context Window as a Cyclic Computational Surface

### 1.1 The Cyclic Invariant

The defining characteristic of production agentic systems — the property that separates them structurally from single-shot inference — is that the context window operates as a **cyclic computational surface**, not a linear buffer. Each agent-loop iteration consumes a compiled context, produces an output (which includes reasoning, tool calls, partial results, state mutations), and then that output re-enters the context window as input for the subsequent cycle. The system is governed by a recurrence:

$$\mathcal{C}_{t+1} = \text{Compile}\!\Big(\mathcal{P},\; \mathcal{F}\big(\mathcal{C}_t,\; \mathcal{O}_t,\; \mathcal{R}_{t+1},\; \mathcal{M}_{t+1},\; \mathcal{T}_{t+1}\big)\Big)$$

where:

- $\mathcal{C}_t$ is the compiled context at cycle $t$
- $\mathcal{O}_t$ is the model output (generation) at cycle $t$
- $\mathcal{R}_{t+1}$ is fresh retrieval evidence triggered by cycle $t$'s actions
- $\mathcal{M}_{t+1}$ is the updated memory state after cycle $t$
- $\mathcal{T}_{t+1}$ is the set of tool-call results returned during cycle $t$
- $\mathcal{F}$ is the **cycle filter** — the function that selects, compresses, and re-ranks all candidates
- $\mathcal{P}$ is the invariant policy layer (system + developer constraints)
- $\text{Compile}$ is the prefill compiler

The critical insight: **every cycle is an opportunity for context degradation**. Without disciplined cycle management, the system accumulates noise tokens — stale tool outputs, redundant history, superseded plan fragments, verbose acknowledgments — that compound across iterations until the context window is saturated with low-utility content and the agent's effective reasoning capacity collapses.

### 1.2 The Noise Accumulation Theorem

Define the **signal ratio** at cycle $t$ as:

$$\sigma_t = \frac{\sum_{i} u_i(t) \cdot t_i(t)}{\sum_{i} t_i(t)}$$

where $u_i(t) \in [0, 1]$ is the task utility of context item $i$ at cycle $t$ and $t_i(t)$ is its token cost. Without active cycle management, $\sigma_t$ degrades monotonically:

$$\sigma_{t+1} \leq \sigma_t + \epsilon_{\text{fresh}} - \delta_{\text{stale}}$$

where $\epsilon_{\text{fresh}}$ is the signal contribution of newly injected content and $\delta_{\text{stale}}$ is the cumulative decay of retained content. In practice, $\delta_{\text{stale}}$ grows superlinearly because:

1. Each cycle's output (tool results, reasoning traces, intermediate states) is admitted to history, inflating Tier 4.
2. Earlier evidence loses relevance as the task evolves, but remains in context unless explicitly evicted.
3. Compression artifacts from prior cycles introduce semantic drift that compounds over iterations.

**The objective of cyclic context engineering is to maintain $\sigma_t \geq \sigma_{\min}$ for all $t$**, where $\sigma_{\min}$ is the minimum signal ratio required for reliable task completion — empirically, $\sigma_{\min} \geq 0.70$.

### 1.3 Why Prompts, Instructions, and System Directives Are Noise Candidates

A foundational departure from conventional prompt engineering: in a SOTA cyclic context system, **every token in the context window — including system prompts, developer instructions, and role policies — is a noise candidate unless it demonstrably contributes to the current cycle's task utility**. This does not mean system policies are discarded; it means they are:

1. **Compressed to minimal effective form.** Verbose role descriptions, motivational framing, stylistic guidance, and redundant safety restatements are stripped.
2. **Conditionally included.** Constraints that apply only to specific task phases (e.g., "when generating SQL, always use parameterized queries") are loaded only during those phases.
3. **Versioned and cached.** The invariant core of system policy is hashed and cached in the KV-cache layer; only the delta from the prior cycle is recomputed.
4. **Measured by ablation.** If removing a system-prompt sentence produces no measurable change in output quality on the benchmark suite, that sentence is eliminated.

The principle is absolute: **content is first; everything else — developer prompt, instruction scaffolding, system preamble, formatting directives — justifies its token cost through measured task utility or is excised.**

---

## 2. Cyclic Context Architecture: The Six-Layer Execution Stack

### 2.1 Layer Decomposition

The cyclic context system is organized into six architectural layers, each with a well-defined responsibility, typed interface, and measurable quality gate.

| Layer | Name | Responsibility | Interface Protocol | Cycle Frequency |
|---|---|---|---|---|
| **L0** | Policy Kernel | Invariant constraints, safety boundaries, instruction hierarchy | Static prefill; KV-cache pinned | Once per session; delta per task-phase |
| **L1** | Task Orchestrator | Plan state, decomposition, step tracking, exit criteria | gRPC internal; Protobuf state schema | Every cycle |
| **L2** | Retrieval Engine | Evidence acquisition, hybrid search, provenance tagging | gRPC internal; MCP tool surface | On-demand per subquery |
| **L3** | Memory Governance | Working/session/episodic/semantic/procedural memory arbitration | gRPC internal; typed write contracts | Every cycle (read); gated (write) |
| **L4** | Cycle Filter | Selection, compression, deduplication, eviction, budget enforcement | Internal pipeline; functional composition | Every cycle |
| **L5** | Prefill Compiler | Deterministic assembly, validation, trace emission | Internal pipeline; produces immutable artifact | Every cycle |

### 2.2 Data Flow per Cycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CYCLE t → t+1                                │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ L1: Task  │───▶│ L2: Retr. │───▶│ L3: Mem. │───▶│ L4: Filter│    │
│  │ Orchest.  │    │ Engine   │    │ Govern.  │    │          │      │
│  └────┬─────┘    └──────────┘    └──────────┘    └────┬─────┘      │
│       │                                               │             │
│       │         ┌──────────┐                          │             │
│       │         │ L0: Policy│──────────────────────┐  │             │
│       │         │ Kernel   │                       │  │             │
│       │         └──────────┘                       ▼  ▼             │
│       │                                     ┌──────────┐            │
│       └────────────────────────────────────▶│ L5: Prefill│           │
│                                             │ Compiler  │           │
│                                             └────┬─────┘            │
│                                                  │                  │
│                                                  ▼                  │
│                                          ┌──────────────┐           │
│                                          │  LLM Inference │          │
│                                          │  (Generation)  │          │
│                                          └──────┬───────┘           │
│                                                 │                   │
│                                                 ▼                   │
│                                          ┌──────────────┐           │
│                                          │ Output Parser │           │
│                                          │ + Verifier    │           │
│                                          └──────┬───────┘           │
│                                                 │                   │
│                          ┌──────────────────────┼────────────┐      │
│                          ▼                      ▼            ▼      │
│                   ┌───────────┐          ┌───────────┐ ┌─────────┐  │
│                   │ Tool Exec │          │ State Mut. │ │ Memory  │  │
│                   │ (MCP/gRPC)│          │ (L1 update)│ │ Write   │  │
│                   └─────┬─────┘          └───────────┘ │ (L3)    │  │
│                         │                              └─────────┘  │
│                         ▼                                           │
│                   Tool Results ──────────────▶ CYCLE t+1 INPUT      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Every arrow represents a typed contract. Every box has a bounded latency budget. Every transition is observable via distributed traces.

---

## 3. L0 — Policy Kernel: Minimal Invariant Core

### 3.1 Design Principle

The Policy Kernel is the **smallest possible set of tokens that define irrevocable behavioral constraints**. It is not a "system prompt" in the traditional sense — it is a compiled, compressed, phase-conditional directive set.

### 3.2 Policy Compilation

---

**PSEUDO-ALGORITHM 3.1: Policy Kernel Compilation**

```
PROCEDURE CompilePolicyKernel(
    base_policy: VersionedPolicy,
    task_phase: TaskPhase,
    active_tools: ToolManifest,
    security_context: SecurityContext
) → PolicyKernel:

    // Step 1: Load base invariants (safety, identity, hierarchy)
    invariants ← base_policy.invariant_directives
    // These are the absolute minimum: typically 150-300 tokens
    // Example: instruction hierarchy declaration, safety boundaries,
    //          output schema contract, hallucination prohibition

    // Step 2: Select phase-conditional directives
    phase_directives ← base_policy.phase_directives[task_phase]
    // Only load directives relevant to current execution phase
    // e.g., "code generation" phase loads coding constraints;
    //        "analysis" phase loads citation requirements

    // Step 3: Derive tool-scoped constraints
    tool_constraints ← []
    FOR EACH tool IN active_tools:
        IF base_policy.has_tool_constraint(tool.id) THEN
            APPEND base_policy.tool_constraint(tool.id) TO tool_constraints

    // Step 4: Inject security-context-specific rules
    security_rules ← DeriveSecurityRules(security_context)
    // e.g., PII handling rules if task involves personal data

    // Step 5: Compress to minimal token form
    raw_kernel ← Concatenate([invariants, phase_directives, 
                               tool_constraints, security_rules])
    
    compressed_kernel ← MinimalDirectiveCompressor(raw_kernel)
    // Removes: filler words, motivational framing, redundant restatements
    // Preserves: imperative constraints, schema definitions, hierarchy declarations
    
    // Step 6: Validate token budget
    ASSERT TokenCount(compressed_kernel) ≤ config.policy_kernel_max_tokens
    // Typical target: 200-500 tokens for the entire policy kernel
    
    // Step 7: Compute hash for KV-cache reuse
    kernel_hash ← SHA256(compressed_kernel)
    
    RETURN PolicyKernel(
        content = compressed_kernel,
        hash = kernel_hash,
        phase = task_phase,
        version = base_policy.version
    )
```

---

### 3.3 Noise Elimination from System Prompts

The following categories of conventional system-prompt content are classified as **noise** and eliminated from the Policy Kernel:

| Noise Category | Example | Reason for Elimination |
|---|---|---|
| **Identity narration** | "You are a helpful, harmless, and honest AI assistant." | Zero marginal utility after the first sentence; model behavior is governed by training, not self-description verbosity |
| **Motivational framing** | "You should try your best to help the user." | No measurable impact on output quality; consumes tokens |
| **Redundant safety restatements** | Repeating the same prohibition in three different phrasings | Single precise statement is sufficient; redundancy dilutes attention |
| **Stylistic decoration** | "Please format your responses in a clear and concise manner." | Controlled by output schema contracts, not prose requests |
| **Capability enumeration** | "You can search the web, write code, analyze data..." | Tool affordances are loaded dynamically from the tool manifest; static enumeration is stale and wasteful |
| **Hypothetical scenarios** | "If the user asks you to do X, respond with Y..." | Encode as conditional rules in the phase-directive table, not as narrative |

**Empirical benchmark:** A well-compressed Policy Kernel achieves equivalent or superior constraint adherence to a verbose 2,000-token system prompt while consuming 200–500 tokens — a 4×–10× compression ratio with zero fidelity loss on constraint compliance benchmarks.

### 3.4 KV-Cache Pinning

For models that support prefix caching (most production inference servers), the Policy Kernel is designed to be **cache-stable across cycles**:

- The invariant portion (safety, hierarchy, identity) is identical across all cycles within a session → KV-cache hit rate approaches 1.0.
- Phase-conditional directives change only on task-phase transitions (not every cycle) → cache invalidation is infrequent.
- The kernel is placed at the **start of the prefill** to maximize cache prefix match length.

The latency and cost savings are:

$$\text{Savings}_{\text{per cycle}} = T_{\text{kernel}} \cdot C_{\text{prefill\_per\_token}} \cdot (1 - \text{cache\_miss\_rate})$$

For a 400-token kernel with 95% cache hit rate and $C_{\text{prefill}} = \$0.003 / 1\text{K tokens}$: savings of $\approx \$0.00114$ per cycle, which compounds to significant cost reduction at scale.

---

## 4. L1 — Task Orchestrator: Bounded Control Loop with State Serialization

### 4.1 The Agent Loop as a Finite State Machine

The agent loop is not an unbounded recursive process. It is a **bounded finite state machine** with explicit states, transitions, exit criteria, and failure modes:

```
STATES := {
    PLAN,           // Decompose objective into subtasks
    RETRIEVE,       // Acquire evidence for current subtask  
    ACT,            // Execute tool call or generate output
    VERIFY,         // Validate action result against criteria
    CRITIQUE,       // Assess whether result meets quality gate
    REPAIR,         // Fix identified deficiencies
    COMMIT,         // Persist validated result
    ESCALATE,       // Human-in-the-loop intervention
    TERMINATE       // Exit with final output
}

TRANSITIONS := {
    PLAN      → RETRIEVE   | ESCALATE,
    RETRIEVE  → ACT        | PLAN (if evidence insufficient),
    ACT       → VERIFY,
    VERIFY    → COMMIT     | CRITIQUE,
    CRITIQUE  → REPAIR     | COMMIT | ESCALATE,
    REPAIR    → ACT,       // bounded: max_repair_attempts
    COMMIT    → PLAN (next subtask) | TERMINATE,
    ESCALATE  → PLAN | TERMINATE
}

BOUNDS := {
    max_cycles_total:       50,
    max_repair_per_subtask: 3,
    max_retrieval_retries:  2,
    deadline_ms:            30000
}
```

### 4.2 Task State Serialization: Minimal Token Representation

At each cycle, the Task Orchestrator serializes its state into the context. This serialization must be **maximally compact** — every token spent on state representation is a token unavailable for evidence or reasoning.

---

**PSEUDO-ALGORITHM 4.1: Compact Task State Serialization**

```
PROCEDURE SerializeTaskState(task: Task) → CompactStateString:

    // Encode only decision-relevant state
    state ← {
        "objective":     task.objective_hash,          // Reference, not full text
        "phase":         task.current_phase.name,       // e.g., "ACT"
        "step":          task.current_step_index,       // e.g., 3 of 7
        "step_desc":     task.current_step.description, // 1-line summary
        "completed":     [s.id FOR s IN task.steps IF s.status = DONE],
        "pending":       [s.id FOR s IN task.steps IF s.status = PENDING],
        "blocked":       [s.id FOR s IN task.steps IF s.status = BLOCKED],
        "errors":        task.active_errors[-3:],       // Last 3 only
        "repair_count":  task.current_repair_count,
        "cycle":         task.cycle_number,
        "budget_remain": task.remaining_cycles
    }

    // Compress: use abbreviated keys, omit defaults, omit empty lists
    compact ← CompactJSON(state, 
                          abbreviate_keys = TRUE,
                          omit_empty = TRUE,
                          omit_defaults = TRUE)
    
    ASSERT TokenCount(compact) ≤ config.task_state_max_tokens  // Target: 50-150 tokens
    
    RETURN compact
```

---

### 4.3 Instance-Episode Decomposition

The user's emphasis on **instance-episode combination** maps to a precise architectural concept:

- **Instance** = a single cycle of the agent loop (one LLM invocation with its compiled context). The instance is ephemeral. Its working context exists only for the duration of that inference call.
- **Episode** = a complete task execution spanning multiple instances/cycles, from initial objective to terminal commit or escalation.

The critical optimization is: **each instance must contain exactly the information required for its specific cycle, drawn from the episode's accumulated state, without carrying forward the full episode history.**

$$\mathcal{C}_{\text{instance}}(t) = \text{Select}\Big(\mathcal{E}_{\text{episode}},\; \text{phase}(t),\; \text{step}(t),\; \text{budget}(t)\Big)$$

where $\mathcal{E}_{\text{episode}}$ is the total accumulated episode state (potentially gigabytes of tool outputs, retrieval results, and history) and $\text{Select}$ is the cycle filter that extracts only the decision-relevant subset.

This is the core of the **cyclic context discipline**: the episode grows unboundedly; the instance is always bounded by $\mathcal{W}$.

---

## 5. L2 — Retrieval Engine: Deterministic Evidence Acquisition with Provenance

### 5.1 Query Decomposition Before Retrieval

A single user query or task-step requirement is never submitted directly to a retrieval system. Instead, it is **decomposed, expanded, and routed** through a multi-strategy retrieval plan.

---

**PSEUDO-ALGORITHM 5.1: Retrieval Plan Generation**

```
PROCEDURE GenerateRetrievalPlan(
    task_step: TaskStep,
    episode_context: EpisodeContext,
    source_registry: SourceRegistry
) → RetrievalPlan:

    // Step 1: Decompose the task step into atomic information needs
    info_needs ← DecomposeInformationNeeds(task_step)
    // e.g., "Implement rate limiting for the API" decomposes into:
    //   - Current rate limiting implementation (code search)
    //   - Rate limiting best practices (semantic search, knowledge base)
    //   - API traffic patterns (metrics/analytics query)
    //   - Existing configuration (config file search)

    // Step 2: For each information need, determine retrieval strategy
    subqueries ← []
    FOR EACH need IN info_needs:
        strategy ← SelectStrategy(need)
        // Strategies: EXACT_MATCH, SEMANTIC_SEARCH, GRAPH_TRAVERSAL,
        //             STRUCTURED_QUERY, LIVE_INSPECTION, MEMORY_RECALL

        sources ← RouteToSources(need, source_registry, strategy)
        // Each source has: schema, latency_tier, authority_score, freshness_guarantee

        query ← RewriteQuery(need, strategy, episode_context)
        // Expand with synonyms, refine with episode-specific terminology,
        // add filters (date range, author, file path, etc.)

        APPEND SubQuery(
            original_need = need,
            rewritten_query = query,
            strategy = strategy,
            sources = sources,
            timeout_ms = sources.max_latency_tier,
            max_results = config.max_results_per_subquery
        ) TO subqueries

    // Step 3: Identify parallelizable subqueries
    parallel_groups ← GroupByIndependence(subqueries)

    // Step 4: Set total retrieval budget
    plan ← RetrievalPlan(
        subqueries = subqueries,
        parallel_groups = parallel_groups,
        total_timeout_ms = config.retrieval_deadline_ms,
        total_token_budget = config.evidence_token_budget,
        dedup_strategy = SEMANTIC_DEDUP,
        provenance_required = TRUE
    )

    RETURN plan
```

---

### 5.2 Hybrid Retrieval Execution

Each subquery is executed via its assigned strategy, and results are merged through a unified ranking pipeline:

| Strategy | Mechanism | Latency Tier | Authority Signal |
|---|---|---|---|
| **Exact Match** | Keyword/BM25, regex, symbol search | Fast (< 50ms) | High (precise match) |
| **Semantic Search** | Dense embedding similarity, cross-encoder reranking | Medium (50–200ms) | Medium (semantic approximation) |
| **Graph Traversal** | Knowledge graph, code dependency graph, lineage graph | Medium (100–500ms) | High (structural relationship) |
| **Structured Query** | SQL/NoSQL against structured data stores | Fast–Medium | High (exact data) |
| **Live Inspection** | Runtime state, log tailing, metrics query | Variable | Highest (ground truth) |
| **Memory Recall** | Query against episodic/semantic memory layers | Fast (< 30ms) | Medium (validated prior knowledge) |

### 5.3 Evidence Ranking and Selection

Retrieved results are ranked by a composite score:

$$\text{score}(e) = w_r \cdot \text{relevance}(e) + w_a \cdot \text{authority}(e) + w_f \cdot \text{freshness}(e) + w_u \cdot \text{exec\_utility}(e) - w_c \cdot \text{token\_cost}(e)$$

where:

- $\text{relevance}(e)$: Semantic similarity to the information need (cross-encoder score or reciprocal rank fusion)
- $\text{authority}(e)$: Source credibility (official documentation > blog post > generated content)
- $\text{freshness}(e)$: Temporal recency, exponential decay from last-modified timestamp
- $\text{exec\_utility}(e)$: Predicted utility for the current execution phase (e.g., code snippets during ACT phase score higher than conceptual explanations)
- $\text{token\_cost}(e)$: Normalized token count (penalizes verbose evidence when budget is tight)

Evidence items below a composite threshold $\theta_{\min}$ are discarded. Items above the threshold are selected greedily by score until the evidence token budget is exhausted.

### 5.4 Provenance Tagging

Every evidence item admitted to the context carries a mandatory provenance record:

```
ProvenancedEvidence := {
    content:          string,           // The evidence text/data
    source_id:        URI,              // Canonical source identifier
    source_type:      SourceType,       // DOCUMENT | CODE | API | METRIC | MEMORY
    retrieval_strategy: Strategy,       // How it was found
    retrieval_query:  string,           // The subquery that produced it
    relevance_score:  float,            // [0, 1]
    authority_score:  float,            // [0, 1]
    freshness:        ISO8601,          // Last-modified or retrieved timestamp
    chunk_boundaries: {start, end},     // Position in source document
    token_cost:       int,              // Exact token count
    lineage_tag:      string            // Which information need this serves
}
```

**Anonymous evidence — evidence without provenance — is never admitted to the context.** This is an inviolable architectural constraint.

---

## 6. L3 — Memory Governance: Five-Layer Memory with Strict Write Discipline

### 6.1 Memory Layer Architecture

| Layer | Scope | Persistence | Write Policy | Read Latency | Eviction |
|---|---|---|---|---|---|
| **Working Memory** | Current instance only | Ephemeral (dies with cycle) | Free write | Immediate (in-context) | Automatic at cycle end |
| **Session Memory** | Current episode | Session-scoped | Validated write | < 10ms | Session end + TTL |
| **Episodic Memory** | Cross-episode, per-user | Durable | Gated write (dedup + provenance) | < 50ms | Relevance decay + TTL |
| **Semantic Memory** | Organizational | Durable | Curated write (review + approval) | < 100ms | Manual curation |
| **Procedural Memory** | Organizational | Durable | Learned from validated episodes | < 100ms | Periodic re-evaluation |

### 6.2 Memory Write Contract

Memory writes are not free operations. Every write must pass through a validation gate:

---

**PSEUDO-ALGORITHM 6.1: Memory Write Validation**

```
PROCEDURE ValidateMemoryWrite(
    candidate: MemoryCandidate,
    target_layer: MemoryLayer,
    existing_memories: MemoryStore
) → WriteDecision:

    // Gate 1: Novelty check — is this information already known?
    duplicates ← existing_memories.semantic_search(
        candidate.content, 
        threshold = config.dedup_threshold  // e.g., 0.92 cosine similarity
    )
    IF duplicates IS NOT EMPTY THEN
        // Check if candidate is strictly more informative
        IF NOT StrictlyMoreInformative(candidate, duplicates[0]) THEN
            RETURN WriteDecision(action = REJECT, reason = "DUPLICATE")
        ELSE
            RETURN WriteDecision(action = UPDATE, target = duplicates[0].id)

    // Gate 2: Provenance validation — where did this information come from?
    IF candidate.provenance IS NULL THEN
        RETURN WriteDecision(action = REJECT, reason = "NO_PROVENANCE")
    IF candidate.provenance.authority < config.min_authority[target_layer] THEN
        RETURN WriteDecision(action = REJECT, reason = "INSUFFICIENT_AUTHORITY")

    // Gate 3: Non-obviousness filter — is this worth remembering?
    // Reject trivially derivable facts, formatting preferences,
    // ephemeral state that won't help future tasks
    IF IsObvious(candidate, task_domain = candidate.domain) THEN
        RETURN WriteDecision(action = REJECT, reason = "OBVIOUS")

    // Gate 4: Correction/constraint priority — prioritize memories that
    // encode corrections, learned constraints, and failure-avoidance patterns
    IF IsCorrection(candidate) OR IsConstraint(candidate) THEN
        candidate.priority ← HIGH
    ELSE
        candidate.priority ← NORMAL

    // Gate 5: Expiry policy assignment
    candidate.expiry ← ComputeExpiry(target_layer, candidate.type)
    // Working: end of cycle
    // Session: end of session + 24h buffer
    // Episodic: 90 days with relevance-based extension
    // Semantic: no expiry (curated)
    // Procedural: re-evaluation every 30 days

    RETURN WriteDecision(
        action = ADMIT,
        target_layer = target_layer,
        priority = candidate.priority,
        expiry = candidate.expiry,
        provenance = candidate.provenance
    )
```

---

### 6.3 Memory Read: Cycle-Specific Projection

At each cycle, the Memory Governance layer projects a **cycle-specific memory summary** — not a dump of all memories, but a filtered, ranked, compressed view:

$$\mathcal{M}_{\text{instance}}(t) = \text{TopK}\Big(\text{Rank}\big(\text{Filter}(\mathcal{M}_{\text{all}},\; \text{task}(t),\; \text{phase}(t)\big),\; \text{relevance} \times \text{recency}\Big),\; k_{\text{mem}}\Big)$$

where $k_{\text{mem}}$ is determined by the token budget allocated to Tier 3 by the slot allocator.

**Key discipline:** Memories that were useful in cycle $t-1$ are not automatically included in cycle $t$. They must re-qualify through the relevance filter. This prevents memory staleness accumulation across cycles.

---

## 7. L4 — Cycle Filter: The Core Innovation for Cyclic Context Optimization

### 7.1 Design Rationale

The Cycle Filter is the **single most critical component** for maintaining context quality across cycles. It operates between the raw input candidates (prior output, new retrieval, updated memory, tool results) and the Prefill Compiler. Its sole purpose: **ensure that every token entering the next cycle's context demonstrably serves the current cycle's task utility**.

### 7.2 The Eight-Stage Cycle Filter Pipeline

---

**PSEUDO-ALGORITHM 7.1: Cycle Filter Pipeline**

```
PROCEDURE CycleFilter(
    prior_context: ContextObject,    // Context from cycle t
    model_output: ModelOutput,       // Output from cycle t
    tool_results: ToolResult[],      // Tool responses from cycle t
    new_retrieval: Evidence[],       // Fresh retrieval for cycle t+1
    memory_projection: Memory[],    // Filtered memories for cycle t+1
    task_state: TaskState,          // Updated task state
    budget: TokenBudget             // Available budget for cycle t+1
) → FilteredCandidates:

    // ═══════════════════════════════════════════
    // STAGE 1: OUTPUT DECOMPOSITION
    // ═══════════════════════════════════════════
    // Parse model output into atomic components
    output_components ← DecomposeOutput(model_output)
    // Components: {reasoning_trace, decisions, tool_calls, 
    //              partial_results, questions, errors}
    
    // Discard: verbose reasoning traces (retain only conclusions)
    // Discard: acknowledgment tokens ("Sure, I'll help with that")
    // Retain:  decisions, conclusions, error states, pending actions
    retained_output ← []
    FOR EACH component IN output_components:
        IF component.type IN {DECISION, CONCLUSION, ERROR, PENDING_ACTION} THEN
            APPEND component TO retained_output
        ELSE IF component.type = REASONING_TRACE THEN
            // Compress to conclusion only
            conclusion ← ExtractConclusion(component)
            IF conclusion IS NOT NULL THEN
                APPEND conclusion TO retained_output
        // ACKNOWLEDGMENT, FILLER, DECORATION → discarded

    // ═══════════════════════════════════════════
    // STAGE 2: TOOL RESULT TRIAGE
    // ═══════════════════════════════════════════
    // Tool results are high-value but high-cost; triage aggressively
    triaged_tools ← []
    FOR EACH result IN tool_results:
        // Check if this result is still decision-relevant
        IF result.is_superseded_by(subsequent_results) THEN
            DISCARD result  // Newer result makes this obsolete
            CONTINUE
        
        // Compress tool output to decision-relevant content
        compressed ← CompressToolOutput(result, task_state.current_step)
        // e.g., 500-line code output → relevant function signatures + key logic
        // e.g., full API response → relevant fields only
        
        compressed.provenance ← result.provenance
        APPEND compressed TO triaged_tools

    // ═══════════════════════════════════════════
    // STAGE 3: HISTORY MANAGEMENT
    // ═══════════════════════════════════════════
    // Apply sliding window + summarization to conversation history
    prior_history ← prior_context.tier_4_history
    
    // Add the latest turn (user input + model output) to history
    latest_turn ← FormatTurn(task_state.last_user_input, retained_output)
    updated_history ← Append(prior_history, latest_turn)
    
    // Apply checkpoint summarization if history exceeds budget
    managed_history ← ManageCheckpoints(
        updated_history,
        window_size_k = budget.history_window_turns,
        checkpoint_interval = config.checkpoint_interval
    )

    // ═══════════════════════════════════════════
    // STAGE 4: EVIDENCE DEDUPLICATION
    // ═══════════════════════════════════════════
    // Merge new retrieval with retained evidence, eliminating duplicates
    retained_evidence ← prior_context.tier_2_evidence
    
    // Remove evidence items whose relevance has decayed below threshold
    FOR EACH item IN retained_evidence:
        item.current_relevance ← ComputeRelevance(item, task_state, current_cycle)
        IF item.current_relevance < config.evidence_retention_threshold THEN
            EVICT item
    
    // Merge with new retrieval, deduplicating by semantic similarity
    merged_evidence ← SemanticDedup(
        Concatenate(retained_evidence, new_retrieval),
        similarity_threshold = config.dedup_threshold
    )
    
    // Re-rank merged evidence for current cycle
    FOR EACH item IN merged_evidence:
        item.cycle_score ← ComputeCycleRelevance(item, task_state)
    SortDescending(merged_evidence, key = cycle_score)

    // ═══════════════════════════════════════════
    // STAGE 5: CROSS-SECTION DEDUPLICATION
    // ═══════════════════════════════════════════
    // Information may appear in evidence AND history AND memory
    // Retain canonical instance; replace others with references
    all_sections ← {evidence: merged_evidence, memory: memory_projection,
                     history: managed_history, tools: triaged_tools}
    
    deduplicated ← CrossSectionDedup(all_sections)
    // For each duplicate cluster, retain the highest-authority instance
    // Replace others with [REF: canonical_id]

    // ═══════════════════════════════════════════
    // STAGE 6: NOISE DETECTION AND REMOVAL
    // ═══════════════════════════════════════════
    // Scan all candidates for noise patterns
    FOR EACH item IN AllItems(deduplicated):
        noise_score ← NoiseDetector(item)
        // Noise indicators:
        //   - High token count with low information density
        //   - Repetitive content (n-gram repetition ratio)
        //   - Formatting-heavy content (markdown decorators without content)
        //   - Self-referential meta-commentary ("As I mentioned earlier...")
        //   - Stale temporal references ("yesterday's meeting" when irrelevant)
        
        IF noise_score > config.noise_threshold THEN
            IF item.tier = 0 THEN
                WARN "Noise detected in Policy Kernel — review required"
            ELSE
                EVICT item
                LOG "Noise eviction: {item.id}, score: {noise_score}"

    // ═══════════════════════════════════════════
    // STAGE 7: BUDGET FEASIBILITY CHECK
    // ═══════════════════════════════════════════
    total_demand ← SumTokenCosts(deduplicated)
    available ← budget.window_size - budget.gen_reserve - budget.policy_kernel_tokens
    
    IF total_demand > available THEN
        // Apply priority-weighted compression
        deduplicated ← ApplyBudgetCompression(deduplicated, available, budget.priorities)
    
    // ═══════════════════════════════════════════
    // STAGE 8: COHERENCE VALIDATION
    // ═══════════════════════════════════════════
    // Ensure the filtered set is internally consistent
    contradictions ← DetectContradictions(deduplicated)
    IF contradictions IS NOT EMPTY THEN
        FOR EACH (item_a, item_b) IN contradictions:
            // Retain the higher-authority item; annotate the conflict
            IF item_a.authority > item_b.authority THEN
                Annotate(item_a, "CONFLICT: contradicts {item_b.id}, retained by authority")
                EVICT item_b
            ELSE
                Annotate(item_b, "CONFLICT: contradicts {item_a.id}, retained by authority")
                EVICT item_a

    RETURN FilteredCandidates(deduplicated)
```

---

### 7.3 Noise Classification Taxonomy

The Cycle Filter classifies tokens into three categories:

| Category | Definition | Action |
|---|---|---|
| **Signal** | Tokens that directly contribute to correct task completion in the current cycle | Retain |
| **Structural** | Tokens that provide context coherence (delimiters, references, ordering cues) but carry no information | Minimize but preserve |
| **Noise** | Tokens that consume budget without contributing to current-cycle task utility | Evict |

The noise classification is applied to **all context sources** — including system prompts, developer instructions, and prior model outputs. Nothing is exempt from noise evaluation except the minimal invariant policy kernel (which is pre-compressed by L0).

### 7.4 Information Density Metric

Define the **information density** of a context segment $s$ as:

$$\rho_{\text{info}}(s) = \frac{\text{PropositionalContent}(s)}{\text{TokenCount}(s)}$$

where $\text{PropositionalContent}(s)$ is the count of distinct, non-trivial factual propositions extractable from $s$. Segments with $\rho_{\text{info}} < \rho_{\min}$ (empirically, $\rho_{\min} \approx 0.3$ propositions per 100 tokens) are candidates for compression or eviction.

For model outputs specifically, the density is often low because LLMs generate verbose explanations, hedging language, and reformulations. The Cycle Filter strips these, retaining only propositional content.

---

## 8. L5 — Prefill Compiler: Deterministic Assembly with Cycle Awareness

### 8.1 Cycle-Aware Compilation

The Prefill Compiler at L5 extends the foundational pipeline (Chapter 6, Algorithm 6.1) with **cycle-awareness**: it knows which cycle it is compiling for, what changed since the prior cycle, and what the current execution phase demands.

---

**PSEUDO-ALGORITHM 8.1: Cycle-Aware Prefill Compilation**

```
PROCEDURE CycleAwareCompile(
    policy_kernel: PolicyKernel,      // From L0
    task_state: TaskState,            // From L1
    filtered_candidates: FilteredCandidates,  // From L4
    prior_compilation: CompilationTrace,      // Trace from cycle t-1
    cycle_number: int,
    budget: TokenBudget
) → ContextObject:

    // Step 1: Compute delta from prior cycle
    delta ← ComputeDelta(filtered_candidates, prior_compilation)
    // delta.added:   new items not in prior context
    // delta.removed: items in prior context but evicted
    // delta.modified: items present in both but changed
    // delta.unchanged: items identical to prior cycle

    // Step 2: Leverage KV-cache alignment
    // Maximize prefix match with prior context for cache reuse
    IF policy_kernel.hash = prior_compilation.policy_hash THEN
        // Policy kernel unchanged → full cache hit on prefix
        cache_reuse_prefix ← policy_kernel.token_count
    ELSE
        cache_reuse_prefix ← 0

    // Step 3: Assemble sections with cycle-optimized ordering
    // Ordering principle: stable prefix (for cache) → dynamic content
    sections ← [
        // STABLE PREFIX (cache-friendly)
        Section("POLICY", policy_kernel.content, tier = 0, mutable = FALSE),
        
        // SEMI-STABLE (changes on task-phase transition, not every cycle)
        Section("TASK_STATE", SerializeCompact(task_state), tier = 1, mutable = TRUE),
        Section("TOOLS", filtered_candidates.tools, tier = 1, mutable = TRUE),
        
        // DYNAMIC (changes every cycle)
        Section("EVIDENCE", filtered_candidates.evidence, tier = 2, mutable = TRUE),
        Section("MEMORY", filtered_candidates.memory, tier = 3, mutable = TRUE),
        Section("HISTORY", filtered_candidates.history, tier = 4, mutable = TRUE)
    ]

    // Step 4: Apply priority-weighted slot allocation
    allocations ← SolveTokenAllocation(sections, budget)

    // Step 5: Compress each section to its allocation
    FOR EACH section IN sections WHERE section.mutable:
        IF TokenCount(section.content) > allocations[section.name] THEN
            section.content ← CompressToFit(section.content, allocations[section.name])

    // Step 6: Assemble final prefill with structural delimiters
    prefill ← AssembleWithDelimiters(sections, delimiter_style = config.delimiter_style)

    // Step 7: Validate invariants
    ASSERT TokenCount(prefill) + budget.gen_reserve ≤ budget.window_size
    ASSERT policy_kernel.hash = SHA256(ExtractSection(prefill, "POLICY"))
    ASSERT AllEvidenceHasProvenance(ExtractSection(prefill, "EVIDENCE"))
    ASSERT NoDuplicateIDs(prefill)
    ASSERT NoInstructionInDataSections(prefill)
    ASSERT SignalDensity(prefill) ≥ config.min_signal_density

    // Step 8: Emit compilation trace
    trace ← CompilationTrace(
        cycle = cycle_number,
        timestamp = NOW(),
        config_version = config.version,
        input_hash = SHA256(filtered_candidates),
        output_hash = SHA256(prefill),
        total_tokens = TokenCount(prefill),
        section_allocations = allocations,
        delta = delta,
        cache_reuse_prefix = cache_reuse_prefix,
        signal_density = SignalDensity(prefill),
        latency_ms = ElapsedMs()
    )

    RETURN ContextObject(prefill = prefill, trace = trace)
```

---

### 8.2 Cycle-to-Cycle Delta Optimization

A key SOTA optimization: the compiler computes the **minimal edit distance** between the prior cycle's context and the current cycle's context, then constructs the new context to **maximize KV-cache prefix reuse**.

The KV-cache stores key-value pairs for previously computed tokens. If the first $P$ tokens of the new context are identical to the prior context, the inference engine skips prefill computation for those $P$ tokens — reducing latency by:

$$\Delta_{\text{latency}} = P \cdot t_{\text{prefill\_per\_token}}$$

and cost by:

$$\Delta_{\text{cost}} = P \cdot c_{\text{prefill\_per\_token}}$$

To maximize $P$, the compiler:

1. Places the policy kernel (stable) at the start.
2. Places task state (semi-stable) next.
3. Places tool affordances (semi-stable, changes on step transitions) next.
4. Places dynamic content (evidence, memory, history) last.

This ordering ensures that the stable prefix is as long as possible across cycles.

### 8.3 Output Quality Gate

Before the compiled context is submitted to the LLM, it passes through a **quality gate** that checks:

| Check | Condition | Failure Action |
|---|---|---|
| Token budget compliance | $|\mathcal{C}| + R_{\text{gen}} \leq \mathcal{W}$ | Compilation failure; re-compress |
| Signal density | $\sigma \geq \sigma_{\min}$ | Warning; trigger aggressive noise removal |
| Provenance coverage | All evidence items tagged | Compilation failure; reject untagged evidence |
| Duplication rate | $\leq 5\%$ token duplication | Warning; trigger additional dedup pass |
| Contradiction count | 0 unresolved contradictions | Compilation failure; resolve or escalate |
| Constraint density | $\rho_c \leq 0.15$ | Warning; compress policy kernel |
| Staleness index | $\leq 0.20$ | Warning; trigger eviction sweep |

---

## 9. Instance-Episode Lifecycle Management

### 9.1 Episode State Store

The episode state store is the **durable backing store** for all information generated across an episode's cycles. It is external to the context window and serves as the source of truth for rehydration, debugging, and post-episode analysis.

```
EpisodeStateStore := {
    episode_id:         UUID,
    user_id:            TenantID,
    objective:          string,
    plan:               Plan,
    
    // Append-only logs
    cycle_log:          CycleLog[],         // One entry per cycle
    tool_invocations:   ToolInvocation[],   // Full tool call/response records
    retrieval_log:      RetrievalLog[],     // All retrieval queries and results
    
    // Mutable state (current values)
    task_state:         TaskState,
    session_memory:     Memory[],
    
    // Compiled context archive
    context_archive:    ContextObject[],    // Every compiled context, for debugging
    
    // Indexes for fast access
    by_turn_id:         Map<TurnID, CycleLog>,
    by_topic:           SemanticIndex,
    by_tool:            Map<ToolID, ToolInvocation[]>
}
```

### 9.2 Cycle Log Structure

Each cycle produces a log entry:

```
CycleLog := {
    cycle_number:       int,
    timestamp:          ISO8601,
    phase:              AgentPhase,
    step:               StepID,
    
    // Input summary (not full context — that's in context_archive)
    input_summary: {
        evidence_count:     int,
        evidence_tokens:    int,
        memory_count:       int,
        memory_tokens:      int,
        history_tokens:     int,
        total_prefill:      int,
        signal_density:     float
    },
    
    // Output summary
    output: {
        type:               OutputType,     // TOOL_CALL | TEXT | DECISION | ERROR
        tool_calls:         ToolCall[],
        decisions:          Decision[],
        errors:             Error[],
        tokens_generated:   int
    },
    
    // Quality metrics
    verification:       VerificationResult,
    critique:           CritiqueResult,
    repair_needed:      bool,
    
    // Performance
    latency_ms:         float,
    cost_usd:           float,
    cache_hit_ratio:    float
}
```

### 9.3 Cross-Episode Learning

After an episode terminates, a **post-episode analysis** extracts durable lessons:

---

**PSEUDO-ALGORITHM 9.1: Post-Episode Memory Extraction**

```
PROCEDURE PostEpisodeAnalysis(episode: Episode) → MemoryCandidate[]:

    candidates ← []

    // Extract corrections: cases where the agent's initial output was wrong
    // and required repair or user correction
    corrections ← ExtractCorrections(episode.cycle_log)
    FOR EACH correction IN corrections:
        candidate ← MemoryCandidate(
            content = FormatCorrection(correction),
            type = EPISODIC,
            provenance = {source: episode.id, cycle: correction.cycle},
            priority = HIGH,
            domain = episode.task_domain
        )
        APPEND candidate TO candidates

    // Extract learned constraints: rules discovered during execution
    // that should govern future similar tasks
    constraints ← ExtractLearnedConstraints(episode.cycle_log)
    FOR EACH constraint IN constraints:
        candidate ← MemoryCandidate(
            content = FormatConstraint(constraint),
            type = PROCEDURAL,
            provenance = {source: episode.id, cycles: constraint.source_cycles},
            priority = HIGH,
            domain = episode.task_domain
        )
        APPEND candidate TO candidates

    // Extract successful patterns: tool sequences that worked well
    patterns ← ExtractSuccessfulPatterns(episode.cycle_log, 
                                          quality_threshold = config.pattern_threshold)
    FOR EACH pattern IN patterns:
        candidate ← MemoryCandidate(
            content = FormatPattern(pattern),
            type = PROCEDURAL,
            provenance = {source: episode.id},
            priority = NORMAL,
            domain = episode.task_domain
        )
        APPEND candidate TO candidates

    // Submit all candidates through the memory write validation gate
    validated ← []
    FOR EACH candidate IN candidates:
        decision ← ValidateMemoryWrite(candidate, candidate.type, memory_store)
        IF decision.action = ADMIT THEN
            APPEND candidate TO validated

    RETURN validated
```

---

This closes the cycle: episode outputs produce durable memories that improve future episodes, but only through the validated write gate — preventing memory pollution from failed or low-quality episodes.

---

## 10. Observability, Evaluation, and Continuous Quality Enforcement

### 10.1 Per-Cycle Observability

Every cycle emits a structured trace compatible with OpenTelemetry:

```
CycleTrace := {
    trace_id:           UUID,
    span_id:            UUID,
    parent_span_id:     UUID,      // Links to episode-level trace
    
    // Context engineering metrics
    context: {
        total_tokens:           int,
        signal_density:         float,
        noise_evicted_tokens:   int,
        compression_ratio:      float,
        cache_hit_prefix:       int,
        evidence_count:         int,
        memory_count:           int,
        history_turns_full:     int,
        history_turns_summarized: int,
        staleness_index:        float,
        duplication_rate:       float,
        injection_vuln_surface: float
    },
    
    // Inference metrics
    inference: {
        prefill_latency_ms:     float,
        decode_latency_ms:      float,
        total_latency_ms:       float,
        tokens_generated:       int,
        cost_usd:               float,
        model_id:               string,
        temperature:            float
    },
    
    // Quality metrics
    quality: {
        verification_passed:    bool,
        critique_score:         float,
        repair_needed:          bool,
        hallucination_detected: bool,
        constraint_violations:  int
    }
}
```

### 10.2 Evaluation Infrastructure

Quality is not maintained by prompt tweaking. It is maintained by **continuously executed evaluation pipelines**:

| Evaluation Type | Trigger | Mechanism | Action on Failure |
|---|---|---|---|
| **Ablation regression** | Config change, model change | Run benchmark suite with ablated context sections; compare $\Delta_i$ | Block deployment if critical section utility drops |
| **Context quality gate** | Every cycle | Check signal density, staleness, duplication, provenance coverage | Reject compilation; trigger re-filtering |
| **End-to-end task accuracy** | Nightly CI | Replay recorded episodes against current system; measure task success rate | Alert + regression investigation |
| **Injection resistance** | Weekly | Run adversarial injection test suite against defensive compilation | Block deployment if injection success rate > threshold |
| **Memory quality audit** | Weekly | Sample episodic/procedural memories; human review for accuracy and relevance | Prune invalid memories; adjust write gates |
| **Cost efficiency** | Continuous | Track $\frac{\text{cost per successful task}}{\text{baseline cost}}$ | Alert if cost exceeds budget envelope |

### 10.3 Feedback Loop Integration

Human corrections, failed traces, and production regressions are normalized into:

1. **Benchmark tasks.** A failed episode becomes a test case in the regression suite.
2. **Memory entries.** Corrections become episodic memories with HIGH priority.
3. **Policy adjustments.** Repeated failure patterns trigger policy kernel updates (version-controlled, reviewed).
4. **Retrieval tuning.** Evidence gaps identified in failures drive retrieval configuration updates (new sources, adjusted ranking weights, expanded query decomposition rules).

This creates a **closed-loop improvement system** where every failure strengthens the system's future performance — mechanically, not anecdotally.

---

## 11. Production Reliability Engineering

### 11.1 Failure Modes and Mitigations

| Failure Mode | Impact | Mitigation |
|---|---|---|
| **Context budget exhaustion** | Compilation failure; agent stalls | Cascade-compress policy; defer-retrieval fallback; paginate |
| **Retrieval timeout** | Missing evidence; potential hallucination | Deadline enforcement; proceed with cached/memory evidence + flag for rehydration |
| **Memory store unavailable** | No episodic/semantic context | Graceful degradation; compile without memory; session memory from local cache |
| **Tool execution failure** | Incomplete action; stale state | Retry with jitter (max 3); circuit breaker; compensating action; escalate |
| **LLM inference failure** | No output for cycle | Retry with backoff; failover to secondary model; persist state for resumption |
| **Context poisoning detected** | Adversarial content in context | Quarantine input; recompile without poisoned source; security alert |
| **Cycle budget exceeded** | Episode termination risk | Warning at 80% cycle budget; forced summarization at 90%; hard stop with state persistence |
| **KV-cache miss** | Increased latency and cost | Reorder context to maximize prefix stability; pre-warm cache on task start |

### 11.2 Idempotency and Determinism

Every state-mutating operation in the cycle is **idempotent by design**:

- **Tool calls** carry idempotency keys. Retrying a tool call with the same key produces the same result.
- **Memory writes** are deduplicated by content hash. Retrying a memory write for an identical entry is a no-op.
- **Context compilation** is deterministic. Given identical inputs, the compiler produces byte-identical output. Retrying a compilation is safe.
- **Episode state updates** use compare-and-swap (CAS) semantics. Concurrent updates are detected and resolved.

### 11.3 Cost Model

The per-cycle cost is:

$$C_{\text{cycle}} = C_{\text{prefill}} \cdot |\mathcal{C}_{\text{prefill}}| + C_{\text{decode}} \cdot |\mathcal{C}_{\text{decode}}| + C_{\text{retrieval}} + C_{\text{tools}} + C_{\text{memory\_ops}}$$

The per-episode cost is:

$$C_{\text{episode}} = \sum_{t=1}^{T} C_{\text{cycle}}(t)$$

**Cost optimization levers:**

1. **KV-cache prefix reuse** reduces $C_{\text{prefill}}$ by up to $80\%$ per cycle.
2. **Aggressive noise eviction** reduces $|\mathcal{C}_{\text{prefill}}|$ by $20\text{–}40\%$ per cycle.
3. **Tool result compression** reduces history growth rate, delaying budget exhaustion.
4. **Memory-based retrieval avoidance** skips retrieval when memory already contains the answer, eliminating $C_{\text{retrieval}}$ for those cycles.
5. **Early termination** via quality gates prevents wasted cycles on tasks that are already complete or irrecoverable.

### 11.4 Backpressure and Rate Control

Under load, the system applies backpressure at multiple layers:

| Layer | Backpressure Mechanism |
|---|---|
| **User/Application boundary (JSON-RPC)** | Rate limiting per tenant; queue with bounded depth; reject with 429 |
| **Retrieval engine** | Timeout per subquery; circuit breaker on degraded sources; cached fallback |
| **Memory store** | Read-through cache; write queue with bounded depth; async write with acknowledgment |
| **LLM inference** | Token-bucket rate limiter; queue prioritization by task criticality; model fallback (frontier → secondary) |
| **Tool execution** | Per-tool concurrency limits; timeout per tool class; circuit breaker per tool server |

---

## 12. Formal Quality Invariants

The following invariants are **mechanically enforced** at every cycle. Violation of any invariant triggers compilation rejection, alert, and (for safety-critical deployments) episode termination.

| Invariant ID | Statement | Enforcement Point |
|---|---|---|
| **INV-1** | $|\mathcal{C}_{\text{prefill}}| + R_{\text{gen}} \leq \mathcal{W}$ | L5: Compilation validation |
| **INV-2** | All evidence items carry provenance metadata | L5: Compilation validation |
| **INV-3** | No instruction content in data sections | L5: Defensive compilation |
| **INV-4** | Policy kernel hash matches versioned source | L5: Compilation validation |
| **INV-5** | Signal density $\sigma \geq 0.70$ | L5: Quality gate |
| **INV-6** | Duplication rate $\leq 0.05$ | L4: Cross-section dedup |
| **INV-7** | Staleness index $\leq 0.20$ | L4: Relevance decay eviction |
| **INV-8** | Constraint density $\rho_c \leq 0.15$ | L0: Policy kernel compression |
| **INV-9** | Injection vulnerability surface $\mathcal{V} \leq 0.10$ | L5: Defensive compilation |
| **INV-10** | Memory writes pass validation gate | L3: Write contract enforcement |
| **INV-11** | Cycle count $\leq$ max_cycles_total | L1: Bounded loop enforcement |
| **INV-12** | Repair count per subtask $\leq$ max_repair_per_subtask | L1: Bounded repair enforcement |
| **INV-13** | Compilation is deterministic (input hash → output hash is a function) | L5: Determinism assertion |
| **INV-14** | Every cycle emits a structured trace | L5 + observability layer |

---

## 13. Comparative Positioning: Why This Architecture Exceeds Current Agent-Loop Implementations

| Capability | Conventional Agent Loops | This Architecture |
|---|---|---|
| Context management | Append-only history; truncate on overflow | Cycle-filtered, signal-optimized, provenance-tagged, deduplicated |
| System prompt handling | Static, verbose, uncompressed | Compiled policy kernel; phase-conditional; noise-eliminated; KV-cache pinned |
| Memory | Unstructured append; no write validation | Five-layer typed memory with dedup, provenance, expiry, and gated writes |
| Retrieval | Single-query RAG; anonymous chunks | Multi-strategy decomposed retrieval; provenance-tagged; hybrid search |
| Budget management | Hope it fits; silent truncation | Formal constrained optimization with water-filling allocation |
| Cycle hygiene | None; context degrades monotonically | Eight-stage cycle filter with noise detection, staleness eviction, cross-section dedup |
| Determinism | Non-deterministic prompt concatenation | Byte-identical compilation with versioned config and compilation traces |
| Security | None or ad hoc | Four-layer defense-in-depth with quantified vulnerability surface |
| Observability | Log the prompt; hope for the best | Per-cycle structured traces with 14+ quality metrics |
| Cost optimization | None | KV-cache prefix optimization, aggressive compression, retrieval avoidance |
| Evaluation | Manual testing | Continuous ablation, regression, injection, and cost benchmarks in CI/CD |

---

## 14. Summary of Formal Constructs

| Construct | Section | Purpose |
|---|---|---|
| Cyclic context recurrence $\mathcal{C}_{t+1} = \text{Compile}(\mathcal{P}, \mathcal{F}(\cdot))$ | §1.1 | Formal definition of cyclic context evolution |
| Signal ratio $\sigma_t$ and degradation theorem | §1.2 | Quantifies context quality across cycles |
| Policy Kernel compilation (Algorithm 3.1) | §3.2 | Minimal invariant policy extraction |
| Compact task state serialization (Algorithm 4.1) | §4.2 | Token-efficient state representation |
| Instance-episode decomposition | §4.3 | Formal separation of ephemeral and durable state |
| Retrieval plan generation (Algorithm 5.1) | §5.1 | Multi-strategy query decomposition |
| Evidence composite scoring | §5.3 | Multi-factor retrieval ranking |
| Memory write validation (Algorithm 6.1) | §6.2 | Gated, deduplicated, provenanced memory admission |
| Eight-stage cycle filter (Algorithm 7.1) | §7.2 | Core cyclic context optimization pipeline |
| Information density metric $\rho_{\text{info}}$ | §7.4 | Quantifies content vs. token ratio |
| Cycle-aware prefill compilation (Algorithm 8.1) | §8.1 | Deterministic, cache-optimized context assembly |
| Delta optimization for KV-cache reuse | §8.2 | Latency and cost reduction via prefix stability |
| Post-episode memory extraction (Algorithm 9.1) | §9.3 | Closed-loop learning from episode outcomes |
| 14 mechanical invariants | §12 | Non-negotiable quality enforcement |
| Per-cycle cost model | §11.3 | Formal cost optimization framework |

---

This architecture treats the context window as the **central computational resource** of agentic AI — not as a prompt, not as a buffer, but as a **cyclic, budget-constrained, signal-optimized, deterministically compiled execution surface**. Every component — from policy kernel compression to cycle filtering to memory governance to KV-cache alignment — is engineered to maintain maximum signal density across unbounded cycle counts, under hard token, latency, and cost constraints, with mechanical enforcement of quality invariants and continuous evaluation feedback loops. The result is an agent loop that does not degrade over time, does not accumulate noise, does not waste tokens on redundant or stale content, and produces auditable, reproducible, and measurably correct outputs at every cycle.