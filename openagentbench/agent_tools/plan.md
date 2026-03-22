

# Agent Tools Sub-Module — SOTA+ Tool Orchestration Architecture

## Unified Protocol Stack, Multi-Model MIMO, Least-Privilege Execution, Per-User Isolation

---

## 1. Module Position and Integration Boundary

### 1.1 Architectural Context

The `agent_tools` sub-module is the **actuation layer** of the agentic platform. It converts agent intent into observable, governed, fault-tolerant world-state transitions. It sits downstream of:

- **`agent_data`** — provides per-user PostgreSQL-partitioned storage, schema management, and typed data access.
- **`agentic_retrieve`** — provides provenance-tagged, hybrid-ranked evidence fragments under latency budgets.
- **`agentic_memory`** — provides five-layer memory hierarchy (working, session, episodic, semantic, procedural) with validated promotion pipelines.

The `agent_tools` module does **not** duplicate any capability of upstream modules. It consumes their outputs (evidence, memory, data) and exposes a unified, typed, protocol-polymorphic surface through which agents actuate tools, invoke external services, delegate to other agents, and interact with computer environments.

### 1.2 Module Responsibilities

| Responsibility | Scope |
|---|---|
| **Tool Registry** | Discovery, registration, schema management, versioning, health tracking across all tool types |
| **Protocol Routing** | Unified dispatch across function calls, MCP, JSON-RPC, gRPC, A2A, SDK wrappers |
| **Invocation Lifecycle** | Six-phase deterministic lifecycle: Request → Validate → Authorize → Execute → Verify → Return |
| **Idempotency Engine** | Caller-generated key management, deduplication store, at-least-once + idempotent receiver semantics |
| **Authorization Gate** | Caller-scoped credential propagation, least-privilege enforcement, approval workflows |
| **Timeout Classification** | Four-tier timeout classes with deadline propagation and async polling |
| **Composite Orchestration** | Multi-step tool chains as DAG executions with compensating actions |
| **Computer Tools** | Browser automation, desktop control, vision-based interaction surfaces |
| **A2A Delegation** | Agent-to-agent task delegation with typed contracts and isolation |
| **SDK Compatibility** | Native integration with `python-openai`, vLLM, Claude MCP, Google A2A |
| **Observability** | Traces, metrics, audit logs, cost attribution, health monitoring |
| **Cache Layer** | Tool result caching with provenance-aware invalidation |
| **Testing Infrastructure** | Schema-driven test generation, behavioral contracts, chaos testing |

### 1.3 Integration Contracts

| Boundary | Protocol | Direction | Contract |
|---|---|---|---|
| Agent Orchestrator → Tools | gRPC (internal) | Invoke | Typed invocation with deadline, idempotency key, caller token |
| Application → Tools | JSON-RPC 2.0 | Invoke + Admin | External tool invocation, registry management |
| Agent → Tool Discovery | MCP | Discovery | Schema announcement, capability negotiation, change subscription |
| Agent → Agent (A2A) | Google A2A Protocol | Delegate | Task cards, artifact exchange, status polling |
| Tools → Data Layer | gRPC (internal) | Read/Write | Per-user partitioned data access via `agent_data` |
| Tools → Retrieval | gRPC (internal) | Read | Evidence retrieval via `agentic_retrieve` |
| Tools → Memory | gRPC (internal) | Read/Write | Memory read for context, write-back for learned tool patterns |
| Tools → External APIs | HTTPS/gRPC | Invoke | SDK-wrapped external service calls |
| Tools → Computer Env | Browser CDP / OS APIs | Control | Browser automation, desktop interaction, vision capture |

### 1.4 Model Compatibility Matrix

| Model Interface | Tool Integration Mode | Supported Operations |
|---|---|---|
| `python-openai` (chat) | Function calling via `tools` parameter | Tool schema injection, structured output parsing, parallel tool calls |
| `python-openai` (completion) | Manual tool-call extraction from completion text | Pattern-matched tool invocation |
| `python-openai` (responses API) | Native tool_use with built-in tools | Web search, code interpreter, file search |
| vLLM (chat) | OpenAI-compatible function calling | Same as `python-openai` chat |
| vLLM (completion) | Guided generation with tool grammars | Constrained decoding for tool call format |
| Claude MCP | Native MCP tool/resource/prompt surfaces | Full MCP lifecycle |
| Google A2A | Agent card exchange, task delegation | Inter-agent tool delegation |
| Multi-modal models | Vision-augmented tool selection | Screenshot analysis, UI element detection |

---

## 2. Tool Type Taxonomy

### 2.1 Formal Classification

Every tool registered in the system belongs to exactly one **type class** $\mathcal{C}_t$, which determines its protocol binding, lifecycle constraints, and governance requirements:

$$
\mathcal{C}_t \in \{\text{FUNCTION}, \text{METHOD}, \text{MCP\_TOOL}, \text{MCP\_RESOURCE}, \text{MCP\_PROMPT}, \text{JSON\_RPC}, \text{GRPC}, \text{A2A\_AGENT}, \text{SDK\_WRAPPED}, \text{BROWSER}, \text{DESKTOP}, \text{VISION}, \text{COMPOSITE}\}
$$

### 2.2 Type Class Properties

| Type Class | Protocol | Statefulness | Latency Tier | Mutation Risk | Isolation |
|---|---|---|---|---|---|
| FUNCTION | In-process call | Stateless | Interactive | Read-only | Process-local |
| METHOD | In-process call | Instance-bound | Interactive | Varies | Object-scoped |
| MCP_TOOL | MCP (stdio/SSE/streamable-HTTP) | Server-defined | Standard | Varies | MCP session |
| MCP_RESOURCE | MCP resource read | Stateless | Interactive | Read-only | MCP session |
| MCP_PROMPT | MCP prompt surface | Stateless | Interactive | Read-only | MCP session |
| JSON_RPC | JSON-RPC 2.0 over HTTP | Stateless/Stateful | Standard | Varies | Request-scoped |
| GRPC | gRPC/Protobuf | Stateless/Stateful | Interactive | Varies | Channel-scoped |
| A2A_AGENT | Google A2A Protocol | Stateful (task) | Long-Running/Async | Varies | Task-isolated |
| SDK_WRAPPED | SDK-specific (OpenAI, etc.) | SDK-managed | Standard | Varies | SDK session |
| BROWSER | Chrome DevTools Protocol | Stateful (page) | Standard/Long | Write | Browser context |
| DESKTOP | OS automation APIs | Stateful (session) | Standard | Write | Desktop session |
| VISION | Image analysis pipeline | Stateless | Standard | Read-only | Request-scoped |
| COMPOSITE | DAG of atomic tools | Composite state | Long-Running | Aggregate | Workspace-isolated |

### 2.3 Formal Tool Definition

Each tool $\mathcal{T}$ is defined as a typed record:

$$
\mathcal{T} = \langle \texttt{id}, \texttt{v}, \mathcal{C}_t, \Sigma_{\text{in}}, \Sigma_{\text{out}}, \Sigma_{\text{err}}, \mathcal{C}_{\text{auth}}, \tau, \mathcal{I}, \mathcal{M}, \mathcal{O}, \mathcal{SE}, h, \text{prov} \rangle
$$

where:

| Symbol | Type | Description |
|---|---|---|
| $\texttt{id}$ | `NamespacedID` | Globally unique, namespace-scoped identifier |
| $\texttt{v}$ | `SemVer` | Semantic version of tool contract |
| $\mathcal{C}_t$ | `TypeClass` | Classification from taxonomy above |
| $\Sigma_{\text{in}}$ | `JSONSchema` | Validated input schema with semantic annotations |
| $\Sigma_{\text{out}}$ | `JSONSchema` | Typed structured output schema with pagination and provenance |
| $\Sigma_{\text{err}}$ | `ErrorEnvelope` | Classified error codes with retryability flags |
| $\mathcal{C}_{\text{auth}}$ | `AuthContract` | Required scopes, credential propagation rules |
| $\tau$ | `TimeoutClass` | One of $\{\tau_I, \tau_S, \tau_L, \tau_A\}$ |
| $\mathcal{I}$ | `IdempotencySpec` | Key derivation, deduplication window, at-least-once semantics |
| $\mathcal{M}$ | `MutationClass` | $\{R, W_r, W_i\}$ — read-only, write-reversible, write-irreversible |
| $\mathcal{O}$ | `ObservabilityContract` | Trace propagation, metric emission, audit log schema |
| $\mathcal{SE}$ | `SideEffectManifest` | Declared resource mutations with operation types |
| $h$ | `HealthScore` | Runtime health in $[0, 1]$, updated by monitoring |
| $\text{prov}$ | `RegistrationProvenance` | Source MCP server, registration timestamp, registrar identity |

---

## 3. Tool Registry Architecture

### 3.1 Registry Data Model

```
TABLE: tool_registry
  COLUMNS:
    tool_id             TEXT        PRIMARY KEY   -- namespace:name
    version             TEXT        NOT NULL      -- semver
    type_class          ENUM(...)   NOT NULL
    input_schema        JSONB       NOT NULL      -- JSON Schema
    output_schema       JSONB       NOT NULL
    error_schema        JSONB       NOT NULL
    auth_contract       JSONB       NOT NULL
    timeout_class       ENUM(interactive, standard, long_running, async)
    idempotency_spec    JSONB       NOT NULL
    mutation_class      ENUM(read_only, write_reversible, write_irreversible)
    side_effect_manifest JSONB      NULLABLE
    observability_contract JSONB    NOT NULL
    health_score        FLOAT       DEFAULT 1.0
    status              ENUM(active, deprecated, quarantined, unavailable)
    deprecation_notice  JSONB       NULLABLE
    source_endpoint     TEXT        NOT NULL      -- MCP server, gRPC endpoint, etc.
    source_type         ENUM(mcp, grpc, jsonrpc, function, a2a, sdk, browser, desktop)
    token_cost_estimate INT         NOT NULL      -- estimated tokens for schema in context
    compressed_description TEXT     NOT NULL      -- one-line for Tier 0 index
    registered_at       TIMESTAMPTZ DEFAULT NOW()
    updated_at          TIMESTAMPTZ DEFAULT NOW()
    contract_tests_hash TEXT        NOT NULL      -- hash of passing test suite
  INDEXES:
    BTREE(type_class, status)
    BTREE(mutation_class, status)
    GIN(input_schema)                             -- for schema-based capability queries
    BTREE(health_score DESC) WHERE status = 'active'
    BTREE(token_cost_estimate ASC)

TABLE: tool_idempotency_store
  COLUMNS:
    idempotency_key     TEXT        PRIMARY KEY
    tool_id             TEXT        NOT NULL
    caller_session_id   UUID        NOT NULL
    result_envelope     JSONB       NOT NULL
    created_at          TIMESTAMPTZ DEFAULT NOW()
    expires_at          TIMESTAMPTZ NOT NULL
  INDEXES:
    BTREE(expires_at)                             -- TTL cleanup
    BTREE(tool_id, caller_session_id)

TABLE: tool_invocation_audit
  PARTITION BY RANGE(created_at)
  COLUMNS:
    audit_id            UUID        PRIMARY KEY
    trace_id            TEXT        NOT NULL
    tool_id             TEXT        NOT NULL
    tool_version        TEXT        NOT NULL
    caller_id           UUID        NOT NULL      -- originating user uu_id
    agent_id            UUID        NOT NULL
    session_id          UUID        NULLABLE
    operation           TEXT        NOT NULL
    input_hash          BYTEA       NOT NULL      -- SHA-256 of canonical input
    mutation_class      ENUM(...)
    auth_decision       ENUM(allow, deny, requires_approval)
    approval_ticket_id  UUID        NULLABLE
    status              ENUM(success, error, timeout, pending)
    error_code          TEXT        NULLABLE
    latency_ms          INT         NOT NULL
    compute_cost        FLOAT       DEFAULT 0
    token_cost          INT         DEFAULT 0
    side_effects        JSONB       NULLABLE
    created_at          TIMESTAMPTZ DEFAULT NOW()
  INDEXES:
    BTREE(caller_id, created_at DESC)
    BTREE(tool_id, created_at DESC)
    BTREE(trace_id)
    BTREE(session_id, created_at DESC) WHERE session_id IS NOT NULL
    BTREE(status) WHERE status != 'success'

TABLE: tool_approval_tickets
  COLUMNS:
    ticket_id           UUID        PRIMARY KEY
    tool_id             TEXT        NOT NULL
    operation           TEXT        NOT NULL
    params_redacted     JSONB       NOT NULL
    requested_by        UUID        NOT NULL      -- caller uu_id
    agent_id            UUID        NOT NULL
    status              ENUM(pending, approved, denied, auto_denied, expired)
    resolution_by       UUID        NULLABLE
    resolution_at       TIMESTAMPTZ NULLABLE
    created_at          TIMESTAMPTZ DEFAULT NOW()
    expires_at          TIMESTAMPTZ NOT NULL
    escalation_level    INT         DEFAULT 0
    dry_run_result      JSONB       NULLABLE
    metadata            JSONB
  INDEXES:
    BTREE(status, expires_at) WHERE status = 'pending'
    BTREE(requested_by, created_at DESC)

TABLE: tool_result_cache (UNLOGGED)
  COLUMNS:
    cache_key           TEXT        PRIMARY KEY   -- HASH(tool_id, version, canonical_params)
    tool_id             TEXT        NOT NULL
    result_data         JSONB       NOT NULL
    created_at          TIMESTAMPTZ DEFAULT NOW()
    expires_at          TIMESTAMPTZ NOT NULL
    hit_count           INT         DEFAULT 0
  INDEXES:
    BTREE(expires_at)
    BTREE(tool_id)
```

### 3.2 Tool Admission Predicate

```
ALGORITHM: ToolAdmissionGate
INPUT:  tool_descriptor: ToolDescriptor, contract_tests: TestSuite
OUTPUT: admission_result: AdmissionResult

1.  violations ← EMPTY_LIST

// ═══════════════════════════════════════════
// CONJUNCT 1: Schema Validity
// ═══════════════════════════════════════════
2.  IF NOT JSONSchemaValid(tool_descriptor.input_schema):
3.      APPEND "Invalid input schema" TO violations
4.  IF NOT JSONSchemaValid(tool_descriptor.output_schema):
5.      APPEND "Invalid output schema" TO violations
6.  IF NOT ErrorEnvelopeConformant(tool_descriptor.error_schema):
7.      APPEND "Error schema does not conform to envelope spec" TO violations

// ═══════════════════════════════════════════
// CONJUNCT 2: Version Registration
// ═══════════════════════════════════════════
8.  IF NOT SemVerValid(tool_descriptor.version):
9.      APPEND "Invalid semantic version" TO violations
10. existing ← ToolRegistry.lookup(tool_descriptor.id, tool_descriptor.version)
11. IF existing IS NOT NULL AND existing.contract_tests_hash == HASH(contract_tests):
12.     RETURN AdmissionResult(ALREADY_REGISTERED)

// ═══════════════════════════════════════════
// CONJUNCT 3: Auth Policy Defined
// ═══════════════════════════════════════════
13. IF tool_descriptor.auth_contract IS NULL OR
       tool_descriptor.auth_contract.required_scopes IS EMPTY:
14.     APPEND "Auth contract undefined or empty scopes" TO violations

// ═══════════════════════════════════════════
// CONJUNCT 4: Timeout Class Assigned
// ═══════════════════════════════════════════
15. IF tool_descriptor.timeout_class NOT IN {interactive, standard, long_running, async}:
16.     APPEND "Timeout class not assigned" TO violations

// ═══════════════════════════════════════════
// CONJUNCT 5: Idempotency Specified
// ═══════════════════════════════════════════
17. IF tool_descriptor.mutation_class != read_only AND
       tool_descriptor.idempotency_spec IS NULL:
18.     APPEND "Mutating tool without idempotency spec" TO violations

// ═══════════════════════════════════════════
// CONJUNCT 6: Mutation Class Labeled
// ═══════════════════════════════════════════
19. IF tool_descriptor.mutation_class IS NULL:
20.     APPEND "Mutation class not declared" TO violations
21. IF tool_descriptor.mutation_class IN {write_reversible, write_irreversible} AND
       tool_descriptor.side_effect_manifest IS NULL:
22.     APPEND "Write tool without side-effect manifest" TO violations

// ═══════════════════════════════════════════
// CONJUNCT 7: Observability Instrumented
// ═══════════════════════════════════════════
23. IF tool_descriptor.observability_contract IS NULL:
24.     APPEND "Observability contract missing" TO violations

// ═══════════════════════════════════════════
// CONJUNCT 8: Contract Tests Passing
// ═══════════════════════════════════════════
25. IF contract_tests IS NOT NULL:
26.     test_result ← RunContractTests(contract_tests, tool_descriptor)
27.     IF NOT test_result.all_passed:
28.         APPEND "Contract tests failed: " + test_result.failures TO violations

// ═══════════════════════════════════════════
// VERDICT
// ═══════════════════════════════════════════
29. IF |violations| > 0:
30.     RETURN AdmissionResult(REJECTED, violations=violations)
31. ELSE:
32.     RETURN AdmissionResult(ADMITTED)
```

$$
\texttt{Admit}(\mathcal{T}) \iff \bigwedge_{i=1}^{8} \texttt{Conjunct}_i(\mathcal{T}) = \top
$$

---

## 4. Protocol Router — Unified Dispatch Engine

### 4.1 Routing Architecture

The Protocol Router is the single point of dispatch for all tool invocations. It accepts a protocol-agnostic `ToolInvocationRequest` and routes to the appropriate protocol adapter based on the tool's type class.

$$
\texttt{Route}: \texttt{ToolInvocationRequest} \times \mathcal{C}_t \rightarrow \texttt{ProtocolAdapter}
$$

### 4.2 Protocol Adapter Registry

| Type Class | Protocol Adapter | Transport | Serialization |
|---|---|---|---|
| FUNCTION | `InProcessAdapter` | Direct call | Native objects |
| METHOD | `InProcessAdapter` | Direct call on instance | Native objects |
| MCP_TOOL | `MCPClientAdapter` | stdio / SSE / Streamable HTTP | JSON |
| MCP_RESOURCE | `MCPResourceAdapter` | stdio / SSE / Streamable HTTP | JSON |
| MCP_PROMPT | `MCPPromptAdapter` | stdio / SSE / Streamable HTTP | JSON |
| JSON_RPC | `JSONRPCAdapter` | HTTP/1.1 or HTTP/2 | JSON |
| GRPC | `GRPCAdapter` | HTTP/2 (gRPC) | Protobuf |
| A2A_AGENT | `A2AClientAdapter` | HTTPS | JSON (A2A spec) |
| SDK_WRAPPED | `SDKAdapter` | SDK-specific | SDK-specific |
| BROWSER | `BrowserAdapter` | Chrome DevTools Protocol | JSON |
| DESKTOP | `DesktopAdapter` | OS-specific APIs | Native |
| VISION | `VisionPipelineAdapter` | In-process + model call | Tensors / JSON |
| COMPOSITE | `CompositeDAGAdapter` | Internal orchestration | Mixed |

### 4.3 Unified Dispatch Algorithm

```
ALGORITHM: UnifiedToolDispatch
INPUT:  request: ToolInvocationRequest, registry: ToolRegistry,
        idempotency_store: IdempotencyStore, auth_engine: AuthEngine,
        cache: ToolResultCache
OUTPUT: response: ToolInvocationResponse

1.  span ← StartTraceSpan("tool.dispatch", trace_id=request.trace_id,
                            tool_id=request.tool_id)

// ═══════════════════════════════════════════
// PHASE 0: TOOL RESOLUTION
// ═══════════════════════════════════════════
2.  tool ← registry.Resolve(request.tool_id, request.version_spec)
3.  IF tool IS NULL:
4.      span.SetStatus(TOOL_NOT_FOUND)
5.      RETURN ErrorEnvelope(code=TOOL_NOT_FOUND, retryable=FALSE)
6.  IF tool.status == quarantined:
7.      RETURN ErrorEnvelope(code=TOOL_QUARANTINED, retryable=FALSE)
8.  IF tool.status == unavailable:
9.      RETURN ErrorEnvelope(code=UPSTREAM_FAILURE, retryable=TRUE,
10.                           retry_after_ms=5000)
11. IF tool.status == deprecated:
12.     span.AddEvent("deprecated_tool_invoked", tool.deprecation_notice)

// ═══════════════════════════════════════════
// PHASE 1: INPUT VALIDATION
// ═══════════════════════════════════════════
13. validation ← ValidateAgainstSchema(request.params, tool.input_schema)
14. IF NOT validation.valid:
15.     span.SetStatus(INVALID_INPUT)
16.     RETURN ErrorEnvelope(code=INVALID_INPUT, details=validation.errors,
17.                           retryable=FALSE)

// ═══════════════════════════════════════════
// PHASE 2: IDEMPOTENCY CHECK
// ═══════════════════════════════════════════
18. IF request.idempotency_key IS NOT NULL:
19.     prior ← idempotency_store.Lookup(request.tool_id, request.idempotency_key)
20.     IF prior IS NOT NULL AND prior.age < tool.idempotency_spec.dedup_window:
21.         span.SetStatus(IDEMPOTENT_REPLAY)
22.         EmitMetric("tool.idempotency.replay", tool_id=tool.id)
23.         RETURN prior.result_envelope

// ═══════════════════════════════════════════
// PHASE 3: AUTHORIZATION
// ═══════════════════════════════════════════
24. auth_decision ← auth_engine.Evaluate(
25.     caller_token=request.caller_token,
26.     required_scopes=tool.auth_contract.required_scopes,
27.     resource_scope=ExtractResourceScope(request.params, tool.side_effect_manifest),
28.     mutation_class=tool.mutation_class
29. )
30. IF auth_decision == DENY:
31.     EmitAuditLog(DENIED, request, tool)
32.     RETURN ErrorEnvelope(code=AUTHORIZATION_DENIED, retryable=FALSE)
33. IF auth_decision == REQUIRES_APPROVAL:
34.     ticket ← CreateApprovalTicket(request, tool)
35.     RETURN PendingEnvelope(approval_ticket_id=ticket.id,
36.                             poll_endpoint=BuildPollEndpoint(ticket.id))

// ═══════════════════════════════════════════
// PHASE 4: CACHE CHECK (read-only tools only)
// ═══════════════════════════════════════════
37. IF tool.mutation_class == read_only:
38.     cache_key ← DeriveCanonicalHash(tool.id, tool.version, request.params)
39.     cached ← cache.Get(cache_key)
40.     IF cached IS NOT NULL AND cached.age < tool.cache_ttl:
41.         EmitMetric("tool.cache.hit", tool_id=tool.id)
42.         RETURN SuccessEnvelope(data=cached.result_data, source="cache")

// ═══════════════════════════════════════════
// PHASE 5: PROTOCOL-SPECIFIC EXECUTION
// ═══════════════════════════════════════════
43. adapter ← ProtocolAdapterRegistry.Get(tool.type_class)
44. deadline ← ComputeEffectiveDeadline(tool.timeout_class, request.deadline)

45. execution_result ← adapter.Execute(
46.     tool=tool,
47.     params=request.params,
48.     caller_token=NarrowCredential(request.caller_token, tool.auth_contract),
49.     deadline=deadline,
50.     trace_context=span.context,
51.     session_context=request.session_context
52. )

53. IF execution_result.timed_out:
54.     span.SetStatus(DEADLINE_EXCEEDED)
55.     RETURN ErrorEnvelope(code=DEADLINE_EXCEEDED, retryable=TRUE)
56. IF execution_result.error IS NOT NULL:
57.     span.SetStatus(EXECUTION_FAILED)
58.     RETURN ErrorEnvelope(code=MapErrorCode(execution_result.error),
59.                           retryable=IsTransient(execution_result.error))

// ═══════════════════════════════════════════
// PHASE 6: OUTPUT VALIDATION
// ═══════════════════════════════════════════
60. output_valid ← ValidateAgainstSchema(execution_result.value, tool.output_schema)
61. IF NOT output_valid.valid:
62.     EmitAlert(TOOL_OUTPUT_SCHEMA_VIOLATION, tool.id)
63.     span.SetStatus(OUTPUT_SCHEMA_VIOLATION)
64.     RETURN ErrorEnvelope(code=INTERNAL_TOOL_ERROR, retryable=FALSE)

// ═══════════════════════════════════════════
// PHASE 7: SIDE-EFFECT VERIFICATION (write tools)
// ═══════════════════════════════════════════
65. IF tool.mutation_class != read_only:
66.     side_effect_check ← VerifySideEffects(
67.         tool, request.params, execution_result.value,
68.         tool.side_effect_manifest
69.     )
70.     IF NOT side_effect_check.consistent:
71.         // Attempt rollback for reversible tools
72.         IF tool.mutation_class == write_reversible:
73.             compensation ← tool.DeriveCompensation(request.params, execution_result.value)
74.             ExecuteCompensation(compensation, deadline)
75.         EmitAlert(SIDE_EFFECT_INCONSISTENCY, tool.id)
76.         RETURN ErrorEnvelope(code=INTERNAL_TOOL_ERROR)

// ═══════════════════════════════════════════
// PHASE 8: STORE RESULTS AND RETURN
// ═══════════════════════════════════════════
77. // Idempotency store write
78. IF request.idempotency_key IS NOT NULL:
79.     idempotency_store.Store(
80.         tool_id=tool.id, key=request.idempotency_key,
81.         result=SuccessEnvelope(data=execution_result.value),
82.         ttl=tool.idempotency_spec.dedup_window
83.     )

84. // Cache write (read-only tools)
85. IF tool.mutation_class == read_only:
86.     cache.Put(cache_key, execution_result.value, ttl=tool.cache_ttl)

87. // Audit log
88. EmitAuditLog(SUCCESS, request, tool, execution_result)

89. // Metrics
90. EmitMetric("tool.invocation.latency_ms", value=execution_result.duration_ms,
91.             labels={tool_id: tool.id, status: "success", mutation: tool.mutation_class})
92. EmitMetric("tool.invocation.count", labels={tool_id: tool.id, status: "success"})

93. span.SetStatus(SUCCESS)
94. span.End()

95. RETURN SuccessEnvelope(
96.     data=execution_result.value,
97.     provenance={tool_id: tool.id, version: tool.version, executed_at: NOW()},
98.     execution_metadata={latency_ms: execution_result.duration_ms,
99.                          cost: execution_result.resource_cost}
100.)
```

---

## 5. Protocol Adapters — Per-Type-Class Execution

### 5.1 MCP Client Adapter

```
ALGORITHM: MCPClientAdapter.Execute
INPUT:  tool: ToolDescriptor, params: JSON, caller_token: Token,
        deadline: Timestamp, trace_context: TraceContext,
        session_context: SessionContext
OUTPUT: execution_result: ExecutionResult

1.  // Resolve MCP connection
2.  mcp_session ← MCPConnectionPool.GetOrCreate(
3.      endpoint=tool.source_endpoint,
4.      transport=tool.mcp_transport,  -- stdio, sse, streamable_http
5.      session_id=session_context.session_id
6.  )

7.  // Capability check
8.  IF NOT mcp_session.HasCapability("tools"):
9.      RETURN ExecutionResult(error=TOOL_NOT_AVAILABLE)

10. // Build MCP tool call
11. mcp_request ← MCPToolCallRequest {
12.     method: "tools/call",
13.     params: {
14.         name: tool.mcp_tool_name,
15.         arguments: params
16.     },
17.     _meta: {
18.         progressToken: GenerateProgressToken()  -- for long-running tools
19.     }
20. }

21. // Execute with deadline
22. TRY:
23.     mcp_response ← mcp_session.SendWithDeadline(mcp_request, deadline)
24.
25.     IF mcp_response.isError:
26.         RETURN ExecutionResult(
27.             error=MapMCPError(mcp_response.error),
28.             duration_ms=elapsed()
29.         )
30.
31.     // Parse MCP content array into structured output
32.     structured_output ← ParseMCPContent(mcp_response.content, tool.output_schema)
33.     RETURN ExecutionResult(
34.         value=structured_output,
35.         duration_ms=elapsed(),
36.         resource_cost=EstimateMCPCost(mcp_response)
37.     )
38.
39. ON_TIMEOUT:
40.     RETURN ExecutionResult(timed_out=TRUE, duration_ms=elapsed())
41. ON_ERROR(e):
42.     RETURN ExecutionResult(error=e, duration_ms=elapsed())
```

### 5.2 A2A Client Adapter (Google Agent-to-Agent Protocol)

```
ALGORITHM: A2AClientAdapter.Execute
INPUT:  tool: ToolDescriptor, params: JSON, caller_token: Token,
        deadline: Timestamp, trace_context: TraceContext,
        session_context: SessionContext
OUTPUT: execution_result: ExecutionResult

1.  // Resolve A2A agent endpoint from tool descriptor
2.  agent_card ← A2ADiscovery.ResolveAgentCard(tool.source_endpoint)
3.  IF agent_card IS NULL:
4.      RETURN ExecutionResult(error=TOOL_NOT_FOUND)

5.  // Verify capability match
6.  IF NOT A2ACapabilityMatch(params, agent_card.skills):
7.      RETURN ExecutionResult(error=PRECONDITION_FAILED)

8.  // Create A2A task
9.  task_request ← A2ATaskRequest {
10.     id: GenerateTaskID(),
11.     message: {
12.         role: "user",
13.         parts: BuildA2AParts(params, tool.input_schema)
14.     },
15.     metadata: {
16.         trace_id: trace_context.trace_id,
17.         caller_id: caller_token.identity,
18.         deadline: deadline
19.     }
20. }

21. // Submit task
22. task_response ← A2AClient.SendTask(agent_card.endpoint, task_request, caller_token)

23. // Handle based on task state
24. SWITCH task_response.status.state:

25.   CASE "completed":
26.     artifacts ← task_response.artifacts
27.     structured_output ← ParseA2AArtifacts(artifacts, tool.output_schema)
28.     RETURN ExecutionResult(value=structured_output, duration_ms=elapsed())

29.   CASE "failed":
30.     RETURN ExecutionResult(error=MapA2AError(task_response.status))

31.   CASE "working", "input-required":
32.     // Long-running or interactive: poll with deadline
33.     RETURN PollA2ATask(task_response.id, agent_card.endpoint,
34.                         deadline, caller_token, tool.output_schema)

35.   CASE "canceled":
36.     RETURN ExecutionResult(error=TASK_CANCELED)

ALGORITHM: PollA2ATask
INPUT:  task_id: string, endpoint: URL, deadline: Timestamp,
        caller_token: Token, output_schema: JSONSchema
OUTPUT: execution_result: ExecutionResult

1.  poll_interval ← A2A_INITIAL_POLL_MS  -- typically 500ms

2.  WHILE NOW() < deadline:
3.      status ← A2AClient.GetTaskStatus(endpoint, task_id, caller_token)
4.
5.      SWITCH status.state:
6.        CASE "completed":
7.          artifacts ← A2AClient.GetTaskArtifacts(endpoint, task_id, caller_token)
8.          RETURN ExecutionResult(
9.              value=ParseA2AArtifacts(artifacts, output_schema),
10.             duration_ms=elapsed()
11.         )
12.       CASE "failed":
13.         RETURN ExecutionResult(error=MapA2AError(status))
14.       CASE "input-required":
15.         // Escalate: requires agent loop to provide additional input
16.         RETURN ExecutionResult(
17.             error=INPUT_REQUIRED,
18.             details=status.message,
19.             retryable=TRUE
20.         )
21.       CASE "working":
22.         EmitProgress(task_id, status.progress)
23.
24.     Sleep(poll_interval)
25.     poll_interval ← MIN(poll_interval * 1.5, A2A_MAX_POLL_MS)

26. // Deadline exceeded
27. A2AClient.CancelTask(endpoint, task_id, caller_token)
28. RETURN ExecutionResult(timed_out=TRUE, duration_ms=elapsed())
```

### 5.3 SDK Wrapper Adapter (OpenAI / vLLM Compatibility)

```
ALGORITHM: SDKAdapter.Execute
INPUT:  tool: ToolDescriptor, params: JSON, caller_token: Token,
        deadline: Timestamp, trace_context: TraceContext,
        session_context: SessionContext
OUTPUT: execution_result: ExecutionResult

1.  // Determine SDK type from tool configuration
2.  sdk_type ← tool.sdk_config.type  -- openai, vllm, anthropic, google, etc.

3.  SWITCH sdk_type:

4.    CASE "openai":
5.      // Use python-openai module
6.      // Tool invocation through function calling is handled by the agent loop
7.      // This adapter handles SDK-specific built-in tools
8.      client ← OpenAIClient(
9.          api_key=SecretStore.Resolve(tool.sdk_config.key_ref),
10.         base_url=tool.sdk_config.base_url,  -- may point to vLLM
11.         timeout=RemainingTime(deadline)
12.     )
13.
14.     SWITCH tool.sdk_config.builtin_tool:
15.       CASE "web_search":
16.         result ← client.responses.create(
17.             model=tool.sdk_config.model,
18.             tools=[{type: "web_search_preview"}],
19.             input=params.query
20.         )
21.       CASE "code_interpreter":
22.         result ← client.responses.create(
23.             model=tool.sdk_config.model,
24.             tools=[{type: "code_interpreter", container: {type: "auto"}}],
25.             input=params.code
26.         )
27.       CASE "file_search":
28.         result ← client.responses.create(
29.             model=tool.sdk_config.model,
30.             tools=[{type: "file_search", vector_store_ids: params.store_ids}],
31.             input=params.query
32.         )
33.       CASE "custom_function":
34.         // Proxy: format as function call result for the calling model
35.         result ← ExecuteFunctionHandler(tool.sdk_config.handler, params)
36.
37.     RETURN ExecutionResult(
38.         value=ParseSDKResponse(result, tool.output_schema),
39.         duration_ms=elapsed(),
40.         resource_cost=EstimateSDKCost(result)
41.     )

42.   CASE "vllm":
43.     // vLLM-served models as tools (e.g., specialized classifier, reranker)
44.     client ← OpenAIClient(
45.         base_url=tool.sdk_config.vllm_endpoint,
46.         api_key="not-needed",  -- local vLLM
47.         timeout=RemainingTime(deadline)
48.     )
49.     result ← client.chat.completions.create(
50.         model=tool.sdk_config.model,
51.         messages=BuildToolMessages(params, tool.sdk_config.system_prompt),
52.         max_tokens=tool.sdk_config.max_tokens,
53.         temperature=tool.sdk_config.temperature
54.     )
55.     RETURN ExecutionResult(
56.         value=ParseVLLMResponse(result, tool.output_schema),
57.         duration_ms=elapsed()
58.     )
```

### 5.4 Browser Automation Adapter

```
ALGORITHM: BrowserAdapter.Execute
INPUT:  tool: ToolDescriptor, params: JSON, caller_token: Token,
        deadline: Timestamp, trace_context: TraceContext,
        session_context: SessionContext
OUTPUT: execution_result: ExecutionResult

1.  // Acquire or create browser context (isolated per session)
2.  browser_ctx ← BrowserPool.AcquireContext(
3.      session_id=session_context.session_id,
4.      isolation=STRICT,  -- separate cookies, storage, etc.
5.      timeout=RemainingTime(deadline)
6.  )

7.  IF browser_ctx IS NULL:
8.      RETURN ExecutionResult(error=RESOURCE_EXHAUSTED)

9.  TRY:
10.     SWITCH params.action:

11.       CASE "navigate":
12.         page ← browser_ctx.Navigate(params.url, wait_until="networkidle",
13.                                       timeout=MIN(30000, RemainingTime(deadline)))
14.         screenshot ← page.Screenshot(full_page=FALSE)
15.         dom_snapshot ← page.GetAccessibilityTree(max_depth=5)
16.         RETURN ExecutionResult(value={
17.             url: page.url,
18.             title: page.title,
19.             accessibility_tree: dom_snapshot,
20.             screenshot_ref: MediaStore.Write(screenshot, ttl=SESSION_TTL)
21.         })

22.       CASE "click":
23.         element ← page.QuerySelector(params.selector)
24.         IF element IS NULL:
25.             // Vision fallback: use vision model to locate element
26.             screenshot ← page.Screenshot()
27.             element_coords ← VisionPipeline.LocateElement(
28.                 screenshot, params.element_description
29.             )
30.             IF element_coords IS NULL:
31.                 RETURN ExecutionResult(error=PRECONDITION_FAILED,
32.                                        details="Element not found")
33.             page.Click(element_coords.x, element_coords.y)
34.         ELSE:
35.             element.Click()
36.         page.WaitForNavigation(timeout=5000)
37.         RETURN ExecutionResult(value={
38.             action: "click", success: TRUE,
39.             resulting_url: page.url
40.         })

41.       CASE "type":
42.         element ← page.QuerySelector(params.selector)
43.         element.Clear()
44.         element.Type(params.text, delay=params.typing_delay)
45.         RETURN ExecutionResult(value={action: "type", success: TRUE})

46.       CASE "extract":
47.         // Extract structured data from page
48.         content ← page.GetTextContent(params.selector)
49.         RETURN ExecutionResult(value={
50.             content: content,
51.             url: page.url
52.         })

53.       CASE "screenshot":
54.         screenshot ← page.Screenshot(
55.             full_page=params.full_page,
56.             clip=params.clip_region
57.         )
58.         ref ← MediaStore.Write(screenshot, ttl=SESSION_TTL)
59.         // Optional: generate text description via vision model
60.         description ← VisionPipeline.Describe(screenshot, max_tokens=200)
61.         RETURN ExecutionResult(value={
62.             screenshot_ref: ref,
63.             description: description,
64.             dimensions: screenshot.dimensions
65.         })

66. FINALLY:
67.     // Do NOT release browser context — session-bound
68.     // Will be released on session close
69.     PASS
```

### 5.5 Vision Pipeline Adapter

```
ALGORITHM: VisionPipelineAdapter.Execute
INPUT:  tool: ToolDescriptor, params: JSON, caller_token: Token,
        deadline: Timestamp, trace_context: TraceContext,
        session_context: SessionContext
OUTPUT: execution_result: ExecutionResult

1.  // Load image from reference or raw data
2.  image ← CASE
3.      WHEN params.image_ref IS NOT NULL:
4.          MediaStore.Read(params.image_ref)
5.      WHEN params.image_base64 IS NOT NULL:
6.          DecodeBase64(params.image_base64)
7.      WHEN params.screenshot_from_browser IS TRUE:
8.          BrowserPool.GetContext(session_context.session_id).Screenshot()
9.      WHEN params.desktop_screenshot IS TRUE:
10.         DesktopCapture.CaptureScreen(params.screen_region)

11. IF image IS NULL:
12.     RETURN ExecutionResult(error=INVALID_INPUT, details="No image source")

13. SWITCH params.task:

14.   CASE "describe":
15.     // Use multi-modal model for image description
16.     description ← OPENAI_INFERENCE(
17.         model=tool.vision_config.model,  -- e.g., gpt-4o or local vLLM with vision
18.         messages=[{
19.             role: "user",
20.             content: [
21.                 {type: "image_url", image_url: {url: ToDataURL(image)}},
22.                 {type: "text", text: params.prompt OR "Describe this image in detail."}
23.             ]
24.         }],
25.         max_tokens=params.max_tokens OR 300,
26.         temperature=0.0
27.     )
28.     RETURN ExecutionResult(value={description: description.content})

29.   CASE "locate_element":
30.     // UI element detection for computer use
31.     result ← VLLM_INFERENCE(
32.         model=tool.vision_config.grounding_model,
33.         messages=[{
34.             role: "user",
35.             content: [
36.                 {type: "image_url", image_url: {url: ToDataURL(image)}},
37.                 {type: "text", text: "Locate the UI element: " + params.element_description
38.                                    + ". Return bounding box coordinates [x1,y1,x2,y2] "
39.                                    + "as fractions of image dimensions."}
40.             ]
41.         }],
42.         max_tokens=100, temperature=0.0
43.     )
44.     coords ← ParseBoundingBox(result.content, image.dimensions)
45.     RETURN ExecutionResult(value={
46.         found: coords IS NOT NULL,
47.         bounding_box: coords,
48.         click_point: CenterOf(coords)
49.     })

50.   CASE "extract_text":
51.     // OCR + structured extraction
52.     ocr_text ← OCREngine.Extract(image)
53.     RETURN ExecutionResult(value={text: ocr_text})

54.   CASE "compare":
55.     // Compare two images (e.g., before/after screenshot)
56.     image_b ← MediaStore.Read(params.compare_to_ref)
57.     diff ← ImageDiff(image, image_b)
58.     RETURN ExecutionResult(value={
59.         similarity: diff.similarity_score,
60.         changed_regions: diff.regions,
61.         description: diff.description
62.     })
```

---

## 6. Composite Tool Orchestration — DAG Execution Engine

### 6.1 Formal Composition Model

A composite tool $\mathcal{T}_\oplus$ is a DAG:

$$
\mathcal{T}_\oplus = (V, E, \phi_{\text{bind}}, \psi_{\text{merge}}, \Pi_{\text{compensate}})
$$

where:
- $V = \{\mathcal{T}_1, \ldots, \mathcal{T}_n\}$ — constituent atomic tools
- $E \subseteq V \times V$ — execution dependencies
- $\phi_{\text{bind}}: \Sigma_{\text{out}}(\mathcal{T}_i) \to \Sigma_{\text{in}}(\mathcal{T}_j)$ — output-to-input binding functions
- $\psi_{\text{merge}}: \prod_i \Sigma_{\text{out}}(\mathcal{T}_i) \to \Sigma_{\text{out}}(\mathcal{T}_\oplus)$ — aggregate result merger
- $\Pi_{\text{compensate}}: V \to (\text{Action} \cup \{\bot\})$ — compensating action map

### 6.2 DAG Execution Algorithm

```
ALGORITHM: CompositeDAGExecution
INPUT:  composite: CompositeSpec, input_params: JSON, deadline: Timestamp,
        caller_token: Token, trace_context: TraceContext
OUTPUT: execution_result: ExecutionResult

1.  // Validate DAG structure
2.  ASSERT IsAcyclic(composite.edges), "Composite DAG contains cycle"
3.  ASSERT Depth(composite.edges) ≤ MAX_COMPOSITE_DEPTH, "DAG too deep"

4.  levels ← TopologicalSort(composite.tools, composite.edges)
5.  intermediate_results ← EMPTY_MAP
6.  committed_stack ← EMPTY_STACK

7.  FOR EACH level IN levels:
8.      // Check deadline
9.      IF NOW() > deadline - SAFETY_MARGIN_MS:
10.         GOTO CompensateAndFail("deadline_approaching")

11.     // Build parallel tasks for this level
12.     parallel_tasks ← EMPTY_LIST
13.     FOR EACH tool_ref IN level:
14.         // Bind inputs from upstream outputs and original params
15.         bound_input ← ApplyBindings(
16.             composite.bindings[tool_ref.id],
17.             input_params,
18.             intermediate_results
19.         )
20.         // Compute per-step deadline
21.         step_deadline ← MIN(
22.             tool_ref.timeout_class.max_duration + NOW(),
23.             deadline - SAFETY_MARGIN_MS
24.         )
25.         APPEND (tool_ref, bound_input, step_deadline) TO parallel_tasks

26.     // Execute level in parallel
27.     results ← PARALLEL_FOR_EACH (tool_ref, input, step_deadline) IN parallel_tasks:
28.         RETURN UnifiedToolDispatch(
29.             ToolInvocationRequest {
30.                 tool_id: tool_ref.id,
31.                 params: input,
32.                 caller_token: caller_token,
33.                 deadline: step_deadline,
34.                 trace_id: trace_context.trace_id,
35.                 idempotency_key: DeriveCompositeStepKey(
36.                     composite.id, tool_ref.id, input
37.                 )
38.             }
39.         )

40.     // Process results
41.     FOR EACH (tool_ref, result) IN results:
42.         IF result.status == ERROR:
43.             IF result.retryable AND tool_ref.retry_budget > 0:
44.                 retry_result ← RetryWithBackoff(
45.                     tool_ref, result.input, tool_ref.retry_budget, deadline
46.                 )
47.                 IF retry_result.status == ERROR:
48.                     GOTO CompensateAndFail("step_failed: " + tool_ref.id)
49.                 result ← retry_result
50.             ELSE:
51.                 GOTO CompensateAndFail("step_failed_non_retryable: " + tool_ref.id)

52.         intermediate_results[tool_ref.id] ← result.data
53.         committed_stack.Push(tool_ref)

54. // All levels complete — merge results
55. composite_output ← ApplyMergeFunction(
56.     composite.merge_function, intermediate_results
57. )
58. RETURN ExecutionResult(value=composite_output, duration_ms=elapsed())

CompensateAndFail:
59. // Reverse-order compensation
60. WHILE committed_stack IS NOT EMPTY:
61.     completed_tool ← committed_stack.Pop()
62.     compensation ← composite.compensations.Get(completed_tool.id)
63.     IF compensation IS NOT NULL:
64.         TRY:
65.             ExecuteCompensation(
66.                 compensation,
67.                 intermediate_results[completed_tool.id],
68.                 deadline
69.             )
70.         ON_ERROR(comp_error):
71.             EmitAlert(COMPENSATION_FAILED, completed_tool.id, comp_error)
72.             // Log but continue compensating remaining steps

73. RETURN ExecutionResult(
74.     error=COMPOSITE_PARTIAL_FAILURE,
75.     details={
76.         failed_step: failure_reason,
77.         completed_steps: [s.id FOR s IN committed_stack_original],
78.         compensations_executed: compensation_count
79.     }
80. )
```

---

## 7. Lazy Tool Loading and Token Budget Optimization

### 7.1 Three-Tier Loading Strategy

$$
C_{\text{tools}} = \underbrace{N \cdot \bar{s}_{\text{index}}}_{\text{Tier 0: always loaded}} + \underbrace{|\mathcal{T}_{\text{selected}}| \cdot \bar{s}_{\text{schema}}}_{\text{Tier 1: plan-selected}} + \underbrace{|\mathcal{T}_{\text{expanded}}| \cdot \bar{s}_{\text{detail}}}_{\text{Tier 2: on-demand}}
$$

### 7.2 Tool Selection as Knapsack Optimization

Given token budget $B_{\text{tools}}$, tool utility estimates $\{u_i\}$, and schema costs $\{s_i\}$:

$$
\max_{\mathbf{x} \in \{0,1\}^N} \sum_{i=1}^{N} u_i \cdot x_i \quad \text{subject to} \quad \sum_{i=1}^{N} s_i \cdot x_i \leq B_{\text{tools}}
$$

Greedy approximation by $u_i / s_i$ ratio:

```
ALGORITHM: LazyToolLoading
INPUT:  task: Task, registry: ToolRegistry, B_tools: int,
        model_config: ModelConfig
OUTPUT: affordance_block: List<ToolSchema>, remaining_budget: int

// ═══════════════════════════════════════════
// TIER 0: Compressed index (always loaded)
// ═══════════════════════════════════════════
1.  index ← registry.GetCompressedIndex()
2.  // index: List<{tool_id, one_sentence, mutation_class, type_class, health_score}>
3.  index_tokens ← TokenCount(index)
4.  remaining ← B_tools - index_tokens
5.  affordance_block ← [FormatTier0(index)]

// ═══════════════════════════════════════════
// TIER 1: Task-relevant tool schemas
// ═══════════════════════════════════════════
6.  // Estimate utility per tool for current task
7.  FOR EACH entry IN index:
8.      entry.utility ← EstimateToolUtility(entry, task)
9.      // Utility factors:
10.     //   - Semantic similarity between task description and tool description
11.     //   - Historical usage frequency for similar tasks (from procedural memory)
12.     //   - Mutation class alignment (read tasks prefer read tools)
13.     //   - Health score (degraded tools get lower utility)
14.     entry.cost ← registry.GetSchemaTokenCost(entry.tool_id)
15.     entry.ratio ← entry.utility / MAX(entry.cost, 1)

16. // Sort by utility/cost ratio descending (greedy knapsack)
17. SORT index BY ratio DESCENDING

18. FOR EACH entry IN index:
19.     IF entry.utility < θ_min_utility: BREAK  -- below relevance threshold
20.     schema ← registry.GetSchema(entry.tool_id, detail=STANDARD)
21.     schema_tokens ← TokenCount(schema)
22.     IF schema_tokens ≤ remaining:
23.         APPEND FormatTier1(schema) TO affordance_block
24.         remaining -= schema_tokens
25.     ELSE:
26.         // Try compressed reference (Tier 2 placeholder)
27.         ref ← registry.GetSchema(entry.tool_id, detail=MINIMAL)
28.         ref_tokens ← TokenCount(ref)
29.         IF ref_tokens ≤ remaining:
30.             APPEND FormatTier2Ref(ref) TO affordance_block
31.             remaining -= ref_tokens
32.         ELSE:
33.             BREAK  -- budget exhausted

34. RETURN (affordance_block, remaining)

FUNCTION: EstimateToolUtility(entry, task) → float
    sim ← SemanticSimilarity(entry.one_sentence, task.objective)
    hist ← ProceduralMemory.ToolUsageFrequency(entry.tool_id, task.task_class)
    mutation_align ← CASE
        WHEN task.requires_mutation AND entry.mutation_class != read_only: 0.3
        WHEN NOT task.requires_mutation AND entry.mutation_class == read_only: 0.2
        ELSE: 0.0
    health ← entry.health_score
    RETURN 0.45 * sim + 0.25 * hist + 0.15 * mutation_align + 0.15 * health
```

---

## 8. Idempotency Engine

### 8.1 Key Derivation

$$
k = \text{HMAC-SHA256}\left(\text{session\_id} \| \text{tool\_id} \| \text{step\_id} \| \text{canonical\_params}\right)
$$

where $\text{canonical\_params}$ is deterministic JSON serialization (sorted keys, normalized whitespace).

### 8.2 Deduplication Window

$$
w_{\text{dedup}}(\mathcal{T}) = \max\left(2 \cdot \tau_{\text{class}}(\mathcal{T})_{\max}, \; w_{\min}\right)
$$

### 8.3 Effective Semantics

$$
\text{at-least-once delivery} + \text{idempotent receiver} = \text{effectively-once execution}
$$

### 8.4 Idempotent Retry with Exponential Backoff

```
ALGORITHM: IdempotentRetry
INPUT:  tool: ToolDescriptor, params: JSON, retry_policy: RetryPolicy,
        session_context: SessionContext
OUTPUT: response: ToolInvocationResponse

1.  idempotency_key ← DeriveIdempotencyKey(
2.      session_context.session_id,
3.      tool.id,
4.      session_context.current_step_id,
5.      CanonicalJSON(params)
6.  )

7.  attempt ← 0
8.  last_error ← NULL

9.  WHILE attempt < retry_policy.max_attempts:
10.     attempt += 1

11.     request ← ToolInvocationRequest {
12.         tool_id: tool.id,
13.         version_spec: tool.version,
14.         params: params,
15.         idempotency_key: idempotency_key,
16.         caller_token: session_context.caller_token,
17.         trace_id: session_context.trace_id,
18.         deadline: ComputeAttemptDeadline(
19.             tool.timeout_class, retry_policy, attempt
20.         )
21.     }

22.     response ← UnifiedToolDispatch(request, ...)

23.     IF response.status == SUCCESS:
24.         RETURN response

25.     IF response.status == ERROR:
26.         IF NOT response.retryable:
27.             RETURN response  -- terminal failure
28.         last_error ← response

29.         // Exponential backoff with jitter
30.         backoff_ms ← retry_policy.base_ms * 2^(attempt - 1)
31.         capped_ms ← MIN(backoff_ms, retry_policy.max_backoff_ms)
32.         jitter ← RANDOM_UNIFORM(-0.25 * capped_ms, +0.25 * capped_ms)
33.         Sleep(MAX(0, capped_ms + jitter))

34. // Exhausted retry budget
35. EmitAlert(RETRY_BUDGET_EXHAUSTED, tool.id, attempt, last_error)
36. RETURN ErrorEnvelope(code=RETRY_EXHAUSTED, last_error=last_error)
```

---

## 9. Authorization and Approval System

### 9.1 Three-Tier Credential Model

$$
\text{Scopes}(C_t) \subseteq \text{Scopes}(C_a) \subseteq \text{Scopes}(C_u)
$$

$$
\text{TTL}(C_t) \leq \text{TTL}(C_a) \leq \text{TTL}(C_u)
$$

where $C_u$ is the user credential, $C_a$ is the agent credential (delegated, time-limited), and $C_t$ is the per-tool-invocation credential (narrowed).

### 9.2 Authorization Decision Function

$$
\texttt{AuthDecision} = f(\mathcal{C}_{\text{auth}}, C_{\text{caller}}, \mathcal{M}_{\text{mutation}}, \texttt{env}, \sigma_{\text{scope}})
$$

```
ALGORITHM: EvaluateAuthorization
INPUT:  tool: ToolDescriptor, caller_token: Token, params: JSON,
        environment: Environment
OUTPUT: decision: AuthDecision

1.  // Step 1: Verify caller identity
2.  identity ← VerifyToken(caller_token)
3.  IF identity IS NULL OR identity.expired:
4.      RETURN DENY

5.  // Step 2: Check scope intersection
6.  effective_scopes ← identity.scopes ∩ tool.auth_contract.required_scopes
7.  IF effective_scopes ⊊ tool.auth_contract.required_scopes:
8.      // Missing required scopes
9.      RETURN DENY

10. // Step 3: Resource scope check
11. resource_scope ← ExtractResourceScope(params, tool.side_effect_manifest)
12. IF NOT identity.resource_access.Covers(resource_scope):
13.     RETURN DENY

14. // Step 4: Approval gate evaluation
15. requires_approval ← EvaluateApprovalPolicy(
16.     tool.mutation_class, identity.trust_level,
17.     environment, resource_scope
18. )

19. IF requires_approval:
20.     RETURN REQUIRES_APPROVAL

21. RETURN ALLOW

FUNCTION: EvaluateApprovalPolicy(mutation_class, trust_level, env, scope) → bool
    // Read-only: never needs approval
    IF mutation_class == read_only: RETURN FALSE
    // Irreversible writes: always need approval
    IF mutation_class == write_irreversible: RETURN TRUE
    // Production + critical scope: always need approval
    IF env == PRODUCTION AND scope.criticality >= CRITICAL: RETURN TRUE
    // High-trust callers in non-prod: auto-approve reversible writes
    IF trust_level >= θ_auto_approve AND env != PRODUCTION: RETURN FALSE
    // Default: require approval
    RETURN TRUE
```

### 9.3 Approval Lifecycle

```
ALGORITHM: ApprovalLifecycle
INPUT:  request: ToolInvocationRequest, tool: ToolDescriptor
OUTPUT: resolution: ApprovalResolution

1.  // Create approval ticket
2.  ticket ← ApprovalTicket {
3.      ticket_id: GenerateUUID(),
4.      tool_id: tool.id,
5.      operation: request.params,
6.      params_redacted: RedactSensitive(request.params, tool.input_schema),
7.      requested_by: request.caller_token.identity,
8.      agent_id: request.agent_id,
9.      status: PENDING,
10.     created_at: NOW(),
11.     expires_at: NOW() + APPROVAL_TIMEOUT,
12.     escalation_level: 0
13. }

14. // Generate dry-run preview for irreversible tools
15. IF tool.mutation_class == write_irreversible:
16.     dry_run ← ExecuteDryRun(tool, request.params)
17.     ticket.dry_run_result ← dry_run

18. // Persist ticket
19. INSERT ticket INTO tool_approval_tickets

20. // Notify approvers
21. NotifyApproverGroup(tool.approval_config.escalation_chain[0], ticket)

22. // Monitor with escalation
23. escalation_idx ← 0
24. WHILE NOW() < ticket.expires_at:
25.     // Check for resolution
26.     status ← ApprovalStore.GetStatus(ticket.ticket_id)
27.     IF status IN {APPROVED, DENIED}:
28.         RETURN ApprovalResolution(status, resolved_by=status.resolver)

29.     // Check escalation threshold
30.     elapsed_fraction ← (NOW() - ticket.created_at) /
31.                          (ticket.expires_at - ticket.created_at)
32.     escalation_threshold ← (escalation_idx + 1) /
33.                              |tool.approval_config.escalation_chain|
34.     IF elapsed_fraction > escalation_threshold AND
35.        escalation_idx < |tool.approval_config.escalation_chain| - 1:
36.         escalation_idx += 1
37.         ticket.escalation_level ← escalation_idx
38.         NotifyApproverGroup(
39.             tool.approval_config.escalation_chain[escalation_idx], ticket
40.         )
41.         EmitEvent(APPROVAL_ESCALATED, ticket.ticket_id, escalation_idx)

42.     Sleep(APPROVAL_POLL_INTERVAL)

43. // Timeout: AUTO-DENY (never auto-approve)
44. UPDATE tool_approval_tickets
45.     SET status = 'auto_denied'
46.     WHERE ticket_id = ticket.ticket_id
47. EmitEvent(APPROVAL_AUTO_DENIED, ticket.ticket_id)
48. RETURN ApprovalResolution(AUTO_DENIED)
```

---

## 10. Timeout Classification and Deadline Propagation

### 10.1 Timeout Class Definitions

| Class | Symbol | Max Duration | Agent Behavior |
|---|---|---|---|
| Interactive | $\tau_I$ | $< 500\text{ms}$ | Synchronous wait |
| Standard | $\tau_S$ | $< 5\text{s}$ | Synchronous wait with progress |
| Long-Running | $\tau_L$ | $< 5\text{min}$ | Checkpoint-based; may interleave |
| Async | $\tau_A$ | $> 5\text{min}$ | Submit-and-poll; continue other work |

### 10.2 Effective Deadline Computation

$$
d_{\text{effective}} = \min\left(\tau_{\text{class}}(\mathcal{T})_{\max},\ d_{\text{agent\_loop}} - t_{\text{elapsed}} - t_{\text{safety}}\right)
$$

### 10.3 Timeout Class Assignment

$$
\tau_{\text{class}}(\mathcal{T}) = \begin{cases}
\tau_I & \text{if } p_{99}(\text{latency}) < 500\text{ms} \wedge \text{no external I/O} \\
\tau_S & \text{if } p_{99}(\text{latency}) < 5\text{s} \wedge \text{bounded I/O} \\
\tau_L & \text{if } p_{99}(\text{latency}) < 5\text{min} \wedge \text{checkpointable} \\
\tau_A & \text{otherwise}
\end{cases}
$$

---

## 11. Tool Result Cache Architecture

### 11.1 Cache Eligibility

Only tools satisfying $\mathcal{M}(\mathcal{T}) = \mathcal{M}_R$ (read-only) are cacheable. Cache keys are derived from the canonical input hash:

$$
k_{\text{cache}} = \text{SHA-256}(\text{tool\_id} \| \text{version} \| \text{CanonicalJSON}(\text{params}))
$$

### 11.2 Cache Operations

```
ALGORITHM: ToolCacheOperations

// READ: Check cache before execution
FUNCTION CacheLookup(tool_id, version, params) → CachedResult OR NULL:
    key ← SHA256(tool_id || version || CanonicalJSON(params))
    result ← QUERY:
        SELECT result_data, created_at FROM tool_result_cache
        WHERE cache_key = :key AND expires_at > NOW()
        LIMIT 1
    IF |result| > 0:
        UPDATE tool_result_cache SET hit_count = hit_count + 1
            WHERE cache_key = :key
        CacheMetrics.record_hit(tool_id)
        RETURN result[0].result_data
    CacheMetrics.record_miss(tool_id)
    RETURN NULL

// WRITE: Store result after execution
FUNCTION CacheStore(tool_id, version, params, result, ttl):
    key ← SHA256(tool_id || version || CanonicalJSON(params))
    INSERT INTO tool_result_cache (cache_key, tool_id, result_data, expires_at)
    VALUES (:key, :tool_id, :result, NOW() + :ttl)
    ON CONFLICT (cache_key) DO UPDATE
        SET result_data = :result, expires_at = NOW() + :ttl, hit_count = 0

// INVALIDATE: On tool schema change or data mutation
FUNCTION CacheInvalidate(tool_id):
    DELETE FROM tool_result_cache WHERE tool_id = :tool_id
    CacheMetrics.record_invalidation(tool_id)
```

---

## 12. MCP Discovery and Change Subscription

```
ALGORITHM: MCPDiscoveryAndSubscription
INPUT:  mcp_endpoints: List<MCPEndpoint>, registry: ToolRegistry
OUTPUT: discovered_tools: List<ToolDescriptor>

1.  all_discovered ← EMPTY_LIST

2.  FOR EACH endpoint IN mcp_endpoints:
3.      TRY:
4.          // Step 1: Initialize MCP connection
5.          session ← MCPSession.Connect(
6.              endpoint.url,
7.              transport=endpoint.transport,  -- stdio, sse, streamable_http
8.              protocol_version=CURRENT_MCP_VERSION
9.          )

10.         // Step 2: Capability negotiation
11.         capabilities ← session.Initialize(
12.             client_info={name: "agent_tools", version: MODULE_VERSION},
13.             client_capabilities={
14.                 tools: {listChanged: TRUE},
15.                 resources: {subscribe: TRUE, listChanged: TRUE},
16.                 prompts: {listChanged: TRUE}
17.             }
18.         )

19.         // Step 3: Discover tools
20.         IF capabilities.tools IS NOT NULL:
21.             tool_list ← session.ListTools(cursor=NULL)
22.             WHILE tool_list IS NOT NULL:
23.                 FOR EACH mcp_tool IN tool_list.tools:
24.                     descriptor ← MapMCPToolToDescriptor(mcp_tool, endpoint)
25.                     admission ← ToolAdmissionGate(descriptor, NULL)
26.                     IF admission == ADMITTED:
27.                         registry.Register(descriptor)
28.                         APPEND descriptor TO all_discovered
29.                     ELSE:
30.                         EmitWarning(TOOL_ADMISSION_FAILED, mcp_tool.name,
31.                                      admission.violations)
32.                 IF tool_list.nextCursor IS NOT NULL:
33.                     tool_list ← session.ListTools(cursor=tool_list.nextCursor)
34.                 ELSE:
35.                     tool_list ← NULL

36.         // Step 4: Discover resources
37.         IF capabilities.resources IS NOT NULL:
38.             resource_list ← session.ListResources(cursor=NULL)
39.             // Similar pagination and registration loop for MCP resources
40.             FOR EACH resource IN resource_list.resources:
41.                 descriptor ← MapMCPResourceToDescriptor(resource, endpoint)
42.                 registry.Register(descriptor)

43.         // Step 5: Discover prompt surfaces
44.         IF capabilities.prompts IS NOT NULL:
45.             prompt_list ← session.ListPrompts(cursor=NULL)
46.             FOR EACH prompt IN prompt_list.prompts:
47.                 descriptor ← MapMCPPromptToDescriptor(prompt, endpoint)
48.                 registry.Register(descriptor)

49.         // Step 6: Subscribe to change notifications
50.         session.OnNotification("notifications/tools/list_changed",
51.             CALLBACK: OnToolsChanged(endpoint, registry))
52.         session.OnNotification("notifications/resources/list_changed",
53.             CALLBACK: OnResourcesChanged(endpoint, registry))
54.         session.OnNotification("notifications/prompts/list_changed",
55.             CALLBACK: OnPromptsChanged(endpoint, registry))

56.     ON_ERROR(e):
57.         EmitWarning(MCP_DISCOVERY_FAILED, endpoint.url, e)
58.         // Mark all tools from this endpoint as unavailable
59.         registry.MarkEndpointUnavailable(endpoint.id)

60. RETURN all_discovered

CALLBACK: OnToolsChanged(endpoint, registry):
    // Re-discover tools from this endpoint
    updated_tools ← DiscoverToolsFromEndpoint(endpoint)
    current_tools ← registry.GetToolsByEndpoint(endpoint.id)
    
    // Diff and update
    added ← updated_tools - current_tools
    removed ← current_tools - updated_tools
    
    FOR EACH tool IN added:
        registry.Register(tool)
    FOR EACH tool IN removed:
        registry.MarkUnavailable(tool.id)
    
    // Invalidate prefill compiler cache
    PrefillCompiler.InvalidateToolAffordanceCache()
```

---

## 13. Tool Observability

### 13.1 Metrics Emission Points

| Metric | Type | Labels | Emission Point |
|---|---|---|---|
| `tool.invocation.count` | Counter | tool_id, version, status, mutation_class, type_class | Phase 8 of dispatch |
| `tool.invocation.latency_ms` | Histogram | tool_id, version, timeout_class | Phase 8 of dispatch |
| `tool.invocation.cost` | Counter | tool_id, cost_category | Phase 8 of dispatch |
| `tool.cache.hit_rate` | Gauge | tool_id | Cache lookup |
| `tool.idempotency.replay_rate` | Gauge | tool_id | Idempotency check |
| `tool.error_rate` | Gauge | tool_id, error_code | Computed from invocation count |
| `tool.timeout_rate` | Gauge | tool_id, timeout_class | Computed from invocation count |
| `tool.approval.pending_count` | Gauge | tool_id | Approval monitor |
| `tool.approval.resolution_ms` | Histogram | tool_id, resolution | Approval lifecycle |
| `tool.health_score` | Gauge | tool_id | Health monitor |
| `tool.schema.validation_failure_rate` | Gauge | tool_id, phase | Validation phases |

### 13.2 Health Score Computation

$$
h(\mathcal{T}) = 1 - \left(0.5 \cdot \frac{\text{error\_rate}}{\epsilon_{\max}} + 0.3 \cdot \frac{p_{99}(\text{latency})}{\tau_{\text{class}}(\mathcal{T})_{\max}} + 0.2 \cdot \frac{\text{timeout\_rate}}{\epsilon_{\text{timeout}}}\right)
$$

Clamped to $[0, 1]$. Tools with $h(\mathcal{T}) < 0.3$ are automatically quarantined.

### 13.3 Health Monitor

```
ALGORITHM: ToolHealthMonitor
INPUT:  registry: ToolRegistry, metrics_store: MetricsStore,
        schedule: CronSchedule (every 60s)
OUTPUT: health_report: HealthReport

1.  ON schedule:
2.  FOR EACH tool IN registry.GetAllActive():
3.      window ← LAST_15_MINUTES

4.      error_rate ← metrics_store.Query(
5.          "rate(tool.invocation.count{tool_id=$tool.id, status='error'}[$window])"
6.      ) / MAX(1, metrics_store.Query(
7.          "rate(tool.invocation.count{tool_id=$tool.id}[$window])"
8.      ))

9.      p99_latency ← metrics_store.Query(
10.         "histogram_quantile(0.99, tool.invocation.latency_ms{tool_id=$tool.id}[$window])"
11.     )

12.     timeout_rate ← metrics_store.Query(
13.         "tool.timeout_rate{tool_id=$tool.id}[$window]"
14.     )

15.     // Compute health score
16.     h ← 1.0 - CLAMP(
17.         0.5 * (error_rate / ε_max) +
18.         0.3 * (p99_latency / tool.timeout_class.max_duration) +
19.         0.2 * (timeout_rate / ε_timeout),
20.         0, 1
21.     )

22.     registry.UpdateHealth(tool.id, h)

23.     // Alerting
24.     IF error_rate > tool.alert_thresholds.error_rate:
25.         EmitAlert(TOOL_ERROR_RATE_HIGH, tool.id, error_rate)
26.     IF p99_latency > 2 * tool.baseline_latency_p99:
27.         EmitAlert(TOOL_LATENCY_REGRESSION, tool.id, p99_latency)
28.     IF timeout_rate > tool.alert_thresholds.timeout_rate:
29.         EmitAlert(TOOL_TIMEOUT_RATE_HIGH, tool.id, timeout_rate)
30.         // Suggest reclassification
31.         suggested ← InferTimeoutClass(p99_latency)
32.         IF suggested != tool.timeout_class:
33.             EmitRecommendation(RECLASSIFY_TIMEOUT, tool.id, suggested)

34.     // Auto-quarantine critically degraded tools
35.     IF h < 0.3:
36.         registry.Quarantine(tool.id, reason="health_score_critical")
37.         EmitAlert(TOOL_QUARANTINED, tool.id, h)
```

---

## 14. Memory Integration — Tool Pattern Learning

### 14.1 Write-Back to Procedural Memory

After successful tool invocations, the system writes tool usage patterns back to procedural memory for future optimization:

```
ALGORITHM: ToolPatternWriteBack
INPUT:  invocation: CompletedInvocation, session_context: SessionContext
OUTPUT: write_back_result: WriteBackResult

1.  // Only write back for successful, non-trivial invocations
2.  IF invocation.status != SUCCESS: RETURN SKIP
3.  IF invocation.latency_ms < TRIVIAL_THRESHOLD_MS: RETURN SKIP

4.  // Check if this tool usage pattern is novel
5.  pattern ← ToolUsagePattern {
6.      tool_id: invocation.tool_id,
7.      task_class: session_context.task_class,
8.      input_schema_hash: SchemaHash(invocation.params),
9.      success: TRUE,
10.     latency_ms: invocation.latency_ms,
11.     cost: invocation.resource_cost,
12.     context_summary: TRUNCATE(session_context.task_description, 200)
13. }

14. // Update history table with tool usage data
15. evidence_record ← {
16.     chunk_id: invocation.trace_id,
17.     source: "tool_invocation:" + invocation.tool_id,
18.     utility_score: ComputeToolUtility(invocation),
19.     was_cited: TRUE,
20.     use_count: 1
21. }

22. // Asynchronous write to history for future retrieval
23. ASYNC: UpdateHistoryWithToolUsage(
24.     session_context.uu_id,
25.     session_context.session_id,
26.     pattern,
27.     evidence_record
28. )

29. // Check promotion to procedural memory
30. similar_patterns ← ProceduralMemory.FindSimilar(
31.     session_context.uu_id, pattern, threshold=0.85
32. )
33. IF |similar_patterns| >= PATTERN_CONSOLIDATION_THRESHOLD:
34.     // This tool usage pattern is recurring → promote
35.     ProceduralMemory.ConsolidatePattern(
36.         session_context.uu_id, similar_patterns + [pattern]
37.     )

38. RETURN WriteBackResult(RECORDED)
```

---

## 15. Model-Specific Tool Call Format Adapters

### 15.1 Format Translation Layer

Different models expect tool calls in different formats. The format adapter translates between the internal tool representation and model-specific formats:

```
ALGORITHM: FormatToolsForModel
INPUT:  selected_tools: List<ToolSchema>, model_config: ModelConfig
OUTPUT: formatted_tools: ModelSpecificToolFormat

1.  SWITCH model_config.provider:

2.    CASE "openai":
3.      // OpenAI function calling format
4.      formatted ← []
5.      FOR EACH tool IN selected_tools:
6.          formatted.append({
7.              type: "function",
8.              function: {
9.                  name: SanitizeToolName(tool.id),  -- OpenAI name constraints
10.                 description: tool.compressed_description,
11.                 parameters: tool.input_schema,
12.                 strict: TRUE  -- enforce strict schema adherence
13.             }
14.         })
15.     RETURN {format: "openai_tools", tools: formatted}

16.   CASE "vllm":
17.     // vLLM uses OpenAI-compatible format
18.     // Same as OpenAI but may need guided decoding grammar
19.     openai_format ← FormatToolsForModel(selected_tools,
20.                                           ModelConfig{provider: "openai"})
21.     IF model_config.use_guided_decoding:
22.         grammar ← BuildToolCallGrammar(selected_tools)
23.         openai_format.guided_decoding_grammar ← grammar
24.     RETURN openai_format

25.   CASE "anthropic":
26.     // Claude tool_use format
27.     formatted ← []
28.     FOR EACH tool IN selected_tools:
29.         formatted.append({
30.             name: tool.id,
31.             description: tool.compressed_description,
32.             input_schema: tool.input_schema
33.         })
34.     RETURN {format: "anthropic_tools", tools: formatted}

35.   CASE "google":
36.     // Google function declarations
37.     formatted ← []
38.     FOR EACH tool IN selected_tools:
39.         formatted.append({
40.             name: tool.id,
41.             description: tool.compressed_description,
42.             parameters: ConvertToGoogleSchema(tool.input_schema)
43.         })
44.     RETURN {format: "google_tools",
45.             function_declarations: formatted}
```

### 15.2 Tool Call Parsing from Model Output

```
ALGORITHM: ParseToolCallFromModelOutput
INPUT:  model_output: ModelResponse, model_config: ModelConfig
OUTPUT: tool_calls: List<ToolCallRequest>

1.  tool_calls ← EMPTY_LIST

2.  SWITCH model_config.provider:

3.    CASE "openai", "vllm":
4.      IF model_output.choices[0].message.tool_calls IS NOT NULL:
5.          FOR EACH tc IN model_output.choices[0].message.tool_calls:
6.              tool_calls.append(ToolCallRequest {
7.                  call_id: tc.id,
8.                  tool_id: ResolveToolName(tc.function.name),
9.                  params: ParseJSON(tc.function.arguments),
10.                 model_format: "openai"
11.             })

12.   CASE "anthropic":
13.     FOR EACH block IN model_output.content:
14.         IF block.type == "tool_use":
15.             tool_calls.append(ToolCallRequest {
16.                 call_id: block.id,
17.                 tool_id: block.name,
18.                 params: block.input,
19.                 model_format: "anthropic"
20.             })

21.   CASE "google":
22.     IF model_output.candidates[0].content.parts IS NOT NULL:
23.         FOR EACH part IN model_output.candidates[0].content.parts:
24.             IF part.function_call IS NOT NULL:
25.                 tool_calls.append(ToolCallRequest {
26.                     call_id: GenerateCallID(),
27.                     tool_id: part.function_call.name,
28.                     params: part.function_call.args,
29.                     model_format: "google"
30.                 })

31. // Validate all parsed tool calls
32. FOR EACH tc IN tool_calls:
33.     tool ← ToolRegistry.Resolve(tc.tool_id)
34.     IF tool IS NULL:
35.         tc.status ← INVALID
36.         tc.error ← "Tool not found: " + tc.tool_id
37.     ELSE:
38.         validation ← ValidateAgainstSchema(tc.params, tool.input_schema)
39.         IF NOT validation.valid:
40.             tc.status ← INVALID
41.             tc.error ← "Input validation failed: " + validation.errors
42.         ELSE:
43.             tc.status ← VALID

44. RETURN tool_calls
```

---

## 16. Tool Testing Infrastructure

### 16.1 Schema-Driven Test Generation

```
ALGORITHM: GenerateToolTestSuite
INPUT:  tool: ToolDescriptor
OUTPUT: test_suite: TestSuite

1.  tests ← EMPTY_LIST

// ═══════════════════════════════════════════
// CATEGORY 1: Input validation tests
// ═══════════════════════════════════════════
2.  // Valid inputs
3.  FOR EACH example IN tool.input_schema.examples:
4.      tests.append(Test("valid_input", input=example, expect=SUCCESS))

5.  // Missing required fields
6.  FOR EACH field IN tool.input_schema.required:
7.      tests.append(Test("missing_" + field,
8.          input=RemoveField(ValidInput(), field),
9.          expect=ERROR, expect_code=INVALID_INPUT))

10. // Boundary conditions
11. FOR EACH prop IN tool.input_schema.properties:
12.     IF prop.maxLength: tests.append(BoundaryTest(prop, "maxLength"))
13.     IF prop.minimum: tests.append(BoundaryTest(prop, "minimum"))
14.     IF prop.maximum: tests.append(BoundaryTest(prop, "maximum"))
15.     IF prop.enum: tests.append(EnumTest(prop))

16. // Additional properties rejection
17. tests.append(Test("reject_additional_props",
18.     input=ValidInput() ∪ {"hallucinated_field": "value"},
19.     expect=ERROR, expect_code=INVALID_INPUT))

// ═══════════════════════════════════════════
// CATEGORY 2: Idempotency tests (mutating tools)
// ═══════════════════════════════════════════
20. IF tool.mutation_class != read_only:
21.     tests.append(IdempotencyTest(tool))

// ═══════════════════════════════════════════
// CATEGORY 3: Authorization tests
// ═══════════════════════════════════════════
22. tests.append(Test("auth_denied_no_scopes",
23.     input=ValidInput(), caller=NoScopeToken(),
24.     expect=ERROR, expect_code=AUTHORIZATION_DENIED))

// ═══════════════════════════════════════════
// CATEGORY 4: Timeout enforcement tests
// ═══════════════════════════════════════════
25. tests.append(Test("timeout_enforcement",
26.     input=SlowInput(2 * tool.timeout_class.max_duration),
27.     expect=ERROR, expect_code=DEADLINE_EXCEEDED))

// ═══════════════════════════════════════════
// CATEGORY 5: Output schema compliance
// ═══════════════════════════════════════════
28. tests.append(Test("output_schema_valid",
29.     input=ValidInput(), expect=SUCCESS,
30.     output_validator=λ out: ValidateAgainstSchema(out, tool.output_schema)))

// ═══════════════════════════════════════════
// CATEGORY 6: Behavioral contracts
// ═══════════════════════════════════════════
31. IF tool.mutation_class == read_only:
32.     tests.append(DeterminismTest(tool, iterations=10))
33. IF tool.mutation_class == write_reversible:
34.     tests.append(ReversibilityTest(tool))

35. RETURN TestSuite(tool_id=tool.id, tests=tests)
```

---

## 17. Operational Invariants

| # | Invariant | Enforcement |
|---|---|---|
| 1 | Every tool has typed, versioned, schema-validated contract | Admission predicate at registration |
| 2 | Every invocation traverses full 8-phase lifecycle | Protocol enforcement in dispatch engine |
| 3 | Every mutating tool is idempotent with caller-generated keys | Idempotency store + dedup window |
| 4 | Authorization is caller-scoped, never agent-ambient | Three-tier credential chain + policy evaluation |
| 5 | Irreversible mutations require approval or dry-run | Approval gates + auto-deny on timeout |
| 6 | Tool schemas lazily loaded under token budget | Knapsack-optimized prefill compilation |
| 7 | Every invocation traced, metered, audit-logged | Observability infrastructure at every phase |
| 8 | Schema evolution follows semver with compatibility checks | CI/CD deploy gates |
| 9 | Side effects declared, detected, audited | Manifests + runtime mutation detection |
| 10 | Behavioral contracts verified via property-based testing | Mandatory CI gates |
| 11 | User isolation: tool results scoped to `uu_id` | Partition pruning + caller token binding |
| 12 | Composite DAGs bounded in depth, compensable on failure | DAG validator + reverse compensation |
| 13 | Tool health continuously monitored with auto-quarantine | Health score < 0.3 triggers quarantine |
| 14 | Cache serves only read-only tools with provenance-aware invalidation | Mutation class gating |
| 15 | A2A delegation preserves caller identity chain | Token delegation + task isolation |

---

## 18. Latency Budget Allocation

| Phase | Operation | Budget (ms) | Parallelism | Fallback |
|---|---|---|---|---|
| 0 | Tool resolution from registry | 1 | — | Reject if not found |
| 1 | Input schema validation | 2 | — | Reject on failure |
| 2 | Idempotency lookup | 3 | — | Replay cached result |
| 3 | Authorization evaluation | 5 | — | Deny on failure |
| 4 | Cache check (read-only) | 3 | — | Cache miss → execute |
| 5 | Protocol-specific execution | Varies by $\tau$ | — | Timeout → retry or fail |
| 6 | Output validation | 2 | — | Quarantine tool on failure |
| 7 | Side-effect verification | 5 | — | Compensate on inconsistency |
| 8 | Result storage (async) | 0 | Async | — |
| **Overhead (excl. execution)** | | **~21ms** | | |

The tool dispatch overhead is $\leq 25\text{ms}$, ensuring that tool invocation latency is dominated by actual tool execution time, not framework overhead.

---

## 19. Above-SOTA Innovation Summary

| Innovation | Mechanism | Advantage |
|---|---|---|
| **Unified protocol router across 13 type classes** | Single dispatch engine with protocol-specific adapters | Any tool type invoked through identical lifecycle |
| **Three-tier lazy loading with knapsack optimization** | $\max \sum u_i x_i$ s.t. $\sum s_i x_i \leq B$ | Optimal context utilization vs. static schema injection |
| **A2A protocol integration as first-class tool type** | Google A2A task delegation with typed contracts | Agent-to-agent delegation with full provenance |
| **Vision-augmented browser tools** | CDP + vision model fallback for element location | Handles dynamic UIs where selectors fail |
| **Composite DAG execution with reverse compensation** | Topological execution + compensating action stack | Safe multi-step tool chains with partial failure recovery |
| **Three-tier credential narrowing** | $\text{Scopes}(C_t) \subseteq \text{Scopes}(C_a) \subseteq \text{Scopes}(C_u)$ | Zero privilege escalation at any layer |
| **Health-score-driven auto-quarantine** | $h < 0.3 \Rightarrow$ quarantine | Degraded tools automatically removed from agent use |
| **Tool pattern write-back to procedural memory** | Successful patterns consolidated via procedural memory promotion | Agent learns optimal tool selection over time |
| **Model-agnostic format translation** | Per-provider adapters for OpenAI, vLLM, Anthropic, Google | Single tool registry serves all model types |
| **Schema-driven test generation** | Automated boundary, idempotency, auth, timeout tests from schema | Contract tests scale with tool count without manual effort |
| **Approval with escalation and auto-deny** | Never auto-approve on timeout; escalation chains | Safety-by-default for irreversible mutations |
| **Side-effect manifest with runtime verification** | Declared $\mathcal{SE}$ vs. detected mutations → quarantine on mismatch | Catches tools that lie about their mutation class |

---

$$
\boxed{\text{Tools} = \text{Typed Infrastructure.}\quad \text{Every invocation: validated, authorized, idempotent, traced, budget-bounded, compensable.}}
$$