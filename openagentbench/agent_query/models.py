"""Typed records for context assembly, decomposition, routing, and compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4

from openagentbench.agent_context.models import (
    CompiledCycleContext,
    CompilationTrace,
    ContextArchiveEntry,
    ContextInvariantReport,
)
from openagentbench.agent_data import SessionRecord
from openagentbench.agent_retrieval import (
    ProtocolType,
    QueryClassification,
    RetrievalMode,
    SelectedModelPlan,
    SourceTable,
)

from .config import DEFAULT_MAX_SUBQUERIES, DEFAULT_TOOL_TOKEN_BUDGET
from .enums import (
    AmbiguityLevel,
    ClarificationMode,
    EmotionalTone,
    ExpertiseLevel,
    IntentClass,
    QueryErrorCode,
    RouteTarget,
)
from .types import JSONValue


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class QueryTemplate:
    sql: str
    params: Mapping[str, object]


@dataclass(slots=True, frozen=True)
class QueryBudgetAllocation:
    total_budget: int
    context_budget: int
    intent_budget: int
    pragmatic_budget: int
    cognitive_budget: int
    rewrite_budget: int
    decomposition_budget: int
    routing_budget: int
    clarification_budget: int
    reserve_budget: int

    def allocated_budget(self) -> int:
        return (
            self.context_budget
            + self.intent_budget
            + self.pragmatic_budget
            + self.cognitive_budget
            + self.rewrite_budget
            + self.decomposition_budget
            + self.routing_budget
            + self.clarification_budget
            + self.reserve_budget
        )


@dataclass(slots=True, frozen=True)
class ContextSourceRecord:
    source: str
    token_count: int
    detail: str


@dataclass(slots=True, frozen=True)
class ToolAffordanceSummary:
    tool_id: str
    description: str
    protocol: str
    token_cost_estimate: int
    relevance_score: float
    required_scopes: tuple[str, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class QueryContextArtifact:
    query_text: str
    session_topic: str
    history_excerpt: tuple[str, ...]
    memory_messages: tuple[dict[str, Any], ...]
    tool_affordances: tuple[ToolAffordanceSummary, ...]
    token_accounting: Mapping[str, int]
    provenance: tuple[ContextSourceRecord, ...]
    remaining_budget: int
    compiled_context: CompiledCycleContext | None = None
    invariant_report: ContextInvariantReport | None = None
    compilation_trace: CompilationTrace | None = None
    archive_entry: ContextArchiveEntry | None = None


@dataclass(slots=True, frozen=True)
class IntentHypothesis:
    intent_class: IntentClass
    confidence: float
    rationale: str
    preferred_protocols: tuple[ProtocolType, ...]
    relevant_tools: tuple[str, ...]
    retrieval_classification: QueryClassification


@dataclass(slots=True, frozen=True)
class PragmaticProfile:
    ambiguity_level: AmbiguityLevel
    clarification_mode: ClarificationMode
    presuppositions: tuple[str, ...]
    implied_constraints: tuple[str, ...]
    missing_slots: tuple[str, ...]
    coreference_risk: float
    behavioral_signals: tuple[str, ...] = ()
    latent_goals: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class CognitiveComplexity:
    complexity_score: float
    requires_decomposition: bool
    estimated_steps: int
    expertise_level: ExpertiseLevel
    emotional_tone: EmotionalTone


@dataclass(slots=True, frozen=True)
class QueryRewritePlan:
    resolved_query: str
    expanded_query: str
    search_query: str
    hypothetical_answer: str
    preserved_constraints: tuple[str, ...]
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ClarificationQuestion:
    question: str
    reason: str
    blocking: bool = True


@dataclass(slots=True, frozen=True)
class RoutedSubQuery:
    step_id: str
    text: str
    route_target: RouteTarget
    protocol: str
    priority: float
    dependencies: tuple[str, ...] = ()
    tool_candidates: tuple[str, ...] = ()
    retrieval_modes: tuple[RetrievalMode, ...] = ()
    target_tables: tuple[SourceTable, ...] = ()


@dataclass(slots=True, frozen=True)
class QueryResolutionRequest:
    user_id: UUID
    session: SessionRecord
    query_text: str
    context_window_size: int | None = None
    tool_token_budget: int = DEFAULT_TOOL_TOKEN_BUDGET
    max_subqueries: int = DEFAULT_MAX_SUBQUERIES
    idempotency_key: str | None = None
    deadline_ms: int | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class QueryResolutionPlan:
    original_query: str
    context: QueryContextArtifact
    budget: QueryBudgetAllocation
    intent: IntentHypothesis
    pragmatic: PragmaticProfile
    cognitive: CognitiveComplexity
    rewrite: QueryRewritePlan
    subqueries: tuple[RoutedSubQuery, ...]
    selected_model_plan: SelectedModelPlan
    needs_clarification: bool
    clarification_questions: tuple[ClarificationQuestion, ...]
    error_code: QueryErrorCode | None = None
    generated_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class QueryAuditRecord:
    audit_id: UUID
    user_id: UUID
    session_id: UUID
    cache_key: str
    ambiguity_level: AmbiguityLevel
    subquery_count: int
    cache_hit: bool
    latency_ms: int
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class QueryCacheEntry:
    cache_key: str
    user_id: UUID
    session_id: UUID
    plan: QueryResolutionPlan
    expires_at: datetime
    hit_count: int = 0
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class QueryResolutionResponse:
    plan: QueryResolutionPlan
    audit_id: UUID
    cache_key: str
    cache_hit: bool
    latency_ms: int


@dataclass(slots=True, frozen=True)
class QueryEndpointCompatibilityReport:
    retrieval_report: Any
    memory_report: Any
    tool_report: Any
    agent_data_endpoints: tuple[Any, ...]
    vllm_endpoints: tuple[Any, ...]
    query_resolve_tool: dict[str, Any]
    query_clarify_tool: dict[str, Any]
    openai_responses_request: dict[str, Any]
    openai_chat_request: dict[str, Any]
    openai_realtime_request: dict[str, Any]
    vllm_responses_request: dict[str, Any]
    vllm_chat_request: dict[str, Any]
    gemini_generate_content_request: dict[str, Any]
    gemini_count_tokens_request: dict[str, Any]


def new_query_audit_record(
    *,
    user_id: UUID,
    session_id: UUID,
    cache_key: str,
    ambiguity_level: AmbiguityLevel,
    subquery_count: int,
    cache_hit: bool,
    latency_ms: int,
) -> QueryAuditRecord:
    return QueryAuditRecord(
        audit_id=uuid4(),
        user_id=user_id,
        session_id=session_id,
        cache_key=cache_key,
        ambiguity_level=ambiguity_level,
        subquery_count=subquery_count,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
    )


__all__ = [
    "ClarificationQuestion",
    "CognitiveComplexity",
    "ContextSourceRecord",
    "IntentHypothesis",
    "PragmaticProfile",
    "QueryAuditRecord",
    "QueryBudgetAllocation",
    "QueryCacheEntry",
    "QueryContextArtifact",
    "QueryEndpointCompatibilityReport",
    "QueryResolutionPlan",
    "QueryResolutionRequest",
    "QueryResolutionResponse",
    "QueryRewritePlan",
    "QueryTemplate",
    "RoutedSubQuery",
    "ToolAffordanceSummary",
    "new_query_audit_record",
]
