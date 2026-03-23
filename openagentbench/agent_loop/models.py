"""Typed records for the agent-loop orchestration engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter_ns
from typing import Any, Mapping
from uuid import UUID, uuid4

from openagentbench.agent_context import (
    CompiledCycleContext,
    CompilationTrace,
    ContextArchiveEntry,
    ContextInvariantReport,
)
from openagentbench.agent_data import HistoryRecord, MemoryRecord, MemoryScope, MemoryTier, SessionRecord
from openagentbench.agent_query import QueryResolutionResponse, RouteTarget
from openagentbench.agent_retrieval import AuthorityTier, MemoryType, SourceTable
from openagentbench.agent_memory import WorkingMemoryItem

from .enums import (
    ActionStatus,
    CognitiveMode,
    EscalationReason,
    LoopPhase,
    MetacognitiveDecision,
    RepairStrategy,
    RootCauseClass,
    SubsystemAvailability,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class LoopPhaseBudget:
    total_budget: int
    plan_budget: int
    retrieve_budget: int
    act_budget: int
    verify_budget: int
    critique_budget: int
    repair_budget: int
    reserve_budget: int

    def allocated_budget(self) -> int:
        return (
            self.plan_budget
            + self.retrieve_budget
            + self.act_budget
            + self.verify_budget
            + self.critique_budget
            + self.repair_budget
            + self.reserve_budget
        )


@dataclass(slots=True, frozen=True)
class LoopPolicy:
    total_token_budget: int = 24_000
    reserve_ratio: float = 0.10
    system1_budget_ratio: float = 0.15
    system1_complexity_threshold: float = 0.45
    system1_intent_confidence_threshold: float = 0.78
    metacognitive_threshold: float = 0.72
    quality_gate_threshold: float = 0.72
    minimum_dimension_threshold: float = 0.55
    evidence_sufficiency_threshold: float = 0.40
    max_iterations: int = 8
    max_repairs: int = 3
    checkpoint_every_phase: bool = True


@dataclass(slots=True, frozen=True)
class LoopExecutionRequest:
    user_id: UUID
    session: SessionRecord
    query_text: str
    agent_id: UUID = field(default_factory=uuid4)
    loop_id: UUID = field(default_factory=uuid4)
    scopes: tuple[str, ...] = (
        "tools.read",
        "tools.write",
        "tools.admin",
        "tools.browser",
        "tools.vision",
        "tools.delegate",
    )
    idempotency_key: str | None = None
    deadline_ms: int | None = None
    force_deliberative: bool = False
    stop_after_phase: LoopPhase | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LoopAction:
    action_id: str
    title: str
    instruction: str
    route_target: RouteTarget
    priority: float
    dependencies: tuple[str, ...] = ()
    tool_id: str | None = None
    protocol: str | None = None
    fallback_tool_ids: tuple[str, ...] = ()
    mutates_state: bool = False
    requires_approval: bool = False
    target_tables: tuple[SourceTable, ...] = ()


@dataclass(slots=True, frozen=True)
class LoopPlan:
    actions: tuple[LoopAction, ...]
    constraints: tuple[str, ...]
    validated: bool
    validation_issues: tuple[str, ...]
    rollback_required: bool
    query_response: QueryResolutionResponse


@dataclass(slots=True, frozen=True)
class PredictedResources:
    total_tokens: int
    estimated_cost: float
    estimated_latency_ms: int
    tool_calls: int
    model_calls: int
    phase_budget: LoopPhaseBudget


@dataclass(slots=True, frozen=True)
class EvidenceItem:
    action_id: str
    content: str
    score: float
    token_count: int
    source_table: SourceTable
    source_id: str
    retrieval_method: str
    authority_tier: AuthorityTier
    freshness_seconds: int
    trace_id: str


@dataclass(slots=True, frozen=True)
class EvidenceBundle:
    items: tuple[EvidenceItem, ...]
    total_tokens: int
    quality_score: float
    sufficiency_by_action: Mapping[str, float]
    trace_ids: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ActionOutcome:
    action_id: str
    status: ActionStatus
    tool_id: str | None
    protocol: str | None
    output: Mapping[str, Any]
    latency_ns: int
    cache_hit: bool = False
    error_code: str | None = None
    error_message: str | None = None
    mutated_state: bool = False
    used_fallback: bool = False
    rolled_back: bool = False


@dataclass(slots=True, frozen=True)
class MetacognitiveAssessment:
    score: float
    decision: MetacognitiveDecision
    tool_success_rate: float
    evidence_coverage: float
    plan_adherence: float
    complexity_match: float
    consistency: float
    notes: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class QualityVector:
    correctness: float
    completeness: float
    coherence: float
    safety: float
    grounded: float
    efficient: float

    def min_dimension(self) -> float:
        return min(
            self.correctness,
            self.completeness,
            self.coherence,
            self.safety,
            self.grounded,
            self.efficient,
        )

    def mean_score(self) -> float:
        return (
            self.correctness
            + self.completeness
            + self.coherence
            + self.safety
            + self.grounded
            + self.efficient
        ) / 6.0


@dataclass(slots=True, frozen=True)
class VerificationDefect:
    defect_id: UUID
    root_cause: RootCauseClass
    severity: float
    message: str
    action_id: str | None = None
    retryable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class VerificationVerdict:
    passed: bool
    quality: QualityVector
    defects: tuple[VerificationDefect, ...]
    grounding_score: float
    tool_success_rate: float
    schema_valid: bool
    notes: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class RepairDecision:
    strategy: RepairStrategy
    rationale: str
    estimated_cost_tokens: int
    defect_id: UUID | None = None
    target_action_id: str | None = None
    replacement_tool_id: str | None = None
    followup_query: str | None = None


@dataclass(slots=True, frozen=True)
class DeferredMemoryWrite:
    target_layer: MemoryTier
    target_scope: MemoryScope
    memory_type: MemoryType
    content: str
    rationale: str
    source_action_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CommittedMemoryWrite:
    memory_id: str
    target_layer: MemoryTier
    target_scope: MemoryScope
    content: str
    rationale: str


@dataclass(slots=True, frozen=True)
class LoopPhaseSpan:
    phase: LoopPhase
    duration_ns: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class LoopExecutionState:
    request: LoopExecutionRequest
    history_records: tuple[HistoryRecord, ...] = ()
    memory_records: tuple[MemoryRecord, ...] = ()
    working_items: tuple[WorkingMemoryItem, ...] = ()
    current_phase: LoopPhase = LoopPhase.CONTEXT_ASSEMBLE
    last_completed_phase: LoopPhase | None = None
    cognitive_mode: CognitiveMode | None = None
    subsystem_status: dict[str, SubsystemAvailability] = field(default_factory=dict)
    query_response: QueryResolutionResponse | None = None
    compiled_context: CompiledCycleContext | None = None
    context_invariants: ContextInvariantReport | None = None
    context_trace: CompilationTrace | None = None
    context_archive: list[ContextArchiveEntry] = field(default_factory=list)
    plan: LoopPlan | None = None
    predicted_resources: PredictedResources | None = None
    evidence: EvidenceBundle | None = None
    action_outcomes: list[ActionOutcome] = field(default_factory=list)
    metacognitive: MetacognitiveAssessment | None = None
    verdict: VerificationVerdict | None = None
    repair_history: list[RepairDecision] = field(default_factory=list)
    deferred_writes: list[DeferredMemoryWrite] = field(default_factory=list)
    committed_writes: list[CommittedMemoryWrite] = field(default_factory=list)
    phase_spans: list[LoopPhaseSpan] = field(default_factory=list)
    output_text: str = ""
    upgraded_from_fast_path: bool = False
    escalated: bool = False
    escalation_reason: EscalationReason | None = None
    iteration: int = 0
    repair_count: int = 0
    metacognitive_retry_count: int = 0
    total_tokens_consumed: int = 0
    started_at_ns: int = field(default_factory=perf_counter_ns)


@dataclass(slots=True)
class LoopCheckpointRecord:
    checkpoint_id: UUID
    loop_id: UUID
    user_id: UUID
    session_id: UUID
    last_completed_phase: LoopPhase | None
    next_phase: LoopPhase
    iteration: int
    repair_count: int
    paused: bool
    state_snapshot: LoopExecutionState
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class LoopAuditRecord:
    audit_id: UUID
    loop_id: UUID
    user_id: UUID
    session_id: UUID
    last_completed_phase: LoopPhase | None
    next_phase: LoopPhase | None
    cognitive_mode: CognitiveMode | None
    paused: bool
    success: bool
    escalated: bool
    iterations: int
    repairs: int
    tool_calls: int
    committed_writes: int
    total_tokens: int
    latency_ns: int
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class LoopExecutionResult:
    loop_id: UUID
    last_completed_phase: LoopPhase | None
    next_phase: LoopPhase | None
    cognitive_mode: CognitiveMode | None
    paused: bool
    upgraded_from_fast_path: bool
    query_response: QueryResolutionResponse | None
    compiled_context: CompiledCycleContext | None
    context_invariants: ContextInvariantReport | None
    context_trace: CompilationTrace | None
    context_archive: tuple[ContextArchiveEntry, ...]
    plan: LoopPlan | None
    predicted_resources: PredictedResources | None
    evidence: EvidenceBundle | None
    action_outcomes: tuple[ActionOutcome, ...]
    metacognitive: MetacognitiveAssessment | None
    verdict: VerificationVerdict | None
    repair_history: tuple[RepairDecision, ...]
    committed_writes: tuple[CommittedMemoryWrite, ...]
    checkpoints: tuple[LoopCheckpointRecord, ...]
    audit_id: UUID
    escalation_reason: EscalationReason | None
    output_text: str
    subsystem_status: Mapping[str, SubsystemAvailability]
    latency_ns: int


def new_checkpoint(state: LoopExecutionState, *, paused: bool) -> LoopCheckpointRecord:
    return LoopCheckpointRecord(
        checkpoint_id=uuid4(),
        loop_id=state.request.loop_id,
        user_id=state.request.user_id,
        session_id=state.request.session.session_id,
        last_completed_phase=state.last_completed_phase,
        next_phase=state.current_phase,
        iteration=state.iteration,
        repair_count=state.repair_count,
        paused=paused,
        state_snapshot=state,
    )


def new_loop_audit_record(
    state: LoopExecutionState,
    *,
    paused: bool,
    success: bool,
) -> LoopAuditRecord:
    return LoopAuditRecord(
        audit_id=uuid4(),
        loop_id=state.request.loop_id,
        user_id=state.request.user_id,
        session_id=state.request.session.session_id,
        last_completed_phase=state.last_completed_phase,
        next_phase=state.current_phase if paused else None,
        cognitive_mode=state.cognitive_mode,
        paused=paused,
        success=success,
        escalated=state.escalated,
        iterations=state.iteration,
        repairs=state.repair_count,
        tool_calls=sum(1 for outcome in state.action_outcomes if outcome.tool_id is not None),
        committed_writes=len(state.committed_writes),
        total_tokens=state.total_tokens_consumed,
        latency_ns=max(perf_counter_ns() - state.started_at_ns, 0),
    )


__all__ = [
    "ActionOutcome",
    "CommittedMemoryWrite",
    "DeferredMemoryWrite",
    "EvidenceBundle",
    "EvidenceItem",
    "LoopAction",
    "LoopAuditRecord",
    "LoopCheckpointRecord",
    "LoopExecutionRequest",
    "LoopExecutionResult",
    "LoopExecutionState",
    "LoopPhaseBudget",
    "LoopPhaseSpan",
    "LoopPlan",
    "LoopPolicy",
    "MetacognitiveAssessment",
    "PredictedResources",
    "QualityVector",
    "RepairDecision",
    "VerificationDefect",
    "VerificationVerdict",
    "new_checkpoint",
    "new_loop_audit_record",
]
