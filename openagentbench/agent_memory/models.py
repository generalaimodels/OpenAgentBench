"""Typed records for working memory, checkpoints, promotion, and compiled memory context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from openagentbench.agent_data import HistoryRecord, MemoryRecord, SessionRecord
from openagentbench.agent_data.enums import MemoryTier
from openagentbench.agent_retrieval import AuthorityTier, MemoryType, Modality, QueryClassification, SelectedModelPlan

from .enums import (
    ConflictStatus,
    MemoryOperation,
    ProcedureMatchMode,
    ProcedureStatus,
    PromotionAction,
    PromotionSource,
)
from .types import JSONValue


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class QueryTemplate:
    sql: str
    params: dict[str, object]


@dataclass(slots=True, frozen=True)
class BudgetAllocation:
    total_budget: int
    working_budget: int
    session_budget: int
    episodic_budget: int
    semantic_budget: int
    procedural_budget: int
    multimodal_budget: int
    reserve_budget: int

    def allocated_budget(self) -> int:
        return (
            self.working_budget
            + self.session_budget
            + self.episodic_budget
            + self.semantic_budget
            + self.procedural_budget
            + self.multimodal_budget
            + self.reserve_budget
        )


@dataclass(slots=True)
class WorkingMemoryItem:
    item_id: UUID
    user_id: UUID
    session_id: UUID
    step_id: UUID
    content_text: str
    token_count: int
    modality: Modality = Modality.TEXT
    utility_score: float = 0.0
    dependency_count: int = 0
    ttl_remaining: int = -1
    carry_forward: bool = False
    binary_ref: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def externalized_label(self) -> str:
        ref = self.binary_ref or f"memory://external/{self.item_id}"
        return f"[Externalized {self.modality.value} artifact: {ref}]"


@dataclass(slots=True, frozen=True)
class WorkingMemorySnapshot:
    items: tuple[WorkingMemoryItem, ...]
    token_used: int
    capacity: int


@dataclass(slots=True, frozen=True)
class SessionTurnMarkers:
    correction_flag: bool
    decision_flag: bool
    segment_boundary: bool


@dataclass(slots=True)
class SessionCheckpointRecord:
    checkpoint_id: UUID
    user_id: UUID
    session_id: UUID
    checkpoint_seq: int
    summary_text: str
    summary_version: int
    turn_count: int
    working_item_ids: tuple[UUID, ...]
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class MemoryFragment:
    layer: MemoryTier
    content: str
    token_count: int
    score: float
    source_id: UUID | None
    modality: Modality = Modality.TEXT
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProcedureMatchResult:
    mode: ProcedureMatchMode
    procedure_id: UUID | None
    match_confidence: float
    estimated_token_savings: int
    content_text: str | None = None


@dataclass(slots=True)
class PromotionCandidate:
    user_id: UUID
    source_id: UUID
    source_layer: MemoryTier
    memory_type: MemoryType
    content_text: str
    token_count: int
    novelty_score: float
    correctness_score: float
    reusability_score: float
    modality: Modality = Modality.TEXT
    session_id: UUID | None = None
    promotion_source: PromotionSource = PromotionSource.AUTOMATED
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PromotionDecision:
    action: PromotionAction
    target_layer: MemoryTier | None
    promotion_score: float
    reason: str


@dataclass(slots=True)
class MemoryAuditRecord:
    audit_id: UUID
    user_id: UUID
    operation: MemoryOperation
    layer: MemoryTier | None
    item_id: UUID | None
    result: str
    caller_id: UUID | None = None
    session_id: UUID | None = None
    latency_ms: int | None = None
    token_delta: int = 0
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class MemoryCacheEntry:
    cache_key: str
    user_id: UUID
    layer: MemoryTier
    payload: dict[str, JSONValue]
    expires_at: datetime
    hit_count: int = 0
    embedding_bucket: str | None = None
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class MemoryCompileRequest:
    user_id: UUID
    session: SessionRecord
    query_text: str
    total_budget: int
    step_id: UUID | None = None
    classification: QueryClassification | None = None
    model_plan: SelectedModelPlan | None = None
    system_prompt_text: str | None = None
    tool_budget: int = 0
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True)
class CompiledMemoryContext:
    messages: list[dict[str, Any]]
    budget: BudgetAllocation
    selected_working: list[WorkingMemoryItem]
    selected_fragments: list[MemoryFragment]
    classification: QueryClassification
    total_tokens: int


@dataclass(slots=True, frozen=True)
class MemoryRepositorySnapshot:
    session: SessionRecord
    history: tuple[HistoryRecord, ...]
    memories: tuple[MemoryRecord, ...]
    working_items: tuple[WorkingMemoryItem, ...]
    checkpoints: tuple[SessionCheckpointRecord, ...]


@dataclass(slots=True, frozen=True)
class MemoryWritePlan:
    memory_tier: MemoryTier
    memory_type: MemoryType
    authority_tier: AuthorityTier
    procedure_status: ProcedureStatus | None = None
    conflict_status: ConflictStatus = ConflictStatus.NONE


@dataclass(slots=True)
class MemoryMaintenanceReport:
    expired_count: int = 0
    archived_count: int = 0
    deduplicated_count: int = 0
    stale_count: int = 0
    cache_evictions: int = 0
    generated_at: datetime = field(default_factory=_utc_now)
    elapsed: timedelta = timedelta()


def new_checkpoint(
    *,
    user_id: UUID,
    session_id: UUID,
    checkpoint_seq: int,
    summary_text: str,
    summary_version: int,
    turn_count: int,
    working_items: Sequence[WorkingMemoryItem],
    metadata: Mapping[str, JSONValue] | None = None,
) -> SessionCheckpointRecord:
    return SessionCheckpointRecord(
        checkpoint_id=uuid4(),
        user_id=user_id,
        session_id=session_id,
        checkpoint_seq=checkpoint_seq,
        summary_text=summary_text,
        summary_version=summary_version,
        turn_count=turn_count,
        working_item_ids=tuple(item.item_id for item in working_items),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "BudgetAllocation",
    "CompiledMemoryContext",
    "MemoryAuditRecord",
    "MemoryCacheEntry",
    "MemoryCompileRequest",
    "MemoryFragment",
    "MemoryMaintenanceReport",
    "MemoryRepositorySnapshot",
    "MemoryWritePlan",
    "ProcedureMatchResult",
    "PromotionCandidate",
    "PromotionDecision",
    "QueryTemplate",
    "SessionCheckpointRecord",
    "SessionTurnMarkers",
    "WorkingMemoryItem",
    "WorkingMemorySnapshot",
    "new_checkpoint",
]
