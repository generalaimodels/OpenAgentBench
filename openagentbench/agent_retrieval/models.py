"""Typed records for the hybrid multi-stream retrieval engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID, uuid5

from .enums import (
    AuthorityTier,
    HumanFeedback,
    LoopStrategy,
    MemoryType,
    ModelExecutionMode,
    ModelRole,
    Modality,
    OutputStream,
    ProtocolType,
    QualityIssue,
    QueryType,
    ReasoningEffort,
    RetrievalMode,
    Role,
    SignalTopology,
    SourceTable,
    TaskOutcome,
)
from .types import EmbeddingVector, JSONValue


@dataclass(slots=True, frozen=True)
class QueryTemplate:
    sql: str
    params: dict[str, object]


@dataclass(slots=True, frozen=True)
class FragmentLocator:
    source_table: SourceTable
    chunk_id: UUID

    def as_cache_key(self) -> str:
        return f"{self.source_table}:{self.chunk_id}"


@dataclass(slots=True, frozen=True)
class RetrievalBias:
    bm25_weight: float
    dense_weight: float
    memory_weight: float
    history_weight: float

    def normalized(self) -> "RetrievalBias":
        total = self.bm25_weight + self.dense_weight + self.memory_weight + self.history_weight
        if total <= 0.0:
            return RetrievalBias(0.25, 0.25, 0.25, 0.25)
        return RetrievalBias(
            bm25_weight=self.bm25_weight / total,
            dense_weight=self.dense_weight / total,
            memory_weight=self.memory_weight / total,
            history_weight=self.history_weight / total,
        )


@dataclass(slots=True, frozen=True)
class TimeWindow:
    start: datetime
    end: datetime
    label: str


@dataclass(slots=True, frozen=True)
class QueryClassification:
    type: QueryType
    retrieval_bias: RetrievalBias
    requires_decomposition: bool
    requires_coreference_resolution: bool
    requires_temporal_scoping: bool
    session_topic_overlap: float
    output_stream: OutputStream
    output_streams: tuple[OutputStream, ...]
    preferred_modalities: tuple[Modality, ...]
    protocol_hints: tuple[ProtocolType, ...]
    model_execution_mode: ModelExecutionMode = ModelExecutionMode.SINGLE_MODEL
    signal_topology: SignalTopology = SignalTopology.SISO
    reasoning_effort: ReasoningEffort = ReasoningEffort.DIRECT
    loop_strategy: LoopStrategy = LoopStrategy.SINGLE_PASS
    model_roles: tuple[ModelRole, ...] = (ModelRole.GENERATION,)


@dataclass(slots=True, frozen=True)
class UserScope:
    uu_id: UUID
    partition_key: int
    acl_scope: tuple[str, ...] = ()


@dataclass(slots=True)
class SessionTurn:
    session_id: UUID
    uu_id: UUID
    turn_index: int
    role: Role
    content_text: str
    created_at: datetime
    tokens_used: int = 0
    tool_calls: tuple[dict[str, JSONValue], ...] | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    expires_at: datetime | None = None

    def turn_id(self) -> UUID:
        return uuid5(self.session_id, f"turn:{self.turn_index}")


@dataclass(slots=True)
class SessionContext:
    turns: tuple[SessionTurn, ...]
    turn_count: int
    last_user_query: str | None
    topic_trajectory: str
    active_tool_context: tuple[str, ...]
    session_start: datetime | None
    session_duration: timedelta


@dataclass(slots=True, frozen=True)
class HistoryEvidence:
    locator: FragmentLocator
    utility_score: float
    use_count: int = 1
    was_cited: bool = False


@dataclass(slots=True)
class HistoryEntry:
    history_id: UUID
    uu_id: UUID
    query_text: str
    query_embedding: EmbeddingVector | None
    response_summary: str | None
    evidence_used: tuple[HistoryEvidence, ...]
    task_outcome: TaskOutcome
    human_feedback: HumanFeedback
    utility_score: float
    negative_flag: bool
    tags: tuple[str, ...]
    metadata: dict[str, JSONValue]
    created_at: datetime
    session_origin: UUID | None = None


@dataclass(slots=True)
class MemoryEntry:
    memory_id: UUID
    uu_id: UUID
    memory_type: MemoryType
    content_text: str
    content_embedding: EmbeddingVector | None
    authority_tier: AuthorityTier
    confidence: float
    source_provenance: dict[str, JSONValue]
    verified_by: tuple[UUID, ...]
    supersedes: tuple[UUID, ...]
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    access_count: int
    last_accessed_at: datetime | None
    content_hash: bytes
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MemorySummary:
    facts: tuple[MemoryEntry, ...]
    preferences: tuple[MemoryEntry, ...]
    corrections: tuple[MemoryEntry, ...]
    constraints: tuple[MemoryEntry, ...]
    procedures: tuple[MemoryEntry, ...]
    total_items: int
    compressed_text: str


@dataclass(slots=True, frozen=True)
class AugmentedQuery:
    original: str
    resolved: str
    constraints: tuple[MemoryEntry, ...]
    preferences: tuple[MemoryEntry, ...]
    temporal_scope: TimeWindow | None
    session_topic: str
    query_class: QueryClassification


@dataclass(slots=True, frozen=True)
class DocumentDescriptor:
    modality: Modality
    mime_type: str
    uri: str | None
    text_projection: str
    page_index: int | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProtocolTrace:
    protocol_type: ProtocolType
    method: str | None
    service_name: str | None
    rpc_id: str | None
    payload_text: str
    latency_ms: int | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ModelCapabilityProfile:
    model_name: str
    role: ModelRole
    supports_text: bool = True
    supports_documents: bool = False
    supports_images: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_tools: bool = False
    supports_json_schema: bool = False


@dataclass(slots=True, frozen=True)
class SelectedModelPlan:
    query_classification: QueryClassification
    primary_model: ModelCapabilityProfile | None
    role_bindings: dict[ModelRole, ModelCapabilityProfile]


@dataclass(slots=True, frozen=True)
class SubQuery:
    text: str
    target_tables: tuple[SourceTable, ...]
    retrieval_modes: tuple[RetrievalMode, ...]
    priority: float
    temporal_scope: TimeWindow | None
    output_stream: OutputStream = OutputStream.TEXT_EVIDENCE
    preferred_modalities: tuple[Modality, ...] = (Modality.TEXT,)
    protocol_filters: tuple[ProtocolType, ...] = ()
    is_original: bool = False
    is_followup: bool = False


@dataclass(slots=True, frozen=True)
class EmbeddedSubQuery:
    subquery: SubQuery
    plain_embedding: EmbeddingVector
    augmented_embedding: EmbeddingVector
    combined_embedding: EmbeddingVector


@dataclass(slots=True)
class ScoredFragment:
    locator: FragmentLocator
    content: str
    score: float
    retrieval_mode: RetrievalMode
    created_at: datetime
    metadata: dict[str, JSONValue]
    subquery_origin: str


@dataclass(slots=True, frozen=True)
class MIMOStreamResults:
    exact: tuple[ScoredFragment, ...]
    semantic: tuple[ScoredFragment, ...]
    history: tuple[ScoredFragment, ...]
    memory: tuple[ScoredFragment, ...]
    negative: frozenset[FragmentLocator]
    streams_completed: int
    streams_timed_out: int


@dataclass(slots=True)
class FusedCandidate:
    locator: FragmentLocator
    content: str
    fused_score: float
    source_streams: frozenset[str]
    per_stream_ranks: dict[str, int]
    per_stream_scores: dict[str, float]
    metadata: dict[str, JSONValue]
    stream_agreement: int


@dataclass(slots=True)
class RankedFragment:
    locator: FragmentLocator
    content: str
    final_score: float
    cross_encoder_score: float
    fused_score: float
    source_streams: frozenset[str]
    metadata: dict[str, JSONValue]


@dataclass(slots=True, frozen=True)
class ProvenanceOrigin:
    source_table: SourceTable
    source_id: str
    chunk_id: UUID
    extraction_timestamp: datetime
    retrieval_method: str


@dataclass(slots=True, frozen=True)
class ProvenanceScoring:
    cross_encoder_score: float
    fused_score: float
    final_score: float
    source_streams: tuple[str, ...]
    stream_agreement: int
    truncated: bool = False


@dataclass(slots=True, frozen=True)
class ProvenanceTrust:
    authority_tier: AuthorityTier
    confidence: float
    is_memory_validated: bool
    is_historically_proven: bool
    negative_flag: bool
    freshness: timedelta


@dataclass(slots=True, frozen=True)
class ProvenanceRecord:
    origin: ProvenanceOrigin
    scoring: ProvenanceScoring
    trust: ProvenanceTrust
    custody_hash: str


@dataclass(slots=True)
class ProvenanceTaggedFragment:
    locator: FragmentLocator
    content: str
    final_score: float
    cross_encoder_score: float
    fused_score: float
    provenance: ProvenanceRecord
    token_count: int


@dataclass(slots=True, frozen=True)
class BudgetReport:
    budget_total: int
    budget_used: int
    budget_remaining: int
    utilization: float
    fragments_included: int
    fragments_excluded: int
    truncation_applied: bool


@dataclass(slots=True, frozen=True)
class QualityAssessment:
    score: float
    issue: QualityIssue | None


@dataclass(slots=True, frozen=True)
class RankingConfig:
    k_exact_per_table: int = 30
    k_semantic_per_table: int = 30
    k_history_queries: int = 20
    k_negative: int = 10
    k_rerank: int = 50
    k_final: int = 10
    diversity_lambda: float = 0.6
    beta_cross: float = 0.7
    beta_fused: float = 0.3
    negative_penalty: float = 0.1
    source_diversity_bonus: float = 0.05
    agreement_multiplier: float = 0.3
    bm25_exact_weight: float = 0.7
    trigram_exact_weight: float = 0.3
    rrf_kappa: int = 60
    utility_threshold: float = 0.5
    similarity_threshold: float = 0.6
    negative_similarity_threshold: float = 0.7
    negative_utility_threshold: float = 0.2


@dataclass(slots=True, frozen=True)
class QualityConfig:
    min_quality_threshold: float = 0.7
    max_iterations: int = 1
    min_useful_tokens: int = 50
    provenance_overhead_per_fragment: int = 40
    staleness_limit_hours: float = 24.0 * 30.0


@dataclass(slots=True, frozen=True)
class RetrievalConfig:
    partition_modulus: int = 256
    max_session_turns: int = 32
    max_memory_items: int = 64
    max_subqueries: int = 5
    memory_summary_token_budget: int = 500
    embedding_dimension: int = 256
    latency_deadline_ms: int = 250
    min_refinement_budget_ms: int = 25
    history_decay_days: float = 30.0
    memory_decay_days: float = 90.0
    session_decay_hours: float = 1.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass(slots=True)
class EvidenceResponse:
    fragments: tuple[ProvenanceTaggedFragment, ...]
    total_candidates_considered: int
    latency_ms: float
    source_coverage: dict[str, int]
    budget_report: BudgetReport
    quality_assessment: QualityAssessment
    streams_completed: int
    streams_timed_out: int
    cache_hit_ratio: float
    retrieval_trace_id: str
