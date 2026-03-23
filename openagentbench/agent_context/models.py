"""Typed records for canonical cyclic context compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID, uuid4

from openagentbench.agent_data.models import HistoryRecord, MemoryRecord, SessionRecord
from openagentbench.agent_memory.models import WorkingMemoryItem
from openagentbench.agent_tools.models import ToolResultTurn


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


ContextProviderName = Literal["openai_responses", "openai_chat", "vllm_responses", "vllm_chat"]


@dataclass(slots=True, frozen=True)
class PolicyKernel:
    content: str
    directives: tuple[str, ...]
    token_count: int
    kernel_hash: str
    phase: str
    version: str = "1.0.0"


@dataclass(slots=True, frozen=True)
class TaskStateProjection:
    content: str
    token_count: int
    phase: str
    cycle_number: int
    current_step: str | None = None
    completed_action_ids: tuple[str, ...] = ()
    pending_action_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ContextSection:
    name: str
    role: str
    content: str
    token_count: int
    mutable: bool
    source_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EvidenceProjectionItem:
    evidence_id: str
    content: str
    source_table: str
    source_id: str
    retrieval_method: str
    authority: str
    freshness_seconds: int
    score: float
    token_count: int
    provenance_tag: str


@dataclass(slots=True, frozen=True)
class EvidenceProjection:
    items: tuple[EvidenceProjectionItem, ...]
    total_tokens: int
    admitted_count: int
    rejected_without_provenance: int
    token_budget: int


@dataclass(slots=True, frozen=True)
class MemoryProjectionItem:
    memory_id: str
    content: str
    layer: str
    scope: str
    score: float
    token_count: int
    source_kind: str


@dataclass(slots=True, frozen=True)
class MemoryProjection:
    items: tuple[MemoryProjectionItem, ...]
    messages: tuple[dict[str, Any], ...]
    total_tokens: int
    token_budget: int
    selected_working_ids: tuple[str, ...] = ()
    selected_memory_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class CycleFilterResult:
    sections: tuple[ContextSection, ...]
    history_messages: tuple[dict[str, Any], ...]
    noise_evicted_tokens: int
    evicted_item_ids: tuple[str, ...]
    duplication_rate: float
    staleness_index: float
    signal_density: float
    rejected_without_provenance: int
    total_tokens: int


@dataclass(slots=True, frozen=True)
class ContextInvariantReport:
    passed: bool
    token_budget_ok: bool
    provenance_coverage_ok: bool
    no_instruction_leakage: bool
    signal_density_ok: bool
    duplication_ok: bool
    staleness_ok: bool
    archive_emitted: bool
    signal_density: float
    duplication_rate: float
    staleness_index: float
    violations: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class CompilationTrace:
    compile_id: UUID
    session_id: UUID
    provider: ContextProviderName
    cycle_number: int
    started_at: datetime
    completed_at: datetime
    input_hash: str
    output_hash: str
    archive_hash: str
    total_tokens: int
    section_allocations: Mapping[str, int]
    stable_prefix_tokens: int
    signal_density: float
    duplication_rate: float
    staleness_index: float
    noise_evicted_tokens: int
    rejected_without_provenance: int


@dataclass(slots=True, frozen=True)
class ContextProviderProfile:
    provider: ContextProviderName
    request_format: str
    supports_streaming: bool
    supports_tools: bool
    stable_prefix_sections: tuple[str, ...]
    unsupported_request_fields: tuple[str, ...] = ()
    notes: str = ""


@dataclass(slots=True, frozen=True)
class ContextArchiveEntry:
    archive_id: UUID
    session_id: UUID
    user_id: UUID
    provider: ContextProviderName
    input_hash: str
    output_hash: str
    archive_hash: str
    created_at: datetime
    section_hashes: Mapping[str, str]
    section_allocations: Mapping[str, int]
    invariant_report: ContextInvariantReport
    trace: CompilationTrace


@dataclass(slots=True, frozen=True)
class ContextCompatibilityReport:
    provider_profiles: tuple[ContextProviderProfile, ...]
    openai_http_endpoints: tuple[Any, ...]
    vllm_endpoints: tuple[Any, ...]
    openai_responses_request: dict[str, Any]
    openai_chat_request: dict[str, Any]
    vllm_responses_request: dict[str, Any]
    vllm_chat_request: dict[str, Any]
    openai_tool_result_items: tuple[dict[str, Any], ...]
    openai_chat_tool_result_messages: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class CompiledCycleContext:
    provider_profile: ContextProviderProfile
    policy_kernel: PolicyKernel
    task_state: TaskStateProjection
    evidence_projection: EvidenceProjection
    memory_projection: MemoryProjection
    filter_result: CycleFilterResult
    messages: tuple[dict[str, Any], ...]
    responses_input: tuple[dict[str, Any], ...]
    chat_messages: tuple[dict[str, Any], ...]
    openai_responses_request: dict[str, Any]
    openai_chat_request: dict[str, Any]
    vllm_responses_request: dict[str, Any]
    vllm_chat_request: dict[str, Any]
    total_tokens: int
    stable_prefix_tokens: int
    output_hash: str
    section_hashes: Mapping[str, str]
    trace: CompilationTrace
    invariant_report: ContextInvariantReport
    archive_entry: ContextArchiveEntry


@dataclass(slots=True, frozen=True)
class ContextCompileRequest:
    user_id: UUID
    session: SessionRecord
    query_text: str
    history: tuple[HistoryRecord, ...] = ()
    memories: tuple[MemoryRecord, ...] = ()
    working_items: tuple[WorkingMemoryItem, ...] = ()
    evidence_items: tuple[Any, ...] = ()
    tool_result_turns: tuple[ToolResultTurn | Mapping[str, Any], ...] = ()
    active_tools: tuple[dict[str, Any], ...] = ()
    provider: ContextProviderName = "openai_responses"
    cycle_number: int = 0
    current_phase: str = "context_assemble"
    current_step: str | None = None
    total_budget: int | None = None
    response_reserve: int | None = None
    tool_budget: int = 0
    history_budget: int | None = None
    memory_budget: int | None = None
    evidence_budget: int | None = None
    system_prompt_text: str | None = None
    phase_directives: tuple[str, ...] = ()
    security_rules: tuple[str, ...] = ()
    completed_action_ids: tuple[str, ...] = ()
    pending_action_ids: tuple[str, ...] = ()
    prior_compiled_context: CompiledCycleContext | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def new_context_archive_entry(
    *,
    session_id: UUID,
    user_id: UUID,
    provider: ContextProviderName,
    input_hash: str,
    output_hash: str,
    archive_hash: str,
    section_hashes: Mapping[str, str],
    section_allocations: Mapping[str, int],
    invariant_report: ContextInvariantReport,
    trace: CompilationTrace,
) -> ContextArchiveEntry:
    return ContextArchiveEntry(
        archive_id=uuid4(),
        session_id=session_id,
        user_id=user_id,
        provider=provider,
        input_hash=input_hash,
        output_hash=output_hash,
        archive_hash=archive_hash,
        created_at=_utc_now(),
        section_hashes=dict(section_hashes),
        section_allocations=dict(section_allocations),
        invariant_report=invariant_report,
        trace=trace,
    )


__all__ = [
    "CompiledCycleContext",
    "CompilationTrace",
    "ContextArchiveEntry",
    "ContextCompatibilityReport",
    "ContextCompileRequest",
    "ContextInvariantReport",
    "ContextProviderName",
    "ContextProviderProfile",
    "ContextSection",
    "CycleFilterResult",
    "EvidenceProjection",
    "EvidenceProjectionItem",
    "MemoryProjection",
    "MemoryProjectionItem",
    "PolicyKernel",
    "TaskStateProjection",
    "new_context_archive_entry",
]
