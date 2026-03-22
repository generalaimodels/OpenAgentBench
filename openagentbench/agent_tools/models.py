"""Typed records for registry, dispatch, state, and compatibility surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from .enums import (
    ApprovalStatus,
    AuthDecision,
    Environment,
    ErrorCode,
    InvocationStatus,
    MutationClass,
    TimeoutClass,
    ToolSourceType,
    ToolStatus,
    TypeClass,
)
from .types import JSONSchema, JSONValue, ToolHandler


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class QueryTemplate:
    sql: str
    params: Mapping[str, object]


@dataclass(slots=True, frozen=True)
class SchemaValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class AuthContract:
    required_scopes: tuple[str, ...]
    approval_required: bool = False
    redacted_fields: tuple[str, ...] = ()
    description: str = ""


@dataclass(slots=True, frozen=True)
class IdempotencySpec:
    enabled: bool = True
    dedup_window_seconds: int = 300
    key_fields: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ObservabilityContract:
    metric_prefix: str
    emit_traces: bool = True
    emit_metrics: bool = True
    emit_audit_logs: bool = True


@dataclass(slots=True, frozen=True)
class SideEffectManifest:
    resources: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    reversible: bool = True
    verification_hints: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class RegistrationProvenance:
    source_endpoint: str
    source_type: ToolSourceType
    registrar_identity: str
    registered_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class ToolSchemaSummary:
    tool_id: str
    compressed_description: str
    mutation_class: MutationClass
    type_class: TypeClass
    health_score: float
    token_cost_estimate: int


@dataclass(slots=True, frozen=True)
class CompositeStep:
    step_id: str
    tool_id: str
    depends_on: tuple[str, ...] = ()
    input_bindings: Mapping[str, str] = field(default_factory=dict)
    static_params: Mapping[str, JSONValue] = field(default_factory=dict)
    retry_budget: int = 0
    compensation_tool_id: str | None = None
    compensation_params: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CompositeToolSpec:
    steps: tuple[CompositeStep, ...]
    output_bindings: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ToolDescriptor:
    tool_id: str
    version: str
    type_class: TypeClass
    input_schema: JSONSchema
    output_schema: JSONSchema
    error_schema: JSONSchema
    auth_contract: AuthContract
    timeout_class: TimeoutClass
    idempotency_spec: IdempotencySpec
    mutation_class: MutationClass
    observability_contract: ObservabilityContract
    source_endpoint: str
    source_type: ToolSourceType
    compressed_description: str
    handler: ToolHandler | None = None
    side_effect_manifest: SideEffectManifest | None = None
    health_score: float = 1.0
    status: ToolStatus = ToolStatus.ACTIVE
    token_cost_estimate: int = 64
    cache_ttl_seconds: int = 0
    deprecation_notice: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    contract_tests_hash: str = "inline"
    composite_spec: CompositeToolSpec | None = None

    def as_summary(self) -> ToolSchemaSummary:
        return ToolSchemaSummary(
            tool_id=self.tool_id,
            compressed_description=self.compressed_description,
            mutation_class=self.mutation_class,
            type_class=self.type_class,
            health_score=self.health_score,
            token_cost_estimate=self.token_cost_estimate,
        )


@dataclass(slots=True, frozen=True)
class ExecutionContext:
    user_id: UUID
    agent_id: UUID
    session_id: UUID | None
    scopes: tuple[str, ...]
    trace_id: str
    environment: Environment = Environment.DEVELOPMENT
    trust_level: int = 0
    request_started_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class ToolInvocationRequest:
    tool_id: str
    params: Mapping[str, JSONValue]
    context: ExecutionContext
    idempotency_key: str | None = None
    version_spec: str | None = None
    requested_deadline_ms: int | None = None
    approval_ticket_id: UUID | None = None


@dataclass(slots=True, frozen=True)
class ErrorEnvelope:
    code: ErrorCode
    message: str
    retryable: bool
    details: Mapping[str, JSONValue] = field(default_factory=dict)
    retry_after_ms: int | None = None


@dataclass(slots=True, frozen=True)
class SuccessEnvelope:
    data: JSONValue
    provenance: Mapping[str, JSONValue]
    execution_metadata: Mapping[str, JSONValue]
    source: str | None = None


@dataclass(slots=True, frozen=True)
class PendingEnvelope:
    approval_ticket_id: UUID
    poll_token: str
    status: ApprovalStatus = ApprovalStatus.PENDING


@dataclass(slots=True, frozen=True)
class ToolInvocationResponse:
    status: InvocationStatus
    success: SuccessEnvelope | None = None
    error: ErrorEnvelope | None = None
    pending: PendingEnvelope | None = None

    @staticmethod
    def from_success(
        data: JSONValue,
        *,
        provenance: Mapping[str, JSONValue],
        execution_metadata: Mapping[str, JSONValue],
        source: str | None = None,
    ) -> "ToolInvocationResponse":
        return ToolInvocationResponse(
            status=InvocationStatus.SUCCESS,
            success=SuccessEnvelope(
                data=data,
                provenance=provenance,
                execution_metadata=execution_metadata,
                source=source,
            ),
        )

    @staticmethod
    def from_error(
        code: ErrorCode,
        message: str,
        *,
        retryable: bool,
        details: Mapping[str, JSONValue] | None = None,
        retry_after_ms: int | None = None,
        timeout: bool = False,
    ) -> "ToolInvocationResponse":
        return ToolInvocationResponse(
            status=InvocationStatus.TIMEOUT if timeout else InvocationStatus.ERROR,
            error=ErrorEnvelope(
                code=code,
                message=message,
                retryable=retryable,
                details=dict(details or {}),
                retry_after_ms=retry_after_ms,
            ),
        )

    @staticmethod
    def from_pending(ticket_id: UUID, *, poll_token: str) -> "ToolInvocationResponse":
        return ToolInvocationResponse(
            status=InvocationStatus.PENDING,
            pending=PendingEnvelope(approval_ticket_id=ticket_id, poll_token=poll_token),
        )


@dataclass(slots=True)
class ToolAuditRecord:
    audit_id: UUID
    trace_id: str
    tool_id: str
    tool_version: str
    caller_id: UUID
    agent_id: UUID
    session_id: UUID | None
    status: InvocationStatus
    auth_decision: AuthDecision
    input_hash: bytes
    latency_ms: int
    mutation_class: MutationClass
    error_code: str | None = None
    token_cost: int = 0
    compute_cost: float = 0.0
    side_effects: Mapping[str, JSONValue] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class ToolApprovalTicket:
    ticket_id: UUID
    tool_id: str
    params_redacted: Mapping[str, JSONValue]
    requested_by: UUID
    agent_id: UUID
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    resolution_by: UUID | None = None
    resolution_at: datetime | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCacheEntry:
    cache_key: str
    tool_id: str
    user_id: UUID
    result_data: JSONValue
    expires_at: datetime
    hit_count: int = 0
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class IdempotencyRecord:
    key: str
    tool_id: str
    user_id: UUID
    session_id: UUID | None
    response: ToolInvocationResponse
    expires_at: datetime
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True, frozen=True)
class AdmissionResult:
    admitted: bool
    violations: tuple[str, ...] = ()
    already_registered: bool = False


@dataclass(slots=True, frozen=True)
class ParsedToolCall:
    call_id: str
    tool_id: str
    params: Mapping[str, JSONValue]
    model_format: str
    valid: bool
    error: str | None = None


@dataclass(slots=True, frozen=True)
class ToolResultTurn:
    call_id: str
    tool_id: str
    output: JSONValue


@dataclass(slots=True, frozen=True)
class ToolEndpointCompatibilityReport:
    retrieval_report: Any
    memory_report: Any
    agent_data_endpoints: Sequence[Any]
    vllm_endpoints: Sequence[Any]
    tool_definitions: tuple[Any, ...]
    openai_tools_format: dict[str, Any]
    vllm_tools_format: dict[str, Any]
    anthropic_tools_format: dict[str, Any]
    google_tools_format: dict[str, Any]
    openai_responses_request: dict[str, Any]
    openai_chat_request: dict[str, Any]
    openai_realtime_request: dict[str, Any]
    openai_audio_speech_request: dict[str, Any]
    openai_audio_transcription_request: dict[str, Any]
    openai_audio_translation_request: dict[str, Any]
    openai_image_generation_request: dict[str, Any]
    openai_image_edit_request: dict[str, Any]
    openai_video_generation_request: dict[str, Any]
    gemini_generate_content_request: dict[str, Any]
    gemini_count_tokens_request: dict[str, Any]
    jsonrpc_invoke_request: dict[str, Any]
    jsonrpc_admin_request: dict[str, Any]
    mcp_initialize_request: dict[str, Any]
    mcp_list_tools_request: dict[str, Any]
    mcp_call_tool_request: dict[str, Any]
    a2a_task_request: dict[str, Any]
    openai_responses_tool_result_items: tuple[dict[str, Any], ...]
    openai_chat_tool_messages: tuple[dict[str, Any], ...]
    anthropic_tool_result_blocks: tuple[dict[str, Any], ...]
    google_tool_result_parts: tuple[dict[str, Any], ...]
    parsed_openai_responses_tool_calls: tuple[ParsedToolCall, ...]
    parsed_openai_chat_tool_calls: tuple[ParsedToolCall, ...]


__all__ = [
    "AdmissionResult",
    "AuthContract",
    "CompositeStep",
    "CompositeToolSpec",
    "ErrorEnvelope",
    "ExecutionContext",
    "IdempotencyRecord",
    "IdempotencySpec",
    "ObservabilityContract",
    "ParsedToolCall",
    "PendingEnvelope",
    "QueryTemplate",
    "RegistrationProvenance",
    "SchemaValidationResult",
    "SideEffectManifest",
    "SuccessEnvelope",
    "ToolApprovalTicket",
    "ToolAuditRecord",
    "ToolCacheEntry",
    "ToolDescriptor",
    "ToolEndpointCompatibilityReport",
    "ToolInvocationRequest",
    "ToolInvocationResponse",
    "ToolResultTurn",
    "ToolSchemaSummary",
]
