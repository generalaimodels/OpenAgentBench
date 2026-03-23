"""Typed records for the universal agent SDK connector fabric."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID, uuid4

from openagentbench.agent_context import CompiledCycleContext
from openagentbench.agent_data import SessionRecord
from openagentbench.agent_tools import (
    ExecutionContext,
    MutationClass,
    ToolDescriptor,
    ToolInvocationResponse,
    ToolSourceType,
)

from .enums import (
    AuthType,
    ConnectorDomain,
    ConnectorHealth,
    InteractionModality,
    OperationState,
    OsType,
    ProtocolName,
    ProviderTarget,
    ResourceScope,
    SafetyLevel,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class CapabilityMatrix:
    tcp_sockets: bool = True
    unix_domain_sockets: bool = False
    process_spawn: bool = True
    file_system: bool = True
    screen_capture: bool = False
    clipboard: bool = False
    named_pipes: bool = False


@dataclass(slots=True, frozen=True)
class OsPlatformSnapshot:
    os_type: OsType
    hostname: str
    environment: Mapping[str, str]
    current_user: str
    capabilities: CapabilityMatrix


@dataclass(slots=True, frozen=True)
class EndpointDescriptor:
    endpoint_id: str
    address: str
    protocol_hints: tuple[ProtocolName, ...]
    auth_type: AuthType = AuthType.NONE
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProtocolCapabilities:
    supports_streaming: bool
    supports_bidirectional: bool
    supports_binary: bool
    auth_types: tuple[AuthType, ...]


@dataclass(slots=True, frozen=True)
class UniversalRequest:
    method: str
    headers: Mapping[str, str]
    body: bytes
    idempotency_key: str | None
    deadline: timedelta
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UniversalResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    latency: timedelta
    protocol: ProtocolName
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ConnectorOperation:
    operation_id: str
    description: str
    tool_id: str
    connector_id: str
    domain: ConnectorDomain
    modality: InteractionModality
    protocol: ProtocolName
    source_type: ToolSourceType
    mutation_class: MutationClass
    required_scopes: tuple[str, ...]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    token_cost_estimate: int
    destructive: bool = False
    supports_streaming: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConnectorDescriptor:
    connector_id: str
    domain: ConnectorDomain
    modality: InteractionModality
    protocol: ProtocolName
    endpoint: EndpointDescriptor
    operations: tuple[ConnectorOperation, ...]
    health: ConnectorHealth = ConnectorHealth.HEALTHY
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AuthCredential:
    credential_id: UUID
    auth_type: AuthType
    token: str
    scopes: frozenset[str]
    issued_at: datetime
    expires_at: datetime | None = None
    source: str = "memory"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_expired(self, *, at: datetime, safety_window: timedelta = timedelta(seconds=60)) -> bool:
        if self.expires_at is None:
            return False
        return at + safety_window >= self.expires_at

    def narrow_to(self, scopes: Sequence[str]) -> "AuthCredential":
        requested = frozenset(scope for scope in scopes if scope in self.scopes)
        return AuthCredential(
            credential_id=self.credential_id,
            auth_type=self.auth_type,
            token=self.token,
            scopes=requested,
            issued_at=self.issued_at,
            expires_at=self.expires_at,
            source=self.source,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True, frozen=True)
class BudgetLimit:
    scope: ResourceScope
    api_calls: int
    tokens: int
    compute_seconds: float
    monetary_cost_usd: float


@dataclass(slots=True, frozen=True)
class CostRecord:
    api_calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    compute_seconds: float = 0.0
    storage_bytes: int = 0
    network_bytes: int = 0
    monetary_cost_usd: float = 0.0


@dataclass(slots=True, frozen=True)
class BudgetApproval:
    approved: bool
    remaining_api_calls: int
    remaining_tokens: int
    remaining_compute_seconds: float
    remaining_cost_usd: float
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class RoutedAction:
    connector_id: str
    operation_id: str
    tool_id: str
    modality: InteractionModality
    protocol: ProtocolName
    auth_type: AuthType
    required_scopes: tuple[str, ...]
    estimated_latency_ms: int
    estimated_cost_usd: float


@dataclass(slots=True, frozen=True)
class OperationHandle:
    handle_id: UUID
    connector_id: str
    operation_id: str
    state: OperationState
    created_at: datetime
    poll_token: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OperationStatus:
    state: OperationState
    progress: float | None = None
    message: str | None = None
    eta: timedelta | None = None
    intermediate_result: Any = None


@dataclass(slots=True, frozen=True)
class AgentSdkInvocationRequest:
    operation: str
    params: Mapping[str, Any]
    connector_id: str | None = None
    task_hint: str | None = None
    target_system: str | None = None
    required_scopes: tuple[str, ...] = ()
    idempotency_key: str | None = None
    requested_deadline_ms: int | None = None
    safety_level: SafetyLevel = SafetyLevel.CAUTIOUS
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentSdkInvocationResult:
    request: AgentSdkInvocationRequest
    route: RoutedAction
    response: ToolInvocationResponse
    execution_context: ExecutionContext
    cost_record: CostRecord
    operation_handle: OperationHandle | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentTaskStep:
    step_id: str
    operation: str
    params: Mapping[str, Any]
    connector_id: str | None = None
    task_hint: str | None = None
    required_scopes: tuple[str, ...] = ()
    allow_failure: bool = False
    requested_deadline_ms: int | None = None
    idempotency_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentTaskSpec:
    task_id: UUID
    objective: str
    steps: tuple[AgentTaskStep, ...]
    provider_target: ProviderTarget = ProviderTarget.OPENAI
    token_budget: int = 1_024
    max_step_iterations: int = 3
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentTaskResult:
    task_id: UUID
    objective: str
    results: tuple[AgentSdkInvocationResult, ...]
    compiled_context: CompiledCycleContext | None
    total_cost: CostRecord
    completed: bool
    failures: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ProjectedToolSurface:
    tool_definitions: tuple[dict[str, Any], ...]
    openai_tools: dict[str, Any]
    vllm_tools: dict[str, Any]
    mcp_descriptors: tuple[dict[str, Any], ...]
    function_descriptors: tuple[dict[str, Any], ...]


@dataclass(slots=True, frozen=True)
class ProviderClientConfig:
    provider: ProviderTarget
    model: str
    embedding_model: str | None = None
    base_url: str | None = None
    prefer_responses_api: bool = True
    reasoning_effort: str | None = None
    store: bool | None = None


@dataclass(slots=True)
class ProviderSuite:
    config: ProviderClientConfig
    text_model: Any | None = None
    embedding_provider: Any | None = None


@dataclass(slots=True, frozen=True)
class AgentSdkCompatibilityReport:
    connector_count: int
    operation_count: int
    provider_targets: tuple[ProviderTarget, ...]
    tool_report: Any
    context_report: Any
    openai_responses_request: dict[str, Any]
    openai_chat_request: dict[str, Any]
    openai_realtime_request: dict[str, Any]
    vllm_responses_request: dict[str, Any]
    vllm_chat_request: dict[str, Any]
    mcp_initialize_request: dict[str, Any]
    mcp_list_tools_request: dict[str, Any]
    mcp_call_tool_request: dict[str, Any]
    jsonrpc_invoke_request: dict[str, Any]
    jsonrpc_registry_request: dict[str, Any]
    a2a_task_request: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ConnectorProjectionRecord:
    connector_id: str
    tool_id: str
    projected_name: str
    protocol: ProtocolName
    modality: InteractionModality
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentSdkSnapshot:
    session: SessionRecord
    platform: OsPlatformSnapshot
    connectors: tuple[ConnectorDescriptor, ...]
    tool_descriptors: tuple[ToolDescriptor, ...]
    generated_at: datetime = field(default_factory=_utc_now)


def new_task_spec(
    *,
    objective: str,
    steps: Sequence[AgentTaskStep],
    provider_target: ProviderTarget = ProviderTarget.OPENAI,
    token_budget: int = 1_024,
    metadata: Mapping[str, Any] | None = None,
) -> AgentTaskSpec:
    return AgentTaskSpec(
        task_id=uuid4(),
        objective=objective,
        steps=tuple(steps),
        provider_target=provider_target,
        token_budget=token_budget,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "AgentSdkCompatibilityReport",
    "AgentSdkInvocationRequest",
    "AgentSdkInvocationResult",
    "AgentSdkSnapshot",
    "AgentTaskResult",
    "AgentTaskSpec",
    "AgentTaskStep",
    "AuthCredential",
    "BudgetApproval",
    "BudgetLimit",
    "CapabilityMatrix",
    "ConnectorDescriptor",
    "ConnectorOperation",
    "ConnectorProjectionRecord",
    "CostRecord",
    "EndpointDescriptor",
    "OperationHandle",
    "OperationStatus",
    "OsPlatformSnapshot",
    "ProjectedToolSurface",
    "ProtocolCapabilities",
    "ProviderClientConfig",
    "ProviderSuite",
    "RoutedAction",
    "UniversalRequest",
    "UniversalResponse",
    "new_task_spec",
]
