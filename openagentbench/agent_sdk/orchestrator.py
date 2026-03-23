"""Universal SDK runtime built on top of the existing OpenAgentBench modules."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from openagentbench.agent_context import ContextCompileRequest, InMemoryContextRepository, compile_context
from openagentbench.agent_data import HistoryRecord, MemoryRecord, SessionRecord
from openagentbench.agent_memory import WorkingMemoryItem
from openagentbench.agent_tools import (
    ErrorCode,
    ExecutionContext,
    OpenAgentBenchToolSuite,
    ToolDescriptor,
    ToolExecutionEngine,
    ToolInvocationRequest,
    ToolInvocationResponse,
    format_tools_for_model,
)

from .auth import AuthManager
from .enums import (
    ConnectorHealth,
    InteractionModality,
    OperationState,
    OsType,
    ProviderTarget,
    ResourceScope,
)
from .governor import CostGovernor
from .models import (
    AgentSdkInvocationRequest,
    AgentSdkInvocationResult,
    AgentSdkSnapshot,
    AgentTaskResult,
    AgentTaskSpec,
    AuthCredential,
    BudgetApproval,
    BudgetLimit,
    CapabilityMatrix,
    ConnectorDescriptor,
    ConnectorProjectionRecord,
    CostRecord,
    OperationHandle,
    OperationStatus,
    OsPlatformSnapshot,
    ProjectedToolSurface,
    ProviderClientConfig,
    ProviderSuite,
    RoutedAction,
)
from .providers import AgentSdkProviderFactory
from .registry import SdkConnectorRegistry, ToolBackedConnectorRuntime
from .repository import InMemoryAgentSdkRepository

_DEFAULT_SCOPES = (
    "tools.read",
    "tools.write",
    "tools.admin",
    "tools.browser",
    "tools.vision",
    "tools.delegate",
    "tools.terminal",
)


def _platform_type() -> OsType:
    system = platform.system().lower()
    if system == "linux":
        return OsType.LINUX
    if system == "darwin":
        return OsType.MACOS
    if system == "windows":
        return OsType.WINDOWS
    return OsType.UNKNOWN


def _platform_snapshot() -> OsPlatformSnapshot:
    os_type = _platform_type()
    capabilities = CapabilityMatrix(
        tcp_sockets=True,
        unix_domain_sockets=os_type in {OsType.LINUX, OsType.MACOS},
        process_spawn=os_type in {OsType.LINUX, OsType.MACOS, OsType.WINDOWS},
        file_system=os_type in {OsType.LINUX, OsType.MACOS, OsType.WINDOWS},
        screen_capture=os_type in {OsType.LINUX, OsType.MACOS, OsType.WINDOWS},
        clipboard=os_type in {OsType.LINUX, OsType.MACOS, OsType.WINDOWS},
        named_pipes=os_type in {OsType.LINUX, OsType.MACOS, OsType.WINDOWS},
    )
    return OsPlatformSnapshot(
        os_type=os_type,
        hostname=platform.node(),
        environment={key: value for key, value in os.environ.items() if key.startswith(("OPEN", "PYTHON", "PATH"))},
        current_user=os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        capabilities=capabilities,
    )


def _descriptor_to_tool_definition(descriptor: ToolDescriptor) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": descriptor.tool_id,
            "description": descriptor.compressed_description,
            "parameters": descriptor.input_schema,
        },
    }


def _estimate_latency_ms(descriptor: ToolDescriptor) -> int:
    base = 120 if descriptor.mutation_class.value == "read_only" else 240
    if descriptor.tool_id.startswith("browser_"):
        return 600
    if descriptor.tool_id.startswith("terminal_"):
        return 450
    if descriptor.tool_id.startswith("a2a_"):
        return 350
    return base


def _response_cost(descriptor: ToolDescriptor, response: ToolInvocationResponse) -> CostRecord:
    payload_bytes = 0
    if response.success is not None:
        payload_bytes = len(json.dumps(response.success.data, sort_keys=True))
    elif response.error is not None:
        payload_bytes = len(response.error.message)
    return CostRecord(
        api_calls=1,
        tokens_input=descriptor.token_cost_estimate,
        tokens_output=max(payload_bytes // 8, 1),
        compute_seconds=_estimate_latency_ms(descriptor) / 1_000.0,
        network_bytes=payload_bytes,
        monetary_cost_usd=round(descriptor.token_cost_estimate / 100_000.0, 6),
    )


@dataclass(slots=True)
class AgentSdk:
    session: SessionRecord
    history: tuple[HistoryRecord, ...] = ()
    memories: tuple[MemoryRecord, ...] = ()
    working_items: tuple[WorkingMemoryItem, ...] = ()
    tool_engine: ToolExecutionEngine = field(default_factory=ToolExecutionEngine)
    auth_manager: AuthManager = field(default_factory=AuthManager)
    cost_governor: CostGovernor = field(default_factory=CostGovernor)
    context_repository: InMemoryContextRepository = field(default_factory=InMemoryContextRepository)
    repository: InMemoryAgentSdkRepository = field(default_factory=InMemoryAgentSdkRepository)
    provider_factory: AgentSdkProviderFactory = field(default_factory=AgentSdkProviderFactory)
    connector_runtime: ToolBackedConnectorRuntime = field(init=False)
    platform: OsPlatformSnapshot = field(default_factory=_platform_snapshot)

    def __post_init__(self) -> None:
        self.connector_runtime = ToolBackedConnectorRuntime(engine=self.tool_engine)
        if ResourceScope.SESSION not in self.cost_governor.limits:
            self.cost_governor.set_limit(
                BudgetLimit(
                    scope=ResourceScope.SESSION,
                    api_calls=10_000,
                    tokens=10_000_000,
                    compute_seconds=50_000.0,
                    monetary_cost_usd=1_000.0,
                )
            )

    @classmethod
    def bootstrap_openagentbench(
        cls,
        *,
        session: SessionRecord,
        history: Sequence[HistoryRecord] = (),
        memories: Sequence[MemoryRecord] = (),
        working_items: Sequence[WorkingMemoryItem] = (),
        extra_descriptors: Sequence[ToolDescriptor] = (),
    ) -> "AgentSdk":
        tool_engine = ToolExecutionEngine()
        OpenAgentBenchToolSuite(
            sessions={session.session_id: session},
            history_by_session={session.session_id: list(history)},
            memories_by_user={session.user_id: list(memories)},
            working_by_session={(session.user_id, session.session_id): list(working_items)},
        ).register_into(tool_engine)
        for descriptor in extra_descriptors:
            tool_engine.register(descriptor)
        sdk = cls(
            session=session,
            history=tuple(history),
            memories=tuple(memories),
            working_items=tuple(working_items),
            tool_engine=tool_engine,
        )
        sdk.sync_connectors()
        return sdk

    def sync_connectors(self) -> None:
        self.connector_runtime.sync_from_tool_engine()

    def list_connectors(self) -> tuple[ConnectorDescriptor, ...]:
        return self.connector_runtime.registry.list_connectors()

    def tool_descriptors(self) -> tuple[ToolDescriptor, ...]:
        return self.tool_engine.registry.list_tools()

    def tool_definitions(self) -> tuple[dict[str, Any], ...]:
        return tuple(_descriptor_to_tool_definition(descriptor) for descriptor in self.tool_descriptors())

    def build_context(
        self,
        *,
        query_text: str,
        provider: str = "openai_responses",
        tool_budget: int = 768,
    ):
        return compile_context(
            ContextCompileRequest(
                user_id=self.session.user_id,
                session=self.session,
                query_text=query_text,
                history=self.history,
                memories=self.memories,
                working_items=self.working_items,
                active_tools=self.tool_definitions(),
                provider=provider,  # type: ignore[arg-type]
                total_budget=self.session.context_window_size,
                response_reserve=self.session.max_response_tokens,
                tool_budget=tool_budget,
                metadata={"objective": query_text, "connector_count": len(self.list_connectors())},
            ),
            repository=self.context_repository,
        )

    def select_tools_for_context(
        self,
        *,
        task_hint: str,
        token_budget: int,
    ) -> tuple[ToolDescriptor, ...]:
        return self.tool_engine.registry.select_tools_for_task(task_objective=task_hint, token_budget=token_budget)

    def project_tool_surface(
        self,
        *,
        task_hint: str,
        token_budget: int,
    ) -> ProjectedToolSurface:
        selected = self.select_tools_for_context(task_hint=task_hint, token_budget=token_budget)
        definitions = tuple(_descriptor_to_tool_definition(descriptor) for descriptor in selected)
        mcp_descriptors = tuple(
            {
                "name": descriptor.tool_id,
                "description": descriptor.compressed_description,
                "inputSchema": descriptor.input_schema,
                "annotations": {
                    "idempotent": descriptor.mutation_class.value == "read_only",
                    "requiredScopes": list(descriptor.auth_contract.required_scopes),
                    "sourceType": descriptor.source_type.value,
                },
            }
            for descriptor in selected
        )
        function_descriptors = tuple(definition["function"] for definition in definitions)
        return ProjectedToolSurface(
            tool_definitions=definitions,
            openai_tools=format_tools_for_model(definitions, "openai"),
            vllm_tools=format_tools_for_model(definitions, "vllm"),
            mcp_descriptors=mcp_descriptors,
            function_descriptors=function_descriptors,
        )

    def build_provider_suite(self, *, client: Any, config: ProviderClientConfig) -> ProviderSuite:
        return self.provider_factory.build(client=client, config=config)

    def build_model_requests(
        self,
        *,
        query_text: str,
        task_hint: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        compiled = self.build_context(query_text=query_text)
        selected_surface = self.project_tool_surface(task_hint=task_hint or query_text, token_budget=768)
        openai_responses = dict(compiled.openai_responses_request)
        openai_responses["tools"] = selected_surface.openai_tools["tools"]
        openai_chat = dict(compiled.openai_chat_request)
        openai_chat["tools"] = selected_surface.openai_tools["tools"]
        openai_chat["tool_choice"] = "auto"
        vllm_responses = dict(compiled.vllm_responses_request)
        vllm_responses["tools"] = selected_surface.vllm_tools["tools"]
        vllm_chat = dict(compiled.vllm_chat_request)
        vllm_chat["tools"] = selected_surface.vllm_tools["tools"]
        vllm_chat["tool_choice"] = "auto"
        return {
            "openai_responses": openai_responses,
            "openai_chat": openai_chat,
            "vllm_responses": vllm_responses,
            "vllm_chat": vllm_chat,
        }

    def route_action(self, request: AgentSdkInvocationRequest) -> RoutedAction:
        connector = self.connector_runtime.registry.select_connector_for_task(
            task_hint=request.task_hint or request.operation,
            operation=request.operation,
            connector_id=request.connector_id,
        )
        if connector is None:
            raise LookupError(f"no connector available for operation '{request.operation}'")
        operation = self.connector_runtime.registry.find_operation(request.operation, connector_id=connector.connector_id)
        if operation is None:
            raise LookupError(f"connector '{connector.connector_id}' does not expose '{request.operation}'")
        descriptor = self.tool_engine.registry.resolve(operation.tool_id)
        if descriptor is None:
            raise LookupError(f"tool '{operation.tool_id}' is not registered")
        return RoutedAction(
            connector_id=connector.connector_id,
            operation_id=operation.operation_id,
            tool_id=operation.tool_id,
            modality=operation.modality,
            protocol=operation.protocol,
            auth_type=connector.endpoint.auth_type,
            required_scopes=operation.required_scopes,
            estimated_latency_ms=_estimate_latency_ms(descriptor),
            estimated_cost_usd=round(descriptor.token_cost_estimate / 100_000.0, 6),
        )

    def execution_context(
        self,
        *,
        scopes: Sequence[str] = _DEFAULT_SCOPES,
        trace_id: str | None = None,
    ) -> ExecutionContext:
        return ExecutionContext(
            user_id=self.session.user_id,
            agent_id=uuid4(),
            session_id=self.session.session_id,
            scopes=tuple(scopes),
            trace_id=trace_id or f"sdk:{self.session.session_id}",
        )

    def invoke(
        self,
        request: AgentSdkInvocationRequest,
        *,
        scopes: Sequence[str] = _DEFAULT_SCOPES,
        trace_id: str | None = None,
    ) -> AgentSdkInvocationResult:
        route = self.route_action(request)
        descriptor = self.tool_engine.registry.resolve(route.tool_id)
        if descriptor is None:
            raise LookupError(f"tool '{route.tool_id}' is not registered")
        approval = self.cost_governor.check_budget(
            scope=ResourceScope.SESSION,
            estimated=CostRecord(
                api_calls=1,
                tokens_input=descriptor.token_cost_estimate,
                monetary_cost_usd=round(descriptor.token_cost_estimate / 100_000.0, 6),
            ),
        )
        if not approval.approved:
            response = ToolInvocationResponse.from_error(
                code=ErrorCode.AUTHORIZATION_DENIED,
                message=approval.reason or "budget rejected",
                retryable=False,
            )
            execution_context = self.execution_context(scopes=scopes, trace_id=trace_id)
            result = AgentSdkInvocationResult(
                request=request,
                route=route,
                response=response,
                execution_context=execution_context,
                cost_record=CostRecord(),
                provenance={"budget_rejected": True},
            )
            self.repository.append_invocation(result)
            return result

        credential = self.auth_manager.resolve_credential(
            endpoint=self.connector_runtime.registry.resolve_connector(route.connector_id).endpoint,  # type: ignore[union-attr]
            scopes=request.required_scopes or route.required_scopes,
            auth_type=route.auth_type,
        )
        effective_scopes = tuple(sorted(set(scopes) | set(credential.scopes) | set(route.required_scopes)))
        execution_context = self.execution_context(scopes=effective_scopes, trace_id=trace_id)
        tool_request = ToolInvocationRequest(
            tool_id=route.tool_id,
            params=request.params,
            context=execution_context,
            idempotency_key=request.idempotency_key,
            requested_deadline_ms=request.requested_deadline_ms,
        )
        response = self.connector_runtime.invoke(tool_request)
        operation_handle = None
        if response.pending is not None:
            operation_handle = OperationHandle(
                handle_id=uuid4(),
                connector_id=route.connector_id,
                operation_id=route.operation_id,
                state=OperationState.PENDING,
                created_at=execution_context.request_started_at,
                poll_token=response.pending.poll_token,
                metadata={"approval_ticket_id": str(response.pending.approval_ticket_id)},
            )
            self.repository.put_handle(operation_handle)
        cost_record = _response_cost(descriptor, response)
        self.cost_governor.record_cost(scope=ResourceScope.SESSION, cost=cost_record)
        result = AgentSdkInvocationResult(
            request=request,
            route=route,
            response=response,
            execution_context=execution_context,
            cost_record=cost_record,
            operation_handle=operation_handle,
            provenance={
                "connector_id": route.connector_id,
                "tool_id": route.tool_id,
                "credential_source": credential.source,
                "credential_auth_type": credential.auth_type.value,
            },
        )
        self.repository.append_invocation(result)
        return result

    def await_operation(self, handle_id: UUID) -> OperationStatus:
        handle = self.repository.get_handle(handle_id)
        if handle is None:
            raise LookupError(f"operation handle '{handle_id}' was not found")
        if handle.poll_token is None:
            return OperationStatus(state=handle.state, progress=1.0 if handle.state is OperationState.SUCCEEDED else 0.0)
        approval_ticket_id = UUID(str(handle.metadata["approval_ticket_id"]))
        ticket = self.tool_engine.state_repository.get_approval_ticket(approval_ticket_id)
        if ticket is None:
            return OperationStatus(state=OperationState.CANCELLED, message="approval ticket expired or was removed")
        if ticket.status.value == "pending":
            return OperationStatus(state=OperationState.PENDING, progress=0.0, message="awaiting approval")
        if ticket.status.value == "approved":
            return OperationStatus(state=OperationState.RUNNING, progress=0.5, message="approved and awaiting replay")
        return OperationStatus(state=OperationState.CANCELLED, message="approval denied")

    def run_task(
        self,
        task: AgentTaskSpec,
        *,
        scopes: Sequence[str] = _DEFAULT_SCOPES,
    ) -> AgentTaskResult:
        compiled = self.build_context(
            query_text=task.objective,
            provider="openai_responses" if task.provider_target is ProviderTarget.OPENAI else "vllm_responses",
            tool_budget=task.token_budget,
        )
        results: list[AgentSdkInvocationResult] = []
        failures: list[str] = []
        for step in task.steps:
            result = self.invoke(
                AgentSdkInvocationRequest(
                    operation=step.operation,
                    params=step.params,
                    connector_id=step.connector_id,
                    task_hint=step.task_hint or task.objective,
                    required_scopes=step.required_scopes,
                    idempotency_key=step.idempotency_key,
                    requested_deadline_ms=step.requested_deadline_ms,
                    metadata=step.metadata,
                ),
                scopes=scopes,
                trace_id=f"sdk-task:{task.task_id}:{step.step_id}",
            )
            results.append(result)
            if result.response.status.value in {"error", "timeout"} and not step.allow_failure:
                failures.append(step.step_id)
                break
        total_cost = CostRecord()
        for result in results:
            total_cost = CostRecord(
                api_calls=total_cost.api_calls + result.cost_record.api_calls,
                tokens_input=total_cost.tokens_input + result.cost_record.tokens_input,
                tokens_output=total_cost.tokens_output + result.cost_record.tokens_output,
                compute_seconds=total_cost.compute_seconds + result.cost_record.compute_seconds,
                storage_bytes=total_cost.storage_bytes + result.cost_record.storage_bytes,
                network_bytes=total_cost.network_bytes + result.cost_record.network_bytes,
                monetary_cost_usd=total_cost.monetary_cost_usd + result.cost_record.monetary_cost_usd,
            )
        return AgentTaskResult(
            task_id=task.task_id,
            objective=task.objective,
            results=tuple(results),
            compiled_context=compiled,
            total_cost=total_cost,
            completed=not failures,
            failures=tuple(failures),
        )

    def snapshot(self) -> AgentSdkSnapshot:
        return AgentSdkSnapshot(
            session=self.session,
            platform=self.platform,
            connectors=self.list_connectors(),
            tool_descriptors=self.tool_descriptors(),
        )

    def projection_records(self) -> tuple[ConnectorProjectionRecord, ...]:
        records: list[ConnectorProjectionRecord] = []
        for connector in self.list_connectors():
            for operation in connector.operations:
                records.append(
                    ConnectorProjectionRecord(
                        connector_id=connector.connector_id,
                        tool_id=operation.tool_id,
                        projected_name=operation.operation_id,
                        protocol=operation.protocol,
                        modality=operation.modality,
                        metadata={"domain": connector.domain.value},
                    )
                )
        return tuple(records)


__all__ = ["AgentSdk"]
