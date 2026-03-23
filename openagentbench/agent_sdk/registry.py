"""Connector projection and routing registry for the universal agent SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from openagentbench.agent_retrieval.scoring import lexical_overlap_score
from openagentbench.agent_tools import (
    ToolDescriptor,
    ToolExecutionEngine,
    ToolInvocationRequest,
    ToolInvocationResponse,
    ToolSourceType,
)

from .enums import AuthType, ConnectorDomain, ConnectorHealth, InteractionModality, ProtocolName
from .models import ConnectorDescriptor, ConnectorOperation, EndpointDescriptor


def _domain_for_tool(tool: ToolDescriptor) -> ConnectorDomain:
    if tool.tool_id.startswith("memory_"):
        return ConnectorDomain.MEMORY
    if tool.tool_id.startswith("data_"):
        return ConnectorDomain.CONTEXT
    if tool.tool_id.startswith("retrieval_"):
        return ConnectorDomain.RETRIEVAL
    if tool.tool_id.startswith("browser_"):
        return ConnectorDomain.BROWSER
    if tool.tool_id.startswith("terminal_"):
        return ConnectorDomain.TERMINAL
    if tool.tool_id.startswith("vision_"):
        return ConnectorDomain.VISION
    if tool.tool_id.startswith("a2a_"):
        return ConnectorDomain.A2A
    if tool.source_type is ToolSourceType.MCP:
        return ConnectorDomain.MCP
    if tool.source_type is ToolSourceType.JSONRPC:
        return ConnectorDomain.JSONRPC
    if tool.source_type is ToolSourceType.FUNCTION:
        return ConnectorDomain.FUNCTION
    return ConnectorDomain.TOOLS


def _modality_for_tool(tool: ToolDescriptor) -> InteractionModality:
    if tool.tool_id.startswith("browser_"):
        return InteractionModality.BROWSER
    if tool.tool_id.startswith("terminal_"):
        return InteractionModality.CLI
    if tool.source_type is ToolSourceType.DESKTOP:
        return InteractionModality.GUI
    return InteractionModality.API


def _protocol_for_tool(tool: ToolDescriptor) -> ProtocolName:
    if tool.tool_id.startswith("browser_"):
        return ProtocolName.WEBSOCKET
    if tool.tool_id.startswith("terminal_"):
        return ProtocolName.PIPE
    if tool.source_type is ToolSourceType.GRPC:
        return ProtocolName.GRPC
    if tool.source_type is ToolSourceType.JSONRPC:
        return ProtocolName.JSONRPC
    if tool.source_type is ToolSourceType.MCP:
        return ProtocolName.MCP
    if tool.source_type is ToolSourceType.A2A:
        return ProtocolName.JSONRPC
    return ProtocolName.HTTP


def _auth_type_for_tool(tool: ToolDescriptor) -> AuthType:
    if tool.tool_id.startswith("browser_") or tool.tool_id.startswith("vision_"):
        return AuthType.BROWSER_SESSION
    if tool.tool_id.startswith("terminal_"):
        return AuthType.SSH_KEY
    return AuthType.BEARER_TOKEN


def _connector_key(tool: ToolDescriptor) -> str:
    domain = _domain_for_tool(tool).value
    source = tool.source_type.value
    if tool.tool_id.startswith("browser_"):
        return "browser"
    if tool.tool_id.startswith("terminal_"):
        return "terminal"
    if tool.tool_id.startswith("vision_"):
        return "vision"
    if tool.tool_id.startswith("a2a_"):
        return "a2a"
    return f"{domain}:{source}"


def build_connector_descriptor(tool_group: Sequence[ToolDescriptor]) -> ConnectorDescriptor:
    if not tool_group:
        raise ValueError("tool_group must not be empty")
    primary = tool_group[0]
    operations = tuple(
        ConnectorOperation(
            operation_id=tool.tool_id,
            description=tool.compressed_description,
            tool_id=tool.tool_id,
            connector_id=_connector_key(primary),
            domain=_domain_for_tool(tool),
            modality=_modality_for_tool(tool),
            protocol=_protocol_for_tool(tool),
            source_type=tool.source_type,
            mutation_class=tool.mutation_class,
            required_scopes=tool.auth_contract.required_scopes,
            input_schema=tool.input_schema,
            output_schema=tool.output_schema,
            token_cost_estimate=tool.token_cost_estimate,
            destructive=tool.mutation_class.value != "read_only",
            metadata={"version": tool.version},
        )
        for tool in tool_group
    )
    mean_health = sum(tool.health_score for tool in tool_group) / max(len(tool_group), 1)
    health = ConnectorHealth.HEALTHY
    if mean_health < 0.75:
        health = ConnectorHealth.DEGRADED
    if mean_health < 0.30:
        health = ConnectorHealth.UNAVAILABLE
    return ConnectorDescriptor(
        connector_id=_connector_key(primary),
        domain=_domain_for_tool(primary),
        modality=_modality_for_tool(primary),
        protocol=_protocol_for_tool(primary),
        endpoint=EndpointDescriptor(
            endpoint_id=_connector_key(primary),
            address=primary.source_endpoint,
            protocol_hints=(_protocol_for_tool(primary),),
            auth_type=_auth_type_for_tool(primary),
            metadata={"source_type": primary.source_type.value},
        ),
        operations=operations,
        health=health,
        metadata={"tool_ids": [tool.tool_id for tool in tool_group]},
    )


@dataclass(slots=True)
class SdkConnectorRegistry:
    connectors: dict[str, ConnectorDescriptor] = field(default_factory=dict)

    def register_connector(self, connector: ConnectorDescriptor) -> None:
        self.connectors[connector.connector_id] = connector

    def register_tool_descriptors(self, tools: Sequence[ToolDescriptor]) -> None:
        grouped: dict[str, list[ToolDescriptor]] = {}
        for tool in tools:
            grouped.setdefault(_connector_key(tool), []).append(tool)
        for group in grouped.values():
            self.register_connector(build_connector_descriptor(group))

    def list_connectors(self) -> tuple[ConnectorDescriptor, ...]:
        ordered = sorted(self.connectors.values(), key=lambda connector: connector.connector_id)
        return tuple(ordered)

    def resolve_connector(self, connector_id: str) -> ConnectorDescriptor | None:
        return self.connectors.get(connector_id)

    def find_operation(self, operation: str, *, connector_id: str | None = None) -> ConnectorOperation | None:
        connectors = [self.connectors[connector_id]] if connector_id and connector_id in self.connectors else self.connectors.values()
        for connector in connectors:
            for operation_descriptor in connector.operations:
                if operation_descriptor.operation_id == operation:
                    return operation_descriptor
        return None

    def select_connector_for_task(
        self,
        *,
        task_hint: str,
        operation: str | None = None,
        connector_id: str | None = None,
    ) -> ConnectorDescriptor | None:
        if connector_id is not None:
            return self.resolve_connector(connector_id)
        ranked: list[tuple[float, ConnectorDescriptor]] = []
        for connector in self.connectors.values():
            relevance = lexical_overlap_score(
                task_hint,
                " ".join(
                    [
                        connector.connector_id,
                        connector.domain.value,
                        *(operation_descriptor.description for operation_descriptor in connector.operations),
                    ]
                ),
            )
            if operation is not None and any(descriptor.operation_id == operation for descriptor in connector.operations):
                relevance += 0.35
            relevance += 0.15 if connector.health is ConnectorHealth.HEALTHY else 0.0
            ranked.append((relevance, connector))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]


@dataclass(slots=True)
class ToolBackedConnectorRuntime:
    engine: ToolExecutionEngine
    registry: SdkConnectorRegistry = field(default_factory=SdkConnectorRegistry)

    def sync_from_tool_engine(self) -> None:
        self.registry.register_tool_descriptors(self.engine.registry.list_tools(include_inactive=True))

    def invoke(self, request: ToolInvocationRequest) -> ToolInvocationResponse:
        return self.engine.dispatch(request)


__all__ = [
    "SdkConnectorRegistry",
    "ToolBackedConnectorRuntime",
    "build_connector_descriptor",
]
