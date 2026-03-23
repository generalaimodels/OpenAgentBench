"""Compatibility helpers for the agent-sdk provider and protocol surfaces."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from openagentbench.agent_context import build_agent_context_compatibility_report
from openagentbench.agent_data import SessionRecord, SessionStatus, hash_normalized_text
from openagentbench.agent_tools import (
    ToolDescriptor,
    build_a2a_task_request,
    build_agent_tools_endpoint_compatibility_report,
    build_jsonrpc_invoke_request,
    build_jsonrpc_registry_request,
    build_mcp_call_tool_request,
    build_mcp_initialize_request,
    build_mcp_list_tools_request,
    build_tool_enabled_openai_realtime_request,
)

from .enums import ProviderTarget
from .models import AgentSdkCompatibilityReport
from .orchestrator import AgentSdk


def _default_models() -> tuple[str, str]:
    context_report = build_agent_context_compatibility_report()
    openai_model = str(context_report.openai_responses_request["model"])
    vllm_model = str(context_report.vllm_responses_request["model"])
    return openai_model, vllm_model


def _sample_session(*, model_id: str) -> SessionRecord:
    now = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
    system_prompt = "You are the OpenAgentBench agent-sdk compatibility probe."
    return SessionRecord(
        session_id=uuid4(),
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id=model_id,
        context_window_size=32_000,
        system_prompt_hash=hash_normalized_text(system_prompt),
        system_prompt_tokens=10,
        max_response_tokens=1_024,
        turn_count=0,
        summary_text="Compatibility probe session for the universal agent SDK.",
        summary_token_count=8,
        system_prompt_text=system_prompt,
    )


def _tool_name(tool_definition: ToolDescriptor | Mapping[str, Any]) -> str:
    if isinstance(tool_definition, ToolDescriptor):
        return tool_definition.tool_id
    function = tool_definition.get("function")
    if not isinstance(function, Mapping):
        raise TypeError("tool definitions must expose a function block")
    return str(function["name"])


def _tool_schema(tool_definition: ToolDescriptor | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(tool_definition, ToolDescriptor):
        return tool_definition.input_schema
    function = tool_definition.get("function")
    if not isinstance(function, Mapping):
        raise TypeError("tool definitions must expose a function block")
    schema = function.get("parameters")
    if not isinstance(schema, Mapping):
        raise TypeError("tool definitions must expose JSON-schema parameters")
    return schema


def _sample_value_from_schema(schema: Mapping[str, Any]) -> Any:
    schema_type = schema.get("type")
    if isinstance(schema_type, list) and schema_type:
        schema_type = schema_type[0]

    enum_values = schema.get("enum")
    if isinstance(enum_values, Sequence) and not isinstance(enum_values, (str, bytes)) and enum_values:
        return enum_values[0]

    if schema_type == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, Mapping):
            return {}
        sample: dict[str, Any] = {}
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for field_name in required:
                if isinstance(field_name, str):
                    field_schema = properties.get(field_name)
                    if isinstance(field_schema, Mapping):
                        sample[field_name] = _sample_value_from_schema(field_schema)
        return sample
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            return [_sample_value_from_schema(items)]
        return []
    if schema_type == "string":
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and min_length > 1:
            return "x" * min(min_length, 16)
        return "sample"
    if schema_type == "integer":
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)):
            return int(minimum)
        return 1
    if schema_type == "number":
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)):
            return float(minimum)
        return 1.0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    return "sample"


def _primary_tool_payload(tool_definitions: Sequence[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not tool_definitions:
        raise ValueError("tool_definitions must not be empty")
    primary = tool_definitions[0]
    tool_name = _tool_name(primary)
    sample_arguments = _sample_value_from_schema(_tool_schema(primary))
    if not isinstance(sample_arguments, dict):
        sample_arguments = {}
    return tool_name, sample_arguments


def _preferred_a2a_tool_name(tool_definitions: Sequence[dict[str, Any]], *, fallback: str) -> str:
    for tool_definition in tool_definitions:
        tool_name = _tool_name(tool_definition)
        if tool_name.startswith("a2a_"):
            return tool_name
    return fallback


def _default_task_hint(sdk: AgentSdk) -> str:
    connectors = sdk.list_connectors()
    if not connectors:
        return "Project compatibility for the registered OpenAgentBench SDK surfaces."

    fragments: list[str] = []
    for connector in connectors:
        fragments.append(connector.connector_id.replace(":", " "))
        fragments.append(connector.domain.value.replace("_", " "))
        for operation in connector.operations[:2]:
            fragments.append(operation.operation_id.replace("_", " "))
    return "Project compatibility across connectors, protocols, and tool surfaces: " + ", ".join(fragments)


def build_agent_sdk_endpoint_compatibility_report(
    *,
    session: SessionRecord | None = None,
    openai_model: str | None = None,
    vllm_model: str | None = None,
    task_hint: str | None = None,
    tool_token_budget: int = 768,
) -> AgentSdkCompatibilityReport:
    default_openai_model, default_vllm_model = _default_models()
    if openai_model is not None:
        effective_openai_model = openai_model
    elif session is not None:
        effective_openai_model = session.model_id
    else:
        effective_openai_model = default_openai_model
    effective_vllm_model = vllm_model or default_vllm_model

    sdk = AgentSdk.bootstrap_openagentbench(session=session or _sample_session(model_id=effective_openai_model))
    sdk.sync_connectors()
    effective_task_hint = task_hint or _default_task_hint(sdk)
    projected_surface = sdk.project_tool_surface(task_hint=effective_task_hint, token_budget=tool_token_budget)
    if not projected_surface.tool_definitions:
        raise AssertionError("agent-sdk compatibility probing requires at least one projected tool")

    primary_tool_name, primary_tool_arguments = _primary_tool_payload(projected_surface.tool_definitions)
    delegate_tool_name = _preferred_a2a_tool_name(projected_surface.tool_definitions, fallback=primary_tool_name)

    model_requests = sdk.build_model_requests(query_text=effective_task_hint, task_hint=effective_task_hint)
    openai_responses_request = deepcopy(model_requests["openai_responses"])
    openai_responses_request["model"] = effective_openai_model
    openai_chat_request = deepcopy(model_requests["openai_chat"])
    openai_chat_request["model"] = effective_openai_model
    openai_realtime_request = build_tool_enabled_openai_realtime_request(
        model=effective_openai_model,
        tools=projected_surface.tool_definitions,
    )
    vllm_responses_request = deepcopy(model_requests["vllm_responses"])
    vllm_responses_request["model"] = effective_vllm_model
    vllm_chat_request = deepcopy(model_requests["vllm_chat"])
    vllm_chat_request["model"] = effective_vllm_model

    tool_report = build_agent_tools_endpoint_compatibility_report(projected_surface.tool_definitions)
    context_report = build_agent_context_compatibility_report()

    a2a_task_request = deepcopy(build_a2a_task_request())
    a2a_task_request["metadata"]["tool_id"] = delegate_tool_name
    a2a_task_request["message"]["parts"] = [
        {
            "type": "text",
            "text": f"Delegate compatibility inspection for task: {effective_task_hint}",
        }
    ]

    connectors = sdk.list_connectors()
    return AgentSdkCompatibilityReport(
        connector_count=len(connectors),
        operation_count=sum(len(connector.operations) for connector in connectors),
        provider_targets=(ProviderTarget.OPENAI, ProviderTarget.VLLM),
        tool_report=tool_report,
        context_report=context_report,
        openai_responses_request=openai_responses_request,
        openai_chat_request=openai_chat_request,
        openai_realtime_request=openai_realtime_request,
        vllm_responses_request=vllm_responses_request,
        vllm_chat_request=vllm_chat_request,
        mcp_initialize_request=build_mcp_initialize_request(),
        mcp_list_tools_request=build_mcp_list_tools_request(),
        mcp_call_tool_request=build_mcp_call_tool_request(
            tool_name=primary_tool_name,
            arguments=primary_tool_arguments,
        ),
        jsonrpc_invoke_request=build_jsonrpc_invoke_request(
            tool_name=primary_tool_name,
            arguments=primary_tool_arguments,
        ),
        jsonrpc_registry_request=build_jsonrpc_registry_request(),
        a2a_task_request=a2a_task_request,
    )


def assert_agent_sdk_endpoint_payload_compatibility(
    report: AgentSdkCompatibilityReport | None = None,
) -> None:
    current = report or build_agent_sdk_endpoint_compatibility_report()
    selected_tool_names = {_tool_name(tool_definition) for tool_definition in current.tool_report.tool_definitions}
    assert current.connector_count > 0
    assert current.operation_count >= current.connector_count
    assert current.openai_responses_request["input"][0]["role"] == "system"
    assert current.openai_responses_request["model"]
    assert "temperature" not in current.openai_responses_request
    assert current.openai_chat_request["tool_choice"] == "auto"
    assert current.openai_chat_request["model"]
    assert current.openai_realtime_request["session"]["tool_choice"] == "auto"
    assert current.openai_realtime_request["session"]["model"]
    assert current.vllm_responses_request["input"][-1]["role"] == "user"
    assert current.vllm_responses_request["model"]
    assert current.vllm_chat_request["messages"][-1]["role"] == "user"
    assert current.vllm_chat_request["model"]
    assert current.mcp_initialize_request["method"] == "initialize"
    assert current.mcp_list_tools_request["method"] == "tools/list"
    assert current.mcp_call_tool_request["method"] == "tools/call"
    assert current.mcp_call_tool_request["params"]["name"] in selected_tool_names
    assert current.jsonrpc_invoke_request["method"] == "tools.invoke"
    assert current.jsonrpc_invoke_request["params"]["tool"] in selected_tool_names
    assert current.jsonrpc_registry_request["method"] == "tools.list"
    assert current.a2a_task_request["metadata"]["tool_id"] in selected_tool_names


__all__ = [
    "assert_agent_sdk_endpoint_payload_compatibility",
    "build_agent_sdk_endpoint_compatibility_report",
]
