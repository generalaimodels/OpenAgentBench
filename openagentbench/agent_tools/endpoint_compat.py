"""Model-format and protocol compatibility helpers for the agent-tools module."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

from openagentbench.agent_data import OPENAI_ENDPOINTS, VLLM_ENDPOINTS
from openagentbench.agent_memory import (
    MemoryEndpointCompatibilityReport,
    build_memory_endpoint_compatibility_report,
)
from openagentbench.agent_retrieval import (
    EndpointCompatibilityReport,
    build_endpoint_compatibility_report,
    build_gemini_count_tokens_request,
    build_gemini_generate_content_request,
    build_openai_audio_speech_request,
    build_openai_audio_transcription_request,
    build_openai_audio_translation_request,
    build_openai_image_edit_request,
    build_openai_image_generation_request,
    build_openai_realtime_session_request,
    build_openai_video_generation_request,
)

from .catalog import build_default_tool_definitions
from .models import ParsedToolCall, ToolDescriptor, ToolEndpointCompatibilityReport, ToolResultTurn
from .registry import validate_against_schema


def sanitize_tool_name(tool_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", tool_name)
    return sanitized[:64] or "tool"


def _normalize_tool_definition(tool: ToolDescriptor | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool, ToolDescriptor):
        return {
            "name": tool.tool_id,
            "description": tool.compressed_description,
            "input_schema": tool.input_schema,
        }
    function = tool.get("function")
    if not isinstance(function, dict):
        raise TypeError("tool definitions must contain a function block")
    return {
        "name": str(function["name"]),
        "description": str(function["description"]),
        "input_schema": dict(function["parameters"]),
    }


def _tool_schema_map(
    selected_tools: Iterable[ToolDescriptor | dict[str, Any]],
    *,
    provider: str | None = None,
) -> dict[str, tuple[str, dict[str, Any]]]:
    normalized = [_normalize_tool_definition(tool) for tool in selected_tools]
    schema_map: dict[str, tuple[str, dict[str, Any]]] = {}
    for entry in normalized:
        canonical_name = entry["name"]
        schema = entry["input_schema"]
        schema_map[canonical_name] = (canonical_name, schema)
        if provider in {"openai", "vllm"}:
            schema_map[sanitize_tool_name(canonical_name)] = (canonical_name, schema)
    return schema_map


def _normalize_tool_result_turn(turn: ToolResultTurn | Mapping[str, Any]) -> ToolResultTurn:
    if isinstance(turn, ToolResultTurn):
        return turn
    return ToolResultTurn(
        call_id=str(turn["call_id"]),
        tool_id=str(turn["tool_id"]),
        output=turn.get("output"),
    )


def _serialize_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, sort_keys=True)


def _google_tool_result_payload(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return dict(output)
    if isinstance(output, list):
        return {"items": list(output)}
    return {"value": output}


def _sample_value_from_schema(schema: Mapping[str, Any]) -> Any:
    schema_type = schema.get("type")
    if isinstance(schema_type, list) and schema_type:
        schema_type = schema_type[0]

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", ())
        sample: dict[str, Any] = {}
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)) and isinstance(properties, Mapping):
            for field_name in required:
                if isinstance(field_name, str) and field_name in properties and isinstance(properties[field_name], Mapping):
                    sample[field_name] = _sample_value_from_schema(properties[field_name])
        return sample
    if schema_type == "array":
        items = schema.get("items", {})
        if isinstance(items, Mapping):
            return [_sample_value_from_schema(items)]
        return []
    if schema_type == "string":
        minimum = schema.get("minLength")
        length = minimum if isinstance(minimum, int) and minimum > 0 else 8
        return "x" * min(length, 16)
    if schema_type == "integer":
        minimum = schema.get("minimum")
        return int(minimum) if isinstance(minimum, (int, float)) else 1
    if schema_type == "number":
        minimum = schema.get("minimum")
        return float(minimum) if isinstance(minimum, (int, float)) else 1.0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    return "sample"


def _guided_decoding_grammar(selected_tools: Sequence[ToolDescriptor | dict[str, Any]]) -> str:
    names = [sanitize_tool_name(_normalize_tool_definition(tool)["name"]) for tool in selected_tools]
    joined = " | ".join(f'"{name}"' for name in names)
    return f"tool_name = {joined}"


def format_tools_for_model(
    selected_tools: Sequence[ToolDescriptor | dict[str, Any]],
    provider: str,
    *,
    use_guided_decoding: bool = False,
) -> dict[str, Any]:
    normalized = [_normalize_tool_definition(tool) for tool in selected_tools]

    if provider == "openai":
        return {
            "format": "openai_tools",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": sanitize_tool_name(tool["name"]),
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                        "strict": True,
                    },
                }
                for tool in normalized
            ],
        }

    if provider == "vllm":
        payload = format_tools_for_model(selected_tools, "openai")
        payload["format"] = "vllm_tools"
        if use_guided_decoding:
            payload["guided_decoding_grammar"] = _guided_decoding_grammar(selected_tools)
        return payload

    if provider == "anthropic":
        return {
            "format": "anthropic_tools",
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }
                for tool in normalized
            ],
        }

    if provider == "google":
        return {
            "format": "google_tools",
            "function_declarations": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                }
                for tool in normalized
            ],
        }

    raise ValueError(f"unsupported provider '{provider}'")


def build_tool_enabled_openai_responses_request(
    *,
    model: str,
    system_prompt: str,
    user_content: Sequence[dict[str, Any]],
    tools: Sequence[ToolDescriptor | dict[str, Any]],
    reasoning_effort: str = "medium",
    max_output_tokens: int = 512,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": list(user_content)},
        ],
        "tools": format_tools_for_model(tools, "openai")["tools"],
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def build_tool_enabled_openai_chat_request(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    tools: Sequence[ToolDescriptor | dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": "https://example.com/diagram.png"}},
                ],
            },
        ],
        "tools": format_tools_for_model(tools, "openai")["tools"],
        "tool_choice": "auto",
        "max_tokens": 384,
        "temperature": 0.0,
    }


def build_openai_responses_tool_result_items(
    tool_results: Sequence[ToolResultTurn | Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "type": "function_call_output",
            "call_id": normalized.call_id,
            "output": _serialize_tool_output(normalized.output),
        }
        for normalized in (_normalize_tool_result_turn(turn) for turn in tool_results)
    )


def build_openai_chat_tool_result_messages(
    tool_results: Sequence[ToolResultTurn | Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "role": "tool",
            "tool_call_id": normalized.call_id,
            "name": sanitize_tool_name(normalized.tool_id),
            "content": _serialize_tool_output(normalized.output),
        }
        for normalized in (_normalize_tool_result_turn(turn) for turn in tool_results)
    )


def build_anthropic_tool_result_blocks(
    tool_results: Sequence[ToolResultTurn | Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "type": "tool_result",
            "tool_use_id": normalized.call_id,
            "content": [{"type": "text", "text": _serialize_tool_output(normalized.output)}],
            "is_error": False,
        }
        for normalized in (_normalize_tool_result_turn(turn) for turn in tool_results)
    )


def build_google_tool_result_parts(
    tool_results: Sequence[ToolResultTurn | Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "functionResponse": {
                "name": normalized.tool_id,
                "response": _google_tool_result_payload(normalized.output),
            }
        }
        for normalized in (_normalize_tool_result_turn(turn) for turn in tool_results)
    )


def build_tool_enabled_openai_realtime_request(
    *,
    model: str,
    tools: Sequence[ToolDescriptor | dict[str, Any]],
) -> dict[str, Any]:
    payload = build_openai_realtime_session_request(
        model=model,
        modalities=("text", "audio"),
        instructions="Coordinate tool calls across text and audio turns.",
    )
    payload["session"]["tools"] = format_tools_for_model(tools, "openai")["tools"]
    payload["session"]["tool_choice"] = "auto"
    return payload


def build_jsonrpc_invoke_request(
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "tool-invoke-1",
        "method": "tools.invoke",
        "params": {
            "tool": tool_name,
            "arguments": arguments,
        },
    }


def build_jsonrpc_registry_request() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "tool-admin-1",
        "method": "tools.list",
        "params": {"include_schemas": True},
    }


def build_mcp_initialize_request() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "mcp-init-1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "clientInfo": {"name": "agent_tools", "version": "0.1.0"},
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": True, "listChanged": True},
                "prompts": {"listChanged": True},
            },
        },
    }


def build_mcp_list_tools_request() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "mcp-tools-1",
        "method": "tools/list",
        "params": {},
    }


def build_mcp_call_tool_request(
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "mcp-call-1",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }


def build_a2a_task_request() -> dict[str, Any]:
    return {
        "id": "a2a-task-1",
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "Delegate browser inspection and memory verification."}],
        },
        "metadata": {
            "trace_id": "trace-agent-tools-1",
            "tool_id": "a2a_delegate",
            "deadline_ms": 60_000,
        },
    }


def _sample_tool_call_fixture(
    tool_definitions: Sequence[ToolDescriptor | dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], tuple[ToolResultTurn, ...]]:
    normalized = [_normalize_tool_definition(tool) for tool in tool_definitions]
    if not normalized:
        raise ValueError("tool_definitions must contain at least one tool")

    sample_tool = normalized[0]
    sample_arguments = _sample_value_from_schema(sample_tool["input_schema"])
    if not isinstance(sample_arguments, dict):
        sample_arguments = {}
    wire_name = sanitize_tool_name(sample_tool["name"])
    responses_output = {
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": wire_name,
                "arguments": json.dumps(sample_arguments, sort_keys=True),
            }
        ]
    }
    chat_output = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": wire_name,
                                "arguments": json.dumps(sample_arguments, sort_keys=True),
                            },
                        }
                    ]
                }
            }
        ]
    }
    tool_results = (
        ToolResultTurn(
            call_id="call_1",
            tool_id=sample_tool["name"],
            output={"status": "ok", "tool": sample_tool["name"]},
        ),
    )
    return responses_output, chat_output, tool_results


def build_agent_tools_endpoint_compatibility_report(
    selected_tools: Sequence[ToolDescriptor | dict[str, Any]] | None = None,
) -> ToolEndpointCompatibilityReport:
    tool_definitions = tuple(selected_tools or build_default_tool_definitions())
    retrieval_report: EndpointCompatibilityReport = build_endpoint_compatibility_report()
    memory_report: MemoryEndpointCompatibilityReport = build_memory_endpoint_compatibility_report()
    sample_responses_output, sample_chat_output, sample_tool_results = _sample_tool_call_fixture(tool_definitions)

    sample_user_content = (
        {"type": "input_text", "text": "Read memory, inspect tool coverage, and emit structured output."},
        {"type": "input_image", "image_url": "https://example.com/diagram.png"},
        {"type": "input_audio", "audio_url": "https://example.com/note.wav"},
    )

    return ToolEndpointCompatibilityReport(
        retrieval_report=retrieval_report,
        memory_report=memory_report,
        agent_data_endpoints=OPENAI_ENDPOINTS,
        vllm_endpoints=VLLM_ENDPOINTS,
        tool_definitions=tool_definitions,
        openai_tools_format=format_tools_for_model(tool_definitions, "openai"),
        vllm_tools_format=format_tools_for_model(tool_definitions, "vllm", use_guided_decoding=True),
        anthropic_tools_format=format_tools_for_model(tool_definitions, "anthropic"),
        google_tools_format=format_tools_for_model(tool_definitions, "google"),
        openai_responses_request=build_tool_enabled_openai_responses_request(
            model="gpt-5-mini",
            system_prompt="You are a tool orchestration model.",
            user_content=sample_user_content,
            tools=tool_definitions,
        ),
        openai_chat_request=build_tool_enabled_openai_chat_request(
            model="gpt-4.1-mini",
            system_prompt="You are a tool orchestration model.",
            user_text="Use the available tools and summarize the current platform state.",
            tools=tool_definitions,
        ),
        openai_realtime_request=build_tool_enabled_openai_realtime_request(
            model="gpt-realtime-mini",
            tools=tool_definitions,
        ),
        openai_audio_speech_request=build_openai_audio_speech_request(
            model="gpt-4o-mini-tts",
            input_text="Read the tool plan aloud.",
        ),
        openai_audio_transcription_request=build_openai_audio_transcription_request(
            model="gpt-4o-mini-transcribe",
            file_name="tool-note.wav",
        ),
        openai_audio_translation_request=build_openai_audio_translation_request(
            model="gpt-4o-mini-transcribe",
            file_name="tool-note.wav",
        ),
        openai_image_generation_request=build_openai_image_generation_request(
            model="gpt-image-1",
            prompt="Generate a tool orchestration diagram.",
        ),
        openai_image_edit_request=build_openai_image_edit_request(
            model="gpt-image-1",
            prompt="Highlight the authorization gate.",
            image_name="tool-graph.png",
            mask_name="mask.png",
        ),
        openai_video_generation_request=build_openai_video_generation_request(
            model="sora-2",
            prompt="Animate the tool-dispatch lifecycle.",
            image_name="tool-graph.png",
        ),
        gemini_generate_content_request=build_gemini_generate_content_request(
            system_instruction="You are a tool protocol compatibility checker.",
            user_text="Summarize the current tool surface and supported protocols.",
        ),
        gemini_count_tokens_request=build_gemini_count_tokens_request(
            system_instruction="You are a tool protocol compatibility checker.",
            user_text="Summarize the current tool surface and supported protocols.",
        ),
        jsonrpc_invoke_request=build_jsonrpc_invoke_request(
            tool_name="memory_read",
            arguments={"query": "postgres", "layer": "semantic"},
        ),
        jsonrpc_admin_request=build_jsonrpc_registry_request(),
        mcp_initialize_request=build_mcp_initialize_request(),
        mcp_list_tools_request=build_mcp_list_tools_request(),
        mcp_call_tool_request=build_mcp_call_tool_request(
            tool_name="memory_read",
            arguments={"query": "global rules"},
        ),
        a2a_task_request=build_a2a_task_request(),
        openai_responses_tool_result_items=build_openai_responses_tool_result_items(sample_tool_results),
        openai_chat_tool_messages=build_openai_chat_tool_result_messages(sample_tool_results),
        anthropic_tool_result_blocks=build_anthropic_tool_result_blocks(sample_tool_results),
        google_tool_result_parts=build_google_tool_result_parts(sample_tool_results),
        parsed_openai_responses_tool_calls=tuple(
            parse_tool_call_from_model_output(
                sample_responses_output,
                "openai",
                known_tools=tool_definitions,
            )
        ),
        parsed_openai_chat_tool_calls=tuple(
            parse_tool_call_from_model_output(
                sample_chat_output,
                "openai",
                known_tools=tool_definitions,
            )
        ),
    )


def _coerce_tool_arguments(raw_arguments: Any) -> tuple[dict[str, Any], str | None]:
    if raw_arguments is None:
        return {}, None
    if isinstance(raw_arguments, str):
        try:
            decoded = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {}, f"invalid JSON arguments: {exc.msg}"
        if not isinstance(decoded, dict):
            return {}, "tool arguments must decode to an object"
        return decoded, None
    if isinstance(raw_arguments, Mapping):
        return dict(raw_arguments), None
    return {}, "tool arguments must be an object"


def _parse_openai_tool_calls(model_output: dict[str, Any]) -> list[tuple[str, str, dict[str, Any], str | None]]:
    output = model_output.get("output")
    if isinstance(output, list):
        parsed: list[tuple[str, str, dict[str, Any], str | None]] = []
        for index, item in enumerate(output):
            if not isinstance(item, Mapping) or item.get("type") != "function_call":
                continue
            arguments, error = _coerce_tool_arguments(item.get("arguments", "{}"))
            parsed.append(
                (
                    str(item.get("call_id") or item.get("id") or f"call_{index}"),
                    str(item.get("name", "")),
                    arguments,
                    error,
                )
            )
        if parsed:
            return parsed

    choices = model_output.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    parsed = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        arguments, error = _coerce_tool_arguments(function.get("arguments", "{}"))
        parsed.append((str(tool_call.get("id", "")), str(function.get("name", "")), arguments, error))
    return parsed


def _parse_anthropic_tool_calls(model_output: dict[str, Any]) -> list[tuple[str, str, dict[str, Any], str | None]]:
    content = model_output.get("content")
    if not isinstance(content, list):
        return []
    parsed: list[tuple[str, str, dict[str, Any], str | None]] = []
    for block in content:
        if block.get("type") != "tool_use":
            continue
        arguments, error = _coerce_tool_arguments(block.get("input", {}))
        parsed.append((str(block.get("id", "")), str(block.get("name", "")), arguments, error))
    return parsed


def _parse_google_tool_calls(model_output: dict[str, Any]) -> list[tuple[str, str, dict[str, Any], str | None]]:
    candidates = model_output.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []
    content = candidates[0].get("content", {})
    parts = content.get("parts")
    if not isinstance(parts, list):
        return []
    parsed: list[tuple[str, str, dict[str, Any], str | None]] = []
    for index, part in enumerate(parts):
        function_call = part.get("functionCall") or part.get("function_call")
        if not isinstance(function_call, dict):
            continue
        arguments, error = _coerce_tool_arguments(function_call.get("args", {}))
        parsed.append(
            (
                f"google-call-{index}",
                str(function_call.get("name", "")),
                arguments,
                error,
            )
        )
    return parsed


def parse_tool_call_from_model_output(
    model_output: dict[str, Any],
    provider: str,
    *,
    known_tools: Sequence[ToolDescriptor | dict[str, Any]] = (),
) -> list[ParsedToolCall]:
    schema_map = _tool_schema_map(known_tools, provider=provider) if known_tools else {}
    if provider in {"openai", "vllm"}:
        raw_calls = _parse_openai_tool_calls(model_output)
    elif provider == "anthropic":
        raw_calls = _parse_anthropic_tool_calls(model_output)
    elif provider == "google":
        raw_calls = _parse_google_tool_calls(model_output)
    else:
        raise ValueError(f"unsupported provider '{provider}'")

    parsed: list[ParsedToolCall] = []
    for call_id, wire_tool_id, params, parse_error in raw_calls:
        schema_binding = schema_map.get(wire_tool_id)
        if schema_binding is None:
            parsed.append(
                ParsedToolCall(
                    call_id=call_id,
                    tool_id=wire_tool_id,
                    params=params,
                    model_format=provider,
                    valid=not schema_map,
                    error=None if not schema_map else f"tool '{wire_tool_id}' is unknown",
                )
            )
            continue
        canonical_tool_id, schema = schema_binding
        if parse_error is not None:
            parsed.append(
                ParsedToolCall(
                    call_id=call_id,
                    tool_id=canonical_tool_id,
                    params=params,
                    model_format=provider,
                    valid=False,
                    error=parse_error,
                )
            )
            continue
        validation = validate_against_schema(params, schema)
        parsed.append(
            ParsedToolCall(
                call_id=call_id,
                tool_id=canonical_tool_id,
                params=params,
                model_format=provider,
                valid=validation.valid,
                error=None if validation.valid else "; ".join(validation.errors),
            )
        )
    return parsed


def assert_agent_tools_endpoint_payload_compatibility(
    report: ToolEndpointCompatibilityReport | None = None,
) -> None:
    current = report or build_agent_tools_endpoint_compatibility_report()
    assert current.retrieval_report.openai_realtime_request["type"] == "session.update"
    assert current.memory_report.memory_read_tool["function"]["name"] == "memory_read"
    assert current.agent_data_endpoints
    assert current.vllm_endpoints
    assert current.openai_tools_format["format"] == "openai_tools"
    assert current.vllm_tools_format["format"] == "vllm_tools"
    assert current.anthropic_tools_format["format"] == "anthropic_tools"
    assert current.google_tools_format["format"] == "google_tools"
    assert current.openai_responses_request["input"][0]["role"] == "system"
    assert "temperature" not in current.openai_responses_request
    assert current.openai_chat_request["messages"][0]["role"] == "system"
    assert current.openai_realtime_request["session"]["tools"]
    assert "input" in current.openai_audio_speech_request
    assert "file" in current.openai_audio_transcription_request
    assert "file" in current.openai_audio_translation_request
    assert "prompt" in current.openai_image_generation_request
    assert "image" in current.openai_image_edit_request
    assert "prompt" in current.openai_video_generation_request
    assert current.jsonrpc_invoke_request["jsonrpc"] == "2.0"
    assert current.mcp_initialize_request["method"] == "initialize"
    assert current.a2a_task_request["message"]["role"] == "user"
    assert current.openai_responses_tool_result_items[0]["type"] == "function_call_output"
    assert current.openai_chat_tool_messages[0]["role"] == "tool"
    assert current.anthropic_tool_result_blocks[0]["type"] == "tool_result"
    assert "functionResponse" in current.google_tool_result_parts[0]
    assert current.parsed_openai_responses_tool_calls[0].valid is True
    assert current.parsed_openai_chat_tool_calls[0].valid is True


__all__ = [
    "assert_agent_tools_endpoint_payload_compatibility",
    "build_a2a_task_request",
    "build_agent_tools_endpoint_compatibility_report",
    "build_anthropic_tool_result_blocks",
    "build_google_tool_result_parts",
    "build_jsonrpc_invoke_request",
    "build_jsonrpc_registry_request",
    "build_mcp_call_tool_request",
    "build_mcp_initialize_request",
    "build_mcp_list_tools_request",
    "build_openai_chat_tool_result_messages",
    "build_openai_responses_tool_result_items",
    "build_tool_enabled_openai_chat_request",
    "build_tool_enabled_openai_realtime_request",
    "build_tool_enabled_openai_responses_request",
    "format_tools_for_model",
    "parse_tool_call_from_model_output",
    "sanitize_tool_name",
]
