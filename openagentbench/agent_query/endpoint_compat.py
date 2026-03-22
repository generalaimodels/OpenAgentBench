"""OpenAI/vLLM compatibility helpers for the query understanding module."""

from __future__ import annotations

from typing import Any

from openagentbench.agent_data import OPENAI_ENDPOINTS, VLLM_ENDPOINTS
from openagentbench.agent_memory import build_memory_endpoint_compatibility_report
from .config import QueryEndpointCompatibilityConfig
from openagentbench.agent_retrieval.endpoint_compat import (
    build_endpoint_compatibility_report,
    build_gemini_count_tokens_request,
    build_gemini_generate_content_request,
    build_openai_chat_completions_request,
    build_openai_realtime_session_request,
    build_openai_responses_request,
    build_vllm_chat_completions_request,
    build_vllm_responses_request,
)
from openagentbench.agent_tools import build_agent_tools_endpoint_compatibility_report
from openagentbench.agent_tools.endpoint_compat import (
    build_tool_enabled_openai_chat_request,
    build_tool_enabled_openai_realtime_request,
    build_tool_enabled_openai_responses_request,
    format_tools_for_model,
)

from .models import QueryEndpointCompatibilityReport


def build_query_resolve_tool_definition(
    config: QueryEndpointCompatibilityConfig | None = None,
) -> dict[str, Any]:
    active_config = config or QueryEndpointCompatibilityConfig()
    return {
        "type": "function",
        "function": {
            "name": active_config.resolve_tool_name,
            "description": "Resolve intent, rewrite the query, decompose it, and route the resulting plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "session_summary": {"type": "string"},
                    "turn_count": {"type": "integer", "minimum": 0, "maximum": 1_000_000},
                    "tool_budget": {"type": "integer", "minimum": 0, "maximum": active_config.tool_budget_maximum},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
}


def build_query_clarify_tool_definition(
    config: QueryEndpointCompatibilityConfig | None = None,
) -> dict[str, Any]:
    active_config = config or QueryEndpointCompatibilityConfig()
    return {
        "type": "function",
        "function": {
            "name": active_config.clarify_tool_name,
            "description": "Produce a concise clarification question when the user request is ambiguous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "missing_slots": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
}


def build_query_endpoint_compatibility_report(
    config: QueryEndpointCompatibilityConfig | None = None,
) -> QueryEndpointCompatibilityReport:
    active_config = config or QueryEndpointCompatibilityConfig()
    retrieval_report = build_endpoint_compatibility_report()
    memory_report = build_memory_endpoint_compatibility_report()
    tool_report = build_agent_tools_endpoint_compatibility_report()
    tools = [
        build_query_resolve_tool_definition(active_config),
        build_query_clarify_tool_definition(active_config),
    ]
    openai_responses = build_tool_enabled_openai_responses_request(
        model=active_config.openai_responses_model,
        system_prompt=active_config.openai_system_prompt,
        user_content=[{"type": "input_text", "text": active_config.openai_user_example}],
        tools=tools,
        reasoning_effort=active_config.reasoning_effort,
    )
    openai_chat = build_tool_enabled_openai_chat_request(
        model=active_config.openai_chat_model,
        system_prompt=active_config.openai_system_prompt,
        user_text=active_config.openai_chat_example,
        tools=tools,
    )
    openai_realtime = build_tool_enabled_openai_realtime_request(
        model=active_config.openai_realtime_model,
        tools=tools,
    )
    vllm_responses = build_vllm_responses_request(
        model=active_config.vllm_responses_model,
        system_prompt=active_config.openai_system_prompt,
        user_input=active_config.vllm_user_example,
        context=(active_config.vllm_context_example,),
        reasoning_effort=active_config.reasoning_effort,
    )
    vllm_responses["tools"] = format_tools_for_model(tools, "vllm")["tools"]
    vllm_chat = build_vllm_chat_completions_request(
        model=active_config.vllm_chat_model,
        system_prompt=active_config.openai_system_prompt,
        user_input=active_config.vllm_chat_example,
        context=(active_config.vllm_chat_context_example,),
    )
    vllm_chat["tools"] = format_tools_for_model(tools, "vllm")["tools"]
    return QueryEndpointCompatibilityReport(
        retrieval_report=retrieval_report,
        memory_report=memory_report,
        tool_report=tool_report,
        agent_data_endpoints=tuple(OPENAI_ENDPOINTS),
        vllm_endpoints=tuple(VLLM_ENDPOINTS),
        query_resolve_tool=tools[0],
        query_clarify_tool=tools[1],
        openai_responses_request=openai_responses,
        openai_chat_request=openai_chat,
        openai_realtime_request=openai_realtime,
        vllm_responses_request=vllm_responses,
        vllm_chat_request=vllm_chat,
        gemini_generate_content_request=build_gemini_generate_content_request(
            system_instruction=active_config.gemini_system_instruction,
            user_text=active_config.gemini_user_text,
        ),
        gemini_count_tokens_request=build_gemini_count_tokens_request(
            system_instruction=active_config.gemini_system_instruction,
            user_text=active_config.gemini_user_text,
        ),
    )


def assert_query_endpoint_payload_compatibility(
    report: QueryEndpointCompatibilityReport | None = None,
) -> None:
    current = report or build_query_endpoint_compatibility_report()
    assert current.retrieval_report.openai_responses_request["model"]
    assert current.memory_report.openai_responses_request["model"]
    assert current.tool_report.openai_tools_format["format"] == "openai_tools"
    assert any(endpoint.path == "/v1/responses" for endpoint in current.agent_data_endpoints)
    assert any(endpoint.path == "/v1/responses" for endpoint in current.vllm_endpoints)
    assert current.query_resolve_tool["function"]["name"] == "query_resolve"
    assert current.query_clarify_tool["function"]["name"] == "query_clarify"
    assert current.openai_responses_request["tools"][0]["type"] == "function"
    assert current.openai_chat_request["tool_choice"] == "auto"
    assert current.openai_realtime_request["session"]["tool_choice"] == "auto"
    assert current.vllm_responses_request["tools"][0]["type"] == "function"
    assert current.vllm_chat_request["tools"][0]["type"] == "function"
    assert "contents" in current.gemini_generate_content_request
    assert "contents" in current.gemini_count_tokens_request


__all__ = [
    "assert_query_endpoint_payload_compatibility",
    "build_query_clarify_tool_definition",
    "build_query_endpoint_compatibility_report",
    "build_query_resolve_tool_definition",
]
