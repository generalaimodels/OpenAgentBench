"""Compatibility helpers for agent-context provider serialization."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from openagentbench.agent_data.enums import SessionStatus
from openagentbench.agent_data.models import SessionRecord
from openagentbench.agent_data.openai_catalog import OPENAI_HTTP_ENDPOINTS
from openagentbench.agent_data.vllm_catalog import VLLM_ENDPOINTS

from .compiler import build_provider_profile, compile_context
from .models import ContextCompatibilityReport, ContextCompileRequest


def _sample_session() -> SessionRecord:
    now = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
    return SessionRecord(
        session_id=uuid4(),
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id="gpt-5.4-mini",
        context_window_size=32_000,
        system_prompt_hash=b"context",
        system_prompt_tokens=16,
        max_response_tokens=1_024,
        turn_count=2,
        summary_text="Compatibility probe session.",
        summary_token_count=4,
        system_prompt_text="You are the OpenAgentBench context compatibility probe.",
    )


def _sample_tools() -> tuple[dict[str, Any], ...]:
    return (
        {
            "type": "function",
            "function": {
                "name": "memory_read",
                "description": "Read validated memory for the current user.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
    )


def _tool_result_items() -> tuple[dict[str, Any], ...]:
    return (
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "{\"result\":\"validated\"}",
        },
    )


def _chat_tool_result_messages() -> tuple[dict[str, Any], ...]:
    return (
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "memory_read",
            "content": "{\"result\":\"validated\"}",
        },
    )


def build_agent_context_compatibility_report() -> ContextCompatibilityReport:
    session = _sample_session()
    compiled = compile_context(
        ContextCompileRequest(
            user_id=session.user_id,
            session=session,
            query_text="Compile a grounded context packet for this session.",
            provider="openai_responses",
            active_tools=_sample_tools(),
            metadata={"objective": "compatibility"},
        )
    )
    return ContextCompatibilityReport(
        provider_profiles=tuple(
            build_provider_profile(provider)
            for provider in ("openai_responses", "openai_chat", "vllm_responses", "vllm_chat")
        ),
        openai_http_endpoints=tuple(OPENAI_HTTP_ENDPOINTS),
        vllm_endpoints=tuple(VLLM_ENDPOINTS),
        openai_responses_request=compiled.openai_responses_request,
        openai_chat_request=compiled.openai_chat_request,
        vllm_responses_request=compiled.vllm_responses_request,
        vllm_chat_request=compiled.vllm_chat_request,
        openai_tool_result_items=_tool_result_items(),
        openai_chat_tool_result_messages=_chat_tool_result_messages(),
    )


def assert_agent_context_payload_compatibility(
    report: ContextCompatibilityReport | None = None,
) -> None:
    current = report or build_agent_context_compatibility_report()
    assert any("POST /v1/responses" == endpoint for endpoint in current.openai_http_endpoints)
    assert any("POST /v1/chat/completions" == endpoint for endpoint in current.openai_http_endpoints)
    assert any(endpoint.path == "/v1/responses" for endpoint in current.vllm_endpoints)
    assert current.openai_responses_request["input"][0]["role"] == "system"
    assert current.openai_chat_request["messages"][-1]["role"] == "user"
    assert current.vllm_responses_request["input"][-1]["role"] == "user"
    assert current.vllm_chat_request["messages"][-1]["role"] == "user"
    assert current.openai_tool_result_items[0]["type"] == "function_call_output"
    assert current.openai_chat_tool_result_messages[0]["role"] == "tool"


__all__ = [
    "assert_agent_context_payload_compatibility",
    "build_agent_context_compatibility_report",
]
