from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from openagentbench.agent_data import SessionRecord, SessionStatus, hash_normalized_text
from openagentbench.agent_sdk import (
    assert_agent_sdk_endpoint_payload_compatibility,
    build_agent_sdk_endpoint_compatibility_report,
    module_root,
    plan_path,
    read_plan,
)


def _session(*, model_id: str) -> SessionRecord:
    now = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
    system_prompt = "You are the agent-sdk test harness."
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
        system_prompt_tokens=8,
        max_response_tokens=1_024,
        turn_count=0,
        summary_text="SDK compatibility test session.",
        summary_token_count=5,
        system_prompt_text=system_prompt,
    )


def test_agent_sdk_public_api_surface_exposes_runtime_assets_and_compatibility() -> None:
    report = build_agent_sdk_endpoint_compatibility_report()

    assert module_root().name == "agent_sdk"
    assert plan_path().name == "plan.md"
    assert "Universal Agent SDK Architecture" in read_plan()
    assert report.connector_count > 0
    assert_agent_sdk_endpoint_payload_compatibility(report)


def test_agent_sdk_endpoint_compatibility_report_uses_supplied_models_and_dynamic_tool_routes() -> None:
    report = build_agent_sdk_endpoint_compatibility_report(
        session=_session(model_id="openai-test-model"),
        vllm_model="vllm-test-model",
    )

    selected_tool_names = {
        tool_definition["function"]["name"]
        for tool_definition in report.tool_report.tool_definitions
    }

    assert report.openai_responses_request["model"] == "openai-test-model"
    assert report.openai_chat_request["model"] == "openai-test-model"
    assert report.openai_realtime_request["session"]["model"] == "openai-test-model"
    assert report.vllm_responses_request["model"] == "vllm-test-model"
    assert report.vllm_chat_request["model"] == "vllm-test-model"
    assert report.jsonrpc_invoke_request["params"]["tool"] in selected_tool_names
    assert report.mcp_call_tool_request["params"]["name"] in selected_tool_names
    assert report.a2a_task_request["metadata"]["tool_id"] in selected_tool_names
