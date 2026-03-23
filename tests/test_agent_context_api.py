from __future__ import annotations

from openagentbench.agent_context import (
    InMemoryContextRepository,
    assert_agent_context_payload_compatibility,
    build_agent_context_compatibility_report,
    compile_context,
    module_root,
    plan_path,
    read_plan,
)


def test_agent_context_public_api_surface_exposes_runtime_assets_and_compatibility() -> None:
    report = build_agent_context_compatibility_report()

    assert module_root().name == "agent_context"
    assert plan_path().name == "plan.md"
    assert "SOTA Context Engineering" in read_plan()
    assert report.openai_responses_request["input"][0]["role"] == "system"
    assert report.vllm_chat_request["messages"][-1]["role"] == "user"
    assert InMemoryContextRepository is not None
    assert compile_context is not None
    assert_agent_context_payload_compatibility(report)
