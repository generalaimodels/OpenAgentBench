from __future__ import annotations

from openagentbench.agent_query import (
    QueryEndpointCompatibilityConfig,
    assert_query_endpoint_payload_compatibility,
    build_query_endpoint_compatibility_report,
)


def test_query_endpoint_payloads_cover_openai_and_vllm_shapes() -> None:
    report = build_query_endpoint_compatibility_report()
    assert_query_endpoint_payload_compatibility(report)

    assert report.retrieval_report.openai_completions_request["model"] == "gpt-3.5-turbo-instruct"
    assert report.memory_report.openai_realtime_request["session"]["model"] == "gpt-realtime-mini"
    assert report.tool_report.vllm_endpoints
    assert report.query_resolve_tool["function"]["name"] == "query_resolve"
    assert report.query_clarify_tool["function"]["name"] == "query_clarify"
    assert report.openai_responses_request["model"] == "gpt-5-mini"
    assert report.openai_chat_request["tool_choice"] == "auto"
    assert report.openai_realtime_request["session"]["tool_choice"] == "auto"
    assert report.vllm_responses_request["tools"][0]["type"] == "function"
    assert report.vllm_chat_request["tools"][0]["type"] == "function"
    assert "contents" in report.gemini_generate_content_request


def test_query_endpoint_payloads_accept_config_overrides() -> None:
    report = build_query_endpoint_compatibility_report(
        QueryEndpointCompatibilityConfig(
            resolve_tool_name="resolve_custom",
            clarify_tool_name="clarify_custom",
            tool_budget_maximum=2048,
            openai_responses_model="custom-openai-responses",
            openai_chat_model="custom-openai-chat",
            openai_realtime_model="custom-openai-realtime",
            vllm_responses_model="custom-vllm-responses",
            vllm_chat_model="custom-vllm-chat",
        )
    )

    assert report.query_resolve_tool["function"]["name"] == "resolve_custom"
    assert report.query_clarify_tool["function"]["name"] == "clarify_custom"
    assert report.query_resolve_tool["function"]["parameters"]["properties"]["tool_budget"]["maximum"] == 2048
    assert report.openai_responses_request["model"] == "custom-openai-responses"
    assert report.openai_chat_request["model"] == "custom-openai-chat"
    assert report.openai_realtime_request["session"]["model"] == "custom-openai-realtime"
    assert report.vllm_responses_request["model"] == "custom-vllm-responses"
    assert report.vllm_chat_request["model"] == "custom-vllm-chat"
