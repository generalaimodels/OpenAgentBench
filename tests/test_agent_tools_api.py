from __future__ import annotations

from openagentbench.agent_tools import (
    ToolExecutionEngine,
    assert_agent_tools_endpoint_payload_compatibility,
    build_agent_tools_endpoint_compatibility_report,
    build_default_tool_definitions,
    format_tools_for_model,
    module_root,
    parse_tool_call_from_model_output,
    plan_path,
    schema_sql_path,
)


def test_public_api_surface_exposes_runtime_assets_and_compatibility_helpers() -> None:
    report = build_agent_tools_endpoint_compatibility_report()
    assert module_root().name == "agent_tools"
    assert plan_path().name == "plan.md"
    assert schema_sql_path().name == "003_agent_tools_schema.sql"
    assert report.memory_report.memory_read_tool["function"]["name"] == "memory_read"
    assert any(endpoint.path == "/v1/videos" for endpoint in report.agent_data_endpoints)
    assert any(endpoint.path == "/score" for endpoint in report.vllm_endpoints)
    assert report.retrieval_report.vllm_responses_request["input"][0]["role"] == "system"
    assert ToolExecutionEngine is not None
    assert_agent_tools_endpoint_payload_compatibility(report)


def test_model_specific_tool_formats_cover_openai_vllm_anthropic_and_google() -> None:
    tool_definitions = build_default_tool_definitions()
    openai_format = format_tools_for_model(tool_definitions, "openai")
    vllm_format = format_tools_for_model(tool_definitions, "vllm", use_guided_decoding=True)
    anthropic_format = format_tools_for_model(tool_definitions, "anthropic")
    google_format = format_tools_for_model(tool_definitions, "google")

    assert openai_format["format"] == "openai_tools"
    assert len(openai_format["tools"]) == len(tool_definitions)
    assert vllm_format["format"] == "vllm_tools"
    assert "guided_decoding_grammar" in vllm_format
    assert anthropic_format["format"] == "anthropic_tools"
    assert google_format["format"] == "google_tools"


def test_parse_tool_call_from_model_output_validates_against_known_schema() -> None:
    tool_definitions = build_default_tool_definitions()
    model_output = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "memory_read",
                                "arguments": '{"query":"postgres","layer":"semantic","top_k":3}',
                            },
                        },
                        {
                            "id": "call_2",
                            "function": {
                                "name": "memory_write",
                                "arguments": '{"target_layer":"semantic"}',
                            },
                        },
                    ]
                }
            }
        ]
    }

    parsed = parse_tool_call_from_model_output(model_output, "openai", known_tools=tool_definitions)
    assert len(parsed) == 2
    assert parsed[0].valid is True
    assert parsed[0].tool_id == "memory_read"
    assert parsed[1].valid is False
    assert "required property missing" in (parsed[1].error or "")
