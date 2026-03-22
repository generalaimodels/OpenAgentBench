from __future__ import annotations

from openagentbench.agent_tools import (
    ToolResultTurn,
    ToolExecutionEngine,
    assert_agent_tools_endpoint_payload_compatibility,
    build_agent_tools_endpoint_compatibility_report,
    build_anthropic_tool_result_blocks,
    build_default_tool_definitions,
    build_google_tool_result_parts,
    build_openai_chat_tool_result_messages,
    build_openai_responses_tool_result_items,
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
    assert "temperature" not in report.openai_responses_request
    assert report.openai_responses_tool_result_items[0]["type"] == "function_call_output"
    assert report.parsed_openai_responses_tool_calls[0].valid is True
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


def test_parse_tool_call_from_openai_responses_restores_canonical_tool_name() -> None:
    known_tools = (
        {
            "type": "function",
            "function": {
                "name": "memory-read",
                "description": "Read memory by query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
    )
    model_output = {
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "memory_read",
                "arguments": '{"query":"postgres"}',
            }
        ]
    }

    parsed = parse_tool_call_from_model_output(model_output, "openai", known_tools=known_tools)
    assert len(parsed) == 1
    assert parsed[0].valid is True
    assert parsed[0].tool_id == "memory-read"
    assert parsed[0].params == {"query": "postgres"}


def test_parse_tool_call_from_model_output_marks_invalid_json_arguments() -> None:
    tool_definitions = build_default_tool_definitions()
    model_output = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "memory_read",
                "arguments": '{"query"',
            }
        ]
    }

    parsed = parse_tool_call_from_model_output(model_output, "openai", known_tools=tool_definitions)
    assert len(parsed) == 1
    assert parsed[0].valid is False
    assert parsed[0].tool_id == "memory_read"
    assert "invalid JSON arguments" in (parsed[0].error or "")


def test_tool_result_payload_builders_cover_openai_anthropic_and_google_shapes() -> None:
    tool_results = (
        ToolResultTurn(
            call_id="call_1",
            tool_id="memory-read",
            output={"summary": "ready"},
        ),
    )

    responses_items = build_openai_responses_tool_result_items(tool_results)
    chat_messages = build_openai_chat_tool_result_messages(tool_results)
    anthropic_blocks = build_anthropic_tool_result_blocks(tool_results)
    google_parts = build_google_tool_result_parts(tool_results)

    assert responses_items == (
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"summary": "ready"}',
        },
    )
    assert chat_messages == (
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "memory_read",
            "content": '{"summary": "ready"}',
        },
    )
    assert anthropic_blocks == (
        {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": [{"type": "text", "text": '{"summary": "ready"}'}],
            "is_error": False,
        },
    )
    assert google_parts == (
        {
            "functionResponse": {
                "name": "memory-read",
                "response": {"summary": "ready"},
            }
        },
    )
