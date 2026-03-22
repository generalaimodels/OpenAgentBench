from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from openagentbench.agent_data import FinishReason, MessageRole

from tests.realtime.support import (
    BudgetLedger,
    CapturedMessage,
    DatabaseHarness,
    LiveTestConfig,
    VendorCapture,
    WireEvent,
    WireProtocolEvent,
    assert_persisted_case,
    estimate_case_cost,
    fixed_tool_declaration,
    fixed_tool_response,
    normalize_capture,
    rewrite_agent_data_schema,
    selection_roundtrip_case,
    stable_uuid,
    text_memory_case,
    tiny_pcm16_audio_bytes,
    tiny_png_bytes,
    tool_roundtrip_case,
)


def _synthetic_capture(vendor: str, *, with_tool: bool = False) -> VendorCapture:
    started_at = datetime(2026, 3, 22, 0, 0, 0, tzinfo=timezone.utc)
    completed_at = started_at + timedelta(milliseconds=420)
    if with_tool:
        messages = (
            CapturedMessage(
                role=MessageRole.USER,
                created_at=started_at,
                content="Call the lookup_test_clock tool with timezone UTC.",
                content_parts=(
                    {"type": "input_text", "text": "Call the lookup_test_clock tool with timezone UTC."},
                ),
            ),
            CapturedMessage(
                role=MessageRole.ASSISTANT,
                created_at=started_at + timedelta(milliseconds=120),
                content=None,
                tool_calls=(
                    {
                        "id": "tool-call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup_test_clock",
                            "arguments": '{"timezone":"UTC"}',
                        },
                    },
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                model_id=f"{vendor}-test-model",
            ),
            CapturedMessage(
                role=MessageRole.TOOL,
                created_at=started_at + timedelta(milliseconds=240),
                content='{"utc_timestamp":"2026-03-22T00:00:00Z","timezone":"UTC","source":"realtime_test_fixture"}',
                name="lookup_test_clock",
                tool_call_id="tool-call-1",
            ),
            CapturedMessage(
                role=MessageRole.ASSISTANT,
                created_at=completed_at,
                content="TOOL_OK 2026-03-22T00:00:00Z",
                model_id=f"{vendor}-test-model",
                finish_reason=FinishReason.STOP,
                completion_tokens=8,
            ),
        )
        protocol_events = (
            WireProtocolEvent(
                protocol_type="tool_call",
                direction="inbound",
                method="lookup_test_clock",
                rpc_id="rpc-1",
                tool_name="lookup_test_clock",
                tool_call_id="tool-call-1",
                message_ordinal=1,
                created_at=started_at + timedelta(milliseconds=110),
                payload={"arguments": {"timezone": "UTC"}},
            ),
            WireProtocolEvent(
                protocol_type="tool_call",
                direction="outbound",
                method="lookup_test_clock",
                rpc_id="rpc-1",
                tool_name="lookup_test_clock",
                tool_call_id="tool-call-1",
                message_ordinal=2,
                created_at=started_at + timedelta(milliseconds=230),
                payload=fixed_tool_response(),
            ),
            WireProtocolEvent(
                protocol_type="json-rpc",
                direction="outbound",
                method="tools/call",
                rpc_id="rpc-2",
                tool_name="lookup_test_clock",
                tool_call_id="tool-call-1",
                created_at=started_at + timedelta(milliseconds=231),
                payload={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": "rpc-2",
                    "params": {"name": "lookup_test_clock"},
                },
            ),
        )
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "TOOL_OK 2026-03-22T00:00:00Z"}],
                }
            ]
        }
    else:
        messages = (
            CapturedMessage(
                role=MessageRole.USER,
                created_at=started_at,
                content="Remember this durable preference exactly: PostgreSQL is my preferred durable memory store.",
                content_parts=(
                    {
                        "type": "input_text",
                        "text": "Remember this durable preference exactly: PostgreSQL is my preferred durable memory store.",
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,synthetic",
                    },
                ),
            ),
            CapturedMessage(
                role=MessageRole.ASSISTANT,
                created_at=started_at + timedelta(milliseconds=200),
                content="ACK_POSTGRESQL",
                model_id=f"{vendor}-test-model",
                finish_reason=FinishReason.STOP,
                completion_tokens=3,
            ),
            CapturedMessage(
                role=MessageRole.USER,
                created_at=started_at + timedelta(milliseconds=300),
                content="What is my preferred durable memory store?",
                content_parts=(
                    {"type": "input_text", "text": "What is my preferred durable memory store?"},
                ),
            ),
            CapturedMessage(
                role=MessageRole.ASSISTANT,
                created_at=completed_at,
                content="PostgreSQL is your preferred durable memory store.",
                model_id=f"{vendor}-test-model",
                finish_reason=FinishReason.STOP,
                completion_tokens=9,
            ),
        )
        protocol_events = ()
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "PostgreSQL is your preferred durable memory store.",
                        }
                    ],
                }
            ]
        }

    stream_events = (
        WireEvent(
            direction="inbound",
            event_type="session.created" if vendor == "openai" else "setupComplete",
            payload={"vendor": vendor},
            created_at=started_at,
        ),
        WireEvent(
            direction="inbound",
            event_type="response.text.delta" if vendor == "openai" else "serverContent",
            payload={"delta": "ACK" if not with_tool else "TOOL"},
            text_delta="ACK" if not with_tool else "TOOL",
            created_at=started_at + timedelta(milliseconds=150),
        ),
        WireEvent(
            direction="inbound",
            event_type="response.done" if vendor == "openai" else "serverContent.turnComplete",
            payload=response_payload,
            text_delta=None,
            created_at=completed_at,
        ),
    )

    return VendorCapture(
        provider=vendor,
        endpoint="/v1/realtime" if vendor == "openai" else "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
        model_id=f"{vendor}-test-model",
        started_at=started_at,
        completed_at=completed_at,
        request_payload={
            "headers": {"authorization": "Bearer secret"},
            "case": "tool" if with_tool else "selection",
        },
        response_payload=response_payload,
        usage_payload={"input_tokens": 24, "output_tokens": 12},
        error_payload=None,
        request_id=f"{vendor}-request-1",
        status_code=101,
        succeeded=True,
        stream_mode=True,
        messages=messages,
        stream_events=stream_events,
        protocol_events=protocol_events,
        metadata={"synthetic": True},
    )


def _postgres_harness() -> DatabaseHarness:
    config = LiveTestConfig.from_env()
    if not config.database_url:
        return DatabaseHarness.in_memory(run_id=f"{config.run_id}_dry")
    return DatabaseHarness(database_url=config.database_url, run_id=f"{config.run_id}_dry")


def test_fixture_generation_is_deterministic() -> None:
    first_png = tiny_png_bytes()
    second_png = tiny_png_bytes()
    assert first_png == second_png
    assert len(first_png) > 100

    first_audio = tiny_pcm16_audio_bytes()
    second_audio = tiny_pcm16_audio_bytes()
    assert first_audio == second_audio
    assert len(first_audio) > 1_000


def test_schema_rewriter_retargets_agent_data_schema() -> None:
    original = "CREATE SCHEMA IF NOT EXISTS agent_data;\nSELECT * FROM agent_data.sessions;"
    rewritten = rewrite_agent_data_schema(original, "agent_data_test_x")
    assert "agent_data_test_x.sessions" in rewritten
    assert "CREATE SCHEMA IF NOT EXISTS agent_data_test_x;" in rewritten


def test_normalize_capture_redacts_secrets_and_is_stable() -> None:
    capture = _synthetic_capture("openai")
    records = normalize_capture(capture, text_memory_case(), run_id="dry-stable")
    again = normalize_capture(capture, text_memory_case(), run_id="dry-stable")

    assert records.session.session_id == again.session.session_id
    assert records.api_invocation.request_payload["headers"]["authorization"] == "[REDACTED]"
    assert records.history[0].content_parts is not None
    assert records.history[0].content_parts[1]["type"] == "input_image"
    assert stable_uuid("dry-stable", "openai", "x") == stable_uuid("dry-stable", "openai", "x")


def test_budget_ledger_enforces_global_cap() -> None:
    ledger = BudgetLedger(run_id="budget-case-dry", max_budget_usd=0.0100)
    cheap = estimate_case_cost("openai", "gpt-realtime-mini", text_memory_case())
    ledger.commit("openai", cheap.total_cost_usd)
    with pytest.raises(RuntimeError):
        ledger.assert_can_spend("gemini", 0.0200)


@pytest.mark.postgres
@pytest.mark.parametrize("vendor", ["openai", "gemini"])
def test_postgres_roundtrip_retrieval_and_selection(vendor: str) -> None:
    harness = _postgres_harness()
    case = selection_roundtrip_case()
    capture = _synthetic_capture(vendor)
    records = normalize_capture(capture, case, run_id=f"dry-db-{vendor}")
    try:
        from tests.realtime.support.persistence import persist_normalized_case

        persist_normalized_case(harness, records)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except Exception as exc:
        if "vector" in str(exc).lower() or "extension" in str(exc).lower():
            pytest.skip(str(exc))
        raise
    try:
        assert_persisted_case(harness, records, case)
    finally:
        config = LiveTestConfig.from_env()
        if config.cleanup_rows:
            harness.delete_case_rows(records)


@pytest.mark.postgres
@pytest.mark.parametrize("vendor", ["openai", "gemini"])
def test_postgres_roundtrip_protocol_rows(vendor: str) -> None:
    harness = _postgres_harness()
    case = tool_roundtrip_case(fixed_tool_declaration(), fixed_tool_response())
    capture = _synthetic_capture(vendor, with_tool=True)
    records = normalize_capture(capture, case, run_id=f"dry-protocol-{vendor}")
    try:
        from tests.realtime.support.persistence import persist_normalized_case

        persist_normalized_case(harness, records)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except Exception as exc:
        if "vector" in str(exc).lower() or "extension" in str(exc).lower():
            pytest.skip(str(exc))
        raise
    try:
        assert len(records.protocol_events) >= 2
        assert_persisted_case(harness, records, case)
    finally:
        config = LiveTestConfig.from_env()
        if config.cleanup_rows:
            harness.delete_case_rows(records)
