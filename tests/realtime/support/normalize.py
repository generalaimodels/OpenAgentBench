"""Normalize vendor captures into the production agent-data records."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid5

from openagentbench.agent_data import (
    APIInvocationRecord,
    APIStreamEventRecord,
    FinishReason,
    HistoryRecord,
    MemoryRecord,
    MemoryScope,
    MemoryTier,
    MessageRole,
    ProvenanceType,
    ProtocolEventRecord,
    SessionRecord,
    SessionStatus,
    hash_normalized_text,
)
from openagentbench.agent_data.types import ContentPart, EmbeddingVector, JSONValue

from .cases import RealtimeCaseSpec
from .types import CapturedMessage, NormalizedCaseRecords, VendorCapture

STABLE_NAMESPACE = UUID("7fe8de09-8b20-4d86-a30b-8d0b4100d176")


def stable_uuid(*parts: object) -> UUID:
    encoded = "::".join(str(part) for part in parts)
    return uuid5(STABLE_NAMESPACE, encoded)


def _approximate_token_count(text: str | None) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))


def _content_text_from_parts(parts: tuple[ContentPart, ...] | None) -> str:
    if not parts:
        return ""
    chunks: list[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return " ".join(chunks)


def _extract_text_from_response_payload(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    value = part.get("text") or part.get("transcript")
                    if isinstance(value, str):
                        chunks.append(value)
        joined = " ".join(chunk for chunk in chunks if chunk)
        return joined or None
    candidate = payload.get("text")
    if isinstance(candidate, str):
        return candidate
    return None


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            lowered = key.lower()
            if lowered in {"authorization", "api_key", "x-goog-api-key"}:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_value(nested)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _embedding(seed: float) -> EmbeddingVector:
    values = [0.0] * 1536
    values[0] = seed
    values[1] = max(seed / 2.0, 0.0001)
    values[2] = 1.0 - min(seed, 0.99)
    return tuple(values)


def _history_record(
    *,
    message: CapturedMessage,
    index: int,
    user_id: UUID,
    session_id: UUID,
    api_call_id: UUID,
    provider: str,
    case: RealtimeCaseSpec,
    run_id: str,
) -> HistoryRecord:
    content_text = message.content or _content_text_from_parts(message.content_parts)
    token_count = (
        message.completion_tokens
        or message.prompt_tokens
        or _approximate_token_count(content_text)
    )
    return HistoryRecord(
        message_id=stable_uuid(run_id, provider, case.name, "message", index),
        session_id=session_id,
        user_id=user_id,
        turn_index=index + 1,
        role=message.role,
        content=message.content,
        content_parts=message.content_parts,
        name=message.name,
        tool_calls=message.tool_calls,
        tool_call_id=message.tool_call_id,
        content_embedding=_embedding(0.01 * (index + 1)),
        content_hash=hash_normalized_text(content_text) if content_text else None,
        token_count=token_count,
        model_id=message.model_id,
        finish_reason=message.finish_reason,
        prompt_tokens=message.prompt_tokens,
        completion_tokens=message.completion_tokens,
        latency_ms=message.latency_ms,
        api_call_id=api_call_id,
        created_at=message.created_at,
        metadata={
            "test_suite": "realtime",
            "vendor": provider,
            "case_name": case.name,
            "run_id": run_id,
            "fixture_version": "v1",
            **message.metadata,
        },
    )


def normalize_capture(capture: VendorCapture, case: RealtimeCaseSpec, *, run_id: str) -> NormalizedCaseRecords:
    provider = capture.provider
    user_id = stable_uuid(run_id, provider, case.name, "user")
    session_id = stable_uuid(run_id, provider, case.name, "session")
    api_call_id = stable_uuid(run_id, provider, case.name, "api")
    system_prompt_hash = hash_normalized_text(case.system_prompt_text)

    session = SessionRecord(
        session_id=session_id,
        user_id=user_id,
        created_at=capture.started_at,
        updated_at=capture.completed_at,
        expires_at=capture.completed_at + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id=capture.model_id,
        context_window_size=32_768,
        system_prompt_hash=system_prompt_hash,
        system_prompt_tokens=_approximate_token_count(case.system_prompt_text),
        temperature=0.1,
        top_p=1.0,
        max_response_tokens=512,
        turn_count=len(capture.messages),
        total_prompt_tokens=int(capture.usage_payload.get("input_tokens", 0) or 0),
        total_completion_tokens=int(capture.usage_payload.get("output_tokens", 0) or 0),
        total_cost_microcents=0,
        metadata={
            "test_suite": "realtime",
            "vendor": provider,
            "case_name": case.name,
            "run_id": run_id,
            "fixture_version": "v1",
            **case.extra_metadata,
            **capture.metadata,
        },
        system_prompt_text=case.system_prompt_text,
    )

    history = tuple(
        _history_record(
            message=message,
            index=index,
            user_id=user_id,
            session_id=session_id,
            api_call_id=api_call_id,
            provider=provider,
            case=case,
            run_id=run_id,
        )
        for index, message in enumerate(capture.messages)
    )

    usage_payload = dict(capture.usage_payload)
    if "input_tokens" not in usage_payload:
        usage_payload["input_tokens"] = case.estimated_input_tokens
    if "output_tokens" not in usage_payload:
        usage_payload["output_tokens"] = case.estimated_output_tokens

    response_text = _extract_text_from_response_payload(capture.response_payload)
    response_payload = _redact_value(capture.response_payload) if capture.response_payload is not None else None
    api_invocation = APIInvocationRecord(
        api_call_id=api_call_id,
        user_id=user_id,
        session_id=session_id,
        provider=provider,
        endpoint=capture.endpoint,
        model_id=capture.model_id,
        request_payload=_redact_value(capture.request_payload),
        response_payload=response_payload,
        usage_payload=_redact_value(usage_payload),
        error_payload=_redact_value(capture.error_payload) if capture.error_payload is not None else None,
        request_id=capture.request_id,
        status_code=capture.status_code,
        started_at=capture.started_at,
        completed_at=capture.completed_at,
        latency_ms=max(int((capture.completed_at - capture.started_at).total_seconds() * 1000), 0),
        succeeded=capture.succeeded,
        stream_mode=capture.stream_mode,
        input_token_count=int(usage_payload.get("input_tokens", 0) or 0),
        output_token_count=int(usage_payload.get("output_tokens", 0) or 0),
        cached_input_token_count=int(usage_payload.get("cached_input_tokens", 0) or 0),
        reasoning_token_count=int(usage_payload.get("reasoning_tokens", 0) or 0),
        metadata={
            "test_suite": "realtime",
            "vendor": provider,
            "case_name": case.name,
            "run_id": run_id,
            "fixture_version": "v1",
            "response_text_preview": response_text,
        },
    )

    stream_events = tuple(
        APIStreamEventRecord(
            user_id=user_id,
            api_call_id=api_call_id,
            event_index=index,
            event_type=event.event_type,
            text_delta=event.text_delta,
            token_count=event.token_count or _approximate_token_count(event.text_delta),
            payload=_redact_value(event.payload),
            binary_payload=event.binary_payload,
            mime_type=event.mime_type,
            created_at=event.created_at,
        )
        for index, event in enumerate(capture.stream_events)
        if event.direction == "inbound"
    )

    message_ids = {index: record.message_id for index, record in enumerate(history)}
    protocol_events = tuple(
        ProtocolEventRecord(
            protocol_event_id=stable_uuid(run_id, provider, case.name, "protocol", index),
            user_id=user_id,
            session_id=session_id,
            api_call_id=api_call_id,
            message_id=message_ids.get(event.message_ordinal) if event.message_ordinal is not None else None,
            protocol_type=event.protocol_type,
            direction=event.direction,
            method=event.method,
            rpc_id=event.rpc_id,
            tool_name=event.tool_name,
            tool_call_id=event.tool_call_id,
            payload=_redact_value(event.payload),
            binary_payload=event.binary_payload,
            mime_type=event.mime_type,
            created_at=event.created_at,
        )
        for index, event in enumerate(capture.protocol_events)
    )

    memories: list[MemoryRecord] = []
    if case.memory_fact_text:
        reference_turn_id = history[0].message_id if history else None
        memories.append(
            MemoryRecord(
                memory_id=stable_uuid(run_id, provider, case.name, "memory", 0),
                user_id=user_id,
                session_id=session_id,
                memory_tier=MemoryTier.SESSION,
                memory_scope=MemoryScope.LOCAL,
                content_text=case.memory_fact_text,
                content_embedding=_embedding(0.75),
                content_hash=hash_normalized_text(case.memory_fact_text),
                provenance_type=ProvenanceType.USER_STATED,
                provenance_turn_id=reference_turn_id,
                confidence=0.98,
                relevance_accumulator=5.0,
                access_count=3,
                last_accessed_at=capture.completed_at,
                created_at=capture.started_at,
                updated_at=capture.completed_at,
                expires_at=capture.completed_at + timedelta(days=1),
                is_active=True,
                is_validated=True,
                token_count=_approximate_token_count(case.memory_fact_text),
                tags=("realtime-test", "selection"),
                metadata={
                    "test_suite": "realtime",
                    "vendor": provider,
                    "case_name": case.name,
                    "run_id": run_id,
                    "fixture_version": "v1",
                },
            )
        )

    return NormalizedCaseRecords(
        session=session,
        api_invocation=api_invocation,
        history=history,
        memories=tuple(memories),
        stream_events=stream_events,
        protocol_events=protocol_events,
        system_prompt_text=case.system_prompt_text,
    )
