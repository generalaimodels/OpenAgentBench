"""Typed capture contracts for realtime normalization and persistence tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openagentbench.agent_data.enums import FinishReason, MessageRole
from openagentbench.agent_data.models import (
    APIInvocationRecord,
    APIStreamEventRecord,
    HistoryRecord,
    MemoryRecord,
    ProtocolEventRecord,
    SessionRecord,
)
from openagentbench.agent_data.types import ContentPart, JSONValue


@dataclass(slots=True, frozen=True)
class WireEvent:
    direction: str
    event_type: str
    payload: dict[str, JSONValue]
    created_at: datetime
    text_delta: str | None = None
    token_count: int = 0
    binary_payload: bytes | None = None
    mime_type: str | None = None


@dataclass(slots=True, frozen=True)
class WireProtocolEvent:
    protocol_type: str
    direction: str
    method: str | None
    created_at: datetime
    payload: dict[str, JSONValue]
    rpc_id: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    message_ordinal: int | None = None
    binary_payload: bytes | None = None
    mime_type: str | None = None


@dataclass(slots=True, frozen=True)
class CapturedMessage:
    role: MessageRole
    created_at: datetime
    content: str | None = None
    content_parts: tuple[ContentPart, ...] | None = None
    name: str | None = None
    tool_calls: tuple[dict[str, Any], ...] | None = None
    tool_call_id: str | None = None
    model_id: str | None = None
    finish_reason: FinishReason | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class VendorCapture:
    provider: str
    endpoint: str
    model_id: str
    started_at: datetime
    completed_at: datetime
    request_payload: dict[str, JSONValue]
    response_payload: dict[str, JSONValue] | None
    usage_payload: dict[str, JSONValue]
    error_payload: dict[str, JSONValue] | None
    request_id: str | None
    status_code: int | None
    succeeded: bool
    stream_mode: bool
    messages: tuple[CapturedMessage, ...]
    stream_events: tuple[WireEvent, ...]
    protocol_events: tuple[WireProtocolEvent, ...]
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class NormalizedCaseRecords:
    session: SessionRecord
    api_invocation: APIInvocationRecord
    history: tuple[HistoryRecord, ...]
    memories: tuple[MemoryRecord, ...]
    stream_events: tuple[APIStreamEventRecord, ...]
    protocol_events: tuple[ProtocolEventRecord, ...]
    system_prompt_text: str
