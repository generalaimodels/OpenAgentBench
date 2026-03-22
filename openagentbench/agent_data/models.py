"""Typed records for sessions, memories, history, and compiled context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence
from uuid import UUID

from .enums import (
    FinishReason,
    MemoryScope,
    MemoryTier,
    MessageRole,
    ProvenanceType,
    SessionStatus,
    TaskType,
)
from .json_codec import dumps
from .types import ContentPart, EmbeddingVector, JSONValue, MessageContent


@dataclass(slots=True, frozen=True)
class ChatMessage:
    role: str
    content: MessageContent = None
    name: str | None = None
    tool_calls: tuple[dict[str, Any], ...] | None = None
    tool_call_id: str | None = None

    def as_openai_dict(self) -> dict[str, Any]:
        content: Any
        if isinstance(self.content, tuple):
            content = list(self.content)
        else:
            content = self.content
        payload: dict[str, Any] = {"role": self.role, "content": content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_calls is not None:
            payload["tool_calls"] = list(self.tool_calls)
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(slots=True, frozen=True)
class ContextBudget:
    total_budget: int
    memory_budget: int
    history_budget: int
    response_reserve: int
    tool_budget: int


@dataclass(slots=True)
class SessionRecord:
    session_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    status: SessionStatus
    model_id: str
    context_window_size: int
    system_prompt_hash: bytes
    system_prompt_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    max_response_tokens: int = 4096
    turn_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_microcents: int = 0
    summary_text: str | None = None
    summary_embedding: EmbeddingVector | None = None
    summary_token_count: int = 0
    parent_session_id: UUID | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    system_prompt_text: str | None = None


@dataclass(slots=True)
class MemoryRecord:
    memory_id: UUID
    user_id: UUID
    session_id: UUID | None
    memory_tier: MemoryTier
    memory_scope: MemoryScope
    content_text: str
    content_embedding: EmbeddingVector | None
    content_hash: bytes
    provenance_type: ProvenanceType
    provenance_turn_id: UUID | None
    confidence: float
    relevance_accumulator: float
    access_count: int
    last_accessed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    is_active: bool
    is_validated: bool
    token_count: int
    superseded_by: UUID | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True)
class HistoryRecord:
    message_id: UUID
    session_id: UUID
    user_id: UUID
    turn_index: int
    role: MessageRole
    content: str | None
    content_parts: tuple[ContentPart, ...] | None
    name: str | None
    tool_calls: tuple[dict[str, Any], ...] | None
    tool_call_id: str | None
    content_embedding: EmbeddingVector | None
    content_hash: bytes | None
    token_count: int
    model_id: str | None
    finish_reason: FinishReason | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None
    api_call_id: UUID | None
    created_at: datetime
    is_compressed: bool = False
    compressed_summary_id: UUID | None = None
    is_pruned: bool = False
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_chat_message(self) -> ChatMessage:
        return ChatMessage(
            role=self.role.as_openai_role(),
            content=self.content_parts or self.content,
            name=self.name,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
        )


@dataclass(slots=True, frozen=True)
class CompileRequest:
    user_id: UUID
    session: SessionRecord
    query_text: str
    tool_token_budget: int = 0
    task_type: TaskType | None = None
    query_embedding: EmbeddingVector | None = None
    memory_budget_override: int | None = None
    history_budget_override: int | None = None
    system_prompt_text: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ScoredMemory:
    memory: MemoryRecord
    score: float
    efficiency: float


@dataclass(slots=True)
class CompiledContext:
    messages: list[dict[str, Any]]
    selected_memories: list[ScoredMemory]
    selected_history: list[HistoryRecord]
    budget: ContextBudget
    task_type: TaskType
    tokens_used: int


@dataclass(slots=True, frozen=True)
class QueryTemplate:
    sql: str
    params: Mapping[str, Any]


def message_payload_size(messages: Sequence[ChatMessage]) -> int:
    total_size = 0
    for message in messages:
        if isinstance(message.content, tuple):
            total_size += len(dumps(message.as_openai_dict()))
        else:
            total_size += len(message.content or "")
    return total_size


@dataclass(slots=True)
class APIInvocationRecord:
    api_call_id: UUID
    user_id: UUID
    session_id: UUID
    provider: str
    endpoint: str
    model_id: str
    request_payload: dict[str, JSONValue]
    response_payload: dict[str, JSONValue] | None
    usage_payload: dict[str, JSONValue]
    error_payload: dict[str, JSONValue] | None
    request_id: str | None
    status_code: int | None
    started_at: datetime
    completed_at: datetime | None
    latency_ms: int | None
    succeeded: bool
    stream_mode: bool
    input_token_count: int = 0
    output_token_count: int = 0
    cached_input_token_count: int = 0
    reasoning_token_count: int = 0
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True)
class APIStreamEventRecord:
    user_id: UUID
    api_call_id: UUID
    event_index: int
    event_type: str
    text_delta: str | None
    token_count: int
    payload: dict[str, JSONValue]
    binary_payload: bytes | None
    mime_type: str | None
    created_at: datetime


@dataclass(slots=True)
class ProtocolEventRecord:
    protocol_event_id: UUID
    user_id: UUID
    session_id: UUID
    api_call_id: UUID | None
    message_id: UUID | None
    protocol_type: str
    direction: str
    method: str | None
    rpc_id: str | None
    tool_name: str | None
    tool_call_id: str | None
    payload: dict[str, JSONValue]
    binary_payload: bytes | None
    mime_type: str | None
    created_at: datetime
