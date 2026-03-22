"""Parameterized SQL templates for the PostgreSQL-backed agent-data module."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID

from .json_codec import dumps
from .models import (
    APIInvocationRecord,
    APIStreamEventRecord,
    HistoryRecord,
    MemoryRecord,
    ProtocolEventRecord,
    QueryTemplate,
    SessionRecord,
)
from .types import EmbeddingVector


def _pgvector_literal(vector: EmbeddingVector) -> str:
    return "[" + ",".join(format(value, ".8g") for value in vector) + "]"


def build_insert_session(record: SessionRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.sessions (
        user_id,
        session_id,
        created_at,
        updated_at,
        expires_at,
        status,
        model_id,
        context_window_size,
        system_prompt_hash,
        system_prompt_tokens,
        temperature,
        top_p,
        max_response_tokens,
        turn_count,
        total_prompt_tokens,
        total_completion_tokens,
        total_cost_microcents,
        summary_text,
        summary_embedding,
        summary_token_count,
        parent_session_id,
        metadata
    ) VALUES (
        %(user_id)s,
        %(session_id)s,
        %(created_at)s,
        %(updated_at)s,
        %(expires_at)s,
        %(status)s,
        %(model_id)s,
        %(context_window_size)s,
        %(system_prompt_hash)s,
        %(system_prompt_tokens)s,
        %(temperature)s,
        %(top_p)s,
        %(max_response_tokens)s,
        %(turn_count)s,
        %(total_prompt_tokens)s,
        %(total_completion_tokens)s,
        %(total_cost_microcents)s,
        %(summary_text)s,
        %(summary_embedding)s,
        %(summary_token_count)s,
        %(parent_session_id)s,
        %(metadata)s::jsonb
    )
    ON CONFLICT (user_id, session_id) DO UPDATE
    SET
        updated_at = EXCLUDED.updated_at,
        expires_at = EXCLUDED.expires_at,
        status = EXCLUDED.status,
        metadata = EXCLUDED.metadata
    """
    params: Mapping[str, Any] = {
        "user_id": record.user_id,
        "session_id": record.session_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "expires_at": record.expires_at,
        "status": int(record.status),
        "model_id": record.model_id,
        "context_window_size": record.context_window_size,
        "system_prompt_hash": record.system_prompt_hash,
        "system_prompt_tokens": record.system_prompt_tokens,
        "temperature": record.temperature,
        "top_p": record.top_p,
        "max_response_tokens": record.max_response_tokens,
        "turn_count": record.turn_count,
        "total_prompt_tokens": record.total_prompt_tokens,
        "total_completion_tokens": record.total_completion_tokens,
        "total_cost_microcents": record.total_cost_microcents,
        "summary_text": record.summary_text,
        "summary_embedding": _pgvector_literal(record.summary_embedding) if record.summary_embedding else None,
        "summary_token_count": record.summary_token_count,
        "parent_session_id": record.parent_session_id,
        "metadata": dumps(record.metadata),
    }
    return QueryTemplate(sql=sql, params=params)


def build_insert_history(record: HistoryRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.conversation_history (
        user_id,
        session_id,
        message_id,
        turn_index,
        role,
        content,
        content_parts,
        name,
        tool_calls,
        tool_call_id,
        content_embedding,
        content_hash,
        token_count,
        model_id,
        finish_reason,
        prompt_tokens,
        completion_tokens,
        latency_ms,
        api_call_id,
        created_at,
        is_compressed,
        compressed_summary_id,
        is_pruned,
        metadata
    ) VALUES (
        %(user_id)s,
        %(session_id)s,
        %(message_id)s,
        %(turn_index)s,
        %(role)s,
        %(content)s,
        %(content_parts)s::jsonb,
        %(name)s,
        %(tool_calls)s::jsonb,
        %(tool_call_id)s,
        %(content_embedding)s,
        %(content_hash)s,
        %(token_count)s,
        %(model_id)s,
        %(finish_reason)s,
        %(prompt_tokens)s,
        %(completion_tokens)s,
        %(latency_ms)s,
        %(api_call_id)s,
        %(created_at)s,
        %(is_compressed)s,
        %(compressed_summary_id)s,
        %(is_pruned)s,
        %(metadata)s::jsonb
    )
    """
    params: Mapping[str, Any] = {
        "user_id": record.user_id,
        "session_id": record.session_id,
        "message_id": record.message_id,
        "turn_index": record.turn_index,
        "role": int(record.role),
        "content": record.content,
        "content_parts": dumps(record.content_parts) if record.content_parts is not None else None,
        "name": record.name,
        "tool_calls": dumps(record.tool_calls) if record.tool_calls is not None else None,
        "tool_call_id": record.tool_call_id,
        "content_embedding": _pgvector_literal(record.content_embedding) if record.content_embedding else None,
        "content_hash": record.content_hash,
        "token_count": record.token_count,
        "model_id": record.model_id,
        "finish_reason": int(record.finish_reason) if record.finish_reason is not None else None,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "latency_ms": record.latency_ms,
        "api_call_id": record.api_call_id,
        "created_at": record.created_at,
        "is_compressed": record.is_compressed,
        "compressed_summary_id": record.compressed_summary_id,
        "is_pruned": record.is_pruned,
        "metadata": dumps(record.metadata),
    }
    return QueryTemplate(sql=sql, params=params)


def build_upsert_memory(record: MemoryRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.memory_store (
        user_id,
        memory_id,
        session_id,
        memory_tier,
        memory_scope,
        content_text,
        content_embedding,
        content_hash,
        provenance_type,
        provenance_turn_id,
        confidence,
        relevance_accumulator,
        access_count,
        last_accessed_at,
        created_at,
        updated_at,
        expires_at,
        is_active,
        is_validated,
        token_count,
        superseded_by,
        tags,
        metadata
    ) VALUES (
        %(user_id)s,
        %(memory_id)s,
        %(session_id)s,
        %(memory_tier)s,
        %(memory_scope)s,
        %(content_text)s,
        %(content_embedding)s,
        %(content_hash)s,
        %(provenance_type)s,
        %(provenance_turn_id)s,
        %(confidence)s,
        %(relevance_accumulator)s,
        %(access_count)s,
        %(last_accessed_at)s,
        %(created_at)s,
        %(updated_at)s,
        %(expires_at)s,
        %(is_active)s,
        %(is_validated)s,
        %(token_count)s,
        %(superseded_by)s,
        %(tags)s,
        %(metadata)s::jsonb
    )
    ON CONFLICT (user_id, memory_id) DO UPDATE
    SET
        session_id = EXCLUDED.session_id,
        memory_tier = EXCLUDED.memory_tier,
        memory_scope = EXCLUDED.memory_scope,
        content_text = EXCLUDED.content_text,
        content_embedding = EXCLUDED.content_embedding,
        content_hash = EXCLUDED.content_hash,
        provenance_type = EXCLUDED.provenance_type,
        provenance_turn_id = EXCLUDED.provenance_turn_id,
        confidence = EXCLUDED.confidence,
        relevance_accumulator = EXCLUDED.relevance_accumulator,
        access_count = EXCLUDED.access_count,
        last_accessed_at = EXCLUDED.last_accessed_at,
        updated_at = EXCLUDED.updated_at,
        expires_at = EXCLUDED.expires_at,
        is_active = EXCLUDED.is_active,
        is_validated = EXCLUDED.is_validated,
        token_count = EXCLUDED.token_count,
        superseded_by = EXCLUDED.superseded_by,
        tags = EXCLUDED.tags,
        metadata = EXCLUDED.metadata
    """
    params: Mapping[str, Any] = {
        "user_id": record.user_id,
        "memory_id": record.memory_id,
        "session_id": record.session_id,
        "memory_tier": int(record.memory_tier),
        "memory_scope": int(record.memory_scope),
        "content_text": record.content_text,
        "content_embedding": _pgvector_literal(record.content_embedding) if record.content_embedding else None,
        "content_hash": record.content_hash,
        "provenance_type": int(record.provenance_type),
        "provenance_turn_id": record.provenance_turn_id,
        "confidence": record.confidence,
        "relevance_accumulator": record.relevance_accumulator,
        "access_count": record.access_count,
        "last_accessed_at": record.last_accessed_at,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "expires_at": record.expires_at,
        "is_active": record.is_active,
        "is_validated": record.is_validated,
        "token_count": record.token_count,
        "superseded_by": record.superseded_by,
        "tags": list(record.tags),
        "metadata": dumps(record.metadata),
    }
    return QueryTemplate(sql=sql, params=params)


def build_fetch_active_history(
    *,
    user_id: UUID,
    session_id: UUID,
    limit: int | None = None,
) -> QueryTemplate:
    sql = """
    SELECT *
    FROM agent_data.conversation_history
    WHERE user_id = %(user_id)s
      AND session_id = %(session_id)s
      AND is_compressed = false
      AND is_pruned = false
    ORDER BY turn_index DESC
    """
    if limit is not None:
        sql += "\nLIMIT %(limit)s"
    params: dict[str, Any] = {"user_id": user_id, "session_id": session_id}
    if limit is not None:
        params["limit"] = limit
    return QueryTemplate(sql=sql, params=params)


def build_fetch_semantic_memories(
    *,
    user_id: UUID,
    query_embedding: EmbeddingVector,
    limit: int = 50,
) -> QueryTemplate:
    sql = """
    SELECT *
    FROM agent_data.memory_store
    WHERE user_id = %(user_id)s
      AND is_active = true
      AND content_embedding IS NOT NULL
    ORDER BY content_embedding <=> %(query_embedding)s::vector
    LIMIT %(limit)s
    """
    return QueryTemplate(
        sql=sql,
        params={
            "user_id": user_id,
            "query_embedding": _pgvector_literal(query_embedding),
            "limit": limit,
        },
    )


def build_fetch_keyword_memories(
    *,
    user_id: UUID,
    query_text: str,
    limit: int = 20,
) -> QueryTemplate:
    sql = """
    SELECT *
    FROM agent_data.memory_store
    WHERE user_id = %(user_id)s
      AND is_active = true
      AND content_text %% %(query_text)s
    ORDER BY similarity(content_text, %(query_text)s) DESC, access_count DESC
    LIMIT %(limit)s
    """
    return QueryTemplate(sql=sql, params={"user_id": user_id, "query_text": query_text, "limit": limit})


def build_insert_api_invocation(record: APIInvocationRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.model_api_calls (
        user_id,
        api_call_id,
        session_id,
        provider,
        endpoint,
        model_id,
        request_payload,
        response_payload,
        usage_payload,
        error_payload,
        request_id,
        status_code,
        started_at,
        completed_at,
        latency_ms,
        succeeded,
        stream_mode,
        input_token_count,
        output_token_count,
        cached_input_token_count,
        reasoning_token_count,
        metadata
    ) VALUES (
        %(user_id)s,
        %(api_call_id)s,
        %(session_id)s,
        %(provider)s,
        %(endpoint)s,
        %(model_id)s,
        %(request_payload)s::jsonb,
        %(response_payload)s::jsonb,
        %(usage_payload)s::jsonb,
        %(error_payload)s::jsonb,
        %(request_id)s,
        %(status_code)s,
        %(started_at)s,
        %(completed_at)s,
        %(latency_ms)s,
        %(succeeded)s,
        %(stream_mode)s,
        %(input_token_count)s,
        %(output_token_count)s,
        %(cached_input_token_count)s,
        %(reasoning_token_count)s,
        %(metadata)s::jsonb
    )
    ON CONFLICT (user_id, api_call_id) DO UPDATE
    SET
        response_payload = EXCLUDED.response_payload,
        usage_payload = EXCLUDED.usage_payload,
        error_payload = EXCLUDED.error_payload,
        request_id = EXCLUDED.request_id,
        status_code = EXCLUDED.status_code,
        completed_at = EXCLUDED.completed_at,
        latency_ms = EXCLUDED.latency_ms,
        succeeded = EXCLUDED.succeeded,
        stream_mode = EXCLUDED.stream_mode,
        input_token_count = EXCLUDED.input_token_count,
        output_token_count = EXCLUDED.output_token_count,
        cached_input_token_count = EXCLUDED.cached_input_token_count,
        reasoning_token_count = EXCLUDED.reasoning_token_count,
        metadata = EXCLUDED.metadata
    """
    return QueryTemplate(
        sql=sql,
        params={
            "user_id": record.user_id,
            "api_call_id": record.api_call_id,
            "session_id": record.session_id,
            "provider": record.provider,
            "endpoint": record.endpoint,
            "model_id": record.model_id,
            "request_payload": dumps(record.request_payload),
            "response_payload": dumps(record.response_payload) if record.response_payload is not None else None,
            "usage_payload": dumps(record.usage_payload),
            "error_payload": dumps(record.error_payload) if record.error_payload is not None else None,
            "request_id": record.request_id,
            "status_code": record.status_code,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
            "latency_ms": record.latency_ms,
            "succeeded": record.succeeded,
            "stream_mode": record.stream_mode,
            "input_token_count": record.input_token_count,
            "output_token_count": record.output_token_count,
            "cached_input_token_count": record.cached_input_token_count,
            "reasoning_token_count": record.reasoning_token_count,
            "metadata": dumps(record.metadata),
        },
    )


def build_insert_stream_event(record: APIStreamEventRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.model_stream_events (
        user_id,
        api_call_id,
        event_index,
        event_type,
        text_delta,
        token_count,
        payload,
        binary_payload,
        mime_type,
        created_at
    ) VALUES (
        %(user_id)s,
        %(api_call_id)s,
        %(event_index)s,
        %(event_type)s,
        %(text_delta)s,
        %(token_count)s,
        %(payload)s::jsonb,
        %(binary_payload)s,
        %(mime_type)s,
        %(created_at)s
    )
    ON CONFLICT (user_id, api_call_id, event_index) DO NOTHING
    """
    return QueryTemplate(
        sql=sql,
        params={
            "user_id": record.user_id,
            "api_call_id": record.api_call_id,
            "event_index": record.event_index,
            "event_type": record.event_type,
            "text_delta": record.text_delta,
            "token_count": record.token_count,
            "payload": dumps(record.payload),
            "binary_payload": record.binary_payload,
            "mime_type": record.mime_type,
            "created_at": record.created_at,
        },
    )


def build_insert_protocol_event(record: ProtocolEventRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.protocol_events (
        user_id,
        protocol_event_id,
        session_id,
        api_call_id,
        message_id,
        protocol_type,
        direction,
        method,
        rpc_id,
        tool_name,
        tool_call_id,
        payload,
        binary_payload,
        mime_type,
        created_at
    ) VALUES (
        %(user_id)s,
        %(protocol_event_id)s,
        %(session_id)s,
        %(api_call_id)s,
        %(message_id)s,
        %(protocol_type)s,
        %(direction)s,
        %(method)s,
        %(rpc_id)s,
        %(tool_name)s,
        %(tool_call_id)s,
        %(payload)s::jsonb,
        %(binary_payload)s,
        %(mime_type)s,
        %(created_at)s
    )
    ON CONFLICT (user_id, protocol_event_id) DO NOTHING
    """
    return QueryTemplate(
        sql=sql,
        params={
            "user_id": record.user_id,
            "protocol_event_id": record.protocol_event_id,
            "session_id": record.session_id,
            "api_call_id": record.api_call_id,
            "message_id": record.message_id,
            "protocol_type": record.protocol_type,
            "direction": record.direction,
            "method": record.method,
            "rpc_id": record.rpc_id,
            "tool_name": record.tool_name,
            "tool_call_id": record.tool_call_id,
            "payload": dumps(record.payload),
            "binary_payload": record.binary_payload,
            "mime_type": record.mime_type,
            "created_at": record.created_at,
        },
    )
