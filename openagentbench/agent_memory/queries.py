"""Parameterized SQL templates for the PostgreSQL-backed memory module."""

from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from openagentbench.agent_data import MemoryScope
from openagentbench.agent_data.enums import MemoryTier
from openagentbench.agent_data.json_codec import dumps

from .models import MemoryAuditRecord, MemoryCacheEntry, QueryTemplate, SessionCheckpointRecord


def build_load_working_memory(
    *,
    user_id: UUID,
    session_id: UUID,
    step_id: UUID | None = None,
) -> QueryTemplate:
    sql = """
    SELECT *
    FROM agent_data.memory_store_working
    WHERE user_id = %(user_id)s
      AND session_id = %(session_id)s
    """
    params: dict[str, Any] = {"user_id": user_id, "session_id": session_id}
    if step_id is not None:
        sql += "\n  AND agent_step_id = %(step_id)s"
        params["step_id"] = step_id
    sql += "\nORDER BY created_at ASC"
    return QueryTemplate(sql=sql, params=params)


def build_load_durable_memories(
    *,
    user_id: UUID,
    session_id: UUID,
    tiers: Sequence[MemoryTier],
    limit: int = 64,
) -> QueryTemplate:
    sql = """
    SELECT *
    FROM agent_data.memory_store
    WHERE user_id = %(user_id)s
      AND memory_tier = ANY(%(tiers)s)
      AND is_active = true
      AND (
            memory_scope = %(global_scope)s
         OR (memory_scope = %(local_scope)s AND session_id = %(session_id)s)
      )
    ORDER BY
        CASE
            WHEN memory_tier = %(session_tier)s THEN 0
            WHEN memory_scope = %(local_scope)s THEN 1
            ELSE 2
        END,
        updated_at DESC
    LIMIT %(limit)s
    """
    params = {
        "user_id": user_id,
        "session_id": session_id,
        "tiers": [int(tier) for tier in tiers],
        "global_scope": int(MemoryScope.GLOBAL),
        "local_scope": int(MemoryScope.LOCAL),
        "session_tier": int(MemoryTier.SESSION),
        "limit": limit,
    }
    return QueryTemplate(sql=sql, params=params)


def build_insert_session_checkpoint(record: SessionCheckpointRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.session_checkpoints (
        user_id,
        session_id,
        checkpoint_id,
        checkpoint_seq,
        summary_text,
        summary_version,
        turn_count,
        working_item_ids,
        metadata,
        created_at
    ) VALUES (
        %(user_id)s,
        %(session_id)s,
        %(checkpoint_id)s,
        %(checkpoint_seq)s,
        %(summary_text)s,
        %(summary_version)s,
        %(turn_count)s,
        %(working_item_ids)s,
        %(metadata)s::jsonb,
        %(created_at)s
    )
    """
    params = {
        "user_id": record.user_id,
        "session_id": record.session_id,
        "checkpoint_id": record.checkpoint_id,
        "checkpoint_seq": record.checkpoint_seq,
        "summary_text": record.summary_text,
        "summary_version": record.summary_version,
        "turn_count": record.turn_count,
        "working_item_ids": [str(item_id) for item_id in record.working_item_ids],
        "metadata": dumps(record.metadata),
        "created_at": record.created_at,
    }
    return QueryTemplate(sql=sql, params=params)


def build_insert_memory_audit_log(record: MemoryAuditRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.memory_audit_log (
        user_id,
        audit_id,
        operation,
        layer,
        item_id,
        caller_id,
        session_id,
        result,
        latency_ms,
        token_delta,
        metadata,
        created_at
    ) VALUES (
        %(user_id)s,
        %(audit_id)s,
        %(operation)s,
        %(layer)s,
        %(item_id)s,
        %(caller_id)s,
        %(session_id)s,
        %(result)s,
        %(latency_ms)s,
        %(token_delta)s,
        %(metadata)s::jsonb,
        %(created_at)s
    )
    """
    params = {
        "user_id": record.user_id,
        "audit_id": record.audit_id,
        "operation": record.operation.value,
        "layer": int(record.layer) if record.layer is not None else None,
        "item_id": record.item_id,
        "caller_id": record.caller_id,
        "session_id": record.session_id,
        "result": record.result,
        "latency_ms": record.latency_ms,
        "token_delta": record.token_delta,
        "metadata": dumps(record.metadata),
        "created_at": record.created_at,
    }
    return QueryTemplate(sql=sql, params=params)


def build_insert_memory_cache_entry(entry: MemoryCacheEntry) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.memory_cache (
        cache_key,
        user_id,
        layer,
        payload,
        hit_count,
        embedding_bucket,
        created_at,
        expires_at
    ) VALUES (
        %(cache_key)s,
        %(user_id)s,
        %(layer)s,
        %(payload)s::jsonb,
        %(hit_count)s,
        %(embedding_bucket)s,
        %(created_at)s,
        %(expires_at)s
    )
    ON CONFLICT (cache_key) DO UPDATE
    SET
        payload = EXCLUDED.payload,
        hit_count = EXCLUDED.hit_count,
        embedding_bucket = EXCLUDED.embedding_bucket,
        created_at = EXCLUDED.created_at,
        expires_at = EXCLUDED.expires_at
    """
    params = {
        "cache_key": entry.cache_key,
        "user_id": entry.user_id,
        "layer": int(entry.layer),
        "payload": dumps(entry.payload),
        "hit_count": entry.hit_count,
        "embedding_bucket": entry.embedding_bucket,
        "created_at": entry.created_at,
        "expires_at": entry.expires_at,
    }
    return QueryTemplate(sql=sql, params=params)


def build_lookup_memory_cache(*, cache_key: str, user_id: UUID, layer: MemoryTier) -> QueryTemplate:
    sql = """
    SELECT *
    FROM agent_data.memory_cache
    WHERE cache_key = %(cache_key)s
      AND user_id = %(user_id)s
      AND layer = %(layer)s
      AND expires_at > now()
    """
    return QueryTemplate(
        sql=sql,
        params={"cache_key": cache_key, "user_id": user_id, "layer": int(layer)},
    )


def build_invalidate_memory_cache(*, user_id: UUID, layer: MemoryTier | None = None) -> QueryTemplate:
    sql = """
    DELETE FROM agent_data.memory_cache
    WHERE user_id = %(user_id)s
    """
    params: dict[str, Any] = {"user_id": user_id}
    if layer is not None:
        sql += "\n  AND layer = %(layer)s"
        params["layer"] = int(layer)
    return QueryTemplate(sql=sql, params=params)


__all__ = [
    "build_insert_memory_audit_log",
    "build_insert_memory_cache_entry",
    "build_insert_session_checkpoint",
    "build_invalidate_memory_cache",
    "build_load_durable_memories",
    "build_load_working_memory",
    "build_lookup_memory_cache",
]
