"""Parameterized SQL templates for the PostgreSQL-backed retrieval module."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence
from uuid import UUID

from openagentbench.agent_data.json_codec import dumps

from .models import HistoryEntry, MemoryEntry, QueryTemplate, SessionTurn, TimeWindow
from .types import EmbeddingVector


def _pgvector_literal(vector: EmbeddingVector) -> str:
    return "[" + ",".join(format(value, ".8g") for value in vector) + "]"


def _temporal_clause(temporal_scope: TimeWindow | None, *, column: str) -> tuple[str, dict[str, object]]:
    if temporal_scope is None:
        return "", {}
    return (
        f" AND {column} >= %(window_start)s AND {column} <= %(window_end)s",
        {"window_start": temporal_scope.start, "window_end": temporal_scope.end},
    )


def build_verify_user_active(*, uu_id: UUID) -> QueryTemplate:
    sql = """
    SELECT 1
    FROM agent_retrieval.users
    WHERE uu_id = %(uu_id)s
      AND status = 'active'
    """
    return QueryTemplate(sql=sql, params={"uu_id": uu_id})


def build_load_session_context(*, uu_id: UUID, session_id: UUID, limit: int) -> QueryTemplate:
    sql = """
    SELECT uu_id, session_id, turn_index, role, content_text, tokens_used, tool_calls, metadata, created_at, expires_at
    FROM agent_retrieval.session
    WHERE uu_id = %(uu_id)s
      AND session_id = %(session_id)s
    ORDER BY turn_index DESC
    LIMIT %(limit)s
    """
    return QueryTemplate(sql=sql, params={"uu_id": uu_id, "session_id": session_id, "limit": limit})


def build_load_memory_summary(*, uu_id: UUID, limit: int) -> QueryTemplate:
    sql = """
    SELECT *
    FROM agent_retrieval.memory
    WHERE uu_id = %(uu_id)s
      AND (expires_at IS NULL OR expires_at > now())
      AND authority_tier IN ('canonical', 'curated', 'derived')
    ORDER BY
        CASE authority_tier
            WHEN 'canonical' THEN 4
            WHEN 'curated' THEN 3
            WHEN 'derived' THEN 2
            ELSE 1
        END DESC,
        confidence DESC,
        access_count DESC
    LIMIT %(limit)s
    """
    return QueryTemplate(sql=sql, params={"uu_id": uu_id, "limit": limit})


def build_insert_session_turn(record: SessionTurn) -> QueryTemplate:
    sql = """
    INSERT INTO agent_retrieval.session (
        uu_id,
        session_id,
        turn_index,
        role,
        content_text,
        content_embedding,
        tokens_used,
        tool_calls,
        metadata,
        created_at,
        expires_at
    ) VALUES (
        %(uu_id)s,
        %(session_id)s,
        %(turn_index)s,
        %(role)s,
        %(content_text)s,
        %(content_embedding)s,
        %(tokens_used)s,
        %(tool_calls)s::jsonb,
        %(metadata)s::jsonb,
        %(created_at)s,
        %(expires_at)s
    )
    ON CONFLICT (uu_id, session_id, turn_index) DO UPDATE
    SET
        content_text = EXCLUDED.content_text,
        content_embedding = EXCLUDED.content_embedding,
        tokens_used = EXCLUDED.tokens_used,
        tool_calls = EXCLUDED.tool_calls,
        metadata = EXCLUDED.metadata,
        expires_at = EXCLUDED.expires_at
    """
    params = {
        "uu_id": record.uu_id,
        "session_id": record.session_id,
        "turn_index": record.turn_index,
        "role": str(record.role),
        "content_text": record.content_text,
        "content_embedding": None,
        "tokens_used": record.tokens_used,
        "tool_calls": dumps(record.tool_calls) if record.tool_calls is not None else None,
        "metadata": dumps(record.metadata),
        "created_at": record.created_at,
        "expires_at": record.expires_at,
    }
    return QueryTemplate(sql=sql, params=params)


def build_insert_history_entry(record: HistoryEntry) -> QueryTemplate:
    sql = """
    INSERT INTO agent_retrieval.history (
        uu_id,
        history_id,
        query_text,
        query_embedding,
        response_summary,
        evidence_used,
        task_outcome,
        human_feedback,
        utility_score,
        negative_flag,
        tags,
        metadata,
        created_at,
        session_origin
    ) VALUES (
        %(uu_id)s,
        %(history_id)s,
        %(query_text)s,
        %(query_embedding)s,
        %(response_summary)s,
        %(evidence_used)s::jsonb,
        %(task_outcome)s,
        %(human_feedback)s,
        %(utility_score)s,
        %(negative_flag)s,
        %(tags)s,
        %(metadata)s::jsonb,
        %(created_at)s,
        %(session_origin)s
    )
    ON CONFLICT (uu_id, history_id) DO UPDATE
    SET
        response_summary = EXCLUDED.response_summary,
        evidence_used = EXCLUDED.evidence_used,
        task_outcome = EXCLUDED.task_outcome,
        human_feedback = EXCLUDED.human_feedback,
        utility_score = EXCLUDED.utility_score,
        negative_flag = EXCLUDED.negative_flag,
        tags = EXCLUDED.tags,
        metadata = EXCLUDED.metadata
    """
    params = {
        "uu_id": record.uu_id,
        "history_id": record.history_id,
        "query_text": record.query_text,
        "query_embedding": _pgvector_literal(record.query_embedding) if record.query_embedding else None,
        "response_summary": record.response_summary,
        "evidence_used": dumps(
            [
                {
                    "chunk_id": str(item.locator.chunk_id),
                    "source": str(item.locator.source_table),
                    "utility_score": item.utility_score,
                    "use_count": item.use_count,
                    "was_cited": item.was_cited,
                }
                for item in record.evidence_used
            ]
        ),
        "task_outcome": str(record.task_outcome),
        "human_feedback": str(record.human_feedback),
        "utility_score": record.utility_score,
        "negative_flag": record.negative_flag,
        "tags": list(record.tags),
        "metadata": dumps(record.metadata),
        "created_at": record.created_at,
        "session_origin": record.session_origin,
    }
    return QueryTemplate(sql=sql, params=params)


def build_upsert_memory_entry(record: MemoryEntry) -> QueryTemplate:
    sql = """
    INSERT INTO agent_retrieval.memory (
        uu_id,
        memory_id,
        memory_type,
        content_text,
        content_embedding,
        authority_tier,
        confidence,
        source_provenance,
        verified_by,
        supersedes,
        created_at,
        updated_at,
        expires_at,
        access_count,
        last_accessed_at,
        content_hash,
        metadata
    ) VALUES (
        %(uu_id)s,
        %(memory_id)s,
        %(memory_type)s,
        %(content_text)s,
        %(content_embedding)s,
        %(authority_tier)s,
        %(confidence)s,
        %(source_provenance)s::jsonb,
        %(verified_by)s,
        %(supersedes)s,
        %(created_at)s,
        %(updated_at)s,
        %(expires_at)s,
        %(access_count)s,
        %(last_accessed_at)s,
        %(content_hash)s,
        %(metadata)s::jsonb
    )
    ON CONFLICT (uu_id, memory_id) DO UPDATE
    SET
        content_text = EXCLUDED.content_text,
        content_embedding = EXCLUDED.content_embedding,
        authority_tier = EXCLUDED.authority_tier,
        confidence = EXCLUDED.confidence,
        source_provenance = EXCLUDED.source_provenance,
        verified_by = EXCLUDED.verified_by,
        supersedes = EXCLUDED.supersedes,
        updated_at = EXCLUDED.updated_at,
        expires_at = EXCLUDED.expires_at,
        access_count = EXCLUDED.access_count,
        last_accessed_at = EXCLUDED.last_accessed_at,
        metadata = EXCLUDED.metadata
    """
    params = {
        "uu_id": record.uu_id,
        "memory_id": record.memory_id,
        "memory_type": str(record.memory_type),
        "content_text": record.content_text,
        "content_embedding": _pgvector_literal(record.content_embedding) if record.content_embedding else None,
        "authority_tier": str(record.authority_tier),
        "confidence": record.confidence,
        "source_provenance": dumps(record.source_provenance),
        "verified_by": list(record.verified_by),
        "supersedes": list(record.supersedes),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "expires_at": record.expires_at,
        "access_count": record.access_count,
        "last_accessed_at": record.last_accessed_at,
        "content_hash": record.content_hash,
        "metadata": dumps(record.metadata),
    }
    return QueryTemplate(sql=sql, params=params)


def build_exact_session_retrieval(
    *,
    uu_id: UUID,
    query_text: str,
    temporal_scope: TimeWindow | None,
    limit: int,
) -> QueryTemplate:
    clause, params = _temporal_clause(temporal_scope, column="created_at")
    sql = f"""
    SELECT *,
           ts_rank_cd(to_tsvector('simple', content_text), plainto_tsquery('simple', %(query_text)s)) AS bm25_score,
           similarity(content_text, %(query_text)s) AS trigram_score
    FROM agent_retrieval.session
    WHERE uu_id = %(uu_id)s
      AND to_tsvector('simple', content_text) @@ plainto_tsquery('simple', %(query_text)s)
      {clause}
    ORDER BY bm25_score DESC, trigram_score DESC
    LIMIT %(limit)s
    """
    params.update({"uu_id": uu_id, "query_text": query_text, "limit": limit})
    return QueryTemplate(sql=sql, params=params)


def build_exact_history_retrieval(
    *,
    uu_id: UUID,
    query_text: str,
    temporal_scope: TimeWindow | None,
    limit: int,
) -> QueryTemplate:
    clause, params = _temporal_clause(temporal_scope, column="created_at")
    sql = f"""
    SELECT *,
           ts_rank_cd(
               to_tsvector('simple', query_text || ' ' || coalesce(response_summary, '')),
               plainto_tsquery('simple', %(query_text)s)
           ) AS bm25_score,
           similarity(query_text, %(query_text)s) AS trigram_score
    FROM agent_retrieval.history
    WHERE uu_id = %(uu_id)s
      AND negative_flag = false
      AND to_tsvector('simple', query_text || ' ' || coalesce(response_summary, ''))
          @@ plainto_tsquery('simple', %(query_text)s)
      {clause}
    ORDER BY bm25_score DESC, trigram_score DESC
    LIMIT %(limit)s
    """
    params.update({"uu_id": uu_id, "query_text": query_text, "limit": limit})
    return QueryTemplate(sql=sql, params=params)


def build_exact_memory_retrieval(
    *,
    uu_id: UUID,
    query_text: str,
    temporal_scope: TimeWindow | None,
    limit: int,
) -> QueryTemplate:
    clause, params = _temporal_clause(temporal_scope, column="updated_at")
    sql = f"""
    SELECT *,
           ts_rank_cd(to_tsvector('simple', content_text), plainto_tsquery('simple', %(query_text)s)) AS bm25_score,
           similarity(content_text, %(query_text)s) AS trigram_score
    FROM agent_retrieval.memory
    WHERE uu_id = %(uu_id)s
      AND (expires_at IS NULL OR expires_at > now())
      AND to_tsvector('simple', content_text) @@ plainto_tsquery('simple', %(query_text)s)
      {clause}
    ORDER BY bm25_score DESC, trigram_score DESC
    LIMIT %(limit)s
    """
    params.update({"uu_id": uu_id, "query_text": query_text, "limit": limit})
    return QueryTemplate(sql=sql, params=params)


def build_semantic_retrieval(
    *,
    table_name: str,
    uu_id: UUID,
    vector_column: str,
    query_embedding: EmbeddingVector,
    temporal_scope: TimeWindow | None,
    created_at_column: str,
    limit: int,
) -> QueryTemplate:
    clause, params = _temporal_clause(temporal_scope, column=created_at_column)
    sql = f"""
    SELECT *
    FROM agent_retrieval.{table_name}
    WHERE uu_id = %(uu_id)s
      {clause}
    ORDER BY {vector_column} <=> %(query_embedding)s::vector
    LIMIT %(limit)s
    """
    params.update(
        {
            "uu_id": uu_id,
            "query_embedding": _pgvector_literal(query_embedding),
            "limit": limit,
        }
    )
    return QueryTemplate(sql=sql, params=params)


def build_touch_memory_access(*, uu_id: UUID, memory_ids: Sequence[UUID], accessed_at: datetime) -> QueryTemplate:
    sql = """
    UPDATE agent_retrieval.memory
    SET access_count = access_count + 1,
        last_accessed_at = %(accessed_at)s,
        updated_at = GREATEST(updated_at, %(accessed_at)s)
    WHERE uu_id = %(uu_id)s
      AND memory_id = ANY(%(memory_ids)s::uuid[])
    """
    return QueryTemplate(
        sql=sql,
        params={
            "uu_id": uu_id,
            "memory_ids": [str(memory_id) for memory_id in memory_ids],
            "accessed_at": accessed_at,
        },
    )
