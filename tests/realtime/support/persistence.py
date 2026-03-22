"""Database harness for replay-grade realtime persistence verification."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable

from agent_data.runtime import schema_sql_path
from openagentbench.agent_data import (
    CompileRequest,
    ContextCompiler,
    FinishReason,
    HistoryRecord,
    MemoryRecord,
    MessageRole,
    QueryTemplate,
    build_fetch_active_history,
    build_fetch_keyword_memories,
    build_fetch_semantic_memories,
    build_insert_api_invocation,
    build_insert_history,
    build_insert_protocol_event,
    build_insert_session,
    build_insert_stream_event,
    build_upsert_memory,
)

from .cases import RealtimeCaseSpec
from .normalize import normalize_capture
from .types import NormalizedCaseRecords, VendorCapture

try:  # pragma: no cover - optional dependency for integration tests
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional dependency for integration tests
    psycopg = None
    dict_row = None


def _require_psycopg() -> None:
    if psycopg is None or dict_row is None:  # pragma: no cover - exercised through skip logic
        raise RuntimeError("psycopg is required for PostgreSQL integration tests")


def _schema_name(run_id: str) -> str:
    sanitized = "".join(character if character.isalnum() else "_" for character in run_id.lower())
    return f"agent_data_test_{sanitized[:32]}"


def rewrite_agent_data_schema(sql: str, schema_name: str) -> str:
    return sql.replace("CREATE SCHEMA IF NOT EXISTS agent_data;", f"CREATE SCHEMA IF NOT EXISTS {schema_name};").replace(
        "agent_data.",
        f"{schema_name}.",
    )


def _parse_vector(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].strip()
            if not inner:
                return tuple()
            return tuple(float(item) for item in inner.split(","))
    return None


def _offline_case_key(records: NormalizedCaseRecords) -> tuple[str, str]:
    return (str(records.session.user_id), str(records.session.session_id))


def _tokenize_text(value: str) -> tuple[str, ...]:
    normalized = []
    current = []
    for character in value.lower():
        if character.isalnum():
            current.append(character)
            continue
        if current:
            normalized.append("".join(current))
            current.clear()
    if current:
        normalized.append("".join(current))
    return tuple(normalized)


def _cosine_similarity(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float:
    if left is None or right is None or not left or not right or len(left) != len(right):
        return -1.0
    dot = sum(lhs * rhs for lhs, rhs in zip(left, right))
    left_norm = math.sqrt(sum(component * component for component in left))
    right_norm = math.sqrt(sum(component * component for component in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot / (left_norm * right_norm)


def execute_query_template(connection: Any, schema_name: str, template: QueryTemplate) -> Any:
    sql = template.sql.replace("agent_data.", f"{schema_name}.")
    with connection.cursor() as cursor:
        cursor.execute(sql, template.params)
        return cursor


def _insert_system_prompt(connection: Any, schema_name: str, records: NormalizedCaseRecords) -> None:
    sql = f"""
    INSERT INTO {schema_name}.system_prompts (
        prompt_hash,
        prompt_text,
        token_count,
        metadata
    ) VALUES (
        %(prompt_hash)s,
        %(prompt_text)s,
        %(token_count)s,
        %(metadata)s::jsonb
    )
    ON CONFLICT (prompt_hash) DO UPDATE
    SET prompt_text = EXCLUDED.prompt_text,
        token_count = EXCLUDED.token_count,
        metadata = EXCLUDED.metadata
    """
    with connection.cursor() as cursor:
        cursor.execute(
            sql,
            {
                "prompt_hash": records.session.system_prompt_hash,
                "prompt_text": records.system_prompt_text,
                "token_count": records.session.system_prompt_tokens,
                "metadata": '{"test_suite":"realtime"}',
            },
        )


def _row_to_history(row: dict[str, Any]) -> HistoryRecord:
    return HistoryRecord(
        message_id=row["message_id"],
        session_id=row["session_id"],
        user_id=row["user_id"],
        turn_index=row["turn_index"],
        role=MessageRole(row["role"]),
        content=row["content"],
        content_parts=tuple(row["content_parts"]) if row["content_parts"] is not None else None,
        name=row["name"],
        tool_calls=tuple(row["tool_calls"]) if row["tool_calls"] is not None else None,
        tool_call_id=row["tool_call_id"],
        content_embedding=_parse_vector(row["content_embedding"]),
        content_hash=row["content_hash"],
        token_count=row["token_count"],
        model_id=row["model_id"],
        finish_reason=FinishReason(row["finish_reason"]) if row["finish_reason"] is not None else None,
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        latency_ms=row["latency_ms"],
        api_call_id=row["api_call_id"],
        created_at=row["created_at"],
        is_compressed=row["is_compressed"],
        compressed_summary_id=row["compressed_summary_id"],
        is_pruned=row["is_pruned"],
        metadata=row["metadata"],
    )


def _row_to_memory(row: dict[str, Any]) -> MemoryRecord:
    from openagentbench.agent_data import MemoryScope, MemoryTier, ProvenanceType

    return MemoryRecord(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        memory_tier=MemoryTier(row["memory_tier"]),
        memory_scope=MemoryScope(row["memory_scope"]),
        content_text=row["content_text"],
        content_embedding=_parse_vector(row["content_embedding"]),
        content_hash=row["content_hash"],
        provenance_type=ProvenanceType(row["provenance_type"]),
        provenance_turn_id=row["provenance_turn_id"],
        confidence=row["confidence"],
        relevance_accumulator=row["relevance_accumulator"],
        access_count=row["access_count"],
        last_accessed_at=row["last_accessed_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
        is_active=row["is_active"],
        is_validated=row["is_validated"],
        token_count=row["token_count"],
        superseded_by=row["superseded_by"],
        tags=tuple(row["tags"]),
        metadata=row["metadata"],
    )


@dataclass(slots=True)
class DatabaseHarness:
    database_url: str | None
    run_id: str
    schema_name: str = field(init=False)
    persisted_cases: dict[tuple[str, str], NormalizedCaseRecords] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.schema_name = _schema_name(self.run_id)

    @property
    def uses_in_memory_store(self) -> bool:
        return not self.database_url

    @classmethod
    def in_memory(cls, run_id: str) -> "DatabaseHarness":
        return cls(database_url=None, run_id=run_id)

    def connect(self) -> Any:
        _require_psycopg()
        return psycopg.connect(self.database_url, autocommit=True, row_factory=dict_row)

    def ensure_schema(self) -> None:
        if self.uses_in_memory_store:
            return
        ddl = rewrite_agent_data_schema(
            Path(schema_sql_path()).read_text(encoding="utf-8"),
            self.schema_name,
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(ddl)

    def delete_case_rows(self, records: NormalizedCaseRecords) -> None:
        if self.uses_in_memory_store:
            self.persisted_cases.pop(_offline_case_key(records), None)
            return
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.schema_name}.model_stream_events WHERE user_id = %s AND api_call_id = %s",
                    (records.session.user_id, records.api_invocation.api_call_id),
                )
                cursor.execute(
                    f"DELETE FROM {self.schema_name}.protocol_events WHERE user_id = %s AND session_id = %s",
                    (records.session.user_id, records.session.session_id),
                )
                cursor.execute(
                    f"DELETE FROM {self.schema_name}.conversation_history WHERE user_id = %s AND session_id = %s",
                    (records.session.user_id, records.session.session_id),
                )
                cursor.execute(
                    f"DELETE FROM {self.schema_name}.memory_store WHERE user_id = %s AND session_id = %s",
                    (records.session.user_id, records.session.session_id),
                )
                cursor.execute(
                    f"DELETE FROM {self.schema_name}.model_api_calls WHERE user_id = %s AND api_call_id = %s",
                    (records.session.user_id, records.api_invocation.api_call_id),
                )
                cursor.execute(
                    f"DELETE FROM {self.schema_name}.sessions WHERE user_id = %s AND session_id = %s",
                    (records.session.user_id, records.session.session_id),
                )


def _load_persisted_offline_case(harness: DatabaseHarness, records: NormalizedCaseRecords) -> NormalizedCaseRecords:
    persisted = harness.persisted_cases.get(_offline_case_key(records))
    if persisted is None:
        raise AssertionError("normalized case was not persisted into the in-memory realtime harness")
    return persisted


def persist_normalized_case(harness: DatabaseHarness, records: NormalizedCaseRecords) -> None:
    if harness.uses_in_memory_store:
        harness.persisted_cases[_offline_case_key(records)] = records
        return
    harness.ensure_schema()
    with harness.connect() as connection:
        _insert_system_prompt(connection, harness.schema_name, records)
        execute_query_template(connection, harness.schema_name, build_insert_session(records.session))
        execute_query_template(connection, harness.schema_name, build_insert_api_invocation(records.api_invocation))
        for history in records.history:
            execute_query_template(connection, harness.schema_name, build_insert_history(history))
        for memory in records.memories:
            execute_query_template(connection, harness.schema_name, build_upsert_memory(memory))
        for event in records.stream_events:
            execute_query_template(connection, harness.schema_name, build_insert_stream_event(event))
        for protocol_event in records.protocol_events:
            execute_query_template(connection, harness.schema_name, build_insert_protocol_event(protocol_event))


def load_active_history_records(harness: DatabaseHarness, records: NormalizedCaseRecords) -> list[HistoryRecord]:
    if harness.uses_in_memory_store:
        persisted = _load_persisted_offline_case(harness, records)
        active_history = [record for record in persisted.history if not record.is_pruned]
        return sorted(active_history, key=lambda record: record.turn_index, reverse=True)
    with harness.connect() as connection:
        cursor = execute_query_template(
            connection,
            harness.schema_name,
            build_fetch_active_history(
                user_id=records.session.user_id,
                session_id=records.session.session_id,
            ),
        )
        rows = cursor.fetchall()
    return [_row_to_history(row) for row in rows]


def load_keyword_memory_records(
    harness: DatabaseHarness,
    records: NormalizedCaseRecords,
    query_text: str,
) -> list[MemoryRecord]:
    if harness.uses_in_memory_store:
        persisted = _load_persisted_offline_case(harness, records)
        query_tokens = set(_tokenize_text(query_text))
        ranked = []
        for memory in persisted.memories:
            if not memory.is_active:
                continue
            content_tokens = set(_tokenize_text(memory.content_text))
            overlap = len(query_tokens & content_tokens)
            ranked.append((overlap, memory.confidence, memory.access_count, memory.created_at, memory))
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        if ranked and ranked[0][0] > 0:
            return [item[-1] for item in ranked]
        fallback = [
            memory
            for memory in sorted(
                persisted.memories,
                key=lambda record: (record.confidence, record.access_count, record.created_at),
                reverse=True,
            )
            if memory.is_active
        ]
        return fallback
    with harness.connect() as connection:
        cursor = execute_query_template(
            connection,
            harness.schema_name,
            build_fetch_keyword_memories(user_id=records.session.user_id, query_text=query_text, limit=10),
        )
        rows = cursor.fetchall()
    return [_row_to_memory(row) for row in rows]


def load_semantic_memory_records(
    harness: DatabaseHarness,
    records: NormalizedCaseRecords,
    query_embedding: tuple[float, ...],
) -> list[MemoryRecord]:
    if harness.uses_in_memory_store:
        persisted = _load_persisted_offline_case(harness, records)
        ranked = []
        for memory in persisted.memories:
            if not memory.is_active:
                continue
            similarity = _cosine_similarity(memory.content_embedding, query_embedding)
            ranked.append((similarity, memory.confidence, memory.access_count, memory.created_at, memory))
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        return [item[-1] for item in ranked if item[0] >= -1.0]
    with harness.connect() as connection:
        cursor = execute_query_template(
            connection,
            harness.schema_name,
            build_fetch_semantic_memories(
                user_id=records.session.user_id,
                query_embedding=query_embedding,
                limit=10,
            ),
        )
        rows = cursor.fetchall()
    return [_row_to_memory(row) for row in rows]


def _fetch_count(connection: Any, schema_name: str, table_name: str, params: Iterable[Any], where_sql: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT count(*) AS count FROM {schema_name}.{table_name} WHERE {where_sql}", tuple(params))
        row = cursor.fetchone()
        return int(row["count"])


def assert_persisted_case(
    harness: DatabaseHarness,
    records: NormalizedCaseRecords,
    case: RealtimeCaseSpec,
) -> None:
    if harness.uses_in_memory_store:
        persisted = _load_persisted_offline_case(harness, records)
        assert persisted.session == records.session
        assert persisted.api_invocation == records.api_invocation
        assert persisted.history == records.history
        assert persisted.stream_events == records.stream_events
        assert persisted.protocol_events == records.protocol_events
        assert persisted.memories == records.memories

        history_rows = sorted(persisted.history, key=lambda record: record.turn_index)
        assert [row.turn_index for row in history_rows] == list(range(1, len(records.history) + 1))
        for expected, row in zip(records.history, history_rows):
            assert row.role == expected.role
            assert row.api_call_id == records.api_invocation.api_call_id
            if expected.content_parts is not None:
                assert row.content_parts == expected.content_parts

        event_rows = sorted(persisted.stream_events, key=lambda record: record.event_index)
        assert [row.event_index for row in event_rows] == list(range(len(records.stream_events)))

        protocol_rows = sorted(
            persisted.protocol_events,
            key=lambda record: (record.created_at, record.protocol_event_id),
        )
        assert len(protocol_rows) == len(records.protocol_events)
        for expected, row in zip(records.protocol_events, protocol_rows):
            assert row.protocol_type == expected.protocol_type
            assert row.direction == expected.direction
            assert row.method == expected.method
            assert row.tool_call_id == expected.tool_call_id
            assert row.rpc_id == expected.rpc_id
    else:
        with harness.connect() as connection:
            session_count = _fetch_count(
                connection,
                harness.schema_name,
                "sessions",
                [records.session.user_id, records.session.session_id],
                "user_id = %s AND session_id = %s",
            )
            assert session_count == 1

            api_count = _fetch_count(
                connection,
                harness.schema_name,
                "model_api_calls",
                [records.session.user_id, records.api_invocation.api_call_id],
                "user_id = %s AND api_call_id = %s",
            )
            assert api_count == 1

            history_count = _fetch_count(
                connection,
                harness.schema_name,
                "conversation_history",
                [records.session.user_id, records.session.session_id],
                "user_id = %s AND session_id = %s",
            )
            assert history_count == len(records.history)

            stream_count = _fetch_count(
                connection,
                harness.schema_name,
                "model_stream_events",
                [records.session.user_id, records.api_invocation.api_call_id],
                "user_id = %s AND api_call_id = %s",
            )
            assert stream_count == len(records.stream_events)

            protocol_count = _fetch_count(
                connection,
                harness.schema_name,
                "protocol_events",
                [records.session.user_id, records.session.session_id],
                "user_id = %s AND session_id = %s",
            )
            assert protocol_count == len(records.protocol_events)

            memory_count = _fetch_count(
                connection,
                harness.schema_name,
                "memory_store",
                [records.session.user_id, records.session.session_id],
                "user_id = %s AND session_id = %s",
            )
            assert memory_count == len(records.memories)

            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT turn_index, role, content, content_parts, api_call_id
                    FROM {harness.schema_name}.conversation_history
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY turn_index ASC
                    """,
                    (records.session.user_id, records.session.session_id),
                )
                history_rows = cursor.fetchall()
                assert [row["turn_index"] for row in history_rows] == list(range(1, len(records.history) + 1))
                for expected, row in zip(records.history, history_rows):
                    assert row["role"] == int(expected.role)
                    assert row["api_call_id"] == records.api_invocation.api_call_id
                    if expected.content_parts is not None:
                        assert tuple(row["content_parts"]) == expected.content_parts

                cursor.execute(
                    f"""
                    SELECT event_index
                    FROM {harness.schema_name}.model_stream_events
                    WHERE user_id = %s AND api_call_id = %s
                    ORDER BY event_index ASC
                    """,
                    (records.session.user_id, records.api_invocation.api_call_id),
                )
                event_rows = cursor.fetchall()
                assert [row["event_index"] for row in event_rows] == list(range(len(records.stream_events)))

                cursor.execute(
                    f"""
                    SELECT protocol_type, direction, method, tool_call_id, rpc_id
                    FROM {harness.schema_name}.protocol_events
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY created_at ASC, protocol_event_id ASC
                    """,
                    (records.session.user_id, records.session.session_id),
                )
                protocol_rows = cursor.fetchall()
                assert len(protocol_rows) == len(records.protocol_events)
                for expected, row in zip(records.protocol_events, protocol_rows):
                    assert row["protocol_type"] == expected.protocol_type
                    assert row["direction"] == expected.direction
                    assert row["method"] == expected.method
                    assert row["tool_call_id"] == expected.tool_call_id
                    assert row["rpc_id"] == expected.rpc_id

    active_history = load_active_history_records(harness, records)
    assert [record.turn_index for record in active_history] == sorted(
        (record.turn_index for record in records.history),
        reverse=True,
    )

    if records.memories:
        keyword_query = case.keyword_query_text or case.selection_query_text or "PostgreSQL"
        keyword_memories = load_keyword_memory_records(harness, records, keyword_query)
        assert keyword_memories, "keyword search should return at least one durable memory row"
        assert keyword_memories[0].memory_id == records.memories[0].memory_id

        semantic_memories = load_semantic_memory_records(
            harness,
            records,
            records.memories[0].content_embedding or tuple([0.0] * 1536),
        )
        assert semantic_memories, "semantic search should return at least one durable memory row"
        assert semantic_memories[0].memory_id == records.memories[0].memory_id

        compiler = ContextCompiler()
        compiled = compiler.compile_context(
            CompileRequest(
                user_id=records.session.user_id,
                session=records.session,
                query_text=case.selection_query_text or case.prompt,
                memory_budget_override=512,
                history_budget_override=512,
            ),
            history=list(reversed(active_history)),
            memories=keyword_memories,
        )
        assert compiled.selected_history, "history suffix selection should not be empty"
        assert compiled.selected_memories, "memory selection should not be empty"


def persist_capture(harness: DatabaseHarness, capture: VendorCapture, case: RealtimeCaseSpec, *, run_id: str) -> NormalizedCaseRecords:
    records = normalize_capture(capture, case, run_id=run_id)
    persist_normalized_case(harness, records)
    return records
