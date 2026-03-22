from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from openagentbench.agent_data.enums import MemoryTier
from openagentbench.agent_memory import (
    MemoryAuditRecord,
    MemoryCacheEntry,
    MemoryOperation,
    SessionCheckpointRecord,
    build_insert_memory_audit_log,
    build_insert_memory_cache_entry,
    build_insert_session_checkpoint,
    build_invalidate_memory_cache,
    build_load_durable_memories,
    build_load_working_memory,
    read_schema_sql,
    schema_sql_path,
)


def test_scope_aware_sql_templates_are_user_scoped() -> None:
    user_id = uuid4()
    session_id = uuid4()

    durable = build_load_durable_memories(
        user_id=user_id,
        session_id=session_id,
        tiers=(MemoryTier.SESSION, MemoryTier.SEMANTIC, MemoryTier.PROCEDURAL),
    )
    working = build_load_working_memory(user_id=user_id, session_id=session_id)

    assert "user_id = %(user_id)s" in durable.sql
    assert "memory_scope" in durable.sql
    assert durable.params["session_id"] == session_id
    assert "user_id = %(user_id)s" in working.sql


def test_insert_templates_include_expected_payload_fields() -> None:
    now = datetime.now(timezone.utc)
    checkpoint = SessionCheckpointRecord(
        checkpoint_id=uuid4(),
        user_id=uuid4(),
        session_id=uuid4(),
        checkpoint_seq=2,
        summary_text="summary",
        summary_version=3,
        turn_count=8,
        working_item_ids=(uuid4(), uuid4()),
        created_at=now,
    )
    audit = MemoryAuditRecord(
        audit_id=uuid4(),
        user_id=checkpoint.user_id,
        operation=MemoryOperation.CHECKPOINT,
        layer=MemoryTier.SESSION,
        item_id=checkpoint.checkpoint_id,
        result="success",
        created_at=now,
    )
    cache = MemoryCacheEntry(
        cache_key="user:semantic:q",
        user_id=checkpoint.user_id,
        layer=MemoryTier.SEMANTIC,
        payload={"status": "ok"},
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )

    checkpoint_query = build_insert_session_checkpoint(checkpoint)
    audit_query = build_insert_memory_audit_log(audit)
    cache_query = build_insert_memory_cache_entry(cache)
    invalidate_query = build_invalidate_memory_cache(user_id=checkpoint.user_id, layer=MemoryTier.SEMANTIC)

    assert checkpoint_query.params["checkpoint_seq"] == 2
    assert len(checkpoint_query.params["working_item_ids"]) == 2
    assert audit_query.params["operation"] == "checkpoint"
    assert cache_query.params["cache_key"] == "user:semantic:q"
    assert "layer = %(layer)s" in invalidate_query.sql


def test_schema_migration_exists_and_contains_memory_extension_tables() -> None:
    schema_text = read_schema_sql()

    assert schema_sql_path().name == "002_agent_memory_schema.sql"
    assert "ALTER TABLE agent_data.memory_store" in schema_text
    assert "CREATE TABLE IF NOT EXISTS agent_data.session_checkpoints" in schema_text
    assert "CREATE TABLE IF NOT EXISTS agent_data.memory_audit_log" in schema_text
    assert "CREATE UNLOGGED TABLE IF NOT EXISTS agent_data.memory_cache" in schema_text
