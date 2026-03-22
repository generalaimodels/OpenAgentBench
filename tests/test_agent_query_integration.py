from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from openagentbench.agent_data import (
    HistoryRecord,
    MemoryRecord,
    MemoryScope,
    MemoryTier,
    MessageRole,
    ProvenanceType,
    SessionRecord,
    SessionStatus,
    hash_normalized_text,
)
from openagentbench.agent_memory import WorkingMemoryItem
from openagentbench.agent_query import QueryResolutionRequest, QueryResolver, read_schema_sql
from openagentbench.agent_retrieval import Modality


def _session() -> SessionRecord:
    now = datetime(2026, 3, 23, 13, 0, 0, tzinfo=timezone.utc)
    return SessionRecord(
        session_id=uuid4(),
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id="gpt-4.1-mini",
        context_window_size=32_000,
        system_prompt_hash=hash_normalized_text("You are an integration test agent."),
        system_prompt_tokens=16,
        max_response_tokens=1_600,
        turn_count=6,
        summary_text="The user is coordinating memory, retrieval, tool routing, and delegated analysis.",
        summary_token_count=15,
        system_prompt_text="You are an integration test agent.",
    )


def _history(session: SessionRecord) -> list[HistoryRecord]:
    now = datetime(2026, 3, 23, 13, 5, 0, tzinfo=timezone.utc)
    return [
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=1,
            role=MessageRole.USER,
            content="Analyze the failure mode and keep the durable database rule in mind.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Analyze the failure mode and keep the durable database rule in mind."),
            token_count=11,
            model_id=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=None,
            api_call_id=None,
            created_at=now - timedelta(minutes=2),
        )
    ]


def _memories(session: SessionRecord) -> list[MemoryRecord]:
    now = datetime(2026, 3, 23, 13, 10, 0, tzinfo=timezone.utc)
    return [
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=None,
            memory_tier=MemoryTier.SEMANTIC,
            memory_scope=MemoryScope.GLOBAL,
            content_text="Global rule: PostgreSQL is the durable source of truth.",
            content_embedding=None,
            content_hash=hash_normalized_text("Global rule: PostgreSQL is the durable source of truth."),
            provenance_type=ProvenanceType.FACT,
            provenance_turn_id=None,
            confidence=0.99,
            relevance_accumulator=5.0,
            access_count=9,
            last_accessed_at=now,
            created_at=now - timedelta(days=4),
            updated_at=now,
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=9,
            tags=("global", "database"),
        ),
    ]


def _working(session: SessionRecord) -> list[WorkingMemoryItem]:
    return [
        WorkingMemoryItem(
            item_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            step_id=uuid4(),
            content_text="Working note: delegated analysis is allowed when reasoning depth is high.",
            token_count=10,
            modality=Modality.TEXT,
        )
    ]


def test_query_module_integrates_memory_tools_and_model_routing() -> None:
    session = _session()
    resolver = QueryResolver()
    response = resolver.resolve(
        QueryResolutionRequest(
            user_id=session.user_id,
            session=session,
            query_text="Analyze this failure deeply, remember the durable database rule, and delegate if needed.",
        ),
        history=_history(session),
        memories=_memories(session),
        working_items=_working(session),
    )

    assert response.plan.selected_model_plan.primary_model is not None
    assert response.plan.context.memory_messages
    assert response.plan.context.tool_affordances
    assert response.plan.subqueries
    assert "agent_query_resolution_cache" in read_schema_sql()
    assert "agent_query_resolution_audit" in read_schema_sql()
