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
from openagentbench.agent_query import QueryResolutionRequest, QueryResolver, RouteTarget
from openagentbench.agent_retrieval import Modality


def _session() -> SessionRecord:
    now = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
    return SessionRecord(
        session_id=uuid4(),
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id="gpt-4o-mini",
        context_window_size=24_000,
        system_prompt_hash=hash_normalized_text("You are a query-understanding test agent."),
        system_prompt_tokens=18,
        max_response_tokens=1_200,
        turn_count=5,
        summary_text="The user is building a memory-aware, tool-aware agent stack.",
        summary_token_count=14,
        system_prompt_text="You are a query-understanding test agent.",
    )


def _history(session: SessionRecord) -> list[HistoryRecord]:
    now = datetime(2026, 3, 23, 12, 5, 0, tzinfo=timezone.utc)
    return [
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=1,
            role=MessageRole.USER,
            content="Remember that PostgreSQL is the durable source of truth.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Remember that PostgreSQL is the durable source of truth."),
            token_count=10,
            model_id=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=None,
            api_call_id=None,
            created_at=now - timedelta(minutes=2),
        ),
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=2,
            role=MessageRole.ASSISTANT,
            content="I will use memory and tools together for later follow-ups.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("I will use memory and tools together for later follow-ups."),
            token_count=11,
            model_id="gpt-4o-mini",
            finish_reason=None,
            prompt_tokens=110,
            completion_tokens=11,
            latency_ms=120,
            api_call_id=None,
            created_at=now - timedelta(minutes=1),
        ),
    ]


def _memories(session: SessionRecord) -> list[MemoryRecord]:
    now = datetime(2026, 3, 23, 12, 10, 0, tzinfo=timezone.utc)
    return [
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            memory_tier=MemoryTier.SESSION,
            memory_scope=MemoryScope.LOCAL,
            content_text="Session rule: verify browser actions before summarizing.",
            content_embedding=None,
            content_hash=hash_normalized_text("Session rule: verify browser actions before summarizing."),
            provenance_type=ProvenanceType.INSTRUCTION,
            provenance_turn_id=None,
            confidence=0.95,
            relevance_accumulator=2.0,
            access_count=2,
            last_accessed_at=now,
            created_at=now - timedelta(hours=1),
            updated_at=now,
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=8,
            tags=("session", "browser"),
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=None,
            memory_tier=MemoryTier.SEMANTIC,
            memory_scope=MemoryScope.GLOBAL,
            content_text="Global semantic rule: PostgreSQL remains the durable source of truth.",
            content_embedding=None,
            content_hash=hash_normalized_text("Global semantic rule: PostgreSQL remains the durable source of truth."),
            provenance_type=ProvenanceType.FACT,
            provenance_turn_id=None,
            confidence=0.99,
            relevance_accumulator=4.0,
            access_count=12,
            last_accessed_at=now,
            created_at=now - timedelta(days=7),
            updated_at=now,
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=10,
            tags=("semantic", "database"),
            metadata={"modality_ref": "memory://semantic/postgres"},
        ),
    ]


def _working(session: SessionRecord) -> list[WorkingMemoryItem]:
    return [
        WorkingMemoryItem(
            item_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            step_id=uuid4(),
            content_text="Working note: browser verification happens before response synthesis.",
            token_count=9,
            modality=Modality.TEXT,
        )
    ]


def test_query_resolver_builds_decomposed_tool_and_memory_routes_and_caches() -> None:
    session = _session()
    resolver = QueryResolver()
    request = QueryResolutionRequest(
        user_id=session.user_id,
        session=session,
        query_text="Use memory to verify my PostgreSQL preference and then open the browser dashboard screenshot.",
    )

    first = resolver.resolve(
        request,
        history=_history(session),
        memories=_memories(session),
        working_items=_working(session),
    )
    second = resolver.resolve(
        request,
        history=_history(session),
        memories=_memories(session),
        working_items=_working(session),
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert first.plan.cognitive.requires_decomposition
    assert "browser_navigate" in first.plan.intent.relevant_tools
    assert "memory_read" in first.plan.intent.relevant_tools
    targets = {item.route_target for item in first.plan.subqueries}
    assert RouteTarget.MEMORY in targets
    assert RouteTarget.TOOL in targets


def test_query_resolver_tracks_psychological_signals_and_latent_goals() -> None:
    session = _session()
    resolver = QueryResolver()
    response = resolver.resolve(
        QueryResolutionRequest(
            user_id=session.user_id,
            session=session,
            query_text=(
                "I am stressed in a controlling environment and need the safest way to solve this "
                "without losing autonomy or increasing risk."
            ),
        ),
        history=_history(session),
        memories=_memories(session),
        working_items=_working(session),
    )

    profile = response.plan.pragmatic
    assert "stress_load" in profile.behavioral_signals
    assert "control_environment" in profile.behavioral_signals
    assert "risk_reduction" in profile.latent_goals
    assert "autonomy_preservation" in profile.latent_goals
