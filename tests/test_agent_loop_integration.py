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
from openagentbench.agent_loop import AgentLoopEngine, CognitiveMode, LoopExecutionRequest, LoopPhase
from openagentbench.agent_memory import WorkingMemoryItem
from openagentbench.agent_retrieval import Modality


def _session() -> SessionRecord:
    now = datetime(2026, 3, 22, 18, 0, 0, tzinfo=timezone.utc)
    return SessionRecord(
        session_id=uuid4(),
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id="gpt-4.1-mini",
        context_window_size=32_000,
        system_prompt_hash=hash_normalized_text("You are the agent-loop integration test harness."),
        system_prompt_tokens=12,
        max_response_tokens=1_200,
        turn_count=4,
        summary_text="The user is coordinating retrieval, memory, and tool execution.",
        summary_token_count=12,
        system_prompt_text="You are the agent-loop integration test harness.",
    )


def _history(session: SessionRecord) -> list[HistoryRecord]:
    now = datetime(2026, 3, 22, 18, 10, 0, tzinfo=timezone.utc)
    return [
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=1,
            role=MessageRole.USER,
            content="Please remember that PostgreSQL is the durable database.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Please remember that PostgreSQL is the durable database."),
            token_count=9,
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
            content="I will preserve that preference and reuse it for future planning.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("I will preserve that preference and reuse it for future planning."),
            token_count=11,
            model_id="gpt-4.1-mini",
            finish_reason=None,
            prompt_tokens=100,
            completion_tokens=11,
            latency_ms=90,
            api_call_id=None,
            created_at=now - timedelta(minutes=1),
        ),
    ]


def _memories(session: SessionRecord) -> list[MemoryRecord]:
    now = datetime(2026, 3, 22, 18, 20, 0, tzinfo=timezone.utc)
    return [
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=None,
            memory_tier=MemoryTier.SEMANTIC,
            memory_scope=MemoryScope.GLOBAL,
            content_text="Global rule: PostgreSQL is the durable database of record.",
            content_embedding=None,
            content_hash=hash_normalized_text("Global rule: PostgreSQL is the durable database of record."),
            provenance_type=ProvenanceType.FACT,
            provenance_turn_id=None,
            confidence=0.99,
            relevance_accumulator=5.0,
            access_count=10,
            last_accessed_at=now,
            created_at=now - timedelta(days=4),
            updated_at=now,
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=10,
            tags=("database", "semantic"),
        )
    ]


def _working(session: SessionRecord) -> list[WorkingMemoryItem]:
    return [
        WorkingMemoryItem(
            item_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            step_id=uuid4(),
            content_text="Working note: preserve durable database preferences in session outcomes.",
            token_count=10,
            modality=Modality.TEXT,
        )
    ]


def test_agent_loop_enforces_write_wall_until_commit_and_supports_resume() -> None:
    session = _session()
    memories = _memories(session)
    engine = AgentLoopEngine()

    paused = engine.execute(
        LoopExecutionRequest(
            user_id=session.user_id,
            session=session,
            query_text="Remember my durable database preference and summarize it for this session.",
            stop_after_phase=LoopPhase.VERIFY,
        ),
        history=_history(session),
        memories=memories,
        working_items=_working(session),
    )

    assert paused.paused is True
    assert paused.last_completed_phase is LoopPhase.VERIFY
    assert paused.next_phase is LoopPhase.COMMIT
    assert paused.committed_writes == ()
    assert len(memories) == 1
    assert any(checkpoint.last_completed_phase is LoopPhase.VERIFY for checkpoint in paused.checkpoints)

    resumed = engine.resume(
        paused.loop_id,
        history=_history(session),
        memories=memories,
        working_items=_working(session),
    )

    assert resumed.paused is False
    assert resumed.last_completed_phase is LoopPhase.COMMIT
    assert resumed.committed_writes
    assert len(memories) > 1


def test_agent_loop_upgrades_from_fast_path_to_deliberative_mode_when_verify_fails() -> None:
    session = _session()
    engine = AgentLoopEngine()

    result = engine.execute(
        LoopExecutionRequest(
            user_id=session.user_id,
            session=session,
            query_text="What is my preference?",
        ),
        history=(),
        memories=[],
        working_items=(),
    )

    assert result.upgraded_from_fast_path is True
    assert result.cognitive_mode is CognitiveMode.SYSTEM2_DELIBERATIVE


def test_agent_loop_uses_tool_fallback_when_primary_tool_is_unauthorized() -> None:
    session = _session()
    engine = AgentLoopEngine()

    result = engine.execute(
        LoopExecutionRequest(
            user_id=session.user_id,
            session=session,
            query_text="Open https://example.com in the browser and compile the context.",
            scopes=("tools.read",),
        ),
        history=_history(session),
        memories=_memories(session),
        working_items=_working(session),
    )

    assert any(outcome.used_fallback for outcome in result.action_outcomes)
    assert any(outcome.status.value == "success" for outcome in result.action_outcomes)
