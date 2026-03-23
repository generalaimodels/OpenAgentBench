from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from openagentbench.agent_context import ContextCompileRequest, InMemoryContextRepository, compile_context
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
from openagentbench.agent_retrieval import Modality


def _session() -> SessionRecord:
    now = datetime(2026, 3, 23, 6, 0, 0, tzinfo=timezone.utc)
    return SessionRecord(
        session_id=uuid4(),
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id="gpt-5.4-mini",
        context_window_size=24_000,
        system_prompt_hash=hash_normalized_text("You are the agent-context test harness."),
        system_prompt_tokens=10,
        max_response_tokens=1_024,
        turn_count=3,
        summary_text="User is iterating on a cyclic agent plan.",
        summary_token_count=9,
        system_prompt_text="You are the agent-context test harness.",
    )


def _history(session: SessionRecord) -> tuple[HistoryRecord, ...]:
    now = datetime(2026, 3, 23, 6, 5, 0, tzinfo=timezone.utc)
    return (
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=1,
            role=MessageRole.USER,
            content="Keep the durable PostgreSQL rule in memory.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Keep the durable PostgreSQL rule in memory."),
            token_count=8,
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
            content="I will preserve the durable database constraint for later steps.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("I will preserve the durable database constraint for later steps."),
            token_count=11,
            model_id="gpt-5.4-mini",
            finish_reason=None,
            prompt_tokens=90,
            completion_tokens=11,
            latency_ms=80,
            api_call_id=None,
            created_at=now - timedelta(minutes=1),
        ),
    )


def _memories(session: SessionRecord) -> tuple[MemoryRecord, ...]:
    now = datetime(2026, 3, 23, 6, 10, 0, tzinfo=timezone.utc)
    return (
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
            relevance_accumulator=6.0,
            access_count=4,
            last_accessed_at=now,
            created_at=now - timedelta(days=3),
            updated_at=now,
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=10,
            tags=("database", "constraint"),
        ),
    )


def _working(session: SessionRecord) -> tuple[WorkingMemoryItem, ...]:
    return (
        WorkingMemoryItem(
            item_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            step_id=uuid4(),
            content_text="Working note: preserve durable state and provenance in the next cycle.",
            token_count=11,
            modality=Modality.TEXT,
            utility_score=0.9,
        ),
    )


def test_agent_context_compiles_deterministically_and_archives_context() -> None:
    session = _session()
    repository = InMemoryContextRepository()
    request = ContextCompileRequest(
        user_id=session.user_id,
        session=session,
        query_text="Plan the next grounded retrieval cycle using memory and history.",
        history=_history(session),
        memories=_memories(session),
        working_items=_working(session),
        active_tools=(
            {
                "type": "function",
                "function": {
                    "name": "memory_read",
                    "description": "Read validated memory for the current user.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
        ),
        metadata={"objective": "grounded_cycle"},
    )

    compiled_one = compile_context(request, repository=repository)
    compiled_two = compile_context(request, repository=repository)

    assert compiled_one.output_hash == compiled_two.output_hash
    assert compiled_one.trace.output_hash == compiled_two.trace.output_hash
    assert compiled_one.invariant_report.passed is True
    assert compiled_one.archive_entry.archive_hash == compiled_one.trace.archive_hash
    assert repository.latest_for_session(session.session_id) is not None
    assert compiled_one.responses_input[-1]["role"] == "user"
    assert compiled_one.openai_chat_request["messages"][-1]["role"] == "user"


def test_agent_context_rejects_unprovenanced_evidence_and_tracks_stable_prefix() -> None:
    session = _session()
    base_request = ContextCompileRequest(
        user_id=session.user_id,
        session=session,
        query_text="Retrieve grounded evidence for the repair plan.",
        history=_history(session),
        memories=_memories(session),
        working_items=_working(session),
        evidence_items=(
            {
                "content": "This evidence is missing provenance fields and should be rejected.",
            },
        ),
    )

    first = compile_context(base_request)
    second = compile_context(replace(base_request, prior_compiled_context=first, cycle_number=1))

    assert first.evidence_projection.rejected_without_provenance == 1
    assert first.invariant_report.provenance_coverage_ok is False
    assert second.stable_prefix_tokens >= first.policy_kernel.token_count
