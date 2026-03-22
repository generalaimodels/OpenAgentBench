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
from openagentbench.agent_memory import (
    MemoryCompileRequest,
    MemoryContextCompiler,
    PromotionCandidate,
    WorkingMemoryBuffer,
    WorkingMemoryItem,
    build_session_checkpoint,
    decide_promotion,
    update_session_summary,
)
from openagentbench.agent_retrieval import MemoryType, Modality


def _session() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id=uuid4(),
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id="gpt-4o-mini",
        context_window_size=16_000,
        system_prompt_hash=hash_normalized_text("You are a memory-aware agent."),
        system_prompt_tokens=16,
        max_response_tokens=1_200,
        summary_text="The user is building a local, global, and session-aware memory system.",
        summary_token_count=15,
        system_prompt_text="You are a memory-aware agent.",
    )


def _memories(session: SessionRecord) -> list[MemoryRecord]:
    now = datetime.now(timezone.utc)
    other_session_id = uuid4()
    return [
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            memory_tier=MemoryTier.SESSION,
            memory_scope=MemoryScope.LOCAL,
            content_text="Current session decided to keep PostgreSQL as the checkpoint store.",
            content_embedding=None,
            content_hash=hash_normalized_text("Current session decided to keep PostgreSQL as the checkpoint store."),
            provenance_type=ProvenanceType.USER_STATED,
            provenance_turn_id=None,
            confidence=0.95,
            relevance_accumulator=2.5,
            access_count=4,
            last_accessed_at=now - timedelta(minutes=5),
            created_at=now - timedelta(hours=1),
            updated_at=now - timedelta(minutes=5),
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=12,
            tags=("session", "decision"),
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            memory_tier=MemoryTier.EPISODIC,
            memory_scope=MemoryScope.LOCAL,
            content_text="Local session episode: the user corrected the TTL policy to keep hot memories longer.",
            content_embedding=None,
            content_hash=hash_normalized_text("Local session episode: the user corrected the TTL policy to keep hot memories longer."),
            provenance_type=ProvenanceType.CORRECTION,
            provenance_turn_id=None,
            confidence=0.90,
            relevance_accumulator=2.0,
            access_count=3,
            last_accessed_at=now - timedelta(minutes=7),
            created_at=now - timedelta(hours=2),
            updated_at=now - timedelta(minutes=7),
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=14,
            tags=("local", "episodic"),
            metadata={"outcome_score": 0.92, "human_feedback_score": 0.3},
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=None,
            memory_tier=MemoryTier.SEMANTIC,
            memory_scope=MemoryScope.GLOBAL,
            content_text="Global rule: PostgreSQL is the durable source of truth for memory persistence.",
            content_embedding=None,
            content_hash=hash_normalized_text("Global rule: PostgreSQL is the durable source of truth for memory persistence."),
            provenance_type=ProvenanceType.FACT,
            provenance_turn_id=None,
            confidence=0.97,
            relevance_accumulator=4.0,
            access_count=12,
            last_accessed_at=now - timedelta(hours=2),
            created_at=now - timedelta(days=10),
            updated_at=now - timedelta(hours=2),
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=14,
            tags=("global", "semantic"),
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=None,
            memory_tier=MemoryTier.PROCEDURAL,
            memory_scope=MemoryScope.GLOBAL,
            content_text="Global procedure: checkpoint session state before summary compression and invalidate cache on promotion.",
            content_embedding=None,
            content_hash=hash_normalized_text("Global procedure: checkpoint session state before summary compression and invalidate cache on promotion."),
            provenance_type=ProvenanceType.INSTRUCTION,
            provenance_turn_id=None,
            confidence=0.98,
            relevance_accumulator=3.8,
            access_count=11,
            last_accessed_at=now - timedelta(hours=1),
            created_at=now - timedelta(days=5),
            updated_at=now - timedelta(hours=1),
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=15,
            tags=("global", "procedure"),
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=other_session_id,
            memory_tier=MemoryTier.SESSION,
            memory_scope=MemoryScope.LOCAL,
            content_text="Foreign local session memory that must never leak into this session.",
            content_embedding=None,
            content_hash=hash_normalized_text("Foreign local session memory that must never leak into this session."),
            provenance_type=ProvenanceType.USER_STATED,
            provenance_turn_id=None,
            confidence=0.99,
            relevance_accumulator=10.0,
            access_count=20,
            last_accessed_at=now,
            created_at=now - timedelta(minutes=10),
            updated_at=now,
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=11,
            tags=("forbidden",),
        ),
    ]


def test_memory_context_compiler_distinguishes_session_local_and_global_memory() -> None:
    session = _session()
    working_item = WorkingMemoryItem(
        item_id=uuid4(),
        user_id=session.user_id,
        session_id=session.session_id,
        step_id=uuid4(),
        content_text="Working note: validate cache invalidation after local promotion.",
        token_count=10,
        modality=Modality.TEXT,
        utility_score=0.8,
        carry_forward=True,
    )
    compiler = MemoryContextCompiler()
    result = compiler.compile_context(
        MemoryCompileRequest(
            user_id=session.user_id,
            session=session,
            query_text="What should this session remember about PostgreSQL, TTL corrections, and checkpointing?",
            total_budget=600,
        ),
        memories=_memories(session),
        working_items=(working_item,),
    )

    contents = [str(message["content"]) for message in result.messages]
    assert any("[Session Memory]" in content for content in contents)
    assert any("[Local Episodic Memory]" in content for content in contents)
    assert any("[Global Semantic Memory]" in content for content in contents)
    assert any("[Global Procedures]" in content for content in contents)
    assert not any("Foreign local session memory" in content for content in contents)
    assert result.selected_working


def test_working_memory_buffer_externalizes_low_utility_multimodal_items_first() -> None:
    user_id = uuid4()
    session_id = uuid4()
    step_id = uuid4()
    buffer = WorkingMemoryBuffer(capacity=80)
    buffer.extend(
        [
            WorkingMemoryItem(
                item_id=uuid4(),
                user_id=user_id,
                session_id=session_id,
                step_id=step_id,
                content_text="Critical code plan for the current task.",
                token_count=28,
                modality=Modality.CODE,
                dependency_count=3,
            ),
            WorkingMemoryItem(
                item_id=uuid4(),
                user_id=user_id,
                session_id=session_id,
                step_id=step_id,
                content_text="Large image artifact that can be externalized.",
                token_count=70,
                modality=Modality.IMAGE,
            ),
        ]
    )

    evicted = buffer.prune_to_capacity(query_text="code plan for the current task")
    assert buffer.token_used <= buffer.capacity
    assert any(item.binary_ref is not None for item in buffer.items if item.modality is Modality.IMAGE)
    assert not evicted or all(item.modality is not Modality.CODE for item in evicted)


def test_session_summary_and_checkpoint_preserve_corrections_and_decisions() -> None:
    session = _session()
    now = datetime.now(timezone.utc)
    turns = [
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=3,
            role=MessageRole.USER,
            content="Correction: keep local memory session-bound and do not leak it globally.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Correction: keep local memory session-bound and do not leak it globally."),
            token_count=15,
            model_id=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=None,
            api_call_id=None,
            created_at=now,
        ),
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=4,
            role=MessageRole.ASSISTANT,
            content="Decision: checkpoint before summary compression.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Decision: checkpoint before summary compression."),
            token_count=8,
            model_id="gpt-4o-mini",
            finish_reason=None,
            prompt_tokens=10,
            completion_tokens=8,
            latency_ms=100,
            api_call_id=None,
            created_at=now,
        ),
    ]

    summary = update_session_summary(
        existing_summary="Existing summary: design the memory module carefully.",
        new_turns=turns,
        max_tokens=80,
    )
    checkpoint = build_session_checkpoint(
        session=session,
        checkpoint_seq=1,
        summary_text=summary,
        summary_version=1,
        turn_count=4,
        working_items=[],
    )

    assert "[Correction]" in summary
    assert "[Decision]" in summary
    assert checkpoint.checkpoint_seq == 1
    assert checkpoint.summary_text == summary


def test_promotion_decision_prefers_global_procedural_target_for_reusable_trace() -> None:
    candidate = PromotionCandidate(
        user_id=uuid4(),
        source_id=uuid4(),
        source_layer=MemoryTier.EPISODIC,
        memory_type=MemoryType.PROCEDURE,
        content_text="Successful repeated trace for cache invalidation and checkpoint ordering.",
        token_count=24,
        novelty_score=0.82,
        correctness_score=0.90,
        reusability_score=0.91,
        modality=Modality.TRACE,
    )
    decision = decide_promotion(candidate)

    assert decision.action.value == "promote"
    assert decision.target_layer is MemoryTier.PROCEDURAL
    assert decision.promotion_score > 0.80
