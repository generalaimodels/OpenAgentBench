from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from openagentbench.agent_data import (
    OPENAI_ENDPOINTS,
    VLLM_ENDPOINTS,
    ContextCompiler,
    CompileRequest,
    HistoryRecord,
    MemoryRecord,
    MemoryScope,
    MemoryTier,
    MessageRole,
    ProvenanceType,
    SessionRecord,
    SessionStatus,
    TaskType,
    allocate_budget,
    hash_normalized_text,
)


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
        context_window_size=8_000,
        system_prompt_hash=hash_normalized_text("You are helpful."),
        system_prompt_tokens=10,
        max_response_tokens=1_000,
        summary_text="The user is building a memory module.",
        summary_token_count=12,
        system_prompt_text="You are helpful.",
    )


def _history(session: SessionRecord) -> list[HistoryRecord]:
    now = datetime.now(timezone.utc)
    return [
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=1,
            role=MessageRole.USER,
            content="Remember that I prefer PostgreSQL.",
            content_parts=(
                {"type": "input_text", "text": "Remember that I prefer PostgreSQL."},
                {"type": "input_image", "image_url": "https://example.com/architecture.png"},
            ),
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Remember that I prefer PostgreSQL."),
            token_count=20,
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
            content="Noted. I will keep PostgreSQL as the primary store.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Noted. I will keep PostgreSQL as the primary store."),
            token_count=24,
            model_id="gpt-4o-mini",
            finish_reason=None,
            prompt_tokens=100,
            completion_tokens=24,
            latency_ms=140,
            api_call_id=None,
            created_at=now - timedelta(minutes=1),
        ),
    ]


def _memories(session: SessionRecord) -> list[MemoryRecord]:
    now = datetime.now(timezone.utc)
    return [
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            memory_tier=MemoryTier.SEMANTIC,
            memory_scope=MemoryScope.GLOBAL,
            content_text="User prefers PostgreSQL for durable memory storage.",
            content_embedding=None,
            content_hash=hash_normalized_text("User prefers PostgreSQL for durable memory storage."),
            provenance_type=ProvenanceType.PREFERENCE,
            provenance_turn_id=None,
            confidence=0.95,
            relevance_accumulator=3.2,
            access_count=4,
            last_accessed_at=now - timedelta(hours=1),
            created_at=now - timedelta(days=2),
            updated_at=now - timedelta(hours=1),
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=16,
            tags=("database", "preference"),
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            memory_tier=MemoryTier.SESSION,
            memory_scope=MemoryScope.LOCAL,
            content_text="Current task is designing a memory module.",
            content_embedding=None,
            content_hash=hash_normalized_text("Current task is designing a memory module."),
            provenance_type=ProvenanceType.USER_STATED,
            provenance_turn_id=None,
            confidence=0.85,
            relevance_accumulator=2.1,
            access_count=2,
            last_accessed_at=now - timedelta(minutes=20),
            created_at=now - timedelta(hours=4),
            updated_at=now - timedelta(minutes=20),
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=False,
            token_count=12,
            tags=("task",),
        ),
    ]


def test_allocate_budget_respects_total_window() -> None:
    budget = allocate_budget(
        context_window_size=10_000,
        system_prompt_tokens=300,
        response_reserve=1_000,
        tool_budget=200,
        task_type=TaskType.KNOWLEDGE_INTENSIVE,
    )
    assert budget.total_budget == 8_500
    assert budget.memory_budget + budget.history_budget == 8_500


def test_compile_context_returns_system_memory_and_history() -> None:
    session = _session()
    compiler = ContextCompiler()
    result = compiler.compile_context(
        CompileRequest(
            user_id=session.user_id,
            session=session,
            query_text="What do you remember about my database preference?",
        ),
        history=_history(session),
        memories=_memories(session),
    )

    assert result.messages[0]["role"] == "system"
    assert any(message["role"] == "user" for message in result.messages)
    assert any("[Memory Context]" in (message["content"] or "") for message in result.messages)
    assert any(isinstance(message["content"], list) for message in result.messages if message["role"] == "user")
    assert result.selected_memories


def test_hash_normalized_text_collapses_whitespace() -> None:
    left = hash_normalized_text("hello   world")
    right = hash_normalized_text("hello world")
    assert left == right


def test_endpoint_catalogs_include_recent_openai_and_vllm_surfaces() -> None:
    paths = {endpoint.path for endpoint in OPENAI_ENDPOINTS}
    vllm_paths = {endpoint.path for endpoint in VLLM_ENDPOINTS}
    assert "/v1/completions" in paths
    assert "/v1/embeddings" in paths
    assert "/v1/videos" in paths
    assert "/v1/moderations" in paths
    assert "/v1/responses" in vllm_paths
    assert "/v1/realtime" in vllm_paths
    assert "/score" in vllm_paths
