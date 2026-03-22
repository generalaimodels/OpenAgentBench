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
from openagentbench.agent_retrieval import Modality
from openagentbench.agent_tools import (
    ExecutionContext,
    OpenAgentBenchToolSuite,
    ToolExecutionEngine,
    ToolInvocationRequest,
    assert_agent_tools_endpoint_payload_compatibility,
    build_agent_tools_endpoint_compatibility_report,
    read_schema_sql,
)


def _session() -> SessionRecord:
    now = datetime(2026, 3, 22, 18, 0, 0, tzinfo=timezone.utc)
    return SessionRecord(
        session_id=uuid4(),
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id="gpt-4o-mini",
        context_window_size=24_000,
        system_prompt_hash=hash_normalized_text("You are an agent-tools integration test."),
        system_prompt_tokens=18,
        max_response_tokens=1_200,
        turn_count=4,
        summary_text="The user is validating tool, memory, and retrieval integration.",
        summary_token_count=16,
        system_prompt_text="You are an agent-tools integration test.",
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
            content="Remember the PostgreSQL preference and checkpoint ordering.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Remember the PostgreSQL preference and checkpoint ordering."),
            token_count=10,
            model_id=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=None,
            api_call_id=None,
            created_at=now - timedelta(minutes=1),
        )
    ]


def _memories(session: SessionRecord) -> list[MemoryRecord]:
    now = datetime(2026, 3, 22, 18, 20, 0, tzinfo=timezone.utc)
    foreign_session_id = uuid4()
    return [
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            memory_tier=MemoryTier.SESSION,
            memory_scope=MemoryScope.LOCAL,
            content_text="Session checkpoint before summary compression.",
            content_embedding=None,
            content_hash=hash_normalized_text("Session checkpoint before summary compression."),
            provenance_type=ProvenanceType.USER_STATED,
            provenance_turn_id=None,
            confidence=0.92,
            relevance_accumulator=2.5,
            access_count=3,
            last_accessed_at=now,
            created_at=now - timedelta(hours=2),
            updated_at=now,
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=6,
            tags=("session", "checkpoint"),
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=None,
            memory_tier=MemoryTier.SEMANTIC,
            memory_scope=MemoryScope.GLOBAL,
            content_text="Global semantic rule: PostgreSQL is the durable source of truth.",
            content_embedding=None,
            content_hash=hash_normalized_text("Global semantic rule: PostgreSQL is the durable source of truth."),
            provenance_type=ProvenanceType.FACT,
            provenance_turn_id=None,
            confidence=0.98,
            relevance_accumulator=5.0,
            access_count=10,
            last_accessed_at=now,
            created_at=now - timedelta(days=10),
            updated_at=now,
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=11,
            tags=("global", "database"),
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=foreign_session_id,
            memory_tier=MemoryTier.SESSION,
            memory_scope=MemoryScope.LOCAL,
            content_text="Foreign session memory that must remain isolated.",
            content_embedding=None,
            content_hash=hash_normalized_text("Foreign session memory that must remain isolated."),
            provenance_type=ProvenanceType.USER_STATED,
            provenance_turn_id=None,
            confidence=1.0,
            relevance_accumulator=10.0,
            access_count=20,
            last_accessed_at=now,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=7,
            tags=("forbidden",),
        ),
    ]


def _working(session: SessionRecord) -> list[WorkingMemoryItem]:
    return [
        WorkingMemoryItem(
            item_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            step_id=uuid4(),
            content_text="Working note: keep memory and tools aligned.",
            token_count=8,
            modality=Modality.TEXT,
        )
    ]


def _context(session: SessionRecord) -> ExecutionContext:
    return ExecutionContext(
        user_id=session.user_id,
        agent_id=uuid4(),
        session_id=session.session_id,
        scopes=(
            "tools.read",
            "tools.write",
            "tools.admin",
            "tools.browser",
            "tools.vision",
            "tools.delegate",
        ),
        trace_id="trace-integration-agent-tools",
    )


def _request(
    *,
    tool_id: str,
    params: dict[str, object],
    context: ExecutionContext,
    idempotency_key: str | None = None,
) -> ToolInvocationRequest:
    return ToolInvocationRequest(
        tool_id=tool_id,
        params=params,
        context=context,
        idempotency_key=idempotency_key,
    )


def test_openagentbench_tool_suite_integrates_data_retrieval_memory_and_tooling() -> None:
    session = _session()
    suite = OpenAgentBenchToolSuite(
        sessions={session.session_id: session},
        history_by_session={session.session_id: _history(session)},
        memories_by_user={session.user_id: _memories(session)},
        working_by_session={(session.user_id, session.session_id): _working(session)},
    )
    engine = ToolExecutionEngine()
    suite.register_into(engine)
    context = _context(session)

    retrieval = engine.dispatch(
        _request(
            tool_id="retrieval_plan",
            params={"query": "Plan and execute with tools until the checkpoint workflow is verified."},
            context=context,
        )
    )
    memory_read = engine.dispatch(
        _request(
            tool_id="memory_read",
            params={"query": "postgres", "layer": "semantic", "top_k": 3},
            context=context,
        )
    )
    compiled = engine.dispatch(
        _request(
            tool_id="data_compile_context",
            params={"query": "What do you remember about my database preference?"},
            context=context,
        )
    )
    memory_write = engine.dispatch(
        _request(
            tool_id="memory_write",
            params={
                "content": "Global rule: always verify tool auth before mutation.",
                "target_layer": "semantic",
                "target_scope": "global",
                "memory_type": "fact",
                "confidence": 0.9,
            },
            context=context,
            idempotency_key="memory-write-1",
        )
    )
    inspect = engine.dispatch(_request(tool_id="memory_inspect", params={}, context=context))
    registry_view = engine.dispatch(
        _request(
            tool_id="tool_registry_list",
            params={"task_hint": "browser screenshot tool", "token_budget": 256},
            context=context,
        )
    )
    browser = engine.dispatch(
        _request(
            tool_id="browser_navigate",
            params={"url": "https://example.com/tools", "capture_screenshot": True},
            context=context,
        )
    )
    vision = engine.dispatch(
        _request(
            tool_id="vision_describe",
            params={"image_ref": "memory://semantic/postgres", "prompt": "Describe the memory artifact."},
            context=context,
        )
    )
    delegate = engine.dispatch(
        _request(
            tool_id="a2a_delegate",
            params={"agent_name": "critic", "task": "Verify the tool auth policy."},
            context=context,
            idempotency_key="delegate-1",
        )
    )

    assert retrieval.success.data["query_type"] == "agentic"
    assert memory_read.success.data["items"]
    assert not any("Foreign session memory" in item["content"] for item in memory_read.success.data["items"])
    assert any("[Memory Context]" in str(message.get("content")) for message in compiled.success.data["messages"])
    assert memory_write.success.data["scope"] == "global"
    assert inspect.success.data["counts_by_tier"]["semantic"] >= 2
    assert any(tool["tool_id"] == "browser_navigate" for tool in registry_view.success.data["tools"])
    assert browser.success.data["screenshot_ref"].startswith("browser://")
    assert "Describe the memory artifact." in vision.success.data["description"]
    assert delegate.success.data["state"] == "working"


def test_tool_endpoint_report_and_schema_assets_cover_full_matrix() -> None:
    report = build_agent_tools_endpoint_compatibility_report()
    assert_agent_tools_endpoint_payload_compatibility(report)
    assert report.openai_realtime_request["session"]["tool_choice"] == "auto"
    assert report.jsonrpc_invoke_request["method"] == "tools.invoke"
    assert report.mcp_call_tool_request["method"] == "tools/call"
    assert report.a2a_task_request["metadata"]["tool_id"] == "a2a_delegate"
    assert report.retrieval_report.vllm_score_request["model"] == "BAAI/bge-reranker-base"
    assert "tool_registry" in read_schema_sql()
