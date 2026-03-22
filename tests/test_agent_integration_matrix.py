from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import agent_data.runtime as agent_data_runtime
from openagentbench.agent_data import (
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
    hash_normalized_text,
)
from openagentbench.agent_memory import (
    MemoryCompileRequest,
    MemoryContextCompiler,
    MemoryProviderSuite,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleTextModel,
    WorkingMemoryItem,
    assert_memory_endpoint_payload_compatibility,
    build_load_durable_memories,
    build_load_working_memory,
    build_memory_endpoint_compatibility_report,
    filter_scoped_memories,
    read_schema_sql as read_memory_schema_sql,
    schema_sql_path as memory_schema_sql_path,
)
from openagentbench.agent_retrieval import (
    LoopStrategy,
    ModelRole,
    ModelRouter,
    Modality,
    OutputStream,
    QueryType,
    ReasoningEffort,
    assert_endpoint_payload_compatibility,
    build_endpoint_compatibility_report,
    build_exact_memory_retrieval,
    build_exact_session_retrieval,
    build_verify_user_active,
    classify_query,
    default_profiles,
    read_schema_sql as read_retrieval_schema_sql,
    schema_sql_path as retrieval_schema_sql_path,
)


@dataclass(frozen=True, slots=True)
class _Scenario:
    name: str
    query: str
    expected_type: QueryType
    expected_loop: LoopStrategy
    expected_reasoning: ReasoningEffort
    expect_multimodal_refs: bool = False
    expect_thinking_role: bool = False
    expect_agentic_roles: bool = False


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
        context_window_size=20_000,
        system_prompt_hash=hash_normalized_text("You are an integration test agent."),
        system_prompt_tokens=14,
        max_response_tokens=1_200,
        turn_count=4,
        summary_text="The user is building an advanced memory stack with retrieval and OpenAI-compatible integrations.",
        summary_token_count=18,
        system_prompt_text="You are an integration test agent.",
    )


def _history(session: SessionRecord, *, multimodal: bool) -> list[HistoryRecord]:
    now = datetime(2026, 3, 22, 18, 10, 0, tzinfo=timezone.utc)
    user_content_parts = None
    user_content = "Please keep session memory and global memory coordinated."
    if multimodal:
        user_content = None
        user_content_parts = (
            {"type": "input_text", "text": "Use this screenshot and audio note to refine memory context."},
            {"type": "input_image", "image_url": "https://example.com/diagram.png"},
            {"type": "input_audio", "audio_url": "https://example.com/note.wav"},
        )
    return [
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=1,
            role=MessageRole.USER,
            content=user_content,
            content_parts=user_content_parts,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("Use multimodal input for memory context." if multimodal else user_content or ""),
            token_count=26 if multimodal else 14,
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
            content="Acknowledged. I will preserve session decisions and global rules separately.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text(
                "Acknowledged. I will preserve session decisions and global rules separately."
            ),
            token_count=13,
            model_id="gpt-4o-mini",
            finish_reason=None,
            prompt_tokens=120,
            completion_tokens=13,
            latency_ms=120,
            api_call_id=None,
            created_at=now - timedelta(minutes=1),
        ),
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
            content_text="Session memory: checkpoint before summary compression.",
            content_embedding=None,
            content_hash=hash_normalized_text("Session memory: checkpoint before summary compression."),
            provenance_type=ProvenanceType.INSTRUCTION,
            provenance_turn_id=None,
            confidence=0.96,
            relevance_accumulator=2.0,
            access_count=3,
            last_accessed_at=now - timedelta(minutes=15),
            created_at=now - timedelta(hours=1),
            updated_at=now - timedelta(minutes=15),
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=8,
            tags=("session", "checkpoint"),
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            memory_tier=MemoryTier.EPISODIC,
            memory_scope=MemoryScope.LOCAL,
            content_text="Local episode: the user corrected cache invalidation after promotion.",
            content_embedding=None,
            content_hash=hash_normalized_text("Local episode: the user corrected cache invalidation after promotion."),
            provenance_type=ProvenanceType.CORRECTION,
            provenance_turn_id=None,
            confidence=0.91,
            relevance_accumulator=2.2,
            access_count=4,
            last_accessed_at=now - timedelta(minutes=10),
            created_at=now - timedelta(hours=2),
            updated_at=now - timedelta(minutes=10),
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=10,
            tags=("local", "correction"),
            metadata={"outcome_score": 0.94, "human_feedback_score": 0.2},
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=None,
            memory_tier=MemoryTier.SEMANTIC,
            memory_scope=MemoryScope.GLOBAL,
            content_text="Global semantic rule: PostgreSQL remains the durable source of truth for memory persistence.",
            content_embedding=None,
            content_hash=hash_normalized_text(
                "Global semantic rule: PostgreSQL remains the durable source of truth for memory persistence."
            ),
            provenance_type=ProvenanceType.FACT,
            provenance_turn_id=None,
            confidence=0.98,
            relevance_accumulator=5.0,
            access_count=10,
            last_accessed_at=now - timedelta(hours=3),
            created_at=now - timedelta(days=10),
            updated_at=now - timedelta(hours=3),
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=12,
            tags=("global", "semantic"),
            metadata={"modality_ref": "memory://semantic/postgres"},
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=None,
            memory_tier=MemoryTier.PROCEDURAL,
            memory_scope=MemoryScope.GLOBAL,
            content_text="Global procedure: checkpoint, summarize, promote, and invalidate cache in that order.",
            content_embedding=None,
            content_hash=hash_normalized_text(
                "Global procedure: checkpoint, summarize, promote, and invalidate cache in that order."
            ),
            provenance_type=ProvenanceType.INSTRUCTION,
            provenance_turn_id=None,
            confidence=0.99,
            relevance_accumulator=4.5,
            access_count=11,
            last_accessed_at=now - timedelta(hours=1),
            created_at=now - timedelta(days=8),
            updated_at=now - timedelta(hours=1),
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=12,
            tags=("global", "procedure"),
            metadata={"modality_ref": "memory://procedure/checkpoint"},
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=foreign_session_id,
            memory_tier=MemoryTier.SESSION,
            memory_scope=MemoryScope.LOCAL,
            content_text="Foreign local memory that must not leak into the active session.",
            content_embedding=None,
            content_hash=hash_normalized_text("Foreign local memory that must not leak into the active session."),
            provenance_type=ProvenanceType.USER_STATED,
            provenance_turn_id=None,
            confidence=1.0,
            relevance_accumulator=100.0,
            access_count=100,
            last_accessed_at=now,
            created_at=now - timedelta(minutes=1),
            updated_at=now,
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=9,
            tags=("forbidden",),
        ),
    ]


def _working_items(session: SessionRecord, *, multimodal: bool) -> tuple[WorkingMemoryItem, ...]:
    items = [
        WorkingMemoryItem(
            item_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            step_id=uuid4(),
            content_text="Working note: keep the active session and global memory hierarchy aligned.",
            token_count=12,
            modality=Modality.TEXT,
            dependency_count=2,
            carry_forward=True,
        )
    ]
    if multimodal:
        items.append(
            WorkingMemoryItem(
                item_id=uuid4(),
                user_id=session.user_id,
                session_id=session.session_id,
                step_id=uuid4(),
                content_text="Image reference for current multimodal memory task.",
                token_count=18,
                modality=Modality.IMAGE,
                binary_ref="https://example.com/diagram.png",
            )
        )
    return tuple(items)


def _assert_openai_message_array(test_case: unittest.TestCase, messages: list[dict[str, object]]) -> None:
    valid_roles = {"system", "user", "assistant", "tool", "function"}
    test_case.assertTrue(messages)
    for message in messages:
        test_case.assertIn(message["role"], valid_roles)
        content = message.get("content")
        test_case.assertTrue(isinstance(content, (str, list)) or content is None)


class CrossModuleIntegrationMatrixTests(unittest.TestCase):
    def test_cross_module_scenario_matrix(self) -> None:
        scenarios = (
            _Scenario(
                name="normal-session-heavy",
                query="Answer this normally in one short paragraph about what this session should remember for checkpointing and cache invalidation.",
                expected_type=QueryType.CONVERSATIONAL,
                expected_loop=LoopStrategy.RETRIEVAL_REFINEMENT,
                expected_reasoning=ReasoningEffort.DIRECT,
            ),
            _Scenario(
                name="multimodal-mimo",
                query="Use the screenshot and audio note to produce JSON and a summary of the memory state.",
                expected_type=QueryType.MULTIMODAL,
                expected_loop=LoopStrategy.RETRIEVAL_REFINEMENT,
                expected_reasoning=ReasoningEffort.DIRECT,
                expect_multimodal_refs=True,
            ),
            _Scenario(
                name="thinking-reasoning",
                query="Think step by step about the tradeoffs of local versus global memory retention.",
                expected_type=QueryType.REASONING,
                expected_loop=LoopStrategy.SINGLE_PASS,
                expected_reasoning=ReasoningEffort.THINKING,
                expect_thinking_role=True,
            ),
            _Scenario(
                name="agentic-loop",
                query="Plan and execute with tools, verify checkpoint ordering, and retry until the procedure is validated.",
                expected_type=QueryType.AGENTIC,
                expected_loop=LoopStrategy.AGENTIC_LOOP,
                expected_reasoning=ReasoningEffort.DIRECT,
                expect_agentic_roles=True,
            ),
        )

        for scenario in scenarios:
            with self.subTest(scenario=scenario.name):
                session = _session()
                history = _history(session, multimodal=scenario.expect_multimodal_refs)
                raw_memories = _memories(session)
                scoped_memories = filter_scoped_memories(memories=raw_memories, session_id=session.session_id)
                working_items = _working_items(session, multimodal=scenario.expect_multimodal_refs)

                classification = classify_query(
                    scenario.query,
                    session.summary_text or "",
                    turn_count=session.turn_count,
                )
                self.assertEqual(classification.type, scenario.expected_type)
                self.assertEqual(classification.loop_strategy, scenario.expected_loop)
                self.assertEqual(classification.reasoning_effort, scenario.expected_reasoning)

                plan = ModelRouter().select(classification, default_profiles())
                self.assertIsNotNone(plan.primary_model)

                memory_context = MemoryContextCompiler().compile_context(
                    MemoryCompileRequest(
                        user_id=session.user_id,
                        session=session,
                        query_text=scenario.query,
                        total_budget=900,
                        classification=classification,
                        model_plan=plan,
                    ),
                    memories=raw_memories,
                    working_items=working_items,
                )
                data_context = ContextCompiler().compile_context(
                    CompileRequest(
                        user_id=session.user_id,
                        session=session,
                        query_text=scenario.query,
                    ),
                    history=history,
                    memories=scoped_memories,
                )

                combined_messages = list(memory_context.messages)
                combined_messages.extend(
                    message
                    for message in data_context.messages
                    if not (
                        message.get("role") == "system"
                        and message.get("content") == session.system_prompt_text
                    )
                )

                _assert_openai_message_array(self, combined_messages)
                combined_text = "\n".join(
                    message["content"]
                    for message in combined_messages
                    if isinstance(message.get("content"), str)
                )
                self.assertIn("[Session Memory]", combined_text)
                self.assertIn("[Global Semantic Memory]", combined_text)
                self.assertNotIn("Foreign local memory that must not leak", combined_text)

                if scenario.expect_multimodal_refs:
                    self.assertIn("[Multimodal References]", combined_text)
                    self.assertTrue(any(isinstance(message.get("content"), list) for message in data_context.messages))
                    self.assertIn(ModelRole.MULTIMODAL, plan.role_bindings)
                    self.assertIn(OutputStream.STRUCTURED_DATA, classification.output_streams)

                if scenario.expect_thinking_role:
                    self.assertIn(ModelRole.THINKING, plan.role_bindings)

                if scenario.expect_agentic_roles:
                    self.assertIn(ModelRole.PLANNER, plan.role_bindings)
                    self.assertIn(ModelRole.EXECUTOR, plan.role_bindings)
                    self.assertIn(ModelRole.CRITIC, plan.role_bindings)
                    self.assertIn("[Global Procedures]", combined_text)

    def test_api_sql_and_runtime_assets_align_across_all_modules(self) -> None:
        retrieval_report = build_endpoint_compatibility_report()
        memory_report = build_memory_endpoint_compatibility_report()
        assert_endpoint_payload_compatibility(retrieval_report)
        assert_memory_endpoint_payload_compatibility(memory_report)

        user_id = uuid4()
        session_id = uuid4()
        retrieval_verify = build_verify_user_active(uu_id=user_id)
        retrieval_session_sql = build_exact_session_retrieval(
            uu_id=user_id,
            query_text="checkpoint",
            temporal_scope=None,
            limit=5,
        )
        retrieval_memory_sql = build_exact_memory_retrieval(
            uu_id=user_id,
            query_text="postgres",
            temporal_scope=None,
            limit=5,
        )
        memory_scope_sql = build_load_durable_memories(
            user_id=user_id,
            session_id=session_id,
            tiers=(MemoryTier.SESSION, MemoryTier.SEMANTIC, MemoryTier.PROCEDURAL),
        )
        working_sql = build_load_working_memory(user_id=user_id, session_id=session_id)

        self.assertIn("uu_id = %(uu_id)s", retrieval_verify.sql)
        self.assertIn("uu_id = %(uu_id)s", retrieval_session_sql.sql)
        self.assertIn("uu_id = %(uu_id)s", retrieval_memory_sql.sql)
        self.assertIn("user_id = %(user_id)s", memory_scope_sql.sql)
        self.assertIn("memory_scope", memory_scope_sql.sql)
        self.assertIn("session_id = %(session_id)s", working_sql.sql)

        self.assertTrue(agent_data_runtime.schema_sql_path().exists())
        self.assertTrue(retrieval_schema_sql_path().exists())
        self.assertTrue(memory_schema_sql_path().exists())
        self.assertIn("CREATE TABLE IF NOT EXISTS agent_data.sessions", agent_data_runtime.read_schema_sql())
        self.assertIn("agent_retrieval", read_retrieval_schema_sql())
        self.assertIn("session_checkpoints", read_memory_schema_sql())

        self.assertEqual(memory_report.memory_read_tool["function"]["name"], "memory_read")
        self.assertEqual(len(memory_report.openai_responses_request["tools"]), 3)
        self.assertEqual(memory_report.retrieval_report.openai_realtime_request["type"], "session.update")

    def test_openai_compatible_provider_suite_roundtrips_for_memory_and_retrieval_use(self) -> None:
        class _EmbeddingsApi:
            @staticmethod
            def create(**_: object) -> object:
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

        class _ResponsesApi:
            @staticmethod
            def create(**_: object) -> object:
                return SimpleNamespace(output_text="checkpoint then summarize then invalidate cache")

        client = SimpleNamespace(embeddings=_EmbeddingsApi(), responses=_ResponsesApi())
        suite = MemoryProviderSuite(
            embedding_provider=OpenAICompatibleEmbeddingProvider(client=client, model="text-embedding-3-small"),
            summarizer_model=OpenAICompatibleTextModel(client=client, model="gpt-4.1-mini"),
            procedure_model=OpenAICompatibleTextModel(client=client, model="gpt-4.1-mini"),
        )

        self.assertEqual(suite.embed_batch(["memory flow"]), [(0.1, 0.2, 0.3)])
        self.assertEqual(
            suite.summarize(
                existing_summary="Existing summary.",
                additions=("Correction: isolate local session memory.",),
                max_tokens=32,
            ),
            "checkpoint then summarize then invalidate cache",
        )
        self.assertEqual(
            suite.synthesize_procedure(
                ("step one: checkpoint", "step two: summarize", "step three: invalidate"),
                max_tokens=48,
            ),
            "checkpoint then summarize then invalidate cache",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
