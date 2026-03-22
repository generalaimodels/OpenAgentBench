from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

from openagentbench.agent_retrieval import (
    AuthorityTier,
    HistoryEntry,
    HistoryEvidence,
    HumanFeedback,
    HybridRetrievalEngine,
    InMemoryRetrievalRepository,
    LoopStrategy,
    MemoryEntry,
    MemoryType,
    ModelExecutionMode,
    ModelRole,
    ModelRouter,
    Modality,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleTextModel,
    OutputStream,
    ProtocolType,
    QueryType,
    ReasoningEffort,
    Role,
    SessionTurn,
    SignalTopology,
    SourceTable,
    TaskOutcome,
    assert_endpoint_payload_compatibility,
    build_endpoint_compatibility_report,
    build_openai_audio_speech_request,
    build_openai_audio_transcription_request,
    build_openai_audio_translation_request,
    build_exact_history_retrieval,
    build_gemini_count_tokens_request,
    build_gemini_generate_content_request,
    build_insert_history_entry,
    build_insert_session_turn,
    build_load_memory_summary,
    build_load_session_context,
    build_openai_chat_completions_request,
    build_openai_completions_request,
    build_openai_embeddings_request,
    build_openai_image_edit_request,
    build_openai_image_generation_request,
    build_openai_moderations_request,
    build_openai_realtime_session_request,
    build_openai_responses_request,
    build_openai_video_generation_request,
    build_vllm_chat_completions_request,
    build_vllm_completions_request,
    build_vllm_embeddings_request,
    build_vllm_realtime_session_request,
    build_vllm_responses_request,
    build_vllm_score_request,
    build_semantic_retrieval,
    build_touch_memory_access,
    build_upsert_memory_entry,
    build_verify_user_active,
    classify_query,
    default_profiles,
    extract_gemini_text,
    module_root,
    read_plan,
    read_schema_sql,
    run_gemini_smoke,
)
from openagentbench.agent_retrieval.models import FragmentLocator


@dataclass(frozen=True, slots=True)
class _SuccessExperiment:
    name: str
    query: str
    expected_type: QueryType
    expected_execution_mode: ModelExecutionMode
    expected_signal_topology: SignalTopology
    expected_reasoning_effort: ReasoningEffort
    expected_loop_strategy: LoopStrategy
    required_roles: tuple[ModelRole, ...]
    required_streams: tuple[OutputStream, ...]
    expected_primary_role: ModelRole | None = None


def _sample_rows() -> tuple[SessionTurn, HistoryEntry, MemoryEntry]:
    now = datetime(2026, 3, 22, 12, 0, 0, tzinfo=timezone.utc)
    uu_id = uuid4()
    session_id = uuid4()
    memory_id = uuid4()

    session_turn = SessionTurn(
        session_id=session_id,
        uu_id=uu_id,
        turn_index=0,
        role=Role.USER,
        content_text="Inspect the retrieval endpoint behavior for JSON-RPC and OpenAI compatibility.",
        created_at=now,
        tokens_used=12,
        metadata={"protocol_type": ProtocolType.JSON_RPC.value, "modality": Modality.RUNTIME.value},
        expires_at=now + timedelta(days=1),
    )
    memory_entry = MemoryEntry(
        memory_id=memory_id,
        uu_id=uu_id,
        memory_type=MemoryType.PROCEDURE,
        content_text="Use the OpenAI-compatible endpoint for embeddings and a local reranker for vLLM.",
        content_embedding=(0.1, 0.2, 0.3),
        authority_tier=AuthorityTier.CANONICAL,
        confidence=0.95,
        source_provenance={"source": "review"},
        verified_by=(),
        supersedes=(),
        created_at=now - timedelta(hours=1),
        updated_at=now,
        expires_at=now + timedelta(days=30),
        access_count=2,
        last_accessed_at=now,
        content_hash=b"memory-entry",
        metadata={"protocol_type": ProtocolType.HTTP.value, "modality": Modality.TEXT.value},
    )
    history_entry = HistoryEntry(
        history_id=uuid4(),
        uu_id=uu_id,
        query_text="Which endpoint should the reranker use?",
        query_embedding=(0.2, 0.1, 0.4),
        response_summary="Use an OpenAI-compatible serving layer.",
        evidence_used=(
            HistoryEvidence(
                locator=FragmentLocator(SourceTable.MEMORY, memory_id),
                utility_score=0.9,
                was_cited=True,
            ),
        ),
        task_outcome=TaskOutcome.SUCCESS,
        human_feedback=HumanFeedback.APPROVED,
        utility_score=0.92,
        negative_flag=False,
        tags=("endpoint", "reranker"),
        metadata={"protocol_type": ProtocolType.HTTP.value},
        created_at=now,
        session_origin=session_id,
    )
    return session_turn, history_entry, memory_entry


class RetrievalApiBuilderTests(unittest.TestCase):
    def test_openai_payload_builders_emit_expected_shapes(self) -> None:
        responses_payload = build_openai_responses_request(
            model="gpt-5-mini",
            system_prompt="You are a retrieval planner.",
            user_input="Summarize the endpoint choice.",
            context=("Use OpenAI embeddings.", "Use vLLM for reranking."),
            reasoning_effort="medium",
        )
        chat_payload = build_openai_chat_completions_request(
            model="gpt-4.1-mini",
            system_prompt="You are a retrieval planner.",
            user_input="Summarize the endpoint choice.",
            context=("Use OpenAI embeddings.",),
        )
        completion_payload = build_openai_completions_request(
            model="gpt-3.5-turbo-instruct",
            prompt="Summarize the endpoint choice.",
        )
        embedding_payload = build_openai_embeddings_request(
            model="text-embedding-3-large",
            inputs=("retrieval", "reranking"),
            dimensions=256,
        )
        moderation_payload = build_openai_moderations_request(
            model="omni-moderation-latest",
            inputs=("Summarize the endpoint choice.",),
        )

        self.assertEqual(responses_payload["input"][0]["role"], "system")
        self.assertEqual(responses_payload["reasoning"]["effort"], "medium")
        self.assertEqual(chat_payload["messages"][0]["role"], "system")
        self.assertEqual(completion_payload["prompt"], "Summarize the endpoint choice.")
        self.assertEqual(embedding_payload["dimensions"], 256)
        self.assertEqual(embedding_payload["input"], ["retrieval", "reranking"])
        self.assertEqual(moderation_payload["input"], ["Summarize the endpoint choice."])

    def test_vllm_payload_builders_emit_openai_compatible_shapes(self) -> None:
        responses_payload = build_vllm_responses_request(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            system_prompt="You are a retrieval planner.",
            user_input="Summarize the endpoint choice.",
            context=("Use OpenAI embeddings.", "Use vLLM for reranking."),
            reasoning_effort="medium",
        )
        chat_payload = build_vllm_chat_completions_request(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            system_prompt="You are a retrieval planner.",
            user_input="Summarize the endpoint choice.",
            context=("Use OpenAI embeddings.",),
        )
        completions_payload = build_vllm_completions_request(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            prompt="Summarize the endpoint choice.",
        )
        embeddings_payload = build_vllm_embeddings_request(
            model="intfloat/e5-small-v2",
            inputs=("retrieval", "reranking"),
        )
        realtime_payload = build_vllm_realtime_session_request(
            model="openai/whisper-small",
            modalities=("audio",),
            instructions="Stay concise.",
        )
        score_payload = build_vllm_score_request(
            model="BAAI/bge-reranker-base",
            text_1="retrieval",
            text_2=("reranking", "vision"),
        )

        self.assertEqual(responses_payload["input"][0]["role"], "system")
        self.assertEqual(chat_payload["messages"][0]["role"], "system")
        self.assertEqual(completions_payload["prompt"], "Summarize the endpoint choice.")
        self.assertEqual(embeddings_payload["input"], ["retrieval", "reranking"])
        self.assertEqual(realtime_payload["type"], "session.update")
        self.assertEqual(score_payload["text_1"], "retrieval")
        self.assertEqual(score_payload["text_2"], ["reranking", "vision"])

    def test_full_openai_endpoint_matrix_payload_builders_emit_expected_shapes(self) -> None:
        realtime_payload = build_openai_realtime_session_request(
            model="gpt-realtime-mini",
            modalities=("text", "audio"),
            instructions="Stay concise.",
        )
        speech_payload = build_openai_audio_speech_request(
            model="gpt-4o-mini-tts",
            input_text="Speak this.",
        )
        transcription_payload = build_openai_audio_transcription_request(
            model="gpt-4o-mini-transcribe",
            file_name="sample.wav",
        )
        translation_payload = build_openai_audio_translation_request(
            model="gpt-4o-mini-transcribe",
            file_name="sample.wav",
        )
        image_generation_payload = build_openai_image_generation_request(
            model="gpt-image-1",
            prompt="Create an architecture diagram.",
        )
        image_edit_payload = build_openai_image_edit_request(
            model="gpt-image-1",
            prompt="Remove the background.",
            image_name="input.png",
            mask_name="mask.png",
        )
        video_generation_payload = build_openai_video_generation_request(
            model="sora-2",
            prompt="Animate the uploaded image.",
            image_name="input.png",
        )

        self.assertEqual(realtime_payload["type"], "session.update")
        self.assertEqual(realtime_payload["session"]["modalities"], ["text", "audio"])
        self.assertEqual(speech_payload["input"], "Speak this.")
        self.assertEqual(transcription_payload["file"], "sample.wav")
        self.assertEqual(translation_payload["file"], "sample.wav")
        self.assertEqual(image_generation_payload["prompt"], "Create an architecture diagram.")
        self.assertEqual(image_edit_payload["image"], "input.png")
        self.assertEqual(image_edit_payload["mask"], "mask.png")
        self.assertEqual(video_generation_payload["image"], "input.png")

    def test_gemini_payload_builders_and_extract_text_roundtrip(self) -> None:
        generate_payload = build_gemini_generate_content_request(
            system_instruction="You are a smoke test.",
            user_text="Return one line.",
        )
        count_payload = build_gemini_count_tokens_request(
            system_instruction="You are a smoke test.",
            user_text="Return one line.",
        )
        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "OK "},
                            {"text": "READY"},
                        ]
                    }
                }
            ]
        }

        self.assertIn("contents", generate_payload)
        self.assertIn("generationConfig", generate_payload)
        self.assertIn("contents", count_payload)
        self.assertEqual(extract_gemini_text(response_payload), "OK READY")

    def test_endpoint_compatibility_report_passes_validation(self) -> None:
        report = build_endpoint_compatibility_report()
        assert_endpoint_payload_compatibility(report)
        self.assertIn("input", report.openai_responses_request)
        self.assertIn("messages", report.openai_chat_request)
        self.assertIn("prompt", report.openai_completions_request)
        self.assertEqual(report.openai_realtime_request["type"], "session.update")
        self.assertIn("input", report.openai_moderations_request)
        self.assertIn("input", report.openai_audio_speech_request)
        self.assertIn("file", report.openai_audio_transcription_request)
        self.assertIn("file", report.openai_audio_translation_request)
        self.assertIn("prompt", report.openai_image_generation_request)
        self.assertIn("image", report.openai_image_edit_request)
        self.assertIn("prompt", report.openai_video_generation_request)
        self.assertIn("input", report.vllm_responses_request)
        self.assertIn("messages", report.vllm_chat_request)
        self.assertIn("prompt", report.vllm_completions_request)
        self.assertEqual(report.vllm_realtime_request["type"], "session.update")
        self.assertIn("text_1", report.vllm_score_request)
        self.assertIn("contents", report.gemini_generate_content_request)

    def test_endpoint_compatibility_report_fails_for_invalid_openai_responses_shape(self) -> None:
        report = build_endpoint_compatibility_report()
        broken = replace(report, openai_responses_request={"input": [{"role": "system"}]})

        with self.assertRaises(AssertionError):
            assert_endpoint_payload_compatibility(broken)

    def test_extract_gemini_text_fails_for_malformed_payload(self) -> None:
        with self.assertRaises(ValueError):
            extract_gemini_text({"candidates": [{"content": {"parts": [{}]}}]})


class RetrievalProviderTests(unittest.TestCase):
    def test_openai_embedding_provider_and_chat_fallback(self) -> None:
        class _EmbeddingsApi:
            @staticmethod
            def create(**_: object) -> object:
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

        class _ChatCompletionsApi:
            @staticmethod
            def create(**_: object) -> object:
                choice = SimpleNamespace(message=SimpleNamespace(content="chat fallback ok"))
                return SimpleNamespace(choices=[choice])

        client = SimpleNamespace(
            embeddings=_EmbeddingsApi(),
            chat=SimpleNamespace(completions=_ChatCompletionsApi()),
        )

        embedding_provider = OpenAICompatibleEmbeddingProvider(client=client, model="text-embedding-3-small")
        text_model = OpenAICompatibleTextModel(
            client=client,
            model="gpt-4.1-mini",
            prefer_responses_api=False,
        )

        self.assertEqual(embedding_provider.embed_batch(["retrieval"]), [(0.1, 0.2, 0.3)])
        self.assertEqual(
            text_model.complete(system_prompt="System", user_input="User", context=("Ctx",)),
            "chat fallback ok",
        )


class RetrievalSqlTemplateTests(unittest.TestCase):
    def test_read_and_write_templates_cover_core_api_surface(self) -> None:
        session_turn, history_entry, memory_entry = _sample_rows()

        session_insert = build_insert_session_turn(session_turn)
        history_insert = build_insert_history_entry(history_entry)
        memory_upsert = build_upsert_memory_entry(memory_entry)
        user_verify = build_verify_user_active(uu_id=session_turn.uu_id)
        session_load = build_load_session_context(uu_id=session_turn.uu_id, session_id=session_turn.session_id, limit=8)
        memory_load = build_load_memory_summary(uu_id=session_turn.uu_id, limit=8)
        history_exact = build_exact_history_retrieval(
            uu_id=session_turn.uu_id,
            query_text="endpoint",
            temporal_scope=None,
            limit=5,
        )
        semantic = build_semantic_retrieval(
            table_name="memory",
            uu_id=session_turn.uu_id,
            vector_column="content_embedding",
            query_embedding=(0.1, 0.2, 0.3),
            temporal_scope=None,
            created_at_column="updated_at",
            limit=5,
        )
        touch = build_touch_memory_access(
            uu_id=session_turn.uu_id,
            memory_ids=(memory_entry.memory_id,),
            accessed_at=memory_entry.updated_at,
        )

        self.assertIn("INSERT INTO agent_retrieval.session", session_insert.sql)
        self.assertIn("INSERT INTO agent_retrieval.history", history_insert.sql)
        self.assertIn("INSERT INTO agent_retrieval.memory", memory_upsert.sql)
        self.assertIn("FROM agent_retrieval.users", user_verify.sql)
        self.assertIn("FROM agent_retrieval.session", session_load.sql)
        self.assertIn("FROM agent_retrieval.memory", memory_load.sql)
        self.assertIn("FROM agent_retrieval.history", history_exact.sql)
        self.assertIn("ORDER BY content_embedding <=> %(query_embedding)s::vector", semantic.sql)
        self.assertIn("UPDATE agent_retrieval.memory", touch.sql)
        self.assertEqual(session_insert.params["uu_id"], session_turn.uu_id)
        self.assertEqual(history_insert.params["uu_id"], history_entry.uu_id)
        self.assertEqual(memory_upsert.params["uu_id"], memory_entry.uu_id)
        self.assertEqual(touch.params["memory_ids"], [str(memory_entry.memory_id)])


class RetrievalRuntimeAndRoutingTests(unittest.TestCase):
    def test_runtime_assets_are_readable(self) -> None:
        self.assertEqual(module_root().name, "agent_retrieval")
        self.assertIn("SOTA+ Hybrid Multi-Tier Retrieval System", read_plan())
        self.assertIn("CREATE SCHEMA IF NOT EXISTS agent_retrieval;", read_schema_sql())

    def test_engine_model_plan_exposes_selective_role_bindings(self) -> None:
        query = "Plan and execute multimodal realtime retrieval, use tools, then verify the result with JSON output."
        classification = classify_query(query, "", turn_count=4)
        repo = InMemoryRetrievalRepository(active_users={uuid4()})
        engine = HybridRetrievalEngine(repository=repo)
        plan = engine.plan_models(query, turn_count=4)

        self.assertEqual(plan.query_classification.type, classification.type)
        self.assertIn(ModelRole.EMBEDDING, plan.role_bindings)
        self.assertIn(ModelRole.GENERATION, plan.role_bindings)
        self.assertIn(ModelRole.PLANNER, plan.role_bindings)
        self.assertIn(ModelRole.EXECUTOR, plan.role_bindings)
        self.assertIsNotNone(plan.primary_model)
        self.assertTrue(any(profile.role is ModelRole.RERANKING for profile in default_profiles()))

    def test_engine_retrieve_raises_for_missing_user(self) -> None:
        engine = HybridRetrievalEngine(repository=InMemoryRetrievalRepository())

        with self.assertRaises(LookupError):
            engine.retrieve(
                "Retrieve the endpoint plan.",
                uu_id=uuid4(),
                session_id=uuid4(),
                token_budget=128,
            )


class RetrievalSuccessExperimentTests(unittest.TestCase):
    def test_success_matrix_covers_normal_mimo_thinking_dual_and_agentic_modes(self) -> None:
        cases = (
            _SuccessExperiment(
                name="normal-single-model",
                query="Answer this normally in one short paragraph.",
                expected_type=QueryType.CONVERSATIONAL,
                expected_execution_mode=ModelExecutionMode.SINGLE_MODEL,
                expected_signal_topology=SignalTopology.SISO,
                expected_reasoning_effort=ReasoningEffort.DIRECT,
                expected_loop_strategy=LoopStrategy.SINGLE_PASS,
                required_roles=(ModelRole.EMBEDDING, ModelRole.GENERATION),
                required_streams=(OutputStream.TEXT_EVIDENCE,),
                expected_primary_role=ModelRole.GENERATION,
            ),
            _SuccessExperiment(
                name="multimodal-mimo",
                query="Use the screenshot, audio clip, and PDF report to produce JSON and a short summary.",
                expected_type=QueryType.MULTIMODAL,
                expected_execution_mode=ModelExecutionMode.SINGLE_MODEL,
                expected_signal_topology=SignalTopology.MIMO,
                expected_reasoning_effort=ReasoningEffort.DIRECT,
                expected_loop_strategy=LoopStrategy.RETRIEVAL_REFINEMENT,
                required_roles=(ModelRole.EMBEDDING, ModelRole.GENERATION, ModelRole.MULTIMODAL),
                required_streams=(
                    OutputStream.VISION_EVIDENCE,
                    OutputStream.STRUCTURED_DATA,
                    OutputStream.TEXT_EVIDENCE,
                ),
                expected_primary_role=ModelRole.MULTIMODAL,
            ),
            _SuccessExperiment(
                name="thinking-reasoning",
                query="Think step by step and compare tradeoffs for this retrieval strategy.",
                expected_type=QueryType.REASONING,
                expected_execution_mode=ModelExecutionMode.SINGLE_MODEL,
                expected_signal_topology=SignalTopology.SISO,
                expected_reasoning_effort=ReasoningEffort.THINKING,
                expected_loop_strategy=LoopStrategy.RETRIEVAL_REFINEMENT,
                required_roles=(ModelRole.EMBEDDING, ModelRole.GENERATION, ModelRole.THINKING),
                required_streams=(OutputStream.TEXT_EVIDENCE,),
                expected_primary_role=ModelRole.THINKING,
            ),
            _SuccessExperiment(
                name="dual-model-reranking",
                query="Use an embedding model and a second rerank stage for the candidates.",
                expected_type=QueryType.CONVERSATIONAL,
                expected_execution_mode=ModelExecutionMode.DUAL_MODEL,
                expected_signal_topology=SignalTopology.SISO,
                expected_reasoning_effort=ReasoningEffort.DIRECT,
                expected_loop_strategy=LoopStrategy.RETRIEVAL_REFINEMENT,
                required_roles=(ModelRole.EMBEDDING, ModelRole.GENERATION, ModelRole.RERANKING),
                required_streams=(OutputStream.TEXT_EVIDENCE,),
                expected_primary_role=ModelRole.GENERATION,
            ),
            _SuccessExperiment(
                name="agentic-loop",
                query="Plan and execute with tools, retry until deployment is verified, and return JSON.",
                expected_type=QueryType.AGENTIC,
                expected_execution_mode=ModelExecutionMode.MULTI_MODEL,
                expected_signal_topology=SignalTopology.MIMO,
                expected_reasoning_effort=ReasoningEffort.DIRECT,
                expected_loop_strategy=LoopStrategy.AGENTIC_LOOP,
                required_roles=(
                    ModelRole.EMBEDDING,
                    ModelRole.GENERATION,
                    ModelRole.PLANNER,
                    ModelRole.EXECUTOR,
                    ModelRole.CRITIC,
                    ModelRole.AGENTIC_LOOP,
                ),
                required_streams=(OutputStream.TOOL_TRACE, OutputStream.STRUCTURED_DATA, OutputStream.RUNTIME_STATE),
                expected_primary_role=ModelRole.GENERATION,
            ),
        )

        router = ModelRouter()
        profiles = default_profiles()

        for case in cases:
            with self.subTest(case=case.name):
                classification = classify_query(case.query, "", turn_count=0)
                plan = router.select(classification, profiles)

                self.assertEqual(classification.type, case.expected_type)
                self.assertEqual(classification.model_execution_mode, case.expected_execution_mode)
                self.assertEqual(classification.signal_topology, case.expected_signal_topology)
                self.assertEqual(classification.reasoning_effort, case.expected_reasoning_effort)
                self.assertEqual(classification.loop_strategy, case.expected_loop_strategy)

                for role in case.required_roles:
                    self.assertIn(role, classification.model_roles)
                    self.assertIn(role, plan.role_bindings)

                for stream in case.required_streams:
                    self.assertIn(stream, classification.output_streams)

                self.assertIsNotNone(plan.primary_model)
                if case.expected_primary_role is not None:
                    self.assertEqual(plan.primary_model.role, case.expected_primary_role)

                if ModelRole.MULTIMODAL in case.required_roles:
                    multimodal_profile = plan.role_bindings[ModelRole.MULTIMODAL]
                    self.assertTrue(
                        multimodal_profile.supports_images
                        or multimodal_profile.supports_documents
                        or multimodal_profile.supports_audio
                    )

                if any(role in case.required_roles for role in (ModelRole.PLANNER, ModelRole.EXECUTOR, ModelRole.CRITIC)):
                    self.assertTrue(plan.role_bindings[ModelRole.PLANNER].supports_tools)
                    self.assertTrue(plan.role_bindings[ModelRole.EXECUTOR].supports_tools)

                if ModelRole.RERANKING in case.required_roles:
                    self.assertEqual(plan.role_bindings[ModelRole.RERANKING].role, ModelRole.RERANKING)


class GeminiSmokeApiTests(unittest.TestCase):
    def test_run_gemini_smoke_requires_api_key(self) -> None:
        original_api_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                env_path = os.path.join(directory, ".env")
                with self.assertRaises(RuntimeError):
                    run_gemini_smoke(env_path=env_path)
        finally:
            if original_api_key is not None:
                os.environ["GEMINI_API_KEY"] = original_api_key


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
