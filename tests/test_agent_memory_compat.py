from __future__ import annotations

from types import SimpleNamespace

from openagentbench.agent_memory import (
    MemoryProviderSuite,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleTextModel,
    assert_memory_endpoint_payload_compatibility,
    build_memory_endpoint_compatibility_report,
)


def test_memory_endpoint_payloads_match_expected_shapes() -> None:
    report = build_memory_endpoint_compatibility_report()
    assert_memory_endpoint_payload_compatibility(report)

    assert report.memory_read_tool["function"]["name"] == "memory_read"
    assert report.memory_write_tool["function"]["name"] == "memory_write"
    assert report.openai_responses_request["input"][0]["role"] == "system"
    assert len(report.openai_chat_request["tools"]) == 3
    assert report.openai_realtime_request["type"] == "session.update"
    assert "file" in report.openai_audio_transcription_request
    assert "image" in report.openai_image_edit_request
    assert "contents" in report.gemini_generate_content_request
    assert "prompt" in report.retrieval_report.openai_completions_request
    assert report.retrieval_report.vllm_chat_request["messages"][0]["role"] == "system"
    assert "text_1" in report.retrieval_report.vllm_score_request
    assert report.retrieval_report.openai_embeddings_request["model"] == "text-embedding-3-small"


def test_memory_provider_suite_supports_openai_compatible_embeddings_and_summaries() -> None:
    class _EmbeddingsAPI:
        @staticmethod
        def create(**_: object) -> object:
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

    class _ResponsesAPI:
        @staticmethod
        def create(**_: object) -> object:
            return SimpleNamespace(output_text="summary preserved with corrections")

    client = SimpleNamespace(embeddings=_EmbeddingsAPI(), responses=_ResponsesAPI())
    suite = MemoryProviderSuite(
        embedding_provider=OpenAICompatibleEmbeddingProvider(client=client, model="text-embedding-3-small"),
        summarizer_model=OpenAICompatibleTextModel(client=client, model="gpt-4.1-mini"),
    )

    assert suite.embed_batch(["memory"]) == [(0.1, 0.2, 0.3)]
    assert suite.summarize(
        existing_summary="",
        additions=("correction: keep local memory isolated",),
        max_tokens=32,
    ) == "summary preserved with corrections"
