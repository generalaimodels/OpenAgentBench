"""vLLM endpoint catalog for OpenAI-compatible and native serving surfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class VLLMEndpointSpec:
    name: str
    path: str
    python_sdk_call: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    supports_mixed_content: bool
    notes: str
    source_url: str


VLLM_ENDPOINTS: tuple[VLLMEndpointSpec, ...] = (
    VLLMEndpointSpec(
        name="completions",
        path="/v1/completions",
        python_sdk_call="client.completions.create(..., base_url='http://host:8000/v1')",
        input_modalities=("text",),
        output_modalities=("text", "structured_data"),
        supports_mixed_content=False,
        notes="OpenAI-compatible text completion surface served by vLLM.",
        source_url="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    ),
    VLLMEndpointSpec(
        name="chat_completions",
        path="/v1/chat/completions",
        python_sdk_call="client.chat.completions.create(..., base_url='http://host:8000/v1')",
        input_modalities=("text", "image", "audio", "mixed"),
        output_modalities=("text", "tool_calls", "structured_data"),
        supports_mixed_content=True,
        notes="OpenAI-compatible chat completion surface with multimodal inputs and tool-use support.",
        source_url="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    ),
    VLLMEndpointSpec(
        name="responses",
        path="/v1/responses",
        python_sdk_call="client.responses.create(..., base_url='http://host:8000/v1')",
        input_modalities=("text", "image", "audio", "file", "mixed"),
        output_modalities=("text", "tool_calls", "structured_data"),
        supports_mixed_content=True,
        notes="OpenAI-compatible Responses API surface for stateful tool and multimodal workflows.",
        source_url="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    ),
    VLLMEndpointSpec(
        name="embeddings",
        path="/v1/embeddings",
        python_sdk_call="client.embeddings.create(..., base_url='http://host:8000/v1')",
        input_modalities=("text",),
        output_modalities=("embedding",),
        supports_mixed_content=False,
        notes="Embedding endpoint for vector retrieval and ranking workloads.",
        source_url="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    ),
    VLLMEndpointSpec(
        name="audio_transcriptions",
        path="/v1/audio/transcriptions",
        python_sdk_call="client.audio.transcriptions.create(..., base_url='http://host:8000/v1')",
        input_modalities=("audio",),
        output_modalities=("text",),
        supports_mixed_content=False,
        notes="OpenAI-compatible transcription endpoint for ASR-capable models.",
        source_url="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    ),
    VLLMEndpointSpec(
        name="audio_translations",
        path="/v1/audio/translations",
        python_sdk_call="client.audio.translations.create(..., base_url='http://host:8000/v1')",
        input_modalities=("audio",),
        output_modalities=("text",),
        supports_mixed_content=False,
        notes="OpenAI-compatible translation endpoint for ASR-capable models.",
        source_url="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    ),
    VLLMEndpointSpec(
        name="realtime",
        path="/v1/realtime",
        python_sdk_call="OpenAI realtime session via WebSocket with base_url='ws://host:8000/v1/realtime'",
        input_modalities=("audio",),
        output_modalities=("text", "events"),
        supports_mixed_content=False,
        notes="Realtime WebSocket endpoint for streaming transcription workflows.",
        source_url="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    ),
    VLLMEndpointSpec(
        name="score",
        path="/score",
        python_sdk_call="POST /score with a scoring model deployment",
        input_modalities=("text", "document"),
        output_modalities=("structured_data", "score"),
        supports_mixed_content=True,
        notes="Native vLLM scoring endpoint for cross-encoder style ranking workloads.",
        source_url="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    ),
)


def vllm_endpoints_for_modality(modality: str) -> list[VLLMEndpointSpec]:
    normalized = modality.strip().lower()
    return [
        endpoint
        for endpoint in VLLM_ENDPOINTS
        if normalized in endpoint.input_modalities or normalized in endpoint.output_modalities
    ]


__all__ = ["VLLM_ENDPOINTS", "VLLMEndpointSpec", "vllm_endpoints_for_modality"]
