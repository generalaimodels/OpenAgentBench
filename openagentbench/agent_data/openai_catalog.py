"""OpenAI endpoint catalog for multimodal compatibility and replay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class OpenAIEndpointSpec:
    name: str
    path: str
    python_sdk_call: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    supports_mixed_content: bool
    notes: str
    source_url: str


OPENAI_ENDPOINTS: tuple[OpenAIEndpointSpec, ...] = (
    OpenAIEndpointSpec(
        name="completions",
        path="/v1/completions",
        python_sdk_call="client.completions.create(...)",
        input_modalities=("text",),
        output_modalities=("text", "structured_data"),
        supports_mixed_content=False,
        notes="Legacy text completion surface that remains relevant for compatibility checks and migration audits.",
        source_url="https://platform.openai.com/docs/api-reference/completions",
    ),
    OpenAIEndpointSpec(
        name="responses",
        path="/v1/responses",
        python_sdk_call="client.responses.create(...)",
        input_modalities=("text", "image", "audio", "file", "mixed"),
        output_modalities=("text", "tool_calls", "structured_data"),
        supports_mixed_content=True,
        notes="Primary unified multimodal endpoint for stateful tool-using workflows.",
        source_url="https://developers.openai.com/api/docs/models",
    ),
    OpenAIEndpointSpec(
        name="chat_completions",
        path="/v1/chat/completions",
        python_sdk_call="client.chat.completions.create(...)",
        input_modalities=("text", "image", "audio", "mixed"),
        output_modalities=("text", "audio", "tool_calls", "structured_data"),
        supports_mixed_content=True,
        notes="OpenAI message-list API with explicit conversational turns.",
        source_url="https://developers.openai.com/api/docs/models/gpt-5",
    ),
    OpenAIEndpointSpec(
        name="realtime",
        path="/v1/realtime",
        python_sdk_call="OpenAI realtime session via WebRTC/WebSocket/SIP",
        input_modalities=("text", "image", "audio", "mixed"),
        output_modalities=("text", "audio", "events"),
        supports_mixed_content=True,
        notes="Low-latency streaming interface for speech and multimodal sessions.",
        source_url="https://developers.openai.com/api/docs/models/all",
    ),
    OpenAIEndpointSpec(
        name="audio_speech",
        path="/v1/audio/speech",
        python_sdk_call="client.audio.speech.create(...)",
        input_modalities=("text",),
        output_modalities=("audio",),
        supports_mixed_content=False,
        notes="Text-to-speech endpoint.",
        source_url="https://developers.openai.com/api/docs/models/all",
    ),
    OpenAIEndpointSpec(
        name="audio_transcriptions",
        path="/v1/audio/transcriptions",
        python_sdk_call="client.audio.transcriptions.create(...)",
        input_modalities=("audio",),
        output_modalities=("text",),
        supports_mixed_content=False,
        notes="Speech-to-text endpoint.",
        source_url="https://developers.openai.com/api/docs/models/all",
    ),
    OpenAIEndpointSpec(
        name="audio_translations",
        path="/v1/audio/translations",
        python_sdk_call="client.audio.translations.create(...)",
        input_modalities=("audio",),
        output_modalities=("text",),
        supports_mixed_content=False,
        notes="Audio translation endpoint.",
        source_url="https://developers.openai.com/api/docs/models/gpt-5",
    ),
    OpenAIEndpointSpec(
        name="images_generations",
        path="/v1/images/generations",
        python_sdk_call="client.images.generate(...)",
        input_modalities=("text", "image"),
        output_modalities=("image",),
        supports_mixed_content=True,
        notes="Text-to-image and image-conditioned generation.",
        source_url="https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=file",
    ),
    OpenAIEndpointSpec(
        name="images_edits",
        path="/v1/images/edits",
        python_sdk_call="client.images.edit(...)",
        input_modalities=("text", "image", "mask"),
        output_modalities=("image",),
        supports_mixed_content=True,
        notes="Image editing endpoint with masks and prompt guidance.",
        source_url="https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=file",
    ),
    OpenAIEndpointSpec(
        name="videos",
        path="/v1/videos",
        python_sdk_call="client.videos.generate(...)",
        input_modalities=("text", "image", "mixed"),
        output_modalities=("video", "audio"),
        supports_mixed_content=True,
        notes="Video generation endpoint for Sora-class models.",
        source_url="https://developers.openai.com/api/docs/models/sora-2",
    ),
    OpenAIEndpointSpec(
        name="embeddings",
        path="/v1/embeddings",
        python_sdk_call="client.embeddings.create(...)",
        input_modalities=("text",),
        output_modalities=("embedding",),
        supports_mixed_content=False,
        notes="Text embeddings endpoint for retrieval and ranking.",
        source_url="https://developers.openai.com/api/docs/models/text-embedding-3-small",
    ),
    OpenAIEndpointSpec(
        name="moderations",
        path="/v1/moderations",
        python_sdk_call="client.moderations.create(...)",
        input_modalities=("text", "image", "mixed"),
        output_modalities=("structured_data", "safety_labels"),
        supports_mixed_content=True,
        notes="Safety classification endpoint for text and image policy checks.",
        source_url="https://platform.openai.com/docs/api-reference/moderations",
    ),
)


def endpoints_for_modality(modality: str) -> list[OpenAIEndpointSpec]:
    normalized = modality.strip().lower()
    return [
        endpoint
        for endpoint in OPENAI_ENDPOINTS
        if normalized in endpoint.input_modalities or normalized in endpoint.output_modalities
    ]
