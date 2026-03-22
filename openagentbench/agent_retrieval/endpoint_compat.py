"""Endpoint compatibility helpers for OpenAI-style and Gemini-style retrieval clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def _filtered_context(context: Sequence[str]) -> list[str]:
    return [item.strip() for item in context if item.strip()]


def build_openai_responses_request(
    *,
    model: str,
    system_prompt: str,
    user_input: str,
    context: Sequence[str] = (),
    max_output_tokens: int = 256,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    user_content: list[dict[str, str]] = []
    filtered_context = _filtered_context(context)
    if filtered_context:
        user_content.append({"type": "input_text", "text": "Context:\n" + "\n".join(filtered_context)})
    user_content.append({"type": "input_text", "text": user_input})

    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ],
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }
    if reasoning_effort is not None and reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def build_openai_chat_completions_request(
    *,
    model: str,
    system_prompt: str,
    user_input: str,
    context: Sequence[str] = (),
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    filtered_context = _filtered_context(context)
    if filtered_context:
        messages.append({"role": "system", "content": "Context:\n" + "\n".join(filtered_context)})
    messages.append({"role": "user", "content": user_input})
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def build_openai_embeddings_request(
    *,
    model: str,
    inputs: Sequence[str],
    dimensions: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "input": list(inputs)}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return payload


def build_openai_realtime_session_request(
    *,
    model: str,
    modalities: Sequence[str] = ("text",),
    instructions: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "session.update",
        "session": {
            "model": model,
            "modalities": list(modalities),
        },
    }
    if instructions:
        payload["session"]["instructions"] = instructions
    return payload


def build_openai_audio_speech_request(
    *,
    model: str,
    input_text: str,
    voice: str = "alloy",
    audio_format: str = "mp3",
) -> dict[str, Any]:
    return {
        "model": model,
        "input": input_text,
        "voice": voice,
        "format": audio_format,
    }


def build_openai_audio_transcription_request(
    *,
    model: str,
    file_name: str,
    response_format: str = "json",
) -> dict[str, Any]:
    return {
        "model": model,
        "file": file_name,
        "response_format": response_format,
    }


def build_openai_audio_translation_request(
    *,
    model: str,
    file_name: str,
    response_format: str = "json",
) -> dict[str, Any]:
    return {
        "model": model,
        "file": file_name,
        "response_format": response_format,
    }


def build_openai_image_generation_request(
    *,
    model: str,
    prompt: str,
    size: str = "1024x1024",
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "size": size,
    }


def build_openai_image_edit_request(
    *,
    model: str,
    prompt: str,
    image_name: str,
    mask_name: str | None = None,
    size: str = "1024x1024",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "image": image_name,
        "size": size,
    }
    if mask_name is not None:
        payload["mask"] = mask_name
    return payload


def build_openai_video_generation_request(
    *,
    model: str,
    prompt: str,
    image_name: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
    }
    if image_name is not None:
        payload["image"] = image_name
    return payload


def build_gemini_generate_content_request(
    *,
    system_instruction: str,
    user_text: str,
    temperature: float = 0.0,
    max_output_tokens: int = 128,
) -> dict[str, Any]:
    combined_text = f"System instruction:\n{system_instruction}\n\nUser request:\n{user_text}"
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": combined_text}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }


def build_gemini_count_tokens_request(*, system_instruction: str, user_text: str) -> dict[str, Any]:
    combined_text = f"System instruction:\n{system_instruction}\n\nUser request:\n{user_text}"
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": combined_text}],
            }
        ],
    }


def extract_gemini_text(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response did not include candidates")
    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise ValueError("Gemini candidate payload is malformed")
    content = first_candidate.get("content")
    if not isinstance(content, dict):
        raise ValueError("Gemini candidate content is missing")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Gemini content parts are missing")
    text_parts = [part.get("text") for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
    if not text_parts:
        raise ValueError("Gemini response did not include text parts")
    return "".join(text_parts).strip()


@dataclass(slots=True, frozen=True)
class EndpointCompatibilityReport:
    openai_responses_request: dict[str, Any]
    openai_chat_request: dict[str, Any]
    openai_embeddings_request: dict[str, Any]
    openai_realtime_request: dict[str, Any]
    openai_audio_speech_request: dict[str, Any]
    openai_audio_transcription_request: dict[str, Any]
    openai_audio_translation_request: dict[str, Any]
    openai_image_generation_request: dict[str, Any]
    openai_image_edit_request: dict[str, Any]
    openai_video_generation_request: dict[str, Any]
    gemini_generate_content_request: dict[str, Any]
    gemini_count_tokens_request: dict[str, Any]


def build_endpoint_compatibility_report() -> EndpointCompatibilityReport:
    system_prompt = "You are a retrieval planner."
    user_input = "Summarize the retrieval state in one line."
    context = ("Previous memory: user prefers PostgreSQL.", "Session topic: endpoint compatibility.")
    return EndpointCompatibilityReport(
        openai_responses_request=build_openai_responses_request(
            model="gpt-5-mini",
            system_prompt=system_prompt,
            user_input=user_input,
            context=context,
            reasoning_effort="medium",
        ),
        openai_chat_request=build_openai_chat_completions_request(
            model="gpt-4o-mini",
            system_prompt=system_prompt,
            user_input=user_input,
            context=context,
        ),
        openai_embeddings_request=build_openai_embeddings_request(
            model="text-embedding-3-small",
            inputs=("endpoint compatibility", "retrieval memory"),
            dimensions=256,
        ),
        openai_realtime_request=build_openai_realtime_session_request(
            model="gpt-realtime-mini",
            modalities=("text", "audio"),
            instructions=system_prompt,
        ),
        openai_audio_speech_request=build_openai_audio_speech_request(
            model="gpt-4o-mini-tts",
            input_text=user_input,
        ),
        openai_audio_transcription_request=build_openai_audio_transcription_request(
            model="gpt-4o-mini-transcribe",
            file_name="sample.wav",
        ),
        openai_audio_translation_request=build_openai_audio_translation_request(
            model="gpt-4o-mini-transcribe",
            file_name="sample.wav",
        ),
        openai_image_generation_request=build_openai_image_generation_request(
            model="gpt-image-1",
            prompt=user_input,
        ),
        openai_image_edit_request=build_openai_image_edit_request(
            model="gpt-image-1",
            prompt=user_input,
            image_name="sample.png",
            mask_name="mask.png",
        ),
        openai_video_generation_request=build_openai_video_generation_request(
            model="sora-2",
            prompt=user_input,
            image_name="sample.png",
        ),
        gemini_generate_content_request=build_gemini_generate_content_request(
            system_instruction=system_prompt,
            user_text=user_input,
        ),
        gemini_count_tokens_request=build_gemini_count_tokens_request(
            system_instruction=system_prompt,
            user_text=user_input,
        ),
    )


def assert_endpoint_payload_compatibility(report: EndpointCompatibilityReport | None = None) -> None:
    current = report or build_endpoint_compatibility_report()

    responses_input = current.openai_responses_request["input"]
    if not isinstance(responses_input, list) or len(responses_input) != 2:
        raise AssertionError("OpenAI Responses payload must contain system and user items")
    if current.openai_chat_request["messages"][0]["role"] != "system":
        raise AssertionError("OpenAI Chat payload must start with a system message")
    if "input" not in current.openai_embeddings_request or not current.openai_embeddings_request["input"]:
        raise AssertionError("OpenAI embeddings payload must include at least one input string")
    if current.openai_realtime_request.get("type") != "session.update":
        raise AssertionError("OpenAI realtime payload must be a session.update event")
    if "input" not in current.openai_audio_speech_request:
        raise AssertionError("OpenAI audio speech payload must include input text")
    if "file" not in current.openai_audio_transcription_request:
        raise AssertionError("OpenAI audio transcription payload must include a file")
    if "file" not in current.openai_audio_translation_request:
        raise AssertionError("OpenAI audio translation payload must include a file")
    if "prompt" not in current.openai_image_generation_request:
        raise AssertionError("OpenAI image generation payload must include a prompt")
    if "image" not in current.openai_image_edit_request:
        raise AssertionError("OpenAI image edit payload must include an image")
    if "prompt" not in current.openai_video_generation_request:
        raise AssertionError("OpenAI video generation payload must include a prompt")
    if "contents" not in current.gemini_generate_content_request:
        raise AssertionError("Gemini generateContent payload must include contents")
    if "contents" not in current.gemini_count_tokens_request:
        raise AssertionError("Gemini countTokens payload must include contents")
