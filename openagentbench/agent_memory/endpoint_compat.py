"""OpenAI-compatible tool definitions and payload-shape checks for the memory module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openagentbench.agent_retrieval.endpoint_compat import (
    EndpointCompatibilityReport,
    assert_endpoint_payload_compatibility,
    build_endpoint_compatibility_report,
    build_gemini_count_tokens_request,
    build_gemini_generate_content_request,
    build_openai_audio_speech_request,
    build_openai_audio_transcription_request,
    build_openai_audio_translation_request,
    build_openai_chat_completions_request,
    build_openai_embeddings_request,
    build_openai_image_edit_request,
    build_openai_image_generation_request,
    build_openai_realtime_session_request,
    build_openai_responses_request,
    build_openai_video_generation_request,
)


def build_memory_read_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read local, global, or session memory fragments for the current user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "layer": {
                        "type": "string",
                        "enum": ["session", "episodic", "semantic", "procedural", "auto"],
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["session", "local", "global", "auto"],
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def build_memory_write_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Write a validated memory item into the correct local, global, or session layer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "target_layer": {
                        "type": "string",
                        "enum": ["session", "episodic", "semantic", "procedural"],
                    },
                    "target_scope": {
                        "type": "string",
                        "enum": ["session", "local", "global"],
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["fact", "preference", "correction", "constraint", "procedure"],
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["content", "target_layer", "target_scope", "memory_type"],
                "additionalProperties": False,
            },
        },
    }


def build_memory_inspect_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "memory_inspect",
            "description": "Inspect memory health, layer utilization, and cache state for the current user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["session", "local", "global", "all"]},
                    "include_audit": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
    }


@dataclass(slots=True, frozen=True)
class MemoryEndpointCompatibilityReport:
    retrieval_report: EndpointCompatibilityReport
    memory_read_tool: dict[str, Any]
    memory_write_tool: dict[str, Any]
    memory_inspect_tool: dict[str, Any]
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


def build_memory_endpoint_compatibility_report() -> MemoryEndpointCompatibilityReport:
    tools = [
        build_memory_read_tool_definition(),
        build_memory_write_tool_definition(),
        build_memory_inspect_tool_definition(),
    ]
    retrieval_report = build_endpoint_compatibility_report()
    responses_request = build_openai_responses_request(
        model="gpt-5-mini",
        system_prompt="You are a memory orchestration model.",
        user_input="Inspect the user's global memory for database preferences.",
        context=("Session summary: the user is designing advanced memory management.",),
        reasoning_effort="medium",
    )
    responses_request["tools"] = tools
    chat_request = build_openai_chat_completions_request(
        model="gpt-4o-mini",
        system_prompt="You are a memory orchestration model.",
        user_input="Read local session memory and summarize it.",
        context=("Global memory: PostgreSQL is the durable store.",),
    )
    chat_request["tools"] = tools
    return MemoryEndpointCompatibilityReport(
        retrieval_report=retrieval_report,
        memory_read_tool=tools[0],
        memory_write_tool=tools[1],
        memory_inspect_tool=tools[2],
        openai_responses_request=responses_request,
        openai_chat_request=chat_request,
        openai_embeddings_request=build_openai_embeddings_request(
            model="text-embedding-3-small",
            inputs=(
                "local memory preference",
                "global semantic rule",
                "session correction",
            ),
            dimensions=256,
        ),
        openai_realtime_request=build_openai_realtime_session_request(
            model="gpt-realtime-mini",
            modalities=("text", "audio"),
            instructions="Coordinate session, local, and global memory streams.",
        ),
        openai_audio_speech_request=build_openai_audio_speech_request(
            model="gpt-4o-mini-tts",
            input_text="Summarize current session memory.",
        ),
        openai_audio_transcription_request=build_openai_audio_transcription_request(
            model="gpt-4o-mini-transcribe",
            file_name="memory-note.wav",
        ),
        openai_audio_translation_request=build_openai_audio_translation_request(
            model="gpt-4o-mini-transcribe",
            file_name="memory-note.wav",
        ),
        openai_image_generation_request=build_openai_image_generation_request(
            model="gpt-image-1",
            prompt="Generate a memory hierarchy diagram.",
        ),
        openai_image_edit_request=build_openai_image_edit_request(
            model="gpt-image-1",
            prompt="Highlight the session-memory layer.",
            image_name="memory-graph.png",
            mask_name="mask.png",
        ),
        openai_video_generation_request=build_openai_video_generation_request(
            model="sora-2",
            prompt="Animate the memory promotion pipeline.",
            image_name="memory-graph.png",
        ),
        gemini_generate_content_request=build_gemini_generate_content_request(
            system_instruction="You are a memory compatibility checker.",
            user_text="Summarize the local and global memory state.",
        ),
        gemini_count_tokens_request=build_gemini_count_tokens_request(
            system_instruction="You are a memory compatibility checker.",
            user_text="Summarize the local and global memory state.",
        ),
    )


def assert_memory_endpoint_payload_compatibility(
    report: MemoryEndpointCompatibilityReport | None = None,
) -> None:
    current = report or build_memory_endpoint_compatibility_report()
    assert_endpoint_payload_compatibility(current.retrieval_report)
    assert current.memory_read_tool["function"]["name"] == "memory_read"
    assert current.memory_write_tool["function"]["name"] == "memory_write"
    assert current.memory_inspect_tool["function"]["name"] == "memory_inspect"
    assert current.openai_responses_request["input"][0]["role"] == "system"
    assert current.openai_chat_request["messages"][0]["role"] == "system"
    assert len(current.openai_responses_request["tools"]) == 3
    assert current.openai_embeddings_request["model"] == "text-embedding-3-small"
    assert current.openai_realtime_request["type"] == "session.update"
    assert "input" in current.openai_audio_speech_request
    assert "file" in current.openai_audio_transcription_request
    assert "file" in current.openai_audio_translation_request
    assert "prompt" in current.openai_image_generation_request
    assert "image" in current.openai_image_edit_request
    assert "prompt" in current.openai_video_generation_request
    assert "contents" in current.gemini_generate_content_request
    assert "contents" in current.gemini_count_tokens_request


__all__ = [
    "MemoryEndpointCompatibilityReport",
    "assert_memory_endpoint_payload_compatibility",
    "build_memory_endpoint_compatibility_report",
    "build_memory_inspect_tool_definition",
    "build_memory_read_tool_definition",
    "build_memory_write_tool_definition",
]
