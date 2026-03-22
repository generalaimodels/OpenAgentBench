"""Deterministic realtime case specifications shared across vendors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class RealtimeCaseSpec:
    name: str
    prompt: str
    expected_text_fragment: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    content_modalities: tuple[str, ...]
    memory_fact_text: str | None = None
    keyword_query_text: str | None = None
    selection_query_text: str | None = None
    system_prompt_text: str = (
        "You are a realtime integration test model. "
        "Follow instructions exactly, keep replies terse, and preserve tool semantics."
    )
    tool_declaration: dict[str, Any] | None = None
    tool_response_payload: dict[str, Any] | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def text_memory_case() -> RealtimeCaseSpec:
    return RealtimeCaseSpec(
        name="text_memory_roundtrip",
        prompt=(
            "Remember this durable preference exactly: PostgreSQL is my preferred durable memory store. "
            "Reply with ACK_POSTGRESQL."
        ),
        expected_text_fragment="ACK_POSTGRESQL",
        estimated_input_tokens=36,
        estimated_output_tokens=12,
        content_modalities=("text",),
        memory_fact_text="User prefers PostgreSQL as the durable memory store.",
        keyword_query_text="preferred durable memory store PostgreSQL",
        selection_query_text="What is my preferred durable memory store?",
    )


def image_roundtrip_case() -> RealtimeCaseSpec:
    return RealtimeCaseSpec(
        name="image_conditioned_roundtrip",
        prompt="Look at the attached test image and reply with IMAGE_OK only.",
        expected_text_fragment="IMAGE_OK",
        estimated_input_tokens=30,
        estimated_output_tokens=8,
        content_modalities=("text", "image"),
        extra_metadata={"requires_image_input": True},
    )


def audio_roundtrip_case() -> RealtimeCaseSpec:
    return RealtimeCaseSpec(
        name="audio_roundtrip",
        prompt="Please answer with the exact word audio.",
        expected_text_fragment="audio",
        estimated_input_tokens=20,
        estimated_output_tokens=8,
        content_modalities=("audio",),
    )


def tool_roundtrip_case(tool_declaration: dict[str, Any], tool_response_payload: dict[str, Any]) -> RealtimeCaseSpec:
    return RealtimeCaseSpec(
        name="tool_roundtrip",
        prompt=(
            "Call the lookup_test_clock tool with timezone UTC, then reply with TOOL_OK and the returned UTC timestamp."
        ),
        expected_text_fragment="TOOL_OK",
        estimated_input_tokens=28,
        estimated_output_tokens=24,
        content_modalities=("text", "tool"),
        tool_declaration=tool_declaration,
        tool_response_payload=tool_response_payload,
    )


def selection_roundtrip_case() -> RealtimeCaseSpec:
    return RealtimeCaseSpec(
        name="selection_roundtrip",
        prompt="Use the stored preference and latest history to answer concisely.",
        expected_text_fragment="PostgreSQL",
        estimated_input_tokens=18,
        estimated_output_tokens=16,
        content_modalities=("text",),
        memory_fact_text="User prefers PostgreSQL for durable memory storage.",
        keyword_query_text="durable memory PostgreSQL",
        selection_query_text="Which database should I use for durable memory?",
    )
