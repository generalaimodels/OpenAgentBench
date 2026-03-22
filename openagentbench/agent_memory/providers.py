"""Provider protocols and adapters for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from openagentbench.agent_data.json_codec import normalize_text
from openagentbench.agent_retrieval.providers import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleTextModel,
    TextModel,
)
from openagentbench.agent_retrieval.scoring import tokenize

from .types import EmbeddingVector


def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = tokenize(text)
    if len(tokens) <= max_tokens:
        return text.strip()
    return " ".join(tokens[:max_tokens]).strip()


@dataclass(slots=True)
class HeuristicMemoryTextModel:
    label: str = "heuristic-memory"

    def complete(
        self,
        *,
        system_prompt: str,
        user_input: str,
        context: Sequence[str] = (),
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        del temperature
        joined_context = " ".join(item.strip() for item in context if item.strip())
        lowered_prompt = system_prompt.lower()
        if "answer yes or no" in lowered_prompt:
            left = normalize_text(context[0] if context else user_input)
            right = normalize_text(context[1] if len(context) > 1 else user_input)
            return "YES" if left == right else "NO"
        if "contradict" in lowered_prompt:
            text = f"{joined_context} {user_input}".lower()
            return "YES" if " not " in f" {text} " and any(token in text for token in ("must", "always", "only")) else "NO"
        combined = " ".join(part for part in (joined_context, user_input) if part).strip()
        if not combined:
            combined = system_prompt
        return _truncate_by_tokens(combined, max_tokens)


@dataclass(slots=True)
class MemoryProviderSuite:
    embedding_provider: EmbeddingProvider = field(default_factory=HashingEmbeddingProvider)
    summarizer_model: TextModel = field(default_factory=HeuristicMemoryTextModel)
    reranker_model: TextModel = field(default_factory=HeuristicMemoryTextModel)
    contradiction_model: TextModel = field(default_factory=HeuristicMemoryTextModel)
    equivalence_model: TextModel = field(default_factory=HeuristicMemoryTextModel)
    procedure_model: TextModel = field(default_factory=HeuristicMemoryTextModel)
    vision_model: TextModel = field(default_factory=HeuristicMemoryTextModel)
    transcription_model: TextModel = field(default_factory=HeuristicMemoryTextModel)

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return list(self.embedding_provider.embed_batch(texts))

    def summarize(self, *, existing_summary: str, additions: Sequence[str], max_tokens: int) -> str:
        context = tuple(item for item in (existing_summary,) if item.strip())
        prompt = "\n".join(line.strip() for line in additions if line.strip())
        text = self.summarizer_model.complete(
            system_prompt=(
                "Update the memory summary. Preserve corrections, decisions, constraints, "
                "open questions, and named procedures."
            ),
            user_input=prompt,
            context=context,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return _truncate_by_tokens(text, max_tokens)

    def are_equivalent(self, left: str, right: str) -> bool:
        response = self.equivalence_model.complete(
            system_prompt="Answer YES or NO only. Are these two statements semantically equivalent?",
            user_input=left,
            context=(right,),
            max_tokens=8,
            temperature=0.0,
        )
        normalized = normalize_text(response).upper()
        return normalized.startswith("YES") or normalize_text(left) == normalize_text(right)

    def contradicts(self, left: str, right: str) -> bool:
        response = self.contradiction_model.complete(
            system_prompt="Answer YES or NO only. Do these two statements contradict each other?",
            user_input=left,
            context=(right,),
            max_tokens=8,
            temperature=0.0,
        )
        normalized = normalize_text(response).upper()
        if normalized.startswith("YES"):
            return True
        left_text = f" {normalize_text(left).lower()} "
        right_text = f" {normalize_text(right).lower()} "
        contradictory_tokens = (" not ", " never ", " no ", " cannot ", " only ")
        return any(token in left_text for token in contradictory_tokens) != any(
            token in right_text for token in contradictory_tokens
        )

    def describe_visual_reference(self, reference: str, *, max_tokens: int = 64) -> str:
        return self.vision_model.complete(
            system_prompt="Describe this visual reference concisely for multimodal memory indexing.",
            user_input=reference,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    def transcribe_reference(self, reference: str, *, max_tokens: int = 96) -> str:
        return self.transcription_model.complete(
            system_prompt="Transcribe or summarize this audio reference for memory indexing.",
            user_input=reference,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    def synthesize_procedure(self, traces: Sequence[str], *, max_tokens: int = 192) -> str:
        return self.procedure_model.complete(
            system_prompt=(
                "Synthesize a reusable procedure with preconditions, ordered steps, "
                "postconditions, and error handling."
            ),
            user_input="\n---\n".join(item.strip() for item in traces if item.strip()),
            max_tokens=max_tokens,
            temperature=0.0,
        )


__all__ = [
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "HeuristicMemoryTextModel",
    "MemoryProviderSuite",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleTextModel",
    "TextModel",
]
