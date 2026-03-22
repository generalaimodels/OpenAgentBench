"""Provider protocols and compatibility adapters for embeddings, planning, and reranking."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .scoring import cosine_similarity, lexical_overlap_score, tokenize
from .types import EmbeddingVector


class EmbeddingProvider(Protocol):
    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed a batch of strings."""


class TextModel(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_input: str,
        context: Sequence[str] = (),
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Return a deterministic string completion for planner-style prompts."""


class QueryPlanner(Protocol):
    def resolve_coreferences(self, query: str, conversation: Sequence[str]) -> str:
        """Resolve follow-up references against recent conversation."""

    def decompose_query(self, query: str, *, max_subqueries: int) -> Sequence[str]:
        """Split a complex query into bounded independent subqueries."""

    def expand_query(self, query: str) -> str:
        """Generate a slightly broader query for refinement loops."""


class CrossEncoderScorer(Protocol):
    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return calibrated relevance scores in [0, 1]."""


def _l2_normalize(values: list[float]) -> EmbeddingVector:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


@dataclass(slots=True)
class HashingEmbeddingProvider:
    dimension: int = 256

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        embeddings: list[EmbeddingVector] = []
        for text in texts:
            vector = [0.0] * self.dimension
            tokens = tokenize(text)
            if not tokens:
                embeddings.append(tuple(vector))
                continue
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[bucket] += sign
            embeddings.append(_l2_normalize(vector))
        return embeddings


@dataclass(slots=True)
class HeuristicQueryPlanner:
    def resolve_coreferences(self, query: str, conversation: Sequence[str]) -> str:
        if not conversation:
            return query.strip()
        lowered = query.strip().lower()
        if lowered.startswith(("it ", "it?", "this ", "that ", "they ", "them ", "those ", "also ", "and ")):
            anchor = conversation[-1].strip().rstrip("?")
            if anchor:
                return f"{anchor}. {query.strip()}"
        return query.strip()

    def decompose_query(self, query: str, *, max_subqueries: int) -> Sequence[str]:
        normalized = query.replace(" then ", ", ").replace(" and ", ", ").replace(" also ", ", ")
        parts = [part.strip(" ,;") for part in normalized.split(",") if len(part.strip(" ,;")) >= 8]
        if not parts:
            return ()
        return tuple(parts[:max_subqueries])

    def expand_query(self, query: str) -> str:
        return f"{query.strip()} relevant facts constraints history"


@dataclass(slots=True)
class HeuristicCrossEncoder:
    embedding_provider: EmbeddingProvider = field(default_factory=HashingEmbeddingProvider)

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        query_embedding = self.embedding_provider.embed_batch([query])[0]
        doc_embeddings = self.embedding_provider.embed_batch(documents)
        scores: list[float] = []
        for document, embedding in zip(documents, doc_embeddings, strict=True):
            semantic = max(cosine_similarity(query_embedding, embedding), 0.0)
            lexical = lexical_overlap_score(query, document)
            scores.append(min(1.0, 0.65 * semantic + 0.35 * lexical))
        return scores


def _extract_openai_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    output = getattr(response, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None) or item.get("content", [])
            for part in content:
                part_text = getattr(part, "text", None) or part.get("text")
                if isinstance(part_text, str) and part_text:
                    return part_text

    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None) or first.get("message", {})
        content = getattr(message, "content", None) or message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                text = getattr(part, "text", None) or part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
    raise ValueError("unable to extract text from OpenAI-compatible response")


def _build_context_block(context: Sequence[str]) -> str | None:
    filtered = [item.strip() for item in context if item.strip()]
    if not filtered:
        return None
    return "Context:\n" + "\n".join(filtered)


def _build_openai_responses_input(
    *,
    system_prompt: str,
    user_input: str,
    context: Sequence[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    context_block = _build_context_block(context)
    if context_block is not None:
        content.append({"type": "input_text", "text": context_block})
    content.append({"type": "input_text", "text": user_input})
    return [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


def _build_openai_chat_messages(
    *,
    system_prompt: str,
    user_input: str,
    context: Sequence[str],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    context_block = _build_context_block(context)
    if context_block is not None:
        messages.append({"role": "system", "content": context_block})
    messages.append({"role": "user", "content": user_input})
    return messages


@dataclass(slots=True)
class OpenAICompatibleEmbeddingProvider:
    client: Any
    model: str
    dimensions: int | None = None

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        kwargs: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        data = getattr(response, "data", None) or response["data"]
        return [tuple(float(value) for value in item.embedding) for item in data]


@dataclass(slots=True)
class OpenAICompatibleTextModel:
    client: Any
    model: str
    prefer_responses_api: bool = True
    reasoning_effort: str | None = None

    def complete(
        self,
        *,
        system_prompt: str,
        user_input: str,
        context: Sequence[str] = (),
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        if self.prefer_responses_api and hasattr(self.client, "responses"):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": _build_openai_responses_input(
                    system_prompt=system_prompt,
                    user_input=user_input,
                    context=context,
                ),
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            if self.reasoning_effort is not None and self.reasoning_effort != "none":
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            response = self.client.responses.create(**kwargs)
            return _extract_openai_text(response).strip()

        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=_build_openai_chat_messages(
                    system_prompt=system_prompt,
                    user_input=user_input,
                    context=context,
                ),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return _extract_openai_text(response).strip()
        raise ValueError("client does not expose an OpenAI-compatible text endpoint")


def _extract_gemini_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                texts = [part.text for part in parts if hasattr(part, "text") and isinstance(part.text, str)]
                if texts:
                    return "".join(texts)
    raise ValueError("unable to extract text from Gemini response")


@dataclass(slots=True)
class GeminiTextModel:
    client: Any
    model: str

    def complete(
        self,
        *,
        system_prompt: str,
        user_input: str,
        context: Sequence[str] = (),
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        prompt_parts = [system_prompt]
        if context:
            prompt_parts.append("Context:\n" + "\n".join(context))
        prompt_parts.append("User:\n" + user_input)
        prompt = "\n\n".join(part for part in prompt_parts if part)

        if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            return _extract_gemini_text(response).strip()

        if hasattr(self.client, "generate_content"):
            response = self.client.generate_content(
                model=self.model,
                contents=prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            return _extract_gemini_text(response).strip()
        raise ValueError("client does not expose a Gemini-compatible generate_content endpoint")


@dataclass(slots=True)
class LLMQueryPlanner:
    text_model: TextModel

    def resolve_coreferences(self, query: str, conversation: Sequence[str]) -> str:
        return self.text_model.complete(
            system_prompt="Resolve all pronouns and references using the conversation. Return only the resolved query.",
            user_input=query,
            context=conversation,
            max_tokens=128,
            temperature=0.0,
        )

    def decompose_query(self, query: str, *, max_subqueries: int) -> Sequence[str]:
        response = self.text_model.complete(
            system_prompt=(
                "Split the query into independent subqueries. "
                "Return one subquery per line with no numbering or commentary."
            ),
            user_input=query,
            max_tokens=256,
            temperature=0.0,
        )
        parts = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]
        return tuple(parts[:max_subqueries])

    def expand_query(self, query: str) -> str:
        return self.text_model.complete(
            system_prompt="Rewrite the query to improve retrieval recall while preserving intent. Return only the rewritten query.",
            user_input=query,
            max_tokens=128,
            temperature=0.2,
        )
