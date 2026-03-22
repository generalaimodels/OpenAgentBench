"""Provider protocols and deterministic adapters for the query module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .config import QueryProviderPolicy
from openagentbench.agent_retrieval.providers import (
    HeuristicQueryPlanner,
    OpenAICompatibleTextModel,
    QueryPlanner,
    TextModel,
)
from openagentbench.agent_retrieval.scoring import tokenize


def _truncate(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = tokenize(text)
    if len(tokens) <= max_tokens:
        return text.strip()
    return " ".join(tokens[:max_tokens]).strip()


@dataclass(slots=True)
class HeuristicQueryTextModel:
    label: str = "heuristic-query"
    hypothetical_prefix: str = "Hypothesis:"
    clarification_template: str = "What specific target should I resolve for: {query}?"
    expansion_suffix: str = "relevant constraints evidence tools and history"

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
        prompt = system_prompt.lower()
        joined_context = " ".join(item.strip() for item in context if item.strip())
        combined = " ".join(part for part in (joined_context, user_input.strip()) if part).strip()
        if "hypothetical answer" in prompt:
            return _truncate(f"{self.hypothetical_prefix} {combined}", max_tokens)
        if "clarifying question" in prompt:
            return _truncate(self.clarification_template.format(query=user_input.strip()), max_tokens)
        if "expand" in prompt:
            return _truncate(f"{user_input.strip()} {self.expansion_suffix}", max_tokens)
        return _truncate(combined or system_prompt, max_tokens)


@dataclass(slots=True)
class QueryProviderSuite:
    planner: QueryPlanner = field(default_factory=HeuristicQueryPlanner)
    intent_model: TextModel = field(default_factory=HeuristicQueryTextModel)
    rewrite_model: TextModel = field(default_factory=HeuristicQueryTextModel)
    hyde_model: TextModel = field(default_factory=HeuristicQueryTextModel)
    clarification_model: TextModel = field(default_factory=HeuristicQueryTextModel)
    policy: QueryProviderPolicy = field(default_factory=QueryProviderPolicy)

    def expand(self, query_text: str, *, context: Sequence[str] = (), max_tokens: int | None = None) -> str:
        return self.rewrite_model.complete(
            system_prompt=self.policy.expand_prompt,
            user_input=query_text,
            context=context,
            max_tokens=self.policy.default_expand_tokens if max_tokens is None else max_tokens,
            temperature=self.policy.deterministic_temperature,
        )

    def hypothetical_answer(self, query_text: str, *, context: Sequence[str] = (), max_tokens: int | None = None) -> str:
        return self.hyde_model.complete(
            system_prompt=self.policy.hypothetical_prompt,
            user_input=query_text,
            context=context,
            max_tokens=self.policy.default_hypothetical_tokens if max_tokens is None else max_tokens,
            temperature=self.policy.deterministic_temperature,
        )

    def clarification(self, query_text: str, *, context: Sequence[str] = (), max_tokens: int | None = None) -> str:
        return self.clarification_model.complete(
            system_prompt=self.policy.clarification_prompt,
            user_input=query_text,
            context=context,
            max_tokens=self.policy.default_clarification_tokens if max_tokens is None else max_tokens,
            temperature=self.policy.deterministic_temperature,
        )


__all__ = [
    "HeuristicQueryTextModel",
    "OpenAICompatibleTextModel",
    "QueryPlanner",
    "QueryProviderSuite",
    "TextModel",
]
