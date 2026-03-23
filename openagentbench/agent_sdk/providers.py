"""Provider runtime helpers for OpenAI Python and vLLM-compatible clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openagentbench.agent_retrieval.providers import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleTextModel,
)

from .enums import ProviderTarget
from .models import ProviderClientConfig, ProviderSuite


@dataclass(slots=True)
class AgentSdkProviderFactory:
    def build(self, *, client: Any, config: ProviderClientConfig) -> ProviderSuite:
        text_model = OpenAICompatibleTextModel(
            client=client,
            model=config.model,
            prefer_responses_api=config.prefer_responses_api,
            reasoning_effort=config.reasoning_effort,
            store=config.store,
        )
        embedding_provider = None
        if config.embedding_model is not None:
            embedding_provider = OpenAICompatibleEmbeddingProvider(client=client, model=config.embedding_model)
        return ProviderSuite(
            config=config,
            text_model=text_model,
            embedding_provider=embedding_provider,
        )

    def build_openai(
        self,
        *,
        client: Any,
        model: str,
        embedding_model: str | None = None,
        reasoning_effort: str | None = None,
        store: bool | None = None,
    ) -> ProviderSuite:
        return self.build(
            client=client,
            config=ProviderClientConfig(
                provider=ProviderTarget.OPENAI,
                model=model,
                embedding_model=embedding_model,
                prefer_responses_api=True,
                reasoning_effort=reasoning_effort,
                store=store,
            ),
        )

    def build_vllm(
        self,
        *,
        client: Any,
        model: str,
        embedding_model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> ProviderSuite:
        return self.build(
            client=client,
            config=ProviderClientConfig(
                provider=ProviderTarget.VLLM,
                model=model,
                embedding_model=embedding_model,
                prefer_responses_api=True,
                reasoning_effort=reasoning_effort,
            ),
        )


__all__ = ["AgentSdkProviderFactory"]
