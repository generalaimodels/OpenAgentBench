"""Model-role routing for selective-purpose retrieval execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .enums import ModelExecutionMode, ModelRole, Modality, OutputStream, ReasoningEffort
from .models import ModelCapabilityProfile, QueryClassification, SelectedModelPlan


@dataclass(slots=True)
class ModelRouter:
    def select(
        self,
        query_classification: QueryClassification,
        profiles: Sequence[ModelCapabilityProfile],
    ) -> SelectedModelPlan:
        bindings: dict[ModelRole, ModelCapabilityProfile] = {}
        for role in query_classification.model_roles:
            candidate = self._select_best_profile(
                role,
                query_classification=query_classification,
                profiles=profiles,
                current_bindings=bindings,
            )
            if candidate is not None:
                bindings[role] = candidate

        primary_model = self._select_primary(query_classification, bindings)
        return SelectedModelPlan(
            query_classification=query_classification,
            primary_model=primary_model,
            role_bindings=bindings,
        )

    def _select_primary(
        self,
        query_classification: QueryClassification,
        bindings: dict[ModelRole, ModelCapabilityProfile],
    ) -> ModelCapabilityProfile | None:
        for preferred_role in (
            ModelRole.MULTIMODAL,
            ModelRole.THINKING,
            ModelRole.GENERATION,
            ModelRole.PLANNER,
            ModelRole.EXECUTOR,
        ):
            if preferred_role in bindings:
                return bindings[preferred_role]
        if query_classification.model_execution_mode is ModelExecutionMode.DUAL_MODEL:
            return bindings.get(ModelRole.RERANKING) or bindings.get(ModelRole.GENERATION)
        return next(iter(bindings.values()), None)

    def _select_best_profile(
        self,
        role: ModelRole,
        *,
        query_classification: QueryClassification,
        profiles: Sequence[ModelCapabilityProfile],
        current_bindings: dict[ModelRole, ModelCapabilityProfile],
    ) -> ModelCapabilityProfile | None:
        scored = [
            (self._score_profile(role, profile, query_classification, current_bindings), profile)
            for profile in profiles
        ]
        scored = [item for item in scored if item[0] > 0.0]
        if not scored:
            return None
        scored.sort(
            key=lambda item: (
                item[0],
                item[1].supports_json_schema,
                item[1].supports_tools,
                item[1].supports_images or item[1].supports_documents,
            ),
            reverse=True,
        )
        return scored[0][1]

    def _score_profile(
        self,
        role: ModelRole,
        profile: ModelCapabilityProfile,
        query_classification: QueryClassification,
        current_bindings: dict[ModelRole, ModelCapabilityProfile],
    ) -> float:
        score = 0.0
        if profile.role is role:
            score += 8.0
        elif role in {ModelRole.THINKING, ModelRole.CRITIC, ModelRole.PLANNER} and profile.role in {
            ModelRole.GENERATION,
            ModelRole.MULTIMODAL,
            ModelRole.THINKING,
        }:
            score += 4.0
        elif role is ModelRole.EXECUTOR and profile.role in {ModelRole.GENERATION, ModelRole.MULTIMODAL}:
            score += 4.0
        elif role is ModelRole.RERANKING and profile.role in {ModelRole.GENERATION, ModelRole.EMBEDDING}:
            score += 2.0

        if role is ModelRole.EMBEDDING and profile.role is ModelRole.EMBEDDING:
            score += 4.0
        if role is ModelRole.MULTIMODAL and (profile.supports_images or profile.supports_documents or profile.supports_audio):
            score += 4.0
        if role in {ModelRole.PLANNER, ModelRole.EXECUTOR, ModelRole.AGENTIC_LOOP} and profile.supports_tools:
            score += 3.0
        if OutputStream.STRUCTURED_DATA in query_classification.output_streams and profile.supports_json_schema:
            score += 2.0
        if self._needs_visual_capability(query_classification) and (profile.supports_images or profile.supports_documents):
            score += 2.0
        if Modality.AUDIO in query_classification.preferred_modalities and profile.supports_audio:
            score += 1.5
        if Modality.VIDEO in query_classification.preferred_modalities and profile.supports_video:
            score += 1.5
        if query_classification.reasoning_effort is not ReasoningEffort.DIRECT and profile.role in {
            ModelRole.THINKING,
            ModelRole.GENERATION,
        }:
            score += 1.5
        if query_classification.model_execution_mode is ModelExecutionMode.MULTI_MODEL and role in {
            ModelRole.PLANNER,
            ModelRole.EXECUTOR,
            ModelRole.CRITIC,
        }:
            score += 1.0

        if profile in current_bindings.values():
            score -= 0.5
        return score

    def _needs_visual_capability(self, query_classification: QueryClassification) -> bool:
        return any(
            modality in query_classification.preferred_modalities
            for modality in (Modality.IMAGE, Modality.DOCUMENT, Modality.VIDEO)
        )


def default_profiles() -> tuple[ModelCapabilityProfile, ...]:
    return (
        ModelCapabilityProfile(
            model_name="text-embedding-3-large",
            role=ModelRole.EMBEDDING,
            supports_text=True,
            supports_documents=True,
        ),
        ModelCapabilityProfile(
            model_name="gpt-4.1-mini",
            role=ModelRole.GENERATION,
            supports_text=True,
            supports_documents=True,
            supports_tools=True,
            supports_json_schema=True,
        ),
        ModelCapabilityProfile(
            model_name="gpt-4.1",
            role=ModelRole.MULTIMODAL,
            supports_text=True,
            supports_documents=True,
            supports_images=True,
            supports_audio=True,
            supports_tools=True,
            supports_json_schema=True,
        ),
        ModelCapabilityProfile(
            model_name="bge-reranker-v2-m3",
            role=ModelRole.RERANKING,
            supports_text=True,
        ),
        ModelCapabilityProfile(
            model_name="o4-mini",
            role=ModelRole.THINKING,
            supports_text=True,
            supports_documents=True,
            supports_tools=True,
            supports_json_schema=True,
        ),
    )
