"""Scoring helpers for working memory utility, episodic recall, and promotion."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from openagentbench.agent_data import MemoryRecord
from openagentbench.agent_data.enums import MemoryTier
from openagentbench.agent_retrieval import Modality
from openagentbench.agent_retrieval.scoring import lexical_overlap_score

from .models import PromotionCandidate, WorkingMemoryItem


def _as_aware(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def working_memory_utility(
    *,
    item: WorkingMemoryItem,
    query_text: str,
    now: datetime | None = None,
) -> float:
    current_time = _as_aware(now or datetime.now(timezone.utc))
    created_at = _as_aware(item.created_at)
    age_seconds = max((current_time - created_at).total_seconds(), 0.0)
    relevance = lexical_overlap_score(query_text, item.content_text)
    recency = 1.0 / (1.0 + age_seconds / 60.0)
    dependency = min(item.dependency_count / 5.0, 1.0)
    modality_value = {
        Modality.TEXT: 1.0,
        Modality.CODE: 1.2,
        Modality.STRUCTURED: 0.9,
        Modality.TRACE: 0.8,
        Modality.RUNTIME: 0.85,
        Modality.DOCUMENT: 0.95,
        Modality.IMAGE: 0.6,
        Modality.AUDIO: 0.6,
        Modality.VIDEO: 0.6,
    }.get(item.modality, 0.75)
    return (
        0.40 * relevance
        + 0.25 * recency
        + 0.20 * dependency
        + 0.15 * modality_value
    )


def memory_record_priority(
    *,
    memory: MemoryRecord,
    query_text: str,
    now: datetime | None = None,
) -> float:
    current_time = _as_aware(now or datetime.now(timezone.utc))
    updated_at = _as_aware(memory.updated_at)
    age_hours = max((current_time - updated_at).total_seconds() / 3600.0, 0.0)
    freshness = math.exp(-0.02 * age_hours)
    lexical = lexical_overlap_score(query_text, memory.content_text)
    access_score = 0.0 if memory.access_count <= 0 else min(math.log1p(memory.access_count) / 3.0, 1.0)
    tier_weight = {
        MemoryTier.WORKING: 0.8,
        MemoryTier.SESSION: 0.9,
        MemoryTier.EPISODIC: 0.95,
        MemoryTier.SEMANTIC: 1.0,
        MemoryTier.PROCEDURAL: 1.1,
    }[memory.memory_tier]
    return tier_weight * (0.45 * lexical + 0.25 * freshness + 0.15 * access_score + 0.15 * memory.confidence)


def episodic_recall_score(
    *,
    memory: MemoryRecord,
    query_text: str,
    now: datetime | None = None,
) -> float:
    current_time = _as_aware(now or datetime.now(timezone.utc))
    created_at = _as_aware(memory.created_at)
    age_days = max((current_time - created_at).total_seconds() / 86400.0, 0.0)
    semantic_similarity = lexical_overlap_score(query_text, memory.content_text)
    recency_decay = math.exp(-0.02 * age_days)
    outcome_quality = float(memory.metadata.get("outcome_score", memory.confidence))
    human_feedback = float(memory.metadata.get("human_feedback_score", 0.0))
    frequency = 0.0 if memory.access_count <= 0 else min(math.log1p(memory.access_count) / 4.0, 1.0)
    return (
        0.40 * semantic_similarity
        + 0.20 * recency_decay
        + 0.25 * max(min(outcome_quality * (1.0 + human_feedback), 1.0), 0.0)
        + 0.15 * frequency
    )


def promotion_score(*, novelty_score: float, correctness_score: float, reusability_score: float) -> float:
    return 0.30 * novelty_score + 0.40 * correctness_score + 0.30 * reusability_score


def candidate_promotion_score(candidate: PromotionCandidate) -> float:
    return promotion_score(
        novelty_score=candidate.novelty_score,
        correctness_score=candidate.correctness_score,
        reusability_score=candidate.reusability_score,
    )


def effective_ttl_days(*, base_days: float, eta: float, access_count: int, mu: float) -> float:
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if eta < 0.0 or eta >= 1.0:
        raise ValueError("eta must be in [0, 1)")
    return base_days * (1.0 - eta * math.exp(-float(access_count) / mu))


__all__ = [
    "candidate_promotion_score",
    "effective_ttl_days",
    "episodic_recall_score",
    "memory_record_priority",
    "promotion_score",
    "working_memory_utility",
]
