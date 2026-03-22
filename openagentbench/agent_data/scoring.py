"""Scoring functions for retrieval, promotion, and conflict handling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

from .enums import ProvenanceType
from .models import MemoryRecord, ScoredMemory
from .types import EmbeddingVector


PROVENANCE_AUTHORITY: dict[ProvenanceType, float] = {
    ProvenanceType.USER_STATED: 1.0,
    ProvenanceType.CORRECTION: 0.95,
    ProvenanceType.INSTRUCTION: 0.90,
    ProvenanceType.PREFERENCE: 0.85,
    ProvenanceType.FACT: 0.80,
    ProvenanceType.TOOL_OUTPUT: 0.70,
    ProvenanceType.SYSTEM_INFERRED: 0.60,
}


@dataclass(slots=True, frozen=True)
class RetrievalWeights:
    semantic_weight: float = 0.45
    freshness_weight: float = 0.20
    access_weight: float = 0.15
    authority_weight: float = 0.20
    freshness_lambda: float = 0.005


@dataclass(slots=True, frozen=True)
class PromotionRule:
    discount_factor: float
    utility_threshold: float
    minimum_access_count: int


PROMOTION_RULES = {
    "L1_TO_L2": PromotionRule(0.01, 0.30, 3),
    "L2_TO_L3": PromotionRule(0.001, 0.50, 10),
    "L2_TO_L4": PromotionRule(0.001, 0.50, 10),
}


def cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("embedding dimensions must match")
    numerator = 0.0
    lhs_norm = 0.0
    rhs_norm = 0.0
    for left, right in zip(lhs, rhs, strict=True):
        numerator += left * right
        lhs_norm += left * left
        rhs_norm += right * right
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return numerator / math.sqrt(lhs_norm * rhs_norm)


def lexical_overlap_score(query_text: str, content_text: str) -> float:
    query_terms = set(query_text.lower().split())
    content_terms = set(content_text.lower().split())
    if not query_terms or not content_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms | content_terms)


def _age_in_hours(reference_time: datetime, *, now: datetime) -> float:
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta = now - reference_time
    return max(delta.total_seconds() / 3600.0, 0.0)


def _access_score(access_count: int, max_access_count: int) -> float:
    if access_count <= 0 or max_access_count <= 0:
        return 0.0
    return math.log1p(access_count) / math.log1p(max_access_count)


def compute_memory_score(
    *,
    query_text: str,
    query_embedding: EmbeddingVector | None,
    memory: MemoryRecord,
    max_access_count: int,
    now: datetime,
    weights: RetrievalWeights,
) -> float:
    semantic_score = lexical_overlap_score(query_text, memory.content_text)
    if query_embedding is not None and memory.content_embedding is not None:
        semantic_score = cosine_similarity(query_embedding, memory.content_embedding)

    reference_time = memory.last_accessed_at or memory.created_at
    freshness_score = math.exp(-weights.freshness_lambda * _age_in_hours(reference_time, now=now))
    access_score = _access_score(memory.access_count, max_access_count)
    authority_score = memory.confidence * PROVENANCE_AUTHORITY[memory.provenance_type]

    return (
        weights.semantic_weight * semantic_score
        + weights.freshness_weight * freshness_score
        + weights.access_weight * access_score
        + weights.authority_weight * authority_score
    )


def score_memories(
    *,
    query_text: str,
    query_embedding: EmbeddingVector | None,
    memories: Iterable[MemoryRecord],
    now: datetime,
    weights: RetrievalWeights,
) -> list[ScoredMemory]:
    active_memories = [memory for memory in memories if memory.is_active and memory.token_count > 0]
    if not active_memories:
        return []

    max_access_count = max(memory.access_count for memory in active_memories)
    scored: list[ScoredMemory] = []
    for memory in active_memories:
        score = compute_memory_score(
            query_text=query_text,
            query_embedding=query_embedding,
            memory=memory,
            max_access_count=max_access_count,
            now=now,
            weights=weights,
        )
        efficiency = score / float(memory.token_count)
        scored.append(ScoredMemory(memory=memory, score=score, efficiency=efficiency))
    return scored


def mean_relevance(memory: MemoryRecord) -> float:
    if memory.access_count <= 0:
        return 0.0
    return memory.relevance_accumulator / float(memory.access_count)


def promotion_utility(memory: MemoryRecord, *, now: datetime, rule: PromotionRule) -> float:
    age_hours = _age_in_hours(memory.created_at, now=now)
    return (mean_relevance(memory) * math.log2(1 + memory.access_count)) / (1 + rule.discount_factor * age_hours)


def should_promote(memory: MemoryRecord, *, now: datetime, rule: PromotionRule) -> bool:
    if memory.access_count < rule.minimum_access_count:
        return False
    return promotion_utility(memory, now=now, rule=rule) > rule.utility_threshold
