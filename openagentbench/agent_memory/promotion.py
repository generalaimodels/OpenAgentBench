"""Promotion decisions and authority-aware conflict handling."""

from __future__ import annotations

from openagentbench.agent_data.enums import MemoryTier
from openagentbench.agent_retrieval import AuthorityTier, MemoryType

from .enums import ConflictResolution, PromotionAction
from .models import PromotionCandidate, PromotionDecision
from .scoring import candidate_promotion_score


AUTHORITY_RANK = {
    AuthorityTier.CANONICAL: 4,
    AuthorityTier.CURATED: 3,
    AuthorityTier.DERIVED: 2,
    AuthorityTier.EPHEMERAL: 1,
}


def determine_target_layer(candidate: PromotionCandidate) -> MemoryTier | None:
    if candidate.source_layer is MemoryTier.WORKING:
        return MemoryTier.SESSION
    if candidate.source_layer is MemoryTier.SESSION:
        return MemoryTier.EPISODIC
    if candidate.source_layer is MemoryTier.EPISODIC:
        if candidate.memory_type is MemoryType.PROCEDURE or candidate.reusability_score >= 0.75:
            return MemoryTier.PROCEDURAL
        if candidate.reusability_score >= 0.45 or candidate.memory_type in {
            MemoryType.CORRECTION,
            MemoryType.CONSTRAINT,
            MemoryType.FACT,
            MemoryType.PREFERENCE,
        }:
            return MemoryTier.SEMANTIC
    return None


def decide_promotion(candidate: PromotionCandidate, *, minimum_score: float = 0.55) -> PromotionDecision:
    target_layer = determine_target_layer(candidate)
    if target_layer is None:
        return PromotionDecision(
            action=PromotionAction.NONE,
            target_layer=None,
            promotion_score=0.0,
            reason="terminal_layer_or_insufficient_reusability",
        )
    score = candidate_promotion_score(candidate)
    if score < minimum_score:
        return PromotionDecision(
            action=PromotionAction.REJECT,
            target_layer=target_layer,
            promotion_score=score,
            reason="below_promotion_threshold",
        )
    return PromotionDecision(
        action=PromotionAction.PROMOTE,
        target_layer=target_layer,
        promotion_score=score,
        reason="validated_for_promotion",
    )


def resolve_authority_conflict(
    *,
    existing_authority: AuthorityTier,
    existing_confidence: float,
    new_authority: AuthorityTier,
    new_confidence: float,
) -> ConflictResolution:
    existing_rank = AUTHORITY_RANK[existing_authority]
    new_rank = AUTHORITY_RANK[new_authority]
    if new_rank > existing_rank:
        if existing_authority is AuthorityTier.CANONICAL:
            return ConflictResolution.DEFER
        return ConflictResolution.ACCEPT_NEW
    if new_rank == existing_rank:
        if new_confidence > existing_confidence + 0.10:
            return ConflictResolution.ACCEPT_NEW
        return ConflictResolution.DEFER
    if new_confidence > 0.95:
        return ConflictResolution.DEFER
    return ConflictResolution.KEEP_EXISTING


__all__ = ["AUTHORITY_RANK", "decide_promotion", "determine_target_layer", "resolve_authority_conflict"]
