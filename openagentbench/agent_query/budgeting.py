"""Budget allocation helpers for query understanding."""

from __future__ import annotations

from .config import QueryBudgetPolicy
from .models import QueryBudgetAllocation


def _normalize(weights: dict[str, float], total_budget: int) -> QueryBudgetAllocation:
    total_weight = sum(weights.values())
    if total_budget <= 0 or total_weight <= 0.0:
        return QueryBudgetAllocation(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    ordered_keys = (
        "context_budget",
        "intent_budget",
        "pragmatic_budget",
        "cognitive_budget",
        "rewrite_budget",
        "decomposition_budget",
        "routing_budget",
        "clarification_budget",
        "reserve_budget",
    )
    normalized = {key: weights[key] / total_weight for key in ordered_keys}
    raw = {key: int(total_budget * normalized[key]) for key in ordered_keys}
    remainder = max(total_budget - sum(raw.values()), 0)
    for key in ordered_keys:
        if remainder <= 0:
            break
        raw[key] += 1
        remainder -= 1
    return QueryBudgetAllocation(total_budget=total_budget, **raw)


def allocate_query_understanding_budget(
    *,
    context_window_size: int,
    query_budget_ratio: float | None = None,
    policy: QueryBudgetPolicy | None = None,
) -> QueryBudgetAllocation:
    active_policy = policy or QueryBudgetPolicy()
    ratio = active_policy.query_budget_ratio if query_budget_ratio is None else query_budget_ratio
    total_budget = max(int(context_window_size * ratio), 0)
    return _normalize(active_policy.stage_weights(), total_budget)


__all__ = ["allocate_query_understanding_budget"]
