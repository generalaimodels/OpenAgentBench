"""Budget allocation helpers for the memory prefill compiler."""

from __future__ import annotations

from dataclasses import replace

from openagentbench.agent_retrieval import ModelRole, QueryClassification, QueryType, ReasoningEffort

from .models import BudgetAllocation


def _normalize_allocations(base: dict[str, float], total_budget: int) -> BudgetAllocation:
    total_weight = sum(base.values())
    if total_budget <= 0 or total_weight <= 0.0:
        return BudgetAllocation(0, 0, 0, 0, 0, 0, 0, 0)

    ordered_keys = (
        "working_budget",
        "session_budget",
        "episodic_budget",
        "semantic_budget",
        "procedural_budget",
        "multimodal_budget",
        "reserve_budget",
    )
    normalized = {key: base[key] / total_weight for key in ordered_keys}
    raw = {key: int(total_budget * normalized[key]) for key in ordered_keys}
    used = sum(raw.values())
    remainder = max(total_budget - used, 0)
    for key in ordered_keys:
        if remainder <= 0:
            break
        raw[key] += 1
        remainder -= 1
    return BudgetAllocation(total_budget=total_budget, **raw)


def allocate_layer_budgets(
    *,
    total_budget: int,
    classification: QueryClassification,
    has_procedure_match: bool = False,
) -> BudgetAllocation:
    base = {
        "working_budget": 0.20,
        "session_budget": 0.20,
        "episodic_budget": 0.15,
        "semantic_budget": 0.25,
        "procedural_budget": 0.10,
        "multimodal_budget": 0.05,
        "reserve_budget": 0.05,
    }

    if classification.type is QueryType.CODE:
        base["semantic_budget"] *= 1.30
        base["episodic_budget"] *= 0.75
    if classification.type is QueryType.DIAGNOSTIC:
        base["episodic_budget"] *= 1.50
        base["semantic_budget"] *= 0.85
    if classification.type is QueryType.CONVERSATIONAL:
        base["session_budget"] *= 1.50
        base["semantic_budget"] *= 0.70
    if classification.type is QueryType.MULTIMODAL:
        base["multimodal_budget"] *= 3.00
        base["working_budget"] *= 1.15
    if classification.type is QueryType.AGENTIC:
        base["procedural_budget"] *= 1.60
        base["reserve_budget"] *= 1.20
        base["session_budget"] *= 0.85
    if classification.reasoning_effort in {ReasoningEffort.THINKING, ReasoningEffort.DELIBERATE, ReasoningEffort.SELF_REFLECTIVE}:
        base["semantic_budget"] *= 1.10
        base["reserve_budget"] *= 1.15
    if ModelRole.MULTIMODAL in classification.model_roles:
        base["multimodal_budget"] *= 1.25
    if has_procedure_match:
        base["procedural_budget"] *= 2.00
        base["episodic_budget"] *= 0.50

    return _normalize_allocations(base, total_budget)


def compute_working_memory_capacity(
    *,
    context_window_size: int,
    system_prompt_tokens: int,
    tool_budget: int,
    output_reserve: int,
    session_claim: int,
    episodic_claim: int,
    semantic_claim: int,
    procedural_claim: int,
    minimum_working_tokens: int = 256,
) -> int:
    available = (
        context_window_size
        - system_prompt_tokens
        - tool_budget
        - output_reserve
        - session_claim
        - episodic_claim
        - semantic_claim
        - procedural_claim
    )
    return max(available, minimum_working_tokens)


def reallocate_for_procedure_match(allocation: BudgetAllocation) -> BudgetAllocation:
    procedural_gain = min(allocation.episodic_budget // 2, allocation.reserve_budget)
    return replace(
        allocation,
        episodic_budget=max(allocation.episodic_budget - procedural_gain, 0),
        procedural_budget=allocation.procedural_budget + procedural_gain,
    )


__all__ = [
    "allocate_layer_budgets",
    "compute_working_memory_capacity",
    "reallocate_for_procedure_match",
]
