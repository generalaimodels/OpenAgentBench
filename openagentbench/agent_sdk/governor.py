"""Bounded resource and cost enforcement for the universal agent SDK."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import ResourceScope
from .models import BudgetApproval, BudgetLimit, CostRecord


def _sum_tokens(cost: CostRecord) -> int:
    return cost.tokens_input + cost.tokens_output


@dataclass(slots=True)
class CostGovernor:
    limits: dict[ResourceScope, BudgetLimit] = field(default_factory=dict)
    usage: dict[ResourceScope, CostRecord] = field(default_factory=dict)

    def set_limit(self, limit: BudgetLimit) -> None:
        self.limits[limit.scope] = limit

    def check_budget(self, *, scope: ResourceScope, estimated: CostRecord) -> BudgetApproval:
        limit = self.limits.get(scope)
        consumed = self.usage.get(scope, CostRecord())
        if limit is None:
            return BudgetApproval(
                approved=True,
                remaining_api_calls=1_000_000,
                remaining_tokens=1_000_000_000,
                remaining_compute_seconds=1_000_000.0,
                remaining_cost_usd=1_000_000.0,
            )

        remaining_api_calls = limit.api_calls - consumed.api_calls - estimated.api_calls
        remaining_tokens = limit.tokens - _sum_tokens(consumed) - _sum_tokens(estimated)
        remaining_compute_seconds = limit.compute_seconds - consumed.compute_seconds - estimated.compute_seconds
        remaining_cost_usd = limit.monetary_cost_usd - consumed.monetary_cost_usd - estimated.monetary_cost_usd
        approved = all(
            (
                remaining_api_calls >= 0,
                remaining_tokens >= 0,
                remaining_compute_seconds >= 0.0,
                remaining_cost_usd >= 0.0,
            )
        )
        reason = None
        if not approved:
            reason = "resource budget exceeded"
        return BudgetApproval(
            approved=approved,
            remaining_api_calls=max(remaining_api_calls, 0),
            remaining_tokens=max(remaining_tokens, 0),
            remaining_compute_seconds=max(remaining_compute_seconds, 0.0),
            remaining_cost_usd=max(remaining_cost_usd, 0.0),
            reason=reason,
        )

    def record_cost(self, *, scope: ResourceScope, cost: CostRecord) -> None:
        existing = self.usage.get(scope, CostRecord())
        self.usage[scope] = CostRecord(
            api_calls=existing.api_calls + cost.api_calls,
            tokens_input=existing.tokens_input + cost.tokens_input,
            tokens_output=existing.tokens_output + cost.tokens_output,
            compute_seconds=existing.compute_seconds + cost.compute_seconds,
            storage_bytes=existing.storage_bytes + cost.storage_bytes,
            network_bytes=existing.network_bytes + cost.network_bytes,
            monetary_cost_usd=existing.monetary_cost_usd + cost.monetary_cost_usd,
        )

    def get_usage(self, *, scope: ResourceScope) -> CostRecord:
        return self.usage.get(scope, CostRecord())


__all__ = ["CostGovernor"]
