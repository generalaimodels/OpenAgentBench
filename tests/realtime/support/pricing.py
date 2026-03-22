"""Budget estimation and guardrails for live realtime tests."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .cases import RealtimeCaseSpec


@dataclass(slots=True, frozen=True)
class CaseCostEstimate:
    vendor: str
    model_id: str
    input_cost_usd: float
    output_cost_usd: float
    audio_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd + self.audio_cost_usd


OPENAI_ESTIMATED_RATES = {
    "input_text_per_1k": 0.0040,
    "output_text_per_1k": 0.0160,
    "audio_input_per_minute": 0.06,
}

GEMINI_ESTIMATED_RATES = {
    "input_text_per_1k": 0.0010,
    "output_text_per_1k": 0.0030,
    "audio_input_per_minute": 0.01,
}


def estimate_case_cost(vendor: str, model_id: str, case: RealtimeCaseSpec) -> CaseCostEstimate:
    rates = OPENAI_ESTIMATED_RATES if vendor == "openai" else GEMINI_ESTIMATED_RATES
    input_cost = (case.estimated_input_tokens / 1000.0) * rates["input_text_per_1k"]
    output_cost = (case.estimated_output_tokens / 1000.0) * rates["output_text_per_1k"]
    audio_cost = 0.0
    if "audio" in case.content_modalities:
        audio_cost = (1.25 / 60.0) * rates["audio_input_per_minute"]
    if "image" in case.content_modalities:
        input_cost += 0.001
    if case.tool_declaration is not None:
        output_cost += 0.0005
    return CaseCostEstimate(
        vendor=vendor,
        model_id=model_id,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        audio_cost_usd=audio_cost,
    )


class BudgetLedger:
    def __init__(self, *, run_id: str, max_budget_usd: float) -> None:
        self._path = Path(tempfile.gettempdir()) / f"openagentbench_live_budget_{run_id}.json"
        self._max_budget_usd = max_budget_usd

    def _read(self) -> dict[str, float]:
        if not self._path.exists():
            return {"total": 0.0, "openai": 0.0, "gemini": 0.0}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self, state: dict[str, float]) -> None:
        self._path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    def current_total(self) -> float:
        return float(self._read()["total"])

    def current_vendor_total(self, vendor: str) -> float:
        return float(self._read()[vendor])

    def assert_can_spend(self, vendor: str, amount_usd: float) -> None:
        state = self._read()
        total_after = state["total"] + amount_usd
        vendor_after = state[vendor] + amount_usd
        vendor_cap = self._max_budget_usd / 2.0
        if total_after > self._max_budget_usd:
            raise RuntimeError(
                f"live realtime budget exceeded: would spend ${total_after:.4f} over ${self._max_budget_usd:.2f}"
            )
        if vendor_after > vendor_cap and total_after > (self._max_budget_usd * 0.85):
            raise RuntimeError(
                f"vendor budget exceeded late in suite: {vendor} would reach ${vendor_after:.4f}"
            )

    def commit(self, vendor: str, amount_usd: float) -> float:
        self.assert_can_spend(vendor, amount_usd)
        state = self._read()
        state["total"] += amount_usd
        state[vendor] += amount_usd
        self._write(state)
        return state["total"]
