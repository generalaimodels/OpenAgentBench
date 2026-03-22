"""Working-memory buffer management and overflow handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from openagentbench.agent_retrieval import Modality

from .models import WorkingMemoryItem, WorkingMemorySnapshot
from .scoring import working_memory_utility


EXPENSIVE_MODALITIES = {Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}


@dataclass(slots=True)
class WorkingMemoryBuffer:
    capacity: int
    items: list[WorkingMemoryItem] = field(default_factory=list)

    @property
    def token_used(self) -> int:
        return sum(item.token_count for item in self.items)

    def snapshot(self) -> WorkingMemorySnapshot:
        return WorkingMemorySnapshot(items=tuple(self.items), token_used=self.token_used, capacity=self.capacity)

    def add(self, item: WorkingMemoryItem) -> None:
        self.items.append(item)

    def extend(self, items: Iterable[WorkingMemoryItem]) -> None:
        self.items.extend(items)

    def prune_to_capacity(self, *, query_text: str, floor_ratio: float = 0.90) -> list[WorkingMemoryItem]:
        evicted: list[WorkingMemoryItem] = []
        for item in self.items:
            item.utility_score = working_memory_utility(item=item, query_text=query_text)

        for item in sorted(self.items, key=lambda current: (-current.token_count, current.utility_score)):
            if self.token_used <= self.capacity:
                break
            if item.modality in EXPENSIVE_MODALITIES and item.utility_score < 0.55 and item.binary_ref is None:
                item.binary_ref = f"memory://external/{item.item_id}"
                item.content_text = item.externalized_label()
                item.token_count = min(item.token_count, 18)

        target = max(int(self.capacity * floor_ratio), 0)
        self.items.sort(key=lambda item: (item.utility_score, item.token_count, item.created_at))
        while self.token_used > target and self.items:
            lowest = self.items[0]
            if lowest.utility_score >= 0.45 and self.token_used <= self.capacity:
                break
            evicted.append(self.items.pop(0))
        return evicted

    def carry_forward_items(self, *, ratio: float = 0.30) -> list[WorkingMemoryItem]:
        carry_limit = int(self.capacity * ratio)
        carried: list[WorkingMemoryItem] = []
        running_total = 0
        for item in sorted(self.items, key=lambda current: (-current.utility_score, current.token_count)):
            if not item.carry_forward:
                continue
            if running_total + item.token_count > carry_limit:
                continue
            carried.append(item)
            running_total += item.token_count
        return carried


__all__ = ["EXPENSIVE_MODALITIES", "WorkingMemoryBuffer"]
