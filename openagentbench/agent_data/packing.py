"""Budget-constrained packing for history and memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import HistoryRecord, ScoredMemory


@dataclass(slots=True, frozen=True)
class HistorySelection:
    records: list[HistoryRecord]
    token_count: int
    truncated: bool


@dataclass(slots=True, frozen=True)
class MemorySelection:
    records: list[ScoredMemory]
    token_count: int


def select_contiguous_history_suffix(
    history: Sequence[HistoryRecord],
    token_budget: int,
) -> HistorySelection:
    ordered = sorted(
        (
            record
            for record in history
            if not record.is_pruned and not record.is_compressed and record.turn_index >= 0
        ),
        key=lambda record: record.turn_index,
    )

    if token_budget <= 0 or not ordered:
        return HistorySelection(records=[], token_count=0, truncated=bool(ordered))

    selected: list[HistoryRecord] = []
    running_total = 0
    for record in reversed(ordered):
        if running_total + record.token_count > token_budget:
            break
        selected.append(record)
        running_total += record.token_count

    selected.reverse()
    truncated = bool(selected) and len(selected) != len(ordered)
    if not selected:
        truncated = bool(ordered)
    return HistorySelection(records=selected, token_count=running_total, truncated=truncated)


def pack_memories(
    memories: Iterable[ScoredMemory],
    token_budget: int,
) -> MemorySelection:
    ordered = sorted(
        memories,
        key=lambda item: (
            -item.efficiency,
            -item.score,
            item.memory.token_count,
            -item.memory.confidence,
        ),
    )

    if token_budget <= 0:
        return MemorySelection(records=[], token_count=0)

    selected: list[ScoredMemory] = []
    running_total = 0
    for memory in ordered:
        if running_total + memory.memory.token_count > token_budget:
            continue
        selected.append(memory)
        running_total += memory.memory.token_count
    return MemorySelection(records=selected, token_count=running_total)
