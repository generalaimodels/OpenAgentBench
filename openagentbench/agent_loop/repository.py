"""Checkpoint and audit storage for the agent-loop module."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol, Sequence
from uuid import UUID

from .models import LoopAuditRecord, LoopCheckpointRecord


class LoopRepository(Protocol):
    def put_checkpoint(self, record: LoopCheckpointRecord) -> None:
        """Persist a checkpoint snapshot for a loop."""

    def latest_checkpoint(self, loop_id: UUID) -> LoopCheckpointRecord | None:
        """Return the newest checkpoint for a loop."""

    def list_checkpoints(self, loop_id: UUID) -> Sequence[LoopCheckpointRecord]:
        """Return all checkpoints for a loop ordered by creation."""

    def insert_audit_record(self, record: LoopAuditRecord) -> None:
        """Persist an audit summary for a loop execution."""

    def list_audit_records(self, loop_id: UUID) -> Sequence[LoopAuditRecord]:
        """Return audits for the loop ordered by creation."""


@dataclass(slots=True)
class InMemoryLoopRepository:
    checkpoints: dict[UUID, list[LoopCheckpointRecord]] = field(default_factory=dict)
    audits: dict[UUID, list[LoopAuditRecord]] = field(default_factory=dict)

    def put_checkpoint(self, record: LoopCheckpointRecord) -> None:
        self.checkpoints.setdefault(record.loop_id, []).append(deepcopy(record))

    def latest_checkpoint(self, loop_id: UUID) -> LoopCheckpointRecord | None:
        rows = self.checkpoints.get(loop_id, ())
        if not rows:
            return None
        return deepcopy(rows[-1])

    def list_checkpoints(self, loop_id: UUID) -> Sequence[LoopCheckpointRecord]:
        return tuple(deepcopy(self.checkpoints.get(loop_id, ())))

    def insert_audit_record(self, record: LoopAuditRecord) -> None:
        self.audits.setdefault(record.loop_id, []).append(record)

    def list_audit_records(self, loop_id: UUID) -> Sequence[LoopAuditRecord]:
        return tuple(self.audits.get(loop_id, ()))


__all__ = ["InMemoryLoopRepository", "LoopRepository"]
