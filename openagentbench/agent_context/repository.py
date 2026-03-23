"""Archive repository for compiled cyclic contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from .models import ContextArchiveEntry


class ContextRepository(Protocol):
    def put_archive_entry(self, entry: ContextArchiveEntry) -> None:
        """Persist a compiled context archive entry."""

    def latest_for_session(self, session_id: UUID) -> ContextArchiveEntry | None:
        """Return the latest archive entry for the session."""

    def list_for_session(self, session_id: UUID) -> tuple[ContextArchiveEntry, ...]:
        """Return archive entries for the session in insertion order."""


@dataclass(slots=True)
class InMemoryContextRepository:
    entries_by_session: dict[UUID, list[ContextArchiveEntry]] = field(default_factory=dict)

    def put_archive_entry(self, entry: ContextArchiveEntry) -> None:
        self.entries_by_session.setdefault(entry.session_id, []).append(entry)

    def latest_for_session(self, session_id: UUID) -> ContextArchiveEntry | None:
        entries = self.entries_by_session.get(session_id)
        if not entries:
            return None
        return entries[-1]

    def list_for_session(self, session_id: UUID) -> tuple[ContextArchiveEntry, ...]:
        return tuple(self.entries_by_session.get(session_id, ()))


__all__ = ["ContextRepository", "InMemoryContextRepository"]
