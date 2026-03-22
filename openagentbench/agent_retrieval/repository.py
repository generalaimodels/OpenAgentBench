"""Storage abstractions for retrieval along with an in-memory reference repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, Sequence
from uuid import UUID

from .enums import SourceTable
from .models import FragmentLocator, HistoryEntry, MemoryEntry, SessionTurn


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RetrievalRepository(Protocol):
    def user_exists(self, uu_id: UUID) -> bool:
        """Return whether the user is active and retrievable."""

    def acl_scope(self, uu_id: UUID) -> tuple[str, ...]:
        """Return the effective ACL scope for the user."""

    def list_session_turns(self, uu_id: UUID, session_id: UUID, *, limit: int) -> Sequence[SessionTurn]:
        """Return session turns for the active session ordered by turn_index ascending."""

    def list_history_entries(self, uu_id: UUID, *, limit: int | None = None) -> Sequence[HistoryEntry]:
        """Return historical retrieval feedback for the user."""

    def list_memory_entries(
        self,
        uu_id: UUID,
        *,
        limit: int | None = None,
        include_expired: bool = False,
    ) -> Sequence[MemoryEntry]:
        """Return durable memory rows for the user."""

    def resolve_fragment(self, uu_id: UUID, locator: FragmentLocator) -> tuple[str, datetime, dict[str, object]] | None:
        """Resolve a previously seen fragment locator back to concrete content."""

    def touch_memories(self, uu_id: UUID, memory_ids: Sequence[UUID], *, accessed_at: datetime | None = None) -> None:
        """Update memory access counters for retrieved memories."""


@dataclass(slots=True)
class InMemoryRetrievalRepository:
    active_users: set[UUID] = field(default_factory=set)
    acl_by_user: dict[UUID, tuple[str, ...]] = field(default_factory=dict)
    sessions: dict[UUID, list[SessionTurn]] = field(default_factory=dict)
    history: dict[UUID, list[HistoryEntry]] = field(default_factory=dict)
    memory: dict[UUID, list[MemoryEntry]] = field(default_factory=dict)

    def user_exists(self, uu_id: UUID) -> bool:
        return uu_id in self.active_users

    def acl_scope(self, uu_id: UUID) -> tuple[str, ...]:
        return self.acl_by_user.get(uu_id, ())

    def list_session_turns(self, uu_id: UUID, session_id: UUID, *, limit: int) -> Sequence[SessionTurn]:
        turns = [
            turn
            for turn in self.sessions.get(uu_id, ())
            if turn.session_id == session_id and (turn.expires_at is None or turn.expires_at > _utc_now())
        ]
        turns.sort(key=lambda turn: turn.turn_index)
        if limit <= 0:
            return ()
        return tuple(turns[-limit:])

    def list_history_entries(self, uu_id: UUID, *, limit: int | None = None) -> Sequence[HistoryEntry]:
        rows = list(self.history.get(uu_id, ()))
        rows.sort(key=lambda row: row.created_at, reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return tuple(rows)

    def list_memory_entries(
        self,
        uu_id: UUID,
        *,
        limit: int | None = None,
        include_expired: bool = False,
    ) -> Sequence[MemoryEntry]:
        now = _utc_now()
        rows = [
            item
            for item in self.memory.get(uu_id, ())
            if include_expired or item.expires_at is None or item.expires_at > now
        ]
        rows.sort(key=lambda item: (item.updated_at, item.created_at), reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return tuple(rows)

    def resolve_fragment(self, uu_id: UUID, locator: FragmentLocator) -> tuple[str, datetime, dict[str, object]] | None:
        if locator.source_table is SourceTable.SESSION:
            for turn in self.sessions.get(uu_id, ()):
                if turn.turn_id() == locator.chunk_id:
                    return (
                        turn.content_text,
                        turn.created_at,
                        {
                            "source_table": str(locator.source_table),
                            "role": str(turn.role),
                            "turn_index": turn.turn_index,
                        },
                    )
            return None

        if locator.source_table in {SourceTable.HISTORY, SourceTable.HISTORY_DERIVED}:
            for row in self.history.get(uu_id, ()):
                if row.history_id == locator.chunk_id:
                    content = row.response_summary or row.query_text
                    return (
                        content,
                        row.created_at,
                        {
                            "source_table": str(SourceTable.HISTORY),
                            "task_outcome": str(row.task_outcome),
                            "utility_score": row.utility_score,
                        },
                    )
            return None

        if locator.source_table is SourceTable.MEMORY:
            for row in self.memory.get(uu_id, ()):
                if row.memory_id == locator.chunk_id:
                    return (
                        row.content_text,
                        row.updated_at,
                        {
                            "source_table": str(SourceTable.MEMORY),
                            "authority_tier": str(row.authority_tier),
                            "confidence": row.confidence,
                            "memory_type": str(row.memory_type),
                            "access_count": row.access_count,
                        },
                    )
            return None
        return None

    def touch_memories(self, uu_id: UUID, memory_ids: Sequence[UUID], *, accessed_at: datetime | None = None) -> None:
        access_time = accessed_at or _utc_now()
        wanted = set(memory_ids)
        if not wanted:
            return
        for row in self.memory.get(uu_id, ()):
            if row.memory_id in wanted:
                row.access_count += 1
                row.last_accessed_at = access_time
