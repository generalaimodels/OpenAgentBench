"""Storage abstractions and an in-memory reference repository for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, Sequence
from uuid import UUID

from openagentbench.agent_data import HistoryRecord, MemoryRecord
from openagentbench.agent_data.enums import MemoryTier

from .models import MemoryAuditRecord, MemoryCacheEntry, SessionCheckpointRecord, WorkingMemoryItem


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRepository(Protocol):
    def user_exists(self, user_id: UUID) -> bool:
        """Return whether the user is active for memory operations."""

    def list_history(self, user_id: UUID, session_id: UUID, *, limit: int | None = None) -> Sequence[HistoryRecord]:
        """Return session history ordered by turn index."""

    def list_memories(
        self,
        user_id: UUID,
        *,
        tiers: Sequence[MemoryTier] | None = None,
        limit: int | None = None,
    ) -> Sequence[MemoryRecord]:
        """Return durable memory rows for the requested tiers."""

    def list_working_items(
        self,
        user_id: UUID,
        session_id: UUID,
        *,
        step_id: UUID | None = None,
    ) -> Sequence[WorkingMemoryItem]:
        """Return working-memory items for a session and optional step."""

    def list_checkpoints(self, user_id: UUID, session_id: UUID) -> Sequence[SessionCheckpointRecord]:
        """Return checkpoints ordered by sequence descending."""

    def upsert_working_item(self, item: WorkingMemoryItem) -> None:
        """Persist or replace a working-memory item in the backing store."""

    def insert_checkpoint(self, checkpoint: SessionCheckpointRecord) -> None:
        """Persist a session checkpoint."""

    def insert_audit_record(self, record: MemoryAuditRecord) -> None:
        """Persist an audit event."""

    def get_cache_entry(self, cache_key: str) -> MemoryCacheEntry | None:
        """Return a non-expired cache entry if one exists."""

    def put_cache_entry(self, entry: MemoryCacheEntry) -> None:
        """Insert or replace a cache entry."""

    def invalidate_cache(self, user_id: UUID, *, tier: MemoryTier | None = None) -> int:
        """Invalidate cache entries for the user and optional tier."""


@dataclass(slots=True)
class InMemoryMemoryRepository:
    active_users: set[UUID] = field(default_factory=set)
    history: dict[tuple[UUID, UUID], list[HistoryRecord]] = field(default_factory=dict)
    memories: dict[UUID, list[MemoryRecord]] = field(default_factory=dict)
    working: dict[tuple[UUID, UUID], list[WorkingMemoryItem]] = field(default_factory=dict)
    checkpoints: dict[tuple[UUID, UUID], list[SessionCheckpointRecord]] = field(default_factory=dict)
    audit_log: dict[UUID, list[MemoryAuditRecord]] = field(default_factory=dict)
    cache: dict[str, MemoryCacheEntry] = field(default_factory=dict)

    def user_exists(self, user_id: UUID) -> bool:
        return user_id in self.active_users

    def list_history(self, user_id: UUID, session_id: UUID, *, limit: int | None = None) -> Sequence[HistoryRecord]:
        rows = sorted(self.history.get((user_id, session_id), ()), key=lambda row: row.turn_index)
        if limit is not None:
            rows = rows[-limit:]
        return tuple(rows)

    def list_memories(
        self,
        user_id: UUID,
        *,
        tiers: Sequence[MemoryTier] | None = None,
        limit: int | None = None,
    ) -> Sequence[MemoryRecord]:
        rows = list(self.memories.get(user_id, ()))
        if tiers is not None:
            tier_set = set(tiers)
            rows = [row for row in rows if row.memory_tier in tier_set]
        rows.sort(key=lambda row: (row.updated_at, row.created_at), reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return tuple(rows)

    def list_working_items(
        self,
        user_id: UUID,
        session_id: UUID,
        *,
        step_id: UUID | None = None,
    ) -> Sequence[WorkingMemoryItem]:
        rows = list(self.working.get((user_id, session_id), ()))
        if step_id is not None:
            rows = [row for row in rows if row.step_id == step_id]
        rows.sort(key=lambda row: row.created_at)
        return tuple(rows)

    def list_checkpoints(self, user_id: UUID, session_id: UUID) -> Sequence[SessionCheckpointRecord]:
        rows = list(self.checkpoints.get((user_id, session_id), ()))
        rows.sort(key=lambda row: row.checkpoint_seq, reverse=True)
        return tuple(rows)

    def upsert_working_item(self, item: WorkingMemoryItem) -> None:
        key = (item.user_id, item.session_id)
        rows = self.working.setdefault(key, [])
        for index, existing in enumerate(rows):
            if existing.item_id == item.item_id:
                rows[index] = item
                return
        rows.append(item)

    def insert_checkpoint(self, checkpoint: SessionCheckpointRecord) -> None:
        self.checkpoints.setdefault((checkpoint.user_id, checkpoint.session_id), []).append(checkpoint)

    def insert_audit_record(self, record: MemoryAuditRecord) -> None:
        self.audit_log.setdefault(record.user_id, []).append(record)

    def get_cache_entry(self, cache_key: str) -> MemoryCacheEntry | None:
        entry = self.cache.get(cache_key)
        if entry is None:
            return None
        if entry.expires_at <= _utc_now():
            self.cache.pop(cache_key, None)
            return None
        return entry

    def put_cache_entry(self, entry: MemoryCacheEntry) -> None:
        self.cache[entry.cache_key] = entry

    def invalidate_cache(self, user_id: UUID, *, tier: MemoryTier | None = None) -> int:
        invalidated = 0
        for key in list(self.cache):
            entry = self.cache[key]
            if entry.user_id != user_id:
                continue
            if tier is not None and entry.layer is not tier:
                continue
            self.cache.pop(key, None)
            invalidated += 1
        return invalidated


__all__ = ["InMemoryMemoryRepository", "MemoryRepository"]
