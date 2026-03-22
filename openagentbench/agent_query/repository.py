"""Storage abstractions and an in-memory repository for query-resolution cache and audit data."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Protocol, Sequence

from .models import QueryAuditRecord, QueryCacheEntry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class QueryRepository(Protocol):
    def get_cache_entry(self, cache_key: str) -> QueryCacheEntry | None:
        """Return a live cache entry if one exists."""

    def put_cache_entry(self, entry: QueryCacheEntry) -> None:
        """Insert or replace a query cache entry."""

    def insert_audit_record(self, record: QueryAuditRecord) -> None:
        """Persist an audit record."""

    def list_audit_records(self) -> Sequence[QueryAuditRecord]:
        """Return all audit records ordered by creation time."""


@dataclass(slots=True)
class InMemoryQueryRepository:
    cache: dict[str, QueryCacheEntry] = field(default_factory=dict)
    audit_log: list[QueryAuditRecord] = field(default_factory=list)

    def get_cache_entry(self, cache_key: str) -> QueryCacheEntry | None:
        entry = self.cache.get(cache_key)
        if entry is None:
            return None
        if entry.expires_at <= _utc_now():
            self.cache.pop(cache_key, None)
            return None
        updated = replace(entry, hit_count=entry.hit_count + 1)
        self.cache[cache_key] = updated
        return updated

    def put_cache_entry(self, entry: QueryCacheEntry) -> None:
        self.cache[entry.cache_key] = entry

    def insert_audit_record(self, record: QueryAuditRecord) -> None:
        self.audit_log.append(record)

    def list_audit_records(self) -> Sequence[QueryAuditRecord]:
        return tuple(sorted(self.audit_log, key=lambda item: item.created_at))


__all__ = ["InMemoryQueryRepository", "QueryRepository"]
