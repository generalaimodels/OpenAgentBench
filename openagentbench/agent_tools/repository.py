"""State-store abstractions and the in-memory reference repository for tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID

from .enums import ApprovalStatus
from .models import IdempotencyRecord, ToolApprovalTicket, ToolAuditRecord, ToolCacheEntry, ToolInvocationResponse


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ToolStateRepository(Protocol):
    def get_idempotency_record(self, tool_id: str, key: str) -> IdempotencyRecord | None:
        """Return a non-expired idempotency record if present."""

    def put_idempotency_record(self, record: IdempotencyRecord) -> None:
        """Persist an idempotency replay entry."""

    def get_cache_entry(self, cache_key: str) -> ToolCacheEntry | None:
        """Return a non-expired cache entry if present."""

    def put_cache_entry(self, entry: ToolCacheEntry) -> None:
        """Persist or replace a tool-result cache entry."""

    def invalidate_cache(self, *, tool_id: str | None = None, user_id: UUID | None = None) -> int:
        """Invalidate cache entries by tool and/or user."""

    def append_audit_record(self, record: ToolAuditRecord) -> None:
        """Persist an audit record."""

    def list_audit_records(self, tool_id: str | None = None) -> tuple[ToolAuditRecord, ...]:
        """Return audit records ordered by insertion time."""

    def create_approval_ticket(self, ticket: ToolApprovalTicket) -> None:
        """Persist a new approval ticket."""

    def get_approval_ticket(self, ticket_id: UUID) -> ToolApprovalTicket | None:
        """Return an approval ticket."""

    def resolve_approval_ticket(self, ticket_id: UUID, response: ToolApprovalTicket) -> None:
        """Persist an updated approval ticket."""


@dataclass(slots=True)
class InMemoryToolStateRepository:
    idempotency: dict[tuple[str, str], IdempotencyRecord] = field(default_factory=dict)
    cache: dict[str, ToolCacheEntry] = field(default_factory=dict)
    audit_log: list[ToolAuditRecord] = field(default_factory=list)
    approvals: dict[UUID, ToolApprovalTicket] = field(default_factory=dict)

    def get_idempotency_record(self, tool_id: str, key: str) -> IdempotencyRecord | None:
        record = self.idempotency.get((tool_id, key))
        if record is None:
            return None
        if record.expires_at <= _utc_now():
            self.idempotency.pop((tool_id, key), None)
            return None
        return record

    def put_idempotency_record(self, record: IdempotencyRecord) -> None:
        self.idempotency[(record.tool_id, record.key)] = record

    def get_cache_entry(self, cache_key: str) -> ToolCacheEntry | None:
        entry = self.cache.get(cache_key)
        if entry is None:
            return None
        if entry.expires_at <= _utc_now():
            self.cache.pop(cache_key, None)
            return None
        entry.hit_count += 1
        return entry

    def put_cache_entry(self, entry: ToolCacheEntry) -> None:
        self.cache[entry.cache_key] = entry

    def invalidate_cache(self, *, tool_id: str | None = None, user_id: UUID | None = None) -> int:
        invalidated = 0
        for cache_key in list(self.cache):
            entry = self.cache[cache_key]
            if tool_id is not None and entry.tool_id != tool_id:
                continue
            if user_id is not None and entry.user_id != user_id:
                continue
            self.cache.pop(cache_key, None)
            invalidated += 1
        return invalidated

    def append_audit_record(self, record: ToolAuditRecord) -> None:
        self.audit_log.append(record)

    def list_audit_records(self, tool_id: str | None = None) -> tuple[ToolAuditRecord, ...]:
        if tool_id is None:
            return tuple(self.audit_log)
        return tuple(record for record in self.audit_log if record.tool_id == tool_id)

    def create_approval_ticket(self, ticket: ToolApprovalTicket) -> None:
        self.approvals[ticket.ticket_id] = ticket

    def get_approval_ticket(self, ticket_id: UUID) -> ToolApprovalTicket | None:
        ticket = self.approvals.get(ticket_id)
        if ticket is None:
            return None
        if ticket.expires_at <= _utc_now() and ticket.status.value == "pending":
            expired = ToolApprovalTicket(
                ticket_id=ticket.ticket_id,
                tool_id=ticket.tool_id,
                params_redacted=ticket.params_redacted,
                requested_by=ticket.requested_by,
                agent_id=ticket.agent_id,
                status=ApprovalStatus.EXPIRED,
                created_at=ticket.created_at,
                expires_at=ticket.expires_at,
                resolution_by=ticket.resolution_by,
                resolution_at=ticket.resolution_at,
                metadata=ticket.metadata,
            )
            self.approvals[ticket_id] = expired
            return expired
        return ticket

    def resolve_approval_ticket(self, ticket_id: UUID, response: ToolApprovalTicket) -> None:
        self.approvals[ticket_id] = response


__all__ = ["InMemoryToolStateRepository", "ToolStateRepository", "ToolInvocationResponse"]
