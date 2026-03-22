"""Memory-specific enumerations not already owned by agent_data or agent_retrieval."""

from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 fallback
    class StrEnum(str, Enum):
        pass


class MemoryOperation(StrEnum):
    READ = "read"
    WRITE = "write"
    PROMOTE = "promote"
    EVICT = "evict"
    MERGE = "merge"
    CONFLICT_RESOLVE = "conflict_resolve"
    EXPIRE = "expire"
    ARCHIVE = "archive"
    GC = "gc"
    CHECKPOINT = "checkpoint"
    CONTAMINATION_CHECK = "contamination_check"


class PromotionAction(StrEnum):
    PROMOTE = "promote"
    REJECT = "reject"
    DEDUPLICATE = "deduplicate"
    DEFER = "defer"
    MERGE = "merge"
    NONE = "none"


class PromotionSource(StrEnum):
    WORKING = "working"
    SESSION = "session"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    MANUAL = "manual"
    AUTOMATED = "automated"


class ConflictStatus(StrEnum):
    NONE = "none"
    DETECTED = "detected"
    RESOLVING = "resolving"
    RESOLVED = "resolved"


class ConflictResolution(StrEnum):
    ACCEPT_NEW = "accept_new"
    KEEP_EXISTING = "keep_existing"
    DEFER = "defer"


class ProcedureStatus(StrEnum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    STAGED = "staged"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ProcedureMatchMode(StrEnum):
    DIRECT_INVOKE = "direct_invoke"
    GUIDED_PLANNING = "guided_planning"
    VERIFICATION_ONLY = "verification_only"
    PLAN_FROM_SCRATCH = "plan_from_scratch"


class RecallMode(StrEnum):
    SUCCESS_BIASED = "success_biased"
    FAILURE_BIASED = "failure_biased"
    CORRECTION_BIASED = "correction_biased"
    ALL = "all"


__all__ = [
    "ConflictResolution",
    "ConflictStatus",
    "MemoryOperation",
    "ProcedureMatchMode",
    "ProcedureStatus",
    "PromotionAction",
    "PromotionSource",
    "RecallMode",
]
