"""Compact enums aligned to PostgreSQL smallint storage."""

from __future__ import annotations

from enum import Enum, IntEnum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 fallback
    class StrEnum(str, Enum):
        pass


class SessionStatus(IntEnum):
    ACTIVE = 1
    PAUSED = 2
    CLOSED = 3
    EXPIRED = 4


class MemoryTier(IntEnum):
    WORKING = 0
    SESSION = 1
    EPISODIC = 2
    SEMANTIC = 3
    PROCEDURAL = 4


class MemoryScope(IntEnum):
    LOCAL = 0
    GLOBAL = 1


class ProvenanceType(IntEnum):
    USER_STATED = 0
    SYSTEM_INFERRED = 1
    CORRECTION = 2
    PREFERENCE = 3
    FACT = 4
    INSTRUCTION = 5
    TOOL_OUTPUT = 6


class MessageRole(IntEnum):
    SYSTEM = 0
    USER = 1
    ASSISTANT = 2
    TOOL = 3
    FUNCTION = 4

    def as_openai_role(self) -> str:
        return {
            MessageRole.SYSTEM: "system",
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.TOOL: "tool",
            MessageRole.FUNCTION: "function",
        }[self]


class FinishReason(IntEnum):
    STOP = 0
    LENGTH = 1
    TOOL_CALLS = 2
    CONTENT_FILTER = 3


class TaskType(StrEnum):
    CONTINUATION = "continuation"
    KNOWLEDGE_INTENSIVE = "knowledge_intensive"
    NEW_SESSION_WITH_HISTORY = "new_session_with_history"
    TOOL_HEAVY = "tool_heavy"
