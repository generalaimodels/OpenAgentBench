"""Enumerations for the agent-tools orchestration layer."""

from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 fallback
    class StrEnum(str, Enum):
        pass


class TypeClass(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    MCP_TOOL = "mcp_tool"
    MCP_RESOURCE = "mcp_resource"
    MCP_PROMPT = "mcp_prompt"
    JSON_RPC = "json_rpc"
    GRPC = "grpc"
    A2A_AGENT = "a2a_agent"
    SDK_WRAPPED = "sdk_wrapped"
    BROWSER = "browser"
    DESKTOP = "desktop"
    VISION = "vision"
    COMPOSITE = "composite"


class ToolSourceType(StrEnum):
    FUNCTION = "function"
    MCP = "mcp"
    JSONRPC = "jsonrpc"
    GRPC = "grpc"
    A2A = "a2a"
    SDK = "sdk"
    BROWSER = "browser"
    DESKTOP = "desktop"
    VISION = "vision"
    COMPOSITE = "composite"


class MutationClass(StrEnum):
    READ_ONLY = "read_only"
    WRITE_REVERSIBLE = "write_reversible"
    WRITE_IRREVERSIBLE = "write_irreversible"


class TimeoutClass(StrEnum):
    INTERACTIVE = "interactive"
    STANDARD = "standard"
    LONG_RUNNING = "long_running"
    ASYNC = "async"

    def max_duration_ms(self) -> int:
        return {
            TimeoutClass.INTERACTIVE: 500,
            TimeoutClass.STANDARD: 5_000,
            TimeoutClass.LONG_RUNNING: 300_000,
            TimeoutClass.ASYNC: 900_000,
        }[self]


class ToolStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    QUARANTINED = "quarantined"
    UNAVAILABLE = "unavailable"


class AuthDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRES_APPROVAL = "requires_approval"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    AUTO_DENIED = "auto_denied"
    EXPIRED = "expired"


class InvocationStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PENDING = "pending"


class ErrorCode(StrEnum):
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_QUARANTINED = "tool_quarantined"
    UPSTREAM_FAILURE = "upstream_failure"
    INVALID_INPUT = "invalid_input"
    AUTHORIZATION_DENIED = "authorization_denied"
    APPROVAL_REQUIRED = "approval_required"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INTERNAL_TOOL_ERROR = "internal_tool_error"
    OUTPUT_SCHEMA_VIOLATION = "output_schema_violation"
    PRECONDITION_FAILED = "precondition_failed"
    RETRY_EXHAUSTED = "retry_exhausted"
    COMPOSITE_PARTIAL_FAILURE = "composite_partial_failure"


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

