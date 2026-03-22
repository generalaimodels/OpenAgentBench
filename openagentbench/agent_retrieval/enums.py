"""Enumerations for the multi-stream retrieval engine."""

from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 fallback
    class StrEnum(str, Enum):
        pass


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class QueryType(StrEnum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    CODE = "code"
    DIAGNOSTIC = "diagnostic"
    CONVERSATIONAL = "conversational"
    MULTIMODAL = "multimodal"
    REASONING = "reasoning"
    AGENTIC = "agentic"


class Modality(StrEnum):
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    STRUCTURED = "structured"
    RUNTIME = "runtime"
    TRACE = "trace"


class OutputStream(StrEnum):
    TEXT_EVIDENCE = "text_evidence"
    CODE_EVIDENCE = "code_evidence"
    STRUCTURED_DATA = "structured_data"
    RUNTIME_STATE = "runtime_state"
    VISION_EVIDENCE = "vision_evidence"
    TOOL_TRACE = "tool_trace"


class RetrievalMode(StrEnum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    HISTORICAL_UTILITY = "historical_utility"
    MEMORY_AUGMENTED = "memory_augmented"
    PROGRESSIVE = "progressive"
    MULTI_HOP = "multi_hop"
    TOOL_STATE_AUGMENTED = "tool_state_augmented"


class SourceTable(StrEnum):
    SESSION = "session"
    HISTORY = "history"
    MEMORY = "memory"
    HISTORY_DERIVED = "history_derived"


class ProtocolType(StrEnum):
    HTTP = "http"
    JSON_RPC = "json_rpc"
    GRPC = "grpc"
    MCP = "mcp"
    FUNCTION_CALL = "function_call"
    TOOL_EVENT = "tool_event"


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    CORRECTION = "correction"
    CONSTRAINT = "constraint"
    PROCEDURE = "procedure"
    SCHEMA = "schema"


class AuthorityTier(StrEnum):
    CANONICAL = "canonical"
    CURATED = "curated"
    DERIVED = "derived"
    EPHEMERAL = "ephemeral"


class TaskOutcome(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class HumanFeedback(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    CORRECTED = "corrected"
    NONE = "none"


class QualityIssue(StrEnum):
    LOW_RELEVANCE = "low_relevance"
    LOW_DIVERSITY = "low_diversity"
    STALE_EVIDENCE = "stale_evidence"
    MISSING_MODALITY = "missing_modality"
    LOOP_DIVERGENCE = "loop_divergence"
    INSUFFICIENT_TOOL_STATE = "insufficient_tool_state"
    INSUFFICIENT_REASONING_SUPPORT = "insufficient_reasoning_support"


class ModelRole(StrEnum):
    EMBEDDING = "embedding"
    GENERATION = "generation"
    RERANKING = "reranking"
    MULTIMODAL = "multimodal"
    THINKING = "thinking"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    AGENTIC_LOOP = "agentic_loop"


class ModelExecutionMode(StrEnum):
    SINGLE_MODEL = "single_model"
    DUAL_MODEL = "dual_model"
    MULTI_MODEL = "multi_model"


class SignalTopology(StrEnum):
    SISO = "siso"
    SIMO = "simo"
    MISO = "miso"
    MIMO = "mimo"


class ReasoningEffort(StrEnum):
    DIRECT = "direct"
    THINKING = "thinking"
    DELIBERATE = "deliberate"
    SELF_REFLECTIVE = "self_reflective"


class LoopStrategy(StrEnum):
    SINGLE_PASS = "single_pass"
    RETRIEVAL_REFINEMENT = "retrieval_refinement"
    TOOL_LOOP = "tool_loop"
    AGENTIC_LOOP = "agentic_loop"
    CRITIQUE_REPAIR = "critique_repair"
