"""Enumerations for the agent-loop orchestration engine."""

from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 fallback
    class StrEnum(str, Enum):
        pass


class LoopPhase(StrEnum):
    CONTEXT_ASSEMBLE = "context_assemble"
    PLAN = "plan"
    DECOMPOSE = "decompose"
    PREDICT = "predict"
    RETRIEVE = "retrieve"
    ACT = "act"
    METACOGNITIVE_CHECK = "metacognitive_check"
    VERIFY = "verify"
    CRITIQUE = "critique"
    REPAIR = "repair"
    ESCALATE = "escalate"
    COMMIT = "commit"
    HALT = "halt"
    FAIL = "fail"


class CognitiveMode(StrEnum):
    SYSTEM1_FAST = "system1_fast"
    SYSTEM2_DELIBERATIVE = "system2_deliberative"
    HYBRID = "hybrid"


class SubsystemAvailability(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ActionStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    SKIPPED = "skipped"


class MetacognitiveDecision(StrEnum):
    PROCEED_TO_VERIFY = "proceed_to_verify"
    RE_RETRIEVE = "re_retrieve"
    RE_EXECUTE = "re_execute"
    EARLY_CRITIQUE = "early_critique"


class RootCauseClass(StrEnum):
    SCHEMA_VIOLATION = "schema_violation"
    FACTUAL_ERROR = "factual_error"
    LOGIC_ERROR = "logic_error"
    INCOMPLETENESS = "incompleteness"
    COHERENCE_FAILURE = "coherence_failure"
    SAFETY_VIOLATION = "safety_violation"
    HALLUCINATION = "hallucination"
    RETRIEVAL_GAP = "retrieval_gap"
    TOOL_FAILURE = "tool_failure"
    TOOL_MISUSE = "tool_misuse"
    MEMORY_STALE = "memory_stale"
    CONTEXT_OVERFLOW = "context_overflow"
    MODEL_CAPABILITY_LIMIT = "model_capability_limit"
    AGENT_DELEGATION_FAILURE = "agent_delegation_failure"


class RepairStrategy(StrEnum):
    CONSTRAINED_REGENERATION = "constrained_regeneration"
    MINIMAL_EDIT = "minimal_edit"
    RE_RETRIEVE_AND_REGENERATE = "re_retrieve_and_regenerate"
    TOOL_SUBSTITUTE = "tool_substitute"
    EXPANDED_RETRIEVAL = "expanded_retrieval"
    MODEL_UPGRADE = "model_upgrade"
    DECOMPOSE_FURTHER = "decompose_further"
    INLINE_EXECUTION = "inline_execution"
    SECTION_REWRITE = "section_rewrite"
    FULL_REGENERATION = "full_regeneration"
    NOOP = "noop"


class EscalationReason(StrEnum):
    PLAN_INFEASIBLE = "plan_infeasible"
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"
    LATENCY_BUDGET_EXCEEDED = "latency_budget_exceeded"
    REPAIR_BUDGET_EXCEEDED = "repair_budget_exceeded"
    LOOP_DETECTED = "loop_detected"
    MEMORY_UNAVAILABLE = "memory_unavailable"
    TOOL_FAILURE = "tool_failure"
    VERIFICATION_FAILURE = "verification_failure"
    CHECKPOINT_MISSING = "checkpoint_missing"
    TERMINAL_CHECKPOINT = "terminal_checkpoint"


__all__ = [
    "ActionStatus",
    "CognitiveMode",
    "EscalationReason",
    "LoopPhase",
    "MetacognitiveDecision",
    "RepairStrategy",
    "RootCauseClass",
    "SubsystemAvailability",
]
