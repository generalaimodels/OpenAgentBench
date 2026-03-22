"""Enumerations for the query understanding and resolution pipeline."""

from __future__ import annotations

from enum import Enum


class QueryStage(str, Enum):
    CONTEXT_ASSEMBLY = "context_assembly"
    INTENT_CLASSIFICATION = "intent_classification"
    PRAGMATIC_ANALYSIS = "pragmatic_analysis"
    COGNITIVE_ESTIMATION = "cognitive_estimation"
    REWRITE = "rewrite"
    DECOMPOSITION = "decomposition"
    ROUTING = "routing"
    CLARIFICATION = "clarification"


class IntentClass(str, Enum):
    INFORMATIONAL = "informational"
    ANALYTICAL = "analytical"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    GENERATIVE = "generative"
    VERIFICATION = "verification"
    META_COGNITIVE = "meta_cognitive"
    VISUAL_ANALYSIS = "visual_analysis"


class AmbiguityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ClarificationMode(str, Enum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class RouteTarget(str, Enum):
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    TOOL = "tool"
    DELEGATION = "delegation"
    SYNTHESIS = "synthesis"


class ExpertiseLevel(str, Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class EmotionalTone(str, Enum):
    NEUTRAL = "neutral"
    URGENT = "urgent"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    EXPLORATORY = "exploratory"


class QueryErrorCode(str, Enum):
    INTENT_AMBIGUOUS = "INTENT_AMBIGUOUS"
    PRESUPPOSITION_VIOLATED = "PRESUPPOSITION_VIOLATED"
    DECOMPOSITION_OVERFLOW = "DECOMPOSITION_OVERFLOW"
    ROUTING_UNAVAILABLE = "ROUTING_UNAVAILABLE"
    TOKEN_BUDGET_EXCEEDED = "TOKEN_BUDGET_EXCEEDED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    MEMORY_READ_FAILURE = "MEMORY_READ_FAILURE"
    TOOL_DISCOVERY_FAILURE = "TOOL_DISCOVERY_FAILURE"


__all__ = [
    "AmbiguityLevel",
    "ClarificationMode",
    "EmotionalTone",
    "ExpertiseLevel",
    "IntentClass",
    "QueryErrorCode",
    "QueryStage",
    "RouteTarget",
]
