"""Deterministic heuristics for ambiguity, complexity, rewriting, and routing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from openagentbench.agent_retrieval import (
    Modality,
    OutputStream,
    ProtocolType,
    QueryClassification,
    QueryType,
    RetrievalMode,
    SourceTable,
)
from openagentbench.agent_retrieval.scoring import count_tokens, lexical_overlap_score, tokenize

from .config import CognitiveHeuristicPolicy, PragmaticHeuristicPolicy, ToolAffordancePolicy
from .enums import (
    AmbiguityLevel,
    ClarificationMode,
    EmotionalTone,
    ExpertiseLevel,
    IntentClass,
    RouteTarget,
)
from .models import CognitiveComplexity, PragmaticProfile, RoutedSubQuery, ToolAffordanceSummary

CONSTRAINT_PATTERN = re.compile(r"\b(without|using|under|must|only|avoid|before|after)\b([^,.;]+)", re.IGNORECASE)
PRESUPPOSITION_PATTERN = re.compile(r"\b(again|continue|same|previous|earlier|already)\b", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class RoutedDecision:
    target: RouteTarget
    protocol: str
    tool_candidates: tuple[str, ...]
    retrieval_modes: tuple[RetrievalMode, ...]
    target_tables: tuple[SourceTable, ...]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_intent_class(
    query_text: str,
    classification: QueryClassification,
    *,
    selected_tools: Sequence[ToolAffordanceSummary],
) -> IntentClass:
    lowered = query_text.lower()
    if any(modality in classification.preferred_modalities for modality in (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO)):
        return IntentClass.VISUAL_ANALYSIS
    active_policy = PragmaticHeuristicPolicy()
    if any(token in lowered for token in active_policy.verification_hints):
        return IntentClass.VERIFICATION
    if any(token in lowered for token in active_policy.memory_hints):
        return IntentClass.META_COGNITIVE
    if any(token in lowered for token in active_policy.navigational_hints):
        return IntentClass.NAVIGATIONAL
    if any(token in lowered for token in active_policy.transactional_hints):
        return IntentClass.TRANSACTIONAL
    if any(token in lowered for token in active_policy.analytical_hints):
        return IntentClass.ANALYTICAL
    if any(token in lowered for token in active_policy.generative_hints):
        return IntentClass.GENERATIVE
    if selected_tools and classification.requires_decomposition:
        return IntentClass.ANALYTICAL
    return IntentClass.INFORMATIONAL


def extract_constraints(query_text: str) -> tuple[str, ...]:
    constraints = []
    for match in CONSTRAINT_PATTERN.finditer(query_text):
        phrase = " ".join(part.strip() for part in match.groups() if part.strip())
        if phrase:
            constraints.append(phrase)
    return tuple(dict.fromkeys(constraints))


def extract_presuppositions(query_text: str) -> tuple[str, ...]:
    matches = [match.group(0).lower() for match in PRESUPPOSITION_PATTERN.finditer(query_text)]
    return tuple(dict.fromkeys(matches))


def analyze_pragmatics(
    query_text: str,
    *,
    conversation: Sequence[str],
    policy: PragmaticHeuristicPolicy | None = None,
) -> PragmaticProfile:
    active_policy = policy or PragmaticHeuristicPolicy()
    lowered = query_text.strip().lower()
    tokens = tokenize(query_text)
    presuppositions = extract_presuppositions(query_text)
    constraints = extract_constraints(query_text)

    missing_slots: list[str] = []
    if tokens and tokens[0] in active_policy.ambiguous_referents:
        missing_slots.append("referent")
    if lowered.startswith(active_policy.followup_action_prefixes) and len(tokens) <= 3:
        missing_slots.append("target")
    if "best" in tokens and all(token not in tokens for token in active_policy.selection_criteria_markers):
        missing_slots.append("selection_criteria")

    coreference_risk = 0.0
    if conversation and tokens and tokens[0] in active_policy.ambiguous_referents:
        coreference_risk = active_policy.high_coreference_risk
    elif conversation and presuppositions:
        coreference_risk = active_policy.medium_coreference_risk

    if len(missing_slots) >= 2 or coreference_risk >= active_policy.required_clarification_risk_threshold:
        ambiguity = AmbiguityLevel.HIGH
        mode = ClarificationMode.REQUIRED
    elif missing_slots or presuppositions:
        ambiguity = AmbiguityLevel.MODERATE
        mode = ClarificationMode.OPTIONAL
    else:
        ambiguity = AmbiguityLevel.LOW
        mode = ClarificationMode.NONE

    behavioral_signals = infer_behavioral_signals(query_text, policy=active_policy)
    latent_goals = infer_latent_goals(query_text, policy=active_policy)

    return PragmaticProfile(
        ambiguity_level=ambiguity,
        clarification_mode=mode,
        presuppositions=presuppositions,
        implied_constraints=constraints,
        missing_slots=tuple(missing_slots),
        coreference_risk=coreference_risk,
        behavioral_signals=behavioral_signals,
        latent_goals=latent_goals,
    )


def infer_expertise_level(query_text: str, *, policy: PragmaticHeuristicPolicy | None = None) -> ExpertiseLevel:
    active_policy = policy or PragmaticHeuristicPolicy()
    lowered = query_text.lower()
    if any(token in lowered for token in active_policy.expert_hints):
        return ExpertiseLevel.EXPERT
    if any(token in lowered for token in active_policy.novice_hints):
        return ExpertiseLevel.NOVICE
    return ExpertiseLevel.INTERMEDIATE


def infer_emotional_tone(query_text: str, *, policy: PragmaticHeuristicPolicy | None = None) -> EmotionalTone:
    active_policy = policy or PragmaticHeuristicPolicy()
    lowered = query_text.lower()
    if any(token in lowered for token in active_policy.urgent_hints):
        return EmotionalTone.URGENT
    if any(token in lowered for token in active_policy.frustrated_hints):
        return EmotionalTone.FRUSTRATED
    if any(token in lowered for token in active_policy.confused_hints):
        return EmotionalTone.CONFUSED
    if query_text.strip().endswith("?"):
        return EmotionalTone.EXPLORATORY
    return EmotionalTone.NEUTRAL


def infer_behavioral_signals(
    query_text: str,
    *,
    policy: PragmaticHeuristicPolicy | None = None,
) -> tuple[str, ...]:
    active_policy = policy or PragmaticHeuristicPolicy()
    lowered = query_text.lower()
    signals: list[str] = []
    if any(token in lowered for token in active_policy.control_environment_hints):
        signals.append("control_environment")
    if any(token in lowered for token in active_policy.stress_hints):
        signals.append("stress_load")
    if any(token in lowered for token in active_policy.frustrated_hints):
        signals.append("frustration")
    if any(token in lowered for token in active_policy.confused_hints):
        signals.append("uncertainty")
    if any(token in lowered for token in active_policy.urgent_hints):
        signals.append("time_pressure")
    return tuple(signals)


def infer_latent_goals(
    query_text: str,
    *,
    policy: PragmaticHeuristicPolicy | None = None,
) -> tuple[str, ...]:
    active_policy = policy or PragmaticHeuristicPolicy()
    lowered = query_text.lower()
    goals: list[str] = []
    if any(token in lowered for token in active_policy.protection_goal_hints):
        goals.append("risk_reduction")
    if any(token in lowered for token in active_policy.autonomy_goal_hints):
        goals.append("autonomy_preservation")
    if any(token in lowered for token in active_policy.resolution_goal_hints):
        goals.append("task_resolution")
    if "understand" in lowered or "analyze" in lowered:
        goals.append("situation_modeling")
    return tuple(dict.fromkeys(goals))


def estimate_cognitive_complexity(
    query_text: str,
    *,
    classification: QueryClassification,
    selected_tool_count: int,
    memory_message_count: int,
    policy: CognitiveHeuristicPolicy | None = None,
) -> CognitiveComplexity:
    active_policy = policy or CognitiveHeuristicPolicy()
    token_count = count_tokens(query_text)
    complexity = min(
        1.0,
        (token_count / active_policy.token_normalizer)
        + (active_policy.decomposition_bonus if classification.requires_decomposition else 0.0),
    )
    complexity += active_policy.tool_bonus * min(selected_tool_count, active_policy.tool_count_cap)
    complexity += active_policy.memory_bonus * min(memory_message_count, active_policy.memory_count_cap)
    if classification.type in {QueryType.AGENTIC, QueryType.CODE, QueryType.MULTIMODAL}:
        complexity += active_policy.advanced_query_bonus
    if OutputStream.STRUCTURED_DATA in classification.output_streams:
        complexity += active_policy.structured_output_bonus
    complexity = min(complexity, 1.0)
    estimated_steps = max(1, min(active_policy.max_estimated_steps, 1 + int(complexity * 5)))
    return CognitiveComplexity(
        complexity_score=complexity,
        requires_decomposition=classification.requires_decomposition or complexity >= active_policy.decomposition_threshold,
        estimated_steps=estimated_steps,
        expertise_level=infer_expertise_level(query_text),
        emotional_tone=infer_emotional_tone(query_text),
    )


def build_search_query(*, resolved_query: str, constraints: Sequence[str], session_topic: str) -> str:
    parts = [resolved_query.strip()]
    if session_topic.strip():
        parts.append(session_topic.strip())
    if constraints:
        parts.append(" ".join(constraints))
    normalized = " ".join(part for part in parts if part).strip()
    return re.sub(r"\s+", " ", normalized)


def rank_tool_affordances(
    query_text: str,
    *,
    tool_affordances: Sequence[ToolAffordanceSummary],
    token_budget: int,
    policy: ToolAffordancePolicy | None = None,
) -> tuple[ToolAffordanceSummary, ...]:
    active_policy = policy or ToolAffordancePolicy()
    if token_budget <= 0:
        return ()

    ranked: list[ToolAffordanceSummary] = []
    for tool in tool_affordances:
        lexical = lexical_overlap_score(query_text, tool.description + " " + tool.tool_id.replace("_", " "))
        score = min(
            1.0,
            active_policy.prior_relevance_weight * tool.relevance_score
            + active_policy.lexical_relevance_weight * lexical,
        )
        ranked.append(
            ToolAffordanceSummary(
                tool_id=tool.tool_id,
                description=tool.description,
                protocol=tool.protocol,
                token_cost_estimate=tool.token_cost_estimate,
                relevance_score=score,
                required_scopes=tool.required_scopes,
                metadata=tool.metadata,
            )
        )
    ranked.sort(key=lambda item: (item.relevance_score / max(item.token_cost_estimate, 1), item.relevance_score), reverse=True)

    selected: list[ToolAffordanceSummary] = []
    used_tokens = 0
    for tool in ranked:
        if tool.relevance_score < active_policy.minimum_selected_relevance:
            continue
        if used_tokens + tool.token_cost_estimate > token_budget:
            continue
        selected.append(tool)
        used_tokens += tool.token_cost_estimate
    return tuple(selected)


def decompose_query_text(
    query_text: str,
    *,
    max_subqueries: int,
    policy: CognitiveHeuristicPolicy | None = None,
) -> tuple[str, ...]:
    active_policy = policy or CognitiveHeuristicPolicy()
    normalized = re.sub(r"\bthen\b|\balso\b", ",", query_text, flags=re.IGNORECASE)
    normalized = normalized.replace(" and ", ", ").replace(";", ",")
    parts = [
        part.strip(" ,")
        for part in normalized.split(",")
        if len(part.strip(" ,")) >= active_policy.minimum_subquery_length
    ]
    if not parts:
        return (query_text.strip(),)
    deduped = tuple(dict.fromkeys(parts))
    return deduped[:max_subqueries]


def infer_route(
    query_text: str,
    *,
    intent_class: IntentClass,
    classification: QueryClassification,
    selected_tools: Sequence[ToolAffordanceSummary],
    policy: ToolAffordancePolicy | None = None,
) -> RoutedDecision:
    active_policy = policy or ToolAffordancePolicy()
    lowered = query_text.lower()
    if any(keyword in lowered for keyword in active_policy.browser_keywords):
        tool_candidates = tuple(
            tool.tool_id
            for tool in selected_tools
            if tool.protocol in active_policy.browser_protocols or tool.tool_id in active_policy.browser_tool_ids
        )
        if tool_candidates:
            protocol = "browser" if "browser_navigate" in tool_candidates else "vision"
            return RoutedDecision(
                target=RouteTarget.TOOL,
                protocol=protocol,
                tool_candidates=tool_candidates,
                retrieval_modes=(RetrievalMode.TOOL_STATE_AUGMENTED, RetrievalMode.SEMANTIC),
                target_tables=(SourceTable.HISTORY,),
            )

    if any(keyword in lowered for keyword in active_policy.terminal_keywords):
        tool_candidates = tuple(
            tool.tool_id
            for tool in selected_tools
            if any(marker in tool.tool_id for marker in active_policy.terminal_tool_markers)
        )
        if tool_candidates:
            return RoutedDecision(
                target=RouteTarget.TOOL,
                protocol="function",
                tool_candidates=tool_candidates,
                retrieval_modes=(RetrievalMode.TOOL_STATE_AUGMENTED, RetrievalMode.EXACT),
                target_tables=(SourceTable.HISTORY, SourceTable.MEMORY),
            )

    if any(keyword in lowered for keyword in active_policy.memory_keywords):
        return RoutedDecision(
            target=RouteTarget.MEMORY,
            protocol="grpc",
            tool_candidates=tuple(tool.tool_id for tool in selected_tools if "memory" in tool.tool_id),
            retrieval_modes=(RetrievalMode.MEMORY_AUGMENTED, RetrievalMode.SEMANTIC),
            target_tables=(SourceTable.MEMORY, SourceTable.SESSION),
        )

    if intent_class is IntentClass.NAVIGATIONAL or any(token in lowered for token in active_policy.navigation_keywords):
        tool_candidates = tuple(
            tool.tool_id
            for tool in selected_tools
            if tool.protocol in active_policy.browser_protocols or tool.tool_id in active_policy.browser_tool_ids
        )
        if tool_candidates:
            protocol = "browser" if "browser_navigate" in tool_candidates else "vision"
            return RoutedDecision(
                target=RouteTarget.TOOL,
                protocol=protocol,
                tool_candidates=tool_candidates,
                retrieval_modes=(RetrievalMode.TOOL_STATE_AUGMENTED, RetrievalMode.SEMANTIC),
                target_tables=(SourceTable.HISTORY,),
            )

    if intent_class in {IntentClass.TRANSACTIONAL, IntentClass.VERIFICATION}:
        tool_candidates = tuple(
            tool.tool_id for tool in selected_tools if tool.protocol in active_policy.transactional_protocols
        )
        if tool_candidates:
            return RoutedDecision(
                target=RouteTarget.TOOL,
                protocol=selected_tools[0].protocol if selected_tools else "function",
                tool_candidates=tool_candidates,
                retrieval_modes=(RetrievalMode.TOOL_STATE_AUGMENTED, RetrievalMode.EXACT),
                target_tables=(SourceTable.MEMORY, SourceTable.HISTORY),
            )

    if intent_class is IntentClass.ANALYTICAL and any(tool.tool_id == "a2a_delegate" for tool in selected_tools):
        return RoutedDecision(
            target=RouteTarget.DELEGATION,
            protocol="a2a",
            tool_candidates=("a2a_delegate",),
            retrieval_modes=(RetrievalMode.MULTI_HOP, RetrievalMode.SEMANTIC),
            target_tables=(SourceTable.HISTORY, SourceTable.MEMORY),
        )

    if classification.type is QueryType.MULTIMODAL:
        return RoutedDecision(
            target=RouteTarget.RETRIEVAL,
            protocol="openai-compatible",
            tool_candidates=(),
            retrieval_modes=(RetrievalMode.SEMANTIC, RetrievalMode.EXACT),
            target_tables=(SourceTable.HISTORY, SourceTable.MEMORY),
        )

    return RoutedDecision(
        target=RouteTarget.RETRIEVAL,
        protocol="grpc",
        tool_candidates=(),
        retrieval_modes=(RetrievalMode.EXACT, RetrievalMode.SEMANTIC),
        target_tables=(SourceTable.HISTORY, SourceTable.MEMORY),
    )


def build_routed_subqueries(
    *,
    fragments: Sequence[str],
    intent_class: IntentClass,
    classification: QueryClassification,
    selected_tools: Sequence[ToolAffordanceSummary],
    tool_policy: ToolAffordancePolicy | None = None,
    cognitive_policy: CognitiveHeuristicPolicy | None = None,
) -> tuple[RoutedSubQuery, ...]:
    active_cognitive_policy = cognitive_policy or CognitiveHeuristicPolicy()
    routed: list[RoutedSubQuery] = []
    for index, fragment in enumerate(fragments):
        decision = infer_route(
            fragment,
            intent_class=intent_class,
            classification=classification,
            selected_tools=selected_tools,
            policy=tool_policy,
        )
        routed.append(
            RoutedSubQuery(
                step_id=f"sq_{index + 1}",
                text=fragment,
                route_target=decision.target,
                protocol=decision.protocol,
                priority=max(
                    active_cognitive_policy.priority_floor,
                    1.0 - (index * active_cognitive_policy.priority_decay),
                ),
                dependencies=((f"sq_{index}",) if index > 0 else ()),
                tool_candidates=decision.tool_candidates,
                retrieval_modes=decision.retrieval_modes,
                target_tables=decision.target_tables,
            )
        )
    return tuple(routed)


def summarize_history(history_lines: Iterable[str], *, max_lines: int = 4) -> tuple[str, ...]:
    filtered = [line.strip() for line in history_lines if line.strip()]
    if not filtered:
        return ()
    return tuple(filtered[-max_lines:])


__all__ = [
    "analyze_pragmatics",
    "build_routed_subqueries",
    "build_search_query",
    "decompose_query_text",
    "estimate_cognitive_complexity",
    "extract_constraints",
    "extract_presuppositions",
    "infer_behavioral_signals",
    "infer_emotional_tone",
    "infer_expertise_level",
    "infer_intent_class",
    "infer_latent_goals",
    "rank_tool_affordances",
    "stable_hash",
    "summarize_history",
]
