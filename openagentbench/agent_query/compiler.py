"""Context assembly for query understanding with memory and tool awareness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from openagentbench.agent_context import ContextCompileRequest, compile_context
from openagentbench.agent_data import HistoryRecord, MemoryRecord
from openagentbench.agent_memory import MemoryCompileRequest, MemoryContextCompiler, WorkingMemoryItem
from openagentbench.agent_retrieval import classify_query
from openagentbench.agent_retrieval.scoring import count_tokens, extract_topic_trajectory
from openagentbench.agent_tools.catalog import build_default_tool_definitions

from .budgeting import allocate_query_understanding_budget
from .config import QueryModuleConfig, ToolAffordancePolicy
from .models import (
    ContextSourceRecord,
    QueryBudgetAllocation,
    QueryContextArtifact,
    QueryResolutionRequest,
    ToolAffordanceSummary,
)
from .scoring import rank_tool_affordances, summarize_history


def _history_line(record: HistoryRecord) -> str:
    content = record.content or ""
    if not content and record.content_parts:
        parts: list[str] = []
        for item in record.content_parts:
            if not isinstance(item, dict):
                continue
            for key in ("text", "image_url", "audio_url", "video_url"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    parts.append(value)
        content = " ".join(parts)
    return f"{record.role.value}: {content}".strip()


def _estimate_tool_tokens(description: str, parameters: dict[str, Any], *, policy: ToolAffordancePolicy) -> int:
    property_count = len(parameters.get("properties", {}))
    return max(policy.minimum_token_floor, count_tokens(description) + (property_count * policy.per_property_token_cost))


def _tool_protocol(name: str, *, policy: ToolAffordancePolicy) -> str:
    if name.startswith(policy.browser_prefixes):
        return "browser"
    if name.startswith(policy.vision_prefixes):
        return "vision"
    if name.startswith(policy.delegation_prefixes):
        return "a2a"
    if name.startswith(policy.memory_prefixes):
        return "grpc"
    return "function"


def _normalize_tool(tool: dict[str, Any], *, policy: ToolAffordancePolicy) -> ToolAffordanceSummary:
    function = tool["function"]
    description = str(function["description"])
    parameters = dict(function["parameters"])
    return ToolAffordanceSummary(
        tool_id=str(function["name"]),
        description=description,
        protocol=_tool_protocol(str(function["name"]), policy=policy),
        token_cost_estimate=_estimate_tool_tokens(description, parameters, policy=policy),
        relevance_score=policy.default_relevance_score,
    )


@dataclass(slots=True)
class QueryContextAssembler:
    memory_compiler: MemoryContextCompiler = field(default_factory=MemoryContextCompiler)
    config: QueryModuleConfig = field(default_factory=QueryModuleConfig)
    tool_definition_factory: Callable[[], Sequence[dict[str, Any]]] = build_default_tool_definitions

    def assemble(
        self,
        request: QueryResolutionRequest,
        *,
        history: Sequence[HistoryRecord] = (),
        memories: Sequence[MemoryRecord] = (),
        working_items: Sequence[WorkingMemoryItem] = (),
        tools: Sequence[dict[str, Any]] = (),
    ) -> tuple[QueryContextArtifact, QueryBudgetAllocation]:
        context_window = request.context_window_size or request.session.context_window_size
        budget = allocate_query_understanding_budget(
            context_window_size=context_window,
            policy=self.config.budget_policy,
        )
        classification = classify_query(
            request.query_text,
            request.session.summary_text or "",
            turn_count=request.session.turn_count,
        )

        tool_definitions = tuple(tools) or tuple(self.tool_definition_factory())
        normalized_tools = tuple(_normalize_tool(tool, policy=self.config.tool_policy) for tool in tool_definitions)
        selected_tools = rank_tool_affordances(
            request.query_text,
            tool_affordances=normalized_tools,
            token_budget=min(request.tool_token_budget, budget.routing_budget + budget.decomposition_budget),
            policy=self.config.tool_policy,
        )

        compiled_context = compile_context(
            ContextCompileRequest(
                user_id=request.user_id,
                session=request.session,
                query_text=request.query_text,
                history=tuple(history),
                memories=tuple(memories),
                working_items=tuple(working_items),
                active_tools=tool_definitions,
                provider="openai_responses",
                total_budget=context_window,
                response_reserve=request.session.max_response_tokens,
                tool_budget=min(request.tool_token_budget, budget.routing_budget + budget.decomposition_budget),
                memory_budget=max(budget.context_budget - min(request.tool_token_budget, budget.routing_budget), 0),
                metadata={
                    "objective": request.query_text,
                    "intent": classification.type.value,
                    "subqueries": request.max_subqueries,
                },
            ),
            classification=classification,
        )

        history_excerpt = summarize_history(_history_line(record) for record in history)
        history_tokens = sum(count_tokens(line) for line in history_excerpt)
        tool_tokens = sum(tool.token_cost_estimate for tool in selected_tools)
        token_accounting = {
            "query": count_tokens(request.query_text),
            "history": history_tokens,
            "memory": compiled_context.memory_projection.total_tokens,
            "tools": tool_tokens,
            "context": compiled_context.total_tokens,
        }
        consumed = sum(token_accounting.values())
        remaining = max(budget.total_budget - consumed, 0)
        session_topic = extract_topic_trajectory(history_excerpt) or (request.session.summary_text or "")

        provenance = (
            ContextSourceRecord(source="query", token_count=token_accounting["query"], detail="raw_query"),
            ContextSourceRecord(source="history", token_count=history_tokens, detail=f"turns={len(history_excerpt)}"),
            ContextSourceRecord(
                source="memory",
                token_count=compiled_context.memory_projection.total_tokens,
                detail=f"messages={len(compiled_context.memory_projection.messages)}",
            ),
            ContextSourceRecord(source="tools", token_count=tool_tokens, detail=f"selected={len(selected_tools)}"),
            ContextSourceRecord(source="context", token_count=compiled_context.total_tokens, detail=compiled_context.provider_profile.provider),
        )

        return (
            QueryContextArtifact(
                query_text=request.query_text,
                session_topic=session_topic,
                history_excerpt=history_excerpt,
                memory_messages=compiled_context.memory_projection.messages,
                tool_affordances=selected_tools,
                token_accounting=token_accounting,
                provenance=provenance,
                remaining_budget=remaining,
                compiled_context=compiled_context,
                invariant_report=compiled_context.invariant_report,
                compilation_trace=compiled_context.trace,
                archive_entry=compiled_context.archive_entry,
            ),
            budget,
        )


__all__ = ["QueryContextAssembler"]
