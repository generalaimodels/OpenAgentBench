"""End-to-end query understanding engine with memory, tool, and routing integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from openagentbench.agent_data import HistoryRecord, MemoryRecord
from openagentbench.agent_retrieval import ModelCapabilityProfile, ModelRouter, default_profiles
from openagentbench.agent_retrieval.scoring import classify_query
from openagentbench.agent_memory import WorkingMemoryItem

from .compiler import QueryContextAssembler
from .config import DEFAULT_QUERY_CACHE_TTL_SECONDS, QueryModuleConfig
from .enums import ClarificationMode
from .models import (
    ClarificationQuestion,
    IntentHypothesis,
    QueryCacheEntry,
    QueryResolutionPlan,
    QueryResolutionRequest,
    QueryResolutionResponse,
    QueryRewritePlan,
    new_query_audit_record,
)
from .providers import QueryProviderSuite
from .repository import InMemoryQueryRepository, QueryRepository
from .scoring import (
    analyze_pragmatics,
    build_routed_subqueries,
    build_search_query,
    decompose_query_text,
    estimate_cognitive_complexity,
    infer_intent_class,
    stable_hash,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class QueryResolver:
    config: QueryModuleConfig = field(default_factory=QueryModuleConfig)
    context_assembler: QueryContextAssembler = field(default_factory=QueryContextAssembler)
    providers: QueryProviderSuite = field(default_factory=QueryProviderSuite)
    repository: QueryRepository = field(default_factory=InMemoryQueryRepository)
    model_router: ModelRouter = field(default_factory=ModelRouter)
    model_profiles: Sequence[ModelCapabilityProfile] = field(default_factory=default_profiles)
    cache_ttl: timedelta = field(
        default_factory=lambda: timedelta(seconds=DEFAULT_QUERY_CACHE_TTL_SECONDS)
    )

    def __post_init__(self) -> None:
        self.context_assembler.config = self.config
        self.providers.policy = self.config.provider_policy
        default_ttl = timedelta(seconds=self.config.resolver_policy.cache_ttl_seconds)
        if self.cache_ttl == timedelta(seconds=DEFAULT_QUERY_CACHE_TTL_SECONDS):
            self.cache_ttl = default_ttl

    def resolve(
        self,
        request: QueryResolutionRequest,
        *,
        history: Sequence[HistoryRecord] = (),
        memories: Sequence[MemoryRecord] = (),
        working_items: Sequence[WorkingMemoryItem] = (),
        tools: Sequence[dict[str, Any]] = (),
    ) -> QueryResolutionResponse:
        started_at = _utc_now()
        cache_key = self._cache_key(request)
        cached = self.repository.get_cache_entry(cache_key)
        if cached is not None:
            latency_ms = self._latency_ms(started_at)
            audit = new_query_audit_record(
                user_id=request.user_id,
                session_id=request.session.session_id,
                cache_key=cache_key,
                ambiguity_level=cached.plan.pragmatic.ambiguity_level,
                subquery_count=len(cached.plan.subqueries),
                cache_hit=True,
                latency_ms=latency_ms,
            )
            self.repository.insert_audit_record(audit)
            return QueryResolutionResponse(
                plan=cached.plan,
                audit_id=audit.audit_id,
                cache_key=cache_key,
                cache_hit=True,
                latency_ms=latency_ms,
            )

        classification = classify_query(
            request.query_text,
            request.session.summary_text or "",
            turn_count=request.session.turn_count,
        )
        context, budget = self.context_assembler.assemble(
            request,
            history=history,
            memories=memories,
            working_items=working_items,
            tools=tools,
        )
        intent_class = infer_intent_class(
            request.query_text,
            classification,
            selected_tools=context.tool_affordances,
        )
        pragmatic = analyze_pragmatics(
            request.query_text,
            conversation=context.history_excerpt,
            policy=self.config.pragmatic_policy,
        )
        cognitive = estimate_cognitive_complexity(
            request.query_text,
            classification=classification,
            selected_tool_count=len(context.tool_affordances),
            memory_message_count=len(context.memory_messages),
            policy=self.config.cognitive_policy,
        )

        resolved_query = self.providers.planner.resolve_coreferences(request.query_text, context.history_excerpt)
        expanded_query = self.providers.expand(
            resolved_query,
            context=(*context.history_excerpt, context.session_topic),
            max_tokens=max(budget.rewrite_budget, self.config.resolver_policy.minimum_expand_tokens),
        )
        hypothetical_answer = self.providers.hypothetical_answer(
            resolved_query,
            context=(expanded_query, context.session_topic),
            max_tokens=max(budget.rewrite_budget, self.config.resolver_policy.minimum_hypothetical_tokens),
        )
        search_query = build_search_query(
            resolved_query=resolved_query,
            constraints=pragmatic.implied_constraints,
            session_topic=context.session_topic,
        )
        model_plan = self.model_router.select(classification, self.model_profiles)
        fragments = decompose_query_text(
            expanded_query if cognitive.requires_decomposition else resolved_query,
            max_subqueries=request.max_subqueries,
            policy=self.config.cognitive_policy,
        )
        subqueries = build_routed_subqueries(
            fragments=fragments,
            intent_class=intent_class,
            classification=classification,
            selected_tools=context.tool_affordances,
            tool_policy=self.config.tool_policy,
            cognitive_policy=self.config.cognitive_policy,
        )

        clarification_questions = self._clarification_questions(
            request.query_text,
            pragmatic=pragmatic,
            context=context,
        )
        plan = QueryResolutionPlan(
            original_query=request.query_text,
            context=context,
            budget=budget,
            intent=IntentHypothesis(
                intent_class=intent_class,
                confidence=max(self.config.resolver_policy.minimum_intent_confidence, 1.0 - pragmatic.coreference_risk),
                rationale=f"classified as {intent_class.value} from query features and protocol hints",
                preferred_protocols=classification.protocol_hints,
                relevant_tools=tuple(tool.tool_id for tool in context.tool_affordances),
                retrieval_classification=classification,
            ),
            pragmatic=pragmatic,
            cognitive=cognitive,
            rewrite=QueryRewritePlan(
                resolved_query=resolved_query,
                expanded_query=expanded_query,
                search_query=search_query,
                hypothetical_answer=hypothetical_answer,
                preserved_constraints=pragmatic.implied_constraints,
                provenance={
                    "history_turns": len(context.history_excerpt),
                    "tool_count": len(context.tool_affordances),
                    "memory_messages": len(context.memory_messages),
                },
            ),
            subqueries=subqueries,
            selected_model_plan=model_plan,
            needs_clarification=pragmatic.clarification_mode is ClarificationMode.REQUIRED,
            clarification_questions=clarification_questions,
        )

        self.repository.put_cache_entry(
            QueryCacheEntry(
                cache_key=cache_key,
                user_id=request.user_id,
                session_id=request.session.session_id,
                plan=plan,
                expires_at=_utc_now() + self.cache_ttl,
            )
        )
        latency_ms = self._latency_ms(started_at)
        audit = new_query_audit_record(
            user_id=request.user_id,
            session_id=request.session.session_id,
            cache_key=cache_key,
            ambiguity_level=pragmatic.ambiguity_level,
            subquery_count=len(subqueries),
            cache_hit=False,
            latency_ms=latency_ms,
        )
        self.repository.insert_audit_record(audit)
        return QueryResolutionResponse(
            plan=plan,
            audit_id=audit.audit_id,
            cache_key=cache_key,
            cache_hit=False,
            latency_ms=latency_ms,
        )

    def _cache_key(self, request: QueryResolutionRequest) -> str:
        base = request.idempotency_key or f"{request.user_id}:{request.session.session_id}:{request.query_text}"
        return stable_hash(base)

    def _clarification_questions(
        self,
        query_text: str,
        *,
        pragmatic,
        context,
    ) -> tuple[ClarificationQuestion, ...]:
        if pragmatic.clarification_mode is ClarificationMode.NONE:
            return ()
        questions: list[ClarificationQuestion] = []
        if pragmatic.missing_slots:
            question_text = self.providers.clarification(
                query_text,
                context=(*context.history_excerpt, context.session_topic),
            )
            questions.append(
                ClarificationQuestion(
                    question=question_text,
                    reason="missing required referent or selection criteria",
                    blocking=pragmatic.clarification_mode is ClarificationMode.REQUIRED,
                )
            )
        if pragmatic.presuppositions:
            questions.append(
                ClarificationQuestion(
                    question=self.config.resolver_policy.continuation_question,
                    reason="the request assumes prior context that may be ambiguous",
                    blocking=pragmatic.clarification_mode is ClarificationMode.REQUIRED,
                )
            )
        return tuple(questions)

    def _latency_ms(self, started_at: datetime) -> int:
        delta = _utc_now() - started_at
        return max(int(delta.total_seconds() * 1000), 0)


__all__ = ["QueryResolver"]
