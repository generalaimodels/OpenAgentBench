"""Orchestration engine for the agent-loop module."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import timedelta
from time import perf_counter_ns
from typing import Any, Sequence
from uuid import UUID, uuid4

from openagentbench.agent_data import HistoryRecord, MemoryRecord, MemoryScope, MemoryTier, SessionRecord
from openagentbench.agent_query import QueryResolutionRequest, QueryResolver, RouteTarget
from openagentbench.agent_retrieval import (
    AuthorityTier,
    HistoryEntry,
    HumanFeedback,
    HybridRetrievalEngine,
    InMemoryRetrievalRepository,
    MemoryEntry,
    MemoryType,
    ProvenanceTaggedFragment,
    QueryType,
    SessionTurn,
    TaskOutcome,
)
from openagentbench.agent_retrieval.enums import Role, SourceTable
from openagentbench.agent_tools import (
    ExecutionContext,
    MutationClass,
    OpenAgentBenchToolSuite,
    ToolDescriptor,
    ToolExecutionEngine,
    ToolInvocationRequest,
)
from openagentbench.agent_tools.catalog import build_default_tool_definitions
from openagentbench.agent_tools.enums import InvocationStatus
from openagentbench.agent_tools.models import ToolInvocationResponse
from openagentbench.agent_memory import WorkingMemoryItem
from openagentbench.agent_retrieval.scoring import count_tokens

from .enums import (
    ActionStatus,
    CognitiveMode,
    EscalationReason,
    LoopPhase,
    MetacognitiveDecision,
    RepairStrategy,
    RootCauseClass,
    SubsystemAvailability,
)
from .models import (
    ActionOutcome,
    CommittedMemoryWrite,
    DeferredMemoryWrite,
    EvidenceBundle,
    EvidenceItem,
    LoopAction,
    LoopExecutionRequest,
    LoopExecutionResult,
    LoopExecutionState,
    LoopPhaseBudget,
    LoopPhaseSpan,
    LoopPlan,
    LoopPolicy,
    MetacognitiveAssessment,
    PredictedResources,
    QualityVector,
    RepairDecision,
    VerificationDefect,
    VerificationVerdict,
    new_checkpoint,
    new_loop_audit_record,
)
from .repository import InMemoryLoopRepository, LoopRepository


def _coalesce_text(record: HistoryRecord) -> str:
    if record.content:
        return record.content
    parts: list[str] = []
    for item in record.content_parts or ():
        if not isinstance(item, dict):
            continue
        for key in ("text", "image_url", "audio_url", "video_url"):
            value = item.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
    return " ".join(parts)


def _extract_url(text: str) -> str | None:
    for token in text.split():
        if token.startswith(("http://", "https://")):
            return token.strip("()[]{}<>,.")
    return None


def _memory_type_for_record(memory: MemoryRecord) -> MemoryType:
    if memory.memory_tier is MemoryTier.PROCEDURAL:
        return MemoryType.PROCEDURE
    if memory.memory_tier is MemoryTier.SEMANTIC:
        return MemoryType.FACT
    if "constraint" in " ".join(memory.tags).lower():
        return MemoryType.CONSTRAINT
    if "preference" in " ".join(memory.tags).lower():
        return MemoryType.PREFERENCE
    if "correction" in " ".join(memory.tags).lower():
        return MemoryType.CORRECTION
    return MemoryType.FACT


def _authority_for_memory(memory: MemoryRecord) -> AuthorityTier:
    if memory.memory_scope is MemoryScope.GLOBAL and memory.is_validated:
        return AuthorityTier.CANONICAL
    if memory.is_validated:
        return AuthorityTier.CURATED
    return AuthorityTier.DERIVED


def _history_entry_from_record(record: HistoryRecord) -> HistoryEntry:
    content = _coalesce_text(record)
    return HistoryEntry(
        history_id=record.message_id,
        uu_id=record.user_id,
        query_text=content,
        query_embedding=None,
        response_summary=content,
        evidence_used=(),
        task_outcome=TaskOutcome.UNKNOWN,
        human_feedback=HumanFeedback.NONE,
        utility_score=0.5,
        negative_flag=False,
        tags=("history", record.role.value),
        metadata={"turn_index": record.turn_index},
        created_at=record.created_at,
        session_origin=record.session_id,
    )


def _session_turn_from_record(record: HistoryRecord) -> SessionTurn:
    role = Role.USER
    if record.role.value == "assistant":
        role = Role.ASSISTANT
    elif record.role.value == "system":
        role = Role.SYSTEM
    elif record.role.value == "tool":
        role = Role.TOOL
    return SessionTurn(
        session_id=record.session_id,
        uu_id=record.user_id,
        turn_index=record.turn_index,
        role=role,
        content_text=_coalesce_text(record),
        created_at=record.created_at,
        tokens_used=record.token_count,
        tool_calls=record.tool_calls,
        metadata=dict(record.metadata),
        expires_at=None,
    )


def _memory_entry_from_record(memory: MemoryRecord) -> MemoryEntry:
    return MemoryEntry(
        memory_id=memory.memory_id,
        uu_id=memory.user_id,
        memory_type=_memory_type_for_record(memory),
        content_text=memory.content_text,
        content_embedding=None,
        authority_tier=_authority_for_memory(memory),
        confidence=memory.confidence,
        source_provenance={"tier": memory.memory_tier.value, "scope": memory.memory_scope.value},
        verified_by=(),
        supersedes=(),
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        expires_at=memory.expires_at,
        access_count=memory.access_count,
        last_accessed_at=memory.last_accessed_at,
        content_hash=memory.content_hash,
        metadata=dict(memory.metadata),
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(slots=True)
class AgentLoopEngine:
    policy: LoopPolicy = field(default_factory=LoopPolicy)
    query_resolver: QueryResolver = field(default_factory=QueryResolver)
    repository: LoopRepository = field(default_factory=InMemoryLoopRepository)

    def execute(
        self,
        request: LoopExecutionRequest,
        *,
        history: Sequence[HistoryRecord] = (),
        memories: Sequence[MemoryRecord] = (),
        working_items: Sequence[WorkingMemoryItem] = (),
    ) -> LoopExecutionResult:
        mutable_memories = memories if isinstance(memories, list) else list(memories)
        state = LoopExecutionState(request=request)
        return self._run(state, history=history, memories=mutable_memories, working_items=working_items)

    def resume(
        self,
        loop_id: UUID,
        *,
        history: Sequence[HistoryRecord] = (),
        memories: Sequence[MemoryRecord] = (),
        working_items: Sequence[WorkingMemoryItem] = (),
        stop_after_phase: LoopPhase | None = None,
    ) -> LoopExecutionResult:
        checkpoint = self.repository.latest_checkpoint(loop_id)
        if checkpoint is None:
            raise LookupError(EscalationReason.CHECKPOINT_MISSING.value)
        state = checkpoint.state_snapshot
        if state.current_phase in {LoopPhase.HALT, LoopPhase.FAIL}:
            return self._finalize(state, paused=False, success=state.current_phase is LoopPhase.HALT)
        state.request = replace(state.request, stop_after_phase=stop_after_phase)
        mutable_memories = memories if isinstance(memories, list) else list(memories)
        return self._run(state, history=history, memories=mutable_memories, working_items=working_items)

    def _run(
        self,
        state: LoopExecutionState,
        *,
        history: Sequence[HistoryRecord],
        memories: list[MemoryRecord],
        working_items: Sequence[WorkingMemoryItem],
    ) -> LoopExecutionResult:
        tool_engine = self._build_tool_engine(state.request, history, memories, working_items)
        retrieval_engine = self._build_retrieval_engine(state.request.session, history, memories, state.request.scopes)

        while state.current_phase not in {LoopPhase.HALT, LoopPhase.FAIL}:
            if state.iteration > self.policy.max_iterations:
                state.escalated = True
                state.escalation_reason = EscalationReason.LOOP_DETECTED
                state.current_phase = LoopPhase.FAIL
                break

            phase = state.current_phase
            started_ns = perf_counter_ns()

            if phase is LoopPhase.CONTEXT_ASSEMBLE:
                self._context_assemble(state, history=history, memories=memories, working_items=working_items)
            elif phase is LoopPhase.PLAN:
                self._plan(state, tool_engine=tool_engine)
            elif phase is LoopPhase.DECOMPOSE:
                self._decompose(state)
            elif phase is LoopPhase.PREDICT:
                self._predict(state)
            elif phase is LoopPhase.RETRIEVE:
                self._retrieve(state, retrieval_engine=retrieval_engine)
            elif phase is LoopPhase.ACT:
                self._act(state, tool_engine=tool_engine)
            elif phase is LoopPhase.METACOGNITIVE_CHECK:
                self._metacognitive_check(state)
            elif phase is LoopPhase.VERIFY:
                self._verify(state)
            elif phase is LoopPhase.CRITIQUE:
                self._critique(state)
            elif phase is LoopPhase.REPAIR:
                self._repair(state, tool_engine=tool_engine, retrieval_engine=retrieval_engine)
            elif phase is LoopPhase.ESCALATE:
                self._escalate(state)
            elif phase is LoopPhase.COMMIT:
                self._commit(state, tool_engine=tool_engine, memories=memories)
            else:
                state.current_phase = LoopPhase.FAIL

            self._record_span(state, phase, started_ns)
            paused = self._should_pause(state, phase)
            self._checkpoint(state, paused=paused)
            if paused:
                return self._finalize(state, paused=True, success=False)

        return self._finalize(state, paused=False, success=state.current_phase is LoopPhase.HALT)

    def _context_assemble(
        self,
        state: LoopExecutionState,
        *,
        history: Sequence[HistoryRecord],
        memories: Sequence[MemoryRecord],
        working_items: Sequence[WorkingMemoryItem],
    ) -> None:
        request = state.request
        query_response = self.query_resolver.resolve(
            QueryResolutionRequest(
                user_id=request.user_id,
                session=request.session,
                query_text=request.query_text,
                idempotency_key=request.idempotency_key,
            ),
            history=history,
            memories=memories,
            working_items=working_items,
            tools=self._query_tool_definitions(),
        )
        state.query_response = query_response
        state.subsystem_status = {
            "agent_query": SubsystemAvailability.READY,
            "agent_memory": SubsystemAvailability.READY if memories or working_items else SubsystemAvailability.DEGRADED,
            "agent_tools": SubsystemAvailability.READY,
            "agent_retrieval": SubsystemAvailability.READY if history or memories else SubsystemAvailability.DEGRADED,
        }
        state.cognitive_mode = self._select_cognitive_mode(state)
        state.total_tokens_consumed += sum(query_response.plan.context.token_accounting.values())
        state.last_completed_phase = LoopPhase.CONTEXT_ASSEMBLE
        state.current_phase = LoopPhase.PLAN

    def _plan(self, state: LoopExecutionState, *, tool_engine: ToolExecutionEngine) -> None:
        query_response = state.query_response
        if query_response is None:
            state.escalated = True
            state.escalation_reason = EscalationReason.PLAN_INFEASIBLE
            state.current_phase = LoopPhase.FAIL
            return

        selected_subqueries = query_response.plan.subqueries
        if state.cognitive_mode is CognitiveMode.SYSTEM1_FAST and selected_subqueries:
            selected_subqueries = selected_subqueries[:1]

        actions: list[LoopAction] = []
        validation_issues: list[str] = []
        for subquery in selected_subqueries:
            tool_id, fallback_tool_ids, descriptor = self._bind_tool(subquery.text, subquery.route_target, subquery.tool_candidates, tool_engine)
            if subquery.route_target in {RouteTarget.MEMORY, RouteTarget.DELEGATION} and tool_id is None:
                validation_issues.append(f"missing dispatchable tool for {subquery.step_id}")
            actions.append(
                LoopAction(
                    action_id=subquery.step_id,
                    title=subquery.text[:64],
                    instruction=subquery.text,
                    route_target=subquery.route_target,
                    priority=subquery.priority,
                    dependencies=subquery.dependencies,
                    tool_id=tool_id,
                    protocol=descriptor.source_type.value if descriptor is not None else subquery.protocol,
                    fallback_tool_ids=fallback_tool_ids,
                    mutates_state=descriptor is not None and descriptor.mutation_class is not MutationClass.READ_ONLY,
                    requires_approval=descriptor is not None and descriptor.auth_contract.approval_required,
                    target_tables=subquery.target_tables,
                )
            )

        if not actions:
            actions.append(
                LoopAction(
                    action_id="sq_1",
                    title=state.request.query_text[:64],
                    instruction=query_response.plan.rewrite.resolved_query,
                    route_target=RouteTarget.RETRIEVAL,
                    priority=1.0,
                    target_tables=(SourceTable.MEMORY, SourceTable.HISTORY),
                )
            )

        state.plan = LoopPlan(
            actions=tuple(actions),
            constraints=query_response.plan.rewrite.preserved_constraints,
            validated=not validation_issues,
            validation_issues=tuple(validation_issues),
            rollback_required=any(action.mutates_state for action in actions),
            query_response=query_response,
        )
        if not state.plan.validated and state.cognitive_mode is CognitiveMode.SYSTEM2_DELIBERATIVE:
            state.escalated = True
            state.escalation_reason = EscalationReason.PLAN_INFEASIBLE
            state.current_phase = LoopPhase.ESCALATE
        else:
            state.last_completed_phase = LoopPhase.PLAN
            state.current_phase = LoopPhase.RETRIEVE if state.cognitive_mode is CognitiveMode.SYSTEM1_FAST else LoopPhase.DECOMPOSE

    def _decompose(self, state: LoopExecutionState) -> None:
        state.last_completed_phase = LoopPhase.DECOMPOSE
        state.current_phase = LoopPhase.PREDICT

    def _predict(self, state: LoopExecutionState) -> None:
        if state.plan is None:
            state.escalated = True
            state.escalation_reason = EscalationReason.PLAN_INFEASIBLE
            state.current_phase = LoopPhase.FAIL
            return
        total_budget = min(self.policy.total_token_budget, state.request.session.context_window_size)
        if state.cognitive_mode is CognitiveMode.SYSTEM1_FAST:
            total_budget = max(int(total_budget * self.policy.system1_budget_ratio), 1_024)
        reserve_budget = int(total_budget * self.policy.reserve_ratio)
        plan_budget = max(int(total_budget * 0.14), 256)
        retrieve_budget = max(int(total_budget * 0.22), 256)
        act_budget = max(int(total_budget * 0.20), 256)
        verify_budget = max(int(total_budget * 0.16), 256)
        critique_budget = max(int(total_budget * 0.09), 128)
        repair_budget = max(int(total_budget * 0.09), 128)
        consumed = plan_budget + retrieve_budget + act_budget + verify_budget + critique_budget + repair_budget
        reserve_budget = max(total_budget - consumed, reserve_budget)
        phase_budget = LoopPhaseBudget(
            total_budget=total_budget,
            plan_budget=plan_budget,
            retrieve_budget=retrieve_budget,
            act_budget=act_budget,
            verify_budget=verify_budget,
            critique_budget=critique_budget,
            repair_budget=repair_budget,
            reserve_budget=reserve_budget,
        )
        tool_calls = sum(1 for action in state.plan.actions if action.tool_id is not None)
        model_calls = 2 if state.cognitive_mode is CognitiveMode.SYSTEM2_DELIBERATIVE else 1
        estimated_tokens = sum(count_tokens(action.instruction) * 8 for action in state.plan.actions) + retrieve_budget // 2 + verify_budget // 3
        estimated_latency = len(state.plan.actions) * 180 + (600 if state.cognitive_mode is CognitiveMode.SYSTEM2_DELIBERATIVE else 120)
        state.predicted_resources = PredictedResources(
            total_tokens=estimated_tokens,
            estimated_cost=round(estimated_tokens / 100_000.0, 6),
            estimated_latency_ms=estimated_latency,
            tool_calls=tool_calls,
            model_calls=model_calls,
            phase_budget=phase_budget,
        )
        while state.predicted_resources.total_tokens > phase_budget.total_budget and len(state.plan.actions) > 1:
            pruned = list(state.plan.actions)
            pruned.pop()
            state.plan = LoopPlan(
                actions=tuple(pruned),
                constraints=state.plan.constraints,
                validated=state.plan.validated,
                validation_issues=state.plan.validation_issues,
                rollback_required=state.plan.rollback_required,
                query_response=state.plan.query_response,
            )
            estimated_tokens = sum(count_tokens(action.instruction) * 8 for action in state.plan.actions) + retrieve_budget // 2 + verify_budget // 3
            state.predicted_resources = PredictedResources(
                total_tokens=estimated_tokens,
                estimated_cost=round(estimated_tokens / 100_000.0, 6),
                estimated_latency_ms=max(len(state.plan.actions) * 180, 180),
                tool_calls=sum(1 for action in state.plan.actions if action.tool_id is not None),
                model_calls=model_calls,
                phase_budget=phase_budget,
            )
        state.last_completed_phase = LoopPhase.PREDICT
        state.current_phase = LoopPhase.RETRIEVE

    def _retrieve(self, state: LoopExecutionState, *, retrieval_engine: HybridRetrievalEngine) -> None:
        if state.plan is None:
            state.current_phase = LoopPhase.FAIL
            return
        items: list[EvidenceItem] = []
        sufficiency_by_action: dict[str, float] = {}
        trace_ids: list[str] = []
        phase_budget = state.predicted_resources.phase_budget if state.predicted_resources is not None else LoopPhaseBudget(
            total_budget=self.policy.total_token_budget,
            plan_budget=0,
            retrieve_budget=max(self.policy.total_token_budget // 4, 256),
            act_budget=0,
            verify_budget=0,
            critique_budget=0,
            repair_budget=0,
            reserve_budget=0,
        )
        per_action_budget = max(phase_budget.retrieve_budget // max(len(state.plan.actions), 1), 64)
        for action in state.plan.actions:
            response = retrieval_engine.retrieve(
                action.instruction,
                uu_id=state.request.user_id,
                session_id=state.request.session.session_id,
                token_budget=per_action_budget,
            )
            trace_ids.append(response.retrieval_trace_id)
            action_items = self._evidence_from_fragments(action.action_id, response.fragments)
            items.extend(action_items)
            sufficiency_by_action[action.action_id] = _clamp(
                (response.quality_assessment.score * 0.65) + (min(len(action_items), 3) / 3.0 * 0.35),
                0.0,
                1.0,
            )

        items.sort(key=lambda item: (item.score / max(item.token_count, 1), item.score), reverse=True)
        running_tokens = 0
        trimmed: list[EvidenceItem] = []
        for item in items:
            if running_tokens + item.token_count > phase_budget.retrieve_budget:
                continue
            trimmed.append(item)
            running_tokens += item.token_count
        quality_score = sum(sufficiency_by_action.values()) / max(len(sufficiency_by_action), 1)
        state.evidence = EvidenceBundle(
            items=tuple(trimmed),
            total_tokens=running_tokens,
            quality_score=quality_score,
            sufficiency_by_action=sufficiency_by_action,
            trace_ids=tuple(trace_ids),
        )
        state.total_tokens_consumed += running_tokens
        state.last_completed_phase = LoopPhase.RETRIEVE
        state.current_phase = LoopPhase.ACT

    def _act(self, state: LoopExecutionState, *, tool_engine: ToolExecutionEngine) -> None:
        if state.plan is None:
            state.current_phase = LoopPhase.FAIL
            return
        outcomes: list[ActionOutcome] = []
        for action in state.plan.actions:
            dependency_errors = [
                outcome
                for outcome in outcomes
                if outcome.action_id in action.dependencies and outcome.status is not ActionStatus.SUCCESS
            ]
            if dependency_errors:
                outcomes.append(
                    ActionOutcome(
                        action_id=action.action_id,
                        status=ActionStatus.SKIPPED,
                        tool_id=action.tool_id,
                        protocol=action.protocol,
                        output={"reason": "dependency_failed"},
                        latency_ns=0,
                        error_code="dependency_failed",
                        error_message="a prerequisite action did not succeed",
                    )
                )
                continue
            if action.tool_id is None:
                synthesized = self._synthesize_action_output(state, action)
                outcomes.append(
                    ActionOutcome(
                        action_id=action.action_id,
                        status=ActionStatus.SUCCESS,
                        tool_id=None,
                        protocol=action.protocol,
                        output=synthesized,
                        latency_ns=0,
                    )
                )
                continue
            outcome = self._dispatch_action(state, action, tool_engine=tool_engine)
            outcomes.append(outcome)
        state.action_outcomes = outcomes
        state.output_text = self._compose_output_text(state)
        self._capture_deferred_writes(state)
        state.last_completed_phase = LoopPhase.ACT
        state.current_phase = LoopPhase.METACOGNITIVE_CHECK

    def _metacognitive_check(self, state: LoopExecutionState) -> None:
        action_count = max(len(state.plan.actions) if state.plan is not None else 0, 1)
        tool_outcomes = [outcome for outcome in state.action_outcomes if outcome.tool_id is not None]
        successful_tool_calls = sum(1 for outcome in tool_outcomes if outcome.status is ActionStatus.SUCCESS)
        tool_success_rate = successful_tool_calls / max(len(tool_outcomes), 1) if tool_outcomes else 1.0
        evidence_coverage = 0.0
        if state.evidence is not None and state.evidence.sufficiency_by_action:
            evidence_coverage = sum(
                1.0 for score in state.evidence.sufficiency_by_action.values() if score >= self.policy.evidence_sufficiency_threshold
            ) / action_count
        plan_adherence = sum(1 for outcome in state.action_outcomes if outcome.status is not ActionStatus.SKIPPED) / action_count
        expected_complex = (
            state.query_response.plan.cognitive.complexity_score if state.query_response is not None else 0.5
        )
        complexity_match = 1.0 if (expected_complex < 0.5) == (state.cognitive_mode is CognitiveMode.SYSTEM1_FAST) else 0.7
        consistency = 1.0 if all(outcome.status is ActionStatus.SUCCESS for outcome in state.action_outcomes) else 0.72
        score = (
            0.26 * tool_success_rate
            + 0.28 * evidence_coverage
            + 0.20 * plan_adherence
            + 0.14 * complexity_match
            + 0.12 * consistency
        )
        if score >= self.policy.metacognitive_threshold:
            decision = MetacognitiveDecision.PROCEED_TO_VERIFY
        elif evidence_coverage < 0.5:
            decision = MetacognitiveDecision.RE_RETRIEVE
        elif tool_success_rate < 0.8:
            decision = MetacognitiveDecision.RE_EXECUTE
        else:
            decision = MetacognitiveDecision.EARLY_CRITIQUE
        if decision in {MetacognitiveDecision.RE_RETRIEVE, MetacognitiveDecision.RE_EXECUTE}:
            if state.metacognitive_retry_count >= 1:
                decision = MetacognitiveDecision.EARLY_CRITIQUE if tool_success_rate < 0.8 else MetacognitiveDecision.PROCEED_TO_VERIFY
            else:
                state.metacognitive_retry_count += 1
        else:
            state.metacognitive_retry_count = 0
        state.metacognitive = MetacognitiveAssessment(
            score=score,
            decision=decision,
            tool_success_rate=tool_success_rate,
            evidence_coverage=evidence_coverage,
            plan_adherence=plan_adherence,
            complexity_match=complexity_match,
            consistency=consistency,
        )
        state.last_completed_phase = LoopPhase.METACOGNITIVE_CHECK
        if decision is MetacognitiveDecision.RE_RETRIEVE and state.repair_count < self.policy.max_repairs:
            state.current_phase = LoopPhase.RETRIEVE
        elif decision is MetacognitiveDecision.RE_EXECUTE and state.repair_count < self.policy.max_repairs:
            state.current_phase = LoopPhase.ACT
        elif decision is MetacognitiveDecision.EARLY_CRITIQUE:
            state.current_phase = LoopPhase.CRITIQUE
        else:
            state.current_phase = LoopPhase.VERIFY

    def _verify(self, state: LoopExecutionState) -> None:
        state.iteration += 1
        action_count = max(len(state.plan.actions) if state.plan is not None else 0, 1)
        successful = sum(1 for outcome in state.action_outcomes if outcome.status is ActionStatus.SUCCESS)
        tool_outcomes = [outcome for outcome in state.action_outcomes if outcome.tool_id is not None]
        tool_success_rate = successful / action_count
        grounding_score = state.evidence.quality_score if state.evidence is not None else 0.0
        schema_valid = all(outcome.status is not ActionStatus.PENDING for outcome in state.action_outcomes)
        defects: list[VerificationDefect] = []
        if any(outcome.tool_id == "memory_write" for outcome in state.action_outcomes):
            defects.append(
                VerificationDefect(
                    defect_id=uuid4(),
                    root_cause=RootCauseClass.SAFETY_VIOLATION,
                    severity=1.0,
                    message="memory writes are only allowed during commit",
                    retryable=False,
                )
            )
        if not schema_valid:
            defects.append(
                VerificationDefect(
                    defect_id=uuid4(),
                    root_cause=RootCauseClass.SCHEMA_VIOLATION,
                    severity=0.75,
                    message="one or more actions remained pending",
                )
            )
        for outcome in state.action_outcomes:
            if outcome.status is ActionStatus.ERROR:
                defects.append(
                    VerificationDefect(
                        defect_id=uuid4(),
                        root_cause=RootCauseClass.TOOL_FAILURE,
                        severity=0.75,
                        message=outcome.error_message or "tool invocation failed",
                        action_id=outcome.action_id,
                    )
                )
            if outcome.status is ActionStatus.SKIPPED:
                defects.append(
                    VerificationDefect(
                        defect_id=uuid4(),
                        root_cause=RootCauseClass.INCOMPLETENESS,
                        severity=0.60,
                        message="action skipped due to earlier failure",
                        action_id=outcome.action_id,
                    )
                )
        if grounding_score < self.policy.evidence_sufficiency_threshold:
            defects.append(
                VerificationDefect(
                    defect_id=uuid4(),
                    root_cause=RootCauseClass.RETRIEVAL_GAP,
                    severity=0.68,
                    message="evidence coverage remained below the grounding threshold",
                )
            )
        correctness = _clamp(tool_success_rate - (0.12 if defects else 0.0), 0.0, 1.0)
        completeness = successful / action_count
        coherence = 1.0 if not defects else max(0.45, 1.0 - len(defects) * 0.12)
        safety = 1.0 if not any(defect.root_cause is RootCauseClass.SAFETY_VIOLATION for defect in defects) else 0.0
        grounded = grounding_score
        efficient = 1.0
        if state.predicted_resources is not None and state.predicted_resources.phase_budget.total_budget > 0:
            efficient = _clamp(
                1.0 - (state.total_tokens_consumed / state.predicted_resources.phase_budget.total_budget),
                0.0,
                1.0,
            )
        quality = QualityVector(
            correctness=correctness,
            completeness=completeness,
            coherence=coherence,
            safety=safety,
            grounded=grounded,
            efficient=efficient,
        )
        passed = (
            not defects
            and quality.mean_score() >= self.policy.quality_gate_threshold
            and quality.min_dimension() >= self.policy.minimum_dimension_threshold
        )
        state.verdict = VerificationVerdict(
            passed=passed,
            quality=quality,
            defects=tuple(defects),
            grounding_score=grounding_score,
            tool_success_rate=tool_success_rate,
            schema_valid=schema_valid,
        )
        state.last_completed_phase = LoopPhase.VERIFY
        if passed:
            state.current_phase = LoopPhase.COMMIT
            return
        if state.cognitive_mode is CognitiveMode.SYSTEM1_FAST and not state.upgraded_from_fast_path:
            state.cognitive_mode = CognitiveMode.SYSTEM2_DELIBERATIVE
            state.upgraded_from_fast_path = True
            state.current_phase = LoopPhase.PLAN
            return
        state.current_phase = LoopPhase.CRITIQUE

    def _critique(self, state: LoopExecutionState) -> None:
        if state.verdict is None or not state.verdict.defects:
            defect = VerificationDefect(
                defect_id=uuid4(),
                root_cause=RootCauseClass.INCOMPLETENESS,
                severity=0.5,
                message="verification did not produce an explicit diagnosis",
            )
            state.verdict = VerificationVerdict(
                passed=False,
                quality=state.verdict.quality if state.verdict is not None else QualityVector(0.5, 0.5, 0.5, 1.0, 0.5, 0.5),
                defects=(defect,),
                grounding_score=state.verdict.grounding_score if state.verdict is not None else 0.0,
                tool_success_rate=state.verdict.tool_success_rate if state.verdict is not None else 0.0,
                schema_valid=state.verdict.schema_valid if state.verdict is not None else False,
            )
        state.last_completed_phase = LoopPhase.CRITIQUE
        state.current_phase = LoopPhase.REPAIR

    def _repair(
        self,
        state: LoopExecutionState,
        *,
        tool_engine: ToolExecutionEngine,
        retrieval_engine: HybridRetrievalEngine,
    ) -> None:
        if state.verdict is None or not state.verdict.defects:
            state.escalated = True
            state.escalation_reason = EscalationReason.VERIFICATION_FAILURE
            state.current_phase = LoopPhase.ESCALATE
            return
        if state.repair_count >= self.policy.max_repairs:
            state.escalated = True
            state.escalation_reason = EscalationReason.REPAIR_BUDGET_EXCEEDED
            state.current_phase = LoopPhase.ESCALATE
            return
        defect = max(state.verdict.defects, key=lambda item: item.severity)
        decision = self._select_repair_strategy(state, defect)
        state.repair_history.append(decision)
        state.repair_count += 1
        if decision.strategy is RepairStrategy.TOOL_SUBSTITUTE and decision.target_action_id and decision.replacement_tool_id:
            action = next((item for item in state.plan.actions if item.action_id == decision.target_action_id), None)
            if action is not None:
                replacement = LoopAction(
                    action_id=action.action_id,
                    title=action.title,
                    instruction=action.instruction,
                    route_target=action.route_target,
                    priority=action.priority,
                    dependencies=action.dependencies,
                    tool_id=decision.replacement_tool_id,
                    protocol=tool_engine.registry.resolve(decision.replacement_tool_id).source_type.value,
                    fallback_tool_ids=tuple(tool_id for tool_id in action.fallback_tool_ids if tool_id != decision.replacement_tool_id),
                    mutates_state=action.mutates_state,
                    requires_approval=action.requires_approval,
                    target_tables=action.target_tables,
                )
                outcome = self._dispatch_action(state, replacement, tool_engine=tool_engine, force_fallback=True)
                self._replace_action_outcome(state, outcome)
        elif decision.strategy is RepairStrategy.EXPANDED_RETRIEVAL:
            original_actions = state.plan.actions if state.plan is not None else ()
            for action in original_actions:
                if decision.target_action_id is not None and action.action_id != decision.target_action_id:
                    continue
                query = decision.followup_query or f"{action.instruction} evidence detail"
                response = retrieval_engine.retrieve(
                    query,
                    uu_id=state.request.user_id,
                    session_id=state.request.session.session_id,
                    token_budget=max(self.policy.total_token_budget // 8, 128),
                )
                merged = list(state.evidence.items if state.evidence is not None else ())
                merged.extend(self._evidence_from_fragments(action.action_id, response.fragments))
                merged.sort(key=lambda item: item.score, reverse=True)
                state.evidence = EvidenceBundle(
                    items=tuple(merged),
                    total_tokens=sum(item.token_count for item in merged),
                    quality_score=max(state.evidence.quality_score if state.evidence is not None else 0.0, response.quality_assessment.score),
                    sufficiency_by_action={
                        **(dict(state.evidence.sufficiency_by_action) if state.evidence is not None else {}),
                        action.action_id: max(
                            response.quality_assessment.score,
                            dict(state.evidence.sufficiency_by_action).get(action.action_id, 0.0) if state.evidence is not None else 0.0,
                        ),
                    },
                    trace_ids=tuple((state.evidence.trace_ids if state.evidence is not None else ()) + (response.retrieval_trace_id,)),
                )
        else:
            target_id = defect.action_id
            if target_id is not None:
                action = next((item for item in state.plan.actions if item.action_id == target_id), None)
                if action is not None:
                    repaired = ActionOutcome(
                        action_id=action.action_id,
                        status=ActionStatus.SUCCESS,
                        tool_id=action.tool_id,
                        protocol=action.protocol,
                        output=self._synthesize_action_output(state, action, repair_hint=decision.rationale),
                        latency_ns=0,
                        used_fallback=True,
                    )
                    self._replace_action_outcome(state, repaired)
        state.output_text = self._compose_output_text(state)
        state.last_completed_phase = LoopPhase.REPAIR
        state.current_phase = LoopPhase.VERIFY

    def _escalate(self, state: LoopExecutionState) -> None:
        state.escalated = True
        if state.escalation_reason is None:
            state.escalation_reason = EscalationReason.VERIFICATION_FAILURE
        state.last_completed_phase = LoopPhase.ESCALATE
        state.current_phase = LoopPhase.FAIL

    def _commit(self, state: LoopExecutionState, *, tool_engine: ToolExecutionEngine, memories: list[MemoryRecord]) -> None:
        context = self._execution_context(state.request)
        existing_hashes = {memory.content_hash for memory in memories}
        committed: list[CommittedMemoryWrite] = []
        for candidate in state.deferred_writes:
            content_hash = self._content_hash(candidate.content)
            if content_hash in existing_hashes:
                continue
            response = tool_engine.dispatch(
                ToolInvocationRequest(
                    tool_id="memory_write",
                    params={
                        "content": candidate.content,
                        "target_layer": candidate.target_layer.name.lower(),
                        "target_scope": "global" if candidate.target_scope is MemoryScope.GLOBAL else "local",
                        "memory_type": candidate.memory_type.value,
                        "confidence": 0.9,
                    },
                    context=context,
                    idempotency_key=f"{state.request.loop_id}:{candidate.target_layer.value}:{content_hash.hex()}",
                )
            )
            if response.status is not InvocationStatus.SUCCESS or response.success is None:
                continue
            payload = response.success.data
            memory_id = str(payload.get("memory_id", ""))
            committed.append(
                CommittedMemoryWrite(
                    memory_id=memory_id,
                    target_layer=candidate.target_layer,
                    target_scope=candidate.target_scope,
                    content=candidate.content,
                    rationale=candidate.rationale,
                )
            )
            existing_hashes.add(content_hash)
        state.committed_writes = committed
        state.last_completed_phase = LoopPhase.COMMIT
        state.current_phase = LoopPhase.HALT

    def _build_tool_engine(
        self,
        request: LoopExecutionRequest,
        history: Sequence[HistoryRecord],
        memories: list[MemoryRecord],
        working_items: Sequence[WorkingMemoryItem],
    ) -> ToolExecutionEngine:
        engine = ToolExecutionEngine()
        OpenAgentBenchToolSuite(
            sessions={request.session.session_id: request.session},
            history_by_session={request.session.session_id: list(history)},
            memories_by_user={request.user_id: memories},
            working_by_session={(request.user_id, request.session.session_id): list(working_items)},
        ).register_into(engine)
        self._register_custom_tools(engine)
        return engine

    def _build_retrieval_engine(
        self,
        session: SessionRecord,
        history: Sequence[HistoryRecord],
        memories: Sequence[MemoryRecord],
        scopes: Sequence[str],
    ) -> HybridRetrievalEngine:
        repository = InMemoryRetrievalRepository(
            active_users={session.user_id},
            acl_by_user={session.user_id: tuple(scopes)},
            sessions={session.user_id: [_session_turn_from_record(record) for record in history]},
            history={session.user_id: [_history_entry_from_record(record) for record in history]},
            memory={session.user_id: [_memory_entry_from_record(memory) for memory in memories]},
        )
        return HybridRetrievalEngine(repository=repository)

    def _select_cognitive_mode(self, state: LoopExecutionState) -> CognitiveMode:
        if state.request.force_deliberative:
            return CognitiveMode.SYSTEM2_DELIBERATIVE
        if state.query_response is None:
            return CognitiveMode.SYSTEM2_DELIBERATIVE
        plan = state.query_response.plan
        requires_decomposition = plan.cognitive.requires_decomposition or len(plan.subqueries) > 1
        requires_reasoning = plan.cognitive.complexity_score >= self.policy.system1_complexity_threshold
        bound_tools = {
            tool_id
            for subquery in plan.subqueries
            for tool_id in subquery.tool_candidates
            if tool_id != "memory_write"
        }
        requires_multi_tool = len(bound_tools) > 2
        lowered = state.request.query_text.lower()
        involves_mutation = any(token in lowered for token in ("create", "update", "delete", "write", "send", "purchase"))
        if (
            not requires_decomposition
            and not requires_reasoning
            and not requires_multi_tool
            and not involves_mutation
            and plan.intent.confidence >= self.policy.system1_intent_confidence_threshold
        ):
            return CognitiveMode.SYSTEM1_FAST
        return CognitiveMode.SYSTEM2_DELIBERATIVE

    def _bind_tool(
        self,
        text: str,
        route_target: RouteTarget,
        tool_candidates: Sequence[str],
        tool_engine: ToolExecutionEngine,
    ) -> tuple[str | None, tuple[str, ...], ToolDescriptor | None]:
        if route_target is RouteTarget.MEMORY:
            descriptor = tool_engine.registry.resolve("memory_read")
            return "memory_read", ("memory_inspect",), descriptor
        if route_target is RouteTarget.DELEGATION:
            descriptor = tool_engine.registry.resolve("a2a_delegate")
            return "a2a_delegate", (), descriptor
        if route_target is not RouteTarget.TOOL:
            return None, (), None
        filtered = [tool_id for tool_id in tool_candidates if tool_id != "memory_write"]
        if not filtered:
            selected = tool_engine.registry.select_tools_for_task(text, token_budget=256)
            filtered = [summary.tool_id for summary in selected if summary.tool_id != "memory_write"]
        if not filtered:
            return None, (), None
        lowered = text.lower()
        ordered = filtered
        if "browser_navigate" in filtered and ("http://" in lowered or "https://" in lowered or "browser" in lowered):
            ordered = ["browser_navigate", *[item for item in filtered if item != "browser_navigate"]]
        elif "vision_describe" in filtered and any(token in lowered for token in ("image", "screenshot", "vision")):
            ordered = ["vision_describe", *[item for item in filtered if item != "vision_describe"]]
        elif "tool_registry_list" in filtered and any(token in lowered for token in ("tool", "registry")):
            ordered = ["tool_registry_list", *[item for item in filtered if item != "tool_registry_list"]]
        elif "tool_endpoint_matrix" in filtered and any(token in lowered for token in ("endpoint", "protocol", "compatibility")):
            ordered = ["tool_endpoint_matrix", *[item for item in filtered if item != "tool_endpoint_matrix"]]
        elif "data_compile_context" in filtered:
            ordered = ["data_compile_context", *[item for item in filtered if item != "data_compile_context"]]
        elif "retrieval_plan" in filtered:
            ordered = ["retrieval_plan", *[item for item in filtered if item != "retrieval_plan"]]
        for fallback_tool in ("data_compile_context", "retrieval_plan", "memory_read", "tool_registry_list"):
            if fallback_tool == ordered[0]:
                continue
            if tool_engine.registry.resolve(fallback_tool) is None:
                continue
            if fallback_tool not in ordered:
                ordered.append(fallback_tool)
        selected_tool = ordered[0]
        descriptor = tool_engine.registry.resolve(selected_tool)
        return selected_tool, tuple(ordered[1:]), descriptor

    def _dispatch_action(
        self,
        state: LoopExecutionState,
        action: LoopAction,
        *,
        tool_engine: ToolExecutionEngine,
        force_fallback: bool = False,
    ) -> ActionOutcome:
        tool_ids = [action.tool_id] if action.tool_id is not None else []
        if force_fallback:
            tool_ids = [action.tool_id]
        tool_ids.extend(tool_id for tool_id in action.fallback_tool_ids if tool_id not in tool_ids)
        for index, tool_id in enumerate(tool_ids):
            if tool_id is None:
                continue
            params = self._build_tool_params(state, tool_id, action)
            started_ns = perf_counter_ns()
            response = tool_engine.dispatch(
                ToolInvocationRequest(
                    tool_id=tool_id,
                    params=params,
                    context=self._execution_context(state.request),
                    idempotency_key=f"{state.request.loop_id}:{action.action_id}:{tool_id}",
                )
            )
            latency_ns = max(perf_counter_ns() - started_ns, 0)
            outcome = self._outcome_from_response(action, tool_id, response, latency_ns, used_fallback=index > 0)
            if outcome.status is ActionStatus.SUCCESS:
                return outcome
        return outcome

    def _build_tool_params(self, state: LoopExecutionState, tool_id: str, action: LoopAction) -> dict[str, Any]:
        session_id = str(state.request.session.session_id)
        lowered = action.instruction.lower()
        url = _extract_url(action.instruction) or _extract_url(state.output_text) or "https://example.com"
        modality = "image" if "image" in lowered else "audio" if "audio" in lowered else "text"
        if tool_id == "memory_read":
            return {"query": action.instruction, "layer": "auto", "top_k": 5, "session_id": session_id}
        if tool_id == "memory_inspect":
            return {"session_id": session_id, "include_audit": False}
        if tool_id == "retrieval_plan":
            return {
                "query": action.instruction,
                "session_summary": state.request.session.summary_text or "",
                "turn_count": state.request.session.turn_count,
            }
        if tool_id == "data_compile_context":
            return {"query": action.instruction, "session_id": session_id, "tool_budget": 512}
        if tool_id == "tool_registry_list":
            return {"task_hint": action.instruction, "token_budget": 384}
        if tool_id == "tool_endpoint_matrix":
            return {}
        if tool_id == "data_endpoint_catalog":
            return {} if modality == "text" else {"modality": modality}
        if tool_id == "browser_navigate":
            return {"url": url, "capture_screenshot": True}
        if tool_id == "vision_describe":
            return {"image_ref": url, "prompt": action.instruction, "max_tokens": 128}
        if tool_id == "a2a_delegate":
            return {
                "agent_name": f"specialist-{action.action_id}",
                "task": action.instruction,
                "artifacts": [{"name": "evidence", "uri": f"evidence://{action.action_id}"}],
            }
        if tool_id == "terminal_execute":
            command = action.instruction.strip()
            lowered_command = command.lower()
            for prefix in ("run the command ", "execute the command ", "run ", "execute "):
                if lowered_command.startswith(prefix):
                    command = command[len(prefix) :]
                    lowered_command = command.lower()
                    break
            for suffix in (" in the terminal", " in terminal"):
                if lowered_command.endswith(suffix):
                    command = command[: -len(suffix)]
                    lowered_command = command.lower()
                    break
            return {"command": command.strip().strip(".")}
        return {"query": action.instruction}

    def _outcome_from_response(
        self,
        action: LoopAction,
        tool_id: str,
        response: ToolInvocationResponse,
        latency_ns: int,
        *,
        used_fallback: bool,
    ) -> ActionOutcome:
        if response.status is InvocationStatus.SUCCESS and response.success is not None:
            return ActionOutcome(
                action_id=action.action_id,
                status=ActionStatus.SUCCESS,
                tool_id=tool_id,
                protocol=action.protocol,
                output=dict(response.success.data) if isinstance(response.success.data, dict) else {"result": response.success.data},
                latency_ns=latency_ns,
                cache_hit=bool(response.success.execution_metadata.get("cache_hit", False)),
                mutated_state=tool_id == "memory_write" or bool(action.mutates_state),
                used_fallback=used_fallback,
            )
        if response.status is InvocationStatus.PENDING and response.pending is not None:
            return ActionOutcome(
                action_id=action.action_id,
                status=ActionStatus.PENDING,
                tool_id=tool_id,
                protocol=action.protocol,
                output={"approval_ticket_id": str(response.pending.approval_ticket_id)},
                latency_ns=latency_ns,
                error_code="approval_required",
                error_message="tool invocation is pending approval",
                used_fallback=used_fallback,
            )
        error_code = None
        error_message = "tool invocation failed"
        if response.error is not None:
            error_code = response.error.code.value
            error_message = response.error.message
        return ActionOutcome(
            action_id=action.action_id,
            status=ActionStatus.ERROR,
            tool_id=tool_id,
            protocol=action.protocol,
            output={},
            latency_ns=latency_ns,
            error_code=error_code,
            error_message=error_message,
            used_fallback=used_fallback,
        )

    def _synthesize_action_output(
        self,
        state: LoopExecutionState,
        action: LoopAction,
        *,
        repair_hint: str | None = None,
    ) -> dict[str, Any]:
        evidence_lines = [
            item.content
            for item in (state.evidence.items if state.evidence is not None else ())
            if item.action_id == action.action_id
        ][:3]
        summary_parts = [action.instruction]
        if evidence_lines:
            summary_parts.append(" | ".join(evidence_lines))
        if repair_hint:
            summary_parts.append(repair_hint)
        summary = " ".join(part for part in summary_parts if part).strip()
        return {"summary": summary[:512], "evidence_count": len(evidence_lines)}

    def _compose_output_text(self, state: LoopExecutionState) -> str:
        parts: list[str] = []
        for outcome in state.action_outcomes:
            payload = outcome.output.get("summary")
            if not isinstance(payload, str):
                payload = str(outcome.output)
            parts.append(f"{outcome.action_id}: {payload}")
        return "\n".join(parts)

    def _capture_deferred_writes(self, state: LoopExecutionState) -> None:
        if not state.output_text:
            return
        candidates: list[DeferredMemoryWrite] = [
            DeferredMemoryWrite(
                target_layer=MemoryTier.SESSION,
                target_scope=MemoryScope.LOCAL,
                memory_type=MemoryType.FACT,
                content=f"Loop outcome: {state.output_text[:240]}",
                rationale="session outcome summary",
            )
        ]
        if state.repair_count > 0:
            candidates.append(
                DeferredMemoryWrite(
                    target_layer=MemoryTier.EPISODIC,
                    target_scope=MemoryScope.LOCAL,
                    memory_type=MemoryType.CORRECTION,
                    content=f"Repair pattern: {state.repair_history[-1].rationale}",
                    rationale="successful repair pattern",
                )
            )
        for constraint in state.plan.constraints if state.plan is not None else ():
            if any(token in constraint.lower() for token in ("must", "only", "before", "after", "without", "using")):
                candidates.append(
                    DeferredMemoryWrite(
                        target_layer=MemoryTier.SEMANTIC,
                        target_scope=MemoryScope.GLOBAL,
                        memory_type=MemoryType.CONSTRAINT,
                        content=constraint,
                        rationale="preserved execution constraint",
                    )
                )
        unique: dict[tuple[MemoryTier, str], DeferredMemoryWrite] = {}
        for candidate in candidates:
            unique[(candidate.target_layer, candidate.content)] = candidate
        state.deferred_writes = list(unique.values())

    def _select_repair_strategy(self, state: LoopExecutionState, defect: VerificationDefect) -> RepairDecision:
        if defect.root_cause is RootCauseClass.TOOL_FAILURE and defect.action_id is not None and state.plan is not None:
            action = next((item for item in state.plan.actions if item.action_id == defect.action_id), None)
            if action is not None and action.fallback_tool_ids:
                return RepairDecision(
                    strategy=RepairStrategy.TOOL_SUBSTITUTE,
                    rationale="retry with a fallback tool binding",
                    estimated_cost_tokens=64,
                    defect_id=defect.defect_id,
                    target_action_id=action.action_id,
                    replacement_tool_id=action.fallback_tool_ids[0],
                )
        if defect.root_cause is RootCauseClass.RETRIEVAL_GAP:
            return RepairDecision(
                strategy=RepairStrategy.EXPANDED_RETRIEVAL,
                rationale="expand evidence retrieval with a more specific query",
                estimated_cost_tokens=192,
                defect_id=defect.defect_id,
                target_action_id=defect.action_id,
                followup_query=f"{state.request.query_text} grounded evidence",
            )
        if defect.root_cause in {RootCauseClass.INCOMPLETENESS, RootCauseClass.LOGIC_ERROR, RootCauseClass.SCHEMA_VIOLATION}:
            return RepairDecision(
                strategy=RepairStrategy.MINIMAL_EDIT,
                rationale="repair the failing action with a bounded local regeneration",
                estimated_cost_tokens=96,
                defect_id=defect.defect_id,
                target_action_id=defect.action_id,
            )
        return RepairDecision(
            strategy=RepairStrategy.SECTION_REWRITE,
            rationale="rewrite the affected section using the current evidence bundle",
            estimated_cost_tokens=160,
            defect_id=defect.defect_id,
            target_action_id=defect.action_id,
        )

    def _replace_action_outcome(self, state: LoopExecutionState, updated: ActionOutcome) -> None:
        replaced = False
        for index, outcome in enumerate(state.action_outcomes):
            if outcome.action_id == updated.action_id:
                state.action_outcomes[index] = updated
                replaced = True
                break
        if not replaced:
            state.action_outcomes.append(updated)

    def _evidence_from_fragments(
        self,
        action_id: str,
        fragments: Sequence[ProvenanceTaggedFragment],
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for fragment in fragments:
            items.append(
                EvidenceItem(
                    action_id=action_id,
                    content=fragment.content,
                    score=fragment.final_score,
                    token_count=fragment.token_count,
                    source_table=fragment.provenance.origin.source_table,
                    source_id=fragment.provenance.origin.source_id,
                    retrieval_method=fragment.provenance.origin.retrieval_method,
                    authority_tier=fragment.provenance.trust.authority_tier,
                    freshness_seconds=max(int(fragment.provenance.trust.freshness.total_seconds()), 0),
                    trace_id=fragment.provenance.custody_hash,
                )
            )
        return items

    def _execution_context(self, request: LoopExecutionRequest) -> ExecutionContext:
        return ExecutionContext(
            user_id=request.user_id,
            agent_id=request.agent_id,
            session_id=request.session.session_id,
            scopes=request.scopes,
            trace_id=str(request.loop_id),
        )

    def _should_pause(self, state: LoopExecutionState, phase: LoopPhase) -> bool:
        return state.request.stop_after_phase is phase and state.current_phase not in {LoopPhase.HALT, LoopPhase.FAIL}

    def _checkpoint(self, state: LoopExecutionState, *, paused: bool) -> None:
        if self.policy.checkpoint_every_phase or paused:
            self.repository.put_checkpoint(new_checkpoint(state, paused=paused))

    def _record_span(self, state: LoopExecutionState, phase: LoopPhase, started_ns: int) -> None:
        duration_ns = max(perf_counter_ns() - started_ns, 0)
        state.phase_spans.append(LoopPhaseSpan(phase=phase, duration_ns=duration_ns))

    def _content_hash(self, text: str) -> bytes:
        return __import__("hashlib").sha256(text.encode("utf-8")).digest()

    def _finalize(self, state: LoopExecutionState, *, paused: bool, success: bool) -> LoopExecutionResult:
        audit = new_loop_audit_record(state, paused=paused, success=success)
        self.repository.insert_audit_record(audit)
        checkpoints = self.repository.list_checkpoints(state.request.loop_id)
        return LoopExecutionResult(
            loop_id=state.request.loop_id,
            last_completed_phase=state.last_completed_phase,
            next_phase=state.current_phase if paused else None,
            cognitive_mode=state.cognitive_mode,
            paused=paused,
            upgraded_from_fast_path=state.upgraded_from_fast_path,
            query_response=state.query_response,
            plan=state.plan,
            predicted_resources=state.predicted_resources,
            evidence=state.evidence,
            action_outcomes=tuple(state.action_outcomes),
            metacognitive=state.metacognitive,
            verdict=state.verdict,
            repair_history=tuple(state.repair_history),
            committed_writes=tuple(state.committed_writes),
            checkpoints=tuple(checkpoints),
            audit_id=audit.audit_id,
            escalation_reason=state.escalation_reason,
            output_text=state.output_text,
            subsystem_status=dict(state.subsystem_status),
            latency_ns=audit.latency_ns,
        )

    def _extra_query_tool_definitions(self) -> tuple[dict[str, Any], ...]:
        return ()

    def _query_tool_definitions(self) -> tuple[dict[str, Any], ...]:
        return (*build_default_tool_definitions(), *self._extra_query_tool_definitions())

    def _register_custom_tools(self, engine: ToolExecutionEngine) -> None:
        del engine


__all__ = ["AgentLoopEngine"]
