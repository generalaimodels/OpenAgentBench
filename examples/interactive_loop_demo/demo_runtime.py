"""Shared runtime for the interactive loop demo and benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from time import perf_counter_ns
from typing import Any, Callable, Literal, Sequence
from uuid import uuid4

from openagentbench.agent_data import (
    HistoryRecord,
    MemoryRecord,
    MemoryScope,
    MemoryTier,
    MessageRole,
    ProvenanceType,
    SessionRecord,
    SessionStatus,
    hash_normalized_text,
)
from openagentbench.agent_loop import LoopExecutionRequest, LoopExecutionResult, LoopPhase
from openagentbench.agent_query import OpenAICompatibleTextModel
from openagentbench.agent_memory import WorkingMemoryItem
from openagentbench.agent_retrieval import Modality

from .demo_engine import InteractiveDemoLoopEngine
from .demo_env import DemoConfig, load_demo_config


ProviderName = Literal["auto", "demo", "openai"]

_DEMO_SCOPES = (
    "tools.read",
    "tools.write",
    "tools.admin",
    "tools.browser",
    "tools.vision",
    "tools.delegate",
    "tools.terminal",
)

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are the final response layer for OpenAgentBench. "
    "Answer the user's latest request using only the supplied framework trace. "
    "Ground every claim in the provided plan, action outcomes, evidence, committed writes, and terminal output. "
    "If information is missing or a tool failed, say that clearly instead of guessing. "
    "Keep the answer technical, concise, and directly useful."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _build_session() -> SessionRecord:
    now = _utc_now()
    return SessionRecord(
        session_id=uuid4(),
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        status=SessionStatus.ACTIVE,
        model_id="demo-agent-loop",
        context_window_size=32_000,
        system_prompt_hash=hash_normalized_text("You are the OpenAgentBench interactive loop demo."),
        system_prompt_tokens=10,
        max_response_tokens=1_500,
        turn_count=0,
        summary_text="Interactive demo session for the six-module OpenAgentBench stack.",
        summary_token_count=11,
        system_prompt_text="You are the OpenAgentBench interactive loop demo.",
    )


def _seed_history(session: SessionRecord) -> list[HistoryRecord]:
    now = _utc_now()
    return [
        HistoryRecord(
            message_id=uuid4(),
            session_id=session.session_id,
            user_id=session.user_id,
            turn_index=1,
            role=MessageRole.SYSTEM,
            content="The demo workspace contains a small Python module and tests.",
            content_parts=None,
            name=None,
            tool_calls=None,
            tool_call_id=None,
            content_embedding=None,
            content_hash=hash_normalized_text("The demo workspace contains a small Python module and tests."),
            token_count=10,
            model_id=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=None,
            api_call_id=None,
            created_at=now,
        )
    ]


def _seed_memories(session: SessionRecord, config: DemoConfig) -> list[MemoryRecord]:
    now = _utc_now()
    return [
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=None,
            memory_tier=MemoryTier.SEMANTIC,
            memory_scope=MemoryScope.GLOBAL,
            content_text="Global rule: PostgreSQL is the durable source of truth for agent memory persistence.",
            content_embedding=None,
            content_hash=hash_normalized_text(
                "Global rule: PostgreSQL is the durable source of truth for agent memory persistence."
            ),
            provenance_type=ProvenanceType.FACT,
            provenance_turn_id=None,
            confidence=0.98,
            relevance_accumulator=5.0,
            access_count=8,
            last_accessed_at=now,
            created_at=now - timedelta(days=7),
            updated_at=now,
            expires_at=None,
            is_active=True,
            is_validated=True,
            token_count=12,
            tags=("semantic", "database"),
        ),
        MemoryRecord(
            memory_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            memory_tier=MemoryTier.SESSION,
            memory_scope=MemoryScope.LOCAL,
            content_text=f"Session note: terminal commands run inside {config.workspace_root}.",
            content_embedding=None,
            content_hash=hash_normalized_text(f"Session note: terminal commands run inside {config.workspace_root}."),
            provenance_type=ProvenanceType.INSTRUCTION,
            provenance_turn_id=None,
            confidence=1.0,
            relevance_accumulator=3.0,
            access_count=3,
            last_accessed_at=now,
            created_at=now - timedelta(hours=1),
            updated_at=now,
            expires_at=now + timedelta(days=1),
            is_active=True,
            is_validated=True,
            token_count=10,
            tags=("session", "terminal"),
        ),
    ]


def _seed_working(session: SessionRecord) -> list[WorkingMemoryItem]:
    now = _utc_now()
    return [
        WorkingMemoryItem(
            item_id=uuid4(),
            user_id=session.user_id,
            session_id=session.session_id,
            step_id=uuid4(),
            content_text="Working memory: prefer safe reversible actions first and summarize terminal output clearly.",
            token_count=12,
            modality=Modality.TEXT,
            created_at=now,
        )
    ]


def _new_history_record(
    session: SessionRecord,
    *,
    role: MessageRole,
    content: str,
    turn_index: int,
    model_id: str | None,
) -> HistoryRecord:
    now = _utc_now()
    return HistoryRecord(
        message_id=uuid4(),
        session_id=session.session_id,
        user_id=session.user_id,
        turn_index=turn_index,
        role=role,
        content=content,
        content_parts=None,
        name=None,
        tool_calls=None,
        tool_call_id=None,
        content_embedding=None,
        content_hash=hash_normalized_text(content),
        token_count=max(len(content.split()), 1),
        model_id=model_id,
        finish_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=None,
        api_call_id=None,
        created_at=now,
    )


def sample_prompts() -> tuple[str, ...]:
    return (
        "Remember that PostgreSQL is the durable memory store and summarize why that matters.",
        "Open https://example.com in the browser and summarize the page state.",
        "Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result.",
        "List the available tools and explain which one can inspect memory.",
    )


@dataclass(slots=True, frozen=True)
class LoopProgressEvent:
    phase: str
    title: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)


def _phase_title(phase: LoopPhase | None) -> str:
    if phase is None:
        return "Unknown"
    return phase.value.replace("_", " ").title()


def _build_progress_event(result: LoopExecutionResult) -> LoopProgressEvent:
    phase = result.last_completed_phase
    if phase is LoopPhase.CONTEXT_ASSEMBLE and result.query_response is not None:
        plan = result.query_response.plan
        primary_route = plan.subqueries[0].route_target.value if plan.subqueries else "none"
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=(
                f"intent={plan.intent.intent_class.value} route={primary_route} "
                f"subqueries={len(plan.subqueries)} clarification={plan.needs_clarification}"
            ),
            metrics={
                "cache_hit": result.query_response.cache_hit,
                "latency_ms": result.query_response.latency_ms,
            },
        )
    if phase is LoopPhase.PLAN and result.plan is not None:
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"actions={len(result.plan.actions)} validated={result.plan.validated} rollback={result.plan.rollback_required}",
            metrics={"constraints": len(result.plan.constraints)},
        )
    if phase is LoopPhase.DECOMPOSE and result.plan is not None:
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"decomposition ready with {len(result.plan.actions)} executable actions",
        )
    if phase is LoopPhase.PREDICT and result.predicted_resources is not None:
        predicted = result.predicted_resources
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=(
                f"tokens={predicted.total_tokens} tool_calls={predicted.tool_calls} "
                f"model_calls={predicted.model_calls}"
            ),
            metrics={
                "estimated_latency_ms": predicted.estimated_latency_ms,
                "estimated_cost": round(predicted.estimated_cost, 4),
            },
        )
    if phase is LoopPhase.RETRIEVE and result.evidence is not None:
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"evidence_items={len(result.evidence.items)} quality={result.evidence.quality_score:.2f}",
            metrics={"trace_ids": len(result.evidence.trace_ids)},
        )
    if phase is LoopPhase.ACT:
        success_count = sum(1 for outcome in result.action_outcomes if outcome.status.value == "success")
        tool_count = sum(1 for outcome in result.action_outcomes if outcome.tool_id is not None)
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"actions_executed={len(result.action_outcomes)} successes={success_count}",
            metrics={"tool_calls": tool_count},
        )
    if phase is LoopPhase.METACOGNITIVE_CHECK and result.metacognitive is not None:
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"decision={result.metacognitive.decision.value} score={result.metacognitive.score:.2f}",
            metrics={"evidence_coverage": round(result.metacognitive.evidence_coverage, 4)},
        )
    if phase is LoopPhase.VERIFY and result.verdict is not None:
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"passed={result.verdict.passed} grounded={result.verdict.grounding_score:.2f}",
            metrics={"defects": len(result.verdict.defects)},
        )
    if phase is LoopPhase.CRITIQUE:
        defect_count = len(result.verdict.defects) if result.verdict is not None else 0
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"critique complete defects={defect_count}",
            metrics={"repairs": len(result.repair_history)},
        )
    if phase is LoopPhase.REPAIR:
        last_strategy = result.repair_history[-1].strategy.value if result.repair_history else "noop"
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"repair_count={len(result.repair_history)} last_strategy={last_strategy}",
        )
    if phase is LoopPhase.COMMIT:
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"committed_writes={len(result.committed_writes)}",
        )
    if phase is LoopPhase.ESCALATE:
        return LoopProgressEvent(
            phase=phase.value,
            title=_phase_title(phase),
            summary=f"escalation={result.escalation_reason.value if result.escalation_reason else 'unknown'}",
        )
    return LoopProgressEvent(
        phase=phase.value if phase is not None else "unknown",
        title=_phase_title(phase),
        summary="phase completed",
    )


@dataclass(slots=True)
class DemoLoopApplication:
    config: DemoConfig = field(default_factory=load_demo_config)
    session: SessionRecord = field(default_factory=_build_session)
    history: list[HistoryRecord] = field(init=False)
    memories: list[MemoryRecord] = field(init=False)
    working_items: list[WorkingMemoryItem] = field(init=False)
    engine: InteractiveDemoLoopEngine = field(init=False)
    last_result: LoopExecutionResult | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.history = _seed_history(self.session)
        self.memories = _seed_memories(self.session, self.config)
        self.working_items = _seed_working(self.session)
        self.engine = InteractiveDemoLoopEngine(demo_config=self.config)

    @property
    def provider_name(self) -> str:
        return "demo"

    @property
    def model_label(self) -> str:
        return "interactive-demo-loop"

    def _append_user_turn(self, query_text: str) -> None:
        user_turn_index = self.session.turn_count + 1
        self.history.append(
            _new_history_record(
                self.session,
                role=MessageRole.USER,
                content=query_text,
                turn_index=user_turn_index,
                model_id=None,
            )
        )
        self.session.turn_count = user_turn_index
        self.session.updated_at = _utc_now()

    def _execute_loop(self, query_text: str) -> LoopExecutionResult:
        return self.engine.execute(
            LoopExecutionRequest(
                user_id=self.session.user_id,
                session=self.session,
                query_text=query_text,
                scopes=_DEMO_SCOPES,
            ),
            history=tuple(self.history),
            memories=self.memories,
            working_items=tuple(self.working_items),
        )

    def _execute_loop_realtime(
        self,
        query_text: str,
        *,
        on_event: Callable[[LoopProgressEvent], None],
    ) -> LoopExecutionResult:
        stop_after_phase = LoopPhase.CONTEXT_ASSEMBLE
        result = self.engine.execute(
            LoopExecutionRequest(
                user_id=self.session.user_id,
                session=self.session,
                query_text=query_text,
                scopes=_DEMO_SCOPES,
                stop_after_phase=stop_after_phase,
            ),
            history=tuple(self.history),
            memories=self.memories,
            working_items=tuple(self.working_items),
        )
        if result.last_completed_phase is not None:
            on_event(_build_progress_event(result))
        while result.paused and result.next_phase not in {None, LoopPhase.HALT, LoopPhase.FAIL}:
            result = self.engine.resume(
                result.loop_id,
                history=tuple(self.history),
                memories=self.memories,
                working_items=tuple(self.working_items),
                stop_after_phase=result.next_phase,
            )
            if result.last_completed_phase is not None:
                on_event(_build_progress_event(result))
        return result

    def _refresh_session_summary(self) -> None:
        self.session.updated_at = _utc_now()
        self.session.summary_text = (
            "Interactive demo session. Recent user goals: "
            + " | ".join(record.content for record in self.history if record.role is MessageRole.USER)[-300:]
        )
        self.session.summary_token_count = max(len((self.session.summary_text or "").split()), 1)

    def _record_result(self, result: LoopExecutionResult, *, model_id: str) -> None:
        assistant_turn_index = self.session.turn_count + 1
        self.history.append(
            _new_history_record(
                self.session,
                role=MessageRole.ASSISTANT,
                content=result.output_text or "The loop completed without a textual output.",
                turn_index=assistant_turn_index,
                model_id=model_id,
            )
        )
        self.session.turn_count = assistant_turn_index
        self._refresh_session_summary()
        self.working_items.append(
            WorkingMemoryItem(
                item_id=uuid4(),
                user_id=self.session.user_id,
                session_id=self.session.session_id,
                step_id=uuid4(),
                content_text=f"Last loop mode={result.cognitive_mode.value if result.cognitive_mode else 'unknown'}; output={result.output_text[:180]}",
                token_count=max(len(result.output_text.split()), 1) if result.output_text else 8,
                modality=Modality.TEXT,
                created_at=_utc_now(),
            )
        )
        if len(self.working_items) > 6:
            self.working_items = self.working_items[-6:]

    def run_query(self, query_text: str) -> LoopExecutionResult:
        started_ns = perf_counter_ns()
        self._append_user_turn(query_text)
        result = self._execute_loop(query_text)
        self._record_result(result, model_id=self.model_label)
        final_result = replace(result, latency_ns=max(perf_counter_ns() - started_ns, 0))
        self.last_result = final_result
        return final_result

    def run_query_stream(self, query_text: str, *, on_delta: Callable[[str], None]) -> LoopExecutionResult:
        result = self.run_query(query_text)
        if result.output_text:
            on_delta(result.output_text)
        return result

    def run_query_realtime(
        self,
        query_text: str,
        *,
        on_event: Callable[[LoopProgressEvent], None],
        on_delta: Callable[[str], None] | None = None,
    ) -> LoopExecutionResult:
        started_ns = perf_counter_ns()
        self._append_user_turn(query_text)
        raw_result = self._execute_loop_realtime(query_text, on_event=on_event)
        result = raw_result
        if on_delta is not None and result.output_text:
            on_delta(result.output_text)
        self._record_result(result, model_id=self.model_label)
        final_result = replace(result, latency_ns=max(perf_counter_ns() - started_ns, 0))
        self.last_result = final_result
        return final_result


def _create_openai_client(config: DemoConfig) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in live environments
        raise RuntimeError("Install the openai package with `python3 -m pip install openai` or `pip install .[openai]`.") from exc
    if not config.openai_api_key_present:
        raise RuntimeError(f"OPENAI_API_KEY is required for provider=openai. Add it to {config.env_path}.")
    return OpenAI(timeout=config.openai_timeout_seconds)


def _build_openai_text_model(client: Any, config: DemoConfig) -> OpenAICompatibleTextModel:
    return OpenAICompatibleTextModel(
        client=client,
        model=config.openai_model,
        reasoning_effort=config.openai_reasoning_effort,
        store=False,
    )


def _truncate_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _compact_output(output: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in output.items():
        if isinstance(value, str):
            compact[key] = _truncate_text(value, 480)
            continue
        if isinstance(value, dict):
            nested: dict[str, Any] = {}
            for nested_key, nested_value in list(value.items())[:8]:
                nested[nested_key] = (
                    _truncate_text(nested_value, 180) if isinstance(nested_value, str) else nested_value
                )
            compact[key] = nested
            continue
        if isinstance(value, list):
            compact[key] = [
                _truncate_text(item, 140) if isinstance(item, str) else item for item in value[:8]
            ]
            continue
        compact[key] = value
    return compact


def _build_synthesis_payload(query_text: str, result: LoopExecutionResult) -> str:
    payload = {
        "user_query": query_text,
        "cognitive_mode": result.cognitive_mode.value if result.cognitive_mode is not None else None,
        "last_completed_phase": result.last_completed_phase.value if result.last_completed_phase is not None else None,
        "escalation_reason": result.escalation_reason.value if result.escalation_reason is not None else None,
        "framework_output": _truncate_text(result.output_text, 1200),
        "plan": [
            {
                "action_id": action.action_id,
                "route_target": action.route_target.value,
                "tool_id": action.tool_id,
                "instruction": _truncate_text(action.instruction, 220),
            }
            for action in (result.plan.actions if result.plan is not None else ())
        ],
        "action_outcomes": [
            {
                "action_id": outcome.action_id,
                "status": outcome.status.value,
                "tool_id": outcome.tool_id,
                "used_fallback": outcome.used_fallback,
                "error_code": outcome.error_code,
                "error_message": outcome.error_message,
                "output": _compact_output(dict(outcome.output)),
            }
            for outcome in result.action_outcomes
        ],
        "evidence": [
            {
                "action_id": item.action_id,
                "content": _truncate_text(item.content, 240),
                "score": round(item.score, 4),
                "source_table": item.source_table.value,
                "authority_tier": item.authority_tier.value,
            }
            for item in (result.evidence.items[:10] if result.evidence is not None else ())
        ],
        "committed_writes": [
            {
                "layer": write.target_layer.value,
                "scope": write.target_scope.value,
                "content": _truncate_text(write.content, 180),
            }
            for write in result.committed_writes
        ],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


@dataclass(slots=True)
class OpenAIDemoLoopApplication(DemoLoopApplication):
    client: Any | None = None
    synthesis_model: OpenAICompatibleTextModel = field(init=False)

    def __post_init__(self) -> None:
        DemoLoopApplication.__post_init__(self)
        self.client = _create_openai_client(self.config) if self.client is None else self.client
        self.synthesis_model = _build_openai_text_model(self.client, self.config)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_label(self) -> str:
        return self.config.openai_model

    def _synthesize_response(
        self,
        query_text: str,
        result: LoopExecutionResult,
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        payload = _build_synthesis_payload(query_text, result)
        if on_delta is None:
            try:
                response = self.synthesis_model.complete(
                    system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
                    user_input=payload,
                    max_tokens=self.config.openai_max_output_tokens,
                    temperature=0.1,
                )
            except Exception:  # pragma: no cover - network/runtime failures fallback to framework text
                return result.output_text
            return response.strip() or result.output_text
        chunks: list[str] = []
        try:
            for delta in self.synthesis_model.stream_complete(
                system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
                user_input=payload,
                max_tokens=self.config.openai_max_output_tokens,
            ):
                if not delta:
                    continue
                chunks.append(delta)
                on_delta(delta)
        except Exception:  # pragma: no cover - network/runtime failures fallback to framework text
            return result.output_text
        response = "".join(chunks).strip()
        return response or result.output_text

    def run_query(self, query_text: str) -> LoopExecutionResult:
        started_ns = perf_counter_ns()
        self._append_user_turn(query_text)
        raw_result = self._execute_loop(query_text)
        result = replace(raw_result, output_text=self._synthesize_response(query_text, raw_result))
        self._record_result(result, model_id=self.model_label)
        final_result = replace(result, latency_ns=max(perf_counter_ns() - started_ns, 0))
        self.last_result = final_result
        return final_result

    def run_query_stream(self, query_text: str, *, on_delta: Callable[[str], None]) -> LoopExecutionResult:
        started_ns = perf_counter_ns()
        self._append_user_turn(query_text)
        raw_result = self._execute_loop(query_text)
        result = replace(
            raw_result,
            output_text=self._synthesize_response(query_text, raw_result, on_delta=on_delta),
        )
        self._record_result(result, model_id=self.model_label)
        final_result = replace(result, latency_ns=max(perf_counter_ns() - started_ns, 0))
        self.last_result = final_result
        return final_result

    def run_query_realtime(
        self,
        query_text: str,
        *,
        on_event: Callable[[LoopProgressEvent], None],
        on_delta: Callable[[str], None] | None = None,
    ) -> LoopExecutionResult:
        started_ns = perf_counter_ns()
        self._append_user_turn(query_text)
        raw_result = self._execute_loop_realtime(query_text, on_event=on_event)
        result = replace(
            raw_result,
            output_text=self._synthesize_response(query_text, raw_result, on_delta=on_delta),
        )
        self._record_result(result, model_id=self.model_label)
        final_result = replace(result, latency_ns=max(perf_counter_ns() - started_ns, 0))
        self.last_result = final_result
        return final_result


def build_application(
    *,
    config: DemoConfig | None = None,
    provider: ProviderName = "auto",
    client: Any | None = None,
) -> DemoLoopApplication | OpenAIDemoLoopApplication:
    resolved_config = load_demo_config() if config is None else config
    selected_provider = provider
    if selected_provider == "auto":
        selected_provider = "openai" if resolved_config.openai_api_key_present else "demo"
    if selected_provider == "openai":
        return OpenAIDemoLoopApplication(config=resolved_config, client=client)
    return DemoLoopApplication(config=resolved_config)


def format_result(result: LoopExecutionResult) -> str:
    lines = [
        f"Mode: {result.cognitive_mode.value if result.cognitive_mode else 'unknown'}",
        f"Last Phase: {result.last_completed_phase.value if result.last_completed_phase else 'none'}",
        f"Latency: {result.latency_ns / 1_000_000:.2f} ms",
        f"Actions: {len(result.action_outcomes)}",
    ]
    if result.plan is not None and result.plan.actions:
        lines.append("Plan:")
        for action in result.plan.actions:
            lines.append(f"- {action.action_id} [{action.route_target.value}] tool={action.tool_id or 'none'} :: {action.instruction}")
    if result.action_outcomes:
        lines.append("Outcomes:")
        for outcome in result.action_outcomes:
            detail = ""
            if "stdout" in outcome.output:
                detail = f" stdout={str(outcome.output['stdout']).strip()[:120]}"
            elif "items" in outcome.output:
                detail = f" items={outcome.output.get('count', len(outcome.output.get('items', [])))}"
            lines.append(
                f"- {outcome.action_id} status={outcome.status.value} tool={outcome.tool_id or 'none'} fallback={outcome.used_fallback}{detail}"
            )
    if result.evidence is not None:
        lines.append(f"Evidence Items: {len(result.evidence.items)}")
    if result.committed_writes:
        lines.append("Committed Memory Writes:")
        for write in result.committed_writes:
            lines.append(f"- {write.target_layer.name.lower()}::{write.target_scope.name.lower()} {write.content[:120]}")
    if result.output_text:
        lines.append("Output:")
        lines.append(result.output_text)
    if result.escalation_reason is not None:
        lines.append(f"Escalation: {result.escalation_reason.value}")
    return "\n".join(lines)


__all__ = [
    "DemoLoopApplication",
    "LoopProgressEvent",
    "OpenAIDemoLoopApplication",
    "ProviderName",
    "build_application",
    "format_result",
    "load_demo_config",
    "sample_prompts",
]
