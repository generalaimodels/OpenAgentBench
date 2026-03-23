"""Stateful realtime CLI runtime for the OpenAgentBench Q&A chatbot example."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping
from uuid import UUID, uuid4

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for the example runtime
    OpenAI = None

from openagentbench.agent_context import CompiledCycleContext, build_agent_context_compatibility_report
from openagentbench.agent_data import (
    FinishReason,
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
from openagentbench.agent_memory import (
    WorkingMemoryBuffer,
    WorkingMemoryItem,
    build_session_checkpoint,
    compute_working_memory_capacity,
    update_session_summary,
)
from openagentbench.agent_query import QueryResolutionRequest, QueryResolutionResponse, QueryResolver
from openagentbench.agent_retrieval import Modality
from openagentbench.agent_sdk import (
    AgentSdk,
    AgentSdkInvocationRequest,
    AgentSdkInvocationResult,
    AgentSdkProviderFactory,
    AgentSdkSnapshot,
    ProjectedToolSurface,
    ProviderSuite,
)
from examples.realtime_qa_chatbot.tools import (
    RealtimeQaLoopEngine,
    build_terminal_execute_descriptor,
    infer_terminal_request,
)

ProviderName = Literal["auto", "openai", "vllm", "heuristic"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_path() -> Path:
    return _project_root() / ".env"


def _strip_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path | None = None) -> dict[str, str]:
    resolved_path = path or _env_path()
    values: dict[str, str] = {}
    if not resolved_path.exists():
        return values
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        normalized_value = _strip_env_value(raw_value)
        values[normalized_key] = normalized_value
        os.environ.setdefault(normalized_key, normalized_value)
    return values


def _default_provider_models() -> tuple[str, str]:
    report = build_agent_context_compatibility_report()
    return (
        str(report.openai_responses_request["model"]),
        str(report.vllm_responses_request["model"]),
    )


def _coalesce_history_text(record: HistoryRecord) -> str:
    if record.content:
        return record.content
    parts: list[str] = []
    for part in record.content_parts or ():
        if not isinstance(part, dict):
            continue
        for key in ("text", "image_url", "audio_url", "video_url"):
            value = part.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
    return " ".join(parts)


def _result_payload(invocation: AgentSdkInvocationResult | None) -> Mapping[str, Any]:
    if invocation is None or invocation.response.success is None:
        return {}
    data = invocation.response.success.data
    if isinstance(data, dict):
        return data
    return {"value": data}


def _safe_round(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _looks_like_structured_dump(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith("sq_1:"):
        return True
    if "{'" in stripped and "primary_model" in stripped:
        return True
    if stripped.count("{") >= 2 and stripped.count("}") >= 2 and "memory_id" in stripped:
        return True
    return False


@dataclass(slots=True, frozen=True)
class ChatbotConfig:
    provider: ProviderName = "auto"
    model: str | None = None
    base_url: str | None = None
    env_file: Path | None = None
    reasoning_effort: str = "medium"
    stream_answer: bool = True
    tool_budget: int = 768
    answer_max_tokens: int = 640
    context_provider: str = "openai_responses"


@dataclass(slots=True, frozen=True)
class RealtimeQaEvent:
    kind: str
    title: str
    message: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatbotTurnArtifacts:
    question: str
    answer: str
    provider: str
    sdk_snapshot: AgentSdkSnapshot
    tool_surface: ProjectedToolSurface
    query_preview: QueryResolutionResponse
    context_preview: CompiledCycleContext
    retrieval_preview: AgentSdkInvocationResult | None
    memory_preview: AgentSdkInvocationResult | None
    registry_preview: AgentSdkInvocationResult | None
    compiled_context_preview: AgentSdkInvocationResult | None
    terminal_followup_preview: AgentSdkInvocationResult | None
    loop_result: LoopExecutionResult
    generated_at: datetime = field(default_factory=_utc_now)


class RealtimeQaChatbot:
    def __init__(self, config: ChatbotConfig | None = None) -> None:
        self.config = config or ChatbotConfig()
        self.environment = load_env_file(self.config.env_file)
        self._default_openai_model, self._default_vllm_model = _default_provider_models()
        self.sdk_provider_factory = AgentSdkProviderFactory()
        self.provider_name = self._resolve_provider_name()
        self.provider_suite, self.provider_model, self.provider_note = self._build_provider_suite()
        self.session = self._build_session(model_id=self._session_model_id())
        self.memories: list[MemoryRecord] = self._bootstrap_memories()
        self.history: list[HistoryRecord] = []
        self.working_buffer = WorkingMemoryBuffer(capacity=self._working_capacity())
        self.query_resolver = QueryResolver()
        self.loop_engine = RealtimeQaLoopEngine(project_root=_project_root())
        self.session_checkpoints: list[Any] = []
        self.last_turn: ChatbotTurnArtifacts | None = None

    def status_snapshot(self) -> dict[str, Any]:
        sdk = self._build_sdk()
        return {
            "provider": self.provider_name,
            "provider_model": self.provider_model,
            "provider_note": self.provider_note,
            "session_id": str(self.session.session_id),
            "user_id": str(self.session.user_id),
            "turn_count": self.session.turn_count,
            "history_records": len(self.history),
            "memory_records": len(self.memories),
            "working_items": len(self.working_buffer.items),
            "working_capacity": self.working_buffer.capacity,
            "connector_count": len(sdk.list_connectors()),
            "tool_count": len(sdk.tool_descriptors()),
        }

    def list_connectors(self) -> tuple[Any, ...]:
        return self._build_sdk().list_connectors()

    def question_history(self) -> tuple[str, ...]:
        return tuple(
            _coalesce_history_text(record).strip()
            for record in self.history
            if record.role is MessageRole.USER and _coalesce_history_text(record).strip()
        )

    def project_tools(self, task_hint: str) -> ProjectedToolSurface:
        return self._build_sdk().project_tool_surface(task_hint=task_hint, token_budget=self.config.tool_budget)

    def preview_query(self, query_text: str) -> QueryResolutionResponse:
        sdk = self._build_sdk()
        return self.query_resolver.resolve(
            QueryResolutionRequest(
                user_id=self.session.user_id,
                session=self.session,
                query_text=query_text,
                tool_token_budget=self.config.tool_budget,
            ),
            history=tuple(self.history),
            memories=tuple(self.memories),
            working_items=tuple(self.working_buffer.items),
            tools=sdk.tool_definitions(),
        )

    def memory_lookup(self, query_text: str) -> Mapping[str, Any]:
        sdk = self._build_sdk()
        result = sdk.invoke(
            AgentSdkInvocationRequest(
                operation="memory_read",
                params={"query": query_text, "top_k": 5},
                task_hint=query_text,
            )
        )
        return _result_payload(result)

    def terminal_execute(
        self,
        command: str,
        *,
        working_dir: str | None = None,
        shell: str = "auto",
        timeout_ms: int = 15000,
        allow_write: bool = False,
    ) -> Mapping[str, Any]:
        sdk = self._build_sdk()
        result = sdk.invoke(
            AgentSdkInvocationRequest(
                operation="terminal_execute",
                params={
                    "command": command,
                    "working_dir": working_dir,
                    "shell": shell,
                    "timeout_ms": timeout_ms,
                    "allow_write": allow_write,
                },
                task_hint=command,
            )
        )
        payload = _result_payload(result)
        if payload:
            return payload
        if result.response.error is not None:
            return {
                "command": command,
                "shell": shell,
                "working_dir": working_dir or str(_project_root()),
                "exit_code": -1,
                "duration_ms": 0,
                "stdout": "",
                "stderr": result.response.error.message,
                "timed_out": result.response.status.value == "timeout",
                "summary": result.response.error.message,
            }
        return {}

    def last_trace_snapshot(self) -> dict[str, Any]:
        if self.last_turn is None:
            return {}
        result = self.last_turn.loop_result
        context_trace = result.context_trace
        verdict = result.verdict
        return {
            "question": self.last_turn.question,
            "provider": self.last_turn.provider,
            "loop_id": str(result.loop_id),
            "cognitive_mode": None if result.cognitive_mode is None else result.cognitive_mode.value,
            "last_completed_phase": None if result.last_completed_phase is None else result.last_completed_phase.value,
            "paused": result.paused,
            "upgraded_from_fast_path": result.upgraded_from_fast_path,
            "signal_density": None if context_trace is None else _safe_round(context_trace.signal_density),
            "duplication_rate": None if context_trace is None else _safe_round(context_trace.duplication_rate),
            "staleness_index": None if context_trace is None else _safe_round(context_trace.staleness_index),
            "stable_prefix_tokens": None if context_trace is None else context_trace.stable_prefix_tokens,
            "evidence_count": 0 if result.evidence is None else len(result.evidence.items),
            "action_count": len(result.action_outcomes),
            "committed_writes": len(result.committed_writes),
            "verification_passed": None if verdict is None else verdict.passed,
            "quality_mean": None if verdict is None else _safe_round(verdict.quality.mean_score()),
            "escalation_reason": None if result.escalation_reason is None else result.escalation_reason.value,
        }

    def answer_realtime(self, query_text: str) -> Iterator[RealtimeQaEvent]:
        normalized_query = " ".join(query_text.strip().split())
        if not normalized_query:
            yield RealtimeQaEvent(kind="error", title="input", message="Please enter a non-empty prompt.")
            return

        query_item = self._new_working_item(
            text=f"Active user question: {normalized_query}",
            utility_score=0.92,
            carry_forward=True,
            metadata={"kind": "user_query"},
        )
        self.working_buffer.add(query_item)
        self.working_buffer.prune_to_capacity(query_text=normalized_query)

        sdk = self._build_sdk()
        sdk_snapshot = sdk.snapshot()
        tool_surface = sdk.project_tool_surface(task_hint=normalized_query, token_budget=self.config.tool_budget)
        yield RealtimeQaEvent(
            kind="sdk",
            title="sdk",
            message=(
                f"Loaded {len(sdk_snapshot.connectors)} connectors and projected "
                f"{len(tool_surface.tool_definitions)} tools for the current question."
            ),
            payload={
                "connectors": len(sdk_snapshot.connectors),
                "tools": len(tool_surface.tool_definitions),
                "provider": self.provider_name,
            },
        )

        query_preview = self.query_resolver.resolve(
            QueryResolutionRequest(
                user_id=self.session.user_id,
                session=self.session,
                query_text=normalized_query,
                tool_token_budget=self.config.tool_budget,
            ),
            history=tuple(self.history),
            memories=tuple(self.memories),
            working_items=tuple(self.working_buffer.items),
            tools=sdk.tool_definitions(),
        )
        yield RealtimeQaEvent(
            kind="query",
            title="query",
            message=(
                f"Intent={query_preview.plan.intent.intent_class.value}, "
                f"subqueries={len(query_preview.plan.subqueries)}, "
                f"clarification={query_preview.plan.needs_clarification}."
            ),
            payload={
                "intent": query_preview.plan.intent.intent_class.value,
                "subqueries": len(query_preview.plan.subqueries),
                "clarification": query_preview.plan.needs_clarification,
            },
        )

        context_provider = "vllm_responses" if self.provider_name == "vllm" else self.config.context_provider
        context_preview = sdk.build_context(
            query_text=normalized_query,
            provider=context_provider,
            tool_budget=self.config.tool_budget,
        )
        context_trace = context_preview.trace
        yield RealtimeQaEvent(
            kind="context",
            title="context",
            message=(
                f"Compiled context: tokens={context_trace.total_tokens}, "
                f"signal={context_trace.signal_density:.3f}, "
                f"duplication={context_trace.duplication_rate:.3f}, "
                f"staleness={context_trace.staleness_index:.3f}."
            ),
            payload={
                "total_tokens": context_trace.total_tokens,
                "signal_density": context_trace.signal_density,
                "duplication_rate": context_trace.duplication_rate,
                "staleness_index": context_trace.staleness_index,
                "stable_prefix_tokens": context_trace.stable_prefix_tokens,
            },
        )

        retrieval_preview = sdk.invoke(
            AgentSdkInvocationRequest(
                operation="retrieval_plan",
                params={
                    "query": normalized_query,
                    "session_summary": self.session.summary_text or "",
                    "turn_count": self.session.turn_count,
                },
                task_hint=normalized_query,
            )
        )
        retrieval_payload = _result_payload(retrieval_preview)
        if retrieval_payload:
            yield RealtimeQaEvent(
                kind="retrieval",
                title="retrieval",
                message=(
                    f"Primary model={retrieval_payload.get('primary_model')}, "
                    f"query_type={retrieval_payload.get('query_type')}, "
                    f"reasoning={retrieval_payload.get('reasoning_effort')}."
                ),
                payload=dict(retrieval_payload),
            )

        loop_request = LoopExecutionRequest(
            user_id=self.session.user_id,
            session=self.session,
            query_text=normalized_query,
            scopes=(
                "tools.read",
                "tools.write",
                "tools.admin",
                "tools.browser",
                "tools.vision",
                "tools.delegate",
                "tools.terminal",
            ),
            stop_after_phase=LoopPhase.CONTEXT_ASSEMBLE,
        )
        loop_result = self.loop_engine.execute(
            loop_request,
            history=tuple(self.history),
            memories=self.memories,
            working_items=tuple(self.working_buffer.items),
        )
        yield self._phase_event(loop_result)

        while loop_result.paused and loop_result.next_phase is not None:
            loop_result = self.loop_engine.resume(
                loop_result.loop_id,
                history=tuple(self.history),
                memories=self.memories,
                working_items=tuple(self.working_buffer.items),
                stop_after_phase=loop_result.next_phase,
            )
            yield self._phase_event(loop_result)

        sdk_after = self._build_sdk()
        memory_preview = sdk_after.invoke(
            AgentSdkInvocationRequest(
                operation="memory_read",
                params={"query": normalized_query, "top_k": 4},
                task_hint=normalized_query,
            )
        )
        registry_preview = sdk_after.invoke(
            AgentSdkInvocationRequest(
                operation="tool_registry_list",
                params={"task_hint": normalized_query, "token_budget": self.config.tool_budget},
                task_hint=normalized_query,
            )
        )
        compiled_context_preview = sdk_after.invoke(
            AgentSdkInvocationRequest(
                operation="data_compile_context",
                params={
                    "query": normalized_query,
                    "session_id": str(self.session.session_id),
                    "tool_budget": min(self.config.tool_budget, 256),
                },
                task_hint=normalized_query,
            )
        )
        terminal_followup_preview: AgentSdkInvocationResult | None = None
        if self._should_run_terminal_followup(normalized_query, loop_result):
            terminal_followup_preview = self._run_terminal_followup(sdk_after, normalized_query)
            terminal_followup_payload = _result_payload(terminal_followup_preview)
            if terminal_followup_payload:
                yield RealtimeQaEvent(
                    kind="diagnostics",
                    title="diagnostics",
                    message=(
                        f"Ran autonomous terminal diagnostics with shell={terminal_followup_payload.get('shell')} "
                        f"exit_code={terminal_followup_payload.get('exit_code')}."
                    ),
                    payload=dict(terminal_followup_payload),
                )

        yield RealtimeQaEvent(
            kind="answer_start",
            title="answer",
            message=f"Synthesizing final grounded answer with provider={self.provider_name}.",
            payload={"provider": self.provider_name, "model": self.provider_model},
        )

        answer_text = ""
        answer_streamed = False
        streamed_chunks: list[str] = []
        synthesis_context = self._answer_context_lines(
            question=normalized_query,
            query_preview=query_preview,
            context_preview=context_preview,
            loop_result=loop_result,
            memory_preview=memory_preview,
            registry_preview=registry_preview,
            compiled_context_preview=compiled_context_preview,
            terminal_followup_preview=terminal_followup_preview,
            sdk_snapshot=sdk_snapshot,
            tool_surface=tool_surface,
        )

        if self.provider_suite is not None and self.provider_suite.text_model is not None:
            text_model = self.provider_suite.text_model
            try:
                if self.config.stream_answer and hasattr(text_model, "stream_complete"):
                    for chunk in text_model.stream_complete(
                        system_prompt=self._answer_system_prompt(),
                        user_input=normalized_query,
                        context=synthesis_context,
                        max_tokens=self.config.answer_max_tokens,
                    ):
                        if not chunk:
                            continue
                        streamed_chunks.append(chunk)
                        answer_streamed = True
                        yield RealtimeQaEvent(
                            kind="answer_delta",
                            title="answer",
                            message=chunk,
                            payload={"provider": self.provider_name},
                        )
                    answer_text = "".join(streamed_chunks).strip()
                if not answer_text:
                    answer_text = text_model.complete(
                        system_prompt=self._answer_system_prompt(),
                        user_input=normalized_query,
                        context=synthesis_context,
                        max_tokens=self.config.answer_max_tokens,
                        temperature=0.0,
                    ).strip()
            except Exception as exc:  # pragma: no cover - runtime fallback path
                if streamed_chunks:
                    answer_text = "".join(streamed_chunks).strip()
                else:
                    self.provider_note = f"provider fallback: {exc}"
                    yield RealtimeQaEvent(
                        kind="provider_fallback",
                        title="provider",
                        message=f"Provider synthesis failed, switching to deterministic fallback: {exc}",
                        payload={"provider": self.provider_name},
                    )
                    answer_text = self._fallback_answer(
                        question=normalized_query,
                        query_preview=query_preview,
                        loop_result=loop_result,
                        memory_preview=memory_preview,
                        terminal_followup_preview=terminal_followup_preview,
                    )
        else:
            answer_text = self._fallback_answer(
                question=normalized_query,
                query_preview=query_preview,
                loop_result=loop_result,
                memory_preview=memory_preview,
                terminal_followup_preview=terminal_followup_preview,
            )
        if not answer_text:
            answer_text = self._fallback_answer(
                question=normalized_query,
                query_preview=query_preview,
                loop_result=loop_result,
                memory_preview=memory_preview,
                terminal_followup_preview=terminal_followup_preview,
            )

        user_record = self._history_record(
            role=MessageRole.USER,
            content=normalized_query,
            turn_index=len(self.history) + 1,
        )
        assistant_record = self._history_record(
            role=MessageRole.ASSISTANT,
            content=answer_text,
            turn_index=len(self.history) + 2,
            model_id=self.provider_model or self.session.model_id,
            finish_reason=FinishReason.STOP,
            prompt_tokens=context_preview.trace.total_tokens,
            completion_tokens=max(len(answer_text.split()), 1),
            latency_ms=max(loop_result.latency_ns // 1_000_000, 0),
        )
        self.history.extend((user_record, assistant_record))
        self.session.turn_count = len(self.history)
        self.session.updated_at = _utc_now()
        self.session.summary_text = update_session_summary(
            existing_summary=self.session.summary_text or "",
            new_turns=(user_record, assistant_record),
            max_tokens=160,
        )
        self.session.summary_token_count = max(len((self.session.summary_text or "").split()), 0)

        answer_item = self._new_working_item(
            text=f"Assistant answer summary: {answer_text[:240]}",
            utility_score=0.68,
            carry_forward=True,
            metadata={"kind": "assistant_answer"},
        )
        self.working_buffer.items = self.working_buffer.carry_forward_items(ratio=0.40)
        self.working_buffer.add(answer_item)
        self.working_buffer.prune_to_capacity(query_text=normalized_query)

        checkpoint = build_session_checkpoint(
            session=self.session,
            checkpoint_seq=len(self.session_checkpoints) + 1,
            summary_text=self.session.summary_text or "",
            summary_version=len(self.session_checkpoints) + 1,
            turn_count=self.session.turn_count,
            working_items=tuple(self.working_buffer.items),
            metadata={
                "provider": self.provider_name,
                "loop_id": str(loop_result.loop_id),
                "audit_id": str(loop_result.audit_id),
            },
        )
        self.session_checkpoints.append(checkpoint)

        self.last_turn = ChatbotTurnArtifacts(
            question=normalized_query,
            answer=answer_text,
            provider=self.provider_name,
            sdk_snapshot=sdk_snapshot,
            tool_surface=tool_surface,
            query_preview=query_preview,
            context_preview=context_preview,
            retrieval_preview=retrieval_preview,
            memory_preview=memory_preview,
            registry_preview=registry_preview,
            compiled_context_preview=compiled_context_preview,
            terminal_followup_preview=terminal_followup_preview,
            loop_result=loop_result,
        )

        if not answer_streamed:
            yield RealtimeQaEvent(
                kind="answer_delta",
                title="answer",
                message=answer_text,
                payload={"provider": self.provider_name},
            )
        yield RealtimeQaEvent(
            kind="answer_final",
            title="answer",
            message=answer_text,
            payload={
                "provider": self.provider_name,
                "model": self.provider_model,
                "streamed": answer_streamed,
                "checkpoint_seq": checkpoint.checkpoint_seq,
            },
        )

    def _build_sdk(self) -> AgentSdk:
        return AgentSdk.bootstrap_openagentbench(
            session=self.session,
            history=tuple(self.history),
            memories=self.memories,
            working_items=tuple(self.working_buffer.items),
            extra_descriptors=(build_terminal_execute_descriptor(_project_root()),),
        )

    def _build_session(self, *, model_id: str) -> SessionRecord:
        now = _utc_now()
        system_prompt = (
            "You are the OpenAgentBench realtime Q&A chatbot. "
            "Use the framework traces as the ground truth for every answer."
        )
        return SessionRecord(
            session_id=uuid4(),
            user_id=uuid4(),
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=1),
            status=SessionStatus.ACTIVE,
            model_id=model_id,
            context_window_size=32_000,
            system_prompt_hash=hash_normalized_text(system_prompt),
            system_prompt_tokens=max(len(system_prompt.split()), 1),
            max_response_tokens=1_024,
            turn_count=0,
            summary_text="Interactive Q&A session over the OpenAgentBench modules.",
            summary_token_count=9,
            system_prompt_text=system_prompt,
        )

    def _session_model_id(self) -> str:
        if self.provider_model:
            return self.provider_model
        if self.provider_name == "vllm":
            return self._vllm_model_name()
        if self.provider_name == "openai":
            return self._openai_model_name()
        return self._default_openai_model

    def _bootstrap_memories(self) -> list[MemoryRecord]:
        now = _utc_now()
        seeds = (
            (
                MemoryTier.SEMANTIC,
                ProvenanceType.FACT,
                "OpenAgentBench module graph: agent_data stores records, agent_query resolves intent, "
                "agent_retrieval ranks evidence, agent_tools exposes tools, agent_context compiles cyclic "
                "context, agent_loop executes bounded phases, and agent_sdk projects connector surfaces.",
                ("modules", "architecture"),
            ),
            (
                MemoryTier.SEMANTIC,
                ProvenanceType.INSTRUCTION,
                "Loop invariant: memory writes should be committed only after verification and commit gating.",
                ("memory", "constraint"),
            ),
            (
                MemoryTier.PROCEDURAL,
                ProvenanceType.SYSTEM_INFERRED,
                "SDK connectors project tool surfaces into OpenAI, vLLM, MCP, JSON-RPC, function, and A2A shapes.",
                ("sdk", "connectors", "protocols"),
            ),
        )
        records: list[MemoryRecord] = []
        for tier, provenance, content, tags in seeds:
            records.append(
                MemoryRecord(
                    memory_id=uuid4(),
                    user_id=self.session.user_id,
                    session_id=None,
                    memory_tier=tier,
                    memory_scope=MemoryScope.GLOBAL,
                    content_text=content,
                    content_embedding=None,
                    content_hash=hash_normalized_text(content),
                    provenance_type=provenance,
                    provenance_turn_id=None,
                    confidence=0.98,
                    relevance_accumulator=4.0,
                    access_count=0,
                    last_accessed_at=None,
                    created_at=now,
                    updated_at=now,
                    expires_at=None,
                    is_active=True,
                    is_validated=True,
                    token_count=max(len(content.split()), 1),
                    tags=tags,
                    metadata={"seeded_by": "realtime_qa_chatbot"},
                )
            )
        return records

    def _working_capacity(self) -> int:
        return compute_working_memory_capacity(
            context_window_size=self.session.context_window_size,
            system_prompt_tokens=self.session.system_prompt_tokens,
            tool_budget=self.config.tool_budget,
            output_reserve=self.session.max_response_tokens,
            session_claim=512,
            episodic_claim=320,
            semantic_claim=640,
            procedural_claim=384,
        )

    def _new_working_item(
        self,
        *,
        text: str,
        utility_score: float,
        carry_forward: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkingMemoryItem:
        return WorkingMemoryItem(
            item_id=uuid4(),
            user_id=self.session.user_id,
            session_id=self.session.session_id,
            step_id=uuid4(),
            content_text=text,
            token_count=max(len(text.split()), 1),
            modality=Modality.TEXT,
            utility_score=utility_score,
            carry_forward=carry_forward,
            metadata=dict(metadata or {}),
        )

    def _history_record(
        self,
        *,
        role: MessageRole,
        content: str,
        turn_index: int,
        model_id: str | None = None,
        finish_reason: FinishReason | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
    ) -> HistoryRecord:
        return HistoryRecord(
            message_id=uuid4(),
            session_id=self.session.session_id,
            user_id=self.session.user_id,
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
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            api_call_id=None,
            created_at=_utc_now(),
        )

    def _resolve_provider_name(self) -> str:
        requested = self.config.provider
        if requested == "auto":
            if os.getenv("OPENAI_API_KEY"):
                return "openai"
            if self._vllm_base_url():
                return "vllm"
            return "heuristic"
        return requested

    def _openai_model_name(self) -> str:
        return (
            self.config.model
            or os.getenv("OPENAGENTBENCH_OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL")
            or self._default_openai_model
        )

    def _vllm_model_name(self) -> str:
        return (
            self.config.model
            or os.getenv("OPENAGENTBENCH_VLLM_MODEL")
            or os.getenv("VLLM_MODEL")
            or self._default_vllm_model
        )

    def _vllm_base_url(self) -> str | None:
        return self.config.base_url or os.getenv("OPENAGENTBENCH_VLLM_BASE_URL") or os.getenv("VLLM_BASE_URL")

    def _build_provider_suite(self) -> tuple[ProviderSuite | None, str | None, str]:
        if self.provider_name == "heuristic":
            return None, None, "No external provider configured; deterministic synthesis is active."
        if OpenAI is None:
            return None, None, "openai package is not installed; deterministic synthesis is active."

        if self.provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None, None, "OPENAI_API_KEY is missing; deterministic synthesis is active."
            model = self._openai_model_name()
            client = OpenAI(api_key=api_key, base_url=self.config.base_url or os.getenv("OPENAI_BASE_URL"))
            suite = self.sdk_provider_factory.build_openai(
                client=client,
                model=model,
                reasoning_effort=self.config.reasoning_effort,
            )
            return suite, model, "OpenAI provider loaded from the local environment."

        if self.provider_name == "vllm":
            base_url = self._vllm_base_url()
            if not base_url:
                return None, None, "VLLM base URL is missing; deterministic synthesis is active."
            model = self._vllm_model_name()
            client = OpenAI(
                api_key=os.getenv("OPENAGENTBENCH_VLLM_API_KEY") or os.getenv("VLLM_API_KEY") or "EMPTY",
                base_url=base_url,
            )
            suite = self.sdk_provider_factory.build_vllm(
                client=client,
                model=model,
                reasoning_effort=self.config.reasoning_effort,
            )
            return suite, model, "vLLM OpenAI-compatible provider loaded from the local environment."

        return None, None, "Unknown provider requested; deterministic synthesis is active."

    def _answer_system_prompt(self) -> str:
        return (
            "You are the realtime OpenAgentBench Q&A demo. "
            "Answer only from the supplied framework trace, evidence, memory, context, and SDK diagnostics. "
            "Do not invent missing facts. If the trace is incomplete, say so explicitly. "
            "When terminal diagnostics are present, state the root cause first and then give the concrete fix. "
            "Prefer precise technical answers with short paragraphs."
        )

    def _should_run_terminal_followup(self, question: str, loop_result: LoopExecutionResult) -> bool:
        lowered = question.lower()
        if not any(token in lowered for token in ("powershell", "pwsh", "terminal", "shell", "bash", "cmd")):
            return False
        if not any(
            token in lowered
            for token in ("help", "fix", "solve", "check", "diagnose", "problem", "issue", "not working", "working properly")
        ):
            return False
        terminal_outcomes = [outcome for outcome in loop_result.action_outcomes if outcome.tool_id == "terminal_execute"]
        if not terminal_outcomes:
            return True
        combined = " ".join(
            " ".join(
                str(outcome.output.get(field, ""))
                for field in ("command", "stdout", "stderr", "summary")
            )
            for outcome in terminal_outcomes
        ).lower()
        return "diag:" not in combined

    def _run_terminal_followup(self, sdk: AgentSdk, question: str) -> AgentSdkInvocationResult:
        inferred = infer_terminal_request(question)
        return sdk.invoke(
            AgentSdkInvocationRequest(
                operation="terminal_execute",
                params={
                    "command": inferred["command"],
                    "shell": inferred["shell"],
                    "timeout_ms": 15000,
                    "max_output_chars": 6000,
                    "allow_write": False,
                },
                task_hint=question,
            )
        )

    def _terminal_diagnosis_summary(
        self,
        *,
        question: str,
        loop_result: LoopExecutionResult,
        terminal_followup_preview: AgentSdkInvocationResult | None,
    ) -> str | None:
        segments: list[str] = []
        for outcome in loop_result.action_outcomes:
            if outcome.tool_id != "terminal_execute":
                continue
            segments.extend(
                str(outcome.output.get(field, ""))
                for field in ("command", "stdout", "stderr", "summary")
            )
        followup_payload = _result_payload(terminal_followup_preview)
        if followup_payload:
            segments.extend(
                str(followup_payload.get(field, ""))
                for field in ("command", "stdout", "stderr", "summary")
            )
        combined = "\n".join(segment for segment in segments if segment).lower()
        lowered_question = question.lower()

        if "powershell" in lowered_question or "pwsh" in lowered_question:
            if "diag: powershell_runtime=missing" in combined:
                return "PowerShell runtime was not found in the current environment."
            if "diag: help_command=" in combined and "diag: sample_help_name=" in combined:
                if lowered_question.startswith("help ") or " help " in f" {lowered_question} ":
                    return (
                        "PowerShell itself is available and the help system resolves correctly. "
                        "The failure is command usage: a free-form sentence was treated as arguments to `help`, "
                        "but `help` expects a concrete command name such as `help Get-ChildItem`. "
                        "Use a real command to troubleshoot the shell, or ask the assistant without starting the line with `help`."
                    )
                return "PowerShell is available and the help system resolves correctly."
            if "a positional parameter cannot be found" in combined:
                return (
                    "PowerShell is responding, but the text was parsed as an invalid command invocation. "
                    "This is a command-syntax problem, not a shell startup failure."
                )
            if "not recognized as the name of a cmdlet" in combined:
                return "PowerShell is running, but the referenced command was not found."

        if "diag: shell=bash" in combined:
            return "The shell diagnostics completed successfully and the local bash environment is available."
        if "diag: shell=powershell" in combined:
            return "The shell diagnostics completed successfully and the local PowerShell environment is available."
        return None

    def _answer_context_lines(
        self,
        *,
        question: str,
        query_preview: QueryResolutionResponse,
        context_preview: CompiledCycleContext,
        loop_result: LoopExecutionResult,
        memory_preview: AgentSdkInvocationResult | None,
        registry_preview: AgentSdkInvocationResult | None,
        compiled_context_preview: AgentSdkInvocationResult | None,
        terminal_followup_preview: AgentSdkInvocationResult | None,
        sdk_snapshot: AgentSdkSnapshot,
        tool_surface: ProjectedToolSurface,
    ) -> tuple[str, ...]:
        context_trace = context_preview.trace
        verdict = loop_result.verdict
        memory_payload = _result_payload(memory_preview)
        registry_payload = _result_payload(registry_preview)
        compiled_payload = _result_payload(compiled_context_preview)
        followup_payload = _result_payload(terminal_followup_preview)
        recent_user_questions = [
            _coalesce_history_text(record).strip()
            for record in self.history
            if record.role is MessageRole.USER
        ][-8:]
        recent_assistant_answers = [
            _coalesce_history_text(record).strip()
            for record in self.history
            if record.role is MessageRole.ASSISTANT
        ][-4:]
        evidence_lines = [
            item.content.replace("\n", " ").strip()
            for item in (loop_result.evidence.items if loop_result.evidence is not None else ())
        ][:4]
        terminal_lines = [
            (
                f"command={outcome.output.get('command')} "
                f"exit_code={outcome.output.get('exit_code')} "
                f"stdout={str(outcome.output.get('stdout', '')).replace(chr(10), ' ')[:400]} "
                f"stderr={str(outcome.output.get('stderr', '')).replace(chr(10), ' ')[:200]}"
            ).strip()
            for outcome in loop_result.action_outcomes
            if outcome.tool_id == "terminal_execute"
        ][:2]
        if followup_payload:
            terminal_lines.append(
                (
                    f"command={followup_payload.get('command')} "
                    f"exit_code={followup_payload.get('exit_code')} "
                    f"stdout={str(followup_payload.get('stdout', '')).replace(chr(10), ' ')[:400]} "
                    f"stderr={str(followup_payload.get('stderr', '')).replace(chr(10), ' ')[:200]}"
                ).strip()
            )
        memory_lines = [
            str(item.get("content", "")).strip()
            for item in memory_payload.get("items", ())
            if isinstance(item, dict)
        ][:3]
        tool_names = [tool_definition["function"]["name"] for tool_definition in tool_surface.tool_definitions[:6]]
        terminal_summary = self._terminal_diagnosis_summary(
            question=question,
            loop_result=loop_result,
            terminal_followup_preview=terminal_followup_preview,
        )
        lines = [
            f"User question: {question}",
            f"Query intent: {query_preview.plan.intent.intent_class.value}",
            f"Resolved query: {query_preview.plan.rewrite.resolved_query}",
            f"Expanded query: {query_preview.plan.rewrite.expanded_query}",
            f"Subquery count: {len(query_preview.plan.subqueries)}",
            f"Connector count: {len(sdk_snapshot.connectors)}",
            f"Projected tools: {', '.join(tool_names)}",
            f"Context signal density: {context_trace.signal_density:.3f}",
            f"Context duplication rate: {context_trace.duplication_rate:.3f}",
            f"Context staleness index: {context_trace.staleness_index:.3f}",
            f"Context stable prefix tokens: {context_trace.stable_prefix_tokens}",
            f"Loop cognitive mode: {None if loop_result.cognitive_mode is None else loop_result.cognitive_mode.value}",
            f"Loop output: {loop_result.output_text or 'No synthesized loop text was produced.'}",
            f"Loop committed writes: {len(loop_result.committed_writes)}",
            f"Tool registry count: {registry_payload.get('count', 0)}",
            f"Compiled legacy context messages: {len(compiled_payload.get('messages', ())) if isinstance(compiled_payload, dict) else 0}",
        ]
        if evidence_lines:
            lines.append("Evidence: " + " | ".join(evidence_lines))
        if recent_user_questions:
            lines.append("Recent user questions: " + " | ".join(recent_user_questions))
        if recent_assistant_answers:
            lines.append("Recent assistant answers: " + " | ".join(recent_assistant_answers))
        if memory_lines:
            lines.append("Memory hits: " + " | ".join(memory_lines))
        if terminal_lines:
            lines.append("Terminal observations: " + " | ".join(terminal_lines))
        if terminal_summary:
            lines.append("Terminal diagnosis: " + terminal_summary)
        if verdict is not None:
            lines.append(
                "Verification: "
                f"passed={verdict.passed}, "
                f"correctness={verdict.quality.correctness:.3f}, "
                f"completeness={verdict.quality.completeness:.3f}, "
                f"grounded={verdict.quality.grounded:.3f}."
            )
        if loop_result.escalation_reason is not None:
            lines.append(f"Escalation: {loop_result.escalation_reason.value}")
        return tuple(lines)

    def _fallback_answer(
        self,
        *,
        question: str,
        query_preview: QueryResolutionResponse,
        loop_result: LoopExecutionResult,
        memory_preview: AgentSdkInvocationResult | None,
        terminal_followup_preview: AgentSdkInvocationResult | None,
    ) -> str:
        parts: list[str] = []
        terminal_summary = self._terminal_diagnosis_summary(
            question=question,
            loop_result=loop_result,
            terminal_followup_preview=terminal_followup_preview,
        )
        if terminal_summary:
            parts.append(terminal_summary)
            terminal_outcomes = [outcome for outcome in loop_result.action_outcomes if outcome.tool_id == "terminal_execute"]
            if terminal_outcomes:
                observed = str(terminal_outcomes[0].output.get("stdout", "")).strip()
                if observed:
                    compact = " | ".join(line.strip() for line in observed.splitlines() if line.strip())
                    parts.append("Observed diagnostic output: " + compact[:320])
        output_text = loop_result.output_text.strip()
        if output_text and not _looks_like_structured_dump(output_text):
            parts.append(output_text)
        elif not terminal_summary:
            parts.append(
                f"Resolved query: {query_preview.plan.rewrite.resolved_query}. "
                f"The loop did not emit a richer synthesized action summary for this turn."
            )
        memory_payload = _result_payload(memory_preview)
        memory_hits = [
            str(item.get("content", "")).strip()
            for item in memory_payload.get("items", ())
            if isinstance(item, dict)
        ][:2]
        if memory_hits and not terminal_summary:
            parts.append("Relevant memory: " + " | ".join(memory_hits))
        if loop_result.verdict is not None and not terminal_summary:
            parts.append(
                "Verification: "
                f"passed={loop_result.verdict.passed}, "
                f"grounding={loop_result.verdict.grounding_score:.3f}, "
                f"tool_success={loop_result.verdict.tool_success_rate:.3f}."
            )
        return "\n\n".join(part for part in parts if part).strip()

    def _phase_event(self, result: LoopExecutionResult) -> RealtimeQaEvent:
        phase = result.last_completed_phase
        if phase is None:
            return RealtimeQaEvent(kind="phase", title="loop", message="No phase was completed.")
        if phase is LoopPhase.CONTEXT_ASSEMBLE:
            trace = result.context_trace
            message = (
                "Context assembled."
                if trace is None
                else (
                    f"Context assembled with signal={trace.signal_density:.3f}, "
                    f"stable_prefix={trace.stable_prefix_tokens}, total_tokens={trace.total_tokens}."
                )
            )
        elif phase is LoopPhase.PLAN and result.plan is not None:
            message = f"Planned {len(result.plan.actions)} action(s) with {len(result.plan.constraints)} constraint(s)."
        elif phase is LoopPhase.RETRIEVE and result.evidence is not None:
            message = (
                f"Retrieved {len(result.evidence.items)} evidence item(s) with "
                f"quality={result.evidence.quality_score:.3f}."
            )
        elif phase is LoopPhase.ACT:
            successful = sum(1 for outcome in result.action_outcomes if outcome.status.value == "success")
            message = f"Executed {len(result.action_outcomes)} action(s); successful={successful}."
        elif phase is LoopPhase.METACOGNITIVE_CHECK and result.metacognitive is not None:
            message = (
                f"Metacognitive decision={result.metacognitive.decision.value}, "
                f"score={result.metacognitive.score:.3f}."
            )
        elif phase is LoopPhase.VERIFY and result.verdict is not None:
            message = (
                f"Verification passed={result.verdict.passed}, "
                f"quality_mean={result.verdict.quality.mean_score():.3f}."
            )
        elif phase is LoopPhase.REPAIR:
            message = f"Applied repair cycle count={len(result.repair_history)}."
        elif phase is LoopPhase.COMMIT:
            message = f"Committed {len(result.committed_writes)} memory write(s)."
        elif phase is LoopPhase.FAIL:
            reason = None if result.escalation_reason is None else result.escalation_reason.value
            message = f"Loop failed with escalation={reason}."
        else:
            message = f"Completed phase {phase.value}."
        return RealtimeQaEvent(
            kind="phase",
            title=phase.value,
            message=message,
            payload={
                "phase": phase.value,
                "cognitive_mode": None if result.cognitive_mode is None else result.cognitive_mode.value,
                "paused": result.paused,
                "next_phase": None if result.next_phase is None else result.next_phase.value,
            },
        )


__all__ = [
    "ChatbotConfig",
    "ChatbotTurnArtifacts",
    "RealtimeQaChatbot",
    "RealtimeQaEvent",
    "load_env_file",
]
