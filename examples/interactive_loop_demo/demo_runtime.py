"""Legacy interactive demo compatibility layer backed by the current loop engine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from openagentbench.agent_data import SessionRecord, SessionStatus, hash_normalized_text
from openagentbench.agent_loop import LoopExecutionRequest, LoopExecutionResult, LoopPhase
from openagentbench.agent_sdk import AgentSdkProviderFactory

from examples.realtime_qa_chatbot.tools import RealtimeQaLoopEngine

from .demo_env import DemoConfig, load_demo_config


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class LoopProgressEvent:
    phase: str
    message: str
    metrics: Mapping[str, Any] = field(default_factory=dict)


def _build_session(*, model_id: str) -> SessionRecord:
    now = _utc_now()
    system_prompt = (
        "You are the OpenAgentBench interactive loop demo. "
        "Use the loop trace and tool outputs as the source of truth."
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
        max_response_tokens=768,
        turn_count=0,
        summary_text="Compatibility demo session over the OpenAgentBench loop.",
        summary_token_count=9,
        system_prompt_text=system_prompt,
    )


def _normalize_query(query: str) -> str:
    lowered = query.lower()
    if "workspace unit tests" in lowered and "python3 -m unittest -q test_demo_stats.py" not in lowered:
        return "Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result."
    return query


def _first_terminal_outcome(result: LoopExecutionResult) -> Mapping[str, Any] | None:
    for outcome in result.action_outcomes:
        if outcome.tool_id == "terminal_execute":
            return outcome.output
    return None


def _deterministic_summary(result: LoopExecutionResult) -> str:
    terminal_output = _first_terminal_outcome(result)
    if terminal_output is not None:
        exit_code = terminal_output.get("exit_code")
        stdout = str(terminal_output.get("stdout", "")).strip()
        stderr = str(terminal_output.get("stderr", "")).strip()
        lines = [f"Terminal command completed with exit_code={exit_code}."]
        if stdout:
            lines.append(stdout[:800])
        if stderr:
            lines.append(stderr[:400])
        return "\n".join(lines).strip()
    if result.output_text.strip():
        return result.output_text.strip()
    return "The loop completed without a richer synthesized terminal summary."


def _provider_context_lines(query: str, result: LoopExecutionResult) -> tuple[str, ...]:
    terminal_output = _first_terminal_outcome(result) or {}
    stdout = str(terminal_output.get("stdout", "")).strip()
    stderr = str(terminal_output.get("stderr", "")).strip()
    return (
        f"User task: {query}",
        f"Last completed phase: {None if result.last_completed_phase is None else result.last_completed_phase.value}",
        f"Loop paused: {result.paused}",
        f"Terminal exit code: {terminal_output.get('exit_code')}",
        f"Terminal stdout: {stdout}",
        f"Terminal stderr: {stderr}",
        f"Committed writes: {len(result.committed_writes)}",
    )


class DemoLoopApplication:
    def __init__(self, config: DemoConfig | None = None) -> None:
        self.config = config or load_demo_config()
        self.provider_name = "demo"
        self.session = _build_session(model_id="gpt-4.1-mini")
        self.loop_engine = RealtimeQaLoopEngine(project_root=self.config.workspace_root)

    def _loop_event(self, result: LoopExecutionResult) -> LoopProgressEvent:
        phase = "loop" if result.last_completed_phase is None else result.last_completed_phase.value
        trace = result.context_trace
        metrics: dict[str, Any] = {}
        if trace is not None:
            metrics = {
                "signal_density": trace.signal_density,
                "stable_prefix_tokens": trace.stable_prefix_tokens,
                "section_allocations": dict(trace.section_allocations),
                "duplication_rate": trace.duplication_rate,
                "staleness_index": trace.staleness_index,
            }
        message = phase
        if phase == "act":
            terminal_output = _first_terminal_outcome(result)
            if terminal_output is not None:
                message = f"Executed terminal command with exit_code={terminal_output.get('exit_code')}."
        elif phase == "verify" and result.verdict is not None:
            message = f"Verification passed={result.verdict.passed}."
        return LoopProgressEvent(phase=phase, message=message, metrics=metrics)

    def _run_loop(
        self,
        query: str,
        *,
        on_event: Callable[[LoopProgressEvent], None] | None = None,
    ) -> LoopExecutionResult:
        normalized_query = _normalize_query(query)
        result = self.loop_engine.execute(
            LoopExecutionRequest(
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
            ),
            history=(),
            memories=(),
            working_items=(),
        )
        if on_event is not None:
            on_event(self._loop_event(result))
        while result.paused and result.next_phase is not None:
            result = self.loop_engine.resume(
                result.loop_id,
                history=(),
                memories=(),
                working_items=(),
                stop_after_phase=result.next_phase,
            )
            if on_event is not None:
                on_event(self._loop_event(result))
        return result

    def run_query(self, query: str) -> LoopExecutionResult:
        result = self._run_loop(query)
        return replace(result, output_text=_deterministic_summary(result))

    def run_query_realtime(
        self,
        query: str,
        *,
        on_event: Callable[[LoopProgressEvent], None],
    ) -> LoopExecutionResult:
        result = self._run_loop(query, on_event=on_event)
        return replace(result, output_text=_deterministic_summary(result))


class OpenAIDemoLoopApplication(DemoLoopApplication):
    def __init__(self, config: DemoConfig | None = None, *, client: Any) -> None:
        super().__init__(config=config)
        self.provider_name = "openai"
        self.client = client
        self.provider_suite = AgentSdkProviderFactory().build_openai(
            client=client,
            model=self.config.openai_model,
            reasoning_effort=self.config.openai_reasoning_effort,
            store=False,
        )

    def _synthesize_with_provider(self, query: str, result: LoopExecutionResult) -> str:
        model = self.provider_suite.text_model
        return model.complete(
            system_prompt=(
                "You are the final response layer for OpenAgentBench. "
                "Produce a compact grounded answer from the loop outputs."
            ),
            user_input=query,
            context=_provider_context_lines(query, result),
            max_tokens=self.config.openai_max_output_tokens,
            temperature=0.0,
        ).strip()

    def run_query(self, query: str) -> LoopExecutionResult:
        result = self._run_loop(query)
        return replace(result, output_text=self._synthesize_with_provider(query, result))

    def run_query_stream(
        self,
        query: str,
        *,
        on_delta: Callable[[str], None],
    ) -> LoopExecutionResult:
        result = self._run_loop(query)
        model = self.provider_suite.text_model
        chunks: list[str] = []
        for chunk in model.stream_complete(
            system_prompt=(
                "You are the final response layer for OpenAgentBench. "
                "Produce a compact grounded answer from the loop outputs."
            ),
            user_input=query,
            context=_provider_context_lines(query, result),
            max_tokens=self.config.openai_max_output_tokens,
        ):
            if not chunk:
                continue
            chunks.append(chunk)
            on_delta(chunk)
        answer = "".join(chunks).strip() or self._synthesize_with_provider(query, result)
        return replace(result, output_text=answer)


def build_application(
    *,
    config: DemoConfig | None = None,
    provider: str = "auto",
    client: Any | None = None,
) -> DemoLoopApplication:
    resolved_config = config or load_demo_config()
    if provider == "auto":
        provider = "openai" if resolved_config.openai_api_key_present else "demo"
    if provider == "openai":
        if client is None:
            raise ValueError("client is required for the compatibility OpenAI demo wrapper")
        return OpenAIDemoLoopApplication(config=resolved_config, client=client)
    return DemoLoopApplication(config=resolved_config)


__all__ = [
    "DemoLoopApplication",
    "LoopProgressEvent",
    "OpenAIDemoLoopApplication",
    "build_application",
]
