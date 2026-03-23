from __future__ import annotations

from openagentbench.agent_loop import AgentLoopEngine, LoopExecutionRequest, LoopPhase
from openagentbench.agent_query import QueryResolutionRequest, QueryResolver

from tests.test_agent_loop_integration import _history, _memories, _session, _working
from examples.interactive_loop_demo.demo_runtime import DemoLoopApplication, load_demo_config


def test_query_and_loop_expose_canonical_agent_context_artifacts() -> None:
    session = _session()
    resolver = QueryResolver()
    resolved = resolver.resolve(
        QueryResolutionRequest(
            user_id=session.user_id,
            session=session,
            query_text="Use memory and history to plan the next grounded step.",
        ),
        history=_history(session),
        memories=_memories(session),
        working_items=_working(session),
    )

    assert resolved.plan.context.compiled_context is not None
    assert resolved.plan.context.invariant_report is not None
    assert resolved.plan.context.compilation_trace is not None

    engine = AgentLoopEngine()
    result = engine.execute(
        LoopExecutionRequest(
            user_id=session.user_id,
            session=session,
            query_text="Summarize my durable database preference with grounded evidence.",
            stop_after_phase=LoopPhase.VERIFY,
        ),
        history=_history(session),
        memories=list(_memories(session)),
        working_items=_working(session),
    )

    assert result.compiled_context is not None
    assert result.context_invariants is not None
    assert result.context_trace is not None
    assert result.context_archive


def test_realtime_demo_context_event_exposes_agent_context_metrics() -> None:
    app = DemoLoopApplication(config=load_demo_config())
    events = []

    app.run_query_realtime(
        "List the available tools and explain which one can inspect memory.",
        on_event=events.append,
    )

    first = events[0]
    assert first.phase == "context_assemble"
    assert "signal_density" in first.metrics
    assert "stable_prefix_tokens" in first.metrics
    assert "section_allocations" in first.metrics
