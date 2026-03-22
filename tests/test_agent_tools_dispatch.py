from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from openagentbench.agent_tools import (
    AuthContract,
    CompositeStep,
    CompositeToolSpec,
    ExecutionContext,
    IdempotencySpec,
    InMemoryToolRegistry,
    MutationClass,
    ObservabilityContract,
    SideEffectManifest,
    TimeoutClass,
    ToolDescriptor,
    ToolExecutionEngine,
    ToolInvocationRequest,
    ToolSourceType,
    ToolStatus,
    TypeClass,
)


def _context(*, scopes: tuple[str, ...]) -> ExecutionContext:
    return ExecutionContext(
        user_id=uuid4(),
        agent_id=uuid4(),
        session_id=uuid4(),
        scopes=scopes,
        trace_id="trace-agent-tools",
        request_started_at=datetime.now(timezone.utc),
    )


def _descriptor(
    *,
    name: str,
    handler,
    mutation_class: MutationClass = MutationClass.READ_ONLY,
    type_class: TypeClass = TypeClass.FUNCTION,
    required_scopes: tuple[str, ...] = ("tools.read",),
    side_effect_manifest: SideEffectManifest | None = None,
    cache_ttl_seconds: int = 0,
    composite_spec: CompositeToolSpec | None = None,
) -> ToolDescriptor:
    return ToolDescriptor(
        tool_id=name,
        version="1.0.0",
        type_class=type_class,
        input_schema={"type": "object", "additionalProperties": True},
        output_schema={"type": "object", "additionalProperties": True},
        error_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "retryable": {"type": "boolean"},
            },
            "required": ["code", "message", "retryable"],
            "additionalProperties": True,
        },
        auth_contract=AuthContract(required_scopes=required_scopes),
        timeout_class=TimeoutClass.STANDARD,
        idempotency_spec=IdempotencySpec(enabled=mutation_class is not MutationClass.READ_ONLY),
        mutation_class=mutation_class,
        observability_contract=ObservabilityContract(metric_prefix=f"tests.{name}"),
        side_effect_manifest=side_effect_manifest,
        source_endpoint="tests://agent_tools",
        source_type=ToolSourceType.FUNCTION if type_class is TypeClass.FUNCTION else ToolSourceType.COMPOSITE,
        compressed_description=f"Test tool {name}",
        handler=handler,
        cache_ttl_seconds=cache_ttl_seconds,
        status=ToolStatus.ACTIVE,
        composite_spec=composite_spec,
    )


def _request(
    *,
    tool_id: str,
    params: dict[str, object],
    context: ExecutionContext,
    idempotency_key: str | None = None,
    approval_ticket_id=None,
) -> ToolInvocationRequest:
    return ToolInvocationRequest(
        tool_id=tool_id,
        params=params,
        context=context,
        idempotency_key=idempotency_key,
        approval_ticket_id=approval_ticket_id,
    )


def test_admission_rejects_mutating_tool_without_side_effect_manifest() -> None:
    registry = InMemoryToolRegistry()
    descriptor = _descriptor(
        name="bad_write",
        handler=lambda params, context: {"ok": True},
        mutation_class=MutationClass.WRITE_REVERSIBLE,
        required_scopes=("tools.write",),
        side_effect_manifest=None,
    )

    admission = registry.register(descriptor)
    assert admission.admitted is False
    assert "side-effect manifest" in " ".join(admission.violations)


def test_dispatch_caches_read_only_tools_and_replays_idempotent_mutations() -> None:
    read_counter = {"count": 0}
    write_counter = {"count": 0}
    engine = ToolExecutionEngine()

    read_tool = _descriptor(
        name="read_config",
        handler=lambda params, context: {"value": params["key"], "seen": read_counter.__setitem__("count", read_counter["count"] + 1) or read_counter["count"]},
        cache_ttl_seconds=120,
    )
    write_tool = _descriptor(
        name="append_note",
        handler=lambda params, context: {"sequence": write_counter.__setitem__("count", write_counter["count"] + 1) or write_counter["count"]},
        mutation_class=MutationClass.WRITE_REVERSIBLE,
        required_scopes=("tools.write",),
        side_effect_manifest=SideEffectManifest(resources=("notes",), operations=("append",)),
    )

    engine.register(read_tool)
    engine.register(write_tool)
    read_context = _context(scopes=("tools.read", "tools.write"))

    first_read = engine.dispatch(_request(tool_id="read_config", params={"key": "alpha"}, context=read_context))
    second_read = engine.dispatch(_request(tool_id="read_config", params={"key": "alpha"}, context=read_context))

    first_write = engine.dispatch(_request(tool_id="append_note", params={"text": "note"}, context=read_context))
    second_write = engine.dispatch(_request(tool_id="append_note", params={"text": "note"}, context=read_context))

    assert first_read.status.value == "success"
    assert second_read.success.source == "cache"
    assert read_counter["count"] == 1
    assert first_write.status.value == "success"
    assert second_write.status.value == "success"
    assert first_write.success.data == second_write.success.data
    assert write_counter["count"] == 1


def test_irreversible_tool_requires_approval_and_runs_after_ticket_approval() -> None:
    executed = {"count": 0}
    engine = ToolExecutionEngine()
    destroy_tool = _descriptor(
        name="destroy_record",
        handler=lambda params, context: {"deleted": params["record_id"], "count": executed.__setitem__("count", executed["count"] + 1) or executed["count"]},
        mutation_class=MutationClass.WRITE_IRREVERSIBLE,
        required_scopes=("tools.write",),
        side_effect_manifest=SideEffectManifest(resources=("records",), operations=("delete",), reversible=False),
    )
    engine.register(destroy_tool)
    context = _context(scopes=("tools.write",))

    pending = engine.dispatch(_request(tool_id="destroy_record", params={"record_id": "r1"}, context=context))
    assert pending.status.value == "pending"
    ticket_id = pending.pending.approval_ticket_id

    engine.approve_ticket(ticket_id, resolver_id=uuid4())
    approved = engine.dispatch(
        _request(
            tool_id="destroy_record",
            params={"record_id": "r1"},
            context=context,
            idempotency_key="destroy-r1",
            approval_ticket_id=ticket_id,
        )
    )

    assert approved.status.value == "success"
    assert approved.success.data["deleted"] == "r1"
    assert executed["count"] == 1


def test_composite_tool_executes_steps_and_merges_output() -> None:
    engine = ToolExecutionEngine()
    prepare = _descriptor(
        name="prepare_text",
        handler=lambda params, context: {"text": str(params["query"]).upper()},
    )
    decorate = _descriptor(
        name="decorate_text",
        handler=lambda params, context: {"final": f"{params['prefix']}::{params['text']}"},
    )
    composite = _descriptor(
        name="composite_text",
        handler=None,
        type_class=TypeClass.COMPOSITE,
        composite_spec=CompositeToolSpec(
            steps=(
                CompositeStep(
                    step_id="prepare",
                    tool_id="prepare_text",
                    input_bindings={"query": "request.query"},
                ),
                CompositeStep(
                    step_id="decorate",
                    tool_id="decorate_text",
                    depends_on=("prepare",),
                    input_bindings={"prefix": "request.prefix", "text": "steps.prepare.text"},
                ),
            ),
            output_bindings={"result": "steps.decorate.final"},
        ),
    )

    engine.register(prepare)
    engine.register(decorate)
    engine.register(composite)

    response = engine.dispatch(
        _request(
            tool_id="composite_text",
            params={"query": "hello", "prefix": "P"},
            context=_context(scopes=("tools.read",)),
        )
    )

    assert response.status.value == "success"
    assert response.success.data == {"result": "P::HELLO"}
