"""Unified dispatch engine for the in-memory agent-tools reference implementation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from math import ceil
from time import perf_counter_ns
from typing import Any, Mapping
from uuid import UUID, uuid4

from openagentbench.agent_data import canonical_json_bytes

from .enums import (
    ApprovalStatus,
    AuthDecision,
    Environment,
    ErrorCode,
    InvocationStatus,
    MutationClass,
    ToolStatus,
    TypeClass,
)
from .models import (
    ExecutionContext,
    IdempotencyRecord,
    ToolApprovalTicket,
    ToolAuditRecord,
    ToolCacheEntry,
    ToolDescriptor,
    ToolInvocationRequest,
    ToolInvocationResponse,
)
from .registry import InMemoryToolRegistry, validate_against_schema
from .repository import InMemoryToolStateRepository, ToolStateRepository


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _derive_hash(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _nested_lookup(payload: Mapping[str, Any], path: str) -> Any:
    if not path:
        return payload
    current: Any = payload
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise KeyError(path)
        current = current[segment]
    return current


def _serialize_uuid(value: UUID | None) -> str:
    return "" if value is None else str(value)


class _CompositeExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolExecutionEngine:
    registry: InMemoryToolRegistry
    state_repository: ToolStateRepository

    def __init__(
        self,
        *,
        registry: InMemoryToolRegistry | None = None,
        state_repository: ToolStateRepository | None = None,
    ) -> None:
        self.registry = registry or InMemoryToolRegistry()
        self.state_repository = state_repository or InMemoryToolStateRepository()

    def register(self, tool: ToolDescriptor):
        return self.registry.register(tool)

    def approve_ticket(self, ticket_id: UUID, *, resolver_id: UUID) -> ToolApprovalTicket | None:
        ticket = self.state_repository.get_approval_ticket(ticket_id)
        if ticket is None:
            return None
        resolved = ToolApprovalTicket(
            ticket_id=ticket.ticket_id,
            tool_id=ticket.tool_id,
            params_redacted=ticket.params_redacted,
            requested_by=ticket.requested_by,
            agent_id=ticket.agent_id,
            status=ApprovalStatus.APPROVED,
            created_at=ticket.created_at,
            expires_at=ticket.expires_at,
            resolution_by=resolver_id,
            resolution_at=_utc_now(),
            metadata=ticket.metadata,
        )
        self.state_repository.resolve_approval_ticket(ticket_id, resolved)
        return resolved

    def deny_ticket(self, ticket_id: UUID, *, resolver_id: UUID) -> ToolApprovalTicket | None:
        ticket = self.state_repository.get_approval_ticket(ticket_id)
        if ticket is None:
            return None
        resolved = ToolApprovalTicket(
            ticket_id=ticket.ticket_id,
            tool_id=ticket.tool_id,
            params_redacted=ticket.params_redacted,
            requested_by=ticket.requested_by,
            agent_id=ticket.agent_id,
            status=ApprovalStatus.DENIED,
            created_at=ticket.created_at,
            expires_at=ticket.expires_at,
            resolution_by=resolver_id,
            resolution_at=_utc_now(),
            metadata=ticket.metadata,
        )
        self.state_repository.resolve_approval_ticket(ticket_id, resolved)
        return resolved

    def dispatch(self, request: ToolInvocationRequest) -> ToolInvocationResponse:
        started_ns = perf_counter_ns()
        tool = self.registry.resolve(request.tool_id, request.version_spec)
        input_hash = hashlib.sha256(canonical_json_bytes(dict(request.params))).digest()

        if tool is None:
            response = ToolInvocationResponse.from_error(
                ErrorCode.TOOL_NOT_FOUND,
                f"tool '{request.tool_id}' was not found",
                retryable=False,
            )
            self._append_audit(tool=None, request=request, response=response, input_hash=input_hash, started_ns=started_ns)
            return response

        if tool.status is ToolStatus.QUARANTINED:
            response = ToolInvocationResponse.from_error(
                ErrorCode.TOOL_QUARANTINED,
                f"tool '{tool.tool_id}' is quarantined",
                retryable=False,
            )
            self._append_audit(tool=tool, request=request, response=response, input_hash=input_hash, started_ns=started_ns)
            return response

        if tool.status is ToolStatus.UNAVAILABLE:
            response = ToolInvocationResponse.from_error(
                ErrorCode.UPSTREAM_FAILURE,
                f"tool '{tool.tool_id}' is unavailable",
                retryable=True,
                retry_after_ms=5_000,
            )
            self._append_audit(tool=tool, request=request, response=response, input_hash=input_hash, started_ns=started_ns)
            return response

        validation = validate_against_schema(dict(request.params), tool.input_schema)
        if not validation.valid:
            response = ToolInvocationResponse.from_error(
                ErrorCode.INVALID_INPUT,
                "tool input failed schema validation",
                retryable=False,
                details={"errors": list(validation.errors)},
            )
            self._append_audit(tool=tool, request=request, response=response, input_hash=input_hash, started_ns=started_ns)
            return response

        idempotency_key = self._resolve_idempotency_key(tool, request)
        if idempotency_key is not None:
            prior = self.state_repository.get_idempotency_record(tool.tool_id, idempotency_key)
            if prior is not None:
                return prior.response

        auth_decision = self._evaluate_authorization(tool, request)
        if auth_decision is AuthDecision.DENY:
            response = ToolInvocationResponse.from_error(
                ErrorCode.AUTHORIZATION_DENIED,
                f"scopes for '{tool.tool_id}' were insufficient",
                retryable=False,
            )
            self._append_audit(
                tool=tool,
                request=request,
                response=response,
                input_hash=input_hash,
                started_ns=started_ns,
                auth_decision=auth_decision,
            )
            return response

        if auth_decision is AuthDecision.REQUIRES_APPROVAL:
            approved_ticket = self._validate_approved_ticket(tool, request)
            if approved_ticket is None:
                ticket = self._create_approval_ticket(tool, request)
                response = ToolInvocationResponse.from_pending(ticket.ticket_id, poll_token=str(ticket.ticket_id))
                self._append_audit(
                    tool=tool,
                    request=request,
                    response=response,
                    input_hash=input_hash,
                    started_ns=started_ns,
                    auth_decision=auth_decision,
                )
                return response

        cache_key = self._cache_key(tool, request)
        if tool.mutation_class is MutationClass.READ_ONLY and tool.cache_ttl_seconds > 0:
            cached = self.state_repository.get_cache_entry(cache_key)
            if cached is not None:
                response = ToolInvocationResponse.from_success(
                    cached.result_data,
                    provenance={"tool_id": tool.tool_id, "version": tool.version, "source": "cache"},
                    execution_metadata={"latency_ms": 0, "cache_hit": True},
                    source="cache",
                )
                self._append_audit(tool=tool, request=request, response=response, input_hash=input_hash, started_ns=started_ns)
                return response

        response = self._execute_tool(tool, request, started_ns)

        if response.status is InvocationStatus.SUCCESS and response.success is not None:
            output_validation = validate_against_schema(response.success.data, tool.output_schema)
            if not output_validation.valid:
                response = ToolInvocationResponse.from_error(
                    ErrorCode.OUTPUT_SCHEMA_VIOLATION,
                    "tool output failed schema validation",
                    retryable=False,
                    details={"errors": list(output_validation.errors)},
                )

        if response.status is InvocationStatus.SUCCESS and idempotency_key is not None:
            self.state_repository.put_idempotency_record(
                IdempotencyRecord(
                    key=idempotency_key,
                    tool_id=tool.tool_id,
                    user_id=request.context.user_id,
                    session_id=request.context.session_id,
                    response=response,
                    expires_at=request.context.request_started_at
                    + timedelta(seconds=tool.idempotency_spec.dedup_window_seconds),
                )
            )

        if response.status is InvocationStatus.SUCCESS and tool.mutation_class is MutationClass.READ_ONLY and tool.cache_ttl_seconds > 0:
            self.state_repository.put_cache_entry(
                ToolCacheEntry(
                    cache_key=cache_key,
                    tool_id=tool.tool_id,
                    user_id=request.context.user_id,
                    result_data=response.success.data if response.success is not None else {},
                    expires_at=request.context.request_started_at + timedelta(seconds=tool.cache_ttl_seconds),
                )
            )

        if response.status is InvocationStatus.SUCCESS and tool.mutation_class is not MutationClass.READ_ONLY:
            self.state_repository.invalidate_cache(user_id=request.context.user_id)

        self._append_audit(tool=tool, request=request, response=response, input_hash=input_hash, started_ns=started_ns)
        return response

    def refresh_tool_health(self, tool_id: str | None = None) -> dict[str, float]:
        targets = [self.registry.resolve(tool_id)] if tool_id is not None else list(self.registry.list_tools(include_inactive=True))
        updated: dict[str, float] = {}
        for tool in targets:
            if tool is None:
                continue
            audits = self.state_repository.list_audit_records(tool.tool_id)
            if not audits:
                updated[tool.tool_id] = tool.health_score
                continue
            total = len(audits)
            error_count = sum(1 for audit in audits if audit.status is InvocationStatus.ERROR)
            timeout_count = sum(1 for audit in audits if audit.status is InvocationStatus.TIMEOUT)
            latencies = sorted(audit.latency_ms for audit in audits)
            p99_index = min(len(latencies) - 1, ceil(len(latencies) * 0.99) - 1)
            p99_latency = latencies[p99_index]
            error_rate = error_count / total
            timeout_rate = timeout_count / total
            latency_ratio = p99_latency / max(tool.timeout_class.max_duration_ms(), 1)
            score = 1.0 - _clamp(0.5 * (error_rate / 0.20) + 0.3 * latency_ratio + 0.2 * (timeout_rate / 0.10), 0.0, 1.0)
            tool.health_score = round(score, 4)
            if tool.health_score < 0.3:
                tool.status = ToolStatus.QUARANTINED
            updated[tool.tool_id] = tool.health_score
        return updated

    def _resolve_idempotency_key(self, tool: ToolDescriptor, request: ToolInvocationRequest) -> str | None:
        if tool.mutation_class is MutationClass.READ_ONLY:
            return None
        if request.idempotency_key:
            return request.idempotency_key
        if not tool.idempotency_spec.enabled:
            return None
        return _derive_hash(
            [
                _serialize_uuid(request.context.user_id).encode("utf-8"),
                _serialize_uuid(request.context.session_id).encode("utf-8"),
                tool.tool_id.encode("utf-8"),
                canonical_json_bytes(dict(request.params)),
            ]
        )

    def _cache_key(self, tool: ToolDescriptor, request: ToolInvocationRequest) -> str:
        return _derive_hash(
            [
                _serialize_uuid(request.context.user_id).encode("utf-8"),
                tool.tool_id.encode("utf-8"),
                tool.version.encode("utf-8"),
                canonical_json_bytes(dict(request.params)),
            ]
        )

    def _evaluate_authorization(self, tool: ToolDescriptor, request: ToolInvocationRequest) -> AuthDecision:
        caller_scopes = set(request.context.scopes)
        required = set(tool.auth_contract.required_scopes)
        if not required.issubset(caller_scopes):
            return AuthDecision.DENY
        if tool.auth_contract.approval_required:
            return AuthDecision.REQUIRES_APPROVAL
        if tool.mutation_class is MutationClass.WRITE_IRREVERSIBLE:
            return AuthDecision.REQUIRES_APPROVAL
        if (
            request.context.environment is Environment.PRODUCTION
            and tool.mutation_class is not MutationClass.READ_ONLY
            and request.context.trust_level < 80
        ):
            return AuthDecision.REQUIRES_APPROVAL
        return AuthDecision.ALLOW

    def _validate_approved_ticket(self, tool: ToolDescriptor, request: ToolInvocationRequest) -> ToolApprovalTicket | None:
        if request.approval_ticket_id is None:
            return None
        ticket = self.state_repository.get_approval_ticket(request.approval_ticket_id)
        if ticket is None:
            return None
        if ticket.tool_id != tool.tool_id or ticket.requested_by != request.context.user_id:
            return None
        if ticket.status is not ApprovalStatus.APPROVED:
            return None
        return ticket

    def _create_approval_ticket(self, tool: ToolDescriptor, request: ToolInvocationRequest) -> ToolApprovalTicket:
        redacted = {
            key: ("<redacted>" if key in tool.auth_contract.redacted_fields else value)
            for key, value in request.params.items()
        }
        ticket = ToolApprovalTicket(
            ticket_id=uuid4(),
            tool_id=tool.tool_id,
            params_redacted=redacted,
            requested_by=request.context.user_id,
            agent_id=request.context.agent_id,
            status=ApprovalStatus.PENDING,
            created_at=request.context.request_started_at,
            expires_at=request.context.request_started_at + timedelta(minutes=10),
            metadata={"trace_id": request.context.trace_id},
        )
        self.state_repository.create_approval_ticket(ticket)
        return ticket

    def _execute_tool(
        self,
        tool: ToolDescriptor,
        request: ToolInvocationRequest,
        started_ns: int,
    ) -> ToolInvocationResponse:
        effective_deadline_ms = tool.timeout_class.max_duration_ms()
        if request.requested_deadline_ms is not None:
            effective_deadline_ms = min(effective_deadline_ms, request.requested_deadline_ms)

        try:
            if tool.type_class is TypeClass.COMPOSITE:
                result = self._execute_composite(tool, request, effective_deadline_ms=effective_deadline_ms)
            else:
                if tool.handler is None:
                    return ToolInvocationResponse.from_error(
                        ErrorCode.UPSTREAM_FAILURE,
                        f"tool '{tool.tool_id}' has no executable handler",
                        retryable=True,
                    )
                result = tool.handler(dict(request.params), request.context)
        except _CompositeExecutionError as exc:
            return ToolInvocationResponse.from_error(
                ErrorCode.COMPOSITE_PARTIAL_FAILURE,
                str(exc),
                retryable=False,
            )
        except Exception as exc:  # pragma: no cover - safety net
            return ToolInvocationResponse.from_error(
                ErrorCode.INTERNAL_TOOL_ERROR,
                f"tool '{tool.tool_id}' raised {exc.__class__.__name__}",
                retryable=False,
                details={"message": str(exc)},
            )

        elapsed_ms = (perf_counter_ns() - started_ns) // 1_000_000
        if elapsed_ms > effective_deadline_ms:
            return ToolInvocationResponse.from_error(
                ErrorCode.DEADLINE_EXCEEDED,
                f"tool '{tool.tool_id}' exceeded its deadline",
                retryable=True,
                timeout=True,
            )

        if isinstance(result, ToolInvocationResponse):
            return result
        return ToolInvocationResponse.from_success(
            result,
            provenance={
                "tool_id": tool.tool_id,
                "version": tool.version,
                "type_class": tool.type_class.value,
                "executed_at": request.context.request_started_at.isoformat(),
            },
            execution_metadata={"latency_ms": elapsed_ms, "cache_hit": False},
        )

    def _execute_composite(
        self,
        tool: ToolDescriptor,
        request: ToolInvocationRequest,
        *,
        effective_deadline_ms: int,
    ) -> Mapping[str, Any]:
        if tool.composite_spec is None:
            raise ValueError("composite tools require a composite spec")

        pending = {step.step_id: step for step in tool.composite_spec.steps}
        results: dict[str, Any] = {}
        successful_steps: list[tuple[Any, Mapping[str, Any]]] = []

        while pending:
            progressed = False
            for step_id, step in list(pending.items()):
                if any(dependency not in results for dependency in step.depends_on):
                    continue
                step_params: dict[str, Any] = dict(step.static_params)
                for key, binding in step.input_bindings.items():
                    step_params[key] = self._resolve_binding(binding, request.params, results)

                step_request = ToolInvocationRequest(
                    tool_id=step.tool_id,
                    params=step_params,
                    context=request.context,
                    idempotency_key=_derive_hash(
                        [
                            tool.tool_id.encode("utf-8"),
                            step.step_id.encode("utf-8"),
                            canonical_json_bytes(step_params),
                        ]
                    ),
                    requested_deadline_ms=effective_deadline_ms,
                    approval_ticket_id=request.approval_ticket_id,
                )
                step_response = self.dispatch(step_request)
                if step_response.status is not InvocationStatus.SUCCESS or step_response.success is None:
                    self._compensate(successful_steps, request)
                    raise _CompositeExecutionError(f"composite step '{step.step_id}' failed")
                results[step.step_id] = step_response.success.data
                successful_steps.append((step, step_params))
                pending.pop(step_id)
                progressed = True
            if not progressed:
                raise _CompositeExecutionError("composite graph contains a cycle or unresolved dependency")

        if not tool.composite_spec.output_bindings:
            return results
        merged: dict[str, Any] = {}
        for key, binding in tool.composite_spec.output_bindings.items():
            merged[key] = self._resolve_binding(binding, request.params, results)
        return merged

    def _resolve_binding(
        self,
        binding: str,
        request_params: Mapping[str, Any],
        step_results: Mapping[str, Any],
    ) -> Any:
        if binding == "request":
            return dict(request_params)
        if binding.startswith("request."):
            return _nested_lookup(request_params, binding.removeprefix("request."))
        if binding.startswith("steps."):
            path = binding.removeprefix("steps.")
            first, _, remainder = path.partition(".")
            if first not in step_results:
                raise KeyError(binding)
            value = step_results[first]
            if not remainder:
                return value
            if not isinstance(value, Mapping):
                raise KeyError(binding)
            return _nested_lookup(value, remainder)
        return binding

    def _compensate(self, successful_steps: list[tuple[Any, Mapping[str, Any]]], request: ToolInvocationRequest) -> None:
        for step, _ in reversed(successful_steps):
            if step.compensation_tool_id is None:
                continue
            compensation_params = dict(step.compensation_params)
            compensation_params.setdefault("composite_input", dict(request.params))
            compensation_params.setdefault("failed_step", step.step_id)
            self.dispatch(
                ToolInvocationRequest(
                    tool_id=step.compensation_tool_id,
                    params=compensation_params,
                    context=request.context,
                    requested_deadline_ms=1_000,
                )
            )

    def _append_audit(
        self,
        *,
        tool: ToolDescriptor | None,
        request: ToolInvocationRequest,
        response: ToolInvocationResponse,
        input_hash: bytes,
        started_ns: int,
        auth_decision: AuthDecision = AuthDecision.ALLOW,
    ) -> None:
        tool_id = request.tool_id if tool is None else tool.tool_id
        tool_version = "" if tool is None else tool.version
        mutation = MutationClass.READ_ONLY if tool is None else tool.mutation_class
        elapsed_ms = (perf_counter_ns() - started_ns) // 1_000_000
        self.state_repository.append_audit_record(
            ToolAuditRecord(
                audit_id=uuid4(),
                trace_id=request.context.trace_id,
                tool_id=tool_id,
                tool_version=tool_version,
                caller_id=request.context.user_id,
                agent_id=request.context.agent_id,
                session_id=request.context.session_id,
                status=response.status,
                auth_decision=auth_decision,
                input_hash=input_hash,
                latency_ms=elapsed_ms,
                mutation_class=mutation,
                error_code=response.error.code.value if response.error is not None else None,
                token_cost=tool.token_cost_estimate if tool is not None else 0,
                side_effects=asdict(tool.side_effect_manifest) if tool is not None and tool.side_effect_manifest else {},
            )
        )


__all__ = ["ToolExecutionEngine"]
