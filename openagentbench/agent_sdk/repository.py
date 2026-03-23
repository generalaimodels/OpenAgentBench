"""In-memory state repository for SDK invocations and long-running handles."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from .models import AgentSdkInvocationResult, OperationHandle


@dataclass(slots=True)
class InMemoryAgentSdkRepository:
    invocations: list[AgentSdkInvocationResult] = field(default_factory=list)
    handles: dict[UUID, OperationHandle] = field(default_factory=dict)

    def append_invocation(self, result: AgentSdkInvocationResult) -> None:
        self.invocations.append(result)
        if result.operation_handle is not None:
            self.handles[result.operation_handle.handle_id] = result.operation_handle

    def list_invocations(self) -> tuple[AgentSdkInvocationResult, ...]:
        return tuple(self.invocations)

    def put_handle(self, handle: OperationHandle) -> None:
        self.handles[handle.handle_id] = handle

    def get_handle(self, handle_id: UUID) -> OperationHandle | None:
        return self.handles.get(handle_id)


__all__ = ["InMemoryAgentSdkRepository"]
