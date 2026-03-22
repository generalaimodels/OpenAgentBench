"""Shared type aliases for the agent-tools module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping

from openagentbench.agent_data.types import JSONValue

JSONSchema = dict[str, Any]
JSONMapping = Mapping[str, JSONValue]

if TYPE_CHECKING:
    from .models import ExecutionContext

ToolHandler = Callable[[JSONMapping, "ExecutionContext"], JSONValue]

