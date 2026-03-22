"""Shared type aliases used across the agent-memory module."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from openagentbench.agent_data.types import ContentPart, EmbeddingVector, JSONValue

TokenCounter: TypeAlias = Callable[[str], int]
MessagePayload: TypeAlias = dict[str, JSONValue]

__all__ = ["ContentPart", "EmbeddingVector", "JSONValue", "MessagePayload", "TokenCounter"]
