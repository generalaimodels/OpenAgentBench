"""Shared type aliases used across the agent-retrieval module."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
EmbeddingVector: TypeAlias = tuple[float, ...]
TokenCounter: TypeAlias = Callable[[str], int]
