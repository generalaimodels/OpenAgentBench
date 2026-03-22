"""Shared type aliases used across the agent-data module."""

from __future__ import annotations

from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
EmbeddingVector: TypeAlias = tuple[float, ...]
ContentPart: TypeAlias = dict[str, JSONValue]
MessageContent: TypeAlias = str | tuple[ContentPart, ...] | None
