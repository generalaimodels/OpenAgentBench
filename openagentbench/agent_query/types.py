"""Shared type aliases for the query module."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | dict[str, "JSONValue"] | list["JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
JSONSchema: TypeAlias = dict[str, Any]
StringMap: TypeAlias = Mapping[str, str]
StringSequence: TypeAlias = Sequence[str]

__all__ = [
    "JSONObject",
    "JSONScalar",
    "JSONSchema",
    "JSONValue",
    "StringMap",
    "StringSequence",
]
