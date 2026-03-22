"""Registry, admission, and schema-validation primitives for tools."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from openagentbench.agent_data import dumps
from openagentbench.agent_retrieval.scoring import count_tokens, lexical_overlap_score

from .enums import MutationClass, ToolStatus
from .models import AdmissionResult, SchemaValidationResult, ToolDescriptor, ToolSchemaSummary
from .types import JSONSchema, JSONValue

SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.\-]+)?$")


def estimate_schema_tokens(schema: JSONSchema) -> int:
    return max(1, count_tokens(dumps(schema)))


def _matches_type(value: JSONValue, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _validate_schema_value(
    value: JSONValue,
    schema: JSONSchema,
    *,
    path: str,
    errors: list[str],
) -> None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _matches_type(value, expected_type):
        errors.append(f"{path}: expected {expected_type}")
        return

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: value not in enum")

    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            errors.append(f"{path}: shorter than minLength={minimum}")
        if isinstance(maximum, int) and len(value) > maximum:
            errors.append(f"{path}: longer than maxLength={maximum}")

    if (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: smaller than minimum={minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: larger than maximum={maximum}")

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", ())
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for field_name in required:
                if isinstance(field_name, str) and field_name not in value:
                    errors.append(f"{path}.{field_name}: required property missing")
        if schema.get("additionalProperties") is False and isinstance(properties, Mapping):
            for field_name in value:
                if field_name not in properties:
                    errors.append(f"{path}.{field_name}: additional property not allowed")
        if isinstance(properties, Mapping):
            for field_name, field_schema in properties.items():
                if field_name not in value or not isinstance(field_schema, Mapping):
                    continue
                _validate_schema_value(value[field_name], dict(field_schema), path=f"{path}.{field_name}", errors=errors)

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_value(item, dict(item_schema), path=f"{path}[{index}]", errors=errors)


def validate_against_schema(value: JSONValue, schema: JSONSchema) -> SchemaValidationResult:
    if not isinstance(schema, Mapping):
        return SchemaValidationResult(valid=False, errors=("schema must be a mapping",))
    errors: list[str] = []
    _validate_schema_value(value, dict(schema), path="$", errors=errors)
    return SchemaValidationResult(valid=not errors, errors=tuple(errors))


def _error_envelope_conformant(schema: JSONSchema) -> bool:
    if not isinstance(schema, Mapping):
        return False
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return False
    return {"code", "message", "retryable"}.issubset(properties)


def admit_tool(tool: ToolDescriptor) -> AdmissionResult:
    violations: list[str] = []

    for name, schema in (
        ("input schema", tool.input_schema),
        ("output schema", tool.output_schema),
        ("error schema", tool.error_schema),
    ):
        if not isinstance(schema, Mapping):
            violations.append(f"{name} must be a mapping")

    if not _error_envelope_conformant(tool.error_schema):
        violations.append("error schema does not conform to the error envelope")
    if not SEMVER_PATTERN.match(tool.version):
        violations.append("tool version must be semver-compatible")
    if not tool.auth_contract.required_scopes:
        violations.append("auth contract must declare at least one scope")
    if tool.mutation_class is not MutationClass.READ_ONLY and not tool.idempotency_spec.enabled:
        violations.append("mutating tools must enable idempotency")
    if tool.mutation_class is not MutationClass.READ_ONLY and tool.side_effect_manifest is None:
        violations.append("mutating tools must declare a side-effect manifest")
    if not tool.observability_contract.metric_prefix.strip():
        violations.append("observability metric prefix must be non-empty")
    if tool.type_class.value != "composite" and tool.handler is None:
        violations.append("non-composite tools must provide a handler")
    if tool.type_class.value == "composite" and tool.composite_spec is None:
        violations.append("composite tools must provide a composite spec")

    return AdmissionResult(admitted=not violations, violations=tuple(violations))


def _parse_version(version: str) -> tuple[int, int, int, str]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", version)
    if match is None:
        return (0, 0, 0, version)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4))


def _mutation_alignment(task_objective: str, tool: ToolDescriptor) -> float:
    lowered = task_objective.lower()
    write_hints = ("write", "create", "update", "delete", "save", "store", "approve", "send", "execute")
    needs_mutation = any(hint in lowered for hint in write_hints)
    if needs_mutation and tool.mutation_class is not MutationClass.READ_ONLY:
        return 0.20
    if not needs_mutation and tool.mutation_class is MutationClass.READ_ONLY:
        return 0.15
    return 0.0


@dataclass(slots=True)
class InMemoryToolRegistry:
    tools: dict[tuple[str, str], ToolDescriptor] = field(default_factory=dict)

    def register(self, tool: ToolDescriptor) -> AdmissionResult:
        admission = admit_tool(tool)
        if not admission.admitted:
            return admission

        key = (tool.tool_id, tool.version)
        existing = self.tools.get(key)
        if existing is not None and existing.contract_tests_hash == tool.contract_tests_hash:
            return AdmissionResult(admitted=True, already_registered=True)

        if tool.token_cost_estimate <= 0:
            tool.token_cost_estimate = estimate_schema_tokens(tool.input_schema)
        self.tools[key] = tool
        return AdmissionResult(admitted=True)

    def resolve(self, tool_id: str, version_spec: str | None = None) -> ToolDescriptor | None:
        if version_spec is not None:
            return self.tools.get((tool_id, version_spec))

        candidates = [tool for (candidate_id, _), tool in self.tools.items() if candidate_id == tool_id]
        if not candidates:
            return None
        candidates.sort(key=lambda tool: _parse_version(tool.version), reverse=True)
        return candidates[0]

    def list_tools(self, *, include_inactive: bool = False) -> tuple[ToolDescriptor, ...]:
        tools = list(self.tools.values())
        if not include_inactive:
            tools = [
                tool
                for tool in tools
                if tool.status in {ToolStatus.ACTIVE, ToolStatus.DEPRECATED}
            ]
        tools.sort(key=lambda tool: (tool.tool_id, _parse_version(tool.version)))
        return tuple(tools)

    def compressed_index(self) -> tuple[ToolSchemaSummary, ...]:
        return tuple(tool.as_summary() for tool in self.list_tools())

    def select_tools_for_task(
        self,
        *,
        task_objective: str,
        token_budget: int,
        minimum_utility: float = 0.05,
    ) -> tuple[ToolDescriptor, ...]:
        ranked: list[tuple[float, ToolDescriptor]] = []
        for tool in self.list_tools():
            base_text = f"{tool.tool_id} {tool.compressed_description}"
            utility = 0.65 * lexical_overlap_score(task_objective, base_text)
            utility += 0.20 * tool.health_score
            utility += _mutation_alignment(task_objective, tool)
            if utility < minimum_utility:
                continue
            cost = max(tool.token_cost_estimate, 1)
            ranked.append((utility / cost, tool))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected: list[ToolDescriptor] = []
        remaining = token_budget
        for _, tool in ranked:
            if tool.token_cost_estimate > remaining:
                continue
            selected.append(tool)
            remaining -= tool.token_cost_estimate
        return tuple(selected)


__all__ = [
    "InMemoryToolRegistry",
    "admit_tool",
    "estimate_schema_tokens",
    "validate_against_schema",
]

