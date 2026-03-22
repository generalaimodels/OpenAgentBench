"""Custom loop engine and terminal tool for the interactive demo."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from openagentbench.agent_loop import AgentLoopEngine
from openagentbench.agent_tools import (
    AuthContract,
    IdempotencySpec,
    MutationClass,
    ObservabilityContract,
    SideEffectManifest,
    TimeoutClass,
    ToolDescriptor,
    ToolExecutionEngine,
    ToolSourceType,
    TypeClass,
)

from .demo_env import DemoConfig, load_demo_config


def build_terminal_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": (
                "Execute a local terminal command inside the demo workspace and return "
                "stdout, stderr, and the exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1},
                    "cwd": {"type": "string", "minLength": 1},
                    "timeout_ms": {"type": "integer", "minimum": 250, "maximum": 30000},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }


def _error_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "message": {"type": "string"},
            "retryable": {"type": "boolean"},
        },
        "required": ["code", "message", "retryable"],
        "additionalProperties": True,
    }


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "exit_code": {"type": "integer"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "duration_ms": {"type": "integer"},
            "timed_out": {"type": "boolean"},
            "write_enabled": {"type": "boolean"},
        },
        "required": ["command", "cwd", "exit_code", "stdout", "stderr", "duration_ms", "timed_out", "write_enabled"],
        "additionalProperties": False,
    }


def _blocked_command(command: str, *, allow_terminal_write: bool) -> str | None:
    normalized = f" {command.lower()} "
    if any(token in normalized for token in ("&&", "||", ";", "|")):
        return "command chaining is disabled in the demo terminal tool"
    blocked = (
        " rm ",
        " rmdir ",
        " del ",
        " move ",
        " mv ",
        " cp ",
        " chmod ",
        " chown ",
        " git reset ",
        " git checkout ",
        " shutdown ",
        " reboot ",
    )
    if any(token in normalized for token in blocked):
        return "destructive or state-resetting commands are blocked in the demo terminal tool"
    if not allow_terminal_write and any(token in normalized for token in (" mkdir ", " touch ", " tee ", " sed -i ", " > ", " >> ")):
        return "write commands are disabled; set OPENAGENTBENCH_DEMO_ALLOW_TERMINAL_WRITE=1 to opt in"
    return None


def _resolve_cwd(workspace_root: Path, requested_cwd: str | None) -> Path:
    if not requested_cwd:
        return workspace_root
    candidate = (workspace_root / requested_cwd).resolve()
    candidate.relative_to(workspace_root)
    return candidate


def _infer_demo_terminal_command(instruction: str) -> str | None:
    lowered = instruction.strip().lower()
    explicit_markers = ("python ", "python3 ", "pytest", "unittest", "bash ", "sh ", "./")
    if any(marker in lowered for marker in explicit_markers):
        return None
    if any(token in lowered for token in ("unit test", "unit tests", "test suite", "run tests", "workspace tests")):
        return "python3 -m unittest discover -q"
    return None


def build_terminal_tool_descriptor(config: DemoConfig) -> ToolDescriptor:
    definition = build_terminal_tool_definition()
    allow_terminal_write = config.allow_terminal_write

    def handler(params: dict[str, Any], _) -> dict[str, Any]:
        command = str(params["command"]).strip()
        if not command:
            raise ValueError("command must not be empty")
        blocked_reason = _blocked_command(command, allow_terminal_write=allow_terminal_write)
        if blocked_reason is not None:
            raise ValueError(blocked_reason)
        requested_cwd = str(params.get("cwd") or "").strip() or None
        cwd = _resolve_cwd(config.workspace_root, requested_cwd)
        timeout_ms = int(params.get("timeout_ms") or config.terminal_timeout_ms)
        args = shlex.split(command)
        started_ns = perf_counter_ns()
        completed = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_ms / 1000.0,
            check=False,
        )
        duration_ms = max(int((perf_counter_ns() - started_ns) / 1_000_000), 0)
        stdout = completed.stdout[-config.max_output_chars :]
        stderr = completed.stderr[-config.max_output_chars :]
        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration_ms": duration_ms,
            "timed_out": False,
            "write_enabled": allow_terminal_write,
        }

    mutation_class = MutationClass.WRITE_REVERSIBLE if allow_terminal_write else MutationClass.READ_ONLY
    side_effect_manifest = None
    if allow_terminal_write:
        side_effect_manifest = SideEffectManifest(
            resources=("examples.interactive_loop_demo.workspace",),
            operations=("workspace_write",),
            reversible=True,
        )
    return ToolDescriptor(
        tool_id="terminal_execute",
        version="1.0.0",
        type_class=TypeClass.FUNCTION,
        input_schema=definition["function"]["parameters"],
        output_schema=_output_schema(),
        error_schema=_error_schema(),
        auth_contract=AuthContract(required_scopes=("tools.terminal",)),
        timeout_class=TimeoutClass.STANDARD,
        idempotency_spec=IdempotencySpec(enabled=allow_terminal_write),
        mutation_class=mutation_class,
        observability_contract=ObservabilityContract(metric_prefix="agent_tools.terminal_execute"),
        source_endpoint="openagentbench://examples/interactive_loop_demo",
        source_type=ToolSourceType.FUNCTION,
        compressed_description=definition["function"]["description"],
        handler=handler,
        side_effect_manifest=side_effect_manifest,
        token_cost_estimate=42,
    )


@dataclass(slots=True)
class InteractiveDemoLoopEngine(AgentLoopEngine):
    demo_config: DemoConfig = field(default_factory=load_demo_config)

    def _extra_query_tool_definitions(self) -> tuple[dict[str, Any], ...]:
        return (build_terminal_tool_definition(),)

    def _register_custom_tools(self, engine: ToolExecutionEngine) -> None:
        admission = engine.register(build_terminal_tool_descriptor(self.demo_config))
        if not admission.admitted:
            raise RuntimeError(f"failed to register terminal_execute: {admission.violations}")

    def _build_tool_params(self, state, tool_id: str, action) -> dict[str, Any]:
        params = AgentLoopEngine._build_tool_params(self, state, tool_id, action)
        if tool_id != "terminal_execute":
            return params
        inferred_command = _infer_demo_terminal_command(action.instruction)
        if inferred_command is not None:
            params["command"] = inferred_command
        return params


__all__ = [
    "InteractiveDemoLoopEngine",
    "build_terminal_tool_definition",
    "build_terminal_tool_descriptor",
]
