"""Example-local tools and loop extensions for the realtime Q&A chatbot."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping

from openagentbench.agent_loop import AgentLoopEngine
from openagentbench.agent_tools import (
    AuthContract,
    ErrorCode,
    IdempotencySpec,
    MutationClass,
    ObservabilityContract,
    SideEffectManifest,
    TimeoutClass,
    ToolDescriptor,
    ToolInvocationResponse,
    ToolSourceType,
    ToolStatus,
    TypeClass,
)

_READ_ONLY_BLOCK_PATTERNS = (
    r"\brm\b",
    r"\brmdir\b",
    r"\bdel\b",
    r"\berase\b",
    r"\bremove-item\b",
    r"\bmove-item\b",
    r"\bcopy-item\b",
    r"\bnew-item\b",
    r"\bset-content\b",
    r"\badd-content\b",
    r"\bout-file\b",
    r"\btruncate\b",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bgit\s+push\b",
    r"\bgit\s+commit\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+checkout\s+--\b",
    r"\bsed\s+-i\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"(?:^|[^<])>(?!>)",
    r">>",
)

_EXPLICIT_COMMAND_PREFIXES = (
    "run the command ",
    "execute the command ",
    "run command ",
    "execute command ",
    "run ",
    "execute ",
)

_EXPLICIT_COMMAND_SUFFIXES = (
    " in the terminal and summarize the result",
    " in the terminal and explain the result",
    " in terminal and summarize the result",
    " in terminal and explain the result",
    " in the terminal",
    " in terminal",
    " and summarize the result",
    " and explain the result",
    " and report the result",
)

_COMMAND_PREFIXES = (
    "python",
    "python3",
    "py",
    "pip",
    "pytest",
    "git",
    "dir",
    "ls",
    "pwd",
    "cd",
    "echo",
    "type",
    "cat",
    "findstr",
    "rg",
    "where",
    "which",
    "npm",
    "node",
    "cargo",
    "go",
    "docker",
    "kubectl",
    "az",
    "aws",
    "pwsh",
    "powershell",
    "cmd",
    "get-childitem",
    "get-command",
    "get-help",
    "help",
)

_NATURAL_LANGUAGE_MARKERS = (
    "help me",
    "solve",
    "fix",
    "check",
    "why",
    "not working",
    "working properly",
    "on your own",
    "for me",
    "again",
    "think",
    "issue",
    "problem",
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "by",
        "check",
        "fix",
        "for",
        "from",
        "help",
        "how",
        "i",
        "is",
        "it",
        "me",
        "my",
        "not",
        "of",
        "on",
        "or",
        "own",
        "please",
        "properly",
        "shell",
        "solve",
        "terminal",
        "the",
        "this",
        "to",
        "we",
        "what",
        "why",
        "working",
        "your",
    }
)


def build_terminal_execute_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": (
                "Execute a local terminal command inside the project workspace for inspection, diagnostics, "
                "tests, repository analysis, and environment verification. Read-only by default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1},
                    "working_dir": {"type": "string", "minLength": 1},
                    "shell": {
                        "type": "string",
                        "enum": ["auto", "bash", "sh", "powershell", "pwsh", "cmd"],
                    },
                    "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 120000},
                    "max_output_chars": {"type": "integer", "minimum": 256, "maximum": 20000},
                    "allow_write": {"type": "boolean"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "shell": {"type": "string"},
            "working_dir": {"type": "string"},
            "exit_code": {"type": "integer"},
            "duration_ms": {"type": "integer"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "timed_out": {"type": "boolean"},
            "summary": {"type": "string"},
        },
        "required": [
            "command",
            "shell",
            "working_dir",
            "exit_code",
            "duration_ms",
            "stdout",
            "stderr",
            "timed_out",
            "summary",
        ],
        "additionalProperties": True,
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


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 16, 0)] + "\n...[truncated]"


def _resolve_working_dir(project_root: Path, working_dir: str | None) -> Path:
    root = project_root.resolve()
    candidate = root if not working_dir else Path(working_dir).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if os.path.commonpath([str(root), str(resolved)]) != str(root):
        raise ValueError("working_dir must stay inside the project workspace")
    if not resolved.exists():
        raise ValueError("working_dir does not exist")
    if not resolved.is_dir():
        raise ValueError("working_dir must be a directory")
    return resolved


def _resolve_shell(shell_name: str) -> tuple[str, list[str]]:
    normalized = shell_name.strip().lower() or "auto"
    if normalized == "auto":
        normalized = "powershell" if os.name == "nt" else "bash"
    if normalized == "bash":
        return "bash", ["bash", "-lc"]
    if normalized == "sh":
        return "sh", ["sh", "-lc"]
    if normalized == "powershell":
        return "powershell", ["powershell", "-NoProfile", "-Command"]
    if normalized == "pwsh":
        return "pwsh", ["pwsh", "-NoProfile", "-Command"]
    if normalized == "cmd":
        return "cmd", ["cmd", "/d", "/s", "/c"]
    raise ValueError(f"unsupported shell '{shell_name}'")


def _is_write_like(command: str) -> bool:
    lowered = command.strip().lower()
    return any(re.search(pattern, lowered) for pattern in _READ_ONLY_BLOCK_PATTERNS)


def _extract_explicit_command(text: str) -> str | None:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in _EXPLICIT_COMMAND_PREFIXES:
        if lowered.startswith(prefix):
            candidate = stripped[len(prefix) :].strip().strip(".")
            lowered_candidate = candidate.lower()
            for suffix in _EXPLICIT_COMMAND_SUFFIXES:
                if lowered_candidate.endswith(suffix):
                    candidate = candidate[: -len(suffix)].strip().strip(".")
                    lowered_candidate = candidate.lower()
            return candidate
    fenced = re.search(r"`([^`]+)`", stripped)
    if fenced is not None:
        return fenced.group(1).strip()
    return None


def _word_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text))


def _looks_like_natural_language_request(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if _extract_explicit_command(stripped) is not None:
        return False
    if stripped.endswith("?"):
        return True
    if any(marker in lowered for marker in _NATURAL_LANGUAGE_MARKERS):
        return True
    words = _word_tokens(stripped)
    if not words:
        return False
    if words[0].lower() == "help" and len(words) >= 4:
        return True
    stopword_count = sum(1 for word in words if word.lower() in _STOPWORDS)
    if len(words) >= 6 and stopword_count >= 2:
        return True
    if lowered.startswith(("can you ", "please ", "help ", "why ", "what ", "how ")):
        return True
    return False


def _looks_like_command(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _extract_explicit_command(stripped) is not None:
        return True
    if _looks_like_natural_language_request(stripped):
        return False
    lowered = stripped.lower()
    first = lowered.split(maxsplit=1)[0]
    if first in _COMMAND_PREFIXES:
        return True
    if stripped.startswith(("./", ".\\", "/", "~")):
        return True
    if any(token in stripped for token in ("|", ">", "<", ";", "&&")):
        return True
    return len(_word_tokens(stripped)) <= 3


def _default_shell_name() -> str:
    return "powershell" if os.name == "nt" else "bash"


def _powershell_diagnostic_command() -> str:
    return (
        "$ErrorActionPreference='Continue'; "
        "Write-Output 'DIAG: shell=powershell'; "
        "Write-Output ('DIAG: ps_version=' + $PSVersionTable.PSVersion.ToString()); "
        "$helpCommand = Get-Command help -ErrorAction SilentlyContinue; "
        "if ($null -ne $helpCommand) { "
        "Write-Output ('DIAG: help_command=' + $helpCommand.CommandType + ':' + $helpCommand.Name) "
        "} else { "
        "Write-Output 'DIAG: help_command=missing' "
        "}; "
        "$helpAlias = Get-Alias help -ErrorAction SilentlyContinue; "
        "if ($null -ne $helpAlias) { "
        "Write-Output ('DIAG: help_alias=' + $helpAlias.Definition) "
        "} else { "
        "Write-Output 'DIAG: help_alias=missing' "
        "}; "
        "$sampleHelp = Get-Help Get-ChildItem -ErrorAction SilentlyContinue; "
        "if ($null -ne $sampleHelp) { "
        "Write-Output ('DIAG: sample_help_name=' + $sampleHelp.Name) "
        "} else { "
        "Write-Output 'DIAG: sample_help_name=missing' "
        "}; "
        "Write-Output ('DIAG: location=' + (Get-Location).Path)"
    )


def _bash_diagnostic_command() -> str:
    return (
        "printf 'DIAG: shell=bash\\n'; "
        "printf 'DIAG: cwd=%s\\n' \"$(pwd)\"; "
        "printf 'DIAG: bash_path=%s\\n' \"$(command -v bash 2>/dev/null || echo missing)\"; "
        "printf 'DIAG: sh_path=%s\\n' \"$(command -v sh 2>/dev/null || echo missing)\"; "
        "printf 'DIAG: python_path=%s\\n' \"$(command -v python 2>/dev/null || command -v python3 2>/dev/null || echo missing)\""
    )


def _bash_powershell_probe_command() -> str:
    script = shlex.quote(_powershell_diagnostic_command())
    return (
        "if pwsh_path=$(command -v pwsh); then "
        f"\"${{pwsh_path}}\" -NoProfile -Command {script}; "
        "elif powershell_path=$(command -v powershell); then "
        f"\"${{powershell_path}}\" -NoProfile -Command {script}; "
        "else "
        "printf 'DIAG: shell=powershell\\nDIAG: powershell_runtime=missing\\n'; "
        "fi"
    )


def infer_terminal_request(text: str) -> dict[str, Any]:
    stripped = text.strip()
    explicit = _extract_explicit_command(stripped)
    if explicit is not None:
        return {
            "command": explicit,
            "shell": _default_shell_name(),
            "mode": "explicit_command",
            "focus": "command",
        }
    if _looks_like_command(stripped):
        return {
            "command": stripped,
            "shell": _default_shell_name(),
            "mode": "direct_command",
            "focus": "command",
        }
    lowered = stripped.lower()
    if "powershell" in lowered or "pwsh" in lowered:
        return {
            "command": _powershell_diagnostic_command() if os.name == "nt" else _bash_powershell_probe_command(),
            "shell": "powershell" if os.name == "nt" else "bash",
            "mode": "diagnostic",
            "focus": "powershell",
        }
    if any(token in lowered for token in ("terminal", "shell", "bash", "cmd")):
        return {
            "command": _bash_diagnostic_command() if os.name != "nt" else _powershell_diagnostic_command(),
            "shell": _default_shell_name(),
            "mode": "diagnostic",
            "focus": "terminal",
        }
    return {
        "command": stripped,
        "shell": _default_shell_name(),
        "mode": "direct_command",
        "focus": "command",
    }


def build_terminal_execute_descriptor(project_root: Path) -> ToolDescriptor:
    definition = build_terminal_execute_tool_definition()

    def handler(params: Mapping[str, Any], _) -> dict[str, Any] | ToolInvocationResponse:
        command = str(params["command"]).strip()
        shell_name, prefix = _resolve_shell(str(params.get("shell") or "auto"))
        resolved_dir = _resolve_working_dir(project_root, str(params["working_dir"]) if params.get("working_dir") else None)
        timeout_ms = int(params.get("timeout_ms") or 15000)
        max_output_chars = int(params.get("max_output_chars") or 6000)
        allow_write = bool(params.get("allow_write", False))

        if not allow_write and _is_write_like(command):
            return ToolInvocationResponse.from_error(
                code=ErrorCode.AUTHORIZATION_DENIED,
                message="terminal command was blocked because it appears to mutate state; set allow_write=true",
                retryable=False,
            )

        started_ns = perf_counter_ns()
        try:
            completed = subprocess.run(
                [*prefix, command],
                cwd=str(resolved_dir),
                capture_output=True,
                text=True,
                timeout=timeout_ms / 1000.0,
                check=False,
            )
        except FileNotFoundError as exc:
            return ToolInvocationResponse.from_error(
                code=ErrorCode.UPSTREAM_FAILURE,
                message=f"terminal shell '{shell_name}' is not available: {exc}",
                retryable=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _truncate(exc.stdout or "", max_output_chars)
            stderr = _truncate(exc.stderr or "", max_output_chars)
            return {
                "command": command,
                "shell": shell_name,
                "working_dir": str(resolved_dir),
                "exit_code": -1,
                "duration_ms": timeout_ms,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": True,
                "summary": f"Command timed out after {timeout_ms} ms.",
            }

        duration_ms = max((perf_counter_ns() - started_ns) // 1_000_000, 0)
        stdout = _truncate(completed.stdout, max_output_chars)
        stderr = _truncate(completed.stderr, max_output_chars)
        summary = (
            f"Command exited with code {completed.returncode}. "
            f"stdout={len(completed.stdout)} chars stderr={len(completed.stderr)} chars."
        )
        return {
            "command": command,
            "shell": shell_name,
            "working_dir": str(resolved_dir),
            "exit_code": int(completed.returncode),
            "duration_ms": int(duration_ms),
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
            "summary": summary,
        }

    return ToolDescriptor(
        tool_id="terminal_execute",
        version="1.0.0",
        type_class=TypeClass.SDK_WRAPPED,
        input_schema=definition["function"]["parameters"],
        output_schema=_output_schema(),
        error_schema=_error_schema(),
        auth_contract=AuthContract(required_scopes=("tools.terminal",)),
        timeout_class=TimeoutClass.LONG_RUNNING,
        idempotency_spec=IdempotencySpec(enabled=True),
        mutation_class=MutationClass.READ_ONLY,
        observability_contract=ObservabilityContract(metric_prefix="examples.realtime_qa_chatbot.terminal"),
        source_endpoint="local://realtime_qa_chatbot/terminal",
        source_type=ToolSourceType.SDK,
        compressed_description=definition["function"]["description"],
        handler=handler,
        side_effect_manifest=SideEffectManifest(resources=("local.terminal",), operations=("inspect",), reversible=True),
        health_score=1.0,
        status=ToolStatus.ACTIVE,
        token_cost_estimate=96,
        cache_ttl_seconds=0,
        metadata={"workspace_root": str(project_root.resolve())},
    )


@dataclass(slots=True)
class RealtimeQaLoopEngine(AgentLoopEngine):
    project_root: Path = Path.cwd()

    def _extra_query_tool_definitions(self) -> tuple[dict[str, Any], ...]:
        return (build_terminal_execute_tool_definition(),)

    def _register_custom_tools(self, engine) -> None:
        engine.register(build_terminal_execute_descriptor(self.project_root))

    def _build_tool_params(self, state, tool_id: str, action):  # type: ignore[override]
        if tool_id != "terminal_execute":
            return AgentLoopEngine._build_tool_params(self, state, tool_id, action)
        inferred = infer_terminal_request(action.instruction)
        return {
            "command": inferred["command"],
            "shell": inferred["shell"],
            "timeout_ms": 15000,
            "max_output_chars": 6000,
            "allow_write": False,
        }


__all__ = [
    "RealtimeQaLoopEngine",
    "build_terminal_execute_descriptor",
    "build_terminal_execute_tool_definition",
    "infer_terminal_request",
]
