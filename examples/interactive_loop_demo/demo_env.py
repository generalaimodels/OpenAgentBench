"""Environment loading and configuration for the interactive loop demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(slots=True, frozen=True)
class DemoConfig:
    project_root: Path
    env_path: Path
    workspace_root: Path
    allow_terminal_write: bool
    terminal_timeout_ms: int
    max_output_chars: int
    openai_api_key_present: bool
    gemini_api_key_present: bool
    openai_model: str
    openai_reasoning_effort: str
    openai_max_output_tokens: int
    openai_timeout_seconds: float


def load_demo_config(env_path: str | os.PathLike[str] | None = None) -> DemoConfig:
    project_root = Path(__file__).resolve().parents[2]
    resolved_env_path = Path(env_path) if env_path is not None else project_root / ".env"
    load_dotenv(resolved_env_path)
    if not resolved_env_path.exists():
        load_dotenv(project_root / ".env.example")
    workspace_root = project_root / "examples" / "interactive_loop_demo" / "workspace"
    return DemoConfig(
        project_root=project_root,
        env_path=resolved_env_path,
        workspace_root=workspace_root,
        allow_terminal_write=_parse_bool(os.environ.get("OPENAGENTBENCH_DEMO_ALLOW_TERMINAL_WRITE"), default=False),
        terminal_timeout_ms=max(int(os.environ.get("OPENAGENTBENCH_DEMO_TERMINAL_TIMEOUT_MS", "4000")), 250),
        max_output_chars=max(int(os.environ.get("OPENAGENTBENCH_DEMO_MAX_OUTPUT_CHARS", "6000")), 512),
        openai_api_key_present=bool(os.environ.get("OPENAI_API_KEY")),
        gemini_api_key_present=bool(os.environ.get("GEMINI_API_KEY")),
        openai_model=os.environ.get("OPENAGENTBENCH_OPENAI_MODEL", "gpt-5.4").strip() or "gpt-5.4",
        openai_reasoning_effort=(
            os.environ.get("OPENAGENTBENCH_OPENAI_REASONING_EFFORT", "medium").strip() or "medium"
        ),
        openai_max_output_tokens=max(int(os.environ.get("OPENAGENTBENCH_OPENAI_MAX_OUTPUT_TOKENS", "700")), 128),
        openai_timeout_seconds=max(
            _parse_float(os.environ.get("OPENAGENTBENCH_OPENAI_TIMEOUT_SECONDS"), default=45.0),
            5.0,
        ),
    )


__all__ = ["DemoConfig", "load_demo_config", "load_dotenv"]
