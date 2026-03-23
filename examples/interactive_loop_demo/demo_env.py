"""Configuration helpers for the legacy interactive loop demo compatibility layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from examples.realtime_qa_chatbot.runtime import load_env_file


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def load_demo_config() -> DemoConfig:
    project_root = _repo_root()
    env_path = project_root / ".env"
    env_values = load_env_file(env_path)
    workspace_root = project_root / "examples" / "interactive_loop_demo" / "workspace"
    return DemoConfig(
        project_root=project_root,
        env_path=env_path,
        workspace_root=workspace_root,
        allow_terminal_write=False,
        terminal_timeout_ms=4_000,
        max_output_chars=6_000,
        openai_api_key_present=bool(env_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")),
        gemini_api_key_present=bool(env_values.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")),
        openai_model=env_values.get("OPENAGENTBENCH_OPENAI_MODEL") or env_values.get("OPENAI_MODEL") or "gpt-5.4",
        openai_reasoning_effort=env_values.get("OPENAGENTBENCH_OPENAI_REASONING_EFFORT") or "medium",
        openai_max_output_tokens=700,
        openai_timeout_seconds=45.0,
    )


__all__ = ["DemoConfig", "load_demo_config"]
