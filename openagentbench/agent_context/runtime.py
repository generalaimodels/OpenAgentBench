"""Convenience helpers for locating agent-context assets from application code."""

from __future__ import annotations

from pathlib import Path


def module_root() -> Path:
    return Path(__file__).resolve().parent


def plan_path() -> Path:
    return module_root() / "plan.md"


def read_plan() -> str:
    return plan_path().read_text(encoding="utf-8")


__all__ = ["module_root", "plan_path", "read_plan"]
