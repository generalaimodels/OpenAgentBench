"""Convenience helpers for locating retrieval assets from application code."""

from __future__ import annotations

from pathlib import Path


def module_root() -> Path:
    return Path(__file__).resolve().parent


def schema_sql_path() -> Path:
    return module_root() / "sql" / "001_agent_retrieval_schema.sql"


def plan_path() -> Path:
    return module_root() / "plan.md"


def read_schema_sql() -> str:
    return schema_sql_path().read_text(encoding="utf-8")


def read_plan() -> str:
    return plan_path().read_text(encoding="utf-8")
