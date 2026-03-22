"""Convenience helpers for locating query-module assets from application code."""

from __future__ import annotations

from pathlib import Path


def module_root() -> Path:
    return Path(__file__).resolve().parent


def plan_path() -> Path:
    return module_root() / "plan.md"


def skills_path() -> Path:
    return module_root() / "skills.md"


def read_plan() -> str:
    return plan_path().read_text(encoding="utf-8")


def read_skills() -> str:
    return skills_path().read_text(encoding="utf-8")


def schema_sql_path() -> Path:
    return module_root().parents[1] / "agent_data" / "sql" / "004_agent_query_schema.sql"


def read_schema_sql() -> str:
    return schema_sql_path().read_text(encoding="utf-8")


__all__ = [
    "module_root",
    "plan_path",
    "read_plan",
    "read_schema_sql",
    "read_skills",
    "schema_sql_path",
    "skills_path",
]
