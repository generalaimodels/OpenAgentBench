"""Convenience helpers for locating the agent-data assets from application code."""

from __future__ import annotations

from pathlib import Path

from openagentbench.agent_data import *  # noqa: F401,F403


def module_root() -> Path:
    return Path(__file__).resolve().parent


def schema_sql_path() -> Path:
    return module_root() / "sql" / "001_agent_data_schema.sql"


def openapi_spec_path() -> Path:
    return module_root() / "api" / "openapi.yaml"


def read_schema_sql() -> str:
    return schema_sql_path().read_text(encoding="utf-8")


def read_openapi_spec() -> str:
    return openapi_spec_path().read_text(encoding="utf-8")

