from __future__ import annotations

from openagentbench.agent_memory import (
    MemoryContextCompiler,
    build_memory_inspect_tool_definition,
    build_memory_read_tool_definition,
    build_memory_write_tool_definition,
    module_root,
    plan_path,
    schema_sql_path,
)


def test_public_api_surface_exposes_runtime_assets_and_tools() -> None:
    assert module_root().name == "agent_memory"
    assert plan_path().name == "plan.md"
    assert schema_sql_path().name == "002_agent_memory_schema.sql"
    assert build_memory_read_tool_definition()["function"]["name"] == "memory_read"
    assert build_memory_write_tool_definition()["function"]["name"] == "memory_write"
    assert build_memory_inspect_tool_definition()["function"]["name"] == "memory_inspect"
    assert MemoryContextCompiler is not None
