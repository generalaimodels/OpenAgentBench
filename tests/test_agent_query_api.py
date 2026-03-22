from __future__ import annotations

from openagentbench.agent_query import (
    QueryResolver,
    build_query_clarify_tool_definition,
    build_query_resolve_tool_definition,
    module_root,
    plan_path,
    read_skills,
    schema_sql_path,
    skills_path,
)


def test_public_api_surface_exposes_runtime_assets_and_query_tools() -> None:
    assert module_root().name == "agent_query"
    assert plan_path().name == "plan.md"
    assert skills_path().name == "skills.md"
    assert schema_sql_path().name == "004_agent_query_schema.sql"
    assert build_query_resolve_tool_definition()["function"]["name"] == "query_resolve"
    assert build_query_clarify_tool_definition()["function"]["name"] == "query_clarify"
    assert "Psychological-State and Pressure Analysis" in read_skills()
    assert QueryResolver is not None
