from __future__ import annotations

from openagentbench.agent_loop import AgentLoopEngine, module_root, plan_path, read_plan


def test_agent_loop_public_api_surface_exposes_runtime_assets() -> None:
    assert module_root().name == "agent_loop"
    assert plan_path().name == "plan.md"
    assert "Agent Loop" in read_plan()
    assert AgentLoopEngine is not None
