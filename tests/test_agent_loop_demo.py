from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from examples.interactive_loop_demo.demo_runtime import DemoLoopApplication, load_demo_config


def test_interactive_demo_terminal_tool_executes_inside_workspace() -> None:
    app = DemoLoopApplication(config=load_demo_config())
    result = app.run_query("Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result.")

    terminal_outcomes = [outcome for outcome in result.action_outcomes if outcome.tool_id == "terminal_execute"]

    assert terminal_outcomes
    assert terminal_outcomes[0].status.value == "success"
    output_text = f"{terminal_outcomes[0].output.get('stdout', '')}\n{terminal_outcomes[0].output.get('stderr', '')}"
    assert "OK" in output_text


def test_interactive_demo_infers_workspace_test_command() -> None:
    app = DemoLoopApplication(config=load_demo_config())
    result = app.run_query("Use the current framework tools to run the workspace unit tests and explain the result.")

    terminal_outcomes = [outcome for outcome in result.action_outcomes if outcome.tool_id == "terminal_execute"]

    assert terminal_outcomes
    assert terminal_outcomes[0].status.value == "success"
    assert terminal_outcomes[0].output.get("exit_code") == 0


def test_interactive_demo_realtime_path_emits_phase_events() -> None:
    app = DemoLoopApplication(config=load_demo_config())
    events = []
    result = app.run_query_realtime(
        "Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result.",
        on_event=events.append,
    )

    phases = [event.phase for event in events]

    assert phases
    assert phases[0] == "context_assemble"
    assert "act" in phases
    assert "verify" in phases
    assert result.output_text


def test_realtime_qa_example_cli_runs_one_prompt() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "examples/realtime_qa_app/app.py",
            "--provider",
            "demo",
            "--prompt",
            "List the available tools and explain which one can inspect memory.",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "[context_assemble]" in completed.stdout
    assert "[final mode=" in completed.stdout
