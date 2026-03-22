from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.interactive_loop_demo.demo_env import DemoConfig
from examples.interactive_loop_demo.demo_runtime import OpenAIDemoLoopApplication, build_application


@dataclass(slots=True)
class _FakeResponse:
    output_text: str


@dataclass(slots=True)
class _FakeResponsesAPI:
    calls: list[dict[str, Any]]

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        input_payload = kwargs.get("input")
        prompt_text = _flatten_input_text(input_payload)
        if "Resolve all pronouns and references" in prompt_text:
            return _FakeResponse("Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result.")
        if "Expand and rewrite the user's task for retrieval" in prompt_text:
            return _FakeResponse("Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result.")
        if "Produce a compact hypothetical answer" in prompt_text:
            return _FakeResponse("Hypothesis: the workspace tests pass.")
        if "Produce one clarifying question" in prompt_text:
            return _FakeResponse("Which workspace target should I inspect?")
        if "final response layer for OpenAgentBench" in prompt_text:
            return _FakeResponse("Synthesized framework answer from the fake OpenAI client.")
        return _FakeResponse("Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result.")


@dataclass(slots=True)
class _FakeClient:
    responses: _FakeResponsesAPI


def _flatten_input_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return "\n".join(parts)


def _build_config() -> DemoConfig:
    project_root = Path("/tmp/openagentbench-demo")
    return DemoConfig(
        project_root=project_root,
        env_path=project_root / ".env",
        workspace_root=Path("/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/examples/interactive_loop_demo/workspace"),
        allow_terminal_write=False,
        terminal_timeout_ms=4000,
        max_output_chars=6000,
        openai_api_key_present=True,
        gemini_api_key_present=False,
        openai_model="gpt-5.4",
        openai_reasoning_effort="medium",
        openai_max_output_tokens=700,
        openai_timeout_seconds=45.0,
    )


def test_openai_demo_application_uses_live_model_for_synthesis() -> None:
    fake_client = _FakeClient(responses=_FakeResponsesAPI(calls=[]))
    app = OpenAIDemoLoopApplication(config=_build_config(), client=fake_client)

    result = app.run_query("Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result.")

    assert result.output_text == "Synthesized framework answer from the fake OpenAI client."
    assert len(fake_client.responses.calls) >= 1
    assert any(call.get("store") is False for call in fake_client.responses.calls)


def test_openai_demo_streaming_falls_back_to_single_delta_without_stream_api() -> None:
    fake_client = _FakeClient(responses=_FakeResponsesAPI(calls=[]))
    app = OpenAIDemoLoopApplication(config=_build_config(), client=fake_client)
    chunks: list[str] = []

    result = app.run_query_stream(
        "Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result.",
        on_delta=chunks.append,
    )

    assert result.output_text == "Synthesized framework answer from the fake OpenAI client."
    assert "".join(chunks) == result.output_text


def test_build_application_auto_selects_openai_when_key_is_present() -> None:
    fake_client = _FakeClient(responses=_FakeResponsesAPI(calls=[]))

    app = build_application(config=_build_config(), provider="auto", client=fake_client)

    assert isinstance(app, OpenAIDemoLoopApplication)
    assert app.provider_name == "openai"
