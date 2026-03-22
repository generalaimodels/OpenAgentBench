"""Interactive REPL for the OpenAgentBench agent-loop demo."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover - script execution support
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os

from examples.interactive_loop_demo.benchmark import run_benchmark
from examples.interactive_loop_demo.demo_runtime import build_application, format_result, load_demo_config, sample_prompts


def _print_banner(app) -> None:
    config = app.config
    print("OpenAgentBench Interactive Loop Demo")
    print(f"Workspace: {config.workspace_root}")
    print(f"Env File: {config.env_path}")
    print(f"Provider: {app.provider_name}")
    print(f"Model: {app.model_label}")
    print(
        "Provider Keys: "
        f"openai={'present' if config.openai_api_key_present else 'demo/absent'} "
        f"gemini={'present' if config.gemini_api_key_present else 'demo/absent'}"
    )
    print(
        "OpenAI Runtime: "
        f"reasoning={config.openai_reasoning_effort} "
        f"max_output_tokens={config.openai_max_output_tokens} "
        f"timeout={config.openai_timeout_seconds:.0f}s"
    )
    print(
        "Terminal Mode: "
        f"{'workspace-write-enabled' if config.allow_terminal_write else 'read-only'} "
        f"(timeout={config.terminal_timeout_ms}ms)"
    )
    print("Type /help for commands.")


def _print_help() -> None:
    print("/help               show commands")
    print("/examples           show sample prompts")
    print("/status             show current session status")
    print("/trace              show the last full framework trace")
    print("/stream [on|off]    enable or disable streamed answer output")
    print("/benchmark [n] [k]  run the benchmark for n iterations and k prompts")
    print("/terminal <cmd>     execute a workspace command through the terminal tool")
    print("/quit               exit the demo")


def _print_status(app) -> None:
    print(f"Provider: {app.provider_name}")
    print(f"Model: {app.model_label}")
    print(f"Session: {app.session.session_id}")
    print(f"Turns: {app.session.turn_count}")
    print(f"History Records: {len(app.history)}")
    print(f"Memory Records: {len(app.memories)}")
    print(f"Working Items: {len(app.working_items)}")
    print(f"Summary: {app.session.summary_text}")


def _run_streaming_query(app, query: str) -> None:
    print("Answer:")
    result = app.run_query_stream(query, on_delta=lambda delta: print(delta, end="", flush=True))
    if result.output_text and not result.output_text.endswith("\n"):
        print()
    print(f"[mode={result.cognitive_mode.value if result.cognitive_mode else 'unknown'} actions={len(result.action_outcomes)} latency={result.latency_ns / 1_000_000:.2f} ms]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAgentBench interactive loop demo.")
    parser.add_argument("--prompt", help="Run one prompt and exit.")
    parser.add_argument("--env-file", help="Override the .env path used by the demo.")
    parser.add_argument(
        "--provider",
        choices=("auto", "demo", "openai"),
        default="auto",
        help="Select demo heuristics or the live OpenAI-backed framework runtime.",
    )
    parser.add_argument("--model", help="Override OPENAGENTBENCH_OPENAI_MODEL.")
    parser.add_argument("--reasoning-effort", help="Override OPENAGENTBENCH_OPENAI_REASONING_EFFORT.")
    parser.add_argument("--stream", action="store_true", help="Stream the final OpenAI answer text when available.")
    args = parser.parse_args()

    if args.model:
        os.environ["OPENAGENTBENCH_OPENAI_MODEL"] = args.model
    if args.reasoning_effort:
        os.environ["OPENAGENTBENCH_OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    config = load_demo_config(args.env_file)
    app = build_application(config=config, provider=args.provider)
    stream_mode = app.provider_name == "openai"
    if args.stream:
        stream_mode = True
    _print_banner(app)

    if args.prompt:
        if stream_mode:
            _run_streaming_query(app, args.prompt)
        else:
            result = app.run_query(args.prompt)
            print(format_result(result))
        return

    while True:
        try:
            raw = input("demo> ").strip()
        except EOFError:
            print()
            break
        if not raw:
            continue
        if raw == "/quit":
            break
        if raw == "/help":
            _print_help()
            continue
        if raw == "/examples":
            for prompt in sample_prompts():
                print(f"- {prompt}")
            continue
        if raw == "/status":
            _print_status(app)
            continue
        if raw == "/trace":
            if app.last_result is None:
                print("No query has been executed yet.")
            else:
                print(format_result(app.last_result))
            continue
        if raw.startswith("/stream"):
            parts = raw.split()
            if len(parts) == 1:
                print(f"Streaming: {'on' if stream_mode else 'off'}")
            elif len(parts) == 2 and parts[1] in {"on", "off"}:
                stream_mode = parts[1] == "on"
                print(f"Streaming: {'on' if stream_mode else 'off'}")
            else:
                print("Usage: /stream [on|off]")
            continue
        if raw.startswith("/benchmark"):
            parts = raw.split()
            iterations = int(parts[1]) if len(parts) > 1 else 2
            prompt_limit = int(parts[2]) if len(parts) > 2 else None
            summary = run_benchmark(
                max(iterations, 1),
                provider=args.provider,
                env_file=args.env_file,
                prompt_limit=prompt_limit,
            )
            print(f"Samples: {summary.samples}")
            print(f"Average Latency: {summary.average_latency_ms:.2f} ms")
            print(f"P50 Latency: {summary.p50_latency_ms:.2f} ms")
            print(f"P95 Latency: {summary.p95_latency_ms:.2f} ms")
            print(f"Average Actions: {summary.average_actions:.2f}")
            print(f"Average Evidence Items: {summary.average_evidence_items:.2f}")
            continue
        query = raw
        if raw.startswith("/terminal "):
            command = raw.removeprefix("/terminal ").strip()
            query = f"Run the command {command} in the terminal and summarize the result."
        if stream_mode:
            _run_streaming_query(app, query)
        else:
            result = app.run_query(query)
            print(format_result(result))


if __name__ == "__main__":  # pragma: no cover
    main()
