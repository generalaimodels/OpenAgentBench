"""Realtime interactive Q&A application built on the OpenAgentBench framework."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover - script execution support
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os
from typing import Callable

from examples.interactive_loop_demo.demo_runtime import (
    LoopProgressEvent,
    build_application,
    format_result,
    load_demo_config,
    sample_prompts,
)


def _print_banner(app) -> None:
    config = app.config
    print("OpenAgentBench Realtime Q&A Demo")
    print(f"Workspace: {config.workspace_root}")
    print(f"Provider: {app.provider_name}")
    print(f"Model: {app.model_label}")
    print("Runtime: phased checkpoint-resume streaming across query, retrieval, tools, memory, and loop")
    print("Type /help for commands.")


def _print_help() -> None:
    print("/help               show commands")
    print("/examples           show sample prompts")
    print("/status             show session state")
    print("/trace              show the last full framework trace")
    print("/terminal <cmd>     run a workspace command through the framework terminal tool")
    print("/quit               exit the application")


def _print_status(app) -> None:
    print(f"Provider: {app.provider_name}")
    print(f"Model: {app.model_label}")
    print(f"Session: {app.session.session_id}")
    print(f"Turns: {app.session.turn_count}")
    print(f"History Records: {len(app.history)}")
    print(f"Memory Records: {len(app.memories)}")
    print(f"Working Items: {len(app.working_items)}")
    print(f"Summary: {app.session.summary_text}")


def _print_event(event: LoopProgressEvent) -> None:
    metric_suffix = ""
    if event.metrics:
        metric_suffix = " | " + " ".join(f"{key}={value}" for key, value in event.metrics.items())
    print(f"[{event.phase}] {event.summary}{metric_suffix}")


def _build_answer_printer() -> Callable[[str], None]:
    printed_header = {"value": False}

    def _printer(delta: str) -> None:
        if not delta:
            return
        if not printed_header["value"]:
            print("Answer:")
            printed_header["value"] = True
        print(delta, end="", flush=True)

    return _printer


def _run_query(app, query: str) -> None:
    print("Framework:")
    answer_printer = _build_answer_printer()
    result = app.run_query_realtime(query, on_event=_print_event, on_delta=answer_printer)
    if result.output_text:
        print()
    print(
        f"[final mode={result.cognitive_mode.value if result.cognitive_mode else 'unknown'} "
        f"phase={result.last_completed_phase.value if result.last_completed_phase else 'none'} "
        f"actions={len(result.action_outcomes)} latency={result.latency_ns / 1_000_000:.2f} ms]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the realtime OpenAgentBench Q&A example.")
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
    args = parser.parse_args()

    if args.model:
        os.environ["OPENAGENTBENCH_OPENAI_MODEL"] = args.model
    if args.reasoning_effort:
        os.environ["OPENAGENTBENCH_OPENAI_REASONING_EFFORT"] = args.reasoning_effort

    config = load_demo_config(args.env_file)
    app = build_application(config=config, provider=args.provider)
    _print_banner(app)

    if args.prompt:
        _run_query(app, args.prompt)
        return

    while True:
        try:
            raw = input("qa> ").strip()
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
        query = raw
        if raw.startswith("/terminal "):
            command = raw.removeprefix("/terminal ").strip()
            query = f"Run the command {command} in the terminal and summarize the result."
        _run_query(app, query)


if __name__ == "__main__":  # pragma: no cover
    main()
