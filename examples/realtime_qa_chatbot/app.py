"""Interactive CLI for the OpenAgentBench realtime Q&A chatbot example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.realtime_qa_chatbot.runtime import ChatbotConfig, RealtimeQaChatbot, RealtimeQaEvent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime Q&A chatbot demo built on the full OpenAgentBench stack.",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "openai", "vllm", "heuristic"),
        default="auto",
        help="Answer synthesis provider. 'auto' prefers OpenAI when OPENAI_API_KEY is available.",
    )
    parser.add_argument("--model", help="Override the provider model id.")
    parser.add_argument("--base-url", help="Override the provider base URL, useful for vLLM.")
    parser.add_argument("--prompt", help="Run a single prompt and exit.")
    parser.add_argument("--tool-budget", type=int, default=768, help="Tool-schema token budget for the SDK surface.")
    parser.add_argument("--answer-max-tokens", type=int, default=640, help="Maximum synthesis tokens per answer.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming answer chunks.")
    return parser


def _help_text() -> str:
    return "\n".join(
        (
            "/help                  Show commands.",
            "/status                Show provider, session, memory, connector, and tool counts.",
            "/modules               Show how the example maps to the OpenAgentBench modules.",
            "/connectors            List SDK connectors and their operations.",
            "/tools [hint]          Show the projected tool surface for a task hint.",
            "/memory <query>        Read current memories through the SDK memory tool.",
            "/plan <query>          Preview the query-resolution plan without running a full loop.",
            "/history               Show the user questions captured in this chat session.",
            "/terminal <command>    Run the example-local terminal tool directly.",
            "/trace                 Show the last loop/context trace snapshot.",
            "/quit                  Exit the chatbot.",
            "Any other input runs a realtime Q&A turn.",
        )
    )


def _format_status(bot: RealtimeQaChatbot) -> str:
    snapshot = bot.status_snapshot()
    return "\n".join(
        (
            f"provider: {snapshot['provider']}",
            f"provider_model: {snapshot['provider_model']}",
            f"provider_note: {snapshot['provider_note']}",
            f"session_id: {snapshot['session_id']}",
            f"user_id: {snapshot['user_id']}",
            f"turn_count: {snapshot['turn_count']}",
            f"history_records: {snapshot['history_records']}",
            f"memory_records: {snapshot['memory_records']}",
            f"working_items: {snapshot['working_items']} / {snapshot['working_capacity']}",
            f"connector_count: {snapshot['connector_count']}",
            f"tool_count: {snapshot['tool_count']}",
        )
    )


def _format_modules() -> str:
    return "\n".join(
        (
            "agent_data: session, history, and memory records for the live chat state.",
            "agent_memory: working buffer, session summaries, and checkpoints after each turn.",
            "agent_query: intent resolution, subquery planning, and clarification analysis.",
            "agent_retrieval: retrieval-plan routing and loop evidence acquisition.",
            "agent_tools: registered framework tools executed through the SDK and loop engine.",
            "agent_context: canonical cyclic context compilation and invariant tracking.",
            "agent_loop: bounded realtime phase execution with checkpoint/resume.",
            "agent_sdk: connector projection, tool surfaces, provider wiring, and protocol views.",
        )
    )


def _format_connectors(bot: RealtimeQaChatbot) -> str:
    connectors = bot.list_connectors()
    if not connectors:
        return "No connectors are registered."
    lines: list[str] = []
    for connector in connectors:
        operation_names = ", ".join(operation.operation_id for operation in connector.operations[:6])
        lines.append(
            f"{connector.connector_id}: domain={connector.domain.value}, "
            f"modality={connector.modality.value}, protocol={connector.protocol.value}, "
            f"operations={operation_names}"
        )
    return "\n".join(lines)


def _format_tools(bot: RealtimeQaChatbot, hint: str) -> str:
    surface = bot.project_tools(hint)
    if not surface.tool_definitions:
        return "No tools were projected for the current hint."
    lines: list[str] = []
    for tool in surface.tool_definitions:
        function = tool["function"]
        lines.append(f"{function['name']}: {function['description']}")
    return "\n".join(lines)


def _format_memory(bot: RealtimeQaChatbot, query: str) -> str:
    payload = bot.memory_lookup(query)
    items = payload.get("items", ())
    if not items:
        return "No memory hits."
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"{item.get('layer')}/{item.get('scope')} score={item.get('score')}: {item.get('content')}"
        )
    return "\n".join(lines)


def _format_plan(bot: RealtimeQaChatbot, query: str) -> str:
    preview = bot.preview_query(query)
    lines = [
        f"intent: {preview.plan.intent.intent_class.value}",
        f"resolved_query: {preview.plan.rewrite.resolved_query}",
        f"expanded_query: {preview.plan.rewrite.expanded_query}",
        f"needs_clarification: {preview.plan.needs_clarification}",
    ]
    if preview.plan.subqueries:
        lines.append("subqueries:")
        for subquery in preview.plan.subqueries:
            tool_candidates = ", ".join(subquery.tool_candidates) or "none"
            lines.append(
                f"  - {subquery.step_id}: route={subquery.route_target.value}, tools={tool_candidates}, text={subquery.text}"
            )
    return "\n".join(lines)


def _format_trace(bot: RealtimeQaChatbot) -> str:
    trace = bot.last_trace_snapshot()
    if not trace:
        return "No turn has been executed yet."
    return "\n".join(f"{key}: {value}" for key, value in trace.items())


def _format_history(bot: RealtimeQaChatbot) -> str:
    questions = bot.question_history()
    if not questions:
        return "No user questions have been captured yet."
    return "\n".join(f"{index}. {question}" for index, question in enumerate(questions, start=1))


def _format_terminal(bot: RealtimeQaChatbot, command: str) -> str:
    payload = bot.terminal_execute(command)
    return "\n".join(
        (
            f"command: {payload.get('command')}",
            f"shell: {payload.get('shell')}",
            f"working_dir: {payload.get('working_dir')}",
            f"exit_code: {payload.get('exit_code')}",
            f"duration_ms: {payload.get('duration_ms')}",
            f"summary: {payload.get('summary')}",
            f"stdout:\n{payload.get('stdout', '')}",
            f"stderr:\n{payload.get('stderr', '')}",
        )
    )


def _print_event(event: RealtimeQaEvent, *, answer_open: bool) -> bool:
    if event.kind == "answer_delta":
        if not answer_open:
            print("assistant> ", end="", flush=True)
        print(event.message, end="", flush=True)
        return True
    if event.kind == "answer_final":
        if answer_open:
            print(flush=True)
        else:
            print(f"assistant> {event.message}")
        print(
            f"[answer] provider={event.payload.get('provider')} "
            f"model={event.payload.get('model')} "
            f"checkpoint={event.payload.get('checkpoint_seq')}"
        )
        return False
    print(f"[{event.title}] {event.message}")
    return answer_open


def _run_turn(bot: RealtimeQaChatbot, prompt: str) -> None:
    answer_open = False
    for event in bot.answer_realtime(prompt):
        answer_open = _print_event(event, answer_open=answer_open)


def _interactive_loop(bot: RealtimeQaChatbot) -> int:
    print("OpenAgentBench Realtime Q&A Chatbot")
    print(_format_status(bot))
    print(_help_text())
    while True:
        try:
            raw = input("qa> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0
        if not raw:
            continue
        if raw == "/quit":
            return 0
        if raw == "/help":
            print(_help_text())
            continue
        if raw == "/status":
            print(_format_status(bot))
            continue
        if raw == "/modules":
            print(_format_modules())
            continue
        if raw == "/connectors":
            print(_format_connectors(bot))
            continue
        if raw.startswith("/tools"):
            hint = raw[6:].strip() or (bot.last_turn.question if bot.last_turn is not None else "general question answering")
            print(_format_tools(bot, hint))
            continue
        if raw.startswith("/memory"):
            query = raw[7:].strip()
            if not query:
                print("Usage: /memory <query>")
                continue
            print(_format_memory(bot, query))
            continue
        if raw.startswith("/plan"):
            query = raw[5:].strip()
            if not query:
                print("Usage: /plan <query>")
                continue
            print(_format_plan(bot, query))
            continue
        if raw == "/history":
            print(_format_history(bot))
            continue
        if raw.startswith("/terminal"):
            command = raw[9:].strip()
            if not command:
                print("Usage: /terminal <command>")
                continue
            print(_format_terminal(bot, command))
            continue
        if raw == "/trace":
            print(_format_trace(bot))
            continue
        _run_turn(bot, raw)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bot = RealtimeQaChatbot(
        ChatbotConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            stream_answer=not args.no_stream,
            tool_budget=args.tool_budget,
            answer_max_tokens=args.answer_max_tokens,
        )
    )
    if args.prompt:
        _run_turn(bot, args.prompt)
        return 0
    return _interactive_loop(bot)


if __name__ == "__main__":
    raise SystemExit(main())
