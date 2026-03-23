"""Legacy CLI wrapper around the interactive loop demo compatibility app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.interactive_loop_demo.demo_env import load_demo_config
from examples.interactive_loop_demo.demo_runtime import DemoLoopApplication


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Legacy realtime Q&A app compatibility wrapper.")
    parser.add_argument("--provider", default="demo")
    parser.add_argument("--prompt")
    args = parser.parse_args(argv)

    app = DemoLoopApplication(config=load_demo_config())

    def emit(event) -> None:
        print(f"[{event.phase}] {event.message}")

    prompt = args.prompt or "List the available tools and explain which one can inspect memory."
    result = app.run_query_realtime(prompt, on_event=emit)
    print(f"[final mode={args.provider}] {result.output_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
