"""Legacy CLI entrypoint for the interactive loop demo compatibility wrapper."""

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
    parser = argparse.ArgumentParser(description="Legacy interactive loop demo compatibility CLI.")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args(argv)
    app = DemoLoopApplication(config=load_demo_config())
    result = app.run_query(args.prompt)
    print(result.output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
