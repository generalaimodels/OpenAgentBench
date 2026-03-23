"""Minimal benchmark entrypoint for the legacy interactive loop demo wrapper."""

from __future__ import annotations

import argparse
from time import perf_counter

from .demo_env import load_demo_config
from .demo_runtime import DemoLoopApplication


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the compatibility interactive loop demo.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result.",
    )
    args = parser.parse_args(argv)
    app = DemoLoopApplication(config=load_demo_config())
    started = perf_counter()
    for _ in range(max(args.iterations, 1)):
        app.run_query(args.prompt)
    elapsed_ms = (perf_counter() - started) * 1000.0
    print(f"iterations={args.iterations} elapsed_ms={elapsed_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
