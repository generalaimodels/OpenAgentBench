"""Performance harness for the interactive loop demo."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover - script execution support
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os
import statistics
from dataclasses import dataclass

from examples.interactive_loop_demo.demo_runtime import build_application, load_demo_config, sample_prompts


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


@dataclass(slots=True, frozen=True)
class BenchmarkSummary:
    samples: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    average_actions: float
    average_evidence_items: float


def run_benchmark(
    iterations: int,
    *,
    provider: str = "auto",
    env_file: str | None = None,
    prompt_limit: int | None = None,
) -> BenchmarkSummary:
    latencies: list[float] = []
    action_counts: list[int] = []
    evidence_counts: list[int] = []
    prompts = sample_prompts()
    if prompt_limit is not None:
        prompts = prompts[: max(prompt_limit, 1)]

    for _ in range(iterations):
        app = build_application(config=load_demo_config(env_file), provider=provider)
        for prompt in prompts:
            result = app.run_query(prompt)
            latencies.append(result.latency_ns / 1_000_000.0)
            action_counts.append(len(result.action_outcomes))
            evidence_counts.append(0 if result.evidence is None else len(result.evidence.items))

    return BenchmarkSummary(
        samples=len(latencies),
        average_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        average_actions=statistics.mean(action_counts) if action_counts else 0.0,
        average_evidence_items=statistics.mean(evidence_counts) if evidence_counts else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAgentBench interactive loop benchmark.")
    parser.add_argument("--iterations", type=int, default=2, help="Number of full prompt suites to execute.")
    parser.add_argument(
        "--provider",
        choices=("auto", "demo", "openai"),
        default="auto",
        help="Select demo heuristics or the live OpenAI-backed framework runtime.",
    )
    parser.add_argument("--env-file", help="Override the .env path used by the demo.")
    parser.add_argument("--model", help="Override OPENAGENTBENCH_OPENAI_MODEL.")
    parser.add_argument("--reasoning-effort", help="Override OPENAGENTBENCH_OPENAI_REASONING_EFFORT.")
    parser.add_argument("--prompt-limit", type=int, help="Only run the first N sample prompts.")
    args = parser.parse_args()
    if args.model:
        os.environ["OPENAGENTBENCH_OPENAI_MODEL"] = args.model
    if args.reasoning_effort:
        os.environ["OPENAGENTBENCH_OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    summary = run_benchmark(
        max(args.iterations, 1),
        provider=args.provider,
        env_file=args.env_file,
        prompt_limit=args.prompt_limit,
    )
    print("Interactive Loop Benchmark")
    print(f"Samples: {summary.samples}")
    print(f"Average Latency: {summary.average_latency_ms:.2f} ms")
    print(f"P50 Latency: {summary.p50_latency_ms:.2f} ms")
    print(f"P95 Latency: {summary.p95_latency_ms:.2f} ms")
    print(f"Average Actions: {summary.average_actions:.2f}")
    print(f"Average Evidence Items: {summary.average_evidence_items:.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()
