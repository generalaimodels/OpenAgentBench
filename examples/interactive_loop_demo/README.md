# Interactive Loop Demo

This example shows the full six-module OpenAgentBench stack working together:

- `agent_data`
- `agent_memory`
- `agent_query`
- `agent_retrieval`
- `agent_tools`
- `agent_loop`

It adds one demo-only local tool: `terminal_execute`.

The CLI now supports two runtime modes:

- `--provider demo` keeps the local deterministic six-module demo
- `--provider openai` uses the real OpenAI Python client and model while still executing through the same framework modules

## Safety Model

- terminal commands are restricted to `examples/interactive_loop_demo/workspace`
- command chaining is disabled
- destructive commands are blocked
- workspace writes are disabled by default

To opt into workspace writes for local experiments:

```bash
OPENAGENTBENCH_DEMO_ALLOW_TERMINAL_WRITE=1 python3 examples/interactive_loop_demo/app.py
```

## Environment

The demo loads `OpenAgentBench/.env` if present, then falls back to `OpenAgentBench/.env.example`.

The provider keys are only surfaced as presence checks in the demo banner.

For the live OpenAI-backed mode, install the client and set a real API key:

```bash
python3 -m pip install openai
```

Optional runtime knobs:

- `OPENAGENTBENCH_OPENAI_MODEL=gpt-5.4`
- `OPENAGENTBENCH_OPENAI_REASONING_EFFORT=medium`
- `OPENAGENTBENCH_OPENAI_MAX_OUTPUT_TOKENS=700`
- `OPENAGENTBENCH_OPENAI_TIMEOUT_SECONDS=45`

## Run The REPL

```bash
python3 examples/interactive_loop_demo/app.py
```

Run the same CLI with the real OpenAI-backed framework mode:

```bash
python3 examples/interactive_loop_demo/app.py --provider openai
```

Useful commands:

- `/examples`
- `/status`
- `/terminal python3 -m unittest -q test_demo_stats.py`
- `/benchmark 2`
- `/benchmark 1 1`

## Run One Prompt

```bash
python3 examples/interactive_loop_demo/app.py --prompt "Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result."
```

Live OpenAI-backed run:

```bash
python3 examples/interactive_loop_demo/app.py --provider openai --prompt "Use the current framework tools to solve this task: run the workspace unit tests and explain what passed."
```

## Run The Benchmark

```bash
python3 examples/interactive_loop_demo/benchmark.py --iterations 2
```

Low-cost live benchmark smoke test:

```bash
python3 examples/interactive_loop_demo/benchmark.py --provider openai --iterations 1 --prompt-limit 1
```
