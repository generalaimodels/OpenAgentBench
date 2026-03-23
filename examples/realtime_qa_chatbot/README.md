# Realtime Q&A Chatbot

This example is a stateful CLI chatbot built on the current OpenAgentBench stack.

It combines the modules directly:

- `agent_data`: session, history, and memory records
- `agent_memory`: working memory, summaries, and checkpoints
- `agent_query`: intent resolution and subquery planning
- `agent_retrieval`: retrieval planning and evidence routing
- `agent_tools`: framework tool execution
- `agent_context`: cyclic context compilation and invariants
- `agent_loop`: bounded realtime phase execution with checkpoint/resume
- `agent_sdk`: connector projection, tool surfaces, and OpenAI/vLLM provider wiring

## Run

From the project root:

```bash
python examples/realtime_qa_chatbot/app.py
```

Run a single prompt:

```bash
python examples/realtime_qa_chatbot/app.py --prompt "How does the framework combine the loop, memory, and SDK layers?"
```

Force deterministic local synthesis:

```bash
python examples/realtime_qa_chatbot/app.py --provider heuristic
```

Force OpenAI:

```bash
python examples/realtime_qa_chatbot/app.py --provider openai
```

Use a vLLM OpenAI-compatible server:

```bash
python examples/realtime_qa_chatbot/app.py --provider vllm --base-url http://localhost:8000/v1
```

## Environment

The runtime automatically loads the project `.env` file.

Recognized variables:

- `OPENAI_API_KEY`
- `OPENAGENTBENCH_OPENAI_MODEL`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL`
- `OPENAGENTBENCH_VLLM_BASE_URL`
- `VLLM_BASE_URL`
- `OPENAGENTBENCH_VLLM_MODEL`
- `VLLM_MODEL`
- `OPENAGENTBENCH_VLLM_API_KEY`
- `VLLM_API_KEY`

If no external provider is configured, the example falls back to deterministic local answer synthesis while still running the full query, context, loop, tool, memory, and SDK pipeline.

## CLI Commands

- `/help`
- `/status`
- `/modules`
- `/connectors`
- `/tools [hint]`
- `/memory <query>`
- `/plan <query>`
- `/history`
- `/terminal <command>`
- `/trace`
- `/quit`
