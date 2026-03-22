# Realtime Q&A App

This example is a realtime Q&A application built on the full OpenAgentBench stack:

- `agent_data`
- `agent_memory`
- `agent_query`
- `agent_retrieval`
- `agent_tools`
- `agent_loop`

Unlike the generic loop demo, this app streams framework progress phase by phase by using the loop checkpoint/resume path:

- context assembly
- planning
- decomposition and prediction
- retrieval
- tool execution
- metacognitive check
- verification, critique, repair, and commit

It then streams the final answer text for the OpenAI-backed mode.

## Run

```bash
python3 examples/realtime_qa_app/app.py
```

Live OpenAI-backed mode:

```bash
python3 examples/realtime_qa_app/app.py --provider openai
```

One-shot prompt:

```bash
python3 examples/realtime_qa_app/app.py --prompt "Use the framework to answer: what tools are available and which tool can inspect memory?"
```

Terminal tool example:

```bash
python3 examples/realtime_qa_app/app.py --prompt "Run the command python3 -m unittest -q test_demo_stats.py in the terminal and summarize the result."
```

## Commands

- `/help`
- `/examples`
- `/status`
- `/trace`
- `/terminal <cmd>`
- `/quit`
