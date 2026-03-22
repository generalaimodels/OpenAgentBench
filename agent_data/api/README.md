# API Contract

[`openapi.yaml`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/api/openapi.yaml) defines the external contract for the memory/session/history module.

[`openai_python_endpoints.md`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/api/openai_python_endpoints.md) documents the OpenAI Python endpoint matrix that this module is designed to preserve and replay.

Core operations:

- create a session
- append one or more messages to a session
- compile a model-ready context window
- upsert or inspect memories
- inspect raw API calls and ordered stream events
- inspect MCP / JSON-RPC / tool-call protocol events
- trigger maintenance jobs such as promotion, expiry, and compaction

The contract is intentionally OpenAI-compatible at the context boundary: compiled responses return `messages` that can flow directly into Chat Completions style endpoints, stored history accepts multimodal content parts so text, image, audio, video references, and mixed payloads do not get flattened away, and protocol ledgers preserve MCP / JSON-RPC / tool-call request-response updates for replay-grade testing.
