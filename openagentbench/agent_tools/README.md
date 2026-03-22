# Agent Tools

This package is the executable reference surface for the `agent_tools` orchestration layer.

It follows the same implementation style as the existing OpenAgentBench modules:

- typed contracts for tools, dispatch, approval, caching, and audit state
- an admission gate plus deterministic schema validation
- an in-memory registry and execution engine for tests and local integration
- protocol and model-format helpers for OpenAI, vLLM, Gemini, JSON-RPC, MCP, and A2A payloads
- integrated tool handlers that compose `agent_data`, `agent_retrieval`, and `agent_memory`
- runtime helpers that expose the package plan and SQL migration assets

The durable schema extension for this module lives in [003_agent_tools_schema.sql](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/sql/003_agent_tools_schema.sql).

