# Agent Memory

This folder is the executable reference surface for the advanced memory-management plan in [plan.md](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_memory/plan.md).

It is intentionally aligned with the existing `agent_data` and `agent_retrieval` module shapes:

- typed records
- provider and repository contracts
- parameterized SQL builders
- runtime asset readers
- OpenAI-compatible compatibility helpers
- in-memory reference behavior for tests
- full OpenAI-compatible endpoint matrix coverage reused from `agent_retrieval`, plus memory-specific tool compatibility

The current implementation is the foundation layer:

- budget allocation
- working-memory compaction
- session marker detection and summary updates
- promotion and authority-conflict decisions
- prefill compilation into OpenAI-compatible messages
- cache/checkpoint/audit SQL templates
- compatibility coverage for:
  - OpenAI `responses`
  - OpenAI `chat/completions`
  - OpenAI `embeddings`
  - OpenAI realtime session payloads
  - OpenAI audio speech, transcription, and translation payloads
  - OpenAI image generation and edit payloads
  - OpenAI video generation payloads
  - Gemini-compatible content and token-count payloads
  - memory read/write/inspect tool definitions

The durable PostgreSQL extension for this module lives in [002_agent_memory_schema.sql](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/sql/002_agent_memory_schema.sql).
