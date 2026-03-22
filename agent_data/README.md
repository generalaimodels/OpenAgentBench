# Agent Data Module

`agent_data/` is the canonical home of the memory, session, history, and context-compilation subsystem for OpenAgentBench.

This folder contains:

- the architecture plan in [`plan.md`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/plan.md)
- the executable PostgreSQL schema in [`sql/001_agent_data_schema.sql`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/sql/001_agent_data_schema.sql)
- the API contract in [`api/openapi.yaml`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/api/openapi.yaml)
- Python entrypoints in [`__init__.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/__init__.py) and [`runtime.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/runtime.py)
- the reusable package implementation in [`openagentbench/agent_data`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_data/__init__.py)

## What Is Implemented

- Typed Python models for sessions, memories, history records, budgets, and compiled contexts.
- Lossless API-call and stream-event records so no request, response, token delta, or multimodal event has to be discarded during testing.
- A protocol-event ledger for MCP, JSON-RPC, and tool-call traffic so read/write updates are preserved exactly.
- A context compiler that emits OpenAI-compatible `messages` arrays with token-budget-aware packing.
- Retrieval scoring with semantic relevance, freshness decay, access-frequency weighting, and provenance authority.
- Greedy memory packing plus contiguous-suffix history packing, matching the plan's bounded-budget design.
- Fast JSON helpers with `orjson` fallback to keep JSONB-heavy paths compact and deterministic.
- Parameterized SQL templates for inserts, upserts, history fetches, raw API-call capture, stream-event capture, semantic retrieval, and trigram retrieval.
- A PostgreSQL schema with hash partitioning, monthly history sub-partitioning, HNSW/vector indexes, trigram search, and expiry/promotion helpers.

## Plan Alignment Notes

The implementation follows the plan closely, with a few production corrections required by PostgreSQL:

- Partitioned tables use composite primary keys that include partition keys. PostgreSQL does not allow global unique constraints that omit partition columns.
- A `system_prompts` registry is included because the plan stores `system_prompt_hash`, while the runtime still needs prompt text for context assembly.
- `conversation_history` uses helper-driven monthly sub-partition creation so the schema remains operational instead of hard-coding every month forever.
- Auxiliary `model_api_calls` and `model_stream_events` tables are added for replay-grade testing so raw API activity and token/event deltas are never lost.
- An auxiliary `protocol_events` table is added so MCP and JSON-RPC traffic does not disappear inside generic metadata blobs.

## Performance Intent

The module is designed around the same objectives as the plan:

- append-only history to minimize MVCC churn
- lossless API trace retention for benchmark replay and debugging
- explicit protocol-event persistence for tool ecosystems such as MCP and JSON-RPC
- pre-computed token counts for read-time budget packing
- JSONB and compact serialization for metadata-heavy writes
- pgvector HNSW indexes for semantic retrieval
- per-user hash partition locality for session, memory, and history access

Actual SOTA or p99 claims still require workload-specific benchmarking, index-tuning, and production telemetry. The code and schema are built to make that validation possible rather than hand-waving it.

## Quick Start

1. Apply [`sql/001_agent_data_schema.sql`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/sql/001_agent_data_schema.sql) to PostgreSQL.
2. Import the Python API from either `agent_data` or `openagentbench.agent_data`.
3. Use [`runtime.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/runtime.py) if you need to discover the schema or OpenAPI assets at runtime.

```python
from agent_data import ContextCompiler, CompileRequest
```
