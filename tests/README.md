# Tests

This folder contains the verification surface for the `agent_data`, `agent_retrieval`, and `agent_memory` modules.

## Files

### [`test_agent_data.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_data.py)

Fast unit-style checks for the production `agent_data` Python module.

Coverage:

- budget allocation
- context compiler message assembly
- multimodal `content_parts` preservation
- normalized hashing for deduplication
- endpoint catalog coverage for embeddings and video endpoints

### [`test_agent_retrieval.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_retrieval.py)

Hybrid retrieval engine checks for:

- multimodal and protocol-aware query classification
- in-memory orchestration
- user isolation
- negative evidence suppression
- provider adapter behavior
- routing and token-budget behavior

### [`test_agent_retrieval_compat.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_retrieval_compat.py)

Compatibility-focused retrieval checks for:

- OpenAI Responses, Chat Completions, and Embeddings request shapes
- Gemini `generateContent` and `countTokens` request shapes
- classification coverage for:
  - normal single-model queries
  - multimodal MIMO queries
  - reasoning queries
  - agentic-loop queries

### [`test_agent_retrieval_api.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_retrieval_api.py)

API-surface checks for:

- OpenAI and Gemini payload builders
- SQL template builders for read/write retrieval paths
- runtime asset readers
- selective model-plan API behavior
- success experiments for:
  - normal single-model retrieval
  - multimodal MIMO retrieval
  - thinking-heavy retrieval
  - dual-model reranking retrieval
  - agentic-loop retrieval
- Gemini smoke failure contract

### [`test_agent_memory.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_memory.py)

Memory-layer execution checks for:

- session, local, and global memory separation
- working-memory compaction and multimodal externalization
- session summary preservation of corrections and decisions
- promotion routing into procedural memory

### [`test_agent_memory_compat.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_memory_compat.py)

Compatibility-focused memory checks for:

- OpenAI-compatible tool-payload shapes for memory operations
- provider-suite interoperability with OpenAI-compatible embeddings and responses

### [`test_agent_memory_sql.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_memory_sql.py)

SQL and schema checks for:

- user-scoped local/global/session retrieval templates
- checkpoint, audit, and cache template construction
- additive schema migration presence for the memory extension

### [`test_agent_memory_api.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_memory_api.py)

API-surface checks for:

- public runtime asset resolution
- exported memory tool definitions
- compiler visibility from the package surface

### [`test_agent_integration_matrix.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_integration_matrix.py)

Cross-module integration checks for:

- `agent_data` + `agent_retrieval` + `agent_memory` combined message flow
- session, local, and global memory isolation inside the integrated stack
- scenario matrix coverage for:
  - normal session-heavy queries
  - multimodal MIMO queries
  - thinking/reasoning queries
  - agentic-loop queries
- API payload compatibility across retrieval and memory tool surfaces
- schema/runtime asset presence across all three modules

## Success Signals

The retrieval experiment suite is designed to make a passing run easy to interpret:

- `normal-single-model` passes when the classifier stays `single_model`, `siso`, `single_pass`, and the router selects a text-generation primary model
- `multimodal-mimo` passes when the classifier promotes the query to multimodal MIMO handling and the router binds a multimodal-capable primary model
- `thinking-reasoning` passes when the classifier raises reasoning effort and the router binds a thinking-capable model
- `dual-model-reranking` passes when the classifier enables dual-model execution and the router binds embedding plus reranking roles
- `agentic-loop` passes when the classifier enters multi-model agentic-loop mode and the router binds planner, executor, and critic roles with tool support

### [`realtime/test_realtime_dry.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/realtime/test_realtime_dry.py)

Zero-network validation for realtime persistence and retrieval.

Coverage:

- deterministic PNG and PCM16 fixture generation
- stable record IDs and payload redaction
- live-test budget guard behavior
- PostgreSQL schema rewrite into an isolated test schema
- synthetic OpenAI and Gemini capture normalization
- PostgreSQL round-trip validation for:
  - `sessions`
  - `conversation_history`
  - `memory_store`
  - `model_api_calls`
  - `model_stream_events`
  - `protocol_events`
- retrieval-path verification:
  - active history reads
  - keyword memory search
  - semantic memory search
  - context selection

## Database Expectations

When `TEST_DATABASE_URL` is set, the dry realtime suite validates:

- inserts and keyed reads for the production schema
- ordering and lossless round-trip of multimodal `content_parts`
- contiguous stream-event ordering
- protocol event ordering and identity preservation
- keyword and semantic retrieval returning the intended memory row first
- compiler selection using rows loaded back from PostgreSQL

The tests use a dedicated per-run schema and only delete test-owned rows when `LIVE_TEST_CLEANUP=1`.

## Environment Variables

- `TEST_DATABASE_URL`
- `RUN_LIVE_REALTIME_TESTS=1`
- `LIVE_TEST_MAX_BUDGET_USD=3.0`
- `LIVE_TEST_VENDOR=all|openai|gemini`
- `OPENAI_REALTIME_MODEL=gpt-realtime-mini`
- `GEMINI_LIVE_MODEL=gemini-2.5-flash-native-audio-preview-12-2025`
- `LIVE_TEST_CLEANUP=1`

## Run Commands

Run the base suite:

```bash
python -m pytest tests/test_agent_data.py tests/test_agent_retrieval.py tests/test_agent_retrieval_compat.py tests/test_agent_retrieval_api.py tests/realtime/test_realtime_dry.py
```

Run retrieval-only compatibility and API checks without `pytest`:

```bash
PYTHONPATH=OpenAgentBench python3 -m unittest \
  discover -s OpenAgentBench/tests -p 'test_agent_retrieval*.py'
```

Run only PostgreSQL-backed dry checks:

```bash
TEST_DATABASE_URL=postgresql://user:pass@host:5432/db \
python -m pytest tests/realtime/test_realtime_dry.py -k postgres
```
