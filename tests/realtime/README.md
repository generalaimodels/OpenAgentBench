# Realtime Tests

This folder contains replay-grade realtime persistence tests for `agent_data`.

## Current Test

### [`test_realtime_dry.py`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/realtime/test_realtime_dry.py)

The dry realtime suite validates:

- deterministic text, image, and audio fixtures
- normalization of synthetic OpenAI and Gemini style captures
- secret redaction before persistence
- stable IDs for reproducible rows
- PostgreSQL writes through the production query builders
- PostgreSQL reads and retrieval checks for:
  - active history retrieval
  - keyword memory retrieval
  - semantic memory retrieval
  - context selection
- protocol ledger verification for tool-call and JSON-RPC-like traffic

## Production Tables Validated

- `sessions`
- `conversation_history`
- `memory_store`
- `model_api_calls`
- `model_stream_events`
- `protocol_events`

## Isolation

- each run uses a dedicated schema derived from the live test run id
- query templates are rewritten from `agent_data.*` into that isolated schema
- cleanup is optional and limited to test-owned rows

## Requirements

For PostgreSQL-backed checks:

- `TEST_DATABASE_URL`
- `pgvector`
- `pg_trgm`

If the database does not expose the required extensions or privileges, the PostgreSQL-backed checks skip rather than silently weakening coverage.

## Run Commands

Dry-only:

```bash
python -m pytest tests/realtime/test_realtime_dry.py
```

Dry plus PostgreSQL:

```bash
TEST_DATABASE_URL=postgresql://user:pass@host:5432/db \
python -m pytest tests/realtime/test_realtime_dry.py -k postgres
```
