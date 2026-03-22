# SQL Assets

[`001_agent_data_schema.sql`](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/agent_data/sql/001_agent_data_schema.sql) creates the PostgreSQL-backed storage layer for:

- `sessions`
- `memory_store`
- `memory_store_working`
- `conversation_history`
- `system_prompts`

The migration also provisions:

- required extensions when available
- hash partitions for session and memory data
- hash plus monthly range sub-partitions for conversation history
- vector, trigram, GIN, and B-tree indexes aligned to the retrieval plan
- helper functions for `updated_at` maintenance and rolling monthly partition creation

Run it with a PostgreSQL role that can create extensions and schema objects if you want the full feature set.
