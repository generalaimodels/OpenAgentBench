-- Query-understanding cache and audit schema for OpenAgentBench.

CREATE TABLE IF NOT EXISTS agent_query_resolution_cache (
    cache_key           TEXT PRIMARY KEY,
    user_id             UUID NOT NULL,
    session_id          UUID NOT NULL,
    plan_data           JSONB NOT NULL,
    expires_at          TIMESTAMPTZ NOT NULL,
    hit_count           INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_query_resolution_cache_expires_at
    ON agent_query_resolution_cache (expires_at);

CREATE INDEX IF NOT EXISTS idx_agent_query_resolution_cache_user_session
    ON agent_query_resolution_cache (user_id, session_id);

CREATE TABLE IF NOT EXISTS agent_query_resolution_audit (
    audit_id            UUID PRIMARY KEY,
    user_id             UUID NOT NULL,
    session_id          UUID NOT NULL,
    cache_key           TEXT NOT NULL,
    ambiguity_level     TEXT NOT NULL,
    subquery_count      INTEGER NOT NULL,
    cache_hit           BOOLEAN NOT NULL DEFAULT FALSE,
    latency_ms          INTEGER NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_query_resolution_audit_user_session
    ON agent_query_resolution_audit (user_id, session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_query_resolution_audit_cache_key
    ON agent_query_resolution_audit (cache_key);
