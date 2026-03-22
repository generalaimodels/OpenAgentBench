BEGIN;

CREATE TABLE IF NOT EXISTS agent_data.tool_registry (
    tool_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    type_class TEXT NOT NULL,
    input_schema JSONB NOT NULL,
    output_schema JSONB NOT NULL,
    error_schema JSONB NOT NULL,
    auth_contract JSONB NOT NULL,
    timeout_class TEXT NOT NULL,
    idempotency_spec JSONB NOT NULL,
    mutation_class TEXT NOT NULL,
    side_effect_manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
    observability_contract JSONB NOT NULL,
    health_score REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'active',
    deprecation_notice TEXT,
    source_endpoint TEXT NOT NULL,
    source_type TEXT NOT NULL,
    token_cost_estimate INTEGER NOT NULL DEFAULT 0,
    compressed_description TEXT NOT NULL,
    contract_tests_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_data.tool_idempotency_store (
    idempotency_key TEXT PRIMARY KEY,
    tool_id TEXT NOT NULL,
    user_id UUID NOT NULL,
    caller_session_id UUID,
    result_envelope JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_data.tool_invocation_audit (
    caller_id UUID NOT NULL,
    audit_id UUID NOT NULL,
    trace_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    tool_version TEXT NOT NULL,
    agent_id UUID NOT NULL,
    session_id UUID,
    auth_decision TEXT NOT NULL,
    status TEXT NOT NULL,
    input_hash BYTEA NOT NULL,
    mutation_class TEXT NOT NULL,
    error_code TEXT,
    latency_ms INTEGER NOT NULL,
    compute_cost REAL NOT NULL DEFAULT 0.0,
    token_cost INTEGER NOT NULL DEFAULT 0,
    side_effects JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (caller_id, audit_id)
) PARTITION BY HASH (caller_id);

CREATE TABLE IF NOT EXISTS agent_data.tool_approval_tickets (
    requested_by UUID NOT NULL,
    ticket_id UUID NOT NULL,
    tool_id TEXT NOT NULL,
    params_redacted JSONB NOT NULL,
    agent_id UUID NOT NULL,
    status TEXT NOT NULL,
    resolution_by UUID,
    resolution_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (requested_by, ticket_id)
) PARTITION BY HASH (requested_by);

CREATE UNLOGGED TABLE IF NOT EXISTS agent_data.tool_result_cache (
    cache_key TEXT PRIMARY KEY,
    tool_id TEXT NOT NULL,
    user_id UUID NOT NULL,
    result_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
);

DO $$
DECLARE
    partition_index INTEGER;
BEGIN
    FOR partition_index IN 0..255 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.tool_invocation_audit_p%1$s PARTITION OF agent_data.tool_invocation_audit FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.tool_approval_tickets_p%1$s PARTITION OF agent_data.tool_approval_tickets FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
    END LOOP;
END
$$;

CREATE INDEX IF NOT EXISTS idx_tool_registry_type_status
    ON agent_data.tool_registry (type_class, status);

CREATE INDEX IF NOT EXISTS idx_tool_registry_mutation_status
    ON agent_data.tool_registry (mutation_class, status);

CREATE INDEX IF NOT EXISTS idx_tool_registry_health
    ON agent_data.tool_registry (health_score DESC)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_tool_registry_token_cost
    ON agent_data.tool_registry (token_cost_estimate ASC);

CREATE INDEX IF NOT EXISTS idx_tool_idempotency_tool_user
    ON agent_data.tool_idempotency_store (tool_id, user_id);

CREATE INDEX IF NOT EXISTS idx_tool_idempotency_expires
    ON agent_data.tool_idempotency_store (expires_at);

CREATE INDEX IF NOT EXISTS idx_tool_audit_trace
    ON agent_data.tool_invocation_audit (trace_id);

CREATE INDEX IF NOT EXISTS idx_tool_audit_tool_time
    ON agent_data.tool_invocation_audit (caller_id, tool_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tool_audit_status
    ON agent_data.tool_invocation_audit (caller_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tool_approval_status
    ON agent_data.tool_approval_tickets (requested_by, status, expires_at);

CREATE INDEX IF NOT EXISTS idx_tool_cache_tool_user
    ON agent_data.tool_result_cache (tool_id, user_id, expires_at);

CREATE INDEX IF NOT EXISTS idx_tool_cache_expires
    ON agent_data.tool_result_cache (expires_at);

COMMIT;
