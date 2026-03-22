BEGIN;

DO $$
BEGIN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS vector';
EXCEPTION
    WHEN undefined_file THEN
        RAISE EXCEPTION 'pgvector (extension name: vector) is required for agent_data';
    WHEN insufficient_privilege THEN
        RAISE EXCEPTION 'insufficient privilege to create extension vector';
END
$$;

DO $$
BEGIN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS pg_trgm';
EXCEPTION
    WHEN undefined_file THEN
        RAISE NOTICE 'pg_trgm is unavailable; trigram recall paths will remain disabled';
    WHEN insufficient_privilege THEN
        RAISE NOTICE 'insufficient privilege to create extension pg_trgm';
END
$$;

DO $$
BEGIN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS pg_stat_statements';
EXCEPTION
    WHEN undefined_file THEN
        RAISE NOTICE 'pg_stat_statements is unavailable';
    WHEN insufficient_privilege THEN
        RAISE NOTICE 'insufficient privilege to create extension pg_stat_statements';
END
$$;

DO $$
BEGIN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS pg_cron';
EXCEPTION
    WHEN undefined_file THEN
        RAISE NOTICE 'pg_cron is unavailable; schedule maintenance from the application layer';
    WHEN insufficient_privilege THEN
        RAISE NOTICE 'insufficient privilege to create extension pg_cron';
END
$$;

CREATE SCHEMA IF NOT EXISTS agent_data;

CREATE OR REPLACE FUNCTION agent_data.touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

CREATE TABLE IF NOT EXISTS agent_data.system_prompts (
    prompt_hash BYTEA PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS agent_data.sessions (
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    status SMALLINT NOT NULL DEFAULT 1 CHECK (status BETWEEN 1 AND 4),
    model_id TEXT NOT NULL,
    context_window_size INTEGER NOT NULL CHECK (context_window_size > 0),
    system_prompt_hash BYTEA NOT NULL,
    system_prompt_tokens INTEGER NOT NULL CHECK (system_prompt_tokens >= 0),
    temperature REAL NOT NULL DEFAULT 0.7 CHECK (temperature >= 0.0),
    top_p REAL NOT NULL DEFAULT 1.0 CHECK (top_p > 0.0 AND top_p <= 1.0),
    max_response_tokens INTEGER NOT NULL DEFAULT 4096 CHECK (max_response_tokens >= 0),
    turn_count INTEGER NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
    total_prompt_tokens BIGINT NOT NULL DEFAULT 0 CHECK (total_prompt_tokens >= 0),
    total_completion_tokens BIGINT NOT NULL DEFAULT 0 CHECK (total_completion_tokens >= 0),
    total_cost_microcents BIGINT NOT NULL DEFAULT 0 CHECK (total_cost_microcents >= 0),
    summary_text TEXT,
    summary_embedding vector(1536),
    summary_token_count INTEGER NOT NULL DEFAULT 0 CHECK (summary_token_count >= 0),
    parent_session_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (user_id, session_id)
) PARTITION BY HASH (user_id);

ALTER TABLE agent_data.sessions SET (fillfactor = 90);

CREATE TABLE IF NOT EXISTS agent_data.memory_store (
    user_id UUID NOT NULL,
    memory_id UUID NOT NULL,
    session_id UUID,
    memory_tier SMALLINT NOT NULL CHECK (memory_tier BETWEEN 0 AND 4),
    memory_scope SMALLINT NOT NULL CHECK (memory_scope BETWEEN 0 AND 1),
    content_text TEXT NOT NULL,
    content_embedding vector(1536),
    content_hash BYTEA NOT NULL,
    provenance_type SMALLINT NOT NULL CHECK (provenance_type BETWEEN 0 AND 6),
    provenance_turn_id UUID,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    relevance_accumulator REAL NOT NULL DEFAULT 0.0,
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_validated BOOLEAN NOT NULL DEFAULT false,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    superseded_by UUID,
    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (user_id, memory_id)
) PARTITION BY HASH (user_id);

ALTER TABLE agent_data.memory_store SET (fillfactor = 85);

CREATE UNLOGGED TABLE IF NOT EXISTS agent_data.memory_store_working (
    user_id UUID NOT NULL,
    memory_id UUID NOT NULL,
    session_id UUID NOT NULL,
    memory_tier SMALLINT NOT NULL DEFAULT 0 CHECK (memory_tier = 0),
    memory_scope SMALLINT NOT NULL DEFAULT 0 CHECK (memory_scope = 0),
    content_text TEXT NOT NULL,
    content_embedding vector(1536),
    content_hash BYTEA NOT NULL,
    provenance_type SMALLINT NOT NULL CHECK (provenance_type BETWEEN 0 AND 6),
    provenance_turn_id UUID,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    relevance_accumulator REAL NOT NULL DEFAULT 0.0,
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_validated BOOLEAN NOT NULL DEFAULT false,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    superseded_by UUID,
    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (user_id, memory_id)
);

ALTER TABLE agent_data.memory_store_working SET (fillfactor = 100);

CREATE TABLE IF NOT EXISTS agent_data.conversation_history (
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    message_id UUID NOT NULL,
    turn_index INTEGER NOT NULL,
    role SMALLINT NOT NULL CHECK (role BETWEEN 0 AND 4),
    content TEXT,
    content_parts JSONB,
    name TEXT,
    tool_calls JSONB,
    tool_call_id TEXT,
    content_embedding vector(1536),
    content_hash BYTEA,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    model_id TEXT,
    finish_reason SMALLINT CHECK (finish_reason BETWEEN 0 AND 3),
    prompt_tokens INTEGER CHECK (prompt_tokens IS NULL OR prompt_tokens >= 0),
    completion_tokens INTEGER CHECK (completion_tokens IS NULL OR completion_tokens >= 0),
    latency_ms INTEGER CHECK (latency_ms IS NULL OR latency_ms >= 0),
    api_call_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_compressed BOOLEAN NOT NULL DEFAULT false,
    compressed_summary_id UUID,
    is_pruned BOOLEAN NOT NULL DEFAULT false,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (user_id, created_at, message_id)
) PARTITION BY HASH (user_id);

ALTER TABLE agent_data.conversation_history SET (fillfactor = 100);

CREATE TABLE IF NOT EXISTS agent_data.model_api_calls (
    user_id UUID NOT NULL,
    api_call_id UUID NOT NULL,
    session_id UUID NOT NULL,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model_id TEXT NOT NULL,
    request_payload JSONB NOT NULL,
    response_payload JSONB,
    usage_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_payload JSONB,
    request_id TEXT,
    status_code INTEGER,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    latency_ms INTEGER CHECK (latency_ms IS NULL OR latency_ms >= 0),
    succeeded BOOLEAN NOT NULL DEFAULT false,
    stream_mode BOOLEAN NOT NULL DEFAULT false,
    input_token_count INTEGER NOT NULL DEFAULT 0 CHECK (input_token_count >= 0),
    output_token_count INTEGER NOT NULL DEFAULT 0 CHECK (output_token_count >= 0),
    cached_input_token_count INTEGER NOT NULL DEFAULT 0 CHECK (cached_input_token_count >= 0),
    reasoning_token_count INTEGER NOT NULL DEFAULT 0 CHECK (reasoning_token_count >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (user_id, api_call_id)
) PARTITION BY HASH (user_id);

ALTER TABLE agent_data.model_api_calls SET (fillfactor = 90);

CREATE TABLE IF NOT EXISTS agent_data.model_stream_events (
    user_id UUID NOT NULL,
    api_call_id UUID NOT NULL,
    event_index INTEGER NOT NULL CHECK (event_index >= 0),
    event_type TEXT NOT NULL,
    text_delta TEXT,
    token_count INTEGER NOT NULL DEFAULT 0 CHECK (token_count >= 0),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    binary_payload BYTEA,
    mime_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, api_call_id, event_index)
) PARTITION BY HASH (user_id);

ALTER TABLE agent_data.model_stream_events SET (fillfactor = 100);

CREATE TABLE IF NOT EXISTS agent_data.protocol_events (
    user_id UUID NOT NULL,
    protocol_event_id UUID NOT NULL,
    session_id UUID NOT NULL,
    api_call_id UUID,
    message_id UUID,
    protocol_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    method TEXT,
    rpc_id TEXT,
    tool_name TEXT,
    tool_call_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    binary_payload BYTEA,
    mime_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, protocol_event_id)
) PARTITION BY HASH (user_id);

ALTER TABLE agent_data.protocol_events SET (fillfactor = 100);

DO $$
DECLARE
    partition_index INTEGER;
BEGIN
    FOR partition_index IN 0..255 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.sessions_p%1$s PARTITION OF agent_data.sessions FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.memory_store_p%1$s PARTITION OF agent_data.memory_store FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.model_api_calls_p%1$s PARTITION OF agent_data.model_api_calls FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.model_stream_events_p%1$s PARTITION OF agent_data.model_stream_events FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.protocol_events_p%1$s PARTITION OF agent_data.protocol_events FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
    END LOOP;
END
$$;

DO $$
DECLARE
    partition_index INTEGER;
BEGIN
    FOR partition_index IN 0..511 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.conversation_history_p%1$s PARTITION OF agent_data.conversation_history FOR VALUES WITH (modulus 512, remainder %1$s) PARTITION BY RANGE (created_at)',
            partition_index
        );
    END LOOP;
END
$$;

CREATE OR REPLACE FUNCTION agent_data.ensure_conversation_history_month_partitions(
    start_month DATE,
    month_count INTEGER
)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    month_offset INTEGER;
    partition_index INTEGER;
    range_start DATE;
    range_end DATE;
    child_partition_name TEXT;
    parent_partition_name TEXT;
BEGIN
    IF date_trunc('month', start_month)::DATE <> start_month THEN
        RAISE EXCEPTION 'start_month must be the first day of a month';
    END IF;

    IF month_count <= 0 THEN
        RAISE EXCEPTION 'month_count must be positive';
    END IF;

    FOR month_offset IN 0..month_count - 1 LOOP
        range_start := (start_month + make_interval(months => month_offset))::DATE;
        range_end := (range_start + INTERVAL '1 month')::DATE;
        FOR partition_index IN 0..511 LOOP
            parent_partition_name := format('conversation_history_p%s', partition_index);
            child_partition_name := format(
                'conversation_history_p%s_%s',
                partition_index,
                to_char(range_start, 'YYYYMM')
            );

            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS agent_data.%I PARTITION OF agent_data.%I FOR VALUES FROM (%L) TO (%L)',
                child_partition_name,
                parent_partition_name,
                range_start,
                range_end
            );
        END LOOP;
    END LOOP;
END;
$$;

SELECT agent_data.ensure_conversation_history_month_partitions(
    (date_trunc('month', now())::DATE - INTERVAL '3 months')::DATE,
    18
);

CREATE INDEX IF NOT EXISTS sessions_user_active_idx
    ON agent_data.sessions (user_id, updated_at DESC)
    INCLUDE (session_id, model_id, turn_count, expires_at)
    WHERE status = 1;

CREATE INDEX IF NOT EXISTS sessions_user_all_idx
    ON agent_data.sessions (user_id, created_at DESC)
    INCLUDE (session_id, status, model_id);

CREATE INDEX IF NOT EXISTS sessions_expiry_idx
    ON agent_data.sessions (expires_at)
    INCLUDE (user_id, session_id)
    WHERE status <> 4;

CREATE INDEX IF NOT EXISTS sessions_summary_embedding_hnsw_idx
    ON agent_data.sessions USING hnsw (summary_embedding vector_cosine_ops)
    WHERE summary_embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS memory_store_user_tier_active_idx
    ON agent_data.memory_store (user_id, memory_tier, memory_scope, access_count DESC)
    INCLUDE (
        memory_id,
        session_id,
        token_count,
        confidence,
        provenance_type,
        created_at,
        last_accessed_at
    )
    WHERE is_active = true;

CREATE INDEX IF NOT EXISTS memory_store_content_hash_idx
    ON agent_data.memory_store (user_id, content_hash)
    INCLUDE (memory_id, updated_at);

CREATE INDEX IF NOT EXISTS memory_store_expiry_idx
    ON agent_data.memory_store (expires_at)
    INCLUDE (user_id, memory_id)
    WHERE expires_at IS NOT NULL AND is_active = true;

CREATE INDEX IF NOT EXISTS memory_store_tags_gin_idx
    ON agent_data.memory_store USING gin (tags)
    WHERE is_active = true;

CREATE INDEX IF NOT EXISTS memory_store_metadata_gin_idx
    ON agent_data.memory_store USING gin (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS memory_store_text_trgm_idx
    ON agent_data.memory_store USING gin (content_text gin_trgm_ops)
    WHERE is_active = true;

CREATE INDEX IF NOT EXISTS memory_store_embedding_hnsw_idx
    ON agent_data.memory_store USING hnsw (content_embedding vector_cosine_ops)
    WHERE is_active = true AND content_embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS memory_store_superseded_idx
    ON agent_data.memory_store (user_id, superseded_by)
    WHERE superseded_by IS NOT NULL;

CREATE INDEX IF NOT EXISTS conversation_history_session_turn_idx
    ON agent_data.conversation_history (user_id, session_id, turn_index DESC)
    INCLUDE (
        message_id,
        role,
        content,
        content_parts,
        name,
        tool_calls,
        tool_call_id,
        token_count,
        created_at
    );

CREATE INDEX IF NOT EXISTS conversation_history_session_active_idx
    ON agent_data.conversation_history (user_id, session_id, turn_index DESC)
    INCLUDE (
        message_id,
        role,
        content,
        content_parts,
        name,
        tool_calls,
        tool_call_id,
        token_count,
        created_at
    )
    WHERE is_compressed = false AND is_pruned = false;

CREATE INDEX IF NOT EXISTS conversation_history_compaction_idx
    ON agent_data.conversation_history (user_id, session_id, created_at)
    INCLUDE (message_id, token_count)
    WHERE is_compressed = false;

CREATE INDEX IF NOT EXISTS conversation_history_embedding_hnsw_idx
    ON agent_data.conversation_history USING hnsw (content_embedding vector_cosine_ops)
    WHERE content_embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS model_api_calls_session_started_idx
    ON agent_data.model_api_calls (user_id, session_id, started_at DESC)
    INCLUDE (
        api_call_id,
        endpoint,
        model_id,
        request_id,
        latency_ms,
        succeeded,
        stream_mode
    );

CREATE INDEX IF NOT EXISTS model_api_calls_request_id_idx
    ON agent_data.model_api_calls (user_id, request_id)
    WHERE request_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS model_stream_events_created_idx
    ON agent_data.model_stream_events (user_id, api_call_id, created_at, event_index);

CREATE INDEX IF NOT EXISTS protocol_events_session_created_idx
    ON agent_data.protocol_events (user_id, session_id, created_at DESC)
    INCLUDE (
        protocol_event_id,
        protocol_type,
        direction,
        method,
        rpc_id,
        tool_name,
        tool_call_id,
        api_call_id,
        message_id
    );

CREATE INDEX IF NOT EXISTS protocol_events_lookup_idx
    ON agent_data.protocol_events (user_id, protocol_type, tool_call_id, rpc_id)
    WHERE tool_call_id IS NOT NULL OR rpc_id IS NOT NULL;

CREATE TRIGGER sessions_touch_updated_at
BEFORE UPDATE ON agent_data.sessions
FOR EACH ROW
EXECUTE FUNCTION agent_data.touch_updated_at();

CREATE TRIGGER memory_store_touch_updated_at
BEFORE UPDATE ON agent_data.memory_store
FOR EACH ROW
EXECUTE FUNCTION agent_data.touch_updated_at();

CREATE TRIGGER memory_store_working_touch_updated_at
BEFORE UPDATE ON agent_data.memory_store_working
FOR EACH ROW
EXECUTE FUNCTION agent_data.touch_updated_at();

COMMENT ON TABLE agent_data.sessions IS
'Session control plane. Composite primary key includes user_id because PostgreSQL partitioned uniqueness requires partition keys.';

COMMENT ON TABLE agent_data.memory_store IS
'Tiered durable memory plane. Vector, trigram, and metadata indexes support hybrid retrieval.';

COMMENT ON TABLE agent_data.memory_store_working IS
'UNLOGGED working-memory cache. Crash loss is acceptable because the state is reconstructible from active session context.';

COMMENT ON TABLE agent_data.conversation_history IS
'Append-dominant message history plane. Hash partitioned by user_id and range sub-partitioned by created_at.';

COMMENT ON TABLE agent_data.model_api_calls IS
'Lossless upstream API-call ledger. Stores raw request/response payloads, usage payloads, request ids, and status for reproducible evals.';

COMMENT ON TABLE agent_data.model_stream_events IS
'Ordered stream-event ledger. Preserves token deltas, multimodal event payloads, and optional binary chunks for replay-grade testing.';

COMMENT ON TABLE agent_data.protocol_events IS
'Protocol ledger for MCP, JSON-RPC, and tool-call traffic. Preserves read/write request-response updates with exact payload ordering.';

COMMIT;
