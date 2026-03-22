BEGIN;

DO $$
BEGIN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS vector';
EXCEPTION
    WHEN undefined_file THEN
        RAISE EXCEPTION 'pgvector (extension name: vector) is required for agent_retrieval';
    WHEN insufficient_privilege THEN
        RAISE EXCEPTION 'insufficient privilege to create extension vector';
END
$$;

DO $$
BEGIN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS pg_trgm';
EXCEPTION
    WHEN undefined_file THEN
        RAISE NOTICE 'pg_trgm is unavailable; trigram retrieval paths will remain disabled';
    WHEN insufficient_privilege THEN
        RAISE NOTICE 'insufficient privilege to create extension pg_trgm';
END
$$;

CREATE SCHEMA IF NOT EXISTS agent_retrieval;

CREATE OR REPLACE FUNCTION agent_retrieval.touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

CREATE TABLE IF NOT EXISTS agent_retrieval.users (
    uu_id UUID PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('active', 'disabled', 'deleted')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_retrieval.session (
    uu_id UUID NOT NULL,
    session_id UUID NOT NULL,
    turn_index INTEGER NOT NULL CHECK (turn_index >= 0),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content_text TEXT NOT NULL,
    content_embedding vector(1536),
    tokens_used INTEGER NOT NULL DEFAULT 0 CHECK (tokens_used >= 0),
    tool_calls JSONB,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    PRIMARY KEY (uu_id, session_id, turn_index)
) PARTITION BY HASH (uu_id);

ALTER TABLE agent_retrieval.session SET (fillfactor = 90);

CREATE TABLE IF NOT EXISTS agent_retrieval.history (
    uu_id UUID NOT NULL,
    history_id UUID NOT NULL,
    query_text TEXT NOT NULL,
    query_embedding vector(1536),
    response_summary TEXT,
    evidence_used JSONB NOT NULL DEFAULT '[]'::jsonb,
    task_outcome TEXT NOT NULL CHECK (task_outcome IN ('success', 'partial', 'failure', 'unknown')),
    human_feedback TEXT NOT NULL DEFAULT 'none' CHECK (human_feedback IN ('approved', 'rejected', 'corrected', 'none')),
    utility_score REAL NOT NULL DEFAULT 0.0 CHECK (utility_score >= 0.0 AND utility_score <= 1.0),
    negative_flag BOOLEAN NOT NULL DEFAULT false,
    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_origin UUID,
    PRIMARY KEY (uu_id, history_id)
) PARTITION BY HASH (uu_id);

ALTER TABLE agent_retrieval.history SET (fillfactor = 90);

CREATE TABLE IF NOT EXISTS agent_retrieval.memory (
    uu_id UUID NOT NULL,
    memory_id UUID NOT NULL,
    memory_type TEXT NOT NULL CHECK (memory_type IN ('fact', 'preference', 'correction', 'constraint', 'procedure', 'schema')),
    content_text TEXT NOT NULL,
    content_embedding vector(1536),
    authority_tier TEXT NOT NULL CHECK (authority_tier IN ('canonical', 'curated', 'derived', 'ephemeral')),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    source_provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    verified_by UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    supersedes UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
    last_accessed_at TIMESTAMPTZ,
    content_hash BYTEA NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (uu_id, memory_id),
    UNIQUE (uu_id, content_hash)
) PARTITION BY HASH (uu_id);

ALTER TABLE agent_retrieval.memory SET (fillfactor = 85);

DO $$
DECLARE
    partition_index INTEGER;
BEGIN
    FOR partition_index IN 0..63 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_retrieval.session_p%1$s PARTITION OF agent_retrieval.session FOR VALUES WITH (modulus 64, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_retrieval.history_p%1$s PARTITION OF agent_retrieval.history FOR VALUES WITH (modulus 64, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_retrieval.memory_p%1$s PARTITION OF agent_retrieval.memory FOR VALUES WITH (modulus 64, remainder %1$s)',
            partition_index
        );
    END LOOP;
END
$$;

CREATE TRIGGER agent_retrieval_touch_users_updated_at
BEFORE UPDATE ON agent_retrieval.users
FOR EACH ROW
EXECUTE FUNCTION agent_retrieval.touch_updated_at();

CREATE TRIGGER agent_retrieval_touch_memory_updated_at
BEFORE UPDATE ON agent_retrieval.memory
FOR EACH ROW
EXECUTE FUNCTION agent_retrieval.touch_updated_at();

CREATE INDEX IF NOT EXISTS idx_retrieval_session_created
    ON agent_retrieval.session (uu_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_session_turn
    ON agent_retrieval.session (uu_id, session_id, turn_index);

CREATE INDEX IF NOT EXISTS idx_retrieval_session_expires
    ON agent_retrieval.session (expires_at);

CREATE INDEX IF NOT EXISTS idx_retrieval_session_text_trgm
    ON agent_retrieval.session USING gin (content_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_retrieval_session_text_fts
    ON agent_retrieval.session USING gin (to_tsvector('simple', content_text));

CREATE INDEX IF NOT EXISTS idx_retrieval_session_vector
    ON agent_retrieval.session USING hnsw (content_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_retrieval_history_utility
    ON agent_retrieval.history (uu_id, utility_score DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_history_created
    ON agent_retrieval.history (uu_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_history_negative
    ON agent_retrieval.history (uu_id, negative_flag);

CREATE INDEX IF NOT EXISTS idx_retrieval_history_outcome
    ON agent_retrieval.history (uu_id, task_outcome);

CREATE INDEX IF NOT EXISTS idx_retrieval_history_tags
    ON agent_retrieval.history USING gin (tags);

CREATE INDEX IF NOT EXISTS idx_retrieval_history_text_trgm
    ON agent_retrieval.history USING gin (query_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_retrieval_history_text_fts
    ON agent_retrieval.history USING gin (to_tsvector('simple', query_text || ' ' || coalesce(response_summary, '')));

CREATE INDEX IF NOT EXISTS idx_retrieval_history_vector
    ON agent_retrieval.history USING hnsw (query_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_retrieval_memory_type
    ON agent_retrieval.memory (uu_id, memory_type);

CREATE INDEX IF NOT EXISTS idx_retrieval_memory_authority
    ON agent_retrieval.memory (uu_id, authority_tier, confidence DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_memory_updated
    ON agent_retrieval.memory (uu_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_memory_expires
    ON agent_retrieval.memory (expires_at);

CREATE INDEX IF NOT EXISTS idx_retrieval_memory_text_trgm
    ON agent_retrieval.memory USING gin (content_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_retrieval_memory_text_fts
    ON agent_retrieval.memory USING gin (to_tsvector('simple', content_text));

CREATE INDEX IF NOT EXISTS idx_retrieval_memory_vector
    ON agent_retrieval.memory USING hnsw (content_embedding vector_cosine_ops);

COMMIT;
