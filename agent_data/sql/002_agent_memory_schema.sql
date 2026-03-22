BEGIN;

ALTER TABLE agent_data.sessions
    ADD COLUMN IF NOT EXISTS summary_version INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_summarized_turn INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS checkpoint_seq INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS checkpoint_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS supported_modalities JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS active_task_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE agent_data.conversation_history
    ADD COLUMN IF NOT EXISTS correction_flag BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS decision_flag BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS segment_boundary BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS message_modality TEXT,
    ADD COLUMN IF NOT EXISTS modality_ref TEXT,
    ADD COLUMN IF NOT EXISTS tool_trace_summary TEXT,
    ADD COLUMN IF NOT EXISTS compression_group_id UUID;

ALTER TABLE agent_data.memory_store_working
    ADD COLUMN IF NOT EXISTS agent_step_id UUID,
    ADD COLUMN IF NOT EXISTS modality TEXT NOT NULL DEFAULT 'text',
    ADD COLUMN IF NOT EXISTS binary_ref TEXT,
    ADD COLUMN IF NOT EXISTS utility_score REAL NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS dependency_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS ttl_remaining INTEGER NOT NULL DEFAULT -1,
    ADD COLUMN IF NOT EXISTS carry_forward BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS source_trace_id UUID;

ALTER TABLE agent_data.memory_store
    ADD COLUMN IF NOT EXISTS memory_type SMALLINT,
    ADD COLUMN IF NOT EXISTS authority_tier SMALLINT,
    ADD COLUMN IF NOT EXISTS modality TEXT NOT NULL DEFAULT 'text',
    ADD COLUMN IF NOT EXISTS modality_ref TEXT,
    ADD COLUMN IF NOT EXISTS promotion_source TEXT,
    ADD COLUMN IF NOT EXISTS promotion_score REAL,
    ADD COLUMN IF NOT EXISTS evaluation_record JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS action_trace JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS outcome_status TEXT,
    ADD COLUMN IF NOT EXISTS procedure_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS procedure_status TEXT,
    ADD COLUMN IF NOT EXISTS procedure_version TEXT,
    ADD COLUMN IF NOT EXISTS test_suite_ref TEXT,
    ADD COLUMN IF NOT EXISTS success_rate REAL,
    ADD COLUMN IF NOT EXISTS conflict_status TEXT NOT NULL DEFAULT 'none',
    ADD COLUMN IF NOT EXISTS staleness_flags INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS promotion_chain JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS agent_data.session_checkpoints (
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    checkpoint_id UUID NOT NULL,
    checkpoint_seq INTEGER NOT NULL CHECK (checkpoint_seq >= 0),
    summary_text TEXT NOT NULL,
    summary_version INTEGER NOT NULL CHECK (summary_version >= 0),
    turn_count INTEGER NOT NULL CHECK (turn_count >= 0),
    working_item_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, checkpoint_id)
) PARTITION BY HASH (user_id);

CREATE TABLE IF NOT EXISTS agent_data.memory_promotion_log (
    user_id UUID NOT NULL,
    log_id UUID NOT NULL,
    source_layer SMALLINT NOT NULL,
    target_layer SMALLINT,
    source_item_id UUID NOT NULL,
    target_item_id UUID,
    promotion_score REAL,
    action TEXT NOT NULL,
    rejection_reason TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, log_id)
) PARTITION BY HASH (user_id);

CREATE TABLE IF NOT EXISTS agent_data.memory_audit_log (
    user_id UUID NOT NULL,
    audit_id UUID NOT NULL,
    operation TEXT NOT NULL,
    layer SMALLINT,
    item_id UUID,
    caller_id UUID,
    session_id UUID,
    result TEXT NOT NULL,
    latency_ms INTEGER,
    token_delta INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, audit_id)
) PARTITION BY HASH (user_id);

CREATE TABLE IF NOT EXISTS agent_data.memory_conflict_queue (
    user_id UUID NOT NULL,
    conflict_id UUID NOT NULL,
    existing_memory_id UUID,
    proposed_memory_id UUID,
    existing_authority SMALLINT,
    proposed_authority SMALLINT,
    resolution_status TEXT NOT NULL DEFAULT 'pending',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, conflict_id)
) PARTITION BY HASH (user_id);

CREATE UNLOGGED TABLE IF NOT EXISTS agent_data.memory_cache (
    cache_key TEXT PRIMARY KEY,
    user_id UUID NOT NULL,
    layer SMALLINT NOT NULL,
    payload JSONB NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0 CHECK (hit_count >= 0),
    embedding_bucket TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);

DO $$
DECLARE
    partition_index INTEGER;
BEGIN
    FOR partition_index IN 0..255 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.session_checkpoints_p%1$s PARTITION OF agent_data.session_checkpoints FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.memory_promotion_log_p%1$s PARTITION OF agent_data.memory_promotion_log FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.memory_audit_log_p%1$s PARTITION OF agent_data.memory_audit_log FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS agent_data.memory_conflict_queue_p%1$s PARTITION OF agent_data.memory_conflict_queue FOR VALUES WITH (modulus 256, remainder %1$s)',
            partition_index
        );
    END LOOP;
END
$$;

CREATE INDEX IF NOT EXISTS idx_history_correction_flag
    ON agent_data.conversation_history (user_id, session_id, created_at DESC)
    WHERE correction_flag = true;

CREATE INDEX IF NOT EXISTS idx_history_decision_flag
    ON agent_data.conversation_history (user_id, session_id, created_at DESC)
    WHERE decision_flag = true;

CREATE INDEX IF NOT EXISTS idx_history_segment_boundary
    ON agent_data.conversation_history (user_id, session_id, created_at DESC)
    WHERE segment_boundary = true;

CREATE INDEX IF NOT EXISTS idx_memory_store_scope_tier
    ON agent_data.memory_store (user_id, memory_scope, memory_tier, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_store_conflict_status
    ON agent_data.memory_store (user_id, conflict_status)
    WHERE conflict_status <> 'none';

CREATE INDEX IF NOT EXISTS idx_memory_store_active_procedures
    ON agent_data.memory_store (user_id, updated_at DESC)
    WHERE memory_tier = 4 AND procedure_status = 'active';

CREATE INDEX IF NOT EXISTS idx_session_checkpoints_session_seq
    ON agent_data.session_checkpoints (user_id, session_id, checkpoint_seq DESC);

CREATE INDEX IF NOT EXISTS idx_memory_audit_layer_time
    ON agent_data.memory_audit_log (user_id, layer, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_cache_user_layer
    ON agent_data.memory_cache (user_id, layer, expires_at);

COMMIT;
