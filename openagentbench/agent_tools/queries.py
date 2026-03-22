"""Parameterized SQL templates for the PostgreSQL-backed agent-tools module."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
from uuid import UUID

from openagentbench.agent_data import dumps

from .models import IdempotencyRecord, QueryTemplate, ToolApprovalTicket, ToolAuditRecord, ToolCacheEntry, ToolDescriptor


def build_upsert_tool_registry(tool: ToolDescriptor) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.tool_registry (
        tool_id,
        version,
        type_class,
        input_schema,
        output_schema,
        error_schema,
        auth_contract,
        timeout_class,
        idempotency_spec,
        mutation_class,
        side_effect_manifest,
        observability_contract,
        health_score,
        status,
        deprecation_notice,
        source_endpoint,
        source_type,
        token_cost_estimate,
        compressed_description,
        contract_tests_hash,
        metadata
    ) VALUES (
        %(tool_id)s,
        %(version)s,
        %(type_class)s,
        %(input_schema)s::jsonb,
        %(output_schema)s::jsonb,
        %(error_schema)s::jsonb,
        %(auth_contract)s::jsonb,
        %(timeout_class)s,
        %(idempotency_spec)s::jsonb,
        %(mutation_class)s,
        %(side_effect_manifest)s::jsonb,
        %(observability_contract)s::jsonb,
        %(health_score)s,
        %(status)s,
        %(deprecation_notice)s,
        %(source_endpoint)s,
        %(source_type)s,
        %(token_cost_estimate)s,
        %(compressed_description)s,
        %(contract_tests_hash)s,
        %(metadata)s::jsonb
    )
    ON CONFLICT (tool_id) DO UPDATE
    SET
        version = EXCLUDED.version,
        type_class = EXCLUDED.type_class,
        input_schema = EXCLUDED.input_schema,
        output_schema = EXCLUDED.output_schema,
        error_schema = EXCLUDED.error_schema,
        auth_contract = EXCLUDED.auth_contract,
        timeout_class = EXCLUDED.timeout_class,
        idempotency_spec = EXCLUDED.idempotency_spec,
        mutation_class = EXCLUDED.mutation_class,
        side_effect_manifest = EXCLUDED.side_effect_manifest,
        observability_contract = EXCLUDED.observability_contract,
        health_score = EXCLUDED.health_score,
        status = EXCLUDED.status,
        deprecation_notice = EXCLUDED.deprecation_notice,
        source_endpoint = EXCLUDED.source_endpoint,
        source_type = EXCLUDED.source_type,
        token_cost_estimate = EXCLUDED.token_cost_estimate,
        compressed_description = EXCLUDED.compressed_description,
        contract_tests_hash = EXCLUDED.contract_tests_hash,
        metadata = EXCLUDED.metadata,
        updated_at = now()
    """
    params = {
        "tool_id": tool.tool_id,
        "version": tool.version,
        "type_class": tool.type_class.value,
        "input_schema": dumps(tool.input_schema),
        "output_schema": dumps(tool.output_schema),
        "error_schema": dumps(tool.error_schema),
        "auth_contract": dumps(asdict(tool.auth_contract)),
        "timeout_class": tool.timeout_class.value,
        "idempotency_spec": dumps(asdict(tool.idempotency_spec)),
        "mutation_class": tool.mutation_class.value,
        "side_effect_manifest": dumps({} if tool.side_effect_manifest is None else asdict(tool.side_effect_manifest)),
        "observability_contract": dumps(asdict(tool.observability_contract)),
        "health_score": tool.health_score,
        "status": tool.status.value,
        "deprecation_notice": tool.deprecation_notice,
        "source_endpoint": tool.source_endpoint,
        "source_type": tool.source_type.value,
        "token_cost_estimate": tool.token_cost_estimate,
        "compressed_description": tool.compressed_description,
        "contract_tests_hash": tool.contract_tests_hash,
        "metadata": dumps(tool.metadata),
    }
    return QueryTemplate(sql=sql, params=params)


def build_insert_idempotency_record(record: IdempotencyRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.tool_idempotency_store (
        idempotency_key,
        tool_id,
        user_id,
        caller_session_id,
        result_envelope,
        created_at,
        expires_at
    ) VALUES (
        %(idempotency_key)s,
        %(tool_id)s,
        %(user_id)s,
        %(caller_session_id)s,
        %(result_envelope)s::jsonb,
        %(created_at)s,
        %(expires_at)s
    )
    """
    params = {
        "idempotency_key": record.key,
        "tool_id": record.tool_id,
        "user_id": record.user_id,
        "caller_session_id": record.session_id,
        "result_envelope": dumps(
            {
                "status": record.response.status.value,
                "success": None if record.response.success is None else asdict(record.response.success),
                "error": None if record.response.error is None else asdict(record.response.error),
                "pending": None if record.response.pending is None else asdict(record.response.pending),
            }
        ),
        "created_at": record.created_at,
        "expires_at": record.expires_at,
    }
    return QueryTemplate(sql=sql, params=params)


def build_lookup_idempotency_record(*, tool_id: str, key: str, user_id: UUID) -> QueryTemplate:
    sql = """
    SELECT *
    FROM agent_data.tool_idempotency_store
    WHERE idempotency_key = %(idempotency_key)s
      AND tool_id = %(tool_id)s
      AND user_id = %(user_id)s
      AND expires_at > now()
    """
    return QueryTemplate(sql=sql, params={"idempotency_key": key, "tool_id": tool_id, "user_id": user_id})


def build_insert_tool_invocation_audit(record: ToolAuditRecord) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.tool_invocation_audit (
        audit_id,
        trace_id,
        tool_id,
        tool_version,
        caller_id,
        agent_id,
        session_id,
        auth_decision,
        status,
        input_hash,
        mutation_class,
        error_code,
        latency_ms,
        compute_cost,
        token_cost,
        side_effects,
        created_at
    ) VALUES (
        %(audit_id)s,
        %(trace_id)s,
        %(tool_id)s,
        %(tool_version)s,
        %(caller_id)s,
        %(agent_id)s,
        %(session_id)s,
        %(auth_decision)s,
        %(status)s,
        %(input_hash)s,
        %(mutation_class)s,
        %(error_code)s,
        %(latency_ms)s,
        %(compute_cost)s,
        %(token_cost)s,
        %(side_effects)s::jsonb,
        %(created_at)s
    )
    """
    params = {
        "audit_id": record.audit_id,
        "trace_id": record.trace_id,
        "tool_id": record.tool_id,
        "tool_version": record.tool_version,
        "caller_id": record.caller_id,
        "agent_id": record.agent_id,
        "session_id": record.session_id,
        "auth_decision": record.auth_decision.value,
        "status": record.status.value,
        "input_hash": record.input_hash,
        "mutation_class": record.mutation_class.value,
        "error_code": record.error_code,
        "latency_ms": record.latency_ms,
        "compute_cost": record.compute_cost,
        "token_cost": record.token_cost,
        "side_effects": dumps(record.side_effects),
        "created_at": record.created_at,
    }
    return QueryTemplate(sql=sql, params=params)


def build_insert_tool_approval_ticket(ticket: ToolApprovalTicket) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.tool_approval_tickets (
        ticket_id,
        tool_id,
        params_redacted,
        requested_by,
        agent_id,
        status,
        resolution_by,
        resolution_at,
        created_at,
        expires_at,
        metadata
    ) VALUES (
        %(ticket_id)s,
        %(tool_id)s,
        %(params_redacted)s::jsonb,
        %(requested_by)s,
        %(agent_id)s,
        %(status)s,
        %(resolution_by)s,
        %(resolution_at)s,
        %(created_at)s,
        %(expires_at)s,
        %(metadata)s::jsonb
    )
    """
    params = {
        "ticket_id": ticket.ticket_id,
        "tool_id": ticket.tool_id,
        "params_redacted": dumps(ticket.params_redacted),
        "requested_by": ticket.requested_by,
        "agent_id": ticket.agent_id,
        "status": ticket.status.value,
        "resolution_by": ticket.resolution_by,
        "resolution_at": ticket.resolution_at,
        "created_at": ticket.created_at,
        "expires_at": ticket.expires_at,
        "metadata": dumps(ticket.metadata),
    }
    return QueryTemplate(sql=sql, params=params)


def build_insert_tool_cache_entry(entry: ToolCacheEntry) -> QueryTemplate:
    sql = """
    INSERT INTO agent_data.tool_result_cache (
        cache_key,
        tool_id,
        user_id,
        result_data,
        created_at,
        expires_at,
        hit_count
    ) VALUES (
        %(cache_key)s,
        %(tool_id)s,
        %(user_id)s,
        %(result_data)s::jsonb,
        %(created_at)s,
        %(expires_at)s,
        %(hit_count)s
    )
    ON CONFLICT (cache_key) DO UPDATE
    SET
        result_data = EXCLUDED.result_data,
        created_at = EXCLUDED.created_at,
        expires_at = EXCLUDED.expires_at,
        hit_count = EXCLUDED.hit_count
    """
    params = {
        "cache_key": entry.cache_key,
        "tool_id": entry.tool_id,
        "user_id": entry.user_id,
        "result_data": dumps(entry.result_data),
        "created_at": entry.created_at,
        "expires_at": entry.expires_at,
        "hit_count": entry.hit_count,
    }
    return QueryTemplate(sql=sql, params=params)


def build_lookup_tool_cache(*, cache_key: str, user_id: UUID) -> QueryTemplate:
    sql = """
    SELECT *
    FROM agent_data.tool_result_cache
    WHERE cache_key = %(cache_key)s
      AND user_id = %(user_id)s
      AND expires_at > now()
    """
    return QueryTemplate(sql=sql, params={"cache_key": cache_key, "user_id": user_id})


def build_invalidate_tool_cache(*, tool_id: str | None = None, user_id: UUID | None = None) -> QueryTemplate:
    sql = """
    DELETE FROM agent_data.tool_result_cache
    WHERE 1 = 1
    """
    params: dict[str, Any] = {}
    if tool_id is not None:
        sql += "\n  AND tool_id = %(tool_id)s"
        params["tool_id"] = tool_id
    if user_id is not None:
        sql += "\n  AND user_id = %(user_id)s"
        params["user_id"] = user_id
    return QueryTemplate(sql=sql, params=params)


__all__ = [
    "build_insert_idempotency_record",
    "build_insert_tool_approval_ticket",
    "build_insert_tool_cache_entry",
    "build_insert_tool_invocation_audit",
    "build_invalidate_tool_cache",
    "build_lookup_idempotency_record",
    "build_lookup_tool_cache",
    "build_upsert_tool_registry",
]
