"""SQL query builders for query-resolution cache and audit persistence."""

from __future__ import annotations

from .models import QueryTemplate


def build_lookup_query_cache() -> QueryTemplate:
    return QueryTemplate(
        sql=(
            "SELECT cache_key, plan_data, expires_at, hit_count "
            "FROM agent_query_resolution_cache "
            "WHERE cache_key = %(cache_key)s AND expires_at > NOW() "
            "LIMIT 1"
        ),
        params={"cache_key": None},
    )


def build_upsert_query_cache() -> QueryTemplate:
    return QueryTemplate(
        sql=(
            "INSERT INTO agent_query_resolution_cache "
            "(cache_key, user_id, session_id, plan_data, expires_at, hit_count) "
            "VALUES (%(cache_key)s, %(user_id)s, %(session_id)s, %(plan_data)s, %(expires_at)s, %(hit_count)s) "
            "ON CONFLICT (cache_key) DO UPDATE SET "
            "plan_data = EXCLUDED.plan_data, expires_at = EXCLUDED.expires_at, hit_count = EXCLUDED.hit_count"
        ),
        params={
            "cache_key": None,
            "user_id": None,
            "session_id": None,
            "plan_data": None,
            "expires_at": None,
            "hit_count": 0,
        },
    )


def build_insert_query_audit() -> QueryTemplate:
    return QueryTemplate(
        sql=(
            "INSERT INTO agent_query_resolution_audit "
            "(audit_id, user_id, session_id, cache_key, ambiguity_level, subquery_count, cache_hit, latency_ms, created_at) "
            "VALUES (%(audit_id)s, %(user_id)s, %(session_id)s, %(cache_key)s, %(ambiguity_level)s, "
            "%(subquery_count)s, %(cache_hit)s, %(latency_ms)s, %(created_at)s)"
        ),
        params={
            "audit_id": None,
            "user_id": None,
            "session_id": None,
            "cache_key": None,
            "ambiguity_level": None,
            "subquery_count": 0,
            "cache_hit": False,
            "latency_ms": 0,
            "created_at": None,
        },
    )


__all__ = [
    "build_insert_query_audit",
    "build_lookup_query_cache",
    "build_upsert_query_cache",
]
