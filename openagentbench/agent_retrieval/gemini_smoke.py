"""Gemini compatibility smoke path for the retrieval module.

This script keeps all work inside the retrieval submodule:
- validates endpoint payload compatibility
- calls a small Gemini model via REST
- optionally writes a session/history/memory sample into PostgreSQL
- reads the rows back through the retrieval SQL templates
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from openagentbench.agent_data.json_codec import hash_normalized_text

from .endpoint_compat import (
    assert_endpoint_payload_compatibility,
    build_endpoint_compatibility_report,
    build_gemini_count_tokens_request,
    build_gemini_generate_content_request,
    extract_gemini_text,
)
from .enums import AuthorityTier, HumanFeedback, MemoryType, ProtocolType, QueryType, Role, TaskOutcome
from .models import FragmentLocator, HistoryEntry, HistoryEvidence, MemoryEntry, SessionTurn
from .queries import (
    build_exact_memory_retrieval,
    build_insert_history_entry,
    build_insert_session_turn,
    build_load_memory_summary,
    build_load_session_context,
    build_touch_memory_access,
    build_upsert_memory_entry,
)
from .runtime import read_schema_sql
from .scoring import classify_query
from .enums import SourceTable

try:  # pragma: no cover - optional runtime dependency
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional runtime dependency
    psycopg = None
    dict_row = None


STABLE_NAMESPACE = UUID("28d5c8de-8df4-42db-b79e-bb7f9e3d9f4e")


def _stable_uuid(*parts: object) -> UUID:
    return uuid5(STABLE_NAMESPACE, "::".join(str(part) for part in parts))


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _request_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    last_error: RuntimeError | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - live only
            body = exc.read().decode("utf-8", errors="replace")
            retriable = exc.code in {429, 500, 502, 503, 504}
            last_error = RuntimeError(f"Gemini request failed with {exc.code}: {body}")
            if not retriable or attempt == 2:
                raise last_error from exc
            time.sleep(1.5 * (attempt + 1))
    if last_error is not None:  # pragma: no cover - defensive
        raise last_error
    raise RuntimeError("Gemini request failed without a captured error")


@dataclass(slots=True, frozen=True)
class GeminiSmokeResult:
    model: str
    prompt: str
    response_text: str
    input_token_count: int | None
    query_type: QueryType
    db_persisted: bool
    schema_name: str | None


def _rewrite_schema(sql: str, schema_name: str) -> str:
    return sql.replace("CREATE SCHEMA IF NOT EXISTS agent_retrieval;", f"CREATE SCHEMA IF NOT EXISTS {schema_name};").replace(
        "agent_retrieval.",
        f"{schema_name}.",
    )


def _execute_template(connection: Any, schema_name: str, template: Any) -> list[dict[str, Any]]:
    sql = template.sql.replace("agent_retrieval.", f"{schema_name}.")
    with connection.cursor() as cursor:
        cursor.execute(sql, template.params)
        if cursor.description is None:
            return []
        return list(cursor.fetchall())


def _persist_sample_rows(
    *,
    database_url: str,
    prompt: str,
    response_text: str,
    model: str,
) -> str:
    if psycopg is None or dict_row is None:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("psycopg is not installed in the active environment")

    now = datetime.now(timezone.utc)
    run_tag = now.strftime("%Y%m%dT%H%M%SZ")
    schema_name = f"agent_retrieval_smoke_{run_tag.lower()}"
    uu_id = _stable_uuid("gemini", "smoke", "user")
    session_id = _stable_uuid("gemini", "smoke", "session")
    history_id = _stable_uuid("gemini", "smoke", "history")
    memory_id = _stable_uuid("gemini", "smoke", "memory")

    session_turn = SessionTurn(
        session_id=session_id,
        uu_id=uu_id,
        turn_index=0,
        role=Role.USER,
        content_text=prompt,
        created_at=now,
        tokens_used=max(len(prompt.split()), 1),
        metadata={
            "test_suite": "agent_retrieval_gemini_smoke",
            "provider": "gemini",
            "model": model,
            "protocol_type": ProtocolType.HTTP.value,
        },
        expires_at=now + timedelta(days=1),
    )
    history_entry = HistoryEntry(
        history_id=history_id,
        uu_id=uu_id,
        query_text=prompt,
        query_embedding=None,
        response_summary=response_text,
        evidence_used=(
            HistoryEvidence(
                locator=FragmentLocator(source_table=SourceTable.SESSION, chunk_id=session_turn.turn_id()),
                utility_score=0.9,
            ),
        ),
        task_outcome=TaskOutcome.SUCCESS,
        human_feedback=HumanFeedback.APPROVED,
        utility_score=0.95,
        negative_flag=False,
        tags=("smoke", "gemini", "endpoint"),
        metadata={"provider": "gemini", "model": model, "test_suite": "agent_retrieval_gemini_smoke"},
        created_at=now,
        session_origin=session_id,
    )
    memory_entry = MemoryEntry(
        memory_id=memory_id,
        uu_id=uu_id,
        memory_type=MemoryType.FACT,
        content_text=response_text,
        content_embedding=None,
        authority_tier=AuthorityTier.CURATED,
        confidence=0.88,
        source_provenance={"provider": "gemini", "model": model, "prompt": prompt},
        verified_by=(),
        supersedes=(),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=7),
        access_count=0,
        last_accessed_at=None,
        content_hash=hash_normalized_text(response_text),
        metadata={"test_suite": "agent_retrieval_gemini_smoke"},
    )

    ddl = _rewrite_schema(read_schema_sql(), schema_name)
    with psycopg.connect(database_url, autocommit=True, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            cursor.execute(
                f"""
                INSERT INTO {schema_name}.users (uu_id, status, metadata)
                VALUES (%s, 'active', %s::jsonb)
                ON CONFLICT (uu_id) DO UPDATE
                SET status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata
                """,
                (uu_id, json.dumps({"test_suite": "agent_retrieval_gemini_smoke"})),
            )
        _execute_template(connection, schema_name, build_insert_session_turn(session_turn))
        _execute_template(connection, schema_name, build_insert_history_entry(history_entry))
        _execute_template(connection, schema_name, build_upsert_memory_entry(memory_entry))

        session_rows = _execute_template(
            connection,
            schema_name,
            build_load_session_context(uu_id=uu_id, session_id=session_id, limit=5),
        )
        if not session_rows or session_rows[0]["content_text"] != prompt:
            raise RuntimeError("session smoke row did not round-trip correctly")

        memory_rows = _execute_template(
            connection,
            schema_name,
            build_load_memory_summary(uu_id=uu_id, limit=5),
        )
        if not memory_rows or memory_rows[0]["content_text"] != response_text:
            raise RuntimeError("memory smoke row did not round-trip correctly")

        exact_rows = _execute_template(
            connection,
            schema_name,
            build_exact_memory_retrieval(
                uu_id=uu_id,
                query_text=response_text.split()[0],
                temporal_scope=None,
                limit=3,
            ),
        )
        if not exact_rows:
            raise RuntimeError("exact memory retrieval returned no rows")

        _execute_template(
            connection,
            schema_name,
            build_touch_memory_access(uu_id=uu_id, memory_ids=(memory_id,), accessed_at=now + timedelta(seconds=1)),
        )
    return schema_name


def run_gemini_smoke(
    *,
    env_path: str | os.PathLike[str] = ".env",
    model: str | None = None,
) -> GeminiSmokeResult:
    _load_dotenv(Path(env_path))
    assert_endpoint_payload_compatibility(build_endpoint_compatibility_report())

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for the Gemini retrieval smoke test")

    selected_model = model or os.getenv("GEMINI_SMALL_MODEL", "gemini-2.5-flash-lite")
    system_instruction = (
        "You are a retrieval compatibility smoke test. "
        "Return only the final answer, keep it deterministic, and do not add explanations."
    )
    prompt = "Answer exactly with: RETRIEVAL_OK PostgreSQL"

    count_tokens_url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{urllib.parse.quote(selected_model, safe='')}:countTokens?key={urllib.parse.quote(api_key, safe='')}"
    )
    generate_url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{urllib.parse.quote(selected_model, safe='')}:generateContent?key={urllib.parse.quote(api_key, safe='')}"
    )

    token_response = _request_json(
        count_tokens_url,
        build_gemini_count_tokens_request(system_instruction=system_instruction, user_text=prompt),
    )
    generation_response = _request_json(
        generate_url,
        build_gemini_generate_content_request(
            system_instruction=system_instruction,
            user_text=prompt,
            temperature=0.0,
            max_output_tokens=96,
        ),
    )
    response_text = extract_gemini_text(generation_response)
    query_type = classify_query(prompt, "", turn_count=0).type

    database_url = os.getenv("AGENT_RETRIEVAL_DATABASE_URL") or os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    schema_name: str | None = None
    if database_url:
        schema_name = _persist_sample_rows(
            database_url=database_url,
            prompt=prompt,
            response_text=response_text,
            model=selected_model,
        )

    return GeminiSmokeResult(
        model=selected_model,
        prompt=prompt,
        response_text=response_text,
        input_token_count=token_response.get("totalTokens"),
        query_type=query_type,
        db_persisted=schema_name is not None,
        schema_name=schema_name,
    )


def main() -> int:
    result = run_gemini_smoke()
    print(json.dumps(
        {
            "model": result.model,
            "query_type": result.query_type.value,
            "input_token_count": result.input_token_count,
            "response_text": result.response_text,
            "db_persisted": result.db_persisted,
            "schema_name": result.schema_name,
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover - executable smoke entrypoint
    raise SystemExit(main())
