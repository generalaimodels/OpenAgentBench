"""Fast JSON helpers with deterministic hashing support."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

try:
    import orjson  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    orjson = None


def dumps(value: Any, *, sort_keys: bool = False) -> str:
    """Serialize JSON using the fastest available backend."""
    if orjson is not None:
        option = orjson.OPT_SORT_KEYS if sort_keys else 0
        return orjson.dumps(value, option=option).decode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    )


def dumpb(value: Any, *, sort_keys: bool = False) -> bytes:
    """Serialize to bytes to avoid a second encode step in hash-heavy paths."""
    if orjson is not None:
        option = orjson.OPT_SORT_KEYS if sort_keys else 0
        return orjson.dumps(value, option=option)
    return dumps(value, sort_keys=sort_keys).encode("utf-8")


def loads(payload: str | bytes | bytearray | memoryview) -> Any:
    """Deserialize JSON from text or bytes."""
    if orjson is not None and not isinstance(payload, str):
        return orjson.loads(payload)
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def normalize_text(text: str) -> str:
    """Canonicalize text before hashing or deduplication."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def canonical_json_bytes(value: Any) -> bytes:
    """Emit a stable byte representation for cache keys and digests."""
    return dumpb(value, sort_keys=True)


def sha256_digest(payload: str | bytes | bytearray | memoryview) -> bytes:
    """Return raw SHA-256 bytes to match PostgreSQL BYTEA storage."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    return hashlib.sha256(payload).digest()


def hash_normalized_text(text: str) -> bytes:
    """Hash normalized text so semantically equivalent spacing deduplicates."""
    return sha256_digest(normalize_text(text))
