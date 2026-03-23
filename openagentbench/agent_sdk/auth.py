"""Credential resolution and scope management for the universal agent SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence
from uuid import uuid4

from .enums import AuthType
from .models import AuthCredential, EndpointDescriptor


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class AuthManager:
    credentials_by_scope: dict[tuple[str, AuthType], list[AuthCredential]] = field(default_factory=dict)

    def register_credential(
        self,
        *,
        domain: str,
        auth_type: AuthType,
        token: str,
        scopes: Sequence[str],
        expires_in: timedelta | None = None,
        source: str = "manual",
    ) -> AuthCredential:
        issued_at = _utc_now()
        credential = AuthCredential(
            credential_id=uuid4(),
            auth_type=auth_type,
            token=token,
            scopes=frozenset(scopes),
            issued_at=issued_at,
            expires_at=issued_at + expires_in if expires_in is not None else None,
            source=source,
        )
        self.credentials_by_scope.setdefault((domain.lower(), auth_type), []).append(credential)
        return credential

    def resolve_credential(
        self,
        *,
        endpoint: EndpointDescriptor,
        scopes: Sequence[str],
        auth_type: AuthType | None = None,
    ) -> AuthCredential:
        target_auth = auth_type or endpoint.auth_type
        now = _utc_now()
        domain = endpoint.address.lower()
        candidates = self.credentials_by_scope.get((domain, target_auth), [])
        for credential in candidates:
            if credential.is_expired(at=now):
                continue
            if set(scopes).issubset(credential.scopes):
                return credential.narrow_to(scopes)

        return AuthCredential(
            credential_id=uuid4(),
            auth_type=target_auth,
            token=f"scoped::{domain}::{target_auth.value}",
            scopes=frozenset(scopes),
            issued_at=now,
            expires_at=None,
            source="inferred",
            metadata={"endpoint_id": endpoint.endpoint_id},
        )


__all__ = ["AuthManager"]
