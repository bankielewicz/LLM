"""Pairing, sessions, and ordered browser-request authentication.

Credentials in this module are intentionally memory-only.  The service owns one
``AuthManager`` per process; replacing that manager (or calling ``restart``)
invalidates every session and bootstrap grant.
"""

from __future__ import annotations

import base64
import hmac
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .errors import ApiError


EXPECTED_HOST = "127.0.0.1:8765"
EXPECTED_ORIGIN = "http://127.0.0.1:8765"
BOOTSTRAP_LIFETIME_SECONDS = 120
SESSION_IDLE_SECONDS = 2 * 60 * 60
SESSION_ABSOLUTE_SECONDS = 12 * 60 * 60
MAX_SESSIONS = 32
_SECRET_BYTES = 32
_DUMMY_SECRET = "A" * 43
_MUTATION_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_ALLOWED_FETCH_SITES = frozenset({"same-origin", "none"})


def _timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _secret_from_bytes(raw: bytes) -> str:
    if len(raw) != _SECRET_BYTES:
        raise RuntimeError("credential entropy source returned the wrong byte count")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _uuid_from_bytes(raw: bytes) -> str:
    if len(raw) != 16:
        raise RuntimeError("identifier entropy source returned the wrong byte count")
    return str(uuid.UUID(bytes=raw, version=4))


def generate_instance_id(
    rng: Callable[[int], bytes] = secrets.token_bytes,
) -> str:
    """Return a CSPRNG UUID suitable for one service instance."""

    return _uuid_from_bytes(rng(16))


def _canonical_uuid(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        return False
    return str(parsed) == value and parsed.version in {1, 2, 3, 4, 5}


@dataclass(frozen=True)
class BootstrapGrant:
    """One memory-only, single-use bootstrap credential."""

    secret: str = field(repr=False)
    issued_at: float
    expires_at: float

    @property
    def url(self) -> str:
        return f"{EXPECTED_ORIGIN}/#/connect/{self.secret}"


@dataclass(frozen=True)
class SessionCredentials:
    """The exact public ``Session`` response plus its private stable subject."""

    session_id: str
    access_token: str = field(repr=False)
    csrf_token: str = field(repr=False)
    instance_id: str
    idle_expires_at: float
    absolute_expires_at: float

    def as_dict(self) -> dict[str, str]:
        """Serialize only fields frozen by the public OpenAPI Session schema."""

        return {
            "access_token": self.access_token,
            "csrf_token": self.csrf_token,
            "instance_id": self.instance_id,
            "idle_expires_at": _timestamp(self.idle_expires_at),
            "absolute_expires_at": _timestamp(self.absolute_expires_at),
        }


@dataclass(frozen=True)
class SessionIdentity:
    """Authenticated request identity; persist ``session_id``, never its token."""

    session_id: str
    access_token: str = field(repr=False)
    instance_id: str
    created_at: float
    idle_expires_at: float
    absolute_expires_at: float


@dataclass
class _SessionRecord:
    session_id: str
    access_token: str = field(repr=False)
    csrf_token: str = field(repr=False)
    created_at: float
    idle_expires_at: float
    absolute_expires_at: float

    def expired(self, now: float) -> bool:
        return now >= self.idle_expires_at or now >= self.absolute_expires_at

    def identity(self, instance_id: str) -> SessionIdentity:
        return SessionIdentity(
            session_id=self.session_id,
            access_token=self.access_token,
            instance_id=instance_id,
            created_at=self.created_at,
            idle_expires_at=self.idle_expires_at,
            absolute_expires_at=self.absolute_expires_at,
        )


def _headers_lower(headers: Mapping[str, str]) -> dict[str, str]:
    """Make header lookup case-insensitive and make duplicates fail closed."""

    lowered: dict[str, str] = {}
    for name, value in headers.items():
        key = str(name).lower()
        text = str(value)
        if key in lowered:
            lowered[key] = f"{lowered[key]},{text}"
        else:
            lowered[key] = text
    return lowered


def _bearer_token(value: str | None) -> str | None:
    if value is None:
        return None
    pieces = value.split(" ")
    if len(pieces) != 2 or pieces[0].casefold() != "bearer" or not pieces[1]:
        return None
    return pieces[1]


class AuthManager:
    """Own instance-local bootstrap grants and browser sessions.

    ``authorize`` implements the RUN-003 order through CSRF.  The HTTP facade
    performs content-type and transport-size checks only after this method
    returns, then parses and validates the body.
    """

    def __init__(
        self,
        instance_id: str,
        *,
        clock: Callable[[], float] = time.time,
        rng: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        if not _canonical_uuid(instance_id):
            raise ValueError("instance_id must be a canonical lowercase UUID")
        self._instance_id = instance_id
        self._clock = clock
        self._rng = rng
        self._bootstrap: BootstrapGrant | None = None
        self._sessions: list[_SessionRecord] = []
        self._lock = threading.RLock()

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def _secret(self) -> str:
        return _secret_from_bytes(self._rng(_SECRET_BYTES))

    def _uuid(self) -> str:
        return _uuid_from_bytes(self._rng(16))

    def issue_bootstrap(self) -> BootstrapGrant:
        """Replace any unconsumed grant with a fresh 120-second secret."""

        with self._lock:
            now = float(self._clock())
            grant = BootstrapGrant(
                secret=self._secret(),
                issued_at=now,
                expires_at=now + BOOTSTRAP_LIFETIME_SECONDS,
            )
            self._bootstrap = grant
            return grant

    def exchange_bootstrap(self, supplied_secret: str) -> SessionCredentials:
        """Consume a valid grant and create independent access/CSRF tokens."""

        with self._lock:
            now = float(self._clock())
            grant = self._bootstrap
            candidate = supplied_secret if isinstance(supplied_secret, str) else ""
            expected = grant.secret if grant is not None else _DUMMY_SECRET
            matches = hmac.compare_digest(candidate, expected)
            if grant is None or not matches or now >= grant.expires_at:
                if grant is not None and now >= grant.expires_at:
                    self._bootstrap = None
                raise ApiError(
                    "AUTH_REQUIRED",
                    "The pairing secret is invalid or no longer available.",
                )

            if self.active_session_count(now=now) >= MAX_SESSIONS:
                raise ApiError(
                    "REGISTRY_LIMIT",
                    "The session limit for this companion instance was reached.",
                    reason_code="SESSION_LIMIT",
                )

            created_at = now
            absolute_expires_at = created_at + SESSION_ABSOLUTE_SECONDS
            idle_expires_at = min(
                created_at + SESSION_IDLE_SECONDS,
                absolute_expires_at,
            )
            record = _SessionRecord(
                session_id=self._uuid(),
                access_token=self._secret(),
                csrf_token=self._secret(),
                created_at=created_at,
                idle_expires_at=idle_expires_at,
                absolute_expires_at=absolute_expires_at,
            )
            self._sessions.append(record)
            self._bootstrap = None
            return SessionCredentials(
                session_id=record.session_id,
                access_token=record.access_token,
                csrf_token=record.csrf_token,
                instance_id=self._instance_id,
                idle_expires_at=record.idle_expires_at,
                absolute_expires_at=record.absolute_expires_at,
            )

    def active_session_count(self, *, now: float | None = None) -> int:
        with self._lock:
            instant = float(self._clock()) if now is None else float(now)
            return sum(not record.expired(instant) for record in self._sessions)

    def _find_session(self, token: str) -> _SessionRecord | None:
        match: _SessionRecord | None = None
        # There are at most 32 live sessions.  Iterate through all retained
        # records so a credential comparison does not return early.
        for record in self._sessions:
            if hmac.compare_digest(token, record.access_token):
                match = record
        return match

    def authorize(
        self,
        method: str,
        headers: Mapping[str, str],
        *,
        session_exchange: bool = False,
        mutation: bool | None = None,
        refresh_idle: bool = True,
    ) -> SessionIdentity | None:
        """Apply Host through CSRF checks in the fixed RUN-003 order.

        Session exchange has no bearer session, so it returns ``None`` after
        Host/OPTIONS/Origin succeed.  ``refresh_idle=False`` is available for
        events emitted after an already-authorized stream connection; stream
        heartbeats themselves must never call this method.
        """

        verb = method.upper()
        request_headers = _headers_lower(headers)
        is_mutation = verb in _MUTATION_METHODS if mutation is None else mutation

        if request_headers.get("host") != EXPECTED_HOST:
            raise ApiError(
                "ORIGIN_REJECTED",
                "The request Host is not permitted.",
                reason_code="HOST_MISMATCH",
            )
        if verb == "OPTIONS":
            raise ApiError(
                "ORIGIN_REJECTED",
                "Browser preflight requests are not permitted.",
                reason_code="PREFLIGHT_REJECTED",
            )

        origin = request_headers.get("origin")
        if origin is not None and origin != EXPECTED_ORIGIN:
            raise ApiError(
                "ORIGIN_REJECTED",
                "The request Origin is not permitted.",
                reason_code="ORIGIN_MISMATCH",
            )
        if origin is None and (is_mutation or session_exchange):
            raise ApiError(
                "ORIGIN_REJECTED",
                "This request requires a same-origin Origin header.",
                reason_code="ORIGIN_MISSING",
            )

        if verb == "GET" and request_headers.get("sec-fetch-site") not in _ALLOWED_FETCH_SITES:
            raise ApiError(
                "ORIGIN_REJECTED",
                "The request fetch site is not permitted.",
                reason_code="FETCH_SITE_REJECTED",
            )

        if session_exchange:
            return None

        token = _bearer_token(request_headers.get("authorization"))
        with self._lock:
            record = self._find_session(token or "")
            if record is None:
                raise ApiError(
                    "AUTH_REQUIRED",
                    "A valid companion session is required.",
                )

            now = float(self._clock())
            if record.expired(now):
                raise ApiError(
                    "SESSION_EXPIRED",
                    "The companion session expired.",
                )

            if is_mutation:
                supplied_csrf = request_headers.get("x-llmf-csrf", "")
                if not hmac.compare_digest(supplied_csrf, record.csrf_token):
                    raise ApiError(
                        "CSRF_REJECTED",
                        "The request CSRF token is missing or invalid.",
                    )

            if refresh_idle:
                record.idle_expires_at = min(
                    now + SESSION_IDLE_SECONDS,
                    record.absolute_expires_at,
                )
            return record.identity(self._instance_id)

    def revoke(self, access_token: str) -> bool:
        """Revoke one token immediately; a later use is AUTH_REQUIRED."""

        with self._lock:
            match = self._find_session(access_token)
            if match is None:
                return False
            self._sessions.remove(match)
            return True

    def revoke_identity(self, identity: SessionIdentity) -> bool:
        return self.revoke(identity.access_token)

    def restart(self, instance_id: str | None = None) -> None:
        """Invalidate every memory-only credential for service restart."""

        with self._lock:
            self._bootstrap = None
            self._sessions.clear()
            if instance_id is not None:
                if not _canonical_uuid(instance_id):
                    raise ValueError("instance_id must be a canonical lowercase UUID")
                self._instance_id = instance_id


__all__ = [
    "AuthManager",
    "BootstrapGrant",
    "SessionCredentials",
    "SessionIdentity",
    "BOOTSTRAP_LIFETIME_SECONDS",
    "SESSION_IDLE_SECONDS",
    "SESSION_ABSOLUTE_SECONDS",
    "MAX_SESSIONS",
    "EXPECTED_HOST",
    "EXPECTED_ORIGIN",
    "generate_instance_id",
]
