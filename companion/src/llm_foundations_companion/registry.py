"""Confined content-addressed artifacts, capacity reservations, and registry reads."""

from __future__ import annotations

import base64
import errno
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import sqlite3
import stat
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol

from .database import Database, utc_now
from .errors import ApiError
from .platform_security import (
    _fsync_directory,
    _reject_linked_components,
    ensure_private_directory,
    is_reparse_point,
    StorageSecurityError,
)
from .schema import canonical_json, load_document


ROOT_QUOTA_BYTES = 50 * 1024 * 1024 * 1024
FREE_SPACE_RESERVE_BYTES = 1024 * 1024 * 1024
_U64_MAX = (1 << 64) - 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ORIGINS = frozenset(
    {"locally_created", "imported", "legacy_imported", "shipped_fixture"}
)
_PREVIEW_POLICIES = frozenset({"text", "metadata_only", "download_only"})
_ROW_LIMITS = {
    "artifact_rows": ("artifacts", 100_000),
    "dataset_rows": ("datasets", 1_000),
    "run_rows": ("runs", 20_000),
    "model_rows": ("models", 512),
    "checkpoint_rows": ("checkpoints", 5_000),
}


class _IdempotencyCommit(Protocol):
    def record_success(
        self,
        connection: sqlite3.Connection,
        *,
        status_code: int,
        response_body: bytes | Mapping[str, Any],
    ) -> bytes: ...


@dataclass(frozen=True)
class StagedArtifact:
    staging_owner: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class RecoveryReport:
    storage_writable: bool
    verified_artifacts: int
    quarantined_entries: int
    failed_check: str | None


def _uuid_text(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a lowercase UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a lowercase UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase UUID")
    return value


def _nonnegative_u64(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= _U64_MAX):
        raise ValueError(f"{label} must be an unsigned 64-bit integer")
    return value


def _semantic_cursor_error() -> ApiError:
    return ApiError(
        "VALIDATION_FAILED",
        "The cursor is not valid for this list.",
        reason_code="SEMANTIC_INVALID",
        field_errors=[{"field_path": "cursor", "message": "Cursor is invalid."}],
    )


class CursorCodec:
    """Authenticated opaque list cursors bound to instance, filters, and order."""

    def __init__(self, instance_id: str, key: bytes) -> None:
        self._instance_id = _uuid_text(instance_id, "instance_id")
        if not isinstance(key, bytes) or len(key) < 32:
            raise ValueError("cursor_key must contain at least 32 bytes")
        self._key = key

    def encode(
        self,
        *,
        schema: str,
        order: str,
        filters: Mapping[str, Any],
        created_at: str,
        item_id: str,
    ) -> str:
        payload = canonical_json(
            {
                "created_at": created_at,
                "filters": dict(filters),
                "instance_id": self._instance_id,
                "item_id": item_id,
                "order": order,
                "schema": schema,
                "version": 1,
            }
        )
        signature = hmac.digest(self._key, payload, "sha256")
        return base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode("ascii")

    def decode(
        self,
        cursor: str,
        *,
        schema: str,
        order: str,
        filters: Mapping[str, Any],
    ) -> tuple[str, str]:
        try:
            if not isinstance(cursor, str) or not cursor or len(cursor) > 4_096:
                raise ValueError
            raw = cursor.encode("ascii")
            decoded = base64.b64decode(
                raw + b"=" * (-len(raw) % 4), altchars=b"-_", validate=True
            )
            if base64.urlsafe_b64encode(decoded).rstrip(b"=") != raw:
                raise ValueError
            if len(decoded) <= 32:
                raise ValueError
            payload, signature = decoded[:-32], decoded[-32:]
            if not hmac.compare_digest(signature, hmac.digest(self._key, payload, "sha256")):
                raise ValueError
            value = json.loads(payload.decode("utf-8"))
            expected = {
                "created_at",
                "filters",
                "instance_id",
                "item_id",
                "order",
                "schema",
                "version",
            }
            if not isinstance(value, dict) or set(value) != expected:
                raise ValueError
            if (
                value["version"] != 1
                or value["instance_id"] != self._instance_id
                or value["schema"] != schema
                or value["order"] != order
                or value["filters"] != dict(filters)
                or not isinstance(value["created_at"], str)
                or not isinstance(value["item_id"], str)
            ):
                raise ValueError
            return value["created_at"], value["item_id"]
        except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise _semantic_cursor_error() from exc


class Registry:
    """Sole filesystem/SQLite join for immutable registered artifacts."""

    def __init__(
        self,
        db: Database,
        root: Path,
        instance_id: str,
        cursor_key: bytes,
    ) -> None:
        candidate = Path(root).absolute().resolve(strict=True)
        if candidate != db.root:
            raise ValueError("registry root must equal the database storage root")
        self.db = db
        self.root = db.root
        self.instance_id = _uuid_text(instance_id, "instance_id")
        self.cursor_codec = CursorCodec(self.instance_id, bytes(cursor_key))
        self._sealed_sha256 = self._load_sealed_sha256()

    @staticmethod
    def _load_sealed_sha256() -> frozenset[str]:
        document = load_document("materialized-manifest.json")
        rows = document.get("files")
        if document.get("format") != "llm-foundations-materialized-fixtures-v1" or not isinstance(rows, list):
            raise RuntimeError("bundled materialized fixture registry is malformed")
        paths: set[str] = set()
        sealed: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "records", "sha256", "utf8_bytes"}:
                raise RuntimeError("bundled materialized fixture row is malformed")
            path = row["path"]
            digest = row["sha256"]
            if not isinstance(path, str) or path in paths or not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise RuntimeError("bundled materialized fixture row is invalid or duplicated")
            paths.add(path)
            if path == "capstone-support-v1/sealed_test.jsonl":
                sealed.add(digest)
        if len(sealed) != 1:
            raise RuntimeError("bundled sealed fixture identity is missing or ambiguous")
        return frozenset(sealed)

    def allocate_artifact_id(self) -> str:
        return str(uuid.uuid4())

    def is_sealed_digest(self, sha256: str) -> bool:
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return sha256 in self._sealed_sha256

    def _root_bytes(self) -> int:
        total = 0
        try:
            for directory, names, files in os.walk(self.root, followlinks=False):
                base = Path(directory)
                for name in names:
                    if is_reparse_point(base / name):
                        raise OSError("linked directory inside storage root")
                for name in files:
                    path = base / name
                    if is_reparse_point(path):
                        raise OSError("linked file inside storage root")
                    info = os.lstat(path)
                    if stat.S_ISREG(info.st_mode):
                        total += info.st_size
                        if total > _U64_MAX:
                            raise OverflowError
            return total
        except (OSError, OverflowError) as exc:
            raise ApiError("STORAGE_UNAVAILABLE", "Storage usage cannot be verified.") from exc

    @staticmethod
    def _table_count(connection: sqlite3.Connection, table: str) -> int:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if exists is None:
            return 0
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def reserve_capacity_in(
        self,
        connection: sqlite3.Connection,
        *,
        owner_kind: str,
        owner_id: str,
        byte_count: int,
        artifact_rows: int = 0,
        dataset_rows: int = 0,
        run_rows: int = 0,
        model_rows: int = 0,
        checkpoint_rows: int = 0,
        reservation_id: str | None = None,
    ) -> str:
        if owner_kind not in {"job", "upload"}:
            raise ValueError("owner_kind must be job or upload")
        _uuid_text(owner_id, "owner_id")
        requested = {
            "byte_count": _nonnegative_u64(byte_count, "byte_count"),
            "artifact_rows": _nonnegative_u64(artifact_rows, "artifact_rows"),
            "dataset_rows": _nonnegative_u64(dataset_rows, "dataset_rows"),
            "run_rows": _nonnegative_u64(run_rows, "run_rows"),
            "model_rows": _nonnegative_u64(model_rows, "model_rows"),
            "checkpoint_rows": _nonnegative_u64(checkpoint_rows, "checkpoint_rows"),
        }
        held = connection.execute(
            """
            SELECT COALESCE(SUM(byte_count), 0), COALESCE(SUM(artifact_rows), 0),
                   COALESCE(SUM(dataset_rows), 0), COALESCE(SUM(run_rows), 0),
                   COALESCE(SUM(model_rows), 0), COALESCE(SUM(checkpoint_rows), 0)
            FROM reservations WHERE state = 'held'
            """
        ).fetchone()
        held_values = dict(zip(requested, (int(value) for value in held), strict=True))
        if self._root_bytes() + held_values["byte_count"] + requested["byte_count"] > ROOT_QUOTA_BYTES:
            raise ApiError(
                "REGISTRY_LIMIT",
                "The storage root quota would be exceeded.",
                reason_code="ROOT_QUOTA_EXCEEDED",
            )
        free = shutil.disk_usage(self.root).free
        required_free = held_values["byte_count"] + requested["byte_count"] + FREE_SPACE_RESERVE_BYTES
        if free < required_free:
            raise ApiError(
                "DISK_FULL",
                "Free storage is below the required reserve.",
                reason_code="INSUFFICIENT_STORAGE",
            )
        for key, (table, limit) in _ROW_LIMITS.items():
            if self._table_count(connection, table) + held_values[key] + requested[key] > limit:
                raise ApiError(
                    "REGISTRY_LIMIT",
                    "A registry count limit would be exceeded.",
                    reason_code="REGISTRY_COUNT_EXCEEDED",
                )
        identity = reservation_id or str(uuid.uuid4())
        _uuid_text(identity, "reservation_id")
        try:
            connection.execute(
                """
                INSERT INTO reservations(
                    reservation_id, owner_kind, owner_id, byte_count,
                    artifact_rows, dataset_rows, run_rows, model_rows,
                    checkpoint_rows, state, created_at, released_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'held', ?, NULL)
                """,
                (
                    identity,
                    owner_kind,
                    owner_id,
                    requested["byte_count"],
                    requested["artifact_rows"],
                    requested["dataset_rows"],
                    requested["run_rows"],
                    requested["model_rows"],
                    requested["checkpoint_rows"],
                    utc_now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ApiError("STATE_CONFLICT", "This capacity owner already has a reservation.") from exc
        return identity

    def reserve_capacity(self, **values: Any) -> str:
        with self.db.transaction() as connection:
            return self.reserve_capacity_in(connection, **values)

    def release_capacity(
        self, connection: sqlite3.Connection, reservation_id: str
    ) -> None:
        _uuid_text(reservation_id, "reservation_id")
        connection.execute(
            """
            UPDATE reservations SET state = 'released', released_at = ?
            WHERE reservation_id = ? AND state = 'held'
            """,
            (utc_now(), reservation_id),
        )

    def release_capacity_now(self, reservation_id: str) -> None:
        with self.db.transaction() as connection:
            self.release_capacity(connection, reservation_id)

    def create_staging_dir(self, staging_owner: str) -> Path:
        _uuid_text(staging_owner, "staging_owner")
        jobs = self.root / "jobs"
        owner = jobs / staging_owner
        staging = owner / "staging"
        ensure_private_directory(jobs)
        ensure_private_directory(owner)
        ensure_private_directory(staging)
        return staging

    @staticmethod
    def _stream_chunks(source: BinaryIO | bytes | bytearray | memoryview) -> Iterator[bytes]:
        if isinstance(source, (bytes, bytearray, memoryview)):
            yield bytes(source)
            return
        read = getattr(source, "read", None)
        if not callable(read):
            raise TypeError("source must be a binary readable stream")
        while True:
            chunk = read(1024 * 1024)
            if chunk is None or chunk == b"":
                return
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise TypeError("source.read() must return bytes")
            yield bytes(chunk)

    def stage_stream(
        self,
        source: BinaryIO | bytes | bytearray | memoryview,
        staging_owner: str,
        *,
        max_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> StagedArtifact:
        if max_bytes is not None:
            _nonnegative_u64(max_bytes, "max_bytes")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise ValueError("expected_sha256 must be lowercase SHA-256")
        staging = self.create_staging_dir(staging_owner)
        path = staging / f"{uuid.uuid4()}.partial"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb", buffering=0) as stream:
                for chunk in self._stream_chunks(source):
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise ApiError("PAYLOAD_TOO_LARGE", "The uploaded content is too large.")
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = stream.write(view)
                        if written is None or written <= 0:
                            raise OSError(errno.EIO, "short write")
                        view = view[written:]
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(staging)
            actual = digest.hexdigest()
            if expected_sha256 is not None and actual != expected_sha256:
                raise ApiError(
                    "VALIDATION_FAILED",
                    "The uploaded content digest does not match.",
                    reason_code="CHECKSUM_MISMATCH",
                )
            return StagedArtifact(staging_owner, path, size, actual)
        except BaseException as exc:
            path.unlink(missing_ok=True)
            try:
                _fsync_directory(staging)
            except OSError:
                pass
            if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
                raise ApiError(
                    "DISK_FULL",
                    "Free storage is below the required reserve.",
                    reason_code="INSUFFICIENT_STORAGE",
                ) from exc
            raise

    @staticmethod
    def _validate_artifact_fields(
        *,
        artifact_type: str,
        display_name: str,
        media_type: str,
        preview_policy: str,
        origin: str,
        job_id: str | None,
        artifact_id: str,
    ) -> None:
        if not isinstance(artifact_type, str) or _ARTIFACT_TYPE_RE.fullmatch(artifact_type) is None:
            raise ValueError("artifact_type is invalid")
        if not isinstance(display_name, str) or not (1 <= len(display_name) <= 120):
            raise ValueError("display_name must contain 1 to 120 scalars")
        if not isinstance(media_type, str) or not (1 <= len(media_type) <= 120):
            raise ValueError("media_type must contain 1 to 120 scalars")
        if preview_policy not in _PREVIEW_POLICIES:
            raise ValueError("preview_policy is invalid")
        if origin not in _ORIGINS:
            raise ValueError("origin is invalid")
        _uuid_text(artifact_id, "artifact_id")
        if job_id is not None:
            _uuid_text(job_id, "job_id")

    def _storage_corrupt(self, message: str) -> ApiError:
        self.db.enter_read_only_recovery("STORAGE_CORRUPT")
        return ApiError(
            "STORAGE_UNAVAILABLE",
            message,
            reason_code="STORAGE_CORRUPT",
        )

    def _verify_object_path(self, path: Path, size: int, digest: str) -> None:
        try:
            _reject_linked_components(path, include_leaf=True)
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                raise OSError("registered object size or type mismatch")
            calculated = hashlib.sha256()
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    calculated.update(chunk)
            if calculated.hexdigest() != digest:
                raise OSError("registered object digest mismatch")
        except (OSError, StorageSecurityError) as exc:
            raise self._storage_corrupt("A registered artifact failed integrity verification.") from exc

    def commit_staged_in(
        self,
        connection: sqlite3.Connection,
        staged: StagedArtifact,
        *,
        artifact_type: str,
        display_name: str,
        media_type: str,
        preview_policy: str,
        origin: str,
        job_id: str | None = None,
        artifact_id: str | None = None,
        preserve_staged: bool = False,
    ) -> dict[str, Any]:
        identity = artifact_id or self.allocate_artifact_id()
        self._validate_artifact_fields(
            artifact_type=artifact_type,
            display_name=display_name,
            media_type=media_type,
            preview_policy=preview_policy,
            origin=origin,
            job_id=job_id,
            artifact_id=identity,
        )
        if staged.path.parent != self.create_staging_dir(staged.staging_owner):
            raise ValueError("staged artifact is outside its owned staging directory")
        _reject_linked_components(staged.path, include_leaf=True)
        object_dir = self.root / "objects" / staged.sha256[:2] / staged.sha256
        ensure_private_directory(object_dir)
        destination = object_dir / staged.sha256
        relative_path = PurePosixPath(
            "objects", staged.sha256[:2], staged.sha256, staged.sha256
        ).as_posix()
        created_at = utc_now()
        if os.path.lexists(destination):
            self._verify_object_path(destination, staged.size, staged.sha256)
            if not preserve_staged:
                staged.path.unlink(missing_ok=True)
                _fsync_directory(staged.path.parent)
        else:
            temporary: Path | None = None
            try:
                if preserve_staged:
                    temporary = object_dir / (
                        f".{staged.sha256}.{uuid.uuid4().hex}.partial"
                    )
                    digest = hashlib.sha256()
                    copied = 0
                    source_flags = (
                        os.O_RDONLY
                        | getattr(os, "O_BINARY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    destination_flags = (
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_BINARY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    source_descriptor = os.open(staged.path, source_flags)
                    try:
                        destination_descriptor = os.open(
                            temporary, destination_flags, 0o600
                        )
                        try:
                            with os.fdopen(
                                source_descriptor, "rb", closefd=False
                            ) as source:
                                with os.fdopen(
                                    destination_descriptor, "wb", closefd=False
                                ) as target:
                                    for chunk in iter(
                                        lambda: source.read(1024 * 1024), b""
                                    ):
                                        copied += len(chunk)
                                        digest.update(chunk)
                                        target.write(chunk)
                                    target.flush()
                                    os.fsync(target.fileno())
                        finally:
                            os.close(destination_descriptor)
                    finally:
                        os.close(source_descriptor)
                    if (
                        copied != staged.size
                        or digest.hexdigest() != staged.sha256
                    ):
                        raise OSError("staged artifact changed before commit")
                    os.replace(temporary, destination)
                    temporary = None
                else:
                    os.replace(staged.path, destination)
                    if os.name != "nt":
                        os.chmod(destination, 0o600)
                _fsync_directory(object_dir)
            except OSError as exc:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                if exc.errno == errno.ENOSPC:
                    raise ApiError(
                        "DISK_FULL",
                        "Free storage is below the required reserve.",
                        reason_code="INSUFFICIENT_STORAGE",
                    ) from exc
                raise ApiError("STORAGE_UNAVAILABLE", "Artifact bytes could not be committed.") from exc
        connection.execute(
            "INSERT OR IGNORE INTO objects(sha256, size_bytes, relative_path, created_at) VALUES (?, ?, ?, ?)",
            (staged.sha256, staged.size, relative_path, created_at),
        )
        object_row = connection.execute(
            "SELECT size_bytes, relative_path FROM objects WHERE sha256 = ?", (staged.sha256,)
        ).fetchone()
        if object_row is None or int(object_row[0]) != staged.size or str(object_row[1]) != relative_path:
            raise self._storage_corrupt("Artifact metadata conflicts with registered bytes.")
        descriptor = {
            "artifact_id": identity,
            "type": artifact_type,
            "display_name": display_name,
            "relative_path": relative_path,
            "size_bytes": staged.size,
            "sha256": staged.sha256,
            "created_at": created_at,
            "media_type": media_type,
            "preview_policy": preview_policy,
            "origin": origin,
            "job_id": job_id,
        }
        connection.execute(
            """
            INSERT INTO artifacts(
                artifact_id, type, display_name, relative_path, size_bytes,
                sha256, created_at, media_type, preview_policy, origin,
                job_id, content_revision, updated_at, deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, NULL)
            """,
            (
                identity,
                artifact_type,
                display_name,
                relative_path,
                staged.size,
                staged.sha256,
                created_at,
                media_type,
                preview_policy,
                origin,
                job_id,
                created_at,
            ),
        )
        return descriptor

    def register_stream(
        self,
        source: BinaryIO | bytes | bytearray | memoryview,
        staging_owner: str,
        *,
        artifact_type: str,
        display_name: str,
        media_type: str,
        preview_policy: str,
        origin: str,
        job_id: str | None = None,
        max_bytes: int | None = None,
        expected_sha256: str | None = None,
        artifact_id: str | None = None,
        idempotency_commit: _IdempotencyCommit | None = None,
        reservation_id: str | None = None,
    ) -> dict[str, Any]:
        staged = self.stage_stream(
            source,
            staging_owner,
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
        )
        try:
            with self.db.transaction() as connection:
                descriptor = self.commit_staged_in(
                    connection,
                    staged,
                    artifact_type=artifact_type,
                    display_name=display_name,
                    media_type=media_type,
                    preview_policy=preview_policy,
                    origin=origin,
                    job_id=job_id,
                    artifact_id=artifact_id,
                )
                if reservation_id is not None:
                    self.release_capacity(connection, reservation_id)
                if idempotency_commit is not None:
                    idempotency_commit.record_success(
                        connection, status_code=201, response_body=descriptor
                    )
                self.db.bump_revision(connection)
            return descriptor
        finally:
            staged.path.unlink(missing_ok=True)

    def accept_job_input(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        canonical_bytes: bytes,
        canonical_sha256: str,
        schema_id: str,
        origin: str,
        reservation: Mapping[str, int],
    ) -> str:
        _uuid_text(job_id, "job_id")
        if not isinstance(canonical_bytes, bytes):
            raise TypeError("canonical_bytes must be bytes")
        if hashlib.sha256(canonical_bytes).hexdigest() != canonical_sha256:
            raise ValueError("canonical_sha256 does not identify canonical_bytes")
        allowed = {
            "byte_count",
            "artifact_rows",
            "dataset_rows",
            "run_rows",
            "model_rows",
            "checkpoint_rows",
        }
        if not isinstance(reservation, Mapping) or set(reservation) - allowed or "byte_count" not in reservation:
            raise ValueError("reservation contains unsupported or missing fields")
        self.reserve_capacity_in(
            connection,
            owner_kind="job",
            owner_id=job_id,
            reservation_id=job_id,
            **dict(reservation),
        )
        staged = self.stage_stream(
            io.BytesIO(canonical_bytes),
            job_id,
            max_bytes=1024 * 1024,
            expected_sha256=canonical_sha256,
        )
        try:
            descriptor = self.commit_staged_in(
                connection,
                staged,
                artifact_type="job_request",
                display_name="job-request.json",
                media_type="application/json",
                preview_policy="metadata_only",
                origin=origin,
                job_id=job_id,
            )
            return str(descriptor["artifact_id"])
        finally:
            staged.path.unlink(missing_ok=True)

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        _uuid_text(artifact_id, "artifact_id")
        with self.db.read() as connection:
            row = connection.execute(
                """
                SELECT artifact_id, type, display_name, relative_path, size_bytes,
                       sha256, created_at, media_type, preview_policy, origin, job_id
                FROM artifacts WHERE artifact_id = ? AND deleted_at IS NULL
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise ApiError("NOT_FOUND", "Artifact was not found.")
        keys = (
            "artifact_id",
            "type",
            "display_name",
            "relative_path",
            "size_bytes",
            "sha256",
            "created_at",
            "media_type",
            "preview_policy",
            "origin",
            "job_id",
        )
        result = dict(zip(keys, row, strict=True))
        result["size_bytes"] = int(result["size_bytes"])
        return result

    def assert_content_readable(
        self, artifact_id: str, *, allow_sealed_internal: bool = False
    ) -> dict[str, Any]:
        descriptor = self.get_artifact(artifact_id)
        if not allow_sealed_internal:
            sealed = descriptor["sha256"] in self._sealed_sha256
            if not sealed:
                with self.db.read() as connection:
                    sealed = connection.execute(
                        "SELECT 1 FROM dataset_splits WHERE sha256 = ? AND sealed = 1 LIMIT 1",
                        (descriptor["sha256"],),
                    ).fetchone() is not None
            if sealed:
                raise ApiError(
                    "STATE_CONFLICT",
                    "Sealed split content requires a consumed release token.",
                    reason_code="SEALED_SPLIT_REQUIRES_TOKEN",
                )
        return descriptor

    def _registered_path(self, descriptor: Mapping[str, Any]) -> Path:
        digest = str(descriptor["sha256"])
        expected = PurePosixPath("objects", digest[:2], digest, digest)
        relative = PurePosixPath(str(descriptor["relative_path"]))
        if relative != expected:
            raise self._storage_corrupt("A registered artifact path is invalid.")
        return self.root.joinpath(*relative.parts)

    def open_verified_artifact(
        self, artifact_id: str, *, allow_sealed_internal: bool = False
    ) -> BinaryIO:
        descriptor = self.assert_content_readable(
            artifact_id, allow_sealed_internal=allow_sealed_internal
        )
        path = self._registered_path(descriptor)
        try:
            _reject_linked_components(path, include_leaf=True)
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            raw_descriptor = os.open(path, flags)
            stream = os.fdopen(raw_descriptor, "rb")
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != int(descriptor["size_bytes"]):
                raise OSError("artifact size or type mismatch")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != descriptor["sha256"]:
                raise OSError("artifact digest mismatch")
            stream.seek(0)
        except (OSError, StorageSecurityError) as exc:
            try:
                stream.close()
            except UnboundLocalError:
                pass
            raise self._storage_corrupt("A registered artifact failed integrity verification.") from exc
        return stream

    def read_artifact_bytes(
        self, artifact_id: str, *, allow_sealed_internal: bool = False
    ) -> bytes:
        with self.open_verified_artifact(
            artifact_id, allow_sealed_internal=allow_sealed_internal
        ) as stream:
            return stream.read()

    def verified_artifact_path(
        self, artifact_id: str, *, allow_sealed_internal: bool = False
    ) -> Path:
        """Return the content-addressed path after verifying its exact bytes."""
        descriptor = self.assert_content_readable(
            artifact_id, allow_sealed_internal=allow_sealed_internal
        )
        with self.open_verified_artifact(
            artifact_id, allow_sealed_internal=allow_sealed_internal
        ):
            pass
        return self._registered_path(descriptor)

    def recover(self) -> RecoveryReport:
        if self.db.read_only:
            return RecoveryReport(False, 0, 0, self.db.recovery_reason)
        verified = 0
        try:
            with self.db.read() as connection:
                rows = tuple(
                    connection.execute(
                        "SELECT sha256, size_bytes, relative_path FROM objects ORDER BY sha256"
                    )
                )
            for row in rows:
                descriptor = {
                    "sha256": str(row[0]),
                    "size_bytes": int(row[1]),
                    "relative_path": str(row[2]),
                }
                path = self._registered_path(descriptor)
                self._verify_object_path(path, descriptor["size_bytes"], descriptor["sha256"])
                verified += 1
        except (ApiError, sqlite3.DatabaseError, OSError, ValueError, TypeError) as exc:
            self.db.enter_read_only_recovery("STORAGE_CORRUPT")
            return RecoveryReport(False, verified, 0, "STORAGE_CORRUPT")

        quarantined = 0
        jobs = self.root / "jobs"
        if jobs.exists():
            try:
                _reject_linked_components(jobs, include_leaf=True)
                quarantine = self.root / "quarantine"
                ensure_private_directory(quarantine)
                for owner in sorted(jobs.iterdir(), key=lambda item: item.name):
                    if is_reparse_point(owner) or not owner.is_dir():
                        raise OSError("unsafe job staging owner")
                    staging = owner / "staging"
                    if not staging.exists():
                        continue
                    _reject_linked_components(staging, include_leaf=True)
                    owner_quarantine = quarantine / owner.name
                    ensure_private_directory(owner_quarantine)
                    for entry in sorted(staging.iterdir(), key=lambda item: item.name):
                        if is_reparse_point(entry):
                            raise OSError("unsafe staging entry")
                        destination = owner_quarantine / f"{entry.name}-{uuid.uuid4()}"
                        os.replace(entry, destination)
                        quarantined += 1
                    _fsync_directory(staging)
                    _fsync_directory(owner_quarantine)
                _fsync_directory(quarantine)
            except (OSError, StorageSecurityError):
                self.db.enter_read_only_recovery("STORAGE_CORRUPT")
                return RecoveryReport(False, verified, quarantined, "STORAGE_CORRUPT")
        return RecoveryReport(True, verified, quarantined, None)

    @staticmethod
    def _limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 100):
            raise ApiError(
                "VALIDATION_FAILED",
                "List limit must be from 1 through 100.",
                reason_code="SCHEMA_INVALID",
                field_errors=[{"field_path": "limit", "message": "Must be from 1 through 100."}],
            )
        return limit

    def list_artifacts(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        limit = self._limit(limit)
        selected_filters = dict(filters or {})
        allowed = {"type", "origin", "job_id"}
        if set(selected_filters) - allowed:
            raise ValueError("unsupported artifact list filter")
        order = "created_at_desc,artifact_id_desc"
        anchor = None
        if cursor:
            anchor = self.cursor_codec.decode(
                cursor,
                schema="artifact-list-v1",
                order=order,
                filters=selected_filters,
            )
        clauses = ["deleted_at IS NULL"]
        parameters: list[Any] = []
        for key in sorted(selected_filters):
            clauses.append(f"{key} = ?")
            parameters.append(selected_filters[key])
        if anchor is not None:
            clauses.append("(created_at < ? OR (created_at = ? AND artifact_id < ?))")
            parameters.extend((anchor[0], anchor[0], anchor[1]))
        sql = (
            "SELECT artifact_id, type, display_name, relative_path, size_bytes, sha256, "
            "created_at, media_type, preview_policy, origin, job_id FROM artifacts WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC, artifact_id DESC LIMIT ?"
        )
        parameters.append(limit + 1)
        with self.db.read() as connection:
            rows = tuple(connection.execute(sql, parameters))
        keys = (
            "artifact_id",
            "type",
            "display_name",
            "relative_path",
            "size_bytes",
            "sha256",
            "created_at",
            "media_type",
            "preview_policy",
            "origin",
            "job_id",
        )
        items = [dict(zip(keys, row, strict=True)) for row in rows[:limit]]
        for item in items:
            item["size_bytes"] = int(item["size_bytes"])
        next_cursor = None
        if len(rows) > limit and items:
            last = items[-1]
            next_cursor = self.cursor_codec.encode(
                schema="artifact-list-v1",
                order=order,
                filters=selected_filters,
                created_at=str(last["created_at"]),
                item_id=str(last["artifact_id"]),
            )
        return {"items": items, "next_cursor": next_cursor}

    @staticmethod
    def _record_table(kind: str) -> tuple[str, str, str]:
        mapping = {
            "datasets": ("datasets", "dataset_id", "manifest_json"),
            "jobs": ("jobs", "job_id", "record_json"),
            "runs": ("runs", "run_id", "record_json"),
            "models": ("models", "model_id", "record_json"),
            "checkpoints": ("checkpoints", "checkpoint_id", "record_json"),
            "tokenizers": ("tokenizers", "tokenizer_id", "record_json"),
        }
        try:
            return mapping[kind]
        except KeyError as exc:
            raise ValueError("unsupported registry kind") from exc

    def get_record(self, kind: str, record_id: str) -> dict[str, Any]:
        _uuid_text(record_id, f"{kind} id")
        table, identifier, payload = self._record_table(kind)
        with self.db.read() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            ).fetchone()
            row = None if exists is None else connection.execute(
                f"SELECT {payload} FROM {table} WHERE {identifier} = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            raise ApiError("NOT_FOUND", f"{kind[:-1].capitalize()} was not found.")
        try:
            value = json.loads(row[0])
        except (TypeError, json.JSONDecodeError) as exc:
            raise self._storage_corrupt("A registry record is malformed.") from exc
        if not isinstance(value, dict):
            raise self._storage_corrupt("A registry record is malformed.")
        return value

    def list_records(
        self,
        kind: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        limit = self._limit(limit)
        table, identifier, payload = self._record_table(kind)
        selected_filters = dict(filters or {})
        filter_columns = {
            "datasets": {"record_format", "origin", "eligibility"},
            "jobs": {"operation", "state", "request_sha256", "idempotency_key"},
            "runs": {"operation", "origin"},
            "models": {"backend", "origin"},
            "checkpoints": {"backend", "origin"},
            "tokenizers": {"origin"},
        }[kind]
        if set(selected_filters) - filter_columns:
            raise ValueError("unsupported registry list filter")
        order = f"created_at_desc,{identifier}_desc"
        anchor = None
        if cursor:
            anchor = self.cursor_codec.decode(
                cursor,
                schema=f"{kind}-list-v1",
                order=order,
                filters=selected_filters,
            )
        with self.db.read() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            ).fetchone()
            if exists is None:
                return {"items": [], "next_cursor": None}
            columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
            clauses: list[str] = []
            if "deleted_at" in columns:
                clauses.append("deleted_at IS NULL")
            parameters: list[Any] = []
            for key in sorted(selected_filters):
                clauses.append(f"{key} = ?")
                parameters.append(selected_filters[key])
            if anchor is not None:
                clauses.append(f"(created_at < ? OR (created_at = ? AND {identifier} < ?))")
                parameters.extend((anchor[0], anchor[0], anchor[1]))
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            sql = (
                f"SELECT {identifier}, created_at, {payload} FROM {table}{where} "
                f"ORDER BY created_at DESC, {identifier} DESC LIMIT ?"
            )
            parameters.append(limit + 1)
            rows = tuple(connection.execute(sql, parameters))
        items: list[dict[str, Any]] = []
        for row in rows[:limit]:
            try:
                value = json.loads(row[2])
            except (TypeError, json.JSONDecodeError) as exc:
                raise self._storage_corrupt("A registry record is malformed.") from exc
            if not isinstance(value, dict):
                raise self._storage_corrupt("A registry record is malformed.")
            items.append(value)
        next_cursor = None
        if len(rows) > limit and rows:
            last = rows[limit - 1]
            next_cursor = self.cursor_codec.encode(
                schema=f"{kind}-list-v1",
                order=order,
                filters=selected_filters,
                created_at=str(last[1]),
                item_id=str(last[0]),
            )
        return {"items": items, "next_cursor": next_cursor}


__all__ = [
    "CursorCodec",
    "FREE_SPACE_RESERVE_BYTES",
    "ROOT_QUOTA_BYTES",
    "RecoveryReport",
    "Registry",
    "StagedArtifact",
]
