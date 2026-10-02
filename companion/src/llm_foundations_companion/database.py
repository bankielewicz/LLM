"""Serialized SQLite transactions, migration custody, and recovery state."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .migrations import CORE_SCHEMA_VERSION, CORE_STATEMENTS
from .platform_security import ensure_private_directory, resolve_storage_root


_SAFE_AUDIT_VALUE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class DatabaseError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class IntegrityReport:
    quick_check: str
    foreign_key_errors: tuple[tuple[object, ...], ...]

    @property
    def ok(self) -> bool:
        return self.quick_check == "ok" and not self.foreign_key_errors


class Database:
    """One process-wide, RLock-serialized SQLite boundary."""

    def __init__(self, root: Path, *, read_only: bool = False) -> None:
        self.root = resolve_storage_root(root, create=not read_only)
        self.path = self.root / "metadata.sqlite3"
        self._requested_read_only = read_only
        self._read_only = read_only
        self._recovery_reason: str | None = None
        self._initialized = False
        self._installation_id: str | None = None
        self._lock = threading.RLock()
        self._local = threading.local()

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def recovery_reason(self) -> str | None:
        return self._recovery_reason

    @property
    def installation_id(self) -> str | None:
        if not self._initialized:
            raise RuntimeError("database is not initialized")
        return self._installation_id

    @property
    def storage_root_display(self) -> str:
        return self.root.name or "storage"

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            existed = self.path.exists()
            if existed:
                probe: sqlite3.Connection | None = None
                try:
                    # The first open of existing bytes is strictly read-only.  In
                    # particular, a future user_version must not be switched to WAL
                    # or otherwise touched by this older runtime.
                    probe = self._connect(read_only=True)
                    version = int(probe.execute("PRAGMA user_version").fetchone()[0])
                    if version > CORE_SCHEMA_VERSION:
                        self._enter_recovery("UNSUPPORTED_FUTURE_MIGRATION")
                        self._initialized = True
                        return
                    report = self._integrity_report(probe)
                    if not report.ok:
                        self._enter_recovery("STORAGE_CORRUPT")
                        self._initialized = True
                        return
                    if version >= 1:
                        # Validate the identity while the existing database is
                        # still open read-only.  A structurally valid database
                        # with missing or malformed root metadata is corrupt and
                        # must not be switched to WAL before recovery.
                        self._load_installation_id(probe)
                    if self._requested_read_only:
                        if version < CORE_SCHEMA_VERSION:
                            self._enter_recovery("MIGRATION_REQUIRED")
                        self._initialized = True
                        return
                except (sqlite3.DatabaseError, OSError, DatabaseError, ValueError, TypeError):
                    self._enter_recovery("STORAGE_CORRUPT")
                    self._initialized = True
                    return
                finally:
                    if probe is not None:
                        probe.close()
            elif self._requested_read_only:
                self._enter_recovery("STORAGE_UNAVAILABLE")
                self._initialized = True
                return

            connection: sqlite3.Connection | None = None
            try:
                connection = self._connect(read_only=False)
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                migrated = version < CORE_SCHEMA_VERSION
                if version < CORE_SCHEMA_VERSION:
                    if existed and self.path.stat().st_size:
                        self._write_validated_backup(connection, version)
                    connection.execute("BEGIN EXCLUSIVE")
                    try:
                        for statement in CORE_STATEMENTS:
                            connection.execute(statement)
                        now = utc_now()
                        connection.execute(
                            "INSERT OR REPLACE INTO migrations VALUES (?, ?, ?)",
                            (
                                CORE_SCHEMA_VERSION,
                                now,
                                f"companion storage schema v{CORE_SCHEMA_VERSION}",
                            ),
                        )
                        connection.execute(
                            "INSERT OR REPLACE INTO schema_components VALUES (?, ?, ?)",
                            ("core", CORE_SCHEMA_VERSION, now),
                        )
                        connection.execute(
                            f"PRAGMA user_version = {CORE_SCHEMA_VERSION}"
                        )
                        if tuple(connection.execute("PRAGMA foreign_key_check")):
                            raise DatabaseError(
                                "STORAGE_CORRUPT",
                                "migration created invalid references",
                            )
                        connection.commit()
                    except BaseException:
                        connection.rollback()
                        raise
                if not existed or migrated:
                    self._ensure_installation_id(connection)
                else:
                    self._load_installation_id(connection)
                report = self._integrity_report(connection)
                if not report.ok:
                    self._enter_recovery("STORAGE_CORRUPT")
                self._initialized = True
            except (sqlite3.DatabaseError, OSError, DatabaseError) as exc:
                self._enter_recovery("STORAGE_CORRUPT")
                self._initialized = True
                if not existed:
                    raise DatabaseError(
                        "STORAGE_UNAVAILABLE",
                        "metadata database initialization failed",
                    ) from exc
            finally:
                if connection is not None:
                    connection.close()

    def close(self) -> None:
        self._initialized = False

    def assert_writable(self) -> None:
        if self._read_only:
            raise DatabaseError(
                "STORAGE_UNAVAILABLE", "storage is in read-only recovery mode"
            )

    def enter_read_only_recovery(self, reason: str) -> None:
        with self._lock:
            self._enter_recovery(reason)

    def _enter_recovery(self, reason: str) -> None:
        self._read_only = True
        self._recovery_reason = reason

    def _connect(self, *, read_only: bool) -> sqlite3.Connection:
        if read_only:
            connection = sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro",
                uri=True,
                timeout=30,
                isolation_level=None,
            )
        else:
            connection = sqlite3.connect(
                self.path, timeout=30, isolation_level=None
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        if read_only:
            connection.execute("PRAGMA query_only = ON")
        else:
            mode = str(
                connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            ).lower()
            if mode != "wal":
                connection.close()
                raise DatabaseError(
                    "STORAGE_UNAVAILABLE", "metadata database did not enter WAL mode"
                )
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextlib.contextmanager
    def transaction(self, mode: str = "IMMEDIATE") -> Iterator[sqlite3.Connection]:
        if mode not in {"DEFERRED", "IMMEDIATE", "EXCLUSIVE"}:
            raise ValueError("unsupported SQLite transaction mode")
        if not self._initialized:
            raise RuntimeError("database is not initialized")
        self.assert_writable()
        with self._lock:
            if getattr(self._local, "in_transaction", False):
                raise RuntimeError("nested Database transactions are not supported")
            connection = self._connect(read_only=False)
            self._local.in_transaction = True
            try:
                connection.execute(f"BEGIN {mode}")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                self._local.in_transaction = False
                connection.close()

    @contextlib.contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        if not self._initialized:
            raise RuntimeError("database is not initialized")
        with self._lock:
            connection = self._connect(read_only=self._read_only)
            try:
                connection.execute("BEGIN")
                yield connection
                connection.rollback()
            finally:
                connection.close()

    @property
    def revision(self) -> int:
        with self.read() as connection:
            row = connection.execute(
                "SELECT database_revision FROM registry_state WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise DatabaseError("STORAGE_CORRUPT", "registry revision is unavailable")
        return int(row[0])

    def bump_revision(self, connection: sqlite3.Connection) -> int:
        connection.execute(
            "UPDATE registry_state SET database_revision = database_revision + 1 "
            "WHERE singleton = 1"
        )
        return int(
            connection.execute(
                "SELECT database_revision FROM registry_state WHERE singleton = 1"
            ).fetchone()[0]
        )

    def install_component_schema(
        self, component: str, version: int, statements: Iterable[str]
    ) -> None:
        if (
            not component
            or not component.replace("_", "").isalnum()
            or version < 1
        ):
            raise ValueError("invalid component schema identity")
        with self.transaction("EXCLUSIVE") as connection:
            row = connection.execute(
                "SELECT version FROM schema_components WHERE component = ?",
                (component,),
            ).fetchone()
            current = int(row[0]) if row else 0
            if current == version:
                return
            if current > version:
                raise DatabaseError(
                    "STORAGE_UNAVAILABLE",
                    "component schema is newer than this runtime",
                )
            for statement in statements:
                if statement.strip():
                    connection.execute(statement)
            connection.execute(
                "INSERT OR REPLACE INTO schema_components VALUES (?, ?, ?)",
                (component, version, utc_now()),
            )

    def integrity_report(self) -> IntegrityReport:
        with self._lock:
            connection = self._connect(read_only=self._read_only)
            try:
                return self._integrity_report(connection)
            finally:
                connection.close()

    @staticmethod
    def _integrity_report(connection: sqlite3.Connection) -> IntegrityReport:
        quick_rows = tuple(connection.execute("PRAGMA quick_check"))
        quick_values = [str(row[0]) for row in quick_rows]
        quick = "ok" if quick_values == ["ok"] else "; ".join(quick_values)
        foreign = tuple(
            tuple(row) for row in connection.execute("PRAGMA foreign_key_check")
        )
        return IntegrityReport(quick, foreign)

    def append_mutation_audit(
        self,
        connection: sqlite3.Connection,
        *,
        request_id: str,
        session_hash: str,
        operation: str,
        entity_ids: Sequence[str],
        outcome: str,
    ) -> None:
        values = (request_id, session_hash, operation, outcome, *entity_ids)
        if any(not isinstance(value, str) or not _SAFE_AUDIT_VALUE.fullmatch(value) for value in values):
            raise ValueError("mutation audit accepts only bounded identifiers")
        entity_json = json.dumps(
            list(entity_ids), ensure_ascii=False, separators=(",", ":")
        )
        connection.execute(
            """
            INSERT INTO mutation_audit(
                occurred_at, request_id, session_hash, operation,
                entity_ids_json, outcome
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                request_id,
                session_hash,
                operation,
                entity_json,
                outcome,
            ),
        )

    def audit_mutation(
        self,
        *,
        request_id: str,
        session_hash: str,
        operation: str,
        entity_ids: Sequence[str],
        outcome: str,
    ) -> None:
        with self.transaction() as connection:
            self.append_mutation_audit(
                connection,
                request_id=request_id,
                session_hash=session_hash,
                operation=operation,
                entity_ids=entity_ids,
                outcome=outcome,
            )

    def _ensure_installation_id(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT value_json FROM root_metadata WHERE key = 'installation_id'"
            ).fetchone()
            if row is None:
                installation_id = str(uuid.uuid4())
                now = utc_now()
                connection.execute(
                    "INSERT INTO root_metadata VALUES ('installation_id', ?, ?, ?)",
                    (json.dumps(installation_id), now, now),
                )
            else:
                installation_id = json.loads(row[0])
            uuid.UUID(installation_id)
            if str(uuid.UUID(installation_id)) != installation_id:
                raise ValueError
            connection.commit()
            self._installation_id = installation_id
        except BaseException:
            connection.rollback()
            raise

    def _load_installation_id(self, connection: sqlite3.Connection) -> None:
        row = connection.execute(
            "SELECT value_json FROM root_metadata WHERE key = 'installation_id'"
        ).fetchone()
        if row is None:
            raise DatabaseError(
                "STORAGE_CORRUPT", "installation identity is unavailable"
            )
        value = json.loads(row[0])
        if str(uuid.UUID(value)) != value:
            raise DatabaseError("STORAGE_CORRUPT", "installation identity is invalid")
        self._installation_id = value

    def _write_validated_backup(
        self, connection: sqlite3.Connection, old_version: int
    ) -> None:
        backup_dir = self.root / "backups"
        ensure_private_directory(backup_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = backup_dir / f"metadata-v{old_version}-{stamp}.sqlite3"
        backup = sqlite3.connect(destination)
        try:
            connection.backup(backup)
            if not self._integrity_report(backup).ok:
                raise DatabaseError(
                    "STORAGE_CORRUPT", "metadata backup validation failed"
                )
        finally:
            backup.close()
        if not destination.is_file() or destination.stat().st_size == 0:
            raise DatabaseError(
                "STORAGE_CORRUPT", "metadata backup was not written"
            )
        with destination.open("rb") as stream:
            os.fsync(stream.fileno())
        if os.name != "nt":
            descriptor = os.open(backup_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
