from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from llm_foundations_companion.database import Database
from llm_foundations_companion.registry import Registry


class _FailAfterRename:
    def __init__(self) -> None:
        self.calls = 0

    def record_success(
        self,
        connection: Any,
        *,
        status_code: int,
        response_body: dict[str, Any],
    ) -> None:
        assert connection.in_transaction
        assert status_code == 201
        assert response_body["sha256"]
        self.calls += 1
        raise RuntimeError("injected failure after content-addressed rename")


def _registry(root: Path) -> tuple[Database, Registry]:
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    return database, registry


def _object_path(root: Path, payload: bytes) -> Path:
    digest = hashlib.sha256(payload).hexdigest()
    return root / "objects" / digest[:2] / digest / digest


def _row_counts(
    database: Database,
    *,
    digest: str,
    artifact_ids: tuple[str, ...],
) -> tuple[int, tuple[int, ...]]:
    with database.read() as connection:
        object_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM objects WHERE sha256 = ?",
                (digest,),
            ).fetchone()[0]
        )
        artifact_counts = tuple(
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()[0]
            )
            for artifact_id in artifact_ids
        )
    return object_count, artifact_counts


def test_default_registration_rollback_retains_young_orphan_without_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    database, registry = _registry(root)
    payload = b"default registration rollback\n"
    digest = hashlib.sha256(payload).hexdigest()
    artifact_id = str(uuid.uuid4())
    owner = str(uuid.uuid4())
    failure = _FailAfterRename()
    revision = database.revision

    with pytest.raises(RuntimeError, match="injected failure"):
        registry.register_stream(
            payload,
            owner,
            artifact_type="rollback_probe",
            display_name="rollback-probe.bin",
            media_type="application/octet-stream",
            preview_policy="download_only",
            origin="locally_created",
            artifact_id=artifact_id,
            idempotency_commit=failure,
        )

    object_path = _object_path(root, payload)
    staging = root / "jobs" / owner / "staging"
    assert failure.calls == 1
    assert database.revision == revision
    assert _row_counts(
        database, digest=digest, artifact_ids=(artifact_id,)
    ) == (0, (0,))
    assert staging.is_dir() and tuple(staging.iterdir()) == ()
    assert object_path.read_bytes() == payload

    reopened_database, reopened_registry = _registry(root)
    report = reopened_registry.recover()

    assert report.storage_writable
    assert report.verified_artifacts == 0
    assert report.quarantined_entries == 0
    assert report.failed_check is None
    assert reopened_database.revision == revision
    assert _row_counts(
        reopened_database, digest=digest, artifact_ids=(artifact_id,)
    ) == (0, (0,))
    # DAT-010 and DEL-007 require retaining a newly created orphan for at
    # least 24 hours; this immediate restart must not collect it.
    assert object_path.read_bytes() == payload


def test_failed_duplicate_registration_preserves_registered_shared_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    database, registry = _registry(root)
    payload = b"shared content-addressed bytes\n"
    digest = hashlib.sha256(payload).hexdigest()
    first_id = str(uuid.uuid4())
    failed_id = str(uuid.uuid4())

    first = registry.register_stream(
        payload,
        str(uuid.uuid4()),
        artifact_type="shared_probe",
        display_name="first.bin",
        media_type="application/octet-stream",
        preview_policy="download_only",
        origin="locally_created",
        artifact_id=first_id,
    )
    revision = database.revision
    failure = _FailAfterRename()

    with pytest.raises(RuntimeError, match="injected failure"):
        registry.register_stream(
            payload,
            str(uuid.uuid4()),
            artifact_type="shared_probe",
            display_name="failed-duplicate.bin",
            media_type="application/octet-stream",
            preview_policy="download_only",
            origin="locally_created",
            artifact_id=failed_id,
            idempotency_commit=failure,
        )

    object_path = _object_path(root, payload)
    assert failure.calls == 1
    assert database.revision == revision
    assert first["sha256"] == digest
    assert _row_counts(
        database,
        digest=digest,
        artifact_ids=(first_id, failed_id),
    ) == (1, (1, 0))
    assert object_path.read_bytes() == payload

    reopened_database, reopened_registry = _registry(root)
    report = reopened_registry.recover()

    assert report.storage_writable
    assert report.verified_artifacts == 1
    assert report.failed_check is None
    assert reopened_database.revision == revision
    assert _row_counts(
        reopened_database,
        digest=digest,
        artifact_ids=(first_id, failed_id),
    ) == (1, (1, 0))
    assert reopened_registry.read_artifact_bytes(first_id) == payload
    assert object_path.read_bytes() == payload
