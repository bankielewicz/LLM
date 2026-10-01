from __future__ import annotations

import io
import os
import sqlite3
import uuid
from pathlib import Path

import pytest

from llm_foundations_companion.database import Database
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.platform_security import (
    InstanceLease,
    StorageSecurityError,
)
from llm_foundations_companion.registry import ROOT_QUOTA_BYTES, Registry


def _registry(tmp_path: Path) -> tuple[Database, Registry]:
    root = tmp_path / "root"
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    return database, registry


def test_database_initializes_wal_full_and_stable_installation(tmp_path: Path) -> None:
    database, _ = _registry(tmp_path)
    installation = database.installation_id
    assert installation is not None
    with database.read() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    reopened = Database(database.root)
    reopened.initialize()
    assert reopened.installation_id == installation
    assert reopened.integrity_report().ok


def test_future_database_is_inspected_read_only_without_inventing_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    path = root / "metadata.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 99")
    connection.execute("CREATE TABLE future_data(value TEXT)")
    connection.execute("INSERT INTO future_data VALUES ('preserve')")
    connection.commit()
    connection.close()
    before = path.read_bytes()
    database = Database(root)
    database.initialize()
    assert database.read_only
    assert database.recovery_reason == "UNSUPPORTED_FUTURE_MIGRATION"
    assert database.installation_id is None
    assert path.read_bytes() == before


def test_current_database_with_missing_identity_enters_recovery_without_writes(
    tmp_path: Path,
) -> None:
    database, _ = _registry(tmp_path)
    path = database.path
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM root_metadata WHERE key = 'installation_id'")
    before = path.read_bytes()
    reopened = Database(database.root)
    reopened.initialize()
    assert reopened.read_only
    assert reopened.recovery_reason == "STORAGE_CORRUPT"
    assert reopened.installation_id is None
    assert path.read_bytes() == before


def test_instance_lease_rejects_link_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    runtime = root / "runtime"
    runtime.mkdir(mode=0o700)
    target = root / "target"
    target.write_text("do not touch", encoding="utf-8")
    (runtime / "instance.lock").symlink_to(target)
    with pytest.raises(StorageSecurityError) as caught:
        InstanceLease(root).acquire()
    assert caught.value.code == "STORAGE_UNAVAILABLE"
    assert target.read_text(encoding="utf-8") == "do not touch"


def test_artifact_commit_reuse_verified_read_and_cursor_binding(tmp_path: Path) -> None:
    _, registry = _registry(tmp_path)
    first = registry.register_stream(
        io.BytesIO(b"same bytes"),
        str(uuid.uuid4()),
        artifact_type="test_blob",
        display_name="one",
        media_type="application/octet-stream",
        preview_policy="download_only",
        origin="imported",
    )
    second = registry.register_stream(
        io.BytesIO(b"same bytes"),
        str(uuid.uuid4()),
        artifact_type="test_blob",
        display_name="two",
        media_type="application/octet-stream",
        preview_policy="download_only",
        origin="imported",
    )
    assert first["artifact_id"] != second["artifact_id"]
    assert first["relative_path"] == second["relative_path"]
    assert registry.read_artifact_bytes(first["artifact_id"]) == b"same bytes"
    page = registry.list_artifacts(limit=1)
    assert len(page["items"]) == 1
    assert page["next_cursor"]
    following = registry.list_artifacts(limit=1, cursor=page["next_cursor"])
    assert len(following["items"]) == 1
    with pytest.raises(ApiError) as caught:
        registry.list_artifacts(
            limit=1,
            cursor=page["next_cursor"][:-1] + ("A" if page["next_cursor"][-1] != "A" else "B"),
        )
    assert (caught.value.code, caught.value.reason_code) == (
        "VALIDATION_FAILED",
        "SEMANTIC_INVALID",
    )


def test_capacity_preflight_checks_quota_and_row_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, registry = _registry(tmp_path)
    monkeypatch.setattr(registry, "_root_bytes", lambda: ROOT_QUOTA_BYTES)
    with pytest.raises(ApiError) as caught:
        registry.reserve_capacity(
            owner_kind="upload",
            owner_id=str(uuid.uuid4()),
            byte_count=1,
        )
    assert (caught.value.code, caught.value.reason_code) == (
        "REGISTRY_LIMIT",
        "ROOT_QUOTA_EXCEEDED",
    )
    monkeypatch.setattr(registry, "_root_bytes", lambda: 0)
    owner = str(uuid.uuid4())
    reservation = registry.reserve_capacity(
        owner_kind="upload",
        owner_id=owner,
        byte_count=1,
        artifact_rows=1,
        reservation_id=owner,
    )
    assert reservation == owner
    with database.transaction() as connection:
        registry.release_capacity(connection, reservation)
    with database.read() as connection:
        assert connection.execute(
            "SELECT state FROM reservations WHERE reservation_id = ?", (reservation,)
        ).fetchone()[0] == "released"


def test_job_request_artifact_and_reservation_share_caller_transaction(
    tmp_path: Path,
) -> None:
    database, registry = _registry(tmp_path)
    job_id = str(uuid.uuid4())
    canonical = b'{"operation":"context_preview"}'
    import hashlib

    digest = hashlib.sha256(canonical).hexdigest()
    with pytest.raises(RuntimeError):
        with database.transaction() as connection:
            registry.accept_job_input(
                connection,
                job_id=job_id,
                canonical_bytes=canonical,
                canonical_sha256=digest,
                schema_id="ContextPreviewRequest",
                origin="locally_created",
                reservation={"byte_count": 256 * 1024 * 1024, "artifact_rows": 16},
            )
            raise RuntimeError("roll back caller")
    with database.read() as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0


def test_recovery_quarantines_staging_and_detects_corrupt_object(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)
    owner = str(uuid.uuid4())
    staging = registry.create_staging_dir(owner)
    abandoned = staging / "abandoned.partial"
    abandoned.write_bytes(b"partial")
    report = registry.recover()
    assert report.storage_writable and report.quarantined_entries == 1
    assert not abandoned.exists()
    descriptor = registry.register_stream(
        io.BytesIO(b"healthy"),
        str(uuid.uuid4()),
        artifact_type="test_blob",
        display_name="healthy",
        media_type="application/octet-stream",
        preview_policy="download_only",
        origin="imported",
    )
    path = registry.root.joinpath(*descriptor["relative_path"].split("/"))
    path.write_bytes(b"corrupt")
    report = registry.recover()
    assert not report.storage_writable
    assert report.failed_check == "STORAGE_CORRUPT"
    assert database.read_only


def test_recovery_treats_linked_staging_root_as_corruption(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (registry.root / "jobs").symlink_to(outside, target_is_directory=True)
    report = registry.recover()
    assert report == type(report)(False, 0, 0, "STORAGE_CORRUPT")
    assert database.read_only
