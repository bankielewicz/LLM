from __future__ import annotations

import base64
import errno
import hashlib
import hmac
import io
import json
import os
import sqlite3
import types
import uuid
from pathlib import Path
from typing import Any

import pytest

import llm_foundations_companion.database as database_module
import llm_foundations_companion.registry as registry_module
from llm_foundations_companion.database import Database, DatabaseError, IntegrityReport
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.registry import CursorCodec, Registry, StagedArtifact
from llm_foundations_companion.schema import canonical_json


def _registry(tmp_path: Path) -> tuple[Database, Registry]:
    root = tmp_path / "root"
    database = Database(root)
    database.initialize()
    return database, Registry(database, root, str(uuid.uuid4()), os.urandom(32))


def _api_error(call: Any, code: str, reason_code: str | None = None) -> ApiError:
    with pytest.raises(ApiError) as caught:
        call()
    assert caught.value.code == code
    assert caught.value.reason_code == reason_code
    return caught.value


def _signed_cursor(key: bytes, value: Any) -> str:
    payload = canonical_json(value)
    signature = hmac.digest(key, payload, "sha256")
    return base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode("ascii")


def _insert_run(
    database: Database,
    run_id: str,
    *,
    created_at: str,
    operation: str = "tokenizer_train",
    record_json: str | None = None,
) -> None:
    record = record_json if record_json is not None else json.dumps(
        {"run_id": run_id, "operation": operation}, separators=(",", ":")
    )
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO runs(
                run_id, format, operation, origin, job_id,
                source_identity_json, created_at, updated_at, record_json
            ) VALUES (?, 'llm-foundations-run-v1', ?, 'locally_created',
                      NULL, NULL, ?, ?, ?)
            """,
            (run_id, operation, created_at, created_at, record),
        )


def test_database_guards_lifecycle_and_transaction_boundaries(tmp_path: Path) -> None:
    database = Database(tmp_path / "root")
    assert database.storage_root_display == "root"
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = database.installation_id
    with pytest.raises(RuntimeError, match="not initialized"):
        with database.read():
            pass
    with pytest.raises(RuntimeError, match="not initialized"):
        with database.transaction():
            pass

    database.initialize()
    installation_id = database.installation_id
    database.initialize()
    assert database.installation_id == installation_id
    with pytest.raises(ValueError, match="transaction mode"):
        with database.transaction("INVALID"):
            pass

    with database.transaction() as connection:
        connection.execute("CREATE TABLE rollback_probe(value TEXT)")
        with pytest.raises(RuntimeError, match="nested"):
            with database.transaction():
                pass
    with pytest.raises(RuntimeError, match="force rollback"):
        with database.transaction() as connection:
            connection.execute("INSERT INTO rollback_probe VALUES ('discard')")
            raise RuntimeError("force rollback")
    with database.read() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rollback_probe").fetchone()[0] == 0

    database.close()
    with pytest.raises(RuntimeError, match="not initialized"):
        with database.read():
            pass


def test_read_only_open_reports_missing_and_migration_required_storage(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir(mode=0o700)
    missing = Database(missing_root, read_only=True)
    missing.initialize()
    assert missing.recovery_reason == "STORAGE_UNAVAILABLE"
    assert missing.installation_id is None

    current, _ = _registry(tmp_path / "existing")
    installation_id = current.installation_id
    with sqlite3.connect(current.path) as connection:
        connection.execute("PRAGMA user_version = 1")
    old = Database(current.root, read_only=True)
    old.initialize()
    assert old.recovery_reason == "MIGRATION_REQUIRED"
    assert old.installation_id == installation_id
    with pytest.raises(DatabaseError) as caught:
        old.assert_writable()
    assert caught.value.code == "STORAGE_UNAVAILABLE"


def test_existing_database_with_invalid_installation_identity_enters_recovery(
    tmp_path: Path,
) -> None:
    database, _ = _registry(tmp_path)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE root_metadata SET value_json = ? WHERE key = 'installation_id'",
            (json.dumps(str(uuid.uuid4()).upper()),),
        )
    reopened = Database(database.root)
    reopened.initialize()
    assert reopened.recovery_reason == "STORAGE_CORRUPT"
    assert reopened.installation_id is None


def test_new_database_initialization_failure_is_storage_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "root")

    def fail_connect(*, read_only: bool) -> sqlite3.Connection:
        raise sqlite3.DatabaseError("simulated open failure")

    monkeypatch.setattr(database, "_connect", fail_connect)
    with pytest.raises(DatabaseError) as caught:
        database.initialize()
    assert caught.value.code == "STORAGE_UNAVAILABLE"
    assert database.recovery_reason == "STORAGE_CORRUPT"


def test_database_migration_rolls_back_invalid_schema_and_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid_sql = (
        "CREATE TABLE migration_probe(value TEXT)",
        "THIS IS NOT VALID SQL",
    )
    monkeypatch.setattr(database_module, "CORE_STATEMENTS", invalid_sql)
    database = Database(tmp_path / "invalid-sql")
    with pytest.raises(DatabaseError) as caught:
        database.initialize()
    assert caught.value.code == "STORAGE_UNAVAILABLE"
    with sqlite3.connect(database.path) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'migration_probe'"
        ).fetchone() is None

    invalid_reference = (
        "CREATE TABLE parent(id INTEGER PRIMARY KEY)",
        """
        CREATE TABLE child(
            parent_id INTEGER REFERENCES parent(id)
                DEFERRABLE INITIALLY DEFERRED
        )
        """,
        "INSERT INTO child(parent_id) VALUES (99)",
    )
    monkeypatch.setattr(database_module, "CORE_STATEMENTS", invalid_reference)
    database = Database(tmp_path / "invalid-reference")
    with pytest.raises(DatabaseError) as caught:
        database.initialize()
    assert caught.value.code == "STORAGE_UNAVAILABLE"


def test_database_enters_recovery_when_integrity_fails_before_or_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current, _ = _registry(tmp_path / "existing")
    monkeypatch.setattr(
        Database,
        "_integrity_report",
        staticmethod(lambda connection: IntegrityReport("bad", ())),
    )
    reopened = Database(current.root)
    reopened.initialize()
    assert reopened.recovery_reason == "STORAGE_CORRUPT"
    assert reopened.installation_id is None

    fresh = Database(tmp_path / "fresh")
    fresh.initialize()
    assert fresh.recovery_reason == "STORAGE_CORRUPT"
    assert fresh.read_only


def test_database_rejects_non_wal_writable_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "root")

    class Result:
        def fetchone(self) -> tuple[str]:
            return ("delete",)

    class Connection:
        row_factory: object | None = None
        closed = False

        def execute(self, statement: str) -> Result:
            return Result()

        def close(self) -> None:
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(database_module.sqlite3, "connect", lambda *a, **k: connection)
    with pytest.raises(DatabaseError, match="WAL mode") as caught:
        database._connect(read_only=False)
    assert caught.value.code == "STORAGE_UNAVAILABLE"
    assert connection.closed


def test_database_component_revision_and_audit_failure_boundaries(tmp_path: Path) -> None:
    database, _ = _registry(tmp_path)
    with database.transaction() as connection:
        connection.execute("DELETE FROM registry_state WHERE singleton = 1")
    with pytest.raises(DatabaseError, match="revision") as caught:
        _ = database.revision
    assert caught.value.code == "STORAGE_CORRUPT"
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO registry_state(singleton, database_revision) VALUES (1, 0)"
        )

    for component, version in (("", 1), ("bad-name", 1), ("valid_name", 0)):
        with pytest.raises(ValueError, match="component schema"):
            database.install_component_schema(component, version, ())
    database.install_component_schema(
        "coverage_probe", 2, ("", "CREATE TABLE coverage_component(value TEXT)")
    )
    database.install_component_schema("coverage_probe", 2, ())
    with pytest.raises(DatabaseError, match="newer") as caught:
        database.install_component_schema("coverage_probe", 1, ())
    assert caught.value.code == "STORAGE_UNAVAILABLE"

    with pytest.raises(ValueError, match="bounded identifiers"):
        database.audit_mutation(
            request_id="contains a space",
            session_hash="session",
            operation="create",
            entity_ids=("entity",),
            outcome="accepted",
        )
    database.audit_mutation(
        request_id="request-1",
        session_hash="session-1",
        operation="create",
        entity_ids=("entity-1", "entity-2"),
        outcome="accepted",
    )
    with database.read() as connection:
        row = connection.execute(
            "SELECT entity_ids_json FROM mutation_audit WHERE request_id = 'request-1'"
        ).fetchone()
    assert json.loads(row[0]) == ["entity-1", "entity-2"]


def test_backup_validation_rejects_corrupt_and_empty_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, _ = _registry(tmp_path / "corrupt")
    with sqlite3.connect(database.path) as connection:
        connection.execute("PRAGMA user_version = 1")
    reports = iter((IntegrityReport("ok", ()), IntegrityReport("bad", ())))
    monkeypatch.setattr(
        Database, "_integrity_report", staticmethod(lambda connection: next(reports))
    )
    reopened = Database(database.root)
    reopened.initialize()
    assert reopened.recovery_reason == "STORAGE_CORRUPT"

    monkeypatch.undo()
    empty = Database(tmp_path / "empty")

    class NoBackup:
        def backup(self, destination: sqlite3.Connection) -> None:
            return None

    with pytest.raises(DatabaseError, match="was not written") as caught:
        empty._write_validated_backup(NoBackup(), 1)  # type: ignore[arg-type]
    assert caught.value.code == "STORAGE_CORRUPT"


@pytest.mark.parametrize("value", (None, "not-a-uuid", str(uuid.uuid4()).upper()))
def test_uuid_boundary_requires_canonical_lowercase_text(value: object) -> None:
    with pytest.raises(ValueError, match="lowercase UUID"):
        registry_module._uuid_text(value, "identifier")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (True, -1, 1 << 64, 1.5))
def test_capacity_values_require_unsigned_64_bit_integers(value: object) -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        registry_module._nonnegative_u64(value, "byte_count")  # type: ignore[arg-type]


def test_cursor_decoder_rejects_structural_signature_and_binding_failures() -> None:
    instance_id = str(uuid.uuid4())
    key = b"k" * 32
    codec = CursorCodec(instance_id, key)
    arguments = {
        "schema": "items-v1",
        "order": "created_at_desc,item_id_desc",
        "filters": {"origin": "imported"},
    }
    valid = codec.encode(
        **arguments,
        created_at="2026-10-01T00:00:00.000Z",
        item_id=str(uuid.uuid4()),
    )
    decoded = base64.urlsafe_b64decode(valid + "=" * (-len(valid) % 4))
    tampered = base64.urlsafe_b64encode(
        decoded[:-1] + bytes([decoded[-1] ^ 1])
    ).rstrip(b"=").decode("ascii")
    wrong_shape = _signed_cursor(key, ["not", "an", "object"])
    wrong_binding = _signed_cursor(
        key,
        {
            "created_at": "2026-10-01T00:00:00.000Z",
            "filters": arguments["filters"],
            "instance_id": instance_id,
            "item_id": str(uuid.uuid4()),
            "order": arguments["order"],
            "schema": arguments["schema"],
            "version": 2,
        },
    )
    for cursor in (
        None,
        "",
        "A" * 4097,
        "AA",
        "Zg==",
        tampered,
        wrong_shape,
        wrong_binding,
    ):
        _api_error(
            lambda cursor=cursor: codec.decode(cursor, **arguments),  # type: ignore[arg-type]
            "VALIDATION_FAILED",
            "SEMANTIC_INVALID",
        )
    with pytest.raises(ValueError, match="32 bytes"):
        CursorCodec(instance_id, b"short")


def test_registry_rejects_mismatched_root_and_invalid_fixture_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, _ = _registry(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="registry root"):
        Registry(database, other, str(uuid.uuid4()), os.urandom(32))

    valid_row = {
        "path": "capstone-support-v1/sealed_test.jsonl",
        "records": 1,
        "sha256": "a" * 64,
        "utf8_bytes": 1,
    }
    invalid_documents = (
        {},
        {"format": "llm-foundations-materialized-fixtures-v1", "files": ["bad-row"]},
        {
            "format": "llm-foundations-materialized-fixtures-v1",
            "files": [{**valid_row, "extra": True}],
        },
        {
            "format": "llm-foundations-materialized-fixtures-v1",
            "files": [{**valid_row, "sha256": "NOT-A-DIGEST"}],
        },
        {
            "format": "llm-foundations-materialized-fixtures-v1",
            "files": [{**valid_row, "path": "ordinary.jsonl"}],
        },
    )
    for document in invalid_documents:
        monkeypatch.setattr(
            registry_module,
            "load_document",
            lambda name, document=document: document,
        )
        with pytest.raises(RuntimeError, match="fixture"):
            Registry._load_sealed_sha256()


def test_registry_usage_and_table_count_fail_closed_on_unsafe_storage(
    tmp_path: Path,
) -> None:
    database, registry = _registry(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (registry.root / "linked").symlink_to(outside)
    _api_error(registry._root_bytes, "STORAGE_UNAVAILABLE")
    (registry.root / "linked").unlink()
    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    (registry.root / "linked-directory").symlink_to(
        outside_directory, target_is_directory=True
    )
    _api_error(registry._root_bytes, "STORAGE_UNAVAILABLE")
    with database.transaction() as connection:
        assert registry._table_count(connection, "table_that_does_not_exist") == 0


def test_capacity_reservation_rejects_invalid_owners_disk_pressure_and_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, registry = _registry(tmp_path)
    owner = str(uuid.uuid4())
    with database.transaction() as connection:
        with pytest.raises(ValueError, match="owner_kind"):
            registry.reserve_capacity_in(
                connection, owner_kind="other", owner_id=owner, byte_count=0
            )

    monkeypatch.setattr(registry, "_root_bytes", lambda: 0)
    monkeypatch.setattr(
        registry_module.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(free=0),
    )
    _api_error(
        lambda: registry.reserve_capacity(
            owner_kind="upload", owner_id=owner, byte_count=1
        ),
        "DISK_FULL",
        "INSUFFICIENT_STORAGE",
    )

    monkeypatch.setattr(
        registry_module.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(
            free=registry_module.FREE_SPACE_RESERVE_BYTES + 1024
        ),
    )
    monkeypatch.setattr(
        registry,
        "_table_count",
        lambda connection, table: registry_module._ROW_LIMITS["artifact_rows"][1]
        if table == "artifacts"
        else 0,
    )
    _api_error(
        lambda: registry.reserve_capacity(
            owner_kind="upload",
            owner_id=owner,
            byte_count=0,
            artifact_rows=1,
        ),
        "REGISTRY_LIMIT",
        "REGISTRY_COUNT_EXCEEDED",
    )

    monkeypatch.setattr(registry, "_table_count", lambda connection, table: 0)
    registry.reserve_capacity(
        owner_kind="upload",
        owner_id=owner,
        reservation_id=owner,
        byte_count=0,
    )
    _api_error(
        lambda: registry.reserve_capacity(
            owner_kind="upload", owner_id=owner, byte_count=0
        ),
        "STATE_CONFLICT",
    )


def test_staging_rejects_bad_streams_limits_and_digests_without_partial_files(
    tmp_path: Path,
) -> None:
    _, registry = _registry(tmp_path)

    class TextReader:
        def read(self, size: int) -> str:
            return "text"

    cases = (
        (object(), {}, TypeError),
        (TextReader(), {}, TypeError),
        (io.BytesIO(b"too large"), {"max_bytes": 3}, ApiError),
        (io.BytesIO(b"digest"), {"expected_sha256": "0" * 64}, ApiError),
    )
    for source, options, exception in cases:
        owner = str(uuid.uuid4())
        staging = registry.create_staging_dir(owner)
        with pytest.raises(exception):
            registry.stage_stream(source, owner, **options)
        assert tuple(staging.iterdir()) == ()
    with pytest.raises(ValueError, match="expected_sha256"):
        registry.stage_stream(
            b"data", str(uuid.uuid4()), expected_sha256="BAD"
        )


def test_staging_maps_disk_exhaustion_and_preserves_the_original_cleanup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, registry = _registry(tmp_path)

    class DiskFullReader:
        def read(self, size: int) -> bytes:
            raise OSError(errno.ENOSPC, "simulated staging exhaustion")

    _api_error(
        lambda: registry.stage_stream(DiskFullReader(), str(uuid.uuid4())),
        "DISK_FULL",
        "INSUFFICIENT_STORAGE",
    )

    def fail_fsync(path: Path) -> None:
        raise OSError(errno.EIO, "simulated cleanup fsync failure")

    monkeypatch.setattr(registry_module, "_fsync_directory", fail_fsync)
    with pytest.raises(TypeError, match="binary readable stream"):
        registry.stage_stream(object(), str(uuid.uuid4()))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_type", "Invalid-Type"),
        ("display_name", ""),
        ("media_type", ""),
        ("preview_policy", "inline"),
        ("origin", "remote"),
        ("job_id", "not-a-uuid"),
    ),
)
def test_artifact_metadata_validation_rejects_out_of_contract_values(
    field: str, value: object
) -> None:
    values: dict[str, Any] = {
        "artifact_type": "test_blob",
        "display_name": "artifact.bin",
        "media_type": "application/octet-stream",
        "preview_policy": "download_only",
        "origin": "imported",
        "job_id": None,
        "artifact_id": str(uuid.uuid4()),
    }
    values[field] = value
    with pytest.raises(ValueError):
        Registry._validate_artifact_fields(**values)


def test_commit_rejects_foreign_staging_path_and_changed_preserved_bytes(
    tmp_path: Path,
) -> None:
    database, registry = _registry(tmp_path)
    owner = str(uuid.uuid4())
    foreign = tmp_path / "foreign.partial"
    foreign.write_bytes(b"bytes")
    staged = StagedArtifact(owner, foreign, 5, hashlib.sha256(b"bytes").hexdigest())
    with database.transaction() as connection:
        with pytest.raises(ValueError, match="owned staging"):
            registry.commit_staged_in(
                connection,
                staged,
                artifact_type="test_blob",
                display_name="foreign.bin",
                media_type="application/octet-stream",
                preview_policy="download_only",
                origin="imported",
            )

    staged = registry.stage_stream(b"before", owner)
    staged.path.write_bytes(b"after!")
    with pytest.raises(ApiError) as caught:
        with database.transaction() as connection:
            registry.commit_staged_in(
                connection,
                staged,
                artifact_type="test_blob",
                display_name="changed.bin",
                media_type="application/octet-stream",
                preview_policy="download_only",
                origin="imported",
                preserve_staged=True,
            )
    assert caught.value.code == "STORAGE_UNAVAILABLE"
    assert staged.path.read_bytes() == b"after!"


def test_commit_maps_disk_exhaustion_and_metadata_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, registry = _registry(tmp_path / "disk-full")

    def no_space(source: Path, destination: Path) -> None:
        raise OSError(errno.ENOSPC, "simulated disk full")

    monkeypatch.setattr(registry_module.os, "replace", no_space)
    _api_error(
        lambda: registry.register_stream(
            b"payload",
            str(uuid.uuid4()),
            artifact_type="test_blob",
            display_name="payload.bin",
            media_type="application/octet-stream",
            preview_policy="download_only",
            origin="imported",
        ),
        "DISK_FULL",
        "INSUFFICIENT_STORAGE",
    )

    monkeypatch.undo()
    database, registry = _registry(tmp_path / "metadata-conflict")
    payload = b"payload"
    digest = hashlib.sha256(payload).hexdigest()
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO objects(sha256, size_bytes, relative_path, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                digest,
                len(payload) + 1,
                f"objects/{digest[:2]}/{digest}/wrong",
                database_module.utc_now(),
            ),
        )
    _api_error(
        lambda: registry.register_stream(
            payload,
            str(uuid.uuid4()),
            artifact_type="test_blob",
            display_name="payload.bin",
            media_type="application/octet-stream",
            preview_policy="download_only",
            origin="imported",
        ),
        "STORAGE_UNAVAILABLE",
        "STORAGE_CORRUPT",
    )
    assert database.read_only


def test_register_releases_reservation_records_idempotency_and_bumps_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, registry = _registry(tmp_path)
    monkeypatch.setattr(registry, "_root_bytes", lambda: 0)
    monkeypatch.setattr(
        registry_module.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(
            free=registry_module.FREE_SPACE_RESERVE_BYTES + 1024
        ),
    )
    owner = str(uuid.uuid4())
    registry.reserve_capacity(
        owner_kind="upload",
        owner_id=owner,
        reservation_id=owner,
        byte_count=7,
        artifact_rows=1,
    )
    before = database.revision

    class IdempotencyCommit:
        calls: list[tuple[int, dict[str, Any], str]] = []

        def record_success(
            self,
            connection: sqlite3.Connection,
            *,
            status_code: int,
            response_body: bytes | dict[str, Any],
        ) -> bytes:
            state = connection.execute(
                "SELECT state FROM reservations WHERE reservation_id = ?", (owner,)
            ).fetchone()[0]
            self.calls.append(
                (status_code, dict(response_body), str(state))  # type: ignore[arg-type]
            )
            return b"recorded"

    commit = IdempotencyCommit()
    descriptor = registry.register_stream(
        b"payload",
        owner,
        artifact_type="test_blob",
        display_name="payload.bin",
        media_type="application/octet-stream",
        preview_policy="download_only",
        origin="imported",
        reservation_id=owner,
        idempotency_commit=commit,
    )
    assert commit.calls == [(201, descriptor, "released")]
    assert database.revision == before + 1


def test_accept_job_input_validates_bytes_digest_and_reservation(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)
    job_id = str(uuid.uuid4())
    payload = b"{}"
    digest = hashlib.sha256(payload).hexdigest()
    cases = (
        (
            {
                "canonical_bytes": bytearray(payload),
                "canonical_sha256": digest,
                "reservation": {"byte_count": 2},
            },
            TypeError,
        ),
        (
            {
                "canonical_bytes": payload,
                "canonical_sha256": "0" * 64,
                "reservation": {"byte_count": 2},
            },
            ValueError,
        ),
        (
            {
                "canonical_bytes": payload,
                "canonical_sha256": digest,
                "reservation": {"artifact_rows": 1},
            },
            ValueError,
        ),
        (
            {
                "canonical_bytes": payload,
                "canonical_sha256": digest,
                "reservation": {"byte_count": 2, "unsupported": 1},
            },
            ValueError,
        ),
    )
    for values, exception in cases:
        with pytest.raises(exception):
            with database.transaction() as connection:
                registry.accept_job_input(
                    connection,
                    job_id=job_id,
                    schema_id="ProbeRequest",
                    origin="locally_created",
                    **values,
                )


def test_artifact_read_boundaries_cover_missing_sealed_and_corrupt_content(
    tmp_path: Path,
) -> None:
    database, registry = _registry(tmp_path)
    _api_error(lambda: registry.get_artifact(str(uuid.uuid4())), "NOT_FOUND")
    descriptor = registry.register_stream(
        b"sealed",
        str(uuid.uuid4()),
        artifact_type="test_blob",
        display_name="sealed.bin",
        media_type="application/octet-stream",
        preview_policy="download_only",
        origin="imported",
    )
    registry._sealed_sha256 = frozenset({descriptor["sha256"]})
    _api_error(
        lambda: registry.assert_content_readable(descriptor["artifact_id"]),
        "STATE_CONFLICT",
        "SEALED_SPLIT_REQUIRES_TOKEN",
    )
    assert registry.assert_content_readable(
        descriptor["artifact_id"], allow_sealed_internal=True
    ) == descriptor
    path = registry.verified_artifact_path(
        descriptor["artifact_id"], allow_sealed_internal=True
    )
    assert path.read_bytes() == b"sealed"
    path.unlink()
    _api_error(
        lambda: registry.read_artifact_bytes(
            descriptor["artifact_id"], allow_sealed_internal=True
        ),
        "STORAGE_UNAVAILABLE",
        "STORAGE_CORRUPT",
    )
    assert database.read_only


def test_sealed_digest_validation_and_recovery_detect_size_tampering(
    tmp_path: Path,
) -> None:
    database, registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        registry.is_sealed_digest("BAD")

    descriptor = registry.register_stream(
        b"original",
        str(uuid.uuid4()),
        artifact_type="test_blob",
        display_name="original.bin",
        media_type="application/octet-stream",
        preview_policy="download_only",
        origin="imported",
    )
    path = registry.root.joinpath(*descriptor["relative_path"].split("/"))
    with path.open("ab") as stream:
        stream.write(b"-larger")
    report = registry.recover()
    assert report.failed_check == "STORAGE_CORRUPT"
    assert database.read_only


def test_registered_path_and_recovery_fail_closed_for_unsafe_staging(
    tmp_path: Path,
) -> None:
    database, registry = _registry(tmp_path / "path")
    _api_error(
        lambda: registry._registered_path(
            {"sha256": "a" * 64, "relative_path": "objects/aa/wrong/path"}
        ),
        "STORAGE_UNAVAILABLE",
        "STORAGE_CORRUPT",
    )
    assert database.read_only
    assert registry.recover().failed_check == "STORAGE_CORRUPT"

    database, registry = _registry(tmp_path / "owner-file")
    jobs = registry.root / "jobs"
    jobs.mkdir(mode=0o700)
    (jobs / "not-a-directory").write_bytes(b"unsafe")
    report = registry.recover()
    assert not report.storage_writable
    assert report.failed_check == "STORAGE_CORRUPT"
    assert database.read_only

    _, registry = _registry(tmp_path / "no-staging")
    owner = registry.root / "jobs" / str(uuid.uuid4())
    owner.mkdir(parents=True)
    report = registry.recover()
    assert report.storage_writable
    assert report.quarantined_entries == 0

    database, registry = _registry(tmp_path / "linked-entry")
    staging = registry.create_staging_dir(str(uuid.uuid4()))
    outside = tmp_path / "outside-entry"
    outside.write_bytes(b"outside")
    (staging / "linked.partial").symlink_to(outside)
    report = registry.recover()
    assert report.failed_check == "STORAGE_CORRUPT"
    assert database.read_only


@pytest.mark.parametrize("limit", (True, 0, 101, 1.5))
def test_list_limit_rejects_non_integer_or_out_of_range_values(
    tmp_path: Path, limit: object
) -> None:
    _, registry = _registry(tmp_path)
    _api_error(
        lambda: registry.list_artifacts(limit=limit),  # type: ignore[arg-type]
        "VALIDATION_FAILED",
        "SCHEMA_INVALID",
    )


def test_artifact_list_filters_are_validated_and_bound_to_cursor(tmp_path: Path) -> None:
    _, registry = _registry(tmp_path)
    for index, origin in enumerate(("imported", "locally_created", "imported")):
        registry.register_stream(
            f"payload-{index}".encode(),
            str(uuid.uuid4()),
            artifact_type="test_blob",
            display_name=f"payload-{index}.bin",
            media_type="application/octet-stream",
            preview_policy="download_only",
            origin=origin,
        )
    with pytest.raises(ValueError, match="unsupported artifact"):
        registry.list_artifacts(filters={"bad": "filter"})
    first = registry.list_artifacts(limit=1, filters={"origin": "imported"})
    assert len(first["items"]) == 1
    assert first["next_cursor"]
    second = registry.list_artifacts(
        limit=1,
        cursor=first["next_cursor"],
        filters={"origin": "imported"},
    )
    assert len(second["items"]) == 1
    _api_error(
        lambda: registry.list_artifacts(
            limit=1,
            cursor=first["next_cursor"],
            filters={"origin": "locally_created"},
        ),
        "VALIDATION_FAILED",
        "SEMANTIC_INVALID",
    )


def test_record_registry_missing_table_kind_filters_and_cursor_pagination(
    tmp_path: Path,
) -> None:
    database, registry = _registry(tmp_path)
    assert registry.list_records("jobs") == {"items": [], "next_cursor": None}
    with pytest.raises(ValueError, match="unsupported registry kind"):
        registry.get_record("unknown", str(uuid.uuid4()))
    with pytest.raises(ValueError, match="unsupported registry list filter"):
        registry.list_records("runs", filters={"backend": "cpu"})

    run_ids = [str(uuid.uuid4()) for _ in range(3)]
    _insert_run(database, run_ids[0], created_at="2026-10-01T00:00:00.000Z")
    _insert_run(database, run_ids[1], created_at="2026-10-01T00:00:01.000Z")
    _insert_run(
        database,
        run_ids[2],
        created_at="2026-10-01T00:00:02.000Z",
        operation="tiny_train",
    )
    assert registry.get_record("runs", run_ids[0])["run_id"] == run_ids[0]
    _api_error(lambda: registry.get_record("runs", str(uuid.uuid4())), "NOT_FOUND")
    first = registry.list_records(
        "runs", limit=1, filters={"operation": "tokenizer_train"}
    )
    assert len(first["items"]) == 1
    assert first["next_cursor"]
    second = registry.list_records(
        "runs",
        limit=1,
        cursor=first["next_cursor"],
        filters={"operation": "tokenizer_train"},
    )
    assert len(second["items"]) == 1
    assert second["items"][0]["run_id"] != first["items"][0]["run_id"]


@pytest.mark.parametrize("record_json", ("not-json", "[]"))
def test_get_record_treats_malformed_persisted_json_as_corruption(
    tmp_path: Path, record_json: str
) -> None:
    database, registry = _registry(tmp_path)
    run_id = str(uuid.uuid4())
    _insert_run(
        database,
        run_id,
        created_at="2026-10-01T00:00:00.000Z",
        record_json=record_json,
    )
    _api_error(
        lambda: registry.get_record("runs", run_id),
        "STORAGE_UNAVAILABLE",
        "STORAGE_CORRUPT",
    )
    assert database.read_only


@pytest.mark.parametrize("record_json", ("not-json", "[]"))
def test_list_records_treats_malformed_persisted_json_as_corruption(
    tmp_path: Path, record_json: str
) -> None:
    database, registry = _registry(tmp_path)
    _insert_run(
        database,
        str(uuid.uuid4()),
        created_at="2026-10-01T00:00:00.000Z",
        record_json=record_json,
    )
    _api_error(
        lambda: registry.list_records("runs"),
        "STORAGE_UNAVAILABLE",
        "STORAGE_CORRUPT",
    )
    assert database.read_only
