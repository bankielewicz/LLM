from __future__ import annotations

import hashlib
import io
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from llm_foundations_companion.database import Database
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.training_store import TrainingStore


_NOW = "2026-10-01T00:00:00.000Z"


def _runtime(tmp_path: Path) -> tuple[Database, Registry, TrainingStore]:
    root = tmp_path / "root"
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    store = TrainingStore(
        database,
        registry,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="1" * 64,
        companion_source_revision="2" * 40,
    )
    return database, registry, store


def _install_checkpoint(database: Database, checkpoint_id: str) -> None:
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO checkpoints(
                checkpoint_id, format, backend, origin, run_id, job_id,
                manifest_sha256, source_identity_json, created_at, updated_at,
                deleted_at, record_json
            ) VALUES (?, 'tiny_v2_portable', 'tiny_v2', 'locally_created',
                      NULL, NULL, ?, NULL, ?, ?, NULL, '{}')
            """,
            (checkpoint_id, "3" * 64, _NOW, _NOW),
        )


def _generation_request(checkpoint_id: str) -> dict[str, Any]:
    return {
        "operation": "generate",
        "checkpoint_id": checkpoint_id,
        "prompt": "hello",
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 17,
    }


def _checkpoint(checkpoint_id: str) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint_id,
        "model_id": str(uuid.uuid4()),
        "manifest_sha256": "3" * 64,
        "checkpoint_sha256": "3" * 64,
        "tokenizer_sha256": "4" * 64,
        "config_sha256": "5" * 64,
        "context": 8,
        "architecture_profile_id": "tiny-v2-standard-v1",
        "runtime_profile": "wsl-cpu",
        "device": "cpu",
    }


def _preview_value(
    checkpoint: dict[str, Any], generation_request: dict[str, Any]
) -> dict[str, Any]:
    subject_identity = {
        "backend": "tiny",
        "model_id": checkpoint["model_id"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "runtime_profile": "wsl-cpu",
        "device": "cpu",
        "context_limit": checkpoint["context"],
    }
    return {
        "backend": "tiny",
        "subject_identity_sha256": hashlib.sha256(
            canonical_json(subject_identity)
        ).hexdigest(),
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "template_sha256": None,
        "device": "cpu",
        "original_input_token_count": 1,
        "input_token_ids": [1],
        "input_token_count": 1,
        "serialized_text": "hello",
        "dropped_message_indices": [],
        "cropped_input_tokens": 0,
        "effective_context_budget": checkpoint["context"],
        "canonical_generation_request_sha256": hashlib.sha256(
            canonical_json(generation_request)
        ).hexdigest(),
    }


def _register_preview(
    registry: Registry,
    payload: bytes,
    *,
    artifact_type: str = "context_preview",
) -> dict[str, Any]:
    return registry.register_stream(
        io.BytesIO(payload),
        str(uuid.uuid4()),
        artifact_type=artifact_type,
        display_name="preview.json",
        media_type="application/json",
        preview_policy="metadata_only",
        origin="locally_created",
    )


def _request_with_preview(
    generation_request: dict[str, Any], descriptor: dict[str, Any]
) -> dict[str, Any]:
    return {
        **generation_request,
        "preview_artifact_id": descriptor["artifact_id"],
        "context_preview_digest": descriptor["sha256"],
    }


def _job_count(database: Database) -> int:
    with database.read() as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if exists is None:
            return 0
        return int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])


def _assert_preview_stale(exc: ApiError) -> None:
    assert exc.code == "VALIDATION_FAILED"
    assert exc.reason_code == "CONTEXT_PREVIEW_STALE"
    assert [dict(item) for item in exc.field_errors] == [
        {
            "field_path": "/preview_artifact_id",
            "message": "The context preview no longer matches this request.",
        }
    ]


def _valid_preview_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Database, Registry, TrainingStore, dict[str, Any], dict[str, Any]]:
    database, registry, store = _runtime(tmp_path)
    checkpoint_id = str(uuid.uuid4())
    _install_checkpoint(database, checkpoint_id)
    checkpoint = _checkpoint(checkpoint_id)
    generation_request = _generation_request(checkpoint_id)
    preview = _register_preview(
        registry, canonical_json(_preview_value(checkpoint, generation_request))
    )
    monkeypatch.setattr(
        store, "_checkpoint_resolution", lambda _checkpoint_id: (checkpoint, [])
    )
    return database, registry, store, generation_request, preview


def test_missing_preview_with_live_checkpoint_is_stale_without_job(
    tmp_path: Path,
) -> None:
    database, _, store = _runtime(tmp_path)
    checkpoint_id = str(uuid.uuid4())
    _install_checkpoint(database, checkpoint_id)
    request = {
        **_generation_request(checkpoint_id),
        "preview_artifact_id": str(uuid.uuid4()),
        "context_preview_digest": "6" * 64,
    }

    with pytest.raises(ApiError) as caught:
        store.resolve(request)

    _assert_preview_stale(caught.value)
    assert _job_count(database) == 0


def test_soft_deleted_preview_with_live_checkpoint_is_stale_without_job(
    tmp_path: Path,
) -> None:
    database, registry, store = _runtime(tmp_path)
    checkpoint_id = str(uuid.uuid4())
    _install_checkpoint(database, checkpoint_id)
    preview = _register_preview(registry, b"{}")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE artifacts SET deleted_at = ? WHERE artifact_id = ?",
            (_NOW, preview["artifact_id"]),
        )

    with pytest.raises(ApiError) as caught:
        store.resolve(_request_with_preview(_generation_request(checkpoint_id), preview))

    _assert_preview_stale(caught.value)
    assert _job_count(database) == 0


def test_missing_checkpoint_keeps_reference_error_when_preview_exists(
    tmp_path: Path,
) -> None:
    database, registry, store = _runtime(tmp_path)
    checkpoint_id = str(uuid.uuid4())
    preview = _register_preview(registry, b"{}")

    with pytest.raises(ApiError) as caught:
        store.resolve(_request_with_preview(_generation_request(checkpoint_id), preview))

    assert caught.value.reason_code == "REFERENCE_MISSING"
    assert [item["field_path"] for item in caught.value.field_errors] == [
        "/checkpoint_id"
    ]
    assert _job_count(database) == 0


def test_missing_checkpoint_and_preview_keep_all_reference_errors(
    tmp_path: Path,
) -> None:
    database, _, store = _runtime(tmp_path)
    request = {
        **_generation_request(str(uuid.uuid4())),
        "preview_artifact_id": str(uuid.uuid4()),
        "context_preview_digest": "6" * 64,
    }

    with pytest.raises(ApiError) as caught:
        store.resolve(request)

    assert caught.value.reason_code == "REFERENCE_MISSING"
    assert [item["field_path"] for item in caught.value.field_errors] == [
        "/checkpoint_id",
        "/preview_artifact_id",
    ]
    assert _job_count(database) == 0


def test_preview_disappearing_during_verified_read_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, registry, store, generation_request, preview = _valid_preview_runtime(
        tmp_path, monkeypatch
    )

    def missing(_artifact_id: str, *, allow_sealed_internal: bool = False) -> bytes:
        del allow_sealed_internal
        raise ApiError("NOT_FOUND", "Artifact was not found.")

    monkeypatch.setattr(registry, "read_artifact_bytes", missing)
    with pytest.raises(ApiError) as caught:
        store.resolve(_request_with_preview(generation_request, preview))

    _assert_preview_stale(caught.value)
    assert _job_count(database) == 0


def test_preview_disappearing_during_input_materialization_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, registry, store, generation_request, preview = _valid_preview_runtime(
        tmp_path, monkeypatch
    )
    original_get = registry.get_artifact
    calls = 0

    def disappearing(artifact_id: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ApiError("NOT_FOUND", "Artifact was not found.")
        return original_get(artifact_id)

    monkeypatch.setattr(registry, "get_artifact", disappearing)
    with pytest.raises(ApiError) as caught:
        store.resolve(_request_with_preview(generation_request, preview))

    assert calls == 3
    _assert_preview_stale(caught.value)
    assert _job_count(database) == 0


def test_preview_storage_corruption_is_not_reclassified_as_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, registry, store, generation_request, preview = _valid_preview_runtime(
        tmp_path, monkeypatch
    )
    path = registry.root.joinpath(*str(preview["relative_path"]).split("/"))
    payload = path.read_bytes()
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    with pytest.raises(ApiError) as caught:
        store.resolve(_request_with_preview(generation_request, preview))

    assert caught.value.code == "STORAGE_UNAVAILABLE"
    assert caught.value.reason_code == "STORAGE_CORRUPT"
    assert _job_count(database) == 0


def test_non_not_found_registry_error_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, registry, store, generation_request, preview = _valid_preview_runtime(
        tmp_path, monkeypatch
    )
    expected = ApiError(
        "STATE_CONFLICT", reason_code="SEALED_SPLIT_REQUIRES_TOKEN"
    )

    def conflict(_artifact_id: str, *, allow_sealed_internal: bool = False) -> bytes:
        del allow_sealed_internal
        raise expected

    monkeypatch.setattr(registry, "read_artifact_bytes", conflict)
    with pytest.raises(ApiError) as caught:
        store.resolve(_request_with_preview(generation_request, preview))

    assert caught.value is expected
    assert _job_count(database) == 0


def test_wrong_artifact_type_is_rejected_before_content_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, registry, store = _runtime(tmp_path)
    checkpoint_id = str(uuid.uuid4())
    _install_checkpoint(database, checkpoint_id)
    checkpoint = _checkpoint(checkpoint_id)
    generation_request = _generation_request(checkpoint_id)
    preview = _register_preview(registry, b"arbitrary bytes", artifact_type="diagnostic")
    monkeypatch.setattr(
        store, "_checkpoint_resolution", lambda _checkpoint_id: (checkpoint, [])
    )

    def unexpected_read(
        _artifact_id: str, *, allow_sealed_internal: bool = False
    ) -> bytes:
        del allow_sealed_internal
        pytest.fail("wrong artifact type must be rejected before content read")

    monkeypatch.setattr(registry, "read_artifact_bytes", unexpected_read)
    with pytest.raises(ApiError) as caught:
        store.resolve(_request_with_preview(generation_request, preview))

    _assert_preview_stale(caught.value)
    assert _job_count(database) == 0
