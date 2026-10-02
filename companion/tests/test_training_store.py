from __future__ import annotations

import io
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from llm_foundations_companion.admission import AdmissionPlanner
from llm_foundations_companion.database import Database
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.training_store import (
    AdmissionPlan,
    TrainingStore,
    tiny_parameter_count,
)


def _store(
    tmp_path: Path, *, identities: list[str] | None = None
) -> tuple[Database, Registry, TrainingStore]:
    root = tmp_path / "root"
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    values = iter(identities or [])
    store = TrainingStore(
        database,
        registry,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="1" * 64,
        companion_source_revision="2" * 40,
        uuid_factory=lambda: next(values),
    )
    return database, registry, store


def test_parameter_count_matches_int003_and_rejects_boolean() -> None:
    untied = tiny_parameter_count(
        vocab_size=257, context=64, width=64, layers=2, tied=False
    )
    tied = tiny_parameter_count(
        vocab_size=257, context=64, width=64, layers=2, tied=True
    )
    assert untied == 137_088
    assert untied - tied == 257 * 64
    with pytest.raises(ValueError):
        tiny_parameter_count(
            vocab_size=True, context=64, width=64, layers=2, tied=False
        )


def test_training_store_import_is_framework_free(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[1] / "src"
    script = (
        "import sys;"
        f"sys.path.insert(0, {str(source_root)!r});"
        "import llm_foundations_companion.admission;"
        "import llm_foundations_companion.training_store;"
        "assert 'torch' not in sys.modules;"
        "assert 'safetensors' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        cwd=tmp_path,
    )


def test_tokenizer_admission_uses_only_eligible_train_text(tmp_path: Path) -> None:
    run_id, tokenizer_id = str(uuid.uuid4()), str(uuid.uuid4())
    database, registry, store = _store(
        tmp_path, identities=[run_id, tokenizer_id]
    )
    dataset_id = str(uuid.uuid4())
    split = registry.register_stream(
        io.BytesIO(b'{"record_id":"b","text":"three"}\n'),
        str(uuid.uuid4()),
        artifact_type="dataset_split",
        display_name="train.jsonl",
        media_type="application/x-ndjson",
        preview_policy="text",
        origin="locally_created",
    )
    now = "2026-10-01T00:00:00.000Z"
    manifest_sha256 = "3" * 64
    manifest = (
        "{"
        '"format":"llm-foundations-dataset-v1",'
        f'"dataset_id":"{dataset_id}",'
        '"name":"tiny",'
        '"record_format":"document_text_v1",'
        '"origin":"locally_created",'
        '"splits":{"train":{'
        f'"artifact_id":"{split["artifact_id"]}",'
        f'"sha256":"{split["sha256"]}",'
        f'"records":1,"utf8_bytes":{split["size_bytes"]},"sealed":false'
        "}},"
        f'"audit_artifact_id":"{split["artifact_id"]}",'
        f'"created_at":"{now}",'
        f'"manifest_sha256":"{manifest_sha256}",'
        '"eligibility":"eligible"'
        "}"
    )
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO datasets(
                dataset_id, format, name, record_format, origin,
                audit_artifact_id, created_at, provenance_note,
                manifest_sha256, eligibility, manifest_json,
                source_identity_json, updated_at
            ) VALUES (?, 'llm-foundations-dataset-v1', 'tiny', 'document_text_v1',
                      'locally_created', ?, ?, NULL, ?, 'eligible', ?, NULL, ?)
            """,
            (
                dataset_id,
                split["artifact_id"],
                now,
                manifest_sha256,
                manifest,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO dataset_splits(
                dataset_id, split_name, artifact_id, sha256,
                records, utf8_bytes, sealed
            ) VALUES (?, 'train', ?, ?, 1, ?, 0)
            """,
            (
                dataset_id,
                split["artifact_id"],
                split["sha256"],
                split["size_bytes"],
            ),
        )
    plan = store.resolve(
        {
            "operation": "tokenizer_train",
            "dataset_id": dataset_id,
            "vocab_size": 257,
            "seed": 17,
            "tokenizer_profile_id": "byte-v1",
        }
    )
    assert plan.run_id == run_id
    assert plan.tokenizer_id == tokenizer_id
    assert plan.resolved["dataset_manifest_sha256"] == manifest_sha256
    assert [item["role"] for item in plan.resolved["_inputs"]] == ["dataset.train"]
    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE datasets
            SET eligibility = 'audit_only',
                manifest_json = ?
            WHERE dataset_id = ?
            """,
            (manifest.replace('"eligibility":"eligible"', '"eligibility":"audit_only"'), dataset_id),
        )
    with pytest.raises(ApiError) as caught:
        store.resolve(
            {
                "operation": "tokenizer_train",
                "dataset_id": dataset_id,
                "vocab_size": 257,
                "seed": 17,
                "tokenizer_profile_id": "byte-v1",
            }
        )
    assert caught.value.code == "VALIDATION_FAILED"
    assert caught.value.reason_code == "DATASET_INELIGIBLE"


def test_reference_stage_collects_sorted_errors_before_semantics_without_mutation(
    tmp_path: Path,
) -> None:
    database, _, store = _store(tmp_path)
    request = {
        "operation": "tiny_train",
        "dataset_id": str(uuid.uuid4()),
        "tokenizer_id": str(uuid.uuid4()),
        "steps": 2,
        "eval_every": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
        "seed": 7,
        "context": 8,
        "width": 17,
        "heads": 2,
        "layers": 1,
        "architecture_profile_id": "tiny-v2-standard-v1",
    }
    with database.read() as connection:
        before = tuple(
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("job_operation_contexts", "reservations")
        )
    with pytest.raises(ApiError) as caught:
        store.resolve(request)
    assert caught.value.code == "VALIDATION_FAILED"
    assert caught.value.reason_code == "REFERENCE_MISSING"
    assert [item["field_path"] for item in caught.value.field_errors] == [
        "/dataset_id",
        "/tokenizer_id",
    ]
    with database.read() as connection:
        after = tuple(
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("job_operation_contexts", "reservations")
        )
    assert after == before


@pytest.mark.parametrize("mismatch", ["tokenizer", "config", "architecture", "vocab"])
def test_valid_declared_checkpoint_identity_mismatch_is_not_storage_corruption(
    tmp_path: Path, mismatch: str
) -> None:
    _, registry, store = _store(tmp_path)
    tokenizer_value = {"format": "teaching-byte-bpe-v1", "merges": []}
    tokenizer_bytes = json.dumps(tokenizer_value, indent=2).encode("utf-8")
    tokenizer = registry.register_stream(
        io.BytesIO(tokenizer_bytes),
        str(uuid.uuid4()),
        artifact_type="checkpoint_file",
        display_name="tokenizer.json",
        media_type="application/json",
        preview_policy="metadata_only",
        origin="locally_created",
    )
    config_value = {
        "format": "tiny-v2-config-v1",
        "architecture_profile_id": "tiny-v2-standard-v1",
        "vocab_size": 258 if mismatch == "vocab" else 257,
        "context": 8,
        "width": 16,
        "heads": 1,
        "layers": 1,
    }
    config = registry.register_stream(
        io.BytesIO(json.dumps(config_value).encode("utf-8")),
        str(uuid.uuid4()),
        artifact_type="checkpoint_file",
        display_name="config.json",
        media_type="application/json",
        preview_policy="metadata_only",
        origin="locally_created",
    )
    tokenizer_fingerprint = hashlib.sha256(
        json.dumps(tokenizer_value, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint = {
        "model_identity": {
            "kind": "tiny_v2",
            "architecture_profile_id": (
                "tiny-v2-weight-tied-v1"
                if mismatch == "architecture"
                else "tiny-v2-standard-v1"
            ),
            "tokenizer_sha256": "a" * 64 if mismatch == "tokenizer" else tokenizer_fingerprint,
            "config_sha256": "b" * 64 if mismatch == "config" else config["sha256"],
        },
        "_files": {
            "tokenizer.json": tokenizer,
            "config.json": config,
        },
    }
    with pytest.raises(ApiError) as caught:
        store._checkpoint_tiny_identity(checkpoint)
    assert caught.value.code == "VALIDATION_FAILED"
    assert caught.value.reason_code == "CHECKPOINT_INCOMPATIBLE"


def test_registered_checkpoint_byte_corruption_remains_storage_failure(tmp_path: Path) -> None:
    _, registry, store = _store(tmp_path)
    tokenizer_value = {"format": "teaching-byte-bpe-v1", "merges": []}
    tokenizer_bytes = json.dumps(tokenizer_value, indent=2).encode("utf-8")
    tokenizer = registry.register_stream(
        io.BytesIO(tokenizer_bytes),
        str(uuid.uuid4()),
        artifact_type="checkpoint_file",
        display_name="tokenizer.json",
        media_type="application/json",
        preview_policy="metadata_only",
        origin="locally_created",
    )
    config_value = {
        "format": "tiny-v2-config-v1",
        "architecture_profile_id": "tiny-v2-standard-v1",
        "vocab_size": 257,
        "context": 8,
        "width": 16,
        "heads": 1,
        "layers": 1,
    }
    config = registry.register_stream(
        io.BytesIO(json.dumps(config_value).encode("utf-8")),
        str(uuid.uuid4()),
        artifact_type="checkpoint_file",
        display_name="config.json",
        media_type="application/json",
        preview_policy="metadata_only",
        origin="locally_created",
    )
    tokenizer_fingerprint = hashlib.sha256(
        json.dumps(tokenizer_value, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint = {
        "model_identity": {
            "kind": "tiny_v2",
            "architecture_profile_id": "tiny-v2-standard-v1",
            "tokenizer_sha256": tokenizer_fingerprint,
            "config_sha256": config["sha256"],
        },
        "_files": {"tokenizer.json": tokenizer, "config.json": config},
    }
    config_path = registry.root.joinpath(*str(config["relative_path"]).split("/"))
    config_path.write_bytes(config_path.read_bytes().replace(b'"context": 8', b'"context": 9'))
    with pytest.raises(ApiError) as caught:
        store._checkpoint_tiny_identity(checkpoint)
    assert caught.value.code == "STORAGE_UNAVAILABLE"
    assert caught.value.reason_code == "STORAGE_CORRUPT"


def test_applied_context_preview_has_closed_capability_response_without_checkpoint_key(
    tmp_path: Path,
) -> None:
    database, _, store = _store(tmp_path)
    model_id = str(uuid.uuid4())
    now = "2026-10-01T00:00:00.000Z"
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO models(
                model_id, format, backend, origin, creating_job_id,
                source_identity_json, created_at, updated_at, record_json
            ) VALUES (?, 'prepared-model-v1', 'applied', 'locally_created',
                      NULL, NULL, ?, ?, '{}')
            """,
            (model_id, now, now),
        )
    with pytest.raises(ApiError) as caught:
        store.resolve(
            {
                "operation": "context_preview",
                "backend": "chat",
                "subject": {"model_id": model_id},
                "messages": [{"role": "user", "content": "Hello"}],
                "max_new_tokens": 8,
                "temperature": 0,
                "top_p": 1,
                "seed": 4,
            }
        )
    assert caught.value.code == "CAPABILITY_UNAVAILABLE"


def test_job_context_is_immutable_and_snapshot_is_exact_and_repeatable(
    tmp_path: Path,
) -> None:
    database, registry, store = _store(tmp_path)
    job_id = str(uuid.uuid4())
    artifact = registry.register_stream(
        io.BytesIO(b"verified input"),
        str(uuid.uuid4()),
        artifact_type="test_input",
        display_name="input.bin",
        media_type="application/octet-stream",
        preview_policy="download_only",
        origin="locally_created",
    )
    reservation = {
        "byte_count": 1,
        "artifact_rows": 1,
        "dataset_rows": 0,
        "run_rows": 0,
        "model_rows": 0,
        "checkpoint_rows": 0,
    }
    plan = AdmissionPlan(
        operation="context_preview",
        reservation=reservation,
        resolved={
            "request": {"operation": "context_preview"},
            "checkpoint": {"checkpoint_id": str(uuid.uuid4())},
            "_inputs": [
                {
                    "role": "checkpoint.config.json",
                    "artifact_id": artifact["artifact_id"],
                    "sha256": artifact["sha256"],
                    "size_bytes": artifact["size_bytes"],
                    "sealed": False,
                }
            ],
        },
        run_id=None,
        model_id=None,
        tokenizer_id=None,
        checkpoint_ids=(),
    )
    with database.transaction() as connection:
        store.create_job_context_in(
            connection,
            job_id=job_id,
            request_sha256="4" * 64,
            plan=plan,
        )
    context = store.get_job_context(job_id)
    assert context.reservation == reservation
    assert context.resolved["request"] == {"operation": "context_preview"}
    first = store.materialize_worker_snapshot(job_id)
    second = store.materialize_worker_snapshot(job_id)
    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    input_path = first.path.parent / "staging" / first.value["inputs"][0]["path"]
    assert input_path.read_bytes() == b"verified input"
    assert first.value["resolved"] == {
        "request": {"operation": "context_preview"},
        "checkpoint": plan.resolved["checkpoint"],
    }
    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction() as connection:
            connection.execute(
                "UPDATE job_operation_contexts SET operation = 'generate' WHERE job_id = ?",
                (job_id,),
            )
    input_path.write_bytes(b"tampered data!")
    with pytest.raises(ApiError) as caught:
        store.materialize_worker_snapshot(job_id)
    assert caught.value.reason_code == "STORAGE_CORRUPT"


def test_real_s2_admission_requires_source_revision(tmp_path: Path) -> None:
    database, registry, _ = _store(tmp_path)
    store = TrainingStore(
        database,
        registry,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="1" * 64,
        companion_source_revision=None,
    )
    with pytest.raises(ApiError) as caught:
        store.resolve({"operation": "tokenizer_train"})
    assert caught.value.code == "CAPABILITY_UNAVAILABLE"


def test_admission_dispatches_s2_and_retains_fallback_plans() -> None:
    class Resolved:
        def resolve(self, request):
            return ("resolved", request["operation"])

    class Fallback:
        def plan(self, request):
            return {"byte_count": 7, "operation": request["operation"]}

    planner = AdmissionPlanner(Resolved(), fallback=Fallback())
    assert planner.resolve({"operation": "evaluate"}) == ("resolved", "evaluate")
    assert planner.resolve({"operation": "model_prepare"}) is None
    assert planner.plan({"operation": "model_prepare"}) == {
        "byte_count": 7,
        "operation": "model_prepare",
    }
