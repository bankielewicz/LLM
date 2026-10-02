from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import llm_foundations_companion.operation_store as operation_module
from llm_foundations_companion.database import Database
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.operation_store import (
    OperationStore,
    OperationStoreError,
    PreparedArtifact,
    PreparedCheckpoint,
    _artifact_metadata,
    _checkpoint_artifact_metadata,
    _mapping_bytes,
)
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.tiny_v2.tokenizer import Tokenizer, serialize_tokenizer
from llm_foundations_companion.training_store import (
    AdmissionPlan,
    JobContext,
    TrainingStore,
)


def _runtime(tmp_path: Path) -> tuple[Database, Registry, TrainingStore, OperationStore]:
    root = tmp_path / str(uuid.uuid4())
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    training = TrainingStore(
        database,
        registry,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="1" * 64,
        companion_source_revision="2" * 40,
    )
    with database.transaction() as connection:
        connection.execute(
            "CREATE TABLE jobs(job_id TEXT PRIMARY KEY, record_json TEXT NOT NULL)"
        )
    return database, registry, training, OperationStore(database, registry, training)


def _reservation(
    *, byte_count: int = 1_000_000, artifact_rows: int = 16, checkpoints: int = 0
) -> dict[str, int]:
    return {
        "byte_count": byte_count,
        "artifact_rows": artifact_rows,
        "dataset_rows": 0,
        "run_rows": 1,
        "model_rows": 1 if checkpoints else 0,
        "checkpoint_rows": checkpoints,
    }


def _tokenizer_plan() -> AdmissionPlan:
    dataset_id = str(uuid.uuid4())
    return AdmissionPlan(
        operation="tokenizer_train",
        reservation=_reservation(),
        resolved={
            "request": {
                "operation": "tokenizer_train",
                "dataset_id": dataset_id,
                "tokenizer_profile_id": "byte-v1",
                "vocab_size": 257,
                "seed": 17,
            },
            "dataset_manifest_sha256": "4" * 64,
            "dataset_bindings": [
                {
                    "dataset_id": dataset_id,
                    "split": "train",
                    "artifact_id": str(uuid.uuid4()),
                    "sha256": "5" * 64,
                }
            ],
            "_inputs": [],
        },
        run_id=str(uuid.uuid4()),
        model_id=None,
        tokenizer_id=str(uuid.uuid4()),
        checkpoint_ids=(),
    )


def _training_plan(*, reservation: dict[str, int] | None = None) -> AdmissionPlan:
    dataset_id = str(uuid.uuid4())
    request = {
        "operation": "tiny_train",
        "dataset_id": dataset_id,
        "tokenizer_id": str(uuid.uuid4()),
        "architecture_profile_id": "tiny-v2-standard-v1",
        "steps": 1,
        "eval_every": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
        "context": 8,
        "width": 16,
        "heads": 1,
        "layers": 1,
        "seed": 17,
    }
    return AdmissionPlan(
        operation="tiny_train",
        reservation=reservation or _reservation(checkpoints=1),
        resolved={
            "request": request,
            "dataset_manifest_sha256": "6" * 64,
            "dataset_bindings": [
                {
                    "dataset_id": dataset_id,
                    "split": split,
                    "artifact_id": str(uuid.uuid4()),
                    "sha256": digest * 64,
                }
                for split, digest in (("train", "7"), ("validation", "8"))
            ],
            "tokenizer": {
                "tokenizer_id": request["tokenizer_id"],
                "tokenizer_type": "byte",
                "vocab_size": 257,
                "tokenizer_sha256": "9" * 64,
                "artifact_id": str(uuid.uuid4()),
            },
            "parameter_count": 10,
            "requested_final_step": 1,
            "_inputs": [],
        },
        run_id=str(uuid.uuid4()),
        model_id=str(uuid.uuid4()),
        tokenizer_id=None,
        checkpoint_ids=(str(uuid.uuid4()),),
    )


def _install_context(
    database: Database,
    training: TrainingStore,
    plan: AdmissionPlan,
) -> tuple[str, JobContext]:
    job_id = str(uuid.uuid4())
    requested = plan.resolved.get("requested_final_step")
    with database.transaction() as connection:
        training.create_job_context_in(
            connection,
            job_id=job_id,
            request_sha256="3" * 64,
            plan=plan,
        )
        connection.execute(
            """
            INSERT INTO reservations(
                reservation_id, owner_kind, owner_id, byte_count,
                artifact_rows, dataset_rows, run_rows, model_rows,
                checkpoint_rows, state, created_at, released_at
            ) VALUES (?, 'job', ?, ?, ?, ?, ?, ?, ?, 'held', ?, NULL)
            """,
            (
                job_id,
                job_id,
                plan.reservation["byte_count"],
                plan.reservation["artifact_rows"],
                plan.reservation["dataset_rows"],
                plan.reservation["run_rows"],
                plan.reservation["model_rows"],
                plan.reservation["checkpoint_rows"],
                "2026-10-01T00:00:00.000Z",
            ),
        )
        connection.execute(
            "INSERT INTO jobs(job_id, record_json) VALUES (?, ?)",
            (
                job_id,
                canonical_json(
                    {
                        "job_id": job_id,
                        "run_id": plan.run_id,
                        "step": 0,
                        "requested_final_step": requested,
                        "checkpoint_boundary": None,
                    }
                ).decode("utf-8"),
            ),
        )
    return job_id, training.get_job_context(job_id)


def _assert_storage_corrupt(exc: ApiError) -> None:
    assert exc.code == "STORAGE_UNAVAILABLE"
    assert exc.reason_code == "STORAGE_CORRUPT"


def test_malformed_worker_json_is_a_protocol_error() -> None:
    with pytest.raises(OperationStoreError) as caught:
        _mapping_bytes(b'{', "worker output")
    assert caught.value.code == "WORKER_PROTOCOL_ERROR"


def test_malformed_worker_jsonl_is_a_protocol_error(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{\n')
    with pytest.raises(OperationStoreError) as caught:
        OperationStore._jsonl(path, "metrics")
    assert caught.value.code == "WORKER_PROTOCOL_ERROR"


def test_artifact_metadata_and_jsonl_shapes_are_closed(tmp_path: Path) -> None:
    assert _artifact_metadata("tokenizer_json", "tokenizer.json")[2:] == (
        "application/json",
        "text",
    )
    assert _artifact_metadata("training_metrics", "metrics.jsonl")[2:] == (
        "application/x-ndjson",
        "text",
    )
    assert _checkpoint_artifact_metadata(str(uuid.uuid4()), "config.json")[2:] == (
        "application/json",
        "metadata_only",
    )
    with pytest.raises(OperationStoreError):
        _artifact_metadata("unknown", "unknown.bin")
    with pytest.raises(OperationStoreError):
        OperationStore._exact_proposal({"a": 1}, {"a", "b"}, "proposal")

    valid = tmp_path / "valid.jsonl"
    valid.write_bytes(b'{"a":1}\n{"a":2}\n')
    assert OperationStore._jsonl(valid, "rows") == [{"a": 1}, {"a": 2}]
    for index, raw in enumerate((b"", b"{}", b"{}\n\n", b"[]\n")):
        path = tmp_path / f"invalid-{index}.jsonl"
        path.write_bytes(raw)
        with pytest.raises(OperationStoreError) as caught:
            OperationStore._jsonl(path, "rows")
        assert caught.value.code == "WORKER_PROTOCOL_ERROR"


def test_output_capacity_maps_missing_and_exceeded_reservations(tmp_path: Path) -> None:
    database, _, _, store = _runtime(tmp_path)
    missing_id = str(uuid.uuid4())
    with pytest.raises(ApiError) as missing:
        store._assert_output_capacity(
            missing_id,
            proposed_bytes=0,
            proposed_rows=0,
            code=None,
            keep_terminal_capacity=False,
        )
    _assert_storage_corrupt(missing.value)

    plan = _tokenizer_plan()
    plan = replace(plan, reservation=_reservation(byte_count=0, artifact_rows=0))
    job_id, _ = _install_context(database, store.training_store, plan)
    with pytest.raises(ApiError) as public:
        store._assert_output_capacity(
            job_id,
            proposed_bytes=1,
            proposed_rows=1,
            code=None,
            keep_terminal_capacity=False,
        )
    assert public.value.code == "REGISTRY_LIMIT"
    assert public.value.reason_code == "ROOT_QUOTA_EXCEEDED"
    with pytest.raises(OperationStoreError) as worker:
        store._assert_output_capacity(
            job_id,
            proposed_bytes=1,
            proposed_rows=1,
            code="WORKER_PROTOCOL_ERROR",
            keep_terminal_capacity=False,
        )
    assert worker.value.code == "WORKER_PROTOCOL_ERROR"


@pytest.mark.parametrize("mode", ["probe", "low-space"])
def test_output_capacity_maps_disk_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    database, _, training, store = _runtime(tmp_path)
    job_id, _ = _install_context(database, training, _tokenizer_plan())
    if mode == "probe":
        def fail_probe(_path: Path) -> Any:
            raise OSError("unavailable")
        monkeypatch.setattr(operation_module.shutil, "disk_usage", fail_probe)
    else:
        monkeypatch.setattr(
            operation_module.shutil,
            "disk_usage",
            lambda _path: SimpleNamespace(free=0),
        )

    with pytest.raises(ApiError) as public:
        store._assert_output_capacity(
            job_id,
            proposed_bytes=1,
            proposed_rows=1,
            code=None,
            keep_terminal_capacity=False,
        )
    assert public.value.code == (
        "STORAGE_UNAVAILABLE" if mode == "probe" else "DISK_FULL"
    )
    with pytest.raises(OperationStoreError) as worker:
        store._assert_output_capacity(
            job_id,
            proposed_bytes=1,
            proposed_rows=1,
            code="CHECKPOINT_INCOMPATIBLE",
            keep_terminal_capacity=False,
        )
    assert worker.value.code == "CHECKPOINT_INCOMPATIBLE"


def test_output_capacity_counts_committed_and_pending_service_copies(
    tmp_path: Path,
) -> None:
    database, registry, training, store = _runtime(tmp_path)
    plan = replace(_tokenizer_plan(), reservation=_reservation(byte_count=10, artifact_rows=3))
    job_id, _ = _install_context(database, training, plan)
    registry.register_stream(
        b"123456",
        job_id,
        artifact_type="generation",
        display_name="committed.json",
        media_type="application/json",
        preview_policy="text",
        origin="locally_created",
        job_id=job_id,
    )
    pending = PreparedArtifact(
        job_id=job_id,
        role="generation",
        staging_name="pending.json",
        artifact_id=str(uuid.uuid4()),
        staged=registry.stage_stream(b"1234", job_id),
    )
    with database.transaction() as connection:
        store.commit_artifact_in(connection, pending)
    with pytest.raises(ApiError) as caught:
        store._assert_output_capacity(
            job_id,
            proposed_bytes=1,
            proposed_rows=0,
            code=None,
            keep_terminal_capacity=False,
        )
    assert caught.value.code == "REGISTRY_LIMIT"


def test_artifact_prepare_validates_proposal_and_preserves_service_copy(
    tmp_path: Path,
) -> None:
    database, registry, training, store = _runtime(tmp_path)
    job_id, _ = _install_context(database, training, _tokenizer_plan())
    digest = hashlib.sha256(b"{}").hexdigest()
    invalid = [
        {},
        {"type": "other", "role": "generation", "staging_name": "x", "size_bytes": 2, "sha256": digest},
        {"type": "artifact_ready", "role": "unknown", "staging_name": "x", "size_bytes": 2, "sha256": digest},
        {"type": "artifact_ready", "role": "generation", "staging_name": "../x", "size_bytes": 2, "sha256": digest},
        {"type": "artifact_ready", "role": "generation", "staging_name": "x", "size_bytes": True, "sha256": digest},
        {"type": "artifact_ready", "role": "generation", "staging_name": "x", "size_bytes": 2, "sha256": "bad"},
    ]
    for proposal in invalid:
        with pytest.raises(OperationStoreError) as caught:
            store.prepare_artifact(job_id, proposal)
        assert caught.value.code == "WORKER_PROTOCOL_ERROR"

    missing = {
        "type": "artifact_ready",
        "role": "generation",
        "staging_name": "missing.json",
        "size_bytes": 2,
        "sha256": digest,
    }
    with pytest.raises(OperationStoreError):
        store.prepare_artifact(job_id, missing)

    path = registry.create_staging_dir(job_id) / "generation.json"
    path.write_bytes(b"{}")
    proposal = {**missing, "staging_name": path.name}
    prepared = store.prepare_artifact(job_id, proposal)
    path.write_bytes(b"mutated")
    with database.transaction() as connection:
        store.commit_artifact_in(connection, prepared)
    assert prepared.staged.path.read_bytes() == b"{}"

    duplicate_path = registry.create_staging_dir(job_id) / "second.json"
    duplicate_path.write_bytes(b"{}")
    with pytest.raises(OperationStoreError, match="twice"):
        store.prepare_artifact(
            job_id, {**proposal, "staging_name": duplicate_path.name}
        )
    with pytest.raises(TypeError):
        with database.transaction() as connection:
            store.commit_artifact_in(connection, object())  # type: ignore[arg-type]
    conflict = PreparedArtifact(
        job_id=job_id,
        role="generation",
        staging_name="conflict.json",
        artifact_id=str(uuid.uuid4()),
        staged=registry.stage_stream(b"other", job_id),
    )
    with pytest.raises(OperationStoreError, match="twice"):
        with database.transaction() as connection:
            store.commit_artifact_in(connection, conflict)


def test_checkpoint_prepare_rejects_nontraining_invalid_and_unsafe_proposals(
    tmp_path: Path,
) -> None:
    database, _, training, store = _runtime(tmp_path)
    tokenizer_job, _ = _install_context(database, training, _tokenizer_plan())
    with pytest.raises(OperationStoreError, match="cannot commit"):
        store.prepare_checkpoint(tokenizer_job, {})

    training_job, _ = _install_context(database, training, _training_plan())
    bad_type = {
        "type": "artifact_ready",
        "staging_name": "checkpoint",
        "manifest_sha256": "a" * 64,
        "files": [],
    }
    with pytest.raises(OperationStoreError, match="type"):
        store.prepare_checkpoint(training_job, bad_type)

    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    digest = hashlib.sha256(b"x").hexdigest()
    proposal = {
        "type": "checkpoint_ready",
        "staging_name": "missing-checkpoint",
        "manifest_sha256": digest,
        "files": [
            {"name": name, "size": 1, "sha256": digest} for name in names
        ],
    }
    with pytest.raises(OperationStoreError, match="staging is unsafe") as unsafe:
        store.prepare_checkpoint(training_job, proposal)
    assert unsafe.value.code == "CHECKPOINT_INCOMPATIBLE"


def test_checkpoint_leases_reject_exhaustion_and_corrupt_ledger(tmp_path: Path) -> None:
    database, _, training, store = _runtime(tmp_path)
    job_id, context = _install_context(database, training, _training_plan())
    with database.read() as connection:
        next_id, committed = store._next_checkpoint_id(connection, context)
    assert next_id == context.checkpoint_ids[0]
    assert committed == []
    with pytest.raises(OperationStoreError, match="leases"):
        with database.read() as connection:
            store._next_checkpoint_id(connection, replace(context, checkpoint_ids=()))

    corrupt_id = str(uuid.uuid4())
    now = "2026-10-01T00:00:00.000Z"
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO checkpoints(
                checkpoint_id, format, backend, origin, run_id, job_id,
                manifest_sha256, source_identity_json, created_at, updated_at,
                record_json
            ) VALUES (?, 'tiny_v2_portable', 'tiny_v2', 'locally_created',
                      NULL, ?, ?, NULL, ?, ?, '{}')
            """,
            (corrupt_id, job_id, "a" * 64, now, now),
        )
    with pytest.raises(ApiError) as caught:
        with database.read() as connection:
            store._checkpoint_rows_in(connection, job_id)
    _assert_storage_corrupt(caught.value)


def test_run_creation_rolls_back_retries_and_rejects_identity_drift(tmp_path: Path) -> None:
    database, _, training, store = _runtime(tmp_path)
    _, context = _install_context(database, training, _tokenizer_plan())
    with pytest.raises(RuntimeError, match="rollback"):
        with database.transaction() as connection:
            created = store._ensure_run_in(connection, context)
            assert created is not None
            raise RuntimeError("rollback")
    with database.read() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id = ?", (context.run_id,)
        ).fetchone()[0] == 0

    with database.transaction() as connection:
        created = store._ensure_run_in(connection, context)
    with database.transaction() as connection:
        assert store._ensure_run_in(connection, context) == created
    drifted = dict(created or {})
    drifted["job_id"] = str(uuid.uuid4())
    with database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET record_json = ? WHERE run_id = ?",
            (canonical_json(drifted).decode("utf-8"), context.run_id),
        )
    with pytest.raises(ApiError) as caught:
        with database.transaction() as connection:
            store._ensure_run_in(connection, context)
    _assert_storage_corrupt(caught.value)


def _descriptor_context(operation: str) -> JobContext:
    dataset_id = str(uuid.uuid4())
    bindings = [
        {
            "dataset_id": dataset_id,
            "split": split,
            "artifact_id": str(uuid.uuid4()),
            "sha256": digest * 64,
        }
        for split, digest in (("train", "6"), ("validation", "7"))
    ]
    request = {
        "operation": "tiny_train",
        "dataset_id": dataset_id,
        "tokenizer_id": str(uuid.uuid4()),
        "architecture_profile_id": "tiny-v2-standard-v1",
        "steps": 1,
        "eval_every": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
        "context": 8,
        "width": 16,
        "heads": 1,
        "layers": 1,
        "seed": 17,
    }
    resolved: dict[str, Any] = {
        "request": request,
        "dataset_bindings": bindings,
    }
    if operation == "tiny_resume":
        parent_id = str(uuid.uuid4())
        resolved = {
            "request": {
                "operation": "tiny_resume",
                "checkpoint_id": parent_id,
                "additional_steps": 1,
            },
            "dataset_bindings": bindings,
            "training_identity": {
                "batch_size": 1,
                "learning_rate": 0.001,
                "context": 8,
                "width": 16,
                "heads": 1,
                "layers": 1,
                "eval_every": 1,
                "save_every": 1,
            },
            "parent_checkpoint": {
                "checkpoint_id": parent_id,
                "step": 0,
                "seed": 17,
                "model_identity": {
                    "kind": "tiny_v2",
                    "architecture_profile_id": "tiny-v2-standard-v1",
                    "config_sha256": "8" * 64,
                    "tokenizer_sha256": "9" * 64,
                },
            },
        }
    return JobContext(
        job_id=str(uuid.uuid4()),
        operation=operation,
        request_sha256="3" * 64,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="1" * 64,
        companion_source_revision="2" * 40,
        run_id=str(uuid.uuid4()),
        model_id=str(uuid.uuid4()),
        tokenizer_id=None,
        checkpoint_ids=(str(uuid.uuid4()),),
        reservation=_reservation(checkpoints=1),
        resolved=resolved,
    )


@pytest.mark.parametrize("operation", ["tiny_train", "tiny_resume"])
def test_checkpoint_descriptor_selects_external_and_in_run_lineage(
    tmp_path: Path, operation: str
) -> None:
    _, _, _, store = _runtime(tmp_path)
    context = _descriptor_context(operation)
    prepared = PreparedCheckpoint(
        context=context,
        checkpoint_id=context.checkpoint_ids[0],
        step=1 if operation == "tiny_resume" else 0,
        manifest_sha256="a" * 64,
        files=(),
        manifest={},
        trainer_state={},
        config={"architecture_profile_id": "tiny-v2-standard-v1"},
        tokenizer_sha256="9" * 64,
        config_sha256="8" * 64,
    )
    external = store._checkpoint_descriptor(
        prepared,
        artifact_ids={"manifest.json": str(uuid.uuid4())},
        previous=None,
        created_at="2026-10-01T00:00:00.000Z",
    )
    expected_external = (
        None
        if operation == "tiny_train"
        else context.resolved["request"]["checkpoint_id"]
    )
    assert external["parent_checkpoint_id"] == expected_external

    previous_id = str(uuid.uuid4())
    chained = store._checkpoint_descriptor(
        prepared,
        artifact_ids={"manifest.json": str(uuid.uuid4())},
        previous={"checkpoint_id": previous_id},
        created_at="2026-10-01T00:00:00.000Z",
    )
    assert chained["parent_checkpoint_id"] == previous_id


@pytest.mark.parametrize("valid_metric", [True, False])
def test_failed_training_retains_only_valid_metrics(
    tmp_path: Path, valid_metric: bool
) -> None:
    database, registry, training, store = _runtime(tmp_path)
    job_id, context = _install_context(database, training, _training_plan())
    metric = {
        "run_id": context.run_id,
        "sequence": 0,
        "step": 0,
        "name": "train_nll_token",
        "value": 1.0,
        "unit": "nats_per_token",
        "protocol_id": "tiny-v2-training-v1",
        "recorded_at": "2026-10-01T00:00:00.000Z",
    }
    metric_raw = canonical_json(metric) + b"\n" if valid_metric else b"[]\n"
    metrics = PreparedArtifact(
        job_id=job_id,
        role="training_metrics",
        staging_name="metrics.jsonl",
        artifact_id=str(uuid.uuid4()),
        staged=registry.stage_stream(metric_raw, job_id),
    )
    extra = PreparedArtifact(
        job_id=job_id,
        role="generation",
        staging_name="generation.json",
        artifact_id=str(uuid.uuid4()),
        staged=registry.stage_stream(b"{}", job_id),
    )
    with database.transaction() as connection:
        store.commit_artifact_in(connection, metrics)
        store.commit_artifact_in(connection, extra)
    terminal = store.prepare_terminal(
        job_id, state="failed", reason_code="WORKER_PROTOCOL_ERROR"
    )
    assert terminal.artifacts == ((metrics,) if valid_metric else ())
    assert set(terminal.discarded_artifacts) == (
        {extra} if valid_metric else {metrics, extra}
    )


def test_terminal_validation_rejects_unexpected_shapes_and_types(tmp_path: Path) -> None:
    database, _, training, store = _runtime(tmp_path)
    job_id, _ = _install_context(database, training, _tokenizer_plan())
    with pytest.raises(ValueError, match="terminal"):
        store.prepare_terminal(job_id, state="running", reason_code="running")
    with pytest.raises(OperationStoreError, match="typed result"):
        store.prepare_terminal(
            job_id, state="completed", reason_code="completed", result=None
        )
    with pytest.raises(OperationStoreError, match="cannot return"):
        store.prepare_terminal(
            job_id,
            state="interrupted",
            reason_code="interrupted",
            result={"operation": "tokenizer_train"},
        )
    with pytest.raises(TypeError):
        with database.transaction() as connection:
            store.finalize_terminal_in(connection, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        store.complete_terminal(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        store.complete_checkpoint(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("case", ["valid", "artifact_identity", "digest"])
def test_context_preview_terminal_binds_exact_artifact(
    tmp_path: Path, case: str
) -> None:
    _, registry, _, store = _runtime(tmp_path)
    job_id = str(uuid.uuid4())
    preview = {
        "backend": "tiny",
        "subject_identity_sha256": "3" * 64,
        "tokenizer_sha256": "4" * 64,
        "template_sha256": None,
        "device": "cpu",
        "original_input_token_count": 1,
        "input_token_ids": [1],
        "input_token_count": 1,
        "serialized_text": "hello",
        "dropped_message_indices": [],
        "cropped_input_tokens": 0,
        "effective_context_budget": 8,
        "canonical_generation_request_sha256": "5" * 64,
    }
    artifact = PreparedArtifact(
        job_id=job_id,
        role="context_preview",
        staging_name="context-preview.json",
        artifact_id=str(uuid.uuid4()),
        staged=registry.stage_stream(canonical_json(preview), job_id),
    )
    context = JobContext(
        job_id=job_id,
        operation="context_preview",
        request_sha256="6" * 64,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="1" * 64,
        companion_source_revision="2" * 40,
        run_id=None,
        model_id=None,
        tokenizer_id=None,
        checkpoint_ids=(),
        reservation=_reservation(),
        resolved={"request": {"operation": "context_preview"}},
    )
    result = {
        "operation": "context_preview",
        "preview_artifact_id": artifact.artifact_id,
        **preview,
        "artifact_ids": [artifact.artifact_id],
        "context_preview_digest": hashlib.sha256(canonical_json(preview)).hexdigest(),
    }
    if case == "artifact_identity":
        result["preview_artifact_id"] = str(uuid.uuid4())
    elif case == "digest":
        result["context_preview_digest"] = "7" * 64

    if case == "valid":
        assert store._validate_completed_result(context, result, (artifact,)) == result
    else:
        with pytest.raises(OperationStoreError) as caught:
            store._validate_completed_result(context, result, (artifact,))
        assert caught.value.code == "WORKER_PROTOCOL_ERROR"


@pytest.mark.parametrize(
    "case", ["valid", "run_identity", "checkpoint_identity", "artifact_binding"]
)
def test_generate_terminal_binds_result_to_context_and_artifact(
    tmp_path: Path, case: str
) -> None:
    _, registry, _, store = _runtime(tmp_path)
    job_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    checkpoint_id = str(uuid.uuid4())
    generated = {
        "job_id": job_id,
        "run_id": run_id,
        "text": "world",
        "generated_token_count": 1,
        "stop_reason": "length",
        "serialized_input": "hello",
    }
    if case == "artifact_binding":
        generated["text"] = "different"
    artifact = PreparedArtifact(
        job_id=job_id,
        role="generation",
        staging_name="generation.json",
        artifact_id=str(uuid.uuid4()),
        staged=registry.stage_stream(canonical_json(generated), job_id),
    )
    context = JobContext(
        job_id=job_id,
        operation="generate",
        request_sha256="8" * 64,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="1" * 64,
        companion_source_revision="2" * 40,
        run_id=run_id,
        model_id=None,
        tokenizer_id=None,
        checkpoint_ids=(),
        reservation=_reservation(),
        resolved={
            "request": {
                "operation": "generate",
                "checkpoint_id": checkpoint_id,
            }
        },
    )
    result = {
        "operation": "generate",
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "generated_text": "world",
        "generated_token_count": 1,
        "stop_reason": "length",
        "serialized_input": "hello",
        "truncated_input_tokens": 0,
        "artifact_ids": [artifact.artifact_id],
    }
    if case == "run_identity":
        result["run_id"] = str(uuid.uuid4())
    elif case == "checkpoint_identity":
        result["checkpoint_id"] = str(uuid.uuid4())

    if case == "valid":
        assert store._validate_completed_result(context, result, (artifact,)) == result
    else:
        with pytest.raises(OperationStoreError) as caught:
            store._validate_completed_result(context, result, (artifact,))
        assert caught.value.code == "WORKER_PROTOCOL_ERROR"


@pytest.mark.parametrize(
    "case",
    ["valid", "invalid_shape", "identity", "record_order", "cardinality", "payload", "summary"],
)
def test_paired_evaluation_reconciles_records_metrics_and_subjects(
    tmp_path: Path, case: str
) -> None:
    _, registry, _, store = _runtime(tmp_path)
    job_id = str(uuid.uuid4())
    evaluation_id = str(uuid.uuid4())
    dataset_id = str(uuid.uuid4())
    records_id = str(uuid.uuid4())
    subjects = [
        {"kind": "tiny_checkpoint", "checkpoint_id": str(uuid.uuid4())}
        for _index in range(2)
    ]
    checkpoints = [
        {
            "checkpoint_sha256": digest * 64,
            "tokenizer_sha256": tokenizer * 64,
        }
        for digest, tokenizer in (("a", "c"), ("b", "d"))
    ]
    summaries = [
        {
            "negative_log_likelihood_sum": 10.0 + 5.0 * index,
            "utf8_bytes_sum": 5,
            "record_count": 2,
            "nll_per_utf8_byte": 2.0 + index,
        }
        for index in range(2)
    ]
    paired = {
        "format": "llm-foundations-tiny-paired-evaluation-v1",
        "evaluation_id": evaluation_id,
        "job_id": job_id,
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": "e" * 64,
        "evaluation_profile_id": "tiny-nll-per-byte-v1",
        "records_artifact_id": records_id,
        "ordered_record_ids": ["record-a", "record-b"],
        "subjects": [
            {
                "subject_index": index,
                "subject": subjects[index],
                **checkpoints[index],
                **summaries[index],
            }
            for index in range(2)
        ],
        "delta_nll_per_utf8_byte": 1.0,
    }
    records = [
        {"subject_index": index, "record_id": record_id}
        for index in range(2)
        for record_id in ("record-a", "record-b")
    ]
    metric_rows = [
        {"sequence": index, "payload": {**summaries[index], "records_artifact_id": records_id}}
        for index in range(2)
    ]
    if case == "invalid_shape":
        paired["format"] = "wrong"
    elif case == "identity":
        paired["dataset_manifest_sha256"] = "f" * 64
    elif case == "record_order":
        records[2], records[3] = records[3], records[2]
    elif case == "cardinality":
        metric_rows.pop()
    elif case == "payload":
        metric_rows[0].pop("payload")
    elif case == "summary":
        metric_rows[0]["payload"]["record_count"] = 3

    records_raw = b"".join(canonical_json(row) + b"\n" for row in records)
    paired_artifact = PreparedArtifact(
        job_id=job_id,
        role="evaluation_paired",
        staging_name="paired.json",
        artifact_id=str(uuid.uuid4()),
        staged=registry.stage_stream(canonical_json(paired), job_id),
    )
    records_artifact = PreparedArtifact(
        job_id=job_id,
        role="evaluation_records",
        staging_name="records.jsonl",
        artifact_id=records_id,
        staged=registry.stage_stream(records_raw, job_id),
    )
    context = JobContext(
        job_id=job_id,
        operation="evaluate",
        request_sha256="1" * 64,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="2" * 64,
        companion_source_revision="3" * 40,
        run_id=evaluation_id,
        model_id=None,
        tokenizer_id=None,
        checkpoint_ids=(),
        reservation=_reservation(),
        resolved={
            "request": {
                "operation": "evaluate",
                "dataset_id": dataset_id,
                "split": "test",
                "evaluation_profile_id": "tiny-nll-per-byte-v1",
                "subjects": subjects,
            },
            "dataset_manifest_sha256": "e" * 64,
            "dataset_bindings": [{"split": "test"}],
            "subjects": [{"checkpoint": item} for item in checkpoints],
        },
    )
    by_role = {
        "evaluation_paired": paired_artifact,
        "evaluation_records": records_artifact,
    }
    if case == "valid":
        assert store._validate_paired(context, by_role, metric_rows) is None
    else:
        with pytest.raises(OperationStoreError) as caught:
            store._validate_paired(context, by_role, metric_rows)
        assert caught.value.code == "WORKER_PROTOCOL_ERROR"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_external", "not durable"),
        ("corrupt_durable", "STORAGE_CORRUPT"),
        ("step_mismatch", "step is inconsistent"),
    ],
)
def test_checkpoint_incumbent_requires_durable_consistent_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    database, registry, _, store = _runtime(tmp_path)
    operation = "tiny_resume" if case == "missing_external" else "tiny_train"
    context, checkpoint_id, values = _checkpoint_metadata_case(operation)
    incumbent_id = str(uuid.uuid4())
    trainer = json.loads(values["trainer_state.json"])
    trainer["incumbent"] = {
        "checkpoint_id": incumbent_id,
        "validation_nll_token": 1.0,
        "step": 2 if case == "step_mismatch" else 0,
    }
    values["trainer_state.json"] = canonical_json(trainer)
    if case == "missing_external":
        monkeypatch.setattr(
            store, "_checkpoint_trainer_incumbent", lambda _parent_id: incumbent_id
        )
    else:
        record_json = "{" if case == "corrupt_durable" else '{"step":1}'
        now = "2026-10-01T00:00:00.000Z"
        with database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO checkpoints(
                    checkpoint_id, format, backend, origin, run_id, job_id,
                    manifest_sha256, source_identity_json, created_at, updated_at,
                    deleted_at, record_json
                ) VALUES (?, 'tiny_v2_portable', 'tiny_v2', 'locally_created',
                          NULL, ?, ?, NULL, ?, ?, NULL, ?)
                """,
                (incumbent_id, context.job_id, "4" * 64, now, now, record_json),
            )
    staged, digest = _stage_checkpoint_metadata(registry, context, values)

    if case == "corrupt_durable":
        with pytest.raises(ApiError) as caught:
            store._validate_checkpoint_metadata(context, checkpoint_id, staged, digest)
        _assert_storage_corrupt(caught.value)
    else:
        with pytest.raises(OperationStoreError, match=message) as caught:
            store._validate_checkpoint_metadata(context, checkpoint_id, staged, digest)
        assert caught.value.code == "CHECKPOINT_INCOMPATIBLE"


def test_partial_checkpoint_prepare_cleans_service_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, registry, training, store = _runtime(tmp_path)
    job_id, _ = _install_context(database, training, _training_plan())
    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    directory = registry.create_staging_dir(job_id) / "partial-checkpoint"
    directory.mkdir()
    for name in names:
        (directory / name).write_bytes(b"x")
    digest = hashlib.sha256(b"x").hexdigest()
    proposal = {
        "type": "checkpoint_ready",
        "staging_name": directory.name,
        "manifest_sha256": digest,
        "files": [
            {
                "name": name,
                "size": 2 if name == "optimizer.safetensors" else 1,
                "sha256": digest,
            }
            for name in names
        ],
    }
    copied: list[Any] = []
    original = store._copy_staged_file

    def capture(*args: Any, **kwargs: Any) -> Any:
        staged = original(*args, **kwargs)
        copied.append(staged)
        return staged

    monkeypatch.setattr(store, "_copy_staged_file", capture)
    with pytest.raises(OperationStoreError, match="verification"):
        store.prepare_checkpoint(job_id, proposal)
    assert len(copied) == 1
    assert not copied[0].path.exists()


def test_checkpoint_commit_failure_rolls_back_and_preserves_retry_bytes(
    tmp_path: Path,
) -> None:
    database, registry, training, store = _runtime(tmp_path)
    job_id, context = _install_context(database, training, _training_plan())
    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    files = tuple(
        (name, registry.stage_stream(f"bytes:{name}".encode(), job_id))
        for name in names
    )
    prepared = PreparedCheckpoint(
        context=context,
        checkpoint_id=context.checkpoint_ids[0],
        step=0,
        manifest_sha256="5" * 64,
        files=files,
        manifest={},
        trainer_state={},
        config={"architecture_profile_id": "invalid"},
        tokenizer_sha256="6" * 64,
        config_sha256="7" * 64,
    )
    with pytest.raises(OperationStoreError, match="descriptor"):
        with database.transaction() as connection:
            store.commit_checkpoint_in(connection, prepared)
    assert all(staged.path.exists() for _name, staged in files)
    with database.read() as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0

    corrected = replace(
        prepared,
        config={"architecture_profile_id": "tiny-v2-standard-v1"},
    )
    with database.transaction() as connection:
        committed = store.commit_checkpoint_in(connection, corrected)
    assert committed.checkpoint_id == context.checkpoint_ids[0]
    with database.read() as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 7
        assert connection.execute("SELECT COUNT(*) FROM checkpoint_files").fetchone()[0] == 7
    store.complete_checkpoint(corrected)
    assert all(not staged.path.exists() for _name, staged in files)


def test_terminal_finalizer_internal_failure_is_atomic_and_retryable(
    tmp_path: Path,
) -> None:
    database, registry, training, store = _runtime(tmp_path)
    plan = _tokenizer_plan()
    job_id, context = _install_context(database, training, plan)
    payload = serialize_tokenizer(Tokenizer())
    path = registry.create_staging_dir(job_id) / "tokenizer.json"
    path.write_bytes(payload)
    artifact = store.prepare_artifact(
        job_id,
        {
            "type": "artifact_ready",
            "role": "tokenizer_json",
            "staging_name": path.name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    with database.transaction() as connection:
        store.commit_artifact_in(connection, artifact)
    result = {
        "operation": "tokenizer_train",
        "tokenizer_id": context.tokenizer_id,
        "tokenizer_type": "byte",
        "vocab_size": 257,
        "tokenizer_sha256": Tokenizer().fingerprint(),
        "artifact_ids": [artifact.artifact_id],
    }
    terminal = store.prepare_terminal(
        job_id, state="completed", reason_code="completed", result=result
    )
    with database.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET record_json = ? WHERE job_id = ?",
            (
                canonical_json(
                    {
                        "job_id": job_id,
                        "run_id": str(uuid.uuid4()),
                        "step": 0,
                        "requested_final_step": None,
                        "checkpoint_boundary": None,
                    }
                ).decode(),
                job_id,
            ),
        )
    with pytest.raises(ApiError) as caught:
        with database.transaction() as connection:
            store.finalize_terminal_in(connection, terminal)
    _assert_storage_corrupt(caught.value)
    assert artifact.staged.path.exists()
    with database.read() as connection:
        for table in ("runs", "artifacts", "tokenizers", "run_artifacts"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

    with database.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET record_json = ? WHERE job_id = ?",
            (
                canonical_json(
                    {
                        "job_id": job_id,
                        "run_id": context.run_id,
                        "step": 0,
                        "requested_final_step": None,
                        "checkpoint_boundary": None,
                    }
                ).decode(),
                job_id,
            ),
        )
    with database.transaction() as connection:
        committed = store.finalize_terminal_in(connection, terminal)
    assert committed.artifact_ids[0] == artifact.artifact_id
    assert len(committed.artifact_ids) == 2
    store.complete_terminal(terminal)
    assert not artifact.staged.path.exists()


def _checkpoint_metadata_case(
    operation: str,
) -> tuple[JobContext, str, dict[str, bytes]]:
    context = _descriptor_context(operation)
    checkpoint_id = context.checkpoint_ids[0]
    tokenizer = Tokenizer()
    tokenizer_raw = serialize_tokenizer(tokenizer)
    resolved = copy.deepcopy(context.resolved)
    resolved["requested_final_step"] = 1
    if operation == "tiny_train":
        resolved["tokenizer"] = {"tokenizer_sha256": tokenizer.fingerprint()}
    else:
        resolved["parent_checkpoint"]["model_identity"][
            "tokenizer_sha256"
        ] = tokenizer.fingerprint()
    context = replace(context, resolved=resolved)
    config = {
        "format": "tiny-v2-config-v1",
        "architecture_profile_id": "tiny-v2-standard-v1",
        "vocab_size": 257,
        "context": 8,
        "width": 16,
        "heads": 1,
        "layers": 1,
    }
    trainer = {
        "format": "tiny-v2-trainer-state-v1",
        "completed_global_step": 0,
        "requested_final_step": 1,
        "eval_every": 1,
        "batch_size": 1,
        "seed": 17,
        "learning_rate": 0.001,
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
        "parameter_steps": {"tokens.weight": 0},
        "incumbent": {
            "checkpoint_id": checkpoint_id,
            "validation_nll_token": 1.25,
            "step": 0,
        },
        "dataset_bindings": [
            {
                "split": item["split"],
                "dataset_id": item["dataset_id"],
                "sha256": item["sha256"],
            }
            for item in resolved["dataset_bindings"]
        ],
        "tokenizer_sha256": tokenizer.fingerprint(),
        "architecture_profile_id": "tiny-v2-standard-v1",
        "runtime_profile": context.runtime_profile,
        "device": context.device,
        "dependency_lock_sha256": context.dependency_lock_sha256,
    }
    if operation == "tiny_resume":
        trainer["incumbent"] = {
            "checkpoint_id": None,
            "validation_nll_token": None,
            "step": None,
        }
    values = {
        "model.safetensors": b"model-safe-bytes",
        "optimizer.safetensors": b"optimizer-safe-bytes",
        "rng.safetensors": b"rng-safe-bytes",
        "tokenizer.json": tokenizer_raw,
        "config.json": canonical_json(config),
        "trainer_state.json": canonical_json(trainer),
    }
    return context, checkpoint_id, values


def _stage_checkpoint_metadata(
    registry: Registry,
    context: JobContext,
    values: dict[str, bytes],
) -> tuple[dict[str, Any], str]:
    manifest = {
        "format": "tiny-v2-checkpoint-v1",
        "portability": "resume",
        "files": [
            {
                "name": name,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for name, raw in values.items()
        ],
        "aliases": {},
    }
    complete = {**values, "manifest.json": canonical_json(manifest)}
    staged = {
        name: registry.stage_stream(raw, context.job_id)
        for name, raw in complete.items()
    }
    return staged, hashlib.sha256(complete["manifest.json"]).hexdigest()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("manifest_identity", "manifest identity"),
        ("metadata_json", "JSON metadata"),
        ("config_fields", "config has invalid fields"),
        ("tokenizer_invalid", "tokenizer is incompatible"),
        ("tokenizer_noncanonical", "serialization is not exact"),
        ("vocab_mismatch", "tokenizer and config differ"),
        ("config_admission", "config differs from admission"),
        ("trainer_identity", "trainer identity differs"),
        ("dataset_identity", "dataset bindings differ"),
        ("tokenizer_fingerprint", "fingerprint differs"),
        ("optimizer_steps", "optimizer steps are inconsistent"),
        ("outside_lineage", "outside this lineage"),
    ],
)
def test_checkpoint_metadata_rejects_independently_valid_identity_drift(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    _, registry, _, store = _runtime(tmp_path)
    context, checkpoint_id, values = _checkpoint_metadata_case("tiny_train")
    passed_digest: str | None = None

    if case == "metadata_json":
        values["trainer_state.json"] = b"{"
    elif case == "config_fields":
        config = json.loads(values["config.json"])
        config["unexpected"] = True
        values["config.json"] = canonical_json(config)
    elif case == "tokenizer_invalid":
        values["tokenizer.json"] = b"{}"
    elif case == "tokenizer_noncanonical":
        values["tokenizer.json"] += b"\n"
    elif case in {"vocab_mismatch", "config_admission"}:
        config = json.loads(values["config.json"])
        config["vocab_size" if case == "vocab_mismatch" else "width"] = (
            258 if case == "vocab_mismatch" else 32
        )
        values["config.json"] = canonical_json(config)
    else:
        trainer = json.loads(values["trainer_state.json"])
        if case == "trainer_identity":
            trainer["dependency_lock_sha256"] = "2" * 64
        elif case == "dataset_identity":
            trainer["dataset_bindings"][0]["sha256"] = "a" * 64
        elif case == "tokenizer_fingerprint":
            trainer["tokenizer_sha256"] = "f" * 64
            resolved = copy.deepcopy(context.resolved)
            resolved["tokenizer"]["tokenizer_sha256"] = "f" * 64
            context = replace(context, resolved=resolved)
        elif case == "optimizer_steps":
            trainer["parameter_steps"]["tokens.weight"] = 1
        elif case == "outside_lineage":
            trainer["incumbent"]["checkpoint_id"] = str(uuid.uuid4())
        values["trainer_state.json"] = canonical_json(trainer)

    staged, digest = _stage_checkpoint_metadata(registry, context, values)
    if case == "manifest_identity":
        passed_digest = "f" * 64
    with pytest.raises(OperationStoreError, match=message) as caught:
        store._validate_checkpoint_metadata(
            context,
            checkpoint_id,
            staged,
            passed_digest or digest,
        )
    assert caught.value.code == "CHECKPOINT_INCOMPATIBLE"


def test_checkpoint_metadata_accepts_bound_train_and_resume_contexts(
    tmp_path: Path,
) -> None:
    _, registry, _, store = _runtime(tmp_path)
    for operation in ("tiny_train", "tiny_resume"):
        context, checkpoint_id, values = _checkpoint_metadata_case(operation)
        staged, digest = _stage_checkpoint_metadata(registry, context, values)
        manifest, trainer, config, _, _, step = store._validate_checkpoint_metadata(
            context,
            checkpoint_id,
            staged,
            digest,
        )
        assert manifest["portability"] == "resume"
        assert trainer["completed_global_step"] == step == 0
        assert config["architecture_profile_id"] == "tiny-v2-standard-v1"


@pytest.mark.parametrize("case", ["missing_parent", "bad_model_identity"])
def test_resume_checkpoint_metadata_rejects_corrupt_admission_context(
    tmp_path: Path,
    case: str,
) -> None:
    _, registry, _, store = _runtime(tmp_path)
    context, checkpoint_id, values = _checkpoint_metadata_case("tiny_resume")
    resolved = copy.deepcopy(context.resolved)
    if case == "missing_parent":
        resolved.pop("parent_checkpoint")
    else:
        resolved["parent_checkpoint"]["model_identity"] = None
    context = replace(context, resolved=resolved)
    staged, digest = _stage_checkpoint_metadata(registry, context, values)

    with pytest.raises(ApiError) as caught:
        store._validate_checkpoint_metadata(
            context,
            checkpoint_id,
            staged,
            digest,
        )
    _assert_storage_corrupt(caught.value)
