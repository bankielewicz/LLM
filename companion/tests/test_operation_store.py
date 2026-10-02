from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from llm_foundations_companion.database import Database
from llm_foundations_companion.operation_store import (
    OperationStore,
    OperationStoreError,
    PreparedArtifact,
)
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.tiny_v2.tokenizer import Tokenizer, serialize_tokenizer
from llm_foundations_companion.training_store import AdmissionPlan, TrainingStore


def _runtime(tmp_path: Path) -> tuple[Database, Registry, TrainingStore, OperationStore]:
    root = tmp_path / "root"
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


def _context(
    database: Database,
    training: TrainingStore,
    *,
    job_id: str,
    plan: AdmissionPlan,
) -> None:
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
                plan.reservation.get("artifact_rows", 0),
                plan.reservation.get("dataset_rows", 0),
                plan.reservation.get("run_rows", 0),
                plan.reservation.get("model_rows", 0),
                plan.reservation.get("checkpoint_rows", 0),
                "2026-10-01T00:00:00.000Z",
            ),
        )
        requested = plan.resolved.get("requested_final_step")
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


def _set_job_projection(
    database: Database,
    job_id: str,
    *,
    step: int,
    requested_final_step: int,
    checkpoint_id: str | None = None,
) -> None:
    with database.transaction() as connection:
        row = connection.execute(
            "SELECT record_json FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        record = json.loads(row[0])
        record["step"] = step
        record["requested_final_step"] = requested_final_step
        record["checkpoint_boundary"] = (
            None
            if checkpoint_id is None
            else {"checkpoint_id": checkpoint_id, "step": step}
        )
        connection.execute(
            "UPDATE jobs SET record_json = ? WHERE job_id = ?",
            (canonical_json(record).decode("utf-8"), job_id),
        )


def _reservation(*, checkpoints: int = 0, models: int = 0) -> dict[str, int]:
    return {
        "byte_count": 1_000_000,
        "artifact_rows": 16,
        "dataset_rows": 0,
        "run_rows": 1,
        "model_rows": models,
        "checkpoint_rows": checkpoints,
    }


def test_operation_store_import_is_framework_free(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[1] / "src"
    script = (
        "import sys;"
        f"sys.path.insert(0, {str(source_root)!r});"
        "import llm_foundations_companion.operation_store;"
        "assert 'torch' not in sys.modules;"
        "assert 'safetensors' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True, cwd=tmp_path)


def test_artifact_proposal_cannot_exceed_held_job_reservation(
    tmp_path: Path,
) -> None:
    database, _registry, training, store = _runtime(tmp_path)
    job_id, run_id, tokenizer_id = (str(uuid.uuid4()) for _ in range(3))
    dataset_id, split_artifact_id = (str(uuid.uuid4()) for _ in range(2))
    plan = AdmissionPlan(
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
                    "artifact_id": split_artifact_id,
                    "sha256": "5" * 64,
                }
            ],
            "_inputs": [],
        },
        run_id=run_id,
        model_id=None,
        tokenizer_id=tokenizer_id,
        checkpoint_ids=(),
    )
    _context(database, training, job_id=job_id, plan=plan)
    with pytest.raises(OperationStoreError, match="held reservation") as caught:
        store.prepare_artifact(
            job_id,
            {
                "type": "artifact_ready",
                "role": "tokenizer_json",
                "staging_name": "oversized.json",
                "size_bytes": plan.reservation["byte_count"],
                "sha256": "a" * 64,
            },
        )
    assert caught.value.code == "WORKER_PROTOCOL_ERROR"


def test_terminal_artifact_is_deferred_and_uses_service_copy(tmp_path: Path) -> None:
    database, registry, training, store = _runtime(tmp_path)
    job_id, run_id, tokenizer_id = (str(uuid.uuid4()) for _ in range(3))
    dataset_id, split_artifact_id = (str(uuid.uuid4()) for _ in range(2))
    plan = AdmissionPlan(
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
                    "artifact_id": split_artifact_id,
                    "sha256": "5" * 64,
                }
            ],
            "_inputs": [],
        },
        run_id=run_id,
        model_id=None,
        tokenizer_id=tokenizer_id,
        checkpoint_ids=(),
    )
    _context(database, training, job_id=job_id, plan=plan)
    payload = serialize_tokenizer(Tokenizer())
    path = registry.create_staging_dir(job_id) / "tokenizer.json"
    path.write_bytes(payload)
    proposal = {
        "type": "artifact_ready",
        "role": "tokenizer_json",
        "staging_name": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    prepared = store.prepare_artifact(job_id, proposal)
    path.write_bytes(b"mutated after prepare")
    with database.transaction() as connection:
        committed = store.commit_artifact_in(connection, prepared)
    with database.read() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE artifact_id = ?",
            (prepared.artifact_id,),
        ).fetchone()[0] == 0
    assert committed.ack["artifact_id"] == prepared.artifact_id

    result = {
        "operation": "tokenizer_train",
        "tokenizer_id": tokenizer_id,
        "tokenizer_type": "byte",
        "vocab_size": 257,
        "tokenizer_sha256": Tokenizer().fingerprint(),
        "artifact_ids": [prepared.artifact_id],
    }
    terminal = store.prepare_terminal(
        job_id, state="completed", reason_code="completed", result=result
    )
    with pytest.raises(RuntimeError, match="forced rollback"):
        with database.transaction() as connection:
            rolled_back = store.finalize_terminal_in(connection, terminal)
            assert rolled_back.artifact_ids[0] == prepared.artifact_id
            raise RuntimeError("forced rollback")
    assert prepared.staged.path.exists()
    retry_terminal = store.prepare_terminal(
        job_id,
        state="completed",
        reason_code="completed",
        result=result,
        finished_at=terminal.finished_at,
    )
    assert retry_terminal.artifacts == terminal.artifacts
    with database.transaction() as connection:
        finalized = store.finalize_terminal_in(connection, retry_terminal)
    store.complete_terminal(retry_terminal)

    assert finalized.result == result
    assert finalized.run_id == run_id
    assert finalized.model_id is None
    assert finalized.artifact_ids[0] == prepared.artifact_id
    assert len(finalized.artifact_ids) == 2
    assert registry.read_artifact_bytes(prepared.artifact_id) == payload
    with database.read() as connection:
        tokenizer = connection.execute(
            "SELECT artifact_id, record_json FROM tokenizers WHERE tokenizer_id = ?",
            (tokenizer_id,),
        ).fetchone()
        assert tokenizer[0] == prepared.artifact_id
        assert json.loads(tokenizer[1])["artifact_ids"] == [prepared.artifact_id]
        assert connection.execute(
            "SELECT artifact_id FROM run_artifacts WHERE run_id = ? AND role = 'tokenizer_json'",
            (run_id,),
        ).fetchone()[0] == prepared.artifact_id
        run_result_id = connection.execute(
            "SELECT artifact_id FROM run_artifacts WHERE run_id = ? AND role = 'run_result'",
            (run_id,),
        ).fetchone()[0]
    assert finalized.artifact_ids[1] == run_result_id
    projection = json.loads(registry.read_artifact_bytes(run_result_id))
    assert projection == {
        "run_id": run_id,
        "status": "completed",
        "requested_final_step": 0,
        "final_step": 0,
        "last_observed_step": 0,
        "last_durable_checkpoint_step": None,
        "finished_at": retry_terminal.finished_at,
    }


@pytest.mark.parametrize(
    ("sequences", "steps"),
    [([1, 0], [0, 1]), ([0, 1], [1, 0])],
)
def test_metric_artifact_requires_increasing_sequence_and_nondecreasing_step(
    tmp_path: Path, sequences: list[int], steps: list[int]
) -> None:
    _database, registry, _training, store = _runtime(tmp_path)
    job_id, run_id, artifact_id = (str(uuid.uuid4()) for _ in range(3))
    rows = [
        {
            "run_id": run_id,
            "sequence": sequence,
            "step": step,
            "name": "train_nll_token",
            "value": 1.0,
            "unit": "nats_per_token",
            "protocol_id": "tiny-v2-training-v1",
            "recorded_at": "2026-10-01T00:00:00.000Z",
        }
        for sequence, step in zip(sequences, steps, strict=True)
    ]
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    staged = registry.stage_stream(raw, job_id)
    artifact = PreparedArtifact(
        job_id=job_id,
        role="training_metrics",
        staging_name="metrics.jsonl",
        artifact_id=artifact_id,
        staged=staged,
    )
    with pytest.raises(OperationStoreError, match="Metric sequence") as caught:
        store._validate_metric_artifact(artifact, run_id)
    assert caught.value.code == "WORKER_PROTOCOL_ERROR"


def test_completed_evaluation_binds_records_to_admitted_split_and_metrics(
    tmp_path: Path,
) -> None:
    database, registry, training, store = _runtime(tmp_path)
    job_id, run_id, dataset_id, split_artifact_id, checkpoint_id = (
        str(uuid.uuid4()) for _ in range(5)
    )
    dataset_raw = canonical_json({"record_id": "record-1", "text": "café"}) + b"\n"
    staged_split = registry.stage_stream(dataset_raw, str(uuid.uuid4()))
    with database.transaction() as connection:
        registry.commit_staged_in(
            connection,
            staged_split,
            artifact_type="dataset_split",
            display_name="validation.jsonl",
            media_type="application/x-ndjson",
            preview_policy="text",
            origin="locally_created",
            artifact_id=split_artifact_id,
        )
    subject = {"kind": "tiny_checkpoint", "checkpoint_id": checkpoint_id}
    plan = AdmissionPlan(
        operation="evaluate",
        reservation=_reservation(),
        resolved={
            "request": {
                "operation": "evaluate",
                "dataset_id": dataset_id,
                "split": "validation",
                "evaluation_profile_id": "tiny-nll-per-byte-v1",
                "subjects": [subject],
            },
            "dataset_manifest_sha256": "4" * 64,
            "dataset_bindings": [
                {
                    "dataset_id": dataset_id,
                    "split": "validation",
                    "artifact_id": split_artifact_id,
                    "sha256": hashlib.sha256(dataset_raw).hexdigest(),
                    "sealed": False,
                }
            ],
            "subjects": [],
            "_inputs": [],
        },
        run_id=run_id,
        model_id=None,
        tokenizer_id=None,
        checkpoint_ids=(),
    )
    _context(database, training, job_id=job_id, plan=plan)

    records = canonical_json(
        {
            "subject_index": True,
            "record_id": "record-1",
            "utf8_bytes": 5,
            "token_count": 5,
            "negative_log_likelihood": 5.0,
            "nll_per_utf8_byte": 1.0,
        }
    ) + b"\n"
    records_path = registry.create_staging_dir(job_id) / "evaluation-records.jsonl"
    records_path.write_bytes(records)
    records_prepared = store.prepare_artifact(
        job_id,
        {
            "type": "artifact_ready",
            "role": "evaluation_records",
            "staging_name": records_path.name,
            "size_bytes": len(records),
            "sha256": hashlib.sha256(records).hexdigest(),
        },
    )
    with database.transaction() as connection:
        store.commit_artifact_in(connection, records_prepared)

    metric = {
        "run_id": run_id,
        "sequence": 0,
        "step": 0,
        "name": "test_nll_bytes_v1",
        "value": 1.0,
        "unit": "nats_per_utf8_byte",
        "protocol_id": "tiny-nll-per-byte-v1",
        "recorded_at": "2026-10-01T00:00:00.000Z",
        "payload": {
            "negative_log_likelihood_sum": 5.0,
            "utf8_bytes_sum": 5,
            "record_count": 1,
            "nll_per_utf8_byte": 1.0,
            "records_artifact_id": records_prepared.artifact_id,
        },
    }
    metrics = canonical_json(metric) + b"\n"
    metrics_path = registry.create_staging_dir(job_id) / "evaluation-metrics.jsonl"
    metrics_path.write_bytes(metrics)
    metrics_prepared = store.prepare_artifact(
        job_id,
        {
            "type": "artifact_ready",
            "role": "evaluation_metrics",
            "staging_name": metrics_path.name,
            "size_bytes": len(metrics),
            "sha256": hashlib.sha256(metrics).hexdigest(),
        },
    )
    with database.transaction() as connection:
        store.commit_artifact_in(connection, metrics_prepared)
    result = {
        "operation": "evaluate",
        "evaluation_id": run_id,
        "subjects": [subject],
        "metrics_artifact_id": metrics_prepared.artifact_id,
        "records_artifact_id": records_prepared.artifact_id,
        "paired_artifact_id": None,
        "artifact_ids": [
            records_prepared.artifact_id,
            metrics_prepared.artifact_id,
        ],
    }
    with pytest.raises(OperationStoreError, match="Evaluation records") as caught:
        store.prepare_terminal(
            job_id,
            state="completed",
            reason_code="completed",
            result=result,
        )
    assert caught.value.code == "WORKER_PROTOCOL_ERROR"


def _checkpoint_payload(checkpoint_id: str, dataset_id: str) -> dict[str, bytes]:
    tokenizer = Tokenizer()
    tokenizer_bytes = serialize_tokenizer(tokenizer)
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
            {"split": "train", "dataset_id": dataset_id, "sha256": "6" * 64},
            {"split": "validation", "dataset_id": dataset_id, "sha256": "7" * 64},
        ],
        "tokenizer_sha256": tokenizer.fingerprint(),
        "architecture_profile_id": "tiny-v2-standard-v1",
        "runtime_profile": "wsl-cpu",
        "device": "cpu",
        "dependency_lock_sha256": "1" * 64,
    }
    values = {
        "model.safetensors": b"model-safe-bytes",
        "optimizer.safetensors": b"optimizer-safe-bytes",
        "rng.safetensors": b"rng-safe-bytes",
        "tokenizer.json": tokenizer_bytes,
        "config.json": canonical_json(config),
        "trainer_state.json": canonical_json(trainer),
    }
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
    values["manifest.json"] = canonical_json(manifest)
    return values


def test_checkpoint_commit_is_atomic_and_failed_terminal_registers_model(
    tmp_path: Path,
) -> None:
    database, registry, training, store = _runtime(tmp_path)
    job_id, run_id, model_id, checkpoint_id = (
        str(uuid.uuid4()) for _ in range(4)
    )
    dataset_id, train_artifact, validation_artifact, tokenizer_artifact = (
        str(uuid.uuid4()) for _ in range(4)
    )
    tokenizer = Tokenizer()
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
    bindings = [
        {
            "dataset_id": dataset_id,
            "split": "train",
            "artifact_id": train_artifact,
            "sha256": "6" * 64,
        },
        {
            "dataset_id": dataset_id,
            "split": "validation",
            "artifact_id": validation_artifact,
            "sha256": "7" * 64,
        },
    ]
    plan = AdmissionPlan(
        operation="tiny_train",
        reservation=_reservation(checkpoints=1, models=1),
        resolved={
            "request": request,
            "dataset_manifest_sha256": "8" * 64,
            "dataset_bindings": bindings,
            "tokenizer": {
                "tokenizer_id": request["tokenizer_id"],
                "tokenizer_type": "byte",
                "vocab_size": 257,
                "tokenizer_sha256": tokenizer.fingerprint(),
                "artifact_id": tokenizer_artifact,
            },
            "parameter_count": 10,
            "requested_final_step": 1,
            "_inputs": [],
        },
        run_id=run_id,
        model_id=model_id,
        tokenizer_id=None,
        checkpoint_ids=(checkpoint_id,),
    )
    _context(database, training, job_id=job_id, plan=plan)
    values = _checkpoint_payload(checkpoint_id, dataset_id)
    directory = registry.create_staging_dir(job_id) / "checkpoint-0"
    directory.mkdir()
    for name, raw in values.items():
        (directory / name).write_bytes(raw)
    proposal = {
        "type": "checkpoint_ready",
        "staging_name": directory.name,
        "manifest_sha256": hashlib.sha256(values["manifest.json"]).hexdigest(),
        "files": [
            {
                "name": name,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for name, raw in values.items()
        ],
    }

    prepared = store.prepare_checkpoint(job_id, proposal)
    (directory / "model.safetensors").write_bytes(b"mutated after prepare")
    with pytest.raises(RuntimeError, match="forced rollback"):
        with database.transaction() as connection:
            rolled_back = store.commit_checkpoint_in(connection, prepared)
            assert rolled_back.checkpoint_id == checkpoint_id
            raise RuntimeError("forced rollback")
    assert all(staged.path.exists() for _name, staged in prepared.files)
    with database.transaction() as connection:
        committed = store.commit_checkpoint_in(connection, prepared)
    store.complete_checkpoint(prepared)
    assert all(not staged.path.exists() for _name, staged in prepared.files)

    assert committed.checkpoint_id == checkpoint_id
    assert committed.step == 0
    assert committed.run_id == run_id
    assert committed.ack["artifact_ids"].keys() == values.keys()
    _set_job_projection(
        database,
        job_id,
        step=committed.step,
        requested_final_step=1,
        checkpoint_id=committed.checkpoint_id,
    )
    model_artifact_id = committed.ack["artifact_ids"]["model.safetensors"]
    assert registry.read_artifact_bytes(model_artifact_id) == values["model.safetensors"]
    with database.read() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM checkpoint_files WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()[0] == 7
        assert connection.execute(
            "SELECT COUNT(*) FROM models WHERE model_id = ?", (model_id,)
        ).fetchone()[0] == 0

    terminal = store.prepare_terminal(
        job_id,
        state="failed",
        reason_code="NONFINITE_TRAINING_VALUE",
        result=None,
    )
    with database.transaction() as connection:
        finalized = store.finalize_terminal_in(connection, terminal)
    store.complete_terminal(terminal)
    assert finalized.result is None
    assert finalized.run_id == run_id
    assert finalized.model_id == model_id
    assert len(finalized.artifact_ids) == 1
    with database.read() as connection:
        record = json.loads(
            connection.execute(
                "SELECT record_json FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()[0]
        )
        run_result_id = connection.execute(
            "SELECT artifact_id FROM run_artifacts WHERE run_id = ? AND role = 'run_result'",
            (run_id,),
        ).fetchone()[0]
    assert record["checkpoint_id"] == checkpoint_id
    assert record["checkpoint_sha256"] == proposal["manifest_sha256"]
    assert finalized.artifact_ids == (run_result_id,)
    projection = json.loads(registry.read_artifact_bytes(run_result_id))
    assert projection == {
        "run_id": run_id,
        "status": "failed",
        "requested_final_step": 1,
        "last_observed_step": 0,
        "last_durable_checkpoint_step": 0,
        "error": "NONFINITE_TRAINING_VALUE",
        "finished_at": terminal.finished_at,
    }


def test_failed_terminal_discards_unvalidated_pending_output(tmp_path: Path) -> None:
    database, registry, training, store = _runtime(tmp_path)
    job_id, run_id, tokenizer_id = (str(uuid.uuid4()) for _ in range(3))
    dataset_id, split_artifact_id = (str(uuid.uuid4()) for _ in range(2))
    plan = AdmissionPlan(
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
                    "artifact_id": split_artifact_id,
                    "sha256": "5" * 64,
                }
            ],
            "_inputs": [],
        },
        run_id=run_id,
        model_id=None,
        tokenizer_id=tokenizer_id,
        checkpoint_ids=(),
    )
    _context(database, training, job_id=job_id, plan=plan)
    payload = serialize_tokenizer(Tokenizer())
    path = registry.create_staging_dir(job_id) / "tokenizer.json"
    path.write_bytes(payload)
    prepared = store.prepare_artifact(
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
        store.commit_artifact_in(connection, prepared)
    staged_path = prepared.staged.path
    terminal = store.prepare_terminal(
        job_id, state="failed", reason_code="WORKER_PROTOCOL_ERROR"
    )
    assert terminal.artifacts == ()
    assert terminal.discarded_artifacts == (prepared,)
    with database.transaction() as connection:
        finalized = store.finalize_terminal_in(connection, terminal)
    store.complete_terminal(terminal)
    assert not staged_path.exists()
    assert len(finalized.artifact_ids) == 1
    with database.read() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE artifact_id = ?",
            (prepared.artifact_id,),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM tokenizers WHERE tokenizer_id = ?",
            (tokenizer_id,),
        ).fetchone()[0] == 0
        roles = connection.execute(
            "SELECT role FROM run_artifacts WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        assert [row[0] for row in roles] == ["run_result"]


def test_missing_operation_context_is_a_terminal_noop(tmp_path: Path) -> None:
    database, _registry, _training, store = _runtime(tmp_path)
    result = {"operation": "future_operation", "value": 1}
    prepared = store.prepare_terminal(
        str(uuid.uuid4()), state="completed", reason_code="completed", result=result
    )
    with database.transaction() as connection:
        committed = store.finalize_terminal_in(connection, prepared)
    store.complete_terminal(prepared)
    assert committed.result == result
    assert committed.run_id is None
    assert committed.model_id is None
    assert committed.artifact_ids == ()
